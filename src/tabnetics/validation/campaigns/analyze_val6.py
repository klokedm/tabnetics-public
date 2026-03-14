#!/usr/bin/env python3
"""Val-6 paired analysis with gate evaluation (G1..G7).

This script evaluates the two-profile Val-6 campaign:
  baseline  – production defaults (MNPO class-pareto-extended, PLS-DA)
  full      – all implemented features (PLS-DA, WSNR, adaptive sizing,
              all oracles, e-value screening, etc.)
and emits JSON + Markdown reports suitable for promotion decisions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, wilcoxon
except Exception:  # pragma: no cover - optional fallback
    pearsonr = None
    wilcoxon = None

from tabnetics.datasets.benchmark_catalog import DATASET_SETS
from tabnetics.benchmarks.gaming_detectors import run_gaming_detectors
from tabnetics.datasets.registry import DATASET_REGISTRY


PAIR_SPECS: List[Tuple[str, str, str]] = [
    ("full", "baseline", "full_vs_baseline"),
]
TIER_ORDER = ["easy", "medium", "hard", "very_hard"]
BLOCKING_GATES = ("G1", "G2", "G3", "G4", "G5", "G6")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    return value


def _status_job_id_from_marker(marker_name: str) -> Optional[str]:
    # val6__baseline__ds02.DONE.ok -> val6/baseline/ds02
    if not marker_name.endswith(".DONE.ok"):
        return None
    core = marker_name[: -len(".DONE.ok")]
    parts = core.split("__")
    if len(parts) != 3:
        return None
    return "/".join(parts)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required aggregate file: {path}")
    return pd.read_csv(path)


def _load_summary(path: Path) -> pd.DataFrame:
    df = _load_csv(path)
    df = df[(df["protocol"] == "holdout") & (df["config"] == "baseline")].copy()
    df["profile_id"] = df["job_id"].astype(str).str.split("/").str[1]
    return df


def _load_runs(path: Path) -> pd.DataFrame:
    df = _load_csv(path)
    df = df[(df["protocol"] == "holdout") & (df["config"] == "baseline")].copy()
    df["profile_id"] = df["job_id"].astype(str).str.split("/").str[1]
    return df


def _class_count_for_dataset(dataset_id: str) -> Optional[int]:
    seen: set[str] = set()
    current = str(dataset_id)
    while current and current not in seen:
        seen.add(current)
        spec = DATASET_REGISTRY.get(current)
        if spec is None:
            return None
        params = spec.params if isinstance(spec.params, dict) else {}
        synthetic = params.get("synthetic_profile")
        if isinstance(synthetic, dict):
            n_classes = synthetic.get("n_classes")
            if n_classes is not None:
                try:
                    return int(n_classes)
                except Exception:
                    pass
        raw = params.get("n_classes")
        if raw is not None:
            try:
                return int(raw)
            except Exception:
                pass
        base = params.get("base_dataset")
        if not base:
            break
        current = str(base)
    return None


def _wilcoxon_p(vals: Iterable[float], *, alternative: str = "greater") -> Optional[float]:
    arr = np.asarray(list(vals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 6:
        return None
    if np.allclose(arr, 0.0, atol=1e-12):
        return 1.0
    if wilcoxon is None:
        return None
    try:
        res = wilcoxon(arr, alternative=alternative, zero_method="wilcox")
    except Exception:
        return None
    p = getattr(res, "pvalue", None)
    if p is None:
        return None
    return float(p)


def _tier_breakdown(dataset_delta: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for tier in TIER_ORDER:
        sub = dataset_delta[dataset_delta["tier"] == tier]
        if sub.empty:
            continue
        vals = sub["delta_balanced_accuracy"].astype(float)
        out[tier] = {
            "n": int(sub.shape[0]),
            "mean_delta": float(vals.mean()),
            "worst_delta": float(vals.min()),
            "best_delta": float(vals.max()),
        }
    return out


def _dataset_level_pair_stats(
    *,
    summary_df: pd.DataFrame,
    cand_profile: str,
    base_profile: str,
) -> pd.DataFrame:
    cand = summary_df[summary_df["profile_id"] == cand_profile][
        [
            "dataset_id",
            "tier",
            "balanced_accuracy_mean",
            "fs_time_sec_mean",
            "dist_time_sec_mean",
            "transform_time_sec_mean",
        ]
    ].copy()
    cand = cand.rename(
        columns={
            "tier": "tier_cand",
            "balanced_accuracy_mean": "ba_cand",
            "fs_time_sec_mean": "fs_time_cand",
            "dist_time_sec_mean": "dist_time_cand",
            "transform_time_sec_mean": "transform_time_cand",
        }
    )
    base = summary_df[summary_df["profile_id"] == base_profile][
        [
            "dataset_id",
            "tier",
            "balanced_accuracy_mean",
            "fs_time_sec_mean",
            "dist_time_sec_mean",
            "transform_time_sec_mean",
        ]
    ].copy()
    base = base.rename(
        columns={
            "tier": "tier_base",
            "balanced_accuracy_mean": "ba_base",
            "fs_time_sec_mean": "fs_time_base",
            "dist_time_sec_mean": "dist_time_base",
            "transform_time_sec_mean": "transform_time_base",
        }
    )
    merged = cand.merge(base, on="dataset_id", how="inner")
    if merged.empty:
        return merged
    merged["tier"] = merged["tier_cand"].fillna(merged["tier_base"]).astype(str)
    merged["delta_balanced_accuracy"] = merged["ba_cand"].astype(float) - merged["ba_base"].astype(float)
    merged["runtime_cand"] = (
        merged["fs_time_cand"].astype(float)
        + merged["dist_time_cand"].astype(float)
        + merged["transform_time_cand"].astype(float)
    )
    merged["runtime_base"] = (
        merged["fs_time_base"].astype(float)
        + merged["dist_time_base"].astype(float)
        + merged["transform_time_base"].astype(float)
    )
    return merged


def _seed_level_pair_stats(
    *,
    runs_df: pd.DataFrame,
    cand_profile: str,
    base_profile: str,
) -> pd.DataFrame:
    cand = runs_df[runs_df["profile_id"] == cand_profile][["dataset_id", "tier", "seed", "balanced_accuracy"]].copy()
    cand = cand.rename(columns={"tier": "tier_cand", "balanced_accuracy": "ba_cand"})
    base = runs_df[runs_df["profile_id"] == base_profile][["dataset_id", "tier", "seed", "balanced_accuracy"]].copy()
    base = base.rename(columns={"tier": "tier_base", "balanced_accuracy": "ba_base"})
    merged = cand.merge(base, on=["dataset_id", "seed"], how="inner")
    if merged.empty:
        return merged
    merged["tier"] = merged["tier_cand"].fillna(merged["tier_base"]).astype(str)
    merged["delta_balanced_accuracy"] = merged["ba_cand"].astype(float) - merged["ba_base"].astype(float)
    merged["class_count"] = merged["dataset_id"].map(_class_count_for_dataset)
    merged["balanced_accuracy"] = merged["ba_cand"].astype(float)
    return merged


def _profile_dataset_map(plan_jobs: Sequence[Mapping[str, Any]]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for job in plan_jobs:
        job_id = str(job.get("job_id", ""))
        parts = job_id.split("/")
        if len(parts) != 3:
            continue
        profile_id = parts[1]
        datasets = [str(x) for x in (dict(job.get("params") or {}).get("datasets") or [])]
        out.setdefault(profile_id, set()).update(datasets)
    return out


def _coverage_from_status(root_out_dir: Path, plan_jobs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    status_dir = root_out_dir / "_status"
    done_jobs: set[str] = set()
    if status_dir.exists():
        for marker in status_dir.glob("val6__*.DONE.ok"):
            parsed = _status_job_id_from_marker(marker.name)
            if parsed:
                done_jobs.add(parsed)
    planned_ids = [str(j.get("job_id", "")) for j in plan_jobs if str(j.get("job_id", ""))]
    pending = sorted([jid for jid in planned_ids if jid not in done_jobs])
    return {
        "jobs_total": len(planned_ids),
        "jobs_done": len(done_jobs),
        "jobs_pending_count": len(pending),
        "jobs_pending": pending,
        "complete": len(done_jobs) == len(planned_ids),
    }


def _pair_report(
    *,
    summary_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    expected_profile_datasets: Mapping[str, set[str]],
    candidate: str,
    baseline: str,
    reliability_threshold: float = 0.12,
) -> Dict[str, Any]:
    ds_delta = _dataset_level_pair_stats(
        summary_df=summary_df, cand_profile=candidate, base_profile=baseline
    )
    seed_delta = _seed_level_pair_stats(
        runs_df=runs_df, cand_profile=candidate, base_profile=baseline
    )

    expected_overlap = len(
        set(expected_profile_datasets.get(candidate, set()))
        & set(expected_profile_datasets.get(baseline, set()))
    )
    observed_overlap = int(ds_delta.shape[0])

    if ds_delta.empty:
        return {
            "candidate": candidate,
            "baseline": baseline,
            "expected_dataset_overlap": expected_overlap,
            "observed_dataset_overlap": observed_overlap,
            "mean_delta": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "tier_breakdown": {},
            "gates": {},
            "gaming_detectors": {},
        }

    vals = ds_delta["delta_balanced_accuracy"].astype(float)
    mean_delta = float(vals.mean())
    wins = int((vals > 0.001).sum())
    ties = int((vals.abs() <= 0.001).sum())
    losses = int((vals < -0.001).sum())
    tier_breakdown = _tier_breakdown(ds_delta)

    # Gate G1: Wilcoxon and positive mean delta.
    g1_p = _wilcoxon_p(vals.to_numpy(dtype=float), alternative="greater")
    g1_ok = bool((g1_p is not None and g1_p < 0.05) and mean_delta > 0.0)

    # Gate G2: extended-dataset mean delta.
    non_rv = {str(ds) for ds in DATASET_SETS.get("validation_all", []) if not str(ds).startswith("rv_")}
    core = {str(ds) for ds in DATASET_SETS.get("core", [])}
    extended = non_rv - core
    ext_sub = ds_delta[ds_delta["dataset_id"].astype(str).isin(extended)]
    g2_mean = float(ext_sub["delta_balanced_accuracy"].mean()) if not ext_sub.empty else float("nan")
    g2_ok = bool(np.isfinite(g2_mean) and g2_mean >= -0.005)

    # Gate G3: easy-tier mean/worst guard.
    easy_sub = ds_delta[ds_delta["tier"].astype(str) == "easy"]
    easy_mean = float(easy_sub["delta_balanced_accuracy"].mean()) if not easy_sub.empty else float("nan")
    easy_worst = float(easy_sub["delta_balanced_accuracy"].min()) if not easy_sub.empty else float("nan")
    g3_ok = bool(
        np.isfinite(easy_mean)
        and np.isfinite(easy_worst)
        and easy_mean >= -0.005
        and easy_worst >= -0.03
    )

    # Gate G4: worst regression after reliability filter on baseline std.
    base_seed = runs_df[runs_df["profile_id"] == baseline][["dataset_id", "balanced_accuracy"]].copy()
    base_seed["balanced_accuracy"] = pd.to_numeric(base_seed["balanced_accuracy"], errors="coerce")
    base_std = base_seed.groupby("dataset_id")["balanced_accuracy"].std(ddof=0)
    reliable = set(base_std[base_std <= float(reliability_threshold)].index.astype(str))
    rel_sub = ds_delta[ds_delta["dataset_id"].astype(str).isin(reliable)]
    g4_worst = float(rel_sub["delta_balanced_accuracy"].min()) if not rel_sub.empty else float("nan")
    g4_ok = bool(np.isfinite(g4_worst) and g4_worst > -0.05)

    # Gate G5: runtime <= 2x baseline on overlap.
    rt_cand = pd.to_numeric(ds_delta["runtime_cand"], errors="coerce")
    rt_base = pd.to_numeric(ds_delta["runtime_base"], errors="coerce")
    rt_ratio = float(np.nanmean(rt_cand) / max(1e-12, float(np.nanmean(rt_base))))
    g5_ok = bool(np.isfinite(rt_ratio) and rt_ratio <= 2.0)

    # Gate G6: anti-gaming diagnostics.
    gt_rows = seed_delta.copy()
    gt = run_gaming_detectors(gt_rows)
    g6_ok = not bool(gt.get("any_flagged", False))

    # Gate G7 (advisory): detrended Wilcoxon (RTM correction).
    detrended_p: Optional[float] = None
    rtm_rho: Optional[float] = None
    if ds_delta.shape[0] >= 6:
        x = ds_delta["ba_base"].astype(float).to_numpy()
        y = ds_delta["delta_balanced_accuracy"].astype(float).to_numpy()
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        y_hat = A @ coef
        resid = y - y_hat
        detrended_p = _wilcoxon_p(resid, alternative="greater")
        if pearsonr is not None:
            try:
                rho_val, _ = pearsonr(x, y)
                if np.isfinite(rho_val):
                    rtm_rho = float(rho_val)
            except Exception:
                rtm_rho = None
    g7_ok = bool(detrended_p is not None and detrended_p < 0.10)

    gates = {
        "G1": {"pass": g1_ok, "p_wilcoxon": g1_p, "mean_delta": mean_delta},
        "G2": {"pass": g2_ok, "extended_mean_delta": g2_mean, "threshold": -0.005},
        "G3": {"pass": g3_ok, "easy_mean_delta": easy_mean, "easy_worst_delta": easy_worst},
        "G4": {
            "pass": g4_ok,
            "worst_reliable_delta": g4_worst,
            "reliability_max_seed_std": float(reliability_threshold),
        },
        "G5": {"pass": g5_ok, "runtime_ratio_vs_baseline": rt_ratio, "threshold": 2.0},
        "G6": {"pass": g6_ok, "any_flagged": bool(gt.get("any_flagged", False))},
        "G7": {"pass": g7_ok, "detrended_wilcoxon_p": detrended_p, "rtm_rho": rtm_rho, "advisory": True},
    }

    return {
        "candidate": candidate,
        "baseline": baseline,
        "expected_dataset_overlap": expected_overlap,
        "observed_dataset_overlap": observed_overlap,
        "mean_delta": mean_delta,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "tier_breakdown": tier_breakdown,
        "gates": gates,
        "gaming_detectors": gt,
    }


def _recommendation(pair_reports: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    report = dict(pair_reports.get("full_vs_baseline") or {})
    gates = dict(report.get("gates") or {})
    blocking_ok = all(bool(dict(gates.get(g, {})).get("pass", False)) for g in BLOCKING_GATES)

    if not blocking_ok:
        return {
            "promote": False,
            "selected_profile": "baseline",
            "reason": "Full profile did not pass all blocking gates G1..G6.",
        }

    return {
        "promote": True,
        "selected_profile": "full",
        "mean_delta": float(report.get("mean_delta", 0.0) or 0.0),
        "reason": "Full profile passed all blocking gates; promoting.",
    }


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# VAL6 Phase Analysis")
    lines.append("")
    cov = dict(report.get("coverage") or {})
    lines.append(f"- Jobs: {cov.get('jobs_done', 0)}/{cov.get('jobs_total', 0)} complete")
    lines.append(f"- Complete: {bool(cov.get('complete', False))}")
    lines.append("")
    lines.append("## Pair Reports")
    lines.append("")

    pair_reports = dict(report.get("pair_reports") or {})
    for pair_id, row_raw in pair_reports.items():
        row = dict(row_raw or {})
        if not row:
            continue
        lines.append(f"### {pair_id}: `{row.get('candidate')}` vs `{row.get('baseline')}`")
        lines.append("")
        lines.append(
            f"- Overlap: {row.get('observed_dataset_overlap', 0)}/{row.get('expected_dataset_overlap', 0)} datasets"
        )
        lines.append(
            f"- Wins/Ties/Losses: {row.get('wins', 0)}/{row.get('ties', 0)}/{row.get('losses', 0)}"
        )
        lines.append(f"- Mean ΔBA: {row.get('mean_delta')}")
        lines.append("")
        lines.append("| Gate | Pass | Detail |")
        lines.append("|---|---|---|")
        gates = dict(row.get("gates") or {})
        for gate_id in ("G1", "G2", "G3", "G4", "G5", "G6", "G7"):
            g = dict(gates.get(gate_id) or {})
            detail = ", ".join(
                f"{k}={v}" for k, v in g.items() if k not in {"pass", "advisory"}
            )
            lines.append(f"| {gate_id} | {bool(g.get('pass', False))} | {detail} |")
        lines.append("")

    rec = dict(report.get("promotion_recommendation") or {})
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"- Promote: {bool(rec.get('promote', False))}")
    lines.append(f"- Selected profile: `{rec.get('selected_profile')}`")
    lines.append(f"- Reason: {rec.get('reason', '')}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root-out-dir",
        type=Path,
        default=Path("run_artifacts/validation-6/val6_6pods_merged"),
        help="Merged Val-6 root directory containing _aggregate outputs.",
    )
    p.add_argument(
        "--plan",
        type=Path,
        default=Path("pod_validation/plan_6_val6.json"),
        help="Val-6 plan json used for expected-coverage accounting.",
    )
    p.add_argument(
        "--reliability-max-seed-std",
        type=float,
        default=0.12,
        help="Reliability filter threshold for G4.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root_out_dir
    agg = root / "_aggregate"
    summary_csv = agg / "benchmark_df_fs_summary__all_jobs.csv"
    runs_csv = agg / "benchmark_df_fs_runs__all_jobs.csv"

    summary_df = _load_summary(summary_csv)
    runs_df = _load_runs(runs_csv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    jobs = list(plan.get("jobs", []))
    profile_datasets = _profile_dataset_map(jobs)
    coverage = _coverage_from_status(root, jobs)

    pair_reports: Dict[str, Any] = {}
    for cand, base, pair_id in PAIR_SPECS:
        pair_reports[pair_id] = _pair_report(
            summary_df=summary_df,
            runs_df=runs_df,
            expected_profile_datasets=profile_datasets,
            candidate=cand,
            baseline=base,
            reliability_threshold=float(args.reliability_max_seed_std),
        )

    report = {
        "coverage": coverage,
        "pair_reports": pair_reports,
        "promotion_recommendation": _recommendation(pair_reports),
    }
    report = _to_builtin(report)

    out_json = agg / "val6_phase_analysis.json"
    out_md = agg / "VAL6_PHASE_ANALYSIS.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(out_md, report)

    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

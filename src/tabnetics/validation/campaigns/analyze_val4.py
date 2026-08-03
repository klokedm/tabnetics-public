#!/usr/bin/env python3
"""Val-4 paired analysis + (optionally provisional) promotion decisions.

This script is designed for Phase-4 tasks T-R-121/T-R-122/T-R-123.
It supports incomplete shard coverage (best-guess mode) and emits
machine-readable + markdown reports for hand-off and post-mortem refresh.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from tabnetics.benchmarks.gaming_detectors import run_gaming_detectors
from tabnetics.datasets.registry import DATASET_REGISTRY
from tabnetics.validation.core.gates import GateConfig
from tabnetics.validation.core.ledger import append_rows, rows_from_val4_report
from tabnetics.validation.core.report import build_side_by_side_gate_report


TIER_ORDER = ["easy", "medium", "hard", "very_hard"]
PAIR_SPECS: List[Tuple[str, str, str]] = [
    ("broad_oracle", "baseline", "B_vs_A"),
    ("new_methods", "broad_oracle", "C_vs_B"),
    ("new_methods", "baseline", "C_vs_A"),
    ("pls_da_guard", "baseline", "D_vs_A"),
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_job_id_from_marker(marker_name: str) -> Optional[str]:
    # val4__baseline__ds02.DONE.ok -> val4/baseline/ds02
    if not marker_name.endswith(".DONE.ok"):
        return None
    core = marker_name[: -len(".DONE.ok")]
    parts = core.split("__")
    if len(parts) != 3:
        return None
    return "/".join(parts)


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
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return value


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


def _build_profile_dataset_map(plan_jobs: Sequence[Mapping[str, Any]]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for job in plan_jobs:
        job_id = str(job.get("job_id", ""))
        parts = job_id.split("/")
        if len(parts) != 3:
            continue
        profile_id = parts[1]
        params = dict(job.get("params") or {})
        datasets = [str(x) for x in (params.get("datasets") or [])]
        out.setdefault(profile_id, set()).update(datasets)
    return out


def _load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["protocol"] == "holdout") & (df["config"] == "baseline")].copy()
    df["profile_id"] = df["job_id"].astype(str).str.split("/").str[1]
    return df


def _load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["protocol"] == "holdout") & (df["config"] == "baseline")].copy()
    df["profile_id"] = df["job_id"].astype(str).str.split("/").str[1]
    return df


def _load_sota(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["profile_id"] = df["job_id"].astype(str).str.split("/").str[1]
    return df


def _dataset_level_pair_stats(
    *,
    summary_df: pd.DataFrame,
    cand_profile: str,
    base_profile: str,
) -> pd.DataFrame:
    cand = summary_df[summary_df["profile_id"] == cand_profile][["dataset_id", "tier", "balanced_accuracy_mean"]].copy()
    cand = cand.rename(columns={"tier": "tier_cand", "balanced_accuracy_mean": "ba_cand"})
    base = summary_df[summary_df["profile_id"] == base_profile][["dataset_id", "tier", "balanced_accuracy_mean"]].copy()
    base = base.rename(columns={"tier": "tier_base", "balanced_accuracy_mean": "ba_base"})
    merged = cand.merge(base, on="dataset_id", how="inner")
    if merged.empty:
        return merged
    merged["tier"] = merged["tier_cand"].fillna(merged["tier_base"]).astype(str)
    merged["delta_balanced_accuracy"] = merged["ba_cand"].astype(float) - merged["ba_base"].astype(float)
    return merged[["dataset_id", "tier", "ba_cand", "ba_base", "delta_balanced_accuracy"]]


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
    return merged[
        [
            "dataset_id",
            "tier",
            "seed",
            "balanced_accuracy",
            "ba_cand",
            "ba_base",
            "delta_balanced_accuracy",
            "class_count",
        ]
    ]


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
            "improved": int((vals > 0.001).sum()),
            "same": int((vals.abs() <= 0.001).sum()),
            "regressed": int((vals < -0.001).sum()),
        }
    return out


def _wilcoxon_p(values: Sequence[float]) -> Optional[float]:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    non_zero = [value for value in vals if abs(value) > 1e-12]
    if not non_zero:
        return 1.0 if vals else None
    try:
        from scipy import stats

        return float(stats.wilcoxon(non_zero, alternative="two-sided").pvalue)
    except Exception:
        return None


def _scorecard_by_profile(sota_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for profile_id, grp in sota_df.groupby("profile_id"):
        status_counts = grp["holdout_status"].value_counts(dropna=False).to_dict()
        n = int(grp.shape[0])
        above = int(status_counts.get("above", 0))
        within = int(status_counts.get("within", 0))
        below = int(status_counts.get("below", 0))
        out[str(profile_id)] = {
            "n": n,
            "above": above,
            "within": within,
            "below": below,
            "pct_above_or_within": float((above + within) / n) if n else None,
        }
    return out


def _coverage_from_status(
    *,
    root_out_dir: Path,
    plan_jobs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    status_dir = root_out_dir / "_status"
    done_jobs: set[str] = set()
    if status_dir.exists():
        for marker in status_dir.glob("val4__*.DONE.ok"):
            parsed = _status_job_id_from_marker(marker.name)
            if parsed:
                done_jobs.add(parsed)
    planned_ids = [str(j.get("job_id", "")) for j in plan_jobs if str(j.get("job_id", ""))]
    pending = sorted([jid for jid in planned_ids if jid not in done_jobs])
    return {
        "jobs_total": len(planned_ids),
        "jobs_done": len(done_jobs),
        "jobs_pending": pending,
        "jobs_pending_count": len(pending),
        "complete": len(done_jobs) == len(planned_ids),
    }


def _pair_report(
    *,
    summary_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    expected_profile_datasets: Mapping[str, set[str]],
    cand_profile: str,
    base_profile: str,
    reliability_threshold: float = 0.12,
) -> Dict[str, Any]:
    dataset_delta = _dataset_level_pair_stats(summary_df=summary_df, cand_profile=cand_profile, base_profile=base_profile)
    seed_delta = _seed_level_pair_stats(runs_df=runs_df, cand_profile=cand_profile, base_profile=base_profile)

    expected_overlap = len(set(expected_profile_datasets.get(cand_profile, set())) & set(expected_profile_datasets.get(base_profile, set())))
    observed_overlap = int(dataset_delta.shape[0])
    complete = bool(expected_overlap > 0 and observed_overlap >= expected_overlap)

    if seed_delta.empty:
        return {
            "candidate": cand_profile,
            "baseline": base_profile,
            "expected_dataset_overlap": expected_overlap,
            "observed_dataset_overlap": observed_overlap,
            "complete_dataset_overlap": complete,
            "observed_seed_rows": 0,
            "mean_delta": None,
            "wilcoxon_p": None,
            "hard_mean_delta": None,
            "tier_breakdown": {},
            "gate": {},
            "gaming_detectors": {},
            "top_positive_datasets": [],
            "top_negative_datasets": [],
        }

    deltas_by_dataset = (
        seed_delta.groupby("dataset_id")["delta_balanced_accuracy"]
        .apply(lambda s: [float(x) for x in s.tolist()])
        .to_dict()
    )
    tier_by_dataset = (
        seed_delta.drop_duplicates("dataset_id")
        .set_index("dataset_id")["tier"]
        .astype(str)
        .to_dict()
    )

    # Compute baseline per-dataset seed std for reliability filter.
    base_runs = runs_df[runs_df["profile_id"] == base_profile]
    baseline_seed_std: Dict[str, float] = {}
    for ds_id, grp in base_runs.groupby("dataset_id"):
        ba_vals = grp["balanced_accuracy"].astype(float)
        if len(ba_vals) > 1:
            baseline_seed_std[str(ds_id)] = float(ba_vals.std(ddof=1))

    # Create configs for reliability-filtered and bootstrap gates (opt-in).
    reliable_cfg: Optional[GateConfig] = None
    bootstrap_cfg: Optional[GateConfig] = None
    if reliability_threshold > 0:
        reliable_cfg = GateConfig(
            mode="strict",
            reliability_max_seed_std=reliability_threshold,
        )
        bootstrap_cfg = GateConfig(
            mode="bootstrap",
            reliability_max_seed_std=reliability_threshold,
            n_bootstrap=2000,
            n_permutations=2000,
            random_seed=42,
        )

    gate = build_side_by_side_gate_report(
        deltas_by_dataset=deltas_by_dataset,
        tier_by_dataset=tier_by_dataset,
        reliable_config=reliable_cfg,
        bootstrap_config=bootstrap_cfg,
        baseline_seed_std_by_dataset=baseline_seed_std or None,
    )
    gaming = run_gaming_detectors(
        seed_delta,
        delta_col="delta_balanced_accuracy",
        metric_col="balanced_accuracy",
        dataset_col="dataset_id",
        seed_col="seed",
        tier_col="tier",
        class_count_col="class_count",
    )

    dataset_delta = dataset_delta.sort_values("delta_balanced_accuracy", ascending=False)
    tier_stats = _tier_breakdown(dataset_delta)

    hard_mean_delta = None
    if "hard" in tier_stats:
        hard_mean_delta = tier_stats["hard"]["mean_delta"]

    return {
        "candidate": cand_profile,
        "baseline": base_profile,
        "expected_dataset_overlap": expected_overlap,
        "observed_dataset_overlap": observed_overlap,
        "complete_dataset_overlap": complete,
        "observed_seed_rows": int(seed_delta.shape[0]),
        "mean_delta": float(dataset_delta["delta_balanced_accuracy"].mean()) if not dataset_delta.empty else None,
        "wilcoxon_p": _wilcoxon_p(dataset_delta["delta_balanced_accuracy"].astype(float).tolist())
        if not dataset_delta.empty
        else None,
        "hard_mean_delta": hard_mean_delta,
        "tier_breakdown": tier_stats,
        "gate": gate,
        "gaming_detectors": gaming,
        "top_positive_datasets": [
            {"dataset_id": str(r.dataset_id), "tier": str(r.tier), "delta": float(r.delta_balanced_accuracy)}
            for r in dataset_delta.head(5).itertuples(index=False)
        ],
        "top_negative_datasets": [
            {"dataset_id": str(r.dataset_id), "tier": str(r.tier), "delta": float(r.delta_balanced_accuracy)}
            for r in dataset_delta.tail(5)
            .sort_values(by="delta_balanced_accuracy", key=lambda s: s.abs(), ascending=False)
            .itertuples(index=False)
        ],
    }


def _choose_promotion(
    *,
    pair_reports: Mapping[str, Mapping[str, Any]],
    allow_partial: bool,
    promotion_gate: str = "strict",
) -> Dict[str, Any]:
    broad = dict(pair_reports.get("B_vs_A") or {})
    new = dict(pair_reports.get("C_vs_A") or {})
    pls = dict(pair_reports.get("D_vs_A") or {})

    def _gate_pass(rep: Mapping[str, Any], gate_key: str) -> bool:
        try:
            return str(rep["gate"][gate_key]["verdict"]) == "PASS"
        except (KeyError, TypeError):
            return False

    def _complete(rep: Mapping[str, Any]) -> bool:
        return bool(rep.get("complete_dataset_overlap", False))

    broad_pass = _gate_pass(broad, promotion_gate)
    new_pass = _gate_pass(new, promotion_gate)
    pls_pass = _gate_pass(pls, promotion_gate)
    broad_complete = _complete(broad)
    new_complete = _complete(new)
    pls_complete = _complete(pls)

    eligible_broad = broad_pass and (allow_partial or broad_complete)
    eligible_new = new_pass and (allow_partial or new_complete)
    eligible_pls = pls_pass and (allow_partial or pls_complete)

    # --- FS profile selection (B vs C vs baseline) ---
    chosen = "baseline"
    decision_reason = f"No candidate passed {promotion_gate} gate."
    confidence = "high" if (broad_complete and new_complete) else "provisional"

    if eligible_broad and eligible_new:
        # Prefer C (new_methods) over B (broad_oracle) when both pass:
        # C includes all of B's methods plus extra eval models and copula
        # derand×5, giving broader coverage at acceptable compute cost.
        chosen = "new_methods"
        decision_reason = f"Both passed {promotion_gate}; new_methods preferred (broader method pool)."
    elif eligible_broad:
        chosen = "broad_oracle"
        decision_reason = f"Only broad_oracle passed {promotion_gate} gate."
    elif eligible_new:
        chosen = "new_methods"
        decision_reason = f"Only new_methods passed {promotion_gate} gate."

    # --- PLS-DA overlay (independent, additive) ---
    pls_recommendation = "disabled"
    pls_reason = f"PLS-DA did not pass {promotion_gate} gate."
    if eligible_pls:
        pls_recommendation = "enabled"
        pls_reason = f"PLS-DA passed {promotion_gate} gate; enable for datasets with n_classes >= 5."

    if not (broad_complete and new_complete):
        confidence = "provisional"

    notes: List[str] = []
    if not broad_complete:
        notes.append("B_vs_A incomplete coverage; decision may change after shard completion.")
    if not new_complete:
        notes.append("C_vs_A incomplete coverage; decision may change after shard completion.")
    if not pls_complete:
        notes.append("D_vs_A incomplete; PLS-DA recommendation is provisional.")

    return {
        "allow_partial_decisions": bool(allow_partial),
        "confidence": confidence,
        "recommended_profile": chosen,
        "decision_reason": decision_reason,
        "pls_da_recommendation": pls_recommendation,
        "pls_da_reason": pls_reason,
        "promotion_gate": promotion_gate,
        "broad_oracle_pass": broad_pass,
        "new_methods_pass": new_pass,
        "pls_da_pass": pls_pass,
        "coverage_complete": {
            "B_vs_A": broad_complete,
            "C_vs_A": new_complete,
            "D_vs_A": pls_complete,
        },
        "notes": notes,
        "rerun_trigger": "Recompute after shard-3 completion and refresh decisions if strict verdicts change.",
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    status = report.get("run_status", {})
    decision = report.get("promotion_decision", {})
    lines.append("# Val-4 Phase Analysis")
    lines.append("")
    lines.append("## Run Status")
    lines.append("")
    lines.append(f"- Jobs done: {status.get('jobs_done')}/{status.get('jobs_total')}")
    lines.append(f"- Jobs pending: {status.get('jobs_pending_count')}")
    lines.append(f"- Complete: {status.get('complete')}")
    lines.append("")
    lines.append("## Promotion Recommendation")
    lines.append("")
    lines.append(f"- Confidence: **{decision.get('confidence')}**")
    lines.append(f"- Promotion gate: **{decision.get('promotion_gate', 'strict')}**")
    lines.append(f"- Recommended FS profile: **{decision.get('recommended_profile')}**")
    lines.append(f"- Reason: {decision.get('decision_reason')}")
    lines.append(f"- PLS-DA overlay: **{decision.get('pls_da_recommendation', 'n/a')}**")
    lines.append(f"- PLS-DA reason: {decision.get('pls_da_reason', 'n/a')}")
    for note in decision.get("notes", []):
        lines.append(f"- Note: {note}")
    lines.append("")
    lines.append("## Pairwise Gate Summary")
    lines.append("")
    lines.append("| Pair | Overlap | Mean ΔBA | Strict | Reliable | Bootstrap | Quantile | Any GT Flag |")
    lines.append("|------|--------:|---------:|--------|----------|-----------|----------|-------------|")
    pair_reports = report.get("pair_reports", {})
    for label in ["B_vs_A", "C_vs_B", "C_vs_A", "D_vs_A"]:
        pr = pair_reports.get(label) or {}
        overlap = f"{pr.get('observed_dataset_overlap', 0)}/{pr.get('expected_dataset_overlap', 0)}"
        mean_delta = pr.get("mean_delta")
        mean_delta_str = "n/a" if mean_delta is None else f"{float(mean_delta):+.4f}"
        strict = (((pr.get("gate") or {}).get("strict") or {}).get("verdict")) or "n/a"
        reliable = (((pr.get("gate") or {}).get("reliable") or {}).get("verdict")) or "n/a"
        bootstrap = (((pr.get("gate") or {}).get("bootstrap") or {}).get("verdict")) or "n/a"
        quant = (((pr.get("gate") or {}).get("quantile") or {}).get("verdict")) or "n/a"
        gt_flag = (((pr.get("gaming_detectors") or {}).get("any_flagged")))
        gt_flag_str = "yes" if gt_flag else "no"
        lines.append(f"| {label} | {overlap} | {mean_delta_str} | {strict} | {reliable} | {bootstrap} | {quant} | {gt_flag_str} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def analyze_val4(
    *,
    root_out_dir: Path,
    plan_path: Path,
    allow_partial_decisions: bool,
    reliability_threshold: float = 0.12,
    promotion_gate: str = "strict",
) -> Dict[str, Any]:
    aggregate_dir = root_out_dir / "_aggregate"
    summary_csv = aggregate_dir / "benchmark_df_fs_summary__all_jobs.csv"
    sota_csv = aggregate_dir / "benchmark_df_fs_sota_comparison__all_jobs.csv"
    runs_csv = aggregate_dir / "benchmark_df_fs_runs__all_jobs.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing aggregate summary CSV: {summary_csv}")
    if not sota_csv.exists():
        raise FileNotFoundError(f"Missing aggregate SOTA CSV: {sota_csv}")
    if not runs_csv.exists():
        raise FileNotFoundError(
            f"Missing aggregate runs CSV: {runs_csv} (rerun pod_validation/aggregate_results.py)"
        )

    plan = _read_json(plan_path)
    plan_jobs = list(plan.get("jobs") or [])
    profile_datasets = _build_profile_dataset_map(plan_jobs)
    run_status = _coverage_from_status(root_out_dir=root_out_dir, plan_jobs=plan_jobs)

    summary_df = _load_summary(summary_csv)
    runs_df = _load_runs(runs_csv)
    sota_df = _load_sota(sota_csv)

    pair_reports: Dict[str, Any] = {}
    for cand, base, label in PAIR_SPECS:
        pair_reports[label] = _pair_report(
            summary_df=summary_df,
            runs_df=runs_df,
            expected_profile_datasets=profile_datasets,
            cand_profile=cand,
            base_profile=base,
            reliability_threshold=reliability_threshold,
        )

    scorecard = _scorecard_by_profile(sota_df)
    promotion = _choose_promotion(
        pair_reports=pair_reports,
        allow_partial=allow_partial_decisions,
        promotion_gate=promotion_gate,
    )

    return {
        "plan_path": str(plan_path),
        "root_out_dir": str(root_out_dir),
        "run_status": run_status,
        "profiles_expected_datasets": {k: sorted(v) for k, v in profile_datasets.items()},
        "scorecard": scorecard,
        "pair_reports": pair_reports,
        "promotion_decision": promotion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Val-4 paired comparisons and promotion recommendations.")
    parser.add_argument(
        "--root-out-dir",
        type=str,
        default="run_artifacts/validation-4/val4_6pods_merged",
        help="Val-4 merged artifact root (contains _status, _aggregate, val4/...).",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="pod_validation/plan_6.json",
        help="Val-4 plan JSON path.",
    )
    parser.add_argument(
        "--allow-partial-decisions",
        action="store_true",
        help="Allow provisional promotion recommendations when shard coverage is incomplete.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional JSON output path (default: <root>/_aggregate/val4_phase_analysis.json).",
    )
    parser.add_argument(
        "--md-out",
        type=str,
        default="",
        help="Optional Markdown output path (default: <root>/_aggregate/VAL4_PHASE_ANALYSIS.md).",
    )
    parser.add_argument(
        "--ledger-out",
        type=str,
        default="",
        help="Optional promotion-ledger JSONL path (default: <root>/_aggregate/promotion_attempts.jsonl).",
    )
    parser.add_argument(
        "--ledger-evidence-link",
        action="append",
        default=[],
        help="Issue comment or artifact URL to attach to each appended ledger row; may be repeated.",
    )
    parser.add_argument(
        "--reliability-threshold",
        type=float,
        default=0.12,
        help="Baseline seed std threshold for reliability filter (0 = disabled). Default: 0.12.",
    )
    parser.add_argument(
        "--promotion-gate",
        type=str,
        default="strict",
        choices=["strict", "reliable", "bootstrap"],
        help="Which gate to use for promotion decisions (default: strict).",
    )
    args = parser.parse_args()

    root_out_dir = Path(args.root_out_dir).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing plan file: {plan_path}")

    report = analyze_val4(
        root_out_dir=root_out_dir,
        plan_path=plan_path,
        allow_partial_decisions=bool(args.allow_partial_decisions),
        reliability_threshold=float(args.reliability_threshold),
        promotion_gate=str(args.promotion_gate),
    )
    report = _to_builtin(report)

    json_out = (
        Path(args.json_out).expanduser().resolve()
        if args.json_out
        else (root_out_dir / "_aggregate" / "val4_phase_analysis.json")
    )
    md_out = (
        Path(args.md_out).expanduser().resolve()
        if args.md_out
        else (root_out_dir / "_aggregate" / "VAL4_PHASE_ANALYSIS.md")
    )
    ledger_out = (
        Path(args.ledger_out).expanduser().resolve()
        if args.ledger_out
        else (root_out_dir / "_aggregate" / "promotion_attempts.jsonl")
    )
    ledger_rows = rows_from_val4_report(
        report,
        campaign="val4",
        evidence_links=list(args.ledger_evidence_link or []),
        source_artifacts={"json_report": json_out, "markdown_report": md_out, "plan": plan_path},
    )
    report["promotion_ledger"] = {
        "path": str(ledger_out),
        "row_count": int(len(ledger_rows)),
        "schema_version": "tabnetics_promotion_ledger_v1",
        "history_authority": "GitHub issues/comments remain canonical; ledger rows are queryable evidence.",
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    ledger_out.parent.mkdir(parents=True, exist_ok=True)

    json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_out.write_text(_render_markdown(report), encoding="utf-8")
    ledger_count = append_rows(ledger_out, ledger_rows)

    print(f"Wrote JSON: {json_out}")
    print(f"Wrote Markdown: {md_out}")
    print(f"Appended ledger rows: {ledger_count} -> {ledger_out}")
    print(f"Recommendation: {report['promotion_decision']['recommended_profile']} ({report['promotion_decision']['confidence']})")


if __name__ == "__main__":
    main()

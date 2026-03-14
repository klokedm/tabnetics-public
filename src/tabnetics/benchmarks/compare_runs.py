#!/usr/bin/env python3
"""
Compare two benchmark runs produced by `python -m tabnetics.benchmarks.cli`.

This is a lightweight helper for the project's implement→validate loop:
- Join two `df_fs_summary.csv` files (baseline vs candidate) on `dataset_id`.
- Emit per-dataset deltas and a tier-level gate summary for promotion decisions.

Example:
  python experiments/compare_df_fs_runs.py \
    --baseline-run run_artifacts/validation/a6_full3_baseline/20260209_204846_df_fs_sota_benchmark \
    --candidate-run run_artifacts/validation/a6_full3_fastpath/20260209_210000_df_fs_sota_benchmark \
    --out-csv run_artifacts/validation/a6_full3_fastpath_vs_baseline_deltas.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


def _import_benchmark_datasets():
    try:
        from tabnetics.benchmarks.runner import BENCHMARK_DATASETS
    except ImportError:
        # Allow running from inside `experiments/`.
        from run_df_fs_sota_benchmark import BENCHMARK_DATASETS  # type: ignore

    return BENCHMARK_DATASETS


def _resolve_summary_csv(p: Path) -> Path:
    if p.is_dir():
        return p / "df_fs_summary.csv"
    return p


def _load_summary(
    run_or_csv: Path,
    *,
    config: str = "baseline",
    protocol: str = "holdout",
) -> pd.DataFrame:
    csv_path = _resolve_summary_csv(run_or_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"dataset_id", "config", "protocol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    subset = df[(df["protocol"] == protocol) & (df["config"] == config)].copy()
    if subset.empty:
        available = sorted({str(x) for x in df["config"].unique()})
        raise ValueError(
            f"No rows for protocol={protocol!r} config={config!r} in {csv_path}. "
            f"Available configs: {available}"
        )
    return subset


def _add_dataset_metadata(df: pd.DataFrame) -> pd.DataFrame:
    benchmark = _import_benchmark_datasets()
    tiers = []
    names = []
    for ds_id in df["dataset_id"].astype(str).tolist():
        spec = benchmark.get(ds_id)
        tiers.append(getattr(spec, "tier", "unknown") if spec is not None else "unknown")
        names.append(getattr(spec, "display_name", ds_id) if spec is not None else ds_id)
    out = df.copy()
    out.insert(1, "tier", tiers)
    out.insert(2, "dataset_name", names)
    return out


def _metric_columns_to_compare(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Sequence[str]:
    # Canonical columns from `df_fs_summary.csv` that are useful for gating and attribution.
    preferred = [
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "hybrid_score_mean",
        "accuracy_mean",
        "selected_features_mean",
        "fs_time_sec_mean",
        "dist_time_sec_mean",
        "transform_time_sec_mean",
        "n_dist_features_transformed_mean",
        "n_dist_rejected_mean",
        "n_dist_skipped_unreliable_mean",
        "n_dist_skipped_block_cv_mean",
        "n_low_gof_downweighted_mean",
        "cdf_block_gating_time_sec_mean",
    ]
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    both = [c for c in preferred if c in cols_a and c in cols_b]
    # Add any additional *_mean columns present in both, for completeness.
    extra = sorted([c for c in cols_a.intersection(cols_b) if c.endswith("_mean") and c not in both])
    return [*both, *extra]


@dataclass(frozen=True)
class GateResult:
    ok: bool
    message: str


def _evaluate_promotion_gates(
    deltas: pd.DataFrame,
    *,
    easy_cap_ba: float = 0.02,
    medium_cap_ba: float = 0.05,
    hard_min_mean_ba: float = 0.0,
) -> GateResult:
    if "tier" not in deltas.columns or "delta_balanced_accuracy" not in deltas.columns:
        return GateResult(ok=False, message="Missing required columns for gate evaluation.")

    # Caps interpreted as: "no dataset in tier regresses more than cap".
    easy_violations = deltas[(deltas["tier"] == "easy") & (deltas["delta_balanced_accuracy"] < -easy_cap_ba)]
    medium_violations = deltas[(deltas["tier"] == "medium") & (deltas["delta_balanced_accuracy"] < -medium_cap_ba)]

    hard = deltas[deltas["tier"] == "hard"]
    hard_mean = float(hard["delta_balanced_accuracy"].mean()) if not hard.empty else 0.0

    problems = []
    if not easy_violations.empty:
        problems.append(f"easy-tier cap violated on {len(easy_violations)} dataset(s)")
    if not medium_violations.empty:
        problems.append(f"medium-tier cap violated on {len(medium_violations)} dataset(s)")
    if hard_mean < hard_min_mean_ba:
        problems.append(f"hard-tier mean delta BA {hard_mean:+.4f} < {hard_min_mean_ba:+.4f}")

    if problems:
        return GateResult(ok=False, message="; ".join(problems))
    return GateResult(ok=True, message="promotion gates satisfied")


def compare_runs(
    *,
    baseline_run_or_csv: Path,
    candidate_run_or_csv: Path,
    baseline_config: str = "baseline",
    candidate_config: str = "baseline",
    protocol: str = "holdout",
    easy_cap_ba: float = 0.02,
    medium_cap_ba: float = 0.05,
    hard_min_mean_ba: float = 0.0,
) -> pd.DataFrame:
    base = _load_summary(baseline_run_or_csv, config=baseline_config, protocol=protocol)
    cand = _load_summary(candidate_run_or_csv, config=candidate_config, protocol=protocol)

    base = _add_dataset_metadata(base)
    cand = _add_dataset_metadata(cand)

    metrics = _metric_columns_to_compare(base, cand)
    keep_cols = ["dataset_id", "tier", "dataset_name", *metrics]
    base_small = base[keep_cols].rename(columns={c: f"baseline_{c}" for c in metrics})
    cand_small = cand[keep_cols].rename(columns={c: f"candidate_{c}" for c in metrics})

    merged = base_small.merge(cand_small, on=["dataset_id", "tier", "dataset_name"], how="inner")
    if merged.empty:
        raise ValueError("No datasets in common between the two runs (after filtering by config/protocol).")

    # Standardize the common deltas first.
    if "balanced_accuracy_mean" in metrics:
        merged["delta_balanced_accuracy"] = merged["candidate_balanced_accuracy_mean"] - merged["baseline_balanced_accuracy_mean"]
    if "macro_f1_mean" in metrics:
        merged["delta_macro_f1"] = merged["candidate_macro_f1_mean"] - merged["baseline_macro_f1_mean"]
    if "hybrid_score_mean" in metrics:
        merged["delta_hybrid"] = merged["candidate_hybrid_score_mean"] - merged["baseline_hybrid_score_mean"]

    # Additional deltas for attribution.
    for c in metrics:
        b = f"baseline_{c}"
        k = f"candidate_{c}"
        d = f"delta_{c}"
        if b in merged.columns and k in merged.columns:
            merged[d] = merged[k] - merged[b]

    # Gate summary (returned as attributes on the DataFrame for the CLI; useful in notebooks too).
    gate = _evaluate_promotion_gates(
        merged,
        easy_cap_ba=easy_cap_ba,
        medium_cap_ba=medium_cap_ba,
        hard_min_mean_ba=hard_min_mean_ba,
    )
    merged.attrs["gate_ok"] = gate.ok
    merged.attrs["gate_message"] = gate.message

    # Stable ordering.
    sort_cols = ["tier"]
    if "delta_balanced_accuracy" in merged.columns:
        sort_cols.append("delta_balanced_accuracy")
    merged = merged.sort_values(sort_cols, ascending=[True, False]).reset_index(drop=True)
    return merged


def _tier_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    if "tier" not in deltas.columns:
        return pd.DataFrame()
    cols = [c for c in ["delta_balanced_accuracy", "delta_macro_f1", "delta_hybrid"] if c in deltas.columns]
    if not cols:
        return pd.DataFrame()
    return deltas.groupby("tier")[cols].agg(["mean", "median", "min", "max", "count"]).reset_index()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run", type=Path, required=True, help="Baseline run dir or df_fs_summary.csv")
    ap.add_argument("--candidate-run", type=Path, required=True, help="Candidate run dir or df_fs_summary.csv")
    ap.add_argument("--baseline-config", default="baseline")
    ap.add_argument("--candidate-config", default="baseline")
    ap.add_argument("--protocol", default="holdout")
    ap.add_argument("--easy-cap-ba", type=float, default=0.02)
    ap.add_argument("--medium-cap-ba", type=float, default=0.05)
    ap.add_argument("--hard-min-mean-ba", type=float, default=0.0)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args(argv)

    deltas = compare_runs(
        baseline_run_or_csv=args.baseline_run,
        candidate_run_or_csv=args.candidate_run,
        baseline_config=args.baseline_config,
        candidate_config=args.candidate_config,
        protocol=args.protocol,
        easy_cap_ba=args.easy_cap_ba,
        medium_cap_ba=args.medium_cap_ba,
        hard_min_mean_ba=args.hard_min_mean_ba,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(args.out_csv, index=False)

    tier_summary = _tier_summary(deltas)
    gate_ok = bool(deltas.attrs.get("gate_ok", False))
    gate_msg = str(deltas.attrs.get("gate_message", ""))

    print(f"Compared {len(deltas)} datasets. Gate: {'PASS' if gate_ok else 'FAIL'} ({gate_msg})")
    if not tier_summary.empty:
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
            print(tier_summary.to_string(index=False))
    print(f"Wrote: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

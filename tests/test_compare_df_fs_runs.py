from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabnetics.benchmarks.compare_runs import compare_runs


def _write_summary_csv(p: Path, *, ba: float) -> None:
    df = pd.DataFrame(
        [
            {
                "dataset_id": "synthetic_easy_dfshift",
                "config": "baseline",
                "protocol": "holdout",
                "balanced_accuracy_mean": ba,
                "macro_f1_mean": ba - 0.01,
                "hybrid_score_mean": ba - 0.005,
                "fs_time_sec_mean": 1.0,
                "dist_time_sec_mean": 2.0,
                "transform_time_sec_mean": 3.0,
            }
        ]
    )
    df.to_csv(p, index=False)


def test_compare_runs_emits_basic_deltas(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    cand = tmp_path / "cand.csv"
    _write_summary_csv(base, ba=0.50)
    _write_summary_csv(cand, ba=0.55)

    deltas = compare_runs(baseline_run_or_csv=base, candidate_run_or_csv=cand)
    assert len(deltas) == 1
    row = deltas.iloc[0].to_dict()
    assert row["dataset_id"] == "synthetic_easy_dfshift"
    assert row["tier"] == "easy"
    assert abs(row["delta_balanced_accuracy"] - 0.05) < 1e-12


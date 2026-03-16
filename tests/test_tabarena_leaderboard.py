from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.benchmarking.tabarena_leaderboard import (
    _compute_results_per_task,
    _compute_winrate,
    build_comparison_artifacts,
    build_tabnetics_rows,
)


def test_build_tabnetics_rows_broadcasts_across_official_folds() -> None:
    official = pd.DataFrame(
        [
            {"dataset": "d1", "fold": 0, "method": "RF (default)", "metric_error": 0.3, "problem_type": "binary", "metric": "roc_auc"},
            {"dataset": "d1", "fold": 1, "method": "RF (default)", "metric_error": 0.4, "problem_type": "binary", "metric": "roc_auc"},
            {"dataset": "d2", "fold": 0, "method": "RF (default)", "metric_error": 0.2, "problem_type": "multiclass", "metric": "log_loss"},
        ]
    )
    local = pd.DataFrame(
        [
            {"dataset_id": "d1", "problem_type": "binary", "metric": "roc_auc", "status": "ok", "metric_error": 0.11, "balanced_accuracy": 0.9, "roc_auc": 0.89, "log_loss": 0.3, "elapsed_sec": 10.0, "selected_features": 5},
            {"dataset_id": "d2", "problem_type": "multiclass", "metric": "log_loss", "status": "ok", "metric_error": 0.55, "balanced_accuracy": 0.7, "roc_auc": 0.0, "log_loss": 0.55, "elapsed_sec": 12.0, "selected_features": 7},
        ]
    )

    rows = build_tabnetics_rows(local, official, method_name="tabnetics-test")

    assert len(rows) == 3
    assert sorted(rows["fold"].tolist()) == [0, 0, 1]
    assert set(rows["method"].unique()) == {"tabnetics-test"}
    assert rows.loc[rows["dataset"] == "d1", "metric_error"].tolist() == [0.11, 0.11]


def test_build_tabnetics_rows_keeps_local_fold_rows_without_broadcast() -> None:
    official = pd.DataFrame(
        [
            {"dataset": "d1", "fold": 0, "method": "RF (default)", "metric_error": 0.3, "problem_type": "binary", "metric": "roc_auc"},
            {"dataset": "d1", "fold": 1, "method": "RF (default)", "metric_error": 0.4, "problem_type": "binary", "metric": "roc_auc"},
        ]
    )
    local = pd.DataFrame(
        [
            {"dataset_id": "d1", "problem_type": "binary", "metric": "roc_auc", "fold": 0, "status": "ok", "metric_error": 0.11, "balanced_accuracy": 0.9, "roc_auc": 0.89, "log_loss": 0.3, "elapsed_sec": 10.0, "selected_features": 5},
            {"dataset_id": "d1", "problem_type": "binary", "metric": "roc_auc", "fold": 1, "status": "ok", "metric_error": 0.22, "balanced_accuracy": 0.8, "roc_auc": 0.78, "log_loss": 0.4, "elapsed_sec": 11.0, "selected_features": 6},
        ]
    )

    rows = build_tabnetics_rows(local, official, method_name="tabnetics-test")

    assert len(rows) == 2
    assert sorted(rows["fold"].tolist()) == [0, 1]
    assert rows.sort_values("fold")["metric_error"].tolist() == [0.11, 0.22]


def test_winrate_uses_equal_dataset_weighting_across_folds() -> None:
    df = pd.DataFrame(
        [
            {"dataset": "d1", "fold": 0, "method": "A", "metric_error": 0.1, "metric": "roc_auc", "problem_type": "binary"},
            {"dataset": "d1", "fold": 0, "method": "B", "metric_error": 0.2, "metric": "roc_auc", "problem_type": "binary"},
            {"dataset": "d1", "fold": 1, "method": "A", "metric_error": 0.1, "metric": "roc_auc", "problem_type": "binary"},
            {"dataset": "d1", "fold": 1, "method": "B", "metric_error": 0.2, "metric": "roc_auc", "problem_type": "binary"},
            {"dataset": "d2", "fold": 0, "method": "A", "metric_error": 0.8, "metric": "roc_auc", "problem_type": "binary"},
            {"dataset": "d2", "fold": 0, "method": "B", "metric_error": 0.2, "metric": "roc_auc", "problem_type": "binary"},
        ]
    )

    results_per_task = _compute_results_per_task(df)
    winrate = _compute_winrate(results_per_task)

    assert math.isclose(float(winrate["A"]), 0.5, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(winrate["B"]), 0.5, rel_tol=0.0, abs_tol=1e-9)


def test_build_comparison_artifacts_tracks_dataset_best_wins_and_losses() -> None:
    official = pd.DataFrame(
        [
            {"dataset": "d1", "fold": 0, "method": "RF (default)", "metric_error": 0.20, "time_train_s": 1.0, "time_infer_s": 0.1, "metric_error_val": 0.2, "seed": 0, "problem_type": "binary", "metric": "roc_auc"},
            {"dataset": "d1", "fold": 0, "method": "ModelB", "metric_error": 0.10, "time_train_s": 1.0, "time_infer_s": 0.1, "metric_error_val": 0.1, "seed": 0, "problem_type": "binary", "metric": "roc_auc"},
            {"dataset": "d2", "fold": 0, "method": "RF (default)", "metric_error": 0.30, "time_train_s": 1.0, "time_infer_s": 0.1, "metric_error_val": 0.3, "seed": 0, "problem_type": "multiclass", "metric": "log_loss"},
            {"dataset": "d2", "fold": 0, "method": "ModelB", "metric_error": 0.40, "time_train_s": 1.0, "time_infer_s": 0.1, "metric_error_val": 0.4, "seed": 0, "problem_type": "multiclass", "metric": "log_loss"},
        ]
    )
    local = pd.DataFrame(
        [
            {"dataset_id": "d1", "problem_type": "binary", "metric": "roc_auc", "seed": 42, "status": "ok", "metric_error": 0.05, "balanced_accuracy": 0.9, "roc_auc": 0.95, "log_loss": 0.2, "model": "mnpo_lr", "elapsed_sec": 10.0, "selected_features": 5},
            {"dataset_id": "d2", "problem_type": "multiclass", "metric": "log_loss", "seed": 42, "status": "ok", "metric_error": 0.50, "balanced_accuracy": 0.6, "roc_auc": 0.0, "log_loss": 0.5, "model": "mnpo_rf", "elapsed_sec": 11.0, "selected_features": 6},
        ]
    )

    artifacts = build_comparison_artifacts(
        local,
        official_results=official,
        method_name="tabnetics-test",
        bootstrap_rounds=8,
    )

    assert artifacts.summary["datasets_covered"] == 2
    assert artifacts.summary["wins_vs_dataset_best"] == 1
    assert artifacts.summary["losses_vs_dataset_best"] == 1
    assert "tabnetics-test" in set(artifacts.leaderboard["method"].unique())

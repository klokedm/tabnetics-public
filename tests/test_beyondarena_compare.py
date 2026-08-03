from __future__ import annotations

import math

import pandas as pd
import pytest

import tabnetics.benchmarks.beyondarena_compare as beyondarena_compare
from tabnetics.benchmarks.beyondarena_compare import (
    OFFICIAL_BEST_TFM_METHOD,
    append_official_best_tfm_rows,
    build_beyondarena_comparison_artifacts,
    build_beyondarena_summary_table,
    inspect_official_beyondarena_results,
    join_beyondarena_results,
    load_public_beyondarena_r2_results,
    normalize_beyondarena_method_name,
    normalize_beyondarena_result_table,
    summarize_beyondarena_pairs,
)


def test_method_name_normalization_preserves_beyondarena_paper_names() -> None:
    assert normalize_beyondarena_method_name("tabpfn") == "TabPFN-2.6"
    assert normalize_beyondarena_method_name("TA-TabPFN-2.6_c1_BAG_L1") == "TabPFN-2.6"
    assert normalize_beyondarena_method_name("LinearModel_c1_BAG_L1") == "Linear/Logistic Regression"
    assert normalize_beyondarena_method_name("tabm_official") == "TabM"
    assert normalize_beyondarena_method_name("TabICL") == "TabICLv2"
    assert normalize_beyondarena_method_name("custom-local") == "custom-local"


def test_official_status_blocks_exact_claims_when_unconfigured() -> None:
    status = inspect_official_beyondarena_results()

    assert status.available is False
    assert status.exact_paired_rows_available is False
    assert "exact paired parity claims are blocked" in status.reason


def test_public_r2_loader_filters_default_config_without_network(monkeypatch) -> None:
    def fake_read_public_r2(method_path: str) -> pd.DataFrame:
        assert method_path == "TA-TabPFN-2.6"
        return pd.DataFrame(
            [
                {
                    "dataset": "d1",
                    "fold": 0,
                    "method": "TA-TabPFN-2.6_c1_BAG_L1",
                    "metric": "roc_auc",
                    "metric_error": 0.20,
                    "problem_type": "binary",
                    "ta_name": "TA-TabPFN-2.6",
                    "ta_suite": "beyond_iid_benchmark_2026",
                },
                {
                    "dataset": "d1",
                    "fold": 0,
                    "method": "TA-TabPFN-2.6_r1_BAG_L1",
                    "metric": "roc_auc",
                    "metric_error": 0.19,
                    "problem_type": "binary",
                    "ta_name": "TA-TabPFN-2.6",
                    "ta_suite": "beyond_iid_benchmark_2026",
                },
            ]
        )

    monkeypatch.setattr(beyondarena_compare, "_read_public_r2_parquet", fake_read_public_r2)

    rows = load_public_beyondarena_r2_results(methods=("TabPFN-2.6",))

    assert len(rows) == 1
    assert rows.loc[0, "dataset_id"] == "d1"
    assert rows.loc[0, "split_id"] == "0"
    assert rows.loc[0, "method"] == "TabPFN-2.6"
    assert rows.loc[0, "metric_value"] == 0.20
    assert rows.loc[0, "metric_error"] == 0.20
    assert rows.loc[0, "metric_value_semantics"] == "error"
    assert bool(rows.loc[0, "lower_is_better"]) is True

    renormalized = normalize_beyondarena_result_table(rows, origin="official")

    assert renormalized.loc[0, "metric_value"] == 0.20
    assert renormalized.loc[0, "metric_error"] == 0.20
    assert renormalized.loc[0, "metric_value_semantics"] == "error"
    assert bool(renormalized.loc[0, "lower_is_better"]) is True


def test_public_r2_error_rows_join_raw_local_metric_rows() -> None:
    official = pd.DataFrame(
        [
            {
                "dataset_id": "d1",
                "split_id": "0",
                "method": "TabPFN-2.6",
                "metric": "roc_auc",
                "metric_value": 0.20,
                "metric_error": 0.20,
                "lower_is_better": True,
                "status": "ok",
            }
        ]
    )
    local = pd.DataFrame(
        [
            {
                "dataset_id": "d1",
                "split_id": "0",
                "method": "TabenticsDiakrino",
                "metric": "roc_auc",
                "metric_value": 0.82,
                "metric_error": 0.18,
                "lower_is_better": False,
                "status": "ok",
            }
        ]
    )

    joined = join_beyondarena_results(official, local)

    assert len(joined) == 1
    assert joined.loc[0, "comparison_value_official"] == 0.20
    assert joined.loc[0, "comparison_value_local"] == 0.18
    assert joined.loc[0, "comparison_value_semantics"] == "error"
    assert bool(joined.loc[0, "comparison_lower_is_better"]) is True
    assert math.isclose(float(joined.loc[0, "comparison_value_delta"]), -0.02)
    assert math.isclose(float(joined.loc[0, "comparison_delta"]), 0.02)
    assert math.isclose(float(joined.loc[0, "source_value_delta"]), 0.62)
    assert pd.isna(joined.loc[0, "raw_metric_delta"])
    assert joined.loc[0, "outcome"] == "win"


def test_direction_mismatch_without_common_error_representation_fails_closed() -> None:
    official = pd.DataFrame(
        [
            {
                "dataset_id": "d1",
                "split_id": "0",
                "method": "TabPFN-2.6",
                "metric": "roc_auc",
                "metric_value": 0.20,
                "lower_is_better": True,
                "status": "ok",
            }
        ]
    )
    local = pd.DataFrame(
        [
            {
                "dataset_id": "d1",
                "split_id": "0",
                "method": "TabenticsDiakrino",
                "metric": "roc_auc",
                "metric_value": 0.82,
                "lower_is_better": False,
                "status": "ok",
            }
        ]
    )

    with pytest.raises(ValueError, match="Metric direction/representation mismatch"):
        join_beyondarena_results(official, local)


def test_non_finite_primary_values_without_finite_error_fallback_fail_closed() -> None:
    official = normalize_beyondarena_result_table(
        pd.DataFrame(
            [
                {
                    "dataset_id": "d1",
                    "split_id": "0",
                    "method": "TabPFN-2.6",
                    "metric": "roc_auc",
                    "score": 0.80,
                }
            ]
        ),
        origin="official",
    )
    local = normalize_beyondarena_result_table(
        pd.DataFrame(
            [
                {
                    "dataset_id": "d1",
                    "split_id": "0",
                    "method": "TabenticsDiakrino",
                    "metric": "roc_auc",
                    "score": float("inf"),
                }
            ]
        ),
        origin="tabnetics",
    )

    with pytest.raises(ValueError, match="Metric direction/representation mismatch"):
        join_beyondarena_results(official, local)


def test_append_official_best_tfm_rows_respects_metric_direction() -> None:
    official = normalize_beyondarena_result_table(
        pd.DataFrame(
            [
                {
                    "dataset": "d1",
                    "split_id": "0",
                    "method": "TabPFN-2.6",
                    "metric": "roc_auc",
                    "score": 0.80,
                },
                {
                    "dataset": "d1",
                    "split_id": "0",
                    "method": "TabICLv2",
                    "metric": "roc_auc",
                    "score": 0.85,
                },
                {
                    "dataset": "d2",
                    "split_id": "0",
                    "method": "TabDPT",
                    "metric": "log_loss",
                    "score": 0.30,
                },
                {
                    "dataset": "d2",
                    "split_id": "0",
                    "method": "TabPFN-2.6",
                    "metric": "log_loss",
                    "score": 0.40,
                },
            ]
        ),
        origin="official",
    )

    augmented = append_official_best_tfm_rows(official)
    best = augmented[augmented["method"].eq(OFFICIAL_BEST_TFM_METHOD)]

    assert len(best) == 2
    by_dataset = {row["dataset_id"]: row for _, row in best.iterrows()}
    assert by_dataset["d1"]["best_tfm_source_method"] == "TabICLv2"
    assert by_dataset["d1"]["metric_value"] == 0.85
    assert by_dataset["d2"]["best_tfm_source_method"] == "TabDPT"
    assert by_dataset["d2"]["metric_value"] == 0.30


def test_append_official_best_tfm_rows_minimizes_public_error_values() -> None:
    public_errors = normalize_beyondarena_result_table(
        pd.DataFrame(
            [
                {
                    "dataset_id": "d1",
                    "split_id": "0",
                    "method": "TabPFN-2.6",
                    "metric": "roc_auc",
                    "metric_value": 0.20,
                    "metric_error": 0.20,
                    "metric_value_semantics": "error",
                    "lower_is_better": True,
                },
                {
                    "dataset_id": "d1",
                    "split_id": "0",
                    "method": "TabICLv2",
                    "metric": "roc_auc",
                    "metric_value": 0.15,
                    "metric_error": 0.15,
                    "metric_value_semantics": "error",
                    "lower_is_better": "True",
                },
            ]
        ),
        origin="official",
    )

    augmented = append_official_best_tfm_rows(public_errors)
    best = augmented[augmented["method"].eq(OFFICIAL_BEST_TFM_METHOD)].iloc[0]

    assert best["best_tfm_source_method"] == "TabICLv2"
    assert best["metric_value"] == 0.15
    assert best["metric_value_semantics"] == "error"
    assert bool(best["lower_is_better"]) is True


def test_paired_join_handles_metric_direction_and_skip_rows() -> None:
    official_raw = pd.DataFrame(
        [
            {
                "dataset": "d1",
                "split_id": "0:0",
                "method": "tabpfn",
                "metric": "roc_auc",
                "score": 0.80,
                "task_type": "iid",
            },
            {
                "dataset": "d1",
                "split_id": "0:1",
                "method": "tabpfn",
                "metric": "roc_auc",
                "score": 0.70,
                "task_type": "iid",
            },
            {
                "dataset": "d2",
                "split_id": "0:0",
                "method": "TabM",
                "metric": "log_loss",
                "score": 0.40,
                "task_type": "temporal",
            },
        ]
    )
    local_raw = pd.DataFrame(
        [
            {
                "dataset_id": "d1",
                "split_id": "0:0",
                "method": "TabenticsDiakrino",
                "model_name": "TabenticsDiakrinoFSClassifier",
                "metric": "roc_auc",
                "metric_value": 0.82,
                "status": "ok",
            },
            {
                "dataset_id": "d1",
                "split_id": "0:1",
                "method": "TabenticsDiakrino",
                "metric": "roc_auc",
                "metric_value": 0.69,
                "status": "ok",
                "seed": 42,
                "device": "gpu",
                "execution_host": "public-gpu-host",
                "execution_lane": "gpu",
                "execution_status": "ok",
                "execution_backend": "tabnetics-diakrino",
                "allow_gpu_execution": True,
                "local_dataset_id": "local-d1",
                "local_split_id": "r0f1",
            },
            {
                "dataset_id": "d2",
                "split_id": "0:0",
                "method": "tabnetics-current",
                "metric": "log_loss",
                "metric_value": 0.35,
                "status": "ok",
            },
            {
                "dataset_id": "d2",
                "split_id": "0:1",
                "method": "tabnetics-current",
                "metric": "log_loss",
                "metric_value": 0.10,
                "status": "skipped",
            },
        ]
    )
    official = normalize_beyondarena_result_table(official_raw, origin="official")
    local = normalize_beyondarena_result_table(local_raw, origin="tabnetics")

    joined = join_beyondarena_results(official, local)

    assert len(joined) == 3
    assert joined["outcome"].tolist() == ["win", "loss", "win"]
    assert joined["comparison_value_semantics"].eq("metric").all()
    assert joined.loc[joined["metric"] == "log_loss", "comparison_delta"].iloc[0] > 0
    assert set(joined["method_official"]) == {"TabPFN-2.6", "TabM"}
    assert joined.loc[joined["split_id"].eq("0:0"), "method_local"].iloc[0] == "TabenticsDiakrino"
    assert joined.loc[joined["split_id"].eq("0:0"), "model_name"].iloc[0] == "TabenticsDiakrinoFSClassifier"
    diakrino_seeded = joined[joined["split_id"].eq("0:1")].iloc[0]
    assert diakrino_seeded["seed"] == 42
    assert diakrino_seeded["device"] == "gpu"
    assert diakrino_seeded["execution_host"] == "public-gpu-host"
    assert diakrino_seeded["execution_lane"] == "gpu"
    assert diakrino_seeded["execution_status"] == "ok"
    assert diakrino_seeded["execution_backend"] == "tabnetics-diakrino"
    assert bool(diakrino_seeded["allow_gpu_execution"]) is True
    assert diakrino_seeded["local_dataset_id"] == "local-d1"
    assert diakrino_seeded["local_split_id"] == "r0f1"


def test_summary_reports_wtl_and_wilcoxon_columns() -> None:
    joined = pd.DataFrame(
        {
            "comparison_delta": [0.2, -0.1, 0.0],
            "outcome": ["win", "loss", "tie"],
            "task_type_official": ["iid", "iid", "grouped"],
        }
    )

    overall = summarize_beyondarena_pairs(joined)
    by_task = summarize_beyondarena_pairs(joined, group_cols=("task_type_official",))

    assert overall.loc[0, "wins"] == 1
    assert overall.loc[0, "ties"] == 1
    assert overall.loc[0, "losses"] == 1
    assert math.isclose(float(overall.loc[0, "mean_delta"]), (0.2 - 0.1) / 3.0)
    assert set(by_task["task_type_official"]) == {"iid", "grouped"}
    assert "wilcoxon_p" in overall.columns


def test_build_comparison_artifacts_emits_joined_and_subgroup_summary() -> None:
    official = pd.DataFrame(
        [
            {
                "dataset": "d1",
                "split_id": "0:0",
                "method": "TabPFN-2.6",
                "metric": "roc_auc",
                "score": 0.80,
                "task_type": "iid",
            },
            {
                "dataset": "d2",
                "split_id": "0:0",
                "method": "TabM",
                "metric": "log_loss",
                "score": 0.40,
                "task_type": "temporal",
            },
        ]
    )
    local = pd.DataFrame(
        [
            {"dataset": "d1", "split_id": "0:0", "method": "TabenticsDiakrino", "metric": "roc_auc", "score": 0.81},
            {"dataset": "d2", "split_id": "0:0", "method": "tabnetics-current", "metric": "log_loss", "score": 0.45},
        ]
    )

    artifacts = build_beyondarena_comparison_artifacts(official, local)

    assert artifacts.status.exact_paired_rows_available is True
    assert len(artifacts.joined) == 3
    assert OFFICIAL_BEST_TFM_METHOD in set(artifacts.joined["method_official"])
    assert not artifacts.summary.empty
    assert {"summary_scope", "summary_value"}.issubset(artifacts.summary.columns)
    assert "all" in set(artifacts.summary["summary_scope"])
    assert "task_type" in set(artifacts.summary["summary_scope"])


def test_summary_table_is_method_pair_and_local_subgroup_aware() -> None:
    official = pd.DataFrame(
        [
            {"dataset": "d1", "split_id": "0", "method": "TabPFN-2.6", "metric": "roc_auc", "score": 0.80},
            {"dataset": "d2", "split_id": "0", "method": "TabPFN-2.6", "metric": "roc_auc", "score": 0.70},
            {"dataset": "d1", "split_id": "0", "method": "TabM", "metric": "roc_auc", "score": 0.78},
        ]
    )
    local = pd.DataFrame(
        [
            {
                "dataset_id": "d1",
                "split_id": "0",
                "method": "TabenticsDiakrino",
                "metric": "roc_auc",
                "score": 0.82,
                "task_type": "iid",
                "problem_type": "classification",
                "size_tier": "tiny",
                "dimensionality": "low",
                "has_text": False,
                "high_cardinality": False,
            },
            {
                "dataset_id": "d2",
                "split_id": "0",
                "method": "TabenticsDiakrino",
                "metric": "roc_auc",
                "score": 0.68,
                "task_type": "temporal",
                "problem_type": "classification",
                "size_tier": "small",
                "dimensionality": "high",
                "has_text": True,
                "high_cardinality": True,
            },
            {
                "dataset_id": "d1",
                "split_id": "0",
                "method": "tabnetics-current",
                "metric": "roc_auc",
                "score": 0.79,
                "task_type": "iid",
                "problem_type": "classification",
                "size_tier": "tiny",
                "dimensionality": "low",
                "has_text": False,
                "high_cardinality": False,
            },
        ]
    )

    artifacts = build_beyondarena_comparison_artifacts(official, local)
    summary = artifacts.summary

    assert "summary_task_type" in artifacts.joined.columns
    assert "summary_size_tier" in artifacts.joined.columns
    expected_scopes = {
        "all",
        "task_type",
        "problem_type",
        "size_tier",
        "dimensionality",
        "has_text",
        "high_cardinality",
    }
    assert expected_scopes.issubset(set(summary["summary_scope"]))

    all_rows = summary[summary["summary_scope"].eq("all")]
    assert len(all_rows) == 6
    assert OFFICIAL_BEST_TFM_METHOD in set(all_rows["method_official"])
    diakrino_all = all_rows[
        all_rows["method_official"].eq("TabPFN-2.6")
        & all_rows["method_local"].eq("TabenticsDiakrino")
    ].iloc[0]
    assert diakrino_all["summary_value"] == "all"
    assert int(diakrino_all["n_pairs"]) == 2
    assert int(diakrino_all["wins"]) == 1
    assert int(diakrino_all["losses"]) == 1

    task_rows = summary[
        summary["summary_scope"].eq("task_type")
        & summary["method_official"].eq("TabPFN-2.6")
        & summary["method_local"].eq("TabenticsDiakrino")
    ]
    assert set(task_rows["summary_value"]) == {"iid", "temporal"}


def test_summary_table_handles_empty_joined_rows() -> None:
    summary = build_beyondarena_summary_table(pd.DataFrame())

    assert list(summary.columns) == [
        "summary_scope",
        "summary_value",
        "n_pairs",
        "wins",
        "ties",
        "losses",
        "mean_delta",
        "median_delta",
        "wilcoxon_p",
    ]
    assert summary.empty

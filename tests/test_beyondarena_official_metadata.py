from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tabnetics.benchmarks.beyondarena_compare import metric_lower_is_better
from tabnetics.datasets.beyondarena import (
    build_beyondarena_current_feasibility_task_rows,
    build_beyondarena_smoke_task_rows,
    load_beyondarena_core_tasks_csv,
    load_beyondarena_task_metadata_csv,
    select_beyondarena_core_dataset_split_rows,
    select_beyondarena_core_task_rows,
)


FIXTURES = Path(__file__).parent / "fixtures" / "beyondarena" / "official_metadata"


def test_official_task_metadata_csv_loads_split_grain_without_parquet() -> None:
    rows = load_beyondarena_task_metadata_csv(FIXTURES / "BeyondArena_tasks_metadata.csv.fixture")
    by_split = {(row.tabarena_task_name, row.split_index): row for row in rows}

    assert len(rows) == 10
    assert len({row.tabarena_task_name for row in rows}) == 6
    assert by_split[("simple_iid_regression-aaaa", "r0f1")].split == 1
    assert by_split[("grouped_energy-bbbb", "r0f2")].split == 2
    assert by_split[("temporal_sales-cccc", "r1f0")].split == 1
    assert by_split[("simple_iid_regression-aaaa", "r0f0")].normalized_task_type == "iid"

    spec = by_split[("wide_genomics-ffff", "r0f0")].to_dataset_spec()
    assert spec.artifact_dir is None
    assert spec.has_dataset is False
    assert spec.is_high_dimensional is True
    assert spec.skip_reason == "official metadata row only; local dataset.parquet not materialized"


def test_core_subset_selects_by_task_name_and_normalized_split() -> None:
    metadata = load_beyondarena_task_metadata_csv(FIXTURES / "BeyondArena_tasks_metadata.csv.fixture")
    core = load_beyondarena_core_tasks_csv(FIXTURES / "BeyondArena_core_tasks.csv.fixture")

    selected = select_beyondarena_core_task_rows(metadata, core)

    assert [row.key for row in selected] == [task.key for task in core]
    assert selected[1].tabarena_task_name == "grouped_energy-bbbb"
    assert selected[1].split_index == "r0f1"
    assert selected[2].tabarena_task_name == "temporal_sales-cccc"
    assert selected[2].split_index == "r1f0"


def test_core_dataset_split_subset_expands_all_official_rows_for_core_datasets() -> None:
    metadata = load_beyondarena_task_metadata_csv(FIXTURES / "BeyondArena_tasks_metadata.csv.fixture")
    core = load_beyondarena_core_tasks_csv(FIXTURES / "BeyondArena_core_tasks.csv.fixture")

    selected = select_beyondarena_core_dataset_split_rows(metadata, core)
    by_dataset = {}
    for row in selected:
        by_dataset.setdefault(row.tabarena_task_name, []).append(row.split_index)

    assert len(selected) == 10
    assert by_dataset["grouped_energy-bbbb"] == ["r0f0", "r0f1", "r0f2"]
    assert by_dataset["temporal_sales-cccc"] == ["r0f0", "r1f0"]
    assert by_dataset["simple_iid_regression-aaaa"] == ["r0f0", "r0f1"]


def test_core_subset_strict_mode_reports_missing_rows() -> None:
    metadata = load_beyondarena_task_metadata_csv(FIXTURES / "BeyondArena_tasks_metadata.csv.fixture")
    missing_core = load_beyondarena_core_tasks_csv(
        FIXTURES / "BeyondArena_core_tasks.csv.fixture"
    ) + load_beyondarena_core_tasks_csv(
        pd.DataFrame({"dataset": ["missing-dataset"], "split": [0]})
    )

    with pytest.raises(ValueError, match="not present in task metadata"):
        select_beyondarena_core_task_rows(metadata, missing_core)


def test_smoke_task_rows_cover_official_metadata_facets() -> None:
    rows = load_beyondarena_task_metadata_csv(FIXTURES / "BeyondArena_tasks_metadata.csv.fixture")

    smoke = build_beyondarena_smoke_task_rows(rows)

    assert len(smoke) == 6
    assert any(row.normalized_task_type == "iid" for row in smoke)
    assert any(row.normalized_task_type == "grouped" for row in smoke)
    assert any(row.normalized_task_type == "temporal" for row in smoke)
    assert any(row.has_text_features for row in smoke)
    assert any(row.has_high_cardinality_features for row in smoke)
    assert any(row.is_high_dimensional for row in smoke)


def test_smoke_task_rows_prefer_lower_cost_facet_matches() -> None:
    rows = load_beyondarena_task_metadata_csv(
        pd.DataFrame(
            [
                {
                    "dataset_name": "plain_iid",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_name": "label",
                    "eval_metric": "roc_auc",
                    "tabarena_task_name": "plain_iid-0001",
                    "task_id_str": "UserTask|1|plain_iid/0001",
                    "data_foundry_uri": "plain_iid/0001",
                    "task_type": "random",
                    "repeat": 0,
                    "fold": 0,
                    "has_text": False,
                    "has_high_cardinality_categorical": False,
                    "num_instances_train": 100,
                    "num_instances_test": 50,
                    "num_cols_after_preprocessing": 10,
                },
                {
                    "dataset_name": "aaa_huge_grouped",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_name": "label",
                    "eval_metric": "roc_auc",
                    "tabarena_task_name": "aaa_huge_grouped-0002",
                    "task_id_str": "UserTask|2|aaa_huge_grouped/0002",
                    "data_foundry_uri": "aaa_huge_grouped/0002",
                    "task_type": "grouped",
                    "repeat": 0,
                    "fold": 0,
                    "has_text": False,
                    "has_high_cardinality_categorical": False,
                    "num_instances_train": 1_000_000,
                    "num_instances_test": 250_000,
                    "num_cols_after_preprocessing": 200,
                },
                {
                    "dataset_name": "zzz_tiny_grouped",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_name": "label",
                    "eval_metric": "roc_auc",
                    "tabarena_task_name": "zzz_tiny_grouped-0003",
                    "task_id_str": "UserTask|3|zzz_tiny_grouped/0003",
                    "data_foundry_uri": "zzz_tiny_grouped/0003",
                    "task_type": "grouped",
                    "repeat": 0,
                    "fold": 0,
                    "has_text": False,
                    "has_high_cardinality_categorical": False,
                    "num_instances_train": 120,
                    "num_instances_test": 60,
                    "num_cols_after_preprocessing": 12,
                },
                {
                    "dataset_name": "temporal_small",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_name": "label",
                    "eval_metric": "roc_auc",
                    "tabarena_task_name": "temporal_small-0004",
                    "task_id_str": "UserTask|4|temporal_small/0004",
                    "data_foundry_uri": "temporal_small/0004",
                    "task_type": "temporal",
                    "repeat": 0,
                    "fold": 0,
                    "has_text": False,
                    "has_high_cardinality_categorical": False,
                    "num_instances_train": 130,
                    "num_instances_test": 65,
                    "num_cols_after_preprocessing": 14,
                },
                {
                    "dataset_name": "text_small",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_name": "label",
                    "eval_metric": "roc_auc",
                    "tabarena_task_name": "text_small-0005",
                    "task_id_str": "UserTask|5|text_small/0005",
                    "data_foundry_uri": "text_small/0005",
                    "task_type": "random",
                    "repeat": 0,
                    "fold": 0,
                    "has_text": True,
                    "has_high_cardinality_categorical": False,
                    "num_instances_train": 140,
                    "num_instances_test": 70,
                    "num_cols_after_preprocessing": 110,
                },
                {
                    "dataset_name": "merchant_small",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_name": "label",
                    "eval_metric": "roc_auc",
                    "tabarena_task_name": "merchant_small-0006",
                    "task_id_str": "UserTask|6|merchant_small/0006",
                    "data_foundry_uri": "merchant_small/0006",
                    "task_type": "random",
                    "repeat": 0,
                    "fold": 0,
                    "has_text": False,
                    "has_high_cardinality_categorical": True,
                    "num_instances_train": 150,
                    "num_instances_test": 75,
                    "num_cols_after_preprocessing": 16,
                },
                {
                    "dataset_name": "wide_small",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_name": "label",
                    "eval_metric": "roc_auc",
                    "tabarena_task_name": "wide_small-0007",
                    "task_id_str": "UserTask|7|wide_small/0007",
                    "data_foundry_uri": "wide_small/0007",
                    "task_type": "random",
                    "repeat": 0,
                    "fold": 0,
                    "has_text": False,
                    "has_high_cardinality_categorical": False,
                    "num_instances_train": 160,
                    "num_instances_test": 80,
                    "num_cols_after_preprocessing": 500,
                },
            ]
        )
    )

    smoke = build_beyondarena_smoke_task_rows(rows)

    assert "aaa_huge_grouped-0002" not in {row.tabarena_task_name for row in smoke}
    assert any(row.tabarena_task_name == "zzz_tiny_grouped-0003" for row in smoke)


def test_current_feasibility_task_rows_prefer_small_plain_classification() -> None:
    rows = load_beyondarena_task_metadata_csv(
        pd.DataFrame(
            [
                {
                    "dataset_name": "wide_text",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_name": "label",
                    "eval_metric": "roc_auc",
                    "tabarena_task_name": "wide_text-aaaa",
                    "task_id_str": "UserTask|1|wide_text/aaaa",
                    "data_foundry_uri": "wide_text/aaaa",
                    "task_type": "random",
                    "repeat": 0,
                    "fold": 0,
                    "has_text": True,
                    "has_high_cardinality_categorical": False,
                    "num_instances_train": 80,
                    "num_instances_test": 40,
                    "num_cols_after_preprocessing": 500,
                },
                {
                    "dataset_name": "tiny_iid",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_name": "label",
                    "eval_metric": "roc_auc",
                    "tabarena_task_name": "tiny_iid-bbbb",
                    "task_id_str": "UserTask|2|tiny_iid/bbbb",
                    "data_foundry_uri": "tiny_iid/bbbb",
                    "task_type": "random",
                    "repeat": 0,
                    "fold": 0,
                    "has_text": False,
                    "has_high_cardinality_categorical": False,
                    "num_instances_train": 120,
                    "num_instances_test": 40,
                    "num_cols_after_preprocessing": 12,
                },
                {
                    "dataset_name": "tiny_regression",
                    "problem_type": "regression",
                    "is_classification": False,
                    "target_name": "y",
                    "eval_metric": "root_mean_squared_error",
                    "tabarena_task_name": "tiny_regression-cccc",
                    "task_id_str": "UserTask|3|tiny_regression/cccc",
                    "data_foundry_uri": "tiny_regression/cccc",
                    "task_type": "random",
                    "repeat": 0,
                    "fold": 0,
                    "has_text": False,
                    "has_high_cardinality_categorical": False,
                    "num_instances_train": 20,
                    "num_instances_test": 20,
                    "num_cols_after_preprocessing": 5,
                },
                {
                    "dataset_name": "small_grouped",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_name": "label",
                    "eval_metric": "roc_auc",
                    "tabarena_task_name": "small_grouped-dddd",
                    "task_id_str": "UserTask|4|small_grouped/dddd",
                    "data_foundry_uri": "small_grouped/dddd",
                    "task_type": "grouped",
                    "repeat": 0,
                    "fold": 0,
                    "has_text": False,
                    "has_high_cardinality_categorical": False,
                    "num_instances_train": 90,
                    "num_instances_test": 30,
                    "num_cols_after_preprocessing": 10,
                },
            ]
        )
    )

    selected = build_beyondarena_current_feasibility_task_rows(rows)

    assert len(selected) == 1
    assert selected[0].tabarena_task_name == "tiny_iid-bbbb"
    assert selected[0].normalized_problem_type == "classification"


def test_official_metric_directions_cover_beyondarena_problem_types() -> None:
    rows = load_beyondarena_task_metadata_csv(FIXTURES / "BeyondArena_tasks_metadata.csv.fixture")
    by_metric = {row.eval_metric: row for row in rows}

    assert by_metric["roc_auc"].metric_lower_is_better is False
    assert by_metric["log_loss"].metric_lower_is_better is True
    assert by_metric["root_mean_squared_error"].metric_lower_is_better is True
    assert metric_lower_is_better("root_mean_squared_error") is True

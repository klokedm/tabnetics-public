from __future__ import annotations

import os
from pathlib import Path

import pytest

from tabnetics.benchmarks.beyondarena_compare import (
    OFFICIAL_BEYONDARENA_METHODS,
    inspect_official_beyondarena_results,
    load_public_beyondarena_r2_results,
)
from tabnetics.datasets.beyondarena import (
    BEYONDARENA_EXPECTED_ACCEPTED_DATASETS,
    BEYONDARENA_EXPECTED_CORE_TASK_ROWS,
    BEYONDARENA_EXPECTED_TASK_METADATA_ROWS,
    build_beyondarena_smoke_task_rows,
    discover_hf_beyondarena_specs,
    discover_local_beyondarena_specs,
    load_beyondarena_core_tasks_csv,
    load_beyondarena_dataset,
    load_beyondarena_splits,
    load_beyondarena_task_metadata_csv,
    select_beyondarena_core_task_rows,
    validate_beyondarena_split_leakage,
)


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "y"}


@pytest.mark.beyondarena_external
def test_upstream_beyondarena_metadata_csv_inventory_matches_expected_counts() -> None:
    if not _truthy_env("BEYONDARENA_RUN_UPSTREAM_METADATA"):
        pytest.skip("set BEYONDARENA_RUN_UPSTREAM_METADATA=1 to query upstream metadata CSVs")

    task_rows = load_beyondarena_task_metadata_csv()
    core_rows = load_beyondarena_core_tasks_csv()
    selected_core = select_beyondarena_core_task_rows(task_rows, core_rows)
    smoke = build_beyondarena_smoke_task_rows(selected_core)

    assert len(task_rows) == BEYONDARENA_EXPECTED_TASK_METADATA_ROWS
    assert len({row.tabarena_task_name for row in task_rows}) == BEYONDARENA_EXPECTED_ACCEPTED_DATASETS
    assert len(core_rows) == BEYONDARENA_EXPECTED_CORE_TASK_ROWS
    assert len(selected_core) == BEYONDARENA_EXPECTED_CORE_TASK_ROWS
    assert len(smoke) >= 5
    assert {"iid", "grouped", "temporal"}.issubset({row.normalized_task_type for row in smoke})
    assert any(row.has_text_features for row in smoke)
    assert any(row.has_high_cardinality_features for row in smoke)
    assert any(row.is_high_dimensional for row in smoke)


@pytest.mark.beyondarena_external
def test_hf_manifest_discovery_indexes_all_configs_without_local_parquet_requirement() -> None:
    if not _truthy_env("BEYONDARENA_RUN_HF_MANIFEST"):
        pytest.skip("set BEYONDARENA_RUN_HF_MANIFEST=1 to query Hugging Face metadata manifests")
    pytest.importorskip("huggingface_hub")

    specs = discover_hf_beyondarena_specs()

    assert len(specs) == BEYONDARENA_EXPECTED_ACCEPTED_DATASETS
    assert len({spec.beyondarena_id for spec in specs}) == BEYONDARENA_EXPECTED_ACCEPTED_DATASETS
    assert {"iid", "grouped", "temporal"}.issubset({spec.task_type for spec in specs})
    assert any(spec.has_text_features or spec.has_text_cache for spec in specs)
    assert any(spec.has_high_cardinality for spec in specs)
    assert any(spec.is_high_dimensional for spec in specs)


@pytest.mark.beyondarena_external
def test_public_r2_official_results_source_is_exact_pair_ready() -> None:
    if not _truthy_env("BEYONDARENA_RUN_PUBLIC_R2"):
        pytest.skip("set BEYONDARENA_RUN_PUBLIC_R2=1 to query public BeyondArena R2 results")

    rows = load_public_beyondarena_r2_results()
    status = inspect_official_beyondarena_results("public-r2")

    assert status.available is True
    assert status.exact_paired_rows_available is True
    assert status.row_count == len(rows)
    assert set(rows["method"]) == set(OFFICIAL_BEYONDARENA_METHODS)
    assert rows["dataset_id"].nunique() == BEYONDARENA_EXPECTED_ACCEPTED_DATASETS
    assert rows["metric_value"].notna().all()
    assert rows["lower_is_better"].eq(True).all()
    assert set(rows["status"]) == {"ok"}


@pytest.mark.beyondarena_external
def test_local_materialized_artifact_root_loads_dataset_and_validates_first_split() -> None:
    artifact_root = os.environ.get("BEYONDARENA_ARTIFACT_ROOT")
    if not artifact_root:
        pytest.skip("set BEYONDARENA_ARTIFACT_ROOT=/path/to/materialized/artifacts to load local parquet")
    pytest.importorskip("pyarrow")

    root = Path(artifact_root)
    specs = discover_local_beyondarena_specs(root)
    materialized = [spec for spec in specs if spec.has_dataset and spec.artifact_dir is not None]
    if not materialized:
        pytest.skip(f"no materialized dataset.parquet artifacts found under {root}")

    spec = materialized[0]
    loaded = load_beyondarena_dataset(spec.artifact_dir)
    split = load_beyondarena_splits(spec.artifact_dir).splits[0]
    leakage = validate_beyondarena_split_leakage(loaded.frame, loaded.spec, split)

    assert loaded.spec.has_dataset is True
    assert loaded.spec.n_samples == len(loaded.frame)
    assert loaded.spec.target_column not in loaded.X.columns
    assert len(loaded.X) == len(loaded.y)
    assert leakage["ok"] is True

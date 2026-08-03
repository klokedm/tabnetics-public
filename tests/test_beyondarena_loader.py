from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from tabnetics.datasets.beyondarena import (
    BeyondArenaUnavailableError,
    build_beyondarena_smoke_subset,
    discover_local_beyondarena_specs,
    load_beyondarena_dataset,
)


FIXTURES = Path(__file__).parent / "fixtures" / "beyondarena"


def test_discover_local_beyondarena_specs_manifest_only() -> None:
    specs = discover_local_beyondarena_specs(FIXTURES)
    by_id = {spec.beyondarena_id: spec for spec in specs}

    assert set(by_id) == {
        "fixture_iid_small",
        "fixture_grouped_small",
        "fixture_temporal_small",
        "fixture_text_high_cardinality",
        "fixture_high_dimensional",
    }
    assert by_id["fixture_iid_small"].task_type == "iid"
    assert by_id["fixture_grouped_small"].task_type == "grouped"
    assert by_id["fixture_temporal_small"].task_type == "temporal"
    assert by_id["fixture_text_high_cardinality"].has_text_features is True
    assert by_id["fixture_text_high_cardinality"].has_high_cardinality is True
    assert by_id["fixture_high_dimensional"].is_high_dimensional is True
    assert all(spec.has_dataset is False for spec in specs)


def test_manifest_only_load_does_not_require_parquet() -> None:
    artifact = FIXTURES / "iid_small" / "v1"
    spec = load_beyondarena_dataset(artifact, manifest_only=True)

    assert spec.beyondarena_id == "fixture_iid_small"
    assert spec.skip_reason == "dataset.parquet not present in local artifact"

    with pytest.raises(BeyondArenaUnavailableError, match="Missing BeyondArena parquet"):
        load_beyondarena_dataset(artifact)


def test_load_beyondarena_dataset_from_local_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    src = FIXTURES / "iid_small" / "v1"
    artifact = tmp_path / "fixture_iid_small" / "v1"
    shutil.copytree(src, artifact)
    pd.DataFrame(
        {
            "x_num": [0.1, 0.2, 0.3, 0.4],
            "x_cat": ["a", "a", "b", "b"],
            "target": ["no", "yes", "no", "yes"],
        }
    ).to_parquet(artifact / "dataset.parquet")

    loaded = load_beyondarena_dataset(artifact)

    assert loaded.spec.beyondarena_id == "fixture_iid_small"
    assert loaded.spec.n_samples == 4
    assert loaded.X.columns.tolist() == ["x_num", "x_cat"]
    assert loaded.y.tolist() == ["no", "yes", "no", "yes"]


def test_build_beyondarena_smoke_subset_covers_representative_flags() -> None:
    specs = discover_local_beyondarena_specs(FIXTURES)
    subset = build_beyondarena_smoke_subset(specs)

    assert "fixture_iid_small" in subset
    assert "fixture_grouped_small" in subset
    assert "fixture_temporal_small" in subset
    assert "fixture_text_high_cardinality" in subset
    assert "fixture_high_dimensional" in subset

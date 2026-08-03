from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tabnetics.benchmarks.beyondarena_materialize import (
    BeyondArenaMaterializeConfig,
    build_beyondarena_materialization_plan,
    main,
    materialize_beyondarena_hf_artifacts,
    summarize_beyondarena_materialization,
)
from tabnetics.datasets.beyondarena import (
    CONTAINER_METADATA,
    DATASET_METADATA,
    DATASET_PARQUET,
    DTYPES_METADATA,
    SPLIT_METADATA,
    TASK_METADATA,
    TEXT_CACHE_BASENAME,
    discover_local_beyondarena_specs,
    load_beyondarena_dataset,
)


FIXTURES = Path(__file__).parent / "fixtures" / "beyondarena"


def _manifest_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "official_fixture_task-0001",
                "dataset_name": "fixture_iid_small",
                "data_foundry_uri": "iid_small/v1",
                "split_id": "r0f0",
                "problem_type": "classification",
                "task_type": "iid",
            },
            {
                "dataset_id": "official_fixture_task-0001",
                "dataset_name": "fixture_iid_small",
                "data_foundry_uri": "iid_small/v1",
                "split_id": "r0f1",
                "problem_type": "classification",
                "task_type": "iid",
            },
        ]
    )


def _available_paths(uri: str = "iid_small/v1", *, include_dataset: bool = False) -> set[str]:
    filenames = {
        CONTAINER_METADATA,
        DATASET_METADATA,
        TASK_METADATA,
        SPLIT_METADATA,
        DTYPES_METADATA,
    }
    if include_dataset:
        filenames.add(DATASET_PARQUET)
    return {f"{uri}/{name}" for name in filenames}


def test_materialization_plan_is_metadata_only_by_default(tmp_path: Path) -> None:
    plan = build_beyondarena_materialization_plan(
        _manifest_frame(),
        out_dir=tmp_path / "artifacts",
        available_paths=_available_paths(),
    )

    assert len(plan) == 5
    assert plan["data_foundry_uri"].nunique() == 1
    assert set(plan["filename"]) == {
        CONTAINER_METADATA,
        DATASET_METADATA,
        TASK_METADATA,
        SPLIT_METADATA,
        DTYPES_METADATA,
    }
    assert DATASET_PARQUET not in set(plan["filename"])
    assert set(plan["status"]) == {"planned"}
    assert set(plan["task_count"]) == {2}
    assert plan["artifact_plan_ready"].eq(True).all()
    assert plan["artifact_materialization_ready"].eq(False).all()
    assert plan["artifact_local_runner_ready"].eq(False).all()
    assert plan["artifact_required_pending_count"].eq(5).all()


def test_materialization_plan_requires_explicit_dataset_and_tolerates_missing_text_cache(tmp_path: Path) -> None:
    plan = build_beyondarena_materialization_plan(
        _manifest_frame(),
        out_dir=tmp_path / "artifacts",
        include_dataset=True,
        include_text_cache=True,
        available_paths=_available_paths(include_dataset=True),
        size_by_remote_path={"iid_small/v1/dataset.parquet": 1234},
    )

    assert DATASET_PARQUET in set(plan["filename"])
    dataset_row = plan[plan["filename"].eq(DATASET_PARQUET)].iloc[0]
    assert int(dataset_row["size_bytes"]) == 1234
    text_row = plan[plan["filename"].eq(TEXT_CACHE_BASENAME)].iloc[0]
    assert bool(text_row["required"]) is False
    assert text_row["status"] == "skipped_optional_missing_remote"
    required = plan[plan["required"].eq(True)]
    assert set(required["status"]) == {"planned"}
    assert plan["artifact_plan_ready"].eq(True).all()
    assert plan["artifact_optional_missing_count"].eq(1).all()
    assert set(plan["artifact_optional_missing_files"]) == {TEXT_CACHE_BASENAME}
    assert plan["artifact_missing_required_files"].eq("").all()


def test_materialization_plan_marks_required_missing_files_as_blockers(tmp_path: Path) -> None:
    plan = build_beyondarena_materialization_plan(
        _manifest_frame(),
        out_dir=tmp_path / "artifacts",
        include_dataset=True,
        include_text_cache=True,
        available_paths=_available_paths(include_dataset=False),
    )

    dataset_row = plan[plan["filename"].eq(DATASET_PARQUET)].iloc[0]
    text_row = plan[plan["filename"].eq(TEXT_CACHE_BASENAME)].iloc[0]
    summary = summarize_beyondarena_materialization(plan)

    assert dataset_row["status"] == "missing_remote"
    assert text_row["status"] == "skipped_optional_missing_remote"
    assert plan["artifact_plan_ready"].eq(False).all()
    assert plan["artifact_required_blocked_count"].eq(1).all()
    assert set(plan["artifact_missing_required_files"]) == {DATASET_PARQUET}
    assert set(plan["artifact_optional_missing_files"]) == {TEXT_CACHE_BASENAME}
    assert summary["artifact_blocked_count"] == 1
    assert summary["materialization_ready"] is False
    assert summary["local_runner_ready"] is False
    assert DATASET_PARQUET in summary["materialization_blocker"]


def test_beyondarena_materialize_cli_dry_run_writes_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = tmp_path / "beyondarena_task_manifest.csv"
    _manifest_frame().to_csv(manifest, index=False)
    out_dir = tmp_path / "artifacts"
    plan_csv = tmp_path / "materialization_plan.csv"

    rc = main(
        [
            "--task-manifest-csv",
            str(manifest),
            "--out-dir",
            str(out_dir),
            "--include-dataset",
            "--dry-run",
            "--plan-csv",
            str(plan_csv),
        ]
    )

    assert rc == 0
    rows = pd.read_csv(plan_csv)
    payload = json.loads(capsys.readouterr().out)
    assert len(rows) == 6
    assert DATASET_PARQUET in set(rows["filename"])
    assert "artifact_local_runner_ready" in rows.columns
    assert payload["artifact_count"] == 1
    assert payload["artifact_pending_count"] == 1
    assert payload["materialization_ready"] is False
    assert payload["local_runner_ready"] is False
    assert payload["planned_size_bytes"] == 0
    assert not out_dir.exists()


def test_beyondarena_materialize_cli_can_require_local_runner_ready(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "beyondarena_task_manifest.csv"
    _manifest_frame().to_csv(manifest, index=False)
    out_dir = tmp_path / "artifacts"
    plan_csv = tmp_path / "materialization_plan.csv"

    rc = main(
        [
            "--task-manifest-csv",
            str(manifest),
            "--out-dir",
            str(out_dir),
            "--include-dataset",
            "--dry-run",
            "--require-local-runner-ready",
            "--plan-csv",
            str(plan_csv),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    rows = pd.read_csv(plan_csv)
    assert rc == 2
    assert payload["require_local_runner_ready"] is True
    assert payload["local_runner_ready"] is False
    assert rows["artifact_local_runner_ready"].eq(False).all()
    assert rows["artifact_required_pending_count"].gt(0).all()


def test_materialize_with_fake_downloader_copies_discoverable_artifact(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    manifest = tmp_path / "beyondarena_task_manifest.csv"
    _manifest_frame().to_csv(manifest, index=False)
    remote_root = tmp_path / "remote"
    remote_artifact = remote_root / "iid_small" / "v1"
    shutil.copytree(FIXTURES / "iid_small" / "v1", remote_artifact)
    pd.DataFrame(
        {
            "x_num": [0.1, 0.2, 0.3, 0.4],
            "x_cat": ["a", "a", "b", "b"],
            "target": ["no", "yes", "no", "yes"],
        }
    ).to_parquet(remote_artifact / DATASET_PARQUET)

    def fake_download(remote_path: str) -> Path:
        path = remote_root / remote_path
        if not path.exists():
            raise FileNotFoundError(remote_path)
        return path

    out_dir = tmp_path / "artifacts"
    rows = materialize_beyondarena_hf_artifacts(
        BeyondArenaMaterializeConfig(
            task_manifest_csv=manifest,
            out_dir=out_dir,
            include_dataset=True,
        ),
        download_fn=fake_download,
        available_paths=_available_paths(include_dataset=True),
    )

    assert set(rows["status"]) == {"downloaded"}
    assert rows["artifact_materialization_ready"].eq(True).all()
    assert rows["artifact_local_runner_ready"].eq(True).all()
    assert summarize_beyondarena_materialization(rows)["local_runner_ready"] is True
    artifact = out_dir / "iid_small" / "v1"
    specs = discover_local_beyondarena_specs(out_dir)
    assert len(specs) == 1
    assert specs[0].has_dataset is True
    loaded = load_beyondarena_dataset(artifact)
    assert loaded.spec.beyondarena_id == "fixture_iid_small"
    assert loaded.spec.n_samples == 4

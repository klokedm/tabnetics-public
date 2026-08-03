from __future__ import annotations

import json
import math
import os
import shutil
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabnetics.benchmarks import beyondarena_local as local_module
from tabnetics.benchmarks.beyondarena_local import (
    BeyondArenaLocalRunConfig,
    main,
    run_local_beyondarena_artifact,
    run_local_beyondarena_artifacts,
)
from tabnetics.benchmarks.beyondarena_plan import align_beyondarena_results_to_manifest
from tabnetics.benchmarks.result_journal import AtomicResultJournal, ResultJournalContextError


FIXTURES = Path(__file__).parent / "fixtures" / "beyondarena"
TASK_METADATA = "task_metadata.predictive-ml-task-mold-v1.json"


def _copy_iid_artifact_with_parquet(tmp_path: Path) -> Path:
    pytest.importorskip("pyarrow")
    src = FIXTURES / "iid_small" / "v1"
    artifact = tmp_path / "iid_small" / "v1"
    shutil.copytree(src, artifact)
    pd.DataFrame(
        {
            "x_num": [0.1, 0.2, 0.3, 0.4, 0.8, 0.9],
            "x_cat": ["a", "a", "b", "b", "a", "b"],
            "target": ["no", "yes", "no", "yes", "no", "yes"],
        }
    ).to_parquet(artifact / "dataset.parquet")
    return artifact


def test_local_runner_materializes_smoke_result_rows(tmp_path: Path) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="sklearn-smoke",
        method="tabnetics-current-smoke",
        model_profile="sklearn_smoke",
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert len(rows) == 1
    row = rows.iloc[0].to_dict()
    assert row["dataset_id"] == "fixture_iid_small"
    assert row["split_id"] == "0:0"
    assert row["method"] == "tabnetics-current-smoke"
    assert row["metric"] == "roc_auc"
    assert row["status"] == "ok"
    assert row["execution_status"] == "ok"
    assert row["lower_is_better"] is False
    assert row["seed"] == 42
    assert row["device"] == "cpu"
    assert row["execution_host"] == "local-ci"
    assert row["execution_lane"] == "cpu"
    assert row["allow_gpu_execution"] is False
    assert row["leakage_ok"] is True
    assert 0.0 <= float(row["metric_value"]) <= 1.0
    assert math.isclose(float(row["metric_error"]), 1.0 - float(row["metric_value"]))


def test_tabnetics_current_enables_svc_probabilities_for_log_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_tabnetics_with_config(config, *args, **kwargs):
        captured["flat_flag"] = bool(config.model_cv_enable_svc_probability)
        captured["nested_flag"] = bool(config.classification.enable_svc_probability)
        return (
            {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "log_loss": 0.25,
                "roc_auc": 1.0,
            },
            {"model_name": "svm_rbf"},
        )

    monkeypatch.setattr(local_module, "_run_tabnetics_with_config", fake_run_tabnetics_with_config)

    values, meta = local_module._run_tabnetics_current(
        np.zeros((6, 2), dtype=float),
        np.array(["a", "b", "a", "b", "a", "b"]),
        np.zeros((2, 2), dtype=float),
        np.array(["a", "b"]),
        dataset_name="probability_required",
        seed=17,
        fast=False,
    )

    assert values["log_loss"] == 0.25
    assert meta["model_name"] == "svm_rbf"
    assert captured == {"flat_flag": True, "nested_flag": True}


def test_local_runner_skips_missing_materialized_parquet() -> None:
    artifact = FIXTURES / "iid_small" / "v1"
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="sklearn-smoke",
        method="tabnetics-current-smoke",
        model_profile="sklearn_smoke",
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["status"] == "skipped"
    assert row["execution_status"] == "skipped_missing_artifact"
    assert "Missing BeyondArena parquet" in row["skip_reason"]


def test_local_runner_sklearn_smoke_supports_regression_schema_rows(tmp_path: Path) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    metadata_path = artifact / TASK_METADATA
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["problem_type"] = "regression"
    metadata["objective_metric_name"] = "root_mean_squared_error"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "x_num": [0.1, 0.2, 0.3, 0.4, 0.8, 0.9],
            "x_cat": ["a", "a", "b", "b", "a", "b"],
            "target": [1.0, 1.2, 0.8, 1.4, 1.6, 1.7],
        }
    ).to_parquet(artifact / "dataset.parquet")
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="sklearn-smoke",
        method="tabnetics-current-smoke",
        model_profile="sklearn_smoke",
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["metric"] == "rmse"
    assert row["status"] == "ok"
    assert row["execution_status"] == "ok"
    assert bool(row["lower_is_better"]) is True
    assert row["model_name"] == "Ridge"
    assert float(row["metric_value"]) >= 0.0
    assert int(row["n_targets"]) == 1


def test_local_runner_tabnetics_current_keeps_regression_fail_closed(tmp_path: Path) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    metadata_path = artifact / TASK_METADATA
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["problem_type"] = "regression"
    metadata["objective_metric_name"] = "root_mean_squared_error"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "x_num": [0.1, 0.2, 0.3, 0.4, 0.8, 0.9],
            "x_cat": ["a", "a", "b", "b", "a", "b"],
            "target": [1.0, 1.2, 0.8, 1.4, 1.6, 1.7],
        }
    ).to_parquet(artifact / "dataset.parquet")
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabnetics-current",
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["metric"] == "rmse"
    assert row["status"] == "skipped"
    assert row["execution_status"] == "skipped_unsupported_regression"
    assert "classification-only" in row["skip_reason"]
    assert "sklearn-smoke only for schema checks" in row["skip_reason"]


def test_tabnetics_pipeline_backends_encode_string_class_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, np.ndarray]] = []

    def fake_run(config, X_train, y_train, X_test, y_test, *, dataset_name, seed):
        captured.append(
            {
                "y_train": np.asarray(y_train),
                "y_test": np.asarray(y_test),
            }
        )
        return (
            {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "log_loss": 0.0,
                "roc_auc": 1.0,
            },
            {"model_name": "FakePipeline"},
        )

    monkeypatch.setattr(local_module, "_run_tabnetics_with_config", fake_run)
    X_train = np.zeros((4, 2), dtype=float)
    X_test = np.ones((2, 2), dtype=float)
    y_train = np.array(["no", "yes", "no", "yes"], dtype=object)
    y_test = np.array(["yes", "no"], dtype=object)

    _, current_meta = local_module._run_tabnetics_current(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_name="string_label_smoke",
        seed=42,
        fast=True,
    )
    _, tabpfn_meta = local_module._run_tabpfn_candidate(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_name="string_label_smoke",
        seed=42,
    )

    assert len(captured) == 2
    for call in captured:
        assert call["y_train"].dtype.kind in {"i", "u"}
        assert call["y_test"].dtype.kind in {"i", "u"}
        assert call["y_train"].tolist() == [0, 1, 0, 1]
        assert call["y_test"].tolist() == [1, 0]
    assert current_meta["class_labels"] == "no|yes"
    assert current_meta["n_classes"] == 2
    assert tabpfn_meta["class_labels"] == "no|yes"
    assert tabpfn_meta["n_classes"] == 2


def test_beyondarena_local_cli_writes_result_csv(tmp_path: Path) -> None:
    _copy_iid_artifact_with_parquet(tmp_path)
    out_csv = tmp_path / "local_results.csv"

    rc = main(
        [
            "--artifact-root",
            str(tmp_path),
            "--out-csv",
            str(out_csv),
            "--backend",
            "sklearn-smoke",
            "--execution-host",
            "host1.example.com",
            "--max-artifacts",
            "1",
        ]
    )

    assert rc == 0
    rows = pd.read_csv(out_csv)
    assert len(rows) == 1
    assert rows.loc[0, "method"] == "tabnetics-current-smoke"
    assert rows.loc[0, "status"] == "ok"
    assert rows.loc[0, "device"] == "cpu"
    assert rows.loc[0, "execution_host"] == "host1.example.com"
    assert rows.loc[0, "execution_lane"] == "cpu"
    assert int(rows.loc[0, "seed"]) == 42


def test_beyondarena_local_cli_refuses_claimed_output_csv(tmp_path: Path) -> None:
    _copy_iid_artifact_with_parquet(tmp_path)
    out_csv = tmp_path / "local_results.csv"
    out_csv.with_name(f"{out_csv.name}.lock").write_text("pid=other\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="output is already claimed"):
        main(
            [
                "--artifact-root",
                str(tmp_path),
                "--out-csv",
                str(out_csv),
                "--backend",
                "sklearn-smoke",
                "--max-artifacts",
                "1",
            ]
        )

    assert not out_csv.exists()


def test_beyondarena_local_cli_reclaims_stale_dead_pid_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_iid_artifact_with_parquet(tmp_path)
    out_csv = tmp_path / "local_results.csv"
    lock_path = out_csv.with_name(f"{out_csv.name}.lock")
    stale_metadata = {
        **local_module._new_lock_metadata(),
        "pid": "424242",
        "pid_start_time": "1",
    }
    lock_path.write_text(
        "".join(f"{key}={value}\n" for key, value in stale_metadata.items()),
        encoding="utf-8",
    )
    monkeypatch.setattr(local_module, "_pid_is_running", lambda pid: False)

    rc = main(
        [
            "--artifact-root",
            str(tmp_path),
            "--out-csv",
            str(out_csv),
            "--backend",
            "sklearn-smoke",
            "--max-artifacts",
            "1",
        ]
    )

    assert rc == 0
    assert not lock_path.exists()
    rows = pd.read_csv(out_csv)
    assert len(rows) == 1
    assert rows.loc[0, "status"] == "ok"


def test_beyondarena_local_cli_does_not_reclaim_foreign_host_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_iid_artifact_with_parquet(tmp_path)
    out_csv = tmp_path / "local_results.csv"
    lock_path = out_csv.with_name(f"{out_csv.name}.lock")
    lock_metadata = {
        **local_module._new_lock_metadata(),
        "hostname": "another-host.example",
        "pid": "424242",
    }
    lock_path.write_text(
        "".join(f"{key}={value}\n" for key, value in lock_metadata.items()),
        encoding="utf-8",
    )
    monkeypatch.setattr(local_module, "_pid_is_running", lambda pid: False)

    with pytest.raises(RuntimeError, match="output is already claimed"):
        main(
            [
                "--artifact-root",
                str(tmp_path),
                "--out-csv",
                str(out_csv),
                "--backend",
                "sklearn-smoke",
            ]
        )

    assert lock_path.exists()
    assert not out_csv.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", None),
        ("claim_id", None),
        ("hostname", None),
        ("boot_id", None),
        ("pid", None),
        ("pid_start_time", None),
        ("version", "3"),
        ("claim_id", "not-a-uuid"),
        ("hostname", ""),
        ("boot_id", "not-a-uuid"),
        ("pid", "0"),
        ("pid", "not-a-pid"),
        ("pid_start_time", ""),
        ("pid_start_time", "0"),
        ("pid_start_time", "not-a-start-time"),
    ],
    ids=lambda value: "missing" if value is None else str(value) or "empty",
)
def test_beyondarena_lock_recovery_rejects_incomplete_or_malformed_v2_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str | None,
) -> None:
    lock_path = tmp_path / "local_results.csv.lock"
    metadata = {
        **local_module._new_lock_metadata(),
        "pid": "424242",
        "pid_start_time": "1",
    }
    if value is None:
        metadata.pop(field)
    else:
        metadata[field] = value
    lock_path.write_text(
        "".join(f"{key}={item}\n" for key, item in metadata.items()),
        encoding="utf-8",
    )
    monkeypatch.setattr(local_module, "_pid_is_running", lambda pid: False)

    assert local_module._lock_is_safely_reclaimable(lock_path) is False
    with pytest.raises(FileExistsError):
        local_module._open_out_csv_lock(lock_path)
    assert lock_path.exists()


def test_beyondarena_lock_recovery_rejects_duplicate_v2_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "local_results.csv.lock"
    metadata = {
        **local_module._new_lock_metadata(),
        "pid": "424242",
        "pid_start_time": "1",
    }
    payload = "".join(f"{key}={value}\n" for key, value in metadata.items())
    lock_path.write_text(f"{payload}pid=424242\n", encoding="utf-8")
    monkeypatch.setattr(local_module, "_pid_is_running", lambda pid: False)

    assert local_module._lock_metadata(lock_path) == {}
    assert local_module._lock_is_safely_reclaimable(lock_path) is False
    with pytest.raises(FileExistsError):
        local_module._open_out_csv_lock(lock_path)
    assert lock_path.exists()


def test_output_claim_cleanup_does_not_unlink_replacement_lock(tmp_path: Path) -> None:
    out_csv = tmp_path / "local_results.csv"
    lock_path = out_csv.with_name(f"{out_csv.name}.lock")

    with local_module._claim_out_csv(out_csv):
        lock_path.unlink()
        replacement = {
            **local_module._new_lock_metadata(),
            "claim_id": "replacement-owner",
        }
        lock_path.write_text(
            "".join(f"{key}={value}\n" for key, value in replacement.items()),
            encoding="utf-8",
        )

    assert lock_path.exists()
    assert local_module._lock_metadata(lock_path)["claim_id"] == "replacement-owner"


def test_checkpoint_identity_is_content_bound_even_when_size_is_unchanged(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"first-state")
    original_stat = checkpoint.stat()
    first = local_module._checkpoint_identity(str(checkpoint))

    checkpoint.write_bytes(b"other-state")
    os.utime(
        checkpoint,
        ns=(int(original_stat.st_atime_ns), int(original_stat.st_mtime_ns)),
    )
    second = local_module._checkpoint_identity(str(checkpoint))

    assert first["size"] == second["size"]
    assert first["sha256"] != second["sha256"]


def test_result_journal_context_binds_runtime_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = local_module._normalize_config(
        local_module.BeyondArenaLocalRunConfig(artifact_root=tmp_path)
    )
    monkeypatch.setattr(
        local_module,
        "_runtime_dependency_identity",
        lambda backend: {
            "backend": backend,
            "python": "3.x",
            "packages": {},
            "tabnetics_source": {"sha256": "abc"},
        },
    )

    context = local_module._result_journal_context(config)

    assert context["runner"] == "beyondarena-local-v4"
    assert context["runtime"]["tabnetics_source"]["sha256"] == "abc"


def test_selected_artifact_inventory_uses_authoritative_checksum_without_hashing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    container_path = artifact / "container_metadata.json"
    container = json.loads(container_path.read_text(encoding="utf-8"))
    container["checksum"] = "a" * 64
    container_path.write_text(json.dumps(container, indent=2) + "\n", encoding="utf-8")
    real_sha256_file = local_module._sha256_file
    hashed_paths: list[Path] = []

    def track_hash(path: Path) -> str:
        hashed_paths.append(Path(path))
        return real_sha256_file(path)

    monkeypatch.setattr(local_module, "_sha256_file", track_hash)
    config = local_module._normalize_config(
        BeyondArenaLocalRunConfig(artifact_root=tmp_path, backend="sklearn-smoke")
    )

    inventory = local_module._selected_artifact_inventory(config)

    assert len(inventory) == 1
    assert inventory[0]["relative_path"] == "iid_small/v1"
    assert inventory[0]["artifact_revision"] == "a" * 64
    assert inventory[0]["dataset"]["identity_kind"] == "authoritative_sha256"
    assert inventory[0]["dataset"]["sha256"] == "a" * 64
    assert artifact / "dataset.parquet" not in hashed_paths
    assert artifact / "experiment_metadata.predictive-ml-splits-mold-v1.json" in hashed_paths


def test_selected_artifact_inventory_records_missing_metadata_and_fallback_dataset_hash(
    tmp_path: Path,
) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    (artifact / "container_metadata.json").unlink()
    config = local_module._normalize_config(
        BeyondArenaLocalRunConfig(artifact_root=tmp_path, backend="sklearn-smoke")
    )

    inventory = local_module._selected_artifact_inventory(config)

    assert len(inventory) == 1
    metadata = {item["path"]: item for item in inventory[0]["metadata"]}
    assert metadata["iid_small/v1/container_metadata.json"]["state"] == "missing"
    assert inventory[0]["dataset"]["identity_kind"] == "computed_sha256"
    assert inventory[0]["dataset"]["sha256"] == local_module._sha256_file(
        artifact / "dataset.parquet"
    )


@pytest.mark.parametrize("mutation", ("dataset", "splits"))
def test_beyondarena_resume_rejects_same_path_artifact_content_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    out_csv = tmp_path / "local_results.csv"
    args = [
        "--artifact-root",
        str(tmp_path),
        "--out-csv",
        str(out_csv),
        "--backend",
        "sklearn-smoke",
    ]
    stable_runtime = {
        "backend": "sklearn-smoke",
        "python": "3.x",
        "packages": {},
        "tabnetics_source": {"sha256": "stable"},
    }
    monkeypatch.setattr(
        local_module,
        "_runtime_dependency_identity",
        lambda backend: stable_runtime,
    )
    assert main(args) == 0
    original_csv = out_csv.read_bytes()

    if mutation == "dataset":
        dataset_path = artifact / "dataset.parquet"
        frame = pd.read_parquet(dataset_path)
        frame.loc[0, "x_num"] = 99.0
        frame.to_parquet(dataset_path)
    else:
        split_path = artifact / "experiment_metadata.predictive-ml-splits-mold-v1.json"
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        split_payload["splits"]["0"]["0"] = [[0, 1, 2], [3, 4, 5]]
        split_path.write_text(json.dumps(split_payload, indent=2) + "\n", encoding="utf-8")

    backend_calls = 0

    def unexpected_backend_call(*args, **kwargs):
        nonlocal backend_calls
        backend_calls += 1
        raise AssertionError("resume reached model execution after an input identity change")

    monkeypatch.setattr(local_module, "_run_sklearn_smoke", unexpected_backend_call)
    with pytest.raises(ResultJournalContextError, match="context does not match"):
        main(args)

    assert backend_calls == 0
    assert out_csv.read_bytes() == original_csv


@pytest.mark.parametrize(
    ("initial", "changed"),
    [
        (
            {"available": False, "version": ""},
            {"available": True, "version": "2.0.0"},
        ),
        (
            {"available": True, "version": "2.0.0"},
            {"available": True, "version": "2.1.0"},
        ),
    ],
    ids=("missing-to-installed", "version-change"),
)
def test_tabpfn_dependency_change_rejects_existing_journal_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial: dict[str, object],
    changed: dict[str, object],
) -> None:
    tabpfn_identity = dict(initial)

    def distribution_identity(package: str) -> dict[str, object]:
        if package == "tabpfn":
            return dict(tabpfn_identity)
        return {"available": True, "version": "stable"}

    monkeypatch.setattr(local_module, "_distribution_identity", distribution_identity)
    monkeypatch.setattr(
        local_module,
        "_package_source_identity",
        lambda: {"root": "test", "python_file_count": 1, "sha256": "stable"},
    )
    monkeypatch.setattr(
        local_module,
        "_tabpfn_default_model_identity",
        lambda package: {"resolution": "package_default_unresolved"},
    )
    config = local_module._normalize_config(
        BeyondArenaLocalRunConfig(artifact_root=tmp_path, backend="tabpfn-candidate")
    )
    first_context = local_module._result_journal_context(config)
    journal_root = tmp_path / "results.journal"
    local_module.AtomicResultJournal(
        journal_root,
        key_fields=local_module._BEYONDARENA_RESULT_KEY_FIELDS,
        context=first_context,
    )

    tabpfn_identity.clear()
    tabpfn_identity.update(changed)
    changed_context = local_module._result_journal_context(config)

    assert first_context["runtime"]["packages"]["tabpfn"] == initial
    assert changed_context["runtime"]["packages"]["tabpfn"] == changed
    with pytest.raises(ResultJournalContextError, match="context does not match"):
        local_module.AtomicResultJournal(
            journal_root,
            key_fields=local_module._BEYONDARENA_RESULT_KEY_FIELDS,
            context=changed_context,
        )


@pytest.mark.parametrize(
    ("initial", "changed"),
    [
        (
            {"available": False, "importable": False, "version": ""},
            {"available": True, "importable": True, "version": "18.0.0"},
        ),
        (
            {"available": True, "importable": False, "version": "18.0.0"},
            {"available": True, "importable": True, "version": "18.0.0"},
        ),
        (
            {"available": True, "importable": True, "version": "18.0.0"},
            {"available": True, "importable": True, "version": "19.0.0"},
        ),
    ],
    ids=("missing-to-installed", "broken-to-usable", "version-change"),
)
def test_parquet_engine_change_rejects_existing_journal_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial: dict[str, object],
    changed: dict[str, object],
) -> None:
    pyarrow_identity = dict(initial)

    def distribution_identity(package: str) -> dict[str, object]:
        if package == "pyarrow":
            return dict(pyarrow_identity)
        if package == "fastparquet":
            return {"available": False, "importable": False, "version": ""}
        return {"available": True, "importable": True, "version": "stable"}

    def engine_usability(engine: str) -> dict[str, object]:
        if engine == "fastparquet":
            return {
                "usable": False,
                "implementation": "",
                "resolution_error": "ImportError",
            }
        usable = bool(
            pyarrow_identity.get("available") is True
            and pyarrow_identity.get("importable") is True
        )
        if not usable:
            return {
                "usable": False,
                "implementation": "",
                "resolution_error": "ImportError",
            }
        return {
            "usable": True,
            "implementation": "pandas.io.parquet.PyArrowImpl",
            "selected_engine": "pyarrow",
        }

    monkeypatch.setattr(local_module, "_distribution_identity", distribution_identity)
    monkeypatch.setattr(local_module, "_parquet_engine_usability", engine_usability)
    monkeypatch.setattr(
        local_module,
        "_package_source_identity",
        lambda: {"root": "test", "python_file_count": 1, "sha256": "stable"},
    )
    config = local_module._normalize_config(
        BeyondArenaLocalRunConfig(artifact_root=tmp_path, backend="sklearn-smoke")
    )
    first_context = local_module._result_journal_context(config)
    journal_root = tmp_path / "results.journal"
    local_module.AtomicResultJournal(
        journal_root,
        key_fields=local_module._BEYONDARENA_RESULT_KEY_FIELDS,
        context=first_context,
    )

    pyarrow_identity.clear()
    pyarrow_identity.update(changed)
    changed_context = local_module._result_journal_context(config)

    assert "fastparquet" in first_context["runtime"]["packages"]
    assert first_context["runtime"]["packages"]["pyarrow"] == initial
    assert changed_context["runtime"]["packages"]["pyarrow"] == changed
    expected_initial_engine = "pyarrow" if initial.get("importable") is True else ""
    expected_changed_engine = "pyarrow" if changed.get("importable") is True else ""
    assert first_context["runtime"]["parquet"]["selected_engine"] == expected_initial_engine
    assert changed_context["runtime"]["parquet"]["selected_engine"] == expected_changed_engine
    with pytest.raises(ResultJournalContextError, match="context does not match"):
        local_module.AtomicResultJournal(
            journal_root,
            key_fields=local_module._BEYONDARENA_RESULT_KEY_FIELDS,
            context=changed_context,
        )


def test_parquet_runtime_identity_records_fastparquet_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packages = {
        "pyarrow": {"available": False, "importable": False, "version": ""},
        "fastparquet": {"available": True, "importable": True, "version": "2026.1.0"},
    }

    def engine_usability(engine: str) -> dict[str, object]:
        if engine == "pyarrow":
            return {"usable": False, "implementation": "", "resolution_error": "ImportError"}
        return {
            "usable": True,
            "implementation": "pandas.io.parquet.FastParquetImpl",
            "selected_engine": "fastparquet",
        }

    monkeypatch.setattr(local_module, "_parquet_engine_usability", engine_usability)

    identity = local_module._parquet_runtime_identity(packages)

    assert identity["pandas_policy"] == "auto"
    assert identity["selected_engine"] == "fastparquet"
    assert identity["engines"]["pyarrow"]["runtime"]["usable"] is False
    assert identity["engines"]["fastparquet"]["runtime"]["usable"] is True
    assert identity["engines"]["fastparquet"]["distribution"]["version"] == "2026.1.0"


def test_tabpfn_model_identity_binds_explicit_model_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "tabpfn-default.ckpt"
    model.write_bytes(b"model-a")
    monkeypatch.setenv("TABPFN_MODEL_PATH", str(model))
    monkeypatch.setenv("TABPFN_MODEL_CACHE_DIR", str(tmp_path / "missing-cache"))
    monkeypatch.setenv("TABPFN_CACHE_DIR", str(tmp_path / "missing-legacy-cache"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "missing-hf-cache"))

    identity = local_module._tabpfn_default_model_identity(
        {"available": True, "version": "2.0.0"}
    )

    assert identity["resolution"] == "cached_or_explicit_model"
    explicit = [item for item in identity["model_files"] if item["source"] == "explicit_model"]
    assert len(explicit) == 1
    assert explicit[0]["sha256"] == local_module._sha256_file(model)


def test_result_journal_context_ignores_resource_only_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_module,
        "_runtime_dependency_identity",
        lambda backend: {"backend": backend, "identity": "stable"},
    )
    first = local_module._normalize_config(
        BeyondArenaLocalRunConfig(
            artifact_root=tmp_path,
            backend="sklearn-smoke",
            execution_host="public_cpu_host_1-a",
            max_workers=8,
            max_in_flight_artifacts=4,
        )
    )
    second = local_module._normalize_config(
        BeyondArenaLocalRunConfig(
            artifact_root=tmp_path,
            backend="sklearn-smoke",
            execution_host="public_cpu_host_1-b",
            max_workers=2,
            max_in_flight_artifacts=1,
        )
    )

    assert local_module._result_journal_context(first) == local_module._result_journal_context(second)


def test_tabiclv2_resume_rejects_checkpoint_path_content_revision_and_semantic_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tabnetics.classification import tabiclv2

    monkeypatch.setattr(
        local_module,
        "_runtime_dependency_identity",
        lambda backend: {"backend": backend, "identity": "stable"},
    )
    first_checkpoint = tmp_path / "first.ckpt"
    second_checkpoint = tmp_path / "second.ckpt"
    first_checkpoint.write_bytes(b"checkpoint-a")
    second_checkpoint.write_bytes(b"checkpoint-a")
    config = local_module._normalize_config(
        BeyondArenaLocalRunConfig(
            artifact_root=tmp_path,
            backend="tabiclv2-candidate",
            tabiclv2_checkpoint=str(first_checkpoint),
        )
    )
    context = local_module._result_journal_context(config)
    journal_root = tmp_path / "tabiclv2.journal"
    AtomicResultJournal(
        journal_root,
        key_fields=local_module._BEYONDARENA_RESULT_KEY_FIELDS,
        context=context,
    )

    path_changed = local_module._result_journal_context(
        replace(config, tabiclv2_checkpoint=str(second_checkpoint))
    )
    with pytest.raises(ResultJournalContextError):
        AtomicResultJournal(
            journal_root,
            key_fields=local_module._BEYONDARENA_RESULT_KEY_FIELDS,
            context=path_changed,
        )

    first_checkpoint.write_bytes(b"checkpoint-b")
    sha_changed = local_module._result_journal_context(config)
    with pytest.raises(ResultJournalContextError):
        AtomicResultJournal(
            journal_root,
            key_fields=local_module._BEYONDARENA_RESULT_KEY_FIELDS,
            context=sha_changed,
        )

    first_checkpoint.write_bytes(b"checkpoint-a")
    monkeypatch.setattr(tabiclv2, "TABICLV2_REVISION", "f" * 40)
    revision_changed = local_module._result_journal_context(config)
    with pytest.raises(ResultJournalContextError):
        AtomicResultJournal(
            journal_root,
            key_fields=local_module._BEYONDARENA_RESULT_KEY_FIELDS,
            context=revision_changed,
        )

    for changed_config in (
        replace(config, tabiclv2_device="cuda:1"),
        replace(config, tabiclv2_max_features=1_999),
    ):
        semantic_change = local_module._result_journal_context(changed_config)
        with pytest.raises(ResultJournalContextError):
            AtomicResultJournal(
                journal_root,
                key_fields=local_module._BEYONDARENA_RESULT_KEY_FIELDS,
                context=semantic_change,
            )


def test_beyondarena_local_cli_resumes_committed_rows_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    split_path = artifact / "experiment_metadata.predictive-ml-splits-mold-v1.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    split_payload["splits"]["0"]["1"] = [[0, 1, 2, 3], [4, 5]]
    split_path.write_text(json.dumps(split_payload, indent=2) + "\n", encoding="utf-8")
    out_csv = tmp_path / "local_results.csv"
    args = [
        "--artifact-root",
        str(tmp_path),
        "--out-csv",
        str(out_csv),
        "--backend",
        "sklearn-smoke",
    ]
    real_backend = local_module._run_sklearn_smoke
    initial_calls = 0

    def interrupt_after_first(*backend_args, **backend_kwargs):
        nonlocal initial_calls
        initial_calls += 1
        if initial_calls == 2:
            raise KeyboardInterrupt("simulated worker interruption")
        return real_backend(*backend_args, **backend_kwargs)

    monkeypatch.setattr(local_module, "_run_sklearn_smoke", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt, match="simulated worker interruption"):
        main(args)

    journal_rows = tuple((tmp_path / "local_results.csv.journal" / "rows").glob("*.json"))
    assert initial_calls == 2
    assert len(journal_rows) == 1
    assert not out_csv.exists()

    resume_calls = 0

    def track_resume(*backend_args, **backend_kwargs):
        nonlocal resume_calls
        resume_calls += 1
        return real_backend(*backend_args, **backend_kwargs)

    monkeypatch.setattr(local_module, "_run_sklearn_smoke", track_resume)
    assert main(args) == 0

    rows = pd.read_csv(out_csv)
    assert resume_calls == 1
    assert len(rows) == 2
    assert rows["split_id"].astype(str).tolist() == ["0:0", "0:1"]
    assert not rows.duplicated(["dataset_id", "split_id", "method", "metric", "seed"]).any()

    assert main(args) == 0
    assert resume_calls == 1


def test_local_runner_uses_official_manifest_ids_for_comparison_join(tmp_path: Path) -> None:
    _copy_iid_artifact_with_parquet(tmp_path)
    manifest = pd.DataFrame(
        [
            {
                "dataset_id": "official_fixture_task-0001",
                "dataset_name": "fixture_iid_small",
                "data_foundry_uri": "iid_small/v1",
                "split": 0,
                "official_split_id": "0",
                "split_id": "r0f0",
                "repeat": 0,
                "fold": 0,
                "task_type": "iid",
                "problem_type": "classification",
                "metric": "roc_auc",
                "size_tier": "tiny",
                "dimensionality": "low",
                "has_text": False,
                "high_cardinality": False,
            }
        ]
    )
    manifest_csv = tmp_path / "beyondarena_task_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    config = BeyondArenaLocalRunConfig(
        artifact_root=tmp_path,
        task_manifest_csv=manifest_csv,
        backend="sklearn-smoke",
        method="tabnetics-current-smoke",
        model_profile="sklearn_smoke",
    )

    rows = run_local_beyondarena_artifacts(config=config)
    aligned = align_beyondarena_results_to_manifest(rows, manifest)

    assert len(rows) == 1
    assert rows.loc[0, "dataset_id"] == "official_fixture_task-0001"
    assert rows.loc[0, "local_dataset_id"] == "fixture_iid_small"
    assert rows.loc[0, "split_id"] == "0"
    assert rows.loc[0, "local_split_id"] == "0:0"
    assert rows.loc[0, "status"] == "ok"
    assert len(aligned) == 1
    assert aligned.loc[0, "dataset_id"] == "official_fixture_task-0001"
    assert aligned.loc[0, "split_id"] == "0"


def test_local_runner_emits_skipped_manifest_rows_when_artifacts_are_missing(tmp_path: Path) -> None:
    artifact_root = tmp_path / "empty_artifacts"
    artifact_root.mkdir()
    manifest = pd.DataFrame(
        [
            {
                "dataset_id": "official_missing_task-0001",
                "dataset_name": "missing_dataset",
                "data_foundry_uri": "missing_dataset/v1",
                "split": 0,
                "official_split_id": "0",
                "split_id": "r0f0",
                "repeat": 0,
                "fold": 0,
                "task_type": "iid",
                "problem_type": "classification",
                "metric": "roc_auc",
                "size_tier": "tiny",
                "dimensionality": "low",
                "has_text": False,
                "high_cardinality": False,
            }
        ]
    )
    manifest_csv = tmp_path / "beyondarena_task_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact_root,
        task_manifest_csv=manifest_csv,
        backend="tabnetics-current",
    )

    rows = run_local_beyondarena_artifacts(config=config)

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["dataset_id"] == "official_missing_task-0001"
    assert row["split_id"] == "0"
    assert row["local_split_id"] == "r0f0"
    assert row["method"] == "tabnetics-current"
    assert row["status"] == "skipped"
    assert row["execution_status"] == "skipped_missing_artifact"
    assert "no materialized local DataFoundry artifact" in row["skip_reason"]
    assert row["device"] == "cpu"
    assert row["execution_host"] == "public_cpu_host_1/public_cpu_host_2"
    assert row["execution_lane"] == "cpu"


def test_local_runner_accounts_for_unmatched_manifest_rows(tmp_path: Path) -> None:
    _copy_iid_artifact_with_parquet(tmp_path)
    manifest = pd.DataFrame(
        [
            {
                "dataset_id": "official_fixture_task-0001",
                "dataset_name": "fixture_iid_small",
                "data_foundry_uri": "iid_small/v1",
                "split": 0,
                "official_split_id": "0",
                "split_id": "r0f0",
                "repeat": 0,
                "fold": 0,
                "task_type": "iid",
                "problem_type": "classification",
                "metric": "roc_auc",
                "size_tier": "tiny",
                "dimensionality": "low",
                "has_text": False,
                "high_cardinality": False,
            },
            {
                "dataset_id": "official_missing_task-0002",
                "dataset_name": "missing_dataset",
                "data_foundry_uri": "missing_dataset/v1",
                "split": 0,
                "official_split_id": "0",
                "split_id": "r0f0",
                "repeat": 0,
                "fold": 0,
                "task_type": "iid",
                "problem_type": "classification",
                "metric": "roc_auc",
                "size_tier": "tiny",
                "dimensionality": "low",
                "has_text": False,
                "high_cardinality": False,
            },
        ]
    )
    manifest_csv = tmp_path / "beyondarena_task_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    config = BeyondArenaLocalRunConfig(
        artifact_root=tmp_path,
        task_manifest_csv=manifest_csv,
        backend="sklearn-smoke",
    )

    rows = run_local_beyondarena_artifacts(config=config)

    assert len(rows) == 2
    by_dataset = {row["dataset_id"]: row for _, row in rows.iterrows()}
    assert by_dataset["official_fixture_task-0001"]["status"] == "ok"
    assert by_dataset["official_missing_task-0002"]["status"] == "skipped"
    assert by_dataset["official_missing_task-0002"]["execution_status"] == "skipped_missing_artifact"


def test_local_runner_uses_configured_max_workers_for_artifact_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _copy_iid_artifact_with_parquet(tmp_path)
    second = tmp_path / "iid_other" / "v1"
    shutil.copytree(first, second)
    seen_workers: list[int] = []
    seen_artifacts: list[str] = []

    real_executor = local_module.ThreadPoolExecutor

    def tracking_executor(*, max_workers: int):
        seen_workers.append(int(max_workers))
        return real_executor(max_workers=max_workers)

    def fake_run_artifact(
        artifact_dir,
        *,
        config,
        manifest_rows,
        result_journal,
        collect_rows,
    ):
        seen_artifacts.append(str(Path(artifact_dir).relative_to(tmp_path)))
        return pd.DataFrame.from_records(
            [
                {
                    "dataset_id": str(Path(artifact_dir).parent.name),
                    "status": "ok",
                }
            ]
        )

    monkeypatch.setattr(local_module, "ThreadPoolExecutor", tracking_executor)
    monkeypatch.setattr(local_module, "run_local_beyondarena_artifact", fake_run_artifact)

    rows = run_local_beyondarena_artifacts(
        config=BeyondArenaLocalRunConfig(
            artifact_root=tmp_path,
            backend="sklearn-smoke",
            max_workers=2,
        )
    )

    assert seen_workers == [2]
    assert sorted(seen_artifacts) == ["iid_other/v1", "iid_small/v1"]
    assert len(rows) == 2
    assert set(rows["status"]) == {"ok"}


def test_bounded_executor_map_caps_submitted_artifact_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_pending: list[int] = []

    class FakeFuture:
        def __init__(self, value: int) -> None:
            self.value = value

        def result(self) -> pd.DataFrame:
            return pd.DataFrame.from_records([{"value": self.value}])

    class FakeExecutor:
        def submit(self, function, task):
            return FakeFuture(function(task))

    def fake_wait(pending, *, return_when):
        assert return_when is local_module.FIRST_COMPLETED
        observed_pending.append(len(pending))
        completed = {next(iter(pending))}
        return completed, set(pending).difference(completed)

    monkeypatch.setattr(local_module, "wait", fake_wait)
    frames = list(
        local_module._bounded_executor_map(
            FakeExecutor(),
            lambda value: value,
            range(7),
            max_in_flight=2,
        )
    )

    assert len(frames) == 7
    assert observed_pending
    assert max(observed_pending) == 2


def test_beyondarena_local_cli_shards_task_manifest_rows(tmp_path: Path) -> None:
    _copy_iid_artifact_with_parquet(tmp_path)
    manifest = pd.DataFrame(
        [
            {
                "dataset_id": "official_fixture_task-0001",
                "dataset_name": "fixture_iid_small",
                "data_foundry_uri": "iid_small/v1",
                "split": 0,
                "official_split_id": "0",
                "split_id": "r0f0",
                "repeat": 0,
                "fold": 0,
                "task_type": "iid",
                "problem_type": "classification",
                "metric": "roc_auc",
                "size_tier": "tiny",
                "dimensionality": "low",
                "has_text": False,
                "high_cardinality": False,
            },
            {
                "dataset_id": "official_missing_task-0002",
                "dataset_name": "missing_dataset",
                "data_foundry_uri": "missing_dataset/v1",
                "split": 0,
                "official_split_id": "0",
                "split_id": "r0f0",
                "repeat": 0,
                "fold": 0,
                "task_type": "iid",
                "problem_type": "classification",
                "metric": "roc_auc",
                "size_tier": "tiny",
                "dimensionality": "low",
                "has_text": False,
                "high_cardinality": False,
            },
        ]
    )
    manifest_csv = tmp_path / "beyondarena_task_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    shard0 = tmp_path / "local_results_shard0.csv"
    shard1 = tmp_path / "local_results_shard1.csv"

    for shard_index, out_csv in ((0, shard0), (1, shard1)):
        assert (
            main(
                [
                    "--artifact-root",
                    str(tmp_path),
                    "--task-manifest-csv",
                    str(manifest_csv),
                    "--out-csv",
                    str(out_csv),
                    "--backend",
                    "sklearn-smoke",
                    "--manifest-shard-index",
                    str(shard_index),
                    "--manifest-shard-count",
                    "2",
                ]
            )
            == 0
        )

    rows0 = pd.read_csv(shard0)
    rows1 = pd.read_csv(shard1)
    assert rows0["dataset_id"].tolist() == ["official_fixture_task-0001"]
    assert rows1["dataset_id"].tolist() == ["official_missing_task-0002"]
    assert rows0.loc[0, "status"] == "ok"
    assert rows1.loc[0, "execution_status"] == "skipped_missing_artifact"
    assert int(rows0.loc[0, "manifest_shard_index"]) == 0
    assert int(rows1.loc[0, "manifest_shard_index"]) == 1
    assert int(rows0.loc[0, "manifest_shard_count"]) == 2
    assert int(rows1.loc[0, "manifest_shard_count"]) == 2

    merged = tmp_path / "local_results_merged.csv"
    assert (
        main(
            [
                "--merge-shard-glob",
                str(tmp_path / "local_results_shard*.csv"),
                "--out-csv",
                str(merged),
            ]
        )
        == 0
    )
    merged_rows = pd.read_csv(merged)
    assert merged_rows["dataset_id"].tolist() == [
        "official_fixture_task-0001",
        "official_missing_task-0002",
    ]


def test_beyondarena_local_merge_refuses_overlapping_shard_keys(tmp_path: Path) -> None:
    row = {
        "dataset_id": "official_fixture_task-0001",
        "split_id": "0",
        "method": "tabnetics-current",
        "metric": "roc_auc",
        "seed": 42,
        "status": "ok",
    }
    pd.DataFrame([row]).to_csv(tmp_path / "local_results_shard0.csv", index=False)
    pd.DataFrame([row]).to_csv(tmp_path / "local_results_shard1.csv", index=False)

    with pytest.raises(ValueError, match="overlap on comparison keys"):
        local_module.merge_beyondarena_local_result_shards(str(tmp_path / "local_results_shard*.csv"))


def test_beyondarena_local_cli_rejects_invalid_manifest_shard_args(tmp_path: Path) -> None:
    _copy_iid_artifact_with_parquet(tmp_path)
    manifest_csv = tmp_path / "beyondarena_task_manifest.csv"
    pd.DataFrame(
        [
            {
                "dataset_id": "official_fixture_task-0001",
                "data_foundry_uri": "iid_small/v1",
                "split_id": "r0f0",
                "metric": "roc_auc",
            }
        ]
    ).to_csv(manifest_csv, index=False)

    with pytest.raises(ValueError, match="0 <= index < shard count"):
        main(
            [
                "--artifact-root",
                str(tmp_path),
                "--task-manifest-csv",
                str(manifest_csv),
                "--out-csv",
                str(tmp_path / "local_results.csv"),
                "--backend",
                "sklearn-smoke",
                "--manifest-shard-index",
                "2",
                "--manifest-shard-count",
                "2",
            ]
        )


def test_local_runner_tabpfn_candidate_requires_explicit_gpu_flag(tmp_path: Path) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabpfn-candidate",
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["method"] == "tabpfn-candidate"
    assert row["model_profile"] == "tabpfn_candidate"
    assert row["status"] == "skipped"
    assert row["execution_status"] == "deferred_gpu_revalidation"
    assert row["device"] == "gpu"
    assert row["execution_host"] == "public-gpu-host"
    assert row["execution_lane"] == "gpu"
    assert bool(row["allow_gpu_execution"]) is False
    assert "--allow-gpu-execution" in row["skip_reason"]
    assert bool(row["leakage_ok"]) is True


def test_local_runner_tabiclv2_candidate_is_explicitly_gpu_deferred(tmp_path: Path) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabiclv2-candidate",
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["method"] == "TabICLv2"
    assert row["model_profile"] == "tabiclv2_candidate"
    assert row["status"] == "skipped"
    assert row["execution_status"] == "deferred_gpu_revalidation"
    assert row["device"] == "gpu"
    assert row["execution_host"] == "public-gpu-host"
    assert row["execution_lane"] == "gpu"
    assert bool(row["allow_gpu_execution"]) is False


def test_local_runner_tabiclv2_requires_explicit_checkpoint_after_gpu_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    monkeypatch.setattr(local_module, "_cuda_available", lambda: True)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabiclv2-candidate",
        allow_gpu_execution=True,
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert rows.loc[0, "status"] == "skipped"
    assert rows.loc[0, "execution_status"] == "skipped_checkpoint_not_configured"
    assert "--tabiclv2-checkpoint" in rows.loc[0, "skip_reason"]


def test_local_runner_tabiclv2_maps_optional_dependency_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    checkpoint = tmp_path / "tabiclv2.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(local_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(
        local_module,
        "_tabiclv2_package_skip",
        lambda: (
            "skipped_optional_dependency_unavailable",
            "TabICLv2 requires tabicl==2.1.1",
        ),
    )
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabiclv2-candidate",
        allow_gpu_execution=True,
        tabiclv2_checkpoint=str(checkpoint),
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert rows.loc[0, "status"] == "skipped"
    assert rows.loc[0, "execution_status"] == "skipped_optional_dependency_unavailable"
    assert "tabicl==2.1.1" in rows.loc[0, "skip_reason"]


def test_local_runner_tabiclv2_dispatches_only_after_all_explicit_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    checkpoint = tmp_path / "tabiclv2.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(local_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(local_module, "_tabiclv2_package_skip", lambda: None)
    captured: dict[str, object] = {}

    def fake_tabiclv2(*args, **kwargs):
        captured.update(kwargs)
        return (
            {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "log_loss": 0.1,
                "roc_auc": 1.0,
            },
            {"model_name": "TabICLClassifier", "checkpoint_sha256": "abc"},
        )

    monkeypatch.setattr(local_module, "_run_tabiclv2_candidate", fake_tabiclv2)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabiclv2-candidate",
        allow_gpu_execution=True,
        tabiclv2_checkpoint=str(checkpoint),
        tabiclv2_device="cuda:0",
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert rows.loc[0, "status"] == "ok"
    assert rows.loc[0, "execution_backend"] == "tabiclv2-candidate"
    assert rows.loc[0, "method"] == "TabICLv2"
    assert captured["checkpoint"] == str(checkpoint)
    assert captured["device"] == "cuda:0"
    assert captured["min_train_rows"] == 300
    assert captured["max_train_rows"] == 100_000
    assert captured["max_features"] == 2_000


def test_local_runner_tabiclv2_maps_published_regime_failure_to_skip_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tabnetics.classification.tabiclv2 import TabICLv2AvailabilityError

    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    checkpoint = tmp_path / "tabiclv2.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(local_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(local_module, "_tabiclv2_package_skip", lambda: None)

    def reject_size(*args, **kwargs):
        raise TabICLv2AvailabilityError(
            "outside published regime",
            status="skipped_tabiclv2_outside_published_regime",
        )

    monkeypatch.setattr(local_module, "_run_tabiclv2_candidate", reject_size)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabiclv2-candidate",
        allow_gpu_execution=True,
        tabiclv2_checkpoint=str(checkpoint),
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert rows.loc[0, "status"] == "skipped"
    assert rows.loc[0, "execution_status"] == "skipped_tabiclv2_outside_published_regime"


def test_tabiclv2_runner_never_fits_label_vocabulary_from_test_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tabnetics.classification import tabiclv2

    observed: dict[str, np.ndarray] = {}

    class FakeTabICLv2:
        def __init__(self, **kwargs) -> None:
            observed["constructor"] = np.array(sorted(kwargs))

        def fit(self, X, y):
            observed["fit_labels"] = np.array(y, copy=True)
            self.classes_ = np.unique(y)
            self.metadata_ = {}
            return self

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    monkeypatch.setattr(tabiclv2, "TabICLv2Classifier", FakeTabICLv2)
    with pytest.raises(tabiclv2.TabICLv2ContractError, match="training-derived class order"):
        local_module._run_tabiclv2_candidate(
            np.ones((3, 2)),
            np.array(["train-a", "train-b", "train-a"]),
            np.ones((1, 2)),
            np.array(["test-only-class"]),
            checkpoint=tmp_path / "unused.ckpt",
            device="cuda",
            seed=3,
            min_train_rows=300,
            max_train_rows=100_000,
            max_features=2_000,
        )

    np.testing.assert_array_equal(
        observed["fit_labels"],
        np.array(["train-a", "train-b", "train-a"]),
    )


def test_local_runner_tabentics_diakrino_defaults_to_native_method_and_defers_gpu(tmp_path: Path) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabnetics-diakrino",
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["method"] == "TabenticsDiakrino"
    assert row["model_profile"] == "tabentics_diakrino_experimental"
    assert row["status"] == "skipped"
    assert row["execution_status"] == "deferred_gpu_revalidation"
    assert row["device"] == "gpu"
    assert row["execution_host"] == "public-gpu-host"
    assert row["execution_lane"] == "gpu"
    assert bool(row["allow_gpu_execution"]) is False
    assert "--allow-gpu-execution" in row["skip_reason"]


def test_local_runner_tabpfn_candidate_with_gpu_flag_skips_without_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    monkeypatch.setattr(local_module, "_cuda_available", lambda: False)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabpfn-candidate",
        allow_gpu_execution=True,
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert len(rows) == 1
    assert rows.loc[0, "status"] == "skipped"
    assert rows.loc[0, "execution_status"] == "skipped_gpu_unavailable"
    assert rows.loc[0, "device"] == "gpu"
    assert rows.loc[0, "execution_host"] == "public-gpu-host"
    assert bool(rows.loc[0, "allow_gpu_execution"]) is True


def test_local_runner_tabpfn_candidate_skips_when_optional_dependency_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    monkeypatch.setattr(local_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(local_module, "_tabpfn_package_available", lambda: False)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabpfn-candidate",
        allow_gpu_execution=True,
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert len(rows) == 1
    assert rows.loc[0, "status"] == "skipped"
    assert rows.loc[0, "execution_status"] == "skipped_optional_dependency_unavailable"
    assert "tabpfn optional dependency" in rows.loc[0, "skip_reason"]


def test_local_runner_tabentics_diakrino_requires_checkpoint_after_gpu_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    monkeypatch.setattr(local_module, "_cuda_available", lambda: True)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabnetics-diakrino",
        allow_gpu_execution=True,
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert len(rows) == 1
    assert rows.loc[0, "method"] == "TabenticsDiakrino"
    assert rows.loc[0, "status"] == "skipped"
    assert rows.loc[0, "execution_status"] == "skipped_native_diakrino_checkpoint_not_configured"
    assert "--tabnetics-diakrino-checkpoint" in rows.loc[0, "skip_reason"]


def test_native_diakrino_config_maps_teacher_refiner_steps_alias() -> None:
    @dataclass(frozen=True)
    class FakeConfig:
        d_model: int = 16
        fs_refiner_steps: int = 4

    cfg = local_module._tabentics_diakrino_config_from_payload(
        FakeConfig,
        {
            "model_config": {
                "d_model": 1024,
                "refiner_steps": 0,
                "ignored_field": "ignored",
            }
        },
    )

    assert cfg.d_model == 1024
    assert cfg.fs_refiner_steps == 0


def test_native_diakrino_loader_treats_unprefixed_teacher_checkpoint_as_fs_teacher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    @dataclass(frozen=True)
    class FakeConfig:
        pass

    class FakeClassifier(torch.nn.Module):
        def __init__(self, config: FakeConfig | None = None) -> None:
            super().__init__()
            self.config = config or FakeConfig()
            self.feature_selector = torch.nn.Module()
            self.feature_selector.stats_encoder = torch.nn.Linear(2, 2)
            self.query_value_encoder = torch.nn.Linear(1, 1)
            for parameter in self.parameters():
                torch.nn.init.zeros_(parameter)

    fake_models = types.ModuleType("tabnetics.classification.tabentics_diakrino_fs_classifier")
    fake_models.TabenticsDiakrinoFSClassifier = FakeClassifier
    fake_models.TabenticsDiakrinoFSClassifierConfig = FakeConfig
    monkeypatch.setitem(sys.modules, "tabnetics.classification.tabentics_diakrino_fs_classifier", fake_models)

    checkpoint = tmp_path / "fs_teacher_with_classifier_key_overlap.pt"
    teacher_stats_weight = torch.full((2, 2), 5.0)
    teacher_stats_bias = torch.full((2,), 6.0)
    torch.save(
        {
            "model_state_dict": {
                "stats_encoder.weight": teacher_stats_weight,
                "stats_encoder.bias": teacher_stats_bias,
                "query_value_encoder.weight": torch.full((1, 1), 7.0),
                "query_value_encoder.bias": torch.full((1,), 8.0),
            },
            "model_config": {},
            "epoch": 3,
            "step": 4,
        },
        checkpoint,
    )

    model, report = local_module._load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_format"] == "fs_teacher"
    assert report["loaded_count"] == 2
    assert report["source_epoch"] == 3
    assert report["source_step"] == 4
    assert torch.equal(model.feature_selector.stats_encoder.weight, teacher_stats_weight)
    assert torch.equal(model.feature_selector.stats_encoder.bias, teacher_stats_bias)
    assert torch.equal(model.query_value_encoder.weight, torch.zeros_like(model.query_value_encoder.weight))
    assert report["discarded_prefixes"]["query_value_encoder"] == 2


def test_classification_values_normalizes_probability_rows() -> None:
    values = local_module._classification_values_from_proba(
        np.array([0, 1, 1]),
        np.array(
            [
                [2.0, 0.0],
                [0.2, 0.6],
                [np.nan, np.inf],
            ]
        ),
        n_classes=2,
    )

    assert values["accuracy"] == pytest.approx(2.0 / 3.0)
    assert np.isfinite(values["log_loss"])
    assert np.isfinite(values["roc_auc"])


def test_local_runner_tabentics_diakrino_calls_native_adapter_when_checkpoint_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy_iid_artifact_with_parquet(tmp_path)
    checkpoint = tmp_path / "native_diakrino.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(local_module, "_cuda_available", lambda: True)

    def fake_native_adapter(*args, **kwargs):
        assert kwargs["checkpoint"] == str(checkpoint)
        assert kwargs["max_features"] == 7
        assert kwargs["batch_size"] == 3
        assert kwargs["support_joint_serving_cache"] is True
        assert kwargs["retry_cuda_oom_microbatch"] is True
        return (
            {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "log_loss": 0.1,
                "roc_auc": 1.0,
            },
            {
                "model_name": "TabenticsDiakrinoFSClassifier",
                "native_diakrino_checkpoint": str(checkpoint),
                "native_diakrino_used_features": 2,
            },
        )

    monkeypatch.setattr(local_module, "_run_tabentics_diakrino_native", fake_native_adapter)
    config = BeyondArenaLocalRunConfig(
        artifact_root=artifact,
        backend="tabnetics-diakrino",
        allow_gpu_execution=True,
        tabentics_diakrino_checkpoint=str(checkpoint),
        tabentics_diakrino_max_features=7,
        tabentics_diakrino_batch_size=3,
        tabentics_diakrino_support_joint_serving_cache=True,
        tabentics_diakrino_retry_cuda_oom_microbatch=True,
        max_splits_per_artifact=1,
    )

    rows = run_local_beyondarena_artifact(artifact, config=config)

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["method"] == "TabenticsDiakrino"
    assert row["status"] == "ok"
    assert row["execution_status"] == "ok"
    assert row["execution_backend"] == "tabnetics-diakrino"
    assert row["native_diakrino_checkpoint"] == str(checkpoint)
    assert int(row["native_diakrino_used_features"]) == 2
    assert float(row["metric_value"]) == 1.0


def test_beyondarena_local_cli_defaults_tabentics_diakrino_method(tmp_path: Path) -> None:
    _copy_iid_artifact_with_parquet(tmp_path)
    out_csv = tmp_path / "local_diakrino_rows.csv"

    rc = main(
        [
            "--artifact-root",
            str(tmp_path),
            "--out-csv",
            str(out_csv),
            "--backend",
            "tabnetics-diakrino",
            "--max-artifacts",
            "1",
        ]
    )

    assert rc == 0
    rows = pd.read_csv(out_csv)
    assert len(rows) == 1
    assert rows.loc[0, "method"] == "TabenticsDiakrino"
    assert rows.loc[0, "model_profile"] == "tabentics_diakrino_experimental"
    assert rows.loc[0, "execution_status"] == "deferred_gpu_revalidation"

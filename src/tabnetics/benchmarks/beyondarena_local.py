"""Local BeyondArena artifact execution for tabnetics comparison rows.

This module is intentionally opt-in: it only reads materialized local
DataFoundry artifacts and never downloads BeyondArena parquet payloads.  The
default backend runs the current tabnetics pipeline on classification splits;
CI can use the explicit ``sklearn-smoke`` backend to exercise result schemas
for classification and regression without producing a tabnetics performance
claim.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import platform
import socket
import tempfile
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from tabnetics.datasets.beyondarena import (
    DATASET_PARQUET,
    TEXT_CACHE_BASENAME,
    BeyondArenaPreprocessingProfile,
    BeyondArenaSplit,
    BeyondArenaUnavailableError,
    apply_beyondarena_preprocessing,
    build_beyondarena_resampling_context,
    discover_local_beyondarena_specs,
    load_beyondarena_dataset,
    load_beyondarena_spec,
    load_beyondarena_splits,
    validate_beyondarena_split_leakage,
)

from .beyondarena_compare import metric_lower_is_better, normalize_beyondarena_metric_name
from .result_journal import AtomicResultJournal


@dataclass(frozen=True)
class BeyondArenaLocalRunConfig:
    """Runtime controls for local BeyondArena result materialization."""

    artifact_root: Path
    task_manifest_csv: Optional[Path] = None
    backend: str = "tabnetics-current"
    method: str = ""
    model_profile: str = ""
    seed: int = 42
    device: str = ""
    execution_host: str = ""
    execution_lane: str = ""
    max_artifacts: Optional[int] = None
    max_splits_per_artifact: Optional[int] = None
    max_workers: int = 1
    max_in_flight_artifacts: Optional[int] = None
    manifest_shard_index: Optional[int] = None
    manifest_shard_count: Optional[int] = None
    allow_gpu_execution: bool = False
    tabentics_diakrino_checkpoint: str = ""
    tabentics_diakrino_max_features: int = 1024
    tabentics_diakrino_batch_size: int = 128
    tabentics_diakrino_support_joint_serving_cache: bool = False
    tabentics_diakrino_retry_cuda_oom_microbatch: bool = False
    # Opt-in CPU execution lane for the native tabnetics-diakrino backend
    # (public_cpu_host_1/public_cpu_host_2 validation); tabpfn-candidate stays GPU-gated.
    tabentics_diakrino_allow_cpu: bool = False
    # Explicit ablation/diagnostic override for mid-training snapshots without a
    # passing head_trust_record; the override is stamped into result-row meta.
    tabentics_diakrino_allow_untrusted_checkpoint: bool = False
    tabiclv2_checkpoint: str = ""
    tabiclv2_device: str = "cuda"
    tabiclv2_min_train_rows: int = 300
    tabiclv2_max_train_rows: int = 100_000
    tabiclv2_max_features: int = 2_000
    on_error: str = "row"  # row | raise


_GPU_BACKENDS = {"tabpfn-candidate", "tabnetics-diakrino", "tabiclv2-candidate"}
_GPU_REVALIDATION_REASON = (
    "requires --allow-gpu-execution after public-gpu-host GPU access/capacity has been revalidated"
)
_BEYONDARENA_RESULT_KEY_FIELDS = ("dataset_id", "split_id", "method", "metric", "seed")
_EXPECTED_ARTIFACT_JSON_FILES = (
    "container_metadata.json",
    "dataset_metadata.dataset-mold-v1.json",
    "dtypes.json",
    "experiment_metadata.predictive-ml-splits-mold-v1.json",
    "task_metadata.predictive-ml-task-mold-v1.json",
)
_LOCK_V2_FIELDS = frozenset(
    {"version", "claim_id", "hostname", "boot_id", "pid", "pid_start_time"}
)
_PACKAGE_IMPORT_NAMES = {
    "huggingface-hub": "huggingface_hub",
    "scikit-learn": "sklearn",
}


def _metric_error_value(metric: str, value: Any) -> Any:
    """Return TabArena-style error scale for a raw local metric value."""

    if value is pd.NA:
        return pd.NA
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return pd.NA
    if not np.isfinite(numeric):
        return pd.NA
    if metric_lower_is_better(metric):
        return numeric
    return 1.0 - numeric


def _default_method_for_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized == "sklearn-smoke":
        return "tabnetics-current-smoke"
    if normalized == "tabpfn-candidate":
        return "tabpfn-candidate"
    if normalized == "tabnetics-diakrino":
        return "TabenticsDiakrino"
    if normalized == "tabiclv2-candidate":
        return "TabICLv2"
    return "tabnetics-current"


def _default_model_profile_for_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized == "sklearn-smoke":
        return "sklearn_smoke"
    if normalized == "tabpfn-candidate":
        return "tabpfn_candidate"
    if normalized == "tabnetics-diakrino":
        return "tabentics_diakrino_experimental"
    if normalized == "tabiclv2-candidate":
        return "tabiclv2_candidate"
    return normalized.replace("-", "_")


def _default_device_for_backend(backend: str) -> str:
    return "gpu" if str(backend).strip().lower() in _GPU_BACKENDS else "cpu"


def _default_execution_lane_for_backend(backend: str) -> str:
    return "gpu" if str(backend).strip().lower() in _GPU_BACKENDS else "cpu"


def _default_execution_host_for_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized == "sklearn-smoke":
        return "local-ci"
    if normalized in _GPU_BACKENDS:
        return "public-gpu-host"
    return "public_cpu_host_1/public_cpu_host_2"


def _normalize_config(config: BeyondArenaLocalRunConfig) -> BeyondArenaLocalRunConfig:
    backend = str(config.backend).strip().lower()
    max_workers = int(max(1, int(config.max_workers or 1)))
    max_in_flight = (
        max_workers
        if config.max_in_flight_artifacts is None
        else int(max(1, int(config.max_in_flight_artifacts)))
    )
    shard_index, shard_count = _validate_manifest_shard(
        config.manifest_shard_index,
        config.manifest_shard_count,
        task_manifest_csv=config.task_manifest_csv,
    )
    if int(config.tabiclv2_min_train_rows) < 300:
        raise ValueError("tabiclv2_min_train_rows cannot relax the published minimum of 300")
    if int(config.tabiclv2_max_train_rows) > 100_000:
        raise ValueError("tabiclv2_max_train_rows cannot exceed the published maximum of 100000")
    if int(config.tabiclv2_max_features) > 2_000:
        raise ValueError("tabiclv2_max_features cannot exceed the published maximum of 2000")
    if int(config.tabiclv2_min_train_rows) > int(config.tabiclv2_max_train_rows):
        raise ValueError("tabiclv2_min_train_rows must be <= tabiclv2_max_train_rows")
    return replace(
        config,
        backend=backend,
        method=str(config.method or _default_method_for_backend(backend)),
        model_profile=str(config.model_profile or _default_model_profile_for_backend(backend)),
        device=str(config.device or _default_device_for_backend(backend)),
        execution_host=str(config.execution_host or _default_execution_host_for_backend(backend)),
        execution_lane=str(config.execution_lane or _default_execution_lane_for_backend(backend)),
        max_workers=max_workers,
        max_in_flight_artifacts=max_in_flight,
        manifest_shard_index=shard_index,
        manifest_shard_count=shard_count,
        tabentics_diakrino_checkpoint=str(config.tabentics_diakrino_checkpoint or ""),
        tabentics_diakrino_max_features=int(max(1, int(config.tabentics_diakrino_max_features or 1024))),
        tabentics_diakrino_batch_size=int(max(1, int(config.tabentics_diakrino_batch_size or 128))),
        tabiclv2_checkpoint=str(config.tabiclv2_checkpoint or ""),
        tabiclv2_device=str(config.tabiclv2_device or "cuda"),
        tabiclv2_min_train_rows=int(max(1, int(config.tabiclv2_min_train_rows or 300))),
        tabiclv2_max_train_rows=int(max(1, int(config.tabiclv2_max_train_rows or 100_000))),
        tabiclv2_max_features=int(max(1, int(config.tabiclv2_max_features or 2_000))),
    )


def _sha256_file(path: Path) -> str:
    source = Path(path)
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = source.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"file changed while computing result-journal identity: {source}")
    return digest.hexdigest()


def _file_identity(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {"path": "", "sha256": ""}
    source = Path(path)
    return {"path": str(source.resolve()), "sha256": _sha256_file(source)}


def _checkpoint_identity(path_value: str) -> dict[str, Any]:
    if not str(path_value).strip():
        return {"path": "", "size": 0, "sha256": ""}
    path = Path(path_value)
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "size": -1, "sha256": ""}
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "sha256": _sha256_file(path),
    }


def _package_source_identity() -> dict[str, Any]:
    """Hash the installed Python source tree that determines benchmark behavior."""

    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    file_count = 0
    for source in sorted(package_root.rglob("*.py"), key=lambda path: path.as_posix()):
        relative = source.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        file_count += 1
    return {
        "root": str(package_root),
        "python_file_count": int(file_count),
        "sha256": digest.hexdigest(),
    }


def _distribution_identity(package: str) -> dict[str, Any]:
    try:
        version = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        available = False
        version = ""
    except Exception as exc:
        return {
            "available": None,
            "importable": None,
            "version": "",
            "resolution_error": type(exc).__name__,
        }
    else:
        available = True
        version = str(version)
    import_name = _PACKAGE_IMPORT_NAMES.get(package, package.replace("-", "_"))
    try:
        importable = importlib.util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        importable = False
    return {"available": available, "importable": importable, "version": version}


def _backend_dependency_packages(backend: str) -> tuple[str, ...]:
    base = {
        "tabnetics",
        "fastparquet",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
    }
    normalized = str(backend).strip().lower()
    if normalized == "tabnetics-current":
        base.update(
            {
                "boruta",
                "catboost",
                "flaml",
                "lightgbm",
                "mapie",
                "optuna",
                "pytabkit",
                "pyvinecopulib",
                "shap",
                "statsmodels",
                "threadpoolctl",
                "torch",
                "xgboost",
            }
        )
    elif normalized == "tabpfn-candidate":
        base.update({"tabpfn", "threadpoolctl", "torch"})
    elif normalized == "tabnetics-diakrino":
        base.update({"threadpoolctl", "torch"})
    elif normalized == "tabiclv2-candidate":
        base.update({"huggingface-hub", "tabicl", "torch"})
    elif normalized == "tabnetics-current-fast":
        base.add("threadpoolctl")
    return tuple(sorted(base))


def _parquet_engine_usability(engine: str) -> dict[str, Any]:
    try:
        from pandas.io.parquet import get_engine

        implementation = get_engine(str(engine))
    except Exception as exc:
        return {
            "usable": False,
            "implementation": "",
            "resolution_error": type(exc).__name__,
        }
    implementation_name = (
        f"{type(implementation).__module__}.{type(implementation).__qualname__}"
    )
    lowered = implementation_name.lower()
    selected_engine = (
        "pyarrow"
        if "pyarrow" in lowered
        else "fastparquet"
        if "fastparquet" in lowered
        else "unknown"
    )
    return {
        "usable": True,
        "implementation": implementation_name,
        "selected_engine": selected_engine,
    }


def _parquet_runtime_identity(packages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    engines = {
        engine: {
            "distribution": packages[engine],
            "runtime": _parquet_engine_usability(engine),
        }
        for engine in ("pyarrow", "fastparquet")
    }
    auto = _parquet_engine_usability("auto")
    try:
        configured_policy = str(pd.get_option("io.parquet.engine"))
    except Exception:
        configured_policy = "unresolved"
    return {
        "pandas_policy": configured_policy,
        "selected_engine": str(auto.get("selected_engine", "")),
        "auto": auto,
        "engines": engines,
    }


def _is_hex_digest(value: str, *, lengths: tuple[int, ...] = (40, 64)) -> bool:
    normalized = str(value).strip().lower()
    return len(normalized) in lengths and all(character in "0123456789abcdef" for character in normalized)


def _model_cache_file_identity(path: Path, *, root: Path) -> dict[str, Any]:
    stat = path.stat()
    resolved = path.resolve()
    resolved_digest = (
        resolved.name.lower()
        if _is_hex_digest(resolved.name) and "blobs" in resolved.parts
        else ""
    )
    identity = {
        "path": path.relative_to(root).as_posix(),
        "resolved_path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "content_address": resolved_digest,
    }
    if not resolved_digest:
        identity["sha256"] = _sha256_file(path)
    return identity


def _tabpfn_default_model_identity(tabpfn_package: dict[str, Any]) -> dict[str, Any]:
    """Identify explicit or cached models selected by TabPFN's default path."""

    environment_names = (
        "TABPFN_MODEL_PATH",
        "TABPFN_MODEL_CACHE_DIR",
        "TABPFN_CACHE_DIR",
        "HF_HOME",
    )
    environment = {name: str(os.environ.get(name, "")) for name in environment_names}
    locations: list[tuple[str, Path]] = []
    if environment["TABPFN_MODEL_PATH"]:
        locations.append(("explicit_model", Path(environment["TABPFN_MODEL_PATH"])))
    for name in ("TABPFN_MODEL_CACHE_DIR", "TABPFN_CACHE_DIR"):
        if environment[name]:
            locations.append((name.lower(), Path(environment[name])))
    locations.append(("default_cache", Path.home() / ".cache" / "tabpfn"))
    locations.append(("legacy_cache", Path.home() / ".tabpfn"))
    hf_home = Path(environment["HF_HOME"]) if environment["HF_HOME"] else Path.home() / ".cache" / "huggingface"
    hf_hub = hf_home / "hub"
    if hf_hub.is_dir():
        for candidate in sorted(hf_hub.iterdir(), key=lambda path: path.name.lower()):
            if candidate.is_dir() and "tabpfn" in candidate.name.lower():
                locations.append(("huggingface_cache", candidate))

    seen: set[str] = set()
    location_records: list[dict[str, Any]] = []
    model_files: list[dict[str, Any]] = []
    model_suffixes = {".ckpt", ".json", ".pt", ".pth", ".safetensors"}
    for source, location in locations:
        canonical = str(location.expanduser().resolve(strict=False))
        if canonical in seen:
            continue
        seen.add(canonical)
        path = Path(canonical)
        state = "file" if path.is_file() else "directory" if path.is_dir() else "missing"
        location_records.append({"source": source, "path": canonical, "state": state})
        if path.is_file():
            model_files.append({"source": source, **_model_cache_file_identity(path, root=path.parent)})
            continue
        if not path.is_dir():
            continue
        for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
            if candidate.is_file() and candidate.suffix.lower() in model_suffixes:
                model_files.append(
                    {"source": source, **_model_cache_file_identity(candidate, root=path)}
                )

    available = (
        tabpfn_package.get("available") is True
        and tabpfn_package.get("importable", True) is not False
    )
    if not available:
        resolution = "package_unavailable"
    elif model_files:
        resolution = "cached_or_explicit_model"
    else:
        resolution = "package_default_unresolved"
    return {
        "resolution": resolution,
        "environment": environment,
        "locations": location_records,
        "model_files": model_files,
    }


def _runtime_dependency_identity(backend: str) -> dict[str, Any]:
    packages = {
        package: _distribution_identity(package)
        for package in _backend_dependency_packages(backend)
    }
    payload: dict[str, Any] = {
        "backend": str(backend).strip().lower(),
        "python": platform.python_version(),
        "packages": packages,
        "parquet": _parquet_runtime_identity(packages),
        "tabnetics_source": _package_source_identity(),
    }
    if str(backend).strip().lower() == "tabpfn-candidate":
        payload["tabpfn_default_model"] = _tabpfn_default_model_identity(
            packages["tabpfn"]
        )
    return payload


def _result_journal_context(config: BeyondArenaLocalRunConfig) -> dict[str, Any]:
    """Bind resume data to inputs and settings that can change result semantics."""

    tabiclv2_identity: dict[str, Any] = {}
    if str(config.backend).strip().lower() == "tabiclv2-candidate":
        from tabnetics.classification.tabiclv2 import tabiclv2_contract_identity

        tabiclv2_identity = {
            "checkpoint": _checkpoint_identity(config.tabiclv2_checkpoint),
            "device": config.tabiclv2_device,
            "min_train_rows": int(config.tabiclv2_min_train_rows),
            "max_train_rows": int(config.tabiclv2_max_train_rows),
            "max_features": int(config.tabiclv2_max_features),
            "contract": tabiclv2_contract_identity(),
        }

    return {
        "runner": "beyondarena-local-v4",
        "runtime": _runtime_dependency_identity(config.backend),
        "artifact_root": str(Path(config.artifact_root).resolve()),
        "selected_artifacts": _selected_artifact_inventory(config),
        "task_manifest": _file_identity(config.task_manifest_csv),
        "backend": config.backend,
        "method": config.method,
        "model_profile": config.model_profile,
        "seed": int(config.seed),
        "device": config.device,
        "execution_lane": config.execution_lane,
        "max_artifacts": config.max_artifacts,
        "max_splits_per_artifact": config.max_splits_per_artifact,
        "manifest_shard_index": config.manifest_shard_index,
        "manifest_shard_count": config.manifest_shard_count,
        "allow_gpu_execution": bool(config.allow_gpu_execution),
        "tabentics_diakrino_checkpoint": _checkpoint_identity(config.tabentics_diakrino_checkpoint),
        "tabentics_diakrino_max_features": int(config.tabentics_diakrino_max_features),
        "tabentics_diakrino_batch_size": int(config.tabentics_diakrino_batch_size),
        "tabentics_diakrino_support_joint_serving_cache": bool(
            config.tabentics_diakrino_support_joint_serving_cache
        ),
        "tabentics_diakrino_retry_cuda_oom_microbatch": bool(
            config.tabentics_diakrino_retry_cuda_oom_microbatch
        ),
        "tabentics_diakrino_allow_cpu": bool(config.tabentics_diakrino_allow_cpu),
        "tabentics_diakrino_allow_untrusted_checkpoint": bool(
            config.tabentics_diakrino_allow_untrusted_checkpoint
        ),
        "tabiclv2": tabiclv2_identity,
        "preprocessing_profile": BeyondArenaPreprocessingProfile().profile_id,
        "on_error": config.on_error,
    }


def _artifact_dirs(root: Path, *, max_artifacts: Optional[int] = None) -> tuple[Path, ...]:
    specs = discover_local_beyondarena_specs(root)
    paths = [spec.artifact_dir for spec in specs if spec.artifact_dir is not None]
    if max_artifacts is not None:
        paths = paths[: int(max_artifacts)]
    return tuple(Path(path) for path in paths)


def _read_task_manifest(source: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if source is None:
        return None
    frame = pd.read_csv(source)
    if frame.empty:
        return frame
    required = {"dataset_id", "split_id", "metric"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"BeyondArena task manifest missing required columns: {missing}")
    return frame


def _validate_manifest_shard(
    shard_index: Optional[int],
    shard_count: Optional[int],
    *,
    task_manifest_csv: Optional[Path],
) -> tuple[Optional[int], Optional[int]]:
    if shard_index is None and shard_count is None:
        return None, None
    if task_manifest_csv is None:
        raise ValueError("manifest sharding requires --task-manifest-csv")
    if shard_index is None or shard_count is None:
        raise ValueError("manifest sharding requires both shard index and shard count")
    index = int(shard_index)
    count = int(shard_count)
    if count <= 0:
        raise ValueError("manifest shard count must be positive")
    if index < 0 or index >= count:
        raise ValueError("manifest shard index must satisfy 0 <= index < shard count")
    return index, count


def _apply_manifest_shard(
    manifest: Optional[pd.DataFrame],
    *,
    shard_index: Optional[int],
    shard_count: Optional[int],
) -> Optional[pd.DataFrame]:
    if manifest is None or shard_index is None or shard_count is None:
        return manifest
    if manifest.empty:
        return manifest.copy()
    source = manifest.reset_index(drop=True).copy()
    positions = np.arange(len(source), dtype=int)
    shard = source.loc[(positions % int(shard_count)) == int(shard_index)].copy()
    return shard.reset_index(drop=True)


def _norm_key(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").strip("/").lower()


def _artifact_manifest_keys(
    artifact_dir: Path,
    *,
    artifact_root: Path,
    spec: Any,
) -> set[str]:
    keys = {
        _norm_key(spec.beyondarena_id),
        _norm_key(spec.dataset_name),
        _norm_key(spec.artifact_revision),
        _norm_key(artifact_dir.name),
        _norm_key(f"{artifact_dir.parent.name}/{artifact_dir.name}"),
    }
    try:
        rel = artifact_dir.relative_to(artifact_root)
        keys.add(_norm_key(rel.as_posix()))
        if rel.name.lower().startswith("v") and rel.parent != Path("."):
            keys.add(_norm_key(rel.parent.as_posix()))
    except ValueError:
        pass
    for value in getattr(spec, "metadata_paths", {}).values():
        if value:
            keys.add(_norm_key(value))
    return {key for key in keys if key}


def _manifest_rows_for_artifact(
    manifest: Optional[pd.DataFrame],
    artifact_dir: Path,
    *,
    artifact_root: Path,
    spec: Any,
) -> Optional[pd.DataFrame]:
    if manifest is None:
        return None
    if manifest.empty:
        return manifest.copy()
    keys = _artifact_manifest_keys(artifact_dir, artifact_root=artifact_root, spec=spec)
    mask = pd.Series(False, index=manifest.index)
    for column in ("data_foundry_uri", "artifact_revision", "dataset_name", "dataset_id", "local_dataset_id"):
        if column not in manifest.columns:
            continue
        mask = mask | manifest[column].map(_norm_key).isin(keys)
    rows = manifest.loc[mask].copy()
    rows["_manifest_row_index"] = rows.index
    return rows.reset_index(drop=True)


def _relative_artifact_path(path: Path, *, artifact_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(artifact_root.resolve())
    except ValueError:
        return str(path.resolve())
    return relative.as_posix() or "."


def _materialized_file_identity(
    path: Path,
    *,
    artifact_root: Path,
    authoritative_sha256: str = "",
) -> dict[str, Any]:
    relative_path = _relative_artifact_path(path, artifact_root=artifact_root)
    if not path.exists():
        return {
            "path": relative_path,
            "state": "missing",
            "size": -1,
            "identity_kind": (
                "authoritative_sha256" if authoritative_sha256 else "missing"
            ),
            "sha256": authoritative_sha256,
        }
    if not path.is_file():
        return {
            "path": relative_path,
            "state": "not_file",
            "size": -1,
            "identity_kind": "invalid",
            "sha256": "",
        }
    stat = path.stat()
    return {
        "path": relative_path,
        "state": "present",
        "size": int(stat.st_size),
        "identity_kind": (
            "authoritative_sha256" if authoritative_sha256 else "computed_sha256"
        ),
        "sha256": authoritative_sha256 or _sha256_file(path),
    }


def _artifact_metadata_inventory(
    artifact_dir: Path,
    *,
    artifact_root: Path,
) -> list[dict[str, Any]]:
    names = set(_EXPECTED_ARTIFACT_JSON_FILES)
    names.update(path.name for path in artifact_dir.glob("*.json"))
    return [
        _materialized_file_identity(
            artifact_dir / name,
            artifact_root=artifact_root,
        )
        for name in sorted(names)
    ]


def _authoritative_container_checksum(artifact_dir: Path) -> str:
    container_path = artifact_dir / "container_metadata.json"
    if not container_path.is_file():
        return ""
    try:
        payload = json.loads(container_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    checksum = str(payload.get("checksum", "")).strip().lower() if isinstance(payload, dict) else ""
    return checksum if _is_hex_digest(checksum, lengths=(64,)) else ""


def _artifact_inventory_entry(
    artifact_dir: Path,
    *,
    artifact_root: Path,
    spec: Any,
) -> dict[str, Any]:
    authoritative_checksum = _authoritative_container_checksum(artifact_dir)
    return {
        "relative_path": _relative_artifact_path(
            artifact_dir,
            artifact_root=artifact_root,
        ),
        "dataset_id": str(spec.beyondarena_id),
        "dataset_name": str(spec.dataset_name),
        "artifact_revision": str(spec.artifact_revision or ""),
        "metadata": _artifact_metadata_inventory(
            artifact_dir,
            artifact_root=artifact_root,
        ),
        "dataset": _materialized_file_identity(
            artifact_dir / DATASET_PARQUET,
            artifact_root=artifact_root,
            authoritative_sha256=authoritative_checksum,
        ),
        "text_cache": _materialized_file_identity(
            artifact_dir / TEXT_CACHE_BASENAME,
            artifact_root=artifact_root,
        ),
    }


def _selected_artifact_inventory(config: BeyondArenaLocalRunConfig) -> list[dict[str, Any]]:
    manifest = _apply_manifest_shard(
        _read_task_manifest(config.task_manifest_csv),
        shard_index=config.manifest_shard_index,
        shard_count=config.manifest_shard_count,
    )
    inventory: list[dict[str, Any]] = []
    for artifact_dir in _artifact_dirs(
        config.artifact_root,
        max_artifacts=config.max_artifacts,
    ):
        spec = load_beyondarena_spec(artifact_dir)
        manifest_rows = _manifest_rows_for_artifact(
            manifest,
            artifact_dir,
            artifact_root=config.artifact_root,
            spec=spec,
        )
        if manifest is not None and (manifest_rows is None or manifest_rows.empty):
            continue
        inventory.append(
            _artifact_inventory_entry(
                artifact_dir,
                artifact_root=config.artifact_root,
                spec=spec,
            )
        )
    inventory.sort(
        key=lambda item: (
            str(item["relative_path"]),
            str(item["dataset_id"]),
            str(item["artifact_revision"]),
        )
    )
    return inventory


def _value_or_default(row: Optional[pd.Series], column: str, default: Any) -> Any:
    if row is None or column not in row.index:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return value


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _split_for_manifest_row(
    splits: Sequence[BeyondArenaSplit],
    row: pd.Series,
) -> Optional[BeyondArenaSplit]:
    repeat = _int_or_none(row.get("repeat"))
    fold = _int_or_none(row.get("fold"))
    if repeat is not None and fold is not None:
        for split in splits:
            if str(split.repeat) == str(repeat) and str(split.fold) == str(fold):
                return split
    aliases = {
        _norm_key(row.get("split_id")),
        _norm_key(row.get("official_split_id")),
        _norm_key(row.get("split")),
    }
    if repeat is not None and fold is not None:
        aliases.add(_norm_key(f"{repeat}:{fold}"))
        aliases.add(_norm_key(f"r{repeat}f{fold}"))
    aliases = {alias for alias in aliases if alias}
    for split in splits:
        split_aliases = {
            _norm_key(split.split_id),
            _norm_key(f"{split.repeat}:{split.fold}"),
            _norm_key(f"r{split.repeat}f{split.fold}"),
        }
        if aliases.intersection(split_aliases):
            return split
    return None


def _placeholder_split_for_manifest_row(row: pd.Series) -> BeyondArenaSplit:
    repeat = str(_value_or_default(row, "repeat", "manifest"))
    fold = str(_value_or_default(row, "fold", "0"))
    split_id = str(_value_or_default(row, "split_id", f"r{repeat}f{fold}"))
    return BeyondArenaSplit(
        split_id=split_id,
        repeat=repeat,
        fold=fold,
        train_indices=tuple(),
        test_indices=tuple(),
        source="manifest_missing_local_split",
    )


def _execution_items(
    splits: Sequence[BeyondArenaSplit],
    manifest_rows: Optional[pd.DataFrame],
) -> tuple[tuple[BeyondArenaSplit, Optional[pd.Series]], ...]:
    if manifest_rows is None:
        return tuple((split, None) for split in splits)
    items: list[tuple[BeyondArenaSplit, Optional[pd.Series]]] = []
    for _, row in manifest_rows.iterrows():
        split = _split_for_manifest_row(splits, row)
        if split is None:
            split = _placeholder_split_for_manifest_row(row)
        items.append((split, row))
    return tuple(items)


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).astype(float)


def _split_arrays(X: pd.DataFrame, y: pd.Series, split: BeyondArenaSplit) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx = list(split.train_indices)
    test_idx = list(split.test_indices)
    X_num = _numeric_frame(X)
    return (
        X_num.iloc[train_idx].to_numpy(dtype=float),
        y.iloc[train_idx].to_numpy(),
        X_num.iloc[test_idx].to_numpy(dtype=float),
        y.iloc[test_idx].to_numpy(),
    )


def _metric_from_mapping(metric: str, values: dict[str, float]) -> float:
    key = normalize_beyondarena_metric_name(metric)
    aliases = {
        "f1": "macro_f1",
        "macro_f1": "macro_f1",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "log_loss": "log_loss",
        "mae": "mae",
        "mean_absolute_error": "mae",
        "mse": "mse",
        "mean_squared_error": "mse",
        "rmse": "rmse",
        "root_mean_squared_error": "rmse",
        "r2": "r2",
    }
    selected = aliases.get(key, key)
    return float(values.get(selected, float("nan")))


def _align_proba(estimator: Any, proba: np.ndarray, *, n_classes: int) -> np.ndarray:
    aligned = np.zeros((int(proba.shape[0]), int(n_classes)), dtype=float)
    classes = np.asarray(getattr(estimator, "classes_", np.arange(proba.shape[1])), dtype=int)
    for src_idx, cls in enumerate(classes.tolist()):
        if 0 <= int(cls) < int(n_classes):
            aligned[:, int(cls)] = proba[:, int(src_idx)]
    row_sum = aligned.sum(axis=1)
    missing = row_sum <= 0
    if np.any(missing):
        aligned[missing, :] = 1.0 / max(1, int(n_classes))
    else:
        aligned = aligned / row_sum[:, None]
    return aligned


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _tabpfn_package_available() -> bool:
    try:
        return importlib.util.find_spec("tabpfn") is not None
    except Exception:
        return False


def _tabiclv2_package_skip() -> Optional[tuple[str, str]]:
    identity = _distribution_identity("tabicl")
    if identity.get("available") is not True or identity.get("importable") is not True:
        return (
            "skipped_optional_dependency_unavailable",
            "TabICLv2 requires the optional tabicl==2.1.1 package",
        )
    if str(identity.get("version", "")) != "2.1.1":
        return (
            "skipped_upstream_contract_mismatch",
            "TabICLv2 requires exact tabicl==2.1.1; "
            f"found {identity.get('version', '') or 'unknown'}",
        )
    return None


def _gpu_backend_skip(
    backend: str,
    config: BeyondArenaLocalRunConfig,
) -> Optional[tuple[str, str]]:
    if backend not in _GPU_BACKENDS:
        return None
    native_cpu_lane = backend == "tabnetics-diakrino" and bool(config.tabentics_diakrino_allow_cpu)
    if not native_cpu_lane:
        if not bool(config.allow_gpu_execution):
            return ("deferred_gpu_revalidation", _GPU_REVALIDATION_REASON)
        if not _cuda_available():
            return (
                "skipped_gpu_unavailable",
                "CUDA device is not visible; run GPU-required BeyondArena backends on public-gpu-host after revalidation",
            )
    if backend == "tabnetics-diakrino":
        checkpoint = str(config.tabentics_diakrino_checkpoint or "").strip()
        if not checkpoint:
            return (
                "skipped_native_diakrino_checkpoint_not_configured",
                (
                    "native Tabnetics Diakrino execution requires an explicit "
                    "--tabnetics-diakrino-checkpoint after public-gpu-host revalidation"
                ),
            )
        if not Path(checkpoint).exists():
            return (
                "skipped_native_diakrino_checkpoint_unavailable",
                f"native Tabnetics Diakrino checkpoint does not exist: {checkpoint}",
            )
    if backend == "tabpfn-candidate" and not _tabpfn_package_available():
        return (
            "skipped_optional_dependency_unavailable",
            "tabpfn optional dependency is unavailable in the current environment",
        )
    if backend == "tabiclv2-candidate":
        checkpoint = str(config.tabiclv2_checkpoint or "").strip()
        if not checkpoint:
            return (
                "skipped_checkpoint_not_configured",
                "TabICLv2 execution requires --tabiclv2-checkpoint resolved at the pinned revision",
            )
        if not Path(checkpoint).is_file():
            return (
                "skipped_checkpoint_unavailable",
                f"TabICLv2 checkpoint is not an existing file: {checkpoint}",
            )
        package_skip = _tabiclv2_package_skip()
        if package_skip is not None:
            return package_skip
    return None


def _run_sklearn_smoke(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    problem_type: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run a deterministic lightweight estimator for CI/schema smoke tests."""

    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if str(problem_type).strip().lower() == "regression":
        y_train_num = pd.to_numeric(pd.Series(y_train), errors="coerce").to_numpy(dtype=float)
        y_test_num = pd.to_numeric(pd.Series(y_test), errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y_train_num).all() or not np.isfinite(y_test_num).all():
            raise ValueError("sklearn-smoke regression requires finite numeric target values")
        if np.unique(y_train_num).size < 2:
            estimator = make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="mean"))
        else:
            estimator = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
        estimator.fit(X_train, y_train_num)
        pred = estimator.predict(X_test)
        mse = float(mean_squared_error(y_test_num, pred))
        try:
            r2 = float(r2_score(y_test_num, pred))
        except Exception:
            r2 = float("nan")
        return (
            {
                "mae": float(mean_absolute_error(y_test_num, pred)),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "r2": r2,
            },
            {
                "model_name": type(estimator[-1]).__name__,
                "n_targets": 1,
            },
        )

    encoder = LabelEncoder().fit(np.concatenate([np.asarray(y_train), np.asarray(y_test)]))
    y_train_enc = encoder.transform(np.asarray(y_train))
    y_test_enc = encoder.transform(np.asarray(y_test))
    n_classes = int(len(encoder.classes_))
    if np.unique(y_train_enc).size < 2:
        estimator = make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="most_frequent"))
    else:
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=int(seed)),
        )
    estimator.fit(X_train, y_train_enc)
    pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        raw_proba = estimator.predict_proba(X_test)
        proba = _align_proba(estimator[-1], raw_proba, n_classes=n_classes)
    else:
        proba = np.full((len(y_test_enc), n_classes), 1.0 / max(1, n_classes), dtype=float)
    proba = _normalize_class_probability_matrix(proba, n_classes=n_classes)

    values = {
        "accuracy": float(accuracy_score(y_test_enc, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_enc, pred)),
        "macro_f1": float(f1_score(y_test_enc, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_test_enc, proba, labels=np.arange(n_classes))) if n_classes > 1 else float("nan"),
        "roc_auc": float("nan"),
    }
    try:
        if n_classes == 2 and np.unique(y_test_enc).size == 2:
            values["roc_auc"] = float(roc_auc_score(y_test_enc, proba[:, 1]))
        elif n_classes > 2 and np.unique(y_test_enc).size > 1:
            values["roc_auc"] = float(roc_auc_score(y_test_enc, proba, multi_class="ovr", average="macro"))
    except Exception:
        values["roc_auc"] = float("nan")
    return values, {"model_name": type(estimator[-1]).__name__, "n_classes": n_classes}


def _result_values_and_meta(result: Any) -> tuple[dict[str, float], dict[str, Any]]:
    values = {
        "accuracy": float(result.accuracy),
        "balanced_accuracy": float(result.balanced_accuracy),
        "macro_f1": float(result.macro_f1),
        "log_loss": float(result.log_loss),
        "roc_auc": float(result.roc_auc),
    }
    meta = {
        "model_name": str(result.model_name),
        "selected_features": int(result.selected_features_count),
        "fs_time_sec": float(result.fs_time_sec),
        "dist_time_sec": float(result.dist_time_sec),
        "transform_time_sec": float(result.transform_time_sec),
        "resampling_context_fingerprint": str(
            result.resampling_context_fingerprint
        ),
        "fit_context_fingerprint": str(result.fit_context_fingerprint),
        "outer_split_fingerprint": str(result.outer_split_fingerprint),
        "resampling_policy": dict(result.resampling_policy or {}),
        "leakage_audit": dict(result.leakage_audit or {}),
    }
    return values, meta


def _build_fast_tabnetics_config(seed: int):
    from tabnetics.pipeline.pipeline import DFFSConfig

    return DFFSConfig(
        random_seed=int(seed),
        fs_fraction=1.0,
        n_final_features=20,
        n_jobs=1,
        apply_cdf_transform=False,
        use_rank_prefilter=False,
        prefilter_union_enabled=False,
        screening_enabled=False,
        folding_method="none",
        selection_strategy="legacy_voting",
        enabled_methods=("mutual_information", "anova_f"),
        fs_portfolio_size=2,
        fs_adaptive_portfolio_sizing_enabled=False,
        auto_router_enabled=False,
        classification_selection_mode="legacy",
        model_candidates=("lr",),
        include_elastic_net_model=False,
        include_rf_model=False,
        include_knn_model=False,
        include_svm_linear_model=False,
        include_dlda_model=False,
        include_nb_model=False,
    )


def _build_tabpfn_candidate_config(seed: int):
    from tabnetics.pipeline.pipeline import DFFSConfig

    return DFFSConfig(
        random_seed=int(seed),
        model_candidates=("tabpfn",),
        include_tabpfn_model=True,
        model_cv_runtime_max_candidates=1,
        include_elastic_net_model=False,
        include_rf_model=False,
        include_knn_model=False,
        include_svm_linear_model=False,
        include_dlda_model=False,
        include_nb_model=False,
        include_vote_ensemble_model=False,
        include_xgb_model=False,
        include_lgbm_model=False,
        include_extra_tree_model=False,
        include_catboost_model=False,
    )


def _run_tabnetics_with_config(
    config: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    dataset_name: str,
    seed: int,
    resampling_context: Any = None,
    resolved_outer_split: Any = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    from tabnetics.pipeline.pipeline import DistributionFeatureSelectionPipeline

    split_kwargs: dict[str, Any] = {}
    if resampling_context is not None or resolved_outer_split is not None:
        if resampling_context is None or resolved_outer_split is None:
            raise ValueError(
                "resampling_context and resolved_outer_split must be supplied together"
            )
        split_kwargs = {
            "split_indices_train": resolved_outer_split.primary.train_indices,
            "split_indices_test": resolved_outer_split.primary.test_indices,
            "resampling_context": resampling_context,
            "resolved_outer_split": resolved_outer_split,
        }
    result = DistributionFeatureSelectionPipeline(config).run_pre_split(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name=dataset_name,
        seed=int(seed),
        **split_kwargs,
    )
    return _result_values_and_meta(result)


def _encode_classification_targets(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder().fit(np.concatenate([np.asarray(y_train), np.asarray(y_test)]))
    y_train_enc = encoder.transform(np.asarray(y_train)).astype(np.int64, copy=False)
    y_test_enc = encoder.transform(np.asarray(y_test)).astype(np.int64, copy=False)
    labels = "|".join(str(label) for label in encoder.classes_.tolist())
    return y_train_enc, y_test_enc, labels, int(len(encoder.classes_))


def _run_tabnetics_current(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    dataset_name: str,
    seed: int,
    fast: bool = False,
    resampling_context: Any = None,
    resolved_outer_split: Any = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    from tabnetics.pipeline.pipeline import DFFSConfig

    config = (
        _build_fast_tabnetics_config(seed)
        if bool(fast)
        else DFFSConfig(
            random_seed=int(seed),
            model_cv_enable_svc_probability=True,
        )
    )
    y_train_enc, y_test_enc, class_labels, n_classes = _encode_classification_targets(y_train, y_test)
    resampling_kwargs: dict[str, Any] = {}
    if resampling_context is not None or resolved_outer_split is not None:
        resampling_kwargs = {
            "resampling_context": resampling_context,
            "resolved_outer_split": resolved_outer_split,
        }
    values, meta = _run_tabnetics_with_config(
        config,
        X_train,
        y_train_enc,
        X_test,
        y_test_enc,
        dataset_name=dataset_name,
        seed=seed,
        **resampling_kwargs,
    )
    meta = {**meta, "class_labels": class_labels, "n_classes": n_classes}
    return values, meta


def _run_tabpfn_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    dataset_name: str,
    seed: int,
    resampling_context: Any = None,
    resolved_outer_split: Any = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    y_train_enc, y_test_enc, class_labels, n_classes = _encode_classification_targets(y_train, y_test)
    resampling_kwargs: dict[str, Any] = {}
    if resampling_context is not None or resolved_outer_split is not None:
        resampling_kwargs = {
            "resampling_context": resampling_context,
            "resolved_outer_split": resolved_outer_split,
        }
    values, meta = _run_tabnetics_with_config(
        _build_tabpfn_candidate_config(seed),
        X_train,
        y_train_enc,
        X_test,
        y_test_enc,
        dataset_name=dataset_name,
        seed=seed,
        **resampling_kwargs,
    )
    meta = {**meta, "class_labels": class_labels, "n_classes": n_classes}
    return values, meta


def _run_tabiclv2_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    checkpoint: str | Path,
    device: str,
    seed: int,
    min_train_rows: int,
    max_train_rows: int,
    max_features: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    from tabnetics.classification.tabiclv2 import (
        TabICLv2Classifier,
        TabICLv2ContractError,
    )

    estimator = TabICLv2Classifier(
        checkpoint_path=checkpoint,
        device=device,
        min_train_rows=int(min_train_rows),
        max_train_rows=int(max_train_rows),
        max_features=int(max_features),
        random_state=int(seed),
    )
    estimator.fit(X_train, y_train)
    proba = estimator.predict_proba(X_test)
    classes = np.asarray(estimator.classes_)
    y_test_encoded = np.empty(len(y_test), dtype=np.int64)
    for index, label in enumerate(np.asarray(y_test).tolist()):
        matches = np.flatnonzero(classes == label)
        if len(matches) != 1:
            raise TabICLv2ContractError(
                "test labels must be present exactly once in the training-derived class order",
                status="failed_tabiclv2_contract",
            )
        y_test_encoded[index] = int(matches[0])
    values = _classification_values_from_proba(
        y_test_encoded,
        proba,
        n_classes=int(len(classes)),
    )
    metadata = dict(estimator.metadata_)
    metadata.update(
        {
            "model_name": "TabICLClassifier",
            "class_labels": "|".join(str(label) for label in classes.tolist()),
            "n_classes": int(len(classes)),
        }
    )
    return values, metadata


def _tabiclv2_skip_for_exception(exc: Exception) -> Optional[tuple[str, str]]:
    from tabnetics.classification.tabiclv2 import TabICLv2AvailabilityError

    if not isinstance(exc, TabICLv2AvailabilityError):
        return None
    status = str(getattr(exc, "status", "skipped_optional_dependency_unavailable"))
    return status, str(exc)


def _tabentics_diakrino_config_from_payload(config_cls: Any, payload: dict[str, Any]) -> Any:
    from tabnetics.classification.diakrino_native import tabentics_diakrino_config_from_payload

    return tabentics_diakrino_config_from_payload(config_cls, payload)


def _torch_load_checkpoint(torch_module: Any, checkpoint: str | Path, *, map_location: Any) -> Any:
    try:
        return torch_module.load(str(checkpoint), map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(str(checkpoint), map_location=map_location)


def _summarize_state_prefixes(keys: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        prefix = str(key).split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _tabentics_diakrino_checkpoint_format(payload: dict[str, Any], state: Any) -> str:
    raw = str(payload.get("checkpoint_format") or payload.get("format") or "").strip().lower()
    normalized = raw.replace("-", "_")
    if normalized in {"classifier", "fs_classifier", "tabentics_diakrino_fs_classifier"}:
        return "classifier"
    if normalized in {"fs_teacher", "teacher", "tabentics_diakrino_fs_teacher"}:
        return "fs_teacher"
    if isinstance(state, dict) and any(str(key).startswith("feature_selector.") for key in state):
        return "classifier"
    return "fs_teacher"


def _load_tabentics_diakrino_fs_teacher_state(
    model: Any,
    state: dict[Any, Any],
    *,
    checkpoint: str | Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    target_state = model.state_dict()
    mapped: dict[str, Any] = {}
    skipped_shape: list[str] = []
    loaded_source_keys: list[str] = []
    for source_key, value in state.items():
        source_name = str(source_key)
        target_key = source_name if source_name.startswith("feature_selector.") else f"feature_selector.{source_name}"
        if target_key not in target_state:
            continue
        if tuple(target_state[target_key].shape) != tuple(getattr(value, "shape", ())):
            skipped_shape.append(
                f"{source_name} -> {target_key}: "
                f"{tuple(getattr(value, 'shape', ()))} vs {tuple(target_state[target_key].shape)}"
            )
            continue
        mapped[target_key] = value
        loaded_source_keys.append(source_name)
    load_result = model.load_state_dict(mapped, strict=False)
    discarded = sorted(set(str(key) for key in state) - set(loaded_source_keys))
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_format": "fs_teacher",
        "loaded_count": int(len(loaded_source_keys)),
        "discarded_count": int(len(discarded)),
        "discarded_prefixes": _summarize_state_prefixes(discarded),
        "skipped_shape": skipped_shape,
        "missing_count": int(len(load_result.missing_keys)),
        "unexpected_count": int(len(load_result.unexpected_keys)),
        "missing_after_partial_load": list(load_result.missing_keys),
        "unexpected_after_partial_load": list(load_result.unexpected_keys),
        "source_epoch": payload.get("epoch"),
        "source_step": payload.get("step"),
    }


def _load_tabentics_diakrino_fs_classifier(
    checkpoint: str | Path,
    *,
    map_location: Any,
) -> tuple[Any, dict[str, Any]]:
    from tabnetics.classification.diakrino_native import load_tabentics_diakrino_fs_classifier

    return load_tabentics_diakrino_fs_classifier(checkpoint, map_location=map_location)


def _select_tabentics_diakrino_features(X_train: np.ndarray, max_features: int) -> np.ndarray:
    n_features = int(X_train.shape[1])
    budget = min(n_features, int(max(1, max_features)))
    if budget >= n_features:
        return np.arange(n_features, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        variance = np.nanvar(np.asarray(X_train, dtype=float), axis=0)
    variance = np.where(np.isfinite(variance), variance, -np.inf)
    order = np.argsort(-variance, kind="stable")
    return np.asarray(order[:budget], dtype=np.int64)


def _normalize_class_probability_matrix(proba: np.ndarray, *, n_classes: int) -> np.ndarray:
    n_classes_int = int(max(1, n_classes))
    values = np.asarray(proba, dtype=float)
    if values.ndim != 2 or values.shape[1] != n_classes_int:
        raise ValueError(
            "classification probability matrix must have shape "
            f"(n_samples, {n_classes_int}); got {tuple(values.shape)}"
        )
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    row_sums = values.sum(axis=1, keepdims=True)
    valid = np.isfinite(row_sums) & (row_sums > 0.0)
    normalized = np.divide(values, row_sums, out=np.zeros_like(values), where=valid)
    if not np.all(valid):
        normalized[~valid[:, 0], :] = 1.0 / float(n_classes_int)
    return normalized


def _classification_values_from_proba(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    n_classes: int,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    proba = _normalize_class_probability_matrix(proba, n_classes=int(n_classes))
    pred = np.argmax(proba, axis=1).astype(int)
    values = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "log_loss": (
            float(log_loss(y_true, proba, labels=np.arange(int(n_classes))))
            if int(n_classes) > 1
            else float("nan")
        ),
        "roc_auc": float("nan"),
    }
    try:
        if int(n_classes) == 2 and np.unique(y_true).size == 2:
            values["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        elif int(n_classes) > 2 and np.unique(y_true).size > 1:
            values["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        values["roc_auc"] = float("nan")
    return values


def _run_tabentics_diakrino_native(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    dataset_name: str,
    seed: int,
    checkpoint: str | Path,
    max_features: int,
    batch_size: int,
    allow_untrusted_checkpoint: bool = False,
    support_joint_serving_cache: bool = False,
    retry_cuda_oom_microbatch: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    from tabnetics.classification.diakrino_native import run_tabentics_diakrino_native

    return run_tabentics_diakrino_native(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_name=dataset_name,
        seed=seed,
        checkpoint=checkpoint,
        max_features=max_features,
        batch_size=batch_size,
        device="auto",
        allow_untrusted_checkpoint=allow_untrusted_checkpoint,
        support_joint_serving_cache=support_joint_serving_cache,
        retry_cuda_oom_microbatch=retry_cuda_oom_microbatch,
    )


def _base_row(
    *,
    spec: Any,
    split: BeyondArenaSplit,
    config: BeyondArenaLocalRunConfig,
    metric: str,
    status: str,
    execution_status: str,
    metric_value: Any = pd.NA,
    skip_reason: str = "",
    manifest_row: Optional[pd.Series] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    official_split = _value_or_default(
        manifest_row,
        "official_split_id",
        _value_or_default(manifest_row, "split", split.split_id),
    )
    payload = {
        "dataset_id": _value_or_default(manifest_row, "dataset_id", spec.beyondarena_id),
        "split_id": str(official_split),
        "local_dataset_id": spec.beyondarena_id,
        "local_split_id": split.split_id,
        "method": config.method,
        "metric": normalize_beyondarena_metric_name(metric),
        "metric_value": metric_value,
        "metric_error": _metric_error_value(metric, metric_value),
        "status": status,
        "execution_status": execution_status,
        "skip_reason": skip_reason,
        "origin": "tabnetics_local_beyondarena",
        "lower_is_better": metric_lower_is_better(metric),
        "task_type": _value_or_default(manifest_row, "task_type", spec.task_type),
        "problem_type": _value_or_default(manifest_row, "problem_type", spec.problem_type),
        "model_profile": config.model_profile,
        "execution_backend": config.backend,
        "seed": int(config.seed),
        "device": config.device,
        "execution_host": config.execution_host,
        "execution_lane": config.execution_lane,
        "manifest_shard_index": config.manifest_shard_index if config.manifest_shard_index is not None else pd.NA,
        "manifest_shard_count": config.manifest_shard_count if config.manifest_shard_count is not None else pd.NA,
        "allow_gpu_execution": bool(config.allow_gpu_execution),
        "artifact_revision": _value_or_default(manifest_row, "data_foundry_uri", spec.artifact_revision),
        "preprocessing_profile": BeyondArenaPreprocessingProfile().profile_id,
    }
    if extra:
        for key, value in extra.items():
            if key not in payload:
                payload[key] = value
    return payload


def _missing_manifest_artifact_rows(
    manifest: pd.DataFrame,
    *,
    config: BeyondArenaLocalRunConfig,
    matched_manifest_indexes: set[int],
) -> pd.DataFrame:
    """Build skipped rows for manifest tasks without local artifacts."""

    records: list[dict[str, Any]] = []
    for idx, row in manifest.iterrows():
        if int(idx) in matched_manifest_indexes:
            continue
        official_split = _value_or_default(
            row,
            "official_split_id",
            _value_or_default(row, "split", _value_or_default(row, "split_id", "")),
        )
        metric = str(_value_or_default(row, "metric", ""))
        records.append(
            {
                "dataset_id": _value_or_default(row, "dataset_id", ""),
                "split_id": str(official_split),
                "local_dataset_id": _value_or_default(row, "local_dataset_id", ""),
                "local_split_id": str(_value_or_default(row, "split_id", "")),
                "method": config.method,
                "metric": normalize_beyondarena_metric_name(metric),
                "metric_value": pd.NA,
                "status": "skipped",
                "execution_status": "skipped_missing_artifact",
                "skip_reason": "no materialized local DataFoundry artifact matched task manifest row",
                "origin": "tabnetics_local_beyondarena",
                "lower_is_better": metric_lower_is_better(metric),
                "task_type": _value_or_default(row, "task_type", ""),
                "problem_type": _value_or_default(row, "problem_type", ""),
                "size_tier": _value_or_default(row, "size_tier", ""),
                "dimensionality": _value_or_default(row, "dimensionality", ""),
                "has_text": _value_or_default(row, "has_text", pd.NA),
                "high_cardinality": _value_or_default(row, "high_cardinality", pd.NA),
                "model_profile": config.model_profile,
                "execution_backend": config.backend,
                "seed": int(config.seed),
                "device": config.device,
                "execution_host": config.execution_host,
                "execution_lane": config.execution_lane,
                "manifest_shard_index": (
                    config.manifest_shard_index if config.manifest_shard_index is not None else pd.NA
                ),
                "manifest_shard_count": (
                    config.manifest_shard_count if config.manifest_shard_count is not None else pd.NA
                ),
                "allow_gpu_execution": bool(config.allow_gpu_execution),
                "artifact_revision": _value_or_default(
                    row,
                    "data_foundry_uri",
                    _value_or_default(row, "artifact_revision", ""),
                ),
                "preprocessing_profile": BeyondArenaPreprocessingProfile().profile_id,
            }
        )
    return pd.DataFrame.from_records(records)


def run_local_beyondarena_artifact(
    artifact_dir: str | Path,
    *,
    config: Optional[BeyondArenaLocalRunConfig] = None,
    manifest_rows: Optional[pd.DataFrame] = None,
    result_journal: Optional[AtomicResultJournal] = None,
    collect_rows: bool = True,
) -> pd.DataFrame:
    """Run one artifact, committing each terminal split row before continuing."""

    artifact_path = Path(artifact_dir)
    cfg = _normalize_config(config or BeyondArenaLocalRunConfig(artifact_root=artifact_path))
    spec = load_beyondarena_spec(artifact_path)
    records: list[dict[str, Any]] = []

    def _emit(record: dict[str, Any]) -> None:
        if result_journal is not None:
            result_journal.commit(record)
        if collect_rows:
            records.append(record)

    if manifest_rows is None and cfg.task_manifest_csv is not None:
        manifest = _read_task_manifest(cfg.task_manifest_csv)
        manifest_rows = _manifest_rows_for_artifact(
            manifest,
            artifact_path,
            artifact_root=cfg.artifact_root,
            spec=spec,
        )
        if manifest_rows is not None and manifest_rows.empty:
            return pd.DataFrame()
    splits = load_beyondarena_splits(artifact_path).splits
    if cfg.max_splits_per_artifact is not None:
        splits = splits[: int(cfg.max_splits_per_artifact)]
    items = _execution_items(splits, manifest_rows)
    metric = spec.objective_metric or "roc_auc"
    if result_journal is not None:
        items = tuple(
            (split, row)
            for split, row in items
            if not result_journal.contains(
                _base_row(
                    spec=spec,
                    split=split,
                    config=cfg,
                    metric=str(_value_or_default(row, "metric", metric)),
                    status="pending",
                    execution_status="pending",
                    manifest_row=row,
                )
            )
        )
    if not items:
        return pd.DataFrame()

    try:
        loaded = load_beyondarena_dataset(artifact_path)
    except BeyondArenaUnavailableError as exc:
        for split, row in items:
            _emit(
                _base_row(
                    spec=spec,
                    split=split,
                    config=cfg,
                    metric=str(_value_or_default(row, "metric", metric)),
                    status="skipped",
                    execution_status="skipped_missing_artifact",
                    skip_reason=str(exc),
                    manifest_row=row,
                )
            )
        return pd.DataFrame.from_records(records)
    except Exception as exc:
        if cfg.on_error == "raise":
            raise
        for split, row in items:
            _emit(
                _base_row(
                    spec=spec,
                    split=split,
                    config=cfg,
                    metric=str(_value_or_default(row, "metric", metric)),
                    status="error",
                    execution_status="exception_loading_artifact",
                    skip_reason=f"{type(exc).__name__}: {exc}",
                    manifest_row=row,
                )
            )
        return pd.DataFrame.from_records(records)

    for split, row in items:
        row_metric = str(_value_or_default(row, "metric", metric))
        backend = cfg.backend.strip().lower()
        if split.source == "manifest_missing_local_split":
            _emit(
                _base_row(
                    spec=spec,
                    split=split,
                    config=cfg,
                    metric=row_metric,
                    status="skipped",
                    execution_status="skipped_missing_split",
                    skip_reason="task manifest split was not present in local DataFoundry split metadata",
                    manifest_row=row,
                )
            )
            continue
        if spec.problem_type != "classification" and backend != "sklearn-smoke":
            _emit(
                _base_row(
                    spec=spec,
                    split=split,
                    config=cfg,
                    metric=row_metric,
                    status="skipped",
                    execution_status="skipped_unsupported_regression",
                    skip_reason=(
                        "current tabnetics/DIAKRINO BeyondArena local execution path is classification-only; "
                        "use sklearn-smoke only for schema checks, not performance claims"
                    ),
                    manifest_row=row,
                )
            )
            continue

        leakage = validate_beyondarena_split_leakage(loaded.frame, spec, split)
        if not bool(leakage.get("ok", False)):
            _emit(
                _base_row(
                    spec=spec,
                    split=split,
                    config=cfg,
                    metric=row_metric,
                    status="skipped",
                    execution_status="skipped_leakage_guard_failed",
                    skip_reason=str(leakage.get("reason", "split leakage guard failed")),
                    manifest_row=row,
                    extra={"leakage_ok": False},
                )
            )
            continue

        try:
            from tabnetics.pipeline.resampling import resolve_holdout

            resampling_context = build_beyondarena_resampling_context(
                loaded.frame,
                spec,
                (split,),
            )
            resolved_outer_split = resolve_holdout(
                resampling_context,
                np.asarray(loaded.y),
                seed=int(cfg.seed),
                purpose="outer",
                supplied_split_id=str(split.split_id),
            )
            resampling_meta = {
                "resampling_context_fingerprint": str(
                    resampling_context.fingerprint
                ),
                "outer_split_fingerprint": str(
                    resolved_outer_split.primary.fingerprint
                ),
                "resampling_policy": resampling_context.policy.to_record(),
                "leakage_audit": resolved_outer_split.primary.audit.to_dict(),
            }
        except Exception as exc:
            _emit(
                _base_row(
                    spec=spec,
                    split=split,
                    config=cfg,
                    metric=row_metric,
                    status="skipped",
                    execution_status="skipped_resampling_contract_failed",
                    skip_reason=f"{type(exc).__name__}: {exc}",
                    manifest_row=row,
                    extra={"leakage_ok": False},
                )
            )
            continue

        backend_skip = _gpu_backend_skip(backend, cfg)
        if backend_skip is not None:
            execution_status, skip_reason = backend_skip
            _emit(
                _base_row(
                    spec=spec,
                    split=split,
                    config=cfg,
                    metric=row_metric,
                    status="skipped",
                    execution_status=execution_status,
                    skip_reason=skip_reason,
                    manifest_row=row,
                    extra={
                        **resampling_meta,
                        "leakage_ok": True,
                        "n_train": int(len(split.train_indices)),
                        "n_test": int(len(split.test_indices)),
                    },
                )
            )
            continue

        try:
            processed = apply_beyondarena_preprocessing(loaded.frame, spec, split=split)
            X_train, y_train, X_test, y_test = _split_arrays(processed.X, processed.y, split)
            if backend == "sklearn-smoke":
                values, meta = _run_sklearn_smoke(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    seed=cfg.seed,
                    problem_type=spec.problem_type,
                )
            elif backend == "tabnetics-current-fast":
                values, meta = _run_tabnetics_current(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    dataset_name=spec.beyondarena_id,
                    seed=cfg.seed,
                    fast=True,
                    resampling_context=resampling_context,
                    resolved_outer_split=resolved_outer_split,
                )
            elif backend == "tabnetics-current":
                values, meta = _run_tabnetics_current(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    dataset_name=spec.beyondarena_id,
                    seed=cfg.seed,
                    fast=False,
                    resampling_context=resampling_context,
                    resolved_outer_split=resolved_outer_split,
                )
            elif backend == "tabpfn-candidate":
                values, meta = _run_tabpfn_candidate(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    dataset_name=spec.beyondarena_id,
                    seed=cfg.seed,
                    resampling_context=resampling_context,
                    resolved_outer_split=resolved_outer_split,
                )
            elif backend == "tabiclv2-candidate":
                values, meta = _run_tabiclv2_candidate(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    checkpoint=cfg.tabiclv2_checkpoint,
                    device=cfg.tabiclv2_device,
                    seed=cfg.seed,
                    min_train_rows=cfg.tabiclv2_min_train_rows,
                    max_train_rows=cfg.tabiclv2_max_train_rows,
                    max_features=cfg.tabiclv2_max_features,
                )
            elif backend == "tabnetics-diakrino":
                values, meta = _run_tabentics_diakrino_native(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    dataset_name=spec.beyondarena_id,
                    seed=cfg.seed,
                    checkpoint=cfg.tabentics_diakrino_checkpoint,
                    max_features=int(cfg.tabentics_diakrino_max_features),
                    batch_size=int(cfg.tabentics_diakrino_batch_size),
                    allow_untrusted_checkpoint=bool(cfg.tabentics_diakrino_allow_untrusted_checkpoint),
                    support_joint_serving_cache=bool(cfg.tabentics_diakrino_support_joint_serving_cache),
                    retry_cuda_oom_microbatch=bool(cfg.tabentics_diakrino_retry_cuda_oom_microbatch),
                )
            else:
                raise ValueError(f"unknown BeyondArena local backend: {cfg.backend!r}")
            meta = {**resampling_meta, **meta}
            metric_value = _metric_from_mapping(row_metric, values)
            row_status = "ok" if np.isfinite(float(metric_value)) else "error"
            _emit(
                _base_row(
                    spec=spec,
                    split=split,
                    config=cfg,
                    metric=row_metric,
                    status=row_status,
                    execution_status="ok" if row_status == "ok" else "metric_unavailable",
                    metric_value=float(metric_value) if row_status == "ok" else pd.NA,
                    skip_reason="" if row_status == "ok" else f"metric {row_metric!r} unavailable",
                    manifest_row=row,
                    extra={
                        **processed.metadata,
                        **meta,
                        "leakage_ok": True,
                        "n_train": int(len(split.train_indices)),
                        "n_test": int(len(split.test_indices)),
                    },
                )
            )
        except Exception as exc:
            if cfg.on_error == "raise":
                raise
            availability_skip = (
                _tabiclv2_skip_for_exception(exc)
                if backend == "tabiclv2-candidate"
                else None
            )
            if availability_skip is not None:
                execution_status, skip_reason = availability_skip
                _emit(
                    _base_row(
                        spec=spec,
                        split=split,
                        config=cfg,
                        metric=row_metric,
                        status="skipped",
                        execution_status=execution_status,
                        skip_reason=skip_reason,
                        manifest_row=row,
                        extra={
                            **resampling_meta,
                            "leakage_ok": True,
                            "n_train": int(len(split.train_indices)),
                            "n_test": int(len(split.test_indices)),
                        },
                    )
                )
                continue
            _emit(
                _base_row(
                    spec=spec,
                    split=split,
                    config=cfg,
                    metric=row_metric,
                    status="error",
                    execution_status="exception",
                    skip_reason=f"{type(exc).__name__}: {exc}",
                    manifest_row=row,
                )
            )
    return pd.DataFrame.from_records(records)


def _bounded_executor_map(
    executor: ThreadPoolExecutor,
    function: Callable[[Any], pd.DataFrame],
    tasks: Iterable[Any],
    *,
    max_in_flight: int,
) -> Iterator[pd.DataFrame]:
    """Submit at most ``max_in_flight`` artifact frames at any time."""

    limit = int(max(1, max_in_flight))
    iterator = iter(tasks)
    pending: set[Future[pd.DataFrame]] = set()

    def _submit_one() -> bool:
        try:
            task = next(iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(function, task))
        return True

    while len(pending) < limit and _submit_one():
        pass
    while pending:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        pending.difference_update(completed)
        for future in completed:
            yield future.result()
            if len(pending) < limit:
                _submit_one()


def run_local_beyondarena_artifacts(
    *,
    config: BeyondArenaLocalRunConfig,
    result_journal: Optional[AtomicResultJournal] = None,
) -> pd.DataFrame:
    """Run local artifacts with bounded scheduling and optional row journaling."""

    config = _normalize_config(config)
    manifest = _apply_manifest_shard(
        _read_task_manifest(config.task_manifest_csv),
        shard_index=config.manifest_shard_index,
        shard_count=config.manifest_shard_count,
    )
    tasks: list[tuple[Path, Optional[pd.DataFrame]]] = []
    matched_manifest_indexes: set[int] = set()
    for artifact_dir in _artifact_dirs(config.artifact_root, max_artifacts=config.max_artifacts):
        spec = load_beyondarena_spec(artifact_dir)
        manifest_rows = _manifest_rows_for_artifact(
            manifest,
            artifact_dir,
            artifact_root=config.artifact_root,
            spec=spec,
        )
        if manifest is not None and (manifest_rows is None or manifest_rows.empty):
            continue
        if manifest_rows is not None and "_manifest_row_index" in manifest_rows.columns:
            matched_manifest_indexes.update(int(idx) for idx in manifest_rows["_manifest_row_index"].tolist())
        tasks.append((artifact_dir, manifest_rows))

    def _run_task(task: tuple[Path, Optional[pd.DataFrame]]) -> pd.DataFrame:
        artifact_dir, manifest_rows = task
        return run_local_beyondarena_artifact(
            artifact_dir,
            config=config,
            manifest_rows=manifest_rows,
            result_journal=result_journal,
            collect_rows=result_journal is None,
        )

    frames: list[pd.DataFrame] = []
    if tasks and int(config.max_workers) > 1:
        worker_count = min(int(config.max_workers), int(config.max_in_flight_artifacts or 1))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for frame in _bounded_executor_map(
                executor,
                _run_task,
                tasks,
                max_in_flight=int(config.max_in_flight_artifacts or 1),
            ):
                if result_journal is None:
                    frames.append(frame)
    else:
        for task in tasks:
            frame = _run_task(task)
            if result_journal is None:
                frames.append(frame)

    if manifest is not None and not manifest.empty:
        missing = _missing_manifest_artifact_rows(
            manifest,
            config=config,
            matched_manifest_indexes=matched_manifest_indexes,
        )
        if not missing.empty:
            if result_journal is None:
                frames.append(missing)
            else:
                for record in missing.to_dict(orient="records"):
                    if not result_journal.contains(record):
                        result_journal.commit(record)
    if result_journal is not None:
        return result_journal.to_frame()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def merge_beyondarena_local_result_shards(shard_glob: str | Path) -> pd.DataFrame:
    """Merge disjoint local-result shard CSVs into one comparison input frame."""

    paths = tuple(sorted(Path(path) for path in glob.glob(str(shard_glob))))
    if not paths:
        raise ValueError(f"no BeyondArena local result shard CSVs matched: {shard_glob}")
    frames = [pd.read_csv(path) for path in paths]
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    key_columns = [
        column
        for column in ("dataset_id", "split_id", "method", "metric", "seed")
        if column in rows.columns
    ]
    if key_columns and not rows.empty:
        duplicated = rows.duplicated(key_columns, keep=False)
        if bool(duplicated.any()):
            sample = rows.loc[duplicated, key_columns].head(5).to_dict(orient="records")
            raise ValueError(f"BeyondArena local result shards overlap on comparison keys: {sample}")
    return rows


@contextmanager
def _claim_out_csv(path: Path) -> Iterator[Path]:
    """Atomically claim one output CSV path for this process."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    fd: Optional[int] = None
    claim_metadata = _new_lock_metadata()
    claim_inode: Optional[tuple[int, int]] = None
    try:
        fd = _open_out_csv_lock(lock_path)
    except FileExistsError as exc:
        raise RuntimeError(
            f"BeyondArena output is already claimed: {path} (lock: {lock_path})"
        ) from exc
    try:
        stat = os.fstat(fd)
        claim_inode = (int(stat.st_dev), int(stat.st_ino))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            for key, value in claim_metadata.items():
                handle.write(f"{key}={value}\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield lock_path
    finally:
        if fd is not None:
            os.close(fd)
        if claim_inode is not None and _lock_is_owned_by_claim(
            lock_path,
            claim_metadata=claim_metadata,
            claim_inode=claim_inode,
        ):
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _lock_metadata(lock_path: Path) -> dict[str, str]:
    try:
        payload = lock_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    metadata: dict[str, str] = {}
    for line in payload.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition("=")
        normalized_key = key.strip()
        if not sep or not normalized_key or normalized_key in metadata:
            return {}
        metadata[normalized_key] = value.strip()
    return metadata


def _lock_owner_pid(lock_path: Path) -> Optional[int]:
    try:
        pid = int(_lock_metadata(lock_path).get("pid", ""))
    except ValueError:
        return None
    return pid if pid > 0 else None


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _process_start_time(pid: int) -> str:
    try:
        payload = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except OSError:
        return ""
    _, separator, suffix = payload.rpartition(")")
    if not separator:
        return ""
    fields = suffix.split()
    return fields[19] if len(fields) > 19 else ""


def _new_lock_metadata() -> dict[str, str]:
    pid = os.getpid()
    return {
        "version": "2",
        "claim_id": uuid.uuid4().hex,
        "hostname": socket.gethostname(),
        "boot_id": _boot_id(),
        "pid": str(pid),
        "pid_start_time": _process_start_time(pid),
    }


def _lock_inode(lock_path: Path) -> Optional[tuple[int, int]]:
    try:
        stat = lock_path.stat(follow_symlinks=False)
    except OSError:
        return None
    return int(stat.st_dev), int(stat.st_ino)


def _lock_is_owned_by_claim(
    lock_path: Path,
    *,
    claim_metadata: dict[str, str],
    claim_inode: tuple[int, int],
) -> bool:
    metadata = _lock_metadata(lock_path)
    return bool(
        metadata.get("claim_id")
        and metadata.get("claim_id") == claim_metadata.get("claim_id")
        and _lock_inode(lock_path) == claim_inode
    )


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _positive_integer(value: str) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(str(value).strip())


def _valid_v2_lock_metadata(metadata: dict[str, str]) -> bool:
    return bool(
        set(metadata) == _LOCK_V2_FIELDS
        and metadata.get("version") == "2"
        and _valid_uuid(metadata.get("claim_id", ""))
        and str(metadata.get("hostname", "")).strip()
        and _valid_uuid(metadata.get("boot_id", ""))
        and _positive_integer(metadata.get("pid", ""))
        and _positive_integer(metadata.get("pid_start_time", ""))
    )


def _lock_is_safely_reclaimable(lock_path: Path) -> bool:
    """Reclaim only a versioned lock proven stale on this exact host.

    Legacy, malformed, and foreign-host locks fail closed because a PID lookup
    on the current host cannot prove that their owner is gone.
    """

    metadata = _lock_metadata(lock_path)
    if not _valid_v2_lock_metadata(metadata):
        return False
    if metadata["hostname"] != socket.gethostname():
        return False
    owner_pid = _lock_owner_pid(lock_path)
    if owner_pid is None:
        return False
    recorded_boot_id = metadata.get("boot_id", "")
    current_boot_id = _boot_id()
    if recorded_boot_id and current_boot_id and recorded_boot_id != current_boot_id:
        return True
    if not _pid_is_running(owner_pid):
        return True
    recorded_start = metadata.get("pid_start_time", "")
    current_start = _process_start_time(owner_pid)
    return bool(recorded_start and current_start and recorded_start != current_start)


def _open_out_csv_lock(lock_path: Path) -> int:
    """Open an exclusive lock, reclaiming only proven same-host leftovers."""

    try:
        return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        stale_inode = _lock_inode(lock_path)
        if not _lock_is_safely_reclaimable(lock_path):
            raise
        if stale_inode is None or _lock_inode(lock_path) != stale_inode:
            raise FileExistsError(lock_path)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)


def _atomic_write_csv(rows: pd.DataFrame, path: Path) -> None:
    """Replace a compatibility CSV only after its bytes are flushed to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.tmp.",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            rows.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument(
        "--merge-shard-glob",
        help="Merge already-written local result shard CSVs into --out-csv and exit.",
    )
    parser.add_argument(
        "--task-manifest-csv",
        type=Path,
        help="Optional beyondarena_task_manifest.csv; when set, output rows use official dataset/split IDs.",
    )
    parser.add_argument(
        "--backend",
        choices=(
            "tabnetics-current",
            "tabnetics-current-fast",
            "tabpfn-candidate",
            "tabiclv2-candidate",
            "tabnetics-diakrino",
            "sklearn-smoke",
        ),
        default="tabnetics-current",
    )
    parser.add_argument("--method", default="")
    parser.add_argument("--model-profile", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="Execution device metadata to write into result rows.")
    parser.add_argument("--execution-host", default="", help="Execution host metadata to write into result rows.")
    parser.add_argument("--execution-lane", default="", help="Execution lane metadata to write into result rows.")
    parser.add_argument("--max-artifacts", type=int)
    parser.add_argument("--max-splits-per-artifact", type=int)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum concurrent local artifact workers for CPU/materialized-artifact execution.",
    )
    parser.add_argument(
        "--max-in-flight-artifacts",
        type=int,
        help=(
            "Hard cap on running plus queued artifact frames; defaults to --max-workers and "
            "may be lowered to enforce a memory-derived concurrency limit."
        ),
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        help=(
            "Atomic row journal used for interruption-safe resume; defaults to "
            "<out-csv>.journal and must be removed explicitly to start a fresh run."
        ),
    )
    parser.add_argument(
        "--manifest-shard-index",
        type=int,
        help="Zero-based shard index for deterministic modulo partitioning of --task-manifest-csv rows.",
    )
    parser.add_argument(
        "--manifest-shard-count",
        type=int,
        help="Total shard count for deterministic modulo partitioning of --task-manifest-csv rows.",
    )
    parser.add_argument(
        "--allow-gpu-execution",
        action="store_true",
        help=(
            "Allow GPU-required BeyondArena backends after public-gpu-host access/capacity has been revalidated; "
            "without this flag they emit deferred skip rows."
        ),
    )
    parser.add_argument(
        "--tabiclv2-checkpoint",
        default="",
        help=(
            "Explicit local tabicl-classifier-v2-20260212.ckpt resolved from jingang/TabICL "
            "at the repository-pinned revision; automatic downloads are disabled."
        ),
    )
    parser.add_argument(
        "--tabiclv2-device",
        default="cuda",
        help="Explicit CUDA device for the opt-in TabICLv2 comparator (for example cuda or cuda:0).",
    )
    parser.add_argument(
        "--tabiclv2-min-train-rows",
        type=int,
        default=300,
        help="Fail-closed lower training-row limit for the published TabICLv2 regime.",
    )
    parser.add_argument(
        "--tabiclv2-max-train-rows",
        type=int,
        default=100_000,
        help="Fail-closed upper training-row limit for the published TabICLv2 regime.",
    )
    parser.add_argument(
        "--tabiclv2-max-features",
        type=int,
        default=2_000,
        help="Fail-closed feature limit for the published TabICLv2 regime.",
    )
    parser.add_argument(
        "--tabnetics-diakrino-checkpoint",
        dest="tabentics_diakrino_checkpoint",
        default="",
        help="Native Tabnetics Diakrino FS-classifier checkpoint path required by --backend tabnetics-diakrino.",
    )
    parser.add_argument(
        "--tabnetics-diakrino-max-features",
        dest="tabentics_diakrino_max_features",
        type=int,
        default=1024,
        help="Maximum features passed to the native Tabnetics Diakrino FS-classifier adapter.",
    )
    parser.add_argument(
        "--tabnetics-diakrino-batch-size",
        dest="tabentics_diakrino_batch_size",
        type=int,
        default=128,
        help="Query batch size for native Tabnetics Diakrino FS-classifier inference.",
    )
    parser.add_argument(
        "--tabnetics-diakrino-allow-cpu",
        dest="tabentics_diakrino_allow_cpu",
        action="store_true",
        help=(
            "Opt-in CPU execution lane for the native tabnetics-diakrino backend "
            "(public_cpu_host_1/public_cpu_host_2 CPU validation); other GPU backends remain gated."
        ),
    )
    parser.add_argument(
        "--tabnetics-diakrino-support-joint-serving-cache",
        dest="tabentics_diakrino_support_joint_serving_cache",
        action="store_true",
        help="Opt in to eval-only support-context caching for native Diakrino inference.",
    )
    parser.add_argument(
        "--tabnetics-diakrino-retry-cuda-oom-microbatch",
        dest="tabentics_diakrino_retry_cuda_oom_microbatch",
        action="store_true",
        help="Retry only a failed native Diakrino query chunk at microbatch 1 after CUDA OOM.",
    )
    parser.add_argument(
        "--tabnetics-diakrino-allow-untrusted-checkpoint",
        dest="tabentics_diakrino_allow_untrusted_checkpoint",
        action="store_true",
        help=(
            "Explicit ablation/diagnostic override: run mid-training snapshots "
            "without a passing head_trust_record; stamped into result-row meta."
        ),
    )
    parser.add_argument("--on-error", choices=("row", "raise"), default="row")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if str(args.merge_shard_glob or "").strip():
        with _claim_out_csv(args.out_csv):
            rows = merge_beyondarena_local_result_shards(str(args.merge_shard_glob))
            _atomic_write_csv(rows, args.out_csv)
        status_counts = rows["status"].value_counts(dropna=False).to_dict() if "status" in rows.columns else {}
        payload = {
            "mode": "merge_shards",
            "merge_shard_glob": str(args.merge_shard_glob),
            "out_csv": str(args.out_csv),
            "row_count": int(len(rows)),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        }
        print(payload)
        return 0
    if args.artifact_root is None:
        raise ValueError("--artifact-root is required unless --merge-shard-glob is set")
    backend = str(args.backend)
    config = BeyondArenaLocalRunConfig(
        artifact_root=args.artifact_root,
        task_manifest_csv=args.task_manifest_csv,
        backend=backend,
        method=str(args.method),
        model_profile=str(args.model_profile),
        seed=int(args.seed),
        device=str(args.device),
        execution_host=str(args.execution_host),
        execution_lane=str(args.execution_lane),
        max_artifacts=args.max_artifacts,
        max_splits_per_artifact=args.max_splits_per_artifact,
        max_workers=int(args.max_workers),
        max_in_flight_artifacts=args.max_in_flight_artifacts,
        manifest_shard_index=args.manifest_shard_index,
        manifest_shard_count=args.manifest_shard_count,
        allow_gpu_execution=bool(args.allow_gpu_execution),
        tabiclv2_checkpoint=str(args.tabiclv2_checkpoint),
        tabiclv2_device=str(args.tabiclv2_device),
        tabiclv2_min_train_rows=int(args.tabiclv2_min_train_rows),
        tabiclv2_max_train_rows=int(args.tabiclv2_max_train_rows),
        tabiclv2_max_features=int(args.tabiclv2_max_features),
        tabentics_diakrino_checkpoint=str(args.tabentics_diakrino_checkpoint),
        tabentics_diakrino_max_features=int(args.tabentics_diakrino_max_features),
        tabentics_diakrino_batch_size=int(args.tabentics_diakrino_batch_size),
        tabentics_diakrino_support_joint_serving_cache=bool(
            args.tabentics_diakrino_support_joint_serving_cache
        ),
        tabentics_diakrino_retry_cuda_oom_microbatch=bool(
            args.tabentics_diakrino_retry_cuda_oom_microbatch
        ),
        tabentics_diakrino_allow_cpu=bool(args.tabentics_diakrino_allow_cpu),
        tabentics_diakrino_allow_untrusted_checkpoint=bool(args.tabentics_diakrino_allow_untrusted_checkpoint),
        on_error=str(args.on_error),
    )
    config = _normalize_config(config)
    journal_dir = args.journal_dir or args.out_csv.with_name(f"{args.out_csv.name}.journal")
    with _claim_out_csv(args.out_csv):
        journal = AtomicResultJournal(
            journal_dir,
            key_fields=_BEYONDARENA_RESULT_KEY_FIELDS,
            context=_result_journal_context(config),
        )
        resumed_row_count = len(journal)
        rows = run_local_beyondarena_artifacts(config=config, result_journal=journal)
        _atomic_write_csv(rows, args.out_csv)
    status_counts = rows["status"].value_counts(dropna=False).to_dict() if "status" in rows.columns else {}
    payload = {
        **asdict(config),
        "artifact_root": str(config.artifact_root),
        "task_manifest_csv": str(config.task_manifest_csv) if config.task_manifest_csv is not None else "",
        "out_csv": str(args.out_csv),
        "journal_dir": str(journal_dir),
        "resumed_row_count": int(resumed_row_count),
        "row_count": int(len(rows)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
    }
    print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BeyondArenaLocalRunConfig",
    "merge_beyondarena_local_result_shards",
    "run_local_beyondarena_artifact",
    "run_local_beyondarena_artifacts",
    "main",
]

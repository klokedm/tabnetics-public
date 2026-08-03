"""Versioned identities for claim-bearing DIAKRINO feature-selection sidecars.

The historical DIAKRINO sidecar format was intentionally permissive: a parquet path was
enough to load scores.  That remains useful for diagnostics, but it cannot establish
that the scores came from the checkpoint, inputs, split, and feature order used by a
validation row.  This module defines the stronger producer/consumer contract used by
the vnext-C campaign.

All derived digests are domain separated by an algorithm and schema-version label.
Array identities bind dtype, shape, and exact C-order bytes.  Artifact identities bind
the exact file bytes and length.  ``DiakrinoExpectedIdentity`` is deliberately small and
JSON-native so a benchmark runner can reconstruct it from its materialized arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DIGEST_ALGORITHM = "sha256"
DIAKRINO_SIDECAR_MANIFEST_SCHEMA_VERSION = "diakrino_sidecar_manifest_v2"
DIAKRINO_SIDECAR_IDENTITY_SCHEMA_VERSION = "diakrino_sidecar_identity_v1"
DIAKRINO_CANONICAL_ARRAY_SCHEMA_VERSION = "diakrino_canonical_array_v1"
DIAKRINO_FEATURE_ORDER_SCHEMA_VERSION = "diakrino_feature_order_v1"
DIAKRINO_CANONICAL_JSON_SCHEMA_VERSION = "diakrino_canonical_json_v1"
DIAKRINO_ARTIFACT_IDENTITY_SCHEMA_VERSION = "diakrino_artifact_identity_v1"
DIAKRINO_PRODUCER_SOURCE_MANIFEST_SCHEMA_VERSION = "diakrino_producer_source_manifest_v1"
DIAKRINO_REQUIRED_CALIBRATION_MODE = "chunk_zscore"

IDENTITY_FRAME_COLUMNS: tuple[str, ...] = (
    "diakrino_identity_schema_version",
    "diakrino_identity_binding_sha256",
    "diakrino_identity_dataset_id",
    "diakrino_identity_checkpoint_sha256",
    "diakrino_identity_model_config_sha256",
    "diakrino_identity_producer_source_sha256",
    "diakrino_identity_source_manifest_sha256",
    "diakrino_identity_input_x_sha256",
    "diakrino_identity_input_y_sha256",
    "diakrino_identity_support_indices_sha256",
    "diakrino_identity_query_indices_sha256",
    "diakrino_identity_feature_order_sha256",
    "diakrino_identity_calibration_mode",
    "diakrino_identity_sidecar_schema_version",
)

__tabnetics_execution_isolated_state__ = {
    "IDENTITY_FRAME_COLUMNS": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}


class DiakrinoSidecarIdentityError(ValueError):
    """Raised when a claim-bearing sidecar identity is absent or mismatched."""


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise DiakrinoSidecarIdentityError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 of exact bytes (artifact-byte identity)."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "+inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return {"__type__": type(value).__qualname__, "__repr__": repr(value)}


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _json_native(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(
    payload: Any,
    *,
    payload_schema_version: str = DIAKRINO_CANONICAL_JSON_SCHEMA_VERSION,
) -> str:
    """Hash canonical JSON with explicit algorithm/schema domain separation."""

    envelope = {
        "digest_algorithm": DIGEST_ALGORITHM,
        "payload_schema_version": str(payload_schema_version),
        "payload": payload,
    }
    return sha256_bytes(_canonical_json_bytes(envelope))


def build_producer_source_manifest(project_root: str | Path) -> dict[str, Any]:
    """Hash the declared Python source closure used to produce DIAKRINO sidecars.

    A single probe-script digest is insufficient because model geometry, episode
    construction, schema helpers, and trust/identity serialization live in imported
    modules.  This explicit closure includes every Python source under
    ``experimental`` and ``scripts/training`` plus the core DIAKRINO runtime helpers.
    """

    root = Path(project_root).resolve()
    candidates: set[Path] = set()
    for relative_root in ("experimental", "scripts/training"):
        folder = root / relative_root
        if folder.is_dir():
            candidates.update(path for path in folder.rglob("*.py") if path.is_file())
    for relative_path in (
        "core/src/tabnetics/classification/tabentics_diakrino_fs_teacher.py",
        "core/src/tabnetics/feature_selection/diakrino_identity.py",
        "core/src/tabnetics/feature_selection/diakrino_closeout.py",
        "core/src/tabnetics/feature_selection/diakrino_sidecar.py",
        "core/src/tabnetics/feature_selection/diakrino_trust.py",
        "core/src/tabnetics/feature_selection/diakrino_views.py",
    ):
        path = root / relative_path
        if path.is_file():
            candidates.add(path)
    if not candidates:
        raise DiakrinoSidecarIdentityError(
            f"producer source closure is empty under project root {root}"
        )
    files = []
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        data = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "digest_algorithm": DIGEST_ALGORITHM,
                "size_bytes": int(len(data)),
                "sha256": sha256_bytes(data),
            }
        )
    payload = {
        "schema_version": DIAKRINO_PRODUCER_SOURCE_MANIFEST_SCHEMA_VERSION,
        "digest_algorithm": DIGEST_ALGORITHM,
        "files": files,
    }
    payload["manifest_sha256"] = canonical_json_sha256(
        payload,
        payload_schema_version=DIAKRINO_PRODUCER_SOURCE_MANIFEST_SCHEMA_VERSION,
    )
    return payload


def validate_producer_source_manifest(payload: Mapping[str, Any]) -> str:
    """Validate a persisted source-closure payload and return its manifest digest."""

    if str(payload.get("schema_version") or "") != DIAKRINO_PRODUCER_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise DiakrinoSidecarIdentityError("unsupported producer source-manifest schema")
    if str(payload.get("digest_algorithm") or "") != DIGEST_ALGORITHM:
        raise DiakrinoSidecarIdentityError("unsupported producer source-manifest digest algorithm")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise DiakrinoSidecarIdentityError("producer source manifest has no files")
    paths: list[str] = []
    for item in files:
        if not isinstance(item, Mapping):
            raise DiakrinoSidecarIdentityError("producer source-manifest file entry is malformed")
        path = str(item.get("path") or "").strip()
        if not path or path.startswith("/") or ".." in Path(path).parts:
            raise DiakrinoSidecarIdentityError("producer source-manifest path is malformed")
        if str(item.get("digest_algorithm") or "") != DIGEST_ALGORITHM:
            raise DiakrinoSidecarIdentityError("producer source file uses an unsupported digest")
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DiakrinoSidecarIdentityError("producer source file size is malformed")
        _require_sha256(item.get("sha256"), field="producer_source.files.sha256")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise DiakrinoSidecarIdentityError("producer source-manifest paths are duplicated or unordered")
    declared = _require_sha256(
        payload.get("manifest_sha256"), field="producer_source.manifest_sha256"
    )
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    observed = canonical_json_sha256(
        unsigned,
        payload_schema_version=DIAKRINO_PRODUCER_SOURCE_MANIFEST_SCHEMA_VERSION,
    )
    if observed != declared:
        raise DiakrinoSidecarIdentityError("producer source-manifest digest is inconsistent")
    return declared


@dataclass(frozen=True)
class CanonicalArrayIdentity:
    """Identity of dtype, shape, and exact C-order array bytes."""

    dtype: str
    shape: tuple[int, ...]
    c_order_sha256: str
    schema_version: str = DIAKRINO_CANONICAL_ARRAY_SCHEMA_VERSION
    digest_algorithm: str = DIGEST_ALGORITHM

    @classmethod
    def from_array(cls, values: Any) -> "CanonicalArrayIdentity":
        array = np.asarray(values)
        if array.dtype.hasobject:
            raise DiakrinoSidecarIdentityError("object arrays have no portable canonical byte identity")
        contiguous = np.ascontiguousarray(array)
        metadata = {
            "digest_algorithm": DIGEST_ALGORITHM,
            "schema_version": DIAKRINO_CANONICAL_ARRAY_SCHEMA_VERSION,
            "dtype": contiguous.dtype.str,
            "shape": [int(value) for value in contiguous.shape],
            "byte_order": "C",
        }
        digest = hashlib.sha256()
        digest.update(_canonical_json_bytes(metadata))
        digest.update(b"\n")
        digest.update(contiguous.tobytes(order="C"))
        return cls(
            dtype=str(contiguous.dtype.str),
            shape=tuple(int(value) for value in contiguous.shape),
            c_order_sha256=digest.hexdigest(),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalArrayIdentity":
        if str(payload.get("schema_version") or "") != DIAKRINO_CANONICAL_ARRAY_SCHEMA_VERSION:
            raise DiakrinoSidecarIdentityError("unsupported canonical-array identity schema")
        if str(payload.get("digest_algorithm") or "") != DIGEST_ALGORITHM:
            raise DiakrinoSidecarIdentityError("unsupported canonical-array digest algorithm")
        raw_shape = payload.get("shape")
        if not isinstance(raw_shape, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_shape
        ):
            raise DiakrinoSidecarIdentityError("canonical-array shape is malformed")
        dtype = str(payload.get("dtype") or "")
        try:
            np.dtype(dtype)
        except Exception as exc:
            raise DiakrinoSidecarIdentityError("canonical-array dtype is malformed") from exc
        return cls(
            dtype=dtype,
            shape=tuple(int(value) for value in raw_shape),
            c_order_sha256=_require_sha256(
                payload.get("c_order_sha256"), field="canonical_array.c_order_sha256"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "digest_algorithm": self.digest_algorithm,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "byte_order": "C",
            "c_order_sha256": self.c_order_sha256,
        }


def exact_int64_vector(values: Any, *, label: str) -> np.ndarray:
    """Return an exact int64 vector without bool/fractional/overflow coercion."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise DiakrinoSidecarIdentityError(f"{label} must be one-dimensional")
    if np.issubdtype(array.dtype, np.bool_):
        raise DiakrinoSidecarIdentityError(f"{label} must not contain booleans")
    if np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.unsignedinteger) and array.size and np.any(
            array > np.iinfo(np.int64).max
        ):
            raise DiakrinoSidecarIdentityError(f"{label} contains int64 overflow")
        return np.ascontiguousarray(array.astype(np.int64, copy=False))
    if np.issubdtype(array.dtype, np.floating):
        numeric = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise DiakrinoSidecarIdentityError(f"{label} contains non-finite values")
        if not np.all(numeric == np.trunc(numeric)):
            raise DiakrinoSidecarIdentityError(f"{label} contains fractional values")
        if numeric.size and (
            np.any(numeric < -(2**63)) or np.any(numeric >= 2**63)
        ):
            raise DiakrinoSidecarIdentityError(f"{label} contains int64 overflow")
        return np.ascontiguousarray(numeric.astype(np.int64))
    raise DiakrinoSidecarIdentityError(f"{label} must contain exact numeric integers")


def exact_int64_scalar(
    value: Any,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    """Return one typed int64 value without lossy JSON/Python coercion."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise DiakrinoSidecarIdentityError(f"{label} must be an exact integer")
    integer = int(value)
    if integer < np.iinfo(np.int64).min or integer > np.iinfo(np.int64).max:
        raise DiakrinoSidecarIdentityError(f"{label} is outside int64 range")
    if minimum is not None and integer < int(minimum):
        raise DiakrinoSidecarIdentityError(f"{label} must be at least {int(minimum)}")
    return integer


def canonical_index_identity(values: Sequence[int] | np.ndarray) -> CanonicalArrayIdentity:
    array = exact_int64_vector(values, label="support/query indices")
    return CanonicalArrayIdentity.from_array(array)


def canonicalize_diakrino_inputs(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Apply the emitter's shared, explicit input representation.

    DIAKRINO emission consumes float32 features and dense first-occurrence-independent
    ``np.unique`` class ids.  Canonical runners often materialize float64 arrays;
    normalizing here makes their independently reconstructed identity describe the
    exact bytes the emitter consumed while retaining dtype in the digest contract.
    """

    X_array = np.asarray(X, dtype=np.float32)
    y_raw = np.asarray(y)
    if X_array.ndim != 2:
        raise DiakrinoSidecarIdentityError("X must be two-dimensional")
    if y_raw.ndim != 1 or y_raw.shape[0] != X_array.shape[0]:
        raise DiakrinoSidecarIdentityError("y must be one-dimensional and row-aligned with X")
    if y_raw.size == 0:
        raise DiakrinoSidecarIdentityError("y must contain at least one row")
    try:
        _, y_array = np.unique(y_raw, return_inverse=True)
    except Exception as exc:
        raise DiakrinoSidecarIdentityError("y labels cannot be canonically remapped") from exc
    return (
        np.ascontiguousarray(X_array, dtype=np.float32),
        np.ascontiguousarray(y_array, dtype=np.int64),
    )


def _typed_identifier(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"type": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise DiakrinoSidecarIdentityError("feature identifiers must be finite")
        return {"type": "float", "value": numeric}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    return {"type": type(value).__qualname__, "value": repr(value)}


@dataclass(frozen=True)
class FeatureOrderIdentity:
    """Identity of ordered original feature identifiers and positions."""

    count: int
    sha256: str
    schema_version: str = DIAKRINO_FEATURE_ORDER_SCHEMA_VERSION
    digest_algorithm: str = DIGEST_ALGORITHM

    @classmethod
    def from_values(
        cls,
        feature_order: Sequence[Any] | None,
        *,
        n_features: int,
    ) -> "FeatureOrderIdentity":
        count = int(n_features)
        if count < 0:
            raise DiakrinoSidecarIdentityError("n_features must be non-negative")
        values: list[Any]
        if feature_order is None:
            values = list(range(count))
        else:
            values = list(feature_order)
        if len(values) != count:
            raise DiakrinoSidecarIdentityError(
                f"feature order length mismatch: expected={count} observed={len(values)}"
            )
        payload = {
            "digest_algorithm": DIGEST_ALGORITHM,
            "schema_version": DIAKRINO_FEATURE_ORDER_SCHEMA_VERSION,
            "entries": [
                {"original_position": int(position), "identifier": _typed_identifier(identifier)}
                for position, identifier in enumerate(values)
            ],
        }
        return cls(count=count, sha256=sha256_bytes(_canonical_json_bytes(payload)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureOrderIdentity":
        if str(payload.get("schema_version") or "") != DIAKRINO_FEATURE_ORDER_SCHEMA_VERSION:
            raise DiakrinoSidecarIdentityError("unsupported feature-order identity schema")
        if str(payload.get("digest_algorithm") or "") != DIGEST_ALGORITHM:
            raise DiakrinoSidecarIdentityError("unsupported feature-order digest algorithm")
        count = payload.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise DiakrinoSidecarIdentityError("feature-order count is malformed")
        return cls(
            count=int(count),
            sha256=_require_sha256(payload.get("sha256"), field="feature_order.sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "digest_algorithm": self.digest_algorithm,
            "count": int(self.count),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArtifactIdentity:
    """Exact byte identity of one persisted artifact."""

    path: str
    size_bytes: int
    sha256: str
    schema_version: str = DIAKRINO_ARTIFACT_IDENTITY_SCHEMA_VERSION
    digest_algorithm: str = DIGEST_ALGORITHM

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        relative_to: str | Path | None = None,
    ) -> "ArtifactIdentity":
        file_path = Path(path)
        data = file_path.read_bytes()
        rendered = str(file_path)
        if relative_to is not None:
            try:
                rendered = str(file_path.resolve().relative_to(Path(relative_to).resolve()))
            except ValueError:
                rendered = str(file_path.resolve())
        return cls(path=rendered, size_bytes=len(data), sha256=sha256_bytes(data))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactIdentity":
        if str(payload.get("schema_version") or "") != DIAKRINO_ARTIFACT_IDENTITY_SCHEMA_VERSION:
            raise DiakrinoSidecarIdentityError("unsupported artifact identity schema")
        if str(payload.get("digest_algorithm") or "") != DIGEST_ALGORITHM:
            raise DiakrinoSidecarIdentityError("unsupported artifact digest algorithm")
        path = str(payload.get("path") or "").strip()
        size = payload.get("size_bytes")
        if not path:
            raise DiakrinoSidecarIdentityError("artifact identity is missing path")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DiakrinoSidecarIdentityError("artifact size_bytes is malformed")
        return cls(
            path=path,
            size_bytes=int(size),
            sha256=_require_sha256(payload.get("sha256"), field="artifact.sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "digest_algorithm": self.digest_algorithm,
            "path": self.path,
            "size_bytes": int(self.size_bytes),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class DiakrinoExpectedIdentity:
    """Small serializable identity a canonical runner reconstructs and supplies."""

    dataset_id: str
    checkpoint_sha256: str
    model_config_sha256: str
    producer_source_sha256: str
    source_manifest_sha256: str
    input_x: CanonicalArrayIdentity
    input_y: CanonicalArrayIdentity
    support_indices: CanonicalArrayIdentity
    query_indices: CanonicalArrayIdentity
    feature_order: FeatureOrderIdentity
    calibration_mode: str
    sidecar_schema_version: int
    schema_version: str = DIAKRINO_SIDECAR_IDENTITY_SCHEMA_VERSION
    digest_algorithm: str = DIGEST_ALGORITHM

    @classmethod
    def from_arrays(
        cls,
        *,
        dataset_id: str,
        checkpoint_sha256: str,
        model_config: Mapping[str, Any] | None = None,
        model_config_sha256: str | None = None,
        producer_source_sha256: str,
        source_manifest_sha256: str,
        X: Any,
        y: Any,
        support_indices: Sequence[int] | np.ndarray,
        query_indices: Sequence[int] | np.ndarray,
        feature_order: Sequence[Any] | None = None,
        calibration_mode: str = DIAKRINO_REQUIRED_CALIBRATION_MODE,
        sidecar_schema_version: int = 1,
    ) -> "DiakrinoExpectedIdentity":
        """Construct from exact arrays.

        ``feature_order=None`` explicitly hashes original positions ``0..F-1``.
        Supply either ``model_config`` (which is canonically hashed here) or an
        already verified ``model_config_sha256`` copied from producer metadata.
        """

        dataset = str(dataset_id).strip()
        if not dataset:
            raise DiakrinoSidecarIdentityError("dataset_id is required")
        X_array, y_array = canonicalize_diakrino_inputs(X, y)
        support = exact_int64_vector(support_indices, label="support indices")
        query = exact_int64_vector(query_indices, label="query indices")
        for label, indices in (("support", support), ("query", query)):
            if indices.ndim != 1:
                raise DiakrinoSidecarIdentityError(f"{label} indices must be one-dimensional")
            if np.any(indices < 0) or np.any(indices >= X_array.shape[0]):
                raise DiakrinoSidecarIdentityError(f"{label} indices are outside the input row range")
            if np.unique(indices).size != indices.size:
                raise DiakrinoSidecarIdentityError(f"{label} indices contain duplicates")
        if np.intersect1d(support, query).size:
            raise DiakrinoSidecarIdentityError("support and query indices overlap")
        assigned = np.sort(np.concatenate((support, query)))
        if not np.array_equal(assigned, np.arange(X_array.shape[0], dtype=np.int64)):
            raise DiakrinoSidecarIdentityError("support/query indices do not partition every input row")
        if model_config is None and model_config_sha256 is None:
            raise DiakrinoSidecarIdentityError("model_config or model_config_sha256 is required")
        if model_config is not None and model_config_sha256 is not None:
            observed = canonical_json_sha256(
                model_config, payload_schema_version="diakrino_model_config_v1"
            )
            if observed != _require_sha256(
                model_config_sha256, field="model_config_sha256"
            ):
                raise DiakrinoSidecarIdentityError("model_config and model_config_sha256 disagree")
            config_digest = observed
        elif model_config is not None:
            config_digest = canonical_json_sha256(
                model_config, payload_schema_version="diakrino_model_config_v1"
            )
        else:
            config_digest = _require_sha256(
                model_config_sha256, field="model_config_sha256"
            )
        calibration = str(calibration_mode).strip().lower()
        if not calibration:
            raise DiakrinoSidecarIdentityError("calibration_mode is required")
        schema = exact_int64_scalar(
            sidecar_schema_version,
            label="sidecar_schema_version",
            minimum=1,
        )
        identity = cls(
            dataset_id=dataset,
            checkpoint_sha256=_require_sha256(
                checkpoint_sha256, field="checkpoint_sha256"
            ),
            model_config_sha256=config_digest,
            producer_source_sha256=_require_sha256(
                producer_source_sha256, field="producer_source_sha256"
            ),
            source_manifest_sha256=_require_sha256(
                source_manifest_sha256, field="source_manifest_sha256"
            ),
            input_x=CanonicalArrayIdentity.from_array(X_array),
            input_y=CanonicalArrayIdentity.from_array(y_array),
            support_indices=canonical_index_identity(support),
            query_indices=canonical_index_identity(query),
            feature_order=FeatureOrderIdentity.from_values(
                feature_order, n_features=int(X_array.shape[1])
            ),
            calibration_mode=calibration,
            sidecar_schema_version=schema,
        )
        identity._validate_dimensions()
        return identity

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiakrinoExpectedIdentity":
        if str(payload.get("schema_version") or "") != DIAKRINO_SIDECAR_IDENTITY_SCHEMA_VERSION:
            raise DiakrinoSidecarIdentityError("unsupported DIAKRINO sidecar identity schema")
        if str(payload.get("digest_algorithm") or "") != DIGEST_ALGORITHM:
            raise DiakrinoSidecarIdentityError("unsupported DIAKRINO identity digest algorithm")
        for field in ("input_x", "input_y", "support_indices", "query_indices", "feature_order"):
            if not isinstance(payload.get(field), Mapping):
                raise DiakrinoSidecarIdentityError(f"DIAKRINO identity field {field!r} is malformed")
        identity = cls(
            dataset_id=str(payload.get("dataset_id") or "").strip(),
            checkpoint_sha256=_require_sha256(
                payload.get("checkpoint_sha256"), field="checkpoint_sha256"
            ),
            model_config_sha256=_require_sha256(
                payload.get("model_config_sha256"), field="model_config_sha256"
            ),
            producer_source_sha256=_require_sha256(
                payload.get("producer_source_sha256"), field="producer_source_sha256"
            ),
            source_manifest_sha256=_require_sha256(
                payload.get("source_manifest_sha256"), field="source_manifest_sha256"
            ),
            input_x=CanonicalArrayIdentity.from_dict(payload["input_x"]),
            input_y=CanonicalArrayIdentity.from_dict(payload["input_y"]),
            support_indices=CanonicalArrayIdentity.from_dict(payload["support_indices"]),
            query_indices=CanonicalArrayIdentity.from_dict(payload["query_indices"]),
            feature_order=FeatureOrderIdentity.from_dict(payload["feature_order"]),
            calibration_mode=str(payload.get("calibration_mode") or "").strip().lower(),
            sidecar_schema_version=exact_int64_scalar(
                payload.get("sidecar_schema_version"),
                label="sidecar_schema_version",
                minimum=1,
            ),
        )
        identity._validate_dimensions()
        supplied_binding = _require_sha256(
            payload.get("binding_sha256"), field="binding_sha256"
        )
        if supplied_binding != identity.binding_sha256:
            raise DiakrinoSidecarIdentityError("DIAKRINO identity binding_sha256 is inconsistent")
        return identity

    def _validate_dimensions(self) -> None:
        if not self.dataset_id:
            raise DiakrinoSidecarIdentityError("dataset_id is required")
        if len(self.input_x.shape) != 2:
            raise DiakrinoSidecarIdentityError("input_x identity must describe a matrix")
        if self.input_y.shape != (self.input_x.shape[0],):
            raise DiakrinoSidecarIdentityError("input_y identity is not row-aligned")
        for label, identity in (
            ("support_indices", self.support_indices),
            ("query_indices", self.query_indices),
        ):
            if len(identity.shape) != 1 or identity.dtype != np.dtype(np.int64).str:
                raise DiakrinoSidecarIdentityError(f"{label} must be a canonical int64 vector")
        if self.support_indices.shape[0] + self.query_indices.shape[0] != self.input_x.shape[0]:
            raise DiakrinoSidecarIdentityError("support/query identity lengths do not cover input rows")
        if self.feature_order.count != self.input_x.shape[1]:
            raise DiakrinoSidecarIdentityError("feature-order identity length does not match X")
        exact_int64_scalar(
            self.sidecar_schema_version,
            label="sidecar_schema_version",
            minimum=1,
        )
        if not self.calibration_mode:
            raise DiakrinoSidecarIdentityError("DIAKRINO calibration/schema identity is malformed")

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "digest_algorithm": self.digest_algorithm,
            "dataset_id": self.dataset_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_config_sha256": self.model_config_sha256,
            "producer_source_sha256": self.producer_source_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "input_x": self.input_x.to_dict(),
            "input_y": self.input_y.to_dict(),
            "support_indices": self.support_indices.to_dict(),
            "query_indices": self.query_indices.to_dict(),
            "feature_order": self.feature_order.to_dict(),
            "calibration_mode": self.calibration_mode,
            "sidecar_schema_version": int(self.sidecar_schema_version),
        }

    @property
    def binding_sha256(self) -> str:
        return canonical_json_sha256(
            self._binding_payload(), payload_schema_version=DIAKRINO_SIDECAR_IDENTITY_SCHEMA_VERSION
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._binding_payload(), "binding_sha256": self.binding_sha256}

    def frame_fields(self) -> dict[str, Any]:
        return {
            "diakrino_identity_schema_version": self.schema_version,
            "diakrino_identity_binding_sha256": self.binding_sha256,
            "diakrino_identity_dataset_id": self.dataset_id,
            "diakrino_identity_checkpoint_sha256": self.checkpoint_sha256,
            "diakrino_identity_model_config_sha256": self.model_config_sha256,
            "diakrino_identity_producer_source_sha256": self.producer_source_sha256,
            "diakrino_identity_source_manifest_sha256": self.source_manifest_sha256,
            "diakrino_identity_input_x_sha256": self.input_x.c_order_sha256,
            "diakrino_identity_input_y_sha256": self.input_y.c_order_sha256,
            "diakrino_identity_support_indices_sha256": self.support_indices.c_order_sha256,
            "diakrino_identity_query_indices_sha256": self.query_indices.c_order_sha256,
            "diakrino_identity_feature_order_sha256": self.feature_order.sha256,
            "diakrino_identity_calibration_mode": self.calibration_mode,
            "diakrino_identity_sidecar_schema_version": int(self.sidecar_schema_version),
        }


def build_dataset_identity_record(
    identity: DiakrinoExpectedIdentity,
    *,
    support_indices: Sequence[int] | np.ndarray,
    query_indices: Sequence[int] | np.ndarray,
    feature_logits_path: str | Path,
    query_class_logits_path: str | Path,
    aux_logits_path: str | Path,
    manifest_root: str | Path,
    feature_embeddings_path: str | Path | None = None,
    inference_views_path: str | Path | None = None,
    native_nulls_path: str | Path | None = None,
    paired_inference_views_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one manifest row after every artifact has reached its final bytes."""

    support = exact_int64_vector(support_indices, label="support indices")
    query = exact_int64_vector(query_indices, label="query indices")
    if canonical_index_identity(support) != identity.support_indices:
        raise DiakrinoSidecarIdentityError("support indices disagree with DIAKRINO identity")
    if canonical_index_identity(query) != identity.query_indices:
        raise DiakrinoSidecarIdentityError("query indices disagree with DIAKRINO identity")
    artifacts = {
        "feature_logits": ArtifactIdentity.from_path(
            feature_logits_path, relative_to=manifest_root
        ).to_dict(),
        "query_class_logits": ArtifactIdentity.from_path(
            query_class_logits_path, relative_to=manifest_root
        ).to_dict(),
        "aux_logits": ArtifactIdentity.from_path(
            aux_logits_path, relative_to=manifest_root
        ).to_dict(),
    }
    if feature_embeddings_path and Path(feature_embeddings_path).is_file():
        artifacts["feature_embeddings"] = ArtifactIdentity.from_path(
            feature_embeddings_path, relative_to=manifest_root
        ).to_dict()
    if inference_views_path and Path(inference_views_path).is_file():
        artifacts["inference_views"] = ArtifactIdentity.from_path(
            inference_views_path, relative_to=manifest_root
        ).to_dict()
    if native_nulls_path and Path(native_nulls_path).is_file():
        artifacts["native_nulls"] = ArtifactIdentity.from_path(
            native_nulls_path, relative_to=manifest_root
        ).to_dict()
    if paired_inference_views_path and Path(paired_inference_views_path).is_file():
        artifacts["paired_inference_views"] = ArtifactIdentity.from_path(
            paired_inference_views_path, relative_to=manifest_root
        ).to_dict()
    return {
        "dataset_id": identity.dataset_id,
        "identity": identity.to_dict(),
        "support_indices": [int(value) for value in support.tolist()],
        "query_indices": [int(value) for value in query.tolist()],
        "artifacts": artifacts,
        # Compatibility paths are informational only; strict consumers use artifacts.
        "feature_logits_path": artifacts["feature_logits"]["path"],
        "query_class_logits_path": artifacts["query_class_logits"]["path"],
        "aux_logits_path": artifacts["aux_logits"]["path"],
        "sidecar_schema_version": int(identity.sidecar_schema_version),
    }


def read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise DiakrinoSidecarIdentityError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DiakrinoSidecarIdentityError(f"{label} must contain a JSON object")
    return dict(payload)


def resolve_manifest_identity_record(
    manifest_path: str | Path,
    *,
    dataset_id: str,
    manifest_payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], DiakrinoExpectedIdentity]:
    """Parse a v2 manifest and resolve exactly one dataset identity record."""

    path = Path(manifest_path)
    if path.suffix.lower() != ".json" or not path.is_file():
        raise DiakrinoSidecarIdentityError(
            "strict DIAKRINO loading requires a v2 manifest JSON; direct parquet/directory paths are diagnostic-only"
        )
    manifest = (
        dict(manifest_payload)
        if manifest_payload is not None
        else read_json_object(path, label="DIAKRINO sidecar manifest")
    )
    if str(manifest.get("schema_version") or "") != DIAKRINO_SIDECAR_MANIFEST_SCHEMA_VERSION:
        raise DiakrinoSidecarIdentityError(
            "DIAKRINO sidecar manifest is legacy or has an unsupported schema version"
        )
    if manifest.get("dry_run") is not False:
        raise DiakrinoSidecarIdentityError("dry-run DIAKRINO manifests are not claim eligible")
    rows = manifest.get("sidecars")
    if not isinstance(rows, list):
        raise DiakrinoSidecarIdentityError("DIAKRINO sidecar manifest has no sidecars list")
    matches = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and str(row.get("dataset_id") or "") == str(dataset_id)
    ]
    if len(matches) != 1:
        raise DiakrinoSidecarIdentityError(
            f"DIAKRINO manifest must contain exactly one record for {dataset_id!r}; found {len(matches)}"
        )
    row = matches[0]
    raw_identity = row.get("identity")
    if not isinstance(raw_identity, Mapping):
        raise DiakrinoSidecarIdentityError("DIAKRINO manifest dataset record has no identity")
    identity = DiakrinoExpectedIdentity.from_dict(raw_identity)
    if identity.dataset_id != str(dataset_id):
        raise DiakrinoSidecarIdentityError("DIAKRINO dataset identity does not match requested dataset")
    row_schema = exact_int64_scalar(
        row.get("sidecar_schema_version"),
        label="manifest sidecar_schema_version",
        minimum=1,
    )
    if row_schema != identity.sidecar_schema_version:
        raise DiakrinoSidecarIdentityError(
            "manifest sidecar schema version disagrees with dataset identity"
        )
    manifest_checkpoint = _require_sha256(
        manifest.get("checkpoint_sha256"), field="manifest.checkpoint_sha256"
    )
    if manifest_checkpoint != identity.checkpoint_sha256:
        raise DiakrinoSidecarIdentityError("manifest and dataset checkpoint identities disagree")
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise DiakrinoSidecarIdentityError("DIAKRINO manifest has no model config payload")
    observed_config = canonical_json_sha256(
        config, payload_schema_version="diakrino_model_config_v1"
    )
    if observed_config != identity.model_config_sha256:
        raise DiakrinoSidecarIdentityError("manifest model config digest disagrees with dataset identity")
    producer_source = manifest.get("producer_source_manifest")
    if not isinstance(producer_source, Mapping):
        raise DiakrinoSidecarIdentityError("DIAKRINO manifest has no producer source-closure payload")
    if validate_producer_source_manifest(producer_source) != identity.producer_source_sha256:
        raise DiakrinoSidecarIdentityError(
            "producer source-closure digest disagrees with dataset identity"
        )
    support = row.get("support_indices")
    query = row.get("query_indices")
    if not isinstance(support, list) or not isinstance(query, list):
        raise DiakrinoSidecarIdentityError("DIAKRINO manifest is missing exact support/query index arrays")
    try:
        support_array = exact_int64_vector(support, label="support indices")
        query_array = exact_int64_vector(query, label="query indices")
    except Exception as exc:
        raise DiakrinoSidecarIdentityError(
            f"DIAKRINO support/query arrays are malformed: {exc}"
        ) from exc
    if support_array.ndim != 1 or query_array.ndim != 1:
        raise DiakrinoSidecarIdentityError("DIAKRINO support/query arrays must be one-dimensional")
    for label, array, declared in (
        ("support", support_array, identity.support_indices),
        ("query", query_array, identity.query_indices),
    ):
        if np.any(array < 0) or np.unique(array).size != array.size:
            raise DiakrinoSidecarIdentityError(f"DIAKRINO {label} indices are negative or duplicated")
        if canonical_index_identity(array) != declared:
            raise DiakrinoSidecarIdentityError(f"DIAKRINO {label} index digest mismatch")
    if np.intersect1d(support_array, query_array).size:
        raise DiakrinoSidecarIdentityError("DIAKRINO support/query indices overlap")
    assigned = np.sort(np.concatenate((support_array, query_array)))
    if not np.array_equal(assigned, np.arange(identity.input_x.shape[0], dtype=np.int64)):
        raise DiakrinoSidecarIdentityError("DIAKRINO support/query indices do not partition every row")
    return manifest, row, identity


def read_verified_artifacts(
    manifest_path: str | Path,
    row: Mapping[str, Any],
    *,
    require_relative_paths: bool = False,
) -> tuple[dict[str, bytes], dict[str, str]]:
    """Read each required artifact once, then verify length and digest.

    Returning the already-hashed bytes lets the strict loader parse those exact bytes
    instead of reopening a path after validation (avoiding a check/use race).
    """

    artifacts = row.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise DiakrinoSidecarIdentityError("DIAKRINO dataset record has no artifact identities")
    root = Path(manifest_path).parent
    data_by_name: dict[str, bytes] = {}
    path_by_name: dict[str, str] = {}
    required = ("feature_logits", "query_class_logits", "aux_logits")
    optional = (
        "feature_embeddings",
        "inference_views",
        "native_nulls",
        "paired_inference_views",
    )
    for name in (*required, *(item for item in optional if item in artifacts)):
        raw = artifacts.get(name)
        if not isinstance(raw, Mapping):
            raise DiakrinoSidecarIdentityError(f"DIAKRINO dataset record is missing {name} artifact")
        expected = ArtifactIdentity.from_dict(raw)
        path = Path(expected.path)
        if require_relative_paths and path.is_absolute():
            raise DiakrinoSidecarIdentityError(
                f"DIAKRINO {name} artifact uses a host-local absolute path"
            )
        if not path.is_absolute():
            path = root / path
        if require_relative_paths:
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise DiakrinoSidecarIdentityError(
                    f"DIAKRINO {name} artifact escapes the portable bundle root"
                ) from exc
        try:
            data = path.read_bytes()
        except Exception as exc:
            raise DiakrinoSidecarIdentityError(f"DIAKRINO artifact is unreadable: {path}") from exc
        if len(data) != expected.size_bytes:
            raise DiakrinoSidecarIdentityError(f"DIAKRINO {name} artifact size mismatch")
        if sha256_bytes(data) != expected.sha256:
            raise DiakrinoSidecarIdentityError(f"DIAKRINO {name} artifact SHA-256 mismatch")
        data_by_name[name] = data
        path_by_name[name] = str(path)
    return data_by_name, path_by_name


def read_verified_artifact_reference(
    manifest_path: str | Path,
    reference: Mapping[str, Any],
    *,
    label: str,
    require_relative_path: bool = False,
) -> tuple[bytes, str, ArtifactIdentity]:
    """Read and verify one manifest-level artifact reference exactly once."""

    expected = ArtifactIdentity.from_dict(reference)
    path = Path(expected.path)
    if require_relative_path and path.is_absolute():
        raise DiakrinoSidecarIdentityError(
            f"DIAKRINO {label} artifact uses a host-local absolute path"
        )
    if not path.is_absolute():
        path = Path(manifest_path).parent / path
    if require_relative_path:
        root = Path(manifest_path).parent.resolve()
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise DiakrinoSidecarIdentityError(
                f"DIAKRINO {label} artifact escapes the portable bundle root"
            ) from exc
    try:
        data = path.read_bytes()
    except Exception as exc:
        raise DiakrinoSidecarIdentityError(f"DIAKRINO {label} artifact is unreadable: {path}") from exc
    if len(data) != expected.size_bytes:
        raise DiakrinoSidecarIdentityError(f"DIAKRINO {label} artifact size mismatch")
    if sha256_bytes(data) != expected.sha256:
        raise DiakrinoSidecarIdentityError(f"DIAKRINO {label} artifact SHA-256 mismatch")
    return data, str(path), expected


def validate_frame_identity(frame: Any, identity: DiakrinoExpectedIdentity, *, artifact: str) -> None:
    """Require every artifact row to carry the complete expected binding."""

    missing = [column for column in IDENTITY_FRAME_COLUMNS if column not in frame.columns]
    if missing:
        raise DiakrinoSidecarIdentityError(
            f"DIAKRINO {artifact} frame is missing identity columns: {missing}"
        )
    expected = identity.frame_fields()
    for column, value in expected.items():
        observed = frame[column]
        if len(observed) == 0:
            raise DiakrinoSidecarIdentityError(f"DIAKRINO {artifact} frame is empty")
        if column == "diakrino_identity_sidecar_schema_version":
            matches = all(
                exact_int64_scalar(
                    item,
                    label=f"DIAKRINO {artifact} row {column}",
                    minimum=1,
                )
                == value
                for item in observed.tolist()
            )
        else:
            matches = all(item == value for item in observed.tolist())
        if not matches:
            raise DiakrinoSidecarIdentityError(
                f"DIAKRINO {artifact} row identity mismatch in {column}"
            )


def bind_qualified_sidecar_manifest(
    source_manifest_path: str | Path,
    qualification_record_path: str | Path,
    output_manifest_path: str | Path,
) -> dict[str, Any]:
    """Create a claim manifest without rerunning DIAKRINO inference.

    This is phase three of the non-circular contract:

    1. emission writes v2 identities/artifacts with feature-selection trust denied;
    2. qualification validates that immutable emission and records exact bindings;
    3. this binder references the same artifact hashes and the immutable qualification
       record, enabling trust only when every binding was qualified from this source
       emission manifest.
    """

    from .diakrino_trust import (  # Local import avoids an identity/trust import cycle.
        default_diakrino_sidecar_trust_record,
        source_emission_qualification_covers_identity,
        trust_record_head_allowed,
    )

    source_path = Path(source_manifest_path).expanduser().resolve()
    qualification_path = Path(qualification_record_path).expanduser().resolve()
    output_path = Path(output_manifest_path).expanduser().resolve()
    if output_path.parent != source_path.parent:
        raise DiakrinoSidecarIdentityError(
            "claim manifest must be written inside the source emission bundle root"
        )
    if output_path == source_path:
        raise DiakrinoSidecarIdentityError("claim manifest must not overwrite the source emission manifest")
    source = read_json_object(source_path, label="DIAKRINO source emission manifest")
    if str(source.get("schema_version") or "") != DIAKRINO_SIDECAR_MANIFEST_SCHEMA_VERSION:
        raise DiakrinoSidecarIdentityError("binder requires a v2 source emission manifest")
    if source.get("dry_run") is not False:
        raise DiakrinoSidecarIdentityError("binder cannot promote a dry-run emission")
    source_phase = source.get("binding_phase")
    if source_phase not in (None, "", "emission_manifest"):
        raise DiakrinoSidecarIdentityError(
            "binder requires an emission-phase manifest, not an already bound claim"
        )
    if source.get("source_emission_manifest_artifact") is not None:
        raise DiakrinoSidecarIdentityError(
            "binder source must not itself reference another source-emission manifest"
        )
    source_trust = source.get("diakrino_sidecar_trust_record")
    if not isinstance(source_trust, Mapping) or trust_record_head_allowed(
        source_trust, "feature_selection"
    ):
        raise DiakrinoSidecarIdentityError(
            "binder source emission must explicitly deny feature-selection trust"
        )
    qualification = read_json_object(
        qualification_path, label="DIAKRINO qualification record"
    )
    source_sha256 = sha256_file(source_path)
    rows = source.get("sidecars")
    if not isinstance(rows, list) or not rows:
        raise DiakrinoSidecarIdentityError("source emission manifest has no sidecars")
    identities: list[DiakrinoExpectedIdentity] = []
    bound_rows: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise DiakrinoSidecarIdentityError("source sidecar record is malformed")
        dataset_id = str(raw.get("dataset_id") or "").strip()
        _, row, identity = resolve_manifest_identity_record(
            source_path,
            dataset_id=dataset_id,
        )
        # Re-hash every referenced artifact before carrying it into the claim manifest.
        read_verified_artifacts(source_path, row, require_relative_paths=True)
        if not source_emission_qualification_covers_identity(
            qualification,
            identity,
            source_manifest_sha256=source_sha256,
        ):
            raise DiakrinoSidecarIdentityError(
                "source-emission qualification does not cover binding "
                f"{identity.binding_sha256}"
            )
        item = json.loads(json.dumps(row))
        artifacts = item.get("artifacts")
        assert isinstance(artifacts, dict)
        for name, artifact_payload in artifacts.items():
            artifact = ArtifactIdentity.from_dict(artifact_payload)
            artifact_path = Path(artifact.path)
            if artifact_path.is_absolute():
                raise DiakrinoSidecarIdentityError(
                    f"source emission artifact {name!r} is not bundle-relative"
                )
            resolved_artifact = (source_path.parent / artifact_path).resolve()
            try:
                resolved_artifact.relative_to(source_path.parent)
            except ValueError as exc:
                raise DiakrinoSidecarIdentityError(
                    f"source emission artifact {name!r} escapes the bundle root"
                ) from exc
            if name == "feature_logits":
                item["feature_logits_path"] = artifact.path
            elif name == "query_class_logits":
                item["query_class_logits_path"] = artifact.path
            elif name == "aux_logits":
                item["aux_logits_path"] = artifact.path
        identities.append(identity)
        bound_rows.append(item)
    checkpoint = source.get("checkpoint_sha256")
    qualification_bytes = qualification_path.read_bytes()
    qualification_sha256 = sha256_bytes(qualification_bytes)
    qualification_relative = Path("qualification_records") / f"{qualification_sha256}.json"
    bundled_qualification = output_path.parent / qualification_relative
    _atomic_publish_bytes(
        bundled_qualification,
        qualification_bytes,
        allow_existing_identical=True,
    )
    trust = default_diakrino_sidecar_trust_record(
        checkpoint_sha256=str(checkpoint or ""),
        qualification_record_path=bundled_qualification,
        qualification_record=qualification,
        required_sidecar_identities=identities,
        source_manifest_sha256=source_sha256,
    )
    trust["qualification_record_path"] = qualification_relative.as_posix()
    if trust.get("exact_identity_coverage") is not True:
        raise DiakrinoSidecarIdentityError("qualification did not enable exact identity coverage")
    claim = json.loads(json.dumps(source))
    claim.update(
        {
            "schema_version": DIAKRINO_SIDECAR_MANIFEST_SCHEMA_VERSION,
            "tool": "tabnetics.feature_selection.diakrino_identity.bind_qualified_sidecar_manifest",
            "dry_run": False,
            "source_emission_manifest_artifact": ArtifactIdentity.from_path(
                source_path, relative_to=output_path.parent
            ).to_dict(),
            "qualification_record_artifact": ArtifactIdentity.from_path(
                bundled_qualification, relative_to=output_path.parent
            ).to_dict(),
            "diakrino_sidecar_trust_record": trust,
            "sidecars": bound_rows,
            "results": bound_rows,
            "binding_phase": "qualified_claim_manifest",
        }
    )
    rendered = (json.dumps(_json_native(claim), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_publish_bytes(output_path, rendered, allow_existing_identical=False)
    return claim


def _atomic_publish_bytes(
    output_path: Path,
    data: bytes,
    *,
    allow_existing_identical: bool,
) -> None:
    """Atomically publish complete bytes without replacing an existing path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if allow_existing_identical and output_path.read_bytes() == data:
            return
        raise DiakrinoSidecarIdentityError(f"refusing to overwrite existing artifact: {output_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Hard-link publication is atomic and fails if another process created
            # the requested output after the preflight check.
            os.link(temporary_name, output_path)
        except FileExistsError as exc:
            raise DiakrinoSidecarIdentityError(
                f"refusing to overwrite existing artifact: {output_path}"
            ) from exc
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


__all__ = [
    "ArtifactIdentity",
    "CanonicalArrayIdentity",
    "DIGEST_ALGORITHM",
    "FeatureOrderIdentity",
    "IDENTITY_FRAME_COLUMNS",
    "DIAKRINO_ARTIFACT_IDENTITY_SCHEMA_VERSION",
    "DIAKRINO_CANONICAL_ARRAY_SCHEMA_VERSION",
    "DIAKRINO_FEATURE_ORDER_SCHEMA_VERSION",
    "DIAKRINO_REQUIRED_CALIBRATION_MODE",
    "DIAKRINO_SIDECAR_IDENTITY_SCHEMA_VERSION",
    "DIAKRINO_SIDECAR_MANIFEST_SCHEMA_VERSION",
    "DiakrinoExpectedIdentity",
    "DiakrinoSidecarIdentityError",
    "build_dataset_identity_record",
    "bind_qualified_sidecar_manifest",
    "build_producer_source_manifest",
    "canonical_index_identity",
    "canonicalize_diakrino_inputs",
    "canonical_json_sha256",
    "read_verified_artifacts",
    "read_verified_artifact_reference",
    "resolve_manifest_identity_record",
    "sha256_bytes",
    "sha256_file",
    "exact_int64_scalar",
    "exact_int64_vector",
    "validate_producer_source_manifest",
    "validate_frame_identity",
]

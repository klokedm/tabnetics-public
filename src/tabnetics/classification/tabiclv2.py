"""Fail-closed adapter for the optional TabICLv2 benchmark comparator.

The adapter deliberately owns no checkpoint discovery or download behavior.  A
caller must provide a local checkpoint whose provenance can be recorded before
the optional upstream package is imported.
"""

from __future__ import annotations

import hashlib
import importlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import re
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


TABICLV2_PACKAGE = "tabicl"
TABICLV2_PACKAGE_VERSION = "2.1.1"
TABICLV2_CLASS_MODULE = "tabicl._sklearn.classifier"
TABICLV2_CLASS_NAME = "TabICLClassifier"
TABICLV2_REPO_ID = "jingang/TabICL"
TABICLV2_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
TABICLV2_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
TABICLV2_CHECKPOINT_SIZE_BYTES = 110_368_038
TABICLV2_CHECKPOINT_SHA256 = (
    "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
)
TABICLV2_LICENSE = "BSD-3-Clause"
TABICLV2_PUBLISHED_MIN_TRAIN_ROWS = 300
TABICLV2_PUBLISHED_MAX_TRAIN_ROWS = 100_000
TABICLV2_PUBLISHED_MAX_FEATURES = 2_000

_CUDA_DEVICE_PATTERN = re.compile(r"cuda(?::(?P<index>[0-9]+))?\Z")


class TabICLv2Error(RuntimeError):
    """Base error carrying a stable runner-facing status code."""

    def __init__(self, message: str, *, status: str) -> None:
        super().__init__(message)
        self.status = status


class TabICLv2AvailabilityError(TabICLv2Error):
    """The pinned comparator cannot truthfully run in this environment."""


class TabICLv2ContractError(TabICLv2Error):
    """Inputs or upstream runtime behavior violate the adapter contract."""

    def __init__(
        self, message: str, *, status: str = "failed_tabiclv2_contract"
    ) -> None:
        super().__init__(message, status=status)


def tabiclv2_contract_identity() -> dict[str, object]:
    """Return immutable-by-convention upstream identity without lazy imports."""

    return {
        "package": {"name": TABICLV2_PACKAGE, "version": TABICLV2_PACKAGE_VERSION},
        "upstream_class": {
            "module": TABICLV2_CLASS_MODULE,
            "name": TABICLV2_CLASS_NAME,
        },
        "checkpoint": {
            "repo_id": TABICLV2_REPO_ID,
            "revision": TABICLV2_REVISION,
            "filename": TABICLV2_CHECKPOINT,
            "size_bytes": TABICLV2_CHECKPOINT_SIZE_BYTES,
            "sha256": TABICLV2_CHECKPOINT_SHA256,
            "identity_semantics": "exact Hugging Face LFS object at the pinned revision",
        },
        "license": TABICLV2_LICENSE,
        "published_limits": {
            "min_train_rows": TABICLV2_PUBLISHED_MIN_TRAIN_ROWS,
            "max_train_rows": TABICLV2_PUBLISHED_MAX_TRAIN_ROWS,
            "max_features": TABICLV2_PUBLISHED_MAX_FEATURES,
            "semantics": "training-context limits documented for TabICLv2 evaluation",
        },
    }


def _availability(message: str, status: str) -> TabICLv2AvailabilityError:
    return TabICLv2AvailabilityError(message, status=status)


def _contract(
    message: str, status: str = "failed_tabiclv2_contract"
) -> TabICLv2ContractError:
    return TabICLv2ContractError(message, status=status)


def _distribution_version() -> str:
    try:
        return importlib_metadata.version(TABICLV2_PACKAGE)
    except importlib_metadata.PackageNotFoundError as exc:
        raise _availability(
            f"optional dependency {TABICLV2_PACKAGE}=={TABICLV2_PACKAGE_VERSION} is not installed",
            "skipped_tabiclv2_dependency_unavailable",
        ) from exc
    except Exception as exc:
        raise _availability(
            f"could not resolve the {TABICLV2_PACKAGE} package version: {exc}",
            "skipped_tabiclv2_dependency_unavailable",
        ) from exc


def _load_upstream_class() -> type[Any]:
    installed_version = _distribution_version()
    if installed_version != TABICLV2_PACKAGE_VERSION:
        raise _availability(
            f"TabICLv2 requires {TABICLV2_PACKAGE}=={TABICLV2_PACKAGE_VERSION}; "
            f"found {installed_version}",
            "skipped_tabiclv2_version_mismatch",
        )
    try:
        module = importlib.import_module(TABICLV2_PACKAGE)
    except Exception as exc:
        raise _availability(
            f"could not import pinned {TABICLV2_PACKAGE} package: {exc}",
            "skipped_tabiclv2_dependency_unavailable",
        ) from exc

    upstream_class = getattr(module, TABICLV2_CLASS_NAME, None)
    if not isinstance(upstream_class, type):
        raise _availability(
            f"{TABICLV2_PACKAGE_VERSION} does not expose {TABICLV2_CLASS_NAME}",
            "skipped_tabiclv2_api_mismatch",
        )
    if (
        upstream_class.__module__ != TABICLV2_CLASS_MODULE
        or upstream_class.__name__ != TABICLV2_CLASS_NAME
        or upstream_class.__qualname__ != TABICLV2_CLASS_NAME
    ):
        raise _availability(
            "TabICLv2 upstream class identity does not match the pinned 2.1.1 API",
            "skipped_tabiclv2_api_mismatch",
        )
    return upstream_class


def _local_file_identity(
    path_value: str | Path,
    *,
    unavailable_status: str,
    description: str,
) -> dict[str, object]:
    try:
        path = Path(path_value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _availability(
            f"{description} is unavailable: {path_value}",
            unavailable_status,
        ) from exc
    if not path.is_file():
        raise _availability(
            f"{description} is not a regular file: {path}",
            unavailable_status,
        )
    try:
        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            raise OSError("checkpoint is empty")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise _availability(
            f"{description} cannot be read: {path}: {exc}",
            unavailable_status,
        ) from exc
    actual_sha256 = digest.hexdigest()
    return {
        "path": str(path),
        "size_bytes": int(size_bytes),
        "sha256": actual_sha256,
    }


def _verify_official_checkpoint_identity(
    identity: dict[str, object], *, description: str
) -> None:
    if (
        identity["size_bytes"] != TABICLV2_CHECKPOINT_SIZE_BYTES
        or identity["sha256"] != TABICLV2_CHECKPOINT_SHA256
    ):
        raise _availability(
            f"{description} size/SHA-256 does not match the official LFS object "
            f"at {TABICLV2_REPO_ID}@{TABICLV2_REVISION}",
            "skipped_tabiclv2_checkpoint_identity_mismatch",
        )


def _pinned_cache_checkpoint_identity() -> dict[str, object]:
    try:
        hub_module = importlib.import_module("huggingface_hub")
        resolver = getattr(hub_module, "hf_hub_download")
    except Exception as exc:
        raise _availability(
            f"could not import pinned Hugging Face local-cache resolver: {exc}",
            "skipped_tabiclv2_checkpoint_provenance_unavailable",
        ) from exc
    try:
        cache_path = resolver(
            repo_id=TABICLV2_REPO_ID,
            filename=TABICLV2_CHECKPOINT,
            revision=TABICLV2_REVISION,
            local_files_only=True,
        )
    except Exception as exc:
        raise _availability(
            "the pinned TabICLv2 checkpoint is not present in the local Hugging Face cache",
            "skipped_tabiclv2_checkpoint_provenance_unavailable",
        ) from exc
    identity = _local_file_identity(
        cache_path,
        unavailable_status="skipped_tabiclv2_checkpoint_provenance_unavailable",
        description="pinned Hugging Face TabICLv2 cache entry",
    )
    _verify_official_checkpoint_identity(
        identity, description="pinned Hugging Face TabICLv2 cache entry"
    )
    return identity


def _checkpoint_identity(checkpoint_path: str | Path) -> dict[str, object]:
    if not isinstance(checkpoint_path, (str, Path)) or not str(checkpoint_path).strip():
        raise _availability(
            "TabICLv2 requires an explicit local checkpoint path",
            "skipped_tabiclv2_checkpoint_not_configured",
        )
    identity = _local_file_identity(
        checkpoint_path,
        unavailable_status="skipped_tabiclv2_checkpoint_unavailable",
        description="TabICLv2 checkpoint",
    )
    _verify_official_checkpoint_identity(identity, description="TabICLv2 checkpoint")
    pinned_cache_identity = _pinned_cache_checkpoint_identity()
    if (
        identity["size_bytes"] != pinned_cache_identity["size_bytes"]
        or identity["sha256"] != pinned_cache_identity["sha256"]
    ):
        raise _availability(
            "TabICLv2 checkpoint does not match the pinned local Hugging Face cache entry",
            "skipped_tabiclv2_checkpoint_identity_mismatch",
        )
    return {
        **identity,
        "repo_id": TABICLV2_REPO_ID,
        "revision": TABICLV2_REVISION,
        "filename": TABICLV2_CHECKPOINT,
        "pinned_cache_path": pinned_cache_identity["path"],
    }


def _numeric_matrix(X: Any, *, operation: str) -> np.ndarray:
    try:
        values = np.asarray(X)
    except Exception as exc:
        raise _contract(
            f"TabICLv2 {operation} X cannot be converted to an array: {exc}"
        ) from exc
    if values.ndim != 2:
        raise _contract(f"TabICLv2 {operation} X must be a two-dimensional matrix")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise _contract(f"TabICLv2 {operation} X must be nonempty")
    if values.dtype.kind not in "iuf":
        raise _contract(f"TabICLv2 {operation} X must contain only real numeric values")
    try:
        numeric = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _contract(
            f"TabICLv2 {operation} X cannot be represented as finite floats"
        ) from exc
    if not np.isfinite(numeric).all():
        raise _contract(f"TabICLv2 {operation} X must contain only finite values")
    return numeric


def _training_labels(y: Any, *, n_rows: int) -> tuple[np.ndarray, LabelEncoder]:
    if y is None:
        raise _contract("TabICLv2 fit requires training labels")
    try:
        labels = np.asarray(y)
    except Exception as exc:
        raise _contract(
            f"TabICLv2 training labels cannot be converted to an array: {exc}"
        ) from exc
    if labels.ndim != 1:
        raise _contract("TabICLv2 training labels must be one-dimensional")
    if labels.shape[0] != n_rows:
        raise _contract("TabICLv2 X and y have inconsistent row counts")
    if labels.shape[0] == 0:
        raise _contract("TabICLv2 training labels must be nonempty")
    if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
        raise _contract(
            "TabICLv2 training labels must not contain missing or non-finite values"
        )
    if labels.dtype.kind == "O":
        for label in labels.tolist():
            if label is None or (
                isinstance(label, (float, np.floating)) and not np.isfinite(label)
            ):
                raise _contract(
                    "TabICLv2 training labels must not contain missing or non-finite values"
                )
    try:
        check_classification_targets(labels)
        encoder = LabelEncoder().fit(labels)
        encoded = encoder.transform(labels)
    except (TypeError, ValueError) as exc:
        raise _contract(
            f"TabICLv2 training labels are not a valid classification target: {exc}"
        ) from exc
    if encoder.classes_.size < 2:
        raise _contract("TabICLv2 requires at least two training classes")
    return np.asarray(encoded, dtype=np.int64), encoder


def _resolve_cuda_device(requested_device: Any) -> tuple[str, Any]:
    if not isinstance(requested_device, str):
        raise _availability(
            "TabICLv2 device must be an explicit CUDA device string",
            "skipped_tabiclv2_cuda_required",
        )
    match = _CUDA_DEVICE_PATTERN.fullmatch(requested_device.strip().lower())
    if match is None:
        raise _availability(
            f"TabICLv2 requires a CUDA device; received {requested_device!r}",
            "skipped_tabiclv2_cuda_required",
        )
    try:
        torch_module = importlib.import_module("torch")
    except Exception as exc:
        raise _availability(
            f"TabICLv2 CUDA validation could not import torch: {exc}",
            "skipped_tabiclv2_cuda_unavailable",
        ) from exc
    try:
        if not bool(torch_module.cuda.is_available()):
            raise _availability(
                "TabICLv2 requires CUDA, but CUDA is not available",
                "skipped_tabiclv2_cuda_unavailable",
            )
        device_count = int(torch_module.cuda.device_count())
        index_text = match.group("index")
        index = (
            int(index_text)
            if index_text is not None
            else int(torch_module.cuda.current_device())
        )
        if index < 0 or index >= device_count:
            raise _availability(
                f"requested TabICLv2 CUDA device cuda:{index} is not visible ({device_count} devices)",
                "skipped_tabiclv2_cuda_unavailable",
            )
    except TabICLv2AvailabilityError:
        raise
    except Exception as exc:
        raise _availability(
            f"TabICLv2 CUDA validation failed: {exc}",
            "skipped_tabiclv2_cuda_unavailable",
        ) from exc
    return f"cuda:{index}", torch_module


def _json_safe_class_order(classes: np.ndarray) -> list[object]:
    result: list[object] = []
    for value in classes.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            result.append(value)
        else:
            result.append({"type": type(value).__name__, "repr": repr(value)})
    return result


class TabICLv2Classifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible, GPU-only wrapper for pinned TabICLv2 inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        min_train_rows: int = TABICLV2_PUBLISHED_MIN_TRAIN_ROWS,
        max_train_rows: int = TABICLV2_PUBLISHED_MAX_TRAIN_ROWS,
        max_features: int = TABICLV2_PUBLISHED_MAX_FEATURES,
        random_state: int = 42,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.min_train_rows = min_train_rows
        self.max_train_rows = max_train_rows
        self.max_features = max_features
        self.random_state = random_state

    def _clear_fitted_state(self) -> None:
        for name in (
            "_estimator_",
            "_label_encoder_",
            "classes_",
            "n_classes_",
            "n_features_in_",
            "n_samples_in_",
            "metadata_",
        ):
            self.__dict__.pop(name, None)

    def _validate_limits(self, *, n_rows: int, n_features: int) -> None:
        values = (self.min_train_rows, self.max_train_rows, self.max_features)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in values
        ):
            raise _contract("TabICLv2 row and feature limits must be integers")
        min_rows = int(self.min_train_rows)
        max_rows = int(self.max_train_rows)
        max_features = int(self.max_features)
        if not (
            TABICLV2_PUBLISHED_MIN_TRAIN_ROWS
            <= min_rows
            <= max_rows
            <= TABICLV2_PUBLISHED_MAX_TRAIN_ROWS
        ):
            raise _contract(
                "TabICLv2 configured row limits cannot widen or invert the published 300..100000 regime"
            )
        if not (1 <= max_features <= TABICLV2_PUBLISHED_MAX_FEATURES):
            raise _contract(
                "TabICLv2 configured feature limit must be within the published 1..2000 regime"
            )
        if n_rows < min_rows or n_rows > max_rows:
            raise _availability(
                f"TabICLv2 fit has {n_rows} rows; configured supported range is {min_rows}..{max_rows}",
                "skipped_tabiclv2_outside_published_regime",
            )
        if n_features > max_features:
            raise _availability(
                f"TabICLv2 fit has {n_features} features; configured supported maximum is {max_features}",
                "skipped_tabiclv2_outside_published_regime",
            )

    def fit(self, X: Any, y: Any) -> TabICLv2Classifier:
        self._clear_fitted_state()
        X_valid = _numeric_matrix(X, operation="fit")
        self._validate_limits(n_rows=X_valid.shape[0], n_features=X_valid.shape[1])
        y_encoded, label_encoder = _training_labels(y, n_rows=X_valid.shape[0])
        checkpoint = _checkpoint_identity(self.checkpoint_path)
        selected_device, _ = _resolve_cuda_device(self.device)
        upstream_class = _load_upstream_class()

        constructor_kwargs = {
            "model_path": checkpoint["path"],
            "allow_auto_download": False,
            "checkpoint_version": TABICLV2_CHECKPOINT,
            "device": selected_device,
            "random_state": self.random_state,
        }
        try:
            estimator = upstream_class(**constructor_kwargs)
        except TypeError as exc:
            raise _availability(
                f"TabICLv2 2.1.1 constructor API mismatch: {exc}",
                "skipped_tabiclv2_api_mismatch",
            ) from exc
        except Exception as exc:
            raise _contract(
                f"TabICLv2 upstream constructor failed: {exc}",
                "failed_tabiclv2_constructor",
            ) from exc
        try:
            estimator.fit(X_valid, y_encoded)
        except Exception as exc:
            raise _contract(
                f"TabICLv2 upstream fit failed: {exc}", "failed_tabiclv2_fit"
            ) from exc
        self._validate_upstream_classes(
            estimator, expected_count=label_encoder.classes_.size
        )

        self._estimator_ = estimator
        self._label_encoder_ = label_encoder
        self.classes_ = np.array(label_encoder.classes_, copy=True)
        self.n_classes_ = int(self.classes_.size)
        self.n_features_in_ = int(X_valid.shape[1])
        self.n_samples_in_ = int(X_valid.shape[0])
        identity = tabiclv2_contract_identity()
        self.metadata_ = {
            **identity,
            "checkpoint": checkpoint,
            "selected_cuda_device": selected_device,
            "effective_limits": {
                "min_train_rows": int(self.min_train_rows),
                "max_train_rows": int(self.max_train_rows),
                "max_features": int(self.max_features),
            },
            "preprocessing": {
                "adapter": "finite real numeric float64 validation; no imputation or scaling",
                "upstream": "TabICL 2.1.1 TransformToNumerical and ensemble preprocessing",
            },
            "class_order": _json_safe_class_order(self.classes_),
            "probability": {
                "kind": "normalized class probability",
                "columns": "class_order",
                "source": "tabicl.TabICLClassifier.predict_proba",
                "validation": "finite, nonnegative, positive-row-sum, exact upstream class order",
            },
        }
        return self

    @staticmethod
    def _validate_upstream_classes(estimator: Any, *, expected_count: int) -> None:
        if not hasattr(estimator, "classes_"):
            raise _contract("TabICLv2 upstream estimator did not expose classes_")
        upstream_classes = np.asarray(estimator.classes_)
        expected = np.arange(expected_count, dtype=np.int64)
        if upstream_classes.ndim != 1 or not np.array_equal(upstream_classes, expected):
            raise _contract(
                "TabICLv2 upstream classes_ does not exactly match encoded training class order"
            )

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(
            self,
            attributes=[
                "_estimator_",
                "_label_encoder_",
                "classes_",
                "n_features_in_",
                "metadata_",
            ],
        )
        X_valid = _numeric_matrix(X, operation="predict")
        if X_valid.shape[1] != self.n_features_in_:
            raise _contract(
                f"TabICLv2 predict has {X_valid.shape[1]} features; fitted data has {self.n_features_in_}"
            )
        self._validate_upstream_classes(
            self._estimator_, expected_count=self.n_classes_
        )
        try:
            raw = self._estimator_.predict_proba(X_valid)
        except Exception as exc:
            raise _contract(
                f"TabICLv2 upstream predict_proba failed: {exc}",
                "failed_tabiclv2_predict_proba",
            ) from exc
        self._validate_upstream_classes(
            self._estimator_, expected_count=self.n_classes_
        )
        try:
            probabilities = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _contract(
                "TabICLv2 predict_proba output is not a numeric matrix"
            ) from exc
        expected_shape = (X_valid.shape[0], self.n_classes_)
        if probabilities.ndim != 2 or probabilities.shape != expected_shape:
            raise _contract(
                f"TabICLv2 predict_proba output must have shape {expected_shape}; got {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all():
            raise _contract("TabICLv2 predict_proba output contains non-finite values")
        if np.any(probabilities < 0):
            raise _contract("TabICLv2 predict_proba output contains negative values")
        row_sums = probabilities.sum(axis=1)
        if not np.isfinite(row_sums).all() or np.any(row_sums <= 0):
            raise _contract(
                "TabICLv2 predict_proba output has a non-positive or non-finite row sum"
            )
        return probabilities / row_sums[:, None]

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        encoded = np.argmax(probabilities, axis=1)
        return self._label_encoder_.inverse_transform(encoded)


__all__ = [
    "TabICLv2AvailabilityError",
    "TabICLv2Classifier",
    "TabICLv2ContractError",
    "TabICLv2Error",
    "tabiclv2_contract_identity",
]

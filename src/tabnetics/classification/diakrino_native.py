"""Native Tabnetics Diakrino classifier inference helpers.

This module keeps the runtime episode encoding in core: robust scaling, per-class
statistics, screening features, selector-informed feature capping, checkpoint
loading, and batched softmax inference.  Torch/model imports are lazy so normal
CPU-only installs are unaffected unless a native DIAKRINO checkpoint is explicitly used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

CLASS_STATS_DIM = 24
MARGINAL_STATS_DIM = 5
SCREENING_FEATURE_DIM = 18
QUANTILE_PROBS = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], dtype=np.float32)
__tabnetics_execution_isolated_state__ = {
    "QUANTILE_PROBS": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": ("numpy",),
    },
}
CLASSIFIER_TRUST_SCHEMA_VERSION = "tabentics_diakrino_classifier_head_trust_v3"
CLASSIFIER_ARTIFACT_MANIFEST_SCHEMA_VERSION = "tabentics_diakrino_classifier_artifacts_v1"
CLASSIFIER_TRUST_RECORD_NAME = "classifier_trust_record.json"
CLASSIFIER_ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
CLASSIFIER_ADMISSION_MIN_TASKS = 20
CLASSIFIER_ADMISSION_MIN_EXAMPLES = 640
CLASSIFIER_ADMISSION_MIN_EXAMPLES_PER_TASK = 32
CLASSIFIER_ADMISSION_MIN_CLASSES = 2
CLASSIFIER_ADMISSION_BOOTSTRAP_SAMPLES = 10000
CLASSIFIER_ADMISSION_ECE_BINS = 10


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _validate_native_training_query(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train_arr = np.asarray(X_train, dtype=np.float32)
    y_train_arr = np.asarray(y_train)
    X_query_arr = np.asarray(X_query, dtype=np.float32)
    if X_train_arr.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {tuple(X_train_arr.shape)}")
    if X_query_arr.ndim != 2:
        raise ValueError(f"X_query must be 2D, got shape {tuple(X_query_arr.shape)}")
    if y_train_arr.ndim != 1:
        raise ValueError(f"y_train must be 1D, got shape {tuple(y_train_arr.shape)}")
    if int(X_train_arr.shape[0]) != int(y_train_arr.shape[0]):
        raise ValueError(
            "X_train and y_train must contain the same number of rows; "
            f"got {int(X_train_arr.shape[0])} and {int(y_train_arr.shape[0])}"
        )
    if int(X_train_arr.shape[0]) <= 0:
        raise ValueError("native Tabnetics Diakrino classification requires at least one training row")
    if int(X_query_arr.shape[0]) <= 0:
        raise ValueError("native Tabnetics Diakrino classification requires at least one query row")
    if int(np.unique(y_train_arr).size) < 2:
        raise ValueError("native Tabnetics Diakrino classification requires at least two classes")
    if int(X_train_arr.shape[1]) <= 0:
        raise ValueError("native Tabnetics Diakrino classification requires at least one feature")
    if int(X_query_arr.shape[1]) != int(X_train_arr.shape[1]):
        raise ValueError(
            "native Tabnetics Diakrino query matrix must have the same feature count as X_train; "
            f"expected {int(X_train_arr.shape[1])}, got {int(X_query_arr.shape[1])}"
        )
    return X_train_arr, y_train_arr, X_query_arr


def fit_robust_scaler(X_support: np.ndarray, *, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        center = np.nanmedian(X_support, axis=0)
        q25 = np.nanpercentile(X_support, 25.0, axis=0)
        q75 = np.nanpercentile(X_support, 75.0, axis=0)
        std = np.nanstd(X_support, axis=0)
    center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    iqr = np.nan_to_num(q75 - q25, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.where(iqr > eps, iqr, np.where(std > eps, std, 1.0))
    return center, scale.astype(np.float32, copy=False)


def apply_robust_scaler(
    X: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    clip_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(X, dtype=np.float32)
    missing = ~np.isfinite(values)
    scaled = (np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) - center[None, :]) / np.maximum(scale[None, :], 1e-6)
    scaled = np.clip(scaled, -float(clip_value), float(clip_value)).astype(np.float32, copy=False)
    scaled[missing] = 0.0
    return scaled, missing


def _validate_marginal_stats_inputs(X: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(X, dtype=np.float32)
    missing_arr = np.asarray(missing, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"native DIAKRINO statistics X must be 2D, got shape {tuple(values.shape)}")
    if missing_arr.ndim != 2:
        raise ValueError(f"native DIAKRINO statistics missing mask must be 2D, got shape {tuple(missing_arr.shape)}")
    if tuple(values.shape) != tuple(missing_arr.shape):
        raise ValueError(
            "native DIAKRINO statistics X and missing mask must have the same shape, "
            f"got {tuple(values.shape)} and {tuple(missing_arr.shape)}"
        )
    if int(values.shape[0]) <= 0:
        raise ValueError("native DIAKRINO statistics X must contain at least one row")
    if int(values.shape[1]) <= 0:
        raise ValueError("native DIAKRINO statistics X must contain at least one feature")
    return values, missing_arr


def _validate_class_stats_inputs(
    X_support: np.ndarray,
    support_missing: np.ndarray,
    y_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, missing = _validate_marginal_stats_inputs(X_support, support_missing)
    labels_raw = np.asarray(y_support)
    if labels_raw.ndim != 1:
        raise ValueError(f"native DIAKRINO class-stat labels must be 1D, got shape {tuple(labels_raw.shape)}")
    if int(labels_raw.shape[0]) != int(values.shape[0]):
        raise ValueError(
            "native DIAKRINO class-stat X and labels must have the same number of rows, "
            f"got {int(values.shape[0])} and {int(labels_raw.shape[0])}"
        )
    try:
        labels = np.asarray(labels_raw, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("native DIAKRINO class-stat labels must be integer-encoded") from exc
    return values, missing, labels


def _validate_screening_inputs(
    marginal: np.ndarray,
    class_stats: np.ndarray,
    class_stats_valid: np.ndarray,
    feature_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    marginal_arr = np.asarray(marginal, dtype=np.float32)
    class_stats_arr = np.asarray(class_stats, dtype=np.float32)
    class_valid_arr = np.asarray(class_stats_valid, dtype=bool)
    feature_valid_arr = np.asarray(feature_valid, dtype=bool)
    if marginal_arr.ndim != 2 or int(marginal_arr.shape[1]) != MARGINAL_STATS_DIM:
        raise ValueError(
            "native DIAKRINO screening marginal stats must have shape "
            f"(features, {MARGINAL_STATS_DIM}), got {tuple(marginal_arr.shape)}"
        )
    if class_stats_arr.ndim != 3 or int(class_stats_arr.shape[2]) != CLASS_STATS_DIM:
        raise ValueError(
            "native DIAKRINO screening class stats must have shape "
            f"(features, classes, {CLASS_STATS_DIM}), got {tuple(class_stats_arr.shape)}"
        )
    if tuple(class_valid_arr.shape) != tuple(class_stats_arr.shape[:2]):
        raise ValueError(
            "native DIAKRINO screening class_stats_valid must match class_stats feature/class shape, "
            f"got {tuple(class_valid_arr.shape)} and {tuple(class_stats_arr.shape[:2])}"
        )
    if feature_valid_arr.ndim != 1:
        raise ValueError(
            f"native DIAKRINO screening feature_valid must be 1D, got shape {tuple(feature_valid_arr.shape)}"
        )
    feature_count = int(marginal_arr.shape[0])
    if int(class_stats_arr.shape[0]) != feature_count or int(feature_valid_arr.shape[0]) != feature_count:
        raise ValueError("native DIAKRINO screening inputs must agree on feature count")
    return marginal_arr, class_stats_arr, class_valid_arr, feature_valid_arr


def marginal_stats_numpy(X: np.ndarray, missing: np.ndarray) -> np.ndarray:
    X_arr, missing_arr = _validate_marginal_stats_inputs(X, missing)
    observed = ~missing_arr
    values = np.where(observed, X_arr, np.nan)
    count = observed.sum(axis=0).astype(np.float32)
    with np.errstate(all="ignore"):
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
    centered = np.where(observed, values - mean[None, :], np.nan)
    with np.errstate(all="ignore"):
        third = np.nanmean(np.power(centered, 3), axis=0)
        fourth = np.nanmean(np.power(centered, 4), axis=0)
    skew = third / np.maximum(np.power(std, 3), 1e-6)
    kurt = fourth / np.maximum(np.power(std, 4), 1e-6)
    observed_fraction = count / max(1, int(X_arr.shape[0]))
    stats = np.stack([mean, std, skew, kurt, observed_fraction], axis=-1)
    return np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def class_stats_numpy(
    X_support: np.ndarray,
    support_missing: np.ndarray,
    y_support: np.ndarray,
    *,
    max_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_arr, missing_arr, labels = _validate_class_stats_inputs(X_support, support_missing, y_support)
    feature_count = int(X_arr.shape[1])
    class_count = max(1, int(max_classes))
    values = np.where(~missing_arr, X_arr, np.nan)
    marginal = marginal_stats_numpy(X_arr, missing_arr)
    stats = np.zeros((feature_count, class_count, CLASS_STATS_DIM), dtype=np.float32)
    valid = np.zeros((feature_count, class_count), dtype=bool)
    support_rows = max(1, int(labels.shape[0]))
    for cls in range(class_count):
        rows_mask = labels == cls
        rows = values[rows_mask]
        rest = values[~rows_mask]
        class_row_count = int(rows.shape[0])
        if class_row_count <= 0:
            continue
        observed = np.isfinite(rows)
        count = observed.sum(axis=0).astype(np.float32)
        valid[:, cls] = count > 0
        with np.errstate(all="ignore"):
            mean = np.nanmean(rows, axis=0)
            std = np.nanstd(rows, axis=0)
            min_value = np.nanmin(rows, axis=0)
            max_value = np.nanmax(rows, axis=0)
            quantiles = np.nanquantile(rows, QUANTILE_PROBS, axis=0).T
            median = np.nanmedian(rows, axis=0)
            q25 = np.nanquantile(rows, 0.25, axis=0)
            q75 = np.nanquantile(rows, 0.75, axis=0)
            rest_mean = np.nanmean(rest, axis=0) if rest.size else np.zeros(feature_count, dtype=np.float32)
            rest_var = np.nanvar(rest, axis=0) if rest.size else np.zeros(feature_count, dtype=np.float32)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
        min_value = np.nan_to_num(min_value, nan=0.0, posinf=0.0, neginf=0.0)
        max_value = np.nan_to_num(max_value, nan=0.0, posinf=0.0, neginf=0.0)
        quantiles = np.nan_to_num(quantiles, nan=0.0, posinf=0.0, neginf=0.0)
        median = np.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0)
        q25 = np.nan_to_num(q25, nan=0.0, posinf=0.0, neginf=0.0)
        q75 = np.nan_to_num(q75, nan=0.0, posinf=0.0, neginf=0.0)
        rest_mean = np.nan_to_num(rest_mean, nan=0.0, posinf=0.0, neginf=0.0)
        rest_var = np.nan_to_num(rest_var, nan=0.0, posinf=0.0, neginf=0.0)
        fisher = np.square(mean - rest_mean) / np.maximum(np.square(std) + rest_var, 1e-6)
        stats[:, cls, 0] = count
        stats[:, cls, 1] = mean
        stats[:, cls, 2] = std
        stats[:, cls, 3] = min_value
        stats[:, cls, 4] = max_value
        stats[:, cls, 5:15] = quantiles
        stats[:, cls, 15] = 1.0 - (count / max(1, class_row_count))
        stats[:, cls, 16] = float(class_row_count) / float(support_rows)
        stats[:, cls, 17] = fisher
        stats[:, cls, 18] = np.abs(mean - marginal[:, 0])
        stats[:, cls, 19] = np.log(np.maximum(std, 1e-6) / np.maximum(marginal[:, 1], 1e-6))
        stats[:, cls, 20] = count / max(1, class_row_count)
        stats[:, cls, 21] = median
        stats[:, cls, 22] = q75 - q25
        stats[:, cls, 23] = float(class_row_count)
    return (
        np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False),
        valid,
        marginal,
    )


def _valid_feature_zscore(values: np.ndarray, valid: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if not np.any(valid):
        return out
    selected = values[valid].astype(np.float32, copy=False)
    scale = float(np.std(selected))
    if scale <= eps:
        return out
    out[valid] = np.clip((selected - float(np.mean(selected))) / scale, -6.0, 6.0)
    return out


def _valid_feature_rank01(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    indices = np.flatnonzero(valid)
    if indices.size <= 1:
        return out
    order = np.argsort(values[indices], kind="stable")
    ranks = np.empty(indices.size, dtype=np.float32)
    ranks[order] = np.arange(indices.size, dtype=np.float32)
    out[indices] = ranks / max(1, order.size - 1)
    return out


def screening_features_numpy(
    marginal: np.ndarray,
    class_stats: np.ndarray,
    class_stats_valid: np.ndarray,
    feature_valid: np.ndarray,
) -> np.ndarray:
    marginal_arr, class_stats_arr, cls_valid, valid = _validate_screening_inputs(
        marginal,
        class_stats,
        class_stats_valid,
        feature_valid,
    )
    mean = marginal_arr[:, 0]
    std = np.clip(marginal_arr[:, 1], 0.0, None)
    skew = marginal_arr[:, 2]
    kurt = marginal_arr[:, 3]
    observed = np.clip(marginal_arr[:, 4], 0.0, 1.0)
    fisher = np.where(cls_valid, np.clip(class_stats_arr[:, :, 17], 0.0, None), 0.0).max(axis=1)
    max_shift = np.where(cls_valid, np.abs(class_stats_arr[:, :, 18]), 0.0).max(axis=1)
    shift_sum = np.where(cls_valid, np.abs(class_stats_arr[:, :, 18]), 0.0).sum(axis=1)
    mean_shift = shift_sum / np.maximum(cls_valid.sum(axis=1), 1)
    log_std_ratio = np.where(cls_valid, np.abs(class_stats_arr[:, :, 19]), 0.0).max(axis=1)
    priors = np.where(cls_valid, np.clip(class_stats_arr[:, :, 16], 0.0, None), 0.0)
    priors = priors / np.maximum(priors.sum(axis=1, keepdims=True), 1e-6)
    class_balance = -(priors * np.log(np.clip(priors, 1e-6, None))).sum(axis=1) / np.log(
        max(2, class_stats_arr.shape[1])
    )
    log_std = np.log1p(std)
    channels = [
        fisher,
        max_shift,
        mean_shift,
        log_std_ratio,
        np.clip(class_balance, 0.0, 1.0),
        log_std,
        np.abs(mean),
        observed,
        np.abs(skew),
        np.log1p(np.abs(kurt)),
        _valid_feature_zscore(fisher, valid),
        _valid_feature_zscore(max_shift, valid),
        _valid_feature_zscore(mean_shift, valid),
        _valid_feature_zscore(log_std, valid),
        _valid_feature_rank01(fisher, valid),
        _valid_feature_rank01(max_shift, valid),
        _valid_feature_rank01(log_std, valid),
        1.0 - observed,
    ]
    features = np.stack(channels, axis=-1).astype(np.float32, copy=False)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return np.where(valid[:, None], features, 0.0).astype(np.float32, copy=False)


def _variance_order(X_train: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        variance = np.nanvar(np.asarray(X_train, dtype=float), axis=0)
    variance = np.where(np.isfinite(variance), variance, -np.inf)
    return np.argsort(-variance, kind="stable")


def _validate_feature_selection_matrix(X_train: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X_train, dtype=np.float32)
    if X_arr.ndim != 2:
        raise ValueError(f"native DIAKRINO feature selection X_train must be 2D, got shape {tuple(X_arr.shape)}")
    if int(X_arr.shape[0]) <= 0:
        raise ValueError("native DIAKRINO feature selection X_train must contain at least one training row")
    if int(X_arr.shape[1]) <= 0:
        raise ValueError("native DIAKRINO feature selection X_train must contain at least one feature")
    return X_arr


def select_tabentics_diakrino_features(
    X_train: np.ndarray,
    max_features: int,
    *,
    sidecar_path: str | Path | None = None,
    dataset_id: str | None = None,
    score_column: str = "feature_selection_logit",
    calibrate: str = "chunk_zscore",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a feature budget for native DIAKRINO inference.

    Sidecar-backed selection uses chunk-calibrated FS-trunk logits and applies the
    discrete-family skip mask.  If the sidecar or score column is unavailable,
    selection falls back to variance top-k.
    """

    X_train_arr = _validate_feature_selection_matrix(X_train)
    n_features = int(X_train_arr.shape[1])
    budget = min(n_features, int(max(1, max_features)))
    if budget >= n_features:
        return np.arange(n_features, dtype=np.int64), {
            "native_diakrino_feature_cap_policy": "all_features",
            "native_diakrino_feature_score_column": "",
            "native_diakrino_sidecar_loaded": False,
            "native_diakrino_sidecar_used": False,
            "native_diakrino_sidecar_status": "not_needed",
            "native_diakrino_sidecar_reason": "feature_budget_covers_all_features",
        }

    selected: list[int] = []
    sidecar_loaded = False
    sidecar_used = False
    sidecar_status = "not_configured"
    sidecar_reason = "sidecar_path_not_configured"
    if sidecar_path:
        sidecar_status = "not_loaded"
        sidecar_reason = "sidecar_unavailable"
        try:
            from tabnetics.feature_selection.diakrino_sidecar import DiakrinoSidecar

            sidecar_dataset_id = str(dataset_id).strip() if dataset_id is not None else ""
            sidecar = DiakrinoSidecar.load(
                sidecar_path,
                dataset_id=sidecar_dataset_id or None,
            )
            sidecar_loaded = sidecar is not None
            scores = None if sidecar is None else sidecar.scalar_scores(str(score_column), calibrate=str(calibrate))
            if scores is not None and int(scores.shape[0]) >= n_features:
                usable = np.asarray(scores[:n_features], dtype=np.float64).copy()
                skip = sidecar.discrete_skip_mask() if sidecar is not None else None
                if skip is not None and int(skip.shape[0]) >= n_features:
                    usable[np.asarray(skip[:n_features], dtype=bool)] = np.nan
                valid = np.isfinite(usable)
                for idx in np.argsort(-np.where(valid, usable, -np.inf), kind="stable"):
                    if not bool(valid[int(idx)]):
                        break
                    selected.append(int(idx))
                    if len(selected) >= budget:
                        break
                sidecar_used = bool(selected)
            if sidecar_used:
                sidecar_status = "used"
                sidecar_reason = "ok"
            elif sidecar_loaded:
                sidecar_status = "loaded_unusable"
                sidecar_reason = (
                    "score_column_missing_or_short"
                    if scores is None or int(scores.shape[0]) < n_features
                    else "no_finite_scores_after_filtering"
                )
        except Exception as exc:
            sidecar_loaded = False
            sidecar_used = False
            sidecar_status = "load_error"
            sidecar_reason = _format_exception_for_meta(exc)

    if len(selected) < budget:
        used = set(selected)
        for idx in _variance_order(X_train_arr):
            item = int(idx)
            if item in used:
                continue
            selected.append(item)
            if len(selected) >= budget:
                break

    policy = (
        f"sidecar_{calibrate}_{score_column}_then_variance"
        if sidecar_used
        else "variance_topk"
    )
    return np.asarray(selected[:budget], dtype=np.int64), {
        "native_diakrino_feature_cap_policy": policy,
        "native_diakrino_feature_score_column": str(score_column) if sidecar_used else "",
        "native_diakrino_sidecar_loaded": bool(sidecar_loaded),
        "native_diakrino_sidecar_used": bool(sidecar_used),
        "native_diakrino_sidecar_status": str(sidecar_status),
        "native_diakrino_sidecar_reason": str(sidecar_reason),
    }


def resolve_torch_device(requested: str | None = None) -> Any:
    """Resolve a torch device without requiring CUDA."""

    import torch

    key = str(requested or "auto").strip().lower()
    if key in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "gpu":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda" or key.startswith("cuda:"):
        return torch.device(key if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _torch_load_checkpoint(torch_module: Any, checkpoint: Any, *, map_location: Any) -> Any:
    source = str(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
    try:
        return torch_module.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(source, map_location=map_location)


def _load_checkpoint_with_identity(
    torch_module: Any,
    checkpoint: str | Path,
    *,
    map_location: Any,
) -> tuple[Any, str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with Path(checkpoint).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
        handle.seek(0)
        payload = _torch_load_checkpoint(torch_module, handle, map_location=map_location)
    return payload, digest.hexdigest(), int(byte_count)


def tabentics_diakrino_config_from_payload(config_cls: Any, payload: dict[str, Any]) -> Any:
    raw = payload.get("config") or payload.get("model_config") or {}
    if not isinstance(raw, dict):
        return config_cls()
    raw = dict(raw)
    if "fs_refiner_steps" not in raw and "refiner_steps" in raw:
        raw["fs_refiner_steps"] = raw["refiner_steps"]
    if "refiner_steps" not in raw and "fs_refiner_steps" in raw:
        raw["refiner_steps"] = raw["fs_refiner_steps"]
    valid = {field.name for field in fields(config_cls)}
    return config_cls(**{key: value for key, value in raw.items() if key in valid})


def _feature_selector_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = (
        payload.get("feature_selector_config")
        or payload.get("fs_teacher_config")
        or payload.get("model_config")
        or payload.get("config")
        or {}
    )
    return {"model_config": raw} if isinstance(raw, dict) else {}


def _payload_has_config_fields(config_cls: Any, payload: dict[str, Any]) -> bool:
    raw = payload.get("config") or payload.get("model_config") or {}
    if not isinstance(raw, dict):
        return False
    raw = dict(raw)
    if "refiner_steps" not in raw and "fs_refiner_steps" in raw:
        raw["refiner_steps"] = raw["fs_refiner_steps"]
    valid = {field.name for field in fields(config_cls)}
    return any(key in valid for key in raw)


def _head_trust_record_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    direct = payload.get("head_trust_record")
    if isinstance(direct, dict):
        return dict(direct)
    summary = payload.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("head_trust_record"), dict):
        return dict(summary["head_trust_record"])
    return {}


def _canonical_state_sha256(state: Any, *, torch_module: Any) -> str:
    if not isinstance(state, dict) or not state:
        raise ValueError("model_state_dict must be a non-empty mapping")
    digest = hashlib.sha256()
    for raw_name, tensor in sorted(state.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_name, str) or not torch_module.is_tensor(tensor):
            raise ValueError("model_state_dict keys must be strings and values must be tensors")
        value = tensor.detach().cpu().contiguous()
        digest.update(raw_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch_module.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json_object_from_file(path: Path) -> tuple[dict[str, Any] | None, bytes | None, str | None]:
    if not path.is_file():
        return None, None, "missing"
    try:
        raw = path.read_bytes()

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_nonfinite_constant(value: str) -> None:
            raise ValueError(f"invalid JSON constant: {value}")

        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError):
        return None, None, "malformed"
    if not isinstance(parsed, dict):
        return None, raw, "malformed"
    return parsed, raw, None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_classifier_artifact_bundle(
    checkpoint: str | Path,
    *,
    payload: dict[str, Any],
    state: Any,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    torch_module: Any,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    manifest_path = checkpoint_path.parent / CLASSIFIER_ARTIFACT_MANIFEST_NAME
    trust_path = checkpoint_path.parent / CLASSIFIER_TRUST_RECORD_NAME
    failures: list[str] = []

    def require(condition: bool, failure: str) -> None:
        if not condition and failure not in failures:
            failures.append(failure)

    manifest, _manifest_raw, manifest_error = _json_object_from_file(manifest_path)
    trust_record, trust_raw, trust_error = _json_object_from_file(trust_path)
    if manifest_error is not None:
        failures.append(f"artifact_manifest_{manifest_error}")
    if trust_error is not None:
        failures.append(f"classifier_trust_record_{trust_error}")

    try:
        checkpoint_state_sha256 = _canonical_state_sha256(state, torch_module=torch_module)
    except (OSError, RuntimeError, TypeError, ValueError):
        checkpoint_state_sha256 = ""
        failures.append("checkpoint_state_malformed")

    if manifest is None or trust_record is None or trust_raw is None:
        return {
            "verified": False,
            "reason": failures[0],
            "failures": failures,
            "manifest_path": str(manifest_path),
            "trust_record_path": str(trust_path),
            "checkpoint_sha256": str(checkpoint_sha256),
            "checkpoint_bytes": int(checkpoint_bytes),
            "checkpoint_state_sha256": str(checkpoint_state_sha256),
            "trust_record_sha256": "",
            "trust_record": {},
        }

    manifest_checkpoint = manifest.get("checkpoint")
    manifest_trust = manifest.get("trust_record")
    summary = payload.get("summary")
    provenance = payload.get("provenance")
    embedded_trust = payload.get("head_trust_record")
    require(
        manifest.get("schema_version") == CLASSIFIER_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifact_manifest_schema_mismatch",
    )
    require(isinstance(manifest_checkpoint, dict), "artifact_manifest_checkpoint_malformed")
    require(isinstance(manifest_trust, dict), "artifact_manifest_trust_record_malformed")
    require(trust_record.get("schema_version") == CLASSIFIER_TRUST_SCHEMA_VERSION, "trust_record_schema_mismatch")
    require(isinstance(summary, dict), "checkpoint_summary_missing")
    require(isinstance(provenance, dict), "checkpoint_provenance_missing")
    require(isinstance(embedded_trust, dict), "embedded_trust_record_missing")

    manifest_checkpoint = manifest_checkpoint if isinstance(manifest_checkpoint, dict) else {}
    manifest_trust = manifest_trust if isinstance(manifest_trust, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    provenance = provenance if isinstance(provenance, dict) else {}
    embedded_trust = embedded_trust if isinstance(embedded_trust, dict) else {}
    summary_trust = summary.get("head_trust_record")
    summary_provenance = summary.get("checkpoint_provenance")
    summary_split = summary.get("split_record")
    provenance_split = provenance.get("split_record")
    summary_provenance = summary_provenance if isinstance(summary_provenance, dict) else {}
    summary_split = summary_split if isinstance(summary_split, dict) else {}
    provenance_split = provenance_split if isinstance(provenance_split, dict) else {}

    trust_record_sha256 = hashlib.sha256(trust_raw).hexdigest()
    require(manifest_checkpoint.get("path") == checkpoint_path.name, "checkpoint_path_mismatch")
    require(trust_record.get("checkpoint_path") == checkpoint_path.name, "trust_checkpoint_path_mismatch")
    require(manifest_trust.get("path") == trust_path.name, "trust_record_path_mismatch")
    require(
        manifest_checkpoint.get("format") == "classifier" and payload.get("checkpoint_format") == "classifier",
        "checkpoint_format_mismatch",
    )
    require(
        manifest_checkpoint.get("artifact_role") == "final_heldout_evaluated_classifier"
        and payload.get("artifact_role") == "final_heldout_evaluated_classifier"
        and provenance.get("artifact_role") == "final_heldout_evaluated_classifier",
        "checkpoint_artifact_role_mismatch",
    )
    require(
        isinstance(manifest_checkpoint.get("bytes"), int)
        and not isinstance(manifest_checkpoint.get("bytes"), bool)
        and manifest_checkpoint.get("bytes") == checkpoint_bytes
        and isinstance(trust_record.get("checkpoint_bytes"), int)
        and not isinstance(trust_record.get("checkpoint_bytes"), bool)
        and trust_record.get("checkpoint_bytes") == checkpoint_bytes,
        "checkpoint_bytes_mismatch",
    )
    require(
        _is_sha256(checkpoint_sha256)
        and manifest_checkpoint.get("sha256") == checkpoint_sha256
        and trust_record.get("checkpoint_sha256") == checkpoint_sha256,
        "checkpoint_sha256_mismatch",
    )
    require(
        _is_sha256(trust_record_sha256) and manifest_trust.get("sha256") == trust_record_sha256,
        "trust_record_sha256_mismatch",
    )

    external_only = {"checkpoint_path", "checkpoint_sha256", "checkpoint_bytes"}
    external_embedded = {key: value for key, value in trust_record.items() if key not in external_only}
    require(external_embedded == embedded_trust, "embedded_trust_record_mismatch")
    require(summary_trust == embedded_trust, "summary_trust_record_mismatch")
    require(summary_provenance == provenance, "summary_provenance_mismatch")
    require(summary_split == provenance_split, "summary_split_record_mismatch")

    def require_hash_identity(failure: str, *values: Any) -> None:
        require(bool(values) and all(_is_sha256(value) for value in values) and len(set(values)) == 1, failure)

    require_hash_identity(
        "checkpoint_state_sha256_mismatch",
        checkpoint_state_sha256,
        manifest_checkpoint.get("state_sha256"),
        trust_record.get("checkpoint_state_sha256"),
        embedded_trust.get("checkpoint_state_sha256"),
        provenance.get("checkpoint_state_sha256"),
        summary_provenance.get("checkpoint_state_sha256"),
    )
    require_hash_identity(
        "split_assignment_sha256_mismatch",
        manifest.get("split_assignment_sha256"),
        trust_record.get("split_assignment_sha256"),
        embedded_trust.get("split_assignment_sha256"),
        provenance_split.get("assignment_sha256"),
        summary_split.get("assignment_sha256"),
    )
    require_hash_identity(
        "row_set_sha256_mismatch",
        trust_record.get("row_set_sha256"),
        embedded_trust.get("row_set_sha256"),
        provenance.get("heldout_row_set_sha256"),
        summary_provenance.get("heldout_row_set_sha256"),
    )
    require_hash_identity(
        "feature_schema_sha256_mismatch",
        trust_record.get("feature_schema_sha256"),
        embedded_trust.get("feature_schema_sha256"),
        provenance.get("heldout_feature_schema_sha256"),
        summary_provenance.get("heldout_feature_schema_sha256"),
    )
    require_hash_identity(
        "data_config_sha256_mismatch",
        manifest.get("data_config_sha256"),
        provenance.get("data_config_sha256"),
        summary_provenance.get("data_config_sha256"),
    )

    serving_config = trust_record.get("effective_serving_config")
    serving_digest = trust_record.get("effective_serving_config_sha256")
    require(isinstance(serving_config, dict), "effective_serving_config_malformed")
    require(
        isinstance(serving_config, dict)
        and serving_config == embedded_trust.get("effective_serving_config")
        and serving_config == provenance.get("effective_serving_config")
        and serving_config == manifest.get("effective_serving_config")
        and serving_config == payload.get("effective_serving_config"),
        "effective_serving_config_mismatch",
    )
    require_hash_identity(
        "effective_serving_config_sha256_mismatch",
        serving_digest,
        embedded_trust.get("effective_serving_config_sha256"),
        provenance.get("effective_serving_config_sha256"),
        manifest.get("effective_serving_config_sha256"),
        payload.get("effective_serving_config_sha256"),
    )
    require(
        isinstance(serving_config, dict)
        and _is_sha256(serving_digest)
        and _canonical_json_sha256(serving_config) == serving_digest,
        "effective_serving_config_digest_invalid",
    )

    manifest_source_hashes = manifest.get("source_hashes")
    require(isinstance(manifest_source_hashes, dict), "source_hashes_malformed")
    if isinstance(manifest_source_hashes, dict):
        required_source_hashes = {
            "warm_start_fs_teacher_sha256",
            "warm_start_classifier_sha256",
            "resume_checkpoint_sha256",
        }
        require(
            set(manifest_source_hashes) == required_source_hashes
            and all(provenance.get(key) == value for key, value in manifest_source_hashes.items())
            and all(value == "" or _is_sha256(value) for value in manifest_source_hashes.values()),
            "source_hashes_mismatch",
        )

    checks = trust_record.get("checks")
    failed_checks = trust_record.get("failed_checks")
    usable = trust_record.get("usable_by_core_candidate")
    trained_classifier_head = trust_record.get("trained_classifier_head")
    require(isinstance(checks, dict) and bool(checks), "trust_record_checks_malformed")
    require(isinstance(failed_checks, list), "trust_record_failed_checks_malformed")
    require(isinstance(usable, bool), "trust_record_usable_flag_malformed")
    require(isinstance(trained_classifier_head, bool), "trust_record_trained_head_flag_malformed")
    require(trust_record.get("gate") == "heldout_query_head_vs_per_episode_chance", "trust_record_gate_mismatch")
    require(
        trust_record.get("evaluation_scope") == "heldout_dataset_or_world_groups",
        "trust_record_evaluation_scope_mismatch",
    )
    if (
        isinstance(checks, dict)
        and isinstance(failed_checks, list)
        and isinstance(usable, bool)
        and isinstance(trained_classifier_head, bool)
    ):
        required_checks = {
            "trained_classifier_head",
            "heldout_evaluation",
            "group_disjoint",
            "minimum_tasks",
            "minimum_examples",
            "minimum_examples_per_task",
            "minimum_classes",
            "task_bootstrap_contract",
            "ece_bins_contract",
            "accuracy_margin",
            "balanced_accuracy_margin",
            "task_accuracy_ci_margin",
            "task_balanced_accuracy_ci_margin",
            "nll_task_bootstrap_margin",
            "ece_threshold_predeclared",
            "ece_task_bootstrap_upper_bound",
            "support_joint_policy",
            "provenance_hashes_present",
        }
        false_checks = {str(key) for key, value in checks.items() if value is not True}
        listed_failures = [str(value) for value in failed_checks]
        require(
            required_checks.issubset(checks)
            and all(isinstance(value, bool) for value in checks.values()),
            "trust_record_checks_malformed",
        )
        require(
            checks.get("trained_classifier_head") is trained_classifier_head,
            "trust_record_trained_head_flag_inconsistent",
        )
        require(
            all(isinstance(value, str) for value in failed_checks)
            and len(listed_failures) == len(set(listed_failures))
            and set(listed_failures) == false_checks,
            "trust_record_failed_checks_inconsistent",
        )
        require(usable == (not false_checks), "trust_record_usable_flag_inconsistent")
        require(
            (usable and trust_record.get("reason") == "heldout_trust_gate_passed")
            or (
                not usable
                and isinstance(trust_record.get("reason"), str)
                and str(trust_record.get("reason")).startswith("heldout_trust_gate_failed:")
            ),
            "trust_record_reason_inconsistent",
        )

        metrics = trust_record.get("metrics")
        admission_policy = serving_config.get("admission_policy") if isinstance(serving_config, dict) else None
        hard_minimums = admission_policy.get("hard_minimums") if isinstance(admission_policy, dict) else None
        heldout_evaluator = provenance.get("heldout_evaluator")
        bootstrap = metrics.get("task_admission_bootstrap") if isinstance(metrics, dict) else None
        task_metrics = metrics.get("task_metrics") if isinstance(metrics, dict) else None

        def finite_number(value: Any) -> bool:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
            try:
                return bool(np.isfinite(float(value)))
            except (OverflowError, TypeError, ValueError):
                return False

        required_hard_minimums = {
            "tasks": CLASSIFIER_ADMISSION_MIN_TASKS,
            "examples": CLASSIFIER_ADMISSION_MIN_EXAMPLES,
            "examples_per_task": CLASSIFIER_ADMISSION_MIN_EXAMPLES_PER_TASK,
            "classes": CLASSIFIER_ADMISSION_MIN_CLASSES,
            "bootstrap_samples": CLASSIFIER_ADMISSION_BOOTSTRAP_SAMPLES,
            "ece_bins": CLASSIFIER_ADMISSION_ECE_BINS,
        }
        require(isinstance(metrics, dict), "trust_record_metrics_malformed")
        require(isinstance(admission_policy, dict), "admission_policy_malformed")
        require(hard_minimums == required_hard_minimums, "admission_hard_minimums_mismatch")
        require(isinstance(heldout_evaluator, dict), "heldout_evaluator_malformed")
        require(isinstance(bootstrap, dict), "task_admission_bootstrap_malformed")
        require(isinstance(task_metrics, list), "task_metrics_malformed")
        metrics = metrics if isinstance(metrics, dict) else {}
        admission_policy = admission_policy if isinstance(admission_policy, dict) else {}
        heldout_evaluator = heldout_evaluator if isinstance(heldout_evaluator, dict) else {}
        bootstrap = bootstrap if isinstance(bootstrap, dict) else {}
        task_metrics = task_metrics if isinstance(task_metrics, list) else []

        def integer_number(value: Any) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        nll_margin = trust_record.get("nll_improvement_margin")
        max_ece = trust_record.get("max_ece")
        requested_minimums = {
            "minimum_tasks": trust_record.get("requested_minimum_tasks"),
            "minimum_examples": trust_record.get("requested_minimum_examples"),
            "minimum_examples_per_task": trust_record.get("requested_minimum_examples_per_task"),
            "minimum_classes": trust_record.get("requested_minimum_classes"),
        }
        effective_minimums = {
            "minimum_tasks": trust_record.get("minimum_tasks"),
            "minimum_examples": trust_record.get("minimum_examples"),
            "minimum_examples_per_task": trust_record.get("minimum_examples_per_task"),
            "minimum_classes": trust_record.get("minimum_classes"),
        }
        policy_requested = {
            "minimum_tasks": admission_policy.get("minimum_tasks") if isinstance(admission_policy, dict) else None,
            "minimum_examples": admission_policy.get("minimum_examples") if isinstance(admission_policy, dict) else None,
            "minimum_examples_per_task": (
                admission_policy.get("minimum_examples_per_task") if isinstance(admission_policy, dict) else None
            ),
            "minimum_classes": admission_policy.get("minimum_classes") if isinstance(admission_policy, dict) else None,
        }
        evaluator_requested = {
            "minimum_tasks": heldout_evaluator.get("minimum_tasks") if isinstance(heldout_evaluator, dict) else None,
            "minimum_examples": heldout_evaluator.get("minimum_examples") if isinstance(heldout_evaluator, dict) else None,
            "minimum_examples_per_task": (
                heldout_evaluator.get("minimum_examples_per_task") if isinstance(heldout_evaluator, dict) else None
            ),
            "minimum_classes": heldout_evaluator.get("minimum_classes") if isinstance(heldout_evaluator, dict) else None,
        }
        require(
            requested_minimums == policy_requested == evaluator_requested,
            "admission_requested_minimums_mismatch",
        )
        require(
            all(integer_number(value) and int(value) >= 0 for value in requested_minimums.values()),
            "admission_requested_minimums_malformed",
        )
        require(
            all(integer_number(value) and int(value) >= 0 for value in effective_minimums.values()),
            "admission_effective_minimums_malformed",
        )
        expected_effective = {
            "minimum_tasks": max(
                CLASSIFIER_ADMISSION_MIN_TASKS,
                int(requested_minimums["minimum_tasks"])
                if integer_number(requested_minimums["minimum_tasks"])
                else 0,
            ),
            "minimum_examples": max(
                CLASSIFIER_ADMISSION_MIN_EXAMPLES,
                int(requested_minimums["minimum_examples"])
                if integer_number(requested_minimums["minimum_examples"])
                else 0,
            ),
            "minimum_examples_per_task": max(
                CLASSIFIER_ADMISSION_MIN_EXAMPLES_PER_TASK,
                int(requested_minimums["minimum_examples_per_task"])
                if integer_number(requested_minimums["minimum_examples_per_task"])
                else 0,
            ),
            "minimum_classes": max(
                CLASSIFIER_ADMISSION_MIN_CLASSES,
                int(requested_minimums["minimum_classes"])
                if integer_number(requested_minimums["minimum_classes"])
                else 0,
            ),
        }
        require(effective_minimums == expected_effective, "admission_effective_minimums_mismatch")
        require(
            finite_number(nll_margin)
            and float(nll_margin) >= 0.0
            and finite_number(max_ece)
            and 0.0 <= float(max_ece) <= 1.0
            and admission_policy.get("nll_improvement_margin") == nll_margin
            and admission_policy.get("max_ece") == max_ece
            and heldout_evaluator.get("nll_improvement_margin") == nll_margin
            and heldout_evaluator.get("max_ece") == max_ece
            and provenance.get("head_nll_improvement_margin") == nll_margin
            and provenance.get("head_max_ece") == max_ece,
            "admission_thresholds_mismatch",
        )

        bootstrap_valid = bool(
            isinstance(bootstrap, dict)
            and bootstrap.get("schema_version") == "tabentics_diakrino_task_admission_bootstrap_v1"
            and bootstrap.get("resampling_unit") == "heldout_dataset_or_world_group"
            and bootstrap.get("group_weighting") == "equal"
            and bootstrap.get("samples") == CLASSIFIER_ADMISSION_BOOTSTRAP_SAMPLES
            and bootstrap.get("lower_quantile") == 0.025
            and bootstrap.get("upper_quantile") == 0.975
            and isinstance(bootstrap.get("seed"), int)
            and not isinstance(bootstrap.get("seed"), bool)
            and admission_policy.get("bootstrap_samples") == CLASSIFIER_ADMISSION_BOOTSTRAP_SAMPLES
            and admission_policy.get("bootstrap_lower_quantile") == 0.025
            and admission_policy.get("bootstrap_upper_quantile") == 0.975
            and admission_policy.get("ece_bins") == CLASSIFIER_ADMISSION_ECE_BINS
            and metrics.get("ece_bins") == CLASSIFIER_ADMISSION_ECE_BINS
        )
        require(bootstrap_valid, "task_admission_bootstrap_contract_mismatch")

        recomputed_nll_lower = recomputed_ece_upper = None
        task_structure_valid = False
        if isinstance(task_metrics, list) and task_metrics:
            try:
                ordered_tasks = sorted(task_metrics, key=lambda item: str(item["group_id"]))
                group_ids = [str(item["group_id"]) for item in ordered_tasks]
                examples = np.asarray([int(item["examples"]) for item in ordered_tasks], dtype=np.int64)
                nll_values = np.asarray([float(item["nll_improvement_vs_chance"]) for item in ordered_tasks])
                ece_values = np.asarray([float(item["expected_calibration_error"]) for item in ordered_tasks])
                task_structure_valid = bool(
                    len(group_ids) == len(set(group_ids))
                    and np.all(examples > 0)
                    and np.all(np.isfinite(nll_values))
                    and np.all(np.isfinite(ece_values))
                    and np.all((ece_values >= 0.0) & (ece_values <= 1.0))
                    and all(
                        np.isclose(
                            float(item["nll_improvement_vs_chance"]),
                            float(item["chance_negative_log_likelihood"])
                            - float(item["negative_log_likelihood"]),
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                        for item in ordered_tasks
                    )
                )
                rng = np.random.default_rng(int(bootstrap["seed"]))
                indices = rng.integers(
                    0,
                    len(ordered_tasks),
                    size=(CLASSIFIER_ADMISSION_BOOTSTRAP_SAMPLES, len(ordered_tasks)),
                )
                recomputed_nll_lower = float(np.quantile(nll_values[indices].mean(axis=1), 0.025))
                recomputed_ece_upper = float(np.quantile(ece_values[indices].mean(axis=1), 0.975))
                nll_bootstrap = bootstrap["nll_improvement_vs_chance"]
                ece_bootstrap = bootstrap["expected_calibration_error"]
                task_structure_valid = bool(
                    task_structure_valid
                    and np.isclose(nll_bootstrap["task_mean"], nll_values.mean(), rtol=0.0, atol=1.0e-12)
                    and np.isclose(
                        nll_bootstrap["lower_quantile_value"], recomputed_nll_lower, rtol=0.0, atol=1.0e-12
                    )
                    and np.isclose(ece_bootstrap["task_mean"], ece_values.mean(), rtol=0.0, atol=1.0e-12)
                    and np.isclose(
                        ece_bootstrap["upper_quantile_value"], recomputed_ece_upper, rtol=0.0, atol=1.0e-12
                    )
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                task_structure_valid = False
        require(task_structure_valid, "task_admission_statistics_mismatch")

        task_example_values = (
            [int(item["examples"]) for item in task_metrics]
            if task_metrics
            and all(
                isinstance(item, dict)
                and integer_number(item.get("examples"))
                and int(item["examples"]) >= 0
                for item in task_metrics
            )
            else []
        )
        task_count_value = metrics.get("task_count") if integer_number(metrics.get("task_count")) else -1
        example_count_value = metrics.get("example_count") if integer_number(metrics.get("example_count")) else -1
        min_task_examples_value = (
            metrics.get("min_task_examples") if integer_number(metrics.get("min_task_examples")) else -1
        )
        min_classes_value = (
            metrics.get("min_episode_classes") if integer_number(metrics.get("min_episode_classes")) else -1
        )
        nll_margin_value = float(nll_margin) if finite_number(nll_margin) else float("inf")
        max_ece_value = float(max_ece) if finite_number(max_ece) else float("-inf")
        eligibility = {
            "minimum_tasks": bool(
                task_count_value == len(task_metrics)
                and len(task_metrics) >= expected_effective["minimum_tasks"]
            ),
            "minimum_examples": bool(
                task_example_values
                and sum(task_example_values) == example_count_value
                and example_count_value >= expected_effective["minimum_examples"]
            ),
            "minimum_examples_per_task": bool(
                task_example_values
                and min(task_example_values) == min_task_examples_value
                and min_task_examples_value >= expected_effective["minimum_examples_per_task"]
            ),
            "minimum_classes": bool(
                min_classes_value >= expected_effective["minimum_classes"]
            ),
            "task_bootstrap_contract": bootstrap_valid,
            "ece_bins_contract": metrics.get("ece_bins") == CLASSIFIER_ADMISSION_ECE_BINS,
            "nll_task_bootstrap_margin": bool(
                recomputed_nll_lower is not None and float(recomputed_nll_lower) >= nll_margin_value
            ),
            "ece_threshold_predeclared": bool(finite_number(max_ece) and 0.0 <= float(max_ece) <= 1.0),
            "ece_task_bootstrap_upper_bound": bool(
                recomputed_ece_upper is not None
                and finite_number(max_ece)
                and float(recomputed_ece_upper) <= max_ece_value
            ),
        }
        require(
            all(checks.get(key) is value for key, value in eligibility.items()),
            "trust_record_statistical_checks_inconsistent",
        )

    return {
        "verified": not failures,
        "reason": "artifact_trust_bundle_verified" if not failures else failures[0],
        "failures": failures,
        "manifest_path": str(manifest_path),
        "trust_record_path": str(trust_path),
        "checkpoint_sha256": str(checkpoint_sha256),
        "checkpoint_bytes": int(checkpoint_bytes),
        "checkpoint_state_sha256": str(checkpoint_state_sha256),
        "trust_record_sha256": str(trust_record_sha256),
        "trust_record": trust_record,
    }


def _summarize_state_prefixes(keys: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        prefix = str(key).split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _checkpoint_format(payload: dict[str, Any], state: Any) -> str:
    raw = str(payload.get("checkpoint_format") or payload.get("format") or "").strip().lower()
    normalized = raw.replace("-", "_")
    if normalized in {"classifier", "fs_classifier", "tabentics_diakrino_fs_classifier"}:
        return "classifier"
    if normalized in {"fs_teacher", "teacher", "tabentics_diakrino_fs_teacher"}:
        return "fs_teacher"
    if isinstance(state, dict) and any(str(key).startswith("feature_selector.") for key in state):
        return "classifier"
    return "fs_teacher"


def _load_fs_teacher_state(
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
        "checkpoint_comparable_for_classification": False,
        "checkpoint_comparability_reason": "fs_teacher_checkpoint_without_trained_classifier_head",
        "loaded_classifier_head_count": 0,
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


def load_tabentics_diakrino_fs_classifier(
    checkpoint: str | Path,
    *,
    map_location: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load a trained classifier checkpoint or reusable FS-teacher trunk.

    Model classes are imported lazily from the core classification package so
    ordinary CPU-only imports do not require torch.
    """

    import torch
    from tabnetics.classification.tabentics_diakrino_fs_classifier import (
        TabenticsDiakrinoFSClassifier,
        TabenticsDiakrinoFSClassifierConfig,
    )
    from tabnetics.classification.tabentics_diakrino_fs_teacher import (
        TabenticsDiakrinoFSTeacher,
        TabenticsDiakrinoFSTeacherConfig,
    )

    payload, checkpoint_sha256, checkpoint_bytes = _load_checkpoint_with_identity(
        torch,
        checkpoint,
        map_location="cpu",
    )
    if not isinstance(payload, dict):
        payload = {"model_state_dict": payload}
    cfg = tabentics_diakrino_config_from_payload(TabenticsDiakrinoFSClassifierConfig, payload)
    model = TabenticsDiakrinoFSClassifier(cfg)
    state = payload.get("model_state_dict", payload)
    checkpoint_format = _checkpoint_format(payload, state)
    report: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_format": checkpoint_format,
        "checkpoint_comparable_for_classification": False,
        "checkpoint_comparability_reason": "unverified_checkpoint_format",
        "loaded_classifier_head_count": 0,
        "loaded_count": 0,
        "missing_count": 0,
        "unexpected_count": 0,
        "checkpoint_usable_by_core_candidate": False,
        "checkpoint_usability_reason": "classifier_trust_record_missing",
        "checkpoint_trust_bundle_verified": False,
        "checkpoint_trust_bundle_reason": "not_applicable",
        "checkpoint_trust_bundle_failures": [],
        "checkpoint_sha256": str(checkpoint_sha256),
        "checkpoint_bytes": int(checkpoint_bytes),
        "checkpoint_state_sha256": "",
        "classifier_trust_record_path": "",
        "artifact_manifest_path": "",
        "head_trust_record": {},
        "embedded_head_trust_record": _head_trust_record_from_payload(payload),
    }
    if isinstance(state, dict) and checkpoint_format == "classifier":
        trust_bundle = _verify_classifier_artifact_bundle(
            checkpoint,
            payload=payload,
            state=state,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_bytes=checkpoint_bytes,
            torch_module=torch,
        )
        head_trust_record = trust_bundle["trust_record"]
        teacher_cfg = tabentics_diakrino_config_from_payload(
            TabenticsDiakrinoFSTeacherConfig,
            _feature_selector_config_payload(payload),
        )
        model.feature_selector = TabenticsDiakrinoFSTeacher(teacher_cfg)
        model_state = model.state_dict()
        source_keys = {str(key) for key in state}
        target_keys = set(model_state)
        unexpected_keys = sorted(source_keys - target_keys)
        missing_keys = sorted(target_keys - source_keys)
        skipped_shape = sorted(
            str(key)
            for key, value in state.items()
            if str(key) in model_state
            and tuple(model_state[str(key)].shape) != tuple(getattr(value, "shape", ()))
        )
        non_tensor_keys = sorted(
            str(key)
            for key, value in state.items()
            if str(key) in model_state and not torch.is_tensor(value)
        )
        matching = {
            str(key): value
            for key, value in state.items()
            if str(key) in model_state
            and torch.is_tensor(value)
            and tuple(model_state[str(key)].shape) == tuple(value.shape)
        }
        load_missing: list[str] = sorted(target_keys - set(matching))
        load_unexpected: list[str] = []
        if matching:
            result = model.load_state_dict(matching, strict=False)
            load_missing = sorted(str(key) for key in result.missing_keys)
            load_unexpected = sorted(str(key) for key in result.unexpected_keys)
        try:
            loaded_state_sha256 = _canonical_state_sha256(model.state_dict(), torch_module=torch)
        except (OSError, RuntimeError, TypeError, ValueError):
            loaded_state_sha256 = ""
        loaded_state_matches = bool(
            _is_sha256(loaded_state_sha256)
            and loaded_state_sha256 == trust_bundle["checkpoint_state_sha256"]
        )
        loaded_classifier_head_count = sum(
            1 for key in matching if str(key).startswith("class_logit_head.")
        )
        complete_state = bool(
            matching
            and len(matching) == len(model_state)
            and not missing_keys
            and not unexpected_keys
            and not skipped_shape
            and not non_tensor_keys
            and not load_missing
            and not load_unexpected
            and loaded_state_matches
        )
        bundle_verified = bool(trust_bundle["verified"])
        trained_classifier_head = bool(head_trust_record.get("trained_classifier_head", False))
        comparable = bool(bundle_verified and complete_state and trained_classifier_head)
        trusted_for_core = bool(
            comparable and head_trust_record.get("usable_by_core_candidate", False)
        )
        if not bundle_verified:
            comparability_reason = str(trust_bundle["reason"])
        elif not complete_state:
            comparability_reason = "classifier_checkpoint_incomplete_state"
        elif not trained_classifier_head:
            comparability_reason = "classifier_checkpoint_without_trained_head"
        else:
            comparability_reason = "trained_classifier_checkpoint"
        usability_reason = (
            str(head_trust_record.get("reason", "head_trust_record_failed"))
            if comparable
            else comparability_reason
        )
        report.update(
            {
                "loaded_count": int(len(matching)),
                "loaded_classifier_head_count": int(loaded_classifier_head_count),
                "checkpoint_complete_state": bool(complete_state),
                "checkpoint_comparable_for_classification": bool(comparable),
                "checkpoint_comparability_reason": str(comparability_reason),
                "missing_count": int(len(load_missing)),
                "unexpected_count": int(len(unexpected_keys) + len(load_unexpected)),
                "missing_after_partial_load": load_missing,
                "unexpected_checkpoint_keys": unexpected_keys,
                "unexpected_after_partial_load": load_unexpected,
                "skipped_shape": skipped_shape,
                "non_tensor_keys": non_tensor_keys,
                "checkpoint_usable_by_core_candidate": bool(trusted_for_core),
                "checkpoint_usability_reason": str(usability_reason),
                "checkpoint_trust_bundle_verified": bool(bundle_verified),
                "checkpoint_trust_bundle_reason": str(trust_bundle["reason"]),
                "checkpoint_trust_bundle_failures": list(trust_bundle["failures"]),
                "checkpoint_state_sha256": str(trust_bundle["checkpoint_state_sha256"]),
                "checkpoint_loaded_state_sha256": str(loaded_state_sha256),
                "checkpoint_loaded_state_matches": bool(loaded_state_matches),
                "classifier_trust_record_sha256": str(trust_bundle["trust_record_sha256"]),
                "classifier_trust_record_path": str(trust_bundle["trust_record_path"]),
                "artifact_manifest_path": str(trust_bundle["manifest_path"]),
                "head_trust_record": head_trust_record,
            }
        )
    elif isinstance(state, dict):
        if _payload_has_config_fields(TabenticsDiakrinoFSTeacherConfig, payload) or not hasattr(model, "feature_selector"):
            teacher_cfg = tabentics_diakrino_config_from_payload(TabenticsDiakrinoFSTeacherConfig, payload)
            model.feature_selector = TabenticsDiakrinoFSTeacher(teacher_cfg)
        report = _load_fs_teacher_state(model, state, checkpoint=checkpoint, payload=payload)
    else:
        model, teacher_report = TabenticsDiakrinoFSClassifier.from_fs_teacher_checkpoint(
            checkpoint,
            config=cfg,
            map_location="cpu",
        )
        report = {
            **teacher_report,
            "checkpoint_format": "fs_teacher",
            "checkpoint_comparable_for_classification": False,
            "checkpoint_comparability_reason": "fs_teacher_checkpoint_without_trained_classifier_head",
            "loaded_classifier_head_count": 0,
        }
    model.to(map_location)
    model.eval()
    return model, report


def normalize_class_probability_matrix(proba: np.ndarray, *, n_classes: int) -> np.ndarray:
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


def classification_values_from_proba(y_true: np.ndarray, proba: np.ndarray, *, n_classes: int) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    proba = normalize_class_probability_matrix(proba, n_classes=int(n_classes))
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


def apply_temperature_to_proba(proba: np.ndarray, *, temperature: float) -> np.ndarray:
    values = normalize_class_probability_matrix(proba, n_classes=int(np.asarray(proba).shape[1]))
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0 or abs(temp - 1.0) <= 1e-12:
        return values
    logits = np.log(np.clip(values, 1e-12, 1.0)) / temp
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return normalize_class_probability_matrix(exp, n_classes=int(values.shape[1]))


class TabenticsDiakrinoNativeOOMError(RuntimeError):
    """CUDA OOM after the bounded native microbatch retry is exhausted."""

    def __init__(self, *, dataset_name: str, query_start: int, requested_batch_size: int) -> None:
        self.dataset_name = str(dataset_name)
        self.query_start = int(query_start)
        self.requested_batch_size = int(requested_batch_size)
        super().__init__(
            "native Tabnetics Diakrino CUDA OOM at query row "
            f"{self.query_start} after retrying microbatch=1 "
            f"(dataset={self.dataset_name!r}, requested_batch_size={self.requested_batch_size})"
        )


def _score_query_chunks_with_oom_retry(
    *,
    total_rows: int,
    requested_batch_size: int,
    score_range: Any,
    retry_enabled: bool,
    is_retryable_oom: Any,
    clear_device_cache: Any,
    dataset_name: str,
) -> tuple[list[np.ndarray], int, int]:
    """Score ordered chunks and retry only a failed chunk at microbatch one."""

    batch_size = int(max(1, requested_batch_size))
    probabilities: list[np.ndarray] = []
    retry_count = 0
    effective_min_batch_size = min(batch_size, int(total_rows)) if int(total_rows) > 0 else 0
    for start in range(0, int(total_rows), batch_size):
        stop = min(int(total_rows), start + batch_size)
        try:
            probabilities.append(score_range(start, stop))
        except RuntimeError as exc:
            if not bool(retry_enabled) or not bool(is_retryable_oom(exc)):
                raise
            if int(stop - start) <= 1:
                raise TabenticsDiakrinoNativeOOMError(
                    dataset_name=dataset_name,
                    query_start=start,
                    requested_batch_size=batch_size,
                ) from exc
            retry_count += 1
            effective_min_batch_size = 1
            clear_device_cache()
            for row_start in range(start, stop):
                try:
                    probabilities.append(score_range(row_start, row_start + 1))
                except RuntimeError as row_exc:
                    if not bool(is_retryable_oom(row_exc)):
                        raise
                    clear_device_cache()
                    raise TabenticsDiakrinoNativeOOMError(
                        dataset_name=dataset_name,
                        query_start=row_start,
                        requested_batch_size=batch_size,
                    ) from row_exc
    return probabilities, int(retry_count), int(effective_min_batch_size)


def predict_tabentics_diakrino_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    *,
    dataset_name: str,
    seed: int,
    checkpoint: str | Path,
    max_features: int,
    batch_size: int,
    device: str | None = None,
    sidecar_path: str | Path | None = None,
    allow_untrusted_checkpoint: bool = False,
    support_joint_serving_cache: bool = False,
    retry_cuda_oom_microbatch: bool = False,
) -> tuple[np.ndarray, dict[str, Any], Any]:
    """Predict class probabilities for query rows from support ``X_train/y_train``.

    The support labels define the class vocabulary.  This keeps the core DFFS
    candidate path leakage-safe while preserving the BeyondArena metric helper
    through ``run_tabentics_diakrino_native`` below.
    """

    X_train_arr, y_train_arr, X_query_arr = _validate_native_training_query(X_train, y_train, X_query)

    import torch
    from sklearn.preprocessing import LabelEncoder
    from tabnetics.classification.tabentics_diakrino_fs_classifier import TabenticsDiakrinoFSClassifierBatch

    encoder = LabelEncoder().fit(y_train_arr)
    y_train_enc = encoder.transform(y_train_arr).astype(np.int64, copy=False)
    n_classes = int(len(encoder.classes_))
    if n_classes < 2:
        raise ValueError("native Tabnetics Diakrino classification requires at least two classes")

    torch_device = resolve_torch_device(device)
    model, load_report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch_device)
    if not bool(load_report.get("checkpoint_comparable_for_classification", False)):
        raise ValueError(
            "native Tabnetics Diakrino checkpoint is not comparable for classification: "
            f"{load_report.get('checkpoint_comparability_reason', 'unknown')}"
        )
    if not bool(load_report.get("checkpoint_usable_by_core_candidate", False)):
        if not bool(allow_untrusted_checkpoint):
            raise ValueError(
                "native Tabnetics Diakrino checkpoint is not trusted for core candidate use: "
                f"{load_report.get('checkpoint_usability_reason', 'unknown')}"
            )
        # Explicit ablation/diagnostic override (mirrors the sidecar trust-record
        # pattern): execution proceeds but the override is stamped into row meta.
    native_cfg = getattr(model, "config", None)
    model_max_classes = int(max(1, int(getattr(native_cfg, "max_classes", max(n_classes, 2)))))
    if n_classes > model_max_classes:
        raise ValueError(
            "native Tabnetics Diakrino checkpoint supports "
            f"max_classes={model_max_classes}, but dataset has {n_classes} classes"
        )
    model_max_features = int(max(1, int(getattr(native_cfg, "max_feature_tokens", max_features))))
    feature_budget = min(int(max(1, max_features)), model_max_features)
    clip_value = float(getattr(native_cfg, "clip_value", 8.0))

    feature_idx, cap_meta = select_tabentics_diakrino_features(
        X_train_arr,
        feature_budget,
        sidecar_path=sidecar_path,
        dataset_id=dataset_name,
    )
    feature_count = int(feature_idx.size)
    if feature_count <= 0:
        raise ValueError("native Tabnetics Diakrino classification requires at least one feature")
    support_selected = np.asarray(X_train_arr[:, feature_idx], dtype=np.float32)
    query_selected = np.asarray(X_query_arr[:, feature_idx], dtype=np.float32)
    center, scale = fit_robust_scaler(support_selected)
    support_scaled, support_missing = apply_robust_scaler(support_selected, center, scale, clip_value=clip_value)
    query_scaled, query_missing = apply_robust_scaler(query_selected, center, scale, clip_value=clip_value)
    feature_valid = np.ones(feature_count, dtype=bool)
    # Train/serve parity for checkpoints trained with the opt-in v-next-C joint
    # channel: feed the (row-capped) scaled support rows so the selector's joint
    # and query-ICL paths see the same inputs they were trained with.
    use_support_joint = bool(getattr(native_cfg, "use_support_joint_channel", False))
    joint_support_scaled = joint_support_missing = joint_support_labels = None
    if use_support_joint:
        support_rows_total = int(support_scaled.shape[0])
        joint_row_cap = 512
        if support_rows_total > joint_row_cap:
            # Deterministic evenly spaced stride, mirroring the teacher's own
            # in-model joint row subsampling.
            row_index = np.round(np.linspace(0, support_rows_total - 1, num=joint_row_cap)).astype(np.int64)
        else:
            row_index = np.arange(support_rows_total, dtype=np.int64)
        joint_support_scaled = support_scaled[row_index]
        joint_support_missing = support_missing[row_index]
        joint_support_labels = y_train_enc[row_index]
    class_stats, class_valid_by_feature, marginal = class_stats_numpy(
        support_scaled,
        support_missing,
        y_train_enc,
        max_classes=model_max_classes,
    )
    class_valid = np.zeros(int(class_stats.shape[1]), dtype=bool)
    class_valid[np.arange(n_classes, dtype=int)] = True
    class_stats_valid = class_valid_by_feature & feature_valid[:, None] & class_valid[None, :]
    screening = screening_features_numpy(marginal, class_stats, class_stats_valid, feature_valid)

    positions = (
        np.linspace(0.0, 1.0, num=max(1, feature_count), dtype=np.float32)
        if feature_count > 1
        else np.zeros(1, dtype=np.float32)
    )

    def make_batch(start: int, stop: int, *, singleton_static: bool) -> TabenticsDiakrinoFSClassifierBatch:
        row_count = int(stop - start)

        def static_tensor(values: np.ndarray, *, dtype: Any) -> Any:
            batched = values[None, ...] if singleton_static else np.repeat(values[None, ...], row_count, axis=0)
            return torch.as_tensor(batched, dtype=dtype, device=torch_device)

        return TabenticsDiakrinoFSClassifierBatch(
            query_values=torch.as_tensor(query_scaled[start:stop], dtype=torch.float32, device=torch_device),
            query_mask=torch.as_tensor(query_missing[start:stop], dtype=torch.bool, device=torch_device),
            marginal_stats=static_tensor(marginal, dtype=torch.float32),
            class_stats=static_tensor(class_stats, dtype=torch.float32),
            class_stats_valid=static_tensor(class_stats_valid, dtype=torch.bool),
            feature_valid_mask=static_tensor(feature_valid, dtype=torch.bool),
            class_valid=static_tensor(class_valid, dtype=torch.bool),
            feature_indices=static_tensor(feature_idx, dtype=torch.long),
            feature_positions=static_tensor(positions, dtype=torch.float32),
            screening_features_input=static_tensor(screening, dtype=torch.float32),
            support_values=(
                None if joint_support_scaled is None else static_tensor(joint_support_scaled, dtype=torch.float32)
            ),
            support_missing=(
                None if joint_support_missing is None else static_tensor(joint_support_missing, dtype=torch.bool)
            ),
            support_labels=(
                None if joint_support_labels is None else static_tensor(joint_support_labels, dtype=torch.long)
            ),
            support_row_valid=(
                None
                if joint_support_labels is None
                else static_tensor(joint_support_labels >= 0, dtype=torch.bool)
            ),
        )

    serving_context = None
    cache_used = bool(support_joint_serving_cache and int(query_scaled.shape[0]) > 0)

    def is_cuda_oom(exc: RuntimeError) -> bool:
        oom_type = getattr(torch, "OutOfMemoryError", ())
        return bool(
            torch_device.type == "cuda"
            and ((oom_type and isinstance(exc, oom_type)) or "out of memory" in str(exc).lower())
        )

    with torch.no_grad():
        if cache_used:
            try:
                serving_context = model.prepare_support_context(make_batch(0, 1, singleton_static=True))
            except RuntimeError as exc:
                if not bool(retry_cuda_oom_microbatch) or not is_cuda_oom(exc):
                    raise
                torch.cuda.empty_cache()
                raise TabenticsDiakrinoNativeOOMError(
                    dataset_name=dataset_name,
                    query_start=0,
                    requested_batch_size=int(max(1, batch_size)),
                ) from exc

        def score_range(start: int, stop: int) -> np.ndarray:
            if serving_context is not None:
                query_values = torch.as_tensor(query_scaled[start:stop], dtype=torch.float32, device=torch_device)
                query_mask = torch.as_tensor(query_missing[start:stop], dtype=torch.bool, device=torch_device)
                logits = model.forward_from_support_context(
                    query_values, query_mask, serving_context
                ).class_logits[:, :n_classes]
            else:
                logits = model(make_batch(start, stop, singleton_static=False)).class_logits[:, :n_classes]
            return torch.softmax(logits, dim=-1).detach().cpu().numpy()

        probabilities, oom_retry_count, effective_min_batch_size = _score_query_chunks_with_oom_retry(
            total_rows=int(query_scaled.shape[0]),
            requested_batch_size=int(batch_size),
            score_range=score_range,
            retry_enabled=bool(retry_cuda_oom_microbatch),
            is_retryable_oom=is_cuda_oom,
            clear_device_cache=torch.cuda.empty_cache,
            dataset_name=dataset_name,
        )
    proba_all = np.concatenate(probabilities, axis=0) if probabilities else np.empty((0, n_classes), dtype=float)
    proba_all = normalize_class_probability_matrix(proba_all, n_classes=n_classes)
    meta = {
        "model_name": "TabenticsDiakrinoFSClassifier",
        "native_diakrino_support_joint_channel": bool(use_support_joint),
        "native_diakrino_joint_support_rows_total": int(support_scaled.shape[0]) if use_support_joint else 0,
        "native_diakrino_joint_support_rows_used": (
            0 if joint_support_scaled is None else int(joint_support_scaled.shape[0])
        ),
        "native_diakrino_untrusted_checkpoint_override": bool(
            allow_untrusted_checkpoint
            and not bool(load_report.get("checkpoint_usable_by_core_candidate", False))
        ),
        "native_diakrino_checkpoint": str(checkpoint),
        "native_diakrino_checkpoint_format": str(load_report.get("checkpoint_format", "")),
        "native_diakrino_checkpoint_comparable": bool(
            load_report.get("checkpoint_comparable_for_classification", False)
        ),
        "native_diakrino_checkpoint_comparability_reason": str(
            load_report.get("checkpoint_comparability_reason", "")
        ),
        "native_diakrino_loaded_classifier_head_count": int(
            load_report.get("loaded_classifier_head_count", 0) or 0
        ),
        "native_diakrino_checkpoint_usable_by_core_candidate": bool(
            load_report.get("checkpoint_usable_by_core_candidate", False)
        ),
        "native_diakrino_checkpoint_usability_reason": str(
            load_report.get("checkpoint_usability_reason", "")
        ),
        "native_diakrino_checkpoint_trust_bundle_verified": bool(
            load_report.get("checkpoint_trust_bundle_verified", False)
        ),
        "native_diakrino_checkpoint_trust_bundle_reason": str(
            load_report.get("checkpoint_trust_bundle_reason", "")
        ),
        "native_diakrino_checkpoint_complete_state": bool(
            load_report.get("checkpoint_complete_state", False)
        ),
        "native_diakrino_checkpoint_sha256": str(load_report.get("checkpoint_sha256", "")),
        "native_diakrino_checkpoint_state_sha256": str(
            load_report.get("checkpoint_state_sha256", "")
        ),
        "native_diakrino_classifier_trust_record_sha256": str(
            load_report.get("classifier_trust_record_sha256", "")
        ),
        "native_diakrino_loaded_count": int(load_report.get("loaded_count", 0) or 0),
        "native_diakrino_max_features": int(max_features),
        "native_diakrino_checkpoint_max_feature_tokens": int(model_max_features),
        "native_diakrino_used_features": int(feature_count),
        "native_diakrino_batch_size": int(batch_size),
        "native_diakrino_support_joint_serving_cache_requested": bool(support_joint_serving_cache),
        "native_diakrino_support_joint_serving_cache_used": bool(cache_used),
        "native_diakrino_cuda_oom_retry_enabled": bool(retry_cuda_oom_microbatch),
        "native_diakrino_cuda_oom_retry_count": int(oom_retry_count),
        "native_diakrino_effective_min_batch_size": int(effective_min_batch_size),
        "native_diakrino_device": str(torch_device),
        "n_classes": int(n_classes),
        "class_labels": "|".join(str(label) for label in encoder.classes_.tolist()),
        **cap_meta,
    }
    return proba_all, meta, encoder


class TabenticsDiakrinoNativeClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible opt-in wrapper for the native Tabnetics Diakrino classifier."""

    def __init__(
        self,
        *,
        checkpoint: str,
        max_features: int = 256,
        batch_size: int = 32,
        device: str = "auto",
        sidecar_path: str = "",
        dataset_id: str = "",
        calibrate_probabilities: bool = True,
        calibration_fraction: float = 0.20,
        calibration_min_samples: int = 12,
        temperature_grid: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
        support_joint_serving_cache: bool = False,
        retry_cuda_oom_microbatch: bool = False,
        random_state: int = 0,
    ) -> None:
        self.checkpoint = checkpoint
        self.max_features = max_features
        self.batch_size = batch_size
        self.device = device
        self.sidecar_path = sidecar_path
        self.dataset_id = dataset_id
        self.calibrate_probabilities = calibrate_probabilities
        self.calibration_fraction = calibration_fraction
        self.calibration_min_samples = calibration_min_samples
        self.temperature_grid = temperature_grid
        self.support_joint_serving_cache = support_joint_serving_cache
        self.retry_cuda_oom_microbatch = retry_cuda_oom_microbatch
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabenticsDiakrinoNativeClassifier":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {tuple(X_arr.shape)}")
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {tuple(y_arr.shape)}")
        if int(X_arr.shape[0]) != int(y_arr.shape[0]):
            raise ValueError(
                "X and y must contain the same number of rows; "
                f"got {int(X_arr.shape[0])} and {int(y_arr.shape[0])}"
            )
        if int(X_arr.shape[0]) <= 0:
            raise ValueError("Tabnetics Diakrino requires at least one training row")
        if int(X_arr.shape[1]) <= 0:
            raise ValueError("Tabnetics Diakrino requires at least one feature")
        if int(np.unique(y_arr).size) < 2:
            raise ValueError("Tabnetics Diakrino requires at least two classes")

        from sklearn.metrics import log_loss
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder().fit(y_arr)
        y_enc = encoder.transform(y_arr).astype(np.int64, copy=False)
        self.classes_ = np.asarray(encoder.classes_)
        self.n_features_in_ = int(X_arr.shape[1])
        self._X_support = X_arr
        self._y_support = y_arr.copy()
        self.temperature_ = 1.0
        self.calibration_meta_ = {
            "native_diakrino_probability_calibration": "disabled",
            "native_diakrino_temperature": 1.0,
        }

        counts = np.bincount(y_enc, minlength=int(len(self.classes_)))
        can_calibrate = (
            bool(self.calibrate_probabilities)
            and int(X_arr.shape[0]) >= int(max(2, self.calibration_min_samples))
            and counts.size >= 2
            and int(np.min(counts)) >= 2
            and float(self.calibration_fraction) > 0.0
        )
        if can_calibrate:
            try:
                train_idx, cal_idx = train_test_split(
                    np.arange(int(X_arr.shape[0])),
                    test_size=float(np.clip(self.calibration_fraction, 0.05, 0.50)),
                    random_state=int(self.random_state),
                    stratify=y_enc,
                )
                if cal_idx.size >= 2 and np.unique(y_enc[cal_idx]).size >= 2:
                    proba, _, cal_encoder = predict_tabentics_diakrino_proba(
                        X_arr[train_idx],
                        y_arr[train_idx],
                        X_arr[cal_idx],
                        dataset_name=str(self.dataset_id or ""),
                        seed=int(self.random_state),
                        checkpoint=str(self.checkpoint),
                        max_features=int(self.max_features),
                        batch_size=int(self.batch_size),
                        device=str(self.device),
                        sidecar_path=str(self.sidecar_path or ""),
                        support_joint_serving_cache=bool(self.support_joint_serving_cache),
                        retry_cuda_oom_microbatch=bool(self.retry_cuda_oom_microbatch),
                    )
                    y_cal = cal_encoder.transform(y_arr[cal_idx]).astype(int, copy=False)
                    best_temp = 1.0
                    best_loss = float("inf")
                    labels = np.arange(int(proba.shape[1]))
                    for temp in tuple(self.temperature_grid or (1.0,)):
                        calibrated = apply_temperature_to_proba(proba, temperature=float(temp))
                        loss = float(log_loss(y_cal, calibrated, labels=labels))
                        if np.isfinite(loss) and loss < best_loss:
                            best_loss = loss
                            best_temp = float(temp)
                    self.temperature_ = float(best_temp)
                    self.calibration_meta_ = {
                        "native_diakrino_probability_calibration": "temperature_holdout",
                        "native_diakrino_temperature": float(best_temp),
                        "native_diakrino_calibration_log_loss": float(best_loss),
                        "native_diakrino_calibration_samples": int(cal_idx.size),
                    }
            except Exception as exc:
                self.calibration_meta_ = {
                    "native_diakrino_probability_calibration": "failed",
                    "native_diakrino_temperature": 1.0,
                    "native_diakrino_calibration_error": _format_exception_for_meta(exc),
                }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_X_support"):
            raise AttributeError("TabenticsDiakrinoNativeClassifier is not fitted.")
        X_query = np.asarray(X, dtype=np.float32)
        if X_query.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {tuple(X_query.shape)}")
        if int(X_query.shape[1]) != int(getattr(self, "n_features_in_", X_query.shape[1])):
            raise ValueError(
                "Tabnetics Diakrino predict_proba expected "
                f"{int(self.n_features_in_)} features, got {int(X_query.shape[1])}"
            )
        proba, meta, _encoder = predict_tabentics_diakrino_proba(
            self._X_support,
            self._y_support,
            X_query,
            dataset_name=str(self.dataset_id or ""),
            seed=int(self.random_state),
            checkpoint=str(self.checkpoint),
            max_features=int(self.max_features),
            batch_size=int(self.batch_size),
            device=str(self.device),
            sidecar_path=str(self.sidecar_path or ""),
            support_joint_serving_cache=bool(self.support_joint_serving_cache),
            retry_cuda_oom_microbatch=bool(self.retry_cuda_oom_microbatch),
        )
        self.native_diakrino_meta_ = {**meta, **dict(getattr(self, "calibration_meta_", {}) or {})}
        return apply_temperature_to_proba(proba, temperature=float(getattr(self, "temperature_", 1.0)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.asarray(self.classes_)[np.argmax(proba, axis=1)]


def _format_exception_for_meta(exc: BaseException) -> str:
    detail = str(exc).strip()
    name = type(exc).__name__
    return name if not detail else f"{name}: {detail}"


def run_tabentics_diakrino_native(
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
    device: str | None = None,
    sidecar_path: str | Path | None = None,
    allow_untrusted_checkpoint: bool = False,
    support_joint_serving_cache: bool = False,
    retry_cuda_oom_microbatch: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    X_train_arr, y_train_arr, X_test_arr = _validate_native_training_query(X_train, y_train, X_test)
    y_test_arr = np.asarray(y_test)
    if y_test_arr.ndim != 1:
        raise ValueError(f"y_test must be 1D, got shape {tuple(y_test_arr.shape)}")
    if int(y_test_arr.shape[0]) != int(X_test_arr.shape[0]):
        raise ValueError(
            "X_test and y_test must contain the same number of rows; "
            f"got {int(X_test_arr.shape[0])} and {int(y_test_arr.shape[0])}"
        )
    from sklearn.preprocessing import LabelEncoder

    metric_encoder = LabelEncoder().fit(y_train_arr)
    try:
        y_test_enc = metric_encoder.transform(y_test_arr).astype(np.int64, copy=False)
    except ValueError as exc:
        raise ValueError("native Tabnetics Diakrino metric labels must all be present in y_train") from exc

    proba_all, meta, encoder = predict_tabentics_diakrino_proba(
        X_train_arr,
        y_train_arr,
        X_test_arr,
        dataset_name=dataset_name,
        seed=seed,
        checkpoint=checkpoint,
        max_features=max_features,
        batch_size=batch_size,
        device=device,
        sidecar_path=sidecar_path,
        allow_untrusted_checkpoint=allow_untrusted_checkpoint,
        support_joint_serving_cache=support_joint_serving_cache,
        retry_cuda_oom_microbatch=retry_cuda_oom_microbatch,
    )
    if tuple(encoder.classes_.tolist()) != tuple(metric_encoder.classes_.tolist()):
        y_test_enc = encoder.transform(y_test_arr).astype(np.int64, copy=False)
    values = classification_values_from_proba(y_test_enc, proba_all, n_classes=int(proba_all.shape[1]))
    return values, meta

"""Runtime auto-router for tabnetics pipeline configuration.

The packaged V25 router uses only dataset-computable descriptors plus candidate
action encodings. It does not consume validation tiers, hard-split labels, or
holdout outcomes at prediction time.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import joblib
import numpy as np
from scipy.stats import entropy as _shannon_entropy

try:
    from tabnetics.datasets.meta_features import extract_meta_features
except Exception:  # pragma: no cover - import fallback for editable edge cases
    from tabnetics.datasets.meta_features import extract_meta_features  # type: ignore


AUTO_ROUTER_ARTIFACT_VERSION = "v25_calibrated_score_router"
SCORE_ROUTER_ARTIFACT_TYPE = "score_expanded_router_v1"

_CHEAP_STAT_KEYS: tuple[str, ...] = (
    "log_n",
    "log_p",
    "log_p_over_n",
    "class_count",
    "class_balance_entropy",
    "correlation_spectrum_decay",
    "heaping_fraction",
    "mean_feature_std",
    "std_feature_std",
    "mean_feature_skew",
    "std_feature_skew",
    "mean_feature_kurtosis",
    "std_feature_kurtosis",
    "mean_missing_fraction",
    "mean_univariate_auc",
    "std_univariate_auc",
    "mean_feature_range",
    "std_feature_range",
    "fraction_binary_features",
)

_CLASSIFICATION_ALIAS_MAP: Dict[str, str] = {
    "selection_mode": "classification_selection_mode",
    "backend": "classification_backend",
    "model_candidates": "model_candidates",
    "exclude_model_candidates": "exclude_model_candidates",
    "oracle_k": "classifier_oracle_k",
    "oracle_weighting_mode": "classifier_oracle_weighting_mode",
    "runtime_containment_enabled": "model_cv_runtime_containment_enabled",
    "runtime_max_candidates": "model_cv_runtime_max_candidates",
    "runtime_high_p_over_n_threshold": "model_cv_runtime_high_p_over_n_threshold",
    "runtime_high_class_threshold": "model_cv_runtime_high_class_threshold",
    "runtime_min_class_count_threshold": "model_cv_runtime_min_class_count_threshold",
    "lr_max_iter": "model_cv_lr_max_iter",
    "use_hybrid_score": "model_cv_use_hybrid_score",
    "hybrid_balanced_weight": "model_cv_balanced_weight",
    "hybrid_macro_f1_weight": "model_cv_macro_f1_weight",
    "conformal_enabled": "classifier_conformal_enabled",
    "conformal_alpha": "classifier_conformal_alpha",
    "conformal_calibration_fraction": "classifier_conformal_calibration_fraction",
    "conformal_min_calibration": "classifier_conformal_min_calibration",
    "conformal_output_sets": "classifier_conformal_output_sets",
    "conformal_method": "classifier_conformal_method",
    "flaml_time_budget": "flaml_time_budget",
    "optuna_time_budget": "optuna_time_budget",
    "optuna_n_trials": "optuna_n_trials",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if not np.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _parse_structured_text(text: str) -> Any:
    candidate = str(text or "").strip()
    if not candidate:
        return candidate
    if candidate.startswith("{") or candidate.startswith("[") or candidate.startswith("("):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except Exception:
                continue
    return candidate


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        rounded = round(float(value))
        if abs(float(value) - rounded) < 1e-12:
            return int(rounded)
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lower = text.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        if lower in {"none", "null", "nan"}:
            return None
        parsed = _parse_structured_text(text)
        if parsed is not text:
            return _clean_scalar(parsed)
        return text
    if isinstance(value, (list, tuple)):
        return [_clean_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _clean_scalar(v) for k, v in value.items()}
    return value


def _sanitize_matrix(X: Any) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.size == 0:
        return X_arr.astype(float, copy=False)
    finite = np.isfinite(X_arr)
    if finite.all():
        return X_arr.astype(float, copy=False)
    cleaned = X_arr.astype(float, copy=True)
    masked = np.where(finite, cleaned, np.nan)
    with np.errstate(all="ignore"):
        fill = np.nanmedian(masked, axis=0)
    fill = np.nan_to_num(fill, nan=0.0, posinf=0.0, neginf=0.0)
    rows, cols = np.where(~finite)
    cleaned[rows, cols] = fill[cols]
    return np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)


def _correlation_spectrum_decay(X: np.ndarray) -> float:
    n, p = X.shape
    if p < 3 or n < 2:
        return 0.0
    rng = np.random.default_rng(42)
    max_cols = 200
    X_sub = X[:, rng.choice(p, max_cols, replace=False)] if p > max_cols else X
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(X_sub, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    tri = np.triu_indices_from(corr, k=1)
    spectrum = np.sort(np.abs(corr[tri]))[::-1]
    if spectrum.size < 3:
        return 0.0
    xs = np.arange(1, spectrum.size + 1, dtype=np.float64)
    ys = np.log(np.clip(spectrum, 1e-8, None))
    coeffs = np.polyfit(xs, ys, 1)
    return float(max(-coeffs[0], 0.0))


def _univariate_auc_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    _n, p = X.shape
    if classes.size < 2:
        return np.full(p, 0.5, dtype=np.float64)
    if classes.size == 2:
        mask_pos = y == classes[1]
        mask_neg = ~mask_pos
        n_pos = int(mask_pos.sum())
        n_neg = int(mask_neg.sum())
        if n_pos == 0 or n_neg == 0:
            return np.full(p, 0.5, dtype=np.float64)
        aucs = np.empty(p, dtype=np.float64)
        for j in range(p):
            ranks = np.argsort(np.argsort(X[:, j])).astype(np.float64) + 1.0
            u_stat = float(ranks[mask_pos].sum()) - n_pos * (n_pos + 1.0) / 2.0
            auc = u_stat / float(n_pos * n_neg)
            aucs[j] = max(float(auc), float(1.0 - auc))
        return aucs
    all_aucs = np.zeros(p, dtype=np.float64)
    for class_id in classes:
        binary_y = (y == class_id).astype(np.int64)
        all_aucs += _univariate_auc_scores(X, binary_y)
    return all_aucs / float(classes.size)


def _feature_cardinality_fractions(X: np.ndarray) -> tuple[float, float]:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0:
        return 0.0, 0.0
    unique_counts = []
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = col[np.isfinite(col)]
        unique_counts.append(int(np.unique(valid).size) if valid.size else 0)
    counts = np.asarray(unique_counts, dtype=float)
    return float(np.mean((counts > 2) & (counts <= 10))), float(np.mean(counts >= 50))


def _entropy_from_hist(values: np.ndarray, bins: int = 10) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    hist, _ = np.histogram(arr, bins=min(int(bins), max(int(arr.size // 2), 2)))
    probs = hist.astype(float)
    total = float(np.sum(probs))
    if total <= 0:
        return 0.0
    probs = probs / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs))) if probs.size else 0.0


def _feature_entropy_summary(X: np.ndarray) -> tuple[float, float]:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0:
        return 0.0, 0.0
    values = np.asarray([_entropy_from_hist(X[:, j]) for j in range(X.shape[1])], dtype=float)
    return float(np.mean(values)), float(np.std(values))


def _class_balance_stats(y: np.ndarray) -> tuple[float, float, float, float]:
    classes, counts = np.unique(np.asarray(y).ravel(), return_counts=True)
    if counts.size == 0:
        return 1.0, 0.0, 0.0, 0.0
    total = float(np.sum(counts))
    probs = counts.astype(float) / max(total, 1.0)
    class_balance_entropy = (
        float(_shannon_entropy(probs) / np.log(classes.size))
        if classes.size > 1
        else 0.0
    )
    balance_ratio = float(np.max(counts) / max(int(np.min(counts)), 1))
    gini = float(1.0 - np.sum(np.square(probs)))
    entropy_y = float(-np.sum(np.where(probs > 0, probs * np.log(probs), 0.0)))
    return balance_ratio, gini, entropy_y, class_balance_entropy


def _cheap_descriptor(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    n, p = X.shape
    y_arr = np.asarray(y).ravel()
    classes = np.unique(y_arr)
    _balance_ratio, _gini, _entropy_y, class_balance_entropy = _class_balance_stats(y_arr)
    X_filled = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    feature_stds = np.nan_to_num(np.nanstd(X, axis=0, ddof=1), nan=0.0)
    centered = X_filled - np.nanmean(X_filled, axis=0, keepdims=True)
    safe_std = np.maximum(feature_stds, 1e-8)
    z = centered / safe_std[None, :]
    feature_skew = np.nan_to_num(np.nanmean(z**3, axis=0), nan=0.0)
    feature_kurtosis = np.nan_to_num(np.nanmean(z**4, axis=0) - 3.0, nan=0.0)
    missing_per_feature = np.mean(~np.isfinite(np.asarray(X, dtype=float)), axis=0) if X.size else np.zeros(p)
    feature_range = np.nan_to_num(np.nanmax(X, axis=0) - np.nanmin(X, axis=0), nan=0.0) if n > 0 else np.zeros(p)
    log_range = np.log1p(np.clip(feature_range, 0.0, None))
    unique_counts = np.asarray([np.unique(X_filled[:, j]).size for j in range(p)], dtype=float) if p else np.zeros(0)
    fraction_binary = float(np.mean(unique_counts <= 2)) if unique_counts.size else 0.0
    heaping_fraction = 0.0
    if p > 0:
        heaped = []
        for j in range(p):
            valid = np.isfinite(X[:, j])
            heaped.append(bool(np.any(valid) and np.allclose(X_filled[valid, j], np.round(X_filled[valid, j]))))
        heaping_fraction = float(np.mean(heaped))
    aucs = _univariate_auc_scores(X_filled, y_arr)
    return {
        "log_n": float(np.log(max(n, 1))),
        "log_p": float(np.log(max(p, 1))),
        "log_p_over_n": float(np.log(max(p, 1) / max(n, 1))),
        "class_count": float(classes.size),
        "class_balance_entropy": float(class_balance_entropy),
        "correlation_spectrum_decay": _correlation_spectrum_decay(X_filled),
        "heaping_fraction": float(heaping_fraction),
        "mean_feature_std": float(np.mean(feature_stds)) if feature_stds.size else 0.0,
        "std_feature_std": float(np.std(feature_stds)) if feature_stds.size else 0.0,
        "mean_feature_skew": float(np.mean(feature_skew)) if feature_skew.size else 0.0,
        "std_feature_skew": float(np.std(feature_skew)) if feature_skew.size else 0.0,
        "mean_feature_kurtosis": float(np.clip(np.mean(feature_kurtosis), -100, 100)) if feature_kurtosis.size else 0.0,
        "std_feature_kurtosis": float(np.clip(np.std(feature_kurtosis), 0, 100)) if feature_kurtosis.size else 0.0,
        "mean_missing_fraction": float(np.mean(missing_per_feature)) if missing_per_feature.size else 0.0,
        "mean_univariate_auc": float(np.mean(aucs)) if aucs.size else 0.5,
        "std_univariate_auc": float(np.std(aucs)) if aucs.size else 0.0,
        "mean_feature_range": float(np.mean(log_range)) if log_range.size else 0.0,
        "std_feature_range": float(np.std(log_range)) if log_range.size else 0.0,
        "fraction_binary_features": fraction_binary,
    }


def compute_dataset_descriptor(
    X: Any,
    y: Any,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the dataset-only descriptor used by the packaged auto-router."""

    meta = dict(metadata or {})
    X_arr = _sanitize_matrix(X)
    y_arr = np.asarray(y).ravel()
    n_samples, n_features = X_arr.shape
    cheap = _cheap_descriptor(X_arr, y_arr)
    expanded = extract_meta_features(
        X_arr,
        y_arr,
        expanded=True,
        skip_distance_matrix=bool(n_features > 200 or n_samples > 2000),
    )
    balance_ratio, gini, entropy_y, _balance_entropy = _class_balance_stats(y_arr)
    lowcard, highcard = _feature_cardinality_fractions(X_arr)
    feat_ent_mean, feat_ent_std = _feature_entropy_summary(X_arr)
    variances = np.nan_to_num(np.nanvar(X_arr, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    signal_fraction = _safe_float(expanded.get("signal_eigenvalue_fraction", 0.0), 0.0)
    feature_vector = {
        **cheap,
        "n": float(n_samples),
        "p": float(n_features),
        "p_over_n": float(n_features / max(n_samples, 1)),
        "class_balance_ratio": float(balance_ratio),
        "class_gini_impurity": float(gini),
        "fraction_lowcard_features": float(lowcard),
        "fraction_highcard_features": float(highcard),
        "mutual_info_mean": 0.0,
        "mutual_info_std": 0.0,
        "feature_entropy_mean": float(feat_ent_mean),
        "feature_entropy_std": float(feat_ent_std),
        "joint_entropy_y": float(entropy_y),
        "noise_signal_ratio": float(max(0.0, 1.0 - signal_fraction)),
        "max_feature_variance": float(np.max(variances) if variances.size else 0.0),
        "fisher_f1": _safe_float(expanded.get("fisher_f1", 0.0)),
        "f2_overlap": _safe_float(expanded.get("f2_overlap", 0.0)),
        "n1_borderline": _safe_float(expanded.get("n1_borderline", 0.0)),
        "n2_nn_ratio": _safe_float(expanded.get("n2_nn_ratio", 0.0)),
        "lsc": _safe_float(expanded.get("lsc", 0.0)),
        "t4_pca_ratio": _safe_float(expanded.get("t4_pca_ratio", 0.0)),
        "intrinsic_dim": _safe_float(expanded.get("intrinsic_dim", 0.0)),
        "correlation_alpha": _safe_float(expanded.get("correlation_alpha", 0.0)),
        "signal_eigenvalue_fraction": float(signal_fraction),
        "indicator_n_below_50": float(n_samples < 50),
        "indicator_p_above_5000": float(n_features > 5000),
        "indicator_p_over_n_over_10": float((n_features / max(n_samples, 1)) > 10.0),
    }
    return {
        "dataset_id": str(meta.get("dataset_id", "") or ""),
        "dataset_name": str(meta.get("dataset_name", "") or ""),
        "feature_vector": {key: float(_safe_float(value, 0.0)) for key, value in feature_vector.items()},
    }


@dataclass
class AutoRouterOutput:
    """Auto-router decision emitted by the packaged V25 score model."""

    enabled_methods: list[str]
    config_overrides: Dict[str, Any]
    classification_overrides: Dict[str, Any]
    confidence: float
    head_confidences: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "auto_router_enabled": True,
            "auto_router_used": True,
            "auto_router_version": str(AUTO_ROUTER_ARTIFACT_VERSION),
            "auto_router_selected_candidate_id": str(self.metadata.get("selected_candidate_id", "")),
            "auto_router_default_candidate_id": str(self.metadata.get("default_candidate_id", "")),
            "auto_router_confidence": float(self.confidence),
            "auto_router_predicted_balanced_accuracy": float(
                self.metadata.get("predicted_balanced_accuracy", self.head_confidences.get("balanced_accuracy", 0.0))
            ),
            "auto_router_predicted_macro_f1": float(
                self.metadata.get("predicted_macro_f1", self.head_confidences.get("macro_f1", 0.0))
            ),
            "auto_router_predicted_utility": float(self.metadata.get("predicted_utility", 0.0)),
            "auto_router_calibrated_utility": float(self.metadata.get("calibrated_utility", 0.0)),
            "auto_router_utility_margin": float(self.metadata.get("utility_margin", 0.0)),
            "auto_router_default_utility_margin": float(self.metadata.get("default_utility_margin", 0.0)),
            "auto_router_beats_default_probability": float(
                self.metadata.get("beats_default_probability", 0.0)
            ),
            "auto_router_policy_defaulted": bool(self.metadata.get("policy_defaulted", False)),
            "auto_router_method_count": int(len(self.enabled_methods or [])),
            "auto_router_enabled_methods": list(self.enabled_methods or []),
            "auto_router_config_overrides": dict(self.config_overrides or {}),
            "auto_router_classification_overrides": dict(self.classification_overrides or {}),
            "auto_router_ranked_candidates": list(self.metadata.get("ranked_candidates", []) or []),
        }


@dataclass
class ScoreRouterConfig:
    balanced_accuracy_weight: float = 0.70
    macro_f1_weight: float = 0.30
    confidence_margin_scale: float = 0.02
    min_utility_gain: float = 0.0
    balanced_accuracy_lcb_offset: float = 0.0
    macro_f1_lcb_offset: float = 0.0
    decision_threshold: float = 0.0
    beats_default_probability_threshold: float = 0.0
    support_penalty_scale: float = 0.0


@dataclass
class ScoreExpandedRouter:
    """Predict candidate BA/F1 scores and select a runnable tabnetics profile."""

    config: ScoreRouterConfig = field(default_factory=ScoreRouterConfig)
    feature_names_: tuple[str, ...] = field(default_factory=tuple, init=False)
    action_feature_names_: tuple[str, ...] = field(default_factory=tuple, init=False)
    candidates_: list[dict[str, Any]] = field(default_factory=list, init=False)
    default_candidate_id_: str = field(default="", init=False)
    feature_median_: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float), init=False, repr=False)
    feature_low_: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float), init=False, repr=False)
    feature_high_: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float), init=False, repr=False)
    score_models_: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    training_metadata_: dict[str, Any] = field(default_factory=dict, init=False)
    fitted_: bool = field(default=False, init=False)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.feature_names_

    def _descriptor_vector(self, descriptor: Mapping[str, Any]) -> np.ndarray:
        payload = dict(descriptor.get("feature_vector", descriptor))
        raw = np.asarray([_safe_float(payload.get(name, 0.0), 0.0) for name in self.feature_names_], dtype=float)
        if self.feature_median_.size == 0:
            return raw.reshape(1, -1)
        arr = raw.reshape(1, -1)
        finite_mask = np.isfinite(arr)
        if not np.all(finite_mask):
            arr = np.where(finite_mask, arr, self.feature_median_[None, :])
        return np.clip(arr, self.feature_low_[None, :], self.feature_high_[None, :])

    def _candidate_action_vector(self, candidate: Mapping[str, Any]) -> np.ndarray:
        values = dict(candidate.get("action_features", {}) or {})
        return np.asarray(
            [_safe_float(values.get(name, 0.0), 0.0) for name in self.action_feature_names_],
            dtype=float,
        ).reshape(1, -1)

    def _candidate_matrix(self, descriptor: Mapping[str, Any]) -> np.ndarray:
        x_desc = self._descriptor_vector(descriptor)
        rows = [
            np.concatenate([x_desc, self._candidate_action_vector(candidate)], axis=1)[0]
            for candidate in self.candidates_
        ]
        if not rows:
            raise RuntimeError("ScoreExpandedRouter has no candidates.")
        return np.asarray(rows, dtype=float)

    def _beats_default_probabilities(self, X: np.ndarray) -> np.ndarray:
        model = self.score_models_.get("beats_default")
        if model is None:
            return np.ones(X.shape[0], dtype=float)
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(X), dtype=float)
            classes = [int(label) for label in getattr(model, "classes_", [0, 1])]
            return probs[:, classes.index(1)] if 1 in classes else np.zeros(X.shape[0], dtype=float)
        raw = np.asarray(model.predict(X), dtype=float)
        return np.clip(np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    def _support_penalties(self) -> np.ndarray:
        values = []
        for candidate in self.candidates_:
            support = max(float(candidate.get("support_dataset_seed_groups", 0) or 0), 1.0)
            values.append(float(self.config.support_penalty_scale) / float(np.sqrt(support)))
        return np.asarray(values, dtype=float)

    def predict(self, descriptor: Mapping[str, Any]) -> AutoRouterOutput:
        if not self.fitted_:
            raise RuntimeError("ScoreExpandedRouter is not fitted.")
        X = self._candidate_matrix(descriptor)
        ba = np.asarray(self.score_models_["balanced_accuracy"].predict(X), dtype=float)
        f1 = np.asarray(self.score_models_["macro_f1"].predict(X), dtype=float)
        ba = np.clip(np.nan_to_num(ba, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        f1 = np.clip(np.nan_to_num(f1, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        beats_default_probability = self._beats_default_probabilities(X)
        ba_lcb = np.clip(ba - float(self.config.balanced_accuracy_lcb_offset), 0.0, 1.0)
        f1_lcb = np.clip(f1 - float(self.config.macro_f1_lcb_offset), 0.0, 1.0)
        raw_utility = float(self.config.balanced_accuracy_weight) * ba + float(self.config.macro_f1_weight) * f1
        utility = (
            float(self.config.balanced_accuracy_weight) * ba_lcb
            + float(self.config.macro_f1_weight) * f1_lcb
            - self._support_penalties()
        )
        order = np.argsort(utility)[::-1]
        top_idx = int(order[0])
        second_utility = float(utility[int(order[1])]) if len(order) > 1 else float(utility[top_idx])
        margin = float(utility[top_idx] - second_utility)
        default_idx = next(
            (idx for idx, candidate in enumerate(self.candidates_) if str(candidate.get("id")) == self.default_candidate_id_),
            top_idx,
        )
        default_margin = float(utility[top_idx] - utility[int(default_idx)])
        policy_defaulted = bool(
            top_idx != int(default_idx)
            and (
                default_margin < max(float(self.config.min_utility_gain), float(self.config.decision_threshold))
                or float(beats_default_probability[top_idx]) < float(self.config.beats_default_probability_threshold)
            )
        )
        if policy_defaulted or default_margin < float(self.config.min_utility_gain):
            top_idx = int(default_idx)
            margin = max(0.0, float(utility[top_idx] - second_utility))
        confidence = 1.0 if policy_defaulted else float(
            np.clip(margin / max(float(self.config.confidence_margin_scale), 1e-9), 0.0, 1.0)
        )
        candidate = dict(self.candidates_[top_idx])
        topk = []
        for idx in order[: min(5, len(order))]:
            item = dict(self.candidates_[int(idx)])
            topk.append(
                {
                    "candidate_id": str(item.get("id", "")),
                    "predicted_balanced_accuracy": float(ba[int(idx)]),
                    "predicted_macro_f1": float(f1[int(idx)]),
                    "predicted_utility": float(raw_utility[int(idx)]),
                    "calibrated_utility": float(utility[int(idx)]),
                    "beats_default_probability": float(beats_default_probability[int(idx)]),
                    "method_count": int(item.get("method_count", 0) or 0),
                    "df_stage_position": item.get("df_stage_position"),
                }
            )
        return AutoRouterOutput(
            enabled_methods=list(candidate.get("enabled_methods", []) or []),
            config_overrides=dict(candidate.get("config_overrides", {}) or {}),
            classification_overrides=dict(candidate.get("classification_overrides", {}) or {}),
            confidence=confidence,
            head_confidences={
                "score_margin": confidence,
                "balanced_accuracy": float(ba[top_idx]),
                "macro_f1": float(f1[top_idx]),
            },
            metadata={
                "router_type": SCORE_ROUTER_ARTIFACT_TYPE,
                "selected_candidate_id": str(candidate.get("id", "")),
                "default_candidate_id": str(self.default_candidate_id_),
                "predicted_balanced_accuracy": float(ba[top_idx]),
                "predicted_macro_f1": float(f1[top_idx]),
                "predicted_balanced_accuracy_lcb": float(ba_lcb[top_idx]),
                "predicted_macro_f1_lcb": float(f1_lcb[top_idx]),
                "predicted_utility": float(raw_utility[top_idx]),
                "calibrated_utility": float(utility[top_idx]),
                "utility_margin": float(margin),
                "default_utility_margin": float(default_margin),
                "beats_default_probability": float(beats_default_probability[top_idx]),
                "policy_defaulted": bool(policy_defaulted),
                "decision_threshold": float(self.config.decision_threshold),
                "ranked_candidates": topk,
            },
        )

    @classmethod
    def from_components(
        cls,
        *,
        feature_names: Sequence[str],
        action_feature_names: Sequence[str],
        candidates: Sequence[Mapping[str, Any]],
        default_candidate_id: str,
        feature_median: Sequence[float],
        feature_low: Sequence[float],
        feature_high: Sequence[float],
        score_models: Mapping[str, Any],
        config: Optional[ScoreRouterConfig] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ScoreExpandedRouter":
        router = cls(config=config or ScoreRouterConfig())
        router.feature_names_ = tuple(str(name) for name in feature_names)
        router.action_feature_names_ = tuple(str(name) for name in action_feature_names)
        router.candidates_ = [dict(candidate) for candidate in candidates]
        router.default_candidate_id_ = str(default_candidate_id)
        router.feature_median_ = np.asarray(feature_median, dtype=float)
        router.feature_low_ = np.asarray(feature_low, dtype=float)
        router.feature_high_ = np.asarray(feature_high, dtype=float)
        router.score_models_ = dict(score_models)
        router.training_metadata_ = dict(metadata or {})
        router.fitted_ = True
        return router

    @classmethod
    def load(cls, model_dir: Path | str) -> "ScoreExpandedRouter":
        path = Path(model_dir)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if str(manifest.get("artifact_type", "")) != SCORE_ROUTER_ARTIFACT_TYPE:
            raise ValueError(f"Not a {SCORE_ROUTER_ARTIFACT_TYPE} artifact: {path}")
        return cls.from_components(
            feature_names=manifest.get("feature_names", []) or [],
            action_feature_names=manifest.get("action_feature_names", []) or [],
            candidates=manifest.get("candidates", []) or [],
            default_candidate_id=str(manifest.get("default_candidate_id", "") or ""),
            feature_median=manifest.get("feature_median", []) or [],
            feature_low=manifest.get("feature_low", []) or [],
            feature_high=manifest.get("feature_high", []) or [],
            score_models=joblib.load(path / "score_models.joblib"),
            config=ScoreRouterConfig(**dict(manifest.get("config", {}) or {})),
            metadata=dict(manifest.get("training_metadata", {}) or {}),
        )


def default_artifact_path() -> Path:
    """Return the bundled V25 router artifact directory."""

    return Path(
        resources.files("tabnetics.auto_router").joinpath(
            "artifacts", AUTO_ROUTER_ARTIFACT_VERSION
        )
    )


@lru_cache(maxsize=4)
def _load_router_cached(model_dir: str) -> ScoreExpandedRouter:
    return ScoreExpandedRouter.load(model_dir)


def load_default_auto_router(model_dir: Optional[Path | str] = None) -> ScoreExpandedRouter:
    """Load the default packaged auto-router, cached by artifact path."""

    path = Path(model_dir) if model_dir is not None else default_artifact_path()
    return _load_router_cached(str(path))


def predict_auto_router(
    X: Any,
    y: Any,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    model_dir: Optional[Path | str] = None,
) -> AutoRouterOutput:
    """Compute a dataset descriptor and emit the V25 auto-router decision."""

    descriptor = compute_dataset_descriptor(X, y, metadata=metadata)
    return load_default_auto_router(model_dir).predict(descriptor)


def _set_nested_distribution_attr(config: Any, key: str, value: Any) -> bool:
    dist_cfg = getattr(config, "dist_config", None)
    if dist_cfg is None:
        return False
    if str(key) == "df_family_set":
        setattr(dist_cfg, "family_set", str(value))
        return True
    return False


def _set_classification_attr(config: Any, key: str, value: Any) -> None:
    cls_cfg = getattr(config, "classification", None)
    if cls_cfg is not None:
        try:
            setattr(cls_cfg, key, value)
        except Exception:
            pass
    flat_key = _CLASSIFICATION_ALIAS_MAP.get(str(key))
    setattr(config, flat_key or str(key), value)


def _set_config_attr(config: Any, key: str, value: Any) -> None:
    key_s = str(key)
    if key_s == "max_train_samples" and (value is None or int(value) <= 0):
        setattr(config, key_s, None)
        return
    inverse_toggles = {
        "disable_cdf_transform": "apply_cdf_transform",
        "disable_cdf_reliability_gate": "cdf_reliability_gate",
        "disable_rank_prefilter": "use_rank_prefilter",
    }
    positive_aliases = {
        "enable_cdf_block_gating_cv": "cdf_block_gating_cv",
        "enable_df_fastpath": "df_fastpath_enabled",
        "enable_fs_adaptive_portfolio_sizing": "fs_adaptive_portfolio_sizing_enabled",
        "enable_fs_ipss_eats_threshold": "fs_ipss_use_eats_threshold",
    }
    if _set_nested_distribution_attr(config, key_s, value):
        setattr(config, key_s, value)
        return
    if key_s in inverse_toggles:
        setattr(config, inverse_toggles[key_s], not bool(value))
        setattr(config, key_s, bool(value))
        return
    if key_s in positive_aliases:
        setattr(config, positive_aliases[key_s], bool(value))
        setattr(config, key_s, bool(value))
        return
    if isinstance(value, list):
        value = tuple(value)
    setattr(config, key_s, value)


def apply_router_output(config: Any, output: AutoRouterOutput) -> Any:
    """Apply an auto-router decision onto a DFFSConfig-like object."""

    enabled_methods = list(output.enabled_methods or [])
    if enabled_methods:
        setattr(config, "enabled_methods", tuple(str(m) for m in enabled_methods if str(m).strip()))
    for key, raw_value in dict(output.config_overrides or {}).items():
        _set_config_attr(config, str(key), _clean_scalar(raw_value))
    for key, raw_value in dict(output.classification_overrides or {}).items():
        _set_classification_attr(config, str(key), _clean_scalar(raw_value))
    setattr(config, "auto_router_enabled", False)
    setattr(config, "auto_router_last_decision", output.to_snapshot())
    return config

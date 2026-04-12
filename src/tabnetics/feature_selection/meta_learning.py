"""Meta-learning selector for routing among supported Val-16 runtime profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from tabnetics.datasets.tier_classifier import EXPANDED_META_FEATURE_KEYS

try:
    from tabnetics.core.compat import make_logistic_regression
except Exception:
    from tabnetics.core.compat import make_logistic_regression  # type: ignore


SUPPORTED_RUNTIME_PROFILES: Tuple[str, ...] = (
    "a_control",
    "d_default",
    "v16_ref",
    "v16_multiomics",
)

LEGACY_MODEL_CANDIDATES: Tuple[str, ...] = (
    "lr",
    "svm_rbf",
    "svm_linear",
    "dlda",
    "knn",
    "rf",
    "nb",
    "elastic_net_lr",
)

PROFILE_METHOD_SETS: Dict[str, Tuple[str, ...]] = {
    "a_control": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
    ),
    "d_default": (
        "stability_lasso",
        "stability_subsample",
        "tigress_stability",
        "subspace_stability",
        "decorrelated_stability",
        "ipss",
        "cluster_stability",
        "rfecv",
        "boruta",
        "gradient_boosting",
        "linear_svm",
        "treeshap",
        "oaenet",
        "mutual_information",
        "anova_f",
        "chi_square",
        "relieff",
        "fcbf",
        "cmim",
        "wmw_auc",
        "joint_auc_l1",
        "ova_ensemble",
        "ecoc_class_aware",
        "joint_multiclass_support",
        "dove_class_specific",
        "sparse_multinomial",
        "nearest_shrunken_centroid",
        "class_pareto_front",
        "hsic_lasso",
        "slce_centroid_encoder",
        "mrmr_jmi",
        "iterative_redundancy_pruning",
        "iterative_redundancy_pruning_bounded",
        "ktsp",
        "copula_knockoff",
    ),
    "v16_ref": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "class_pareto_front",
        "boruta",
        "copula_knockoff",
        "decorrelated_stability",
        "relieff",
        "stability_lasso",
        "rfecv",
        "hsic_lasso",
        "joint_multiclass_support",
        "ipss",
    ),
    "v16_multiomics": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "class_pareto_front",
        "boruta",
        "copula_knockoff",
        "decorrelated_stability",
        "relieff",
        "stability_lasso",
        "rfecv",
        "hsic_lasso",
        "joint_multiclass_support",
        "ipss",
    ),
}

META_FEATURE_KEYS: Tuple[str, ...] = tuple(EXPANDED_META_FEATURE_KEYS)

DEFAULT_RECORDS_PATH = Path(__file__).with_name("meta_learning_records.json")
DEFAULT_FALLBACK_PROFILE = "v16_ref"


def _ordered_unique_profile_labels(records: Sequence[Dict[str, Any]]) -> Tuple[str, ...]:
    labels: List[str] = []
    seen = set()
    for row in records:
        label = str(row.get("best_profile", "") or "").strip()
        if not label or label in seen:
            continue
        labels.append(label)
        seen.add(label)
    return tuple(labels)


def load_training_payload(path: Optional[Path] = None) -> Dict[str, Any]:
    records_path = DEFAULT_RECORDS_PATH if path is None else Path(path)
    with records_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        out = dict(payload)
        out["records"] = [dict(row) for row in out.get("records", [])]
        return out
    return {
        "schema_version": 0,
        "records": [dict(row) for row in payload],
    }


def _resolve_profile_labels(
    payload: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
) -> Tuple[str, ...]:
    configured = [
        str(label).strip()
        for label in list(payload.get("profile_labels") or [])
        if str(label).strip()
    ]
    if configured:
        return tuple(configured)
    inferred = _ordered_unique_profile_labels(records)
    return inferred or SUPPORTED_RUNTIME_PROFILES


def _resolve_fallback_profile(
    payload: Dict[str, Any],
    profile_labels: Sequence[str],
) -> str:
    fallback = str(payload.get("fallback_profile", DEFAULT_FALLBACK_PROFILE) or DEFAULT_FALLBACK_PROFILE).strip()
    if fallback in profile_labels:
        return fallback
    if DEFAULT_FALLBACK_PROFILE in profile_labels:
        return DEFAULT_FALLBACK_PROFILE
    return str(profile_labels[0]) if profile_labels else DEFAULT_FALLBACK_PROFILE


def _resolve_runtime_profile_overlays(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = dict(payload.get("runtime_profile_overlays", {}) or {})
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        profile_id = str(key).strip()
        if not profile_id or not isinstance(value, dict):
            continue
        out[profile_id] = dict(value)
    return out


def normalize_historical_profile(profile_id: str) -> Optional[str]:
    key = str(profile_id or "").strip().lower()
    mapping = {
        "a_control": "a_control",
        "a01_simple_anchor_after_fs": "a_control",
        "d_default": "d_default",
        "a02_default_anchor_after_fs": "d_default",
        "v20_b01_default_anchor": "d_default",
        "v14_ref": "v16_ref",
        "v14_ipss": "v16_ref",
        "v15_ref_ipss": "v16_ref",
        "a03_ref_anchor_after_fs": "v16_ref",
        "v14_multiomics_adapter": "v16_multiomics",
        "v15_multiomics_adapter": "v16_multiomics",
        "v16_ref": "v16_ref",
        "v16_multiomics": "v16_multiomics",
        "v20_b02_ref_anchor": "v16_ref",
        "v20_b03_mnpo_ref_anchor": "v16_ref",
        "v20_b04_val19_new_mnpo": "v16_ref",
        "v20_c01_candidate_a_full64": "V20_C01_candidate_a_full64",
        "v20_c02_candidate_b_full64": "V20_C02_candidate_b_full64",
        "v20_c03_candidate_c_full64": "V20_C03_candidate_c_full64",
        "v20_c04_current_default_full64": "V20_C04_current_default_full64",
    }
    return mapping.get(key)


def compute_meta_learning_features(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    try:
        from tabnetics.datasets.meta_features import extract_meta_features
    except Exception:
        from tabnetics.datasets.meta_features import extract_meta_features  # type: ignore

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()
    base = {
        str(k): float(v)
        for k, v in extract_meta_features(X_arr, y_arr, expanded=True).items()
    }
    return {key: float(base.get(key, 0.0)) for key in META_FEATURE_KEYS}


def load_training_records(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    payload = load_training_payload(path)
    return [dict(row) for row in payload.get("records", [])]


def _set_classification_attr(config: Any, key: str, value: Any) -> None:
    cls_cfg = getattr(config, "classification", None)
    if cls_cfg is not None:
        setattr(cls_cfg, key, value)
    legacy_aliases = {
        "selection_mode": "classification_selection_mode",
        "backend": "classification_backend",
        "model_candidates": "model_candidates",
        "runtime_max_candidates": "model_cv_runtime_max_candidates",
        "runtime_containment_enabled": "model_cv_runtime_containment_enabled",
        "stage2_tree_complexity_penalty_enabled": "stage2_tree_complexity_penalty_enabled",
        "stage2_tree_complexity_penalty_strength": "stage2_tree_complexity_penalty_strength",
        "stage2_ratio_augmentation_enabled": "stage2_ratio_augmentation_enabled",
        "stage2_ratio_max_features": "stage2_ratio_max_features",
        "stage2_ratio_selection_method": "stage2_ratio_selection_method",
        "conformal_enabled": "classifier_conformal_enabled",
        "conformal_alpha": "classifier_conformal_alpha",
        "conformal_calibration_fraction": "classifier_conformal_calibration_fraction",
        "conformal_min_calibration": "classifier_conformal_min_calibration",
        "conformal_method": "classifier_conformal_method",
    }
    flat_key = legacy_aliases.get(key)
    if flat_key is not None:
        setattr(config, flat_key, value)


def _apply_legacy_model_candidates(config: Any) -> None:
    candidates = tuple(LEGACY_MODEL_CANDIDATES)
    _set_classification_attr(config, "model_candidates", candidates)
    include_map = {
        "include_elastic_net_model": "elastic_net_lr",
        "include_rf_model": "rf",
        "include_knn_model": "knn",
        "include_svm_linear_model": "svm_linear",
        "include_dlda_model": "dlda",
        "include_nb_model": "nb",
        "include_nsc_model": "nsc",
        "include_pls_da_model": "pls_da_classifier",
        "include_gpc_model": "gpc",
        "include_vote_ensemble_model": "vote_ensemble",
        "include_xgb_model": "xgb",
        "include_lgbm_model": "lgbm",
        "include_extra_tree_model": "extra_tree",
        "include_catboost_model": "catboost",
        "include_tabpfn_model": "tabpfn",
    }
    selected = set(candidates)
    for attr, token in include_map.items():
        value = bool(token in selected)
        setattr(config, attr, value)
        cls_cfg = getattr(config, "classification", None)
        if cls_cfg is not None and hasattr(cls_cfg, attr):
            setattr(cls_cfg, attr, value)
    _set_classification_attr(config, "runtime_max_candidates", 8)


_COMMON_RUNTIME_PROFILE_OVERRIDES: Dict[str, Any] = {
    "dist_criterion": "simple",
    "folding_method": "pls_da",
    "prefilter_rnaseq_nb_lrt_enabled": True,
    "prefilter_rnaseq_nb_lrt_alpha": 0.10,
    "multiomics_adapter": "none",
    "multiomics_integrator": "mb_plsda",
    "multiomics_n_components": 2,
    "prefilter_bh_ttest_enabled": True,
    "prefilter_variance_floor_enabled": True,
    "fs_mrmr_mi_redundancy_enabled": True,
    "fs_fold_preference_mode": "vote",
    "fs_use_conformal_efficiency": False,
    "fs_conformal_efficiency_method": "split",
    "fs_oracle_weight_js_shrinkage": False,
    "fs_payoff_shrinkage_kappa": 0.0,
}

_COMMON_RUNTIME_CLASSIFICATION_OVERRIDES: Dict[str, Any] = {
    "selection_mode": "legacy",
    "backend": "sklearn",
    "runtime_containment_enabled": True,
    "stage2_tree_complexity_penalty_enabled": True,
    "stage2_tree_complexity_penalty_strength": 0.1,
    "stage2_ratio_augmentation_enabled": True,
    "stage2_ratio_max_features": 16,
    "stage2_ratio_selection_method": "correlation",
    "conformal_enabled": True,
    "conformal_alpha": 0.10,
    "conformal_calibration_fraction": 0.25,
    "conformal_min_calibration": 20,
    "conformal_method": "split",
}

_PROFILE_RUNTIME_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "a_control": {
        "mnpo_performance_oracle_mode": "single",
        "use_diversity_oracle": False,
        "fs_oracle_weighting_mode": "uniform",
        "fs_adaptive_portfolio_sizing_enabled": False,
        "prefilter_union_enabled": False,
        "prefilter_wsnr_enabled": False,
        "prefilter_strategies": ("mi_ftest_blend",),
        "screening_enabled": False,
        "eval_models_enabled": False,
        "regime_gating_enabled": False,
        "fs_copula_derandomize_runs": 3,
    },
    "d_default": {
        "mnpo_performance_oracle_mode": "multi_model_oracles",
        "use_diversity_oracle": True,
        "fs_oracle_weighting_mode": "banzhaf",
        "fs_adaptive_portfolio_sizing_enabled": True,
        "fs_adaptive_size_min": 4,
        "fs_adaptive_size_max": 8,
        "fs_portfolio_size_guard": "warn",
        "prefilter_union_enabled": True,
        "prefilter_wsnr_enabled": True,
        "prefilter_strategies": ("mi_ftest_blend", "rf_importance", "wsnr", "bh_fdr"),
        "screening_enabled": True,
        "screening_method": "evalue",
        "eval_models_enabled": True,
        "eval_models": ("lr_l2", "linear_svc", "rf_small"),
        "eval_aggregate": "mean",
        "fs_diversity_oracle_mode": "legacy_jaccard",
        "regime_gating_enabled": False,
        "fs_copula_derandomize_runs": 3,
    },
    "v16_ref": {
        "mnpo_performance_oracle_mode": "multi_model_oracles",
        "use_diversity_oracle": True,
        "fs_oracle_weighting_mode": "banzhaf",
        "fs_adaptive_portfolio_sizing_enabled": True,
        "fs_adaptive_size_min": 4,
        "fs_adaptive_size_max": 8,
        "fs_portfolio_size_guard": "warn",
        "prefilter_union_enabled": True,
        "prefilter_wsnr_enabled": True,
        "prefilter_strategies": ("mi_ftest_blend", "rf_importance", "wsnr", "bh_fdr"),
        "screening_enabled": True,
        "screening_method": "evalue",
        "eval_models_enabled": True,
        "eval_models": ("lr_l2", "linear_svc", "rf_small"),
        "eval_aggregate": "mean",
        "fs_diversity_oracle_mode": "legacy_jaccard",
        "regime_gating_enabled": True,
        "regime_gating_difficulty_source": "historical",
        "regime_gating_target_tier": "very_hard",
        "regime_gating_min_samples_per_class": 7.0,
        "regime_gating_very_hard_min_classes": 5,
        "regime_gating_low_p_over_n_threshold": 0.0,
        "regime_gating_simple_methods": tuple(PROFILE_METHOD_SETS["a_control"]),
        "regime_gating_very_hard_portfolio_max_methods": 4,
        "regime_gating_very_hard_copula_derandomize_runs": 5,
        "regime_gating_extreme_multiclass_enabled": True,
        "regime_gating_extreme_multiclass_threshold": 8,
        "regime_gating_extreme_multiclass_min_samples_per_class": 11.0,
        "fs_copula_derandomize_runs": 5,
        "fs_max_selected_features_ratio": 0.5,
        "fs_max_selected_features_cap": 500,
        "fs_stability_threshold_method": "fixed",
    },
    "v16_multiomics": {
        "mnpo_performance_oracle_mode": "multi_model_oracles",
        "use_diversity_oracle": True,
        "fs_oracle_weighting_mode": "banzhaf",
        "fs_adaptive_portfolio_sizing_enabled": True,
        "fs_adaptive_size_min": 4,
        "fs_adaptive_size_max": 8,
        "fs_portfolio_size_guard": "warn",
        "prefilter_union_enabled": True,
        "prefilter_wsnr_enabled": True,
        "prefilter_strategies": ("mi_ftest_blend", "rf_importance", "wsnr", "bh_fdr"),
        "screening_enabled": True,
        "screening_method": "evalue",
        "eval_models_enabled": True,
        "eval_models": ("lr_l2", "linear_svc", "rf_small"),
        "eval_aggregate": "mean",
        "fs_diversity_oracle_mode": "legacy_jaccard",
        "regime_gating_enabled": True,
        "regime_gating_difficulty_source": "historical",
        "regime_gating_target_tier": "very_hard",
        "regime_gating_min_samples_per_class": 7.0,
        "regime_gating_very_hard_min_classes": 5,
        "regime_gating_low_p_over_n_threshold": 0.0,
        "regime_gating_simple_methods": tuple(PROFILE_METHOD_SETS["a_control"]),
        "regime_gating_very_hard_portfolio_max_methods": 4,
        "regime_gating_very_hard_copula_derandomize_runs": 5,
        "regime_gating_extreme_multiclass_enabled": True,
        "regime_gating_extreme_multiclass_threshold": 8,
        "regime_gating_extreme_multiclass_min_samples_per_class": 11.0,
        "fs_copula_derandomize_runs": 5,
        "fs_max_selected_features_ratio": 0.5,
        "fs_max_selected_features_cap": 500,
        "fs_stability_threshold_method": "fixed",
        "multiomics_adapter": "split_halves",
        "multiomics_integrator": "mb_plsda",
        "multiomics_n_components": 2,
    },
}


def _apply_runtime_profile_overlay_dict(
    config: Any,
    profile_id: str,
    overlay: Dict[str, Any],
) -> Any:
    enabled_methods = tuple(
        str(method).strip()
        for method in list(overlay.get("enabled_methods") or [])
        if str(method).strip()
    )
    if enabled_methods:
        setattr(config, "enabled_methods", enabled_methods)
    config_overrides = dict(
        overlay.get("config_overrides", overlay.get("overrides", {})) or {}
    )
    for key, value in config_overrides.items():
        setattr(config, str(key), value)
    classification_overrides = dict(overlay.get("classification_overrides", {}) or {})
    for key, value in classification_overrides.items():
        _set_classification_attr(config, str(key), value)
    if not enabled_methods and not config_overrides and not classification_overrides:
        raise ValueError(f"Runtime profile overlay {profile_id!r} is empty.")
    return config


def apply_runtime_profile_overlay(
    config: Any,
    profile_id: str,
    *,
    runtime_profile_overlays: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Any:
    profile_raw = str(profile_id or "").strip()
    custom_overlays = {
        str(key): dict(value)
        for key, value in dict(runtime_profile_overlays or {}).items()
        if str(key).strip()
    }
    if profile_raw in custom_overlays:
        return _apply_runtime_profile_overlay_dict(
            config,
            profile_raw,
            custom_overlays[profile_raw],
        )
    lowered_custom = {str(key).strip().lower(): dict(value) for key, value in custom_overlays.items()}
    profile = str(profile_raw).lower()
    if profile in lowered_custom:
        return _apply_runtime_profile_overlay_dict(
            config,
            profile_raw,
            lowered_custom[profile],
        )
    if profile not in PROFILE_METHOD_SETS:
        raise ValueError(f"Unsupported runtime profile: {profile_id!r}")

    setattr(config, "enabled_methods", tuple(PROFILE_METHOD_SETS[profile]))
    for key, value in _COMMON_RUNTIME_PROFILE_OVERRIDES.items():
        setattr(config, key, value)
    for key, value in _COMMON_RUNTIME_CLASSIFICATION_OVERRIDES.items():
        _set_classification_attr(config, key, value)
    _apply_legacy_model_candidates(config)
    for key, value in _PROFILE_RUNTIME_OVERRIDES[profile].items():
        setattr(config, key, value)
    return config


@dataclass
class MetaLearningSelector:
    mode: str = "decision_tree"
    confidence_threshold: float = 0.55
    records_path: Optional[Path] = field(default_factory=lambda: DEFAULT_RECORDS_PATH)
    random_state: int = 42
    model_: Any = field(default=None, init=False, repr=False)
    classes_: Tuple[str, ...] = field(default_factory=tuple, init=False)
    feature_names_: Tuple[str, ...] = field(default_factory=lambda: META_FEATURE_KEYS, init=False)
    profile_labels_: Tuple[str, ...] = field(default_factory=tuple, init=False)
    fallback_profile_: str = field(default=DEFAULT_FALLBACK_PROFILE, init=False)
    runtime_profile_overlays_: Dict[str, Dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    fitted_: bool = field(default=False, init=False)

    def _make_model(self):
        key = str(self.mode or "decision_tree").strip().lower()
        if key == "logistic":
            return make_logistic_regression(
                solver="lbfgs",
                penalty="l2",
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=int(self.random_state),
            )
        return DecisionTreeClassifier(max_depth=4, random_state=int(self.random_state))

    def _records_or_load(self, records: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if records is None:
            payload = load_training_payload(self.records_path)
            rows = [dict(row) for row in payload.get("records", [])]
            self.profile_labels_ = _resolve_profile_labels(payload, rows)
            self.fallback_profile_ = _resolve_fallback_profile(payload, self.profile_labels_)
            self.runtime_profile_overlays_ = _resolve_runtime_profile_overlays(payload)
            return rows
        rows = [dict(row) for row in records]
        self.profile_labels_ = _ordered_unique_profile_labels(rows) or SUPPORTED_RUNTIME_PROFILES
        self.fallback_profile_ = (
            DEFAULT_FALLBACK_PROFILE
            if DEFAULT_FALLBACK_PROFILE in self.profile_labels_
            else str(self.profile_labels_[0])
        )
        self.runtime_profile_overlays_ = {}
        return rows

    def _matrix_from_records(self, records: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        X_rows: List[List[float]] = []
        y_rows: List[str] = []
        valid_profiles = set(self.profile_labels_ or SUPPORTED_RUNTIME_PROFILES)
        for row in records:
            feats = dict(row.get("meta_features", {}) or {})
            label = str(row.get("best_profile", "") or "")
            if label not in valid_profiles:
                continue
            X_rows.append([float(feats.get(name, 0.0) or 0.0) for name in self.feature_names_])
            y_rows.append(label)
        return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=object)

    def fit(self, records: Optional[Sequence[Dict[str, Any]]] = None) -> "MetaLearningSelector":
        rows = self._records_or_load(records)
        X, y = self._matrix_from_records(rows)
        if X.shape[0] < 2 or np.unique(y).size < 2:
            raise ValueError("Insufficient meta-learning records to fit selector.")
        self.model_ = self._make_model()
        self.model_.fit(X, y)
        self.classes_ = tuple(str(v) for v in getattr(self.model_, "classes_", np.unique(y)).tolist())
        if not self.profile_labels_:
            self.profile_labels_ = tuple(self.classes_)
        if self.fallback_profile_ not in self.profile_labels_:
            self.fallback_profile_ = (
                DEFAULT_FALLBACK_PROFILE
                if DEFAULT_FALLBACK_PROFILE in self.profile_labels_
                else str(self.profile_labels_[0])
            )
        self.fitted_ = True
        return self

    def _predict_with_model(self, feature_map: Dict[str, float]) -> Dict[str, Any]:
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("MetaLearningSelector is not fitted.")
        x_row = np.asarray(
            [[float(feature_map.get(name, 0.0) or 0.0) for name in self.feature_names_]],
            dtype=float,
        )
        if hasattr(self.model_, "predict_proba"):
            probs = np.asarray(self.model_.predict_proba(x_row), dtype=float).ravel()
        else:
            pred = str(self.model_.predict(x_row)[0])
            probs = np.zeros(len(self.classes_), dtype=float)
            if pred in self.classes_:
                probs[self.classes_.index(pred)] = 1.0
        if probs.size != len(self.classes_):
            aligned = np.zeros(len(self.classes_), dtype=float)
            pred = str(self.model_.predict(x_row)[0])
            if pred in self.classes_:
                aligned[self.classes_.index(pred)] = 1.0
            probs = aligned
        top_idx = int(np.argmax(probs)) if probs.size else -1
        confidence = float(probs[top_idx]) if top_idx >= 0 else 0.0
        predicted = str(self.classes_[top_idx]) if top_idx >= 0 else str(self.fallback_profile_)
        fallback = bool(confidence < float(self.confidence_threshold))
        selected = str(self.fallback_profile_) if fallback else predicted
        return {
            "meta_learning_profile_selected": str(selected),
            "meta_learning_profile_raw": str(predicted),
            "meta_learning_confidence": float(confidence),
            "meta_learning_fallback_applied": bool(fallback),
            "meta_learning_candidate_profiles": list(
                self.classes_ or self.profile_labels_ or SUPPORTED_RUNTIME_PROFILES
            ),
            "meta_learning_fallback_profile": str(self.fallback_profile_),
            "meta_learning_feature_vector": {
                str(name): float(feature_map.get(name, 0.0) or 0.0) for name in self.feature_names_
            },
        }

    def predict(self, feature_map: Dict[str, float]) -> Dict[str, Any]:
        return self._predict_with_model(feature_map)

    def predict_from_arrays(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        return self.predict(compute_meta_learning_features(X, y))

    def evaluate(self, records: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
        rows = self._records_or_load(records)
        valid_profiles = set(self.profile_labels_ or SUPPORTED_RUNTIME_PROFILES)
        fallback_profile = str(self.fallback_profile_ or DEFAULT_FALLBACK_PROFILE)
        usable = [
            row
            for row in rows
            if str(row.get("best_profile", "") or "") in valid_profiles
        ]
        if len(usable) < 2:
            return {
                "n_records": int(len(usable)),
                "routed_mean_ba": float("nan"),
                "static_fallback_mean_ba": float("nan"),
                "routed_minus_fallback": float("nan"),
                "static_v16_ref_mean_ba": float("nan"),
                "routed_minus_v16_ref": float("nan"),
                "label_accuracy": float("nan"),
                "fallback_profile": fallback_profile,
                "per_dataset": [],
            }

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in usable:
            dataset_id = str(row.get("dataset_id", "") or "").strip()
            if not dataset_id:
                continue
            grouped.setdefault(dataset_id, []).append(row)
        if len(grouped) < 2:
            return {
                "n_records": int(len(usable)),
                "n_datasets": int(len(grouped)),
                "routed_mean_ba": float("nan"),
                "static_fallback_mean_ba": float("nan"),
                "routed_minus_fallback": float("nan"),
                "static_v16_ref_mean_ba": float("nan"),
                "routed_minus_v16_ref": float("nan"),
                "label_accuracy": float("nan"),
                "fallback_profile": fallback_profile,
                "per_dataset": [],
            }

        per_dataset: List[Dict[str, Any]] = []
        y_true: List[str] = []
        y_pred: List[str] = []
        routed_scores: List[float] = []
        ref_scores: List[float] = []
        for dataset_id in sorted(grouped.keys()):
            holdout_rows = list(grouped.get(dataset_id) or [])
            if not holdout_rows:
                continue
            holdout = dict(holdout_rows[0])
            train_rows = [
                row
                for other_dataset, rows_for_dataset in grouped.items()
                if other_dataset != dataset_id
                for row in rows_for_dataset
            ]
            try:
                selector = MetaLearningSelector(
                    mode=str(self.mode),
                    confidence_threshold=float(self.confidence_threshold),
                    records_path=self.records_path,
                    random_state=int(self.random_state),
                ).fit(train_rows)
                pred = selector.predict(dict(holdout.get("meta_features", {}) or {}))
            except Exception:
                pred = {
                    "meta_learning_profile_selected": fallback_profile,
                    "meta_learning_confidence": 0.0,
                    "meta_learning_fallback_applied": True,
                    "meta_learning_candidate_profiles": list(self.profile_labels_ or SUPPORTED_RUNTIME_PROFILES),
                    "meta_learning_fallback_profile": fallback_profile,
                }
            routed_profile = str(pred.get("meta_learning_profile_selected", fallback_profile) or fallback_profile)
            profile_scores = dict(holdout.get("profile_scores", {}) or {})
            routed_scores.append(
                float(profile_scores.get(routed_profile, profile_scores.get(fallback_profile, 0.0)) or 0.0)
            )
            ref_scores.append(float(profile_scores.get(fallback_profile, 0.0) or 0.0))
            truth = str(holdout.get("best_profile", fallback_profile) or fallback_profile)
            y_true.append(truth)
            y_pred.append(routed_profile)
            per_dataset.append(
                {
                    "dataset_id": str(dataset_id),
                    "predicted_profile": routed_profile,
                    "best_profile": truth,
                    "confidence": float(pred.get("meta_learning_confidence", 0.0) or 0.0),
                    "fallback_applied": bool(pred.get("meta_learning_fallback_applied", False)),
                    "routed_balanced_accuracy": float(routed_scores[-1]),
                    "static_fallback_balanced_accuracy": float(ref_scores[-1]),
                    "static_v16_ref_balanced_accuracy": float(ref_scores[-1]),
                }
            )
        label_accuracy = float(np.mean(np.asarray(y_true, dtype=object) == np.asarray(y_pred, dtype=object)))
        routed_mean = float(np.mean(routed_scores))
        ref_mean = float(np.mean(ref_scores))
        return {
            "n_records": int(len(usable)),
            "n_datasets": int(len(grouped)),
            "routed_mean_ba": float(routed_mean),
            "static_fallback_mean_ba": float(ref_mean),
            "routed_minus_fallback": float(routed_mean - ref_mean),
            "static_v16_ref_mean_ba": float(ref_mean),
            "routed_minus_v16_ref": float(routed_mean - ref_mean),
            "label_accuracy": float(label_accuracy),
            "fallback_profile": fallback_profile,
            "per_dataset": per_dataset,
        }

"""Integrated distribution-fitting + feature-selection pipeline.

This module implements a reproducible pipeline aligned with:
- DistributionFittingPipeline.md (backward-compatible distribution fitter wrapper,
  data auditing, confidence-set diagnostics, abstain/rejection gate)
- FeatureExtractorImplementation.md (strict 80/20 holdout, train-only feature
  selection on train subsample, model fit on full train selected features)

Core protocol:
1. Reproducible train/test split.
2. Fit univariate distribution transforms on full train set.
3. Run feature selection on a train subset (fs_fraction of train).
4. Train classifier on full train using selected features.
5. Evaluate on untouched test set.
"""

from __future__ import annotations

import base64
import datetime as _dt
import warnings
from collections import Counter
import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from itertools import combinations
import logging  # Added for T-AUDIT-001-FIX-004
import math
import pickle
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import os as _os_setup
import zlib

from tabnetics.core.runtime import configure_runtime_environment, has_nvidia_gpu

configure_runtime_environment()
_HAS_NVIDIA_GPU = has_nvidia_gpu()
del _os_setup

import numpy as np
import scipy.stats as sps
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import PolynomialCountSketch, RBFSampler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss as sklearn_log_loss, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

try:
    from tabnetics.core.compat import make_logistic_regression
except Exception as exc:
    from tabnetics.core.compat import make_logistic_regression  # type: ignore

try:
    from tabnetics.distribution.selector import UnifiedDistributionSelectorV6
except Exception as exc:
    from tabnetics.distribution.selector import UnifiedDistributionSelectorV6

try:
    from tabnetics.feature_selection.config import FeatureSelectorConfig
except Exception as exc:
    try:
        from tabnetics.feature_selection.config import FeatureSelectorConfig  # type: ignore
    except Exception as exc:
        FeatureSelectorConfig = None  # type: ignore

try:
    from tabnetics.classification.backends import (
        ClassifierBackend,
        FLAMLBackend,
        MNPOClassifierBackend,
        OptunaBackend,
        SklearnBackend,
    )
except Exception as exc:
    from tabnetics.classification.backends import (  # type: ignore
        ClassifierBackend,
        FLAMLBackend,
        MNPOClassifierBackend,
        OptunaBackend,
        SklearnBackend,
    )

try:
    from tabnetics.feature_selection.conformal import compute_split_conformal_sets, compute_mapie_conformal_sets, mapie_available
except Exception as exc:
    from tabnetics.feature_selection.conformal import compute_split_conformal_sets, compute_mapie_conformal_sets, mapie_available  # type: ignore

try:
    from tabnetics.feature_selection.prefilter import (
        apply_batch_correction_model,
        fit_batch_correction_model,
    )
except Exception as exc:
    from tabnetics.feature_selection.prefilter import (  # type: ignore
        apply_batch_correction_model,
        fit_batch_correction_model,
    )

from tabnetics.domains.base import base_dataset_name, resolve_dataset_catalog_context
from tabnetics.domains.bio import apply_multiomics_adapter_train_test
from tabnetics.domains.face import apply_face_domain_projection

try:
    from xgboost import XGBClassifier as _XGBClassifier
except Exception as exc:
    _XGBClassifier = None

try:
    from tabpfn import TabPFNClassifier as _TabPFNClassifier
except Exception as exc:
    _TabPFNClassifier = None

# Defence-in-depth: if torch was loaded (e.g. via tabpfn), constrain its
# internal thread-pool and inter-op threads to 1 to avoid contention and
# potential cuBLAS mutex deadlocks in forked/spawned workers.
try:
    import torch as _torch_setup
    _torch_setup.set_num_threads(1)
    _torch_setup.set_num_interop_threads(1)
    del _torch_setup
except Exception:
    pass


def _load_feature_selector_cls():
    try:
        from tabnetics.feature_selection import FeatureSelector
    except Exception as exc:
        from tabnetics.feature_selection import FeatureSelector
    return FeatureSelector


# Set up logger for this module (T-AUDIT-001-FIX-004)
logger = logging.getLogger(__name__)

_SPARSE_SCREENING_MODE_ALIAS = {
    "strong": "prefilter_aggressive",
    "gap_safe": "prefilter_balanced",
    "slores": "prefilter_conservative",
}
_SPARSE_SCREENING_MODE_VALID = {
    "none",
    "prefilter_aggressive",
    "prefilter_balanced",
    "prefilter_conservative",
}
_DEPRECATED_TOGGLE_WARNED: Set[str] = set()

_POSITIVE_ONLY_FAMILIES: Set[str] = {
    "expon",
    "gamma",
    "lognorm",
    "weibull_min",
    "pareto",
    "invweibull",
    "invgauss",
    "geninvgauss",
    "invgamma",
    "fisk",
    "genpareto",
    "gengamma",
}
_UNIT_INTERVAL_ONLY_FAMILIES: Set[str] = {"beta", "powerlaw", "triang", "johnsonsb"}


def _warn_deprecated_toggle_once(key: str, message: str) -> None:
    if key in _DEPRECATED_TOGGLE_WARNED:
        return
    _DEPRECATED_TOGGLE_WARNED.add(str(key))
    warnings.warn(message, DeprecationWarning)


def _canonicalize_sparse_screening_mode(mode: Any, *, warn_deprecated: bool = False) -> str:
    raw = str(mode if mode is not None else "none").strip().lower()
    if raw in _SPARSE_SCREENING_MODE_ALIAS:
        canonical = _SPARSE_SCREENING_MODE_ALIAS[raw]
        if warn_deprecated:
            warnings.warn(
                f"fs_sparse_multinomial_screening_mode='{raw}' is deprecated; "
                f"use '{canonical}' instead.",
                DeprecationWarning,
            )
        raw = canonical
    if raw not in _SPARSE_SCREENING_MODE_VALID:
        raw = "none"
    return raw


def _warn_deprecated_df_fastpath_config(config: Any) -> None:
    """Emit one-time warnings for legacy no-op DF fast-path toggles."""
    if bool(getattr(config, "df_fastpath_enabled", False)):
        _warn_deprecated_toggle_once(
            "df_fastpath_enabled",
            "DFFSConfig.df_fastpath_enabled is deprecated and ignored; the DF fast-path was removed.",
        )
    if str(getattr(config, "df_fastpath_trigger", "small_n_or_low_unique") or "small_n_or_low_unique") != "small_n_or_low_unique":
        _warn_deprecated_toggle_once(
            "df_fastpath_trigger",
            "DFFSConfig.df_fastpath_trigger is deprecated and ignored; the DF fast-path was removed.",
        )
    if int(getattr(config, "df_fastpath_small_n_threshold", 250) or 250) != 250:
        _warn_deprecated_toggle_once(
            "df_fastpath_small_n_threshold",
            "DFFSConfig.df_fastpath_small_n_threshold is deprecated and ignored; the DF fast-path was removed.",
        )
    if float(getattr(config, "df_fastpath_unique_ratio_threshold", 0.05) or 0.05) != 0.05:
        _warn_deprecated_toggle_once(
            "df_fastpath_unique_ratio_threshold",
            "DFFSConfig.df_fastpath_unique_ratio_threshold is deprecated and ignored; the DF fast-path was removed.",
        )
    if int(getattr(config, "df_fastpath_n_unique_threshold", 12) or 12) != 12:
        _warn_deprecated_toggle_once(
            "df_fastpath_n_unique_threshold",
            "DFFSConfig.df_fastpath_n_unique_threshold is deprecated and ignored; the DF fast-path was removed.",
        )


class _IdentityFeatureSelector:
    """Minimal selector used by low p/n FS-bypass mode."""

    def __init__(self, n_features: int):
        n = int(max(0, n_features))
        self._selected = np.arange(n, dtype=int)
        self.mnpo_diagnostics_ = {}

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)

    def get_selected_features_indices(self) -> np.ndarray:
        return np.asarray(self._selected, dtype=int)


@dataclass
class SupportProfile:
    inferred_support: str
    frac_zero: float
    min_value: float
    max_value: float
    unique_ratio: float
    is_near_constant: bool


@dataclass
class DataAuditReport:
    n_raw: int
    n_clean: int
    n_missing: int
    n_unique: int
    support: SupportProfile
    has_heaping: bool
    heaping_delta: Optional[float]
    outlier_fraction: float
    # Optional diagnostics flags (opt-in via DistributionFitterConfig).
    frac_negative: float = 0.0
    is_integer_like: bool = False
    zero_inflated: bool = False
    dip_stat: Optional[float] = None
    mode_count: Optional[int] = None
    is_multimodal: Optional[bool] = None
    too_few_unique: bool = False


@dataclass
class DistributionFitSummary:
    feature_index: int
    family: Optional[str]
    params: Optional[Tuple[float, ...]]
    cvm_p: float
    ks_p: float
    simple_score: float
    confidence_set: Tuple[str, ...]
    rejected: bool
    audit: DataAuditReport
    # Optional extended diagnostics (opt-in via DistributionFitterConfig).
    ad_stat: Optional[float] = None
    ad_p: Optional[float] = None
    qq_r2: Optional[float] = None
    pp_r2: Optional[float] = None
    pp_mae: Optional[float] = None
    aic: Optional[float] = None
    aicc: Optional[float] = None
    bic: Optional[float] = None
    loglik: Optional[float] = None
    crps: Optional[float] = None
    # Opt-in CRPS uncertainty decomposition (Gaussian ensemble surrogate):
    # total = aleatoric + epistemic.
    crps_uq_total: Optional[float] = None
    crps_uq_aleatoric: Optional[float] = None
    crps_uq_epistemic: Optional[float] = None
    preq_loglik_mean: Optional[float] = None
    cv_loglik_mean: Optional[float] = None
    cv_loglik_std: Optional[float] = None
    cv_score: Optional[float] = None
    fit_method: Optional[str] = None
    mnpo_weight: Optional[float] = None
    rejection_reason: Optional[str] = None
    selected_family_support: Optional[str] = None
    candidates_pre_filter: Optional[int] = None
    candidates_post_filter: Optional[int] = None


@dataclass
class DistributionFitterConfig:
    robust_mode: bool = True
    use_adaptive_strategy: bool = True
    use_lrt: bool = True
    use_cv: bool = False
    compute_budget: str = "standard"  # fast | standard | thorough
    use_support_filtering: bool = True
    rejection_gate: bool = True
    rejection_p_threshold: float = 0.01
    confidence_margin: float = 0.05
    # Candidate-family library selection (opt-in to preserve baseline behavior).
    family_set: str = "v6"  # v6 | extended | flex
    # Diagnostics / additional GOF metrics.
    # Val-10 Promotion (Review 12): both profiles used these; promoted to default.
    compute_ad: bool = True
    ad_bootstrap_samples: int = 0
    compute_qq_pp: bool = True
    compute_dip: bool = True
    dip_hist_bins: int = 40
    # Advanced DF: interval likelihood for heaped/rounded features.
    # Val-10 Promotion (Review 12): both profiles used this; promoted to default.
    interval_likelihood: bool = True
    interval_delta_override: float = 0.0
    # Optional prescreen: limit fitted candidate count using L-moment ratios.
    # Val-10 Promotion (Review 12): both profiles used this with max_candidates=12; promoted to default.
    use_lmoment_prescreen: bool = True
    lmoment_prescreen_max_candidates: int = 12
    # Estimator (opt-in): default preserves MLE baseline.
    estimator: str = "mle"  # mle | mps
    mps_maxiter: int = 250
    mps_tol: float = 1e-6
    # Proper scoring rule diagnostics.
    # Val-10 Promotion (Review 12): both profiles used this; promoted to default.
    compute_crps: bool = True
    crps_mc_samples: int = 96
    crps_data_subsample: int = 256
    # Diagnostics: decompose CRPS-based uncertainty into aleatoric vs epistemic
    # components using the Gaussian-ensemble surrogate from arXiv:2509.26610.
    # Val-10 Promotion (Review 12): both profiles used this; promoted to default.
    compute_crps_uq_decomposition: bool = True
    # MNPO: TriTrust weighting (enabled by default when criterion=mnpo_oracle; still opt-in overall).
    mnpo_use_tritrust: bool = True
    # MNPO: include CRPS as an additional oracle when criterion=mnpo_oracle (opt-in).
    mnpo_include_crps: bool = False
    # MNPO: include a cheap prequential/holdout predictive oracle (opt-in).
    mnpo_include_preq: bool = False
    # MNPO: additional oracle extensions (all opt-in; defaults preserve baseline behavior).
    mnpo_use_tail_risk_oracle: bool = False
    mnpo_tail_risk_alpha: float = 0.33
    mnpo_use_qre_smoothing: bool = False
    mnpo_qre_temperature_gamma: float = 1.0
    mnpo_use_oracle_redundancy_penalty: bool = False
    mnpo_compute_tremble_sensitivity: bool = False
    preq_holdout_fraction: float = 0.20
    preq_min_train: int = 20
    preq_max_test_points: int = 128
    random_state: Optional[int] = None
    n_jobs: int = 1


@dataclass
class ClassificationConfig:
    """Stage-2 classifier backend configuration."""

    selection_mode: str = "legacy"  # legacy | mnpo_hybrid | tune_first
    backend: str = "sklearn"  # sklearn | flaml | optuna
    flaml_time_budget: int = 60
    optuna_time_budget: int = 120
    optuna_n_trials: int = 25
    flaml_estimator_list: Tuple[str, ...] = ("lgbm", "xgboost", "rf", "extra_tree", "lrl2")
    flaml_metric: str = "macro_f1"
    # Val-11 Promotion (Profile D): expanded 8-model candidate set with runtime cap.
    model_candidates: Tuple[str, ...] = (
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
    )
    exclude_model_candidates: Tuple[str, ...] = tuple()
    regime_candidate_exclusions: Tuple[str, ...] = tuple()
    oracle_complexity_prior_overrides: Tuple[str, ...] = tuple()
    include_elastic_net_model: bool = True
    include_rf_model: bool = True
    include_knn_model: bool = True
    include_svm_linear_model: bool = True
    include_dlda_model: bool = True
    include_nsc_model: bool = False
    include_pls_da_model: bool = False
    include_gpc_model: bool = False
    include_nb_model: bool = True
    include_vote_ensemble_model: bool = False
    include_rp_ensemble_model: bool = False
    include_dbda_model: bool = False
    include_gqda_model: bool = False
    include_bc_svm_linear_model: bool = False
    include_sglnn_model: bool = False
    include_xgb_model: bool = False
    include_lgbm_model: bool = False
    include_extra_tree_model: bool = False
    include_catboost_model: bool = False
    include_tabpfn_model: bool = False
    use_hybrid_score: bool = False
    hybrid_balanced_weight: float = 0.6
    hybrid_macro_f1_weight: float = 0.4
    # Val-11 Promotion (Profile D): runtime containment + explicit 8-candidate cap.
    runtime_containment_enabled: bool = True
    runtime_max_candidates: int = 8
    runtime_high_p_over_n_threshold: float = 40.0
    runtime_high_class_threshold: int = 6
    runtime_min_class_count_threshold: int = 12
    # T-R-246: train-test gap overfitting guard for tree-based models.
    # Val-11 Promotion (Profile D): enabled with threshold 0.15.
    stage2_max_train_test_gap: float = 0.15
    stage2_tree_complexity_penalty_enabled: bool = True
    stage2_tree_complexity_penalty_strength: float = 0.1
    lr_max_iter: int = 10000
    # T-R-206: safety guards.
    min_n_for_automl: int = 50
    min_n_per_class_for_cv: int = 5
    min_n_per_class_for_automl: int = 10
    max_p_over_n_for_automl: int = 200
    # MNPO hybrid classifier-oracle controls (used when selection_mode=mnpo_hybrid).
    oracle_k: int = 1
    oracle_weighting_mode: str = "tritrust"  # tritrust | uniform | shapley | banzhaf
    oracle_include_robustness: bool = True
    oracle_include_complexity: bool = True
    oracle_include_calibration: bool = True
    oracle_include_james_stein: bool = True
    oracle_include_cvar: bool = False
    oracle_cvar_alpha: float = 0.33
    oracle_use_dynamic_complexity: bool = False
    oracle_portfolio_diversity: bool = False
    oracle_portfolio_overlap_threshold: float = 0.75
    oracle_portfolio_corr_threshold: float = 0.85
    oracle_enable_hoeffding_racing: bool = True
    oracle_hoeffding_delta: float = 0.10
    oracle_enable_bbc: bool = True
    oracle_bbc_bootstrap_rounds: int = 200
    oracle_bbc_ci_level: float = 0.90
    oracle_enable_ensemble: bool = False
    oracle_multiclass_ensemble_threshold: int = 4  # auto-enable ensemble for K >= this
    oracle_ensemble_voting_mode: str = "hard"  # hard | soft (B2: soft voting with Nash weights)
    oracle_greedy_ensemble: bool = False  # B1: Caruana-style greedy forward selection with replacement
    oracle_greedy_ensemble_rounds: int = 10  # B1: max rounds for greedy forward selection
    oracle_candidate_pruning: bool = False  # B3: pre-game Banzhaf pruning of negative-contribution candidates
    oracle_candidate_pruning_threshold: float = 0.0  # B3: prune candidates with Banzhaf value <= this
    oracle_incumbent_early_stopping: bool = False  # B8: incumbent-relative Hoeffding racing
    oracle_behavior_profile: str = "current"  # current | val18_compat
    oracle_use_per_family_flaml: bool = True
    # T-CS-030: post-FS ratio augmentation at Stage-2 classifier training.
    # Val-10 Promotion (Review 12): both profiles used this; promoted to default.
    stage2_ratio_augmentation_enabled: bool = True
    stage2_ratio_max_features: int = 16
    stage2_ratio_selection_method: str = "correlation"  # correlation | ktsp
    stage2_ratio_epsilon: float = 1e-6
    # T-R-214: split-conformal prediction-set diagnostics.
    # Val-10 Promotion (Review 12): both profiles used this; promoted to default.
    conformal_enabled: bool = True
    conformal_alpha: float = 0.10
    conformal_calibration_fraction: float = 0.25
    conformal_min_calibration: int = 20
    conformal_output_sets: bool = False
    # VAL12_Suggestions §2.3: conformal method selection (split, aps, raps, cross).
    conformal_method: str = "split"
    n_jobs: int = 1

    def __post_init__(self) -> None:
        selection_mode = str(self.selection_mode or "legacy").strip().lower()
        if selection_mode not in {"legacy", "mnpo_hybrid", "tune_first"}:
            raise ValueError("classification.selection_mode must be one of: legacy, mnpo_hybrid, tune_first")
        self.selection_mode = selection_mode
        mode = str(self.backend or "sklearn").strip().lower()
        if mode not in {"sklearn", "flaml", "optuna"}:
            raise ValueError("classification.backend must be one of: sklearn, flaml, optuna")
        self.backend = mode
        self.flaml_time_budget = int(max(1, self.flaml_time_budget))
        self.optuna_time_budget = int(max(1, self.optuna_time_budget))
        self.optuna_n_trials = int(max(1, self.optuna_n_trials))
        self.flaml_estimator_list = tuple(str(e).strip() for e in self.flaml_estimator_list if str(e).strip())
        self.flaml_metric = str(self.flaml_metric or "macro_f1").strip().lower()
        self.model_candidates = tuple(str(c).strip() for c in self.model_candidates if str(c).strip())
        self.exclude_model_candidates = tuple(
            str(c).strip() for c in getattr(self, "exclude_model_candidates", tuple()) if str(c).strip()
        )
        self.regime_candidate_exclusions = tuple(
            str(c).strip() for c in getattr(self, "regime_candidate_exclusions", tuple()) if str(c).strip()
        )
        self.oracle_complexity_prior_overrides = tuple(
            str(c).strip()
            for c in getattr(self, "oracle_complexity_prior_overrides", tuple())
            if str(c).strip()
        )
        self.hybrid_balanced_weight = float(max(0.0, self.hybrid_balanced_weight))
        self.hybrid_macro_f1_weight = float(max(0.0, self.hybrid_macro_f1_weight))
        self.runtime_max_candidates = int(max(0, self.runtime_max_candidates))
        self.runtime_high_p_over_n_threshold = float(max(0.0, self.runtime_high_p_over_n_threshold))
        self.runtime_high_class_threshold = int(max(2, self.runtime_high_class_threshold))
        self.runtime_min_class_count_threshold = int(max(1, self.runtime_min_class_count_threshold))
        self.lr_max_iter = int(max(500, self.lr_max_iter))
        self.min_n_for_automl = int(max(2, self.min_n_for_automl))
        self.min_n_per_class_for_cv = int(max(2, self.min_n_per_class_for_cv))
        self.min_n_per_class_for_automl = int(max(2, self.min_n_per_class_for_automl))
        self.max_p_over_n_for_automl = int(max(1, self.max_p_over_n_for_automl))
        self.oracle_k = int(max(1, self.oracle_k))
        oracle_mode = str(self.oracle_weighting_mode or "tritrust").strip().lower()
        if oracle_mode not in {"tritrust", "uniform", "shapley", "banzhaf"}:
            oracle_mode = "tritrust"
        self.oracle_weighting_mode = oracle_mode
        oracle_behavior_profile = str(self.oracle_behavior_profile or "current").strip().lower()
        if oracle_behavior_profile not in {"current", "val18_compat"}:
            oracle_behavior_profile = "current"
        self.oracle_behavior_profile = oracle_behavior_profile
        self.oracle_cvar_alpha = float(np.clip(self.oracle_cvar_alpha, 1e-3, 0.95))
        self.oracle_hoeffding_delta = float(np.clip(self.oracle_hoeffding_delta, 1e-6, 0.99))
        self.oracle_bbc_bootstrap_rounds = int(max(0, self.oracle_bbc_bootstrap_rounds))
        self.oracle_bbc_ci_level = float(np.clip(self.oracle_bbc_ci_level, 0.50, 0.999))
        self.oracle_multiclass_ensemble_threshold = int(max(0, self.oracle_multiclass_ensemble_threshold))
        ensemble_voting_mode = str(getattr(self, "oracle_ensemble_voting_mode", "hard") or "hard").strip().lower()
        if ensemble_voting_mode not in {"hard", "soft"}:
            ensemble_voting_mode = "hard"
        self.oracle_ensemble_voting_mode = ensemble_voting_mode
        self.oracle_greedy_ensemble_rounds = int(max(1, getattr(self, "oracle_greedy_ensemble_rounds", 10) or 10))
        self.oracle_candidate_pruning_threshold = float(
            getattr(self, "oracle_candidate_pruning_threshold", 0.0) or 0.0
        )
        self.oracle_portfolio_overlap_threshold = float(
            np.clip(self.oracle_portfolio_overlap_threshold, 0.0, 1.0)
        )
        self.oracle_portfolio_corr_threshold = float(
            np.clip(self.oracle_portfolio_corr_threshold, 0.0, 1.0)
        )
        self.stage2_ratio_max_features = int(max(0, self.stage2_ratio_max_features))
        stage2_ratio_method = str(self.stage2_ratio_selection_method or "correlation").strip().lower()
        if stage2_ratio_method not in {"correlation", "ktsp"}:
            stage2_ratio_method = "correlation"
        self.stage2_ratio_selection_method = stage2_ratio_method
        self.stage2_ratio_epsilon = float(max(1e-12, self.stage2_ratio_epsilon))
        self.conformal_alpha = float(np.clip(self.conformal_alpha, 1e-4, 0.49))
        self.conformal_calibration_fraction = float(np.clip(self.conformal_calibration_fraction, 0.05, 0.95))
        self.conformal_min_calibration = int(max(2, self.conformal_min_calibration))
        conformal_method = str(getattr(self, "conformal_method", "split") or "split").strip().lower()
        if conformal_method not in {"split", "aps", "raps", "cross"}:
            conformal_method = "split"
        self.conformal_method = str(conformal_method)


@dataclass
class DFFSConfig:
    random_seed: int = 42
    test_size: float = 0.20
    # Optional: Cap training set size to enforce HDLSS conditions on large datasets.
    # If set, this overrides test_size (effectively maximizing test_size).
    max_train_samples: Optional[int] = None
    fs_fraction: float = 0.40
    n_final_features: int = 50
    n_jobs: int = 1

    # RP-1: Ratio / log-ratio feature construction (opt-in).
    #
    # When enabled, append up to `max_ratio_features` new log-ratio columns
    # computed from pre-screened feature pairs using a cheap supervised signal
    # (k-TSP-style reversal gap or correlation heuristics).
    #
    # Disabled by default to preserve baseline behavior.
    enable_ratio_features: bool = False
    ratio_pool_size: int = 80
    ratio_selection_method: str = "ktsp"  # ktsp | correlation
    ratio_max_pairs: int = 12000
    max_ratio_features: int = 30
    ratio_epsilon: float = 1e-6
    ratio_include_originals: bool = True
    ratio_abs_value: bool = False
    ratio_require_positive: bool = True

    # Distribution fitting controls
    dist_config: DistributionFitterConfig = field(default_factory=DistributionFitterConfig)
    dist_criterion: str = "simple"
    apply_cdf_transform: bool = True
    df_stage_position: str = "after_fs"  # before_fs | after_fs
    # Deprecated (T-DS2): legacy DF fast-path controls are retained as no-op
    # compatibility fields so old configs/CLI flags do not crash.
    # The heuristic path has been fully removed.
    df_fastpath_enabled: bool = False
    df_fastpath_trigger: str = "small_n_or_low_unique"  # small_n | low_unique | small_n_or_low_unique | small_n_and_low_unique
    df_fastpath_small_n_threshold: int = 250
    df_fastpath_unique_ratio_threshold: float = 0.05
    df_fastpath_n_unique_threshold: int = 12
    cdf_reliability_gate: bool = True
    cdf_min_gof_p: float = 0.005
    cdf_max_confidence_set: int = 8
    cdf_skip_heaped_features: bool = False
    cdf_block_gating_cv: bool = False
    cdf_block_gating_n_blocks: int = 4
    cdf_block_gating_min_block_size: int = 8
    cdf_block_gating_cv_splits: int = 2
    cdf_block_gating_max_blocks: int = 6
    cdf_block_gating_time_budget_sec: float = 12.0
    cdf_block_gating_min_improvement: float = 0.0
    max_dist_features: Optional[int] = 256
    low_gof_downweighting: bool = True
    low_gof_threshold: float = 0.01
    low_gof_weight: float = 0.50
    multimodal_fallback: str = "gmm"  # none | gmm | rank_transform

    # Additional reliability signal from distribution stability
    use_distribution_stability_weight: bool = False
    stability_bootstrap: int = 3

    # Feature prefilter inspired by stability/rank integration literature
    use_rank_prefilter: bool = True
    prefilter_top_k: Optional[int] = 600
    # T-003: Configurable MI/F-test blend weights for Tier 1 prefilter scoring.
    # Defaults reproduce the historical hard-coded 60/40 blend.
    prefilter_mi_weight: float = 0.60
    prefilter_f_weight: float = 0.40
    # T-R-127: optional multi-strategy prefilter union.
    # Val-11 Promotion (Profile D): multi-strategy prefilter enabled by default.
    prefilter_union_enabled: bool = True
    # Val-13 Promotion (T-R-265): added bh_fdr strategy to prefilter union.
    prefilter_strategies: Tuple[str, ...] = ("mi_ftest_blend", "rf_importance", "wsnr", "bh_fdr")
    prefilter_nondefault_budget_fraction: float = 0.10
    # T-R-265: BH-adjusted FDR prefilter toggle (controls whether bh_fdr is in strategies).
    prefilter_bh_ttest_enabled: bool = True
    prefilter_bh_ttest_alpha: float = 0.05
    # T-R-272: variance floor — remove near-constant features before main FS.
    prefilter_variance_floor_enabled: bool = True
    prefilter_variance_floor_threshold: float = 1e-6
    prefilter_variance_floor_mode_freq: float = 0.99
    # T-R-172: binary WSNR prefilter strategy (union strategy).
    # Val-11 Promotion (Profile D): WSNR enabled by default.
    prefilter_wsnr_enabled: bool = True
    prefilter_data_domain: str = "auto"  # auto | rnaseq | generic
    prefilter_rnaseq_transform_enabled: bool = True
    prefilter_rnaseq_transform_force: bool = False
    # Val-10 Promotion (Review 12): both profiles used this; promoted to default.
    prefilter_rnaseq_nb_lrt_enabled: bool = True
    prefilter_rnaseq_nb_lrt_alpha: float = 0.10
    # T-R-145/T-R-149: optional train-fold-only batch correction (opt-in).
    batch_correction: str = "none"  # none | combat | combat_seq | cdf_center | center_scale
    batch_correction_combat_prior_strength: float = 8.0
    batch_correction_cdf_n_quantiles: int = 33
    batch_correction_cdf_clip_low: float = 0.01
    batch_correction_cdf_clip_high: float = 0.99

    # Base scaler applied before distribution transform & classification.
    scaler_mode: str = "standard"  # standard | robust | quantile

    # Tier 2 interaction-aware screening (T-004).
    #
    # This runs *after* Tier 1 prefilter and before method-level selection.
    # Val-11 Promotion (Profile D): e-value screening enabled by default.
    screening_enabled: bool = True
    screening_method: str = "evalue"  # "stir" | "evalue" | "none"
    screening_pool_cap: int = 2000
    screening_stir_n_neighbors: int = 10
    screening_stir_n_iter: int = 50
    screening_stir_keep_fraction: float = 0.5
    screening_stir_min_features: int = 20
    screening_evalue_alpha: float = 0.20
    screening_evalue_min_features: int = 20
    # A24/A23 controls: optional folding stage applied after rank prefilter.
    # T-R-171: production default promotes PLS-DA folding with class-count gate.
    folding_method: str = "pls_da"  # none | rff | tensor_sketch | pls_da
    folding_n_components: int = 512
    # When None, gamma is auto-computed as 1/n_features (sklearn "scale" default
    # for unit-variance standardized features).
    folding_rff_gamma: Optional[float] = None
    # A23: supervised PLS-DA folding controls (opt-in via folding_method=pls_da).
    folding_pls_components: int = 32
    folding_pls_scale: bool = True
    # Val-3 Guardrail: PLS-DA regressions observed on datasets with <5 classes
    # (e.g. -0.08 TOX at C=4). Gains only reliable at C>=5 (r=+0.378).
    folding_pls_min_classes: int = 5
    # T-R-177: continuous PLS-DA gating controls.
    folding_pls_min_n_per_class: int = 3
    folding_pls_max_imbalance_ratio: float = 6.0
    folding_prefilter_k: Optional[int] = None
    # A21 controls: opt-in face-domain projection (Fisherfaces-style PCA->LDA).
    enable_face_domain_projection: bool = False
    use_balanced_fs_subsample: bool = False
    fs_min_per_class: int = 2
    # OP1 operational controls:
    # - per-method timeout to prevent single selector blow-ups
    # - higher liblinear iteration cap for LinearSVC-based selectors
    fs_method_timeout_seconds: float = 0.0
    fs_linear_svm_max_iter: int = 10000
    # OP1.2 runtime-aware FS candidate racing (opt-in).
    fs_runtime_racing_enabled: bool = False
    fs_runtime_racing_proxy_splits: int = 1
    fs_runtime_racing_keep_fraction: float = 0.60
    fs_runtime_racing_min_candidates: int = 4
    fs_runtime_racing_runtime_weight: float = 0.15
    # OP1.3: confidence-aware successive-halving racing redesign (opt-in).
    fs_runtime_racing_mode: str = "single_stage"  # single_stage | successive_halving
    fs_runtime_racing_stages: int = 2
    fs_runtime_racing_confidence_bound: str = "none"  # none | hoeffding | bernstein
    fs_runtime_racing_delta: float = 0.10

    # Feature selector controls (FeatureExtractorImplementation defaults)
    selection_strategy: str = "mnpo_portfolio"  # mnpo_portfolio | legacy_voting
    enabled_methods: Tuple[str, ...] = (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        # Opt-in multiclass-only refinement (guarded by fs_ova_min_classes).
        "ova_ensemble",
        # Val-3 Promotion: A28 class-specific Pareto selector (guarded by min_classes).
        "class_pareto_front",
    )
    # Optional composed FeatureSelectorConfig (T-A3-010). When provided, pipeline
    # builds the selector via FeatureSelector.from_config(fs_config) and applies
    # run-specific overrides (seed/enabled methods) before execution.
    fs_config: Optional[Any] = None
    # T-P3-002: easy-tier lockout ("do no harm") controls.
    tier_lockout_enabled: bool = False
    tier_lockout_tier: str = "easy"
    tier_lockout_difficulty_source: str = "historical"  # historical | meta_features
    # Fallback stack used when lockout triggers (production default profile).
    tier_lockout_fallback_methods: Tuple[str, ...] = (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "joint_multiclass_support",
    )
    # T-P3-005: tier-conditional method routing controls.
    tier_routing_enabled: bool = False
    tier_routing_difficulty_classifier: str = "meta_features"  # historical | meta_features
    tier_routing_table: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # T-R-257/T-R-258/T-R-259: regime-conditional FS gating (Val-12, opt-in).
    # Routes Profile-D defaults to safer/simple behavior on known failure regimes.
    regime_gating_enabled: bool = False
    regime_gating_difficulty_source: str = "historical"  # historical | meta_features
    regime_gating_target_tier: str = "very_hard"
    regime_gating_min_samples_per_class: float = 15.0
    # CC-1 (Val-15): disable low-p/n bypass by default; set >0 to re-enable.
    regime_gating_low_p_over_n_threshold: float = 0.0
    # Simple fallback stack (Profile-A-like) used when very-hard gate triggers.
    regime_gating_simple_methods: Tuple[str, ...] = (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
    )
    # Very-hard safeguard overrides (T-R-258).
    regime_gating_very_hard_portfolio_max_methods: int = 4
    regime_gating_very_hard_copula_derandomize_runs: int = 5
    # Low p/n safeguard behavior (T-R-259/T-R-269): fast univariate filter replaces all_features.
    # Val-13 Fix (T-R-269): 'all_features' caused catastrophic regression (madelon -0.118,
    # dorothea -0.186). 'fast_univariate_filter' applies ANOVA F-test/Welch t-test and keeps
    # top min(200, p/2) features instead of bypassing FS entirely.
    regime_gating_low_p_over_n_mode: str = "fast_univariate_filter"  # fast_univariate_filter | all_features
    # T-R-269: max features to keep when fast_univariate_filter is active.
    regime_gating_low_p_over_n_filter_max_k: int = 200
    # T-R-270: Gate 1 class-count qualifier — only trigger very-hard fallback when c >= this.
    # Prevents over-triggering on low-class datasets like glioma_50 (c=4, lost -0.050).
    regime_gating_very_hard_min_classes: int = 5
    # T-R-268: extreme multiclass gate (Gate 3) — when c >= threshold AND n/c >= guard,
    # keep MNPO FS but switch classifier to OVA ensemble mode.
    regime_gating_extreme_multiclass_enabled: bool = True
    regime_gating_extreme_multiclass_threshold: int = 8
    regime_gating_extreme_multiclass_min_samples_per_class: float = 11.0
    # MNPO portfolio sizing controls.
    #
    # Note: `portfolio_size` is the maximum number of selector candidates that
    # participate in the MNPO portfolio weighting step (not the final number of
    # selected features). Defaults preserve the current production baseline.
    fs_portfolio_size: int = 5
    # Val-11 Promotion (Profile D): adaptive portfolio sizing with warn guard.
    fs_portfolio_size_guard: str = "warn"  # none | warn | raise
    fs_adaptive_portfolio_sizing_enabled: bool = True
    fs_adaptive_size_min: Optional[int] = 4
    fs_adaptive_size_max: Optional[int] = 8
    fs_adaptive_sizing_variance_penalty: bool = False
    fs_adaptive_sizing_variance_penalty_strength: float = 0.5
    # T-R-266: Pareto-front adaptive portfolio sizing (opt-in, replaces fixed min/max bounds).
    fs_pareto_portfolio_sizing_enabled: bool = False
    # T-R-271: stability-weighted portfolio aggregation (opt-in).
    fs_stability_weighted_aggregation_enabled: bool = False
    fs_mnpo_paradigm_aware_prior_enabled: bool = False
    fs_mnpo_interaction_floor: float = 0.12
    fs_rashomon_enabled: bool = False
    fs_rashomon_max_models: int = 12
    fs_rashomon_score_tolerance: float = 0.01
    # Optional: exclude selected methods from the MNPO candidate library when
    # synthetic consensus candidates are present (helps avoid double-counting
    # in portfolio-sizing experiments).
    #
    # Disabled by default (empty tuple) to preserve baseline behavior.
    fs_mnpo_consensus_exclude_methods: Tuple[str, ...] = tuple()
    # When >0, protect the top-k candidates (by MNPO equilibrium weight) from
    # consensus-exclusion filtering.
    fs_mnpo_consensus_exclude_protect_top_k: int = 0
    # Optional: disable synthetic consensus candidates when experimenting with
    # larger MNPO portfolios (A8.1 diagnostic). Enabled by default to preserve
    # baseline behavior.
    fs_mnpo_include_legacy_consensus: bool = True
    fs_mnpo_include_majority_consensus: bool = True
    # Validation/benchmark runtime defaults use a smaller inner CV than the
    # library defaults; expose it so Val-18 fold-resolution profiles can
    # request 5x5 explicitly without changing baseline behavior.
    fs_inner_cv_splits: int = 3
    fs_inner_cv_repeats: int = 1
    use_tritrust: bool = True
    use_stability_oracle: bool = True
    use_complexity_oracle: bool = True
    use_robust_oracle: bool = True
    # Val-11 Promotion (Profile D): diversity oracle enabled by default.
    use_diversity_oracle: bool = True
    # T-R-181/T-R-184/T-R-186: new oracle controls.
    fs_use_cvar_oracle: bool = False
    fs_cvar_alpha: float = 0.33
    # Val-11 Promotion (Profile D): Banzhaf oracle weighting (lower variance than
    # tritrust/Shapley under noisy evaluations; Wang & Jia 2022).
    fs_oracle_weighting_mode: str = "banzhaf"  # tritrust | uniform | shapley | banzhaf
    fs_shapley_n_coalitions_max: int = 2048
    fs_shapley_bayesian_shrinkage: bool = False
    fs_shapley_bayesian_prior_strength: float = 8.0
    fs_use_interaction_oracle: bool = False
    fs_interaction_oracle_min_n_train: int = 150
    fs_interaction_oracle_pool_size_cap: int = 64
    fs_interaction_oracle_pair_cap: int = 20000
    fs_use_ubayfs_oracle: bool = False
    fs_ubayfs_n_bootstrap: int = 32
    fs_ubayfs_min_n: int = 100
    fs_ubayfs_prior_weight: float = 0.0
    fs_use_conformal_uq: bool = False
    fs_conformal_uq_alpha: float = 0.10
    fs_conformal_uq_min_folds: int = 5
    fs_fold_preference_mode: str = "logistic"  # Val-17 CC-V17-4: CLP promotion (p=0.173 trending, included in full-stack p=0.033)
    fs_use_conformal_efficiency: bool = True  # Val-17 CC-V17-3: full-stack promotion
    fs_conformal_efficiency_method: str = "aps"  # Val-17 CC-V17-3: full-stack promotion
    fs_oracle_weight_js_shrinkage: bool = True  # Val-17 CC-V17-3: full-stack promotion
    fs_payoff_shrinkage_kappa: float = 0.15  # Val-17 CC-V17-3: full-stack promotion (p=0.033)
    # Additional MNPO oracle extensions (opt-in; defaults preserve baseline behavior).
    use_tail_risk_oracle: bool = False
    tail_risk_alpha: float = 0.33
    use_regret_oracle: bool = False
    use_qre_smoothing: bool = False
    qre_temperature_gamma: float = 1.0
    use_oracle_redundancy_penalty: bool = False
    compute_tremble_sensitivity: bool = False
    # T-002: construct per-model performance oracles when multi-classifier fold
    # scoring is enabled (see eval_models_enabled below).
    # Val-11 Promotion (Profile D): multi-model oracle mode for richer FS signal.
    mnpo_performance_oracle_mode: str = "multi_model_oracles"  # "single" | "multi_model_oracles"
    fs_diversity_oracle_mode: str = "legacy_jaccard"
    fs_diversity_redundancy_weight: float = 0.6
    fs_diversity_complementarity_weight: float = 0.35
    fs_performance_balanced_weight: float = 0.6
    fs_performance_macro_f1_weight: float = 0.4
    fs_performance_use_adaptive_imbalance: bool = False
    fs_performance_imbalance_ratio_trigger: float = 1.75
    fs_performance_min_classes_for_adaptive: int = 3
    # T-001: Multi-classifier evaluation proxy.
    # Val-11 Promotion (Profile D): multi-classifier eval enabled by default.
    eval_models_enabled: bool = True
    eval_models: Tuple[str, ...] = ("lr_l2", "linear_svc", "rf_small")
    eval_aggregate: str = "mean"  # "mean" | "min" | "cvar"
    eval_cvar_alpha: float = 0.33
    fs_rank_aggregation_mode: str = "none"
    fs_wrapper_refine_enabled: bool = False
    fs_wrapper_refine_top_k: int = 24
    fs_wrapper_refine_max_add: int = 12
    fs_wrapper_refine_min_gain: float = 1e-4
    fs_ova_negative_ratio: float = 2.0
    # Empirically, OVA can help on hard-tier multiclass datasets but can be harmful
    # (cap-violating) on easier/medium datasets with few classes. Keep it gated by
    # default (>=5 classes) and only run it when explicitly appropriate.
    fs_ova_min_classes: int = 5
    # Optional OVA stabilization controls for rare-class multiclass settings.
    # Defaults preserve legacy behavior (min_pos_samples=2, uniform weighting).
    fs_ova_min_pos_samples: int = 2
    fs_ova_class_weight_mode: str = "uniform"  # uniform | sqrt_pos | pos | log_pos | inv_pos | inv_sqrt_pos | inv_log_pos
    fs_ova_aggregation_mode: str = "mean"  # mean | p_norm
    fs_ova_aggregation_p: float = 4.0
    fs_ova_linear_backend: str = "linear_svm_l1"
    # A29 extension: optional probability-calibration weighting in OVA scoring.
    fs_ova_enable_calibration: bool = False
    fs_ova_calibration_cv: int = 3
    # A12 pilot controls: ECOC class-aware multiclass decomposition selector (opt-in).
    fs_ecoc_min_classes: int = 4
    fs_ecoc_max_ovo_pairs: int = 8
    fs_ecoc_random_code_bits: int = 4
    fs_ecoc_class_complexity_weight: float = 1.0
    fs_ecoc_include_ova_tasks: bool = True
    fs_ecoc_negative_ratio: float = 2.0
    # A14 pilot controls: joint multinomial shared-support multiclass selector (opt-in).
    fs_joint_multiclass_min_classes: int = 3
    fs_joint_multiclass_max_features: int = 256
    fs_joint_multiclass_path_grid_size: int = 6
    fs_joint_multiclass_min_c: float = 0.05
    fs_joint_multiclass_max_c: float = 1.6
    fs_joint_multiclass_l1_ratio: float = 0.55
    fs_joint_multiclass_univariate_blend: float = 0.20
    # A19 pilot controls: class-specific relevance matrix + DOvE-style multiclass path (opt-in).
    fs_dove_min_classes: int = 3
    fs_dove_max_pairs_per_class: int = 4
    fs_dove_path_grid_size: int = 5
    fs_dove_specificity_weight: float = 0.35
    fs_dove_minority_boost: float = 0.50
    # A20 pilot controls: sparse multinomial multiclass backend (opt-in).
    fs_sparse_multinomial_min_classes: int = 3
    fs_sparse_multinomial_max_features: int = 320
    fs_sparse_multinomial_path_grid_size: int = 6
    fs_sparse_multinomial_min_c: float = 0.05
    fs_sparse_multinomial_max_c: float = 1.6
    fs_sparse_multinomial_backend: str = "mixed"  # l1 | elasticnet | mixed
    fs_sparse_multinomial_l1_ratio: float = 0.70
    fs_sparse_multinomial_univariate_blend: float = 0.20
    fs_sparse_multinomial_max_iter: int = 5000
    # A20-R1 runtime containment: screening-first sparse multinomial controls.
    fs_sparse_multinomial_screening_mode: str = "none"  # none | prefilter_aggressive | prefilter_balanced | prefilter_conservative
    fs_sparse_multinomial_screening_keep_fraction: float = 1.0
    fs_sparse_multinomial_screening_min_features: int = 64
    fs_sparse_multinomial_screening_fallback_on_failure: bool = True
    # A22 controls: nearest shrunken centroids selector (PAM/NSC-style).
    fs_nsc_shrinkage_grid_size: int = 6
    fs_nsc_min_classes: int = 3
    fs_nsc_thresholding_mode: str = "soft"  # soft | hard | order | auto
    fs_nsc_order_quantile: float = 0.75
    fs_nsc_deep_shrinkage_search: bool = False
    # A28 controls: class-specific Pareto-front multiclass selector (opt-in).
    fs_class_pareto_min_classes: int = 3
    fs_class_pareto_top_per_class: int = 64
    fs_class_pareto_global_fraction: float = 0.40
    fs_class_pareto_minority_boost: float = 0.50
    fs_class_pareto_kw_weight: float = 0.25
    # T-R-166 controls: classical SDR selectors (SIR/SAVE/PFC).
    fs_sdr_min_classes: int = 3
    fs_sdr_prefilter_max_features: int = 512
    fs_sdr_n_components: int = 3
    fs_sdr_covariance_ridge: float = 1e-3
    # A26: per-class quota allocation overlay for class-specific selectors.
    # Val-3 Promotion: enabled by default.
    fs_per_class_quota_enabled: bool = True
    fs_per_class_quota_min_per_class: int = 1
    fs_per_class_quota_max_fraction: float = 0.60
    # A25 controls: HSIC Lasso kernelized selector pilot (opt-in).
    fs_hsic_lasso_alpha: float = 0.01
    fs_hsic_lasso_prefilter_max_features: int = 128
    fs_hsic_lasso_feature_sigma: float = 0.0
    fs_hsic_lasso_target_sigma: float = 0.0
    fs_hsic_lasso_relevance_blend: float = 0.20
    fs_hsic_lasso_max_iter: int = 4000
    # Val-11 Promotion (Profile D): MI redundancy in mRMR enabled by default.
    fs_mrmr_mi_redundancy_enabled: bool = True
    fs_mrmr_mi_n_bins: int = 8
    fs_cmim_min_samples: int = 60
    fs_cmim_n_bins: int = 8
    fs_fcbf_n_bins: int = 8
    fs_ipss_path_grid_size: int = 7
    fs_ipss_min_c: float = 0.08
    fs_ipss_max_c: float = 1.20
    fs_ipss_target_fdr: float = 0.15
    fs_ipss_null_shuffle_rounds: int = 1
    fs_ipss_use_eats_threshold: bool = False
    fs_ipss_eats_exclusion_quantile: float = 0.90
    fs_ipss_eats_min_threshold: float = 0.45
    fs_ipss_importance_model: str = "linear_svm"
    # Optional IPSS gating: skip IPSS on regimes where it tends to displace
    # strong simple rankers (e.g. small-n binary/few-class problems).
    #
    # Disabled by default (0/0.0) to preserve baseline behavior.
    fs_ipss_gate_min_classes: int = 0
    fs_ipss_gate_min_p_over_n: float = 0.0
    fs_cluster_stability_corr_threshold: float = 0.85
    fs_cluster_stability_max_per_cluster: int = 2
    fs_cluster_stability_min_cluster_freq: float = 0.55
    fs_stability_use_loss_guided_validation: bool = False
    fs_stability_validation_fraction: float = 0.25
    fs_stability_validation_quantile: float = 0.40
    fs_stability_validation_min_samples: int = 6
    fs_copula_knockoff_draws: int = 30
    fs_copula_alpha_kn: float = 0.10
    fs_copula_alpha_ebh: float = 0.20
    fs_copula_truncation_level: Optional[int] = None
    fs_copula_generator: str = "copula"  # copula | deepdrk
    fs_copula_deepdrk_latent_fraction: float = 0.35
    fs_copula_deepdrk_noise_scale: float = 1.0
    # Val-13 Promotion (T-R-267): 5 derandomization runs for copula knockoffs.
    fs_copula_derandomize_runs: int = 5
    fs_copula_stabilizer_runs: int = 1
    fs_copula_stabilizer_use_ebh: bool = False
    fs_copula_stabilizer_seed_stride: int = 997
    # T-P3-006: importance uncertainty reporting (diagnostic only).
    fs_importance_uq_enabled: bool = False
    fs_importance_uq_min_cv_folds: int = 3
    fs_decorrelated_stability_eps: float = 1e-3
    # VAL12_Suggestions §3.3: Group-aware feature selection (opt-in, experimental).
    fs_group_sparse_lasso_alpha: float = 0.1
    fs_group_sparse_lasso_distance_threshold: float = 0.7
    # VAL12_Suggestions §4.2: TabPFN benchmark lane (opt-in, benchmark only).
    benchmark_tabpfn_enabled: bool = False
    # VAL12_Suggestions §5.1: EATS threshold calibration (opt-in).
    fs_stability_threshold_method: str = "fixed"  # fixed | eats | cpss
    fs_stability_target_pfer: float = 1.0
    # VAL12_Suggestions §5.2: knockpy/GRIP2 benchmark lane (opt-in).
    benchmark_knockoff_enabled: bool = False
    # VAL12_Suggestions §5.3: IPSS benchmark mode (opt-in).
    benchmark_ipss_enabled: bool = False
    # A13 pilot controls: ieGENES-style iterative redundancy pruning wrapper (opt-in).
    fs_iterative_pruning_pool_factor: float = 2.5
    fs_iterative_pruning_max_rounds: int = 32
    fs_iterative_pruning_min_improvement: float = -0.002
    # Maximum accepted cumulative loss (sum of per-round deltas) before pruning stops.
    fs_iterative_pruning_max_cumulative_loss: float = 0.02
    fs_iterative_pruning_redundancy_weight: float = 0.65
    fs_iterative_pruning_bounded_prefilter_cap: int = 220
    fs_iterative_pruning_bounded_candidate_fraction: float = 0.35
    fs_iterative_pruning_bounded_min_candidates: int = 4
    fs_iterative_pruning_bounded_max_evaluations: int = 48
    fs_iterative_pruning_bounded_max_runtime_seconds: float = 30.0
    fs_iterative_pruning_bounded_enable_class_gating: bool = True
    fs_iterative_pruning_bounded_multiclass_scale: float = 0.70
    fs_iterative_pruning_bounded_imbalance_trigger: float = 2.5
    fs_iterative_pruning_bounded_imbalance_scale: float = 0.75
    # A16 pilot controls: CPSS-style stability overlay for bounded iterative pruning (opt-in).
    fs_iterative_pruning_bounded_use_cpss_overlay: bool = False
    fs_iterative_pruning_bounded_cpss_pairs: int = 4
    fs_iterative_pruning_bounded_cpss_stability_threshold: float = 0.60
    fs_iterative_pruning_bounded_cpss_min_stable_features: int = 2
    fs_iterative_pruning_bounded_cpss_min_jaccard: float = 0.35
    fs_iterative_pruning_bounded_cpss_max_score_drop: float = 0.005
    # A17 pilot controls: class-dominance-aware Pareto prefilter for iterative wrappers (opt-in).
    fs_iterative_pruning_class_pareto_prefilter_enabled: bool = False
    fs_iterative_pruning_class_pareto_min_classes: int = 3
    fs_iterative_pruning_class_pareto_top_per_class: int = 64
    fs_iterative_pruning_class_pareto_global_fraction: float = 0.40
    fs_iterative_pruning_class_pareto_minority_boost: float = 0.50
    # A18 pilot controls: stability-gated class-Pareto follow-up (opt-in).
    fs_iterative_pruning_class_pareto_stability_gate_enabled: bool = False
    fs_iterative_pruning_class_pareto_stability_subsamples: int = 6
    fs_iterative_pruning_class_pareto_stability_fraction: float = 0.70
    fs_iterative_pruning_class_pareto_stability_threshold: float = 0.55
    fs_iterative_pruning_class_pareto_stability_min_overlap: float = 0.50
    fs_iterative_pruning_class_pareto_stability_min_stable_features: int = 4
    fs_iterative_pruning_class_pareto_stability_fallback_on_failure: bool = True
    # Optional decorrelated-stability gating: skip decorrelation when the
    # prefiltered feature pool is already near-orthogonal.
    #
    # Disabled by default (0.0) to preserve baseline behavior.
    fs_decorrelated_stability_min_max_abs_corr: float = 0.0

    # Post-FS feature count safety cap (VAL12_Suggestions / Independent Review E.1).
    # Prevents feature explosion when MNPO FS retains too many features
    # (e.g., dorothea 612 features → overfitting).  Hard architectural guard.
    fs_max_selected_features_ratio: float = 0.5   # max = n_train * ratio
    fs_max_selected_features_cap: int = 500        # absolute cap

    # Final model selection controls
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    # Legacy flat fields (read into `classification` in __post_init__ for
    # backward compatibility with existing configs/tests/scripts).
    classification_selection_mode: str = "legacy"  # legacy | mnpo_hybrid | tune_first
    # Val-11 Promotion (Profile D): 8-model candidate set.
    model_candidates: Tuple[str, ...] = (
        "lr", "svm_rbf", "svm_linear", "dlda", "knn", "rf", "nb", "elastic_net_lr",
    )
    exclude_model_candidates: Tuple[str, ...] = tuple()
    classifier_regime_candidate_exclusions: Tuple[str, ...] = tuple()
    classifier_oracle_complexity_prior_overrides: Tuple[str, ...] = tuple()
    include_elastic_net_model: bool = True
    include_rf_model: bool = True
    include_knn_model: bool = True
    include_svm_linear_model: bool = True
    include_dlda_model: bool = True
    include_nsc_model: bool = False
    include_pls_da_model: bool = False
    include_gpc_model: bool = False
    include_nb_model: bool = True
    include_vote_ensemble_model: bool = False
    include_rp_ensemble_model: bool = False
    include_dbda_model: bool = False
    include_gqda_model: bool = False
    include_bc_svm_linear_model: bool = False
    include_sglnn_model: bool = False
    include_xgb_model: bool = False
    include_lgbm_model: bool = False
    include_extra_tree_model: bool = False
    include_catboost_model: bool = False
    include_tabpfn_model: bool = False
    model_cv_lr_max_iter: int = 10000
    model_cv_use_hybrid_score: bool = False
    model_cv_balanced_weight: float = 0.6
    model_cv_macro_f1_weight: float = 0.4
    # OP1.1 runtime containment: reduce model-CV candidate breadth on
    # regimes that repeatedly trigger long-tail checkpoint tasks.
    # Val-10 Promotion (Review 12): both profiles used this; promoted to default.
    model_cv_runtime_containment_enabled: bool = True
    # If >0, hard-cap candidate count after containment prioritization.
    # Val-11 Promotion (Profile D): explicit 8-candidate cap.
    model_cv_runtime_max_candidates: int = 8
    # Auto-containment triggers (used when max_candidates <= 0).
    model_cv_runtime_high_p_over_n_threshold: float = 40.0
    model_cv_runtime_high_class_threshold: int = 6
    model_cv_runtime_min_class_count_threshold: int = 12
    classifier_oracle_k: int = 1
    classifier_oracle_weighting_mode: str = "tritrust"
    classifier_oracle_include_robustness: bool = True
    classifier_oracle_include_complexity: bool = True
    classifier_oracle_include_calibration: bool = True
    classifier_oracle_include_james_stein: bool = True
    classifier_oracle_include_cvar: bool = False
    classifier_oracle_cvar_alpha: float = 0.33
    classifier_oracle_use_dynamic_complexity: bool = False
    classifier_oracle_portfolio_diversity: bool = False
    classifier_oracle_portfolio_overlap_threshold: float = 0.75
    classifier_oracle_portfolio_corr_threshold: float = 0.85
    classifier_oracle_enable_hoeffding_racing: bool = True
    classifier_oracle_hoeffding_delta: float = 0.10
    classifier_oracle_enable_bbc: bool = True
    classifier_oracle_bbc_bootstrap_rounds: int = 200
    classifier_oracle_bbc_ci_level: float = 0.90
    classifier_oracle_enable_ensemble: bool = False
    classifier_oracle_ensemble_voting_mode: str = "hard"
    classifier_oracle_greedy_ensemble: bool = False
    classifier_oracle_greedy_ensemble_rounds: int = 10
    classifier_oracle_candidate_pruning: bool = False
    classifier_oracle_candidate_pruning_threshold: float = 0.0
    classifier_oracle_incumbent_early_stopping: bool = False
    classifier_oracle_behavior_profile: str = "current"
    classifier_oracle_use_per_family_flaml: bool = True
    # VAL12_Suggestions §2.1-2.2: opt-in calibration reporting (ECE alongside Brier).
    calibration_reporting_enabled: bool = False
    # Val-10 Promotion (Review 12): both profiles used this; promoted to default.
    stage2_ratio_augmentation_enabled: bool = True
    stage2_ratio_max_features: int = 16
    stage2_ratio_selection_method: str = "correlation"
    stage2_ratio_epsilon: float = 1e-6

    # Optional MAQC-II style selector/classifier pairing (opt-in).
    #
    # When enabled, the pipeline evaluates multiple feature-selector method
    # stacks ("selectors") and chooses the selector+classifier pair with the
    # best downstream model-CV score on the strict-holdout training split.
    #
    # This is intentionally *not* the default because it adds compute and can
    # complicate ablations; it is primarily meant for targeted multiclass/HDLSS
    # settings where pairing dominates simple metric reweighting.
    enable_maqc_pairing: bool = False
    maqc_pairing_method_sets: Tuple[Tuple[str, ...], ...] = tuple()
    maqc_pairing_method_set_names: Tuple[str, ...] = tuple()
    maqc_pairing_min_improvement: float = 0.0
    maqc_pairing_min_improvement_se_mult: float = 0.0
    maqc_pairing_score_mode: str = "raw_cv"  # raw_cv | nested_cv | nested_bbc
    maqc_pairing_outer_splits: int = 3
    maqc_pairing_outer_repeats: int = 1
    maqc_pairing_min_train_per_class: int = 2
    maqc_pairing_seed_stride: int = 997
    maqc_pairing_bbc_bootstrap_rounds: int = 200
    maqc_pairing_bbc_ci_level: float = 0.90
    multiomics_adapter: str = "none"
    multiomics_integrator: str = "mb_plsda"
    multiomics_n_components: int = 2
    multiomics_feature_blocks: Optional[Dict[str, Tuple[int, ...]]] = None
    meta_learning_selector_mode: str = "none"
    meta_learning_confidence_threshold: float = 0.55

    def __post_init__(self) -> None:
        self.folding_pls_min_classes = int(max(2, self.folding_pls_min_classes))
        self.folding_pls_min_n_per_class = int(max(1, self.folding_pls_min_n_per_class))
        self.folding_pls_max_imbalance_ratio = float(max(1.0, self.folding_pls_max_imbalance_ratio))
        fallback_mode = str(getattr(self, "multimodal_fallback", "none") or "none").strip().lower()
        if fallback_mode in {"rank", "rank_gaussian", "quantile"}:
            fallback_mode = "rank_transform"
        if fallback_mode not in {"none", "gmm", "rank_transform"}:
            fallback_mode = "none"
        self.multimodal_fallback = str(fallback_mode)
        df_stage_position = str(getattr(self, "df_stage_position", "after_fs") or "after_fs").strip().lower()
        if df_stage_position in {"before", "pre_fs", "pre"}:
            df_stage_position = "before_fs"
        elif df_stage_position in {"after", "post_fs", "post"}:
            df_stage_position = "after_fs"
        if df_stage_position not in {"before_fs", "after_fs"}:
            df_stage_position = "after_fs"
        self.df_stage_position = str(df_stage_position)
        self.prefilter_data_domain = str(self.prefilter_data_domain or "auto").strip().lower()
        if self.prefilter_data_domain not in {"auto", "rnaseq", "generic"}:
            self.prefilter_data_domain = "auto"
        self.prefilter_bh_ttest_alpha = float(
            np.clip(float(getattr(self, "prefilter_bh_ttest_alpha", 0.05) or 0.05), 1e-6, 0.5)
        )
        self.prefilter_rnaseq_nb_lrt_alpha = float(np.clip(self.prefilter_rnaseq_nb_lrt_alpha, 1e-6, 0.5))
        batch_mode = str(getattr(self, "batch_correction", "none") or "none").strip().lower()
        batch_aliases = {
            "cdf": "cdf_center",
            "cdf-center": "cdf_center",
            "cdfcenter": "cdf_center",
            "center_cdf": "cdf_center",
            "combat-seq": "combat_seq",
            "combatseq": "combat_seq",
            "center-scale": "center_scale",
            "centerscale": "center_scale",
        }
        batch_mode = batch_aliases.get(batch_mode, batch_mode)
        if batch_mode not in {"none", "combat", "combat_seq", "cdf_center", "center_scale"}:
            batch_mode = "none"
        self.batch_correction = str(batch_mode)
        self.batch_correction_combat_prior_strength = float(
            max(0.0, float(getattr(self, "batch_correction_combat_prior_strength", 8.0) or 0.0))
        )
        self.batch_correction_cdf_n_quantiles = int(
            max(7, int(getattr(self, "batch_correction_cdf_n_quantiles", 33) or 33))
        )
        low = float(np.clip(getattr(self, "batch_correction_cdf_clip_low", 0.01), 0.0, 0.49))
        high = float(np.clip(getattr(self, "batch_correction_cdf_clip_high", 0.99), 0.51, 1.0))
        if high <= low + 1e-3:
            low, high = 0.01, 0.99
        self.batch_correction_cdf_clip_low = float(low)
        self.batch_correction_cdf_clip_high = float(high)

        # Validate scaler_mode.
        _sm = str(getattr(self, "scaler_mode", "standard") or "standard").strip().lower()
        if _sm not in {"standard", "robust", "quantile"}:
            _sm = "standard"
        self.scaler_mode = _sm

        self.fs_sdr_min_classes = int(max(2, int(getattr(self, "fs_sdr_min_classes", 3) or 3)))
        self.fs_sdr_prefilter_max_features = int(
            max(16, int(getattr(self, "fs_sdr_prefilter_max_features", 512) or 512))
        )
        self.fs_sdr_n_components = int(max(1, int(getattr(self, "fs_sdr_n_components", 3) or 3)))
        self.fs_sdr_covariance_ridge = float(
            max(1e-8, float(getattr(self, "fs_sdr_covariance_ridge", 1e-3) or 1e-3))
        )
        self.fs_interaction_oracle_min_n_train = int(max(2, self.fs_interaction_oracle_min_n_train))
        self.fs_interaction_oracle_pool_size_cap = int(max(4, self.fs_interaction_oracle_pool_size_cap))
        self.fs_interaction_oracle_pair_cap = int(max(1, self.fs_interaction_oracle_pair_cap))
        fs_fold_pref = str(getattr(self, "fs_fold_preference_mode", "vote") or "vote").strip().lower()
        if fs_fold_pref not in {"vote", "logistic"}:
            fs_fold_pref = "vote"
        self.fs_fold_preference_mode = str(fs_fold_pref)
        fs_conf_eff_method = str(
            getattr(self, "fs_conformal_efficiency_method", "split") or "split"
        ).strip().lower()
        if fs_conf_eff_method not in {"split", "aps"}:
            fs_conf_eff_method = "split"
        self.fs_conformal_efficiency_method = str(fs_conf_eff_method)
        self.fs_payoff_shrinkage_kappa = float(
            max(0.0, float(getattr(self, "fs_payoff_shrinkage_kappa", 0.0) or 0.0))
        )
        selection_strategy = str(
            getattr(self, "selection_strategy", "mnpo_portfolio") or "mnpo_portfolio"
        ).strip().lower()
        if selection_strategy not in {"mnpo_portfolio", "legacy_voting"}:
            selection_strategy = "mnpo_portfolio"
        self.selection_strategy = str(selection_strategy)
        multiomics_adapter = str(getattr(self, "multiomics_adapter", "none") or "none").strip().lower()
        if multiomics_adapter in {"metadata", "metadata_block", "feature_metadata", "diablo_blocks"}:
            multiomics_adapter = "metadata_blocks"
        if multiomics_adapter not in {"none", "split_halves", "metadata_blocks"}:
            multiomics_adapter = "none"
        self.multiomics_adapter = str(multiomics_adapter)
        multiomics_integrator = str(
            getattr(self, "multiomics_integrator", "mb_plsda") or "mb_plsda"
        ).strip().lower()
        if multiomics_integrator not in {"mb_plsda", "mint"}:
            multiomics_integrator = "mb_plsda"
        self.multiomics_integrator = str(multiomics_integrator)
        self.multiomics_n_components = int(
            max(1, int(getattr(self, "multiomics_n_components", 2) or 2))
        )
        raw_multiomics_blocks = getattr(self, "multiomics_feature_blocks", None)
        normalized_multiomics_blocks: Optional[Dict[str, Tuple[int, ...]]] = None
        if isinstance(raw_multiomics_blocks, dict):
            block_map: Dict[str, Tuple[int, ...]] = {}
            for raw_name, raw_indices in raw_multiomics_blocks.items():
                name = str(raw_name).strip()
                if not name:
                    continue
                try:
                    idx_arr = np.asarray(list(raw_indices), dtype=int).ravel()
                except Exception:
                    continue
                cleaned = tuple(
                    sorted({int(idx) for idx in idx_arr.tolist() if int(idx) >= 0})
                )
                if cleaned:
                    block_map[name] = cleaned
            if len(block_map) >= 2:
                normalized_multiomics_blocks = dict(block_map)
        self.multiomics_feature_blocks = normalized_multiomics_blocks
        meta_learning_mode = str(
            getattr(self, "meta_learning_selector_mode", "none") or "none"
        ).strip().lower()
        if meta_learning_mode not in {"none", "decision_tree", "logistic"}:
            meta_learning_mode = "none"
        self.meta_learning_selector_mode = str(meta_learning_mode)
        self.meta_learning_confidence_threshold = float(
            np.clip(
                float(getattr(self, "meta_learning_confidence_threshold", 0.55) or 0.55),
                0.0,
                1.0,
            )
        )
        regime_source = str(
            getattr(self, "regime_gating_difficulty_source", "historical") or "historical"
        ).strip().lower()
        if regime_source not in {"historical", "meta_features"}:
            regime_source = "historical"
        self.regime_gating_difficulty_source = str(regime_source)
        regime_tier = str(
            getattr(self, "regime_gating_target_tier", "very_hard") or "very_hard"
        ).strip().lower()
        if regime_tier not in {"easy", "medium", "hard", "very_hard"}:
            regime_tier = "very_hard"
        self.regime_gating_target_tier = str(regime_tier)
        self.regime_gating_min_samples_per_class = float(
            max(1.0, float(getattr(self, "regime_gating_min_samples_per_class", 15.0) or 15.0))
        )
        self.regime_gating_low_p_over_n_threshold = float(
            max(0.0, float(getattr(self, "regime_gating_low_p_over_n_threshold", 0.0) or 0.0))
        )
        raw_simple_methods = getattr(self, "regime_gating_simple_methods", tuple()) or tuple()
        simple_methods_list: List[str] = []
        simple_seen: Set[str] = set()
        for method in raw_simple_methods:
            key = str(method).strip()
            if not key or key in simple_seen:
                continue
            simple_methods_list.append(key)
            simple_seen.add(key)
        simple_methods = tuple(simple_methods_list)
        if not simple_methods:
            simple_methods = (
                "gradient_boosting",
                "linear_svm",
                "mutual_information",
                "anova_f",
                "mrmr_jmi",
            )
        self.regime_gating_simple_methods = tuple(simple_methods)
        self.regime_gating_very_hard_portfolio_max_methods = int(
            max(1, int(getattr(self, "regime_gating_very_hard_portfolio_max_methods", 4) or 4))
        )
        self.regime_gating_very_hard_copula_derandomize_runs = int(
            max(1, int(getattr(self, "regime_gating_very_hard_copula_derandomize_runs", 5) or 5))
        )
        low_p_mode = str(
            getattr(self, "regime_gating_low_p_over_n_mode", "fast_univariate_filter")
            or "fast_univariate_filter"
        ).strip().lower()
        if low_p_mode not in {"all_features", "fast_univariate_filter"}:
            low_p_mode = "fast_univariate_filter"
        self.regime_gating_low_p_over_n_mode = str(low_p_mode)
        stability_threshold_method = str(
            getattr(self, "fs_stability_threshold_method", "fixed") or "fixed"
        ).strip().lower()
        if stability_threshold_method not in {"fixed", "eats", "cpss"}:
            stability_threshold_method = "fixed"
        self.fs_stability_threshold_method = str(stability_threshold_method)
        self.fs_stability_target_pfer = float(
            max(1e-6, float(getattr(self, "fs_stability_target_pfer", 1.0) or 1.0))
        )
        fs_cap_ratio = float(getattr(self, "fs_max_selected_features_ratio", 0.5) or 0.5)
        if not np.isfinite(fs_cap_ratio):
            fs_cap_ratio = 0.5
        self.fs_max_selected_features_ratio = float(np.clip(fs_cap_ratio, 1e-3, 10.0))
        self.fs_max_selected_features_cap = int(
            max(1, int(getattr(self, "fs_max_selected_features_cap", 500)))
        )

        if self.classification is None:
            self.classification = ClassificationConfig()
        elif not isinstance(self.classification, ClassificationConfig):
            self.classification = ClassificationConfig(**dict(self.classification))

        defaults = ClassificationConfig()
        c = self.classification

        # One-time legacy read migration into nested config.
        mappings = (
            ("classification_selection_mode", "selection_mode"),
            ("model_candidates", "model_candidates"),
            ("exclude_model_candidates", "exclude_model_candidates"),
            ("classifier_regime_candidate_exclusions", "regime_candidate_exclusions"),
            ("classifier_oracle_complexity_prior_overrides", "oracle_complexity_prior_overrides"),
            ("include_elastic_net_model", "include_elastic_net_model"),
            ("include_rf_model", "include_rf_model"),
            ("include_knn_model", "include_knn_model"),
            ("include_svm_linear_model", "include_svm_linear_model"),
            ("include_dlda_model", "include_dlda_model"),
            ("include_nsc_model", "include_nsc_model"),
            ("include_pls_da_model", "include_pls_da_model"),
            ("include_gpc_model", "include_gpc_model"),
            ("include_nb_model", "include_nb_model"),
            ("include_vote_ensemble_model", "include_vote_ensemble_model"),
            ("include_rp_ensemble_model", "include_rp_ensemble_model"),
            ("include_xgb_model", "include_xgb_model"),
            ("include_lgbm_model", "include_lgbm_model"),
            ("include_extra_tree_model", "include_extra_tree_model"),
            ("include_catboost_model", "include_catboost_model"),
            ("include_tabpfn_model", "include_tabpfn_model"),
            ("model_cv_lr_max_iter", "lr_max_iter"),
            ("model_cv_use_hybrid_score", "use_hybrid_score"),
            ("model_cv_balanced_weight", "hybrid_balanced_weight"),
            ("model_cv_macro_f1_weight", "hybrid_macro_f1_weight"),
            ("model_cv_runtime_containment_enabled", "runtime_containment_enabled"),
            ("model_cv_runtime_max_candidates", "runtime_max_candidates"),
            ("model_cv_runtime_high_p_over_n_threshold", "runtime_high_p_over_n_threshold"),
            ("model_cv_runtime_high_class_threshold", "runtime_high_class_threshold"),
            ("model_cv_runtime_min_class_count_threshold", "runtime_min_class_count_threshold"),
            ("classifier_oracle_k", "oracle_k"),
            ("classifier_oracle_weighting_mode", "oracle_weighting_mode"),
            ("classifier_oracle_include_robustness", "oracle_include_robustness"),
            ("classifier_oracle_include_complexity", "oracle_include_complexity"),
            ("classifier_oracle_include_calibration", "oracle_include_calibration"),
            ("classifier_oracle_include_james_stein", "oracle_include_james_stein"),
            ("classifier_oracle_include_cvar", "oracle_include_cvar"),
            ("classifier_oracle_cvar_alpha", "oracle_cvar_alpha"),
            ("classifier_oracle_use_dynamic_complexity", "oracle_use_dynamic_complexity"),
            ("classifier_oracle_portfolio_diversity", "oracle_portfolio_diversity"),
            ("classifier_oracle_portfolio_overlap_threshold", "oracle_portfolio_overlap_threshold"),
            ("classifier_oracle_portfolio_corr_threshold", "oracle_portfolio_corr_threshold"),
            ("classifier_oracle_enable_hoeffding_racing", "oracle_enable_hoeffding_racing"),
            ("classifier_oracle_hoeffding_delta", "oracle_hoeffding_delta"),
            ("classifier_oracle_enable_bbc", "oracle_enable_bbc"),
            ("classifier_oracle_bbc_bootstrap_rounds", "oracle_bbc_bootstrap_rounds"),
            ("classifier_oracle_bbc_ci_level", "oracle_bbc_ci_level"),
            ("classifier_oracle_enable_ensemble", "oracle_enable_ensemble"),
            ("classifier_oracle_ensemble_voting_mode", "oracle_ensemble_voting_mode"),
            ("classifier_oracle_greedy_ensemble", "oracle_greedy_ensemble"),
            ("classifier_oracle_greedy_ensemble_rounds", "oracle_greedy_ensemble_rounds"),
            ("classifier_oracle_candidate_pruning", "oracle_candidate_pruning"),
            ("classifier_oracle_candidate_pruning_threshold", "oracle_candidate_pruning_threshold"),
            ("classifier_oracle_incumbent_early_stopping", "oracle_incumbent_early_stopping"),
            ("classifier_oracle_behavior_profile", "oracle_behavior_profile"),
            ("classifier_oracle_use_per_family_flaml", "oracle_use_per_family_flaml"),
            ("stage2_ratio_augmentation_enabled", "stage2_ratio_augmentation_enabled"),
            ("stage2_ratio_max_features", "stage2_ratio_max_features"),
            ("stage2_ratio_selection_method", "stage2_ratio_selection_method"),
            ("stage2_ratio_epsilon", "stage2_ratio_epsilon"),
        )
        for old_name, new_name in mappings:
            legacy_val = getattr(self, old_name)
            if legacy_val != getattr(defaults, new_name):
                setattr(c, new_name, legacy_val)

        # Normalize nested config and mirror back for existing accessors.
        c.__post_init__()
        self.classification_selection_mode = str(c.selection_mode)
        self.model_candidates = tuple(c.model_candidates)
        self.exclude_model_candidates = tuple(c.exclude_model_candidates)
        self.classifier_regime_candidate_exclusions = tuple(c.regime_candidate_exclusions)
        self.classifier_oracle_complexity_prior_overrides = tuple(c.oracle_complexity_prior_overrides)
        self.include_elastic_net_model = bool(c.include_elastic_net_model)
        self.include_rf_model = bool(c.include_rf_model)
        self.include_knn_model = bool(c.include_knn_model)
        self.include_svm_linear_model = bool(c.include_svm_linear_model)
        self.include_dlda_model = bool(c.include_dlda_model)
        self.include_nsc_model = bool(c.include_nsc_model)
        self.include_pls_da_model = bool(c.include_pls_da_model)
        self.include_gpc_model = bool(c.include_gpc_model)
        self.include_nb_model = bool(c.include_nb_model)
        self.include_vote_ensemble_model = bool(c.include_vote_ensemble_model)
        self.include_rp_ensemble_model = bool(getattr(c, "include_rp_ensemble_model", False))
        self.include_xgb_model = bool(c.include_xgb_model)
        self.include_lgbm_model = bool(c.include_lgbm_model)
        self.include_extra_tree_model = bool(c.include_extra_tree_model)
        self.include_catboost_model = bool(c.include_catboost_model)
        self.include_tabpfn_model = bool(c.include_tabpfn_model)
        self.model_cv_lr_max_iter = int(c.lr_max_iter)
        self.model_cv_use_hybrid_score = bool(c.use_hybrid_score)
        self.model_cv_balanced_weight = float(c.hybrid_balanced_weight)
        self.model_cv_macro_f1_weight = float(c.hybrid_macro_f1_weight)
        self.model_cv_runtime_containment_enabled = bool(c.runtime_containment_enabled)
        self.model_cv_runtime_max_candidates = int(c.runtime_max_candidates)
        self.model_cv_runtime_high_p_over_n_threshold = float(c.runtime_high_p_over_n_threshold)
        self.model_cv_runtime_high_class_threshold = int(c.runtime_high_class_threshold)
        self.model_cv_runtime_min_class_count_threshold = int(c.runtime_min_class_count_threshold)
        self.classifier_oracle_k = int(c.oracle_k)
        self.classifier_oracle_weighting_mode = str(c.oracle_weighting_mode)
        self.classifier_oracle_include_robustness = bool(c.oracle_include_robustness)
        self.classifier_oracle_include_complexity = bool(c.oracle_include_complexity)
        self.classifier_oracle_include_calibration = bool(c.oracle_include_calibration)
        self.classifier_oracle_include_james_stein = bool(c.oracle_include_james_stein)
        self.classifier_oracle_include_cvar = bool(c.oracle_include_cvar)
        self.classifier_oracle_cvar_alpha = float(c.oracle_cvar_alpha)
        self.classifier_oracle_use_dynamic_complexity = bool(c.oracle_use_dynamic_complexity)
        self.classifier_oracle_portfolio_diversity = bool(c.oracle_portfolio_diversity)
        self.classifier_oracle_portfolio_overlap_threshold = float(c.oracle_portfolio_overlap_threshold)
        self.classifier_oracle_portfolio_corr_threshold = float(c.oracle_portfolio_corr_threshold)
        self.classifier_oracle_enable_hoeffding_racing = bool(c.oracle_enable_hoeffding_racing)
        self.classifier_oracle_hoeffding_delta = float(c.oracle_hoeffding_delta)
        self.classifier_oracle_enable_bbc = bool(c.oracle_enable_bbc)
        self.classifier_oracle_bbc_bootstrap_rounds = int(c.oracle_bbc_bootstrap_rounds)
        self.classifier_oracle_bbc_ci_level = float(c.oracle_bbc_ci_level)
        self.classifier_oracle_enable_ensemble = bool(c.oracle_enable_ensemble)
        self.classifier_oracle_ensemble_voting_mode = str(c.oracle_ensemble_voting_mode)
        self.classifier_oracle_greedy_ensemble = bool(c.oracle_greedy_ensemble)
        self.classifier_oracle_greedy_ensemble_rounds = int(c.oracle_greedy_ensemble_rounds)
        self.classifier_oracle_candidate_pruning = bool(c.oracle_candidate_pruning)
        self.classifier_oracle_candidate_pruning_threshold = float(c.oracle_candidate_pruning_threshold)
        self.classifier_oracle_incumbent_early_stopping = bool(c.oracle_incumbent_early_stopping)
        self.classifier_oracle_behavior_profile = str(c.oracle_behavior_profile)
        self.classifier_oracle_use_per_family_flaml = bool(c.oracle_use_per_family_flaml)
        self.stage2_ratio_augmentation_enabled = bool(c.stage2_ratio_augmentation_enabled)
        self.stage2_ratio_max_features = int(c.stage2_ratio_max_features)
        self.stage2_ratio_selection_method = str(c.stage2_ratio_selection_method)
        self.stage2_ratio_epsilon = float(c.stage2_ratio_epsilon)


@dataclass
class PipelineRunResult:
    dataset_name: str
    seed: int
    n_samples_total: int
    n_features_total: int
    n_train: int
    n_test: int
    n_fs_subset: int

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    hybrid_score: float
    roc_auc: float
    log_loss: float
    roc_curve_type: str
    roc_auc_source: str
    roc_curve_points: Tuple[Tuple[float, float], ...]
    roc_curves_by_method: Dict[str, Dict[str, Any]]

    selected_features_count: int
    selected_feature_indices_original: Tuple[int, ...]
    model_name: str

    fs_time_sec: float
    dist_time_sec: float
    transform_time_sec: float

    n_dist_features_fitted: int
    n_dist_features_transformed: int
    n_dist_rejected: int
    n_dist_skipped_unreliable: int
    n_dist_skipped_block_cv: int
    n_low_gof_downweighted: int
    mean_dist_stability_weight: float
    cdf_block_gating_time_sec: float
    cdf_block_gating_budget_hit: bool
    cdf_block_gating_blocks_evaluated: int
    cdf_block_gating_blocks_applied: int

    split_indices_train: Tuple[int, ...]
    split_indices_test: Tuple[int, ...]

    distribution_summaries: List[DistributionFitSummary] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    model_bundle: Dict[str, Any] = field(default_factory=dict)
    run_diagnostics: Dict[str, Any] = field(default_factory=dict)


def _json_safe(value: Any) -> Any:
    """Convert nested values into deterministic JSON-safe structures."""
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        if np.isfinite(val):
            return val
        return None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception as exc:
            pass
    return str(value)


def _dist_summary_to_dict(summary: DistributionFitSummary) -> Dict[str, Any]:
    return _json_safe({
        "feature_index": int(summary.feature_index),
        "family": None if summary.family is None else str(summary.family),
        "params": None if summary.params is None else [float(p) for p in summary.params],
        "cvm_p": float(summary.cvm_p) if np.isfinite(summary.cvm_p) else None,
        "ks_p": float(summary.ks_p) if np.isfinite(summary.ks_p) else None,
        "simple_score": float(summary.simple_score) if np.isfinite(summary.simple_score) else None,
        "confidence_set": [str(v) for v in tuple(summary.confidence_set or tuple())],
        "rejected": bool(summary.rejected),
        "audit": {
            "n_raw": int(summary.audit.n_raw),
            "n_clean": int(summary.audit.n_clean),
            "n_missing": int(summary.audit.n_missing),
            "n_unique": int(summary.audit.n_unique),
            "support": {
                "inferred_support": str(summary.audit.support.inferred_support),
                "frac_zero": float(summary.audit.support.frac_zero),
                "min_value": float(summary.audit.support.min_value),
                "max_value": float(summary.audit.support.max_value),
                "unique_ratio": float(summary.audit.support.unique_ratio),
                "is_near_constant": bool(summary.audit.support.is_near_constant),
            },
            "has_heaping": bool(summary.audit.has_heaping),
            "heaping_delta": (
                float(summary.audit.heaping_delta)
                if isinstance(summary.audit.heaping_delta, (int, float, np.floating))
                and np.isfinite(float(summary.audit.heaping_delta))
                else None
            ),
            "outlier_fraction": float(summary.audit.outlier_fraction),
            "frac_negative": float(summary.audit.frac_negative),
            "is_integer_like": bool(summary.audit.is_integer_like),
            "zero_inflated": bool(summary.audit.zero_inflated),
            "dip_stat": (
                float(summary.audit.dip_stat)
                if isinstance(summary.audit.dip_stat, (int, float, np.floating))
                and np.isfinite(float(summary.audit.dip_stat))
                else None
            ),
            "mode_count": (
                int(summary.audit.mode_count)
                if isinstance(summary.audit.mode_count, (int, np.integer))
                else None
            ),
            "is_multimodal": (
                bool(summary.audit.is_multimodal)
                if isinstance(summary.audit.is_multimodal, (bool, np.bool_))
                else None
            ),
            "too_few_unique": bool(summary.audit.too_few_unique),
        },
        "ad_stat": summary.ad_stat,
        "ad_p": summary.ad_p,
        "qq_r2": summary.qq_r2,
        "pp_r2": summary.pp_r2,
        "pp_mae": summary.pp_mae,
        "aic": summary.aic,
        "aicc": summary.aicc,
        "bic": summary.bic,
        "loglik": summary.loglik,
        "crps": summary.crps,
        "crps_uq_total": summary.crps_uq_total,
        "crps_uq_aleatoric": summary.crps_uq_aleatoric,
        "crps_uq_epistemic": summary.crps_uq_epistemic,
        "preq_loglik_mean": summary.preq_loglik_mean,
        "cv_loglik_mean": summary.cv_loglik_mean,
        "cv_loglik_std": summary.cv_loglik_std,
        "cv_score": summary.cv_score,
        "fit_method": summary.fit_method,
        "mnpo_weight": summary.mnpo_weight,
        "rejection_reason": (
            None if summary.rejection_reason is None else str(summary.rejection_reason)
        ),
        "selected_family_support": (
            None if summary.selected_family_support is None else str(summary.selected_family_support)
        ),
        "candidates_pre_filter": (
            None if summary.candidates_pre_filter is None else int(summary.candidates_pre_filter)
        ),
        "candidates_post_filter": (
            None if summary.candidates_post_filter is None else int(summary.candidates_post_filter)
        ),
    })


def _serialize_distribution_summary(summary: DistributionFitSummary) -> Dict[str, Any]:
    return _dist_summary_to_dict(summary)


def _encode_pickle_payload(obj: Any) -> str:
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    packed = zlib.compress(raw, level=6)
    return base64.b64encode(packed).decode("ascii")


def _decode_pickle_payload(payload: str) -> Any:
    text = str(payload or "").strip()
    if not text:
        return None
    packed = base64.b64decode(text.encode("ascii"))
    raw = zlib.decompress(packed)
    return pickle.loads(raw)


class DFFSReproducibleModel:
    """Inference helper that can be reconstructed from a JSON model bundle."""

    schema_version = "1.0"

    def __init__(
        self,
        *,
        n_input_features: int,
        imputer: Any,
        batch_model: Optional[Dict[str, Any]],
        face_meta: Dict[str, Any],
        face_pca: Optional[Any],
        face_lda: Optional[Any],
        ratio_meta: Dict[str, Any],
        scaler_base: Any,
        distribution_plan: Dict[str, Any],
        prefilter_indices: Sequence[int],
        folding_meta: Dict[str, Any],
        folding_transformer: Optional[Any],
        folding_standardize_mean: Optional[np.ndarray],
        folding_standardize_scale: Optional[np.ndarray],
        selector: Any,
        stage2_ratio_meta: Dict[str, Any],
        classifier_model: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.n_input_features = int(max(0, n_input_features))
        self.imputer = imputer
        self.batch_model = dict(batch_model or {}) if isinstance(batch_model, dict) else None
        self.face_meta = dict(face_meta or {})
        self.face_pca = face_pca
        self.face_lda = face_lda
        self.ratio_meta = dict(ratio_meta or {})
        self.scaler_base = scaler_base
        self.distribution_plan = dict(distribution_plan or {})
        self.prefilter_indices = np.asarray(list(prefilter_indices), dtype=int).ravel()
        self.folding_meta = dict(folding_meta or {})
        self.folding_transformer = folding_transformer
        self.folding_standardize_mean = (
            None if folding_standardize_mean is None else np.asarray(folding_standardize_mean, dtype=float).ravel()
        )
        self.folding_standardize_scale = (
            None if folding_standardize_scale is None else np.asarray(folding_standardize_scale, dtype=float).ravel()
        )
        self.selector = selector
        self.stage2_ratio_meta = dict(stage2_ratio_meta or {})
        self.classifier_model = classifier_model
        self.metadata = dict(metadata or {})
        # Distribution objects are looked up by family for deterministic CDF transforms.
        self._dist_fitter = DistributionFitter(DistributionFitterConfig())

    @staticmethod
    def _ensure_2d_numeric(X: np.ndarray, *, expected_features: int) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D X, got shape {arr.shape}")
        if int(arr.shape[1]) != int(expected_features):
            raise ValueError(
                f"Expected {int(expected_features)} input features, got {int(arr.shape[1])}."
            )
        return arr

    def _apply_stage1_ratio_features(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if not bool(self.ratio_meta.get("ratio_features_applied", False)):
            return x
        pairs = list(self.ratio_meta.get("ratio_pairs", []) or [])
        if not pairs:
            return x
        eps = float(self.ratio_meta.get("ratio_epsilon", 1e-6) or 1e-6)
        eps = float(max(1e-12, eps))
        abs_value = bool(self.ratio_meta.get("ratio_abs_value", False))
        include_originals = bool(self.ratio_meta.get("ratio_include_originals", True))

        cols: List[np.ndarray] = []
        for entry in pairs:
            a = int(entry.get("numerator", -1))
            b = int(entry.get("denominator", -1))
            if a < 0 or b < 0 or a >= x.shape[1] or b >= x.shape[1]:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.log((x[:, a] + eps) / (x[:, b] + eps))
            ratio = np.asarray(np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
            if abs_value:
                ratio = np.abs(ratio)
            cols.append(ratio.reshape(-1, 1))
        if not cols:
            return x
        x_ratio = np.hstack(cols)
        if include_originals:
            return np.hstack([x, x_ratio])
        return x_ratio

    def _apply_distribution_transforms(self, X_model_input: np.ndarray, X_base: np.ndarray) -> np.ndarray:
        x_raw = np.asarray(X_model_input, dtype=float)
        x_out = np.asarray(X_base, dtype=float).copy()
        plan = dict(self.distribution_plan or {})
        features = list(plan.get("feature_plans", []) or [])
        for entry in features:
            if not bool(entry.get("applied", False)):
                continue
            feat_idx = int(entry.get("feature_index", -1))
            if feat_idx < 0 or feat_idx >= x_out.shape[1] or feat_idx >= x_raw.shape[1]:
                continue
            family = str(entry.get("family", "") or "").strip()

            # ── GMM / rank_transform multimodal fallback path ──
            fallback_meta = dict(entry.get("fallback_meta") or {})
            fallback_mode = str(fallback_meta.get("fallback_mode", "") or "").strip().lower()
            if family.startswith("multimodal_fallback_") and fallback_mode:
                mu = float(entry.get("train_mean", 0.0) or 0.0)
                sigma = float(entry.get("train_std", 1.0) or 1.0)
                if not np.isfinite(sigma) or sigma <= 1e-12:
                    sigma = 1.0
                weight = float(entry.get("weight", 1.0) or 1.0)
                data = np.asarray(x_raw[:, feat_idx], dtype=float)

                if fallback_mode == "gmm":
                    components = list(fallback_meta.get("components") or [])
                    if not components:
                        continue
                    means = np.array([float(c["mean"]) for c in components], dtype=float)
                    stds = np.array([float(c["std"]) for c in components], dtype=float)
                    weights_gmm = np.array([float(c["weight"]) for c in components], dtype=float)
                    # Reuse the static mixture CDF → norm.ppf transform
                    cdf_vals = DistributionFeatureSelectionPipeline._mixture_cdf_1d(
                        data, means=means, stds=stds, weights=weights_gmm,
                    )
                    gauss = sps.norm.ppf(cdf_vals)
                elif fallback_mode == "rank_transform":
                    # Rank-based Gaussian transform (monotone, no stored params needed)
                    ranks = np.argsort(np.argsort(data)).astype(float)
                    n = float(data.shape[0])
                    u = (ranks + 0.5) / n
                    gauss = sps.norm.ppf(np.clip(u, 1e-8, 1.0 - 1e-8))
                else:
                    continue

                gauss = np.nan_to_num(np.asarray(gauss, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                z = ((gauss - mu) / sigma) * weight
                x_out[:, feat_idx] = np.asarray(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
                continue

            # ── Standard parametric CDF path ──
            params = list(entry.get("params", []) or [])
            if not family or not params:
                continue
            dist_obj = self._dist_fitter._base_distributions.get(family)
            if dist_obj is None:
                continue
            mu = float(entry.get("train_mean", 0.0) or 0.0)
            sigma = float(entry.get("train_std", 1.0) or 1.0)
            if not np.isfinite(sigma) or sigma <= 1e-12:
                sigma = 1.0
            weight = float(entry.get("weight", 1.0) or 1.0)
            data = np.asarray(x_raw[:, feat_idx], dtype=float)
            cdf_vals = dist_obj.cdf(data, *params)
            cdf_vals = np.clip(cdf_vals, 1e-8, 1.0 - 1e-8)
            gauss = sps.norm.ppf(cdf_vals)
            z = ((gauss - mu) / sigma) * weight
            x_out[:, feat_idx] = np.asarray(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
        return x_out

    def _apply_folding(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if not bool(self.folding_meta.get("folding_applied", False)):
            return x
        if self.folding_transformer is None:
            raise RuntimeError("Folding was applied during training, but transformer state is missing in bundle.")
        x_fold = np.asarray(self.folding_transformer.transform(x), dtype=float)
        if x_fold.ndim == 1:
            x_fold = x_fold.reshape(-1, 1)
        if (
            self.folding_standardize_mean is not None
            and self.folding_standardize_scale is not None
            and self.folding_standardize_mean.size == x_fold.shape[1]
            and self.folding_standardize_scale.size == x_fold.shape[1]
        ):
            x_fold = (x_fold - self.folding_standardize_mean) / self.folding_standardize_scale
        return np.asarray(np.nan_to_num(x_fold, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)

    def _apply_stage2_ratio_features(self, X_selected: np.ndarray) -> np.ndarray:
        x = np.asarray(X_selected, dtype=float)
        if not bool(self.stage2_ratio_meta.get("stage2_ratio_features_applied", False)):
            return x
        pairs = list(self.stage2_ratio_meta.get("stage2_ratio_pairs", []) or [])
        if not pairs:
            return x
        eps = float(self.stage2_ratio_meta.get("stage2_ratio_epsilon", 1e-6) or 1e-6)
        eps = float(max(1e-12, eps))
        cols: List[np.ndarray] = []
        for entry in pairs:
            a = int(entry.get("numerator", -1))
            b = int(entry.get("denominator", -1))
            if a < 0 or b < 0 or a >= x.shape[1] or b >= x.shape[1]:
                continue
            num_shift = float(entry.get("numerator_shift", 0.0) or 0.0)
            den_shift = float(entry.get("denominator_shift", 0.0) or 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.log((x[:, a] + num_shift + eps) / (x[:, b] + den_shift + eps))
            ratio = np.asarray(np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
            cols.append(ratio.reshape(-1, 1))
        if not cols:
            return x
        return np.hstack([x, np.hstack(cols)])

    def transform(self, X: np.ndarray, *, batch_labels: Optional[Sequence[Any]] = None) -> np.ndarray:
        x = self._ensure_2d_numeric(X, expected_features=self.n_input_features)
        x_imp = np.asarray(self.imputer.transform(x), dtype=float)

        mode = str((self.batch_model or {}).get("mode", "none") or "none").strip().lower()
        if mode != "none":
            if batch_labels is None:
                raise ValueError(
                    "This model bundle uses batch correction; provide `batch_labels` to transform/predict."
                )
            batch_arr = np.asarray(list(batch_labels), dtype=object).ravel()
            if int(batch_arr.size) != int(x_imp.shape[0]):
                raise ValueError(
                    f"batch_labels has {batch_arr.size} rows but X has {x_imp.shape[0]} rows."
                )
            x_imp, _, _ = apply_batch_correction_model(
                x_imp,
                x_imp,
                model=self.batch_model,
                batch_labels_train=batch_arr,
                batch_labels_test=batch_arr,
            )

        x_model_input = x_imp
        if bool(self.face_meta.get("face_projection_applied", False)):
            if self.face_pca is None:
                raise RuntimeError("Face-domain projection was applied during training, but PCA state is missing.")
            x_model_input = np.asarray(self.face_pca.transform(x_model_input), dtype=float)
            mode = str(self.face_meta.get("face_projection_mode", "pca_only") or "pca_only").strip().lower()
            if mode == "pca_lda" and self.face_lda is not None:
                x_model_input = np.asarray(self.face_lda.transform(x_model_input), dtype=float)
            if x_model_input.ndim == 1:
                x_model_input = x_model_input.reshape(-1, 1)

        x_model_input = self._apply_stage1_ratio_features(x_model_input)
        x_base = np.asarray(self.scaler_base.transform(x_model_input), dtype=float)
        dist_plan = dict(self.distribution_plan or {})
        stage_position = str(dist_plan.get("df_stage_position", "before_fs") or "before_fs").strip().lower()
        pre_idx = np.asarray(self.prefilter_indices, dtype=int).ravel()
        if stage_position == "after_fs":
            if pre_idx.size <= 0:
                pre_idx = np.arange(x_base.shape[1], dtype=int)
            x_pref_raw = np.asarray(x_model_input[:, pre_idx], dtype=float)
            x_pref_base = np.asarray(x_base[:, pre_idx], dtype=float)
            source_space = str(dist_plan.get("df_stage_source_space", "prefilter_raw") or "prefilter_raw").strip().lower()
            if source_space == "folded_selector_input":
                # Backward compatibility: older bundles stored the latent-space
                # DF order and therefore require folding before selector replay.
                x_fold = self._apply_folding(x_pref_base)
                x_sel = np.asarray(self.selector.transform(x_fold), dtype=float)
                x_sel_raw = np.asarray(x_sel, dtype=float)
                x_sel_base = np.asarray(x_sel, dtype=float)
                x_sel = self._apply_distribution_transforms(x_sel_raw, x_sel_base)
            else:
                x_sel = np.asarray(self.selector.transform(x_pref_base), dtype=float)
                selected_idx = None
                if hasattr(self.selector, "get_selected_features_indices"):
                    try:
                        selected_idx = np.asarray(self.selector.get_selected_features_indices(), dtype=int).ravel()
                    except Exception:
                        selected_idx = None
                if selected_idx is None or selected_idx.size != x_sel.shape[1]:
                    x_sel_raw = np.asarray(x_sel, dtype=float)
                    x_sel_base = np.asarray(x_sel, dtype=float)
                else:
                    x_sel_raw = np.asarray(x_pref_raw[:, selected_idx], dtype=float)
                    x_sel_base = np.asarray(x_pref_base[:, selected_idx], dtype=float)
                x_sel = self._apply_distribution_transforms(x_sel_raw, x_sel_base)
                x_sel = self._apply_folding(x_sel)
        else:
            x_trans = self._apply_distribution_transforms(x_model_input, x_base)
            if pre_idx.size <= 0:
                pre_idx = np.arange(x_trans.shape[1], dtype=int)
            x_pref = np.asarray(x_trans[:, pre_idx], dtype=float)
            x_fold = self._apply_folding(x_pref)
            x_sel = np.asarray(self.selector.transform(x_fold), dtype=float)
        x_stage2 = self._apply_stage2_ratio_features(x_sel)
        return np.asarray(np.nan_to_num(x_stage2, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)

    def predict(self, X: np.ndarray, *, batch_labels: Optional[Sequence[Any]] = None) -> np.ndarray:
        x_t = self.transform(X, batch_labels=batch_labels)
        return np.asarray(self.classifier_model.predict(x_t))

    def predict_proba(self, X: np.ndarray, *, batch_labels: Optional[Sequence[Any]] = None) -> np.ndarray:
        x_t = self.transform(X, batch_labels=batch_labels)
        if not hasattr(self.classifier_model, "predict_proba"):
            raise AttributeError("Model does not expose predict_proba.")
        return np.asarray(self.classifier_model.predict_proba(x_t), dtype=float)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": "df_fs_model_bundle",
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "metadata": _json_safe(dict(self.metadata or {})),
            "n_input_features": int(self.n_input_features),
            "inference_contract": {
                "same_column_order_required": True,
                "column_identifier_mode": "index",
                "batch_labels_required": bool(
                    str((self.batch_model or {}).get("mode", "none") or "none").strip().lower() != "none"
                ),
            },
            "stages": {
                "face_projection": _json_safe(self.face_meta),
                "ratio_stage1": _json_safe(self.ratio_meta),
                "distribution_stage": _json_safe(self.distribution_plan),
                "folding_stage": _json_safe(self.folding_meta),
                "stage2_ratio": _json_safe(self.stage2_ratio_meta),
                "prefilter_indices": [int(i) for i in np.asarray(self.prefilter_indices, dtype=int).ravel().tolist()],
            },
            "serialization": {
                "format": "pickle+zlib+base64",
                "imputer": _encode_pickle_payload(self.imputer),
                "batch_model": _encode_pickle_payload(self.batch_model),
                "face_pca": _encode_pickle_payload(self.face_pca),
                "face_lda": _encode_pickle_payload(self.face_lda),
                "scaler_base": _encode_pickle_payload(self.scaler_base),
                "folding_transformer": _encode_pickle_payload(self.folding_transformer),
                "folding_standardize_mean": _encode_pickle_payload(self.folding_standardize_mean),
                "folding_standardize_scale": _encode_pickle_payload(self.folding_standardize_scale),
                "selector": _encode_pickle_payload(self.selector),
                "classifier_model": _encode_pickle_payload(self.classifier_model),
            },
        }

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "DFFSReproducibleModel":
        ser = dict(payload.get("serialization") or {})
        meta = dict(payload.get("metadata") or {})
        stages = dict(payload.get("stages") or {})
        return cls(
            n_input_features=int(payload.get("n_input_features", 0) or 0),
            imputer=_decode_pickle_payload(str(ser.get("imputer", ""))),
            batch_model=_decode_pickle_payload(str(ser.get("batch_model", ""))),
            face_meta=dict(stages.get("face_projection") or {}),
            face_pca=_decode_pickle_payload(str(ser.get("face_pca", ""))),
            face_lda=_decode_pickle_payload(str(ser.get("face_lda", ""))),
            ratio_meta=dict(stages.get("ratio_stage1") or {}),
            scaler_base=_decode_pickle_payload(str(ser.get("scaler_base", ""))),
            distribution_plan=dict(stages.get("distribution_stage") or {}),
            prefilter_indices=list(stages.get("prefilter_indices") or []),
            folding_meta=dict(stages.get("folding_stage") or {}),
            folding_transformer=_decode_pickle_payload(str(ser.get("folding_transformer", ""))),
            folding_standardize_mean=_decode_pickle_payload(str(ser.get("folding_standardize_mean", ""))),
            folding_standardize_scale=_decode_pickle_payload(str(ser.get("folding_standardize_scale", ""))),
            selector=_decode_pickle_payload(str(ser.get("selector", ""))),
            stage2_ratio_meta=dict(stages.get("stage2_ratio") or {}),
            classifier_model=_decode_pickle_payload(str(ser.get("classifier_model", ""))),
            metadata=meta,
        )


def load_df_fs_model_bundle(path: str) -> DFFSReproducibleModel:
    import json

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return DFFSReproducibleModel.from_json_dict(payload)


def _gaussian_expected_abs(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """E|Z| for Z ~ Normal(mu, sigma^2) (vectorized)."""
    mu_arr = np.asarray(mu, dtype=float)
    sig_arr = np.asarray(sigma, dtype=float)
    out = np.full_like(mu_arr, np.nan, dtype=float)
    finite = np.isfinite(mu_arr) & np.isfinite(sig_arr) & (sig_arr >= 0.0)
    if not np.any(finite):
        return out

    mu2 = mu_arr[finite]
    sig2 = sig_arr[finite]

    # Degenerate normal (sigma=0): |mu|.
    out2 = np.abs(mu2)
    mask = sig2 > 0.0
    if np.any(mask):
        m = mu2[mask]
        s = sig2[mask]
        a = m / s
        pdf = sps.norm.pdf(a)
        cdf = sps.norm.cdf(a)
        out2[mask] = 2.0 * s * pdf + m * (2.0 * cdf - 1.0)

    out[finite] = out2
    return out


def _crps_uq_decompose_gaussian_ensemble(
    means: Sequence[float],
    stds: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    """CRPS-based uncertainty decomposition for a Gaussian ensemble surrogate.

    Uses the CRPS entropy identity:
      H_CRPS(P) = E_{Y~P}[CRPS(P, Y)] = 0.5 * E|X - X'|, X,X'~P i.i.d.

    For a Gaussian mixture surrogate with weights w_i and components N(mu_i, std_i^2),
    the entropy is:
      H_mix = 0.5 * sum_{i,j} w_i w_j E|N(mu_i-mu_j, std_i^2+std_j^2)|.

    Aleatoric is the mixture-of-entropies:
      H_alea = sum_i w_i * (std_i / sqrt(pi)).

    Epistemic is the residual:
      H_epi = H_mix - H_alea.

    Reference: Fishkov et al. (2025), arXiv:2509.26610.
    """
    mu = np.asarray(list(means), dtype=float).ravel()
    sig = np.asarray(list(stds), dtype=float).ravel()
    if mu.size == 0 or sig.size == 0 or mu.size != sig.size:
        raise ValueError("means/stds must be non-empty arrays of the same length")

    keep = np.isfinite(mu) & np.isfinite(sig) & (sig >= 0.0)
    if not np.any(keep):
        raise ValueError("no finite Gaussian surrogate members")
    mu = mu[keep]
    sig = sig[keep]

    m = int(mu.size)
    if weights is None:
        w = np.full(m, 1.0 / float(m), dtype=float)
    else:
        w = np.asarray(list(weights), dtype=float).ravel()
        if w.size != int(keep.size):
            raise ValueError("weights length must match means/stds length")
        w = w[keep]
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite")
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            w = np.full(m, 1.0 / float(m), dtype=float)
        else:
            w = w / w_sum

    # Aleatoric: E[H(N(mu_i, sig_i^2))] = sum w_i * sig_i / sqrt(pi).
    alea = float(np.sum(w * (sig / math.sqrt(math.pi))))

    # Total entropy of the mixture surrogate via pairwise E|Normal(mu_ij, sig_ij^2)|.
    mu_ij = mu[:, None] - mu[None, :]
    sig_ij = np.sqrt(np.maximum(0.0, sig[:, None] ** 2 + sig[None, :] ** 2))
    e_abs = _gaussian_expected_abs(mu_ij, sig_ij)
    e_abs = np.nan_to_num(e_abs, nan=0.0, posinf=0.0, neginf=0.0)
    total = 0.5 * float(np.sum((w[:, None] * w[None, :]) * e_abs))

    epi = float(max(0.0, total - alea))
    return float(total), float(alea), float(epi)


class DistributionFitter:
    """Wrapper around UnifiedDistributionSelectorV6 with auditing and diagnostics."""

    def __init__(self, config: Optional[DistributionFitterConfig] = None):
        self.config = config or DistributionFitterConfig()
        family_set = str(getattr(self.config, "family_set", "v6") or "v6").strip().lower()
        if family_set in {"flex", "flexible", "v6_flex"}:
            self._base_distributions = UnifiedDistributionSelectorV6._get_flex_distributions()
        elif family_set in {"extended", "v6_extended", "ext"}:
            self._base_distributions = UnifiedDistributionSelectorV6._get_extended_distributions()
        else:
            self._base_distributions = UnifiedDistributionSelectorV6._get_default_distributions()

        # Budget-dependent defaults.
        if self.config.compute_budget == "fast":
            self.config.use_cv = False
            self.config.use_lrt = False
        elif self.config.compute_budget == "thorough":
            self.config.use_cv = True
            self.config.use_lrt = True

    def audit_data(self, data: np.ndarray) -> DataAuditReport:
        arr = np.asarray(data, dtype=float).ravel()
        finite_mask = np.isfinite(arr)
        clean = arr[finite_mask]

        if clean.size == 0:
            support = SupportProfile(
                inferred_support="unknown",
                frac_zero=0.0,
                min_value=float("nan"),
                max_value=float("nan"),
                unique_ratio=0.0,
                is_near_constant=True,
            )
            return DataAuditReport(
                n_raw=int(arr.size),
                n_clean=0,
                n_missing=int(arr.size),
                n_unique=0,
                support=support,
                has_heaping=False,
                heaping_delta=None,
                outlier_fraction=0.0,
            )

        min_val = float(np.min(clean))
        max_val = float(np.max(clean))
        frac_zero = float(np.mean(np.isclose(clean, 0.0, atol=1e-12)))
        frac_negative = float(np.mean(clean < -1e-12))
        n_unique = int(np.unique(clean).size)
        unique_ratio = float(n_unique / max(1, clean.size))
        is_near_constant = bool(np.std(clean) < 1e-10 or unique_ratio < 0.01)
        too_few_unique = bool(n_unique <= 5 and int(clean.size) >= 50)
        is_integer_like = self._is_integer_like(clean, n_unique=n_unique)
        zero_inflated = bool(frac_zero >= 0.10 and min_val >= -1e-10 and max_val > 0.0 and int(clean.size) >= 30)

        inferred_support = "real"
        if min_val >= -1e-10:
            inferred_support = "positive"
        if min_val >= -1e-10 and max_val <= 1.0 + 1e-10:
            inferred_support = "unit_interval"

        has_heaping, heaping_delta = self._detect_heaping(clean)
        outlier_fraction = self._estimate_outlier_fraction(clean)
        dip_stat = None
        mode_count = None
        is_multimodal = None
        if bool(getattr(self.config, "compute_dip", False)):
            dip_stat = self._compute_simple_dip_stat(clean)
            bins = int(getattr(self.config, "dip_hist_bins", 40) or 40)
            mode_count = self._estimate_mode_count(clean, bins=bins)
            is_multimodal = bool(int(mode_count or 0) >= 2)

        support = SupportProfile(
            inferred_support=inferred_support,
            frac_zero=frac_zero,
            min_value=min_val,
            max_value=max_val,
            unique_ratio=unique_ratio,
            is_near_constant=is_near_constant,
        )

        return DataAuditReport(
            n_raw=int(arr.size),
            n_clean=int(clean.size),
            n_missing=int(arr.size - clean.size),
            n_unique=n_unique,
            support=support,
            has_heaping=has_heaping,
            heaping_delta=heaping_delta,
            outlier_fraction=float(outlier_fraction),
            frac_negative=float(frac_negative),
            is_integer_like=bool(is_integer_like),
            zero_inflated=bool(zero_inflated),
            dip_stat=None if dip_stat is None else float(dip_stat),
            mode_count=None if mode_count is None else int(mode_count),
            is_multimodal=None if is_multimodal is None else bool(is_multimodal),
            too_few_unique=bool(too_few_unique),
        )

    @staticmethod
    def _detect_heaping(clean: np.ndarray) -> Tuple[bool, Optional[float]]:
        if clean.size < 30:
            return False, None

        best_delta = None
        best_score = 0.0
        # Heuristic deltas; extend as needed.
        for delta in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
            snapped = np.isclose(clean / delta, np.round(clean / delta), atol=0.02)
            score = float(np.mean(snapped))
            if score > best_score:
                best_score = score
                best_delta = float(delta)

        # Signal strong heaping when a large fraction lands on grid points.
        return bool(best_score >= 0.40), best_delta

    @staticmethod
    def _estimate_outlier_fraction(clean: np.ndarray) -> float:
        if clean.size < 10:
            return 0.0
        q1, q3 = np.percentile(clean, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-12:
            return 0.0
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        frac = float(np.mean((clean < lower) | (clean > upper)))
        return frac

    @staticmethod
    def _is_integer_like(clean: np.ndarray, *, n_unique: Optional[int] = None) -> bool:
        """Heuristic: detect discrete/integer-like values masquerading as continuous."""
        if clean.size < 10:
            return False
        if n_unique is None:
            n_unique = int(np.unique(clean).size)
        if int(n_unique) <= 2:
            return True
        # Only consider it integer-like when most values are close to an integer.
        atol = 1e-8
        frac_int = float(np.mean(np.isclose(clean, np.round(clean), atol=atol)))
        return bool(frac_int >= 0.98 and int(n_unique) <= int(max(12, 0.10 * clean.size)))

    @staticmethod
    def _compute_simple_dip_stat(clean: np.ndarray) -> float:
        """Cheap dip-like statistic (not Hartigan's exact dip): max |F_n - U| on [min, max]."""
        if clean.size < 10:
            return 0.0
        x = np.sort(np.asarray(clean, dtype=float).ravel())
        if x.size < 2:
            return 0.0
        x0 = float(x[0])
        x1 = float(x[-1])
        if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
            return 0.0
        n = int(x.size)
        empirical = (np.arange(1, n + 1, dtype=float) / float(n))
        uniform = (x - x0) / (x1 - x0)
        stat = float(np.max(np.abs(empirical - uniform)))
        if not np.isfinite(stat):
            return 0.0
        return float(np.clip(stat, 0.0, 1.0))

    @staticmethod
    def _estimate_mode_count(clean: np.ndarray, *, bins: int = 40) -> int:
        """Count histogram modes using light smoothing; intended as a cheap mixture flag."""
        if clean.size < 30:
            return 1
        x = np.asarray(clean, dtype=float).ravel()
        if x.size < 2 or not np.isfinite(np.min(x)) or not np.isfinite(np.max(x)):
            return 1
        if float(np.max(x)) <= float(np.min(x)):
            return 1

        nbins = int(max(10, min(120, bins)))
        hist, _ = np.histogram(x, bins=nbins)
        if hist.size < 3 or int(np.sum(hist)) <= 0:
            return 1

        # Smooth to reduce spurious local maxima on small samples.
        kernel = np.array([1.0, 2.0, 1.0], dtype=float)
        smooth = np.convolve(hist.astype(float), kernel / float(np.sum(kernel)), mode="same")
        peak_thr = 0.05 * float(np.max(smooth))
        if not np.isfinite(peak_thr) or peak_thr <= 0.0:
            return 1

        peaks = 0
        for i in range(1, smooth.size - 1):
            if smooth[i] >= peak_thr and smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
                peaks += 1
        return int(max(1, peaks))

    def generate_candidates(self, audit: DataAuditReport) -> Dict[str, sps.rv_continuous]:
        if not self.config.use_support_filtering:
            return dict(self._base_distributions)

        support = audit.support.inferred_support
        families = set(self._base_distributions.keys())

        if support == "unit_interval":
            allowed = {"beta", "uniform", "triang", "powerlaw", "johnsonsb", "norm", "t", "laplace"}
            families = families.intersection(allowed)
        elif support == "positive":
            banned = {"johnsonsb"}
            families = families.difference(banned)
        else:
            # Real support: keep everything except hard unit-interval-only families when ranges are broad.
            if audit.support.max_value - audit.support.min_value > 2.5:
                families = families.difference(set(_UNIT_INTERVAL_ONLY_FAMILIES))
            # Real-valued features with meaningful negative mass should not evaluate
            # positive-only families (e.g., proteomics M-values).
            if float(getattr(audit, "frac_negative", 0.0) or 0.0) >= 0.05:
                families = families.difference(set(_POSITIVE_ONLY_FAMILIES))

        # Heaped data: keep families with smoother CDF shape flexibility.
        if audit.has_heaping:
            families.update({"norm", "laplace", "t", "johnsonsu"})

        if not families:
            families = set(self._base_distributions.keys())

        return {name: self._base_distributions[name] for name in sorted(families)}

    @staticmethod
    def _family_support_class(family: Optional[str]) -> Optional[str]:
        if family is None:
            return None
        name = str(family)
        if name in _POSITIVE_ONLY_FAMILIES:
            return "positive"
        if name in _UNIT_INTERVAL_ONLY_FAMILIES:
            return "unit_interval"
        return "real"

    @staticmethod
    def _is_support_conflict(audit: DataAuditReport, family_support: Optional[str]) -> bool:
        if family_support is None:
            return False
        if str(getattr(audit.support, "inferred_support", "")) != "real":
            return False
        if str(family_support) != "positive":
            return False
        return bool(float(getattr(audit, "frac_negative", 0.0) or 0.0) >= 0.05)

    def _build_selector(
        self,
        distributions: Dict[str, sps.rv_continuous],
        *,
        random_state: Optional[int] = None,
        compute_crps: Optional[bool] = None,
        interval_likelihood: bool = False,
        interval_delta: float = 0.0,
    ) -> UnifiedDistributionSelectorV6:
        compute_crps_effective = bool(getattr(self.config, "compute_crps", False)) if compute_crps is None else bool(compute_crps)
        return UnifiedDistributionSelectorV6(
            distributions=distributions,
            robust_mode=self.config.robust_mode,
            use_adaptive_strategy=self.config.use_adaptive_strategy,
            use_lrt=self.config.use_lrt,
            use_cv=self.config.use_cv,
            n_jobs=int(getattr(self.config, "n_jobs", 1) or 1),
            use_lmoment_prescreen=bool(getattr(self.config, "use_lmoment_prescreen", False)),
            lmoment_prescreen_max_candidates=int(getattr(self.config, "lmoment_prescreen_max_candidates", 0) or 0),
            fit_estimator=str(getattr(self.config, "estimator", "mle") or "mle"),
            mps_maxiter=int(getattr(self.config, "mps_maxiter", 250) or 250),
            mps_tol=float(getattr(self.config, "mps_tol", 1e-6) or 1e-6),
            compute_ad=bool(getattr(self.config, "compute_ad", False)),
            ad_bootstrap_samples=int(getattr(self.config, "ad_bootstrap_samples", 0) or 0),
            compute_qq_pp=bool(getattr(self.config, "compute_qq_pp", False)),
            interval_likelihood=bool(interval_likelihood),
            interval_delta=float(max(0.0, interval_delta)),
            compute_crps=compute_crps_effective,
            crps_mc_samples=int(getattr(self.config, "crps_mc_samples", 96) or 96),
            crps_data_subsample=int(getattr(self.config, "crps_data_subsample", 256) or 256),
            random_state=random_state,
            mnpo_use_tritrust=bool(getattr(self.config, "mnpo_use_tritrust", True)),
            mnpo_include_crps=bool(getattr(self.config, "mnpo_include_crps", False)),
            mnpo_include_preq=bool(getattr(self.config, "mnpo_include_preq", False)),
            mnpo_use_qre_smoothing=bool(getattr(self.config, "mnpo_use_qre_smoothing", False)),
            mnpo_qre_temperature_gamma=float(getattr(self.config, "mnpo_qre_temperature_gamma", 1.0) or 1.0),
            mnpo_use_oracle_redundancy_penalty=bool(
                getattr(self.config, "mnpo_use_oracle_redundancy_penalty", False)
            ),
            mnpo_compute_tremble_sensitivity=bool(
                getattr(self.config, "mnpo_compute_tremble_sensitivity", False)
            ),
            preq_holdout_fraction=float(getattr(self.config, "preq_holdout_fraction", 0.20) or 0.20),
            preq_min_train=int(getattr(self.config, "preq_min_train", 20) or 20),
            preq_max_test_points=int(getattr(self.config, "preq_max_test_points", 128) or 128),
        )

    def select_best_distribution(
        self,
        data: np.ndarray,
        criterion: str = "simple",
        feature_index: int = -1,
        audit: Optional[DataAuditReport] = None,
    ) -> DistributionFitSummary:
        audit = self.audit_data(data) if audit is None else audit
        candidates_pre_filter = int(len(self._base_distributions))

        if audit.n_clean < 10 or audit.support.is_near_constant:
            if audit.n_clean < 10:
                reason = "insufficient_clean"
            elif bool(audit.support.is_near_constant):
                reason = "near_constant"
            else:
                reason = "fit_failed"
            return DistributionFitSummary(
                feature_index=int(feature_index),
                family=None,
                params=None,
                cvm_p=float("nan"),
                ks_p=float("nan"),
                simple_score=float("nan"),
                confidence_set=tuple(),
                rejected=True,
                audit=audit,
                rejection_reason=reason,
                selected_family_support=None,
                candidates_pre_filter=candidates_pre_filter,
                candidates_post_filter=None,
            )

        candidates = self.generate_candidates(audit)
        candidates_post_filter = int(len(candidates))
        selector_seed = None
        if getattr(self.config, "random_state", None) is not None:
            base = int(getattr(self.config, "random_state"))
            # Derive a stable per-feature seed for bootstrap diagnostics.
            stride = 997
            offset = int(max(0, int(feature_index))) if int(feature_index) >= 0 else 0
            selector_seed = base + stride * offset
        crit = str(criterion).strip().lower()
        want_crps = bool(getattr(self.config, "compute_crps", False)) or crit == "crps" or (
            crit == "mnpo_oracle" and bool(getattr(self.config, "mnpo_include_crps", False))
        )
        interval_enabled = bool(getattr(self.config, "interval_likelihood", False))
        interval_delta = float(getattr(self.config, "interval_delta_override", 0.0) or 0.0)
        if interval_enabled and interval_delta <= 0.0:
            if bool(getattr(audit, "has_heaping", False)) and getattr(audit, "heaping_delta", None) is not None:
                try:
                    interval_delta = float(getattr(audit, "heaping_delta"))
                except Exception as exc:
                    interval_delta = 0.0
            elif bool(getattr(audit, "is_integer_like", False)):
                interval_delta = 1.0
        interval_enabled = bool(interval_enabled and interval_delta > 0.0)

        selector = self._build_selector(
            candidates,
            random_state=selector_seed,
            compute_crps=want_crps,
            interval_likelihood=interval_enabled,
            interval_delta=interval_delta,
        )
        best_name, best_result, all_results = selector.select_best_distribution(data, criterion=criterion, verbose=False)

        if best_name is None or best_result is None:
            return DistributionFitSummary(
                feature_index=int(feature_index),
                family=None,
                params=None,
                cvm_p=float("nan"),
                ks_p=float("nan"),
                simple_score=float("nan"),
                confidence_set=tuple(),
                rejected=True,
                audit=audit,
                rejection_reason="fit_failed",
                selected_family_support=None,
                candidates_pre_filter=candidates_pre_filter,
                candidates_post_filter=candidates_post_filter,
            )

        score_best = float(getattr(best_result, "simple_score", np.nan))
        confidence_set = self._build_confidence_set(all_results, score_best)

        cvm_p = float(getattr(best_result, "cvm_p", np.nan))
        ks_p = float(getattr(best_result, "ks_p", np.nan))
        selected_family_support = self._family_support_class(best_name)
        support_conflict = self._is_support_conflict(audit, selected_family_support)
        rejected = self._is_rejected(best_name, cvm_p, ks_p)
        rejection_reason: Optional[str] = None
        if rejected:
            rejection_reason = "support_conflict" if support_conflict else "gof_reject"

        cv_meta = getattr(best_result, "cv_result", None)
        cv_loglik_mean = None
        cv_loglik_std = None
        cv_score = None
        if cv_meta is not None and int(getattr(cv_meta, "successful_folds", 0) or 0) > 0:
            try:
                cv_loglik_mean = float(getattr(cv_meta, "cv_loglik_mean", float("nan")))
                cv_loglik_std = float(getattr(cv_meta, "cv_loglik_std", float("nan")))
                cv_score = float(getattr(cv_meta, "cv_score", float("nan")))
            except Exception as exc:
                cv_loglik_mean = None
                cv_loglik_std = None
                cv_score = None

        crps_uq_total = None
        crps_uq_aleatoric = None
        crps_uq_epistemic = None
        if bool(getattr(self.config, "compute_crps_uq_decomposition", False)):
            try:
                member_names = list(confidence_set) if confidence_set else [str(best_name)]
                means: List[float] = []
                stds: List[float] = []
                weights: List[float] = []
                for name in member_names:
                    fr = None
                    for r in all_results:
                        if not getattr(r, "success", False):
                            continue
                        if str(getattr(r, "name", "")) != str(name):
                            continue
                        if getattr(r, "params", None) is None:
                            continue
                        fr = r
                        break
                    if fr is None:
                        continue
                    dist_obj = candidates.get(str(name))
                    if dist_obj is None:
                        continue
                    params = tuple(getattr(fr, "params"))
                    try:
                        mu = float(dist_obj.mean(*params))
                        std = float(dist_obj.std(*params))
                    except Exception as exc:
                        continue
                    if not (np.isfinite(mu) and np.isfinite(std) and std >= 0.0):
                        continue
                    means.append(float(mu))
                    stds.append(float(std))
                    w_raw = getattr(fr, "mnpo_weight", None)
                    w_val = float("nan")
                    if w_raw is not None:
                        try:
                            w_val = float(w_raw)
                        except Exception as exc:
                            w_val = float("nan")
                    weights.append(w_val)

                if means:
                    w = None
                    if crit == "mnpo_oracle":
                        w_arr = np.asarray(weights, dtype=float)
                        if np.all(np.isfinite(w_arr)) and float(np.sum(w_arr)) > 0.0:
                            w = w_arr.tolist()
                    crps_uq_total, crps_uq_aleatoric, crps_uq_epistemic = _crps_uq_decompose_gaussian_ensemble(
                        means, stds, weights=w
                    )
            except Exception as exc:
                crps_uq_total = None
                crps_uq_aleatoric = None
                crps_uq_epistemic = None

        return DistributionFitSummary(
            feature_index=int(feature_index),
            family=str(best_name),
            params=tuple(best_result.params) if getattr(best_result, "params", None) is not None else None,
            cvm_p=cvm_p,
            ks_p=ks_p,
            simple_score=score_best,
            confidence_set=confidence_set,
            rejected=rejected,
            audit=audit,
            ad_stat=(None if getattr(best_result, "ad_stat", None) is None else float(getattr(best_result, "ad_stat"))),
            ad_p=(None if getattr(best_result, "ad_p", None) is None else float(getattr(best_result, "ad_p"))),
            qq_r2=(None if getattr(best_result, "qq_r2", None) is None else float(getattr(best_result, "qq_r2"))),
            pp_r2=(None if getattr(best_result, "pp_r2", None) is None else float(getattr(best_result, "pp_r2"))),
            pp_mae=(None if getattr(best_result, "pp_mae", None) is None else float(getattr(best_result, "pp_mae"))),
            aic=(None if not np.isfinite(getattr(best_result, "aic", np.nan)) else float(getattr(best_result, "aic"))),
            aicc=(None if not np.isfinite(getattr(best_result, "aicc", np.nan)) else float(getattr(best_result, "aicc"))),
            bic=(None if not np.isfinite(getattr(best_result, "bic", np.nan)) else float(getattr(best_result, "bic"))),
            loglik=(None if not np.isfinite(getattr(best_result, "loglik", np.nan)) else float(getattr(best_result, "loglik"))),
            crps=(None if getattr(best_result, "crps", None) is None or not np.isfinite(float(getattr(best_result, "crps"))) else float(getattr(best_result, "crps"))),
            crps_uq_total=None if crps_uq_total is None or not np.isfinite(float(crps_uq_total)) else float(crps_uq_total),
            crps_uq_aleatoric=None
            if crps_uq_aleatoric is None or not np.isfinite(float(crps_uq_aleatoric))
            else float(crps_uq_aleatoric),
            crps_uq_epistemic=None
            if crps_uq_epistemic is None or not np.isfinite(float(crps_uq_epistemic))
            else float(crps_uq_epistemic),
            preq_loglik_mean=(
                None
                if getattr(best_result, "preq_loglik_mean", None) is None
                or not np.isfinite(float(getattr(best_result, "preq_loglik_mean")))
                else float(getattr(best_result, "preq_loglik_mean"))
            ),
            cv_loglik_mean=(None if cv_loglik_mean is None or not np.isfinite(cv_loglik_mean) else float(cv_loglik_mean)),
            cv_loglik_std=(None if cv_loglik_std is None or not np.isfinite(cv_loglik_std) else float(cv_loglik_std)),
            cv_score=(None if cv_score is None or not np.isfinite(cv_score) else float(cv_score)),
            fit_method=(None if getattr(best_result, "fit_method", None) is None else str(getattr(best_result, "fit_method"))),
            mnpo_weight=(None if getattr(best_result, "mnpo_weight", None) is None else float(getattr(best_result, "mnpo_weight"))),
            rejection_reason=rejection_reason,
            selected_family_support=selected_family_support,
            candidates_pre_filter=candidates_pre_filter,
            candidates_post_filter=candidates_post_filter,
        )

    def _build_confidence_set(self, all_results: Sequence[Any], score_best: float) -> Tuple[str, ...]:
        if not np.isfinite(score_best):
            return tuple()

        margin = max(1e-6, float(self.config.confidence_margin))
        keep: List[str] = []
        for r in all_results:
            if not getattr(r, "success", False):
                continue
            score = float(getattr(r, "simple_score", np.inf))
            if np.isfinite(score) and score <= score_best + margin:
                keep.append(str(getattr(r, "name", "")))
        return tuple(sorted(set(keep)))

    def _is_rejected(self, family: Optional[str], cvm_p: float, ks_p: float) -> bool:
        if family is None:
            return True
        if not self.config.rejection_gate:
            return False
        thr = float(self.config.rejection_p_threshold)
        cvm_bad = (not np.isfinite(cvm_p)) or cvm_p < thr
        ks_bad = (not np.isfinite(ks_p)) or ks_p < thr
        return bool(cvm_bad and ks_bad)


class DistributionFeatureSelectionPipeline:
    """Integrated DF+FS pipeline with strict leakage-safe protocol."""

    def __init__(self, config: Optional[DFFSConfig] = None):
        self.config = config or DFFSConfig()
        _warn_deprecated_df_fastpath_config(self.config)
        self.dist_fitter = DistributionFitter(self.config.dist_config)
        self._warned_missing_model_backends: Set[str] = set()
        self._last_distribution_plan: Dict[str, Any] = {}
        self._last_face_projection_state: Dict[str, Any] = {}
        self._last_folding_state: Dict[str, Any] = {}

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "dataset",
        seed: Optional[int] = None,
        batch_labels: Optional[Sequence[Any]] = None,
        *,
        capture_artifacts: bool = False,
        capture_diagnostics: bool = False,
    ) -> PipelineRunResult:
        seed = int(self.config.random_seed if seed is None else seed)
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D X, got shape {X_arr.shape}")
        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()
        batch_arr: Optional[np.ndarray]
        if batch_labels is None:
            batch_arr = None
        else:
            batch_arr = np.asarray(list(batch_labels), dtype=object).ravel()
            if int(batch_arr.size) != int(X_arr.shape[0]):
                raise ValueError(
                    f"batch_labels has {batch_arr.size} rows but X has {X_arr.shape[0]}."
                )

        idx_all = np.arange(X_arr.shape[0], dtype=int)
        train_idx, test_idx = self._split_indices(idx_all, y_arr, seed)

        return self.run_pre_split(
            X_train=X_arr[train_idx],
            y_train=y_arr[train_idx],
            X_test=X_arr[test_idx],
            y_test=y_arr[test_idx],
            dataset_name=dataset_name,
            seed=seed,
            split_indices_train=train_idx,
            split_indices_test=test_idx,
            batch_labels_train=(
                None
                if batch_arr is None
                else np.asarray(batch_arr[train_idx], dtype=object)
            ),
            batch_labels_test=(
                None
                if batch_arr is None
                else np.asarray(batch_arr[test_idx], dtype=object)
            ),
            capture_artifacts=bool(capture_artifacts),
            capture_diagnostics=bool(capture_diagnostics),
        )

    def _resolve_meta_learning_runtime_config(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[Optional[DFFSConfig], Dict[str, Any]]:
        mode = str(getattr(self.config, "meta_learning_selector_mode", "none") or "none").strip().lower()
        if mode == "none":
            return None, {}
        try:
            from tabnetics.feature_selection.meta_learning import (
                MetaLearningSelector,
                SUPPORTED_RUNTIME_PROFILES,
                apply_runtime_profile_overlay,
            )
        except Exception:
            from tabnetics.feature_selection.meta_learning import (  # type: ignore
                MetaLearningSelector,
                SUPPORTED_RUNTIME_PROFILES,
                apply_runtime_profile_overlay,
            )

        selected_meta: Dict[str, Any]
        try:
            selector = MetaLearningSelector(
                mode=str(mode),
                confidence_threshold=float(
                    getattr(self.config, "meta_learning_confidence_threshold", 0.55) or 0.55
                ),
                random_state=int(getattr(self.config, "random_seed", 42) or 42),
            ).fit()
            selected_meta = selector.predict_from_arrays(
                np.asarray(X_train, dtype=float),
                np.asarray(y_train).ravel(),
            )
        except Exception as exc:
            selected_meta = {
                "meta_learning_profile_selected": "v16_ref",
                "meta_learning_profile_raw": "v16_ref",
                "meta_learning_confidence": 0.0,
                "meta_learning_fallback_applied": True,
                "meta_learning_candidate_profiles": list(SUPPORTED_RUNTIME_PROFILES),
                "meta_learning_error": str(type(exc).__name__),
            }

        active_config = copy.deepcopy(self.config)
        active_config.meta_learning_selector_mode = "none"
        apply_runtime_profile_overlay(
            active_config,
            str(selected_meta.get("meta_learning_profile_selected", "v16_ref") or "v16_ref"),
        )
        return active_config, selected_meta

    def _apply_multiomics_adapter_train_test(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        *,
        seed: int,
        batch_labels_train: Optional[np.ndarray] = None,
        batch_labels_test: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        del seed  # Adapter behavior is deterministic for a fixed train/test split.
        return apply_multiomics_adapter_train_test(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            adapter_mode=str(getattr(self.config, "multiomics_adapter", "none") or "none"),
            integrator=str(getattr(self.config, "multiomics_integrator", "mb_plsda") or "mb_plsda"),
            n_components=int(getattr(self.config, "multiomics_n_components", 2) or 2),
            feature_blocks=getattr(self.config, "multiomics_feature_blocks", None),
            batch_labels_train=batch_labels_train,
            batch_labels_test=batch_labels_test,
        )

    @staticmethod
    def _empty_distribution_stage_meta() -> Dict[str, Any]:
        return {
            "transform_time_sec": 0.0,
            "n_fitted": 0,
            "n_transformed": 0,
            "n_rejected": 0,
            "n_skipped_unreliable": 0,
            "n_skipped_block_cv": 0,
            "n_downweighted": 0,
            "mean_stability_weight": 1.0,
            "cdf_block_gating_time_sec": 0.0,
            "cdf_block_gating_budget_hit": 0.0,
            "cdf_block_gating_blocks_evaluated": 0,
            "cdf_block_gating_blocks_applied": 0,
        }

    def _df_stage_position(self) -> str:
        stage_position = str(getattr(self.config, "df_stage_position", "after_fs") or "after_fs").strip().lower()
        return "before_fs" if stage_position == "before_fs" else "after_fs"

    def _annotate_distribution_stage_meta(
        self,
        meta: Dict[str, Any],
        *,
        stage_position: str,
        source_space: str,
    ) -> Dict[str, Any]:
        out = dict(self._empty_distribution_stage_meta())
        out.update(dict(meta or {}))
        out["df_stage_position"] = str(stage_position)
        out["df_stage_source_space"] = str(source_space)
        plan = dict(self._last_distribution_plan or {})
        plan["df_stage_position"] = str(stage_position)
        plan["df_stage_source_space"] = str(source_space)
        self._last_distribution_plan = plan
        return out

    def _apply_post_selection_distribution_transform(
        self,
        *,
        X_train_selected: np.ndarray,
        X_test_selected: np.ndarray,
        selected_indices: Sequence[int],
        y_train: np.ndarray,
        seed: int,
        source_space: str,
        source_raw_train: Optional[np.ndarray] = None,
        source_raw_test: Optional[np.ndarray] = None,
        source_base_train: Optional[np.ndarray] = None,
        source_base_test: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[DistributionFitSummary], Dict[str, Any], float]:
        if self._df_stage_position() != "after_fs":
            meta = self._annotate_distribution_stage_meta(
                {},
                stage_position="before_fs",
                source_space="model_input",
            )
            return (
                np.asarray(X_train_selected, dtype=float),
                np.asarray(X_test_selected, dtype=float),
                [],
                meta,
                0.0,
            )

        selected_train = np.asarray(X_train_selected, dtype=float)
        selected_test = np.asarray(X_test_selected, dtype=float)
        effective_source_space = str(source_space or "selected_selector_input")
        raw_train = selected_train
        raw_test = selected_test
        base_train = selected_train
        base_test = selected_test

        if effective_source_space != "folded_selector_input":
            idx = np.asarray(list(selected_indices), dtype=int).ravel()
            raw_full_train = None if source_raw_train is None else np.asarray(source_raw_train, dtype=float)
            raw_full_test = None if source_raw_test is None else np.asarray(source_raw_test, dtype=float)
            base_full_train = None if source_base_train is None else np.asarray(source_base_train, dtype=float)
            base_full_test = None if source_base_test is None else np.asarray(source_base_test, dtype=float)
            can_slice = (
                raw_full_train is not None
                and raw_full_test is not None
                and base_full_train is not None
                and base_full_test is not None
                and idx.size > 0
                and raw_full_train.ndim == 2
                and raw_full_test.ndim == 2
                and base_full_train.ndim == 2
                and base_full_test.ndim == 2
                and int(np.min(idx)) >= 0
                and int(np.max(idx)) < raw_full_train.shape[1]
                and int(np.max(idx)) < raw_full_test.shape[1]
                and int(np.max(idx)) < base_full_train.shape[1]
                and int(np.max(idx)) < base_full_test.shape[1]
            )
            if can_slice and int(idx.size) == int(selected_train.shape[1]):
                raw_train = np.asarray(raw_full_train[:, idx], dtype=float)
                raw_test = np.asarray(raw_full_test[:, idx], dtype=float)
                base_train = np.asarray(base_full_train[:, idx], dtype=float)
                base_test = np.asarray(base_full_test[:, idx], dtype=float)
            else:
                effective_source_space = "selected_selector_input"

        dist_start = self._timer()
        X_train_out, X_test_out, summaries, meta = self._distribution_transform_block(
            np.asarray(raw_train, dtype=float),
            np.asarray(raw_test, dtype=float),
            np.asarray(base_train, dtype=float),
            np.asarray(base_test, dtype=float),
            np.asarray(y_train),
            int(seed),
            np.random.default_rng(int(seed)),
        )
        time_sec = float(self._timer() - dist_start)
        meta = self._annotate_distribution_stage_meta(
            meta,
            stage_position="after_fs",
            source_space=effective_source_space,
        )
        return X_train_out, X_test_out, summaries, meta, time_sec

    def run_pre_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str = "dataset",
        seed: Optional[int] = None,
        split_indices_train: Optional[Sequence[int]] = None,
        split_indices_test: Optional[Sequence[int]] = None,
        batch_labels_train: Optional[Sequence[Any]] = None,
        batch_labels_test: Optional[Sequence[Any]] = None,
        *,
        capture_artifacts: bool = False,
        capture_diagnostics: bool = False,
        _meta_learning_resolution: Optional[Dict[str, Any]] = None,
    ) -> PipelineRunResult:
        """Run DF+FS pipeline on an explicit train/test split.

        This is primarily used for protocol overlays (e.g., repeated nested CV audits)
        where the caller controls the split and the pipeline must not re-split.
        """
        seed = int(self.config.random_seed if seed is None else seed)
        rng = np.random.default_rng(seed)

        X_train_arr = np.asarray(X_train, dtype=float)
        y_train_arr = np.asarray(y_train)
        X_test_arr = np.asarray(X_test, dtype=float)
        y_test_arr = np.asarray(y_test)

        if X_train_arr.ndim != 2:
            raise ValueError(f"Expected 2D X_train, got shape {X_train_arr.shape}")
        if X_test_arr.ndim != 2:
            raise ValueError(f"Expected 2D X_test, got shape {X_test_arr.shape}")
        if X_train_arr.shape[1] != X_test_arr.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: X_train has {X_train_arr.shape[1]} "
                f"features but X_test has {X_test_arr.shape[1]}."
            )
        if y_train_arr.ndim != 1:
            y_train_arr = y_train_arr.ravel()
        if y_test_arr.ndim != 1:
            y_test_arr = y_test_arr.ravel()
        if X_train_arr.shape[0] != y_train_arr.shape[0]:
            raise ValueError(
                f"Row mismatch: X_train has {X_train_arr.shape[0]} rows but y_train has {y_train_arr.shape[0]}."
            )
        if X_test_arr.shape[0] != y_test_arr.shape[0]:
            raise ValueError(
                f"Row mismatch: X_test has {X_test_arr.shape[0]} rows but y_test has {y_test_arr.shape[0]}."
            )
        if batch_labels_train is None:
            batch_train_arr = None
        else:
            batch_train_arr = np.asarray(list(batch_labels_train), dtype=object).ravel()
            if int(batch_train_arr.size) != int(X_train_arr.shape[0]):
                raise ValueError(
                    f"batch_labels_train has {batch_train_arr.size} rows but X_train has {X_train_arr.shape[0]}."
                )
        if batch_labels_test is None:
            batch_test_arr = None
        else:
            batch_test_arr = np.asarray(list(batch_labels_test), dtype=object).ravel()
            if int(batch_test_arr.size) != int(X_test_arr.shape[0]):
                raise ValueError(
                    f"batch_labels_test has {batch_test_arr.size} rows but X_test has {X_test_arr.shape[0]}."
                )

        if _meta_learning_resolution is None:
            active_config, meta_learning_resolution = self._resolve_meta_learning_runtime_config(
                X_train_arr,
                y_train_arr,
            )
            if active_config is not None:
                delegated = DistributionFeatureSelectionPipeline(active_config)
                return delegated.run_pre_split(
                    X_train=X_train_arr,
                    y_train=y_train_arr,
                    X_test=X_test_arr,
                    y_test=y_test_arr,
                    dataset_name=dataset_name,
                    seed=seed,
                    split_indices_train=split_indices_train,
                    split_indices_test=split_indices_test,
                    batch_labels_train=batch_train_arr,
                    batch_labels_test=batch_test_arr,
                    capture_artifacts=bool(capture_artifacts),
                    capture_diagnostics=bool(capture_diagnostics),
                    _meta_learning_resolution=meta_learning_resolution,
                )
        meta_learning_resolution = dict(_meta_learning_resolution or {})

        n_total = int(X_train_arr.shape[0] + X_test_arr.shape[0])
        n_features = int(X_train_arr.shape[1])

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_arr)
        X_test_imp = imputer.transform(X_test_arr)

        X_train_imp, X_test_imp, multiomics_meta = self._apply_multiomics_adapter_train_test(
            X_train_imp,
            y_train_arr,
            X_test_imp,
            seed=seed,
            batch_labels_train=batch_train_arr,
            batch_labels_test=batch_test_arr,
        )

        batch_model, batch_fit_meta = fit_batch_correction_model(
            X_train_imp,
            batch_labels=batch_train_arr,
            mode=str(getattr(self.config, "batch_correction", "none") or "none"),
            combat_prior_strength=float(
                getattr(self.config, "batch_correction_combat_prior_strength", 8.0) or 8.0
            ),
            cdf_center_n_quantiles=int(
                getattr(self.config, "batch_correction_cdf_n_quantiles", 33) or 33
            ),
            cdf_center_clip_quantiles=(
                float(getattr(self.config, "batch_correction_cdf_clip_low", 0.01) or 0.01),
                float(getattr(self.config, "batch_correction_cdf_clip_high", 0.99) or 0.99),
            ),
        )
        X_train_imp, X_test_imp, batch_apply_meta = apply_batch_correction_model(
            X_train_imp,
            X_test_imp,
            model=batch_model,
            batch_labels_train=batch_train_arr,
            batch_labels_test=batch_test_arr,
        )

        X_train_model_input, X_test_model_input, face_meta = self._maybe_apply_face_domain_projection(
            X_train_imp=X_train_imp,
            y_train=y_train_arr,
            X_test_imp=X_test_imp,
            dataset_name=str(dataset_name),
            seed=seed,
        )

        X_train_model_input, X_test_model_input, ratio_meta = self._ratio_feature_generation(
            X_train_imp=X_train_model_input,
            y_train=y_train_arr,
            X_test_imp=X_test_model_input,
            seed=seed,
            face_projection_applied=bool(face_meta.get("face_projection_applied", False)),
        )

        _scaler_mode = str(getattr(self.config, "scaler_mode", "standard") or "standard").strip().lower()
        if _scaler_mode == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler_base = RobustScaler()
        elif _scaler_mode == "quantile":
            from sklearn.preprocessing import QuantileTransformer
            scaler_base = QuantileTransformer(output_distribution="normal", random_state=seed)
        else:
            scaler_base = StandardScaler()
        X_train_base = scaler_base.fit_transform(X_train_model_input)
        X_test_base = scaler_base.transform(X_test_model_input)
        df_stage_position = self._df_stage_position()
        if df_stage_position == "before_fs":
            dist_start = self._timer()
            X_train_trans, X_test_trans, dist_summaries, dist_meta = self._distribution_transform_block(
                X_train_model_input,
                X_test_model_input,
                X_train_base,
                X_test_base,
                y_train_arr,
                seed,
                rng,
            )
            dist_time_sec = float(self._timer() - dist_start)
            dist_meta = self._annotate_distribution_stage_meta(
                dist_meta,
                stage_position="before_fs",
                source_space="model_input",
            )
        else:
            X_train_trans = np.asarray(X_train_base, dtype=float)
            X_test_trans = np.asarray(X_test_base, dtype=float)
            dist_summaries = []
            self._last_distribution_plan = {
                "schema_version": "1.0",
                "apply_cdf_transform": bool(self.config.apply_cdf_transform),
                "n_input_features": 0,
                "dist_feature_indices": [],
                "feature_plans": [],
            }
            dist_meta = self._annotate_distribution_stage_meta(
                {},
                stage_position="after_fs",
                source_space="pending",
            )
            dist_time_sec = 0.0

        fs_start = self._timer()
        folding_prefilter_k = getattr(self.config, "folding_prefilter_k", None)
        prefilter_top_k_override = None
        if str(getattr(self.config, "folding_method", "pls_da") or "pls_da").strip().lower() != "none":
            if folding_prefilter_k is not None:
                try:
                    prefilter_top_k_override = int(max(1, int(folding_prefilter_k)))
                except Exception as exc:
                    prefilter_top_k_override = None
        X_train_fs_input, X_test_fs_input, prefilter_idx = self._rank_prefilter(
            X_train_trans,
            y_train_arr,
            X_test_trans,
            seed,
            top_k_override=prefilter_top_k_override,
        )
        X_train_prefilter_raw = np.asarray(X_train_model_input[:, prefilter_idx], dtype=float)
        X_test_prefilter_raw = np.asarray(X_test_model_input[:, prefilter_idx], dtype=float)
        X_train_prefilter_base = np.asarray(X_train_base[:, prefilter_idx], dtype=float)
        X_test_prefilter_base = np.asarray(X_test_base[:, prefilter_idx], dtype=float)
        post_df_source_space = "prefilter_raw"
        post_df_source_raw_train = X_train_prefilter_raw
        post_df_source_raw_test = X_test_prefilter_raw
        post_df_source_base_train = X_train_prefilter_base
        post_df_source_base_test = X_test_prefilter_base
        if df_stage_position == "after_fs":
            folding_meta: Dict[str, Any] = {}
            self._last_folding_state = {}
        else:
            X_train_fs_input, X_test_fs_input, folding_meta = self._apply_folding_stage(
                X_train_fs_input=X_train_fs_input,
                X_test_fs_input=X_test_fs_input,
                y_train=y_train_arr,
                seed=seed,
            )
            if bool(folding_meta.get("folding_applied", False)):
                post_df_source_space = "folded_selector_input"
                post_df_source_raw_train = np.asarray(X_train_fs_input, dtype=float)
                post_df_source_raw_test = np.asarray(X_test_fs_input, dtype=float)
                post_df_source_base_train = np.asarray(X_train_fs_input, dtype=float)
                post_df_source_base_test = np.asarray(X_test_fs_input, dtype=float)

        fs_idx_local = self._sample_fs_indices(
            y_train_arr,
            fs_fraction=self.config.fs_fraction,
            seed=seed,
            use_balanced=self.config.use_balanced_fs_subsample,
            min_per_class=self.config.fs_min_per_class,
        )
        X_fs = X_train_fs_input[fs_idx_local]
        y_fs = y_train_arr[fs_idx_local]

        fs_result = self._run_feature_selection(
            X_fs,
            y_fs,
            X_train_fs_input,
            X_test_fs_input,
            y_train_arr,
            y_test_arr,
            seed,
            dataset_name=str(dataset_name),
            post_df_source_raw_train=post_df_source_raw_train,
            post_df_source_raw_test=post_df_source_raw_test,
            post_df_source_base_train=post_df_source_base_train,
            post_df_source_base_test=post_df_source_base_test,
            post_df_source_space=post_df_source_space,
        )
        fs_time_sec = float(self._timer() - fs_start)
        late_folding_meta = dict(fs_result.pop("_folding_meta", {}) or {})
        late_folding_state = dict(fs_result.pop("_folding_state", {}) or {})
        if df_stage_position == "after_fs":
            dist_summaries = list(fs_result.pop("_post_df_summaries", []) or [])
            post_df_meta = dict(fs_result.pop("_post_df_meta", {}) or {})
            folding_meta = late_folding_meta
            self._last_folding_state = dict(late_folding_state or {})
            dist_meta = self._annotate_distribution_stage_meta(
                post_df_meta,
                stage_position="after_fs",
                source_space=str(post_df_meta.get("df_stage_source_space", post_df_source_space)),
            )
            dist_time_sec = float(fs_result.pop("_post_df_time_sec", 0.0) or 0.0)
            fs_time_sec = float(max(0.0, fs_time_sec - dist_time_sec))

        selected_local = np.asarray(fs_result["selected_indices"], dtype=int)
        if bool(face_meta.get("face_projection_applied", False)) or (
            df_stage_position != "after_fs" and bool(folding_meta.get("folding_applied", False))
        ):
            selected_original = tuple()
        else:
            selected_original = tuple(int(prefilter_idx[i]) for i in selected_local if 0 <= i < len(prefilter_idx))

        if split_indices_train is None:
            train_idx_out = tuple()
        else:
            train_idx_out = tuple(int(i) for i in np.asarray(split_indices_train, dtype=int).ravel().tolist())

        if split_indices_test is None:
            test_idx_out = tuple()
        else:
            test_idx_out = tuple(int(i) for i in np.asarray(split_indices_test, dtype=int).ravel().tolist())

        snapshot = self._config_snapshot()
        snapshot["df_stage_position_effective"] = str(dist_meta.get("df_stage_position", df_stage_position))
        snapshot["df_stage_source_space"] = str(dist_meta.get("df_stage_source_space", "model_input"))
        effective = fs_result.get("effective_enabled_methods")
        if effective is not None:
            snapshot["enabled_methods"] = [str(m) for m in effective]
            snapshot["effective_enabled_methods"] = [str(m) for m in effective]
        if "enabled_methods_source" in fs_result:
            snapshot["enabled_methods_source"] = str(fs_result["enabled_methods_source"])
        for key in (
            "maqc_pairing_enabled",
            "maqc_pairing_selected_fs_name",
            "maqc_pairing_selected_cv_score",
            "maqc_pairing_candidate_count",
            "maqc_pairing_evaluated_count",
            "maqc_pairing_failed_count",
        ):
            if key in fs_result:
                snapshot[key] = fs_result[key]
        for key in (
            "tier_policy_applied",
            "tier_policy_mode",
            "tier_policy_target_tier",
            "tier_policy_resolved_tier",
            "tier_policy_source",
            "tier_policy_enabled_methods_before",
            "tier_policy_enabled_methods_after",
            "tier_policy_meta_features",
        ):
            if key in fs_result:
                snapshot[key] = fs_result[key]
        for key in (
            "regime_policy_applied",
            "regime_policy_mode",
            "regime_policy_reason",
            "regime_policy_enabled",
            "regime_policy_target_tier",
            "regime_policy_tier",
            "regime_policy_tier_source",
            "regime_policy_tier_meta_features",
            "regime_policy_n_samples",
            "regime_policy_n_features",
            "regime_policy_n_classes",
            "regime_policy_samples_per_class",
            "regime_policy_p_over_n",
            "regime_policy_min_samples_per_class_threshold",
            "regime_policy_low_p_over_n_threshold",
            "regime_policy_trigger_very_hard",
            "regime_policy_trigger_low_p_over_n",
            "regime_policy_enabled_methods_before",
            "regime_policy_enabled_methods_after",
            "regime_policy_enabled_methods_source",
            "regime_policy_bypass_fs",
            "regime_policy_bypass_mode",
            "regime_policy_selector_overrides",
            "selector_overrides_applied",
        ):
            if key in fs_result:
                snapshot[key] = fs_result[key]
        for key in (
            "importance_uq_enabled",
            "importance_uq_computed",
            "importance_uq_reason",
            "importance_uq_n_folds",
            "importance_uq_unstable_threshold",
            "importance_uq_unstable_feature_count",
            "importance_uq_unstable_feature_indices",
        ):
            if key in fs_result:
                snapshot[key] = fs_result[key]
        for key in (
            "model_cv_runtime_containment_enabled",
            "model_cv_runtime_containment_applied",
            "model_cv_runtime_containment_reason",
            "model_cv_requested_candidates",
            "model_cv_effective_candidates",
            "model_cv_dropped_candidates",
            "model_cv_runtime_cap",
            "model_cv_runtime_max_candidates_cfg",
            "model_cv_runtime_p_over_n",
            "model_cv_runtime_n_classes",
            "model_cv_runtime_min_class_count",
            "model_cv_runtime_regime_high_p_over_n",
            "model_cv_runtime_regime_high_class",
            "model_cv_runtime_regime_sparse_class",
            "model_cv_constructed_candidates",
            "model_cv_candidate_build_failures",
            "model_cv_failed_candidates",
            "model_cv_candidate_failure_reasons",
            "model_cv_evaluated_candidates",
            "model_cv_candidate_scores",
            "classification_backend_requested",
            "classification_backend_used",
            "classification_backend_fallback_reason",
            "classification_stage2_wall_seconds",
            "roc_auc",
            "roc_curve_type",
            "roc_auc_source",
            "roc_curve_points",
            "roc_curves_by_method",
            "classifier_conformal_enabled",
            "classifier_conformal_applied",
            "classifier_conformal_skip_reason",
            "classifier_conformal_alpha",
            "classifier_conformal_calibration_fraction",
            "classifier_conformal_min_calibration",
            "classifier_conformal_calibration_size",
            "classifier_conformal_fit_size",
            "classifier_conformal_qhat",
            "classifier_conformal_threshold",
            "classifier_conformal_set_size_mean",
            "classifier_conformal_set_size_median",
            "classifier_conformal_singleton_rate",
            "classifier_conformal_empty_set_rate",
            "classifier_conformal_coverage",
            "classifier_conformal_classes",
            "classifier_conformal_prediction_sets",
            "classifier_conformal_method",
            "classifier_conformal_mapie_enabled",
            "classifier_conformal_mapie_applied",
            "classifier_conformal_mapie_skip_reason",
            "classifier_conformal_mapie_method",
            "classifier_conformal_mapie_alpha",
            "classifier_conformal_mapie_set_size_mean",
            "classifier_conformal_mapie_set_size_median",
            "classifier_conformal_mapie_singleton_rate",
            "classifier_conformal_mapie_empty_set_rate",
            "classifier_conformal_mapie_coverage",
            "classifier_conformal_mapie_classes",
            "classifier_conformal_mapie_prediction_sets",
            "classification_flaml_time_budget_raw",
            "classification_flaml_budget_divisor",
            "classification_flaml_time_budget_divided",
            "classification_flaml_time_budget_effective",
            "classification_optuna_time_budget_raw",
            "classification_optuna_budget_divisor",
            "classification_optuna_time_budget_divided",
            "classification_optuna_time_budget_effective",
            "classification_optuna_n_trials",
            "classification_selection_mode",
            "classification_mnpo_use_per_family_flaml",
            "classification_mnpo_oracle_k",
            "optuna_time_budget",
            "optuna_n_trials_requested",
            "optuna_n_trials_completed",
            "optuna_best_family",
            "optuna_best_params",
            "optuna_best_value",
            "optuna_used_tuned_params",
            "optuna_failure_reason",
            "classification_regime",
            "classification_regime_pool",
            "classification_regime_dropped_candidates",
            "mnpo_selected_classifier",
            "mnpo_selected_candidates",
            "mnpo_candidate_weights",
            "mnpo_oracle_weights",
            "mnpo_candidate_stats",
            "mnpo_race_meta",
            "mnpo_hpo_mode",
            "mnpo_hpo_budget",
            "mnpo_hpo_meta",
            "mnpo_hpo_by_family",
            "stage2_ratio_augmentation_enabled",
            "stage2_ratio_features_applied",
            "stage2_ratio_features_reason",
            "stage2_ratio_selection_method",
            "stage2_ratio_max_features",
            "stage2_ratio_epsilon",
            "stage2_ratio_pool_size_effective",
            "stage2_ratio_pairs_considered",
            "stage2_ratio_features_added",
            "stage2_ratio_feature_start_index",
            "stage2_ratio_pairs",
        ):
            if key in fs_result:
                snapshot[key] = fs_result[key]
        for key, val in face_meta.items():
            snapshot[str(key)] = val
        for key, val in ratio_meta.items():
            snapshot[str(key)] = val
        for key, val in folding_meta.items():
            snapshot[str(key)] = val
        for key, val in multiomics_meta.items():
            snapshot[str(key)] = val
        for key, val in batch_fit_meta.items():
            snapshot[str(key)] = val
        for key, val in batch_apply_meta.items():
            snapshot[str(key)] = val
        for key, val in meta_learning_resolution.items():
            snapshot[str(key)] = val

        model_bundle: Dict[str, Any] = {}
        if bool(capture_artifacts):
            try:
                stage2_ratio_meta = {
                    str(k): v
                    for k, v in fs_result.items()
                    if str(k).startswith("stage2_ratio_")
                }
                face_state = dict(self._last_face_projection_state or {})
                folding_state = dict(self._last_folding_state or {})
                fitted_selector = fs_result.get("_fitted_selector")
                fitted_model = fs_result.get("_fitted_model")
                if fitted_selector is None or fitted_model is None:
                    raise RuntimeError("Missing fitted selector/model state from feature-selection stage.")
                model_bundle_obj = DFFSReproducibleModel(
                    n_input_features=int(n_features),
                    imputer=imputer,
                    batch_model=batch_model if isinstance(batch_model, dict) else None,
                    face_meta=face_meta,
                    face_pca=face_state.get("pca_model"),
                    face_lda=face_state.get("lda_model"),
                    ratio_meta=ratio_meta,
                    scaler_base=scaler_base,
                    distribution_plan=dict(self._last_distribution_plan or {}),
                    prefilter_indices=tuple(int(i) for i in np.asarray(prefilter_idx, dtype=int).ravel().tolist()),
                    folding_meta=folding_meta,
                    folding_transformer=folding_state.get("transformer"),
                    folding_standardize_mean=folding_state.get("standardize_mean"),
                    folding_standardize_scale=folding_state.get("standardize_scale"),
                    selector=fitted_selector,
                    stage2_ratio_meta=stage2_ratio_meta,
                    classifier_model=fitted_model,
                    metadata={
                        "dataset_name": str(dataset_name),
                        "seed": int(seed),
                        "n_train": int(X_train_arr.shape[0]),
                        "n_test": int(X_test_arr.shape[0]),
                        "model_name": str(fs_result.get("model_name", "")),
                        "selected_features_count": int(fs_result.get("selected_features", 0) or 0),
                        "selected_feature_indices_original": [
                            int(i) for i in tuple(selected_original)
                        ],
                    },
                )
                model_bundle = model_bundle_obj.to_json_dict()
            except Exception as exc:
                model_bundle = {
                    "schema_version": "1.0",
                    "artifact_type": "df_fs_model_bundle",
                    "artifact_error": str(type(exc).__name__),
                    "artifact_error_message": str(exc),
                    "dataset_name": str(dataset_name),
                    "seed": int(seed),
                }

        run_diagnostics: Dict[str, Any] = {}
        if bool(capture_diagnostics):
            classifier_oracle_meta: Dict[str, Any] = {}
            for key, val in fs_result.items():
                key_s = str(key)
                if key_s.startswith("_"):
                    continue
                if (
                    key_s.startswith("model_cv_")
                    or key_s.startswith("classification_")
                    or key_s.startswith("mnpo_")
                    or key_s.startswith("maqc_pairing_")
                ):
                    classifier_oracle_meta[key_s] = _json_safe(val)

            run_diagnostics = _json_safe({
                "schema_version": "1.0",
                "artifact_type": "df_fs_run_diagnostics",
                "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "dataset_name": str(dataset_name),
                "seed": int(seed),
                "split": {
                    "train_indices": [int(i) for i in train_idx_out],
                    "test_indices": [int(i) for i in test_idx_out],
                    "n_train": int(X_train_arr.shape[0]),
                    "n_test": int(X_test_arr.shape[0]),
                },
                "metrics": {
                    "accuracy": float(fs_result.get("accuracy", float("nan"))),
                    "balanced_accuracy": float(fs_result.get("balanced_accuracy", float("nan"))),
                    "macro_f1": float(fs_result.get("macro_f1", float("nan"))),
                    "hybrid_score": float(fs_result.get("hybrid_score", float("nan"))),
                    "roc_auc": float(fs_result.get("roc_auc", float("nan"))),
                    "log_loss": float(fs_result.get("log_loss", float("nan"))),
                    "model_name": str(fs_result.get("model_name", "")),
                },
                "timing_seconds": {
                    "distribution_fit": float(dist_time_sec),
                    "feature_selection_plus_classification": float(fs_time_sec),
                    "distribution_transform_only": float(dist_meta.get("transform_time_sec", 0.0) or 0.0),
                },
                "pipeline_stages": {
                    "imputation": {
                        "strategy": "median",
                        "statistics": _json_safe(np.asarray(getattr(imputer, "statistics_", np.array([])), dtype=float)),
                    },
                    "batch_correction_fit": _json_safe(batch_fit_meta),
                    "batch_correction_apply": _json_safe(batch_apply_meta),
                    "face_projection": _json_safe(face_meta),
                    "ratio_stage1": _json_safe(ratio_meta),
                    "distribution_stage": {
                        "summary": _json_safe(dist_meta),
                        "transform_plan": _json_safe(dict(self._last_distribution_plan or {})),
                        "feature_summaries": [
                            _serialize_distribution_summary(summary) for summary in list(dist_summaries or [])
                        ],
                    },
                    "prefilter": {
                        "selected_indices_after_prefilter": [
                            int(i) for i in np.asarray(prefilter_idx, dtype=int).ravel().tolist()
                        ],
                        "selected_count": int(np.asarray(prefilter_idx, dtype=int).size),
                    },
                    "folding_stage": _json_safe(folding_meta),
                    "feature_selection": {
                        "selected_indices_local": [
                            int(i) for i in np.asarray(fs_result.get("selected_indices", tuple()), dtype=int).ravel().tolist()
                        ],
                        "selected_indices_original": [int(i) for i in tuple(selected_original)],
                        "selection_summary": _json_safe(fs_result.get("fs_selection_summary", {})),
                        "detailed": _json_safe(fs_result.get("fs_diagnostics", {})),
                    },
                    "classifier_selection": _json_safe(classifier_oracle_meta),
                },
            })

        return PipelineRunResult(
            dataset_name=str(dataset_name),
            seed=seed,
            n_samples_total=n_total,
            n_features_total=n_features,
            n_train=int(X_train_arr.shape[0]),
            n_test=int(X_test_arr.shape[0]),
            n_fs_subset=int(len(fs_idx_local)),
            accuracy=float(fs_result["accuracy"]),
            balanced_accuracy=float(fs_result["balanced_accuracy"]),
            macro_f1=float(fs_result["macro_f1"]),
            hybrid_score=float(fs_result["hybrid_score"]),
            roc_auc=float(fs_result.get("roc_auc", float("nan"))),
            log_loss=float(fs_result.get("log_loss", float("nan"))),
            roc_curve_type=str(fs_result.get("roc_curve_type", "unavailable")),
            roc_auc_source=str(fs_result.get("roc_auc_source", "unavailable")),
            roc_curve_points=tuple(
                (
                    float(pt[0]),
                    float(pt[1]),
                )
                for pt in (fs_result.get("roc_curve_points") or tuple())
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ),
            roc_curves_by_method=dict(fs_result.get("roc_curves_by_method") or {}),
            selected_features_count=int(fs_result["selected_features"]),
            selected_feature_indices_original=selected_original,
            model_name=str(fs_result["model_name"]),
            fs_time_sec=float(fs_time_sec),
            dist_time_sec=float(dist_time_sec),
            transform_time_sec=float(dist_meta["transform_time_sec"]),
            n_dist_features_fitted=int(dist_meta["n_fitted"]),
            n_dist_features_transformed=int(dist_meta["n_transformed"]),
            n_dist_rejected=int(dist_meta["n_rejected"]),
            n_dist_skipped_unreliable=int(dist_meta["n_skipped_unreliable"]),
            n_dist_skipped_block_cv=int(dist_meta["n_skipped_block_cv"]),
            n_low_gof_downweighted=int(dist_meta["n_downweighted"]),
            mean_dist_stability_weight=float(dist_meta["mean_stability_weight"]),
            cdf_block_gating_time_sec=float(dist_meta["cdf_block_gating_time_sec"]),
            cdf_block_gating_budget_hit=bool(dist_meta["cdf_block_gating_budget_hit"]),
            cdf_block_gating_blocks_evaluated=int(dist_meta["cdf_block_gating_blocks_evaluated"]),
            cdf_block_gating_blocks_applied=int(dist_meta["cdf_block_gating_blocks_applied"]),
            split_indices_train=train_idx_out,
            split_indices_test=test_idx_out,
            distribution_summaries=dist_summaries,
            config_snapshot=snapshot,
            model_bundle=model_bundle,
            run_diagnostics=run_diagnostics,
        )

    def _fit_transform_one_feature(
        self,
        feat_idx: int,
        X_train_imp: np.ndarray,
        X_test_imp: np.ndarray,
        seed: int,
    ) -> Dict[str, Any]:
        """Fit DF and compute CDF-Gaussian transform for a single feature.

        Thread-safe: no shared mutable state is modified.
        Returns a dict with all per-feature results for aggregation.
        """
        train_col = X_train_imp[:, feat_idx]
        test_col = X_test_imp[:, feat_idx]
        audit = self.dist_fitter.audit_data(train_col)
        fallback_payload = self._try_multimodal_fallback_transform(
            feat_idx=int(feat_idx),
            train_col=train_col,
            test_col=test_col,
            audit=audit,
            seed=int(seed),
            criterion=str(getattr(self.config, "dist_criterion", "simple") or "simple"),
        )

        if fallback_payload is not None:
            return {
                "feat_idx": int(feat_idx),
                "summary": fallback_payload["summary"],
                "rejected": False,
                "skipped_unreliable": False,
                "downweighted": False,
                "stability_weight": None,
                "weight": 1.0,
                "train_mean": float(fallback_payload["train_mean"]),
                "train_std": float(fallback_payload["train_std"]),
                "apply_reason": "multimodal_fallback",
                "fallback_meta": dict(fallback_payload.get("fallback_meta") or {}),
                "train_z": np.asarray(fallback_payload["train_z"], dtype=float),
                "test_z": np.asarray(fallback_payload["test_z"], dtype=float),
            }

        summary = self.dist_fitter.select_best_distribution(
            train_col,
            criterion=self.config.dist_criterion,
            feature_index=int(feat_idx),
            audit=audit,
        )

        result: Dict[str, Any] = {
            "feat_idx": int(feat_idx),
            "summary": summary,
            "rejected": False,
            "skipped_unreliable": False,
            "downweighted": False,
            "stability_weight": None,
            "weight": 1.0,
            "train_mean": None,
            "train_std": None,
            "apply_reason": "pending",
            "fallback_meta": {},
            "train_z": None,
            "test_z": None,
        }

        if summary.rejected or summary.family is None or summary.params is None:
            result["rejected"] = True
            result["apply_reason"] = "rejected_or_missing_family"
            return result

        if not self._should_apply_cdf_transform(summary):
            result["skipped_unreliable"] = True
            result["apply_reason"] = "cdf_reliability_gate"
            return result

        dist_obj = self.dist_fitter._base_distributions.get(summary.family)
        if dist_obj is None:
            result["rejected"] = True
            result["apply_reason"] = "missing_distribution_object"
            return result

        train_g = self._cdf_gaussian_transform(train_col, dist_obj, summary.params)
        test_g = self._cdf_gaussian_transform(test_col, dist_obj, summary.params)

        weight = 1.0
        gof_p = self._combined_gof_p(summary)
        if self.config.low_gof_downweighting and gof_p < self.config.low_gof_threshold:
            weight *= float(self.config.low_gof_weight)
            result["downweighted"] = True

        if self.config.use_distribution_stability_weight:
            stability = self._family_stability_bootstrap(
                train_col,
                expected_family=summary.family,
                n_bootstrap=self.config.stability_bootstrap,
                seed=seed + int(feat_idx),
            )
            w_stability = 0.5 + 0.5 * stability
            weight *= w_stability
            result["stability_weight"] = w_stability

        mu = float(np.mean(train_g))
        sigma = float(np.std(train_g))
        if sigma < 1e-8:
            sigma = 1.0
        train_z = (train_g - mu) / sigma * weight
        test_z = (test_g - mu) / sigma * weight
        result["train_z"] = train_z
        result["test_z"] = test_z
        result["weight"] = float(weight)
        result["train_mean"] = float(mu)
        result["train_std"] = float(sigma)
        result["apply_reason"] = "ok"
        return result

    def _distribution_transform_block(
        self,
        X_train_imp: np.ndarray,
        X_test_imp: np.ndarray,
        X_train_base: np.ndarray,
        X_test_base: np.ndarray,
        y_train: np.ndarray,
        seed: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, List[DistributionFitSummary], Dict[str, Any]]:
        X_train_out = X_train_base.copy()
        X_test_out = X_test_base.copy()

        if not self.config.apply_cdf_transform:
            self._last_distribution_plan = {
                "schema_version": "1.0",
                "apply_cdf_transform": False,
                "n_input_features": int(X_train_imp.shape[1]) if X_train_imp.ndim == 2 else 0,
                "dist_feature_indices": [],
                "feature_plans": [],
            }
            return X_train_out, X_test_out, [], {
                "transform_time_sec": 0.0,
                "n_fitted": 0,
                "n_transformed": 0,
                "n_rejected": 0,
                "n_skipped_unreliable": 0,
                "n_skipped_block_cv": 0,
                "n_downweighted": 0,
                "mean_stability_weight": 1.0,
                "cdf_block_gating_time_sec": 0.0,
                "cdf_block_gating_budget_hit": 0.0,
                "cdf_block_gating_blocks_evaluated": 0,
                "cdf_block_gating_blocks_applied": 0,
            }

        n_features = X_train_imp.shape[1]
        if n_features == 0:
            self._last_distribution_plan = {
                "schema_version": "1.0",
                "apply_cdf_transform": True,
                "n_input_features": 0,
                "dist_feature_indices": [],
                "feature_plans": [],
            }
            return X_train_out, X_test_out, [], {
                "transform_time_sec": 0.0,
                "n_fitted": 0,
                "n_transformed": 0,
                "n_rejected": 0,
                "n_skipped_unreliable": 0,
                "n_skipped_block_cv": 0,
                "n_downweighted": 0,
                "mean_stability_weight": 1.0,
                "cdf_block_gating_time_sec": 0.0,
                "cdf_block_gating_budget_hit": 0.0,
                "cdf_block_gating_blocks_evaluated": 0,
                "cdf_block_gating_blocks_applied": 0,
            }

        max_dist = n_features if self.config.max_dist_features is None else int(max(1, self.config.max_dist_features))
        max_dist = int(min(max_dist, n_features))

        variances = np.nanvar(X_train_imp, axis=0)
        dist_feature_indices = np.argsort(variances)[::-1][:max_dist]

        transform_start = self._timer()
        summaries: List[DistributionFitSummary] = []
        summary_map: Dict[int, DistributionFitSummary] = {}
        n_rejected = 0
        n_transformed = 0
        n_skipped_unreliable = 0
        n_skipped_block_cv = 0
        n_downweighted = 0
        stability_weights: List[float] = []
        transformed_payloads: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        feature_plans: Dict[int, Dict[str, Any]] = {}

        # Determine effective parallelism for DF feature loop.
        _df_n_jobs = int(getattr(self.config, "n_jobs", 1) or 1)
        if _df_n_jobs == -1:
            import os as _os
            _df_n_jobs = _os.cpu_count() or 1

        fit_results: List[Dict[str, Any]]
        if _df_n_jobs > 1 and len(dist_feature_indices) > 1:
            # Parallel per-feature DF fitting (DF-1).
            try:
                from tabnetics.core.compat import Parallel, delayed
            except Exception as exc:
                from tabnetics.core.compat import Parallel, delayed  # type: ignore
            fit_results = Parallel(n_jobs=_df_n_jobs, prefer="threads")(
                delayed(self._fit_transform_one_feature)(
                    int(feat_idx), X_train_imp, X_test_imp, seed,
                )
                for feat_idx in dist_feature_indices
            )
        else:
            fit_results = [
                self._fit_transform_one_feature(int(feat_idx), X_train_imp, X_test_imp, seed)
                for feat_idx in dist_feature_indices
            ]

        for r in fit_results:
            summaries.append(r["summary"])
            summary_map[int(r["feat_idx"])] = r["summary"]
            feat_idx = int(r["feat_idx"])
            summary = r["summary"]
            feature_plans[feat_idx] = {
                "feature_index": int(feat_idx),
                "family": None if summary.family is None else str(summary.family),
                "params": (
                    None
                    if summary.params is None
                    else [float(v) for v in tuple(summary.params)]
                ),
                "rejected": bool(r.get("rejected", False)),
                "skipped_unreliable": bool(r.get("skipped_unreliable", False)),
                "downweighted": bool(r.get("downweighted", False)),
                "stability_weight": (
                    None
                    if r.get("stability_weight", None) is None
                    else float(r.get("stability_weight"))
                ),
                "weight": float(r.get("weight", 1.0) or 1.0),
                "train_mean": (
                    None
                    if r.get("train_mean", None) is None
                    else float(r.get("train_mean"))
                ),
                "train_std": (
                    None
                    if r.get("train_std", None) is None
                    else float(r.get("train_std"))
                ),
                "apply_reason": str(r.get("apply_reason", "")),
                "fallback_meta": dict(r.get("fallback_meta") or {}),
                "applied": False,
            }
            if r["rejected"]:
                n_rejected += 1
            elif r["skipped_unreliable"]:
                n_skipped_unreliable += 1
            else:
                if r["downweighted"]:
                    n_downweighted += 1
                if r["stability_weight"] is not None:
                    stability_weights.append(r["stability_weight"])
                if r["train_z"] is not None and r["test_z"] is not None:
                    transformed_payloads[int(r["feat_idx"])] = (r["train_z"], r["test_z"])

        apply_features: Set[int] = set(transformed_payloads.keys())
        block_stats: Dict[str, float] = {
            "n_blocks_evaluated": 0,
            "n_blocks_applied": 0,
            "n_features_skipped_cv": 0,
            "time_sec": 0.0,
            "budget_hit": 0.0,
        }
        if self.config.cdf_block_gating_cv and transformed_payloads:
            apply_features, block_stats = self._apply_cdf_block_gating_cv(
                X_train_base=X_train_base,
                y_train=y_train,
                transformed_payloads=transformed_payloads,
                summary_map=summary_map,
                seed=seed,
            )
            n_skipped_block_cv = int(block_stats["n_features_skipped_cv"])

        for feat_idx in sorted(apply_features):
            train_z, test_z = transformed_payloads[feat_idx]
            X_train_out[:, feat_idx] = train_z
            X_test_out[:, feat_idx] = test_z
            n_transformed += 1
            if feat_idx in feature_plans:
                feature_plans[int(feat_idx)]["applied"] = True
        for feat_idx, plan in feature_plans.items():
            if bool(plan.get("applied", False)):
                continue
            if bool(plan.get("rejected", False)) or bool(plan.get("skipped_unreliable", False)):
                continue
            reason = str(plan.get("apply_reason", "") or "")
            if reason == "ok":
                plan["apply_reason"] = "cdf_block_gating_cv"

        transform_time_sec = self._timer() - transform_start
        mean_stability_weight = float(np.mean(stability_weights)) if stability_weights else 1.0

        meta = {
            "transform_time_sec": float(transform_time_sec),
            "n_fitted": int(len(summaries)),
            "n_transformed": int(n_transformed),
            "n_rejected": int(n_rejected),
            "n_skipped_unreliable": int(n_skipped_unreliable),
            "n_skipped_block_cv": int(n_skipped_block_cv),
            "n_downweighted": int(n_downweighted),
            "mean_stability_weight": float(mean_stability_weight),
            "cdf_block_gating_time_sec": float(block_stats["time_sec"]),
            "cdf_block_gating_budget_hit": float(block_stats["budget_hit"]),
            "cdf_block_gating_blocks_evaluated": int(block_stats["n_blocks_evaluated"]),
            "cdf_block_gating_blocks_applied": int(block_stats["n_blocks_applied"]),
        }
        self._last_distribution_plan = {
            "schema_version": "1.0",
            "apply_cdf_transform": True,
            "n_input_features": int(n_features),
            "dist_feature_indices": [int(i) for i in np.asarray(dist_feature_indices, dtype=int).ravel().tolist()],
            "feature_plans": [
                dict(feature_plans[idx]) for idx in sorted(feature_plans.keys())
            ],
        }
        return X_train_out, X_test_out, summaries, meta

    @staticmethod
    def _combined_gof_p(summary: DistributionFitSummary) -> float:
        pvals: List[float] = []
        if np.isfinite(summary.cvm_p):
            pvals.append(float(summary.cvm_p))
        if np.isfinite(summary.ks_p):
            pvals.append(float(summary.ks_p))
        if not pvals:
            return 0.0
        return float(min(pvals))

    def _build_cdf_profile_blocks(
        self,
        feature_indices: Sequence[int],
        summary_map: Dict[int, DistributionFitSummary],
    ) -> List[List[int]]:
        features = [int(i) for i in feature_indices]
        if not features:
            return []

        # Sort by a compact GOF/profile key to group similar reliability behavior.
        def _profile_key(feat_idx: int) -> Tuple[float, float, float, int]:
            summary = summary_map.get(feat_idx)
            if summary is None:
                return (0.0, 0.0, 1.0, 0)
            return (
                float(self._combined_gof_p(summary)),
                -float(len(summary.confidence_set)),
                -float(summary.audit.outlier_fraction),
                int(bool(summary.audit.has_heaping)),
            )

        ordered = sorted(features, key=_profile_key, reverse=True)
        min_block_size = int(max(1, self.config.cdf_block_gating_min_block_size))
        n_blocks = int(max(1, self.config.cdf_block_gating_n_blocks))

        if len(ordered) < min_block_size:
            return [ordered]

        n_blocks = int(min(n_blocks, max(1, len(ordered) // min_block_size)))
        if n_blocks <= 1:
            return [ordered]

        blocks_raw = np.array_split(np.asarray(ordered, dtype=int), n_blocks)
        blocks = [list(map(int, b.tolist())) for b in blocks_raw if b.size > 0]
        return blocks if blocks else [ordered]

    def _block_cv_score(self, X_block: np.ndarray, y_train: np.ndarray, seed: int) -> float:
        y_arr = np.asarray(y_train)
        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) < 2 or counts.min() < 2:
            return float("nan")

        n_splits = int(max(2, min(int(self.config.cdf_block_gating_cv_splits), int(counts.min()))))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = make_logistic_regression(
            random_state=seed,
            max_iter=2000,
            solver="lbfgs",
            penalty="l2",
            class_weight="balanced",
        )
        try:
            scores = cross_val_score(model, X_block, y_arr, cv=cv, scoring="balanced_accuracy")
            return float(np.mean(scores))
        except Exception as exc:
            return float("nan")

    def _apply_cdf_block_gating_cv(
        self,
        X_train_base: np.ndarray,
        y_train: np.ndarray,
        transformed_payloads: Dict[int, Tuple[np.ndarray, np.ndarray]],
        summary_map: Dict[int, DistributionFitSummary],
        seed: int,
    ) -> Tuple[Set[int], Dict[str, float]]:
        feature_indices = sorted(int(i) for i in transformed_payloads.keys())
        if not feature_indices:
            return set(), {
                "n_blocks_evaluated": 0,
                "n_blocks_applied": 0,
                "n_features_skipped_cv": 0,
                "time_sec": 0.0,
                "budget_hit": 0.0,
            }

        blocks = self._build_cdf_profile_blocks(feature_indices, summary_map)
        if not blocks:
            return set(feature_indices), {
                "n_blocks_evaluated": 0,
                "n_blocks_applied": 0,
                "n_features_skipped_cv": 0,
                "time_sec": 0.0,
                "budget_hit": 0.0,
            }

        max_blocks = int(max(1, self.config.cdf_block_gating_max_blocks))
        budget_sec = float(max(0.0, self.config.cdf_block_gating_time_budget_sec))
        min_gain = float(self.config.cdf_block_gating_min_improvement)

        start = self._timer()
        selected: Set[int] = set()
        n_blocks_evaluated = 0
        n_blocks_applied = 0
        n_features_skipped = 0
        budget_hit = False

        for block_idx, block_feats in enumerate(blocks):
            if n_blocks_evaluated >= max_blocks:
                selected.update(block_feats)
                continue

            if (self._timer() - start) >= budget_sec:
                budget_hit = True
                selected.update(block_feats)
                continue

            X_base_block = X_train_base[:, block_feats]
            X_cdf_block = np.column_stack([transformed_payloads[int(i)][0] for i in block_feats])

            base_score = self._block_cv_score(X_base_block, y_train, seed=seed + block_idx * 17 + 1)
            cdf_score = self._block_cv_score(X_cdf_block, y_train, seed=seed + block_idx * 17 + 2)
            n_blocks_evaluated += 1

            # Fall back to per-feature gating whenever CV scoring is not reliable.
            if not (np.isfinite(base_score) and np.isfinite(cdf_score)):
                selected.update(block_feats)
                continue

            if cdf_score >= base_score + min_gain:
                selected.update(block_feats)
                n_blocks_applied += 1
            else:
                n_features_skipped += len(block_feats)

        stats = {
            "n_blocks_evaluated": int(n_blocks_evaluated),
            "n_blocks_applied": int(n_blocks_applied),
            "n_features_skipped_cv": int(n_features_skipped),
            "time_sec": float(self._timer() - start),
            "budget_hit": float(budget_hit),
        }
        return selected, stats

    def _should_apply_cdf_transform(self, summary: DistributionFitSummary) -> bool:
        if not self.config.cdf_reliability_gate:
            return True
        if self.config.cdf_skip_heaped_features and summary.audit.has_heaping:
            return False

        gof_p = self._combined_gof_p(summary)
        if gof_p < float(self.config.cdf_min_gof_p):
            return False

        cset_size = len(summary.confidence_set)
        if cset_size == 0:
            return False
        if cset_size > int(max(1, self.config.cdf_max_confidence_set)):
            return False
        return True

    def _rank_gaussian_transform_train_test(self, train_col: np.ndarray, test_col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rank-based inverse-normal transform using the train empirical CDF.

        This is a cheap fallback for small-n / low-unique features where
        parametric DF can be unstable or expensive.
        """
        train = np.asarray(train_col, dtype=float).ravel()
        test = np.asarray(test_col, dtype=float).ravel()
        n = int(train.size)
        if n <= 1:
            return np.zeros_like(train), np.zeros_like(test)

        ranks = sps.rankdata(train, method="average")
        eps = float(max(1e-8, 0.5 / float(n)))

        q_train = (np.asarray(ranks, dtype=float) - 0.5) / float(n)
        q_train = np.clip(q_train, eps, 1.0 - eps)
        train_g = sps.norm.ppf(q_train)

        train_sorted = np.sort(train)
        left = np.searchsorted(train_sorted, test, side="left")
        right = np.searchsorted(train_sorted, test, side="right")
        mid = (np.asarray(left, dtype=float) + np.asarray(right, dtype=float)) / 2.0
        q_test = mid / float(n)
        q_test = np.clip(q_test, eps, 1.0 - eps)
        test_g = sps.norm.ppf(q_test)

        train_g = np.nan_to_num(np.asarray(train_g, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        test_g = np.nan_to_num(np.asarray(test_g, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        return train_g, test_g

    @staticmethod
    def _mixture_cdf_1d(values: np.ndarray, *, means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float).ravel()
        cdf = np.zeros(x.shape[0], dtype=float)
        for idx in range(int(means.size)):
            mu = float(means[idx])
            sig = float(max(1e-9, stds[idx]))
            w = float(weights[idx])
            cdf += w * np.asarray(sps.norm.cdf(x, loc=mu, scale=sig), dtype=float)
        cdf = np.clip(np.nan_to_num(cdf, nan=0.5, posinf=1.0, neginf=0.0), 1e-8, 1.0 - 1e-8)
        return cdf

    def _gmm_gaussian_transform_train_test(
        self,
        train_col: np.ndarray,
        test_col: np.ndarray,
        *,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        train = np.asarray(train_col, dtype=float).ravel()
        test = np.asarray(test_col, dtype=float).ravel()
        x_train = train.reshape(-1, 1)
        best_model: Optional[GaussianMixture] = None
        best_bic = float("inf")
        bic_by_k: Dict[str, float] = {}
        fit_candidates: Dict[str, Dict[str, Any]] = {}
        for n_components in (2, 3):
            if int(train.size) <= int(n_components):
                continue
            model = GaussianMixture(
                n_components=int(n_components),
                covariance_type="full",
                random_state=int(seed) + int(n_components) * 97,
                reg_covar=1e-6,
                max_iter=300,
            )
            model.fit(x_train)
            bic = float(model.bic(x_train))
            aic = float(model.aic(x_train))
            bic_by_k[str(int(n_components))] = bic
            fit_candidates[str(int(n_components))] = {
                "bic": float(bic),
                "aic": float(aic),
                "converged": bool(getattr(model, "converged_", False)),
                "n_iter": int(getattr(model, "n_iter_", 0) or 0),
                "lower_bound": float(getattr(model, "lower_bound_", float("nan"))),
            }
            if np.isfinite(bic) and bic < best_bic:
                best_bic = bic
                best_model = model
        if best_model is None:
            raise RuntimeError("No valid GMM fit for multimodal fallback")

        means = np.asarray(best_model.means_, dtype=float).ravel()
        cov = np.asarray(best_model.covariances_, dtype=float)
        if cov.ndim == 3:
            vars_ = cov[:, 0, 0]
        elif cov.ndim == 2:
            vars_ = cov[:, 0]
        else:
            vars_ = cov.ravel()
        stds = np.sqrt(np.maximum(1e-12, np.asarray(vars_, dtype=float)))
        weights = np.asarray(best_model.weights_, dtype=float).ravel()
        total_w = float(np.sum(weights))
        if not np.isfinite(total_w) or total_w <= 0.0:
            weights = np.full(means.size, 1.0 / float(max(1, means.size)), dtype=float)
        else:
            weights = weights / total_w

        order = np.argsort(means)
        means = means[order]
        stds = stds[order]
        weights = weights[order]

        selected_aic = float(best_model.aic(x_train))
        selected_bic = float(best_model.bic(x_train))
        train_loglik_mean = float(best_model.score(x_train))
        train_loglik_total = float(train_loglik_mean * float(train.size))
        min_pairwise_separation_z = None
        if int(means.size) >= 2:
            min_sep = float("inf")
            for i in range(int(means.size)):
                for j in range(i + 1, int(means.size)):
                    denom = math.sqrt(float(stds[i] ** 2 + stds[j] ** 2))
                    sep = float(abs(means[i] - means[j]) / max(1e-9, denom))
                    if np.isfinite(sep):
                        min_sep = min(min_sep, sep)
            if np.isfinite(min_sep):
                min_pairwise_separation_z = float(min_sep)
        entropy = -float(np.sum(weights * np.log(np.maximum(1e-12, weights))))
        effective_components = float(np.exp(entropy))

        train_cdf = self._mixture_cdf_1d(train, means=means, stds=stds, weights=weights)
        test_cdf = self._mixture_cdf_1d(test, means=means, stds=stds, weights=weights)
        train_g = sps.norm.ppf(train_cdf)
        test_g = sps.norm.ppf(test_cdf)
        train_g = np.nan_to_num(np.asarray(train_g, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        test_g = np.nan_to_num(np.asarray(test_g, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        meta = {
            "fallback_mode": "gmm",
            "selected_components": int(means.size),
            "bic_by_components": dict(bic_by_k),
            "components": [
                {
                    "weight": float(weights[idx]),
                    "mean": float(means[idx]),
                    "std": float(stds[idx]),
                }
                for idx in range(int(means.size))
            ],
            "fit_diagnostics": {
                "criterion": "bic",
                "selected_components": int(means.size),
                "selected_bic": float(selected_bic),
                "selected_aic": float(selected_aic),
                "train_loglik_mean": float(train_loglik_mean),
                "train_loglik_total": float(train_loglik_total),
                "converged": bool(getattr(best_model, "converged_", False)),
                "n_iter": int(getattr(best_model, "n_iter_", 0) or 0),
                "lower_bound": float(getattr(best_model, "lower_bound_", float("nan"))),
                "min_component_weight": float(np.min(weights)),
                "max_component_weight": float(np.max(weights)),
                "effective_components": float(effective_components),
                "min_pairwise_separation_z": (
                    None if min_pairwise_separation_z is None else float(min_pairwise_separation_z)
                ),
                "candidate_metrics": dict(fit_candidates),
            },
        }
        return train_g, test_g, meta

    def _try_multimodal_fallback_transform(
        self,
        *,
        feat_idx: int,
        train_col: np.ndarray,
        test_col: np.ndarray,
        audit: DataAuditReport,
        seed: int,
        criterion: str,
    ) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self.config.dist_config, "compute_dip", False)):
            return None
        if not bool(getattr(audit, "is_multimodal", False)):
            return None

        mode = str(getattr(self.config, "multimodal_fallback", "none") or "none").strip().lower()
        if mode in {"rank", "rank_gaussian", "quantile"}:
            mode = "rank_transform"
        if mode not in {"gmm", "rank_transform"}:
            return None

        crit = str(criterion or "simple").strip().lower()
        mnpo_routed = bool(crit == "mnpo_oracle")
        candidates_post = int(len(self.dist_fitter.generate_candidates(audit)))
        summary = DistributionFitSummary(
            feature_index=int(feat_idx),
            family=f"multimodal_fallback_{mode}",
            params=None,
            cvm_p=float("nan"),
            ks_p=float("nan"),
            simple_score=float("nan"),
            confidence_set=tuple(),
            rejected=False,
            audit=audit,
            rejection_reason="multimodal_fallback",
            selected_family_support="real",
            candidates_pre_filter=int(len(self.dist_fitter._base_distributions)),
            candidates_post_filter=candidates_post,
            fit_method=("multimodal_gmm_bic" if mode == "gmm" else "multimodal_rank_transform"),
            mnpo_weight=(1.0 if mnpo_routed else None),
        )

        fallback_meta: Dict[str, Any] = {
            "fallback_mode": str(mode),
            "is_multimodal": bool(audit.is_multimodal),
            "mode_count": None if audit.mode_count is None else int(audit.mode_count),
        }
        try:
            if mode == "gmm":
                train_g, test_g, gmm_meta = self._gmm_gaussian_transform_train_test(
                    train_col,
                    test_col,
                    seed=int(seed) + int(feat_idx) * 4099,
                )
                fallback_meta.update(gmm_meta)
                fit_diag = dict(fallback_meta.get("fit_diagnostics") or {})
                bic_val = fit_diag.get("selected_bic")
                aic_val = fit_diag.get("selected_aic")
                loglik_total_val = fit_diag.get("train_loglik_total")
                if bic_val is not None and np.isfinite(float(bic_val)):
                    summary.bic = float(bic_val)
                if aic_val is not None and np.isfinite(float(aic_val)):
                    summary.aic = float(aic_val)
                if loglik_total_val is not None and np.isfinite(float(loglik_total_val)):
                    summary.loglik = float(loglik_total_val)
            else:
                train_g, test_g = self._rank_gaussian_transform_train_test(train_col, test_col)
        except Exception as exc:
            logger.warning(
                "Multimodal GMM fallback failed for feature %d (%s); using rank transform fallback.",
                int(feat_idx),
                exc,
            )
            train_g, test_g = self._rank_gaussian_transform_train_test(train_col, test_col)
            fallback_meta["fallback_mode"] = "rank_transform"
            fallback_meta["fallback_reason"] = "gmm_failure"
            summary.family = "multimodal_fallback_rank_transform"
            summary.fit_method = "multimodal_rank_transform"

        mu = float(np.mean(train_g))
        sigma = float(np.std(train_g))
        if sigma < 1e-8:
            sigma = 1.0
        train_z = (train_g - mu) / sigma
        test_z = (test_g - mu) / sigma
        fallback_meta["mnpo_context"] = {
            "criterion": str(crit),
            "included_in_mnpo": bool(mnpo_routed),
            "assigned_weight": (None if summary.mnpo_weight is None else float(summary.mnpo_weight)),
            "routing_reason": "multimodal_fallback_bypasses_unimodal_family_competition",
        }
        return {
            "summary": summary,
            "train_z": np.asarray(train_z, dtype=float),
            "test_z": np.asarray(test_z, dtype=float),
            "train_mean": float(mu),
            "train_std": float(sigma),
            "fallback_meta": fallback_meta,
        }

    def audit_distribution_summaries(self, summaries: Sequence[DistributionFitSummary]) -> Dict[str, Any]:
        """Summarize DF rejection and CDF reliability-gate behavior for a fitted feature batch.

        This is intended for empirical debugging of datasets with high DF rejection/skip rates
        (e.g., `lung_gordon`) without changing any production defaults.
        """
        rows = list(summaries or [])
        support_counts: Counter[str] = Counter()
        family_counts_all: Counter[str] = Counter()
        family_counts_non_rejected: Counter[str] = Counter()
        rejection_reasons: Counter[str] = Counter()
        cdf_skip_reasons: Counter[str] = Counter()

        n_heaped = 0
        n_non_rejected = 0
        n_cdf_should_apply = 0
        n_support_conflict_selected = 0
        n_true_gof_failure = 0
        n_multimodal_fallback = 0

        for summary in rows:
            support_counts[str(summary.audit.support.inferred_support)] += 1
            if bool(summary.audit.has_heaping):
                n_heaped += 1

            if summary.family is not None:
                family_counts_all[str(summary.family)] += 1

            conflict_selected = bool(
                str(summary.audit.support.inferred_support) == "real"
                and str(summary.selected_family_support or "") == "positive"
                and float(getattr(summary.audit, "frac_negative", 0.0) or 0.0) >= 0.05
            )
            if conflict_selected:
                n_support_conflict_selected += 1

            reason_raw = summary.rejection_reason
            reason = None if reason_raw is None else str(reason_raw)
            is_multimodal_fallback = bool(reason == "multimodal_fallback")
            if is_multimodal_fallback:
                n_multimodal_fallback += 1

            rejected = bool(
                (not is_multimodal_fallback)
                and (summary.rejected or summary.family is None or summary.params is None)
            )
            if rejected:
                if reason is None:
                    if summary.family is None or summary.params is None:
                        if int(summary.audit.n_clean) < 10:
                            reason = "insufficient_clean"
                        elif bool(summary.audit.support.is_near_constant):
                            reason = "near_constant"
                        else:
                            reason = "fit_failed"
                    else:
                        reason = "gof_reject"
                if reason == "gof_reject":
                    n_true_gof_failure += 1
                rejection_reasons[str(reason)] += 1
                continue

            n_non_rejected += 1
            if summary.family is not None:
                family_counts_non_rejected[str(summary.family)] += 1

            if is_multimodal_fallback:
                n_cdf_should_apply += 1
                continue

            skip_reason: Optional[str] = None
            if self.config.cdf_reliability_gate:
                if self.config.cdf_skip_heaped_features and summary.audit.has_heaping:
                    skip_reason = "heaping"
                else:
                    gof_p = self._combined_gof_p(summary)
                    if gof_p < float(self.config.cdf_min_gof_p):
                        skip_reason = "low_gof"
                    else:
                        cset_size = len(summary.confidence_set)
                        if cset_size == 0:
                            skip_reason = "confidence_empty"
                        elif cset_size > int(max(1, self.config.cdf_max_confidence_set)):
                            skip_reason = "confidence_too_large"

            if skip_reason is None:
                n_cdf_should_apply += 1
            else:
                cdf_skip_reasons[skip_reason] += 1

        out_rejection_reasons = dict(rejection_reasons)
        if n_multimodal_fallback > 0:
            out_rejection_reasons["multimodal_fallback"] = int(n_multimodal_fallback)

        return {
            "n_total": int(len(rows)),
            "n_rejected": int(sum(rejection_reasons.values())),
            "rejection_reasons": out_rejection_reasons,
            "support_counts": dict(support_counts),
            "n_heaped": int(n_heaped),
            "family_counts_all": dict(family_counts_all),
            "family_counts_non_rejected": dict(family_counts_non_rejected),
            "n_non_rejected": int(n_non_rejected),
            "cdf_gate_enabled": bool(self.config.cdf_reliability_gate),
            "cdf_skip_heaped_features": bool(self.config.cdf_skip_heaped_features),
            "cdf_min_gof_p": float(self.config.cdf_min_gof_p),
            "cdf_max_confidence_set": int(self.config.cdf_max_confidence_set),
            "n_cdf_should_apply": int(n_cdf_should_apply),
            "n_cdf_skipped": int(sum(cdf_skip_reasons.values())),
            "cdf_skip_reasons": dict(cdf_skip_reasons),
            "support_conflict_selected": int(n_support_conflict_selected),
            "true_gof_failure": int(n_true_gof_failure),
            "multimodal_fallback": int(n_multimodal_fallback),
        }

    def _cdf_gaussian_transform(
        self,
        data: np.ndarray,
        dist_obj: sps.rv_continuous,
        params: Sequence[float],
    ) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        cdf_vals = dist_obj.cdf(arr, *params)
        cdf_vals = np.clip(cdf_vals, 1e-8, 1 - 1e-8)
        return sps.norm.ppf(cdf_vals)

    def _family_stability_bootstrap(
        self,
        train_col: np.ndarray,
        expected_family: str,
        n_bootstrap: int,
        seed: int,
    ) -> float:
        clean = np.asarray(train_col, dtype=float)
        clean = clean[np.isfinite(clean)]
        if clean.size < 20:
            return 1.0

        rng = np.random.default_rng(seed)
        n_bootstrap = int(max(2, n_bootstrap))

        # Pre-generate all bootstrap samples for reproducibility.
        samples = []
        for _ in range(n_bootstrap):
            idx = rng.choice(np.arange(clean.size), size=clean.size, replace=True)
            samples.append(clean[idx])

        _n_jobs = int(getattr(self.config, "n_jobs", 1) or 1)
        if _n_jobs == -1:
            import os as _os
            _n_jobs = _os.cpu_count() or 1

        def _check_one(sample):
            summary = self.dist_fitter.select_best_distribution(
                sample, criterion=self.config.dist_criterion
            )
            return 1 if summary.family == expected_family else 0

        if _n_jobs > 1 and n_bootstrap > 2:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(_n_jobs, n_bootstrap)) as executor:
                hits = sum(executor.map(_check_one, samples))
        else:
            hits = sum(_check_one(s) for s in samples)

        return float(hits / n_bootstrap)

    @staticmethod
    def _base_dataset_name(dataset_name: str) -> str:
        return base_dataset_name(dataset_name)

    def _resolve_dataset_catalog_context(self, dataset_name: str) -> Dict[str, Any]:
        return resolve_dataset_catalog_context(dataset_name).as_dict()

    @staticmethod
    def _normalize_method_set(methods: Sequence[str]) -> Tuple[str, ...]:
        normalized: List[str] = []
        seen: Set[str] = set()
        for method in methods or ():
            key = str(method).strip()
            if not key or key in seen:
                continue
            normalized.append(key)
            seen.add(key)
        return tuple(normalized)

    @staticmethod
    def _meta_features_to_tier(meta: Dict[str, float]) -> str:
        """Map dataset meta-features to a coarse difficulty tier.

        Heuristic-only classifier used for optional routing/lockout controls.
        """
        p_over_n = float(meta.get("p_over_n", 0.0) or 0.0)
        class_count = float(meta.get("class_count", 2.0) or 2.0)
        entropy = float(meta.get("class_balance_entropy", 1.0) or 0.0)
        n_samples = float(meta.get("n", 0.0) or 0.0)

        if p_over_n >= 220.0 or class_count >= 8.0 or n_samples < 40.0:
            return "very_hard"
        if p_over_n >= 90.0 or class_count >= 5.0 or entropy < 0.45:
            return "hard"
        if p_over_n >= 30.0 or class_count >= 3.0 or entropy < 0.70:
            return "medium"
        return "easy"

    def _resolve_dataset_tier(
        self,
        dataset_name: str,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        source: str,
    ) -> Tuple[str, str, Dict[str, float]]:
        mode = str(source or "historical").strip().lower()
        if mode not in {"historical", "meta_features"}:
            mode = "historical"

        context = self._resolve_dataset_catalog_context(dataset_name)
        historical_tier = str(context.get("tier", "") or "").strip().lower()

        if mode == "historical" and historical_tier:
            return historical_tier, "historical", {}

        if mode == "meta_features":
            try:
                try:
                    from tabnetics.datasets.meta_features import extract_meta_features
                except Exception as exc:
                    from tabnetics.datasets.meta_features import extract_meta_features  # type: ignore
                meta = extract_meta_features(np.asarray(X_ref, dtype=float), np.asarray(y_ref))
                tier = self._meta_features_to_tier(meta)
                return str(tier), "meta_features", {str(k): float(v) for k, v in meta.items()}
            except Exception as exc:
                # Fall through to historical/default if extraction fails.
                pass

        if historical_tier:
            return historical_tier, "historical_fallback", {}
        return "unknown", "none", {}

    def _resolve_regime_policy(
        self,
        dataset_name: str,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        configured_methods: Tuple[str, ...],
    ) -> Dict[str, Any]:
        X_arr = np.asarray(X_ref, dtype=float)
        y_arr = np.asarray(y_ref).ravel()
        n_samples = int(X_arr.shape[0]) if X_arr.ndim == 2 else int(y_arr.size)
        n_features = int(X_arr.shape[1]) if X_arr.ndim == 2 and X_arr.size > 0 else 0
        class_count = int(np.unique(y_arr).size) if y_arr.size > 0 else 0
        p_over_n = float(n_features / max(1, n_samples))
        samples_per_class = float(n_samples / max(1, class_count if class_count > 0 else 1))

        source = str(
            getattr(self.config, "regime_gating_difficulty_source", "historical") or "historical"
        ).strip().lower()
        tier, tier_source, tier_meta = self._resolve_dataset_tier(
            dataset_name=dataset_name,
            X_ref=X_arr,
            y_ref=y_arr,
            source=source,
        )

        target_tier = str(
            getattr(self.config, "regime_gating_target_tier", "very_hard") or "very_hard"
        ).strip().lower()
        min_samples_per_class = float(
            getattr(self.config, "regime_gating_min_samples_per_class", 15.0) or 15.0
        )
        low_p_over_n_threshold = float(
            getattr(self.config, "regime_gating_low_p_over_n_threshold", 0.0) or 0.0
        )

        very_hard_trigger = bool(tier == target_tier) or bool(samples_per_class < min_samples_per_class)
        low_p_over_n_trigger = bool(low_p_over_n_threshold > 0.0) and bool(p_over_n < low_p_over_n_threshold)

        out: Dict[str, Any] = {
            "regime_policy_applied": False,
            "regime_policy_mode": "none",
            "regime_policy_reason": "none",
            "regime_policy_enabled": bool(getattr(self.config, "regime_gating_enabled", False)),
            "regime_policy_target_tier": target_tier,
            "regime_policy_tier": str(tier),
            "regime_policy_tier_source": str(tier_source),
            "regime_policy_tier_meta_features": dict(tier_meta or {}),
            "regime_policy_n_samples": int(n_samples),
            "regime_policy_n_features": int(n_features),
            "regime_policy_n_classes": int(class_count),
            "regime_policy_samples_per_class": float(samples_per_class),
            "regime_policy_p_over_n": float(p_over_n),
            "regime_policy_min_samples_per_class_threshold": float(min_samples_per_class),
            "regime_policy_low_p_over_n_threshold": float(low_p_over_n_threshold),
            "regime_policy_trigger_very_hard": bool(very_hard_trigger),
            "regime_policy_trigger_low_p_over_n": bool(low_p_over_n_trigger),
            "regime_policy_enabled_methods_before": list(configured_methods),
            "regime_policy_enabled_methods_after": list(configured_methods),
            "regime_policy_enabled_methods": tuple(configured_methods),
            "regime_policy_enabled_methods_source": "config",
            "regime_policy_bypass_fs": False,
            "regime_policy_bypass_mode": "",
            "regime_policy_selector_overrides": {},
        }

        if not bool(getattr(self.config, "regime_gating_enabled", False)):
            return out

        # Gate 2 (T-R-259) takes precedence to avoid unnecessary FS overhead when n >> p.
        if low_p_over_n_trigger:
            bypass_mode = str(
                getattr(self.config, "regime_gating_low_p_over_n_mode", "fast_univariate_filter") or "fast_univariate_filter"
            ).strip().lower()
            out.update(
                {
                    "regime_policy_applied": True,
                    "regime_policy_mode": "low_p_over_n_bypass",
                    "regime_policy_reason": f"p_over_n<{low_p_over_n_threshold:g}",
                    "regime_policy_enabled_methods_source": "regime_gate:low_p_over_n",
                    "regime_policy_bypass_fs": True,
                    "regime_policy_bypass_mode": str(bypass_mode),
                    "regime_policy_low_p_over_n_filter_max_k": int(
                        getattr(self.config, "regime_gating_low_p_over_n_filter_max_k", 200) or 200
                    ),
                }
            )
            return out

        # Gate 1/3 (T-R-257/T-R-258): very-hard and low-samples-per-class safeguards.
        # T-R-270: Gate 1 class-count qualifier — only trigger when c >= min_classes.
        min_classes_for_gate1 = int(
            max(0, int(getattr(self.config, "regime_gating_very_hard_min_classes", 5) or 5))
        )
        if min_classes_for_gate1 > 0 and class_count < min_classes_for_gate1:
            very_hard_trigger = False
            out["regime_policy_trigger_very_hard"] = False
            out["regime_policy_reason"] = f"class_count({class_count})<min_classes({min_classes_for_gate1})"
        if very_hard_trigger:
            simple_methods = self._normalize_method_set(
                getattr(self.config, "regime_gating_simple_methods", tuple()) or tuple()
            )
            if not simple_methods:
                simple_methods = configured_methods
            very_hard_cap = int(
                max(1, int(getattr(self.config, "regime_gating_very_hard_portfolio_max_methods", 4) or 4))
            )
            very_hard_cap = int(min(very_hard_cap, max(1, len(simple_methods))))
            adaptive_min_cfg = getattr(self.config, "fs_adaptive_size_min", None)
            adaptive_min = int(max(1, very_hard_cap))
            if adaptive_min_cfg is not None:
                adaptive_min = int(min(very_hard_cap, max(1, int(adaptive_min_cfg))))
            selector_overrides: Dict[str, Any] = {
                "selection_strategy": "legacy_voting",
                "fs_portfolio_size": int(very_hard_cap),
                "fs_adaptive_portfolio_sizing_enabled": True,
                "fs_adaptive_size_min": int(adaptive_min),
                "fs_adaptive_size_max": int(very_hard_cap),
                "fs_copula_derandomize_runs": int(
                    max(
                        1,
                        int(getattr(self.config, "regime_gating_very_hard_copula_derandomize_runs", 5) or 5),
                    )
                ),
            }
            out.update(
                {
                    "regime_policy_applied": True,
                    "regime_policy_mode": "very_hard_fallback",
                    "regime_policy_reason": "tier_or_low_samples_per_class",
                    "regime_policy_enabled_methods_after": list(simple_methods),
                    "regime_policy_enabled_methods": tuple(simple_methods),
                    "regime_policy_enabled_methods_source": "regime_gate:very_hard",
                    "regime_policy_selector_overrides": dict(selector_overrides),
                }
            )
            return out

        # Gate 3 (T-R-268): extreme multiclass classifier recovery gate.
        # When c >= threshold AND n/c >= guard: keep MNPO FS but request
        # OVA ensemble classifier mode for better multi-class decomposition.
        extreme_mc_enabled = bool(getattr(self.config, "regime_gating_extreme_multiclass_enabled", True))
        extreme_mc_threshold = int(
            getattr(self.config, "regime_gating_extreme_multiclass_threshold", 8) or 8
        )
        extreme_mc_min_spc = float(
            getattr(self.config, "regime_gating_extreme_multiclass_min_samples_per_class", 11.0) or 11.0
        )
        if (
            extreme_mc_enabled
            and class_count >= extreme_mc_threshold
            and samples_per_class >= extreme_mc_min_spc
        ):
            multiclass_overrides: Dict[str, Any] = {
                "classification_selection_mode": "mnpo_hybrid",
                "include_vote_ensemble_model": True,
            }
            out.update(
                {
                    "regime_policy_applied": True,
                    "regime_policy_mode": "extreme_multiclass_recovery",
                    "regime_policy_reason": (
                        f"class_count({class_count})>={extreme_mc_threshold} "
                        f"and spc({samples_per_class:.1f})>={extreme_mc_min_spc:.1f}"
                    ),
                    "regime_policy_enabled_methods_source": "regime_gate:extreme_multiclass",
                    "regime_policy_selector_overrides": dict(multiclass_overrides),
                    "regime_policy_bypass_fs": False,
                }
            )
            return out

        return out

    def _resolve_method_policy(
        self,
        dataset_name: str,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
    ) -> Dict[str, Any]:
        configured = self._normalize_method_set(self.config.enabled_methods)
        if not configured:
            configured = tuple()

        policy: Dict[str, Any] = {
            "enabled_methods": configured,
            "enabled_methods_source": "config",
            "tier_policy_applied": False,
            "tier_policy_mode": "none",
            "tier_policy_target_tier": "",
            "tier_policy_resolved_tier": "unknown",
            "tier_policy_source": "none",
            "tier_policy_meta_features": {},
            "tier_policy_enabled_methods_before": list(configured),
            "tier_policy_enabled_methods_after": list(configured),
            "regime_policy_applied": False,
            "regime_policy_mode": "none",
            "regime_policy_reason": "none",
            "regime_policy_enabled": bool(getattr(self.config, "regime_gating_enabled", False)),
            "regime_policy_target_tier": str(
                getattr(self.config, "regime_gating_target_tier", "very_hard") or "very_hard"
            ).strip().lower(),
            "regime_policy_tier": "unknown",
            "regime_policy_tier_source": "none",
            "regime_policy_tier_meta_features": {},
            "regime_policy_n_samples": int(np.asarray(X_ref).shape[0]) if np.asarray(X_ref).ndim == 2 else int(np.asarray(y_ref).size),
            "regime_policy_n_features": int(np.asarray(X_ref).shape[1]) if np.asarray(X_ref).ndim == 2 and np.asarray(X_ref).size > 0 else 0,
            "regime_policy_n_classes": int(np.unique(np.asarray(y_ref).ravel()).size) if np.asarray(y_ref).size > 0 else 0,
            "regime_policy_samples_per_class": 0.0,
            "regime_policy_p_over_n": 0.0,
            "regime_policy_min_samples_per_class_threshold": float(
                getattr(self.config, "regime_gating_min_samples_per_class", 15.0) or 15.0
            ),
            "regime_policy_low_p_over_n_threshold": float(
                getattr(self.config, "regime_gating_low_p_over_n_threshold", 0.0) or 0.0
            ),
            "regime_policy_trigger_very_hard": False,
            "regime_policy_trigger_low_p_over_n": False,
            "regime_policy_enabled_methods_before": list(configured),
            "regime_policy_enabled_methods_after": list(configured),
            "regime_policy_enabled_methods_source": "config",
            "regime_policy_bypass_fs": False,
            "regime_policy_bypass_mode": "",
            "regime_policy_selector_overrides": {},
        }

        regime = self._resolve_regime_policy(
            dataset_name=dataset_name,
            X_ref=X_ref,
            y_ref=y_ref,
            configured_methods=configured,
        )
        for key, value in regime.items():
            policy[key] = value
        if bool(regime.get("regime_policy_applied", False)):
            methods = tuple(regime.get("regime_policy_enabled_methods") or configured)
            policy["enabled_methods"] = methods
            policy["enabled_methods_source"] = str(
                regime.get("regime_policy_enabled_methods_source", "regime_gate")
            )
            policy["regime_policy_enabled_methods_after"] = list(methods)
            return policy

        lockout_enabled = bool(getattr(self.config, "tier_lockout_enabled", False))
        routing_enabled = bool(getattr(self.config, "tier_routing_enabled", False))
        if not lockout_enabled and not routing_enabled:
            return policy

        # Resolve tier for lockout/routing independently so each policy can use
        # its own source setting.
        lock_tier, lock_source, lock_meta = self._resolve_dataset_tier(
            dataset_name=dataset_name,
            X_ref=X_ref,
            y_ref=y_ref,
            source=str(getattr(self.config, "tier_lockout_difficulty_source", "historical") or "historical"),
        )
        route_tier, route_source, route_meta = self._resolve_dataset_tier(
            dataset_name=dataset_name,
            X_ref=X_ref,
            y_ref=y_ref,
            source=str(getattr(self.config, "tier_routing_difficulty_classifier", "meta_features") or "meta_features"),
        )

        # Lockout has precedence over routing when both are enabled.
        if lockout_enabled:
            target_tier = str(getattr(self.config, "tier_lockout_tier", "easy") or "easy").strip().lower()
            if lock_tier == target_tier:
                fallback = self._normalize_method_set(
                    getattr(self.config, "tier_lockout_fallback_methods", tuple()) or tuple()
                )
                if not fallback:
                    fallback = configured
                policy.update(
                    {
                        "enabled_methods": fallback,
                        "enabled_methods_source": f"tier_lockout:{target_tier}",
                        "tier_policy_applied": True,
                        "tier_policy_mode": "lockout",
                        "tier_policy_target_tier": target_tier,
                        "tier_policy_resolved_tier": lock_tier,
                        "tier_policy_source": lock_source,
                        "tier_policy_meta_features": lock_meta,
                        "tier_policy_enabled_methods_after": list(fallback),
                    }
                )
                return policy

        if routing_enabled:
            table = getattr(self.config, "tier_routing_table", {}) or {}
            routed_raw = None
            if isinstance(table, dict):
                routed_raw = table.get(route_tier)
            routed_methods = self._normalize_method_set(routed_raw or tuple())
            if routed_methods:
                policy.update(
                    {
                        "enabled_methods": routed_methods,
                        "enabled_methods_source": f"tier_routing:{route_tier}",
                        "tier_policy_applied": True,
                        "tier_policy_mode": "routing",
                        "tier_policy_target_tier": route_tier,
                        "tier_policy_resolved_tier": route_tier,
                        "tier_policy_source": route_source,
                        "tier_policy_meta_features": route_meta,
                        "tier_policy_enabled_methods_after": list(routed_methods),
                    }
                )
                return policy

        # No policy applied but expose the most specific resolved tier context.
        if lockout_enabled:
            policy["tier_policy_resolved_tier"] = lock_tier
            policy["tier_policy_source"] = lock_source
            policy["tier_policy_meta_features"] = lock_meta
        elif routing_enabled:
            policy["tier_policy_resolved_tier"] = route_tier
            policy["tier_policy_source"] = route_source
            policy["tier_policy_meta_features"] = route_meta
        return policy

    def _maybe_apply_face_domain_projection(
        self,
        X_train_imp: np.ndarray,
        y_train: np.ndarray,
        X_test_imp: np.ndarray,
        dataset_name: str,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        context = resolve_dataset_catalog_context(dataset_name)
        result = apply_face_domain_projection(
            X_train_imp=X_train_imp,
            y_train=y_train,
            X_test_imp=X_test_imp,
            enabled=bool(getattr(self.config, "enable_face_domain_projection", False)),
            dataset_name=str(dataset_name),
            dataset_context=context,
            seed=seed,
        )
        self._last_face_projection_state = {
            "pca_model": result.state.pca_model,
            "lda_model": result.state.lda_model,
            "dataset_name": str(result.state.dataset_name),
            "seed": int(result.state.seed),
        }
        return result.X_train, result.X_test, result.meta

    def _ratio_feature_generation(
        self,
        X_train_imp: np.ndarray,
        y_train: np.ndarray,
        X_test_imp: np.ndarray,
        seed: int,
        face_projection_applied: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Optional pre-DF log-ratio feature construction stage (RP-1).

        Generates up to `max_ratio_features` log-ratio features from a small
        screened pool of columns, then appends them to the design matrix.
        """
        method = str(getattr(self.config, "ratio_selection_method", "ktsp") or "ktsp").strip().lower()
        if method not in {"ktsp", "correlation"}:
            method = "ktsp"

        meta: Dict[str, Any] = {
            "enable_ratio_features": bool(getattr(self.config, "enable_ratio_features", False)),
            "ratio_features_applied": False,
            "ratio_features_reason": "disabled",
            "ratio_selection_method": str(method),
            "ratio_pool_size": int(max(0, getattr(self.config, "ratio_pool_size", 0) or 0)),
            "ratio_pool_size_effective": 0,
            "ratio_max_pairs": int(max(0, getattr(self.config, "ratio_max_pairs", 0) or 0)),
            "max_ratio_features": int(max(0, getattr(self.config, "max_ratio_features", 0) or 0)),
            "ratio_epsilon": float(getattr(self.config, "ratio_epsilon", 1e-6) or 1e-6),
            "ratio_include_originals": bool(getattr(self.config, "ratio_include_originals", True)),
            "ratio_abs_value": bool(getattr(self.config, "ratio_abs_value", False)),
            "ratio_require_positive": bool(getattr(self.config, "ratio_require_positive", True)),
            "ratio_features_added": 0,
            "ratio_feature_start_index": int(X_train_imp.shape[1]) if X_train_imp.ndim == 2 else 0,
            "ratio_pairs": [],
        }

        if not bool(getattr(self.config, "enable_ratio_features", False)):
            return X_train_imp, X_test_imp, meta
        if bool(face_projection_applied):
            meta["ratio_features_reason"] = "skipped_due_to_face_projection"
            return X_train_imp, X_test_imp, meta

        X_train_arr = np.asarray(X_train_imp, dtype=float)
        X_test_arr = np.asarray(X_test_imp, dtype=float)
        y_arr = np.asarray(y_train)
        if X_train_arr.ndim != 2 or X_test_arr.ndim != 2:
            meta["ratio_features_reason"] = "invalid_input_rank"
            return X_train_imp, X_test_imp, meta
        if X_train_arr.shape[1] != X_test_arr.shape[1]:
            meta["ratio_features_reason"] = "feature_dim_mismatch"
            return X_train_imp, X_test_imp, meta

        n_features = int(X_train_arr.shape[1])
        if n_features < 2:
            meta["ratio_features_reason"] = "insufficient_features"
            return X_train_imp, X_test_imp, meta

        pool_size = int(max(2, min(n_features, int(meta["ratio_pool_size"]))))
        max_pairs = int(max(1, int(meta["ratio_max_pairs"]))) if int(meta["ratio_max_pairs"]) > 0 else 0
        max_ratio = int(max(0, int(meta["max_ratio_features"])))
        if max_ratio <= 0:
            meta["ratio_features_reason"] = "max_ratio_features_zero"
            return X_train_imp, X_test_imp, meta

        eps = float(max(1e-12, float(meta["ratio_epsilon"])))
        require_positive = bool(meta["ratio_require_positive"])

        # Screen a small candidate pool using the same cheap rankers as the
        # downstream rank-prefilter (MI + ANOVA-F + linear SVM).
        mi = self._safe_mi(X_train_arr, y_arr, seed)
        fs = self._safe_fscore(X_train_arr, y_arr)
        svm = self._safe_linear_svm_scores(X_train_arr, y_arr, seed)
        combined = 0.45 * self._normalize01(mi) + 0.35 * self._normalize01(fs) + 0.20 * self._normalize01(svm)
        order = np.argsort(combined)[::-1]
        pool_idx = np.asarray(order[:pool_size], dtype=int).ravel()
        pool_idx = np.array(sorted(set(int(i) for i in pool_idx if 0 <= int(i) < n_features)), dtype=int)

        if require_positive:
            mins = np.nanmin(X_train_arr[:, pool_idx], axis=0) if pool_idx.size else np.array([], dtype=float)
            pos_mask = np.isfinite(mins) & (mins > 0.0)
            pool_idx = pool_idx[pos_mask] if pos_mask.size else pool_idx

        meta["ratio_pool_size_effective"] = int(pool_idx.size)
        if pool_idx.size < 2:
            meta["ratio_features_reason"] = "insufficient_positive_pool" if require_positive else "insufficient_pool"
            return X_train_imp, X_test_imp, meta

        rng = np.random.default_rng(int(seed) + 15173)
        pair_candidates = list(combinations(range(pool_idx.size), 2))
        if max_pairs > 0 and len(pair_candidates) > max_pairs:
            sampled = rng.choice(len(pair_candidates), size=max_pairs, replace=False)
            pair_candidates = [pair_candidates[int(i)] for i in sampled]

        feature_pair_scores: List[Tuple[float, int, int]] = []
        classes = np.unique(y_arr)

        if method == "correlation":
            # Correlation-guided pairing (unsupervised heuristic): favor moderately-to-strongly
            # correlated pairs so ratios can remove shared components.
            X_pool = X_train_arr[:, pool_idx]
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(X_pool, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            for a_local, b_local in pair_candidates:
                a = int(pool_idx[a_local])
                b = int(pool_idx[b_local])
                score = float(abs(corr[a_local, b_local]))
                feature_pair_scores.append((score, a, b))
        else:
            # k-TSP-style reversal-gap scoring (supervised, works for binary and multiclass).
            if classes.size < 2:
                meta["ratio_features_reason"] = "insufficient_classes"
                return X_train_imp, X_test_imp, meta

            class_masks = {cls: (y_arr == cls) for cls in classes}
            for a_local, b_local in pair_candidates:
                a = int(pool_idx[a_local])
                b = int(pool_idx[b_local])
                cmp_vec = X_train_arr[:, a] > X_train_arr[:, b]
                class_probs = []
                for cls in classes:
                    mask = class_masks[cls]
                    if np.sum(mask) == 0:
                        class_probs.append(0.5)
                    else:
                        class_probs.append(float(np.mean(cmp_vec[mask])))
                best_gap = 0.0
                for i, j in combinations(range(len(class_probs)), 2):
                    gap = abs(class_probs[i] - class_probs[j])
                    if gap > best_gap:
                        best_gap = gap
                feature_pair_scores.append((float(best_gap), a, b))

        if not feature_pair_scores:
            meta["ratio_features_reason"] = "no_pairs_scored"
            return X_train_imp, X_test_imp, meta

        feature_pair_scores.sort(key=lambda row: row[0], reverse=True)
        selected_pairs = feature_pair_scores[: min(max_ratio, len(feature_pair_scores))]
        if not selected_pairs:
            meta["ratio_features_reason"] = "no_pairs_selected"
            return X_train_imp, X_test_imp, meta

        ratio_cols_train: List[np.ndarray] = []
        ratio_cols_test: List[np.ndarray] = []
        ratio_pairs_meta: List[Dict[str, Any]] = []

        for score, a, b in selected_pairs:
            num_train = X_train_arr[:, int(a)]
            den_train = X_train_arr[:, int(b)]
            num_test = X_test_arr[:, int(a)]
            den_test = X_test_arr[:, int(b)]
            with np.errstate(divide="ignore", invalid="ignore"):
                r_train = np.log((num_train + eps) / (den_train + eps))
                r_test = np.log((num_test + eps) / (den_test + eps))
            if bool(meta["ratio_abs_value"]):
                r_train = np.abs(r_train)
                r_test = np.abs(r_test)
            if not (np.all(np.isfinite(r_train)) and np.all(np.isfinite(r_test))):
                continue
            ratio_cols_train.append(np.asarray(r_train, dtype=float).reshape(-1, 1))
            ratio_cols_test.append(np.asarray(r_test, dtype=float).reshape(-1, 1))
            ratio_pairs_meta.append({"numerator": int(a), "denominator": int(b), "score": float(score)})

        if not ratio_cols_train:
            meta["ratio_features_reason"] = "nonfinite_ratio_features"
            return X_train_imp, X_test_imp, meta

        X_ratio_train = np.hstack(ratio_cols_train)
        X_ratio_test = np.hstack(ratio_cols_test)
        if bool(meta["ratio_include_originals"]):
            X_train_out = np.hstack([X_train_arr, X_ratio_train])
            X_test_out = np.hstack([X_test_arr, X_ratio_test])
            meta["ratio_feature_start_index"] = int(X_train_arr.shape[1])
        else:
            X_train_out = X_ratio_train
            X_test_out = X_ratio_test
            meta["ratio_feature_start_index"] = 0

        meta["ratio_features_applied"] = True
        meta["ratio_features_reason"] = "ok"
        meta["ratio_features_added"] = int(X_ratio_train.shape[1])
        meta["ratio_pairs"] = ratio_pairs_meta
        return X_train_out, X_test_out, meta

    def _apply_folding_stage(
        self,
        X_train_fs_input: np.ndarray,
        X_test_fs_input: np.ndarray,
        y_train: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        method = str(getattr(self.config, "folding_method", "pls_da") or "pls_da").strip().lower()
        if method not in {"none", "rff", "tensor_sketch", "pls_da"}:
            method = "none"
        n_in = int(X_train_fs_input.shape[1]) if X_train_fs_input.ndim == 2 else 0
        folding_gamma_raw = getattr(self.config, "folding_rff_gamma", None)
        folding_gamma = None if folding_gamma_raw is None else float(folding_gamma_raw)
        meta: Dict[str, Any] = {
            "folding_method": method,
            "folding_applied": False,
            "folding_reason": "disabled" if method == "none" else "pending",
            "folding_input_dim": int(n_in),
            "folding_output_dim": int(n_in),
            "folding_n_components": int(max(2, int(getattr(self.config, "folding_n_components", 512) or 512))),
            "folding_rff_gamma": folding_gamma,
            "folding_pls_components_requested": int(max(2, int(getattr(self.config, "folding_pls_components", 32) or 32))),
            "folding_pls_components_used": 0,
            "folding_pls_scale": bool(getattr(self.config, "folding_pls_scale", True)),
            "folding_pls_min_classes": int(getattr(self.config, "folding_pls_min_classes", 5) or 5),
            "folding_pls_min_n_per_class": int(
                getattr(self.config, "folding_pls_min_n_per_class", 3) or 3
            ),
            "folding_pls_max_imbalance_ratio": float(
                getattr(self.config, "folding_pls_max_imbalance_ratio", 6.0) or 6.0
            ),
            "folding_prefilter_k": getattr(self.config, "folding_prefilter_k", None),
            "folding_standardize_applied": False,
            "folding_standardize_constant_dims": 0,
            "folding_standardize_min_train_std": float("nan"),
        }
        state: Dict[str, Any] = {
            "transformer": None,
            "standardize_mean": None,
            "standardize_scale": None,
            "seed": int(seed),
            "method": str(method),
        }
        self._last_folding_state = dict(state)
        if method == "none":
            return X_train_fs_input, X_test_fs_input, meta
        if n_in <= 0:
            meta["folding_reason"] = "empty_input"
            return X_train_fs_input, X_test_fs_input, meta

        n_components = int(max(2, int(getattr(self.config, "folding_n_components", 512) or 512)))
        try:
            if method == "rff":
                gamma_raw = getattr(self.config, "folding_rff_gamma", None)
                gamma = None if gamma_raw is None else float(gamma_raw)
                if gamma is None or (not np.isfinite(gamma)) or gamma <= 0.0:
                    gamma = 1.0 / max(1.0, float(n_in))
                folding = RBFSampler(
                    gamma=gamma,
                    n_components=n_components,
                    random_state=int(seed),
                )
                X_train_out = np.asarray(folding.fit_transform(X_train_fs_input), dtype=float)
                X_test_out = np.asarray(folding.transform(X_test_fs_input), dtype=float)
                meta["folding_rff_gamma"] = float(gamma)
                state["transformer"] = folding
            elif method == "pls_da":
                y_arr = np.asarray(y_train)
                classes = np.unique(y_arr)
                can_use_pls, guard_reason, guard_meta = self._can_use_pls_da(y_arr)
                meta.update(guard_meta)
                if not can_use_pls:
                    meta["folding_reason"] = f"pls_da_{guard_reason}"
                    return X_train_fs_input, X_test_fs_input, meta

                requested_components = int(max(2, int(getattr(self.config, "folding_pls_components", 32) or 32)))
                max_allowed = int(
                    max(
                        1,
                        min(
                            requested_components,
                            n_in,
                            max(1, int(X_train_fs_input.shape[0]) - 1),
                            max(1, int(classes.size) - 1),
                        ),
                    )
                )
                _, y_codes = np.unique(y_arr, return_inverse=True)
                y_codes = np.asarray(y_codes, dtype=int)
                y_onehot = np.eye(classes.size, dtype=float)[y_codes]

                pls = PLSRegression(
                    n_components=max_allowed,
                    scale=bool(getattr(self.config, "folding_pls_scale", True)),
                    max_iter=500,
                    tol=1e-06,
                )
                pls.fit(X_train_fs_input, y_onehot)
                X_train_out = np.asarray(pls.transform(X_train_fs_input), dtype=float)
                X_test_out = np.asarray(pls.transform(X_test_fs_input), dtype=float)
                state["transformer"] = pls
                meta["folding_pls_components_used"] = int(
                    X_train_out.shape[1] if X_train_out.ndim == 2 else 0
                )
                if int(meta["folding_pls_components_used"]) <= 0:
                    meta["folding_reason"] = "pls_da_empty_projection"
                    return X_train_fs_input, X_test_fs_input, meta
            else:
                folding = PolynomialCountSketch(
                    degree=2,
                    n_components=n_components,
                    random_state=int(seed),
                )
                X_train_out = np.asarray(folding.fit_transform(X_train_fs_input), dtype=float)
                X_test_out = np.asarray(folding.transform(X_test_fs_input), dtype=float)
                state["transformer"] = folding

            # Folded feature maps (especially RFF) can have globally low raw variance by
            # construction, which can trip downstream variance-threshold gates. Normalize
            # with train-only statistics to keep the folded representation usable.
            if X_train_out.ndim == 2 and X_train_out.shape[1] > 0:
                train_mean = np.asarray(np.mean(X_train_out, axis=0), dtype=float)
                train_std = np.asarray(np.std(X_train_out, axis=0), dtype=float)
                finite_std = np.where(np.isfinite(train_std), train_std, 0.0)
                safe_scale = np.where(finite_std > 1e-12, finite_std, 1.0)
                X_train_out = (X_train_out - train_mean) / safe_scale
                X_test_out = (X_test_out - train_mean) / safe_scale
                meta["folding_standardize_applied"] = True
                meta["folding_standardize_constant_dims"] = int(np.sum(finite_std <= 1e-12))
                meta["folding_standardize_min_train_std"] = float(np.nanmin(finite_std)) if finite_std.size else float("nan")
                state["standardize_mean"] = np.asarray(train_mean, dtype=float).ravel()
                state["standardize_scale"] = np.asarray(safe_scale, dtype=float).ravel()

            meta["folding_applied"] = True
            meta["folding_reason"] = "ok"
            meta["folding_output_dim"] = int(X_train_out.shape[1]) if X_train_out.ndim == 2 else 0
            meta["folding_n_components"] = int(meta["folding_output_dim"])
            self._last_folding_state = dict(state)
            return X_train_out, X_test_out, meta
        except Exception as exc:
            # FIX HIGH-003: Log exception instead of silent fallback (T-AUDIT-001-FIX-004)
            logger.warning(f"Folding transform failed with {type(exc).__name__}: {exc}; falling back to original features")
            meta["folding_reason"] = f"failed:{type(exc).__name__}"
            self._last_folding_state = dict(state)
            return X_train_fs_input, X_test_fs_input, meta

    def _can_use_pls_da(self, y_train: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
        y_arr = np.asarray(y_train).ravel()
        classes, counts = np.unique(y_arr, return_counts=True)
        n_classes = int(classes.size)
        min_count = int(np.min(counts)) if counts.size > 0 else 0
        max_count = int(np.max(counts)) if counts.size > 0 else 0
        imbalance_ratio = float(max_count / max(1, min_count)) if counts.size > 0 else float("inf")

        min_classes = int(getattr(self.config, "folding_pls_min_classes", 5) or 5)
        min_n_per_class = int(getattr(self.config, "folding_pls_min_n_per_class", 3) or 3)
        max_imbalance_ratio = float(
            getattr(self.config, "folding_pls_max_imbalance_ratio", 6.0) or 6.0
        )

        meta = {
            "folding_pls_n_classes": int(n_classes),
            "folding_pls_min_class_count": int(min_count),
            "folding_pls_max_class_count": int(max_count),
            "folding_pls_observed_imbalance_ratio": float(imbalance_ratio),
            "folding_pls_min_classes": int(min_classes),
            "folding_pls_min_n_per_class": int(min_n_per_class),
            "folding_pls_max_imbalance_ratio": float(max_imbalance_ratio),
        }

        if n_classes < min_classes:
            return False, "insufficient_classes", meta
        if min_count < min_n_per_class:
            return False, "insufficient_per_class", meta
        if imbalance_ratio > max_imbalance_ratio:
            return False, "class_imbalance", meta
        return True, "ok", meta

    def _rank_prefilter(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        seed: int,
        top_k_override: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.config.use_rank_prefilter:
            idx = np.arange(X_train.shape[1], dtype=int)
            return X_train, X_test, idx

        n_features = X_train.shape[1]
        if n_features <= 1:
            idx = np.arange(n_features, dtype=int)
            return X_train, X_test, idx

        if top_k_override is not None:
            top_k = int(max(1, top_k_override))
        else:
            top_k = n_features if self.config.prefilter_top_k is None else int(max(1, self.config.prefilter_top_k))
        if top_k >= n_features:
            idx = np.arange(n_features, dtype=int)
            return X_train, X_test, idx

        mi = self._safe_mi(X_train, y_train, seed)
        fs = self._safe_fscore(X_train, y_train)
        svm = self._safe_linear_svm_scores(X_train, y_train, seed)

        combined = 0.45 * self._normalize01(mi) + 0.35 * self._normalize01(fs) + 0.20 * self._normalize01(svm)
        keep_idx = np.argsort(combined)[::-1][:top_k]
        keep_idx = np.array(sorted(set(int(i) for i in keep_idx)), dtype=int)

        return X_train[:, keep_idx], X_test[:, keep_idx], keep_idx

    @staticmethod
    def _safe_mi(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
        try:
            out = mutual_info_classif(X, y, random_state=seed)
            return np.asarray(out, dtype=float)
        except Exception as exc:
            return np.zeros(X.shape[1], dtype=float)

    @staticmethod
    def _safe_fscore(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            f_vals, _ = f_classif(X, y)
            return np.asarray(f_vals, dtype=float)
        except Exception as exc:
            return np.zeros(X.shape[1], dtype=float)

    @staticmethod
    def _safe_linear_svm_scores(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
        def _coef_scores(clf: LinearSVC) -> np.ndarray:
            clf.fit(X, y)
            coef = np.asarray(clf.coef_, dtype=float)
            if coef.ndim == 2:
                scores = np.mean(np.abs(coef), axis=0)
            else:
                scores = np.abs(coef)
            return np.asarray(scores, dtype=float)

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", category=ConvergenceWarning)
                l1_clf = LinearSVC(
                    penalty="l1",
                    dual=False,
                    C=0.5,
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=12000,
                    tol=1e-3,
                )
                scores = _coef_scores(l1_clf)
                has_conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in caught)
                if not has_conv_warn:
                    return scores
        except Exception as exc:
            pass

        try:
            l2_clf = LinearSVC(
                penalty="l2",
                C=0.3,
                class_weight="balanced",
                random_state=seed,
                max_iter=8000,
                tol=1e-3,
            )
            return _coef_scores(l2_clf)
        except Exception as exc:
            return np.zeros(X.shape[1], dtype=float)

    @staticmethod
    def _normalize01(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size == 0:
            return arr
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo + 1e-12:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        labels = np.unique(y_true_arr)
        if labels.size == 0:
            return 0.0
        if labels.size == 1:
            return float(np.mean(y_pred_arr == labels[0]))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            return float(recall_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))

    @staticmethod
    def _safe_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        labels = np.unique(y_true_arr)
        if labels.size == 0:
            return 0.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            return float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))

    @staticmethod
    def _safe_log_loss(model: Any, X_eval: np.ndarray, y_true: np.ndarray) -> float:
        """Compute log_loss safely, returning NaN if predict_proba is unavailable."""
        try:
            if not hasattr(model, "predict_proba"):
                return float("nan")
            y_proba = model.predict_proba(np.asarray(X_eval, dtype=float))
            return float(sklearn_log_loss(np.asarray(y_true), y_proba))
        except Exception:
            return float("nan")

    @staticmethod
    def _downsample_roc_curve_points(
        fpr: np.ndarray,
        tpr: np.ndarray,
        *,
        max_points: int = 256,
    ) -> Tuple[Tuple[float, float], ...]:
        x = np.asarray(fpr, dtype=float).ravel()
        y = np.asarray(tpr, dtype=float).ravel()
        n = int(min(x.size, y.size))
        if n <= 0:
            return tuple()
        x = x[:n]
        y = y[:n]
        if n > int(max_points):
            idx = np.linspace(0, n - 1, num=int(max_points), dtype=int)
            idx = np.unique(idx)
            x = x[idx]
            y = y[idx]
        pts: List[Tuple[float, float]] = []
        for xv, yv in zip(x, y):
            if not (np.isfinite(xv) and np.isfinite(yv)):
                continue
            pts.append(
                (
                    float(np.clip(xv, 0.0, 1.0)),
                    float(np.clip(yv, 0.0, 1.0)),
                )
            )
        if not pts:
            return tuple()
        if pts[0] != (0.0, 0.0):
            pts.insert(0, (0.0, 0.0))
        if pts[-1] != (1.0, 1.0):
            pts.append((1.0, 1.0))
        return tuple(pts)

    @staticmethod
    def _align_score_matrix_to_classes(
        scores: np.ndarray,
        *,
        target_classes: np.ndarray,
        score_classes: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        arr = np.asarray(scores, dtype=float)
        targets = np.asarray(target_classes).ravel()
        if arr.ndim != 2 or arr.shape[0] <= 0 or targets.size <= 0:
            return None
        if arr.shape[1] != int(targets.size):
            return None
        if score_classes is None:
            return arr
        cols = np.asarray(score_classes).ravel()
        if cols.size != arr.shape[1]:
            return arr
        order: List[int] = []
        for cls in targets:
            idx = np.where(cols == cls)[0]
            if idx.size <= 0:
                return None
            order.append(int(idx[0]))
        return arr[:, np.asarray(order, dtype=int)]

    @staticmethod
    def _hard_voting_vote_fraction_scores(
        model: VotingClassifier,
        X_eval: np.ndarray,
        *,
        target_classes: np.ndarray,
    ) -> Optional[np.ndarray]:
        if str(getattr(model, "voting", "")).strip().lower() != "hard":
            return None
        estimators = list(getattr(model, "estimators_", []) or [])
        if not estimators:
            return None
        x = np.asarray(X_eval, dtype=float)
        classes = np.asarray(target_classes).ravel()
        if x.ndim != 2 or classes.size < 2:
            return None
        class_to_col = {c: i for i, c in enumerate(classes.tolist())}
        votes = np.zeros((x.shape[0], classes.size), dtype=float)
        used = 0
        for est in estimators:
            try:
                pred = np.asarray(est.predict(x)).ravel()
            except Exception as exc:
                continue
            if pred.size != x.shape[0]:
                continue
            used += 1
            for row_idx, label in enumerate(pred):
                col = class_to_col.get(label, None)
                if col is None:
                    continue
                votes[row_idx, int(col)] += 1.0
        if used <= 0:
            return None
        return votes / float(used)

    def _extract_roc_score_payload(
        self,
        *,
        model: Any,
        X_eval: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], str]:
        x = np.asarray(X_eval, dtype=float)
        y_arr = np.asarray(y_true).ravel()
        classes = np.unique(y_arr)
        if x.ndim != 2 or classes.size < 2:
            return None, "unavailable"

        if hasattr(model, "predict_proba"):
            try:
                proba = np.asarray(model.predict_proba(x), dtype=float)
                if proba.ndim == 1:
                    return proba.ravel(), "predict_proba"
                if proba.ndim == 2:
                    aligned = self._align_score_matrix_to_classes(
                        proba,
                        target_classes=classes,
                        score_classes=getattr(model, "classes_", None),
                    )
                    if aligned is not None:
                        if classes.size == 2:
                            return aligned[:, 1], "predict_proba"
                        return aligned, "predict_proba"
            except Exception as exc:
                pass

        if hasattr(model, "decision_function"):
            try:
                decision = np.asarray(model.decision_function(x), dtype=float)
                if decision.ndim == 1:
                    return decision.ravel(), "decision_function"
                if decision.ndim == 2:
                    aligned = self._align_score_matrix_to_classes(
                        decision,
                        target_classes=classes,
                        score_classes=getattr(model, "classes_", None),
                    )
                    if aligned is not None:
                        if classes.size == 2:
                            return aligned[:, 1], "decision_function"
                        return aligned, "decision_function"
            except Exception as exc:
                pass

        if isinstance(model, VotingClassifier):
            vote_scores = self._hard_voting_vote_fraction_scores(
                model,
                x,
                target_classes=classes,
            )
            if vote_scores is not None:
                if classes.size == 2:
                    return vote_scores[:, 1], "hard_vote_fraction"
                return vote_scores, "hard_vote_fraction"

        pred_arr = np.asarray(y_pred).ravel()
        if pred_arr.size != x.shape[0]:
            return None, "unavailable"
        if classes.size == 2:
            return np.asarray(pred_arr == classes[1], dtype=float), "predicted_label"
        one_hot = np.zeros((pred_arr.size, classes.size), dtype=float)
        class_to_col = {c: i for i, c in enumerate(classes.tolist())}
        for row_idx, label in enumerate(pred_arr):
            col = class_to_col.get(label, None)
            if col is None:
                continue
            one_hot[row_idx, int(col)] = 1.0
        return one_hot, "predicted_label"

    def _safe_roc_auc_bundle(
        self,
        *,
        model: Any,
        X_eval: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        has_predict_proba = callable(getattr(model, "predict_proba", None))
        has_decision_function = callable(getattr(model, "decision_function", None))
        supports_hard_vote_fraction = bool(
            isinstance(model, VotingClassifier)
            and str(getattr(model, "voting", "")).strip().lower() == "hard"
        )
        out: Dict[str, Any] = {
            "roc_auc": float("nan"),
            "roc_curve_type": "unavailable",
            "roc_auc_source": "unavailable",
            "roc_curve_points": tuple(),
            "roc_curve_ova": {},
            "roc_auc_macro_ovr": float("nan"),
            "roc_auc_micro_ovr": float("nan"),
            "roc_auc_weighted_ovr": float("nan"),
            "roc_metric_capabilities": {
                "has_predict_proba": bool(has_predict_proba),
                "has_decision_function": bool(has_decision_function),
                "supports_hard_vote_fraction": bool(supports_hard_vote_fraction),
                "supports_predicted_label_fallback": True,
            },
        }
        y_arr = np.asarray(y_true).ravel()
        classes = np.unique(y_arr)
        if classes.size < 2:
            out["roc_auc_source"] = "single_class"
            out["roc_metric_capabilities"]["selected_source"] = "single_class"
            return out

        scores, source = self._extract_roc_score_payload(
            model=model,
            X_eval=X_eval,
            y_true=y_arr,
            y_pred=y_pred,
        )
        out["roc_auc_source"] = str(source)
        out["roc_metric_capabilities"]["selected_source"] = str(source)
        if scores is None:
            return out

        try:
            if classes.size == 2:
                score_vec = np.asarray(scores, dtype=float).ravel()
                if score_vec.size != y_arr.size:
                    return out
                y_pos = np.asarray(y_arr == classes[1], dtype=int)
                finite = np.isfinite(score_vec)
                if int(np.sum(finite)) < 2:
                    return out
                y_pos = y_pos[finite]
                score_vec = score_vec[finite]
                if np.unique(y_pos).size < 2:
                    return out
                auc_val = float(roc_auc_score(y_pos, score_vec))
                fpr, tpr, _ = roc_curve(y_pos, score_vec)
                out["roc_auc"] = float(auc_val)
                out["roc_curve_type"] = "binary"
                out["roc_curve_points"] = self._downsample_roc_curve_points(fpr, tpr)
                out["roc_auc_macro_ovr"] = float(auc_val)
                out["roc_auc_micro_ovr"] = float(auc_val)
                out["roc_auc_weighted_ovr"] = float(auc_val)
                out["roc_curve_ova"] = {
                    str(classes[1]): {
                        "roc_auc": float(auc_val),
                        "roc_curve_points": [[float(pt[0]), float(pt[1])] for pt in out["roc_curve_points"]],
                    }
                }
                return out

            score_mat = np.asarray(scores, dtype=float)
            if score_mat.ndim != 2 or score_mat.shape[0] != y_arr.size:
                return out
            aligned = self._align_score_matrix_to_classes(
                score_mat,
                target_classes=classes,
                score_classes=None,
            )
            if aligned is None:
                return out
            finite_rows = np.all(np.isfinite(aligned), axis=1)
            if int(np.sum(finite_rows)) < 2:
                return out
            y_valid = y_arr[finite_rows]
            s_valid = aligned[finite_rows]
            y_bin = np.zeros((y_valid.size, classes.size), dtype=int)
            class_to_col = {c: i for i, c in enumerate(classes.tolist())}
            for row_idx, label in enumerate(y_valid):
                col = class_to_col.get(label, None)
                if col is not None:
                    y_bin[row_idx, int(col)] = 1
            macro_auc = float(roc_auc_score(y_bin, s_valid, average="macro", multi_class="ovr"))
            weighted_auc = float(roc_auc_score(y_bin, s_valid, average="weighted", multi_class="ovr"))
            fpr, tpr, _ = roc_curve(y_bin.ravel(), s_valid.ravel())
            micro_auc = float(roc_auc_score(y_bin.ravel(), s_valid.ravel()))
            ova_curves: Dict[str, Dict[str, Any]] = {}
            for class_idx, class_label in enumerate(classes):
                y_one = np.asarray(y_valid == class_label, dtype=int)
                if np.unique(y_one).size < 2:
                    continue
                class_scores = np.asarray(s_valid[:, class_idx], dtype=float).ravel()
                finite = np.isfinite(class_scores)
                if int(np.sum(finite)) < 2:
                    continue
                y_one = y_one[finite]
                class_scores = class_scores[finite]
                if np.unique(y_one).size < 2:
                    continue
                cfpr, ctpr, _ = roc_curve(y_one, class_scores)
                c_auc = float(roc_auc_score(y_one, class_scores))
                ova_curves[str(class_label)] = {
                    "roc_auc": float(c_auc),
                    "roc_curve_points": [
                        [float(pt[0]), float(pt[1])]
                        for pt in self._downsample_roc_curve_points(cfpr, ctpr)
                    ],
                }
            auc_val = float(macro_auc)
            out["roc_auc"] = float(auc_val)
            out["roc_curve_type"] = "ovr_micro"
            out["roc_curve_points"] = self._downsample_roc_curve_points(fpr, tpr)
            out["roc_auc_macro_ovr"] = float(macro_auc)
            out["roc_auc_micro_ovr"] = float(micro_auc)
            out["roc_auc_weighted_ovr"] = float(weighted_auc)
            out["roc_curve_ova"] = ova_curves
            return out
        except Exception as exc:
            return out

    @staticmethod
    def _serialize_roc_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        points = []
        for pt in (meta.get("roc_curve_points") or tuple()):
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            try:
                points.append([float(pt[0]), float(pt[1])])
            except Exception as exc:
                continue
        ova_in = dict(meta.get("roc_curve_ova") or {})
        ova_out: Dict[str, Dict[str, Any]] = {}
        for key, val in ova_in.items():
            if not isinstance(val, dict):
                continue
            class_points = []
            for cpt in (val.get("roc_curve_points") or []):
                if not isinstance(cpt, (list, tuple)) or len(cpt) < 2:
                    continue
                try:
                    class_points.append([float(cpt[0]), float(cpt[1])])
                except Exception as exc:
                    continue
            ova_out[str(key)] = {
                "roc_auc": float(val.get("roc_auc", float("nan"))),
                "roc_curve_points": class_points,
            }
        return {
            "roc_auc": float(meta.get("roc_auc", float("nan"))),
            "roc_curve_type": str(meta.get("roc_curve_type", "unavailable")),
            "roc_auc_source": str(meta.get("roc_auc_source", "unavailable")),
            "roc_curve_points": points,
            "roc_curve_ova": ova_out,
            "roc_auc_macro_ovr": float(meta.get("roc_auc_macro_ovr", float("nan"))),
            "roc_auc_micro_ovr": float(meta.get("roc_auc_micro_ovr", float("nan"))),
            "roc_auc_weighted_ovr": float(meta.get("roc_auc_weighted_ovr", float("nan"))),
            "roc_metric_capabilities": dict(meta.get("roc_metric_capabilities") or {}),
        }

    def _collect_roc_curves_by_method(
        self,
        *,
        model: Any,
        model_name: str,
        X_eval: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        root_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        root = dict(root_meta or {})
        if not root:
            root = self._safe_roc_auc_bundle(
                model=model,
                X_eval=X_eval,
                y_true=y_true,
                y_pred=y_pred,
            )
        out[str(model_name)] = self._serialize_roc_meta(root)

        if not isinstance(model, VotingClassifier):
            return out
        fitted_estimators = list(getattr(model, "estimators_", []) or [])
        named_estimators = list(getattr(model, "estimators", []) or [])
        for idx, est in enumerate(fitted_estimators):
            if idx < len(named_estimators) and isinstance(named_estimators[idx], (list, tuple)) and len(named_estimators[idx]) >= 1:
                name = str(named_estimators[idx][0])
            else:
                name = f"member_{idx:02d}"
            try:
                member_pred = np.asarray(est.predict(np.asarray(X_eval, dtype=float))).ravel()
            except Exception as exc:
                continue
            member_meta = self._safe_roc_auc_bundle(
                model=est,
                X_eval=X_eval,
                y_true=y_true,
                y_pred=member_pred,
            )
            out[str(name)] = self._serialize_roc_meta(member_meta)
        return out

    def _run_feature_selection(
        self,
        X_fs: np.ndarray,
        y_fs: np.ndarray,
        X_train_full: np.ndarray,
        X_test_full: np.ndarray,
        y_train_full: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        dataset_name: str = "dataset",
        post_df_source_raw_train: Optional[np.ndarray] = None,
        post_df_source_raw_test: Optional[np.ndarray] = None,
        post_df_source_base_train: Optional[np.ndarray] = None,
        post_df_source_base_test: Optional[np.ndarray] = None,
        post_df_source_space: str = "prefilter_raw",
    ) -> Dict[str, Any]:
        # T-R-272: apply variance floor before any feature-selection logic.
        variance_floor_meta: Dict[str, Any] = {
            "variance_floor_enabled": bool(getattr(self.config, "prefilter_variance_floor_enabled", True)),
            "variance_floor_n_removed": 0,
            "variance_floor_n_before": int(X_train_full.shape[1]),
        }
        if bool(getattr(self.config, "prefilter_variance_floor_enabled", True)):
            vf_threshold = float(getattr(self.config, "prefilter_variance_floor_threshold", 1e-6))
            vf_mode_freq = float(getattr(self.config, "prefilter_variance_floor_mode_freq", 0.99))
            n_features_before = int(X_train_full.shape[1])
            keep_mask = np.ones(n_features_before, dtype=bool)
            for j in range(n_features_before):
                col = X_train_full[:, j]
                if np.var(col) < vf_threshold:
                    keep_mask[j] = False
                    continue
                # Check mode frequency (most common value frequency)
                vals, counts = np.unique(col, return_counts=True)
                if counts.max() / len(col) > vf_mode_freq:
                    keep_mask[j] = False
            n_removed = int(n_features_before - keep_mask.sum())
            if n_removed > 0 and keep_mask.sum() > 0:
                X_fs = X_fs[:, keep_mask]
                X_train_full = X_train_full[:, keep_mask]
                X_test_full = X_test_full[:, keep_mask]
                if (
                    post_df_source_raw_train is not None
                    and np.asarray(post_df_source_raw_train).ndim == 2
                    and np.asarray(post_df_source_raw_train).shape[1] == n_features_before
                ):
                    post_df_source_raw_train = np.asarray(post_df_source_raw_train, dtype=float)[:, keep_mask]
                if (
                    post_df_source_raw_test is not None
                    and np.asarray(post_df_source_raw_test).ndim == 2
                    and np.asarray(post_df_source_raw_test).shape[1] == n_features_before
                ):
                    post_df_source_raw_test = np.asarray(post_df_source_raw_test, dtype=float)[:, keep_mask]
                if (
                    post_df_source_base_train is not None
                    and np.asarray(post_df_source_base_train).ndim == 2
                    and np.asarray(post_df_source_base_train).shape[1] == n_features_before
                ):
                    post_df_source_base_train = np.asarray(post_df_source_base_train, dtype=float)[:, keep_mask]
                if (
                    post_df_source_base_test is not None
                    and np.asarray(post_df_source_base_test).ndim == 2
                    and np.asarray(post_df_source_base_test).shape[1] == n_features_before
                ):
                    post_df_source_base_test = np.asarray(post_df_source_base_test, dtype=float)[:, keep_mask]
                logger.info(
                    "T-R-272 variance floor: removed %d / %d near-constant features for %s",
                    n_removed, n_features_before, dataset_name,
                )
            elif keep_mask.sum() == 0:
                # Edge case: all features would be removed — keep all.
                n_removed = 0
                logger.warning(
                    "T-R-272 variance floor: would remove ALL features; keeping all for %s",
                    dataset_name,
                )
            variance_floor_meta["variance_floor_n_removed"] = n_removed

        candidate = self._choose_selector_candidate(
            X_fs=X_fs,
            y_fs=y_fs,
            X_train_full=X_train_full,
            X_test_full=X_test_full,
            y_train_full=y_train_full,
            seed=seed,
            dataset_name=dataset_name,
            post_df_source_raw_train=post_df_source_raw_train,
            post_df_source_raw_test=post_df_source_raw_test,
            post_df_source_base_train=post_df_source_base_train,
            post_df_source_base_test=post_df_source_base_test,
            post_df_source_space=post_df_source_space,
        )

        X_train_sel = np.asarray(candidate["X_train_sel"], dtype=float)
        X_test_sel = np.asarray(candidate["X_test_sel"], dtype=float)
        model = candidate["model"]
        model_name = str(candidate["model_name"])

        model.fit(X_train_sel, y_train_full)
        y_pred = model.predict(X_test_sel)

        bal_acc = self._safe_balanced_accuracy(y_test, y_pred)
        macro_f1 = self._safe_macro_f1(y_test, y_pred)
        _log_loss_val = self._safe_log_loss(model, X_test_sel, y_test)
        roc_meta = self._safe_roc_auc_bundle(
            model=model,
            X_eval=X_test_sel,
            y_true=y_test,
            y_pred=y_pred,
        )
        roc_curves_by_method = self._collect_roc_curves_by_method(
            model=model,
            model_name=str(model_name),
            X_eval=X_test_sel,
            y_true=y_test,
            y_pred=y_pred,
            root_meta=roc_meta,
        )
        cls_cfg = self._classification_cfg()
        conformal_meta: Dict[str, Any] = {
            "classifier_conformal_enabled": bool(getattr(cls_cfg, "conformal_enabled", False)),
            "classifier_conformal_method": str(
                getattr(cls_cfg, "conformal_method", "split") or "split"
            ).strip().lower(),
            "classifier_conformal_applied": False,
            "classifier_conformal_skip_reason": "disabled",
            "classifier_conformal_alpha": float(getattr(cls_cfg, "conformal_alpha", 0.10) or 0.10),
            "classifier_conformal_calibration_fraction": float(
                getattr(cls_cfg, "conformal_calibration_fraction", 0.25) or 0.25
            ),
            "classifier_conformal_min_calibration": int(
                getattr(cls_cfg, "conformal_min_calibration", 20) or 20
            ),
            "classifier_conformal_calibration_size": 0,
            "classifier_conformal_fit_size": 0,
            "classifier_conformal_qhat": float("nan"),
            "classifier_conformal_threshold": float("nan"),
            "classifier_conformal_set_size_mean": float("nan"),
            "classifier_conformal_set_size_median": float("nan"),
            "classifier_conformal_singleton_rate": float("nan"),
            "classifier_conformal_empty_set_rate": float("nan"),
            "classifier_conformal_coverage": float("nan"),
            "classifier_conformal_classes": [],
            "classifier_conformal_prediction_sets": [],
            "classifier_conformal_mapie_enabled": False,
            "classifier_conformal_mapie_applied": False,
            "classifier_conformal_mapie_skip_reason": "not_requested",
            "classifier_conformal_mapie_method": "none",
            "classifier_conformal_mapie_alpha": float("nan"),
            "classifier_conformal_mapie_set_size_mean": float("nan"),
            "classifier_conformal_mapie_set_size_median": float("nan"),
            "classifier_conformal_mapie_singleton_rate": float("nan"),
            "classifier_conformal_mapie_empty_set_rate": float("nan"),
            "classifier_conformal_mapie_coverage": float("nan"),
            "classifier_conformal_mapie_classes": [],
            "classifier_conformal_mapie_prediction_sets": [],
        }
        if bool(getattr(cls_cfg, "conformal_enabled", False)):
            try:
                split_meta = compute_split_conformal_sets(
                    model=model,
                    X_train=np.asarray(X_train_sel, dtype=float),
                    y_train=np.asarray(y_train_full).ravel(),
                    X_eval=np.asarray(X_test_sel, dtype=float),
                    y_eval=np.asarray(y_test).ravel(),
                    alpha=float(getattr(cls_cfg, "conformal_alpha", 0.10) or 0.10),
                    calibration_fraction=float(
                        getattr(cls_cfg, "conformal_calibration_fraction", 0.25) or 0.25
                    ),
                    min_calibration=int(getattr(cls_cfg, "conformal_min_calibration", 20) or 20),
                    seed=int(seed),
                    include_prediction_sets=bool(getattr(cls_cfg, "conformal_output_sets", False)),
                )
                conformal_meta.update(split_meta)
            except Exception as exc:
                conformal_meta["classifier_conformal_applied"] = False
                conformal_meta["classifier_conformal_skip_reason"] = str(type(exc).__name__)
                conformal_meta["classifier_conformal_prediction_sets"] = []

            # VAL12_Suggestions §2.3: MAPIE APS/RAPS/cross conformal (opt-in).
            _conformal_method = str(getattr(cls_cfg, "conformal_method", "split") or "split").strip().lower()
            if _conformal_method in {"aps", "raps", "cross"}:
                try:
                    mapie_meta = compute_mapie_conformal_sets(
                        model=model,
                        X_train=np.asarray(X_train_sel, dtype=float),
                        y_train=np.asarray(y_train_full).ravel(),
                        X_eval=np.asarray(X_test_sel, dtype=float),
                        y_eval=np.asarray(y_test).ravel(),
                        alpha=float(getattr(cls_cfg, "conformal_alpha", 0.10) or 0.10),
                        method=_conformal_method,
                        seed=int(seed),
                        include_prediction_sets=bool(getattr(cls_cfg, "conformal_output_sets", False)),
                    )
                    conformal_meta.update(mapie_meta)
                except Exception as exc:
                    conformal_meta["classifier_conformal_mapie_applied"] = False
                    conformal_meta["classifier_conformal_mapie_skip_reason"] = str(type(exc).__name__)

        selected_indices = candidate.get("selected_indices")
        if selected_indices is None:
            selected_indices = np.arange(X_train_sel.shape[1], dtype=int)

        out: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "log_loss": _log_loss_val,
            "hybrid_score": float(0.6 * bal_acc + 0.4 * macro_f1),
            "roc_auc": float(roc_meta.get("roc_auc", float("nan"))),
            "roc_curve_type": str(roc_meta.get("roc_curve_type", "unavailable")),
            "roc_auc_source": str(roc_meta.get("roc_auc_source", "unavailable")),
            "roc_curve_points": tuple(roc_meta.get("roc_curve_points", tuple()) or tuple()),
            "roc_curves_by_method": dict(roc_curves_by_method),
            "selected_features": int(X_train_sel.shape[1]),
            "selected_indices": tuple(int(i) for i in np.asarray(selected_indices, dtype=int).tolist()),
            "model_name": str(model_name),
            "effective_enabled_methods": tuple(str(m) for m in candidate.get("enabled_methods") or tuple(self.config.enabled_methods)),
            "enabled_methods_source": str(candidate.get("enabled_methods_source", "config")),
        }
        out.update(conformal_meta)
        # T-R-272: attach variance floor diagnostics.
        out.update(variance_floor_meta)

        model_cv_meta = candidate.get("model_cv_meta")
        if isinstance(model_cv_meta, dict):
            out.update(model_cv_meta)

        # Optional pairing diagnostics (kept lightweight and JSON-safe).
        pairing_meta = candidate.get("pairing_meta")
        if isinstance(pairing_meta, dict):
            out.update(pairing_meta)

        stage2_ratio_meta = candidate.get("stage2_ratio_meta")
        if isinstance(stage2_ratio_meta, dict):
            out.update(stage2_ratio_meta)

        # Tier-policy metadata (T-P3-002 / T-P3-005)
        for key in (
            "tier_policy_applied",
            "tier_policy_mode",
            "tier_policy_target_tier",
            "tier_policy_resolved_tier",
            "tier_policy_source",
            "tier_policy_enabled_methods_before",
            "tier_policy_enabled_methods_after",
            "tier_policy_meta_features",
        ):
            if key in candidate:
                out[key] = candidate[key]
        for key in (
            "regime_policy_applied",
            "regime_policy_mode",
            "regime_policy_reason",
            "regime_policy_enabled",
            "regime_policy_target_tier",
            "regime_policy_tier",
            "regime_policy_tier_source",
            "regime_policy_tier_meta_features",
            "regime_policy_n_samples",
            "regime_policy_n_features",
            "regime_policy_n_classes",
            "regime_policy_samples_per_class",
            "regime_policy_p_over_n",
            "regime_policy_min_samples_per_class_threshold",
            "regime_policy_low_p_over_n_threshold",
            "regime_policy_trigger_very_hard",
            "regime_policy_trigger_low_p_over_n",
            "regime_policy_enabled_methods_before",
            "regime_policy_enabled_methods_after",
            "regime_policy_enabled_methods_source",
            "regime_policy_bypass_fs",
            "regime_policy_bypass_mode",
            "regime_policy_selector_overrides",
            "selector_overrides_applied",
        ):
            if key in candidate:
                out[key] = candidate[key]
        for key in (
            "importance_uq_enabled",
            "importance_uq_computed",
            "importance_uq_reason",
            "importance_uq_n_folds",
            "importance_uq_unstable_threshold",
            "importance_uq_unstable_feature_count",
            "importance_uq_unstable_feature_indices",
        ):
            if key in candidate:
                out[key] = candidate[key]

        for key in ("fs_selection_summary", "fs_diagnostics"):
            if key in candidate:
                out[key] = candidate[key]

        out["_fitted_selector"] = candidate.get("_fitted_selector")
        out["_selection_result"] = candidate.get("_selection_result")
        out["_fitted_model"] = model
        out["_post_df_summaries"] = list(candidate.get("_post_df_summaries") or [])
        out["_post_df_meta"] = dict(candidate.get("_post_df_meta") or {})
        out["_post_df_time_sec"] = float(candidate.get("_post_df_time_sec", 0.0) or 0.0)
        out["_folding_meta"] = dict(candidate.get("_folding_meta") or {})
        out["_folding_state"] = dict(candidate.get("_folding_state") or {})

        return out

    def _build_feature_selector(
        self,
        seed: int,
        enabled_methods: Sequence[str],
        selector_overrides: Optional[Dict[str, Any]] = None,
    ):
        FeatureSelector = _load_feature_selector_cls()
        overrides = dict(selector_overrides or {})
        portfolio_size = int(
            max(
                1,
                overrides.get("fs_portfolio_size", getattr(self.config, "fs_portfolio_size", 5)),
            )
        )
        adaptive_enabled = bool(
            overrides.get(
                "fs_adaptive_portfolio_sizing_enabled",
                getattr(self.config, "fs_adaptive_portfolio_sizing_enabled", False),
            )
        )
        adaptive_min = overrides.get("fs_adaptive_size_min", getattr(self.config, "fs_adaptive_size_min", None))
        adaptive_max = overrides.get("fs_adaptive_size_max", getattr(self.config, "fs_adaptive_size_max", None))
        selection_strategy = str(
            overrides.get(
                "selection_strategy",
                getattr(self.config, "selection_strategy", "mnpo_portfolio"),
            )
            or "mnpo_portfolio"
        ).strip().lower()
        if selection_strategy not in {"mnpo_portfolio", "legacy_voting"}:
            selection_strategy = "mnpo_portfolio"
        copula_derandomize_runs = int(
            max(
                1,
                overrides.get(
                    "fs_copula_derandomize_runs",
                    getattr(self.config, "fs_copula_derandomize_runs", 1),
                )
                or 1,
            )
        )
        if adaptive_enabled and (adaptive_min is None or adaptive_max is None):
            # Val-7 compatibility: preserve adaptive sizing when bounds are omitted.
            default_min = int(max(1, portfolio_size - 2))
            default_max = int(max(portfolio_size, portfolio_size + 2))
            if adaptive_min is None:
                adaptive_min = int(default_min)
            if adaptive_max is None:
                adaptive_max = int(default_max)

        fs_cfg = getattr(self.config, "fs_config", None)
        if fs_cfg is not None and FeatureSelectorConfig is not None:
            if isinstance(fs_cfg, FeatureSelectorConfig):
                cfg_obj = copy.deepcopy(fs_cfg)
            else:
                cfg_obj = fs_cfg
            try:
                # Run-level overrides remain owned by DFFSConfig / benchmark args.
                setattr(cfg_obj, "random_state", int(seed))
                setattr(cfg_obj, "problem_type", "classification")
                setattr(cfg_obj, "selection_strategy", str(selection_strategy))
                setattr(cfg_obj, "enabled_methods", set(enabled_methods))
                setattr(cfg_obj, "method_timeout_seconds", float(getattr(self.config, "fs_method_timeout_seconds", 0.0) or 0.0))
                setattr(cfg_obj, "linear_svm_max_iter", int(getattr(self.config, "fs_linear_svm_max_iter", 10000) or 10000))
                setattr(cfg_obj, "parallel_n_jobs", int(getattr(self.config, "n_jobs", 1) or 1))
                stability_cfg = getattr(cfg_obj, "stability", None)
                if stability_cfg is not None:
                    setattr(
                        stability_cfg,
                        "stability_target_pfer",
                        float(getattr(self.config, "fs_stability_target_pfer", 1.0) or 1.0),
                    )
                mnpo_cfg = getattr(cfg_obj, "mnpo", None)
                if mnpo_cfg is not None:
                    setattr(
                        mnpo_cfg,
                        "inner_cv_splits",
                        int(getattr(self.config, "fs_inner_cv_splits", 3) or 3),
                    )
                    setattr(
                        mnpo_cfg,
                        "inner_cv_repeats",
                        int(getattr(self.config, "fs_inner_cv_repeats", 1) or 1),
                    )
                    setattr(mnpo_cfg, "mirror_descent_steps", 100)
                    setattr(mnpo_cfg, "portfolio_size", int(portfolio_size))
                    setattr(
                        mnpo_cfg,
                        "payoff_shrinkage_kappa",
                        float(getattr(self.config, "fs_payoff_shrinkage_kappa", 0.0) or 0.0),
                    )
                    oracle_cfg = getattr(mnpo_cfg, "oracle", None)
                    if oracle_cfg is not None:
                        setattr(
                            oracle_cfg,
                            "fold_preference_mode",
                            str(getattr(self.config, "fs_fold_preference_mode", "vote") or "vote"),
                        )
                        setattr(
                            oracle_cfg,
                            "use_conformal_efficiency",
                            bool(getattr(self.config, "fs_use_conformal_efficiency", False)),
                        )
                        setattr(
                            oracle_cfg,
                            "conformal_efficiency_method",
                            str(
                                getattr(self.config, "fs_conformal_efficiency_method", "split")
                                or "split"
                            ),
                        )
                        setattr(
                            oracle_cfg,
                            "oracle_weight_js_shrinkage",
                            bool(getattr(self.config, "fs_oracle_weight_js_shrinkage", False)),
                        )
                return FeatureSelector.from_config(cfg_obj)
            except Exception as exc:
                warnings.warn(
                    f"fs_config path failed ({exc}); falling back to legacy DFFSConfig->FeatureSelector mapping.",
                    RuntimeWarning,
                )

        return FeatureSelector(
            n_bootstrap_iterations=3,
            random_state=seed,
            problem_type="classification",
            selection_strategy=str(selection_strategy),
            inner_cv_splits=int(getattr(self.config, "fs_inner_cv_splits", 3) or 3),
            inner_cv_repeats=int(getattr(self.config, "fs_inner_cv_repeats", 1) or 1),
            mirror_descent_steps=100,
            portfolio_size=int(portfolio_size),
            enabled_methods=list(enabled_methods),
            prefilter_mi_weight=float(getattr(self.config, "prefilter_mi_weight", 0.60) or 0.60),
            prefilter_f_weight=float(getattr(self.config, "prefilter_f_weight", 0.40) or 0.40),
            prefilter_union_enabled=bool(getattr(self.config, "prefilter_union_enabled", False)),
            prefilter_strategies=tuple(
                getattr(self.config, "prefilter_strategies", ("mi_ftest_blend",)) or ("mi_ftest_blend",)
            ),
            prefilter_nondefault_budget_fraction=float(
                getattr(self.config, "prefilter_nondefault_budget_fraction", 0.10) or 0.10
            ),
            prefilter_wsnr_enabled=bool(getattr(self.config, "prefilter_wsnr_enabled", False)),
            prefilter_data_domain=str(getattr(self.config, "prefilter_data_domain", "auto") or "auto"),
            prefilter_rnaseq_transform_enabled=bool(
                getattr(self.config, "prefilter_rnaseq_transform_enabled", True)
            ),
            prefilter_rnaseq_transform_force=bool(
                getattr(self.config, "prefilter_rnaseq_transform_force", False)
            ),
            prefilter_rnaseq_nb_lrt_enabled=bool(
                getattr(self.config, "prefilter_rnaseq_nb_lrt_enabled", False)
            ),
            prefilter_rnaseq_nb_lrt_alpha=float(
                getattr(self.config, "prefilter_rnaseq_nb_lrt_alpha", 0.10) or 0.10
            ),
            screening_enabled=bool(getattr(self.config, "screening_enabled", False)),
            screening_method=str(getattr(self.config, "screening_method", "none") or "none"),
            screening_pool_cap=int(getattr(self.config, "screening_pool_cap", 2000) or 2000),
            screening_stir_n_neighbors=int(getattr(self.config, "screening_stir_n_neighbors", 10) or 10),
            screening_stir_n_iter=int(getattr(self.config, "screening_stir_n_iter", 50) or 50),
            screening_stir_keep_fraction=float(
                getattr(self.config, "screening_stir_keep_fraction", 0.5) or 0.5
            ),
            screening_stir_min_features=int(getattr(self.config, "screening_stir_min_features", 20) or 20),
            screening_evalue_alpha=float(getattr(self.config, "screening_evalue_alpha", 0.20) or 0.20),
            screening_evalue_min_features=int(
                getattr(self.config, "screening_evalue_min_features", 20) or 20
            ),
            eval_models_enabled=bool(getattr(self.config, "eval_models_enabled", False)),
            eval_models=tuple(getattr(self.config, "eval_models", ("lr_l2", "linear_svc", "rf_small")) or ()),
            eval_aggregate=str(getattr(self.config, "eval_aggregate", "mean") or "mean"),
            eval_cvar_alpha=float(getattr(self.config, "eval_cvar_alpha", 0.33) or 0.33),
            performance_oracle_mode=str(getattr(self.config, "mnpo_performance_oracle_mode", "single") or "single"),
            portfolio_size_guard=str(getattr(self.config, "fs_portfolio_size_guard", "none") or "none"),
            mnpo_consensus_exclude_methods=list(getattr(self.config, "fs_mnpo_consensus_exclude_methods", ()) or ()),
            mnpo_consensus_exclude_protect_top_k=int(
                getattr(self.config, "fs_mnpo_consensus_exclude_protect_top_k", 0) or 0
            ),
            mnpo_include_legacy_consensus=bool(getattr(self.config, "fs_mnpo_include_legacy_consensus", True)),
            mnpo_include_majority_consensus=bool(getattr(self.config, "fs_mnpo_include_majority_consensus", True)),
            runtime_racing_enabled=bool(getattr(self.config, "fs_runtime_racing_enabled", False)),
            runtime_racing_proxy_splits=int(getattr(self.config, "fs_runtime_racing_proxy_splits", 1) or 1),
            runtime_racing_keep_fraction=float(getattr(self.config, "fs_runtime_racing_keep_fraction", 0.60) or 0.60),
            runtime_racing_min_candidates=int(getattr(self.config, "fs_runtime_racing_min_candidates", 4) or 4),
            runtime_racing_runtime_weight=float(
                getattr(self.config, "fs_runtime_racing_runtime_weight", 0.15) or 0.15
            ),
            runtime_racing_mode=str(getattr(self.config, "fs_runtime_racing_mode", "single_stage") or "single_stage"),
            runtime_racing_stages=int(getattr(self.config, "fs_runtime_racing_stages", 2) or 2),
            runtime_racing_confidence_bound=str(
                getattr(self.config, "fs_runtime_racing_confidence_bound", "none") or "none"
            ),
            runtime_racing_delta=float(getattr(self.config, "fs_runtime_racing_delta", 0.10) or 0.10),
            use_tritrust=self.config.use_tritrust,
            use_stability_oracle=self.config.use_stability_oracle,
            use_complexity_oracle=self.config.use_complexity_oracle,
            use_robust_oracle=self.config.use_robust_oracle,
            use_diversity_oracle=self.config.use_diversity_oracle,
            use_cvar=bool(getattr(self.config, "fs_use_cvar_oracle", False)),
            cvar_alpha=float(getattr(self.config, "fs_cvar_alpha", 0.33) or 0.33),
            oracle_weighting_mode=str(
                getattr(self.config, "fs_oracle_weighting_mode", "tritrust") or "tritrust"
            ),
            shapley_n_coalitions_max=int(
                getattr(self.config, "fs_shapley_n_coalitions_max", 4096) or 4096
            ),
            shapley_bayesian_shrinkage=bool(
                getattr(self.config, "fs_shapley_bayesian_shrinkage", False)
            ),
            shapley_bayesian_prior_strength=float(
                getattr(self.config, "fs_shapley_bayesian_prior_strength", 8.0) or 8.0
            ),
            use_interaction_oracle=bool(getattr(self.config, "fs_use_interaction_oracle", False)),
            interaction_oracle_min_n_train=int(
                getattr(self.config, "fs_interaction_oracle_min_n_train", 150) or 150
            ),
            interaction_oracle_pool_size_cap=int(
                getattr(self.config, "fs_interaction_oracle_pool_size_cap", 64) or 64
            ),
            interaction_oracle_pair_cap=int(
                getattr(self.config, "fs_interaction_oracle_pair_cap", 20000) or 20000
            ),
            use_ubayfs=bool(getattr(self.config, "fs_use_ubayfs_oracle", False)),
            ubayfs_n_bootstrap=int(getattr(self.config, "fs_ubayfs_n_bootstrap", 32) or 32),
            ubayfs_min_n=int(getattr(self.config, "fs_ubayfs_min_n", 100) or 100),
            ubayfs_prior_weight=float(getattr(self.config, "fs_ubayfs_prior_weight", 0.0) or 0.0),
            use_conformal_uq=bool(getattr(self.config, "fs_use_conformal_uq", False)),
            conformal_uq_alpha=float(getattr(self.config, "fs_conformal_uq_alpha", 0.10) or 0.10),
            conformal_uq_min_folds=int(getattr(self.config, "fs_conformal_uq_min_folds", 5) or 5),
            fold_preference_mode=str(
                getattr(self.config, "fs_fold_preference_mode", "vote") or "vote"
            ),
            use_conformal_efficiency=bool(
                getattr(self.config, "fs_use_conformal_efficiency", False)
            ),
            conformal_efficiency_method=str(
                getattr(self.config, "fs_conformal_efficiency_method", "split") or "split"
            ),
            oracle_weight_js_shrinkage=bool(
                getattr(self.config, "fs_oracle_weight_js_shrinkage", False)
            ),
            payoff_shrinkage_kappa=float(
                getattr(self.config, "fs_payoff_shrinkage_kappa", 0.0) or 0.0
            ),
            use_tail_risk_oracle=False,
            tail_risk_alpha=float(getattr(self.config, "tail_risk_alpha", 0.33) or 0.33),
            use_regret_oracle=False,
            use_qre_smoothing=bool(getattr(self.config, "use_qre_smoothing", False)),
            qre_temperature_gamma=float(getattr(self.config, "qre_temperature_gamma", 1.0) or 1.0),
            use_oracle_redundancy_penalty=bool(getattr(self.config, "use_oracle_redundancy_penalty", False)),
            compute_tremble_sensitivity=bool(getattr(self.config, "compute_tremble_sensitivity", False)),
            diversity_oracle_mode=self.config.fs_diversity_oracle_mode,
            diversity_redundancy_weight=self.config.fs_diversity_redundancy_weight,
            diversity_complementarity_weight=self.config.fs_diversity_complementarity_weight,
            performance_balanced_weight=self.config.fs_performance_balanced_weight,
            performance_macro_f1_weight=self.config.fs_performance_macro_f1_weight,
            performance_use_adaptive_imbalance=self.config.fs_performance_use_adaptive_imbalance,
            performance_imbalance_ratio_trigger=self.config.fs_performance_imbalance_ratio_trigger,
            performance_min_classes_for_adaptive=self.config.fs_performance_min_classes_for_adaptive,
            rank_aggregation_mode=self.config.fs_rank_aggregation_mode,
            wrapper_refine_enabled=self.config.fs_wrapper_refine_enabled,
            wrapper_refine_top_k=self.config.fs_wrapper_refine_top_k,
            wrapper_refine_max_add=self.config.fs_wrapper_refine_max_add,
            wrapper_refine_min_gain=self.config.fs_wrapper_refine_min_gain,
            adaptive_portfolio_sizing_enabled=bool(adaptive_enabled),
            adaptive_size_min=adaptive_min,
            adaptive_size_max=adaptive_max,
            adaptive_sizing_variance_penalty=bool(
                getattr(self.config, "fs_adaptive_sizing_variance_penalty", False)
            ),
            adaptive_sizing_variance_penalty_strength=float(
                getattr(self.config, "fs_adaptive_sizing_variance_penalty_strength", 0.5) or 0.5
            ),
            # T-R-266: Pareto-front portfolio sizing.
            pareto_portfolio_sizing_enabled=bool(
                getattr(self.config, "fs_pareto_portfolio_sizing_enabled", False)
            ),
            # T-R-271: stability-weighted portfolio aggregation.
            stability_weighted_aggregation_enabled=bool(
                getattr(self.config, "fs_stability_weighted_aggregation_enabled", False)
            ),
            mnpo_paradigm_aware_prior_enabled=bool(
                getattr(self.config, "fs_mnpo_paradigm_aware_prior_enabled", False)
            ),
            mnpo_interaction_floor=float(
                getattr(self.config, "fs_mnpo_interaction_floor", 0.12) or 0.12
            ),
            rashomon_enabled=bool(getattr(self.config, "fs_rashomon_enabled", False)),
            rashomon_max_models=int(getattr(self.config, "fs_rashomon_max_models", 12) or 12),
            rashomon_score_tolerance=float(
                getattr(self.config, "fs_rashomon_score_tolerance", 0.01) or 0.01
            ),
            ova_negative_ratio=self.config.fs_ova_negative_ratio,
            ova_min_classes=self.config.fs_ova_min_classes,
            ova_min_pos_samples=self.config.fs_ova_min_pos_samples,
            ova_class_weight_mode=self.config.fs_ova_class_weight_mode,
            ova_aggregation_mode=self.config.fs_ova_aggregation_mode,
            ova_aggregation_p=self.config.fs_ova_aggregation_p,
            ova_linear_backend=self.config.fs_ova_linear_backend,
            ova_enable_calibration=bool(getattr(self.config, "fs_ova_enable_calibration", False)),
            ova_calibration_cv=int(getattr(self.config, "fs_ova_calibration_cv", 3) or 3),
            ecoc_min_classes=self.config.fs_ecoc_min_classes,
            ecoc_max_ovo_pairs=self.config.fs_ecoc_max_ovo_pairs,
            ecoc_random_code_bits=self.config.fs_ecoc_random_code_bits,
            ecoc_class_complexity_weight=self.config.fs_ecoc_class_complexity_weight,
            ecoc_include_ova_tasks=self.config.fs_ecoc_include_ova_tasks,
            ecoc_negative_ratio=self.config.fs_ecoc_negative_ratio,
            joint_multiclass_min_classes=self.config.fs_joint_multiclass_min_classes,
            joint_multiclass_max_features=self.config.fs_joint_multiclass_max_features,
            joint_multiclass_path_grid_size=self.config.fs_joint_multiclass_path_grid_size,
            joint_multiclass_min_c=self.config.fs_joint_multiclass_min_c,
            joint_multiclass_max_c=self.config.fs_joint_multiclass_max_c,
            joint_multiclass_l1_ratio=self.config.fs_joint_multiclass_l1_ratio,
            joint_multiclass_univariate_blend=self.config.fs_joint_multiclass_univariate_blend,
            dove_min_classes=int(getattr(self.config, "fs_dove_min_classes", 3) or 3),
            dove_max_pairs_per_class=int(getattr(self.config, "fs_dove_max_pairs_per_class", 4) or 4),
            dove_path_grid_size=int(getattr(self.config, "fs_dove_path_grid_size", 5) or 5),
            dove_specificity_weight=float(getattr(self.config, "fs_dove_specificity_weight", 0.35) or 0.35),
            dove_minority_boost=float(getattr(self.config, "fs_dove_minority_boost", 0.50) or 0.50),
            sparse_multinomial_min_classes=int(
                getattr(self.config, "fs_sparse_multinomial_min_classes", 3) or 3
            ),
            sparse_multinomial_max_features=int(
                getattr(self.config, "fs_sparse_multinomial_max_features", 320) or 320
            ),
            sparse_multinomial_path_grid_size=int(
                getattr(self.config, "fs_sparse_multinomial_path_grid_size", 6) or 6
            ),
            sparse_multinomial_min_c=float(getattr(self.config, "fs_sparse_multinomial_min_c", 0.05) or 0.05),
            sparse_multinomial_max_c=float(getattr(self.config, "fs_sparse_multinomial_max_c", 1.6) or 1.6),
            sparse_multinomial_backend=str(
                getattr(self.config, "fs_sparse_multinomial_backend", "mixed") or "mixed"
            ),
            sparse_multinomial_l1_ratio=float(
                getattr(self.config, "fs_sparse_multinomial_l1_ratio", 0.70) or 0.70
            ),
            sparse_multinomial_univariate_blend=float(
                getattr(self.config, "fs_sparse_multinomial_univariate_blend", 0.20) or 0.20
            ),
            sparse_multinomial_max_iter=int(
                getattr(self.config, "fs_sparse_multinomial_max_iter", 5000) or 5000
            ),
            sparse_multinomial_screening_mode=_canonicalize_sparse_screening_mode(
                getattr(self.config, "fs_sparse_multinomial_screening_mode", "none"),
                warn_deprecated=True,
            ),
            sparse_multinomial_screening_keep_fraction=float(
                getattr(self.config, "fs_sparse_multinomial_screening_keep_fraction", 1.0) or 1.0
            ),
            sparse_multinomial_screening_min_features=int(
                getattr(self.config, "fs_sparse_multinomial_screening_min_features", 64) or 64
            ),
            sparse_multinomial_screening_fallback_on_failure=bool(
                getattr(self.config, "fs_sparse_multinomial_screening_fallback_on_failure", True)
            ),
            nsc_shrinkage_grid_size=int(
                getattr(self.config, "fs_nsc_shrinkage_grid_size", 6) or 6
            ),
            nsc_min_classes=int(
                getattr(self.config, "fs_nsc_min_classes", 3) or 3
            ),
            nsc_thresholding_mode=str(
                getattr(self.config, "fs_nsc_thresholding_mode", "soft") or "soft"
            ),
            nsc_order_quantile=float(
                getattr(self.config, "fs_nsc_order_quantile", 0.75) or 0.75
            ),
            nsc_deep_shrinkage_search=bool(
                getattr(self.config, "fs_nsc_deep_shrinkage_search", False)
            ),
            class_pareto_min_classes=int(
                getattr(self.config, "fs_class_pareto_min_classes", 3) or 3
            ),
            class_pareto_top_per_class=int(
                getattr(self.config, "fs_class_pareto_top_per_class", 64) or 64
            ),
            class_pareto_global_fraction=float(
                getattr(self.config, "fs_class_pareto_global_fraction", 0.40) or 0.40
            ),
            class_pareto_minority_boost=float(
                getattr(self.config, "fs_class_pareto_minority_boost", 0.50) or 0.50
            ),
            class_pareto_kw_weight=float(
                getattr(self.config, "fs_class_pareto_kw_weight", 0.25) or 0.25
            ),
            sdr_min_classes=int(
                getattr(self.config, "fs_sdr_min_classes", 3) or 3
            ),
            sdr_prefilter_max_features=int(
                getattr(self.config, "fs_sdr_prefilter_max_features", 512) or 512
            ),
            sdr_n_components=int(
                getattr(self.config, "fs_sdr_n_components", 3) or 3
            ),
            sdr_covariance_ridge=float(
                getattr(self.config, "fs_sdr_covariance_ridge", 1e-3) or 1e-3
            ),
            per_class_quota_enabled=bool(
                getattr(self.config, "fs_per_class_quota_enabled", False)
            ),
            per_class_quota_min_per_class=int(
                getattr(self.config, "fs_per_class_quota_min_per_class", 1) or 1
            ),
            per_class_quota_max_fraction=float(
                getattr(self.config, "fs_per_class_quota_max_fraction", 0.60) or 0.60
            ),
            hsic_lasso_alpha=float(
                getattr(self.config, "fs_hsic_lasso_alpha", 0.01) or 0.01
            ),
            hsic_lasso_prefilter_max_features=int(
                getattr(self.config, "fs_hsic_lasso_prefilter_max_features", 128) or 128
            ),
            hsic_lasso_feature_sigma=float(
                getattr(self.config, "fs_hsic_lasso_feature_sigma", 0.0) or 0.0
            ),
            hsic_lasso_target_sigma=float(
                getattr(self.config, "fs_hsic_lasso_target_sigma", 0.0) or 0.0
            ),
            hsic_lasso_relevance_blend=float(
                getattr(self.config, "fs_hsic_lasso_relevance_blend", 0.20) or 0.20
            ),
            hsic_lasso_max_iter=int(
                getattr(self.config, "fs_hsic_lasso_max_iter", 4000) or 4000
            ),
            mrmr_mi_redundancy_enabled=bool(
                getattr(self.config, "fs_mrmr_mi_redundancy_enabled", False)
            ),
            mrmr_mi_n_bins=int(getattr(self.config, "fs_mrmr_mi_n_bins", 8) or 8),
            cmim_min_samples=int(getattr(self.config, "fs_cmim_min_samples", 60) or 60),
            cmim_n_bins=int(getattr(self.config, "fs_cmim_n_bins", 8) or 8),
            fcbf_n_bins=int(getattr(self.config, "fs_fcbf_n_bins", 8) or 8),
            ipss_path_grid_size=self.config.fs_ipss_path_grid_size,
            ipss_min_c=self.config.fs_ipss_min_c,
            ipss_max_c=self.config.fs_ipss_max_c,
            ipss_target_fdr=self.config.fs_ipss_target_fdr,
            ipss_null_shuffle_rounds=self.config.fs_ipss_null_shuffle_rounds,
            ipss_use_eats_threshold=self.config.fs_ipss_use_eats_threshold,
            ipss_eats_exclusion_quantile=self.config.fs_ipss_eats_exclusion_quantile,
            ipss_eats_min_threshold=self.config.fs_ipss_eats_min_threshold,
            ipss_importance_model=self.config.fs_ipss_importance_model,
            ipss_gate_min_classes=int(getattr(self.config, "fs_ipss_gate_min_classes", 0) or 0),
            ipss_gate_min_p_over_n=float(getattr(self.config, "fs_ipss_gate_min_p_over_n", 0.0) or 0.0),
            ktsp_k_pairs=16,
            mrmr_redundancy_weight=0.55,
            stability_selection_threshold=0.6,
            stability_threshold_method=str(
                getattr(self.config, "fs_stability_threshold_method", "fixed") or "fixed"
            ),
            stability_target_pfer=float(
                getattr(self.config, "fs_stability_target_pfer", 1.0) or 1.0
            ),
            stability_subsample_fraction=0.5,
            stability_use_loss_guided_validation=self.config.fs_stability_use_loss_guided_validation,
            stability_validation_fraction=self.config.fs_stability_validation_fraction,
            stability_validation_quantile=self.config.fs_stability_validation_quantile,
            stability_validation_min_samples=self.config.fs_stability_validation_min_samples,
            method_timeout_seconds=float(getattr(self.config, "fs_method_timeout_seconds", 0.0) or 0.0),
            linear_svm_max_iter=int(getattr(self.config, "fs_linear_svm_max_iter", 10000) or 10000),
            cluster_stability_corr_threshold=self.config.fs_cluster_stability_corr_threshold,
            cluster_stability_max_per_cluster=self.config.fs_cluster_stability_max_per_cluster,
            cluster_stability_min_cluster_freq=self.config.fs_cluster_stability_min_cluster_freq,
            copula_knockoff_draws=self.config.fs_copula_knockoff_draws,
            copula_alpha_kn=self.config.fs_copula_alpha_kn,
            copula_alpha_ebh=self.config.fs_copula_alpha_ebh,
            copula_truncation_level=self.config.fs_copula_truncation_level,
            copula_generator=str(getattr(self.config, "fs_copula_generator", "copula") or "copula"),
            copula_deepdrk_latent_fraction=float(
                getattr(self.config, "fs_copula_deepdrk_latent_fraction", 0.35) or 0.35
            ),
            copula_deepdrk_noise_scale=float(
                getattr(self.config, "fs_copula_deepdrk_noise_scale", 1.0) or 1.0
            ),
            copula_derandomize_runs=int(copula_derandomize_runs),
            copula_stabilizer_runs=self.config.fs_copula_stabilizer_runs,
            copula_stabilizer_use_ebh=self.config.fs_copula_stabilizer_use_ebh,
            copula_stabilizer_seed_stride=self.config.fs_copula_stabilizer_seed_stride,
            importance_uq_enabled=bool(getattr(self.config, "fs_importance_uq_enabled", False)),
            importance_uq_min_cv_folds=int(getattr(self.config, "fs_importance_uq_min_cv_folds", 3) or 3),
            group_sparse_lasso_alpha=float(
                getattr(self.config, "fs_group_sparse_lasso_alpha", 0.1) or 0.1
            ),
            group_sparse_lasso_distance_threshold=float(
                getattr(self.config, "fs_group_sparse_lasso_distance_threshold", 0.7) or 0.7
            ),
            decorrelated_stability_eps=self.config.fs_decorrelated_stability_eps,
            iterative_pruning_pool_factor=self.config.fs_iterative_pruning_pool_factor,
            iterative_pruning_max_rounds=self.config.fs_iterative_pruning_max_rounds,
            iterative_pruning_min_improvement=self.config.fs_iterative_pruning_min_improvement,
            iterative_pruning_max_cumulative_loss=float(
                getattr(self.config, "fs_iterative_pruning_max_cumulative_loss", 0.02) or 0.02
            ),
            iterative_pruning_redundancy_weight=self.config.fs_iterative_pruning_redundancy_weight,
            iterative_pruning_bounded_prefilter_cap=self.config.fs_iterative_pruning_bounded_prefilter_cap,
            iterative_pruning_bounded_candidate_fraction=self.config.fs_iterative_pruning_bounded_candidate_fraction,
            iterative_pruning_bounded_min_candidates=self.config.fs_iterative_pruning_bounded_min_candidates,
            iterative_pruning_bounded_max_evaluations=self.config.fs_iterative_pruning_bounded_max_evaluations,
            iterative_pruning_bounded_max_runtime_seconds=self.config.fs_iterative_pruning_bounded_max_runtime_seconds,
            iterative_pruning_bounded_enable_class_gating=self.config.fs_iterative_pruning_bounded_enable_class_gating,
            iterative_pruning_bounded_multiclass_scale=self.config.fs_iterative_pruning_bounded_multiclass_scale,
            iterative_pruning_bounded_imbalance_trigger=self.config.fs_iterative_pruning_bounded_imbalance_trigger,
            iterative_pruning_bounded_imbalance_scale=self.config.fs_iterative_pruning_bounded_imbalance_scale,
            iterative_pruning_bounded_use_cpss_overlay=bool(
                getattr(self.config, "fs_iterative_pruning_bounded_use_cpss_overlay", False)
            ),
            iterative_pruning_bounded_cpss_pairs=int(
                getattr(self.config, "fs_iterative_pruning_bounded_cpss_pairs", 4) or 4
            ),
            iterative_pruning_bounded_cpss_stability_threshold=float(
                getattr(self.config, "fs_iterative_pruning_bounded_cpss_stability_threshold", 0.60) or 0.60
            ),
            iterative_pruning_bounded_cpss_min_stable_features=int(
                getattr(self.config, "fs_iterative_pruning_bounded_cpss_min_stable_features", 2) or 2
            ),
            iterative_pruning_bounded_cpss_min_jaccard=float(
                getattr(self.config, "fs_iterative_pruning_bounded_cpss_min_jaccard", 0.35) or 0.35
            ),
            iterative_pruning_bounded_cpss_max_score_drop=float(
                getattr(self.config, "fs_iterative_pruning_bounded_cpss_max_score_drop", 0.005) or 0.005
            ),
            iterative_pruning_class_pareto_prefilter_enabled=bool(
                getattr(self.config, "fs_iterative_pruning_class_pareto_prefilter_enabled", False)
            ),
            iterative_pruning_class_pareto_min_classes=int(
                getattr(self.config, "fs_iterative_pruning_class_pareto_min_classes", 3) or 3
            ),
            iterative_pruning_class_pareto_top_per_class=int(
                getattr(self.config, "fs_iterative_pruning_class_pareto_top_per_class", 64) or 64
            ),
            iterative_pruning_class_pareto_global_fraction=float(
                getattr(self.config, "fs_iterative_pruning_class_pareto_global_fraction", 0.40) or 0.40
            ),
            iterative_pruning_class_pareto_minority_boost=float(
                getattr(self.config, "fs_iterative_pruning_class_pareto_minority_boost", 0.50) or 0.50
            ),
            iterative_pruning_class_pareto_stability_gate_enabled=bool(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_gate_enabled", False)
            ),
            iterative_pruning_class_pareto_stability_subsamples=int(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_subsamples", 6) or 6
            ),
            iterative_pruning_class_pareto_stability_fraction=float(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_fraction", 0.70) or 0.70
            ),
            iterative_pruning_class_pareto_stability_threshold=float(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_threshold", 0.55) or 0.55
            ),
            iterative_pruning_class_pareto_stability_min_overlap=float(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_min_overlap", 0.50) or 0.50
            ),
            iterative_pruning_class_pareto_stability_min_stable_features=int(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_min_stable_features", 4) or 4
            ),
            iterative_pruning_class_pareto_stability_fallback_on_failure=bool(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_fallback_on_failure", True)
            ),
            decorrelated_stability_min_max_abs_corr=float(
                getattr(self.config, "fs_decorrelated_stability_min_max_abs_corr", 0.0) or 0.0
            ),
            parallel_n_jobs=int(getattr(self.config, "n_jobs", 1) or 1),
        )

    def _evaluate_selector_candidate(
        self,
        X_fs: np.ndarray,
        y_fs: np.ndarray,
        X_train_full: np.ndarray,
        X_test_full: np.ndarray,
        y_train_full: np.ndarray,
        seed: int,
        enabled_methods: Sequence[str],
        candidate_name: str,
        selector_overrides: Optional[Dict[str, Any]] = None,
        post_df_source_raw_train: Optional[np.ndarray] = None,
        post_df_source_raw_test: Optional[np.ndarray] = None,
        post_df_source_base_train: Optional[np.ndarray] = None,
        post_df_source_base_test: Optional[np.ndarray] = None,
        post_df_source_space: str = "prefilter_raw",
    ) -> Dict[str, Any]:
        seed_seq = np.random.SeedSequence(int(seed))
        child_streams = seed_seq.spawn(5)
        selector_seed = int(child_streams[0].generate_state(1, dtype=np.uint32)[0])
        folding_seed = int(child_streams[1].generate_state(1, dtype=np.uint32)[0])
        stage2_seed = int(child_streams[2].generate_state(1, dtype=np.uint32)[0])
        model_seed = int(child_streams[3].generate_state(1, dtype=np.uint32)[0])
        post_df_seed = int(child_streams[4].generate_state(1, dtype=np.uint32)[0])

        selector = self._build_feature_selector(
            seed=selector_seed,
            enabled_methods=enabled_methods,
            selector_overrides=selector_overrides,
        )
        _, selection_result = selector.fit_transform(
            X_fs,
            y_fs,
            n_final_features=int(self.config.n_final_features),
            return_result_object=True,
        )

        X_train_sel = selector.transform(X_train_full)
        X_test_sel = selector.transform(X_test_full)

        # --- Post-FS feature count safety cap (VAL12_Suggestions) ---
        _fs_cap_applied = False
        _n_train = np.asarray(X_train_sel).shape[0]
        _n_sel = np.asarray(X_train_sel).shape[1]
        _fs_max = int(min(
            max(1, _n_train * self.config.fs_max_selected_features_ratio),
            max(1, self.config.fs_max_selected_features_cap),
        ))
        if _n_sel > _fs_max:
            # Rank features by importance and keep top-k.
            _imp: Dict[int, float] = {}
            if selection_result is not None:
                _imp = dict(getattr(selection_result, "feature_importance_mean", {}) or {})
            if _imp:
                # Importance keys are indices into original feature space;
                # map them to column positions in X_train_sel via selected indices.
                _sel_idx = selector.get_selected_features_indices()
                if _sel_idx is not None:
                    _sel_idx = np.asarray(_sel_idx, dtype=int)
                    _scores = np.array(
                        [float(_imp.get(int(orig), 0.0)) for orig in _sel_idx],
                        dtype=float,
                    )
                else:
                    _scores = np.zeros(_n_sel, dtype=float)
            else:
                # No importance available — use variance as proxy.
                _scores = np.var(np.asarray(X_train_sel, dtype=float), axis=0)
            _keep_cols = np.argsort(_scores)[::-1][:_fs_max]
            _keep_cols = np.sort(_keep_cols)  # preserve original order
            X_train_sel = np.asarray(X_train_sel, dtype=float)[:, _keep_cols]
            X_test_sel = np.asarray(X_test_sel, dtype=float)[:, _keep_cols]
            _fs_cap_applied = True
            logger.warning(
                "Post-FS cap truncation applied (candidate=%s, seed=%d): selected=%d -> kept=%d "
                "(n_train=%d, ratio=%.3f, cap=%d).",
                str(candidate_name),
                int(seed),
                int(_n_sel),
                int(_fs_max),
                int(_n_train),
                float(self.config.fs_max_selected_features_ratio),
                int(self.config.fs_max_selected_features_cap),
            )
        # --- end safety cap ---

        selected_indices = selector.get_selected_features_indices()
        if selected_indices is None:
            selected_indices = np.arange(np.asarray(X_train_sel).shape[1], dtype=int)

        # If the safety cap was applied, remap selected_indices to the kept subset.
        if _fs_cap_applied:
            selected_indices = np.asarray(selected_indices, dtype=int)
            selected_indices = selected_indices[_keep_cols]

        X_train_post_df, X_test_post_df, post_df_summaries, post_df_meta, post_df_time_sec = (
            self._apply_post_selection_distribution_transform(
                X_train_selected=np.asarray(X_train_sel, dtype=float),
                X_test_selected=np.asarray(X_test_sel, dtype=float),
                selected_indices=np.asarray(selected_indices, dtype=int).ravel().tolist(),
                y_train=np.asarray(y_train_full).ravel(),
                seed=int(post_df_seed),
                source_space=str(post_df_source_space or "prefilter_raw"),
                source_raw_train=post_df_source_raw_train,
                source_raw_test=post_df_source_raw_test,
                source_base_train=post_df_source_base_train,
                source_base_test=post_df_source_base_test,
            )
        )

        if self._df_stage_position() == "after_fs":
            X_train_fold, X_test_fold, folding_meta = self._apply_folding_stage(
                X_train_fs_input=np.asarray(X_train_post_df, dtype=float),
                X_test_fs_input=np.asarray(X_test_post_df, dtype=float),
                y_train=np.asarray(y_train_full).ravel(),
                seed=int(folding_seed),
            )
            folding_state = dict(self._last_folding_state or {})
        else:
            X_train_fold = np.asarray(X_train_post_df, dtype=float)
            X_test_fold = np.asarray(X_test_post_df, dtype=float)
            folding_meta = {}
            folding_state = {}

        X_train_stage2, X_test_stage2, stage2_ratio_meta = self._stage2_ratio_augmentation(
            X_train_sel=np.asarray(X_train_fold, dtype=float),
            y_train=np.asarray(y_train_full).ravel(),
            X_test_sel=np.asarray(X_test_fold, dtype=float),
            seed=int(stage2_seed),
        )

        # T-R-268: when extreme multiclass gate fires, temporarily enable vote_ensemble.
        _restore_vote_ensemble: Optional[bool] = None
        if isinstance(selector_overrides, dict) and selector_overrides.get("include_vote_ensemble_model"):
            cls_cfg_ref = self._classification_cfg()
            _restore_vote_ensemble = cls_cfg_ref.include_vote_ensemble_model
            cls_cfg_ref.include_vote_ensemble_model = True

        model, model_name, cv_score, cv_std, cv_n, cv_meta = self._select_model_via_cv_scored(
            X_train_stage2, y_train_full, model_seed
        )

        # Restore the original vote_ensemble setting.
        if _restore_vote_ensemble is not None:
            try:
                cls_cfg_ref.include_vote_ensemble_model = _restore_vote_ensemble
            except Exception:
                pass

        uq_meta = {}
        unstable_indices: List[int] = []
        if selection_result is not None:
            try:
                uq_meta = dict(getattr(selection_result, "importance_uq", {}) or {})
            except Exception as exc:
                uq_meta = {}
            try:
                unstable_indices = [
                    int(i) for i in (getattr(selection_result, "unstable_feature_indices", []) or [])
                ]
            except Exception as exc:
                unstable_indices = []
        unstable_indices = sorted(set(unstable_indices))

        fs_selection_summary: Dict[str, Any] = {}
        fs_diagnostics: Dict[str, Any] = {}
        if selection_result is not None:
            try:
                fs_selection_summary = _json_safe(selection_result.to_summary_dict())
            except Exception as exc:
                fs_selection_summary = {}
            try:
                fs_diagnostics = _json_safe(
                    {
                        "selected_feature_indices": [
                            int(i)
                            for i in np.asarray(
                                getattr(selection_result, "selected_feature_indices", np.array([], dtype=int)),
                                dtype=int,
                            ).ravel().tolist()
                        ],
                        "selected_feature_votes": dict(getattr(selection_result, "selected_feature_votes", {}) or {}),
                        "all_features_info": dict(getattr(selection_result, "all_features_info", {}) or {}),
                        "method_results": dict(getattr(selection_result, "method_results", {}) or {}),
                        "eliminated_features": dict(getattr(selection_result, "eliminated_features", {}) or {}),
                        "feature_importance_mean": dict(
                            getattr(selection_result, "feature_importance_mean", {}) or {}
                        ),
                        "feature_importance_variance": dict(
                            getattr(selection_result, "feature_importance_variance", {}) or {}
                        ),
                        "unstable_feature_indices": [
                            int(i)
                            for i in list(getattr(selection_result, "unstable_feature_indices", []) or [])
                        ],
                        "importance_uq": dict(getattr(selection_result, "importance_uq", {}) or {}),
                        "config": dict(getattr(selection_result, "config", {}) or {}),
                        "mnpo_diagnostics": dict(getattr(selector, "mnpo_diagnostics_", {}) or {}),
                    }
                )
            except Exception as exc:
                fs_diagnostics = {}

        return {
            "candidate_name": str(candidate_name),
            "enabled_methods": tuple(str(m) for m in enabled_methods),
            "X_train_sel": X_train_stage2,
            "X_test_sel": X_test_stage2,
            "selected_indices": tuple(int(i) for i in np.asarray(selected_indices, dtype=int).tolist()),
            "model": model,
            "model_name": str(model_name),
            "model_cv_score": float(cv_score) if np.isfinite(cv_score) else float("nan"),
            "model_cv_score_std": float(cv_std) if np.isfinite(cv_std) else float("nan"),
            "model_cv_score_n_splits": int(cv_n) if int(cv_n) > 0 else 0,
            "model_cv_meta": dict(cv_meta or {}),
            "stage2_ratio_meta": dict(stage2_ratio_meta or {}),
            "seed_schedule": {
                "root_seed": int(seed),
                "selector_seed": int(selector_seed),
                "folding_seed": int(folding_seed),
                "stage2_seed": int(stage2_seed),
                "model_seed": int(model_seed),
                "post_df_seed": int(post_df_seed),
            },
            "selector_overrides_applied": dict(selector_overrides or {}),
            "importance_uq_enabled": bool(uq_meta.get("importance_uq_enabled", False)),
            "importance_uq_computed": bool(uq_meta.get("importance_uq_computed", False)),
            "importance_uq_reason": str(uq_meta.get("importance_uq_reason", "disabled")),
            "importance_uq_n_folds": int(uq_meta.get("importance_uq_n_folds", 0) or 0),
            "importance_uq_unstable_threshold": float(uq_meta.get("unstable_threshold", 0.0) or 0.0),
            "importance_uq_unstable_feature_count": int(len(unstable_indices)),
            "importance_uq_unstable_feature_indices": unstable_indices,
            "fs_selection_summary": dict(fs_selection_summary or {}),
            "fs_diagnostics": dict(fs_diagnostics or {}),
            "fs_cap_applied": _fs_cap_applied,
            "fs_cap_meta": {
                "original_n_selected": int(_n_sel),
                "max_allowed": int(_fs_max),
                "n_train": int(_n_train),
                "ratio": float(self.config.fs_max_selected_features_ratio),
                "cap": int(self.config.fs_max_selected_features_cap),
            } if _fs_cap_applied else {},
            "_fitted_selector": selector,
            "_selection_result": selection_result,
            "_post_df_summaries": list(post_df_summaries or []),
            "_post_df_meta": dict(post_df_meta or {}),
            "_post_df_time_sec": float(post_df_time_sec),
            "_folding_meta": dict(folding_meta or {}),
            "_folding_state": dict(folding_state or {}),
        }

    @staticmethod
    def _fast_univariate_filter(
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        max_k: int = 200,
    ) -> np.ndarray:
        """T-R-269: Fast univariate feature filter for low p/n regime.

        Uses one-way ANOVA F-statistics for each feature (binary and multiclass),
        keeps top min(max_k, p/2) features by F-statistic. O(p) runtime, no
        game-theoretic overhead.
        """
        from scipy import stats as _stats

        X_arr = np.asarray(X_train, dtype=float)
        y_arr = np.asarray(y_train).ravel()
        n_features = X_arr.shape[1] if X_arr.ndim == 2 else 0
        if n_features == 0:
            return np.array([], dtype=int)

        k = int(min(max_k, max(1, n_features // 2)))
        if n_features <= k:
            return np.arange(n_features, dtype=int)

        classes = np.unique(y_arr)
        n_classes = int(classes.size)

        if n_classes < 2:
            return np.arange(min(k, n_features), dtype=int)

        # Compute F-statistics (works for both binary and multiclass)
        try:
            groups = [X_arr[y_arr == c] for c in classes]
            f_stats = np.zeros(n_features, dtype=float)
            for j in range(n_features):
                col_groups = [g[:, j] for g in groups if g.shape[0] > 0]
                # Filter groups with zero variance to avoid NaN
                col_groups = [g for g in col_groups if g.size > 0]
                if len(col_groups) < 2:
                    f_stats[j] = 0.0
                    continue
                try:
                    f_val, _ = _stats.f_oneway(*col_groups)
                    f_stats[j] = float(f_val) if np.isfinite(f_val) else 0.0
                except Exception:
                    f_stats[j] = 0.0
            top_indices = np.argsort(f_stats)[::-1][:k]
            return np.sort(top_indices)
        except Exception:
            # Fallback: keep first k features
            return np.arange(min(k, n_features), dtype=int)

    def _evaluate_selector_bypass_candidate(
        self,
        X_train_full: np.ndarray,
        X_test_full: np.ndarray,
        y_train_full: np.ndarray,
        seed: int,
        enabled_methods: Sequence[str],
        candidate_name: str,
        bypass_mode: str,
        filter_max_k: int = 200,
        post_df_source_raw_train: Optional[np.ndarray] = None,
        post_df_source_raw_test: Optional[np.ndarray] = None,
        post_df_source_base_train: Optional[np.ndarray] = None,
        post_df_source_base_test: Optional[np.ndarray] = None,
        post_df_source_space: str = "prefilter_raw",
    ) -> Dict[str, Any]:
        seed_seq = np.random.SeedSequence(int(seed))
        child_streams = seed_seq.spawn(4)
        folding_seed = int(child_streams[0].generate_state(1, dtype=np.uint32)[0])
        stage2_seed = int(child_streams[1].generate_state(1, dtype=np.uint32)[0])
        model_seed = int(child_streams[2].generate_state(1, dtype=np.uint32)[0])
        post_df_seed = int(child_streams[3].generate_state(1, dtype=np.uint32)[0])

        X_train_sel = np.asarray(X_train_full, dtype=float)
        X_test_sel = np.asarray(X_test_full, dtype=float)

        # T-R-269: fast_univariate_filter mode — one-way ANOVA F-statistics
        # to select informative features instead of keeping all features.
        actual_bypass_mode = str(bypass_mode).strip().lower()
        if actual_bypass_mode == "fast_univariate_filter":
            selected_indices = self._fast_univariate_filter(
                X_train_sel, y_train_full, max_k=int(max(1, filter_max_k)),
            )
            X_train_sel = X_train_sel[:, selected_indices]
            X_test_sel = X_test_sel[:, selected_indices]
        else:
            # Legacy all_features fallback
            selected_indices = np.arange(X_train_sel.shape[1], dtype=int)

        X_train_post_df, X_test_post_df, post_df_summaries, post_df_meta, post_df_time_sec = (
            self._apply_post_selection_distribution_transform(
                X_train_selected=np.asarray(X_train_sel, dtype=float),
                X_test_selected=np.asarray(X_test_sel, dtype=float),
                selected_indices=np.asarray(selected_indices, dtype=int).ravel().tolist(),
                y_train=np.asarray(y_train_full).ravel(),
                seed=int(post_df_seed),
                source_space=str(post_df_source_space or "prefilter_raw"),
                source_raw_train=post_df_source_raw_train,
                source_raw_test=post_df_source_raw_test,
                source_base_train=post_df_source_base_train,
                source_base_test=post_df_source_base_test,
            )
        )

        if self._df_stage_position() == "after_fs":
            X_train_fold, X_test_fold, folding_meta = self._apply_folding_stage(
                X_train_fs_input=np.asarray(X_train_post_df, dtype=float),
                X_test_fs_input=np.asarray(X_test_post_df, dtype=float),
                y_train=np.asarray(y_train_full).ravel(),
                seed=int(folding_seed),
            )
            folding_state = dict(self._last_folding_state or {})
        else:
            X_train_fold = np.asarray(X_train_post_df, dtype=float)
            X_test_fold = np.asarray(X_test_post_df, dtype=float)
            folding_meta = {}
            folding_state = {}

        X_train_stage2, X_test_stage2, stage2_ratio_meta = self._stage2_ratio_augmentation(
            X_train_sel=np.asarray(X_train_fold, dtype=float),
            y_train=np.asarray(y_train_full).ravel(),
            X_test_sel=np.asarray(X_test_fold, dtype=float),
            seed=int(stage2_seed),
        )
        model, model_name, cv_score, cv_std, cv_n, cv_meta = self._select_model_via_cv_scored(
            X_train_stage2, y_train_full, model_seed
        )

        fs_selection_summary = {
            "selection_strategy": "regime_bypass_all_features",
            "regime_policy_bypass_mode": str(bypass_mode),
            "selected_feature_count": int(selected_indices.size),
            "selected_feature_indices": [int(i) for i in selected_indices.tolist()],
            "reason": "low_p_over_n",
        }
        fs_diagnostics = {
            "regime_bypass": True,
            "regime_policy_bypass_mode": str(bypass_mode),
        }

        return {
            "candidate_name": str(candidate_name),
            "enabled_methods": tuple(str(m) for m in enabled_methods),
            "X_train_sel": X_train_stage2,
            "X_test_sel": X_test_stage2,
            "selected_indices": tuple(int(i) for i in selected_indices.tolist()),
            "model": model,
            "model_name": str(model_name),
            "model_cv_score": float(cv_score) if np.isfinite(cv_score) else float("nan"),
            "model_cv_score_std": float(cv_std) if np.isfinite(cv_std) else float("nan"),
            "model_cv_score_n_splits": int(cv_n) if int(cv_n) > 0 else 0,
            "model_cv_meta": dict(cv_meta or {}),
            "stage2_ratio_meta": dict(stage2_ratio_meta or {}),
            "seed_schedule": {
                "root_seed": int(seed),
                "selector_seed": -1,
                "folding_seed": int(folding_seed),
                "stage2_seed": int(stage2_seed),
                "model_seed": int(model_seed),
                "post_df_seed": int(post_df_seed),
            },
            "selector_overrides_applied": {
                "selection_strategy": "regime_bypass_all_features",
            },
            "importance_uq_enabled": False,
            "importance_uq_computed": False,
            "importance_uq_reason": "selector_bypassed",
            "importance_uq_n_folds": 0,
            "importance_uq_unstable_threshold": 0.0,
            "importance_uq_unstable_feature_count": 0,
            "importance_uq_unstable_feature_indices": [],
            "fs_selection_summary": dict(fs_selection_summary),
            "fs_diagnostics": dict(fs_diagnostics),
            "_fitted_selector": _IdentityFeatureSelector(int(X_train_sel.shape[1])),
            "_selection_result": None,
            "_post_df_summaries": list(post_df_summaries or []),
            "_post_df_meta": dict(post_df_meta or {}),
            "_post_df_time_sec": float(post_df_time_sec),
            "_folding_meta": dict(folding_meta or {}),
            "_folding_state": dict(folding_state or {}),
        }

    def _choose_selector_candidate(
        self,
        X_fs: np.ndarray,
        y_fs: np.ndarray,
        X_train_full: np.ndarray,
        X_test_full: np.ndarray,
        y_train_full: np.ndarray,
        seed: int,
        dataset_name: str = "dataset",
        post_df_source_raw_train: Optional[np.ndarray] = None,
        post_df_source_raw_test: Optional[np.ndarray] = None,
        post_df_source_base_train: Optional[np.ndarray] = None,
        post_df_source_base_test: Optional[np.ndarray] = None,
        post_df_source_space: str = "prefilter_raw",
    ) -> Dict[str, Any]:
        """Choose the selector+classifier pairing candidate, defaulting to the configured selector."""
        policy = self._resolve_method_policy(
            dataset_name=dataset_name,
            X_ref=X_train_full,
            y_ref=y_train_full,
        )
        configured_methods = tuple(policy.get("enabled_methods", tuple(self.config.enabled_methods)))
        configured_source = str(policy.get("enabled_methods_source", "config"))
        selector_overrides = dict(policy.get("regime_policy_selector_overrides", {}) or {})
        policy_keys = (
            "tier_policy_applied",
            "tier_policy_mode",
            "tier_policy_target_tier",
            "tier_policy_resolved_tier",
            "tier_policy_source",
            "tier_policy_enabled_methods_before",
            "tier_policy_enabled_methods_after",
            "tier_policy_meta_features",
            "regime_policy_applied",
            "regime_policy_mode",
            "regime_policy_reason",
            "regime_policy_enabled",
            "regime_policy_target_tier",
            "regime_policy_tier",
            "regime_policy_tier_source",
            "regime_policy_tier_meta_features",
            "regime_policy_n_samples",
            "regime_policy_n_features",
            "regime_policy_n_classes",
            "regime_policy_samples_per_class",
            "regime_policy_p_over_n",
            "regime_policy_min_samples_per_class_threshold",
            "regime_policy_low_p_over_n_threshold",
            "regime_policy_trigger_very_hard",
            "regime_policy_trigger_low_p_over_n",
            "regime_policy_enabled_methods_before",
            "regime_policy_enabled_methods_after",
            "regime_policy_enabled_methods_source",
            "regime_policy_bypass_fs",
            "regime_policy_bypass_mode",
            "regime_policy_selector_overrides",
        )

        if bool(policy.get("regime_policy_bypass_fs", False)):
            out = self._evaluate_selector_bypass_candidate(
                X_train_full=X_train_full,
                X_test_full=X_test_full,
                y_train_full=y_train_full,
                seed=seed,
                enabled_methods=configured_methods,
                candidate_name="configured_enabled_methods",
                bypass_mode=str(policy.get("regime_policy_bypass_mode", "fast_univariate_filter") or "fast_univariate_filter"),
                filter_max_k=int(policy.get("regime_policy_low_p_over_n_filter_max_k", 200) or 200),
                post_df_source_raw_train=post_df_source_raw_train,
                post_df_source_raw_test=post_df_source_raw_test,
                post_df_source_base_train=post_df_source_base_train,
                post_df_source_base_test=post_df_source_base_test,
                post_df_source_space=post_df_source_space,
            )
            out["enabled_methods_source"] = configured_source
            for key in policy_keys:
                if key in policy:
                    out[key] = policy[key]
            return out

        # When tier lockout/routing is applied, force the routed stack directly.
        if bool(policy.get("tier_policy_applied", False)):
            out = self._evaluate_selector_candidate(
                X_fs=X_fs,
                y_fs=y_fs,
                X_train_full=X_train_full,
                X_test_full=X_test_full,
                y_train_full=y_train_full,
                seed=seed,
                enabled_methods=configured_methods,
                candidate_name="configured_enabled_methods",
                selector_overrides=selector_overrides,
                post_df_source_raw_train=post_df_source_raw_train,
                post_df_source_raw_test=post_df_source_raw_test,
                post_df_source_base_train=post_df_source_base_train,
                post_df_source_base_test=post_df_source_base_test,
                post_df_source_space=post_df_source_space,
            )
            out["enabled_methods_source"] = configured_source
            for key in policy_keys:
                if key in policy:
                    out[key] = policy[key]
            return out

        if not bool(self.config.enable_maqc_pairing):
            out = self._evaluate_selector_candidate(
                X_fs=X_fs,
                y_fs=y_fs,
                X_train_full=X_train_full,
                X_test_full=X_test_full,
                y_train_full=y_train_full,
                seed=seed,
                enabled_methods=configured_methods,
                candidate_name="configured_enabled_methods",
                selector_overrides=selector_overrides,
                post_df_source_raw_train=post_df_source_raw_train,
                post_df_source_raw_test=post_df_source_raw_test,
                post_df_source_base_train=post_df_source_base_train,
                post_df_source_base_test=post_df_source_base_test,
                post_df_source_space=post_df_source_space,
            )
            out["enabled_methods_source"] = configured_source
            for key in policy_keys:
                if key in policy:
                    out[key] = policy[key]
            return out

        classes, counts = np.unique(np.asarray(y_train_full), return_counts=True)
        if len(classes) < 2 or int(counts.min()) < 2:
            # MAQC pairing relies on downstream CV scores. If CV cannot run due
            # to tiny per-class counts, skip pairing to avoid wasted compute.
            out = self._evaluate_selector_candidate(
                X_fs=X_fs,
                y_fs=y_fs,
                X_train_full=X_train_full,
                X_test_full=X_test_full,
                y_train_full=y_train_full,
                seed=seed,
                enabled_methods=configured_methods,
                candidate_name="configured_enabled_methods",
                selector_overrides=selector_overrides,
                post_df_source_raw_train=post_df_source_raw_train,
                post_df_source_raw_test=post_df_source_raw_test,
                post_df_source_base_train=post_df_source_base_train,
                post_df_source_base_test=post_df_source_base_test,
                post_df_source_space=post_df_source_space,
            )
            out["enabled_methods_source"] = "maqc_pairing_skipped_no_cv"
            out["pairing_meta"] = {
                "maqc_pairing_enabled": False,
                "maqc_pairing_skipped": True,
                "maqc_pairing_skip_reason": "insufficient_class_counts",
                "maqc_pairing_candidate_count": 0,
                "maqc_pairing_evaluated_count": 0,
                "maqc_pairing_failed_count": 0,
            }
            for key in policy_keys:
                if key in policy:
                    out[key] = policy[key]
            return out

        # Build candidate list from config, but always include the configured
        # enabled_methods so pairing can't accidentally exclude the baseline.
        raw_sets = list(self.config.maqc_pairing_method_sets or ())
        raw_names = list(self.config.maqc_pairing_method_set_names or ())
        if len(raw_names) != len(raw_sets):
            raw_names = [f"candidate_{i}" for i in range(len(raw_sets))]

        raw_sets.append(configured_methods)
        raw_names.append("configured_enabled_methods")

        deduped: List[Tuple[str, Tuple[str, ...]]] = []
        seen: Set[Tuple[str, ...]] = set()
        for name, methods in zip(raw_names, raw_sets):
            methods_t = tuple(str(m) for m in methods)
            if not methods_t or methods_t in seen:
                continue
            deduped.append((str(name), methods_t))
            seen.add(methods_t)

        best: Optional[Dict[str, Any]] = None
        configured_candidate: Optional[Dict[str, Any]] = None
        best_score = -np.inf
        evaluated = 0
        failed = 0

        for cand_name, methods in deduped:
            try:
                cand = self._evaluate_selector_candidate(
                    X_fs=X_fs,
                    y_fs=y_fs,
                    X_train_full=X_train_full,
                    X_test_full=X_test_full,
                    y_train_full=y_train_full,
                    seed=seed,
                    enabled_methods=methods,
                    candidate_name=cand_name,
                    selector_overrides=selector_overrides,
                    post_df_source_raw_train=post_df_source_raw_train,
                    post_df_source_raw_test=post_df_source_raw_test,
                    post_df_source_base_train=post_df_source_base_train,
                    post_df_source_base_test=post_df_source_base_test,
                    post_df_source_space=post_df_source_space,
                )
                evaluated += 1
            except Exception as exc:
                failed += 1
                continue

            score = float(cand.get("model_cv_score", float("nan")))
            score_cmp = score if np.isfinite(score) else -np.inf
            if cand_name == "configured_enabled_methods":
                configured_candidate = cand
            if best is None:
                best = cand
                best_score = score_cmp
            elif score_cmp > best_score:
                best = cand
                best_score = score_cmp

        if best is None:
            # Should be rare: all candidates raised exceptions.
            best = self._evaluate_selector_candidate(
                X_fs=X_fs,
                y_fs=y_fs,
                X_train_full=X_train_full,
                X_test_full=X_test_full,
                y_train_full=y_train_full,
                seed=seed,
                enabled_methods=configured_methods,
                candidate_name="configured_enabled_methods",
                selector_overrides=selector_overrides,
                post_df_source_raw_train=post_df_source_raw_train,
                post_df_source_raw_test=post_df_source_raw_test,
                post_df_source_base_train=post_df_source_base_train,
                post_df_source_base_test=post_df_source_base_test,
                post_df_source_space=post_df_source_space,
            )
            configured_candidate = best
            best_score = float(best.get("model_cv_score", float("nan")))
        else:
            # If no candidate produced a finite CV score, default to the configured selector
            # to keep behavior predictable.
            has_finite = False
            if np.isfinite(best_score):
                has_finite = True
            if not has_finite and configured_candidate is not None:
                best = configured_candidate
                best_score = float(best.get("model_cv_score", float("nan")))

        raw_best = best
        raw_best_score = float(raw_best.get("model_cv_score", float("nan")))

        selected = raw_best
        selected_source = "maqc_pairing"
        reverted = False
        revert_reason = ""

        base_mean = float("nan")
        base_std = float("nan")
        base_n = 0

        best_mean = float(raw_best.get("model_cv_score", float("nan")))
        best_std = float(raw_best.get("model_cv_score_std", float("nan")))
        best_n = int(raw_best.get("model_cv_score_n_splits", 0) or 0)

        improvement = float("nan")
        combined_se = float("nan")

        if configured_candidate is not None:
            base_mean = float(configured_candidate.get("model_cv_score", float("nan")))
            base_std = float(configured_candidate.get("model_cv_score_std", float("nan")))
            base_n = int(configured_candidate.get("model_cv_score_n_splits", 0) or 0)

        abs_thr = float(max(0.0, self.config.maqc_pairing_min_improvement))
        se_mult = float(max(0.0, self.config.maqc_pairing_min_improvement_se_mult))

        if (
            configured_candidate is not None
            and raw_best is not configured_candidate
            and (abs_thr > 0.0 or se_mult > 0.0)
            and np.isfinite(best_mean)
            and np.isfinite(base_mean)
        ):
            improvement = float(best_mean - base_mean)

            base_se = float("nan")
            best_se = float("nan")
            if np.isfinite(base_std) and base_n > 1:
                base_se = float(base_std / float(np.sqrt(float(base_n))))
            if np.isfinite(best_std) and best_n > 1:
                best_se = float(best_std / float(np.sqrt(float(best_n))))

            if np.isfinite(base_se) and np.isfinite(best_se):
                combined_se = float(np.sqrt(base_se * base_se + best_se * best_se))
            elif np.isfinite(base_se):
                combined_se = float(base_se)
            elif np.isfinite(best_se):
                combined_se = float(best_se)

            abs_ok = True
            se_ok = True

            if abs_thr > 0.0 and improvement < abs_thr:
                abs_ok = False
                revert_reason = "below_min_improvement"
            if se_mult > 0.0:
                required = float("inf")
                if np.isfinite(combined_se):
                    required = float(se_mult * combined_se)
                if improvement < required:
                    se_ok = False
                    revert_reason = f"{revert_reason}|below_min_improvement_se".strip("|")

            if not abs_ok or not se_ok:
                selected = configured_candidate
                selected_source = "maqc_pairing_reverted"
                reverted = True

        selected_score = float(selected.get("model_cv_score", float("nan")))
        selected["enabled_methods_source"] = str(selected_source)
        selected["pairing_meta"] = {
            "maqc_pairing_enabled": True,
            "maqc_pairing_selected_fs_name": str(selected.get("candidate_name", "")),
            "maqc_pairing_selected_cv_score": float(selected_score) if np.isfinite(selected_score) else float("nan"),
            "maqc_pairing_selected_cv_score_std": float(selected.get("model_cv_score_std", float("nan"))),
            "maqc_pairing_selected_cv_n_splits": int(selected.get("model_cv_score_n_splits", 0) or 0),
            "maqc_pairing_raw_best_fs_name": str(raw_best.get("candidate_name", "")),
            "maqc_pairing_raw_best_cv_score": float(best_mean) if np.isfinite(best_mean) else float("nan"),
            "maqc_pairing_raw_best_cv_score_std": float(best_std) if np.isfinite(best_std) else float("nan"),
            "maqc_pairing_configured_cv_score": float(base_mean) if np.isfinite(base_mean) else float("nan"),
            "maqc_pairing_configured_cv_score_std": float(base_std) if np.isfinite(base_std) else float("nan"),
            "maqc_pairing_improvement": float(improvement) if np.isfinite(improvement) else float("nan"),
            "maqc_pairing_improvement_se": float(combined_se) if np.isfinite(combined_se) else float("nan"),
            "maqc_pairing_min_improvement": float(abs_thr),
            "maqc_pairing_min_improvement_se_mult": float(se_mult),
            "maqc_pairing_reverted": bool(reverted),
            "maqc_pairing_revert_reason": str(revert_reason),
            "maqc_pairing_candidate_count": int(len(deduped)),
            "maqc_pairing_evaluated_count": int(evaluated),
            "maqc_pairing_failed_count": int(failed),
        }
        for key in policy_keys:
            if key in policy:
                selected[key] = policy[key]
        return selected

    def _split_indices(self, idx_all: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        y_arr = np.asarray(y)
        n_total = int(y_arr.shape[0])

        # Determine split policy.
        #
        # If max_train_samples is set, honor it as an absolute cap on the train
        # split size (Artificial HDLSS protocol). In cap mode, keep at least
        # 20% of samples for test (80/20 minimum holdout) and use the
        # remainder as test.
        forced_train_n: Optional[int] = None
        if self.config.max_train_samples is not None and self.config.max_train_samples > 0 and n_total >= 4:
            train_n = int(self.config.max_train_samples)
            train_n = int(max(2, train_n))
            min_test_n = int(max(2, np.ceil(0.20 * float(n_total))))
            train_n = int(min(train_n, n_total - min_test_n))
            forced_train_n = int(train_n)
        else:
            # Standard bounds check for ratio-based splitting.
            test_size = float(max(0.05, min(0.95, float(self.config.test_size))))

        classes, counts = np.unique(y_arr, return_counts=True)

        split_kwargs: Dict[str, Any] = {"random_state": seed}
        if forced_train_n is not None:
            split_kwargs["train_size"] = int(forced_train_n)
        else:
            split_kwargs["test_size"] = float(test_size)

        if len(classes) >= 2 and counts.min() >= 2:
            try:
                train_idx, test_idx = train_test_split(
                    idx_all,
                    stratify=y_arr,
                    **split_kwargs,
                )
            except ValueError:
                # Stratified splitting can fail when forced_train_n is too small
                # relative to the number of classes; fall back to unstratified.
                train_idx, test_idx = train_test_split(
                    idx_all,
                    **split_kwargs,
                )
        else:
            train_idx, test_idx = train_test_split(idx_all, **split_kwargs)

        return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)

    @staticmethod
    def _sample_fs_indices(
        y_train: np.ndarray,
        fs_fraction: float,
        seed: int,
        use_balanced: bool = True,
        min_per_class: int = 2,
    ) -> np.ndarray:
        y_arr = np.asarray(y_train)
        fs_fraction = float(max(0.05, min(1.0, fs_fraction)))
        n = int(y_arr.shape[0])
        if fs_fraction >= 0.999:
            return np.arange(n, dtype=int)

        n_fs = int(max(2, round(fs_fraction * n)))
        classes, counts = np.unique(y_arr, return_counts=True)
        rng = np.random.default_rng(seed)

        if use_balanced and len(classes) >= 2:
            # Guarantee at least one per class if possible, or up to min_per_class
            min_per_class = int(max(1, min_per_class))
            
            # Pass 1: Ensure absolute minimum coverage (1 per class)
            chosen: List[int] = []
            pool_indices: List[int] = []
            
            # Group indices by class
            class_indices: Dict[int, List[int]] = {}
            for cls in classes:
                cls_idx = np.where(y_arr == cls)[0]
                cls_idx = np.asarray(cls_idx, dtype=int)
                rng.shuffle(cls_idx)
                class_indices[cls] = cls_idx.tolist()
            
            # Take 1 from each class first (Hard Constraint)
            for cls in classes:
                indices = class_indices[cls]
                if indices:
                    chosen.append(indices.pop(0))
            
            # Pass 2: Satisfy min_per_class (if > 1)
            if min_per_class > 1:
                for cls in classes:
                    indices = class_indices[cls]
                    current_count = 1  # We already took 1
                    while current_count < min_per_class and indices and len(chosen) < n_fs:
                        chosen.append(indices.pop(0))
                        current_count += 1
            
            # Pass 3: Fill remaining quota with stratified probability (proportional to remaining size)
            # Collect all remaining available indices
            for cls in classes:
                pool_indices.extend(class_indices[cls])
            
            remaining_slots = int(max(0, n_fs - len(chosen)))
            if remaining_slots > 0 and pool_indices:
                rng.shuffle(pool_indices) # Shuffle the remaining pool for random fill
                chosen.extend(pool_indices[:remaining_slots])

            if len(chosen) >= 2:
                return np.asarray(sorted(set(chosen)), dtype=int)

        if len(classes) >= 2 and counts.min() >= 2:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=fs_fraction, random_state=seed)
            try:
                idx, _ = next(splitter.split(np.zeros((n, 1)), y_arr))
                return np.asarray(idx, dtype=int)
            except ValueError:
                pass

        idx = rng.choice(np.arange(n), size=n_fs, replace=False)
        return np.asarray(idx, dtype=int)

    def _select_model_via_cv(self, X_train: np.ndarray, y_train: np.ndarray, seed: int):
        model, model_name, _, _, _, _ = self._select_model_via_cv_scored(
            X_train=X_train, y_train=y_train, seed=seed
        )
        return model, model_name

    def _classification_cfg(self) -> ClassificationConfig:
        cfg = getattr(self.config, "classification", None)
        if isinstance(cfg, ClassificationConfig):
            return cfg
        return ClassificationConfig(
            selection_mode=str(getattr(self.config, "classification_selection_mode", "legacy") or "legacy"),
            backend=str(getattr(self.config, "classification_backend", "sklearn") or "sklearn"),
            flaml_time_budget=int(getattr(self.config, "flaml_time_budget", 60) or 60),
            optuna_time_budget=int(getattr(self.config, "optuna_time_budget", 120) or 120),
            optuna_n_trials=int(getattr(self.config, "optuna_n_trials", 25) or 25),
            model_candidates=tuple(getattr(self.config, "model_candidates", ("lr", "svm_rbf"))),
            include_elastic_net_model=bool(getattr(self.config, "include_elastic_net_model", False)),
            include_rf_model=bool(getattr(self.config, "include_rf_model", False)),
            include_knn_model=bool(getattr(self.config, "include_knn_model", False)),
            include_svm_linear_model=bool(getattr(self.config, "include_svm_linear_model", False)),
            include_dlda_model=bool(getattr(self.config, "include_dlda_model", False)),
            include_nsc_model=bool(getattr(self.config, "include_nsc_model", False)),
            include_pls_da_model=bool(getattr(self.config, "include_pls_da_model", False)),
            include_gpc_model=bool(getattr(self.config, "include_gpc_model", False)),
            include_nb_model=bool(getattr(self.config, "include_nb_model", False)),
            include_vote_ensemble_model=bool(getattr(self.config, "include_vote_ensemble_model", False)),
            include_rp_ensemble_model=bool(getattr(self.config, "include_rp_ensemble_model", False)),
            include_dbda_model=bool(getattr(self.config, "include_dbda_model", False)),
            include_gqda_model=bool(getattr(self.config, "include_gqda_model", False)),
            include_bc_svm_linear_model=bool(getattr(self.config, "include_bc_svm_linear_model", False)),
            include_sglnn_model=bool(getattr(self.config, "include_sglnn_model", False)),
            include_xgb_model=bool(getattr(self.config, "include_xgb_model", False)),
            include_lgbm_model=bool(getattr(self.config, "include_lgbm_model", False)),
            include_extra_tree_model=bool(getattr(self.config, "include_extra_tree_model", False)),
            include_catboost_model=bool(getattr(self.config, "include_catboost_model", False)),
            include_tabpfn_model=bool(getattr(self.config, "include_tabpfn_model", False)),
            lr_max_iter=int(getattr(self.config, "model_cv_lr_max_iter", 10000) or 10000),
            use_hybrid_score=bool(getattr(self.config, "model_cv_use_hybrid_score", False)),
            hybrid_balanced_weight=float(getattr(self.config, "model_cv_balanced_weight", 0.6) or 0.6),
            hybrid_macro_f1_weight=float(getattr(self.config, "model_cv_macro_f1_weight", 0.4) or 0.4),
            runtime_containment_enabled=bool(
                getattr(self.config, "model_cv_runtime_containment_enabled", False)
            ),
            runtime_max_candidates=int(getattr(self.config, "model_cv_runtime_max_candidates", 0) or 0),
            runtime_high_p_over_n_threshold=float(
                getattr(self.config, "model_cv_runtime_high_p_over_n_threshold", 40.0) or 40.0
            ),
            runtime_high_class_threshold=int(
                getattr(self.config, "model_cv_runtime_high_class_threshold", 6) or 6
            ),
            runtime_min_class_count_threshold=int(
                getattr(self.config, "model_cv_runtime_min_class_count_threshold", 12) or 12
            ),
            oracle_k=int(getattr(self.config, "classifier_oracle_k", 1) or 1),
            oracle_weighting_mode=str(
                getattr(self.config, "classifier_oracle_weighting_mode", "tritrust") or "tritrust"
            ),
            oracle_include_robustness=bool(
                getattr(self.config, "classifier_oracle_include_robustness", True)
            ),
            oracle_include_complexity=bool(
                getattr(self.config, "classifier_oracle_include_complexity", True)
            ),
            oracle_include_calibration=bool(
                getattr(self.config, "classifier_oracle_include_calibration", True)
            ),
            oracle_include_james_stein=bool(
                getattr(self.config, "classifier_oracle_include_james_stein", True)
            ),
            oracle_include_cvar=bool(
                getattr(self.config, "classifier_oracle_include_cvar", False)
            ),
            oracle_cvar_alpha=float(
                getattr(self.config, "classifier_oracle_cvar_alpha", 0.33) or 0.33
            ),
            oracle_use_dynamic_complexity=bool(
                getattr(self.config, "classifier_oracle_use_dynamic_complexity", False)
            ),
            oracle_portfolio_diversity=bool(
                getattr(self.config, "classifier_oracle_portfolio_diversity", False)
            ),
            oracle_portfolio_overlap_threshold=float(
                getattr(self.config, "classifier_oracle_portfolio_overlap_threshold", 0.75) or 0.75
            ),
            oracle_portfolio_corr_threshold=float(
                getattr(self.config, "classifier_oracle_portfolio_corr_threshold", 0.85) or 0.85
            ),
            oracle_enable_hoeffding_racing=bool(
                getattr(self.config, "classifier_oracle_enable_hoeffding_racing", True)
            ),
            oracle_hoeffding_delta=float(
                getattr(self.config, "classifier_oracle_hoeffding_delta", 0.10) or 0.10
            ),
            oracle_enable_bbc=bool(getattr(self.config, "classifier_oracle_enable_bbc", True)),
            oracle_bbc_bootstrap_rounds=int(
                getattr(self.config, "classifier_oracle_bbc_bootstrap_rounds", 200) or 200
            ),
            oracle_bbc_ci_level=float(
                getattr(self.config, "classifier_oracle_bbc_ci_level", 0.90) or 0.90
            ),
            oracle_enable_ensemble=bool(
                getattr(self.config, "classifier_oracle_enable_ensemble", False)
            ),
            oracle_use_per_family_flaml=bool(
                getattr(self.config, "classifier_oracle_use_per_family_flaml", True)
            ),
            stage2_ratio_augmentation_enabled=bool(
                getattr(self.config, "stage2_ratio_augmentation_enabled", False)
            ),
            stage2_ratio_max_features=int(
                getattr(self.config, "stage2_ratio_max_features", 16) or 16
            ),
            stage2_ratio_selection_method=str(
                getattr(self.config, "stage2_ratio_selection_method", "correlation") or "correlation"
            ),
            stage2_ratio_epsilon=float(
                getattr(self.config, "stage2_ratio_epsilon", 1e-6) or 1e-6
            ),
            conformal_enabled=bool(
                getattr(self.config, "classifier_conformal_enabled", False)
            ),
            conformal_alpha=float(
                getattr(self.config, "classifier_conformal_alpha", 0.10) or 0.10
            ),
            conformal_calibration_fraction=float(
                getattr(self.config, "classifier_conformal_calibration_fraction", 0.25) or 0.25
            ),
            conformal_min_calibration=int(
                getattr(self.config, "classifier_conformal_min_calibration", 20) or 20
            ),
            conformal_output_sets=bool(
                getattr(self.config, "classifier_conformal_output_sets", False)
            ),
        )

    def _apply_model_cv_runtime_containment(
        self,
        candidate_names: Sequence[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[List[str], Dict[str, Any]]:
        cls_cfg = self._classification_cfg()
        requested = [str(name) for name in candidate_names if str(name)]
        X_arr = np.asarray(X_train, dtype=float)
        y_arr = np.asarray(y_train)

        n_train = int(X_arr.shape[0]) if X_arr.ndim == 2 else int(y_arr.shape[0])
        n_features = int(X_arr.shape[1]) if X_arr.ndim == 2 and X_arr.size > 0 else 0
        p_over_n = float(n_features / max(1, n_train))
        classes, counts = np.unique(y_arr, return_counts=True)
        n_classes = int(classes.size)
        min_class_count = int(counts.min()) if counts.size > 0 else 0

        enabled = bool(cls_cfg.runtime_containment_enabled)
        high_p_over_n = p_over_n >= float(max(0.0, cls_cfg.runtime_high_p_over_n_threshold))
        high_class = n_classes >= int(max(2, cls_cfg.runtime_high_class_threshold))
        sparse_class = (
            min_class_count > 0
            and min_class_count <= int(max(1, cls_cfg.runtime_min_class_count_threshold))
        )

        max_candidates_cfg = int(cls_cfg.runtime_max_candidates)
        auto_cap = len(requested)
        if high_p_over_n and high_class:
            auto_cap = 3
        elif high_p_over_n or high_class or sparse_class:
            auto_cap = 4
        target_cap = max_candidates_cfg if max_candidates_cfg > 0 else auto_cap
        target_cap = int(max(1, min(target_cap, len(requested) if requested else 1)))

        meta: Dict[str, Any] = {
            "model_cv_runtime_containment_enabled": bool(enabled),
            "model_cv_runtime_containment_applied": False,
            "model_cv_runtime_containment_reason": "disabled" if not enabled else "not_needed",
            "model_cv_requested_candidates": tuple(requested),
            "model_cv_effective_candidates": tuple(requested),
            "model_cv_dropped_candidates": tuple(),
            "model_cv_runtime_cap": int(target_cap),
            "model_cv_runtime_max_candidates_cfg": int(max_candidates_cfg),
            "model_cv_runtime_p_over_n": float(p_over_n),
            "model_cv_runtime_n_classes": int(n_classes),
            "model_cv_runtime_min_class_count": int(min_class_count),
            "model_cv_runtime_regime_high_p_over_n": bool(high_p_over_n),
            "model_cv_runtime_regime_high_class": bool(high_class),
            "model_cv_runtime_regime_sparse_class": bool(sparse_class),
        }

        if not requested:
            meta["model_cv_effective_candidates"] = ("lr",)
            return ["lr"], meta

        if (not enabled) or target_cap >= len(requested):
            return list(requested), meta

        keep: List[str] = []
        for anchor in ("lr", "svm_rbf"):
            if anchor in requested and anchor not in keep:
                keep.append(anchor)

        priority = (
            "dlda",
            "shrinkage_lda",
            "nsc",
            "rp_ensemble",
            "pls_da_classifier",
            "nb",
            "svm_linear",
            "knn",
            "vote_ensemble",
            "elastic_net_lr",
            "rf",
            "lgbm",
            "extra_tree",
            "catboost",
            "gpc",
            "xgb",
            "tabpfn",
        )
        for name in priority:
            if len(keep) >= target_cap:
                break
            if name in requested and name not in keep:
                keep.append(name)

        for name in requested:
            if len(keep) >= target_cap:
                break
            if name not in keep:
                keep.append(name)

        dropped = [name for name in requested if name not in keep]
        reason_parts = []
        if high_p_over_n:
            reason_parts.append("high_p_over_n")
        if high_class:
            reason_parts.append("high_class_count")
        if sparse_class:
            reason_parts.append("sparse_class_count")
        if not reason_parts:
            reason_parts.append("hard_cap")

        meta["model_cv_runtime_containment_applied"] = bool(len(dropped) > 0)
        meta["model_cv_runtime_containment_reason"] = "+".join(reason_parts)
        meta["model_cv_effective_candidates"] = tuple(keep)
        meta["model_cv_dropped_candidates"] = tuple(dropped)
        return keep, meta

    def _make_sklearn_backend(self, candidate_names: Sequence[str]) -> SklearnBackend:
        cls_cfg = self._classification_cfg()
        return SklearnBackend(
            candidate_names=tuple(candidate_names),
            lr_max_iter=int(cls_cfg.lr_max_iter),
            use_hybrid_score=bool(cls_cfg.use_hybrid_score),
            hybrid_balanced_weight=float(cls_cfg.hybrid_balanced_weight),
            hybrid_macro_f1_weight=float(cls_cfg.hybrid_macro_f1_weight),
            max_train_test_gap=float(cls_cfg.stage2_max_train_test_gap),
            tree_complexity_penalty_enabled=bool(cls_cfg.stage2_tree_complexity_penalty_enabled),
            tree_complexity_penalty_strength=float(cls_cfg.stage2_tree_complexity_penalty_strength),
            n_jobs=int(getattr(cls_cfg, 'n_jobs', 1) or 1),
            build_xgb_model_fn=self._build_xgb_model,
            build_tabpfn_model_fn=self._build_tabpfn_model,
            warn_missing_backend_fn=self._warn_missing_model_backend,
        )

    def _get_classifier_backend(self, *, candidate_names: Sequence[str]) -> Tuple[ClassifierBackend, Dict[str, Any]]:
        cls_cfg = self._classification_cfg()
        selection_mode = str(cls_cfg.selection_mode or "legacy").strip().lower()
        backend_name = str(cls_cfg.backend or "sklearn").strip().lower()
        meta: Dict[str, Any] = {
            "classification_selection_mode": str(selection_mode),
            "classification_backend_requested": backend_name,
            "classification_backend_used": backend_name if selection_mode == "legacy" else "mnpo_hybrid",
        }

        base_optuna_budget = int(max(1, cls_cfg.optuna_time_budget))
        adjusted_optuna_budget = int(base_optuna_budget)
        base_budget = int(max(1, cls_cfg.flaml_time_budget))
        adjusted_budget = int(base_budget)
        if bool(getattr(self.config, "enable_maqc_pairing", False)):
            n_candidates = int(max(1, len(candidate_names)))
            raw_budget = float(base_budget) / float(n_candidates)
            meta["classification_flaml_time_budget_raw"] = int(base_budget)
            meta["classification_flaml_budget_divisor"] = int(n_candidates)
            meta["classification_flaml_time_budget_divided"] = float(raw_budget)
            adjusted_budget = int(max(1, raw_budget))

            raw_optuna_budget = float(base_optuna_budget) / float(n_candidates)
            meta["classification_optuna_time_budget_raw"] = int(base_optuna_budget)
            meta["classification_optuna_budget_divisor"] = int(n_candidates)
            meta["classification_optuna_time_budget_divided"] = float(raw_optuna_budget)
            adjusted_optuna_budget = int(max(1, raw_optuna_budget))

        # Phase 6: new hybrid selector path (regime-gated MNPO classifier oracle).
        if selection_mode in {"mnpo_hybrid", "tune_first"}:
            use_per_family_flaml = bool(getattr(cls_cfg, "oracle_use_per_family_flaml", True))
            tune_first_enabled = selection_mode == "tune_first"
            if backend_name != "flaml":
                use_per_family_flaml = False
                meta["classification_backend_fallback_reason"] = "mnpo_per_family_flaml_disabled_backend_not_flaml"
            if use_per_family_flaml and adjusted_budget < 15:
                warnings.warn(
                    "MNPO hybrid: per-family FLAML disabled because divided time budget fell below 15s floor.",
                    RuntimeWarning,
                )
                use_per_family_flaml = False
                meta["classification_backend_fallback_reason"] = "mnpo_flaml_budget_below_floor"
            if tune_first_enabled and not use_per_family_flaml:
                meta["classification_selection_mode_effective"] = "mnpo_hybrid"
                meta.setdefault(
                    "classification_selection_mode_fallback_reason",
                    "tune_first_requires_per_family_flaml",
                )

            backend = MNPOClassifierBackend(
                candidate_names=tuple(candidate_names),
                exclude_candidate_names=tuple(getattr(cls_cfg, "exclude_model_candidates", tuple()) or tuple()),
                regime_candidate_exclusions=tuple(
                    getattr(cls_cfg, "regime_candidate_exclusions", tuple()) or tuple()
                ),
                oracle_complexity_prior_overrides=tuple(
                    getattr(cls_cfg, "oracle_complexity_prior_overrides", tuple()) or tuple()
                ),
                oracle_k=int(cls_cfg.oracle_k),
                oracle_weighting_mode=str(cls_cfg.oracle_weighting_mode),
                oracle_include_robustness=bool(cls_cfg.oracle_include_robustness),
                oracle_include_complexity=bool(cls_cfg.oracle_include_complexity),
                oracle_include_calibration=bool(cls_cfg.oracle_include_calibration),
                oracle_include_james_stein=bool(cls_cfg.oracle_include_james_stein),
                oracle_include_cvar=bool(getattr(cls_cfg, "oracle_include_cvar", False)),
                oracle_cvar_alpha=float(getattr(cls_cfg, "oracle_cvar_alpha", 0.33) or 0.33),
                oracle_use_dynamic_complexity=bool(
                    getattr(cls_cfg, "oracle_use_dynamic_complexity", False)
                ),
                oracle_portfolio_diversity=bool(
                    getattr(cls_cfg, "oracle_portfolio_diversity", False)
                ),
                oracle_portfolio_overlap_threshold=float(
                    getattr(cls_cfg, "oracle_portfolio_overlap_threshold", 0.75) or 0.75
                ),
                oracle_portfolio_corr_threshold=float(
                    getattr(cls_cfg, "oracle_portfolio_corr_threshold", 0.85) or 0.85
                ),
                enable_hoeffding_racing=bool(cls_cfg.oracle_enable_hoeffding_racing),
                hoeffding_delta=float(cls_cfg.oracle_hoeffding_delta),
                enable_bbc=bool(cls_cfg.oracle_enable_bbc),
                bbc_bootstrap_rounds=int(cls_cfg.oracle_bbc_bootstrap_rounds),
                bbc_ci_level=float(cls_cfg.oracle_bbc_ci_level),
                enable_ensemble=bool(cls_cfg.oracle_enable_ensemble),
                multiclass_ensemble_threshold=int(getattr(cls_cfg, 'oracle_multiclass_ensemble_threshold', 4)),
                ensemble_voting_mode=str(getattr(cls_cfg, "oracle_ensemble_voting_mode", "hard") or "hard"),
                greedy_ensemble=bool(getattr(cls_cfg, "oracle_greedy_ensemble", False)),
                greedy_ensemble_rounds=int(getattr(cls_cfg, "oracle_greedy_ensemble_rounds", 10) or 10),
                candidate_pruning=bool(getattr(cls_cfg, "oracle_candidate_pruning", False)),
                candidate_pruning_threshold=float(getattr(cls_cfg, "oracle_candidate_pruning_threshold", 0.0) or 0.0),
                incumbent_early_stopping=bool(getattr(cls_cfg, "oracle_incumbent_early_stopping", False)),
                oracle_behavior_profile=str(getattr(cls_cfg, "oracle_behavior_profile", "current") or "current"),
                flaml_time_budget=int(max(1, adjusted_budget)),
                flaml_metric=str(cls_cfg.flaml_metric),
                flaml_n_jobs=1,
                use_per_family_flaml=bool(use_per_family_flaml),
                tune_first=bool(tune_first_enabled and use_per_family_flaml),
                min_n_for_automl=int(cls_cfg.min_n_for_automl),
                min_n_per_class_for_automl=int(cls_cfg.min_n_per_class_for_automl),
                min_n_per_class_for_cv=int(cls_cfg.min_n_per_class_for_cv),
                max_p_over_n_for_automl=int(cls_cfg.max_p_over_n_for_automl),
                lr_max_iter=int(cls_cfg.lr_max_iter),
                use_hybrid_score=bool(cls_cfg.use_hybrid_score),
                hybrid_balanced_weight=float(cls_cfg.hybrid_balanced_weight),
                hybrid_macro_f1_weight=float(cls_cfg.hybrid_macro_f1_weight),
                n_jobs=int(getattr(cls_cfg, 'n_jobs', 1) or 1),
                build_xgb_model_fn=self._build_xgb_model,
                build_tabpfn_model_fn=self._build_tabpfn_model,
                warn_missing_backend_fn=self._warn_missing_model_backend,
            )
            meta["classification_backend_used"] = "mnpo_hybrid"
            meta["classification_flaml_time_budget_effective"] = int(max(1, adjusted_budget))
            meta["classification_mnpo_use_per_family_flaml"] = bool(use_per_family_flaml)
            meta["classification_mnpo_oracle_k"] = int(cls_cfg.oracle_k)
            meta["classification_tune_first_enabled"] = bool(tune_first_enabled and use_per_family_flaml)
            return backend, meta

        if backend_name != "flaml":
            if backend_name == "optuna":
                backend = OptunaBackend(
                    candidate_names=tuple(candidate_names),
                    time_budget=int(adjusted_optuna_budget),
                    n_trials=int(cls_cfg.optuna_n_trials),
                    min_n_for_automl=int(cls_cfg.min_n_for_automl),
                    min_n_per_class_for_cv=int(cls_cfg.min_n_per_class_for_cv),
                    min_n_per_class_for_automl=int(cls_cfg.min_n_per_class_for_automl),
                    max_p_over_n_for_automl=int(cls_cfg.max_p_over_n_for_automl),
                    lr_max_iter=int(cls_cfg.lr_max_iter),
                    use_hybrid_score=bool(cls_cfg.use_hybrid_score),
                    hybrid_balanced_weight=float(cls_cfg.hybrid_balanced_weight),
                    hybrid_macro_f1_weight=float(cls_cfg.hybrid_macro_f1_weight),
                    n_jobs=int(getattr(cls_cfg, 'n_jobs', 1) or 1),
                    build_xgb_model_fn=self._build_xgb_model,
                    build_tabpfn_model_fn=self._build_tabpfn_model,
                    warn_missing_backend_fn=self._warn_missing_model_backend,
                )
                meta["classification_optuna_time_budget_effective"] = int(adjusted_optuna_budget)
                meta["classification_optuna_n_trials"] = int(cls_cfg.optuna_n_trials)
                return backend, meta
            return self._make_sklearn_backend(candidate_names), meta

        if bool(getattr(self.config, "enable_maqc_pairing", False)) and float(
            meta.get("classification_flaml_time_budget_divided", 0.0)
        ) < 15.0:
            warnings.warn(
                "FLAML backend skipped: divided time budget below 15s floor under MAQC pairing.",
                RuntimeWarning,
            )
            meta["classification_backend_used"] = "sklearn"
            meta["classification_backend_fallback_reason"] = "flaml_budget_below_floor"
            return self._make_sklearn_backend(candidate_names), meta
        adjusted_budget = int(max(15, adjusted_budget))

        backend = FLAMLBackend(
            time_budget=int(adjusted_budget),
            estimator_list=tuple(cls_cfg.flaml_estimator_list),
            metric=str(cls_cfg.flaml_metric),
            min_n_for_automl=int(cls_cfg.min_n_for_automl),
            min_n_per_class_for_cv=int(cls_cfg.min_n_per_class_for_cv),
            min_n_per_class_for_automl=int(cls_cfg.min_n_per_class_for_automl),
            max_p_over_n_for_automl=int(cls_cfg.max_p_over_n_for_automl),
        )
        meta["classification_flaml_time_budget_effective"] = int(adjusted_budget)
        return backend, meta

    def _stage2_ratio_augmentation(
        self,
        X_train_sel: np.ndarray,
        y_train: np.ndarray,
        X_test_sel: np.ndarray,
        *,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Optional post-FS ratio augmentation applied at Stage-2 model training."""
        cls_cfg = self._classification_cfg()
        enabled = bool(getattr(cls_cfg, "stage2_ratio_augmentation_enabled", False))
        method = str(getattr(cls_cfg, "stage2_ratio_selection_method", "correlation") or "correlation").strip().lower()
        if method not in {"correlation", "ktsp"}:
            method = "correlation"

        meta: Dict[str, Any] = {
            "stage2_ratio_augmentation_enabled": bool(enabled),
            "stage2_ratio_features_applied": False,
            "stage2_ratio_features_reason": "disabled",
            "stage2_ratio_selection_method": str(method),
            "stage2_ratio_max_features": int(max(0, getattr(cls_cfg, "stage2_ratio_max_features", 0) or 0)),
            "stage2_ratio_epsilon": float(getattr(cls_cfg, "stage2_ratio_epsilon", 1e-6) or 1e-6),
            "stage2_ratio_pool_size_effective": 0,
            "stage2_ratio_pairs_considered": 0,
            "stage2_ratio_features_added": 0,
            "stage2_ratio_feature_start_index": int(X_train_sel.shape[1]) if np.asarray(X_train_sel).ndim == 2 else 0,
            "stage2_ratio_pairs": [],
        }
        if not enabled:
            return np.asarray(X_train_sel, dtype=float), np.asarray(X_test_sel, dtype=float), meta

        x_tr = np.asarray(X_train_sel, dtype=float)
        x_te = np.asarray(X_test_sel, dtype=float)
        y_arr = np.asarray(y_train).ravel()
        if x_tr.ndim != 2 or x_te.ndim != 2:
            meta["stage2_ratio_features_reason"] = "invalid_input_rank"
            return x_tr, x_te, meta
        if x_tr.shape[1] != x_te.shape[1]:
            meta["stage2_ratio_features_reason"] = "feature_dim_mismatch"
            return x_tr, x_te, meta
        if x_tr.shape[1] < 2:
            meta["stage2_ratio_features_reason"] = "insufficient_features"
            return x_tr, x_te, meta
        max_ratio = int(max(0, meta["stage2_ratio_max_features"]))
        if max_ratio <= 0:
            meta["stage2_ratio_features_reason"] = "max_ratio_features_zero"
            return x_tr, x_te, meta

        eps = float(max(1e-12, float(meta["stage2_ratio_epsilon"])))
        n_features = int(x_tr.shape[1])
        pool_size = int(max(2, min(n_features, max(8, 4 * max_ratio))))

        mi = self._safe_mi(x_tr, y_arr, seed)
        fs = self._safe_fscore(x_tr, y_arr)
        svm = self._safe_linear_svm_scores(x_tr, y_arr, seed)
        combined = 0.45 * self._normalize01(mi) + 0.35 * self._normalize01(fs) + 0.20 * self._normalize01(svm)
        order = np.argsort(np.asarray(combined, dtype=float))[::-1]
        pool_idx = np.asarray(order[:pool_size], dtype=int).ravel()
        pool_idx = np.array(sorted(set(int(i) for i in pool_idx if 0 <= int(i) < n_features)), dtype=int)
        meta["stage2_ratio_pool_size_effective"] = int(pool_idx.size)
        if pool_idx.size < 2:
            meta["stage2_ratio_features_reason"] = "insufficient_pool"
            return x_tr, x_te, meta

        pair_candidates = list(combinations(range(int(pool_idx.size)), 2))
        max_pairs = int(min(len(pair_candidates), max(256, 256 * max_ratio)))
        if len(pair_candidates) > max_pairs:
            rng = np.random.default_rng(int(seed) + 19331)
            sampled = rng.choice(len(pair_candidates), size=max_pairs, replace=False)
            pair_candidates = [pair_candidates[int(i)] for i in sampled]
        meta["stage2_ratio_pairs_considered"] = int(len(pair_candidates))

        feature_pair_scores: List[Tuple[float, int, int]] = []
        if method == "correlation":
            x_pool = x_tr[:, pool_idx]
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(x_pool, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            for a_local, b_local in pair_candidates:
                a = int(pool_idx[a_local])
                b = int(pool_idx[b_local])
                feature_pair_scores.append((float(abs(corr[a_local, b_local])), a, b))
        else:
            classes = np.unique(y_arr)
            if classes.size < 2:
                meta["stage2_ratio_features_reason"] = "insufficient_classes"
                return x_tr, x_te, meta
            class_masks = {cls: (y_arr == cls) for cls in classes}
            for a_local, b_local in pair_candidates:
                a = int(pool_idx[a_local])
                b = int(pool_idx[b_local])
                cmp_vec = x_tr[:, a] > x_tr[:, b]
                class_probs = []
                for cls in classes:
                    mask = class_masks[cls]
                    if int(np.sum(mask)) <= 0:
                        class_probs.append(0.5)
                    else:
                        class_probs.append(float(np.mean(cmp_vec[mask])))
                best_gap = 0.0
                for i, j in combinations(range(len(class_probs)), 2):
                    best_gap = max(best_gap, abs(float(class_probs[i]) - float(class_probs[j])))
                feature_pair_scores.append((float(best_gap), a, b))

        if not feature_pair_scores:
            meta["stage2_ratio_features_reason"] = "no_pairs_scored"
            return x_tr, x_te, meta
        feature_pair_scores.sort(key=lambda row: row[0], reverse=True)
        selected_pairs = feature_pair_scores[: min(max_ratio, len(feature_pair_scores))]
        if not selected_pairs:
            meta["stage2_ratio_features_reason"] = "no_pairs_selected"
            return x_tr, x_te, meta

        ratio_cols_train: List[np.ndarray] = []
        ratio_cols_test: List[np.ndarray] = []
        ratio_pairs_meta: List[Dict[str, Any]] = []
        for score, a, b in selected_pairs:
            a_i = int(a)
            b_i = int(b)
            num_train = x_tr[:, a_i]
            den_train = x_tr[:, b_i]
            num_test = x_te[:, a_i]
            den_test = x_te[:, b_i]
            num_shift = float(max(0.0, -float(np.min(num_train)) + eps))
            den_shift = float(max(0.0, -float(np.min(den_train)) + eps))
            with np.errstate(divide="ignore", invalid="ignore"):
                r_train = np.log((num_train + num_shift + eps) / (den_train + den_shift + eps))
                r_test = np.log((num_test + num_shift + eps) / (den_test + den_shift + eps))
            if not (np.all(np.isfinite(r_train)) and np.all(np.isfinite(r_test))):
                continue
            ratio_cols_train.append(np.asarray(r_train, dtype=float).reshape(-1, 1))
            ratio_cols_test.append(np.asarray(r_test, dtype=float).reshape(-1, 1))
            ratio_pairs_meta.append(
                {
                    "numerator": int(a_i),
                    "denominator": int(b_i),
                    "score": float(score),
                    "numerator_shift": float(num_shift),
                    "denominator_shift": float(den_shift),
                    "epsilon": float(eps),
                }
            )

        if not ratio_cols_train:
            meta["stage2_ratio_features_reason"] = "nonfinite_ratio_features"
            return x_tr, x_te, meta

        x_ratio_train = np.hstack(ratio_cols_train)
        x_ratio_test = np.hstack(ratio_cols_test)
        x_train_out = np.hstack([x_tr, x_ratio_train])
        x_test_out = np.hstack([x_te, x_ratio_test])

        meta["stage2_ratio_features_applied"] = True
        meta["stage2_ratio_features_reason"] = "ok"
        meta["stage2_ratio_features_added"] = int(x_ratio_train.shape[1])
        meta["stage2_ratio_pairs"] = ratio_pairs_meta
        return x_train_out, x_test_out, meta

    def _select_model_via_cv_scored(self, X_train: np.ndarray, y_train: np.ndarray, seed: int):
        requested_candidates = self._resolved_model_candidates()
        candidate_names, runtime_meta = self._apply_model_cv_runtime_containment(
            candidate_names=requested_candidates,
            X_train=X_train,
            y_train=y_train,
        )
        if not candidate_names:
            candidate_names = ["lr"]
            runtime_meta["model_cv_effective_candidates"] = ("lr",)

        classes, counts = np.unique(np.asarray(y_train).ravel(), return_counts=True)
        n_classes = int(classes.size)
        class_counts = np.asarray(counts, dtype=int)

        backend, dispatch_meta = self._get_classifier_backend(candidate_names=candidate_names)
        runtime_meta.update(dispatch_meta)

        supports = backend.supports_dataset(
            n_samples=int(np.asarray(X_train).shape[0]),
            n_features=int(np.asarray(X_train).shape[1]),
            n_classes=int(n_classes),
            class_counts=class_counts,
        )
        if (not supports) and backend.name() in {"flaml", "optuna"}:
            runtime_meta["classification_backend_used"] = "sklearn"
            runtime_meta["classification_backend_fallback_reason"] = "dataset_not_supported"
            backend = self._make_sklearn_backend(candidate_names=candidate_names)

        stage2_start = self._timer()
        try:
            model, model_name, score, std, n_splits, backend_meta = backend.fit_and_select(
                np.asarray(X_train, dtype=float),
                np.asarray(y_train).ravel(),
                seed=int(seed),
                n_classes=int(n_classes),
                class_counts=class_counts,
                cv_splits=5,
                scoring="balanced_accuracy",
            )
        except (ImportError, RuntimeError) as exc:
            if backend.name() in {"flaml", "optuna"}:
                warnings.warn(
                    f"{backend.name().upper()} backend unavailable/unsupported ({exc}); falling back to sklearn backend.",
                    RuntimeWarning,
                )
                runtime_meta["classification_backend_used"] = "sklearn"
                runtime_meta["classification_backend_fallback_reason"] = str(type(exc).__name__)
                fallback = self._make_sklearn_backend(candidate_names=candidate_names)
                model, model_name, score, std, n_splits, backend_meta = fallback.fit_and_select(
                    np.asarray(X_train, dtype=float),
                    np.asarray(y_train).ravel(),
                    seed=int(seed),
                    n_classes=int(n_classes),
                    class_counts=class_counts,
                    cv_splits=5,
                    scoring="balanced_accuracy",
                )
            else:
                raise
        runtime_meta["classification_stage2_wall_seconds"] = float(max(0.0, self._timer() - stage2_start))

        if isinstance(backend_meta, dict):
            runtime_meta.update(dict(backend_meta))
        runtime_meta["classification_backend_used"] = str(
            runtime_meta.get("classification_backend_used", getattr(backend, "name", lambda: "sklearn")())
        )
        return model, model_name, float(score), float(std), int(n_splits), runtime_meta

    def _resolved_model_candidates(self) -> List[str]:
        cls_cfg = self._classification_cfg()
        ordered: List[str] = []
        seen: Set[str] = set()
        seen_alias_groups: Set[str] = set()
        alias_groups = {
            "dlda": "lda_shrink",
            "shrinkage_lda": "lda_shrink",
        }

        def _append(name: str) -> None:
            model_name = str(name).strip()
            if not model_name:
                return
            alias_key = str(alias_groups.get(model_name, model_name))
            if alias_key in seen_alias_groups:
                return
            if model_name not in seen:
                ordered.append(model_name)
                seen.add(model_name)
                seen_alias_groups.add(alias_key)

        for name in cls_cfg.model_candidates:
            _append(name)

        if cls_cfg.include_elastic_net_model:
            _append("elastic_net_lr")
        if cls_cfg.include_rf_model:
            _append("rf")
        if cls_cfg.include_knn_model:
            _append("knn")
        if cls_cfg.include_svm_linear_model:
            _append("svm_linear")
        if cls_cfg.include_dlda_model:
            _append("dlda")
        if bool(getattr(cls_cfg, "include_nsc_model", False)):
            _append("nsc")
        if bool(getattr(cls_cfg, "include_pls_da_model", False)):
            _append("pls_da_classifier")
        if bool(getattr(cls_cfg, "include_gpc_model", False)):
            _append("gpc")
        if bool(cls_cfg.include_nb_model):
            _append("nb")
        if bool(cls_cfg.include_vote_ensemble_model):
            _append("vote_ensemble")
        if bool(getattr(cls_cfg, "include_rp_ensemble_model", False)):
            _append("rp_ensemble")
        if bool(getattr(cls_cfg, "include_dbda_model", False)):
            _append("dbda")
        if bool(getattr(cls_cfg, "include_gqda_model", False)):
            _append("gqda")
        if bool(getattr(cls_cfg, "include_bc_svm_linear_model", False)):
            _append("bc_svm_linear")
        if bool(getattr(cls_cfg, "include_sglnn_model", False)):
            _append("sglnn")
        if bool(cls_cfg.include_xgb_model):
            _append("xgb")
        if bool(getattr(cls_cfg, "include_lgbm_model", False)):
            _append("lgbm")
        if bool(getattr(cls_cfg, "include_extra_tree_model", False)):
            _append("extra_tree")
        if bool(getattr(cls_cfg, "include_catboost_model", False)):
            _append("catboost")
        if bool(cls_cfg.include_tabpfn_model):
            _append("tabpfn")
        excluded_alias_groups = {
            str(alias_groups.get(str(name).strip(), str(name).strip()))
            for name in tuple(getattr(cls_cfg, "exclude_model_candidates", tuple()) or tuple())
            if str(name).strip()
        }
        return [
            name
            for name in ordered
            if str(alias_groups.get(name, name)) not in excluded_alias_groups
        ]

    @staticmethod
    def _format_optional_candidate_failure(exc: BaseException) -> str:
        detail = str(exc).strip()
        name = type(exc).__name__
        return name if not detail else f"{name}: {detail}"

    def _build_xgb_model(self, y_train: np.ndarray, seed: int):
        if _XGBClassifier is None:
            return None, "optional dependency unavailable"
        n_classes = int(np.unique(np.asarray(y_train)).size)
        params: Dict[str, Any] = {
            "n_estimators": 220,
            "max_depth": 5,
            "learning_rate": 0.06,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": 1,
            "tree_method": "hist",
            "verbosity": 0,
            "use_label_encoder": False,
        }
        if n_classes <= 2:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes
            params["eval_metric"] = "mlogloss"
        try:
            return _XGBClassifier(**params), None
        except Exception as exc:
            return None, self._format_optional_candidate_failure(exc)

    def _build_tabpfn_model(self, seed: int):
        if _TabPFNClassifier is None:
            return None, "optional dependency unavailable"
        # Force CPU device when no NVIDIA GPU is present to avoid
        # cuBLAS initialisation deadlock in loky workers.
        _device = "auto" if _HAS_NVIDIA_GPU else "cpu"
        attempts: List[Dict[str, Any]] = [
            {"random_state": seed, "device": _device},
            {"device": _device},
            {"random_state": seed},
            {},
        ]
        last_failure_reason: Optional[str] = None
        for kwargs in attempts:
            try:
                return _TabPFNClassifier(**kwargs), None
            except TypeError as exc:
                last_failure_reason = self._format_optional_candidate_failure(exc)
                continue
            except Exception as exc:
                last_failure_reason = self._format_optional_candidate_failure(exc)
                continue
        return None, str(last_failure_reason or "constructor_rejected_all_attempts")

    def _warn_missing_model_backend(
        self,
        model_name: str,
        package_name: str,
        reason: Optional[str] = None,
    ) -> None:
        if model_name in self._warned_missing_model_backends:
            return
        detail = str(reason or "").strip()
        if detail:
            message = (
                f"Requested model candidate '{model_name}' could not be constructed via "
                f"optional dependency '{package_name}' ({detail}). Skipping this candidate."
            )
        else:
            message = (
                f"Requested model candidate '{model_name}' but optional dependency "
                f"'{package_name}' is unavailable. Skipping this candidate."
            )
        warnings.warn(
            message,
            RuntimeWarning,
        )
        self._warned_missing_model_backends.add(model_name)

    @staticmethod
    def _timer() -> float:
        # Isolated method for easier testing/mocking if needed.
        import time

        return float(time.perf_counter())

    def _config_snapshot(self) -> Dict[str, Any]:
        return {
            "random_seed": self.config.random_seed,
            "test_size": self.config.test_size,
            "max_train_samples": self.config.max_train_samples,
            "fs_fraction": self.config.fs_fraction,
            "n_final_features": self.config.n_final_features,
            "enable_ratio_features": bool(getattr(self.config, "enable_ratio_features", False)),
            "ratio_pool_size": int(getattr(self.config, "ratio_pool_size", 0) or 0),
            "ratio_selection_method": str(getattr(self.config, "ratio_selection_method", "ktsp") or "ktsp"),
            "ratio_max_pairs": int(getattr(self.config, "ratio_max_pairs", 0) or 0),
            "max_ratio_features": int(getattr(self.config, "max_ratio_features", 0) or 0),
            "ratio_epsilon": float(getattr(self.config, "ratio_epsilon", 1e-6) or 1e-6),
            "ratio_include_originals": bool(getattr(self.config, "ratio_include_originals", True)),
            "ratio_abs_value": bool(getattr(self.config, "ratio_abs_value", False)),
            "ratio_require_positive": bool(getattr(self.config, "ratio_require_positive", True)),
            "selection_strategy": str(
                getattr(self.config, "selection_strategy", "mnpo_portfolio") or "mnpo_portfolio"
            ),
            "multiomics_adapter": str(getattr(self.config, "multiomics_adapter", "none") or "none"),
            "multiomics_integrator": str(getattr(self.config, "multiomics_integrator", "mb_plsda") or "mb_plsda"),
            "multiomics_n_components": int(getattr(self.config, "multiomics_n_components", 2) or 2),
            "multiomics_feature_blocks": {
                str(name): list(indices)
                for name, indices in (
                    getattr(self.config, "multiomics_feature_blocks", {}) or {}
                ).items()
            },
            "meta_learning_selector_mode": str(
                getattr(self.config, "meta_learning_selector_mode", "none") or "none"
            ),
            "meta_learning_confidence_threshold": float(
                getattr(self.config, "meta_learning_confidence_threshold", 0.55) or 0.55
            ),
            "dist_criterion": self.config.dist_criterion,
            "dist_config__family_set": str(getattr(self.config.dist_config, "family_set", "v6") or "v6"),
            "dist_config__robust_mode": bool(getattr(self.config.dist_config, "robust_mode", True)),
            "dist_config__use_adaptive_strategy": bool(getattr(self.config.dist_config, "use_adaptive_strategy", True)),
            "dist_config__use_lrt": bool(getattr(self.config.dist_config, "use_lrt", True)),
            "dist_config__use_cv": bool(getattr(self.config.dist_config, "use_cv", False)),
            "dist_config__compute_budget": str(getattr(self.config.dist_config, "compute_budget", "standard") or "standard"),
            "dist_config__use_support_filtering": bool(getattr(self.config.dist_config, "use_support_filtering", True)),
            "dist_config__rejection_gate": bool(getattr(self.config.dist_config, "rejection_gate", True)),
            "dist_config__rejection_p_threshold": float(getattr(self.config.dist_config, "rejection_p_threshold", 0.01) or 0.01),
            "dist_config__confidence_margin": float(getattr(self.config.dist_config, "confidence_margin", 0.05) or 0.05),
            "dist_config__compute_ad": bool(getattr(self.config.dist_config, "compute_ad", False)),
            "dist_config__ad_bootstrap_samples": int(getattr(self.config.dist_config, "ad_bootstrap_samples", 0) or 0),
            "dist_config__compute_qq_pp": bool(getattr(self.config.dist_config, "compute_qq_pp", False)),
            "dist_config__compute_dip": bool(getattr(self.config.dist_config, "compute_dip", False)),
            "dist_config__dip_hist_bins": int(getattr(self.config.dist_config, "dip_hist_bins", 40) or 40),
            "dist_config__interval_likelihood": bool(getattr(self.config.dist_config, "interval_likelihood", False)),
            "dist_config__interval_delta_override": float(getattr(self.config.dist_config, "interval_delta_override", 0.0) or 0.0),
            "dist_config__use_lmoment_prescreen": bool(getattr(self.config.dist_config, "use_lmoment_prescreen", False)),
            "dist_config__lmoment_prescreen_max_candidates": int(
                getattr(self.config.dist_config, "lmoment_prescreen_max_candidates", 0) or 0
            ),
            "dist_config__estimator": str(getattr(self.config.dist_config, "estimator", "mle") or "mle"),
            "dist_config__mps_maxiter": int(getattr(self.config.dist_config, "mps_maxiter", 250) or 250),
            "dist_config__mps_tol": float(getattr(self.config.dist_config, "mps_tol", 1e-6) or 1e-6),
            "dist_config__compute_crps": bool(getattr(self.config.dist_config, "compute_crps", False)),
            "dist_config__crps_mc_samples": int(getattr(self.config.dist_config, "crps_mc_samples", 96) or 96),
            "dist_config__crps_data_subsample": int(getattr(self.config.dist_config, "crps_data_subsample", 256) or 256),
            "dist_config__compute_crps_uq_decomposition": bool(
                getattr(self.config.dist_config, "compute_crps_uq_decomposition", False)
            ),
            "dist_config__mnpo_use_tritrust": bool(getattr(self.config.dist_config, "mnpo_use_tritrust", True)),
            "dist_config__mnpo_include_crps": bool(getattr(self.config.dist_config, "mnpo_include_crps", False)),
            "dist_config__mnpo_include_preq": bool(getattr(self.config.dist_config, "mnpo_include_preq", False)),
            "dist_config__mnpo_use_tail_risk_oracle": False,
            "dist_config__mnpo_tail_risk_alpha": float(getattr(self.config.dist_config, "mnpo_tail_risk_alpha", 0.33) or 0.33),
            "dist_config__mnpo_use_qre_smoothing": bool(
                getattr(self.config.dist_config, "mnpo_use_qre_smoothing", False)
            ),
            "dist_config__mnpo_qre_temperature_gamma": float(
                getattr(self.config.dist_config, "mnpo_qre_temperature_gamma", 1.0) or 1.0
            ),
            "dist_config__mnpo_use_oracle_redundancy_penalty": bool(
                getattr(self.config.dist_config, "mnpo_use_oracle_redundancy_penalty", False)
            ),
            "dist_config__mnpo_compute_tremble_sensitivity": bool(
                getattr(self.config.dist_config, "mnpo_compute_tremble_sensitivity", False)
            ),
            "dist_config__preq_holdout_fraction": float(getattr(self.config.dist_config, "preq_holdout_fraction", 0.20) or 0.20),
            "dist_config__preq_min_train": int(getattr(self.config.dist_config, "preq_min_train", 20) or 20),
            "dist_config__preq_max_test_points": int(getattr(self.config.dist_config, "preq_max_test_points", 128) or 128),
            "dist_config__random_state": getattr(self.config.dist_config, "random_state", None),
            "multimodal_fallback": str(getattr(self.config, "multimodal_fallback", "none") or "none"),
            "apply_cdf_transform": self.config.apply_cdf_transform,
            "df_stage_position": str(getattr(self.config, "df_stage_position", "after_fs") or "after_fs"),
            "df_fastpath_enabled": self.config.df_fastpath_enabled,
            "df_fastpath_trigger": self.config.df_fastpath_trigger,
            "df_fastpath_small_n_threshold": self.config.df_fastpath_small_n_threshold,
            "df_fastpath_unique_ratio_threshold": self.config.df_fastpath_unique_ratio_threshold,
            "df_fastpath_n_unique_threshold": self.config.df_fastpath_n_unique_threshold,
            "cdf_reliability_gate": self.config.cdf_reliability_gate,
            "cdf_min_gof_p": self.config.cdf_min_gof_p,
            "cdf_max_confidence_set": self.config.cdf_max_confidence_set,
            "cdf_skip_heaped_features": self.config.cdf_skip_heaped_features,
            "cdf_block_gating_cv": self.config.cdf_block_gating_cv,
            "cdf_block_gating_n_blocks": self.config.cdf_block_gating_n_blocks,
            "cdf_block_gating_min_block_size": self.config.cdf_block_gating_min_block_size,
            "cdf_block_gating_cv_splits": self.config.cdf_block_gating_cv_splits,
            "cdf_block_gating_max_blocks": self.config.cdf_block_gating_max_blocks,
            "cdf_block_gating_time_budget_sec": self.config.cdf_block_gating_time_budget_sec,
            "cdf_block_gating_min_improvement": self.config.cdf_block_gating_min_improvement,
            "max_dist_features": self.config.max_dist_features,
            "low_gof_downweighting": self.config.low_gof_downweighting,
            "low_gof_threshold": self.config.low_gof_threshold,
            "low_gof_weight": self.config.low_gof_weight,
            "use_distribution_stability_weight": self.config.use_distribution_stability_weight,
            "stability_bootstrap": self.config.stability_bootstrap,
            "use_rank_prefilter": self.config.use_rank_prefilter,
            "prefilter_top_k": self.config.prefilter_top_k,
            "prefilter_mi_weight": float(getattr(self.config, "prefilter_mi_weight", 0.60) or 0.60),
            "prefilter_f_weight": float(getattr(self.config, "prefilter_f_weight", 0.40) or 0.40),
            "prefilter_union_enabled": bool(getattr(self.config, "prefilter_union_enabled", False)),
            "prefilter_strategies": list(
                getattr(self.config, "prefilter_strategies", ("mi_ftest_blend",)) or ("mi_ftest_blend",)
            ),
            "prefilter_nondefault_budget_fraction": float(
                getattr(self.config, "prefilter_nondefault_budget_fraction", 0.10) or 0.10
            ),
            "prefilter_bh_ttest_enabled": bool(getattr(self.config, "prefilter_bh_ttest_enabled", False)),
            "prefilter_bh_ttest_alpha": float(
                getattr(self.config, "prefilter_bh_ttest_alpha", 0.05) or 0.05
            ),
            "prefilter_wsnr_enabled": bool(getattr(self.config, "prefilter_wsnr_enabled", False)),
            "prefilter_data_domain": str(getattr(self.config, "prefilter_data_domain", "auto") or "auto"),
            "prefilter_rnaseq_transform_enabled": bool(
                getattr(self.config, "prefilter_rnaseq_transform_enabled", True)
            ),
            "prefilter_rnaseq_transform_force": bool(
                getattr(self.config, "prefilter_rnaseq_transform_force", False)
            ),
            "prefilter_rnaseq_nb_lrt_enabled": bool(
                getattr(self.config, "prefilter_rnaseq_nb_lrt_enabled", False)
            ),
            "prefilter_rnaseq_nb_lrt_alpha": float(
                getattr(self.config, "prefilter_rnaseq_nb_lrt_alpha", 0.10) or 0.10
            ),
            "batch_correction": str(getattr(self.config, "batch_correction", "none") or "none"),
            "batch_correction_combat_prior_strength": float(
                getattr(self.config, "batch_correction_combat_prior_strength", 8.0) or 8.0
            ),
            "batch_correction_cdf_n_quantiles": int(
                getattr(self.config, "batch_correction_cdf_n_quantiles", 33) or 33
            ),
            "batch_correction_cdf_clip_low": float(
                getattr(self.config, "batch_correction_cdf_clip_low", 0.01) or 0.01
            ),
            "batch_correction_cdf_clip_high": float(
                getattr(self.config, "batch_correction_cdf_clip_high", 0.99) or 0.99
            ),
            "screening_enabled": bool(getattr(self.config, "screening_enabled", False)),
            "screening_method": str(getattr(self.config, "screening_method", "none") or "none"),
            "screening_evalue_alpha": float(getattr(self.config, "screening_evalue_alpha", 0.20) or 0.20),
            "screening_evalue_min_features": int(
                getattr(self.config, "screening_evalue_min_features", 20) or 20
            ),
            "screening_pool_cap": int(getattr(self.config, "screening_pool_cap", 2000) or 2000),
            "screening_stir_n_neighbors": int(getattr(self.config, "screening_stir_n_neighbors", 10) or 10),
            "screening_stir_n_iter": int(getattr(self.config, "screening_stir_n_iter", 50) or 50),
            "screening_stir_keep_fraction": float(
                getattr(self.config, "screening_stir_keep_fraction", 0.5) or 0.5
            ),
            "screening_stir_min_features": int(getattr(self.config, "screening_stir_min_features", 20) or 20),
            "folding_method": str(getattr(self.config, "folding_method", "pls_da") or "pls_da"),
            "folding_n_components": int(getattr(self.config, "folding_n_components", 512) or 512),
            "folding_rff_gamma": (
                None
                if getattr(self.config, "folding_rff_gamma", None) is None
                else float(getattr(self.config, "folding_rff_gamma"))
            ),
            "folding_pls_components": int(getattr(self.config, "folding_pls_components", 32) or 32),
            "folding_pls_scale": bool(getattr(self.config, "folding_pls_scale", True)),
            "folding_pls_min_classes": int(getattr(self.config, "folding_pls_min_classes", 5) or 5),
            "folding_pls_min_n_per_class": int(
                getattr(self.config, "folding_pls_min_n_per_class", 3) or 3
            ),
            "folding_pls_max_imbalance_ratio": float(
                getattr(self.config, "folding_pls_max_imbalance_ratio", 6.0) or 6.0
            ),
            "folding_prefilter_k": getattr(self.config, "folding_prefilter_k", None),
            "enable_face_domain_projection": bool(
                getattr(self.config, "enable_face_domain_projection", False)
            ),
            "use_balanced_fs_subsample": self.config.use_balanced_fs_subsample,
            "fs_min_per_class": self.config.fs_min_per_class,
            "fs_method_timeout_seconds": float(getattr(self.config, "fs_method_timeout_seconds", 0.0) or 0.0),
            "fs_linear_svm_max_iter": int(getattr(self.config, "fs_linear_svm_max_iter", 10000) or 10000),
            "fs_runtime_racing_enabled": bool(getattr(self.config, "fs_runtime_racing_enabled", False)),
            "fs_runtime_racing_proxy_splits": int(getattr(self.config, "fs_runtime_racing_proxy_splits", 1) or 1),
            "fs_runtime_racing_keep_fraction": float(
                getattr(self.config, "fs_runtime_racing_keep_fraction", 0.60) or 0.60
            ),
            "fs_runtime_racing_min_candidates": int(
                getattr(self.config, "fs_runtime_racing_min_candidates", 4) or 4
            ),
            "fs_runtime_racing_runtime_weight": float(
                getattr(self.config, "fs_runtime_racing_runtime_weight", 0.15) or 0.15
            ),
            "fs_runtime_racing_mode": str(
                getattr(self.config, "fs_runtime_racing_mode", "single_stage") or "single_stage"
            ),
            "fs_runtime_racing_stages": int(getattr(self.config, "fs_runtime_racing_stages", 2) or 2),
            "fs_runtime_racing_confidence_bound": str(
                getattr(self.config, "fs_runtime_racing_confidence_bound", "none") or "none"
            ),
            "fs_runtime_racing_delta": float(getattr(self.config, "fs_runtime_racing_delta", 0.10) or 0.10),
            "enabled_methods": list(self.config.enabled_methods),
            "tier_lockout_enabled": bool(getattr(self.config, "tier_lockout_enabled", False)),
            "tier_lockout_tier": str(getattr(self.config, "tier_lockout_tier", "easy") or "easy"),
            "tier_lockout_difficulty_source": str(
                getattr(self.config, "tier_lockout_difficulty_source", "historical") or "historical"
            ),
            "tier_lockout_fallback_methods": list(
                getattr(self.config, "tier_lockout_fallback_methods", tuple()) or tuple()
            ),
            "tier_routing_enabled": bool(getattr(self.config, "tier_routing_enabled", False)),
            "tier_routing_difficulty_classifier": str(
                getattr(self.config, "tier_routing_difficulty_classifier", "meta_features") or "meta_features"
            ),
            "tier_routing_table": {
                str(k): list(v) if isinstance(v, (list, tuple, set)) else [str(v)]
                for k, v in (getattr(self.config, "tier_routing_table", {}) or {}).items()
            },
            "regime_gating_enabled": bool(getattr(self.config, "regime_gating_enabled", False)),
            "regime_gating_difficulty_source": str(
                getattr(self.config, "regime_gating_difficulty_source", "historical") or "historical"
            ),
            "regime_gating_target_tier": str(
                getattr(self.config, "regime_gating_target_tier", "very_hard") or "very_hard"
            ),
            "regime_gating_min_samples_per_class": float(
                getattr(self.config, "regime_gating_min_samples_per_class", 15.0) or 15.0
            ),
            "regime_gating_low_p_over_n_threshold": float(
                getattr(self.config, "regime_gating_low_p_over_n_threshold", 0.0) or 0.0
            ),
            "regime_gating_simple_methods": list(
                getattr(self.config, "regime_gating_simple_methods", tuple()) or tuple()
            ),
            "regime_gating_very_hard_portfolio_max_methods": int(
                getattr(self.config, "regime_gating_very_hard_portfolio_max_methods", 4) or 4
            ),
            "regime_gating_very_hard_copula_derandomize_runs": int(
                getattr(self.config, "regime_gating_very_hard_copula_derandomize_runs", 5) or 5
            ),
            "regime_gating_low_p_over_n_mode": str(
                getattr(self.config, "regime_gating_low_p_over_n_mode", "fast_univariate_filter")
                or "fast_univariate_filter"
            ),
            "fs_portfolio_size": int(getattr(self.config, "fs_portfolio_size", 5)),
            "fs_portfolio_size_guard": str(getattr(self.config, "fs_portfolio_size_guard", "none") or "none"),
            "fs_adaptive_portfolio_sizing_enabled": bool(
                getattr(self.config, "fs_adaptive_portfolio_sizing_enabled", False)
            ),
            "fs_adaptive_size_min": (
                None
                if getattr(self.config, "fs_adaptive_size_min", None) is None
                else int(getattr(self.config, "fs_adaptive_size_min"))
            ),
            "fs_adaptive_size_max": (
                None
                if getattr(self.config, "fs_adaptive_size_max", None) is None
                else int(getattr(self.config, "fs_adaptive_size_max"))
            ),
            "fs_adaptive_sizing_variance_penalty": bool(
                getattr(self.config, "fs_adaptive_sizing_variance_penalty", False)
            ),
            "fs_adaptive_sizing_variance_penalty_strength": float(
                getattr(self.config, "fs_adaptive_sizing_variance_penalty_strength", 0.5) or 0.5
            ),
            "fs_mnpo_paradigm_aware_prior_enabled": bool(
                getattr(self.config, "fs_mnpo_paradigm_aware_prior_enabled", False)
            ),
            "fs_mnpo_interaction_floor": float(
                getattr(self.config, "fs_mnpo_interaction_floor", 0.12) or 0.12
            ),
            "fs_rashomon_enabled": bool(getattr(self.config, "fs_rashomon_enabled", False)),
            "fs_rashomon_max_models": int(getattr(self.config, "fs_rashomon_max_models", 12) or 12),
            "fs_rashomon_score_tolerance": float(
                getattr(self.config, "fs_rashomon_score_tolerance", 0.01) or 0.01
            ),
            "eval_models_enabled": bool(getattr(self.config, "eval_models_enabled", False)),
            "eval_models": list(getattr(self.config, "eval_models", ("lr_l2", "linear_svc", "rf_small")) or ()),
            "eval_aggregate": str(getattr(self.config, "eval_aggregate", "mean") or "mean"),
            "eval_cvar_alpha": float(getattr(self.config, "eval_cvar_alpha", 0.33) or 0.33),
            "mnpo_performance_oracle_mode": str(
                getattr(self.config, "mnpo_performance_oracle_mode", "single") or "single"
            ).strip().lower(),
            "fs_mnpo_consensus_exclude_methods": list(getattr(self.config, "fs_mnpo_consensus_exclude_methods", ()) or ()),
            "fs_mnpo_consensus_exclude_protect_top_k": int(
                getattr(self.config, "fs_mnpo_consensus_exclude_protect_top_k", 0) or 0
            ),
            "fs_mnpo_include_legacy_consensus": bool(getattr(self.config, "fs_mnpo_include_legacy_consensus", True)),
            "fs_mnpo_include_majority_consensus": bool(getattr(self.config, "fs_mnpo_include_majority_consensus", True)),
            "fs_inner_cv_splits": int(getattr(self.config, "fs_inner_cv_splits", 3) or 3),
            "fs_inner_cv_repeats": int(getattr(self.config, "fs_inner_cv_repeats", 1) or 1),
            "use_tritrust": self.config.use_tritrust,
            "use_stability_oracle": self.config.use_stability_oracle,
            "use_complexity_oracle": self.config.use_complexity_oracle,
            "use_robust_oracle": self.config.use_robust_oracle,
            "use_diversity_oracle": self.config.use_diversity_oracle,
            "fs_use_cvar_oracle": bool(getattr(self.config, "fs_use_cvar_oracle", False)),
            "fs_cvar_alpha": float(getattr(self.config, "fs_cvar_alpha", 0.33) or 0.33),
            "fs_oracle_weighting_mode": str(
                getattr(self.config, "fs_oracle_weighting_mode", "tritrust") or "tritrust"
            ),
            "fs_shapley_n_coalitions_max": int(
                getattr(self.config, "fs_shapley_n_coalitions_max", 4096) or 4096
            ),
            "fs_shapley_bayesian_shrinkage": bool(
                getattr(self.config, "fs_shapley_bayesian_shrinkage", False)
            ),
            "fs_shapley_bayesian_prior_strength": float(
                getattr(self.config, "fs_shapley_bayesian_prior_strength", 8.0) or 8.0
            ),
            "fs_use_interaction_oracle": bool(
                getattr(self.config, "fs_use_interaction_oracle", False)
            ),
            "fs_interaction_oracle_min_n_train": int(
                getattr(self.config, "fs_interaction_oracle_min_n_train", 150) or 150
            ),
            "fs_interaction_oracle_pool_size_cap": int(
                getattr(self.config, "fs_interaction_oracle_pool_size_cap", 64) or 64
            ),
            "fs_interaction_oracle_pair_cap": int(
                getattr(self.config, "fs_interaction_oracle_pair_cap", 20000) or 20000
            ),
            "fs_use_ubayfs_oracle": bool(getattr(self.config, "fs_use_ubayfs_oracle", False)),
            "fs_ubayfs_n_bootstrap": int(
                getattr(self.config, "fs_ubayfs_n_bootstrap", 32) or 32
            ),
            "fs_ubayfs_min_n": int(getattr(self.config, "fs_ubayfs_min_n", 100) or 100),
            "fs_ubayfs_prior_weight": float(
                getattr(self.config, "fs_ubayfs_prior_weight", 0.0) or 0.0
            ),
            "fs_use_conformal_uq": bool(getattr(self.config, "fs_use_conformal_uq", False)),
            "fs_conformal_uq_alpha": float(
                getattr(self.config, "fs_conformal_uq_alpha", 0.10) or 0.10
            ),
            "fs_conformal_uq_min_folds": int(
                getattr(self.config, "fs_conformal_uq_min_folds", 5) or 5
            ),
            "fs_fold_preference_mode": str(
                getattr(self.config, "fs_fold_preference_mode", "vote") or "vote"
            ),
            "fs_use_conformal_efficiency": bool(
                getattr(self.config, "fs_use_conformal_efficiency", False)
            ),
            "fs_conformal_efficiency_method": str(
                getattr(self.config, "fs_conformal_efficiency_method", "split") or "split"
            ),
            "fs_oracle_weight_js_shrinkage": bool(
                getattr(self.config, "fs_oracle_weight_js_shrinkage", False)
            ),
            "fs_payoff_shrinkage_kappa": float(
                getattr(self.config, "fs_payoff_shrinkage_kappa", 0.0) or 0.0
            ),
            "use_tail_risk_oracle": False,
            "tail_risk_alpha": float(getattr(self.config, "tail_risk_alpha", 0.33) or 0.33),
            "use_regret_oracle": False,
            "use_qre_smoothing": bool(getattr(self.config, "use_qre_smoothing", False)),
            "qre_temperature_gamma": float(getattr(self.config, "qre_temperature_gamma", 1.0) or 1.0),
            "use_oracle_redundancy_penalty": bool(getattr(self.config, "use_oracle_redundancy_penalty", False)),
            "compute_tremble_sensitivity": bool(getattr(self.config, "compute_tremble_sensitivity", False)),
            "fs_diversity_oracle_mode": self.config.fs_diversity_oracle_mode,
            "fs_diversity_redundancy_weight": self.config.fs_diversity_redundancy_weight,
            "fs_diversity_complementarity_weight": self.config.fs_diversity_complementarity_weight,
            "fs_performance_balanced_weight": self.config.fs_performance_balanced_weight,
            "fs_performance_macro_f1_weight": self.config.fs_performance_macro_f1_weight,
            "fs_performance_use_adaptive_imbalance": self.config.fs_performance_use_adaptive_imbalance,
            "fs_performance_imbalance_ratio_trigger": self.config.fs_performance_imbalance_ratio_trigger,
            "fs_performance_min_classes_for_adaptive": self.config.fs_performance_min_classes_for_adaptive,
            "fs_rank_aggregation_mode": self.config.fs_rank_aggregation_mode,
            "fs_wrapper_refine_enabled": self.config.fs_wrapper_refine_enabled,
            "fs_wrapper_refine_top_k": self.config.fs_wrapper_refine_top_k,
            "fs_wrapper_refine_max_add": self.config.fs_wrapper_refine_max_add,
            "fs_wrapper_refine_min_gain": self.config.fs_wrapper_refine_min_gain,
            "fs_ova_negative_ratio": self.config.fs_ova_negative_ratio,
            "fs_ova_min_classes": self.config.fs_ova_min_classes,
            "fs_ova_min_pos_samples": self.config.fs_ova_min_pos_samples,
            "fs_ova_class_weight_mode": self.config.fs_ova_class_weight_mode,
            "fs_ova_aggregation_mode": self.config.fs_ova_aggregation_mode,
            "fs_ova_aggregation_p": self.config.fs_ova_aggregation_p,
            "fs_ova_linear_backend": self.config.fs_ova_linear_backend,
            "fs_ova_enable_calibration": bool(getattr(self.config, "fs_ova_enable_calibration", False)),
            "fs_ova_calibration_cv": int(getattr(self.config, "fs_ova_calibration_cv", 3) or 3),
            "fs_ecoc_min_classes": self.config.fs_ecoc_min_classes,
            "fs_ecoc_max_ovo_pairs": self.config.fs_ecoc_max_ovo_pairs,
            "fs_ecoc_random_code_bits": self.config.fs_ecoc_random_code_bits,
            "fs_ecoc_class_complexity_weight": self.config.fs_ecoc_class_complexity_weight,
            "fs_ecoc_include_ova_tasks": self.config.fs_ecoc_include_ova_tasks,
            "fs_ecoc_negative_ratio": self.config.fs_ecoc_negative_ratio,
            "fs_joint_multiclass_min_classes": self.config.fs_joint_multiclass_min_classes,
            "fs_joint_multiclass_max_features": self.config.fs_joint_multiclass_max_features,
            "fs_joint_multiclass_path_grid_size": self.config.fs_joint_multiclass_path_grid_size,
            "fs_joint_multiclass_min_c": self.config.fs_joint_multiclass_min_c,
            "fs_joint_multiclass_max_c": self.config.fs_joint_multiclass_max_c,
            "fs_joint_multiclass_l1_ratio": self.config.fs_joint_multiclass_l1_ratio,
            "fs_joint_multiclass_univariate_blend": self.config.fs_joint_multiclass_univariate_blend,
            "fs_dove_min_classes": int(getattr(self.config, "fs_dove_min_classes", 3) or 3),
            "fs_dove_max_pairs_per_class": int(getattr(self.config, "fs_dove_max_pairs_per_class", 4) or 4),
            "fs_dove_path_grid_size": int(getattr(self.config, "fs_dove_path_grid_size", 5) or 5),
            "fs_dove_specificity_weight": float(getattr(self.config, "fs_dove_specificity_weight", 0.35) or 0.35),
            "fs_dove_minority_boost": float(getattr(self.config, "fs_dove_minority_boost", 0.50) or 0.50),
            "fs_sparse_multinomial_min_classes": int(
                getattr(self.config, "fs_sparse_multinomial_min_classes", 3) or 3
            ),
            "fs_sparse_multinomial_max_features": int(
                getattr(self.config, "fs_sparse_multinomial_max_features", 320) or 320
            ),
            "fs_sparse_multinomial_path_grid_size": int(
                getattr(self.config, "fs_sparse_multinomial_path_grid_size", 6) or 6
            ),
            "fs_sparse_multinomial_min_c": float(getattr(self.config, "fs_sparse_multinomial_min_c", 0.05) or 0.05),
            "fs_sparse_multinomial_max_c": float(getattr(self.config, "fs_sparse_multinomial_max_c", 1.6) or 1.6),
            "fs_sparse_multinomial_backend": str(getattr(self.config, "fs_sparse_multinomial_backend", "mixed") or "mixed"),
            "fs_sparse_multinomial_l1_ratio": float(
                getattr(self.config, "fs_sparse_multinomial_l1_ratio", 0.70) or 0.70
            ),
            "fs_sparse_multinomial_univariate_blend": float(
                getattr(self.config, "fs_sparse_multinomial_univariate_blend", 0.20) or 0.20
            ),
            "fs_sparse_multinomial_max_iter": int(
                getattr(self.config, "fs_sparse_multinomial_max_iter", 5000) or 5000
            ),
            "fs_sparse_multinomial_screening_mode": _canonicalize_sparse_screening_mode(
                getattr(self.config, "fs_sparse_multinomial_screening_mode", "none"),
                warn_deprecated=False,
            ),
            "fs_sparse_multinomial_screening_keep_fraction": float(
                getattr(self.config, "fs_sparse_multinomial_screening_keep_fraction", 1.0) or 1.0
            ),
            "fs_sparse_multinomial_screening_min_features": int(
                getattr(self.config, "fs_sparse_multinomial_screening_min_features", 64) or 64
            ),
            "fs_sparse_multinomial_screening_fallback_on_failure": bool(
                getattr(self.config, "fs_sparse_multinomial_screening_fallback_on_failure", True)
            ),
            "fs_nsc_shrinkage_grid_size": int(
                getattr(self.config, "fs_nsc_shrinkage_grid_size", 6) or 6
            ),
            "fs_nsc_min_classes": int(
                getattr(self.config, "fs_nsc_min_classes", 3) or 3
            ),
            "fs_nsc_thresholding_mode": str(
                getattr(self.config, "fs_nsc_thresholding_mode", "soft") or "soft"
            ),
            "fs_nsc_order_quantile": float(
                getattr(self.config, "fs_nsc_order_quantile", 0.75) or 0.75
            ),
            "fs_nsc_deep_shrinkage_search": bool(
                getattr(self.config, "fs_nsc_deep_shrinkage_search", False)
            ),
            "fs_class_pareto_min_classes": int(
                getattr(self.config, "fs_class_pareto_min_classes", 3) or 3
            ),
            "fs_class_pareto_top_per_class": int(
                getattr(self.config, "fs_class_pareto_top_per_class", 64) or 64
            ),
            "fs_class_pareto_global_fraction": float(
                getattr(self.config, "fs_class_pareto_global_fraction", 0.40) or 0.40
            ),
            "fs_class_pareto_minority_boost": float(
                getattr(self.config, "fs_class_pareto_minority_boost", 0.50) or 0.50
            ),
            "fs_class_pareto_kw_weight": float(
                getattr(self.config, "fs_class_pareto_kw_weight", 0.25) or 0.25
            ),
            "fs_sdr_min_classes": int(
                getattr(self.config, "fs_sdr_min_classes", 3) or 3
            ),
            "fs_sdr_prefilter_max_features": int(
                getattr(self.config, "fs_sdr_prefilter_max_features", 512) or 512
            ),
            "fs_sdr_n_components": int(
                getattr(self.config, "fs_sdr_n_components", 3) or 3
            ),
            "fs_sdr_covariance_ridge": float(
                getattr(self.config, "fs_sdr_covariance_ridge", 1e-3) or 1e-3
            ),
            "fs_per_class_quota_enabled": bool(getattr(self.config, "fs_per_class_quota_enabled", False)),
            "fs_per_class_quota_min_per_class": int(
                getattr(self.config, "fs_per_class_quota_min_per_class", 1) or 1
            ),
            "fs_per_class_quota_max_fraction": float(
                getattr(self.config, "fs_per_class_quota_max_fraction", 0.60) or 0.60
            ),
            "fs_hsic_lasso_alpha": float(
                getattr(self.config, "fs_hsic_lasso_alpha", 0.01) or 0.01
            ),
            "fs_hsic_lasso_prefilter_max_features": int(
                getattr(self.config, "fs_hsic_lasso_prefilter_max_features", 128) or 128
            ),
            "fs_hsic_lasso_feature_sigma": float(
                getattr(self.config, "fs_hsic_lasso_feature_sigma", 0.0) or 0.0
            ),
            "fs_hsic_lasso_target_sigma": float(
                getattr(self.config, "fs_hsic_lasso_target_sigma", 0.0) or 0.0
            ),
            "fs_hsic_lasso_relevance_blend": float(
                getattr(self.config, "fs_hsic_lasso_relevance_blend", 0.20) or 0.20
            ),
            "fs_hsic_lasso_max_iter": int(
                getattr(self.config, "fs_hsic_lasso_max_iter", 4000) or 4000
            ),
            "fs_mrmr_mi_redundancy_enabled": bool(
                getattr(self.config, "fs_mrmr_mi_redundancy_enabled", False)
            ),
            "fs_mrmr_mi_n_bins": int(getattr(self.config, "fs_mrmr_mi_n_bins", 8) or 8),
            "fs_cmim_min_samples": int(getattr(self.config, "fs_cmim_min_samples", 60) or 60),
            "fs_cmim_n_bins": int(getattr(self.config, "fs_cmim_n_bins", 8) or 8),
            "fs_fcbf_n_bins": int(getattr(self.config, "fs_fcbf_n_bins", 8) or 8),
            "fs_ipss_path_grid_size": self.config.fs_ipss_path_grid_size,
            "fs_ipss_min_c": self.config.fs_ipss_min_c,
            "fs_ipss_max_c": self.config.fs_ipss_max_c,
            "fs_ipss_target_fdr": self.config.fs_ipss_target_fdr,
            "fs_ipss_null_shuffle_rounds": self.config.fs_ipss_null_shuffle_rounds,
            "fs_ipss_use_eats_threshold": self.config.fs_ipss_use_eats_threshold,
            "fs_ipss_eats_exclusion_quantile": self.config.fs_ipss_eats_exclusion_quantile,
            "fs_ipss_eats_min_threshold": self.config.fs_ipss_eats_min_threshold,
            "fs_ipss_importance_model": self.config.fs_ipss_importance_model,
            "fs_ipss_gate_min_classes": int(getattr(self.config, "fs_ipss_gate_min_classes", 0) or 0),
            "fs_ipss_gate_min_p_over_n": float(getattr(self.config, "fs_ipss_gate_min_p_over_n", 0.0) or 0.0),
            "fs_cluster_stability_corr_threshold": self.config.fs_cluster_stability_corr_threshold,
            "fs_cluster_stability_max_per_cluster": self.config.fs_cluster_stability_max_per_cluster,
            "fs_cluster_stability_min_cluster_freq": self.config.fs_cluster_stability_min_cluster_freq,
            "fs_stability_threshold_method": str(
                getattr(self.config, "fs_stability_threshold_method", "fixed") or "fixed"
            ),
            "fs_stability_target_pfer": float(
                getattr(self.config, "fs_stability_target_pfer", 1.0) or 1.0
            ),
            "fs_stability_use_loss_guided_validation": self.config.fs_stability_use_loss_guided_validation,
            "fs_stability_validation_fraction": self.config.fs_stability_validation_fraction,
            "fs_stability_validation_quantile": self.config.fs_stability_validation_quantile,
            "fs_stability_validation_min_samples": self.config.fs_stability_validation_min_samples,
            "fs_copula_knockoff_draws": self.config.fs_copula_knockoff_draws,
            "fs_copula_alpha_kn": self.config.fs_copula_alpha_kn,
            "fs_copula_alpha_ebh": self.config.fs_copula_alpha_ebh,
            "fs_copula_truncation_level": self.config.fs_copula_truncation_level,
            "fs_copula_generator": str(getattr(self.config, "fs_copula_generator", "copula") or "copula"),
            "fs_copula_deepdrk_latent_fraction": float(
                getattr(self.config, "fs_copula_deepdrk_latent_fraction", 0.35) or 0.35
            ),
            "fs_copula_deepdrk_noise_scale": float(
                getattr(self.config, "fs_copula_deepdrk_noise_scale", 1.0) or 1.0
            ),
            "fs_copula_derandomize_runs": int(
                getattr(self.config, "fs_copula_derandomize_runs", 1) or 1
            ),
            "fs_copula_stabilizer_runs": self.config.fs_copula_stabilizer_runs,
            "fs_copula_stabilizer_use_ebh": self.config.fs_copula_stabilizer_use_ebh,
            "fs_copula_stabilizer_seed_stride": self.config.fs_copula_stabilizer_seed_stride,
            "fs_importance_uq_enabled": bool(getattr(self.config, "fs_importance_uq_enabled", False)),
            "fs_importance_uq_min_cv_folds": int(getattr(self.config, "fs_importance_uq_min_cv_folds", 3) or 3),
            "fs_decorrelated_stability_eps": self.config.fs_decorrelated_stability_eps,
            "fs_iterative_pruning_pool_factor": self.config.fs_iterative_pruning_pool_factor,
            "fs_iterative_pruning_max_rounds": self.config.fs_iterative_pruning_max_rounds,
            "fs_iterative_pruning_min_improvement": self.config.fs_iterative_pruning_min_improvement,
            "fs_iterative_pruning_max_cumulative_loss": float(
                getattr(self.config, "fs_iterative_pruning_max_cumulative_loss", 0.02) or 0.02
            ),
            "fs_iterative_pruning_redundancy_weight": self.config.fs_iterative_pruning_redundancy_weight,
            "fs_iterative_pruning_bounded_prefilter_cap": self.config.fs_iterative_pruning_bounded_prefilter_cap,
            "fs_iterative_pruning_bounded_candidate_fraction": self.config.fs_iterative_pruning_bounded_candidate_fraction,
            "fs_iterative_pruning_bounded_min_candidates": self.config.fs_iterative_pruning_bounded_min_candidates,
            "fs_iterative_pruning_bounded_max_evaluations": self.config.fs_iterative_pruning_bounded_max_evaluations,
            "fs_iterative_pruning_bounded_max_runtime_seconds": self.config.fs_iterative_pruning_bounded_max_runtime_seconds,
            "fs_iterative_pruning_bounded_enable_class_gating": self.config.fs_iterative_pruning_bounded_enable_class_gating,
            "fs_iterative_pruning_bounded_multiclass_scale": self.config.fs_iterative_pruning_bounded_multiclass_scale,
            "fs_iterative_pruning_bounded_imbalance_trigger": self.config.fs_iterative_pruning_bounded_imbalance_trigger,
            "fs_iterative_pruning_bounded_imbalance_scale": self.config.fs_iterative_pruning_bounded_imbalance_scale,
            "fs_iterative_pruning_bounded_use_cpss_overlay": bool(
                getattr(self.config, "fs_iterative_pruning_bounded_use_cpss_overlay", False)
            ),
            "fs_iterative_pruning_bounded_cpss_pairs": int(
                getattr(self.config, "fs_iterative_pruning_bounded_cpss_pairs", 4) or 4
            ),
            "fs_iterative_pruning_bounded_cpss_stability_threshold": float(
                getattr(self.config, "fs_iterative_pruning_bounded_cpss_stability_threshold", 0.60) or 0.60
            ),
            "fs_iterative_pruning_bounded_cpss_min_stable_features": int(
                getattr(self.config, "fs_iterative_pruning_bounded_cpss_min_stable_features", 2) or 2
            ),
            "fs_iterative_pruning_bounded_cpss_min_jaccard": float(
                getattr(self.config, "fs_iterative_pruning_bounded_cpss_min_jaccard", 0.35) or 0.35
            ),
            "fs_iterative_pruning_bounded_cpss_max_score_drop": float(
                getattr(self.config, "fs_iterative_pruning_bounded_cpss_max_score_drop", 0.005) or 0.005
            ),
            "fs_iterative_pruning_class_pareto_prefilter_enabled": bool(
                getattr(self.config, "fs_iterative_pruning_class_pareto_prefilter_enabled", False)
            ),
            "fs_iterative_pruning_class_pareto_min_classes": int(
                getattr(self.config, "fs_iterative_pruning_class_pareto_min_classes", 3) or 3
            ),
            "fs_iterative_pruning_class_pareto_top_per_class": int(
                getattr(self.config, "fs_iterative_pruning_class_pareto_top_per_class", 64) or 64
            ),
            "fs_iterative_pruning_class_pareto_global_fraction": float(
                getattr(self.config, "fs_iterative_pruning_class_pareto_global_fraction", 0.40) or 0.40
            ),
            "fs_iterative_pruning_class_pareto_minority_boost": float(
                getattr(self.config, "fs_iterative_pruning_class_pareto_minority_boost", 0.50) or 0.50
            ),
            "fs_iterative_pruning_class_pareto_stability_gate_enabled": bool(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_gate_enabled", False)
            ),
            "fs_iterative_pruning_class_pareto_stability_subsamples": int(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_subsamples", 6) or 6
            ),
            "fs_iterative_pruning_class_pareto_stability_fraction": float(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_fraction", 0.70) or 0.70
            ),
            "fs_iterative_pruning_class_pareto_stability_threshold": float(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_threshold", 0.55) or 0.55
            ),
            "fs_iterative_pruning_class_pareto_stability_min_overlap": float(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_min_overlap", 0.50) or 0.50
            ),
            "fs_iterative_pruning_class_pareto_stability_min_stable_features": int(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_min_stable_features", 4) or 4
            ),
            "fs_iterative_pruning_class_pareto_stability_fallback_on_failure": bool(
                getattr(self.config, "fs_iterative_pruning_class_pareto_stability_fallback_on_failure", True)
            ),
            "fs_decorrelated_stability_min_max_abs_corr": float(
                getattr(self.config, "fs_decorrelated_stability_min_max_abs_corr", 0.0) or 0.0
            ),
            "classification_backend": self._classification_cfg().backend,
            "classification_selection_mode": str(self._classification_cfg().selection_mode),
            "flaml_time_budget": int(self._classification_cfg().flaml_time_budget),
            "optuna_time_budget": int(self._classification_cfg().optuna_time_budget),
            "optuna_n_trials": int(self._classification_cfg().optuna_n_trials),
            "flaml_estimator_list": list(self._classification_cfg().flaml_estimator_list),
            "flaml_metric": str(self._classification_cfg().flaml_metric),
            "model_candidates": list(self._classification_cfg().model_candidates),
            "exclude_model_candidates": list(
                getattr(self._classification_cfg(), "exclude_model_candidates", tuple()) or tuple()
            ),
            "classifier_regime_candidate_exclusions": list(
                getattr(self._classification_cfg(), "regime_candidate_exclusions", tuple()) or tuple()
            ),
            "classifier_oracle_complexity_prior_overrides": list(
                getattr(self._classification_cfg(), "oracle_complexity_prior_overrides", tuple()) or tuple()
            ),
            "include_elastic_net_model": bool(self._classification_cfg().include_elastic_net_model),
            "include_rf_model": bool(self._classification_cfg().include_rf_model),
            "include_knn_model": bool(self._classification_cfg().include_knn_model),
            "include_svm_linear_model": bool(self._classification_cfg().include_svm_linear_model),
            "include_dlda_model": bool(self._classification_cfg().include_dlda_model),
            "include_nsc_model": bool(self._classification_cfg().include_nsc_model),
            "include_pls_da_model": bool(self._classification_cfg().include_pls_da_model),
            "include_gpc_model": bool(self._classification_cfg().include_gpc_model),
            "include_nb_model": bool(self._classification_cfg().include_nb_model),
            "include_vote_ensemble_model": bool(self._classification_cfg().include_vote_ensemble_model),
            "include_rp_ensemble_model": bool(
                getattr(self._classification_cfg(), "include_rp_ensemble_model", False)
            ),
            "include_xgb_model": bool(self._classification_cfg().include_xgb_model),
            "include_lgbm_model": bool(self._classification_cfg().include_lgbm_model),
            "include_extra_tree_model": bool(self._classification_cfg().include_extra_tree_model),
            "include_catboost_model": bool(self._classification_cfg().include_catboost_model),
            "include_tabpfn_model": bool(self._classification_cfg().include_tabpfn_model),
            "model_cv_lr_max_iter": int(self._classification_cfg().lr_max_iter),
            "model_cv_use_hybrid_score": bool(self._classification_cfg().use_hybrid_score),
            "model_cv_balanced_weight": float(self._classification_cfg().hybrid_balanced_weight),
            "model_cv_macro_f1_weight": float(self._classification_cfg().hybrid_macro_f1_weight),
            "model_cv_runtime_containment_enabled": bool(self._classification_cfg().runtime_containment_enabled),
            "model_cv_runtime_max_candidates": int(self._classification_cfg().runtime_max_candidates),
            "model_cv_runtime_high_p_over_n_threshold": float(self._classification_cfg().runtime_high_p_over_n_threshold),
            "model_cv_runtime_high_class_threshold": int(self._classification_cfg().runtime_high_class_threshold),
            "model_cv_runtime_min_class_count_threshold": int(
                self._classification_cfg().runtime_min_class_count_threshold
            ),
            "classification_min_n_for_automl": int(self._classification_cfg().min_n_for_automl),
            "classification_min_n_per_class_for_cv": int(self._classification_cfg().min_n_per_class_for_cv),
            "classification_min_n_per_class_for_automl": int(
                self._classification_cfg().min_n_per_class_for_automl
            ),
            "classification_max_p_over_n_for_automl": int(self._classification_cfg().max_p_over_n_for_automl),
            "classifier_oracle_k": int(self._classification_cfg().oracle_k),
            "classifier_oracle_weighting_mode": str(self._classification_cfg().oracle_weighting_mode),
            "classifier_oracle_include_robustness": bool(
                self._classification_cfg().oracle_include_robustness
            ),
            "classifier_oracle_include_complexity": bool(
                self._classification_cfg().oracle_include_complexity
            ),
            "classifier_oracle_include_calibration": bool(
                self._classification_cfg().oracle_include_calibration
            ),
            "classifier_oracle_include_james_stein": bool(
                self._classification_cfg().oracle_include_james_stein
            ),
            "classifier_oracle_include_cvar": bool(
                self._classification_cfg().oracle_include_cvar
            ),
            "classifier_oracle_cvar_alpha": float(self._classification_cfg().oracle_cvar_alpha),
            "classifier_oracle_use_dynamic_complexity": bool(
                self._classification_cfg().oracle_use_dynamic_complexity
            ),
            "classifier_oracle_portfolio_diversity": bool(
                self._classification_cfg().oracle_portfolio_diversity
            ),
            "classifier_oracle_portfolio_overlap_threshold": float(
                self._classification_cfg().oracle_portfolio_overlap_threshold
            ),
            "classifier_oracle_portfolio_corr_threshold": float(
                self._classification_cfg().oracle_portfolio_corr_threshold
            ),
            "classifier_oracle_enable_hoeffding_racing": bool(
                self._classification_cfg().oracle_enable_hoeffding_racing
            ),
            "classifier_oracle_hoeffding_delta": float(self._classification_cfg().oracle_hoeffding_delta),
            "classifier_oracle_enable_bbc": bool(self._classification_cfg().oracle_enable_bbc),
            "classifier_oracle_bbc_bootstrap_rounds": int(
                self._classification_cfg().oracle_bbc_bootstrap_rounds
            ),
            "classifier_oracle_bbc_ci_level": float(self._classification_cfg().oracle_bbc_ci_level),
            "classifier_oracle_enable_ensemble": bool(self._classification_cfg().oracle_enable_ensemble),
            "classifier_oracle_behavior_profile": str(
                self._classification_cfg().oracle_behavior_profile
            ),
            "classifier_oracle_use_per_family_flaml": bool(
                self._classification_cfg().oracle_use_per_family_flaml
            ),
            "stage2_ratio_augmentation_enabled": bool(
                self._classification_cfg().stage2_ratio_augmentation_enabled
            ),
            "stage2_ratio_max_features": int(self._classification_cfg().stage2_ratio_max_features),
            "stage2_ratio_selection_method": str(
                self._classification_cfg().stage2_ratio_selection_method
            ),
            "stage2_ratio_epsilon": float(self._classification_cfg().stage2_ratio_epsilon),
            "classifier_conformal_enabled": bool(self._classification_cfg().conformal_enabled),
            "classifier_conformal_alpha": float(self._classification_cfg().conformal_alpha),
            "classifier_conformal_calibration_fraction": float(
                self._classification_cfg().conformal_calibration_fraction
            ),
            "classifier_conformal_min_calibration": int(
                self._classification_cfg().conformal_min_calibration
            ),
            "classifier_conformal_output_sets": bool(
                self._classification_cfg().conformal_output_sets
            ),
            "classifier_conformal_method": str(
                getattr(self._classification_cfg(), "conformal_method", "split") or "split"
            ).strip().lower(),
            "enable_maqc_pairing": self.config.enable_maqc_pairing,
            "maqc_pairing_method_set_names": list(self.config.maqc_pairing_method_set_names or ()),
            "maqc_pairing_method_sets": [list(methods) for methods in (self.config.maqc_pairing_method_sets or ())],
            "maqc_pairing_min_improvement": self.config.maqc_pairing_min_improvement,
            "maqc_pairing_min_improvement_se_mult": self.config.maqc_pairing_min_improvement_se_mult,
            "dist_config": {
                "family_set": str(getattr(self.config.dist_config, "family_set", "v6") or "v6"),
                "robust_mode": self.config.dist_config.robust_mode,
                "use_adaptive_strategy": self.config.dist_config.use_adaptive_strategy,
                "use_lrt": self.config.dist_config.use_lrt,
                "use_cv": self.config.dist_config.use_cv,
                "compute_budget": self.config.dist_config.compute_budget,
                "use_support_filtering": self.config.dist_config.use_support_filtering,
                "rejection_gate": self.config.dist_config.rejection_gate,
                "rejection_p_threshold": self.config.dist_config.rejection_p_threshold,
                "confidence_margin": self.config.dist_config.confidence_margin,
                "compute_ad": bool(getattr(self.config.dist_config, "compute_ad", False)),
                "ad_bootstrap_samples": int(getattr(self.config.dist_config, "ad_bootstrap_samples", 0) or 0),
                "compute_qq_pp": bool(getattr(self.config.dist_config, "compute_qq_pp", False)),
                "compute_dip": bool(getattr(self.config.dist_config, "compute_dip", False)),
                "dip_hist_bins": int(getattr(self.config.dist_config, "dip_hist_bins", 40) or 40),
                "use_lmoment_prescreen": bool(getattr(self.config.dist_config, "use_lmoment_prescreen", False)),
                "lmoment_prescreen_max_candidates": int(
                    getattr(self.config.dist_config, "lmoment_prescreen_max_candidates", 0) or 0
                ),
                "estimator": str(getattr(self.config.dist_config, "estimator", "mle") or "mle"),
                "mps_maxiter": int(getattr(self.config.dist_config, "mps_maxiter", 250) or 250),
                "mps_tol": float(getattr(self.config.dist_config, "mps_tol", 1e-6) or 1e-6),
                "compute_crps": bool(getattr(self.config.dist_config, "compute_crps", False)),
                "crps_mc_samples": int(getattr(self.config.dist_config, "crps_mc_samples", 96) or 96),
                "crps_data_subsample": int(getattr(self.config.dist_config, "crps_data_subsample", 256) or 256),
                "compute_crps_uq_decomposition": bool(
                    getattr(self.config.dist_config, "compute_crps_uq_decomposition", False)
                ),
                "mnpo_include_crps": bool(getattr(self.config.dist_config, "mnpo_include_crps", False)),
                "random_state": getattr(self.config.dist_config, "random_state", None),
            },
            "multimodal_fallback": str(getattr(self.config, "multimodal_fallback", "none") or "none"),
        }

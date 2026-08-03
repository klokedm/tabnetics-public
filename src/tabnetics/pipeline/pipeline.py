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
from functools import wraps
from itertools import combinations
import logging  # Added for T-AUDIT-001-FIX-004
import math
import pickle
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import os as _os_setup
import zlib

from tabnetics.core.runtime import (
    configure_runtime_environment,
    has_nvidia_gpu,
    resolve_sklearn_n_jobs,
    sklearn_n_jobs_scope,
)

configure_runtime_environment()
_HAS_NVIDIA_GPU = has_nvidia_gpu()
del _os_setup

import numpy as np
import scipy.stats as sps
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import PolynomialCountSketch, RBFSampler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss as sklearn_log_loss, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .resampling import (
    FitResamplingContext,
    ResolvedSplitPlan,
    ResamplingContractError,
    ResamplingPolicy,
    SplitAssignment,
    coerce_sample_weights,
    ensure_fit_resampling_context,
    require_supported_resampling,
    resolve_assignment,
    resolve_cv,
    resolve_fit_subsample,
    resolve_holdout,
    typed_scalar_key,
)
from .balancing import (  # noqa: E402
    TrainingBalanceConfig,
    TrainingBalanceContractError,
    TrainingBalanceResult,
    apply_training_balance,
)

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

# Feature selection is part of every canonical benchmark execution.  Keep the
# class reference captured at module import so a later ``sys.modules`` shadow
# cannot redirect the pipeline's lazy selector lookup.
from tabnetics.feature_selection import FeatureSelector as _CANONICAL_FEATURE_SELECTOR

try:
    from tabnetics.classification.backends import (
        ClassifierBackend,
        FLAMLBackend,
        MNPOClassifierBackend,
        NativeCategoricalStage2Adapter,
        NativeCategoricalStage2Error,
        OptunaBackend,
        SampleWeightRoutingError,
        SklearnBackend,
        fit_native_categorical_stage2_singleton,
        fit_estimator_with_sample_weight,
        resolve_native_categorical_stage2_adapter,
    )
except Exception as exc:
    from tabnetics.classification.backends import (  # type: ignore
        ClassifierBackend,
        FLAMLBackend,
        MNPOClassifierBackend,
        NativeCategoricalStage2Adapter,
        NativeCategoricalStage2Error,
        OptunaBackend,
        SampleWeightRoutingError,
        SklearnBackend,
        fit_native_categorical_stage2_singleton,
        fit_estimator_with_sample_weight,
        resolve_native_categorical_stage2_adapter,
    )

from tabnetics.classification.conformance import (
    FittedClassifierDescriptor,
    ProbabilityRequirement,
    extract_probability_matrix,
    inspect_fitted_classifier,
)
from tabnetics.classification.registry import (
    ClassifierCapabilityOverrides,
    ClassifierRuntimeFacts,
    DEFAULT_CLASSIFIER_REGISTRY,
    SupportLevel,
    resolve_classifier_capabilities,
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
from tabnetics.datasets.tier_classifier import (
    adaptive_prefilter_top_k as _adaptive_prefilter_top_k,
    normalized_complexity_score as _normalized_complexity_score,
    predict_tier_with_details as _predict_tier_with_details,
)
from tabnetics.datasets.schema import (
    DatasetSchema,
    FeatureRole,
    SchemaContractError,
    infer_dataset_schema,
)

from .preprocessing import (
    FeatureSelectorRuntimeFacts,
    FoldLocalPreprocessor,
    NativeCategoricalStage2Bridge,
    TypedInputCapabilityError,
    admit_feature_selector_methods,
    guarded_sparse_to_dense,
    is_sparse_input,
    is_typed_input,
)

try:
    from sklearn.frozen import FrozenEstimator
except Exception:
    FrozenEstimator = None  # type: ignore

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


def _load_feature_selector_cls(
    _sealed_feature_selector: type = _CANONICAL_FEATURE_SELECTOR,
):
    """Return the import-time selector reference used by benchmark execution."""

    return _sealed_feature_selector


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
_DIAKRINO_REGIME_CONDITIONAL_METHODS: Tuple[str, ...] = ("diakrino_prior", "diakrino_screening_prior")
_DIAKRINO_PROTECTED_CORE_EXCLUDED_METHODS: Tuple[str, ...] = (
    "diakrino_prior",
    "diakrino_screening_prior",
    "diakrino_conformal_selection",
)
_DIAKRINO_REGIME_CONDITIONAL_ALLOWED_REGIMES: Tuple[str, ...] = (
    "hdlss_extreme",
    "hdlss_moderate",
)
_DEPRECATED_TOGGLE_WARNED: Set[str] = set()
__tabnetics_execution_isolated_state__ = {
    "_HAS_NVIDIA_GPU": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}
__tabnetics_execution_ephemeral_globals__ = ("_DEPRECATED_TOGGLE_WARNED",)

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


class _FixedIndexFeatureSelector:
    """Replay a finalized feature set while retaining selector diagnostics."""

    def __init__(self, selected_indices: Sequence[int], base_selector: Optional[Any] = None):
        self._selected = np.asarray(list(selected_indices), dtype=int).ravel()
        self.base_selector = base_selector
        self.mnpo_diagnostics_ = dict(getattr(base_selector, "mnpo_diagnostics_", {}) or {})

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, self._selected]

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
    # Opt-in tiny/heaped-data guard: skip scipy families with shape params
    # unless there are enough distinct values to identify them.
    flex_family_distinct_gate_enabled: bool = False
    flex_family_min_distinct: int = 15
    # Opt-in simple-score retention: a >=3-param selected family must beat the
    # best simple family by more than this V6 simple_score margin.
    flex_family_retention_margin: float = 0.0
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
    # DIAKRINO population-family priors over TRAIN-split fitting (native integration §3;
    # opt-in, default off; scipy MLE stays ground truth — these only shrink/route/skip).
    # All gated on a persisted sidecar; graceful no-op when absent.
    diakrino_family_prescreen_enabled: bool = False   # §3.1 intersect candidate families w/ top-K
    diakrino_family_prescreen_top_k: int = 4
    diakrino_family_prescreen_keep_mandatory: bool = True  # never empty the candidate set
    diakrino_family_prior_lambda: float = 0.0          # §3.1b soft log-P family prior; 0.0 == no-op
    diakrino_skip_fit_discrete_enabled: bool = False  # §3.6 family argmax in discrete ids -> rank-gaussian
    diakrino_warm_start_enabled: bool = False         # §3.2 BLOCKED: param head degenerate (see PARAM_PROBE_PLAN.md)
    diakrino_sidecar_path: str = ""                    # dir/manifest/feature_logits parquet; "" => disabled
    diakrino_sidecar_dataset_id: str = ""              # optional dataset id for sidecar roots/manifests
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
    include_tabentics_diakrino_model: bool = False
    # T-DATA-NATIVE-STAGE2-01: narrow, explicitly requested native categorical
    # route.  The default remains the numeric Stage-2 path.
    native_categorical_stage2_enabled: bool = False
    native_categorical_stage2_estimator: str = ""
    tabentics_diakrino_checkpoint: str = ""
    tabentics_diakrino_max_features: int = 256
    tabentics_diakrino_batch_size: int = 32
    tabentics_diakrino_support_joint_serving_cache: bool = False
    tabentics_diakrino_retry_cuda_oom_microbatch: bool = False
    tabentics_diakrino_device: str = "auto"
    tabentics_diakrino_calibrate_probabilities: bool = True
    tabentics_diakrino_calibration_fraction: float = 0.20
    use_hybrid_score: bool = False
    hybrid_balanced_weight: float = 0.6
    hybrid_macro_f1_weight: float = 0.4
    # Val-11 Promotion (Profile D): runtime containment + explicit 8-candidate cap.
    runtime_containment_enabled: bool = True
    runtime_max_candidates: int = 8
    runtime_high_p_over_n_threshold: float = 40.0
    runtime_high_class_threshold: int = 6
    runtime_min_class_count_threshold: int = 12
    enable_svc_probability: bool = False
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
    oracle_include_worst_class_recall: bool = True
    oracle_include_james_stein: bool = True
    oracle_complexity_shrinkage: bool = False
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
    oracle_include_diakrino_family_meta: bool = False  # report-only DIAKRINO family/logit meta-features
    # T-R-380: opt-in post-hoc probability calibration stage.
    posthoc_calibration_enabled: bool = False
    posthoc_calibration_method: str = "sigmoid"  # sigmoid | isotonic
    posthoc_calibration_fraction: float = 0.20
    posthoc_calibration_min_calibration: int = 20
    posthoc_calibration_refinement_stopping: bool = True
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
        self.n_jobs = resolve_sklearn_n_jobs(getattr(self, "n_jobs", 1))
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
        self.native_categorical_stage2_enabled = bool(
            getattr(self, "native_categorical_stage2_enabled", False)
        )
        self.native_categorical_stage2_estimator = str(
            getattr(self, "native_categorical_stage2_estimator", "") or ""
        ).strip()
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
        method = str(self.posthoc_calibration_method or "sigmoid").strip().lower()
        if method not in {"sigmoid", "isotonic"}:
            method = "sigmoid"
        self.posthoc_calibration_method = method
        self.posthoc_calibration_fraction = float(
            np.clip(float(self.posthoc_calibration_fraction), 0.05, 0.50)
        )
        self.posthoc_calibration_min_calibration = int(
            max(2, int(self.posthoc_calibration_min_calibration))
        )
        self.posthoc_calibration_refinement_stopping = bool(
            self.posthoc_calibration_refinement_stopping
        )
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

    # T-CORE-BALANCE-01: opt-in training-fold balancing.  This state is never
    # replayed at inference and the default preserves the existing fit path.
    training_balance: TrainingBalanceConfig = field(
        default_factory=TrainingBalanceConfig
    )

    # T-DATA-TYPED-01: opt-in typed input boundary.  Legacy dense numeric
    # ndarrays retain the historical direct coercion path while this is false.
    typed_input_enabled: bool = False
    typed_text_encoding: str = "tfidf_hash"  # tfidf_hash | length_hash | drop
    typed_text_hash_buckets: int = 16  # TF-IDF hash buckets
    typed_sparse_dense_max_elements: int = 2_000_000

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

    # ── DIAKRINO native integration (opt-in, default off; all no-op without a sidecar) ──
    diakrino_sidecar_path: str = ""                       # dir/manifest/feature_logits parquet ("" => disabled)
    diakrino_sidecar_dataset_id: str = ""                 # optional dataset id for sidecar roots/manifests
    # DIAKRINO prefilter augmentation. ``protected_union`` keeps the classical
    # prefilter/selector path independent, then appends a separately bounded DIAKRINO
    # set. ``legacy_fixed_budget_blend`` is retained only for exact experiment
    # replay; it can evict classical incumbents and must not be described as a
    # union or used for promotion evidence.
    diakrino_prefilter_enabled: bool = False
    diakrino_prefilter_mode: str = "protected_union"      # protected_union | legacy_fixed_budget_blend
    # Compatibility activation scalar; protected_union uses only its sign,
    # while legacy_fixed_budget_blend uses its clipped magnitude as the blend weight.
    diakrino_prefilter_lambda: float = 0.0                # <=0 (or disabled) == bit-identical baseline
    diakrino_prefilter_max_extras: int = 50               # separate DIAKRINO-only addition budget
    diakrino_prefilter_score_column: str = "prior_logit"  # calibration-safe surface only
    diakrino_prefilter_shadow_probe_indices: Tuple[int, ...] = tuple()
    # §3.3 CDF-trust gate: low family-confidence (high normalized entropy) routes the
    # feature to the existing cheap rank-gaussian fallback (no-drop). Decision persisted
    # to feature_plans for replay parity. threshold>=1.0 (or disabled) == no-op.
    diakrino_cdf_trust_gate_enabled: bool = False
    diakrino_cdf_trust_entropy_threshold: float = 1.01    # normalized family entropy in [0,1]; >=1.0 never fires
    diakrino_cdf_trust_fallback: str = "rank_gaussian"    # rank_gaussian | drop
    # §3.4 entropy stability surrogate (replaces the bootstrap stability weight)
    diakrino_stability_surrogate_enabled: bool = False
    # T-DIAKRINO-IMP-08: when enabled, explicitly configured DIAKRINO candidate selectors
    # run only in HDLSS regimes. Default off preserves the enabled_methods contract.
    diakrino_regime_conditional: bool = False
    # DIAKRINO feature-selector candidate/oracle bridge. These flags only pass a persisted
    # sidecar into opt-in DIAKRINO selector methods / MNPO oracles; empty path + false
    # oracle flags preserve the baseline selector exactly.
    diakrino_prior_score_column: str = "prior_logit"
    diakrino_screening_score_column: str = "screening_logit"
    diakrino_prior_calibrate: str = "chunk_zscore"
    diakrino_prior_top_k: int = 0
    diakrino_conformal_selection_enabled: bool = False
    diakrino_conformal_target_fdp: float = 0.20
    diakrino_conformal_calibrate: str = "chunk_zscore"
    diakrino_conformal_null_fraction: float = 0.50
    diakrino_conformal_min_null_scores: int = 4
    diakrino_conformal_max_features: int = 0
    diakrino_conformal_qualification_record: str = ""
    fs_use_diakrino_relevance_oracle: bool = False
    fs_diakrino_relevance_min_n_train: int = 100
    fs_diakrino_relevance_score_column: str = "prior_logit"

    # Feature prefilter inspired by stability/rank integration literature
    use_rank_prefilter: bool = True
    prefilter_top_k: Optional[int] = 600
    prefilter_adaptive_top_k: bool = False
    prefilter_adaptive_top_k_scaling: float = 0.5
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
    fs_method_max_rss_mb: float = 0.0
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
    tier_classifier_mode: str = "heuristic"  # heuristic | learned
    tier_classifier_model_path: str = ""
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
    regime_gating_min_samples_per_class: float = 7.0
    regime_gating_use_expanded_features: bool = False
    regime_gating_min_fisher_f1: float = 0.10
    regime_gating_max_n1_borderline: float = 0.40
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
    fs_oracle_enable_stability: bool = True
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
    fs_oracle_complexity_conditioning: bool = False
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
    # T-CORE-HARDEN-MRMR-RUNTIME-01: zero means unlimited and preserves the
    # historical selector behavior. Any nonzero cap is explicitly opt-in.
    fs_mrmr_max_unique_pair_evaluations: int = 0
    fs_mrmr_max_runtime_seconds: float = 0.0
    fs_mrmr_budget_fallback_mode: str = "empty"
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
    fs_pathway_group_sparse_lasso_n_groups: int = 50
    fs_pathway_group_sparse_lasso_max_group_size: int = 50
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
    include_tabentics_diakrino_model: bool = False
    native_categorical_stage2_enabled: bool = False
    native_categorical_stage2_estimator: str = ""
    tabentics_diakrino_checkpoint: str = ""
    tabentics_diakrino_max_features: int = 256
    tabentics_diakrino_batch_size: int = 32
    tabentics_diakrino_support_joint_serving_cache: bool = False
    tabentics_diakrino_retry_cuda_oom_microbatch: bool = False
    tabentics_diakrino_device: str = "auto"
    tabentics_diakrino_calibrate_probabilities: bool = True
    tabentics_diakrino_calibration_fraction: float = 0.20
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
    model_cv_enable_svc_probability: bool = False
    classifier_oracle_k: int = 1
    classifier_oracle_weighting_mode: str = "tritrust"
    classifier_oracle_include_robustness: bool = True
    classifier_oracle_include_complexity: bool = True
    classifier_oracle_include_calibration: bool = True
    classifier_oracle_enable_worst_class: bool = True
    classifier_oracle_include_james_stein: bool = True
    classifier_oracle_complexity_shrinkage: bool = False
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
    classifier_oracle_include_diakrino_family_meta: bool = False
    # VAL12_Suggestions §2.1-2.2: opt-in calibration reporting (ECE alongside Brier).
    calibration_reporting_enabled: bool = False
    classifier_posthoc_calibration_enabled: bool = False
    classifier_posthoc_calibration_method: str = "sigmoid"
    classifier_posthoc_calibration_fraction: float = 0.20
    classifier_posthoc_calibration_min_calibration: int = 20
    classifier_posthoc_calibration_refinement_stopping: bool = True
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
    # Nested pairing is intentionally bounded only when an explicit limit is
    # supplied. A partial nested evaluation is never promoted to raw-CV.
    maqc_pairing_max_outer_evaluations: int = 0
    maqc_pairing_max_runtime_seconds: float = 0.0
    multiomics_adapter: str = "none"
    multiomics_integrator: str = "mb_plsda"
    multiomics_n_components: int = 2
    multiomics_feature_blocks: Optional[Dict[str, Tuple[int, ...]]] = None
    # V25 packaged score router. Keep opt-in until its frozen registered-holdout
    # campaign passes the portfolio promotion gate.
    auto_router_enabled: bool = False
    auto_router_artifact_path: str = ""
    auto_router_fail_open: bool = True
    auto_router_descriptor_ood_gate_enabled: bool = False
    # Requires a validated cross-fit uncertainty artifact embedded in the router.
    # Disabled by default; legacy router behavior is unchanged unless enabled.
    auto_router_crossfit_uncertainty_enabled: bool = False
    diakrino_router_dispersion_descriptor_enabled: bool = False
    meta_learning_selector_mode: str = "none"
    meta_learning_confidence_threshold: float = 0.55
    meta_learning_records_path: str = ""

    def __post_init__(self) -> None:
        self.n_jobs = resolve_sklearn_n_jobs(getattr(self, "n_jobs", 1))
        self.typed_input_enabled = bool(getattr(self, "typed_input_enabled", False))
        typed_text_encoding = str(
            getattr(self, "typed_text_encoding", "tfidf_hash") or "tfidf_hash"
        ).strip().lower()
        if typed_text_encoding not in {"tfidf_hash", "length_hash", "drop"}:
            typed_text_encoding = "tfidf_hash"
        self.typed_text_encoding = str(typed_text_encoding)
        self.typed_text_hash_buckets = int(
            max(1, int(getattr(self, "typed_text_hash_buckets", 16) or 16))
        )
        self.typed_sparse_dense_max_elements = int(
            max(0, int(getattr(self, "typed_sparse_dense_max_elements", 2_000_000) or 0))
        )
        try:
            self.fs_method_max_rss_mb = float(self.fs_method_max_rss_mb)
        except (TypeError, ValueError) as exc:
            raise ValueError("fs_method_max_rss_mb must be a finite float >= 0") from exc
        if (
            not math.isfinite(self.fs_method_max_rss_mb)
            or self.fs_method_max_rss_mb < 0.0
        ):
            raise ValueError("fs_method_max_rss_mb must be a finite float >= 0")
        try:
            self.fs_mrmr_max_unique_pair_evaluations = int(
                self.fs_mrmr_max_unique_pair_evaluations
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fs_mrmr_max_unique_pair_evaluations must be an integer >= 0"
            ) from exc
        if self.fs_mrmr_max_unique_pair_evaluations < 0:
            raise ValueError(
                "fs_mrmr_max_unique_pair_evaluations must be >= 0 (0 means unlimited)"
            )
        try:
            self.fs_mrmr_max_runtime_seconds = float(
                self.fs_mrmr_max_runtime_seconds
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fs_mrmr_max_runtime_seconds must be a finite float >= 0"
            ) from exc
        if (
            not math.isfinite(self.fs_mrmr_max_runtime_seconds)
            or self.fs_mrmr_max_runtime_seconds < 0.0
        ):
            raise ValueError(
                "fs_mrmr_max_runtime_seconds must be a finite float >= 0 (0 means unlimited)"
            )
        self.fs_mrmr_budget_fallback_mode = str(
            self.fs_mrmr_budget_fallback_mode or "empty"
        ).strip().lower()
        if self.fs_mrmr_budget_fallback_mode not in {"empty", "relevance_only"}:
            raise ValueError(
                "fs_mrmr_budget_fallback_mode must be 'empty' or 'relevance_only'"
            )
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
        maqc_score_mode = str(
            getattr(self, "maqc_pairing_score_mode", "raw_cv") or "raw_cv"
        ).strip().lower()
        if maqc_score_mode not in {"raw_cv", "nested_cv", "nested_bbc"}:
            raise ValueError(
                "maqc_pairing_score_mode must be 'raw_cv', 'nested_cv', or "
                "'nested_bbc'."
            )
        self.maqc_pairing_score_mode = str(maqc_score_mode)
        # ``raw_cv`` is the established path.  Preserve its formerly inert
        # nested-only fields exactly; strict coercion begins only when a nested
        # evaluator is explicitly requested.
        if self.maqc_pairing_score_mode != "raw_cv":
            self.maqc_pairing_outer_splits = int(
                getattr(self, "maqc_pairing_outer_splits", 3) or 0
            )
            if self.maqc_pairing_outer_splits < 2:
                raise ValueError("maqc_pairing_outer_splits must be >= 2.")
            self.maqc_pairing_outer_repeats = int(
                getattr(self, "maqc_pairing_outer_repeats", 1) or 0
            )
            if self.maqc_pairing_outer_repeats < 1:
                raise ValueError("maqc_pairing_outer_repeats must be >= 1.")
            self.maqc_pairing_min_train_per_class = int(
                getattr(self, "maqc_pairing_min_train_per_class", 2) or 0
            )
            if self.maqc_pairing_min_train_per_class < 1:
                raise ValueError("maqc_pairing_min_train_per_class must be >= 1.")
            self.maqc_pairing_seed_stride = int(
                getattr(self, "maqc_pairing_seed_stride", 997) or 0
            )
            if self.maqc_pairing_seed_stride < 1:
                raise ValueError("maqc_pairing_seed_stride must be >= 1.")
            self.maqc_pairing_bbc_bootstrap_rounds = int(
                getattr(self, "maqc_pairing_bbc_bootstrap_rounds", 200) or 0
            )
            if (
                self.maqc_pairing_score_mode == "nested_bbc"
                and self.maqc_pairing_bbc_bootstrap_rounds < 10
            ):
                raise ValueError(
                    "maqc_pairing_bbc_bootstrap_rounds must be >= 10."
                )
            self.maqc_pairing_bbc_ci_level = float(
                getattr(self, "maqc_pairing_bbc_ci_level", 0.90) or 0.0
            )
            if (
                self.maqc_pairing_score_mode == "nested_bbc"
                and not 0.0 < self.maqc_pairing_bbc_ci_level < 1.0
            ):
                raise ValueError(
                    "maqc_pairing_bbc_ci_level must be strictly between 0 and 1."
                )
            self.maqc_pairing_max_outer_evaluations = int(
                getattr(self, "maqc_pairing_max_outer_evaluations", 0) or 0
            )
            if self.maqc_pairing_max_outer_evaluations < 0:
                raise ValueError("maqc_pairing_max_outer_evaluations must be >= 0.")
            self.maqc_pairing_max_runtime_seconds = float(
                getattr(self, "maqc_pairing_max_runtime_seconds", 0.0) or 0.0
            )
            if (
                not math.isfinite(self.maqc_pairing_max_runtime_seconds)
                or self.maqc_pairing_max_runtime_seconds < 0.0
            ):
                raise ValueError(
                    "maqc_pairing_max_runtime_seconds must be a finite float >= 0."
                )
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
        self.auto_router_enabled = bool(getattr(self, "auto_router_enabled", False))
        self.auto_router_artifact_path = str(
            getattr(self, "auto_router_artifact_path", "") or ""
        ).strip()
        self.auto_router_fail_open = bool(getattr(self, "auto_router_fail_open", True))
        self.auto_router_descriptor_ood_gate_enabled = bool(
            getattr(self, "auto_router_descriptor_ood_gate_enabled", False)
        )
        self.auto_router_crossfit_uncertainty_enabled = bool(
            getattr(self, "auto_router_crossfit_uncertainty_enabled", False)
        )
        meta_learning_mode = str(
            getattr(self, "meta_learning_selector_mode", "none") or "none"
        ).strip().lower()
        if meta_learning_mode not in {"none", "decision_tree", "logistic", "flag_selector_v1"}:
            meta_learning_mode = "none"
        self.meta_learning_selector_mode = str(meta_learning_mode)
        self.meta_learning_confidence_threshold = float(
            np.clip(
                float(getattr(self, "meta_learning_confidence_threshold", 0.55) or 0.55),
                0.0,
                1.0,
            )
        )
        self.meta_learning_records_path = str(
            getattr(self, "meta_learning_records_path", "") or ""
        ).strip()
        tier_classifier_mode = str(
            getattr(self, "tier_classifier_mode", "heuristic") or "heuristic"
        ).strip().lower()
        if tier_classifier_mode not in {"heuristic", "learned"}:
            tier_classifier_mode = "heuristic"
        self.tier_classifier_mode = str(tier_classifier_mode)
        self.tier_classifier_model_path = str(
            getattr(self, "tier_classifier_model_path", "") or ""
        ).strip()
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
            max(1.0, float(getattr(self, "regime_gating_min_samples_per_class", 7.0) or 7.0))
        )
        self.regime_gating_use_expanded_features = bool(
            getattr(self, "regime_gating_use_expanded_features", False)
        )
        self.regime_gating_min_fisher_f1 = float(
            max(0.0, float(getattr(self, "regime_gating_min_fisher_f1", 0.10) or 0.10))
        )
        self.regime_gating_max_n1_borderline = float(
            np.clip(
                float(getattr(self, "regime_gating_max_n1_borderline", 0.40) or 0.40),
                0.0,
                1.0,
            )
        )
        self.regime_gating_low_p_over_n_threshold = float(
            max(0.0, float(getattr(self, "regime_gating_low_p_over_n_threshold", 0.0) or 0.0))
        )
        self.prefilter_adaptive_top_k = bool(getattr(self, "prefilter_adaptive_top_k", False))
        self.prefilter_adaptive_top_k_scaling = float(
            max(0.0, float(getattr(self, "prefilter_adaptive_top_k_scaling", 0.5) or 0.5))
        )
        self.diakrino_sidecar_path = str(getattr(self, "diakrino_sidecar_path", "") or "")
        self.diakrino_sidecar_dataset_id = str(getattr(self, "diakrino_sidecar_dataset_id", "") or "")
        diakrino_prefilter_mode = str(
            getattr(self, "diakrino_prefilter_mode", "protected_union") or "protected_union"
        ).strip().lower()
        if diakrino_prefilter_mode not in {"protected_union", "legacy_fixed_budget_blend"}:
            diakrino_prefilter_mode = "protected_union"
        self.diakrino_prefilter_mode = str(diakrino_prefilter_mode)
        self.diakrino_prefilter_max_extras = int(
            max(0, int(getattr(self, "diakrino_prefilter_max_extras", 50) or 0))
        )
        try:
            shadow_probe_indices = np.asarray(
                list(getattr(self, "diakrino_prefilter_shadow_probe_indices", tuple()) or tuple()),
                dtype=int,
            ).ravel()
        except Exception:
            shadow_probe_indices = np.array([], dtype=int)
        self.diakrino_prefilter_shadow_probe_indices = tuple(
            sorted({int(i) for i in shadow_probe_indices.tolist() if int(i) >= 0})
        )
        self.diakrino_prior_score_column = str(getattr(self, "diakrino_prior_score_column", "prior_logit") or "prior_logit")
        self.diakrino_screening_score_column = str(
            getattr(self, "diakrino_screening_score_column", "screening_logit") or "screening_logit"
        )
        diakrino_prior_calibrate = str(getattr(self, "diakrino_prior_calibrate", "chunk_zscore") or "chunk_zscore").strip()
        if diakrino_prior_calibrate not in {
            "none",
            "chunk_zscore",
            "chunk_rank01",
            "chunk_ecdf",
            "chunk_minmax",
            "chunk_robust_iqr",
            "chunk_softmax_temp",
            "blend",
        }:
            diakrino_prior_calibrate = "chunk_zscore"
        self.diakrino_prior_calibrate = str(diakrino_prior_calibrate)
        self.diakrino_prior_top_k = int(max(0, int(getattr(self, "diakrino_prior_top_k", 0) or 0)))
        self.diakrino_conformal_selection_enabled = bool(
            getattr(self, "diakrino_conformal_selection_enabled", False)
        )
        _diakrino_conf_target = getattr(self, "diakrino_conformal_target_fdp", 0.20)
        if _diakrino_conf_target is None:
            _diakrino_conf_target = 0.20
        self.diakrino_conformal_target_fdp = float(
            min(1.0, max(0.0, float(_diakrino_conf_target)))
        )
        diakrino_conformal_calibrate = str(
            getattr(self, "diakrino_conformal_calibrate", "chunk_zscore") or "chunk_zscore"
        ).strip()
        if diakrino_conformal_calibrate not in {
            "chunk_zscore",
            "chunk_rank01",
            "chunk_ecdf",
            "chunk_minmax",
            "chunk_robust_iqr",
            "chunk_softmax_temp",
            "blend",
        }:
            diakrino_conformal_calibrate = "chunk_zscore"
        self.diakrino_conformal_calibrate = str(diakrino_conformal_calibrate)
        _diakrino_conf_null_fraction = getattr(self, "diakrino_conformal_null_fraction", 0.50)
        if _diakrino_conf_null_fraction is None:
            _diakrino_conf_null_fraction = 0.50
        self.diakrino_conformal_null_fraction = float(
            min(1.0, max(0.0, float(_diakrino_conf_null_fraction)))
        )
        self.diakrino_conformal_min_null_scores = int(
            max(1, int(getattr(self, "diakrino_conformal_min_null_scores", 4) or 4))
        )
        self.diakrino_conformal_max_features = int(
            max(0, int(getattr(self, "diakrino_conformal_max_features", 0) or 0))
        )
        self.diakrino_conformal_qualification_record = str(
            getattr(self, "diakrino_conformal_qualification_record", "") or ""
        )
        self.fs_use_diakrino_relevance_oracle = bool(getattr(self, "fs_use_diakrino_relevance_oracle", False))
        self.fs_diakrino_relevance_min_n_train = int(
            max(2, int(getattr(self, "fs_diakrino_relevance_min_n_train", 100) or 100))
        )
        self.fs_diakrino_relevance_score_column = str(
            getattr(self, "fs_diakrino_relevance_score_column", "prior_logit") or "prior_logit"
        )
        self.fs_oracle_complexity_conditioning = bool(
            getattr(self, "fs_oracle_complexity_conditioning", False)
        )
        self.classifier_oracle_complexity_shrinkage = bool(
            getattr(self, "classifier_oracle_complexity_shrinkage", False)
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
        stability_oracle_enabled = bool(getattr(self, "use_stability_oracle", True)) and bool(
            getattr(self, "fs_oracle_enable_stability", True)
        )
        self.use_stability_oracle = bool(stability_oracle_enabled)
        self.fs_oracle_enable_stability = bool(stability_oracle_enabled)

        if self.training_balance is None:
            self.training_balance = TrainingBalanceConfig()
        elif not isinstance(self.training_balance, TrainingBalanceConfig):
            self.training_balance = TrainingBalanceConfig.from_mapping(
                self.training_balance
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
            ("include_tabentics_diakrino_model", "include_tabentics_diakrino_model"),
            (
                "native_categorical_stage2_enabled",
                "native_categorical_stage2_enabled",
            ),
            (
                "native_categorical_stage2_estimator",
                "native_categorical_stage2_estimator",
            ),
            ("tabentics_diakrino_checkpoint", "tabentics_diakrino_checkpoint"),
            ("tabentics_diakrino_max_features", "tabentics_diakrino_max_features"),
            ("tabentics_diakrino_batch_size", "tabentics_diakrino_batch_size"),
            (
                "tabentics_diakrino_support_joint_serving_cache",
                "tabentics_diakrino_support_joint_serving_cache",
            ),
            (
                "tabentics_diakrino_retry_cuda_oom_microbatch",
                "tabentics_diakrino_retry_cuda_oom_microbatch",
            ),
            ("tabentics_diakrino_device", "tabentics_diakrino_device"),
            ("tabentics_diakrino_calibrate_probabilities", "tabentics_diakrino_calibrate_probabilities"),
            ("tabentics_diakrino_calibration_fraction", "tabentics_diakrino_calibration_fraction"),
            ("model_cv_lr_max_iter", "lr_max_iter"),
            ("model_cv_use_hybrid_score", "use_hybrid_score"),
            ("model_cv_balanced_weight", "hybrid_balanced_weight"),
            ("model_cv_macro_f1_weight", "hybrid_macro_f1_weight"),
            ("model_cv_runtime_containment_enabled", "runtime_containment_enabled"),
            ("model_cv_runtime_max_candidates", "runtime_max_candidates"),
            ("model_cv_runtime_high_p_over_n_threshold", "runtime_high_p_over_n_threshold"),
            ("model_cv_runtime_high_class_threshold", "runtime_high_class_threshold"),
            ("model_cv_runtime_min_class_count_threshold", "runtime_min_class_count_threshold"),
            ("model_cv_enable_svc_probability", "enable_svc_probability"),
            ("classifier_oracle_k", "oracle_k"),
            ("classifier_oracle_weighting_mode", "oracle_weighting_mode"),
            ("classifier_oracle_include_robustness", "oracle_include_robustness"),
            ("classifier_oracle_include_complexity", "oracle_include_complexity"),
            ("classifier_oracle_include_calibration", "oracle_include_calibration"),
            ("classifier_oracle_enable_worst_class", "oracle_include_worst_class_recall"),
            ("classifier_oracle_include_james_stein", "oracle_include_james_stein"),
            ("classifier_oracle_complexity_shrinkage", "oracle_complexity_shrinkage"),
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
            ("classifier_oracle_include_diakrino_family_meta", "oracle_include_diakrino_family_meta"),
            ("classifier_posthoc_calibration_enabled", "posthoc_calibration_enabled"),
            ("classifier_posthoc_calibration_method", "posthoc_calibration_method"),
            ("classifier_posthoc_calibration_fraction", "posthoc_calibration_fraction"),
            ("classifier_posthoc_calibration_min_calibration", "posthoc_calibration_min_calibration"),
            (
                "classifier_posthoc_calibration_refinement_stopping",
                "posthoc_calibration_refinement_stopping",
            ),
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
        c.n_jobs = int(self.n_jobs)
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
        self.include_tabentics_diakrino_model = bool(getattr(c, "include_tabentics_diakrino_model", False))
        self.native_categorical_stage2_enabled = bool(
            getattr(c, "native_categorical_stage2_enabled", False)
        )
        self.native_categorical_stage2_estimator = str(
            getattr(c, "native_categorical_stage2_estimator", "") or ""
        )
        self.tabentics_diakrino_checkpoint = str(getattr(c, "tabentics_diakrino_checkpoint", "") or "")
        self.tabentics_diakrino_max_features = int(max(1, int(getattr(c, "tabentics_diakrino_max_features", 256) or 256)))
        self.tabentics_diakrino_batch_size = int(max(1, int(getattr(c, "tabentics_diakrino_batch_size", 32) or 32)))
        self.tabentics_diakrino_support_joint_serving_cache = bool(
            getattr(c, "tabentics_diakrino_support_joint_serving_cache", False)
        )
        self.tabentics_diakrino_retry_cuda_oom_microbatch = bool(
            getattr(c, "tabentics_diakrino_retry_cuda_oom_microbatch", False)
        )
        self.tabentics_diakrino_device = str(getattr(c, "tabentics_diakrino_device", "auto") or "auto")
        self.tabentics_diakrino_calibrate_probabilities = bool(
            getattr(c, "tabentics_diakrino_calibrate_probabilities", True)
        )
        self.tabentics_diakrino_calibration_fraction = float(
            np.clip(float(getattr(c, "tabentics_diakrino_calibration_fraction", 0.20) or 0.20), 0.0, 0.5)
        )
        self.model_cv_lr_max_iter = int(c.lr_max_iter)
        self.model_cv_use_hybrid_score = bool(c.use_hybrid_score)
        self.model_cv_balanced_weight = float(c.hybrid_balanced_weight)
        self.model_cv_macro_f1_weight = float(c.hybrid_macro_f1_weight)
        self.model_cv_runtime_containment_enabled = bool(c.runtime_containment_enabled)
        self.model_cv_runtime_max_candidates = int(c.runtime_max_candidates)
        self.model_cv_runtime_high_p_over_n_threshold = float(c.runtime_high_p_over_n_threshold)
        self.model_cv_runtime_high_class_threshold = int(c.runtime_high_class_threshold)
        self.model_cv_runtime_min_class_count_threshold = int(c.runtime_min_class_count_threshold)
        self.model_cv_enable_svc_probability = bool(c.enable_svc_probability)
        self.classifier_oracle_k = int(c.oracle_k)
        self.classifier_oracle_weighting_mode = str(c.oracle_weighting_mode)
        self.classifier_oracle_include_robustness = bool(c.oracle_include_robustness)
        self.classifier_oracle_include_complexity = bool(c.oracle_include_complexity)
        self.classifier_oracle_include_calibration = bool(c.oracle_include_calibration)
        self.classifier_oracle_enable_worst_class = bool(c.oracle_include_worst_class_recall)
        self.classifier_oracle_include_james_stein = bool(c.oracle_include_james_stein)
        self.classifier_oracle_complexity_shrinkage = bool(
            getattr(c, "oracle_complexity_shrinkage", False)
        )
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
        self.classifier_oracle_include_diakrino_family_meta = bool(c.oracle_include_diakrino_family_meta)
        self.classifier_posthoc_calibration_enabled = bool(c.posthoc_calibration_enabled)
        self.classifier_posthoc_calibration_method = str(c.posthoc_calibration_method)
        self.classifier_posthoc_calibration_fraction = float(c.posthoc_calibration_fraction)
        self.classifier_posthoc_calibration_min_calibration = int(c.posthoc_calibration_min_calibration)
        self.classifier_posthoc_calibration_refinement_stopping = bool(
            c.posthoc_calibration_refinement_stopping
        )
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
    classification_fitted_probability_kind: str = "unknown"
    classification_fitted_probability_source: str = "unavailable"
    classification_final_fitted_descriptor: Dict[str, Any] = field(
        default_factory=dict
    )
    resampling_context_fingerprint: str = ""
    fit_context_fingerprint: str = ""
    outer_split_fingerprint: str = ""
    resampling_policy: Dict[str, Any] = field(default_factory=dict)
    leakage_audit: Dict[str, Any] = field(default_factory=dict)
    resampling_trace: Dict[str, Any] = field(default_factory=dict)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    model_input_schema: Dict[str, Any] = field(default_factory=dict)
    selected_feature_schema: Dict[str, Any] = field(default_factory=dict)
    typed_preprocessing: Dict[str, Any] = field(default_factory=dict)


class NestedPairingEvaluationError(RuntimeError):
    """Fail closed when an opt-in nested pairing contract cannot be honored."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.diagnostics = _json_safe(dict(diagnostics or {}))


@dataclass(frozen=True)
class _NestedPairingFoldCapture:
    """Ephemeral row-aligned predictions for one candidate/outer fold.

    This object deliberately exists only while ``run_pre_split`` is resolving a
    nested pairing. Its raw vectors are used for the BBC procedure estimate and
    are reduced to aggregates before a public result is constructed.
    """

    candidate_name: str
    fold_index: int
    split_fingerprint: str
    test_indices: Tuple[int, ...]
    test_row_ids_fingerprint: str
    y_true: Tuple[Any, ...]
    y_pred: Tuple[Any, ...]
    sample_weights: Tuple[float, ...]
    balanced_accuracy: float
    leakage_audit: Dict[str, Any]


@dataclass(frozen=True)
class _NestedPairingRawContext:
    """Raw outer-train inputs needed to rebuild every nested fold safely."""

    X_train: Any
    y_train: np.ndarray
    schema: DatasetSchema | None
    batch_labels: np.ndarray | None
    fit_resampling_context: FitResamplingContext
    outer_plan: ResolvedSplitPlan
    dataset_name: str
    seed: int


@dataclass
class FittedPipelineComponents:
    """Train-only fitted state consumed by the public sklearn estimator.

    This deliberately contains no held-out metrics, assignments, or prediction
    artifacts.  Model/selector policy is resolved through the configured
    training-only CV path, then the selected model is refit on every supplied
    training row.
    """

    runtime_model: Any
    classes: np.ndarray
    fit_resampling_context: FitResamplingContext
    config_snapshot: Dict[str, Any]
    model_name: str
    source_schema: DatasetSchema | None = None
    model_input_schema: DatasetSchema | None = None
    selected_feature_schema: Dict[str, Any] = field(default_factory=dict)
    typed_preprocessor: FoldLocalPreprocessor | None = None
    typed_sparse_dense_max_elements: int = 0
    variance_keep_indices: Tuple[int, ...] = tuple()
    selected_model_input_indices: Tuple[int, ...] = tuple()
    fit_provenance: Dict[str, Any] = field(default_factory=dict)
    last_inference_schema_report: Dict[str, Any] = field(default_factory=dict)

    def _prepare_input(
        self,
        X: Any,
        *,
        alignment: str = "strict",
    ) -> np.ndarray:
        """Validate and transform one inference matrix without refitting state."""

        checked_X = X
        if self.source_schema is not None:
            aligner = getattr(self.source_schema, "align_inference_input", None)
            if callable(aligner):
                checked_X, report = aligner(X, alignment_mode=alignment)
                self.last_inference_schema_report = dict(report.to_record())
            else:  # Compatibility with bundles produced before the estimator API.
                if str(alignment).strip().lower() != "strict":
                    raise SchemaContractError(
                        "Inference alignment is unavailable for this fitted schema contract."
                    )
                self.source_schema.validate_input(X)
        elif self.typed_preprocessor is not None:
            raise RuntimeError("Typed fitted components are missing their input schema.")

        if self.typed_preprocessor is None:
            return np.asarray(checked_X, dtype=float)

        transformed = self.typed_preprocessor.transform_with_schema(
            checked_X,
            schema=self.source_schema,
            output_mode="numeric",
        )
        if is_sparse_input(transformed.X):
            return guarded_sparse_to_dense(
                transformed.X,
                max_elements=int(self.typed_sparse_dense_max_elements),
                callsite="fitted_estimator_inference",
            ).astype(float, copy=False)
        return np.asarray(transformed.X, dtype=float)

    def transform(self, X: Any, *, batch_labels: Optional[Sequence[Any]] = None, alignment: str = "strict") -> np.ndarray:
        matrix = self._prepare_input(X, alignment=alignment)
        return np.asarray(
            self.runtime_model.transform(matrix, batch_labels=batch_labels), dtype=float
        )

    def predict(self, X: Any, *, batch_labels: Optional[Sequence[Any]] = None, alignment: str = "strict") -> np.ndarray:
        matrix = self._prepare_input(X, alignment=alignment)
        return np.asarray(
            self.runtime_model.predict(matrix, batch_labels=batch_labels)
        )

    def predict_proba(self, X: Any, *, batch_labels: Optional[Sequence[Any]] = None, alignment: str = "strict") -> np.ndarray:
        matrix = self._prepare_input(X, alignment=alignment)
        return np.asarray(
            self.runtime_model.predict_proba(matrix, batch_labels=batch_labels),
            dtype=float,
        )


@dataclass(frozen=True)
class _NativeCategoricalStage2Context:
    """Private, pre-CV native view context for the narrow Stage-2 route."""

    preprocessor: FoldLocalPreprocessor
    bridge: NativeCategoricalStage2Bridge
    train_transformed: Any
    test_transformed: Any
    raw_train: Any
    source_schema: DatasetSchema
    adapter: NativeCategoricalStage2Adapter
    classifier_name: str
    resolved_capabilities: Mapping[str, Any]
    selector_numeric_positions: tuple[int, ...] = tuple()

    def with_selector_numeric_positions(
        self,
        positions: Sequence[int],
    ) -> "_NativeCategoricalStage2Context":
        normalized = tuple(int(value) for value in positions)
        return _NativeCategoricalStage2Context(
            preprocessor=self.preprocessor,
            bridge=self.bridge,
            train_transformed=self.train_transformed,
            test_transformed=self.test_transformed,
            raw_train=self.raw_train,
            source_schema=self.source_schema,
            adapter=self.adapter,
            classifier_name=self.classifier_name,
            resolved_capabilities=dict(self.resolved_capabilities),
            selector_numeric_positions=normalized,
        )

    def selected_views(
        self,
        selector_positions: Sequence[int],
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        if not self.selector_numeric_positions:
            raise TypedInputCapabilityError(
                "native_stage2_selector_context_unavailable",
                "Native Stage-2 routing reached candidate CV without the precomputed selector-to-numeric mapping.",
                diagnostics={
                    "numeric_schema_fingerprint": self.bridge.numeric_schema_fingerprint,
                    "native_schema_fingerprint": self.bridge.native_schema_fingerprint,
                },
            )
        normalized: list[int] = []
        for raw_position in selector_positions:
            if isinstance(raw_position, (bool, np.bool_)):
                raise TypedInputCapabilityError(
                    "native_stage2_selector_position_non_integer",
                    "Native Stage-2 selector positions must be integer indices.",
                    diagnostics={"position": repr(raw_position)},
                )
            try:
                position = int(raw_position)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypedInputCapabilityError(
                    "native_stage2_selector_position_non_integer",
                    "Native Stage-2 selector positions must be integer indices.",
                    diagnostics={"position": repr(raw_position)},
                ) from exc
            if position != raw_position:
                raise TypedInputCapabilityError(
                    "native_stage2_selector_position_non_integer",
                    "Native Stage-2 selector positions must be integer indices.",
                    diagnostics={"position": repr(raw_position)},
                )
            if not 0 <= position < len(self.selector_numeric_positions):
                raise TypedInputCapabilityError(
                    "native_stage2_selector_position_out_of_range",
                    "Native Stage-2 routing received an out-of-range selector position.",
                    diagnostics={
                        "position": int(position),
                        "selector_width": len(self.selector_numeric_positions),
                    },
                )
            normalized.append(position)
        numeric_positions = tuple(
            int(self.selector_numeric_positions[position]) for position in normalized
        )
        train_view, train_record = self.preprocessor.select_native_stage2_view(
            self.train_transformed,
            bridge=self.bridge,
            selected_numeric_positions=numeric_positions,
        )
        test_view, test_record = self.preprocessor.select_native_stage2_view(
            self.test_transformed,
            bridge=self.bridge,
            selected_numeric_positions=numeric_positions,
        )
        if dict(train_record) != dict(test_record):
            raise TypedInputCapabilityError(
                "native_stage2_train_test_view_mismatch",
                "Native Stage-2 train/test selected views disagree on immutable schema provenance.",
                diagnostics={
                    "train_record": dict(train_record),
                    "test_record": dict(test_record),
                },
            )
        if not tuple(train_record.get("selected_categorical_columns", ())):
            raise TypedInputCapabilityError(
                "native_stage2_no_selected_categorical_columns",
                "The explicit native Stage-2 route selected no categorical feature.",
                diagnostics={
                    "selected_numeric_positions": list(numeric_positions),
                    "selected_native_columns": list(
                        train_record.get("selected_native_columns", ())
                    ),
                },
            )
        return train_view, test_view, train_record

    @staticmethod
    def _take_rows(values: Any, positions: np.ndarray) -> Any:
        if hasattr(values, "iloc"):
            return values.iloc[positions].copy()
        return np.asarray(values)[positions].copy()

    def fold_view_factory(
        self,
        *,
        selected_numeric_positions: Sequence[int],
    ) -> Any:
        """Build native CV views from fold-train-only typed state.

        The outer preprocessor remains available only for final fit.  Inner CV
        must never slice its already-fitted native view because doing so leaks
        category vocabulary and imputation state from held-out fold rows.
        """

        numeric_positions = tuple(int(value) for value in selected_numeric_positions)
        if not numeric_positions:
            raise TypedInputCapabilityError(
                "native_stage2_selection_empty",
                "Native Stage-2 CV requires at least one selected numeric feature.",
            )
        if len(set(numeric_positions)) != len(numeric_positions):
            raise TypedInputCapabilityError(
                "native_stage2_selection_duplicate",
                "Native Stage-2 CV refuses duplicate selected numeric positions.",
                diagnostics={"selected_numeric_positions": list(numeric_positions)},
            )
        invalid = [
            int(position)
            for position in numeric_positions
            if not 0 <= int(position) < len(self.bridge.numeric_feature_names)
        ]
        if invalid:
            raise TypedInputCapabilityError(
                "native_stage2_selection_out_of_range",
                "Native Stage-2 CV received an out-of-range numeric selection position.",
                diagnostics={"invalid_positions": invalid},
            )
        expected_numeric_columns = tuple(
            self.bridge.numeric_feature_names[position]
            for position in numeric_positions
        )
        expected_native_columns = tuple(
            self.bridge.native_feature_names[
                self.bridge.native_position_by_numeric_position[position]
            ]
            for position in numeric_positions
        )

        def build_fold_views(
            train_positions: np.ndarray,
            validation_positions: np.ndarray,
        ) -> tuple[Any, Any, Sequence[str], Mapping[str, Any]]:
            raw_fold_train = self._take_rows(self.raw_train, train_positions)
            raw_fold_validation = self._take_rows(
                self.raw_train, validation_positions
            )
            fold_preprocessor = clone(self.preprocessor)
            fold_preprocessor.fit(raw_fold_train, schema=self.source_schema)
            fold_bridge = fold_preprocessor.native_stage2_bridge()
            if (
                str(fold_bridge.source_schema_fingerprint)
                != str(self.bridge.source_schema_fingerprint)
            ):
                raise TypedInputCapabilityError(
                    "native_stage2_fold_source_schema_mismatch",
                    "Fold-local native Stage-2 preprocessing changed the admitted source schema.",
                    diagnostics={
                        "expected_source_schema_fingerprint": self.bridge.source_schema_fingerprint,
                        "observed_source_schema_fingerprint": fold_bridge.source_schema_fingerprint,
                    },
                )
            if tuple(fold_bridge.numeric_feature_names) != tuple(
                self.bridge.numeric_feature_names
            ) or tuple(fold_bridge.native_feature_names) != tuple(
                self.bridge.native_feature_names
            ):
                raise TypedInputCapabilityError(
                    "native_stage2_fold_feature_identity_mismatch",
                    "Fold-local native Stage-2 preprocessing changed numeric/native feature identity or order.",
                    diagnostics={
                        "expected_numeric_feature_names": list(
                            self.bridge.numeric_feature_names
                        ),
                        "observed_numeric_feature_names": list(
                            fold_bridge.numeric_feature_names
                        ),
                        "expected_native_feature_names": list(
                            self.bridge.native_feature_names
                        ),
                        "observed_native_feature_names": list(
                            fold_bridge.native_feature_names
                        ),
                    },
                )
            fold_positions_by_name = {
                str(name): int(position)
                for position, name in enumerate(fold_bridge.numeric_feature_names)
            }
            if len(fold_positions_by_name) != len(fold_bridge.numeric_feature_names):
                raise TypedInputCapabilityError(
                    "native_stage2_fold_duplicate_numeric_columns",
                    "Fold-local native Stage-2 numeric schema contains duplicate feature names.",
                )
            try:
                fold_numeric_positions = tuple(
                    fold_positions_by_name[name] for name in expected_numeric_columns
                )
            except KeyError as exc:
                raise TypedInputCapabilityError(
                    "native_stage2_fold_selected_name_missing",
                    "Fold-local native Stage-2 schema cannot map a selected outer numeric feature by name.",
                    diagnostics={
                        "selected_numeric_columns": list(expected_numeric_columns),
                        "fold_numeric_feature_names": list(
                            fold_bridge.numeric_feature_names
                        ),
                    },
                ) from exc
            for name, outer_position, fold_position in zip(
                expected_numeric_columns,
                numeric_positions,
                fold_numeric_positions,
            ):
                outer_feature = self.preprocessor.numeric_schema_.features[
                    int(outer_position)
                ]
                fold_feature = fold_preprocessor.numeric_schema_.features[
                    int(fold_position)
                ]
                outer_lineage = next(
                    record
                    for record in self.preprocessor.numeric_schema_.lineage
                    if record.output_name == name
                )
                fold_lineage = next(
                    record
                    for record in fold_preprocessor.numeric_schema_.lineage
                    if record.output_name == name
                )
                if (
                    str(outer_feature.source_name) != str(fold_feature.source_name)
                    or str(outer_lineage.operation) != str(fold_lineage.operation)
                    or tuple(outer_lineage.input_names)
                    != tuple(fold_lineage.input_names)
                    or str(outer_lineage.source_schema_hash)
                    != str(fold_lineage.source_schema_hash)
                ):
                    raise TypedInputCapabilityError(
                        "native_stage2_fold_selected_lineage_mismatch",
                        "Fold-local native Stage-2 feature lineage does not match the selected outer numeric feature.",
                        diagnostics={
                            "feature": name,
                            "outer_numeric_position": int(outer_position),
                            "fold_numeric_position": int(fold_position),
                            "outer_source_name": str(outer_feature.source_name),
                            "fold_source_name": str(fold_feature.source_name),
                            "outer_operation": str(outer_lineage.operation),
                            "fold_operation": str(fold_lineage.operation),
                        },
                    )
            capability_overrides = ClassifierCapabilityOverrides(
                categorical_input=SupportLevel.SUPPORTED
            )
            train_transformed = fold_preprocessor.transform_for_classifier(
                raw_fold_train,
                classifier_name=self.classifier_name,
                dependency_facts=self.adapter.dependency_facts,
                capability_overrides=capability_overrides,
            )
            validation_transformed = fold_preprocessor.transform_for_classifier(
                raw_fold_validation,
                classifier_name=self.classifier_name,
                dependency_facts=self.adapter.dependency_facts,
                capability_overrides=capability_overrides,
            )
            train_view, train_record = fold_preprocessor.select_native_stage2_view(
                train_transformed,
                bridge=fold_bridge,
                selected_numeric_positions=fold_numeric_positions,
            )
            validation_view, validation_record = (
                fold_preprocessor.select_native_stage2_view(
                    validation_transformed,
                    bridge=fold_bridge,
                    selected_numeric_positions=fold_numeric_positions,
                )
            )
            if dict(train_record) != dict(validation_record):
                raise TypedInputCapabilityError(
                    "native_stage2_fold_train_validation_view_mismatch",
                    "Fold-local native Stage-2 train/validation views disagree on immutable provenance.",
                    diagnostics={
                        "train_record": dict(train_record),
                        "validation_record": dict(validation_record),
                    },
                )
            if tuple(train_record["selected_numeric_columns"]) != expected_numeric_columns or tuple(
                train_record["selected_native_columns"]
            ) != expected_native_columns:
                raise TypedInputCapabilityError(
                    "native_stage2_fold_selected_lineage_mismatch",
                    "Fold-local native Stage-2 selected view does not map the outer numeric selection exactly.",
                    diagnostics={
                        "expected_numeric_columns": list(expected_numeric_columns),
                        "observed_numeric_columns": list(
                            train_record["selected_numeric_columns"]
                        ),
                        "expected_native_columns": list(expected_native_columns),
                        "observed_native_columns": list(
                            train_record["selected_native_columns"]
                        ),
                    },
                )
            categorical_columns = tuple(
                str(value)
                for value in train_record.get("selected_categorical_columns", ())
            )
            if not categorical_columns:
                raise TypedInputCapabilityError(
                    "native_stage2_no_selected_categorical_columns",
                    "Fold-local native Stage-2 CV selected no categorical feature.",
                )
            return (
                train_view,
                validation_view,
                categorical_columns,
                {
                    "source_schema_fingerprint": fold_bridge.source_schema_fingerprint,
                    "numeric_schema_fingerprint": fold_bridge.numeric_schema_fingerprint,
                    "native_schema_fingerprint": fold_bridge.native_schema_fingerprint,
                    "selected_numeric_positions": list(
                        train_record["selected_numeric_positions"]
                    ),
                    "outer_selected_numeric_positions": list(numeric_positions),
                    "selected_numeric_columns": list(
                        train_record["selected_numeric_columns"]
                    ),
                    "selected_native_positions": list(
                        train_record["selected_native_positions"]
                    ),
                    "selected_native_columns": list(
                        train_record["selected_native_columns"]
                    ),
                    "selected_category_vocabularies": dict(
                        train_record["selected_category_vocabularies"]
                    ),
                    "preprocessor_fit_rows": int(fold_preprocessor.fit_row_count_),
                    "preprocessor_scope": "fold_train_only",
                },
            )

        return build_fold_views


class IncompleteFeatureSelectionError(RuntimeError):
    """Fail-closed selector outcome that must not enter Stage 2.

    A capped selector can intentionally return no features.  This error keeps
    that outcome distinct from an arbitrary empty matrix so callers can record
    a truthful skipped run rather than a downstream classifier failure.
    """

    code = "incomplete_feature_selection"

    def __init__(
        self,
        *,
        candidate_name: str,
        selection_aggregation_status: str,
        selector_candidate_statuses: Mapping[str, Any],
        selected_feature_count: int,
    ) -> None:
        statuses = copy.deepcopy(dict(selector_candidate_statuses or {}))
        mrmr_status = statuses.get("mrmr_jmi", {})
        if not isinstance(mrmr_status, Mapping):
            mrmr_status = {}
        mrmr_execution = {
            field: mrmr_status.get(field)
            for field in (
                "status",
                "complete",
                "incomplete",
                "fallback_applied",
                "budget_status",
                "budget_exhausted",
                "stop_reason",
            )
        }
        self.candidate_name = str(candidate_name)
        self.selection_aggregation_status = str(selection_aggregation_status)
        self.diagnostics = MappingProxyType(
            {
                "candidate_name": self.candidate_name,
                "selection_aggregation_status": self.selection_aggregation_status,
                "selection_aggregation_fail_closed": True,
                "selected_feature_count": int(selected_feature_count),
                "selector_candidate_statuses": statuses,
                "mrmr_jmi_execution": mrmr_execution,
            }
        )
        super().__init__(
            "Feature selection produced no usable features after a fail-closed "
            f"selector outcome for candidate {self.candidate_name!r} "
            f"({self.selection_aggregation_status})."
        )


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


class UnsafeLegacyBundleError(ValueError):
    """Raised before any legacy arbitrary-pickle payload is decoded."""


class DFFSReproducibleModel:
    """Legacy trusted-runtime inference helper backed by arbitrary pickle.

    This v1 format is retained only for explicitly trusted local artifacts.  It
    is not a safe interchange format; use the versioned safe bundle codec for
    a supported non-executable inference route.
    """

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
        variance_keep_indices: Optional[Sequence[int]] = None,
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
        self.variance_keep_indices = np.asarray(
            list(variance_keep_indices or tuple()), dtype=int
        ).ravel()
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
                    train_sorted_raw = fallback_meta.get("train_sorted")
                    train_sorted = None
                    if train_sorted_raw is not None:
                        try:
                            train_sorted = np.asarray(list(train_sorted_raw), dtype=float).ravel()
                            train_sorted = train_sorted[np.isfinite(train_sorted)]
                        except Exception:
                            train_sorted = None
                    if train_sorted is not None and train_sorted.size > 1:
                        train_sorted = np.sort(train_sorted)
                        n = float(train_sorted.size)
                        left = np.searchsorted(train_sorted, data, side="left")
                        right = np.searchsorted(train_sorted, data, side="right")
                        mid = (np.asarray(left, dtype=float) + np.asarray(right, dtype=float)) / 2.0
                        eps = float(max(1e-8, 0.5 / n))
                        u = np.clip(mid / n, eps, 1.0 - eps)
                    else:
                        ranks = np.argsort(np.argsort(data)).astype(float)
                        n = float(max(1, data.shape[0]))
                        u = np.clip((ranks + 0.5) / n, 1e-8, 1.0 - 1e-8)
                    gauss = sps.norm.ppf(u)
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

    def _apply_variance_keep(self, X: np.ndarray) -> np.ndarray:
        """Replay the train-only variance-floor routing map before selection."""

        x = np.asarray(X, dtype=float)
        keep = np.asarray(self.variance_keep_indices, dtype=int).ravel()
        if keep.size == 0:
            return x
        if np.any(keep < 0) or np.any(keep >= x.shape[1]):
            raise RuntimeError(
                "Variance-floor routing indices are incompatible with the inference matrix."
            )
        if len(set(int(value) for value in keep.tolist())) != int(keep.size):
            raise RuntimeError("Variance-floor routing indices must be unique.")
        return np.asarray(x[:, keep], dtype=float)

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
                x_fold = self._apply_variance_keep(x_fold)
                x_sel = np.asarray(self.selector.transform(x_fold), dtype=float)
                x_sel_raw = np.asarray(x_sel, dtype=float)
                x_sel_base = np.asarray(x_sel, dtype=float)
                x_sel = self._apply_distribution_transforms(x_sel_raw, x_sel_base)
            else:
                x_pref_raw = self._apply_variance_keep(x_pref_raw)
                x_pref_base = self._apply_variance_keep(x_pref_base)
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
            x_fold = self._apply_variance_keep(x_fold)
            x_sel = np.asarray(self.selector.transform(x_fold), dtype=float)
        x_stage2 = self._apply_stage2_ratio_features(x_sel)
        return np.asarray(np.nan_to_num(x_stage2, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)

    def predict(self, X: np.ndarray, *, batch_labels: Optional[Sequence[Any]] = None) -> np.ndarray:
        x_t = self.transform(X, batch_labels=batch_labels)
        return np.asarray(self.classifier_model.predict(x_t))

    def predict_proba(self, X: np.ndarray, *, batch_labels: Optional[Sequence[Any]] = None) -> np.ndarray:
        x_t = self.transform(X, batch_labels=batch_labels)
        fitted = dict(
            (self.metadata or {}).get("classification_final_fitted_descriptor")
            or {}
        )
        if fitted and (
            str(fitted.get("fitted_probability_kind", "unknown"))
            not in {"native", "calibrated", "score_derived"}
            or str(fitted.get("matrix_observation", "unobserved")) != "passed"
        ):
            raise AttributeError("Model has no admitted fitted probability matrix.")
        if not callable(getattr(self.classifier_model, "predict_proba", None)):
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
                "variance_keep_indices": [
                    int(i)
                    for i in np.asarray(
                        self.variance_keep_indices, dtype=int
                    ).ravel().tolist()
                ],
            },
            "serialization": {
                "format": "pickle+zlib+base64",
                "trust_mode": "legacy_unsafe_pickle",
                "trusted_legacy_pickle_required": True,
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
    def from_json_dict(
        cls,
        payload: Dict[str, Any],
        *,
        trusted_legacy_pickle: bool = False,
    ) -> "DFFSReproducibleModel":
        """Load v1 state only after an explicit local-trust decision."""

        # This is a security boundary before any pickle payload is decoded.
        # Do not coerce values here: strings such as "false" and integers such
        # as 1 are truthy but are not an explicit caller trust decision.
        if trusted_legacy_pickle is not True:
            raise UnsafeLegacyBundleError(
                "Legacy DFFS bundles embed arbitrary pickle and are unsafe to load by "
                "default. Re-run with trusted_legacy_pickle=True only for an artifact "
                "you explicitly trust."
            )
        ser = dict(payload.get("serialization") or {})
        meta = dict(payload.get("metadata") or {})
        stages = dict(payload.get("stages") or {})
        model = cls(
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
            variance_keep_indices=list(stages.get("variance_keep_indices") or []),
            folding_meta=dict(stages.get("folding_stage") or {}),
            folding_transformer=_decode_pickle_payload(str(ser.get("folding_transformer", ""))),
            folding_standardize_mean=_decode_pickle_payload(str(ser.get("folding_standardize_mean", ""))),
            folding_standardize_scale=_decode_pickle_payload(str(ser.get("folding_standardize_scale", ""))),
            selector=_decode_pickle_payload(str(ser.get("selector", ""))),
            stage2_ratio_meta=dict(stages.get("stage2_ratio") or {}),
            classifier_model=_decode_pickle_payload(str(ser.get("classifier_model", ""))),
            metadata=meta,
        )
        model.metadata["legacy_load_trust"] = {
            "trusted_legacy_pickle": True,
            "serialization_format": str(ser.get("format", "pickle+zlib+base64")),
        }
        return model


def load_df_fs_model_bundle(
    path: str,
    *,
    trusted_legacy_pickle: bool = False,
) -> DFFSReproducibleModel:
    import json

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return DFFSReproducibleModel.from_json_dict(
        payload,
        trusted_legacy_pickle=trusted_legacy_pickle,
    )


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

    # §3.1 family prescreen — mandatory floor of common continuous families that the
    # DIAKRINO shortlist can never prune away (so a wrong prediction degrades cost, not
    # correctness; scipy GOF still picks the final family from the kept set).
    _DIAKRINO_PRESCREEN_MANDATORY: frozenset = frozenset(
        {"norm", "t", "johnsonsu", "gamma", "lognorm", "weibull_min", "laplace", "beta", "uniform"}
    )

    @property
    def _diakrino_sidecar(self):
        """Lazily-loaded DIAKRINO sidecar for family prescreen (None when disabled/absent)."""
        cached = getattr(self, "_diakrino_sidecar_cache", "unset")
        if cached != "unset":
            return cached
        sc = None
        path = str(getattr(self.config, "diakrino_sidecar_path", "") or "")
        dataset_id = (
            str(getattr(self.config, "diakrino_sidecar_dataset_id", "") or "")
            or str(getattr(self, "_active_diakrino_dataset_id", "") or "")
            or None
        )
        if path:
            try:
                from tabnetics.feature_selection.diakrino_sidecar import DiakrinoSidecar
                sc = DiakrinoSidecar.load(path, dataset_id=dataset_id)
            except Exception:
                sc = None
        self._diakrino_sidecar_cache = sc
        return sc

    def _diakrino_family_shortlist(self, feature_index: int, support_filtered: set) -> Optional[set]:
        """Top-K predicted continuous family NAMES for ``feature_index``, intersected
        with the support-filtered set and unioned with the mandatory floor.  Returns
        ``None`` (no prescreen) when disabled, sidecar absent, index out of range, or
        the family head is too uncertain (high entropy)."""
        if not bool(getattr(self.config, "diakrino_family_prescreen_enabled", False)):
            return None
        if feature_index is None or int(feature_index) < 0:
            return None
        sc = self._diakrino_sidecar
        if sc is None or int(feature_index) >= sc.n_features:
            return None
        ent = sc.family_entropy(normalized=True)
        # high family entropy => head undecided => fall through to the full set
        if ent is not None and float(ent[int(feature_index)]) > 0.92:
            return None
        topk = sc.family_topk_continuous(int(getattr(self.config, "diakrino_family_prescreen_top_k", 4)))
        if topk is None:
            return None
        names = {sc.family_id_to_name(fid) for fid in topk[int(feature_index)]}
        names.discard(None)
        keep = (support_filtered & names)
        if bool(getattr(self.config, "diakrino_family_prescreen_keep_mandatory", True)):
            keep = keep | (support_filtered & self._DIAKRINO_PRESCREEN_MANDATORY)
        return keep or None  # never return an empty set

    def _diakrino_family_log_prior_by_name(self, feature_index: int) -> Optional[Dict[str, float]]:
        """Return continuous-family ``log P_diakrino(family)`` values keyed by scipy name."""
        if feature_index is None or int(feature_index) < 0:
            return None
        sc = self._diakrino_sidecar
        if sc is None or int(feature_index) >= sc.n_features:
            return None
        logits = sc.family_logits()
        if logits is None or int(feature_index) >= int(logits.shape[0]):
            return None

        try:
            from tabnetics.feature_selection.diakrino_sidecar import N_CONTINUOUS_FAMILIES
        except Exception:
            N_CONTINUOUS_FAMILIES = 31  # type: ignore

        row = np.asarray(logits[int(feature_index), : int(N_CONTINUOUS_FAMILIES)], dtype=float)
        finite = np.isfinite(row)
        if row.size == 0 or not np.any(finite):
            return None

        z = np.full_like(row, -np.inf, dtype=float)
        z[finite] = row[finite] - float(np.max(row[finite]))
        exp_z = np.exp(z)
        denom = float(np.sum(exp_z))
        if not np.isfinite(denom) or denom <= 0.0:
            return None

        log_probs = z - float(np.log(denom))
        by_name: Dict[str, float] = {}
        for family_id, log_prob in enumerate(log_probs.tolist()):
            if not np.isfinite(float(log_prob)):
                continue
            name = sc.family_id_to_name(int(family_id))
            if not name:
                continue
            by_name[str(name)] = float(log_prob)
        return by_name or None

    def _apply_diakrino_family_prior_selection(
        self,
        best_name: Optional[str],
        best_result: Optional[Any],
        all_results: Sequence[Any],
        *,
        criterion: str,
        feature_index: int,
    ) -> Tuple[Optional[str], Optional[Any]]:
        """Softly bias simple family selection by DIAKRINO family log-probability."""
        lam = float(getattr(self.config, "diakrino_family_prior_lambda", 0.0) or 0.0)
        if lam <= 0.0:
            return best_name, best_result
        crit = str(criterion or "simple").strip().lower()
        if crit not in {"", "simple"}:
            return best_name, best_result

        log_prior = self._diakrino_family_log_prior_by_name(int(feature_index))
        if not log_prior:
            return best_name, best_result

        ranked: List[Tuple[float, int, Any]] = []
        for idx, result in enumerate(all_results or []):
            if not getattr(result, "success", False):
                continue
            score = float(getattr(result, "simple_score", np.inf))
            name = str(getattr(result, "name", "") or "")
            log_p = log_prior.get(name)
            if not (np.isfinite(score) and log_p is not None and np.isfinite(float(log_p))):
                continue
            adjusted = score - lam * float(log_p)
            if np.isfinite(adjusted):
                ranked.append((float(adjusted), int(idx), result))

        if not ranked:
            return best_name, best_result

        ranked.sort(key=lambda item: (item[0], item[1]))
        chosen = ranked[0][2]
        return str(getattr(chosen, "name", "") or ""), chosen

    @staticmethod
    def _scipy_family_param_count(dist: Any) -> int:
        """Count scipy continuous family params as shape args + loc + scale."""
        shapes = getattr(dist, "shapes", None)
        if shapes is None:
            return 2
        shape_text = str(shapes).strip()
        if not shape_text:
            return 2
        shape_count = len([part for part in shape_text.split(",") if str(part).strip()])
        return int(shape_count + 2)

    def _family_param_count(self, family_name: str) -> int:
        dist = self._base_distributions.get(str(family_name))
        if dist is None:
            return 0
        return self._scipy_family_param_count(dist)

    def _apply_distinct_value_family_gate(
        self,
        families: Set[str],
        audit: DataAuditReport,
    ) -> Set[str]:
        if not bool(getattr(self.config, "flex_family_distinct_gate_enabled", False)):
            return set(families)

        min_distinct = int(max(1, int(getattr(self.config, "flex_family_min_distinct", 15) or 15)))
        n_unique = int(getattr(audit, "n_unique", 0) or 0)
        if n_unique >= min_distinct:
            return set(families)

        gated = {
            str(name)
            for name in set(families)
            if self._family_param_count(str(name)) < 3
        }
        if gated:
            return gated

        # Defensive fallback for unusual custom candidate sets: keep any simple
        # family from the active library rather than reintroducing gated families.
        return {
            str(name)
            for name in self._base_distributions
            if self._family_param_count(str(name)) < 3
        } or set(families)

    def _apply_flexible_family_retention_margin(
        self,
        best_name: Optional[str],
        best_result: Optional[Any],
        all_results: Sequence[Any],
        *,
        criterion: str,
    ) -> Tuple[Optional[str], Optional[Any]]:
        margin = float(max(0.0, float(getattr(self.config, "flex_family_retention_margin", 0.0) or 0.0)))
        if margin <= 0.0 or best_name is None or best_result is None:
            return best_name, best_result
        crit = str(criterion or "simple").strip().lower()
        if crit not in {"", "simple"}:
            return best_name, best_result
        if self._family_param_count(str(best_name)) < 3:
            return best_name, best_result

        best_score = float(getattr(best_result, "simple_score", np.inf))
        if not np.isfinite(best_score):
            return best_name, best_result

        simple_candidates: List[Tuple[float, int, str, Any]] = []
        for result in all_results or []:
            if not getattr(result, "success", False):
                continue
            name = str(getattr(result, "name", "") or "")
            if not name or self._family_param_count(name) >= 3:
                continue
            score = float(getattr(result, "simple_score", np.inf))
            if not np.isfinite(score):
                continue
            simple_candidates.append((float(score), self._family_param_count(name), name, result))

        if not simple_candidates:
            return best_name, best_result

        simple_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        simple_score, _, simple_name, simple_result = simple_candidates[0]
        if simple_score <= best_score + margin:
            return simple_name, simple_result
        return best_name, best_result

    def generate_candidates(self, audit: DataAuditReport, feature_index: int = -1) -> Dict[str, sps.rv_continuous]:
        families = set(self._base_distributions.keys())

        if not self.config.use_support_filtering:
            if not bool(getattr(self.config, "flex_family_distinct_gate_enabled", False)):
                return dict(self._base_distributions)
            families = self._apply_distinct_value_family_gate(families, audit)
            return {
                name: self._base_distributions[name]
                for name in self._base_distributions
                if name in families
            }

        support = audit.support.inferred_support

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

        families = self._apply_distinct_value_family_gate(families, audit)

        # §3.1 DIAKRINO family prescreen (opt-in): narrow to the DIAKRINO top-K continuous families
        # for this feature, always keeping the mandatory floor.  Replay-safe: the family
        # scipy ultimately selects persists into feature_plans as usual; the inference
        # path never consults the sidecar.
        shortlist = self._diakrino_family_shortlist(int(feature_index), set(families))
        if shortlist:
            families = shortlist

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

        candidates = self.generate_candidates(audit, feature_index=int(feature_index))
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
        best_name, best_result = self._apply_diakrino_family_prior_selection(
            best_name,
            best_result,
            all_results,
            criterion=criterion,
            feature_index=int(feature_index),
        )
        best_name, best_result = self._apply_flexible_family_retention_margin(
            best_name,
            best_result,
            all_results,
            criterion=criterion,
        )

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


def _scope_pipeline_sklearn_n_jobs(method):
    """Bind the configured worker cap for one pipeline call only."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        n_jobs = resolve_sklearn_n_jobs(getattr(self.config, "n_jobs", 1))
        with sklearn_n_jobs_scope(n_jobs):
            return method(self, *args, **kwargs)

    return wrapped


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
        self._last_diakrino_prefilter_state: Dict[str, Any] = {}
        self._diakrino_prefilter_identity_available = True

    @property
    def _diakrino_transform_sidecar(self):
        """Lazily loaded DIAKRINO sidecar for distribution-stage route/weight gates."""
        dist_path = str(getattr(self.config.dist_config, "diakrino_sidecar_path", "") or "")
        pipe_path = str(getattr(self.config, "diakrino_sidecar_path", "") or "")
        dist_dataset_id = str(getattr(self.config.dist_config, "diakrino_sidecar_dataset_id", "") or "")
        pipe_dataset_id = str(getattr(self.config, "diakrino_sidecar_dataset_id", "") or "")
        active_dataset_id = str(getattr(self, "_active_diakrino_dataset_id", "") or "")
        dataset_id = dist_dataset_id or pipe_dataset_id or active_dataset_id
        path = dist_path or pipe_path
        cached = getattr(self, "_diakrino_transform_sidecar_cache", None)
        if isinstance(cached, tuple) and cached[0] == path and cached[1] == dataset_id:
            return cached[2]
        sc = None
        if path:
            try:
                from tabnetics.feature_selection.diakrino_sidecar import DiakrinoSidecar
                sc = DiakrinoSidecar.load(path, dataset_id=dataset_id or None)
            except Exception:
                sc = None
        self._diakrino_transform_sidecar_cache = (path, dataset_id, sc)
        return sc

    def _diakrino_sidecar_resolution_diagnostics(self, n_features: int) -> Dict[str, Any]:
        active_dataset_id = str(getattr(self, "_active_diakrino_dataset_id", "") or "")

        def inspect(label: str, path: str, dataset_id: str) -> Dict[str, Any]:
            row: Dict[str, Any] = {
                "consumer": str(label),
                "configured_path": str(path or ""),
                "requested_dataset_id": str(dataset_id or ""),
                "active_dataset_id": active_dataset_id,
                "loaded": False,
                "n_features_expected": int(n_features),
                "n_features_match": False,
            }
            if not path:
                row["status"] = "disabled_empty_path"
                return row
            try:
                from tabnetics.feature_selection.diakrino_sidecar import DiakrinoSidecar
                sc = DiakrinoSidecar.load(path, dataset_id=str(dataset_id or "") or None)
            except Exception as exc:
                row["status"] = "load_error"
                row["error_type"] = type(exc).__name__
                row["error"] = str(exc)
                return row
            if sc is None:
                row["status"] = "missing_or_unreadable"
                return row
            try:
                row.update(sc.resolution_diagnostics())
            except Exception:
                row["loaded"] = True
                row["n_features"] = int(getattr(sc, "n_features", 0) or 0)
            row["status"] = "loaded"
            row["n_features_match"] = bool(int(getattr(sc, "n_features", 0) or 0) == int(n_features))
            return row

        pipe_path = str(getattr(self.config, "diakrino_sidecar_path", "") or "")
        dist_path = str(getattr(self.config.dist_config, "diakrino_sidecar_path", "") or "")
        pipe_dataset_id = (
            str(getattr(self.config, "diakrino_sidecar_dataset_id", "") or "")
            or active_dataset_id
        )
        dist_dataset_id = (
            str(getattr(self.config.dist_config, "diakrino_sidecar_dataset_id", "") or "")
            or pipe_dataset_id
        )
        return {
            "schema_version": "1.0",
            "active_dataset_id": active_dataset_id,
            "pipeline": inspect("pipeline", pipe_path, pipe_dataset_id),
            "distribution": inspect("distribution", dist_path or pipe_path, dist_dataset_id),
        }

    def _diakrino_router_dispersion_descriptor(self) -> Dict[str, Any]:
        if not bool(getattr(self.config, "diakrino_router_dispersion_descriptor_enabled", False)):
            return {"enabled": False}
        path = str(getattr(self.config, "diakrino_sidecar_path", "") or "")
        if not path:
            return {"enabled": True, "loaded": False, "reason": "empty_sidecar_path"}
        try:
            from tabnetics.feature_selection.diakrino_sidecar import DiakrinoSidecar

            sc = DiakrinoSidecar.load(
                path,
                dataset_id=str(getattr(self.config, "diakrino_sidecar_dataset_id", "") or "") or None,
            )
        except Exception as exc:
            return {
                "enabled": True,
                "loaded": False,
                "reason": "load_error",
                "error_type": str(type(exc).__name__),
                "error": str(exc),
            }
        if sc is None:
            return {"enabled": True, "loaded": False, "reason": "missing_or_unreadable"}
        summary = sc.selection_dispersion_summary(column="feature_selection_logit", calibrate="chunk_zscore")
        if summary is None:
            summary = sc.selection_dispersion_summary(column="prior_logit", calibrate="chunk_zscore")
        return {
            "enabled": True,
            "loaded": summary is not None,
            "promotion_gate": "beats per-chunk z-score by >= +0.01 in paired portfolio validation",
            "summary": dict(summary or {}),
        }

    def _diakrino_family_agreement_audit(self, feature_plans: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
        enabled = bool(getattr(self.config.dist_config, "diakrino_family_prescreen_enabled", False)) or bool(
            getattr(self.config.dist_config, "diakrino_skip_fit_discrete_enabled", False)
        )
        if not enabled:
            return {"enabled": False}
        sc = self._diakrino_transform_sidecar
        if sc is None:
            return {"enabled": True, "loaded": False, "reason": "missing_or_unreadable"}
        ids = sc.family_argmax_ids()
        if ids is None:
            return {"enabled": True, "loaded": False, "reason": "missing_family_logits"}
        ids_arr = np.asarray(ids, dtype=np.int64).ravel()
        compared = 0
        agreed = 0
        discrete = 0
        rows: List[Dict[str, Any]] = []
        for feat_idx, plan in sorted(dict(feature_plans).items()):
            i = int(feat_idx)
            if i < 0 or i >= ids_arr.size:
                continue
            family_id = int(ids_arr[i])
            if family_id >= 31:
                discrete += 1
                rows.append(
                    {
                        "feature_index": i,
                        "diakrino_family_id": int(family_id),
                        "diakrino_family_name": None,
                        "fitted_family": None if plan.get("family") is None else str(plan.get("family")),
                        "agreement": False,
                        "skip_reason": "diakrino_discrete_or_nuisance_family",
                    }
                )
                continue
            diakrino_name = sc.family_id_to_name(family_id)
            fitted = None if plan.get("family") is None else str(plan.get("family"))
            if not diakrino_name or not fitted or fitted.startswith("multimodal_fallback"):
                continue
            same = str(diakrino_name) == str(fitted)
            compared += 1
            agreed += int(bool(same))
            rows.append(
                {
                    "feature_index": i,
                    "diakrino_family_id": int(family_id),
                    "diakrino_family_name": str(diakrino_name),
                    "fitted_family": str(fitted),
                    "agreement": bool(same),
                }
            )
        return {
            "enabled": True,
            "loaded": True,
            "n_features_audited": int(len(feature_plans)),
            "n_compared": int(compared),
            "n_agree": int(agreed),
            "agreement_rate": float(agreed / compared) if compared > 0 else float("nan"),
            "n_diakrino_discrete_or_nuisance": int(discrete),
            "examples": rows[:20],
        }

    @staticmethod
    def _take_input_rows(X: Any, indices: Sequence[int]) -> Any:
        """Slice rows without converting DataFrames into column selection."""

        positions = np.asarray(indices, dtype=int).ravel()
        if hasattr(X, "iloc"):
            return X.iloc[positions].copy()
        if is_sparse_input(X):
            return X[positions]
        return np.asarray(X)[positions]

    def _typed_frontend_requested(
        self,
        X: Any,
        *,
        schema: DatasetSchema | None,
    ) -> bool:
        """An explicit schema is itself an opt-in request for the new boundary."""

        return bool(
            schema is not None
            or bool(getattr(self.config, "typed_input_enabled", False))
            and is_typed_input(X)
        )

    def _prepare_typed_train_test_inputs(
        self,
        *,
        X_train: Any,
        X_test: Any,
        schema: DatasetSchema | None,
    ) -> tuple[np.ndarray, np.ndarray, FoldLocalPreprocessor | None, Dict[str, Any]]:
        """Fit typed state on train and adapt the legacy numeric core safely."""

        typed_requested = self._typed_frontend_requested(X_train, schema=schema)
        if not typed_requested:
            return (
                np.asarray(X_train, dtype=float),
                np.asarray(X_test, dtype=float),
                None,
                {},
            )

        source_schema = infer_dataset_schema(X_train, schema=schema)
        source_schema.validate_input(X_test)
        preprocessor = FoldLocalPreprocessor(
            text_encoding=str(
                getattr(self.config, "typed_text_encoding", "tfidf_hash")
                or "tfidf_hash"
            ),
            text_hash_buckets=int(
                getattr(self.config, "typed_text_hash_buckets", 16) or 16
            ),
        )
        train_typed = preprocessor.fit_transform_with_schema(
            X_train,
            schema=source_schema,
            output_mode="numeric",
        )
        test_typed = preprocessor.transform_with_schema(
            X_test,
            schema=source_schema,
            output_mode="numeric",
        )
        raw_is_sparse = bool(is_sparse_input(X_train))
        adapter = "numeric"
        if raw_is_sparse:
            adapter = "bounded_dense"
            train_arr = guarded_sparse_to_dense(
                train_typed.X,
                max_elements=int(
                    getattr(self.config, "typed_sparse_dense_max_elements", 0)
                    or 0
                ),
                callsite="pipeline_train",
            ).astype(float, copy=False)
            test_arr = guarded_sparse_to_dense(
                test_typed.X,
                max_elements=int(
                    getattr(self.config, "typed_sparse_dense_max_elements", 0)
                    or 0
                ),
                callsite="pipeline_test",
            ).astype(float, copy=False)
        else:
            train_arr = np.asarray(train_typed.X, dtype=float)
            test_arr = np.asarray(test_typed.X, dtype=float)

        context: Dict[str, Any] = {
            "enabled": True,
            "adapter": adapter,
            "source_schema": source_schema.to_record(),
            "model_input_schema": train_typed.schema.to_record(),
            "source_schema_fingerprint": source_schema.fingerprint,
            "model_input_schema_fingerprint": train_typed.schema.fingerprint,
            "preprocessor": {
                "class": "FoldLocalPreprocessor",
                "cloneable": True,
                "fit_rows": int(preprocessor.fit_row_count_),
                "input_has_missing_train": bool(preprocessor.input_has_missing_),
                "text_encoding": str(preprocessor.text_encoding),
                "text_hash_buckets": int(preprocessor.text_hash_buckets),
                "native_categorical_route": {
                    "available": False,
                    "status": "disabled_by_config",
                    "reason": "native_categorical_stage2_disabled",
                    "pipeline_route": "numeric_adapter_for_feature_selection",
                },
                "sparse": {
                    "input_is_sparse": raw_is_sparse,
                    "preserved_by_preprocessor": raw_is_sparse,
                    "numeric_core_bridge": adapter,
                    "max_dense_elements": int(
                        getattr(self.config, "typed_sparse_dense_max_elements", 0)
                        or 0
                    ),
                },
            },
        }
        return train_arr, test_arr, preprocessor, context

    def _prepare_native_categorical_stage2_context(
        self,
        *,
        typed_requested: bool,
        preprocessor: FoldLocalPreprocessor | None,
        X_train: Any,
        X_test: Any,
        source_schema: DatasetSchema | None,
        fit_resampling_context: FitResamplingContext,
        sample_weight_requested: bool,
    ) -> _NativeCategoricalStage2Context | None:
        """Admit the explicit singleton native route before any candidate CV.

        This is intentionally a narrow capability boundary.  It fails before
        numeric fallback whenever the request would need a mixed candidate,
        replay, transformation, resampling, calibration, or weighting contract
        that this v1 adapter does not own.
        """

        cls_cfg = self._classification_cfg()
        if not bool(getattr(cls_cfg, "native_categorical_stage2_enabled", False)):
            return None
        requested_name = str(
            getattr(cls_cfg, "native_categorical_stage2_estimator", "") or ""
        ).strip()
        if not requested_name:
            raise TypedInputCapabilityError(
                "native_stage2_estimator_required",
                "Native categorical Stage-2 routing requires an explicit CatBoost or LightGBM estimator.",
            )
        if not typed_requested or preprocessor is None or source_schema is None:
            raise TypedInputCapabilityError(
                "native_stage2_typed_input_required",
                "Native categorical Stage-2 routing requires the opt-in typed DataFrame boundary.",
                diagnostics={"typed_requested": bool(typed_requested)},
            )
        if bool(is_sparse_input(X_train)) or bool(is_sparse_input(X_test)):
            raise TypedInputCapabilityError(
                "native_stage2_sparse_unavailable",
                "Native categorical Stage-2 routing does not support sparse input.",
            )
        if not any(
            feature.role is FeatureRole.CATEGORICAL
            for feature in source_schema.features
        ):
            raise TypedInputCapabilityError(
                "native_stage2_categorical_input_required",
                "Native categorical Stage-2 routing requires at least one categorical source feature.",
                diagnostics={"source_schema_fingerprint": source_schema.fingerprint},
            )

        unsupported: list[str] = []
        if str(getattr(cls_cfg, "backend", "sklearn") or "sklearn").strip().lower() != "sklearn":
            unsupported.append("classification_backend_not_sklearn")
        if str(getattr(cls_cfg, "selection_mode", "legacy") or "legacy").strip().lower() != "legacy":
            unsupported.append("classification_selection_mode_not_legacy")
        if bool(getattr(self.config, "enable_maqc_pairing", False)):
            unsupported.append("maqc_pairing")
        if bool(sample_weight_requested):
            unsupported.append("sample_weights")
        if str(fit_resampling_context.policy.kind) != "iid":
            unsupported.append("structured_resampling")
        if bool(getattr(self.config, "enable_ratio_features", False)):
            unsupported.append("stage1_ratio_features")
        if bool(getattr(cls_cfg, "stage2_ratio_augmentation_enabled", False)):
            unsupported.append("stage2_ratio_features")
        if bool(getattr(self.config, "apply_cdf_transform", False)):
            unsupported.append("distribution_transform")
        if str(getattr(self.config, "df_stage_position", "after_fs") or "after_fs").strip().lower() != "after_fs":
            unsupported.append("distribution_stage_position")
        if str(getattr(self.config, "folding_method", "none") or "none").strip().lower() != "none":
            unsupported.append("folding")
        if str(getattr(self.config, "batch_correction", "none") or "none").strip().lower() != "none":
            unsupported.append("batch_correction")
        if str(getattr(self.config, "multiomics_adapter", "none") or "none").strip().lower() != "none":
            unsupported.append("multiomics_adapter")
        if bool(getattr(self.config, "enable_face_domain_projection", False)):
            unsupported.append("face_projection")
        if bool(getattr(self.config, "calibration_reporting_enabled", False)):
            unsupported.append("calibration_reporting")
        if bool(getattr(cls_cfg, "posthoc_calibration_enabled", False)):
            unsupported.append("posthoc_calibration")
        if bool(getattr(cls_cfg, "conformal_enabled", False)):
            unsupported.append("conformal")
        if bool(getattr(cls_cfg, "use_hybrid_score", False)):
            unsupported.append("hybrid_classifier_score")
        if bool(getattr(self.config, "tier_lockout_enabled", False)):
            unsupported.append("tier_lockout")
        if bool(getattr(self.config, "tier_routing_enabled", False)):
            unsupported.append("tier_routing")
        if bool(getattr(self.config, "regime_gating_enabled", False)):
            unsupported.append("regime_gating")
        if str(getattr(self.config, "meta_learning_selector_mode", "none") or "none").strip().lower() != "none":
            unsupported.append("meta_learning_selector")
        if bool(getattr(self.config, "auto_router_enabled", False)):
            unsupported.append("auto_router")
        if bool(getattr(self.config, "diakrino_prefilter_enabled", False)):
            unsupported.append("diakrino_prefilter")
        if unsupported:
            raise TypedInputCapabilityError(
                "native_stage2_unsupported_composition",
                "Native categorical Stage-2 routing rejects unsupported pipeline composition before fitting.",
                diagnostics={
                    "unsupported": sorted(set(unsupported)),
                    "requested_estimator": requested_name,
                },
            )

        try:
            requested_spec = DEFAULT_CLASSIFIER_REGISTRY.get(requested_name)
        except Exception as exc:
            raise TypedInputCapabilityError(
                "native_stage2_classifier_unknown",
                "Native categorical Stage-2 routing requires a registered CatBoost or LightGBM classifier.",
                diagnostics={"requested_estimator": requested_name},
            ) from exc
        canonical_name = str(requested_spec.name)
        candidate_names = tuple(self._resolved_model_candidates())
        canonical_candidates: list[str] = []
        for candidate_name in candidate_names:
            try:
                canonical_candidates.append(
                    str(DEFAULT_CLASSIFIER_REGISTRY.get(str(candidate_name)).name)
                )
            except Exception as exc:
                raise TypedInputCapabilityError(
                    "native_stage2_candidate_unknown",
                    "Native categorical Stage-2 routing refuses an unknown configured classifier candidate.",
                    diagnostics={"candidate_name": str(candidate_name)},
                ) from exc
        if len(candidate_names) != 1 or canonical_candidates != [canonical_name]:
            raise TypedInputCapabilityError(
                "native_stage2_mixed_candidates",
                "Native categorical Stage-2 routing requires one configured candidate matching the explicit estimator.",
                diagnostics={
                    "requested_estimator": requested_name,
                    "requested_canonical_name": canonical_name,
                    "configured_candidates": list(candidate_names),
                    "configured_canonical_candidates": canonical_candidates,
                },
            )
        try:
            adapter = resolve_native_categorical_stage2_adapter(canonical_name)
        except NativeCategoricalStage2Error as exc:
            raise TypedInputCapabilityError(
                exc.code,
                str(exc),
                diagnostics=dict(exc.diagnostics),
            ) from exc
        capability_overrides = ClassifierCapabilityOverrides(
            categorical_input=SupportLevel.SUPPORTED
        )
        resolved = resolve_classifier_capabilities(
            canonical_name,
            runtime=ClassifierRuntimeFacts(input_has_categorical=True),
            dependency_facts=adapter.dependency_facts,
            overrides=capability_overrides,
        )
        if not resolved.is_available:
            raise TypedInputCapabilityError(
                "native_stage2_classifier_unavailable",
                "The explicit native categorical Stage-2 classifier is not concretely admitted.",
                diagnostics={
                    "canonical_name": canonical_name,
                    "availability": resolved.availability.value,
                    "availability_reasons": list(resolved.availability_reasons),
                },
            )
        try:
            train_transformed = preprocessor.transform_for_classifier(
                X_train,
                classifier_name=canonical_name,
                dependency_facts=adapter.dependency_facts,
                capability_overrides=capability_overrides,
            )
            test_transformed = preprocessor.transform_for_classifier(
                X_test,
                classifier_name=canonical_name,
                dependency_facts=adapter.dependency_facts,
                capability_overrides=capability_overrides,
            )
            bridge = preprocessor.native_stage2_bridge()
        except TypedInputCapabilityError:
            raise
        except Exception as exc:
            raise TypedInputCapabilityError(
                "native_stage2_adapter_preparation_failed",
                "Native categorical Stage-2 routing could not prepare its validated DataFrame context.",
                diagnostics={"error_type": type(exc).__name__},
            ) from exc
        if train_transformed.output_mode != "native_categorical" or test_transformed.output_mode != "native_categorical":
            raise TypedInputCapabilityError(
                "native_stage2_adapter_view_unavailable",
                "The admitted native categorical classifier did not receive a native categorical view.",
                diagnostics={
                    "train_output_mode": train_transformed.output_mode,
                    "test_output_mode": test_transformed.output_mode,
                },
            )
        resolved_record = {
            "canonical_name": resolved.canonical_name,
            "availability": resolved.availability.value,
            "availability_reasons": list(resolved.availability_reasons),
            "dependency_status": resolved.dependency_status.value,
            "categorical_input": resolved.categorical_input.value,
            "adapter_identity": adapter.adapter_identity,
        }
        return _NativeCategoricalStage2Context(
            preprocessor=preprocessor,
            bridge=bridge,
            train_transformed=train_transformed,
            test_transformed=test_transformed,
            raw_train=X_train,
            source_schema=source_schema,
            adapter=adapter,
            classifier_name=canonical_name,
            resolved_capabilities=resolved_record,
        )

    def _validate_train_only_diakrino_eligibility(
        self,
        *,
        source_schema: DatasetSchema | None = None,
        model_input_schema: DatasetSchema | None = None,
    ) -> None:
        """Admit only replayable protected-union prefilter compositions."""

        if not bool(getattr(self.config, "diakrino_prefilter_enabled", False)):
            return
        mode = str(
            getattr(self.config, "diakrino_prefilter_mode", "protected_union")
            or "protected_union"
        ).strip().lower()
        if mode != "protected_union":
            raise TypedInputCapabilityError(
                "train_only_diakrino_legacy_prefilter_unsupported",
                "Train-only fitted components support only the replayable "
                "DIAKRINO protected_union prefilter mode.",
                diagnostics={"mode": mode, "supported_mode": "protected_union"},
            )

        identity_breaks: list[str] = []
        if bool(getattr(self.config, "enable_face_domain_projection", False)):
            identity_breaks.append("face_projection")
        if bool(getattr(self.config, "enable_ratio_features", False)):
            identity_breaks.append("ratio_generation")
        if self._df_stage_position() != "after_fs" and str(
            getattr(self.config, "folding_method", "none") or "none"
        ).strip().lower() != "none":
            identity_breaks.append("pre_fs_folding")
        if source_schema is not None and model_input_schema is not None:
            if (
                int(source_schema.n_features) != int(model_input_schema.n_features)
                or tuple(source_schema.feature_names)
                != tuple(model_input_schema.feature_names)
            ):
                identity_breaks.append("typed_feature_identity_changed")
        if identity_breaks:
            raise TypedInputCapabilityError(
                "train_only_diakrino_original_identity_required",
                "Train-only DIAKRINO protected_union replay requires an "
                "identity-preserving original feature space before prefiltering.",
                diagnostics={
                    "mode": mode,
                    "identity_breaks": list(dict.fromkeys(identity_breaks)),
                },
            )

    def _validate_train_only_diakrino_fitted_state(
        self,
        *,
        runtime_model: DFFSReproducibleModel,
        fit_provenance: Mapping[str, Any],
        source_schema: DatasetSchema | None,
        model_input_schema: DatasetSchema | None,
        selected_model_input_indices: Sequence[int],
        selected_feature_schema: Mapping[str, Any],
    ) -> None:
        """Fail closed unless the fitted protected-union route is self-consistent."""

        if not bool(getattr(self.config, "diakrino_prefilter_enabled", False)):
            return

        def indices(
            value: Any,
            *,
            field: str,
            width: int,
            allow_empty: bool = True,
        ) -> np.ndarray:
            raw = [] if value is None else list(value)
            if any(
                isinstance(item, (bool, np.bool_))
                or not isinstance(item, (int, np.integer))
                for item in raw
            ):
                raise TypedInputCapabilityError(
                    "train_only_diakrino_prefilter_indices_invalid",
                    "Fitted DIAKRINO replay indices must be exact integers.",
                    diagnostics={"field": field},
                )
            result = np.asarray(raw, dtype=int).ravel()
            if (
                (not allow_empty and result.size == 0)
                or len(set(int(item) for item in result.tolist())) != int(result.size)
                or np.any(result < 0)
                or np.any(result >= int(width))
            ):
                raise TypedInputCapabilityError(
                    "train_only_diakrino_prefilter_indices_invalid",
                    "Fitted DIAKRINO replay indices are empty, duplicated, or out of range.",
                    diagnostics={
                        "field": field,
                        "width": int(width),
                        "indices": [int(item) for item in result.tolist()],
                    },
                )
            return result

        if fit_provenance.get("schema_version") != "tabnetics_train_only_components_v1":
            raise TypedInputCapabilityError(
                "train_only_diakrino_provenance_invalid",
                "Fitted DIAKRINO state requires the train-only provenance schema.",
                diagnostics={"field": "fit_provenance.schema_version"},
            )
        state = fit_provenance.get("diakrino_prefilter")
        if not isinstance(state, Mapping):
            raise TypedInputCapabilityError(
                "train_only_diakrino_provenance_invalid",
                "Fitted DIAKRINO prefilter provenance is missing or malformed.",
                diagnostics={"field": "diakrino_prefilter"},
            )
        if (
            state.get("schema_version") != "1.0"
            or state.get("configured") is not True
            or str(state.get("mode") or "") != "protected_union"
            or state.get("protection_active") is not True
            or state.get("original_identity_available") is not True
            or str(state.get("original_identity_reason") or "")
            != "original_feature_indices"
        ):
            raise TypedInputCapabilityError(
                "train_only_diakrino_provenance_invalid",
                "Fitted DIAKRINO provenance does not prove protected-union original identity.",
                diagnostics={
                    "mode": state.get("mode"),
                    "configured": state.get("configured"),
                    "protection_active": state.get("protection_active"),
                    "original_identity_available": state.get(
                        "original_identity_available"
                    ),
                    "original_identity_reason": state.get(
                        "original_identity_reason"
                    ),
                },
            )

        width = int(runtime_model.n_input_features)
        runtime_prefilter = indices(
            runtime_model.prefilter_indices,
            field="runtime.prefilter_indices",
            width=width,
            allow_empty=False,
        )
        initial = indices(
            state.get("initial_original_indices"),
            field="provenance.initial_original_indices",
            width=width,
            allow_empty=False,
        )
        if not np.array_equal(runtime_prefilter, initial):
            raise TypedInputCapabilityError(
                "train_only_diakrino_prefilter_indices_invalid",
                "Runtime prefilter indices disagree with fitted DIAKRINO provenance.",
                diagnostics={
                    "runtime": runtime_prefilter.tolist(),
                    "provenance": initial.tolist(),
                },
            )
        variance = indices(
            runtime_model.variance_keep_indices,
            field="runtime.variance_keep_indices",
            width=int(initial.size),
            allow_empty=False,
        )
        active = indices(
            state.get("active_original_indices"),
            field="provenance.active_original_indices",
            width=width,
            allow_empty=False,
        )
        expected_active = initial[variance]
        if not np.array_equal(active, expected_active):
            raise TypedInputCapabilityError(
                "train_only_diakrino_prefilter_indices_invalid",
                "Active DIAKRINO identities disagree with runtime variance replay.",
                diagnostics={
                    "active": active.tolist(),
                    "expected_active": expected_active.tolist(),
                },
            )

        classical = indices(
            state.get("classical_pool_original_indices"),
            field="provenance.classical_pool_original_indices",
            width=width,
            allow_empty=False,
        )
        extras = indices(
            state.get("diakrino_extra_original_indices"),
            field="provenance.diakrino_extra_original_indices",
            width=width,
        )
        ranked = indices(
            state.get("diakrino_ranked_candidate_original_indices"),
            field="provenance.diakrino_ranked_candidate_original_indices",
            width=width,
        )
        initial_set = set(int(item) for item in initial.tolist())
        if (
            not set(int(item) for item in classical.tolist()).issubset(initial_set)
            or not set(int(item) for item in extras.tolist()).issubset(initial_set)
            or not set(int(item) for item in ranked.tolist()).issubset(initial_set)
            or initial_set
            != set(int(item) for item in classical.tolist())
            | set(int(item) for item in extras.tolist())
        ):
            raise TypedInputCapabilityError(
                "train_only_diakrino_prefilter_indices_invalid",
                "Fitted DIAKRINO source sets disagree with the runtime prefilter union.",
                diagnostics={"field": "prefilter_source_sets"},
            )
        if state.get("applied") is False and (extras.size or ranked.size):
            raise TypedInputCapabilityError(
                "train_only_diakrino_provenance_invalid",
                "A deterministic DIAKRINO abstention cannot retain fitted additions.",
                diagnostics={"reason": state.get("reason")},
            )

        selector_getter = getattr(
            runtime_model.selector, "get_selected_features_indices", None
        )
        if not callable(selector_getter):
            raise TypedInputCapabilityError(
                "train_only_diakrino_selected_mapping_invalid",
                "The fitted selector does not expose replayable selected indices.",
            )
        selector_indices = indices(
            selector_getter(),
            field="runtime.selector_indices",
            width=int(active.size),
            allow_empty=False,
        )
        expected_selected = active[selector_indices]
        declared_selected = indices(
            selected_model_input_indices,
            field="components.selected_model_input_indices",
            width=width,
            allow_empty=False,
        )
        if not np.array_equal(declared_selected, expected_selected):
            raise TypedInputCapabilityError(
                "train_only_diakrino_selected_mapping_invalid",
                "Selected source identities disagree with the replayable selector route.",
                diagnostics={
                    "declared": declared_selected.tolist(),
                    "expected": expected_selected.tolist(),
                },
            )
        if (
            source_schema is None
            or model_input_schema is None
            or int(source_schema.n_features) != width
            or tuple(source_schema.feature_names)
            != tuple(model_input_schema.feature_names)
        ):
            raise TypedInputCapabilityError(
                "train_only_diakrino_selected_mapping_invalid",
                "Selected DIAKRINO identities require matching immutable source schemas.",
            )
        try:
            selected_schema = DatasetSchema.from_record(selected_feature_schema)
        except (SchemaContractError, TypeError, ValueError) as exc:
            raise TypedInputCapabilityError(
                "train_only_diakrino_selected_mapping_invalid",
                "Selected DIAKRINO feature schema is missing or malformed.",
            ) from exc
        expected_names = tuple(
            source_schema.feature_names[int(item)] for item in expected_selected.tolist()
        )
        if tuple(selected_schema.feature_names) != expected_names:
            raise TypedInputCapabilityError(
                "train_only_diakrino_selected_mapping_invalid",
                "Selected DIAKRINO feature names disagree with source identity mapping.",
                diagnostics={
                    "selected_names": list(selected_schema.feature_names),
                    "expected_names": list(expected_names),
                },
            )
        augmentation = fit_provenance.get("diakrino_protected_augmentation")
        if isinstance(augmentation, Mapping) and augmentation:
            final_original = indices(
                augmentation.get("final_original_indices"),
                field="provenance.diakrino_protected_augmentation.final_original_indices",
                width=width,
                allow_empty=False,
            )
            if not np.array_equal(final_original, expected_selected):
                raise TypedInputCapabilityError(
                    "train_only_diakrino_selected_mapping_invalid",
                    "Protected-union selection provenance disagrees with runtime replay.",
                    diagnostics={"field": "final_original_indices"},
                )

    def _fit_components(
        self,
        X: Any,
        y: np.ndarray,
        dataset_name: str = "dataset",
        seed: Optional[int] = None,
        batch_labels: Optional[Sequence[Any]] = None,
        *,
        sample_weight: Optional[Sequence[float]] = None,
        schema: DatasetSchema | None = None,
        resampling_context: Optional[FitResamplingContext] = None,
    ) -> FittedPipelineComponents:
        """Fit reusable state on all supplied rows without evaluation artifacts.

        ``X_apply`` below is an inference transform view of the *same* training
        matrix.  It is required by legacy stage helpers that jointly fit and
        transform, but it has no labels, no split identity, and is never used
        for metrics or policy selection.  Selector and classifier policy are
        chosen exclusively by their configured training-only CV routines.
        """

        seed = int(self.config.random_seed if seed is None else seed)
        y_arr = np.asarray(y).ravel()
        if y_arr.ndim != 1 or y_arr.size < 2:
            raise ValueError("fit_components requires a non-empty one-dimensional label vector.")

        cls_cfg = self._classification_cfg()
        self._validate_train_only_diakrino_eligibility()
        unsupported: list[str] = []
        if bool(getattr(cls_cfg, "native_categorical_stage2_enabled", False)):
            unsupported.append("native_categorical_stage2")
        if bool(getattr(cls_cfg, "posthoc_calibration_enabled", False)):
            unsupported.append("posthoc_calibration")
        if str(getattr(self.config, "multiomics_adapter", "none") or "none").strip().lower() != "none":
            unsupported.append("multiomics_adapter")
        if bool(getattr(self.config, "auto_router_enabled", False)):
            unsupported.append("auto_router")
        if str(getattr(self.config, "meta_learning_selector_mode", "none") or "none").strip().lower() != "none":
            unsupported.append("meta_learning_selector")
        if unsupported:
            raise TypedInputCapabilityError(
                "train_only_components_unsupported_composition",
                "The train-only estimator refuses options whose fitted inference "
                "state is not yet replayable without an evaluation split or external artifact.",
                diagnostics={"unsupported": sorted(unsupported)},
            )

        typed_requested = self._typed_frontend_requested(X, schema=schema)
        # This is a transform target only, not a hidden holdout.  The helper
        # fits its preprocessor exactly once on X and returns a same-row apply
        # view used by downstream transform-oriented helpers.
        X_train_arr, X_apply_arr, typed_preprocessor, typed_preprocessing = (
            self._prepare_typed_train_test_inputs(
                X_train=X,
                X_test=X,
                schema=schema,
            )
        )
        if X_train_arr.ndim != 2 or X_apply_arr.ndim != 2:
            raise ValueError("fit_components requires a two-dimensional feature matrix.")
        if int(X_train_arr.shape[0]) != int(y_arr.size):
            raise ValueError(
                f"Row mismatch: X has {X_train_arr.shape[0]} rows but y has {y_arr.size}."
            )
        if int(X_train_arr.shape[1]) != int(X_apply_arr.shape[1]):
            raise ValueError("Training and inference-transform views have incompatible feature widths.")

        source_schema = (
            typed_preprocessor.input_schema_
            if typed_preprocessor is not None
            else DatasetSchema.from_input(X)
        )
        # The numeric core only changes values (imputation/scaling), not the
        # feature contract.  Retaining this schema binds ordinary DataFrame
        # names and dtypes at inference and in the safe-bundle manifest.
        model_input_schema = (
            typed_preprocessor.get_output_schema(output_mode="numeric")
            if typed_preprocessor is not None
            else source_schema
        )
        if typed_requested and typed_preprocessor is None:
            raise RuntimeError("Typed fit requested but no fitted preprocessor was created.")
        self._validate_train_only_diakrino_eligibility(
            source_schema=source_schema,
            model_input_schema=model_input_schema,
        )

        supplied_weights = (
            tuple()
            if sample_weight is None
            else coerce_sample_weights(
                sample_weight,
                n_rows=int(y_arr.size),
                field_name="sample_weight",
                require_positive_mass=True,
            )
        )
        if resampling_context is None:
            fit_context = FitResamplingContext.iid(
                int(y_arr.size),
                sample_weights=supplied_weights or None,
            )
        else:
            fit_context = ensure_fit_resampling_context(
                resampling_context,
                n_rows=int(y_arr.size),
            )
            if supplied_weights and tuple(fit_context.sample_weights) != tuple(supplied_weights):
                raise ResamplingContractError(
                    "sample_weight does not match resampling_context.sample_weights.",
                    code="sample_weight_context_mismatch",
                    diagnostics={"n_rows": int(y_arr.size)},
                )

        if batch_labels is None:
            batch_arr = (
                np.asarray(fit_context.batch_ids, dtype=object)
                if fit_context.batch_ids
                else None
            )
        else:
            batch_arr = np.asarray(list(batch_labels), dtype=object).ravel()
            if int(batch_arr.size) != int(y_arr.size):
                raise ValueError(
                    f"batch_labels has {batch_arr.size} rows but X has {y_arr.size} rows."
                )
            if fit_context.batch_ids and tuple(
                typed_scalar_key(value) for value in batch_arr.tolist()
            ) != tuple(typed_scalar_key(value) for value in fit_context.batch_ids):
                raise ResamplingContractError(
                    "batch_labels and resampling_context.batch_ids describe different row metadata.",
                    code="batch_identity_mismatch",
                    diagnostics={"n_rows": int(y_arr.size)},
                )

        active_diakrino_dataset_id = self._base_dataset_name(str(dataset_name or "dataset"))
        if str(getattr(self, "_active_diakrino_dataset_id", "") or "") != active_diakrino_dataset_id:
            self._diakrino_transform_sidecar_cache = None
            try:
                setattr(self.dist_fitter, "_diakrino_sidecar_cache", "unset")
            except Exception:
                pass
        self._active_diakrino_dataset_id = active_diakrino_dataset_id
        try:
            setattr(self.dist_fitter, "_active_diakrino_dataset_id", active_diakrino_dataset_id)
        except Exception:
            pass

        self._active_resampling_plans = {}
        self._active_fit_resampling_context = fit_context
        self._active_fs_resampling_context = None
        self._typed_feature_selector_runtime = None
        self._typed_feature_selector_admission = {}
        fit_sample_weight = (
            None
            if not fit_context.sample_weights
            else np.asarray(fit_context.sample_weights, dtype=float)
        )
        self._sample_weight_provenance = {
            "sample_weight_requested": bool(fit_sample_weight is not None),
            "sample_weight_feature_selection_consumed": False,
            "sample_weight_stage2_fit_consumed": False,
            "sample_weight_stage2_cv_consumed": False,
            "sample_weight_posthoc_calibration_consumed": False,
            "sample_weight_metrics_consumed": False,
        }
        self._typed_feature_selector_sample_weight_requested = bool(
            fit_sample_weight is not None
        )
        if typed_requested:
            self._typed_feature_selector_runtime = FeatureSelectorRuntimeFacts(
                input_is_sparse=bool(is_sparse_input(X)),
                input_has_categorical=bool(
                    source_schema is not None
                    and any(
                        feature.role is FeatureRole.CATEGORICAL
                        for feature in source_schema.features
                    )
                ),
                input_has_missing=bool(
                    typed_preprocessor is not None
                    and getattr(typed_preprocessor, "input_has_missing_", False)
                ),
                sample_weight_requested=False,
                structured_resampling_requested=fit_context.policy.kind != "iid",
                fold_local_adapter=str(
                    typed_preprocessing.get("adapter", "numeric") or "numeric"
                ),
                structured_output_required=False,
            )

        rng = np.random.default_rng(seed)
        n_features = int(X_train_arr.shape[1])
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_arr)
        X_apply_imp = np.asarray(X_train_imp, dtype=float).copy()

        batch_model, batch_fit_meta = fit_batch_correction_model(
            X_train_imp,
            batch_labels=batch_arr,
            mode=str(getattr(self.config, "batch_correction", "none") or "none"),
            combat_prior_strength=float(
                getattr(self.config, "batch_correction_combat_prior_strength", 8.0)
                or 8.0
            ),
            cdf_center_n_quantiles=int(
                getattr(self.config, "batch_correction_cdf_n_quantiles", 33) or 33
            ),
            cdf_center_clip_quantiles=(
                float(getattr(self.config, "batch_correction_cdf_clip_low", 0.01) or 0.01),
                float(getattr(self.config, "batch_correction_cdf_clip_high", 0.99) or 0.99),
            ),
        )
        X_train_imp, X_apply_imp, batch_apply_meta = apply_batch_correction_model(
            X_train_imp,
            X_apply_imp,
            model=batch_model,
            batch_labels_train=batch_arr,
            batch_labels_test=batch_arr,
        )
        X_train_model_input, X_apply_model_input, face_meta = (
            self._maybe_apply_face_domain_projection(
                X_train_imp=X_train_imp,
                y_train=y_arr,
                X_test_imp=X_apply_imp,
                dataset_name=str(dataset_name),
                seed=seed,
            )
        )
        X_train_model_input, X_apply_model_input, ratio_meta = (
            self._ratio_feature_generation(
                X_train_imp=X_train_model_input,
                y_train=y_arr,
                X_test_imp=X_apply_model_input,
                seed=seed,
                face_projection_applied=bool(
                    face_meta.get("face_projection_applied", False)
                ),
            )
        )

        identity_breaks: list[str] = []
        if bool(face_meta.get("face_projection_applied", False)):
            identity_breaks.append("face_projection")
        if bool(ratio_meta.get("ratio_features_applied", False)):
            identity_breaks.append("ratio_generation")
        if int(X_train_model_input.shape[1]) != int(n_features):
            identity_breaks.append("feature_width_changed")
        if self._df_stage_position() != "after_fs" and str(
            getattr(self.config, "folding_method", "pls_da") or "pls_da"
        ).strip().lower() != "none":
            identity_breaks.append("pre_fs_folding")
        self._diakrino_prefilter_identity_available = not bool(identity_breaks)
        self._diakrino_prefilter_identity_reason = (
            "original_feature_indices"
            if not identity_breaks
            else "+".join(dict.fromkeys(identity_breaks))
        )

        scaler_mode = str(
            getattr(self.config, "scaler_mode", "standard") or "standard"
        ).strip().lower()
        if scaler_mode == "robust":
            from sklearn.preprocessing import RobustScaler

            scaler_base = RobustScaler()
        elif scaler_mode == "quantile":
            from sklearn.preprocessing import QuantileTransformer

            scaler_base = QuantileTransformer(
                output_distribution="normal", random_state=seed
            )
        else:
            scaler_base = StandardScaler()
        X_train_base = scaler_base.fit_transform(X_train_model_input)
        X_apply_base = scaler_base.transform(X_apply_model_input)

        df_stage_position = self._df_stage_position()
        if df_stage_position == "before_fs":
            X_train_trans, X_apply_trans, _, dist_meta = (
                self._distribution_transform_block(
                    X_train_model_input,
                    X_apply_model_input,
                    X_train_base,
                    X_apply_base,
                    y_arr,
                    seed,
                    rng,
                )
            )
            dist_meta = self._annotate_distribution_stage_meta(
                dist_meta,
                stage_position="before_fs",
                source_space="model_input",
            )
        else:
            X_train_trans = np.asarray(X_train_base, dtype=float)
            X_apply_trans = np.asarray(X_apply_base, dtype=float)
            self._last_distribution_plan = {
                "schema_version": "1.0",
                "apply_cdf_transform": bool(self.config.apply_cdf_transform),
                "n_input_features": 0,
                "dist_feature_indices": [],
                "feature_plans": [],
            }
            dist_meta = self._annotate_distribution_stage_meta(
                {}, stage_position="after_fs", source_space="pending"
            )

        folding_prefilter_k = getattr(self.config, "folding_prefilter_k", None)
        prefilter_top_k_override = None
        if str(
            getattr(self.config, "folding_method", "pls_da") or "pls_da"
        ).strip().lower() != "none" and folding_prefilter_k is not None:
            try:
                prefilter_top_k_override = int(max(1, int(folding_prefilter_k)))
            except (TypeError, ValueError, OverflowError):
                prefilter_top_k_override = None
        X_train_fs_input, X_apply_fs_input, prefilter_idx = self._rank_prefilter(
            X_train_trans,
            y_arr,
            X_apply_trans,
            seed,
            top_k_override=prefilter_top_k_override,
        )
        X_train_prefilter_raw = np.asarray(
            X_train_model_input[:, prefilter_idx], dtype=float
        )
        X_apply_prefilter_raw = np.asarray(
            X_apply_model_input[:, prefilter_idx], dtype=float
        )
        X_train_prefilter_base = np.asarray(X_train_base[:, prefilter_idx], dtype=float)
        X_apply_prefilter_base = np.asarray(X_apply_base[:, prefilter_idx], dtype=float)
        post_df_source_space = "prefilter_raw"
        if df_stage_position == "after_fs":
            folding_meta: Dict[str, Any] = {}
            self._last_folding_state = {}
        else:
            X_train_fs_input, X_apply_fs_input, folding_meta = (
                self._apply_folding_stage(
                    X_train_fs_input=X_train_fs_input,
                    X_test_fs_input=X_apply_fs_input,
                    y_train=y_arr,
                    seed=seed,
                )
            )
            if bool(folding_meta.get("folding_applied", False)):
                post_df_source_space = "folded_selector_input"
                X_train_prefilter_raw = np.asarray(X_train_fs_input, dtype=float)
                X_apply_prefilter_raw = np.asarray(X_apply_fs_input, dtype=float)
                X_train_prefilter_base = np.asarray(X_train_fs_input, dtype=float)
                X_apply_prefilter_base = np.asarray(X_apply_fs_input, dtype=float)

        if fit_context.policy.kind == "iid":
            fs_idx_local = self._sample_fs_indices(
                y_arr,
                fs_fraction=self.config.fs_fraction,
                seed=seed,
                use_balanced=self.config.use_balanced_fs_subsample,
                min_per_class=self.config.fs_min_per_class,
            )
        else:
            fs_idx_local = np.asarray(
                resolve_fit_subsample(
                    fit_context,
                    y_arr,
                    fraction=float(self.config.fs_fraction),
                    seed=seed,
                    balanced=bool(self.config.use_balanced_fs_subsample),
                    min_per_class=int(self.config.fs_min_per_class),
                ),
                dtype=int,
            )
        fs_context = fit_context.take(
            fs_idx_local,
            parent_split_fingerprint=fit_context.fingerprint,
        )
        X_fs = np.asarray(X_train_fs_input[fs_idx_local], dtype=float)

        variance_keep_indices = np.arange(X_train_fs_input.shape[1], dtype=int)
        if bool(getattr(self.config, "prefilter_variance_floor_enabled", True)):
            threshold = float(
                getattr(self.config, "prefilter_variance_floor_threshold", 1e-6)
            )
            mode_frequency = float(
                getattr(self.config, "prefilter_variance_floor_mode_freq", 0.99)
            )
            keep_mask = np.ones(X_train_fs_input.shape[1], dtype=bool)
            for column_index in range(X_train_fs_input.shape[1]):
                column = X_train_fs_input[:, column_index]
                if np.var(column) < threshold:
                    keep_mask[column_index] = False
                    continue
                _, counts = np.unique(column, return_counts=True)
                if counts.max() / len(column) > mode_frequency:
                    keep_mask[column_index] = False
            if bool(np.any(keep_mask)) and not bool(np.all(keep_mask)):
                variance_keep_indices = np.flatnonzero(keep_mask).astype(int)
                self._update_diakrino_prefilter_active_mask(keep_mask)
                X_fs = X_fs[:, keep_mask]
                X_train_fs_input = X_train_fs_input[:, keep_mask]
                X_apply_fs_input = X_apply_fs_input[:, keep_mask]
                if X_train_prefilter_raw.shape[1] == keep_mask.size:
                    X_train_prefilter_raw = X_train_prefilter_raw[:, keep_mask]
                    X_apply_prefilter_raw = X_apply_prefilter_raw[:, keep_mask]
                    X_train_prefilter_base = X_train_prefilter_base[:, keep_mask]
                    X_apply_prefilter_base = X_apply_prefilter_base[:, keep_mask]

        candidate = self._choose_selector_candidate(
            X_fs=X_fs,
            y_fs=y_arr[fs_idx_local],
            X_train_full=X_train_fs_input,
            X_test_full=X_apply_fs_input,
            y_train_full=y_arr,
            seed=seed,
            dataset_name=str(dataset_name),
            fit_resampling_context=fit_context,
            fs_resampling_context=fs_context,
            post_df_source_raw_train=X_train_prefilter_raw,
            post_df_source_raw_test=X_apply_prefilter_raw,
            post_df_source_base_train=X_train_prefilter_base,
            post_df_source_base_test=X_apply_prefilter_base,
            post_df_source_space=post_df_source_space,
        )
        model = candidate["model"]
        model_name = str(candidate["model_name"])
        model_cv_meta = dict(candidate.get("model_cv_meta") or {})
        if bool(model_cv_meta.get("model_cv_sample_weight_cv_routed", False)):
            self._sample_weight_provenance["sample_weight_stage2_cv_consumed"] = True
        X_train_selected = np.asarray(candidate["X_train_sel"], dtype=float)
        final_balance = self._apply_final_training_balance(
            X_train_selected,
            y_arr,
            seed=int(seed),
            fit_context=fit_context,
            sample_weight=fit_sample_weight,
            callsite="fit_components_final_fit",
        )
        if final_balance is None:
            final_X_train = X_train_selected
            final_y_train = y_arr
            final_sample_weight = fit_sample_weight
        else:
            final_X_train = final_balance.X
            final_y_train = final_balance.y
            final_sample_weight = final_balance.sample_weight
            model_cv_meta["training_balance_final_fit_provenance"] = (
                final_balance.provenance.to_dict()
            )
        final_weight_route = fit_estimator_with_sample_weight(
            model,
            final_X_train,
            final_y_train,
            sample_weight=final_sample_weight,
        )
        if final_sample_weight is not None:
            self._sample_weight_provenance["sample_weight_stage2_fit_consumed"] = True
        base_descriptor = self._inspect_selected_classifier(
            model,
            final_X_train,
            model_cv_meta=model_cv_meta,
            effective_model_name=model_name,
            sample_weight_requested=final_sample_weight is not None,
            sample_weight_routed_observation=final_weight_route,
        )
        final_descriptor = self._inspect_selected_classifier(
            model,
            final_X_train,
            model_cv_meta=model_cv_meta,
            effective_model_name=model_name,
            requested_device=base_descriptor.requested_device,
            sample_weight_requested=final_sample_weight is not None,
            sample_weight_routed_observation=final_weight_route,
        )
        descriptor_metadata = self._fitted_descriptor_metadata(
            base_descriptor=base_descriptor,
            final_descriptor=final_descriptor,
        )

        if df_stage_position == "after_fs":
            folding_meta = dict(candidate.get("_folding_meta", {}) or {})
            self._last_folding_state = dict(candidate.get("_folding_state", {}) or {})
            post_df_meta = dict(candidate.get("_post_df_meta", {}) or {})
            if post_df_meta:
                dist_meta = self._annotate_distribution_stage_meta(
                    post_df_meta,
                    stage_position="after_fs",
                    source_space=str(
                        post_df_meta.get("df_stage_source_space", post_df_source_space)
                    ),
                )

        selected_local = np.asarray(candidate.get("selected_indices", tuple()), dtype=int)
        selected_model_input_indices: Tuple[int, ...] = tuple()
        if not bool(face_meta.get("face_projection_applied", False)) and not (
            df_stage_position != "after_fs"
            and bool(folding_meta.get("folding_applied", False))
        ):
            selected_model_input_indices = tuple(
                int(prefilter_idx[variance_keep_indices[index]])
                for index in selected_local
                if 0 <= index < variance_keep_indices.size
                and 0 <= variance_keep_indices[index] < prefilter_idx.size
            )
        selected_feature_schema: Dict[str, Any] = {}
        if model_input_schema is not None and selected_model_input_indices:
            source_to_model_identity = bool(
                source_schema is not None
                and source_schema.n_features == model_input_schema.n_features
                and source_schema.feature_names == model_input_schema.feature_names
            )
            if not identity_breaks:
                try:
                    selected_feature_schema = model_input_schema.select(
                        selected_model_input_indices,
                        operation="feature_selection_output",
                    ).to_record()
                except (SchemaContractError, ValueError, IndexError) as exc:
                    selected_feature_schema = {
                        "status": "unavailable",
                        "reason": type(exc).__name__,
                    }
            if not source_to_model_identity:
                selected_model_input_indices = tuple()

        # Candidate evaluation owns the fitted post-selection ratio route as a
        # nested record.  Preserve it verbatim so runtime replay augments the
        # selected matrix to the exact width used by the final classifier.
        stage2_ratio_meta = dict(candidate.get("stage2_ratio_meta") or {})
        face_state = dict(getattr(self, "_last_face_projection_state", {}) or {})
        folding_state = dict(getattr(self, "_last_folding_state", {}) or {})
        fitted_selector = candidate.get("_fitted_selector")
        if fitted_selector is None:
            raise RuntimeError("Feature-selection fit did not produce a reusable selector.")
        runtime_model = DFFSReproducibleModel(
            n_input_features=int(n_features),
            imputer=imputer,
            batch_model=batch_model if isinstance(batch_model, dict) else None,
            face_meta=face_meta,
            face_pca=face_state.get("pca_model"),
            face_lda=face_state.get("lda_model"),
            ratio_meta=ratio_meta,
            scaler_base=scaler_base,
            distribution_plan=dict(getattr(self, "_last_distribution_plan", {}) or {}),
            prefilter_indices=tuple(
                int(value) for value in np.asarray(prefilter_idx, dtype=int).ravel().tolist()
            ),
            folding_meta=folding_meta,
            folding_transformer=folding_state.get("transformer"),
            folding_standardize_mean=folding_state.get("standardize_mean"),
            folding_standardize_scale=folding_state.get("standardize_scale"),
            selector=fitted_selector,
            stage2_ratio_meta=stage2_ratio_meta,
            classifier_model=model,
            variance_keep_indices=tuple(
                int(value) for value in variance_keep_indices.tolist()
            ),
            metadata={
                "dataset_name": str(dataset_name),
                "seed": int(seed),
                "n_train": int(y_arr.size),
                "model_name": model_name,
                "training_only": True,
                "evaluation_metrics_emitted": False,
                "classification_final_fitted_descriptor": _json_safe(
                    descriptor_metadata.get("classification_final_fitted_descriptor", {})
                ),
            },
        )
        snapshot = self._config_snapshot()
        snapshot.update(
            {
                "train_only_components": True,
                "fit_context": _json_safe(
                    fit_context.to_metadata(
                        sample_weights_consumed=fit_sample_weight is not None,
                        sample_weight_usage=(
                            "stage2_cv_and_final_fit"
                            if fit_sample_weight is not None
                            else "not_requested"
                        ),
                    )
                ),
                "typed_preprocessing": _json_safe(typed_preprocessing),
                "classification_selected_identity": _json_safe(
                    model_cv_meta.get("classification_selected_identity", {})
                ),
                "classification_final_fitted_descriptor": _json_safe(
                    descriptor_metadata.get("classification_final_fitted_descriptor", {})
                ),
                "variance_keep_indices": [
                    int(value) for value in variance_keep_indices.tolist()
                ],
            }
        )
        fit_provenance = {
            "schema_version": "tabnetics_train_only_components_v1",
            "training_only": True,
            "evaluation_metrics_emitted": False,
            "n_fit_rows": int(y_arr.size),
            "fit_context_fingerprint": str(fit_context.fingerprint),
            "fit_policy": "configured_training_cv_then_full_refit",
            "sample_weight_provenance": _json_safe(self._sample_weight_provenance),
            "batch_fit": _json_safe(batch_fit_meta),
            "batch_apply": _json_safe(batch_apply_meta),
            "distribution_stage": _json_safe(dist_meta),
            "feature_selection": _json_safe(
                candidate.get("fs_selection_summary", {})
            ),
            "diakrino_sidecar_resolution": _json_safe(
                self._diakrino_sidecar_resolution_diagnostics(int(n_features))
            ),
            "diakrino_prefilter": _json_safe(
                dict(getattr(self, "_last_diakrino_prefilter_state", {}) or {})
            ),
            "diakrino_protected_augmentation": _json_safe(
                dict(candidate.get("diakrino_protected_augmentation", {}) or {})
            ),
            "classifier_selection": _json_safe(model_cv_meta),
            "training_balance": _json_safe(
                {
                    "config": self._training_balance_cfg().to_dict(),
                    "final_fit": dict(
                        getattr(self, "_training_balance_final_provenance", {}) or {}
                    ),
                }
            ),
        }
        classes = np.asarray(getattr(model, "classes_", np.unique(y_arr))).ravel()
        if classes.size < 2:
            raise RuntimeError("The fitted classifier does not expose a valid class order.")
        self._validate_train_only_diakrino_fitted_state(
            runtime_model=runtime_model,
            fit_provenance=fit_provenance,
            source_schema=source_schema,
            model_input_schema=model_input_schema,
            selected_model_input_indices=selected_model_input_indices,
            selected_feature_schema=selected_feature_schema,
        )
        return FittedPipelineComponents(
            runtime_model=runtime_model,
            classes=classes,
            fit_resampling_context=fit_context,
            config_snapshot=_json_safe(snapshot),
            model_name=model_name,
            source_schema=source_schema,
            model_input_schema=model_input_schema,
            selected_feature_schema=_json_safe(selected_feature_schema),
            typed_preprocessor=typed_preprocessor,
            typed_sparse_dense_max_elements=int(
                getattr(self.config, "typed_sparse_dense_max_elements", 0) or 0
            ),
            variance_keep_indices=tuple(
                int(value) for value in variance_keep_indices.tolist()
            ),
            selected_model_input_indices=selected_model_input_indices,
            fit_provenance=_json_safe(fit_provenance),
        )

    @_scope_pipeline_sklearn_n_jobs
    def run(
        self,
        X: Any,
        y: np.ndarray,
        dataset_name: str = "dataset",
        seed: Optional[int] = None,
        batch_labels: Optional[Sequence[Any]] = None,
        *,
        sample_weight: Optional[Sequence[float]] = None,
        schema: DatasetSchema | None = None,
        resampling_context: Optional[FitResamplingContext] = None,
        supplied_split_id: Optional[str] = None,
        capture_artifacts: bool = False,
        capture_diagnostics: bool = False,
    ) -> PipelineRunResult:
        seed = int(self.config.random_seed if seed is None else seed)
        typed_requested = self._typed_frontend_requested(X, schema=schema)
        input_schema: DatasetSchema | None = None
        if typed_requested:
            input_schema = infer_dataset_schema(X, schema=schema)
            n_rows = int(X.shape[0])
            n_columns = int(X.shape[1])
            if n_columns != int(input_schema.n_features):
                raise SchemaContractError(
                    "Input width does not match the resolved typed input schema."
                )
            X_arr: np.ndarray | None = None
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim != 2:
                raise ValueError(f"Expected 2D X, got shape {X_arr.shape}")
            n_rows = int(X_arr.shape[0])
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()
        if int(y_arr.size) != n_rows:
            raise ValueError(f"y has {y_arr.size} rows but X has {n_rows}.")
        supplied_sample_weights = coerce_sample_weights(
            sample_weight,
            n_rows=n_rows,
            field_name="sample_weight",
        )
        batch_arr: Optional[np.ndarray]
        if batch_labels is None:
            batch_arr = None
        else:
            batch_arr = np.asarray(list(batch_labels), dtype=object).ravel()
            if int(batch_arr.size) != n_rows:
                raise ValueError(
                    f"batch_labels has {batch_arr.size} rows but X has {n_rows}."
                )

        if resampling_context is None:
            full_context = FitResamplingContext(
                n_rows=n_rows,
                batch_ids=(
                    tuple()
                    if batch_arr is None
                    else tuple(batch_arr.tolist())
                ),
                sample_weights=supplied_sample_weights,
                policy=ResamplingPolicy(kind="iid"),
            )
        else:
            full_context = ensure_fit_resampling_context(
                resampling_context,
                n_rows=n_rows,
            )
            if supplied_sample_weights:
                if not full_context.sample_weights:
                    raise ResamplingContractError(
                        "sample_weight was supplied separately but the resampling "
                        "context has no aligned sample_weights vector.",
                        code="sample_weight_context_missing",
                        diagnostics={"n_rows": n_rows},
                    )
                if tuple(full_context.sample_weights) != tuple(supplied_sample_weights):
                    raise ResamplingContractError(
                        "sample_weight does not match resampling_context.sample_weights.",
                        code="sample_weight_context_mismatch",
                        diagnostics={"n_rows": n_rows},
                    )
            if batch_arr is not None and full_context.batch_ids:
                context_batch_keys = tuple(
                    typed_scalar_key(value) for value in full_context.batch_ids
                )
                supplied_batch_keys = tuple(
                    typed_scalar_key(value) for value in batch_arr.tolist()
                )
                if context_batch_keys != supplied_batch_keys:
                    raise ResamplingContractError(
                        "batch_labels and resampling_context.batch_ids describe different row metadata.",
                        code="batch_identity_mismatch",
                        diagnostics={"n_rows": n_rows},
                    )

        outer_plan = self._resolve_outer_split_plan(
            y_arr,
            seed=seed,
            resampling_context=full_context,
            supplied_split_id=supplied_split_id,
        )
        train_idx = np.asarray(outer_plan.primary.train_indices, dtype=int)
        test_idx = np.asarray(outer_plan.primary.test_indices, dtype=int)

        return self.run_pre_split(
            X_train=(
                self._take_input_rows(X, train_idx)
                if typed_requested
                else X_arr[train_idx]
            ),
            y_train=y_arr[train_idx],
            X_test=(
                self._take_input_rows(X, test_idx)
                if typed_requested
                else X_arr[test_idx]
            ),
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
            sample_weight_train=(
                None
                if not full_context.sample_weights
                else np.asarray(full_context.sample_weights, dtype=float)[train_idx]
            ),
            sample_weight_test=(
                None
                if not full_context.sample_weights
                else np.asarray(full_context.sample_weights, dtype=float)[test_idx]
            ),
            schema=input_schema,
            resampling_context=full_context,
            resolved_outer_split=outer_plan,
            capture_artifacts=bool(capture_artifacts),
            capture_diagnostics=bool(capture_diagnostics),
        )

    def _resolve_meta_learning_runtime_config(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[Optional[DFFSConfig], Dict[str, Any]]:
        if bool(getattr(self.config, "auto_router_enabled", False)):
            selected_meta: Dict[str, Any]
            try:
                from tabnetics.auto_router import apply_router_output, predict_auto_router

                model_path = str(getattr(self.config, "auto_router_artifact_path", "") or "").strip()
                diakrino_dispersion = self._diakrino_router_dispersion_descriptor()
                router_metadata: Dict[str, Any] = {}
                if bool(diakrino_dispersion.get("enabled", False)):
                    router_metadata["diakrino_router_dispersion_descriptor"] = diakrino_dispersion
                output = predict_auto_router(
                    np.asarray(X_train, dtype=float),
                    np.asarray(y_train).ravel(),
                    metadata=router_metadata,
                    model_dir=(Path(model_path) if model_path else None),
                    descriptor_ood_gate_enabled=bool(
                        getattr(self.config, "auto_router_descriptor_ood_gate_enabled", False)
                    ),
                    crossfit_uncertainty_enabled=bool(
                        getattr(self.config, "auto_router_crossfit_uncertainty_enabled", False)
                    ),
                )
                selected_meta = output.to_snapshot()
                if router_metadata:
                    selected_meta["auto_router_diakrino_dispersion_descriptor"] = diakrino_dispersion
                active_config = copy.deepcopy(self.config)
                apply_router_output(active_config, output)
                active_config.meta_learning_selector_mode = "none"
                active_config.auto_router_enabled = False
                return active_config, selected_meta
            except Exception as exc:
                selected_meta = {
                    "auto_router_enabled": True,
                    "auto_router_used": False,
                    "auto_router_version": "v25_calibrated_score_router",
                    "auto_router_error": str(type(exc).__name__),
                    "auto_router_error_message": str(exc),
                    "auto_router_fail_open": bool(getattr(self.config, "auto_router_fail_open", True)),
                }
                if not bool(getattr(self.config, "auto_router_fail_open", True)):
                    raise
                active_config = copy.deepcopy(self.config)
                active_config.auto_router_enabled = False
                return active_config, selected_meta
        mode = str(getattr(self.config, "meta_learning_selector_mode", "none") or "none").strip().lower()
        if mode == "none":
            return None, {}
        if mode == "flag_selector_v1":
            try:
                from meta.data_adapter import compute_descriptor
            except Exception:
                compute_descriptor = None  # type: ignore[assignment]
            try:
                from tabnetics.feature_selection.flag_selector_runtime import FlagSelector, apply_selector_output
            except Exception:
                from tabnetics.feature_selection.flag_selector_runtime import FlagSelector, apply_selector_output  # type: ignore
            selected_meta: Dict[str, Any]
            selector = None
            try:
                model_path = str(getattr(self.config, "meta_learning_records_path", "") or "").strip()
                if not model_path:
                    raise ValueError("flag_selector_v1 requires meta_learning_records_path to point at an artifact directory.")
                selector = FlagSelector.load(Path(model_path))
                if compute_descriptor is None:
                    raise ImportError("meta.data_adapter.compute_descriptor is unavailable for flag_selector_v1.")
                descriptor = compute_descriptor(
                    np.asarray(X_train, dtype=float),
                    np.asarray(y_train).ravel(),
                    metadata={},
                )
                output = selector.predict(descriptor)
                selected_meta = output.to_snapshot()
            except Exception as exc:
                fallback_profile = (
                    str(getattr(getattr(selector, "config", None), "fallback_profile", "v16_ref") or "v16_ref")
                    if selector is not None
                    else "v16_ref"
                )
                selected_meta = {
                    "selector_predicted_profile": fallback_profile,
                    "selector_raw_profile": fallback_profile,
                    "selector_confidence": 0.0,
                    "selector_raw_confidence": 0.0,
                    "selector_used": False,
                    "selector_abstain": True,
                    "selector_fallback_profile": fallback_profile,
                    "selector_ranked_topk": [fallback_profile],
                    "meta_learning_profile_selected": fallback_profile,
                    "meta_learning_profile_raw": fallback_profile,
                    "meta_learning_confidence": 0.0,
                    "meta_learning_fallback_applied": True,
                    "meta_learning_candidate_profiles": [fallback_profile],
                    "meta_learning_fallback_profile": fallback_profile,
                    "meta_learning_error": str(type(exc).__name__),
                }
                output = None
            active_config = copy.deepcopy(self.config)
            active_config.meta_learning_selector_mode = "none"
            if output is not None:
                apply_selector_output(active_config, output)
            return active_config, selected_meta
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
        selector = None
        try:
            selector = MetaLearningSelector(
                mode=str(mode),
                confidence_threshold=float(
                    getattr(self.config, "meta_learning_confidence_threshold", 0.55) or 0.55
                ),
                records_path=(
                    Path(str(getattr(self.config, "meta_learning_records_path", "")).strip())
                    if str(getattr(self.config, "meta_learning_records_path", "")).strip()
                    else None
                ),
                random_state=int(getattr(self.config, "random_seed", 42) or 42),
            ).fit()
            selected_meta = selector.predict_from_arrays(
                np.asarray(X_train, dtype=float),
                np.asarray(y_train).ravel(),
            )
        except Exception as exc:
            fallback_profile = (
                str(getattr(selector, "fallback_profile_", "v16_ref") or "v16_ref")
                if selector is not None
                else "v16_ref"
            )
            selected_meta = {
                "meta_learning_profile_selected": fallback_profile,
                "meta_learning_profile_raw": fallback_profile,
                "meta_learning_confidence": 0.0,
                "meta_learning_fallback_applied": True,
                "meta_learning_candidate_profiles": list(
                    getattr(selector, "profile_labels_", SUPPORTED_RUNTIME_PROFILES)
                    if selector is not None
                    else SUPPORTED_RUNTIME_PROFILES
                ),
                "meta_learning_fallback_profile": fallback_profile,
                "meta_learning_error": str(type(exc).__name__),
            }

        active_config = copy.deepcopy(self.config)
        active_config.meta_learning_selector_mode = "none"
        apply_runtime_profile_overlay(
            active_config,
            str(selected_meta.get("meta_learning_profile_selected", "v16_ref") or "v16_ref"),
            runtime_profile_overlays=getattr(selector, "runtime_profile_overlays_", None),
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

    def _prepare_pre_split_resampling(
        self,
        *,
        y_train: np.ndarray,
        y_test: np.ndarray,
        split_indices_train: Optional[Sequence[int]],
        split_indices_test: Optional[Sequence[int]],
        batch_labels_train: Optional[np.ndarray],
        batch_labels_test: Optional[np.ndarray],
        sample_weight_train: Optional[Sequence[float]] = None,
        sample_weight_test: Optional[Sequence[float]] = None,
        resampling_context: Optional[FitResamplingContext],
        resolved_outer_split: Optional[ResolvedSplitPlan],
    ) -> Tuple[
        FitResamplingContext,
        FitResamplingContext,
        ResolvedSplitPlan,
        Tuple[int, ...],
        Tuple[int, ...],
    ]:
        """Validate an explicit split and derive its aligned fit-row context."""

        n_train = int(np.asarray(y_train).ravel().size)
        n_test = int(np.asarray(y_test).ravel().size)
        n_observed = n_train + n_test
        provided_train_weights = coerce_sample_weights(
            sample_weight_train,
            n_rows=n_train,
            field_name="sample_weight_train",
        )
        provided_test_weights = coerce_sample_weights(
            sample_weight_test,
            n_rows=n_test,
            field_name="sample_weight_test",
        )
        if bool(provided_train_weights) != bool(provided_test_weights):
            raise ResamplingContractError(
                "sample_weight_train and sample_weight_test must be supplied together.",
                code="partial_sample_weights",
                diagnostics={"n_train": n_train, "n_test": n_test},
            )
        provided_train = (
            None
            if split_indices_train is None
            else tuple(
                int(value)
                for value in np.asarray(split_indices_train, dtype=int).ravel().tolist()
            )
        )
        provided_test = (
            None
            if split_indices_test is None
            else tuple(
                int(value)
                for value in np.asarray(split_indices_test, dtype=int).ravel().tolist()
            )
        )
        if (provided_train is None) != (provided_test is None):
            raise ResamplingContractError(
                "split_indices_train and split_indices_test must be supplied together.",
                code="partial_split_indices",
                diagnostics={"n_train": n_train, "n_test": n_test},
            )
        if provided_train is not None and len(provided_train) != n_train:
            raise ResamplingContractError(
                "split_indices_train length does not match X_train/y_train.",
                code="split_partition_length_mismatch",
                diagnostics={
                    "partition": "train",
                    "expected_rows": n_train,
                    "observed_rows": len(provided_train),
                },
            )
        if provided_test is not None and len(provided_test) != n_test:
            raise ResamplingContractError(
                "split_indices_test length does not match X_test/y_test.",
                code="split_partition_length_mismatch",
                diagnostics={
                    "partition": "test",
                    "expected_rows": n_test,
                    "observed_rows": len(provided_test),
                },
            )

        if resampling_context is None:
            if resolved_outer_split is not None:
                raise ResamplingContractError(
                    "resolved_outer_split requires its full-row resampling_context.",
                    code="resolved_split_context_missing",
                    diagnostics={"purpose": resolved_outer_split.purpose},
                )
            public_train = (
                tuple(range(n_train))
                if provided_train is None
                else provided_train
            )
            public_test = (
                tuple(range(n_train, n_observed))
                if provided_test is None
                else provided_test
            )
            row_ids = (*public_train, *public_test)
            batch_ids: tuple[Any, ...] = tuple()
            if batch_labels_train is not None or batch_labels_test is not None:
                train_batch = (
                    (None,) * n_train
                    if batch_labels_train is None
                    else tuple(batch_labels_train.tolist())
                )
                test_batch = (
                    (None,) * n_test
                    if batch_labels_test is None
                    else tuple(batch_labels_test.tolist())
                )
                batch_ids = (*train_batch, *test_batch)
            full_context = FitResamplingContext(
                n_rows=n_observed,
                row_ids=row_ids,
                batch_ids=batch_ids,
                sample_weights=(
                    (*provided_train_weights, *provided_test_weights)
                    if provided_train_weights
                    else tuple()
                ),
                policy=ResamplingPolicy(kind="iid"),
            )
            assignment = SplitAssignment(
                scope="outer",
                split_id="pre-split",
                train_indices=tuple(range(n_train)),
                test_indices=tuple(range(n_train, n_observed)),
                source="caller_pre_split",
                parent_context_fingerprint=full_context.base_fingerprint,
            )
            full_y = tuple(np.asarray(y_train).ravel().tolist()) + tuple(
                np.asarray(y_test).ravel().tolist()
            )
            outer_plan = resolve_assignment(
                full_context,
                assignment,
                y=full_y,
                purpose="outer",
            )
            train_idx_out = public_train
            test_idx_out = public_test
        else:
            if not isinstance(resampling_context, FitResamplingContext):
                raise ResamplingContractError(
                    "resampling_context must be a FitResamplingContext.",
                    code="invalid_resampling_context",
                    diagnostics={
                        "context_type": (
                            f"{type(resampling_context).__module__}."
                            f"{type(resampling_context).__qualname__}"
                        )
                    },
                )
            full_context = resampling_context
            if provided_train_weights:
                if not full_context.sample_weights:
                    raise ResamplingContractError(
                        "Explicit sample weights require aligned "
                        "resampling_context.sample_weights.",
                        code="sample_weight_context_missing",
                        diagnostics={"context_rows": full_context.n_rows},
                    )
            if resolved_outer_split is not None:
                outer_plan = resolved_outer_split
                if outer_plan.context_fingerprint != full_context.fingerprint:
                    raise ResamplingContractError(
                        "resolved_outer_split belongs to a different row context.",
                        code="resolved_split_context_mismatch",
                        diagnostics={
                            "expected_context_fingerprint": full_context.fingerprint,
                            "observed_context_fingerprint": outer_plan.context_fingerprint,
                        },
                    )
                assignment = outer_plan.primary.assignment
            else:
                if provided_train is None or provided_test is None:
                    raise ResamplingContractError(
                        "A full-row resampling_context requires explicit split indices or a resolved split plan.",
                        code="pre_split_assignment_missing",
                        diagnostics={
                            "context_rows": full_context.n_rows,
                            "n_train": n_train,
                            "n_test": n_test,
                        },
                    )
                assignment = SplitAssignment(
                    scope="outer",
                    split_id="pre-split",
                    train_indices=provided_train,
                    test_indices=provided_test,
                    source="caller_pre_split",
                    allow_unassigned=(
                        len(set(provided_train).union(provided_test))
                        < full_context.n_rows
                    ),
                    parent_context_fingerprint=full_context.base_fingerprint,
                )
                full_y: Optional[tuple[Any, ...]] = None
                if len(set(provided_train).union(provided_test)) == full_context.n_rows:
                    staged: list[Any] = [None] * full_context.n_rows
                    for position, value in zip(
                        provided_train,
                        np.asarray(y_train).ravel().tolist(),
                    ):
                        staged[position] = value
                    for position, value in zip(
                        provided_test,
                        np.asarray(y_test).ravel().tolist(),
                    ):
                        staged[position] = value
                    full_y = tuple(staged)
                outer_plan = resolve_assignment(
                    full_context,
                    assignment,
                    y=full_y,
                    purpose="outer",
                )
            if len(assignment.train_indices) != n_train or len(assignment.test_indices) != n_test:
                raise ResamplingContractError(
                    "Resolved split partition sizes do not match the provided arrays.",
                    code="split_partition_length_mismatch",
                    diagnostics={
                        "resolved_train_rows": len(assignment.train_indices),
                        "provided_train_rows": n_train,
                        "resolved_test_rows": len(assignment.test_indices),
                        "provided_test_rows": n_test,
                    },
                )
            if provided_train is not None and provided_train != assignment.train_indices:
                raise ResamplingContractError(
                    "split_indices_train does not match resolved_outer_split.",
                    code="resolved_split_indices_mismatch",
                    diagnostics={"partition": "train"},
                )
            if provided_test is not None and provided_test != assignment.test_indices:
                raise ResamplingContractError(
                    "split_indices_test does not match resolved_outer_split.",
                    code="resolved_split_indices_mismatch",
                    diagnostics={"partition": "test"},
                )
            train_idx_out = assignment.train_indices
            test_idx_out = assignment.test_indices

        if provided_train_weights:
            expected_train_weights = tuple(
                full_context.sample_weights[index]
                for index in outer_plan.primary.train_indices
            )
            expected_test_weights = tuple(
                full_context.sample_weights[index]
                for index in outer_plan.primary.test_indices
            )
            if tuple(provided_train_weights) != expected_train_weights:
                raise ResamplingContractError(
                    "sample_weight_train is not aligned with the resolved split context.",
                    code="sample_weight_context_mismatch",
                    diagnostics={"partition": "train", "n_rows": n_train},
                )
            if tuple(provided_test_weights) != expected_test_weights:
                raise ResamplingContractError(
                    "sample_weight_test is not aligned with the resolved split context.",
                    code="sample_weight_context_mismatch",
                    diagnostics={"partition": "test", "n_rows": n_test},
                )

        fit_context = full_context.take(
            outer_plan.primary.train_indices,
            parent_split_fingerprint=outer_plan.primary.fingerprint,
        )
        return (
            full_context,
            fit_context,
            outer_plan,
            tuple(train_idx_out),
            tuple(test_idx_out),
        )

    def _maqc_pairing_candidates(
        self,
        configured_methods: Sequence[str],
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Return deterministic MAQC candidate stacks including the baseline."""

        raw_sets = list(self.config.maqc_pairing_method_sets or ())
        raw_names = list(self.config.maqc_pairing_method_set_names or ())
        if len(raw_names) != len(raw_sets):
            raw_names = [f"candidate_{index}" for index in range(len(raw_sets))]
        active_names = [
            str(name).strip() or f"candidate_{index}"
            for index, (name, methods) in enumerate(zip(raw_names, raw_sets))
            if tuple(str(method) for method in methods)
        ]
        if len(set(active_names)) != len(active_names):
            raise NestedPairingEvaluationError(
                "nested_pairing_duplicate_candidate_name",
                "Nested MAQC pairing requires unique names for every configured selector candidate.",
                diagnostics={"candidate_names": list(active_names)},
            )
        raw_sets.append(tuple(str(method) for method in configured_methods))
        raw_names.append("configured_enabled_methods")

        deduped: List[Tuple[str, Tuple[str, ...]]] = []
        seen: Set[Tuple[str, ...]] = set()
        seen_names: Set[str] = set()
        for name, methods in zip(raw_names, raw_sets):
            methods_t = tuple(str(method) for method in methods)
            if not methods_t or methods_t in seen:
                continue
            candidate_name = str(name).strip() or f"candidate_{len(deduped)}"
            if candidate_name in seen_names:
                raise NestedPairingEvaluationError(
                    "nested_pairing_duplicate_candidate_name",
                    "Nested MAQC pairing requires unique names for every active selector candidate.",
                    diagnostics={"candidate_name": str(candidate_name)},
                )
            deduped.append((candidate_name, methods_t))
            seen.add(methods_t)
            seen_names.add(candidate_name)
        if not deduped:
            raise NestedPairingEvaluationError(
                "nested_pairing_no_candidates",
                "Nested MAQC pairing requires at least one non-empty selector stack.",
            )
        return tuple(deduped)

    def _nested_pairing_mode(self) -> str:
        """Return the validated opt-in pairing score mode."""

        mode = str(getattr(self.config, "maqc_pairing_score_mode", "raw_cv") or "raw_cv")
        return mode.strip().lower()

    def _nested_pairing_unsupported_reasons(
        self,
        *,
        external_feature_scores: Optional[np.ndarray],
    ) -> Tuple[str, ...]:
        """List configurations whose fold-local identity cannot yet be proven."""

        reasons: List[str] = []
        cls_cfg = self._classification_cfg()
        if bool(getattr(cls_cfg, "native_categorical_stage2_enabled", False)):
            reasons.append("native_categorical_stage2")
        if external_feature_scores is not None:
            reasons.append("external_feature_scores")
        if bool(getattr(self.config, "diakrino_prefilter_enabled", False)):
            reasons.append("diakrino_prefilter")
        if bool(getattr(self.config, "diakrino_conformal_selection_enabled", False)):
            reasons.append("diakrino_conformal_selection")
        if bool(getattr(self.config, "fs_use_diakrino_relevance_oracle", False)):
            reasons.append("diakrino_relevance_oracle")
        if bool(getattr(self.config, "use_diakrino_selector_prior", False)):
            reasons.append("diakrino_selector_prior")
        if bool(getattr(self.config, "diakrino_regime_conditional", False)):
            reasons.append("diakrino_regime_conditional")
        if str(getattr(self.config, "diakrino_sidecar_path", "") or "").strip():
            reasons.append("diakrino_sidecar")
        if str(
            getattr(self.config.dist_config, "diakrino_sidecar_path", "") or ""
        ).strip():
            reasons.append("diakrino_distribution_sidecar")
        if bool(getattr(self.config, "diakrino_cdf_trust_gate_enabled", False)):
            reasons.append("diakrino_cdf_trust_gate")
        if bool(getattr(self.config.dist_config, "diakrino_skip_fit_discrete_enabled", False)):
            reasons.append("diakrino_distribution_skip_fit")
        if float(
            getattr(self.config.dist_config, "diakrino_family_prior_lambda", 0.0) or 0.0
        ) > 0.0:
            reasons.append("diakrino_distribution_prior")
        pairing_methods = tuple(
            str(method)
            for method_set in (
                tuple(getattr(self.config, "maqc_pairing_method_sets", tuple()) or tuple())
                + (tuple(getattr(self.config, "enabled_methods", tuple()) or tuple()),)
            )
            for method in method_set
        )
        if any("diakrino" in method.strip().lower() for method in pairing_methods):
            reasons.append("diakrino_selector_method")
        classifier_candidates = {
            str(name).strip().lower()
            for name in tuple(getattr(cls_cfg, "model_candidates", tuple()) or tuple())
        }
        if (
            bool(getattr(cls_cfg, "include_tabpfn_model", False))
            or bool(getattr(cls_cfg, "include_tabentics_diakrino_model", False))
            or "tabpfn" in classifier_candidates
            or "tabentics_diakrino" in classifier_candidates
        ):
            reasons.append("diakrino_classifier")
        if bool(getattr(self.config, "auto_router_enabled", False)):
            reasons.append("auto_router")
        if str(
            getattr(self.config, "meta_learning_selector_mode", "none") or "none"
        ).strip().lower() != "none":
            reasons.append("meta_learning_selector")
        if str(getattr(self.config, "multiomics_adapter", "none") or "none").strip().lower() != "none":
            reasons.append("multiomics_adapter")
        if bool(getattr(self.config, "regime_gating_enabled", False)):
            reasons.append("regime_gating")
        if bool(getattr(self.config, "tier_lockout_enabled", False)):
            reasons.append("tier_lockout")
        if bool(getattr(self.config, "tier_routing_enabled", False)):
            reasons.append("tier_routing")
        if str(getattr(self.config, "tier_classifier_mode", "heuristic") or "heuristic").strip().lower() != "heuristic":
            reasons.append("learned_tier_router")
        if str(getattr(cls_cfg, "backend", "sklearn") or "sklearn").strip().lower() != "sklearn":
            reasons.append("classification_backend_not_sklearn")
        if str(getattr(cls_cfg, "selection_mode", "legacy") or "legacy").strip().lower() != "legacy":
            reasons.append("classification_selection_mode_not_legacy")
        return tuple(sorted(set(reasons)))

    def _prepare_nested_pairing_raw_context(
        self,
        *,
        X_train: Any,
        y_train: np.ndarray,
        schema: DatasetSchema | None,
        batch_labels: np.ndarray | None,
        fit_resampling_context: FitResamplingContext,
        dataset_name: str,
        seed: int,
        configured_methods: Sequence[str],
        external_feature_scores: Optional[np.ndarray],
        reentrancy_guard: bool,
    ) -> _NestedPairingRawContext | None:
        """Resolve raw-row outer folds before any learned typed transform runs."""

        if not bool(getattr(self.config, "enable_maqc_pairing", False)):
            return None
        mode = self._nested_pairing_mode()
        if mode == "raw_cv":
            return None
        if reentrancy_guard:
            raise NestedPairingEvaluationError(
                "nested_pairing_reentrancy",
                "Nested MAQC pairing cannot recursively invoke another nested pairing.",
            )

        unsupported = self._nested_pairing_unsupported_reasons(
            external_feature_scores=external_feature_scores,
        )
        if unsupported:
            raise NestedPairingEvaluationError(
                "nested_pairing_unsupported_composition",
                "Nested MAQC pairing cannot prove fold-local identity for the requested composition.",
                diagnostics={"unsupported": list(unsupported), "mode": mode},
            )
        if mode == "nested_bbc":
            if int(self.config.maqc_pairing_outer_repeats) != 1:
                raise NestedPairingEvaluationError(
                    "nested_bbc_repeats_unsupported",
                    "Nested BBC v1 requires maqc_pairing_outer_repeats=1 so each row has one OOF prediction.",
                    diagnostics={
                        "outer_repeats": int(self.config.maqc_pairing_outer_repeats)
                    },
                )
            if fit_resampling_context.policy.kind not in {"iid", "stratified"}:
                raise NestedPairingEvaluationError(
                    "nested_bbc_resampling_policy_unsupported",
                    "Nested BBC v1 supports only IID or stratified row bootstraps; use nested_cv for grouped, temporal, or supplied policies.",
                    diagnostics={"policy": fit_resampling_context.policy.kind},
                )

        classes, counts = np.unique(np.asarray(y_train).ravel(), return_counts=True)
        if classes.size < 2:
            raise NestedPairingEvaluationError(
                "nested_pairing_insufficient_classes",
                "Nested MAQC pairing requires at least two observed classes.",
            )
        if int(counts.min()) < int(self.config.maqc_pairing_outer_splits):
            raise NestedPairingEvaluationError(
                "nested_pairing_outer_stratification_impossible",
                "Nested MAQC pairing cannot allocate every class across the requested outer folds.",
                diagnostics={
                    "minimum_class_count": int(counts.min()),
                    "outer_splits": int(self.config.maqc_pairing_outer_splits),
                },
            )

        candidates = self._maqc_pairing_candidates(configured_methods)
        policy_kind = str(fit_resampling_context.policy.kind)
        outer_plan = self._resolve_inner_split_plan(
            fit_resampling_context,
            np.asarray(y_train).ravel(),
            purpose="maqc_pairing_nested_outer",
            n_splits=int(self.config.maqc_pairing_outer_splits),
            n_repeats=int(self.config.maqc_pairing_outer_repeats),
            seed=int(seed) + int(self.config.maqc_pairing_seed_stride),
            stratified=policy_kind in {"iid", "stratified", "stratified_group"},
            shuffle=policy_kind not in {"blocked_temporal", "supplied"},
        )
        expected_outer_folds = int(
            self.config.maqc_pairing_outer_splits
            * self.config.maqc_pairing_outer_repeats
        )
        if len(outer_plan.splits) != expected_outer_folds:
            raise NestedPairingEvaluationError(
                "nested_pairing_outer_fold_count",
                "Nested MAQC pairing requires the resolved outer plan to match its declared split and repeat count.",
                diagnostics={
                    "policy": str(policy_kind),
                    "expected_outer_folds": int(expected_outer_folds),
                    "resolved_outer_folds": int(len(outer_plan.splits)),
                },
            )
        split_fingerprints = [str(split.fingerprint) for split in outer_plan.splits]
        if len(set(split_fingerprints)) != len(split_fingerprints):
            raise NestedPairingEvaluationError(
                "nested_pairing_duplicate_outer_fold",
                "Nested MAQC pairing requires unique resolved outer-fold fingerprints.",
                diagnostics={"resolved_outer_folds": int(len(split_fingerprints))},
            )
        for split in outer_plan.splits:
            fold_y = np.asarray(y_train).ravel()[np.asarray(split.train_indices, dtype=int)]
            _, fold_counts = np.unique(fold_y, return_counts=True)
            if (
                fold_counts.size < classes.size
                or int(fold_counts.min())
                < int(self.config.maqc_pairing_min_train_per_class)
            ):
                raise NestedPairingEvaluationError(
                    "nested_pairing_fold_train_class_support",
                    "A nested outer-train fold lacks the configured minimum per-class support.",
                    diagnostics={
                        "split_fingerprint": split.fingerprint,
                        "minimum_train_per_class": int(
                            self.config.maqc_pairing_min_train_per_class
                        ),
                    },
                )

        planned = int(len(candidates) * len(outer_plan.splits))
        cap = int(self.config.maqc_pairing_max_outer_evaluations)
        if cap > 0 and planned > cap:
            raise NestedPairingEvaluationError(
                "nested_pairing_evaluation_cap",
                "Nested MAQC pairing would exceed its explicit outer-evaluation cap.",
                diagnostics={"planned": planned, "cap": cap},
            )
        return _NestedPairingRawContext(
            X_train=X_train,
            y_train=np.asarray(y_train).ravel().copy(),
            schema=schema,
            batch_labels=(
                None if batch_labels is None else np.asarray(batch_labels, dtype=object).copy()
            ),
            fit_resampling_context=fit_resampling_context,
            outer_plan=outer_plan,
            dataset_name=str(dataset_name),
            seed=int(seed),
        )

    @staticmethod
    def _nested_pairing_fold_seed(
        *,
        seed: int,
        seed_stride: int,
        candidate_index: int,
        fold_index: int,
    ) -> int:
        """Derive a deterministic, independent stream for one candidate/fold."""

        stream = np.random.SeedSequence(
            [
                int(seed),
                int(seed_stride),
                int(candidate_index),
                int(fold_index),
            ]
        )
        return int(stream.generate_state(1, dtype=np.uint32)[0])

    def _nested_pairing_check_runtime(
        self,
        *,
        started_at: float,
        phase: str,
        planned_evaluations: int,
        completed_evaluations: int,
    ) -> float:
        """Enforce the explicit nested-evaluation wall-clock contract."""

        elapsed = float(max(0.0, self._timer() - float(started_at)))
        cap = float(getattr(self.config, "maqc_pairing_max_runtime_seconds", 0.0) or 0.0)
        if cap > 0.0 and elapsed > cap:
            raise NestedPairingEvaluationError(
                "nested_pairing_runtime_cap",
                "Nested MAQC pairing exceeded its explicit runtime cap; the entire evaluation must be rerun.",
                diagnostics={
                    "phase": str(phase),
                    "elapsed_seconds": float(elapsed),
                    "cap_seconds": float(cap),
                    "planned_evaluations": int(planned_evaluations),
                    "completed_evaluations": int(completed_evaluations),
                    "resume_policy": "rerun_all_atomic",
                },
            )
        return elapsed

    def _capture_nested_pairing_fold(
        self,
        *,
        candidate_name: str,
        fold_index: int,
        raw_context: _NestedPairingRawContext,
        split: Any,
        resolved_fold_plan: ResolvedSplitPlan,
        result: PipelineRunResult,
        prediction_capture: Mapping[str, Any],
    ) -> _NestedPairingFoldCapture:
        """Validate and freeze one private clone prediction capture."""

        if not isinstance(prediction_capture, Mapping):
            raise NestedPairingEvaluationError(
                "nested_pairing_prediction_capture_missing",
                "Nested evaluation did not provide a private prediction capture.",
                diagnostics={
                    "candidate_name": str(candidate_name),
                    "fold_index": int(fold_index),
                },
            )
        capture = dict(prediction_capture)
        expected_test = tuple(int(value) for value in split.test_indices)
        expected_train = tuple(int(value) for value in split.train_indices)
        if tuple(int(value) for value in result.split_indices_train) != expected_train:
            raise NestedPairingEvaluationError(
                "nested_pairing_train_assignment_mismatch",
                "A nested clone did not honor its resolved outer-train assignment.",
                diagnostics={
                    "candidate_name": str(candidate_name),
                    "fold_index": int(fold_index),
                },
            )
        if tuple(int(value) for value in result.split_indices_test) != expected_test:
            raise NestedPairingEvaluationError(
                "nested_pairing_test_assignment_mismatch",
                "A nested clone did not honor its resolved outer-test assignment.",
                diagnostics={
                    "candidate_name": str(candidate_name),
                    "fold_index": int(fold_index),
                },
            )
        if str(result.outer_split_fingerprint) != str(
            resolved_fold_plan.primary.fingerprint
        ):
            raise NestedPairingEvaluationError(
                "nested_pairing_split_fingerprint_mismatch",
                "A nested clone returned a different resolved split fingerprint.",
                diagnostics={
                    "candidate_name": str(candidate_name),
                    "fold_index": int(fold_index),
                    "expected_split_fingerprint": str(
                        resolved_fold_plan.primary.fingerprint
                    ),
                    "observed_split_fingerprint": str(result.outer_split_fingerprint),
                },
            )
        if not bool(dict(result.leakage_audit or {}).get("ok", False)):
            raise NestedPairingEvaluationError(
                "nested_pairing_clone_leakage_audit_failed",
                "A nested clone failed its outer-fold leakage audit.",
                diagnostics={
                    "candidate_name": str(candidate_name),
                    "fold_index": int(fold_index),
                    "leakage_reason": str(
                        dict(result.leakage_audit or {}).get("reason", "unknown")
                    ),
                },
            )

        y_true = tuple(capture.get("y_true", tuple()) or tuple())
        y_pred = tuple(capture.get("y_pred", tuple()) or tuple())
        weights = tuple(
            float(value)
            for value in (capture.get("sample_weights", tuple()) or tuple())
        )
        if not (len(y_true) == len(y_pred) == len(weights) == len(expected_test)):
            raise NestedPairingEvaluationError(
                "nested_pairing_prediction_capture_alignment",
                "A nested clone returned prediction vectors that are not aligned to its test fold.",
                diagnostics={
                    "candidate_name": str(candidate_name),
                    "fold_index": int(fold_index),
                    "expected_rows": int(len(expected_test)),
                    "y_true_rows": int(len(y_true)),
                    "y_pred_rows": int(len(y_pred)),
                    "sample_weight_rows": int(len(weights)),
                },
            )

        expected_y = tuple(
            np.asarray(raw_context.y_train, dtype=object)
            .ravel()[np.asarray(expected_test, dtype=int)]
            .tolist()
        )
        if tuple(typed_scalar_key(value) for value in y_true) != tuple(
            typed_scalar_key(value) for value in expected_y
        ):
            raise NestedPairingEvaluationError(
                "nested_pairing_prediction_label_mismatch",
                "A nested clone prediction capture is not aligned to the raw outer-fold labels.",
                diagnostics={
                    "candidate_name": str(candidate_name),
                    "fold_index": int(fold_index),
                    "n_rows": int(len(expected_test)),
                },
            )
        expected_weights = (
            tuple(1.0 for _ in expected_test)
            if not raw_context.fit_resampling_context.sample_weights
            else tuple(
                float(raw_context.fit_resampling_context.sample_weights[index])
                for index in expected_test
            )
        )
        if not np.allclose(
            np.asarray(weights, dtype=float),
            np.asarray(expected_weights, dtype=float),
            equal_nan=False,
        ):
            raise NestedPairingEvaluationError(
                "nested_pairing_prediction_weight_mismatch",
                "A nested clone prediction capture is not aligned to the declared test weights.",
                diagnostics={
                    "candidate_name": str(candidate_name),
                    "fold_index": int(fold_index),
                    "n_rows": int(len(expected_test)),
                },
            )
        recomputed_score = self._safe_balanced_accuracy(
            np.asarray(y_true),
            np.asarray(y_pred),
            sample_weight=np.asarray(weights, dtype=float),
        )
        if not np.isfinite(recomputed_score) or not np.isclose(
            float(result.balanced_accuracy),
            float(recomputed_score),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise NestedPairingEvaluationError(
                "nested_pairing_prediction_score_mismatch",
                "A nested clone score does not match its private row-level prediction capture.",
                diagnostics={
                    "candidate_name": str(candidate_name),
                    "fold_index": int(fold_index),
                    "reported_balanced_accuracy": float(result.balanced_accuracy),
                    "recomputed_balanced_accuracy": float(recomputed_score),
                },
            )
        test_context = raw_context.fit_resampling_context.take(
            expected_test,
            parent_split_fingerprint=resolved_fold_plan.primary.fingerprint,
        )
        return _NestedPairingFoldCapture(
            candidate_name=str(candidate_name),
            fold_index=int(fold_index),
            split_fingerprint=str(resolved_fold_plan.primary.fingerprint),
            test_indices=expected_test,
            test_row_ids_fingerprint=str(test_context.row_ids_fingerprint),
            y_true=y_true,
            y_pred=y_pred,
            sample_weights=weights,
            balanced_accuracy=float(recomputed_score),
            leakage_audit=_json_safe(dict(result.leakage_audit or {})),
        )

    def _select_nested_pairing_from_fold_scores(
        self,
        *,
        candidates: Sequence[Tuple[str, Tuple[str, ...]]],
        scores_by_candidate: Mapping[str, np.ndarray],
        configured_name: str,
    ) -> Dict[str, Any]:
        """Apply the single paired outer-fold selection and reversion rule."""

        ordered_names = tuple(str(name) for name, _ in candidates)
        if not ordered_names or str(configured_name) not in scores_by_candidate:
            raise NestedPairingEvaluationError(
                "nested_pairing_selection_inputs_invalid",
                "Nested pairing selection requires a configured candidate and complete score matrix.",
            )
        expected_folds: Optional[int] = None
        for name in ordered_names:
            scores = np.asarray(scores_by_candidate.get(name, tuple()), dtype=float)
            if scores.ndim != 1 or scores.size < 2 or not bool(np.all(np.isfinite(scores))):
                raise NestedPairingEvaluationError(
                    "nested_pairing_selection_score_coverage",
                    "Nested pairing selection requires at least two finite paired scores per candidate.",
                    diagnostics={"candidate_name": str(name)},
                )
            if expected_folds is None:
                expected_folds = int(scores.size)
            elif int(scores.size) != expected_folds:
                raise NestedPairingEvaluationError(
                    "nested_pairing_selection_fold_mismatch",
                    "Nested pairing candidates must be compared on identical outer-fold coverage.",
                    diagnostics={
                        "candidate_name": str(name),
                        "expected_folds": int(expected_folds),
                        "observed_folds": int(scores.size),
                    },
                )

        raw_best_name = ordered_names[0]
        raw_best_scores = np.asarray(scores_by_candidate[raw_best_name], dtype=float)
        raw_best_mean = float(np.mean(raw_best_scores))
        for name in ordered_names[1:]:
            candidate_scores = np.asarray(scores_by_candidate[name], dtype=float)
            candidate_mean = float(np.mean(candidate_scores))
            if candidate_mean > raw_best_mean:
                raw_best_name = str(name)
                raw_best_scores = candidate_scores
                raw_best_mean = float(candidate_mean)

        base_scores = np.asarray(scores_by_candidate[str(configured_name)], dtype=float)
        selected_name = str(raw_best_name)
        reverted = False
        revert_reason = ""
        improvement = float("nan")
        paired_se = float("nan")
        abs_thr = float(max(0.0, self.config.maqc_pairing_min_improvement))
        se_mult = float(max(0.0, self.config.maqc_pairing_min_improvement_se_mult))
        if str(raw_best_name) != str(configured_name):
            differences = np.asarray(raw_best_scores - base_scores, dtype=float)
            if not bool(np.all(np.isfinite(differences))):
                raise NestedPairingEvaluationError(
                    "nested_pairing_unpaired_score_coverage",
                    "Nested candidate comparisons require complete paired outer-fold scores.",
                    diagnostics={
                        "best_candidate": str(raw_best_name),
                        "configured_candidate": str(configured_name),
                    },
                )
            improvement = float(np.mean(differences))
            paired_se = float(
                np.std(differences, ddof=1) / np.sqrt(float(differences.size))
            )
            below_abs = bool(abs_thr > 0.0 and improvement < abs_thr)
            below_se = bool(
                se_mult > 0.0 and improvement < float(se_mult * paired_se)
            )
            if below_abs or below_se:
                selected_name = str(configured_name)
                reverted = True
                revert_reason = "|".join(
                    value
                    for value, active in (
                        ("below_min_improvement", below_abs),
                        ("below_min_improvement_se", below_se),
                    )
                    if active
                )
        selected_scores = np.asarray(scores_by_candidate[selected_name], dtype=float)
        return {
            "raw_best_name": str(raw_best_name),
            "raw_best_scores": raw_best_scores,
            "configured_scores": base_scores,
            "selected_name": str(selected_name),
            "selected_scores": selected_scores,
            "reverted": bool(reverted),
            "revert_reason": str(revert_reason),
            "improvement": float(improvement),
            "paired_se": float(paired_se),
            "min_improvement": float(abs_thr),
            "min_improvement_se_mult": float(se_mult),
        }

    def _nested_pairing_bbc_metadata(
        self,
        *,
        raw_context: _NestedPairingRawContext,
        candidates: Sequence[Tuple[str, Tuple[str, ...]]],
        captures_by_candidate: Mapping[str, Sequence[_NestedPairingFoldCapture]],
        configured_candidate_name: str,
        selected_candidate_name: str,
        started_at: float,
        planned_evaluations: int,
        completed_evaluations: int,
    ) -> Dict[str, Any]:
        """Estimate the deployed paired-fold policy without persisting OOF rows."""

        n_rows = int(np.asarray(raw_context.y_train).ravel().size)
        if n_rows < 2:
            raise NestedPairingEvaluationError(
                "nested_bbc_insufficient_oof_rows",
                "Nested BBC requires at least two row-level OOF observations.",
                diagnostics={"n_rows": int(n_rows)},
            )
        candidate_names = tuple(str(name) for name, _ in candidates)
        outer_splits = tuple(raw_context.outer_plan.splits)
        if len(outer_splits) < 2:
            raise NestedPairingEvaluationError(
                "nested_bbc_insufficient_outer_folds",
                "Nested BBC requires at least two paired outer folds.",
                diagnostics={"outer_folds": int(len(outer_splits))},
            )
        if str(configured_candidate_name) not in candidate_names:
            raise NestedPairingEvaluationError(
                "nested_bbc_configured_candidate_missing",
                "Nested BBC cannot reconstruct the deployed configured candidate.",
                diagnostics={
                    "configured_candidate": str(configured_candidate_name)
                },
            )
        expected_y = tuple(np.asarray(raw_context.y_train).ravel().tolist())
        expected_y_array = np.asarray(expected_y)
        expected_weights = np.asarray(
            (
                tuple(1.0 for _ in range(n_rows))
                if not raw_context.fit_resampling_context.sample_weights
                else raw_context.fit_resampling_context.sample_weights
            ),
            dtype=float,
        )
        if expected_weights.size != n_rows:
            raise NestedPairingEvaluationError(
                "nested_bbc_weight_coverage_invalid",
                "Nested BBC requires one declared evaluation weight per OOF row.",
                diagnostics={
                    "n_rows": int(n_rows),
                    "weight_rows": int(expected_weights.size),
                },
            )
        fold_positions: List[np.ndarray] = []
        for fold_index, split in enumerate(outer_splits):
            positions = np.asarray(split.test_indices, dtype=int)
            if (
                positions.size == 0
                or np.any(positions < 0)
                or np.any(positions >= n_rows)
            ):
                raise NestedPairingEvaluationError(
                    "nested_bbc_oof_index_invalid",
                    "Nested BBC received an invalid outer-fold OOF row position.",
                    diagnostics={"fold_index": int(fold_index)},
                )
            fold_positions.append(positions)
        predictions: Dict[str, np.ndarray] = {
            name: np.empty(n_rows, dtype=object) for name in candidate_names
        }
        seen_by_candidate: Dict[str, np.ndarray] = {
            name: np.zeros(n_rows, dtype=bool) for name in candidate_names
        }
        captures_by_candidate_fold: Dict[
            str, Dict[int, _NestedPairingFoldCapture]
        ] = {}

        for name in candidate_names:
            captures = tuple(captures_by_candidate.get(name, tuple()) or tuple())
            if len(captures) != len(outer_splits):
                raise NestedPairingEvaluationError(
                    "nested_bbc_candidate_fold_coverage",
                    "Nested BBC requires complete candidate/fold OOF coverage.",
                    diagnostics={
                        "candidate_name": str(name),
                        "expected_folds": int(len(outer_splits)),
                        "observed_folds": int(len(captures)),
                    },
                )
            captures_for_fold: Dict[int, _NestedPairingFoldCapture] = {}
            for capture in captures:
                fold_index = int(capture.fold_index)
                if fold_index < 0 or fold_index >= len(outer_splits):
                    raise NestedPairingEvaluationError(
                        "nested_bbc_fold_index_invalid",
                        "Nested BBC received an invalid candidate fold index.",
                        diagnostics={
                            "candidate_name": str(name),
                            "fold_index": int(fold_index),
                        },
                    )
                if fold_index in captures_for_fold:
                    raise NestedPairingEvaluationError(
                        "nested_bbc_duplicate_candidate_fold",
                        "Nested BBC requires one capture per candidate and outer fold.",
                        diagnostics={
                            "candidate_name": str(name),
                            "fold_index": int(fold_index),
                        },
                    )
                expected_split = outer_splits[fold_index]
                positions = fold_positions[fold_index]
                if (
                    str(capture.split_fingerprint) != str(expected_split.fingerprint)
                    or tuple(int(value) for value in capture.test_indices)
                    != tuple(int(value) for value in positions.tolist())
                ):
                    raise NestedPairingEvaluationError(
                        "nested_bbc_fold_provenance_mismatch",
                        "Nested BBC candidate captures must retain their original outer-fold membership.",
                        diagnostics={
                            "candidate_name": str(name),
                            "fold_index": int(fold_index),
                        },
                    )
                if bool(np.any(seen_by_candidate[name][positions])):
                    raise NestedPairingEvaluationError(
                        "nested_bbc_oof_duplicate_row",
                        "Nested BBC v1 requires exactly one OOF prediction per row.",
                        diagnostics={
                            "candidate_name": str(name),
                            "fold_index": int(fold_index),
                        },
                    )
                capture_y = tuple(capture.y_true)
                expected_fold_y = tuple(expected_y[index] for index in positions.tolist())
                if tuple(typed_scalar_key(value) for value in capture_y) != tuple(
                    typed_scalar_key(value) for value in expected_fold_y
                ):
                    raise NestedPairingEvaluationError(
                        "nested_bbc_oof_label_alignment",
                        "Nested BBC candidate captures disagree on raw row labels.",
                        diagnostics={
                            "candidate_name": str(name),
                            "fold_index": int(fold_index),
                        },
                    )
                if not np.allclose(
                    np.asarray(capture.sample_weights, dtype=float),
                    expected_weights[positions],
                    equal_nan=False,
                ):
                    raise NestedPairingEvaluationError(
                        "nested_bbc_oof_weight_alignment",
                        "Nested BBC candidate captures disagree on declared row weights.",
                        diagnostics={
                            "candidate_name": str(name),
                            "fold_index": int(fold_index),
                        },
                    )
                predictions[name][positions] = list(capture.y_pred)
                seen_by_candidate[name][positions] = True
                captures_for_fold[fold_index] = capture
            if not bool(np.all(seen_by_candidate[name])):
                raise NestedPairingEvaluationError(
                    "nested_bbc_oof_incomplete_coverage",
                    "Nested BBC v1 requires every outer-train row to have one OOF prediction per candidate.",
                    diagnostics={
                        "candidate_name": str(name),
                        "covered_rows": int(np.sum(seen_by_candidate[name])),
                        "expected_rows": int(n_rows),
                    },
                )
            if len(captures_for_fold) != len(outer_splits):
                raise NestedPairingEvaluationError(
                    "nested_bbc_candidate_fold_coverage",
                    "Nested BBC requires a capture for every candidate and outer fold.",
                    diagnostics={
                        "candidate_name": str(name),
                        "expected_folds": int(len(outer_splits)),
                        "observed_folds": int(len(captures_for_fold)),
                    },
                )
            captures_by_candidate_fold[name] = captures_for_fold

        label_keys = np.empty(n_rows, dtype=object)
        label_keys[:] = [typed_scalar_key(value) for value in expected_y]
        class_keys = tuple(sorted(set(label_keys.tolist())))
        if len(class_keys) < 2:
            raise NestedPairingEvaluationError(
                "nested_bbc_insufficient_classes",
                "Nested BBC requires at least two classes in its OOF row universe.",
                diagnostics={"n_classes": int(len(class_keys))},
            )
        for key in class_keys:
            class_mask = np.asarray(
                [value == key for value in label_keys.tolist()], dtype=bool
            )
            if not float(np.sum(expected_weights[class_mask])) > 0.0:
                raise NestedPairingEvaluationError(
                    "nested_bbc_zero_class_weight",
                    "Nested BBC cannot score a class with zero declared weight mass.",
                    diagnostics={"n_classes": int(len(class_keys))},
                )

        prediction_arrays = {
            name: np.asarray(predictions[name].tolist()) for name in candidate_names
        }
        if str(selected_candidate_name) not in prediction_arrays:
            raise NestedPairingEvaluationError(
                "nested_bbc_selected_candidate_missing",
                "Nested BBC cannot find the candidate selected from nested outer-fold evidence.",
                diagnostics={"selected_candidate": str(selected_candidate_name)},
            )
        deployed_reference_scores = np.asarray(
            [
                float(
                    captures_by_candidate_fold[str(selected_candidate_name)][
                        fold_index
                    ].balanced_accuracy
                )
                for fold_index in range(len(outer_splits))
            ],
            dtype=float,
        )
        if not bool(np.all(np.isfinite(deployed_reference_scores))):
            raise NestedPairingEvaluationError(
                "nested_bbc_fold_reference_score_invalid",
                "Nested BBC could not compute the deployed selected candidate's fold-mean reference.",
                diagnostics={"selected_candidate": str(selected_candidate_name)},
            )
        deployed_reference_score = float(np.mean(deployed_reference_scores))

        def class_supported(indices: np.ndarray) -> bool:
            if indices.size == 0:
                return False
            for key in class_keys:
                mask = np.asarray(
                    [value == key for value in label_keys[indices].tolist()],
                    dtype=bool,
                )
                if not bool(np.any(mask)):
                    return False
                if not float(np.sum(expected_weights[indices][mask])) > 0.0:
                    return False
            return True

        rounds = int(self.config.maqc_pairing_bbc_bootstrap_rounds)
        max_attempts = int(max(rounds, 10 * rounds))
        seed_stream = np.random.SeedSequence(
            [
                int(raw_context.seed),
                int(self.config.maqc_pairing_seed_stride),
                3004,
            ]
        )
        bootstrap_seed = int(seed_stream.generate_state(1, dtype=np.uint32)[0])
        rng = np.random.default_rng(bootstrap_seed)
        oob_scores: List[float] = []
        winner_counts = {name: 0 for name in candidate_names}
        raw_best_counts = {name: 0 for name in candidate_names}
        reverted_draws = 0
        revert_reason_counts: Dict[str, int] = {}
        rejected = {
            "missing_inbag_class": 0,
            "missing_oob_class": 0,
            "invalid_candidate_score": 0,
        }
        oob_seen = np.zeros(n_rows, dtype=bool)
        oob_row_draws = 0
        attempted = 0
        while len(oob_scores) < rounds and attempted < max_attempts:
            attempted += 1
            self._nested_pairing_check_runtime(
                started_at=started_at,
                phase="nested_bbc_bootstrap",
                planned_evaluations=planned_evaluations,
                completed_evaluations=completed_evaluations,
            )
            inbag_by_fold: List[np.ndarray] = []
            oob_by_fold: List[np.ndarray] = []
            rejected_draw = False
            for positions in fold_positions:
                inbag_local = np.asarray(
                    rng.integers(0, positions.size, size=positions.size), dtype=int
                )
                oob_mask = np.ones(positions.size, dtype=bool)
                oob_mask[inbag_local] = False
                inbag = positions[inbag_local]
                oob = positions[np.flatnonzero(oob_mask)]
                if not class_supported(inbag):
                    rejected["missing_inbag_class"] += 1
                    rejected_draw = True
                    break
                if not class_supported(oob):
                    rejected["missing_oob_class"] += 1
                    rejected_draw = True
                    break
                inbag_by_fold.append(inbag)
                oob_by_fold.append(oob)
            if rejected_draw:
                continue

            inbag_scores_by_candidate: Dict[str, np.ndarray] = {}
            for name in candidate_names:
                scores = np.asarray(
                    [
                        self._safe_balanced_accuracy(
                            expected_y_array[inbag],
                            prediction_arrays[name][inbag],
                            sample_weight=expected_weights[inbag],
                        )
                        for inbag in inbag_by_fold
                    ],
                    dtype=float,
                )
                if not bool(np.all(np.isfinite(scores))):
                    rejected["invalid_candidate_score"] += 1
                    rejected_draw = True
                    break
                inbag_scores_by_candidate[name] = scores
            if rejected_draw:
                continue
            try:
                bootstrap_selection = self._select_nested_pairing_from_fold_scores(
                    candidates=candidates,
                    scores_by_candidate=inbag_scores_by_candidate,
                    configured_name=str(configured_candidate_name),
                )
            except NestedPairingEvaluationError:
                rejected["invalid_candidate_score"] += 1
                continue
            winner_name = str(bootstrap_selection["selected_name"])
            oob_fold_scores = np.asarray(
                [
                    self._safe_balanced_accuracy(
                        expected_y_array[oob],
                        prediction_arrays[winner_name][oob],
                        sample_weight=expected_weights[oob],
                    )
                    for oob in oob_by_fold
                ],
                dtype=float,
            )
            if not bool(np.all(np.isfinite(oob_fold_scores))):
                rejected["invalid_candidate_score"] += 1
                continue
            oob_scores.append(float(np.mean(oob_fold_scores)))
            winner_counts[winner_name] += 1
            raw_best_name = str(bootstrap_selection["raw_best_name"])
            raw_best_counts[raw_best_name] += 1
            if bool(bootstrap_selection["reverted"]):
                reverted_draws += 1
                reason = str(bootstrap_selection["revert_reason"])
                revert_reason_counts[reason] = int(
                    revert_reason_counts.get(reason, 0) + 1
                )
            for oob in oob_by_fold:
                oob_seen[oob] = True
                oob_row_draws += int(oob.size)

        if len(oob_scores) != rounds:
            raise NestedPairingEvaluationError(
                "nested_bbc_valid_draws_incomplete",
                "Nested BBC could not produce the predeclared number of valid paired OOB draws.",
                diagnostics={
                    "requested_valid_draws": int(rounds),
                    "valid_draws": int(len(oob_scores)),
                    "attempted_draws": int(attempted),
                    "max_attempts": int(max_attempts),
                    "rejected_draws": dict(rejected),
                },
            )
        score_array = np.asarray(oob_scores, dtype=float)
        alpha_low = float((1.0 - float(self.config.maqc_pairing_bbc_ci_level)) / 2.0)
        alpha_high = float(1.0 - alpha_low)
        corrected = float(np.mean(score_array))
        return {
            "maqc_pairing_bbc_score_space": "bbc_oob_fold_mean",
            "maqc_pairing_bbc_bootstrap_unit": "within_outer_fold_rows",
            "maqc_pairing_bbc_selection_rule": "paired_outer_fold_mean_with_reversion",
            "maqc_pairing_bbc_bootstrap_seed": int(bootstrap_seed),
            "maqc_pairing_bbc_bootstrap_rounds": int(rounds),
            "maqc_pairing_bbc_ci_level": float(self.config.maqc_pairing_bbc_ci_level),
            "maqc_pairing_bbc_corrected_score": float(corrected),
            "maqc_pairing_bbc_reference_score_space": "nested_oof_fold_mean",
            "maqc_pairing_bbc_reference_nested_oof_fold_mean": float(
                deployed_reference_score
            ),
            "maqc_pairing_bbc_correction": float(
                corrected - deployed_reference_score
            ),
            "maqc_pairing_bbc_ci_low": float(np.quantile(score_array, alpha_low)),
            "maqc_pairing_bbc_ci_high": float(np.quantile(score_array, alpha_high)),
            "maqc_pairing_bbc_oob_score_std": float(
                np.std(score_array, ddof=1) if score_array.size > 1 else 0.0
            ),
            "maqc_pairing_bbc_attempted_draws": int(attempted),
            "maqc_pairing_bbc_valid_draws": int(score_array.size),
            "maqc_pairing_bbc_rejected_draws": dict(rejected),
            "maqc_pairing_bbc_max_attempts": int(max_attempts),
            "maqc_pairing_bbc_oob_unique_rows": int(np.sum(oob_seen)),
            "maqc_pairing_bbc_oob_row_draws": int(oob_row_draws),
            "maqc_pairing_bbc_oob_fold_count": int(len(outer_splits)),
            "maqc_pairing_bbc_selection_frequency": {
                name: float(winner_counts[name] / float(score_array.size))
                for name in candidate_names
            },
            "maqc_pairing_bbc_raw_best_frequency": {
                name: float(raw_best_counts[name] / float(score_array.size))
                for name in candidate_names
            },
            "maqc_pairing_bbc_reverted_draws": int(reverted_draws),
            "maqc_pairing_bbc_revert_reason_frequency": {
                reason: float(count / float(score_array.size))
                for reason, count in sorted(revert_reason_counts.items())
            },
        }

    def _run_nested_pairing_evaluation(
        self,
        *,
        raw_context: _NestedPairingRawContext,
        configured_methods: Sequence[str],
    ) -> Dict[str, Any]:
        """Run every candidate/fold clone and select only from paired outer evidence."""

        mode = self._nested_pairing_mode()
        if mode not in {"nested_cv", "nested_bbc"}:
            raise NestedPairingEvaluationError(
                "nested_pairing_mode_not_active",
                "Nested pairing evaluation was invoked outside a nested score mode.",
                diagnostics={"mode": str(mode)},
            )
        candidates = self._maqc_pairing_candidates(configured_methods)
        configured_methods_t = tuple(str(method) for method in configured_methods)
        configured_name = next(
            (
                str(name)
                for name, methods in candidates
                if tuple(methods) == configured_methods_t
            ),
            None,
        )
        if configured_name is None:
            raise NestedPairingEvaluationError(
                "nested_pairing_configured_candidate_missing",
                "Nested MAQC pairing could not retain the configured selector stack as its baseline.",
            )

        planned = int(len(candidates) * len(raw_context.outer_plan.splits))
        cap = int(self.config.maqc_pairing_max_outer_evaluations)
        if cap > 0 and planned > cap:
            raise NestedPairingEvaluationError(
                "nested_pairing_evaluation_cap",
                "Nested MAQC pairing would exceed its explicit outer-evaluation cap.",
                diagnostics={"planned": int(planned), "cap": int(cap)},
            )

        started_at = self._timer()
        captures_by_candidate: Dict[str, List[_NestedPairingFoldCapture]] = {
            str(name): [] for name, _ in candidates
        }
        completed = 0
        raw_y = np.asarray(raw_context.y_train).ravel()
        raw_weights = raw_context.fit_resampling_context.sample_weights
        for candidate_index, (candidate_name, methods) in enumerate(candidates):
            for fold_index, split in enumerate(raw_context.outer_plan.splits):
                self._nested_pairing_check_runtime(
                    started_at=started_at,
                    phase="nested_outer_fold_start",
                    planned_evaluations=planned,
                    completed_evaluations=completed,
                )
                fold_plan = resolve_assignment(
                    raw_context.fit_resampling_context,
                    split.assignment,
                    y=raw_y,
                    purpose=f"maqc_pairing_nested_outer_fold_{fold_index}",
                )
                if str(fold_plan.primary.fingerprint) != str(split.fingerprint):
                    raise NestedPairingEvaluationError(
                        "nested_pairing_resolved_fold_mismatch",
                        "The nested fold could not be rebound to its raw-row resolved assignment.",
                        diagnostics={
                            "candidate_name": str(candidate_name),
                            "fold_index": int(fold_index),
                            "expected_split_fingerprint": str(split.fingerprint),
                            "observed_split_fingerprint": str(fold_plan.primary.fingerprint),
                        },
                    )
                train_indices = np.asarray(split.train_indices, dtype=int)
                test_indices = np.asarray(split.test_indices, dtype=int)
                fold_config = copy.deepcopy(self.config)
                fold_config.enable_maqc_pairing = False
                fold_config.maqc_pairing_score_mode = "raw_cv"
                fold_config.maqc_pairing_method_sets = tuple()
                fold_config.maqc_pairing_method_set_names = tuple()
                fold_config.enabled_methods = tuple(str(method) for method in methods)
                fold_seed = self._nested_pairing_fold_seed(
                    seed=int(raw_context.seed),
                    seed_stride=int(self.config.maqc_pairing_seed_stride),
                    candidate_index=int(candidate_index),
                    fold_index=int(fold_index),
                )
                fold_pipeline = DistributionFeatureSelectionPipeline(fold_config)
                fold_prediction_captures: List[Mapping[str, Any]] = []

                def capture_fold_predictions(payload: Mapping[str, Any]) -> None:
                    if not isinstance(payload, Mapping):
                        raise NestedPairingEvaluationError(
                            "nested_pairing_prediction_capture_invalid",
                            "The nested clone emitted a non-mapping prediction capture.",
                            diagnostics={
                                "candidate_name": str(candidate_name),
                                "fold_index": int(fold_index),
                            },
                        )
                    fold_prediction_captures.append(dict(payload))

                try:
                    fold_result = fold_pipeline.run_pre_split(
                        self._take_input_rows(raw_context.X_train, train_indices),
                        raw_y[train_indices],
                        self._take_input_rows(raw_context.X_train, test_indices),
                        raw_y[test_indices],
                        dataset_name=str(raw_context.dataset_name),
                        seed=int(fold_seed),
                        split_indices_train=tuple(int(value) for value in train_indices.tolist()),
                        split_indices_test=tuple(int(value) for value in test_indices.tolist()),
                        batch_labels_train=(
                            None
                            if raw_context.batch_labels is None
                            else raw_context.batch_labels[train_indices]
                        ),
                        batch_labels_test=(
                            None
                            if raw_context.batch_labels is None
                            else raw_context.batch_labels[test_indices]
                        ),
                        sample_weight_train=(
                            None
                            if not raw_weights
                            else tuple(raw_weights[index] for index in train_indices.tolist())
                        ),
                        sample_weight_test=(
                            None
                            if not raw_weights
                            else tuple(raw_weights[index] for index in test_indices.tolist())
                        ),
                        schema=raw_context.schema,
                        resampling_context=raw_context.fit_resampling_context,
                        resolved_outer_split=fold_plan,
                        _capture_evaluation_predictions=True,
                        _evaluation_prediction_sink=capture_fold_predictions,
                        _nested_pairing_reentrancy=True,
                    )
                except NestedPairingEvaluationError:
                    raise
                except Exception as exc:
                    raise NestedPairingEvaluationError(
                        "nested_pairing_candidate_fold_failed",
                        "A nested selector candidate/fold failed; partial evidence cannot be promoted.",
                        diagnostics={
                            "candidate_name": str(candidate_name),
                            "fold_index": int(fold_index),
                            "exception_type": type(exc).__name__,
                        },
                    ) from exc
                if len(fold_prediction_captures) != 1:
                    raise NestedPairingEvaluationError(
                        "nested_pairing_prediction_capture_count",
                        "Nested evaluation requires exactly one in-memory prediction capture per candidate/fold.",
                        diagnostics={
                            "candidate_name": str(candidate_name),
                            "fold_index": int(fold_index),
                            "capture_count": int(len(fold_prediction_captures)),
                        },
                    )
                captures_by_candidate[str(candidate_name)].append(
                    self._capture_nested_pairing_fold(
                        candidate_name=str(candidate_name),
                        fold_index=int(fold_index),
                        raw_context=raw_context,
                        split=split,
                        resolved_fold_plan=fold_plan,
                        result=fold_result,
                        prediction_capture=fold_prediction_captures[0],
                    )
                )
                completed += 1
                self._nested_pairing_check_runtime(
                    started_at=started_at,
                    phase="nested_outer_fold_complete",
                    planned_evaluations=planned,
                    completed_evaluations=completed,
                )

        scores_by_candidate: Dict[str, np.ndarray] = {}
        summaries: List[Dict[str, Any]] = []
        for candidate_name, _ in candidates:
            captures = tuple(captures_by_candidate[str(candidate_name)])
            if len(captures) != len(raw_context.outer_plan.splits):
                raise NestedPairingEvaluationError(
                    "nested_pairing_candidate_fold_coverage",
                    "Nested pairing requires complete coverage for every candidate and outer fold.",
                    diagnostics={
                        "candidate_name": str(candidate_name),
                        "expected_folds": int(len(raw_context.outer_plan.splits)),
                        "observed_folds": int(len(captures)),
                    },
                )
            scores = np.asarray(
                [float(capture.balanced_accuracy) for capture in captures], dtype=float
            )
            if not bool(np.all(np.isfinite(scores))):
                raise NestedPairingEvaluationError(
                    "nested_pairing_nonfinite_fold_score",
                    "Nested pairing requires finite score coverage for every candidate/fold.",
                    diagnostics={"candidate_name": str(candidate_name)},
                )
            scores_by_candidate[str(candidate_name)] = scores
            summaries.append(
                {
                    "candidate_name": str(candidate_name),
                    "nested_oof_mean": float(np.mean(scores)),
                    "nested_oof_std": float(
                        np.std(scores, ddof=1) if scores.size > 1 else 0.0
                    ),
                    "nested_oof_fold_count": int(scores.size),
                    "nested_oof_test_rows": int(
                        sum(len(capture.test_indices) for capture in captures)
                    ),
                    "nested_oof_split_fingerprints": [
                        str(capture.split_fingerprint) for capture in captures
                    ],
                }
            )

        selection = self._select_nested_pairing_from_fold_scores(
            candidates=candidates,
            scores_by_candidate=scores_by_candidate,
            configured_name=str(configured_name),
        )
        raw_best_name = str(selection["raw_best_name"])
        selected_name = str(selection["selected_name"])
        selected_source = f"maqc_pairing_{mode}"
        if bool(selection["reverted"]):
            selected_source = f"maqc_pairing_{mode}_reverted"
        base_scores = np.asarray(selection["configured_scores"], dtype=float)
        best_scores = np.asarray(selection["raw_best_scores"], dtype=float)
        improvement = float(selection["improvement"])
        paired_se = float(selection["paired_se"])
        abs_thr = float(selection["min_improvement"])
        se_mult = float(selection["min_improvement_se_mult"])
        reverted = bool(selection["reverted"])
        revert_reason = str(selection["revert_reason"])

        selected_methods = next(
            tuple(methods) for name, methods in candidates if str(name) == selected_name
        )
        selected_scores = scores_by_candidate[selected_name]
        elapsed = self._nested_pairing_check_runtime(
            started_at=started_at,
            phase="nested_selection_complete",
            planned_evaluations=planned,
            completed_evaluations=completed,
        )
        pairing_meta: Dict[str, Any] = {
            "maqc_pairing_enabled": True,
            "maqc_pairing_score_mode": str(mode),
            "maqc_pairing_score_space": "nested_oof",
            "maqc_pairing_selected_fs_name": str(selected_name),
            "maqc_pairing_selected_cv_score": float(np.mean(selected_scores)),
            "maqc_pairing_selected_cv_score_std": float(
                np.std(selected_scores, ddof=1) if selected_scores.size > 1 else 0.0
            ),
            "maqc_pairing_selected_cv_n_splits": int(selected_scores.size),
            "maqc_pairing_raw_best_fs_name": str(raw_best_name),
            "maqc_pairing_raw_best_cv_score": float(np.mean(best_scores)),
            "maqc_pairing_raw_best_cv_score_std": float(
                np.std(best_scores, ddof=1) if best_scores.size > 1 else 0.0
            ),
            "maqc_pairing_configured_fs_name": str(configured_name),
            "maqc_pairing_configured_cv_score": float(np.mean(base_scores)),
            "maqc_pairing_configured_cv_score_std": float(
                np.std(base_scores, ddof=1) if base_scores.size > 1 else 0.0
            ),
            "maqc_pairing_improvement": float(improvement),
            "maqc_pairing_improvement_se": float(paired_se),
            "maqc_pairing_min_improvement": float(abs_thr),
            "maqc_pairing_min_improvement_se_mult": float(se_mult),
            "maqc_pairing_reverted": bool(reverted),
            "maqc_pairing_revert_reason": str(revert_reason),
            "maqc_pairing_candidate_count": int(len(candidates)),
            "maqc_pairing_evaluated_count": int(len(candidates)),
            "maqc_pairing_failed_count": 0,
            "maqc_pairing_outer_plan_fingerprint": str(raw_context.outer_plan.fingerprint),
            "maqc_pairing_outer_context_fingerprint": str(
                raw_context.fit_resampling_context.fingerprint
            ),
            "maqc_pairing_outer_row_ids_fingerprint": str(
                raw_context.fit_resampling_context.row_ids_fingerprint
            ),
            "maqc_pairing_outer_policy": str(
                raw_context.fit_resampling_context.policy.kind
            ),
            "maqc_pairing_outer_fold_count": int(len(raw_context.outer_plan.splits)),
            "maqc_pairing_outer_evaluations_planned": int(planned),
            "maqc_pairing_outer_evaluations_completed": int(completed),
            "maqc_pairing_max_outer_evaluations": int(cap),
            "maqc_pairing_max_runtime_seconds": float(
                self.config.maqc_pairing_max_runtime_seconds
            ),
            "maqc_pairing_elapsed_seconds": float(elapsed),
            "maqc_pairing_resume_policy": "rerun_all_atomic",
            "maqc_pairing_candidate_score_summary": summaries,
        }
        if mode == "nested_bbc":
            pairing_meta.update(
                self._nested_pairing_bbc_metadata(
                    raw_context=raw_context,
                    candidates=candidates,
                    captures_by_candidate=captures_by_candidate,
                    configured_candidate_name=str(configured_name),
                    selected_candidate_name=str(selected_name),
                    started_at=started_at,
                    planned_evaluations=planned,
                    completed_evaluations=completed,
                )
            )
            pairing_meta["maqc_pairing_elapsed_seconds"] = self._nested_pairing_check_runtime(
                started_at=started_at,
                phase="nested_bbc_complete",
                planned_evaluations=planned,
                completed_evaluations=completed,
            )
        return {
            "selected_name": str(selected_name),
            "selected_methods": tuple(str(method) for method in selected_methods),
            "selected_source": str(selected_source),
            "pairing_meta": pairing_meta,
        }

    def _choose_nested_selector_candidate(
        self,
        *,
        raw_context: _NestedPairingRawContext,
        configured_methods: Sequence[str],
        X_fs: np.ndarray,
        y_fs: np.ndarray,
        X_train_full: np.ndarray,
        X_test_full: np.ndarray,
        y_train_full: np.ndarray,
        seed: int,
        dataset_name: str,
        selector_overrides: Optional[Dict[str, Any]],
        post_df_source_raw_train: Optional[np.ndarray],
        post_df_source_raw_test: Optional[np.ndarray],
        post_df_source_base_train: Optional[np.ndarray],
        post_df_source_base_test: Optional[np.ndarray],
        post_df_source_space: str,
    ) -> Dict[str, Any]:
        """Fit exactly one full-outer candidate after nested-only selection."""

        evidence = self._run_nested_pairing_evaluation(
            raw_context=raw_context,
            configured_methods=configured_methods,
        )
        try:
            selected = self._evaluate_selector_candidate(
                X_fs=X_fs,
                y_fs=y_fs,
                X_train_full=X_train_full,
                X_test_full=X_test_full,
                y_train_full=y_train_full,
                seed=seed,
                enabled_methods=tuple(evidence["selected_methods"]),
                candidate_name=str(evidence["selected_name"]),
                dataset_name=dataset_name,
                selector_overrides=selector_overrides,
                post_df_source_raw_train=post_df_source_raw_train,
                post_df_source_raw_test=post_df_source_raw_test,
                post_df_source_base_train=post_df_source_base_train,
                post_df_source_base_test=post_df_source_base_test,
                post_df_source_space=post_df_source_space,
            )
        except Exception as exc:
            raise NestedPairingEvaluationError(
                "nested_pairing_final_fit_failed",
                "The nested-selected selector stack could not complete its one full-outer-train fit.",
                diagnostics={
                    "selected_candidate": str(evidence["selected_name"]),
                    "exception_type": type(exc).__name__,
                },
            ) from exc
        pairing_meta = dict(evidence["pairing_meta"])
        pairing_meta.update(
            {
                "maqc_pairing_final_fit_candidate_name": str(
                    evidence["selected_name"]
                ),
                "maqc_pairing_final_fit_model_cv_score": float(
                    selected.get("model_cv_score", float("nan"))
                ),
                "maqc_pairing_final_fit_model_cv_score_std": float(
                    selected.get("model_cv_score_std", float("nan"))
                ),
                "maqc_pairing_final_fit_model_cv_n_splits": int(
                    selected.get("model_cv_score_n_splits", 0) or 0
                ),
                "maqc_pairing_final_fit_model_cv_used_for_selection": False,
            }
        )
        selected["enabled_methods_source"] = str(evidence["selected_source"])
        selected["pairing_meta"] = pairing_meta
        return selected

    def run_pre_split(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_test: Any,
        y_test: np.ndarray,
        dataset_name: str = "dataset",
        seed: Optional[int] = None,
        split_indices_train: Optional[Sequence[int]] = None,
        split_indices_test: Optional[Sequence[int]] = None,
        batch_labels_train: Optional[Sequence[Any]] = None,
        batch_labels_test: Optional[Sequence[Any]] = None,
        *,
        sample_weight_train: Optional[Sequence[float]] = None,
        sample_weight_test: Optional[Sequence[float]] = None,
        schema: DatasetSchema | None = None,
        resampling_context: Optional[FitResamplingContext] = None,
        resolved_outer_split: Optional[ResolvedSplitPlan] = None,
        capture_artifacts: bool = False,
        capture_diagnostics: bool = False,
        external_feature_scores: Optional[np.ndarray] = None,
        _meta_learning_resolution: Optional[Dict[str, Any]] = None,
        _capture_evaluation_predictions: bool = False,
        _evaluation_prediction_sink: Optional[Any] = None,
        _nested_pairing_reentrancy: bool = False,
    ) -> PipelineRunResult:
        """Run DF+FS pipeline on an explicit train/test split.

        This is primarily used for protocol overlays (e.g., repeated nested CV audits)
        where the caller controls the split and the pipeline must not re-split.

        ``external_feature_scores`` (native integration §2.3): an optional dense vector
        of per-feature relevance scores indexed by ORIGINAL feature column
        (length == ``X_train.shape[1]``), supplied pre-slice. Consumed only by the
        opt-in DIAKRINO prefilter mode. The default ``protected_union`` mode fits the
        classical selector independently, then appends bounded DIAKRINO extras by original
        feature identity. ``None`` (the default) is a strict no-op.
        """
        seed = int(self.config.random_seed if seed is None else seed)
        rng = np.random.default_rng(seed)
        active_diakrino_dataset_id = self._base_dataset_name(str(dataset_name or "dataset"))
        if str(getattr(self, "_active_diakrino_dataset_id", "") or "") != active_diakrino_dataset_id:
            self._diakrino_transform_sidecar_cache = None
            try:
                setattr(self.dist_fitter, "_diakrino_sidecar_cache", "unset")
            except Exception:
                pass
        self._active_diakrino_dataset_id = active_diakrino_dataset_id
        try:
            setattr(self.dist_fitter, "_active_diakrino_dataset_id", active_diakrino_dataset_id)
        except Exception:
            pass

        typed_requested = self._typed_frontend_requested(X_train, schema=schema)
        y_train_arr = np.asarray(y_train).ravel()
        y_test_arr = np.asarray(y_test).ravel()
        raw_train_shape = tuple(
            int(value)
            for value in getattr(X_train, "shape", np.asarray(X_train).shape)
        )
        raw_test_shape = tuple(
            int(value)
            for value in getattr(X_test, "shape", np.asarray(X_test).shape)
        )
        if len(raw_train_shape) != 2:
            raise ValueError(f"Expected 2D X_train, got shape {raw_train_shape}")
        if len(raw_test_shape) != 2:
            raise ValueError(f"Expected 2D X_test, got shape {raw_test_shape}")
        if raw_train_shape[1] != raw_test_shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: X_train has {raw_train_shape[1]} "
                f"features but X_test has {raw_test_shape[1]}."
            )
        if raw_train_shape[0] != y_train_arr.shape[0]:
            raise ValueError(
                f"Row mismatch: X_train has {raw_train_shape[0]} rows but y_train has {y_train_arr.shape[0]}."
            )
        if raw_test_shape[0] != y_test_arr.shape[0]:
            raise ValueError(
                f"Row mismatch: X_test has {raw_test_shape[0]} rows but y_test has {y_test_arr.shape[0]}."
            )
        if batch_labels_train is None:
            batch_train_arr = None
        else:
            batch_train_arr = np.asarray(list(batch_labels_train), dtype=object).ravel()
            if int(batch_train_arr.size) != int(raw_train_shape[0]):
                raise ValueError(
                    f"batch_labels_train has {batch_train_arr.size} rows but X_train has {raw_train_shape[0]}."
                )
        if batch_labels_test is None:
            batch_test_arr = None
        else:
            batch_test_arr = np.asarray(list(batch_labels_test), dtype=object).ravel()
            if int(batch_test_arr.size) != int(raw_test_shape[0]):
                raise ValueError(
                    f"batch_labels_test has {batch_test_arr.size} rows but X_test has {raw_test_shape[0]}."
                )
        if bool(
            getattr(
                self._classification_cfg(),
                "native_categorical_stage2_enabled",
                False,
            )
        ):
            if not typed_requested:
                raise TypedInputCapabilityError(
                    "native_stage2_typed_input_required",
                    "Native categorical Stage-2 routing requires the opt-in typed DataFrame boundary.",
                    diagnostics={"typed_requested": False},
                )
            if bool(is_sparse_input(X_train)) or bool(is_sparse_input(X_test)):
                raise TypedInputCapabilityError(
                    "native_stage2_sparse_unavailable",
                    "Native categorical Stage-2 routing does not support sparse input.",
                )
        (
            full_resampling_context,
            fit_resampling_context,
            outer_split_plan,
            train_idx_out,
            test_idx_out,
        ) = self._prepare_pre_split_resampling(
            y_train=y_train_arr,
            y_test=y_test_arr,
            split_indices_train=split_indices_train,
            split_indices_test=split_indices_test,
            batch_labels_train=batch_train_arr,
            batch_labels_test=batch_test_arr,
            sample_weight_train=sample_weight_train,
            sample_weight_test=sample_weight_test,
            resampling_context=resampling_context,
            resolved_outer_split=resolved_outer_split,
        )
        self._active_resampling_plans: Dict[str, Any] = {}
        self._active_fit_resampling_context = fit_resampling_context
        self._active_fs_resampling_context = None
        self._typed_feature_selector_runtime: FeatureSelectorRuntimeFacts | None = None
        self._typed_feature_selector_admission: Dict[str, Any] = {}
        nested_pairing_raw_context = self._prepare_nested_pairing_raw_context(
            X_train=X_train,
            y_train=y_train_arr,
            schema=schema,
            batch_labels=batch_train_arr,
            fit_resampling_context=fit_resampling_context,
            dataset_name=dataset_name,
            seed=seed,
            configured_methods=tuple(str(method) for method in self.config.enabled_methods),
            external_feature_scores=external_feature_scores,
            reentrancy_guard=bool(_nested_pairing_reentrancy),
        )
        typed_preprocessor: FoldLocalPreprocessor | None = None
        source_input_schema: DatasetSchema | None = None
        model_input_schema: DatasetSchema | None = None
        typed_preprocessing: Dict[str, Any] = {}
        if typed_requested:
            (
                X_train_arr,
                X_test_arr,
                typed_preprocessor,
                typed_preprocessing,
            ) = self._prepare_typed_train_test_inputs(
                X_train=X_train,
                X_test=X_test,
                schema=schema,
            )
            source_input_schema = typed_preprocessor.input_schema_
            model_input_schema = typed_preprocessor.get_output_schema(
                output_mode="numeric"
            )
        else:
            X_train_arr = np.asarray(X_train, dtype=float)
            X_test_arr = np.asarray(X_test, dtype=float)

        if X_train_arr.ndim != 2:
            raise ValueError(f"Expected 2D X_train, got shape {X_train_arr.shape}")
        if X_test_arr.ndim != 2:
            raise ValueError(f"Expected 2D X_test, got shape {X_test_arr.shape}")
        if X_train_arr.shape[1] != X_test_arr.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: X_train has {X_train_arr.shape[1]} "
                f"features but X_test has {X_test_arr.shape[1]}."
            )
        # DIAKRINO native integration §2.3: validate + stage the optional external per-feature
        # score vector (dense, ORIGINAL-index, pre-slice).  None => strict no-op.
        if typed_requested and external_feature_scores is not None:
            raise TypedInputCapabilityError(
                "typed_external_feature_scores_unavailable",
                "External feature scores are indexed by the legacy numeric input "
                "matrix and cannot be safely mapped through typed derived-feature "
                "lineage yet.",
                diagnostics={
                    "source_schema_fingerprint": (
                        "" if source_input_schema is None else source_input_schema.fingerprint
                    ),
                    "model_schema_fingerprint": (
                        "" if model_input_schema is None else model_input_schema.fingerprint
                    ),
                },
            )
        if external_feature_scores is None:
            self._diakrino_external_feature_scores = None
        else:
            _ext = np.asarray(external_feature_scores, dtype=float).ravel()
            if _ext.shape[0] != X_train_arr.shape[1]:
                raise ValueError(
                    f"external_feature_scores has length {_ext.shape[0]} but X_train has "
                    f"{X_train_arr.shape[1]} features (must be dense, original-index, pre-slice)."
                )
            self._diakrino_external_feature_scores = _ext
        if X_train_arr.shape[0] != y_train_arr.shape[0]:
            raise ValueError(
                f"Row mismatch: X_train has {X_train_arr.shape[0]} rows but y_train has {y_train_arr.shape[0]}."
            )
        if X_test_arr.shape[0] != y_test_arr.shape[0]:
            raise ValueError(
                f"Row mismatch: X_test has {X_test_arr.shape[0]} rows but y_test has {y_test_arr.shape[0]}."
            )
        fit_sample_weight = (
            None
            if not fit_resampling_context.sample_weights
            else np.asarray(
                coerce_sample_weights(
                    fit_resampling_context.sample_weights,
                    n_rows=int(y_train_arr.size),
                    field_name="fit_sample_weights",
                    require_positive_mass=True,
                ),
                dtype=float,
            )
        )
        test_sample_weight = (
            None
            if not full_resampling_context.sample_weights
            else np.asarray(
                coerce_sample_weights(
                    tuple(
                        full_resampling_context.sample_weights[index]
                        for index in outer_split_plan.primary.test_indices
                    ),
                    n_rows=int(y_test_arr.size),
                    field_name="test_sample_weights",
                    require_positive_mass=True,
                ),
                dtype=float,
            )
        )
        self._sample_weight_provenance = {
            "sample_weight_requested": bool(fit_sample_weight is not None),
            "sample_weight_feature_selection_consumed": False,
            "sample_weight_stage2_fit_consumed": False,
            "sample_weight_stage2_cv_consumed": False,
            "sample_weight_posthoc_calibration_consumed": False,
            "sample_weight_metrics_consumed": bool(test_sample_weight is not None),
        }
        self._typed_feature_selector_sample_weight_requested = bool(
            fit_sample_weight is not None
        )
        if typed_requested:
            self._typed_feature_selector_runtime = FeatureSelectorRuntimeFacts(
                input_is_sparse=bool(is_sparse_input(X_train)),
                input_has_categorical=bool(
                    source_input_schema is not None
                    and any(
                        feature.role is FeatureRole.CATEGORICAL
                        for feature in source_input_schema.features
                    )
                ),
                input_has_missing=bool(
                    typed_preprocessor is not None
                    and getattr(typed_preprocessor, "input_has_missing_", False)
                ),
                # Feature selection remains intentionally unweighted.  The
                # Stage-2 classifier contract records its separate consumption.
                sample_weight_requested=False,
                structured_resampling_requested=(
                    fit_resampling_context.policy.kind != "iid"
                ),
                fold_local_adapter=str(
                    typed_preprocessing.get("adapter", "numeric") or "numeric"
                ),
                structured_output_required=False,
            )
        native_stage2_context = self._prepare_native_categorical_stage2_context(
            typed_requested=typed_requested,
            preprocessor=typed_preprocessor,
            X_train=X_train,
            X_test=X_test,
            source_schema=source_input_schema,
            fit_resampling_context=fit_resampling_context,
            sample_weight_requested=bool(fit_sample_weight is not None),
        )
        if native_stage2_context is not None:
            typed_preprocessing.setdefault("preprocessor", {})[
                "native_categorical_route"
            ] = {
                "available": True,
                "status": "context_prepared_before_selector_cv",
                "reason": "selected_view_pending_numeric_feature_selection",
                "pipeline_route": "native_categorical_singleton_pending_selection",
                "adapter_identity": native_stage2_context.adapter.adapter_identity,
                "canonical_name": native_stage2_context.classifier_name,
                "source_schema_fingerprint": native_stage2_context.bridge.source_schema_fingerprint,
                "numeric_schema_fingerprint": native_stage2_context.bridge.numeric_schema_fingerprint,
                "native_schema_fingerprint": native_stage2_context.bridge.native_schema_fingerprint,
            }
        if fit_resampling_context.policy.kind != "iid":
            dist_cfg = getattr(self, "dist_fitter", None)
            dist_cfg = getattr(dist_cfg, "config", self.config.dist_config)
            if bool(getattr(dist_cfg, "use_cv", False)):
                require_supported_resampling(
                    fit_resampling_context,
                    callsite="distribution_selector_cv_after_finite_filter",
                    supported_policies=tuple(),
                )
            if bool(getattr(dist_cfg, "mnpo_include_preq", False)):
                require_supported_resampling(
                    fit_resampling_context,
                    callsite="distribution_selector_preq_after_finite_filter",
                    supported_policies=tuple(),
                )
        if full_resampling_context.batch_ids:
            expected_train_batch = tuple(
                full_resampling_context.batch_ids[index]
                for index in outer_split_plan.primary.train_indices
            )
            expected_test_batch = tuple(
                full_resampling_context.batch_ids[index]
                for index in outer_split_plan.primary.test_indices
            )
            if batch_train_arr is not None and tuple(
                typed_scalar_key(value) for value in batch_train_arr.tolist()
            ) != tuple(typed_scalar_key(value) for value in expected_train_batch):
                raise ResamplingContractError(
                    "batch_labels_train is not aligned with the resolved split context.",
                    code="batch_identity_mismatch",
                    diagnostics={"partition": "train", "n_rows": len(expected_train_batch)},
                )
            if batch_test_arr is not None and tuple(
                typed_scalar_key(value) for value in batch_test_arr.tolist()
            ) != tuple(typed_scalar_key(value) for value in expected_test_batch):
                raise ResamplingContractError(
                    "batch_labels_test is not aligned with the resolved split context.",
                    code="batch_identity_mismatch",
                    diagnostics={"partition": "test", "n_rows": len(expected_test_batch)},
                )

        if _meta_learning_resolution is None:
            active_config, meta_learning_resolution = self._resolve_meta_learning_runtime_config(
                X_train_arr,
                y_train_arr,
            )
            if active_config is not None:
                if typed_requested:
                    raise TypedInputCapabilityError(
                        "typed_meta_learning_runtime_unavailable",
                        "The meta-learning runtime overlay cannot yet preserve the "
                        "fitted typed preprocessor and its lineage. Disable the overlay "
                        "or use the legacy numeric input route until #236 provides a "
                        "replayable fitted bundle.",
                        diagnostics={
                            "source_schema_fingerprint": (
                                ""
                                if source_input_schema is None
                                else source_input_schema.fingerprint
                            ),
                        },
                    )
                delegated = DistributionFeatureSelectionPipeline(active_config)
                return delegated.run_pre_split(
                    X_train=X_train_arr,
                    y_train=y_train_arr,
                    X_test=X_test_arr,
                    y_test=y_test_arr,
                    dataset_name=dataset_name,
                    seed=seed,
                    # The resolved plan indexes positions in its full-row
                    # context.  The caller may instead have supplied durable
                    # external row IDs, which are only appropriate at the
                    # public boundary and would fail the delegated contract.
                    split_indices_train=outer_split_plan.primary.train_indices,
                    split_indices_test=outer_split_plan.primary.test_indices,
                    batch_labels_train=batch_train_arr,
                    batch_labels_test=batch_test_arr,
                    sample_weight_train=fit_sample_weight,
                    sample_weight_test=test_sample_weight,
                    resampling_context=full_resampling_context,
                    resolved_outer_split=outer_split_plan,
                    capture_artifacts=bool(capture_artifacts),
                    capture_diagnostics=bool(capture_diagnostics),
                    _meta_learning_resolution=meta_learning_resolution,
                )
        meta_learning_resolution = dict(_meta_learning_resolution or {})

        n_total = int(full_resampling_context.n_rows)
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

        identity_breaks: List[str] = []
        if bool(face_meta.get("face_projection_applied", False)):
            identity_breaks.append("face_projection")
        if bool(ratio_meta.get("ratio_features_applied", False)):
            identity_breaks.append("ratio_generation")
        if bool(multiomics_meta.get("multiomics_adapter_applied", False)):
            identity_breaks.append("multiomics_adapter")
        if int(X_train_model_input.shape[1]) != int(n_features):
            identity_breaks.append("feature_width_changed")
        if self._df_stage_position() != "after_fs" and str(
            getattr(self.config, "folding_method", "pls_da") or "pls_da"
        ).strip().lower() != "none":
            identity_breaks.append("pre_fs_folding")
        self._diakrino_prefilter_identity_available = not bool(identity_breaks)
        self._diakrino_prefilter_identity_reason = (
            "original_feature_indices" if not identity_breaks else "+".join(dict.fromkeys(identity_breaks))
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
        if native_stage2_context is not None:
            native_stage2_context = native_stage2_context.with_selector_numeric_positions(
                np.asarray(prefilter_idx, dtype=int).ravel().tolist()
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

        if fit_resampling_context.policy.kind == "iid":
            fs_idx_local = self._sample_fs_indices(
                y_train_arr,
                fs_fraction=self.config.fs_fraction,
                seed=seed,
                use_balanced=self.config.use_balanced_fs_subsample,
                min_per_class=self.config.fs_min_per_class,
            )
        else:
            fs_idx_local = np.asarray(
                resolve_fit_subsample(
                    fit_resampling_context,
                    y_train_arr,
                    fraction=float(self.config.fs_fraction),
                    seed=int(seed),
                    balanced=bool(self.config.use_balanced_fs_subsample),
                    min_per_class=int(self.config.fs_min_per_class),
                ),
                dtype=int,
            )
        fs_resampling_context = fit_resampling_context.take(
            fs_idx_local,
            parent_split_fingerprint=outer_split_plan.primary.fingerprint,
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
            fit_resampling_context=fit_resampling_context,
            fs_resampling_context=fs_resampling_context,
            post_df_source_raw_train=post_df_source_raw_train,
            post_df_source_raw_test=post_df_source_raw_test,
            post_df_source_base_train=post_df_source_base_train,
            post_df_source_base_test=post_df_source_base_test,
            post_df_source_space=post_df_source_space,
            sample_weight_train=fit_sample_weight,
            sample_weight_test=test_sample_weight,
            native_stage2_context=native_stage2_context,
            nested_pairing_raw_context=nested_pairing_raw_context,
            capture_evaluation_predictions=bool(_capture_evaluation_predictions),
        )
        # Nested BBC consumes this through an internal callback while the clone
        # is still on the stack. It is never placed on PipelineRunResult.
        evaluation_prediction_capture = fs_result.pop(
            "_evaluation_prediction_capture", None
        )
        if bool(_capture_evaluation_predictions):
            if not isinstance(evaluation_prediction_capture, Mapping):
                raise NestedPairingEvaluationError(
                    "nested_pairing_prediction_capture_missing",
                    "The requested private nested-evaluation capture was not produced.",
                )
            if not callable(_evaluation_prediction_sink):
                raise NestedPairingEvaluationError(
                    "nested_pairing_prediction_sink_missing",
                    "Private nested-evaluation capture requires an in-memory callback sink.",
                )
            try:
                _evaluation_prediction_sink(evaluation_prediction_capture)
            except NestedPairingEvaluationError:
                raise
            except Exception as exc:
                raise NestedPairingEvaluationError(
                    "nested_pairing_prediction_sink_failed",
                    "The private nested-evaluation callback rejected a fold capture.",
                    diagnostics={"exception_type": type(exc).__name__},
                ) from exc
        native_stage2_route = dict(fs_result.get("native_stage2_route") or {})
        if typed_preprocessor is not None and native_stage2_route:
            typed_preprocessing.setdefault("preprocessor", {})[
                "native_categorical_route"
            ] = _json_safe(native_stage2_route)
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
        selected_model_input_indices: Tuple[int, ...] = tuple()
        if bool(face_meta.get("face_projection_applied", False)) or (
            df_stage_position != "after_fs" and bool(folding_meta.get("folding_applied", False))
        ):
            selected_original = tuple()
        else:
            diakrino_protected_diag = dict(fs_result.get("diakrino_protected_augmentation", {}) or {})
            if bool(diakrino_protected_diag.get("applied", False)):
                selected_model_input_indices = tuple(
                    int(i)
                    for i in list(diakrino_protected_diag.get("final_original_indices", []) or [])
                )
            else:
                active_input_indices = np.asarray(
                    fs_result.get(
                        "variance_floor_active_input_indices",
                        np.arange(len(prefilter_idx), dtype=int),
                    ),
                    dtype=int,
                ).ravel()
                selected_model_input_indices = tuple(
                    int(prefilter_idx[active_input_indices[index]])
                    for index in selected_local
                    if 0 <= index < active_input_indices.size
                    and 0 <= active_input_indices[index] < len(prefilter_idx)
                )
            selected_original = selected_model_input_indices
            # The downstream classifier receives this final set, which can
            # differ from the selector's pre-cap/top-k count in every arm.
            # Emit a count that is cryptographically consistent with the
            # original-index identity recorded alongside the result.
            fs_result["selected_features"] = len(selected_model_input_indices)

        selected_feature_schema: Dict[str, Any] = {}
        if model_input_schema is not None:
            pipeline_preserves_model_identity = not bool(identity_breaks)
            source_to_model_identity = bool(
                source_input_schema is not None
                and source_input_schema.n_features == model_input_schema.n_features
                and source_input_schema.feature_names == model_input_schema.feature_names
            )
            if pipeline_preserves_model_identity and selected_model_input_indices:
                try:
                    selected_feature_schema = model_input_schema.select(
                        selected_model_input_indices,
                        operation="feature_selection_output",
                    ).to_record()
                except (SchemaContractError, ValueError, IndexError) as exc:
                    selected_feature_schema = {
                        "status": "unavailable",
                        "reason": str(type(exc).__name__),
                    }
            elif not pipeline_preserves_model_identity:
                selected_feature_schema = {
                    "status": "unavailable",
                    "reason": "pipeline_feature_identity_changed",
                    "identity_breaks": list(identity_breaks),
                }
            if not source_to_model_identity:
                # The legacy integer field cannot represent one-to-many text/date
                # expansion or dropped non-model columns. Preserve only the richer
                # model-input schema instead of reporting a false raw-column index.
                selected_original = tuple()

        snapshot = self._config_snapshot()
        sample_weight_requested = bool(fit_sample_weight is not None)
        resampling_trace = {
            "schema_version": "1.0",
            "full_context": full_resampling_context.to_metadata(
                sample_weights_consumed=sample_weight_requested,
                sample_weight_usage=(
                    "stage2_fit_calibration_and_metrics"
                    if sample_weight_requested
                    else "not_consumed"
                ),
            ),
            "fit_context": fit_resampling_context.to_metadata(
                sample_weights_consumed=sample_weight_requested,
                sample_weight_usage=(
                    "stage2_cv_fit_and_calibration"
                    if sample_weight_requested
                    else "not_consumed"
                ),
            ),
            "outer_plan": outer_split_plan.to_metadata(),
            "inner_plans": dict(
                getattr(self, "_active_resampling_plans", {}) or {}
            ),
        }
        snapshot["resampling"] = _json_safe(resampling_trace)
        snapshot["sample_weight_provenance"] = _json_safe(
            fs_result.get("sample_weight_provenance", {})
        )
        if source_input_schema is not None and model_input_schema is not None:
            snapshot["typed_preprocessing"] = _json_safe(typed_preprocessing)
            snapshot["input_schema"] = source_input_schema.to_record()
            snapshot["model_input_schema"] = model_input_schema.to_record()
            snapshot["selected_feature_schema"] = _json_safe(selected_feature_schema)
            snapshot["selected_model_input_indices"] = [
                int(index) for index in selected_model_input_indices
            ]
            snapshot["typed_feature_selector_admission"] = _json_safe(
                fs_result.get("typed_feature_selector_admission", {})
            )
            if native_stage2_route:
                snapshot["native_categorical_stage2_route"] = _json_safe(
                    native_stage2_route
                )
        snapshot["df_stage_position_effective"] = str(dist_meta.get("df_stage_position", df_stage_position))
        snapshot["df_stage_source_space"] = str(dist_meta.get("df_stage_source_space", "model_input"))
        if isinstance(fs_result.get("diakrino_protected_augmentation"), dict) and bool(
            fs_result.get("diakrino_protected_augmentation")
        ):
            snapshot["diakrino_protected_augmentation"] = _json_safe(
                fs_result.get("diakrino_protected_augmentation", {})
            )
        effective = fs_result.get("effective_enabled_methods")
        if effective is not None:
            snapshot["enabled_methods"] = [str(m) for m in effective]
            snapshot["effective_enabled_methods"] = [str(m) for m in effective]
        requested = fs_result.get("requested_enabled_methods")
        if requested is not None:
            snapshot["requested_enabled_methods"] = [str(m) for m in requested]
        if "enabled_methods_source" in fs_result:
            snapshot["enabled_methods_source"] = str(fs_result["enabled_methods_source"])
        if self._nested_pairing_mode() != "raw_cv":
            for key, value in fs_result.items():
                if str(key).startswith("maqc_pairing_"):
                    snapshot[str(key)] = _json_safe(value)
        else:
            # Keep the established raw-CV snapshot surface byte-compatible.
            for key in (
                "maqc_pairing_enabled",
                "maqc_pairing_score_space",
                "maqc_pairing_selected_fs_name",
                "maqc_pairing_selected_cv_score",
                "maqc_pairing_augmented_cv_score",
                "maqc_pairing_augmented_selected_feature_count",
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
            "diakrino_regime_conditional_enabled",
            "diakrino_regime_conditional_applied",
            "diakrino_regime_conditional_reason",
            "diakrino_regime_conditional_regime",
            "diakrino_regime_conditional_allowed",
            "diakrino_regime_conditional_allowed_regimes",
            "diakrino_regime_conditional_methods",
            "diakrino_regime_conditional_removed_methods",
            "diakrino_regime_conditional_enabled_methods_before",
            "diakrino_regime_conditional_enabled_methods_after",
            "selector_overrides_applied",
            "feature_selection_resampling",
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
            "model_cv_svc_probability_enabled",
            "model_cv_svc_probability_candidates",
            "model_cv_candidate_wall_seconds",
            "model_cv_svc_probability_runtime_note",
            "model_cv_structured_resampling_policy",
            "model_cv_structured_resampling_excluded",
            "model_cv_resampling_plan",
            "model_cv_resampling_source",
            "model_cv_sample_weight_requested",
            "model_cv_sample_weight_cv_routed",
            "model_cv_sample_weight_routes",
            "model_cv_sample_weight_excluded_candidates",
            "model_cv_sample_weight_admitted_candidates",
            "classification_backend_requested",
            "classification_backend_used",
            "classification_backend_fallback_reason",
            "classification_stage2_wall_seconds",
            "classification_selected_identity",
            "classification_selected_requested_name",
            "classification_selected_outward_name",
            "classification_selected_canonical_name",
            "classification_selected_effective_model_name",
            "classification_selected_composite_identity",
            "classification_selected_fallback_reason",
            "classification_base_fitted_descriptor",
            "classification_final_fitted_descriptor",
            "classification_fitted_probability_kind",
            "classification_fitted_probability_source",
            "classification_fitted_class_order",
            "classification_fitted_matrix_observation",
            "classification_fitted_matrix_reason",
            "classification_fitted_argmax_observation",
            "classification_fitted_serialization_observation",
            "calibration_sample_weight_consumed",
            "log_loss_available",
            "log_loss_skip_reason",
            "log_loss_reason",
            "log_loss_requirement",
            "log_loss_probability_kind",
            "log_loss_probability_source",
            "log_loss_class_order",
            "log_loss_class_alignment",
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
            "classifier_conformal_score_source",
            "classifier_conformal_used_predict_proba",
            "classifier_conformal_model_supports_predict_proba",
            "classifier_conformal_model_probability_kind",
            "classifier_conformal_conformity_kind",
            "classifier_conformal_calibration_score_source",
            "classifier_conformal_evaluation_score_source",
            "classifier_conformal_class_order",
            "classifier_conformal_source_consistent",
            "classifier_conformal_probability_required",
            "classifier_conformal_probability_claim",
            "classifier_conformal_source_errors",
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
            "classifier_conformal_mapie_probability_requirement",
            "classifier_conformal_mapie_probability_kind",
            "classifier_conformal_mapie_probability_source",
            "classifier_conformal_mapie_probability_admitted",
            "classifier_conformal_mapie_probability_reason",
            "classifier_conformal_mapie_class_order",
            "calibration_reporting_enabled",
            "calibration_metrics_available",
            "calibration_brier",
            "calibration_ece",
            "calibration_n_eval",
            "calibration_n_classes",
            "calibration_skip_reason",
            "calibration_probability_kind",
            "calibration_probability_source",
            "calibration_class_order",
            "calibration_class_alignment",
            "classifier_posthoc_calibration_enabled",
            "classifier_posthoc_calibration_applied",
            "classifier_posthoc_calibration_method",
            "classifier_posthoc_calibration_fraction",
            "classifier_posthoc_calibration_min_calibration",
            "classifier_posthoc_calibration_refinement_stopping",
            "classifier_posthoc_calibration_skip_reason",
            "classifier_posthoc_calibration_size",
            "classifier_posthoc_calibration_fit_size",
            "classifier_posthoc_calibration_base_brier",
            "classifier_posthoc_calibration_base_ece",
            "classifier_posthoc_calibration_calibrated_brier",
            "classifier_posthoc_calibration_calibrated_ece",
            "classifier_posthoc_calibration_probability_kind",
            "classifier_posthoc_calibration_probability_source",
            "classifier_posthoc_calibration_sample_weight_requested",
            "classifier_posthoc_calibration_sample_weight_consumed",
            "classifier_posthoc_calibration_sample_weight_route",
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
                if typed_preprocessor is not None:
                    raise TypedInputCapabilityError(
                        "typed_inference_bundle_pending",
                        "The current reproducible bundle accepts only the legacy "
                        "numeric matrix. A typed fitted preprocessor must be packaged "
                        "with the final estimator by the #236 inference-bundle work.",
                    )
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
                    variance_keep_indices=tuple(
                        int(value)
                        for value in np.asarray(
                            fs_result.get(
                                "variance_floor_active_input_indices",
                                tuple(),
                            ),
                            dtype=int,
                        ).ravel().tolist()
                    ),
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
                        "resampling": _json_safe(resampling_trace),
                        "classification_selected_identity": _json_safe(
                            fs_result.get("classification_selected_identity", {})
                        ),
                        "classification_base_fitted_descriptor": _json_safe(
                            fs_result.get("classification_base_fitted_descriptor", {})
                        ),
                        "classification_final_fitted_descriptor": _json_safe(
                            fs_result.get("classification_final_fitted_descriptor", {})
                        ),
                        "classification_fitted_probability_kind": str(
                            fs_result.get(
                                "classification_fitted_probability_kind",
                                "unknown",
                            )
                        ),
                        "classification_fitted_probability_source": str(
                            fs_result.get(
                                "classification_fitted_probability_source",
                                "unavailable",
                            )
                        ),
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

        run_diagnostics: Dict[str, Any] = {
            "schema_version": "1.0",
            "artifact_type": "df_fs_resampling_trace",
            "dataset_name": str(dataset_name),
            "seed": int(seed),
            "resampling": _json_safe(resampling_trace),
        }
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
            if isinstance(fs_result.get("native_stage2_route"), Mapping):
                classifier_oracle_meta["native_categorical_stage2_route"] = (
                    _json_safe(fs_result["native_stage2_route"])
                )

            run_diagnostics = _json_safe({
                "schema_version": "1.0",
                "artifact_type": "df_fs_run_diagnostics",
                "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "dataset_name": str(dataset_name),
                "seed": int(seed),
                "resampling": _json_safe(resampling_trace),
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
                "diakrino_sidecar_resolution": _json_safe(
                    self._diakrino_sidecar_resolution_diagnostics(int(n_features))
                ),
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
                        **(
                            {
                                "diakrino_prefilter": _json_safe(
                                    dict(getattr(self, "_last_diakrino_prefilter_state", {}) or {})
                                )
                            }
                            if self._diakrino_prefilter_runtime_enabled()
                            else {}
                        ),
                    },
                    "folding_stage": _json_safe(folding_meta),
                    "feature_selection": {
                        "selected_indices_local": [
                            int(i) for i in np.asarray(fs_result.get("selected_indices", tuple()), dtype=int).ravel().tolist()
                        ],
                        "selected_indices_original": [int(i) for i in tuple(selected_original)],
                        "selection_summary": _json_safe(fs_result.get("fs_selection_summary", {})),
                        "detailed": _json_safe(fs_result.get("fs_diagnostics", {})),
                        **(
                            {
                                "diakrino_protected_augmentation": _json_safe(
                                    fs_result.get("diakrino_protected_augmentation", {})
                                )
                            }
                            if bool(fs_result.get("diakrino_protected_augmentation"))
                            else {}
                        ),
                    },
                    "classifier_selection": _json_safe(classifier_oracle_meta),
                },
            })

        if source_input_schema is not None and model_input_schema is not None:
            run_diagnostics["typed_input"] = _json_safe(
                {
                    "preprocessing": typed_preprocessing,
                    "input_schema": source_input_schema.to_record(),
                    "model_input_schema": model_input_schema.to_record(),
                    "selected_feature_schema": selected_feature_schema,
                    "selected_model_input_indices": [
                        int(index) for index in selected_model_input_indices
                    ],
                    "feature_selector_admission": fs_result.get(
                        "typed_feature_selector_admission", {}
                    ),
                }
            )

        return PipelineRunResult(
            dataset_name=str(dataset_name),
            seed=seed,
            n_samples_total=n_total,
            n_features_total=(
                int(source_input_schema.n_features)
                if source_input_schema is not None
                else n_features
            ),
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
            classification_fitted_probability_kind=str(
                fs_result.get("classification_fitted_probability_kind", "unknown")
            ),
            classification_fitted_probability_source=str(
                fs_result.get(
                    "classification_fitted_probability_source", "unavailable"
                )
            ),
            classification_final_fitted_descriptor=dict(
                fs_result.get("classification_final_fitted_descriptor") or {}
            ),
            resampling_context_fingerprint=str(full_resampling_context.fingerprint),
            fit_context_fingerprint=str(fit_resampling_context.fingerprint),
            outer_split_fingerprint=str(outer_split_plan.primary.fingerprint),
            resampling_policy=_json_safe(full_resampling_context.policy.to_record()),
            leakage_audit=_json_safe(outer_split_plan.primary.audit.to_dict()),
            resampling_trace=_json_safe(resampling_trace),
            input_schema=(
                {} if source_input_schema is None else source_input_schema.to_record()
            ),
            model_input_schema=(
                {} if model_input_schema is None else model_input_schema.to_record()
            ),
            selected_feature_schema=_json_safe(selected_feature_schema),
            typed_preprocessing=_json_safe(typed_preprocessing),
        )

    def _diakrino_feature_entropy(self, feat_idx: int) -> Optional[float]:
        sc = self._diakrino_transform_sidecar
        if sc is None or int(feat_idx) < 0 or int(feat_idx) >= int(getattr(sc, "n_features", 0)):
            return None
        ent = sc.family_entropy(normalized=True)
        if ent is None or int(feat_idx) >= int(np.asarray(ent).shape[0]):
            return None
        try:
            val = float(np.asarray(ent, dtype=float).ravel()[int(feat_idx)])
        except Exception:
            return None
        if not np.isfinite(val):
            return None
        return float(np.clip(val, 0.0, 1.0))

    def _diakrino_discrete_skip_family_id(self, feat_idx: int) -> Optional[int]:
        if not bool(getattr(self.config.dist_config, "diakrino_skip_fit_discrete_enabled", False)):
            return None
        sc = self._diakrino_transform_sidecar
        if sc is None or int(feat_idx) < 0 or int(feat_idx) >= int(getattr(sc, "n_features", 0)):
            return None
        mask = sc.discrete_skip_mask()
        ids = sc.family_argmax_ids()
        if mask is None or ids is None or int(feat_idx) >= int(np.asarray(mask).shape[0]):
            return None
        try:
            if bool(np.asarray(mask).ravel()[int(feat_idx)]):
                return int(np.asarray(ids).ravel()[int(feat_idx)])
        except Exception:
            return None
        return None

    def _diakrino_cdf_trust_fallback_mode(self, feat_idx: int) -> Optional[Tuple[str, float]]:
        if not bool(getattr(self.config, "diakrino_cdf_trust_gate_enabled", False)):
            return None
        threshold = float(getattr(self.config, "diakrino_cdf_trust_entropy_threshold", 1.01) or 1.01)
        entropy = self._diakrino_feature_entropy(int(feat_idx))
        if entropy is None or not (entropy > threshold):
            return None
        mode = str(getattr(self.config, "diakrino_cdf_trust_fallback", "rank_gaussian") or "rank_gaussian").strip().lower()
        if mode in {"drop", "none", "skip"}:
            return "drop", float(entropy)
        return "rank_transform", float(entropy)

    def _rank_transform_fallback_result(
        self,
        *,
        feat_idx: int,
        train_col: np.ndarray,
        test_col: np.ndarray,
        summary: DistributionFitSummary,
        apply_reason: str,
        fallback_reason: str,
        fit_method: str,
        fallback_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        original_family = None if summary.family is None else str(summary.family)
        original_params = None if summary.params is None else [float(v) for v in tuple(summary.params)]
        train_g, test_g = self._rank_gaussian_transform_train_test(train_col, test_col)
        train_sorted = np.sort(np.asarray(train_col, dtype=float).ravel())
        train_sorted = train_sorted[np.isfinite(train_sorted)]
        meta = {
            "fallback_mode": "rank_transform",
            "fallback_reason": str(fallback_reason),
            "selected_family_before_fallback": original_family,
            "selected_params_before_fallback": original_params,
            "train_sorted": [float(v) for v in train_sorted.tolist()],
        }
        meta.update(dict(fallback_meta or {}))

        summary.family = "multimodal_fallback_rank_transform"
        summary.params = None
        summary.rejected = False
        summary.rejection_reason = str(fallback_reason)
        summary.selected_family_support = "real"
        summary.fit_method = str(fit_method)

        mu = float(np.mean(train_g))
        sigma = float(np.std(train_g))
        if sigma < 1e-8:
            sigma = 1.0
        return {
            "feat_idx": int(feat_idx),
            "summary": summary,
            "rejected": False,
            "skipped_unreliable": False,
            "downweighted": False,
            "stability_weight": None,
            "stability_source": None,
            "weight": 1.0,
            "train_mean": float(mu),
            "train_std": float(sigma),
            "apply_reason": str(apply_reason),
            "fallback_meta": meta,
            "train_z": (np.asarray(train_g, dtype=float) - mu) / sigma,
            "test_z": (np.asarray(test_g, dtype=float) - mu) / sigma,
        }

    def _diakrino_entropy_stability_surrogate(self, feat_idx: int) -> Optional[float]:
        if not bool(getattr(self.config, "diakrino_stability_surrogate_enabled", False)):
            return None
        entropy = self._diakrino_feature_entropy(int(feat_idx))
        if entropy is None:
            return None
        return float(np.clip(1.0 - entropy, 0.0, 1.0))

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
        discrete_family_id = self._diakrino_discrete_skip_family_id(int(feat_idx))
        if discrete_family_id is not None:
            summary = DistributionFitSummary(
                feature_index=int(feat_idx),
                family="multimodal_fallback_rank_transform",
                params=None,
                cvm_p=float("nan"),
                ks_p=float("nan"),
                simple_score=float("nan"),
                confidence_set=tuple(),
                rejected=False,
                audit=audit,
                rejection_reason="diakrino_skip_fit_discrete",
                selected_family_support="real",
                candidates_pre_filter=int(len(self.dist_fitter._base_distributions)),
                candidates_post_filter=0,
                fit_method="diakrino_skip_fit_discrete_rank_transform",
            )
            return self._rank_transform_fallback_result(
                feat_idx=int(feat_idx),
                train_col=train_col,
                test_col=test_col,
                summary=summary,
                apply_reason="diakrino_skip_fit_discrete_rank_gaussian",
                fallback_reason="diakrino_skip_fit_discrete",
                fit_method="diakrino_skip_fit_discrete_rank_transform",
                fallback_meta={"diakrino_family_argmax_id": int(discrete_family_id)},
            )

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
                "stability_source": None,
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
            "stability_source": None,
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

        cdf_trust = self._diakrino_cdf_trust_fallback_mode(int(feat_idx))
        if cdf_trust is not None:
            cdf_mode, entropy = cdf_trust
            if cdf_mode == "drop":
                result["skipped_unreliable"] = True
                result["apply_reason"] = "diakrino_cdf_trust_drop"
                result["fallback_meta"] = {
                    "fallback_mode": "drop",
                    "fallback_reason": "diakrino_cdf_trust_low_confidence",
                    "diakrino_family_entropy": float(entropy),
                }
                return result
            return self._rank_transform_fallback_result(
                feat_idx=int(feat_idx),
                train_col=train_col,
                test_col=test_col,
                summary=summary,
                apply_reason="diakrino_cdf_trust_rank_gaussian",
                fallback_reason="diakrino_cdf_trust_low_confidence",
                fit_method="diakrino_cdf_trust_rank_transform",
                fallback_meta={"diakrino_family_entropy": float(entropy)},
            )

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
            stability = self._diakrino_entropy_stability_surrogate(int(feat_idx))
            if stability is None:
                stability = self._family_stability_bootstrap(
                    train_col,
                    expected_family=summary.family,
                    n_bootstrap=self.config.stability_bootstrap,
                    seed=seed + int(feat_idx),
                )
                result["stability_source"] = "bootstrap"
            else:
                result["stability_source"] = "diakrino_entropy_surrogate"
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
                "stability_source": (
                    None
                    if r.get("stability_source", None) is None
                    else str(r.get("stability_source"))
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
        diakrino_family_agreement = self._diakrino_family_agreement_audit(feature_plans)

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
            "diakrino_family_agreement_audit": diakrino_family_agreement,
        }
        self._last_distribution_plan = {
            "schema_version": "1.0",
            "apply_cdf_transform": True,
            "n_input_features": int(n_features),
            "dist_feature_indices": [int(i) for i in np.asarray(dist_feature_indices, dtype=int).ravel().tolist()],
            "diakrino_family_agreement_audit": diakrino_family_agreement,
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

    def _block_cv_score(
        self,
        X_block: np.ndarray,
        y_train: np.ndarray,
        seed: int,
        cv_plan: Optional[ResolvedSplitPlan] = None,
    ) -> float:
        y_arr = np.asarray(y_train)
        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) < 2 or counts.min() < 2:
            return float("nan")

        n_splits = int(max(2, min(int(self.config.cdf_block_gating_cv_splits), int(counts.min()))))
        cv: Any
        if cv_plan is None:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
        else:
            cv = [
                (
                    np.asarray(train, dtype=int),
                    np.asarray(test, dtype=int),
                )
                for train, test in cv_plan.index_pairs()
            ]
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
        structured_cv_plan: Optional[ResolvedSplitPlan] = None
        fit_context = getattr(self, "_active_fit_resampling_context", None)
        if (
            isinstance(fit_context, FitResamplingContext)
            and fit_context.policy.kind != "iid"
        ):
            classes, counts = np.unique(np.asarray(y_train).ravel(), return_counts=True)
            if len(classes) >= 2 and int(np.min(counts)) >= 2:
                n_splits = int(
                    max(
                        2,
                        min(
                            int(self.config.cdf_block_gating_cv_splits),
                            int(np.min(counts)),
                        ),
                    )
                )
                structured_cv_plan = self._resolve_inner_split_plan(
                    fit_context,
                    np.asarray(y_train).ravel(),
                    purpose="cdf_block_gating_cv",
                    n_splits=n_splits,
                    n_repeats=1,
                    seed=int(seed),
                    stratified=True,
                    shuffle=True,
                )

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

            cv_kwargs = (
                {}
                if structured_cv_plan is None
                else {"cv_plan": structured_cv_plan}
            )
            base_score = self._block_cv_score(
                X_base_block,
                y_train,
                seed=seed + block_idx * 17 + 1,
                **cv_kwargs,
            )
            cdf_score = self._block_cv_score(
                X_cdf_block,
                y_train,
                seed=seed + block_idx * 17 + 2,
                **cv_kwargs,
            )
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

        if str(fallback_meta.get("fallback_mode", "") or "") == "rank_transform":
            train_sorted = np.sort(np.asarray(train_col, dtype=float).ravel())
            train_sorted = train_sorted[np.isfinite(train_sorted)]
            fallback_meta["train_sorted"] = [float(v) for v in train_sorted.tolist()]

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

    def _admit_typed_selector_methods(
        self,
        enabled_methods: Sequence[str],
        *,
        source: str,
    ) -> Tuple[str, ...]:
        """Resolve typed-input FS admission before the selector receives data.

        The legacy numeric core sees only the fold-local numeric adapter.  This
        method makes that adaptation explicit and refuses a method set that has
        no capability-honored candidate instead of silently changing the set.
        """

        requested = self._normalize_method_set(enabled_methods)
        runtime = getattr(self, "_typed_feature_selector_runtime", None)
        if runtime is None:
            return requested
        try:
            admitted, rejected, records = admit_feature_selector_methods(
                requested,
                runtime=runtime,
            )
        except KeyError as exc:
            raise TypedInputCapabilityError(
                "unknown_typed_feature_selector_method",
                "A typed-input selector method is not present in the authoritative "
                "feature-selector registry.",
                diagnostics={
                    "requested_methods": list(requested),
                    "source": str(source),
                    "error": str(exc),
                },
            ) from exc

        self._typed_feature_selector_admission = {
            "requested_methods": list(requested),
            "admitted_methods": list(admitted),
            "rejected_methods": dict(rejected),
            "source": str(source),
            "runtime": {
                "input_is_sparse": bool(runtime.input_is_sparse),
                "input_has_categorical": bool(runtime.input_has_categorical),
                "input_has_missing": bool(runtime.input_has_missing),
                "sample_weight_requested": bool(runtime.sample_weight_requested),
                "sample_weight_stage2_only": bool(
                    getattr(
                        self,
                        "_typed_feature_selector_sample_weight_requested",
                        False,
                    )
                ),
                "sample_weight_consumed": False,
                "structured_resampling_requested": bool(
                    runtime.structured_resampling_requested
                ),
                "fold_local_adapter": str(runtime.fold_local_adapter),
                "structured_output_required": bool(
                    runtime.structured_output_required
                ),
            },
            "capabilities": [record.to_record() for record in records],
        }
        if not admitted:
            raise TypedInputCapabilityError(
                "no_typed_feature_selector_methods_admitted",
                "No configured feature-selection method can honor the typed input "
                "and resampling contract.",
                diagnostics=dict(self._typed_feature_selector_admission),
            )
        return tuple(admitted)

    def _meta_features_to_tier(self, meta: Dict[str, float]) -> str:
        prediction = _predict_tier_with_details(
            meta,
            mode=str(getattr(self.config, "tier_classifier_mode", "heuristic") or "heuristic"),
            model_path=(
                Path(str(getattr(self.config, "tier_classifier_model_path", "")).strip())
                if str(getattr(self.config, "tier_classifier_model_path", "")).strip()
                else None
            ),
        )
        return str(prediction.tier)

    def _resolve_dataset_tier(
        self,
        dataset_name: str,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        source: str,
    ) -> Tuple[str, str, Dict[str, Any]]:
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
                tier_mode = str(getattr(self.config, "tier_classifier_mode", "heuristic") or "heuristic")
                use_expanded = bool(
                    tier_mode == "learned"
                    or getattr(self.config, "regime_gating_use_expanded_features", False)
                    or getattr(self.config, "prefilter_adaptive_top_k", False)
                )
                meta = extract_meta_features(
                    np.asarray(X_ref, dtype=float),
                    np.asarray(y_ref),
                    expanded=bool(use_expanded),
                )
                prediction = _predict_tier_with_details(
                    meta,
                    mode=tier_mode,
                    model_path=(
                        Path(str(getattr(self.config, "tier_classifier_model_path", "")).strip())
                        if str(getattr(self.config, "tier_classifier_model_path", "")).strip()
                        else None
                    ),
                )
                meta_payload = {str(k): float(v) for k, v in meta.items()}
                meta_payload["complexity_score"] = float(_normalized_complexity_score(meta_payload))
                meta_payload["tier_classifier_mode"] = str(prediction.mode)
                meta_payload["tier_classifier_model_source"] = str(prediction.model_source)
                meta_payload["tier_classifier_confidence"] = float(prediction.confidence)
                meta_payload["tier_classifier_fallback_applied"] = bool(prediction.fallback_applied)
                meta_payload["tier_classifier_model_path"] = str(prediction.model_path)
                meta_payload["tier_classifier_model_sha256"] = str(prediction.model_sha256)
                meta_payload["tier_classifier_model_size_bytes"] = int(
                    prediction.model_size_bytes
                )
                meta_payload["tier_classifier_fallback_reason"] = str(
                    prediction.fallback_reason
                )
                meta_payload["tier_classifier_prediction"] = prediction.to_snapshot()
                return str(prediction.tier), "meta_features", meta_payload
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
            getattr(self.config, "regime_gating_min_samples_per_class", 7.0) or 7.0
        )
        low_p_over_n_threshold = float(
            getattr(self.config, "regime_gating_low_p_over_n_threshold", 0.0) or 0.0
        )
        use_expanded_features = bool(
            getattr(self.config, "regime_gating_use_expanded_features", False)
        )
        fisher_f1 = float(tier_meta.get("fisher_f1", 0.0) or 0.0)
        n1_borderline = float(tier_meta.get("n1_borderline", 0.0) or 0.0)
        fisher_threshold = float(
            getattr(self.config, "regime_gating_min_fisher_f1", 0.10) or 0.10
        )
        n1_threshold = float(
            getattr(self.config, "regime_gating_max_n1_borderline", 0.40) or 0.40
        )
        expanded_trigger = bool(use_expanded_features) and (
            bool(fisher_f1 > 0.0 and fisher_f1 <= fisher_threshold)
            or bool(n1_borderline >= n1_threshold)
        )

        very_hard_trigger = (
            bool(tier == target_tier)
            or bool(samples_per_class < min_samples_per_class)
            or bool(expanded_trigger)
        )
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
            "regime_policy_use_expanded_features": bool(use_expanded_features),
            "regime_policy_fisher_f1": float(fisher_f1),
            "regime_policy_n1_borderline": float(n1_borderline),
            "regime_policy_min_fisher_f1_threshold": float(fisher_threshold),
            "regime_policy_max_n1_borderline_threshold": float(n1_threshold),
            "regime_policy_trigger_very_hard": bool(very_hard_trigger),
            "regime_policy_trigger_low_p_over_n": bool(low_p_over_n_trigger),
            "regime_policy_trigger_expanded_features": bool(expanded_trigger),
            "regime_policy_enabled_methods_before": list(configured_methods),
            "regime_policy_enabled_methods_after": list(configured_methods),
            "regime_policy_enabled_methods": tuple(configured_methods),
            "regime_policy_enabled_methods_source": "config",
            "regime_policy_bypass_fs": False,
            "regime_policy_bypass_mode": "",
            "regime_policy_selector_overrides": {},
            "diakrino_regime_conditional_enabled": bool(getattr(self.config, "diakrino_regime_conditional", False)),
            "diakrino_regime_conditional_applied": False,
            "diakrino_regime_conditional_reason": "disabled",
            "diakrino_regime_conditional_regime": "unknown",
            "diakrino_regime_conditional_allowed": False,
            "diakrino_regime_conditional_allowed_regimes": list(_DIAKRINO_REGIME_CONDITIONAL_ALLOWED_REGIMES),
            "diakrino_regime_conditional_methods": list(_DIAKRINO_REGIME_CONDITIONAL_METHODS),
            "diakrino_regime_conditional_removed_methods": [],
            "diakrino_regime_conditional_enabled_methods_before": list(configured_methods),
            "diakrino_regime_conditional_enabled_methods_after": list(configured_methods),
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

    def _apply_diakrino_regime_conditional_policy(
        self,
        policy: Dict[str, Any],
        X_ref: np.ndarray,
        y_ref: np.ndarray,
    ) -> Dict[str, Any]:
        """Remove DIAKRINO candidate selectors outside HDLSS regimes when opted in."""
        out = dict(policy or {})
        enabled = bool(getattr(self.config, "diakrino_regime_conditional", False))
        methods_before = tuple(str(m) for m in out.get("enabled_methods", tuple()) or tuple())
        meta: Dict[str, Any] = {
            "diakrino_regime_conditional_enabled": bool(enabled),
            "diakrino_regime_conditional_applied": False,
            "diakrino_regime_conditional_reason": "disabled",
            "diakrino_regime_conditional_regime": "unknown",
            "diakrino_regime_conditional_allowed": False,
            "diakrino_regime_conditional_allowed_regimes": list(_DIAKRINO_REGIME_CONDITIONAL_ALLOWED_REGIMES),
            "diakrino_regime_conditional_methods": list(_DIAKRINO_REGIME_CONDITIONAL_METHODS),
            "diakrino_regime_conditional_removed_methods": [],
            "diakrino_regime_conditional_enabled_methods_before": list(methods_before),
            "diakrino_regime_conditional_enabled_methods_after": list(methods_before),
        }
        if not enabled:
            out.update(meta)
            return out

        X_arr = np.asarray(X_ref)
        y_arr = np.asarray(y_ref).ravel()
        n_samples = int(X_arr.shape[0]) if X_arr.ndim == 2 else int(y_arr.size)
        n_features = int(X_arr.shape[1]) if X_arr.ndim == 2 and X_arr.size > 0 else 0
        try:
            from tabnetics.classification.backends import classify_regime

            regime = str(classify_regime(n_samples=n_samples, n_features=n_features))
        except Exception:
            n = int(max(1, n_samples))
            p_over_n = float(n_features) / float(n)
            if n < 50 or p_over_n > 500.0:
                regime = "hdlss_extreme"
            elif n < 200 and p_over_n > 50.0:
                regime = "hdlss_moderate"
            else:
                regime = "standard"

        allowed = regime in set(_DIAKRINO_REGIME_CONDITIONAL_ALLOWED_REGIMES)
        meta["diakrino_regime_conditional_regime"] = str(regime)
        meta["diakrino_regime_conditional_allowed"] = bool(allowed)
        if allowed:
            meta["diakrino_regime_conditional_reason"] = "allowed_regime"
            out.update(meta)
            return out

        methods_after = tuple(
            m for m in methods_before if m not in set(_DIAKRINO_REGIME_CONDITIONAL_METHODS)
        )
        removed = [m for m in methods_before if m in set(_DIAKRINO_REGIME_CONDITIONAL_METHODS)]
        if not removed:
            meta["diakrino_regime_conditional_reason"] = "no_diakrino_methods"
            out.update(meta)
            return out

        source = str(out.get("enabled_methods_source", "config") or "config")
        out["enabled_methods"] = methods_after
        out["enabled_methods_source"] = f"{source}+diakrino_regime_conditional:{regime}"
        meta.update(
            {
                "diakrino_regime_conditional_applied": True,
                "diakrino_regime_conditional_reason": "removed_non_hdlss",
                "diakrino_regime_conditional_removed_methods": list(removed),
                "diakrino_regime_conditional_enabled_methods_after": list(methods_after),
            }
        )
        out.update(meta)
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
                getattr(self.config, "regime_gating_min_samples_per_class", 7.0) or 7.0
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
            return self._apply_diakrino_regime_conditional_policy(policy, X_ref, y_ref)

        lockout_enabled = bool(getattr(self.config, "tier_lockout_enabled", False))
        routing_enabled = bool(getattr(self.config, "tier_routing_enabled", False))
        if not lockout_enabled and not routing_enabled:
            return self._apply_diakrino_regime_conditional_policy(policy, X_ref, y_ref)

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
                return self._apply_diakrino_regime_conditional_policy(policy, X_ref, y_ref)

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
                return self._apply_diakrino_regime_conditional_policy(policy, X_ref, y_ref)

        # No policy applied but expose the most specific resolved tier context.
        if lockout_enabled:
            policy["tier_policy_resolved_tier"] = lock_tier
            policy["tier_policy_source"] = lock_source
            policy["tier_policy_meta_features"] = lock_meta
        elif routing_enabled:
            policy["tier_policy_resolved_tier"] = route_tier
            policy["tier_policy_source"] = route_source
            policy["tier_policy_meta_features"] = route_meta
        return self._apply_diakrino_regime_conditional_policy(policy, X_ref, y_ref)

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
        n_features = int(X_train.shape[1])
        mode = str(
            getattr(self.config, "diakrino_prefilter_mode", "protected_union") or "protected_union"
        ).strip().lower()
        enabled = bool(getattr(self.config, "diakrino_prefilter_enabled", False))
        lam = float(getattr(self.config, "diakrino_prefilter_lambda", 0.0) or 0.0)
        top_k = int(n_features)
        combined: Optional[np.ndarray] = None
        classical_reason = "rank_prefilter_disabled"
        classical_keep_idx = np.arange(n_features, dtype=int)

        if bool(self.config.use_rank_prefilter) and n_features > 1:
            if top_k_override is not None:
                top_k = int(max(1, top_k_override))
            else:
                top_k = (
                    n_features
                    if self.config.prefilter_top_k is None
                    else int(max(1, self.config.prefilter_top_k))
                )
            if bool(getattr(self.config, "prefilter_adaptive_top_k", False)) and top_k_override is None:
                try:
                    try:
                        from tabnetics.datasets.meta_features import extract_meta_features
                    except Exception:
                        from tabnetics.datasets.meta_features import extract_meta_features  # type: ignore
                    meta = extract_meta_features(
                        np.asarray(X_train, dtype=float),
                        np.asarray(y_train).ravel(),
                        expanded=True,
                    )
                    top_k = _adaptive_prefilter_top_k(
                        base_top_k=int(top_k),
                        n_features=int(n_features),
                        meta_features=meta,
                        scaling_factor=float(
                            getattr(self.config, "prefilter_adaptive_top_k_scaling", 0.5) or 0.5
                        ),
                    )
                except Exception:
                    pass
            if top_k < n_features:
                mi = self._safe_mi(X_train, y_train, seed)
                fs = self._safe_fscore(X_train, y_train)
                svm = self._safe_linear_svm_scores(X_train, y_train, seed)
                combined = (
                    0.45 * self._normalize01(mi)
                    + 0.35 * self._normalize01(fs)
                    + 0.20 * self._normalize01(svm)
                )
                classical_keep_idx = np.argsort(combined)[::-1][:top_k]
                classical_keep_idx = np.array(
                    sorted(set(int(i) for i in classical_keep_idx)), dtype=int
                )
                classical_reason = "classical_rank_prefilter"
            else:
                classical_reason = "classical_pool_covers_all_features"
        elif n_features <= 1:
            classical_reason = "insufficient_features"

        if not enabled or lam <= 0.0:
            self._record_diakrino_prefilter_state(
                mode=mode,
                reason="disabled" if not enabled else "nonpositive_lambda",
                initial_original_indices=classical_keep_idx,
                classical_pool_original_indices=classical_keep_idx,
            )
            return (
                X_train[:, classical_keep_idx],
                X_test[:, classical_keep_idx],
                classical_keep_idx,
            )

        if mode == "legacy_fixed_budget_blend":
            if combined is None:
                self._record_diakrino_prefilter_state(
                    mode=mode,
                    reason=f"legacy_noop:{classical_reason}",
                    initial_original_indices=classical_keep_idx,
                    classical_pool_original_indices=classical_keep_idx,
                )
                return (
                    X_train[:, classical_keep_idx],
                    X_test[:, classical_keep_idx],
                    classical_keep_idx,
                )
            legacy_scores = self._apply_diakrino_legacy_fixed_budget_blend(
                combined,
                n_features=n_features,
            )
            keep_idx = np.argsort(legacy_scores)[::-1][:top_k]
            keep_idx = np.array(sorted(set(int(i) for i in keep_idx)), dtype=int)
            self._record_diakrino_prefilter_state(
                mode=mode,
                reason="legacy_fixed_budget_blend_can_evict",
                initial_original_indices=keep_idx,
                classical_pool_original_indices=classical_keep_idx,
                applied=not np.array_equal(keep_idx, classical_keep_idx),
                legacy_evicted_original_indices=np.setdiff1d(
                    classical_keep_idx, keep_idx, assume_unique=True
                ),
            )
            return X_train[:, keep_idx], X_test[:, keep_idx], keep_idx

        max_extras = int(max(0, getattr(self.config, "diakrino_prefilter_max_extras", 0) or 0))
        classical_set = set(int(i) for i in classical_keep_idx.tolist())
        shadow_set = set(
            int(i)
            for i in tuple(
                getattr(self.config, "diakrino_prefilter_shadow_probe_indices", tuple()) or tuple()
            )
        )
        protection_reason = "protected_union_active"
        ranked_candidates: List[int] = []
        outside_candidates: List[int] = []
        shadow_candidates: List[int] = []
        agreement_count = 0
        agreement_denominator = 0
        eligible_outside_candidate_count = 0
        valid_finite_candidate_count = 0
        budget_scan_count = 0
        budget_scan_exhausted = False

        if not bool(getattr(self, "_diakrino_prefilter_identity_available", True)):
            protection_reason = (
                "original_identity_unavailable:"
                f"{getattr(self, '_diakrino_prefilter_identity_reason', 'unknown')}"
            )
        elif max_extras <= 0:
            protection_reason = "zero_extra_budget"
        else:
            scores = self._resolve_diakrino_prefilter_scores(int(n_features))
            if scores is None:
                protection_reason = "missing_or_misaligned_scores"
            else:
                raw_scores = np.asarray(scores, dtype=float).ravel()
                finite_scores = raw_scores[np.isfinite(raw_scores)]
                valid_finite_candidate_count = int(finite_scores.size)
                if finite_scores.size <= 1 or float(np.ptp(finite_scores)) <= 1e-12:
                    protection_reason = "noninformative_scores"
                else:
                    finite_indices = np.flatnonzero(np.isfinite(raw_scores))
                    finite_order = np.argsort(
                        -raw_scores[finite_indices], kind="mergesort"
                    )
                    diakrino_ranked = [int(i) for i in finite_indices[finite_order].tolist()]
                    admissible_ranked = [int(i) for i in diakrino_ranked if int(i) not in shadow_set]
                    eligible_outside = [
                        int(i) for i in admissible_ranked if int(i) not in classical_set
                    ]
                    eligible_outside_candidate_count = int(len(eligible_outside))
                    outside_candidates = eligible_outside[:max_extras]
                    if outside_candidates:
                        last_admitted = int(outside_candidates[-1])
                        budget_scan_count = int(diakrino_ranked.index(last_admitted) + 1)
                    elif diakrino_ranked:
                        budget_scan_count = int(len(diakrino_ranked))
                    # "Exhausted" means the ranking ended before the outside-extra
                    # budget was filled, not merely that the final rank was inspected.
                    budget_scan_exhausted = bool(
                        len(outside_candidates) < max_extras
                        and budget_scan_count >= len(diakrino_ranked)
                    )
                    shadow_candidates = [
                        int(i)
                        for i in diakrino_ranked[:budget_scan_count]
                        if int(i) in shadow_set
                    ]
                    union_set = classical_set | set(outside_candidates)
                    ranked_candidates = [
                        int(i) for i in admissible_ranked if int(i) in union_set
                    ]
                    agreement_denominator = int(min(top_k, len(diakrino_ranked)))
                    agreement_count = int(
                        len(classical_set & set(diakrino_ranked[:agreement_denominator]))
                    )

        keep_idx = np.array(sorted(classical_set | set(outside_candidates)), dtype=int)
        self._record_diakrino_prefilter_state(
            mode=mode,
            reason=str(protection_reason),
            initial_original_indices=keep_idx,
            classical_pool_original_indices=classical_keep_idx,
            diakrino_extra_original_indices=np.asarray(outside_candidates, dtype=int),
            diakrino_ranked_candidate_original_indices=np.asarray(ranked_candidates, dtype=int),
            applied=bool(ranked_candidates),
            protection_active=True,
            diakrino_addition_budget=max_extras,
            diakrino_classical_agreement_count=agreement_count,
            diakrino_classical_agreement_denominator=agreement_denominator,
            diakrino_eligible_outside_candidate_count=eligible_outside_candidate_count,
            diakrino_admitted_outside_candidate_count=len(outside_candidates),
            diakrino_valid_finite_candidate_count=valid_finite_candidate_count,
            diakrino_budget_scan_count=budget_scan_count,
            diakrino_budget_scan_exhausted=budget_scan_exhausted,
            shadow_probe_candidate_original_indices=np.asarray(shadow_candidates, dtype=int),
            shadow_probe_candidate_denominator=budget_scan_count,
        )

        return X_train[:, keep_idx], X_test[:, keep_idx], keep_idx

    def _diakrino_prefilter_runtime_enabled(self) -> bool:
        return bool(getattr(self.config, "diakrino_prefilter_enabled", False)) and float(
            getattr(self.config, "diakrino_prefilter_lambda", 0.0) or 0.0
        ) > 0.0

    def _record_diakrino_prefilter_state(
        self,
        *,
        mode: str,
        reason: str,
        initial_original_indices: np.ndarray,
        classical_pool_original_indices: np.ndarray,
        diakrino_extra_original_indices: Optional[np.ndarray] = None,
        diakrino_ranked_candidate_original_indices: Optional[np.ndarray] = None,
        applied: bool = False,
        protection_active: bool = False,
        diakrino_addition_budget: int = 0,
        diakrino_classical_agreement_count: int = 0,
        diakrino_classical_agreement_denominator: int = 0,
        diakrino_eligible_outside_candidate_count: int = 0,
        diakrino_admitted_outside_candidate_count: int = 0,
        diakrino_valid_finite_candidate_count: int = 0,
        diakrino_budget_scan_count: int = 0,
        diakrino_budget_scan_exhausted: bool = False,
        shadow_probe_candidate_original_indices: Optional[np.ndarray] = None,
        shadow_probe_candidate_denominator: int = 0,
        legacy_evicted_original_indices: Optional[np.ndarray] = None,
    ) -> None:
        initial = np.asarray(initial_original_indices, dtype=int).ravel()
        classical = np.asarray(classical_pool_original_indices, dtype=int).ravel()
        extras = np.asarray(
            [] if diakrino_extra_original_indices is None else diakrino_extra_original_indices,
            dtype=int,
        ).ravel()
        ranked_candidates = np.asarray(
            []
            if diakrino_ranked_candidate_original_indices is None
            else diakrino_ranked_candidate_original_indices,
            dtype=int,
        ).ravel()
        shadow = np.asarray(
            []
            if shadow_probe_candidate_original_indices is None
            else shadow_probe_candidate_original_indices,
            dtype=int,
        ).ravel()
        legacy_evicted = np.asarray(
            [] if legacy_evicted_original_indices is None else legacy_evicted_original_indices,
            dtype=int,
        ).ravel()
        agreement_denominator = int(max(0, diakrino_classical_agreement_denominator))
        self._last_diakrino_prefilter_state = {
            "schema_version": "1.0",
            "configured": bool(getattr(self.config, "diakrino_prefilter_enabled", False)),
            "mode": str(mode),
            "applied": bool(applied),
            "protection_active": bool(protection_active),
            "reason": str(reason),
            "original_identity_available": bool(
                getattr(self, "_diakrino_prefilter_identity_available", True)
            ),
            "original_identity_reason": str(
                getattr(self, "_diakrino_prefilter_identity_reason", "original_feature_indices")
            ),
            "initial_original_indices": tuple(int(i) for i in initial.tolist()),
            "active_original_indices": tuple(int(i) for i in initial.tolist()),
            "classical_pool_original_indices": tuple(int(i) for i in classical.tolist()),
            "diakrino_extra_original_indices": tuple(int(i) for i in extras.tolist()),
            "diakrino_ranked_candidate_original_indices": tuple(
                int(i) for i in ranked_candidates.tolist()
            ),
            "diakrino_addition_budget": int(max(0, diakrino_addition_budget)),
            "diakrino_classical_agreement_count": int(max(0, diakrino_classical_agreement_count)),
            "diakrino_classical_agreement_denominator": agreement_denominator,
            "diakrino_classical_agreement_rate": (
                float(diakrino_classical_agreement_count / agreement_denominator)
                if agreement_denominator > 0
                else float("nan")
            ),
            "diakrino_eligible_outside_candidate_count": int(
                max(0, diakrino_eligible_outside_candidate_count)
            ),
            "diakrino_admitted_outside_candidate_count": int(
                max(0, diakrino_admitted_outside_candidate_count)
            ),
            "diakrino_valid_finite_candidate_count": int(
                max(0, diakrino_valid_finite_candidate_count)
            ),
            "diakrino_budget_scan_count": int(max(0, diakrino_budget_scan_count)),
            "diakrino_budget_scan_exhausted": bool(diakrino_budget_scan_exhausted),
            "shadow_probe_candidate_original_indices": tuple(int(i) for i in shadow.tolist()),
            "shadow_probe_candidate_count": int(shadow.size),
            "shadow_probe_candidate_denominator": int(
                max(0, shadow_probe_candidate_denominator)
            ),
            "shadow_probe_candidate_fraction": (
                float(shadow.size / int(shadow_probe_candidate_denominator))
                if int(shadow_probe_candidate_denominator) > 0
                else float("nan")
            ),
            "legacy_evicted_original_indices": tuple(int(i) for i in legacy_evicted.tolist()),
        }

    def _update_diakrino_prefilter_active_mask(self, keep_mask: np.ndarray) -> None:
        state = dict(getattr(self, "_last_diakrino_prefilter_state", {}) or {})
        active = np.asarray(state.get("active_original_indices", tuple()), dtype=int).ravel()
        mask = np.asarray(keep_mask, dtype=bool).ravel()
        if active.size != mask.size:
            return
        removed = active[~mask]
        state["active_original_indices"] = tuple(int(i) for i in active[mask].tolist())
        state["variance_floor_removed_original_indices"] = tuple(
            int(i) for i in removed.tolist()
        )
        state["variance_floor_removed_count"] = int(removed.size)
        self._last_diakrino_prefilter_state = state

    def _diakrino_protected_selection_context(self, n_features: int) -> Optional[Dict[str, Any]]:
        state = dict(getattr(self, "_last_diakrino_prefilter_state", {}) or {})
        if str(state.get("mode", "")) != "protected_union" or not bool(
            state.get("protection_active", False)
        ):
            return None
        active = np.asarray(state.get("active_original_indices", tuple()), dtype=int).ravel()
        if active.size != int(n_features):
            return None
        active_lookup = {int(original): int(local) for local, original in enumerate(active.tolist())}
        classical_original = [
            int(i)
            for i in tuple(state.get("classical_pool_original_indices", tuple()) or tuple())
            if int(i) in active_lookup
        ]
        ranked_candidates_original = [
            int(i)
            for i in tuple(
                state.get("diakrino_ranked_candidate_original_indices", tuple()) or tuple()
            )
            if int(i) in active_lookup
        ]
        if not classical_original:
            return None
        return {
            "state": state,
            "active_original_indices": active,
            "classical_original_indices": np.asarray(classical_original, dtype=int),
            "classical_local_indices": np.asarray(
                [active_lookup[int(i)] for i in classical_original], dtype=int
            ),
            "ranked_candidate_original_indices": np.asarray(
                ranked_candidates_original, dtype=int
            ),
            "ranked_candidate_local_indices": np.asarray(
                [active_lookup[int(i)] for i in ranked_candidates_original], dtype=int
            ),
            "active_local_by_original": active_lookup,
        }

    def _diakrino_closeout_admitted_pairs(
        self,
        *,
        X_support: np.ndarray,
        y_support: np.ndarray,
        seed: int,
        protected_ctx: Mapping[str, Any],
        protected_core_indices: np.ndarray,
    ) -> Optional[Tuple[List[Tuple[int, int]], Dict[str, Any]]]:
        """Apply an explicitly staged #229 decision over the immutable core.

        Evidence is staged by the canonical runner only after strict sidecar
        validation.  Absence is a strict no-op, preserving P1/P2 and defaults.
        """

        evidence = getattr(self, "_diakrino_closeout_evidence", None)
        if not isinstance(evidence, Mapping):
            return None
        from tabnetics.feature_selection.diakrino_closeout import (
            DIAKRINO_CLOSEOUT_ARMS,
            DiakrinoCloseoutConfig,
            DiakrinoCloseoutError,
            dynamic_addition_budget,
            finalize_closeout_decision,
            jmi_admit,
            matched_control_additions,
            native_null_proposals,
            support_only_classical_scores,
        )

        arm = str(evidence.get("arm") or "").strip().lower()
        if arm not in DIAKRINO_CLOSEOUT_ARMS:
            raise DiakrinoCloseoutError(f"unsupported staged closeout arm: {arm!r}")
        active_original = np.asarray(
            protected_ctx["active_original_indices"], dtype=int
        ).ravel()
        width = int(evidence.get("n_features", -1))
        if width <= 0 or np.any(active_original < 0) or np.any(active_original >= width):
            raise DiakrinoCloseoutError("closeout evidence feature identity is misaligned")

        def active_vector(name: str) -> np.ndarray:
            values = np.asarray(evidence.get(name), dtype=np.float64).reshape(-1)
            if values.shape != (width,) or not np.all(np.isfinite(values)):
                raise DiakrinoCloseoutError(
                    f"closeout evidence {name} is missing, non-finite, or misaligned"
                )
            return values[active_original]

        real_rank = active_vector("real_rank")
        core = tuple(int(value) for value in np.asarray(protected_core_indices).ravel())
        budget = dynamic_addition_budget(len(core))
        diagnostics: Dict[str, Any] = {
            "closeout_schema_version": "diakrino_closeout_decision_v1",
            "closeout_arm": arm,
        }
        additions: Tuple[int, ...]
        if arm in {
            "protected_native_null_abstain",
            "protected_native_null_jmi",
        }:
            raw_views = np.asarray(evidence.get("ranks_by_view"), dtype=np.float64)
            if raw_views.ndim != 2 or raw_views.shape[1] != width:
                raise DiakrinoCloseoutError("closeout view ranks are missing or misaligned")
            threshold_values = evidence.get("thresholds")
            if not isinstance(threshold_values, Mapping):
                raise DiakrinoCloseoutError("closeout thresholds are missing")
            threshold_keys = {
                "shadow_margin_min",
                "label_null_margin_min",
                "rank_std_max",
                "selected_set_stability_min",
                "proposal_pool_multiplier",
                "discretization_bins",
            }
            if set(threshold_values) != threshold_keys:
                raise DiakrinoCloseoutError("closeout thresholds are incomplete or ambiguous")
            config = DiakrinoCloseoutConfig(**dict(threshold_values))
            proposals, null_diagnostics = native_null_proposals(
                real_rank=real_rank,
                shadow_rank=active_vector("shadow_rank"),
                label_null_rank=active_vector("label_null_rank"),
                rank_std=active_vector("rank_std"),
                ranks_by_view=raw_views[:, active_original],
                protected_core=core,
                config=config,
            )
            diagnostics.update(null_diagnostics)
            diagnostics["proposal_original_indices"] = [
                int(active_original[index]) for index in proposals
            ]
            if arm == "protected_native_null_jmi":
                additions, ledger = jmi_admit(
                    np.asarray(X_support, dtype=np.float64),
                    np.asarray(y_support).reshape(-1),
                    protected_core=core,
                    proposals=proposals,
                    real_rank=real_rank,
                    budget=budget,
                    discretization_bins=int(config.discretization_bins),
                )
                diagnostics["jmi_ledger"] = [dict(row) for row in ledger]
            else:
                additions = tuple(proposals[:budget])
        else:
            realized = evidence.get("p4_realized_additions")
            if isinstance(realized, bool) or not isinstance(realized, (int, np.integer)):
                raise DiakrinoCloseoutError(
                    "matched controls require exact P4 realized additions"
                )
            if int(realized) < 0 or int(realized) > budget:
                raise DiakrinoCloseoutError("P4 realized additions violate dynamic budget")
            classical = support_only_classical_scores(
                np.asarray(X_support, dtype=np.float64),
                np.asarray(y_support).reshape(-1),
            )
            additions = matched_control_additions(
                arm,
                protected_core=core,
                realized_additions=int(realized),
                n_features=int(active_original.size),
                seed=int(seed),
                classical_scores=classical,
                diakrino_ranks=real_rank,
            )
            diagnostics.update(
                {
                    "p4_realized_additions": int(realized),
                    "control_budget_matched": len(additions) == int(realized),
                    "classical_score_source": "support_mi_anova_average_ranks_v1",
                }
            )
        decision = finalize_closeout_decision(
            arm,
            protected_core=core,
            additions=additions,
            n_features=int(active_original.size),
            reason="native_null_abstain" if not additions else "closeout_admitted",
            diagnostics=diagnostics,
        )
        diagnostics.update(
            {
                "abstained": decision.abstained,
                "fallback_exact": decision.fallback_exact,
                "realized_additions": decision.realized_additions,
                "addition_budget": decision.addition_budget,
            }
        )
        local_by_original = {
            int(original): int(local)
            for local, original in enumerate(active_original.tolist())
        }
        admitted_pairs = [
            (int(active_original[local]), int(local)) for local in decision.additions
        ]
        if any(original not in local_by_original for original, _ in admitted_pairs):
            raise AssertionError("closeout decision escaped the active feature identity")
        return admitted_pairs, diagnostics

    def _resolve_diakrino_prefilter_scores(self, n_features: int) -> Optional[np.ndarray]:
        """Return a dense per-feature DIAKRINO relevance vector aligned to the prefilter's
        columns (length ``n_features``), or ``None`` when unavailable/misaligned.

        Prefers the caller-supplied ``external_feature_scores`` channel (staged on the
        instance by :meth:`run_pre_split`); falls back to a persisted sidecar.  Returns
        ``None`` on any width mismatch so a downstream pruning change can never silently
        misalign scores to the wrong features.
        """
        ext = getattr(self, "_diakrino_external_feature_scores", None)
        if ext is not None:
            ext = np.asarray(ext, dtype=float).ravel()
            return ext if ext.shape[0] == int(n_features) else None
        path = str(getattr(self.config, "diakrino_sidecar_path", "") or "")
        dataset_id = (
            str(getattr(self.config, "diakrino_sidecar_dataset_id", "") or "")
            or str(getattr(self, "_active_diakrino_dataset_id", "") or "")
            or None
        )
        if not path:
            return None
        try:
            from tabnetics.feature_selection.diakrino_sidecar import DiakrinoSidecar
            sc = DiakrinoSidecar.load(path, dataset_id=dataset_id)
            if sc is None or sc.n_features != int(n_features):
                return None
            col = str(getattr(self.config, "diakrino_prefilter_score_column", "prior_logit"))
            vec = sc.scalar_scores(col, calibrate="chunk_zscore")
            return vec if (vec is not None and vec.shape[0] == int(n_features)) else None
        except Exception:
            return None

    def _apply_diakrino_legacy_fixed_budget_blend(
        self,
        combined: np.ndarray,
        *,
        n_features: int,
    ) -> np.ndarray:
        """Reproduce the retired DIAKRINO blend that can evict fixed-budget incumbents."""
        if not bool(getattr(self.config, "diakrino_prefilter_enabled", False)):
            return combined
        lam = float(getattr(self.config, "diakrino_prefilter_lambda", 0.0) or 0.0)
        if lam <= 0.0:
            return combined
        scores = self._resolve_diakrino_prefilter_scores(int(n_features))
        if scores is None:
            return combined
        lam = min(1.0, lam)
        return (1.0 - lam) * np.asarray(combined, dtype=float) + lam * self._normalize01(scores)

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
    def _safe_balanced_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> float:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        weights = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).ravel()
        )
        labels = np.unique(y_true_arr)
        if labels.size == 0:
            return 0.0
        if labels.size == 1:
            if weights is None:
                return float(np.mean(y_pred_arr == labels[0]))
            return float(np.average(y_pred_arr == labels[0], weights=weights))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            if weights is None:
                return float(
                    recall_score(
                        y_true_arr,
                        y_pred_arr,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                )
            return float(
                recall_score(
                    y_true_arr,
                    y_pred_arr,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                    sample_weight=weights,
                )
            )

    @staticmethod
    def _safe_macro_f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> float:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        weights = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).ravel()
        )
        labels = np.unique(y_true_arr)
        if labels.size == 0:
            return 0.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            if weights is None:
                return float(
                    f1_score(
                        y_true_arr,
                        y_pred_arr,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                )
            return float(
                f1_score(
                    y_true_arr,
                    y_pred_arr,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                    sample_weight=weights,
                )
            )

    def _inspect_selected_classifier(
        self,
        model: Any,
        X_probe: Any | None,
        *,
        model_cv_meta: Dict[str, Any],
        effective_model_name: str,
        requested_device: Optional[str] = None,
        sample_weight_requested: bool = False,
        sample_weight_routed_observation: str = "not_requested",
    ) -> FittedClassifierDescriptor:
        identity = dict(model_cv_meta.get("classification_selected_identity") or {})
        if not identity:
            raise RuntimeError("classifier_selection_identity_missing")
        canonical_name = identity.get("canonical_name")
        cls_cfg = self._classification_cfg()
        native_diakrino_requested = str(
            getattr(cls_cfg, "tabentics_diakrino_device", "auto") or "auto"
        )
        members = [dict(value) for value in tuple(identity.get("members") or ())]
        for member in members:
            if member.get("canonical_name") == "tabentics_diakrino":
                member["requested_device"] = native_diakrino_requested
        if members:
            identity["members"] = members
        if requested_device is None and canonical_name == "tabentics_diakrino":
            requested_device = native_diakrino_requested
        if requested_device is None and canonical_name == "tabpfn":
            configured_device = getattr(
                model, "_tabnetics_requested_device", None
            )
            if configured_device is None:
                get_params = getattr(model, "get_params", None)
                if callable(get_params):
                    try:
                        configured_device = dict(
                            get_params(deep=False) or {}
                        ).get("device")
                    except Exception:
                        configured_device = None
            if configured_device is not None:
                requested_device = str(configured_device)
        probe_payload = (
            None
            if X_probe is None
            else np.asarray(X_probe, dtype=float)[: min(16, len(X_probe))]
        )
        return inspect_fitted_classifier(
            model,
            canonical_name=canonical_name,
            registry_anchor_name=identity.get("registry_anchor_name"),
            backend=str(
                model_cv_meta.get("classification_backend_used", "unknown")
            ),
            requested_name=identity.get("requested_name"),
            outward_name=identity.get("outward_name"),
            effective_model_name=str(effective_model_name),
            selection_identity=identity,
            config={
                "enable_svc_probability": bool(
                    getattr(
                        self._classification_cfg(),
                        "enable_svc_probability",
                        False,
                    )
                ),
                "tabentics_diakrino_calibrate_probabilities": bool(
                    getattr(
                        self._classification_cfg(),
                        "tabentics_diakrino_calibrate_probabilities",
                        True,
                    )
                ),
            },
            probe_X=probe_payload,
            sample_weight_requested=bool(sample_weight_requested),
            sample_weight_routed_observation=str(
                sample_weight_routed_observation
            ),
            requested_device=requested_device,
        )

    @staticmethod
    def _fitted_descriptor_metadata(
        *,
        base_descriptor: FittedClassifierDescriptor,
        final_descriptor: FittedClassifierDescriptor,
    ) -> Dict[str, Any]:
        final = final_descriptor.to_dict()
        return {
            "classification_base_fitted_descriptor": base_descriptor.to_dict(),
            "classification_final_fitted_descriptor": final,
            "classification_fitted_probability_kind": str(
                final["fitted_probability_kind"]
            ),
            "classification_fitted_probability_source": str(
                final["probability_source"]
            ),
            "classification_fitted_class_order": list(final["class_order"]),
            "classification_fitted_matrix_observation": str(
                final["matrix_observation"]
            ),
            "classification_fitted_matrix_reason": str(final["matrix_reason"]),
            "classification_fitted_argmax_observation": str(
                final["argmax_observation"]
            ),
            "classification_fitted_serialization_observation": str(
                final["pickle_observation"]
            ),
        }

    @staticmethod
    def _safe_log_loss_with_meta(
        model: Any,
        X_eval: np.ndarray,
        y_true: np.ndarray,
        descriptor: Optional[FittedClassifierDescriptor],
        *,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "log_loss_available": False,
            "log_loss_skip_reason": "fitted_descriptor_required",
            "log_loss_probability_kind": "unknown",
            "log_loss_probability_source": "unavailable",
            "log_loss_class_order": [],
            "log_loss_class_alignment": "failed",
        }
        if descriptor is None:
            return float("nan"), meta
        classes = np.asarray(getattr(model, "classes_", ())).ravel()
        result = extract_probability_matrix(
            model,
            np.asarray(X_eval, dtype=float),
            descriptor,
            requirement=ProbabilityRequirement.MATRIX,
            target_classes=classes,
        )
        meta.update(result.metadata(prefix="log_loss"))
        meta["log_loss_skip_reason"] = str(result.reason)
        if not result.available or result.matrix is None:
            return float("nan"), meta
        try:
            if sample_weight is None:
                value = float(
                    sklearn_log_loss(
                        np.asarray(y_true),
                        np.asarray(result.matrix, dtype=float),
                        labels=classes,
                    )
                )
            else:
                value = float(
                    sklearn_log_loss(
                        np.asarray(y_true),
                        np.asarray(result.matrix, dtype=float),
                        labels=classes,
                        sample_weight=np.asarray(sample_weight, dtype=float).ravel(),
                    )
                )
        except Exception as exc:
            meta["log_loss_available"] = False
            meta["log_loss_skip_reason"] = f"metric:{type(exc).__name__}"
            return float("nan"), meta
        meta["log_loss_available"] = True
        meta["log_loss_skip_reason"] = "ok"
        return value, meta

    @staticmethod
    def _safe_log_loss(
        model: Any,
        X_eval: np.ndarray,
        y_true: np.ndarray,
        descriptor: Optional[FittedClassifierDescriptor] = None,
        *,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> float:
        value, _ = DistributionFeatureSelectionPipeline._safe_log_loss_with_meta(
            model,
            X_eval,
            y_true,
            descriptor,
            sample_weight=sample_weight,
        )
        return float(value)

    def _safe_probability_calibration_metrics(
        self,
        model: Any,
        X_eval: np.ndarray,
        y_true: np.ndarray,
        descriptor: Optional[FittedClassifierDescriptor] = None,
        *,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "calibration_metrics_available": False,
            "calibration_brier": float("nan"),
            "calibration_ece": float("nan"),
            "calibration_n_eval": 0,
            "calibration_n_classes": 0,
            "calibration_skip_reason": "unavailable",
            "calibration_probability_kind": "unknown",
            "calibration_probability_source": "unavailable",
            "calibration_class_order": [],
            "calibration_class_alignment": "failed",
            "calibration_sample_weight_consumed": bool(sample_weight is not None),
        }
        if descriptor is None:
            out["calibration_skip_reason"] = "fitted_descriptor_required"
            return out
        y_arr = np.asarray(y_true).ravel()
        weights = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).ravel()
        )
        if weights is not None:
            if int(weights.size) != int(y_arr.size):
                out["calibration_skip_reason"] = "sample_weight_length_mismatch"
                return out
            if (
                not np.all(np.isfinite(weights))
                or np.any(weights < 0.0)
                or not float(np.sum(weights)) > 0.0
            ):
                out["calibration_skip_reason"] = "sample_weight_invalid"
                return out
        classes = np.asarray(getattr(model, "classes_", ())).ravel()
        if y_arr.size == 0 or classes.size < 2:
            out["calibration_skip_reason"] = "degenerate_labels"
            return out
        matrix_result = extract_probability_matrix(
            model,
            np.asarray(X_eval, dtype=float),
            descriptor,
            requirement=ProbabilityRequirement.MATRIX,
            target_classes=classes,
        )
        matrix_meta = matrix_result.metadata(prefix="calibration")
        out["calibration_probability_kind"] = matrix_meta[
            "calibration_probability_kind"
        ]
        out["calibration_probability_source"] = matrix_meta[
            "calibration_probability_source"
        ]
        out["calibration_class_order"] = matrix_meta["calibration_class_order"]
        out["calibration_class_alignment"] = matrix_meta[
            "calibration_class_alignment"
        ]
        if not matrix_result.available or matrix_result.matrix is None:
            out["calibration_skip_reason"] = str(matrix_result.reason)
            return out
        aligned = np.asarray(matrix_result.matrix, dtype=float)
        if aligned.ndim != 2 or aligned.shape[0] != y_arr.size or aligned.shape[1] != classes.size:
            out["calibration_skip_reason"] = "invalid_probability_shape"
            return out
        finite_rows = np.all(np.isfinite(aligned), axis=1)
        if int(np.sum(finite_rows)) < 2:
            out["calibration_skip_reason"] = "insufficient_finite_probabilities"
            return out
        p = np.clip(aligned[finite_rows], 0.0, 1.0)
        row_sums = np.sum(p, axis=1, keepdims=True)
        row_sums[row_sums <= 1e-12] = 1.0
        p = p / row_sums
        y_valid = y_arr[finite_rows]
        weights_valid = None if weights is None else weights[finite_rows]
        if weights_valid is not None and not float(np.sum(weights_valid)) > 0.0:
            out["calibration_skip_reason"] = "sample_weight_zero_finite_mass"
            return out
        class_to_idx = {c: i for i, c in enumerate(classes.tolist())}
        true_idx = np.asarray([class_to_idx.get(c, -1) for c in y_valid], dtype=int)
        if np.any(true_idx < 0):
            out["calibration_skip_reason"] = "label_mapping_failed"
            return out
        one_hot = np.zeros_like(p, dtype=float)
        one_hot[np.arange(true_idx.size), true_idx] = 1.0
        if classes.size == 2:
            brier_values = (p[:, 1] - one_hot[:, 1]) ** 2
        else:
            brier_values = np.sum((p - one_hot) ** 2, axis=1)
        brier = (
            float(np.mean(brier_values))
            if weights_valid is None
            else float(np.average(brier_values, weights=weights_valid))
        )
        confidences = np.max(p, axis=1)
        correct = (np.argmax(p, axis=1) == true_idx).astype(float)
        ece = 0.0
        bins = np.linspace(0.0, 1.0, 16)
        for b in range(15):
            lo = float(bins[b])
            hi = float(bins[b + 1])
            if b == 14:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            n_bin = int(np.sum(mask))
            if n_bin <= 0:
                continue
            if weights_valid is None:
                ece += (float(n_bin) / float(confidences.size)) * abs(
                    float(np.mean(correct[mask])) - float(np.mean(confidences[mask]))
                )
            else:
                bin_weight = float(np.sum(weights_valid[mask]))
                if bin_weight <= 0.0:
                    continue
                ece += (bin_weight / float(np.sum(weights_valid))) * abs(
                    float(np.average(correct[mask], weights=weights_valid[mask]))
                    - float(
                        np.average(
                            confidences[mask], weights=weights_valid[mask]
                        )
                    )
                )
        out.update(
            {
                "calibration_metrics_available": True,
                "calibration_brier": float(brier),
                "calibration_ece": float(ece),
                "calibration_n_eval": int(y_valid.size),
                "calibration_n_classes": int(classes.size),
                "calibration_skip_reason": "ok",
            }
        )
        return out

    def _apply_posthoc_calibration(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        seed: int,
        base_descriptor: Optional[FittedClassifierDescriptor] = None,
        selection_identity: Optional[Dict[str, Any]] = None,
        backend: str = "unknown",
        fit_resampling_context: Optional[FitResamplingContext] = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        cls_cfg = self._classification_cfg()
        enabled = bool(getattr(cls_cfg, "posthoc_calibration_enabled", False))
        method = str(getattr(cls_cfg, "posthoc_calibration_method", "sigmoid") or "sigmoid").strip().lower()
        if method in {"platt", "temperature"}:
            method = "sigmoid"
        if method not in {"sigmoid", "isotonic"}:
            method = "sigmoid"
        fraction = float(np.clip(float(getattr(cls_cfg, "posthoc_calibration_fraction", 0.20) or 0.20), 0.05, 0.50))
        min_cal = int(max(2, int(getattr(cls_cfg, "posthoc_calibration_min_calibration", 20) or 20)))
        refine_stop = bool(getattr(cls_cfg, "posthoc_calibration_refinement_stopping", True))
        meta: Dict[str, Any] = {
            "classifier_posthoc_calibration_enabled": bool(enabled),
            "classifier_posthoc_calibration_applied": False,
            "classifier_posthoc_calibration_method": str(method),
            "classifier_posthoc_calibration_fraction": float(fraction),
            "classifier_posthoc_calibration_min_calibration": int(min_cal),
            "classifier_posthoc_calibration_refinement_stopping": bool(refine_stop),
            "classifier_posthoc_calibration_skip_reason": "disabled",
            "classifier_posthoc_calibration_size": 0,
            "classifier_posthoc_calibration_fit_size": int(np.asarray(y_train).ravel().size),
            "classifier_posthoc_calibration_base_brier": float("nan"),
            "classifier_posthoc_calibration_base_ece": float("nan"),
            "classifier_posthoc_calibration_calibrated_brier": float("nan"),
            "classifier_posthoc_calibration_calibrated_ece": float("nan"),
            "classifier_posthoc_calibration_probability_kind": "unknown",
            "classifier_posthoc_calibration_probability_source": "unavailable",
            "classifier_posthoc_calibration_sample_weight_requested": bool(
                sample_weight is not None
            ),
            "classifier_posthoc_calibration_sample_weight_consumed": False,
            "classifier_posthoc_calibration_sample_weight_route": "not_requested",
        }
        if not enabled:
            return model, meta
        x = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train).ravel()
        weights = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).ravel()
        )
        if weights is not None:
            if int(weights.size) != int(y.size):
                raise SampleWeightRoutingError(
                    "sample_weight_posthoc_calibration_length_mismatch"
                )
            if (
                not np.all(np.isfinite(weights))
                or np.any(weights < 0.0)
                or not float(np.sum(weights)) > 0.0
            ):
                raise SampleWeightRoutingError(
                    "sample_weight_posthoc_calibration_invalid"
                )
        if base_descriptor is None:
            meta["classifier_posthoc_calibration_skip_reason"] = (
                "fitted_descriptor_required"
            )
            return model, meta
        base_admission = extract_probability_matrix(
            model,
            x[: min(16, len(x))],
            base_descriptor,
            requirement=ProbabilityRequirement.MATRIX,
            target_classes=np.asarray(getattr(model, "classes_", ())).ravel(),
        )
        meta["classifier_posthoc_calibration_probability_kind"] = (
            base_descriptor.fitted_probability_kind.value
        )
        meta["classifier_posthoc_calibration_probability_source"] = str(
            base_descriptor.probability_source
        )
        if not base_admission.available:
            meta["classifier_posthoc_calibration_skip_reason"] = str(
                base_admission.reason
            )
            return model, meta
        classes, counts = np.unique(y, return_counts=True)
        if x.ndim != 2 or y.size != x.shape[0] or classes.size < 2:
            meta["classifier_posthoc_calibration_skip_reason"] = "degenerate_training_data"
            return model, meta
        if int(np.min(counts)) < 2:
            meta["classifier_posthoc_calibration_skip_reason"] = "insufficient_class_counts"
            return model, meta
        n_cal = int(round(float(y.size) * fraction))
        n_cal = int(max(min_cal, classes.size * 2, n_cal))
        if n_cal >= y.size:
            meta["classifier_posthoc_calibration_skip_reason"] = "insufficient_calibration_holdout"
            return model, meta
        if (
            isinstance(fit_resampling_context, FitResamplingContext)
            and fit_resampling_context.policy.kind != "iid"
        ):
            calibration_plan = resolve_holdout(
                fit_resampling_context,
                y,
                seed=int(seed),
                train_size=int(y.size - n_cal),
                purpose="posthoc_calibration",
            )
            fit_idx = np.asarray(
                calibration_plan.primary.train_indices, dtype=int
            )
            cal_idx = np.asarray(
                calibration_plan.primary.test_indices, dtype=int
            )
            meta["classifier_posthoc_calibration_resampling"] = (
                calibration_plan.to_metadata()
            )
            trace = getattr(self, "_active_resampling_plans", None)
            if isinstance(trace, dict):
                trace["posthoc_calibration"] = calibration_plan.to_metadata()
        else:
            try:
                splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=n_cal,
                    random_state=int(seed),
                )
                fit_idx, cal_idx = next(splitter.split(x, y))
            except Exception as exc:
                meta["classifier_posthoc_calibration_skip_reason"] = str(
                    type(exc).__name__
                )
                return model, meta
        fit_counts = np.unique(y[fit_idx], return_counts=True)[1]
        cal_classes, cal_counts = np.unique(y[cal_idx], return_counts=True)
        if cal_classes.size != classes.size or int(np.min(fit_counts)) < 1 or int(np.min(cal_counts)) < 1:
            meta["classifier_posthoc_calibration_skip_reason"] = "split_missing_class"
            return model, meta
        try:
            base = clone(model)
            if weights is None:
                base.fit(x[fit_idx], y[fit_idx])
                base_weight_route = "not_requested"
            else:
                base_weight_route = fit_estimator_with_sample_weight(
                    base,
                    x[fit_idx],
                    y[fit_idx],
                    weights[fit_idx],
                )
        except SampleWeightRoutingError:
            raise
        except Exception as exc:
            meta["classifier_posthoc_calibration_skip_reason"] = f"base_fit_{type(exc).__name__}"
            return model, meta
        identity = dict(selection_identity or {})
        try:
            fitted_base_descriptor = inspect_fitted_classifier(
                base,
                canonical_name=identity.get("canonical_name"),
                registry_anchor_name=identity.get("registry_anchor_name"),
                backend=str(backend),
                requested_name=identity.get("requested_name"),
                outward_name=identity.get("outward_name"),
                effective_model_name=identity.get("effective_model_name"),
                selection_identity=identity,
                config={
                    "enable_svc_probability": bool(
                        getattr(self._classification_cfg(), "enable_svc_probability", False)
                    )
                },
                probe_X=x[fit_idx][: min(16, len(fit_idx))],
                sample_weight_requested=weights is not None,
                sample_weight_routed_observation=base_weight_route,
                requested_device=base_descriptor.requested_device,
            )
        except Exception as exc:
            meta["classifier_posthoc_calibration_skip_reason"] = (
                f"base_descriptor_{type(exc).__name__}"
            )
            return model, meta
        base_metrics = self._safe_probability_calibration_metrics(
            base,
            x[cal_idx],
            y[cal_idx],
            fitted_base_descriptor,
            sample_weight=(None if weights is None else weights[cal_idx]),
        )
        try:
            if FrozenEstimator is not None:
                calibrator = CalibratedClassifierCV(
                    estimator=FrozenEstimator(base),  # type: ignore[operator]
                    method=method,
                )
            else:
                calibrator = CalibratedClassifierCV(
                    estimator=base,
                    method=method,
                    cv="prefit",
                )
            if weights is None:
                calibrator.fit(x[cal_idx], y[cal_idx])
                calibration_weight_route = "not_requested"
            else:
                calibration_weight_route = fit_estimator_with_sample_weight(
                    calibrator,
                    x[cal_idx],
                    y[cal_idx],
                    weights[cal_idx],
                )
        except SampleWeightRoutingError:
            raise
        except Exception as exc:
            meta["classifier_posthoc_calibration_skip_reason"] = f"calibrator_{type(exc).__name__}"
            return model, meta
        try:
            calibrated_descriptor = inspect_fitted_classifier(
                calibrator,
                canonical_name=identity.get("canonical_name"),
                registry_anchor_name=identity.get("registry_anchor_name"),
                backend=str(backend),
                requested_name=identity.get("requested_name"),
                outward_name=identity.get("outward_name"),
                effective_model_name=identity.get("effective_model_name"),
                selection_identity=identity,
                config={
                    "enable_svc_probability": bool(
                        getattr(self._classification_cfg(), "enable_svc_probability", False)
                    )
                },
                probe_X=x[cal_idx][: min(16, len(cal_idx))],
                sample_weight_requested=weights is not None,
                sample_weight_routed_observation=(
                    f"{base_weight_route}+{calibration_weight_route}"
                ),
                requested_device=base_descriptor.requested_device,
            )
        except Exception as exc:
            meta["classifier_posthoc_calibration_skip_reason"] = (
                f"calibrated_descriptor_{type(exc).__name__}"
            )
            return model, meta
        cal_metrics = self._safe_probability_calibration_metrics(
            calibrator,
            x[cal_idx],
            y[cal_idx],
            calibrated_descriptor,
            sample_weight=(None if weights is None else weights[cal_idx]),
        )
        base_brier = float(base_metrics.get("calibration_brier", float("nan")))
        base_ece = float(base_metrics.get("calibration_ece", float("nan")))
        cal_brier = float(cal_metrics.get("calibration_brier", float("nan")))
        cal_ece = float(cal_metrics.get("calibration_ece", float("nan")))
        meta.update(
            {
                "classifier_posthoc_calibration_size": int(cal_idx.size),
                "classifier_posthoc_calibration_fit_size": int(fit_idx.size),
                "classifier_posthoc_calibration_base_brier": float(base_brier),
                "classifier_posthoc_calibration_base_ece": float(base_ece),
                "classifier_posthoc_calibration_calibrated_brier": float(cal_brier),
                "classifier_posthoc_calibration_calibrated_ece": float(cal_ece),
                "classifier_posthoc_calibration_sample_weight_consumed": bool(
                    weights is not None
                ),
                "classifier_posthoc_calibration_sample_weight_route": (
                    "not_requested"
                    if weights is None
                    else f"{base_weight_route}+{calibration_weight_route}"
                ),
            }
        )
        if refine_stop:
            if np.isfinite(base_brier) and np.isfinite(cal_brier) and cal_brier > base_brier + 1e-12:
                meta["classifier_posthoc_calibration_skip_reason"] = "refinement_stopping_brier_worse"
                return model, meta
            if (
                not np.isfinite(base_brier)
                and np.isfinite(base_ece)
                and np.isfinite(cal_ece)
                and cal_ece > base_ece + 1e-12
            ):
                meta["classifier_posthoc_calibration_skip_reason"] = "refinement_stopping_ece_worse"
                return model, meta
        meta["classifier_posthoc_calibration_applied"] = True
        meta["classifier_posthoc_calibration_skip_reason"] = "ok"
        meta["classifier_posthoc_calibration_probability_kind"] = (
            calibrated_descriptor.fitted_probability_kind.value
        )
        meta["classifier_posthoc_calibration_probability_source"] = str(
            calibrated_descriptor.probability_source
        )
        return calibrator, meta

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
        sample_weight: Optional[Sequence[float]] = None,
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
        weights = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).ravel()
        )
        if weights is not None and (
            int(weights.size) != int(y_arr.size)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not float(np.sum(weights)) > 0.0
        ):
            out["roc_auc_source"] = "sample_weight_invalid"
            out["roc_metric_capabilities"]["selected_source"] = "sample_weight_invalid"
            return out
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
                weight_vec = None if weights is None else weights[finite]
                if np.unique(y_pos).size < 2:
                    return out
                if weight_vec is None:
                    auc_val = float(roc_auc_score(y_pos, score_vec))
                    fpr, tpr, _ = roc_curve(y_pos, score_vec)
                else:
                    auc_val = float(
                        roc_auc_score(y_pos, score_vec, sample_weight=weight_vec)
                    )
                    fpr, tpr, _ = roc_curve(
                        y_pos, score_vec, sample_weight=weight_vec
                    )
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
            weights_valid = None if weights is None else weights[finite_rows]
            y_bin = np.zeros((y_valid.size, classes.size), dtype=int)
            class_to_col = {c: i for i, c in enumerate(classes.tolist())}
            for row_idx, label in enumerate(y_valid):
                col = class_to_col.get(label, None)
                if col is not None:
                    y_bin[row_idx, int(col)] = 1
            if weights_valid is None:
                macro_auc = float(
                    roc_auc_score(
                        y_bin, s_valid, average="macro", multi_class="ovr"
                    )
                )
                weighted_auc = float(
                    roc_auc_score(
                        y_bin, s_valid, average="weighted", multi_class="ovr"
                    )
                )
                fpr, tpr, _ = roc_curve(y_bin.ravel(), s_valid.ravel())
                micro_auc = float(roc_auc_score(y_bin.ravel(), s_valid.ravel()))
            else:
                macro_auc = float(
                    roc_auc_score(
                        y_bin,
                        s_valid,
                        average="macro",
                        multi_class="ovr",
                        sample_weight=weights_valid,
                    )
                )
                weighted_auc = float(
                    roc_auc_score(
                        y_bin,
                        s_valid,
                        average="weighted",
                        multi_class="ovr",
                        sample_weight=weights_valid,
                    )
                )
                micro_weights = np.repeat(weights_valid, classes.size)
                fpr, tpr, _ = roc_curve(
                    y_bin.ravel(), s_valid.ravel(), sample_weight=micro_weights
                )
                micro_auc = float(
                    roc_auc_score(
                        y_bin.ravel(), s_valid.ravel(), sample_weight=micro_weights
                    )
                )
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
                class_weights = (
                    None if weights_valid is None else weights_valid[finite]
                )
                if np.unique(y_one).size < 2:
                    continue
                if class_weights is None:
                    cfpr, ctpr, _ = roc_curve(y_one, class_scores)
                    c_auc = float(roc_auc_score(y_one, class_scores))
                else:
                    cfpr, ctpr, _ = roc_curve(
                        y_one, class_scores, sample_weight=class_weights
                    )
                    c_auc = float(
                        roc_auc_score(
                            y_one, class_scores, sample_weight=class_weights
                        )
                    )
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
        sample_weight: Optional[Sequence[float]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        root = dict(root_meta or {})
        if not root:
            root = self._safe_roc_auc_bundle(
                model=model,
                X_eval=X_eval,
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
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
                sample_weight=sample_weight,
            )
            out[str(name)] = self._serialize_roc_meta(member_meta)
        return out

    def _finalize_native_categorical_stage2(
        self,
        *,
        candidate: Mapping[str, Any],
        y_train_full: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        variance_floor_meta: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Final-fit the admitted singleton without crossing the numeric bridge."""

        adapter = candidate.get("_native_stage2_adapter")
        if not isinstance(adapter, NativeCategoricalStage2Adapter):
            raise TypedInputCapabilityError(
                "native_stage2_adapter_missing",
                "Native categorical Stage-2 candidate lost its concrete adapter before final fit.",
            )
        route = dict(candidate.get("native_stage2_route") or {})
        if not bool(route.get("enabled", False)):
            raise TypedInputCapabilityError(
                "native_stage2_route_missing",
                "Native categorical Stage-2 finalization requires a persisted candidate route record.",
            )
        X_train_sel = candidate.get("X_train_sel")
        X_test_sel = candidate.get("X_test_sel")
        categorical_columns = tuple(
            str(value) for value in route.get("selected_categorical_columns", ())
        )
        if not categorical_columns:
            raise TypedInputCapabilityError(
                "native_stage2_no_selected_categorical_columns",
                "Native categorical Stage-2 finalization requires selected categorical columns.",
            )
        model = candidate.get("model")
        if model is None:
            raise TypedInputCapabilityError(
                "native_stage2_model_missing",
                "Native categorical Stage-2 candidate did not provide an admitted estimator instance.",
            )
        y_train_arr = np.asarray(y_train_full).ravel()
        y_test_arr = np.asarray(y_test).ravel()
        started = self._timer()
        try:
            adapter.fit(
                model,
                X_train_sel,
                y_train_arr,
                categorical_columns=categorical_columns,
            )
            y_pred = adapter.predict(
                model,
                X_test_sel,
                categorical_columns=categorical_columns,
            )
            proba = adapter.predict_proba(
                model,
                X_test_sel,
                categorical_columns=categorical_columns,
            )
        except NativeCategoricalStage2Error as exc:
            raise TypedInputCapabilityError(
                exc.code,
                str(exc),
                diagnostics=dict(exc.diagnostics),
            ) from exc
        if int(y_pred.size) != int(y_test_arr.size):
            raise TypedInputCapabilityError(
                "native_stage2_final_prediction_row_mismatch",
                "Native categorical Stage-2 final prediction row count is invalid.",
                diagnostics={"expected_rows": int(y_test_arr.size), "observed_rows": int(y_pred.size)},
            )
        classes = np.asarray(getattr(model, "classes_", np.unique(y_train_arr))).ravel()
        probability_reason = "ok"
        probability_available = True
        if (
            proba.ndim != 2
            or int(proba.shape[0]) != int(y_test_arr.size)
            or int(proba.shape[1]) != int(classes.size)
            or not np.all(np.isfinite(proba))
            or np.any(proba < 0.0)
        ):
            probability_available = False
            probability_reason = "native_predict_proba_invalid_shape_or_values"
        elif not np.allclose(np.sum(proba, axis=1), 1.0, atol=1e-6, rtol=1e-6):
            probability_available = False
            probability_reason = "native_predict_proba_not_simplex"

        bal_acc = self._safe_balanced_accuracy(y_test_arr, y_pred)
        macro_f1 = self._safe_macro_f1(y_test_arr, y_pred)
        accuracy = float(accuracy_score(y_test_arr, y_pred))
        log_loss_value = float("nan")
        if probability_available:
            try:
                log_loss_value = float(
                    sklearn_log_loss(y_test_arr, proba, labels=classes)
                )
            except Exception as exc:
                probability_reason = f"log_loss:{type(exc).__name__}"

        roc_meta: Dict[str, Any] = {
            "roc_auc": float("nan"),
            "roc_curve_type": "unavailable",
            "roc_auc_source": "native_predict_proba",
            "roc_curve_points": tuple(),
            "roc_curve_ova": {},
            "roc_auc_macro_ovr": float("nan"),
            "roc_auc_micro_ovr": float("nan"),
            "roc_auc_weighted_ovr": float("nan"),
            "roc_metric_capabilities": {
                "has_predict_proba": True,
                "has_decision_function": False,
                "supports_hard_vote_fraction": False,
                "supports_predicted_label_fallback": False,
                "selected_source": "native_predict_proba",
            },
        }
        if probability_available:
            try:
                observed_classes = np.unique(y_test_arr)
                if observed_classes.size == 2:
                    positive = observed_classes[1]
                    matching = np.where(classes == positive)[0]
                    if matching.size == 1:
                        score_vec = np.asarray(proba[:, int(matching[0])], dtype=float)
                        y_positive = np.asarray(y_test_arr == positive, dtype=int)
                        auc = float(roc_auc_score(y_positive, score_vec))
                        fpr, tpr, _ = roc_curve(y_positive, score_vec)
                        points = self._downsample_roc_curve_points(fpr, tpr)
                        roc_meta.update(
                            {
                                "roc_auc": auc,
                                "roc_curve_type": "binary",
                                "roc_curve_points": points,
                                "roc_auc_macro_ovr": auc,
                                "roc_auc_micro_ovr": auc,
                                "roc_auc_weighted_ovr": auc,
                                "roc_curve_ova": {
                                    str(positive): {
                                        "roc_auc": auc,
                                        "roc_curve_points": [
                                            [float(point[0]), float(point[1])]
                                            for point in points
                                        ],
                                    }
                                },
                            }
                        )
                elif observed_classes.size > 2 and set(observed_classes.tolist()) == set(classes.tolist()):
                    auc = float(
                        roc_auc_score(
                            y_test_arr,
                            proba,
                            labels=classes,
                            average="macro",
                            multi_class="ovr",
                        )
                    )
                    roc_meta.update(
                        {
                            "roc_auc": auc,
                            "roc_curve_type": "ovr_macro",
                            "roc_auc_macro_ovr": auc,
                        }
                    )
            except Exception:
                pass

        model_cv_meta = dict(candidate.get("model_cv_meta") or {})
        model_cv_meta["classification_stage2_wall_seconds"] = float(
            max(0.0, self._timer() - started)
        )
        selection_identity = dict(
            model_cv_meta.get("classification_selected_identity") or {}
        )
        model_name = str(candidate.get("model_name") or adapter.canonical_name)
        base_descriptor = self._inspect_selected_classifier(
            model,
            None,
            model_cv_meta=model_cv_meta,
            effective_model_name=model_name,
            sample_weight_requested=False,
            sample_weight_routed_observation="native_dataframe_unweighted",
        )
        final_descriptor = self._inspect_selected_classifier(
            model,
            None,
            model_cv_meta=model_cv_meta,
            effective_model_name=model_name,
            requested_device=base_descriptor.requested_device,
            sample_weight_requested=False,
            sample_weight_routed_observation="native_dataframe_unweighted",
        )
        descriptor_meta = self._fitted_descriptor_metadata(
            base_descriptor=base_descriptor,
            final_descriptor=final_descriptor,
        )
        route.update(
            {
                "status": "final_fit_predict_proba_complete",
                "final_fit_rows": int(y_train_arr.size),
                "final_predict_rows": int(y_test_arr.size),
                "final_predict_proba_rows": int(proba.shape[0])
                if proba.ndim == 2
                else 0,
                "final_probability_validation": probability_reason,
                "final_model_name": model_name,
            }
        )

        log_loss_meta = {
            "log_loss_available": bool(
                probability_available and np.isfinite(log_loss_value)
            ),
            "log_loss_skip_reason": "ok"
            if probability_available and np.isfinite(log_loss_value)
            else probability_reason,
            "log_loss_reason": "native_predict_proba",
            "log_loss_requirement": ProbabilityRequirement.MATRIX.value,
            "log_loss_probability_kind": "native",
            "log_loss_probability_source": "native_dataframe_predict_proba",
            "log_loss_class_order": [str(value) for value in classes.tolist()],
            "log_loss_class_alignment": "native_adapter_checked",
        }
        calibration_meta = {
            "calibration_reporting_enabled": False,
            "calibration_metrics_available": False,
            "calibration_brier": float("nan"),
            "calibration_ece": float("nan"),
            "calibration_n_eval": 0,
            "calibration_n_classes": 0,
            "calibration_skip_reason": "native_stage2_disabled_by_contract",
            "calibration_probability_kind": "native",
            "calibration_probability_source": "native_dataframe_predict_proba",
            "calibration_class_order": [str(value) for value in classes.tolist()],
            "calibration_class_alignment": "native_adapter_checked",
            "calibration_sample_weight_consumed": False,
        }
        posthoc_meta = {
            "classifier_posthoc_calibration_enabled": False,
            "classifier_posthoc_calibration_applied": False,
            "classifier_posthoc_calibration_method": "disabled",
            "classifier_posthoc_calibration_fraction": 0.0,
            "classifier_posthoc_calibration_min_calibration": 0,
            "classifier_posthoc_calibration_refinement_stopping": False,
            "classifier_posthoc_calibration_skip_reason": "native_stage2_disabled_by_contract",
            "classifier_posthoc_calibration_size": 0,
            "classifier_posthoc_calibration_fit_size": int(y_train_arr.size),
            "classifier_posthoc_calibration_sample_weight_requested": False,
            "classifier_posthoc_calibration_sample_weight_consumed": False,
            "classifier_posthoc_calibration_sample_weight_route": "not_requested",
        }
        conformal_meta = {
            "classifier_conformal_enabled": False,
            "classifier_conformal_applied": False,
            "classifier_conformal_skip_reason": "native_stage2_disabled_by_contract",
            "classifier_conformal_method": "disabled",
            "classifier_conformal_alpha": float("nan"),
            "classifier_conformal_calibration_fraction": float("nan"),
            "classifier_conformal_min_calibration": 0,
            "classifier_conformal_calibration_size": 0,
            "classifier_conformal_fit_size": 0,
            "classifier_conformal_prediction_sets": [],
        }
        out: Dict[str, Any] = {
            "accuracy": accuracy,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "log_loss": log_loss_value,
            "hybrid_score": float(0.6 * bal_acc + 0.4 * macro_f1),
            "roc_auc": float(roc_meta["roc_auc"]),
            "roc_curve_type": str(roc_meta["roc_curve_type"]),
            "roc_auc_source": str(roc_meta["roc_auc_source"]),
            "roc_curve_points": tuple(roc_meta["roc_curve_points"]),
            "roc_curves_by_method": {
                model_name: self._serialize_roc_meta(roc_meta)
            },
            "selected_features": int(X_train_sel.shape[1]),
            "selected_indices": tuple(
                int(value)
                for value in np.asarray(
                    candidate.get("selected_indices", tuple()), dtype=int
                ).ravel().tolist()
            ),
            "model_name": model_name,
            "effective_enabled_methods": tuple(
                str(value)
                for value in candidate.get("enabled_methods")
                or tuple(self.config.enabled_methods)
            ),
            "requested_enabled_methods": tuple(
                str(value)
                for value in candidate.get("requested_enabled_methods")
                or candidate.get("enabled_methods")
                or tuple(self.config.enabled_methods)
            ),
            "enabled_methods_source": str(
                candidate.get("enabled_methods_source", "config")
            ),
            "feature_selection_resampling": dict(
                candidate.get("feature_selection_resampling") or {}
            ),
            "sample_weight_provenance": dict(
                getattr(self, "_sample_weight_provenance", {}) or {}
            ),
            "native_stage2_route": route,
            "_fitted_selector": candidate.get("_fitted_selector"),
            "_selection_result": candidate.get("_selection_result"),
            "_fitted_model": model,
            "_post_df_summaries": list(candidate.get("_post_df_summaries") or []),
            "_post_df_meta": dict(candidate.get("_post_df_meta") or {}),
            "_post_df_time_sec": float(candidate.get("_post_df_time_sec", 0.0) or 0.0),
            "_folding_meta": dict(candidate.get("_folding_meta") or {}),
            "_folding_state": dict(candidate.get("_folding_state") or {}),
        }
        out.update(dict(variance_floor_meta))
        out.update(model_cv_meta)
        out.update(dict(candidate.get("stage2_ratio_meta") or {}))
        out.update(descriptor_meta)
        out.update(log_loss_meta)
        out.update(calibration_meta)
        out.update(posthoc_meta)
        out.update(conformal_meta)
        typed_admission = dict(
            getattr(self, "_typed_feature_selector_admission", {}) or {}
        )
        if typed_admission:
            out["typed_feature_selector_admission"] = _json_safe(typed_admission)
        for key in (
            "importance_uq_enabled",
            "importance_uq_computed",
            "importance_uq_reason",
            "importance_uq_n_folds",
            "importance_uq_unstable_threshold",
            "importance_uq_unstable_feature_count",
            "importance_uq_unstable_feature_indices",
            "fs_selection_summary",
            "fs_diagnostics",
            "selector_overrides_applied",
        ):
            if key in candidate:
                out[key] = candidate[key]
        if selection_identity:
            out["classification_selected_identity"] = selection_identity
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
        fit_resampling_context: Optional[FitResamplingContext] = None,
        fs_resampling_context: Optional[FitResamplingContext] = None,
        post_df_source_raw_train: Optional[np.ndarray] = None,
        post_df_source_raw_test: Optional[np.ndarray] = None,
        post_df_source_base_train: Optional[np.ndarray] = None,
        post_df_source_base_test: Optional[np.ndarray] = None,
        post_df_source_space: str = "prefilter_raw",
        sample_weight_train: Optional[np.ndarray] = None,
        sample_weight_test: Optional[np.ndarray] = None,
        native_stage2_context: _NativeCategoricalStage2Context | None = None,
        nested_pairing_raw_context: _NestedPairingRawContext | None = None,
        capture_evaluation_predictions: bool = False,
    ) -> Dict[str, Any]:
        # T-R-272: apply variance floor before any feature-selection logic.
        variance_floor_meta: Dict[str, Any] = {
            "variance_floor_enabled": bool(getattr(self.config, "prefilter_variance_floor_enabled", True)),
            "variance_floor_n_removed": 0,
            "variance_floor_n_before": int(X_train_full.shape[1]),
        }
        selector_input_indices = np.arange(int(X_train_full.shape[1]), dtype=int)
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
                selector_input_indices = selector_input_indices[keep_mask]
                self._update_diakrino_prefilter_active_mask(keep_mask)
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
        variance_floor_meta["variance_floor_active_input_indices"] = tuple(
            int(value) for value in selector_input_indices.tolist()
        )
        if native_stage2_context is not None:
            base_positions = tuple(native_stage2_context.selector_numeric_positions)
            if not base_positions:
                raise TypedInputCapabilityError(
                    "native_stage2_selector_context_unavailable",
                    "Native Stage-2 routing has no precomputed numeric selector mapping.",
                )
            if np.any(selector_input_indices < 0) or np.any(
                selector_input_indices >= len(base_positions)
            ):
                raise TypedInputCapabilityError(
                    "native_stage2_selector_mapping_out_of_range",
                    "Variance-floor selection cannot be mapped to the native Stage-2 numeric schema.",
                    diagnostics={
                        "active_indices": [int(value) for value in selector_input_indices.tolist()],
                        "selector_mapping_width": len(base_positions),
                    },
                )
            native_stage2_context = native_stage2_context.with_selector_numeric_positions(
                [base_positions[int(value)] for value in selector_input_indices.tolist()]
            )

        candidate = self._choose_selector_candidate(
            X_fs=X_fs,
            y_fs=y_fs,
            X_train_full=X_train_full,
            X_test_full=X_test_full,
            y_train_full=y_train_full,
            seed=seed,
            dataset_name=dataset_name,
            fit_resampling_context=fit_resampling_context,
            fs_resampling_context=fs_resampling_context,
            post_df_source_raw_train=post_df_source_raw_train,
            post_df_source_raw_test=post_df_source_raw_test,
            post_df_source_base_train=post_df_source_base_train,
            post_df_source_base_test=post_df_source_base_test,
            post_df_source_space=post_df_source_space,
            native_stage2_context=native_stage2_context,
            nested_pairing_raw_context=nested_pairing_raw_context,
        )

        if native_stage2_context is not None:
            return self._finalize_native_categorical_stage2(
                candidate=candidate,
                y_train_full=y_train_full,
                y_test=y_test,
                seed=int(seed),
                variance_floor_meta=variance_floor_meta,
            )

        X_train_sel = np.asarray(candidate["X_train_sel"], dtype=float)
        X_test_sel = np.asarray(candidate["X_test_sel"], dtype=float)
        model = candidate["model"]
        model_name = str(candidate["model_name"])
        model_cv_meta = dict(candidate.get("model_cv_meta") or {})
        if bool(model_cv_meta.get("model_cv_sample_weight_cv_routed", False)):
            self._sample_weight_provenance["sample_weight_stage2_cv_consumed"] = True
        selection_identity = dict(
            model_cv_meta.get("classification_selected_identity") or {}
        )

        final_balance = self._apply_final_training_balance(
            X_train_sel,
            y_train_full,
            seed=int(seed),
            fit_context=fit_resampling_context,
            sample_weight=sample_weight_train,
            callsite="run_feature_selection_final_fit",
        )
        if final_balance is None:
            final_X_train = X_train_sel
            final_y_train = y_train_full
            final_sample_weight = sample_weight_train
        else:
            final_X_train = final_balance.X
            final_y_train = final_balance.y
            final_sample_weight = final_balance.sample_weight
            model_cv_meta["training_balance_final_fit_provenance"] = (
                final_balance.provenance.to_dict()
            )
        final_weight_route = fit_estimator_with_sample_weight(
            model,
            final_X_train,
            final_y_train,
            sample_weight=final_sample_weight,
        )
        if final_sample_weight is not None:
            self._sample_weight_provenance["sample_weight_stage2_fit_consumed"] = True
        base_fitted_descriptor = self._inspect_selected_classifier(
            model,
            final_X_train,
            model_cv_meta=model_cv_meta,
            effective_model_name=model_name,
            sample_weight_requested=final_sample_weight is not None,
            sample_weight_routed_observation=final_weight_route,
        )
        cls_cfg = self._classification_cfg()
        fitted_selection_identity = dict(selection_identity)
        if base_fitted_descriptor.members:
            fitted_selection_identity["members"] = [
                member.to_dict() for member in base_fitted_descriptor.members
            ]
        model, posthoc_calibration_meta = self._apply_posthoc_calibration(
            model,
            X_train_sel,
            y_train_full,
            seed=int(seed),
            base_descriptor=base_fitted_descriptor,
            selection_identity=fitted_selection_identity,
            backend=str(
                model_cv_meta.get("classification_backend_used", "unknown")
            ),
            fit_resampling_context=fit_resampling_context,
            sample_weight=sample_weight_train,
        )
        if bool(
            posthoc_calibration_meta.get(
                "classifier_posthoc_calibration_sample_weight_consumed", False
            )
        ):
            self._sample_weight_provenance[
                "sample_weight_posthoc_calibration_consumed"
            ] = True
        if bool(posthoc_calibration_meta.get("classifier_posthoc_calibration_applied", False)):
            model_name = f"{model_name}_calibrated_{posthoc_calibration_meta.get('classifier_posthoc_calibration_method', 'sigmoid')}"
        final_fitted_descriptor = self._inspect_selected_classifier(
            model,
            final_X_train,
            model_cv_meta=model_cv_meta,
            effective_model_name=model_name,
            requested_device=base_fitted_descriptor.requested_device,
            sample_weight_requested=final_sample_weight is not None,
            sample_weight_routed_observation=str(
                posthoc_calibration_meta.get(
                    "classifier_posthoc_calibration_sample_weight_route",
                    final_weight_route,
                )
            ),
        )
        fitted_descriptor_meta = self._fitted_descriptor_metadata(
            base_descriptor=base_fitted_descriptor,
            final_descriptor=final_fitted_descriptor,
        )
        if (
            isinstance(model, VotingClassifier)
            and str(getattr(model, "voting", "")).lower() == "soft"
            and not final_fitted_descriptor.probability_matrix_available
        ):
            raise RuntimeError(
                "soft_vote_probability_conformance_failed:"
                + str(final_fitted_descriptor.matrix_reason)
            )
        y_pred = model.predict(X_test_sel)

        bal_acc = self._safe_balanced_accuracy(
            y_test, y_pred, sample_weight=sample_weight_test
        )
        macro_f1 = self._safe_macro_f1(
            y_test, y_pred, sample_weight=sample_weight_test
        )
        _log_loss_val, log_loss_meta = self._safe_log_loss_with_meta(
            model,
            X_test_sel,
            y_test,
            final_fitted_descriptor,
            sample_weight=sample_weight_test,
        )
        calibration_eval_meta: Dict[str, Any] = {
            "calibration_reporting_enabled": bool(getattr(self.config, "calibration_reporting_enabled", False)),
            "calibration_metrics_available": False,
            "calibration_brier": float("nan"),
            "calibration_ece": float("nan"),
            "calibration_n_eval": 0,
            "calibration_n_classes": 0,
            "calibration_skip_reason": "disabled",
            "calibration_probability_kind": (
                final_fitted_descriptor.fitted_probability_kind.value
            ),
            "calibration_probability_source": str(
                final_fitted_descriptor.probability_source
            ),
            "calibration_class_order": [
                value.to_dict() for value in final_fitted_descriptor.class_order
            ],
            "calibration_class_alignment": "unobserved",
        }
        if bool(getattr(self.config, "calibration_reporting_enabled", False)) or bool(
            posthoc_calibration_meta.get("classifier_posthoc_calibration_enabled", False)
        ):
            calibration_eval_meta.update(
                self._safe_probability_calibration_metrics(
                model,
                X_test_sel,
                y_test,
                final_fitted_descriptor,
                sample_weight=sample_weight_test,
                )
            )
        roc_meta = self._safe_roc_auc_bundle(
            model=model,
            X_eval=X_test_sel,
            y_true=y_test,
            y_pred=y_pred,
            sample_weight=sample_weight_test,
        )
        roc_curves_by_method = self._collect_roc_curves_by_method(
            model=model,
            model_name=str(model_name),
            X_eval=X_test_sel,
            y_true=y_test,
            y_pred=y_pred,
            root_meta=roc_meta,
            sample_weight=sample_weight_test,
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
            "classifier_conformal_model_probability_kind": (
                final_fitted_descriptor.fitted_probability_kind.value
            ),
            "classifier_conformal_conformity_kind": "unavailable",
            "classifier_conformal_score_source": "unavailable",
            "classifier_conformal_calibration_score_source": "unavailable",
            "classifier_conformal_evaluation_score_source": "unavailable",
            "classifier_conformal_class_order": [],
            "classifier_conformal_source_consistent": False,
            "classifier_conformal_probability_required": False,
            "classifier_conformal_probability_claim": False,
            "classifier_conformal_used_predict_proba": False,
            "classifier_conformal_model_supports_predict_proba": False,
            "classifier_conformal_source_errors": {},
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
            "classifier_conformal_mapie_probability_requirement": (
                ProbabilityRequirement.MATRIX.value
            ),
            "classifier_conformal_mapie_probability_kind": (
                final_fitted_descriptor.fitted_probability_kind.value
            ),
            "classifier_conformal_mapie_probability_source": str(
                final_fitted_descriptor.probability_source
            ),
            "classifier_conformal_mapie_probability_admitted": False,
            "classifier_conformal_mapie_probability_reason": "not_requested",
            "classifier_conformal_mapie_class_order": [],
        }
        if bool(getattr(cls_cfg, "conformal_enabled", False)) and sample_weight_train is not None:
            # Split/APS/RAPS refit internally without a weighted contract.  Do
            # not present an unweighted set as a weighted decision artifact.
            conformal_meta["classifier_conformal_skip_reason"] = (
                "sample_weight_unsupported"
            )
        elif bool(getattr(cls_cfg, "conformal_enabled", False)):
            structured_conformal = bool(
                isinstance(fit_resampling_context, FitResamplingContext)
                and fit_resampling_context.policy.kind != "iid"
            )
            split_calibration_indices = None
            if structured_conformal:
                n_conformal = int(np.asarray(y_train_full).ravel().size)
                n_conformal_classes = len(
                    {
                        typed_scalar_key(value)
                        for value in np.asarray(
                            y_train_full, dtype=object
                        ).ravel().tolist()
                    }
                )
                split_min_cal = int(
                    max(2, getattr(cls_cfg, "conformal_min_calibration", 20) or 20)
                )
                split_min_required = int(
                    max(
                        split_min_cal + n_conformal_classes,
                        2 * n_conformal_classes + 2,
                    )
                )
                if n_conformal >= split_min_required:
                    split_cal_size = int(
                        max(
                            split_min_cal,
                            round(
                                float(
                                    getattr(
                                        cls_cfg,
                                        "conformal_calibration_fraction",
                                        0.25,
                                    )
                                    or 0.25
                                )
                                * n_conformal
                            ),
                        )
                    )
                    split_cal_size = int(
                        min(
                            split_cal_size,
                            n_conformal - max(2, n_conformal_classes),
                        )
                    )
                    if split_cal_size >= split_min_cal:
                        split_plan = resolve_holdout(
                            fit_resampling_context,
                            np.asarray(y_train_full).ravel(),
                            seed=int(seed),
                            train_size=int(n_conformal - split_cal_size),
                            purpose="classifier_conformal_split",
                        )
                        split_calibration_indices = (
                            split_plan.primary.train_indices,
                            split_plan.primary.test_indices,
                        )
                        self._active_resampling_plans[
                            "classifier_conformal_split"
                        ] = split_plan.to_metadata()
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
                    include_score_source=True,
                    classifier_identity=selection_identity,
                    classifier_backend=str(
                        model_cv_meta.get("classification_backend_used", "unknown")
                    ),
                    calibration_indices=split_calibration_indices,
                )
                conformal_meta.update(split_meta)
            except Exception as exc:
                conformal_meta["classifier_conformal_applied"] = False
                conformal_meta["classifier_conformal_skip_reason"] = str(type(exc).__name__)
                conformal_meta["classifier_conformal_prediction_sets"] = []

            # VAL12_Suggestions §2.3: MAPIE APS/RAPS/cross conformal (opt-in).
            _conformal_method = str(getattr(cls_cfg, "conformal_method", "split") or "split").strip().lower()
            if _conformal_method in {"aps", "raps", "cross"}:
                mapie_calibration_indices = None
                if structured_conformal and _conformal_method in {"aps", "raps"}:
                    mapie_n = int(np.asarray(y_train_full).ravel().size)
                    mapie_classes = n_conformal_classes
                    mapie_cal_size = int(max(20, round(0.25 * mapie_n)))
                    mapie_cal_size = int(
                        min(mapie_cal_size, mapie_n - max(2, mapie_classes))
                    )
                    if mapie_cal_size >= mapie_classes and mapie_cal_size >= 10:
                        mapie_plan = resolve_holdout(
                            fit_resampling_context,
                            np.asarray(y_train_full).ravel(),
                            seed=int(seed),
                            train_size=int(mapie_n - mapie_cal_size),
                            purpose="classifier_conformal_mapie",
                        )
                        mapie_calibration_indices = (
                            mapie_plan.primary.train_indices,
                            mapie_plan.primary.test_indices,
                        )
                        self._active_resampling_plans[
                            "classifier_conformal_mapie"
                        ] = mapie_plan.to_metadata()
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
                        classifier_descriptor=final_fitted_descriptor,
                        classifier_identity=selection_identity,
                        classifier_backend=str(
                            model_cv_meta.get(
                                "classification_backend_used", "unknown"
                            )
                        ),
                        calibration_indices=mapie_calibration_indices,
                        structured_resampling=structured_conformal,
                    )
                    conformal_meta.update(mapie_meta)
                except Exception as exc:
                    conformal_meta["classifier_conformal_mapie_applied"] = False
                    conformal_meta["classifier_conformal_mapie_skip_reason"] = str(type(exc).__name__)

        selected_indices = candidate.get("selected_indices")
        if selected_indices is None:
            selected_indices = np.arange(X_train_sel.shape[1], dtype=int)

        accuracy = (
            float(accuracy_score(y_test, y_pred))
            if sample_weight_test is None
            else float(accuracy_score(y_test, y_pred, sample_weight=sample_weight_test))
        )
        out: Dict[str, Any] = {
            "accuracy": accuracy,
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
            "requested_enabled_methods": tuple(
                str(m)
                for m in candidate.get("requested_enabled_methods")
                or candidate.get("enabled_methods")
                or tuple(self.config.enabled_methods)
            ),
            "enabled_methods_source": str(candidate.get("enabled_methods_source", "config")),
            "feature_selection_resampling": dict(
                candidate.get("feature_selection_resampling") or {}
            ),
            "sample_weight_provenance": dict(
                getattr(self, "_sample_weight_provenance", {}) or {}
            ),
        }
        out.update(calibration_eval_meta)
        out.update(log_loss_meta)
        out.update(posthoc_calibration_meta)
        out.update(fitted_descriptor_meta)
        out.update(conformal_meta)
        # T-R-272: attach variance floor diagnostics.
        out.update(variance_floor_meta)
        typed_admission = dict(
            getattr(self, "_typed_feature_selector_admission", {}) or {}
        )
        if typed_admission:
            out["typed_feature_selector_admission"] = _json_safe(typed_admission)

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

        diakrino_protected_augmentation = candidate.get("diakrino_protected_augmentation")
        if isinstance(diakrino_protected_augmentation, dict) and diakrino_protected_augmentation:
            out["diakrino_protected_augmentation"] = dict(diakrino_protected_augmentation)

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
            "diakrino_regime_conditional_enabled",
            "diakrino_regime_conditional_applied",
            "diakrino_regime_conditional_reason",
            "diakrino_regime_conditional_regime",
            "diakrino_regime_conditional_allowed",
            "diakrino_regime_conditional_allowed_regimes",
            "diakrino_regime_conditional_methods",
            "diakrino_regime_conditional_removed_methods",
            "diakrino_regime_conditional_enabled_methods_before",
            "diakrino_regime_conditional_enabled_methods_after",
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
        if bool(capture_evaluation_predictions):
            y_capture = tuple(np.asarray(y_test, dtype=object).ravel().tolist())
            pred_capture = tuple(np.asarray(y_pred, dtype=object).ravel().tolist())
            if len(y_capture) != len(pred_capture):
                raise NestedPairingEvaluationError(
                    "nested_pairing_prediction_length_mismatch",
                    "The private nested-evaluation prediction capture is not row aligned.",
                    diagnostics={
                        "y_true_rows": int(len(y_capture)),
                        "y_pred_rows": int(len(pred_capture)),
                    },
                )
            weight_capture = (
                tuple(1.0 for _ in range(len(y_capture)))
                if sample_weight_test is None
                else tuple(
                    float(value)
                    for value in np.asarray(sample_weight_test, dtype=float)
                    .ravel()
                    .tolist()
                )
            )
            if len(weight_capture) != len(y_capture):
                raise NestedPairingEvaluationError(
                    "nested_pairing_prediction_weight_length_mismatch",
                    "The private nested-evaluation weights are not row aligned.",
                    diagnostics={
                        "y_true_rows": int(len(y_capture)),
                        "sample_weight_rows": int(len(weight_capture)),
                    },
                )
            out["_evaluation_prediction_capture"] = {
                "y_true": y_capture,
                "y_pred": pred_capture,
                "sample_weights": weight_capture,
            }

        return out

    @staticmethod
    def _protected_classical_methods(enabled_methods: Sequence[str]) -> Tuple[str, ...]:
        resolved = tuple(
            str(method)
            for method in enabled_methods
            if str(method) not in set(_DIAKRINO_PROTECTED_CORE_EXCLUDED_METHODS)
        )
        return resolved or ("mutual_information", "anova_f")

    def _build_feature_selector(
        self,
        seed: int,
        enabled_methods: Sequence[str],
        dataset_name: str = "",
        selector_overrides: Optional[Dict[str, Any]] = None,
        protected_classical_core: bool = False,
    ):
        FeatureSelector = _load_feature_selector_cls()
        overrides = dict(selector_overrides or {})
        resolved_enabled_methods = tuple(str(m) for m in enabled_methods)
        if bool(protected_classical_core):
            resolved_enabled_methods = self._protected_classical_methods(
                resolved_enabled_methods
            )
        enabled_methods = resolved_enabled_methods
        diakrino_sidecar_for_selector = (
            "" if bool(protected_classical_core) else str(getattr(self.config, "diakrino_sidecar_path", "") or "")
        )
        diakrino_relevance_oracle_for_selector = bool(
            not protected_classical_core
            and getattr(self.config, "fs_use_diakrino_relevance_oracle", False)
        )
        diakrino_conformal_for_selector = bool(
            not protected_classical_core
            and getattr(self.config, "diakrino_conformal_selection_enabled", False)
        )
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
        diakrino_dataset_id = (
            str(getattr(self.config, "diakrino_sidecar_dataset_id", "") or "")
            or self._base_dataset_name(str(dataset_name or getattr(self, "_active_diakrino_dataset_id", "") or ""))
        )
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
            cfg_obj = copy.deepcopy(fs_cfg)
            try:
                # Run-level overrides remain owned by DFFSConfig / benchmark args.
                setattr(cfg_obj, "random_state", int(seed))
                setattr(cfg_obj, "problem_type", "classification")
                setattr(cfg_obj, "selection_strategy", str(selection_strategy))
                setattr(cfg_obj, "enabled_methods", set(enabled_methods))
                setattr(cfg_obj, "method_timeout_seconds", float(getattr(self.config, "fs_method_timeout_seconds", 0.0) or 0.0))
                setattr(
                    cfg_obj,
                    "method_max_rss_mb",
                    float(getattr(self.config, "fs_method_max_rss_mb", 0.0) or 0.0),
                )
                setattr(cfg_obj, "linear_svm_max_iter", int(getattr(self.config, "fs_linear_svm_max_iter", 10000) or 10000))
                setattr(cfg_obj, "parallel_n_jobs", int(getattr(self.config, "n_jobs", 1) or 1))
                methods_cfg = getattr(cfg_obj, "methods", None)
                if methods_cfg is not None:
                    setattr(
                        methods_cfg,
                        "mrmr_max_unique_pair_evaluations",
                        int(
                            getattr(
                                self.config,
                                "fs_mrmr_max_unique_pair_evaluations",
                                0,
                            )
                            or 0
                        ),
                    )
                    setattr(
                        methods_cfg,
                        "mrmr_max_runtime_seconds",
                        float(
                            getattr(self.config, "fs_mrmr_max_runtime_seconds", 0.0)
                            or 0.0
                        ),
                    )
                    setattr(
                        methods_cfg,
                        "mrmr_budget_fallback_mode",
                        str(
                            getattr(
                                self.config,
                                "fs_mrmr_budget_fallback_mode",
                                "empty",
                            )
                            or "empty"
                        ),
                    )
                    setattr(
                        methods_cfg,
                        "diakrino_prior_sidecar_path",
                        str(diakrino_sidecar_for_selector),
                    )
                    setattr(methods_cfg, "diakrino_prior_dataset_id", str(diakrino_dataset_id or ""))
                    setattr(
                        methods_cfg,
                        "diakrino_prior_score_column",
                        str(getattr(self.config, "diakrino_prior_score_column", "prior_logit") or "prior_logit"),
                    )
                    setattr(
                        methods_cfg,
                        "diakrino_screening_score_column",
                        str(
                            getattr(self.config, "diakrino_screening_score_column", "screening_logit")
                            or "screening_logit"
                        ),
                    )
                    setattr(
                        methods_cfg,
                        "diakrino_prior_calibrate",
                        str(getattr(self.config, "diakrino_prior_calibrate", "chunk_zscore") or "chunk_zscore"),
                    )
                    setattr(
                        methods_cfg,
                        "diakrino_prior_top_k",
                        int(getattr(self.config, "diakrino_prior_top_k", 0) or 0),
                    )
                    setattr(
                        methods_cfg,
                        "diakrino_conformal_selection_enabled",
                        bool(diakrino_conformal_for_selector),
                    )
                    setattr(
                        methods_cfg,
                        "diakrino_conformal_target_fdp",
                        float(
                            0.20
                            if getattr(self.config, "diakrino_conformal_target_fdp", 0.20) is None
                            else getattr(self.config, "diakrino_conformal_target_fdp", 0.20)
                        ),
                    )
                    setattr(
                        methods_cfg,
                        "diakrino_conformal_calibrate",
                        str(getattr(self.config, "diakrino_conformal_calibrate", "chunk_zscore") or "chunk_zscore"),
                    )
                    setattr(
                        methods_cfg,
                        "diakrino_conformal_null_fraction",
                        float(
                            0.50
                            if getattr(self.config, "diakrino_conformal_null_fraction", 0.50) is None
                            else getattr(self.config, "diakrino_conformal_null_fraction", 0.50)
                        ),
                    )
                    setattr(
                        methods_cfg,
                        "diakrino_conformal_min_null_scores",
                        int(getattr(self.config, "diakrino_conformal_min_null_scores", 4) or 4),
                    )
                    setattr(
                        methods_cfg,
                        "diakrino_conformal_max_features",
                        int(getattr(self.config, "diakrino_conformal_max_features", 0) or 0),
                    )
                    setattr(
                        methods_cfg,
                        "diakrino_conformal_qualification_record",
                        str(getattr(self.config, "diakrino_conformal_qualification_record", "") or ""),
                    )
                stability_cfg = getattr(cfg_obj, "stability", None)
                if stability_cfg is not None:
                    setattr(
                        stability_cfg,
                        "stability_target_pfer",
                        float(getattr(self.config, "fs_stability_target_pfer", 1.0) or 1.0),
                    )
                mnpo_cfg = getattr(cfg_obj, "mnpo", None)
                if mnpo_cfg is not None:
                    if bool(protected_classical_core):
                        setattr(mnpo_cfg, "use_diakrino_selector_prior", False)
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
                        if bool(protected_classical_core):
                            setattr(oracle_cfg, "use_diakrino_selector_prior", False)
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
                        setattr(
                            oracle_cfg,
                            "complexity_conditioning",
                            bool(getattr(self.config, "fs_oracle_complexity_conditioning", False)),
                        )
                        setattr(
                            oracle_cfg,
                            "use_diakrino_relevance_oracle",
                            bool(diakrino_relevance_oracle_for_selector),
                        )
                        setattr(
                            oracle_cfg,
                            "diakrino_relevance_min_n_train",
                            int(getattr(self.config, "fs_diakrino_relevance_min_n_train", 100) or 100),
                        )
                        setattr(
                            oracle_cfg,
                            "diakrino_relevance_score_column",
                            str(
                                getattr(self.config, "fs_diakrino_relevance_score_column", "prior_logit")
                                or "prior_logit"
                            ),
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
            diakrino_prior_sidecar_path=str(diakrino_sidecar_for_selector),
            diakrino_prior_dataset_id=str(diakrino_dataset_id or ""),
            diakrino_prior_score_column=str(getattr(self.config, "diakrino_prior_score_column", "prior_logit") or "prior_logit"),
            diakrino_screening_score_column=str(
                getattr(self.config, "diakrino_screening_score_column", "screening_logit") or "screening_logit"
            ),
            diakrino_prior_calibrate=str(getattr(self.config, "diakrino_prior_calibrate", "chunk_zscore") or "chunk_zscore"),
            diakrino_prior_top_k=int(getattr(self.config, "diakrino_prior_top_k", 0) or 0),
            diakrino_conformal_selection_enabled=bool(diakrino_conformal_for_selector),
            diakrino_conformal_target_fdp=float(
                0.20
                if getattr(self.config, "diakrino_conformal_target_fdp", 0.20) is None
                else getattr(self.config, "diakrino_conformal_target_fdp", 0.20)
            ),
            diakrino_conformal_calibrate=str(
                getattr(self.config, "diakrino_conformal_calibrate", "chunk_zscore") or "chunk_zscore"
            ),
            diakrino_conformal_null_fraction=float(
                0.50
                if getattr(self.config, "diakrino_conformal_null_fraction", 0.50) is None
                else getattr(self.config, "diakrino_conformal_null_fraction", 0.50)
            ),
            diakrino_conformal_min_null_scores=int(
                getattr(self.config, "diakrino_conformal_min_null_scores", 4) or 4
            ),
            diakrino_conformal_max_features=int(getattr(self.config, "diakrino_conformal_max_features", 0) or 0),
            diakrino_conformal_qualification_record=str(
                getattr(self.config, "diakrino_conformal_qualification_record", "") or ""
            ),
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
            use_diakrino_relevance_oracle=bool(diakrino_relevance_oracle_for_selector),
            use_diakrino_selector_prior=False,
            diakrino_relevance_min_n_train=int(getattr(self.config, "fs_diakrino_relevance_min_n_train", 100) or 100),
            diakrino_relevance_score_column=str(
                getattr(self.config, "fs_diakrino_relevance_score_column", "prior_logit") or "prior_logit"
            ),
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
            complexity_conditioning=bool(
                getattr(self.config, "fs_oracle_complexity_conditioning", False)
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
            mrmr_max_unique_pair_evaluations=int(
                getattr(self.config, "fs_mrmr_max_unique_pair_evaluations", 0) or 0
            ),
            mrmr_max_runtime_seconds=float(
                getattr(self.config, "fs_mrmr_max_runtime_seconds", 0.0) or 0.0
            ),
            mrmr_budget_fallback_mode=str(
                getattr(self.config, "fs_mrmr_budget_fallback_mode", "empty")
                or "empty"
            ),
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
            method_max_rss_mb=float(
                getattr(self.config, "fs_method_max_rss_mb", 0.0) or 0.0
            ),
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
            pathway_group_sparse_lasso_n_groups=int(
                getattr(self.config, "fs_pathway_group_sparse_lasso_n_groups", 50) or 50
            ),
            pathway_group_sparse_lasso_max_group_size=int(
                getattr(self.config, "fs_pathway_group_sparse_lasso_max_group_size", 50) or 50
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
        dataset_name: str = "",
        selector_overrides: Optional[Dict[str, Any]] = None,
        post_df_source_raw_train: Optional[np.ndarray] = None,
        post_df_source_raw_test: Optional[np.ndarray] = None,
        post_df_source_base_train: Optional[np.ndarray] = None,
        post_df_source_base_test: Optional[np.ndarray] = None,
        post_df_source_space: str = "prefilter_raw",
        append_diakrino_extras: bool = True,
        native_stage2_context: _NativeCategoricalStage2Context | None = None,
    ) -> Dict[str, Any]:
        seed_seq = np.random.SeedSequence(int(seed))
        child_streams = seed_seq.spawn(5)
        selector_seed = int(child_streams[0].generate_state(1, dtype=np.uint32)[0])
        folding_seed = int(child_streams[1].generate_state(1, dtype=np.uint32)[0])
        stage2_seed = int(child_streams[2].generate_state(1, dtype=np.uint32)[0])
        model_seed = int(child_streams[3].generate_state(1, dtype=np.uint32)[0])
        post_df_seed = int(child_streams[4].generate_state(1, dtype=np.uint32)[0])

        protected_ctx = self._diakrino_protected_selection_context(int(X_train_full.shape[1]))
        protected_enabled = protected_ctx is not None
        requested_enabled_methods = tuple(str(m) for m in enabled_methods)
        admitted_enabled_methods = self._admit_typed_selector_methods(
            requested_enabled_methods,
            source=f"candidate:{candidate_name}",
        )
        effective_enabled_methods = (
            self._protected_classical_methods(admitted_enabled_methods)
            if protected_enabled
            else admitted_enabled_methods
        )
        classical_local = (
            np.asarray(protected_ctx["classical_local_indices"], dtype=int)
            if protected_enabled
            else np.arange(X_train_full.shape[1], dtype=int)
        )
        selector = self._build_feature_selector(
            seed=selector_seed,
            enabled_methods=effective_enabled_methods,
            dataset_name=dataset_name,
            selector_overrides=selector_overrides,
            protected_classical_core=bool(protected_enabled),
        )
        fs_context = getattr(self, "_active_fs_resampling_context", None)
        split_plan_provider = None
        resampling_policy = "iid"
        if isinstance(fs_context, FitResamplingContext):
            resampling_policy = str(fs_context.policy.kind)
            if resampling_policy != "iid":
                if fs_context.n_rows != int(np.asarray(y_fs).ravel().size):
                    raise ResamplingContractError(
                        "Feature-selection context is not aligned with y_fs.",
                        code="context_row_mismatch",
                        diagnostics={
                            "context_rows": fs_context.n_rows,
                            "y_rows": int(np.asarray(y_fs).ravel().size),
                            "callsite": "feature_selector",
                        },
                    )

                def split_plan_provider(**request):
                    return self._resolve_inner_split_plan(
                        fs_context,
                        request["y"],
                        purpose=str(request["purpose"]),
                        n_splits=int(request["n_splits"]),
                        n_repeats=int(request["n_repeats"]),
                        seed=int(selector_seed),
                        stratified=bool(request["stratified"]),
                        shuffle=bool(request.get("shuffle", True)),
                    )

        selector_resampling_kwargs: Dict[str, Any] = {}
        if split_plan_provider is not None or resampling_policy != "iid":
            selector_resampling_kwargs = {
                "resampling_plan_provider": split_plan_provider,
                "resampling_policy": resampling_policy,
            }
        _, selection_result = selector.fit_transform(
            np.asarray(X_fs, dtype=float)[:, classical_local],
            y_fs,
            n_final_features=int(self.config.n_final_features),
            return_result_object=True,
            **selector_resampling_kwargs,
        )

        selection_config = getattr(selection_result, "config", {})
        if not isinstance(selection_config, Mapping):
            selection_config = {}
        selected_for_guard = np.asarray(
            getattr(selection_result, "selected_feature_indices", tuple())
        ).ravel()
        if bool(
            selection_config.get("selection_aggregation_fail_closed", False)
        ) and selected_for_guard.size == 0:
            candidate_statuses = selection_config.get(
                "selector_candidate_statuses", {}
            )
            if not isinstance(candidate_statuses, Mapping):
                candidate_statuses = {}
            raise IncompleteFeatureSelectionError(
                candidate_name=str(candidate_name),
                selection_aggregation_status=str(
                    selection_config.get(
                        "selection_aggregation_status",
                        "fail_closed_no_eligible_selector_evidence",
                    )
                ),
                selector_candidate_statuses=candidate_statuses,
                selected_feature_count=int(selected_for_guard.size),
            )

        X_train_sel = selector.transform(np.asarray(X_train_full, dtype=float)[:, classical_local])
        X_test_sel = selector.transform(np.asarray(X_test_full, dtype=float)[:, classical_local])

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

        selector_selected_indices = selector.get_selected_features_indices()
        if selector_selected_indices is None:
            selector_selected_indices = np.arange(np.asarray(X_train_sel).shape[1], dtype=int)

        # If the safety cap was applied, remap selected_indices to the kept subset.
        if _fs_cap_applied:
            selector_selected_indices = np.asarray(selector_selected_indices, dtype=int)
            selector_selected_indices = selector_selected_indices[_keep_cols]

        selector_selected_indices = np.asarray(selector_selected_indices, dtype=int).ravel()
        selected_indices = np.asarray(
            classical_local[selector_selected_indices]
            if protected_enabled
            else selector_selected_indices,
            dtype=int,
        ).ravel()
        diakrino_protected_diag: Dict[str, Any] = {}
        fitted_selector: Any = selector
        if _fs_cap_applied:
            # The cap is applied after the base selector.  Persist a selector
            # in the pre-selector coordinate system so inference replays the
            # retained columns rather than the uncapped selector output.
            fitted_selector = _FixedIndexFeatureSelector(
                selector_selected_indices,
                base_selector=selector,
            )
        if protected_enabled and protected_ctx is not None:
            protected_core_indices = np.asarray(selected_indices, dtype=int).ravel()
            active_original = np.asarray(
                protected_ctx["active_original_indices"], dtype=int
            ).ravel()
            protected_original = active_original[protected_core_indices]
            protected_original_set = set(int(i) for i in protected_original.tolist())
            candidate_pairs = list(
                zip(
                    np.asarray(
                        protected_ctx["ranked_candidate_original_indices"], dtype=int
                    ).ravel().tolist(),
                    np.asarray(
                        protected_ctx["ranked_candidate_local_indices"], dtype=int
                    ).ravel().tolist(),
                )
            )
            addition_budget = int(
                max(0, protected_ctx["state"].get("diakrino_addition_budget", 0) or 0)
            )
            closeout_result = (
                self._diakrino_closeout_admitted_pairs(
                    X_support=np.asarray(X_train_full, dtype=float),
                    y_support=np.asarray(y_train_full).reshape(-1),
                    seed=int(seed),
                    protected_ctx=protected_ctx,
                    protected_core_indices=protected_core_indices,
                )
                if bool(append_diakrino_extras)
                else None
            )
            admitted_pairs = (
                list(closeout_result[0])
                if closeout_result is not None
                else (
                    [
                        (int(original), int(local))
                        for original, local in candidate_pairs
                        if int(original) not in protected_original_set
                    ][:addition_budget]
                    if bool(append_diakrino_extras)
                    else []
                )
            )
            extra_indices = np.asarray(
                [int(local) for _, local in admitted_pairs], dtype=int
            )
            selected_indices = np.concatenate([protected_core_indices, extra_indices]).astype(int)
            X_train_sel = np.asarray(X_train_full, dtype=float)[:, selected_indices]
            X_test_sel = np.asarray(X_test_full, dtype=float)[:, selected_indices]

            extra_original = active_original[extra_indices]
            final_original = active_original[selected_indices]
            protected_retained = set(int(i) for i in protected_original.tolist()) & set(
                int(i) for i in final_original.tolist()
            )
            active_lookup = dict(protected_ctx["active_local_by_original"])
            replay_indices = np.asarray(
                [active_lookup[int(i)] for i in final_original.tolist()], dtype=int
            )
            fitted_selector = _FixedIndexFeatureSelector(replay_indices, base_selector=selector)
            state = dict(protected_ctx["state"])
            shadow_count = int(state.get("shadow_probe_candidate_count", 0) or 0)
            shadow_fraction = state.get("shadow_probe_candidate_fraction", float("nan"))
            agreement_rate = state.get("diakrino_classical_agreement_rate", float("nan"))
            diakrino_protected_diag = {
                "schema_version": "1.0",
                "enabled": True,
                "applied": True,
                "mode": "protected_union",
                "protected_core_source": "classical_selector_without_diakrino_methods_or_oracles",
                "protection_reason": str(state.get("reason", "protected_union_active")),
                "requested_enabled_methods": [str(m) for m in requested_enabled_methods],
                "protected_effective_methods": [str(m) for m in effective_enabled_methods],
                "augmentation_deferred_for_pairing": bool(not append_diakrino_extras),
                "protected_core_size": int(protected_original.size),
                "protected_core_original_indices": [int(i) for i in protected_original.tolist()],
                "protected_core_retained_size": int(len(protected_retained)),
                "protected_core_retention_rate": float(
                    len(protected_retained) / max(1, int(protected_original.size))
                ),
                "diakrino_addition_budget": int(state.get("diakrino_addition_budget", 0) or 0),
                "diakrino_ranked_candidate_count": int(len(candidate_pairs)),
                "diakrino_eligible_outside_candidate_count": int(
                    state.get("diakrino_eligible_outside_candidate_count", 0) or 0
                ),
                "diakrino_admitted_outside_candidate_count": int(
                    state.get("diakrino_admitted_outside_candidate_count", 0) or 0
                ),
                "diakrino_valid_finite_candidate_count": int(
                    state.get("diakrino_valid_finite_candidate_count", 0) or 0
                ),
                "diakrino_budget_scan_count": int(
                    state.get("diakrino_budget_scan_count", 0) or 0
                ),
                "diakrino_budget_scan_exhausted": bool(
                    state.get("diakrino_budget_scan_exhausted", False)
                ),
                "diakrino_additions": int(extra_original.size),
                "diakrino_extra_original_indices": [int(i) for i in extra_original.tolist()],
                "diakrino_classical_agreement_count": int(
                    state.get("diakrino_classical_agreement_count", 0) or 0
                ),
                "diakrino_classical_agreement_rate": float(agreement_rate),
                "shadow_probe_candidate_count": shadow_count,
                "shadow_probe_candidate_denominator": int(
                    state.get("shadow_probe_candidate_denominator", 0) or 0
                ),
                "shadow_probe_candidate_fraction": float(shadow_fraction),
                "shadow_probe_rejected_count": shadow_count,
                "final_feature_count": int(final_original.size),
                "final_original_indices": [int(i) for i in final_original.tolist()],
                "final_stability_available": False,
                "final_stability_rate": float("nan"),
                "final_stability_source": "unavailable",
            }
            if closeout_result is not None:
                diakrino_protected_diag.update(dict(closeout_result[1]))
                diakrino_protected_diag["mode"] = str(closeout_result[1]["closeout_arm"])
                diakrino_protected_diag["diakrino_addition_budget"] = int(
                    closeout_result[1]["addition_budget"]
                )
                diakrino_protected_diag["diakrino_admitted_outside_candidate_count"] = int(
                    closeout_result[1]["realized_additions"]
                )
                diakrino_protected_diag["protection_reason"] = (
                    "native_null_abstain"
                    if bool(closeout_result[1]["abstained"])
                    else "closeout_admitted"
                )

        native_stage2_route: Dict[str, Any] = {}
        if native_stage2_context is not None:
            X_train_stage2, X_test_stage2, selected_native_record = (
                native_stage2_context.selected_views(selected_indices)
            )
            selected_categorical_columns = tuple(
                str(value)
                for value in selected_native_record.get(
                    "selected_categorical_columns", tuple()
                )
            )
            model, model_name, cv_score, cv_std, cv_n, cv_meta = (
                fit_native_categorical_stage2_singleton(
                    adapter=native_stage2_context.adapter,
                    X_train=X_train_stage2,
                    y_train=y_train_full,
                    categorical_columns=selected_categorical_columns,
                    seed=int(model_seed),
                    fold_view_factory=native_stage2_context.fold_view_factory(
                        selected_numeric_positions=selected_native_record[
                            "selected_numeric_positions"
                        ],
                    ),
                    cv_splits=5,
                    max_train_test_gap=float(
                        getattr(
                            self._classification_cfg(),
                            "stage2_max_train_test_gap",
                            0.0,
                        )
                        or 0.0
                    ),
                    n_jobs=resolve_sklearn_n_jobs(
                        getattr(self.config, "n_jobs", 1)
                    ),
                )
            )
            post_df_summaries = []
            post_df_meta = {
                "transform_time_sec": 0.0,
                "n_fitted": 0,
                "n_transformed": 0,
                "n_rejected": 0,
                "n_skipped_unreliable": 0,
                "n_skipped_block_cv": 0,
                "n_downweighted": 0,
                "mean_stability_weight": float("nan"),
                "cdf_block_gating_time_sec": 0.0,
                "cdf_block_gating_budget_hit": False,
                "cdf_block_gating_blocks_evaluated": 0,
                "cdf_block_gating_blocks_applied": 0,
                "native_stage2_transform_bypass": "admitted_native_dataframe",
            }
            post_df_time_sec = 0.0
            folding_meta = {"folding_applied": False, "reason": "native_stage2_disabled"}
            folding_state = {}
            stage2_ratio_meta = {
                "stage2_ratio_augmentation_enabled": False,
                "stage2_ratio_features_applied": False,
                "stage2_ratio_features_reason": "native_stage2_disabled",
                "stage2_ratio_selection_method": "unavailable",
                "stage2_ratio_max_features": 0,
                "stage2_ratio_epsilon": float(
                    getattr(self._classification_cfg(), "stage2_ratio_epsilon", 1e-6)
                    or 1e-6
                ),
                "stage2_ratio_pool_size_effective": 0,
                "stage2_ratio_pairs_considered": 0,
                "stage2_ratio_features_added": 0,
                "stage2_ratio_feature_start_index": int(X_train_stage2.shape[1]),
                "stage2_ratio_pairs": [],
            }
            native_stage2_route = {
                "schema_version": "1.0",
                "enabled": True,
                "status": "cv_complete_final_fit_pending",
                "canonical_name": native_stage2_context.classifier_name,
                "adapter_identity": native_stage2_context.adapter.adapter_identity,
                "resolved_capabilities": dict(
                    native_stage2_context.resolved_capabilities
                ),
                **dict(selected_native_record),
                "cv_score": float(cv_score) if np.isfinite(cv_score) else float("nan"),
                "cv_score_std": float(cv_std) if np.isfinite(cv_std) else float("nan"),
                "cv_n_splits": int(cv_n),
                "cv_preprocessor_scope": str(
                    dict(cv_meta or {})
                    .get("native_categorical_stage2", {})
                    .get("cv_preprocessor_scope", "unavailable")
                ),
                "cv_fold_views": list(
                    dict(cv_meta or {})
                    .get("native_categorical_stage2", {})
                    .get("cv_fold_views", [])
                    or []
                ),
            }
        else:
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
        if diakrino_protected_diag:
            protected_core_set = set(
                int(i)
                for i in np.asarray(selector_selected_indices, dtype=int).ravel().tolist()
            )
            unstable_core = protected_core_set & set(int(i) for i in unstable_indices)
            if bool(uq_meta.get("importance_uq_computed", False)) and protected_core_set:
                diakrino_protected_diag["final_stability_available"] = True
                diakrino_protected_diag["final_stability_rate"] = float(
                    1.0 - (len(unstable_core) / len(protected_core_set))
                )
                diakrino_protected_diag["final_stability_source"] = (
                    "importance_uq_unstable_complement"
                )
                diakrino_protected_diag["final_stability_scope"] = "protected_core"
            diakrino_protected_diag["protected_core_unstable_count"] = int(len(unstable_core))

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

        fs_resampling_diag = dict(
            getattr(selector, "resampling_diagnostics_", {}) or {}
        )
        effective_selector_methods = tuple(
            str(method)
            for method in (
                fs_resampling_diag.get("effective_methods")
                or (
                    sorted(getattr(selector, "enabled_methods", ()))
                    if getattr(selector, "enabled_methods", None) is not None
                    else effective_enabled_methods
                )
            )
        )
        return {
            "candidate_name": str(candidate_name),
            "enabled_methods": effective_selector_methods,
            "requested_enabled_methods": tuple(str(m) for m in requested_enabled_methods),
            "feature_selection_resampling": fs_resampling_diag,
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
            "diakrino_protected_augmentation": dict(diakrino_protected_diag or {}),
            "native_stage2_route": dict(native_stage2_route or {}),
            "fs_cap_applied": _fs_cap_applied,
            "fs_cap_meta": {
                "original_n_selected": int(_n_sel),
                "max_allowed": int(_fs_max),
                "n_train": int(_n_train),
                "ratio": float(self.config.fs_max_selected_features_ratio),
                "cap": int(self.config.fs_max_selected_features_cap),
            } if _fs_cap_applied else {},
            "_fitted_selector": fitted_selector,
            "_selection_result": selection_result,
            "_post_df_summaries": list(post_df_summaries or []),
            "_post_df_meta": dict(post_df_meta or {}),
            "_post_df_time_sec": float(post_df_time_sec),
            "_folding_meta": dict(folding_meta or {}),
            "_folding_state": dict(folding_state or {}),
            "_native_stage2_adapter": (
                None
                if native_stage2_context is None
                else native_stage2_context.adapter
            ),
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
        dataset_name: str = "",
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

        protected_ctx = self._diakrino_protected_selection_context(int(X_train_full.shape[1]))
        protected_enabled = protected_ctx is not None
        requested_enabled_methods = tuple(str(m) for m in enabled_methods)
        effective_enabled_methods = (
            self._protected_classical_methods(requested_enabled_methods)
            if protected_enabled
            else requested_enabled_methods
        )
        classical_local = (
            np.asarray(protected_ctx["classical_local_indices"], dtype=int)
            if protected_enabled
            else np.arange(X_train_full.shape[1], dtype=int)
        )
        X_train_sel = np.asarray(X_train_full, dtype=float)[:, classical_local]
        X_test_sel = np.asarray(X_test_full, dtype=float)[:, classical_local]

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

        selected_indices = np.asarray(classical_local[selected_indices], dtype=int).ravel()
        diakrino_protected_diag: Dict[str, Any] = {}
        if actual_bypass_mode == "fast_univariate_filter":
            # The low-p/n bypass still selects a subset.  Persist those input
            # coordinates instead of using an identity transformer over the
            # already-sliced training matrix.
            fitted_selector: Any = _FixedIndexFeatureSelector(selected_indices)
        else:
            fitted_selector = _IdentityFeatureSelector(int(X_train_sel.shape[1]))
        if protected_enabled and protected_ctx is not None:
            protected_core_indices = np.asarray(selected_indices, dtype=int).ravel()
            active_original = np.asarray(
                protected_ctx["active_original_indices"], dtype=int
            ).ravel()
            protected_original = active_original[protected_core_indices]
            protected_original_set = set(int(i) for i in protected_original.tolist())
            candidate_pairs = list(
                zip(
                    np.asarray(
                        protected_ctx["ranked_candidate_original_indices"], dtype=int
                    ).ravel().tolist(),
                    np.asarray(
                        protected_ctx["ranked_candidate_local_indices"], dtype=int
                    ).ravel().tolist(),
                )
            )
            addition_budget = int(
                max(0, protected_ctx["state"].get("diakrino_addition_budget", 0) or 0)
            )
            closeout_result = self._diakrino_closeout_admitted_pairs(
                X_support=np.asarray(X_train_full, dtype=float),
                y_support=np.asarray(y_train_full).reshape(-1),
                seed=int(seed),
                protected_ctx=protected_ctx,
                protected_core_indices=protected_core_indices,
            )
            admitted_pairs = (
                list(closeout_result[0])
                if closeout_result is not None
                else [
                    (int(original), int(local))
                    for original, local in candidate_pairs
                    if int(original) not in protected_original_set
                ][:addition_budget]
            )
            extra_indices = np.asarray(
                [int(local) for _, local in admitted_pairs], dtype=int
            )
            selected_indices = np.concatenate([protected_core_indices, extra_indices]).astype(int)
            X_train_sel = np.asarray(X_train_full, dtype=float)[:, selected_indices]
            X_test_sel = np.asarray(X_test_full, dtype=float)[:, selected_indices]
            extra_original = active_original[extra_indices]
            final_original = active_original[selected_indices]
            protected_retained = set(int(i) for i in protected_original.tolist()) & set(
                int(i) for i in final_original.tolist()
            )
            active_lookup = dict(protected_ctx["active_local_by_original"])
            fitted_selector = _FixedIndexFeatureSelector(
                [active_lookup[int(i)] for i in final_original.tolist()]
            )
            state = dict(protected_ctx["state"])
            diakrino_protected_diag = {
                "schema_version": "1.0",
                "enabled": True,
                "applied": True,
                "mode": "protected_union",
                "protected_core_source": f"classical_regime_bypass:{actual_bypass_mode}",
                "protection_reason": str(state.get("reason", "protected_union_active")),
                "requested_enabled_methods": [str(m) for m in requested_enabled_methods],
                "protected_effective_methods": [str(m) for m in effective_enabled_methods],
                "protected_core_size": int(protected_original.size),
                "protected_core_original_indices": [int(i) for i in protected_original.tolist()],
                "protected_core_retained_size": int(len(protected_retained)),
                "protected_core_retention_rate": float(
                    len(protected_retained) / max(1, int(protected_original.size))
                ),
                "diakrino_addition_budget": int(state.get("diakrino_addition_budget", 0) or 0),
                "diakrino_ranked_candidate_count": int(len(candidate_pairs)),
                "diakrino_eligible_outside_candidate_count": int(
                    state.get("diakrino_eligible_outside_candidate_count", 0) or 0
                ),
                "diakrino_admitted_outside_candidate_count": int(
                    state.get("diakrino_admitted_outside_candidate_count", 0) or 0
                ),
                "diakrino_valid_finite_candidate_count": int(
                    state.get("diakrino_valid_finite_candidate_count", 0) or 0
                ),
                "diakrino_budget_scan_count": int(
                    state.get("diakrino_budget_scan_count", 0) or 0
                ),
                "diakrino_budget_scan_exhausted": bool(
                    state.get("diakrino_budget_scan_exhausted", False)
                ),
                "diakrino_additions": int(extra_original.size),
                "diakrino_extra_original_indices": [int(i) for i in extra_original.tolist()],
                "diakrino_classical_agreement_count": int(
                    state.get("diakrino_classical_agreement_count", 0) or 0
                ),
                "diakrino_classical_agreement_rate": float(
                    state.get("diakrino_classical_agreement_rate", float("nan"))
                ),
                "shadow_probe_candidate_count": int(
                    state.get("shadow_probe_candidate_count", 0) or 0
                ),
                "shadow_probe_candidate_denominator": int(
                    state.get("shadow_probe_candidate_denominator", 0) or 0
                ),
                "shadow_probe_candidate_fraction": float(
                    state.get("shadow_probe_candidate_fraction", float("nan"))
                ),
                "shadow_probe_rejected_count": int(
                    state.get("shadow_probe_candidate_count", 0) or 0
                ),
                "final_feature_count": int(final_original.size),
                "final_original_indices": [int(i) for i in final_original.tolist()],
                "final_stability_available": False,
                "final_stability_rate": float("nan"),
                "final_stability_source": "selector_bypassed",
            }
            if closeout_result is not None:
                diakrino_protected_diag.update(dict(closeout_result[1]))
                diakrino_protected_diag["mode"] = str(closeout_result[1]["closeout_arm"])
                diakrino_protected_diag["diakrino_addition_budget"] = int(
                    closeout_result[1]["addition_budget"]
                )
                diakrino_protected_diag["diakrino_admitted_outside_candidate_count"] = int(
                    closeout_result[1]["realized_additions"]
                )
                diakrino_protected_diag["protection_reason"] = (
                    "native_null_abstain"
                    if bool(closeout_result[1]["abstained"])
                    else "closeout_admitted"
                )

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
            "enabled_methods": tuple(str(m) for m in effective_enabled_methods),
            "requested_enabled_methods": tuple(str(m) for m in requested_enabled_methods),
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
            "diakrino_protected_augmentation": dict(diakrino_protected_diag or {}),
            "_fitted_selector": fitted_selector,
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
        fit_resampling_context: Optional[FitResamplingContext] = None,
        fs_resampling_context: Optional[FitResamplingContext] = None,
        post_df_source_raw_train: Optional[np.ndarray] = None,
        post_df_source_raw_test: Optional[np.ndarray] = None,
        post_df_source_base_train: Optional[np.ndarray] = None,
        post_df_source_base_test: Optional[np.ndarray] = None,
        post_df_source_space: str = "prefilter_raw",
        native_stage2_context: _NativeCategoricalStage2Context | None = None,
        nested_pairing_raw_context: _NestedPairingRawContext | None = None,
    ) -> Dict[str, Any]:
        """Choose the selector+classifier pairing candidate, defaulting to the configured selector."""
        self._active_fit_resampling_context = fit_resampling_context
        self._active_fs_resampling_context = fs_resampling_context
        policy = self._resolve_method_policy(
            dataset_name=dataset_name,
            X_ref=X_train_full,
            y_ref=y_train_full,
        )
        configured_methods = tuple(policy.get("enabled_methods", tuple(self.config.enabled_methods)))
        configured_source = str(policy.get("enabled_methods_source", "config"))
        configured_methods = self._admit_typed_selector_methods(
            configured_methods,
            source=configured_source,
        )
        selector_overrides = dict(policy.get("regime_policy_selector_overrides", {}) or {})
        if native_stage2_context is not None:
            if bool(policy.get("regime_policy_bypass_fs", False)) or bool(
                policy.get("tier_policy_applied", False)
            ):
                raise TypedInputCapabilityError(
                    "native_stage2_selector_policy_unsupported",
                    "Native categorical Stage-2 routing refuses selector-policy rewrites before candidate CV.",
                    diagnostics={
                        "regime_policy_bypass_fs": bool(
                            policy.get("regime_policy_bypass_fs", False)
                        ),
                        "tier_policy_applied": bool(
                            policy.get("tier_policy_applied", False)
                        ),
                    },
                )
            out = self._evaluate_selector_candidate(
                X_fs=X_fs,
                y_fs=y_fs,
                X_train_full=X_train_full,
                X_test_full=X_test_full,
                y_train_full=y_train_full,
                seed=seed,
                enabled_methods=configured_methods,
                candidate_name="configured_enabled_methods",
                dataset_name=dataset_name,
                selector_overrides=selector_overrides,
                post_df_source_raw_train=post_df_source_raw_train,
                post_df_source_raw_test=post_df_source_raw_test,
                post_df_source_base_train=post_df_source_base_train,
                post_df_source_base_test=post_df_source_base_test,
                post_df_source_space=post_df_source_space,
                native_stage2_context=native_stage2_context,
            )
            out["enabled_methods_source"] = configured_source
            return out
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
            "diakrino_regime_conditional_enabled",
            "diakrino_regime_conditional_applied",
            "diakrino_regime_conditional_reason",
            "diakrino_regime_conditional_regime",
            "diakrino_regime_conditional_allowed",
            "diakrino_regime_conditional_allowed_regimes",
            "diakrino_regime_conditional_methods",
            "diakrino_regime_conditional_removed_methods",
            "diakrino_regime_conditional_enabled_methods_before",
            "diakrino_regime_conditional_enabled_methods_after",
        )

        if nested_pairing_raw_context is not None:
            out = self._choose_nested_selector_candidate(
                raw_context=nested_pairing_raw_context,
                configured_methods=configured_methods,
                X_fs=X_fs,
                y_fs=y_fs,
                X_train_full=X_train_full,
                X_test_full=X_test_full,
                y_train_full=y_train_full,
                seed=seed,
                dataset_name=dataset_name,
                selector_overrides=selector_overrides,
                post_df_source_raw_train=post_df_source_raw_train,
                post_df_source_raw_test=post_df_source_raw_test,
                post_df_source_base_train=post_df_source_base_train,
                post_df_source_base_test=post_df_source_base_test,
                post_df_source_space=post_df_source_space,
            )
            for key in policy_keys:
                if key in policy:
                    out[key] = policy[key]
            return out

        if bool(policy.get("regime_policy_bypass_fs", False)):
            out = self._evaluate_selector_bypass_candidate(
                X_train_full=X_train_full,
                X_test_full=X_test_full,
                y_train_full=y_train_full,
                seed=seed,
                enabled_methods=configured_methods,
                candidate_name="configured_enabled_methods",
                dataset_name=dataset_name,
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
                dataset_name=dataset_name,
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
                dataset_name=dataset_name,
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
                dataset_name=dataset_name,
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
        protected_pairing = self._diakrino_protected_selection_context(
            int(X_train_full.shape[1])
        ) is not None

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
                    dataset_name=dataset_name,
                    selector_overrides=selector_overrides,
                    post_df_source_raw_train=post_df_source_raw_train,
                    post_df_source_raw_test=post_df_source_raw_test,
                    post_df_source_base_train=post_df_source_base_train,
                    post_df_source_base_test=post_df_source_base_test,
                    post_df_source_space=post_df_source_space,
                    append_diakrino_extras=bool(not protected_pairing),
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
                dataset_name=dataset_name,
                selector_overrides=selector_overrides,
                post_df_source_raw_train=post_df_source_raw_train,
                post_df_source_raw_test=post_df_source_raw_test,
                post_df_source_base_train=post_df_source_base_train,
                post_df_source_base_test=post_df_source_base_test,
                post_df_source_space=post_df_source_space,
                append_diakrino_extras=bool(not protected_pairing),
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
        selected_classical = selected
        pairing_meta = {
            "maqc_pairing_enabled": True,
            "maqc_pairing_selected_fs_name": str(selected_classical.get("candidate_name", "")),
            "maqc_pairing_selected_cv_score": float(selected_score) if np.isfinite(selected_score) else float("nan"),
            "maqc_pairing_selected_cv_score_std": float(selected_classical.get("model_cv_score_std", float("nan"))),
            "maqc_pairing_selected_cv_n_splits": int(selected_classical.get("model_cv_score_n_splits", 0) or 0),
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

        if protected_pairing:
            pairing_meta["maqc_pairing_score_space"] = (
                "classical_only_before_diakrino_augmentation"
            )
            selected_requested_methods = tuple(
                str(m)
                for m in selected_classical.get("requested_enabled_methods")
                or selected_classical.get("enabled_methods")
                or configured_methods
            )
            selected = self._evaluate_selector_candidate(
                X_fs=X_fs,
                y_fs=y_fs,
                X_train_full=X_train_full,
                X_test_full=X_test_full,
                y_train_full=y_train_full,
                seed=seed,
                enabled_methods=selected_requested_methods,
                candidate_name=str(selected_classical.get("candidate_name", "configured_enabled_methods")),
                dataset_name=dataset_name,
                selector_overrides=selector_overrides,
                post_df_source_raw_train=post_df_source_raw_train,
                post_df_source_raw_test=post_df_source_raw_test,
                post_df_source_base_train=post_df_source_base_train,
                post_df_source_base_test=post_df_source_base_test,
                post_df_source_space=post_df_source_space,
                append_diakrino_extras=True,
            )
            pairing_meta["maqc_pairing_augmented_cv_score"] = float(
                selected.get("model_cv_score", float("nan"))
            )
            pairing_meta["maqc_pairing_augmented_selected_feature_count"] = int(
                len(tuple(selected.get("selected_indices", tuple()) or tuple()))
            )

        selected["enabled_methods_source"] = str(selected_source)
        selected["pairing_meta"] = pairing_meta
        for key in policy_keys:
            if key in policy:
                selected[key] = policy[key]
        return selected

    def _outer_split_request(self, n_total: int) -> Tuple[float, Optional[int]]:
        """Return the legacy test ratio and optional absolute training cap."""

        forced_train_n: Optional[int] = None
        if (
            self.config.max_train_samples is not None
            and self.config.max_train_samples > 0
            and int(n_total) >= 4
        ):
            train_n = int(max(2, int(self.config.max_train_samples)))
            min_test_n = int(max(2, np.ceil(0.20 * float(n_total))))
            train_n = int(min(train_n, int(n_total) - min_test_n))
            forced_train_n = int(train_n)
        test_size = float(max(0.05, min(0.95, float(self.config.test_size))))
        return test_size, forced_train_n

    def _resolve_outer_split_plan(
        self,
        y: np.ndarray,
        *,
        seed: int,
        resampling_context: FitResamplingContext,
        supplied_split_id: Optional[str] = None,
    ) -> ResolvedSplitPlan:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()
        context = ensure_fit_resampling_context(
            resampling_context,
            n_rows=int(y_arr.shape[0]),
        )
        test_size, forced_train_n = self._outer_split_request(int(y_arr.shape[0]))
        return resolve_holdout(
            context,
            y_arr,
            seed=int(seed),
            test_size=float(test_size),
            train_size=forced_train_n,
            purpose="outer",
            supplied_split_id=supplied_split_id,
        )

    def _resolve_inner_split_plan(
        self,
        context: FitResamplingContext,
        y: Sequence[Any],
        *,
        purpose: str,
        n_splits: int,
        n_repeats: int,
        seed: int,
        stratified: bool,
        shuffle: bool = True,
    ) -> ResolvedSplitPlan:
        plan = resolve_cv(
            context,
            y,
            n_splits=int(n_splits),
            seed=int(seed),
            purpose=str(purpose),
            n_repeats=int(n_repeats),
            stratified=bool(stratified),
            shuffle=bool(shuffle),
        )
        trace = getattr(self, "_active_resampling_plans", None)
        if isinstance(trace, dict):
            trace[str(purpose)] = plan.to_metadata()
        return plan

    def _split_indices(
        self,
        idx_all: np.ndarray,
        y: np.ndarray,
        seed: int,
        resampling_context: Optional[FitResamplingContext] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compatibility wrapper around the authoritative outer resolver."""

        index_values = np.asarray(idx_all, dtype=int).ravel()
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()
        if int(index_values.size) != int(y_arr.size):
            raise ValueError(
                f"idx_all has {index_values.size} rows but y has {y_arr.size}."
            )
        context = ensure_fit_resampling_context(
            resampling_context,
            n_rows=int(y_arr.size),
        )
        plan = self._resolve_outer_split_plan(
            y_arr,
            seed=int(seed),
            resampling_context=context,
        )
        train_positions = np.asarray(plan.primary.train_indices, dtype=int)
        test_positions = np.asarray(plan.primary.test_indices, dtype=int)
        return (
            np.asarray(index_values[train_positions], dtype=int),
            np.asarray(index_values[test_positions], dtype=int),
        )

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

    def _training_balance_cfg(self) -> TrainingBalanceConfig:
        value = getattr(self.config, "training_balance", None)
        if isinstance(value, TrainingBalanceConfig):
            return value
        if value is None:
            return TrainingBalanceConfig()
        return TrainingBalanceConfig.from_mapping(value)

    def _validate_training_balance_composition(
        self,
        *,
        fit_context: FitResamplingContext | None,
        callsite: str,
    ) -> TrainingBalanceConfig:
        balance = self._training_balance_cfg()
        if not balance.enabled:
            return balance
        cls_cfg = self._classification_cfg()
        unsupported: list[str] = []
        if str(cls_cfg.selection_mode).strip().lower() != "legacy":
            unsupported.append(f"selection_mode:{cls_cfg.selection_mode}")
        if str(cls_cfg.backend).strip().lower() != "sklearn":
            unsupported.append(f"backend:{cls_cfg.backend}")
        if bool(getattr(cls_cfg, "native_categorical_stage2_enabled", False)):
            unsupported.append("native_categorical_stage2")
        if bool(getattr(cls_cfg, "posthoc_calibration_enabled", False)):
            unsupported.append("posthoc_calibration_internal_refit")
        if bool(getattr(cls_cfg, "conformal_enabled", False)):
            unsupported.append("conformal_internal_refit")
        if bool(getattr(cls_cfg, "enable_svc_probability", False)):
            unsupported.append("svc_probability_internal_refit")
        if bool(getattr(cls_cfg, "include_tabpfn_model", False)):
            unsupported.append("tabpfn_opaque_backend")
        if bool(getattr(cls_cfg, "include_tabentics_diakrino_model", False)):
            unsupported.append("tabentics_diakrino_native_backend")
        if bool(getattr(self.config, "typed_input_enabled", False)):
            unsupported.append("typed_or_mixed_input")
        if unsupported:
            raise TrainingBalanceContractError(
                "Training balancing v1 rejects classifier compositions with opaque or internal refits.",
                code="training_balance_unsupported_composition",
                diagnostics={"callsite": str(callsite), "unsupported": sorted(unsupported)},
            )
        if fit_context is not None and fit_context.policy.kind not in {"iid", "stratified"}:
            raise TrainingBalanceContractError(
                "Training balancing v1 supports only iid and stratified resampling policies.",
                code="unsupported_resampling_policy",
                diagnostics={
                    "callsite": str(callsite),
                    "policy": str(fit_context.policy.kind),
                },
            )
        return balance

    def _apply_final_training_balance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        seed: int,
        fit_context: FitResamplingContext | None,
        sample_weight: np.ndarray | None,
        callsite: str,
    ) -> TrainingBalanceResult | None:
        balance = self._validate_training_balance_composition(
            fit_context=fit_context,
            callsite=callsite,
        )
        if not balance.enabled:
            return None
        result = apply_training_balance(
            np.asarray(X_train, dtype=float),
            np.asarray(y_train).ravel(),
            config=balance,
            pipeline_seed=int(seed),
            context=fit_context,
            sample_weight=sample_weight,
        )
        records = dict(getattr(self, "_training_balance_final_provenance", {}) or {})
        records[str(callsite)] = result.provenance.to_dict()
        self._training_balance_final_provenance = records
        return result

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
            include_tabentics_diakrino_model=bool(getattr(self.config, "include_tabentics_diakrino_model", False)),
            native_categorical_stage2_enabled=bool(
                getattr(self.config, "native_categorical_stage2_enabled", False)
            ),
            native_categorical_stage2_estimator=str(
                getattr(self.config, "native_categorical_stage2_estimator", "") or ""
            ),
            tabentics_diakrino_checkpoint=str(getattr(self.config, "tabentics_diakrino_checkpoint", "") or ""),
            tabentics_diakrino_max_features=int(getattr(self.config, "tabentics_diakrino_max_features", 256) or 256),
            tabentics_diakrino_batch_size=int(getattr(self.config, "tabentics_diakrino_batch_size", 32) or 32),
            tabentics_diakrino_support_joint_serving_cache=bool(
                getattr(self.config, "tabentics_diakrino_support_joint_serving_cache", False)
            ),
            tabentics_diakrino_retry_cuda_oom_microbatch=bool(
                getattr(self.config, "tabentics_diakrino_retry_cuda_oom_microbatch", False)
            ),
            tabentics_diakrino_device=str(getattr(self.config, "tabentics_diakrino_device", "auto") or "auto"),
            tabentics_diakrino_calibrate_probabilities=bool(
                getattr(self.config, "tabentics_diakrino_calibrate_probabilities", True)
            ),
            tabentics_diakrino_calibration_fraction=float(
                getattr(self.config, "tabentics_diakrino_calibration_fraction", 0.20) or 0.20
            ),
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
            enable_svc_probability=bool(
                getattr(self.config, "model_cv_enable_svc_probability", False)
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
            oracle_include_worst_class_recall=bool(
                getattr(self.config, "classifier_oracle_enable_worst_class", True)
            ),
            oracle_include_james_stein=bool(
                getattr(self.config, "classifier_oracle_include_james_stein", True)
            ),
            oracle_complexity_shrinkage=bool(
                getattr(self.config, "classifier_oracle_complexity_shrinkage", False)
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
            oracle_include_diakrino_family_meta=bool(
                getattr(self.config, "classifier_oracle_include_diakrino_family_meta", False)
            ),
            posthoc_calibration_enabled=bool(
                getattr(self.config, "classifier_posthoc_calibration_enabled", False)
            ),
            posthoc_calibration_method=str(
                getattr(self.config, "classifier_posthoc_calibration_method", "sigmoid") or "sigmoid"
            ),
            posthoc_calibration_fraction=float(
                getattr(self.config, "classifier_posthoc_calibration_fraction", 0.20) or 0.20
            ),
            posthoc_calibration_min_calibration=int(
                getattr(self.config, "classifier_posthoc_calibration_min_calibration", 20) or 20
            ),
            posthoc_calibration_refinement_stopping=bool(
                getattr(self.config, "classifier_posthoc_calibration_refinement_stopping", True)
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
            n_jobs=resolve_sklearn_n_jobs(getattr(self.config, "n_jobs", 1)),
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
            "tabentics_diakrino",
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
            enable_svc_probability=bool(getattr(cls_cfg, "enable_svc_probability", False)),
            build_xgb_model_fn=self._build_xgb_model,
            build_tabpfn_model_fn=self._build_tabpfn_model,
            build_tabentics_diakrino_model_fn=self._build_tabentics_diakrino_model,
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
                oracle_include_worst_class_recall=bool(
                    getattr(cls_cfg, "oracle_include_worst_class_recall", True)
                ),
                oracle_include_james_stein=bool(cls_cfg.oracle_include_james_stein),
                oracle_complexity_shrinkage=bool(
                    getattr(cls_cfg, "oracle_complexity_shrinkage", False)
                ),
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
                oracle_include_diakrino_family_meta=bool(
                    getattr(cls_cfg, "oracle_include_diakrino_family_meta", False)
                ),
                diakrino_sidecar_path=str(getattr(self.config, "diakrino_sidecar_path", "") or ""),
                diakrino_sidecar_dataset_id=str(
                    getattr(self.config, "diakrino_sidecar_dataset_id", "") or ""
                ),
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
                build_tabentics_diakrino_model_fn=self._build_tabentics_diakrino_model,
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
                    build_tabentics_diakrino_model_fn=self._build_tabentics_diakrino_model,
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
            n_jobs=int(cls_cfg.n_jobs),
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
            # Route admission is train-only.  Rejecting a pair because an
            # evaluation/inference covariate happens to fall outside the
            # training support would change the final model width from the
            # held-out matrix.  The runtime route uses the same deterministic
            # finite-value sanitisation for those out-of-support values.
            if not np.all(np.isfinite(r_train)):
                continue
            r_test = np.asarray(
                np.nan_to_num(r_test, nan=0.0, posinf=0.0, neginf=0.0),
                dtype=float,
            )
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
        fit_context = getattr(self, "_active_fit_resampling_context", None)
        training_balance = self._validate_training_balance_composition(
            fit_context=(fit_context if isinstance(fit_context, FitResamplingContext) else None),
            callsite="classifier_selection_cv",
        )
        sample_weight_arr: Optional[np.ndarray] = None
        if isinstance(fit_context, FitResamplingContext) and fit_context.sample_weights:
            sample_weight_arr = np.asarray(
                coerce_sample_weights(
                    fit_context.sample_weights,
                    n_rows=int(np.asarray(y_train).ravel().size),
                    field_name="classifier_selection_sample_weights",
                    require_positive_mass=True,
                ),
                dtype=float,
            )
        if sample_weight_arr is not None:
            cls_cfg = self._classification_cfg()
            capability_config = {
                "enable_svc_probability": bool(
                    getattr(cls_cfg, "enable_svc_probability", False)
                ),
                "tabentics_diakrino_calibrate_probabilities": bool(
                    getattr(
                        cls_cfg,
                        "tabentics_diakrino_calibrate_probabilities",
                        True,
                    )
                ),
            }
            weighted_admitted: List[str] = []
            weighted_excluded: Dict[str, str] = {}
            for candidate_name in candidate_names:
                resolved = resolve_classifier_capabilities(
                    str(candidate_name),
                    runtime=ClassifierRuntimeFacts(
                        n_classes=max(2, n_classes),
                        sample_weight_requested=True,
                        structured_resampling_requested=bool(
                            fit_context is not None
                            and fit_context.policy.kind != "iid"
                        ),
                    ),
                    config=capability_config,
                )
                if resolved.is_available:
                    weighted_admitted.append(str(candidate_name))
                else:
                    reasons = tuple(resolved.availability_reasons)
                    weighted_excluded[str(candidate_name)] = (
                        str(reasons[0]) if reasons else "sample_weight:unsupported"
                    )
            if not weighted_admitted:
                raise RuntimeError(
                    "sample_weight_no_effectively_supported_classifier:"
                    + ",".join(sorted(weighted_excluded))
                )
            candidate_names = weighted_admitted
            runtime_meta["model_cv_effective_candidates"] = tuple(candidate_names)
            runtime_meta["model_cv_sample_weight_requested"] = True
            runtime_meta["model_cv_sample_weight_excluded_candidates"] = dict(
                weighted_excluded
            )
            runtime_meta["model_cv_sample_weight_admitted_candidates"] = tuple(
                weighted_admitted
            )
        structured_policy = bool(
            isinstance(fit_context, FitResamplingContext)
            and fit_context.policy.kind != "iid"
        )
        if structured_policy:
            if fit_context.n_rows != int(np.asarray(y_train).ravel().size):
                raise ResamplingContractError(
                    "Classifier context is not aligned with y_train.",
                    code="context_row_mismatch",
                    diagnostics={
                        "context_rows": fit_context.n_rows,
                        "y_rows": int(np.asarray(y_train).ravel().size),
                        "callsite": "classifier_selection",
                    },
                )
            admitted_candidates: List[str] = []
            excluded_candidates: Dict[str, str] = {}
            cls_cfg = self._classification_cfg()
            capability_config = {
                "enable_svc_probability": bool(
                    getattr(cls_cfg, "enable_svc_probability", False)
                ),
                "tabentics_diakrino_calibrate_probabilities": bool(
                    getattr(cls_cfg, "tabentics_diakrino_calibrate_probabilities", True)
                ),
            }
            for candidate_name in candidate_names:
                resolved = resolve_classifier_capabilities(
                    str(candidate_name),
                    runtime=ClassifierRuntimeFacts(
                        n_classes=max(2, n_classes),
                        structured_resampling_requested=True,
                    ),
                    config=capability_config,
                )
                if resolved.structured_resampling is SupportLevel.SUPPORTED:
                    admitted_candidates.append(str(candidate_name))
                else:
                    excluded_candidates[str(candidate_name)] = (
                        "non_iid_internal_resampling_unsupported:"
                        f"classifier:{resolved.canonical_name}:"
                        f"{resolved.structured_resampling.value}"
                    )
            candidate_names = admitted_candidates
            if not candidate_names:
                if sample_weight_arr is not None:
                    raise RuntimeError(
                        "sample_weight_no_supported_classifier_after_structured_resampling"
                    )
                candidate_names = ["lr"]
            runtime_meta["model_cv_structured_resampling_policy"] = str(
                fit_context.policy.kind
            )
            runtime_meta["model_cv_structured_resampling_excluded"] = dict(
                excluded_candidates
            )
            runtime_meta["model_cv_effective_candidates"] = tuple(candidate_names)

        classifier_cv_plan: Optional[ResolvedSplitPlan] = None
        classifier_cv_pairs = None
        if structured_policy and class_counts.size > 0 and int(np.min(class_counts)) >= 2:
            requested_cv_splits = int(max(2, min(5, int(np.min(class_counts)))))
            classifier_cv_plan = self._resolve_inner_split_plan(
                fit_context,
                np.asarray(y_train).ravel(),
                purpose="classifier_selection_cv",
                n_splits=requested_cv_splits,
                n_repeats=1,
                seed=int(seed),
                stratified=True,
                shuffle=True,
            )
            classifier_cv_pairs = classifier_cv_plan.index_pairs()
            runtime_meta["model_cv_resampling_plan"] = classifier_cv_plan.to_metadata()

        backend, dispatch_meta = self._get_classifier_backend(candidate_names=candidate_names)
        runtime_meta.update(dispatch_meta)

        if training_balance.enabled and not isinstance(backend, SklearnBackend):
            raise TrainingBalanceContractError(
                "Training balancing v1 is implemented only by the sklearn Stage-2 backend.",
                code="training_balance_backend_unsupported",
                diagnostics={"backend": str(backend.name())},
            )

        if sample_weight_arr is not None and not isinstance(backend, SklearnBackend):
            raise RuntimeError(
                "sample_weight_backend_unsupported:" + str(backend.name())
            )

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
            if sample_weight_arr is None:
                model, model_name, score, std, n_splits, backend_meta = backend.fit_and_select(
                    np.asarray(X_train, dtype=float),
                    np.asarray(y_train).ravel(),
                    seed=int(seed),
                    n_classes=int(n_classes),
                    class_counts=class_counts,
                    cv_splits=5,
                    scoring="balanced_accuracy",
                    cv_plan=classifier_cv_pairs,
                    training_balance=training_balance,
                    balance_context=fit_context,
                )
            else:
                model, model_name, score, std, n_splits, backend_meta = backend.fit_and_select(
                    np.asarray(X_train, dtype=float),
                    np.asarray(y_train).ravel(),
                    seed=int(seed),
                    n_classes=int(n_classes),
                    class_counts=class_counts,
                    cv_splits=5,
                    scoring="balanced_accuracy",
                    cv_plan=classifier_cv_pairs,
                    sample_weight=sample_weight_arr,
                    training_balance=training_balance,
                    balance_context=fit_context,
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
                    cv_plan=classifier_cv_pairs,
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
        if bool(getattr(cls_cfg, "include_tabentics_diakrino_model", False)):
            _append("tabentics_diakrino")
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
                candidate = _TabPFNClassifier(**kwargs)
                if kwargs.get("device") is not None:
                    try:
                        setattr(
                            candidate,
                            "_tabnetics_requested_device",
                            str(kwargs["device"]),
                        )
                    except Exception:
                        pass
                return candidate, None
            except TypeError as exc:
                last_failure_reason = self._format_optional_candidate_failure(exc)
                continue
            except Exception as exc:
                last_failure_reason = self._format_optional_candidate_failure(exc)
                continue
        return None, str(last_failure_reason or "constructor_rejected_all_attempts")

    def _build_tabentics_diakrino_model(self, seed: int):
        cls_cfg = self._classification_cfg()
        checkpoint = str(getattr(cls_cfg, "tabentics_diakrino_checkpoint", "") or "").strip()
        if not checkpoint:
            return None, "native DIAKRINO checkpoint not configured"
        p = Path(checkpoint)
        if not p.exists():
            return None, f"native DIAKRINO checkpoint unavailable: {checkpoint}"
        try:
            import torch
            from tabnetics.classification.diakrino_native import (
                TabenticsDiakrinoNativeClassifier,
                load_tabentics_diakrino_fs_classifier,
            )

            requested_device = str(getattr(cls_cfg, "tabentics_diakrino_device", "auto") or "auto").strip().lower()
            if (
                (requested_device in {"cuda", "gpu"} or requested_device.startswith("cuda:"))
                and not torch.cuda.is_available()
            ):
                return None, "native DIAKRINO checkpoint requested CUDA but CUDA is unavailable"
            preflight_model, preflight = load_tabentics_diakrino_fs_classifier(p, map_location=torch.device("cpu"))
            del preflight_model
            if not bool(preflight.get("checkpoint_comparable_for_classification", False)):
                return (
                    None,
                    "native DIAKRINO checkpoint non-comparable for classification: "
                    f"{preflight.get('checkpoint_comparability_reason', 'unknown')}",
                )
            if not bool(preflight.get("checkpoint_usable_by_core_candidate", False)):
                return (
                    None,
                    "native DIAKRINO checkpoint not usable by core candidate: "
                    f"{preflight.get('checkpoint_usability_reason', 'unknown')}",
                )

            return (
                TabenticsDiakrinoNativeClassifier(
                    checkpoint=str(p),
                    max_features=int(max(1, int(getattr(cls_cfg, "tabentics_diakrino_max_features", 256) or 256))),
                    batch_size=int(max(1, int(getattr(cls_cfg, "tabentics_diakrino_batch_size", 32) or 32))),
                    support_joint_serving_cache=bool(
                        getattr(cls_cfg, "tabentics_diakrino_support_joint_serving_cache", False)
                    ),
                    retry_cuda_oom_microbatch=bool(
                        getattr(cls_cfg, "tabentics_diakrino_retry_cuda_oom_microbatch", False)
                    ),
                    device=str(getattr(cls_cfg, "tabentics_diakrino_device", "auto") or "auto"),
                    sidecar_path=str(getattr(self.config, "diakrino_sidecar_path", "") or ""),
                    dataset_id=str(
                        getattr(self.config, "diakrino_sidecar_dataset_id", "") or ""
                    )
                    or str(getattr(self, "_active_diakrino_dataset_id", "") or ""),
                    calibrate_probabilities=bool(
                        getattr(cls_cfg, "tabentics_diakrino_calibrate_probabilities", True)
                    ),
                    calibration_fraction=float(
                        getattr(cls_cfg, "tabentics_diakrino_calibration_fraction", 0.20) or 0.20
                    ),
                    random_state=int(seed),
                ),
                None,
            )
        except Exception as exc:
            return None, self._format_optional_candidate_failure(exc)

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
            "training_balance": self._training_balance_cfg().to_dict(),
            "training_balance_provenance": _json_safe(
                dict(getattr(self, "_training_balance_final_provenance", {}) or {})
            ),
            "test_size": self.config.test_size,
            "max_train_samples": self.config.max_train_samples,
            "fs_fraction": self.config.fs_fraction,
            "n_final_features": self.config.n_final_features,
            "n_jobs": resolve_sklearn_n_jobs(getattr(self.config, "n_jobs", 1)),
            "classification_n_jobs": int(self._classification_cfg().n_jobs),
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
            "auto_router_enabled": bool(
                getattr(self.config, "auto_router_enabled", False)
            ),
            "auto_router_artifact_path": str(
                getattr(self.config, "auto_router_artifact_path", "") or ""
            ),
            "auto_router_fail_open": bool(
                getattr(self.config, "auto_router_fail_open", True)
            ),
            "auto_router_descriptor_ood_gate_enabled": bool(
                getattr(self.config, "auto_router_descriptor_ood_gate_enabled", False)
            ),
            "auto_router_crossfit_uncertainty_enabled": bool(
                getattr(self.config, "auto_router_crossfit_uncertainty_enabled", False)
            ),
            "auto_router_last_decision": dict(
                getattr(self.config, "auto_router_last_decision", {}) or {}
            ),
            "meta_learning_selector_mode": str(
                getattr(self.config, "meta_learning_selector_mode", "none") or "none"
            ),
            "meta_learning_confidence_threshold": float(
                getattr(self.config, "meta_learning_confidence_threshold", 0.55) or 0.55
            ),
            "meta_learning_records_path": str(
                getattr(self.config, "meta_learning_records_path", "") or ""
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
            "dist_config__flex_family_distinct_gate_enabled": bool(
                getattr(self.config.dist_config, "flex_family_distinct_gate_enabled", False)
            ),
            "dist_config__flex_family_min_distinct": int(
                getattr(self.config.dist_config, "flex_family_min_distinct", 15) or 15
            ),
            "dist_config__flex_family_retention_margin": float(
                getattr(self.config.dist_config, "flex_family_retention_margin", 0.0) or 0.0
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
            "dist_config__diakrino_family_prior_lambda": float(
                getattr(self.config.dist_config, "diakrino_family_prior_lambda", 0.0) or 0.0
            ),
            "dist_config__diakrino_skip_fit_discrete_enabled": bool(
                getattr(self.config.dist_config, "diakrino_skip_fit_discrete_enabled", False)
            ),
            "dist_config__diakrino_sidecar_path": str(getattr(self.config.dist_config, "diakrino_sidecar_path", "") or ""),
            "dist_config__diakrino_sidecar_dataset_id": str(
                getattr(self.config.dist_config, "diakrino_sidecar_dataset_id", "") or ""
            ),
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
            "diakrino_sidecar_path": str(getattr(self.config, "diakrino_sidecar_path", "") or ""),
            "diakrino_sidecar_dataset_id": str(getattr(self.config, "diakrino_sidecar_dataset_id", "") or ""),
            "diakrino_sidecar_dataset_id_effective": str(
                getattr(self.config, "diakrino_sidecar_dataset_id", "") or ""
            )
            or str(getattr(self, "_active_diakrino_dataset_id", "") or ""),
            **(
                {
                    "diakrino_prefilter_enabled": True,
                    "diakrino_prefilter_mode": str(
                        getattr(self.config, "diakrino_prefilter_mode", "protected_union")
                        or "protected_union"
                    ),
                    "diakrino_prefilter_lambda": float(
                        getattr(self.config, "diakrino_prefilter_lambda", 0.0) or 0.0
                    ),
                    "diakrino_prefilter_max_extras": int(
                        getattr(self.config, "diakrino_prefilter_max_extras", 0) or 0
                    ),
                    "diakrino_prefilter_score_column": str(
                        getattr(self.config, "diakrino_prefilter_score_column", "prior_logit")
                        or "prior_logit"
                    ),
                    "diakrino_prefilter_shadow_probe_indices": [
                        int(i)
                        for i in tuple(
                            getattr(
                                self.config,
                                "diakrino_prefilter_shadow_probe_indices",
                                tuple(),
                            )
                            or tuple()
                        )
                    ],
                }
                if self._diakrino_prefilter_runtime_enabled()
                else {}
            ),
            "diakrino_prior_score_column": str(getattr(self.config, "diakrino_prior_score_column", "prior_logit") or "prior_logit"),
            "diakrino_screening_score_column": str(
                getattr(self.config, "diakrino_screening_score_column", "screening_logit") or "screening_logit"
            ),
            "diakrino_prior_calibrate": str(getattr(self.config, "diakrino_prior_calibrate", "chunk_zscore") or "chunk_zscore"),
            "diakrino_prior_top_k": int(getattr(self.config, "diakrino_prior_top_k", 0) or 0),
            "diakrino_conformal_selection_enabled": bool(
                getattr(self.config, "diakrino_conformal_selection_enabled", False)
            ),
            "diakrino_conformal_target_fdp": float(
                0.20
                if getattr(self.config, "diakrino_conformal_target_fdp", 0.20) is None
                else getattr(self.config, "diakrino_conformal_target_fdp", 0.20)
            ),
            "diakrino_conformal_calibrate": str(
                getattr(self.config, "diakrino_conformal_calibrate", "chunk_zscore") or "chunk_zscore"
            ),
            "diakrino_conformal_normalization_family": (
                "chunk_zscore"
                if str(getattr(self.config, "diakrino_conformal_calibrate", "chunk_zscore") or "chunk_zscore")
                == "chunk_zscore"
                else str(getattr(self.config, "diakrino_conformal_calibrate", "chunk_zscore") or "chunk_zscore")
            ),
            "diakrino_conformal_calibration": (
                "within_chunk_mean_std_then_split_conformal_bh"
                if str(getattr(self.config, "diakrino_conformal_calibrate", "chunk_zscore") or "chunk_zscore")
                == "chunk_zscore"
                else "split_conformal_bh"
            ),
            "diakrino_conformal_zscore_applied": bool(
                str(getattr(self.config, "diakrino_conformal_calibrate", "chunk_zscore") or "chunk_zscore")
                == "chunk_zscore"
            ),
            "diakrino_conformal_null_fraction": float(
                0.50
                if getattr(self.config, "diakrino_conformal_null_fraction", 0.50) is None
                else getattr(self.config, "diakrino_conformal_null_fraction", 0.50)
            ),
            "diakrino_conformal_min_null_scores": int(
                getattr(self.config, "diakrino_conformal_min_null_scores", 4) or 4
            ),
            "diakrino_conformal_max_features": int(
                getattr(self.config, "diakrino_conformal_max_features", 0) or 0
            ),
            "diakrino_conformal_qualification_record": str(
                getattr(self.config, "diakrino_conformal_qualification_record", "") or ""
            ),
            "diakrino_cdf_trust_gate_enabled": bool(getattr(self.config, "diakrino_cdf_trust_gate_enabled", False)),
            "diakrino_cdf_trust_entropy_threshold": float(
                getattr(self.config, "diakrino_cdf_trust_entropy_threshold", 1.01) or 1.01
            ),
            "diakrino_cdf_trust_fallback": str(getattr(self.config, "diakrino_cdf_trust_fallback", "rank_gaussian") or "rank_gaussian"),
            "diakrino_stability_surrogate_enabled": bool(getattr(self.config, "diakrino_stability_surrogate_enabled", False)),
            "diakrino_regime_conditional": bool(getattr(self.config, "diakrino_regime_conditional", False)),
            "use_rank_prefilter": self.config.use_rank_prefilter,
            "prefilter_top_k": self.config.prefilter_top_k,
            "prefilter_adaptive_top_k": bool(
                getattr(self.config, "prefilter_adaptive_top_k", False)
            ),
            "prefilter_adaptive_top_k_scaling": float(
                getattr(self.config, "prefilter_adaptive_top_k_scaling", 0.5) or 0.5
            ),
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
            "fs_method_max_rss_mb": float(
                getattr(self.config, "fs_method_max_rss_mb", 0.0) or 0.0
            ),
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
            "tier_classifier_mode": str(
                getattr(self.config, "tier_classifier_mode", "heuristic") or "heuristic"
            ),
            "tier_classifier_model_path": str(
                getattr(self.config, "tier_classifier_model_path", "") or ""
            ),
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
                getattr(self.config, "regime_gating_min_samples_per_class", 7.0) or 7.0
            ),
            "regime_gating_use_expanded_features": bool(
                getattr(self.config, "regime_gating_use_expanded_features", False)
            ),
            "regime_gating_min_fisher_f1": float(
                getattr(self.config, "regime_gating_min_fisher_f1", 0.10) or 0.10
            ),
            "regime_gating_max_n1_borderline": float(
                getattr(self.config, "regime_gating_max_n1_borderline", 0.40) or 0.40
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
            "fs_oracle_enable_stability": bool(
                getattr(self.config, "fs_oracle_enable_stability", self.config.use_stability_oracle)
            ),
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
            "fs_use_diakrino_relevance_oracle": bool(
                getattr(self.config, "fs_use_diakrino_relevance_oracle", False)
            ),
            "fs_diakrino_relevance_min_n_train": int(
                getattr(self.config, "fs_diakrino_relevance_min_n_train", 100) or 100
            ),
            "fs_diakrino_relevance_score_column": str(
                getattr(self.config, "fs_diakrino_relevance_score_column", "prior_logit") or "prior_logit"
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
            "fs_oracle_complexity_conditioning": bool(
                getattr(self.config, "fs_oracle_complexity_conditioning", False)
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
            "fs_mrmr_max_unique_pair_evaluations": int(
                getattr(self.config, "fs_mrmr_max_unique_pair_evaluations", 0) or 0
            ),
            "fs_mrmr_max_runtime_seconds": float(
                getattr(self.config, "fs_mrmr_max_runtime_seconds", 0.0) or 0.0
            ),
            "fs_mrmr_budget_fallback_mode": str(
                getattr(self.config, "fs_mrmr_budget_fallback_mode", "empty")
                or "empty"
            ),
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
            "fs_pathway_group_sparse_lasso_n_groups": int(
                getattr(self.config, "fs_pathway_group_sparse_lasso_n_groups", 50) or 50
            ),
            "fs_pathway_group_sparse_lasso_max_group_size": int(
                getattr(self.config, "fs_pathway_group_sparse_lasso_max_group_size", 50) or 50
            ),
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
            "include_tabentics_diakrino_model": bool(
                getattr(self._classification_cfg(), "include_tabentics_diakrino_model", False)
            ),
            "native_categorical_stage2_enabled": bool(
                getattr(
                    self._classification_cfg(),
                    "native_categorical_stage2_enabled",
                    False,
                )
            ),
            "native_categorical_stage2_estimator": str(
                getattr(
                    self._classification_cfg(),
                    "native_categorical_stage2_estimator",
                    "",
                )
                or ""
            ),
            "tabentics_diakrino_checkpoint": str(
                getattr(self._classification_cfg(), "tabentics_diakrino_checkpoint", "") or ""
            ),
            "tabentics_diakrino_max_features": int(
                getattr(self._classification_cfg(), "tabentics_diakrino_max_features", 256) or 256
            ),
            "tabentics_diakrino_batch_size": int(
                getattr(self._classification_cfg(), "tabentics_diakrino_batch_size", 32) or 32
            ),
            "tabentics_diakrino_support_joint_serving_cache": bool(
                getattr(self._classification_cfg(), "tabentics_diakrino_support_joint_serving_cache", False)
            ),
            "tabentics_diakrino_retry_cuda_oom_microbatch": bool(
                getattr(self._classification_cfg(), "tabentics_diakrino_retry_cuda_oom_microbatch", False)
            ),
            "tabentics_diakrino_device": str(
                getattr(self._classification_cfg(), "tabentics_diakrino_device", "auto") or "auto"
            ),
            "tabentics_diakrino_calibrate_probabilities": bool(
                getattr(self._classification_cfg(), "tabentics_diakrino_calibrate_probabilities", True)
            ),
            "tabentics_diakrino_calibration_fraction": float(
                getattr(self._classification_cfg(), "tabentics_diakrino_calibration_fraction", 0.20) or 0.20
            ),
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
            "model_cv_enable_svc_probability": bool(
                getattr(self._classification_cfg(), "enable_svc_probability", False)
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
            "classifier_oracle_enable_worst_class": bool(
                getattr(self._classification_cfg(), "oracle_include_worst_class_recall", True)
            ),
            "classifier_oracle_include_james_stein": bool(
                self._classification_cfg().oracle_include_james_stein
            ),
            "classifier_oracle_complexity_shrinkage": bool(
                getattr(self._classification_cfg(), "oracle_complexity_shrinkage", False)
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
            "classifier_posthoc_calibration_enabled": bool(
                getattr(self._classification_cfg(), "posthoc_calibration_enabled", False)
            ),
            "classifier_posthoc_calibration_method": str(
                getattr(self._classification_cfg(), "posthoc_calibration_method", "sigmoid")
            ),
            "classifier_posthoc_calibration_fraction": float(
                getattr(self._classification_cfg(), "posthoc_calibration_fraction", 0.20)
            ),
            "classifier_posthoc_calibration_min_calibration": int(
                getattr(self._classification_cfg(), "posthoc_calibration_min_calibration", 20)
            ),
            "classifier_posthoc_calibration_refinement_stopping": bool(
                getattr(self._classification_cfg(), "posthoc_calibration_refinement_stopping", True)
            ),
            "calibration_reporting_enabled": bool(
                getattr(self.config, "calibration_reporting_enabled", False)
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
            **(
                {
                    "maqc_pairing_score_mode": str(
                        self.config.maqc_pairing_score_mode
                    ),
                    "maqc_pairing_outer_splits": int(
                        self.config.maqc_pairing_outer_splits
                    ),
                    "maqc_pairing_outer_repeats": int(
                        self.config.maqc_pairing_outer_repeats
                    ),
                    "maqc_pairing_min_train_per_class": int(
                        self.config.maqc_pairing_min_train_per_class
                    ),
                    "maqc_pairing_seed_stride": int(
                        self.config.maqc_pairing_seed_stride
                    ),
                    "maqc_pairing_bbc_bootstrap_rounds": int(
                        self.config.maqc_pairing_bbc_bootstrap_rounds
                    ),
                    "maqc_pairing_bbc_ci_level": float(
                        self.config.maqc_pairing_bbc_ci_level
                    ),
                    "maqc_pairing_max_outer_evaluations": int(
                        self.config.maqc_pairing_max_outer_evaluations
                    ),
                    "maqc_pairing_max_runtime_seconds": float(
                        self.config.maqc_pairing_max_runtime_seconds
                    ),
                }
                if self._nested_pairing_mode() != "raw_cv"
                else {}
            ),
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
                "flex_family_distinct_gate_enabled": bool(
                    getattr(self.config.dist_config, "flex_family_distinct_gate_enabled", False)
                ),
                "flex_family_min_distinct": int(
                    getattr(self.config.dist_config, "flex_family_min_distinct", 15) or 15
                ),
                "flex_family_retention_margin": float(
                    getattr(self.config.dist_config, "flex_family_retention_margin", 0.0) or 0.0
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
                "diakrino_family_prior_lambda": float(
                    getattr(self.config.dist_config, "diakrino_family_prior_lambda", 0.0) or 0.0
                ),
                "diakrino_skip_fit_discrete_enabled": bool(
                    getattr(self.config.dist_config, "diakrino_skip_fit_discrete_enabled", False)
                ),
                "diakrino_sidecar_path": str(getattr(self.config.dist_config, "diakrino_sidecar_path", "") or ""),
                "diakrino_sidecar_dataset_id": str(getattr(self.config.dist_config, "diakrino_sidecar_dataset_id", "") or ""),
                "random_state": getattr(self.config.dist_config, "random_state", None),
            },
            "multimodal_fallback": str(getattr(self.config, "multimodal_fallback", "none") or "none"),
            "diakrino_cdf_trust_gate_enabled": bool(getattr(self.config, "diakrino_cdf_trust_gate_enabled", False)),
            "diakrino_cdf_trust_entropy_threshold": float(
                getattr(self.config, "diakrino_cdf_trust_entropy_threshold", 1.01) or 1.01
            ),
            "diakrino_cdf_trust_fallback": str(getattr(self.config, "diakrino_cdf_trust_fallback", "rank_gaussian") or "rank_gaussian"),
            "diakrino_stability_surrogate_enabled": bool(getattr(self.config, "diakrino_stability_surrogate_enabled", False)),
        }

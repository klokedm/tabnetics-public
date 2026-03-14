import concurrent.futures
import functools
import logging
from dataclasses import dataclass, field
from itertools import combinations
import math
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
import signal
import warnings
import zlib
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    KFold,
    LeaveOneOut,
    cross_val_score,
    RepeatedStratifiedKFold,
    RepeatedKFold
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif, # For classification
    f_regression, # For regression
    mutual_info_classif, # For classification
    mutual_info_regression, # For regression
    RFECV
)
from sklearn.linear_model import LassoCV, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# -- Copula knockoff selector (lives in parent experiments package) --
try:
    from ..copula_knockoff_selector import CopulaKnockoffSelector
except Exception as exc:
    try:
        from tabnetics.feature_selection.copula import CopulaKnockoffSelector
    except Exception as exc:
        try:
            from copula_knockoff_selector import CopulaKnockoffSelector
        except Exception as exc:
            CopulaKnockoffSelector = None

# -- Registry & stability (same package) --
from .registry import METHOD_REGISTRY, MethodSpec, get_method_weights, get_experimental_keys
from .stability_base import (
    SubsampleStability as _SubsampleStability,
    TigressStability as _TigressStability,
    SubspaceStability as _SubspaceStability,
    DecorrelatedStability as _DecorrelatedStability,
    ClusterStability as _ClusterStability,
)

import scipy.stats as sps

# NOTE: Scoped warning suppression is preferred over module-level.
# Module-level RuntimeWarning filter removed in audit T-A2-003.

# -- MNPO core (lives in parent experiments package) --
try:
    from ..mnpo_core import (
        aggregate_payoff_matrix as _mnpo_aggregate_payoff_matrix,
        apply_oracle_redundancy_penalty as _mnpo_apply_oracle_redundancy_penalty,
        fold_regret_mean_max as _mnpo_fold_regret_mean_max,
        fit_tritrust_weights as _mnpo_fit_tritrust_weights,
        lower_tail_cvar as _mnpo_lower_tail_cvar,
        matrix_from_scalar_scores as _mnpo_matrix_from_scalar_scores,
        mirror_descent_reference_regularized as _mnpo_mirror_descent_reference_regularized,
        normalize_vector_01 as _mnpo_normalize_vector_01,
        pairwise_pref_from_scalar as _mnpo_pairwise_pref_from_scalar,
        tremble_oracle_matrices as _mnpo_tremble_oracle_matrices,
    )
except Exception as exc:
    try:
        from tabnetics.core.mnpo import (
            aggregate_payoff_matrix as _mnpo_aggregate_payoff_matrix,
            apply_oracle_redundancy_penalty as _mnpo_apply_oracle_redundancy_penalty,
            fold_regret_mean_max as _mnpo_fold_regret_mean_max,
            fit_tritrust_weights as _mnpo_fit_tritrust_weights,
            lower_tail_cvar as _mnpo_lower_tail_cvar,
            matrix_from_scalar_scores as _mnpo_matrix_from_scalar_scores,
            mirror_descent_reference_regularized as _mnpo_mirror_descent_reference_regularized,
            normalize_vector_01 as _mnpo_normalize_vector_01,
            pairwise_pref_from_scalar as _mnpo_pairwise_pref_from_scalar,
            tremble_oracle_matrices as _mnpo_tremble_oracle_matrices,
        )
    except Exception as exc:
        from tabnetics.core.mnpo import (  # type: ignore
            aggregate_payoff_matrix as _mnpo_aggregate_payoff_matrix,
            apply_oracle_redundancy_penalty as _mnpo_apply_oracle_redundancy_penalty,
            fold_regret_mean_max as _mnpo_fold_regret_mean_max,
            fit_tritrust_weights as _mnpo_fit_tritrust_weights,
            lower_tail_cvar as _mnpo_lower_tail_cvar,
            matrix_from_scalar_scores as _mnpo_matrix_from_scalar_scores,
            mirror_descent_reference_regularized as _mnpo_mirror_descent_reference_regularized,
            normalize_vector_01 as _mnpo_normalize_vector_01,
            pairwise_pref_from_scalar as _mnpo_pairwise_pref_from_scalar,
            tremble_oracle_matrices as _mnpo_tremble_oracle_matrices,
        )


# Logger for feature selector operations (P0-3: replaces print() calls)
logger = logging.getLogger(__name__)

# -- Extracted modules (Phase 6) --
from .result import FeatureSelectionResult
from .scoring import _FeatureScoreCache
from .config import OracleConfig


class FeatureSelector:
    """
    Advanced Feature Selector with two strategies:
    1. `mnpo_portfolio` (default): MNPO (Nash Multi-Portfolio Optimization) selection.
    2. `legacy_voting`: legacy weighted ensemble voting.
    
    Methods:
    1. Stability Selection with Lasso
    2. RFECV (Recursive Feature Elimination with Cross-Validation)
    3. BorutaPy
    4. Gradient Boosting
    5. Linear SVM
    6. Mutual Information
    7. ANOVA F-test
    8. mRMR/JMI-style redundancy-aware selection
    9. k-TSP pairwise rank-rule selection
    10. Complementary-subsampling stability selection
    11. Integrated path stability selection (IPSS) with optional EATS thresholding
    12. Cluster stability selection for correlated feature groups
    13. GA/SVM-RFE (Genetic Algorithm with SVM-RFE) - conditional
    14. Derandomized Truncated D-vine Copula Knockoffs with e-values to control the false discovery rate(https://arxiv.org/pdf/2407.14002)
    15. Decorrelated stability selection (correlation-whitened sparse subsampling)
    16. Subspace stability selection (equivalent correlated-subspace models)
    17. TIGRESS-style randomized stability-path selection
    18. Optional rank-aggregation synthetic candidate (Borda or RRA)
    19. Optional ranking+wrapper greedy refinement over MNPO vote ranking
    20. Optional multiclass one-vs-all ensemble feature selector
    21. ECOC class-aware decomposition selector (multiclass)
    22. Iterative redundancy-pruning wrapper selector
    23. Joint multinomial shared-support selector (multiclass)
    24. Runtime-bounded iterative redundancy-pruning wrapper selector
    25. Class-specific relevance matrix with DOvE-style multiclass path
    26. Sparse multinomial multiclass selector (group-regularized path)
    27. Nearest shrunken centroids selector (PAM/NSC-style multiclass)
    28. Class-specific Pareto-front multiclass selector
    29. HSIC Lasso-style kernelized selector
    30. Univariate Wilcoxon–Mann–Whitney AUC filter (binary-only)
    
    Candidate generators use Leave-One-Out Cross-Validation (LOOCV) where applicable.
    """

    def __init__(self, 
                 n_folds=5, 
                 n_bootstrap_iterations=10,
                 random_state=42, 
                 problem_type='classification',
                 variance_threshold=0.01, 
                 correlation_threshold=0.90,
                 use_pca=False, 
                 n_components=0.95,
                 selection_strategy='mnpo_portfolio',
                 inner_cv_splits=5,
                 inner_cv_repeats=3,
                 pairwise_delta=0.01,
                 mirror_descent_steps=300,
                 mirror_descent_eta=0.15,
                 mirror_descent_lambda=0.08,
                 portfolio_size=6,
                 adaptive_portfolio_sizing_enabled=False,
                 adaptive_size_min=None,
                 adaptive_size_max=None,
                 adaptive_sizing_variance_penalty=False,
                 adaptive_sizing_variance_penalty_strength=0.5,
                 # T-R-266: Pareto-front portfolio sizing (opt-in).
                 pareto_portfolio_sizing_enabled=False,
                 # T-R-271: stability-weighted portfolio aggregation (opt-in).
                 stability_weighted_aggregation_enabled=False,
                 rashomon_enabled=False,
                 rashomon_max_models=12,
                 rashomon_score_tolerance=0.01,
                 use_tritrust=True,
                 use_stability_oracle=True,
                 use_complexity_oracle=True,
                 use_robust_oracle=True,
                 use_diversity_oracle=True,
                 use_cvar=False,
                 cvar_alpha=0.33,
                 use_tail_risk_oracle=False,
                 tail_risk_alpha=0.33,
                 use_regret_oracle=False,
                 use_qre_smoothing=False,
                 qre_temperature_gamma=1.0,
                 use_oracle_redundancy_penalty=False,
                 disable_redundancy_penalty_binary=True,
                 disable_class_pareto_binary=True,
                 compute_tremble_sensitivity=False,
                 oracle_weighting_mode="tritrust",
                 shapley_n_coalitions_max=4096,
                 shapley_bayesian_shrinkage=False,
                 shapley_bayesian_prior_strength=8.0,
                 use_interaction_oracle=False,
                 interaction_oracle_min_n_train=150,
                 interaction_oracle_pool_size_cap=64,
                 interaction_oracle_pair_cap=20000,
                 use_ubayfs=False,
                 ubayfs_n_bootstrap=32,
                 ubayfs_min_n=100,
                 ubayfs_prior_weight=0.0,
                 use_conformal_uq=False,
                 conformal_uq_alpha=0.10,
                 conformal_uq_min_folds=5,
                 fold_preference_mode="vote",
                 use_conformal_efficiency=False,
                 conformal_efficiency_method="split",
                 oracle_weight_js_shrinkage=False,
                 payoff_shrinkage_kappa=0.0,
                 diversity_oracle_mode='legacy_jaccard',
                 diversity_redundancy_weight=0.6,
                 diversity_complementarity_weight=0.35,
                 performance_balanced_weight=0.6,
                 performance_macro_f1_weight=0.4,
                 performance_use_adaptive_imbalance=False,
                 performance_imbalance_ratio_trigger=1.75,
                 performance_min_classes_for_adaptive=3,
                 rank_aggregation_mode='none',
                 wrapper_refine_enabled=False,
                 wrapper_refine_top_k=24,
                 wrapper_refine_max_add=12,
                 wrapper_refine_min_gain=1e-4,
                 ova_negative_ratio=2.0,
                 ova_min_classes=3,
                 ova_min_pos_samples=2,
                 ova_class_weight_mode="uniform",
                 ova_aggregation_mode="mean",
                 ova_aggregation_p=4.0,
                 ova_linear_backend="linear_svm_l1",
                 ova_enable_calibration=False,
                 ova_calibration_cv=3,
                 ecoc_min_classes=4,
                 ecoc_max_ovo_pairs=8,
                 ecoc_random_code_bits=4,
                 ecoc_class_complexity_weight=1.0,
                 ecoc_include_ova_tasks=True,
                 ecoc_negative_ratio=2.0,
                 joint_multiclass_min_classes=3,
                 joint_multiclass_max_features=256,
                 joint_multiclass_path_grid_size=6,
                 joint_multiclass_min_c=0.05,
                 joint_multiclass_max_c=1.6,
                 joint_multiclass_l1_ratio=0.55,
                 joint_multiclass_univariate_blend=0.20,
                 dove_min_classes=3,
                 dove_max_pairs_per_class=4,
                 dove_path_grid_size=5,
                 dove_specificity_weight=0.35,
                 dove_minority_boost=0.50,
                 sparse_multinomial_min_classes=3,
                 sparse_multinomial_max_features=320,
                 sparse_multinomial_path_grid_size=6,
                 sparse_multinomial_min_c=0.05,
                 sparse_multinomial_max_c=1.6,
                 sparse_multinomial_backend="mixed",
                 sparse_multinomial_l1_ratio=0.70,
                 sparse_multinomial_univariate_blend=0.20,
                 sparse_multinomial_max_iter=5000,
                 sparse_multinomial_screening_mode="none",
                 sparse_multinomial_screening_keep_fraction=1.0,
                 sparse_multinomial_screening_min_features=64,
                 sparse_multinomial_screening_fallback_on_failure=True,
                 nsc_shrinkage_grid_size=6,
                 nsc_min_classes=3,
                 nsc_thresholding_mode="soft",
                 nsc_order_quantile=0.75,
                 nsc_deep_shrinkage_search=False,
                 class_pareto_min_classes=3,
                 class_pareto_top_per_class=64,
                 class_pareto_global_fraction=0.40,
                 class_pareto_minority_boost=0.50,
                 class_pareto_kw_weight=0.25,
                 sdr_min_classes=3,
                 sdr_prefilter_max_features=512,
                 sdr_n_components=3,
                 sdr_covariance_ridge=1e-3,
                 per_class_quota_enabled=False,
                 per_class_quota_min_per_class=1,
                 per_class_quota_max_fraction=0.60,
                 hsic_lasso_alpha=0.01,
                 hsic_lasso_prefilter_max_features=128,
                 hsic_lasso_feature_sigma=0.0,
                 hsic_lasso_target_sigma=0.0,
                 hsic_lasso_relevance_blend=0.20,
                 hsic_lasso_max_iter=4000,
                 hsic_lasso_binary_delta_enabled=True,
                 hsic_lasso_binary_delta_min_samples=30,
                 slce_prefilter_max_features=1024,
                 slce_min_samples=30,
                 slce_ridge=1.0,
                 treeshap_min_samples=50,
                 treeshap_n_estimators=200,
                 treeshap_multi_seed_runs=3,
                 oaenet_min_samples=40,
                 oaenet_prescreen_max_features=512,
                 oaenet_l1_ratio=0.5,
                 oaenet_c_grid_size=6,
                 ktsp_max_features=120,
                 ktsp_k_pairs=24,
                 ktsp_max_pairs=12000,
                 mrmr_max_features=320,
                 mrmr_redundancy_weight=0.55,
                 mrmr_mi_redundancy_enabled=False,
                 mrmr_mi_n_bins=8,
                 cmim_min_samples=60,
                 cmim_n_bins=8,
                 fcbf_n_bins=8,
                 iterative_pruning_pool_factor=2.5,
                 iterative_pruning_max_rounds=32,
                 iterative_pruning_min_improvement=-0.002,
                 iterative_pruning_max_cumulative_loss=0.02,
                 iterative_pruning_redundancy_weight=0.65,
                 iterative_pruning_bounded_prefilter_cap=220,
                 iterative_pruning_bounded_candidate_fraction=0.35,
                 iterative_pruning_bounded_min_candidates=4,
                 iterative_pruning_bounded_max_evaluations=48,
                 iterative_pruning_bounded_max_runtime_seconds=30.0,
                 iterative_pruning_bounded_enable_class_gating=True,
                 iterative_pruning_bounded_multiclass_scale=0.70,
                 iterative_pruning_bounded_imbalance_trigger=2.5,
                 iterative_pruning_bounded_imbalance_scale=0.75,
                 iterative_pruning_bounded_use_cpss_overlay=False,
                 iterative_pruning_bounded_cpss_pairs=4,
                 iterative_pruning_bounded_cpss_stability_threshold=0.60,
                 iterative_pruning_bounded_cpss_min_stable_features=2,
                 iterative_pruning_bounded_cpss_min_jaccard=0.35,
                 iterative_pruning_bounded_cpss_max_score_drop=0.005,
                 iterative_pruning_class_pareto_prefilter_enabled=False,
                 iterative_pruning_class_pareto_min_classes=3,
                 iterative_pruning_class_pareto_top_per_class=64,
                 iterative_pruning_class_pareto_global_fraction=0.40,
                 iterative_pruning_class_pareto_minority_boost=0.50,
                 iterative_pruning_class_pareto_stability_gate_enabled=False,
                 iterative_pruning_class_pareto_stability_subsamples=6,
                 iterative_pruning_class_pareto_stability_fraction=0.70,
                 iterative_pruning_class_pareto_stability_threshold=0.55,
                 iterative_pruning_class_pareto_stability_min_overlap=0.50,
                 iterative_pruning_class_pareto_stability_min_stable_features=4,
                 iterative_pruning_class_pareto_stability_fallback_on_failure=True,
                 stability_subsample_fraction=0.5,
                 stability_selection_threshold=0.6,
                 stability_threshold_method='fixed',
                 stability_target_pfer=1.0,
                 stability_use_loss_guided_validation=False,
                 stability_validation_fraction=0.25,
                 stability_validation_quantile=0.40,
                 stability_validation_min_samples=6,
                 ipss_path_grid_size=7,
                 ipss_min_c=0.08,
                 ipss_max_c=1.20,
                 ipss_target_fdr=0.15,
                 ipss_null_shuffle_rounds=1,
                 ipss_use_eats_threshold=False,
                 ipss_eats_exclusion_quantile=0.90,
                 ipss_eats_min_threshold=0.45,
                 ipss_importance_model='linear_svm',
                 cluster_stability_corr_threshold=0.85,
                 cluster_stability_max_per_cluster=2,
                 cluster_stability_min_cluster_freq=0.55,
                 copula_knockoff_draws=30,
                 copula_alpha_kn=0.1,
                 copula_alpha_ebh=0.2,
                 copula_truncation_level=5,
                 copula_generator="copula",
                 copula_deepdrk_latent_fraction=0.35,
                 copula_deepdrk_noise_scale=1.0,
                 copula_derandomize_runs=1,
                 copula_stabilizer_runs=1,
                 copula_stabilizer_use_ebh=False,
                 copula_stabilizer_seed_stride=997,
                 importance_uq_enabled=False,
                 importance_uq_min_cv_folds=3,
                 group_sparse_lasso_alpha=0.1,
                 group_sparse_lasso_distance_threshold=0.7,
                 decorrelated_stability_eps=1e-3,
                 decorrelated_stability_min_max_abs_corr=0.0,
                 ipss_gate_min_classes=0,
                 ipss_gate_min_p_over_n=0.0,
                 portfolio_size_guard="none",
                 mnpo_include_legacy_consensus=True,
                 mnpo_include_majority_consensus=True,
                 mnpo_consensus_exclude_methods=None,
                 mnpo_consensus_exclude_protect_top_k=0,
                 mnpo_paradigm_aware_prior_enabled=False,
                 mnpo_interaction_floor=0.12,
                 runtime_racing_enabled=False,
                 runtime_racing_proxy_splits=1,
                 runtime_racing_keep_fraction=0.60,
                 runtime_racing_min_candidates=4,
                 runtime_racing_runtime_weight=0.15,
                 runtime_racing_mode="single_stage",
                 runtime_racing_stages=2,
                 runtime_racing_confidence_bound="none",
                 runtime_racing_delta=0.10,
                 complexity_use_runtime_penalty=False,
                 method_timeout_seconds=0.0,
                 parallel_n_jobs=1,
                 linear_svm_max_iter=10000,
                 enabled_methods=None,
                 prefilter_mi_weight=0.60,
                 prefilter_f_weight=0.40,
                 prefilter_union_enabled=False,
                 prefilter_strategies=("mi_ftest_blend",),
                 prefilter_nondefault_budget_fraction=0.10,
                 prefilter_wsnr_enabled=False,
                 prefilter_wsnr_stabilize_counts=True,
                 prefilter_data_domain="auto",
                 prefilter_rnaseq_transform_enabled=True,
                 prefilter_rnaseq_transform_force=False,
                 prefilter_rnaseq_nb_lrt_enabled=False,
                 prefilter_rnaseq_nb_lrt_alpha=0.10,
                 screening_enabled=False,
                 screening_method="none",
                 screening_pool_cap=2000,
                 screening_stir_n_neighbors=10,
                 screening_stir_n_iter=50,
                 screening_stir_keep_fraction=0.5,
                 screening_stir_min_features=20,
                 screening_evalue_alpha=0.20,
                 screening_evalue_min_features=20,
                 eval_models_enabled=False,
                 eval_models=("lr_l2", "linear_svc", "rf_small"),
                 eval_aggregate="mean",
                 eval_cvar_alpha=0.33,
                 eval_failure_strict_mode=False,
                 eval_model_weight_strategy="fixed",
                 performance_oracle_mode="single"):
        """
        Args:
            n_folds (int): Number of folds for cross-validation.
            n_bootstrap_iterations (int): Number of bootstrap iterations for each method.
            random_state (int): Random seed for reproducibility.
            problem_type (str): 'classification' or 'regression'.
            variance_threshold (float): Threshold for variance-based feature removal.
            correlation_threshold (float): Threshold for correlation-based feature removal.
            use_pca (bool): Whether to use PCA for dimensionality reduction.
            n_components (float): Number of components to keep in PCA.
            selection_strategy (str): 'mnpo_portfolio' or 'legacy_voting'.
            inner_cv_splits (int): Inner CV folds for oracle evaluation.
            inner_cv_repeats (int): Number of inner CV repeats.
            pairwise_delta (float): Practical margin for pairwise preference ties.
            mirror_descent_steps (int): Iterations for MNPO mirror descent.
            mirror_descent_eta (float): Mirror descent step size.
            mirror_descent_lambda (float): KL regularization toward reference prior.
            portfolio_size (int): Maximum number of selector candidates in final portfolio.
            use_tritrust (bool): Whether to learn trust/ignore/flip oracle weights.
            use_stability_oracle (bool): Use stability oracle in MNPO aggregation.
            use_complexity_oracle (bool): Use complexity oracle in MNPO aggregation.
            use_robust_oracle (bool): Use robustness oracle in MNPO aggregation.
            use_diversity_oracle (bool): Use diversity oracle in MNPO aggregation.
            use_tail_risk_oracle (bool): Add tail-risk oracle (CVaR over inner-CV folds; opt-in).
            tail_risk_alpha (float): Tail mass used by CVaR (mean of worst ceil(alpha*n_folds) folds).
            use_regret_oracle (bool): Add fold-regret oracle (mean/max regret vs per-fold best; opt-in).
            use_qre_smoothing (bool): QRE-style smoothing for scalar-oracle preferences (opt-in).
            qre_temperature_gamma (float): QRE temperature multiplier (higher = smoother preferences).
            use_oracle_redundancy_penalty (bool): Discount redundant oracle weights via Spearman redundancy penalty (opt-in).
            disable_redundancy_penalty_binary (bool): When ``n_classes == 2``, force ``use_oracle_redundancy_penalty=False``.
            disable_class_pareto_binary (bool): When ``n_classes == 2``, raise class-Pareto class count gate to 3.
            compute_tremble_sensitivity (bool): Compute tremble-sensitivity diagnostic (opt-in).
            diversity_oracle_mode (str): 'legacy_jaccard' or 'mi_redundancy'.
            diversity_redundancy_weight (float): Redundancy penalty in MI diversity mode.
            diversity_complementarity_weight (float): Complementarity reward in MI diversity mode.
            performance_balanced_weight (float): Balanced-accuracy weight in performance oracle.
            performance_macro_f1_weight (float): Macro-F1 weight in performance oracle.
            performance_use_adaptive_imbalance (bool): Adapt perf weights for imbalanced multiclass folds.
            performance_imbalance_ratio_trigger (float): Imbalance ratio threshold for adaptive weights.
            performance_min_classes_for_adaptive (int): Min classes to enable adaptive weighting.
            rank_aggregation_mode (str): Optional rank aggregation candidate ('none', 'borda', 'rra').
            wrapper_refine_enabled (bool): Enable post-MNPO greedy wrapper refinement.
            wrapper_refine_top_k (int): Top vote-ranked pool size considered by wrapper refinement.
            wrapper_refine_max_add (int): Max features chosen in wrapper phase before vote-fill.
            wrapper_refine_min_gain (float): Minimum score gain required to continue greedy wrapper adds.
            ova_negative_ratio (float): Negative:positive cap ratio in OVA class-balanced sampling.
            ova_min_classes (int): Minimum number of classes required to run OVA selection (multiclass-only gating).
            ova_min_pos_samples (int): Minimum positive samples required to include a class in OVA aggregation.
            ova_class_weight_mode (str): Optional class-aggregation weighting ('uniform', 'sqrt_pos', 'pos', 'log_pos',
                'inv_pos', 'inv_sqrt_pos', 'inv_log_pos').
            ova_aggregation_mode (str): OVA class-score aggregation across classes ('mean', 'p_norm').
            ova_aggregation_p (float): p for 'p_norm' aggregation (p>1 emphasizes class-specific peaks vs mean).
            ova_linear_backend (str): Linear scorer used inside OVA ('linear_svm_l1' or 'elastic_net_lr').
            ova_enable_calibration (bool): Enable class-conditional probability-calibration weighting in OVA scoring.
            ova_calibration_cv (int): CV folds used by OVA calibration stage.
            ecoc_min_classes (int): Minimum class count required to run ECOC class-aware selector.
            ecoc_max_ovo_pairs (int): Max number of confusable class-pair (OVO) ECOC tasks.
            ecoc_random_code_bits (int): Number of random ECOC dichotomy tasks.
            ecoc_class_complexity_weight (float): Weight multiplier for class-complexity-aware ECOC task weighting.
            ecoc_include_ova_tasks (bool): Include one-vs-all tasks in ECOC decomposition.
            ecoc_negative_ratio (float): Negative:positive cap ratio for ECOC one-vs-all tasks.
            joint_multiclass_min_classes (int): Minimum class count required for joint multinomial selector.
            joint_multiclass_max_features (int): Maximum prefiltered pool size for joint selector fitting.
            joint_multiclass_path_grid_size (int): Number of C values in multinomial regularization path.
            joint_multiclass_min_c (float): Minimum C for multinomial path.
            joint_multiclass_max_c (float): Maximum C for multinomial path.
            joint_multiclass_l1_ratio (float): Elastic-net L1 ratio for multinomial path fitting.
            joint_multiclass_univariate_blend (float): Blend weight for MI/F-score relevance in final joint scores.
            dove_min_classes (int): Minimum class count required for DOvE-style class-specific selector.
            dove_max_pairs_per_class (int): Maximum one-vs-each class pairs per class in DOvE selector.
            dove_path_grid_size (int): Number of support-size scaling steps in DOvE path search.
            dove_specificity_weight (float): Weight of class-specificity term in DOvE final scoring.
            dove_minority_boost (float): Inverse-count weighting exponent for minority-class emphasis.
            sparse_multinomial_min_classes (int): Minimum class count required for sparse multinomial selector.
            sparse_multinomial_max_features (int): Maximum prefiltered pool size for sparse multinomial fitting.
            sparse_multinomial_path_grid_size (int): Number of C values in sparse multinomial path.
            sparse_multinomial_min_c (float): Minimum C for sparse multinomial path.
            sparse_multinomial_max_c (float): Maximum C for sparse multinomial path.
            sparse_multinomial_backend (str): 'l1', 'elasticnet', or 'mixed' sparse multinomial backend.
            sparse_multinomial_l1_ratio (float): Elastic-net l1_ratio when sparse backend includes elastic-net.
            sparse_multinomial_univariate_blend (float): Blend weight for MI/F-score relevance in sparse scores.
            sparse_multinomial_max_iter (int): max_iter for sparse multinomial logistic fits.
            sparse_multinomial_screening_mode (str): Runtime-containment screening mode
                ('none', 'prefilter_aggressive', 'prefilter_balanced', 'prefilter_conservative').
                Mode semantics:
                - prefilter_aggressive: strong-rule-inspired screening.
                - prefilter_balanced: GAP-safe-inspired screening.
                - prefilter_conservative: Slores-inspired screening.
                Deprecated aliases ('strong', 'gap_safe', 'slores') are accepted with warning.
            sparse_multinomial_screening_keep_fraction (float): Fraction of pooled features retained by screening.
            sparse_multinomial_screening_min_features (int): Minimum retained features after screening.
            sparse_multinomial_screening_fallback_on_failure (bool): Fallback to unscreened pool if screened path fails.
            nsc_shrinkage_grid_size (int): Number of shrinkage values evaluated by nearest shrunken centroids selector.
            nsc_min_classes (int): Minimum class count required to run nearest shrunken centroids selector.
            nsc_thresholding_mode (str): NSC thresholding mode: 'soft', 'hard', 'quantile_hard', or 'auto'.
            nsc_order_quantile (float): Quantile used by NSC quantile_hard-threshold mode.
            nsc_deep_shrinkage_search (bool): Expand NSC shrinkage grid with quantile-based deltas.
            class_pareto_min_classes (int): Minimum class count required to run class-specific Pareto selector.
            class_pareto_top_per_class (int): Per-class candidate budget before Pareto dominance filtering.
            class_pareto_global_fraction (float): Additional global relevance pool fraction in Pareto candidate set.
            class_pareto_minority_boost (float): Minority-class weighting exponent for Pareto class scoring.
            class_pareto_kw_weight (float): Rank-based (KW-family) weight in class-Pareto per-class scoring.
            sdr_min_classes (int): Minimum class count required for SDR selectors (SIR/SAVE/PFC).
            sdr_prefilter_max_features (int): Max prefiltered pool size for SDR covariance estimation.
            sdr_n_components (int): Number of SDR directions used to score features.
            sdr_covariance_ridge (float): Ridge regularization added to SDR covariance matrices.
            per_class_quota_enabled (bool): Enable per-class quota overlay for class-specific selectors.
            per_class_quota_min_per_class (int): Minimum requested selected features per class under quota overlay.
            per_class_quota_max_fraction (float): Maximum fraction of final support that quota can force.
            hsic_lasso_alpha (float): L1 regularization strength for HSIC Lasso-style kernel regression.
            hsic_lasso_prefilter_max_features (int): Max prefiltered candidate features for HSIC Lasso fitting.
            hsic_lasso_feature_sigma (float): RBF sigma for feature kernels (<=0 uses per-feature median heuristic).
            hsic_lasso_target_sigma (float): RBF sigma for regression target kernel (<=0 uses median heuristic).
            hsic_lasso_relevance_blend (float): Blend of HSIC relevance into HSIC Lasso sparse coefficients.
            hsic_lasso_max_iter (int): max_iter for HSIC Lasso solver.
            ktsp_max_features (int): Max prefiltered feature count for k-TSP pair search.
            ktsp_k_pairs (int): Number of top pairwise rules used by k-TSP voting.
            ktsp_max_pairs (int): Hard cap on candidate feature pairs evaluated by k-TSP.
            mrmr_max_features (int): Max prefiltered candidate pool for mRMR/JMI-style selection.
            mrmr_redundancy_weight (float): Redundancy penalty weight in mRMR criterion.
            iterative_pruning_pool_factor (float): Initial candidate pool size multiplier for iterative pruning.
            iterative_pruning_max_rounds (int): Max removal rounds for iterative redundancy pruning.
            iterative_pruning_min_improvement (float): Minimum accepted score delta when pruning a feature.
            iterative_pruning_max_cumulative_loss (float): Maximum accepted cumulative loss (sum of per-round deltas).
            iterative_pruning_redundancy_weight (float): Redundancy-vs-relevance tradeoff in pruning priority.
            iterative_pruning_bounded_prefilter_cap (int): Max prefilter pool for runtime-bounded iterative pruning.
            iterative_pruning_bounded_candidate_fraction (float): Fraction of removal candidates evaluated per round.
            iterative_pruning_bounded_min_candidates (int): Minimum candidate removals evaluated per round.
            iterative_pruning_bounded_max_evaluations (int): Hard cap on wrapper score evaluations.
            iterative_pruning_bounded_max_runtime_seconds (float): Wall-clock cap for bounded iterative pruning.
            iterative_pruning_bounded_enable_class_gating (bool): Enable class-aware candidate-budget gating.
            iterative_pruning_bounded_multiclass_scale (float): Candidate-budget multiplier when classes > 2.
            iterative_pruning_bounded_imbalance_trigger (float): Imbalance ratio threshold for budget downscaling.
            iterative_pruning_bounded_imbalance_scale (float): Candidate-budget multiplier under heavy imbalance.
            iterative_pruning_bounded_use_cpss_overlay (bool): Enable CPSS-style stability overlay for bounded pruning.
            iterative_pruning_bounded_cpss_pairs (int): Number of complementary pairs used by CPSS overlay.
            iterative_pruning_bounded_cpss_stability_threshold (float): Stability frequency threshold for CPSS support.
            iterative_pruning_bounded_cpss_min_stable_features (int): Minimum stable-support size required to switch.
            iterative_pruning_bounded_cpss_min_jaccard (float): Minimum Jaccard overlap vs base bounded support.
            iterative_pruning_bounded_cpss_max_score_drop (float): Maximum allowed wrapper-score drop when switching.
            iterative_pruning_class_pareto_prefilter_enabled (bool): Enable class-dominance-aware Pareto prefilter.
            iterative_pruning_class_pareto_min_classes (int): Min class count required for Pareto prefilter.
            iterative_pruning_class_pareto_top_per_class (int): Per-class candidate budget before Pareto ranking.
            iterative_pruning_class_pareto_global_fraction (float): Global relevance contribution in Pareto candidate pool.
            iterative_pruning_class_pareto_minority_boost (float): Minority-class weighting exponent in Pareto prefilter.
            iterative_pruning_class_pareto_stability_gate_enabled (bool): Enable conservative stability gate on Pareto output.
            iterative_pruning_class_pareto_stability_subsamples (int): Number of stratified subsample rounds used by the stability gate.
            iterative_pruning_class_pareto_stability_fraction (float): Subsample train fraction used by the stability gate.
            iterative_pruning_class_pareto_stability_threshold (float): Stability-frequency threshold for stable Pareto candidates.
            iterative_pruning_class_pareto_stability_min_overlap (float): Minimum stable-overlap recall required to keep raw Pareto support.
            iterative_pruning_class_pareto_stability_min_stable_features (int): Minimum stable-support size required by the stability gate.
            iterative_pruning_class_pareto_stability_fallback_on_failure (bool): Fallback to global prefilter when stability gate fails hard.
            stability_subsample_fraction (float): Subsample fraction for complementary stability runs.
            stability_selection_threshold (float): Frequency threshold for stable feature acceptance.
            stability_threshold_method (str): Stability threshold mode ('fixed' | 'eats' | 'cpss').
            stability_target_pfer (float): Target PFER budget used by CPSS threshold calibration.
            stability_use_loss_guided_validation (bool): Enable OOS loss-guided fit filtering in stability subsampling.
            stability_validation_fraction (float): Fraction of each subsample reserved for validation.
            stability_validation_quantile (float): Quantile threshold used to keep high-quality subsample fits.
            stability_validation_min_samples (int): Minimum validation sample count for loss-guided split.
            ipss_path_grid_size (int): Number of regularization levels in IPSS path integration.
            ipss_min_c (float): Minimum C for linear-model IPSS path.
            ipss_max_c (float): Maximum C for linear-model IPSS path.
            ipss_target_fdr (float): Target FDR level used for q-value filtering in IPSS.
            ipss_null_shuffle_rounds (int): Number of shuffled-null rounds for IPSS calibration.
            ipss_use_eats_threshold (bool): Enable EATS-style data-adaptive threshold calibration.
            ipss_eats_exclusion_quantile (float): Null-score quantile for EATS exclusion filter.
            ipss_eats_min_threshold (float): Lower bound for EATS stable-threshold output.
            ipss_importance_model (str): Base model for IPSS path ('linear_svm', 'gradient_boosting', 'random_forest').
            cluster_stability_corr_threshold (float): Correlation threshold used to form clusters.
            cluster_stability_max_per_cluster (int): Max selected features per cluster in first pass.
            cluster_stability_min_cluster_freq (float): Min cluster stability frequency for priority pass.
            copula_knockoff_draws (int): Number of DTDCKe draws for copula knockoff aggregation.
            copula_alpha_kn (float): Per-draw knockoff threshold FDR target.
            copula_alpha_ebh (float): e-BH aggregation FDR target.
            copula_truncation_level (int or None): Optional D-vine truncation depth.
            copula_stabilizer_runs (int): Number of repeated outer copula runs for stabilization.
            copula_stabilizer_use_ebh (bool): Apply e-BH to stabilizer e-values from repeated supports.
            copula_stabilizer_seed_stride (int): Seed increment between stabilizer outer runs.
            importance_uq_enabled (bool): Enable reporting-only per-feature importance uncertainty diagnostics.
            importance_uq_min_cv_folds (int): Minimum successful folds required for importance-UQ reporting.
            decorrelated_stability_eps (float): Ridge regularizer for correlation inverse-sqrt transform.
            decorrelated_stability_min_max_abs_corr (float): Optional correlation gate for decorrelated stability.
                If >0, decorrelated stability returns an empty result when the prefiltered feature pool's
                `max_abs_pairwise_corr` falls below this threshold.
            ipss_gate_min_classes (int): Optional IPSS gate; if >0, IPSS runs only when `n_classes >= gate`.
            ipss_gate_min_p_over_n (float): Optional IPSS gate; if >0, IPSS runs only when `p/n >= gate`.
                If both IPSS gates are set (>0), IPSS runs when either condition is satisfied (OR-gate).
            portfolio_size_guard (str): Optional guard for `portfolio_size` vs enabled method count.
                One of: 'none', 'warn', 'raise'. Disabled by default.
            mnpo_include_legacy_consensus (bool): Whether to include the synthetic `legacy_consensus` candidate
                in the MNPO selector candidate library. Enabled by default to preserve baseline behavior.
            mnpo_include_majority_consensus (bool): Whether to include the synthetic `majority_consensus` candidate
                in the MNPO selector candidate library. Enabled by default to preserve baseline behavior.
            mnpo_consensus_exclude_methods (list[str] or None): Optional method names to exclude from the MNPO
                candidate library when synthetic consensus candidates are present. This can help avoid
                double-counting core methods that already feed `legacy_consensus` / `majority_consensus`
                when increasing `portfolio_size`. Disabled by default.
            mnpo_consensus_exclude_protect_top_k (int): When >0, protects the top-k candidates (by MNPO
                equilibrium weight) from exclusion. This is useful when you only want to exclude
                candidates that would otherwise enter the portfolio *because* `portfolio_size` was increased.
            runtime_racing_enabled (bool): Enable runtime-aware candidate racing before full MNPO evaluation.
            runtime_racing_proxy_splits (int): Number of inner-CV splits used for racing proxy scoring.
            runtime_racing_keep_fraction (float): Fraction of candidates retained for full MNPO evaluation.
            runtime_racing_min_candidates (int): Minimum number of candidates retained after racing.
            runtime_racing_runtime_weight (float): Runtime penalty strength in racing proxy scoring.
            runtime_racing_mode (str): Racing mode ('single_stage' or 'successive_halving').
            runtime_racing_stages (int): Number of stages used by successive-halving racing.
            runtime_racing_confidence_bound (str): Confidence bound for elimination ('none', 'hoeffding', 'bernstein').
            runtime_racing_delta (float): Tail probability for confidence-bound eliminations.
            complexity_use_runtime_penalty (bool): If True, penalize selectors by measured wall-clock runtime.
                Default is False because wall-clock times can vary run-to-run and harm determinism.
            method_timeout_seconds (float): Optional per-method wall-clock timeout (seconds). <=0 disables.
            parallel_n_jobs (int): Number of parallel threads for FS method dispatch. 1=sequential, -1=all CPUs.
            linear_svm_max_iter (int): Max iterations for all LinearSVC-based selectors (liblinear backend).
            enabled_methods (list[str] or None): Optional method allow-list.
            prefilter_mi_weight (float): Weight for MI scores in prefilter blend (default 0.60).
            prefilter_f_weight (float): Weight for F-test scores in prefilter blend (default 0.40).
        """
        self.n_folds = n_folds
        self.n_bootstrap_iterations = n_bootstrap_iterations
        self.random_state = random_state
        self.problem_type = problem_type
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.use_pca = use_pca
        self.n_components = n_components
        self.selection_strategy = selection_strategy
        self.inner_cv_splits = inner_cv_splits
        self.inner_cv_repeats = inner_cv_repeats
        self.pairwise_delta = pairwise_delta
        self.mirror_descent_steps = mirror_descent_steps
        self.mirror_descent_eta = mirror_descent_eta
        self.mirror_descent_lambda = mirror_descent_lambda
        self.portfolio_size = int(max(1, portfolio_size))
        self.adaptive_portfolio_sizing_enabled = bool(adaptive_portfolio_sizing_enabled)
        self.adaptive_size_min = None if adaptive_size_min is None else int(max(1, adaptive_size_min))
        self.adaptive_size_max = None if adaptive_size_max is None else int(max(1, adaptive_size_max))
        self.adaptive_sizing_variance_penalty = bool(adaptive_sizing_variance_penalty)
        self.adaptive_sizing_variance_penalty_strength = float(
            max(0.0, adaptive_sizing_variance_penalty_strength)
        )
        self.pareto_portfolio_sizing_enabled = bool(pareto_portfolio_sizing_enabled)
        self.stability_weighted_aggregation_enabled = bool(stability_weighted_aggregation_enabled)
        if self.adaptive_portfolio_sizing_enabled:
            if self.adaptive_size_min is None or self.adaptive_size_max is None:
                raise ValueError(
                    "adaptive portfolio sizing requires both adaptive_size_min and adaptive_size_max"
                )
            if self.adaptive_size_min > self.adaptive_size_max:
                raise ValueError(
                    "adaptive sizing bounds must satisfy 1 <= adaptive_size_min <= adaptive_size_max"
                )
            if not (self.adaptive_size_min <= self.portfolio_size <= self.adaptive_size_max):
                raise ValueError(
                    "portfolio_size must lie within [adaptive_size_min, adaptive_size_max] when adaptive sizing is enabled"
                )
        self.rashomon_enabled = bool(rashomon_enabled)
        self.rashomon_max_models = int(max(2, rashomon_max_models))
        self.rashomon_score_tolerance = float(max(0.0, rashomon_score_tolerance))
        self.portfolio_size_guard = str(portfolio_size_guard).strip().lower() if portfolio_size_guard is not None else "none"
        if self.portfolio_size_guard not in {"none", "warn", "raise"}:
            self.portfolio_size_guard = "none"
        self.mnpo_include_legacy_consensus = bool(mnpo_include_legacy_consensus)
        self.mnpo_include_majority_consensus = bool(mnpo_include_majority_consensus)
        if mnpo_consensus_exclude_methods is None:
            self.mnpo_consensus_exclude_methods = tuple()
        else:
            cleaned = []
            for name in mnpo_consensus_exclude_methods:
                s = str(name).strip().lower()
                if s:
                    cleaned.append(s)
            self.mnpo_consensus_exclude_methods = tuple(sorted(set(cleaned)))
        self.mnpo_consensus_exclude_protect_top_k = int(max(0, mnpo_consensus_exclude_protect_top_k))
        self.mnpo_paradigm_aware_prior_enabled = bool(mnpo_paradigm_aware_prior_enabled)
        self.mnpo_interaction_floor = float(np.clip(mnpo_interaction_floor, 0.0, 0.90))
        self.runtime_racing_enabled = bool(runtime_racing_enabled)
        self.runtime_racing_proxy_splits = int(max(1, runtime_racing_proxy_splits))
        self.runtime_racing_keep_fraction = float(np.clip(runtime_racing_keep_fraction, 0.10, 1.0))
        self.runtime_racing_min_candidates = int(max(1, runtime_racing_min_candidates))
        self.runtime_racing_runtime_weight = float(np.clip(runtime_racing_runtime_weight, 0.0, 2.0))
        self.runtime_racing_mode = str(runtime_racing_mode).strip().lower() if runtime_racing_mode is not None else "single_stage"
        if self.runtime_racing_mode not in {"single_stage", "successive_halving"}:
            self.runtime_racing_mode = "single_stage"
        self.runtime_racing_stages = int(max(1, runtime_racing_stages))
        self.runtime_racing_confidence_bound = (
            str(runtime_racing_confidence_bound).strip().lower()
            if runtime_racing_confidence_bound is not None
            else "none"
        )
        if self.runtime_racing_confidence_bound not in {"none", "hoeffding", "bernstein"}:
            self.runtime_racing_confidence_bound = "none"
        self.runtime_racing_delta = float(np.clip(runtime_racing_delta, 1e-6, 0.49))
        self.use_tritrust = bool(use_tritrust)
        self.use_stability_oracle = bool(use_stability_oracle)
        self.use_complexity_oracle = bool(use_complexity_oracle)
        self.complexity_use_runtime_penalty = bool(complexity_use_runtime_penalty)
        self.method_timeout_seconds = float(max(0.0, method_timeout_seconds))
        self.parallel_n_jobs = int(parallel_n_jobs) if parallel_n_jobs is not None else 1
        self.linear_svm_max_iter = int(max(1000, linear_svm_max_iter))
        self.use_robust_oracle = bool(use_robust_oracle)
        self.use_diversity_oracle = bool(use_diversity_oracle)
        self.use_cvar = bool(use_cvar)
        self.cvar_alpha = float(np.clip(cvar_alpha, 0.01, 1.0))
        self.use_tail_risk_oracle = bool(use_tail_risk_oracle)
        self.tail_risk_alpha = float(np.clip(tail_risk_alpha, 0.01, 1.0))
        self.use_regret_oracle = bool(use_regret_oracle)
        self.use_qre_smoothing = bool(use_qre_smoothing)
        self.qre_temperature_gamma = float(max(1e-6, qre_temperature_gamma))
        self.disable_redundancy_penalty_binary = bool(disable_redundancy_penalty_binary)
        self.disable_class_pareto_binary = bool(disable_class_pareto_binary)
        self.use_oracle_redundancy_penalty = bool(use_oracle_redundancy_penalty)
        self.compute_tremble_sensitivity = bool(compute_tremble_sensitivity)
        self.oracle_weighting_mode = (
            str(oracle_weighting_mode).strip().lower() if oracle_weighting_mode is not None else "tritrust"
        )
        if self.oracle_weighting_mode not in {"tritrust", "uniform", "shapley", "banzhaf"}:
            self.oracle_weighting_mode = "tritrust"
        self.shapley_n_coalitions_max = int(max(2, shapley_n_coalitions_max))
        self.shapley_bayesian_shrinkage = bool(shapley_bayesian_shrinkage)
        self.shapley_bayesian_prior_strength = float(max(1e-6, shapley_bayesian_prior_strength))
        self.use_interaction_oracle = bool(use_interaction_oracle)
        self.interaction_oracle_min_n_train = int(max(2, interaction_oracle_min_n_train))
        self.interaction_oracle_pool_size_cap = int(max(4, interaction_oracle_pool_size_cap))
        self.interaction_oracle_pair_cap = int(max(1, interaction_oracle_pair_cap))
        self.use_ubayfs = bool(use_ubayfs)
        self.ubayfs_n_bootstrap = int(max(1, ubayfs_n_bootstrap))
        self.ubayfs_min_n = int(max(1, ubayfs_min_n))
        self.ubayfs_prior_weight = float(np.clip(ubayfs_prior_weight, 0.0, 1.0))
        self.use_conformal_uq = bool(use_conformal_uq)
        self.conformal_uq_alpha = float(np.clip(conformal_uq_alpha, 1e-3, 0.49))
        self.conformal_uq_min_folds = int(max(2, conformal_uq_min_folds))
        self.fold_preference_mode = (
            str(fold_preference_mode).strip().lower()
            if fold_preference_mode is not None
            else "vote"
        )
        if self.fold_preference_mode not in {"vote", "logistic"}:
            self.fold_preference_mode = "vote"
        self.use_conformal_efficiency = bool(use_conformal_efficiency)
        self.conformal_efficiency_method = (
            str(conformal_efficiency_method).strip().lower()
            if conformal_efficiency_method is not None
            else "split"
        )
        if self.conformal_efficiency_method not in {"split", "aps"}:
            self.conformal_efficiency_method = "split"
        self.oracle_weight_js_shrinkage = bool(oracle_weight_js_shrinkage)
        self.payoff_shrinkage_kappa = float(max(0.0, payoff_shrinkage_kappa))
        self.diversity_oracle_mode = (
            str(diversity_oracle_mode).strip().lower() if diversity_oracle_mode is not None else "legacy_jaccard"
        )
        if self.diversity_oracle_mode not in {"legacy_jaccard", "mi_redundancy", "pid_mi", "complementarity"}:
            self.diversity_oracle_mode = "legacy_jaccard"
        self.diversity_redundancy_weight = float(np.clip(diversity_redundancy_weight, 0.0, 2.0))
        self.diversity_complementarity_weight = float(np.clip(diversity_complementarity_weight, 0.0, 2.0))
        self.oracle = OracleConfig(
            pairwise_delta=float(self.pairwise_delta),
            use_tritrust=bool(self.use_tritrust),
            use_stability_oracle=bool(self.use_stability_oracle),
            use_complexity_oracle=bool(self.use_complexity_oracle),
            use_robust_oracle=bool(self.use_robust_oracle),
            use_diversity_oracle=bool(self.use_diversity_oracle),
            use_cvar=bool(self.use_cvar),
            cvar_alpha=float(self.cvar_alpha),
            use_qre_smoothing=bool(self.use_qre_smoothing),
            qre_temperature_gamma=float(self.qre_temperature_gamma),
            use_oracle_redundancy_penalty=bool(self.use_oracle_redundancy_penalty),
            compute_tremble_sensitivity=bool(self.compute_tremble_sensitivity),
            diversity_mode=str(self.diversity_oracle_mode),
            diversity_redundancy_weight=float(self.diversity_redundancy_weight),
            diversity_complementarity_weight=float(self.diversity_complementarity_weight),
            performance_oracle_mode=str(getattr(self, "performance_oracle_mode", "single") or "single"),
            weighting_mode=str(self.oracle_weighting_mode),
            shapley_n_coalitions_max=int(self.shapley_n_coalitions_max),
            shapley_bayesian_shrinkage=bool(self.shapley_bayesian_shrinkage),
            shapley_bayesian_prior_strength=float(self.shapley_bayesian_prior_strength),
            use_interaction_oracle=bool(self.use_interaction_oracle),
            interaction_oracle_min_n_train=int(self.interaction_oracle_min_n_train),
            interaction_oracle_pool_size_cap=int(self.interaction_oracle_pool_size_cap),
            interaction_oracle_pair_cap=int(self.interaction_oracle_pair_cap),
            use_ubayfs=bool(self.use_ubayfs),
            ubayfs_n_bootstrap=int(self.ubayfs_n_bootstrap),
            ubayfs_min_n=int(self.ubayfs_min_n),
            ubayfs_prior_weight=float(self.ubayfs_prior_weight),
            use_conformal_uq=bool(self.use_conformal_uq),
            conformal_uq_alpha=float(self.conformal_uq_alpha),
            conformal_uq_min_folds=int(self.conformal_uq_min_folds),
            fold_preference_mode=str(self.fold_preference_mode),
            use_conformal_efficiency=bool(self.use_conformal_efficiency),
            conformal_efficiency_method=str(self.conformal_efficiency_method),
            oracle_weight_js_shrinkage=bool(self.oracle_weight_js_shrinkage),
        )
        self.performance_balanced_weight = float(max(0.0, performance_balanced_weight))
        self.performance_macro_f1_weight = float(max(0.0, performance_macro_f1_weight))
        self.performance_use_adaptive_imbalance = bool(performance_use_adaptive_imbalance)
        self.performance_imbalance_ratio_trigger = float(max(1.0, performance_imbalance_ratio_trigger))
        self.performance_min_classes_for_adaptive = int(max(2, performance_min_classes_for_adaptive))
        self.rank_aggregation_mode = str(rank_aggregation_mode).strip().lower()
        if self.rank_aggregation_mode not in {"none", "borda", "rra"}:
            self.rank_aggregation_mode = "none"
        self.wrapper_refine_enabled = bool(wrapper_refine_enabled)
        self.wrapper_refine_top_k = int(max(4, wrapper_refine_top_k))
        self.wrapper_refine_max_add = int(max(1, wrapper_refine_max_add))
        self.wrapper_refine_min_gain = float(max(0.0, wrapper_refine_min_gain))
        self.ova_negative_ratio = float(np.clip(ova_negative_ratio, 1.0, 10.0))
        self.ova_min_classes = int(max(3, ova_min_classes))
        self.ova_min_pos_samples = int(max(2, ova_min_pos_samples))
        self.ova_class_weight_mode = str(ova_class_weight_mode).strip().lower() if ova_class_weight_mode is not None else "uniform"
        if self.ova_class_weight_mode not in {
            "uniform",
            "sqrt_pos",
            "pos",
            "log_pos",
            "inv_pos",
            "inv_sqrt_pos",
            "inv_log_pos",
        }:
            self.ova_class_weight_mode = "uniform"
        self.ova_aggregation_mode = (
            str(ova_aggregation_mode).strip().lower() if ova_aggregation_mode is not None else "mean"
        )
        if self.ova_aggregation_mode in {"pnorm", "p-norm", "p_norm"}:
            self.ova_aggregation_mode = "p_norm"
        if self.ova_aggregation_mode not in {"mean", "p_norm"}:
            self.ova_aggregation_mode = "mean"
        try:
            p = float(ova_aggregation_p)
        except Exception as exc:
            p = 4.0
        if not np.isfinite(p) or p <= 0.0:
            p = 4.0
        self.ova_aggregation_p = float(p)
        self.ova_linear_backend = str(ova_linear_backend).strip().lower() if ova_linear_backend is not None else "linear_svm_l1"
        if self.ova_linear_backend in {"svm", "linear_svm"}:
            self.ova_linear_backend = "linear_svm_l1"
        if self.ova_linear_backend not in {"linear_svm_l1", "elastic_net_lr"}:
            self.ova_linear_backend = "linear_svm_l1"
        self.ova_enable_calibration = bool(ova_enable_calibration)
        self.ova_calibration_cv = int(max(2, ova_calibration_cv))
        self.ecoc_min_classes = int(max(3, ecoc_min_classes))
        self.ecoc_max_ovo_pairs = int(max(0, ecoc_max_ovo_pairs))
        self.ecoc_random_code_bits = int(max(0, ecoc_random_code_bits))
        self.ecoc_class_complexity_weight = float(np.clip(ecoc_class_complexity_weight, 0.0, 5.0))
        self.ecoc_include_ova_tasks = bool(ecoc_include_ova_tasks)
        self.ecoc_negative_ratio = float(np.clip(ecoc_negative_ratio, 1.0, 10.0))
        self.joint_multiclass_min_classes = int(max(3, joint_multiclass_min_classes))
        self.joint_multiclass_max_features = int(max(32, joint_multiclass_max_features))
        self.joint_multiclass_path_grid_size = int(max(2, joint_multiclass_path_grid_size))
        self.joint_multiclass_min_c = float(max(1e-3, joint_multiclass_min_c))
        self.joint_multiclass_max_c = float(max(self.joint_multiclass_min_c + 1e-3, joint_multiclass_max_c))
        self.joint_multiclass_l1_ratio = float(np.clip(joint_multiclass_l1_ratio, 0.0, 1.0))
        self.joint_multiclass_univariate_blend = float(np.clip(joint_multiclass_univariate_blend, 0.0, 0.80))
        self.dove_min_classes = int(max(3, dove_min_classes))
        self.dove_max_pairs_per_class = int(max(1, dove_max_pairs_per_class))
        self.dove_path_grid_size = int(max(2, dove_path_grid_size))
        self.dove_specificity_weight = float(np.clip(dove_specificity_weight, 0.0, 0.90))
        self.dove_minority_boost = float(np.clip(dove_minority_boost, 0.0, 2.0))
        self.sparse_multinomial_min_classes = int(max(3, sparse_multinomial_min_classes))
        self.sparse_multinomial_max_features = int(max(32, sparse_multinomial_max_features))
        self.sparse_multinomial_path_grid_size = int(max(2, sparse_multinomial_path_grid_size))
        self.sparse_multinomial_min_c = float(max(1e-3, sparse_multinomial_min_c))
        self.sparse_multinomial_max_c = float(
            max(self.sparse_multinomial_min_c + 1e-3, sparse_multinomial_max_c)
        )
        self.sparse_multinomial_backend = (
            str(sparse_multinomial_backend).strip().lower()
            if sparse_multinomial_backend is not None
            else "mixed"
        )
        if self.sparse_multinomial_backend not in {"l1", "elasticnet", "mixed"}:
            self.sparse_multinomial_backend = "mixed"
        self.sparse_multinomial_l1_ratio = float(np.clip(sparse_multinomial_l1_ratio, 0.0, 1.0))
        self.sparse_multinomial_univariate_blend = float(
            np.clip(sparse_multinomial_univariate_blend, 0.0, 0.80)
        )
        self.sparse_multinomial_max_iter = int(max(1000, sparse_multinomial_max_iter))
        screening_mode_raw = (
            str(sparse_multinomial_screening_mode).strip().lower()
            if sparse_multinomial_screening_mode is not None
            else "none"
        )
        screening_alias_map = {
            "strong": "prefilter_aggressive",
            "gap_safe": "prefilter_balanced",
            "slores": "prefilter_conservative",
        }
        self.sparse_multinomial_screening_mode_deprecated_alias_used = False
        if screening_mode_raw in screening_alias_map:
            self.sparse_multinomial_screening_mode_deprecated_alias_used = True
            warnings.warn(
                f"sparse_multinomial_screening_mode='{screening_mode_raw}' is deprecated; "
                f"use '{screening_alias_map[screening_mode_raw]}' instead.",
                DeprecationWarning,
            )
            screening_mode_raw = screening_alias_map[screening_mode_raw]
        if screening_mode_raw not in {
            "none",
            "prefilter_aggressive",
            "prefilter_balanced",
            "prefilter_conservative",
        }:
            screening_mode_raw = "none"
        self.sparse_multinomial_screening_mode = screening_mode_raw
        self.sparse_multinomial_screening_keep_fraction = float(
            np.clip(sparse_multinomial_screening_keep_fraction, 0.05, 1.0)
        )
        self.sparse_multinomial_screening_min_features = int(max(8, sparse_multinomial_screening_min_features))
        self.sparse_multinomial_screening_fallback_on_failure = bool(
            sparse_multinomial_screening_fallback_on_failure
        )
        self.nsc_shrinkage_grid_size = int(max(2, nsc_shrinkage_grid_size))
        self.nsc_min_classes = int(max(2, nsc_min_classes))
        self.nsc_thresholding_mode = str(nsc_thresholding_mode).strip().lower()
        if self.nsc_thresholding_mode in {"pnorm", "p-norm", "p_norm"}:
            self.nsc_thresholding_mode = "soft"
        if self.nsc_thresholding_mode == "order":
            self.nsc_thresholding_mode = "quantile_hard"  # backward compat rename
        if self.nsc_thresholding_mode not in {"soft", "hard", "quantile_hard", "auto"}:
            self.nsc_thresholding_mode = "soft"
        self.nsc_order_quantile = float(np.clip(nsc_order_quantile, 0.50, 0.99))
        self.nsc_deep_shrinkage_search = bool(nsc_deep_shrinkage_search)
        self.class_pareto_min_classes = int(max(2, class_pareto_min_classes))
        self.class_pareto_top_per_class = int(max(4, class_pareto_top_per_class))
        self.class_pareto_global_fraction = float(np.clip(class_pareto_global_fraction, 0.0, 1.0))
        self.class_pareto_minority_boost = float(np.clip(class_pareto_minority_boost, 0.0, 2.0))
        self.class_pareto_kw_weight = float(np.clip(class_pareto_kw_weight, 0.0, 0.80))
        self.sdr_min_classes = int(max(2, sdr_min_classes))
        self.sdr_prefilter_max_features = int(max(16, sdr_prefilter_max_features))
        self.sdr_n_components = int(max(1, sdr_n_components))
        self.sdr_covariance_ridge = float(max(1e-8, sdr_covariance_ridge))
        self.per_class_quota_enabled = bool(per_class_quota_enabled)
        self.per_class_quota_min_per_class = int(max(1, per_class_quota_min_per_class))
        self.per_class_quota_max_fraction = float(np.clip(per_class_quota_max_fraction, 0.05, 1.0))
        self.hsic_lasso_alpha = float(np.clip(hsic_lasso_alpha, 1e-6, 10.0))
        self.hsic_lasso_prefilter_max_features = int(max(8, hsic_lasso_prefilter_max_features))
        self.hsic_lasso_feature_sigma = float(max(0.0, hsic_lasso_feature_sigma))
        self.hsic_lasso_target_sigma = float(max(0.0, hsic_lasso_target_sigma))
        self.hsic_lasso_relevance_blend = float(np.clip(hsic_lasso_relevance_blend, 0.0, 1.0))
        self.hsic_lasso_max_iter = int(max(1000, hsic_lasso_max_iter))
        self.hsic_lasso_binary_delta_enabled = bool(hsic_lasso_binary_delta_enabled)
        self.hsic_lasso_binary_delta_min_samples = int(max(2, hsic_lasso_binary_delta_min_samples))
        self.slce_prefilter_max_features = int(max(8, slce_prefilter_max_features))
        self.slce_min_samples = int(max(2, slce_min_samples))
        self.slce_ridge = float(max(1e-8, slce_ridge))
        self.treeshap_min_samples = int(max(2, treeshap_min_samples))
        self.treeshap_n_estimators = int(max(50, treeshap_n_estimators))
        self.treeshap_multi_seed_runs = int(max(1, treeshap_multi_seed_runs))
        self.oaenet_min_samples = int(max(2, oaenet_min_samples))
        self.oaenet_prescreen_max_features = int(max(16, oaenet_prescreen_max_features))
        self.oaenet_l1_ratio = float(np.clip(oaenet_l1_ratio, 0.05, 0.95))
        self.oaenet_c_grid_size = int(max(3, oaenet_c_grid_size))
        self.ktsp_max_features = int(max(16, ktsp_max_features))
        self.ktsp_k_pairs = int(max(4, ktsp_k_pairs))
        self.ktsp_max_pairs = int(max(100, ktsp_max_pairs))
        self.mrmr_max_features = int(max(24, mrmr_max_features))
        self.mrmr_redundancy_weight = float(np.clip(mrmr_redundancy_weight, 0.0, 2.0))
        self.mrmr_mi_redundancy_enabled = bool(mrmr_mi_redundancy_enabled)
        self.mrmr_mi_n_bins = int(max(2, mrmr_mi_n_bins))
        self.cmim_min_samples = int(max(2, cmim_min_samples))
        self.cmim_n_bins = int(max(2, cmim_n_bins))
        self.fcbf_n_bins = int(max(2, fcbf_n_bins))
        self.iterative_pruning_pool_factor = float(np.clip(iterative_pruning_pool_factor, 1.2, 8.0))
        self.iterative_pruning_max_rounds = int(max(1, iterative_pruning_max_rounds))
        self.iterative_pruning_min_improvement = float(np.clip(iterative_pruning_min_improvement, -0.10, 0.10))
        self.iterative_pruning_max_cumulative_loss = float(
            np.clip(iterative_pruning_max_cumulative_loss, 0.0, 0.50)
        )
        self.iterative_pruning_redundancy_weight = float(np.clip(iterative_pruning_redundancy_weight, 0.0, 1.0))
        self.iterative_pruning_bounded_prefilter_cap = int(max(48, iterative_pruning_bounded_prefilter_cap))
        self.iterative_pruning_bounded_candidate_fraction = float(
            np.clip(iterative_pruning_bounded_candidate_fraction, 0.05, 1.0)
        )
        self.iterative_pruning_bounded_min_candidates = int(max(1, iterative_pruning_bounded_min_candidates))
        self.iterative_pruning_bounded_max_evaluations = int(max(1, iterative_pruning_bounded_max_evaluations))
        self.iterative_pruning_bounded_max_runtime_seconds = float(
            np.clip(iterative_pruning_bounded_max_runtime_seconds, 1.0, 1200.0)
        )
        self.iterative_pruning_bounded_enable_class_gating = bool(iterative_pruning_bounded_enable_class_gating)
        self.iterative_pruning_bounded_multiclass_scale = float(
            np.clip(iterative_pruning_bounded_multiclass_scale, 0.10, 1.0)
        )
        self.iterative_pruning_bounded_imbalance_trigger = float(max(1.0, iterative_pruning_bounded_imbalance_trigger))
        self.iterative_pruning_bounded_imbalance_scale = float(
            np.clip(iterative_pruning_bounded_imbalance_scale, 0.10, 1.0)
        )
        self.iterative_pruning_bounded_use_cpss_overlay = bool(iterative_pruning_bounded_use_cpss_overlay)
        self.iterative_pruning_bounded_cpss_pairs = int(max(1, iterative_pruning_bounded_cpss_pairs))
        self.iterative_pruning_bounded_cpss_stability_threshold = float(
            np.clip(iterative_pruning_bounded_cpss_stability_threshold, 0.35, 0.99)
        )
        self.iterative_pruning_bounded_cpss_min_stable_features = int(
            max(1, iterative_pruning_bounded_cpss_min_stable_features)
        )
        self.iterative_pruning_bounded_cpss_min_jaccard = float(
            np.clip(iterative_pruning_bounded_cpss_min_jaccard, 0.0, 1.0)
        )
        self.iterative_pruning_bounded_cpss_max_score_drop = float(
            np.clip(iterative_pruning_bounded_cpss_max_score_drop, 0.0, 0.20)
        )
        self.iterative_pruning_class_pareto_prefilter_enabled = bool(iterative_pruning_class_pareto_prefilter_enabled)
        self.iterative_pruning_class_pareto_min_classes = int(max(2, iterative_pruning_class_pareto_min_classes))
        self.iterative_pruning_class_pareto_top_per_class = int(max(4, iterative_pruning_class_pareto_top_per_class))
        self.iterative_pruning_class_pareto_global_fraction = float(
            np.clip(iterative_pruning_class_pareto_global_fraction, 0.0, 1.0)
        )
        self.iterative_pruning_class_pareto_minority_boost = float(
            np.clip(iterative_pruning_class_pareto_minority_boost, 0.0, 2.0)
        )
        self.iterative_pruning_class_pareto_stability_gate_enabled = bool(
            iterative_pruning_class_pareto_stability_gate_enabled
        )
        self.iterative_pruning_class_pareto_stability_subsamples = int(
            max(1, iterative_pruning_class_pareto_stability_subsamples)
        )
        self.iterative_pruning_class_pareto_stability_fraction = float(
            np.clip(iterative_pruning_class_pareto_stability_fraction, 0.35, 0.95)
        )
        self.iterative_pruning_class_pareto_stability_threshold = float(
            np.clip(iterative_pruning_class_pareto_stability_threshold, 0.35, 0.99)
        )
        self.iterative_pruning_class_pareto_stability_min_overlap = float(
            np.clip(iterative_pruning_class_pareto_stability_min_overlap, 0.0, 1.0)
        )
        self.iterative_pruning_class_pareto_stability_min_stable_features = int(
            max(1, iterative_pruning_class_pareto_stability_min_stable_features)
        )
        self.iterative_pruning_class_pareto_stability_fallback_on_failure = bool(
            iterative_pruning_class_pareto_stability_fallback_on_failure
        )
        self.stability_subsample_fraction = float(np.clip(stability_subsample_fraction, 0.35, 0.8))
        self.stability_selection_threshold = float(np.clip(stability_selection_threshold, 0.45, 0.95))
        self.stability_threshold_method = str(stability_threshold_method or "fixed").strip().lower()
        if self.stability_threshold_method not in {"fixed", "eats", "cpss"}:
            self.stability_threshold_method = "fixed"
        self.stability_target_pfer = float(max(1e-6, stability_target_pfer))
        self.stability_use_loss_guided_validation = bool(stability_use_loss_guided_validation)
        self.stability_validation_fraction = float(np.clip(stability_validation_fraction, 0.10, 0.50))
        self.stability_validation_quantile = float(np.clip(stability_validation_quantile, 0.05, 0.95))
        self.stability_validation_min_samples = int(max(3, stability_validation_min_samples))
        self.ipss_path_grid_size = int(max(3, ipss_path_grid_size))
        self.ipss_min_c = float(max(1e-3, ipss_min_c))
        self.ipss_max_c = float(max(self.ipss_min_c + 1e-3, ipss_max_c))
        self.ipss_target_fdr = float(np.clip(ipss_target_fdr, 1e-3, 0.5))
        self.ipss_null_shuffle_rounds = int(max(1, ipss_null_shuffle_rounds))
        self.ipss_use_eats_threshold = bool(ipss_use_eats_threshold)
        self.ipss_eats_exclusion_quantile = float(np.clip(ipss_eats_exclusion_quantile, 0.5, 0.995))
        self.ipss_eats_min_threshold = float(np.clip(ipss_eats_min_threshold, 0.30, 0.95))
        self.ipss_importance_model = str(ipss_importance_model).strip().lower()
        if self.ipss_importance_model not in {"linear_svm", "gradient_boosting", "random_forest"}:
            self.ipss_importance_model = "linear_svm"
        self.ipss_gate_min_classes = int(max(0, ipss_gate_min_classes))
        self.ipss_gate_min_p_over_n = float(max(0.0, ipss_gate_min_p_over_n))
        self.cluster_stability_corr_threshold = float(np.clip(cluster_stability_corr_threshold, 0.35, 0.99))
        self.cluster_stability_max_per_cluster = int(max(1, cluster_stability_max_per_cluster))
        self.cluster_stability_min_cluster_freq = float(np.clip(cluster_stability_min_cluster_freq, 0.15, 0.99))
        self.copula_knockoff_draws = int(max(1, copula_knockoff_draws))
        self.copula_alpha_kn = float(np.clip(copula_alpha_kn, 1e-4, 0.49))
        self.copula_alpha_ebh = float(np.clip(copula_alpha_ebh, 1e-4, 0.49))
        self.copula_truncation_level = (
            None if copula_truncation_level is None else int(max(1, copula_truncation_level))
        )
        self.copula_generator = str(copula_generator or "copula").strip().lower()
        if self.copula_generator not in {"copula", "deepdrk"}:
            self.copula_generator = "copula"
        self.copula_deepdrk_latent_fraction = float(np.clip(copula_deepdrk_latent_fraction, 0.05, 1.0))
        self.copula_deepdrk_noise_scale = float(max(0.0, copula_deepdrk_noise_scale))
        self.copula_derandomize_runs = int(max(1, copula_derandomize_runs))
        self.copula_stabilizer_runs = int(max(1, copula_stabilizer_runs))
        self.copula_stabilizer_use_ebh = bool(copula_stabilizer_use_ebh)
        self.copula_stabilizer_seed_stride = int(max(1, copula_stabilizer_seed_stride))
        self.importance_uq_enabled = bool(importance_uq_enabled)
        self.importance_uq_min_cv_folds = int(max(1, importance_uq_min_cv_folds))
        self.group_sparse_lasso_alpha = float(np.clip(group_sparse_lasso_alpha, 0.001, 10.0))
        self.group_sparse_lasso_distance_threshold = float(np.clip(group_sparse_lasso_distance_threshold, 0.1, 1.0))
        self.decorrelated_stability_eps = float(np.clip(decorrelated_stability_eps, 1e-6, 1.0))
        self.decorrelated_stability_min_max_abs_corr = float(
            np.clip(decorrelated_stability_min_max_abs_corr, 0.0, 1.0)
        )
        self.enabled_methods = set(enabled_methods) if enabled_methods is not None else None
        self.prefilter_mi_weight = float(np.clip(prefilter_mi_weight, 0.0, 1.0))
        self.prefilter_f_weight = float(np.clip(prefilter_f_weight, 0.0, 1.0))
        self.prefilter_union_enabled = bool(prefilter_union_enabled)
        if prefilter_strategies is None:
            self.prefilter_strategies = ("mi_ftest_blend",)
        else:
            cleaned = tuple(
                str(name).strip().lower()
                for name in prefilter_strategies
                if str(name).strip()
            )
            self.prefilter_strategies = cleaned if cleaned else ("mi_ftest_blend",)
        self.prefilter_wsnr_enabled = bool(prefilter_wsnr_enabled)
        self.prefilter_wsnr_stabilize_counts = bool(prefilter_wsnr_stabilize_counts)
        self.prefilter_data_domain = str(prefilter_data_domain or "auto").strip().lower()
        if self.prefilter_data_domain not in {"auto", "rnaseq", "generic"}:
            self.prefilter_data_domain = "auto"
        self.prefilter_rnaseq_transform_enabled = bool(prefilter_rnaseq_transform_enabled)
        self.prefilter_rnaseq_transform_force = bool(prefilter_rnaseq_transform_force)
        self.prefilter_rnaseq_nb_lrt_enabled = bool(prefilter_rnaseq_nb_lrt_enabled)
        self.prefilter_rnaseq_nb_lrt_alpha = float(
            np.clip(prefilter_rnaseq_nb_lrt_alpha, 1e-6, 0.5)
        )
        if self.prefilter_wsnr_enabled and "wsnr" not in set(self.prefilter_strategies):
            self.prefilter_strategies = tuple(list(self.prefilter_strategies) + ["wsnr"])
        self.prefilter_nondefault_budget_fraction = float(
            np.clip(prefilter_nondefault_budget_fraction, 0.01, 0.50)
        )
        # ── Tier 2 screening (T-004) ──────────────────────────────
        self.screening_enabled = bool(screening_enabled)
        self.screening_method = str(screening_method).strip().lower() if screening_method else "none"
        self.screening_pool_cap = int(max(1, screening_pool_cap))
        self.screening_stir_n_neighbors = int(max(1, screening_stir_n_neighbors))
        self.screening_stir_n_iter = int(max(1, screening_stir_n_iter))
        self.screening_stir_keep_fraction = float(np.clip(screening_stir_keep_fraction, 0.01, 1.0))
        self.screening_stir_min_features = int(max(1, screening_stir_min_features))
        self.screening_evalue_alpha = float(np.clip(screening_evalue_alpha, 1e-4, 0.95))
        self.screening_evalue_min_features = int(max(1, screening_evalue_min_features))
        # ── Multi-classifier evaluation proxy (T-001) ────────────────
        self.eval_models_enabled = bool(eval_models_enabled)
        if eval_models is None:
            self.eval_models = ("lr_l2", "linear_svc", "rf_small")
        else:
            self.eval_models = tuple(str(m).strip().lower() for m in eval_models)
        self.eval_aggregate = str(eval_aggregate).strip().lower() if eval_aggregate else "mean"
        if self.eval_aggregate not in {"mean", "min", "cvar"}:
            self.eval_aggregate = "mean"
        self.eval_cvar_alpha = float(np.clip(eval_cvar_alpha, 0.01, 1.0))
        self.eval_failure_strict_mode = bool(eval_failure_strict_mode)
        self.eval_model_weight_strategy = str(eval_model_weight_strategy or "fixed").strip().lower()
        if self.eval_model_weight_strategy not in {"fixed", "learned"}:
            self.eval_model_weight_strategy = "fixed"
        self._eval_multimodel_fold_log = []  # per-model diagnostics accumulator
        # ── Multi-model MNPO oracles (T-002) ──────────────────────────
        _pom = str(performance_oracle_mode).strip().lower() if performance_oracle_mode else "single"
        if _pom not in {"single", "multi_model_oracles"}:
            _pom = "single"
        self.performance_oracle_mode = _pom
        # Keep the nested oracle config in sync with runtime attributes.
        self.oracle.performance_oracle_mode = str(self.performance_oracle_mode)
        self.oracle.pairwise_delta = float(self.pairwise_delta)
        self.oracle.use_tritrust = bool(self.use_tritrust)
        self.oracle.use_stability_oracle = bool(self.use_stability_oracle)
        self.oracle.use_complexity_oracle = bool(self.use_complexity_oracle)
        self.oracle.use_robust_oracle = bool(self.use_robust_oracle)
        self.oracle.use_diversity_oracle = bool(self.use_diversity_oracle)
        self.oracle.use_cvar = bool(self.use_cvar)
        self.oracle.cvar_alpha = float(self.cvar_alpha)
        self.oracle.use_qre_smoothing = bool(self.use_qre_smoothing)
        self.oracle.qre_temperature_gamma = float(self.qre_temperature_gamma)
        self.oracle.use_oracle_redundancy_penalty = bool(self.use_oracle_redundancy_penalty)
        self.oracle.compute_tremble_sensitivity = bool(self.compute_tremble_sensitivity)
        self.oracle.diversity_mode = str(self.diversity_oracle_mode)
        self.oracle.diversity_redundancy_weight = float(self.diversity_redundancy_weight)
        self.oracle.diversity_complementarity_weight = float(self.diversity_complementarity_weight)
        self.oracle.weighting_mode = str(self.oracle_weighting_mode)
        self.oracle.shapley_n_coalitions_max = int(self.shapley_n_coalitions_max)
        self.oracle.shapley_bayesian_shrinkage = bool(self.shapley_bayesian_shrinkage)
        self.oracle.shapley_bayesian_prior_strength = float(self.shapley_bayesian_prior_strength)
        self.oracle.use_interaction_oracle = bool(self.use_interaction_oracle)
        self.oracle.interaction_oracle_min_n_train = int(self.interaction_oracle_min_n_train)
        self.oracle.interaction_oracle_pool_size_cap = int(self.interaction_oracle_pool_size_cap)
        self.oracle.interaction_oracle_pair_cap = int(self.interaction_oracle_pair_cap)
        self.oracle.use_ubayfs = bool(self.use_ubayfs)
        self.oracle.ubayfs_n_bootstrap = int(self.ubayfs_n_bootstrap)
        self.oracle.ubayfs_min_n = int(self.ubayfs_min_n)
        self.oracle.ubayfs_prior_weight = float(self.ubayfs_prior_weight)
        self.oracle.use_conformal_uq = bool(self.use_conformal_uq)
        self.oracle.conformal_uq_alpha = float(self.conformal_uq_alpha)
        self.oracle.conformal_uq_min_folds = int(self.conformal_uq_min_folds)
        self.oracle.fold_preference_mode = str(self.fold_preference_mode)
        self.oracle.use_conformal_efficiency = bool(self.use_conformal_efficiency)
        self.oracle.conformal_efficiency_method = str(self.conformal_efficiency_method)
        self.oracle.oracle_weight_js_shrinkage = bool(self.oracle_weight_js_shrinkage)
        if self.enabled_methods is not None and self.portfolio_size_guard != "none":
            n_enabled = int(len(self.enabled_methods))
            if self.portfolio_size < n_enabled:
                msg = (
                    f"portfolio_size={self.portfolio_size} is smaller than enabled_methods={n_enabled}; "
                    "this can cause method displacement or 'no-effect' additions when expanding the candidate set. "
                    "Increase portfolio_size or disable this guard."
                )
                if self.portfolio_size_guard == "raise":
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning)
                # Enforce minimum breadth under guarded mode to prevent displacement.
                self.portfolio_size = int(n_enabled)
                if self.adaptive_portfolio_sizing_enabled:
                    if self.adaptive_size_max is None or int(self.adaptive_size_max) < int(self.portfolio_size):
                        self.adaptive_size_max = int(self.portfolio_size)
                    if self.adaptive_size_min is not None and int(self.adaptive_size_min) > int(self.adaptive_size_max):
                        self.adaptive_size_min = int(self.adaptive_size_max)
        self.scaler_ = None
        self.pca_ = None
        self.feature_importance_plot_ = None
        self._feature_importance_plot_payload = None
        self.selection_result_ = None
        self.mnpo_diagnostics_ = {}
        
        # Initialize scoring functions based on problem type
        if self.problem_type == 'classification':
            self.f_scorer = f_classif
            self.mi_scorer = mutual_info_classif
        elif self.problem_type == 'regression':
            self.f_scorer = f_regression
            self.mi_scorer = mutual_info_regression
        else:
            raise ValueError("problem_type must be 'classification' or 'regression'")

    # ------------------------------------------------------------------
    #  GPU availability check (T-P3-INFRA-002)
    # ------------------------------------------------------------------
    @functools.cached_property
    def _gpu_available(self) -> bool:
        """Return True if a CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ------------------------------------------------------------------
    #  Factory: construct from structured config dataclass
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config) -> 'FeatureSelector':
        """Construct a ``FeatureSelector`` from a :class:`FeatureSelectorConfig`.

        Parameters not represented in the config dataclass fall back to
        their ``__init__`` defaults.  This is an *additional* entry point —
        the existing ``__init__`` API is unchanged.

        Parameters
        ----------
        config : FeatureSelectorConfig
            A structured configuration object (see
            ``tabnetics.feature_selection.config``).

        Returns
        -------
        FeatureSelector
        """
        adaptive_enabled = bool(getattr(config.mnpo, "adaptive_portfolio_sizing_enabled", False))
        adaptive_min = getattr(config.mnpo, "adaptive_size_min", None)
        adaptive_max = getattr(config.mnpo, "adaptive_size_max", None)
        ref_size = int(config.mnpo.portfolio_size)
        if adaptive_enabled:
            if adaptive_min is None or adaptive_max is None:
                raise ValueError(
                    "adaptive portfolio sizing requires both adaptive_size_min and adaptive_size_max"
                )
            adaptive_min_i = int(adaptive_min)
            adaptive_max_i = int(adaptive_max)
            if adaptive_min_i < 1 or adaptive_max_i < adaptive_min_i:
                raise ValueError("adaptive sizing bounds must satisfy 1 <= adaptive_size_min <= adaptive_size_max")
            if not (adaptive_min_i <= ref_size <= adaptive_max_i):
                raise ValueError(
                    "portfolio_size must lie within [adaptive_size_min, adaptive_size_max] when adaptive sizing is enabled"
                )

        oracle_cfg = getattr(config.mnpo, "oracle", None)

        def _oracle_get(name: str, fallback: Any) -> Any:
            if oracle_cfg is None:
                return fallback
            return getattr(oracle_cfg, name, fallback)

        return cls(
            # ── General / core ────────────────────────────────────────
            n_folds=config.n_folds,
            n_bootstrap_iterations=config.n_bootstrap_iterations,
            random_state=config.random_state,
            problem_type=config.problem_type,
            variance_threshold=config.variance_threshold,
            correlation_threshold=config.correlation_threshold,
            use_pca=config.use_pca,
            n_components=config.n_components,
            selection_strategy=config.selection_strategy,
            enabled_methods=config.enabled_methods,
            method_timeout_seconds=config.method_timeout_seconds,
            parallel_n_jobs=getattr(config, 'parallel_n_jobs', 1),
            linear_svm_max_iter=config.linear_svm_max_iter,
            # ── MNPO / portfolio ──────────────────────────────────────
            inner_cv_splits=config.mnpo.inner_cv_splits,
            inner_cv_repeats=config.mnpo.inner_cv_repeats,
            pairwise_delta=float(_oracle_get("pairwise_delta", config.mnpo.pairwise_delta)),
            mirror_descent_steps=config.mnpo.mirror_descent_steps,
            mirror_descent_eta=config.mnpo.mirror_descent_eta,
            mirror_descent_lambda=config.mnpo.mirror_descent_lambda,
            portfolio_size=config.mnpo.portfolio_size,
            adaptive_portfolio_sizing_enabled=adaptive_enabled,
            adaptive_size_min=adaptive_min,
            adaptive_size_max=adaptive_max,
            adaptive_sizing_variance_penalty=bool(
                getattr(config.mnpo, "adaptive_sizing_variance_penalty", False)
            ),
            adaptive_sizing_variance_penalty_strength=float(
                getattr(config.mnpo, "adaptive_sizing_variance_penalty_strength", 0.5) or 0.5
            ),
            pareto_portfolio_sizing_enabled=bool(
                getattr(config.mnpo, "pareto_portfolio_sizing_enabled", False)
            ),
            stability_weighted_aggregation_enabled=bool(
                getattr(config.mnpo, "stability_weighted_aggregation_enabled", False)
            ),
            rashomon_enabled=bool(getattr(config.mnpo, "rashomon_enabled", False)),
            rashomon_max_models=int(getattr(config.mnpo, "rashomon_max_models", 12) or 12),
            rashomon_score_tolerance=float(
                getattr(config.mnpo, "rashomon_score_tolerance", 0.01) or 0.01
            ),
            portfolio_size_guard=config.mnpo.portfolio_size_guard,
            use_tritrust=bool(_oracle_get("use_tritrust", config.mnpo.use_tritrust)),
            use_stability_oracle=bool(
                _oracle_get("use_stability_oracle", config.mnpo.use_stability_oracle)
            ),
            use_complexity_oracle=bool(
                _oracle_get("use_complexity_oracle", config.mnpo.use_complexity_oracle)
            ),
            use_robust_oracle=bool(_oracle_get("use_robust_oracle", config.mnpo.use_robust_oracle)),
            use_diversity_oracle=bool(
                _oracle_get("use_diversity_oracle", config.mnpo.use_diversity_oracle)
            ),
            use_cvar=bool(_oracle_get("use_cvar", getattr(config.mnpo, "use_cvar", False))),
            cvar_alpha=float(_oracle_get("cvar_alpha", getattr(config.mnpo, "cvar_alpha", 0.33))),
            use_oracle_redundancy_penalty=bool(
                _oracle_get(
                    "use_oracle_redundancy_penalty",
                    getattr(config.mnpo, "use_oracle_redundancy_penalty", False),
                )
            ),
            use_tail_risk_oracle=config.mnpo.use_tail_risk_oracle,
            tail_risk_alpha=config.mnpo.tail_risk_alpha,
            disable_redundancy_penalty_binary=config.mnpo.disable_redundancy_penalty_binary,
            disable_class_pareto_binary=config.mnpo.disable_class_pareto_binary,
            compute_tremble_sensitivity=bool(
                _oracle_get("compute_tremble_sensitivity", getattr(config.mnpo, "compute_tremble_sensitivity", False))
            ),
            oracle_weighting_mode=str(
                _oracle_get("weighting_mode", getattr(config.mnpo, "oracle_weighting_mode", "tritrust"))
            ),
            shapley_n_coalitions_max=int(
                _oracle_get("shapley_n_coalitions_max", getattr(config.mnpo, "shapley_n_coalitions_max", 4096))
            ),
            shapley_bayesian_shrinkage=bool(
                _oracle_get(
                    "shapley_bayesian_shrinkage",
                    getattr(config.mnpo, "shapley_bayesian_shrinkage", False),
                )
            ),
            shapley_bayesian_prior_strength=float(
                _oracle_get(
                    "shapley_bayesian_prior_strength",
                    getattr(config.mnpo, "shapley_bayesian_prior_strength", 8.0),
                )
            ),
            use_interaction_oracle=bool(
                _oracle_get("use_interaction_oracle", getattr(config.mnpo, "use_interaction_oracle", False))
            ),
            interaction_oracle_min_n_train=int(
                _oracle_get(
                    "interaction_oracle_min_n_train",
                    getattr(config.mnpo, "interaction_oracle_min_n_train", 150),
                )
            ),
            interaction_oracle_pool_size_cap=int(
                _oracle_get(
                    "interaction_oracle_pool_size_cap",
                    getattr(config.mnpo, "interaction_oracle_pool_size_cap", 64),
                )
            ),
            interaction_oracle_pair_cap=int(
                _oracle_get(
                    "interaction_oracle_pair_cap",
                    getattr(config.mnpo, "interaction_oracle_pair_cap", 20000),
                )
            ),
            use_ubayfs=bool(_oracle_get("use_ubayfs", getattr(config.mnpo, "use_ubayfs", False))),
            ubayfs_n_bootstrap=int(
                _oracle_get("ubayfs_n_bootstrap", getattr(config.mnpo, "ubayfs_n_bootstrap", 32))
            ),
            ubayfs_min_n=int(_oracle_get("ubayfs_min_n", getattr(config.mnpo, "ubayfs_min_n", 100))),
            ubayfs_prior_weight=float(
                _oracle_get("ubayfs_prior_weight", getattr(config.mnpo, "ubayfs_prior_weight", 0.0))
            ),
            use_conformal_uq=bool(
                _oracle_get("use_conformal_uq", getattr(config.mnpo, "use_conformal_uq", False))
            ),
            conformal_uq_alpha=float(
                _oracle_get("conformal_uq_alpha", getattr(config.mnpo, "conformal_uq_alpha", 0.10))
            ),
            conformal_uq_min_folds=int(
                _oracle_get(
                    "conformal_uq_min_folds",
                    getattr(config.mnpo, "conformal_uq_min_folds", 5),
                )
            ),
            fold_preference_mode=str(
                _oracle_get("fold_preference_mode", getattr(config.mnpo, "fold_preference_mode", "vote"))
            ),
            use_conformal_efficiency=bool(
                _oracle_get(
                    "use_conformal_efficiency",
                    getattr(config.mnpo, "use_conformal_efficiency", False),
                )
            ),
            conformal_efficiency_method=str(
                _oracle_get(
                    "conformal_efficiency_method",
                    getattr(config.mnpo, "conformal_efficiency_method", "split"),
                )
            ),
            oracle_weight_js_shrinkage=bool(
                _oracle_get(
                    "oracle_weight_js_shrinkage",
                    getattr(config.mnpo, "oracle_weight_js_shrinkage", False),
                )
            ),
            payoff_shrinkage_kappa=float(
                getattr(config.mnpo, "payoff_shrinkage_kappa", 0.0) or 0.0
            ),
            diversity_oracle_mode=str(
                _oracle_get("diversity_mode", config.mnpo.diversity_oracle_mode)
            ),
            diversity_redundancy_weight=float(
                _oracle_get(
                    "diversity_redundancy_weight",
                    getattr(config.mnpo, "diversity_redundancy_weight", 0.60),
                )
            ),
            diversity_complementarity_weight=float(
                _oracle_get(
                    "diversity_complementarity_weight",
                    getattr(config.mnpo, "diversity_complementarity_weight", 0.35),
                )
            ),
            rank_aggregation_mode=config.mnpo.rank_aggregation_mode,
            complexity_use_runtime_penalty=config.mnpo.complexity_use_runtime_penalty,
            mnpo_include_legacy_consensus=config.mnpo.mnpo_include_legacy_consensus,
            mnpo_include_majority_consensus=config.mnpo.mnpo_include_majority_consensus,
            mnpo_paradigm_aware_prior_enabled=getattr(
                config.mnpo,
                "mnpo_paradigm_aware_prior_enabled",
                False,
            ),
            mnpo_interaction_floor=getattr(config.mnpo, "mnpo_interaction_floor", 0.12),
            # ── Stability ─────────────────────────────────────────────
            stability_subsample_fraction=config.stability.stability_subsample_fraction,
            stability_selection_threshold=config.stability.stability_selection_threshold,
            stability_threshold_method=getattr(config.stability, "stability_threshold_method", "fixed"),
            stability_target_pfer=float(
                getattr(config.stability, "stability_target_pfer", 1.0) or 1.0
            ),
            stability_use_loss_guided_validation=config.stability.stability_use_loss_guided_validation,
            stability_validation_fraction=config.stability.stability_validation_fraction,
            stability_validation_quantile=config.stability.stability_validation_quantile,
            stability_validation_min_samples=config.stability.stability_validation_min_samples,
            cluster_stability_corr_threshold=config.stability.cluster_stability_corr_threshold,
            cluster_stability_max_per_cluster=config.stability.cluster_stability_max_per_cluster,
            cluster_stability_min_cluster_freq=config.stability.cluster_stability_min_cluster_freq,
            decorrelated_stability_eps=config.stability.decorrelated_stability_eps,
            decorrelated_stability_min_max_abs_corr=config.stability.decorrelated_stability_min_max_abs_corr,
            ipss_path_grid_size=config.stability.ipss_path_grid_size,
            ipss_min_c=config.stability.ipss_min_c,
            ipss_max_c=config.stability.ipss_max_c,
            ipss_target_fdr=config.stability.ipss_target_fdr,
            ipss_use_eats_threshold=config.stability.ipss_use_eats_threshold,
            ipss_importance_model=config.stability.ipss_importance_model,
            ipss_gate_min_classes=config.stability.ipss_gate_min_classes,
            ipss_gate_min_p_over_n=config.stability.ipss_gate_min_p_over_n,
            # ── Wrapper / iterative pruning ───────────────────────────
            wrapper_refine_enabled=config.wrapper.wrapper_refine_enabled,
            wrapper_refine_top_k=config.wrapper.wrapper_refine_top_k,
            wrapper_refine_max_add=config.wrapper.wrapper_refine_max_add,
            wrapper_refine_min_gain=config.wrapper.wrapper_refine_min_gain,
            iterative_pruning_pool_factor=config.wrapper.iterative_pruning_pool_factor,
            iterative_pruning_max_rounds=config.wrapper.iterative_pruning_max_rounds,
            iterative_pruning_min_improvement=config.wrapper.iterative_pruning_min_improvement,
            iterative_pruning_max_cumulative_loss=config.wrapper.iterative_pruning_max_cumulative_loss,
            iterative_pruning_redundancy_weight=config.wrapper.iterative_pruning_redundancy_weight,
            iterative_pruning_bounded_prefilter_cap=config.wrapper.iterative_pruning_bounded_prefilter_cap,
            iterative_pruning_bounded_max_evaluations=config.wrapper.iterative_pruning_bounded_max_evaluations,
            iterative_pruning_bounded_max_runtime_seconds=config.wrapper.iterative_pruning_bounded_max_runtime_seconds,
            iterative_pruning_bounded_use_cpss_overlay=config.wrapper.iterative_pruning_bounded_use_cpss_overlay,
            iterative_pruning_bounded_enable_class_gating=config.wrapper.iterative_pruning_bounded_enable_class_gating,
            # ── Multiclass ────────────────────────────────────────────
            ova_negative_ratio=config.multiclass.ova_negative_ratio,
            ova_min_classes=config.multiclass.ova_min_classes,
            ova_class_weight_mode=config.multiclass.ova_class_weight_mode,
            ova_aggregation_mode=config.multiclass.ova_aggregation_mode,
            ecoc_min_classes=config.multiclass.ecoc_min_classes,
            nsc_min_classes=config.multiclass.nsc_min_classes,
            nsc_thresholding_mode=config.multiclass.nsc_thresholding_mode,
            nsc_shrinkage_grid_size=config.multiclass.nsc_shrinkage_grid_size,
            class_pareto_min_classes=config.multiclass.class_pareto_min_classes,
            class_pareto_top_per_class=config.multiclass.class_pareto_top_per_class,
            # ── Copula knockoff ───────────────────────────────────────
            copula_knockoff_draws=config.copula.copula_knockoff_draws,
            copula_alpha_kn=config.copula.copula_alpha_kn,
            copula_alpha_ebh=config.copula.copula_alpha_ebh,
            copula_truncation_level=config.copula.copula_truncation_level,
            copula_generator=getattr(config.copula, "copula_generator", "copula"),
            copula_deepdrk_latent_fraction=getattr(config.copula, "copula_deepdrk_latent_fraction", 0.35),
            copula_deepdrk_noise_scale=getattr(config.copula, "copula_deepdrk_noise_scale", 1.0),
            copula_derandomize_runs=getattr(config.copula, "copula_derandomize_runs", 1),
            copula_stabilizer_runs=config.copula.copula_stabilizer_runs,
            # ── Method-specific ───────────────────────────────────────
            mrmr_max_features=config.methods.mrmr_max_features,
            mrmr_redundancy_weight=config.methods.mrmr_redundancy_weight,
            mrmr_mi_redundancy_enabled=getattr(config.methods, "mrmr_mi_redundancy_enabled", False),
            mrmr_mi_n_bins=getattr(config.methods, "mrmr_mi_n_bins", 8),
            cmim_min_samples=getattr(config.methods, "cmim_min_samples", 60),
            cmim_n_bins=getattr(config.methods, "cmim_n_bins", 8),
            fcbf_n_bins=getattr(config.methods, "fcbf_n_bins", 8),
            ktsp_max_features=config.methods.ktsp_max_features,
            ktsp_k_pairs=config.methods.ktsp_k_pairs,
            hsic_lasso_alpha=config.methods.hsic_lasso_alpha,
            hsic_lasso_prefilter_max_features=config.methods.hsic_lasso_prefilter_max_features,
            hsic_lasso_binary_delta_enabled=getattr(
                config.methods,
                "hsic_lasso_binary_delta_enabled",
                True,
            ),
            hsic_lasso_binary_delta_min_samples=getattr(
                config.methods,
                "hsic_lasso_binary_delta_min_samples",
                30,
            ),
            slce_prefilter_max_features=getattr(
                config.methods,
                "slce_prefilter_max_features",
                1024,
            ),
            slce_min_samples=getattr(
                config.methods,
                "slce_min_samples",
                30,
            ),
            slce_ridge=getattr(
                config.methods,
                "slce_ridge",
                1.0,
            ),
            treeshap_min_samples=getattr(config.methods, "treeshap_min_samples", 50),
            treeshap_n_estimators=getattr(config.methods, "treeshap_n_estimators", 200),
            treeshap_multi_seed_runs=getattr(config.methods, "treeshap_multi_seed_runs", 3),
            oaenet_min_samples=getattr(config.methods, "oaenet_min_samples", 40),
            oaenet_prescreen_max_features=getattr(
                config.methods, "oaenet_prescreen_max_features", 512
            ),
            oaenet_l1_ratio=getattr(config.methods, "oaenet_l1_ratio", 0.5),
            oaenet_c_grid_size=getattr(config.methods, "oaenet_c_grid_size", 6),
            sdr_min_classes=getattr(config.methods, "sdr_min_classes", 3),
            sdr_prefilter_max_features=getattr(
                config.methods, "sdr_prefilter_max_features", 512
            ),
            sdr_n_components=getattr(config.methods, "sdr_n_components", 3),
            sdr_covariance_ridge=getattr(
                config.methods, "sdr_covariance_ridge", 1e-3
            ),
            # ── Prefilter blend ───────────────────────────────────────────
            prefilter_mi_weight=config.prefilter.mi_weight,
            prefilter_f_weight=config.prefilter.f_weight,
            prefilter_union_enabled=getattr(config.prefilter, "union_enabled", False),
            prefilter_strategies=getattr(config.prefilter, "strategies", ("mi_ftest_blend",)),
            prefilter_nondefault_budget_fraction=getattr(
                config.prefilter,
                "nondefault_budget_fraction",
                0.10,
            ),
            prefilter_wsnr_enabled=bool(getattr(config.prefilter, "wsnr_enabled", False)),
            prefilter_wsnr_stabilize_counts=bool(
                getattr(config.prefilter, "wsnr_stabilize_counts", True)
            ),
            prefilter_data_domain=str(getattr(config.prefilter, "data_domain", "auto") or "auto"),
            prefilter_rnaseq_transform_enabled=bool(
                getattr(config.prefilter, "rnaseq_transform_enabled", True)
            ),
            prefilter_rnaseq_transform_force=bool(
                getattr(config.prefilter, "rnaseq_transform_force", False)
            ),
            prefilter_rnaseq_nb_lrt_enabled=bool(
                getattr(config.prefilter, "rnaseq_nb_lrt_enabled", False)
            ),
            prefilter_rnaseq_nb_lrt_alpha=float(
                getattr(config.prefilter, "rnaseq_nb_lrt_alpha", 0.10) or 0.10
            ),
            # ── Tier 2 screening (T-004) ──────────────────────────────
            screening_enabled=config.screening.enabled,
            screening_method=config.screening.method,
            screening_pool_cap=config.screening.pool_cap,
            screening_stir_n_neighbors=config.screening.stir_n_neighbors,
            screening_stir_n_iter=config.screening.stir_n_iter,
            screening_stir_keep_fraction=config.screening.stir_keep_fraction,
            screening_stir_min_features=config.screening.stir_min_features,
            screening_evalue_alpha=getattr(config.screening, "evalue_alpha", 0.20),
            screening_evalue_min_features=getattr(config.screening, "evalue_min_features", 20),
            # ── Multi-classifier evaluation proxy ─────────────────────
            eval_models_enabled=config.evaluation.eval_models_enabled,
            eval_models=config.evaluation.eval_models,
            eval_aggregate=config.evaluation.eval_aggregate,
            eval_cvar_alpha=config.evaluation.eval_cvar_alpha,
            eval_failure_strict_mode=config.evaluation.eval_failure_strict_mode,
            eval_model_weight_strategy=config.evaluation.eval_model_weight_strategy,
            # ── Multi-model MNPO oracles (T-002) ──────────────────────
            performance_oracle_mode=str(
                _oracle_get("performance_oracle_mode", config.mnpo.performance_oracle_mode)
            ),
            # ── Importance uncertainty (reporting only) ───────────────
            importance_uq_enabled=bool(
                getattr(getattr(config, "importance_uq", None), "enabled", False)
            ),
            importance_uq_min_cv_folds=int(
                getattr(getattr(config, "importance_uq", None), "min_cv_folds", 3) or 3
            ),
        )

    # ------------------------------------------------------------------
    #  Copula knock-off selection  (TDCKe)
    # ------------------------------------------------------------------
    def _copula_knockoff_selection(self, X, y, n_target_features):
        """Wrapper so the copula selector plugs into the voting scheme."""
        from .methods.knockoff import copula_knockoff_selection
        return copula_knockoff_selection(
            X, y, n_target_features,
            CopulaKnockoffSelectorClass=CopulaKnockoffSelector,
            copula_knockoff_draws=self.copula_knockoff_draws,
            copula_alpha_kn=self.copula_alpha_kn,
            copula_alpha_ebh=self.copula_alpha_ebh,
            copula_truncation_level=self.copula_truncation_level,
            copula_generator=self.copula_generator,
            copula_deepdrk_latent_fraction=self.copula_deepdrk_latent_fraction,
            copula_deepdrk_noise_scale=self.copula_deepdrk_noise_scale,
            copula_derandomize_runs=self.copula_derandomize_runs,
            copula_stabilizer_runs=self.copula_stabilizer_runs,
            copula_stabilizer_use_ebh=self.copula_stabilizer_use_ebh,
            copula_stabilizer_seed_stride=self.copula_stabilizer_seed_stride,
            random_state=self.random_state,
        )

    def _get_cv_splitter(self, y):
        """
        Get appropriate cross-validation splitter based on problem type.

        DEPRECATED: This always returned LOOCV. Use _get_cv_strategy() instead
        for consistent, adaptive CV strategy (P1-3: CV unification).

        Kept for backward compatibility; delegates to _get_cv_strategy().
        """
        n_samples = len(y)
        return self._get_cv_strategy(n_samples, y, purpose='method_internal')

    def _get_cv_strategy(self, n_samples, y, purpose='method_internal'):
        """Unified CV strategy selection. Delegates to cv.get_cv_strategy."""
        from .cv import get_cv_strategy
        return get_cv_strategy(n_samples, y, self.problem_type, self.random_state, self.inner_cv_splits, self.inner_cv_repeats, purpose)

    def _calculate_target_features(self, n_features_total, n_final_features=None):
        """Calculate target number of features for initial selection."""
        if n_final_features is not None:
            # Select twice the target, but no more than 50% of features
            n_target = min(2 * n_final_features, max(n_final_features, n_features_total // 2))
        else:
            # Default: select 30% of features or 20, whichever is larger
            n_target = max(int(0.3 * n_features_total), min(20, n_features_total))
        return n_target

    def _remove_correlated_features(self, X_df, threshold=0.90):
        """Remove highly correlated features."""
        from .preprocessing import remove_correlated_features
        return remove_correlated_features(X_df, threshold)

    def _prefilter_feature_pool(self, X, y, max_features):
        """
        Build a compact candidate pool before expensive combinatorial selectors.
        Combines MI and F-test relevance where available.

        Uses _score_cache if available (P1-1: eliminates redundant computations).
        """
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            return np.array([], dtype=int)
        n_samples, n_features = X_arr.shape
        if n_samples < 2 or n_features <= 0:
            return np.array([], dtype=int)

        if n_features <= max_features:
            return np.arange(n_features, dtype=int)

        X_score_source = np.asarray(X_arr, dtype=float)
        rnaseq_meta: Dict[str, Any] = {
            "rnaseq_transform_applied": False,
            "rnaseq_transform_reason": "disabled",
            "is_rnaseq": False,
        }
        try:
            from .prefilter import _rnaseq_transform

            X_score_source, rnaseq_meta = _rnaseq_transform(
                X_arr,
                data_domain=str(getattr(self, "prefilter_data_domain", "auto") or "auto"),
                enabled=bool(getattr(self, "prefilter_rnaseq_transform_enabled", True)),
                force=bool(getattr(self, "prefilter_rnaseq_transform_force", False)),
            )
        except Exception as exc:
            logger.exception("RNA-seq transform stage failed; using raw features for prefilter scoring.")
            X_score_source = np.asarray(X_arr, dtype=float)
            rnaseq_meta = {
                "rnaseq_transform_applied": False,
                "rnaseq_transform_reason": "transform_exception",
                "is_rnaseq": False,
            }

        # Use cache if available, otherwise compute on-demand
        use_cache = bool(
            hasattr(self, '_score_cache')
            and self._score_cache is not None
            and X_score_source.shape == X_arr.shape
            and np.allclose(X_score_source, X_arr, equal_nan=True)
        )
        if use_cache:
            mi_scores = self._score_cache.mi_scores
            mi_norm = self._score_cache._safe_normalize(mi_scores)
            f_scores, _ = self._score_cache.f_scores
            f_norm = self._score_cache._safe_normalize(f_scores)
        else:
            # Legacy path (if not using fit_transform or cache unavailable)
            try:
                mi_scores = np.asarray(
                    self.mi_scorer(X_score_source, y, random_state=self.random_state),
                    dtype=float,
                ).ravel()
            except Exception as exc:
                mi_scores = np.zeros(n_features, dtype=float)
            mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
            mi_norm = self._normalize_vector_01(mi_scores)

            try:
                f_scores, _ = self.f_scorer(X_score_source, y)
                f_scores = np.asarray(f_scores, dtype=float).ravel()
            except Exception as exc:
                f_scores = np.zeros(n_features, dtype=float)
            f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
            f_norm = self._normalize_vector_01(f_scores)

        # Default blend: 60% MI, 40% F-test (see audit §6.4 for rationale)
        combined = (
            float(self.prefilter_mi_weight) * np.asarray(mi_norm, dtype=float)
            + float(self.prefilter_f_weight) * np.asarray(f_norm, dtype=float)
        )

        if bool(getattr(self, "prefilter_rnaseq_nb_lrt_enabled", False)):
            try:
                from .prefilter import rnaseq_nb_lrt_scores

                nb_scores, nb_meta = rnaseq_nb_lrt_scores(
                    X_arr,
                    np.asarray(y),
                    data_domain=str(getattr(self, "prefilter_data_domain", "auto") or "auto"),
                    alpha=float(getattr(self, "prefilter_rnaseq_nb_lrt_alpha", 0.10) or 0.10),
                )
                if bool(nb_meta.get("rnaseq_nb_lrt_applied", False)):
                    nb_norm = self._normalize_vector_01(nb_scores)
                    combined = 0.70 * combined + 0.30 * np.asarray(nb_norm, dtype=float)
            except Exception as exc:
                logger.exception("RNA-seq NB-LRT prefilter stage failed; continuing with MI/F-test blend.")

        combined = np.asarray(self._normalize_vector_01(combined), dtype=float).ravel()
        default_selected = np.argsort(combined)[::-1][:max_features]
        default_selected = np.array(sorted(set(int(i) for i in default_selected)), dtype=int)

        if not bool(getattr(self, "prefilter_union_enabled", False)):
            return default_selected

        try:
            from .prefilter import build_prefilter_union_pool

            union_selected = build_prefilter_union_pool(
                X_score_source,
                np.asarray(y),
                max_features=int(max_features),
                strategies=tuple(getattr(self, "prefilter_strategies", ("mi_ftest_blend",))),
                nondefault_budget_fraction=float(
                    getattr(self, "prefilter_nondefault_budget_fraction", 0.10)
                ),
                base_scores=combined,
                mi_scores=np.asarray(mi_norm, dtype=float).ravel(),
                f_scores=np.asarray(f_norm, dtype=float).ravel(),
                normalize_fn=self._normalize_vector_01,
                random_state=int(self.random_state),
                problem_type=str(self.problem_type),
                wsnr_stabilize_counts=bool(
                    getattr(self, "prefilter_wsnr_stabilize_counts", True)
                ),
                wsnr_data_domain=str(getattr(self, "prefilter_data_domain", "auto") or "auto"),
            )
            if union_selected.size > 0:
                return union_selected
        except Exception as exc:
            logger.exception("Prefilter union failed; falling back to default MI/F-test blend.")
        return default_selected

    def _stability_selection_lasso(self, X, y, n_target_features):
        """Perform stability selection using Lasso with bootstrap sampling."""
        from .methods.embedded import stability_selection_lasso
        return stability_selection_lasso(X, y, n_target_features, self.n_bootstrap_iterations, self.problem_type, self.random_state, self._get_cv_splitter)

    def _rfe_cv_selection(self, X, y, n_target_features):
        """Perform RFECV with multiple runs."""
        from .methods.embedded import rfe_cv_selection
        return rfe_cv_selection(X, y, n_target_features, self.n_bootstrap_iterations, self.problem_type, self.random_state, self._get_cv_splitter)

    def _boruta_selection(self, X, y, n_target_features):
        """Perform Boruta selection with multiple runs."""
        from .methods.embedded import boruta_selection
        return boruta_selection(X, y, n_target_features, self.n_bootstrap_iterations, self.problem_type, self.random_state)

    def _gradient_boosting_selection(self, X, y, n_target_features):
        """Perform Gradient Boosting feature selection."""
        from .methods.embedded import gradient_boosting_selection
        return gradient_boosting_selection(X, y, n_target_features, self.n_bootstrap_iterations, self.problem_type, self.random_state)

    def _linear_svm_selection(self, X, y, n_target_features):
        """Perform Linear SVM feature selection."""
        from .methods.embedded import linear_svm_selection
        return linear_svm_selection(X, y, n_target_features, self.n_bootstrap_iterations, self.problem_type, self.random_state, self.linear_svm_max_iter)

    def _mutual_information_selection(self, X, y, n_target_features):
        """Perform Mutual Information based selection."""
        from .methods.filter import mutual_information_selection
        return mutual_information_selection(X, y, n_target_features, self.mi_scorer, self.random_state)

    def _anova_f_selection(self, X, y, n_target_features):
        """Perform ANOVA F-test based selection."""
        from .methods.filter import anova_f_selection
        return anova_f_selection(X, y, n_target_features, self.f_scorer)

    def _chi_square_selection(self, X, y, n_target_features):
        """Perform Chi-Square univariate filter selection."""
        from .methods.filter import chi_square_selection
        return chi_square_selection(X, y, n_target_features)

    def _relieff_selection(self, X, y, n_target_features):
        """Perform ReliefF instance-based filter selection."""
        from .methods.filter import relieff_selection
        return relieff_selection(X, y, n_target_features)

    def _wmw_auc_selection(self, X, y, n_target_features):
        """
        Univariate AUC ranking via the Wilcoxon-Mann-Whitney statistic (binary-only).
        """
        from .methods.filter import wmw_auc_selection
        return wmw_auc_selection(X, y, n_target_features, self.problem_type)

    def _joint_auc_l1_selection(self, X, y, n_target_features):
        """
        Joint AUC-aware L1 feature selector (binary-only).
        """
        from .methods.embedded import joint_auc_l1_selection
        return joint_auc_l1_selection(X, y, n_target_features, self.problem_type, self.random_state, self._prefilter_feature_pool)

    def _treeshap_selection(self, X, y, n_target_features):
        from .methods.embedded import treeshap_selection

        return treeshap_selection(
            X,
            y,
            n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            n_bootstrap_iterations=self.n_bootstrap_iterations,
            min_samples=self.treeshap_min_samples,
            n_estimators=self.treeshap_n_estimators,
            multi_seed_runs=self.treeshap_multi_seed_runs,
        )

    def _oaenet_selection(self, X, y, n_target_features):
        from .methods.embedded import oaenet_adaptive_selection

        return oaenet_adaptive_selection(
            X,
            y,
            n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            min_samples=self.oaenet_min_samples,
            prescreen_max_features=self.oaenet_prescreen_max_features,
            l1_ratio=self.oaenet_l1_ratio,
            c_grid_size=self.oaenet_c_grid_size,
        )

    def _ova_ensemble_selection(self, X, y, n_target_features):
        from .methods.multiclass import ova_ensemble_selection
        return ova_ensemble_selection(
            X, y, n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            ova_min_classes=self.ova_min_classes,
            ova_min_pos_samples=self.ova_min_pos_samples,
            ova_negative_ratio=self.ova_negative_ratio,
            ova_linear_backend=self.ova_linear_backend,
            linear_svm_max_iter=self.linear_svm_max_iter,
            ova_enable_calibration=self.ova_enable_calibration,
            ova_calibration_cv=self.ova_calibration_cv,
            ova_class_weight_mode=self.ova_class_weight_mode,
            ova_aggregation_mode=self.ova_aggregation_mode,
            ova_aggregation_p=self.ova_aggregation_p,
            normalize_fn=self._normalize_vector_01,
        )

    @staticmethod
    def _label_to_key(label):
        from .methods.multiclass import label_to_key
        return label_to_key(label)

    def _estimate_ova_calibration_reliability(self, X_sub, y_sub):
        from .methods.multiclass import estimate_ova_calibration_reliability
        return estimate_ova_calibration_reliability(
            X_sub, y_sub,
            ova_calibration_cv=self.ova_calibration_cv,
            ova_linear_backend=self.ova_linear_backend,
            random_state=self.random_state,
            linear_svm_max_iter=self.linear_svm_max_iter,
        )

    def _apply_per_class_quota_overlay(
        self,
        selected_indices,
        ranked_indices,
        class_rankings,
        n_target_features,
    ):
        from .methods.multiclass import apply_per_class_quota_overlay
        return apply_per_class_quota_overlay(
            selected_indices, ranked_indices, class_rankings, n_target_features,
            per_class_quota_enabled=self.per_class_quota_enabled,
            per_class_quota_min_per_class=self.per_class_quota_min_per_class,
            per_class_quota_max_fraction=self.per_class_quota_max_fraction,
        )

    def _ecoc_binary_relevance_scores(self, X_sub, y_sub, n_features):
        from .methods.multiclass import ecoc_binary_relevance_scores
        return ecoc_binary_relevance_scores(
            X_sub, y_sub, n_features,
            random_state=self.random_state,
            ova_linear_backend=self.ova_linear_backend,
            linear_svm_max_iter=self.linear_svm_max_iter,
            normalize_fn=self._normalize_vector_01,
        )

    def _ecoc_class_complexity_weights(self, X, y, classes):
        from .methods.multiclass import ecoc_class_complexity_weights
        return ecoc_class_complexity_weights(
            X, y, classes,
            ecoc_class_complexity_weight=self.ecoc_class_complexity_weight,
            normalize_fn=self._normalize_vector_01,
        )

    def _ecoc_class_aware_selection(self, X, y, n_target_features):
        from .methods.multiclass import ecoc_class_aware_selection
        return ecoc_class_aware_selection(
            X, y, n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            ecoc_min_classes=self.ecoc_min_classes,
            ecoc_class_complexity_weight=self.ecoc_class_complexity_weight,
            ecoc_include_ova_tasks=self.ecoc_include_ova_tasks,
            ecoc_negative_ratio=self.ecoc_negative_ratio,
            ecoc_max_ovo_pairs=self.ecoc_max_ovo_pairs,
            ecoc_random_code_bits=self.ecoc_random_code_bits,
            ova_linear_backend=self.ova_linear_backend,
            linear_svm_max_iter=self.linear_svm_max_iter,
            normalize_fn=self._normalize_vector_01,
        )

    def _nearest_shrunken_centroid_selection(self, X, y, n_target_features):
        from .methods.multiclass import nearest_shrunken_centroid_selection
        return nearest_shrunken_centroid_selection(
            X, y, n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            nsc_min_classes=self.nsc_min_classes,
            nsc_shrinkage_grid_size=self.nsc_shrinkage_grid_size,
            nsc_thresholding_mode=getattr(self, "nsc_thresholding_mode", "soft"),
            nsc_order_quantile=self.nsc_order_quantile,
            nsc_deep_shrinkage_search=self.nsc_deep_shrinkage_search,
            per_class_quota_enabled=self.per_class_quota_enabled,
            per_class_quota_min_per_class=self.per_class_quota_min_per_class,
            per_class_quota_max_fraction=self.per_class_quota_max_fraction,
            normalize_fn=self._normalize_vector_01,
        )

    def _class_specific_pareto_front_selection(self, X, y, n_target_features, class_pareto_min_classes: Optional[int] = None):
        from .methods.multiclass import class_specific_pareto_front_selection
        if class_pareto_min_classes is None:
            class_pareto_min_classes = self.class_pareto_min_classes
        return class_specific_pareto_front_selection(
            X, y, n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            class_pareto_min_classes=int(max(2, int(class_pareto_min_classes))),
            class_pareto_top_per_class=self.class_pareto_top_per_class,
            class_pareto_global_fraction=self.class_pareto_global_fraction,
            class_pareto_minority_boost=self.class_pareto_minority_boost,
            class_pareto_kw_weight=self.class_pareto_kw_weight,
            prefilter_fn=self._prefilter_feature_pool,
            binary_class_prefilter_fn=self._binary_class_prefilter_scores,
            normalize_fn=self._normalize_vector_01,
            mi_scorer_fn=self.mi_scorer,
            f_scorer_fn=self.f_scorer,
            per_class_quota_enabled=self.per_class_quota_enabled,
            per_class_quota_min_per_class=self.per_class_quota_min_per_class,
            per_class_quota_max_fraction=self.per_class_quota_max_fraction,
        )

    def _sir_sdr_selection(self, X, y, n_target_features):
        from .methods.sdr import sir_sdr_selection

        return sir_sdr_selection(
            X,
            y,
            n_target_features,
            problem_type=self.problem_type,
            sdr_min_classes=self.sdr_min_classes,
            sdr_prefilter_max_features=self.sdr_prefilter_max_features,
            sdr_n_components=self.sdr_n_components,
            sdr_covariance_ridge=self.sdr_covariance_ridge,
            prefilter_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
            random_state=self.random_state,
        )

    def _save_sdr_selection(self, X, y, n_target_features):
        from .methods.sdr import save_sdr_selection

        return save_sdr_selection(
            X,
            y,
            n_target_features,
            problem_type=self.problem_type,
            sdr_min_classes=self.sdr_min_classes,
            sdr_prefilter_max_features=self.sdr_prefilter_max_features,
            sdr_n_components=self.sdr_n_components,
            sdr_covariance_ridge=self.sdr_covariance_ridge,
            prefilter_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
            random_state=self.random_state,
        )

    def _pfc_sdr_selection(self, X, y, n_target_features):
        from .methods.sdr import pfc_sdr_selection

        return pfc_sdr_selection(
            X,
            y,
            n_target_features,
            problem_type=self.problem_type,
            sdr_min_classes=self.sdr_min_classes,
            sdr_prefilter_max_features=self.sdr_prefilter_max_features,
            sdr_n_components=self.sdr_n_components,
            sdr_covariance_ridge=self.sdr_covariance_ridge,
            prefilter_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
            random_state=self.random_state,
        )

    def _hsic_lasso_selection(self, X, y, n_target_features):
        from .methods.hsic import hsic_lasso_selection
        return hsic_lasso_selection(
            X, y, n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            hsic_lasso_prefilter_max_features=self.hsic_lasso_prefilter_max_features,
            hsic_lasso_target_sigma=self.hsic_lasso_target_sigma,
            hsic_lasso_feature_sigma=self.hsic_lasso_feature_sigma,
            hsic_lasso_alpha=self.hsic_lasso_alpha,
            hsic_lasso_max_iter=self.hsic_lasso_max_iter,
            hsic_lasso_relevance_blend=self.hsic_lasso_relevance_blend,
            hsic_lasso_binary_delta_enabled=self.hsic_lasso_binary_delta_enabled,
            hsic_lasso_binary_delta_min_samples=self.hsic_lasso_binary_delta_min_samples,
            prefilter_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
            rbf_kernel_1d_fn=self._rbf_kernel_1d,
            center_kernel_matrix_fn=self._center_kernel_matrix,
        )

    def _slce_centroid_encoder_selection(self, X, y, n_target_features):
        from .methods.slce import slce_centroid_encoder_selection
        return slce_centroid_encoder_selection(
            X,
            y,
            n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            slce_prefilter_max_features=self.slce_prefilter_max_features,
            slce_min_samples=self.slce_min_samples,
            slce_ridge=self.slce_ridge,
            prefilter_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
        )

    def _joint_multiclass_support_selection(self, X, y, n_target_features):
        from .methods.multiclass import joint_multiclass_support_selection
        return joint_multiclass_support_selection(
            X, y, n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            joint_multiclass_min_classes=self.joint_multiclass_min_classes,
            joint_multiclass_max_features=self.joint_multiclass_max_features,
            joint_multiclass_min_c=self.joint_multiclass_min_c,
            joint_multiclass_max_c=self.joint_multiclass_max_c,
            joint_multiclass_path_grid_size=self.joint_multiclass_path_grid_size,
            joint_multiclass_l1_ratio=self.joint_multiclass_l1_ratio,
            joint_multiclass_univariate_blend=self.joint_multiclass_univariate_blend,
            prefilter_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
            mi_scorer_fn=self.mi_scorer,
            f_scorer_fn=self.f_scorer,
        )

    def _dove_class_specific_selection(self, X, y, n_target_features):
        from .methods.multiclass import dove_class_specific_selection
        return dove_class_specific_selection(
            X, y, n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            dove_min_classes=self.dove_min_classes,
            dove_max_pairs_per_class=self.dove_max_pairs_per_class,
            dove_minority_boost=self.dove_minority_boost,
            dove_specificity_weight=self.dove_specificity_weight,
            dove_path_grid_size=self.dove_path_grid_size,
            ecoc_class_complexity_weight=self.ecoc_class_complexity_weight,
            ova_linear_backend=self.ova_linear_backend,
            linear_svm_max_iter=self.linear_svm_max_iter,
            normalize_fn=self._normalize_vector_01,
        )

    def _sparse_multinomial_screen_candidates(self, X_pool, y_arr, n_target_features):
        from .methods.multiclass import sparse_multinomial_screen_candidates
        return sparse_multinomial_screen_candidates(
            X_pool, y_arr, n_target_features,
            sparse_multinomial_screening_mode=self.sparse_multinomial_screening_mode,
            sparse_multinomial_screening_keep_fraction=self.sparse_multinomial_screening_keep_fraction,
            sparse_multinomial_screening_min_features=self.sparse_multinomial_screening_min_features,
            random_state=self.random_state,
            normalize_fn=self._normalize_vector_01,
            mi_scorer_fn=self.mi_scorer,
            f_scorer_fn=self.f_scorer,
        )

    def _sparse_multinomial_selection(self, X, y, n_target_features):
        from .methods.multiclass import sparse_multinomial_selection
        return sparse_multinomial_selection(
            X, y, n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            sparse_multinomial_min_classes=self.sparse_multinomial_min_classes,
            sparse_multinomial_max_features=self.sparse_multinomial_max_features,
            sparse_multinomial_min_c=self.sparse_multinomial_min_c,
            sparse_multinomial_max_c=self.sparse_multinomial_max_c,
            sparse_multinomial_path_grid_size=self.sparse_multinomial_path_grid_size,
            sparse_multinomial_l1_ratio=self.sparse_multinomial_l1_ratio,
            sparse_multinomial_univariate_blend=self.sparse_multinomial_univariate_blend,
            sparse_multinomial_backend=self.sparse_multinomial_backend,
            sparse_multinomial_max_iter=self.sparse_multinomial_max_iter,
            sparse_multinomial_screening_mode=self.sparse_multinomial_screening_mode,
            sparse_multinomial_screening_keep_fraction=self.sparse_multinomial_screening_keep_fraction,
            sparse_multinomial_screening_min_features=self.sparse_multinomial_screening_min_features,
            sparse_multinomial_screening_fallback_on_failure=self.sparse_multinomial_screening_fallback_on_failure,
            prefilter_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
            mi_scorer_fn=self.mi_scorer,
            f_scorer_fn=self.f_scorer,
        )

    def _mrmr_jmi_selection(self, X, y, n_target_features):
        """
        Redundancy-aware forward selection.
        Uses MI relevance with average absolute-correlation redundancy penalty.
        """
        from .methods.filter import mrmr_jmi_selection
        return mrmr_jmi_selection(
            X, y, n_target_features,
            random_state=self.random_state,
            mi_scorer=self.mi_scorer,
            prefilter_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
            mrmr_max_features=self.mrmr_max_features,
            mrmr_redundancy_weight=self.mrmr_redundancy_weight,
            mrmr_mi_redundancy_enabled=self.mrmr_mi_redundancy_enabled,
            mrmr_mi_n_bins=self.mrmr_mi_n_bins,
        )

    def _fcbf_selection(self, X, y, n_target_features):
        """Fast correlation-based filter (FCBF, symmetric uncertainty)."""
        from .methods.filter import fcbf_selection

        return fcbf_selection(
            X,
            y,
            n_target_features,
            prefilter_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
            max_features=self.mrmr_max_features,
            n_bins=self.fcbf_n_bins,
        )

    def _cmim_selection(self, X, y, n_target_features):
        """Conditional MI Max-Dependency forward selection."""
        from .methods.filter import cmim_selection

        return cmim_selection(
            X,
            y,
            n_target_features,
            random_state=self.random_state,
            mi_scorer=self.mi_scorer,
            prefilter_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
            max_features=self.mrmr_max_features,
            min_samples=self.cmim_min_samples,
            n_bins=self.cmim_n_bins,
        )

    def _binary_class_prefilter_scores(self, X, y_bin, include_kw=False, kw_weight=0.25):
        """Binary relevance scores used by class-aware prefilter components."""
        from .prefilter import binary_class_prefilter_scores
        return binary_class_prefilter_scores(
            X, y_bin,
            random_state=self.random_state,
            normalize_fn=self._normalize_vector_01,
            include_kw=include_kw,
            kw_weight=kw_weight,
        )

    @staticmethod
    def _center_kernel_matrix(K):
        """Center a Gram matrix with H K H."""
        from .prefilter import center_kernel_matrix
        return center_kernel_matrix(K)

    @staticmethod
    def _rbf_kernel_1d(values, sigma):
        """Build a 1D RBF kernel from a vector and a sigma value."""
        from .prefilter import rbf_kernel_1d
        return rbf_kernel_1d(values, sigma)

    def _pareto_prefilter_stability_support(self, X_candidate, y_arr, target):
        """Estimate candidate support frequencies under stratified subsampling."""
        from .prefilter import pareto_prefilter_stability_support
        return pareto_prefilter_stability_support(
            X_candidate, y_arr, target,
            stability_subsamples=self.iterative_pruning_class_pareto_stability_subsamples,
            stability_fraction=self.iterative_pruning_class_pareto_stability_fraction,
            random_state=self.random_state,
            mi_scorer=self.mi_scorer,
            f_scorer=self.f_scorer,
            normalize_fn=self._normalize_vector_01,
            mi_weight=self.prefilter_mi_weight,
            f_weight=self.prefilter_f_weight,
        )

    def _class_dominance_pareto_prefilter(self, X, y, max_features):
        """
        A17: class-dominance-aware Pareto prefilter for iterative wrapper selectors.
        """
        from .prefilter import class_dominance_pareto_prefilter
        return class_dominance_pareto_prefilter(
            X, y, max_features,
            prefilter_pool_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
            mi_scorer=self.mi_scorer,
            f_scorer=self.f_scorer,
            random_state=self.random_state,
            problem_type=self.problem_type,
            iterative_pruning_class_pareto_prefilter_enabled=self.iterative_pruning_class_pareto_prefilter_enabled,
            iterative_pruning_class_pareto_min_classes=self.iterative_pruning_class_pareto_min_classes,
            iterative_pruning_class_pareto_top_per_class=self.iterative_pruning_class_pareto_top_per_class,
            iterative_pruning_class_pareto_global_fraction=self.iterative_pruning_class_pareto_global_fraction,
            iterative_pruning_class_pareto_minority_boost=self.iterative_pruning_class_pareto_minority_boost,
            iterative_pruning_class_pareto_stability_gate_enabled=self.iterative_pruning_class_pareto_stability_gate_enabled,
            iterative_pruning_class_pareto_stability_subsamples=self.iterative_pruning_class_pareto_stability_subsamples,
            iterative_pruning_class_pareto_stability_fraction=self.iterative_pruning_class_pareto_stability_fraction,
            iterative_pruning_class_pareto_stability_threshold=self.iterative_pruning_class_pareto_stability_threshold,
            iterative_pruning_class_pareto_stability_min_overlap=self.iterative_pruning_class_pareto_stability_min_overlap,
            iterative_pruning_class_pareto_stability_min_stable_features=self.iterative_pruning_class_pareto_stability_min_stable_features,
            iterative_pruning_class_pareto_stability_fallback_on_failure=self.iterative_pruning_class_pareto_stability_fallback_on_failure,
            mi_weight=self.prefilter_mi_weight,
            f_weight=self.prefilter_f_weight,
        )

    def _iterative_pruning_cpss_overlay(self, X_pool, y_arr, target, base_selected, base_relevance, base_score):
        """A16: CPSS-style complementary-pairs stability overlay for bounded pruning."""
        from .methods.wrapper import iterative_pruning_cpss_overlay
        return iterative_pruning_cpss_overlay(
            X_pool, y_arr, target, base_selected, base_relevance, base_score,
            use_cpss_overlay=self.iterative_pruning_bounded_use_cpss_overlay,
            cpss_pairs=self.iterative_pruning_bounded_cpss_pairs,
            cpss_stability_threshold=self.iterative_pruning_bounded_cpss_stability_threshold,
            cpss_min_stable_features=self.iterative_pruning_bounded_cpss_min_stable_features,
            cpss_min_jaccard=self.iterative_pruning_bounded_cpss_min_jaccard,
            cpss_max_score_drop=self.iterative_pruning_bounded_cpss_max_score_drop,
            random_state=self.random_state,
            mi_scorer=self.mi_scorer,
            f_scorer=self.f_scorer,
            normalize_fn=self._normalize_vector_01,
            wrapper_score_fn=self._wrapper_refine_subset_score,
        )

    def _iterative_redundancy_pruning_core(self, X, y, n_target_features, runtime_bounded=False):
        """ieGENES-style iterative wrapper pruning core."""
        from .methods.wrapper import iterative_redundancy_pruning_core
        return iterative_redundancy_pruning_core(
            X, y, n_target_features, runtime_bounded=runtime_bounded,
            prefilter_fn=self._prefilter_feature_pool,
            pareto_prefilter_fn=self._class_dominance_pareto_prefilter,
            normalize_fn=self._normalize_vector_01,
            wrapper_score_fn=self._wrapper_refine_subset_score,
            cpss_overlay_fn=self._iterative_pruning_cpss_overlay,
            mi_scorer=self.mi_scorer,
            f_scorer=self.f_scorer,
            score_cache=getattr(self, '_score_cache', None),
            problem_type=self.problem_type,
            random_state=self.random_state,
            mrmr_max_features=self.mrmr_max_features,
            iterative_pruning_pool_factor=self.iterative_pruning_pool_factor,
            iterative_pruning_redundancy_weight=self.iterative_pruning_redundancy_weight,
            iterative_pruning_max_rounds=self.iterative_pruning_max_rounds,
            iterative_pruning_min_improvement=self.iterative_pruning_min_improvement,
            iterative_pruning_max_cumulative_loss=self.iterative_pruning_max_cumulative_loss,
            iterative_pruning_class_pareto_prefilter_enabled=self.iterative_pruning_class_pareto_prefilter_enabled,
            iterative_pruning_class_pareto_top_per_class=self.iterative_pruning_class_pareto_top_per_class,
            iterative_pruning_class_pareto_global_fraction=self.iterative_pruning_class_pareto_global_fraction,
            iterative_pruning_class_pareto_minority_boost=self.iterative_pruning_class_pareto_minority_boost,
            iterative_pruning_bounded_prefilter_cap=self.iterative_pruning_bounded_prefilter_cap,
            iterative_pruning_bounded_max_evaluations=self.iterative_pruning_bounded_max_evaluations,
            iterative_pruning_bounded_max_runtime_seconds=self.iterative_pruning_bounded_max_runtime_seconds,
            iterative_pruning_bounded_candidate_fraction=self.iterative_pruning_bounded_candidate_fraction,
            iterative_pruning_bounded_min_candidates=self.iterative_pruning_bounded_min_candidates,
            iterative_pruning_bounded_enable_class_gating=self.iterative_pruning_bounded_enable_class_gating,
            iterative_pruning_bounded_multiclass_scale=self.iterative_pruning_bounded_multiclass_scale,
            iterative_pruning_bounded_imbalance_trigger=self.iterative_pruning_bounded_imbalance_trigger,
            iterative_pruning_bounded_imbalance_scale=self.iterative_pruning_bounded_imbalance_scale,
            iterative_pruning_bounded_use_cpss_overlay=self.iterative_pruning_bounded_use_cpss_overlay,
            iterative_pruning_bounded_cpss_pairs=self.iterative_pruning_bounded_cpss_pairs,
            iterative_pruning_bounded_cpss_stability_threshold=self.iterative_pruning_bounded_cpss_stability_threshold,
            iterative_pruning_bounded_cpss_min_stable_features=self.iterative_pruning_bounded_cpss_min_stable_features,
            iterative_pruning_bounded_cpss_min_jaccard=self.iterative_pruning_bounded_cpss_min_jaccard,
            iterative_pruning_bounded_cpss_max_score_drop=self.iterative_pruning_bounded_cpss_max_score_drop,
        )

    def _iterative_redundancy_pruning_selection(self, X, y, n_target_features):
        return self._iterative_redundancy_pruning_core(
            X,
            y,
            n_target_features=n_target_features,
            runtime_bounded=False,
        )

    def _iterative_redundancy_pruning_bounded_selection(self, X, y, n_target_features):
        return self._iterative_redundancy_pruning_core(
            X,
            y,
            n_target_features=n_target_features,
            runtime_bounded=True,
        )

    def _ktsp_selection(self, X, y, n_target_features):
        """
        k-TSP-inspired pairwise rank-rule selector.
        """
        from .methods.pairwise import ktsp_selection
        return ktsp_selection(
            X, y, n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            prefilter_fn=self._prefilter_feature_pool,
            ktsp_max_features=self.ktsp_max_features,
            ktsp_max_pairs=self.ktsp_max_pairs,
            ktsp_k_pairs=self.ktsp_k_pairs,
        )

    @staticmethod
    def _bh_qvalues(p_values):
        """Benjamini-Hochberg q-value transform with monotone adjustment."""
        from .methods.embedded import bh_qvalues
        return bh_qvalues(p_values)

    def _ipss_path_grid(self):
        """Build the regularization path grid for IPSS."""
        from .methods.embedded import ipss_path_grid
        return ipss_path_grid(
            ipss_importance_model=self.ipss_importance_model,
            ipss_min_c=self.ipss_min_c,
            ipss_max_c=self.ipss_max_c,
            ipss_path_grid_size=self.ipss_path_grid_size,
        )

    def _fit_ipss_model_importance(self, X_sub, y_sub, path_level, seed_shift):
        """Fit a single IPSS model and return feature importance vector."""
        from .methods.embedded import fit_ipss_model_importance
        return fit_ipss_model_importance(
            X_sub, y_sub, path_level, seed_shift,
            problem_type=self.problem_type,
            random_state=self.random_state,
            ipss_importance_model=self.ipss_importance_model,
            linear_svm_max_iter=self.linear_svm_max_iter,
        )

    def _estimate_ipss_statistics(self, X_pool, y, path_grid, seed_offset=0):
        """Estimate IPSS selection statistics over the regularization path."""
        from .methods.embedded import estimate_ipss_statistics
        return estimate_ipss_statistics(
            X_pool, y, path_grid, seed_offset=seed_offset,
            stability_subsample_fraction=self.stability_subsample_fraction,
            n_bootstrap_iterations=self.n_bootstrap_iterations,
            random_state=self.random_state,
            problem_type=self.problem_type,
            ipss_importance_model=self.ipss_importance_model,
            linear_svm_max_iter=self.linear_svm_max_iter,
            normalize_fn=self._normalize_vector_01,
        )

    def _select_eats_threshold(self, integrated_scores, null_scores):
        """Select the EATS elbow-adaptive threshold for IPSS."""
        from .methods.embedded import select_eats_threshold
        return select_eats_threshold(
            integrated_scores, null_scores,
            ipss_eats_exclusion_quantile=self.ipss_eats_exclusion_quantile,
            ipss_eats_min_threshold=self.ipss_eats_min_threshold,
            stability_selection_threshold=self.stability_selection_threshold,
        )

    def _ipss_selection(self, X, y, n_target_features):
        """Integrated path stability selection with optional EATS threshold calibration."""
        from .methods.embedded import ipss_selection
        return ipss_selection(
            X, y, n_target_features,
            problem_type=self.problem_type,
            random_state=self.random_state,
            mrmr_max_features=self.mrmr_max_features,
            prefilter_fn=self._prefilter_feature_pool,
            normalize_fn=self._normalize_vector_01,
            stability_subsample_fraction=self.stability_subsample_fraction,
            n_bootstrap_iterations=self.n_bootstrap_iterations,
            ipss_importance_model=self.ipss_importance_model,
            ipss_min_c=self.ipss_min_c,
            ipss_max_c=self.ipss_max_c,
            ipss_path_grid_size=self.ipss_path_grid_size,
            linear_svm_max_iter=self.linear_svm_max_iter,
            ipss_null_shuffle_rounds=self.ipss_null_shuffle_rounds,
            ipss_use_eats_threshold=self.ipss_use_eats_threshold,
            ipss_target_fdr=self.ipss_target_fdr,
            ipss_eats_exclusion_quantile=self.ipss_eats_exclusion_quantile,
            ipss_eats_min_threshold=self.ipss_eats_min_threshold,
            stability_selection_threshold=self.stability_selection_threshold,
            ipss_gate_min_classes=getattr(self, "ipss_gate_min_classes", 0),
            ipss_gate_min_p_over_n=getattr(self, "ipss_gate_min_p_over_n", 0.0),
        )

    def _stability_subsample_selection(self, X, y, n_target_features):
        """Complementary-subsampling stability selection (delegates to StabilitySelectionBase)."""
        runner = _SubsampleStability(
            subsample_fraction=self.stability_subsample_fraction,
            selection_threshold=self.stability_selection_threshold,
            selection_threshold_method=self.stability_threshold_method,
            selection_target_pfer=self.stability_target_pfer,
            eats_exclusion_quantile=self.ipss_eats_exclusion_quantile,
            eats_min_threshold=self.ipss_eats_min_threshold,
            n_bootstrap_iterations=self.n_bootstrap_iterations,
            random_state=self.random_state,
            problem_type=self.problem_type,
            linear_svm_max_iter=self.linear_svm_max_iter,
            mrmr_max_features=self.mrmr_max_features,
            use_loss_guided_validation=self.stability_use_loss_guided_validation,
            validation_fraction=self.stability_validation_fraction,
            validation_quantile=self.stability_validation_quantile,
            validation_min_samples=self.stability_validation_min_samples,
            parallel_n_jobs=getattr(self, 'parallel_n_jobs', 1),
        )
        return runner.run(
            X, y, n_target_features, self._prefilter_feature_pool,
            fit_score_fn=self._fit_and_score_fold,
        )

    def _tigress_stability_selection(self, X, y, n_target_features):
        """TIGRESS-style randomised stability-path selection (delegates to StabilitySelectionBase)."""
        runner = _TigressStability(
            subsample_fraction=self.stability_subsample_fraction,
            selection_threshold=self.stability_selection_threshold,
            selection_threshold_method=self.stability_threshold_method,
            selection_target_pfer=self.stability_target_pfer,
            eats_exclusion_quantile=self.ipss_eats_exclusion_quantile,
            eats_min_threshold=self.ipss_eats_min_threshold,
            n_bootstrap_iterations=self.n_bootstrap_iterations,
            random_state=self.random_state,
            problem_type=self.problem_type,
            linear_svm_max_iter=self.linear_svm_max_iter,
            mrmr_max_features=self.mrmr_max_features,
            ipss_min_c=self.ipss_min_c,
            ipss_max_c=self.ipss_max_c,
            ipss_path_grid_size=self.ipss_path_grid_size,
            parallel_n_jobs=getattr(self, 'parallel_n_jobs', 1),
        )
        results, scores = runner.run(
            X, y, n_target_features, self._prefilter_feature_pool,
        )
        if not results:
            return self._stability_subsample_selection(X, y, n_target_features)
        return results, scores

    def _subspace_stability_selection(self, X, y, n_target_features):
        """Subspace stability selection (delegates to StabilitySelectionBase)."""
        subsample_runner = _SubsampleStability(
            subsample_fraction=self.stability_subsample_fraction,
            selection_threshold=self.stability_selection_threshold,
            selection_threshold_method=self.stability_threshold_method,
            selection_target_pfer=self.stability_target_pfer,
            eats_exclusion_quantile=self.ipss_eats_exclusion_quantile,
            eats_min_threshold=self.ipss_eats_min_threshold,
            n_bootstrap_iterations=self.n_bootstrap_iterations,
            random_state=self.random_state,
            problem_type=self.problem_type,
            linear_svm_max_iter=self.linear_svm_max_iter,
            mrmr_max_features=self.mrmr_max_features,
            use_loss_guided_validation=self.stability_use_loss_guided_validation,
            validation_fraction=self.stability_validation_fraction,
            validation_quantile=self.stability_validation_quantile,
            validation_min_samples=self.stability_validation_min_samples,
            parallel_n_jobs=getattr(self, 'parallel_n_jobs', 1),
        )
        runner = _SubspaceStability(
            subsample_stability=subsample_runner,
            corr_threshold=self.cluster_stability_corr_threshold,
            selection_threshold=self.stability_selection_threshold,
        )
        return runner.run(
            X, y, n_target_features, self._prefilter_feature_pool,
            fit_score_fn=self._fit_and_score_fold,
        )

    def _decorrelated_stability_selection(self, X, y, n_target_features):
        """Decorrelated stability selection (delegates to StabilitySelectionBase)."""
        runner = _DecorrelatedStability(
            subsample_fraction=self.stability_subsample_fraction,
            selection_threshold=self.stability_selection_threshold,
            selection_threshold_method=self.stability_threshold_method,
            selection_target_pfer=self.stability_target_pfer,
            eats_exclusion_quantile=self.ipss_eats_exclusion_quantile,
            eats_min_threshold=self.ipss_eats_min_threshold,
            n_bootstrap_iterations=self.n_bootstrap_iterations,
            random_state=self.random_state,
            problem_type=self.problem_type,
            linear_svm_max_iter=self.linear_svm_max_iter,
            mrmr_max_features=self.mrmr_max_features,
            decorrelated_stability_eps=self.decorrelated_stability_eps,
            decorrelated_stability_min_max_abs_corr=self.decorrelated_stability_min_max_abs_corr,
            parallel_n_jobs=getattr(self, 'parallel_n_jobs', 1),
        )
        results, scores = runner.run(
            X, y, n_target_features, self._prefilter_feature_pool,
        )
        if not results:
            return self._stability_subsample_selection(X, y, n_target_features)
        return results, scores

    def _build_correlation_clusters(self, X_pool):
        """Build connected components using absolute-correlation thresholding."""
        runner = _ClusterStability(
            subsample_fraction=self.stability_subsample_fraction,
            selection_threshold=self.stability_selection_threshold,
            selection_threshold_method=self.stability_threshold_method,
            selection_target_pfer=self.stability_target_pfer,
            eats_exclusion_quantile=self.ipss_eats_exclusion_quantile,
            eats_min_threshold=self.ipss_eats_min_threshold,
            n_bootstrap_iterations=self.n_bootstrap_iterations,
            random_state=self.random_state,
            problem_type=self.problem_type,
            linear_svm_max_iter=self.linear_svm_max_iter,
            mrmr_max_features=self.mrmr_max_features,
            cluster_stability_corr_threshold=self.cluster_stability_corr_threshold,
            cluster_stability_max_per_cluster=self.cluster_stability_max_per_cluster,
            cluster_stability_min_cluster_freq=self.cluster_stability_min_cluster_freq,
            parallel_n_jobs=getattr(self, 'parallel_n_jobs', 1),
        )
        return runner._build_correlation_clusters(X_pool)

    def _cluster_stability_selection(self, X, y, n_target_features):
        """Cluster-aware stability selection (delegates to StabilitySelectionBase)."""
        runner = _ClusterStability(
            subsample_fraction=self.stability_subsample_fraction,
            selection_threshold=self.stability_selection_threshold,
            selection_threshold_method=self.stability_threshold_method,
            selection_target_pfer=self.stability_target_pfer,
            eats_exclusion_quantile=self.ipss_eats_exclusion_quantile,
            eats_min_threshold=self.ipss_eats_min_threshold,
            n_bootstrap_iterations=self.n_bootstrap_iterations,
            random_state=self.random_state,
            problem_type=self.problem_type,
            linear_svm_max_iter=self.linear_svm_max_iter,
            mrmr_max_features=self.mrmr_max_features,
            cluster_stability_corr_threshold=self.cluster_stability_corr_threshold,
            cluster_stability_max_per_cluster=self.cluster_stability_max_per_cluster,
            cluster_stability_min_cluster_freq=self.cluster_stability_min_cluster_freq,
            parallel_n_jobs=getattr(self, 'parallel_n_jobs', 1),
        )
        results, scores = runner.run(
            X, y, n_target_features, self._prefilter_feature_pool,
        )
        if not results:
            return self._stability_subsample_selection(X, y, n_target_features)
        return results, scores

    def _group_sparse_lasso_selection(self, X, y, n_target_features):
        """Group sparse lasso feature selection (VAL12_Suggestions §3.3)."""
        from .methods.group_fs import group_sparse_lasso_selection
        return group_sparse_lasso_selection(
            X, y, n_target_features,
            alpha=self.group_sparse_lasso_alpha,
            group_distance_threshold=self.group_sparse_lasso_distance_threshold,
            random_state=self.random_state,
        )

    def _random_selection(self, X, y, n_target_features):
        """Random feature selection baseline."""
        from .methods.filter import random_selection
        return random_selection(X, y, n_target_features, self.random_state)

    def _calculate_weighted_votes(self, method_results, n_features):
        """Calculate weighted votes for each feature across all methods with enhanced weights for Boruta and GA/SVM-RFE."""
        from .aggregation import calculate_weighted_votes
        return calculate_weighted_votes(method_results, n_features)

    def _build_feature_importance_figure(self, feature_votes, selected_indices):
        from .visualization import build_feature_importance_figure
        return build_feature_importance_figure(feature_votes, selected_indices)

    def _close_feature_importance_figure(self, fig) -> None:
        from .visualization import close_feature_importance_figure
        close_feature_importance_figure(fig)

    def _plot_feature_importance(self, feature_votes, selected_indices):
        """Plot feature importance based on votes."""
        payload_votes = np.asarray(feature_votes, dtype=float).ravel()
        payload_selected = np.asarray(selected_indices, dtype=int).ravel()
        self._feature_importance_plot_payload = (payload_votes.copy(), payload_selected.copy())

        # Close previous handle if still open to avoid figure accumulation across tests/runs.
        prev_fig = self.feature_importance_plot_
        if prev_fig is not None:
            self._close_feature_importance_figure(prev_fig)

        fig = self._build_feature_importance_figure(payload_votes, payload_selected)
        self.feature_importance_plot_ = fig
        if fig is None:
            return None
        # Keep only one live handle and release it from pyplot state immediately.
        self._close_feature_importance_figure(fig)
        return fig

    def _safe_normalize_scores(self, all_scores, selected_indices, n_features):
        """Normalize any score container into a non-negative [0,1] vector."""
        from .aggregation import safe_normalize_scores
        return safe_normalize_scores(all_scores, selected_indices, n_features)

    @contextmanager
    def _method_timeout(self, method_name: str):
        seconds = float(max(0.0, self.method_timeout_seconds))
        if seconds <= 0.0:
            yield
            return
        if not hasattr(signal, "SIGALRM"):
            yield
            return

        def _raise_timeout(signum, frame):
            raise TimeoutError(f"{method_name} timed out after {seconds:.1f}s")

        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)

    def _dispatch_method(self, method_name, method_fn, contract, X_uncorr, y, n_target, class_pareto_min_classes):
        """Dispatch a single FS method, handling contracts and special args."""
        n_samples = int(X_uncorr.shape[0])
        n_features = int(X_uncorr.shape[1])
        classes = np.unique(y)
        n_classes = int(classes.size)
        if contract is not None:
            if contract.supports_dataset(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
            ):
                return contract.compute(X_uncorr, y, n_target)
            else:
                return (
                    {
                        "selected_indices": np.array([], dtype=int),
                        "scores": {},
                        "skipped_by_contract": True,
                        "contract_runtime_class": str(contract.estimated_runtime_class),
                    },
                    {},
                )
        elif method_name == "class_pareto_front" and class_pareto_min_classes is not None:
            return method_fn(
                X_uncorr,
                y,
                n_target,
                class_pareto_min_classes=class_pareto_min_classes,
            )
        else:
            return method_fn(X_uncorr, y, n_target)

    def _run_single_method(
        self,
        method_name: str,
        method_fn,
        contract,
        X_uncorr: np.ndarray,
        y: np.ndarray,
        n_target: int,
        class_pareto_min_classes,
        use_timeout: bool = True,
        method_seed: int | None = None,
    ) -> Tuple[str, tuple, float]:
        """Run a single FS method.  Returns ``(method_name, result_tuple, runtime_seconds)``.

        Args:
            method_seed: Optional per-method seed override. When running methods
                in parallel, each method receives a unique seed derived from
                ``self.random_state + method_index`` (VAL12_Suggestions §1.2).
        """
        # Propagate parallel_n_jobs to sklearn estimators inside FS methods (FS-2).
        from .methods.embedded import set_sklearn_n_jobs
        set_sklearn_n_jobs(int(getattr(self, "parallel_n_jobs", 1) or 1))
        # Per-method seed override for statistical independence in parallel dispatch.
        _orig_rs = self.random_state
        if method_seed is not None:
            self.random_state = int(method_seed)

        t0 = perf_counter()
        try:
            if use_timeout:
                with self._method_timeout(method_name):
                    result = self._dispatch_method(
                        method_name, method_fn, contract,
                        X_uncorr, y, n_target, class_pareto_min_classes,
                    )
            else:
                result = self._dispatch_method(
                    method_name, method_fn, contract,
                    X_uncorr, y, n_target, class_pareto_min_classes,
                )
            runtime = perf_counter() - t0
            return method_name, result, runtime
        except TimeoutError:
            runtime = perf_counter() - t0
            result = (
                {
                    "selected_indices": np.array([], dtype=int),
                    "scores": {},
                    "timed_out": True,
                    "timeout_seconds": float(self.method_timeout_seconds),
                },
                {},
            )
            logger.warning("%s timed out after %.1fs", method_name, float(self.method_timeout_seconds))
            return method_name, result, runtime
        except Exception as exc:
            runtime = perf_counter() - t0
            logger.error("%s failed: %s", method_name, exc)
            return method_name, ({}, {}), runtime
        finally:
            self.random_state = _orig_rs

    def _derive_method_seed(self, method_name: str) -> Optional[int]:
        """Derive a stable per-method seed from the selector root seed.

        The mapping is independent of execution mode and task ordering, so the
        same method receives the same RNG stream in sequential and parallel
        execution, and also when unrelated methods are added or removed.
        """
        if self.random_state is None:
            return None
        base_seed = int(self.random_state)
        method_key = int(zlib.crc32(str(method_name).encode("utf-8")) & 0xFFFFFFFF)
        seed_seq = np.random.SeedSequence(
            [
                np.uint32(base_seed & 0xFFFFFFFF),
                np.uint32(method_key),
            ]
        )
        return int(seed_seq.generate_state(1, dtype=np.uint32)[0] % np.uint32(2**31 - 1))

    def _run_selection_methods(self, X_uncorr, y, n_target, class_pareto_min_classes: Optional[int] = None):
        """Run candidate selector methods and capture per-method runtime."""
        method_results = {}
        method_runtimes = {}
        n_classes = int(np.unique(np.asarray(y)).size)
        n_samples = int(np.asarray(X_uncorr).shape[0])
        n_features = int(np.asarray(X_uncorr).shape[1])
        if n_samples > 0 and n_features > 0 and (float(n_features) / float(max(1, n_samples))) >= 20.0:
            msg = (
                f"HDLSS regime detected (p={n_features}, n={n_samples}, p/n={n_features/max(1,n_samples):.1f}); "
                "runtime/variance may increase in this regime."
            )
            warnings.warn(msg, RuntimeWarning)
            logger.warning(msg)

        # T-P3-008: proof-of-concept contract adapters for selected methods.
        method_contracts = {}
        try:
            from .contracts import build_default_method_contracts

            method_contracts = build_default_method_contracts(self)
        except Exception as exc:
            method_contracts = {}

        # Build method_registry from the canonical METHOD_REGISTRY.
        method_registry = []
        for _key, _spec in METHOD_REGISTRY.items():
            if _spec.maturity == 'deprecated':
                continue
            if _spec.requires_gpu and not self._gpu_available:
                logger.info("Skipping %s: requires GPU", _key)
                continue
            _fn = getattr(self, _spec.fn_name, None)
            if _fn is None:
                continue
            method_registry.append((_key, f"- {_spec.label}...", _fn))

        logger.info("Applying %d selection methods to %d features",
                   len([m for m in method_registry if self.enabled_methods is None or m[0] in self.enabled_methods]),
                   X_uncorr.shape[1])

        # Build tasks list (filtered by enabled_methods).
        tasks = []
        for method_name, message, method_fn in method_registry:
            if self.enabled_methods is not None and method_name not in self.enabled_methods:
                continue
            contract = method_contracts.get(method_name)
            tasks.append((method_name, message, method_fn, contract))

        _fs_n_jobs = int(getattr(self, "parallel_n_jobs", 1) or 1)
        if _fs_n_jobs == -1:
            import os as _os
            _fs_n_jobs = _os.cpu_count() or 1

        if _fs_n_jobs > 1 and len(tasks) > 1:
            # ── Parallel FS method dispatch (FS-1) ────────────────────
            logger.info("Running %d FS methods with %d parallel workers", len(tasks), _fs_n_jobs)
            with concurrent.futures.ThreadPoolExecutor(max_workers=_fs_n_jobs) as executor:
                futures = {}
                for method_name, message, method_fn, contract in tasks:
                    logger.info("%s", message)
                    # Derive per-method seed for statistical independence in a
                    # way that is stable across sequential/parallel execution.
                    _method_seed = self._derive_method_seed(method_name)
                    future = executor.submit(
                        self._run_single_method,
                        method_name, method_fn, contract,
                        X_uncorr, y, n_target, class_pareto_min_classes,
                        False,  # use_timeout — signal-based timeout not safe in threads
                        _method_seed,
                    )
                    futures[future] = method_name

                for future in concurrent.futures.as_completed(futures):
                    mname, result, runtime = future.result()
                    method_results[mname] = result
                    method_runtimes[mname] = runtime
        else:
            # ── Sequential execution (original behaviour) ─────────────
            for method_name, message, method_fn, contract in tasks:
                logger.info("%s", message)
                logger.debug("Running %s", method_name)
                _method_seed = self._derive_method_seed(method_name)
                mname, result, runtime = self._run_single_method(
                    method_name, method_fn, contract,
                    X_uncorr, y, n_target, class_pareto_min_classes,
                    True,  # use_timeout — signal-based timeout OK in main thread
                    _method_seed,
                )
                method_results[mname] = result
                method_runtimes[mname] = runtime

        # GA/SVM-RFE REMOVED 2026-02-15 (global state mutation bugs)

        return method_results, method_runtimes

    def _get_inner_cv_splits(self, X, y):
        """Create repeated inner CV splits for noisy pairwise preference estimation."""
        from .cv import get_inner_cv_splits
        return get_inner_cv_splits(X, y, self.problem_type, self.random_state, self.inner_cv_splits, self.inner_cv_repeats)

    def _augment_training_data(self, X_train, y_train):
        """Lightweight counterfactual-style augmentation."""
        from .cv import augment_training_data
        return augment_training_data(X_train, y_train, self.problem_type, self.random_state)

    @staticmethod
    def _safe_balanced_accuracy(y_true, y_pred):
        from .cv import safe_balanced_accuracy
        return safe_balanced_accuracy(y_true, y_pred)

    @staticmethod
    def _safe_macro_f1(y_true, y_pred):
        from .cv import safe_macro_f1
        return safe_macro_f1(y_true, y_pred)

    def _compute_fold_conformal_efficiency(self, X_train, y_train, X_val, y_val):
        """Compute a fold-level conformal efficiency diagnostic when enabled."""
        if (not bool(getattr(self, "use_conformal_efficiency", False))) or self.problem_type != "classification":
            return {}
        y_train_arr = np.asarray(y_train).ravel()
        if np.unique(y_train_arr).size < 2:
            return {
                "conformal_efficiency_method": str(
                    getattr(self, "conformal_efficiency_method", "split") or "split"
                ),
                "conformal_efficiency_applied": False,
                "conformal_efficiency_skip_reason": "single_class",
                "conformal_singleton_rate": float("nan"),
            }
        try:
            try:
                from .conformal import compute_conformal_singleton_rate
            except Exception as exc:
                from tabnetics.feature_selection.conformal import compute_conformal_singleton_rate  # type: ignore

            model = make_logistic_regression(
                solver="lbfgs",
                penalty="l2",
                C=1.0,
                max_iter=getattr(self, "model_cv_lr_max_iter", 2000),
                class_weight="balanced",
                random_state=self.random_state,
            )
            return dict(
                compute_conformal_singleton_rate(
                    model=model,
                    X_train=np.asarray(X_train, dtype=float),
                    y_train=y_train_arr,
                    X_eval=np.asarray(X_val, dtype=float),
                    y_eval=np.asarray(y_val).ravel(),
                    alpha=float(getattr(self, "conformal_uq_alpha", 0.10) or 0.10),
                    method=str(
                        getattr(self, "conformal_efficiency_method", "split") or "split"
                    ),
                    seed=int(self.random_state),
                )
            )
        except Exception as exc:
            return {
                "conformal_efficiency_method": str(
                    getattr(self, "conformal_efficiency_method", "split") or "split"
                ),
                "conformal_efficiency_applied": False,
                "conformal_efficiency_skip_reason": str(type(exc).__name__),
                "conformal_singleton_rate": float("nan"),
            }

    def _fit_and_score_fold(self, X_train, y_train, X_val, y_val):
        """Fit a low-capacity downstream model and return scalar score + prediction signal.

        When ``eval_models_enabled`` is True, delegates to multi-classifier
        evaluation with fixed-weight aggregation (T-001).

        Returns
        -------
        2-tuple ``(score, signal)`` when single-model (default), or
        3-tuple ``(score, signal, per_model_scores)`` when multi-model is
        enabled **and** ``performance_oracle_mode == "multi_model_oracles"``
        (T-002).  portfolio.evaluate_candidate_library handles both shapes.
        """
        fold_meta = self._compute_fold_conformal_efficiency(X_train, y_train, X_val, y_val)
        if self.eval_models_enabled:
            from .cv import fit_and_score_fold_multimodel, CVEvaluationContext
            agg_score, signal, per_model = fit_and_score_fold_multimodel(
                X_train, y_train, X_val, y_val,
                self.problem_type, self.random_state,
                self.eval_models, self.eval_aggregate, self.eval_cvar_alpha,
                self.performance_balanced_weight, self.performance_macro_f1_weight,
                self.performance_use_adaptive_imbalance,
                self.performance_imbalance_ratio_trigger,
                self.performance_min_classes_for_adaptive,
                getattr(self, 'model_cv_lr_max_iter', 2000),
                eval_failure_strict_mode=self.eval_failure_strict_mode,
                eval_model_weight_strategy=self.eval_model_weight_strategy,
                evaluation_context=CVEvaluationContext(
                    purpose="evaluation_fold",
                    allow_learned_model_aggregation=False,
                ),
            )
            # Accumulate per-model diagnostics for inspection
            self._eval_multimodel_fold_log.append(per_model)
            # When multi_model_oracles mode is active, pass per-model
            # scores downstream so portfolio.py can collect them (T-002).
            if self.performance_oracle_mode == "multi_model_oracles":
                if fold_meta:
                    return float(agg_score), signal, per_model, fold_meta
                return float(agg_score), signal, per_model
            if fold_meta:
                return float(agg_score), signal, fold_meta
            return float(agg_score), signal

        from .cv import fit_and_score_fold
        score, signal = fit_and_score_fold(
            X_train, y_train, X_val, y_val,
            self.problem_type, self.random_state,
            self.performance_balanced_weight, self.performance_macro_f1_weight,
            self.performance_use_adaptive_imbalance,
            self.performance_imbalance_ratio_trigger,
            self.performance_min_classes_for_adaptive,
            getattr(self, 'model_cv_lr_max_iter', 2000),
        )
        if fold_meta:
            return float(score), signal, fold_meta
        return float(score), signal

    def _resolve_performance_weights(self, y_train):
        """Resolve performance-oracle weights. Delegates to cv.resolve_performance_weights."""
        from .cv import resolve_performance_weights
        return resolve_performance_weights(
            y_train, self.problem_type,
            self.performance_balanced_weight, self.performance_macro_f1_weight,
            self.performance_use_adaptive_imbalance,
            self.performance_imbalance_ratio_trigger,
            self.performance_min_classes_for_adaptive,
        )

    def _runtime_race_candidates(self, X, y, candidates):
        """
        Runtime-aware racing pass before full MNPO evaluation.
        Supports single-stage filtering or successive-halving elimination with
        optional Hoeffding/Bernstein confidence-bound safety checks.
        """
        from .mnpo.portfolio import runtime_race_candidates
        return runtime_race_candidates(
            X, y, candidates,
            get_inner_cv_splits_fn=self._get_inner_cv_splits,
            fit_and_score_fold_fn=self._fit_and_score_fold,
            runtime_racing_enabled=self.runtime_racing_enabled,
            runtime_racing_mode=self.runtime_racing_mode,
            runtime_racing_proxy_splits=self.runtime_racing_proxy_splits,
            runtime_racing_keep_fraction=self.runtime_racing_keep_fraction,
            runtime_racing_min_candidates=self.runtime_racing_min_candidates,
            runtime_racing_runtime_weight=self.runtime_racing_runtime_weight,
            runtime_racing_stages=self.runtime_racing_stages,
            runtime_racing_confidence_bound=self.runtime_racing_confidence_bound,
            runtime_racing_delta=self.runtime_racing_delta,
        )

    def _evaluate_candidate_library(self, X, y, candidates):
        """Evaluate candidate selectors and return oracle-ready statistics."""
        from .mnpo.portfolio import evaluate_candidate_library
        return evaluate_candidate_library(
            X, y, candidates,
            get_inner_cv_splits_fn=self._get_inner_cv_splits,
            fit_and_score_fold_fn=self._fit_and_score_fold,
            augment_training_data_fn=self._augment_training_data,
            use_robust_oracle=self.use_robust_oracle,
            complexity_use_runtime_penalty=self.complexity_use_runtime_penalty,
        )

    def _pairwise_pref_from_fold_scores(self, scores_i, scores_j):
        """Preference probability from repeated-CV score arrays with explicit ties."""
        from .mnpo.oracles import pairwise_pref_from_fold_scores
        return pairwise_pref_from_fold_scores(scores_i, scores_j, pairwise_delta=self.pairwise_delta)

    def _pairwise_pref_from_scalar(self, scalar_i, scalar_j, tie_margin=0.02, temperature=None):
        """Preference probability from scalar oracle values."""
        from .mnpo.oracles import pairwise_pref_from_scalar
        return pairwise_pref_from_scalar(scalar_i, scalar_j, tie_margin=tie_margin, temperature=temperature)

    def _estimate_oracle_preferences(self, candidate_names, evaluation):
        """Build pairwise preference matrices for enabled oracles."""
        from .mnpo.oracles import estimate_oracle_preferences
        return estimate_oracle_preferences(
            candidate_names, evaluation,
            pairwise_delta=self.pairwise_delta,
            use_tail_risk_oracle=False,
            tail_risk_alpha=self.tail_risk_alpha,
            use_qre_smoothing=self.use_qre_smoothing,
            qre_temperature_gamma=self.qre_temperature_gamma,
            use_regret_oracle=False,
            use_stability_oracle=self.use_stability_oracle,
            use_complexity_oracle=self.use_complexity_oracle,
            use_robust_oracle=self.use_robust_oracle,
            use_diversity_oracle=self.use_diversity_oracle,
            use_cvar=self.use_cvar,
            cvar_alpha=self.cvar_alpha,
            use_ubayfs=self.use_ubayfs,
            ubayfs_n_bootstrap=self.ubayfs_n_bootstrap,
            ubayfs_min_n=self.ubayfs_min_n,
            ubayfs_prior_weight=self.ubayfs_prior_weight,
            use_conformal_uq=self.use_conformal_uq,
            conformal_uq_alpha=self.conformal_uq_alpha,
            conformal_uq_min_folds=self.conformal_uq_min_folds,
            fold_preference_mode=self.fold_preference_mode,
            use_conformal_efficiency=self.use_conformal_efficiency,
            conformal_efficiency_method=self.conformal_efficiency_method,
            diversity_oracle_mode=self.diversity_oracle_mode,
            diversity_redundancy_weight=self.diversity_redundancy_weight,
            diversity_complementarity_weight=self.diversity_complementarity_weight,
            weighting_mode=self.oracle_weighting_mode,
            shapley_n_coalitions_max=self.shapley_n_coalitions_max,
            oracle_config=self.oracle,
        )

    def _fit_tritrust_weights(self, oracle_matrices):
        """TriTrust-style trust/ignore/flip calibration from oracle agreement."""
        from .mnpo.oracles import fit_tritrust_weights
        return fit_tritrust_weights(oracle_matrices)

    def _aggregate_payoff_matrix(self, oracle_matrices, oracle_weights):
        """Aggregate oracle preferences into an anti-symmetric payoff matrix."""
        from .mnpo.oracles import aggregate_payoff_matrix
        return aggregate_payoff_matrix(oracle_matrices, oracle_weights)

    def _normalize_vector_01(self, values):
        """Min-max normalize to [0,1] with safe fallback."""
        from .mnpo.oracles import normalize_vector_01
        return normalize_vector_01(values)

    @staticmethod
    def _normalized_mutual_info(values_a, values_b, n_bins=16):
        """Estimate normalized mutual information from two score vectors."""
        from .mnpo.oracles import normalized_mutual_info
        return normalized_mutual_info(values_a, values_b, n_bins=n_bins)

    @staticmethod
    def _discretize_signal(values: np.ndarray, n_bins: int = 8) -> np.ndarray:
        """Discretize a 1D signal into integer bins for MI/PID estimates."""
        from .mnpo.oracles import discretize_signal
        return discretize_signal(values, n_bins=n_bins)

    @staticmethod
    def _entropy_discrete(values: np.ndarray) -> float:
        from .mnpo.oracles import entropy_discrete
        return entropy_discrete(values)

    @staticmethod
    def _mutual_information_discrete(x: np.ndarray, y: np.ndarray) -> float:
        from .mnpo.oracles import mutual_information_discrete
        return mutual_information_discrete(x, y)

    @staticmethod
    def _pid_imin(signal_a: np.ndarray, signal_b: np.ndarray, target: np.ndarray, n_bins: int = 8) -> Tuple[float, float, float, float, float]:
        """Compute a lightweight I_min PID (redundancy, unique_a, unique_b, synergy, H(target))."""
        from .mnpo.oracles import pid_imin
        return pid_imin(signal_a, signal_b, target, n_bins=n_bins)

    def _mirror_descent_mnpo(self, payoff, reference_prior):
        """Reference-regularized mirror descent on the selector simplex."""
        from .mnpo.oracles import mirror_descent_mnpo
        return mirror_descent_mnpo(
            payoff, reference_prior,
            mirror_descent_steps=self.mirror_descent_steps,
            mirror_descent_eta=self.mirror_descent_eta,
            mirror_descent_lambda=self.mirror_descent_lambda,
        )

    def _extract_portfolio(self, candidate_names, candidate_weights, evaluation):
        """Select top weighted but complementary selector candidates."""
        from .mnpo.portfolio import extract_portfolio
        return extract_portfolio(
            candidate_names, candidate_weights, evaluation,
            portfolio_size=self.portfolio_size,
            use_diversity_oracle=self.use_diversity_oracle,
        )

    def _mnpo_aggregate_feature_votes(self, candidates, candidate_names, n_features):
        """Aggregate feature scores from portfolio-weighted candidates."""
        from .mnpo.portfolio import mnpo_aggregate_feature_votes
        return mnpo_aggregate_feature_votes(candidates, candidate_names, n_features)

    def _wrapper_refine_subset_score(self, X, y, feature_subset):
        """Score one feature subset using the same inner-CV fit/score pathway as MNPO."""
        from .mnpo.consensus import wrapper_refine_subset_score
        return wrapper_refine_subset_score(
            X, y, feature_subset,
            get_inner_cv_splits_fn=self._get_inner_cv_splits,
            fit_and_score_fold_fn=self._fit_and_score_fold,
        )

    def _apply_wrapper_refinement(self, X, y, vote_ranking, n_final_features):
        """
        Greedy wrapper refinement over top-ranked features.
        Uses inner-CV score improvements and falls back to vote ranking fill.
        """
        from .mnpo.consensus import apply_wrapper_refinement
        return apply_wrapper_refinement(
            X, y, vote_ranking, n_final_features,
            wrapper_refine_enabled=self.wrapper_refine_enabled,
            wrapper_refine_top_k=self.wrapper_refine_top_k,
            wrapper_refine_max_add=self.wrapper_refine_max_add,
            wrapper_refine_min_gain=self.wrapper_refine_min_gain,
            get_inner_cv_splits_fn=self._get_inner_cv_splits,
            fit_and_score_fold_fn=self._fit_and_score_fold,
        )

    def _build_rank_aggregation_candidate(self, source_candidates, n_target, n_final_features, n_features):
        """
        Build a synthetic candidate from per-method rankings.
        Supports:
          - Borda: sum of reversed-rank points.
          - RRA: robust rank aggregation score via order-statistic Beta CDF p-values.
        """
        from .mnpo.consensus import build_rank_aggregation_candidate
        return build_rank_aggregation_candidate(
            source_candidates, n_target, n_final_features, n_features,
            rank_aggregation_mode=self.rank_aggregation_mode,
            normalize_fn=self._normalize_vector_01,
        )

    def _mnpo_select_features(
        self,
        X_uncorr,
        y,
        n_target,
        n_final_features,
        method_results,
        method_runtimes,
        use_oracle_redundancy_penalty: Optional[bool] = None,
    ):
        """MNPO-inspired selector strategy."""
        if use_oracle_redundancy_penalty is None:
            use_oracle_redundancy_penalty = self.use_oracle_redundancy_penalty
        from .mnpo.portfolio import mnpo_select_features
        return mnpo_select_features(
            X_uncorr, y, n_target, n_final_features, method_results, method_runtimes,
            safe_normalize_scores_fn=self._safe_normalize_scores,
            calculate_weighted_votes_fn=self._calculate_weighted_votes,
            get_inner_cv_splits_fn=self._get_inner_cv_splits,
            fit_and_score_fold_fn=self._fit_and_score_fold,
            augment_training_data_fn=self._augment_training_data,
            oracle=self.oracle,
            mnpo_include_legacy_consensus=self.mnpo_include_legacy_consensus,
            mnpo_include_majority_consensus=self.mnpo_include_majority_consensus,
            mnpo_consensus_exclude_methods=self.mnpo_consensus_exclude_methods,
            mnpo_consensus_exclude_protect_top_k=self.mnpo_consensus_exclude_protect_top_k,
            use_tritrust=self.use_tritrust,
            use_oracle_redundancy_penalty=bool(use_oracle_redundancy_penalty),
            compute_tremble_sensitivity=self.compute_tremble_sensitivity,
            mirror_descent_steps=self.mirror_descent_steps,
            mirror_descent_eta=self.mirror_descent_eta,
            mirror_descent_lambda=self.mirror_descent_lambda,
            wrapper_refine_enabled=self.wrapper_refine_enabled,
            rank_aggregation_mode=self.rank_aggregation_mode,
            portfolio_size=self.portfolio_size,
            adaptive_portfolio_sizing_enabled=self.adaptive_portfolio_sizing_enabled,
            adaptive_size_min=self.adaptive_size_min,
            adaptive_size_max=self.adaptive_size_max,
            adaptive_sizing_variance_penalty=self.adaptive_sizing_variance_penalty,
            adaptive_sizing_variance_penalty_strength=self.adaptive_sizing_variance_penalty_strength,
            pareto_portfolio_sizing_enabled=self.pareto_portfolio_sizing_enabled,
            stability_weighted_aggregation_enabled=self.stability_weighted_aggregation_enabled,
            use_diversity_oracle=self.use_diversity_oracle,
            runtime_racing_enabled=self.runtime_racing_enabled,
            runtime_racing_mode=self.runtime_racing_mode,
            runtime_racing_proxy_splits=self.runtime_racing_proxy_splits,
            runtime_racing_keep_fraction=self.runtime_racing_keep_fraction,
            runtime_racing_min_candidates=self.runtime_racing_min_candidates,
            runtime_racing_runtime_weight=self.runtime_racing_runtime_weight,
            runtime_racing_stages=self.runtime_racing_stages,
            runtime_racing_confidence_bound=self.runtime_racing_confidence_bound,
            runtime_racing_delta=self.runtime_racing_delta,
            use_robust_oracle=self.use_robust_oracle,
            complexity_use_runtime_penalty=self.complexity_use_runtime_penalty,
            pairwise_delta=self.pairwise_delta,
            use_cvar=self.use_cvar,
            cvar_alpha=self.cvar_alpha,
            use_tail_risk_oracle=False,
            tail_risk_alpha=self.tail_risk_alpha,
            use_qre_smoothing=self.use_qre_smoothing,
            qre_temperature_gamma=self.qre_temperature_gamma,
            use_regret_oracle=False,
            use_stability_oracle=self.use_stability_oracle,
            use_complexity_oracle=self.use_complexity_oracle,
            diversity_oracle_mode=self.diversity_oracle_mode,
            oracle_weighting_mode=self.oracle_weighting_mode,
            shapley_n_coalitions_max=self.shapley_n_coalitions_max,
            shapley_bayesian_shrinkage=self.shapley_bayesian_shrinkage,
            shapley_bayesian_prior_strength=self.shapley_bayesian_prior_strength,
            use_interaction_oracle=self.use_interaction_oracle,
            interaction_oracle_min_n_train=self.interaction_oracle_min_n_train,
            interaction_oracle_pool_size_cap=self.interaction_oracle_pool_size_cap,
            interaction_oracle_pair_cap=self.interaction_oracle_pair_cap,
            use_ubayfs=self.use_ubayfs,
            ubayfs_n_bootstrap=self.ubayfs_n_bootstrap,
            ubayfs_min_n=self.ubayfs_min_n,
            ubayfs_prior_weight=self.ubayfs_prior_weight,
            use_conformal_uq=self.use_conformal_uq,
            conformal_uq_alpha=self.conformal_uq_alpha,
            conformal_uq_min_folds=self.conformal_uq_min_folds,
            fold_preference_mode=self.fold_preference_mode,
            use_conformal_efficiency=self.use_conformal_efficiency,
            conformal_efficiency_method=self.conformal_efficiency_method,
            oracle_weight_js_shrinkage=self.oracle_weight_js_shrinkage,
            payoff_shrinkage_kappa=self.payoff_shrinkage_kappa,
            diversity_redundancy_weight=self.diversity_redundancy_weight,
            diversity_complementarity_weight=self.diversity_complementarity_weight,
            wrapper_refine_top_k=self.wrapper_refine_top_k,
            wrapper_refine_max_add=self.wrapper_refine_max_add,
            wrapper_refine_min_gain=self.wrapper_refine_min_gain,
            performance_oracle_mode=self.performance_oracle_mode,
            mnpo_paradigm_aware_prior_enabled=self.mnpo_paradigm_aware_prior_enabled,
            mnpo_interaction_floor=self.mnpo_interaction_floor,
            rashomon_enabled=self.rashomon_enabled,
            rashomon_max_models=self.rashomon_max_models,
            rashomon_score_tolerance=self.rashomon_score_tolerance,
            random_state=self.random_state,
        )

    def fit_transform(self, X, y, n_final_features=30, return_result_object=True):
        """
        Apply ensemble feature selection with voting.
        
        Args:
            X: Input features
            y: Target variable
            n_final_features: Number of features to select
            return_result_object: If True, returns FeatureSelectionResult object
            
        Returns:
            If return_result_object is True: (transformed_X, FeatureSelectionResult)
            Otherwise: transformed_X
        """
        if not isinstance(X, np.ndarray): 
            X = np.array(X)
        if not isinstance(y, np.ndarray): 
            y = np.array(y)
        # Reset per-fit diagnostics accumulator so repeated calls do not leak memory.
        self._eval_multimodel_fold_log = []
        
        n_samples, n_features = X.shape
        if n_samples > 0 and n_features > 0 and (float(n_features) / float(max(1, n_samples))) >= 20.0:
            msg = (
                f"HDLSS regime detected (p={n_features}, n={n_samples}, p/n={n_features/max(1,n_samples):.1f}); "
                "runtime/variance may increase in this regime."
            )
            warnings.warn(msg, RuntimeWarning)
            logger.warning(msg)
        logger.info("Starting ensemble feature selection. Shape: %s", X.shape)
        
        # Track eliminated features
        eliminated_features = {}
        feature_mapping = np.arange(n_features)  # Maps current indices to original
        
        # Step 1: Variance filtering
        var_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_var_filtered = var_selector.fit_transform(X)
        var_support = var_selector.get_support()
        eliminated_features['low_variance'] = feature_mapping[~var_support].tolist()
        feature_mapping = feature_mapping[var_support]
        
        logger.info("After variance filtering: %s", X_var_filtered.shape)
        
        # Step 2: Scaling
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_var_filtered)
        
        # Step 3: Correlation removal
        df = pd.DataFrame(X_scaled)
        df_uncorr, dropped_corr = self._remove_correlated_features(df, self.correlation_threshold)
        X_uncorr = df_uncorr.values
        
        # Update feature mapping
        kept_cols = [int(col) for col in df_uncorr.columns]
        eliminated_features['high_correlation'] = [feature_mapping[i] for i in range(len(feature_mapping)) 
                                                  if i not in kept_cols]
        feature_mapping = feature_mapping[kept_cols]
        
        logger.info("After correlation removal: %d features remaining", X_uncorr.shape[1])

        # Initialize score cache (P1-1: eliminates redundant MI/F-test computations)
        self._score_cache = _FeatureScoreCache(
            X_uncorr, y, random_state=self.random_state, problem_type=self.problem_type
        )

        # GA/SVM-RFE auto-enabling logic REMOVED 2026-02-15
        # (method deprecated due to global state mutation bugs)

        # Calculate target features for initial selection
        n_target = self._calculate_target_features(X_uncorr.shape[1], n_final_features)
        classes = np.unique(np.asarray(y).ravel())
        n_classes = int(classes.size)

        class_pareto_min_classes_effective = int(self.class_pareto_min_classes)
        use_oracle_redundancy_penalty_effective = bool(self.use_oracle_redundancy_penalty)
        binary_redundancy_penalty_disabled = False
        binary_class_pareto_override = False
        if n_classes == 2:
            if self.disable_redundancy_penalty_binary and self.use_oracle_redundancy_penalty:
                use_oracle_redundancy_penalty_effective = False
                binary_redundancy_penalty_disabled = True
            if self.disable_class_pareto_binary:
                class_pareto_min_classes_effective = int(max(3, self.class_pareto_min_classes))
                binary_class_pareto_override = True

        # Step 3b: Tier 2 screening (T-004) — interaction-aware pool reduction
        if self.screening_enabled and self.screening_method != "none":
            try:
                from .methods.screening import screen_features

                screened_idx = screen_features(
                    X_uncorr, y,
                    enabled=self.screening_enabled,
                    method=self.screening_method,
                    pool_cap=self.screening_pool_cap,
                    stir_n_neighbors=self.screening_stir_n_neighbors,
                    stir_n_iter=self.screening_stir_n_iter,
                    stir_keep_fraction=self.screening_stir_keep_fraction,
                    stir_min_features=self.screening_stir_min_features,
                    evalue_alpha=self.screening_evalue_alpha,
                    evalue_min_features=self.screening_evalue_min_features,
                    random_state=self.random_state,
                )
                if screened_idx is not None:
                    X_uncorr = X_uncorr[:, screened_idx]
                    feature_mapping = feature_mapping[screened_idx]
                    # Rebuild score cache for the reduced pool
                    self._score_cache = _FeatureScoreCache(
                        X_uncorr, y, random_state=self.random_state,
                        problem_type=self.problem_type,
                    )
                    n_target = self._calculate_target_features(
                        X_uncorr.shape[1], n_final_features
                    )
                    logger.info(
                        "Tier 2 screening (%s): %d features retained",
                        self.screening_method, X_uncorr.shape[1],
                    )
            except Exception as exc:
                logger.exception(
                    "Tier 2 screening failed; proceeding without screening"
                )

        # Step 4: Apply all selection methods
        method_results, method_runtimes = self._run_selection_methods(
            X_uncorr,
            y,
            n_target,
            class_pareto_min_classes=(
                class_pareto_min_classes_effective if binary_class_pareto_override else None
            ),
        )

        # Step 5: Aggregate methods into final feature votes
        mnpo_summary = None
        self.mnpo_diagnostics_ = {}
        if self.selection_strategy == 'mnpo_portfolio':
            mnpo_output = self._mnpo_select_features(
                X_uncorr,
                y,
                n_target,
                n_final_features,
                method_results,
                method_runtimes,
                use_oracle_redundancy_penalty=use_oracle_redundancy_penalty_effective,
            )
            if mnpo_output is not None:
                selected_indices_local, feature_votes, feature_details, mnpo_diagnostics, mnpo_summary = mnpo_output
                self.mnpo_diagnostics_ = mnpo_diagnostics
            else:
                logger.warning("MNPO strategy fallback: using legacy weighted voting.")
                feature_votes, feature_details = self._calculate_weighted_votes(method_results, X_uncorr.shape[1])
                selected_indices_local = np.argsort(feature_votes)[::-1][:n_final_features]
        else:
            feature_votes, feature_details = self._calculate_weighted_votes(method_results, X_uncorr.shape[1])
            selected_indices_local = np.argsort(feature_votes)[::-1][:n_final_features]

        # Step 6: Ensure final indices are valid and padded from vote ranking if needed.
        vote_ranking = np.argsort(feature_votes)[::-1]
        if selected_indices_local is None:
            selected_indices_local = np.array([], dtype=int)
        selected_indices_local = np.asarray(selected_indices_local, dtype=int).ravel()
        ordered_selected = []
        seen_selected = set()
        for idx in selected_indices_local:
            idx_i = int(idx)
            if not (0 <= idx_i < X_uncorr.shape[1]) or idx_i in seen_selected:
                continue
            seen_selected.add(idx_i)
            ordered_selected.append(idx_i)
        selected_indices_local = np.asarray(ordered_selected, dtype=int)
        if selected_indices_local.size < n_final_features:
            fill = []
            selected_set = set(int(i) for i in selected_indices_local.tolist())
            for idx in vote_ranking:
                idx_i = int(idx)
                if idx_i in selected_set:
                    continue
                fill.append(idx_i)
                selected_set.add(idx_i)
                if selected_indices_local.size + len(fill) >= n_final_features:
                    break
            if fill:
                selected_indices_local = np.concatenate(
                    [selected_indices_local, np.asarray(fill, dtype=int)]
                )
        selected_indices_local = selected_indices_local[:n_final_features]
        
        # Map back to original indices
        selected_indices_original = feature_mapping[selected_indices_local]

        # Optional reporting-only uncertainty diagnostics for feature importances.
        importance_uq_meta: Dict[str, Any] = {
            "importance_uq_enabled": bool(self.importance_uq_enabled),
            "importance_uq_computed": False,
            "importance_uq_reason": "disabled",
            "importance_uq_n_folds": 0,
            "unstable_threshold": 0.0,
        }
        feature_importance_mean: Dict[int, float] = {}
        feature_importance_variance: Dict[int, float] = {}
        unstable_feature_indices: List[int] = []
        if self.importance_uq_enabled:
            try:
                from .cv import compute_feature_importance_uq

                uq = compute_feature_importance_uq(
                    X_uncorr,
                    y,
                    problem_type=self.problem_type,
                    random_state=self.random_state,
                    inner_cv_splits=self.inner_cv_splits,
                    inner_cv_repeats=self.inner_cv_repeats,
                    min_cv_folds=self.importance_uq_min_cv_folds,
                    model_cv_lr_max_iter=self.linear_svm_max_iter,
                )
                importance_uq_meta.update(
                    {
                        "importance_uq_enabled": bool(uq.get("importance_uq_enabled", True)),
                        "importance_uq_computed": bool(uq.get("importance_uq_computed", False)),
                        "importance_uq_reason": str(uq.get("importance_uq_reason", "unknown")),
                        "importance_uq_n_folds": int(uq.get("importance_uq_n_folds", 0) or 0),
                        "unstable_threshold": float(uq.get("unstable_threshold", 0.0) or 0.0),
                    }
                )
                imp_mean_local = np.asarray(
                    uq.get("importance_mean", np.zeros(X_uncorr.shape[1], dtype=float)),
                    dtype=float,
                ).ravel()
                imp_var_local = np.asarray(
                    uq.get("importance_variance", np.zeros(X_uncorr.shape[1], dtype=float)),
                    dtype=float,
                ).ravel()
                imp_mean_local = np.nan_to_num(imp_mean_local, nan=0.0, posinf=0.0, neginf=0.0)
                imp_var_local = np.nan_to_num(imp_var_local, nan=0.0, posinf=0.0, neginf=0.0)
                for local_idx, orig_idx in enumerate(feature_mapping.tolist()):
                    if local_idx >= imp_mean_local.size or local_idx >= imp_var_local.size:
                        continue
                    feature_importance_mean[int(orig_idx)] = float(imp_mean_local[local_idx])
                    feature_importance_variance[int(orig_idx)] = float(imp_var_local[local_idx])

                unstable_local = np.asarray(
                    uq.get("unstable_feature_indices", np.array([], dtype=int)),
                    dtype=int,
                ).ravel()
                for idx in unstable_local.tolist():
                    idx_i = int(idx)
                    if 0 <= idx_i < feature_mapping.size:
                        unstable_feature_indices.append(int(feature_mapping[idx_i]))
                unstable_feature_indices = sorted(set(unstable_feature_indices))
            except Exception as exc:
                logger.exception("Importance UQ computation failed; continuing without UQ diagnostics")
                importance_uq_meta.update(
                    {
                        "importance_uq_computed": False,
                        "importance_uq_reason": "uq_exception",
                        "importance_uq_n_folds": 0,
                        "unstable_threshold": 0.0,
                    }
                )
        
        # Create comprehensive result object
        all_features_info = {}
        for orig_idx in range(n_features):
            if orig_idx in eliminated_features.get('low_variance', []):
                reason = 'low_variance'
            elif orig_idx in eliminated_features.get('high_correlation', []):
                reason = 'high_correlation'
            else:
                reason = None
                
            all_features_info[orig_idx] = {
                'eliminated': reason is not None,
                'elimination_reason': reason,
                'votes': 0,
                'method_details': {}
            }
        
        # Add voting details for non-eliminated features
        for local_idx, orig_idx in enumerate(feature_mapping):
            all_features_info[orig_idx]['votes'] = feature_votes[local_idx]
            all_features_info[orig_idx]['method_details'] = feature_details[local_idx]
        
        result_method_results = {method: results for method, (results, _) in method_results.items()}
        if mnpo_summary is not None:
            result_method_results['mnpo_portfolio'] = mnpo_summary

        # Create result object
        result = FeatureSelectionResult(
            selected_feature_indices=selected_indices_original,
            selected_feature_votes={idx: feature_votes[np.where(feature_mapping == idx)[0][0]] 
                                   for idx in selected_indices_original},
            all_features_info=all_features_info,
            method_results=result_method_results,
            eliminated_features=eliminated_features,
            feature_importance_mean=feature_importance_mean,
            feature_importance_variance=feature_importance_variance,
            unstable_feature_indices=unstable_feature_indices,
            importance_uq=importance_uq_meta,
            config={
                'n_final_features': n_final_features,
                'n_target_features': n_target,
                'n_bootstrap_iterations': self.n_bootstrap_iterations,
                'n_classes': int(n_classes),
                'variance_threshold': self.variance_threshold,
                'correlation_threshold': self.correlation_threshold,
                # 'use_ga_svm_rfe': REMOVED 2026-02-15 (global state mutation bugs)
                'selection_strategy': self.selection_strategy,
                'inner_cv_splits': self.inner_cv_splits,
                'inner_cv_repeats': self.inner_cv_repeats,
                'pairwise_delta': self.pairwise_delta,
                'portfolio_size': self.portfolio_size,
                'adaptive_portfolio_sizing_enabled': self.adaptive_portfolio_sizing_enabled,
                'adaptive_size_min': self.adaptive_size_min,
                'adaptive_size_max': self.adaptive_size_max,
                'adaptive_sizing_variance_penalty': bool(self.adaptive_sizing_variance_penalty),
                'adaptive_sizing_variance_penalty_strength': float(self.adaptive_sizing_variance_penalty_strength),
                'rashomon_enabled': self.rashomon_enabled,
                'rashomon_max_models': self.rashomon_max_models,
                'rashomon_score_tolerance': self.rashomon_score_tolerance,
                'mnpo_consensus_exclude_methods': list(self.mnpo_consensus_exclude_methods),
                'mnpo_consensus_exclude_protect_top_k': int(self.mnpo_consensus_exclude_protect_top_k),
                'mnpo_paradigm_aware_prior_enabled': bool(self.mnpo_paradigm_aware_prior_enabled),
                'mnpo_interaction_floor': float(self.mnpo_interaction_floor),
                'runtime_racing_enabled': self.runtime_racing_enabled,
                'runtime_racing_proxy_splits': self.runtime_racing_proxy_splits,
                'runtime_racing_keep_fraction': self.runtime_racing_keep_fraction,
                'runtime_racing_min_candidates': self.runtime_racing_min_candidates,
                'runtime_racing_runtime_weight': self.runtime_racing_runtime_weight,
                'runtime_racing_mode': self.runtime_racing_mode,
                'runtime_racing_stages': self.runtime_racing_stages,
                'runtime_racing_confidence_bound': self.runtime_racing_confidence_bound,
                'runtime_racing_delta': self.runtime_racing_delta,
                'use_tritrust': self.use_tritrust,
                'use_stability_oracle': self.use_stability_oracle,
                'use_complexity_oracle': self.use_complexity_oracle,
                'complexity_use_runtime_penalty': self.complexity_use_runtime_penalty,
                'method_timeout_seconds': self.method_timeout_seconds,
                'parallel_n_jobs': self.parallel_n_jobs,
                'linear_svm_max_iter': self.linear_svm_max_iter,
                'use_robust_oracle': self.use_robust_oracle,
                'use_diversity_oracle': self.use_diversity_oracle,
                'use_cvar': bool(self.use_cvar),
                'cvar_alpha': float(self.cvar_alpha),
                'use_tail_risk_oracle': self.use_tail_risk_oracle,
                'tail_risk_alpha': self.tail_risk_alpha,
                'use_regret_oracle': self.use_regret_oracle,
                'use_qre_smoothing': self.use_qre_smoothing,
                'qre_temperature_gamma': self.qre_temperature_gamma,
                'use_oracle_redundancy_penalty': self.use_oracle_redundancy_penalty,
                'use_oracle_redundancy_penalty_effective': bool(use_oracle_redundancy_penalty_effective),
                'disable_redundancy_penalty_binary': bool(self.disable_redundancy_penalty_binary),
                'binary_redundancy_penalty_disabled': int(bool(binary_redundancy_penalty_disabled)),
                'compute_tremble_sensitivity': self.compute_tremble_sensitivity,
                'oracle_weighting_mode': str(self.oracle_weighting_mode),
                'shapley_n_coalitions_max': int(self.shapley_n_coalitions_max),
                'shapley_bayesian_shrinkage': bool(self.shapley_bayesian_shrinkage),
                'shapley_bayesian_prior_strength': float(self.shapley_bayesian_prior_strength),
                'use_interaction_oracle': bool(self.use_interaction_oracle),
                'interaction_oracle_min_n_train': int(self.interaction_oracle_min_n_train),
                'interaction_oracle_pool_size_cap': int(self.interaction_oracle_pool_size_cap),
                'interaction_oracle_pair_cap': int(self.interaction_oracle_pair_cap),
                'use_ubayfs': bool(self.use_ubayfs),
                'ubayfs_n_bootstrap': int(self.ubayfs_n_bootstrap),
                'ubayfs_min_n': int(self.ubayfs_min_n),
                'ubayfs_prior_weight': float(self.ubayfs_prior_weight),
                'use_conformal_uq': bool(self.use_conformal_uq),
                'conformal_uq_alpha': float(self.conformal_uq_alpha),
                'conformal_uq_min_folds': int(self.conformal_uq_min_folds),
                'diversity_oracle_mode': self.diversity_oracle_mode,
                'diversity_redundancy_weight': self.diversity_redundancy_weight,
                'diversity_complementarity_weight': self.diversity_complementarity_weight,
                'performance_balanced_weight': self.performance_balanced_weight,
                'performance_macro_f1_weight': self.performance_macro_f1_weight,
                'performance_use_adaptive_imbalance': self.performance_use_adaptive_imbalance,
                'performance_imbalance_ratio_trigger': self.performance_imbalance_ratio_trigger,
                'performance_min_classes_for_adaptive': self.performance_min_classes_for_adaptive,
                'rank_aggregation_mode': self.rank_aggregation_mode,
                'wrapper_refine_enabled': self.wrapper_refine_enabled,
                'wrapper_refine_top_k': self.wrapper_refine_top_k,
                'wrapper_refine_max_add': self.wrapper_refine_max_add,
                'wrapper_refine_min_gain': self.wrapper_refine_min_gain,
                'ova_negative_ratio': self.ova_negative_ratio,
                'ova_min_classes': self.ova_min_classes,
                'ova_min_pos_samples': self.ova_min_pos_samples,
                'ova_class_weight_mode': self.ova_class_weight_mode,
                'ova_aggregation_mode': self.ova_aggregation_mode,
                'ova_aggregation_p': self.ova_aggregation_p,
                'ova_linear_backend': self.ova_linear_backend,
                'ova_enable_calibration': self.ova_enable_calibration,
                'ova_calibration_cv': self.ova_calibration_cv,
                'ecoc_min_classes': self.ecoc_min_classes,
                'ecoc_max_ovo_pairs': self.ecoc_max_ovo_pairs,
                'ecoc_random_code_bits': self.ecoc_random_code_bits,
                'ecoc_class_complexity_weight': self.ecoc_class_complexity_weight,
                'ecoc_include_ova_tasks': self.ecoc_include_ova_tasks,
                'ecoc_negative_ratio': self.ecoc_negative_ratio,
                'joint_multiclass_min_classes': self.joint_multiclass_min_classes,
                'joint_multiclass_max_features': self.joint_multiclass_max_features,
                'joint_multiclass_path_grid_size': self.joint_multiclass_path_grid_size,
                'joint_multiclass_min_c': self.joint_multiclass_min_c,
                'joint_multiclass_max_c': self.joint_multiclass_max_c,
                'joint_multiclass_l1_ratio': self.joint_multiclass_l1_ratio,
                'joint_multiclass_univariate_blend': self.joint_multiclass_univariate_blend,
                'dove_min_classes': self.dove_min_classes,
                'dove_max_pairs_per_class': self.dove_max_pairs_per_class,
                'dove_path_grid_size': self.dove_path_grid_size,
                'dove_specificity_weight': self.dove_specificity_weight,
                'dove_minority_boost': self.dove_minority_boost,
                'sparse_multinomial_min_classes': self.sparse_multinomial_min_classes,
                'sparse_multinomial_max_features': self.sparse_multinomial_max_features,
                'sparse_multinomial_path_grid_size': self.sparse_multinomial_path_grid_size,
                'sparse_multinomial_min_c': self.sparse_multinomial_min_c,
                'sparse_multinomial_max_c': self.sparse_multinomial_max_c,
                'sparse_multinomial_backend': self.sparse_multinomial_backend,
                'sparse_multinomial_l1_ratio': self.sparse_multinomial_l1_ratio,
                'sparse_multinomial_univariate_blend': self.sparse_multinomial_univariate_blend,
                'sparse_multinomial_max_iter': self.sparse_multinomial_max_iter,
                'sparse_multinomial_screening_mode': self.sparse_multinomial_screening_mode,
                'sparse_multinomial_screening_keep_fraction': self.sparse_multinomial_screening_keep_fraction,
                'sparse_multinomial_screening_min_features': self.sparse_multinomial_screening_min_features,
                'sparse_multinomial_screening_fallback_on_failure': self.sparse_multinomial_screening_fallback_on_failure,
                'nsc_shrinkage_grid_size': self.nsc_shrinkage_grid_size,
                'nsc_min_classes': self.nsc_min_classes,
                'nsc_thresholding_mode': self.nsc_thresholding_mode,
                'nsc_order_quantile': self.nsc_order_quantile,
                'nsc_deep_shrinkage_search': self.nsc_deep_shrinkage_search,
                'class_pareto_min_classes': self.class_pareto_min_classes,
                'class_pareto_min_classes_effective': int(class_pareto_min_classes_effective),
                'class_pareto_top_per_class': self.class_pareto_top_per_class,
                'disable_class_pareto_binary': bool(self.disable_class_pareto_binary),
                'binary_class_pareto_override': int(bool(binary_class_pareto_override)),
                'class_pareto_global_fraction': self.class_pareto_global_fraction,
                'class_pareto_minority_boost': self.class_pareto_minority_boost,
                'class_pareto_kw_weight': self.class_pareto_kw_weight,
                'sdr_min_classes': int(self.sdr_min_classes),
                'sdr_prefilter_max_features': int(self.sdr_prefilter_max_features),
                'sdr_n_components': int(self.sdr_n_components),
                'sdr_covariance_ridge': float(self.sdr_covariance_ridge),
                'per_class_quota_enabled': self.per_class_quota_enabled,
                'per_class_quota_min_per_class': self.per_class_quota_min_per_class,
                'per_class_quota_max_fraction': self.per_class_quota_max_fraction,
                'hsic_lasso_alpha': self.hsic_lasso_alpha,
                'hsic_lasso_prefilter_max_features': self.hsic_lasso_prefilter_max_features,
                'hsic_lasso_feature_sigma': self.hsic_lasso_feature_sigma,
                'hsic_lasso_target_sigma': self.hsic_lasso_target_sigma,
                'hsic_lasso_relevance_blend': self.hsic_lasso_relevance_blend,
                'hsic_lasso_max_iter': self.hsic_lasso_max_iter,
                'hsic_lasso_binary_delta_enabled': bool(self.hsic_lasso_binary_delta_enabled),
                'hsic_lasso_binary_delta_min_samples': int(self.hsic_lasso_binary_delta_min_samples),
                'slce_prefilter_max_features': int(self.slce_prefilter_max_features),
                'slce_min_samples': int(self.slce_min_samples),
                'slce_ridge': float(self.slce_ridge),
                'treeshap_min_samples': int(self.treeshap_min_samples),
                'treeshap_n_estimators': int(self.treeshap_n_estimators),
                'treeshap_multi_seed_runs': int(self.treeshap_multi_seed_runs),
                'oaenet_min_samples': int(self.oaenet_min_samples),
                'oaenet_prescreen_max_features': int(self.oaenet_prescreen_max_features),
                'oaenet_l1_ratio': float(self.oaenet_l1_ratio),
                'oaenet_c_grid_size': int(self.oaenet_c_grid_size),
                'ktsp_max_features': self.ktsp_max_features,
                'ktsp_k_pairs': self.ktsp_k_pairs,
                'ktsp_max_pairs': self.ktsp_max_pairs,
                'mrmr_max_features': self.mrmr_max_features,
                'mrmr_redundancy_weight': self.mrmr_redundancy_weight,
                'mrmr_mi_redundancy_enabled': bool(self.mrmr_mi_redundancy_enabled),
                'mrmr_mi_n_bins': int(self.mrmr_mi_n_bins),
                'cmim_min_samples': int(self.cmim_min_samples),
                'cmim_n_bins': int(self.cmim_n_bins),
                'fcbf_n_bins': int(self.fcbf_n_bins),
                'iterative_pruning_pool_factor': self.iterative_pruning_pool_factor,
                'iterative_pruning_max_rounds': self.iterative_pruning_max_rounds,
                'iterative_pruning_min_improvement': self.iterative_pruning_min_improvement,
                'iterative_pruning_max_cumulative_loss': self.iterative_pruning_max_cumulative_loss,
                'iterative_pruning_redundancy_weight': self.iterative_pruning_redundancy_weight,
                'iterative_pruning_bounded_prefilter_cap': self.iterative_pruning_bounded_prefilter_cap,
                'iterative_pruning_bounded_candidate_fraction': self.iterative_pruning_bounded_candidate_fraction,
                'iterative_pruning_bounded_min_candidates': self.iterative_pruning_bounded_min_candidates,
                'iterative_pruning_bounded_max_evaluations': self.iterative_pruning_bounded_max_evaluations,
                'iterative_pruning_bounded_max_runtime_seconds': self.iterative_pruning_bounded_max_runtime_seconds,
                'iterative_pruning_bounded_enable_class_gating': self.iterative_pruning_bounded_enable_class_gating,
                'iterative_pruning_bounded_multiclass_scale': self.iterative_pruning_bounded_multiclass_scale,
                'iterative_pruning_bounded_imbalance_trigger': self.iterative_pruning_bounded_imbalance_trigger,
                'iterative_pruning_bounded_imbalance_scale': self.iterative_pruning_bounded_imbalance_scale,
                'iterative_pruning_bounded_use_cpss_overlay': self.iterative_pruning_bounded_use_cpss_overlay,
                'iterative_pruning_bounded_cpss_pairs': self.iterative_pruning_bounded_cpss_pairs,
                'iterative_pruning_bounded_cpss_stability_threshold': self.iterative_pruning_bounded_cpss_stability_threshold,
                'iterative_pruning_bounded_cpss_min_stable_features': self.iterative_pruning_bounded_cpss_min_stable_features,
                'iterative_pruning_bounded_cpss_min_jaccard': self.iterative_pruning_bounded_cpss_min_jaccard,
                'iterative_pruning_bounded_cpss_max_score_drop': self.iterative_pruning_bounded_cpss_max_score_drop,
                'iterative_pruning_class_pareto_prefilter_enabled': self.iterative_pruning_class_pareto_prefilter_enabled,
                'iterative_pruning_class_pareto_min_classes': self.iterative_pruning_class_pareto_min_classes,
                'iterative_pruning_class_pareto_top_per_class': self.iterative_pruning_class_pareto_top_per_class,
                'iterative_pruning_class_pareto_global_fraction': self.iterative_pruning_class_pareto_global_fraction,
                'iterative_pruning_class_pareto_minority_boost': self.iterative_pruning_class_pareto_minority_boost,
                'iterative_pruning_class_pareto_stability_gate_enabled': self.iterative_pruning_class_pareto_stability_gate_enabled,
                'iterative_pruning_class_pareto_stability_subsamples': self.iterative_pruning_class_pareto_stability_subsamples,
                'iterative_pruning_class_pareto_stability_fraction': self.iterative_pruning_class_pareto_stability_fraction,
                'iterative_pruning_class_pareto_stability_threshold': self.iterative_pruning_class_pareto_stability_threshold,
                'iterative_pruning_class_pareto_stability_min_overlap': self.iterative_pruning_class_pareto_stability_min_overlap,
                'iterative_pruning_class_pareto_stability_min_stable_features': self.iterative_pruning_class_pareto_stability_min_stable_features,
                'iterative_pruning_class_pareto_stability_fallback_on_failure': self.iterative_pruning_class_pareto_stability_fallback_on_failure,
                'stability_subsample_fraction': self.stability_subsample_fraction,
                'stability_selection_threshold': self.stability_selection_threshold,
                'stability_threshold_method': self.stability_threshold_method,
                'stability_target_pfer': self.stability_target_pfer,
                'stability_use_loss_guided_validation': self.stability_use_loss_guided_validation,
                'stability_validation_fraction': self.stability_validation_fraction,
                'stability_validation_quantile': self.stability_validation_quantile,
                'stability_validation_min_samples': self.stability_validation_min_samples,
                'ipss_path_grid_size': self.ipss_path_grid_size,
                'ipss_min_c': self.ipss_min_c,
                'ipss_max_c': self.ipss_max_c,
                'ipss_target_fdr': self.ipss_target_fdr,
                'ipss_null_shuffle_rounds': self.ipss_null_shuffle_rounds,
                'ipss_use_eats_threshold': self.ipss_use_eats_threshold,
                'ipss_eats_exclusion_quantile': self.ipss_eats_exclusion_quantile,
                'ipss_eats_min_threshold': self.ipss_eats_min_threshold,
                'ipss_importance_model': self.ipss_importance_model,
                'cluster_stability_corr_threshold': self.cluster_stability_corr_threshold,
                'cluster_stability_max_per_cluster': self.cluster_stability_max_per_cluster,
                'cluster_stability_min_cluster_freq': self.cluster_stability_min_cluster_freq,
                'copula_knockoff_draws': self.copula_knockoff_draws,
                'copula_alpha_kn': self.copula_alpha_kn,
                'copula_alpha_ebh': self.copula_alpha_ebh,
                'copula_truncation_level': self.copula_truncation_level,
                'copula_generator': self.copula_generator,
                'copula_deepdrk_latent_fraction': self.copula_deepdrk_latent_fraction,
                'copula_deepdrk_noise_scale': self.copula_deepdrk_noise_scale,
                'copula_derandomize_runs': int(self.copula_derandomize_runs),
                'copula_stabilizer_runs': self.copula_stabilizer_runs,
                'copula_stabilizer_use_ebh': self.copula_stabilizer_use_ebh,
                'copula_stabilizer_seed_stride': self.copula_stabilizer_seed_stride,
                'prefilter_mi_weight': float(self.prefilter_mi_weight),
                'prefilter_f_weight': float(self.prefilter_f_weight),
                'prefilter_union_enabled': bool(self.prefilter_union_enabled),
                'prefilter_strategies': list(self.prefilter_strategies),
                'prefilter_nondefault_budget_fraction': float(self.prefilter_nondefault_budget_fraction),
                'prefilter_wsnr_enabled': bool(getattr(self, "prefilter_wsnr_enabled", False)),
                'prefilter_wsnr_stabilize_counts': bool(
                    getattr(self, "prefilter_wsnr_stabilize_counts", True)
                ),
                'prefilter_data_domain': str(getattr(self, "prefilter_data_domain", "auto")),
                'prefilter_rnaseq_transform_enabled': bool(
                    getattr(self, "prefilter_rnaseq_transform_enabled", True)
                ),
                'prefilter_rnaseq_transform_force': bool(
                    getattr(self, "prefilter_rnaseq_transform_force", False)
                ),
                'prefilter_rnaseq_nb_lrt_enabled': bool(
                    getattr(self, "prefilter_rnaseq_nb_lrt_enabled", False)
                ),
                'prefilter_rnaseq_nb_lrt_alpha': float(
                    getattr(self, "prefilter_rnaseq_nb_lrt_alpha", 0.10)
                ),
                'screening_enabled': bool(self.screening_enabled),
                'screening_method': str(self.screening_method),
                'screening_pool_cap': int(self.screening_pool_cap),
                'screening_stir_n_neighbors': int(self.screening_stir_n_neighbors),
                'screening_stir_n_iter': int(self.screening_stir_n_iter),
                'screening_stir_keep_fraction': float(self.screening_stir_keep_fraction),
                'screening_stir_min_features': int(self.screening_stir_min_features),
                'screening_evalue_alpha': float(self.screening_evalue_alpha),
                'screening_evalue_min_features': int(self.screening_evalue_min_features),
                'importance_uq_enabled': bool(self.importance_uq_enabled),
                'importance_uq_min_cv_folds': int(self.importance_uq_min_cv_folds),
                'importance_uq_computed': bool(importance_uq_meta.get('importance_uq_computed', False)),
                'importance_uq_reason': str(importance_uq_meta.get('importance_uq_reason', 'disabled')),
                'importance_uq_n_folds': int(importance_uq_meta.get('importance_uq_n_folds', 0) or 0),
                'decorrelated_stability_eps': self.decorrelated_stability_eps,
                'enabled_methods': sorted(self.enabled_methods) if self.enabled_methods is not None else None,
            }
        )
        
        self.selection_result_ = result
        self.selected_features_indices_ = selected_indices_original
        self.feature_scores_ = result.selected_feature_votes
        
        # Create visualization
        self._plot_feature_importance(feature_votes, selected_indices_local)

        logger.info("Feature selection completed: selected %d/%d features",
                   len(selected_indices_original), n_features)

        if self.selection_strategy == 'mnpo_portfolio' and self.mnpo_diagnostics_:
            logger.debug("MNPO portfolio summary:")
            portfolio = self.mnpo_diagnostics_.get('portfolio_weights', {})
            oracle_weights = self.mnpo_diagnostics_.get('oracle_weights', {})
            logger.debug("  Portfolio candidates: %s", self.mnpo_diagnostics_.get('portfolio_candidates', []))
            logger.debug("  Portfolio weights: %s", portfolio)
            logger.debug("  Oracle trust weights: %s", oracle_weights)
        else:
            logger.debug("Method weights used in voting:")
            method_weights = get_method_weights()
            for method, weight in method_weights.items():
                if method in method_results:
                    logger.debug("  %s: %sx weight", method, weight)
        
        # Transform data
        X_selected = X[:, selected_indices_original]
        
        if return_result_object:
            return X_selected, result
        else:
            return X_selected

    def transform(self, X):
        """Transform new data using selected features."""
        # Defensive reset for repeated inference-only calls on long-lived instances.
        self._eval_multimodel_fold_log = []
        if self.selected_features_indices_ is None:
            raise RuntimeError("Selector has not been fitted yet.")
        if not isinstance(X, np.ndarray): 
            X = np.array(X)
        return X[:, self.selected_features_indices_]

    def get_selected_features_indices(self):
        """Get indices of selected features."""
        return self.selected_features_indices_

    def get_feature_scores(self):
        """Get feature scores (votes)."""
        return self.feature_scores_

    def get_feature_importance_plot(self):
        """Get feature importance plot."""
        if self._feature_importance_plot_payload is None:
            return None
        votes, selected = self._feature_importance_plot_payload
        fig = self._build_feature_importance_figure(votes, selected)
        self.feature_importance_plot_ = fig
        return fig

    def get_selection_result(self):
        """Get comprehensive selection result object."""
        return self.selection_result_

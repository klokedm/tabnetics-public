"""Configuration dataclasses for FeatureSelector.

Provides a structured, grouped API for the ~288 parameters that
``FeatureSelector.__init__`` currently accepts as flat keyword arguments.

Usage::

    from tabnetics.feature_selection.config import FeatureSelectorConfig, MNPOConfig
    cfg = FeatureSelectorConfig(
        random_state=0,
        mnpo=MNPOConfig(portfolio_size=8),
    )
    fs = FeatureSelector.from_config(cfg)

All default values are kept in exact sync with ``FeatureSelector.__init__``
so that ``FeatureSelector.from_config(FeatureSelectorConfig())`` produces
an object identical to ``FeatureSelector()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Optional, Set, Tuple
import warnings


# T-R-243: default penalty map for weak selectors.
DEFAULT_SELECTOR_PENALTY_MAP: Dict[str, float] = {
    "treeshap": 0.0,
    "dove_class_specific": 0.0,
    "sparse_multinomial": 0.0,
}

# ── MNPO / Portfolio ──────────────────────────────────────────────────
@dataclass
class OracleConfig:
    """MNPO oracle controls (Phase 6A, T-R-180).

    Note (R9 mitigation): aggregation-layer alternatives (including Shapley)
    must be implemented via ``weighting_mode``. Separate parallel Shapley
    pathways are not permitted.
    """

    pairwise_delta: float = 0.01
    use_tritrust: bool = True
    use_stability_oracle: bool = True
    use_complexity_oracle: bool = True
    use_robust_oracle: bool = True
    use_diversity_oracle: bool = True
    use_cvar: bool = False
    cvar_alpha: float = 0.33
    use_qre_smoothing: bool = False
    qre_temperature_gamma: float = 1.0
    fold_preference_mode: str = "vote"  # vote | logistic
    use_oracle_redundancy_penalty: bool = False
    compute_tremble_sensitivity: bool = False
    diversity_mode: str = "legacy_jaccard"  # legacy_jaccard | mi_redundancy | pid_mi | complementarity
    diversity_redundancy_weight: float = 0.60
    diversity_complementarity_weight: float = 0.35
    performance_oracle_mode: str = "single"  # single | multi_model_oracles
    weighting_mode: str = "tritrust"  # tritrust | uniform | shapley
    shapley_n_coalitions_max: int = 4096
    # T-R-179: continuous Bayesian shrinkage for Shapley weighting mode.
    shapley_bayesian_shrinkage: bool = False
    shapley_bayesian_prior_strength: float = 8.0
    # T-R-142: interaction-density oracle (hard-gated at n_train >= 150).
    use_interaction_oracle: bool = False
    interaction_oracle_min_n_train: int = 150
    interaction_oracle_pool_size_cap: int = 64
    interaction_oracle_pair_cap: int = 20000
    use_ubayfs: bool = False
    ubayfs_n_bootstrap: int = 32
    ubayfs_min_n: int = 100
    ubayfs_prior_weight: float = 0.0
    use_conformal_uq: bool = False
    conformal_uq_alpha: float = 0.10
    conformal_uq_min_folds: int = 5
    use_conformal_efficiency: bool = False
    conformal_efficiency_method: str = "split"  # split | aps
    oracle_weight_js_shrinkage: bool = False
    complexity_conditioning: bool = False
    # DIAKRINO relevance oracle (native integration §2.2; opt-in, default off).
    # Scores how much chunk-calibrated DIAKRINO per-feature relevance mass each candidate
    # method captured, biasing method TRUST without ever selecting features.  Consumes
    # one MNPO_ORACLE_COUNT_CAP slot -> ship only inside a reduced-oracle preset.
    use_diakrino_relevance_oracle: bool = False
    diakrino_relevance_min_n_train: int = 100
    diakrino_relevance_score_column: str = "prior_logit"  # calibration-safe surface
    use_diakrino_selector_prior: bool = False
    diakrino_selector_prior_weight: float = 1.0
    diakrino_selector_prior_calibration: str = "current_checkpoint_20260628"
    diakrino_selector_prior_qualification_record: str = ""

    def __post_init__(self) -> None:
        mode = str(self.diversity_mode).strip().lower() if self.diversity_mode is not None else "legacy_jaccard"
        if mode not in {"legacy_jaccard", "mi_redundancy", "pid_mi", "complementarity"}:
            mode = "legacy_jaccard"
        self.diversity_mode = mode

        pom = str(self.performance_oracle_mode).strip().lower() if self.performance_oracle_mode is not None else "single"
        if pom not in {"single", "multi_model_oracles"}:
            pom = "single"
        self.performance_oracle_mode = pom

        wm = str(self.weighting_mode).strip().lower() if self.weighting_mode is not None else "tritrust"
        if wm not in {"tritrust", "uniform", "shapley", "banzhaf"}:
            wm = "tritrust"
        self.weighting_mode = wm
        fpm = str(self.fold_preference_mode).strip().lower() if self.fold_preference_mode is not None else "vote"
        if fpm not in {"vote", "logistic"}:
            fpm = "vote"
        self.fold_preference_mode = fpm
        cem = (
            str(self.conformal_efficiency_method).strip().lower()
            if self.conformal_efficiency_method is not None
            else "split"
        )
        if cem not in {"split", "aps"}:
            cem = "split"
        self.conformal_efficiency_method = cem

        self.pairwise_delta = float(max(0.0, self.pairwise_delta))
        self.cvar_alpha = float(min(1.0, max(0.01, self.cvar_alpha)))
        self.qre_temperature_gamma = float(max(1e-6, self.qre_temperature_gamma))
        self.diversity_redundancy_weight = float(min(2.0, max(0.0, self.diversity_redundancy_weight)))
        self.diversity_complementarity_weight = float(min(2.0, max(0.0, self.diversity_complementarity_weight)))
        self.shapley_n_coalitions_max = int(max(2, self.shapley_n_coalitions_max))
        self.shapley_bayesian_prior_strength = float(
            max(1e-6, self.shapley_bayesian_prior_strength)
        )
        self.interaction_oracle_min_n_train = int(max(2, self.interaction_oracle_min_n_train))
        self.interaction_oracle_pool_size_cap = int(max(4, self.interaction_oracle_pool_size_cap))
        self.interaction_oracle_pair_cap = int(max(1, self.interaction_oracle_pair_cap))
        self.ubayfs_n_bootstrap = int(max(1, self.ubayfs_n_bootstrap))
        self.ubayfs_min_n = int(max(1, self.ubayfs_min_n))
        self.ubayfs_prior_weight = float(min(1.0, max(0.0, self.ubayfs_prior_weight)))
        self.conformal_uq_alpha = float(min(0.49, max(1e-3, self.conformal_uq_alpha)))
        self.conformal_uq_min_folds = int(max(2, self.conformal_uq_min_folds))
        self.diakrino_selector_prior_weight = float(
            min(1.0, max(0.0, self.diakrino_selector_prior_weight))
        )
        mode = (
            str(self.diakrino_selector_prior_calibration).strip().lower()
            if self.diakrino_selector_prior_calibration is not None
            else "current_checkpoint_20260628"
        )
        if mode in {"raw", "off"}:
            mode = "none"
        if mode not in {"current_checkpoint_20260628", "none"}:
            mode = "current_checkpoint_20260628"
        self.diakrino_selector_prior_calibration = mode
        self.diakrino_selector_prior_qualification_record = str(
            self.diakrino_selector_prior_qualification_record or ""
        )

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "OracleConfig":
        """Create an OracleConfig from a named preset.

        Presets (T-R-248 oracle pruning experiment):
          - 'perf_only': Performance oracle only (1 oracle).
          - 'perf_complexity': Performance + complexity (2 oracles).
          - 'perf_complexity_stability': Performance + complexity + stability (3 oracles).
          - 'full': All default oracles enabled (5 oracles, current default).
          - 'minimal_cvar': Performance + CVaR only (2 oracles, tail-risk focus).
        """
        presets = {
            "perf_only": dict(
                use_stability_oracle=False,
                use_complexity_oracle=False,
                use_robust_oracle=False,
                use_diversity_oracle=False,
            ),
            "perf_complexity": dict(
                use_stability_oracle=False,
                use_robust_oracle=False,
                use_diversity_oracle=False,
            ),
            "perf_complexity_stability": dict(
                use_robust_oracle=False,
                use_diversity_oracle=False,
            ),
            "full": dict(),  # All defaults.
            "minimal_cvar": dict(
                use_stability_oracle=False,
                use_complexity_oracle=False,
                use_robust_oracle=False,
                use_diversity_oracle=False,
                use_cvar=True,
                cvar_alpha=0.33,
            ),
        }
        preset_key = str(preset).strip().lower()
        if preset_key not in presets:
            raise ValueError(
                f"Unknown oracle preset {preset!r}. "
                f"Available: {sorted(presets.keys())}"
            )
        kwargs = dict(presets[preset_key])
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class MNPOConfig:
    """Mirror-descent Nash portfolio optimisation parameters."""

    inner_cv_splits: int = 5
    inner_cv_repeats: int = 3
    mirror_descent_steps: int = 300
    mirror_descent_eta: float = 0.15
    mirror_descent_lambda: float = 0.08
    portfolio_size: int = 6
    portfolio_size_guard: str = "none"
    oracle: OracleConfig = field(default_factory=OracleConfig)
    # Deprecated flat aliases (kept for compatibility, mapped into oracle.*).
    pairwise_delta: float = 0.01
    use_tritrust: bool = True
    use_stability_oracle: bool = True
    use_complexity_oracle: bool = True
    use_robust_oracle: bool = True
    use_diversity_oracle: bool = True
    disable_redundancy_penalty_binary: bool = True
    disable_class_pareto_binary: bool = True
    # Deprecated (T-DS3): retained for config compatibility, ignored at runtime.
    use_tail_risk_oracle: bool = False
    tail_risk_alpha: float = 0.33
    diversity_oracle_mode: str = "legacy_jaccard"
    diversity_redundancy_weight: float = 0.60
    diversity_complementarity_weight: float = 0.35
    rank_aggregation_mode: str = "none"
    complexity_use_runtime_penalty: bool = False
    use_qre_smoothing: bool = False
    qre_temperature_gamma: float = 1.0
    use_oracle_redundancy_penalty: bool = False
    compute_tremble_sensitivity: bool = False
    performance_oracle_mode: str = "single"
    oracle_weighting_mode: str = "tritrust"
    shapley_n_coalitions_max: int = 4096
    shapley_bayesian_shrinkage: bool = False
    shapley_bayesian_prior_strength: float = 8.0
    fold_preference_mode: str = "vote"
    use_interaction_oracle: bool = False
    interaction_oracle_min_n_train: int = 150
    interaction_oracle_pool_size_cap: int = 64
    interaction_oracle_pair_cap: int = 20000
    # DIAKRINO relevance oracle (native integration §2.2; opt-in, default off).
    use_diakrino_relevance_oracle: bool = False
    diakrino_relevance_min_n_train: int = 100
    diakrino_relevance_score_column: str = "prior_logit"
    use_diakrino_selector_prior: bool = False
    diakrino_selector_prior_weight: float = 1.0
    diakrino_selector_prior_calibration: str = "current_checkpoint_20260628"
    diakrino_selector_prior_qualification_record: str = ""
    use_cvar: bool = False
    cvar_alpha: float = 0.33
    use_ubayfs: bool = False
    ubayfs_n_bootstrap: int = 32
    ubayfs_min_n: int = 100
    ubayfs_prior_weight: float = 0.0
    use_conformal_efficiency: bool = False
    conformal_efficiency_method: str = "split"
    oracle_weight_js_shrinkage: bool = False
    complexity_conditioning: bool = False
    payoff_shrinkage_kappa: float = 0.0
    mnpo_include_legacy_consensus: bool = True
    mnpo_include_majority_consensus: bool = True
    adaptive_portfolio_sizing_enabled: bool = False
    adaptive_size_min: Optional[int] = None
    adaptive_size_max: Optional[int] = None
    # T-R-179: penalize noisy candidate-weight distributions in adaptive sizing.
    adaptive_sizing_variance_penalty: bool = False
    adaptive_sizing_variance_penalty_strength: float = 0.5
    mnpo_paradigm_aware_prior_enabled: bool = False
    mnpo_interaction_floor: float = 0.12
    rashomon_enabled: bool = False
    rashomon_max_models: int = 12
    rashomon_score_tolerance: float = 0.01
    # T-R-243: penalty map for weak selectors (0.0=exclude, <1.0=downweight).
    selector_penalty_map: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.oracle is None:
            self.oracle = OracleConfig()

        alias_map = {
            "pairwise_delta": "pairwise_delta",
            "use_tritrust": "use_tritrust",
            "use_stability_oracle": "use_stability_oracle",
            "use_complexity_oracle": "use_complexity_oracle",
            "use_robust_oracle": "use_robust_oracle",
            "use_diversity_oracle": "use_diversity_oracle",
            "diversity_oracle_mode": "diversity_mode",
            "diversity_redundancy_weight": "diversity_redundancy_weight",
            "diversity_complementarity_weight": "diversity_complementarity_weight",
            "use_qre_smoothing": "use_qre_smoothing",
            "qre_temperature_gamma": "qre_temperature_gamma",
            "use_oracle_redundancy_penalty": "use_oracle_redundancy_penalty",
            "compute_tremble_sensitivity": "compute_tremble_sensitivity",
            "performance_oracle_mode": "performance_oracle_mode",
            "oracle_weighting_mode": "weighting_mode",
            "shapley_n_coalitions_max": "shapley_n_coalitions_max",
            "shapley_bayesian_shrinkage": "shapley_bayesian_shrinkage",
            "shapley_bayesian_prior_strength": "shapley_bayesian_prior_strength",
            "fold_preference_mode": "fold_preference_mode",
            "use_interaction_oracle": "use_interaction_oracle",
            "interaction_oracle_min_n_train": "interaction_oracle_min_n_train",
            "interaction_oracle_pool_size_cap": "interaction_oracle_pool_size_cap",
            "interaction_oracle_pair_cap": "interaction_oracle_pair_cap",
            "use_cvar": "use_cvar",
            "cvar_alpha": "cvar_alpha",
            "use_ubayfs": "use_ubayfs",
            "ubayfs_n_bootstrap": "ubayfs_n_bootstrap",
            "ubayfs_min_n": "ubayfs_min_n",
            "ubayfs_prior_weight": "ubayfs_prior_weight",
            "use_conformal_efficiency": "use_conformal_efficiency",
            "conformal_efficiency_method": "conformal_efficiency_method",
            "oracle_weight_js_shrinkage": "oracle_weight_js_shrinkage",
            "complexity_conditioning": "complexity_conditioning",
            "use_diakrino_selector_prior": "use_diakrino_selector_prior",
            "diakrino_selector_prior_weight": "diakrino_selector_prior_weight",
            "diakrino_selector_prior_calibration": "diakrino_selector_prior_calibration",
            "diakrino_selector_prior_qualification_record": "diakrino_selector_prior_qualification_record",
        }
        default_oracle = OracleConfig()
        for legacy_name, oracle_name in alias_map.items():
            legacy_val = getattr(self, legacy_name)
            default_val = getattr(default_oracle, oracle_name)
            if legacy_val != default_val:
                warnings.warn(
                    f"MNPOConfig.{legacy_name} is deprecated; use MNPOConfig.oracle.{oracle_name}.",
                    DeprecationWarning,
                )
                setattr(self.oracle, oracle_name, legacy_val)

        # Deprecated in runtime path (kept for config compatibility only).
        if bool(self.use_tail_risk_oracle):
            warnings.warn(
                "MNPOConfig.use_tail_risk_oracle is deprecated and ignored at runtime.",
                DeprecationWarning,
            )
        if float(self.tail_risk_alpha) != 0.33:
            warnings.warn(
                "MNPOConfig.tail_risk_alpha is deprecated and ignored unless use_tail_risk_oracle is set.",
                DeprecationWarning,
            )
        self.adaptive_sizing_variance_penalty_strength = float(
            max(0.0, self.adaptive_sizing_variance_penalty_strength)
        )
        self.payoff_shrinkage_kappa = float(max(0.0, getattr(self, "payoff_shrinkage_kappa", 0.0) or 0.0))
        if self.selector_penalty_map is None:
            self.selector_penalty_map = dict(DEFAULT_SELECTOR_PENALTY_MAP)
        # Re-sanitize nested oracle after legacy alias overrides.
        if hasattr(self.oracle, "__post_init__"):
            self.oracle.__post_init__()


# ── Stability Selection ──────────────────────────────────────────────
@dataclass
class StabilityConfig:
    """Stability selection, cluster-stability, decorrelated-stability,
    and IPSS parameters."""

    # Core stability
    stability_subsample_fraction: float = 0.5
    stability_selection_threshold: float = 0.6
    stability_threshold_method: str = "fixed"
    stability_target_pfer: float = 1.0
    stability_use_loss_guided_validation: bool = False
    stability_validation_fraction: float = 0.25
    stability_validation_quantile: float = 0.40
    stability_validation_min_samples: int = 6

    # Cluster stability
    cluster_stability_corr_threshold: float = 0.85
    cluster_stability_max_per_cluster: int = 2
    cluster_stability_min_cluster_freq: float = 0.55

    # Decorrelated stability
    decorrelated_stability_eps: float = 1e-3
    decorrelated_stability_min_max_abs_corr: float = 0.0

    # IPSS (integrated path stability selection)
    ipss_path_grid_size: int = 7
    ipss_min_c: float = 0.08
    ipss_max_c: float = 1.20
    ipss_target_fdr: float = 0.15
    ipss_use_eats_threshold: bool = False
    ipss_importance_model: str = "linear_svm"
    ipss_gate_min_classes: int = 0
    ipss_gate_min_p_over_n: float = 0.0


# ── Wrapper / Iterative Pruning ───────────────────────────────────────
@dataclass
class WrapperConfig:
    """Wrapper refinement and iterative-pruning parameters."""

    # Greedy wrapper refinement
    wrapper_refine_enabled: bool = False
    wrapper_refine_top_k: int = 24
    wrapper_refine_max_add: int = 12
    wrapper_refine_min_gain: float = 1e-4

    # Iterative pruning (unbounded)
    iterative_pruning_pool_factor: float = 2.5
    iterative_pruning_max_rounds: int = 32
    iterative_pruning_min_improvement: float = -0.002
    iterative_pruning_max_cumulative_loss: float = 0.02
    iterative_pruning_redundancy_weight: float = 0.65

    # Iterative pruning (bounded / runtime-capped)
    iterative_pruning_bounded_prefilter_cap: int = 220
    iterative_pruning_bounded_max_evaluations: int = 48
    iterative_pruning_bounded_max_runtime_seconds: float = 30.0
    iterative_pruning_bounded_use_cpss_overlay: bool = False
    iterative_pruning_bounded_enable_class_gating: bool = True


# ── Multiclass ────────────────────────────────────────────────────────
@dataclass
class MulticlassConfig:
    """One-vs-All, ECOC, NSC, and class-Pareto multiclass parameters."""

    # OVA
    ova_negative_ratio: float = 2.0
    ova_min_classes: int = 3
    ova_class_weight_mode: str = "uniform"
    ova_aggregation_mode: str = "mean"

    # ECOC
    ecoc_min_classes: int = 4

    # NSC (nearest shrunken centroids)
    nsc_min_classes: int = 3
    nsc_thresholding_mode: str = "soft"
    nsc_shrinkage_grid_size: int = 6

    # Class-specific Pareto
    class_pareto_min_classes: int = 3
    class_pareto_top_per_class: int = 64


# ── Copula Knockoff ───────────────────────────────────────────────────
@dataclass
class CopulaConfig:
    """D-vine copula knockoff (DTDCKe) parameters."""

    copula_knockoff_draws: int = 30
    copula_alpha_kn: float = 0.1
    copula_alpha_ebh: float = 0.2
    copula_truncation_level: Optional[int] = 5
    copula_generator: str = "copula"  # "copula" | "deepdrk"
    copula_deepdrk_latent_fraction: float = 0.35
    copula_deepdrk_noise_scale: float = 1.0
    # T-R-110: derandomized knockoffs (default=1 preserves legacy single-run behavior).
    copula_derandomize_runs: int = 1
    copula_stabilizer_runs: int = 1
    # Wall-clock budget for the full copula knockoff method (across all
    # derandomize runs and inner draws). 0/None disables. The selector caps
    # ``trunc_lvl`` independently — this budget is a defence-in-depth stop
    # for unexpected blow-ups (e.g. LassoCV non-convergence in HDLSS).
    copula_time_budget_seconds: float = 1800.0
    copula_vine_num_threads: Optional[int] = None
    copula_lasso_max_iter: int = 5000
    copula_lasso_n_jobs: int = 1


# ── Prefilter blend ───────────────────────────────────────────────────
@dataclass
class PrefilterConfig:
    """Prefilter blend configuration for feature pool reduction.

    Controls the MI/F-test relevance blend used by
    ``pareto_prefilter_stability_support()`` and
    ``class_dominance_pareto_prefilter()`` in ``prefilter.py``.

    Default values reproduce the original hard-coded 60/40 MI/F-test
    blend exactly (bit-for-bit).

    Note
    ----
    The *binary_class_prefilter_scores()* function uses a **different**
    blend (45/35/20 base ratios with optional KW).  That blend is
    intentionally **not** covered here and will be configurable in a
    future follow-up.
    """

    enabled: bool = True
    strategy: str = "blend_v1"  # Default: current 60/40 MI/F-test blend
    mi_weight: float = 0.60
    f_weight: float = 0.40
    # T-R-127: optional multi-strategy union pool.
    union_enabled: bool = False
    strategies: Tuple[str, ...] = ("mi_ftest_blend",)
    nondefault_budget_fraction: float = 0.10
    wsnr_enabled: bool = False
    wsnr_stabilize_counts: bool = True
    data_domain: str = "auto"  # auto | rnaseq | generic
    rnaseq_transform_enabled: bool = True
    rnaseq_transform_force: bool = False
    rnaseq_nb_lrt_enabled: bool = False
    rnaseq_nb_lrt_alpha: float = 0.10
    # Future: effect_size_weight, kw_rank_weight, stir_*, rf_*


# ── Tier 2 screening ──────────────────────────────────────────────────
@dataclass
class ScreeningConfig:
    """Tier 2 interaction-aware screening configuration.

    When ``enabled`` is *True* and ``method`` is set (e.g. ``"stir"``),
    a screening pass runs **after** Tier 1 prefilter to further reduce
    the feature pool using interaction-aware scoring (STIR / ReliefF).

    See ArchitectureRefactor.md §14, RISK-2 for boundary rationale.
    """

    enabled: bool = False
    method: str = "none"           # "stir" | "evalue" | "none"
    pool_cap: int = 2000           # Max features to screen (safety cap)
    runtime_budget_sec: float = 60.0  # Reserved for future wall-clock timeout
    # STIR parameters
    stir_n_neighbors: int = 10
    stir_n_iter: int = 50
    stir_keep_fraction: float = 0.5   # Keep top 50 % of features
    stir_min_features: int = 20       # Never reduce below this
    # E-value screening (T-R-111)
    evalue_alpha: float = 0.20
    evalue_min_features: int = 20


# ── Multi-classifier evaluation proxy ─────────────────────────────────
@dataclass
class EvaluationConfig:
    """Multi-classifier evaluation proxy configuration.

    When ``eval_models_enabled`` is *False* (default), the evaluation
    pipeline uses the existing single-LR fold scorer (bit-for-bit
    identical to baseline).  When *True*, multiple classifiers are
    fitted independently on each training fold and scored on the
    validation fold; their scores are combined with a **fixed-weight**
    aggregation (mean, min, or CVaR).  No learned/adaptive weights
    are permitted in Stage 1 (see ArchitectureRefactor.md §14, RISK-1).
    """

    eval_models_enabled: bool = False  # OFF by default → baseline behavior
    eval_models: tuple = ("lr_l2", "linear_svc", "rf_small")
    eval_aggregate: str = "mean"       # "mean", "min", "cvar"
    eval_cvar_alpha: float = 0.33      # CVaR tail fraction (used when eval_aggregate="cvar")
    eval_failure_strict_mode: bool = False  # True -> raise model eval errors instead of fallback score=0
    eval_model_weight_strategy: str = "fixed"  # "fixed" (allowed), "learned" (blocked in eval folds)


# ── Candidate-method specifics ────────────────────────────────────────
@dataclass
class MethodConfig:
    """Per-method hyper-parameters for mRMR, k-TSP, and HSIC Lasso."""

    mrmr_max_features: int = 320
    mrmr_redundancy_weight: float = 0.55
    mrmr_mi_redundancy_enabled: bool = False
    mrmr_mi_n_bins: int = 8
    # Zero keeps the legacy unlimited selector.  Any positive value is an
    # opt-in hard cap for ordered redundancy-pair computations.
    mrmr_max_unique_pair_evaluations: int = 0
    mrmr_max_runtime_seconds: float = 0.0
    # Default fails closed on a budget breach.  Relevance-only fill is opt-in.
    mrmr_budget_fallback_mode: str = "empty"
    cmim_min_samples: int = 60
    cmim_n_bins: int = 8
    fcbf_n_bins: int = 8
    ktsp_max_features: int = 120
    ktsp_k_pairs: int = 24
    hsic_lasso_alpha: float = 0.01
    hsic_lasso_prefilter_max_features: int = 128
    hsic_lasso_binary_delta_enabled: bool = True
    hsic_lasso_binary_delta_min_samples: int = 30
    slce_prefilter_max_features: int = 1024
    slce_min_samples: int = 30
    slce_ridge: float = 1.0
    # T-R-158: TreeSHAP embedded selector (opt-in via enabled_methods).
    treeshap_min_samples: int = 50
    treeshap_n_estimators: int = 200
    treeshap_multi_seed_runs: int = 3
    # T-R-161: OAENet adaptive elastic-net selector (opt-in).
    oaenet_min_samples: int = 40
    oaenet_prescreen_max_features: int = 512
    oaenet_l1_ratio: float = 0.5
    oaenet_c_grid_size: int = 6
    # T-R-166: classical SDR method controls (SIR/SAVE/PFC).
    sdr_min_classes: int = 3
    sdr_prefilter_max_features: int = 512
    sdr_n_components: int = 3
    sdr_covariance_ridge: float = 1e-3
    # DIAKRINO candidate methods (native integration §2.1; opt-in via enabled_methods only —
    # both methods are default_enabled=False in the registry).  Read a persisted sidecar
    # parquet; degrade to a graceful skip when absent.  Calibration-safe columns only.
    diakrino_prior_sidecar_path: str = ""          # dir or feature_logits parquet; "" = disabled/skip
    diakrino_prior_dataset_id: str = ""            # optional dataset id for sidecar roots/manifests
    diakrino_prior_score_column: str = "prior_logit"
    diakrino_screening_score_column: str = "screening_logit"
    diakrino_prior_calibrate: str = "chunk_zscore"  # none | chunk_* modes | blend
    diakrino_prior_top_k: int = 0                    # 0 => use the selector's default n_target
    # DIAKRINO schema-v2 conformal selection scaffold (T-DIAKRINO-NAT-10).
    # Requires enabled_methods={"diakrino_conformal_selection", ...} AND this toggle.
    # Diagnostics intentionally mirror replay outputs: normalize/calibration/z-score.
    diakrino_conformal_selection_enabled: bool = False
    diakrino_conformal_target_fdp: float = 0.20
    diakrino_conformal_calibrate: str = "chunk_zscore"
    diakrino_conformal_null_fraction: float = 0.50
    diakrino_conformal_min_null_scores: int = 4
    diakrino_conformal_max_features: int = 0
    diakrino_conformal_qualification_record: str = ""

    def __post_init__(self) -> None:
        try:
            self.mrmr_max_unique_pair_evaluations = int(
                self.mrmr_max_unique_pair_evaluations
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "mrmr_max_unique_pair_evaluations must be an integer >= 0"
            ) from exc
        if self.mrmr_max_unique_pair_evaluations < 0:
            raise ValueError(
                "mrmr_max_unique_pair_evaluations must be >= 0 (0 means unlimited)"
            )
        try:
            self.mrmr_max_runtime_seconds = float(self.mrmr_max_runtime_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "mrmr_max_runtime_seconds must be a finite float >= 0"
            ) from exc
        if (
            not math.isfinite(self.mrmr_max_runtime_seconds)
            or self.mrmr_max_runtime_seconds < 0.0
        ):
            raise ValueError(
                "mrmr_max_runtime_seconds must be a finite float >= 0 (0 means unlimited)"
            )
        self.mrmr_budget_fallback_mode = str(
            self.mrmr_budget_fallback_mode or "empty"
        ).strip().lower()
        if self.mrmr_budget_fallback_mode not in {"empty", "relevance_only"}:
            raise ValueError(
                "mrmr_budget_fallback_mode must be 'empty' or 'relevance_only'"
            )
        self.diakrino_prior_sidecar_path = str(self.diakrino_prior_sidecar_path or "")
        self.diakrino_prior_dataset_id = str(self.diakrino_prior_dataset_id or "")
        self.diakrino_prior_score_column = str(self.diakrino_prior_score_column or "prior_logit")
        self.diakrino_screening_score_column = str(self.diakrino_screening_score_column or "screening_logit")
        self.diakrino_prior_calibrate = str(self.diakrino_prior_calibrate or "chunk_zscore").strip().lower()
        self.diakrino_prior_top_k = int(max(0, int(self.diakrino_prior_top_k or 0)))
        self.diakrino_conformal_selection_enabled = bool(self.diakrino_conformal_selection_enabled)
        self.diakrino_conformal_target_fdp = float(min(1.0, max(0.0, self.diakrino_conformal_target_fdp)))
        self.diakrino_conformal_calibrate = str(self.diakrino_conformal_calibrate or "chunk_zscore").strip().lower()
        self.diakrino_conformal_null_fraction = float(
            min(1.0, max(0.0, self.diakrino_conformal_null_fraction))
        )
        self.diakrino_conformal_min_null_scores = int(max(1, int(self.diakrino_conformal_min_null_scores or 1)))
        self.diakrino_conformal_max_features = int(max(0, int(self.diakrino_conformal_max_features or 0)))
        self.diakrino_conformal_qualification_record = str(self.diakrino_conformal_qualification_record or "")


# ── Top-level ─────────────────────────────────────────────────────────
@dataclass
class FeatureSelectorConfig:
    """Top-level configuration for ``FeatureSelector``.

    General / core parameters live directly on this dataclass.
    Domain-specific groups are nested as sub-configs (``mnpo``,
    ``stability``, ``wrapper``, ``multiclass``, ``copula``, ``methods``).

    Parameters not covered here fall back to their ``__init__`` defaults
    when constructed via ``FeatureSelector.from_config()``.
    """

    # ── General / core ────────────────────────────────────────────────
    n_folds: int = 5
    n_bootstrap_iterations: int = 10
    random_state: int = 42
    problem_type: str = "classification"
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.90
    use_pca: bool = False
    n_components: float = 0.95
    selection_strategy: str = "mnpo_portfolio"
    enabled_methods: Optional[Set[str]] = None
    method_timeout_seconds: float = 0.0
    method_max_rss_mb: float = 0.0
    parallel_n_jobs: int = 1
    linear_svm_max_iter: int = 10000

    # ── Nested sub-configs ────────────────────────────────────────────
    mnpo: MNPOConfig = field(default_factory=MNPOConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    wrapper: WrapperConfig = field(default_factory=WrapperConfig)
    multiclass: MulticlassConfig = field(default_factory=MulticlassConfig)
    copula: CopulaConfig = field(default_factory=CopulaConfig)
    methods: MethodConfig = field(default_factory=MethodConfig)
    prefilter: PrefilterConfig = field(default_factory=PrefilterConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self):
        """Validate top-level config invariants early."""
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if self.n_bootstrap_iterations < 1:
            raise ValueError(
                f"n_bootstrap_iterations must be >= 1, got {self.n_bootstrap_iterations}"
            )
        if not (0.0 <= self.variance_threshold <= 1.0):
            raise ValueError(
                f"variance_threshold must be in [0, 1], got {self.variance_threshold}"
            )
        if not (0.0 < self.correlation_threshold <= 1.0):
            raise ValueError(
                f"correlation_threshold must be in (0, 1], got {self.correlation_threshold}"
            )
        if self.problem_type not in ("classification", "regression"):
            raise ValueError(
                f"problem_type must be 'classification' or 'regression', got {self.problem_type!r}"
            )
        try:
            self.method_max_rss_mb = float(self.method_max_rss_mb)
        except (TypeError, ValueError) as exc:
            raise ValueError("method_max_rss_mb must be a finite float >= 0") from exc
        if (
            not math.isfinite(self.method_max_rss_mb)
            or self.method_max_rss_mb < 0.0
        ):
            raise ValueError("method_max_rss_mb must be a finite float >= 0")
        self.random_state = int(self.random_state) % (2**32)

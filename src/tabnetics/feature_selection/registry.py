"""Canonical registry of all feature selection methods.

Single source of truth for method keys, labels, weights, and metadata.
Replaces the triple-duplicated definitions previously in feature_selector.py.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MethodSpec:
    """Specification for a single feature selection method."""
    key: str                          # Internal key, e.g., 'stability_lasso'
    label: str                        # Display label
    fn_name: str                      # Method name on FeatureSelector
    legacy_weight: float = 1.0        # Weight for legacy voting
    maturity: str = 'stable'          # 'experimental', 'stable', 'deprecated'
    paradigm: str = 'filter'          # method category
    min_classes: int = 2              # Minimum classes required
    binary_only: bool = False         # Only supports binary
    requires_multiclass: bool = False # Requires 3+ classes
    requires_gpu: bool = False        # Requires CUDA GPU
    structured_resampling: str = 'supported'  # supported | conditional | unsupported
    default_enabled: bool = True      # If False, skipped unless explicitly in enabled_methods
                                      # (hard opt-in gate independent of profile membership;
                                      #  enabled_methods=None means run-all for default_enabled methods)


def _build_registry() -> dict:
    """Build the canonical method registry.

    Every entry was reconciled from three previously-duplicated sources:
      1. _calculate_weighted_votes()  – weights
      2. _run_selection_methods()     – (key, label, fn_name) tuples
      3. fit_transform() display dict – weights (debug logging)
    plus the experimental_methods set in _mnpo_select_features().

    Reconciliation notes
    --------------------
    * 'k tsp' in the old weight dict was a typo for 'ktsp'; the weight (1.1)
      was never actually applied.  Fixed here as 'ktsp'.
    * 'wmw_auc' and 'joint_auc_l1' had weights in _calculate_weighted_votes()
      but were missing from the fit_transform() display dict.  Both are now
      included with their canonical weights.
    * 'copula_knockoff' had weight 4 (int) in fit_transform() and 4.0 (float)
      in _calculate_weighted_votes().  Canonicalised to 4.0.
    * GA/SVM-RFE was previously removed (2026-02-15) and is excluded.
    """
    specs = [
        # --- Stability methods ---
        MethodSpec(
            key='stability_lasso',
            label='Stability Selection (Lasso)',
            fn_name='_stability_selection_lasso',
            legacy_weight=1.0,
            maturity='stable',
            paradigm='stability',
        ),
        MethodSpec(
            key='stability_subsample',
            label='Stability Selection (complementary subsampling)',
            fn_name='_stability_subsample_selection',
            legacy_weight=1.2,
            maturity='experimental',
            paradigm='stability',
        ),
        MethodSpec(
            key='tigress_stability',
            label='TIGRESS-style Stability Selection',
            fn_name='_tigress_stability_selection',
            legacy_weight=1.2,
            maturity='experimental',
            paradigm='stability',
        ),
        MethodSpec(
            key='subspace_stability',
            label='Subspace Stability Selection (equivalent correlated subsets)',
            fn_name='_subspace_stability_selection',
            legacy_weight=1.2,
            maturity='experimental',
            paradigm='stability',
        ),
        MethodSpec(
            key='decorrelated_stability',
            label='Decorrelated Stability Selection',
            fn_name='_decorrelated_stability_selection',
            legacy_weight=1.25,
            maturity='stable',
            paradigm='stability',
        ),
        MethodSpec(
            key='ipss',
            label='Integrated Path Stability Selection (IPSS)',
            fn_name='_ipss_selection',
            legacy_weight=1.25,
            maturity='experimental',
            paradigm='stability',
        ),
        MethodSpec(
            key='cluster_stability',
            label='Cluster Stability Selection (correlation-aware)',
            fn_name='_cluster_stability_selection',
            legacy_weight=1.25,
            maturity='experimental',
            paradigm='stability',
        ),

        # --- Wrapper methods ---
        MethodSpec(
            key='rfecv',
            label='Recursive Feature Elimination',
            fn_name='_rfe_cv_selection',
            legacy_weight=1.0,
            maturity='stable',
            paradigm='wrapper',
        ),
        MethodSpec(
            key='boruta',
            label='Boruta',
            fn_name='_boruta_selection',
            legacy_weight=2.0,
            maturity='stable',
            paradigm='wrapper',
        ),

        # --- Embedded methods ---
        MethodSpec(
            key='gradient_boosting',
            label='Gradient Boosting',
            fn_name='_gradient_boosting_selection',
            legacy_weight=1.0,
            maturity='stable',
            paradigm='embedded',
        ),
        MethodSpec(
            key='linear_svm',
            label='Linear SVM',
            fn_name='_linear_svm_selection',
            legacy_weight=1.0,
            maturity='stable',
            paradigm='embedded',
        ),
        MethodSpec(
            key='treeshap',
            label='TreeSHAP embedded selector',
            fn_name='_treeshap_selection',
            legacy_weight=1.1,
            maturity='experimental',
            paradigm='embedded',
        ),
        MethodSpec(
            key='oaenet',
            label='OAENet adaptive elastic-net selector',
            fn_name='_oaenet_selection',
            legacy_weight=1.1,
            maturity='experimental',
            paradigm='embedded',
        ),

        # --- Filter methods ---
        MethodSpec(
            key='mutual_information',
            label='Mutual Information',
            fn_name='_mutual_information_selection',
            legacy_weight=1.0,
            maturity='stable',
            paradigm='filter',
        ),
        MethodSpec(
            key='anova_f',
            label='ANOVA F-test',
            fn_name='_anova_f_selection',
            legacy_weight=1.0,
            maturity='stable',
            paradigm='filter',
        ),
        MethodSpec(
            key='chi_square',
            label='Chi-Square univariate filter',
            fn_name='_chi_square_selection',
            legacy_weight=1.0,
            maturity='experimental',
            paradigm='filter',
        ),
        MethodSpec(
            key='relieff',
            label='ReliefF instance-based filter',
            fn_name='_relieff_selection',
            legacy_weight=1.0,
            maturity='experimental',
            paradigm='filter',
        ),
        MethodSpec(
            key='fcbf',
            label='FCBF correlation-based filter',
            fn_name='_fcbf_selection',
            legacy_weight=1.1,
            maturity='experimental',
            paradigm='filter',
        ),
        MethodSpec(
            key='cmim',
            label='CMIM conditional MI filter',
            fn_name='_cmim_selection',
            legacy_weight=1.25,
            maturity='experimental',
            paradigm='filter',
        ),

        # --- Pairwise / binary-aware methods ---
        MethodSpec(
            key='wmw_auc',
            label='WMW univariate AUC filter',
            fn_name='_wmw_auc_selection',
            legacy_weight=1.0,
            maturity='stable',
            paradigm='pairwise',
            binary_only=False,
        ),
        MethodSpec(
            key='joint_auc_l1',
            label='Joint AUC-aware L1 selector (binary-only)',
            fn_name='_joint_auc_l1_selection',
            legacy_weight=1.15,
            maturity='stable',
            paradigm='pairwise',
            binary_only=True,
        ),

        # --- Multiclass methods ---
        MethodSpec(
            key='ova_ensemble',
            label='OVA multiclass ensemble selection',
            fn_name='_ova_ensemble_selection',
            legacy_weight=1.1,
            maturity='experimental',
            paradigm='multiclass',
            requires_multiclass=True,
        ),
        MethodSpec(
            key='ecoc_class_aware',
            label='ECOC class-aware decomposition selection',
            fn_name='_ecoc_class_aware_selection',
            legacy_weight=1.2,
            maturity='experimental',
            paradigm='multiclass',
            requires_multiclass=True,
        ),
        MethodSpec(
            key='joint_multiclass_support',
            label='Joint multiclass shared-support selection',
            fn_name='_joint_multiclass_support_selection',
            legacy_weight=1.25,
            maturity='experimental',
            paradigm='multiclass',
            requires_multiclass=True,
        ),
        MethodSpec(
            key='dove_class_specific',
            label='DOvE-style class-specific multiclass selection',
            fn_name='_dove_class_specific_selection',
            legacy_weight=1.25,
            maturity='experimental',
            paradigm='multiclass',
            requires_multiclass=True,
        ),
        MethodSpec(
            key='sparse_multinomial',
            label='Sparse multinomial multiclass selection',
            fn_name='_sparse_multinomial_selection',
            legacy_weight=1.30,
            maturity='experimental',
            paradigm='multiclass',
            requires_multiclass=True,
        ),
        MethodSpec(
            key='nearest_shrunken_centroid',
            label='Nearest shrunken centroids multiclass selection',
            fn_name='_nearest_shrunken_centroid_selection',
            legacy_weight=1.20,
            maturity='stable',
            paradigm='multiclass',
            requires_multiclass=True,
        ),
        MethodSpec(
            key='class_pareto_front',
            label='Class-specific Pareto-front multiclass selection',
            fn_name='_class_specific_pareto_front_selection',
            legacy_weight=1.25,
            maturity='experimental',
            paradigm='multiclass',
            requires_multiclass=True,
        ),
        MethodSpec(
            key='sir_sdr',
            label='SIR sufficient-dimension reduction selector',
            fn_name='_sir_sdr_selection',
            legacy_weight=1.15,
            maturity='experimental',
            paradigm='multiclass',
            requires_multiclass=True,
        ),
        MethodSpec(
            key='save_sdr',
            label='SAVE sufficient-dimension reduction selector',
            fn_name='_save_sdr_selection',
            legacy_weight=1.15,
            maturity='experimental',
            paradigm='multiclass',
            requires_multiclass=True,
        ),
        MethodSpec(
            key='pfc_sdr',
            label='PFC sufficient-dimension reduction selector',
            fn_name='_pfc_sdr_selection',
            legacy_weight=1.15,
            maturity='experimental',
            paradigm='multiclass',
            requires_multiclass=True,
        ),

        # --- Information-theoretic methods ---
        MethodSpec(
            key='hsic_lasso',
            label='HSIC Lasso-style kernelized selection',
            fn_name='_hsic_lasso_selection',
            legacy_weight=1.25,
            maturity='experimental',
            paradigm='filter',
        ),
        MethodSpec(
            key='slce_centroid_encoder',
            label='SLCE centroid-encoder selection',
            fn_name='_slce_centroid_encoder_selection',
            legacy_weight=1.25,
            maturity='experimental',
            paradigm='embedded',
            binary_only=True,
        ),
        MethodSpec(
            key='mrmr_jmi',
            label='mRMR/JMI redundancy-aware selection',
            fn_name='_mrmr_jmi_selection',
            legacy_weight=1.15,
            maturity='experimental',
            paradigm='filter',
        ),

        # --- Redundancy-pruning wrappers ---
        MethodSpec(
            key='iterative_redundancy_pruning',
            label='Iterative redundancy-pruning wrapper',
            fn_name='_iterative_redundancy_pruning_selection',
            legacy_weight=1.2,
            maturity='experimental',
            paradigm='wrapper',
        ),
        MethodSpec(
            key='iterative_redundancy_pruning_bounded',
            label='Iterative redundancy-pruning wrapper (runtime-bounded)',
            fn_name='_iterative_redundancy_pruning_bounded_selection',
            legacy_weight=1.2,
            maturity='experimental',
            paradigm='wrapper',
        ),

        # --- Pairwise rank methods ---
        MethodSpec(
            key='ktsp',
            label='k-TSP pairwise rank selection',
            fn_name='_ktsp_selection',
            legacy_weight=1.1,
            maturity='experimental',
            paradigm='pairwise',
        ),

        # --- Knockoff methods ---
        MethodSpec(
            key='copula_knockoff',
            label='Copula knock-off selection',
            fn_name='_copula_knockoff_selection',
            legacy_weight=4.0,
            maturity='stable',
            paradigm='knockoff',
        ),

        # --- Group-aware methods (VAL12_Suggestions §3.3) ---
        MethodSpec(
            key='group_sparse_lasso',
            label='Group sparse lasso',
            fn_name='_group_sparse_lasso_selection',
            legacy_weight=1.0,
            maturity='experimental',
            paradigm='embedded',
        ),
        MethodSpec(
            key='pathway_group_sparse_lasso',
            label='Pathway-proxy group sparse lasso',
            fn_name='_pathway_group_sparse_lasso_selection',
            legacy_weight=1.0,
            maturity='experimental',
            paradigm='embedded',
            default_enabled=False,
        ),

        # --- DIAKRINO candidate methods (native integration §2.1) ---
        # Opt-in only: default_enabled=False -> skipped unless explicitly named in
        # enabled_methods, even under run-all.  maturity='experimental' keeps them out
        # of consensus anchoring.  CPU-only (read a persisted sidecar; no GPU inference).
        MethodSpec(
            key='diakrino_prior',
            label='DIAKRINO prior-relevance selector',
            fn_name='_diakrino_prior_selection',
            legacy_weight=1.0,
            maturity='experimental',
            paradigm='embedded',
            requires_gpu=False,
            default_enabled=False,
        ),
        MethodSpec(
            key='diakrino_screening_prior',
            label='DIAKRINO screening-relevance selector',
            fn_name='_diakrino_screening_selection',
            legacy_weight=1.0,
            maturity='experimental',
            paradigm='embedded',
            requires_gpu=False,
            default_enabled=False,
        ),
        MethodSpec(
            key='diakrino_conformal_selection',
            label='DIAKRINO conformal FDP-controlled selector',
            fn_name='_diakrino_conformal_selection',
            legacy_weight=1.0,
            maturity='experimental',
            paradigm='filter',
            requires_gpu=False,
            default_enabled=False,
        ),

        # --- Baseline methods ---
        MethodSpec(
            key='random',
            label='Random feature selection (baseline)',
            fn_name='_random_selection',
            legacy_weight=1.0,
            maturity='experimental',
            paradigm='filter',
        ),
    ]
    structured_unsupported = {
        'stability_lasso',
        'stability_subsample',
        'tigress_stability',
        'subspace_stability',
        'decorrelated_stability',
        'ipss',
        'cluster_stability',
        'rfecv',
        'oaenet',
        'joint_auc_l1',
        'ova_ensemble',
        'nearest_shrunken_centroid',
        'iterative_redundancy_pruning',
        'iterative_redundancy_pruning_bounded',
    }
    structured_conditional = {
        'boruta',
        'treeshap',
        'diakrino_conformal_selection',
    }
    specs = [
        replace(spec, structured_resampling='unsupported')
        if spec.key in structured_unsupported
        else replace(spec, structured_resampling='conditional')
        if spec.key in structured_conditional
        else spec
        for spec in specs
    ]
    return {s.key: s for s in specs}


METHOD_REGISTRY: dict[str, MethodSpec] = _build_registry()
__tabnetics_execution_isolated_state__ = {
    "METHOD_REGISTRY": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}

# ---------------------------------------------------------------------------
# Derived helpers — replace hard-coded sets / dicts throughout the codebase.
# ---------------------------------------------------------------------------


def get_method_weights() -> dict[str, float]:
    """Return {key: legacy_weight} for all registered methods."""
    return {k: v.legacy_weight for k, v in METHOD_REGISTRY.items()}


def get_experimental_keys() -> set[str]:
    """Return the set of keys whose maturity is 'experimental'."""
    return {k for k, v in METHOD_REGISTRY.items() if v.maturity == 'experimental'}


def method_excluded_by_default(spec: 'MethodSpec', enabled_methods) -> bool:
    """Hard opt-in gate for ``default_enabled=False`` methods.

    Returns True (skip) when a method opts out of default execution and is NOT
    explicitly listed in ``enabled_methods``.  With ``enabled_methods=None``
    (the run-all default), such a method is still skipped — so registering it
    never changes production selection unless a caller opts in by name.
    """
    if spec.default_enabled:
        return False
    return enabled_methods is None or spec.key not in enabled_methods

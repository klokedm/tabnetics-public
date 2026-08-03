"""Benchmark method-set profiles.

Shared profile constants imported by the runner, planner, and tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class BeyondArenaModelParity:
    """Paper-method to tabnetics backend mapping for BeyondArena comparisons."""

    paper_name: str
    normalized_name: str
    tabnetics_backend: Optional[str]
    availability: str
    dependency: str
    device: str
    local_backend: Optional[str] = None
    sample_limit: str = ""
    feature_limit: str = ""
    tuning_mode: str = "paper-default-or-local-default"
    skip_reason: str = ""
    install_hint: str = ""
    compatibility_scope: str = ""
    execution_guard: str = ""
    fallback_status: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {str(k): "" if v is None else str(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class DiakrinoValidationProfile:
    """Deferred DIAKRINO validation arm metadata.

    These profiles are intentionally descriptive: campaign execution remains
    controlled by the validation plan and issue workflow, while tests can assert
    that each DIAKRINO lever stays opt-in and names real config/CLI surfaces.
    """

    profile_id: str
    description: str
    fs_method_set: str
    comparison_baseline: str
    runner_surface: str
    cli_args: Tuple[str, ...] = tuple()
    config_toggles: Tuple[str, ...] = tuple()
    required_artifacts: Tuple[str, ...] = tuple()
    requires_sidecar: bool = False
    requires_qualification_record: bool = False
    execution_status: str = "deferred"
    pinned_classifier: str = ""
    normalization: str = ""
    calibration: str = ""
    expected_output_columns: Tuple[str, ...] = tuple()
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return dict(asdict(self))


BEYONDARENA_MODEL_PARITY: Tuple[BeyondArenaModelParity, ...] = (
    BeyondArenaModelParity(
        paper_name="Linear/Logistic Regression",
        normalized_name="Linear/Logistic Regression",
        tabnetics_backend="lr",
        availability="available",
        dependency="scikit-learn",
        device="cpu",
        tuning_mode="local default",
    ),
    BeyondArenaModelParity(
        paper_name="Random Forest",
        normalized_name="Random Forest",
        tabnetics_backend="rf",
        availability="available",
        dependency="scikit-learn",
        device="cpu",
        tuning_mode="local default/tuned when benchmark backend enables tuning",
    ),
    BeyondArenaModelParity(
        paper_name="ExtraTrees",
        normalized_name="ExtraTrees",
        tabnetics_backend="extra_tree",
        availability="available",
        dependency="scikit-learn",
        device="cpu",
        tuning_mode="local default/tuned when benchmark backend enables tuning",
    ),
    BeyondArenaModelParity(
        paper_name="CatBoost",
        normalized_name="CatBoost",
        tabnetics_backend="catboost",
        availability="optional",
        dependency="catboost",
        device="cpu",
        tuning_mode="local default/tuned when benchmark backend enables tuning",
        skip_reason="catboost optional dependency unavailable",
        fallback_status="skipped_optional_dependency_unavailable",
    ),
    BeyondArenaModelParity(
        paper_name="LightGBM",
        normalized_name="LightGBM",
        tabnetics_backend="lgbm",
        availability="optional",
        dependency="lightgbm",
        device="cpu",
        tuning_mode="local default/tuned when benchmark backend enables tuning",
        skip_reason="lightgbm optional dependency unavailable",
        fallback_status="skipped_optional_dependency_unavailable",
    ),
    BeyondArenaModelParity(
        paper_name="XGBoost",
        normalized_name="XGBoost",
        tabnetics_backend="xgb",
        availability="optional",
        dependency="xgboost",
        device="cpu",
        tuning_mode="local default/tuned when benchmark backend enables tuning",
        skip_reason="xgboost optional dependency unavailable",
        fallback_status="skipped_optional_dependency_unavailable",
    ),
    BeyondArenaModelParity(
        paper_name="RealMLP",
        normalized_name="RealMLP",
        tabnetics_backend="realmlp_td",
        availability="optional",
        dependency="pytabkit",
        device="cpu/gpu-optional",
        tuning_mode="official pytabkit TD wrapper",
        skip_reason="pytabkit RealMLP_TD_Classifier unavailable",
        install_hint="Install the pytabkit extra/environment used by tabnetics validation.",
        compatibility_scope="official pytabkit wrapper path, not a reimplementation",
        fallback_status="skipped_optional_dependency_unavailable",
    ),
    BeyondArenaModelParity(
        paper_name="TabM",
        normalized_name="TabM",
        tabnetics_backend="tabm_official",
        availability="optional",
        dependency="pytabkit",
        device="cpu/gpu-optional",
        tuning_mode="official pytabkit TabM_D wrapper",
        skip_reason="pytabkit TabM_D_Classifier unavailable",
        install_hint="Install the pytabkit extra/environment used by tabnetics validation.",
        compatibility_scope="official pytabkit wrapper path, not a reimplementation",
        fallback_status="skipped_optional_dependency_unavailable",
    ),
    BeyondArenaModelParity(
        paper_name="TabDPT",
        normalized_name="TabDPT",
        tabnetics_backend=None,
        availability="unavailable",
        dependency="upstream TabDPT package/checkpoint",
        device="gpu-preferred",
        tuning_mode="deterministic skip stub",
        skip_reason="TabDPT backend is not integrated in tabnetics core; rows may be represented as skipped.",
        install_hint="Integrate upstream TabDPT under an opt-in backend before executing this method.",
        compatibility_scope="result-schema skip stub only",
        execution_guard="gpu-required shards must use public-gpu-host after revalidation",
        fallback_status="skipped_backend_not_integrated",
    ),
    BeyondArenaModelParity(
        paper_name="TabPFN-2.6",
        normalized_name="TabPFN-2.6",
        tabnetics_backend="tabpfn",
        availability="optional",
        dependency="tabpfn",
        device="gpu-preferred",
        sample_limit="installed tabpfn package/checkpoint limit; verify before execution",
        feature_limit="installed tabpfn package/checkpoint limit; verify before execution",
        tuning_mode="zero-shot/local TabPFN hook; official baseline rows load from public-r2",
        skip_reason="tabpfn package, checkpoint, GPU, or size guard unavailable",
        install_hint="Install tabpfn and run GPU-required shards on public-gpu-host unless another GPU host is validated.",
        compatibility_scope=(
            "official TabPFN-2.6 comparison rows are public-r2 baselines; local tabpfn execution "
            "uses the installed optional package and is not treated as native Tabnetics Diakrino"
        ),
        execution_guard="BeyondArena DIAKRINO local rows require --allow-gpu-execution after public-gpu-host revalidation",
        fallback_status=(
            "deferred_gpu_revalidation | skipped_gpu_unavailable | "
            "skipped_optional_dependency_unavailable | package size/constructor guard"
        ),
    ),
    BeyondArenaModelParity(
        paper_name="TabICLv2",
        normalized_name="TabICLv2",
        tabnetics_backend=None,
        local_backend="tabiclv2-candidate",
        availability="optional",
        dependency="tabicl==2.1.1 plus pinned jingang/TabICL checkpoint",
        device="gpu-required",
        sample_limit="published evaluation regime: 300-100000 training rows",
        feature_limit="published evaluation regime: at most 2000 features",
        tuning_mode="upstream TabICLv2 defaults with an explicit revision-pinned local checkpoint",
        skip_reason="exact TabICLv2 package, pinned checkpoint, CUDA, or published size regime unavailable",
        install_hint="Install the tabiclv2 optional extra and resolve the pinned checkpoint before public-gpu-host execution.",
        compatibility_scope="isolated opt-in local comparator; excluded from the Stage-2 production backend registry",
        execution_guard="requires --backend tabiclv2-candidate and --allow-gpu-execution on public-gpu-host",
        fallback_status=(
            "deferred_gpu_revalidation | skipped_gpu_unavailable | "
            "skipped_optional_dependency_unavailable | skipped_checkpoint_unavailable | "
            "skipped_tabiclv2_outside_published_regime | skipped_tabiclv2_api_mismatch"
        ),
    ),
)

BEYONDARENA_PARITY_BACKENDS: Tuple[str, ...] = tuple(
    item.tabnetics_backend
    for item in BEYONDARENA_MODEL_PARITY
    if item.tabnetics_backend is not None
)


def beyondarena_parity_inventory() -> List[Dict[str, str]]:
    """Return a stable, JSON-serializable BeyondArena method inventory."""

    return [item.to_dict() for item in BEYONDARENA_MODEL_PARITY]

FS_METHOD_SETS: Dict[str, Tuple[str, ...]] = {
    "strict_plus_mrmr": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
    ),
    "strict_plus_mrmr_auc": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "wmw_auc",
    ),
    # C-FS6 pilot: joint AUC-aware sparse wrapper (binary-only).
    "strict_plus_mrmr_auc_joint_l1": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "wmw_auc",
        "joint_auc_l1",
    ),
    "mnpo_cmim_fcbf_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "cmim",
        "fcbf",
    ),
    # T-DIAKRINO-IMP-01: strict MNPO stack plus opt-in DIAKRINO candidate selectors.
    "mnpo_diakrino_probe": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "diakrino_prior",
        "diakrino_screening_prior",
    ),
    "mnpo_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ktsp",
        "stability_subsample",
    ),
    "mnpo_ipss_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ipss",
    ),
    # A10 bundle candidate: combine OVA (multiclass-only) with IPSS.
    "mnpo_ova_ipss_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "ipss",
    ),
    "mnpo_cluster_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "cluster_stability",
    ),
    "mnpo_decorrelated_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "decorrelated_stability",
    ),
    # A9 bundle candidate: combine OVA (multiclass-only) with decorrelated stability.
    "mnpo_ova_decorrelated_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "decorrelated_stability",
    ),
    "mnpo_subspace_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "subspace_stability",
    ),
    "mnpo_copula_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "copula_knockoff",
    ),
    "mnpo_tigress_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "tigress_stability",
    ),
    "mnpo_rankagg_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "stability_subsample",
    ),
    "mnpo_ova_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
    ),
    # A12 pilot: ECOC class-aware decomposition selector added on top of baseline.
    "mnpo_ecoc_class_aware_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "ecoc_class_aware",
    ),
    # A13 pilot: ieGENES-style iterative redundancy-pruning wrapper added on top of baseline.
    "mnpo_iterative_pruning_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "iterative_redundancy_pruning",
    ),
    # A14 pilot: joint multinomial shared-support multiclass selector.
    "mnpo_joint_multiclass_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "joint_multiclass_support",
    ),
    # A19 pilot: class-specific relevance matrix + DOvE-style multiclass path.
    "mnpo_dove_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "dove_class_specific",
    ),
    # A20 pilot: sparse multinomial multiclass backend.
    "mnpo_sparse_multinomial_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "sparse_multinomial",
    ),
    # A22 pilot: nearest shrunken centroids multiclass selector.
    "mnpo_nsc_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "nearest_shrunken_centroid",
    ),
    # Standalone A22 selector (single-method portfolio) for dispatch/debug.
    "a22_nsc": (
        "nearest_shrunken_centroid",
    ),
    # A27 pilot: NSC threshold variants (configured via fs_nsc_* flags).
    "mnpo_nsc_threshold_variants_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "nearest_shrunken_centroid",
    ),
    # A28 pilot: class-specific Pareto-front multiclass selector.
    "mnpo_class_pareto_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "class_pareto_front",
    ),
    # Standalone A28 selector (single-method portfolio) for dispatch/debug.
    "a28_class_pareto": (
        "class_pareto_front",
    ),
    # Broad profile (T-R-131): production-safe expansion of the promoted stack.
    "mnpo_broad_stable": (
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
    ),
    # Val-14 core stack: production-safe MNPO baseline used for isolated effects.
    "mnpo_v14_core": (
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
    ),
    # Val-14 isolated method contrasts.
    "mnpo_v14_core_plus_ipss": (
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
    "mnpo_v14_core_plus_group_sparse": (
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
        "group_sparse_lasso",
        "pathway_group_sparse_lasso",
    ),
    # Broad-all profile (T-R-131): exhaustive non-deprecated candidate stack.
    "mnpo_broad_all": (
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
    # T-R-132 validation bundles (toggle overlays are applied in _build_base_config).
    "mnpo_broad_bundle_a": (
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
    ),
    "mnpo_broad_bundle_b": (
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
    ),
    "mnpo_broad_bundle_c": (
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
    ),
    # Val-4 candidate profile C: mnpo_broad_stable (14 methods) + cmim + fcbf.
    # Used in build_jobs_validation4 as the "new_methods" candidate.
    # cmim gates at n>=60 (config default); fcbf has no hard sample-count gate.
    # Do NOT add runtime-racing, Rashomon, or wrapper-refine to this set —
    # those are confounders or untested at scale on broad stacks.
    "mnpo_broad_val4": (
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
        "cmim",
        "fcbf",
    ),
    # A25 pilot: HSIC Lasso kernelized selector.
    "mnpo_hsic_lasso_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "hsic_lasso",
    ),
    # Standalone A25 selector (single-method portfolio) for dispatch/debug.
    "a25_hsic_lasso": (
        "hsic_lasso",
    ),
    # A29 pilot: SLCE centroid encoder selector.
    "mnpo_slce_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "slce_centroid_encoder",
    ),
    # Standalone A29 selector (single-method portfolio) for dispatch/debug.
    "a29_slce": (
        "slce_centroid_encoder",
    ),
    # Combined A19+A20 pilot bundle (opt-in).
    "mnpo_dove_sparse_multinomial_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "dove_class_specific",
        "sparse_multinomial",
    ),
    # A15 pilot: runtime-bounded iterative pruning wrapper variant.
    "mnpo_iterative_pruning_bounded_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "iterative_redundancy_pruning_bounded",
    ),
    # A16/A17 aliases: same method stack, differentiated by control flags.
    "mnpo_iterative_pruning_bounded_cpss_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "iterative_redundancy_pruning_bounded",
    ),
    "mnpo_iterative_pruning_bounded_pareto_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "iterative_redundancy_pruning_bounded",
    ),
    # A18 alias: bounded iterative pruning + class-Pareto prefilter + stability gate.
    "mnpo_iterative_pruning_bounded_pareto_stability_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "iterative_redundancy_pruning_bounded",
    ),
    # Optional A9+A10 triple bundle (opt-in).
    "mnpo_ova_decorrelated_ipss_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "decorrelated_stability",
        "ipss",
    ),
    # Chi-Square + ReliefF extended portfolio (experimental, T-FS-CHI-003/T-FS-REL-003).
    "mnpo_chi_relief_extended": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "chi_square",
        "relieff",
    ),
}


DIAKRINO_REPLAY_CALIBRATION_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "normalization_mode",
    "normalization_family",
    "calibration_mode",
    "zscore_applied",
    "nn_probe_normalize",
    "nn_probe_normalization_family",
    "nn_probe_calibration",
    "nn_probe_zscore_applied",
    "nn_probe_calibration_summary",
    "nn_probe_raw_chunk_mean_range",
    "nn_probe_chunk_zscore_chunk_mean_range",
    "nn_probe_chunk_zscore_drift_shrink_ratio",
    "nn_probe_chunk_zscore_shrink_pass",
    "nn_probe_chunk_logit_mean_drift_count",
    "nn_probe_chunk_logit_mean_drift_json",
)


DIAKRINO_VALIDATION_PROFILE_ORDER: Tuple[str, ...] = (
    "diakrino_baseline_strict_pinned_lr",
    "diakrino_fusion_w075_zscore_nn_union_pinned_lr",
    "diakrino_distribution_gates_strict_pinned_lr",
    "diakrino_selector_prior_qualified_strict_pinned_lr",
    "diakrino_svc_probability_pool_baseline",
    "diakrino_svc_probability_candidate_pool",
)


DIAKRINO_VALIDATION_PROFILES: Dict[str, DiakrinoValidationProfile] = {
    "diakrino_baseline_strict_pinned_lr": DiakrinoValidationProfile(
        profile_id="diakrino_baseline_strict_pinned_lr",
        description="Strict MNPO+MRMR baseline with downstream classifier pinned for DIAKRINO deltas.",
        fs_method_set="strict_plus_mrmr",
        comparison_baseline="",
        runner_surface="research/run_nn_fs_tabnetics_validation.py",
        cli_args=(
            "--role-ids",
            "none",
            "--profile-ids",
            "strict_plus_mrmr",
            "--force-classifier",
            "lr",
        ),
        pinned_classifier="lr",
        notes="No DIAKRINO sidecar, oracle, distribution, or SVC-probability toggle is enabled.",
    ),
    "diakrino_fusion_w075_zscore_nn_union_pinned_lr": DiakrinoValidationProfile(
        profile_id="diakrino_fusion_w075_zscore_nn_union_pinned_lr",
        description="DIAKRINO replay arm carrying the S1 survivor: prior/screening fusion, chunk_zscore, nn_union.",
        fs_method_set="strict_plus_mrmr",
        comparison_baseline="diakrino_baseline_strict_pinned_lr",
        runner_surface="research/run_nn_fs_tabnetics_validation.py",
        cli_args=(
            "--role-ids",
            "none",
            "--profile-ids",
            "strict_plus_mrmr",
            "--ranker",
            "probe_logits",
            "--probe-logits-score-column",
            "fuse:prior+screening",
            "--probe-fusion-prior-weight",
            "0.75",
            "--probe-normalize",
            "chunk_zscore",
            "--nn-modes",
            "nn_union",
            "--force-classifier",
            "lr",
        ),
        required_artifacts=("T-DIAKRINO-NAT-01 sidecars", "T-DIAKRINO-NAT-02 dataset-id resolution"),
        requires_sidecar=True,
        pinned_classifier="lr",
        normalization="chunk_zscore",
        calibration="within_chunk_mean_std_then_global_rank01",
        expected_output_columns=DIAKRINO_REPLAY_CALIBRATION_OUTPUT_COLUMNS,
    ),
    "diakrino_distribution_gates_strict_pinned_lr": DiakrinoValidationProfile(
        profile_id="diakrino_distribution_gates_strict_pinned_lr",
        description="Strict baseline plus DIAKRINO distribution-family prescreen, skip-fit, trust, and stability gates.",
        fs_method_set="strict_plus_mrmr",
        comparison_baseline="diakrino_baseline_strict_pinned_lr",
        runner_surface="research/run_nn_fs_tabnetics_validation.py",
        cli_args=(
            "--role-ids",
            "none",
            "--profile-ids",
            "strict_plus_mrmr",
            "--diakrino-family-prescreen",
            "--diakrino-family-prior-lambda",
            "0.0",
            "--diakrino-skip-fit-discrete",
            "--diakrino-cdf-trust-gate",
            "--diakrino-stability-surrogate",
            "--force-classifier",
            "lr",
        ),
        config_toggles=(
            "dist_config.diakrino_family_prescreen_enabled",
            "dist_config.diakrino_skip_fit_discrete_enabled",
            "diakrino_cdf_trust_gate_enabled",
            "diakrino_stability_surrogate_enabled",
        ),
        required_artifacts=("T-DIAKRINO-NAT-01 sidecars", "T-DIAKRINO-NAT-02 dataset-id resolution"),
        requires_sidecar=True,
        pinned_classifier="lr",
        notes="The soft family prior remains off at lambda=0.0 unless a separate arm predeclares a nonzero value.",
    ),
    "diakrino_selector_prior_qualified_strict_pinned_lr": DiakrinoValidationProfile(
        profile_id="diakrino_selector_prior_qualified_strict_pinned_lr",
        description="Strict baseline plus qualified DIAKRINO selector-prior weights, fail-closed by checkpoint record.",
        fs_method_set="strict_plus_mrmr",
        comparison_baseline="diakrino_baseline_strict_pinned_lr",
        runner_surface="DFFSConfig/FeatureSelectorConfig overlay",
        config_toggles=(
            "fs_config.mnpo.oracle.use_diakrino_selector_prior",
            "fs_config.mnpo.oracle.diakrino_selector_prior_qualification_record",
        ),
        required_artifacts=(
            "diakrino_checkpoint_qualification_record.json",
            "T-DIAKRINO-NAT-01 sidecars",
            "T-DIAKRINO-NAT-02 dataset-id resolution",
        ),
        requires_sidecar=True,
        requires_qualification_record=True,
        pinned_classifier="lr",
        normalization="chunk_zscore",
        calibration="current_checkpoint_20260628",
        notes="Consumers must deny use when the qualification record is missing, stale, or has any required failed gate.",
    ),
    "diakrino_svc_probability_pool_baseline": DiakrinoValidationProfile(
        profile_id="diakrino_svc_probability_pool_baseline",
        description="Classifier-pool baseline for the SVC probability/log-loss toggle contrast.",
        fs_method_set="strict_plus_mrmr",
        comparison_baseline="",
        runner_surface="DFFSConfig/BeyondArenaLocalRunConfig overlay",
        pinned_classifier="candidate_pool:lr,svm_rbf,svm_linear,nb,vote_ensemble",
        notes=(
            "Uses the same strict feature-selection stack but keeps the classifier pool active "
            "so the SVC-probability toggle has a behavioral surface."
        ),
    ),
    "diakrino_svc_probability_candidate_pool": DiakrinoValidationProfile(
        profile_id="diakrino_svc_probability_candidate_pool",
        description="Classifier-pool baseline plus the opt-in SVC probability/log-loss candidate evaluation path.",
        fs_method_set="strict_plus_mrmr",
        comparison_baseline="diakrino_svc_probability_pool_baseline",
        runner_surface="DFFSConfig/BeyondArenaLocalRunConfig overlay",
        config_toggles=("model_cv_enable_svc_probability",),
        pinned_classifier="candidate_pool:lr,svm_rbf,svm_linear,nb,vote_ensemble",
        notes=(
            "Included because prior experiment runs identified classifier-probability handling as a confound. "
            "Campaign rows must report model_cv_candidate_wall_seconds and conformal score-source telemetry."
        ),
    ),
}

__tabnetics_execution_isolated_state__ = {
    "BEYONDARENA_MODEL_PARITY": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "DIAKRINO_VALIDATION_PROFILES": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "_SOTA_PROTOCOL_OVERRIDES": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}


def diakrino_validation_profile_inventory() -> List[Dict[str, object]]:
    """Return DIAKRINO validation profile metadata in planned comparison order."""

    return [DIAKRINO_VALIDATION_PROFILES[key].to_dict() for key in DIAKRINO_VALIDATION_PROFILE_ORDER]

# ---------------------------------------------------------------------------
# Val-18 singleton method sets (auto-generated from METHOD_REGISTRY).
#
# Each FS method gets a ``singleton_{method}`` entry containing only that
# single method.  These are consumed by the Val-18 M-RAW / M-SCAFFOLD
# singleton sweep profiles.
# ---------------------------------------------------------------------------

_VAL18_SINGLETON_METHODS: Tuple[str, ...] = (
    "anova_f",
    "boruta",
    "chi_square",
    "class_pareto_front",
    "cluster_stability",
    "cmim",
    "copula_knockoff",
    "decorrelated_stability",
    "dove_class_specific",
    "ecoc_class_aware",
    "fcbf",
    "gradient_boosting",
    "group_sparse_lasso",
    "pathway_group_sparse_lasso",
    "hsic_lasso",
    "ipss",
    "iterative_redundancy_pruning",
    "iterative_redundancy_pruning_bounded",
    "joint_auc_l1",
    "joint_multiclass_support",
    "ktsp",
    "linear_svm",
    "mrmr_jmi",
    "mutual_information",
    "nearest_shrunken_centroid",
    "oaenet",
    "ova_ensemble",
    "pfc_sdr",
    "relieff",
    "rfecv",
    "save_sdr",
    "sir_sdr",
    "slce_centroid_encoder",
    "sparse_multinomial",
    "stability_lasso",
    "stability_subsample",
    "subspace_stability",
    "tigress_stability",
    "treeshap",
    "wmw_auc",
    "random",
)

for _method in _VAL18_SINGLETON_METHODS:
    _key = f"singleton_{_method}"
    if _key not in FS_METHOD_SETS:
        FS_METHOD_SETS[_key] = (_method,)
del _method, _key

# ---------------------------------------------------------------------------
# Published-SOTA evaluation classifier mapping (per the validation guide).
#
# Motivation:
# Our FS/DF pipelines are the methods under test. For fair comparison to
# published results, the downstream evaluation classifier should match what
# the literature used for each dataset. This mapping drives the opt-in
# `--use-sota-matched-classifiers` benchmark config.
# ---------------------------------------------------------------------------

# Per-dataset evaluation classifiers historically used as a loose
# classifier-family proxy.  Keep this source private: public consumers use the
# typed records below so that a runnable proxy is never described as an exact
# protocol reproduction.
_LEGACY_DATASET_SOTA_CLASSIFIERS: Dict[str, Tuple[str, ...]] = {
    # Easy tier
    # Li 2004: SVM / kNN / DLDA; we treat SVM here as RBF to match common
    # microarray SVM settings and avoid mixing both linear+rbf in the same entry.
    "leukemia_golub": ("svm_rbf", "knn", "dlda"),
    "dlbcl_shipp": ("svm_linear", "knn"),
    "ovarian_petricoin": ("svm_linear",),
    "srbct_khan": ("svm_rbf", "rf"),
    "prostate_singh": ("svm_rbf", "svm_linear", "knn", "dlda"),
    "mll_microarray": ("svm_rbf", "svm_linear", "knn"),
    "orlraws10p": ("svm_linear", "knn"),
    "warp_pie10p": ("svm_linear", "knn"),
    "pixraw10p": ("svm_linear", "knn"),
    # Medium tier
    "colon_alon": ("svm_linear", "knn", "dlda"),
    "cns_pomeroy": ("svm_linear", "knn", "dlda"),
    "lung_gordon": ("svm_rbf", "knn"),
    "breast_vantveer": ("svm_linear", "knn", "dlda"),
    "gli_85": ("svm_linear", "knn", "dlda"),
    "smk_can_187": ("svm_linear", "knn", "dlda"),
    "cll_sub_111": ("svm_linear", "knn", "dlda"),
    "tox_171": ("svm_linear", "knn", "dlda"),
    "glioma_50_4class": ("svm_linear", "knn", "dlda"),
    "brain_tumor_2_50_4class": ("svm_linear", "knn", "dlda"),
    "leukemia_1_72_3class": ("svm_linear", "knn", "dlda"),
    "hf_breast_ge_mubashir1837": ("svm_linear", "knn", "dlda"),
    "lymphoma_3": ("svm_linear", "knn", "dlda"),
    # NIPS 2003 feature-selection challenge datasets are commonly benchmarked
    # with linear models (SVM/logistic) and optionally kNN.
    "arcene_nips03": ("svm_linear", "lr"),
    "madelon_nips03": ("svm_linear", "lr", "knn"),
    "gisette_nips03": ("svm_linear", "lr"),
    "dexter_nips03": ("svm_linear", "lr"),
    # Hard tier
    "nci60_ross": ("svm_rbf", "knn", "dlda"),
    "gcm_ramaswamy": ("svm_linear", "knn", "dlda", "rf"),
    "tumor11_su": ("svm_linear", "knn"),
    "tumor9_openml": ("svm_linear", "knn", "dlda"),
    "lymphoma_9": ("svm_linear", "knn", "dlda"),
    "lymphoma_11": ("svm_linear", "knn", "dlda"),
    "gla_bra_180": ("svm_linear", "knn", "dlda"),
    "carcinom_11class": ("svm_linear", "knn"),
    "nci9_60_9class": ("svm_linear", "knn", "dlda"),
    "nci_61_8class": ("svm_linear", "knn", "dlda"),
    "dorothea_nips03": ("svm_linear", "lr"),
    # Very hard tier
    # Strict-holdout is our own protocol; match our baseline evaluation pool.
    "nci60_strict_holdout": ("lr", "svm_rbf"),
    "cumida_leukemia_subtypes": ("svm_rbf", "rf", "knn"),
    "cumida_brain_gse50161": ("svm_linear", "knn", "dlda", "rf"),
    "cumida_breast_gse45827": ("svm_linear", "knn", "dlda", "rf"),
    # ---- CuMiDa GEO microarray datasets (Val-9 expansion) ----
    "cumida_gastric_gse54129": ("svm_linear", "knn", "dlda"),
    "cumida_colorectal_gse44861": ("svm_linear", "knn", "dlda", "rf"),
    "cumida_headneck_gse12452": ("svm_linear", "knn", "dlda", "rf"),
    "cumida_lung_gse19804": ("svm_linear", "knn", "dlda", "rf"),
    "cumida_ovarian_gse26712": ("svm_linear", "knn", "dlda", "rf"),
    "cumida_pancreatic_gse16515": ("svm_linear", "knn", "dlda"),
    "cumida_prostate_gse6919": ("svm_linear", "knn", "dlda", "rf"),
    "cumida_renal_gse53757": ("svm_linear", "knn", "dlda", "rf"),
    # ---- XENA TCGA RNA-seq datasets (Val-9 expansion) ----
    "xena_tcga_hnsc_hpv": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_brca": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_coad_cms": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_gbm": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_kirc": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_lgg": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_lihc": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_luad": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_ov": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_prad": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_skcm": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_stad": ("svm_linear", "knn", "dlda", "rf"),
    "xena_tcga_ucec": ("svm_linear", "knn", "dlda", "rf"),
    # ---- Reduced-version (rv_) benchmarks (Val-9 expansion) ----
    "rv_usps": ("svm_linear", "knn"),
    "rv_basehock": ("svm_linear", "knn"),
    "rv_pcmac": ("svm_linear", "knn"),
    "rv_relathe": ("svm_linear", "knn"),
    "rv_coil20": ("svm_linear", "knn"),
    "rv_warpar10p": ("svm_linear", "knn"),
    "rv_lung_discrete": ("svm_linear", "knn", "dlda"),
    "rv_yale": ("svm_linear", "knn"),
    "rv_isolet": ("svm_linear", "knn"),
    "rv_xena_tcga_blca": ("svm_linear", "knn", "dlda", "rf"),
    "rv_xena_tcga_cesc": ("svm_linear", "knn", "dlda", "rf"),
    "rv_xena_tcga_thca": ("svm_linear", "knn", "dlda", "rf"),
}


SOTA_MATCH_STATUS_EXACT = "exact"
SOTA_MATCH_STATUS_FAMILY_PROXY = "family_proxy"
SOTA_MATCH_STATUS_UNAVAILABLE = "unavailable"
_SOTA_MATCH_STATUSES = frozenset(
    {
        SOTA_MATCH_STATUS_EXACT,
        SOTA_MATCH_STATUS_FAMILY_PROXY,
        SOTA_MATCH_STATUS_UNAVAILABLE,
    }
)


@dataclass(frozen=True)
class SotaMatchedClassifierProtocol:
    """One published-classifier comparison contract.

    ``candidates`` are executable tabnetics classifier identities.  Their
    presence does not by itself claim a protocol reproduction: ``match_status``
    is persisted with every opt-in run and distinguishes a faithful rule from a
    family proxy or an unavailable published rule.
    """

    candidates: Tuple[str, ...]
    match_status: str
    source: str
    selector_classifier_coupling: str = "none_declared"
    structured_state_required: bool = False
    structured_rule_id: str = ""
    unavailable_requirements: Tuple[str, ...] = tuple()
    notes: str = ""

    def __post_init__(self) -> None:
        status = str(self.match_status).strip().lower()
        if status not in _SOTA_MATCH_STATUSES:
            raise ValueError(f"Unknown SOTA match status: {self.match_status!r}.")
        candidates = tuple(str(item).strip() for item in self.candidates if str(item).strip())
        if len(candidates) != len(set(candidates)):
            raise ValueError("SOTA protocol candidates must not contain duplicates.")
        if status == SOTA_MATCH_STATUS_UNAVAILABLE and candidates:
            raise ValueError("Unavailable SOTA protocols cannot advertise executable candidates.")
        if status != SOTA_MATCH_STATUS_UNAVAILABLE and not candidates:
            raise ValueError("Executable SOTA protocol statuses require at least one candidate.")
        source = str(self.source).strip()
        coupling = str(self.selector_classifier_coupling).strip()
        if not source:
            raise ValueError("SOTA protocol source must be non-empty.")
        if not coupling:
            raise ValueError("SOTA protocol selector/classifier coupling must be non-empty.")
        requirements = tuple(
            str(item).strip()
            for item in self.unavailable_requirements
            if str(item).strip()
        )
        if len(requirements) != len(set(requirements)):
            raise ValueError("Unavailable SOTA requirements must not contain duplicates.")
        if status == SOTA_MATCH_STATUS_EXACT and requirements:
            raise ValueError("Exact SOTA protocols cannot list unavailable requirements.")
        structured_state_required = bool(self.structured_state_required)
        structured_rule_id = str(self.structured_rule_id).strip()
        if structured_state_required and not structured_rule_id:
            raise ValueError(
                "Structured SOTA protocols must declare a structured_rule_id."
            )
        if not structured_state_required and structured_rule_id:
            raise ValueError(
                "Only structured SOTA protocols may declare a structured_rule_id."
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "match_status", status)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "selector_classifier_coupling", coupling)
        object.__setattr__(self, "structured_state_required", structured_state_required)
        object.__setattr__(self, "structured_rule_id", structured_rule_id)
        object.__setattr__(self, "unavailable_requirements", requirements)
        object.__setattr__(self, "notes", str(self.notes).strip())

    def to_record(self) -> Dict[str, object]:
        """Return JSON-safe provenance for one matched-protocol run."""

        return {
            "schema_version": "sota_matched_classifier_protocol_v1",
            "match_status": self.match_status,
            "candidates": list(self.candidates),
            "source": self.source,
            "selector_classifier_coupling": self.selector_classifier_coupling,
            "structured_state_required": bool(self.structured_state_required),
            "structured_rule_id": self.structured_rule_id,
            "unavailable_requirements": list(self.unavailable_requirements),
            "notes": self.notes,
        }


_SOTA_PROTOCOL_OVERRIDES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "leukemia_golub": {
            "source": "Golub et al. (1999), Science 286:531-537; core/the validation guide",
            "selector_classifier_coupling": "signal_to_noise_top_50 -> golub_weighted_vote",
            "unavailable_requirements": ("golub_weighted_vote",),
            "notes": "Legacy SVM/kNN/DLDA candidates are family proxies, not the published weighted-vote rule.",
        },
        "dlbcl_shipp": {
            "source": "Shipp et al. (2002), Nature Medicine 8:68-74; core/the validation guide",
            "selector_classifier_coupling": "supervised_feature_selection -> published_weighted_vote_or_knn",
            "unavailable_requirements": ("published_weighted_vote",),
            "notes": "The executable candidates are family proxies until the weighted-vote rule is registered.",
        },
        "gcm_ramaswamy": {
            "source": "Ramaswamy et al. (2001) published weighted-vote protocol; core/the validation guide",
            "selector_classifier_coupling": "one_vs_all_gene_ranking -> published_weighted_vote",
            "unavailable_requirements": ("published_weighted_vote",),
            "notes": "The executable candidates are family proxies until the weighted-vote rule is registered.",
        },
    }
)


def _family_proxy_protocol(dataset_id: str, candidates: Tuple[str, ...]) -> SotaMatchedClassifierProtocol:
    override = dict(_SOTA_PROTOCOL_OVERRIDES.get(str(dataset_id), {}))
    return SotaMatchedClassifierProtocol(
        candidates=tuple(candidates),
        match_status=SOTA_MATCH_STATUS_FAMILY_PROXY,
        source=str(
            override.get(
                "source",
                "core/the validation guide published classifier-family reference",
            )
        ),
        selector_classifier_coupling=str(
            override.get("selector_classifier_coupling", "none_declared")
        ),
        structured_state_required=bool(override.get("structured_state_required", False)),
        unavailable_requirements=tuple(override.get("unavailable_requirements", tuple())),
        notes=str(override.get("notes", "Candidate family is a proxy, not an exact paper reproduction.")),
    )


DATASET_SOTA_MATCHED_PROTOCOLS: Mapping[str, SotaMatchedClassifierProtocol] = MappingProxyType(
    {
        dataset_id: _family_proxy_protocol(dataset_id, candidates)
        for dataset_id, candidates in _LEGACY_DATASET_SOTA_CLASSIFIERS.items()
    }
)

# Compatibility view for callers that only need executable candidate names.
# New code must use DATASET_SOTA_MATCHED_PROTOCOLS so match status and coupling
# cannot be discarded.
DATASET_SOTA_CLASSIFIERS: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        dataset_id: protocol.candidates
        for dataset_id, protocol in DATASET_SOTA_MATCHED_PROTOCOLS.items()
        if protocol.match_status != SOTA_MATCH_STATUS_UNAVAILABLE
    }
)

__all__ = [
    "BEYONDARENA_MODEL_PARITY",
    "BEYONDARENA_PARITY_BACKENDS",
    "DATASET_SOTA_CLASSIFIERS",
    "DATASET_SOTA_MATCHED_PROTOCOLS",
    "FS_METHOD_SETS",
    "DIAKRINO_REPLAY_CALIBRATION_OUTPUT_COLUMNS",
    "DIAKRINO_VALIDATION_PROFILES",
    "DIAKRINO_VALIDATION_PROFILE_ORDER",
    "DiakrinoValidationProfile",
    "SOTA_MATCH_STATUS_EXACT",
    "SOTA_MATCH_STATUS_FAMILY_PROXY",
    "SOTA_MATCH_STATUS_UNAVAILABLE",
    "SotaMatchedClassifierProtocol",
    "beyondarena_parity_inventory",
    "diakrino_validation_profile_inventory",
]

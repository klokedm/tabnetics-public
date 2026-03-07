"""Benchmark method-set profiles.

Shared profile constants imported by the runner, planner, and tests.
"""

from __future__ import annotations

from typing import Dict, Tuple

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

# ---------------------------------------------------------------------------
# Published-SOTA evaluation classifier mapping (per VALIDATION.md).
#
# Motivation:
# Our FS/DF pipelines are the methods under test. For fair comparison to
# published results, the downstream evaluation classifier should match what
# the literature used for each dataset. This mapping drives the opt-in
# `--use-sota-matched-classifiers` benchmark config.
# ---------------------------------------------------------------------------

# Per-dataset evaluation classifiers to match the published SOTA family.
# Keys are FS-pipeline datasets from VALIDATION.md (and any additional FS suites
# we add to the validation catalog).
DATASET_SOTA_CLASSIFIERS: Dict[str, Tuple[str, ...]] = {
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


def _build_integrated_parent_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ds_id, spec in CATALOG.items():
        if str(getattr(spec, "pipeline", "")).strip().lower() != "integrated":
            continue

__all__ = ["FS_METHOD_SETS"]

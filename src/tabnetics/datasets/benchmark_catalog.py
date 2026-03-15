"""Benchmark dataset catalog and suite helpers.

Extracted from run_df_fs_sota_benchmark.py (T-A3-011).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from tabnetics.sota_protocols import (
        KNOWN_SOTA_SOURCE_CONFIDENCE,
        sota_claim_scope_for_confidence,
    )
except Exception as exc:
    from tabnetics.sota_protocols import (  # type: ignore
        KNOWN_SOTA_SOURCE_CONFIDENCE,
        sota_claim_scope_for_confidence,
    )

try:
    from tabnetics.datasets.validation_catalog import CATALOG
except Exception as exc:
    from tabnetics.datasets.validation_catalog import CATALOG



@dataclass(frozen=True)
class BenchmarkDatasetSpec:
    dataset_id: str
    display_name: str
    tier: str
    source_kind: str  # synthetic | validation_catalog
    sota_holdout_bal_acc: Tuple[float, float]
    sota_inflated_bal_acc: Tuple[float, float]
    fs_fraction: float
    n_final_features: int
    domain: str = "genomics"
    platform: str = "cDNA"
    sota_source_confidence: str = "proxy"
    sota_claim_scope: str = "positioning_only"
    max_train_samples: Optional[int] = None
    notes: str = ""
    validation_dataset_id: Optional[str] = None
    validation_pipeline: Optional[str] = None
    validation_scenario: Optional[str] = None


TIER_SOTA_DEFAULTS_STRICT: Dict[str, Tuple[float, float]] = {
    "easy": (0.90, 1.00),
    "medium": (0.75, 0.89),
    "hard": (0.50, 0.74),
    "very_hard": (0.00, 0.49),
}


TIER_SOTA_DEFAULTS_INFLATED: Dict[str, Tuple[float, float]] = {
    "easy": (0.95, 1.00),
    "medium": (0.85, 0.97),
    "hard": (0.65, 0.90),
    "very_hard": (0.45, 0.75),
}


# Per-dataset protocol families:
# - strict_holdout: protocol-comparable target for our benchmark policy
# - inflated: published ranges often reported under LOOCV / repeated CV / best-of-N
KNOWN_SOTA_PROTOCOL_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "synthetic_easy_dfshift": {"strict_holdout": (0.90, 1.00), "inflated": (0.95, 1.00)},
    "synthetic_medium_mixed": {"strict_holdout": (0.72, 0.88), "inflated": (0.80, 0.93)},
    "synthetic_very_hard_sparse": {"strict_holdout": (0.30, 0.55), "inflated": (0.40, 0.65)},
    "leukemia_golub": {"strict_holdout": (0.93, 1.00), "inflated": (0.98, 1.00)},
    "dlbcl_shipp": {"strict_holdout": (0.90, 0.98), "inflated": (0.95, 1.00)},
    "colon_alon": {"strict_holdout": (0.80, 0.93), "inflated": (0.88, 0.96)},
    "gli_85": {"strict_holdout": (0.75, 0.87), "inflated": (0.80, 0.92)},
    "cns_pomeroy": {"strict_holdout": (0.74, 0.86), "inflated": (0.80, 0.90)},
    "breast_vantveer": {"strict_holdout": (0.75, 0.87), "inflated": (0.80, 0.90)},
    "ovarian_petricoin": {"strict_holdout": (0.95, 1.00), "inflated": (0.99, 1.00)},
    "srbct_khan": {"strict_holdout": (0.92, 1.00), "inflated": (0.97, 1.00)},
    "prostate_singh": {"strict_holdout": (0.88, 0.95), "inflated": (0.92, 1.00)},
    "smk_can_187": {"strict_holdout": (0.65, 0.80), "inflated": (0.75, 0.90)},
    "mll_microarray": {"strict_holdout": (0.88, 0.96), "inflated": (0.95, 1.00)},
    "lymphoma_3": {"strict_holdout": (0.80, 0.90), "inflated": (0.87, 0.97)},
    "lung_gordon": {"strict_holdout": (0.75, 0.88), "inflated": (0.82, 0.95)},
    "gcm_ramaswamy": {"strict_holdout": (0.50, 0.65), "inflated": (0.60, 0.78)},
    "tumor11_su": {"strict_holdout": (0.55, 0.80), "inflated": (0.65, 0.98)},  # upper raised: Alweshah 2026 HBO wrapper 80.4% under 80/20 holdout
    "lymphoma_9": {"strict_holdout": (0.50, 0.70), "inflated": (0.65, 0.85)},
    "lymphoma_11": {"strict_holdout": (0.45, 0.65), "inflated": (0.60, 0.80)},
    "hf_breast_ge_mubashir1837": {"strict_holdout": (0.50, 0.70), "inflated": (0.75, 0.89)},
    "cumida_brain_gse50161": {"strict_holdout": (0.88, 0.98), "inflated": (0.92, 1.00)},
    "cumida_breast_gse45827": {"strict_holdout": (0.82, 0.95), "inflated": (0.90, 0.98)},
    "cumida_leukemia_subtypes": {"strict_holdout": (0.85, 0.97), "inflated": (0.95, 1.00)},
    "nci60_ross": {"strict_holdout": (0.40, 0.55), "inflated": (0.55, 0.75)},
    "nci60_strict_holdout": {"strict_holdout": (0.30, 0.50), "inflated": (0.55, 0.78)},
    "tumor9_openml": {"strict_holdout": (0.40, 0.62), "inflated": (0.50, 0.78)},
    # Scikit-feature / ASU expansion (2024-2025+ anchors).
    "cll_sub_111": {"strict_holdout": (0.78, 0.90), "inflated": (0.85, 0.95)},
    "tox_171": {"strict_holdout": (0.84, 0.93), "inflated": (0.93, 0.98)},
    "gla_bra_180": {"strict_holdout": (0.58, 0.72), "inflated": (0.68, 0.82)},
    "carcinom_11class": {"strict_holdout": (0.72, 0.88), "inflated": (0.90, 0.98)},
    "glioma_50_4class": {"strict_holdout": (0.88, 0.97), "inflated": (0.96, 1.00)},
    "brain_tumor_2_50_4class": {"strict_holdout": (0.86, 0.95), "inflated": (0.95, 1.00)},
    "leukemia_1_72_3class": {"strict_holdout": (0.85, 0.96), "inflated": (0.95, 1.00)},
    "nci9_60_9class": {"strict_holdout": (0.35, 0.52), "inflated": (0.50, 0.62)},
    "nci_61_8class": {"strict_holdout": (0.50, 0.68), "inflated": (0.65, 0.75)},
    "orlraws10p": {"strict_holdout": (0.95, 1.00), "inflated": (0.98, 1.00)},
    "warp_pie10p": {"strict_holdout": (0.95, 1.00), "inflated": (0.98, 1.00)},
    "pixraw10p": {"strict_holdout": (0.96, 1.00), "inflated": (0.99, 1.00)},
    # NIPS 2003 Feature Selection Challenge (protocol differs; these are coarse anchors).
    "arcene_nips03": {"strict_holdout": (0.75, 0.89), "inflated": (0.85, 0.97)},
    "madelon_nips03": {"strict_holdout": (0.75, 0.89), "inflated": (0.85, 0.97)},
    "gisette_nips03": {"strict_holdout": (0.75, 0.89), "inflated": (0.85, 0.97)},
    "dexter_nips03": {"strict_holdout": (0.75, 0.89), "inflated": (0.85, 0.97)},
    "dorothea_nips03": {"strict_holdout": (0.50, 0.74), "inflated": (0.65, 0.90)},
    # Extended-only CuMiDa datasets
    "cumida_prostate_gse6919": {"strict_holdout": (0.62, 0.78), "inflated": (0.72, 0.85)},
    "cumida_ovarian_gse26712": {"strict_holdout": (0.90, 0.98), "inflated": (0.95, 1.00)},
    "cumida_lung_gse19804": {"strict_holdout": (0.85, 0.95), "inflated": (0.92, 0.99)},
    "cumida_colorectal_gse44861": {"strict_holdout": (0.88, 0.96), "inflated": (0.93, 0.99)},
    "cumida_gastric_gse54129": {"strict_holdout": (0.90, 0.98), "inflated": (0.95, 1.00)},
    "cumida_pancreatic_gse16515": {"strict_holdout": (0.80, 0.92), "inflated": (0.88, 0.97)},
    "cumida_renal_gse53757": {"strict_holdout": (0.80, 0.92), "inflated": (0.88, 0.97)},
    "cumida_headneck_gse12452": {"strict_holdout": (0.78, 0.90), "inflated": (0.85, 0.95)},
    # Extended-only UCSC Xena / TCGA datasets
    "xena_tcga_brca": {"strict_holdout": (0.55, 0.72), "inflated": (0.65, 0.85)},
    "xena_tcga_luad": {"strict_holdout": (0.55, 0.70), "inflated": (0.65, 0.82)},
    "xena_tcga_ucec": {"strict_holdout": (0.50, 0.68), "inflated": (0.60, 0.80)},
    "xena_tcga_lgg": {"strict_holdout": (0.55, 0.72), "inflated": (0.65, 0.85)},
    "xena_tcga_kirc": {"strict_holdout": (0.50, 0.68), "inflated": (0.60, 0.80)},
    # Extended-only UCSC Xena / TCGA expansion (Phase 2)
    "xena_tcga_hnsc_hpv": {"strict_holdout": (0.82, 0.93), "inflated": (0.90, 0.99)},
    "xena_tcga_skcm": {"strict_holdout": (0.78, 0.90), "inflated": (0.85, 0.95)},
    "xena_tcga_gbm": {"strict_holdout": (0.55, 0.72), "inflated": (0.70, 0.90)},
    "xena_tcga_stad": {"strict_holdout": (0.58, 0.75), "inflated": (0.72, 0.90)},
    "xena_tcga_lihc": {"strict_holdout": (0.52, 0.68), "inflated": (0.65, 0.82)},
    "xena_tcga_ov": {"strict_holdout": (0.48, 0.65), "inflated": (0.62, 0.80)},
    "xena_tcga_prad": {"strict_holdout": (0.45, 0.65), "inflated": (0.60, 0.80)},
    "xena_tcga_coad_cms": {"strict_holdout": (0.58, 0.75), "inflated": (0.72, 0.90)},
    # Results-validation datasets (rv_ prefix) — VALIDATION_RESULTS.md
    "rv_warpar10p": {"strict_holdout": (0.70, 0.85), "inflated": (0.80, 0.92)},
    "rv_yale": {"strict_holdout": (0.55, 0.72), "inflated": (0.65, 0.85)},
    "rv_lung_discrete": {"strict_holdout": (0.60, 0.78), "inflated": (0.72, 0.90)},
    "rv_coil20": {"strict_holdout": (0.85, 0.95), "inflated": (0.92, 0.99)},
    "rv_relathe": {"strict_holdout": (0.82, 0.92), "inflated": (0.88, 0.96)},
    "rv_pcmac": {"strict_holdout": (0.80, 0.90), "inflated": (0.86, 0.95)},
    "rv_isolet": {"strict_holdout": (0.60, 0.78), "inflated": (0.72, 0.88)},
    "rv_basehock": {"strict_holdout": (0.90, 0.97), "inflated": (0.95, 0.99)},
    "rv_usps": {"strict_holdout": (0.90, 0.97), "inflated": (0.95, 0.99)},
    "rv_xena_tcga_thca": {"strict_holdout": (0.55, 0.72), "inflated": (0.65, 0.85)},
    "rv_xena_tcga_blca": {"strict_holdout": (0.50, 0.68), "inflated": (0.60, 0.82)},
    "rv_xena_tcga_cesc": {"strict_holdout": (0.75, 0.88), "inflated": (0.82, 0.95)},
}


def _sota_ranges_for_dataset(dataset_id: str, tier: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ds_id = str(dataset_id).strip()
    if ds_id in KNOWN_SOTA_PROTOCOL_RANGES:
        ref = KNOWN_SOTA_PROTOCOL_RANGES[ds_id]
        return tuple(ref["strict_holdout"]), tuple(ref["inflated"])
    # Integrated datasets inherit their parent's strict/inflated references when possible.
    try:
        cat_spec = CATALOG.get(ds_id)
    except Exception as exc:
        cat_spec = None
    if cat_spec is not None and str(getattr(cat_spec, "pipeline", "")).strip().lower() == "integrated":
        parent = None
        try:
            parent = cat_spec.params.get("base_dataset")
        except Exception as exc:
            parent = None
        parent = str(parent).strip() if parent else ""
        if parent and parent in KNOWN_SOTA_PROTOCOL_RANGES:
            ref = KNOWN_SOTA_PROTOCOL_RANGES[parent]
            return tuple(ref["strict_holdout"]), tuple(ref["inflated"])
    return (
        TIER_SOTA_DEFAULTS_STRICT.get(tier, (0.50, 0.80)),
        TIER_SOTA_DEFAULTS_INFLATED.get(tier, (0.60, 0.90)),
    )


def _sota_source_confidence_for_dataset(dataset_id: str, tier: str) -> str:
    ds_id = str(dataset_id).strip()
    if ds_id.startswith("synthetic_"):
        return "synthetic"
    if ds_id in KNOWN_SOTA_SOURCE_CONFIDENCE:
        return str(KNOWN_SOTA_SOURCE_CONFIDENCE[ds_id])
    try:
        cat_spec = CATALOG.get(ds_id)
    except Exception:
        cat_spec = None
    if cat_spec is not None and str(getattr(cat_spec, "pipeline", "")).strip().lower() == "integrated":
        parent = None
        try:
            parent = cat_spec.params.get("base_dataset")
        except Exception:
            parent = None
        parent = str(parent).strip() if parent else ""
        if parent and parent in KNOWN_SOTA_SOURCE_CONFIDENCE:
            return str(KNOWN_SOTA_SOURCE_CONFIDENCE[parent])
    if ds_id.startswith("rv_"):
        return "proxy"
    return "discounted"


def _sota_claim_scope_for_confidence(confidence: str) -> str:
    return str(sota_claim_scope_for_confidence(confidence))


def _synthetic_specs() -> Dict[str, BenchmarkDatasetSpec]:
    return {
        "synthetic_easy_dfshift": BenchmarkDatasetSpec(
            dataset_id="synthetic_easy_dfshift",
            display_name="Synthetic Easy DF-Shift",
            tier="easy",
            source_kind="synthetic",
            domain="synthetic",
            platform="synthetic",
            sota_source_confidence="synthetic",
            sota_claim_scope="benchmark_only",
            sota_holdout_bal_acc=_sota_ranges_for_dataset("synthetic_easy_dfshift", "easy")[0],
            sota_inflated_bal_acc=_sota_ranges_for_dataset("synthetic_easy_dfshift", "easy")[1],
            fs_fraction=0.40,
            n_final_features=40,
            notes="High-separation synthetic with class-conditional marginal shifts.",
        ),
        "synthetic_medium_mixed": BenchmarkDatasetSpec(
            dataset_id="synthetic_medium_mixed",
            display_name="Synthetic Medium Mixed",
            tier="medium",
            source_kind="synthetic",
            domain="synthetic",
            platform="synthetic",
            sota_source_confidence="synthetic",
            sota_claim_scope="benchmark_only",
            sota_holdout_bal_acc=_sota_ranges_for_dataset("synthetic_medium_mixed", "medium")[0],
            sota_inflated_bal_acc=_sota_ranges_for_dataset("synthetic_medium_mixed", "medium")[1],
            fs_fraction=0.40,
            n_final_features=50,
            notes="Moderate overlap with skew/heavy-tail perturbations.",
        ),
        "synthetic_very_hard_sparse": BenchmarkDatasetSpec(
            dataset_id="synthetic_very_hard_sparse",
            display_name="Synthetic Very Hard Sparse",
            tier="very_hard",
            source_kind="synthetic",
            domain="synthetic",
            platform="synthetic",
            sota_source_confidence="synthetic",
            sota_claim_scope="benchmark_only",
            sota_holdout_bal_acc=_sota_ranges_for_dataset("synthetic_very_hard_sparse", "very_hard")[0],
            sota_inflated_bal_acc=_sota_ranges_for_dataset("synthetic_very_hard_sparse", "very_hard")[1],
            fs_fraction=0.40,
            n_final_features=60,
            notes="Multiclass sparse-signal problem with contamination/heaping.",
        ),
    }


def _validation_specs() -> Dict[str, BenchmarkDatasetSpec]:
    out: Dict[str, BenchmarkDatasetSpec] = {}
    for ds_id, spec in CATALOG.items():
        if spec.pipeline not in {"fs", "integrated"}:
            continue
        fs_fraction = float(spec.params.get("fs_fraction", 0.40))
        n_final_features = int(spec.params.get("n_final_features", 50))
        max_train_samples = spec.params.get("max_train_samples")
        if max_train_samples is not None:
            try:
                max_train_samples = int(max_train_samples)
            except Exception as exc:
                max_train_samples = None
        scenario = spec.params.get("scenario")
        strict_holdout, inflated = _sota_ranges_for_dataset(ds_id, spec.tier)
        confidence = _sota_source_confidence_for_dataset(ds_id, spec.tier)
        out[ds_id] = BenchmarkDatasetSpec(
            dataset_id=ds_id,
            display_name=str(spec.display_name),
            tier=str(spec.tier),
            source_kind="validation_catalog",
            domain=str(getattr(spec, "domain", "genomics") or "genomics"),
            platform=str(getattr(spec, "platform", "cDNA") or "cDNA"),
            validation_dataset_id=ds_id,
            validation_pipeline=str(spec.pipeline),
            validation_scenario=str(scenario) if scenario is not None else None,
            sota_source_confidence=confidence,
            sota_claim_scope=_sota_claim_scope_for_confidence(confidence),
            sota_holdout_bal_acc=strict_holdout,
            sota_inflated_bal_acc=inflated,
            fs_fraction=fs_fraction,
            n_final_features=n_final_features,
            max_train_samples=max_train_samples,
            notes=f"Validation catalog dataset ({spec.pipeline}).",
        )
    return out


def _build_benchmark_datasets() -> Dict[str, BenchmarkDatasetSpec]:
    specs: Dict[str, BenchmarkDatasetSpec] = {}
    specs.update(_synthetic_specs())
    specs.update(_validation_specs())
    return specs


BENCHMARK_DATASETS: Dict[str, BenchmarkDatasetSpec] = _build_benchmark_datasets()


def _build_dataset_sets() -> Dict[str, List[str]]:
    synthetic = [k for k, v in BENCHMARK_DATASETS.items() if v.source_kind == "synthetic"]
    validation = [k for k, v in BENCHMARK_DATASETS.items() if v.source_kind == "validation_catalog"]
    validation_fs = [k for k, v in BENCHMARK_DATASETS.items() if v.validation_pipeline == "fs"]
    validation_integrated = [k for k, v in BENCHMARK_DATASETS.items() if v.validation_pipeline == "integrated"]

    sets: Dict[str, List[str]] = {
        "smoke": [ds for ds in ("synthetic_easy_dfshift", "leukemia_golub", "nci60_ross") if ds in BENCHMARK_DATASETS],
        "all": list(BENCHMARK_DATASETS.keys()),
        "real": validation,
        "synthetic": synthetic,
        "validation_all": validation,
        "validation_fs_all": validation_fs,
        "validation_integrated_all": validation_integrated,
    }

    # "extended" includes all datasets; "core" excludes extended_only datasets.
    extended_only_ids = set()
    for ds_id in BENCHMARK_DATASETS:
        cat_spec = CATALOG.get(ds_id)
        if cat_spec is not None and getattr(cat_spec, "extended_only", False):
            extended_only_ids.add(ds_id)
    sets["extended"] = list(BENCHMARK_DATASETS.keys())
    sets["core"] = [k for k in BENCHMARK_DATASETS if k not in extended_only_ids]

    for tier in ("easy", "medium", "hard", "very_hard"):
        sets[tier] = [k for k, v in BENCHMARK_DATASETS.items() if v.tier == tier]
        sets[f"validation_{tier}"] = [k for k, v in BENCHMARK_DATASETS.items() if v.source_kind == "validation_catalog" and v.tier == tier]

    # Suite-style sets to split long RunPod jobs across multiple pods.
    suite_defs: Dict[str, Tuple[str, ...]] = {
        # Microarray/proteomics benchmarks available on OpenML.
        "suite_microarray_easy_openml": (
            "leukemia_golub",
            "dlbcl_shipp",
            "ovarian_petricoin",
            "srbct_khan",
            "prostate_singh",
            "mll_microarray",
        ),
        "suite_microarray_medium_openml": (
            "colon_alon",
            "cns_pomeroy",
            "breast_vantveer",
            "gli_85",
            "smk_can_187",
            "lymphoma_3",
        ),
        "suite_microarray_hard_openml": (
            "nci60_ross",
            "gcm_ramaswamy",
            "tumor11_su",
            "tumor9_openml",
            "lymphoma_9",
            "lymphoma_11",
            "lung_gordon",
        ),
        # Scikit-feature/ASU multiclass microarray expansion (2026-02).
        "suite_scikit_feature_microarray": (
            "cll_sub_111",
            "tox_171",
            "gla_bra_180",
            "carcinom_11class",
            "glioma_50_4class",
            "brain_tumor_2_50_4class",
            "leukemia_1_72_3class",
            "nci_61_8class",
            "nci9_60_9class",
        ),
        # Non-genomics HDLSS sanity-check suite.
        "suite_non_genomics_hdlss": (
            "orlraws10p",
            "warp_pie10p",
            "pixraw10p",
        ),
        "suite_microarray_very_hard": (
            "nci60_strict_holdout",
        ),
        # Manual-or-synth datasets (require local env var paths for real data).
        "suite_manual_fs": (
            "cumida_leukemia_subtypes",
            "cumida_brain_gse50161",
            "cumida_breast_gse45827",
            "hf_breast_ge_mubashir1837",
        ),
        # NIPS 2003 feature-selection challenge benchmarks.
        "suite_nips03_dense": (
            "arcene_nips03",
            "madelon_nips03",
        ),
        "suite_nips03_sparse": (
            "dexter_nips03",
            "dorothea_nips03",
            "gisette_nips03",
        ),
        # Integrated DF+FS scenarios (runs on parent FS datasets via catalog inheritance).
        "suite_integrated_cdf": (
            "int_cdf_leukemia",
            "int_cdf_colon",
            "int_cdf_nci60",
        ),
        "suite_integrated_reliability": (
            "int_low_gof_downweighting",
            "int_distribution_stability_confidence",
        ),
        # Synthetic diagnostics.
        "suite_synthetic": (
            "synthetic_easy_dfshift",
            "synthetic_medium_mixed",
            "synthetic_very_hard_sparse",
        ),
        # Curated quick-check sets (for single-seed or 2-seed validation when a full
        # 3-seed sweep is not warranted).
        "quick_easy_guard": (
            "leukemia_golub",
            "dlbcl_shipp",
            "srbct_khan",
        ),
        "quick_medium_gap": (
            "colon_alon",
            "cns_pomeroy",
            "gli_85",
            "smk_can_187",
        ),
        "quick_hard_multiclass": (
            "nci60_ross",
            "gcm_ramaswamy",
            "tumor11_su",
            "tumor9_openml",
            "lymphoma_11",
            "nci9_60_9class",
            "nci_61_8class",
            "lung_gordon",
            "carcinom_11class",
        ),
        "quick_decorrelation": (
            "nci60_ross",
            "nci60_strict_holdout",
            "gcm_ramaswamy",
            "tumor11_su",
        ),
        "quick_scikit_multiclass_hard": (
            "nci9_60_9class",
            "nci_61_8class",
            "carcinom_11class",
            "cll_sub_111",
        ),
        "quick_extreme_p": (
            "gla_bra_180",
            "glioma_50_4class",
        ),
        "quick_non_genomics_hdlss": (
            "orlraws10p",
            "warp_pie10p",
            "pixraw10p",
        ),
        "quick_df_fastpath": (
            "leukemia_golub",
            "colon_alon",
            "nci60_ross",
            "nci60_strict_holdout",
        ),
        "quick_manual_loaders": (
            "cumida_leukemia_subtypes",
            "cumida_brain_gse50161",
            "cumida_breast_gse45827",
            "hf_breast_ge_mubashir1837",
        ),

        "quick_integrated": (
            "int_cdf_leukemia",
            "int_cdf_colon",
            "int_low_gof_downweighting",
        ),
        # Extended-only dataset suites.
        "suite_extended_cumida": (
            "cumida_prostate_gse6919",
            "cumida_ovarian_gse26712",
            "cumida_lung_gse19804",
            "cumida_colorectal_gse44861",
            "cumida_gastric_gse54129",
            "cumida_pancreatic_gse16515",
            "cumida_renal_gse53757",
            "cumida_headneck_gse12452",
        ),
        "suite_extended_xena_tcga": (
            "xena_tcga_brca",
            "xena_tcga_luad",
            "xena_tcga_ucec",
            "xena_tcga_lgg",
            "xena_tcga_kirc",
            "xena_tcga_hnsc_hpv",
            "xena_tcga_skcm",
            "xena_tcga_gbm",
            "xena_tcga_stad",
            "xena_tcga_lihc",
            "xena_tcga_ov",
            "xena_tcga_prad",
            "xena_tcga_coad_cms",
        ),
        # Results-validation suite (independent hold-out, see VALIDATION_RESULTS.md).
        "suite_results_validation": (
            "rv_warpar10p",
            "rv_yale",
            "rv_lung_discrete",
            "rv_coil20",
            "rv_relathe",
            "rv_pcmac",
            "rv_isolet",
            "rv_basehock",
            "rv_usps",
            "rv_xena_tcga_thca",
            "rv_xena_tcga_blca",
            "rv_xena_tcga_cesc",
        ),
    }
    for name, ds_ids in suite_defs.items():
        sets[name] = [ds_id for ds_id in ds_ids if ds_id in BENCHMARK_DATASETS]

    if not sets["smoke"] and sets["all"]:
        sets["smoke"] = sets["all"][: min(3, len(sets["all"]))]
    return sets


DATASET_SETS: Dict[str, List[str]] = _build_dataset_sets()

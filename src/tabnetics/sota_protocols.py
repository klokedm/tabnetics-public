"""Lightweight SOTA protocol metadata for reporting and benchmark annotation.

This module is intentionally dependency-light so offline analysis scripts can
import it without pulling in dataset loaders or external ML packages.
"""

from __future__ import annotations

from typing import Dict


# Confidence of the best strict-holdout anchor currently available for direct
# scorecard use. This is orthogonal to the numeric strict/inflated ranges:
# - direct: close protocol match (holdout / independent validation / fixed test set)
# - discounted: exact dataset exists, but strongest literature numbers rely on
#   weaker protocols (LOOCV / repeated CV / best-of-N / challenge splits / etc.)
# - proxy: no clean exact-task benchmark was pinned down; range is still a
#   dataset-family or task proxy
# - synthetic: internal synthetic benchmark, not a published external SOTA row
KNOWN_SOTA_SOURCE_CONFIDENCE: Dict[str, str] = {
    "leukemia_golub": "direct",
    "dlbcl_shipp": "discounted",
    "ovarian_petricoin": "direct",
    "srbct_khan": "direct",
    "prostate_singh": "discounted",
    "mll_microarray": "discounted",
    "colon_alon": "discounted",
    "cns_pomeroy": "discounted",
    "lung_gordon": "discounted",
    "breast_vantveer": "direct",
    "gli_85": "discounted",
    "smk_can_187": "discounted",
    "cll_sub_111": "discounted",
    "tox_171": "discounted",
    "brain_tumor_2_50_4class": "discounted",
    "leukemia_1_72_3class": "direct",
    "lymphoma_3": "discounted",
    "gcm_ramaswamy": "direct",
    "tumor11_su": "direct",
    "tumor9_openml": "discounted",
    "nci60_ross": "discounted",
    "nci60_strict_holdout": "direct",
    "nci9_60_9class": "discounted",
    "nci_61_8class": "proxy",
    "lymphoma_9": "proxy",
    "lymphoma_11": "proxy",
    "hf_breast_ge_mubashir1837": "proxy",
    "carcinom_11class": "discounted",
    "gla_bra_180": "discounted",
    "glioma_50_4class": "proxy",
    "arcene_nips03": "discounted",
    "madelon_nips03": "discounted",
    "gisette_nips03": "discounted",
    "dexter_nips03": "discounted",
    "dorothea_nips03": "discounted",
    "orlraws10p": "discounted",
    "warp_pie10p": "discounted",
    "pixraw10p": "discounted",
    "rv_basehock": "proxy",
    "cumida_leukemia_subtypes": "discounted",
    "cumida_brain_gse50161": "discounted",
    "cumida_breast_gse45827": "discounted",
    "cumida_prostate_gse6919": "discounted",
    "cumida_ovarian_gse26712": "discounted",
    "cumida_lung_gse19804": "discounted",
    "cumida_colorectal_gse44861": "discounted",
    "cumida_gastric_gse54129": "proxy",
    "cumida_pancreatic_gse16515": "proxy",
    "cumida_renal_gse53757": "direct",
    "cumida_headneck_gse12452": "proxy",
    "xena_tcga_brca": "proxy",
    "xena_tcga_luad": "proxy",
    "xena_tcga_ucec": "proxy",
    "xena_tcga_lgg": "proxy",
    "xena_tcga_kirc": "proxy",
    "xena_tcga_hnsc_hpv": "discounted",
    "xena_tcga_skcm": "discounted",
    "xena_tcga_gbm": "discounted",
    "xena_tcga_stad": "discounted",
    "xena_tcga_lihc": "proxy",
    "xena_tcga_ov": "discounted",
    "xena_tcga_prad": "proxy",
    "xena_tcga_coad_cms": "direct",
}


SOTA_CLAIM_SCOPE_BY_CONFIDENCE: Dict[str, str] = {
    "direct": "hard_claim",
    "discounted": "qualified_claim",
    "proxy": "positioning_only",
    "synthetic": "benchmark_only",
}


def sota_claim_scope_for_confidence(confidence: str) -> str:
    raw = str(confidence or "proxy").strip().lower()
    return str(SOTA_CLAIM_SCOPE_BY_CONFIDENCE.get(raw, "positioning_only"))

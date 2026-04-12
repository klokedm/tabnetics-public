"""DatasetSpec registry (single source of truth for validation + benchmark datasets)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    pipeline: str
    tier: str
    loader_kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    # Registry/benchmark metadata (filled in finalize step)
    source_kind: str = "validation_catalog"  # synthetic | validation_catalog | dist_benchmark
    domain: str = "genomics"  # genomics | non_genomic | synthetic
    platform: str = "cDNA"  # Affy HG-U95 | Affy HG-U133A | cDNA | mass-spec | synthetic
    sota_holdout_bal_acc: Optional[Tuple[float, float]] = None
    sota_inflated_bal_acc: Optional[Tuple[float, float]] = None
    provenance: str = ""
    extended_only: bool = False  # True = only included in the expanded/extended catalog


def _build_dataset_registry() -> Dict[str, DatasetSpec]:
    fs_specs = {
        "leukemia_golub": DatasetSpec(
            dataset_id="leukemia_golub",
            display_name="Leukemia (Golub)",
            pipeline="fs",
            tier="easy",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "leukemia", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 72,
                    "n_features": 7129,
                    "n_classes": 2,
                    # OpenML `leukemia` v1: 47/25 class split.
                    "weights": [47 / 72.0, 25 / 72.0],
                    "difficulty": "easy",
                },
                "n_final_features": 50,
            },
        ),
        "dlbcl_shipp": DatasetSpec(
            dataset_id="dlbcl_shipp",
            display_name="DLBCL (Shipp)",
            pipeline="fs",
            tier="easy",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "DLBCL", "version": 1},
                    {"name": "dlbcl", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 77,
                    # OpenML `DLBCL` v1 uses 5,469 genes in the published-preprocessed variant.
                    "n_features": 5469,
                    "n_classes": 2,
                    "weights": [58 / 77.0, 19 / 77.0],
                    "difficulty": "easy",
                },
                "n_final_features": 50,
            },
        ),
        "ovarian_petricoin": DatasetSpec(
            dataset_id="ovarian_petricoin",
            display_name="Ovarian Cancer (Petricoin)",
            pipeline="fs",
            tier="easy",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "ovarian", "version": 1},
                    {"name": "Ovarian", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 253,
                    "n_features": 15154,
                    "n_classes": 2,
                    "weights": [162 / 253.0, 91 / 253.0],
                    "difficulty": "easy",
                },
                "n_final_features": 60,
            },
        ),
        "srbct_khan": DatasetSpec(
            dataset_id="srbct_khan",
            display_name="SRBCT (Khan)",
            pipeline="fs",
            tier="easy",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "SRBCT", "version": 1},
                    {"name": "srbct", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 83,
                    "n_features": 2308,
                    "n_classes": 4,
                    # OpenML `SRBCT` v1: [29, 11, 18, 25] class split.
                    "weights": [29 / 83.0, 11 / 83.0, 18 / 83.0, 25 / 83.0],
                    "difficulty": "easy",
                },
                "n_final_features": 50,
            },
        ),
        "prostate_singh": DatasetSpec(
            dataset_id="prostate_singh",
            display_name="Prostate Cancer (Singh)",
            pipeline="fs",
            tier="easy",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "prostate", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 102,
                    "n_features": 12600,
                    "n_classes": 2,
                    "weights": [52 / 102.0, 50 / 102.0],
                    "difficulty": "easy",
                },
                "n_final_features": 50,
            },
        ),
        "mll_microarray": DatasetSpec(
            dataset_id="mll_microarray",
            display_name="MLL (OpenML)",
            pipeline="fs",
            tier="easy",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "MLL", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 72,
                    "n_features": 12582,
                    "n_classes": 3,
                    # OpenML `MLL` v1: 24/28/20 class split.
                    "weights": [24 / 72.0, 28 / 72.0, 20 / 72.0],
                    "difficulty": "easy",
                },
                "n_final_features": 60,
            },
        ),
        "gli_85": DatasetSpec(
            dataset_id="gli_85",
            display_name="Glioma (GLI_85)",
            pipeline="fs",
            tier="medium",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "GLI", "version": 1},
                ],
                "synthetic_profile": {
                    # OpenML `GLI` v1: (85, 22283), binary.
                    "n_samples": 85,
                    "n_features": 22283,
                    "n_classes": 2,
                    # OpenML target split: 26 / 59.
                    "weights": [26 / 85.0, 59 / 85.0],
                    "difficulty": "medium",
                },
                "n_final_features": 60,
            },
        ),
        "smk_can_187": DatasetSpec(
            dataset_id="smk_can_187",
            display_name="SMK_CAN_187 (Smoking)",
            pipeline="fs",
            tier="medium",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "SMK", "version": 1},
                ],
                "synthetic_profile": {
                    # OpenML `SMK` v1: (187, 19993), binary.
                    "n_samples": 187,
                    "n_features": 19993,
                    "n_classes": 2,
                    # OpenML target split: 90 / 97.
                    "weights": [90 / 187.0, 97 / 187.0],
                    "difficulty": "medium",
                },
                "n_final_features": 60,
            },
        ),
        # --- Scikit-feature/ASU expansion candidates (2026-02): multiclass HDLSS ---
        "cll_sub_111": DatasetSpec(
            dataset_id="cll_sub_111",
            display_name="CLL_SUB_111 (Scikit-feature)",
            pipeline="fs",
            tier="medium",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/CLL-SUB-111.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/CLL-SUB-111.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 111,
                    "n_features": 11340,
                    "n_classes": 3,
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
        ),
        "tox_171": DatasetSpec(
            dataset_id="tox_171",
            display_name="TOX_171 (Scikit-feature)",
            pipeline="fs",
            tier="medium",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/TOX-171.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/TOX-171.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 171,
                    "n_features": 5748,
                    "n_classes": 4,
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
        ),
        "gla_bra_180": DatasetSpec(
            dataset_id="gla_bra_180",
            display_name="GLA-BRA-180 (Scikit-feature)",
            pipeline="fs",
            tier="hard",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/GLA-BRA-180.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/GLA-BRA-180.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 180,
                    "n_features": 49151,
                    "n_classes": 4,
                    "difficulty": "hard",
                },
                "n_final_features": 120,
            },
        ),
        "carcinom_11class": DatasetSpec(
            dataset_id="carcinom_11class",
            display_name="Carcinom 11-class (Scikit-feature)",
            pipeline="fs",
            tier="hard",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/Carcinom.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 174,
                    "n_features": 9182,
                    "n_classes": 11,
                    "difficulty": "hard",
                },
                "n_final_features": 90,
            },
        ),
        "glioma_50_4class": DatasetSpec(
            dataset_id="glioma_50_4class",
            display_name="Glioma (50 x 4434, 4-class)",
            pipeline="fs",
            tier="medium",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/GLIOMA.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 50,
                    "n_features": 4434,
                    "n_classes": 4,
                    "difficulty": "medium",
                },
                "n_final_features": 70,
            },
        ),
        # Nutt et al. 2003, Cancer Research 63(7):1602-1607.
        # 50 samples, 12625 genes (Affymetrix U95Av2), 4 classes: CG/CO/NG/NO.
        # Source: biolab.si (University of Ljubljana / Orange Data Mining).
        "brain_tumor_2_50_4class": DatasetSpec(
            dataset_id="brain_tumor_2_50_4class",
            display_name="Brain Tumor 2 — Nutt 2003 (50 x 12625, 4-class)",
            pipeline="fs",
            tier="medium",
            loader_kind="tab_url_or_synth",
            params={
                "tab_url_options": [
                    {
                        "url": "https://file.biolab.si/biolab/supp/bi-cancer/projections/_datasets/glioblastoma.tab",
                    },
                ],
                "openml_options": [
                    {"name": "Brain_Tumor2", "version": 1},
                    {"name": "Brain_Tumor_2", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 50,
                    "n_features": 12625,
                    "n_classes": 4,
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
        ),
        # Armstrong et al. 2002, Nature Genetics 30(1):41-47 (MLL leukemia).
        # 72 samples, 12533 genes (Affymetrix U95Av2), 3 classes: ALL/AML/MLL.
        # Source: biolab.si (University of Ljubljana / Orange Data Mining).
        "leukemia_1_72_3class": DatasetSpec(
            dataset_id="leukemia_1_72_3class",
            display_name="MLL Leukemia — Armstrong 2002 (72 x 12533, 3-class)",
            pipeline="fs",
            tier="medium",
            loader_kind="tab_url_or_synth",
            params={
                "tab_url_options": [
                    {
                        "url": "https://file.biolab.si/biolab/supp/bi-cancer/projections/_datasets/MLL.tab",
                    },
                ],
                "openml_options": [
                    {"name": "Leukemia1", "version": 1},
                    {"name": "leukemia1", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 72,
                    "n_features": 12533,
                    "n_classes": 3,
                    "difficulty": "medium",
                },
                "n_final_features": 70,
            },
        ),
        # NOTE: A legacy duplicate leukemia variant was removed from the catalog.
        # (same Armstrong 2002 MLL data with a different feature-filtering threshold
        # from the GEMS benchmark, which is now dead: gems-system.org).
        "nci9_60_9class": DatasetSpec(
            dataset_id="nci9_60_9class",
            display_name="NCI9 (60 x 9712, 9-class)",
            pipeline="fs",
            tier="hard",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "NCI9", "version": 1},
                    {"name": "9_Tumors", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 60,
                    "n_features": 9712,
                    "n_classes": 9,
                    "difficulty": "hard",
                },
                "n_final_features": 80,
            },
        ),
        "nci_61_8class": DatasetSpec(
            dataset_id="nci_61_8class",
            display_name="NCI (61 x 5244, 8-class proxy)",
            pipeline="fs",
            tier="hard",
            loader_kind="nci60_proxy_or_synth",
            params={
                # OpenML no longer exposes the legacy NCI-61 dataset. We build a
                # deterministic real-data proxy from ISLR NCI60 by:
                # - mapping K562* to LEUKEMIA and MCF7* to BREAST,
                # - dropping UNKNOWN and PROSTATE,
                # - retaining top-variance genes to 5,244 features.
                "proxy_target_features": 5244,
                "synthetic_profile": {
                    "n_samples": 61,
                    "n_features": 5244,
                    "n_classes": 8,
                    "difficulty": "hard",
                },
                "n_final_features": 80,
            },
        ),
        # --- Non-genomics HDLSS additions used in recent FS papers ---
        "orlraws10p": DatasetSpec(
            dataset_id="orlraws10p",
            display_name="ORLraws10P (Face)",
            pipeline="fs",
            tier="easy",
            loader_kind="face_proxy_or_synth",
            params={
                "domain": "face",
                "face_resize_shape": (112, 92),
                "synthetic_profile": {
                    "n_samples": 100,
                    "n_features": 10304,
                    "n_classes": 10,
                    "difficulty": "easy",
                },
                "n_final_features": 90,
            },
        ),
        "warp_pie10p": DatasetSpec(
            dataset_id="warp_pie10p",
            display_name="warpPIE10P (Face)",
            pipeline="fs",
            tier="easy",
            loader_kind="face_proxy_or_synth",
            params={
                "domain": "face",
                "face_resize_shape": (55, 44),
                "synthetic_profile": {
                    "n_samples": 210,
                    "n_features": 2420,
                    "n_classes": 10,
                    "difficulty": "easy",
                },
                "n_final_features": 80,
            },
        ),
        "pixraw10p": DatasetSpec(
            dataset_id="pixraw10p",
            display_name="pixraw10P (Face)",
            pipeline="fs",
            tier="easy",
            loader_kind="face_proxy_or_synth",
            params={
                "domain": "face",
                "face_resize_shape": (100, 100),
                "synthetic_profile": {
                    "n_samples": 100,
                    "n_features": 10000,
                    "n_classes": 10,
                    "difficulty": "easy",
                },
                "n_final_features": 90,
            },
        ),
        "hf_breast_ge_mubashir1837": DatasetSpec(
            dataset_id="hf_breast_ge_mubashir1837",
            display_name="Breast Cancer Gene Expression (HF: mubashir1837)",
            pipeline="fs",
            # Routed to hard: tiny-n (51) HDLSS RNA-seq benchmark with no protocol-stable
            # published SOTA under strict holdout (placeholder strict range ~0.50–0.70).
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "local_path_env": "HF_BREAST_GE_PATH",
                "default_local_path": "train_data/huggingface/mubashir1837_breast_cancer_ge/cleaned_expression.csv",
                "synthetic_profile": {
                    # HuggingFace dataset: 51 samples, ~28k genes, binary response label.
                    "n_samples": 51,
                    "n_features": 28278,
                    "n_classes": 2,
                    "weights": [26 / 51.0, 25 / 51.0],
                    "difficulty": "hard",
                },
                "n_final_features": 80,
                "source_policy": "real_only",
            },
        ),
        "lymphoma_3": DatasetSpec(
            dataset_id="lymphoma_3",
            display_name="Lymphoma-3 (OpenML)",
            pipeline="fs",
            tier="medium",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "Lymphoma-3", "version": 1},
                ],
                "synthetic_profile": {
                    # OpenML `Lymphoma-3` v1: (66, 4026), 3 classes.
                    "n_samples": 66,
                    "n_features": 4026,
                    "n_classes": 3,
                    # OpenML target split: 11 / 46 / 9.
                    "weights": [11 / 66.0, 46 / 66.0, 9 / 66.0],
                    "difficulty": "medium",
                },
                "n_final_features": 60,
            },
        ),
        "lymphoma_9": DatasetSpec(
            dataset_id="lymphoma_9",
            display_name="Lymphoma-9 (OpenML)",
            pipeline="fs",
            tier="hard",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "Lymphoma-9", "version": 1},
                ],
                "synthetic_profile": {
                    # OpenML `Lymphoma-9` v1: (96, 4026), 9 classes.
                    "n_samples": 96,
                    "n_features": 4026,
                    "n_classes": 9,
                    # OpenML target split: 10/11/46/9/2/2/6/4/6 (ABB/CLL/DLBCL/FL/GCB/NIL/RAT/RBB/TCL).
                    "weights": [
                        10 / 96.0,
                        11 / 96.0,
                        46 / 96.0,
                        9 / 96.0,
                        2 / 96.0,
                        2 / 96.0,
                        6 / 96.0,
                        4 / 96.0,
                        6 / 96.0,
                    ],
                    "difficulty": "hard",
                },
                "n_final_features": 70,
            },
        ),
        "lymphoma_11": DatasetSpec(
            dataset_id="lymphoma_11",
            display_name="Lymphoma-11 (OpenML)",
            pipeline="fs",
            tier="hard",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "Lymphoma-11", "version": 1},
                ],
                "synthetic_profile": {
                    # OpenML `Lymphoma-11` v1: (96, 4026), 11 classes.
                    "n_samples": 96,
                    "n_features": 4026,
                    "n_classes": 11,
                    # OpenML target split: 10/23/11/1/9/2/22/2/6/4/6
                    # (ABB/ACL/CLL/DLBCL/FL/GCB/GCL/NIL/RAT/RBB/TCL).
                    "weights": [
                        10 / 96.0,
                        23 / 96.0,
                        11 / 96.0,
                        1 / 96.0,
                        9 / 96.0,
                        2 / 96.0,
                        22 / 96.0,
                        2 / 96.0,
                        6 / 96.0,
                        4 / 96.0,
                        6 / 96.0,
                    ],
                    "difficulty": "hard",
                },
                "n_final_features": 70,
            },
        ),
        "tumor9_openml": DatasetSpec(
            dataset_id="tumor9_openml",
            display_name="9-Tumors (OpenML)",
            pipeline="fs",
            tier="hard",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "9_Tumors", "version": 1},
                ],
                "synthetic_profile": {
                    # OpenML `9_Tumors` v1: (60, 5726), 9 classes.
                    "n_samples": 60,
                    "n_features": 5726,
                    "n_classes": 9,
                    "difficulty": "hard",
                },
                "n_final_features": 60,
            },
        ),
        "arcene_nips03": DatasetSpec(
            dataset_id="arcene_nips03",
            display_name="ARCENE (NIPS 2003 FS Challenge)",
            pipeline="fs",
            tier="medium",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "arcene", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 200,
                    "n_features": 10000,
                    "n_classes": 2,
                    "weights": [0.5, 0.5],
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
        ),
        "madelon_nips03": DatasetSpec(
            dataset_id="madelon_nips03",
            display_name="MADELON (NIPS 2003 FS Challenge)",
            pipeline="fs",
            tier="medium",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "madelon", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 2600,
                    "n_features": 500,
                    "n_classes": 2,
                    "weights": [0.5, 0.5],
                    "difficulty": "medium",
                },
                "n_final_features": 50,
            },
        ),
        "gisette_nips03": DatasetSpec(
            dataset_id="gisette_nips03",
            display_name="GISETTE (NIPS 2003 FS Challenge)",
            pipeline="fs",
            tier="medium",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    # OpenML `gisette` v1 is inactive/retired; prefer active v2 mirror.
                    {"name": "gisette", "version": 2},
                    {"data_id": 41026},
                ],
                "synthetic_profile": {
                    "n_samples": 7000,
                    "n_features": 5001,
                    "n_classes": 2,
                    "weights": [0.5, 0.5],
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
        ),
        "dexter_nips03": DatasetSpec(
            dataset_id="dexter_nips03",
            display_name="DEXTER (NIPS 2003 FS Challenge)",
            pipeline="fs",
            tier="medium",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "dexter", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 600,
                    "n_features": 20001,
                    "n_classes": 2,
                    "weights": [0.5, 0.5],
                    "difficulty": "medium",
                },
                "openml_feature_cap": 20001,
                "n_final_features": 120,
                "source_policy": "real_only",
            },
        ),
        "dorothea_nips03": DatasetSpec(
            dataset_id="dorothea_nips03",
            display_name="DOROTHEA (NIPS 2003 FS Challenge)",
            pipeline="fs",
            tier="hard",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"data_id": 4137},
                    {"name": "dorothea", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 1150,
                    "n_features": 100001,
                    "n_classes": 2,
                    # OpenML `Dorothea` v1: 1038/112 class split.
                    "weights": [1038 / 1150.0, 112 / 1150.0],
                    "difficulty": "hard",
                },
                # Dorothea is extremely sparse and very wide; cap features before densifying.
                "openml_feature_cap": 25000,
                "n_final_features": 200,
            },
        ),
        "colon_alon": DatasetSpec(
            dataset_id="colon_alon",
            display_name="Colon Cancer (Alon)",
            pipeline="fs",
            tier="medium",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "colon", "version": 1},
                    {"name": "Colon", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 62,
                    "n_features": 2000,
                    "n_classes": 2,
                    "weights": [40 / 62.0, 22 / 62.0],
                    "difficulty": "medium",
                },
                "n_final_features": 40,
            },
        ),
        "cns_pomeroy": DatasetSpec(
            dataset_id="cns_pomeroy",
            display_name="CNS / Brain Tumors (Pomeroy)",
            pipeline="fs",
            tier="medium",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "CNS", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 60,
                    "n_features": 7129,
                    "n_classes": 2,
                    # OpenML `CNS` v1: 39/21 class split.
                    "weights": [39 / 60.0, 21 / 60.0],
                    "difficulty": "medium",
                },
                "n_final_features": 45,
            },
        ),
        "lung_gordon": DatasetSpec(
            dataset_id="lung_gordon",
            # NOTE: `lung_gordon` is a legacy id retained for compatibility.
            # The automatically-loadable OpenML dataset is a 5-class lung benchmark.
            display_name="Lung Cancer (OpenML Lung, 5-class)",
            pipeline="fs",
            # Routed to hard: 5-class with a 6-sample minority class makes strict 80/20
            # holdout gate-unreliable (single-error ΔBA ≈ 0.20 when n_test(min)=1).
            tier="hard",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "lung", "version": 1},
                ],
                "synthetic_profile": {
                    # OpenML `Lung` v1: (203, 12600), 5 classes with heavy imbalance.
                    "n_samples": 203,
                    "n_features": 12600,
                    "n_classes": 5,
                    "weights": [139 / 203.0, 17 / 203.0, 6 / 203.0, 21 / 203.0, 20 / 203.0],
                    "difficulty": "hard",
                },
                "n_final_features": 60,
            },
        ),
        "breast_vantveer": DatasetSpec(
            dataset_id="breast_vantveer",
            display_name="Breast Cancer Prognosis (van 't Veer)",
            pipeline="fs",
            tier="medium",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    # Microarray-tagged OpenML dataset; matches (97, 24481) profile.
                    {"name": "Breast", "version": 3},
                ],
                "synthetic_profile": {
                    "n_samples": 97,
                    "n_features": 24481,
                    "n_classes": 2,
                    "weights": [46 / 97.0, 51 / 97.0],
                    "difficulty": "medium",
                },
                "n_final_features": 60,
            },
        ),
        "nci60_ross": DatasetSpec(
            dataset_id="nci60_ross",
            display_name="NCI60 (Ross)",
            pipeline="fs",
            tier="hard",
            loader_kind="nci60_url_or_synth",
            params={
                "synthetic_profile": {
                    "n_samples": 60,
                    "n_features": 6830,
                    "n_classes": 9,
                    "weights": [0.18, 0.15, 0.12, 0.12, 0.11, 0.1, 0.09, 0.08, 0.05],
                    "difficulty": "hard",
                },
                "n_final_features": 50,
            },
        ),
        "gcm_ramaswamy": DatasetSpec(
            dataset_id="gcm_ramaswamy",
            display_name="GCM (Ramaswamy)",
            pipeline="fs",
            tier="hard",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "GCM", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 198,
                    "n_features": 16063,
                    "n_classes": 14,
                    "weights": [
                        0.11,
                        0.1,
                        0.09,
                        0.08,
                        0.08,
                        0.07,
                        0.07,
                        0.07,
                        0.06,
                        0.06,
                        0.06,
                        0.05,
                        0.05,
                        0.05,
                    ],
                    "difficulty": "hard",
                },
                "n_final_features": 70,
            },
        ),
        "tumor11_su": DatasetSpec(
            dataset_id="tumor11_su",
            display_name="11-Tumor (Su)",
            pipeline="fs",
            tier="hard",
            loader_kind="openml_or_synth",
            params={
                "openml_options": [
                    {"name": "11_Tumors", "version": 1},
                ],
                "synthetic_profile": {
                    "n_samples": 174,
                    "n_features": 12533,
                    "n_classes": 11,
                    "weights": [
                        0.13,
                        0.12,
                        0.1,
                        0.1,
                        0.09,
                        0.09,
                        0.08,
                        0.08,
                        0.07,
                        0.07,
                        0.07,
                    ],
                    "difficulty": "hard",
                },
                "n_final_features": 65,
            },
        ),
        "nci60_strict_holdout": DatasetSpec(
            dataset_id="nci60_strict_holdout",
            display_name="NCI60 Strict Holdout",
            pipeline="fs",
            tier="very_hard",
            loader_kind="nci60_url_or_synth",
            params={
                "synthetic_profile": {
                    "n_samples": 60,
                    "n_features": 6830,
                    "n_classes": 9,
                    "weights": [0.2, 0.15, 0.13, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06],
                    "difficulty": "very_hard",
                },
                "fs_fraction": 0.4,
                "n_final_features": 24,
            },
        ),
        "cumida_leukemia_subtypes": DatasetSpec(
            dataset_id="cumida_leukemia_subtypes",
            display_name="CuMiDa Leukemia Subtypes",
            pipeline="fs",
            tier="very_hard",
            loader_kind="manual_or_synth",
            params={
                "local_path_env": "CUMIDA_LEUKEMIA_PATH",
                "default_local_path": "train_data/cumida/Leukemia_GSE28497.arff",
                "synthetic_profile": {
                    # CuMiDa Leukemia GSE28497 export shipped in-repo.
                    "n_samples": 281,
                    "n_features": 22283,
                    "n_classes": 7,
                    "weights": [0.24, 0.2, 0.16, 0.14, 0.1, 0.09, 0.07],
                    "difficulty": "very_hard",
                },
                "n_final_features": 70,
            },
        ),
        "cumida_brain_gse50161": DatasetSpec(
            dataset_id="cumida_brain_gse50161",
            display_name="CuMiDa Brain Subtypes (GSE50161)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "local_path_env": "CUMIDA_BRAIN_PATH",
                "default_local_path": "train_data/cumida/Brain_GSE50161.arff",
                # Real-data profile reference:
                # 108 valid samples, 4 classes (ependymoma=46, glioblastoma=34,
                # medulloblastoma=22, normal=6).
                "synthetic_profile": {
                    "n_samples": 108,
                    "n_features": 54675,
                    "n_classes": 4,
                    "weights": [46 / 108.0, 34 / 108.0, 22 / 108.0, 6 / 108.0],
                    "difficulty": "hard",
                },
                "n_final_features": 120,
            },
        ),
        "cumida_breast_gse45827": DatasetSpec(
            dataset_id="cumida_breast_gse45827",
            display_name="CuMiDa Breast Subtypes (GSE45827)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "local_path_env": "CUMIDA_BREAST_PATH",
                "default_local_path": "train_data/cumida/Breast_GSE45827.arff",
                "synthetic_profile": {
                    # CuMiDa Breast GSE45827: 151 samples, 54,675 probes, 6 classes
                    # (basal, HER, cell_line, normal, luminal_A, luminal_B).
                    "n_samples": 151,
                    "n_features": 54675,
                    "n_classes": 6,
                    # Class counts from CuMiDa `Breast_GSE45827_classes.txt`:
                    # basal=41, HER=30, cell_line=14, normal=7, luminal_A=29, luminal_B=30.
                    "weights": [
                        41 / 151.0,
                        30 / 151.0,
                        14 / 151.0,
                        7 / 151.0,
                        29 / 151.0,
                        30 / 151.0,
                    ],
                    "difficulty": "hard",
                },
                "n_final_features": 140,
            },
        ),
        # ---------------------------------------------------------------
        # Extended-only CuMiDa datasets (not in original validation bundle)
        # Each uses a pre-compiled NPZ file in train_data/extended/
        # ---------------------------------------------------------------
        "cumida_prostate_gse6919": DatasetSpec(
            dataset_id="cumida_prostate_gse6919",
            display_name="CuMiDa Prostate (GSE6919)",
            pipeline="fs",
            tier="medium",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/cumida_prostate_gse6919.npz",
                "synthetic_profile": {
                    "n_samples": 171,
                    "n_features": 12625,
                    "n_classes": 2,
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
            extended_only=True,
        ),
        "cumida_ovarian_gse26712": DatasetSpec(
            dataset_id="cumida_ovarian_gse26712",
            display_name="CuMiDa Ovarian (GSE26712)",
            pipeline="fs",
            tier="medium",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/cumida_ovarian_gse26712.npz",
                "synthetic_profile": {
                    "n_samples": 195,
                    "n_features": 22283,
                    "n_classes": 2,
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
            extended_only=True,
        ),
        "cumida_lung_gse19804": DatasetSpec(
            dataset_id="cumida_lung_gse19804",
            display_name="CuMiDa Lung (GSE19804)",
            pipeline="fs",
            tier="medium",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/cumida_lung_gse19804.npz",
                "synthetic_profile": {
                    "n_samples": 120,
                    "n_features": 54675,
                    "n_classes": 2,
                    "difficulty": "medium",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "cumida_colorectal_gse44861": DatasetSpec(
            dataset_id="cumida_colorectal_gse44861",
            display_name="CuMiDa Colorectal (GSE44861)",
            pipeline="fs",
            tier="medium",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/cumida_colorectal_gse44861.npz",
                "synthetic_profile": {
                    "n_samples": 111,
                    "n_features": 22277,
                    "n_classes": 2,
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
            extended_only=True,
        ),
        "cumida_gastric_gse54129": DatasetSpec(
            dataset_id="cumida_gastric_gse54129",
            display_name="CuMiDa Gastric (GSE54129)",
            pipeline="fs",
            tier="easy",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/cumida_gastric_gse54129.npz",
                "synthetic_profile": {
                    "n_samples": 132,
                    "n_features": 54675,
                    "n_classes": 2,
                    "difficulty": "easy",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "cumida_pancreatic_gse16515": DatasetSpec(
            dataset_id="cumida_pancreatic_gse16515",
            display_name="CuMiDa Pancreatic (GSE16515)",
            pipeline="fs",
            tier="medium",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/cumida_pancreatic_gse16515.npz",
                "synthetic_profile": {
                    "n_samples": 52,
                    "n_features": 54613,
                    "n_classes": 2,
                    "difficulty": "medium",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "cumida_renal_gse53757": DatasetSpec(
            dataset_id="cumida_renal_gse53757",
            display_name="CuMiDa Renal (GSE53757)",
            pipeline="fs",
            tier="medium",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/cumida_renal_gse53757.npz",
                "synthetic_profile": {
                    "n_samples": 144,
                    "n_features": 54675,
                    "n_classes": 2,
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
            extended_only=True,
        ),
        "cumida_headneck_gse12452": DatasetSpec(
            dataset_id="cumida_headneck_gse12452",
            display_name="CuMiDa Head/Neck (GSE12452)",
            pipeline="fs",
            tier="medium",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/cumida_headneck_gse12452.npz",
                "synthetic_profile": {
                    "n_samples": 41,
                    "n_features": 54675,
                    "n_classes": 2,
                    "difficulty": "medium",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        # ---------------------------------------------------------------
        # Extended-only UCSC Xena / TCGA datasets (gene expression, tabular)
        # Pre-compiled NPZ files in train_data/extended/ from UCSC Xena TCGA hub
        # ---------------------------------------------------------------
        "xena_tcga_brca": DatasetSpec(
            dataset_id="xena_tcga_brca",
            display_name="TCGA-BRCA Breast Cancer (Xena, PAM50 5-subtype)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_brca.npz",
                "synthetic_profile": {
                    "n_samples": 956,
                    "n_features": 20530,
                    "n_classes": 5,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "xena_tcga_luad": DatasetSpec(
            dataset_id="xena_tcga_luad",
            display_name="TCGA-LUAD Lung Adenocarcinoma (Xena, 3-subtype)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_luad.npz",
                "synthetic_profile": {
                    "n_samples": 275,
                    "n_features": 20530,
                    "n_classes": 3,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "xena_tcga_ucec": DatasetSpec(
            dataset_id="xena_tcga_ucec",
            display_name="TCGA-UCEC Uterine (Xena, histological 3-type)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_ucec.npz",
                "synthetic_profile": {
                    "n_samples": 190,
                    "n_features": 20530,
                    "n_classes": 3,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "xena_tcga_lgg": DatasetSpec(
            dataset_id="xena_tcga_lgg",
            display_name="TCGA-LGG Lower Grade Glioma (Xena, 3-subtype)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_lgg.npz",
                "synthetic_profile": {
                    "n_samples": 529,
                    "n_features": 20530,
                    "n_classes": 3,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "xena_tcga_kirc": DatasetSpec(
            dataset_id="xena_tcga_kirc",
            display_name="TCGA-KIRC Kidney Clear Cell (Xena, stage-based)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_kirc.npz",
                "synthetic_profile": {
                    "n_samples": 606,
                    "n_features": 20530,
                    "n_classes": 4,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        # ---------------------------------------------------------------
        # Extended-only UCSC Xena / TCGA expansion (Phase 2)
        # ---------------------------------------------------------------
        "xena_tcga_hnsc_hpv": DatasetSpec(
            dataset_id="xena_tcga_hnsc_hpv",
            display_name="TCGA-HNSC Head & Neck (Xena, HPV binary)",
            pipeline="fs",
            tier="medium",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_hnsc_hpv.npz",
                "synthetic_profile": {
                    "n_samples": 114,
                    "n_features": 20530,
                    "n_classes": 2,
                    "difficulty": "medium",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "xena_tcga_skcm": DatasetSpec(
            dataset_id="xena_tcga_skcm",
            display_name="TCGA-SKCM Melanoma (Xena, primary vs metastatic)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_skcm.npz",
                "synthetic_profile": {
                    "n_samples": 472,
                    "n_features": 20530,
                    "n_classes": 2,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "xena_tcga_gbm": DatasetSpec(
            dataset_id="xena_tcga_gbm",
            display_name="TCGA-GBM Glioblastoma (Xena, 4-subtype)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_gbm.npz",
                "synthetic_profile": {
                    "n_samples": 164,
                    "n_features": 20530,
                    "n_classes": 4,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "xena_tcga_stad": DatasetSpec(
            dataset_id="xena_tcga_stad",
            display_name="TCGA-STAD Stomach (Xena, histological 3-group)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_stad.npz",
                "synthetic_profile": {
                    "n_samples": 448,
                    "n_features": 20530,
                    "n_classes": 3,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "xena_tcga_lihc": DatasetSpec(
            dataset_id="xena_tcga_lihc",
            display_name="TCGA-LIHC Liver (Xena, histologic grade 3-class)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_lihc.npz",
                "synthetic_profile": {
                    "n_samples": 415,
                    "n_features": 20530,
                    "n_classes": 3,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "xena_tcga_ov": DatasetSpec(
            dataset_id="xena_tcga_ov",
            display_name="TCGA-OV Ovarian (Xena, histologic grade G2 vs G3)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_ov.npz",
                "synthetic_profile": {
                    "n_samples": 299,
                    "n_features": 20530,
                    "n_classes": 2,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "xena_tcga_prad": DatasetSpec(
            dataset_id="xena_tcga_prad",
            display_name="TCGA-PRAD Prostate (Xena, Gleason score)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_prad.npz",
                "synthetic_profile": {
                    "n_samples": 550,
                    "n_features": 20530,
                    "n_classes": 5,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "xena_tcga_coad_cms": DatasetSpec(
            dataset_id="xena_tcga_coad_cms",
            display_name="TCGA-COAD Colorectal (Xena, histological 2-class)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/extended/xena_tcga_coad_cms.npz",
                "synthetic_profile": {
                    "n_samples": 323,
                    "n_features": 20530,
                    "n_classes": 2,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),

        # ── Results-validation datasets (rv_ prefix) ────────────────────────
        # Independent hold-out catalog for final results validation.
        # Not used in development tuning; see VALIDATION_RESULTS.md.

        # Group A: HDLSS small-n (scikit-feature)
        "rv_warpar10p": DatasetSpec(
            dataset_id="rv_warpar10p",
            display_name="warpAR10P (Scikit-feature, Face)",
            pipeline="fs",
            tier="medium",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/warpAR10P.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/warpAR10P.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 130,
                    "n_features": 2400,
                    "n_classes": 10,
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
            extended_only=True,
        ),
        "rv_yale": DatasetSpec(
            dataset_id="rv_yale",
            display_name="Yale Faces (Scikit-feature)",
            pipeline="fs",
            tier="hard",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/Yale.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/Yale.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 165,
                    "n_features": 1024,
                    "n_classes": 15,
                    "difficulty": "hard",
                },
                "n_final_features": 60,
            },
            extended_only=True,
        ),
        "rv_lung_discrete": DatasetSpec(
            dataset_id="rv_lung_discrete",
            display_name="Lung Discrete (Scikit-feature)",
            pipeline="fs",
            tier="hard",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/lung_discrete.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/lung_discrete.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 73,
                    "n_features": 325,
                    "n_classes": 7,
                    "difficulty": "hard",
                },
                "n_final_features": 50,
            },
            extended_only=True,
        ),

        # Group B: Moderate-scale datasets (scikit-feature)
        "rv_coil20": DatasetSpec(
            dataset_id="rv_coil20",
            display_name="COIL20 Objects (Scikit-feature)",
            pipeline="fs",
            tier="medium",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/COIL20.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/COIL20.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 1440,
                    "n_features": 1024,
                    "n_classes": 20,
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
            extended_only=True,
        ),
        "rv_relathe": DatasetSpec(
            dataset_id="rv_relathe",
            display_name="RELATHE Text (Scikit-feature)",
            pipeline="fs",
            tier="medium",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/RELATHE.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/RELATHE.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 1427,
                    "n_features": 4322,
                    "n_classes": 2,
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
            extended_only=True,
        ),
        "rv_pcmac": DatasetSpec(
            dataset_id="rv_pcmac",
            display_name="PCMAC Text (Scikit-feature)",
            pipeline="fs",
            tier="medium",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/PCMAC.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/PCMAC.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 1943,
                    "n_features": 3289,
                    "n_classes": 2,
                    "difficulty": "medium",
                },
                "n_final_features": 80,
            },
            extended_only=True,
        ),
        "rv_isolet": DatasetSpec(
            dataset_id="rv_isolet",
            display_name="ISOLET Audio (Scikit-feature)",
            pipeline="fs",
            tier="hard",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/Isolet.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/Isolet.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 1560,
                    "n_features": 617,
                    "n_classes": 26,
                    "difficulty": "hard",
                },
                "n_final_features": 80,
            },
            extended_only=True,
        ),
        "rv_basehock": DatasetSpec(
            dataset_id="rv_basehock",
            display_name="BASEHOCK Text (Scikit-feature)",
            pipeline="fs",
            tier="easy",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/BASEHOCK.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/BASEHOCK.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 1993,
                    "n_features": 4862,
                    "n_classes": 2,
                    "difficulty": "easy",
                },
                "n_final_features": 80,
            },
            extended_only=True,
        ),

        # Group C: Large-scale dataset (scikit-feature)
        "rv_usps": DatasetSpec(
            dataset_id="rv_usps",
            display_name="USPS Handwritten Digits (Scikit-feature)",
            pipeline="fs",
            tier="easy",
            loader_kind="mat_url_or_synth",
            params={
                "mat_url_options": [
                    {
                        "url": "https://jundongl.github.io/scikit-feature/OLD/datasets/USPS.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                    {
                        "url": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/USPS.mat",
                        "x_key": "X",
                        "y_key": "Y",
                    },
                ],
                "synthetic_profile": {
                    "n_samples": 9298,
                    "n_features": 256,
                    "n_classes": 10,
                    "difficulty": "easy",
                },
                "n_final_features": 60,
            },
            extended_only=True,
        ),

        # Group D: UCSC Xena / TCGA RNA-seq (manual NPZ)
        "rv_xena_tcga_thca": DatasetSpec(
            dataset_id="rv_xena_tcga_thca",
            display_name="TCGA-THCA Thyroid Cancer (Xena, 3-subtype)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/results_validation/xena_tcga_thca.npz",
                "synthetic_profile": {
                    "n_samples": 500,
                    "n_features": 20530,
                    "n_classes": 3,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "rv_xena_tcga_blca": DatasetSpec(
            dataset_id="rv_xena_tcga_blca",
            display_name="TCGA-BLCA Bladder Cancer (Xena, 5-subtype)",
            pipeline="fs",
            tier="hard",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/results_validation/xena_tcga_blca.npz",
                "synthetic_profile": {
                    "n_samples": 400,
                    "n_features": 20530,
                    "n_classes": 5,
                    "difficulty": "hard",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
        "rv_xena_tcga_cesc": DatasetSpec(
            dataset_id="rv_xena_tcga_cesc",
            display_name="TCGA-CESC Cervical Cancer (Xena, 2-class)",
            pipeline="fs",
            tier="medium",
            loader_kind="manual_or_synth",
            params={
                "default_local_path": "train_data/results_validation/xena_tcga_cesc.npz",
                "synthetic_profile": {
                    "n_samples": 300,
                    "n_features": 20530,
                    "n_classes": 2,
                    "difficulty": "medium",
                },
                "n_final_features": 100,
            },
            extended_only=True,
        ),
    }

    df_specs = {
        "df_synthetic_parametric": DatasetSpec(
            dataset_id="df_synthetic_parametric",
            display_name="Synthetic Controlled Parametric",
            pipeline="df",
            tier="easy",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "synthetic_parametric"},
        ),
        "df_actuarial_losses": DatasetSpec(
            dataset_id="df_actuarial_losses",
            display_name="Actuarial Loss Distributions",
            pipeline="df",
            tier="easy",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "actuarial"},
        ),
        "df_reliability_survival": DatasetSpec(
            dataset_id="df_reliability_survival",
            display_name="Reliability / Survival",
            pipeline="df",
            tier="easy",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "reliability"},
        ),
        "df_financial_returns": DatasetSpec(
            dataset_id="df_financial_returns",
            display_name="Financial Returns",
            pipeline="df",
            tier="medium",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "financial"},
        ),
        "df_hydrology_rainfall": DatasetSpec(
            dataset_id="df_hydrology_rainfall",
            display_name="Hydrology / Rainfall",
            pipeline="df",
            tier="medium",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "hydrology"},
        ),
        "df_internet_traffic": DatasetSpec(
            dataset_id="df_internet_traffic",
            display_name="Internet Traffic / File Sizes",
            pipeline="df",
            tier="medium",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "internet"},
        ),
        "df_contaminated_samples": DatasetSpec(
            dataset_id="df_contaminated_samples",
            display_name="Contaminated / Outlier-Heavy Samples",
            pipeline="df",
            tier="hard",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "contaminated"},
        ),
        "df_heaped_data": DatasetSpec(
            dataset_id="df_heaped_data",
            display_name="Heaped / Rounded Data",
            pipeline="df",
            tier="hard",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "heaped"},
        ),
        "df_tail_discrimination": DatasetSpec(
            dataset_id="df_tail_discrimination",
            display_name="Short vs Long Tail Discrimination",
            pipeline="df",
            tier="hard",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "tail_discrimination"},
        ),
        "df_near_symmetric_small_n": DatasetSpec(
            dataset_id="df_near_symmetric_small_n",
            display_name="Near-Symmetric Heavy-Tailed (Small n)",
            pipeline="df",
            tier="very_hard",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "near_symmetric"},
        ),
        "df_out_of_library": DatasetSpec(
            dataset_id="df_out_of_library",
            display_name="Misspecified / Out-of-Library",
            pipeline="df",
            tier="very_hard",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "out_of_library"},
        ),
        "df_multimodal_mixtures": DatasetSpec(
            dataset_id="df_multimodal_mixtures",
            display_name="Multi-Modal / Mixtures",
            pipeline="df",
            tier="very_hard",
            loader_kind="dist_benchmark",
            source_kind="dist_benchmark",
            params={"profile": "mixtures"},
        ),
    }

    integrated_specs = {
        "int_cdf_leukemia": DatasetSpec(
            dataset_id="int_cdf_leukemia",
            display_name="Integrated: Leukemia with CDF Transform",
            pipeline="integrated",
            tier="easy",
            loader_kind="integrated",
            params={"base_dataset": "leukemia_golub", "scenario": "cdf_transform"},
        ),
        "int_cdf_colon": DatasetSpec(
            dataset_id="int_cdf_colon",
            display_name="Integrated: Colon with CDF Transform",
            pipeline="integrated",
            tier="medium",
            loader_kind="integrated",
            params={"base_dataset": "colon_alon", "scenario": "cdf_transform"},
        ),
        "int_cdf_nci60": DatasetSpec(
            dataset_id="int_cdf_nci60",
            display_name="Integrated: NCI60 with CDF Transform",
            pipeline="integrated",
            tier="hard",
            loader_kind="integrated",
            params={"base_dataset": "nci60_ross", "scenario": "cdf_transform"},
        ),
        "int_low_gof_downweighting": DatasetSpec(
            dataset_id="int_low_gof_downweighting",
            display_name="Integrated: Low-GOF Downweighting",
            pipeline="integrated",
            tier="hard",
            loader_kind="integrated",
            params={"base_dataset": "colon_alon", "scenario": "low_gof_downweighting"},
        ),
        "int_distribution_stability_confidence": DatasetSpec(
            dataset_id="int_distribution_stability_confidence",
            display_name="Integrated: Distribution Stability Confidence",
            pipeline="integrated",
            tier="very_hard",
            loader_kind="integrated",
            params={"base_dataset": "nci60_ross", "scenario": "stability_signal"},
        ),
    }

    catalog: Dict[str, DatasetSpec] = {}
    catalog.update(fs_specs)
    catalog.update(df_specs)
    catalog.update(integrated_specs)
    return catalog

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

# strict_holdout vs inflated protocol families (source: VALIDATION.md)
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
    "tumor11_su": {"strict_holdout": (0.55, 0.80), "inflated": (0.65, 0.98)},
    "lymphoma_9": {"strict_holdout": (0.50, 0.70), "inflated": (0.65, 0.85)},
    "lymphoma_11": {"strict_holdout": (0.45, 0.65), "inflated": (0.60, 0.80)},
    "hf_breast_ge_mubashir1837": {"strict_holdout": (0.50, 0.70), "inflated": (0.75, 0.89)},
    "cumida_brain_gse50161": {"strict_holdout": (0.88, 0.98), "inflated": (0.92, 1.00)},
    "cumida_breast_gse45827": {"strict_holdout": (0.82, 0.95), "inflated": (0.90, 0.98)},
    "cumida_leukemia_subtypes": {"strict_holdout": (0.85, 0.97), "inflated": (0.95, 1.00)},
    "nci60_ross": {"strict_holdout": (0.40, 0.55), "inflated": (0.60, 0.79)},
    "nci60_strict_holdout": {"strict_holdout": (0.30, 0.50), "inflated": (0.60, 0.79)},
    "tumor9_openml": {"strict_holdout": (0.45, 0.65), "inflated": (0.65, 0.84)},
    "cll_sub_111": {"strict_holdout": (0.78, 0.90), "inflated": (0.85, 0.95)},
    "tox_171": {"strict_holdout": (0.84, 0.93), "inflated": (0.93, 0.98)},
    "gla_bra_180": {"strict_holdout": (0.58, 0.72), "inflated": (0.68, 0.89)},
    "carcinom_11class": {"strict_holdout": (0.72, 0.88), "inflated": (0.90, 0.98)},
    "glioma_50_4class": {"strict_holdout": (0.88, 0.97), "inflated": (0.96, 1.00)},
    "brain_tumor_2_50_4class": {"strict_holdout": (0.86, 0.95), "inflated": (0.95, 1.00)},
    "leukemia_1_72_3class": {"strict_holdout": (0.85, 0.96), "inflated": (0.95, 1.00)},
    "nci9_60_9class": {"strict_holdout": (0.50, 0.70), "inflated": (0.66, 0.81)},
    "nci_61_8class": {"strict_holdout": (0.50, 0.68), "inflated": (0.65, 0.75)},
    "orlraws10p": {"strict_holdout": (0.95, 1.00), "inflated": (0.98, 1.00)},
    "warp_pie10p": {"strict_holdout": (0.95, 1.00), "inflated": (0.98, 1.00)},
    "pixraw10p": {"strict_holdout": (0.96, 1.00), "inflated": (0.99, 1.00)},
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

def sota_ranges_for_dataset(dataset_id: str, tier: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ds_id = str(dataset_id)
    tier_norm = str(tier).strip().lower()
    ranges = KNOWN_SOTA_PROTOCOL_RANGES.get(ds_id)
    if ranges is None:
        strict = TIER_SOTA_DEFAULTS_STRICT.get(tier_norm, (0.0, 1.0))
        inflated = TIER_SOTA_DEFAULTS_INFLATED.get(tier_norm, strict)
        return strict, inflated
    strict = tuple(ranges.get("strict_holdout") or TIER_SOTA_DEFAULTS_STRICT.get(tier_norm, (0.0, 1.0)))
    inflated = tuple(ranges.get("inflated") or TIER_SOTA_DEFAULTS_INFLATED.get(tier_norm, strict))
    return (float(strict[0]), float(strict[1])), (float(inflated[0]), float(inflated[1]))

def infer_domain(spec: DatasetSpec) -> str:
    ds_id = str(spec.dataset_id).strip().lower()
    kind = str(spec.loader_kind)
    if str(spec.source_kind) == "synthetic":
        return "synthetic"
    if kind == "face_proxy_or_synth":
        return "non_genomic"
    if ds_id.endswith("_nips03"):
        return "non_genomic"
    explicit = str((spec.params or {}).get("domain", "")).strip().lower()
    if explicit in {"genomics", "non_genomic", "synthetic"}:
        return explicit
    return "genomics"


def infer_platform(spec: DatasetSpec) -> str:
    ds_id = str(spec.dataset_id).strip().lower()
    params = dict(spec.params or {})

    explicit = str(params.get("platform", "") or "").strip()
    if explicit:
        return explicit

    if str(spec.source_kind) == "synthetic":
        return "synthetic"
    if ds_id == "ovarian_petricoin":
        return "mass-spec"

    affy_hgu95 = {
        "leukemia_golub",
        "dlbcl_shipp",
        "prostate_singh",
        "mll_microarray",
        "nci60_ross",
        "nci60_strict_holdout",
    }
    if ds_id in affy_hgu95:
        return "Affy HG-U95"

    affy_hgu133a = {
        "colon_alon",
        "gli_85",
        "smk_can_187",
        "lymphoma_3",
        "lung_gordon",
        "gcm_ramaswamy",
        "tumor11_su",
        "lymphoma_9",
        "lymphoma_11",
    }
    if ds_id in affy_hgu133a:
        return "Affy HG-U133A"

    return "cDNA"

def infer_provenance(spec: DatasetSpec) -> str:
    kind = str(spec.loader_kind)
    params = dict(spec.params or {})
    if kind == "openml_or_synth":
        opts = list(params.get("openml_options", []) or [])
        if opts:
            return f"openml:{opts[0]}"
        return "openml"
    if kind in {"mat_url_or_synth", "tab_url_or_synth"}:
        key = "mat_url_options" if kind == "mat_url_or_synth" else "tab_url_options"
        opts = list(params.get(key, []) or [])
        if opts and isinstance(opts[0], dict) and opts[0].get("url"):
            return str(opts[0].get("url"))
        return kind
    if kind in {"nci60_url_or_synth", "nci60_proxy_or_synth"}:
        return kind
    if kind == "face_proxy_or_synth":
        return "sklearn:olivetti_faces"
    if kind == "manual_or_synth":
        env_name = params.get("local_path_env")
        if env_name:
            return f"env:{env_name}"
        default_path = params.get("default_local_path")
        if default_path:
            return f"local:{default_path}"
        return "manual"
    if kind == "integrated":
        base = params.get("base_dataset")
        return f"integrated_base:{base}"
    if kind == "dist_benchmark":
        return "dist_benchmark"
    return kind

def _finalize_spec(spec: DatasetSpec) -> DatasetSpec:
    prov = infer_provenance(spec)
    dom = infer_domain(spec)
    platform = infer_platform(spec)
    out = replace(spec, provenance=str(prov), domain=str(dom), platform=str(platform))
    if out.pipeline in {"fs", "integrated"} and out.source_kind in {"synthetic", "validation_catalog"}:
        sota_id = out.dataset_id
        if out.pipeline == "integrated":
            # Keep benchmark behavior stable: integrated datasets inherit their parent
            # dataset's SOTA ranges when available (per VALIDATION.md / benchmark runner).
            base = None
            try:
                base = out.params.get("base_dataset")
            except Exception as exc:
                base = None
            base = str(base).strip() if base else ""
            if base and base in KNOWN_SOTA_PROTOCOL_RANGES:
                sota_id = base

        strict, inflated = sota_ranges_for_dataset(sota_id, out.tier)
        out = replace(out, sota_holdout_bal_acc=strict, sota_inflated_bal_acc=inflated)
    return out

def _build_registry() -> Dict[str, DatasetSpec]:
    reg = _build_dataset_registry()
    # Synthetic benchmark datasets (from run_df_fs_sota_benchmark.py)
    reg.update({
        "synthetic_easy_dfshift": DatasetSpec(
            dataset_id="synthetic_easy_dfshift",
            display_name="Synthetic Easy DF-Shift",
            pipeline="fs",
            tier="easy",
            loader_kind="synthetic_benchmark",
            params={
                "fs_fraction": 0.40,
                "n_final_features": 40,
                "notes": "High-separation synthetic with class-conditional marginal shifts.",
            },
            source_kind="synthetic",
        ),
        "synthetic_medium_mixed": DatasetSpec(
            dataset_id="synthetic_medium_mixed",
            display_name="Synthetic Medium Mixed",
            pipeline="fs",
            tier="medium",
            loader_kind="synthetic_benchmark",
            params={
                "fs_fraction": 0.40,
                "n_final_features": 50,
                "notes": "Moderate overlap with skew/heavy-tail perturbations.",
            },
            source_kind="synthetic",
        ),
        "synthetic_very_hard_sparse": DatasetSpec(
            dataset_id="synthetic_very_hard_sparse",
            display_name="Synthetic Very Hard Sparse",
            pipeline="fs",
            tier="very_hard",
            loader_kind="synthetic_benchmark",
            params={
                "fs_fraction": 0.40,
                "n_final_features": 60,
                "notes": "Multiclass sparse-signal problem with contamination/heaping.",
            },
            source_kind="synthetic",
        ),
    })
    return {k: _finalize_spec(v) for k, v in reg.items()}

DATASET_REGISTRY: Dict[str, DatasetSpec] = _build_registry()

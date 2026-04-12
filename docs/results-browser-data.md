---
title: Browser Data Guide
nav_order: 7
---

This page summarizes the public data currently exposed through the [Results Browser](results-browser.md).

The current HDLSS browser bundle exposes **54,882** run rows across **271** profiles and **63** datasets.

Those rows are surfaced in the browser as four linked tables: dataset summaries, profile summaries, dataset/profile aggregates, and the underlying per-seed run explorer.

## Campaign Scope Filter

The browser scope filter is driven by campaign-level dataset coverage. Broad-panel campaigns stay in the full view, while diagnostic-only campaigns stay in the diagnostic view.

| Filter | Meaning |
| --- | --- |
| All campaigns | Every campaign/profile slice currently exposed by the public HDLSS benchmark bundle. |
| Full benchmark panel | Profiles evaluated across the broad public benchmark panel (typically 62-63 datasets). |
| Diagnostic subset | Profiles evaluated on the 24-dataset diagnostic subset used for narrower probes and weighting checks. |

`Val-20 Pipeline` remains in the full-panel group because the campaign as a whole publishes against the broad benchmark surface, even though some profiles are narrower probes.

## Campaigns

| Campaign | Code | Scope | Profiles | Datasets | Families | What it covers |
| --- | --- | --- | --- | --- | --- | --- |
| Val-18 Anchor | `val18_anchors` | Full benchmark panel | 8 | 63 | Anchor | Baseline and bypass anchor controls across the broad benchmark panel. |
| Val-18 Classifier | `val18_classifiers` | Full benchmark panel | 40 | 63 | Clf Only, Pool, Stage | Classifier-pool and classifier-ablation runs across the broad benchmark panel. |
| Val-18 Clf Oracle | `val18_cls_oracle_wt` | Diagnostic subset | 9 | 24 | Clf Oracle | Classifier-oracle weighting diagnostics on the diagnostic subset. |
| Val-18 FS | `val18_singletons` | Full benchmark panel | 78 | 63 | FS RAW, FS SCAF | Raw and scaffolded feature-selection singletons across the broad benchmark panel. |
| Val-18 MNPO | `val18_mnpo` | Full benchmark panel | 23 | 63 | Oracle | MNPO oracle weighting and ablation runs across the broad benchmark panel. |
| Val-18 Stage | `val18_stage` | Diagnostic subset | 37 | 24 | DistFit, Stage, V18 Prefilter | Pipeline stage-ordering probes on the diagnostic subset. |
| Val-19 Bridge | `val19_classifiers` | Full benchmark panel | 13 | 63 | Clf Only, V19 Bridge | Val-19 bridge specialists and classifier promotions across the broad benchmark panel. |
| Val-19 Random FS | `val19_random_fs` | Full benchmark panel | 2 | 63 | FS RAW, FS SCAF | Randomized feature-selection controls across the broad benchmark panel. |
| Val-20 Bridge | `val20_bridge` | Full benchmark panel | 6 | 63 | V20 Bridge | Val-20 bridge reruns that align anchor and promotion candidates on the broad benchmark panel. |
| Val-20 Ensemble | `val20_ensemble` | Diagnostic subset | 13 | 24 | V20 Ensemble, V20 Oracle Diag | Diagnostic-subset ensemble extraction and voting probes. |
| Val-20 FLAML | `val20_flaml` | Diagnostic subset | 22 | 24 | V20 FLAML | Diagnostic-subset FLAML and frontier tuning probes. |
| Val-20 LR Mitigation | `val20_lr` | Full benchmark panel | 3 | 62 | V20 LR Mitigation | Broad-panel logistic-regression overselection countermeasures. |
| Val-20 Pipeline | `val20_pipeline` | Full benchmark panel | 6 | 63 | V18 Prefilter | Pipeline simplification probes that span both the broad benchmark panel and the diagnostic subset. |
| Val-20 TabPFN | `val20_tabpfn` | Full benchmark panel | 3 | 63 | V20 TabPFN | Broad-panel TabPFN gating checks. |
| Val-20 Tune-First | `val20_tune_first` | Diagnostic subset | 8 | 24 | V20 Tune-First | Diagnostic-subset tune-first classifier-selection probes. |

## Families

| Family | Code | Profiles | Datasets | Scope | Campaigns | What it isolates |
| --- | --- | --- | --- | --- | --- | --- |
| Anchor | `A` | 8 | 63 | Full benchmark panel | Val-18 Anchor | Baseline and bypass anchor controls. |
| Clf Only | `C_ONLY` | 30 | 24 | Full benchmark panel | Val-18 Classifier, Val-19 Bridge | Single-classifier bridge and specialist promotions. |
| Clf Oracle | `W` | 9 | 24 | Diagnostic subset | Val-18 Clf Oracle | Classifier-oracle reweighting diagnostics. |
| DistFit | `D` | 19 | 24 | Diagnostic subset | Val-18 Stage | Distribution-fitting and CDF-transform variants. |
| FS RAW | `M_RAW` | 40 | 63 | Full benchmark panel | Val-18 FS, Val-19 Random FS | Raw feature-selection singleton runs. |
| FS SCAF | `M_SCAFFOLD` | 40 | 63 | Full benchmark panel | Val-18 FS, Val-19 Random FS | Scaffolded feature-selection singleton runs. |
| Oracle | `N` | 23 | 63 | Full benchmark panel | Val-18 MNPO | MNPO oracle weighting and ablation profiles. |
| Pool | `C` | 12 | 63 | Full benchmark panel | Val-18 Classifier | Classifier-pool comparisons and ensemble candidate pools. |
| Stage | `S` | 7 | 24 | Full benchmark panel, Diagnostic subset | Val-18 Classifier, Val-18 Stage | Stage-ordering probes. |
| V18 Prefilter | `P` | 22 | 63 | Full benchmark panel, Diagnostic subset | Val-18 Stage, Val-20 Pipeline | Prefilter and dimensionality-folding probes. |
| V19 Bridge | `V` | 6 | 63 | Full benchmark panel | Val-19 Bridge | Val-19 bridge reruns and promotions. |
| V20 Bridge | `B` | 6 | 63 | Full benchmark panel | Val-20 Bridge | Public browser family in the benchmark bundle. |
| V20 Ensemble | `E` | 7 | 24 | Diagnostic subset | Val-20 Ensemble | Val-20 ensemble extraction and voting variants. |
| V20 FLAML | `F` | 22 | 24 | Diagnostic subset | Val-20 FLAML | Val-20 FLAML and tuned promotion probes. |
| V20 LR Mitigation | `L` | 3 | 62 | Full benchmark panel | Val-20 LR Mitigation | Val-20 logistic-regression mitigation profiles. |
| V20 Oracle Diag | `O` | 6 | 24 | Diagnostic subset | Val-20 Ensemble | Public browser family in the benchmark bundle. |
| V20 TabPFN | `T` | 3 | 63 | Full benchmark panel | Val-20 TabPFN | Val-20 TabPFN gating checks. |
| V20 Tune-First | `TF` | 8 | 24 | Diagnostic subset | Val-20 Tune-First | Val-20 tune-first bridge profiles. |

## Datasets

| Dataset | Tier | Domain | Platform | Samples / Features | Current top profile | SOTA band |
| --- | --- | --- | --- | --- | --- | --- |
| 11-Tumor (Su) (`tumor11_su`) | Hard | genomics | Affy HG-U133A | 174 / 12,533 | N01_core_legacy_voting | 0.55-0.80 (above) |
| 9-Tumors (OpenML) (`tumor9_openml`) | Hard | genomics | cDNA | 60 / 5,726 | V19_C03_old_regime_mnpo_full64 | 0.45-0.65 (above) |
| ARCENE (NIPS 2003 FS Challenge) (`arcene_nips03`) | Medium | non_genomic | cDNA | 200 / 10,000 | P12_fold_tensor_sketch | 0.75-0.89 (above) |
| BASEHOCK Text (Scikit-feature) (`rv_basehock`) | Easy | genomics | cDNA | 1,993 / 4,862 | P14_lowpn_fast_filter | 0.90-0.97 (within) |
| Brain Tumor 2 — Nutt 2003 (50 x 12625, 4-class) (`brain_tumor_2_50_4class`) | Medium | genomics | cDNA | 50 / 12,625 | V19_C03_old_regime_mnpo_full64 | 0.86-0.95 (above) |
| Breast Cancer Gene Expression (HF: mubashir1837) (`hf_breast_ge_mubashir1837`) | Hard | genomics | cDNA | 51 / 28,278 | A01_simple_anchor_after_fs | 0.50-0.70 (above) |
| Breast Cancer Prognosis (van 't Veer) (`breast_vantveer`) | Medium | genomics | cDNA | 97 / 24,481 | N07_core_banzhaf_no_payoff | 0.75-0.87 (within) |
| Carcinom 11-class (Scikit-feature) (`carcinom_11class`) | Hard | genomics | cDNA | 174 / 9,182 | N32_core_banzhaf_perf_complex_stab | 0.72-0.88 (above) |
| CLL_SUB_111 (Scikit-feature) (`cll_sub_111`) | Medium | genomics | cDNA | 111 / 11,340 | N41_core_banzhaf_perf_only_5x5 | 0.78-0.90 (above) |
| CNS / Brain Tumors (Pomeroy) (`cns_pomeroy`) | Medium | genomics | cDNA | 60 / 7,129 | M_SCAFFOLD_boruta | 0.74-0.86 (within) |
| Colon Cancer (Alon) (`colon_alon`) | Medium | genomics | Affy HG-U133A | 62 / 2,000 | C02_pool_legacy_plus_tabpfn | 0.80-0.93 (above) |
| CuMiDa Brain Subtypes (GSE50161) (`cumida_brain_gse50161`) | Hard | genomics | cDNA | 108 / 54,675 | A01_simple_anchor_after_fs | 0.88-0.98 (above) |
| CuMiDa Breast Subtypes (GSE45827) (`cumida_breast_gse45827`) | Hard | genomics | cDNA | 151 / 54,675 | A02_default_anchor_after_fs | 0.82-0.95 (above) |
| CuMiDa Colorectal (GSE44861) (`cumida_colorectal_gse44861`) | Medium | genomics | cDNA | 111 / 22,277 | A01_simple_anchor_after_fs | 0.88-0.96 (within) |
| CuMiDa Gastric (GSE54129) (`cumida_gastric_gse54129`) | Easy | genomics | cDNA | 132 / 54,675 | A01_simple_anchor_after_fs | 0.90-0.98 (above) |
| CuMiDa Head/Neck (GSE12452) (`cumida_headneck_gse12452`) | Medium | genomics | cDNA | 41 / 54,675 | A02_default_anchor_after_fs | 0.78-0.90 (above) |
| CuMiDa Leukemia Subtypes (`cumida_leukemia_subtypes`) | Very Hard | genomics | cDNA | 281 / 22,283 | A02_default_anchor_after_fs | 0.85-0.97 (within) |
| CuMiDa Lung (GSE19804) (`cumida_lung_gse19804`) | Medium | genomics | cDNA | 120 / 54,675 | A07_ref_before_fs | 0.85-0.95 (above) |
| CuMiDa Ovarian (GSE26712) (`cumida_ovarian_gse26712`) | Medium | genomics | cDNA | 195 / 22,283 | A01_simple_anchor_after_fs | 0.90-0.98 (above) |
| CuMiDa Pancreatic (GSE16515) (`cumida_pancreatic_gse16515`) | Medium | genomics | cDNA | 52 / 54,613 | A05_skip_df_ref | 0.80-0.92 (above) |
| CuMiDa Prostate (GSE6919) (`cumida_prostate_gse6919`) | Medium | genomics | cDNA | 171 / 12,625 | V20_L02_diversity_top3_full64 | 0.62-0.78 (above) |
| CuMiDa Renal (GSE53757) (`cumida_renal_gse53757`) | Medium | genomics | cDNA | 144 / 54,675 | A03_ref_anchor_after_fs | 0.80-0.92 (above) |
| DEXTER (NIPS 2003 FS Challenge) (`dexter_nips03`) | Medium | non_genomic | cDNA | 600 / 20,000 | N30_core_banzhaf_perf_only | 0.75-0.89 (above) |
| DLBCL (Shipp) (`dlbcl_shipp`) | Easy | genomics | Affy HG-U95 | 77 / 5,469 | A02_default_anchor_after_fs | 0.90-0.98 (above) |
| DOROTHEA (NIPS 2003 FS Challenge) (`dorothea_nips03`) | Hard | non_genomic | cDNA | 1,150 / 10,000 | D07_df_cv_after | 0.50-0.74 (above) |
| GCM (Ramaswamy) (`gcm_ramaswamy`) | Hard | genomics | Affy HG-U133A | 190 / 16,063 | D18_multimodal_rank | 0.50-0.65 (above) |
| GISETTE (NIPS 2003 FS Challenge) (`gisette_nips03`) | Medium | non_genomic | cDNA | 7,000 / 5,000 | C04_pool_mnpo_plus_tabpfn | 0.75-0.89 (above) |
| GLA-BRA-180 (Scikit-feature) (`gla_bra_180`) | Hard | genomics | cDNA | 180 / 49,151 | V19_C03_old_regime_mnpo_full64 | 0.58-0.72 (above) |
| Glioma (50 x 4434, 4-class) (`glioma_50_4class`) | Medium | genomics | cDNA | 50 / 4,434 | N11_core_banzhaf_cvar | 0.88-0.97 (within) |
| Glioma (GLI_85) (`gli_85`) | Medium | genomics | Affy HG-U133A | 85 / 22,283 | A05_skip_df_ref | 0.75-0.87 (above) |
| Leukemia (Golub) (`leukemia_golub`) | Easy | genomics | Affy HG-U95 | 72 / 7,129 | A01_simple_anchor_after_fs | 0.93-1.00 (within) |
| Lung Cancer (OpenML Lung, 5-class) (`lung_gordon`) | Hard | genomics | Affy HG-U133A | 203 / 12,600 | A08_ref_no_regime_gating | 0.75-0.88 (above) |
| Lymphoma-11 (OpenML) (`lymphoma_11`) | Hard | genomics | Affy HG-U133A | 96 / 4,026 | N11_core_banzhaf_cvar | 0.45-0.65 (above) |
| Lymphoma-3 (OpenML) (`lymphoma_3`) | Medium | genomics | Affy HG-U133A | 66 / 4,026 | A01_simple_anchor_after_fs | 0.80-0.90 (above) |
| Lymphoma-9 (OpenML) (`lymphoma_9`) | Hard | genomics | Affy HG-U133A | 96 / 4,026 | M_RAW_iterative_redundancy_pruning_bounded | 0.50-0.70 (above) |
| MADELON (NIPS 2003 FS Challenge) (`madelon_nips03`) | Medium | non_genomic | cDNA | 2,600 / 500 | C10_only_tabpfn_full64 | 0.75-0.89 (within) |
| MLL (OpenML) (`mll_microarray`) | Easy | genomics | Affy HG-U95 | 72 / 12,582 | A02_default_anchor_after_fs | 0.88-0.96 (above) |
| MLL Leukemia — Armstrong 2002 (72 x 12533, 3-class) (`leukemia_1_72_3class`) | Medium | genomics | cDNA | 72 / 12,533 | A01_simple_anchor_after_fs | 0.85-0.96 (above) |
| NCI (61 x 5244, 8-class proxy) (`nci_61_8class`) | Hard | genomics | cDNA | 61 / 5,244 | A07_ref_before_fs | 0.50-0.68 (above) |
| NCI60 (Ross) (`nci60_ross`) | Hard | genomics | Affy HG-U95 | 64 / 6,830 | M_RAW_chi_square | 0.40-0.55 (within) |
| NCI60 Strict Holdout (`nci60_strict_holdout`) | Very Hard | genomics | Affy HG-U95 | 64 / 6,830 | P15_lowpn_all_features | 0.30-0.50 (above) |
| NCI9 (60 x 9712, 9-class) (`nci9_60_9class`) | Hard | genomics | cDNA | 60 / 5,726 | V20_F01_flaml_30s_mnpo_diag24 | 0.50-0.70 (within) |
| ORLraws10P (Face) (`orlraws10p`) | Easy | non_genomic | cDNA | 100 / 10,304 | A01_simple_anchor_after_fs | 0.95-1.00 (within) |
| Ovarian Cancer (Petricoin) (`ovarian_petricoin`) | Easy | genomics | mass-spec | 253 / 15,154 | A01_simple_anchor_after_fs | 0.95-1.00 (within) |
| pixraw10P (Face) (`pixraw10p`) | Easy | non_genomic | cDNA | 100 / 10,000 | A03_ref_anchor_after_fs | 0.96-1.00 (within) |
| Prostate Cancer (Singh) (`prostate_singh`) | Easy | genomics | Affy HG-U95 | 102 / 12,600 | A07_ref_before_fs | 0.88-0.95 (above) |
| SMK_CAN_187 (Smoking) (`smk_can_187`) | Medium | genomics | Affy HG-U133A | 187 / 19,993 | M_SCAFFOLD_subspace_stability | 0.65-0.80 (within) |
| SRBCT (Khan) (`srbct_khan`) | Easy | genomics | cDNA | 83 / 2,308 | A01_simple_anchor_after_fs | 0.92-1.00 (within) |
| TCGA-BRCA Breast Cancer (Xena, PAM50 5-subtype) (`xena_tcga_brca`) | Hard | genomics | cDNA | 956 / 20,530 | A07_ref_before_fs | 0.55-0.72 (above) |
| TCGA-COAD Colorectal (Xena, histological 2-class) (`xena_tcga_coad_cms`) | Hard | genomics | cDNA | 323 / 20,530 | V19_C01_old_regime_legacy_full64 | 0.58-0.75 (above) |
| TCGA-GBM Glioblastoma (Xena, 4-subtype) (`xena_tcga_gbm`) | Hard | genomics | cDNA | 164 / 20,530 | V19_C01_old_regime_legacy_full64 | 0.55-0.72 (above) |
| TCGA-HNSC Head & Neck (Xena, HPV binary) (`xena_tcga_hnsc_hpv`) | Medium | genomics | cDNA | 114 / 20,530 | A05_skip_df_ref | 0.82-0.93 (above) |
| TCGA-KIRC Kidney Clear Cell (Xena, stage-based) (`xena_tcga_kirc`) | Hard | genomics | cDNA | 606 / 20,530 | V20_B03_mnpo_ref_anchor | 0.50-0.68 (below) |
| TCGA-LGG Lower Grade Glioma (Xena, 3-subtype) (`xena_tcga_lgg`) | Hard | genomics | cDNA | 529 / 20,530 | V19_C04_new_regime_mnpo_full64 | 0.55-0.72 (within) |
| TCGA-LIHC Liver (Xena, histologic grade 3-class) (`xena_tcga_lihc`) | Hard | genomics | cDNA | 415 / 20,530 | V19_C03_old_regime_mnpo_full64 | 0.52-0.68 (within) |
| TCGA-LUAD Lung Adenocarcinoma (Xena, 3-subtype) (`xena_tcga_luad`) | Hard | genomics | cDNA | 275 / 20,530 | A07_ref_before_fs | 0.55-0.70 (above) |
| TCGA-OV Ovarian (Xena, histologic grade G2 vs G3) (`xena_tcga_ov`) | Hard | genomics | cDNA | 299 / 20,530 | A01_simple_anchor_after_fs | 0.48-0.65 (above) |
| TCGA-PRAD Prostate (Xena, Gleason score) (`xena_tcga_prad`) | Hard | genomics | cDNA | 550 / 20,530 | V19_C04_new_regime_mnpo_full64 | 0.45-0.65 (within) |
| TCGA-SKCM Melanoma (Xena, primary vs metastatic) (`xena_tcga_skcm`) | Hard | genomics | cDNA | 472 / 20,530 | N41_core_banzhaf_perf_only_5x5 | 0.78-0.90 (within) |
| TCGA-STAD Stomach (Xena, histological 3-group) (`xena_tcga_stad`) | Hard | genomics | cDNA | 448 / 20,530 | V20_L01_lr_prior_reduced_full64 | 0.58-0.75 (within) |
| TCGA-UCEC Uterine (Xena, histological 3-type) (`xena_tcga_ucec`) | Hard | genomics | cDNA | 190 / 20,530 | V19_C01_old_regime_legacy_full64 | 0.50-0.68 (above) |
| TOX_171 (Scikit-feature) (`tox_171`) | Medium | genomics | cDNA | 171 / 5,748 | A01_simple_anchor_after_fs | 0.84-0.93 (above) |
| warpPIE10P (Face) (`warp_pie10p`) | Easy | non_genomic | cDNA | 210 / 2,420 | A07_ref_before_fs | 0.95-1.00 (within) |

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

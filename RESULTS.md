# Benchmark Results

> **Status:** These results are from validation campaigns Val-18 and Val-19, run during active development. A peer-reviewed article with full methodology, ablation studies, and statistical analysis is in preparation.

---

## Overview

Tabnetics is evaluated on a catalog of **63 benchmark datasets** spanning binary and multiclass classification tasks in the HDLSS regime (high-dimensional, low sample size). Datasets range from 41 to 7,000 samples, 500 to 100,001 features, and 2 to 14 classes. The evaluation protocol uses multiple random seeds per dataset, stratified train/test splits (80/20), and reports **balanced accuracy** (macro-averaged recall) as the primary metric.

Results below are from **Val-18** (48,906 rows) and **Val-19** (790 rows) — combined validation campaigns with 49,696 successful runs across 199 pipeline profiles.

---

## Aggregate results

| Metric | Value |
|---|---|
| Benchmark datasets | 63 |
| Pipeline profiles evaluated | 199 |
| Total runs (dataset × seed × profile) | 49,696 |
| Datasets with BA ≥ 0.90 | 27 / 63 |
| Datasets with BA ≥ 0.80 | 39 / 63 |
| Perfect classification (BA = 1.0) | 7 datasets |
| SOTA comparison: above / within / below | 18 / 25 / 20 |

---

## Dataset difficulty spectrum

The 63 benchmark datasets span a wide difficulty range. Each bar shows the best balanced accuracy achieved across all 199 profiles.

![Dataset Difficulty Spectrum](assets/images/dataset_difficulty_spectrum.png)

Tier assignments: 11 easy (BA ≥ 0.85), 23 medium (0.70–0.85), 27 hard (BA < 0.70), and 2 very hard.

---

## Experiment families

Tabnetics validation is organized into experiment families, each isolating a different pipeline component. The box plots below show the BA distribution across profiles within each family:

![Profile BA Distribution by Family](assets/images/family_overview.png)

| Family | Profiles | Description |
|---|---|---|
| Anchor (A) | 8 | Baseline and bypass controls |
| FS RAW / SCAFFOLD (M) | 40 + 40 | Individual feature selection methods (raw vs scaffold pipeline) |
| Oracle (N) | 23 | MNPO oracle weighting and component ablation |
| Distribution Fitting (D) | 19 | CDF-based distribution pre-processing variants |
| Classifier (C, C_ONLY) | 12 + 24 | Classifier pool and individual classifier experiments |
| Prefilter / Folding (P) | 16 | Variance gating, dimension folding experiments |
| Clf. Oracle Weighting (W) | 9 | Cross-stage classifier-oracle reweighting |
| Stage (S) | 7 | Pipeline stage ordering experiments |

---

## Per-dataset balanced accuracy

Best balanced accuracy across all profiles on each dataset. Datasets are grouped by difficulty tier.

### Easy tier (11 datasets)

| Dataset | Samples | Features | Classes | Best BA | Source |
|---|---|---|---|---|---|
| SRBCT (Khan) | 83 | 2,308 | 4 | **1.000** | [OpenML](https://www.openml.org/d/1106) |
| Ovarian Cancer (Petricoin) | 253 | 15,154 | 2 | **1.000** | [OpenML](https://www.openml.org/d/1166) |
| CuMiDa Gastric (GSE54129) | 132 | 54,675 | 2 | **1.000** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54129) |
| Leukemia (Golub) | 72 | 7,129 | 2 | **0.990** | [OpenML](https://www.openml.org/d/1104) |
| MLL Leukemia (Armstrong) | 72 | 12,582 | 3 | **0.989** | [OpenML](https://www.openml.org/d/1108) |
| Prostate Cancer (Singh) | 102 | 12,600 | 2 | **0.971** | [OpenML](https://www.openml.org/d/1107) |
| BASEHOCK Text | 1,993 | 4,862 | 2 | **0.969** | [Scikit-feature](https://jundongl.github.io/scikit-feature/) |
| ORLraws10P (Face) | 100 | 10,304 | 10 | **0.961** | Scikit-feature |
| DLBCL (Shipp) | 77 | 5,469 | 2 | **0.950** | [OpenML](https://www.openml.org/d/1102) |
| warpPIE10P (Face) | 210 | 2,420 | 10 | **0.947** | Scikit-feature |
| pixraw10P (Face) | 100 | 10,000 | 10 | **0.950** | Scikit-feature |

### Medium tier (23 datasets)

| Dataset | Samples | Features | Classes | Best BA | Source |
|---|---|---|---|---|---|
| Lymphoma-3 | 66 | 4,026 | 3 | **1.000** | [OpenML](https://www.openml.org/d/377) |
| CuMiDa Ovarian (GSE26712) | 195 | 22,283 | 2 | **1.000** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE26712) |
| CuMiDa Head/Neck (GSE12452) | 41 | 54,675 | 2 | **1.000** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12452) |
| CuMiDa Renal (GSE53757) | 144 | 54,675 | 2 | **0.986** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53757) |
| GISETTE (NIPS 2003) | 7,000 | 5,001 | 2 | **0.979** | [OpenML](https://www.openml.org/d/41027) |
| MLL Leukemia 3-class | 72 | 12,533 | 3 | **0.972** | [Armstrong et al. 2002](https://doi.org/10.1038/ng765) |
| Colon Cancer (Alon) | 62 | 2,000 | 2 | **0.968** | [OpenML](https://www.openml.org/d/1105) |
| CuMiDa Lung (GSE19804) | 120 | 54,675 | 2 | **0.950** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19804) |
| TCGA-HNSC HPV | 114 | 20,530 | 2 | **0.938** | [UCSC Xena](https://xenabrowser.net/) |
| CuMiDa Colorectal (GSE44861) | 111 | 22,277 | 2 | **0.932** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44861) |
| DEXTER (NIPS 2003) | 600 | 20,001 | 2 | **0.922** | [OpenML](https://www.openml.org/d/4136) |
| CuMiDa Pancreatic (GSE16515) | 52 | 54,613 | 2 | **0.917** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE16515) |
| TOX_171 | 171 | 5,748 | 4 | **0.905** | [Scikit-feature](https://jundongl.github.io/scikit-feature/) |
| GLI_85 Glioma | 85 | 22,283 | 2 | **0.878** | [OpenML](https://www.openml.org/d/1111) |
| CLL_SUB_111 | 111 | 11,340 | 3 | **0.859** | [Scikit-feature](https://jundongl.github.io/scikit-feature/) |
| MADELON (NIPS 2003) | 2,600 | 500 | 2 | **0.856** | [OpenML](https://www.openml.org/d/1485) |
| ARCENE (NIPS 2003) | 200 | 10,000 | 2 | **0.834** | [OpenML](https://www.openml.org/d/1458) |
| Brain Tumor 2 (Nutt) | 50 | 12,625 | 4 | **0.817** | [BioLab](https://doi.org/10.1158/0008-5472.CAN-02-4243) |
| CuMiDa Prostate (GSE6919) | 171 | 12,625 | 2 | **0.813** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE6919) |
| Glioma 4-class | 50 | 4,434 | 4 | **0.800** | [Scikit-feature](https://jundongl.github.io/scikit-feature/) |
| SMK_CAN_187 | 187 | 19,993 | 2 | 0.753 | [OpenML](https://www.openml.org/d/1095) |
| Breast Cancer (van 't Veer) | 97 | 24,481 | 2 | 0.711 | [OpenML](https://www.openml.org/d/1168) |
| CNS / Brain (Pomeroy) | 60 | 7,129 | 2 | 0.663 | [OpenML](https://www.openml.org/d/1100) |

### Hard tier (27 datasets)

| Dataset | Samples | Features | Classes | Best BA | Source |
|---|---|---|---|---|---|
| CuMiDa Breast (GSE45827) | 151 | 54,675 | 6 | **1.000** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827) |
| Lung Cancer (Gordon) | 203 | 12,600 | 5 | **0.984** | [OpenML](https://www.openml.org/d/1109) |
| CuMiDa Brain (GSE50161) | 108 | 54,675 | 4 | **0.974** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50161) |
| Carcinom 11-class | 174 | 9,182 | 11 | **0.967** | [Scikit-feature](https://jundongl.github.io/scikit-feature/) |
| 11-Tumor (Su) | 174 | 12,533 | 11 | **0.964** | [OpenML](https://www.openml.org/d/1113) |
| Lymphoma-9 | 96 | 4,026 | 9 | **0.958** | [OpenML](https://www.openml.org/d/378) |
| TCGA-BRCA Breast | 956 | 20,530 | 5 | **0.910** | [UCSC Xena](https://xenabrowser.net/) |
| TCGA-SKCM Melanoma | 472 | 20,530 | 2 | **0.880** | [UCSC Xena](https://xenabrowser.net/) |
| Lymphoma-11 | 96 | 4,026 | 11 | **0.828** | [OpenML](https://www.openml.org/d/378) |
| DOROTHEA (NIPS 2003) | 1,150 | 100,001 | 2 | 0.793 | [OpenML](https://www.openml.org/d/4137) |
| NCI 8-class | 61 | 5,244 | 8 | 0.792 | NCI60 |
| TCGA-LUAD Lung | 275 | 20,530 | 3 | 0.774 | [UCSC Xena](https://xenabrowser.net/) |
| TCGA-GBM Glioblastoma | 164 | 20,530 | 4 | 0.767 | [UCSC Xena](https://xenabrowser.net/) |
| Breast Gene Expression (HF) | 51 | 28,278 | 2 | 0.757 | [HuggingFace](https://huggingface.co/datasets/mubashir1837/Breast_cancer_gene_expression) |
| TCGA-COAD Colorectal | 323 | 20,530 | 2 | 0.756 | [UCSC Xena](https://xenabrowser.net/) |
| GCM (Ramaswamy) | 198 | 16,063 | 14 | 0.704 | [OpenML](https://www.openml.org/d/1112) |
| GLA-BRA-180 | 180 | 49,151 | 4 | 0.678 | [Scikit-feature](https://jundongl.github.io/scikit-feature/) |
| TCGA-UCEC Uterine | 190 | 20,530 | 3 | 0.673 | [UCSC Xena](https://xenabrowser.net/) |
| TCGA-OV Ovarian | 299 | 20,530 | 2 | 0.625 | [UCSC Xena](https://xenabrowser.net/) |
| NCI9 (9-class) | 60 | 9,712 | 9 | 0.613 | [OpenML](https://www.openml.org/d/1115) |
| TCGA-LGG Glioma | 529 | 20,530 | 3 | 0.558 | [UCSC Xena](https://xenabrowser.net/) |
| TCGA-STAD Stomach | 448 | 20,530 | 3 | 0.549 | [UCSC Xena](https://xenabrowser.net/) |
| 9-Tumors | 60 | 5,726 | 9 | 0.538 | [OpenML](https://www.openml.org/d/1114) |
| NCI60 (Ross) | 60 | 6,830 | 9 | 0.518 | [OpenML](https://www.openml.org/d/1116) |
| TCGA-LIHC Liver | 415 | 20,530 | 3 | 0.492 | [UCSC Xena](https://xenabrowser.net/) |
| TCGA-KIRC Kidney | 606 | 20,530 | 4 | 0.476 | [UCSC Xena](https://xenabrowser.net/) |
| TCGA-PRAD Prostate | 550 | 20,530 | 5 | 0.374 | [UCSC Xena](https://xenabrowser.net/) |

### Very hard tier (2 datasets)

| Dataset | Samples | Features | Classes | Best BA | Source |
|---|---|---|---|---|---|
| CuMiDa Leukemia Subtypes | 281 | 22,283 | 7 | **0.865** | [CuMiDa](https://sbcb.inf.ufrgs.br/cumida) |
| NCI60 Strict Holdout | 60 | 6,830 | 9 | 0.546 | NCI60 |

---

## MNPO oracle weighting

The MNPO aggregation framework combines multiple feature selection methods using game-theoretic weighting. The oracle weighting hierarchy is a key architectural finding:

![Oracle Weighting Hierarchy](assets/images/oracle_weighting_hierarchy.png)

**Banzhaf > TriTrust > Shapley > Uniform** — Banzhaf power-index weighting consistently outperforms all alternatives, including Shapley values and simple uniform averaging. The hierarchy is stable across all difficulty tiers.

JS divergence is the most impactful oracle component (removing it costs −0.018 BA). A 2-oracle configuration (performance + JS divergence) could reduce computational cost by approximately 50% with minimal accuracy loss.

---

## Feature selection: engineered vs random baseline

A key finding across 40 feature selection methods tested in both RAW and SCAFFOLD pipelines:

![Engineered FS vs Random Baseline](assets/images/fs_vs_random_volcano.png)

Only **15 of 39** engineered FS methods (38%) significantly beat the random baseline (Wilcoxon, p < 0.1), with a mean advantage of just **+0.0065 BA**. This "FS paradox" highlights that the MNPO architecture's value lies in its **ensemble averaging** across methods rather than finding the single best feature selector.

---

## Classifier pool

Individual classifiers evaluated in isolation across the benchmark catalog:

![Classifier Rankings](assets/images/classifier_rankings.png)

Top classifiers: TabPFN, DLDA/Shrinkage LDA, Elastic Net LR, and RP Ensemble consistently rank highest. SGLNN and PLS-DA are identified as harmful (below the pool average) and candidates for removal.

---

## SOTA comparison

Val-18 best profiles are compared to published results for each dataset. Comparison confidence is categorized by protocol match quality:

| Confidence Level | Datasets | Above SOTA | Within Range | Below SOTA |
|---|---|---|---|---|
| Direct (protocol-matched) | 10 | 3 | 5 | 2 |
| Discounted (close protocol) | 35 | 10 | 13 | 12 |
| Proxy (positioning only) | 18 | 5 | 7 | 6 |
| **Total** | **63** | **18** | **25** | **20** |

Val-18 profiles achieve BA approximately 0.80 on HDLSS microarray benchmarks, which is competitive with published results for this domain (0.78–0.85 typical range).

---

## TabArena comparison

To contextualize performance on general tabular data, tabnetics was evaluated on [TabArena](https://huggingface.co/spaces/TabArena/TabArena-Leaderboard) — a broad benchmark suite spanning general tabular classification (not HDLSS-specific):

| Metric | Value |
|---|---|
| Elo rating | 1002.4 |
| Rank | 37 / 45 |
| Win rate | 0.266 |
| Mean BA | 0.744 |

The approximately 200-point Elo gap to competitive defaults (XGBoost 1205, EBM 1229) and approximately 400-point gap to tuned ensembles (RealMLP 1449, TabM 1414) reflects that tabnetics is designed for **HDLSS bioinformatics**, not general tabular data. On TabArena's general benchmarks (n >> p), gradient-boosted trees with hyperparameter tuning dominate — an expected and well-documented result.

---

## Validation protocol

- **Split:** Stratified train/test split (80/20) with multiple random seeds per dataset (median 5 seeds).
- **Metric:** Balanced accuracy (macro-averaged recall), which accounts for class imbalance.
- **Leakage prevention:** All distribution fitting, feature selection, and model selection are performed on training data only. Test data is never seen during preprocessing.
- **Statistical testing:** Pairwise profile comparisons use Wilcoxon signed-rank tests on per-dataset balanced accuracy, with Benjamini–Hochberg FDR correction. Effect sizes reported as Hodges–Lehmann estimators.
- **Reproducibility:** All datasets are available through OpenML, GEO, Scikit-feature, UCSC Xena, or HuggingFace.

---

## Dataset sources

The 63 benchmark datasets come from established sources in the HDLSS classification literature:

| Source | Count | Description |
|---|---|---|
| [OpenML](https://www.openml.org/) | 20 | Standardized ML benchmark repository |
| [UCSC Xena](https://xenabrowser.net/) (TCGA) | 13 | TCGA RNA-seq gene expression (20,530 genes) |
| [GEO](https://www.ncbi.nlm.nih.gov/geo/) / [CuMiDa](https://sbcb.inf.ufrgs.br/cumida) | 12 | NCBI Gene Expression Omnibus (curated microarray) |
| [Scikit-feature](https://jundongl.github.io/scikit-feature/) | 8 | Feature selection benchmark datasets |
| Face recognition / text | 4 | ORLraws10P, warpPIE10P, pixraw10P, BASEHOCK |
| Other | 6 | NCI60, HuggingFace, BioLab, NIPS 2003 challenge |

For reproducible validation runs, Tabnetics packages many of these public datasets into a HuggingFace bundle. The bundle is an operational mirror of the public upstream sources above, not a separate private dataset collection.

### Key references

- **Golub et al.** ["Molecular classification of cancer: class discovery and class prediction by gene expression monitoring."](https://doi.org/10.1126/science.286.5439.531) *Science* 286(5439):531–537, 1999. — Leukemia dataset.
- **Armstrong et al.** ["MLL translocations specify a distinct gene expression profile that distinguishes a unique leukemia."](https://doi.org/10.1038/ng765) *Nature Genetics* 30:41–47, 2002. — MLL leukemia dataset.
- **Khan et al.** ["Classification and diagnostic prediction of cancers using gene expression profiling and artificial neural networks."](https://doi.org/10.1038/89044) *Nature Medicine* 7:673–679, 2001. — SRBCT dataset.
- **Shipp et al.** ["Diffuse large B-cell lymphoma outcome prediction by gene-expression profiling and supervised machine learning."](https://www.nature.com/articles/nm0102-68) *Nature Medicine* 8:68–74, 2002. — DLBCL dataset.
- **Feltes et al.** ["CuMiDa: An extensively curated microarray database for benchmarking and testing of machine learning approaches."](https://doi.org/10.1089/cmb.2018.0238) *J. Computational Biology* 26(4):376–386, 2019. — CuMiDa datasets.
- **de Souto et al.** ["Clustering cancer gene expression data: a comparative study."](https://doi.org/10.1186/1471-2105-9-497) *BMC Bioinformatics* 9:497, 2008. — Multi-dataset benchmark design.
- **Guyon et al.** [*Design of experiments for the NIPS 2003 variable selection benchmark*](https://www.sambuz.com/doc/design-of-experiments-for-the-nips-2003-variable-pdf-document-667666). — ARCENE, MADELON, DEXTER, DOROTHEA, GISETTE.
- **TCGA Research Network.** ["Comprehensive genomic characterization defines human glioblastoma genes and core pathways."](https://pubmed.ncbi.nlm.nih.gov/18772890/) *Nature* 455:1061–1068, 2008. — TCGA datasets.
- **Goldman et al.** ["Visualizing and interpreting cancer genomics data via the Xena platform."](https://doi.org/10.1038/s41587-020-0546-8) *Nature Biotechnology* 38:675–678, 2020. — UCSC Xena browser.
- **Hollmann et al.** ["TabPFN: A transformer that solves small tabular classification problems in a second."](https://arxiv.org/abs/2207.01848) *ICLR* 2023. — TabPFN classifier.
- **Banzhaf, J. F.** ["Weighted voting doesn't work: a mathematical analysis."](https://doi.org/10.2307/1933169) *Rutgers Law Review* 19:317–343, 1965. — Banzhaf power index used in MNPO weighting.

---

## Ongoing work

A peer-reviewed article presenting the full methodology, ablation studies, and extended results is in preparation. The article will cover:

- Formal description of the MNPO aggregation framework
- Ablation of each pipeline stage (prefilter, distribution fitting, feature selection, classification)
- Comparison with SOTA AutoML methods (FLAML, AutoGluon, TabPFN) on the full benchmark catalog
- Analysis of failure modes on very-hard multiclass datasets (9–14 classes, $n < 100$)
- The feature selection paradox: why ensemble averaging outperforms individual method selection
- Extended validation on held-out datasets not used during development

Results in this document will be updated as validation campaigns continue.

---

*This documentation is auto-generated from internal notes and sources with the support of rule-based transformations and generative AI. Errors are possible — please report any issues via [Discussions](https://github.com/klokedm/tabnetics-public/discussions).*

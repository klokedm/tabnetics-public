# Benchmark Results

> **Status:** These results are from validation campaigns run during active development. A peer-reviewed article with full methodology, ablation studies, and statistical analysis is in preparation.

---

## Overview

Tabnetics is evaluated on a catalog of **35 benchmark datasets** spanning binary and multiclass classification tasks in the HDLSS regime. Datasets range from 50 to 2,600 samples, 500 to 100,000 features, and 2 to 14 classes. The evaluation protocol uses multiple random seeds per dataset (9 seeds), stratified train/test splits (80/20), and reports **balanced accuracy** (macro-averaged recall) as the primary metric.

Results below are from **Val-15** — a complete validation campaign with 2,832 successful runs across 9 pipeline profiles.

---

## Aggregate results

| Metric | Value |
|---|---|
| Benchmark datasets | 35 |
| Pipeline profiles evaluated | 9 |
| Total runs (dataset × seed × profile) | 2,832 |
| Overall mean balanced accuracy (best profile) | **0.803** |
| Datasets with BA ≥ 0.90 | 12 / 35 |
| Datasets with BA ≥ 0.80 | 20 / 35 |
| Perfect classification (BA = 1.0) | 3 datasets |

---

## Per-dataset balanced accuracy

Mean balanced accuracy across 9 random seeds for the best-performing profile on each dataset. Datasets are grouped by difficulty tier.

### Easy tier

| Dataset | Samples | Features | Classes | Mean BA | Source |
|---|---|---|---|---|---|
| Leukemia (Golub) | 72 | 7,129 | 2 | **0.961** | [OpenML](https://www.openml.org/d/1104) |
| DLBCL (Shipp) | 77 | 5,469 | 2 | **0.884** | [OpenML](https://www.openml.org/d/1102) |
| Ovarian Cancer (Petricoin) | 253 | 15,154 | 2 | **0.995** | [OpenML](https://www.openml.org/d/1166) |
| SRBCT (Khan) | 83 | 2,308 | 4 | **1.000** | [OpenML](https://www.openml.org/d/1106) |
| Prostate Cancer (Singh) | 102 | 12,600 | 2 | **0.942** | [OpenML](https://www.openml.org/d/1107) |
| CuMiDa Gastric (GSE54129) | 132 | 54,675 | 2 | **1.000** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54129) |

### Medium tier

| Dataset | Samples | Features | Classes | Mean BA | Source |
|---|---|---|---|---|---|
| MLL Leukemia (Armstrong 2002) | 72 | 12,533 | 3 | **0.936** | [Armstrong et al. 2002](https://doi.org/10.1038/ng765) |
| GLI_85 Glioma | 85 | 22,283 | 2 | **0.845** | [OpenML](https://www.openml.org/d/1111) |
| CNS / Brain Tumors (Pomeroy) | 60 | 7,129 | 2 | **0.806** | [OpenML](https://www.openml.org/d/1100) |
| Colon Cancer (Alon) | 62 | 2,000 | 2 | **0.865** | [OpenML](https://www.openml.org/d/1105) |
| CLL_SUB_111 | 111 | 11,340 | 3 | **0.836** | [Scikit-feature](https://jundongl.github.io/scikit-feature/) |
| Breast Cancer (van 't Veer) | 97 | 24,481 | 2 | 0.648 | [OpenML](https://www.openml.org/d/1168) |
| Glioma 4-class | 50 | 4,434 | 4 | 0.704 | [Scikit-feature](https://jundongl.github.io/scikit-feature/) |
| CuMiDa Colorectal (GSE44861) | 111 | 22,277 | 2 | 0.796 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44861) |
| ARCENE (NIPS 2003) | 200 | 10,000 | 2 | **0.865** | [OpenML](https://www.openml.org/d/1458) |
| MADELON (NIPS 2003) | 2,600 | 500 | 2 | 0.694 | [OpenML](https://www.openml.org/d/1485) |
| TOX_171 | 171 | 5,748 | 4 | **0.863** | [Scikit-feature](https://jundongl.github.io/scikit-feature/) |

### Hard tier

| Dataset | Samples | Features | Classes | Mean BA | Source |
|---|---|---|---|---|---|
| Lymphoma-3 | 66 | 4,026 | 3 | **0.989** | [OpenML](https://www.openml.org/d/377) |
| Lymphoma-9 | 96 | 4,026 | 9 | **0.907** | [OpenML](https://www.openml.org/d/378) |
| Carcinom 11-class | 174 | 9,182 | 11 | **0.976** | [Scikit-feature](https://jundongl.github.io/scikit-feature/) |
| 11-Tumor (Su) | 174 | 12,533 | 11 | **0.939** | [OpenML](https://www.openml.org/d/1113) |
| CuMiDa Brain (GSE50161) | 108 | 54,675 | 4 | **0.895** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50161) |
| CuMiDa Renal (GSE53757) | 144 | 54,675 | 2 | **0.958** | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53757) |
| CuMiDa Leukemia Subtypes | 281 | 22,283 | 7 | **0.820** | [CuMiDa](https://sbcb.inf.ufrgs.br/cumida) |
| GCM (Ramaswamy) | 198 | 16,063 | 14 | 0.564 | [OpenML](https://www.openml.org/d/1112) |
| GLA-BRA-180 | 180 | 49,151 | 4 | — | [Scikit-feature](https://jundongl.github.io/scikit-feature/) |
| DOROTHEA (NIPS 2003) | 1,150 | 100,001 | 2 | **1.000** | [OpenML](https://www.openml.org/d/4137) |
| NCI (8-class proxy) | 61 | 5,244 | 8 | **0.806** | NCI60 |
| NCI9 (9-class) | 60 | 9,712 | 9 | 0.493 | [OpenML](https://www.openml.org/d/1115) |
| Breast Gene Expression (HF) | 51 | 28,278 | 2 | 0.633 | [HuggingFace](https://huggingface.co/datasets/mubashir1837/Breast_cancer_gene_expression) |
| TCGA-SKCM Melanoma | 472 | 20,530 | 2 | **0.845** | [UCSC Xena](https://xenabrowser.net/) |
| TCGA-COAD Colorectal | 323 | 20,530 | 2 | 0.748 | [UCSC Xena](https://xenabrowser.net/) |
| TCGA-GBM Glioblastoma | 164 | 20,530 | 4 | 0.672 | [UCSC Xena](https://xenabrowser.net/) |
| TCGA-LGG Lower Grade Glioma | 529 | 20,530 | 3 | 0.555 | [UCSC Xena](https://xenabrowser.net/) |

### Very hard tier

| Dataset | Samples | Features | Classes | Mean BA | Source |
|---|---|---|---|---|---|
| NCI60 Strict Holdout | 60 | 6,830 | 9 | 0.445 | NCI60 |
| 9-Tumors | 60 | 5,726 | 9 | 0.472 | [OpenML](https://www.openml.org/d/1114) |

---

## Profile comparison

Nine pipeline profiles were compared using Wilcoxon signed-rank tests with Benjamini–Hochberg correction (35 paired datasets, 9 seeds each).

| Rank | Profile | Mean BA | Description |
|---|---|---|---|
| 1 | Best profile | **0.803** | Distribution fitting + MNPO portfolio FS + multi-omics adapter |
| 2 | Core pipeline | 0.780 | Distribution fitting + MNPO portfolio FS |
| 3 | Reference | 0.779 | MNPO portfolio FS with IPSS |
| 4–6 | Conformal variants | 0.779 | + MAPIE conformal prediction (zero BA effect by design) |
| 7 | Control | 0.777 | Simple baseline without distribution fitting |
| 8 | Default | 0.777 | Production default |
| 9 | No regime fallback | 0.773 | Regime gating disabled |

Key findings:
- The MNPO portfolio consistently outperforms single-method baselines.
- Distribution fitting as preprocessing contributes a small but consistent positive effect.
- Conformal prediction adds uncertainty quantification without affecting balanced accuracy (by design).
- Very-hard fallback routing (regime gating) is beneficial on extreme HDLSS datasets (+0.025 BA on 5 trigger datasets).

---

## Validation protocol

- **Split:** Stratified train/test split (80/20) with 9 random seeds per dataset.
- **Metric:** Balanced accuracy (macro-averaged recall), which accounts for class imbalance.
- **Leakage prevention:** All distribution fitting, feature selection, and model selection are performed on training data only. Test data is never seen during preprocessing.
- **Statistical testing:** Pairwise profile comparisons use Wilcoxon signed-rank tests on per-dataset balanced accuracy, with Benjamini–Hochberg FDR correction. Effect sizes reported as rank-biserial correlation ($r$) and Cohen's $d$.
- **Reproducibility:** All datasets are available through OpenML, GEO, Scikit-feature, or HuggingFace.

---

## Dataset sources

The 35 benchmark datasets come from established sources in the HDLSS classification literature:

| Source | Count | Description |
|---|---|---|
| [OpenML](https://www.openml.org/) | 19 | Standardized ML benchmark repository |
| [GEO](https://www.ncbi.nlm.nih.gov/geo/) | 5 | NCBI Gene Expression Omnibus (via [CuMiDa](https://sbcb.inf.ufrgs.br/cumida)) |
| [Scikit-feature](https://jundongl.github.io/scikit-feature/) | 5 | Feature selection benchmark datasets |
| [UCSC Xena](https://xenabrowser.net/) | 4 | TCGA RNA-seq gene expression (20,530 genes) |
| Other | 2 | NCI60 cell line panel, HuggingFace |

### Key references

- **Golub et al.** "Molecular classification of cancer: class discovery and class prediction by gene expression monitoring." *Science* 286(5439):531–537, 1999. — Leukemia dataset.
- **Armstrong et al.** "MLL translocations specify a distinct gene expression profile that distinguishes a unique leukemia." *Nature Genetics* 30:41–47, 2002. — MLL leukemia dataset.
- **Nutt et al.** "Gene expression-based classification of malignant gliomas." *Cancer Research* 63(7):1602–1607, 2003. — Brain Tumor 2 dataset.
- **Khan et al.** "Classification and diagnostic prediction of cancers using gene expression profiling and artificial neural networks." *Nature Medicine* 7:673–679, 2001. — SRBCT dataset.
- **Shipp et al.** "Diffuse large B-cell lymphoma outcome prediction by gene-expression profiling." *New England Journal of Medicine* 346(25):1937–1947, 2002. — DLBCL dataset.
- **Feltes et al.** "CuMiDa: An extensively curated microarray database for benchmarking and testing of machine learning approaches." *J. Computational Biology* 26(4):376–386, 2019. — CuMiDa datasets.
- **de Souto et al.** "Clustering cancer gene expression data: a comparative study." *BMC Bioinformatics* 9:497, 2008. — Multi-dataset benchmark design.
- **Guyon et al.** "Design of experiments for the NIPS 2003 variable selection benchmark." NIPS 2003 Feature Selection Challenge. — ARCENE, MADELON, DEXTER, DOROTHEA, GISETTE.
- **TCGA Research Network.** "Comprehensive genomic characterization defines human glioblastoma genes and core pathways." *Nature* 455:1061–1068, 2008. — TCGA datasets.
- **Goldman et al.** "Visualizing and interpreting cancer genomics data via the Xena platform." *Nature Biotechnology* 38:675–678, 2020. — UCSC Xena browser.

---

## Extended catalog

Beyond the 35 core benchmark datasets, tabnetics includes an extended catalog of **93 total datasets** covering:

- **21 additional genomics datasets** (CuMiDa tissue-specific panels, 11 TCGA cancer types via UCSC Xena)
- **12 results-validation holdout datasets** (independent evaluation, not used during development)
- **12 distribution-fitting benchmarks** (synthetic parametric, financial, actuarial, hydrology scenarios)
- **5 integrated pipeline benchmarks** (end-to-end CDF transform + FS evaluation)
- **3 synthetic HDLSS generators** (controlled difficulty for unit testing)

See [USING.md](USING.md) for dataset group names and CLI usage.

---

## Ongoing work

A peer-reviewed article presenting the full methodology, ablation studies, and extended results is in preparation. The article will cover:

- Formal description of the MNPO aggregation framework
- Ablation of each pipeline stage (prefilter, distribution fitting, feature selection, classification)
- Comparison with SOTA AutoML methods (FLAML, AutoGluon, TabPFN) on the full benchmark catalog
- Analysis of failure modes on very-hard multiclass datasets (9–14 classes, $n < 100$)
- Extended validation on held-out datasets not used during development

Results in this document will be updated as validation campaigns continue.

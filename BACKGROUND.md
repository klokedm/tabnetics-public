# Background — Methods and References

Tabnetics implements both **novel contributions** developed as part of this project and **established methods** from the feature selection, distribution fitting, and HDLSS classification literature. This document maps each component to its theoretical foundation.

---

## Novel contributions

These components were developed specifically for tabnetics and do not correspond to a single prior publication.

### MNPO — Nash Multi-Portfolio Optimization

The core aggregation engine. MNPO frames feature-selection method combination as a cooperative game: each candidate method is a player, and a set of oracles (performance, stability, complexity, robustness, diversity) evaluate portfolios. A Nash bargaining solution selects the Pareto-optimal portfolio. Key novel elements:

- **Multi-oracle pairwise preference framework** — oracles cast pairwise preferences between candidate method subsets; these are fused via weighted voting or Banzhaf indices.
- **Banzhaf / Shapley weighting for oracles** — oracle influence weights are computed from cooperative game theory rather than fixed by hand.
- **CVaR oracle** — a tail-risk oracle that optimizes conditional value-at-risk over fold-level balanced accuracy.
- **Complementarity oracle** — measures feature-set complementarity via partial information decomposition (PID) or mutual-information redundancy terms.
- **Oracle redundancy penalty** — detects and down-weights oracles whose recommendations are collinear, preventing double-counting.
- **Adaptive portfolio sizing** — the number of methods retained by MNPO scales with dataset difficulty and the distribution of oracle scores.

### Regime-gated pipeline routing

A lightweight regime detector classifies datasets into HDLSS tiers (extreme, moderate, mild) and routes each tier to a pre-configured pipeline profile. This avoids running expensive methods (e.g., copula knockoffs on $n < 40$ datasets) where they are statistically unreliable.

### Distribution fitting as a preprocessing stage

While individual distribution families are standard, using distribution fitting as a CDF-based preprocessing step inside a feature-selection pipeline — with bootstrap-calibrated goodness-of-fit tests, L-moment prescreening, and multimodal fallback — is a pipeline-level contribution.

### Tri-gate validation protocol

A three-level promotion framework (method-gate → portfolio-gate → campaign-gate) ensures that pipeline changes are validated at the portfolio level with paired statistical tests (Wilcoxon signed-rank) across the full benchmark catalog.

---

## Implemented methods

Each section lists the methods implemented in tabnetics and the papers they are based on.

### Feature selection — stability-based

| Method | Reference |
|---|---|
| Stability Selection (Lasso) | Meinshausen & Bühlmann. "Stability selection." *J. Royal Statistical Society B*, 72(4):417–473, 2010. |
| Complementary Subsampling | Shah & Samworth. "Variable selection with error control." *J. Royal Statistical Society B*, 75(1):55–80, 2013. |
| TIGRESS | Haury et al. "TIGRESS: Trustful Inference of Gene REgulation using Stability Selection." *BMC Systems Biology*, 6:145, 2012. |
| IPSS (Integrated Path Stability Selection) | Melikechi et al. "Integrated path stability selection." arXiv:2403.15877, 2024. |
| Cluster Stability Selection | Faletto & Bien. "Cluster stability selection." *Computational Statistics & Data Analysis*, 177:107NY, 2022. |

### Feature selection — knockoff-based

| Method | Reference |
|---|---|
| Copula Knockoffs (D-vine) | Román-Vásquez et al. "Vine copula knockoff filter for high-dimensional controlled variable selection." arXiv:2410.00650, 2024. |
| Knockoff Filter (general framework) | Candès et al. "Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection." *J. Royal Statistical Society B*, 80(3):551–577, 2018. |
| Derandomized Knockoffs | Ren & Candès. "Derandomizing knockoffs." arXiv:2205.00556, 2022. |

### Feature selection — filter and information-theoretic

| Method | Reference |
|---|---|
| mRMR (Minimum Redundancy Maximum Relevance) | Peng, Long & Ding. "Feature selection based on mutual information." *IEEE Trans. Pattern Analysis & Machine Intelligence*, 27(8):1226–1238, 2005. |
| JMI (Joint Mutual Information) | Yang & Moody. "Data visualization and feature selection: new algorithms for nongaussian data." *NIPS*, 1999. |
| CMIM (Conditional Mutual Information Maximisation) | Fleuret. "Fast binary feature selection with conditional mutual information." *JMLR*, 5:1531–1555, 2004. |
| FCBF (Fast Correlation-Based Filter) | Yu & Liu. "Efficient feature selection via analysis of relevance and redundancy." *JMLR*, 5:1205–1224, 2004. |
| HSIC Lasso | Climente-González et al. "Block HSIC Lasso: model-free biomarker detection for ultra-high dimensional data." *Bioinformatics*, 35(14):i427–i435, 2019. |

### Feature selection — tree and wrapper

| Method | Reference |
|---|---|
| Boruta | Kursa & Rudnicki. "Feature selection with the Boruta package." *J. Statistical Software*, 36(11):1–13, 2010. |
| RFECV (Recursive Feature Elimination) | Guyon et al. "Gene selection for cancer classification using support vector machines." *Machine Learning*, 46:389–422, 2002. |
| TreeSHAP | Lundberg et al. "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence*, 2:56–67, 2020. |

### Feature selection — multiclass-specific

| Method | Reference |
|---|---|
| Nearest Shrunken Centroids | Tibshirani et al. "Diagnosis of multiple cancer types by shrunken centroids of gene expression." *PNAS*, 99(10):6567–6572, 2002. |
| OVA ensemble | Extension of standard one-vs-all decomposition for feature selection. |
| ECOC class-aware decomposition | Dietterich & Bakiri. "Solving multiclass learning problems via error-correcting output codes." *JAIR*, 2:263–286, 1995. |
| SIR / SAVE / PFC (sufficient dimension reduction) | Li. "Sufficient dimension reduction: methods and applications with R." *CRC Press*, 2018. |

### Feature selection — pairwise and AUC-based

| Method | Reference |
|---|---|
| WMW AUC filter | Mann & Whitney U-statistic applied as a univariate filter. |
| k-TSP (k Top Scoring Pairs) | Tan et al. "Simple decision rules for classifying human cancers from gene expression profiles." *Bioinformatics*, 21(20):3896–3904, 2005. |
| Joint AUC+L1 selector | AUC-weighted L1-penalized logistic regression for binary problems. |

### Feature selection — game-theoretic weights

| Concept | Reference |
|---|---|
| Banzhaf value (oracle weighting) | Wang & Jia. "Data Banzhaf: A Robust Data Valuation Framework for Machine Learning." *AISTATS*, 2023. |
| Kernel Banzhaf | Liu et al. "KernelSHAP-IQ: Weighted Least Square Optimization for Shapley Interactions." arXiv:2405.10852, 2024. |
| Shapley value | Shapley. "A value for n-person games." *Contributions to the Theory of Games*, 2:307–317, 1953. |
| QRE (Quantal Response Equilibrium) | McKelvey & Palfrey. "Quantal response equilibria for normal form games." *Games and Economic Behavior*, 10(1):6–38, 1995. |

### Distribution fitting

| Component | Reference |
|---|---|
| Parametric families (20+) | Standard implementations: normal, log-normal, gamma, Weibull, beta, GEV, GPD, Johnson $S_B$/$S_U$, skew-normal, folded-normal, inverse-Gaussian, Burr III/XII, Dagum, sinh-arcsinh, etc. via `scipy.stats`. |
| L-moment prescreening | Hosking. "L-moments: analysis and estimation of distributions using linear combinations of order statistics." *J. Royal Statistical Society B*, 52(1):105–124, 1990. |
| Bootstrap-calibrated GOF | Parametric bootstrap following Efron & Tibshirani (1994) to calibrate Kolmogorov–Smirnov and Cramér–von Mises p-values for small samples. |
| Maximum product spacing (MPS) | Ranneby. "The maximum spacing method. An estimation method related to the maximum likelihood method." *Scandinavian J. Statistics*, 11(2):93–112, 1984. |
| CRPS scoring | Gneiting & Raftery. "Strictly proper scoring rules, prediction, and estimation." *JASA*, 102(477):359–378, 2007. |

### Batch correction

| Method | Reference |
|---|---|
| ComBat | Johnson, Li & Rabinovic. "Adjusting batch effects in microarray expression data using empirical Bayes methods." *Biostatistics*, 8(1):118–127, 2007. |

### Classification

| Method | Reference |
|---|---|
| PLS-DA | Barker & Rayens. "Partial least squares for discrimination." *J. Chemometrics*, 17(3):166–173, 2003. |
| DLDA (Diagonal LDA) | Dudoit, Fridlyand & Speed. "Comparison of discrimination methods for the classification of tumors using gene expression data." *JASA*, 97(457):77–87, 2002. |
| TabPFN | Hollmann et al. "Accurate predictions on small data with a tabular foundation model." *Nature*, 637:319–326, 2025. |
| Conformal prediction (MAPIE) | Taquet et al. "MAPIE: an open-source library for distribution-free uncertainty quantification." arXiv:2207.12274, 2022. |
| UBayFS | Jenul et al. "UBayFS: An R package for user guided feature selection." *JOSS*, 7(79):4848, 2022. |

### Multi-omics

| Component | Reference |
|---|---|
| DIABLO-style multi-block PLS | Singh et al. "DIABLO: an integrative approach for identifying key molecular drivers from multi-omics assays." *Bioinformatics*, 35(17):3055–3062, 2019. |
| MINT batch correction | Rohart et al. "MINT: a multivariate integrative method to identify reproducible molecular signatures across independent experiments and platforms." *BMC Bioinformatics*, 18:128, 2017. |

---

## Benchmark datasets

Tabnetics includes a curated registry of HDLSS benchmark datasets. Key sources:

| Source | Reference |
|---|---|
| CuMiDa (curated microarrays) | Feltes et al. "CuMiDa: An extensively curated microarray database for benchmarking and testing of machine learning approaches." *J. Computational Biology*, 26(4):376–386, 2019. |
| de Souto benchmark | de Souto et al. "Clustering cancer gene expression data: a comparative study." *BMC Bioinformatics*, 9:497, 2008. |
| Statnikov multi-category | Statnikov et al. "A comprehensive comparison of random forests and support vector machines for microarray-based cancer classification." *BMC Bioinformatics*, 9:319, 2008. |
| MAQC-II consortium | Shi et al. "The MicroArray Quality Control (MAQC)-II study of common practices for the development and validation of microarray-based predictive models." *Nature Biotechnology*, 28:827–838, 2010. |
| Leukemia (Golub) | Golub et al. "Molecular classification of cancer: class discovery and class prediction by gene expression monitoring." *Science*, 286(5439):531–537, 1999. |
| MLL leukemia | Armstrong et al. "MLL translocations specify a distinct gene expression profile that distinguishes a unique leukemia." *Nature Genetics*, 30:41–47, 2002. |
| Glioma (Nutt) | Nutt et al. "Gene expression-based classification of malignant gliomas." *Cancer Research*, 63(7):1602–1607, 2003. |

---

## Further reading

- Brown, Pocock, Zhao & Luján. "Conditional likelihood maximisation: a unifying framework for information theoretic feature selection." *JMLR*, 13:27–66, 2012. (Unified view of MI, JMI, CMIM, mRMR.)
- Huang, Pocock & Zhao. "Feature selection using EATS threshold." *IEEE Access*, 2025. (Screening criterion used in Tier-2.)
- Candès, Fan, Janson & Lv. "Panning for gold." *J. Royal Statistical Society B*, 2018. (Knockoff theory underpinning copula knockoffs.)

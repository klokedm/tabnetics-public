---
title: Methods and References
nav_order: 3
---

# Background — Methods and References

Tabnetics implements both **novel contributions** developed as part of this project and **established methods** from the feature selection, distribution fitting, and HDLSS classification literature. This document maps each component to its theoretical foundation.

---

## Novel contributions

These components were developed specifically for tabnetics or adapted here to the HDLSS setting. Where there is a direct conceptual precursor, it is cited explicitly.

### MNPO — Nash Multi-Portfolio Optimization

The core aggregation engine. Tabnetics' MNPO formulates method-portfolio selection as a multiplayer game and solves for a Nash equilibrium via KL-regularized mirror descent on the method-weight simplex (Freund & Schapire, 1999). The multiplayer Nash framing draws conceptual inspiration from Wu et al., [*Multiplayer Nash Preference Optimization*](https://arxiv.org/abs/2509.23102), which generalizes Nash-style optimization from two-player to multiplayer preference settings in the context of LLM alignment.

However, the tabnetics adaptation differs from Wu et al. in several fundamental ways that make it a distinct contribution rather than a direct application:

- **Fixed methods vs. evolving policies** — In Wu et al., players are LLM policies that update their parameters across iterations. In tabnetics, the 36 feature-selection methods are fixed algorithms; the "game" determines portfolio weights over them, not policy updates.
- **Heterogeneous oracles vs. shared preference model** — Wu et al.'s formal convergence guarantees (§3.1 of the paper) apply only to the homogeneous case where all players share one preference oracle. Tabnetics uses 2–11 heterogeneous oracles (performance, stability, complexity, etc.), placing it in the general-sum regime where the paper explicitly states no formal convergence guarantees hold (§3.3).
- **Small-sample regime** — Wu et al. train on 60K+ samples with unlimited reward-model queries. Tabnetics estimates each pairwise preference from 5 CV folds, yielding a 6-point discrete scale with substantial quantization noise.
- **Pairwise, not Plackett-Luce** — The paper's key multiplayer theoretical contribution (Plackett-Luce listwise comparisons) is not used; tabnetics constructs standard pairwise preference matrices.

The solver itself — mirror descent on a simplex with KL regularization toward a reference prior — is mathematically well-established independently of the Wu et al. paper. What Tabnetics adds to HDLSS:

- **Method portfolios instead of policy populations** — mirror descent selects portfolio weights over heterogeneous HDLSS selectors and classifier candidates.
- **HDLSS-specific oracle utilities** — the utility matrix is built from balanced accuracy, stability, complexity, robustness, and diversity signals that matter when `p >> n`.
- **Pipeline-level integration** — the Nash portfolio is embedded inside a distribution-aware, regime-aware, validation-gated HDLSS pipeline rather than used as a standalone optimization objective.

Key novel elements inside this HDLSS adaptation:

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

Implementation note: several methods and benchmark backends are exposed through optional third-party libraries (for example Boruta, SHAP, MAPIE, FLAML, TabPFN, and pytabkit). Their upstream licenses/terms still apply when those integrations are enabled; see [Using Tabnetics -> Third-party integrations and licenses](USING.md#third-party-integrations-and-licenses).

### Feature selection — stability-based

| Method | Reference |
|---|---|
| Stability Selection (Lasso) | Meinshausen & Bühlmann. ["Stability selection."](https://doi.org/10.1111/j.1467-9868.2010.00740.x) *J. Royal Statistical Society B*, 72(4):417–473, 2010. |
| Complementary Subsampling | Shah & Samworth. ["Variable selection with error control."](https://doi.org/10.1111/j.1467-9868.2012.01046.x) *J. Royal Statistical Society B*, 75(1):55–80, 2013. |
| TIGRESS | Haury et al. ["TIGRESS: Trustful Inference of Gene REgulation using Stability Selection."](https://doi.org/10.1186/1752-0509-6-145) *BMC Systems Biology*, 6:145, 2012. |
| IPSS (Integrated Path Stability Selection) | Melikechi et al. ["Integrated path stability selection."](https://arxiv.org/abs/2403.15877) arXiv:2403.15877, 2024. |
| Cluster Stability Selection | Faletto & Bien. ["Cluster stability selection."](https://doi.org/10.1016/j.csda.2022.107579) *Computational Statistics & Data Analysis*, 177:107579, 2022. |

### Feature selection — knockoff-based

| Method | Reference |
|---|---|
| Copula Knockoffs (D-vine) | Román-Vásquez et al. ["Vine copula knockoff filter for high-dimensional controlled variable selection."](https://arxiv.org/abs/2410.00650) arXiv:2410.00650, 2024. |
| Knockoff Filter (general framework) | Candès et al. ["Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection."](https://doi.org/10.1111/rssb.12265) *J. Royal Statistical Society B*, 80(3):551–577, 2018. |
| Derandomized Knockoffs | Ren & Candès. ["Derandomizing knockoffs."](https://arxiv.org/abs/2205.00556) arXiv:2205.00556, 2022. |

### Feature selection — filter and information-theoretic

| Method | Reference |
|---|---|
| mRMR (Minimum Redundancy Maximum Relevance) | Peng, Long & Ding. ["Feature selection based on mutual information."](https://doi.org/10.1109/TPAMI.2005.159) *IEEE Trans. Pattern Analysis & Machine Intelligence*, 27(8):1226–1238, 2005. |
| JMI (Joint Mutual Information) | Yang & Moody. ["Data visualization and feature selection: new algorithms for nongaussian data."](https://papers.nips.cc/paper_files/paper/1999/hash/8c01a75941549a705cf7275e41b21f0d-Abstract.html) *NIPS*, 1999. |
| CMIM (Conditional Mutual Information Maximisation) | Fleuret. ["Fast binary feature selection with conditional mutual information."](https://jmlr.org/papers/v5/fleuret04a.html) *JMLR*, 5:1531–1555, 2004. |
| FCBF (Fast Correlation-Based Filter) | Yu & Liu. ["Efficient feature selection via analysis of relevance and redundancy."](https://jmlr.org/papers/v5/yu04a.html) *JMLR*, 5:1205–1224, 2004. |
| HSIC Lasso | Climente-González et al. ["Block HSIC Lasso: model-free biomarker detection for ultra-high dimensional data."](https://doi.org/10.1093/bioinformatics/btz333) *Bioinformatics*, 35(14):i427–i435, 2019. |

### Feature selection — tree and wrapper

| Method | Reference |
|---|---|
| Boruta | Kursa & Rudnicki. ["Feature selection with the Boruta package."](https://doi.org/10.18637/jss.v036.i11) *J. Statistical Software*, 36(11):1–13, 2010. |
| RFECV (Recursive Feature Elimination) | Guyon et al. ["Gene selection for cancer classification using support vector machines."](https://doi.org/10.1023/A:1012487302797) *Machine Learning*, 46:389–422, 2002. |
| TreeSHAP | Lundberg et al. ["From local explanations to global understanding with explainable AI for trees."](https://doi.org/10.1038/s42256-019-0138-9) *Nature Machine Intelligence*, 2:56–67, 2020. |

### Feature selection — multiclass-specific

| Method | Reference |
|---|---|
| Nearest Shrunken Centroids | Tibshirani et al. ["Diagnosis of multiple cancer types by shrunken centroids of gene expression."](https://doi.org/10.1073/pnas.082099299) *PNAS*, 99(10):6567–6572, 2002. |
| OVA ensemble | Rifkin & Klautau. ["In defense of one-vs-all classification."](https://jmlr.org/papers/v5/rifkin04a.html) *JMLR*, 5:101–141, 2004. Representative source for the one-vs-all decomposition that the Tabnetics selector adapts to feature ranking. |
| ECOC class-aware decomposition | Dietterich & Bakiri. ["Solving multiclass learning problems via error-correcting output codes."](https://doi.org/10.1613/jair.105) *JAIR*, 2:263–286, 1995. |
| SIR / SAVE / PFC (sufficient dimension reduction) | Li. ["Sliced inverse regression for dimension reduction."](https://doi.org/10.1080/01621459.1991.10475035) *JASA*, 86(414):316–327, 1991; Cook & Weisberg. ["Discussion of Li (1991)."](https://doi.org/10.1080/01621459.1991.10475036) *JASA*, 86(414):328–332, 1991; Cook. ["Principal fitted components for dimension reduction in regression."](https://doi.org/10.1214/08-STS275) *Statistical Science*, 22(1):1–26, 2008. |

### Feature selection — pairwise and AUC-based

| Method | Reference |
|---|---|
| WMW AUC filter | Bamber. ["The area above the ordinal dominance graph and the area below the receiver operating characteristic graph."](https://doi.org/10.1016/0022-2496(75)90001-2) *Journal of Mathematical Psychology*, 12(4):387–415, 1975. Provides the ROC/AUC interpretation behind the Wilcoxon-Mann-Whitney ranking used here. |
| k-TSP (k Top Scoring Pairs) | Tan et al. ["Simple decision rules for classifying human cancers from gene expression profiles."](https://doi.org/10.1093/bioinformatics/bti631) *Bioinformatics*, 21(20):3896–3904, 2005. |
| Joint AUC+L1 selector | Ma et al. ["Prediction-based structured variable selection through the receiver operating characteristic curves."](https://doi.org/10.1111/j.1541-0420.2010.01533.x) *Biometrics*, 67(3):896–905, 2011. Representative source for sparse ROC/AUC-aware logistic selection in the same family as the Tabnetics implementation. |

### Feature selection — game-theoretic weights

| Concept | Reference |
|---|---|
| Multiplayer Nash preference framing (conceptual inspiration for Tabnetics MNPO) | Wu et al. [*Multiplayer Nash Preference Optimization*](https://arxiv.org/abs/2509.23102). arXiv:2509.23102, 2025. Tabnetics draws conceptual inspiration from the multiplayer Nash framing but differs fundamentally in player semantics, oracle structure, and data regime (see MNPO section above). |
| Multiplicative weights / online mirror descent | Freund & Schapire. ["Adaptive game playing using multiplicative weights."](https://doi.org/10.1006/game.1999.0738) *Games and Economic Behavior*, 29(1–2):79–103, 1999. The algorithmic foundation for MNPO's equilibrium solver. |
| Banzhaf value (oracle weighting) | Wang & Jia. ["Data Banzhaf: A Robust Data Valuation Framework for Machine Learning."](https://arxiv.org/abs/2205.15466) *AISTATS*, 2023. |
| Kernel Banzhaf | Liu et al. ["KernelSHAP-IQ: Weighted Least Square Optimization for Shapley Interactions."](https://arxiv.org/abs/2405.10852) arXiv:2405.10852, 2024. |
| Shapley value | Shapley. ["A value for n-person games."](https://doi.org/10.1017/CBO9780511528446.003) *Contributions to the Theory of Games*, 2:307–317, 1953. |
| QRE (Quantal Response Equilibrium) | McKelvey & Palfrey. ["Quantal response equilibria for normal form games."](https://doi.org/10.1006/game.1995.1023) *Games and Economic Behavior*, 10(1):6–38, 1995. |

### Distribution fitting

| Component | Reference |
|---|---|
| Parametric families (20+) | Standard implementations: normal, log-normal, gamma, Weibull, beta, GEV, GPD, Johnson $S_B$/$S_U$, skew-normal, folded-normal, inverse-Gaussian, Burr III/XII, Dagum, sinh-arcsinh, etc. via `scipy.stats`. |
| L-moment prescreening | Hosking. ["L-moments: analysis and estimation of distributions using linear combinations of order statistics."](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x) *J. Royal Statistical Society B*, 52(1):105–124, 1990. |
| Bootstrap-calibrated GOF | Parametric bootstrap following Efron & Tibshirani. [*An Introduction to the Bootstrap*](https://www.routledge.com/An-Introduction-to-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317), 1994, to calibrate Kolmogorov–Smirnov and Cramér–von Mises p-values for small samples. |
| Maximum product spacing (MPS) | Ranneby. ["The maximum spacing method. An estimation method related to the maximum likelihood method."](https://www.jstor.org/stable/4615946) *Scandinavian J. Statistics*, 11(2):93–112, 1984. |
| CRPS scoring | Gneiting & Raftery. ["Strictly proper scoring rules, prediction, and estimation."](https://doi.org/10.1198/016214506000001437) *JASA*, 102(477):359–378, 2007. |

### Batch correction

| Method | Reference |
|---|---|
| ComBat | Johnson, Li & Rabinovic. ["Adjusting batch effects in microarray expression data using empirical Bayes methods."](https://doi.org/10.1093/biostatistics/kxj037) *Biostatistics*, 8(1):118–127, 2007. |

### Classification

| Method | Reference |
|---|---|
| PLS-DA | Barker & Rayens. ["Partial least squares for discrimination."](https://doi.org/10.1002/cem.785) *J. Chemometrics*, 17(3):166–173, 2003. |
| DLDA (Diagonal LDA) | Dudoit, Fridlyand & Speed. ["Comparison of discrimination methods for the classification of tumors using gene expression data."](https://doi.org/10.1198/016214502753479248) *JASA*, 97(457):77–87, 2002. |
| TabPFN | Hollmann et al. ["Accurate predictions on small data with a tabular foundation model."](https://doi.org/10.1038/s41586-024-08328-6) *Nature*, 637:319–326, 2025. |
| TabM | Gorishniy et al. ["TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling."](https://openreview.net/forum?id=jJSqueaKkc) *ICLR*, 2025. Two backends: numpy approximation (`tabm`) and official PyTorch implementation via pytabkit (`tabm_official`). |
| RealMLP | Holzmüller et al. ["Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data."](https://openreview.net/forum?id=BVBbFqWXmj) *NeurIPS*, 2024. Two backends: numpy approximation (`realmlp`) and official RealMLP-TD via pytabkit (`realmlp_td`). |
| CPDA (Copula Probabilistic DA) | Internal contribution. Copula-based probabilistic discriminant analysis for HDLSS classification; fits marginal CDFs per class and models joint dependence via a Gaussian copula. |
| pytabkit | Holzmüller et al. ["Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data."](https://openreview.net/forum?id=BVBbFqWXmj) *NeurIPS*, 2024. Optional backend library providing sklearn-compatible wrappers for TabM and RealMLP-TD. |
| Conformal prediction (MAPIE) | Taquet et al. ["MAPIE: an open-source library for distribution-free uncertainty quantification."](https://arxiv.org/abs/2207.12274) arXiv:2207.12274, 2022. |
| UBayFS | Jenul et al. ["UBayFS: An R package for user guided feature selection."](https://doi.org/10.21105/joss.04848) *JOSS*, 7(79):4848, 2022. |

Tabnetics treats conformal prediction as an **uncertainty and efficiency layer**, not as a point-accuracy optimizer. For the singleton-rate / compactness interpretation used in the validation analyses, see [Wang, Sun & Dobriban 2025](https://doi.org/10.48550/arXiv.2509.24095) and [Hallberg Szabadváry et al. 2025](https://doi.org/10.1016/j.mlwa.2025.100664).

### Multi-omics

| Component | Reference |
|---|---|
| DIABLO-style multi-block PLS | Singh et al. ["DIABLO: an integrative approach for identifying key molecular drivers from multi-omics assays."](https://doi.org/10.1093/bioinformatics/bty1054) *Bioinformatics*, 35(17):3055–3062, 2019. |
| MINT batch correction | Rohart et al. ["MINT: a multivariate integrative method to identify reproducible molecular signatures across independent experiments and platforms."](https://doi.org/10.1186/s12859-017-1553-8) *BMC Bioinformatics*, 18:128, 2017. |
| Multi-omics review (cancer) | Cai et al. ["Machine learning for multi-omics data integration in cancer."](https://doi.org/10.1016/j.isci.2022.103798) *iScience*, 25(2):103798, 2022. |

---

## Benchmark datasets

Tabnetics includes a curated registry of HDLSS benchmark datasets. Key sources:

| Source | Reference |
|---|---|
| CuMiDa (curated microarrays) | Feltes et al. ["CuMiDa: An extensively curated microarray database for benchmarking and testing of machine learning approaches."](https://doi.org/10.1089/cmb.2018.0238) *J. Computational Biology*, 26(4):376–386, 2019. |
| de Souto benchmark | de Souto et al. ["Clustering cancer gene expression data: a comparative study."](https://doi.org/10.1186/1471-2105-9-497) *BMC Bioinformatics*, 9:497, 2008. |
| Statnikov multi-category | Statnikov et al. ["A comprehensive comparison of random forests and support vector machines for microarray-based cancer classification."](https://doi.org/10.1186/1471-2105-9-319) *BMC Bioinformatics*, 9:319, 2008. |
| MAQC-II consortium | Shi et al. ["The MicroArray Quality Control (MAQC)-II study of common practices for the development and validation of microarray-based predictive models."](https://doi.org/10.1038/nbt.1665) *Nature Biotechnology*, 28:827–838, 2010. |
| Leukemia (Golub) | Golub et al. ["Molecular classification of cancer: class discovery and class prediction by gene expression monitoring."](https://doi.org/10.1126/science.286.5439.531) *Science*, 286(5439):531–537, 1999. |
| MLL leukemia | Armstrong et al. ["MLL translocations specify a distinct gene expression profile that distinguishes a unique leukemia."](https://doi.org/10.1038/ng765) *Nature Genetics*, 30:41–47, 2002. |
| Glioma (Nutt) | Nutt et al. ["Gene expression-based classification of malignant gliomas."](https://pubmed.ncbi.nlm.nih.gov/12670911/) *Cancer Research*, 63(7):1602–1607, 2003. |

---

## Further reading

- Brown, Pocock, Zhao & Luján. ["Conditional likelihood maximisation: a unifying framework for information theoretic feature selection."](https://jmlr.org/papers/v13/brown12a.html) *JMLR*, 13:27–66, 2012. (Unified view of MI, JMI, CMIM, mRMR.)
- Huang, Pocock & Zhao. "Feature selection using EATS threshold." *IEEE Access*, 2025. (Screening criterion used in Tier-2.)
- Candès, Fan, Janson & Lv. ["Panning for gold."](https://doi.org/10.1111/rssb.12265) *J. Royal Statistical Society B*, 2018. (Knockoff theory underpinning copula knockoffs.)

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

---
title: Home
nav_order: 1
nav_exclude: false
---

# Tabnetics

A Python toolkit for **high-dimensional, low-sample-size (HDLSS)** tabular classification. Tabnetics grew out of the review paper [*Machine learning on small size samples: A synthetic knowledge synthesis*](https://doi.org/10.1177/00368504211029777), which provided the library's initial theoretical background for small-sample learning. The library combines distribution-aware preprocessing, portfolio-based feature selection, and game-theoretic method aggregation into a single pipeline designed for settings where `p >> n`.

**Homepage:** [tabnetics.org](https://tabnetics.org)

```bash
pip install tabnetics
```

This base install now matches the ready-to-run public runtime surface and ships every currently packaged direct dependency except TabPFN. For the fully loaded opt-in stack, including TabPFN, use `pip install "tabnetics[full]"`.

Leakage-safe training balancing is also opt-in: install
`pip install "tabnetics[balancing]"` and select one of `smote`,
`propensity_match`, `random_over`, or `random_under` through
`TrainingBalanceConfig`. The default remains `none`; validation/test rows are
never resampled. See the [usage guide](https://tabnetics.org/USING.html#training-class-balancing).

Licensed under [Apache 2.0](https://github.com/klokedm/tabnetics-public/blob/main/LICENSE).

Tabnetics depends on several third-party libraries with separate licenses and use terms. Those upstream terms still apply when you install or enable the related Tabnetics feature. Review [Third-party integrations and licenses](https://tabnetics.org/USING.html#third-party-integrations-and-licenses) before using the packaged dependencies or the `full` extra.

## Documentation map

- [Using Tabnetics](USING.md) for end-to-end usage, CSV workflows, benchmarks, multi-omics examples, and third-party integration/license notes.
- [Auto Router](AUTO_ROUTER.md) for the default V25 router model, rationale, usage, and evidence summary.
- [Methods and References](BACKGROUND.md) for MNPO positioning, implemented methods, and literature anchors.
- [Benchmark Results](RESULTS.md) for validation summaries and dataset provenance.
- [Tabnetics Diakrino Results](TABENTICS_DIAKRINO_RESULTS.md) for the paired feature-ranking campaign and its scope.
- [Results Browser](results-browser.md) for interactive filtering, charts, and result tables backed by static browser-side data bundles.
- [Profile Config Browser](profile-configs.md) for the fixed settings, observed run-time variations, and seed sets behind each published profile.
- [Browser Data Guide](results-browser-data.md) for the campaign, family, and dataset catalog behind the browser filters.
- [How It Works](how-it-works.md) for the module-level flow across the library.
- [Reference](reference/index.md) for AST-derived package and API summaries.

**Contact:** [marko@tabnetics.org](mailto:marko@tabnetics.org) &ensp;|&ensp; **[Discussions](https://github.com/klokedm/tabnetics-public/discussions)**

## What's new in 1.5.0

**Tabnetics Diakrino — a neural HDLSS oracle, now available (gated).** Tabnetics 1.5.0 introduces Tabnetics Diakrino (from the Greek *διακρίνω*, "to discern"): a ~279M-parameter in-context transformer that reads an entire high-dimensional table at once and proposes which features matter. It ships inside the package as an **opt-in** feature-selection sidecar under `tabnetics.classification`, and the trained weights live separately as a **gated model** on Hugging Face — [`klokedm/tabnetics-diakrino`](https://huggingface.co/klokedm/tabnetics-diakrino) (manual access approval, Apache-2.0).

**A research component, not a new default.** Diakrino is opt-in on purpose. In controlled tests its ranking is *not* additive to the classical selector union on real HDLSS panels, so it is meant to be combined with the classical stack rather than substituted for it. The [Diakrino results page](TABENTICS_DIAKRINO_RESULTS.md) has the paired campaign evidence and the limitations, stated plainly. A peer-reviewed article describing the work is in preparation.

**Pre-trained on EuroHPC MareNostrum 5.** Part of the model was pre-trained on the EuroHPC supercomputer MareNostrum 5 at the Barcelona Supercomputing Center; see [Acknowledgements](#acknowledgements).

## What's new in 1.1.0

**Packaged auto router, explicit opt-in.** Tabnetics ships the V25 calibrated score-router inside the package. Enable it with `DFFSConfig(auto_router_enabled=True)` to compute training-data descriptors, predict balanced accuracy and macro-F1 for supported candidate profiles, and apply its calibrated policy before the pipeline starts. No network download is required. The default remains off until V25 passes a frozen registered-holdout campaign.

**Router evidence published.** The V25 router was trained with 10-fold dataset-level CV on 57 training datasets and 513 policy groups, excluding the frozen holdout IDs and using only dataset-computable descriptors. Its calibrated policy improves mean balanced accuracy by **+0.0038** and macro-F1 by **+0.0053** versus the current default-like candidate on the training-CV policy groups, with **124 / 513** non-default selections and **264 / 513** policy-defaulted selections. The latest available frozen-router holdout evidence predates V25 and remains labeled as context rather than claimed V25 holdout validation.

**Results browser refreshed.** The public interactive browser now includes an Auto Router tab with the V25 policy summary, per-dataset out-of-fold deltas, candidate selection counts, and holdout status alongside the Val-18 through Val-21 benchmark bundle.

**Manual flags remain the default.** `DFFSConfig()` keeps the router off, so explicit method/config flags remain authoritative. Router experiments must opt in with `auto_router_enabled=True`.

## When to use Tabnetics

Tabnetics is built for tabular classification problems where the number of features greatly exceeds the number of samples:

- **Transcriptomics** — microarray and RNA-seq gene expression
- **Proteomics and metabolomics** — mass-spec feature matrices
- **Other HDLSS settings** — any structured tabular problem with `p >> n`

In these regimes the dominant failure modes are not model selection — they are unstable preprocessing, brittle feature selection, information leakage, and inflated validation estimates. Tabnetics addresses all four.

What Tabnetics adds to the HDLSS problem is not just another selector: it turns many unstable HDLSS choices into a multiplayer portfolio game. Feature-selection methods and classifier candidates are treated as competing players, oracle scores become the payoff structure, and the resulting MNPO equilibrium is used to select a robust portfolio under small-sample constraints.

**[Usage guide →](https://tabnetics.org/USING.html)** · **[Methods & references →](https://tabnetics.org/BACKGROUND.html)** · **[Benchmark results →](https://tabnetics.org/RESULTS.html)** · **[Tabnetics Diakrino results →](https://tabnetics.org/TABENTICS_DIAKRINO_RESULTS.html)** · **[Results browser →](https://tabnetics.org/results-browser.html)** · **[Announcements](https://github.com/klokedm/tabnetics-public/discussions/categories/announcements)**

## Benchmark results

Tabnetics has been evaluated on **63 primary HDLSS benchmark datasets** (41-7,000 samples, 500-100,001 features, 2-14 classes) drawn from OpenML, GEO, CuMiDa, Scikit-feature, and UCSC Xena/TCGA, plus **7 additional Val-21 phase-2 RV holdout datasets**. Across the consolidated Val-18 through Val-21 evidence base, the refreshed local results summarize **57,217 runs** with **278 pipeline profiles**. On the primary panel, best profiles exceed published strict-holdout ranges on **29 of 63 datasets**, fall within range on **26**, and fall below range on **8**. Detailed per-dataset results, statistical comparisons, and article references are available in [RESULTS.md](https://tabnetics.org/RESULTS.html), and the static [results browser](https://tabnetics.org/results-browser.html) exposes the same public surfaces as interactive charts and tables. A refreshed cross-campaign technical report with full methodology, sidecar analysis, and next-experiment recommendations is available.

## Auto router

The bundled V25 router is available as an opt-in method/config selector. When enabled, it chooses among supported, already-tested candidates using only descriptors available on a new dataset: sample and feature counts, class-balance statistics, feature-distribution summaries, complexity measures, and candidate action encodings. It does not use holdout labels, validation tiers, or dataset identity.

V25 remains opt-in because its reported gains are training-corpus OOF evidence and its frozen registered-holdout evaluation is pending. See [Auto Router](https://tabnetics.org/AUTO_ROUTER.html) for the model card, evidence summary, opt-in flag, and direct inspection API.

## Key ideas

1. **Distribution-aware preprocessing.** Each feature is fitted to a parametric family (from 20+ candidates) using goodness-of-fit testing, bootstrap calibration, and L-moment prescreening. CDF-based transforms replace ad-hoc normalization.

2. **Portfolio feature selection.** Forty feature-selection paths are available overall, including 39 engineered selectors plus a random-baseline reference used in validation. The benchmark MNPO portfolios combine stability selectors, copula knockoffs, tree-based importance, mutual-information filters, IPSS, HSIC-Lasso, and more into a single robust HDLSS feature portfolio. MNPO builds pairwise preference matrices from multiple oracles (performance, stability, complexity, etc.) and solves for a Nash equilibrium via KL-regularized mirror descent. The multiplayer game framing draws conceptual inspiration from Wu et al.'s [*Multiplayer Nash Preference Optimization*](https://arxiv.org/abs/2509.23102), though the HDLSS adaptation is a distinct contribution with different players, oracles, and data regime (see [BACKGROUND.md](https://tabnetics.org/BACKGROUND.html) for details).

3. **Regime-aware classification.** An MNPO-based classifier oracle picks from regime-appropriate pools. The HDLSS extreme pool now spans **28** candidate backends, including linear/GLM baselines, LDA/QDA families, bias-corrected HDLSS discriminants, PLS and sparse PLS variants, random-projection and random-Fourier lifts, nearest-subspace and spatial-median geometry, copula-style discriminants, confusion-pursuit, DWD / ECOC wrappers, and lightweight or full-fidelity deep-tabular models (TabM, RealMLP). The HDLSS moderate pool carries **29** backends by adding RBF SVM, GPC, KNN, vote ensembles, and TabPFN on top of the HDLSS-safe core. Standard-regime routing can also unlock tree families such as RF, Extra Trees, XGBoost, LightGBM, and CatBoost. The base package now includes the always-shipped boosted-tree and `pytabkit` backends directly; TabPFN remains the only kept opt-in classifier path, and custom environments still degrade gracefully if any third-party backend is absent.

4. **Strict validation.** All learned preprocessing and selection is train-only. The HuggingFace bundle is the authoritative reproducibility mirror of the public upstream datasets used for validation. Synthetic fallback is not allowed for evidence-bearing runs.

## Quick start

The quick-start path assumes the recommended expanded install above, so the broader optional selector and backend surface is already available when you need it.

```python
from tabnetics.pipeline import DistributionFeatureSelectionPipeline, DFFSConfig

config = DFFSConfig(random_seed=42)
pipeline = DistributionFeatureSelectionPipeline(config)

result = pipeline.run(X, y, dataset_name="my_dataset", seed=42)

print(f"Accuracy: {result.accuracy:.3f}")
print(f"Selected features: {result.selected_features}")
```

The quick-start config uses the fixed manual defaults. To evaluate the packaged router, use `DFFSConfig(auto_router_enabled=True)`.

## Operational defaults

The packaged runtime currently follows the promoted post-review workflow:

- `auto_router_enabled=False` is the default; set it to `True` to let the V25 packaged score-router choose a supported method/config profile from the training split before DF/FS/classification runs.
- `df_stage_position="after_fs"` is the default, so distribution fitting runs on the feature space that actually survives selection.
- Evidence-bearing benchmark and validation runs treat the HuggingFace bundle as the authoritative reproducibility mirror of the public upstream datasets and default to `dataset_integrity_policy="error"`.
- Conformal prediction is opt-in and should be interpreted as an uncertainty layer (coverage, prediction-set size, singleton rate), not as a balanced-accuracy optimizer.
- Training balancing is default-off. When explicitly enabled, it is fitted only on each classifier-CV training fold and final-fit training rows; held-out prevalence is untouched.
- `multiomics_adapter="split_halves"` is a benchmark-time shortcut; real multi-omics studies should use explicit blocks with `tabnetics.multiomics`.

## Command line workflows

Editable installs expose installed wrappers, and every wrapper has the same packaged `python -m ...` equivalent:

```bash
tabnetics-benchmark --datasets leukemia_golub --seeds 11 23 37
tabnetics-validation-plan --plan-kind validation17 --num-pods 4
tabnetics-validation-suite --dataset-sets fs_easy --seeds 11 23 37
```

The corresponding module entrypoints are:

- `python -m tabnetics.benchmarks.cli`
- `python -m tabnetics.validation.generate_plan`
- `python -m tabnetics.validation.core.shard_runner`
- `python -m tabnetics.validation.suite`

## Selected literature anchors

The full methods table lives in [BACKGROUND.md](https://tabnetics.org/BACKGROUND.html). For a quick orientation, these are the main papers behind the current public positioning:

- Kokol. [*Machine learning on small size samples: A synthetic knowledge synthesis*](https://doi.org/10.1177/00368504211029777) — the original HDLSS review context behind the library.
- Freund & Schapire. [*Adaptive game playing using multiplicative weights*](https://doi.org/10.1006/game.1999.0738) — the mirror-descent / multiplicative-weights foundation used by the MNPO solver.
- Wu et al. [*Multiplayer Nash Preference Optimization*](https://arxiv.org/abs/2509.23102) — conceptual inspiration for the multiplayer Nash framing, not the solver implementation.
- Marron et al. [*Distance-Weighted Discrimination*](https://doi.org/10.1198/016214507000000677), Lê Cao et al. [*Sparse PLS discriminant analysis*](https://doi.org/10.1186/1471-2105-12-253), and Dietterich & Bakiri [*Error-Correcting Output Codes*](https://doi.org/10.1613/jair.105) — key anchors behind the expanded HDLSS classifier pool.
- Rahimi & Recht [*Random Features for Large-Scale Kernel Machines*](https://papers.neurips.cc/paper/3182-random-features-for-large-scale-kernel-machines), Hall et al. [*Median-Based Classifiers for High-Dimensional Data*](https://doi.org/10.1198/jasa.2009.tm08107), Tsuda [*Subspace classifier in the Hilbert space*](https://doi.org/10.1016/S0167-8655(99)00023-9), and Han et al. [*CODA*](https://jmlr.org/papers/v14/han13a.html) — the kernel, robust, subspace, and copula-style classifier families reflected in the current backend surface.
- Singh et al. [*DIABLO*](https://doi.org/10.1093/bioinformatics/bty1054) and Rohart et al. [*MINT*](https://doi.org/10.1186/s12859-017-1553-8) — the reference points for explicit multi-omics integration.
- Taquet et al. [*MAPIE: an open-source library for distribution-free uncertainty quantification*](https://arxiv.org/abs/2207.12274) — the conformal/UQ reference behind the classifier-side uncertainty outputs.

## Package structure

| Subpackage | Purpose |
|---|---|
| `tabnetics.auto_router` | Packaged V25 router model, dataset descriptor builder, and runtime profile application |
| `tabnetics.core` | MNPO game-theoretic primitives, sklearn compatibility layer, runtime configuration |
| `tabnetics.distribution` | Univariate distribution fitting (20+ families), bootstrap GOF, CDF-based transforms |
| `tabnetics.feature_selection` | 30 selection methods, MNPO portfolio aggregation, copula knockoffs, stability selectors |
| `tabnetics.classification` | Regime-aware classifier pools, classical and specialist HDLSS backends, MNPO classifier oracle, and conformal helpers |
| `tabnetics.pipeline` | End-to-end DF+FS+classification pipeline with leakage prevention |
| `tabnetics.datasets` | Dataset registry, HuggingFace/OpenML loaders, meta-feature extraction |
| `tabnetics.domains` | Domain adapters (bioinformatics prefilters, face-domain projection) |
| `tabnetics.multiomics` | Multi-block PLS-DA (DIABLO-style) and MINT batch-correction integration |
| `tabnetics.benchmarks` | Benchmark runner, method-set profiles, SOTA comparison, gaming detection |
| `tabnetics.validation` | Validation campaign planner, shard execution, promotion gates |

## Feature selection methods

The `FeatureSelector` supports 40 methods out of the box, including 39 engineered selectors plus a random-baseline reference path:

| Category | Methods |
|---|---|
| **Stability selectors** | Lasso stability, subspace stability, decorrelated stability, cluster stability, TIGRESS |
| **Wrapper methods** | RFECV (SVM, RF, LR), Boruta |
| **Filter methods** | ANOVA F-test, mutual information, mRMR, JMI, CMIM, FCBF, Wilcoxon AUC |
| **Tree-based** | GBDT importance, TreeSHAP, random forest |
| **Knockoff methods** | Copula knockoff (D-vine, FDR-controlled via e-values), derandomized knockoffs |
| **Embedded** | OA-Elastic Net, Joint AUC+L1, HSIC-Lasso |
| **Other** | IPSS, k-TSP, OVA/ECOC wrappers, Rashomon importance |

Methods are aggregated via MNPO with configurable oracle presets (`minimal`, `perf_only`, `perf_complexity`, `full`, etc.).

See [BACKGROUND.md](https://tabnetics.org/BACKGROUND.html) for the full list of implemented papers, [USING.md](https://tabnetics.org/USING.html) for detailed usage, and [RESULTS.md](https://tabnetics.org/RESULTS.html) for benchmark results.

## Installation

Recommended install for the quick-start and usage-guide examples:

```bash
pip install tabnetics
```

Full install profile (same package plus the opt-in TabPFN path):

```bash
pip install "tabnetics[full]"
```

## Requirements

- Python >= 3.11
- Plain `tabnetics` is the ready-to-run public runtime surface
- Optional install profile: `full` adds the TabPFN path

## Acknowledgements

Part of the **Tabnetics Diakrino** model (see [Tabnetics Diakrino results](TABENTICS_DIAKRINO_RESULTS.md)) was pre-trained on the EuroHPC supercomputer **MareNostrum 5** within the scope of a EuroHPC Joint Undertaking project:

This work was granted access to the EuroHPC supercomputer MareNostrum 5, hosted by the Barcelona Supercomputing Center (BSC-CNS, Spain), through a EuroHPC AI Factory Fast Lane Access call (project EHPC-AIF-2026FL01-314, "Neural HDLSS Oracle for High-Dimensional Tabular AI"). Part of the Tabnetics Diakrino model was pre-trained on MareNostrum 5 within the scope of this EuroHPC project.

We acknowledge the EuroHPC Joint Undertaking and the Barcelona Supercomputing Center for the allocation and support.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

# Tabnetics

A Python toolkit for **high-dimensional, low-sample-size (HDLSS)** tabular classification. Tabnetics grew out of the review paper [*Machine learning on small size samples: A synthetic knowledge synthesis*](https://doi.org/10.1177/00368504211029777), which provided the library's initial theoretical background for small-sample learning. The library combines distribution-aware preprocessing, portfolio-based feature selection, and game-theoretic method aggregation into a single pipeline designed for settings where `p >> n`.

**Homepage:** [tabnetics.org](https://tabnetics.org)

```bash
pip install tabnetics
```

This base install now matches the ready-to-run public runtime surface and ships every currently packaged direct dependency except TabPFN. For the fully loaded opt-in stack, including TabPFN, use `pip install "tabnetics[full]"`.

Licensed under [Apache 2.0](https://github.com/klokedm/tabnetics-public/blob/main/LICENSE).

Tabnetics depends on several third-party libraries with separate licenses and use terms. Those upstream terms still apply when you install or enable the related Tabnetics feature. Review [Third-party integrations and licenses](https://tabnetics.org/USING.html#third-party-integrations-and-licenses) before using the packaged dependencies or the `full` extra.

## What's new in 0.5.2

**Post-cleanup evidence refresh.** The public benchmark pages now reflect the reconciled Val-18/Val-19/Val-20 evidence base after duplicate cleanup and loader hardening: **54,882 successful runs** across **271 pipeline profiles** on **63 HDLSS datasets**. The best profiles clear **31/63** datasets at balanced accuracy `>= 0.90`, achieve **8** perfect-classification datasets, and land **29 / 26 / 8** in the strict-holdout SOTA split (**above / within / below**).

**Cross-campaign technical report.** A unified technical report covering the full Val-18/Val-19/Val-20 evidence base is now available. The report presents matched-overlap paired comparisons with Wilcoxon signed-rank tests, per-tier decomposition, and a pre-registered hypothesis scorecard covering 22 testable hypotheses across the three campaigns. Key findings: BH filtering is strongly validated as a safety feature (disabling costs −0.046 BA on 63 datasets, p < 1e-4), FLAML at 60 s is the cleanest frontier improvement (+0.003 BA, p = 0.017), and the full FS+DF pipeline scaffold remains directionally beneficial.

**Expanded classifier surface.** The HDLSS extreme-regime pool includes **28** classifiers and the moderate-regime pool includes **29**, covering bias-corrected DA/SVM variants, CPDA, copula-style and robust classifiers, RFF / nearest-subspace geometries, HDRDA-style regularized discriminants, DWD, sparse PLS-DA, ECOC lifts, and the optional full-fidelity `pytabkit` paths for TabM and RealMLP.

## When to use Tabnetics

Tabnetics is built for tabular classification problems where the number of features greatly exceeds the number of samples:

- **Transcriptomics** — microarray and RNA-seq gene expression
- **Proteomics and metabolomics** — mass-spec feature matrices
- **Other HDLSS settings** — any structured tabular problem with `p >> n`

In these regimes the dominant failure modes are not model selection — they are unstable preprocessing, brittle feature selection, information leakage, and inflated validation estimates. Tabnetics addresses all four.

What Tabnetics adds to the HDLSS problem is not just another selector: it turns many unstable HDLSS choices into a multiplayer portfolio game. Feature-selection methods and classifier candidates are treated as competing players, oracle scores become the payoff structure, and the resulting MNPO equilibrium is used to select a robust portfolio under small-sample constraints.

**[Usage guide →](https://tabnetics.org/USING.html)** · **[Methods & references →](https://tabnetics.org/BACKGROUND.html)** · **[Benchmark results →](https://tabnetics.org/RESULTS.html)** · **[Results browser →](https://tabnetics.org/results-browser.html)** · **[Announcements](https://github.com/klokedm/tabnetics-public/discussions/categories/announcements)**

## Call for collaboration

We are actively looking for **testers**, **collaborators**, and **co-authors** to help validate Tabnetics on real-world HDLSS datasets, shape the companion article, and improve the codebase. If you work with high-dimensional tabular data — transcriptomics, proteomics, metabolomics, or similar — we would love to hear from you. See the [Discussions](https://github.com/klokedm/tabnetics-public/discussions) page for ongoing conversations, or open a new thread to introduce your use case.

## Citation

If you use Tabnetics in research, cite the repository for the specific version you used. The library is still under active development, and a companion paper will be published after the current testing and validation cycle is complete.

Repository URL: https://github.com/klokedm/tabnetics-public

```bibtex
@software{kokol_tabnetics_2026,
  author = {Kokol, Marko},
  title = {Tabnetics},
  year = {2026},
  url = {https://github.com/klokedm/tabnetics-public}
}
```

## Benchmark results

Tabnetics has been evaluated on **63 HDLSS benchmark datasets** (41–7,000 samples, 500–100,001 features, 2–14 classes) drawn from OpenML, GEO, CuMiDa, Scikit-feature, and UCSC Xena/TCGA. Across the reconciled Val-18/Val-19/Val-20 evidence base, the public results now summarize **54,882 runs** with **271 pipeline profiles**. The best profiles exceed published strict-holdout SOTA on **29 of 63 datasets (46%)** and match or exceed published ranges on **55 of 63 (87%)**. Eight datasets achieve perfect classification and 31 exceed 0.90 balanced accuracy. Detailed per-dataset results, statistical comparisons, and article references are available in [RESULTS.md](https://tabnetics.org/RESULTS.html), and the static [results browser](https://tabnetics.org/results-browser.html) exposes the same public surfaces as interactive charts and tables. A cross-campaign technical report with full methodology and ablation studies is available, and a peer-reviewed article is in preparation.

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

## Operational defaults

The packaged runtime currently follows the promoted post-review workflow:

- `df_stage_position="after_fs"` is the default, so distribution fitting runs on the feature space that actually survives selection.
- Evidence-bearing benchmark and validation runs treat the HuggingFace bundle as the authoritative reproducibility mirror of the public upstream datasets and default to `dataset_integrity_policy="error"`.
- Conformal prediction is opt-in and should be interpreted as an uncertainty layer (coverage, prediction-set size, singleton rate), not as a balanced-accuracy optimizer.
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

## Development

```bash
git clone https://github.com/klokedm/tabnetics-public.git
cd tabnetics-public
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0 — see [LICENSE](https://github.com/klokedm/tabnetics-public/blob/main/LICENSE).

---

*This documentation is auto-generated from internal notes and sources with the support of rule-based transformations and generative AI. Errors are possible — please report any issues via [Discussions](https://github.com/klokedm/tabnetics-public/discussions).*

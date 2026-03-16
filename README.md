# Tabnetics

A Python toolkit for **high-dimensional, low-sample-size (HDLSS)** tabular classification. Tabnetics grew out of the review paper [*Machine learning on small size samples: A synthetic knowledge synthesis*](https://doi.org/10.1177/00368504211029777), which provided the library's initial theoretical background for small-sample learning. The library combines distribution-aware preprocessing, portfolio-based feature selection, and game-theoretic method aggregation into a single pipeline designed for settings where `p >> n`.

**Homepage:** [tabnetics.org](https://tabnetics.org)

```bash
pip install "tabnetics[feature-selection-optional,benchmarks]"
```

This recommended install enables the expanded selector/backend library set immediately. For the lean core-only package, use `pip install tabnetics`.

Licensed under [Apache 2.0](LICENSE).

Optional integrations and benchmark backends can rely on third-party libraries with separate licenses and use terms. Those upstream terms still apply when you enable the related Tabnetics feature. Review [Third-party integrations and licenses](https://tabnetics.org/USING#third-party-integrations-and-licenses) before using extras such as TabPFN, FLAML, XGBoost, LightGBM, CatBoost, MAPIE, Boruta, SHAP, or pyvinecopulib.

## What's new in 0.5.0

**Beta release line.** `tabnetics==0.5.0` is the first coordinated beta-marked public release in the current package line. The PyPI metadata, GitHub-facing docs, and generated site now all present the library as **Development Status :: 4 - Beta**.

**Latest validation snapshot.** The public benchmark pages now track the newest merged Val-18/Val-19 mirrors: **55,117 successful runs** across **210 pipeline profiles** on **63 HDLSS datasets**. The best profiles clear **31/63** datasets at balanced accuracy `>= 0.90`, achieve **7** perfect-classification datasets, and land **33 / 19 / 11** in the strict-holdout SOTA split (**above / within / below**).

**Val-19 results are fully surfaced.** The release now includes the completed Val-19 added-classifier bridge profiles and the random feature-selection baseline, replacing the older partial Val-19 snapshot in the public benchmark summaries.

**TabArena snapshot refreshed.** The published general-tabular comparison now uses the latest merged 38-dataset TabArena run: **Elo 1012.1**, **binary Elo 1008.3**, **multiclass Elo 1027.3**, and **3 wins / 35 losses / 0 ties** against the current official dataset-best rows. See [TABARENA_RESULTS.md](https://tabnetics.org/TABARENA_RESULTS) for details.

**Expanded classifier surface remains part of the beta release.** The HDLSS extreme-regime pool includes 22 classifiers and the moderate-regime pool expands to 27, including bias-corrected DA/SVM variants, CPDA, Copula-DA, and the optional full-fidelity `pytabkit` paths for TabM and RealMLP.

**Pipeline and docs polish.** The public docs homepage now carries the release notes directly, and the benchmark/docs generation path is aligned with the latest public figures and package metadata for release publishing.

## When to use Tabnetics

Tabnetics is built for tabular classification problems where the number of features greatly exceeds the number of samples:

- **Transcriptomics** — microarray and RNA-seq gene expression
- **Proteomics and metabolomics** — mass-spec feature matrices
- **Other HDLSS settings** — any structured tabular problem with `p >> n`

In these regimes the dominant failure modes are not model selection — they are unstable preprocessing, brittle feature selection, information leakage, and inflated validation estimates. Tabnetics addresses all four.

What Tabnetics adds to the HDLSS problem is not just another selector: it turns many unstable HDLSS choices into a multiplayer portfolio game. Feature-selection methods and classifier candidates are treated as competing players, oracle scores become the payoff structure, and the resulting MNPO equilibrium is used to select a robust portfolio under small-sample constraints.

**[Usage guide →](https://tabnetics.org/USING)** · **[Methods & references →](https://tabnetics.org/BACKGROUND)** · **[Benchmark results →](https://tabnetics.org/RESULTS)** · **[TabArena results →](https://tabnetics.org/TABARENA_RESULTS)** · **[Announcements](https://github.com/klokedm/tabnetics-public/discussions/categories/announcements)**

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

Tabnetics has been evaluated on **63 HDLSS benchmark datasets** (41–7,000 samples, 500–100,001 features, 2–14 classes) drawn from OpenML, GEO, CuMiDa, Scikit-feature, and UCSC Xena/TCGA. Across the latest merged Val-18/Val-19 mirrors, the public results now summarize **55,117 runs** with **210 pipeline profiles**. The best profiles exceed published strict-holdout SOTA on **33 of 63 datasets (52%)** and match or exceed published ranges on **52 of 63 (83%)**. Seven datasets achieve perfect classification and 31 exceed 0.90 balanced accuracy. Detailed per-dataset results, statistical comparisons, and article references are available in [RESULTS.md](https://tabnetics.org/RESULTS). A peer-reviewed article with full methodology and ablation studies is in preparation.

**General tabular data (TabArena).** To provide a transparent reference on non-HDLSS data, Tabnetics was evaluated on the [TabArena](https://arxiv.org/abs/2506.08828) benchmark, a NeurIPS 2025 suite of 38 general tabular classification datasets. Using the `general` profile with XGBoost and LightGBM enabled, and scoring the merged local artifact with the same Elo-based leaderboard machinery used by TabArena, the current informational snapshot gives `tabnetics (general)` an overall **Elo 1012.1**, with **binary Elo 1008.3** and **multiclass Elo 1027.3** across the full 38-dataset coverage. A dedicated `general_tabular` profile is now available for N>>p datasets — it uses a tree-weighted classifier pool, skips HDLSS-specific machinery (screening, folding, CDF transforms), and adapts feature selection by the sample-to-feature ratio. Full results and caveats are in [TABARENA_RESULTS.md](https://tabnetics.org/TABARENA_RESULTS); profile details are in [USING.md](https://tabnetics.org/USING#tabarena-pipeline-profiles).

## Key ideas

1. **Distribution-aware preprocessing.** Each feature is fitted to a parametric family (from 20+ candidates) using goodness-of-fit testing, bootstrap calibration, and L-moment prescreening. CDF-based transforms replace ad-hoc normalization.

2. **Portfolio feature selection.** Forty feature-selection paths are available overall, including 39 engineered selectors plus a random-baseline reference used in validation. The benchmark MNPO portfolios combine stability selectors, copula knockoffs, tree-based importance, mutual-information filters, IPSS, HSIC-Lasso, and more into a single robust HDLSS feature portfolio. MNPO builds pairwise preference matrices from multiple oracles (performance, stability, complexity, etc.) and solves for a Nash equilibrium via KL-regularized mirror descent. The multiplayer game framing draws conceptual inspiration from Wu et al.'s [*Multiplayer Nash Preference Optimization*](https://arxiv.org/abs/2509.23102), though the HDLSS adaptation is a distinct contribution with different players, oracles, and data regime (see [BACKGROUND.md](https://tabnetics.org/BACKGROUND) for details).

3. **Regime-aware classification.** An MNPO-based classifier oracle picks from regime-appropriate pools. The HDLSS extreme pool includes 22 classifiers spanning linear/GLM, LDA, SVM, PLS-DA, NSC, naive Bayes, random projection, distance-based DA, QDA, neural network, kernel approximation, subspace/manifold, robust distance, copula/generative, shrinkage/confusion-pursuit, and deep tabular families. The deep tabular backends (TabM, RealMLP) are available as both lightweight numpy-only approximations and as full-fidelity PyTorch implementations via optional `pytabkit` integration. Moderate-regime pools add RBF SVM, GPC, KNN, vote ensembles, and TabPFN, while standard-regime routing can also unlock tree families such as RF, Extra Trees, XGBoost, LightGBM, and CatBoost. All optional backends (CatBoost, LightGBM, TabPFN, pytabkit) degrade gracefully when not installed.

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

**TabArena benchmark** (general tabular data evaluation against leaderboard-best methods):

```bash
python -m experiments.benchmarking.tabarena_benchmark \
    --dataset-sets all --profile general --seeds 42 \
    --max-workers 12 --max-train-samples 50000 --task-timeout-sec 3600
```

**TabArena results page regeneration** (official-style leaderboard scoring for the local artifact):

```bash
python -m scripts.analysis.generate_tabarena_results \
    --results-csv run_artifacts/tabarena_general_archml/tabarena_results.csv \
    run_artifacts/tabarena_general_archml_rerun6/tabarena_results.csv \
    --output-dir run_artifacts/tabarena_general_archml \
    --write-markdown core/TABARENA_RESULTS.md
```

See [TABARENA_RESULTS.md](https://tabnetics.org/TABARENA_RESULTS) for results and interpretation.

## Selected literature anchors

The full methods table lives in [BACKGROUND.md](https://tabnetics.org/BACKGROUND). For a quick orientation, these are the main papers behind the current public positioning:

- Kokol. [*Machine learning on small size samples: A synthetic knowledge synthesis*](https://doi.org/10.1177/00368504211029777) — the original HDLSS review context behind the library.
- Freund & Schapire. [*Adaptive game playing using multiplicative weights*](https://doi.org/10.1006/game.1999.0738) — the mirror-descent / multiplicative-weights foundation used by the MNPO solver.
- Wu et al. [*Multiplayer Nash Preference Optimization*](https://arxiv.org/abs/2509.23102) — conceptual inspiration for the multiplayer Nash framing, not the solver implementation.
- Singh et al. [*DIABLO*](https://doi.org/10.1093/bioinformatics/bty1054) and Rohart et al. [*MINT*](https://doi.org/10.1186/s12859-017-1553-8) — the reference points for explicit multi-omics integration.
- Taquet et al. [*MAPIE: an open-source library for distribution-free uncertainty quantification*](https://arxiv.org/abs/2207.12274) — the conformal/UQ reference behind the classifier-side uncertainty outputs.

## Package structure

| Subpackage | Purpose |
|---|---|
| `tabnetics.core` | MNPO game-theoretic primitives, sklearn compatibility layer, runtime configuration |
| `tabnetics.distribution` | Univariate distribution fitting (20+ families), bootstrap GOF, CDF-based transforms |
| `tabnetics.feature_selection` | 30 selection methods, MNPO portfolio aggregation, copula knockoffs, stability selectors |
| `tabnetics.classification` | Regime-aware classifier pools, MNPO classifier oracle, PLS-DA, conformal helpers |
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

See [BACKGROUND.md](https://tabnetics.org/BACKGROUND) for the full list of implemented papers, [USING.md](https://tabnetics.org/USING) for detailed usage, and [RESULTS.md](https://tabnetics.org/RESULTS) for benchmark results.

## Installation

Recommended broad install for the quick-start and usage-guide examples:

```bash
pip install "tabnetics[feature-selection-optional,benchmarks]"
```

Minimal core-only install (numpy, pandas, scipy, scikit-learn):

```bash
pip install tabnetics
```

Feature-selection extras only (boruta, copula support, conformal prediction, statsmodels-based multiple-testing correction):

```bash
pip install tabnetics[feature-selection-optional]
```

With full benchmark support (FLAML, LightGBM, XGBoost, TabPFN, etc.):

```bash
pip install tabnetics[benchmarks]
```

## Requirements

- Python >= 3.11
- numpy, pandas, scipy, scikit-learn (core)
- See `pyproject.toml` for optional dependency groups

## Development

```bash
git clone https://github.com/klokedm/tabnetics-public.git
cd tabnetics-public
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*This documentation is auto-generated from internal notes and sources with the support of rule-based transformations and generative AI. Errors are possible — please report any issues via [Discussions](https://github.com/klokedm/tabnetics-public/discussions).*

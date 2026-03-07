# Tabnetics

A Python toolkit for **high-dimensional, low-sample-size (HDLSS)** tabular classification. Tabnetics grew out of the review paper [*Machine learning on small size samples: A synthetic knowledge synthesis*](https://doi.org/10.1177/00368504211029777), which provided the library's initial theoretical background for small-sample learning. The library combines distribution-aware preprocessing, portfolio-based feature selection, and game-theoretic method aggregation into a single pipeline designed for settings where `p >> n`.

```bash
pip install tabnetics
```

Licensed under [Apache 2.0](LICENSE).

## When to use Tabnetics

Tabnetics is built for tabular classification problems where the number of features greatly exceeds the number of samples:

- **Transcriptomics** — microarray and RNA-seq gene expression
- **Proteomics and metabolomics** — mass-spec feature matrices
- **Other HDLSS settings** — any structured tabular problem with `p >> n`

In these regimes the dominant failure modes are not model selection — they are unstable preprocessing, brittle feature selection, information leakage, and inflated validation estimates. Tabnetics addresses all four.

**[Usage guide →](USING.md)** · **[Methods & references →](BACKGROUND.md)** · **[Benchmark results →](RESULTS.md)**

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

Tabnetics has been evaluated on **35 HDLSS benchmark datasets** (50–2,600 samples, 500–100,000 features, 2–14 classes) drawn from OpenML, GEO, Scikit-feature, and UCSC Xena/TCGA. Across 2,800+ runs with 9 random seeds per dataset, the pipeline achieves a mean balanced accuracy of **0.80**, with 12 of 35 datasets above 0.90 and 3 reaching perfect classification. The MNPO portfolio consistently outperforms single-method baselines, and distribution-aware preprocessing contributes a small but consistent positive effect. Detailed per-dataset results, statistical comparisons, and dataset provenance are available in [RESULTS.md](RESULTS.md). A peer-reviewed article with full methodology and ablation studies is in preparation.

## Key ideas

1. **Distribution-aware preprocessing.** Each feature is fitted to a parametric family (from 20+ candidates) using goodness-of-fit testing, bootstrap calibration, and L-moment prescreening. CDF-based transforms replace ad-hoc normalization.

2. **Portfolio feature selection.** Thirty feature-selection methods — stability selectors, copula knockoffs, tree-based importance, mutual-information filters, IPSS, HSIC-Lasso, and more — are run together. A game-theoretic oracle (MNPO — Nash Multi-Portfolio Optimization) aggregates their outputs into a single robust feature set.

3. **Regime-aware classification.** An MNPO-based classifier oracle picks from regime-appropriate pools (LR, SVM, LDA, PLS-DA, NSC for extreme HDLSS; plus RF, XGBoost, CatBoost, TabPFN for moderate regimes).

4. **Strict validation.** All learned preprocessing and selection is train-only. HuggingFace-hosted datasets are the authoritative source. Synthetic fallback is not allowed for evidence-bearing runs.

## Quick start

```python
from tabnetics.pipeline import DistributionFeatureSelectionPipeline, DFFSConfig

config = DFFSConfig(random_seed=42)
pipeline = DistributionFeatureSelectionPipeline(config)

result = pipeline.run(X, y, dataset_name="my_dataset", seed=42)

print(f"Accuracy: {result.accuracy:.3f}")
print(f"Selected features: {result.selected_features}")
```

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

The `FeatureSelector` supports 30 methods out of the box, including:

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

See [BACKGROUND.md](BACKGROUND.md) for the full list of implemented papers, [USING.md](USING.md) for detailed usage, and [RESULTS.md](RESULTS.md) for benchmark results.

## Installation

Core dependencies (numpy, pandas, scipy, scikit-learn):

```bash
pip install tabnetics
```

With optional feature-selection extras (boruta, copula support, conformal prediction):

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

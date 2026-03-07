# Using Tabnetics

This guide covers practical usage of tabnetics — from running the full pipeline on your own data to configuring individual components.

---

## Table of contents

- [Full pipeline](#full-pipeline)
- [Configuration](#configuration)
- [Standalone feature selection](#standalone-feature-selection)
- [Standalone distribution fitting](#standalone-distribution-fitting)
- [Running benchmarks](#running-benchmarks)
- [Datasets](#datasets)
- [Oracle presets](#oracle-presets)
- [Feature selection methods](#feature-selection-methods)
- [Method profiles (benchmark)](#method-profiles)
- [Multi-omics](#multi-omics)

---

## Full pipeline

The main entry point is `DistributionFeatureSelectionPipeline`. It handles train/test splitting, distribution fitting, feature selection, and classification in a single leakage-safe call.

```python
from tabnetics.pipeline import DistributionFeatureSelectionPipeline, DFFSConfig
import numpy as np

# X: (n_samples, n_features) array
# y: (n_samples,) array of class labels

config = DFFSConfig(
    random_seed=42,
    test_size=0.20,
    n_final_features=50,
    n_jobs=4,
)

pipeline = DistributionFeatureSelectionPipeline(config)
result = pipeline.run(X, y, dataset_name="my_dataset")

print(f"Balanced accuracy: {result.balanced_accuracy:.3f}")
print(f"Model:             {result.model_name}")
print(f"Features selected: {result.selected_features_count}")
print(f"Feature indices:   {result.selected_feature_indices_original}")
```

### Pre-split mode

If you manage your own splits (e.g., for nested cross-validation), use `run_pre_split()`:

```python
result = pipeline.run_pre_split(
    X_train, y_train, X_test, y_test,
    dataset_name="my_dataset",
    seed=42,
)
```

### Result object

`PipelineRunResult` contains:

| Field | Description |
|---|---|
| `accuracy` | Test-set accuracy |
| `balanced_accuracy` | Balanced accuracy (macro-averaged recall) |
| `macro_f1` | Macro F1 score |
| `hybrid_score` | Weighted combination of balanced accuracy and macro F1 |
| `roc_auc` | ROC AUC (binary or OvR multiclass) |
| `selected_features_count` | Number of features selected |
| `selected_feature_indices_original` | Indices into the original feature matrix |
| `model_name` | Name of the classifier chosen by the oracle |
| `distribution_summaries` | Per-feature distribution fit results |
| `fs_time_sec` | Feature selection wall time |
| `dist_time_sec` | Distribution fitting wall time |

---

## Configuration

`DFFSConfig` is a dataclass with ~120 fields. Most have sensible defaults. The key groups are:

### Core

| Parameter | Default | Description |
|---|---|---|
| `random_seed` | `42` | Global random seed |
| `test_size` | `0.20` | Fraction held out for test |
| `n_final_features` | `50` | Target number of features after selection |
| `n_jobs` | `1` | Parallelism for FS methods and distribution fitting |
| `fs_fraction` | `0.40` | Fraction of training data used for feature selection |

### Distribution fitting

| Parameter | Default | Description |
|---|---|---|
| `dist_criterion` | `"simple"` | Criterion: `simple`, `cvm_p`, `ks_p`, `aic`, `bic`, `aicc`, `cv`, `cv_loglik`, `crps`, `mnpo_oracle` |
| `apply_cdf_transform` | `True` | Apply CDF transform after fitting |
| `df_stage_position` | `"after_fs"` | `before_fs` or `after_fs` |
| `max_dist_features` | `256` | Max features to distribution-fit (skip rest) |

### Prefilter (Tier 1)

| Parameter | Default | Description |
|---|---|---|
| `use_rank_prefilter` | `True` | Enable univariate prefilter before FS |
| `prefilter_top_k` | `600` | Keep top-k features from prefilter |
| `prefilter_strategies` | `("mi_ftest_blend","rf_importance","wsnr","bh_fdr")` | Prefilter scoring strategies |
| `batch_correction` | `"none"` | Batch correction: `none`, `combat`, `combat_seq`, `cdf_center`, `center_scale` |

### Screening (Tier 2)

| Parameter | Default | Description |
|---|---|---|
| `screening_enabled` | `True` | Enable interaction-aware screening |
| `screening_method` | `"evalue"` | Screening method: `stir`, `evalue`, `none` |

### Feature selection

| Parameter | Default | Description |
|---|---|---|
| `enabled_methods` | *7-method default stack* | Tuple of method keys to run |
| `fs_portfolio_size` | `5` | MNPO portfolio max candidates |
| `fs_oracle_weighting_mode` | `"banzhaf"` | Oracle weighting: `tritrust`, `uniform`, `shapley`, `banzhaf` |

### Classification

| Parameter | Default | Description |
|---|---|---|
| `model_candidates` | `("lr","svm_rbf","svm_linear","dlda","knn","rf","nb","elastic_net_lr")` | Classifier pool |
| `folding_method` | `"pls_da"` | Dimensionality reduction: `none`, `rff`, `tensor_sketch`, `pls_da` |
| `scaler_mode` | `"standard"` | Input scaling: `standard`, `robust`, `quantile` |

### Opt-in features

| Parameter | Default | Description |
|---|---|---|
| `enable_ratio_features` | `False` | Construct log-ratio features (pairs of original features) |
| `regime_gating_enabled` | `False` | Route datasets to regime-appropriate profiles |
| `eval_models_enabled` | `True` | Multi-classifier evaluation proxy |

---

## Standalone feature selection

Use `FeatureSelector` directly when you only need feature selection without the full pipeline:

```python
from tabnetics.feature_selection import FeatureSelector

fs = FeatureSelector(
    random_state=42,
    selection_strategy="mnpo_portfolio",
    portfolio_size=6,
    n_folds=5,
    n_bootstrap_iterations=10,
)

X_selected, fs_result = fs.fit_transform(
    X_train, y_train, n_final_features=30, return_result_object=True
)

# Selected feature indices
print(fs_result.selected_feature_indices)

# Per-method results
for method, info in fs_result.method_results.items():
    print(f"{method}: {len(info.get('selected', []))} features")
```

### Configuring the MNPO oracle

```python
from tabnetics.feature_selection import OracleConfig

oracle = OracleConfig.from_preset("full")
# Presets: perf_only, perf_complexity, perf_complexity_stability, full, minimal_cvar

# Or customize:
oracle = OracleConfig(
    weighting_mode="banzhaf",
    diversity_mode="mi_redundancy",
    use_cvar=True,
    cvar_alpha=0.33,
)
```

---

## Standalone distribution fitting

Fit a single feature to the best parametric distribution:

```python
from tabnetics.distribution.selector import UnifiedDistributionSelectorV6

selector = UnifiedDistributionSelectorV6(robust_mode=True, n_jobs=1)
best_name, best_fit, all_fits = selector.select_best_distribution(
    data_column,          # 1-D numpy array
    criterion="simple",   # or cvm_p, ks_p, aic, bic, cv, crps, mnpo_oracle
)

print(f"Best distribution: {best_name}")
print(f"Parameters: {best_fit.params}")
print(f"KS p-value: {best_fit.ks_pvalue:.4f}")
```

Available criteria:

| Criterion | Description |
|---|---|
| `simple` | Fast prescreening with KS p-value and CVM ranking (default) |
| `cvm_p` | Cramér–von Mises p-value |
| `ks_p` | Kolmogorov–Smirnov p-value |
| `aic` / `bic` / `aicc` | Information criteria |
| `cv` / `cv_loglik` | Cross-validated log-likelihood |
| `crps` | Continuous ranked probability score |
| `mnpo_oracle` | Game-theoretic multi-criterion selection |

---

## Running benchmarks

### Command line

```bash
# Run on a specific dataset
tabnetics-benchmark --datasets leukemia_golub --seeds 11 23 37

# Run on a named dataset group
tabnetics-benchmark --dataset-sets fs_easy --max-workers 8

# Run with a specific method profile
tabnetics-benchmark --datasets leukemia_golub dlbcl_shipp \
    --fs-method-set mnpo_broad_stable \
    --seeds 42

# Full benchmark with distribution fitting diagnostics
tabnetics-benchmark --dataset-sets core \
    --dist-criterion simple \
    --df-stage-position after_fs \
    --compute-budget standard \
    --max-workers 16 \
    --seeds 11 23 37
```

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--datasets` | — | Space-separated dataset IDs |
| `--dataset-sets` | — | Named groups: `core`, `fs_easy`, `fs_medium`, `fs_hard`, `smoke` |
| `--seeds` | `11 23 37` | Random seeds for repeated runs |
| `--max-workers` | `1` | Parallel dataset workers |
| `--test-size` | `0.20` | Test split fraction |
| `--fs-method-set` | — | Named method profile (see [profiles](#method-profiles)) |
| `--dist-criterion` | `simple` | Distribution fitting criterion |
| `--df-stage-position` | `after_fs` | `before_fs` or `after_fs` |
| `--compute-budget` | `standard` | `fast`, `standard`, `thorough` |
| `--prefilter-top-k` | `600` | Prefilter feature count |
| `--screening-enabled` | `False` | Enable Tier-2 screening |
| `--screening-method` | `none` | `stir`, `evalue`, `none` |
| `--eval-models-enabled` | `False` | Multi-classifier evaluation proxy |
| `--task-timeout-sec` | `300` | Per-dataset timeout |
| `--quiet-worker-logs` | `False` | Suppress worker output |
| `--enable-nestedcv-audit` | `False` | Nested CV robustness audit |

### Programmatic

```python
from tabnetics.benchmarks.cli import main

# Equivalent to CLI invocation
import sys
sys.argv = ["tabnetics-benchmark", "--datasets", "leukemia_golub", "--seeds", "42"]
main()
```

---

## Datasets

Tabnetics ships with a registry of 70+ HDLSS benchmark datasets. Most are loaded from HuggingFace.

### Registry

```python
from tabnetics.datasets import CATALOG, DATASET_SETS

# List all registered datasets
for ds_id, spec in CATALOG.items():
    print(f"{ds_id}: {spec.display_name} ({spec.n_samples}×{spec.n_features}, {spec.n_classes} classes)")

# List named dataset groups
print(list(DATASET_SETS.keys()))
# ['all', 'smoke', 'core', 'extended', 'fs_all', 'fs_easy', 'fs_medium', ...]
```

### Dataset groups

| Group | Description |
|---|---|
| `smoke` | 3 datasets for quick sanity checks |
| `core` | All non-extended datasets |
| `extended` | Full catalog including CuMiDa and TCGA |
| `fs_easy` / `fs_medium` / `fs_hard` / `fs_very_hard` | FS pipeline datasets by difficulty tier |

### Example datasets

| ID | Name | Samples | Features | Classes |
|---|---|---|---|---|
| `leukemia_golub` | Leukemia (Golub) | 72 | 7,129 | 2 |
| `dlbcl_shipp` | DLBCL (Shipp) | 77 | 5,469 | 2 |
| `ovarian_petricoin` | Ovarian Cancer (Petricoin) | 253 | 15,154 | 2 |
| `srbct_khan` | SRBCT (Khan) | 83 | 2,308 | 4 |
| `prostate_singh` | Prostate Cancer (Singh) | 102 | 12,600 | 2 |
| `carcinom_11class` | Carcinom 11-class | 174 | 9,182 | 11 |
| `nci9_60_9class` | NCI9 | 60 | 9,712 | 9 |
| `gla_bra_180` | GLA-BRA-180 | 180 | 49,151 | 4 |

---

## Oracle presets

The MNPO oracle can be configured with presets via `OracleConfig.from_preset()`:

| Preset | Oracles | Use case |
|---|---|---|
| `perf_only` | Performance | Fastest; single-criterion selection |
| `perf_complexity` | Performance + Complexity | Prefer simpler feature sets |
| `perf_complexity_stability` | Performance + Complexity + Stability | Add bootstrap stability |
| `full` | Performance + Stability + Complexity + Robust + Diversity | Production default — all 5 oracles |
| `minimal_cvar` | Performance + CVaR | Tail-risk focus for small datasets |

---

## Feature selection methods

All 35+ methods in the registry, grouped by paradigm:

### Stability

| Key | Label |
|---|---|
| `stability_lasso` | Stability Selection (Lasso) |
| `stability_subsample` | Stability Selection (complementary subsampling) |
| `tigress_stability` | TIGRESS-style Stability Selection |
| `subspace_stability` | Subspace Stability Selection |
| `decorrelated_stability` | Decorrelated Stability Selection |
| `ipss` | Integrated Path Stability Selection (IPSS) |
| `cluster_stability` | Cluster Stability Selection |

### Wrapper

| Key | Label |
|---|---|
| `rfecv` | Recursive Feature Elimination |
| `boruta` | Boruta |
| `iterative_redundancy_pruning` | Iterative redundancy-pruning wrapper |
| `iterative_redundancy_pruning_bounded` | Iterative redundancy-pruning wrapper (runtime-bounded) |

### Filter

| Key | Label |
|---|---|
| `mutual_information` | Mutual Information |
| `anova_f` | ANOVA F-test |
| `chi_square` | Chi-Square univariate filter |
| `relieff` | ReliefF instance-based filter |
| `fcbf` | FCBF correlation-based filter |
| `cmim` | CMIM conditional MI filter |
| `hsic_lasso` | HSIC Lasso-style kernelized selection |
| `mrmr_jmi` | mRMR/JMI redundancy-aware selection |

### Embedded

| Key | Label |
|---|---|
| `gradient_boosting` | Gradient Boosting |
| `linear_svm` | Linear SVM |
| `treeshap` | TreeSHAP embedded selector |
| `oaenet` | OAENet adaptive elastic-net selector |
| `slce_centroid_encoder` | SLCE centroid-encoder selection |
| `group_sparse_lasso` | Group sparse lasso |

### Pairwise / AUC

| Key | Label |
|---|---|
| `wmw_auc` | WMW univariate AUC filter |
| `joint_auc_l1` | Joint AUC-aware L1 selector (binary only) |
| `ktsp` | k-TSP pairwise rank selection |

### Knockoff

| Key | Label |
|---|---|
| `copula_knockoff` | Copula knock-off selection |

### Multiclass

| Key | Label |
|---|---|
| `ova_ensemble` | OVA multiclass ensemble selection |
| `ecoc_class_aware` | ECOC class-aware decomposition selection |
| `joint_multiclass_support` | Joint multiclass shared-support selection |
| `dove_class_specific` | DOvE-style class-specific multiclass selection |
| `sparse_multinomial` | Sparse multinomial multiclass selection |
| `nearest_shrunken_centroid` | Nearest shrunken centroids multiclass selection |
| `class_pareto_front` | Class-specific Pareto-front multiclass selection |
| `sir_sdr` / `save_sdr` / `pfc_sdr` | Sufficient dimension reduction selectors |

---

## Method profiles

Named method sets for benchmark runs. Use with `--fs-method-set <profile>` on the CLI.

| Profile | Methods | Notes |
|---|---|---|
| `strict_plus_mrmr` | GB, SVM, MI, ANOVA, mRMR | 5-method baseline |
| `strict_plus_mrmr_auc` | Baseline + WMW AUC | 6-method |
| `mnpo_copula_extended` | Baseline + copula knockoff | Knockoff expansion |
| `mnpo_ipss_extended` | Baseline + IPSS | Stability expansion |
| `mnpo_broad_stable` | 14 production-safe methods | Adds Boruta, copula knockoff, decorrelated stability, ReliefF, stability lasso, RFECV, HSIC lasso |
| `mnpo_v14_core` | 15 methods | broad_stable + joint multiclass support |
| `mnpo_v14_core_plus_ipss` | 16 methods | v14_core + IPSS |
| `mnpo_broad_all` | 36 methods | Exhaustive — all non-deprecated selectors |

---

## Multi-omics

Tabnetics includes DIABLO-style multi-block PLS-DA and MINT batch correction for multi-omics integration:

```python
from tabnetics.multiomics import MultiBlockPLSDA

# X_blocks: list of arrays, one per omics layer
# y: shared class labels
model = MultiBlockPLSDA(n_components=3)
model.fit(X_blocks, y)
scores = model.transform(X_blocks)
```

See `tabnetics.multiomics` for full API.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---
title: Using Tabnetics
nav_order: 2
---

# Using Tabnetics

This guide covers practical usage of tabnetics — from running the full pipeline on your own data to configuring individual components.

For the ready-to-run setup used across this guide, install:

```bash
pip install tabnetics
```

That base install now matches the ready-to-run public runtime surface and includes every currently shipped direct dependency except TabPFN. For the fully loaded opt-in stack, including TabPFN, use `pip install "tabnetics[full]"`. Optional integrations still keep their own upstream licenses/terms; see [Third-party integrations and licenses](#third-party-integrations-and-licenses).

---

## Table of contents

- [Full pipeline](#full-pipeline)
- [Auto router](#auto-router)
- [Third-party integrations and licenses](#third-party-integrations-and-licenses)
- [Your own CSV files](#your-own-csv-files)
- [Configuration](#configuration)
- [Standalone feature selection](#standalone-feature-selection)
- [Standalone distribution fitting](#standalone-distribution-fitting)
- [Running benchmarks](#running-benchmarks)
- [Validation campaigns](#validation-campaigns)
- [Datasets](#datasets)
- [Reproducibility and data source policy](#reproducibility-and-data-source-policy)
- [Uncertainty and conformal outputs](#uncertainty-and-conformal-outputs)
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

This default config uses the packaged V25 auto-router. It computes a training-only dataset descriptor, selects a supported method/config candidate, then runs the delegated pipeline with the router disabled to avoid recursion.

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

### Current runtime defaults

The packaged defaults match the current benchmark/validation path:

- `auto_router_enabled=True` is the default, so the V25 calibrated score-router chooses the supported profile from the training split before distribution fitting, feature selection, and classification.
- `df_stage_position="after_fs"` is the promoted default, so the distribution-fitting stage operates on the selected feature space rather than the full raw matrix.
- Evidence-bearing benchmark and validation runs default to `allow_synthetic_fallback=False` and `dataset_integrity_policy="error"`.
- Conformal prediction remains opt-in and is interpreted as uncertainty output, not as a balanced-accuracy gain mechanism.

## Auto router

The auto-router is the recommended entry point for ordinary use. It avoids asking users to pick validation-campaign flags manually and instead selects among supported, already-tested candidates using descriptors computed directly from the training data.

```python
from tabnetics.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline

config = DFFSConfig(random_seed=42, n_jobs=4)
result = DistributionFeatureSelectionPipeline(config).run(X, y, dataset_name="my_dataset")
```

To opt out and keep explicit/manual configuration:

```python
config = DFFSConfig(auto_router_enabled=False)
```

To inspect the router decision before running a full pipeline:

```python
from tabnetics.auto_router import predict_auto_router

decision = predict_auto_router(X_train, y_train)
print(decision.metadata["selected_candidate_id"])
print(decision.enabled_methods)
```

The V25 router predicts balanced accuracy and macro-F1 for 12 finite candidate profiles and applies a conservative calibrated policy. It uses only dataset-computable descriptors and candidate action encodings; it does not use holdout labels, validation tiers, or dataset identity.

## Third-party integrations and licenses

Tabnetics itself is licensed under Apache 2.0, but it depends on several upstream projects with their own licenses or terms. Installing Tabnetics or enabling the `full` extra does not relicense those projects: you still need to comply with the upstream terms for each library you install or activate.

This table covers the direct runtime dependencies declared for the shipped public install surface (`tabnetics` plus opt-in `tabnetics[full]`). It is not a full transitive dependency manifest for every wheel dependency; for redistribution or commercial deployment, review the exact upstream license text for the versions you ship.

| Library | Used for in Tabnetics | Surface | Audited upstream license / terms | Notes |
|---|---|---|---|---|
| [NumPy](https://pypi.org/project/numpy/) (`numpy`) | Core ndarray/matrix operations across the pipeline, selectors, backends, and validation code | Base package | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | PyPI metadata reflects the main BSD-3-Clause project license plus bundled permissive notices. |
| [pandas](https://pypi.org/project/pandas/) (`pandas`) | Dataset loaders, metadata tables, reporting, and artifact assembly | Base package | BSD 3-Clause License | PyPI metadata leads with BSD-3-Clause text. |
| [SciPy](https://pypi.org/project/scipy/) (`scipy`) | Statistical tests, optimization, sparse helpers, and MAT-file IO | Base package | BSD License | PyPI metadata marks SciPy as BSD-licensed. |
| [scikit-learn](https://pypi.org/project/scikit-learn/) (`scikit-learn`) | Estimator interface, preprocessing, CV, metrics, and many baseline models | Base package | BSD-3-Clause | Core runtime dependency. |
| [threadpoolctl](https://github.com/joblib/threadpoolctl) (`threadpoolctl`) | Worker-side thread caps for benchmark/runtime process control | Base package | BSD-3-Clause | Used in the benchmark runner to keep BLAS/OpenMP workers bounded. |
| [tqdm](https://pypi.org/project/tqdm/) (`tqdm`) | Progress bars in copula sampling and data-generation helpers exposed through shipped code paths | Base package | MPL-2.0 AND MIT | Mixed permissive/weak-copyleft metadata; preserve upstream notices if redistributing modified package files. |
| [Boruta](https://github.com/danielhomola/boruta_py) (`boruta`) | Boruta feature selection | Base package; `boruta` method | BSD 3 clause | Included in plain `tabnetics`. |
| [SHAP](https://pypi.org/project/shap/) (`shap`) | TreeSHAP values for the `treeshap` selector | Base package; `treeshap` method | MIT License | Included in plain `tabnetics`. |
| [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib/) (`pyvinecopulib`) | Vine-copula knockoff generation | Base package; `copula_knockoff` methods | MIT | Included in plain `tabnetics`. |
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) (`mapie`) | Conformal prediction / uncertainty wrappers | Base package; conformal classifier outputs | BSD-3-Clause | Included in plain `tabnetics`. |
| [statsmodels](https://www.statsmodels.org/) (`statsmodels`) | BH / multiple-testing correction in prefiltering | Base package | BSD License | Included in plain `tabnetics`. |
| [datasets](https://github.com/huggingface/datasets) (`datasets`) | HuggingFace dataset bundle loading for benchmark/validation workflows | Base package | Apache 2.0 | Included in plain `tabnetics`. |
| [FLAML](https://github.com/microsoft/FLAML) (`flaml`) | AutoML classifier backend/oracle candidates | Base package; `classification_backend="flaml"` | MIT License | Included in plain `tabnetics`. |
| [Optuna](https://pypi.org/project/optuna/) (`optuna`) | Hyperparameter-search classifier backend/oracle candidates | Base package; `classification_backend="optuna"` | MIT License | Included in plain `tabnetics`. |
| [pytabkit](https://github.com/dholzmueller/pytabkit) (`pytabkit`) | Official TabM / RealMLP-TD sklearn-compatible backends | Base package; `tabm_official`, `realmlp_td` | Apache-2.0 | Included in plain `tabnetics`; paired with PyTorch. |
| [PyTorch](https://pytorch.org/) (`torch`) | Runtime for official `pytabkit` backends and benchmark worker/runtime paths | Base package | BSD-3-Clause | Included in plain `tabnetics`. |
| [LightGBM](https://github.com/microsoft/LightGBM) (`lightgbm`) | Gradient-boosted tree classifier candidate | Base package | MIT | Included in plain `tabnetics`. |
| [XGBoost](https://pypi.org/project/xgboost/) (`xgboost`) | Gradient-boosted tree classifier candidate | Base package | Apache-2.0 | Included in plain `tabnetics`. |
| [CatBoost](https://catboost.ai/) (`catboost`) | Gradient-boosted tree classifier candidate | Base package | Apache License, Version 2.0 | Included in plain `tabnetics`. |
| [TabPFN](https://pypi.org/project/tabpfn/) (`tabpfn`) | Foundation-model classifier candidate | `full` extra only; `tabpfn` classifier path | Prior Labs License (Apache 2.0 with additional attribution) | Kept opt-in in `tabnetics[full]`; upstream packaging terms add attribution obligations, and the default TabPFN-2.5 weights remain non-commercial. |

---

## Your own CSV files

For ad-hoc datasets, the simplest path is to load your CSV with `pandas`, keep metadata columns separate, and pass only numeric features into the pipeline. The examples below assume a label column plus a few metadata fields such as sample IDs, study IDs, or batch IDs.

### 1. Default settings

```python
import pandas as pd

from tabnetics.pipeline import DistributionFeatureSelectionPipeline, DFFSConfig

df = pd.read_csv("data/my_hdlss_dataset.csv")

label_col = "label"
metadata_cols = ["sample_id", "study_id", "batch_id"]
feature_cols = [c for c in df.columns if c not in metadata_cols + [label_col]]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
y = df[label_col].to_numpy()

pipeline = DistributionFeatureSelectionPipeline(DFFSConfig(random_seed=42))
result = pipeline.run(X, y, dataset_name="my_hdlss_dataset", seed=42)

print(f"Balanced accuracy: {result.balanced_accuracy:.3f}")
print(f"Selected features: {result.selected_features_count}")
```

This CSV example also uses the auto-router. Add `auto_router_enabled=False` to `DFFSConfig` if you want to force manual defaults.

### 2. Different profile + additional flags

If you want one of the benchmark method profiles on your own CSV, import the profile from `tabnetics.benchmarks.profiles` and layer extra flags on top of it:

```python
import pandas as pd

from tabnetics.benchmarks.profiles import FS_METHOD_SETS
from tabnetics.pipeline import DistributionFeatureSelectionPipeline, DFFSConfig

df = pd.read_csv("data/my_hdlss_dataset.csv")

label_col = "label"
metadata_cols = ["sample_id", "study_id", "batch_id"]
feature_cols = [c for c in df.columns if c not in metadata_cols + [label_col]]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
y = df[label_col].to_numpy()

profile_name = "mnpo_v14_core_plus_ipss"
config = DFFSConfig(
    random_seed=42,
    auto_router_enabled=False,
    enabled_methods=FS_METHOD_SETS[profile_name],
    n_final_features=75,
    prefilter_top_k=800,
    screening_enabled=True,
    screening_method="evalue",
    eval_models_enabled=True,
    df_stage_position="after_fs",
    max_dist_features=512,
)

pipeline = DistributionFeatureSelectionPipeline(config)
result = pipeline.run(X, y, dataset_name="my_hdlss_profiled_csv", seed=42)

print(f"Profile:           {profile_name}")
print(f"Chosen model:      {result.model_name}")
print(f"Balanced accuracy: {result.balanced_accuracy:.3f}")
```

### 3. Multi-omics correction on specific fields

For real multi-omics CSVs with explicit blocks such as transcriptomics, proteomics, or metabolomics, define the block columns yourself and fit the correction/integration step on the training split only. This keeps the workflow leakage-safe.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tabnetics.multiomics import MINTIntegrator
from tabnetics.pipeline import DistributionFeatureSelectionPipeline, DFFSConfig

df = pd.read_csv("data/my_multiomics_dataset.csv")

label_col = "label"
study_fields = ["study_id", "plate_id"]
transcript_cols = [c for c in df.columns if c.startswith("rna_")]
protein_cols = [c for c in df.columns if c.startswith("prot_")]
clinical_cols = ["age", "stage_code"]

y = df[label_col].to_numpy()
study_labels = (
    df[study_fields[0]].astype(str)
    + "__"
    + df[study_fields[1]].astype(str)
).to_numpy()

idx = np.arange(len(df))
train_idx, test_idx = train_test_split(
    idx,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

train_blocks = [
    (
        df.iloc[train_idx][transcript_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float),
        "transcriptomics",
    ),
    (
        df.iloc[train_idx][protein_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float),
        "proteomics",
    ),
]
test_blocks = [
    (
        df.iloc[test_idx][transcript_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float),
        "transcriptomics",
    ),
    (
        df.iloc[test_idx][protein_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float),
        "proteomics",
    ),
]

mint = MINTIntegrator(n_components=2)
mint.fit(train_blocks, y[train_idx], study_labels[train_idx])
Z_train = mint.transform(train_blocks, study_labels[train_idx])
Z_test = mint.transform(test_blocks, study_labels[test_idx])

X_train = np.hstack([
    Z_train,
    df.iloc[train_idx][clinical_cols]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=float),
])
X_test = np.hstack([
    Z_test,
    df.iloc[test_idx][clinical_cols]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=float),
])

pipeline = DistributionFeatureSelectionPipeline(DFFSConfig(random_seed=42))
result = pipeline.run_pre_split(
    X_train,
    y[train_idx],
    X_test,
    y[test_idx],
    dataset_name="my_multiomics_dataset",
    seed=42,
)

print(f"Balanced accuracy: {result.balanced_accuracy:.3f}")
```

If you want explicit multi-block integration without study/batch correction, replace `MINTIntegrator` with `MultiBlockPLSDA`.
If those block matrices contain missing values, impute them on the training split before calling `fit(...)` and `transform(...)`.

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
| `model_candidates` | `("lr","svm_rbf","svm_linear","dlda","knn","rf","nb","elastic_net_lr")` | Classifier pool (legacy default; regime pools override this when regime gating is enabled) |
| `folding_method` | `"pls_da"` | Dimensionality reduction: `none`, `rff`, `tensor_sketch`, `pls_da` |
| `scaler_mode` | `"standard"` | Input scaling: `standard`, `robust`, `quantile` |

#### Regime-aware classifier pools

When `regime_gating_enabled=True`, the pipeline ignores `model_candidates` and uses
regime-appropriate pools defined in `REGIME_POOLS`:

| Regime | Pool size | Key families |
|--------|:---------:|-------------|
| `hdlss_extreme` (n/p < 0.1) | 22 | LR, elastic-net LR, SVM-linear, BC-SVM, RP-ensemble, DLDA, shrinkage-LDA, NSC, PLS-DA, NB, DBDA, GQDA, SGLNN, RFF-LR, near-subspace, spatial-median-DA, copula-DA, CPDA, TabM, RealMLP, TabM-official\*, RealMLP-TD\* |
| `hdlss_moderate` (0.1 ≤ n/p < 1) | 27 | All of extreme + SVM-RBF, GPC, KNN, vote-ensemble, TabPFN |
| `standard` (n/p ≥ 1) | All available | Full pool including tree models (RF, XGBoost, LightGBM, CatBoost) |

\*Included in plain `tabnetics` and `tabnetics[full]`. If `pytabkit` is missing in a custom environment, these paths still fail gracefully.

**Dual implementations.** TabM and RealMLP each have two backends:
- `tabm` / `realmlp`: numpy-only approximations (zero extra deps, millisecond fit times on HDLSS data)
- `tabm_official` / `realmlp_td`: official PyTorch implementations from `pytabkit` (higher fidelity, requires PyTorch)

Both variants are in the regime pools. The oracle evaluates whichever are available and selects the best combination.

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
| `--dataset-integrity-policy` | `error` | Fail, skip, or fallback when class-diversity sanity checks fail |
| `--compute-budget` | `standard` | `fast`, `standard`, `thorough` |
| `--prefilter-top-k` | `600` | Prefilter feature count |
| `--screening-enabled` | `False` | Enable Tier-2 screening |
| `--screening-method` | `none` | `stir`, `evalue`, `none` |
| `--eval-models-enabled` | `False` | Multi-classifier evaluation proxy |
| `--multiomics-adapter` | `none` | Benchmark-time shortcut for synthetic block construction (`split_halves`) |
| `--enable-classifier-conformal` | `False` | Turn on classifier-side conformal diagnostics |
| `--classifier-conformal-method` | `split` | `split`, `aps`, `raps`, or `cross` |
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

Validation-catalog benchmark datasets are not allowed to silently fall back to synthetic proxies. When those datasets are selected, the benchmark CLI requires the authoritative HuggingFace bundle via `TABNETICS_HF_ORG` or `TABNETICS_HF_REPO_ID`. That bundle is an operational mirror of the public upstream datasets rather than a separate private corpus.

---

## Validation campaigns

Tabnetics ships three packaged validation surfaces:

- `tabnetics-validation-plan` / `python -m tabnetics.validation.generate_plan`
- `tabnetics-validation-shard` / `python -m tabnetics.validation.core.shard_runner`
- `tabnetics-validation-suite` / `python -m tabnetics.validation.suite`

Typical workflow:

```bash
export TABNETICS_HF_ORG=klokedm
export TABNETICS_HF_REPO_ID=klokedm/tabnetics-validation

# 1. Build a sharded campaign plan.
tabnetics-validation-plan --plan-kind validation17 --num-pods 4

# 2. Run a small local slice with the unified suite.
tabnetics-validation-suite \
    --dataset-sets fs_easy \
    --seeds 11 23 37 \
    --dataset-integrity-policy error \
    --output-dir run_artifacts/validation/unified_suite

# 3. Execute one shard from a generated plan.
tabnetics-validation-shard \
    --shard-id 1 \
    --plan pod_validation/plan_4_val17.json \
    --shards pod_validation/shards_4_val17.json
```

Use the suite for smaller slices or component ablations. Use the plan + shard flow when you want a reproducible multi-job campaign across a larger benchmark catalog.

---

## Datasets

Tabnetics ships with a registry of 70+ HDLSS benchmark datasets. Operationally, many validation-catalog runs are loaded through the HuggingFace bundle, but the underlying data come from public upstream sources such as OpenML, GEO, CuMiDa, UCSC Xena, public HuggingFace datasets, and Orange / University of Ljubljana (`biolab.si`) mirrors where applicable.

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

## Reproducibility and data source policy

The public Pages site reflects the packaged implementation, not earlier draft workflows. The operational rules to keep in mind are:

- `df_stage_position="after_fs"` is the current default in the packaged pipeline and benchmark CLI.
- Validation-catalog and evidence-bearing benchmark runs treat the HuggingFace bundle as the authoritative reproducibility mirror of the public upstream sources.
- The benchmark CLI and validation suite default to `--no-synthetic-fallback`; if you explicitly re-enable fallback for non-evidence workflows, do so knowingly and record the source policy in your artifacts.
- `dataset_integrity_policy="error"` is the default so mislabeled or collapsed class structures fail fast instead of silently contaminating results.
- The built-in `multiomics_adapter="split_halves"` is intended for benchmark-time stress tests; use explicit omics blocks with `MultiBlockPLSDA` or `MINTIntegrator` on real data.

If you are preparing a public benchmark or a paper-facing validation run, keep the HF bundle, no-synthetic-fallback, and integrity-error settings unchanged.

---

## Uncertainty and conformal outputs

Classifier-side conformal prediction is available as an opt-in diagnostic layer:

```bash
tabnetics-benchmark \
    --datasets leukemia_golub \
    --enable-classifier-conformal \
    --classifier-conformal-method aps \
    --classifier-conformal-output-sets
```

Interpret those outputs as uncertainty diagnostics:

- coverage and mean prediction-set size
- singleton rate / compactness of the prediction sets
- optional per-sample prediction sets when artifact size is acceptable

They are **not** expected to improve balanced accuracy. This follows the MAPIE interpretation in [Taquet et al. 2022](https://arxiv.org/abs/2207.12274): conformal wrappers calibrate prediction sets around a fixed base classifier. For singleton-oriented efficiency and reject-option interpretations, see [Wang, Sun & Dobriban 2025](https://doi.org/10.48550/arXiv.2509.24095) and [Hallberg Szabadváry et al. 2025](https://doi.org/10.1016/j.mlwa.2025.100664).

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

## TabArena pipeline profiles

Pipeline-level profiles for the [TabArena](https://arxiv.org/abs/2506.08828) general tabular benchmark.
Use with `--profile <name>` via `python -m experiments.benchmarking.tabarena_benchmark`.

| Profile | Classifier pool | Key settings | Best for |
|---|---|---|---|
| `hdlss` | LR, SVM-RBF, RF, KNN, elastic-net, NB | CDF transform on, HDLSS screening/folding active | HDLSS baseline (p >> n) |
| `general` | LR, SVM-RBF, RF, KNN, elastic-net, NB, XGBoost, LightGBM | CDF off, FLAML-backed MNPO hybrid selection, 10-candidate cap | Moderate general tabular |
| `general_full` (default) | 16 classifiers (full surface excl. TabPFN) | CDF off, FLAML-backed MNPO, no runtime cap, tritrust oracle k=3, ensemble, post-Val-17 FS defaults | Broad general tabular |
| `general_tabular` | 12 tree-weighted classifiers (RF, ExtraTrees, XGBoost, LightGBM, CatBoost, LR, elastic-net, SVM-linear, SVM-RBF, KNN, NB, copula-DA) | CDF off, legacy selection with FLAML tuning, no screening/folding, adaptive FS fraction by N/p ratio, no HDLSS machinery | N >> p datasets (many samples, few features) |

The `general_tabular` profile adapts feature selection by the sample-to-feature ratio:
- **N/p > 100 or p ≤ 20**: `fs_fraction=1.0` (keep all features through FS)
- **N/p > 10 or p ≤ 50**: `fs_fraction=0.90`
- **Otherwise**: `fs_fraction=0.50`

Prefilter is only activated when `p > 200` (with BH correction disabled per Val-18 P03 evidence). FS still uses the post-Val-17 Banzhaf-weighted defaults, while classifier selection now stays on the legacy FLAML path because the strongest MNPO-collapse evidence so far is HDLSS/small-sample specific rather than general-tabular. Classifier-side conformal uses APS (per Val-18 C07 evidence). Screening and folding are always off (these are HDLSS-specific). HDLSS-oriented classifiers (GPC, PLS-DA, NSC, vote\_ensemble, shrinkage\_LDA) are excluded in favour of tree ensembles plus `copula_da` as an extra generative family.

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

For the full pipeline, `multiomics_adapter="split_halves"` is the built-in benchmark-style shortcut: it derives two blocks from the first and second half of the feature matrix. For real CSVs with named omics fields, prefer the explicit field-based pattern in [Your own CSV files](#your-own-csv-files) and build the blocks yourself with `MultiBlockPLSDA` or `MINTIntegrator`.

For the broader modeling context, see [Cai et al. 2022](https://doi.org/10.1016/j.isci.2022.103798), which reviews why multi-omics integration remains a meaningful lever for cancer classification even when small-sample settings make the engineering path harder.

See `tabnetics.multiomics` for full API.

---

## License

Apache 2.0 — see [LICENSE](https://github.com/klokedm/tabnetics-public/blob/main/LICENSE).

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

# ClassifierOracle: Deep Meta-Learning Oracle for Classifier Selection

**Status:** PROPOSAL  
**Date:** 2026-03-09  
**Module:** `core/src/tabnetics/experimental/classifier_oracle.py`

---

## 1. Motivation & Core Idea

The final stage of the tabnetics pipeline selects a classifier (LogisticRegression, SVM, GradientBoosting, TabPFN, etc.) and its hyperparameters to train on the distribution-transformed, feature-selected data. Currently this is handled by:

- Fixed default classifier (LogisticRegression with L2 penalty)
- Optional exhaustive grid search over classifier families
- MNPO-style aggregation across classifier candidates (not yet production)

**ClassifierOracle proposes:** train a deep neural network to predict which classifier + hyperparameter configuration will perform best on a given dataset, using meta-knowledge distilled from exhaustive evaluation on large datasets with the same subsampling paradigm as FSOracle and DFOracle.

### The Unified Oracle Vision

ClassifierOracle completes the meta-learned oracle trilogy:

```
New Small Dataset (n=100-200)
    │
    ├── FSOracle → optimal feature subset
    ├── DFOracle → optimal distribution per feature
    └── ClassifierOracle → optimal classifier + hyperparameters
```

Together, the three oracles transform the tabnetics pipeline from a computationally expensive search (run all methods, aggregate via MNPO) into an instant prediction (~100ms for all three).

### Why Classifier Selection Matters for Small Samples

The "Squeezing Lemons" study (Knauer & Rodner, 2024, arXiv:2405.07662) demonstrated that on 44 datasets with n≤500, L2-regularized logistic regression matches AutoML systems. But this is an *average* result — on specific dataset types (highly nonlinear boundaries, many classes, imbalanced), different classifiers dominate. ClassifierOracle learns *when* each classifier excels.

---

## 2. Related Work & Novelty

### Directly Related

| Reference | Key Insight | How ClassifierOracle Differs |
|-----------|-------------|------------------------------|
| **Nápoles et al.** (2022, arXiv:2210.14687) | Meta-learning for joint algorithm selection + hyperparameter tuning; 62 meta-features; CNN on tabular meta-data; 91% accuracy on synthetic, 87% on real | ClassifierOracle uses *subsample-based* episodes from large real datasets (not synthetic meta-datasets); conditions on FSOracle/DFOracle outputs for pipeline-aware prediction |
| **TabPFN** (Hollmann et al., 2022, arXiv:2207.01848) | Implicitly selects its own "algorithm" via in-context learning on training data | ClassifierOracle selects among *existing* sklearn classifiers, complementing rather than replacing TabPFN |
| **HyperFast** (Bonet et al., 2024, arXiv:2402.14335) | Hypernetwork generates classifier weights directly from training data | ClassifierOracle selects *which* classifier to use, while HyperFast generates a single neural classifier; these are complementary approaches |
| **AutoGluon/Auto-sklearn** | AutoML via Bayesian optimization over classifier+hyperparam space | ClassifierOracle replaces expensive online search with instant prediction; can serve as warm-start for AutoML |
| **Dataset2Vec** (Jomaa et al., 2019, arXiv:1905.11063) | Dataset meta-features for hyperparameter transfer across datasets | ClassifierOracle extends to predict the full classifier configuration, not just hyperparameters |
| **Beel et al.** (2020, arXiv:2006.12328) | Siamese meta-learning for algorithm selection using "Algorithm-Performance Personas" | ClassifierOracle uses a direct prediction approach rather than Siamese similarity; the "persona" concept informs our dataset embedding |
| **Kostovska et al.** (2023, arXiv:2310.10685) | Portfolio selection for algorithm selection (PS-AAS); SHAP-based vs performance-based meta-representations | ClassifierOracle adopts the insight that smaller, focused portfolios outperform larger ones; limits candidate classifier space |
| **WPFS** (Margeloiu et al., 2022, arXiv:2211.15616) | Weight Predictor Network for small-sample data; reduces parameters via auxiliary networks | ClassifierOracle's architecture draws on the weight-prediction paradigm for parameter-efficient design |
| **CARTE** (Kim et al., 2024, arXiv:2402.16785) | Context-aware tabular representations via graph attention networks | ClassifierOracle's dataset encoder could use CARTE-style entity embeddings |
| **Squeezing Lemons** (Knauer & Rodner, 2024, arXiv:2405.07662) | LogReg matches AutoML on n≤500 tabular data | Sets the baseline to beat; ClassifierOracle must add value beyond default LogReg |
| **Ranking Architectures** (Dubatovka et al., 2019, arXiv:1911.11481) | Pairwise ranking loss for neural architecture selection conditioned on task meta-features | ClassifierOracle adopts ranking loss for classifier ordering rather than regression on accuracy |

### Novel Contributions

1. **Pipeline-Aware Classifier Selection:** ClassifierOracle conditions on the outputs of FSOracle (selected features) and DFOracle (distribution fits), making it the first classifier selector that accounts for upstream pipeline decisions. This captures the interaction: the best classifier depends on *which features were selected* and *how their distributions were transformed*.

2. **Subsample-to-Full Extrapolation for Algorithm Selection:** Prior meta-learning for algorithm selection uses entire datasets as meta-instances. ClassifierOracle uses *subsamples* of large datasets, creating orders of magnitude more training episodes while establishing ground truth from full-data evaluation.

3. **Hyperparameter Prediction via Mixture Density Network:** Instead of a fixed set of hyperparameter configurations, ClassifierOracle predicts a *distribution* over hyperparameter space using a mixture density network, capturing both the best configuration and the uncertainty about it.

---

## 3. Architecture

### 3.1 Input Representation

The ClassifierOracle receives a rich dataset description combining raw statistics with upstream oracle outputs:

```
Dataset-Level Meta-Features (dim=32):
  # Basic statistics
  n_samples, n_features, n_selected_features (from FSOracle)
  n_classes, class_balance_entropy, minority_class_fraction
  
  # Geometric properties
  mean_feature_correlation, max_feature_correlation
  intrinsic_dimensionality_estimate
  first_4_PCA_explained_variance_ratios  (4 values)
  samples_per_feature_ratio (n/p)
  
  # Complexity measures
  fisher_discriminant_ratio (avg across features)
  volume_of_overlap_region
  maximum_individual_feature_efficiency
  collective_feature_efficiency
  
  # Class separability
  mean_silhouette_score
  class_overlap_score (fraction of samples in overlapping regions)
  
  # Upstream oracle context
  fs_oracle_confidence (mean confidence of FSOracle)
  df_oracle_confidence (mean confidence of DFOracle)
  mean_best_dist_bic (from DFOracle)
  fraction_normal_features (from DFOracle)
  fraction_heavy_tailed_features (from DFOracle)
  
  # Data quality
  noise_to_signal_ratio_estimate
  outlier_fraction (IQR-based)

Per-Feature Summary (dim=p_selected × 8):
  For each FSOracle-selected feature:
    - importance_score (from FSOracle)
    - best_distribution_family_id (from DFOracle, embedded)
    - distribution_confidence (from DFOracle)
    - transformed_skewness (after distribution transform)
    - transformed_kurtosis
    - mutual_info_with_target
    - correlation_with_top_feature
    - univariate_auc

Pairwise Interactions (dim=p_selected × p_selected):
  - Correlation matrix of selected features (post-transform)
```

### 3.2 Network Architecture: ClassifierOracle

```
Input: dataset_meta (32,), feature_summaries (p_sel, 8), corr_matrix (p_sel, p_sel)

1. Dataset Encoder:
   a. Meta-feature branch: Linear(32, d_model) → dataset_emb (d_model,)
   b. Feature set branch:
      - Linear(8, d_model) per feature → (p_sel, d_model)
      - SetTransformer / DeepSet pooling → feature_set_emb (d_model,)
   c. Correlation branch:
      - Flatten upper triangle → Linear(p_sel*(p_sel-1)//2, d_model//2)
      - Or: SVD → top-k singular values → Linear(k, d_model//2)
      → corr_emb (d_model//2,)
   d. Combine: dataset_repr = MLP([dataset_emb; feature_set_emb; corr_emb])
      → (d_model,)

2. Classifier Family Prediction Head:
   - Linear(d_model, n_classifier_families) → (n_families,)
   - Softmax → classifier_probs
   - n_families = 8 (see candidate list below)

3. Per-Family Hyperparameter Prediction Heads:
   For each classifier family k:
   - MixtureDensityNetwork(d_model, n_hyperparams_k, n_components=3)
     → means (3, n_hp), variances (3, n_hp), mixture_weights (3,)
   - This predicts a *distribution* over hyperparameter space

4. Confidence Head:
   - Linear(d_model, 1) → Sigmoid → confidence ∈ [0,1]

Output:
  - classifier_probs: (n_families,) — probability of each classifier family
  - hyperparams: Dict[family, MixtureDensityParams] — predicted HP distributions
  - confidence: float — oracle confidence (low → fall back to default LogReg)
```

**Parameter count estimate:** ~400K for d_model=128

### 3.3 Candidate Classifier Families

```python
CLASSIFIER_FAMILIES = {
    0: {
        "name": "logistic_regression",
        "hyperparams": ["C", "penalty", "solver"],  # C ∈ [1e-4, 1e4], penalty ∈ {l1, l2, elasticnet}
        "n_hyperparams": 3,
    },
    1: {
        "name": "linear_svm",
        "hyperparams": ["C", "kernel"],  # C ∈ [1e-3, 1e3], kernel ∈ {linear, rbf}
        "n_hyperparams": 2,
    },
    2: {
        "name": "gradient_boosting",
        "hyperparams": ["n_estimators", "max_depth", "learning_rate", "subsample"],
        "n_hyperparams": 4,
    },
    3: {
        "name": "random_forest",
        "hyperparams": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
        "n_hyperparams": 4,
    },
    4: {
        "name": "knn",
        "hyperparams": ["n_neighbors", "weights", "metric"],
        "n_hyperparams": 3,
    },
    5: {
        "name": "mlp",
        "hyperparams": ["hidden_layer_sizes", "learning_rate_init", "alpha", "activation"],
        "n_hyperparams": 4,
    },
    6: {
        "name": "tabpfn",
        "hyperparams": ["N_ensemble_configurations"],
        "n_hyperparams": 1,
    },
    7: {
        "name": "naive_bayes",
        "hyperparams": ["var_smoothing"],
        "n_hyperparams": 1,
    },
}
```

---

## 4. Training Procedure

### 4.1 Data Generation Pipeline

```python
# Pseudocode for ClassifierOracle training data generation
CLASSIFIER_CONFIGS = generate_classifier_grid()  # ~200 configs across 8 families

for dataset in large_datasets:
    X_full, y_full = load_dataset(dataset)

    # Step 1: Run full FS + DF pipeline on full data
    fs_result_full = run_feature_selection(X_full, y_full, method="exhaustive")
    df_result_full = run_distribution_fitting(X_full[:, fs_result_full.selected])

    # Step 2: Evaluate all classifier configs on full data (5-fold CV)
    gt_results = {}
    for config in CLASSIFIER_CONFIGS:
        scores = cross_val_score(
            config.classifier, X_full[:, fs_result_full.selected],
            y_full, cv=5, scoring="balanced_accuracy"
        )
        gt_results[config.id] = scores.mean()

    gt_best_family = get_best_family(gt_results)
    gt_best_config = get_best_config(gt_results)
    gt_ranking = rank_all_configs(gt_results)

    # Step 3: Generate subsample episodes
    for episode in range(N_EPISODES):
        n_sub = random.randint(100, 200)
        idx = random.sample(range(len(X_full)), n_sub)
        X_sub, y_sub = X_full[idx], y_full[idx]

        # Step 4: Run FSOracle + DFOracle on subsample (or compute meta-features)
        meta_features = compute_dataset_meta_features(X_sub, y_sub)
        fs_context = run_fs_oracle(X_sub, y_sub)  # or compute from subsample
        df_context = run_df_oracle(X_sub[:, fs_context.selected])

        # Step 5: Training example
        yield {
            "meta_features": meta_features,
            "feature_summaries": compute_feature_summaries(X_sub, y_sub, fs_context, df_context),
            "corr_matrix": np.corrcoef(X_sub[:, fs_context.selected].T),
            "gt_family": gt_best_family,
            "gt_config": gt_best_config,
            "gt_ranking": gt_ranking,  # full ranking for ranking loss
        }
```

### 4.2 Ground Truth Construction

For each large dataset, exhaustively evaluate ~200 classifier configurations:

```python
CLASSIFIER_GRID = {
    "logistic_regression": {
        "C": [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000],
        "penalty": ["l1", "l2", "elasticnet"],
        "solver": ["saga"],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.8, 1.0],
    },
    "random_forest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, 20, None],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "linear_svm": {
        "C": [0.01, 0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11, 21],
        "weights": ["uniform", "distance"],
    },
    "mlp": {
        "hidden_layer_sizes": [(64,), (128,), (64,32), (128,64)],
        "learning_rate_init": [1e-3, 1e-4],
        "alpha": [1e-4, 1e-3, 1e-2],
    },
    "tabpfn": {
        "N_ensemble_configurations": [4, 16, 32],
    },
    "naive_bayes": {
        "var_smoothing": [1e-9, 1e-7, 1e-5],
    },
}
```

**Ground truth labels:**
- **Family label:** The classifier family with the highest mean BA across its best configuration
- **Config label:** The specific (family, hyperparam) tuple with highest BA
- **Ranking:** All 200 configs ranked by BA for ranking loss
- **BA vector:** All 200 BA scores for regression loss (optional)

### 4.3 Loss Function

```python
loss = (
    λ_1 * family_ce(pred_family_probs, gt_family)           # Which family?
  + λ_2 * ranking_loss(pred_config_scores, gt_ranking)       # Rank configs correctly
  + λ_3 * hp_nll(pred_hp_mixture, gt_best_hp)               # Predict hyperparams
  + λ_4 * ba_regression(pred_ba, gt_ba)                     # Predict expected accuracy
)
```

- **Family CE:** Cross-entropy with label smoothing (0.1) for classifier family classification
- **Ranking loss:** ListMLE over all ~200 configurations — the oracle must rank good configs above bad ones
- **HP NLL:** Negative log-likelihood of ground-truth hyperparameters under the predicted mixture density — the oracle must place probability mass near the best hyperparameters
- **BA regression (optional):** MSE loss predicting expected balanced accuracy — useful for confidence calibration

Default: λ_1=1.0, λ_2=0.5, λ_3=0.3, λ_4=0.1

### 4.4 Training Configuration

```yaml
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 0.01
scheduler: CosineAnnealingWarmRestarts (T_0=50, T_mult=2)
batch_size: 32  # smaller batch due to expensive meta-feature computation
epochs: 200
early_stopping: patience=20 on validation family accuracy
gradient_clipping: max_norm=1.0

# Data augmentation
feature_subset_perturbation: randomly drop/add 1-2 features from FSOracle selection
meta_feature_noise: gaussian(σ=0.03)
subsample_size_variation: uniform(80, 250)
class_subsample_strategy: stratified (preserve class proportions)
```

---

## 5. Pipeline-Aware Training (Novel)

The key novelty of ClassifierOracle is **pipeline-awareness**: it conditions on upstream decisions.

### 5.1 Cascaded Oracle Training

```
Phase 1: Train FSOracle independently (as in FSOracle.md)
Phase 2: Train DFOracle independently (as in DFOracle.md)
Phase 3: Train ClassifierOracle conditioned on FSOracle + DFOracle outputs

During Phase 3, for each training episode:
  1. Run FSOracle on the subsample → feature selection
  2. Run DFOracle on selected features → distribution fits
  3. Compute ClassifierOracle inputs from FSOracle/DFOracle outputs
  4. Ground truth: best classifier on FULL data with FULL pipeline
```

### 5.2 End-to-End Fine-Tuning (Optional)

After initial cascaded training, fine-tune all three oracles jointly:

```
Loss_total = Loss_FS + Loss_DF + Loss_Classifier + λ_pipeline * Loss_pipeline

where Loss_pipeline = -log(BA(FSOracle_features, DFOracle_transforms, ClassifierOracle_model))
```

This requires differentiable classification (e.g., differentiable LogReg), which limits the classifier space but enables end-to-end gradient flow.

---

## 6. Integration with Tabnetics

### 6.1 Standalone Pipeline

```python
class OraclePipeline:
    """Complete meta-learned pipeline: FS → DF → Classifier in ~200ms."""

    def __init__(self):
        self.fs_oracle = FSOracle("fs_oracle_v1.pt")
        self.df_oracle = DFOracle("df_oracle_v1.pt")
        self.clf_oracle = ClassifierOracle("clf_oracle_v1.pt")

    def fit_predict(self, X_train, y_train, X_test):
        # Step 1: Feature selection (~50ms)
        importance = self.fs_oracle.predict_importance(X_train, y_train)
        selected = importance > threshold  # or top-k

        # Step 2: Distribution fitting + transform (~50ms)
        for j in np.where(selected)[0]:
            dist, params, _ = self.df_oracle.select_distribution(X_train[:, j])
            X_train[:, j] = transform_to_gaussian(X_train[:, j], dist, params)
            X_test[:, j] = transform_to_gaussian(X_test[:, j], dist, params)

        # Step 3: Classifier selection (~50ms)
        clf_family, clf_hp, conf = self.clf_oracle.predict(
            X_train[:, selected], y_train,
            fs_context=importance, df_context=...
        )

        # Step 4: Train and predict
        clf = instantiate_classifier(clf_family, clf_hp)
        clf.fit(X_train[:, selected], y_train)
        return clf.predict(X_test[:, selected])
```

### 6.2 MNPO Integration

```python
class ClassifierOraclePreference:
    """Neural oracle for classifier selection preferences in MNPO."""

    def __init__(self, model_path: str = "clf_oracle_v1.pt"):
        self.model = load_pretrained_classifier_oracle(model_path)

    def compute_preference_matrix(
        self, X: np.ndarray, y: np.ndarray,
        candidates: List[ClassifierConfig],
        fs_context: dict, df_context: dict
    ) -> np.ndarray:
        """Pairwise preferences over classifier candidates."""
        meta = compute_dataset_meta_features(X, y)
        probs, _, _ = self.model.predict(meta, fs_context, df_context)

        # Score each candidate config
        scores = []
        for config in candidates:
            family_score = probs[config.family_id]
            hp_score = self.model.score_hyperparams(config)
            scores.append(family_score * hp_score)

        return pairwise_pref_from_scalar(np.array(scores))
```

### 6.3 Hybrid Mode: Oracle-Guided Search

Instead of predicting the best classifier directly, use ClassifierOracle to guide a focused search:

```python
# Oracle predicts top-2 families + hyperparameter distributions
top_families, hp_distributions = clf_oracle.predict_top_k(X, y, k=2)

# Run focused search only on predicted families
for family in top_families:
    hp_samples = hp_distributions[family].sample(n=5)  # 5 configs per family
    for hp in hp_samples:
        score = cross_val_score(family, hp, X, y)

# Total: 10 configs evaluated instead of 200
```

---

## 7. Evaluation Protocol

### 7.1 Metrics

| Metric | Description |
|--------|-------------|
| **Family Accuracy** | Top-1 accuracy of classifier family prediction |
| **Top-3 Family Accuracy** | Ground truth family in top-3 predictions |
| **Config Rank Correlation** | Spearman ρ between predicted and true configuration ranking |
| **BA Regret** | BA(predicted_classifier) - BA(best_classifier) on held-out data |
| **BA vs Default LogReg** | How much ClassifierOracle improves over always using LogReg |
| **BA vs AutoGluon** | How ClassifierOracle compares to AutoML (adjusting for runtime) |
| **Runtime** | Oracle prediction time (target: <50ms per dataset) |
| **Calibration** | How well confidence scores correlate with actual prediction quality |

### 7.2 Baselines

1. **Always LogReg** (current tabnetics default)
2. **Always GradientBoosting** (strong default for tabular data)
3. **Random classifier selection** (lower bound)
4. **Full grid search** (200 configs, 5-fold CV — gold standard, expensive)
5. **Auto-sklearn** / **AutoGluon** (AutoML systems, time-limited to 60s)
6. **Meta-learning with hand-crafted meta-features** (Nápoles et al. approach)

---

## 8. Source Datasets

Same 15+ large datasets as FSOracle/DFOracle (see FSOracle.md §5). For ClassifierOracle, each dataset produces ~1000 episodes (subsample + full pipeline evaluation), yielding ~15K training episodes.

**Critical difference:** ClassifierOracle ground truth requires running ~200 classifier configs × 5-fold CV on each full dataset. This is computationally expensive (~2-4 hours per dataset on `lab01`). Plan to run on `lab01` + `lab02` in parallel.

---

## 9. Implementation Phases

### Phase 1: Ground Truth Generation (Week 1-3)
- [ ] Implement exhaustive classifier grid evaluation pipeline
- [ ] Run on all 15+ large datasets (lab01 + lab02)
- [ ] Cache all results (dataset, config) → BA scores
- [ ] Validate ground truth stability (re-run 3× with different seeds)
- [ ] Depends on: FSOracle Phase 1 (datasets), DFOracle Phase 1 (distributions)

### Phase 2: Meta-Feature Engineering (Week 2-3)
- [ ] Implement dataset meta-feature computation (32 features)
- [ ] Implement per-feature summary computation
- [ ] Implement pipeline-context features (from FSOracle/DFOracle)
- [ ] Validate meta-features are informative (logistic regression sanity check)

### Phase 3: Model Architecture (Week 3-4)
- [ ] Implement ClassifierOracle network in PyTorch
- [ ] Implement Mixture Density Network for hyperparameter heads
- [ ] Implement variable-size feature set handling
- [ ] Unit tests

### Phase 4: Training (Week 4-5)
- [ ] Cascaded training (using pretrained FSOracle + DFOracle)
- [ ] Cross-dataset validation
- [ ] Hyperparameter search for the oracle itself
- [ ] Save best checkpoints
- [ ] Depends on: FSOracle Phase 3, DFOracle Phase 4

### Phase 5: Integration & Evaluation (Week 5-6)
- [ ] Implement OraclePipeline (full FS→DF→Classifier)
- [ ] Run on tabnetics benchmark (67 datasets)
- [ ] Compare all modes vs baselines
- [ ] Ablation studies
- [ ] End-to-end fine-tuning experiment
- [ ] Document findings

---

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Classifier rankings don't transfer from large to small datasets | High | High | Key risk — mitigate by validating on HDLSS during training; use dataset-size-aware meta-features; include augmented small datasets in training |
| Ground truth generation is prohibitively expensive | Medium | High | Parallelize across lab01+lab02; use early stopping in CV; cache aggressively |
| Pipeline-aware training creates strong coupling between oracles | Medium | Medium | Allow ClassifierOracle to operate without FSOracle/DFOracle context (zero-masked inputs) |
| MixtureDensityNetwork training is unstable | Medium | Medium | Use LogSumExp stabilization; pre-train with fixed mixture weights; temperature scaling |
| TabPFN only works on small datasets — can't get ground truth from large data | Low | Medium | For TabPFN, evaluate on 1K-sample subsets of large data; or exclude TabPFN from large-data ground truth and include only in HDLSS validation |
| Too few training datasets (15) for good generalization | Medium | High | Augment with synthetic datasets; use aggressive regularization; consider transfer from OpenML-CC18 benchmark results |

---

## 11. Success Criteria

1. **Family Accuracy > 60%** on cross-dataset validation (above random=12.5% for 8 families)
2. **BA Regret < 2%** compared to exhaustive grid search
3. **ClassifierOracle > Always-LogReg** on ≥60% of benchmark datasets
4. **OraclePipeline (FS+DF+Clf) ≥ tabnetics MNPO** balanced accuracy on benchmark
5. **Full inference time < 200ms** for the complete oracle pipeline (FS+DF+Clf)
6. **Confidence calibration:** predictions with confidence > 0.8 should have <1% BA regret

---

## 12. Novel Ideas for Validation

### 12.1 Cross-Task Transfer Learning

Train on classification tasks, fine-tune for regression (MSE → ranked performance):
- Opens the door to oracle-based regression pipeline without full retraining

### 12.2 Active Learning for Oracle Improvement

When ClassifierOracle has low confidence:
1. Run the top-2 predicted classifiers (targeted search)
2. Use the real outcome to update the oracle (online fine-tuning)
3. Over many datasets, the oracle continuously improves

### 12.3 Interpretability via Attention Analysis

Analyze which meta-features the ClassifierOracle attends to when selecting each classifier family:
- If "samples_per_feature_ratio" drives LogReg selection → confirms bias-variance theory
- If "class_overlap_score" drives GBM selection → confirms nonlinearity detection
- Publishable analysis of "what makes each classifier the right choice"

---

## 13. References

1. Nápoles, G. et al. (2022). "Which is the best model for my data?" arXiv:2210.14687
2. Hollmann, N. et al. (2022). "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second." arXiv:2207.01848
3. Bonet, D. et al. (2024). "HyperFast: Instant Classification for Tabular Data." arXiv:2402.14335
4. Jomaa, H.S. et al. (2019). "Dataset2Vec: Learning Dataset Meta-Features." arXiv:1905.11063
5. Kim, M.J., Grinsztajn, L., Varoquaux, G. (2024). "CARTE: Pretraining and Transfer for Tabular Learning." arXiv:2402.16785
6. Knauer, R., Rodner, E. (2024). "Squeezing Lemons with Hammers." arXiv:2405.07662
7. Margeloiu, A. et al. (2022). "Weight Predictor Network with Feature Selection for Small Sample Tabular Biomedical Data." arXiv:2211.15616
8. Beel, J. et al. (2020). "Siamese Meta-Learning and Algorithm Selection with 'Algorithm-Performance Personas'." arXiv:2006.12328
9. Kostovska, A. et al. (2023). "PS-AAS: Portfolio Selection for Automated Algorithm Selection." arXiv:2310.10685
10. Dubatovka, A. et al. (2019). "Ranking architectures using meta-learning." arXiv:1911.11481
11. Rivolli, A. et al. (2018). "Characterizing classification datasets: a study of meta-features for meta-learning." arXiv:1808.10406
12. Franz, A. et al. (2025). "Universal Embeddings of Tabular Data." arXiv:2507.05904
13. tabnetics internal: `core/src/tabnetics/classification/` — existing classifier backends
14. tabnetics internal: `run_artifacts/MNPO_ORACLE_LITERATURE_REVIEW.md` — MNPO limitations analysis

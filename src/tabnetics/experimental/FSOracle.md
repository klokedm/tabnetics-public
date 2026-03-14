# FSOracle: Deep Meta-Learning Oracle for Feature Selection

**Status:** PROPOSAL  
**Date:** 2026-03-09  
**Module:** `core/src/tabnetics/experimental/fs_oracle.py`

---

## 1. Motivation & Core Idea

The existing MNPO feature selection pipeline aggregates ~12+ methods (stability, wrapper, embedded, filter, knockoff) via pairwise preference matrices and mirror-descent Nash equilibrium. The MNPO Oracle Literature Review (2026-02-27) identified a fundamental limitation: with HDLSS data (n<200), the estimation noise in the payoff matrix overwhelms the oracle signal, and no oracle redesign within the classical framework can fix this.

**FSOracle proposes a fundamentally different approach:** instead of running all methods and aggregating post-hoc, train a deep neural network to directly predict which features to select, using meta-knowledge distilled from large datasets.

### The Subsampling Trick

1. **Source large datasets** (≥100k records, 50+ features) from OpenML, UCI, Kaggle
2. **Establish ground truth** on the full dataset: run exhaustive feature selection with multiple methods, identify optimal feature subsets via held-out classification accuracy
3. **Generate training episodes** by randomly sampling small subsets (100–200 records) from the large dataset
4. **Train the oracle** to predict the ground-truth feature importance/selection from the small subsample's statistical signature

This creates a natural supervised learning problem: the network learns to extrapolate from small-sample statistics to full-dataset feature relevance.

---

## 2. Related Work & Novelty

### Directly Related

| Reference | Key Insight | How FSOracle Differs |
|-----------|-------------|---------------------|
| **Dataset2Vec** (Jomaa et al., 2019, arXiv:1905.11063) | Learns dataset meta-features via DeepSet on predictor/target pairs; predicts dataset similarity for hyperparameter transfer | FSOracle uses a similar set-based encoding but predicts *feature-level* importance rather than dataset-level embeddings |
| **TabPFN** (Hollmann et al., 2022, arXiv:2207.01848) | Prior-Data Fitted Network performs in-context classification via transformer trained on synthetic causal datasets | FSOracle applies the PFN paradigm to *feature selection* rather than classification; uses real data subsamples rather than synthetic priors |
| **Scaling TabPFN** (Feuer et al., 2023, arXiv:2311.10609) | Studies sketching and feature selection for TabPFN; notes key differences vs conventional models | Confirms the need for feature-selection-aware PFNs; FSOracle addresses this directly |
| **HyperFast** (Bonet et al., 2024, arXiv:2402.14335) | Meta-trained hypernetwork generates task-specific classifiers in a single forward pass | FSOracle generates task-specific *feature selectors* rather than classifiers; shares the hypernetwork paradigm |
| **CARTE** (Kim et al., 2024, arXiv:2402.16785) | Context-aware graph representation for tabular data with open vocabulary via string embeddings | FSOracle could use CARTE-style entity embeddings for column-name-aware feature encoding |
| **Nápoles et al.** (2022, arXiv:2210.14687) | Meta-learning for model selection using 62 meta-features + synthetic data augmentation; CNN for tabular meta-classification | FSOracle extends this from model selection to feature selection; uses real-data subsampling instead of synthetic generation |
| **Rivolli et al.** (2018, arXiv:1808.10406) | Systematizes 62+ meta-features for dataset characterization; proposes MFE tool | FSOracle uses a superset of these meta-features as input representation |
| **WPFS** (Margeloiu et al., 2022, arXiv:2211.15616) | Weight Predictor Network with Feature Selection for small-sample biomedical data; reduces learnable parameters via auxiliary networks | Validates the principle of parameter-efficient architectures for HDLSS; FSOracle is complementary — it learns the selector rather than the classifier |
| **Beel et al.** (2020, arXiv:2006.12328) | Siamese meta-learning for algorithm selection using "Algorithm-Performance Personas" | FSOracle uses a similar idea of learning from "alike-performing" dataset episodes |
| **Squeezing Lemons** (Knauer & Rodner, 2024, arXiv:2405.07662) | L2-regularized logistic regression matches AutoML/TabPFN on small tabular datasets (n≤500) | Establishes a strong baseline to beat; FSOracle must outperform simple baselines on small samples |

### Novel Contributions

1. **Subsample-to-Full Extrapolation for Feature Selection:** No prior work trains a neural network to predict optimal feature subsets from subsampled statistics of large datasets. This is distinct from both meta-learning (which transfers across datasets) and PFN (which uses synthetic priors).

2. **Per-Feature Attention with Dataset Context:** The architecture processes each feature's statistical signature while attending to inter-feature correlations, producing per-feature importance scores rather than dataset-level embeddings.

3. **MNPO Integration via Learned Oracle:** FSOracle acts as a new oracle in the MNPO framework — its per-feature importance scores produce a pairwise preference matrix over feature subsets, which MNPO aggregates with classical oracles (performance, stability, complexity).

---

## 3. Architecture

### 3.1 Input Representation

For a dataset with `p` features and `n` samples (subsample), compute per-feature statistics:

```
Per-feature vector (dim=24):
  - Statistical moments: mean, std, skewness, excess_kurtosis, cv
  - Quantiles: q25, median, q75, iqr
  - Boolean indicators: is_positive, frac_negative, is_symmetric, has_heavy_tails
  - Distribution hints: exponential_cv_score, uniform_variance_score, lognormal_score
  - L-moments: l_cv, l_skew, l_kurtosis
  - Tail: hazard_slope
  - Shape: peak_to_average_ratio, ratio_mid50, dip_stat, hill_estimator

Dataset-level context vector (dim=16):
  - n_samples, n_features, n_classes, class_balance_entropy
  - mean_feature_correlation, max_feature_correlation
  - mean_mutual_info_with_target, max_mutual_info_with_target
  - first_4_PCA_explained_variance_ratios
  - intrinsic_dimensionality_estimate
  - noise_to_signal_ratio_estimate
  - feature_redundancy_score (mean pairwise |ρ|)

Pairwise feature interaction matrix (dim=p×p):
  - Pearson correlation matrix (upper triangle)
  - Mutual information matrix (optional, expensive)
```

**Total input shape:** `(p, 24)` per-feature + `(16,)` context + `(p, p)` correlation

### 3.2 Network Architecture: SetTransformer-based FSOracle

```
Input: per_feature_stats (p, 24), context (16,), correlation_matrix (p, p)

1. Feature Embedding Layer:
   - Linear(24, d_model) per feature → (p, d_model)
   - Add positional/index encoding (learned, not fixed — features have no natural order)

2. Context Injection:
   - Linear(16, d_model) → context_emb (d_model,)
   - Broadcast-add or FiLM conditioning: feature_embs = feature_embs * γ(context) + β(context)

3. Correlation-Aware Attention:
   - Use correlation matrix as attention bias in self-attention layers
   - attention_logits += α * correlation_matrix  (learned scalar α)

4. Transformer Encoder (L=4 layers, h=4 heads):
   - MultiHeadSelfAttention with correlation bias
   - LayerNorm + FFN (d_model → 4*d_model → d_model)
   - Dropout(0.1)

5. Feature Importance Head:
   - Linear(d_model, 1) → (p, 1) → squeeze → (p,)
   - Sigmoid activation → feature_importance ∈ [0, 1]^p

6. Feature Selection Head (optional, differentiable):
   - Gumbel-Softmax top-k selection
   - Or: threshold at learned τ

Output:
  - feature_importance: (p,) ∈ [0,1] — continuous importance scores
  - selected_features: binary mask (p,) — hard selection via threshold/top-k
```

**Parameter count estimate:** ~500K parameters for d_model=128, L=4, p≤500

### 3.3 Alternative: HyperNetwork Variant

Instead of directly outputting importance scores, generate the weights of a small feature-selector network:

```
Meta-input: dataset statistics → HyperNetwork → weights of TaskNet
TaskNet: X_subsample → feature_importance
```

This is closer to HyperFast's approach and may generalize better to variable feature counts.

---

## 4. Training Procedure

### 4.1 Data Generation Pipeline

```python
# Pseudocode for training data generation
for dataset in large_datasets:  # 50+ datasets, each >=100k rows
    X_full, y_full = load_dataset(dataset)

    # Step 1: Establish ground truth feature importance on full dataset
    gt_importance = compute_ground_truth_importance(X_full, y_full)
    # Methods: permutation importance (RF/GB), SHAP, stability selection,
    #          recursive elimination — averaged across methods

    for episode in range(N_EPISODES_PER_DATASET):  # 500-2000 episodes
        # Step 2: Random subsample
        n_sub = random.randint(100, 200)
        idx = random.sample(range(len(X_full)), n_sub)
        X_sub, y_sub = X_full[idx], y_full[idx]

        # Step 3: Compute input features from subsample
        per_feature_stats = compute_per_feature_stats(X_sub)
        context = compute_dataset_context(X_sub, y_sub)
        corr_matrix = np.corrcoef(X_sub.T)

        # Step 4: Create training example
        yield (per_feature_stats, context, corr_matrix), gt_importance
```

### 4.2 Ground Truth Construction

The ground truth feature importance vector is computed on the FULL dataset using an ensemble of methods:

1. **Permutation importance** (Random Forest, n_repeats=10) — captures nonlinear relevance
2. **Gradient Boosting feature importance** (gain-based) — captures split utility
3. **SHAP values** (TreeSHAP on XGBoost) — captures marginal contribution
4. **Stability selection** (subsampling Lasso, 500 iterations) — captures selection stability
5. **Mutual information** (k-NN estimator) — captures any statistical dependence

The ensemble importance is the rank-averaged (Borda count) importance across methods, normalized to [0, 1].

**Why rank-averaging?** Different methods have different scales (SHAP vs MI vs permutation). Rank averaging produces a robust, scale-free consensus.

### 4.3 Loss Function

Multi-objective loss combining:

```python
loss = (
    λ_1 * ranking_loss(pred_importance, gt_importance)    # ListMLE or ListNet
  + λ_2 * bce_loss(pred_selection, gt_top_k_mask)         # Binary selection
  + λ_3 * correlation_loss(pred_importance, gt_importance) # Spearman ρ surrogate
)
```

- **Ranking loss (primary):** The oracle must rank features correctly, not predict exact importance values. Use ListMLE (Xia et al., 2008) or ApproxNDCG.
- **Selection loss:** Cross-entropy on the top-k binary mask (k = ground-truth optimal number of features, determined by validation accuracy plateau).
- **Correlation loss:** Differentiable Spearman rank correlation to ensure monotonic relationship.

Default: λ_1=1.0, λ_2=0.5, λ_3=0.3

### 4.4 Training Configuration

```yaml
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 0.01
scheduler: CosineAnnealingWarmRestarts (T_0=50, T_mult=2)
batch_size: 64  # 64 episodes per batch
epochs: 200
early_stopping: patience=20 on validation ranking loss
gradient_clipping: max_norm=1.0

# Data augmentation
feature_permutation: true  # randomly permute feature order each episode
noise_injection: gaussian(σ=0.05) on per-feature stats
subsample_size_variation: uniform(80, 250)  # vary subsample size
```

### 4.5 Validation Strategy

- **In-distribution:** Hold out 20% of episodes from each large dataset
- **Cross-dataset:** Hold out 5 complete datasets (never seen subsamples during training)
- **Target-regime:** Evaluate on the existing tabnetics HDLSS benchmark (72–253 samples)

The cross-dataset evaluation is critical: the oracle must generalize to unseen dataset structures, not just unseen subsamples of known datasets.

---

## 5. Source Datasets for Training

Large classification datasets from OpenML for oracle training:

| Dataset | OpenML ID | Samples | Features | Classes | Domain |
|---------|-----------|---------|----------|---------|--------|
| covertype | 1596 | 581,012 | 54 | 7 | Forest cover type |
| higgs | 23512 | 98,050 | 28 | 2 | Physics |
| jannis | 45019 | 83,733 | 54 | 4 | Tabular |
| helena | 41169 | 65,196 | 27 | 100 | Tabular |
| dionis | 41167 | 416,188 | 60 | 355 | Tabular |
| MiniBooNE | 41150 | 130,064 | 50 | 2 | Particle physics |
| CIFAR-10 (tabular features) | — | 60,000 | 3072→PCA(100) | 10 | Vision tabular |
| adult | 1590 | 48,842 | 14 | 2 | Census |
| bank-marketing | 1461 | 45,211 | 16 | 2 | Marketing |
| connect-4 | 40668 | 67,557 | 42 | 3 | Game |
| fashion-mnist (tabular) | — | 70,000 | 784→PCA(100) | 10 | Vision tabular |
| KDD Cup 99 | 1113 | 494,021 | 41 | 23 | Network intrusion |
| poker-hand | 354 | 1,025,010 | 10 | 10 | Card game |
| shuttle | 40685 | 58,000 | 9 | 7 | NASA shuttle |

**Minimum requirement:** 15+ datasets, combined ≥50K training episodes.

**Feature augmentation:** For datasets with <50 features, augment with random noise features (known irrelevant) to train the oracle to reject noise.

---

## 6. MNPO Integration

FSOracle integrates into the existing MNPO framework as a new oracle:

```python
# In core/src/tabnetics/feature_selection/mnpo/oracles.py

class FSOraclePreference:
    """Neural oracle for feature selection preferences."""

    def __init__(self, model_path: str = "fs_oracle_v1.pt"):
        self.model = load_pretrained_fs_oracle(model_path)

    def compute_preference_matrix(
        self, X: np.ndarray, y: np.ndarray, candidates: List[np.ndarray]
    ) -> np.ndarray:
        """Compute pairwise preferences over candidate feature subsets.

        Args:
            X: Training data (n, p)
            y: Labels (n,)
            candidates: List of feature index arrays (from other FS methods)

        Returns:
            Preference matrix P where P[i,j] = Pr(candidate_i > candidate_j)
        """
        # Get oracle's feature importance scores
        importance = self.model.predict_importance(X, y)  # (p,)

        # Score each candidate by mean oracle importance of selected features
        scores = np.array([importance[c].mean() for c in candidates])

        # Convert to pairwise preferences (logistic model)
        return pairwise_pref_from_scalar(scores)
```

### Integration Modes

1. **Oracle mode (default):** FSOracle provides one pairwise preference matrix to MNPO alongside classical oracles (performance, stability, complexity). Low cost (~50ms forward pass), high complementarity.

2. **Standalone mode:** FSOracle directly selects features without MNPO. Use the importance scores and a threshold/top-k rule. For rapid prototyping or when MNPO overhead is unacceptable.

3. **Warm-start mode:** FSOracle's top-k features seed the candidate pool for classical methods (stability selection, RFECV), reducing the search space from p to k features. Then MNPO aggregates the refined candidates.

---

## 7. Evaluation Protocol

### 7.1 Metrics

| Metric | Description |
|--------|-------------|
| **Feature Ranking Correlation** | Spearman ρ between predicted and ground-truth feature importance |
| **Top-k Precision** | Fraction of oracle's top-k that overlap with ground-truth top-k |
| **Top-k Recall** | Fraction of ground-truth top-k recovered by oracle |
| **Downstream BA** | Balanced accuracy when classifying with oracle-selected features |
| **BA vs MNPO** | Paired comparison with current MNPO pipeline on tabnetics benchmark |
| **BA vs Simple** | Paired comparison with best-single-method baseline |
| **Runtime** | Oracle inference time (target: <100ms per dataset) |

### 7.2 Ablation Studies

1. **Subsample size:** How much does oracle quality degrade as n_sub decreases from 200 to 50?
2. **Number of training datasets:** Learning curve over 5, 10, 15, 20+ source datasets
3. **Architecture:** SetTransformer vs MLP vs GNN on correlation graph
4. **Input features:** Full 24-feature per-column stats vs minimal (mean, std, skew, kurt only)
5. **Ground truth method:** Ensemble vs single best method (permutation importance alone)
6. **Correlation matrix:** With vs without correlation-aware attention bias

---

## 8. Implementation Phases

### Phase 1: Data Pipeline & Ground Truth (Week 1-2)
- [ ] Implement `OracleTrainingDataGenerator` class
- [ ] Download and cache 15+ large OpenML datasets
- [ ] Implement ground-truth feature importance computation (5-method ensemble)
- [ ] Generate and store training episodes (HDF5/parquet format)
- [ ] Unit tests for data pipeline

### Phase 2: Model Architecture (Week 2-3)
- [ ] Implement `SetTransformerFSOracle` in PyTorch
- [ ] Implement variable-size feature handling (padding + masking)
- [ ] Implement ranking loss (ListMLE) and selection loss
- [ ] Unit tests for forward pass and loss computation

### Phase 3: Training & Validation (Week 3-4)
- [ ] Training loop with episode sampling
- [ ] Cross-dataset validation
- [ ] Hyperparameter search (d_model, n_layers, learning rate)
- [ ] Save best checkpoints

### Phase 4: MNPO Integration (Week 4-5)
- [ ] Implement `FSOraclePreference` wrapper
- [ ] Register FSOracle as optional oracle in MNPO portfolio
- [ ] Run on tabnetics benchmark (67 datasets)
- [ ] Compare FSOracle+MNPO vs MNPO vs Simple

### Phase 5: Ablation & Paper-Ready Results (Week 5-6)
- [ ] Full ablation studies
- [ ] Cross-domain transfer evaluation (train on genomics, test on non-genomics)
- [ ] Runtime benchmarking
- [ ] Document findings in the project archive

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Large datasets don't transfer to HDLSS regime | Medium | High | Validate on HDLSS during training; add HDLSS datasets to validation |
| Subsample statistics too noisy for reliable extrapolation | Medium | High | Average importance over multiple subsamples at inference; increase episode diversity |
| Overfitting to source dataset structures | Medium | Medium | Cross-dataset validation; diverse training corpus; dropout/regularization |
| Correlation matrix too large for many features (p>500) | Low | Medium | Use top-k principal correlations or random projection |
| Ground truth feature importance is method-dependent | Low | Medium | Rank-averaged ensemble is robust; validate with held-out accuracy |

---

## 10. Success Criteria

1. **Cross-dataset Top-k Precision > 0.6** for k=10 on held-out datasets
2. **Feature Ranking Spearman ρ > 0.5** on held-out datasets
3. **FSOracle+MNPO ≥ MNPO** balanced accuracy on tabnetics benchmark (non-inferior)
4. **FSOracle standalone ≥ Simple baseline** on ≥50% of benchmark datasets
5. **Inference time < 100ms** per dataset on CPU

---

## 11. References

1. Jomaa, H.S., Schmidt-Thieme, L., Grabocka, J. (2019). "Dataset2Vec: Learning Dataset Meta-Features." arXiv:1905.11063
2. Hollmann, N., Müller, S., Eggensperger, K., Hutter, F. (2022). "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second." arXiv:2207.01848
3. Feuer, B., Hegde, C., Cohen, N. (2023). "Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks." arXiv:2311.10609
4. Bonet, D., Mas Montserrat, D., Giró-i-Nieto, X., Ioannidis, A.G. (2024). "HyperFast: Instant Classification for Tabular Data." arXiv:2402.14335
5. Kim, M.J., Grinsztajn, L., Varoquaux, G. (2024). "CARTE: Pretraining and Transfer for Tabular Learning." arXiv:2402.16785
6. Nápoles, G. et al. (2022). "Which is the best model for my data?" arXiv:2210.14687
7. Rivolli, A. et al. (2018). "Characterizing classification datasets: a study of meta-features for meta-learning." arXiv:1808.10406
8. Margeloiu, A. et al. (2022). "Weight Predictor Network with Feature Selection for Small Sample Tabular Biomedical Data." arXiv:2211.15616
9. Knauer, R., Rodner, E. (2024). "Squeezing Lemons with Hammers: An Evaluation of AutoML and Tabular Deep Learning for Data-Scarce Classification Applications." arXiv:2405.07662
10. Beel, J. et al. (2020). "Siamese Meta-Learning and Algorithm Selection with 'Algorithm-Performance Personas'." arXiv:2006.12328
11. Franz, A. et al. (2025). "Universal Embeddings of Tabular Data." arXiv:2507.05904

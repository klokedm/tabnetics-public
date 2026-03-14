# DFOracle: Deep Meta-Learning Oracle for Distribution Fitting

**Status:** PROPOSAL  
**Date:** 2026-03-09  
**Module:** `core/src/tabnetics/experimental/df_oracle.py`

---

## 1. Motivation & Core Idea

Tabnetics currently selects the best-fit distribution for each feature using parametric bootstrap GOF (Cramér-von Mises/KS), BIC-based selection, or MNPO-aggregated oracle preferences across ~20+ scipy.stats families. The existing `s2_distribution_classifier.py` experiment already explored a discriminative neural network for distribution identification from 24 statistical features — but it operates on *single features in isolation* and was trained on synthetic data.

**DFOracle proposes a fundamentally stronger approach:** train a deep neural network to predict the optimal distribution family *and parameters* for each feature, using ground-truth labels derived from exhaustive fitting on large real-world datasets, with subsample-based training episodes.

### Why This Matters

Distribution fitting is the foundation of tabnetics' synthetic data generation pipeline. Choosing the wrong distribution family propagates errors through:
- Synthetic feature generation → wrong correlation structure
- Downstream classification → wrong decision boundaries
- Knockoff generation → invalid knockoffs → inflated false discovery rate

With HDLSS data (n<200), the sample size is insufficient for reliable goodness-of-fit testing. The bootstrap GOF p-values have low statistical power, and BIC tends to overfit. A meta-learned oracle that has seen the "shape" of real data across thousands of subsampling episodes can make better distribution choices than per-sample statistical tests.

### The Subsampling Trick (Same Paradigm as FSOracle)

1. **Source large datasets** (≥100k records) from OpenML
2. **Fit all candidate distributions** on the full dataset — the best-fit (by BIC + bootstrap GOF on full data) is the ground truth
3. **Generate training episodes** by subsampling 100–200 records
4. **Train the oracle** to predict the full-data best distribution from subsample statistics

---

## 2. Related Work & Novelty

### Directly Related

| Reference | Key Insight | How DFOracle Differs |
|-----------|-------------|---------------------|
| **s2_distribution_classifier.py** (tabnetics internal) | 16-class distribution classifier trained on synthetic data; 24 statistical features + per-distribution GOF scores | DFOracle trains on *real data subsamples* (not synthetic), uses inter-feature context, and predicts distribution parameters alongside family |
| **Dataset2Vec** (Jomaa et al., 2019, arXiv:1905.11063) | DeepSet architecture for dataset meta-features | DFOracle uses a similar set-based architecture but specializes for distribution identification per feature |
| **TabPFN** (Hollmann et al., 2022, arXiv:2207.01848) | In-context learning from synthetic causal model prior | DFOracle could adopt the PFN paradigm: encode raw data values (not just summary statistics) for richer distribution identification |
| **Drift-Resilient TabPFN** (Helli et al., 2024, arXiv:2411.10634) | Uses SCMs with temporal shifts; models changing distributions | DFOracle focuses on static distribution identification but the SCM prior design is relevant for training data generation |
| **Nápoles et al.** (2022, arXiv:2210.14687) | 62 meta-features for model selection; synthetic data augmentation for meta-training | DFOracle uses a subset of meta-features specialized for distribution shape characterization |
| **Multi-Branch Contrastive Network** (tabnetics, experiments/nn/) | Distribution identification via contrastive learning on branching architectures | DFOracle extends beyond contrastive pretraining to direct supervised prediction with subsample-based ground truth |

### Novel Contributions

1. **Real-Data Ground Truth for Distribution Selection:** Prior work (including tabnetics' s2) trains distribution classifiers on *synthetic* samples from known distributions. DFOracle trains on *real data* where the ground truth is the distribution that best fits the full dataset — this captures real-world distribution shapes that synthetic training misses (mixture distributions, contamination, truncation).

2. **Joint Distribution Family + Parameter Prediction:** Instead of just classifying the family and then fitting parameters separately, DFOracle predicts both in a single forward pass, enabling end-to-end optimization of the full distribution specification.

3. **Context-Aware Distribution Selection:** DFOracle conditions each feature's distribution prediction on the other features in the dataset (via cross-attention), capturing inter-feature dependencies that affect distribution choice (e.g., a feature that is a ratio of two other features should be fit differently).

---

## 3. Architecture

### 3.1 Input Representation

For each feature in the dataset (subsample of n=100–200 records):

```
Per-feature statistical summary (dim=24, same as existing DistributionFeatures):
  mean, std, skewness, excess_kurtosis, cv
  q25, median, q75, iqr
  is_positive, frac_negative, is_symmetric, has_heavy_tails
  exponential_cv_score, uniform_variance_score, lognormal_score
  l_cv, l_skew, l_kurtosis, hazard_slope
  peak_to_average_ratio, ratio_mid50, dip_stat
  hill_estimator, excess_mean_over_q

Raw value embedding (dim=d_raw, optional):
  - Sort the n values of the feature
  - Apply 1D CNN or set-encoder to the sorted values
  - This captures distributional shape beyond summary statistics
  - Handles multimodality, outliers, truncation

Per-distribution candidate scores (dim=16×6=96):
  For each of 16 candidate distributions:
    - ks_stat, ks_p, cvm_stat, cvm_p, aic, bic
  (Computed on the subsample — noisy but informative)
```

**Total per-feature input:** 24 + d_raw + 96 = ~184+ dimensions

### 3.2 Network Architecture: DFOracle Transformer

```
Input: features_stats (p, 24), raw_values (p, n_sorted), candidate_scores (p, 16, 6)

1. Per-Feature Encoder:
   a. Stats branch: Linear(24, d_model//2) → stats_emb (p, d_model//2)
   b. Raw branch: Conv1D(1, d_model//4, kernel=7) on sorted values
      → AdaptiveAvgPool → raw_emb (p, d_model//4)
   c. Candidate branch: Linear(96, d_model//4) → cand_emb (p, d_model//4)
   d. Concatenate: feature_emb = [stats_emb; raw_emb; cand_emb] (p, d_model)

2. Cross-Feature Transformer (L=3 layers, h=4 heads):
   - Self-attention across features — each feature attends to all others
   - Captures inter-feature distributional dependencies
   - LayerNorm + FFN (d_model → 4*d_model → d_model)
   - Dropout(0.1)

3. Distribution Classification Head:
   - Linear(d_model, n_distributions) → (p, 16)
   - Softmax → per-feature distribution probabilities

4. Parameter Prediction Head:
   - Linear(d_model, max_params × n_distributions) → (p, 16×4)
   - Reshape to (p, 16, 4)
   - Only the predicted distribution's parameters are used
   - Huber loss on parameters (robust to outliers)

5. Confidence Head (optional):
   - Linear(d_model, 1) → (p, 1) → Sigmoid
   - Estimates oracle confidence; low confidence → fall back to classical method

Output:
  - dist_probs: (p, 16) — probability of each distribution family per feature
  - dist_params: (p, 16, 4) — predicted parameters per distribution per feature
  - confidence: (p,) — oracle confidence per feature
```

**Parameter count estimate:** ~800K for d_model=128, L=3

### 3.3 Alternative: Mixture-of-Experts per Distribution Family

```
Shared encoder → (p, d_model)
Gate network: Linear(d_model, 16) → Softmax → gating weights
Expert 1 (norm):  Linear(d_model, 2) → (loc, scale)
Expert 2 (expon): Linear(d_model, 1) → (scale)
Expert 3 (gamma): Linear(d_model, 2) → (shape, scale)
...
Expert 16 (johnsonsb): Linear(d_model, 4) → (a, b, loc, scale)

Output: gate-weighted combination of expert predictions
```

This is more parameter-efficient and better handles the different parameter counts per distribution.

---

## 4. Training Procedure

### 4.1 Data Generation Pipeline

```python
# Pseudocode for DFOracle training data generation
CANDIDATE_DISTS = [
    "norm", "expon", "uniform", "weibull_min", "gamma", "lognorm", "beta",
    "t", "laplace", "pareto", "gumbel_l", "gumbel_r",
    "powerlaw", "triang", "johnsonsu", "johnsonsb"
]

for dataset in large_datasets:
    X_full, y_full = load_dataset(dataset)

    for feature_idx in range(X_full.shape[1]):
        col = X_full[:, feature_idx]

        # Step 1: Ground truth from full data
        gt_family, gt_params, gt_bic = exhaustive_distribution_fit(
            col, CANDIDATE_DISTS, method="mle", criterion="bic+bootstrap_gof"
        )

        for episode in range(N_EPISODES):
            # Step 2: Subsample
            n_sub = random.randint(100, 200)
            idx = random.sample(range(len(col)), n_sub)
            col_sub = col[idx]

            # Step 3: Compute features from subsample
            stats = compute_distribution_features(col_sub)
            sorted_values = np.sort(col_sub)
            candidate_scores = fit_all_candidates(col_sub, CANDIDATE_DISTS)

            # Step 4: Training example
            yield {
                "stats": stats,               # (24,)
                "sorted_values": sorted_values, # (n_sub,)
                "candidate_scores": candidate_scores,  # (16, 6)
                "gt_family": gt_family,        # int (0-15)
                "gt_params": gt_params,        # (4,) padded
                "gt_bic": gt_bic,              # float
            }
```

### 4.2 Ground Truth Construction

For each feature of each large dataset, fit all 16 candidate distributions on the full data:

1. **MLE fitting** with scipy.stats.fit() + fallback to moment matching
2. **BIC computation** (penalizes complexity)
3. **Bootstrap GOF** (1000 parametric bootstraps, Cramér-von Mises statistic)
4. **Ground truth = distribution with best (lowest) BIC among those not rejected by GOF** (p > 0.05)

**Handling ambiguity:** When multiple distributions are statistically indistinguishable (BIC within 2 units), label all of them as acceptable and use label smoothing in the loss function.

### 4.3 Loss Function

```python
loss = (
    λ_1 * cross_entropy(pred_dist_probs, gt_family_label, label_smoothing=0.1)
  + λ_2 * parameter_loss(pred_params[:, gt_family], gt_params)
  + λ_3 * bic_ranking_loss(pred_bic_implicit, gt_bic_ranking)
)

# Parameter loss: Huber loss (robust) on log-transformed scale parameters
def parameter_loss(pred, target):
    # Scale parameters are strictly positive → log transform
    mask_scale = is_scale_param(target)
    loss_scale = huber(log(pred[mask_scale] + ε), log(target[mask_scale] + ε))
    loss_other = huber(pred[~mask_scale], target[~mask_scale])
    return loss_scale + loss_other
```

Default: λ_1=1.0, λ_2=0.3, λ_3=0.2

### 4.4 Training Configuration

```yaml
optimizer: AdamW
learning_rate: 3e-4
weight_decay: 0.01
scheduler: OneCycleLR (max_lr=3e-4, pct_start=0.1)
batch_size: 128  # 128 feature-episodes per batch
epochs: 150
early_stopping: patience=15 on validation cross-entropy
gradient_clipping: max_norm=1.0

# Data augmentation
add_noise_features: true  # inject features with known distributions as extra supervision
value_jitter: uniform(0.95, 1.05) multiplicative noise on raw values
subsample_size_variation: uniform(80, 250)
```

---

## 5. Source Datasets

Same large datasets as FSOracle (see FSOracle.md §5), but each feature becomes an independent training example. With 15+ datasets × 50+ features × 1000 episodes = ~750K+ training examples.

**Additional synthetic augmentation:** Generate features from known distributions with controlled parameters to fill gaps (e.g., rarely-observed distributions like Johnsonsu, Pareto). This combines the strengths of the existing s2 approach (synthetic known-distribution training) with real-data subsampling.

---

## 6. Integration with Tabnetics Distribution Pipeline

### 6.1 As MNPO Oracle

```python
# In core/src/tabnetics/distribution/selector.py

class DFOracleSelector:
    """Neural oracle for distribution selection."""

    def __init__(self, model_path: str = "df_oracle_v1.pt"):
        self.model = load_pretrained_df_oracle(model_path)

    def select_distribution(
        self, data: np.ndarray, candidate_dists: List[str]
    ) -> Tuple[str, Dict[str, float], float]:
        """Select best distribution for a 1D data array.

        Returns:
            (family_name, parameters, confidence)
        """
        stats = compute_distribution_features(data)
        sorted_vals = np.sort(data)
        cand_scores = fit_all_candidates(data, candidate_dists)

        probs, params, conf = self.model.predict(stats, sorted_vals, cand_scores)
        best_idx = probs.argmax()
        return candidate_dists[best_idx], params[best_idx], conf

    def compute_preference_matrix(
        self, data: np.ndarray, candidates: List[str]
    ) -> np.ndarray:
        """Pairwise preferences for MNPO aggregation."""
        probs, _, _ = self.model.predict_for_feature(data)
        # Convert distribution probabilities to pairwise preferences
        scores = np.array([probs[DIST_NAME_TO_ID[c]] for c in candidates])
        return pairwise_pref_from_scalar(scores)
```

### 6.2 Integration Modes

1. **Oracle mode:** DFOracle provides a pairwise preference matrix to the existing MNPO distribution selection pipeline. Compatible with bootstrap GOF and BIC oracles.

2. **Standalone mode:** DFOracle directly selects the distribution family and estimates parameters in a single forward pass (~5ms per feature). No GOF testing needed.

3. **Hybrid mode:** DFOracle narrows the candidate set from 16 to top-3 distributions, then classical methods (bootstrap GOF, BIC) do final selection from the reduced set. This combines neural speed with statistical rigor.

4. **Confidence-gated mode:** Use DFOracle when confidence > 0.8; fall back to classical methods when confidence is low. This handles out-of-distribution features (not seen during training) gracefully.

---

## 7. Comparison with Existing s2 Distribution Classifier

| Aspect | s2_distribution_classifier | DFOracle |
|--------|---------------------------|----------|
| Training data | Synthetic (known distributions) | Real data subsamples + synthetic |
| Ground truth | Known distribution family | Best-fit on full dataset (BIC+GOF) |
| Inter-feature context | None (per-feature independent) | Cross-feature attention |
| Parameter prediction | No (classification only) | Yes (joint family + params) |
| Confidence estimation | No | Yes |
| Raw value input | No (summary stats only) | Yes (sorted values via 1D CNN) |
| Variable sample size | Fixed | Variable (80–250) |
| MNPO integration | No | Yes (pairwise preference output) |

DFOracle is a strict superset of s2's capabilities. The s2 model can serve as a baseline for comparison.

---

## 8. Evaluation Protocol

### 8.1 Metrics

| Metric | Description |
|--------|-------------|
| **Family Accuracy** | Top-1 accuracy of distribution family classification |
| **Top-3 Accuracy** | Ground truth in top-3 predicted families |
| **Parameter MSE** | Mean squared error of predicted vs true parameters (on standardized scale) |
| **BIC Regret** | BIC(predicted_dist) - BIC(best_dist) on full data |
| **Downstream KL** | KL divergence between predicted and true distribution |
| **Synthetic Data Quality** | Classification accuracy using synthetic data generated from oracle's distribution vs classical method |
| **Runtime** | Oracle inference time (target: <10ms per feature) |

### 8.2 Baselines

1. **BIC selection** (current default in tabnetics)
2. **Bootstrap GOF selection** (current gold standard)
3. **s2_distribution_classifier** (existing neural approach)
4. **MNPO distribution aggregation** (full pipeline)
5. **Always-normal** (trivial baseline)

---

## 9. Implementation Phases

### Phase 1: Ground Truth Generation (Week 1-2)
- [ ] Implement exhaustive distribution fitting pipeline for full datasets
- [ ] Handle edge cases: bounded features, discrete features, zero-inflated features
- [ ] Generate and cache ground-truth distribution labels for all features in training corpus
- [ ] Validate ground truth against manual inspection on 20+ features

### Phase 2: Training Data Pipeline (Week 2-3)
- [ ] Implement episode generator (subsample → features → label)
- [ ] Implement sorted-value preprocessing and padding
- [ ] Implement candidate-score computation (batch-optimized)
- [ ] Store training data in HDF5 with lazy loading

### Phase 3: Model Architecture (Week 3-4)
- [ ] Implement DFOracle Transformer in PyTorch
- [ ] Implement MoE variant for parameter prediction
- [ ] Implement confidence head with calibration
- [ ] Unit tests for all components

### Phase 4: Training (Week 4-5)
- [ ] Training loop with cross-dataset validation
- [ ] Compare Transformer vs MoE architectures
- [ ] Hyperparameter search
- [ ] Save best checkpoints

### Phase 5: Integration & Evaluation (Week 5-6)
- [ ] Implement MNPO integration wrapper
- [ ] Run on tabnetics benchmark datasets
- [ ] Compare all integration modes vs baselines
- [ ] Ablation studies
- [ ] Document findings

---

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Real distributions don't match any candidate family | Medium | Medium | Add "none/empirical" class; train to recognize when no parametric family fits well |
| Parameter prediction is ill-conditioned for some families | Medium | Medium | Use robust Huber loss; predict normalized/standardized parameters |
| Cross-feature attention adds noise for independent features | Low | Low | Ablate with/without cross-attention; attention masking for uncorrelated features |
| Training requires fitting 16 distributions per feature per episode | High | Medium | Pre-compute and cache all candidate fits; parallelize with joblib |
| Sorted-value input is sensitive to outliers | Medium | Low | Use robust sorting (winsorize at 1st/99th percentile) |

---

## 11. Success Criteria

1. **Top-1 Family Accuracy > 75%** on cross-dataset validation (human expert is ~80-85%)
2. **Top-3 Family Accuracy > 92%**
3. **BIC Regret < 5** on average (vs selecting best by BIC on full data)
4. **KL Divergence < 0.1** between oracle-predicted and true distribution (on full data)
5. **Runtime < 10ms per feature** on CPU
6. **Synthetic data quality:** Classification with oracle-generated synthetic data within 2% BA of classical method

---

## 12. References

1. Jomaa, H.S., Schmidt-Thieme, L., Grabocka, J. (2019). "Dataset2Vec: Learning Dataset Meta-Features." arXiv:1905.11063
2. Hollmann, N. et al. (2022). "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second." arXiv:2207.01848
3. Helli, K. et al. (2024). "Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data." arXiv:2411.10634
4. Nápoles, G. et al. (2022). "Which is the best model for my data?" arXiv:2210.14687
5. Rivolli, A. et al. (2018). "Characterizing classification datasets: a study of meta-features for meta-learning." arXiv:1808.10406
6. Knauer, R., Rodner, E. (2024). "Squeezing Lemons with Hammers." arXiv:2405.07662
7. Wang, W. et al. (2024). "A Survey on Self-Supervised Learning for Non-Sequential Tabular Data." arXiv:2402.01204
8. tabnetics internal: `experiments/nn/s2_distribution_classifier.py` — baseline neural distribution classifier
9. tabnetics internal: `experiments/unified_dist_selection_v6.py` — production distribution selection pipeline

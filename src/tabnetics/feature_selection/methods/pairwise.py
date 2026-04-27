"""Pairwise feature selection methods."""
import logging
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


def ktsp_selection(
    X, y, n_target_features,
    problem_type, random_state, prefilter_fn,
    ktsp_max_features, ktsp_max_pairs, ktsp_k_pairs,
):
    """
    k-TSP-inspired pairwise rank-rule selector.
    Pair search is run on a prefiltered feature pool to keep runtime tractable.
    """
    n_samples, n_features = np.asarray(X).shape
    if int(n_samples) < 2 or int(n_features) == 0 or problem_type != 'classification':
        return {}, {}

    classes = np.unique(y)
    # k-TSP is primarily a binary relative-expression reversal method.
    # For multiclass tasks, one-vs-rest extensions can be unstable in tiny n settings.
    if classes.size != 2:
        return {}, {}

    pool_idx = prefilter_fn(X, y, max_features=min(ktsp_max_features, n_features))
    if pool_idx.size < 2:
        return {}, {}

    pair_candidates = list(combinations(range(pool_idx.size), 2))
    if len(pair_candidates) > ktsp_max_pairs:
        rng = np.random.default_rng(random_state)
        sampled = rng.choice(len(pair_candidates), size=ktsp_max_pairs, replace=False)
        pair_candidates = [pair_candidates[i] for i in sampled]

    class_masks = {cls: (y == cls) for cls in classes}
    feature_pair_scores = []
    for a_local, b_local in pair_candidates:
        a = pool_idx[a_local]
        b = pool_idx[b_local]
        cmp_vec = X[:, a] > X[:, b]
        class_probs = []
        for cls in classes:
            mask = class_masks[cls]
            if np.sum(mask) == 0:
                class_probs.append(0.5)
            else:
                class_probs.append(float(np.mean(cmp_vec[mask])))

        best_gap = 0.0
        for i, j in combinations(range(len(class_probs)), 2):
            gap = abs(class_probs[i] - class_probs[j])
            if gap > best_gap:
                best_gap = gap
        feature_pair_scores.append((best_gap, int(a), int(b)))

    if not feature_pair_scores:
        return {}, {}

    feature_pair_scores.sort(key=lambda t: t[0], reverse=True)
    k_pairs = int(min(max(2, ktsp_k_pairs), len(feature_pair_scores)))
    top_pairs = feature_pair_scores[:k_pairs]

    feature_scores = np.zeros(n_features, dtype=float)
    for score, a, b in top_pairs:
        feature_scores[a] += float(score)
        feature_scores[b] += float(score)

    selected_indices = np.argsort(feature_scores)[::-1][:n_target_features]
    selected_indices = np.array(
        [int(i) for i in selected_indices if feature_scores[int(i)] > 0],
        dtype=int,
    )
    if selected_indices.size < min(n_target_features, pool_idx.size):
        fallback = [int(i) for i in pool_idx if int(i) not in set(selected_indices.tolist())]
        need = min(n_target_features, pool_idx.size) - selected_indices.size
        selected_indices = np.concatenate([selected_indices, np.array(fallback[:need], dtype=int)])

    results = {
        'selected_indices': selected_indices[:n_target_features],
        'scores': {int(idx): float(feature_scores[idx]) for idx in selected_indices[:n_target_features]},
        'top_pairs': [(int(a), int(b), float(score)) for score, a, b in top_pairs[:20]],
        'pool_size': int(pool_idx.size),
    }
    return results, {i: float(feature_scores[i]) for i in range(n_features)}

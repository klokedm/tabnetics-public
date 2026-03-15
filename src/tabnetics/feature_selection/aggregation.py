"""Aggregation utilities for feature selection ensemble voting."""
import numpy as np

from .registry import get_method_weights


def calculate_weighted_votes(method_results, n_features):
    """Calculate weighted votes for each feature across all methods with enhanced weights for Boruta and GA/SVM-RFE."""
    feature_votes = np.zeros(n_features)
    feature_details = {i: {} for i in range(n_features)}

    # Method weights sourced from the canonical METHOD_REGISTRY.
    method_weights = get_method_weights()

    for method_name, (results, all_scores) in method_results.items():
        if not results or 'selected_indices' not in results:
            continue

        selected_indices = results['selected_indices']
        scores = results.get('scores', {})

        # Get method weight multiplier
        method_weight = method_weights.get(method_name, 1.0)

        # Calculate votes based on ranking with method-specific weights
        for rank, idx in enumerate(selected_indices):
            if idx >= n_features:  # Safety check
                continue

            # Base weight: exponential decay based on rank
            base_weight = 0.9 ** rank

            # Apply method-specific multiplier
            final_weight = base_weight * method_weight

            feature_votes[idx] += final_weight

            # Store details
            feature_details[idx][method_name] = {
                'selected': True,
                'rank': rank + 1,
                'score': scores.get(idx, 0),
                'base_weight': base_weight,
                'method_weight': method_weight,
                'final_weight': final_weight
            }

        # Mark unselected features
        for idx in range(n_features):
            if idx not in selected_indices:
                feature_details[idx][method_name] = {
                    'selected': False,
                    'score': all_scores.get(idx, 0) if all_scores else 0,
                    'method_weight': method_weight
                }

    return feature_votes, feature_details


def safe_normalize_scores(all_scores, selected_indices, n_features):
    """Normalize any score container into a non-negative [0,1] vector."""
    score_vector = np.zeros(n_features, dtype=float)

    if isinstance(all_scores, dict):
        for idx, value in all_scores.items():
            idx_int = int(idx)
            if 0 <= idx_int < n_features:
                score_vector[idx_int] = float(value) if np.isfinite(value) else 0.0
    elif all_scores is not None:
        arr = np.asarray(all_scores, dtype=float).ravel()
        upto = min(n_features, arr.size)
        score_vector[:upto] = np.nan_to_num(arr[:upto], nan=0.0, posinf=0.0, neginf=0.0)

    score_vector = np.nan_to_num(score_vector, nan=0.0, posinf=0.0, neginf=0.0)
    min_score = float(np.min(score_vector)) if score_vector.size else 0.0
    if min_score < 0:
        score_vector = score_vector - min_score

    max_score = float(np.max(score_vector)) if score_vector.size else 0.0
    if max_score > 0:
        score_vector = score_vector / max_score

    selected = np.array(sorted(set(int(i) for i in selected_indices if 0 <= int(i) < n_features)), dtype=int)
    if selected.size > 0:
        rank_weights = np.linspace(1.0, 0.2, num=selected.size)
        for weight, idx in zip(rank_weights, selected):
            score_vector[idx] = max(score_vector[idx], float(weight))

    if np.max(score_vector) > 0:
        score_vector = score_vector / np.max(score_vector)

    return score_vector

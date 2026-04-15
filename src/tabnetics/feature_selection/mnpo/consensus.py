"""MNPO rank aggregation, wrapper refinement, and legacy consensus.

Standalone functions extracted from FeatureSelector (base.py).
"""

import numpy as np
import scipy.stats as sps
from typing import Dict, Any


# ---------------------------------------------------------------------------
# 1. wrapper_refine_subset_score
# ---------------------------------------------------------------------------
def wrapper_refine_subset_score(X, y, feature_subset,
                                get_inner_cv_splits_fn, fit_and_score_fold_fn):
    """Score one feature subset using the same inner-CV fit/score pathway as MNPO."""
    subset = np.asarray(feature_subset, dtype=int).ravel()
    subset = np.array(sorted(set(int(i) for i in subset if 0 <= int(i) < X.shape[1])), dtype=int)
    if subset.size == 0:
        return float("-inf")

    try:
        splits = get_inner_cv_splits_fn(X, y)
    except Exception as exc:
        return float("-inf")
    if not splits:
        return float("-inf")

    scores = []
    for train_idx, val_idx in splits:
        X_train = X[train_idx][:, subset]
        y_train = y[train_idx]
        X_val = X[val_idx][:, subset]
        y_val = y[val_idx]
        try:
            result = fit_and_score_fold_fn(X_train, y_train, X_val, y_val)
            fold_score = result[0] if isinstance(result, tuple) else result
        except Exception as exc:
            continue
        if np.isfinite(fold_score):
            scores.append(float(fold_score))

    if not scores:
        return float("-inf")
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 2. apply_wrapper_refinement
# ---------------------------------------------------------------------------
def apply_wrapper_refinement(
    X, y, vote_ranking, n_final_features,
    *,
    wrapper_refine_enabled,
    wrapper_refine_top_k,
    wrapper_refine_max_add,
    wrapper_refine_min_gain,
    get_inner_cv_splits_fn,
    fit_and_score_fold_fn,
):
    """
    Greedy wrapper refinement over top-ranked features.
    Uses inner-CV score improvements and falls back to vote ranking fill.
    """
    ranking_raw = np.asarray(vote_ranking, dtype=int).ravel()
    seen = set()
    ordered = []
    for idx in ranking_raw:
        idx_i = int(idx)
        if not (0 <= idx_i < X.shape[1]) or idx_i in seen:
            continue
        seen.add(idx_i)
        ordered.append(idx_i)
    ranking = np.asarray(ordered, dtype=int)
    target = int(max(1, min(int(n_final_features), ranking.size)))

    metadata: Dict[str, Any] = {
        "wrapper_refine_enabled": bool(wrapper_refine_enabled),
        "wrapper_refine_applied": False,
        "wrapper_refine_pool_size": 0,
        "wrapper_refine_wrapper_cap": 0,
        "wrapper_refine_min_gain": float(wrapper_refine_min_gain),
        "wrapper_refine_evaluations": 0,
        "wrapper_refine_score_initial": float("nan"),
        "wrapper_refine_score_final": float("nan"),
        "wrapper_refine_stop_reason": "disabled",
    }

    if ranking.size == 0:
        metadata["wrapper_refine_stop_reason"] = "empty_ranking"
        return np.array([], dtype=int), metadata
    if not wrapper_refine_enabled:
        return ranking[:target], metadata

    pool_size = int(min(max(1, wrapper_refine_top_k), ranking.size))
    pool = np.asarray(ranking[:pool_size], dtype=int)
    wrapper_cap = int(min(target, pool.size, max(1, wrapper_refine_max_add)))
    metadata["wrapper_refine_pool_size"] = int(pool_size)
    metadata["wrapper_refine_wrapper_cap"] = int(wrapper_cap)

    selected = [int(pool[0])]
    best_score = wrapper_refine_subset_score(
        X, y, selected, get_inner_cv_splits_fn, fit_and_score_fold_fn,
    )
    metadata["wrapper_refine_score_initial"] = float(best_score if np.isfinite(best_score) else float("nan"))
    evaluations = 1

    stop_reason = "wrapper_cap_reached"
    while len(selected) < wrapper_cap:
        best_feat = None
        best_candidate_score = float("-inf")
        for feat in pool:
            feat_i = int(feat)
            if feat_i in selected:
                continue
            cand = selected + [feat_i]
            score = wrapper_refine_subset_score(
                X, y, cand, get_inner_cv_splits_fn, fit_and_score_fold_fn,
            )
            evaluations += 1
            if score > best_candidate_score:
                best_candidate_score = score
                best_feat = feat_i

        if best_feat is None:
            stop_reason = "no_candidate"
            break

        if np.isfinite(best_score) and np.isfinite(best_candidate_score):
            if best_candidate_score <= (best_score + wrapper_refine_min_gain):
                stop_reason = "min_gain_not_met"
                break

        selected.append(int(best_feat))
        best_score = float(best_candidate_score)

    # Fill remainder from vote ranking to preserve target cardinality.
    selected_set = set(int(i) for i in selected)
    for feat in ranking:
        feat_i = int(feat)
        if feat_i in selected_set:
            continue
        selected.append(feat_i)
        selected_set.add(feat_i)
        if len(selected) >= target:
            break

    refined = np.asarray(selected[:target], dtype=int)
    baseline = np.asarray(ranking[:target], dtype=int)
    metadata["wrapper_refine_applied"] = bool(
        refined.size == baseline.size and np.any(refined != baseline)
    )
    metadata["wrapper_refine_evaluations"] = int(evaluations)
    metadata["wrapper_refine_score_final"] = float(best_score if np.isfinite(best_score) else float("nan"))
    metadata["wrapper_refine_stop_reason"] = str(stop_reason)

    return refined, metadata


# ---------------------------------------------------------------------------
# 3. build_rank_aggregation_candidate
# ---------------------------------------------------------------------------
def build_rank_aggregation_candidate(
    source_candidates, n_target, n_final_features, n_features,
    *,
    rank_aggregation_mode,
    normalize_fn,
):
    """
    Build a synthetic candidate from per-method rankings.
    Supports:
      - Borda: sum of reversed-rank points.
      - RRA: robust rank aggregation score via order-statistic Beta CDF p-values.
    """
    mode = str(rank_aggregation_mode).strip().lower()
    if mode == "none":
        return None
    if len(source_candidates) < 2 or n_features <= 0:
        return None

    rank_cap = int(min(n_features, max(8, int(n_target), int(n_final_features))))
    default_rank = float(rank_cap + 1)
    source_names = list(source_candidates.keys())
    n_sources = int(len(source_names))

    rank_matrix = np.full((n_sources, n_features), default_rank, dtype=float)
    for row, source_name in enumerate(source_names):
        score_vector = np.asarray(
            source_candidates[source_name].get("score_vector", np.zeros(n_features, dtype=float)),
            dtype=float,
        ).ravel()
        if score_vector.size != n_features:
            padded = np.zeros(n_features, dtype=float)
            upto = int(min(n_features, score_vector.size))
            if upto > 0:
                padded[:upto] = score_vector[:upto]
            score_vector = padded
        score_vector = np.nan_to_num(score_vector, nan=0.0, posinf=0.0, neginf=0.0)
        ranked = np.argsort(score_vector)[::-1][:rank_cap]
        if ranked.size > 0:
            rank_matrix[row, ranked] = np.arange(1, ranked.size + 1, dtype=float)

    method_result: Dict[str, Any] = {
        "method": f"rank_aggregate_{mode}",
        "rank_aggregation_mode": mode,
        "rank_aggregation_n_methods": n_sources,
        "rank_aggregation_rank_cap": int(rank_cap),
        "rank_aggregation_source_methods": list(source_names),
    }

    if mode == "borda":
        points = np.maximum(0.0, default_rank - rank_matrix)
        agg_scores = np.sum(points, axis=0)
    else:
        normalized_ranks = np.clip(rank_matrix / default_rank, 1e-12, 1.0)
        sorted_ranks = np.sort(normalized_ranks, axis=0)
        p_values = np.ones(n_features, dtype=float)
        for k in range(n_sources):
            cdf_vals = sps.beta.cdf(sorted_ranks[k], a=float(k + 1), b=float(n_sources - k))
            cdf_vals = np.nan_to_num(cdf_vals, nan=1.0, posinf=1.0, neginf=1.0)
            p_values = np.minimum(p_values, cdf_vals)
        agg_scores = -np.log10(np.clip(p_values, 1e-12, 1.0))
        method_result["rank_aggregation_p_values"] = np.asarray(p_values, dtype=float)

    agg_scores = np.asarray(normalize_fn(agg_scores), dtype=float).ravel()
    agg_scores = np.nan_to_num(agg_scores, nan=0.0, posinf=0.0, neginf=0.0)
    if agg_scores.size != n_features:
        padded = np.zeros(n_features, dtype=float)
        upto = int(min(n_features, agg_scores.size))
        if upto > 0:
            padded[:upto] = agg_scores[:upto]
        agg_scores = padded
    if np.max(agg_scores) <= 0:
        return None

    k_select = int(min(n_features, max(1, int(n_target), int(n_final_features))))
    selected = np.argsort(agg_scores)[::-1][:k_select]
    selected = np.asarray(selected, dtype=int)

    method_result["selected_indices"] = selected
    method_result["scores"] = {int(i): float(agg_scores[i]) for i in selected}
    candidate_name = f"rank_aggregate_{mode}"
    candidate_payload = {
        "selected_indices": selected,
        "score_vector": agg_scores,
        "runtime_sec": 0.0,
        "method_result": method_result,
    }
    return candidate_name, candidate_payload

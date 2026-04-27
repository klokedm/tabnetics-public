"""Wrapper feature selection methods (iterative pruning)."""

import numpy as np
from sklearn.covariance import LedoitWolf
from time import perf_counter


def iterative_pruning_cpss_overlay(
    X_pool, y_arr, target, base_selected, base_relevance, base_score, *,
    use_cpss_overlay,
    cpss_pairs,
    cpss_stability_threshold,
    cpss_min_stable_features,
    cpss_min_jaccard,
    cpss_max_score_drop,
    random_state,
    mi_scorer,
    f_scorer,
    normalize_fn,
    wrapper_score_fn,
):
    """
    A16: CPSS-style complementary-pairs stability overlay for bounded pruning.
    """
    n_samples, n_pool = X_pool.shape
    base_selected = np.asarray(base_selected, dtype=int).ravel()
    out = {
        "iterative_pruning_cpss_overlay_enabled": bool(use_cpss_overlay),
        "iterative_pruning_cpss_switch_applied": False,
        "iterative_pruning_cpss_switch_reason": "disabled",
        "iterative_pruning_cpss_pairs": int(cpss_pairs),
        "iterative_pruning_cpss_subruns": 0,
        "iterative_pruning_cpss_stability_threshold": float(cpss_stability_threshold),
        "iterative_pruning_cpss_min_stable_features": int(cpss_min_stable_features),
        "iterative_pruning_cpss_min_jaccard": float(cpss_min_jaccard),
        "iterative_pruning_cpss_max_score_drop": float(cpss_max_score_drop),
        "iterative_pruning_cpss_stable_feature_count": 0,
        "iterative_pruning_cpss_overlap_jaccard": float("nan"),
        "iterative_pruning_cpss_overlap_recall": float("nan"),
        "iterative_pruning_cpss_base_score": float(base_score) if np.isfinite(base_score) else float("nan"),
        "iterative_pruning_cpss_overlay_score": float("nan"),
        "iterative_pruning_cpss_overlay_score_evaluated": False,
        "iterative_pruning_cpss_selected_local": np.asarray(base_selected, dtype=int),
        "iterative_pruning_cpss_freq_q50": float("nan"),
        "iterative_pruning_cpss_freq_q90": float("nan"),
    }

    if not use_cpss_overlay:
        return out
    if n_pool <= 0:
        out["iterative_pruning_cpss_switch_reason"] = "empty_pool"
        return out
    if n_samples < 8:
        out["iterative_pruning_cpss_switch_reason"] = "insufficient_samples"
        return out
    if np.unique(y_arr).size < 2:
        out["iterative_pruning_cpss_switch_reason"] = "single_class"
        return out

    rng = np.random.default_rng(random_state + 1337)
    support_counts = np.zeros(n_pool, dtype=float)
    n_subruns = 0
    target = int(max(1, min(target, n_pool)))

    for pair_idx in range(int(cpss_pairs)):
        perm = rng.permutation(n_samples)
        half = int(perm.size // 2)
        if half < 4:
            continue
        subsets = (perm[:half], perm[half: 2 * half])
        for subset_idx in subsets:
            subset_idx = np.asarray(subset_idx, dtype=int)
            if subset_idx.size < 4:
                continue
            y_sub = np.asarray(y_arr[subset_idx])
            if np.unique(y_sub).size < 2:
                continue
            X_sub = np.asarray(X_pool[subset_idx], dtype=float)
            try:
                mi_scores = np.asarray(mi_scorer(X_sub, y_sub, random_state=random_state), dtype=float).ravel()
            except Exception as exc:
                mi_scores = np.zeros(n_pool, dtype=float)
            mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                f_scores, _ = f_scorer(X_sub, y_sub)
                f_scores = np.asarray(f_scores, dtype=float).ravel()
            except Exception as exc:
                f_scores = np.zeros(n_pool, dtype=float)
            f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
            local = 0.60 * normalize_fn(mi_scores) + 0.40 * normalize_fn(f_scores)
            local = np.asarray(normalize_fn(local), dtype=float).ravel()
            if local.size != n_pool:
                continue
            chosen = np.argsort(local)[::-1][:target]
            support_counts[np.asarray(chosen, dtype=int)] += 1.0
            n_subruns += 1

    out["iterative_pruning_cpss_subruns"] = int(n_subruns)
    if n_subruns <= 0:
        out["iterative_pruning_cpss_switch_reason"] = "no_valid_subruns"
        return out

    freq = support_counts / float(n_subruns)
    stable_idx = np.where(freq >= float(cpss_stability_threshold))[0]
    out["iterative_pruning_cpss_stable_feature_count"] = int(stable_idx.size)
    out["iterative_pruning_cpss_freq_q50"] = float(np.quantile(freq, 0.50))
    out["iterative_pruning_cpss_freq_q90"] = float(np.quantile(freq, 0.90))

    if stable_idx.size < int(cpss_min_stable_features):
        out["iterative_pruning_cpss_switch_reason"] = "insufficient_stable_support"
        return out

    base_norm = normalize_fn(base_relevance)
    stable_boost = np.zeros(n_pool, dtype=float)
    stable_boost[stable_idx] = 1.0
    combined = 0.55 * freq + 0.25 * base_norm + 0.20 * stable_boost
    combined = np.asarray(normalize_fn(combined), dtype=float).ravel()
    candidate = np.argsort(combined)[::-1][:target]
    candidate = np.asarray(candidate, dtype=int)
    if candidate.size == 0:
        out["iterative_pruning_cpss_switch_reason"] = "no_candidate"
        return out

    base_set = set(int(i) for i in np.asarray(base_selected, dtype=int).tolist())
    cand_set = set(int(i) for i in candidate.tolist())
    union = len(base_set.union(cand_set))
    inter = len(base_set.intersection(cand_set))
    jaccard = float(inter / union) if union > 0 else 1.0
    recall = float(inter / max(1, len(base_set)))
    out["iterative_pruning_cpss_overlap_jaccard"] = float(jaccard)
    out["iterative_pruning_cpss_overlap_recall"] = float(recall)

    if jaccard < float(cpss_min_jaccard):
        out["iterative_pruning_cpss_switch_reason"] = "overlap_below_threshold"
        return out

    overlay_score = wrapper_score_fn(X_pool, y_arr, candidate)
    out["iterative_pruning_cpss_overlay_score_evaluated"] = True
    if np.isfinite(overlay_score):
        out["iterative_pruning_cpss_overlay_score"] = float(overlay_score)
    else:
        out["iterative_pruning_cpss_switch_reason"] = "overlay_score_not_finite"
        return out

    if np.isfinite(base_score):
        if float(overlay_score) < float(base_score) - float(cpss_max_score_drop):
            out["iterative_pruning_cpss_switch_reason"] = "score_drop_exceeds_threshold"
            return out

    out["iterative_pruning_cpss_switch_applied"] = True
    out["iterative_pruning_cpss_switch_reason"] = "switch_applied"
    out["iterative_pruning_cpss_selected_local"] = np.asarray(candidate, dtype=int)
    return out


def iterative_redundancy_pruning_core(
    X, y, n_target_features, runtime_bounded=False, *,
    # Callables
    prefilter_fn,
    pareto_prefilter_fn,
    normalize_fn,
    wrapper_score_fn,
    cpss_overlay_fn,
    mi_scorer,
    f_scorer,
    # Data
    score_cache,
    problem_type,
    random_state,
    mrmr_max_features,
    # Pruning config
    iterative_pruning_pool_factor,
    iterative_pruning_redundancy_weight,
    iterative_pruning_max_rounds,
    iterative_pruning_min_improvement,
    iterative_pruning_max_cumulative_loss,
    # Pareto prefilter config
    iterative_pruning_class_pareto_prefilter_enabled,
    iterative_pruning_class_pareto_top_per_class,
    iterative_pruning_class_pareto_global_fraction,
    iterative_pruning_class_pareto_minority_boost,
    # Bounded config
    iterative_pruning_bounded_prefilter_cap,
    iterative_pruning_bounded_max_evaluations,
    iterative_pruning_bounded_max_runtime_seconds,
    iterative_pruning_bounded_candidate_fraction,
    iterative_pruning_bounded_min_candidates,
    iterative_pruning_bounded_enable_class_gating,
    iterative_pruning_bounded_multiclass_scale,
    iterative_pruning_bounded_imbalance_trigger,
    iterative_pruning_bounded_imbalance_scale,
    # CPSS config
    iterative_pruning_bounded_use_cpss_overlay,
    iterative_pruning_bounded_cpss_pairs,
    iterative_pruning_bounded_cpss_stability_threshold,
    iterative_pruning_bounded_cpss_min_stable_features,
    iterative_pruning_bounded_cpss_min_jaccard,
    iterative_pruning_bounded_cpss_max_score_drop,
):
    """
    ieGENES-style iterative wrapper pruning core.
    When ``runtime_bounded=True`` (A15), candidate evaluations are explicitly
    budgeted by prefilter cap, per-round candidate budget, max evaluations,
    max runtime, and class-aware candidate gating.
    """
    n_samples, n_features = X.shape
    if n_features == 0:
        return {}, {}

    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    n_classes = int(classes.size)
    min_class_count = int(np.min(counts)) if counts.size else 0
    imbalance_ratio = float(np.max(counts) / max(1, min_class_count)) if counts.size else 1.0

    pool_cap = int(
        min(
            n_features,
            max(mrmr_max_features, min(640, max(96, 8 * int(max(1, n_target_features))))),
        )
    )
    if runtime_bounded:
        pool_cap = int(min(pool_cap, int(iterative_pruning_bounded_prefilter_cap)))

    method_name = "iterative_redundancy_pruning_bounded" if runtime_bounded else "iterative_redundancy_pruning"

    if problem_type == "classification" and iterative_pruning_class_pareto_prefilter_enabled:
        pool_idx, pareto_meta = pareto_prefilter_fn(X, y_arr, max_features=pool_cap)
    else:
        pool_idx = prefilter_fn(X, y_arr, max_features=pool_cap)
        pareto_meta = {
            "iterative_pruning_pareto_prefilter_enabled": bool(iterative_pruning_class_pareto_prefilter_enabled),
            "iterative_pruning_pareto_prefilter_applied": False,
            "iterative_pruning_pareto_prefilter_reason": (
                "non_classification_problem" if problem_type != "classification" else "disabled"
            ),
            "iterative_pruning_pareto_prefilter_candidate_universe": int(pool_idx.size),
            "iterative_pruning_pareto_prefilter_front_size": 0,
            "iterative_pruning_pareto_prefilter_classes_used": 0,
            "iterative_pruning_pareto_prefilter_top_per_class": int(iterative_pruning_class_pareto_top_per_class),
            "iterative_pruning_pareto_prefilter_global_fraction": float(
                iterative_pruning_class_pareto_global_fraction
            ),
            "iterative_pruning_pareto_prefilter_minority_boost": float(
                iterative_pruning_class_pareto_minority_boost
            ),
            "iterative_pruning_pareto_prefilter_class_top_hits": {},
        }

    if pool_idx.size == 0:
        return {}, {}

    X_pool = np.asarray(X[:, pool_idx], dtype=float)
    n_pool = int(X_pool.shape[1])
    target = int(min(max(1, n_target_features), n_pool))

    if n_pool <= target:
        scores = np.zeros(n_features, dtype=float)
        scores[pool_idx] = 1.0
        selected = np.asarray(pool_idx[:target], dtype=int)
        results = {
            "selected_indices": selected,
            "scores": {int(idx): float(scores[int(idx)]) for idx in selected},
            "method": method_name,
            "iterative_pruning_initial_size": int(n_pool),
            "iterative_pruning_final_size": int(selected.size),
            "iterative_pruning_rounds": 0,
            "iterative_pruning_pool_size": int(n_pool),
            "iterative_pruning_runtime_bounded": bool(runtime_bounded),
            "iterative_pruning_stop_reason": "pool_not_larger_than_target",
            "iterative_pruning_evaluations": 0,
        }
        results.update(pareto_meta)
        return results, {i: float(scores[i]) for i in range(n_features)}

    # Use score cache if available (P1-1: eliminates redundant MI/F-test
    # computations — scores already computed on full X_uncorr in fit_transform).
    # pool_idx indexes into the same X_uncorr columns the cache was built on.
    if score_cache is not None:
        mi = np.asarray(score_cache.mi_scores[pool_idx], dtype=float).ravel()
        mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
        f_vals = np.asarray(score_cache.f_scores[0][pool_idx], dtype=float).ravel()
        f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        try:
            mi = np.asarray(mi_scorer(X_pool, y_arr, random_state=random_state), dtype=float).ravel()
        except Exception as exc:
            mi = np.zeros(n_pool, dtype=float)
        mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            f_vals, _ = f_scorer(X_pool, y_arr)
            f_vals = np.asarray(f_vals, dtype=float).ravel()
        except Exception as exc:
            f_vals = np.zeros(n_pool, dtype=float)
        f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    base_relevance = 0.60 * normalize_fn(mi) + 0.40 * normalize_fn(f_vals)
    base_relevance = np.asarray(base_relevance, dtype=float).ravel()
    base_relevance = np.nan_to_num(base_relevance, nan=0.0, posinf=0.0, neginf=0.0)

    init_size = int(
        min(
            n_pool,
            max(
                target + 2,
                int(round(iterative_pruning_pool_factor * float(target))),
            ),
        )
    )
    current = np.asarray(np.argsort(base_relevance)[::-1][:init_size], dtype=int)
    if current.size <= target:
        current = np.asarray(np.argsort(base_relevance)[::-1][:target], dtype=int)

    evaluations = 0
    removed_features = []
    removed_deltas = []
    cumulative_delta = 0.0
    rounds = 0
    stop_reason = "max_rounds_reached"
    start_t = perf_counter()
    per_round_candidate_budget = []

    max_evaluations = int(iterative_pruning_bounded_max_evaluations) if runtime_bounded else 0
    max_runtime_seconds = (
        float(iterative_pruning_bounded_max_runtime_seconds) if runtime_bounded else float("inf")
    )
    candidate_fraction = (
        float(iterative_pruning_bounded_candidate_fraction) if runtime_bounded else 1.0
    )
    min_candidates = int(iterative_pruning_bounded_min_candidates) if runtime_bounded else 1
    class_gating = bool(runtime_bounded and iterative_pruning_bounded_enable_class_gating)

    best_score = wrapper_score_fn(X_pool, y_arr, current)
    evaluations += 1
    if not np.isfinite(best_score):
        best_score = float("-inf")
        stop_reason = "initial_score_not_finite"

    corr_estimator = "corrcoef"
    corr_shrinkage = float("nan")
    if int(X_pool.shape[0]) < 2 or int(X_pool.shape[1]) < 2:
        corr_estimator = "degenerate"
        corr_pool = np.zeros((int(X_pool.shape[1]), int(X_pool.shape[1])), dtype=float)
    elif int(X_pool.shape[0]) < 2 * int(X_pool.shape[1]):
        corr_estimator = "ledoit_wolf"
        try:
            lw = LedoitWolf().fit(X_pool)
            cov = np.asarray(lw.covariance_, dtype=float)
            corr_shrinkage = float(getattr(lw, "shrinkage_", float("nan")))
            std = np.sqrt(np.maximum(0.0, np.diag(cov)))
            denom = np.outer(std, std)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-12)
            corr_pool = np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
        except Exception as exc:
            corr_estimator = "corrcoef_fallback"
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(X_pool, rowvar=False)
            corr_pool = np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    else:
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(X_pool, rowvar=False)
        corr_pool = np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    np.fill_diagonal(corr_pool, 0.0)

    while current.size > target and rounds < int(iterative_pruning_max_rounds):
        if runtime_bounded and (perf_counter() - start_t) >= max_runtime_seconds:
            stop_reason = "runtime_budget_exhausted"
            break
        if runtime_bounded and max_evaluations > 0 and evaluations >= max_evaluations:
            stop_reason = "evaluation_budget_exhausted"
            break

        rounds += 1
        local_corr = corr_pool[np.ix_(current, current)]
        mean_corr = np.mean(local_corr, axis=1) if local_corr.size else np.zeros(current.size, dtype=float)
        rel_local = base_relevance[current]
        priority = (
            float(iterative_pruning_redundancy_weight) * normalize_fn(mean_corr)
            - (1.0 - float(iterative_pruning_redundancy_weight)) * normalize_fn(rel_local)
        )
        candidate_order = np.asarray(np.argsort(priority)[::-1], dtype=int)

        if runtime_bounded:
            budget = int(max(min_candidates, int(np.ceil(candidate_fraction * float(candidate_order.size)))))
            if class_gating and n_classes > 2:
                budget = int(max(1, int(np.ceil(budget * float(iterative_pruning_bounded_multiclass_scale)))))
            if class_gating and imbalance_ratio >= float(iterative_pruning_bounded_imbalance_trigger):
                budget = int(max(1, int(np.ceil(budget * float(iterative_pruning_bounded_imbalance_scale)))))
            budget = int(min(candidate_order.size, max(1, budget)))
            candidate_order = candidate_order[:budget]
            per_round_candidate_budget.append(int(budget))
        else:
            per_round_candidate_budget.append(int(candidate_order.size))

        best_remove_idx = None
        best_remove_score = float("-inf")
        for local_idx in candidate_order.tolist():
            if runtime_bounded and (perf_counter() - start_t) >= max_runtime_seconds:
                stop_reason = "runtime_budget_exhausted"
                break
            if runtime_bounded and max_evaluations > 0 and evaluations >= max_evaluations:
                stop_reason = "evaluation_budget_exhausted"
                break

            local_idx = int(local_idx)
            trial = np.delete(current, local_idx)
            trial_score = wrapper_score_fn(X_pool, y_arr, trial)
            evaluations += 1
            if not np.isfinite(trial_score):
                continue
            if trial_score > best_remove_score:
                best_remove_score = float(trial_score)
                best_remove_idx = int(local_idx)

        if stop_reason in {"runtime_budget_exhausted", "evaluation_budget_exhausted"}:
            break

        if best_remove_idx is None:
            stop_reason = "no_valid_removal_candidate"
            break

        delta = float(best_remove_score - best_score) if np.isfinite(best_score) else float("inf")
        if np.isfinite(delta) and delta < float(iterative_pruning_min_improvement):
            stop_reason = "min_improvement_not_met"
            break
        delta_eff = float(delta) if np.isfinite(delta) else 0.0
        if (
            np.isfinite(delta_eff)
            and float(iterative_pruning_max_cumulative_loss) > 0.0
            and (cumulative_delta + delta_eff) < -float(iterative_pruning_max_cumulative_loss)
        ):
            stop_reason = "cumulative_loss_budget_exhausted"
            break

        removed_feature_local = int(current[best_remove_idx])
        removed_features.append(int(pool_idx[removed_feature_local]))
        removed_deltas.append(float(delta_eff))
        cumulative_delta += float(delta_eff)
        current = np.delete(current, best_remove_idx)
        best_score = float(best_remove_score)

        if current.size <= target:
            stop_reason = "target_size_reached"
            break

    if current.size > target:
        corr_local = corr_pool[np.ix_(current, current)]
        mean_corr = np.mean(corr_local, axis=1) if corr_local.size else np.zeros(current.size, dtype=float)
        rel_local = base_relevance[current]
        ranking = np.argsort(
            0.65 * normalize_fn(rel_local)
            + 0.35 * (1.0 - normalize_fn(mean_corr))
        )[::-1]
        current = current[np.asarray(ranking[:target], dtype=int)]

    selected_local = np.asarray(current[:target], dtype=int)
    cpss_meta = {
        "iterative_pruning_cpss_overlay_enabled": bool(runtime_bounded and iterative_pruning_bounded_use_cpss_overlay),
        "iterative_pruning_cpss_switch_applied": False,
        "iterative_pruning_cpss_switch_reason": "disabled",
        "iterative_pruning_cpss_pairs": int(iterative_pruning_bounded_cpss_pairs),
        "iterative_pruning_cpss_subruns": 0,
        "iterative_pruning_cpss_stability_threshold": float(iterative_pruning_bounded_cpss_stability_threshold),
        "iterative_pruning_cpss_min_stable_features": int(iterative_pruning_bounded_cpss_min_stable_features),
        "iterative_pruning_cpss_min_jaccard": float(iterative_pruning_bounded_cpss_min_jaccard),
        "iterative_pruning_cpss_max_score_drop": float(iterative_pruning_bounded_cpss_max_score_drop),
        "iterative_pruning_cpss_stable_feature_count": 0,
        "iterative_pruning_cpss_overlap_jaccard": float("nan"),
        "iterative_pruning_cpss_overlap_recall": float("nan"),
        "iterative_pruning_cpss_base_score": float(best_score) if np.isfinite(best_score) else float("nan"),
        "iterative_pruning_cpss_overlay_score": float("nan"),
        "iterative_pruning_cpss_overlay_score_evaluated": False,
        "iterative_pruning_cpss_freq_q50": float("nan"),
        "iterative_pruning_cpss_freq_q90": float("nan"),
    }
    if runtime_bounded and iterative_pruning_bounded_use_cpss_overlay:
        cpss_meta = cpss_overlay_fn(
            X_pool=X_pool,
            y_arr=y_arr,
            target=target,
            base_selected=selected_local,
            base_relevance=base_relevance,
            base_score=best_score,
        )
        if bool(cpss_meta.get("iterative_pruning_cpss_overlay_score_evaluated", False)):
            evaluations += 1
        if bool(cpss_meta.get("iterative_pruning_cpss_switch_applied", False)):
            selected_local = np.asarray(cpss_meta.get("iterative_pruning_cpss_selected_local", selected_local), dtype=int)
            overlay_score = float(cpss_meta.get("iterative_pruning_cpss_overlay_score", float("nan")))
            if np.isfinite(overlay_score):
                best_score = float(overlay_score)

    selected_local = np.array(
        sorted(set(int(i) for i in np.asarray(selected_local, dtype=int).tolist() if 0 <= int(i) < n_pool)),
        dtype=int,
    )
    if selected_local.size > target:
        selected_local = selected_local[:target]
    if selected_local.size < target:
        fill_order = np.argsort(base_relevance)[::-1]
        present = set(int(i) for i in selected_local.tolist())
        for idx in fill_order.tolist():
            idx_i = int(idx)
            if idx_i in present:
                continue
            selected_local = np.append(selected_local, idx_i)
            present.add(idx_i)
            if selected_local.size >= target:
                break
    selected_local = np.asarray(selected_local[:target], dtype=int)
    selected_global = np.asarray(pool_idx[selected_local], dtype=int)

    corr_local = corr_pool[np.ix_(selected_local, selected_local)]
    selected_mean_corr = np.mean(corr_local, axis=1) if corr_local.size else np.zeros(selected_local.size, dtype=float)
    selected_signal = 0.65 * normalize_fn(base_relevance[selected_local]) + 0.35 * (
        1.0 - normalize_fn(selected_mean_corr)
    )
    if bool(cpss_meta.get("iterative_pruning_cpss_switch_applied", False)):
        stable_boost = np.zeros(selected_local.size, dtype=float)
        stable_boost[:] = 1.0
        selected_signal = 0.60 * normalize_fn(base_relevance[selected_local]) + 0.25 * (
            1.0 - normalize_fn(selected_mean_corr)
        ) + 0.15 * stable_boost

    all_scores = np.zeros(n_features, dtype=float)
    all_scores[pool_idx[selected_local]] = np.asarray(selected_signal, dtype=float)
    if np.max(all_scores) <= 0:
        all_scores[selected_global] = np.linspace(1.0, 0.5, num=max(1, selected_global.size))
    all_scores = normalize_fn(all_scores)

    selected_order = np.argsort(all_scores[selected_global])[::-1]
    selected_global = selected_global[np.asarray(selected_order, dtype=int)]

    results = {
        "selected_indices": np.asarray(selected_global, dtype=int),
        "scores": {int(idx): float(all_scores[int(idx)]) for idx in np.asarray(selected_global, dtype=int).tolist()},
        "method": method_name,
        "iterative_pruning_initial_size": int(init_size),
        "iterative_pruning_final_size": int(selected_global.size),
        "iterative_pruning_rounds": int(rounds),
        "iterative_pruning_pool_size": int(pool_idx.size),
        "iterative_pruning_pool_factor": float(iterative_pruning_pool_factor),
        "iterative_pruning_redundancy_weight": float(iterative_pruning_redundancy_weight),
        "iterative_pruning_min_improvement": float(iterative_pruning_min_improvement),
        "iterative_pruning_max_cumulative_loss": float(iterative_pruning_max_cumulative_loss),
        "iterative_pruning_cumulative_delta": float(cumulative_delta),
        "iterative_pruning_max_rounds": int(iterative_pruning_max_rounds),
        "iterative_pruning_removed_features": list(removed_features),
        "iterative_pruning_removed_deltas": list(removed_deltas),
        "iterative_pruning_best_score_final": float(best_score) if np.isfinite(best_score) else float("nan"),
        "iterative_pruning_evaluations": int(evaluations),
        "iterative_pruning_stop_reason": str(stop_reason),
        "iterative_pruning_runtime_bounded": bool(runtime_bounded),
        "iterative_pruning_runtime_sec": float(max(0.0, perf_counter() - start_t)),
        "iterative_pruning_candidate_budgets": list(per_round_candidate_budget),
        "iterative_pruning_corr_estimator": str(corr_estimator),
        "iterative_pruning_corr_shrinkage": float(corr_shrinkage),
        "iterative_pruning_corr_n": int(X_pool.shape[0]),
        "iterative_pruning_corr_p": int(X_pool.shape[1]),
        "iterative_pruning_classes": int(n_classes),
        "iterative_pruning_min_class_count": int(min_class_count),
        "iterative_pruning_imbalance_ratio": float(imbalance_ratio),
    }
    results.update(pareto_meta)
    results.update(cpss_meta)
    if runtime_bounded:
        results.update(
            {
                "iterative_pruning_bounded_prefilter_cap": int(iterative_pruning_bounded_prefilter_cap),
                "iterative_pruning_bounded_candidate_fraction": float(
                    iterative_pruning_bounded_candidate_fraction
                ),
                "iterative_pruning_bounded_min_candidates": int(iterative_pruning_bounded_min_candidates),
                "iterative_pruning_bounded_max_evaluations": int(iterative_pruning_bounded_max_evaluations),
                "iterative_pruning_bounded_max_runtime_seconds": float(
                    iterative_pruning_bounded_max_runtime_seconds
                ),
                "iterative_pruning_bounded_enable_class_gating": bool(
                    iterative_pruning_bounded_enable_class_gating
                ),
                "iterative_pruning_bounded_multiclass_scale": float(
                    iterative_pruning_bounded_multiclass_scale
                ),
                "iterative_pruning_bounded_imbalance_trigger": float(
                    iterative_pruning_bounded_imbalance_trigger
                ),
                "iterative_pruning_bounded_imbalance_scale": float(
                    iterative_pruning_bounded_imbalance_scale
                ),
            }
        )
    return results, {i: float(all_scores[i]) for i in range(n_features)}

"""Prefilter methods for feature pool reduction (extracted from base.py).

Standalone functions that implement class-aware Pareto prefiltering for
iterative wrapper selectors.  Each was previously a method on FeatureSelector.
"""

import numpy as np
import scipy.stats as sps
from scipy.special import gammaln
from sklearn.feature_selection import mutual_info_classif, f_classif

from tabnetics.core.runtime import get_sklearn_n_jobs as _get_sklearn_n_jobs
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Any, Dict, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# label_to_key — imported for class_dominance_pareto_prefilter
# ---------------------------------------------------------------------------
try:
    from .methods.multiclass import label_to_key as _label_to_key
except Exception as exc:
    try:
        from tabnetics.feature_selection.methods.multiclass import label_to_key as _label_to_key
    except Exception as exc:
        def _label_to_key(label):  # type: ignore[misc]
            return str(label)


# ---------------------------------------------------------------------------
# 1. binary_class_prefilter_scores
# ---------------------------------------------------------------------------
def binary_class_prefilter_scores(X, y_bin, random_state, normalize_fn,
                                  include_kw=False, kw_weight=0.25):
    """Binary relevance scores used by class-aware prefilter components."""
    n_features = int(X.shape[1])
    if n_features == 0:
        return np.zeros(0, dtype=float)

    try:
        mi_scores = np.asarray(mutual_info_classif(X, y_bin, random_state=random_state), dtype=float).ravel()
    except Exception as exc:
        mi_scores = np.zeros(n_features, dtype=float)
    mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        f_scores, _ = f_classif(X, y_bin)
        f_scores = np.asarray(f_scores, dtype=float).ravel()
    except Exception as exc:
        f_scores = np.zeros(n_features, dtype=float)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

    kw_scores = np.zeros(n_features, dtype=float)
    if bool(include_kw):
        try:
            ranked = np.apply_along_axis(sps.rankdata, 0, np.asarray(X, dtype=float))
            kw_scores, _ = f_classif(ranked, y_bin)
            kw_scores = np.asarray(kw_scores, dtype=float).ravel()
        except Exception as exc:
            kw_scores = np.zeros(n_features, dtype=float)
        kw_scores = np.nan_to_num(kw_scores, nan=0.0, posinf=0.0, neginf=0.0)

    pos_mask = np.asarray(y_bin, dtype=int) == 1
    neg_mask = ~pos_mask
    if int(np.sum(pos_mask)) >= 1 and int(np.sum(neg_mask)) >= 1:
        pos_mu = np.asarray(np.mean(X[pos_mask], axis=0), dtype=float).ravel()
        neg_mu = np.asarray(np.mean(X[neg_mask], axis=0), dtype=float).ravel()
        std_all = np.asarray(np.std(X, axis=0), dtype=float).ravel()
        effect = np.abs(pos_mu - neg_mu) / np.maximum(1e-8, std_all)
        effect = np.nan_to_num(effect, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        effect = np.zeros(n_features, dtype=float)

    kw_w = float(np.clip(kw_weight, 0.0, 0.80)) if bool(include_kw) else 0.0
    remaining = float(max(1e-12, 1.0 - kw_w))
    base_mi = 0.45 / 0.80
    base_f = 0.35 / 0.80
    base_eff = 0.20 / 0.80
    combined = (
        remaining * base_mi * normalize_fn(mi_scores)
        + remaining * base_f * normalize_fn(f_scores)
        + remaining * base_eff * normalize_fn(effect)
        + kw_w * normalize_fn(kw_scores)
    )
    return np.asarray(normalize_fn(combined), dtype=float).ravel()


# ---------------------------------------------------------------------------
# 2. center_kernel_matrix
# ---------------------------------------------------------------------------
def center_kernel_matrix(K):
    """Center a Gram matrix with H K H."""
    K_arr = np.asarray(K, dtype=float)
    n = int(K_arr.shape[0])
    if n <= 1:
        return np.asarray(K_arr, dtype=float)
    one = np.full((n, n), 1.0 / float(n), dtype=float)
    centered = K_arr - one @ K_arr - K_arr @ one + one @ K_arr @ one
    return np.asarray(np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)


# ---------------------------------------------------------------------------
# 3. rbf_kernel_1d
# ---------------------------------------------------------------------------
def rbf_kernel_1d(values, sigma):
    """Build a 1D RBF kernel from a vector and a sigma value."""
    x = np.asarray(values, dtype=float).ravel()
    if x.size == 0:
        return np.zeros((0, 0), dtype=float), 0.0
    sq = (x[:, None] - x[None, :]) ** 2
    sigma_use = float(sigma)
    if not np.isfinite(sigma_use) or sigma_use <= 0.0:
        sq_flat = np.asarray(sq, dtype=float).ravel()
        sq_flat = sq_flat[np.isfinite(sq_flat) & (sq_flat > 1e-12)]
        if sq_flat.size > 0:
            sigma_use = float(np.sqrt(0.5 * np.median(sq_flat)))
        else:
            sigma_use = 1.0
    sigma_use = float(max(1e-6, sigma_use))
    K = np.exp(-sq / (2.0 * sigma_use * sigma_use))
    return np.asarray(np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0), dtype=float), float(sigma_use)


# ---------------------------------------------------------------------------
# 4. pareto_prefilter_stability_support
# ---------------------------------------------------------------------------
def pareto_prefilter_stability_support(
    X_candidate, y_arr, target,
    stability_subsamples, stability_fraction, random_state,
    mi_scorer, f_scorer, normalize_fn,
    mi_weight=0.60, f_weight=0.40,
):
    """Estimate candidate support frequencies under stratified subsampling."""
    n_samples, n_pool = X_candidate.shape
    target = int(max(1, min(int(target), n_pool)))
    if n_pool <= 0:
        return np.zeros(0, dtype=float), 0, "empty_candidate_pool"
    if n_samples < 8:
        return np.zeros(n_pool, dtype=float), 0, "insufficient_samples"
    if np.unique(y_arr).size < 2:
        return np.zeros(n_pool, dtype=float), 0, "single_class"

    n_rounds = int(max(1, stability_subsamples))
    train_fraction = float(stability_fraction)
    support_counts = np.zeros(n_pool, dtype=float)
    valid_rounds = 0

    splitter = StratifiedShuffleSplit(
        n_splits=n_rounds,
        train_size=train_fraction,
        random_state=int(random_state + 1987),
    )
    try:
        split_iter = splitter.split(np.zeros((n_samples, 1), dtype=float), y_arr)
    except Exception as exc:
        return np.zeros(n_pool, dtype=float), 0, "split_failed"

    for train_idx, _ in split_iter:
        train_idx = np.asarray(train_idx, dtype=int).ravel()
        if train_idx.size < 4:
            continue
        y_sub = np.asarray(y_arr[train_idx]).ravel()
        if np.unique(y_sub).size < 2:
            continue
        X_sub = np.asarray(X_candidate[train_idx], dtype=float)
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
        if mi_scores.size != n_pool or f_scores.size != n_pool:
            continue
        rel = mi_weight * normalize_fn(mi_scores) + f_weight * normalize_fn(f_scores)
        rel = np.asarray(normalize_fn(rel), dtype=float).ravel()
        chosen = np.argsort(rel)[::-1][:target]
        support_counts[np.asarray(chosen, dtype=int)] += 1.0
        valid_rounds += 1

    if valid_rounds <= 0:
        return np.zeros(n_pool, dtype=float), 0, "no_valid_subsamples"

    return np.asarray(support_counts / float(valid_rounds), dtype=float), int(valid_rounds), "ok"


# ---------------------------------------------------------------------------
# 5. class_dominance_pareto_prefilter
# ---------------------------------------------------------------------------
def class_dominance_pareto_prefilter(
    X, y, max_features,
    *,
    prefilter_pool_fn,
    normalize_fn,
    mi_scorer,
    f_scorer,
    random_state,
    problem_type,
    iterative_pruning_class_pareto_prefilter_enabled,
    iterative_pruning_class_pareto_min_classes,
    iterative_pruning_class_pareto_top_per_class,
    iterative_pruning_class_pareto_global_fraction,
    iterative_pruning_class_pareto_minority_boost,
    iterative_pruning_class_pareto_stability_gate_enabled,
    iterative_pruning_class_pareto_stability_subsamples,
    iterative_pruning_class_pareto_stability_fraction,
    iterative_pruning_class_pareto_stability_threshold,
    iterative_pruning_class_pareto_stability_min_overlap,
    iterative_pruning_class_pareto_stability_min_stable_features,
    iterative_pruning_class_pareto_stability_fallback_on_failure,
    mi_weight=0.60,
    f_weight=0.40,
):
    """
    A17: class-dominance-aware Pareto prefilter for iterative wrapper selectors.
    """
    y_arr = np.asarray(y)
    n_samples, n_features = X.shape
    max_features = int(max(1, min(n_features, max_features)))
    fallback = prefilter_pool_fn(X, y_arr, max_features=max_features)

    metadata = {
        "iterative_pruning_pareto_prefilter_enabled": bool(iterative_pruning_class_pareto_prefilter_enabled),
        "iterative_pruning_pareto_prefilter_applied": False,
        "iterative_pruning_pareto_prefilter_reason": "disabled",
        "iterative_pruning_pareto_prefilter_candidate_universe": int(fallback.size),
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
        "iterative_pruning_pareto_stability_gate_enabled": bool(
            iterative_pruning_class_pareto_stability_gate_enabled
        ),
        "iterative_pruning_pareto_stability_gate_applied": False,
        "iterative_pruning_pareto_stability_gate_reason": "disabled",
        "iterative_pruning_pareto_stability_subsamples": int(
            iterative_pruning_class_pareto_stability_subsamples
        ),
        "iterative_pruning_pareto_stability_valid_subsamples": 0,
        "iterative_pruning_pareto_stability_fraction": float(
            iterative_pruning_class_pareto_stability_fraction
        ),
        "iterative_pruning_pareto_stability_threshold": float(
            iterative_pruning_class_pareto_stability_threshold
        ),
        "iterative_pruning_pareto_stability_min_overlap": float(
            iterative_pruning_class_pareto_stability_min_overlap
        ),
        "iterative_pruning_pareto_stability_min_stable_features": int(
            iterative_pruning_class_pareto_stability_min_stable_features
        ),
        "iterative_pruning_pareto_stability_stable_feature_count": 0,
        "iterative_pruning_pareto_stability_overlap_recall": float("nan"),
        "iterative_pruning_pareto_stability_repair_applied": False,
        "iterative_pruning_pareto_stability_repair_replaced_fraction": 0.0,
        "iterative_pruning_pareto_stability_fallback_on_failure": bool(
            iterative_pruning_class_pareto_stability_fallback_on_failure
        ),
        "iterative_pruning_pareto_stability_fallback_triggered": False,
    }
    if not iterative_pruning_class_pareto_prefilter_enabled:
        return fallback, metadata
    if problem_type != "classification":
        metadata["iterative_pruning_pareto_prefilter_reason"] = "non_classification_problem"
        return fallback, metadata

    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.size < int(iterative_pruning_class_pareto_min_classes):
        metadata["iterative_pruning_pareto_prefilter_reason"] = "insufficient_classes"
        return fallback, metadata
    if counts.size == 0 or int(np.min(counts)) < 2:
        metadata["iterative_pruning_pareto_prefilter_reason"] = "insufficient_class_support"
        return fallback, metadata

    global_cap = int(
        min(
            n_features,
            max(
                max_features,
                max_features + int(np.ceil(float(iterative_pruning_class_pareto_global_fraction) * max_features)),
            ),
        )
    )
    global_pool = prefilter_pool_fn(X, y_arr, max_features=global_cap)
    candidate_set = set(int(i) for i in np.asarray(global_pool, dtype=int).tolist())
    top_per_class = int(min(n_features, max(4, iterative_pruning_class_pareto_top_per_class)))
    max_count = int(np.max(counts))

    class_scores: Dict[str, np.ndarray] = {}
    class_top_sets: Dict[str, set] = {}
    class_weights: Dict[str, float] = {}
    for cls, cnt in zip(classes.tolist(), counts.tolist()):
        y_bin = (y_arr == cls).astype(int)
        if int(np.sum(y_bin == 1)) < 2 or int(np.sum(y_bin == 0)) < 2:
            continue
        score = binary_class_prefilter_scores(X, y_bin, random_state=random_state, normalize_fn=normalize_fn)
        if score.size != n_features:
            continue
        weight = float(
            (max_count / max(1.0, float(cnt))) ** float(iterative_pruning_class_pareto_minority_boost)
        )
        weighted = np.asarray(score * weight, dtype=float).ravel()
        key = str(_label_to_key(cls))
        class_scores[key] = weighted
        class_weights[key] = float(weight)
        top_idx = np.argsort(weighted)[::-1][:top_per_class]
        top_set = set(int(i) for i in np.asarray(top_idx, dtype=int).tolist())
        class_top_sets[key] = top_set
        candidate_set.update(top_set)

    if len(class_scores) < 2:
        metadata["iterative_pruning_pareto_prefilter_reason"] = "insufficient_valid_class_scores"
        return fallback, metadata

    candidate_idx = np.array(sorted(candidate_set), dtype=int)
    if candidate_idx.size == 0:
        metadata["iterative_pruning_pareto_prefilter_reason"] = "empty_candidate_pool"
        return fallback, metadata

    # Rank matrix (lower rank is better) used for Pareto dominance.
    rank_rows = []
    for cls_key in class_scores:
        scores = np.asarray(class_scores[cls_key][candidate_idx], dtype=float).ravel()
        order = np.argsort(scores)[::-1]
        ranks = np.empty(order.size, dtype=float)
        ranks[order] = np.arange(1, order.size + 1, dtype=float)
        rank_rows.append(ranks)
    rank_matrix = np.vstack(rank_rows)

    m = int(candidate_idx.size)
    dominated_by = np.zeros(m, dtype=int)
    dominates = np.zeros(m, dtype=int)
    for i in range(m):
        vi = rank_matrix[:, i]
        for j in range(m):
            if i == j:
                continue
            vj = rank_matrix[:, j]
            i_dominates_j = bool(np.all(vi <= vj) and np.any(vi < vj))
            j_dominates_i = bool(np.all(vj <= vi) and np.any(vj < vi))
            if i_dominates_j:
                dominates[i] += 1
            if j_dominates_i:
                dominated_by[i] += 1

    front_mask = dominated_by == 0
    front_size = int(np.sum(front_mask))

    # Add global multiclass relevance signal for tiebreaks.
    X_candidate = np.asarray(X[:, candidate_idx], dtype=float)
    try:
        mi_global = np.asarray(mi_scorer(X_candidate, y_arr, random_state=random_state), dtype=float).ravel()
    except Exception as exc:
        mi_global = np.zeros(candidate_idx.size, dtype=float)
    mi_global = np.nan_to_num(mi_global, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        f_global, _ = f_scorer(X_candidate, y_arr)
        f_global = np.asarray(f_global, dtype=float).ravel()
    except Exception as exc:
        f_global = np.zeros(candidate_idx.size, dtype=float)
    f_global = np.nan_to_num(f_global, nan=0.0, posinf=0.0, neginf=0.0)
    global_signal = mi_weight * normalize_fn(mi_global) + f_weight * normalize_fn(f_global)
    global_signal = np.asarray(normalize_fn(global_signal), dtype=float).ravel()

    pareto_signal = 1.0 - normalize_fn(np.asarray(dominated_by, dtype=float))
    dominates_signal = normalize_fn(np.asarray(dominates, dtype=float))
    combined = 0.55 * pareto_signal + 0.25 * dominates_signal + 0.20 * global_signal
    combined = np.asarray(normalize_fn(combined), dtype=float).ravel()

    selected_local = np.argsort(combined)[::-1][:max_features]
    selected = np.asarray(candidate_idx[selected_local], dtype=int)
    if selected.size == 0:
        metadata["iterative_pruning_pareto_prefilter_reason"] = "selection_empty_after_pareto"
        return fallback, metadata

    metadata["iterative_pruning_pareto_prefilter_candidate_universe"] = int(candidate_idx.size)
    metadata["iterative_pruning_pareto_prefilter_front_size"] = int(front_size)
    metadata["iterative_pruning_pareto_prefilter_classes_used"] = int(len(class_scores))
    metadata["iterative_pruning_pareto_prefilter_class_weights"] = class_weights

    if bool(iterative_pruning_class_pareto_stability_gate_enabled):
        metadata["iterative_pruning_pareto_stability_gate_applied"] = True
        X_candidate = np.asarray(X[:, candidate_idx], dtype=float)
        support_freq, valid_subsamples, support_reason = pareto_prefilter_stability_support(
            X_candidate,
            y_arr,
            target=max_features,
            stability_subsamples=iterative_pruning_class_pareto_stability_subsamples,
            stability_fraction=iterative_pruning_class_pareto_stability_fraction,
            random_state=random_state,
            mi_scorer=mi_scorer,
            f_scorer=f_scorer,
            normalize_fn=normalize_fn,
            mi_weight=mi_weight,
            f_weight=f_weight,
        )
        metadata["iterative_pruning_pareto_stability_valid_subsamples"] = int(valid_subsamples)
        if support_reason != "ok":
            metadata["iterative_pruning_pareto_stability_gate_reason"] = str(support_reason)
        else:
            stable_local = np.where(
                np.asarray(support_freq, dtype=float).ravel()
                >= float(iterative_pruning_class_pareto_stability_threshold)
            )[0]
            stable_set = set(int(i) for i in np.asarray(stable_local, dtype=int).tolist())
            selected_local_set = set(int(i) for i in np.asarray(selected_local, dtype=int).tolist())
            inter_count = int(len(selected_local_set.intersection(stable_set)))
            overlap_recall = float(inter_count / max(1, len(selected_local_set)))
            metadata["iterative_pruning_pareto_stability_stable_feature_count"] = int(stable_local.size)
            metadata["iterative_pruning_pareto_stability_overlap_recall"] = float(overlap_recall)

            if stable_local.size < int(iterative_pruning_class_pareto_stability_min_stable_features):
                metadata["iterative_pruning_pareto_stability_gate_reason"] = "insufficient_stable_features"
                if bool(iterative_pruning_class_pareto_stability_fallback_on_failure):
                    metadata["iterative_pruning_pareto_stability_fallback_triggered"] = True
                    metadata["iterative_pruning_pareto_prefilter_reason"] = "stability_gate_fallback"
                    return fallback, metadata
            elif overlap_recall < float(iterative_pruning_class_pareto_stability_min_overlap):
                freq_norm = normalize_fn(np.asarray(support_freq, dtype=float).ravel())
                stable_bonus = np.zeros(candidate_idx.size, dtype=float)
                stable_bonus[np.asarray(stable_local, dtype=int)] = 1.0
                repaired_score = 0.55 * combined + 0.30 * freq_norm + 0.15 * stable_bonus
                repaired_score = np.asarray(normalize_fn(repaired_score), dtype=float).ravel()
                repaired_local = np.argsort(repaired_score)[::-1][:max_features]
                repaired_set = set(int(i) for i in np.asarray(repaired_local, dtype=int).tolist())
                replaced_frac = float(
                    1.0 - (len(selected_local_set.intersection(repaired_set)) / max(1, len(selected_local_set)))
                )
                selected_local = np.asarray(repaired_local, dtype=int)
                selected = np.asarray(candidate_idx[selected_local], dtype=int)
                metadata["iterative_pruning_pareto_stability_repair_applied"] = True
                metadata["iterative_pruning_pareto_stability_repair_replaced_fraction"] = float(replaced_frac)
                metadata["iterative_pruning_pareto_stability_gate_reason"] = "repair_applied"
            else:
                metadata["iterative_pruning_pareto_stability_gate_reason"] = "pass"

    class_hits = {}
    selected_set = set(int(i) for i in selected.tolist())
    for cls_key, top_set in class_top_sets.items():
        class_hits[cls_key] = int(len(selected_set.intersection(top_set)))

    metadata.update(
        {
            "iterative_pruning_pareto_prefilter_applied": True,
            "iterative_pruning_pareto_prefilter_reason": (
                "ok_stability_repaired"
                if bool(metadata.get("iterative_pruning_pareto_stability_repair_applied", False))
                else "ok"
            ),
            "iterative_pruning_pareto_prefilter_class_top_hits": class_hits,
        }
    )
    return selected, metadata


# ---------------------------------------------------------------------------
# 6. Multi-strategy prefilter union (T-R-127)
# ---------------------------------------------------------------------------
def build_prefilter_union_pool(
    X,
    y,
    *,
    max_features: int,
    strategies: Sequence[str],
    nondefault_budget_fraction: float,
    base_scores: np.ndarray,
    mi_scores: np.ndarray,
    f_scores: np.ndarray,
    normalize_fn,
    random_state: int,
    problem_type: str,
    wsnr_stabilize_counts: bool = True,
    wsnr_data_domain: str = "auto",
) -> np.ndarray:
    """Compose prefilter pools from multiple strategies and return a capped union."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()
    if X_arr.ndim != 2:
        return np.array([], dtype=int)
    n_samples, n_features = X_arr.shape
    if n_samples < 2 or n_features <= 0:
        return np.array([], dtype=int)
    max_features = int(max(1, min(max_features, n_features)))
    if n_features <= max_features:
        return np.arange(n_features, dtype=int)

    cleaned = [str(s).strip().lower() for s in strategies if str(s).strip()]
    if not cleaned:
        cleaned = ["mi_ftest_blend"]
    if "mi_ftest_blend" not in cleaned:
        cleaned = ["mi_ftest_blend"] + cleaned

    budget_nondefault = int(
        max(
            1,
            min(
                max_features,
                np.ceil(float(max_features) * float(np.clip(nondefault_budget_fraction, 0.01, 0.50))),
            ),
        )
    )

    strategy_scores: Dict[str, np.ndarray] = {}
    selected_by_strategy: Dict[str, np.ndarray] = {}

    base = np.asarray(base_scores, dtype=float).ravel()
    if base.size != n_features:
        base = np.zeros(n_features, dtype=float)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    strategy_scores["mi_ftest_blend"] = np.asarray(normalize_fn(base), dtype=float).ravel()
    selected_by_strategy["mi_ftest_blend"] = np.argsort(strategy_scores["mi_ftest_blend"])[::-1][:max_features]

    for strategy in cleaned:
        if strategy == "mi_ftest_blend":
            continue
        if strategy == "rf_importance":
            scores = np.zeros(n_features, dtype=float)
            try:
                if str(problem_type).strip().lower() == "classification":
                    from sklearn.ensemble import RandomForestClassifier

                    model = RandomForestClassifier(
                        n_estimators=128,
                        random_state=int(random_state),
                        n_jobs=_get_sklearn_n_jobs(),
                    )
                else:
                    from sklearn.ensemble import RandomForestRegressor

                    model = RandomForestRegressor(
                        n_estimators=128,
                        random_state=int(random_state),
                        n_jobs=_get_sklearn_n_jobs(),
                    )
                model.fit(X_arr, y_arr)
                scores = np.asarray(getattr(model, "feature_importances_", scores), dtype=float).ravel()
            except Exception as exc:
                scores = np.zeros(n_features, dtype=float)
            scores = np.asarray(normalize_fn(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)), dtype=float)
            strategy_scores[strategy] = scores
            selected_by_strategy[strategy] = np.argsort(scores)[::-1][:budget_nondefault]
            continue

        if strategy == "relieff_scores":
            scores = np.zeros(n_features, dtype=float)
            try:
                from .methods.filter import relieff_selection

                _, all_scores = relieff_selection(
                    X_arr,
                    y_arr,
                    n_target_features=budget_nondefault,
                    n_neighbors=min(10, max(1, n_samples - 1)),
                )
                score_vec = np.array(
                    [float(all_scores.get(i, 0.0)) for i in range(n_features)],
                    dtype=float,
                )
                scores = score_vec
            except Exception as exc:
                scores = np.zeros(n_features, dtype=float)
            scores = np.asarray(normalize_fn(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)), dtype=float)
            strategy_scores[strategy] = scores
            selected_by_strategy[strategy] = np.argsort(scores)[::-1][:budget_nondefault]
            continue

        if strategy == "wsnr":
            scores = np.zeros(n_features, dtype=float)
            try:
                scores = np.asarray(
                    wsnr_scores(
                        X_arr,
                        y_arr,
                        wsnr_stabilize_counts=bool(wsnr_stabilize_counts),
                        data_domain=str(wsnr_data_domain),
                    ),
                    dtype=float,
                ).ravel()
            except Exception as exc:
                scores = np.zeros(n_features, dtype=float)
            if scores.size != n_features:
                scores = np.zeros(n_features, dtype=float)
            scores = np.asarray(normalize_fn(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)), dtype=float)
            strategy_scores[strategy] = scores
            selected_by_strategy[strategy] = np.argsort(scores)[::-1][:budget_nondefault]
            continue

        # T-R-265: BH-adjusted FDR prefilter (Welch t-test for binary, ANOVA F-test for multiclass)
        if strategy == "bh_fdr":
            scores = np.zeros(n_features, dtype=float)
            try:
                from scipy import stats as _stats
                from statsmodels.stats.multitest import multipletests as _multipletests

                classes = np.unique(y_arr)
                n_classes_local = int(classes.size)
                pvals = np.ones(n_features, dtype=float)

                if n_classes_local == 2:
                    # Binary: Welch's t-test per feature
                    mask0 = y_arr == classes[0]
                    mask1 = y_arr == classes[1]
                    for j in range(n_features):
                        try:
                            _, pv = _stats.ttest_ind(
                                X_arr[mask0, j], X_arr[mask1, j], equal_var=False,
                            )
                            pvals[j] = float(pv) if np.isfinite(pv) else 1.0
                        except Exception:
                            pvals[j] = 1.0
                elif n_classes_local > 2:
                    # Multiclass: one-way ANOVA F-test per feature
                    groups = [X_arr[y_arr == c] for c in classes]
                    for j in range(n_features):
                        try:
                            col_groups = [g[:, j] for g in groups if g.shape[0] > 0]
                            if len(col_groups) >= 2:
                                f_val, pv = _stats.f_oneway(*col_groups)
                                pvals[j] = float(pv) if np.isfinite(pv) else 1.0
                            else:
                                pvals[j] = 1.0
                        except Exception:
                            pvals[j] = 1.0

                # BH correction
                reject, _, _, _ = _multipletests(pvals, alpha=0.05, method="fdr_bh")
                bh_selected = np.where(reject)[0]

                # Use -log10(pval) as score for aggregation
                scores = np.clip(-np.log10(np.clip(pvals, 1e-300, 1.0)), 0.0, 300.0)
                scores = np.asarray(
                    normalize_fn(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)), dtype=float,
                )
                strategy_scores[strategy] = scores

                if bh_selected.size > 0:
                    # Use BH-significant features (capped to budget)
                    if bh_selected.size > budget_nondefault:
                        # Rank by score within significant set
                        ranked = bh_selected[np.argsort(scores[bh_selected])[::-1]]
                        selected_by_strategy[strategy] = ranked[:budget_nondefault]
                    else:
                        selected_by_strategy[strategy] = bh_selected
                else:
                    # No features pass FDR; fall back to top by raw p-value
                    selected_by_strategy[strategy] = np.argsort(scores)[::-1][:budget_nondefault]
            except ImportError:
                # statsmodels not available; use raw F-test fallback
                selected_by_strategy[strategy] = np.argsort(scores)[::-1][:budget_nondefault]
            except Exception:
                selected_by_strategy[strategy] = np.argsort(scores)[::-1][:budget_nondefault]
            continue

    union_set = set()
    for idx in selected_by_strategy.values():
        union_set.update(int(i) for i in np.asarray(idx, dtype=int).tolist())
    if not union_set:
        return np.array([], dtype=int)

    union_idx = np.array(sorted(union_set), dtype=int)
    if union_idx.size <= max_features:
        return union_idx

    # Aggregate per-strategy scores to prioritize union members.
    score_accum = np.zeros(n_features, dtype=float)
    for strategy, scores in strategy_scores.items():
        w = 1.0 if strategy == "mi_ftest_blend" else 0.5
        score_accum += float(w) * np.asarray(scores, dtype=float).ravel()
    score_accum = np.nan_to_num(score_accum, nan=0.0, posinf=0.0, neginf=0.0)
    ranked_union = union_idx[np.argsort(score_accum[union_idx])[::-1]]
    selected = np.array(sorted(set(int(i) for i in ranked_union[:max_features])), dtype=int)
    return selected


# ---------------------------------------------------------------------------
# 7. WSNR binary prefilter strategy (T-R-172)
# ---------------------------------------------------------------------------
def _wsnr_count_like_heuristic(X: np.ndarray, *, zero_threshold: float = 0.30) -> bool:
    """Heuristic detector for sparse count-like matrices."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return False
    finite = np.isfinite(arr)
    if not np.any(finite):
        return False
    vals = arr[finite]
    if vals.size == 0:
        return False
    if float(np.min(vals)) < -1e-12:
        return False
    integer_like = float(np.mean(np.isclose(vals, np.round(vals), atol=1e-8)))
    zero_frac = float(np.mean(np.isclose(vals, 0.0, atol=1e-12)))
    return bool(integer_like >= 0.70 and zero_frac >= float(max(0.0, zero_threshold)))


def _detect_rnaseq_data(
    X: np.ndarray,
    *,
    data_domain: str = "auto",
    min_zero_fraction: float = 0.20,
    min_integer_like: float = 0.70,
) -> Tuple[bool, Dict[str, Any]]:
    """Detect RNA-seq-like count matrices.

    Detection is domain-aware:
    - `data_domain="rnaseq"` forces True.
    - otherwise we require nonnegative values, integer-like support, and sparsity.
    """
    domain = str(data_domain or "auto").strip().lower()
    arr = np.asarray(X, dtype=float)
    meta: Dict[str, Any] = {
        "data_domain": domain,
        "forced_rnaseq_domain": bool(domain == "rnaseq"),
        "is_rnaseq": False,
        "reason": "undetermined",
        "integer_like_fraction": 0.0,
        "zero_fraction": 0.0,
        "min_value": float("nan"),
    }

    if arr.ndim != 2 or arr.size == 0:
        meta["reason"] = "invalid_shape"
        return False, meta

    finite = np.isfinite(arr)
    if not np.any(finite):
        meta["reason"] = "no_finite_values"
        return False, meta

    vals = arr[finite]
    meta["min_value"] = float(np.min(vals))
    if domain == "rnaseq":
        meta["is_rnaseq"] = True
        meta["reason"] = "forced_domain"
        return True, meta

    if float(np.min(vals)) < -1e-12:
        meta["reason"] = "negative_values"
        return False, meta

    integer_like = float(np.mean(np.isclose(vals, np.round(vals), atol=1e-8)))
    zero_frac = float(np.mean(np.isclose(vals, 0.0, atol=1e-12)))
    meta["integer_like_fraction"] = float(integer_like)
    meta["zero_fraction"] = float(zero_frac)

    if integer_like >= float(min_integer_like) and zero_frac >= float(min_zero_fraction):
        meta["is_rnaseq"] = True
        meta["reason"] = "count_like_heuristic"
        return True, meta

    meta["reason"] = "not_count_like"
    return False, meta


def _compute_tmm_size_factors(
    counts: np.ndarray,
    *,
    logratio_trim: float = 0.30,
    sum_trim: float = 0.05,
) -> np.ndarray:
    """Compute approximate TMM size factors with robust trimming.

    This is a lightweight implementation suitable for feature prefiltering.
    """
    arr = np.asarray(counts, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.ones(max(0, arr.shape[0]), dtype=float)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    lib_sizes = np.asarray(np.sum(arr, axis=1), dtype=float).ravel()
    n_samples = int(arr.shape[0])
    valid = lib_sizes > 0.0
    if not np.any(valid):
        return np.ones(n_samples, dtype=float)

    # Use the sample closest to median library size as the reference.
    lib_valid = lib_sizes[valid]
    ref_lib = float(np.median(lib_valid))
    ref_idx = int(np.argmin(np.abs(lib_sizes - ref_lib)))
    ref = np.asarray(arr[ref_idx], dtype=float).ravel()
    ref_lib = float(max(1.0, lib_sizes[ref_idx]))

    norm_factors = np.ones(n_samples, dtype=float)
    for i in range(n_samples):
        if not valid[i]:
            continue
        obs = np.asarray(arr[i], dtype=float).ravel()
        obs_lib = float(max(1.0, lib_sizes[i]))
        mask = (obs > 0.0) & (ref > 0.0)
        if not np.any(mask):
            continue
        obs_use = obs[mask]
        ref_use = ref[mask]

        log_obs = np.log2(obs_use / obs_lib)
        log_ref = np.log2(ref_use / ref_lib)
        m_vals = log_obs - log_ref
        a_vals = 0.5 * (log_obs + log_ref)

        lo_m, hi_m = np.quantile(m_vals, [float(logratio_trim), float(1.0 - logratio_trim)])
        lo_a, hi_a = np.quantile(a_vals, [float(sum_trim), float(1.0 - sum_trim)])
        keep = (m_vals >= lo_m) & (m_vals <= hi_m) & (a_vals >= lo_a) & (a_vals <= hi_a)
        if not np.any(keep):
            continue

        m_keep = m_vals[keep]
        obs_keep = obs_use[keep]
        ref_keep = ref_use[keep]
        # Delta-method variance approximation for log-ratios.
        weights = 1.0 / np.maximum(1e-12, (1.0 / obs_keep) + (1.0 / ref_keep))
        denom = float(np.sum(weights))
        if denom <= 0.0:
            continue
        tmm_log2 = float(np.sum(weights * m_keep) / denom)
        norm_factors[i] = float(max(1e-8, 2.0 ** tmm_log2))

    size_factors = lib_sizes * norm_factors
    valid_sf = size_factors > 0.0
    if not np.any(valid_sf):
        return np.ones(n_samples, dtype=float)
    scale = float(np.median(size_factors[valid_sf]))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    size_factors = size_factors / scale
    size_factors = np.asarray(
        np.nan_to_num(size_factors, nan=1.0, posinf=1.0, neginf=1.0),
        dtype=float,
    )
    size_factors = np.maximum(size_factors, 1e-8)
    return size_factors


def _rnaseq_transform(
    X: np.ndarray,
    *,
    data_domain: str = "auto",
    enabled: bool = True,
    force: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply log2(CPM + 1) + TMM normalization for RNA-seq count data."""
    arr = np.asarray(X, dtype=float)
    base_meta: Dict[str, Any] = {
        "rnaseq_transform_enabled": bool(enabled),
        "rnaseq_transform_forced": bool(force),
        "rnaseq_transform_applied": False,
        "rnaseq_transform_reason": "disabled",
        "rnaseq_size_factor_min": float("nan"),
        "rnaseq_size_factor_max": float("nan"),
        "rnaseq_size_factor_median": float("nan"),
    }
    if arr.ndim != 2:
        base_meta["rnaseq_transform_reason"] = "invalid_shape"
        return arr, base_meta

    is_rnaseq, detect_meta = _detect_rnaseq_data(arr, data_domain=data_domain)
    meta = dict(base_meta)
    meta.update(detect_meta)
    if not bool(enabled):
        meta["rnaseq_transform_reason"] = "disabled"
        return arr, meta
    if not (bool(force) or bool(is_rnaseq)):
        meta["rnaseq_transform_reason"] = "not_rnaseq"
        return arr, meta

    counts = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    counts = np.clip(counts, a_min=0.0, a_max=None)
    if counts.shape[1] == 0:
        meta["rnaseq_transform_reason"] = "empty_features"
        return counts, meta

    size_factors = _compute_tmm_size_factors(counts)
    cpm = (counts / size_factors[:, None]) * 1e6
    transformed = np.log2(cpm + 1.0)
    transformed = np.asarray(
        np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=float,
    )
    meta["rnaseq_transform_applied"] = True
    meta["rnaseq_transform_reason"] = "ok"
    meta["rnaseq_size_factor_min"] = float(np.min(size_factors)) if size_factors.size else float("nan")
    meta["rnaseq_size_factor_max"] = float(np.max(size_factors)) if size_factors.size else float("nan")
    meta["rnaseq_size_factor_median"] = float(np.median(size_factors)) if size_factors.size else float("nan")
    return transformed, meta


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        return np.zeros(0, dtype=float)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    p = np.clip(p, 0.0, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    m = float(p.size)
    q_ranked = np.zeros_like(ranked)
    running = 1.0
    for i in range(ranked.size - 1, -1, -1):
        rank = float(i + 1)
        val = float(ranked[i] * m / rank)
        running = min(running, val)
        q_ranked[i] = running
    q = np.zeros_like(q_ranked)
    q[order] = np.clip(q_ranked, 0.0, 1.0)
    return q


def rnaseq_nb_lrt_scores(
    X: np.ndarray,
    y: np.ndarray,
    *,
    data_domain: str = "auto",
    alpha: float = 0.10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute RNA-seq Negative-Binomial LRT scores with BH-FDR control."""
    arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()
    n_features = int(arr.shape[1]) if arr.ndim == 2 else 0
    out_scores = np.zeros(max(0, n_features), dtype=float)
    meta: Dict[str, Any] = {
        "rnaseq_nb_lrt_applied": False,
        "rnaseq_nb_lrt_reason": "uninitialized",
        "rnaseq_nb_lrt_alpha": float(np.clip(alpha, 1e-6, 0.5)),
        "rnaseq_nb_lrt_significant_count": 0,
        "rnaseq_nb_lrt_q_values": np.ones(max(0, n_features), dtype=float),
    }

    if arr.ndim != 2 or arr.shape[0] < 4 or n_features <= 0:
        meta["rnaseq_nb_lrt_reason"] = "invalid_shape"
        return out_scores, meta

    is_rnaseq, detect_meta = _detect_rnaseq_data(arr, data_domain=data_domain)
    meta.update(detect_meta)
    if not bool(is_rnaseq):
        meta["rnaseq_nb_lrt_reason"] = "not_rnaseq"
        return out_scores, meta

    classes, y_codes = np.unique(y_arr, return_inverse=True)
    n_classes = int(classes.size)
    if n_classes < 2:
        meta["rnaseq_nb_lrt_reason"] = "single_class"
        return out_scores, meta

    counts = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    counts = np.clip(counts, a_min=0.0, a_max=None)
    size_factors = _compute_tmm_size_factors(counts)
    norm_counts = counts / size_factors[:, None]
    norm_counts = np.asarray(
        np.nan_to_num(norm_counts, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=float,
    )

    p_values = np.ones(n_features, dtype=float)
    df = int(max(1, n_classes - 1))
    for j in range(n_features):
        x = np.asarray(norm_counts[:, j], dtype=float).ravel()
        mu = float(np.mean(x))
        var = float(np.var(x, ddof=1)) if x.size > 1 else float(mu)
        if (not np.isfinite(mu)) or mu <= 0.0:
            continue
        alpha_j = float(max(1e-8, (var - mu) / max(1e-8, mu * mu)))
        r = float(1.0 / alpha_j)

        # Null model: shared mean.
        mu0 = np.full(x.size, float(max(1e-8, mu)), dtype=float)
        p0 = r / (r + mu0)
        ll0 = (
            gammaln(x + r)
            - gammaln(r)
            - gammaln(x + 1.0)
            + r * np.log(np.clip(p0, 1e-12, 1.0))
            + x * np.log(np.clip(1.0 - p0, 1e-12, 1.0))
        )

        # Alternative model: class-wise means.
        mu1 = np.zeros_like(x, dtype=float)
        for cls_idx in range(n_classes):
            mask = y_codes == cls_idx
            if not np.any(mask):
                continue
            cls_mean = float(np.mean(x[mask]))
            mu1[mask] = float(max(1e-8, cls_mean))
        p1 = r / (r + mu1)
        ll1 = (
            gammaln(x + r)
            - gammaln(r)
            - gammaln(x + 1.0)
            + r * np.log(np.clip(p1, 1e-12, 1.0))
            + x * np.log(np.clip(1.0 - p1, 1e-12, 1.0))
        )

        lrt = float(max(0.0, 2.0 * (np.sum(ll1) - np.sum(ll0))))
        p_values[j] = float(sps.chi2.sf(lrt, df=df))

    q_values = _bh_fdr(p_values)
    scores = -np.log10(np.clip(q_values, 1e-12, 1.0))
    scores = np.asarray(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)

    alpha_eff = float(np.clip(alpha, 1e-6, 0.5))
    sig_count = int(np.sum(q_values <= alpha_eff))
    meta["rnaseq_nb_lrt_applied"] = True
    meta["rnaseq_nb_lrt_reason"] = "ok"
    meta["rnaseq_nb_lrt_significant_count"] = int(sig_count)
    meta["rnaseq_nb_lrt_q_values"] = np.asarray(q_values, dtype=float)
    return scores, meta


def _wsnr_maybe_stabilize_counts(
    X: np.ndarray,
    *,
    data_domain: str = "auto",
    wsnr_stabilize_counts: bool = True,
) -> np.ndarray:
    """Apply lightweight log1p library-size stabilization for count-like inputs."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        return np.asarray(arr, dtype=float)
    if not bool(wsnr_stabilize_counts):
        return np.asarray(arr, dtype=float)

    domain = str(data_domain or "auto").strip().lower()
    force_rnaseq = domain == "rnaseq"
    if not force_rnaseq and not _wsnr_count_like_heuristic(arr):
        return np.asarray(arr, dtype=float)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    lib_sizes = np.asarray(np.sum(arr, axis=1), dtype=float).ravel()
    valid = lib_sizes > 0.0
    if not np.any(valid):
        return np.log1p(arr)
    median_lib = float(np.median(lib_sizes[valid]))
    if not np.isfinite(median_lib) or median_lib <= 0.0:
        median_lib = 1.0
    size_factors = np.ones(arr.shape[0], dtype=float)
    size_factors[valid] = lib_sizes[valid] / median_lib
    size_factors = np.maximum(size_factors, 1e-8)
    stabilized = np.log1p(arr / size_factors[:, None])
    stabilized = np.asarray(
        np.nan_to_num(stabilized, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=float,
    )
    return stabilized


def wsnr_scores(X, y, *, wsnr_stabilize_counts: bool = True, data_domain: str = "auto") -> np.ndarray:
    """Compute weighted signal-to-noise ratio scores for binary classification.

    Returns a zero vector when the target is not binary or class diversity
    is insufficient for a stable estimate.
    """
    X_arr = _wsnr_maybe_stabilize_counts(
        np.asarray(X, dtype=float),
        data_domain=str(data_domain),
        wsnr_stabilize_counts=bool(wsnr_stabilize_counts),
    )
    y_arr = np.asarray(y).ravel()
    if X_arr.ndim != 2:
        return np.zeros(0, dtype=float)
    n_samples, n_features = X_arr.shape
    if n_samples < 2 or n_features <= 0:
        return np.zeros(max(0, n_features), dtype=float)

    classes = np.unique(y_arr)
    if classes.size != 2:
        return np.zeros(n_features, dtype=float)

    c0, c1 = classes.tolist()
    m0 = np.asarray(y_arr == c0, dtype=bool)
    m1 = np.asarray(y_arr == c1, dtype=bool)
    n0 = int(np.sum(m0))
    n1 = int(np.sum(m1))
    if n0 < 1 or n1 < 1:
        return np.zeros(n_features, dtype=float)

    x0 = np.asarray(X_arr[m0], dtype=float)
    x1 = np.asarray(X_arr[m1], dtype=float)
    mu0 = np.asarray(np.nanmean(x0, axis=0), dtype=float).ravel()
    mu1 = np.asarray(np.nanmean(x1, axis=0), dtype=float).ravel()
    var0 = np.asarray(np.nanvar(x0, axis=0), dtype=float).ravel()
    var1 = np.asarray(np.nanvar(x1, axis=0), dtype=float).ravel()

    # Class-frequency-weighted noise stabilizes scores under mild imbalance.
    # Use data-scale-aware minimum noise to prevent extreme scores (~1e6) on
    # constant-within-class features (zero variance).  The floor is the median
    # pooled std × 1e-3 (or 1e-8 absolute minimum), ensuring WSNR scores stay
    # bounded even for degenerate features.
    n_total = float(max(1, n0 + n1))
    w0 = float(n0 / n_total)
    w1 = float(n1 / n_total)
    pooled_var = w0 * var0 + w1 * var1
    _median_std = float(np.median(np.sqrt(np.maximum(1e-30, pooled_var))))
    _min_noise = max(1e-8, _median_std * 1e-3)
    noise = np.sqrt(np.maximum(_min_noise ** 2, pooled_var))
    signal = np.abs(mu1 - mu0)
    scores = signal / noise
    scores = np.asarray(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), dtype=float).ravel()
    return scores


# ---------------------------------------------------------------------------
# 8. E-value utilities (T-R-111)
# ---------------------------------------------------------------------------
def pvalues_to_evalues(p_values: np.ndarray) -> np.ndarray:
    """Convert p-values to e-values via e = 1 / p (clipped for stability)."""
    pvals = np.asarray(p_values, dtype=float).ravel()
    if pvals.size == 0:
        return np.zeros(0, dtype=float)
    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
    pvals = np.clip(pvals, 1e-12, 1.0)
    e_vals = 1.0 / pvals
    return np.asarray(np.nan_to_num(e_vals, nan=0.0, posinf=1e12, neginf=0.0), dtype=float)


def ebh_support(e_values: np.ndarray, alpha: float) -> np.ndarray:
    """e-BH support set from an e-value vector."""
    e_vals = np.asarray(e_values, dtype=float).ravel()
    if e_vals.size == 0:
        return np.array([], dtype=int)
    e_vals = np.nan_to_num(e_vals, nan=0.0, posinf=1e12, neginf=0.0)
    p = int(e_vals.size)
    order = np.argsort(e_vals)[::-1]
    e_sorted = e_vals[order]
    thresh = (np.arange(1, p + 1, dtype=float) * float(alpha)) / float(p)
    thresh = np.maximum(thresh, 1e-12)
    valid = np.where(e_sorted >= (1.0 / thresh))[0]
    if valid.size == 0:
        return np.array([], dtype=int)
    k_hat = int(valid.max())
    return np.sort(order[: k_hat + 1])


# ---------------------------------------------------------------------------
# 9. Batch-correction helpers (T-R-145, T-R-149)
# ---------------------------------------------------------------------------
_BATCH_CORRECTION_MODES = {"none", "combat", "cdf_center", "combat_seq", "center_scale"}


def _canonicalize_batch_correction_mode(mode: Any) -> str:
    raw = str(mode if mode is not None else "none").strip().lower()
    alias = {
        "cdf": "cdf_center",
        "cdf-center": "cdf_center",
        "cdfcenter": "cdf_center",
        "center_cdf": "cdf_center",
        "combat-seq": "combat_seq",
        "combatseq": "combat_seq",
        "center-scale": "center_scale",
        "centerscale": "center_scale",
    }
    raw = alias.get(raw, raw)
    if raw not in _BATCH_CORRECTION_MODES:
        raw = "none"
    return raw


def _batch_label_key(label: Any) -> Any:
    if label is None:
        return "__missing__"
    try:
        if isinstance(label, float) and np.isnan(label):
            return "__missing__"
    except Exception as exc:
        pass
    return _label_to_key(label)


def _fit_encode_batch_labels(
    batch_labels: Optional[Sequence[Any]],
    n_rows: int,
) -> Tuple[Optional[np.ndarray], Dict[Any, int], Dict[Any, int]]:
    if batch_labels is None:
        return None, {}, {}
    labels = np.asarray(list(batch_labels), dtype=object).ravel()
    if int(labels.size) != int(n_rows):
        raise ValueError(
            f"batch_labels has {labels.size} rows but expected {n_rows}."
        )
    codes = np.full(int(n_rows), -1, dtype=int)
    label_to_code: Dict[Any, int] = {}
    batch_sizes: Dict[Any, int] = {}
    for i, raw in enumerate(labels.tolist()):
        key = _batch_label_key(raw)
        if key not in label_to_code:
            label_to_code[key] = int(len(label_to_code))
            batch_sizes[key] = 0
        code = int(label_to_code[key])
        codes[i] = code
        batch_sizes[key] = int(batch_sizes.get(key, 0) + 1)
    return codes, label_to_code, batch_sizes


def _transform_encode_batch_labels(
    batch_labels: Optional[Sequence[Any]],
    n_rows: int,
    label_to_code: Dict[Any, int],
) -> Tuple[Optional[np.ndarray], int]:
    if batch_labels is None:
        return None, int(n_rows)
    labels = np.asarray(list(batch_labels), dtype=object).ravel()
    if int(labels.size) != int(n_rows):
        raise ValueError(
            f"batch_labels has {labels.size} rows but expected {n_rows}."
        )
    codes = np.full(int(n_rows), -1, dtype=int)
    unknown = 0
    for i, raw in enumerate(labels.tolist()):
        key = _batch_label_key(raw)
        if key in label_to_code:
            codes[i] = int(label_to_code[key])
        else:
            unknown += 1
    return codes, int(unknown)


def _fit_combat_model(
    X_train: np.ndarray,
    batch_codes: np.ndarray,
    *,
    prior_strength: float,
) -> Dict[str, Any]:
    arr = np.asarray(X_train, dtype=float)
    arr = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    n_features = int(arr.shape[1]) if arr.ndim == 2 else 0
    if arr.ndim != 2 or arr.shape[0] < 2 or n_features <= 0:
        raise ValueError("ComBat fit requires a 2D matrix with >=2 rows and >=1 feature.")

    codes = np.asarray(batch_codes, dtype=int).ravel()
    unique_codes = np.unique(codes[codes >= 0])
    if unique_codes.size < 2:
        raise ValueError("ComBat requires at least 2 distinct training batches.")

    global_mean = np.asarray(np.mean(arr, axis=0), dtype=float).ravel()
    pooled_var = np.asarray(np.var(arr, axis=0, ddof=1), dtype=float).ravel()
    pooled_var = np.maximum(np.nan_to_num(pooled_var, nan=1e-8, posinf=1e6, neginf=1e-8), 1e-8)
    pooled_std = np.sqrt(pooled_var)

    n_batches = int(unique_codes.size)
    gamma = np.zeros((n_batches, n_features), dtype=float)
    delta = np.ones((n_batches, n_features), dtype=float)
    batch_counts = np.zeros(n_batches, dtype=int)

    prior = float(max(0.0, prior_strength))
    for code in unique_codes.tolist():
        code_i = int(code)
        mask = codes == code_i
        n_k = int(np.sum(mask))
        if n_k <= 0:
            continue
        batch_counts[code_i] = n_k
        Xk = np.asarray(arr[mask], dtype=float)
        mu_k = np.asarray(np.mean(Xk, axis=0), dtype=float).ravel()
        if n_k > 1:
            var_k = np.asarray(np.var(Xk, axis=0, ddof=1), dtype=float).ravel()
            var_k = np.maximum(np.nan_to_num(var_k, nan=1e-8, posinf=1e6, neginf=1e-8), 1e-8)
        else:
            var_k = pooled_var.copy()

        # Empirical-Bayes-style shrinkage toward global location/scale.
        w = float(n_k / max(1.0, n_k + prior))
        gamma_hat = mu_k - global_mean
        delta_hat = np.sqrt(var_k) / np.maximum(1e-8, pooled_std)
        gamma[code_i] = w * gamma_hat
        delta[code_i] = 1.0 + w * (delta_hat - 1.0)

    delta = np.asarray(np.clip(delta, 0.20, 5.0), dtype=float)
    return {
        "global_mean": global_mean,
        "gamma": gamma,
        "delta": delta,
        "batch_counts": batch_counts,
        "prior_strength": float(prior),
    }


def _apply_combat_model(
    X: np.ndarray,
    batch_codes: np.ndarray,
    *,
    global_mean: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
) -> Tuple[np.ndarray, int]:
    arr = np.asarray(X, dtype=float)
    arr = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    out = arr.copy()
    codes = np.asarray(batch_codes, dtype=int).ravel()
    unknown = int(np.sum(codes < 0))
    if out.ndim != 2 or out.shape[0] != codes.size:
        raise ValueError("ComBat apply input shape mismatch.")

    for code in np.unique(codes[codes >= 0]).tolist():
        code_i = int(code)
        if code_i < 0 or code_i >= int(gamma.shape[0]):
            unknown += int(np.sum(codes == code_i))
            continue
        idx = np.where(codes == code_i)[0]
        if idx.size <= 0:
            continue
        out[idx] = (
            ((out[idx] - global_mean[None, :] - gamma[code_i][None, :]) / delta[code_i][None, :])
            + global_mean[None, :]
        )

    out = np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    return out, int(unknown)


def _fit_cdf_center_model(
    X_train: np.ndarray,
    batch_codes: np.ndarray,
    *,
    n_quantiles: int,
    clip_quantiles: Tuple[float, float],
) -> Dict[str, Any]:
    arr = np.asarray(X_train, dtype=float)
    arr = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] <= 0:
        raise ValueError("cdf_center fit requires a 2D matrix with >=2 rows and >=1 feature.")
    codes = np.asarray(batch_codes, dtype=int).ravel()
    unique_codes = np.unique(codes[codes >= 0])
    if unique_codes.size < 2:
        raise ValueError("cdf_center requires at least 2 distinct training batches.")

    n_q = int(max(7, n_quantiles))
    q_low = float(np.clip(clip_quantiles[0], 0.0, 0.49))
    q_high = float(np.clip(clip_quantiles[1], 0.51, 1.0))
    if q_high <= q_low + 1e-3:
        q_low, q_high = 0.01, 0.99
    q_grid = np.linspace(q_low, q_high, n_q, dtype=float)

    global_q = np.asarray(np.quantile(arr, q_grid, axis=0), dtype=float)
    batch_q: Dict[int, np.ndarray] = {}
    batch_counts: Dict[int, int] = {}
    for code in unique_codes.tolist():
        code_i = int(code)
        idx = np.where(codes == code_i)[0]
        if idx.size <= 0:
            continue
        batch_counts[code_i] = int(idx.size)
        batch_q[code_i] = np.asarray(np.quantile(arr[idx], q_grid, axis=0), dtype=float)

    return {
        "q_grid": q_grid,
        "global_quantiles": global_q,
        "batch_quantiles": batch_q,
        "batch_counts": batch_counts,
    }


def _monotone_quantile_values(qvals: np.ndarray) -> np.ndarray:
    arr = np.asarray(qvals, dtype=float).ravel()
    if arr.size <= 1:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum.accumulate(arr)


def _apply_cdf_map_vector(
    values: np.ndarray,
    *,
    src_quantiles: np.ndarray,
    dst_quantiles: np.ndarray,
    q_grid: np.ndarray,
) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    src = _monotone_quantile_values(src_quantiles)
    dst = _monotone_quantile_values(dst_quantiles)
    q = np.asarray(q_grid, dtype=float).ravel()
    if x.size == 0 or src.size != q.size or dst.size != q.size:
        return np.asarray(x, dtype=float)
    if float(np.ptp(src)) < 1e-12:
        # Degenerate source distribution: use location shift fallback.
        shift = float(np.mean(dst) - np.mean(src))
        return np.asarray(x + shift, dtype=float)
    ranks = np.interp(x, src, q, left=float(q[0]), right=float(q[-1]))
    mapped = np.interp(ranks, q, dst, left=float(dst[0]), right=float(dst[-1]))
    return np.asarray(mapped, dtype=float)


def _apply_cdf_center_model(
    X: np.ndarray,
    batch_codes: np.ndarray,
    *,
    q_grid: np.ndarray,
    global_quantiles: np.ndarray,
    batch_quantiles: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, int]:
    arr = np.asarray(X, dtype=float)
    arr = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    out = arr.copy()
    codes = np.asarray(batch_codes, dtype=int).ravel()
    if out.ndim != 2 or out.shape[0] != codes.size:
        raise ValueError("cdf_center apply input shape mismatch.")

    unknown = int(np.sum(codes < 0))
    n_features = int(out.shape[1])
    for code in np.unique(codes[codes >= 0]).tolist():
        code_i = int(code)
        idx = np.where(codes == code_i)[0]
        if idx.size <= 0:
            continue
        if code_i not in batch_quantiles:
            unknown += int(idx.size)
            continue
        src_q = np.asarray(batch_quantiles[code_i], dtype=float)
        if src_q.ndim != 2:
            unknown += int(idx.size)
            continue
        X_block = out[idx].copy()
        for j in range(n_features):
            X_block[:, j] = _apply_cdf_map_vector(
                X_block[:, j],
                src_quantiles=src_q[:, j],
                dst_quantiles=np.asarray(global_quantiles[:, j], dtype=float),
                q_grid=q_grid,
            )
        out[idx] = X_block

    out = np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    return out, int(unknown)


# ---------------------------------------------------------------------------
# 9b. ComBat-seq (count-preserving batch correction for RNA-seq count data)
# ---------------------------------------------------------------------------

def _fit_combat_seq_model(
    X_train: np.ndarray,
    batch_codes: np.ndarray,
    *,
    prior_strength: float,
) -> Dict[str, Any]:
    """Fit a count-preserving ComBat-seq model on training data.

    This is an adaptation of ComBat for count data (Johnson et al. 2007,
    Zhang et al. 2020 for ComBat-seq).  The key difference from standard
    ComBat is that corrections are applied as multiplicative log-space
    adjustments so that the output remains non-negative (preserving count
    semantics).  A small pseudocount is added before log-transform and
    removed after correction.
    """
    arr = np.asarray(X_train, dtype=float)
    arr = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    n_features = int(arr.shape[1]) if arr.ndim == 2 else 0
    if arr.ndim != 2 or arr.shape[0] < 2 or n_features <= 0:
        raise ValueError("ComBat-seq fit requires 2D matrix with >=2 rows and >=1 feature.")

    codes = np.asarray(batch_codes, dtype=int).ravel()
    unique_codes = np.unique(codes[codes >= 0])
    if unique_codes.size < 2:
        raise ValueError("ComBat-seq requires at least 2 distinct training batches.")

    # Work in log1p space to preserve count semantics.
    log_arr = np.log1p(np.clip(arr, 0.0, None))
    global_mean = np.asarray(np.mean(log_arr, axis=0), dtype=float).ravel()
    pooled_var = np.asarray(np.var(log_arr, axis=0, ddof=1), dtype=float).ravel()
    pooled_var = np.maximum(np.nan_to_num(pooled_var, nan=1e-8, posinf=1e6, neginf=1e-8), 1e-8)
    pooled_std = np.sqrt(pooled_var)

    n_batches = int(unique_codes.size)
    gamma = np.zeros((n_batches, n_features), dtype=float)
    delta = np.ones((n_batches, n_features), dtype=float)
    batch_counts = np.zeros(n_batches, dtype=int)

    prior = float(max(0.0, prior_strength))
    for code in unique_codes.tolist():
        code_i = int(code)
        mask = codes == code_i
        n_k = int(np.sum(mask))
        if n_k <= 0:
            continue
        batch_counts[code_i] = n_k
        Xk = np.asarray(log_arr[mask], dtype=float)
        mu_k = np.asarray(np.mean(Xk, axis=0), dtype=float).ravel()
        if n_k > 1:
            var_k = np.asarray(np.var(Xk, axis=0, ddof=1), dtype=float).ravel()
            var_k = np.maximum(np.nan_to_num(var_k, nan=1e-8, posinf=1e6, neginf=1e-8), 1e-8)
        else:
            var_k = pooled_var.copy()

        w = float(n_k / max(1.0, n_k + prior))
        gamma_hat = mu_k - global_mean
        delta_hat = np.sqrt(var_k) / np.maximum(1e-8, pooled_std)
        gamma[code_i] = w * gamma_hat
        delta[code_i] = 1.0 + w * (delta_hat - 1.0)

    delta = np.asarray(np.clip(delta, 0.20, 5.0), dtype=float)
    return {
        "global_mean": global_mean,
        "gamma": gamma,
        "delta": delta,
        "batch_counts": batch_counts,
        "prior_strength": float(prior),
    }


def _apply_combat_seq_model(
    X: np.ndarray,
    batch_codes: np.ndarray,
    *,
    global_mean: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Apply ComBat-seq correction, returning non-negative (count-like) values."""
    arr = np.asarray(X, dtype=float)
    arr = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    log_arr = np.log1p(np.clip(arr, 0.0, None))
    out = log_arr.copy()
    codes = np.asarray(batch_codes, dtype=int).ravel()
    unknown = int(np.sum(codes < 0))
    if out.ndim != 2 or out.shape[0] != codes.size:
        raise ValueError("ComBat-seq apply input shape mismatch.")

    for code in np.unique(codes[codes >= 0]).tolist():
        code_i = int(code)
        if code_i < 0 or code_i >= int(gamma.shape[0]):
            unknown += int(np.sum(codes == code_i))
            continue
        idx = np.where(codes == code_i)[0]
        if idx.size <= 0:
            continue
        out[idx] = (
            ((out[idx] - global_mean[None, :] - gamma[code_i][None, :]) / delta[code_i][None, :])
            + global_mean[None, :]
        )

    # Invert log1p to recover count-like values (non-negative).
    result = np.expm1(np.clip(out, 0.0, 30.0))  # clip to avoid overflow
    result = np.asarray(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    result = np.clip(result, 0.0, None)  # enforce non-negativity
    return result, int(unknown)


# ---------------------------------------------------------------------------
# 9c. Center-scale batch correction (simple per-batch location/scale)
# ---------------------------------------------------------------------------

def _fit_center_scale_model(
    X_train: np.ndarray,
    batch_codes: np.ndarray,
) -> Dict[str, Any]:
    """Fit per-batch centering and scaling (no shrinkage)."""
    arr = np.asarray(X_train, dtype=float)
    arr = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    n_features = int(arr.shape[1]) if arr.ndim == 2 else 0
    if arr.ndim != 2 or arr.shape[0] < 2 or n_features <= 0:
        raise ValueError("center_scale fit requires 2D matrix with >=2 rows and >=1 feature.")

    codes = np.asarray(batch_codes, dtype=int).ravel()
    unique_codes = np.unique(codes[codes >= 0])
    if unique_codes.size < 2:
        raise ValueError("center_scale requires at least 2 distinct training batches.")

    global_mean = np.asarray(np.mean(arr, axis=0), dtype=float).ravel()
    global_std = np.asarray(np.std(arr, axis=0, ddof=1), dtype=float).ravel()
    global_std = np.maximum(np.nan_to_num(global_std, nan=1.0, posinf=1.0, neginf=1.0), 1e-8)

    n_batches = int(unique_codes.size)
    batch_means = np.zeros((n_batches, n_features), dtype=float)
    batch_stds = np.ones((n_batches, n_features), dtype=float)
    batch_counts = np.zeros(n_batches, dtype=int)

    for code in unique_codes.tolist():
        code_i = int(code)
        mask = codes == code_i
        n_k = int(np.sum(mask))
        if n_k <= 0:
            continue
        batch_counts[code_i] = n_k
        Xk = np.asarray(arr[mask], dtype=float)
        batch_means[code_i] = np.asarray(np.mean(Xk, axis=0), dtype=float).ravel()
        if n_k > 1:
            s = np.asarray(np.std(Xk, axis=0, ddof=1), dtype=float).ravel()
            batch_stds[code_i] = np.maximum(np.nan_to_num(s, nan=1.0), 1e-8)
        else:
            batch_stds[code_i] = global_std.copy()

    return {
        "global_mean": global_mean,
        "global_std": global_std,
        "batch_means": batch_means,
        "batch_stds": batch_stds,
        "batch_counts": batch_counts,
    }


def _apply_center_scale_model(
    X: np.ndarray,
    batch_codes: np.ndarray,
    *,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    batch_means: np.ndarray,
    batch_stds: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Apply per-batch center-scale correction: z-score per batch, then rescale to global."""
    arr = np.asarray(X, dtype=float)
    arr = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    out = arr.copy()
    codes = np.asarray(batch_codes, dtype=int).ravel()
    unknown = int(np.sum(codes < 0))
    if out.ndim != 2 or out.shape[0] != codes.size:
        raise ValueError("center_scale apply input shape mismatch.")

    for code in np.unique(codes[codes >= 0]).tolist():
        code_i = int(code)
        if code_i < 0 or code_i >= int(batch_means.shape[0]):
            unknown += int(np.sum(codes == code_i))
            continue
        idx = np.where(codes == code_i)[0]
        if idx.size <= 0:
            continue
        # z-score with batch parameters, then rescale to global
        out[idx] = (
            ((out[idx] - batch_means[code_i][None, :]) / batch_stds[code_i][None, :])
            * global_std[None, :]
            + global_mean[None, :]
        )

    out = np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    return out, int(unknown)


def fit_batch_correction_model(
    X_train: np.ndarray,
    *,
    batch_labels: Optional[Sequence[Any]],
    mode: str = "none",
    combat_prior_strength: float = 8.0,
    cdf_center_n_quantiles: int = 33,
    cdf_center_clip_quantiles: Tuple[float, float] = (0.01, 0.99),
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fit a train-fold-only batch-correction model.

    Supported modes:
    - ``none``: no correction
    - ``combat``: empirical-Bayes location/scale shrinkage
    - ``combat_seq``: count-preserving ComBat in log1p space (RNA-seq)
    - ``cdf_center``: per-center quantile mapping to pooled reference
    - ``center_scale``: simple per-batch centering and scaling
    """
    arr = np.asarray(X_train, dtype=float)
    n_rows = int(arr.shape[0]) if arr.ndim == 2 else 0
    mode_req = _canonicalize_batch_correction_mode(mode)

    meta: Dict[str, Any] = {
        "batch_correction_mode_requested": str(mode_req),
        "batch_correction_mode_applied": "none",
        "batch_correction_applied": False,
        "batch_correction_fit_reason": "disabled",
        "batch_correction_n_batches": 0,
        "batch_correction_batch_sizes": {},
    }
    model: Dict[str, Any] = {
        "mode": "none",
        "n_features": int(arr.shape[1]) if arr.ndim == 2 else 0,
        "label_to_code": {},
    }
    if mode_req == "none":
        meta["batch_correction_fit_reason"] = "mode_none"
        return model, meta

    batch_codes, label_to_code, batch_sizes = _fit_encode_batch_labels(batch_labels, n_rows)
    if batch_codes is None:
        meta["batch_correction_fit_reason"] = "missing_batch_labels"
        return model, meta
    if len(label_to_code) < 2:
        meta["batch_correction_fit_reason"] = "single_batch"
        meta["batch_correction_n_batches"] = int(len(label_to_code))
        meta["batch_correction_batch_sizes"] = {
            str(k): int(v) for k, v in batch_sizes.items()
        }
        return model, meta

    try:
        if mode_req == "combat":
            payload = _fit_combat_model(
                arr,
                batch_codes,
                prior_strength=float(max(0.0, combat_prior_strength)),
            )
        elif mode_req == "combat_seq":
            payload = _fit_combat_seq_model(
                arr,
                batch_codes,
                prior_strength=float(max(0.0, combat_prior_strength)),
            )
        elif mode_req == "cdf_center":
            payload = _fit_cdf_center_model(
                arr,
                batch_codes,
                n_quantiles=int(max(7, cdf_center_n_quantiles)),
                clip_quantiles=(
                    float(cdf_center_clip_quantiles[0]),
                    float(cdf_center_clip_quantiles[1]),
                ),
            )
        elif mode_req == "center_scale":
            payload = _fit_center_scale_model(arr, batch_codes)
        else:
            payload = {}
    except Exception as exc:
        meta["batch_correction_fit_reason"] = f"fit_failed:{type(exc).__name__}"
        return model, meta

    model = {
        "mode": str(mode_req),
        "n_features": int(arr.shape[1]),
        "label_to_code": dict(label_to_code),
        "batch_sizes": {str(k): int(v) for k, v in batch_sizes.items()},
        "payload": payload,
    }
    meta.update(
        {
            "batch_correction_mode_applied": str(mode_req),
            "batch_correction_applied": True,
            "batch_correction_fit_reason": "ok",
            "batch_correction_n_batches": int(len(label_to_code)),
            "batch_correction_batch_sizes": dict(model["batch_sizes"]),
        }
    )
    return model, meta


def apply_batch_correction_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    model: Optional[Dict[str, Any]],
    batch_labels_train: Optional[Sequence[Any]],
    batch_labels_test: Optional[Sequence[Any]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply a previously fitted batch-correction model to train/test arrays."""
    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    mode = _canonicalize_batch_correction_mode(
        (model or {}).get("mode", "none")
    )
    meta: Dict[str, Any] = {
        "batch_correction_apply_reason": "mode_none",
        "batch_correction_train_rows_corrected": 0,
        "batch_correction_test_rows_corrected": 0,
        "batch_correction_unknown_train_batches": int(Xtr.shape[0]) if mode != "none" else 0,
        "batch_correction_unknown_test_batches": int(Xte.shape[0]) if mode != "none" else 0,
    }
    if mode == "none":
        return np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float), meta
    if not isinstance(model, dict):
        meta["batch_correction_apply_reason"] = "missing_model"
        return np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float), meta

    label_to_code = dict(model.get("label_to_code") or {})
    train_codes, train_unknown = _transform_encode_batch_labels(
        batch_labels_train,
        int(Xtr.shape[0]),
        label_to_code,
    )
    test_codes, test_unknown = _transform_encode_batch_labels(
        batch_labels_test,
        int(Xte.shape[0]),
        label_to_code,
    )
    if train_codes is None:
        meta["batch_correction_apply_reason"] = "missing_train_labels"
        return np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float), meta
    if test_codes is None:
        meta["batch_correction_apply_reason"] = "missing_test_labels"
        return np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float), meta

    payload = dict(model.get("payload") or {})
    try:
        if mode == "combat":
            Xtr_corr, train_unknown_apply = _apply_combat_model(
                Xtr,
                train_codes,
                global_mean=np.asarray(payload.get("global_mean"), dtype=float),
                gamma=np.asarray(payload.get("gamma"), dtype=float),
                delta=np.asarray(payload.get("delta"), dtype=float),
            )
            Xte_corr, test_unknown_apply = _apply_combat_model(
                Xte,
                test_codes,
                global_mean=np.asarray(payload.get("global_mean"), dtype=float),
                gamma=np.asarray(payload.get("gamma"), dtype=float),
                delta=np.asarray(payload.get("delta"), dtype=float),
            )
        elif mode == "combat_seq":
            Xtr_corr, train_unknown_apply = _apply_combat_seq_model(
                Xtr,
                train_codes,
                global_mean=np.asarray(payload.get("global_mean"), dtype=float),
                gamma=np.asarray(payload.get("gamma"), dtype=float),
                delta=np.asarray(payload.get("delta"), dtype=float),
            )
            Xte_corr, test_unknown_apply = _apply_combat_seq_model(
                Xte,
                test_codes,
                global_mean=np.asarray(payload.get("global_mean"), dtype=float),
                gamma=np.asarray(payload.get("gamma"), dtype=float),
                delta=np.asarray(payload.get("delta"), dtype=float),
            )
        elif mode == "cdf_center":
            Xtr_corr, train_unknown_apply = _apply_cdf_center_model(
                Xtr,
                train_codes,
                q_grid=np.asarray(payload.get("q_grid"), dtype=float),
                global_quantiles=np.asarray(payload.get("global_quantiles"), dtype=float),
                batch_quantiles=dict(payload.get("batch_quantiles") or {}),
            )
            Xte_corr, test_unknown_apply = _apply_cdf_center_model(
                Xte,
                test_codes,
                q_grid=np.asarray(payload.get("q_grid"), dtype=float),
                global_quantiles=np.asarray(payload.get("global_quantiles"), dtype=float),
                batch_quantiles=dict(payload.get("batch_quantiles") or {}),
            )
        elif mode == "center_scale":
            Xtr_corr, train_unknown_apply = _apply_center_scale_model(
                Xtr,
                train_codes,
                global_mean=np.asarray(payload.get("global_mean"), dtype=float),
                global_std=np.asarray(payload.get("global_std"), dtype=float),
                batch_means=np.asarray(payload.get("batch_means"), dtype=float),
                batch_stds=np.asarray(payload.get("batch_stds"), dtype=float),
            )
            Xte_corr, test_unknown_apply = _apply_center_scale_model(
                Xte,
                test_codes,
                global_mean=np.asarray(payload.get("global_mean"), dtype=float),
                global_std=np.asarray(payload.get("global_std"), dtype=float),
                batch_means=np.asarray(payload.get("batch_means"), dtype=float),
                batch_stds=np.asarray(payload.get("batch_stds"), dtype=float),
            )
        else:
            Xtr_corr, Xte_corr = np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float)
            train_unknown_apply, test_unknown_apply = int(Xtr.shape[0]), int(Xte.shape[0])
    except Exception as exc:
        meta["batch_correction_apply_reason"] = f"apply_failed:{type(exc).__name__}"
        return np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float), meta

    train_unknown_total = int(max(train_unknown, train_unknown_apply))
    test_unknown_total = int(max(test_unknown, test_unknown_apply))
    meta.update(
        {
            "batch_correction_apply_reason": "ok",
            "batch_correction_train_rows_corrected": int(max(0, Xtr.shape[0] - train_unknown_total)),
            "batch_correction_test_rows_corrected": int(max(0, Xte.shape[0] - test_unknown_total)),
            "batch_correction_unknown_train_batches": int(train_unknown_total),
            "batch_correction_unknown_test_batches": int(test_unknown_total),
        }
    )
    return (
        np.asarray(np.nan_to_num(Xtr_corr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float),
        np.asarray(np.nan_to_num(Xte_corr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float),
        meta,
    )

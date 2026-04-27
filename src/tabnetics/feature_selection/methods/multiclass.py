"""Multiclass-aware feature selection methods."""
import math
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import f_classif, mutual_info_classif

from tabnetics.core.runtime import get_sklearn_n_jobs as _get_sklearn_n_jobs
try:
    from tabnetics.core.compat import make_logistic_regression
except Exception as exc:
    from tabnetics.core.compat import make_logistic_regression  # type: ignore
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..cv import safe_balanced_accuracy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def label_to_key(label):
    """Convert a class label to a hashable dictionary key."""
    if isinstance(label, (np.integer, int)):
        return int(label)
    return str(label)


# ---------------------------------------------------------------------------
# OVA helpers
# ---------------------------------------------------------------------------

def estimate_ova_calibration_reliability(X_sub, y_sub, *, ova_calibration_cv,
                                         ova_linear_backend, random_state,
                                         linear_svm_max_iter):
    """
    Estimate class-conditional OVA calibration reliability in [0, 1].
    Uses a calibrated linear model and Brier-style reliability scaling.
    """
    y_bin = np.asarray(y_sub, dtype=int).ravel()
    if y_bin.size == 0:
        return 1.0
    n_pos = int(np.sum(y_bin == 1))
    n_neg = int(np.sum(y_bin == 0))
    if min(n_pos, n_neg) < 2:
        return 1.0

    cv = int(max(2, int(ova_calibration_cv)))
    cv = int(min(cv, n_pos, n_neg))
    if cv < 2:
        return 1.0

    if str(ova_linear_backend) == "elastic_net_lr":
        base_estimator = make_logistic_regression(
            random_state=random_state,
            max_iter=5000,
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5,
            C=0.4,
            class_weight="balanced",
        )
        X_fit = StandardScaler().fit_transform(np.asarray(X_sub, dtype=float))
    else:
        base_estimator = LinearSVC(
            penalty="l1",
            dual=False,
            random_state=random_state,
            max_iter=linear_svm_max_iter,
            class_weight="balanced",
        )
        X_fit = np.asarray(X_sub, dtype=float)

    try:
        calibrator = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=cv,
        )
        calibrator.fit(X_fit, y_bin)
        probs = calibrator.predict_proba(X_fit)
        if probs.ndim != 2 or probs.shape[1] < 2:
            return 1.0
        p_pos = np.asarray(probs[:, 1], dtype=float).ravel()
        p_pos = np.nan_to_num(p_pos, nan=0.5, posinf=1.0, neginf=0.0)
        p_pos = np.clip(p_pos, 1e-6, 1.0 - 1e-6)
        brier = float(np.mean((p_pos - y_bin) ** 2))
        # Binary Brier score ranges in [0, 0.25] for calibrated probabilities.
        reliability = 1.0 - min(1.0, brier / 0.25)
        return float(np.clip(reliability, 0.0, 1.0))
    except Exception as exc:
        return 1.0


# ---------------------------------------------------------------------------
# Per-class quota overlay
# ---------------------------------------------------------------------------

def apply_per_class_quota_overlay(
    selected_indices: np.ndarray,
    ranked_indices: np.ndarray,
    class_rankings: Dict[Any, List[int]],
    n_target_features: int,
    *,
    per_class_quota_enabled: bool,
    per_class_quota_min_per_class: int,
    per_class_quota_max_fraction: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Optional per-class quota overlay for class-specific selectors.
    Injects class-ranked features when class coverage is under quota.
    """
    selected = np.asarray(selected_indices, dtype=int).ravel()
    ranked = np.asarray(ranked_indices, dtype=int).ravel()
    n_target = int(max(1, n_target_features))

    meta: Dict[str, Any] = {
        "enabled": bool(per_class_quota_enabled),
        "applied": False,
        "quota_min_per_class": int(per_class_quota_min_per_class),
        "quota_max_fraction": float(per_class_quota_max_fraction),
        "quota_budget": 0,
        "forced_additions": [],
        "forced_by_class": {},
        "class_hits_before": {},
        "class_hits_after": {},
        "selected_count_before": int(selected.size),
        "selected_count_after": int(selected.size),
    }

    if not bool(per_class_quota_enabled):
        return selected, meta
    if n_target <= 0 or ranked.size == 0 or not class_rankings:
        return selected, meta

    ranked_unique: List[int] = []
    ranked_seen = set()
    for idx in ranked.tolist():
        idx_i = int(idx)
        if idx_i < 0 or idx_i in ranked_seen:
            continue
        ranked_seen.add(idx_i)
        ranked_unique.append(idx_i)
    if not ranked_unique:
        return selected, meta

    selected_set = set(int(i) for i in selected.tolist() if int(i) in ranked_seen)
    if not selected_set:
        selected_set = set(ranked_unique[:n_target])

    cleaned_rankings: Dict[Any, List[int]] = {}
    for cls_key, ranking in class_rankings.items():
        clean: List[int] = []
        seen_cls = set()
        for idx in ranking:
            idx_i = int(idx)
            if idx_i in ranked_seen and idx_i not in seen_cls:
                clean.append(idx_i)
                seen_cls.add(idx_i)
        if clean:
            cleaned_rankings[cls_key] = clean
    if not cleaned_rankings:
        return np.asarray([idx for idx in ranked_unique if idx in selected_set][:n_target], dtype=int), meta

    quota_budget = int(
        min(
            n_target,
            max(0, int(np.floor(float(per_class_quota_max_fraction) * float(n_target)))),
        )
    )
    meta["quota_budget"] = int(quota_budget)
    if quota_budget <= 0:
        trimmed = [idx for idx in ranked_unique if idx in selected_set][:n_target]
        return np.asarray(trimmed, dtype=int), meta

    forced_by_class: Dict[Any, int] = {}
    forced_additions: List[int] = []
    remaining_budget = int(quota_budget)
    min_per_class = int(max(1, per_class_quota_min_per_class))

    for cls_key in cleaned_rankings.keys():
        ranking = cleaned_rankings[cls_key]
        hits_before = int(sum(1 for idx in ranking[:n_target] if idx in selected_set))
        meta["class_hits_before"][cls_key] = int(hits_before)

        need = int(max(0, min_per_class - hits_before))
        added = 0
        while need > 0 and remaining_budget > 0:
            next_idx = None
            for idx in ranking:
                if idx not in selected_set:
                    next_idx = int(idx)
                    break
            if next_idx is None:
                break
            selected_set.add(next_idx)
            forced_additions.append(next_idx)
            added += 1
            need -= 1
            remaining_budget -= 1
        forced_by_class[cls_key] = int(added)

    final_ranked = [idx for idx in ranked_unique if idx in selected_set]
    if len(final_ranked) < n_target:
        for idx in ranked_unique:
            if idx not in selected_set:
                final_ranked.append(idx)
            if len(final_ranked) >= n_target:
                break
    if len(final_ranked) > n_target:
        forced_set = set(int(i) for i in forced_additions)
        keep: List[int] = []
        for idx in final_ranked:
            if idx in forced_set and idx not in keep:
                keep.append(int(idx))
        for idx in final_ranked:
            if idx in keep:
                continue
            keep.append(int(idx))
            if len(keep) >= n_target:
                break
        final_ranked = keep[:n_target]

    selected_out = np.asarray(final_ranked[:n_target], dtype=int)
    selected_out_set = set(int(i) for i in selected_out.tolist())
    for cls_key, ranking in cleaned_rankings.items():
        meta["class_hits_after"][cls_key] = int(
            sum(1 for idx in ranking[:n_target] if idx in selected_out_set)
        )

    meta.update(
        {
            "applied": bool(len(forced_additions) > 0),
            "forced_additions": [int(i) for i in forced_additions],
            "forced_by_class": dict(forced_by_class),
            "selected_count_after": int(selected_out.size),
        }
    )
    return selected_out, meta


# ---------------------------------------------------------------------------
# ECOC helpers
# ---------------------------------------------------------------------------

def ecoc_binary_relevance_scores(X_sub, y_sub, n_features, *, random_state,
                                  ova_linear_backend, linear_svm_max_iter,
                                  normalize_fn,
                                  score_cache=None,
                                  cache_context_key=None):
    """Binary relevance scores for ECOC dichotomy tasks."""
    if isinstance(score_cache, dict) and cache_context_key is not None:
        cached = score_cache.get(cache_context_key)
        if cached is not None:
            return np.asarray(cached, dtype=float).copy()

    try:
        mi_scores = np.asarray(mutual_info_classif(X_sub, y_sub, random_state=random_state), dtype=float).ravel()
    except Exception as exc:
        mi_scores = np.zeros(n_features, dtype=float)
    mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        f_scores, _ = f_classif(X_sub, y_sub)
        f_scores = np.asarray(f_scores, dtype=float).ravel()
    except Exception as exc:
        f_scores = np.zeros(n_features, dtype=float)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

    linear_scores = np.zeros(n_features, dtype=float)
    if ova_linear_backend == "elastic_net_lr":
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sub)
            clf = make_logistic_regression(
                random_state=random_state,
                max_iter=5000,
                solver="saga",
                penalty="elasticnet",
                l1_ratio=0.5,
                C=0.4,
                class_weight="balanced",
            )
            clf.fit(X_scaled, y_sub)
            coef = np.asarray(clf.coef_, dtype=float)
            if coef.ndim == 2:
                linear_scores = np.mean(np.abs(coef), axis=0)
            else:
                linear_scores = np.abs(coef)
        except Exception as exc:
            pass
    else:
        try:
            svm = LinearSVC(
                penalty="l1",
                dual=False,
                random_state=random_state,
                max_iter=linear_svm_max_iter,
                class_weight="balanced",
            )
            svm.fit(X_sub, y_sub)
            coef = np.asarray(svm.coef_, dtype=float)
            if coef.ndim == 2:
                linear_scores = np.mean(np.abs(coef), axis=0)
            else:
                linear_scores = np.abs(coef)
        except Exception as exc:
            pass

    linear_scores = np.nan_to_num(np.asarray(linear_scores, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    if linear_scores.size != n_features:
        padded = np.zeros(n_features, dtype=float)
        upto = int(min(n_features, linear_scores.size))
        if upto > 0:
            padded[:upto] = linear_scores[:upto]
        linear_scores = padded

    combined = (
        0.45 * normalize_fn(mi_scores)
        + 0.35 * normalize_fn(f_scores)
        + 0.20 * normalize_fn(linear_scores)
    )
    combined = np.asarray(combined, dtype=float).ravel()
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    if isinstance(score_cache, dict) and cache_context_key is not None:
        score_cache[cache_context_key] = np.asarray(combined, dtype=float).copy()
    return combined


def ecoc_class_complexity_weights(X, y, classes, *, ecoc_class_complexity_weight,
                                   normalize_fn):
    """Compute per-class complexity scores and weights."""
    n_features = int(X.shape[1])
    class_to_idx = {cls: np.where(y == cls)[0] for cls in classes}
    centroids = {}
    within_disp = {}
    for cls in classes:
        idx = class_to_idx[cls]
        if idx.size == 0:
            centroids[cls] = np.zeros(n_features, dtype=float)
            within_disp[cls] = 0.0
            continue
        X_cls = X[idx]
        mu = np.asarray(np.mean(X_cls, axis=0), dtype=float).ravel()
        centroids[cls] = mu
        diffs = X_cls - mu[None, :]
        within_disp[cls] = float(np.mean(np.sum(diffs * diffs, axis=1))) if diffs.size else 0.0

    nearest_dist = {cls: float("inf") for cls in classes}
    for i, cls_i in enumerate(classes):
        for j in range(i + 1, len(classes)):
            cls_j = classes[j]
            dist = float(np.linalg.norm(centroids[cls_i] - centroids[cls_j]))
            if dist < nearest_dist[cls_i]:
                nearest_dist[cls_i] = dist
            if dist < nearest_dist[cls_j]:
                nearest_dist[cls_j] = dist

    complexity_raw = []
    for cls in classes:
        d = nearest_dist.get(cls, float("inf"))
        if not np.isfinite(d):
            d = 0.0
        complexity_raw.append(float(within_disp.get(cls, 0.0)) / max(1e-8, d))
    complexity_raw = np.asarray(complexity_raw, dtype=float).ravel()
    complexity_norm = normalize_fn(complexity_raw)

    out_complexity = {}
    out_weights = {}
    for cls, comp in zip(classes, complexity_norm):
        key = label_to_key(cls)
        out_complexity[key] = float(comp)
        out_weights[key] = float(1.0 + ecoc_class_complexity_weight * float(comp))
    return out_complexity, out_weights


# ---------------------------------------------------------------------------
# OVA ensemble selector
# ---------------------------------------------------------------------------

def ova_ensemble_selection(X, y, n_target_features, *, problem_type, random_state,
                           ova_min_classes, ova_min_pos_samples, ova_negative_ratio,
                           ova_linear_backend, linear_svm_max_iter,
                           ova_enable_calibration, ova_calibration_cv,
                           ova_class_weight_mode, ova_aggregation_mode,
                           ova_aggregation_p, normalize_fn):
    """Multiclass one-vs-all ensemble selector."""
    if problem_type != "classification" or X.shape[1] == 0:
        return {}, {}

    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    if classes.size < ova_min_classes:
        return {}, {}

    n_features = int(X.shape[1])
    rng = np.random.default_rng(random_state)
    per_class_target = int(max(1, np.ceil(n_target_features / max(1, classes.size))))

    aggregated_scores = np.zeros(n_features, dtype=float)
    class_selected: Dict[Any, List[int]] = {}
    class_weights: Dict[Any, float] = {}
    class_base_weights: Dict[Any, float] = {}
    class_calibration_reliability: Dict[Any, float] = {}
    classes_used = 0
    weight_sum = 0.0
    binary_task_score_cache: Dict[Any, np.ndarray] = {}

    for cls in classes:
        y_bin = (y_arr == cls).astype(int)
        pos_idx = np.where(y_bin == 1)[0]
        neg_idx = np.where(y_bin == 0)[0]
        if pos_idx.size < ova_min_pos_samples or neg_idx.size < 2:
            continue

        max_neg = int(max(pos_idx.size, round(pos_idx.size * ova_negative_ratio)))
        if neg_idx.size > max_neg:
            neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)

        subset_idx = np.concatenate([pos_idx, neg_idx])
        rng.shuffle(subset_idx)
        X_sub = X[subset_idx]
        y_sub = y_bin[subset_idx]
        if np.unique(y_sub).size < 2:
            continue

        cache_key = (
            "ova",
            label_to_key(cls),
            int(X_sub.shape[0]),
            int(np.sum(y_sub == 1)),
            int(np.sum(y_sub == 0)),
            str(ova_linear_backend),
            int(linear_svm_max_iter),
            int(n_features),
        )
        combined = ecoc_binary_relevance_scores(
            X_sub,
            y_sub,
            n_features=n_features,
            random_state=random_state,
            ova_linear_backend=ova_linear_backend,
            linear_svm_max_iter=linear_svm_max_iter,
            normalize_fn=normalize_fn,
            score_cache=binary_task_score_cache,
            cache_context_key=cache_key,
        )

        top_idx = np.argsort(combined)[::-1][:per_class_target]
        cls_key = label_to_key(cls)
        class_selected[cls_key] = [int(i) for i in np.asarray(top_idx, dtype=int).tolist()]

        cal_reliability = 1.0
        if bool(ova_enable_calibration):
            cal_reliability = float(estimate_ova_calibration_reliability(
                X_sub, y_sub,
                ova_calibration_cv=ova_calibration_cv,
                ova_linear_backend=ova_linear_backend,
                random_state=random_state,
                linear_svm_max_iter=linear_svm_max_iter,
            ))
            if not np.isfinite(cal_reliability):
                cal_reliability = 1.0
            cal_reliability = float(np.clip(cal_reliability, 0.15, 1.0))
        class_calibration_reliability[cls_key] = float(cal_reliability)

        if ova_class_weight_mode == "sqrt_pos":
            cls_weight = float(math.sqrt(float(pos_idx.size)))
        elif ova_class_weight_mode == "pos":
            cls_weight = float(pos_idx.size)
        elif ova_class_weight_mode == "log_pos":
            cls_weight = float(math.log1p(float(pos_idx.size)))
        elif ova_class_weight_mode == "inv_sqrt_pos":
            cls_weight = float(1.0 / max(1e-12, math.sqrt(float(pos_idx.size))))
        elif ova_class_weight_mode == "inv_pos":
            cls_weight = float(1.0 / max(1e-12, float(pos_idx.size)))
        elif ova_class_weight_mode == "inv_log_pos":
            cls_weight = float(1.0 / max(1e-12, math.log1p(float(pos_idx.size))))
        else:
            cls_weight = 1.0

        cls_weight = float(max(0.0, cls_weight))
        class_base_weights[cls_key] = float(cls_weight)
        effective_weight = float(cls_weight * cal_reliability)
        if ova_aggregation_mode == "p_norm":
            p = float(ova_aggregation_p)
            if not np.isfinite(p) or p <= 0.0:
                p = 1.0
            aggregated_scores += effective_weight * np.power(combined, p)
        else:
            aggregated_scores += effective_weight * combined
        class_weights[cls_key] = float(effective_weight)
        classes_used += 1
        weight_sum += effective_weight

    if classes_used <= 0 or weight_sum <= 0.0:
        return {}, {}

    if ova_aggregation_mode == "p_norm":
        p = float(ova_aggregation_p)
        if not np.isfinite(p) or p <= 0.0:
            aggregated_scores = aggregated_scores / float(weight_sum)
        else:
            aggregated_scores = np.power(aggregated_scores / float(weight_sum), 1.0 / p)
    else:
        aggregated_scores = aggregated_scores / float(weight_sum)
    aggregated_scores = normalize_fn(aggregated_scores)
    selected_indices = np.argsort(aggregated_scores)[::-1][: int(min(n_features, max(1, n_target_features)))]

    results = {
        "selected_indices": np.asarray(selected_indices, dtype=int),
        "scores": {int(idx): float(aggregated_scores[int(idx)]) for idx in np.asarray(selected_indices, dtype=int).tolist()},
        "method": "ova_ensemble",
        "ova_classes_total": int(classes.size),
        "ova_classes_used": int(classes_used),
        "ova_negative_ratio": float(ova_negative_ratio),
        "ova_min_classes": int(ova_min_classes),
        "ova_min_pos_samples": int(ova_min_pos_samples),
        "ova_class_weight_mode": str(ova_class_weight_mode),
        "ova_aggregation_mode": str(ova_aggregation_mode),
        "ova_aggregation_p": float(ova_aggregation_p),
        "ova_class_weights": class_weights,
        "ova_class_base_weights": class_base_weights,
        "ova_linear_backend": str(ova_linear_backend),
        "ova_enable_calibration": bool(ova_enable_calibration),
        "ova_calibration_cv": int(ova_calibration_cv),
        "ova_class_calibration_reliability": class_calibration_reliability,
        "ova_per_class_target": int(per_class_target),
        "ova_class_selected_indices": class_selected,
    }
    all_scores = {i: float(aggregated_scores[i]) for i in range(n_features)}
    return results, all_scores


# ---------------------------------------------------------------------------
# ECOC class-aware selector
# ---------------------------------------------------------------------------

def ecoc_class_aware_selection(X, y, n_target_features, *, problem_type,
                               random_state, ecoc_min_classes,
                               ecoc_class_complexity_weight,
                               ecoc_include_ova_tasks, ecoc_negative_ratio,
                               ecoc_max_ovo_pairs, ecoc_random_code_bits,
                               ova_linear_backend, linear_svm_max_iter,
                               normalize_fn):
    """
    ECOC-aware multiclass selector:
    combines OVA, confusable OVO pairs, and random ECOC dichotomies with
    class-complexity-aware task weighting.
    """
    if problem_type != "classification" or X.shape[1] == 0:
        return {}, {}

    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    if classes.size < ecoc_min_classes:
        return {}, {}

    n_features = int(X.shape[1])
    rng = np.random.default_rng(random_state + 211)
    class_complexity, class_weights = ecoc_class_complexity_weights(
        X, y_arr, classes.tolist(),
        ecoc_class_complexity_weight=ecoc_class_complexity_weight,
        normalize_fn=normalize_fn,
    )

    aggregated_scores = np.zeros(n_features, dtype=float)
    tasks_used = 0
    tasks_metadata = []
    total_weight = 0.0
    binary_task_score_cache: Dict[Any, np.ndarray] = {}

    def _partition_key(pos_classes, neg_classes):
        pos_key = tuple(sorted(label_to_key(c) for c in pos_classes))
        neg_key = tuple(sorted(label_to_key(c) for c in neg_classes))
        return min((pos_key, neg_key), (neg_key, pos_key))

    def _task_weight(pos_classes, neg_classes):
        pos_w = [float(class_weights.get(label_to_key(cls), 1.0)) for cls in pos_classes]
        neg_w = [float(class_weights.get(label_to_key(cls), 1.0)) for cls in neg_classes]
        all_w = pos_w + neg_w
        return float(np.mean(all_w)) if all_w else 1.0

    def _run_task(name, pos_classes, neg_classes, pairwise_subset):
        nonlocal tasks_used, total_weight
        pos_set = set(pos_classes)
        neg_set = set(neg_classes)
        if not pos_set or not neg_set:
            return

        if pairwise_subset:
            keep_mask = np.isin(y_arr, list(pos_set | neg_set))
            subset_idx = np.where(keep_mask)[0]
        else:
            subset_idx = np.arange(y_arr.size, dtype=int)

        if subset_idx.size < 4:
            return

        y_bin = np.isin(y_arr[subset_idx], list(pos_set)).astype(int)
        pos_idx = np.where(y_bin == 1)[0]
        neg_idx = np.where(y_bin == 0)[0]
        if pos_idx.size < 2 or neg_idx.size < 2:
            return

        if not pairwise_subset:
            max_neg = int(max(pos_idx.size, round(pos_idx.size * ecoc_negative_ratio)))
            if neg_idx.size > max_neg:
                neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
            keep_local = np.concatenate([pos_idx, neg_idx])
            rng.shuffle(keep_local)
            subset_idx = subset_idx[keep_local]
            y_bin = y_bin[keep_local]
            pos_idx = np.where(y_bin == 1)[0]
            neg_idx = np.where(y_bin == 0)[0]
            if pos_idx.size < 2 or neg_idx.size < 2:
                return

        X_sub = X[subset_idx]
        combined = ecoc_binary_relevance_scores(
            X_sub, y_bin, n_features=n_features,
            random_state=random_state,
            ova_linear_backend=ova_linear_backend,
            linear_svm_max_iter=linear_svm_max_iter,
            normalize_fn=normalize_fn,
            score_cache=binary_task_score_cache,
            cache_context_key=(
                "ecoc",
                str(name),
                int(X_sub.shape[0]),
                int(np.sum(y_bin == 1)),
                int(np.sum(y_bin == 0)),
                str(ova_linear_backend),
                int(linear_svm_max_iter),
                int(n_features),
            ),
        )
        if not np.any(np.isfinite(combined)):
            return

        w = float(max(1e-8, _task_weight(pos_classes, neg_classes)))
        aggregated_scores[:] = aggregated_scores + w * combined
        total_weight += w
        tasks_used += 1
        tasks_metadata.append(
            {
                "task_name": str(name),
                "task_weight": float(w),
                "n_pos": int(pos_idx.size),
                "n_neg": int(neg_idx.size),
                "pairwise_subset": bool(pairwise_subset),
                "pos_classes": [label_to_key(c) for c in pos_classes],
                "neg_classes": [label_to_key(c) for c in neg_classes],
            }
        )

    # OVA tasks.
    if ecoc_include_ova_tasks:
        for cls in classes.tolist():
            neg = [c for c in classes.tolist() if c != cls]
            _run_task(
                name=f"ova_{label_to_key(cls)}",
                pos_classes=[cls],
                neg_classes=neg,
                pairwise_subset=False,
            )

    # OVO tasks on nearest-centroid (most confusable) class pairs.
    class_centroids = {}
    for cls in classes.tolist():
        idx = np.where(y_arr == cls)[0]
        class_centroids[cls] = np.asarray(np.mean(X[idx], axis=0), dtype=float).ravel()
    pair_rows = []
    for i, cls_i in enumerate(classes.tolist()):
        for j in range(i + 1, len(classes)):
            cls_j = classes.tolist()[j]
            dist = float(np.linalg.norm(class_centroids[cls_i] - class_centroids[cls_j]))
            pair_rows.append((dist, cls_i, cls_j))
    pair_rows.sort(key=lambda row: row[0])
    if ecoc_max_ovo_pairs > 0:
        for _, cls_i, cls_j in pair_rows[: int(ecoc_max_ovo_pairs)]:
            _run_task(
                name=f"ovo_{label_to_key(cls_i)}_vs_{label_to_key(cls_j)}",
                pos_classes=[cls_i],
                neg_classes=[cls_j],
                pairwise_subset=True,
            )

    # Random ECOC dichotomy tasks.
    used_partitions = set()
    for bit_idx in range(int(ecoc_random_code_bits)):
        cls_perm = np.asarray(classes.tolist(), dtype=object)
        rng.shuffle(cls_perm)
        split = int(np.ceil(0.5 * cls_perm.size))
        split = int(max(1, min(cls_perm.size - 1, split)))
        pos_classes = tuple(cls_perm[:split].tolist())
        neg_classes = tuple(cls_perm[split:].tolist())
        key = _partition_key(pos_classes, neg_classes)
        if key in used_partitions:
            continue
        used_partitions.add(key)
        _run_task(
            name=f"rand_ecoc_bit_{bit_idx + 1}",
            pos_classes=list(pos_classes),
            neg_classes=list(neg_classes),
            pairwise_subset=False,
        )

    if tasks_used <= 0 or total_weight <= 0.0:
        return {}, {}

    aggregated_scores = aggregated_scores / float(total_weight)
    aggregated_scores = normalize_fn(aggregated_scores)
    selected_indices = np.argsort(aggregated_scores)[::-1][: int(min(n_features, max(1, n_target_features)))]

    results = {
        "selected_indices": np.asarray(selected_indices, dtype=int),
        "scores": {int(idx): float(aggregated_scores[int(idx)]) for idx in np.asarray(selected_indices, dtype=int).tolist()},
        "method": "ecoc_class_aware",
        "ecoc_classes_total": int(classes.size),
        "ecoc_min_classes": int(ecoc_min_classes),
        "ecoc_max_ovo_pairs": int(ecoc_max_ovo_pairs),
        "ecoc_random_code_bits": int(ecoc_random_code_bits),
        "ecoc_class_complexity_weight": float(ecoc_class_complexity_weight),
        "ecoc_include_ova_tasks": bool(ecoc_include_ova_tasks),
        "ecoc_negative_ratio": float(ecoc_negative_ratio),
        "ecoc_tasks_used": int(tasks_used),
        "ecoc_task_metadata": list(tasks_metadata),
        "ecoc_class_complexity_scores": dict(class_complexity),
        "ecoc_class_weights": dict(class_weights),
    }
    all_scores = {i: float(aggregated_scores[i]) for i in range(n_features)}
    return results, all_scores


# ---------------------------------------------------------------------------
# Nearest shrunken centroid selector
# ---------------------------------------------------------------------------

def nearest_shrunken_centroid_selection(X, y, n_target_features, *, problem_type,
                                        random_state, nsc_min_classes,
                                        nsc_shrinkage_grid_size,
                                        nsc_thresholding_mode, nsc_order_quantile,
                                        nsc_deep_shrinkage_search,
                                        per_class_quota_enabled,
                                        per_class_quota_min_per_class,
                                        per_class_quota_max_fraction,
                                        normalize_fn):
    """
    PAM/NSC-style multiclass selector:
    1) standardize class-vs-global centroid contrasts,
    2) apply configurable thresholding mode (soft/hard/order),
    3) rank by max surviving class-wise signal per feature.
    """
    if problem_type != "classification" or X.shape[1] == 0:
        return {}, {}

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.size < int(nsc_min_classes):
        return {}, {}
    if np.min(counts) < 2:
        return {}, {}

    n_samples, n_features = X_arr.shape
    n_target = int(min(max(1, n_target_features), n_features))

    class_means = np.zeros((classes.size, n_features), dtype=float)
    residual_ss = np.zeros(n_features, dtype=float)
    for row_idx, cls in enumerate(classes.tolist()):
        idx = np.where(y_arr == cls)[0]
        X_cls = X_arr[idx]
        mu = np.asarray(np.mean(X_cls, axis=0), dtype=float).ravel()
        class_means[row_idx] = mu
        diffs = X_cls - mu[None, :]
        residual_ss += np.sum(diffs * diffs, axis=0)

    global_mean = np.asarray(np.mean(X_arr, axis=0), dtype=float).ravel()
    pooled_denom = float(max(1, n_samples - int(classes.size)))
    pooled_sd = np.sqrt(np.maximum(residual_ss / pooled_denom, 1e-12))
    finite_sd = pooled_sd[np.isfinite(pooled_sd)]
    s0 = float(np.median(finite_sd)) if finite_sd.size > 0 else 0.0
    scale = np.maximum(1e-8, pooled_sd + s0)
    # Tibshirani et al. (2002): class-size normalization m_k = sqrt(1/n_k - 1/n).
    # This prevents systematic over/under-shrinkage across imbalanced classes.
    mk = np.sqrt(
        np.maximum(
            (1.0 / np.maximum(counts.astype(float), 1.0)) - (1.0 / float(max(1, n_samples))),
            1e-12,
        )
    )
    mk = np.asarray(mk, dtype=float).ravel()
    mk_by_class = {
        label_to_key(cls): float(mk_i)
        for cls, mk_i in zip(classes.tolist(), mk.tolist())
    }
    d_raw = (class_means - global_mean[None, :]) / (mk[:, None] * scale[None, :])
    d_raw = np.nan_to_num(np.asarray(d_raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    base_signal = np.max(np.abs(d_raw), axis=0)
    base_signal = np.asarray(base_signal, dtype=float).ravel()
    abs_raw = np.abs(d_raw)

    mode_requested = str(nsc_thresholding_mode or "soft").strip().lower()
    if mode_requested == "order":
        mode_requested = "quantile_hard"  # backward compat rename
    if mode_requested == "auto":
        threshold_modes = ["soft", "hard", "quantile_hard"]
    elif mode_requested in {"soft", "hard", "quantile_hard"}:
        threshold_modes = [mode_requested]
    else:
        mode_requested = "soft"
        threshold_modes = ["soft"]

    d_max = float(np.max(np.abs(d_raw))) if d_raw.size > 0 else 0.0
    if not np.isfinite(d_max) or d_max <= 0.0:
        selected = np.argsort(base_signal)[::-1][:n_target]
        out_scores = normalize_fn(base_signal)
        ranked_full = np.argsort(np.asarray(out_scores, dtype=float).ravel())[::-1]
        class_rankings = {}
        for row_idx, cls in enumerate(classes.tolist()):
            row = np.asarray(abs_raw[row_idx], dtype=float).ravel()
            order = np.argsort(row)[::-1]
            positives = [int(i) for i in order.tolist() if float(row[int(i)]) > 0.0]
            if positives:
                class_rankings[label_to_key(cls)] = positives
        selected_quota, quota_meta = apply_per_class_quota_overlay(
            np.asarray(selected, dtype=int),
            np.asarray(ranked_full, dtype=int),
            class_rankings,
            n_target,
            per_class_quota_enabled=per_class_quota_enabled,
            per_class_quota_min_per_class=per_class_quota_min_per_class,
            per_class_quota_max_fraction=per_class_quota_max_fraction,
        )
        results = {
            "selected_indices": np.asarray(selected_quota, dtype=int),
            "scores": {
                int(idx): float(out_scores[int(idx)])
                for idx in np.asarray(selected_quota, dtype=int).tolist()
            },
            "method": "nearest_shrunken_centroid",
            "nsc_classes_total": int(classes.size),
            "nsc_min_classes": int(nsc_min_classes),
            "nsc_shrinkage_grid_size": int(nsc_shrinkage_grid_size),
            "nsc_thresholding_mode_requested": str(mode_requested),
            "nsc_thresholding_mode": "soft",
            "nsc_thresholding_modes_evaluated": list(threshold_modes),
            "nsc_order_quantile": float(nsc_order_quantile),
            "nsc_deep_shrinkage_search": bool(nsc_deep_shrinkage_search),
            "nsc_best_delta": 0.0,
            "nsc_best_proxy_balanced_accuracy": float("nan"),
            "nsc_path_metadata": [],
            "nsc_nonzero_features": int(np.sum(base_signal > 0)),
            "nsc_scale_s0": float(s0),
            "nsc_class_size_normalization": dict(mk_by_class),
            "nsc_class_size_normalization_min": float(np.min(mk)) if mk.size > 0 else float("nan"),
            "nsc_class_size_normalization_max": float(np.max(mk)) if mk.size > 0 else float("nan"),
            "nsc_per_class_quota_enabled": bool(quota_meta.get("enabled", False)),
            "nsc_per_class_quota_applied": bool(quota_meta.get("applied", False)),
            "nsc_per_class_quota_budget": int(quota_meta.get("quota_budget", 0)),
            "nsc_per_class_quota_forced_additions": list(quota_meta.get("forced_additions", [])),
            "nsc_per_class_quota_forced_by_class": dict(quota_meta.get("forced_by_class", {})),
            "nsc_per_class_quota_class_hits_before": dict(quota_meta.get("class_hits_before", {})),
            "nsc_per_class_quota_class_hits_after": dict(quota_meta.get("class_hits_after", {})),
            "nsc_per_class_quota_meta": dict(quota_meta),
        }
        all_scores = {i: float(out_scores[i]) for i in range(n_features)}
        return results, all_scores

    try:
        delta_grid = np.linspace(
            0.0,
            max(0.0, 0.95 * d_max),
            num=int(max(2, nsc_shrinkage_grid_size)),
        )
    except Exception as exc:
        delta_grid = np.asarray([0.0, 0.25 * d_max, 0.50 * d_max], dtype=float)
    if bool(nsc_deep_shrinkage_search):
        try:
            q_grid = np.linspace(0.15, 0.95, num=int(max(4, 2 * nsc_shrinkage_grid_size)))
            q_vals = np.quantile(abs_raw.ravel(), q_grid)
            q_vals = np.asarray(q_vals, dtype=float).ravel()
            q_vals = q_vals[np.isfinite(q_vals) & (q_vals >= 0.0)]
            if q_vals.size > 0:
                delta_grid = np.concatenate([np.asarray(delta_grid, dtype=float).ravel(), q_vals], axis=0)
        except Exception as exc:
            pass
    delta_grid = np.unique(np.asarray(delta_grid, dtype=float).ravel())

    min_class_count = int(np.min(counts)) if counts.size > 0 else 0
    n_splits = int(max(2, min(3, min_class_count)))
    can_cv = bool(min_class_count >= 2 and classes.size >= 2 and n_splits >= 2)

    path_rows = []
    best_proxy = -np.inf
    best_delta = 0.0
    best_mode = "soft"
    best_scores = np.asarray(base_signal, dtype=float).ravel()
    best_selected = np.argsort(best_scores)[::-1][:n_target]
    best_nonzero = int(np.sum(best_scores > 0))
    best_abs = np.asarray(abs_raw, dtype=float)

    mode_priority = {"soft": 0, "hard": 1, "quantile_hard": 2}

    def _apply_threshold(mode_name, delta_value):
        delta_value = float(max(0.0, delta_value))
        if mode_name == "hard":
            return d_raw * (abs_raw >= delta_value)
        if mode_name == "quantile_hard":
            q = float(np.clip(nsc_order_quantile, 0.50, 0.99))
            class_thresh = np.quantile(abs_raw, q, axis=1, keepdims=True)
            thresh = np.maximum(delta_value, class_thresh)
            return d_raw * (abs_raw >= thresh)
        return np.sign(d_raw) * np.maximum(abs_raw - delta_value, 0.0)

    for mode in threshold_modes:
        for delta in delta_grid.tolist():
            shrunk = _apply_threshold(mode, float(delta))
            signal = np.max(np.abs(shrunk), axis=0)
            signal = np.nan_to_num(np.asarray(signal, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
            if not np.any(signal > 0):
                signal = np.asarray(base_signal, dtype=float).ravel()

            selected = np.argsort(signal)[::-1][:n_target]
            proxy = float("nan")
            if can_cv and selected.size > 0:
                try:
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                    fold_scores = []
                    X_sel = np.asarray(X_arr[:, selected], dtype=float)
                    for tr_idx, te_idx in cv.split(X_sel, y_arr):
                        X_tr = X_sel[tr_idx]
                        y_tr = y_arr[tr_idx]
                        X_te = X_sel[te_idx]
                        y_te = y_arr[te_idx]

                        cls_vals = np.unique(y_tr)
                        centroids = np.vstack([np.mean(X_tr[y_tr == cls], axis=0) for cls in cls_vals])
                        # Nearest-centroid prediction in selected NSC subspace.
                        diff = X_te[:, None, :] - centroids[None, :, :]
                        dists = np.sum(diff * diff, axis=2)
                        pred = np.asarray([cls_vals[idx] for idx in np.argmin(dists, axis=1)], dtype=y_te.dtype)
                        fold_scores.append(safe_balanced_accuracy(y_te, pred))
                    if fold_scores:
                        proxy = float(np.mean(np.asarray(fold_scores, dtype=float)))
                except Exception as exc:
                    proxy = float("nan")

            proxy_cmp = float(proxy) if np.isfinite(proxy) else -np.inf
            nonzero = int(np.sum(signal > 0))
            score_mass = float(np.mean(signal[selected])) if selected.size > 0 else 0.0
            path_rows.append(
                {
                    "mode": str(mode),
                    "delta": float(delta),
                    "proxy_balanced_accuracy": float(proxy) if np.isfinite(proxy) else float("nan"),
                    "nonzero_features": int(nonzero),
                    "selected_score_mass": float(score_mass),
                }
            )

            choose = False
            if proxy_cmp > best_proxy + 1e-12:
                choose = True
            elif abs(proxy_cmp - best_proxy) <= 1e-12:
                if nonzero < best_nonzero:
                    choose = True
                elif nonzero == best_nonzero:
                    if float(delta) > float(best_delta):
                        choose = True
                    elif (
                        abs(float(delta) - float(best_delta)) <= 1e-12
                        and mode_priority.get(mode, 0) > mode_priority.get(best_mode, 0)
                    ):
                        choose = True
            if choose:
                best_proxy = proxy_cmp
                best_delta = float(delta)
                best_mode = str(mode)
                best_scores = signal
                best_selected = np.asarray(selected, dtype=int)
                best_nonzero = int(nonzero)
                best_abs = np.asarray(np.abs(shrunk), dtype=float)

    best_scores_norm = normalize_fn(best_scores)
    best_scores_norm = np.asarray(best_scores_norm, dtype=float).ravel()
    ranked_full = np.argsort(best_scores_norm)[::-1]
    class_rankings = {}
    for row_idx, cls in enumerate(classes.tolist()):
        row = np.asarray(best_abs[row_idx], dtype=float).ravel()
        if row.size != n_features:
            continue
        order = np.argsort(row)[::-1]
        positives = [int(i) for i in order.tolist() if float(row[int(i)]) > 0.0]
        if positives:
            class_rankings[label_to_key(cls)] = positives
    selected_indices, quota_meta = apply_per_class_quota_overlay(
        np.asarray(best_selected, dtype=int),
        np.asarray(ranked_full, dtype=int),
        class_rankings,
        n_target,
        per_class_quota_enabled=per_class_quota_enabled,
        per_class_quota_min_per_class=per_class_quota_min_per_class,
        per_class_quota_max_fraction=per_class_quota_max_fraction,
    )
    results = {
        "selected_indices": selected_indices,
        "scores": {int(idx): float(best_scores_norm[int(idx)]) for idx in selected_indices.tolist()},
        "method": "nearest_shrunken_centroid",
        "nsc_classes_total": int(classes.size),
        "nsc_min_classes": int(nsc_min_classes),
        "nsc_shrinkage_grid_size": int(nsc_shrinkage_grid_size),
        "nsc_thresholding_mode_requested": str(mode_requested),
        "nsc_thresholding_mode": str(best_mode),
        "nsc_thresholding_modes_evaluated": list(threshold_modes),
        "nsc_order_quantile": float(nsc_order_quantile),
        "nsc_deep_shrinkage_search": bool(nsc_deep_shrinkage_search),
        "nsc_best_delta": float(best_delta),
        "nsc_best_proxy_balanced_accuracy": float(best_proxy) if np.isfinite(best_proxy) else float("nan"),
        "nsc_path_metadata": list(path_rows),
        "nsc_nonzero_features": int(best_nonzero),
        "nsc_scale_s0": float(s0),
        "nsc_class_size_normalization": dict(mk_by_class),
        "nsc_class_size_normalization_min": float(np.min(mk)) if mk.size > 0 else float("nan"),
        "nsc_class_size_normalization_max": float(np.max(mk)) if mk.size > 0 else float("nan"),
        "nsc_per_class_quota_enabled": bool(quota_meta.get("enabled", False)),
        "nsc_per_class_quota_applied": bool(quota_meta.get("applied", False)),
        "nsc_per_class_quota_budget": int(quota_meta.get("quota_budget", 0)),
        "nsc_per_class_quota_forced_additions": list(quota_meta.get("forced_additions", [])),
        "nsc_per_class_quota_forced_by_class": dict(quota_meta.get("forced_by_class", {})),
        "nsc_per_class_quota_class_hits_before": dict(quota_meta.get("class_hits_before", {})),
        "nsc_per_class_quota_class_hits_after": dict(quota_meta.get("class_hits_after", {})),
        "nsc_per_class_quota_meta": dict(quota_meta),
    }
    all_scores = {i: float(best_scores_norm[i]) for i in range(n_features)}
    return results, all_scores


# ---------------------------------------------------------------------------
# Class-specific Pareto front selector
# ---------------------------------------------------------------------------

def class_specific_pareto_front_selection(X, y, n_target_features, *, problem_type,
                                          random_state, class_pareto_min_classes,
                                          class_pareto_top_per_class,
                                          class_pareto_global_fraction,
                                          class_pareto_minority_boost,
                                          class_pareto_kw_weight,
                                          prefilter_fn, binary_class_prefilter_fn,
                                          normalize_fn, mi_scorer_fn, f_scorer_fn,
                                          per_class_quota_enabled,
                                          per_class_quota_min_per_class,
                                          per_class_quota_max_fraction):
    """
    A28: class-specific Pareto-front multiclass selector.
    Builds one-vs-rest class relevance vectors, then applies Pareto dominance
    to prioritize features that balance class coverage.
    """
    if problem_type != "classification" or X.shape[1] == 0:
        return {}, {}

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    n_samples, n_features = X_arr.shape
    n_target = int(min(max(1, n_target_features), n_features))
    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.size < int(class_pareto_min_classes):
        return {}, {}
    if counts.size == 0 or int(np.min(counts)) < 2:
        return {}, {}

    global_cap = int(
        min(
            n_features,
            max(
                n_target,
                n_target + int(np.ceil(float(class_pareto_global_fraction) * n_target)),
            ),
        )
    )
    global_pool = prefilter_fn(X_arr, y_arr, max_features=global_cap)
    candidate_set = set(int(i) for i in np.asarray(global_pool, dtype=int).tolist())

    top_per_class = int(min(n_features, max(4, class_pareto_top_per_class)))
    max_count = int(np.max(counts))
    class_scores: Dict[str, np.ndarray] = {}
    class_weights: Dict[str, float] = {}
    class_top_sets: Dict[str, set] = {}

    for cls, cnt in zip(classes.tolist(), counts.tolist()):
        y_bin = (y_arr == cls).astype(int)
        if int(np.sum(y_bin == 1)) < 2 or int(np.sum(y_bin == 0)) < 2:
            continue
        score = binary_class_prefilter_fn(
            X_arr,
            y_bin,
            include_kw=True,
            kw_weight=float(class_pareto_kw_weight),
        )
        if score.size != n_features:
            continue
        weight = float(
            (float(max_count) / max(1.0, float(cnt))) ** float(class_pareto_minority_boost)
        )
        weighted = np.asarray(score * weight, dtype=float).ravel()
        key = str(label_to_key(cls))
        class_scores[key] = weighted
        class_weights[key] = float(weight)
        top_idx = np.argsort(weighted)[::-1][:top_per_class]
        top_set = set(int(i) for i in np.asarray(top_idx, dtype=int).tolist())
        class_top_sets[key] = top_set
        candidate_set.update(top_set)

    if len(class_scores) < 2:
        return {}, {}

    candidate_idx = np.array(sorted(candidate_set), dtype=int)
    if candidate_idx.size == 0:
        return {}, {}

    # Pareto ranking over per-class rank vectors (lower rank is better).
    rank_rows = []
    for cls_key in class_scores.keys():
        cls_vals = np.asarray(class_scores[cls_key][candidate_idx], dtype=float).ravel()
        order = np.argsort(cls_vals)[::-1]
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

    X_candidate = np.asarray(X_arr[:, candidate_idx], dtype=float)
    try:
        mi_global = np.asarray(mi_scorer_fn(X_candidate, y_arr, random_state=random_state), dtype=float).ravel()
    except Exception as exc:
        mi_global = np.zeros(candidate_idx.size, dtype=float)
    mi_global = np.nan_to_num(mi_global, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        f_global, _ = f_scorer_fn(X_candidate, y_arr)
        f_global = np.asarray(f_global, dtype=float).ravel()
    except Exception as exc:
        f_global = np.zeros(candidate_idx.size, dtype=float)
    f_global = np.nan_to_num(f_global, nan=0.0, posinf=0.0, neginf=0.0)
    global_signal = 0.60 * normalize_fn(mi_global) + 0.40 * normalize_fn(f_global)
    global_signal = np.asarray(normalize_fn(global_signal), dtype=float).ravel()

    pareto_signal = 1.0 - normalize_fn(np.asarray(dominated_by, dtype=float))
    dominates_signal = normalize_fn(np.asarray(dominates, dtype=float))
    combined = 0.55 * pareto_signal + 0.25 * dominates_signal + 0.20 * global_signal
    combined = np.asarray(normalize_fn(combined), dtype=float).ravel()

    ranked_local = np.argsort(combined)[::-1]
    ranked_global = np.asarray(candidate_idx[ranked_local], dtype=int)
    selected_local = ranked_local[:n_target]
    selected_indices = np.asarray(candidate_idx[selected_local], dtype=int)
    if selected_indices.size == 0:
        return {}, {}

    class_rankings: Dict[Any, List[int]] = {}
    for cls_key in class_scores.keys():
        vals = np.asarray(class_scores[cls_key][candidate_idx], dtype=float).ravel()
        order = np.argsort(vals)[::-1]
        positives = [
            int(candidate_idx[int(i)])
            for i in order.tolist()
            if float(vals[int(i)]) > 0.0
        ]
        if positives:
            class_rankings[cls_key] = positives
    selected_indices, quota_meta = apply_per_class_quota_overlay(
        selected_indices,
        ranked_global,
        class_rankings,
        n_target,
        per_class_quota_enabled=per_class_quota_enabled,
        per_class_quota_min_per_class=per_class_quota_min_per_class,
        per_class_quota_max_fraction=per_class_quota_max_fraction,
    )

    all_scores_vec = np.zeros(n_features, dtype=float)
    all_scores_vec[candidate_idx] = combined
    all_scores_vec = np.asarray(normalize_fn(all_scores_vec), dtype=float).ravel()

    selected_set = set(int(i) for i in selected_indices.tolist())
    class_hits = {}
    for cls_key, top_set in class_top_sets.items():
        class_hits[cls_key] = int(len(selected_set.intersection(top_set)))

    results = {
        "selected_indices": selected_indices,
        "scores": {
            int(idx): float(all_scores_vec[int(idx)])
            for idx in selected_indices.tolist()
        },
        "method": "class_pareto_front",
        "class_pareto_classes_total": int(classes.size),
        "class_pareto_classes_used": int(len(class_scores)),
        "class_pareto_min_classes": int(class_pareto_min_classes),
        "class_pareto_top_per_class": int(class_pareto_top_per_class),
        "class_pareto_global_fraction": float(class_pareto_global_fraction),
        "class_pareto_minority_boost": float(class_pareto_minority_boost),
        "class_pareto_kw_weight": float(class_pareto_kw_weight),
        "class_pareto_candidate_universe": int(candidate_idx.size),
        "class_pareto_front_size": int(front_size),
        "class_pareto_class_weights": dict(class_weights),
        "class_pareto_class_top_hits": dict(class_hits),
        "class_pareto_rank_matrix_shape": tuple(int(v) for v in rank_matrix.shape),
        "class_pareto_runtime_note": "O(m^2) Pareto dominance over candidate pool",
        "class_pareto_per_class_quota_enabled": bool(quota_meta.get("enabled", False)),
        "class_pareto_per_class_quota_applied": bool(quota_meta.get("applied", False)),
        "class_pareto_per_class_quota_budget": int(quota_meta.get("quota_budget", 0)),
        "class_pareto_per_class_quota_forced_additions": list(quota_meta.get("forced_additions", [])),
        "class_pareto_per_class_quota_forced_by_class": dict(quota_meta.get("forced_by_class", {})),
        "class_pareto_per_class_quota_class_hits_before": dict(quota_meta.get("class_hits_before", {})),
        "class_pareto_per_class_quota_class_hits_after": dict(quota_meta.get("class_hits_after", {})),
        "class_pareto_per_class_quota_meta": dict(quota_meta),
    }
    all_scores = {i: float(all_scores_vec[i]) for i in range(n_features)}
    return results, all_scores


# ---------------------------------------------------------------------------
# Joint multiclass support selector
# ---------------------------------------------------------------------------

def joint_multiclass_support_selection(X, y, n_target_features, *, problem_type,
                                       random_state, joint_multiclass_min_classes,
                                       joint_multiclass_max_features,
                                       joint_multiclass_min_c,
                                       joint_multiclass_max_c,
                                       joint_multiclass_path_grid_size,
                                       joint_multiclass_l1_ratio,
                                       joint_multiclass_univariate_blend,
                                       prefilter_fn, normalize_fn,
                                       mi_scorer_fn, f_scorer_fn):
    """
    Joint multiclass shared-support selector (A14 pilot).
    Fits a multinomial elastic-net path and ranks features by cross-class
    coefficient norms plus path support frequency.
    """
    if problem_type != "classification" or X.shape[1] == 0:
        return {}, {}

    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.size < int(joint_multiclass_min_classes):
        return {}, {}
    if np.min(counts) < 2:
        return {}, {}

    n_features = int(X.shape[1])
    pool_cap = int(
        min(
            n_features,
            max(int(joint_multiclass_max_features), int(max(1, n_target_features))),
        )
    )
    pool_idx = prefilter_fn(X, y_arr, max_features=pool_cap)
    if pool_idx.size == 0:
        return {}, {}

    X_pool = np.asarray(X[:, pool_idx], dtype=float)
    try:
        scaler = StandardScaler()
        X_pool_scaled = scaler.fit_transform(X_pool)
    except Exception as exc:
        X_pool_scaled = X_pool

    try:
        c_grid = np.geomspace(
            max(1e-3, float(joint_multiclass_min_c)),
            max(float(joint_multiclass_min_c) + 1e-3, float(joint_multiclass_max_c)),
            num=int(max(2, joint_multiclass_path_grid_size)),
        )
    except Exception as exc:
        c_grid = np.asarray([0.1, 0.3, 1.0], dtype=float)

    group_acc = np.zeros(pool_idx.size, dtype=float)
    support_acc = np.zeros(pool_idx.size, dtype=float)
    total_weight = 0.0
    fitted_models = 0
    path_metadata = []

    class_count_map = {
        label_to_key(cls): int(cnt)
        for cls, cnt in zip(classes.tolist(), counts.tolist())
    }

    for c_val in np.asarray(c_grid, dtype=float).ravel().tolist():
        c_val = float(c_val)
        if not np.isfinite(c_val) or c_val <= 0.0:
            continue
        try:
            clf = make_logistic_regression(
                random_state=random_state,
                max_iter=5000,
                solver="saga",
                penalty="elasticnet",
                l1_ratio=float(joint_multiclass_l1_ratio),
                C=c_val,
                class_weight="balanced",
                n_jobs=_get_sklearn_n_jobs(),
            )
            clf.fit(X_pool_scaled, y_arr)
            coef = np.asarray(clf.coef_, dtype=float)
            if coef.ndim == 1:
                coef = coef[None, :]
            group_norm = np.sqrt(np.sum(np.square(coef), axis=0))
            group_norm = np.nan_to_num(group_norm, nan=0.0, posinf=0.0, neginf=0.0)
            if group_norm.size != pool_idx.size:
                resized = np.zeros(pool_idx.size, dtype=float)
                upto = int(min(pool_idx.size, group_norm.size))
                if upto > 0:
                    resized[:upto] = group_norm[:upto]
                group_norm = resized

            if np.any(group_norm > 0):
                positive = group_norm[group_norm > 0]
                support_threshold = float(max(1e-8, np.quantile(positive, 0.20)))
            else:
                support_threshold = 1e-8
            support_mask = group_norm >= support_threshold
            support_density = float(np.mean(support_mask))
            path_weight = float(1.0 + 0.5 * (1.0 - support_density))

            group_acc = group_acc + path_weight * normalize_fn(group_norm)
            support_acc = support_acc + path_weight * support_mask.astype(float)
            total_weight += path_weight
            fitted_models += 1
            path_metadata.append(
                {
                    "c_value": float(c_val),
                    "support_threshold": float(support_threshold),
                    "active_features": int(np.sum(support_mask)),
                    "support_density": float(support_density),
                    "path_weight": float(path_weight),
                    "status": "ok",
                }
            )
        except Exception as exc:
            path_metadata.append(
                {
                    "c_value": float(c_val),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    if fitted_models <= 0 or total_weight <= 0.0:
        return {}, {}

    group_signal = normalize_fn(group_acc / float(total_weight))
    support_signal = normalize_fn(support_acc / float(total_weight))
    joint_signal = 0.72 * group_signal + 0.28 * support_signal

    try:
        mi = np.asarray(mi_scorer_fn(X_pool, y_arr, random_state=random_state), dtype=float).ravel()
    except Exception as exc:
        mi = np.zeros(pool_idx.size, dtype=float)
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        f_vals, _ = f_scorer_fn(X_pool, y_arr)
        f_vals = np.asarray(f_vals, dtype=float).ravel()
    except Exception as exc:
        f_vals = np.zeros(pool_idx.size, dtype=float)
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    univariate_signal = 0.50 * normalize_fn(mi) + 0.50 * normalize_fn(f_vals)

    blend = float(joint_multiclass_univariate_blend)
    pooled_scores = (1.0 - blend) * joint_signal + blend * univariate_signal
    pooled_scores = normalize_fn(pooled_scores)

    all_scores = np.zeros(n_features, dtype=float)
    all_scores[pool_idx] = np.asarray(pooled_scores, dtype=float)
    all_scores = normalize_fn(all_scores)

    k_target = int(min(n_features, max(1, n_target_features)))
    selected = np.argsort(all_scores)[::-1][:k_target]

    results = {
        "selected_indices": np.asarray(selected, dtype=int),
        "scores": {int(idx): float(all_scores[int(idx)]) for idx in np.asarray(selected, dtype=int).tolist()},
        "method": "joint_multiclass_support",
        "joint_multiclass_classes_total": int(classes.size),
        "joint_multiclass_class_counts": class_count_map,
        "joint_multiclass_pool_size": int(pool_idx.size),
        "joint_multiclass_path_grid_size": int(joint_multiclass_path_grid_size),
        "joint_multiclass_min_c": float(joint_multiclass_min_c),
        "joint_multiclass_max_c": float(joint_multiclass_max_c),
        "joint_multiclass_l1_ratio": float(joint_multiclass_l1_ratio),
        "joint_multiclass_univariate_blend": float(joint_multiclass_univariate_blend),
        "joint_multiclass_fitted_models": int(fitted_models),
        "joint_multiclass_path_metadata": list(path_metadata),
        "joint_multiclass_mean_support_density": float(np.mean(support_signal)),
    }
    return results, {i: float(all_scores[i]) for i in range(n_features)}


# ---------------------------------------------------------------------------
# DOvE class-specific selector
# ---------------------------------------------------------------------------

def dove_class_specific_selection(X, y, n_target_features, *, problem_type,
                                   random_state, dove_min_classes,
                                   dove_max_pairs_per_class, dove_minority_boost,
                                   dove_specificity_weight, dove_path_grid_size,
                                   ecoc_class_complexity_weight,
                                   ova_linear_backend, linear_svm_max_iter,
                                   normalize_fn):
    """
    Class-specific relevance matrix with a lightweight DOvE-style path search.
    Uses per-class one-vs-each pair tasks to build a class-by-feature matrix,
    then searches support-size scales to maximize class coverage and relevance.
    """
    if problem_type != "classification" or X.shape[1] == 0:
        return {}, {}

    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.size < int(dove_min_classes):
        return {}, {}
    if int(np.min(counts)) < 2:
        return {}, {}

    n_features = int(X.shape[1])
    class_keys = [label_to_key(cls) for cls in classes.tolist()]
    class_to_indices = {cls: np.where(y_arr == cls)[0] for cls in classes.tolist()}
    centroid_by_class = {
        cls: np.asarray(np.mean(X[idx], axis=0), dtype=float).ravel()
        for cls, idx in class_to_indices.items()
    }
    class_complexity, class_weights = ecoc_class_complexity_weights(
        X, y_arr, classes.tolist(),
        ecoc_class_complexity_weight=ecoc_class_complexity_weight,
        normalize_fn=normalize_fn,
    )

    relevance_matrix = np.zeros((classes.size, n_features), dtype=float)
    task_count = np.zeros(classes.size, dtype=float)
    task_meta = []

    for cls_i, cls in enumerate(classes.tolist()):
        distances = []
        for other in classes.tolist():
            if other == cls:
                continue
            dist = float(np.linalg.norm(centroid_by_class[cls] - centroid_by_class[other]))
            distances.append((dist, other))
        distances.sort(key=lambda row: row[0])
        max_pairs = int(min(max(1, dove_max_pairs_per_class), len(distances)))

        for _, other in distances[:max_pairs]:
            idx_pos = class_to_indices[cls]
            idx_neg = class_to_indices[other]
            if idx_pos.size < 2 or idx_neg.size < 2:
                continue

            subset_idx = np.concatenate([idx_pos, idx_neg])
            if subset_idx.size < 4:
                continue

            y_bin = np.isin(y_arr[subset_idx], [cls]).astype(int)
            X_sub = X[subset_idx]

            pair_scores = ecoc_binary_relevance_scores(
                X_sub, y_bin, n_features=n_features,
                random_state=random_state,
                ova_linear_backend=ova_linear_backend,
                linear_svm_max_iter=linear_svm_max_iter,
                normalize_fn=normalize_fn,
            )
            if not np.any(np.isfinite(pair_scores)):
                continue

            cls_key = label_to_key(cls)
            other_key = label_to_key(other)
            base_w = 0.5 * (
                float(class_weights.get(cls_key, 1.0))
                + float(class_weights.get(other_key, 1.0))
            )
            count_w = float(1.0 / max(1.0, float(idx_pos.size) ** float(dove_minority_boost)))
            pair_weight = float(max(1e-8, base_w * count_w))

            relevance_matrix[cls_i] = relevance_matrix[cls_i] + pair_weight * np.asarray(pair_scores, dtype=float)
            task_count[cls_i] += pair_weight

            task_meta.append(
                {
                    "class": cls_key,
                    "vs_class": other_key,
                    "n_pos": int(idx_pos.size),
                    "n_neg": int(idx_neg.size),
                    "task_weight": float(pair_weight),
                }
            )

    used_rows = task_count > 0
    if not np.any(used_rows):
        return {}, {}

    for row_idx in range(relevance_matrix.shape[0]):
        if task_count[row_idx] > 0:
            relevance_matrix[row_idx] = relevance_matrix[row_idx] / float(task_count[row_idx])
        relevance_matrix[row_idx] = normalize_fn(relevance_matrix[row_idx])

    class_weight_vector = []
    for cls, cnt in zip(classes.tolist(), counts.tolist()):
        cls_key = label_to_key(cls)
        complexity_w = float(class_weights.get(cls_key, 1.0))
        count_w = float(1.0 / max(1.0, float(cnt) ** float(dove_minority_boost)))
        class_weight_vector.append(complexity_w * count_w)
    class_weight_vector = np.asarray(class_weight_vector, dtype=float)
    class_weight_vector = np.nan_to_num(class_weight_vector, nan=0.0, posinf=0.0, neginf=0.0)
    if np.sum(class_weight_vector) <= 0:
        class_weight_vector = np.ones(classes.size, dtype=float)
    class_weight_vector = class_weight_vector / np.sum(class_weight_vector)

    aggregated_signal = np.dot(class_weight_vector, relevance_matrix)
    aggregated_signal = normalize_fn(aggregated_signal)
    specificity_signal = np.std(relevance_matrix, axis=0)
    specificity_signal = normalize_fn(specificity_signal)
    score_vector = (1.0 - float(dove_specificity_weight)) * aggregated_signal + float(
        dove_specificity_weight
    ) * specificity_signal
    score_vector = normalize_fn(score_vector)

    base_per_class = int(max(1, np.ceil(max(1, n_target_features) / max(1, classes.size))))
    path_scales = np.linspace(0.60, 1.40, num=int(max(2, dove_path_grid_size)))
    path_metadata = []
    best_objective = float("-inf")
    best_support = np.array([], dtype=int)

    for step_idx, scale in enumerate(path_scales, start=1):
        per_class_k = int(max(1, round(base_per_class * float(scale))))
        support_parts = []
        used_class_count = 0

        for row_idx in range(relevance_matrix.shape[0]):
            row = np.asarray(relevance_matrix[row_idx], dtype=float).ravel()
            if np.max(row) <= 0:
                continue
            top_idx = np.argsort(row)[::-1][:per_class_k]
            if top_idx.size <= 0:
                continue
            used_class_count += 1
            support_parts.extend(int(i) for i in top_idx.tolist())

        if not support_parts:
            continue

        support = np.array(sorted(set(int(i) for i in support_parts if 0 <= int(i) < n_features)), dtype=int)
        if support.size == 0:
            continue

        support_scores = np.asarray(score_vector[support], dtype=float)
        relevance_term = float(np.mean(support_scores)) if support_scores.size else 0.0
        coverage_term = float(used_class_count / max(1, classes.size))

        redundancy_term = 0.0
        if support.size > 1:
            corr_subset = support[: min(80, support.size)]
            try:
                corr = np.corrcoef(np.asarray(X[:, corr_subset], dtype=float), rowvar=False)
                if isinstance(corr, np.ndarray) and corr.ndim == 2:
                    upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
                    if upper.size > 0:
                        redundancy_term = float(np.nanmean(upper))
            except Exception as exc:
                redundancy_term = 0.0

        objective = float(relevance_term + 0.25 * coverage_term - 0.15 * redundancy_term)
        path_metadata.append(
            {
                "path_step": int(step_idx),
                "scale": float(scale),
                "per_class_top_k": int(per_class_k),
                "support_size": int(support.size),
                "classes_used": int(used_class_count),
                "coverage_term": float(coverage_term),
                "redundancy_term": float(redundancy_term),
                "objective": float(objective),
            }
        )

        if objective > best_objective:
            best_objective = objective
            best_support = support

    if best_support.size == 0:
        best_support = np.argsort(score_vector)[::-1][: int(max(1, n_target_features))]

    best_support_set = set(int(i) for i in best_support.tolist())
    ranked_full = np.argsort(score_vector)[::-1]
    support_ranked = [int(i) for i in ranked_full.tolist() if int(i) in best_support_set]
    remaining_ranked = [int(i) for i in ranked_full.tolist() if int(i) not in best_support_set]
    ordered = np.asarray(support_ranked + remaining_ranked, dtype=int)

    k_target = int(min(n_features, max(1, n_target_features)))
    selected = ordered[:k_target]

    class_top_features = {}
    for cls_key, row in zip(class_keys, relevance_matrix):
        top_idx = np.argsort(row)[::-1][: int(min(10, n_features))]
        class_top_features[cls_key] = [int(i) for i in np.asarray(top_idx, dtype=int).tolist()]

    results = {
        "selected_indices": np.asarray(selected, dtype=int),
        "scores": {int(idx): float(score_vector[int(idx)]) for idx in np.asarray(selected, dtype=int).tolist()},
        "method": "dove_class_specific",
        "dove_classes_total": int(classes.size),
        "dove_classes_used": int(np.sum(used_rows)),
        "dove_min_classes": int(dove_min_classes),
        "dove_max_pairs_per_class": int(dove_max_pairs_per_class),
        "dove_path_grid_size": int(dove_path_grid_size),
        "dove_specificity_weight": float(dove_specificity_weight),
        "dove_minority_boost": float(dove_minority_boost),
        "dove_class_keys": list(class_keys),
        "dove_class_counts": {label_to_key(cls): int(cnt) for cls, cnt in zip(classes.tolist(), counts.tolist())},
        "dove_class_weights": {label_to_key(cls): float(w) for cls, w in zip(classes.tolist(), class_weight_vector)},
        "dove_class_top_features": dict(class_top_features),
        "dove_pair_tasks": list(task_meta),
        "dove_path_metadata": list(path_metadata),
        "dove_best_path_objective": float(best_objective if np.isfinite(best_objective) else 0.0),
        "class_specific_relevance_matrix": np.asarray(relevance_matrix, dtype=float),
    }
    all_scores = {i: float(score_vector[i]) for i in range(n_features)}
    return results, all_scores


# ---------------------------------------------------------------------------
# Sparse multinomial screening helper
# ---------------------------------------------------------------------------

def sparse_multinomial_screen_candidates(X_pool, y_arr, n_target_features, *,
                                          sparse_multinomial_screening_mode,
                                          sparse_multinomial_screening_keep_fraction,
                                          sparse_multinomial_screening_min_features,
                                          random_state, normalize_fn,
                                          mi_scorer_fn, f_scorer_fn):
    """
    Runtime-containment screening for sparse multinomial path fitting.
    Provides three isolated rule-family toggles:
    - prefilter_aggressive: strong-rule-inspired
    - prefilter_balanced: GAP-safe-inspired
    - prefilter_conservative: Slores-inspired
    """
    n_pool = int(X_pool.shape[1]) if X_pool.ndim == 2 else 0
    mode = str(sparse_multinomial_screening_mode).strip().lower()
    keep_fraction = float(np.clip(sparse_multinomial_screening_keep_fraction, 0.05, 1.0))
    min_keep = int(max(8, sparse_multinomial_screening_min_features, int(max(1, n_target_features))))

    meta = {
        "mode": str(mode),
        "rule_family": "none",
        "requested_keep_fraction": float(keep_fraction),
        "requested_min_features": int(sparse_multinomial_screening_min_features),
        "initial_pool_size": int(n_pool),
        "retained_pool_size": int(n_pool),
        "keep_target": int(n_pool),
        "keep_upper_cap": int(n_pool),
        "score_threshold": 0.0,
        "safety_floor": 0.0,
        "applied": False,
        "status": "disabled" if mode == "none" else "pending",
    }
    if mode == "none" or n_pool <= 0:
        return np.arange(n_pool, dtype=int), meta

    mode_cfg = {
        "prefilter_aggressive": {
            "rule_family": "strong_rule_surrogate",
            "keep_scale": 0.80,
            "upper_cap_scale": 1.00,
            "floor_quantile": 0.45,
            "weights": {
                "relevance": 0.20,
                "spread": 0.10,
                "variance": 0.05,
                "grad_peak": 0.45,
                "grad_mean": 0.20,
            },
        },
        "prefilter_balanced": {
            "rule_family": "gap_safe_surrogate",
            "keep_scale": 1.00,
            "upper_cap_scale": 1.05,
            "floor_quantile": 0.25,
            "weights": {
                "relevance": 0.25,
                "spread": 0.15,
                "variance": 0.15,
                "grad_peak": 0.25,
                "grad_mean": 0.20,
            },
        },
        "prefilter_conservative": {
            "rule_family": "slores_surrogate",
            "keep_scale": 1.20,
            "upper_cap_scale": 1.25,
            "floor_quantile": 0.10,
            "weights": {
                "relevance": 0.25,
                "spread": 0.20,
                "variance": 0.20,
                "grad_peak": 0.20,
                "grad_mean": 0.15,
            },
        },
    }.get(mode)
    if mode_cfg is None:
        return np.arange(n_pool, dtype=int), meta

    requested_keep = int(max(min_keep, int(np.ceil(keep_fraction * float(n_pool)))))
    requested_keep = int(min(n_pool, max(1, requested_keep)))
    keep_n = int(max(min_keep, int(np.ceil(float(mode_cfg["keep_scale"]) * float(requested_keep)))))
    keep_n = int(min(n_pool, max(1, keep_n)))
    keep_upper_cap = int(max(keep_n, int(np.ceil(float(mode_cfg["upper_cap_scale"]) * float(keep_n)))))
    keep_upper_cap = int(min(n_pool, max(1, keep_upper_cap)))

    X_arr = np.asarray(X_pool, dtype=float)
    if X_arr.ndim != 2 or X_arr.shape[1] <= 0:
        meta["status"] = "invalid_pool"
        return np.arange(n_pool, dtype=int), meta
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    y_local = np.asarray(y_arr)
    try:
        mi_scores = np.asarray(
            mi_scorer_fn(X_arr, y_local, random_state=random_state),
            dtype=float,
        ).ravel()
    except Exception as exc:
        mi_scores = np.zeros(n_pool, dtype=float)
    mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        f_scores, _ = f_scorer_fn(X_arr, y_local)
        f_scores = np.asarray(f_scores, dtype=float).ravel()
    except Exception as exc:
        f_scores = np.zeros(n_pool, dtype=float)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    rel = 0.55 * normalize_fn(mi_scores) + 0.45 * normalize_fn(f_scores)
    rel = np.asarray(normalize_fn(rel), dtype=float).ravel()

    try:
        classes = np.unique(y_local)
        if classes.size >= 2:
            class_means = []
            for cls in classes.tolist():
                mask = y_local == cls
                if int(np.sum(mask)) <= 0:
                    continue
                class_means.append(np.mean(X_arr[mask], axis=0).ravel())
            if len(class_means) >= 2:
                means = np.vstack(class_means)
                spread = np.mean(np.abs(means - np.mean(means, axis=0, keepdims=True)), axis=0)
            else:
                spread = np.zeros(n_pool, dtype=float)
        else:
            spread = np.zeros(n_pool, dtype=float)
    except Exception as exc:
        spread = np.zeros(n_pool, dtype=float)
    spread = np.asarray(normalize_fn(spread), dtype=float).ravel()

    try:
        variance = np.asarray(np.var(X_arr, axis=0), dtype=float).ravel()
    except Exception as exc:
        variance = np.zeros(n_pool, dtype=float)
    variance = np.asarray(normalize_fn(variance), dtype=float).ravel()

    grad_peak = np.zeros(n_pool, dtype=float)
    grad_mean = np.zeros(n_pool, dtype=float)
    try:
        n_samples = int(X_arr.shape[0])
        if n_samples > 0:
            x_scaled = StandardScaler().fit_transform(X_arr)
            grad_rows = []
            for cls in np.unique(y_local).tolist():
                y_bin = (y_local == cls).astype(float)
                if int(np.sum(y_bin)) <= 0 or int(np.sum(y_bin)) >= y_bin.size:
                    continue
                y_center = y_bin - float(np.mean(y_bin))
                grad = np.abs(np.dot(y_center, x_scaled)) / float(max(1, n_samples))
                grad_rows.append(np.asarray(grad, dtype=float).ravel())
            if grad_rows:
                grad_matrix = np.vstack(grad_rows)
                grad_peak = np.max(grad_matrix, axis=0)
                grad_mean = np.mean(grad_matrix, axis=0)
    except Exception as exc:
        grad_peak = np.zeros(n_pool, dtype=float)
        grad_mean = np.zeros(n_pool, dtype=float)
    grad_peak = np.asarray(normalize_fn(grad_peak), dtype=float).ravel()
    grad_mean = np.asarray(normalize_fn(grad_mean), dtype=float).ravel()

    weights = dict(mode_cfg["weights"])
    signal = (
        float(weights.get("relevance", 0.0)) * rel
        + float(weights.get("spread", 0.0)) * spread
        + float(weights.get("variance", 0.0)) * variance
        + float(weights.get("grad_peak", 0.0)) * grad_peak
        + float(weights.get("grad_mean", 0.0)) * grad_mean
    )
    signal = np.asarray(normalize_fn(signal), dtype=float).ravel()

    # Quantile threshold + gradient-based safety floor to avoid dropping
    # potentially active coordinates in rule-inspired screening modes.
    keep_ratio = float(keep_n) / float(max(1, n_pool))
    rank_quantile = float(np.clip(1.0 - keep_ratio, 0.0, 1.0))
    rank_threshold = float(np.quantile(signal, rank_quantile))
    floor_base = np.maximum(grad_peak, grad_mean)
    safety_floor = float(np.quantile(floor_base, float(mode_cfg["floor_quantile"])))
    threshold = float(max(rank_threshold, safety_floor))

    retained = np.where(signal >= threshold)[0]
    if retained.size < keep_n:
        retained = np.argsort(signal)[::-1][:keep_n]
    if retained.size > keep_upper_cap:
        retained = np.argsort(signal)[::-1][:keep_upper_cap]
    retained = np.array(sorted(set(int(i) for i in retained.tolist())), dtype=int)
    if retained.size <= 0:
        meta["status"] = "empty_retained"
        return np.arange(n_pool, dtype=int), meta

    meta.update(
        {
            "rule_family": str(mode_cfg["rule_family"]),
            "retained_pool_size": int(retained.size),
            "keep_target": int(keep_n),
            "keep_upper_cap": int(keep_upper_cap),
            "score_threshold": float(threshold),
            "safety_floor": float(safety_floor),
            "applied": bool(retained.size < n_pool),
            "status": "ok",
            "signal_component_means": {
                "relevance": float(np.mean(rel)),
                "spread": float(np.mean(spread)),
                "variance": float(np.mean(variance)),
                "grad_peak": float(np.mean(grad_peak)),
                "grad_mean": float(np.mean(grad_mean)),
            },
        }
    )
    return retained, meta


# ---------------------------------------------------------------------------
# Sparse multinomial selector
# ---------------------------------------------------------------------------

def sparse_multinomial_selection(X, y, n_target_features, *, problem_type,
                                  random_state, sparse_multinomial_min_classes,
                                  sparse_multinomial_max_features,
                                  sparse_multinomial_min_c,
                                  sparse_multinomial_max_c,
                                  sparse_multinomial_path_grid_size,
                                  sparse_multinomial_l1_ratio,
                                  sparse_multinomial_univariate_blend,
                                  sparse_multinomial_backend,
                                  sparse_multinomial_max_iter,
                                  sparse_multinomial_screening_mode,
                                  sparse_multinomial_screening_keep_fraction,
                                  sparse_multinomial_screening_min_features,
                                  sparse_multinomial_screening_fallback_on_failure,
                                  prefilter_fn, normalize_fn,
                                  mi_scorer_fn, f_scorer_fn):
    """
    Sparse multinomial backend with group-regularized path scoring.
    Aggregates multinomial coefficient group norms and class-support
    frequencies across regularization strengths.
    """
    if problem_type != "classification" or X.shape[1] == 0:
        return {}, {}

    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.size < int(sparse_multinomial_min_classes):
        return {}, {}
    if int(np.min(counts)) < 2:
        return {}, {}

    n_features = int(X.shape[1])
    pool_cap = int(
        min(
            n_features,
            max(int(sparse_multinomial_max_features), int(max(1, n_target_features))),
        )
    )
    pool_idx = prefilter_fn(X, y_arr, max_features=pool_cap)
    if pool_idx.size == 0:
        return {}, {}

    X_pool_full = np.asarray(X[:, pool_idx], dtype=float)
    screen_local_idx, screening_meta = sparse_multinomial_screen_candidates(
        X_pool_full,
        y_arr,
        int(max(1, n_target_features)),
        sparse_multinomial_screening_mode=sparse_multinomial_screening_mode,
        sparse_multinomial_screening_keep_fraction=sparse_multinomial_screening_keep_fraction,
        sparse_multinomial_screening_min_features=sparse_multinomial_screening_min_features,
        random_state=random_state,
        normalize_fn=normalize_fn,
        mi_scorer_fn=mi_scorer_fn,
        f_scorer_fn=f_scorer_fn,
    )
    screen_local_idx = np.asarray(screen_local_idx, dtype=int).ravel()
    if screen_local_idx.size <= 0:
        screening_meta["status"] = "empty_after_screening"
        if not bool(sparse_multinomial_screening_fallback_on_failure):
            return {}, {}
        screen_local_idx = np.arange(pool_idx.size, dtype=int)

    screening_applied = bool(
        screening_meta.get("applied", False) and screen_local_idx.size < pool_idx.size
    )
    if screening_applied:
        screened_pool_idx = np.asarray(pool_idx[screen_local_idx], dtype=int)
        screened_X_pool = np.asarray(X_pool_full[:, screen_local_idx], dtype=float)
    else:
        screened_pool_idx = np.asarray(pool_idx, dtype=int)
        screened_X_pool = np.asarray(X_pool_full, dtype=float)

    try:
        c_grid = np.geomspace(
            max(1e-3, float(sparse_multinomial_min_c)),
            max(float(sparse_multinomial_min_c) + 1e-3, float(sparse_multinomial_max_c)),
            num=int(max(2, sparse_multinomial_path_grid_size)),
        )
    except Exception as exc:
        c_grid = np.asarray([0.1, 0.3, 1.0], dtype=float)

    backend_mode = str(sparse_multinomial_backend)
    if backend_mode == "l1":
        penalties = [("l1", None)]
    elif backend_mode == "elasticnet":
        penalties = [("elasticnet", float(sparse_multinomial_l1_ratio))]
    else:
        penalties = [("l1", None), ("elasticnet", float(sparse_multinomial_l1_ratio))]

    attempt_specs = [("screened", screened_pool_idx, screened_X_pool)]
    if screening_applied and bool(sparse_multinomial_screening_fallback_on_failure):
        attempt_specs.append(("fallback_full", np.asarray(pool_idx, dtype=int), np.asarray(X_pool_full, dtype=float)))

    fit_pool_idx = None
    fit_X_pool = None
    sparse_signal = None
    support_signal = None
    fitted_models = 0
    path_metadata = []
    fitted_attempt = "none"
    fallback_used = False

    for attempt_name, attempt_pool_idx, attempt_X_pool in attempt_specs:
        signal_acc = np.zeros(attempt_pool_idx.size, dtype=float)
        support_acc = np.zeros(attempt_pool_idx.size, dtype=float)
        total_weight = 0.0
        fitted_models = 0
        path_metadata = []

        try:
            scaler = StandardScaler()
            attempt_X_pool_scaled = scaler.fit_transform(np.asarray(attempt_X_pool, dtype=float))
        except Exception as exc:
            attempt_X_pool_scaled = np.asarray(attempt_X_pool, dtype=float)

        for penalty, l1_ratio in penalties:
            for c_val in np.asarray(c_grid, dtype=float).ravel().tolist():
                c_val = float(c_val)
                if not np.isfinite(c_val) or c_val <= 0.0:
                    continue
                try:
                    clf_kwargs = {
                        "random_state": random_state,
                        "max_iter": int(sparse_multinomial_max_iter),
                        "solver": "saga",
                        "penalty": str(penalty),
                        "C": c_val,
                        "class_weight": "balanced",
                        "n_jobs": 1,
                    }
                    if penalty == "elasticnet":
                        clf_kwargs["l1_ratio"] = float(
                            l1_ratio if l1_ratio is not None else sparse_multinomial_l1_ratio
                        )
                    clf = make_logistic_regression(**clf_kwargs)
                    clf.fit(attempt_X_pool_scaled, y_arr)

                    coef = np.asarray(clf.coef_, dtype=float)
                    if coef.ndim == 1:
                        coef = coef[None, :]
                    abs_coef = np.abs(coef)
                    group_norm = np.sqrt(np.sum(np.square(abs_coef), axis=0))
                    class_presence = np.mean(abs_coef > 1e-8, axis=0)
                    class_peak = np.max(abs_coef, axis=0)

                    group_norm = np.nan_to_num(group_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    class_presence = np.nan_to_num(class_presence, nan=0.0, posinf=0.0, neginf=0.0)
                    class_peak = np.nan_to_num(class_peak, nan=0.0, posinf=0.0, neginf=0.0)

                    if group_norm.size != attempt_pool_idx.size:
                        resized = np.zeros(attempt_pool_idx.size, dtype=float)
                        upto = int(min(attempt_pool_idx.size, group_norm.size))
                        if upto > 0:
                            resized[:upto] = group_norm[:upto]
                        group_norm = resized
                    if class_presence.size != attempt_pool_idx.size:
                        resized = np.zeros(attempt_pool_idx.size, dtype=float)
                        upto = int(min(attempt_pool_idx.size, class_presence.size))
                        if upto > 0:
                            resized[:upto] = class_presence[:upto]
                        class_presence = resized
                    if class_peak.size != attempt_pool_idx.size:
                        resized = np.zeros(attempt_pool_idx.size, dtype=float)
                        upto = int(min(attempt_pool_idx.size, class_peak.size))
                        if upto > 0:
                            resized[:upto] = class_peak[:upto]
                        class_peak = resized

                    sparse_signal_row = (
                        0.55 * normalize_fn(group_norm)
                        + 0.25 * normalize_fn(class_presence)
                        + 0.20 * normalize_fn(class_peak)
                    )
                    sparse_signal_row = np.nan_to_num(
                        np.asarray(sparse_signal_row, dtype=float),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                    support_density = float(np.mean(class_presence > 0.0))
                    path_weight = float(1.0 + 0.4 * (1.0 - support_density))

                    signal_acc = signal_acc + path_weight * sparse_signal_row
                    support_acc = support_acc + path_weight * (class_presence > 0.0).astype(float)
                    total_weight += path_weight
                    fitted_models += 1

                    path_metadata.append(
                        {
                            "screening_attempt": str(attempt_name),
                            "penalty": str(penalty),
                            "l1_ratio": None if l1_ratio is None else float(l1_ratio),
                            "c_value": float(c_val),
                            "support_density": float(support_density),
                            "path_weight": float(path_weight),
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    path_metadata.append(
                        {
                            "screening_attempt": str(attempt_name),
                            "penalty": str(penalty),
                            "l1_ratio": None if l1_ratio is None else float(l1_ratio),
                            "c_value": float(c_val),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

        if fitted_models > 0 and total_weight > 0.0:
            fit_pool_idx = np.asarray(attempt_pool_idx, dtype=int)
            fit_X_pool = np.asarray(attempt_X_pool, dtype=float)
            sparse_signal = normalize_fn(signal_acc / float(total_weight))
            support_signal = normalize_fn(support_acc / float(total_weight))
            fitted_attempt = str(attempt_name)
            fallback_used = bool(attempt_name == "fallback_full")
            break

    if fit_pool_idx is None or fit_X_pool is None or sparse_signal is None or support_signal is None:
        return {}, {}

    pooled_scores = 0.75 * sparse_signal + 0.25 * support_signal

    try:
        mi = np.asarray(mi_scorer_fn(fit_X_pool, y_arr, random_state=random_state), dtype=float).ravel()
    except Exception as exc:
        mi = np.zeros(fit_pool_idx.size, dtype=float)
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        f_vals, _ = f_scorer_fn(fit_X_pool, y_arr)
        f_vals = np.asarray(f_vals, dtype=float).ravel()
    except Exception as exc:
        f_vals = np.zeros(fit_pool_idx.size, dtype=float)
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    univariate_signal = 0.50 * normalize_fn(mi) + 0.50 * normalize_fn(f_vals)

    blend = float(sparse_multinomial_univariate_blend)
    pooled_scores = (1.0 - blend) * pooled_scores + blend * univariate_signal
    pooled_scores = normalize_fn(pooled_scores)

    all_scores = np.zeros(n_features, dtype=float)
    all_scores[fit_pool_idx] = np.asarray(pooled_scores, dtype=float)
    all_scores = normalize_fn(all_scores)

    k_target = int(min(n_features, max(1, n_target_features)))
    selected = np.argsort(all_scores)[::-1][:k_target]

    results = {
        "selected_indices": np.asarray(selected, dtype=int),
        "scores": {int(idx): float(all_scores[int(idx)]) for idx in np.asarray(selected, dtype=int).tolist()},
        "method": "sparse_multinomial",
        "sparse_multinomial_classes_total": int(classes.size),
        "sparse_multinomial_class_counts": {
            label_to_key(cls): int(cnt) for cls, cnt in zip(classes.tolist(), counts.tolist())
        },
        "sparse_multinomial_pool_size": int(fit_pool_idx.size),
        "sparse_multinomial_pool_size_initial": int(pool_idx.size),
        "sparse_multinomial_backend": str(sparse_multinomial_backend),
        "sparse_multinomial_path_grid_size": int(sparse_multinomial_path_grid_size),
        "sparse_multinomial_min_c": float(sparse_multinomial_min_c),
        "sparse_multinomial_max_c": float(sparse_multinomial_max_c),
        "sparse_multinomial_l1_ratio": float(sparse_multinomial_l1_ratio),
        "sparse_multinomial_univariate_blend": float(sparse_multinomial_univariate_blend),
        "sparse_multinomial_fitted_models": int(fitted_models),
        "sparse_multinomial_path_metadata": list(path_metadata),
        "sparse_multinomial_mean_support_density": float(np.mean(support_signal)),
        "sparse_multinomial_screening_mode": str(sparse_multinomial_screening_mode),
        "sparse_multinomial_screening_keep_fraction": float(sparse_multinomial_screening_keep_fraction),
        "sparse_multinomial_screening_min_features": int(sparse_multinomial_screening_min_features),
        "sparse_multinomial_screening_fallback_on_failure": bool(
            sparse_multinomial_screening_fallback_on_failure
        ),
        "sparse_multinomial_screening_applied": bool(screening_applied),
        "sparse_multinomial_screening_initial_pool_size": int(
            screening_meta.get("initial_pool_size", pool_idx.size)
        ),
        "sparse_multinomial_screening_retained_pool_size": int(
            screening_meta.get("retained_pool_size", fit_pool_idx.size)
        ),
        "sparse_multinomial_screening_rule_family": str(
            screening_meta.get("rule_family", "none")
        ),
        "sparse_multinomial_screening_keep_target": int(
            screening_meta.get("keep_target", fit_pool_idx.size)
        ),
        "sparse_multinomial_screening_keep_upper_cap": int(
            screening_meta.get("keep_upper_cap", fit_pool_idx.size)
        ),
        "sparse_multinomial_screening_score_threshold": float(
            screening_meta.get("score_threshold", 0.0)
        ),
        "sparse_multinomial_screening_safety_floor": float(
            screening_meta.get("safety_floor", 0.0)
        ),
        "sparse_multinomial_screening_attempt": str(fitted_attempt),
        "sparse_multinomial_screening_fallback_used": bool(fallback_used),
        "sparse_multinomial_screening_status": str(screening_meta.get("status", "unknown")),
    }
    return results, {i: float(all_scores[i]) for i in range(n_features)}

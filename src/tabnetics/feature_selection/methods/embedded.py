"""Embedded feature selection methods (use sklearn estimators)."""
import logging
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from tabnetics.core.compat import make_logistic_regression
except Exception as exc:
    from tabnetics.core.compat import make_logistic_regression  # type: ignore

try:
    from boruta import BorutaPy
except Exception as exc:  # pragma: no cover
    BorutaPy = None  # type: ignore

logger = logging.getLogger(__name__)
from tabnetics.core.runtime import get_sklearn_n_jobs as _get_sklearn_n_jobs
from tabnetics.core.runtime import set_sklearn_n_jobs


def _invalid_xy_shape(X) -> bool:
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        return True
    n_samples, n_features = X_arr.shape
    return int(n_samples) < 2 or int(n_features) <= 0


def stability_selection_lasso(X, y, n_target_features, n_bootstrap_iterations, problem_type, random_state, cv_splitter_fn):
    """Perform stability selection using Lasso with bootstrap sampling."""
    n_samples, n_features = np.asarray(X).shape
    if int(n_samples) < 2 or int(n_features) == 0:
        return {}, {}

    feature_scores = np.zeros(n_features)
    feature_counts = np.zeros(n_features)

    for i in range(n_bootstrap_iterations):
        # Bootstrap sampling — use per-iteration seeded RNG for reproducibility
        _rng = np.random.default_rng(random_state + i if random_state is not None else None)
        bootstrap_indices = _rng.choice(n_samples, size=int(n_samples * 0.8), replace=True)
        X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]

        if len(np.unique(y_bootstrap)) < 2 and problem_type == 'classification':
            continue

        try:
            # Fit LassoCV with LOOCV
            cv_splitter = cv_splitter_fn(y_bootstrap)
            lasso_cv = LassoCV(cv=cv_splitter, random_state=random_state + i, max_iter=1000, n_jobs=_get_sklearn_n_jobs())
            lasso_cv.fit(X_bootstrap, y_bootstrap)

            # Get feature importance (absolute coefficients)
            coef_abs = np.abs(lasso_cv.coef_)
            selected_mask = coef_abs > 1e-5

            feature_counts[selected_mask] += 1
            feature_scores[selected_mask] += coef_abs[selected_mask]

        except Exception as e:
            logger.warning("Lasso iteration %d failed: %s", i, e)
            continue

    # Average scores
    feature_scores = np.where(feature_counts > 0, feature_scores / feature_counts, 0)

    # Select top features
    selected_indices = np.argsort(feature_scores)[::-1][:n_target_features]

    results = {
        'selected_indices': selected_indices,
        'scores': {idx: feature_scores[idx] for idx in selected_indices},
        'selection_frequency': feature_counts / n_bootstrap_iterations
    }

    return results, {i: feature_scores[i] for i in range(n_features)}


def rfe_cv_selection(X, y, n_target_features, n_bootstrap_iterations, problem_type, random_state, cv_splitter_fn):
    """Perform RFECV with multiple runs."""
    if _invalid_xy_shape(X):
        return {}, {}

    feature_rankings_sum = np.zeros(X.shape[1])
    feature_selected_count = np.zeros(X.shape[1])

    for i in range(n_bootstrap_iterations):
        # Use different random state for each iteration
        if problem_type == 'classification':
            estimator = make_logistic_regression(
                solver='saga',
                penalty='l1',
                C=0.1,
                max_iter=1000,
                n_jobs=_get_sklearn_n_jobs(),
                random_state=random_state + i,
            )
        else:
            estimator = Lasso(alpha=0.01, random_state=random_state + i, max_iter=1000)

        try:
            cv_splitter = cv_splitter_fn(y)
            selector = RFECV(estimator=estimator, step=0.1, cv=cv_splitter,
                           min_features_to_select=min(5, X.shape[1]), n_jobs=_get_sklearn_n_jobs())
            selector.fit(X, y)

            # Track rankings (lower is better)
            feature_rankings_sum += selector.ranking_
            feature_selected_count[selector.support_] += 1

        except Exception as e:
            logger.warning("RFECV iteration %d failed: %s", i, e)
            continue

    # Average rankings
    avg_rankings = feature_rankings_sum / n_bootstrap_iterations

    # Select features based on average ranking
    selected_indices = np.argsort(avg_rankings)[:n_target_features]

    results = {
        'selected_indices': selected_indices,
        'scores': {idx: 1.0 / (avg_rankings[idx] + 1) for idx in selected_indices},
        'avg_rankings': avg_rankings,
        'selection_frequency': feature_selected_count / n_bootstrap_iterations
    }

    return results, {i: 1.0 / (avg_rankings[i] + 1) for i in range(len(avg_rankings))}


def boruta_selection(X, y, n_target_features, n_bootstrap_iterations, problem_type, random_state):
    """Perform Boruta selection with multiple runs."""
    if _invalid_xy_shape(X):
        return {}, {}

    feature_importance_sum = np.zeros(X.shape[1])
    feature_selected_count = np.zeros(X.shape[1])

    for i in range(n_bootstrap_iterations):
        if problem_type == 'classification':
            rf = RandomForestClassifier(n_jobs=_get_sklearn_n_jobs(), max_depth=7,
                                      random_state=random_state + i, n_estimators=100)
        else:
            rf = RandomForestRegressor(n_jobs=_get_sklearn_n_jobs(), max_depth=7,
                                     random_state=random_state + i, n_estimators=100)

        try:
            boruta = BorutaPy(estimator=rf, n_estimators='auto',
                            random_state=random_state + i, max_iter=100)
            boruta.fit(X, y)

            # Track selections and importance
            if hasattr(boruta, 'support_'):
                feature_selected_count[boruta.support_] += 1
            if hasattr(boruta, 'feature_importances_'):
                feature_importance_sum += boruta.feature_importances_

        except Exception as e:
            logger.warning("Boruta iteration %d failed: %s", i, e)
            continue

    # Average importance
    avg_importance = feature_importance_sum / n_bootstrap_iterations

    # Select top features
    selected_indices = np.argsort(avg_importance)[::-1][:n_target_features]

    results = {
        'selected_indices': selected_indices,
        'scores': {idx: avg_importance[idx] for idx in selected_indices},
        'avg_importance': avg_importance,
        'selection_frequency': feature_selected_count / n_bootstrap_iterations
    }

    return results, {i: avg_importance[i] for i in range(len(avg_importance))}


def gradient_boosting_selection(X, y, n_target_features, n_bootstrap_iterations, problem_type, random_state):
    """Perform Gradient Boosting feature selection."""
    if _invalid_xy_shape(X):
        return {}, {}

    feature_importance_sum = np.zeros(X.shape[1])

    for i in range(n_bootstrap_iterations):
        if problem_type == 'classification':
            gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state + i)
        else:
            gb = GradientBoostingRegressor(n_estimators=100, random_state=random_state + i)

        try:
            gb.fit(X, y)
            feature_importance_sum += gb.feature_importances_
        except Exception as e:
            logger.warning("Gradient Boosting iteration %d failed: %s", i, e)
            continue

    # Average importance
    avg_importance = feature_importance_sum / n_bootstrap_iterations

    # Select top features
    selected_indices = np.argsort(avg_importance)[::-1][:n_target_features]

    results = {
        'selected_indices': selected_indices,
        'scores': {idx: avg_importance[idx] for idx in selected_indices},
        'avg_importance': avg_importance
    }

    return results, {i: avg_importance[i] for i in range(len(avg_importance))}


def linear_svm_selection(X, y, n_target_features, n_bootstrap_iterations, problem_type, random_state, linear_svm_max_iter):
    """Perform Linear SVM feature selection."""
    if _invalid_xy_shape(X):
        return {}, {}

    feature_weights_sum = np.zeros(X.shape[1])

    for i in range(n_bootstrap_iterations):
        try:
            if problem_type == 'classification':
                svm = LinearSVC(
                    penalty='l1',
                    dual=False,
                    random_state=random_state + i,
                    max_iter=linear_svm_max_iter,
                )
                svm.fit(X, y)
                weights = np.abs(svm.coef_[0])
            else:
                # Use Lasso for regression
                lasso = Lasso(alpha=0.01, random_state=random_state + i)
                lasso.fit(X, y)
                weights = np.abs(lasso.coef_)

            feature_weights_sum += weights

        except Exception as e:
            logger.warning("Linear SVM iteration %d failed: %s", i, e)
            continue

    # Average weights
    avg_weights = feature_weights_sum / n_bootstrap_iterations

    # Select top features
    selected_indices = np.argsort(avg_weights)[::-1][:n_target_features]

    results = {
        'selected_indices': selected_indices,
        'scores': {idx: avg_weights[idx] for idx in selected_indices},
        'avg_weights': avg_weights
    }

    return results, {i: avg_weights[i] for i in range(len(avg_weights))}


def joint_auc_l1_selection(X, y, n_target_features, problem_type, random_state, prefilter_fn):
    """
    Joint AUC-aware L1 feature selector (binary-only).

    Practical implementation: choose an L1-logistic regularization strength via
    CV ROC-AUC, then select features by absolute coefficient magnitude.
    """
    if problem_type != "classification" or _invalid_xy_shape(X):
        return {}, {}

    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    if classes.size != 2:
        return {}, {}

    pos_label = classes.max()
    y_bin = (y_arr == pos_label).astype(int)
    counts = np.bincount(y_bin, minlength=2)
    if int(np.min(counts)) < 2:
        return {}, {}

    n_features = int(X.shape[1])
    max_pool = int(min(n_features, max(64, min(256, 12 * int(max(1, n_target_features))))))
    pool_idx = prefilter_fn(X, y_bin, max_features=max_pool)
    if pool_idx.size == 0:
        return {}, {}

    X_pool = np.asarray(X[:, pool_idx], dtype=float)
    # Small CV for throughput; this is an opt-in method used inside MNPO.
    n_splits = int(max(2, min(5, int(np.min(counts)))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    c_grid = np.logspace(-2.0, 1.5, num=6)
    best_c = float(c_grid[0])
    best_score = float("-inf")
    score_trace: List[Tuple[float, float]] = []

    for c in c_grid:
        model = make_pipeline(
            StandardScaler(),
            make_logistic_regression(
                solver="saga",
                penalty="l1",
                C=float(c),
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=_get_sklearn_n_jobs(),
            ),
        )
        try:
            scores = cross_val_score(model, X_pool, y_bin, cv=cv, scoring="roc_auc")
            scores = np.asarray(scores, dtype=float)
            scores = scores[np.isfinite(scores)]
            mean_score = float(np.mean(scores)) if scores.size else float("-inf")
        except Exception as exc:
            mean_score = float("-inf")
        score_trace.append((float(c), float(mean_score)))
        if mean_score > best_score:
            best_score = float(mean_score)
            best_c = float(c)

    model_final = make_pipeline(
        StandardScaler(),
        make_logistic_regression(
            solver="saga",
            penalty="l1",
            C=float(best_c),
            max_iter=8000,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=_get_sklearn_n_jobs(),
        ),
    )
    try:
        model_final.fit(X_pool, y_bin)
        lr = model_final.named_steps.get("logisticregression")
        coef = getattr(lr, "coef_", None)
    except Exception as exc:
        return {}, {}

    if coef is None:
        return {}, {}

    coef = np.asarray(coef, dtype=float)
    if coef.ndim == 2:
        coef = coef[0]
    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
    abs_coef = np.abs(coef)
    if abs_coef.size != pool_idx.size:
        return {}, {}

    # Prefer sparse support, but always output exactly n_target_features when possible.
    ranked_local = np.argsort(abs_coef)[::-1]
    selected_local = ranked_local[: int(min(n_target_features, ranked_local.size))]
    selected_indices = np.asarray([int(pool_idx[int(i)]) for i in selected_local], dtype=int)

    scores_out = {int(idx): float(abs_coef[int(local_idx)]) for local_idx, idx in zip(selected_local, selected_indices)}
    all_scores = np.zeros(n_features, dtype=float)
    for local_idx, idx in zip(selected_local, selected_indices):
        all_scores[int(idx)] = float(abs_coef[int(local_idx)])
    if np.max(all_scores) > 0:
        all_scores = all_scores / float(np.max(all_scores))

    results = {
        "selected_indices": selected_indices,
        "scores": scores_out,
        "all_scores": all_scores,
        "positive_label": int(pos_label) if np.issubdtype(classes.dtype, np.integer) else str(pos_label),
        "best_c": float(best_c),
        "best_cv_auc": float(best_score) if np.isfinite(best_score) else float("nan"),
        "cv_auc_trace": [(float(c), float(score)) for c, score in score_trace],
        "pool_size": int(pool_idx.size),
    }
    return results, {i: float(all_scores[i]) for i in range(n_features)}


# ---------------------------------------------------------------------------
#  IPSS (Integrated Path Stability Selection) helpers
# ---------------------------------------------------------------------------

def bh_qvalues(p_values):
    """Benjamini-Hochberg q-value transform with monotone adjustment."""
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        return p
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


def ipss_path_grid(*, ipss_importance_model, ipss_min_c, ipss_max_c, ipss_path_grid_size):
    """Build the regularization path grid for IPSS."""
    if ipss_importance_model == "linear_svm":
        return np.logspace(
            np.log10(ipss_min_c),
            np.log10(ipss_max_c),
            num=ipss_path_grid_size,
        )
    # Tree-based models do not expose a native regularization path.
    # We emulate a path by varying quantile cutoffs over feature importances.
    return np.linspace(0.55, 0.95, num=ipss_path_grid_size)


def fit_ipss_model_importance(
    X_sub, y_sub, path_level, seed_shift, *,
    problem_type, random_state, ipss_importance_model, linear_svm_max_iter,
):
    """Fit a single IPSS model and return feature importance vector."""
    if problem_type == 'classification' and np.unique(y_sub).size < 2:
        return None

    if ipss_importance_model == "linear_svm":
        if problem_type == 'classification':
            model = LinearSVC(
                penalty='l1',
                dual=False,
                C=float(path_level),
                random_state=random_state + seed_shift,
                max_iter=linear_svm_max_iter,
            )
            model.fit(X_sub, y_sub)
            weights = np.abs(model.coef_)
            if weights.ndim == 2:
                weights = np.mean(weights, axis=0)
            return np.asarray(weights, dtype=float).ravel()

        alpha = float(np.clip(1.0 / max(1e-8, float(path_level) * X_sub.shape[0]), 1e-4, 0.25))
        model = Lasso(alpha=alpha, random_state=random_state + seed_shift, max_iter=3000)
        model.fit(X_sub, y_sub)
        return np.abs(np.asarray(model.coef_, dtype=float).ravel())

    if ipss_importance_model == "gradient_boosting":
        if problem_type == 'classification':
            model = GradientBoostingClassifier(n_estimators=120, random_state=random_state + seed_shift)
        else:
            model = GradientBoostingRegressor(n_estimators=120, random_state=random_state + seed_shift)
        model.fit(X_sub, y_sub)
        raw = np.asarray(getattr(model, "feature_importances_", np.zeros(X_sub.shape[1])), dtype=float).ravel()
    else:  # random_forest
        if problem_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=160,
                class_weight='balanced',
                n_jobs=_get_sklearn_n_jobs(),
                random_state=random_state + seed_shift,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=160,
                n_jobs=_get_sklearn_n_jobs(),
                random_state=random_state + seed_shift,
            )
        model.fit(X_sub, y_sub)
        raw = np.asarray(getattr(model, "feature_importances_", np.zeros(X_sub.shape[1])), dtype=float).ravel()

    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(raw > 0):
        return raw
    cutoff = float(np.quantile(raw, float(path_level)))
    sparse = np.clip(raw - cutoff, a_min=0.0, a_max=None)
    return sparse


def estimate_ipss_statistics(
    X_pool, y, path_grid, seed_offset=0, *,
    stability_subsample_fraction, n_bootstrap_iterations, random_state,
    problem_type, ipss_importance_model, linear_svm_max_iter, normalize_fn,
):
    """Estimate IPSS selection statistics over the regularization path."""
    n_samples, n_pool = X_pool.shape
    if n_pool == 0:
        return None

    subsample_size = int(max(4, round(stability_subsample_fraction * n_samples)))
    subsample_size = int(min(max(2, subsample_size), max(2, n_samples - 1)))
    total_rounds = int(max(2, n_bootstrap_iterations))
    rng = np.random.default_rng(random_state + seed_offset)

    n_path = len(path_grid)
    selection_counts = np.zeros((n_path, n_pool), dtype=float)
    coef_sum = np.zeros((n_path, n_pool), dtype=float)
    path_fit_counts = np.zeros(n_path, dtype=float)

    for i in range(total_rounds):
        subset = rng.choice(np.arange(n_samples), size=subsample_size, replace=False)
        complement_mask = np.ones(n_samples, dtype=bool)
        complement_mask[subset] = False
        complement = np.where(complement_mask)[0]

        for j, idx_group in enumerate([subset, complement]):
            if idx_group.size < 3:
                continue
            for p_idx, path_level in enumerate(path_grid):
                seed_shift = (2 * i + j) * n_path + p_idx + seed_offset
                try:
                    coeffs = fit_ipss_model_importance(
                        X_pool[idx_group],
                        y[idx_group],
                        path_level=path_level,
                        seed_shift=int(seed_shift),
                        problem_type=problem_type,
                        random_state=random_state,
                        ipss_importance_model=ipss_importance_model,
                        linear_svm_max_iter=linear_svm_max_iter,
                    )
                except Exception as exc:
                    coeffs = None
                if coeffs is None:
                    continue
                coeffs = np.asarray(coeffs, dtype=float).ravel()
                if coeffs.size != n_pool:
                    continue
                coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
                coef_sum[p_idx] += coeffs
                selection_counts[p_idx] += (coeffs > 1e-8).astype(float)
                path_fit_counts[p_idx] += 1.0

    valid_mask = path_fit_counts > 0
    if not np.any(valid_mask):
        return None

    selection_freq_path = np.zeros_like(selection_counts)
    selection_freq_path[valid_mask] = (
        selection_counts[valid_mask]
        / path_fit_counts[valid_mask][:, None]
    )
    mean_coef_path = np.zeros_like(coef_sum)
    mean_coef_path[valid_mask] = coef_sum[valid_mask] / path_fit_counts[valid_mask][:, None]

    path_x = np.linspace(0.0, 1.0, num=n_path)
    integrated_freq = np.trapezoid(selection_freq_path, x=path_x, axis=0)
    max_freq = np.max(selection_freq_path, axis=0)
    avg_coef = np.average(mean_coef_path, axis=0, weights=np.maximum(path_fit_counts, 1e-12))
    coef_norm = normalize_fn(avg_coef)
    integrated_score = (
        0.65 * integrated_freq
        + 0.25 * max_freq
        + 0.10 * coef_norm
    )

    return {
        "integrated_freq": integrated_freq,
        "max_freq": max_freq,
        "coef_norm": coef_norm,
        "integrated_score": integrated_score,
        "selection_freq_path": selection_freq_path,
        "path_fit_counts": path_fit_counts,
    }


def select_eats_threshold(
    integrated_scores, null_scores, *,
    ipss_eats_exclusion_quantile, ipss_eats_min_threshold, stability_selection_threshold,
):
    """Select the EATS elbow-adaptive threshold for IPSS."""
    scores = np.asarray(integrated_scores, dtype=float).ravel()
    null = np.asarray(null_scores, dtype=float).ravel()
    if scores.size == 0:
        return float(stability_selection_threshold), {}

    exclusion_floor = float(
        np.quantile(null, ipss_eats_exclusion_quantile)
    ) if null.size > 0 else float(stability_selection_threshold)
    exclusion_floor = float(np.clip(exclusion_floor, 0.0, 1.0))

    eligible = scores[scores >= exclusion_floor]
    if eligible.size < 3:
        threshold = max(
            float(ipss_eats_min_threshold),
            exclusion_floor,
            float(stability_selection_threshold),
        )
        return float(threshold), {
            "eats_exclusion_floor": float(exclusion_floor),
            "eats_elbow_threshold": float(threshold),
            "eats_n_threshold_candidates": int(eligible.size),
        }

    thresholds = np.unique(np.round(eligible, decimals=6))
    thresholds = np.sort(thresholds)[::-1]
    selected_counts = np.asarray([(scores >= t).sum() for t in thresholds], dtype=float)

    if thresholds.size < 3 or np.max(selected_counts) <= np.min(selected_counts) + 1e-9:
        elbow_threshold = float(thresholds[min(0, thresholds.size - 1)])
    else:
        x = np.linspace(0.0, 1.0, num=thresholds.size)
        y = (selected_counts - selected_counts.min()) / (selected_counts.max() - selected_counts.min() + 1e-12)
        baseline = np.linspace(y[0], y[-1], num=y.size)
        distances = np.abs(y - baseline)
        elbow_idx = int(np.argmax(distances))
        elbow_threshold = float(thresholds[elbow_idx])

    threshold = max(
        float(ipss_eats_min_threshold),
        float(exclusion_floor),
        float(elbow_threshold),
    )
    threshold = float(np.clip(threshold, 0.0, 1.0))
    return threshold, {
        "eats_exclusion_floor": float(exclusion_floor),
        "eats_elbow_threshold": float(elbow_threshold),
        "eats_n_threshold_candidates": int(thresholds.size),
    }


def ipss_selection(
    X, y, n_target_features, *,
    problem_type, random_state, mrmr_max_features, prefilter_fn, normalize_fn,
    stability_subsample_fraction, n_bootstrap_iterations,
    ipss_importance_model, ipss_min_c, ipss_max_c, ipss_path_grid_size,
    linear_svm_max_iter,
    ipss_null_shuffle_rounds, ipss_use_eats_threshold, ipss_target_fdr,
    ipss_eats_exclusion_quantile, ipss_eats_min_threshold, stability_selection_threshold,
    ipss_gate_min_classes, ipss_gate_min_p_over_n,
):
    """
    Integrated path stability selection with optional EATS threshold calibration.
    """
    n_samples, n_features = np.asarray(X).shape
    if int(n_samples) < 2 or int(n_features) == 0:
        return {}, {}

    gate_min_classes = int(max(0, ipss_gate_min_classes))
    gate_min_p_over_n = float(max(0.0, ipss_gate_min_p_over_n))
    use_class_gate = (problem_type == "classification") and (gate_min_classes > 0)
    use_ratio_gate = gate_min_p_over_n > 0.0
    if use_class_gate or use_ratio_gate:
        y_arr = np.asarray(y)
        n_classes = int(np.unique(y_arr).size) if problem_type == "classification" else 0
        p_over_n = float(n_features) / float(max(1, n_samples))
        passes = []
        if use_class_gate:
            passes.append(n_classes >= gate_min_classes)
        if use_ratio_gate:
            passes.append(p_over_n >= gate_min_p_over_n)
        if not any(passes):
            results = {
                "selected_indices": np.array([], dtype=int),
                "scores": {},
                "method": "ipss",
                "ipss_gated": True,
                "ipss_gate_reason": "gate_not_satisfied",
                "ipss_gate_min_classes": int(gate_min_classes),
                "ipss_gate_min_p_over_n": float(gate_min_p_over_n),
                "ipss_gate_n_classes": int(n_classes),
                "ipss_gate_p_over_n": float(p_over_n),
            }
            return results, {}

    pool_cap = int(
        min(
            n_features,
            max(mrmr_max_features, min(640, max(96, 8 * int(max(1, n_target_features))))),
        )
    )
    pool_idx = prefilter_fn(X, y, max_features=pool_cap)
    if pool_idx.size == 0:
        return {}, {}

    X_pool = X[:, pool_idx]
    path_grid = ipss_path_grid(
        ipss_importance_model=ipss_importance_model,
        ipss_min_c=ipss_min_c,
        ipss_max_c=ipss_max_c,
        ipss_path_grid_size=ipss_path_grid_size,
    )
    observed = estimate_ipss_statistics(
        X_pool, y, path_grid=path_grid, seed_offset=0,
        stability_subsample_fraction=stability_subsample_fraction,
        n_bootstrap_iterations=n_bootstrap_iterations,
        random_state=random_state,
        problem_type=problem_type,
        ipss_importance_model=ipss_importance_model,
        linear_svm_max_iter=linear_svm_max_iter,
        normalize_fn=normalize_fn,
    )
    if observed is None:
        return {}, {}

    integrated_score_pool = np.asarray(observed["integrated_score"], dtype=float).ravel()
    integrated_freq_pool = np.asarray(observed["integrated_freq"], dtype=float).ravel()
    max_freq_pool = np.asarray(observed["max_freq"], dtype=float).ravel()

    rng = np.random.default_rng(random_state + 7919)
    null_scores_parts = []
    for rep in range(ipss_null_shuffle_rounds):
        y_perm = rng.permutation(y)
        null_est = estimate_ipss_statistics(
            X_pool,
            y_perm,
            path_grid=path_grid,
            seed_offset=1000 + rep,
            stability_subsample_fraction=stability_subsample_fraction,
            n_bootstrap_iterations=n_bootstrap_iterations,
            random_state=random_state,
            problem_type=problem_type,
            ipss_importance_model=ipss_importance_model,
            linear_svm_max_iter=linear_svm_max_iter,
            normalize_fn=normalize_fn,
        )
        if null_est is None:
            continue
        null_scores_parts.append(np.asarray(null_est["integrated_score"], dtype=float).ravel())
    null_scores = (
        np.concatenate(null_scores_parts)
        if null_scores_parts
        else np.array([], dtype=float)
    )

    if null_scores.size > 0:
        p_values = np.array(
            [
                (1.0 + float(np.sum(null_scores >= s))) / (1.0 + float(null_scores.size))
                for s in integrated_score_pool
            ],
            dtype=float,
        )
    else:
        p_values = np.clip(1.0 - integrated_score_pool, 0.0, 1.0)
    q_values_pool = bh_qvalues(p_values)

    if ipss_use_eats_threshold:
        stable_threshold, eats_meta = select_eats_threshold(
            integrated_score_pool, null_scores,
            ipss_eats_exclusion_quantile=ipss_eats_exclusion_quantile,
            ipss_eats_min_threshold=ipss_eats_min_threshold,
            stability_selection_threshold=stability_selection_threshold,
        )
    else:
        stable_threshold = float(stability_selection_threshold)
        eats_meta = {
            "eats_exclusion_floor": float("nan"),
            "eats_elbow_threshold": float("nan"),
            "eats_n_threshold_candidates": 0,
        }

    stable_idx = np.where(
        (integrated_score_pool >= stable_threshold)
        & (q_values_pool <= ipss_target_fdr)
    )[0]
    if stable_idx.size == 0:
        stable_idx = np.where(integrated_score_pool >= stable_threshold)[0]

    rank_signal = integrated_score_pool - 0.15 * q_values_pool
    stable_ranked = (
        stable_idx[np.argsort(rank_signal[stable_idx])[::-1]]
        if stable_idx.size > 0
        else np.array([], dtype=int)
    )
    ranked_all = np.argsort(rank_signal)[::-1]

    selected_local = []
    for idx in stable_ranked:
        selected_local.append(int(idx))
        if len(selected_local) >= n_target_features:
            break
    if len(selected_local) < n_target_features:
        for idx in ranked_all:
            idx = int(idx)
            if idx not in selected_local:
                selected_local.append(idx)
            if len(selected_local) >= n_target_features:
                break

    selected_indices = pool_idx[np.array(selected_local[:n_target_features], dtype=int)]

    integrated_score = np.zeros(n_features, dtype=float)
    integrated_score[pool_idx] = integrated_score_pool
    integrated_frequency = np.zeros(n_features, dtype=float)
    integrated_frequency[pool_idx] = integrated_freq_pool
    max_frequency = np.zeros(n_features, dtype=float)
    max_frequency[pool_idx] = max_freq_pool
    q_values = np.ones(n_features, dtype=float)
    q_values[pool_idx] = q_values_pool

    results = {
        "selected_indices": selected_indices,
        "scores": {int(idx): float(integrated_score[idx]) for idx in selected_indices},
        "selection_frequency": integrated_frequency,
        "stability_score": integrated_score,
        "selection_frequency_max": max_frequency,
        "q_values": q_values,
        "n_fits": int(np.sum(observed["path_fit_counts"])),
        "pool_size": int(pool_idx.size),
        "path_grid": [float(v) for v in np.asarray(path_grid, dtype=float).tolist()],
        "path_fit_counts": [int(v) for v in np.asarray(observed["path_fit_counts"], dtype=float).tolist()],
        "stable_threshold": float(stable_threshold),
        "target_fdr": float(ipss_target_fdr),
        "n_null_scores": int(null_scores.size),
        "ipss_importance_model": ipss_importance_model,
        "ipss_use_eats_threshold": bool(ipss_use_eats_threshold),
        "ipss_gated": False,
        "ipss_gate_reason": "",
        "ipss_gate_min_classes": int(gate_min_classes),
        "ipss_gate_min_p_over_n": float(gate_min_p_over_n),
        "ipss_gate_n_classes": int(np.unique(np.asarray(y)).size) if problem_type == "classification" else 0,
        "ipss_gate_p_over_n": float(n_features) / float(max(1, n_samples)),
        **eats_meta,
    }
    return results, {i: float(integrated_score[i]) for i in range(n_features)}


def ipss_benchmark_evaluate(
    ipss_result,
    *,
    y_true=None,
    y_pred=None,
):
    """Extract benchmark-mode diagnostics from an IPSS result dict.

    This is a post-hoc analysis function (VAL12_Suggestions §5.3) that
    takes the result dict returned by ``ipss_selection`` and computes
    additional benchmark metrics.

    Parameters
    ----------
    ipss_result : dict
        Result dict from ``ipss_selection``.
    y_true : array-like or None
        True labels (for BA computation).
    y_pred : array-like or None
        Predicted labels (for BA computation).

    Returns
    -------
    report : dict
        Benchmark report with keys:
        - ``selected_count``: number of selected features
        - ``stable_threshold``: threshold used for selection
        - ``target_fdr``: target FDR
        - ``eats_calibrated``: whether EATS was used
        - ``eats_elbow_threshold``: EATS elbow if used
        - ``pool_size``: size of the pre-filtered candidate pool
        - ``selection_frequency_stats``: summary stats of selection frequencies
        - ``balanced_accuracy``: BA if y_true/y_pred provided
    """
    if not ipss_result or not isinstance(ipss_result, dict):
        return {"error": "empty_or_invalid_ipss_result"}

    report = {
        "selected_count": int(
            len(ipss_result.get("selected_indices", []))
        ),
        "stable_threshold": float(ipss_result.get("stable_threshold", 0.0)),
        "target_fdr": float(ipss_result.get("target_fdr", 0.0)),
        "ipss_use_eats_threshold": bool(
            ipss_result.get("ipss_use_eats_threshold", False)
        ),
        "eats_elbow_threshold": float(
            ipss_result.get("eats_elbow_threshold", float("nan"))
        ),
        "pool_size": int(ipss_result.get("pool_size", 0)),
        "n_fits": int(ipss_result.get("n_fits", 0)),
        "ipss_gated": bool(ipss_result.get("ipss_gated", False)),
        "ipss_gate_reason": str(ipss_result.get("ipss_gate_reason", "")),
    }

    # Selection frequency summary.
    sel_freq = ipss_result.get("selection_frequency", None)
    if sel_freq is not None:
        sel_freq_arr = np.asarray(sel_freq, dtype=float)
        nonzero = sel_freq_arr[sel_freq_arr > 0]
        report["selection_frequency_stats"] = {
            "mean": float(np.mean(nonzero)) if nonzero.size > 0 else 0.0,
            "median": float(np.median(nonzero)) if nonzero.size > 0 else 0.0,
            "max": float(np.max(nonzero)) if nonzero.size > 0 else 0.0,
            "n_nonzero": int(nonzero.size),
        }
    else:
        report["selection_frequency_stats"] = {}

    # Q-value summary if available.
    q_vals = ipss_result.get("q_values", None)
    if q_vals is not None:
        q_arr = np.asarray(q_vals, dtype=float)
        selected_idx = ipss_result.get("selected_indices", np.array([], dtype=int))
        if len(selected_idx) > 0:
            sel_q = q_arr[selected_idx]
            report["selected_q_value_stats"] = {
                "mean": float(np.mean(sel_q)),
                "median": float(np.median(sel_q)),
                "max": float(np.max(sel_q)),
                "fdr_estimate": float(np.mean(sel_q)),
            }
        else:
            report["selected_q_value_stats"] = {}
    else:
        report["selected_q_value_stats"] = {}

    # Balanced accuracy if labels provided.
    if y_true is not None and y_pred is not None:
        from sklearn.metrics import balanced_accuracy_score
        try:
            ba = balanced_accuracy_score(
                np.asarray(y_true).ravel(),
                np.asarray(y_pred).ravel(),
            )
            report["balanced_accuracy"] = float(ba)
        except Exception:
            report["balanced_accuracy"] = float("nan")

    return report


def treeshap_selection(
    X,
    y,
    n_target_features,
    *,
    problem_type,
    random_state,
    n_bootstrap_iterations,
    min_samples=50,
    n_estimators=200,
    max_depth=None,
    multi_seed_runs=3,
):
    """TreeSHAP selector with robust fallback when SHAP is unavailable."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()
    if X_arr.ndim != 2:
        return {}, {}
    n_samples, n_features = X_arr.shape
    if n_samples < int(max(2, min_samples)) or n_features <= 0:
        return (
            {
                "selected_indices": np.array([], dtype=int),
                "scores": {},
                "method": "treeshap",
                "skipped": True,
                "skip_reason": "min_n_gate",
                "min_samples": int(min_samples),
                "n_samples": int(n_samples),
            },
            {},
        )

    def _normalize(v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=float).ravel()
        if arr.size == 0:
            return arr
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        rng = hi - lo
        if rng <= 1e-12:
            return np.zeros_like(arr)
        return (arr - lo) / rng

    use_shap = False
    shap_module = None
    try:
        import shap as _shap  # type: ignore

        shap_module = _shap
        use_shap = True
    except Exception as exc:
        use_shap = False

    score_accum = np.zeros(n_features, dtype=float)
    valid_runs = 0
    n_runs = int(max(1, multi_seed_runs))
    for run in range(n_runs):
        seed = int(random_state + 97 * run)
        try:
            if str(problem_type).lower() == "classification":
                model = RandomForestClassifier(
                    n_estimators=int(max(50, n_estimators)),
                    random_state=seed,
                    class_weight="balanced_subsample",
                    n_jobs=_get_sklearn_n_jobs(),
                    max_depth=max_depth,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=int(max(50, n_estimators)),
                    random_state=seed,
                    n_jobs=_get_sklearn_n_jobs(),
                    max_depth=max_depth,
                )
            model.fit(X_arr, y_arr)

            run_scores = None
            if use_shap and shap_module is not None:
                try:
                    explainer = shap_module.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_arr)
                    if isinstance(shap_values, list):
                        mats = [np.asarray(v, dtype=float) for v in shap_values]
                        mats = [m for m in mats if m.ndim == 2 and m.shape[1] == n_features]
                        if mats:
                            run_scores = np.mean(np.mean(np.abs(np.stack(mats, axis=0)), axis=0), axis=0)
                    else:
                        m = np.asarray(shap_values, dtype=float)
                        if m.ndim == 3 and m.shape[-1] == n_features:
                            run_scores = np.mean(np.abs(m), axis=(0, 1))
                        elif m.ndim == 2 and m.shape[1] == n_features:
                            run_scores = np.mean(np.abs(m), axis=0)
                except Exception as exc:
                    run_scores = None

            if run_scores is None:
                run_scores = np.asarray(
                    getattr(model, "feature_importances_", np.zeros(n_features, dtype=float)),
                    dtype=float,
                ).ravel()
            if run_scores.size != n_features:
                continue
            score_accum += np.asarray(run_scores, dtype=float)
            valid_runs += 1
        except Exception as exc:
            continue

    if valid_runs <= 0:
        return {}, {}
    avg_scores = score_accum / float(valid_runs)
    avg_scores = _normalize(avg_scores)
    selected_indices = np.argsort(avg_scores)[::-1][: int(max(1, min(n_target_features, n_features)))]
    results = {
        "selected_indices": np.asarray(selected_indices, dtype=int),
        "scores": {int(i): float(avg_scores[i]) for i in np.asarray(selected_indices, dtype=int).tolist()},
        "method": "treeshap",
        "shap_available": bool(use_shap),
        "multi_seed_runs": int(n_runs),
        "valid_runs": int(valid_runs),
        "min_samples": int(min_samples),
    }
    return results, {int(i): float(avg_scores[i]) for i in range(n_features)}


def oaenet_adaptive_selection(
    X,
    y,
    n_target_features,
    *,
    problem_type,
    random_state,
    min_samples=40,
    prescreen_max_features=512,
    l1_ratio=0.5,
    c_grid_size=6,
):
    """Outcome-adaptive elastic-net selector (no causal claims)."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()
    if X_arr.ndim != 2:
        return {}, {}
    n_samples, n_features = X_arr.shape
    if n_features <= 0:
        return {}, {}
    if n_samples < int(max(2, min_samples)):
        return (
            {
                "selected_indices": np.array([], dtype=int),
                "scores": {},
                "method": "oaenet",
                "skipped": True,
                "skip_reason": "min_n_gate",
                "min_samples": int(min_samples),
                "n_samples": int(n_samples),
            },
            {},
        )

    def _normalize(v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=float).ravel()
        if arr.size == 0:
            return arr
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        rng = hi - lo
        if rng <= 1e-12:
            return np.zeros_like(arr)
        return (arr - lo) / rng

    from sklearn.feature_selection import f_classif, mutual_info_classif

    # Prescreen to cap runtime on extreme HDLSS pools.
    pool_cap = int(max(16, min(n_features, prescreen_max_features)))
    try:
        mi = np.asarray(
            mutual_info_classif(X_arr, y_arr, random_state=int(random_state)),
            dtype=float,
        ).ravel()
    except Exception as exc:
        mi = np.zeros(n_features, dtype=float)
    try:
        fvals, _ = f_classif(X_arr, y_arr)
        fvals = np.asarray(fvals, dtype=float).ravel()
    except Exception as exc:
        fvals = np.zeros(n_features, dtype=float)
    relevance = 0.5 * _normalize(mi) + 0.5 * _normalize(fvals)
    prescreen_idx = np.argsort(relevance)[::-1][:pool_cap]
    prescreen_idx = np.asarray(sorted(set(int(i) for i in prescreen_idx)), dtype=int)
    if prescreen_idx.size == 0:
        return {}, {}

    X_pool = np.asarray(X_arr[:, prescreen_idx], dtype=float)
    rel_pool = np.asarray(relevance[prescreen_idx], dtype=float)
    # Outcome-adaptive penalty weights: stronger shrinkage on weakly relevant features.
    weights = 1.0 / np.maximum(1e-6, rel_pool + 1e-3)
    weights = np.asarray(weights / max(1e-12, np.median(weights)), dtype=float)
    X_weighted = X_pool / weights[None, :]

    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.size < 2:
        return {}, {}
    n_splits = int(max(2, min(5, int(np.min(counts)))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    c_grid = np.logspace(-2.0, 1.2, num=int(max(3, c_grid_size)))

    def _make_oaenet_estimator(c_val: float, *, max_iter: int):
        base = make_logistic_regression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=float(np.clip(l1_ratio, 0.05, 0.95)),
            C=float(c_val),
            max_iter=int(max_iter),
            class_weight="balanced",
            random_state=int(random_state),
            n_jobs=_get_sklearn_n_jobs(),
        )
        if int(classes.size) > 2:
            return OneVsRestClassifier(base, n_jobs=_get_sklearn_n_jobs())
        return base

    best_c = float(c_grid[0])
    best_score = float("-inf")
    for c_val in c_grid:
        model = _make_oaenet_estimator(float(c_val), max_iter=6000)
        try:
            scores = np.asarray(
                cross_val_score(model, X_weighted, y_arr, cv=cv, scoring="balanced_accuracy"),
                dtype=float,
            ).ravel()
            scores = scores[np.isfinite(scores)]
            if scores.size == 0:
                continue
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_c = float(c_val)
        except Exception as exc:
            continue

    final = _make_oaenet_estimator(float(best_c), max_iter=8000)
    try:
        final.fit(X_weighted, y_arr)
        if hasattr(final, "coef_"):
            coef = np.asarray(final.coef_, dtype=float)
        else:
            # OneVsRestClassifier path (multi-class): aggregate per-binary model coefficients.
            estimators = list(getattr(final, "estimators_", []) or [])
            coef_rows = []
            for est in estimators:
                coef_est = np.asarray(getattr(est, "coef_", np.array([])), dtype=float)
                if coef_est.size == 0:
                    continue
                if coef_est.ndim == 1:
                    coef_rows.append(coef_est.reshape(1, -1))
                elif coef_est.ndim == 2:
                    coef_rows.append(coef_est)
            if not coef_rows:
                return {}, {}
            coef = np.vstack(coef_rows)
    except Exception as exc:
        return {}, {}

    if coef.ndim == 2:
        coef_abs = np.mean(np.abs(coef), axis=0)
    else:
        coef_abs = np.abs(np.asarray(coef, dtype=float).ravel())
    if coef_abs.size != prescreen_idx.size:
        return {}, {}

    # Undo feature scaling before ranking importance.
    score_pool = np.asarray(coef_abs / np.maximum(1e-8, weights), dtype=float)
    score_pool = _normalize(score_pool)
    full_scores = np.zeros(n_features, dtype=float)
    full_scores[prescreen_idx] = score_pool
    selected = np.argsort(full_scores)[::-1][: int(max(1, min(n_target_features, n_features)))]
    selected = np.asarray(selected, dtype=int)

    results = {
        "selected_indices": selected,
        "scores": {int(i): float(full_scores[i]) for i in selected.tolist()},
        "method": "oaenet",
        "prescreen_size": int(prescreen_idx.size),
        "prescreen_cap": int(pool_cap),
        "best_c": float(best_c),
        "best_cv_balanced_accuracy": float(best_score) if np.isfinite(best_score) else float("nan"),
    }
    return results, {int(i): float(full_scores[i]) for i in range(n_features)}

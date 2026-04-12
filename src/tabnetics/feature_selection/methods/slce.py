"""SLCE (Supervised Linear Centroid Encoder) feature selection."""

from __future__ import annotations

import numpy as np


def slce_centroid_encoder_selection(
    X,
    y,
    n_target_features,
    *,
    problem_type,
    random_state,
    slce_prefilter_max_features,
    slce_min_samples,
    slce_ridge,
    prefilter_fn,
    normalize_fn,
):
    """Binary SLCE selector using a ridge-regularized centroid-separation direction."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()
    if X_arr.ndim != 2:
        return {}, {}
    n_samples, n_features = X_arr.shape
    if int(n_samples) < 2 or int(n_features) <= 0:
        return {}, {}
    if str(problem_type).strip().lower() != "classification":
        return {}, {}

    classes = np.unique(y_arr)
    if classes.size != 2:
        return {}, {}
    if int(n_samples) < int(max(2, slce_min_samples)):
        return {}, {}

    n_target = int(min(max(1, n_target_features), n_features))
    pool_cap = int(min(n_features, max(n_target, int(slce_prefilter_max_features))))
    pool_idx = np.asarray(prefilter_fn(X_arr, y_arr, max_features=pool_cap), dtype=int).ravel()
    if pool_idx.size == 0:
        return {}, {}

    X_pool = np.asarray(X_arr[:, pool_idx], dtype=float)
    X_pool = np.nan_to_num(X_pool, nan=0.0, posinf=0.0, neginf=0.0)
    col_mu = np.asarray(np.mean(X_pool, axis=0), dtype=float).ravel()
    col_sd = np.asarray(np.std(X_pool, axis=0, ddof=1), dtype=float).ravel()
    col_sd = np.where(np.isfinite(col_sd) & (col_sd > 1e-8), col_sd, 1.0)
    X_pool = (X_pool - col_mu[None, :]) / col_sd[None, :]

    c0, c1 = classes.tolist()
    m0 = np.asarray(y_arr == c0, dtype=bool)
    m1 = np.asarray(y_arr == c1, dtype=bool)
    n0 = int(np.sum(m0))
    n1 = int(np.sum(m1))
    if n0 < 1 or n1 < 1:
        return {}, {}

    mu0 = np.asarray(np.mean(X_pool[m0], axis=0), dtype=float).ravel()
    mu1 = np.asarray(np.mean(X_pool[m1], axis=0), dtype=float).ravel()
    d = np.asarray(mu1 - mu0, dtype=float).ravel()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.linalg.norm(d)) <= 1e-12:
        return {}, {}

    # Within-class centered design for ridge-regularized scatter inversion.
    Xw = np.asarray(X_pool, dtype=float).copy()
    Xw[m0] = X_pool[m0] - mu0[None, :]
    Xw[m1] = X_pool[m1] - mu1[None, :]
    lam = float(max(1e-8, slce_ridge))
    n_float = float(max(1, n_samples))
    sqrt_n = float(np.sqrt(n_float))

    inv_lam_d = d / lam
    gram = np.asarray((Xw @ Xw.T) / (n_float * lam), dtype=float)
    B = np.eye(n_samples, dtype=float) + gram
    rhs = np.asarray((Xw @ inv_lam_d) / sqrt_n, dtype=float).ravel()

    try:
        alpha = np.linalg.solve(B, rhs)
        w = inv_lam_d - (Xw.T @ alpha) / (sqrt_n * lam)
        solver = "woodbury"
    except Exception as exc:
        # Fallback path is rarely needed but keeps the method robust.
        A = (Xw.T @ Xw) / n_float + lam * np.eye(Xw.shape[1], dtype=float)
        w = np.linalg.solve(A, d)
        solver = "direct"

    w = np.asarray(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), dtype=float).ravel()
    score_pool = np.abs(w)
    if not np.any(score_pool > 0):
        score_pool = np.abs(d)
    score_pool = np.asarray(normalize_fn(score_pool), dtype=float).ravel()

    selected_local = np.argsort(score_pool)[::-1][:n_target]
    selected_indices = np.asarray(pool_idx[selected_local], dtype=int)
    if selected_indices.size == 0:
        return {}, {}

    all_scores = np.zeros(n_features, dtype=float)
    all_scores[pool_idx] = score_pool
    all_scores = np.asarray(normalize_fn(all_scores), dtype=float).ravel()

    results = {
        "selected_indices": selected_indices,
        "scores": {
            int(idx): float(all_scores[int(idx)])
            for idx in selected_indices.tolist()
        },
        "method": "slce_centroid_encoder",
        "slce_pool_size": int(pool_idx.size),
        "slce_ridge": float(lam),
        "slce_solver": str(solver),
        "slce_min_samples": int(max(2, slce_min_samples)),
        "slce_class_counts": {
            int(c0) if np.issubdtype(classes.dtype, np.integer) else str(c0): int(n0),
            int(c1) if np.issubdtype(classes.dtype, np.integer) else str(c1): int(n1),
        },
    }
    all_scores_dict = {i: float(all_scores[i]) for i in range(n_features)}
    return results, all_scores_dict

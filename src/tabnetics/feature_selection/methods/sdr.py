"""Classical sufficient dimension-reduction (SDR) feature selectors.

Implements opt-in SIR/SAVE/PFC scoring for multiclass HDLSS settings.
These methods return standard method payloads compatible with FeatureSelector.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def _invalid_xy_shape(X: np.ndarray) -> bool:
    arr = np.asarray(X)
    if arr.ndim != 2:
        return True
    n_samples, n_features = arr.shape
    return int(n_samples) < 4 or int(n_features) <= 0


def _safe_normalize(scores: np.ndarray, normalize_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    arr = np.asarray(scores, dtype=float).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    try:
        norm = np.asarray(normalize_fn(arr), dtype=float).ravel()
        if norm.size == arr.size:
            return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        pass
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if (not np.isfinite(mn)) or (not np.isfinite(mx)) or (mx - mn) < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / max(1e-12, mx - mn)


def _candidate_pool(
    X: np.ndarray,
    y: np.ndarray,
    *,
    prefilter_fn: Callable[..., np.ndarray],
    prefilter_max_features: int,
) -> np.ndarray:
    n_features = int(np.asarray(X).shape[1])
    cap = int(max(4, min(n_features, int(prefilter_max_features))))
    try:
        cand = np.asarray(prefilter_fn(X, y, max_features=cap), dtype=int).ravel()
    except Exception as exc:
        cand = np.arange(cap, dtype=int)
    if cand.size <= 0:
        cand = np.arange(cap, dtype=int)
    cand = cand[(cand >= 0) & (cand < n_features)]
    cand = np.unique(cand)
    if cand.size < 2:
        cand = np.arange(min(n_features, max(2, cap)), dtype=int)
    return np.asarray(cand, dtype=int)


def _safe_covariance(X: np.ndarray, *, ridge: float) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    p = int(arr.shape[1])
    if p <= 0:
        return np.zeros((0, 0), dtype=float)
    if arr.shape[0] <= 1:
        return np.eye(p, dtype=float) * float(max(1e-8, ridge))
    cov = np.asarray(np.cov(arr, rowvar=False, ddof=1), dtype=float)
    if cov.ndim != 2 or cov.shape[0] != p or cov.shape[1] != p:
        cov = np.eye(p, dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov += np.eye(p, dtype=float) * float(max(1e-8, ridge))
    return cov


def _solve_generalized_sdr(
    kernel: np.ndarray,
    covariance: np.ndarray,
    *,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray]:
    k_mat = np.asarray(kernel, dtype=float)
    s_mat = np.asarray(covariance, dtype=float)
    p = int(k_mat.shape[0])
    if p <= 0:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)
    if k_mat.shape != (p, p) or s_mat.shape != (p, p):
        return np.zeros((p, 0), dtype=float), np.zeros(0, dtype=float)

    pinv_s = np.linalg.pinv(s_mat)
    mat = pinv_s @ k_mat
    mat = 0.5 * (mat + mat.T)
    evals, evecs = np.linalg.eigh(mat)
    order = np.argsort(evals)[::-1]
    keep = int(max(1, min(int(n_components), p)))
    order = order[:keep]
    evals_keep = np.asarray(evals[order], dtype=float)
    evecs_keep = np.asarray(evecs[:, order], dtype=float)
    return evecs_keep, evals_keep


def _direction_scores(directions: np.ndarray, eigvals: np.ndarray) -> np.ndarray:
    vecs = np.asarray(directions, dtype=float)
    vals = np.asarray(eigvals, dtype=float).ravel()
    if vecs.ndim != 2 or vecs.shape[0] <= 0 or vecs.shape[1] <= 0:
        return np.zeros(max(0, vecs.shape[0] if vecs.ndim == 2 else 0), dtype=float)
    w = np.clip(vals, a_min=0.0, a_max=None)
    if w.size != vecs.shape[1] or float(np.sum(w)) <= 1e-12:
        w = np.ones(vecs.shape[1], dtype=float)
    weighted = np.abs(vecs) * w[None, :]
    return np.asarray(np.sum(weighted, axis=1), dtype=float).ravel()


def _finalize_result(
    pooled_scores: np.ndarray,
    candidate_idx: np.ndarray,
    *,
    n_target_features: int,
    n_features_total: int,
    normalize_fn: Callable[[np.ndarray], np.ndarray],
    method_name: str,
) -> Tuple[Dict[str, Any], Dict[int, float]]:
    scores_pool = _safe_normalize(pooled_scores, normalize_fn)
    candidate = np.asarray(candidate_idx, dtype=int).ravel()
    if scores_pool.size != candidate.size:
        return {}, {}

    top_k = int(max(1, min(int(n_target_features), int(candidate.size))))
    order = np.argsort(scores_pool)[::-1][:top_k]
    selected = np.asarray(candidate[order], dtype=int).ravel()

    all_scores = np.zeros(int(max(0, n_features_total)), dtype=float)
    all_scores[candidate] = np.asarray(scores_pool, dtype=float)

    result = {
        "selected_indices": selected,
        "scores": {int(i): float(all_scores[int(i)]) for i in selected.tolist()},
        "all_scores": np.asarray(all_scores, dtype=float),
        "sdr_method": str(method_name),
    }
    all_scores_dict = {int(i): float(all_scores[i]) for i in range(int(all_scores.size))}
    return result, all_scores_dict


def _prepare_sdr_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    problem_type: str,
    sdr_min_classes: int,
    prefilter_fn: Callable[..., np.ndarray],
    prefilter_max_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if str(problem_type) != "classification":
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()
    if _invalid_xy_shape(arr):
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    classes, y_codes = np.unique(y_arr, return_inverse=True)
    if int(classes.size) < int(max(2, sdr_min_classes)):
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    candidate_idx = _candidate_pool(
        arr,
        y_arr,
        prefilter_fn=prefilter_fn,
        prefilter_max_features=int(prefilter_max_features),
    )
    X_pool = np.asarray(arr[:, candidate_idx], dtype=float)
    X_pool = np.asarray(np.nan_to_num(X_pool, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    if X_pool.ndim != 2 or X_pool.shape[0] < 4 or X_pool.shape[1] <= 0:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    return X_pool, np.asarray(y_codes, dtype=int), candidate_idx


def sir_sdr_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_target_features: int,
    *,
    problem_type: str,
    sdr_min_classes: int,
    sdr_prefilter_max_features: int,
    sdr_n_components: int,
    sdr_covariance_ridge: float,
    prefilter_fn: Callable[..., np.ndarray],
    normalize_fn: Callable[[np.ndarray], np.ndarray],
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[int, float]]:
    del random_state  # Deterministic implementation.
    X_pool, y_codes, candidate_idx = _prepare_sdr_data(
        X,
        y,
        problem_type=problem_type,
        sdr_min_classes=sdr_min_classes,
        prefilter_fn=prefilter_fn,
        prefilter_max_features=sdr_prefilter_max_features,
    )
    if X_pool.size <= 0:
        return {}, {}

    n_samples, p = X_pool.shape
    classes = np.unique(y_codes)
    mu = np.asarray(np.mean(X_pool, axis=0), dtype=float).ravel()

    sir_kernel = np.zeros((p, p), dtype=float)
    for cls in classes.tolist():
        mask = y_codes == int(cls)
        n_h = int(np.sum(mask))
        if n_h <= 0:
            continue
        p_h = float(n_h) / float(max(1, n_samples))
        mu_h = np.asarray(np.mean(X_pool[mask], axis=0), dtype=float).ravel()
        d = mu_h - mu
        sir_kernel += p_h * np.outer(d, d)

    cov = _safe_covariance(X_pool, ridge=float(sdr_covariance_ridge))
    n_dirs = int(max(1, min(int(sdr_n_components), int(classes.size - 1), p)))
    dirs, evals = _solve_generalized_sdr(sir_kernel, cov, n_components=n_dirs)
    pooled_scores = _direction_scores(dirs, evals)
    return _finalize_result(
        pooled_scores,
        candidate_idx,
        n_target_features=n_target_features,
        n_features_total=int(np.asarray(X).shape[1]),
        normalize_fn=normalize_fn,
        method_name="sir",
    )


def save_sdr_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_target_features: int,
    *,
    problem_type: str,
    sdr_min_classes: int,
    sdr_prefilter_max_features: int,
    sdr_n_components: int,
    sdr_covariance_ridge: float,
    prefilter_fn: Callable[..., np.ndarray],
    normalize_fn: Callable[[np.ndarray], np.ndarray],
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[int, float]]:
    del random_state  # Deterministic implementation.
    X_pool, y_codes, candidate_idx = _prepare_sdr_data(
        X,
        y,
        problem_type=problem_type,
        sdr_min_classes=sdr_min_classes,
        prefilter_fn=prefilter_fn,
        prefilter_max_features=sdr_prefilter_max_features,
    )
    if X_pool.size <= 0:
        return {}, {}

    n_samples, p = X_pool.shape
    classes = np.unique(y_codes)
    cov = _safe_covariance(X_pool, ridge=float(sdr_covariance_ridge))

    evals_cov, evecs_cov = np.linalg.eigh(cov)
    evals_cov = np.clip(np.asarray(evals_cov, dtype=float), a_min=1e-8, a_max=None)
    whitening = np.asarray(
        evecs_cov @ np.diag(1.0 / np.sqrt(evals_cov)) @ evecs_cov.T,
        dtype=float,
    )
    mu = np.asarray(np.mean(X_pool, axis=0), dtype=float).ravel()
    Z = np.asarray((X_pool - mu[None, :]) @ whitening, dtype=float)

    I = np.eye(p, dtype=float)
    save_kernel = np.zeros((p, p), dtype=float)
    for cls in classes.tolist():
        mask = y_codes == int(cls)
        n_h = int(np.sum(mask))
        if n_h <= 1:
            continue
        p_h = float(n_h) / float(max(1, n_samples))
        cov_h = _safe_covariance(Z[mask], ridge=float(sdr_covariance_ridge))
        d = I - cov_h
        save_kernel += p_h * (d @ d)

    n_dirs = int(max(1, min(int(sdr_n_components), int(classes.size - 1), p)))
    dirs_z, evals = _solve_generalized_sdr(save_kernel, np.eye(p, dtype=float), n_components=n_dirs)
    dirs_orig = np.asarray(whitening @ dirs_z, dtype=float)
    pooled_scores = _direction_scores(dirs_orig, evals)
    return _finalize_result(
        pooled_scores,
        candidate_idx,
        n_target_features=n_target_features,
        n_features_total=int(np.asarray(X).shape[1]),
        normalize_fn=normalize_fn,
        method_name="save",
    )


def pfc_sdr_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_target_features: int,
    *,
    problem_type: str,
    sdr_min_classes: int,
    sdr_prefilter_max_features: int,
    sdr_n_components: int,
    sdr_covariance_ridge: float,
    prefilter_fn: Callable[..., np.ndarray],
    normalize_fn: Callable[[np.ndarray], np.ndarray],
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[int, float]]:
    del random_state  # Deterministic implementation.
    X_pool, y_codes, candidate_idx = _prepare_sdr_data(
        X,
        y,
        problem_type=problem_type,
        sdr_min_classes=sdr_min_classes,
        prefilter_fn=prefilter_fn,
        prefilter_max_features=sdr_prefilter_max_features,
    )
    if X_pool.size <= 0:
        return {}, {}

    n_samples, p = X_pool.shape
    classes = np.unique(y_codes)
    c = int(classes.size)
    if c < 2:
        return {}, {}

    X_center = np.asarray(X_pool - np.mean(X_pool, axis=0, keepdims=True), dtype=float)
    # Use centered one-hot response basis for the fitted inverse model.
    H = np.zeros((n_samples, c), dtype=float)
    H[np.arange(n_samples, dtype=int), y_codes.astype(int)] = 1.0
    H = H - np.mean(H, axis=0, keepdims=True)
    H_pinv = np.linalg.pinv(H)
    X_hat = np.asarray(H @ (H_pinv @ X_center), dtype=float)

    cov = _safe_covariance(X_center, ridge=float(sdr_covariance_ridge))
    pfc_kernel = _safe_covariance(X_hat, ridge=float(sdr_covariance_ridge))
    n_dirs = int(max(1, min(int(sdr_n_components), int(c - 1), p)))
    dirs, evals = _solve_generalized_sdr(pfc_kernel, cov, n_components=n_dirs)
    pooled_scores = _direction_scores(dirs, evals)
    return _finalize_result(
        pooled_scores,
        candidate_idx,
        n_target_features=n_target_features,
        n_features_total=int(np.asarray(X).shape[1]),
        normalize_fn=normalize_fn,
        method_name="pfc",
    )

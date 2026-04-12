"""HSIC Lasso feature selection."""
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

logger = logging.getLogger(__name__)


def hsic_lasso_selection(X, y, n_target_features, *, problem_type, random_state,
                          hsic_lasso_prefilter_max_features, hsic_lasso_target_sigma,
                          hsic_lasso_feature_sigma, hsic_lasso_alpha,
                          hsic_lasso_max_iter, hsic_lasso_relevance_blend,
                          hsic_lasso_binary_delta_enabled,
                          hsic_lasso_binary_delta_min_samples,
                          prefilter_fn, normalize_fn, rbf_kernel_1d_fn,
                          center_kernel_matrix_fn):
    """
    A25: HSIC Lasso pilot (kernelized relevance + sparse non-negative L1 fit).
    """
    if X.shape[1] == 0:
        return {}, {}

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    n_samples, n_features = X_arr.shape
    if n_samples < 6:
        return {}, {}

    n_target = int(min(max(1, n_target_features), n_features))
    pool_cap = int(
        min(
            n_features,
            max(n_target, int(hsic_lasso_prefilter_max_features)),
        )
    )
    pool_idx = prefilter_fn(X_arr, y_arr, max_features=pool_cap)
    if pool_idx.size == 0:
        return {}, {}

    X_pool = np.asarray(X_arr[:, pool_idx], dtype=float)
    X_pool = np.nan_to_num(X_pool, nan=0.0, posinf=0.0, neginf=0.0)
    # Column-standardize before kernel construction for scale robustness.
    col_mu = np.mean(X_pool, axis=0)
    col_sd = np.std(X_pool, axis=0, ddof=1)
    col_sd = np.where(np.isfinite(col_sd) & (col_sd > 1e-8), col_sd, 1.0)
    X_pool = (X_pool - col_mu[None, :]) / col_sd[None, :]

    target_kernel_kind = "rbf"
    binary_delta_applied = False
    if problem_type == "classification":
        y_codes = pd.Categorical(y_arr).codes.astype(int)
        n_classes = int(np.unique(y_codes).size)
        min_n = int(max(2, hsic_lasso_binary_delta_min_samples))
        use_binary_delta = (
            bool(hsic_lasso_binary_delta_enabled)
            and n_classes == 2
            and n_samples >= min_n
        )
        if use_binary_delta:
            L = (y_codes[:, None] == y_codes[None, :]).astype(float)
            target_sigma_eff = float("nan")
            target_kernel_kind = "delta_binary"
            binary_delta_applied = True
        else:
            L, target_sigma_eff = rbf_kernel_1d_fn(
                y_codes.astype(float),
                float(hsic_lasso_target_sigma),
            )
            target_kernel_kind = "rbf_label"
    else:
        L, target_sigma_eff = rbf_kernel_1d_fn(y_arr, float(hsic_lasso_target_sigma))
        target_kernel_kind = "rbf_target"
    Lc = center_kernel_matrix_fn(L)
    y_vec = np.asarray(Lc, dtype=float).ravel()
    y_vec = np.nan_to_num(y_vec, nan=0.0, posinf=0.0, neginf=0.0)
    y_vec = y_vec - float(np.mean(y_vec))
    y_norm = float(np.linalg.norm(y_vec))
    if not np.isfinite(y_norm) or y_norm <= 1e-12:
        return {}, {}
    y_vec = y_vec / y_norm

    hsic_scores = np.zeros(pool_idx.size, dtype=float)
    kernel_cols = []
    feature_sigmas = np.zeros(pool_idx.size, dtype=float)
    skipped = 0

    denom = float(max(1, n_samples - 1)) ** 2
    for j in range(pool_idx.size):
        K, sigma_eff = rbf_kernel_1d_fn(X_pool[:, j], float(hsic_lasso_feature_sigma))
        feature_sigmas[j] = float(sigma_eff)
        Kc = center_kernel_matrix_fn(K)
        hsic = float(np.sum(Kc * Lc) / max(1e-12, denom))
        if not np.isfinite(hsic):
            hsic = 0.0
        hsic_scores[j] = max(0.0, hsic)

        k_vec = np.asarray(Kc, dtype=float).ravel()
        k_vec = np.nan_to_num(k_vec, nan=0.0, posinf=0.0, neginf=0.0)
        k_vec = k_vec - float(np.mean(k_vec))
        k_norm = float(np.linalg.norm(k_vec))
        if not np.isfinite(k_norm) or k_norm <= 1e-12:
            skipped += 1
            kernel_cols.append(np.zeros_like(y_vec))
        else:
            kernel_cols.append(k_vec / k_norm)

    Z = np.column_stack(kernel_cols) if kernel_cols else np.zeros((y_vec.size, 0), dtype=float)
    coef = np.zeros(pool_idx.size, dtype=float)
    converged = True
    if Z.shape[1] > 0:
        try:
            model = Lasso(
                alpha=float(hsic_lasso_alpha),
                positive=True,
                fit_intercept=False,
                max_iter=int(hsic_lasso_max_iter),
                random_state=random_state,
            )
            model.fit(Z, y_vec)
            coef = np.asarray(model.coef_, dtype=float).ravel()
            coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
            coef = np.maximum(coef, 0.0)
            n_iter = int(getattr(model, "n_iter_", 0) or 0)
            converged = bool(n_iter < int(hsic_lasso_max_iter))
        except Exception as exc:
            coef = np.zeros(pool_idx.size, dtype=float)
            converged = False

    coef_norm = np.asarray(normalize_fn(coef), dtype=float).ravel()
    hsic_norm = np.asarray(normalize_fn(hsic_scores), dtype=float).ravel()
    blend = float(np.clip(hsic_lasso_relevance_blend, 0.0, 1.0))
    combined = (1.0 - blend) * coef_norm + blend * hsic_norm
    combined = np.asarray(normalize_fn(combined), dtype=float).ravel()
    if not np.any(combined > 0):
        combined = np.asarray(hsic_norm, dtype=float).ravel()

    selected_local = np.argsort(combined)[::-1][:n_target]
    selected_indices = np.asarray(pool_idx[selected_local], dtype=int)
    if selected_indices.size == 0:
        return {}, {}

    all_scores_vec = np.zeros(n_features, dtype=float)
    all_scores_vec[pool_idx] = combined
    all_scores_vec = np.asarray(normalize_fn(all_scores_vec), dtype=float).ravel()

    finite_sigmas = feature_sigmas[np.isfinite(feature_sigmas) & (feature_sigmas > 0.0)]
    sigma_median = float(np.median(finite_sigmas)) if finite_sigmas.size > 0 else float("nan")
    n_nonzero = int(np.sum(coef > 0))

    results = {
        "selected_indices": selected_indices,
        "scores": {
            int(idx): float(all_scores_vec[int(idx)])
            for idx in selected_indices.tolist()
        },
        "method": "hsic_lasso",
        "hsic_lasso_alpha": float(hsic_lasso_alpha),
        "hsic_lasso_prefilter_max_features": int(hsic_lasso_prefilter_max_features),
        "hsic_lasso_pool_size": int(pool_idx.size),
        "hsic_lasso_relevance_blend": float(hsic_lasso_relevance_blend),
        "hsic_lasso_feature_sigma": float(hsic_lasso_feature_sigma),
        "hsic_lasso_feature_sigma_median_effective": float(sigma_median) if np.isfinite(sigma_median) else float("nan"),
        "hsic_lasso_target_sigma": float(hsic_lasso_target_sigma),
        "hsic_lasso_target_sigma_effective": float(target_sigma_eff) if np.isfinite(target_sigma_eff) else float("nan"),
        "hsic_lasso_target_kernel": str(target_kernel_kind),
        "hsic_lasso_binary_delta_applied": bool(binary_delta_applied),
        "hsic_lasso_binary_delta_min_samples": int(max(2, hsic_lasso_binary_delta_min_samples)),
        "hsic_lasso_nonzero_coefficients": int(n_nonzero),
        "hsic_lasso_solver_converged": bool(converged),
        "hsic_lasso_kernel_column_skips": int(skipped),
        "hsic_lasso_max_iter": int(hsic_lasso_max_iter),
    }
    all_scores = {i: float(all_scores_vec[i]) for i in range(n_features)}
    return results, all_scores

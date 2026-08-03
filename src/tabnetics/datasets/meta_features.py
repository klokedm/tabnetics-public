"""Dataset meta-feature extraction helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import entropy as _shannon_entropy


def _sanitize_matrix(X: np.ndarray) -> np.ndarray:
    """Return a float64 matrix with non-finite values replaced column-wise."""

    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.size == 0:
        return X_arr
    finite = np.isfinite(X_arr)
    if finite.all():
        return X_arr
    cleaned = X_arr.copy()
    masked = np.where(finite, cleaned, np.nan)
    with np.errstate(all="ignore"):
        fill = np.nanmedian(masked, axis=0)
    fill = np.nan_to_num(fill, nan=0.0, posinf=0.0, neginf=0.0)
    rows, cols = np.where(~finite)
    cleaned[rows, cols] = fill[cols]
    return np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)


def extract_meta_features(
    X: np.ndarray,
    y: np.ndarray,
    *,
    expanded: bool = False,
    skip_distance_matrix: bool = False,
) -> Dict[str, float]:
    """Extract dataset meta-features useful for tier assignment and analysis.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Feature matrix.
    y : array-like, shape (n,)
        Target vector (class labels).
    expanded : bool, default False
        When True, additionally compute 9 dataset-complexity measures from the
        Ho & Basu / Lorena / pymfe taxonomy (T-R-398).  Existing callers that
        do not pass ``expanded`` are unaffected.

    Returns
    -------
    dict
        Always contains the 7 base features.  When *expanded* is True the dict
        also contains: ``fisher_f1``, ``f2_overlap``, ``n1_borderline``,
        ``n2_nn_ratio``, ``lsc``, ``t4_pca_ratio``, ``intrinsic_dim``,
        ``correlation_alpha``, ``signal_eigenvalue_fraction``.
    """
    X = _sanitize_matrix(X)
    y = np.asarray(y)

    n, p = X.shape if X.ndim == 2 else (X.shape[0], 1)

    classes, counts = np.unique(y, return_counts=True)
    class_count = float(len(classes))
    if class_count > 1:
        proportions = counts / counts.sum()
        raw_entropy = _shannon_entropy(proportions, base=np.e)
        class_balance_entropy = float(raw_entropy / np.log(class_count))
    else:
        class_balance_entropy = 0.0

    p_over_n = float(p) / float(n) if n > 0 else 0.0

    # --- correlation spectrum decay (+ keep sorted abs-corr for alpha) ---
    abs_corr_sorted: np.ndarray | None = None
    if p >= 3 and n >= 2:
        rng = np.random.RandomState(42)
        max_cols = 200
        if p > max_cols:
            col_idx = rng.choice(p, max_cols, replace=False)
            X_sub = X[:, col_idx]
        else:
            X_sub = X
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(X_sub, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        tri_idx = np.triu_indices_from(corr, k=1)
        abs_corr_sorted = np.sort(np.abs(corr[tri_idx]))[::-1]
        xs = np.arange(len(abs_corr_sorted), dtype=float)
        try:
            def _exp_decay(x, a, b):
                return a * np.exp(-b * x)

            popt, _ = curve_fit(
                _exp_decay,
                xs,
                abs_corr_sorted,
                p0=[abs_corr_sorted[0] if len(abs_corr_sorted) else 1.0, 0.01],
                maxfev=2000,
            )
            correlation_spectrum_decay = float(max(popt[1], 0.0))
        except Exception:
            correlation_spectrum_decay = 0.0
    else:
        correlation_spectrum_decay = 0.0

    if p > 0:
        heaping_count = 0
        for j in range(p):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0 and np.all(valid == np.round(valid)):
                heaping_count += 1
        heaping_fraction = float(heaping_count) / float(p)
    else:
        heaping_fraction = 0.0

    feats: Dict[str, float] = {
        "n": float(n),
        "p": float(p),
        "p_over_n": float(p_over_n),
        "class_count": float(class_count),
        "class_balance_entropy": float(class_balance_entropy),
        "correlation_spectrum_decay": float(correlation_spectrum_decay),
        "heaping_fraction": float(heaping_fraction),
    }

    if not expanded:
        return feats

    # =================================================================
    # Expanded complexity meta-features (T-R-398)
    # =================================================================

    # -- Feature-overlap measures (O(np × k²), cheap) ----------------
    feats["fisher_f1"] = _fisher_f1(X, y, classes)
    feats["f2_overlap"] = _f2_overlap(X, y, classes)

    # -- Eigenvalue-based measures (share eigenvalues) ----------------
    eigenvalues = _safe_eigenvalues(X, n, p)
    feats["t4_pca_ratio"] = _t4_pca_ratio(eigenvalues, p)
    feats["signal_eigenvalue_fraction"] = _signal_eigenvalue_fraction(
        eigenvalues, n, p,
    )

    # -- Correlation-based fractal measure ----------------------------
    feats["correlation_alpha"] = _correlation_alpha(abs_corr_sorted)

    # -- Distance-based measures (share distance matrix) --------------
    if skip_distance_matrix:
        feats["n1_borderline"] = 0.0
        feats["n2_nn_ratio"] = 0.0
        feats["lsc"] = 0.0
        feats["intrinsic_dim"] = _intrinsic_dim_knn(X, n)
    else:
        dist_matrix = _pairwise_distances(X, n)
        feats["n1_borderline"] = _n1_borderline(dist_matrix, y)
        feats["n2_nn_ratio"] = _n2_nn_ratio(dist_matrix, y)
        feats["lsc"] = _lsc(dist_matrix, y)
        feats["intrinsic_dim"] = _intrinsic_dim_mle(dist_matrix, n)

    return feats


# =====================================================================
# Private helpers — expanded meta-features
# =====================================================================


def _fisher_f1(X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Maximum Fisher's discriminant ratio (pymfe complexity.f1).

    For each feature, compute (mu_i - mu_j)^2 / (var_i + var_j) over
    all class pairs; return the global maximum.
    """
    if len(classes) < 2:
        return 0.0
    _n, p = X.shape
    max_f1 = 0.0
    for ci in range(len(classes)):
        for cj in range(ci + 1, len(classes)):
            mask_i = y == classes[ci]
            mask_j = y == classes[cj]
            mu_i = np.mean(X[mask_i], axis=0)
            mu_j = np.mean(X[mask_j], axis=0)
            var_i = (
                np.var(X[mask_i], axis=0, ddof=1)
                if mask_i.sum() > 1
                else np.zeros(p)
            )
            var_j = (
                np.var(X[mask_j], axis=0, ddof=1)
                if mask_j.sum() > 1
                else np.zeros(p)
            )
            denom = var_i + var_j
            with np.errstate(divide="ignore", invalid="ignore"):
                f1_vals = np.where(denom > 0, (mu_i - mu_j) ** 2 / denom, 0.0)
            pair_max = float(np.max(f1_vals))
            if pair_max > max_f1:
                max_f1 = pair_max
    return max_f1


def _f2_overlap(X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Mean per-feature class overlap fraction (adapted pymfe complexity.f2).

    Standard F2 uses a product which collapses to 0 for large p; we use the
    mean overlap fraction instead, which is more informative for HDLSS data.
    """
    if len(classes) < 2:
        return 0.0
    _n, p = X.shape
    overlap_sum = 0.0
    for j in range(p):
        col = X[:, j]
        max_overlap = 0.0
        for ci in range(len(classes)):
            for cj in range(ci + 1, len(classes)):
                vals_i = col[y == classes[ci]]
                vals_j = col[y == classes[cj]]
                minmax = min(float(np.max(vals_i)), float(np.max(vals_j)))
                maxmin = max(float(np.min(vals_i)), float(np.min(vals_j)))
                overlap = max(0.0, minmax - maxmin)
                full_range = max(float(np.max(vals_i)), float(np.max(vals_j))) - min(
                    float(np.min(vals_i)), float(np.min(vals_j))
                )
                if full_range > 0:
                    frac = overlap / full_range
                    if frac > max_overlap:
                        max_overlap = frac
        overlap_sum += max_overlap
    return overlap_sum / max(p, 1)


def _safe_eigenvalues(X: np.ndarray, n: int, p: int) -> np.ndarray:
    """Eigenvalues of the sample covariance, HDLSS-safe via Gram matrix."""
    if n < 2 or p <= 0:
        return np.zeros(0, dtype=np.float64)
    X_c = _sanitize_matrix(X) - np.mean(X, axis=0, keepdims=True)
    divisor = max(n - 1, 1)
    try:
        if n <= p:
            gram = X_c @ X_c.T / divisor
            eigs = np.linalg.eigvalsh(gram)
        else:
            cov = X_c.T @ X_c / divisor
            eigs = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        try:
            singular_values = np.linalg.svd(X_c, full_matrices=False, compute_uv=False)
        except np.linalg.LinAlgError:
            return np.zeros(min(n, p), dtype=np.float64)
        eigs = np.square(singular_values) / divisor
    eigs = np.asarray(eigs, dtype=np.float64)
    eigs = np.nan_to_num(eigs, nan=0.0, posinf=0.0, neginf=0.0)
    eigs = np.clip(eigs, 0.0, None)
    return np.sort(eigs)[::-1]


def _t4_pca_ratio(eigenvalues: np.ndarray, p: int) -> float:
    """Ratio of PCA dimensions (95% variance) to original dimension (pymfe t4)."""
    if len(eigenvalues) == 0 or p == 0:
        return 0.0
    pos = eigenvalues[eigenvalues > 0]
    if len(pos) == 0:
        return 0.0
    total = float(np.sum(pos))
    if total <= 0:
        return 0.0
    cumvar = np.cumsum(pos) / total
    n_components = int(np.searchsorted(cumvar, 0.95)) + 1
    return float(n_components) / float(p)


def _signal_eigenvalue_fraction(
    eigenvalues: np.ndarray, n: int, p: int,
) -> float:
    """Fraction of eigenvalues above the Marchenko-Pastur noise threshold."""
    pos = eigenvalues[eigenvalues > 0]
    if len(pos) < 2:
        return 0.0
    sigma2 = float(np.median(pos))
    gamma = float(p) / float(max(n, 1))
    lambda_plus = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
    n_signal = int(np.sum(pos > lambda_plus))
    return float(n_signal) / float(len(pos))


def _correlation_alpha(abs_corr_sorted: np.ndarray | None) -> float:
    """DFA scaling exponent on sorted abs-correlation spectrum (Kokol α-metric)."""
    if abs_corr_sorted is None or len(abs_corr_sorted) < 8:
        return 0.5  # default: random-walk exponent

    series = abs_corr_sorted
    profile = np.cumsum(series - np.mean(series))
    N = len(profile)

    min_s = 4
    max_s = N // 4
    if max_s < min_s:
        return 0.5

    scales = np.unique(
        np.logspace(np.log10(min_s), np.log10(max_s), 15).astype(int),
    )
    scales = scales[scales >= min_s]
    if len(scales) < 3:
        return 0.5

    log_s_list = []
    log_f_list = []
    for s in scales:
        n_win = N // s
        if n_win < 1:
            continue
        rms_sum = 0.0
        count = 0
        for w in range(n_win):
            seg = profile[w * s : (w + 1) * s]
            x = np.arange(s, dtype=float)
            coeffs = np.polyfit(x, seg, 1)
            trend = np.polyval(coeffs, x)
            rms_sum += float(np.sum((seg - trend) ** 2))
            count += s
        if count > 0 and rms_sum > 0:
            log_s_list.append(np.log(float(s)))
            log_f_list.append(np.log(np.sqrt(rms_sum / count)))

    if len(log_s_list) < 2:
        return 0.5

    log_s_arr = np.array(log_s_list)
    log_f_arr = np.array(log_f_list)
    valid = np.isfinite(log_s_arr) & np.isfinite(log_f_arr)
    if valid.sum() < 2:
        return 0.5

    try:
        coeffs = np.polyfit(log_s_arr[valid], log_f_arr[valid], 1)
    except Exception:
        return 0.5
    return float(coeffs[0])


def _pairwise_distances(X: np.ndarray, n: int) -> np.ndarray:
    """Euclidean distance matrix (n × n).  Shared by N1, N2, LSC, intrinsic_dim."""
    from scipy.spatial.distance import cdist

    return cdist(X, X, metric="euclidean")


def _n1_borderline(dist_matrix: np.ndarray, y: np.ndarray) -> float:
    """Fraction of MST edges connecting different classes (pymfe complexity.n1)."""
    from scipy.sparse.csgraph import minimum_spanning_tree

    n = len(y)
    if n < 2:
        return 0.0
    mst = minimum_spanning_tree(dist_matrix)
    coo = mst.tocoo()
    n_edges = len(coo.row)
    if n_edges == 0:
        return 0.0
    border = int(np.sum(y[coo.row] != y[coo.col]))
    return float(border) / float(n_edges)


def _n2_nn_ratio(dist_matrix: np.ndarray, y: np.ndarray) -> float:
    """Ratio of intra-class to inter-class NN distances (pymfe complexity.n2)."""
    n = len(y)
    if n < 3:
        return 1.0
    intra_total = 0.0
    inter_total = 0.0
    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf  # exclude self
        same = y == y[i]
        same[i] = False
        diff = ~same
        diff[i] = False
        if same.any():
            intra_total += float(np.min(dists[same]))
        if diff.any():
            inter_total += float(np.min(dists[diff]))
    if inter_total > 0:
        return float(intra_total / inter_total)
    return 1.0


def _lsc(dist_matrix: np.ndarray, y: np.ndarray) -> float:
    """Local set average cardinality (pymfe complexity.lsc)."""
    n = len(y)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf
        diff = y != y[i]
        if not diff.any():
            continue
        enemy_dist = float(np.min(dists[diff]))
        local_size = int(np.sum((dist_matrix[i] < enemy_dist) & (np.arange(n) != i)))
        total += local_size
    return float(total / n)


def _intrinsic_dim_mle(dist_matrix: np.ndarray, n: int) -> float:
    """MLE intrinsic dimensionality (Levina & Bickel 2005)."""
    if n < 5:
        return 0.0
    k = min(10, n - 2)
    if k < 2:
        return 0.0
    local_dims = []
    for i in range(n):
        dists = np.sort(dist_matrix[i])
        dists = dists[1:]  # skip self (distance 0)
        if len(dists) < k:
            continue
        T = dists[:k]
        T_k = T[-1]
        if T_k <= 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratios = np.log(T_k / T[: k - 1])
        valid = np.isfinite(log_ratios) & (log_ratios > 0)
        if valid.sum() == 0:
            continue
        m_k = float(np.mean(log_ratios[valid]))
        if m_k > 0:
            local_dims.append(1.0 / m_k)
    if len(local_dims) == 0:
        return 0.0
    return float(np.mean(local_dims))


def _intrinsic_dim_knn(X: np.ndarray, n: int, *, k: int = 10) -> float:
    """MLE intrinsic dimensionality via k-NN (Levina & Bickel 2005).

    Uses scipy BallTree for O(n*k*log(n)) instead of the O(n^2) full
    distance matrix.
    """
    if n < 5:
        return 0.0
    try:
        from sklearn.neighbors import BallTree  # type: ignore
    except Exception:
        return 0.0

    k_use = min(k, n - 2)
    if k_use < 2:
        return 0.0
    X_clean = _sanitize_matrix(X)
    try:
        tree = BallTree(X_clean)
        dists, _ = tree.query(X_clean, k=k_use + 1)  # +1 because first neighbor is self
    except Exception:
        return 0.0
    dists = dists[:, 1:]  # drop self-distance column
    T_k = dists[:, -1]  # k-th neighbor distance
    valid_mask = T_k > 0
    if not valid_mask.any():
        return 0.0
    local_dims = []
    for i in np.where(valid_mask)[0]:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratios = np.log(T_k[i] / dists[i, : k_use - 1])
        finite = np.isfinite(log_ratios) & (log_ratios > 0)
        if finite.sum() == 0:
            continue
        m_k = float(np.mean(log_ratios[finite]))
        if m_k > 0:
            local_dims.append(1.0 / m_k)
    return float(np.mean(local_dims)) if local_dims else 0.0


__all__ = ["extract_meta_features"]

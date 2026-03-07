"""Group-aware feature selection via sparse group lasso (VAL12_Suggestions §3.3).

Provides:
1. ``discover_feature_groups`` — automatic group discovery via hierarchical
   clustering on feature correlations.
2. ``group_sparse_lasso_selection`` — sparse group lasso (L1 + L2,1 penalty)
   feature selection that respects feature group structure.

The method uses a custom coordinate-descent / proximal-gradient solver so that
no external ``group_lasso`` package is required (zero new dependencies).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


# ---------------------------------------------------------------------------
# 1. Automatic group discovery
# ---------------------------------------------------------------------------

def discover_feature_groups(
    X: np.ndarray,
    *,
    method: str = "correlation_clustering",
    distance_threshold: float = 0.7,
    max_group_size: int = 50,
) -> np.ndarray:
    """Discover feature groups via hierarchical clustering on correlations.

    Parameters
    ----------
    X : (n_samples, n_features) array
    method : str
        ``"correlation_clustering"`` (default).
    distance_threshold : float
        Correlation-distance threshold for cutting the dendrogram. Features
        with ``1 - |corr| < distance_threshold`` end up in the same group.
    max_group_size : int
        Cap on group size; larger clusters are split arbitrarily.

    Returns
    -------
    groups : (n_features,) int array — group label for each feature.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.shape[1] <= 1:
        return np.zeros(max(1, arr.shape[1] if arr.ndim == 2 else 0), dtype=int)

    n_features = int(arr.shape[1])
    # Compute correlation matrix.
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    # Convert to distance: 1 - |corr|.
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)

    # Extract condensed distance matrix.
    condensed = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            condensed.append(float(dist[i, j]))
    condensed = np.asarray(condensed, dtype=float)
    if condensed.size == 0:
        return np.zeros(n_features, dtype=int)

    Z = linkage(condensed, method="average")
    thresh = float(np.clip(distance_threshold, 0.01, 1.99))
    labels = fcluster(Z, t=thresh, criterion="distance")
    groups = np.asarray(labels, dtype=int) - 1  # 0-indexed

    # Enforce max_group_size by splitting large groups.
    max_gs = int(max(2, max_group_size))
    next_group = int(groups.max()) + 1
    for g in range(int(groups.max()) + 1):
        members = np.where(groups == g)[0]
        if members.size > max_gs:
            n_splits = int(np.ceil(members.size / max_gs))
            for s in range(1, n_splits):
                chunk = members[s * max_gs : (s + 1) * max_gs]
                groups[chunk] = next_group
                next_group += 1

    return groups


# ---------------------------------------------------------------------------
# 2. Proximal operator for group lasso (L2,1 penalty)
# ---------------------------------------------------------------------------

def _prox_group_l21(
    coef: np.ndarray,
    groups: np.ndarray,
    lam_group: float,
) -> np.ndarray:
    """Proximal operator for L2,1 (group lasso) penalty."""
    out = coef.copy()
    for g in range(int(groups.max()) + 1):
        idx = np.where(groups == g)[0]
        if idx.size == 0:
            continue
        block = out[idx]
        norm = float(np.linalg.norm(block))
        scale = max(0.0, 1.0 - lam_group * np.sqrt(float(idx.size)) / max(norm, 1e-12))
        out[idx] = block * scale
    return out


def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Element-wise soft thresholding (proximal L1)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


# ---------------------------------------------------------------------------
# 3. Sparse group lasso solver (proximal gradient descent)
# ---------------------------------------------------------------------------

def _sparse_group_lasso_coef(
    X: np.ndarray,
    y_binary: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    max_iter: int = 500,
    tol: float = 1e-5,
) -> np.ndarray:
    """Fit sparse group lasso via proximal gradient descent.

    Minimises::

        (1/2n) ||y - X w||^2  +  alpha * l1_ratio * ||w||_1
                               +  alpha * (1 - l1_ratio) * sum_g sqrt(|g|) ||w_g||_2

    Parameters
    ----------
    X : (n_samples, n_features)
    y_binary : (n_samples,) — ±1 encoded binary label
    groups : (n_features,) int — group assignment
    alpha : float — overall regularisation strength
    l1_ratio : float — fraction of penalty devoted to L1 vs group L2,1
    max_iter : int
    tol : float — convergence tolerance on relative coefficient change

    Returns
    -------
    coef : (n_features,) array of fitted coefficients
    """
    n, p = X.shape
    coef = np.zeros(p, dtype=float)
    lam_l1 = float(alpha * l1_ratio)
    lam_group = float(alpha * (1.0 - l1_ratio))

    # Lipschitz constant estimate for step size.
    eigmax = float(np.linalg.norm(X, ord=2) ** 2) / float(max(1, n))
    step = 1.0 / max(eigmax, 1e-8)

    for iteration in range(int(max_iter)):
        residual = y_binary - X @ coef
        grad = -(X.T @ residual) / float(max(1, n))
        # Gradient step.
        coef_new = coef - step * grad
        # L1 proximal.
        coef_new = _soft_threshold(coef_new, step * lam_l1)
        # Group L2,1 proximal.
        coef_new = _prox_group_l21(coef_new, groups, step * lam_group)
        # Check convergence.
        diff = float(np.linalg.norm(coef_new - coef))
        norm_old = float(np.linalg.norm(coef))
        if diff < tol * max(1.0, norm_old):
            coef = coef_new
            break
        coef = coef_new

    return coef


# ---------------------------------------------------------------------------
# 4. Main selection function (follows FS method interface)
# ---------------------------------------------------------------------------

def group_sparse_lasso_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_target_features: int,
    *,
    groups: Optional[np.ndarray] = None,
    group_discovery_method: str = "correlation_clustering",
    group_distance_threshold: float = 0.7,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    random_state: int = 0,
) -> Tuple[Dict[str, Any], Dict[int, float]]:
    """Perform group-aware sparse feature selection.

    Parameters
    ----------
    X : (n_samples, n_features)
    y : (n_samples,) class labels (binary or multiclass)
    n_target_features : int
    groups : optional (n_features,) int array of group labels
        If None, groups are discovered automatically.
    group_discovery_method : str
    group_distance_threshold : float
    alpha : float
    l1_ratio : float
    random_state : int

    Returns
    -------
    results : dict with ``selected_indices``, ``scores``, etc.
    all_scores : {feature_idx: float}
    """
    arr = np.asarray(X, dtype=float)
    labels = np.asarray(y).ravel()
    n_samples, n_features = arr.shape
    n_target = int(min(max(1, n_target_features), n_features))

    rng = np.random.RandomState(int(random_state))

    # Standardise features.
    mu = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0)
    sd[sd < 1e-8] = 1.0
    X_std = (arr - mu) / sd

    # Discover groups if not provided.
    if groups is None:
        groups = discover_feature_groups(
            X_std,
            method=group_discovery_method,
            distance_threshold=group_distance_threshold,
        )
    groups = np.asarray(groups, dtype=int).ravel()
    if groups.size != n_features:
        groups = np.arange(n_features, dtype=int)

    # For multiclass: run one-vs-rest and aggregate importance.
    classes = np.unique(labels)
    importance = np.zeros(n_features, dtype=float)

    for cls in classes:
        y_binary = np.where(labels == cls, 1.0, -1.0)
        coef = _sparse_group_lasso_coef(
            X_std, y_binary, groups,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )
        importance += np.abs(coef)

    # Normalise.
    imp_max = float(np.max(importance))
    if imp_max > 1e-12:
        importance = importance / imp_max

    # Select top features by importance.
    order = np.argsort(-importance)
    selected = np.sort(order[:n_target])

    all_scores = {int(i): float(importance[i]) for i in range(n_features)}
    scores = {int(i): float(importance[i]) for i in selected}

    n_groups_total = int(groups.max() + 1)
    n_groups_selected = int(len(set(groups[selected].tolist())))

    results = {
        "selected_indices": selected,
        "scores": scores,
        "all_scores": all_scores,
        "n_groups_total": n_groups_total,
        "n_groups_selected": n_groups_selected,
        "group_labels": groups.tolist(),
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
    }
    return results, all_scores

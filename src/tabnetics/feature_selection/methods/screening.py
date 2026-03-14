"""Tier 2 interaction-aware screening methods (T-004).

This module implements screening methods that capture feature interactions,
complementing the Tier 1 univariate/bivariate prefilter in ``prefilter.py``.

Architecture (ArchitectureRefactor.md §14 RISK-2):
- Tier 1 (prefilter.py): cheap, always-on, univariate (MI, F-test)
- Tier 2 (this module): expensive, opt-in, interaction-aware (STIR/ReliefF)

Execution flow:
    Tier 1 prefilter → Tier 2 screening → method selection
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.feature_selection import f_classif

logger = logging.getLogger(__name__)

try:
    from ..prefilter import pvalues_to_evalues, ebh_support
except Exception as exc:
    try:
        from tabnetics.feature_selection.prefilter import pvalues_to_evalues, ebh_support
    except Exception as exc:
        def pvalues_to_evalues(p_values):  # type: ignore[misc]
            p = np.asarray(p_values, dtype=float).ravel()
            p = np.clip(np.nan_to_num(p, nan=1.0), 1e-12, 1.0)
            return np.asarray(1.0 / p, dtype=float)

        def ebh_support(e_values, alpha):  # type: ignore[misc]
            e_vals = np.asarray(e_values, dtype=float).ravel()
            if e_vals.size == 0:
                return np.array([], dtype=int)
            order = np.argsort(e_vals)[::-1]
            e_sorted = e_vals[order]
            p = int(e_vals.size)
            thresh = np.maximum((np.arange(1, p + 1, dtype=float) * float(alpha)) / float(p), 1e-12)
            valid = np.where(e_sorted >= (1.0 / thresh))[0]
            if valid.size == 0:
                return np.array([], dtype=int)
            return np.sort(order[: int(valid.max()) + 1])


# ---------------------------------------------------------------------------
# STIR / ReliefF scoring
# ---------------------------------------------------------------------------

def _nearest_neighbors(
    X: np.ndarray,
    idx: int,
    y: np.ndarray,
    same_class: bool,
    k: int,
) -> np.ndarray:
    """Find *k* nearest neighbours of ``X[idx]`` from same/different class.

    Parameters
    ----------
    X : (n, p) standardised feature matrix
    idx : row index of the query instance
    y : (n,) label array
    same_class : if True, search within the same class; else different class
    k : number of neighbours to return

    Returns
    -------
    indices : 1-d int array of neighbour row indices (may be shorter than *k*
        if fewer candidates are available)
    """
    label = y[idx]
    if same_class:
        mask = (y == label)
    else:
        mask = (y != label)
    mask[idx] = False  # exclude the query itself

    candidate_idx = np.flatnonzero(mask)
    if candidate_idx.size == 0:
        return np.array([], dtype=int)

    # Euclidean distances to candidates
    diffs = X[candidate_idx] - X[idx]
    dists = np.einsum("ij,ij->i", diffs, diffs)  # squared L2
    k_eff = min(k, candidate_idx.size)
    # argpartition is O(n) vs O(n log n) for argsort
    if k_eff < candidate_idx.size:
        part = np.argpartition(dists, k_eff - 1)[:k_eff]
    else:
        part = np.arange(candidate_idx.size)
    return candidate_idx[part]


def compute_stir_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 10,
    n_iter: int = 50,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Compute STIR/ReliefF-style feature importance scores.

    For each iteration:
    1. Sample a random instance
    2. Find *k* nearest neighbours of same class ("hits")
    3. Find *k* nearest neighbours of different class ("misses")
    4. Update feature weights: features that differ more from misses
       but agree with hits get higher scores

    For **multiclass** problems, miss contributions are weighted by
    the prior probability of each foreign class (standard ReliefF
    multi-class extension).

    Parameters
    ----------
    X : (n_samples, n_features)
    y : (n_samples,)
    n_neighbors : number of nearest neighbours per query
    n_iter : number of sampling iterations (hard cap, not wall-clock)
    random_state : seed for deterministic sampling

    Returns
    -------
    scores : (n_features,) array; higher = more relevant.
        All values are finite (NaN/Inf replaced by 0).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    # --- Edge cases --------------------------------------------------------
    if n_features == 0:
        return np.zeros(0, dtype=np.float64)

    classes, class_counts = np.unique(y, return_counts=True)
    n_classes = classes.size

    if n_classes < 2:
        # Single class → no discrimination possible
        logger.warning("compute_stir_scores: single class detected; returning zeros")
        return np.zeros(n_features, dtype=np.float64)

    if n_samples < 2:
        logger.warning("compute_stir_scores: n_samples < 2; returning zeros")
        return np.zeros(n_features, dtype=np.float64)

    # --- Standardise features (zero-mean, unit-var) ------------------------
    col_std = np.std(X, axis=0, ddof=0)
    constant_mask = col_std < 1e-12
    col_std[constant_mask] = 1.0  # avoid divide-by-zero; constant cols get 0 weight
    X_std = (X - np.mean(X, axis=0)) / col_std

    # --- Class priors for multiclass weighting of misses -------------------
    priors = class_counts.astype(np.float64) / n_samples
    class_to_prior = dict(zip(classes, priors))

    # --- Main loop ---------------------------------------------------------
    rng = np.random.RandomState(random_state)
    weights = np.zeros(n_features, dtype=np.float64)

    effective_iter = min(n_iter, n_samples)  # can't sample more than n

    for _ in range(effective_iter):
        idx = rng.randint(0, n_samples)
        label_i = y[idx]

        # --- Hits (same class) ---
        hit_idx = _nearest_neighbors(X_std, idx, y, same_class=True, k=n_neighbors)
        if hit_idx.size > 0:
            hit_diffs = np.abs(X_std[idx] - X_std[hit_idx])
            mean_hit_diff = np.mean(hit_diffs, axis=0)
            weights -= mean_hit_diff

        # --- Misses (different classes) ---
        for c in classes:
            if c == label_i:
                continue
            # Prior weight = P(class) / (1 - P(label_i))
            p_c = class_to_prior[c]
            p_label = class_to_prior[label_i]
            denom = 1.0 - p_label
            if denom < 1e-12:
                continue
            miss_weight = p_c / denom

            # Neighbours of class c
            class_mask = (y == c)
            class_candidates = np.flatnonzero(class_mask)
            if class_candidates.size == 0:
                continue

            k_eff = min(n_neighbors, class_candidates.size)
            diffs_c = X_std[class_candidates] - X_std[idx]
            dists_c = np.einsum("ij,ij->i", diffs_c, diffs_c)
            if k_eff < class_candidates.size:
                part = np.argpartition(dists_c, k_eff - 1)[:k_eff]
            else:
                part = np.arange(class_candidates.size)
            miss_nn = class_candidates[part]

            miss_diffs = np.abs(X_std[idx] - X_std[miss_nn])
            mean_miss_diff = np.mean(miss_diffs, axis=0)
            weights += miss_weight * mean_miss_diff

    # Normalise by number of iterations
    if effective_iter > 0:
        weights /= effective_iter

    # Zero-out constant features
    weights[constant_mask] = 0.0

    # Sanitise
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return weights


# ---------------------------------------------------------------------------
# Screening wrapper
# ---------------------------------------------------------------------------
def _screen_features_evalue(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    min_features: int,
) -> Optional[np.ndarray]:
    """Tier 2 e-value screening (p→e conversion + e-BH support)."""
    n_samples, n_features = X.shape
    if n_features == 0 or n_samples < 2:
        return np.array([], dtype=int)
    y_arr = np.asarray(y).ravel()
    if np.unique(y_arr).size < 2:
        return None

    try:
        _, pvals = f_classif(X, y_arr)
        pvals = np.asarray(pvals, dtype=float).ravel()
    except Exception as exc:
        logger.exception("E-value screening: failed to compute p-values")
        return None
    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
    e_vals = pvalues_to_evalues(pvals)
    support = ebh_support(e_vals, alpha=float(alpha))
    support = np.asarray(
        sorted(set(int(i) for i in np.asarray(support, dtype=int).tolist() if 0 <= int(i) < n_features)),
        dtype=int,
    )
    if support.size < int(max(1, min_features)):
        fallback_k = int(min(n_features, max(1, min_features)))
        fallback = np.argsort(e_vals)[::-1][:fallback_k]
        support = np.asarray(sorted(set(int(i) for i in fallback.tolist())), dtype=int)
    if support.size >= n_features:
        return None
    return support


def screen_features(
    X: np.ndarray,
    y: np.ndarray,
    *,
    enabled: bool = False,
    method: str = "none",
    pool_cap: int = 2000,
    stir_n_neighbors: int = 10,
    stir_n_iter: int = 50,
    stir_keep_fraction: float = 0.5,
    stir_min_features: int = 20,
    evalue_alpha: float = 0.2,
    evalue_min_features: int = 20,
    random_state: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Apply Tier 2 STIR screening to reduce the feature pool.

    Parameters
    ----------
    X : (n_samples, n_features) — already prefiltered by Tier 1
    y : (n_samples,) — labels
    enabled : screening toggle
    method : screening method ("stir" | "none")
    pool_cap : safety cap on features to screen
    stir_n_neighbors, stir_n_iter : STIR hyper-parameters
    stir_keep_fraction : fraction of features to retain (0, 1]
    stir_min_features : never drop below this many features
    random_state : seed

    Returns
    -------
    selected_indices : 1-d int array of column indices to keep
        (relative to *X*), or ``None`` if screening is disabled /
        not applicable.
    """
    if not enabled or method == "none":
        return None

    n_samples, n_features = X.shape

    if n_features == 0:
        return np.array([], dtype=int)

    # --- Pool cap: subsample columns if too wide -------------------------
    if n_features > pool_cap:
        logger.info(
            "Tier 2 screening: pool_cap=%d reached (p=%d); subsampling columns",
            pool_cap, n_features,
        )
        rng = np.random.RandomState(random_state)
        keep = rng.choice(n_features, size=pool_cap, replace=False)
        keep.sort()
        X_screen = X[:, keep]
        index_map = keep
    else:
        X_screen = X
        index_map = np.arange(n_features)

    method_norm = str(method or "none").strip().lower()
    if method_norm not in {"stir", "evalue"}:
        logger.warning("Unknown screening method %r; skipping", method)
        return None

    if method_norm == "evalue":
        selected_screen = _screen_features_evalue(
            X_screen,
            y,
            alpha=float(evalue_alpha),
            min_features=int(evalue_min_features),
        )
        if selected_screen is None:
            return None
        selected_screen = np.asarray(selected_screen, dtype=int)
        selected = index_map[selected_screen]
        logger.info(
            "Tier 2 e-value screening: %d → %d features retained (alpha=%.3f, min=%d)",
            n_features,
            selected.size,
            float(evalue_alpha),
            int(evalue_min_features),
        )
        return np.asarray(selected, dtype=int)

    # method_norm == "stir"
    try:
        scores = compute_stir_scores(
            X_screen, y,
            n_neighbors=stir_n_neighbors,
            n_iter=stir_n_iter,
            random_state=random_state,
        )
    except Exception as exc:
        logger.exception("STIR screening failed; falling back to no screening")
        return None

    n_keep = max(
        int(np.ceil(stir_keep_fraction * len(scores))),
        min(stir_min_features, len(scores)),
    )
    n_keep = min(n_keep, len(scores))
    if n_keep >= len(scores):
        return None

    top_idx = np.argsort(scores)[::-1][:n_keep]
    top_idx.sort()
    selected = index_map[top_idx]
    logger.info(
        "Tier 2 STIR screening: %d → %d features retained "
        "(keep_fraction=%.2f, min_features=%d)",
        n_features, selected.size, stir_keep_fraction, stir_min_features,
    )
    return np.asarray(selected, dtype=int)


def screen_features_stir(
    X: np.ndarray,
    y: np.ndarray,
    *,
    enabled: bool = False,
    method: str = "none",
    pool_cap: int = 2000,
    stir_n_neighbors: int = 10,
    stir_n_iter: int = 50,
    stir_keep_fraction: float = 0.5,
    stir_min_features: int = 20,
    random_state: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Backward-compatible wrapper for prior callers/tests."""
    return screen_features(
        X,
        y,
        enabled=enabled,
        method=method,
        pool_cap=pool_cap,
        stir_n_neighbors=stir_n_neighbors,
        stir_n_iter=stir_n_iter,
        stir_keep_fraction=stir_keep_fraction,
        stir_min_features=stir_min_features,
        random_state=random_state,
    )

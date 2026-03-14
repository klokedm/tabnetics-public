"""Filter-based feature selection methods."""
import logging

import numpy as np
import scipy.stats as sps

logger = logging.getLogger(__name__)


def _invalid_xy_shape(X) -> bool:
    """Return True when selector entry points should short-circuit."""
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        return True
    n_samples, n_features = X_arr.shape
    return int(n_samples) < 2 or int(n_features) <= 0


def _safe_rank_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Rank-based discretization for MI/CMI estimators."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return np.zeros(0, dtype=int)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Constant vectors carry no information; assign a single bin.
    if float(np.std(arr)) < 1e-12:
        return np.zeros(arr.size, dtype=int)
    # Preserve low-cardinality discrete structure (e.g., binary labels) rather
    # than collapsing to one bin via quantile edge de-duplication.
    unique_vals = np.unique(arr)
    if 2 <= unique_vals.size <= int(max(2, min(32, n_bins))):
        _, inv = np.unique(arr, return_inverse=True)
        return np.asarray(inv, dtype=int)
    n_bins_eff = int(max(2, min(int(n_bins), arr.size)))
    q = np.linspace(0.0, 1.0, n_bins_eff + 1)
    edges = np.quantile(arr, q)
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        return np.zeros(arr.size, dtype=int)
    # Drop duplicate edges to avoid empty bins.
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(arr.size, dtype=int)
    # np.digitize expects interior edges for fixed bins.
    binned = np.digitize(arr, edges[1:-1], right=False)
    return np.asarray(np.clip(binned, 0, max(0, edges.size - 2)), dtype=int)


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float).ravel()
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _mutual_information_discrete(x_disc: np.ndarray, y_disc: np.ndarray) -> float:
    x = np.asarray(x_disc, dtype=int).ravel()
    y = np.asarray(y_disc, dtype=int).ravel()
    n = int(min(x.size, y.size))
    if n <= 1:
        return 0.0
    x = x[:n]
    y = y[:n]
    x_u, x_inv = np.unique(x, return_inverse=True)
    y_u, y_inv = np.unique(y, return_inverse=True)
    joint = np.zeros((x_u.size, y_u.size), dtype=float)
    np.add.at(joint, (x_inv, y_inv), 1.0)
    px = np.sum(joint, axis=1)
    py = np.sum(joint, axis=0)
    total = float(np.sum(joint))
    if total <= 0.0:
        return 0.0
    px = px / total
    py = py / total
    pxy = joint / total
    nz = pxy > 0.0
    if not np.any(nz):
        return 0.0
    ii, jj = np.where(nz)
    vals = pxy[ii, jj] * np.log(pxy[ii, jj] / (px[ii] * py[jj]))
    return float(np.sum(vals))


def _conditional_mutual_information_discrete(
    x_disc: np.ndarray,
    y_disc: np.ndarray,
    z_disc: np.ndarray,
) -> float:
    """Estimate I(X;Y|Z) using plug-in discrete entropies."""
    x = np.asarray(x_disc, dtype=int).ravel()
    y = np.asarray(y_disc, dtype=int).ravel()
    z = np.asarray(z_disc, dtype=int).ravel()
    n = int(min(x.size, y.size, z.size))
    if n <= 1:
        return 0.0
    x = x[:n]
    y = y[:n]
    z = z[:n]

    z_unique = np.unique(z)
    total = float(n)
    cmi = 0.0
    for z_val in z_unique:
        mask = z == z_val
        nz = int(np.sum(mask))
        if nz <= 1:
            continue
        weight = float(nz) / total
        cmi += weight * _mutual_information_discrete(x[mask], y[mask])
    return float(max(0.0, cmi))


def _compute_symmetric_uncertainty(x_disc: np.ndarray, y_disc: np.ndarray) -> float:
    mi = _mutual_information_discrete(x_disc, y_disc)
    hx = _entropy_from_counts(np.bincount(np.asarray(x_disc, dtype=int)))
    hy = _entropy_from_counts(np.bincount(np.asarray(y_disc, dtype=int)))
    denom = hx + hy
    if denom <= 1e-12:
        return 0.0
    return float(max(0.0, 2.0 * mi / denom))


def _mi_redundancy_score(
    X_pool: np.ndarray,
    X_disc: np.ndarray,
    idx: int,
    selected_local: list[int],
    *,
    mode: str,
    random_state: int,
) -> float:
    if not selected_local:
        return 0.0
    if mode == "pearson":
        vals = [
            float(abs(np.corrcoef(X_pool[:, idx], X_pool[:, j])[0, 1]))
            for j in selected_local
        ]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else 0.0

    if mode == "binned_mi":
        vals = [
            _mutual_information_discrete(X_disc[:, idx], X_disc[:, j])
            for j in selected_local
        ]
        vals = [float(v) for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else 0.0

    # mode == "knn_mi"
    try:
        from sklearn.feature_selection import mutual_info_regression
    except Exception as exc:
        return 0.0
    vals = []
    xi = np.asarray(X_pool[:, idx], dtype=float).reshape(-1, 1)
    for j in selected_local:
        yj = np.asarray(X_pool[:, j], dtype=float).ravel()
        try:
            score = mutual_info_regression(
                xi,
                yj,
                random_state=int(random_state),
                n_neighbors=3,
            )
            vals.append(float(np.asarray(score, dtype=float).ravel()[0]))
        except Exception as exc:
            continue
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def mutual_information_selection(X, y, n_target_features, mi_scorer, random_state):
    """Perform Mutual Information based selection."""
    if _invalid_xy_shape(X):
        return {}, {}

    mi_scores = mi_scorer(X, y, random_state=random_state)
    selected_indices = np.argsort(mi_scores)[::-1][:n_target_features]

    results = {
        'selected_indices': selected_indices,
        'scores': {idx: mi_scores[idx] for idx in selected_indices},
        'all_scores': mi_scores
    }

    return results, {i: mi_scores[i] for i in range(len(mi_scores))}


def random_selection(X, y, n_target_features, random_state):
    """Select features uniformly at random (baseline).

    This is a *null-hypothesis baseline*: features are chosen without
    examining ``X`` or ``y``.  Comparing other methods against this
    baseline reveals whether they extract genuine signal or merely
    benefit from dimensionality reduction.

    The selection is deterministic for a given ``random_state``.
    """
    if _invalid_xy_shape(X):
        return {}, {}

    n_features = X.shape[1]
    k = min(int(n_target_features), n_features)
    rng = np.random.RandomState(int(random_state))
    selected_indices = rng.choice(n_features, size=k, replace=False)
    scores = np.ones(n_features, dtype=float)

    results = {
        'selected_indices': selected_indices,
        'scores': {idx: 1.0 for idx in selected_indices},
        'all_scores': scores,
    }
    return results, {i: 1.0 for i in range(n_features)}


def anova_f_selection(X, y, n_target_features, f_scorer):
    """Perform ANOVA F-test based selection."""
    if _invalid_xy_shape(X):
        return {}, {}

    f_scores, _ = f_scorer(X, y)
    selected_indices = np.argsort(f_scores)[::-1][:n_target_features]

    results = {
        'selected_indices': selected_indices,
        'scores': {idx: f_scores[idx] for idx in selected_indices},
        'all_scores': f_scores
    }

    return results, {i: f_scores[i] for i in range(len(f_scores))}


def _wmw_auc_multiclass(X, y_arr, classes, n_target_features):
    """Compute per-feature M metric (Hand & Till 2001) for multi-class settings.

    For each feature j and each unordered class pair (i, k), the WMW AUC is
    computed via rank-sum statistics.  The direction-invariant pairwise
    separation ``max(AUC_{ik}, 1 - AUC_{ik})`` is averaged across all
    C*(C-1)/2 pairs, yielding the M metric per feature.  The final score is
    mapped to [0, 1] as ``2 * (mean_sep - 0.5)``.

    References
    ----------
    Hand, D.J. & Till, R.J. (2001). A Simple Generalisation of the Area Under
        the ROC Curve for Multiple Class Classification Problems. Machine
        Learning 45(2), 171-186. DOI: 10.1023/A:1010920819831
    Yang, Z. et al. (2022). Learning with Multiclass AUC: Theory and
        Algorithms. IEEE TPAMI. arXiv: 2107.13171
    """
    n_features = int(X.shape[1])
    scores = np.zeros(n_features, dtype=float)
    m_scores = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        col = np.asarray(X[:, j], dtype=float).ravel()
        if not np.any(np.isfinite(col)) or float(np.std(col)) < 1e-12:
            continue
        if np.any(~np.isfinite(col)):
            med = float(np.nanmedian(col))
            col = np.nan_to_num(col, nan=med, posinf=med, neginf=med)

        pair_seps = []
        for ci_idx, ci in enumerate(classes):
            mask_i = y_arr == ci
            n_i = int(np.sum(mask_i))
            if n_i < 1:
                continue
            for cj_idx, cj in enumerate(classes):
                if cj_idx <= ci_idx:          # upper triangle only
                    continue
                mask_j = y_arr == cj
                n_j = int(np.sum(mask_j))
                if n_j < 1:
                    continue
                pair_mask = np.logical_or(mask_i, mask_j)
                pair_vals = np.asarray(col[pair_mask], dtype=float)
                if pair_vals.size < 2:
                    continue
                pair_labels = np.asarray(y_arr[pair_mask])
                pair_ranks = sps.rankdata(pair_vals, method="average")
                mask_i_pair = pair_labels == ci
                u_ij = float(np.sum(pair_ranks[mask_i_pair])) - float(n_i * (n_i + 1) / 2.0)
                denom = float(n_i * n_j)
                if denom <= 0:
                    continue
                auc = float(np.clip(u_ij / denom, 0.0, 1.0))
                pair_seps.append(max(auc, 1.0 - auc))

        if pair_seps:
            mean_sep = float(np.mean(pair_seps))
            m_scores[j] = mean_sep
            scores[j] = float(max(0.0, (mean_sep - 0.5) * 2.0))

    sel = np.argsort(scores)[::-1][: int(min(n_target_features, n_features))]
    return (
        {
            "selected_indices": sel,
            "scores": {int(i): float(scores[i]) for i in sel},
            "all_scores": scores,
            "m_metric_scores": m_scores,
            "n_classes": len(classes),
        },
        {i: float(scores[i]) for i in range(n_features)},
    )


def wmw_auc_selection(X, y, n_target_features, problem_type):
    """
    Univariate AUC ranking via the Wilcoxon-Mann-Whitney statistic.

    For binary problems the standard WMW AUC is used.  For multi-class
    problems the M metric (Hand & Till 2001) — the average of pairwise
    direction-invariant AUCs across all C*(C-1)/2 class pairs — is computed
    via ``_wmw_auc_multiclass``.

    Scoring is direction-invariant: ``score = 2 * |AUC - 0.5|`` in [0,1].
    This makes the method stable under label flipping and treats inverse
    separation as equally informative.
    """
    if problem_type != "classification" or _invalid_xy_shape(X):
        return {}, {}

    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    if classes.size < 2:
        return {}, {}
    if classes.size > 2:
        return _wmw_auc_multiclass(X, y_arr, classes, n_target_features)

    # --- Binary path ---
    # Use the larger label as "positive" only to define the AUC direction.
    # The final score is symmetric in direction.
    pos_label = classes.max()
    pos_mask = y_arr == pos_label
    n_pos = int(np.sum(pos_mask))
    n_neg = int(y_arr.size - n_pos)
    if n_pos < 1 or n_neg < 1:
        return {}, {}

    n_features = int(X.shape[1])
    scores = np.zeros(n_features, dtype=float)
    aucs = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        col = np.asarray(X[:, j], dtype=float).ravel()
        if col.size == 0:
            continue

        if not np.any(np.isfinite(col)):
            continue

        # Impute within-column NaNs to avoid rankdata failures.
        if np.any(~np.isfinite(col)):
            med = np.nanmedian(col)
            if not np.isfinite(med):
                med = 0.0
            col = np.nan_to_num(col, nan=float(med), posinf=float(med), neginf=float(med))

        if float(np.std(col)) < 1e-12:
            continue

        ranks = sps.rankdata(col, method="average")
        rank_sum_pos = float(np.sum(ranks[pos_mask]))
        u_stat = rank_sum_pos - float(n_pos * (n_pos + 1) / 2.0)
        denom = float(n_pos * n_neg)
        if denom <= 0:
            continue

        auc = float(u_stat / denom)
        if not np.isfinite(auc):
            continue
        auc = float(np.clip(auc, 0.0, 1.0))
        aucs[j] = auc

        sep = max(auc, 1.0 - auc)
        scores[j] = float(max(0.0, (sep - 0.5) * 2.0))

    selected_indices = np.argsort(scores)[::-1][: int(min(n_target_features, n_features))]
    results = {
        "selected_indices": selected_indices,
        "scores": {int(idx): float(scores[int(idx)]) for idx in selected_indices},
        "all_scores": scores,
        "auc_scores": aucs,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "positive_label": int(pos_label) if np.issubdtype(classes.dtype, np.integer) else str(pos_label),
    }
    return results, {i: float(scores[i]) for i in range(n_features)}


def mrmr_jmi_selection(
    X, y, n_target_features,
    random_state, mi_scorer, prefilter_fn, normalize_fn,
    mrmr_max_features, mrmr_redundancy_weight,
    mrmr_mi_redundancy_enabled=False,
    mrmr_mi_n_bins=8,
):
    """
    Redundancy-aware forward selection.
    Uses MI relevance with a configurable redundancy penalty:
      - legacy (default): absolute Pearson correlation
      - MI tiered mode (opt-in): Pearson for n<80, binned MI for 80-149,
        k-NN MI for n>=150.
    """
    n_samples, n_features = np.asarray(X).shape
    if int(n_samples) < 2 or int(n_features) == 0:
        return {}, {}

    pool_idx = prefilter_fn(X, y, max_features=min(mrmr_max_features, n_features))
    if pool_idx.size == 0:
        return {}, {}
    X_pool = X[:, pool_idx]

    try:
        relevance = np.asarray(mi_scorer(X_pool, y, random_state=random_state), dtype=float).ravel()
    except Exception as exc:
        relevance = np.zeros(X_pool.shape[1], dtype=float)
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)
    relevance_norm = normalize_fn(relevance)

    k_target = int(min(max(1, n_target_features), X_pool.shape[1]))
    selected_local = []
    selected_set = set()
    criterion_scores = np.zeros(X_pool.shape[1], dtype=float)

    redundancy_mode = "pearson"
    X_disc = None
    if bool(mrmr_mi_redundancy_enabled):
        if int(n_samples) < 80:
            redundancy_mode = "pearson"
        elif int(n_samples) < 150:
            redundancy_mode = "binned_mi"
        else:
            redundancy_mode = "knn_mi"
        if redundancy_mode == "binned_mi":
            X_disc = np.column_stack(
                [_safe_rank_bin(X_pool[:, j], n_bins=int(mrmr_mi_n_bins)) for j in range(X_pool.shape[1])]
            )

    if X_pool.shape[1] <= 1:
        selected_local = [0]
        criterion_scores[0] = float(relevance_norm[0]) if relevance_norm.size else 0.0
    else:
        first = int(np.argmax(relevance_norm))
        selected_local.append(first)
        selected_set.add(first)
        criterion_scores[first] = relevance_norm[first]

        while len(selected_local) < k_target:
            best_idx = None
            best_score = -np.inf
            for idx in range(X_pool.shape[1]):
                if idx in selected_set:
                    continue
                redundancy = _mi_redundancy_score(
                    X_pool,
                    np.asarray(X_disc) if X_disc is not None else np.zeros((0, 0), dtype=int),
                    idx,
                    selected_local,
                    mode=str(redundancy_mode),
                    random_state=int(random_state),
                )
                score = float(relevance_norm[idx] - mrmr_redundancy_weight * redundancy)
                criterion_scores[idx] = score
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected_local.append(int(best_idx))
            selected_set.add(int(best_idx))

    selected_indices = pool_idx[np.array(selected_local, dtype=int)]
    selected_indices = selected_indices[:n_target_features]

    all_scores = np.zeros(n_features, dtype=float)
    all_scores[pool_idx] = criterion_scores
    all_scores = np.nan_to_num(all_scores, nan=0.0, posinf=0.0, neginf=0.0)
    min_score = float(np.min(all_scores)) if all_scores.size else 0.0
    if min_score < 0:
        all_scores = all_scores - min_score

    results = {
        'selected_indices': selected_indices,
        'scores': {int(idx): float(all_scores[idx]) for idx in selected_indices},
        'criterion': 'mrmr_jmi_mi' if bool(mrmr_mi_redundancy_enabled) else 'mrmr_jmi',
        'redundancy_mode': str(redundancy_mode),
        'pool_size': int(pool_idx.size),
    }
    return results, {i: float(all_scores[i]) for i in range(n_features)}


def fcbf_selection(
    X,
    y,
    n_target_features,
    *,
    prefilter_fn,
    normalize_fn,
    max_features=320,
    n_bins=8,
):
    """Fast Correlation-Based Filter (FCBF) using symmetric uncertainty."""
    if _invalid_xy_shape(X):
        return {}, {}
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()
    n_samples, n_features = X_arr.shape
    if int(n_samples) < 2 or int(n_features) <= 0:
        return {}, {}

    pool_idx = prefilter_fn(X_arr, y_arr, max_features=min(int(max_features), n_features))
    if pool_idx.size == 0:
        return {}, {}

    X_pool = np.asarray(X_arr[:, pool_idx], dtype=float)
    X_disc = np.column_stack([_safe_rank_bin(X_pool[:, j], n_bins=int(n_bins)) for j in range(X_pool.shape[1])])
    y_disc = _safe_rank_bin(y_arr, n_bins=max(2, int(min(16, np.unique(y_arr).size + 1))))

    su_with_target = np.zeros(X_pool.shape[1], dtype=float)
    for j in range(X_pool.shape[1]):
        su_with_target[j] = _compute_symmetric_uncertainty(X_disc[:, j], y_disc)
    su_with_target = np.nan_to_num(su_with_target, nan=0.0, posinf=0.0, neginf=0.0)
    ranked = list(np.argsort(su_with_target)[::-1].tolist())

    target_k = int(min(max(1, n_target_features), X_pool.shape[1]))
    selected_local: list[int] = []
    selected_set = set()
    removed = set()
    for i in ranked:
        if i in removed:
            continue
        selected_local.append(int(i))
        selected_set.add(int(i))
        for j in ranked:
            if j == i or j in removed:
                continue
            su_ij = _compute_symmetric_uncertainty(X_disc[:, i], X_disc[:, j])
            # Classic FCBF pruning criterion.
            if su_ij >= su_with_target[j]:
                removed.add(int(j))
        if len(selected_local) >= target_k:
            break

    # FCBF pruning can be aggressive on HDLSS; backfill from ranked SU to honor
    # top-k contract without changing the pruning pass itself.
    if len(selected_local) < target_k:
        for idx in ranked:
            idx = int(idx)
            if idx in selected_set:
                continue
            selected_local.append(idx)
            selected_set.add(idx)
            if len(selected_local) >= target_k:
                break

    selected_local = selected_local[:target_k]
    selected_indices = pool_idx[np.asarray(selected_local, dtype=int)]
    all_scores = np.zeros(n_features, dtype=float)
    all_scores[pool_idx] = normalize_fn(su_with_target)
    all_scores = np.nan_to_num(all_scores, nan=0.0, posinf=0.0, neginf=0.0)

    results = {
        "selected_indices": np.asarray(selected_indices, dtype=int),
        "scores": {int(idx): float(all_scores[int(idx)]) for idx in selected_indices},
        "criterion": "fcbf_su",
        "pool_size": int(pool_idx.size),
    }
    return results, {i: float(all_scores[i]) for i in range(n_features)}


def cmim_selection(
    X,
    y,
    n_target_features,
    *,
    random_state,
    mi_scorer,
    prefilter_fn,
    normalize_fn,
    max_features=320,
    min_samples=60,
    n_bins=8,
):
    """CMIM greedy forward selection with binned conditional MI."""
    if _invalid_xy_shape(X):
        return {}, {}

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()
    n_samples, n_features = X_arr.shape
    if int(n_samples) < int(max(2, min_samples)) or int(n_features) <= 0:
        return {}, {}

    pool_idx = prefilter_fn(X_arr, y_arr, max_features=min(int(max_features), n_features))
    if pool_idx.size == 0:
        return {}, {}
    X_pool = np.asarray(X_arr[:, pool_idx], dtype=float)

    try:
        relevance = np.asarray(mi_scorer(X_pool, y_arr, random_state=int(random_state)), dtype=float).ravel()
    except Exception as exc:
        relevance = np.zeros(X_pool.shape[1], dtype=float)
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)

    X_disc = np.column_stack([_safe_rank_bin(X_pool[:, j], n_bins=int(n_bins)) for j in range(X_pool.shape[1])])
    y_disc = _safe_rank_bin(y_arr, n_bins=max(2, int(min(16, np.unique(y_arr).size + 1))))
    relevance_norm = normalize_fn(relevance)

    k_target = int(min(max(1, n_target_features), X_pool.shape[1]))
    selected_local: list[int] = []
    selected_set = set()
    cmim_scores = np.zeros(X_pool.shape[1], dtype=float)

    first = int(np.argmax(relevance_norm))
    selected_local.append(first)
    selected_set.add(first)
    cmim_scores[first] = float(relevance_norm[first])

    while len(selected_local) < k_target:
        best_idx = None
        best_score = -np.inf
        for idx in range(X_pool.shape[1]):
            if idx in selected_set:
                continue
            if not selected_local:
                score = float(relevance_norm[idx])
            else:
                cmi_vals = [
                    _conditional_mutual_information_discrete(X_disc[:, idx], y_disc, X_disc[:, sel])
                    for sel in selected_local
                ]
                # CMIM objective (min conditional MI) with a small relevance tie-break.
                # This preserves linear-data agreement with mRMR while keeping
                # interaction sensitivity on non-linear datasets.
                cmi_min = float(np.min(np.asarray(cmi_vals, dtype=float))) if cmi_vals else 0.0
                score = float(0.85 * cmi_min + 0.15 * relevance_norm[idx])
            cmim_scores[idx] = float(score)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected_local.append(int(best_idx))
        selected_set.add(int(best_idx))

    selected_indices = pool_idx[np.asarray(selected_local, dtype=int)]
    selected_indices = selected_indices[: int(max(1, n_target_features))]

    all_scores = np.zeros(n_features, dtype=float)
    all_scores[pool_idx] = normalize_fn(cmim_scores)
    all_scores = np.nan_to_num(all_scores, nan=0.0, posinf=0.0, neginf=0.0)
    results = {
        "selected_indices": np.asarray(selected_indices, dtype=int),
        "scores": {int(idx): float(all_scores[int(idx)]) for idx in selected_indices},
        "criterion": "cmim",
        "pool_size": int(pool_idx.size),
        "min_samples_gate": int(min_samples),
    }
    return results, {i: float(all_scores[i]) for i in range(n_features)}


def chi_square_selection(X, y, n_target_features):
    """Perform Chi-Square univariate filter selection.

    Requires non-negative features, so MinMaxScaler is applied internally.
    Uses sklearn.feature_selection.chi2 for scoring.
    """
    if _invalid_xy_shape(X):
        return {}, {}

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import chi2

    X_arr = np.asarray(X, dtype=float)

    # NaN guard: chi2 requires finite non-negative inputs. Impute NaN with
    # column median (preserves feature ranking better than zero-fill).
    nan_mask = np.isnan(X_arr)
    if nan_mask.any():
        col_medians = np.nanmedian(X_arr, axis=0)
        # If an entire column is NaN, median is NaN — fill with 0.
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        inds = np.where(nan_mask)
        X_arr[inds] = col_medians[inds[1]]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_arr)

    chi2_scores, _ = chi2(X_scaled, y)

    if np.any(np.isnan(chi2_scores)):
        logger.warning(
            "Chi-Square produced NaN scores for %d features (likely constant after scaling); replacing with 0.",
            int(np.sum(np.isnan(chi2_scores))),
        )
        chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)

    selected_indices = np.argsort(chi2_scores)[::-1][:n_target_features]

    results = {
        'selected_indices': selected_indices,
        'scores': {int(idx): float(chi2_scores[idx]) for idx in selected_indices},
        'all_scores': chi2_scores,
    }
    return results, {i: float(chi2_scores[i]) for i in range(len(chi2_scores))}


def relieff_selection(X, y, n_target_features, n_neighbors=10):
    """ReliefF instance-based feature selection (pure numpy/sklearn implementation).

    For each sample, finds k nearest hits (same class) and k nearest misses
    (different class) and computes feature weights as the average difference
    in feature values to nearest misses minus the average difference to nearest
    hits.  Higher weight ⇒ more discriminative feature.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    n_target_features : int
    n_neighbors : int, default=10
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).ravel()
    n_samples, n_features = X_arr.shape

    if n_samples < 2 or n_features == 0:
        return {}, {}

    # NaN guard: NearestNeighbors requires finite inputs. Impute NaN with
    # column median (preserves distance structure better than zero-fill).
    nan_mask = np.isnan(X_arr)
    if nan_mask.any():
        col_medians = np.nanmedian(X_arr, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        inds = np.where(nan_mask)
        X_arr[inds] = col_medians[inds[1]]

    classes, class_counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        # Single class — cannot compute hits/misses
        scores = np.zeros(n_features, dtype=float)
        selected_indices = np.arange(min(n_target_features, n_features))
        results = {
            'selected_indices': selected_indices,
            'scores': {int(idx): 0.0 for idx in selected_indices},
            'all_scores': scores,
        }
        return results, {i: 0.0 for i in range(n_features)}

    from sklearn.neighbors import NearestNeighbors

    # Precompute per-feature range for normalisation
    feat_range = np.ptp(X_arr, axis=0)
    feat_range[feat_range == 0] = 1.0  # avoid division by zero

    weights = np.zeros(n_features, dtype=float)

    # Build per-class neighbour indices
    class_indices = {c: np.where(y_arr == c)[0] for c in classes}
    class_prior = {c: cnt / n_samples for c, cnt in zip(classes, class_counts)}

    # Fit a NearestNeighbors model per class
    nn_models = {}
    for c in classes:
        idx_c = class_indices[c]
        k = min(n_neighbors + 1, len(idx_c))  # +1 because query point may be in same set
        if k < 1:
            continue
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
        nn.fit(X_arr[idx_c])
        nn_models[c] = nn

    for i in range(n_samples):
        xi = X_arr[i : i + 1]
        yi = y_arr[i]

        # --- Nearest hits (same class) ---
        if yi in nn_models:
            idx_c = class_indices[yi]
            k_hit = min(n_neighbors + 1, len(idx_c))
            dists, local_idxs = nn_models[yi].kneighbors(xi, n_neighbors=k_hit)
            global_idxs = idx_c[local_idxs[0]]
            # Exclude the sample itself
            mask = global_idxs != i
            hit_idxs = global_idxs[mask][:n_neighbors]
            if len(hit_idxs) > 0:
                diff_hit = np.mean(np.abs(X_arr[hit_idxs] - xi) / feat_range, axis=0)
            else:
                diff_hit = np.zeros(n_features, dtype=float)
        else:
            diff_hit = np.zeros(n_features, dtype=float)

        # --- Nearest misses (different classes, probability-weighted) ---
        diff_miss = np.zeros(n_features, dtype=float)
        for c in classes:
            if c == yi:
                continue
            if c not in nn_models:
                continue
            idx_c = class_indices[c]
            k_miss = min(n_neighbors, len(idx_c))
            if k_miss < 1:
                continue
            dists, local_idxs = nn_models[c].kneighbors(xi, n_neighbors=k_miss)
            miss_idxs = idx_c[local_idxs[0]]
            denom = max(1e-12, 1.0 - class_prior[yi])
            w = class_prior[c] / denom
            diff_miss += w * np.mean(np.abs(X_arr[miss_idxs] - xi) / feat_range, axis=0)

        weights += diff_miss - diff_hit

    weights /= n_samples

    selected_indices = np.argsort(weights)[::-1][:n_target_features]

    results = {
        'selected_indices': selected_indices,
        'scores': {int(idx): float(weights[idx]) for idx in selected_indices},
        'all_scores': weights,
    }
    return results, {i: float(weights[i]) for i in range(n_features)}

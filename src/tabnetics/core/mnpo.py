"""Shared MNPO (Nash Multi-Portfolio Optimization) utilities.

This module centralizes small pieces of reusable MNPO machinery used by both
feature selection (FS) and distribution fitting (DF). The goal is to avoid
having multiple subtly divergent copies of TriTrust weighting, payoff
aggregation, and mirror-descent dynamics.

All helpers are intentionally lightweight, NumPy-only, and keep defaults
conservative to preserve existing behavior in callers.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import warnings


def normalize_vector_01(values: Sequence[float]) -> np.ndarray:
    """Min-max normalize to [0, 1] with a safe fallback to 0.5 when constant."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if abs(max_v - min_v) < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - min_v) / (max_v - min_v)


def pairwise_pref_from_scalar(
    scalar_i: float,
    scalar_j: float,
    *,
    tie_margin: float = 0.02,
    temperature: Optional[float] = None,
) -> float:
    """Convert scalar oracle values into a preference probability p(i beats j).

    Notes:
        This is a logistic preference model with an explicit tie band:
        - If |scalar_i - scalar_j| <= tie_margin, return 0.5 exactly.
        - Otherwise, return sigmoid((scalar_i - scalar_j) / scale).

        If `temperature` is provided (QRE-style smoothing), it acts as the scale.
        Larger temperatures push preferences closer to 0.5.
    """
    diff = float(scalar_i - scalar_j)
    if abs(diff) <= float(tie_margin):
        return 0.5
    if temperature is not None:
        try:
            scale = float(temperature)
        except Exception as exc:
            scale = float("nan")
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float("nan")
    else:
        scale = float("nan")
    if not np.isfinite(scale):
        scale = max(float(tie_margin), 1e-4)
    scale = max(scale, 1e-6)
    x = float(diff / scale)
    x = float(np.clip(x, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def pairwise_pref_from_fold_scores(
    scores_i: Sequence[float],
    scores_j: Sequence[float],
    *,
    pairwise_delta: float,
) -> float:
    """Empirical pairwise preference from repeated-fold scores with tie handling."""
    arr_i = np.asarray(scores_i, dtype=float).ravel()
    arr_j = np.asarray(scores_j, dtype=float).ravel()
    n = int(min(arr_i.size, arr_j.size))
    if n <= 0:
        return 0.5
    delta = np.asarray(arr_i[:n] - arr_j[:n], dtype=float).ravel()
    finite = np.isfinite(delta)
    if int(np.sum(finite)) <= 0:
        return 0.5
    d = delta[finite]
    thr = float(max(0.0, pairwise_delta))
    wins = float(np.mean(d > thr))
    losses = float(np.mean(d < -thr))
    ties = float(max(0.0, 1.0 - wins - losses))
    return float(wins + 0.5 * ties)


def pairwise_pref_logistic(
    scores_i: Sequence[float],
    scores_j: Sequence[float],
    *,
    pairwise_delta: float = 0.0,
    epsilon: float = 1e-6,
) -> float:
    """Continuous pairwise preference from paired fold-score differences.

    Uses a sigmoid-transformed paired t-statistic with a practical-equivalence
    margin. Differences whose mean magnitude stays within ``pairwise_delta``
    return ``0.5`` exactly; larger effects are smoothed rather than hard-voted.
    """
    arr_i = np.asarray(scores_i, dtype=float).ravel()
    arr_j = np.asarray(scores_j, dtype=float).ravel()
    n = int(min(arr_i.size, arr_j.size))
    if n <= 0:
        return 0.5
    delta = np.asarray(arr_i[:n] - arr_j[:n], dtype=float).ravel()
    finite = np.isfinite(delta)
    if int(np.sum(finite)) <= 0:
        return 0.5
    d = np.asarray(delta[finite], dtype=float).ravel()
    if d.size <= 0:
        return 0.5

    mean_d = float(np.mean(d))
    delta_margin = float(max(0.0, float(pairwise_delta)))
    if np.all(np.abs(d) <= delta_margin) or abs(mean_d) <= delta_margin:
        return 0.5
    effect = float(np.sign(mean_d) * max(0.0, abs(mean_d) - delta_margin))

    if d.size <= 1:
        scale = max(float(epsilon), delta_margin, abs(effect), 1e-6)
        x = float(np.clip(effect / scale, -50.0, 50.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    std_d = float(np.std(d, ddof=1))
    if not np.isfinite(std_d):
        std_d = 0.0
    se = float(std_d / np.sqrt(float(max(1, d.size))))
    if se <= float(epsilon):
        se = max(float(epsilon), delta_margin, 1e-6)

    t_stat = float(effect / max(se, float(epsilon)))
    t_stat = float(np.clip(t_stat, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-t_stat)))


def james_stein_shrinkage(
    weights: Dict[str, float],
    *,
    effective_n: Optional[float] = None,
    sigma2: Optional[float] = None,
) -> Dict[str, float]:
    """Shrink noisy oracle weights toward the grand mean (positive-part JS).

    When ``sigma2`` is not supplied, we use a conservative bounded-support
    variance model derived from ``effective_n``:
    - weights in ``[0, 1]`` use ``sigma2 <= 0.25 / n_eff``
    - otherwise use ``sigma2 <= 1.0 / n_eff`` for weights in ``[-1, 1]``
    """
    keys = [str(k) for k in weights.keys()]
    arr = np.asarray([float(weights[k]) for k in keys], dtype=float)
    k = int(arr.size)
    if k <= 2:
        return dict(weights)
    mu = float(np.mean(arr))
    centered = arr - mu
    ss = float(np.sum(centered * centered))
    if ss <= 1e-12:
        return {key: float(mu) for key in keys}
    sigma2_val: float
    if sigma2 is None:
        n_eff = float(max(1.0, float(effective_n) if effective_n is not None else 1.0))
        if bool(np.all(arr >= -1e-12) and np.all(arr <= 1.0 + 1e-12)):
            sigma2_val = 0.25 / n_eff
        else:
            sigma2_val = 1.0 / n_eff
    else:
        sigma2_val = float(sigma2)
    if not np.isfinite(sigma2_val) or sigma2_val < 0.0:
        sigma2_val = 0.0
    shrink = float(max(0.0, min(1.0, 1.0 - ((k - 3.0) * sigma2_val / max(1e-12, ss)))))
    out = mu + shrink * centered
    return {key: float(out[idx]) for idx, key in enumerate(keys)}


def qre_temperature(
    values: Sequence[float],
    *,
    gamma: float = 1.0,
    tie_margin: float = 0.02,
    min_temperature: float = 1e-4,
) -> float:
    """Adaptive QRE temperature based on the dispersion of oracle utilities.

    We use: T = max(std(values), tie_margin, min_temperature) * gamma
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        base = max(float(tie_margin), float(min_temperature))
    else:
        std = float(np.std(arr))
        if not np.isfinite(std):
            std = 0.0
        base = max(std, float(tie_margin), float(min_temperature))
    g = float(gamma)
    if not np.isfinite(g) or g <= 0.0:
        g = 1.0
    return float(max(float(min_temperature), base * g))


def matrix_from_scalar_scores(
    scores: Sequence[float],
    *,
    tie_margin: float,
    use_qre_smoothing: bool,
    qre_temperature_gamma: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a pairwise preference matrix from scalar utilities.

    Returns:
        (matrix, meta) where meta includes the effective temperature.
    """
    vals = np.asarray(scores, dtype=float).ravel()
    m = int(vals.size)
    mat = np.full((m, m), 0.5, dtype=float)
    meta: Dict[str, Any] = {"temperature": None, "use_qre_smoothing": bool(use_qre_smoothing)}

    temperature: Optional[float] = None
    if bool(use_qre_smoothing) and m > 1:
        temperature = qre_temperature(
            vals,
            gamma=float(qre_temperature_gamma),
            tie_margin=float(tie_margin),
        )
        meta["temperature"] = float(temperature)

    for i, j in combinations(range(m), 2):
        p_ij = pairwise_pref_from_scalar(
            float(vals[i]),
            float(vals[j]),
            tie_margin=float(tie_margin),
            temperature=temperature,
        )
        mat[i, j] = p_ij
        mat[j, i] = 1.0 - p_ij
    np.fill_diagonal(mat, 0.5)
    return mat, meta


def lower_tail_cvar(values: Sequence[float], *, alpha: float) -> float:
    """Lower-tail CVaR (a.k.a. expected shortfall) via worst-k averaging.

    For small sample counts (e.g., few CV folds), CVaR is approximated by taking
    k = ceil(alpha * n) and returning the mean of the k smallest values.
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan")
    a = float(alpha)
    if not np.isfinite(a):
        a = 0.33
    a = float(np.clip(a, 0.0, 1.0))
    if a <= 0.0:
        k = 1
    else:
        k = int(np.ceil(a * n))
    k = int(max(1, min(n, k)))
    worst = np.sort(arr)[:k]
    return float(np.mean(worst))


def fold_regret_mean_max(score_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-candidate fold regret relative to the per-fold best.

    Args:
        score_matrix: shape (m_candidates, n_folds) where higher is better.

    Returns:
        (mean_regret, max_regret) each shape (m_candidates,). Lower regret is better.
    """
    mat = np.asarray(score_matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("score_matrix must be 2D (m, n_folds)")
    if mat.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    best = np.nanmax(mat, axis=0)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    regrets = best[None, :] - np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mean_regret = np.mean(regrets, axis=1) if regrets.shape[1] else np.zeros(regrets.shape[0], dtype=float)
    max_regret = np.max(regrets, axis=1) if regrets.shape[1] else np.zeros(regrets.shape[0], dtype=float)
    return np.asarray(mean_regret, dtype=float), np.asarray(max_regret, dtype=float)


def _rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    """Rank data with average ranks for ties (1..n). NumPy-only."""
    x = np.asarray(values, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    # Average ranks for ties in sorted order.
    x_sorted = x[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and x_sorted[end] == x_sorted[start]:
            end += 1
        if end - start > 1:
            avg = float(np.mean(ranks[order[start:end]]))
            ranks[order[start:end]] = avg
        start = end
    return ranks


def spearman_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation (rho) with average-tie ranks. NumPy-only."""
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    n = int(min(x.size, y.size))
    if n < 3:
        return 0.0
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return 0.0
    x = x[mask]
    y = y[mask]
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    rx = rx - float(np.mean(rx))
    ry = ry - float(np.mean(ry))
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.sum(rx * ry) / denom, -1.0, 1.0))


def apply_oracle_redundancy_penalty(
    oracle_weights: Dict[str, float],
    oracle_scores: Dict[str, Sequence[float]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Discount oracle weights when their scalar scores are redundant.

    Implements:
        w_o <- w_o * (1 - max_{o' != o} |rho(o,o')|^2)
    where rho is Spearman correlation computed on per-candidate oracle score vectors.
    """
    names = [str(k) for k in oracle_scores.keys()]
    meta: Dict[str, Any] = {
        "penalties": {},
        "max_abs_rho": {},
        "pairwise_rho": {},
    }
    if len(names) < 2:
        return dict(oracle_weights), meta

    pairwise: Dict[Tuple[str, str], float] = {}
    for i, j in combinations(range(len(names)), 2):
        a = names[i]
        b = names[j]
        rho = spearman_correlation(oracle_scores.get(a, ()), oracle_scores.get(b, ()))
        pairwise[(a, b)] = float(rho)
        pairwise[(b, a)] = float(rho)
    meta["pairwise_rho"] = {f"{a}|{b}": float(r) for (a, b), r in pairwise.items() if a < b}

    out = dict(oracle_weights)
    for name in names:
        max_abs = 0.0
        for other in names:
            if other == name:
                continue
            rho = float(pairwise.get((name, other), 0.0))
            max_abs = max(max_abs, abs(rho))
        penalty = float(max(0.0, 1.0 - max_abs * max_abs))
        meta["penalties"][name] = float(penalty)
        meta["max_abs_rho"][name] = float(max_abs)
        if name in out:
            out[name] = float(out[name]) * penalty
    return out, meta


def tremble_oracle_matrices(
    oracle_matrices: Dict[str, np.ndarray],
    *,
    epsilon: float,
) -> Dict[str, np.ndarray]:
    """Shrink all oracle preferences toward 0.5 (uninformative) by epsilon."""
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        eps = 0.0
    eps = float(np.clip(eps, 0.0, 1.0))
    out: Dict[str, np.ndarray] = {}
    for name, mat in oracle_matrices.items():
        arr = np.asarray(mat, dtype=float)
        trembled = (1.0 - eps) * arr + eps * 0.5
        trembled = np.clip(trembled, 0.0, 1.0)
        np.fill_diagonal(trembled, 0.5)
        out[str(name)] = trembled
    return out


def fit_tritrust_weights(
    oracle_matrices: Dict[str, np.ndarray],
    *,
    reference: str,
    allow_negative: bool,
    no_flip_oracles: Optional[Set[str]] = None,
    ref_delta_threshold: float = 0.05,
    oracle_delta_threshold: float = 0.03,
    reliability_threshold: float = 0.10,
) -> Dict[str, float]:
    """TriTrust-style trust/ignore/(optional flip) calibration via agreement.

    Args:
        oracle_matrices: dict of name -> pairwise preference matrix in [0, 1].
        reference: oracle name used as the agreement anchor.
        allow_negative: if True, reliability may be negative (flip). If False,
            negative reliability is treated as 0 (ignore).
        no_flip_oracles: optional set of oracle names that should never be
            flipped; negative reliability is clipped to 0 for these.
    """
    if reference not in oracle_matrices:
        return {name: 1.0 for name in oracle_matrices}

    no_flip_oracles = set(no_flip_oracles or set())

    ref = np.asarray(oracle_matrices[reference], dtype=float)
    n = int(ref.shape[0])
    total_pairs = max(1, n * (n - 1) // 2)
    weights: Dict[str, float] = {reference: 1.0}

    for name, mat in oracle_matrices.items():
        if name == reference:
            continue
        other = np.asarray(mat, dtype=float)

        agreements: List[float] = []
        confident_pairs = 0
        for i, j in combinations(range(n), 2):
            ref_delta = float(ref[i, j] - 0.5)
            other_delta = float(other[i, j] - 0.5)
            if abs(ref_delta) < float(ref_delta_threshold) or abs(other_delta) < float(oracle_delta_threshold):
                continue
            confident_pairs += 1
            agreements.append(1.0 if np.sign(ref_delta) == np.sign(other_delta) else 0.0)

        if confident_pairs == 0:
            weights[name] = 0.0
            continue

        agreement = float(np.mean(agreements))
        coverage = float(confident_pairs) / float(total_pairs)
        reliability = (2.0 * agreement - 1.0) * coverage

        if name in no_flip_oracles and reliability < 0:
            reliability = 0.0
        if not allow_negative and reliability < 0:
            reliability = 0.0

        if abs(reliability) < float(reliability_threshold):
            reliability = 0.0

        if allow_negative:
            weights[name] = float(np.clip(reliability, -1.0, 1.0))
        else:
            weights[name] = float(np.clip(reliability, 0.0, 1.0))

    return weights


def fit_shapley_weights(
    oracle_matrices: Dict[str, np.ndarray],
    *,
    reference: str = "performance",
    max_coalitions: int = 4096,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Exact Shapley oracle weights from coalition value differences.

    Coalition value is agreement with the reference oracle matrix:
      v(S) = 1 - mean(abs(A_S - A_ref))
    where A_S is the equal-weight aggregate payoff matrix from coalition S.
    """
    names = [str(k) for k in oracle_matrices.keys()]
    k = int(len(names))
    if k == 0:
        return {}, {"applied": False, "reason": "no_oracles", "n_oracles": 0}
    if k == 1:
        return {names[0]: 1.0}, {"applied": True, "reason": "single_oracle", "n_oracles": 1}

    coalition_count = int(2 ** k)
    max_coalitions_i = int(max(2, max_coalitions))
    if coalition_count > max_coalitions_i:
        warnings.warn(
            "Shapley weighting skipped: coalition count exceeds shapley_n_coalitions_max.",
            RuntimeWarning,
        )
        uniform = {name: 1.0 / float(k) for name in names}
        return uniform, {
            "applied": False,
            "reason": "coalition_cap_exceeded",
            "n_oracles": int(k),
            "coalition_count": int(coalition_count),
            "max_coalitions": int(max_coalitions_i),
        }

    ref_name = str(reference) if str(reference) in oracle_matrices else names[0]
    ref = np.asarray(oracle_matrices[ref_name], dtype=float)
    eps = 1e-12
    fact = [1.0]
    for i in range(1, k + 1):
        fact.append(float(fact[-1] * i))
    fact_k = float(fact[k])

    coalition_value: Dict[int, float] = {0: 0.0}

    def _aggregate_from_mask(mask: int) -> np.ndarray:
        mats = {}
        for idx, name in enumerate(names):
            if (mask >> idx) & 1:
                mats[name] = np.asarray(oracle_matrices[name], dtype=float)
        if not mats:
            return np.full_like(ref, 0.5, dtype=float)
        w = {name: 1.0 for name in mats}
        return aggregate_payoff_matrix(mats, w)

    for mask in range(1, coalition_count):
        agg = _aggregate_from_mask(mask)
        agreement = 1.0 - float(np.mean(np.abs(agg - ref)))
        coalition_value[mask] = float(np.clip(agreement, -1.0, 1.0))

    shapley = np.zeros(k, dtype=float)
    for i in range(k):
        bit = 1 << i
        for mask in range(coalition_count):
            if mask & bit:
                continue
            s_size = bin(mask).count("1")  # Python 3.9-compatible popcount
            coeff = float(fact[s_size] * fact[k - s_size - 1] / max(eps, fact_k))
            marginal = float(coalition_value[mask | bit] - coalition_value.get(mask, 0.0))
            shapley[i] += coeff * marginal

    shapley = np.asarray(np.nan_to_num(shapley, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    if float(np.sum(np.abs(shapley))) <= eps:
        weights = np.full(k, 1.0 / float(k), dtype=float)
    else:
        shift = float(np.min(shapley))
        shifted = shapley - shift if shift < 0 else shapley.copy()
        if float(np.sum(shifted)) <= eps:
            weights = np.full(k, 1.0 / float(k), dtype=float)
        else:
            weights = shifted / float(np.sum(shifted))

    return (
        {name: float(weights[idx]) for idx, name in enumerate(names)},
        {
            "applied": True,
            "reason": "ok",
            "reference": str(ref_name),
            "n_oracles": int(k),
            "coalition_count": int(coalition_count),
            "max_coalitions": int(max_coalitions_i),
            "raw_shapley": {name: float(shapley[idx]) for idx, name in enumerate(names)},
        },
    )


def aggregate_payoff_matrix(
    oracle_matrices: Dict[str, np.ndarray],
    oracle_weights: Dict[str, float],
) -> np.ndarray:
    """Aggregate preference matrices into an anti-symmetric payoff matrix."""
    m = int(next(iter(oracle_matrices.values())).shape[0])
    payoff = np.zeros((m, m), dtype=float)
    eps = 1e-6

    for i, j in combinations(range(m), 2):
        combined_logit = 0.0
        for name, mat in oracle_matrices.items():
            w = float(oracle_weights.get(name, 1.0))
            p_ij = float(np.clip(mat[i, j], eps, 1.0 - eps))
            combined_logit += w * float(np.log(p_ij / (1.0 - p_ij)))
        combined_logit = float(np.clip(combined_logit, -20.0, 20.0))
        p_ij = float(1.0 / (1.0 + np.exp(-combined_logit)))
        payoff[i, j] = 2.0 * p_ij - 1.0
        payoff[j, i] = -payoff[i, j]

    return payoff


def shrink_payoff_matrix(
    payoff: np.ndarray,
    *,
    kappa: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Shrink an anti-symmetric payoff matrix toward zero."""
    mat = np.asarray(payoff, dtype=float)
    out = np.array(mat, copy=True, dtype=float)
    kappa_val = float(max(0.0, kappa))
    meta: Dict[str, Any] = {
        "applied": False,
        "kappa": float(kappa_val),
        "variance": float("nan"),
        "alpha": 0.0,
    }
    if out.ndim != 2 or out.shape[0] != out.shape[1] or out.size <= 0 or kappa_val <= 0.0:
        return out, meta
    iu = np.triu_indices(out.shape[0], k=1)
    upper = np.asarray(out[iu], dtype=float).ravel()
    upper = upper[np.isfinite(upper)]
    if upper.size <= 0:
        return out, meta
    var_hat = float(np.var(upper, ddof=1)) if upper.size > 1 else 0.0
    if not np.isfinite(var_hat) or var_hat < 0.0:
        var_hat = 0.0
    alpha = float(np.clip((kappa_val ** 2) / max(1e-12, (kappa_val ** 2) + var_hat), 0.0, 1.0))
    out *= float(1.0 - alpha)
    meta.update(
        {
            "applied": bool(alpha > 0.0),
            "variance": float(var_hat),
            "alpha": float(alpha),
        }
    )
    return out, meta


class _HistoryList(list):
    """List subclass that can carry a ``kl_values`` attribute."""
    kl_values: List[float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_values = []


def mirror_descent_reference_regularized(
    payoff: np.ndarray,
    reference_prior: np.ndarray,
    *,
    steps: int,
    eta: float,
    lambda_: float,
    tol_kl: float = 1e-7,
    return_history: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    """Reference-regularized mirror descent on the simplex.

    Returns:
        If return_history is False: equilibrium weights p.
        If return_history is True: (p, history) where history includes the
        initial prior.  The ``kl_history`` attribute is attached to the
        returned ``history`` list as ``history.kl_values`` (list of floats,
        one per iteration).
    """
    eps = 1e-10
    p = np.asarray(reference_prior, dtype=float).copy()
    p = np.clip(p, eps, None)
    p = p / float(np.sum(p))
    history = _HistoryList([p.copy()])
    kl_history: List[float] = []

    for _ in range(int(max(1, steps))):
        utility = payoff @ p
        log_ratio = np.log(np.clip(p, eps, None)) - np.log(np.clip(reference_prior, eps, None))
        exponent = float(eta) * utility - float(lambda_) * log_ratio
        exponent = exponent - float(np.max(exponent))
        p_new = p * np.exp(exponent)
        total = float(np.sum(p_new))
        if total <= eps:
            break
        p_new = p_new / total
        kl = float(
            np.sum(p_new * (np.log(np.clip(p_new, eps, None)) - np.log(np.clip(p, eps, None))))
        )
        kl_history.append(kl)
        history.append(p_new.copy())
        p = p_new
        if kl < float(tol_kl):
            break

    if return_history:
        history.kl_values = kl_history
        return p, history
    return p


def compute_banzhaf_values(
    oracle_matrices: Dict[str, np.ndarray],
    *,
    value_fn: Optional[Any] = None,
    reference: str = "performance",
    max_coalitions: int = 8192,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Compute Banzhaf values for oracle weighting.

    Unlike Shapley, Banzhaf values use uniform coalition weighting
    (each coalition equally likely) rather than size-dependent weighting.
    This gives strictly larger safety margin under noisy evaluations
    (Wang & Jia 2022, arXiv:2205.15466).

    Parameters
    ----------
    oracle_matrices : dict
        Oracle name -> m×m pairwise preference matrix.
    value_fn : callable, optional
        Coalition value function v(S) -> float.  If None, uses agreement
        with the reference oracle (same as Shapley).
    reference : str
        Reference oracle name for default value function.
    max_coalitions : int
        Maximum coalitions to enumerate.  If 2^k exceeds this, falls
        back to ``kernel_banzhaf_values`` (sampling estimator).

    Returns
    -------
    weights : dict
        Oracle name -> Banzhaf weight (normalized to sum to 1).
    meta : dict
        Diagnostics including raw Banzhaf values and method used.
    """
    names = [str(k) for k in oracle_matrices.keys()]
    k = len(names)
    if k == 0:
        return {}, {"applied": False, "reason": "no_oracles", "n_oracles": 0}
    if k == 1:
        return {names[0]: 1.0}, {"applied": True, "reason": "single_oracle", "n_oracles": 1}

    coalition_count = 2 ** k
    if coalition_count > max_coalitions:
        return kernel_banzhaf_values(
            oracle_matrices,
            reference=reference,
            n_samples=min(max_coalitions, 4096),
        )

    ref_name = str(reference) if str(reference) in oracle_matrices else names[0]
    ref = np.asarray(oracle_matrices[ref_name], dtype=float)
    eps = 1e-12

    # Default value function: agreement with reference oracle.
    if value_fn is None:
        def value_fn(mats_subset):
            if not mats_subset:
                return 0.0
            w = {n: 1.0 for n in mats_subset}
            agg = aggregate_payoff_matrix(mats_subset, w)
            return float(1.0 - np.mean(np.abs(agg - ref)))

    # Pre-compute coalition values.
    coalition_vals = {}
    for mask in range(coalition_count):
        subset = {}
        for idx, name in enumerate(names):
            if (mask >> idx) & 1:
                subset[name] = np.asarray(oracle_matrices[name], dtype=float)
        coalition_vals[mask] = float(value_fn(subset))

    # Banzhaf: uniform weighting over coalitions (1/2^(k-1) per marginal).
    banzhaf = np.zeros(k, dtype=float)
    for i in range(k):
        bit = 1 << i
        marginals = []
        for mask in range(coalition_count):
            if mask & bit:
                continue  # i already in coalition
            marginal = coalition_vals[mask | bit] - coalition_vals[mask]
            marginals.append(marginal)
        banzhaf[i] = float(np.mean(marginals)) if marginals else 0.0

    banzhaf = np.nan_to_num(banzhaf, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to weights (shift non-negative, then normalize).
    if float(np.sum(np.abs(banzhaf))) <= eps:
        weights = np.full(k, 1.0 / float(k), dtype=float)
    else:
        shift = float(np.min(banzhaf))
        shifted = banzhaf - shift if shift < 0 else banzhaf.copy()
        total = float(np.sum(shifted))
        if total <= eps:
            weights = np.full(k, 1.0 / float(k), dtype=float)
        else:
            weights = shifted / total

    return (
        {name: float(weights[idx]) for idx, name in enumerate(names)},
        {
            "applied": True,
            "reason": "ok",
            "method": "exact",
            "reference": str(ref_name),
            "n_oracles": k,
            "coalition_count": coalition_count,
            "raw_banzhaf": {name: float(banzhaf[idx]) for idx, name in enumerate(names)},
        },
    )


def kernel_banzhaf_values(
    oracle_matrices: Dict[str, np.ndarray],
    *,
    reference: str = "performance",
    n_samples: int = 4096,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Kernel Banzhaf regression estimator (Liu et al. 2024, arXiv:2410.08336).

    Samples random coalitions uniformly and estimates Banzhaf values via
    weighted least-squares regression of coalition values on binary
    membership indicators.

    Parameters
    ----------
    oracle_matrices : dict
        Oracle name -> m×m pairwise preference matrix.
    reference : str
        Reference oracle name.
    n_samples : int
        Number of random coalitions to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    weights : dict
        Oracle name -> Banzhaf weight (normalized).
    meta : dict
        Diagnostics.
    """
    names = [str(k) for k in oracle_matrices.keys()]
    k = len(names)
    if k == 0:
        return {}, {"applied": False, "reason": "no_oracles", "n_oracles": 0}
    if k == 1:
        return {names[0]: 1.0}, {"applied": True, "reason": "single_oracle", "n_oracles": 1}

    ref_name = str(reference) if str(reference) in oracle_matrices else names[0]
    ref = np.asarray(oracle_matrices[ref_name], dtype=float)
    rng = np.random.RandomState(seed)
    eps = 1e-12

    # Sample binary membership vectors uniformly.
    Z = rng.randint(0, 2, size=(n_samples, k)).astype(float)
    values = np.zeros(n_samples, dtype=float)

    for s in range(n_samples):
        subset = {}
        for idx, name in enumerate(names):
            if Z[s, idx] > 0.5:
                subset[name] = np.asarray(oracle_matrices[name], dtype=float)
        if not subset:
            values[s] = 0.0
        else:
            w = {n: 1.0 for n in subset}
            agg = aggregate_payoff_matrix(subset, w)
            values[s] = float(1.0 - np.mean(np.abs(agg - ref)))

    # OLS regression: values = Z @ beta + intercept.
    Z_aug = np.hstack([np.ones((n_samples, 1)), Z])
    try:
        beta, _, _, _ = np.linalg.lstsq(Z_aug, values, rcond=None)
        banzhaf = beta[1:]  # Exclude intercept.
    except np.linalg.LinAlgError:
        banzhaf = np.full(k, 1.0 / float(k), dtype=float)

    banzhaf = np.nan_to_num(banzhaf, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize.
    if float(np.sum(np.abs(banzhaf))) <= eps:
        weights = np.full(k, 1.0 / float(k), dtype=float)
    else:
        shift = float(np.min(banzhaf))
        shifted = banzhaf - shift if shift < 0 else banzhaf.copy()
        total = float(np.sum(shifted))
        if total <= eps:
            weights = np.full(k, 1.0 / float(k), dtype=float)
        else:
            weights = shifted / total

    return (
        {name: float(weights[idx]) for idx, name in enumerate(names)},
        {
            "applied": True,
            "reason": "ok",
            "method": "kernel_regression",
            "reference": str(ref_name),
            "n_oracles": k,
            "n_samples": n_samples,
            "seed": seed,
            "raw_banzhaf": {name: float(banzhaf[idx]) for idx, name in enumerate(names)},
        },
    )

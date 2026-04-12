"""MNPO oracle preference estimation, PID, and diversity scoring.

Standalone functions extracted from FeatureSelector (base.py).
"""

import logging
import numpy as np
from itertools import combinations
from typing import Any, Dict, Tuple
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

from tabnetics.datasets.tier_classifier import (
    heuristic_tier as _heuristic_dataset_tier,
    normalized_complexity_score as _normalized_complexity_score,
)

try:
    from tabnetics.core.compat import make_logistic_regression
except Exception as exc:
    from tabnetics.core.compat import make_logistic_regression  # type: ignore

logger = logging.getLogger(__name__)

# -- MNPO core imports --
try:
    from ...mnpo_core import (
        aggregate_payoff_matrix as _mnpo_aggregate_payoff_matrix,
        fit_tritrust_weights as _mnpo_fit_tritrust_weights,
        lower_tail_cvar as _mnpo_lower_tail_cvar,
        matrix_from_scalar_scores as _mnpo_matrix_from_scalar_scores,
        mirror_descent_reference_regularized as _mnpo_mirror_descent_reference_regularized,
        normalize_vector_01 as _mnpo_normalize_vector_01,
        pairwise_pref_from_fold_scores as _mnpo_pairwise_pref_from_fold_scores,
        pairwise_pref_logistic as _mnpo_pairwise_pref_logistic,
        pairwise_pref_from_scalar as _mnpo_pairwise_pref_from_scalar,
    )
except Exception as exc:
    try:
        from tabnetics.core.mnpo import (
            aggregate_payoff_matrix as _mnpo_aggregate_payoff_matrix,
            fit_tritrust_weights as _mnpo_fit_tritrust_weights,
            lower_tail_cvar as _mnpo_lower_tail_cvar,
            matrix_from_scalar_scores as _mnpo_matrix_from_scalar_scores,
            mirror_descent_reference_regularized as _mnpo_mirror_descent_reference_regularized,
            normalize_vector_01 as _mnpo_normalize_vector_01,
            pairwise_pref_from_fold_scores as _mnpo_pairwise_pref_from_fold_scores,
            pairwise_pref_logistic as _mnpo_pairwise_pref_logistic,
            pairwise_pref_from_scalar as _mnpo_pairwise_pref_from_scalar,
        )
    except Exception as exc:
        from tabnetics.core.mnpo import (  # type: ignore[import-untyped]
            aggregate_payoff_matrix as _mnpo_aggregate_payoff_matrix,
            fit_tritrust_weights as _mnpo_fit_tritrust_weights,
            lower_tail_cvar as _mnpo_lower_tail_cvar,
            matrix_from_scalar_scores as _mnpo_matrix_from_scalar_scores,
            mirror_descent_reference_regularized as _mnpo_mirror_descent_reference_regularized,
            normalize_vector_01 as _mnpo_normalize_vector_01,
            pairwise_pref_from_fold_scores as _mnpo_pairwise_pref_from_fold_scores,
            pairwise_pref_logistic as _mnpo_pairwise_pref_logistic,
            pairwise_pref_from_scalar as _mnpo_pairwise_pref_from_scalar,
        )


# ---------------------------------------------------------------------------
# 1. pairwise_pref_from_fold_scores
# ---------------------------------------------------------------------------
def pairwise_pref_from_fold_scores(scores_i, scores_j, pairwise_delta):
    """Preference probability from repeated-CV score arrays with explicit ties."""
    return float(
        _mnpo_pairwise_pref_from_fold_scores(
            scores_i,
            scores_j,
            pairwise_delta=float(pairwise_delta),
        )
    )


def pairwise_pref_logistic(scores_i, scores_j, pairwise_delta):
    """Continuous logistic preference from repeated-CV score arrays."""
    return float(
        _mnpo_pairwise_pref_logistic(
            scores_i,
            scores_j,
            pairwise_delta=float(pairwise_delta),
        )
    )


# ---------------------------------------------------------------------------
# 2. pairwise_pref_from_scalar
# ---------------------------------------------------------------------------
def pairwise_pref_from_scalar(scalar_i, scalar_j, tie_margin=0.02, temperature=None):
    """Preference probability from scalar oracle values."""
    return float(
        _mnpo_pairwise_pref_from_scalar(
            float(scalar_i),
            float(scalar_j),
            tie_margin=float(tie_margin),
            temperature=temperature,
        )
    )


def _compute_conformal_interval_width(fold_scores, alpha=0.10) -> float:
    """Cross-conformal interval width proxy from fold-score dispersion."""
    scores = np.asarray(fold_scores, dtype=float).ravel()
    if scores.size == 0:
        return float("inf")
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("inf")
    alpha_eff = float(np.clip(alpha, 1e-3, 0.49))
    med = float(np.median(scores))
    abs_dev = np.abs(scores - med)
    width = float(np.quantile(abs_dev, 1.0 - alpha_eff))
    if not np.isfinite(width):
        return float("inf")
    return float(max(0.0, width))


def compute_shapley_bayesian_shrinkage(
    *,
    evaluation: Dict[str, Dict[str, Any]],
    candidate_names,
    prior_strength: float,
) -> Dict[str, Any]:
    """Compute a continuous shrinkage factor for Shapley weights.

    Shrinkage follows a simple empirical-Bayes form:
      lambda = prior_strength / (prior_strength + n_eff)
    where n_eff is the minimum observed fold count across candidates.
    """
    folds = []
    for name in candidate_names:
        try:
            vals = np.asarray(
                evaluation.get(name, {}).get("performance_scores", np.asarray([], dtype=float)),
                dtype=float,
            ).ravel()
            folds.append(int(vals.size))
        except Exception as exc:
            folds.append(0)
    n_eff = int(min(folds)) if folds else 0
    prior = float(max(1e-6, prior_strength))
    lam = float(np.clip(prior / (prior + max(0, n_eff)), 0.0, 0.95))
    return {
        "n_eff_folds": int(n_eff),
        "prior_strength": float(prior),
        "shrinkage_lambda": float(lam),
    }


def apply_bayesian_shrinkage_to_weights(
    weights: Dict[str, float],
    *,
    shrinkage_lambda: float,
) -> Dict[str, float]:
    """Shrink oracle weights toward a uniform prior mass."""
    if not isinstance(weights, dict) or not weights:
        return {}
    names = list(weights.keys())
    vec = np.asarray([float(weights.get(n, 0.0)) for n in names], dtype=float)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    if vec.size == 0:
        return {}
    lam = float(np.clip(shrinkage_lambda, 0.0, 0.95))
    uni = np.full(vec.size, 1.0 / float(max(1, vec.size)), dtype=float)
    shrunk = (1.0 - lam) * vec + lam * uni
    shrunk = np.asarray(np.nan_to_num(shrunk, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    total = float(np.sum(shrunk))
    if total <= 1e-12:
        shrunk = uni
    else:
        shrunk = shrunk / total
    return {str(names[i]): float(shrunk[i]) for i in range(len(names))}


def _entropy_from_codes(codes: np.ndarray) -> float:
    arr = np.asarray(codes, dtype=int).ravel()
    if arr.size == 0:
        return 0.0
    _, inv = np.unique(arr, return_inverse=True)
    counts = np.bincount(inv)
    probs = counts.astype(float) / float(max(1, np.sum(counts)))
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


def _mutual_info_codes(x_codes: np.ndarray, y_codes: np.ndarray) -> float:
    x = np.asarray(x_codes, dtype=int).ravel()
    y = np.asarray(y_codes, dtype=int).ravel()
    n = int(min(x.size, y.size))
    if n < 4:
        return 0.0
    x = x[:n]
    y = y[:n]
    _, x_inv = np.unique(x, return_inverse=True)
    _, y_inv = np.unique(y, return_inverse=True)
    nx = int(np.max(x_inv)) + 1 if x_inv.size else 0
    ny = int(np.max(y_inv)) + 1 if y_inv.size else 0
    if nx <= 1 or ny <= 1:
        return 0.0
    table = np.zeros((nx, ny), dtype=float)
    np.add.at(table, (x_inv, y_inv), 1.0)
    pxy = table / float(n)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    denom = px * py
    mask = pxy > 0
    if not np.any(mask):
        return 0.0
    return float(np.sum(pxy[mask] * np.log(np.clip(pxy[mask] / np.clip(denom[mask], 1e-12, None), 1e-12, None))))


def _discretize_feature(values: np.ndarray, n_bins: int = 3) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return np.zeros(0, dtype=int)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if np.allclose(arr, arr[0]):
        return np.zeros(arr.size, dtype=int)
    bins = int(max(2, min(8, n_bins)))
    try:
        edges = np.quantile(arr, np.linspace(0.0, 1.0, bins + 1))
        edges = np.unique(np.asarray(edges, dtype=float))
    except Exception as exc:
        return np.zeros(arr.size, dtype=int)
    if edges.size <= 2:
        return np.zeros(arr.size, dtype=int)
    return np.asarray(np.digitize(arr, edges[1:-1], right=False), dtype=int)


def _compute_interaction_density_score(
    X_pool: np.ndarray,
    y: np.ndarray,
    selected_indices: np.ndarray,
    *,
    pool_size_cap: int,
    pair_cap: int,
) -> float:
    x_arr = np.asarray(X_pool, dtype=float)
    y_arr = np.asarray(y).ravel()
    if x_arr.ndim != 2 or x_arr.shape[0] < 8:
        return 0.0
    idx = np.asarray(selected_indices, dtype=int).ravel()
    idx = np.array(sorted(set(int(i) for i in idx if 0 <= int(i) < x_arr.shape[1])), dtype=int)
    if idx.size < 2:
        return 0.0
    cap = int(max(2, pool_size_cap))
    if idx.size > cap:
        idx = idx[:cap]

    _, y_codes = np.unique(y_arr, return_inverse=True)
    y_codes = np.asarray(y_codes, dtype=int)
    y_entropy = _entropy_from_codes(y_codes)
    if y_entropy <= 1e-12:
        return 0.0

    feat_codes = {}
    feat_mi = {}
    for f_idx in idx.tolist():
        code = _discretize_feature(x_arr[:, int(f_idx)], n_bins=3)
        feat_codes[int(f_idx)] = code
        feat_mi[int(f_idx)] = _mutual_info_codes(code, y_codes)

    total_pairs = 0
    synergy_sum = 0.0
    cap_pairs = int(max(1, pair_cap))
    for i_pos, j_pos in combinations(range(idx.size), 2):
        if total_pairs >= cap_pairs:
            break
        fi = int(idx[i_pos])
        fj = int(idx[j_pos])
        ci = feat_codes.get(fi)
        cj = feat_codes.get(fj)
        if ci is None or cj is None or ci.size != cj.size:
            continue
        joint_code = ci * 16 + cj
        mi_joint = _mutual_info_codes(joint_code, y_codes)
        mi_base = max(float(feat_mi.get(fi, 0.0)), float(feat_mi.get(fj, 0.0)))
        synergy = float(max(0.0, mi_joint - mi_base))
        synergy_sum += synergy
        total_pairs += 1

    if total_pairs <= 0:
        return 0.0
    # Normalize by target entropy to keep scale stable across datasets.
    return float((synergy_sum / float(total_pairs)) / max(1e-12, y_entropy))


# ---------------------------------------------------------------------------
# 3. estimate_oracle_preferences
# ---------------------------------------------------------------------------
def estimate_oracle_preferences(
    candidate_names, evaluation,
    *,
    pairwise_delta,
    fold_preference_mode="vote",
    use_cvar=False,
    cvar_alpha=0.33,
    use_tail_risk_oracle,
    tail_risk_alpha,
    use_qre_smoothing,
    qre_temperature_gamma,
    use_regret_oracle,
    use_stability_oracle,
    use_complexity_oracle,
    use_robust_oracle,
    use_diversity_oracle,
    diversity_oracle_mode,
    diversity_redundancy_weight,
    diversity_complementarity_weight,
    performance_oracle_mode="single",
    weighting_mode="tritrust",
    shapley_n_coalitions_max=4096,
    use_interaction_oracle=False,
    interaction_oracle_min_n_train=150,
    interaction_oracle_pool_size_cap=64,
    interaction_oracle_pair_cap=20000,
    use_ubayfs=False,
    ubayfs_n_bootstrap=32,
    ubayfs_min_n=100,
    ubayfs_prior_weight=0.0,
    use_conformal_uq=False,
    conformal_uq_alpha=0.10,
    conformal_uq_min_folds=5,
    use_conformal_efficiency=False,
    conformal_efficiency_method="split",
    complexity_conditioning=False,
    X_pool=None,
    oracle_config=None,
    random_state=42,
):
    """Build pairwise preference matrices for enabled oracles."""
    m = len(candidate_names)
    oracle_matrices = {}
    oracle_scores = {}
    oracle_components = {}
    oracle_pairwise_meta = {}

    def _oc(name: str, fallback):
        if oracle_config is None:
            return fallback
        return getattr(oracle_config, name, fallback)

    pairwise_delta = float(_oc("pairwise_delta", pairwise_delta))
    fold_preference_mode = str(_oc("fold_preference_mode", fold_preference_mode) or "vote").strip().lower()
    if fold_preference_mode not in {"vote", "logistic"}:
        fold_preference_mode = "vote"
    use_cvar = bool(_oc("use_cvar", use_cvar))
    cvar_alpha = float(_oc("cvar_alpha", cvar_alpha))
    use_stability_oracle = bool(_oc("use_stability_oracle", use_stability_oracle))
    use_complexity_oracle = bool(_oc("use_complexity_oracle", use_complexity_oracle))
    use_robust_oracle = bool(_oc("use_robust_oracle", use_robust_oracle))
    use_diversity_oracle = bool(_oc("use_diversity_oracle", use_diversity_oracle))
    diversity_oracle_mode = str(_oc("diversity_mode", diversity_oracle_mode))
    diversity_redundancy_weight = float(
        _oc("diversity_redundancy_weight", diversity_redundancy_weight)
    )
    diversity_complementarity_weight = float(
        _oc("diversity_complementarity_weight", diversity_complementarity_weight)
    )
    performance_oracle_mode = str(_oc("performance_oracle_mode", performance_oracle_mode))
    weighting_mode = str(_oc("weighting_mode", weighting_mode))
    shapley_n_coalitions_max = int(_oc("shapley_n_coalitions_max", shapley_n_coalitions_max))
    use_interaction_oracle = bool(_oc("use_interaction_oracle", use_interaction_oracle))
    interaction_oracle_min_n_train = int(
        _oc("interaction_oracle_min_n_train", interaction_oracle_min_n_train)
    )
    interaction_oracle_pool_size_cap = int(
        _oc("interaction_oracle_pool_size_cap", interaction_oracle_pool_size_cap)
    )
    interaction_oracle_pair_cap = int(_oc("interaction_oracle_pair_cap", interaction_oracle_pair_cap))
    use_ubayfs = bool(_oc("use_ubayfs", use_ubayfs))
    ubayfs_n_bootstrap = int(_oc("ubayfs_n_bootstrap", ubayfs_n_bootstrap))
    ubayfs_min_n = int(_oc("ubayfs_min_n", ubayfs_min_n))
    ubayfs_prior_weight = float(_oc("ubayfs_prior_weight", ubayfs_prior_weight))
    use_conformal_uq = bool(_oc("use_conformal_uq", use_conformal_uq))
    conformal_uq_alpha = float(_oc("conformal_uq_alpha", conformal_uq_alpha))
    conformal_uq_min_folds = int(_oc("conformal_uq_min_folds", conformal_uq_min_folds))
    use_conformal_efficiency = bool(
        _oc("use_conformal_efficiency", use_conformal_efficiency)
    )
    conformal_efficiency_method = str(
        _oc("conformal_efficiency_method", conformal_efficiency_method) or "split"
    ).strip().lower()
    if conformal_efficiency_method not in {"split", "aps"}:
        conformal_efficiency_method = "split"
    complexity_conditioning = bool(_oc("complexity_conditioning", complexity_conditioning))

    fold_preference_fn = (
        pairwise_pref_logistic if fold_preference_mode == "logistic" else pairwise_pref_from_fold_scores
    )

    performance_matrix = np.full((m, m), 0.5, dtype=float)
    for i, j in combinations(range(m), 2):
        name_i = candidate_names[i]
        name_j = candidate_names[j]
        p_ij = fold_preference_fn(
            evaluation[name_i]['performance_scores'],
            evaluation[name_j]['performance_scores'],
            pairwise_delta=pairwise_delta,
        )
        performance_matrix[i, j] = p_ij
        performance_matrix[j, i] = 1.0 - p_ij
    np.fill_diagonal(performance_matrix, 0.5)
    oracle_matrices['performance'] = performance_matrix
    oracle_scores['performance'] = np.array(
        [evaluation[name]['performance_mean'] for name in candidate_names], dtype=float
    )
    oracle_components["performance"] = {
        "fold_preference_mode": str(fold_preference_mode),
    }

    if use_cvar:
        cvar_scores = []
        for name in candidate_names:
            scores = np.asarray(evaluation[name].get("performance_scores", np.asarray([], dtype=float)), dtype=float)
            cvar_val = _mnpo_lower_tail_cvar(scores, alpha=float(cvar_alpha))
            if not np.isfinite(cvar_val):
                cvar_val = float(np.nanmean(scores)) if scores.size else 0.0
            cvar_scores.append(float(cvar_val))
        cvar_scores_arr = np.asarray(cvar_scores, dtype=float)
        cvar_matrix, meta = _mnpo_matrix_from_scalar_scores(
            cvar_scores_arr,
            tie_margin=0.01,
            use_qre_smoothing=use_qre_smoothing,
            qre_temperature_gamma=qre_temperature_gamma,
        )
        oracle_matrices["cvar"] = cvar_matrix
        oracle_scores["cvar"] = cvar_scores_arr
        oracle_pairwise_meta["cvar"] = dict(meta)

    if use_conformal_uq:
        min_required = int(max(2, conformal_uq_min_folds))
        fold_counts = [
            int(
                np.asarray(
                    evaluation[name].get("performance_scores", np.asarray([], dtype=float)),
                    dtype=float,
                ).size
            )
            for name in candidate_names
        ]
        min_folds_seen = int(min(fold_counts)) if fold_counts else 0
        if min_folds_seen < min_required:
            logger.warning(
                "Conformal UQ oracle skipped: min folds=%d below conformal_uq_min_folds=%d.",
                int(min_folds_seen),
                int(min_required),
            )
            oracle_components["conformal_reliability"] = {
                "enabled": True,
                "applied": False,
                "reason": "min_folds_gate",
                "min_folds_seen": int(min_folds_seen),
                "conformal_uq_min_folds": int(min_required),
                "conformal_uq_alpha": float(np.clip(conformal_uq_alpha, 1e-3, 0.49)),
            }
        else:
            widths = np.array(
                [
                    _compute_conformal_interval_width(
                        evaluation[name].get("performance_scores", np.asarray([], dtype=float)),
                        alpha=float(conformal_uq_alpha),
                    )
                    for name in candidate_names
                ],
                dtype=float,
            )
            widths = np.nan_to_num(widths, nan=np.inf, posinf=np.inf, neginf=np.inf)
            reliability = 1.0 / np.maximum(1e-8, widths)
            reliability = np.asarray(
                np.nan_to_num(reliability, nan=0.0, posinf=1e8, neginf=0.0),
                dtype=float,
            )
            conformal_matrix, meta = _mnpo_matrix_from_scalar_scores(
                reliability,
                tie_margin=0.01,
                use_qre_smoothing=use_qre_smoothing,
                qre_temperature_gamma=qre_temperature_gamma,
            )
            oracle_matrices["conformal_reliability"] = conformal_matrix
            oracle_scores["conformal_reliability"] = reliability
            oracle_components["conformal_reliability"] = {
                "enabled": True,
                "applied": True,
                "reason": "ok",
                "conformal_uq_alpha": float(np.clip(conformal_uq_alpha, 1e-3, 0.49)),
                "conformal_uq_min_folds": int(min_required),
                "min_folds_seen": int(min_folds_seen),
                "interval_widths": widths.tolist(),
            }
            oracle_pairwise_meta["conformal_reliability"] = dict(meta)

    if use_conformal_efficiency:
        min_required = 3
        fold_counts = [
            int(
                np.asarray(
                    evaluation[name].get("conformal_singleton_rates", np.asarray([], dtype=float)),
                    dtype=float,
                ).size
            )
            for name in candidate_names
        ]
        min_folds_seen = int(min(fold_counts)) if fold_counts else 0
        if min_folds_seen < min_required:
            logger.warning(
                "Conformal efficiency oracle skipped: min folds=%d below required=%d.",
                int(min_folds_seen),
                int(min_required),
            )
            oracle_components["conformal_efficiency"] = {
                "enabled": True,
                "applied": False,
                "reason": "min_folds_gate",
                "conformal_efficiency_method": str(conformal_efficiency_method),
                "min_folds_seen": int(min_folds_seen),
                "min_required_folds": int(min_required),
                "conformal_singleton_rate_mean": {},
                "conformal_singleton_rate_std": {},
            }
        else:
            target_coverage = float(1.0 - np.clip(conformal_uq_alpha, 1e-3, 0.49))
            mean_singleton_scores = []
            std_singleton_scores = []
            mean_coverage_scores = []
            std_coverage_scores = []
            utility_scores = []
            for name in candidate_names:
                singleton_vals = np.asarray(
                    evaluation[name].get("conformal_singleton_rates", np.asarray([], dtype=float)),
                    dtype=float,
                ).ravel()
                singleton_vals = singleton_vals[np.isfinite(singleton_vals)]
                coverage_vals = np.asarray(
                    evaluation[name].get("conformal_coverages", np.asarray([], dtype=float)),
                    dtype=float,
                ).ravel()
                coverage_vals = coverage_vals[np.isfinite(coverage_vals)]
                mean_singleton = float(np.mean(singleton_vals)) if singleton_vals.size else 0.0
                std_singleton = float(np.std(singleton_vals, ddof=1)) if singleton_vals.size > 1 else 0.0
                mean_coverage = float(np.mean(coverage_vals)) if coverage_vals.size else float("nan")
                std_coverage = float(np.std(coverage_vals, ddof=1)) if coverage_vals.size > 1 else 0.0
                mean_singleton_scores.append(mean_singleton)
                std_singleton_scores.append(std_singleton)
                mean_coverage_scores.append(mean_coverage)
                std_coverage_scores.append(std_coverage)
                if np.isfinite(mean_coverage) and mean_coverage >= target_coverage:
                    utility_scores.append(float(1.0 + mean_singleton))
                elif np.isfinite(mean_coverage):
                    utility_scores.append(float(-1.0 - max(0.0, target_coverage - mean_coverage)))
                else:
                    utility_scores.append(-2.0)
            utility_scores_arr = np.asarray(utility_scores, dtype=float)
            conformal_matrix, meta = _mnpo_matrix_from_scalar_scores(
                utility_scores_arr,
                tie_margin=0.01,
                use_qre_smoothing=use_qre_smoothing,
                qre_temperature_gamma=qre_temperature_gamma,
            )
            oracle_matrices["conformal_efficiency"] = conformal_matrix
            oracle_scores["conformal_efficiency"] = utility_scores_arr
            oracle_pairwise_meta["conformal_efficiency"] = dict(meta)
            oracle_components["conformal_efficiency"] = {
                "enabled": True,
                "applied": True,
                "reason": "ok",
                "conformal_efficiency_method": str(conformal_efficiency_method),
                "conformal_efficiency_target_coverage": float(target_coverage),
                "min_folds_seen": int(min_folds_seen),
                "min_required_folds": int(min_required),
                "conformal_singleton_rate_mean": {
                    str(candidate_names[i]): float(mean_singleton_scores[i]) for i in range(len(candidate_names))
                },
                "conformal_singleton_rate_std": {
                    str(candidate_names[i]): float(std_singleton_scores[i]) for i in range(len(candidate_names))
                },
                "conformal_coverage_mean": {
                    str(candidate_names[i]): float(mean_coverage_scores[i]) for i in range(len(candidate_names))
                },
                "conformal_coverage_std": {
                    str(candidate_names[i]): float(std_coverage_scores[i]) for i in range(len(candidate_names))
                },
                "conformal_efficiency_utility": {
                    str(candidate_names[i]): float(utility_scores_arr[i]) for i in range(len(candidate_names))
                },
            }

    # ── T-002: Per-model performance oracle matrices ──────────────────
    if performance_oracle_mode == "multi_model_oracles" and m > 0:
        # Collect model keys present across all candidates
        all_model_keys = set()
        for name in candidate_names:
            by_model = evaluation[name].get('performance_scores_by_model', {})
            all_model_keys.update(by_model.keys())
        # Filter to keys with actual per-fold data (skip _fallback_*, _single_class)
        model_keys = sorted(
            k for k in all_model_keys
            if not k.startswith('_')
        )
        if model_keys:
            for model_key in model_keys:
                model_matrix = np.full((m, m), 0.5, dtype=float)
                model_means = np.zeros(m, dtype=float)
                for idx, name in enumerate(candidate_names):
                    by_model = evaluation[name].get('performance_scores_by_model', {})
                    scores = by_model.get(model_key, np.array([], dtype=float))
                    model_means[idx] = float(np.mean(scores)) if len(scores) > 0 else 0.0
                for i, j in combinations(range(m), 2):
                    scores_i = evaluation[candidate_names[i]].get(
                        'performance_scores_by_model', {}
                    ).get(model_key, np.array([], dtype=float))
                    scores_j = evaluation[candidate_names[j]].get(
                        'performance_scores_by_model', {}
                    ).get(model_key, np.array([], dtype=float))
                    if len(scores_i) > 0 and len(scores_j) > 0:
                        p_ij = fold_preference_fn(
                            scores_i, scores_j, pairwise_delta=pairwise_delta,
                        )
                    else:
                        p_ij = 0.5
                    model_matrix[i, j] = p_ij
                    model_matrix[j, i] = 1.0 - p_ij
                np.fill_diagonal(model_matrix, 0.5)
                oracle_name = f"performance_{model_key}"
                oracle_matrices[oracle_name] = model_matrix
                oracle_scores[oracle_name] = model_means
            logger.debug(
                "T-002: Created %d per-model performance oracles: %s",
                len(model_keys),
                [f"performance_{k}" for k in model_keys],
            )
        else:
            logger.debug(
                "T-002: performance_oracle_mode='multi_model_oracles' but no "
                "per-model scores available; falling back to single oracle."
            )

    # T-DS3: Tail-risk and regret oracle pathways removed.
    # Keep compatibility kwargs in the signature for older callers.
    _ = (use_tail_risk_oracle, tail_risk_alpha, use_regret_oracle)

    if use_stability_oracle:
        stability_scores = np.array([evaluation[name]['stability'] for name in candidate_names], dtype=float)
        stability_matrix, meta = _mnpo_matrix_from_scalar_scores(
            stability_scores,
            tie_margin=0.015,
            use_qre_smoothing=use_qre_smoothing,
            qre_temperature_gamma=qre_temperature_gamma,
        )
        oracle_matrices['stability'] = stability_matrix
        oracle_scores['stability'] = stability_scores
        oracle_pairwise_meta["stability"] = dict(meta)

    if use_complexity_oracle:
        complexity_scores = np.array([evaluation[name]['complexity'] for name in candidate_names], dtype=float)
        complexity_matrix, meta = _mnpo_matrix_from_scalar_scores(
            complexity_scores,
            tie_margin=0.01,
            use_qre_smoothing=use_qre_smoothing,
            qre_temperature_gamma=qre_temperature_gamma,
        )
        oracle_matrices['complexity'] = complexity_matrix
        oracle_scores['complexity'] = complexity_scores
        oracle_pairwise_meta["complexity"] = dict(meta)

    if use_robust_oracle:
        robustness_scores = np.array([evaluation[name]['robustness'] for name in candidate_names], dtype=float)
        robustness_matrix, meta = _mnpo_matrix_from_scalar_scores(
            robustness_scores,
            tie_margin=0.01,
            use_qre_smoothing=use_qre_smoothing,
            qre_temperature_gamma=qre_temperature_gamma,
        )
        oracle_matrices['robustness'] = robustness_matrix
        oracle_scores['robustness'] = robustness_scores
        oracle_pairwise_meta["robustness"] = dict(meta)

    if use_interaction_oracle:
        n_train = 0
        if X_pool is not None:
            try:
                n_train = int(np.asarray(X_pool).shape[0])
            except Exception as exc:
                n_train = 0
        if n_train <= 0 and candidate_names:
            n_train = int(max(0, evaluation[candidate_names[0]].get("n_samples", 0)))

        min_n_gate = int(max(2, interaction_oracle_min_n_train))
        if n_train < min_n_gate:
            logger.warning(
                "Interaction oracle skipped: n_train=%d below minimum %d.",
                int(n_train),
                int(min_n_gate),
            )
            oracle_components["interaction_density"] = {
                "enabled": True,
                "applied": False,
                "reason": "min_n_gate",
                "n_train": int(n_train),
                "interaction_oracle_min_n_train": int(min_n_gate),
                "interaction_oracle_pool_size_cap": int(max(4, interaction_oracle_pool_size_cap)),
                "interaction_oracle_pair_cap": int(max(1, interaction_oracle_pair_cap)),
            }
        elif X_pool is None:
            oracle_components["interaction_density"] = {
                "enabled": True,
                "applied": False,
                "reason": "missing_X_pool",
                "n_train": int(n_train),
                "interaction_oracle_min_n_train": int(min_n_gate),
                "interaction_oracle_pool_size_cap": int(max(4, interaction_oracle_pool_size_cap)),
                "interaction_oracle_pair_cap": int(max(1, interaction_oracle_pair_cap)),
            }
        else:
            scores = np.zeros(m, dtype=float)
            for idx, name in enumerate(candidate_names):
                selected = np.asarray(
                    evaluation.get(name, {}).get("selected_indices", np.asarray([], dtype=int)),
                    dtype=int,
                ).ravel()
                scores[idx] = _compute_interaction_density_score(
                    np.asarray(X_pool, dtype=float),
                    np.asarray(evaluation.get(name, {}).get("target_signal", np.asarray([]))),
                    selected,
                    pool_size_cap=int(max(4, interaction_oracle_pool_size_cap)),
                    pair_cap=int(max(1, interaction_oracle_pair_cap)),
                )
            interaction_matrix, meta = _mnpo_matrix_from_scalar_scores(
                np.asarray(scores, dtype=float),
                tie_margin=0.01,
                use_qre_smoothing=use_qre_smoothing,
                qre_temperature_gamma=qre_temperature_gamma,
            )
            oracle_matrices["interaction_density"] = interaction_matrix
            oracle_scores["interaction_density"] = np.asarray(scores, dtype=float)
            oracle_pairwise_meta["interaction_density"] = dict(meta)
            oracle_components["interaction_density"] = {
                "enabled": True,
                "applied": True,
                "reason": "ok",
                "n_train": int(n_train),
                "interaction_oracle_min_n_train": int(min_n_gate),
                "interaction_oracle_pool_size_cap": int(max(4, interaction_oracle_pool_size_cap)),
                "interaction_oracle_pair_cap": int(max(1, interaction_oracle_pair_cap)),
            }

    if use_ubayfs and m > 1:
        n_samples_eff = int(max(0, evaluation[candidate_names[0]].get("n_samples", 0)))
        n_features_eff = int(max(1, evaluation[candidate_names[0]].get("n_features", 1)))
        if n_samples_eff < int(max(1, ubayfs_min_n)):
            oracle_components["ubayfs"] = {
                "enabled": True,
                "applied": False,
                "reason": "min_n_gate",
                "n_samples": int(n_samples_eff),
                "ubayfs_min_n": int(max(1, ubayfs_min_n)),
            }
        else:
            if float(ubayfs_prior_weight) > 0.5:
                logger.warning(
                    "UBayFS prior_weight=%.3f exceeds 0.5; posterior may be prior-dominated.",
                    float(ubayfs_prior_weight),
                )

            eps = 1e-12
            b_runs = int(max(1, ubayfs_n_bootstrap))
            prior_w = float(np.clip(ubayfs_prior_weight, 0.0, 1.0))
            rng = np.random.RandomState(int(random_state) % (2**32))

            posterior_list = []
            for name in candidate_names:
                selected = np.asarray(evaluation[name].get("selected_indices", np.asarray([], dtype=int)), dtype=int).ravel()
                selected = np.asarray(
                    sorted(set(int(i) for i in selected if 0 <= int(i) < n_features_eff)),
                    dtype=int,
                )

                counts = np.zeros(n_features_eff, dtype=float)
                if selected.size > 0:
                    draw_size = int(max(1, selected.size))
                    for _ in range(b_runs):
                        draw = rng.choice(selected, size=draw_size, replace=True)
                        counts[np.unique(draw)] += 1.0
                    empirical = counts / max(eps, float(np.sum(counts)))
                else:
                    empirical = np.full(n_features_eff, 1.0 / float(n_features_eff), dtype=float)

                prior = np.full(n_features_eff, 1.0 / float(n_features_eff), dtype=float)
                post = (1.0 - prior_w) * empirical + prior_w * prior
                post = np.asarray(np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
                if float(np.sum(post)) <= eps:
                    post = np.full(n_features_eff, 1.0 / float(n_features_eff), dtype=float)
                else:
                    post = post / float(np.sum(post))
                posterior_list.append(post)

            posterior = np.vstack(posterior_list)

            def _kl(p: np.ndarray, q: np.ndarray) -> float:
                p_safe = np.clip(np.asarray(p, dtype=float), eps, 1.0)
                q_safe = np.clip(np.asarray(q, dtype=float), eps, 1.0)
                return float(np.sum(p_safe * np.log(p_safe / q_safe)))

            ub_matrix = np.full((m, m), 0.5, dtype=float)
            ub_scores = np.zeros(m, dtype=float)
            scale = 0.50
            for i, j in combinations(range(m), 2):
                d_ij = _kl(posterior[i], posterior[j])
                d_ji = _kl(posterior[j], posterior[i])
                x = float(np.clip((d_ji - d_ij) / max(eps, scale), -20.0, 20.0))
                p_ij = float(1.0 / (1.0 + np.exp(-x)))
                ub_matrix[i, j] = p_ij
                ub_matrix[j, i] = 1.0 - p_ij
            np.fill_diagonal(ub_matrix, 0.5)

            for i in range(m):
                vals = [_kl(posterior[i], posterior[j]) for j in range(m) if j != i]
                ub_scores[i] = -float(np.mean(vals)) if vals else 0.0

            oracle_matrices["ubayfs"] = ub_matrix
            oracle_scores["ubayfs"] = np.asarray(ub_scores, dtype=float)
            oracle_components["ubayfs"] = {
                "enabled": True,
                "applied": True,
                "reason": "ok",
                "n_samples": int(n_samples_eff),
                "n_features": int(n_features_eff),
                "ubayfs_n_bootstrap": int(b_runs),
                "ubayfs_prior_weight": float(prior_w),
            }
            oracle_pairwise_meta["ubayfs"] = {
                "n_bootstrap": int(b_runs),
                "prior_weight": float(prior_w),
                "min_n_gate": int(max(1, ubayfs_min_n)),
            }

    if use_diversity_oracle and m > 1:
        effective_diversity_mode = str(diversity_oracle_mode or "legacy_jaccard").strip().lower()
        if effective_diversity_mode not in {"legacy_jaccard", "mi_redundancy", "pid_mi", "complementarity"}:
            effective_diversity_mode = "legacy_jaccard"
        if effective_diversity_mode == "complementarity" and int(m) <= 4:
            logger.warning(
                "Complementarity diversity mode requested with m=%d<=4; falling back to legacy_jaccard.",
                int(m),
            )
            effective_diversity_mode = "legacy_jaccard"

        diversity_scores = np.zeros(m, dtype=float)
        performance_norm = normalize_vector_01(
            np.array([evaluation[name]['performance_mean'] for name in candidate_names], dtype=float)
        )
        relevance_scores = performance_norm.copy()
        redundancy_means = np.zeros(m, dtype=float)
        complementarity_means = np.zeros(m, dtype=float)

        for i in range(m):
            name_i = candidate_names[i]
            signal_i = evaluation[name_i]['prediction_signal']
            target_signal = evaluation[name_i].get('target_signal', np.asarray([], dtype=int))
            set_i = set(map(int, evaluation[name_i]['selected_indices'].tolist()))
            score_i = evaluation[name_i].get('score_vector', np.array([], dtype=float))
            pairwise_redundancy = []
            pairwise_complementarity = []
            pairwise_relevance = []

            for j in range(m):
                if i == j:
                    continue
                name_j = candidate_names[j]
                signal_j = evaluation[name_j]['prediction_signal']
                set_j = set(map(int, evaluation[name_j]['selected_indices'].tolist()))
                score_j = evaluation[name_j].get('score_vector', np.array([], dtype=float))

                n = min(signal_i.size, signal_j.size)
                if n > 2:
                    corr = np.corrcoef(signal_i[:n], signal_j[:n])[0, 1]
                    corr = float(corr) if np.isfinite(corr) else 0.0
                else:
                    corr = 0.0
                pred_corr = abs(corr)

                union = len(set_i.union(set_j))
                overlap = (len(set_i.intersection(set_j)) / union) if union > 0 else 1.0

                if effective_diversity_mode == 'pid_mi':
                    red, unique_i, _, syn, hy = pid_imin(
                        signal_i,
                        signal_j,
                        target_signal,
                        n_bins=8,
                    )
                    if hy <= 1e-12:
                        redundancy = 0.5
                        complementarity = 0.5
                    else:
                        redundancy = float(red / hy)
                        complementarity = float((unique_i + syn) / hy)
                elif effective_diversity_mode == 'mi_redundancy':
                    mi_redundancy = normalized_mutual_info(score_i, score_j, n_bins=16)
                    redundancy = 0.50 * mi_redundancy + 0.30 * overlap + 0.20 * pred_corr
                    disjointness = 1.0 - overlap
                    prediction_disagreement = 1.0 - pred_corr
                    score_disagreement = 1.0 - mi_redundancy
                    complementarity = 0.50 * disjointness + 0.30 * prediction_disagreement + 0.20 * score_disagreement
                elif effective_diversity_mode == "complementarity":
                    redundancy = pred_corr
                    complementarity = 1.0 - pred_corr
                else:
                    redundancy = 0.40 * overlap + 0.60 * pred_corr
                    complementarity = 1.0 - redundancy
                pairwise_redundancy.append(float(np.clip(redundancy, 0.0, 1.0)))
                pairwise_complementarity.append(float(np.clip(complementarity, 0.0, 1.0)))
                pairwise_relevance.append(float(max(1e-6, performance_norm[j])))

            if effective_diversity_mode in {'mi_redundancy', 'pid_mi', 'complementarity'}:
                if pairwise_redundancy:
                    rel_weights = np.asarray(pairwise_relevance, dtype=float)
                    rel_weights = rel_weights / max(1e-12, np.sum(rel_weights))
                    mean_redundancy = float(np.average(pairwise_redundancy, weights=rel_weights))
                    mean_complementarity = float(np.average(pairwise_complementarity, weights=rel_weights))
                else:
                    mean_redundancy = 0.5
                    mean_complementarity = 0.5
                redundancy_means[i] = mean_redundancy
                complementarity_means[i] = mean_complementarity
                diversity_scores[i] = float(
                    np.clip(
                        relevance_scores[i]
                        - diversity_redundancy_weight * mean_redundancy
                        + diversity_complementarity_weight * mean_complementarity,
                        -1.0,
                        1.5,
                    )
                )
            else:
                diversity_scores[i] = float(
                    np.mean([1.0 - v for v in pairwise_redundancy]) if pairwise_redundancy else 0.5
                )

        if effective_diversity_mode in {'mi_redundancy', 'pid_mi', 'complementarity'}:
            diversity_scores = normalize_vector_01(diversity_scores)
            oracle_components['diversity'] = {
                'relevance': relevance_scores,
                'redundancy': redundancy_means,
                'complementarity': complementarity_means,
                'mode': str(effective_diversity_mode),
            }

        diversity_matrix, meta = _mnpo_matrix_from_scalar_scores(
            diversity_scores,
            tie_margin=0.01,
            use_qre_smoothing=use_qre_smoothing,
            qre_temperature_gamma=qre_temperature_gamma,
        )
        oracle_matrices['diversity'] = diversity_matrix
        oracle_scores['diversity'] = diversity_scores
        oracle_pairwise_meta["diversity"] = dict(meta)

    complexity_meta: Dict[str, Any] = {
        "enabled": bool(complexity_conditioning),
        "applied": False,
        "reason": "disabled" if not complexity_conditioning else "uninitialized",
    }
    if complexity_conditioning:
        if X_pool is None or not candidate_names:
            complexity_meta["reason"] = "missing_x_pool"
        else:
            try:
                try:
                    from tabnetics.datasets.meta_features import extract_meta_features
                except Exception:
                    from tabnetics.datasets.meta_features import extract_meta_features  # type: ignore
                y_signal = np.asarray(
                    evaluation.get(candidate_names[0], {}).get("target_signal", np.asarray([], dtype=int))
                ).ravel()
                x_arr = np.asarray(X_pool, dtype=float)
                if x_arr.ndim != 2 or x_arr.shape[0] != y_signal.size or y_signal.size < 4:
                    complexity_meta["reason"] = "invalid_inputs"
                else:
                    meta_features = {
                        str(k): float(v)
                        for k, v in extract_meta_features(x_arr, y_signal, expanded=True).items()
                    }
                    complexity_meta.update(
                        {
                            "applied": True,
                            "reason": "ok",
                            "meta_features": dict(meta_features),
                            "complexity_score": float(_normalized_complexity_score(meta_features)),
                            "dataset_tier": str(_heuristic_dataset_tier(meta_features)),
                        }
                    )
            except Exception as exc:
                complexity_meta["reason"] = str(type(exc).__name__)
    oracle_components["complexity_conditioning"] = dict(complexity_meta)
    oracle_pairwise_meta["complexity_conditioning"] = dict(complexity_meta)

    # Shared oracle-stability diagnostics contract (PREREQ-B).
    oracle_components["oracle_config"] = {
        "weighting_mode": str(weighting_mode),
        "fold_preference_mode": str(fold_preference_mode),
        "shapley_n_coalitions_max": int(max(2, shapley_n_coalitions_max)),
        "use_interaction_oracle": bool(use_interaction_oracle),
        "interaction_oracle_min_n_train": int(max(2, interaction_oracle_min_n_train)),
        "interaction_oracle_pool_size_cap": int(max(4, interaction_oracle_pool_size_cap)),
        "interaction_oracle_pair_cap": int(max(1, interaction_oracle_pair_cap)),
        "use_conformal_efficiency": bool(use_conformal_efficiency),
        "conformal_efficiency_method": str(conformal_efficiency_method),
        "complexity_conditioning": bool(complexity_conditioning),
    }
    oracle_stability = compute_oracle_stability_diagnostics(oracle_scores)
    oracle_components["oracle_stability"] = dict(oracle_stability)
    oracle_pairwise_meta["oracle_stability"] = dict(oracle_stability)

    return oracle_matrices, oracle_scores, oracle_components, oracle_pairwise_meta


def compute_oracle_stability_diagnostics(oracle_scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute deterministic oracle agreement diagnostics from score vectors.

    Returns rank-based pairwise agreement plus an aggregate summary to support
    promotion gating and auditability.
    """

    def _rankdata(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size == 0:
            return np.asarray([], dtype=float)
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty(arr.size, dtype=float)
        ranks[order] = np.arange(arr.size, dtype=float)
        return ranks

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        n = int(min(a.size, b.size))
        if n < 2:
            return 0.0
        x = np.asarray(a[:n], dtype=float)
        y = np.asarray(b[:n], dtype=float)
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denom <= 1e-12:
            return 0.0
        return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))

    keys = sorted(str(k) for k in (oracle_scores or {}).keys())
    if len(keys) < 2:
        return {
            "n_oracles": int(len(keys)),
            "pair_count": 0,
            "mean_rank_correlation": 1.0 if len(keys) == 1 else 0.0,
            "min_rank_correlation": 1.0 if len(keys) == 1 else 0.0,
            "pairwise_rank_correlation": {},
        }

    pairwise: Dict[str, float] = {}
    values = []
    for key_i, key_j in combinations(keys, 2):
        ri = _rankdata(np.asarray(oracle_scores.get(key_i, np.asarray([])), dtype=float))
        rj = _rankdata(np.asarray(oracle_scores.get(key_j, np.asarray([])), dtype=float))
        corr = _pearson(ri, rj)
        pairwise[f"{key_i}__{key_j}"] = float(corr)
        values.append(float(corr))

    arr = np.asarray(values, dtype=float) if values else np.asarray([], dtype=float)
    return {
        "n_oracles": int(len(keys)),
        "pair_count": int(arr.size),
        "mean_rank_correlation": float(np.mean(arr)) if arr.size else 0.0,
        "min_rank_correlation": float(np.min(arr)) if arr.size else 0.0,
        "pairwise_rank_correlation": pairwise,
    }


def compute_rashomon_importance_bounds(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    *,
    random_state: int,
    max_models: int = 12,
    score_tolerance: float = 0.01,
    cv_splits: int = 3,
) -> Dict[str, Any]:
    """Compute post-selection Rashomon-style importance bounds.

    A model enters the Rashomon set when its CV balanced accuracy is within
    ``score_tolerance`` of the best candidate model.
    """
    idx = np.asarray(feature_indices, dtype=int).ravel()
    idx = np.array(sorted(set(int(i) for i in idx if int(i) >= 0)), dtype=int)
    x_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    n, p = x_arr.shape

    out: Dict[str, Any] = {
        "rashomon_enabled": True,
        "rashomon_computed": False,
        "rashomon_reason": "uninitialized",
        "rashomon_n_features": int(idx.size),
        "rashomon_n_models_total": 0,
        "rashomon_n_models_kept": 0,
        "rashomon_best_score": None,
        "rashomon_score_tolerance": float(max(0.0, score_tolerance)),
        "importance_bounds": {},
    }

    if idx.size == 0 or n <= 2 or p <= 0:
        out["rashomon_reason"] = "invalid_inputs"
        return out
    if np.max(idx) >= p:
        out["rashomon_reason"] = "index_out_of_bounds"
        return out

    x_sub = x_arr[:, idx]
    classes = np.unique(y_arr)
    if classes.size < 2:
        out["rashomon_reason"] = "single_class"
        return out

    max_models_i = int(max(2, max_models))
    cv_splits_i = int(max(2, min(cv_splits, n)))
    min_class = int(np.min(np.bincount(np.searchsorted(classes, y_arr))))
    if classes.size >= 2 and min_class >= cv_splits_i:
        splitter = StratifiedKFold(n_splits=cv_splits_i, shuffle=True, random_state=int(random_state))
        split_iter = splitter.split(x_sub, y_arr)
    else:
        splitter = KFold(n_splits=cv_splits_i, shuffle=True, random_state=int(random_state))
        split_iter = splitter.split(x_sub)

    splits = list(split_iter)
    if not splits:
        out["rashomon_reason"] = "no_cv_splits"
        return out

    c_grid = np.logspace(-2, 1.2, num=max_models_i)
    model_records = []
    # Use saga solver: supports L1 + multiclass in all sklearn versions
    # (liblinear dropped multiclass support in sklearn 1.8).
    import warnings as _w
    for m_idx, c_val in enumerate(c_grid):
        fold_scores = []
        for train_idx, val_idx in splits:
            x_tr = x_sub[train_idx]
            y_tr = y_arr[train_idx]
            x_va = x_sub[val_idx]
            y_va = y_arr[val_idx]
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore", (FutureWarning, UserWarning))
                    clf = make_logistic_regression(
                        penalty="l1",
                        solver="saga",
                        C=float(c_val),
                        max_iter=5000,
                        random_state=int(random_state + m_idx),
                    )
                    clf.fit(x_tr, y_tr)
                    pred = clf.predict(x_va)
                fold_scores.append(float(balanced_accuracy_score(y_va, pred)))
            except Exception as exc:
                continue
        if not fold_scores:
            continue
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore", (FutureWarning, UserWarning))
                clf_full = make_logistic_regression(
                    penalty="l1",
                    solver="saga",
                    C=float(c_val),
                    max_iter=5000,
                    random_state=int(random_state + 10_000 + m_idx),
                )
            clf_full.fit(x_sub, y_arr)
            coef = np.asarray(clf_full.coef_, dtype=float)
            importance = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef).ravel()
        except Exception as exc:
            continue
        model_records.append(
            {
                "c": float(c_val),
                "score": float(np.mean(np.asarray(fold_scores, dtype=float))),
                "importance": np.asarray(importance, dtype=float).ravel(),
            }
        )

    out["rashomon_n_models_total"] = int(len(model_records))
    if not model_records:
        out["rashomon_reason"] = "no_successful_models"
        return out

    scores = np.asarray([row["score"] for row in model_records], dtype=float)
    best = float(np.max(scores))
    tol = float(max(0.0, score_tolerance))
    keep = [row for row in model_records if float(row["score"]) >= best - tol]
    out["rashomon_best_score"] = float(best)
    out["rashomon_n_models_kept"] = int(len(keep))
    if not keep:
        out["rashomon_reason"] = "empty_rashomon_set"
        return out

    imp = np.vstack([np.asarray(row["importance"], dtype=float).ravel() for row in keep])
    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
    imp_min = np.min(imp, axis=0)
    imp_max = np.max(imp, axis=0)
    imp_mean = np.mean(imp, axis=0)

    out["importance_bounds"] = {
        int(idx[j]): {
            "min": float(imp_min[j]),
            "max": float(imp_max[j]),
            "mean": float(imp_mean[j]),
        }
        for j in range(idx.size)
    }
    out["rashomon_computed"] = True
    out["rashomon_reason"] = "ok"
    return out


# ---------------------------------------------------------------------------
# 4. fit_tritrust_weights
# ---------------------------------------------------------------------------
def fit_tritrust_weights(oracle_matrices, reference_oracle=None):
    """TriTrust-style trust/ignore/flip calibration from oracle agreement.

    Parameters
    ----------
    oracle_matrices : dict[str, np.ndarray]
        Oracle name → m×m pairwise preference matrix.
    reference_oracle : str or None
        Which oracle to calibrate against. ``None`` (default) auto-selects:
        ``performance_lr_l2`` if present (T-002 multi-model mode), else
        ``performance``.
    """
    if reference_oracle is None:
        if "performance_lr_l2" in oracle_matrices:
            reference_oracle = "performance_lr_l2"
        else:
            reference_oracle = "performance"
    return dict(
        _mnpo_fit_tritrust_weights(
            oracle_matrices,
            reference=reference_oracle,
            allow_negative=True,
            no_flip_oracles={"diversity"},
            ref_delta_threshold=0.05,
            oracle_delta_threshold=0.03,
            reliability_threshold=0.10,
        )
    )


# ---------------------------------------------------------------------------
# 5. aggregate_payoff_matrix
# ---------------------------------------------------------------------------
def aggregate_payoff_matrix(oracle_matrices, oracle_weights):
    """Aggregate oracle preferences into an anti-symmetric payoff matrix."""
    return _mnpo_aggregate_payoff_matrix(oracle_matrices, oracle_weights)


# ---------------------------------------------------------------------------
# 6. normalize_vector_01
# ---------------------------------------------------------------------------
def normalize_vector_01(values):
    """Min-max normalize to [0,1] with safe fallback."""
    return _mnpo_normalize_vector_01(values)


# ---------------------------------------------------------------------------
# 7. normalized_mutual_info
# ---------------------------------------------------------------------------
def normalized_mutual_info(values_a, values_b, n_bins=16):
    """Estimate normalized mutual information from two score vectors."""
    a = np.asarray(values_a, dtype=float).ravel()
    b = np.asarray(values_b, dtype=float).ravel()
    n = int(min(a.size, b.size))
    if n < 8:
        return 0.0
    a = np.nan_to_num(a[:n], nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b[:n], nan=0.0, posinf=0.0, neginf=0.0)
    if (np.std(a) < 1e-12) or (np.std(b) < 1e-12):
        return 0.0

    bins = int(max(4, min(n_bins, n // 3)))
    try:
        hist2d, _, _ = np.histogram2d(a, b, bins=bins)
    except Exception as exc:
        return 0.0
    if np.sum(hist2d) <= 0:
        return 0.0

    pxy = hist2d / np.sum(hist2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    mask = pxy > 0
    if not np.any(mask):
        return 0.0

    pxpy = np.outer(px, py)
    mi = float(np.sum(pxy[mask] * np.log(pxy[mask] / np.clip(pxpy[mask], 1e-12, None))))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    denom = max(1e-12, min(hx, hy))
    return float(np.clip(mi / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 8. discretize_signal
# ---------------------------------------------------------------------------
def discretize_signal(values, n_bins=8):
    """Discretize a 1D signal into integer bins for MI/PID estimates."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return np.asarray([], dtype=int)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Treat near-integer small-cardinality signals as categorical codes.
    rounded = np.round(arr)
    if np.all(np.isclose(arr, rounded)) and np.unique(rounded).size <= max(2, int(n_bins)):
        _, inv = np.unique(rounded.astype(int), return_inverse=True)
        return np.asarray(inv, dtype=int)

    bins = int(max(2, min(int(n_bins), max(2, int(arr.size // 12)))))
    try:
        edges = np.quantile(arr, np.linspace(0.0, 1.0, bins + 1))
    except Exception as exc:
        return np.zeros(arr.size, dtype=int)
    edges = np.unique(np.asarray(edges, dtype=float))
    if edges.size <= 2:
        return np.zeros(arr.size, dtype=int)
    return np.asarray(np.digitize(arr, edges[1:-1], right=False), dtype=int)


# ---------------------------------------------------------------------------
# 9. entropy_discrete
# ---------------------------------------------------------------------------
def entropy_discrete(values):
    arr = np.asarray(values, dtype=int).ravel()
    if arr.size == 0:
        return 0.0
    _, inv = np.unique(arr, return_inverse=True)
    counts = np.bincount(inv)
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts.astype(float) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# 10. mutual_information_discrete
# ---------------------------------------------------------------------------
def mutual_information_discrete(x, y):
    x_arr = np.asarray(x, dtype=int).ravel()
    y_arr = np.asarray(y, dtype=int).ravel()
    n = int(min(x_arr.size, y_arr.size))
    if n < 2:
        return 0.0
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    _, x_inv = np.unique(x_arr, return_inverse=True)
    _, y_inv = np.unique(y_arr, return_inverse=True)
    nx = int(np.max(x_inv)) + 1 if x_inv.size else 0
    ny = int(np.max(y_inv)) + 1 if y_inv.size else 0
    if nx <= 1 or ny <= 1:
        return 0.0

    counts = np.zeros((nx, ny), dtype=float)
    np.add.at(counts, (x_inv, y_inv), 1.0)
    pxy = counts / float(n)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    denom = px * py
    mask = pxy > 0
    if not np.any(mask):
        return 0.0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / np.clip(denom[mask], 1e-12, None))))


# ---------------------------------------------------------------------------
# 11. pid_imin
# ---------------------------------------------------------------------------
def pid_imin(signal_a, signal_b, target, n_bins=8):
    """Compute a lightweight I_min PID (redundancy, unique_a, unique_b, synergy, H(target))."""
    s1 = discretize_signal(signal_a, n_bins=n_bins)
    s2 = discretize_signal(signal_b, n_bins=n_bins)
    y = np.asarray(target).ravel()
    if y.size == 0 or s1.size == 0 or s2.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    y_vals, y_inv = np.unique(y, return_inverse=True)
    y_inv = np.asarray(y_inv, dtype=int)
    n = int(min(s1.size, s2.size, y_inv.size))
    if n < 8 or y_vals.size < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    s1 = s1[:n]
    s2 = s2[:n]
    y_inv = y_inv[:n]

    # Mutual informations in nats.
    mi1 = mutual_information_discrete(s1, y_inv)
    mi2 = mutual_information_discrete(s2, y_inv)
    joint_code = s1 * (int(np.max(s2)) + 1) + s2
    mij = mutual_information_discrete(joint_code, y_inv)
    hy = entropy_discrete(y_inv)
    if hy <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0, float(hy)

    # Specific information I(S; y) per outcome for I_min redundancy.
    def _specific_info(s_codes):
        _, s_inv = np.unique(s_codes, return_inverse=True)
        ns = int(np.max(s_inv)) + 1 if s_inv.size else 0
        ny = int(np.max(y_inv)) + 1 if y_inv.size else 0
        counts = np.zeros((ns, ny), dtype=float)
        np.add.at(counts, (s_inv, y_inv), 1.0)
        count_y = np.sum(counts, axis=0)
        count_s = np.sum(counts, axis=1)
        total = float(np.sum(count_y))
        out = np.zeros(ny, dtype=float)
        if total <= 0:
            return out
        p_y = count_y / total
        for k in range(ny):
            if count_y[k] <= 0:
                continue
            p_s_given_y = counts[:, k] / float(count_y[k])
            # p(y|s) for fixed y=k across s.
            p_y_given_s = np.zeros(ns, dtype=float)
            mask_s = count_s > 0
            p_y_given_s[mask_s] = counts[mask_s, k] / count_s[mask_s]
            mask = p_s_given_y > 0
            if not np.any(mask) or p_y[k] <= 0:
                continue
            out[k] = float(np.sum(p_s_given_y[mask] * np.log(np.clip(p_y_given_s[mask] / p_y[k], 1e-12, None))))
        return out

    spec1 = _specific_info(s1)
    spec2 = _specific_info(s2)

    # Redundancy: sum_y p(y) * min(I(S1;y), I(S2;y))
    _, counts_y = np.unique(y_inv, return_counts=True)
    p_y = counts_y.astype(float) / float(np.sum(counts_y))
    red = float(np.sum(p_y * np.minimum(spec1, spec2)))
    red = float(np.clip(red, 0.0, max(0.0, min(mi1, mi2))))
    unique1 = float(max(0.0, mi1 - red))
    unique2 = float(max(0.0, mi2 - red))
    synergy = float(max(0.0, mij - mi1 - mi2 + red))
    return float(red), float(unique1), float(unique2), float(synergy), float(hy)


# ---------------------------------------------------------------------------
# 12. mirror_descent_mnpo
# ---------------------------------------------------------------------------
def mirror_descent_mnpo(payoff, reference_prior,
                        mirror_descent_steps, mirror_descent_eta,
                        mirror_descent_lambda):
    """Reference-regularized mirror descent on the selector simplex."""
    return _mnpo_mirror_descent_reference_regularized(
        np.asarray(payoff, dtype=float),
        np.asarray(reference_prior, dtype=float),
        steps=int(max(1, int(mirror_descent_steps))),
        eta=float(mirror_descent_eta),
        lambda_=float(mirror_descent_lambda),
        tol_kl=1e-7,
        return_history=True,
    )

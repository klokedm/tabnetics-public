"""MNPO portfolio orchestration and candidate evaluation.

Standalone functions extracted from FeatureSelector (base.py).
"""

import logging
import numpy as np
from typing import Dict, Any, List, Mapping, Tuple, Sequence

from tabnetics.datasets.tier_classifier import adjust_oracle_weights_for_complexity
from tabnetics.feature_selection.diakrino_trust import (
    DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT,
    DIAKRINO_SELECTOR_PRIOR_CALIBRATION_NONE,
    DIAKRINO_SELECTOR_PRIOR_CURRENT_ANCHOR,
    DIAKRINO_SELECTOR_PRIOR_CURRENT_MAX_BLEND,
    DIAKRINO_SELECTOR_PRIOR_CURRENT_RAW_WEIGHT,
    selector_prior_calibration_from_trust_record,
)

# -- MNPO core imports (used directly by mnpo_select_features) --
try:
        from ...mnpo_core import (
            apply_oracle_redundancy_penalty as _mnpo_apply_oracle_redundancy_penalty,
            fit_shapley_weights as _mnpo_fit_shapley_weights,
            james_stein_shrinkage as _mnpo_james_stein_shrinkage,
            mirror_descent_reference_regularized as _mnpo_mirror_descent_reference_regularized,
            shrink_payoff_matrix as _mnpo_shrink_payoff_matrix,
            tremble_oracle_matrices as _mnpo_tremble_oracle_matrices,
        )
except Exception as exc:
    try:
        from tabnetics.core.mnpo import (
            apply_oracle_redundancy_penalty as _mnpo_apply_oracle_redundancy_penalty,
            fit_shapley_weights as _mnpo_fit_shapley_weights,
            james_stein_shrinkage as _mnpo_james_stein_shrinkage,
            mirror_descent_reference_regularized as _mnpo_mirror_descent_reference_regularized,
            shrink_payoff_matrix as _mnpo_shrink_payoff_matrix,
            tremble_oracle_matrices as _mnpo_tremble_oracle_matrices,
        )
    except Exception as exc:
        from tabnetics.core.mnpo import (  # type: ignore[import-untyped]
            apply_oracle_redundancy_penalty as _mnpo_apply_oracle_redundancy_penalty,
            fit_shapley_weights as _mnpo_fit_shapley_weights,
            james_stein_shrinkage as _mnpo_james_stein_shrinkage,
            mirror_descent_reference_regularized as _mnpo_mirror_descent_reference_regularized,
            shrink_payoff_matrix as _mnpo_shrink_payoff_matrix,
            tremble_oracle_matrices as _mnpo_tremble_oracle_matrices,
        )

# -- Registry (for experimental keys) --
try:
    from ..registry import METHOD_REGISTRY, get_experimental_keys
except Exception as exc:
    try:
        from tabnetics.feature_selection.registry import METHOD_REGISTRY, get_experimental_keys
    except Exception as exc:
        METHOD_REGISTRY = {}
        def get_experimental_keys():  # type: ignore[misc]
            return set()

# -- Sibling modules --
from .oracles import (
    apply_bayesian_shrinkage_to_weights,
    compute_shapley_bayesian_shrinkage,
    estimate_oracle_preferences,
    fit_tritrust_weights,
    aggregate_payoff_matrix,
    normalize_vector_01,
    mirror_descent_mnpo,
)
from .consensus import (
    apply_wrapper_refinement,
    build_rank_aggregation_candidate,
)

# Portfolio diversity thresholds (T-R-192).
PORTFOLIO_REDUNDANCY_OVERLAP_THRESHOLD = 0.80
PORTFOLIO_REDUNDANCY_CORR_THRESHOLD = 0.90
# Val-9 enables 11 concurrent oracles in the full MNPO stack; keep one-slot headroom.
MNPO_ORACLE_COUNT_CAP = 12

logger = logging.getLogger(__name__)

DIAKRINO_SELECTOR_POOL_TO_CORE_METHODS: Dict[str, Tuple[str, ...]] = {
    "mnpo_broad_stable": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "ova_ensemble",
        "class_pareto_front",
    ),
    "strict_plus_mrmr": (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
    ),
    "boruta": ("boruta",),
    "copula_knockoff": ("copula_knockoff",),
    "stability_lasso": ("stability_lasso",),
}

DIAKRINO_SELECTOR_PRIOR_CALIBRATION_MODES = (
    DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT,
    DIAKRINO_SELECTOR_PRIOR_CALIBRATION_NONE,
)
# Evidence anchor from the 2026-06-28 GOAL_IMPROVE pass:
# #160 global DIAKRINO weighting was flat/negative, #167 regime gating did not promote,
# and #164 killed structural consumers for this checkpoint.  Therefore future
# emitted selector-pool vectors are weak evidence: preserve a strict baseline bias,
# keep specialist selector floors, and let MNPO/TriTrust dominate the final game.

def _normalise_selector_pool_weights(weights: Dict[str, float] | None) -> Dict[str, float]:
    if not weights:
        return {}
    out: Dict[str, float] = {}
    for name in DIAKRINO_SELECTOR_POOL_TO_CORE_METHODS:
        try:
            val = float(dict(weights).get(name, 0.0))
        except Exception:
            val = 0.0
        if np.isfinite(val) and val > 0.0:
            out[name] = float(val)
    total = float(sum(out.values()))
    if total <= 0.0:
        return {}
    return {name: float(val / total) for name, val in out.items()}


def calibrate_diakrino_selector_prior(
    selector_prior: Dict[str, float] | None,
    *,
    calibration: str = DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT,
    trust_record: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, float] | None, Dict[str, Any]]:
    """Calibrate DIAKRINO selector-pool weights for the current checkpoint evidence.

    The current validation pass did not justify letting DIAKRINO selector routing replace
    MNPO's own reference prior.  The default calibration therefore shrinks any future
    emitted vector toward a strict-plus-MRMR-heavy evidence anchor and caps the later
    MNPO blend.  ``calibration='none'`` keeps the raw normalized vector for ablations.
    """
    mode = str(calibration or DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT).strip().lower()
    if mode in {"raw", "off"}:
        mode = DIAKRINO_SELECTOR_PRIOR_CALIBRATION_NONE
    if mode not in DIAKRINO_SELECTOR_PRIOR_CALIBRATION_MODES:
        mode = DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT
    trust_calibration = selector_prior_calibration_from_trust_record(trust_record)
    trust_mode = str(trust_calibration.get("mode") or "").strip().lower()
    if trust_record is not None and trust_mode != mode:
        raise ValueError(
            f"DIAKRINO selector-prior calibration mode {mode!r} disagrees with sidecar trust record {trust_mode!r}"
        )

    raw = _normalise_selector_pool_weights(selector_prior)
    meta: Dict[str, Any] = {
        "mode": mode,
        "applied": False,
        "reason": "missing_selector_prior",
        "raw_weights": dict(raw),
        "anchor_weights": {},
        "raw_weight": 1.0,
        "max_blend_weight": 1.0,
    }
    if not raw:
        return None, meta
    if mode == DIAKRINO_SELECTOR_PRIOR_CALIBRATION_NONE:
        meta.update({"applied": False, "reason": "raw_no_calibration", "weights": dict(raw)})
        return raw, meta

    anchor = _normalise_selector_pool_weights(dict(trust_calibration.get("anchor_weights") or {}))
    raw_weight = float(np.clip(float(trust_calibration.get("raw_weight", 0.0)), 0.0, 1.0))
    calibrated = {
        name: (1.0 - raw_weight) * float(anchor.get(name, 0.0)) + raw_weight * float(raw.get(name, 0.0))
        for name in DIAKRINO_SELECTOR_POOL_TO_CORE_METHODS
    }
    calibrated = _normalise_selector_pool_weights(calibrated)
    meta.update(
        {
            "applied": True,
            "reason": "current_checkpoint_validation_shrinkage",
            "weights": dict(calibrated),
            "anchor_weights": dict(anchor),
            "raw_weight": raw_weight,
            "max_blend_weight": float(trust_calibration.get("max_blend_weight", 1.0)),
            "evidence": dict(trust_calibration.get("evidence") or {}),
        }
    )
    return calibrated, meta


def build_diakrino_selector_reference_prior(
    base_prior: np.ndarray,
    candidate_names: Sequence[str],
    selector_prior: Dict[str, float] | None,
    *,
    enabled: bool,
    blend_weight: float,
    calibration: str = DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT,
    trust_record: Mapping[str, Any] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Blend DIAKRINO selector-pool weights into the MNPO reference prior.

    Disabled or missing selector data returns ``base_prior`` exactly.
    """
    base = np.asarray(base_prior, dtype=float).ravel()
    meta: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "missing_selector_prior",
        "blend_weight": float(np.clip(blend_weight, 0.0, 1.0)),
        "requested_blend_weight": float(np.clip(blend_weight, 0.0, 1.0)),
        "effective_blend_weight": 0.0,
        "calibration": {
            "mode": str(calibration or DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT),
            "applied": False,
            "reason": "not_run",
        },
        "pool_to_core": {k: list(v) for k, v in DIAKRINO_SELECTOR_POOL_TO_CORE_METHODS.items()},
        "mapped_pool_weights": {},
        "candidate_prior": {},
    }
    if not enabled:
        return base_prior, meta
    names = [str(n) for n in candidate_names]
    if base.size != len(names) or base.size == 0:
        meta["reason"] = "invalid_base_prior"
        return base_prior, meta
    selector_prior, calibration_meta = calibrate_diakrino_selector_prior(
        selector_prior,
        calibration=str(calibration or DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT),
        trust_record=trust_record,
    )
    meta["calibration"] = calibration_meta
    if not selector_prior:
        return base_prior, meta

    diakrino = np.full(len(names), 1.0 / float(max(1, len(names))), dtype=float)
    mapped_total = 0.0
    mapped_weights: Dict[str, float] = {}
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    for pool_name, raw_weight in dict(selector_prior).items():
        try:
            weight = float(raw_weight)
        except Exception:
            continue
        if not np.isfinite(weight) or weight <= 0.0:
            continue
        mapped = [
            method
            for method in DIAKRINO_SELECTOR_POOL_TO_CORE_METHODS.get(str(pool_name), tuple())
            if method in name_to_idx
        ]
        if not mapped:
            continue
        share = weight / float(len(mapped))
        for method in mapped:
            diakrino[name_to_idx[method]] += share
        mapped_total += weight
        mapped_weights[str(pool_name)] = float(weight)

    if mapped_total <= 0.0:
        meta["reason"] = "no_mapped_candidates"
        return base_prior, meta

    diakrino = np.clip(diakrino, 1e-12, None)
    diakrino = diakrino / float(np.sum(diakrino))
    requested_w = float(np.clip(blend_weight, 0.0, 1.0))
    max_w = float(np.clip(calibration_meta.get("max_blend_weight", 1.0), 0.0, 1.0))
    w = min(requested_w, max_w)
    if w <= 0.0:
        meta["reason"] = "zero_blend_weight"
        return base_prior, meta
    out = (1.0 - w) * base + w * diakrino
    out = np.clip(out, 1e-12, None)
    out = out / float(np.sum(out))
    meta.update(
        {
            "applied": True,
            "reason": "applied",
            "blend_weight": float(w),
            "effective_blend_weight": float(w),
            "mapped_pool_weights": mapped_weights,
            "candidate_prior": {name: float(out[idx]) for idx, name in enumerate(names)},
        }
    )
    return out, meta


def enforce_oracle_count_cap(
    oracle_matrices: Dict[str, np.ndarray],
    *,
    cap: int = MNPO_ORACLE_COUNT_CAP,
) -> Dict[str, Any]:
    """Enforce hard cap on active oracle matrices (T-R-208 / R11)."""
    count = int(len(oracle_matrices or {}))
    cap_i = int(max(1, cap))
    meta = {
        "oracle_count": int(count),
        "oracle_cap": int(cap_i),
        "oracle_cap_warning": bool(count >= max(1, cap_i - 1)),
        "oracle_cap_violation": bool(count > cap_i),
    }
    if count > cap_i:
        raise ValueError(
            f"MNPO oracle cap exceeded: active_oracles={count} > cap={cap_i}. "
            "Retire at least one oracle or switch to an aggregation-layer alternative."
        )
    if count >= max(1, cap_i - 1):
        logger.warning(
            "MNPO oracle count approaching cap: active_oracles=%d, cap=%d.",
            int(count),
            int(cap_i),
        )
    return meta


# ---------------------------------------------------------------------------
# 1. runtime_race_candidates
# ---------------------------------------------------------------------------
def runtime_race_candidates(
    X, y, candidates,
    *,
    get_inner_cv_splits_fn,
    fit_and_score_fold_fn,
    runtime_racing_enabled,
    runtime_racing_mode,
    runtime_racing_proxy_splits,
    runtime_racing_keep_fraction,
    runtime_racing_min_candidates,
    runtime_racing_runtime_weight,
    runtime_racing_stages,
    runtime_racing_confidence_bound,
    runtime_racing_delta,
):
    """
    Runtime-aware racing pass before full MNPO evaluation.
    Supports single-stage filtering or successive-halving elimination with
    optional Hoeffding/Bernstein confidence-bound safety checks.
    """
    race_mode = str(runtime_racing_mode).strip().lower()
    if race_mode not in {"single_stage", "successive_halving"}:
        race_mode = "single_stage"
    cb_mode = str(runtime_racing_confidence_bound).strip().lower()
    if cb_mode not in {"none", "hoeffding", "bernstein"}:
        cb_mode = "none"

    meta = {
        "runtime_racing_enabled": bool(runtime_racing_enabled),
        "runtime_racing_applied": False,
        "runtime_racing_mode": str(race_mode),
        "runtime_racing_proxy_splits": int(runtime_racing_proxy_splits),
        "runtime_racing_keep_fraction": float(runtime_racing_keep_fraction),
        "runtime_racing_min_candidates": int(runtime_racing_min_candidates),
        "runtime_racing_runtime_weight": float(runtime_racing_runtime_weight),
        "runtime_racing_stages": int(runtime_racing_stages),
        "runtime_racing_confidence_bound": str(cb_mode),
        "runtime_racing_delta": float(runtime_racing_delta),
        "runtime_racing_initial_candidates": int(len(candidates)),
        "runtime_racing_kept_candidates": int(len(candidates)),
        "runtime_racing_kept_names": list(candidates.keys()),
        "runtime_racing_dropped_names": [],
        "runtime_racing_proxy_scores": {},
        "runtime_racing_stage_history": [],
    }

    if not runtime_racing_enabled:
        return candidates, meta
    if len(candidates) <= int(max(1, runtime_racing_min_candidates)):
        return candidates, meta

    try:
        splits = get_inner_cv_splits_fn(X, y)
    except Exception as exc:
        return candidates, meta
    if not splits:
        return candidates, meta

    n_proxy_total = int(max(1, min(len(splits), runtime_racing_proxy_splits)))
    split_cache = splits[:n_proxy_total]
    min_keep = int(max(1, runtime_racing_min_candidates))
    initial_names = list(candidates.keys())

    evaluated_names = []
    for name in initial_names:
        selected = np.asarray(
            candidates[name].get("selected_indices", np.array([], dtype=int)),
            dtype=int,
        ).ravel()
        selected = np.array(
            sorted(set(int(i) for i in selected if 0 <= int(i) < X.shape[1])),
            dtype=int,
        )
        if selected.size > 0:
            evaluated_names.append(name)
    if len(evaluated_names) <= min_keep:
        return candidates, meta

    def _confidence_radius(scores_arr):
        if cb_mode == "none":
            return 0.0
        vals = np.asarray(scores_arr, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        if n <= 0:
            return float("inf")
        delta_val = float(np.clip(runtime_racing_delta, 1e-6, 0.49))
        if cb_mode == "hoeffding":
            return float(np.sqrt(np.log(1.0 / delta_val) / max(1e-12, 2.0 * n)))
        var = float(np.var(vals, ddof=1)) if n > 1 else 0.0
        log_term = float(np.log(3.0 / delta_val))
        radius = np.sqrt(max(0.0, 2.0 * var * log_term / max(1e-12, n)))
        radius += 3.0 * log_term / max(1.0, float(n))
        return float(radius)

    def _evaluate(name_list, split_count):
        split_count = int(max(1, min(n_proxy_total, split_count)))
        proxy_splits = split_cache[:split_count]
        scored_local = []
        details_local = {}
        for name in name_list:
            candidate = candidates.get(name, {})
            selected = np.asarray(
                candidate.get("selected_indices", np.array([], dtype=int)),
                dtype=int,
            ).ravel()
            selected = np.array(
                sorted(set(int(i) for i in selected if 0 <= int(i) < X.shape[1])),
                dtype=int,
            )
            if selected.size <= 0:
                continue

            fold_scores = []
            for train_idx, val_idx in proxy_splits:
                X_train = X[train_idx][:, selected]
                y_train = y[train_idx]
                X_val = X[val_idx][:, selected]
                y_val = y[val_idx]
                try:
                    result = fit_and_score_fold_fn(X_train, y_train, X_val, y_val)
                    fold_score = result[0] if isinstance(result, tuple) else result
                except Exception as exc:
                    continue
                if np.isfinite(fold_score):
                    fold_scores.append(float(fold_score))

            fold_arr = np.asarray(fold_scores, dtype=float)
            perf_proxy = float(np.mean(fold_arr)) if fold_arr.size > 0 else float("-inf")
            perf_std = float(np.std(fold_arr, ddof=1)) if fold_arr.size > 1 else 0.0
            radius = _confidence_radius(fold_arr)
            lower_bound = float(perf_proxy - radius) if np.isfinite(perf_proxy) else float("-inf")
            upper_bound = float(perf_proxy + radius) if np.isfinite(perf_proxy) else float("-inf")

            complexity = float(1.0 - (selected.size / max(1, X.shape[1])))
            runtime_sec = float(max(0.0, candidate.get("runtime_sec", 0.0)))
            runtime_pen = float(np.log1p(runtime_sec))
            proxy_score = float(
                perf_proxy + 0.05 * complexity - runtime_racing_runtime_weight * runtime_pen
            )

            scored_local.append((proxy_score, name))
            details_local[name] = {
                "proxy_score": float(proxy_score),
                "proxy_performance": float(perf_proxy),
                "proxy_performance_std": float(perf_std),
                "proxy_n_folds": int(fold_arr.size),
                "proxy_complexity": float(complexity),
                "runtime_sec": float(runtime_sec),
                "lower_perf_bound": float(lower_bound),
                "upper_perf_bound": float(upper_bound),
                "bound_radius": float(radius if np.isfinite(radius) else np.inf),
                "split_count": int(split_count),
            }

        scored_local.sort(key=lambda row: row[0], reverse=True)
        return scored_local, details_local

    def _bound_filter(ranked_names, details_map):
        if cb_mode == "none":
            return list(ranked_names), float("nan")
        lowers = []
        for nm in ranked_names:
            lb = float(details_map.get(nm, {}).get("lower_perf_bound", float("-inf")))
            if np.isfinite(lb):
                lowers.append(lb)
        if not lowers:
            return list(ranked_names), float("nan")
        best_lower = float(np.max(np.asarray(lowers, dtype=float)))
        survivors = []
        for nm in ranked_names:
            ub = float(details_map.get(nm, {}).get("upper_perf_bound", float("-inf")))
            if np.isfinite(ub) and ub >= best_lower:
                survivors.append(nm)
        if len(survivors) < min_keep:
            return list(ranked_names), float(best_lower)
        return survivors, float(best_lower)

    proxy_details: Dict[str, Dict[str, Any]] = {}
    stage_history: List[Dict[str, Any]] = []
    current_names = list(evaluated_names)

    if race_mode == "single_stage":
        scored, details = _evaluate(current_names, n_proxy_total)
        if not scored:
            return candidates, meta
        for nm, payload in details.items():
            proxy_details[nm] = payload
        ranked = [name for _, name in scored]
        keep_n = int(max(min_keep, np.ceil(float(runtime_racing_keep_fraction) * len(ranked))))
        keep_n = int(min(len(ranked), keep_n))
        bounded, best_lower = _bound_filter(ranked, details)
        keep_pool = bounded if bounded else ranked
        keep_n = int(min(len(keep_pool), keep_n))
        kept = list(keep_pool[:keep_n])
        dropped = [name for name in current_names if name not in set(kept)]
        stage_history.append(
            {
                "stage": 1,
                "mode": "single_stage",
                "split_count": int(n_proxy_total),
                "candidates_in": int(len(current_names)),
                "candidates_after_bound": int(len(keep_pool)),
                "keep_target": int(keep_n),
                "kept_names": list(kept),
                "dropped_names": list(dropped),
                "best_lower_bound": float(best_lower) if np.isfinite(best_lower) else float("nan"),
            }
        )
        current_names = kept
    else:
        stages = int(max(1, runtime_racing_stages))
        target_final = int(max(min_keep, np.ceil(float(runtime_racing_keep_fraction) * len(current_names))))
        target_final = int(min(len(current_names), target_final))
        for stage_idx in range(stages):
            if len(current_names) <= target_final:
                break
            split_count = int(np.ceil(float(stage_idx + 1) / float(stages) * float(n_proxy_total)))
            scored, details = _evaluate(current_names, split_count)
            if not scored:
                break
            for nm, payload in details.items():
                proxy_details[nm] = payload
            ranked = [name for _, name in scored]
            bounded, best_lower = _bound_filter(ranked, details)
            keep_pool = bounded if bounded else ranked

            progress = float(stage_idx + 1) / float(stages)
            stage_target = int(
                np.ceil(
                    len(evaluated_names)
                    - progress * float(len(evaluated_names) - target_final)
                )
            )
            stage_target = int(max(target_final, min(len(current_names), stage_target)))
            keep_n = int(max(min_keep, min(len(keep_pool), stage_target)))
            kept = list(keep_pool[:keep_n])
            dropped = [name for name in current_names if name not in set(kept)]
            stage_history.append(
                {
                    "stage": int(stage_idx + 1),
                    "mode": "successive_halving",
                    "split_count": int(split_count),
                    "candidates_in": int(len(current_names)),
                    "candidates_after_bound": int(len(keep_pool)),
                    "keep_target": int(keep_n),
                    "kept_names": list(kept),
                    "dropped_names": list(dropped),
                    "best_lower_bound": float(best_lower) if np.isfinite(best_lower) else float("nan"),
                }
            )
            current_names = kept
            if len(current_names) <= target_final:
                break

    keep_set = set(current_names)
    filtered = {name: payload for name, payload in candidates.items() if name in keep_set}
    if not filtered:
        return candidates, meta

    dropped = [name for name in candidates.keys() if name not in keep_set]
    meta.update(
        {
            "runtime_racing_applied": bool(len(filtered) < len(candidates)),
            "runtime_racing_kept_candidates": int(len(filtered)),
            "runtime_racing_kept_names": list(filtered.keys()),
            "runtime_racing_dropped_names": list(dropped),
            "runtime_racing_proxy_scores": dict(proxy_details),
            "runtime_racing_stage_history": list(stage_history),
        }
    )
    return filtered, meta


# ---------------------------------------------------------------------------
# 2. evaluate_candidate_library
# ---------------------------------------------------------------------------
def evaluate_candidate_library(
    X, y, candidates,
    *,
    get_inner_cv_splits_fn,
    fit_and_score_fold_fn,
    augment_training_data_fn,
    use_robust_oracle,
    complexity_use_runtime_penalty,
):
    """Evaluate candidate selectors and return oracle-ready statistics."""
    splits = get_inner_cv_splits_fn(X, y)
    target_signal_parts = []
    for _, val_idx in splits:
        target_signal_parts.append(np.asarray(y[val_idx]).ravel())
    target_signal = (
        np.concatenate(target_signal_parts) if target_signal_parts else np.asarray([], dtype=int)
    )
    evaluation = {}

    for name, candidate in candidates.items():
        selected_indices = candidate['selected_indices']
        if selected_indices.size == 0:
            selected_indices = np.array([int(np.argmax(candidate['score_vector']))], dtype=int)

        perf_scores = []
        robust_scores = []
        pred_signal_parts = []
        per_model_fold_scores: Dict[str, list] = {}  # T-002: per-model fold accumulator
        conformal_singleton_rates: List[float] = []
        conformal_coverages: List[float] = []
        conformal_fold_meta: List[Dict[str, Any]] = []
        fold_exception_count = 0
        robust_fold_exception_count = 0
        model_failure_total = 0.0
        model_failure_by_model: Dict[str, float] = {}

        for train_idx, val_idx in splits:
            X_train = X[train_idx][:, selected_indices]
            y_train = y[train_idx]
            X_val = X[val_idx][:, selected_indices]
            y_val = y[val_idx]

            try:
                result = fit_and_score_fold_fn(X_train, y_train, X_val, y_val)
                fold_per_model: Dict[str, Any] = {}
                fold_meta: Dict[str, Any] = {}
                if isinstance(result, tuple):
                    if len(result) == 4:
                        fold_score, fold_signal, fold_per_model, fold_meta = result
                    elif len(result) == 3:
                        fold_score, fold_signal, third = result
                        third_dict = dict(third or {}) if isinstance(third, dict) else {}
                        if "conformal_singleton_rate" in third_dict or "conformal_efficiency_method" in third_dict:
                            fold_meta = third_dict
                        else:
                            fold_per_model = third_dict
                    else:
                        fold_score, fold_signal = result[0], result[1]
                    fold_per_model = dict(fold_per_model or {})
                    fold_meta = dict(fold_meta or {})
                    model_failure_total += float(fold_per_model.get("_evaluation_failures_total", 0.0))
                    for diag_key, diag_value in fold_per_model.items():
                        if diag_key.startswith("_evaluation_failures_") and diag_key != "_evaluation_failures_total":
                            model_key = diag_key.replace("_evaluation_failures_", "", 1)
                            model_failure_by_model[model_key] = float(
                                model_failure_by_model.get(model_key, 0.0) + float(diag_value)
                            )
                    for model_key, model_score in fold_per_model.items():
                        if str(model_key).startswith("_"):
                            continue
                        per_model_fold_scores.setdefault(model_key, []).append(float(model_score))
                else:
                    fold_score, fold_signal = result
                    fold_meta = {}
            except Exception as exc:
                fold_score = float("nan")  # FIX T-A3-FIX-004: NaN, not 0.0
                fold_signal = np.zeros(len(val_idx), dtype=float)
                fold_meta = {}
                fold_exception_count += 1
            perf_scores.append(float(fold_score))
            pred_signal_parts.append(np.asarray(fold_signal, dtype=float).ravel())
            if fold_meta:
                conformal_fold_meta.append(dict(fold_meta))
                rate = float(fold_meta.get("conformal_singleton_rate", float("nan")))
                if np.isfinite(rate):
                    conformal_singleton_rates.append(float(rate))
                coverage = float(fold_meta.get("conformal_coverage", float("nan")))
                if np.isfinite(coverage):
                    conformal_coverages.append(float(coverage))

            if use_robust_oracle:
                X_aug, y_aug = augment_training_data_fn(X_train, y_train)
                try:
                    robust_result = fit_and_score_fold_fn(X_aug, y_aug, X_val, y_val)
                    aug_score = robust_result[0] if isinstance(robust_result, tuple) else robust_result
                except Exception as exc:
                    aug_score = fold_score
                    robust_fold_exception_count += 1
                robust_scores.append(float(aug_score - abs(aug_score - fold_score)))

        perf_scores = np.asarray(perf_scores, dtype=float)
        robust_scores = np.asarray(robust_scores, dtype=float) if robust_scores else np.array([], dtype=float)
        pred_signal = np.concatenate(pred_signal_parts) if pred_signal_parts else np.array([], dtype=float)
        conformal_arr = np.asarray(conformal_singleton_rates, dtype=float)
        conformal_cov_arr = np.asarray(conformal_coverages, dtype=float)

        method_result = candidate.get('method_result', {})
        selection_frequency = method_result.get('selection_frequency')
        if selection_frequency is not None and selected_indices.size > 0:
            sel_freq_arr = np.asarray(selection_frequency, dtype=float).ravel()
            selected_freq = sel_freq_arr[selected_indices] if sel_freq_arr.size > int(np.max(selected_indices)) else []
            freq_score = float(np.mean(selected_freq)) if len(selected_freq) > 0 else 0.5
        else:
            freq_score = 0.5

        perf_var = float(np.nanvar(perf_scores)) if perf_scores.size else 1.0
        perf_stability = 1.0 / (1.0 + perf_var)
        selector_stability = _compute_selector_stability_signal(
            method_result,
            selected_indices,
            n_features=int(X.shape[1]),
        )
        stability = float(selector_stability)

        complexity = 1.0 - (len(selected_indices) / max(X.shape[1], 1))
        if complexity_use_runtime_penalty:
            complexity -= 0.05 * np.log1p(candidate.get('runtime_sec', 0.0))
        complexity = float(complexity)

        evaluation[name] = {
            'performance_scores': perf_scores,
            'performance_mean': float(np.nanmean(perf_scores)) if perf_scores.size else 0.0,
            'stability': stability,
            'selector_stability': float(selector_stability),
            'performance_stability': float(perf_stability),
            'selection_frequency_score': float(freq_score),
            'complexity': complexity,
            'robustness': float(np.nanmean(robust_scores)) if robust_scores.size else (
                float(np.nanmean(perf_scores)) if perf_scores.size else 0.0
            ),
            'prediction_signal': pred_signal,
            'target_signal': target_signal,
            'selected_indices': selected_indices,
            'score_vector': np.asarray(candidate.get('score_vector', np.zeros(X.shape[1], dtype=float)), dtype=float),
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            # T-002: per-model fold scores for multi-model oracle construction
            'performance_scores_by_model': {
                k: np.asarray(v, dtype=float) for k, v in per_model_fold_scores.items()
            } if per_model_fold_scores else {},
            'conformal_singleton_rates': conformal_arr,
            'conformal_singleton_rate_mean': (
                float(np.mean(conformal_arr)) if conformal_arr.size else float("nan")
            ),
            'conformal_singleton_rate_std': (
                float(np.std(conformal_arr, ddof=1)) if conformal_arr.size > 1 else 0.0
            ),
            'conformal_coverages': conformal_cov_arr,
            'conformal_coverage_mean': (
                float(np.mean(conformal_cov_arr)) if conformal_cov_arr.size else float("nan")
            ),
            'conformal_coverage_std': (
                float(np.std(conformal_cov_arr, ddof=1)) if conformal_cov_arr.size > 1 else 0.0
            ),
            'conformal_fold_meta': list(conformal_fold_meta),
            'evaluation_failures': {
                'fold_exceptions': int(fold_exception_count),
                'robust_fold_exceptions': int(robust_fold_exception_count),
                'model_failures_total': float(model_failure_total),
                'model_failures_by_model': {
                    str(k): float(v) for k, v in model_failure_by_model.items()
                },
            },
        }

    return evaluation


# ---------------------------------------------------------------------------
# 3. extract_portfolio
# ---------------------------------------------------------------------------
def extract_portfolio(candidate_names, candidate_weights, evaluation,
                      *, portfolio_size, use_diversity_oracle):
    """Select top weighted but complementary selector candidates."""
    m = len(candidate_names)
    if m == 0:
        return []

    max_k = int(max(1, min(portfolio_size, m)))
    ranked = list(np.argsort(candidate_weights)[::-1])
    selected = [ranked[0]]

    for idx in ranked[1:]:
        if len(selected) >= max_k:
            break

        if not use_diversity_oracle:
            selected.append(idx)
            continue

        name_i = candidate_names[idx]
        set_i = set(map(int, evaluation[name_i]['selected_indices'].tolist()))
        signal_i = evaluation[name_i]['prediction_signal']

        is_redundant = False
        for prev_idx in selected:
            name_j = candidate_names[prev_idx]
            set_j = set(map(int, evaluation[name_j]['selected_indices'].tolist()))
            signal_j = evaluation[name_j]['prediction_signal']

            union = len(set_i.union(set_j))
            overlap = (len(set_i.intersection(set_j)) / union) if union > 0 else 1.0

            n = min(signal_i.size, signal_j.size)
            corr = 0.0
            if n > 2:
                corr = np.corrcoef(signal_i[:n], signal_j[:n])[0, 1]
                corr = float(corr) if np.isfinite(corr) else 0.0

            if (
                overlap > PORTFOLIO_REDUNDANCY_OVERLAP_THRESHOLD
                or abs(corr) > PORTFOLIO_REDUNDANCY_CORR_THRESHOLD
            ):
                is_redundant = True
                break

        if not is_redundant:
            selected.append(idx)

    if len(selected) < max_k:
        for idx in ranked:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= max_k:
                break

    return selected


def resolve_adaptive_portfolio_size(
    candidate_weights: np.ndarray,
    *,
    portfolio_size: int,
    adaptive_enabled: bool,
    adaptive_size_min: int | None,
    adaptive_size_max: int | None,
    adaptive_sizing_variance_penalty: bool = False,
    adaptive_sizing_variance_penalty_strength: float = 0.5,
) -> Tuple[int, Dict[str, Any]]:
    """Resolve effective MNPO portfolio size with bounded adaptive sizing.

    Heuristic:
      1) pick first k whose cumulative mass reaches 85% (`k_mass`)
      2) pick elbow from largest adjacent drop in sorted weights (`k_elbow`)
      3) average both and clip to [adaptive_size_min, adaptive_size_max]

    When disabled or inputs are invalid, this returns ``portfolio_size``.
    """
    requested = int(max(1, portfolio_size))
    weights = np.asarray(candidate_weights, dtype=float).ravel()
    n_candidates = int(max(0, weights.size))
    fallback = int(max(1, min(requested, max(1, n_candidates))))

    meta: Dict[str, Any] = {
        "adaptive_enabled": bool(adaptive_enabled),
        "requested_size": int(requested),
        "effective_size": int(fallback),
        "bounds": {
            "min": None if adaptive_size_min is None else int(adaptive_size_min),
            "max": None if adaptive_size_max is None else int(adaptive_size_max),
        },
        "reason": "disabled",
        "k_mass": None,
        "k_elbow": None,
        "mass_target": 0.85,
        "variance_penalty_enabled": bool(adaptive_sizing_variance_penalty),
        "variance_penalty_strength": float(max(0.0, adaptive_sizing_variance_penalty_strength)),
        "variance_penalty_term": 0.0,
    }

    if not adaptive_enabled:
        return int(fallback), meta
    if adaptive_size_min is None or adaptive_size_max is None:
        meta["reason"] = "missing_bounds"
        return int(fallback), meta

    lo = int(max(1, adaptive_size_min))
    hi = int(max(lo, adaptive_size_max))
    if lo > hi:
        meta["reason"] = "invalid_bounds"
        return int(fallback), meta

    if n_candidates <= 0:
        meta["reason"] = "no_candidates"
        return int(max(lo, min(hi, requested))), meta

    sorted_w = np.sort(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0))[::-1]
    sorted_w = np.clip(sorted_w, 0.0, None)
    total = float(np.sum(sorted_w))
    if total <= 1e-12:
        meta["reason"] = "degenerate_weights"
        fallback = int(np.clip(requested, lo, min(hi, n_candidates)))
        meta["effective_size"] = int(fallback)
        return int(fallback), meta

    norm_w = sorted_w / total
    cum_mass = np.cumsum(norm_w)
    k_mass = int(np.searchsorted(cum_mass, 0.85, side="left") + 1)
    if norm_w.size <= 1:
        k_elbow = 1
    else:
        diffs = norm_w[:-1] - norm_w[1:]
        k_elbow = int(np.argmax(diffs) + 1)

    target = int(round(0.5 * float(k_mass + k_elbow)))
    if bool(adaptive_sizing_variance_penalty):
        var = float(np.var(norm_w, ddof=1)) if norm_w.size > 1 else 0.0
        penalty_strength = float(max(0.0, adaptive_sizing_variance_penalty_strength))
        penalty_term = float(np.clip(np.ceil(penalty_strength * var * float(n_candidates)), 0.0, float(n_candidates)))
        target = int(max(1, target - int(penalty_term)))
        meta["variance_penalty_term"] = float(penalty_term)
    upper = int(min(hi, n_candidates))
    effective = int(np.clip(target, lo, upper))

    meta.update(
        {
            "reason": "ok",
            "k_mass": int(k_mass),
            "k_elbow": int(k_elbow),
            "effective_size": int(effective),
        }
    )
    return int(effective), meta


def resolve_pareto_portfolio_size(
    candidate_weights: np.ndarray,
    *,
    portfolio_size: int,
    n_available_methods: int,
) -> Tuple[int, Dict[str, Any]]:
    """Pareto-front portfolio sizing (T-R-266).

    For each candidate portfolio size k ∈ [2, min(12, n_available_methods)]:
      - Compute mean weight (proxy for expected BA) of the top-k methods.
      - Compute sparsity = 1/k.
    Build a Pareto front (maximize both), select the knee-point via
    distance-to-utopia heuristic.

    Falls back to ``portfolio_size`` if the Pareto front is degenerate (< 3 points).
    """
    weights = np.asarray(candidate_weights, dtype=float).ravel()
    n = int(max(0, weights.size))
    requested = int(max(1, portfolio_size))
    n_methods = int(max(1, n_available_methods))

    meta: Dict[str, Any] = {
        "pareto_enabled": True,
        "requested_size": int(requested),
        "effective_size": int(requested),
        "n_candidates": int(n),
        "n_methods": int(n_methods),
        "reason": "ok",
        "pareto_front_sizes": [],
        "pareto_front_quality": [],
        "pareto_front_sparsity": [],
        "knee_index": -1,
    }

    if n <= 1:
        meta["reason"] = "insufficient_candidates"
        meta["effective_size"] = int(max(1, min(requested, n)))
        return int(meta["effective_size"]), meta

    sorted_w = np.sort(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0))[::-1]
    sorted_w = np.clip(sorted_w, 0.0, None)
    total = float(np.sum(sorted_w))
    if total <= 1e-12:
        meta["reason"] = "degenerate_weights"
        return int(requested), meta

    norm_w = sorted_w / total

    k_max = int(min(12, n_methods, n))
    k_min = 2
    if k_max < k_min:
        meta["reason"] = "range_too_small"
        meta["effective_size"] = int(max(1, min(requested, n)))
        return int(meta["effective_size"]), meta

    # Compute quality (mean weight of top-k) and sparsity (1/k) for each k.
    sizes = list(range(k_min, k_max + 1))
    qualities = []
    sparsities = []
    for k in sizes:
        q = float(np.mean(norm_w[:k]))
        s = 1.0 / float(k)
        qualities.append(q)
        sparsities.append(s)

    qualities_arr = np.array(qualities, dtype=float)
    sparsities_arr = np.array(sparsities, dtype=float)

    # Pareto front: non-dominated points (maximize both quality and sparsity).
    n_pts = len(sizes)
    dominated = np.zeros(n_pts, dtype=bool)
    for i in range(n_pts):
        for j in range(n_pts):
            if i == j:
                continue
            if qualities_arr[j] >= qualities_arr[i] and sparsities_arr[j] >= sparsities_arr[i]:
                if qualities_arr[j] > qualities_arr[i] or sparsities_arr[j] > sparsities_arr[i]:
                    dominated[i] = True
                    break

    pareto_mask = ~dominated
    n_pareto = int(pareto_mask.sum())

    if n_pareto < 3:
        meta["reason"] = "degenerate_front"
        meta["effective_size"] = int(requested)
        meta["pareto_front_sizes"] = [sizes[i] for i in range(n_pts) if pareto_mask[i]]
        return int(meta["effective_size"]), meta

    # Knee point: distance to utopia (maximum quality, maximum sparsity).
    pareto_q = qualities_arr[pareto_mask]
    pareto_s = sparsities_arr[pareto_mask]
    pareto_sizes = [sizes[i] for i in range(n_pts) if pareto_mask[i]]

    # Normalize to [0, 1] for distance calculation.
    q_range = float(pareto_q.max() - pareto_q.min()) or 1.0
    s_range = float(pareto_s.max() - pareto_s.min()) or 1.0
    q_norm = (pareto_q - pareto_q.min()) / q_range
    s_norm = (pareto_s - pareto_s.min()) / s_range

    # Utopia point: (1, 1) in normalized space.
    distances = np.sqrt((1.0 - q_norm) ** 2 + (1.0 - s_norm) ** 2)
    knee_idx = int(np.argmin(distances))
    effective = int(pareto_sizes[knee_idx])

    meta.update({
        "reason": "ok",
        "effective_size": int(effective),
        "pareto_front_sizes": pareto_sizes,
        "pareto_front_quality": [float(v) for v in pareto_q],
        "pareto_front_sparsity": [float(v) for v in pareto_s],
        "knee_index": int(knee_idx),
    })
    return int(effective), meta


def _is_interaction_candidate(name: str) -> bool:
    """Heuristic interaction-capability tag for paradigm-aware prior."""
    key = str(name).strip().lower()
    spec = METHOD_REGISTRY.get(key)
    paradigm = str(getattr(spec, "paradigm", "")).strip().lower() if spec is not None else ""
    if paradigm in {"pairwise", "knockoff"}:
        return True
    # Filter/multiclass methods that explicitly model local interactions or
    # class-conditional structure.
    if key in {
        "relieff",
        "cmim",
        "dove_class_specific",
        "class_pareto_front",
        "joint_auc_l1",
        "ktsp",
    }:
        return True
    return False


def apply_paradigm_aware_prior_floor(
    reference_prior: np.ndarray,
    candidate_names: Sequence[str],
    *,
    interaction_floor: float,
):
    """Guarantee a minimum prior mass for interaction-capable candidates."""
    prior = np.asarray(reference_prior, dtype=float).ravel()
    names = [str(n) for n in candidate_names]
    n = int(min(prior.size, len(names)))
    if n <= 0:
        return prior, {
            "enabled": True,
            "applied": False,
            "reason": "empty",
            "interaction_floor": float(interaction_floor),
            "interaction_mass_before": 0.0,
            "interaction_mass_after": 0.0,
            "interaction_candidates": [],
        }
    prior = np.asarray(prior[:n], dtype=float)
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.sum(prior)) <= 1e-12:
        prior = np.full(n, 1.0 / float(n), dtype=float)
    else:
        prior = prior / float(np.sum(prior))

    floor = float(np.clip(interaction_floor, 0.0, 0.95))
    interaction_mask = np.array([_is_interaction_candidate(nm) for nm in names[:n]], dtype=bool)
    interaction_names = [names[i] for i in range(n) if interaction_mask[i]]
    current = float(np.sum(prior[interaction_mask])) if np.any(interaction_mask) else 0.0
    meta = {
        "enabled": True,
        "applied": False,
        "reason": "no_change",
        "interaction_floor": float(floor),
        "interaction_mass_before": float(current),
        "interaction_mass_after": float(current),
        "interaction_candidates": interaction_names,
    }
    if floor <= 0.0:
        meta["reason"] = "zero_floor"
        return prior, meta
    if not np.any(interaction_mask):
        meta["reason"] = "no_interaction_candidates"
        return prior, meta
    if current >= floor - 1e-12:
        meta["reason"] = "already_satisfied"
        return prior, meta

    non_mask = ~interaction_mask
    non_mass = float(np.sum(prior[non_mask]))
    deficit = float(floor - current)
    if non_mass <= 1e-12 or deficit <= 1e-12:
        meta["reason"] = "insufficient_noninteraction_mass"
        return prior, meta

    shift = float(min(deficit, non_mass))
    updated = prior.copy()
    updated[non_mask] = updated[non_mask] * ((non_mass - shift) / max(non_mass, 1e-12))
    inter_mass = float(np.sum(updated[interaction_mask]))
    if inter_mass <= 1e-12:
        updated[interaction_mask] += shift / float(max(1, int(np.sum(interaction_mask))))
    else:
        updated[interaction_mask] += shift * (updated[interaction_mask] / inter_mass)
    updated = np.asarray(np.nan_to_num(updated, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    if float(np.sum(updated)) <= 1e-12:
        updated = np.full(n, 1.0 / float(n), dtype=float)
    else:
        updated = updated / float(np.sum(updated))
    after = float(np.sum(updated[interaction_mask]))
    meta.update(
        {
            "applied": True,
            "reason": "mass_shifted",
            "interaction_mass_after": float(after),
            "mass_shift": float(after - current),
        }
    )
    return updated, meta


# ---------------------------------------------------------------------------
# 4. mnpo_aggregate_feature_votes
# ---------------------------------------------------------------------------
def mnpo_aggregate_feature_votes(candidates, candidate_names, n_features):
    """Aggregate feature scores from portfolio-weighted candidates."""
    feature_votes = np.zeros(n_features, dtype=float)
    feature_details = {i: {} for i in range(n_features)}

    for candidate_name in candidate_names:
        candidate = candidates[candidate_name]
        score_vector = candidate['score_vector']
        selected_indices = candidate['selected_indices']
        selected_set = set(map(int, selected_indices.tolist()))
        rank_lookup = {int(idx): int(rank + 1) for rank, idx in enumerate(selected_indices)}
        portfolio_weight = float(candidate.get('portfolio_weight', 0.0))

        if portfolio_weight > 0:
            feature_votes += portfolio_weight * score_vector

        for idx in range(n_features):
            details = {
                'selected': idx in selected_set,
                'score': float(score_vector[idx]),
                'portfolio_weight': portfolio_weight,
                'method_weight': portfolio_weight,
            }
            if idx in rank_lookup:
                details['rank'] = rank_lookup[idx]
            feature_details[idx][candidate_name] = details

    return feature_votes, feature_details


# ---------------------------------------------------------------------------
# 4b. Kuncheva stability index (T-R-271)
# ---------------------------------------------------------------------------

def compute_kuncheva_stability_index(
    selection_frequency: np.ndarray,
    n_selected: int,
    n_runs: int,
) -> float:
    """Compute the Kuncheva stability index from selection frequency vectors.

    The Kuncheva stability index (KSI) measures the consistency of feature
    selection across multiple runs.  Given the per-feature selection frequency
    vector ``f`` (fraction of runs each feature was selected) produced by
    derandomized/bootstrap methods, KSI is computed as:

        KSI = (1/C(R,2)) * Σ_{i<j} [ (|S_i ∩ S_j| - k²/p) / (k - k²/p) ]

    This is equivalent to:

        KSI = [ p * Σ f_j² - k² ] / [ k * (p - k) ]

    where:
       - ``p`` = total number of features
       - ``k`` = average number of selected features per run  (≈ n_selected)
       - ``f_j`` = fraction of runs that selected feature j
       - R = number of runs

    Returns a value in [-1, 1] where:
       - 1.0 means perfect agreement (all runs select exactly the same features)
       - 0.0 means random-level agreement
       - negative means less-than-random agreement

    For single-run methods (n_runs <= 1) or degenerate cases (k=0 or k=p),
    return 1.0 (no evidence of instability).

    References:
        Kuncheva, L.I. (2007). A stability index for feature selection.
        Proc. 25th IASTED Int'l Conf. on AI and Applications, pp. 390-395.
    """
    p = len(selection_frequency)
    k = float(n_selected)
    if n_runs <= 1 or k <= 0 or k >= p:
        return 1.0
    # Σ f_j²  (sum of squared selection frequencies)
    sum_f_sq = float(np.sum(selection_frequency ** 2))
    denom = k * (p - k)
    if denom <= 0:
        return 1.0
    ksi = (p * sum_f_sq - k * k) / denom
    # Clamp to [-1, 1] for numerical safety.
    return float(np.clip(ksi, -1.0, 1.0))


def _mean_pairwise_jaccard(supports: Sequence[np.ndarray]) -> float:
    valid = []
    for support in supports:
        arr = np.asarray(support, dtype=int).ravel()
        arr = np.asarray(sorted(set(int(v) for v in arr if int(v) >= 0)), dtype=int)
        if arr.size > 0:
            valid.append(set(int(v) for v in arr.tolist()))
    if len(valid) < 2:
        return 1.0
    vals = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            union = len(valid[i].union(valid[j]))
            vals.append(float(len(valid[i].intersection(valid[j])) / union) if union > 0 else 1.0)
    return float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 1.0


def _extract_support_telemetry(
    method_result: Mapping[str, Any],
    *,
    n_features: int,
) -> List[np.ndarray]:
    support_keys = (
        "bootstrap_supports",
        "bootstrap_selected_indices",
        "selection_sets",
        "selected_sets",
        "support_sets",
    )
    for key in support_keys:
        raw = method_result.get(key)
        if raw is None:
            continue
        supports: List[np.ndarray] = []
        if isinstance(raw, (list, tuple)):
            for item in raw:
                item_arr = np.asarray(item)
                if item_arr.ndim == 1 and item_arr.dtype == bool and item_arr.size == int(n_features):
                    supports.append(np.where(item_arr)[0])
                else:
                    supports.append(np.asarray(item, dtype=int).ravel())
        elif isinstance(raw, np.ndarray) and raw.ndim == 2:
            mat = np.asarray(raw)
            if mat.shape[1] == int(n_features) and (
                mat.dtype == bool or np.issubdtype(mat.dtype, np.number)
            ):
                for row in mat:
                    supports.append(np.where(np.asarray(row, dtype=float) > 0.0)[0])
            else:
                for row in mat:
                    supports.append(np.asarray(row, dtype=int).ravel())
        else:
            arr = np.asarray(raw, dtype=object)
            if arr.ndim != 1:
                continue
            for item in arr.tolist():
                supports.append(np.asarray(item, dtype=int).ravel())
        if len(supports) >= 2:
            return supports
    return []


def _compute_selector_stability_signal(
    method_result: Mapping[str, Any],
    selected_indices: np.ndarray,
    *,
    n_features: int,
) -> float:
    """Bootstrap/top-k selector stability signal normalized to [0, 1]."""
    selected = np.asarray(selected_indices, dtype=int).ravel()
    selected = np.asarray(
        sorted(set(int(i) for i in selected if 0 <= int(i) < int(n_features))),
        dtype=int,
    )
    if selected.size <= 0 or int(n_features) <= 0:
        return 1.0

    supports = _extract_support_telemetry(method_result, n_features=int(n_features))
    if len(supports) >= 2:
        return _mean_pairwise_jaccard(supports)

    freq = None
    for key in (
        "selection_frequency",
        "copula_stabilizer_support_frequency",
        "support_frequency",
        "bootstrap_selection_frequency",
    ):
        if key in method_result:
            try:
                freq = np.asarray(method_result[key], dtype=float).ravel()
            except Exception:
                freq = None
            if freq is not None:
                break

    if freq is not None and freq.size > 0:
        probs = np.zeros(int(n_features), dtype=float)
        usable = int(min(probs.size, freq.size))
        probs[:usable] = np.clip(np.nan_to_num(freq[:usable], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        if np.any(probs > 0.0):
            expected_intersection = float(np.sum(probs * probs))
            expected_union = float(np.sum(2.0 * probs - probs * probs))
            expected_jaccard = expected_intersection / expected_union if expected_union > 1e-12 else 1.0
            selected_capture = float(np.mean(probs[selected])) if selected.size else expected_jaccard
            return float(np.clip(0.70 * expected_jaccard + 0.30 * selected_capture, 0.0, 1.0))

    for key in ("selector_stability_score", "stability_score"):
        if key not in method_result:
            continue
        try:
            val = np.asarray(method_result[key], dtype=float)
        except Exception:
            continue
        if val.ndim == 0:
            scalar = float(val)
        elif val.size > int(np.max(selected)):
            scalar = float(np.mean(val.ravel()[selected]))
        else:
            scalar = float(np.mean(val.ravel())) if val.size else 1.0
        if np.isfinite(scalar):
            return float(np.clip(scalar, 0.0, 1.0))

    return 1.0


def _compute_selector_stability_scores(
    candidates: Dict[str, Dict[str, Any]],
    racing_candidates: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Compute a top-k selector stability score in [0, 1] for each candidate."""
    scores: Dict[str, float] = {}
    for name in racing_candidates:
        cand = candidates.get(name) or racing_candidates.get(name)
        if cand is None:
            scores[name] = 1.0
            continue
        mr = cand.get("method_result", {})
        selected = np.asarray(cand.get("selected_indices", np.asarray([], dtype=int)), dtype=int)
        score_vector = np.asarray(cand.get("score_vector", np.asarray([], dtype=float)), dtype=float).ravel()
        n_features = int(score_vector.size) if score_vector.size else int(np.max(selected) + 1) if selected.size else 0
        scores[name] = _compute_selector_stability_signal(
            mr,
            selected,
            n_features=int(max(0, n_features)),
        )
    return scores


_RECOGNIZED_BUDGET_STOP_REASONS = frozenset(
    {
        "runtime_budget_exhausted",
        "unique_pair_evaluation_budget_exhausted",
    }
)


def _optional_execution_bool(
    payload: Mapping[str, Any], field: str
) -> Tuple[Any, bool]:
    """Read an optional execution-state boolean without truthiness coercion."""

    if field not in payload or payload.get(field) is None:
        return None, False
    value = payload.get(field)
    if isinstance(value, (bool, np.bool_)):
        return bool(value), False
    return None, True


def _optional_execution_text(
    payload: Mapping[str, Any], field: str
) -> Tuple[str, bool]:
    """Read optional execution-state text while rejecting non-string payloads."""

    if field not in payload or payload.get(field) is None:
        return "", False
    value = payload.get(field)
    if isinstance(value, str):
        return str(value).strip(), False
    return "", True


def selector_result_eligibility(results: Any) -> Dict[str, Any]:
    """Classify one selector result before it enters an aggregation path.

    Legacy selectors without execution-state fields retain their historical
    eligible path. An explicitly incomplete selector cannot be treated as normal
    evidence. The opt-in mRMR ``relevance_only`` fallback remains available to
    legacy voting, but never enters MNPO or its synthetic consensus candidates.
    """

    payload = results if isinstance(results, Mapping) else {}
    complete, malformed_complete = _optional_execution_bool(payload, "complete")
    explicit_incomplete, malformed_incomplete = _optional_execution_bool(
        payload, "incomplete"
    )
    fallback_applied, malformed_fallback = _optional_execution_bool(
        payload, "fallback_applied"
    )
    budget_exhausted_reported, malformed_budget_exhausted = (
        _optional_execution_bool(payload, "budget_exhausted")
    )
    fallback_mode, malformed_fallback_mode = _optional_execution_text(
        payload, "budget_fallback_mode_requested"
    )
    budget_status, malformed_budget_status = _optional_execution_text(
        payload, "budget_status"
    )
    stop_reason, malformed_stop_reason = _optional_execution_text(
        payload, "stop_reason"
    )
    malformed_fields = [
        field
        for field, malformed in (
            ("complete", malformed_complete),
            ("incomplete", malformed_incomplete),
            ("fallback_applied", malformed_fallback),
            ("budget_exhausted", malformed_budget_exhausted),
            ("budget_fallback_mode_requested", malformed_fallback_mode),
            ("budget_status", malformed_budget_status),
            ("stop_reason", malformed_stop_reason),
        )
        if malformed
    ]
    budget_status_normalized = budget_status.lower()
    stop_reason_normalized = stop_reason.lower()
    budget_exhausted = bool(
        budget_exhausted_reported is True
        or budget_status_normalized == "exhausted"
        or stop_reason_normalized in _RECOGNIZED_BUDGET_STOP_REASONS
    )
    incomplete = bool(
        explicit_incomplete is True
        or complete is False
        or budget_exhausted
        or malformed_fields
    )

    status = "eligible"
    mnpo_candidate_eligible = True
    mnpo_consensus_eligible = True
    legacy_vote_eligible = True
    fail_closed = False
    if malformed_fields:
        status = "malformed_execution_state"
        mnpo_candidate_eligible = False
        mnpo_consensus_eligible = False
        legacy_vote_eligible = False
        fail_closed = True
    elif incomplete:
        if fallback_applied is True and fallback_mode.lower() == "relevance_only":
            status = "relevance_only_fallback"
            mnpo_candidate_eligible = False
            mnpo_consensus_eligible = False
        else:
            status = "incomplete_excluded"
            mnpo_candidate_eligible = False
            mnpo_consensus_eligible = False
            legacy_vote_eligible = False
            fail_closed = True
    elif fallback_applied is True:
        status = "fallback_excluded"
        mnpo_candidate_eligible = False
        mnpo_consensus_eligible = False
        legacy_vote_eligible = False

    return {
        "status": status,
        "complete": complete,
        "incomplete": bool(incomplete),
        "fallback_applied": bool(fallback_applied is True),
        "budget_status": budget_status,
        "budget_exhausted": bool(budget_exhausted),
        "stop_reason": stop_reason,
        "execution_state_malformed": bool(malformed_fields),
        "execution_state_malformed_fields": list(malformed_fields),
        "mnpo_candidate_eligible": bool(mnpo_candidate_eligible),
        "mnpo_consensus_eligible": bool(mnpo_consensus_eligible),
        "legacy_vote_eligible": bool(legacy_vote_eligible),
        "fail_closed": bool(fail_closed),
    }


def resolve_method_result_eligibility(
    method_results: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Resolve eligibility for the standard ``(result, scores)`` method map."""

    resolved: Dict[str, Dict[str, Any]] = {}
    for name, entry in method_results.items():
        result = entry[0] if isinstance(entry, (tuple, list)) and entry else entry
        resolved[str(name)] = selector_result_eligibility(result)
    return resolved


# ---------------------------------------------------------------------------
# 5. mnpo_select_features — main MNPO orchestration
# ---------------------------------------------------------------------------
def mnpo_select_features(
    X_uncorr, y, n_target, n_final_features, method_results, method_runtimes,
    *,
    # Callables from base.py
    safe_normalize_scores_fn,
    calculate_weighted_votes_fn,
    get_inner_cv_splits_fn,
    fit_and_score_fold_fn,
    augment_training_data_fn,
    oracle=None,
    diakrino_relevance_vector=None,  # §2.2 per-feature DIAKRINO relevance, aligned to X_uncorr columns
    diakrino_selector_prior=None,    # §4 per-dataset DIAKRINO selector-pool weights
    diakrino_selector_prior_trust_record=None,  # artifact-side selector-prior calibration contract
    # MNPO config
    mnpo_include_legacy_consensus,
    mnpo_include_majority_consensus,
    mnpo_consensus_exclude_methods,
    mnpo_consensus_exclude_protect_top_k,
    use_tritrust,
    use_oracle_redundancy_penalty,
    compute_tremble_sensitivity,
    mirror_descent_steps,
    mirror_descent_eta,
    mirror_descent_lambda,
    wrapper_refine_enabled,
    rank_aggregation_mode,
    # Portfolio config
    portfolio_size,
    adaptive_portfolio_sizing_enabled=False,
    adaptive_size_min=None,
    adaptive_size_max=None,
    adaptive_sizing_variance_penalty=False,
    adaptive_sizing_variance_penalty_strength=0.5,
    # T-R-266: Pareto-front portfolio sizing (opt-in).
    pareto_portfolio_sizing_enabled=False,
    # T-R-271: stability-weighted portfolio aggregation (opt-in).
    stability_weighted_aggregation_enabled=False,
    use_diversity_oracle,
    # Racing config
    runtime_racing_enabled,
    runtime_racing_mode,
    runtime_racing_proxy_splits,
    runtime_racing_keep_fraction,
    runtime_racing_min_candidates,
    runtime_racing_runtime_weight,
    runtime_racing_stages,
    runtime_racing_confidence_bound,
    runtime_racing_delta,
    # Evaluation config
    use_robust_oracle,
    complexity_use_runtime_penalty,
    # Oracle config
    pairwise_delta,
    use_cvar=False,
    cvar_alpha=0.33,
    use_tail_risk_oracle,
    tail_risk_alpha,
    use_qre_smoothing,
    qre_temperature_gamma,
    use_regret_oracle,
    use_stability_oracle,
    use_complexity_oracle,
    diversity_oracle_mode,
    oracle_weighting_mode="tritrust",
    shapley_n_coalitions_max=4096,
    shapley_bayesian_shrinkage=False,
    shapley_bayesian_prior_strength=8.0,
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
    fold_preference_mode="vote",
    use_diakrino_selector_prior=False,
    diakrino_selector_prior_weight=1.0,
    diakrino_selector_prior_calibration=DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT,
    use_conformal_efficiency=False,
    conformal_efficiency_method="split",
    oracle_weight_js_shrinkage=False,
    complexity_conditioning=False,
    payoff_shrinkage_kappa=0.0,
    diversity_redundancy_weight,
    diversity_complementarity_weight,
    # Wrapper refinement config
    wrapper_refine_top_k,
    wrapper_refine_max_add,
    wrapper_refine_min_gain,
    # Multi-model oracle config (T-002)
    performance_oracle_mode="single",
    # T-R-128: optional paradigm-aware reference prior floor.
    mnpo_paradigm_aware_prior_enabled=False,
    mnpo_interaction_floor=0.12,
    # T-P3-011: post-selection Rashomon importance bounds
    rashomon_enabled=False,
    rashomon_max_models=12,
    rashomon_score_tolerance=0.01,
    # T-R-243: per-selector weight penalty map.
    selector_penalty_map=None,
    random_state=42,
):
    """MNPO-inspired selector strategy."""
    # T-R-180: nested OracleConfig support with backward-compatible flat params.
    if oracle is None:
        try:
            from ..config import OracleConfig  # local import to avoid import-cycle at module load
            oracle = OracleConfig(
                pairwise_delta=float(pairwise_delta),
                use_tritrust=bool(use_tritrust),
                use_stability_oracle=bool(use_stability_oracle),
                use_complexity_oracle=bool(use_complexity_oracle),
                use_robust_oracle=bool(use_robust_oracle),
                use_diversity_oracle=bool(use_diversity_oracle),
                use_cvar=bool(use_cvar),
                cvar_alpha=float(cvar_alpha),
                use_qre_smoothing=bool(use_qre_smoothing),
                qre_temperature_gamma=float(qre_temperature_gamma),
                use_oracle_redundancy_penalty=bool(use_oracle_redundancy_penalty),
                compute_tremble_sensitivity=bool(compute_tremble_sensitivity),
                diversity_mode=str(diversity_oracle_mode),
                diversity_redundancy_weight=float(diversity_redundancy_weight),
                diversity_complementarity_weight=float(diversity_complementarity_weight),
                performance_oracle_mode=str(performance_oracle_mode),
                weighting_mode=str(oracle_weighting_mode),
                shapley_n_coalitions_max=int(shapley_n_coalitions_max),
                shapley_bayesian_shrinkage=bool(shapley_bayesian_shrinkage),
                shapley_bayesian_prior_strength=float(shapley_bayesian_prior_strength),
                use_ubayfs=bool(use_ubayfs),
                ubayfs_n_bootstrap=int(ubayfs_n_bootstrap),
                ubayfs_min_n=int(ubayfs_min_n),
                ubayfs_prior_weight=float(ubayfs_prior_weight),
                use_conformal_uq=bool(use_conformal_uq),
                conformal_uq_alpha=float(conformal_uq_alpha),
                conformal_uq_min_folds=int(conformal_uq_min_folds),
                fold_preference_mode=str(fold_preference_mode),
                use_diakrino_selector_prior=bool(use_diakrino_selector_prior),
                diakrino_selector_prior_weight=float(diakrino_selector_prior_weight),
                diakrino_selector_prior_calibration=str(diakrino_selector_prior_calibration),
                use_conformal_efficiency=bool(use_conformal_efficiency),
                conformal_efficiency_method=str(conformal_efficiency_method),
                oracle_weight_js_shrinkage=bool(oracle_weight_js_shrinkage),
                complexity_conditioning=bool(complexity_conditioning),
            )
        except Exception as exc:
            oracle = None

    def _oc(name: str, fallback):
        if oracle is None:
            return fallback
        return getattr(oracle, name, fallback)

    pairwise_delta = float(_oc("pairwise_delta", pairwise_delta))
    use_tritrust = bool(_oc("use_tritrust", use_tritrust))
    use_stability_oracle = bool(_oc("use_stability_oracle", use_stability_oracle))
    use_complexity_oracle = bool(_oc("use_complexity_oracle", use_complexity_oracle))
    use_robust_oracle = bool(_oc("use_robust_oracle", use_robust_oracle))
    use_diversity_oracle = bool(_oc("use_diversity_oracle", use_diversity_oracle))
    use_cvar = bool(_oc("use_cvar", use_cvar))
    cvar_alpha = float(_oc("cvar_alpha", cvar_alpha))
    use_qre_smoothing = bool(_oc("use_qre_smoothing", use_qre_smoothing))
    qre_temperature_gamma = float(_oc("qre_temperature_gamma", qre_temperature_gamma))
    use_oracle_redundancy_penalty = bool(
        _oc("use_oracle_redundancy_penalty", use_oracle_redundancy_penalty)
    )
    compute_tremble_sensitivity = bool(
        _oc("compute_tremble_sensitivity", compute_tremble_sensitivity)
    )
    diversity_oracle_mode = str(_oc("diversity_mode", diversity_oracle_mode))
    diversity_redundancy_weight = float(
        _oc("diversity_redundancy_weight", diversity_redundancy_weight)
    )
    diversity_complementarity_weight = float(
        _oc("diversity_complementarity_weight", diversity_complementarity_weight)
    )
    performance_oracle_mode = str(_oc("performance_oracle_mode", performance_oracle_mode))
    oracle_weighting_mode = str(_oc("weighting_mode", oracle_weighting_mode))
    shapley_n_coalitions_max = int(_oc("shapley_n_coalitions_max", shapley_n_coalitions_max))
    shapley_bayesian_shrinkage = bool(_oc("shapley_bayesian_shrinkage", False))
    shapley_bayesian_prior_strength = float(_oc("shapley_bayesian_prior_strength", 8.0))
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
    fold_preference_mode = str(_oc("fold_preference_mode", fold_preference_mode) or "vote")
    use_diakrino_selector_prior = bool(_oc("use_diakrino_selector_prior", use_diakrino_selector_prior))
    diakrino_selector_prior_weight = float(
        np.clip(_oc("diakrino_selector_prior_weight", diakrino_selector_prior_weight), 0.0, 1.0)
    )
    diakrino_selector_prior_calibration = str(
        _oc("diakrino_selector_prior_calibration", diakrino_selector_prior_calibration)
        or DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT
    )
    use_conformal_efficiency = bool(
        _oc("use_conformal_efficiency", use_conformal_efficiency)
    )
    conformal_efficiency_method = str(
        _oc("conformal_efficiency_method", conformal_efficiency_method) or "split"
    )
    oracle_weight_js_shrinkage = bool(
        _oc("oracle_weight_js_shrinkage", oracle_weight_js_shrinkage)
    )
    complexity_conditioning = bool(_oc("complexity_conditioning", complexity_conditioning))
    payoff_shrinkage_kappa = float(max(0.0, payoff_shrinkage_kappa))

    n_features = X_uncorr.shape[1]
    candidates = {}
    consensus_source_methods = set()
    method_result_eligibility = resolve_method_result_eligibility(method_results)
    degraded_excluded_methods = [
        name
        for name, status in method_result_eligibility.items()
        if not bool(status["mnpo_candidate_eligible"])
    ]

    for method_name, (results, all_scores) in method_results.items():
        status = method_result_eligibility.get(str(method_name), {})
        if not bool(status.get("mnpo_candidate_eligible", True)):
            continue
        if not results or 'selected_indices' not in results:
            continue
        selected = np.array(
            sorted(set(int(i) for i in results['selected_indices'] if 0 <= int(i) < n_features)),
            dtype=int,
        )
        if selected.size == 0:
            continue
        score_vector = safe_normalize_scores_fn(all_scores, selected, n_features)
        candidates[method_name] = {
            'selected_indices': selected,
            'score_vector': score_vector,
            'runtime_sec': float(method_runtimes.get(method_name, 0.0)),
            'method_result': results,
        }

    # Add synthetic consensus candidates for richer selector competition.
    if candidates:
        # Keep consensus references anchored to core/base selectors so newly added
        # experimental methods do not destabilize majority/legacy candidates.
        experimental_methods = get_experimental_keys()
        consensus_method_results = {
            name: payload
            for name, payload in method_results.items()
            if name not in experimental_methods
            and bool(
                method_result_eligibility.get(str(name), {}).get(
                    "mnpo_consensus_eligible", True
                )
            )
        }
        if not consensus_method_results:
            consensus_method_results = {
                name: payload
                for name, payload in method_results.items()
                if bool(
                    method_result_eligibility.get(str(name), {}).get(
                        "mnpo_consensus_eligible", True
                    )
                )
            }
        consensus_source_methods = set(consensus_method_results.keys())

        legacy_votes, _ = calculate_weighted_votes_fn(consensus_method_results, n_features)
        legacy_scores = normalize_vector_01(legacy_votes)
        legacy_selected = np.argsort(legacy_scores)[::-1][:max(n_target, n_final_features)]
        if mnpo_include_legacy_consensus:
            candidates['legacy_consensus'] = {
                'selected_indices': np.array(legacy_selected, dtype=int),
                'score_vector': legacy_scores,
                'runtime_sec': float(np.mean(list(method_runtimes.values())) if method_runtimes else 0.0),
                'method_result': {
                    'selected_indices': np.array(legacy_selected, dtype=int),
                    'scores': {int(i): float(legacy_scores[i]) for i in legacy_selected},
                    'method': 'legacy_consensus'
                },
            }

        selection_counts = np.zeros(n_features, dtype=float)
        for _, (results, _) in consensus_method_results.items():
            if not results or 'selected_indices' not in results:
                continue
            valid_indices = [
                int(idx) for idx in results['selected_indices']
                if 0 <= int(idx) < n_features
            ]
            if valid_indices:
                selection_counts[np.array(valid_indices, dtype=int)] += 1.0
        if mnpo_include_majority_consensus and np.max(selection_counts) > 0:
            consensus_scores = selection_counts / np.max(selection_counts)
            consensus_selected = np.argsort(consensus_scores)[::-1][:max(n_target, n_final_features)]
            candidates['majority_consensus'] = {
                'selected_indices': np.array(consensus_selected, dtype=int),
                'score_vector': consensus_scores,
                'runtime_sec': 0.0,
                'method_result': {
                    'selected_indices': np.array(consensus_selected, dtype=int),
                    'scores': {int(i): float(consensus_scores[i]) for i in consensus_selected},
                    'method': 'majority_consensus'
                },
            }

        rank_candidate_sources = {
            name: candidates[name] for name in consensus_method_results.keys() if name in candidates
        }
        if len(rank_candidate_sources) < 2:
            rank_candidate_sources = {
                name: payload
                for name, payload in candidates.items()
                if not name.endswith("_consensus") and not name.startswith("rank_aggregate_")
            }
        rank_candidate = build_rank_aggregation_candidate(
            rank_candidate_sources,
            n_target=n_target,
            n_final_features=n_final_features,
            n_features=n_features,
            rank_aggregation_mode=rank_aggregation_mode,
            normalize_fn=normalize_vector_01,
        )
        if rank_candidate is not None:
            rank_name, rank_payload = rank_candidate
            candidates[rank_name] = rank_payload

    if not candidates:
        return None

    racing_candidates, racing_meta = runtime_race_candidates(
        X_uncorr, y, candidates,
        get_inner_cv_splits_fn=get_inner_cv_splits_fn,
        fit_and_score_fold_fn=fit_and_score_fold_fn,
        runtime_racing_enabled=runtime_racing_enabled,
        runtime_racing_mode=runtime_racing_mode,
        runtime_racing_proxy_splits=runtime_racing_proxy_splits,
        runtime_racing_keep_fraction=runtime_racing_keep_fraction,
        runtime_racing_min_candidates=runtime_racing_min_candidates,
        runtime_racing_runtime_weight=runtime_racing_runtime_weight,
        runtime_racing_stages=runtime_racing_stages,
        runtime_racing_confidence_bound=runtime_racing_confidence_bound,
        runtime_racing_delta=runtime_racing_delta,
    )
    if not racing_candidates:
        racing_candidates = candidates
        racing_meta = {
            "runtime_racing_enabled": bool(runtime_racing_enabled),
            "runtime_racing_applied": False,
            "runtime_racing_mode": str(runtime_racing_mode),
            "runtime_racing_proxy_splits": int(runtime_racing_proxy_splits),
            "runtime_racing_keep_fraction": float(runtime_racing_keep_fraction),
            "runtime_racing_min_candidates": int(runtime_racing_min_candidates),
            "runtime_racing_runtime_weight": float(runtime_racing_runtime_weight),
            "runtime_racing_stages": int(runtime_racing_stages),
            "runtime_racing_confidence_bound": str(runtime_racing_confidence_bound),
            "runtime_racing_delta": float(runtime_racing_delta),
            "runtime_racing_initial_candidates": int(len(candidates)),
            "runtime_racing_kept_candidates": int(len(candidates)),
            "runtime_racing_kept_names": list(candidates.keys()),
            "runtime_racing_dropped_names": [],
            "runtime_racing_proxy_scores": {},
            "runtime_racing_stage_history": [],
        }

    candidate_names = list(racing_candidates.keys())
    evaluation = evaluate_candidate_library(
        X_uncorr, y, racing_candidates,
        get_inner_cv_splits_fn=get_inner_cv_splits_fn,
        fit_and_score_fold_fn=fit_and_score_fold_fn,
        augment_training_data_fn=augment_training_data_fn,
        use_robust_oracle=use_robust_oracle,
        complexity_use_runtime_penalty=complexity_use_runtime_penalty,
    )
    oracle_matrices, oracle_scores, oracle_components, oracle_pairwise_meta = estimate_oracle_preferences(
        candidate_names, evaluation,
        pairwise_delta=pairwise_delta,
        use_cvar=use_cvar,
        cvar_alpha=cvar_alpha,
        use_tail_risk_oracle=use_tail_risk_oracle,
        tail_risk_alpha=tail_risk_alpha,
        use_qre_smoothing=use_qre_smoothing,
        qre_temperature_gamma=qre_temperature_gamma,
        use_regret_oracle=use_regret_oracle,
        use_stability_oracle=use_stability_oracle,
        use_complexity_oracle=use_complexity_oracle,
        use_robust_oracle=use_robust_oracle,
        use_diversity_oracle=use_diversity_oracle,
        diversity_oracle_mode=diversity_oracle_mode,
        diversity_redundancy_weight=diversity_redundancy_weight,
        diversity_complementarity_weight=diversity_complementarity_weight,
        performance_oracle_mode=performance_oracle_mode,
        weighting_mode=oracle_weighting_mode,
        shapley_n_coalitions_max=shapley_n_coalitions_max,
        use_interaction_oracle=use_interaction_oracle,
        interaction_oracle_min_n_train=interaction_oracle_min_n_train,
        interaction_oracle_pool_size_cap=interaction_oracle_pool_size_cap,
        interaction_oracle_pair_cap=interaction_oracle_pair_cap,
        use_ubayfs=use_ubayfs,
        ubayfs_n_bootstrap=ubayfs_n_bootstrap,
        ubayfs_min_n=ubayfs_min_n,
        ubayfs_prior_weight=ubayfs_prior_weight,
        use_conformal_uq=use_conformal_uq,
        conformal_uq_alpha=conformal_uq_alpha,
        conformal_uq_min_folds=conformal_uq_min_folds,
        fold_preference_mode=fold_preference_mode,
        use_conformal_efficiency=use_conformal_efficiency,
        conformal_efficiency_method=conformal_efficiency_method,
        complexity_conditioning=complexity_conditioning,
        X_pool=np.asarray(X_uncorr, dtype=float),
        diakrino_relevance_vector=diakrino_relevance_vector,
        oracle_config=oracle,
        random_state=random_state,
    )
    oracle_cap_meta = enforce_oracle_count_cap(oracle_matrices, cap=MNPO_ORACLE_COUNT_CAP)
    oracle_fold_counts = [
        int(
            np.isfinite(
                np.asarray(
                    evaluation.get(name, {}).get("performance_scores", np.asarray([], dtype=float)),
                    dtype=float,
                ).ravel()
            ).sum()
        )
        for name in candidate_names
    ]
    oracle_effective_n = float(min(oracle_fold_counts)) if oracle_fold_counts else 1.0

    weighting_mode = str(oracle_weighting_mode or "tritrust").strip().lower()
    if weighting_mode not in {"tritrust", "uniform", "shapley", "banzhaf"}:
        weighting_mode = "tritrust"
    if weighting_mode == "shapley":
        oracle_weights, shapley_meta = _mnpo_fit_shapley_weights(
            oracle_matrices,
            reference="performance",
            max_coalitions=int(max(2, shapley_n_coalitions_max)),
        )
        if bool(shapley_bayesian_shrinkage):
            shrinkage_meta = compute_shapley_bayesian_shrinkage(
                evaluation=evaluation,
                candidate_names=candidate_names,
                prior_strength=float(shapley_bayesian_prior_strength),
            )
            oracle_weights = apply_bayesian_shrinkage_to_weights(
                oracle_weights,
                shrinkage_lambda=float(shrinkage_meta.get("shrinkage_lambda", 0.0)),
            )
            if isinstance(shapley_meta, dict):
                shapley_meta = dict(shapley_meta)
                shapley_meta["bayesian_shrinkage"] = dict(shrinkage_meta)
        oracle_weights_tritrust = fit_tritrust_weights(oracle_matrices)
    elif weighting_mode == "banzhaf":
        try:
            from ...mnpo_core import compute_banzhaf_values as _compute_banzhaf_values
        except Exception:
            try:
                from tabnetics.core.mnpo import compute_banzhaf_values as _compute_banzhaf_values
            except Exception:
                from tabnetics.core.mnpo import compute_banzhaf_values as _compute_banzhaf_values  # type: ignore[import-untyped]
        oracle_weights, shapley_meta = _compute_banzhaf_values(
            oracle_matrices,
            reference="performance",
            max_coalitions=int(max(2, shapley_n_coalitions_max)),
        )
        if bool(shapley_bayesian_shrinkage):
            shrinkage_meta = compute_shapley_bayesian_shrinkage(
                evaluation=evaluation,
                candidate_names=candidate_names,
                prior_strength=float(shapley_bayesian_prior_strength),
            )
            oracle_weights = apply_bayesian_shrinkage_to_weights(
                oracle_weights,
                shrinkage_lambda=float(shrinkage_meta.get("shrinkage_lambda", 0.0)),
            )
            if isinstance(shapley_meta, dict):
                shapley_meta = dict(shapley_meta)
                shapley_meta["bayesian_shrinkage"] = dict(shrinkage_meta)
        if bool(oracle_weight_js_shrinkage):
            oracle_weights = _mnpo_james_stein_shrinkage(
                oracle_weights,
                effective_n=oracle_effective_n,
            )
            if isinstance(shapley_meta, dict):
                shapley_meta = dict(shapley_meta)
                shapley_meta["oracle_weight_js_shrinkage"] = {
                    "applied": True,
                    "mode": "banzhaf",
                    "effective_n": float(oracle_effective_n),
                }
        oracle_weights_tritrust = fit_tritrust_weights(oracle_matrices)
    elif weighting_mode == "uniform":
        oracle_weights = {name: 1.0 for name in oracle_matrices}
        shapley_meta = {"applied": False, "reason": "uniform_mode"}
        oracle_weights_tritrust = fit_tritrust_weights(oracle_matrices)
    else:
        if use_tritrust:
            oracle_weights = fit_tritrust_weights(oracle_matrices)
            shapley_meta = {"applied": False, "reason": "tritrust_mode"}
            oracle_weights_tritrust = dict(oracle_weights)
        else:
            oracle_weights = {name: 1.0 for name in oracle_matrices}
            shapley_meta = {"applied": False, "reason": "tritrust_disabled"}
            oracle_weights_tritrust = fit_tritrust_weights(oracle_matrices)

    if bool(complexity_conditioning):
        complexity_meta = dict(oracle_components.get("complexity_conditioning", {}) or {})
        meta_features = dict(complexity_meta.get("meta_features", {}) or {})
        if meta_features:
            oracle_weights_before = {
                str(name): float(value) for name, value in dict(oracle_weights).items()
            }
            oracle_weights = adjust_oracle_weights_for_complexity(
                oracle_weights,
                meta_features,
            )
            complexity_meta["oracle_weights_before"] = dict(oracle_weights_before)
            complexity_meta["oracle_weights_after"] = {
                str(name): float(value) for name, value in dict(oracle_weights).items()
            }
            oracle_components["complexity_conditioning"] = dict(complexity_meta)

    oracle_redundancy_meta = None
    if use_oracle_redundancy_penalty:
        oracle_weights, oracle_redundancy_meta = _mnpo_apply_oracle_redundancy_penalty(
            dict(oracle_weights),
            oracle_scores,
        )

    payoff = aggregate_payoff_matrix(oracle_matrices, oracle_weights)
    payoff_shrinkage_meta = {
        "payoff_shrinkage_kappa": float(payoff_shrinkage_kappa),
        "payoff_shrinkage_applied": False,
    }
    if payoff_shrinkage_kappa > 0.0:
        payoff, payoff_shrinkage_meta = _mnpo_shrink_payoff_matrix(
            payoff,
            kappa=float(payoff_shrinkage_kappa),
        )
    perf_prior = normalize_vector_01(oracle_scores.get('performance', np.ones(len(candidate_names))))
    complexity_prior = normalize_vector_01(
        np.array([evaluation[name]['complexity'] for name in candidate_names], dtype=float)
    )
    reference_prior = 0.70 * perf_prior + 0.30 * complexity_prior + 1e-6
    reference_prior = reference_prior / np.sum(reference_prior)
    reference_prior, diakrino_selector_prior_meta = build_diakrino_selector_reference_prior(
        reference_prior,
        candidate_names,
        diakrino_selector_prior,
        enabled=bool(use_diakrino_selector_prior),
        blend_weight=float(diakrino_selector_prior_weight),
        calibration=str(diakrino_selector_prior_calibration),
        trust_record=diakrino_selector_prior_trust_record,
    )
    if bool(mnpo_paradigm_aware_prior_enabled):
        reference_prior, paradigm_prior_meta = apply_paradigm_aware_prior_floor(
            reference_prior,
            candidate_names,
            interaction_floor=float(mnpo_interaction_floor),
        )
    else:
        paradigm_prior_meta = {
            "enabled": False,
            "applied": False,
            "reason": "disabled",
            "interaction_floor": float(mnpo_interaction_floor),
            "interaction_mass_before": float("nan"),
            "interaction_mass_after": float("nan"),
            "interaction_candidates": [],
        }

    p_star, trajectory = mirror_descent_mnpo(
        payoff, reference_prior,
        mirror_descent_steps=mirror_descent_steps,
        mirror_descent_eta=mirror_descent_eta,
        mirror_descent_lambda=mirror_descent_lambda,
    )

    # --- T-VR-13: extract mirror-descent diagnostics ---
    _kl_trajectory: List[float] = []
    try:
        _kl_trajectory = list(getattr(trajectory, 'kl_values', []) or [])
    except Exception:
        pass
    # Weight trajectory: cap at last 20 iterations to avoid bloat.
    _weight_trajectory_cap = 20
    _weight_trajectory: List[List[float]] = []
    if len(trajectory) > _weight_trajectory_cap:
        _weight_trajectory = [
            [float(v) for v in w.ravel()] for w in trajectory[-_weight_trajectory_cap:]
        ]
    else:
        _weight_trajectory = [
            [float(v) for v in w.ravel()] for w in trajectory
        ]

    # --- T-VR-13: preference quantization stats (H2 hypothesis) ---
    _quantized_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    _quantization_stats: Dict[str, Any] = {}
    try:
        for oname, omat in oracle_matrices.items():
            mat = np.asarray(omat, dtype=float)
            n = mat.size
            if n > 0:
                counts = {}
                for level in _quantized_levels:
                    c = int(np.sum(np.abs(mat - level) < 1e-6))
                    counts[str(level)] = c
                total_quantized = sum(counts.values())
                _quantization_stats[oname] = {
                    "quantized_counts": counts,
                    "quantized_fraction": float(total_quantized) / float(n),
                    "total_entries": int(n),
                }
    except Exception:
        pass

    # T-R-243: apply selector penalties to mirror-descent weights.
    selector_penalty_applied = {}
    if selector_penalty_map:
        for idx, name in enumerate(candidate_names):
            if name in selector_penalty_map:
                penalty = float(selector_penalty_map[name])
                old_w = float(p_star[idx])
                p_star[idx] *= penalty
                selector_penalty_applied[name] = {
                    "penalty": penalty,
                    "weight_before": old_w,
                    "weight_after": float(p_star[idx]),
                }
        # Re-normalize to simplex after penalty application.
        total = float(np.sum(p_star))
        if total > 0:
            p_star = p_star / total
        else:
            p_star = np.full(len(p_star), 1.0 / max(1, len(p_star)))

    # T-R-271: stability-weighted portfolio aggregation.
    stability_weight_meta: Dict[str, Any] = {}
    if stability_weighted_aggregation_enabled:
        stability_scores = _compute_selector_stability_scores(
            candidates, racing_candidates,
        )
        any_non_trivial = any(s < 1.0 for s in stability_scores.values())
        if any_non_trivial:
            for idx, name in enumerate(candidate_names):
                stab = stability_scores.get(name, 1.0)
                old_w = float(p_star[idx])
                p_star[idx] *= stab
                stability_weight_meta[name] = {
                    "stability_score": stab,
                    "weight_before": old_w,
                    "weight_after": float(p_star[idx]),
                }
            # Re-normalize to simplex.
            total = float(np.sum(p_star))
            if total > 0:
                p_star = p_star / total
            else:
                p_star = np.full(len(p_star), 1.0 / max(1, len(p_star)))
        else:
            for name in candidate_names:
                stability_weight_meta[name] = {
                    "stability_score": stability_scores.get(name, 1.0),
                    "note": "all_trivial_no_reweight",
                }

    tremble_meta = None
    if compute_tremble_sensitivity and len(candidate_names) > 1:
        eps = 0.01
        trembled = _mnpo_tremble_oracle_matrices(oracle_matrices, epsilon=float(eps))
        payoff_eps = aggregate_payoff_matrix(trembled, oracle_weights)
        p_eps = _mnpo_mirror_descent_reference_regularized(
            np.asarray(payoff_eps, dtype=float),
            np.asarray(reference_prior, dtype=float),
            steps=int(max(1, int(mirror_descent_steps))),
            eta=float(mirror_descent_eta),
            lambda_=float(mirror_descent_lambda),
            tol_kl=1e-7,
            return_history=False,
        )
        p_eps = np.asarray(p_eps, dtype=float).ravel()
        diff = p_eps - np.asarray(p_star, dtype=float).ravel()
        abs_shift = np.abs(diff)
        tremble_meta = {
            "epsilon": float(eps),
            "l1": float(np.sum(abs_shift)),
            "l2": float(np.sqrt(np.sum(diff * diff))),
            "linf": float(np.max(abs_shift)) if abs_shift.size else 0.0,
            "max_shift": float(np.max(abs_shift)) if abs_shift.size else 0.0,
            "shift_by_candidate": {
                str(candidate_names[i]): float(abs_shift[i]) for i in range(min(len(candidate_names), abs_shift.size))
            },
        }

    consensus_excluded = []
    eligible_indices = list(range(len(candidate_names)))
    if mnpo_consensus_exclude_methods and (
        ("legacy_consensus" in racing_candidates) or ("majority_consensus" in racing_candidates)
    ):
        protect_idx = set()
        if mnpo_consensus_exclude_protect_top_k > 0:
            protect_idx = set(
                np.argsort(p_star)[::-1][: min(len(candidate_names), mnpo_consensus_exclude_protect_top_k)].tolist()
            )
        exclude_set = set(mnpo_consensus_exclude_methods)
        filtered = []
        for idx, name in enumerate(candidate_names):
            if (
                name in exclude_set
                and (not consensus_source_methods or name in consensus_source_methods)
                and idx not in protect_idx
            ):
                consensus_excluded.append(name)
                continue
            filtered.append(idx)
        if filtered:
            eligible_indices = filtered
        else:
            consensus_excluded = []

    if len(eligible_indices) != len(candidate_names):
        if bool(pareto_portfolio_sizing_enabled):
            effective_portfolio_size, adaptive_size_meta = resolve_pareto_portfolio_size(
                p_star[eligible_indices],
                portfolio_size=portfolio_size,
                n_available_methods=len(eligible_indices),
            )
        else:
            effective_portfolio_size, adaptive_size_meta = resolve_adaptive_portfolio_size(
                p_star[eligible_indices],
                portfolio_size=portfolio_size,
                adaptive_enabled=bool(adaptive_portfolio_sizing_enabled),
                adaptive_size_min=adaptive_size_min,
                adaptive_size_max=adaptive_size_max,
                adaptive_sizing_variance_penalty=bool(adaptive_sizing_variance_penalty),
                adaptive_sizing_variance_penalty_strength=float(adaptive_sizing_variance_penalty_strength),
            )
        eligible_names = [candidate_names[i] for i in eligible_indices]
        eligible_weights = np.asarray(p_star[eligible_indices], dtype=float)
        portfolio_rel = extract_portfolio(
            eligible_names, eligible_weights, evaluation,
            portfolio_size=effective_portfolio_size,
            use_diversity_oracle=use_diversity_oracle,
        )
        portfolio_indices = [eligible_indices[i] for i in portfolio_rel]
    else:
        if bool(pareto_portfolio_sizing_enabled):
            effective_portfolio_size, adaptive_size_meta = resolve_pareto_portfolio_size(
                p_star,
                portfolio_size=portfolio_size,
                n_available_methods=len(candidate_names),
            )
        else:
            effective_portfolio_size, adaptive_size_meta = resolve_adaptive_portfolio_size(
                p_star,
                portfolio_size=portfolio_size,
                adaptive_enabled=bool(adaptive_portfolio_sizing_enabled),
                adaptive_size_min=adaptive_size_min,
                adaptive_size_max=adaptive_size_max,
                adaptive_sizing_variance_penalty=bool(adaptive_sizing_variance_penalty),
                adaptive_sizing_variance_penalty_strength=float(adaptive_sizing_variance_penalty_strength),
            )
        portfolio_indices = extract_portfolio(
            candidate_names, p_star, evaluation,
            portfolio_size=effective_portfolio_size,
            use_diversity_oracle=use_diversity_oracle,
        )
    portfolio_weights_raw = p_star[portfolio_indices]
    if np.sum(portfolio_weights_raw) <= 0:
        portfolio_weights = np.full(len(portfolio_indices), 1.0 / max(1, len(portfolio_indices)))
    else:
        portfolio_weights = portfolio_weights_raw / np.sum(portfolio_weights_raw)

    for idx, weight in zip(portfolio_indices, portfolio_weights):
        racing_candidates[candidate_names[idx]]['portfolio_weight'] = float(weight)

    feature_votes, feature_details = mnpo_aggregate_feature_votes(racing_candidates, candidate_names, n_features)
    if np.max(feature_votes) <= 0:
        return None

    vote_ranking = np.argsort(feature_votes)[::-1]
    wrapper_refine_meta = {
        "wrapper_refine_enabled": bool(wrapper_refine_enabled),
        "wrapper_refine_applied": False,
    }
    if wrapper_refine_enabled:
        selected_indices_local, wrapper_refine_meta = apply_wrapper_refinement(
            X_uncorr,
            y,
            vote_ranking=vote_ranking,
            n_final_features=n_final_features,
            wrapper_refine_enabled=wrapper_refine_enabled,
            wrapper_refine_top_k=wrapper_refine_top_k,
            wrapper_refine_max_add=wrapper_refine_max_add,
            wrapper_refine_min_gain=wrapper_refine_min_gain,
            get_inner_cv_splits_fn=get_inner_cv_splits_fn,
            fit_and_score_fold_fn=fit_and_score_fold_fn,
        )
        if selected_indices_local.size == 0:
            selected_indices_local = np.asarray(vote_ranking[:n_final_features], dtype=int)
    else:
        selected_indices_local = np.asarray(vote_ranking[:n_final_features], dtype=int)
    rank_aggregation_candidate = next(
        (name for name in candidate_names if name.startswith("rank_aggregate_")),
        None,
    )

    if bool(rashomon_enabled):
        from .oracles import compute_rashomon_importance_bounds

        if 10 <= int(selected_indices_local.size) <= 50:
            rashomon_meta = compute_rashomon_importance_bounds(
                X_uncorr,
                y,
                selected_indices_local,
                random_state=int(random_state),
                max_models=int(max(2, rashomon_max_models)),
                score_tolerance=float(max(0.0, rashomon_score_tolerance)),
                cv_splits=3,
            )
        else:
            rashomon_meta = {
                "rashomon_enabled": True,
                "rashomon_computed": False,
                "rashomon_reason": "outside_k_range",
                "rashomon_n_features": int(selected_indices_local.size),
                "rashomon_n_models_total": 0,
                "rashomon_n_models_kept": 0,
                "rashomon_best_score": None,
                "rashomon_score_tolerance": float(max(0.0, rashomon_score_tolerance)),
                "importance_bounds": {},
            }
    else:
        rashomon_meta = {
            "rashomon_enabled": False,
            "rashomon_computed": False,
            "rashomon_reason": "disabled",
            "rashomon_n_features": int(selected_indices_local.size),
            "rashomon_n_models_total": 0,
            "rashomon_n_models_kept": 0,
            "rashomon_best_score": None,
            "rashomon_score_tolerance": float(max(0.0, rashomon_score_tolerance)),
            "importance_bounds": {},
        }

    method_result_summary = {
        'selected_indices': selected_indices_local,
        'scores': {int(i): float(feature_votes[i]) for i in selected_indices_local},
        'portfolio_candidates': [candidate_names[i] for i in portfolio_indices],
        'portfolio_size_requested': int(portfolio_size),
        'portfolio_size_effective': int(effective_portfolio_size),
        'adaptive_portfolio_sizing': dict(adaptive_size_meta),
        'portfolio_weights': {
            candidate_names[i]: float(w) for i, w in zip(portfolio_indices, portfolio_weights)
        },
        'oracle_weights': {k: float(v) for k, v in oracle_weights.items()},
        'oracle_weights_tritrust': {k: float(v) for k, v in oracle_weights_tritrust.items()},
        'oracle_weighting_mode': str(weighting_mode),
        'oracle_weighting_meta': dict(shapley_meta) if isinstance(shapley_meta, dict) else {},
        'oracle_weight_js_shrinkage': bool(oracle_weight_js_shrinkage and weighting_mode == "banzhaf"),
        'complexity_conditioning': bool(complexity_conditioning),
        'oracle_cap_meta': dict(oracle_cap_meta),
        'shapley_bayesian_shrinkage': bool(shapley_bayesian_shrinkage),
        'shapley_bayesian_prior_strength': float(shapley_bayesian_prior_strength),
        'payoff_shrinkage_meta': dict(payoff_shrinkage_meta),
        'use_interaction_oracle': bool(use_interaction_oracle),
        'interaction_oracle_min_n_train': int(interaction_oracle_min_n_train),
        'interaction_oracle_pool_size_cap': int(interaction_oracle_pool_size_cap),
        'interaction_oracle_pair_cap': int(interaction_oracle_pair_cap),
        'oracle_pairwise_meta': dict(oracle_pairwise_meta) if isinstance(oracle_pairwise_meta, dict) else {},
        'oracle_redundancy_meta': dict(oracle_redundancy_meta) if isinstance(oracle_redundancy_meta, dict) else {},
        'tremble_sensitivity': dict(tremble_meta) if isinstance(tremble_meta, dict) else {},
        'selector_penalty_applied': dict(selector_penalty_applied) if selector_penalty_applied else {},
        'stability_weight_meta': dict(stability_weight_meta) if stability_weight_meta else {},
        'diakrino_selector_prior': dict(diakrino_selector_prior_meta),
        'paradigm_aware_prior': dict(paradigm_prior_meta) if isinstance(paradigm_prior_meta, dict) else {},
        'rank_aggregation_mode': str(rank_aggregation_mode),
        'rank_aggregation_candidate': rank_aggregation_candidate,
        'rashomon_importance': dict(rashomon_meta),
        'mnpo_consensus_excluded_methods': list(consensus_excluded),
        'selector_candidate_statuses': {
            name: dict(status)
            for name, status in method_result_eligibility.items()
        },
        'mnpo_degraded_excluded_methods': list(degraded_excluded_methods),
        'mnpo_consensus_source_methods': sorted(consensus_source_methods),
        'wrapper_refinement': dict(wrapper_refine_meta),
        'runtime_racing_enabled': bool(racing_meta.get("runtime_racing_enabled", False)),
        'runtime_racing_applied': bool(racing_meta.get("runtime_racing_applied", False)),
        'runtime_racing_mode': str(racing_meta.get("runtime_racing_mode", runtime_racing_mode)),
        'runtime_racing_confidence_bound': str(
            racing_meta.get("runtime_racing_confidence_bound", runtime_racing_confidence_bound)
        ),
        'runtime_racing_delta': float(racing_meta.get("runtime_racing_delta", runtime_racing_delta)),
        'runtime_racing_stages': int(racing_meta.get("runtime_racing_stages", runtime_racing_stages)),
        'runtime_racing_initial_candidates': int(racing_meta.get("runtime_racing_initial_candidates", len(candidates))),
        'runtime_racing_kept_candidates': int(racing_meta.get("runtime_racing_kept_candidates", len(candidate_names))),
        'runtime_racing_kept_names': list(racing_meta.get("runtime_racing_kept_names", candidate_names)),
        'runtime_racing_dropped_names': list(racing_meta.get("runtime_racing_dropped_names", [])),
        'runtime_racing_proxy_scores': dict(racing_meta.get("runtime_racing_proxy_scores", {})),
        'runtime_racing_stage_history': list(racing_meta.get("runtime_racing_stage_history", [])),
        'adaptive_sizing_variance_penalty': bool(adaptive_sizing_variance_penalty),
        'adaptive_sizing_variance_penalty_strength': float(adaptive_sizing_variance_penalty_strength),
    }

    def _serialize_component_value(vals):
        if isinstance(vals, dict):
            return {
                str(k): _serialize_component_value(v)
                for k, v in vals.items()
            }
        if np.isscalar(vals):
            try:
                return float(vals)
            except Exception as exc:
                return vals
        arr = np.asarray(vals)
        if arr.dtype == object:
            return [_serialize_component_value(v) for v in arr.ravel().tolist()]
        return [float(v) for v in arr.ravel()]

    diagnostics = {
        'candidate_names': candidate_names,
        'candidate_weights': {name: float(p_star[i]) for i, name in enumerate(candidate_names)},
        'portfolio_candidates': method_result_summary['portfolio_candidates'],
        'portfolio_weights': method_result_summary['portfolio_weights'],
        'oracle_weights': method_result_summary['oracle_weights'],
        'oracle_weights_tritrust': method_result_summary['oracle_weights_tritrust'],
        'oracle_pairwise_meta': method_result_summary['oracle_pairwise_meta'],
        'oracle_cap_meta': method_result_summary['oracle_cap_meta'],
        'oracle_redundancy_meta': method_result_summary['oracle_redundancy_meta'],
        'payoff_shrinkage_meta': dict(payoff_shrinkage_meta),
        'tremble_sensitivity': method_result_summary['tremble_sensitivity'],
        'paradigm_aware_prior': method_result_summary['paradigm_aware_prior'],
        'diakrino_selector_prior': method_result_summary['diakrino_selector_prior'],
        'mnpo_consensus_excluded_methods': list(consensus_excluded),
        'selector_candidate_statuses': {
            name: dict(status)
            for name, status in method_result_eligibility.items()
        },
        'mnpo_degraded_excluded_methods': list(degraded_excluded_methods),
        'mnpo_consensus_source_methods': sorted(consensus_source_methods),
        'oracle_scores': {k: [float(v) for v in np.asarray(vals).ravel()] for k, vals in oracle_scores.items()},
        'oracle_components': {
            name: {
                comp: _serialize_component_value(vals)
                for comp, vals in components.items()
            }
            for name, components in oracle_components.items()
        },
        'trajectory_steps': len(trajectory),
        'rank_aggregation_mode': str(rank_aggregation_mode),
        'rank_aggregation_candidate': rank_aggregation_candidate,
        'wrapper_refinement': dict(wrapper_refine_meta),
        'runtime_racing': {
            "runtime_racing_enabled": bool(racing_meta.get("runtime_racing_enabled", False)),
            "runtime_racing_applied": bool(racing_meta.get("runtime_racing_applied", False)),
            "runtime_racing_mode": str(racing_meta.get("runtime_racing_mode", runtime_racing_mode)),
            "runtime_racing_confidence_bound": str(
                racing_meta.get("runtime_racing_confidence_bound", runtime_racing_confidence_bound)
            ),
            "runtime_racing_delta": float(racing_meta.get("runtime_racing_delta", runtime_racing_delta)),
            "runtime_racing_stages": int(racing_meta.get("runtime_racing_stages", runtime_racing_stages)),
            "runtime_racing_initial_candidates": int(
                racing_meta.get("runtime_racing_initial_candidates", len(candidates))
            ),
            "runtime_racing_kept_candidates": int(
                racing_meta.get("runtime_racing_kept_candidates", len(candidate_names))
            ),
            "runtime_racing_kept_names": list(racing_meta.get("runtime_racing_kept_names", candidate_names)),
            "runtime_racing_dropped_names": list(racing_meta.get("runtime_racing_dropped_names", [])),
            "runtime_racing_proxy_scores": dict(racing_meta.get("runtime_racing_proxy_scores", {})),
            "runtime_racing_stage_history": list(racing_meta.get("runtime_racing_stage_history", [])),
        },
        'evaluation_failures': {
            name: dict(evaluation.get(name, {}).get('evaluation_failures', {}))
            for name in candidate_names
        },
        # --- T-VR-13: new diagnostics ---
        'mirror_descent_kl_trajectory': _kl_trajectory,
        'mirror_descent_weight_trajectory': _weight_trajectory,
        'preference_quantization': _quantization_stats,
    }

    return selected_indices_local, feature_votes, feature_details, diagnostics, method_result_summary

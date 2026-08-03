"""Copula knockoff feature selection method."""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def copula_knockoff_selection(
    X, y, n_target_features, *,
    CopulaKnockoffSelectorClass,
    copula_knockoff_draws,
    copula_alpha_kn,
    copula_alpha_ebh,
    copula_truncation_level,
    copula_generator,
    copula_deepdrk_latent_fraction,
    copula_deepdrk_noise_scale,
    copula_derandomize_runs,
    copula_stabilizer_runs,
    copula_stabilizer_use_ebh,
    copula_stabilizer_seed_stride,
    random_state,
    copula_time_budget_seconds=None,
    copula_vine_num_threads=None,
    copula_lasso_max_iter=5000,
    copula_lasso_n_jobs=1,
):
    """
    Wrapper so the copula selector plugs into the voting scheme.
    """
    if CopulaKnockoffSelectorClass is None:
        return {}, {}
    X_arr = np.asarray(X)
    if X_arr.ndim != 2 or int(X_arr.shape[0]) < 2 or int(X_arr.shape[1]) == 0:
        return {}, {}

    def _ebh_support_local(e_values: np.ndarray, alpha: float) -> np.ndarray:
        e_vals = np.asarray(e_values, dtype=float).ravel()
        if e_vals.size == 0:
            return np.array([], dtype=int)
        e_vals = np.nan_to_num(e_vals, nan=0.0, posinf=0.0, neginf=0.0)
        p = int(e_vals.size)
        order = np.argsort(e_vals)[::-1]
        e_sorted = e_vals[order]
        thresh = (np.arange(1, p + 1, dtype=float) * float(alpha)) / float(p)
        thresh = np.maximum(thresh, 1e-12)
        valid = np.where(e_sorted >= (1.0 / thresh))[0]
        if valid.size == 0:
            return np.array([], dtype=int)
        k_hat = int(valid.max())
        return np.sort(order[: k_hat + 1])

    n_features = int(X.shape[1])
    derandomize_runs = int(max(1, copula_derandomize_runs))
    legacy_stabilizer_runs = int(max(1, copula_stabilizer_runs))
    derandomized_mode = bool(derandomize_runs > 1)
    n_runs = int(derandomize_runs if derandomized_mode else legacy_stabilizer_runs)
    weight_records = []
    e_value_records = []
    support_counts = np.zeros(n_features, dtype=float)
    single_run_support = np.array([], dtype=int)
    effective_truncation = {}
    low_info_reasons = {}
    low_info_nonzero_e_values = []
    low_info_support_sizes = []
    completed_outer_runs = 0
    outer_budget_exhausted = False

    # Slice the wall-clock budget across outer derandomize runs so a single
    # pathological run cannot exhaust the entire envelope.
    if copula_time_budget_seconds is None or float(copula_time_budget_seconds) <= 0.0:
        outer_deadline_end = None
        per_run_budget = None
    else:
        outer_deadline_end = time.monotonic() + float(copula_time_budget_seconds)
        per_run_budget = float(copula_time_budget_seconds) / float(max(1, n_runs))

    for run_idx in range(n_runs):
        if outer_deadline_end is not None and time.monotonic() >= outer_deadline_end:
            logger.warning(
                "Copula knockoff: outer time budget exhausted after %d/%d derandomize runs.",
                run_idx, n_runs,
            )
            outer_budget_exhausted = True
            break

        if outer_deadline_end is None:
            run_budget = None
        else:
            remaining = float(outer_deadline_end - time.monotonic())
            run_budget = float(max(1.0, min(per_run_budget or remaining, remaining)))

        run_seed = int(random_state + run_idx * copula_stabilizer_seed_stride)
        ck = CopulaKnockoffSelectorClass(
            M=copula_knockoff_draws,
            alpha_kn=copula_alpha_kn,
            alpha_ebh=copula_alpha_ebh,
            truncation_level=copula_truncation_level,
            generator=str(copula_generator or "copula"),
            deepdrk_latent_fraction=float(copula_deepdrk_latent_fraction),
            deepdrk_noise_scale=float(copula_deepdrk_noise_scale),
            show_progress=False,
            random_state=run_seed,
            time_budget_seconds=run_budget,
            vine_num_threads=copula_vine_num_threads,
            lasso_max_iter=int(copula_lasso_max_iter),
            lasso_n_jobs=int(copula_lasso_n_jobs),
        ).fit(X, y)
        completed_outer_runs += 1

        weights = np.asarray(ck.get_weights(), dtype=float).ravel()
        if weights.size != n_features:
            padded = np.zeros(n_features, dtype=float)
            upto = int(min(n_features, weights.size))
            padded[:upto] = weights[:upto]
            weights = padded
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weight_records.append(weights)
        e_vals = np.asarray(getattr(ck, "e_avg_", np.zeros(n_features, dtype=float)), dtype=float).ravel()
        if e_vals.size != n_features:
            padded_e = np.zeros(n_features, dtype=float)
            upto = int(min(n_features, e_vals.size))
            padded_e[:upto] = e_vals[:upto]
            e_vals = padded_e
        e_value_records.append(np.nan_to_num(e_vals, nan=0.0, posinf=0.0, neginf=0.0))

        support = np.array(
            sorted(set(int(i) for i in np.asarray(ck.get_support(), dtype=int).tolist() if 0 <= int(i) < n_features)),
            dtype=int,
        )
        if run_idx == 0:
            single_run_support = support
        if support.size > 0:
            support_counts[support] += 1.0
        if not effective_truncation:
            effective_truncation = dict(getattr(ck, "truncation_level_effective_", {}))
        low_info = dict(getattr(ck, "low_information_diagnostics_", {}) or {})
        reason_code = str(low_info.get("reason_code", "unknown"))
        low_info_reasons[reason_code] = int(low_info_reasons.get(reason_code, 0) + 1)
        low_info_nonzero_e_values.append(float(low_info.get("n_nonzero_e_values", 0.0)))
        low_info_support_sizes.append(float(low_info.get("n_support", float(support.size))))

    if not weight_records:
        # Outer budget exhausted before any inner run completed.
        weights = np.zeros(n_features, dtype=float)
    else:
        weights = np.mean(np.vstack(weight_records), axis=0)
    completed_denominator = float(max(1, completed_outer_runs or n_runs))
    support_frequency = support_counts / completed_denominator
    stabilizer_enabled = bool(n_runs > 1)
    stabilizer_e_values = support_frequency / max(float(copula_alpha_kn), 1e-8)
    derandomized_e_values = (
        np.mean(np.vstack(e_value_records), axis=0)
        if e_value_records
        else np.zeros(n_features, dtype=float)
    )

    if derandomized_mode:
        # T-R-110: derandomized knockoffs aggregate per-run e-values, then run e-BH.
        support = _ebh_support_local(derandomized_e_values, alpha=float(copula_alpha_ebh))
    elif copula_stabilizer_use_ebh:
        support = _ebh_support_local(stabilizer_e_values, alpha=float(copula_alpha_ebh))
    elif n_runs == 1:
        support = np.asarray(single_run_support, dtype=int)
    else:
        support = np.where(support_frequency >= 0.5)[0]
        support = np.asarray(support, dtype=int)

    stabilizer_fallback_used = False
    if support.size == 0:
        support = np.argsort(weights)[::-1][: int(max(1, n_target_features))]
        support = np.asarray(support, dtype=int)
        stabilizer_fallback_used = True

    # pick the top-k as "selected" to stay consistent
    if support.size > n_target_features:
        ranked_support = support[np.argsort(weights[support])[::-1]]
        support = ranked_support[:n_target_features]

    results = {
        "selected_indices": support,
        "scores": {idx: weights[idx] for idx in support},
        "method": "copula_knockoff",
        "copula_knockoff_draws": int(copula_knockoff_draws),
        "copula_alpha_kn": float(copula_alpha_kn),
        "copula_alpha_ebh": float(copula_alpha_ebh),
        "copula_truncation_level": (
            None if copula_truncation_level is None else int(copula_truncation_level)
        ),
        "copula_generator": str(copula_generator or "copula"),
        "copula_deepdrk_latent_fraction": float(copula_deepdrk_latent_fraction),
        "copula_deepdrk_noise_scale": float(copula_deepdrk_noise_scale),
        "copula_effective_truncation_level": dict(effective_truncation),
        "copula_derandomized_mode": bool(derandomized_mode),
        "copula_derandomize_runs": int(derandomize_runs),
        "copula_derandomized_e_values": np.asarray(derandomized_e_values, dtype=float),
        "copula_stabilizer_enabled": bool(stabilizer_enabled),
        "copula_stabilizer_runs": int(n_runs),
        "copula_stabilizer_legacy_runs": int(legacy_stabilizer_runs),
        "copula_stabilizer_use_ebh": bool(copula_stabilizer_use_ebh),
        "copula_stabilizer_seed_stride": int(copula_stabilizer_seed_stride),
        "copula_stabilizer_support_frequency": np.asarray(support_frequency, dtype=float),
        "copula_stabilizer_e_values": np.asarray(stabilizer_e_values, dtype=float),
        "copula_stabilizer_fallback_used": bool(stabilizer_fallback_used),
        "copula_low_information": {
            "reason_counts": {str(k): int(v) for k, v in low_info_reasons.items()},
            "mean_nonzero_e_values": float(np.mean(low_info_nonzero_e_values)) if low_info_nonzero_e_values else 0.0,
            "mean_support_size_before_stabilizer": float(np.mean(low_info_support_sizes)) if low_info_support_sizes else 0.0,
            "empty_support_pre_stabilizer_runs": int(
                np.sum(np.asarray(low_info_support_sizes, dtype=float) <= 0.0)
            ) if low_info_support_sizes else 0,
            "fallback_used": bool(stabilizer_fallback_used),
        },
        "copula_time_budget_seconds": (
            None
            if copula_time_budget_seconds is None or float(copula_time_budget_seconds) <= 0.0
            else float(copula_time_budget_seconds)
        ),
        "copula_completed_outer_runs": int(completed_outer_runs),
        "copula_outer_budget_exhausted": bool(outer_budget_exhausted),
    }
    all_scores = {i: weights[i] for i in range(X.shape[1])}
    return results, all_scores

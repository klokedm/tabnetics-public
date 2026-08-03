"""Gate computation utilities (reporting-layer only).

This module is intentionally dependency-light (stdlib + numpy) so that gate logic
can be unit-tested without pulling the full ML stack into the reporting layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


Tier = str  # "easy" | "medium" | "hard" | "very_hard" | "unknown"


@dataclass(frozen=True)
class GateConfig:
    """Promotion gate configuration for reporting.

    Notes:
    - `mode="strict"` is the primary promotion gate definition.
    - `mode in {"quantile","bootstrap"}` are reporting-only until validated.
    """

    mode: str = "strict"  # "strict" | "quantile" | "bootstrap"
    easy_threshold: float = -0.02
    medium_threshold: float = -0.05
    hard_threshold: float = 0.01
    easy_trim: int = 0
    medium_trim: int = 0

    # Catastrophic veto is currently advisory-mode only; the other
    # anti-false-positive constraints apply to all gate modes.
    catastrophic_veto_threshold: float = -0.10
    overall_threshold: float = 0.01
    # Practical-significance floor on the observed primary BA delta.  This is
    # separate from tier safety thresholds so statistically clear but negligible
    # wins do not pass promotion gates.
    min_practical_effect: float = 0.01
    min_effect_dz: float = 0.0

    # Optional profile-level secondary metric no-regression check.  Pass paired
    # candidate-minus-baseline secondary deltas to compute_gate; default-off.
    secondary_regression_tolerance: float | None = None

    # Bootstrap gate controls (reporting-only).
    alpha: float = 0.05
    n_bootstrap: int = 5000
    n_permutations: int = 5000
    random_seed: int = 0

    # Reliability filter — exclude datasets with baseline seed std above threshold.
    # Set to 0.0 (default) to disable filtering.  Recommended: 0.12 for HDLSS.
    reliability_max_seed_std: float = 0.0


@dataclass(frozen=True)
class GateResult:
    verdict: str  # "PASS" | "FAIL"
    overall_mean: float
    hard_mean: float
    very_hard_mean: float
    worst_easy: float
    worst_medium: float
    catastrophic_min: float
    trimmed_easy: Tuple[Tuple[str, float], ...]
    trimmed_medium: Tuple[Tuple[str, float], ...]
    vetoed: bool
    reliability_excluded: Tuple[str, ...] = ()
    effect_floor_passed: bool = True
    effect_size_dz: float = float("nan")
    secondary_regression_passed: bool = True
    secondary_mean_delta: float = float("nan")
    secondary_worst_delta: float = float("nan")
    secondary_regression_tolerance: float | None = None


@dataclass(frozen=True)
class BootstrapGateResult:
    verdict: str  # "PASS" | "FAIL"
    ci_overall_mean: Tuple[float, float]
    ci_hard_mean: Tuple[float, float]
    ci_worst_easy: Tuple[float, float]
    ci_worst_medium: Tuple[float, float]
    p_overall_mean: float
    p_hard_mean: float
    p_worst_easy: float
    p_worst_medium: float
    vetoed: bool
    reliability_excluded: Tuple[str, ...] = ()
    effect_floor_passed: bool = True
    secondary_regression_passed: bool = True
    secondary_mean_delta: float = float("nan")
    secondary_worst_delta: float = float("nan")
    secondary_regression_tolerance: float | None = None


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.mean(np.asarray(xs, dtype=float)))


def _min(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.min(np.asarray(xs, dtype=float)))


def _cohen_dz(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float("inf") if float(arr[0]) > 0.0 else (float("-inf") if float(arr[0]) < 0.0 else 0.0)
    sd = float(np.std(arr, ddof=1))
    if sd <= 1e-12:
        mean = float(np.mean(arr))
        return float("inf") if mean > 0.0 else (float("-inf") if mean < 0.0 else 0.0)
    return float(np.mean(arr) / sd)


def _tier_values(
    dataset_mean_delta: Mapping[str, float],
    tier_by_dataset: Mapping[str, Tier],
) -> Dict[Tier, List[Tuple[str, float]]]:
    out: Dict[Tier, List[Tuple[str, float]]] = {}
    for ds_id, delta in dataset_mean_delta.items():
        tier = str(tier_by_dataset.get(str(ds_id), "unknown"))
        out.setdefault(tier, []).append((str(ds_id), float(delta)))
    return out


def _trim_worst(values: Sequence[Tuple[str, float]], k: int) -> Tuple[Tuple[Tuple[str, float], ...], List[Tuple[str, float]]]:
    if k <= 0:
        return tuple(), list(values)
    # Clamp k so at least one value is kept (FIX: T-A3-FIX-002).
    k = min(k, max(0, len(values) - 1))
    if k == 0:
        return tuple(), list(values)
    ordered = sorted(values, key=lambda t: (t[1], t[0]))
    trimmed = tuple(ordered[:k])
    kept = ordered[k:]
    return trimmed, kept


def _apply_reliability_filter(
    deltas_by_dataset: Mapping[str, Sequence[float]],
    config: GateConfig,
    baseline_seed_std_by_dataset: Mapping[str, float] | None,
) -> Tuple[Dict[str, Sequence[float]], Tuple[str, ...]]:
    """Filter out datasets whose baseline seed std exceeds *config.reliability_max_seed_std*.

    Returns ``(filtered_deltas, excluded_dataset_ids)``.  When the filter is
    disabled (threshold <= 0 or no std data provided), returns all data with no
    exclusions — fully backward-compatible.
    """
    if config.reliability_max_seed_std <= 0 or not baseline_seed_std_by_dataset:
        return dict(deltas_by_dataset), ()
    excluded: list[str] = []
    filtered: Dict[str, Sequence[float]] = {}
    for ds_id, seq in deltas_by_dataset.items():
        std_val = baseline_seed_std_by_dataset.get(str(ds_id))
        if std_val is not None and std_val > config.reliability_max_seed_std:
            excluded.append(str(ds_id))
        else:
            filtered[str(ds_id)] = seq
    return filtered, tuple(sorted(excluded))


def _effect_floor_passed(overall_mean: float, hard_mean: float, effect_size_dz: float, config: GateConfig) -> bool:
    min_effect = float(config.min_practical_effect)
    min_dz = float(config.min_effect_dz)
    ok_effect = True
    if min_effect > 0.0:
        ok_effect = (
            (np.isnan(overall_mean) or overall_mean >= min_effect)
            and (np.isnan(hard_mean) or hard_mean >= min_effect)
        )
    ok_dz = True
    if min_dz > 0.0:
        ok_dz = bool(np.isfinite(effect_size_dz) and effect_size_dz >= min_dz)
    return bool(ok_effect and ok_dz)


def _secondary_summary(
    *,
    secondary_deltas_by_dataset: Mapping[str, Sequence[float]] | None,
    tolerance: float | None,
    allowed_dataset_ids: Iterable[str],
) -> tuple[bool, float, float]:
    if tolerance is None:
        return True, float("nan"), float("nan")
    if secondary_deltas_by_dataset is None:
        return False, float("nan"), float("nan")
    allowed = {str(item) for item in allowed_dataset_ids}
    values: list[float] = []
    for ds_id, seq in secondary_deltas_by_dataset.items():
        if str(ds_id) not in allowed:
            continue
        arr = np.asarray(list(seq), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        values.append(float(np.mean(arr)))
    if not values:
        return False, float("nan"), float("nan")
    mean_delta = float(np.mean(np.asarray(values, dtype=float)))
    worst_delta = float(np.min(np.asarray(values, dtype=float)))
    return bool(worst_delta >= -float(tolerance)), mean_delta, worst_delta


def compute_gate(
    *,
    deltas_by_dataset: Mapping[str, Sequence[float]],
    tier_by_dataset: Mapping[str, Tier],
    config: GateConfig,
    baseline_seed_std_by_dataset: Mapping[str, float] | None = None,
    secondary_deltas_by_dataset: Mapping[str, Sequence[float]] | None = None,
) -> GateResult:
    """Compute strict/quantile gate verdict and diagnostics.

    Input deltas are per-dataset sequences (typically per-seed paired deltas).
    Tier aggregation uses per-dataset mean deltas.
    Catastrophic veto uses the minimum per-seed delta across all observations.
    """

    filtered_deltas, reliability_excluded = _apply_reliability_filter(
        deltas_by_dataset, config, baseline_seed_std_by_dataset,
    )

    dataset_mean_delta: Dict[str, float] = {}
    catastrophic_min = float("nan")
    for ds_id, seq in filtered_deltas.items():
        arr = np.asarray(list(seq), dtype=float)
        if arr.size == 0:
            continue
        dataset_mean_delta[str(ds_id)] = float(np.mean(arr))
        v = float(np.min(arr))
        catastrophic_min = v if np.isnan(catastrophic_min) else float(min(catastrophic_min, v))

    tiers = _tier_values(dataset_mean_delta, tier_by_dataset)
    all_means = [v for _, v in tiers.get("easy", []) + tiers.get("medium", []) + tiers.get("hard", []) + tiers.get("very_hard", [])]
    overall_mean = _mean(all_means)
    hard_mean = _mean([v for _, v in tiers.get("hard", [])])
    very_hard_mean = _mean([v for _, v in tiers.get("very_hard", [])])
    effect_size_dz = _cohen_dz(list(dataset_mean_delta.values()))

    easy_vals = tiers.get("easy", [])
    med_vals = tiers.get("medium", [])
    worst_easy = _min([v for _, v in easy_vals])
    worst_medium = _min([v for _, v in med_vals])

    trimmed_easy: Tuple[Tuple[str, float], ...] = tuple()
    trimmed_medium: Tuple[Tuple[str, float], ...] = tuple()

    vetoed = False
    if config.mode in ("quantile", "bootstrap"):
        if not np.isnan(catastrophic_min) and catastrophic_min < float(config.catastrophic_veto_threshold):
            vetoed = True

    # Quantile trimming affects only the easy/medium worst-case checks.
    easy_kept = easy_vals
    med_kept = med_vals
    if config.mode == "quantile":
        trimmed_easy, easy_kept = _trim_worst(easy_vals, int(config.easy_trim))
        trimmed_medium, med_kept = _trim_worst(med_vals, int(config.medium_trim))
        worst_easy = _min([v for _, v in easy_kept]) if easy_kept else float("nan")
        worst_medium = _min([v for _, v in med_kept]) if med_kept else float("nan")

    # Gate conditions.
    # NaN means the tier has no datasets — treat as vacuously satisfied (FIX: T-A3-FIX-001).
    ok_hard = np.isnan(hard_mean) or (hard_mean >= float(config.hard_threshold))
    ok_easy = np.isnan(worst_easy) or (worst_easy >= float(config.easy_threshold))
    ok_med = np.isnan(worst_medium) or (worst_medium >= float(config.medium_threshold))

    ok_overall = True
    if config.mode in ("strict", "quantile", "bootstrap"):
        ok_overall = np.isnan(overall_mean) or (overall_mean >= float(config.overall_threshold))
    effect_floor_ok = _effect_floor_passed(overall_mean, hard_mean, effect_size_dz, config)
    secondary_ok, secondary_mean, secondary_worst = _secondary_summary(
        secondary_deltas_by_dataset=secondary_deltas_by_dataset,
        tolerance=config.secondary_regression_tolerance,
        allowed_dataset_ids=filtered_deltas.keys(),
    )

    verdict = "PASS" if (
        ok_hard
        and ok_easy
        and ok_med
        and ok_overall
        and effect_floor_ok
        and secondary_ok
        and (not vetoed)
    ) else "FAIL"
    return GateResult(
        verdict=verdict,
        overall_mean=float(overall_mean) if not np.isnan(overall_mean) else float("nan"),
        hard_mean=float(hard_mean) if not np.isnan(hard_mean) else float("nan"),
        very_hard_mean=float(very_hard_mean) if not np.isnan(very_hard_mean) else float("nan"),
        worst_easy=float(worst_easy) if not np.isnan(worst_easy) else float("nan"),
        worst_medium=float(worst_medium) if not np.isnan(worst_medium) else float("nan"),
        catastrophic_min=float(catastrophic_min) if not np.isnan(catastrophic_min) else float("nan"),
        trimmed_easy=trimmed_easy,
        trimmed_medium=trimmed_medium,
        vetoed=bool(vetoed),
        reliability_excluded=reliability_excluded,
        effect_floor_passed=bool(effect_floor_ok),
        effect_size_dz=float(effect_size_dz) if np.isfinite(effect_size_dz) else float(effect_size_dz),
        secondary_regression_passed=bool(secondary_ok),
        secondary_mean_delta=float(secondary_mean) if np.isfinite(secondary_mean) else float("nan"),
        secondary_worst_delta=float(secondary_worst) if np.isfinite(secondary_worst) else float("nan"),
        secondary_regression_tolerance=config.secondary_regression_tolerance,
    )


def compute_bootstrap_gate(
    *,
    deltas_by_dataset: Mapping[str, Sequence[float]],
    tier_by_dataset: Mapping[str, Tier],
    config: GateConfig,
    baseline_seed_std_by_dataset: Mapping[str, float] | None = None,
    secondary_deltas_by_dataset: Mapping[str, Sequence[float]] | None = None,
) -> BootstrapGateResult:
    """Bootstrap + permutation diagnostics for the gate metrics.

    - Bootstrap: resample datasets (per tier) with replacement to form CI bands.
    - Permutation: within-dataset sign-flip at the per-seed delta level (paired under a symmetric null)
      to estimate one-sided p-values for the reported gate metrics.

    Verdict uses CI lower bounds vs thresholds + catastrophic veto + overall threshold.
    """

    filtered_deltas, reliability_excluded = _apply_reliability_filter(
        deltas_by_dataset, config, baseline_seed_std_by_dataset,
    )

    strict_like = GateConfig(
        mode="bootstrap",
        easy_threshold=config.easy_threshold,
        medium_threshold=config.medium_threshold,
        hard_threshold=config.hard_threshold,
        easy_trim=0,
        medium_trim=0,
        catastrophic_veto_threshold=config.catastrophic_veto_threshold,
        overall_threshold=config.overall_threshold,
        min_practical_effect=config.min_practical_effect,
        min_effect_dz=config.min_effect_dz,
        secondary_regression_tolerance=config.secondary_regression_tolerance,
        alpha=config.alpha,
        n_bootstrap=config.n_bootstrap,
        n_permutations=config.n_permutations,
        random_seed=config.random_seed,
        reliability_max_seed_std=config.reliability_max_seed_std,
    )

    base = compute_gate(
        deltas_by_dataset=filtered_deltas,
        tier_by_dataset=tier_by_dataset,
        config=strict_like,
        secondary_deltas_by_dataset=secondary_deltas_by_dataset,
    )

    # Extract per-dataset mean deltas by tier.
    dataset_mean_delta: Dict[str, float] = {}
    for ds_id, seq in filtered_deltas.items():
        vals = list(seq)
        if not vals:
            continue
        dataset_mean_delta[str(ds_id)] = float(np.mean(np.asarray(vals, dtype=float)))
    tiers = _tier_values(dataset_mean_delta, tier_by_dataset)
    rng = np.random.default_rng(int(config.random_seed))
    B = int(max(100, config.n_bootstrap))
    P = int(max(200, config.n_permutations))
    alpha = float(config.alpha)

    def _bootstrap_ci(values: Sequence[float], *, stat: str) -> Tuple[float, float]:
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            return (float("nan"), float("nan"))
        n = int(arr.size)
        idx = rng.integers(0, n, size=(B, n))
        samples = arr[idx]
        if stat == "mean":
            stats = np.mean(samples, axis=1)
        elif stat == "min":
            stats = np.min(samples, axis=1)
        else:
            raise ValueError(f"Unknown stat: {stat}")
        lo = float(np.quantile(stats, alpha / 2.0))
        hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
        return (lo, hi)

    # Permutation p-values (within-dataset, per-seed sign-flips).
    deltas_arr_by_dataset: Dict[str, np.ndarray] = {}
    for ds_id, seq in filtered_deltas.items():
        arr = np.asarray(list(seq), dtype=float)
        if arr.size:
            deltas_arr_by_dataset[str(ds_id)] = arr

    all_ids = [ds_id for ds_id, _ in tiers.get("easy", []) + tiers.get("medium", []) + tiers.get("hard", []) + tiers.get("very_hard", [])]
    easy_ids = [ds_id for ds_id, _ in tiers.get("easy", [])]
    med_ids = [ds_id for ds_id, _ in tiers.get("medium", [])]
    hard_ids = [ds_id for ds_id, _ in tiers.get("hard", [])]

    all_means = [float(dataset_mean_delta[str(ds_id)]) for ds_id in all_ids if str(ds_id) in dataset_mean_delta]
    hard_means = [float(dataset_mean_delta[str(ds_id)]) for ds_id in hard_ids if str(ds_id) in dataset_mean_delta]
    easy_means = [float(dataset_mean_delta[str(ds_id)]) for ds_id in easy_ids if str(ds_id) in dataset_mean_delta]
    med_means = [float(dataset_mean_delta[str(ds_id)]) for ds_id in med_ids if str(ds_id) in dataset_mean_delta]

    ci_overall = _bootstrap_ci(all_means, stat="mean")
    ci_hard = _bootstrap_ci(hard_means, stat="mean")
    ci_easy = _bootstrap_ci(easy_means, stat="min")
    ci_med = _bootstrap_ci(med_means, stat="min")

    p_overall = p_hard = p_easy = p_med = float("nan")
    if all_ids:
        id_to_idx = {str(ds_id): i for i, ds_id in enumerate(all_ids)}
        easy_idx = np.asarray([id_to_idx[str(ds_id)] for ds_id in easy_ids if str(ds_id) in id_to_idx], dtype=int)
        med_idx = np.asarray([id_to_idx[str(ds_id)] for ds_id in med_ids if str(ds_id) in id_to_idx], dtype=int)
        hard_idx = np.asarray([id_to_idx[str(ds_id)] for ds_id in hard_ids if str(ds_id) in id_to_idx], dtype=int)

        perm_overall = np.empty(P, dtype=float)
        perm_hard = np.empty(P, dtype=float)
        perm_worst_easy = np.empty(P, dtype=float)
        perm_worst_med = np.empty(P, dtype=float)

        for i in range(P):
            # Per-dataset permuted mean delta (per-seed sign flips).
            perm_means = np.empty(len(all_ids), dtype=float)
            for ds_id, j in id_to_idx.items():
                arr = deltas_arr_by_dataset.get(str(ds_id))
                if arr is None or arr.size == 0:
                    perm_means[j] = 0.0
                    continue
                signs = rng.choice(np.asarray([-1.0, 1.0]), size=arr.size)
                perm_means[j] = float(np.mean(arr * signs))

            perm_overall[i] = float(np.mean(perm_means))
            perm_hard[i] = float(np.mean(perm_means[hard_idx])) if hard_idx.size else float("nan")
            perm_worst_easy[i] = float(np.min(perm_means[easy_idx])) if easy_idx.size else float("nan")
            perm_worst_med[i] = float(np.min(perm_means[med_idx])) if med_idx.size else float("nan")

        # Larger is better for all tracked gate metrics.
        p_overall = float(np.mean(perm_overall >= float(base.overall_mean))) if not np.isnan(base.overall_mean) else float("nan")
        p_hard = float(np.mean(perm_hard >= float(base.hard_mean))) if not np.isnan(base.hard_mean) else float("nan")
        p_easy = float(np.mean(perm_worst_easy >= float(base.worst_easy))) if not np.isnan(base.worst_easy) else float("nan")
        p_med = float(np.mean(perm_worst_med >= float(base.worst_medium))) if not np.isnan(base.worst_medium) else float("nan")

    # NaN CI means the tier has no datasets — treat as vacuously satisfied (FIX: T-A3-FIX-001).
    ok_hard = np.isnan(ci_hard[0]) or (ci_hard[0] >= float(config.hard_threshold))
    ok_easy = np.isnan(ci_easy[0]) or (ci_easy[0] >= float(config.easy_threshold))
    ok_med = np.isnan(ci_med[0]) or (ci_med[0] >= float(config.medium_threshold))
    ok_overall = np.isnan(ci_overall[0]) or (ci_overall[0] >= float(config.overall_threshold))
    effect_floor_ok = bool(base.effect_floor_passed)
    if float(config.min_practical_effect) > 0.0:
        effect_floor_ok = bool(
            effect_floor_ok
            and (np.isnan(ci_overall[0]) or ci_overall[0] >= float(config.min_practical_effect))
            and (np.isnan(ci_hard[0]) or ci_hard[0] >= float(config.min_practical_effect))
        )
    verdict = "PASS" if (
        ok_hard
        and ok_easy
        and ok_med
        and ok_overall
        and effect_floor_ok
        and bool(base.secondary_regression_passed)
        and (not base.vetoed)
    ) else "FAIL"

    return BootstrapGateResult(
        verdict=verdict,
        ci_overall_mean=ci_overall,
        ci_hard_mean=ci_hard,
        ci_worst_easy=ci_easy,
        ci_worst_medium=ci_med,
        p_overall_mean=p_overall,
        p_hard_mean=p_hard,
        p_worst_easy=p_easy,
        p_worst_medium=p_med,
        vetoed=bool(base.vetoed),
        reliability_excluded=reliability_excluded,
        effect_floor_passed=bool(effect_floor_ok),
        secondary_regression_passed=bool(base.secondary_regression_passed),
        secondary_mean_delta=float(base.secondary_mean_delta)
        if np.isfinite(base.secondary_mean_delta)
        else float("nan"),
        secondary_worst_delta=float(base.secondary_worst_delta)
        if np.isfinite(base.secondary_worst_delta)
        else float("nan"),
        secondary_regression_tolerance=config.secondary_regression_tolerance,
    )

"""Tests for DF diagnostics correctness and CRPS UQ decomposition properties (T-009).

Validates that:
1. Diagnostic flags (is_near_constant, is_integer_like, too_few_unique,
   zero_inflated, has_heaping) trigger correctly on synthetic data.
2. CRPS UQ decomposition is numerically stable and correctly signed.
3. Results are deterministic with fixed seeds and stable across seeds.

Uses 6 synthetic distribution families:
  normal, skew (lognormal), heavy-tail (t df=2), zero-inflated,
  heaped/rounded, integer-like (Poisson).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from tabnetics.pipeline.pipeline import (
    DataAuditReport,
    DistributionFitter,
    DistributionFitterConfig,
    DistributionFitSummary,
    SupportProfile,
    _crps_uq_decompose_gaussian_ensemble,
)


# ---------------------------------------------------------------------------
# Synthetic data generators (deterministic via fixed seed)
# ---------------------------------------------------------------------------


def _gen_normal(seed: int = 42, n: int = 300) -> np.ndarray:
    """Clean N(0, 1) — no diagnostic flags expected."""
    return np.random.default_rng(seed).normal(loc=0.0, scale=1.0, size=n).astype(float)


def _gen_skew(seed: int = 42, n: int = 300) -> np.ndarray:
    """Lognormal — right-skewed, positive support."""
    return np.random.default_rng(seed).lognormal(mean=0.0, sigma=1.0, size=n).astype(float)


def _gen_heavy_tail(seed: int = 42, n: int = 300) -> np.ndarray:
    """Student-t with df=2 — heavy tails, real support."""
    rng = np.random.default_rng(seed)
    return (rng.standard_t(df=2, size=n)).astype(float)


def _gen_zero_inflated(seed: int = 42, n: int = 300) -> np.ndarray:
    """30% zeros + Gamma(2,1) — should trigger zero_inflated flag."""
    rng = np.random.default_rng(seed)
    n_zero = int(n * 0.30)
    n_cont = n - n_zero
    return np.concatenate([np.zeros(n_zero), rng.gamma(shape=2.0, scale=1.0, size=n_cont)]).astype(float)


def _gen_heaped(seed: int = 42, n: int = 300) -> np.ndarray:
    """Normal rounded to nearest 5 — should trigger has_heaping flag."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(loc=50.0, scale=15.0, size=n)
    return (np.round(raw / 5.0) * 5.0).astype(float)


def _gen_integer_like(seed: int = 42, n: int = 300) -> np.ndarray:
    """Poisson(lam=3) — should trigger is_integer_like flag."""
    rng = np.random.default_rng(seed)
    return rng.poisson(lam=3, size=n).astype(float)


def _gen_near_constant(seed: int = 42, n: int = 300) -> np.ndarray:
    """Near-constant: all values within 1e-12."""
    return np.full(n, 7.0, dtype=float)


def _gen_too_few_unique(seed: int = 42, n: int = 300) -> np.ndarray:
    """Only 3 distinct values among >=50 samples — should trigger too_few_unique."""
    rng = np.random.default_rng(seed)
    return rng.choice([1.0, 2.0, 3.0], size=n).astype(float)


# ---------------------------------------------------------------------------
# Shared fitter factory
# ---------------------------------------------------------------------------

def _make_fitter(**overrides) -> DistributionFitter:
    defaults = dict(
        robust_mode=True,
        use_adaptive_strategy=True,
        use_lrt=False,
        use_cv=False,
        use_support_filtering=True,
        random_state=42,
    )
    defaults.update(overrides)
    return DistributionFitter(DistributionFitterConfig(**defaults))


# ===================================================================
# Group 1: Diagnostic Flag Tests — 6 distribution families
# ===================================================================


class TestDiagnosticFlagsNormal:
    """Clean N(0,1) data — no diagnostic flags should trigger."""

    def test_no_heaping(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_normal())
        assert audit.has_heaping is False or audit.has_heaping == False  # noqa: E712

    def test_no_zero_inflated(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_normal())
        assert bool(audit.zero_inflated) is False

    def test_no_integer_like(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_normal())
        assert bool(audit.is_integer_like) is False

    def test_not_near_constant(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_normal())
        assert bool(audit.support.is_near_constant) is False

    def test_not_too_few_unique(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_normal())
        assert bool(audit.too_few_unique) is False

    def test_positive_clean_count(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_normal())
        assert int(audit.n_clean) == 300


class TestDiagnosticFlagsSkew:
    """Right-skewed lognormal data — positive support expected."""

    def test_positive_support(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_skew())
        assert audit.support.inferred_support == "positive"

    def test_min_positive(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_skew())
        assert float(audit.support.min_value) > 0.0

    def test_not_near_constant(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_skew())
        assert bool(audit.support.is_near_constant) is False


class TestDiagnosticFlagsHeavyTail:
    """Student-t(df=2) — heavy tails, real support, large outlier fraction."""

    def test_real_support(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_heavy_tail())
        assert audit.support.inferred_support == "real"

    def test_nontrivial_outlier_fraction(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_heavy_tail())
        # t(2) should have detectable outliers under 3-IQR rule
        assert float(audit.outlier_fraction) > 0.0

    def test_not_near_constant(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_heavy_tail())
        assert bool(audit.support.is_near_constant) is False


class TestDiagnosticFlagsZeroInflated:
    """30% zeros + Gamma — zero_inflated flag must trigger."""

    def test_zero_inflated_flag(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_zero_inflated())
        assert bool(audit.zero_inflated) is True

    def test_frac_zero_above_threshold(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_zero_inflated())
        assert float(audit.support.frac_zero) >= 0.10

    def test_positive_support(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_zero_inflated())
        assert audit.support.inferred_support == "positive"


class TestDiagnosticFlagsHeaped:
    """Normal rounded to nearest 5 — has_heaping must trigger."""

    def test_has_heaping_flag(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_heaped())
        assert bool(audit.has_heaping) is True

    def test_heaping_delta_positive(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_heaped())
        assert audit.heaping_delta is not None
        assert float(audit.heaping_delta) > 0.0

    def test_not_near_constant(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_heaped())
        assert bool(audit.support.is_near_constant) is False


class TestDiagnosticFlagsIntegerLike:
    """Poisson(lam=3) — is_integer_like must trigger."""

    def test_is_integer_like_flag(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_integer_like())
        assert bool(audit.is_integer_like) is True

    def test_n_unique_bounded(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_integer_like())
        # Poisson(3) typically has <= 12 distinct values among 300 samples
        assert int(audit.n_unique) <= max(12, int(0.10 * 300))

    def test_positive_support(self):
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_integer_like())
        assert audit.support.inferred_support == "positive"


# ===================================================================
# Group 2: CRPS UQ Decomposition Properties
# ===================================================================


class TestCRPSUQDecomposition:
    """Validate the Gaussian-ensemble CRPS UQ decomposition function."""

    def test_identical_ensemble_epistemic_zero(self):
        """Identical members → epistemic ≈ 0."""
        total, alea, epi = _crps_uq_decompose_gaussian_ensemble(
            means=[0.0, 0.0, 0.0],
            stds=[1.0, 1.0, 1.0],
            weights=[1.0 / 3, 1.0 / 3, 1.0 / 3],
        )
        assert np.isfinite(total)
        assert np.isfinite(alea)
        assert np.isfinite(epi)
        assert abs(epi) < 1e-10, f"epistemic should be ~0 for identical members, got {epi}"
        assert abs(total - alea) < 1e-10

    def test_diverse_ensemble_epistemic_positive(self):
        """Diverse members (different means) → epistemic > 0."""
        total, alea, epi = _crps_uq_decompose_gaussian_ensemble(
            means=[0.0, 5.0, -5.0],
            stds=[1.0, 1.0, 1.0],
        )
        assert np.isfinite(total)
        assert np.isfinite(alea)
        assert np.isfinite(epi)
        assert epi > 0.01, f"epistemic should be > 0 for diverse ensemble, got {epi}"
        assert total >= alea - 1e-10

    def test_epistemic_increases_with_diversity(self):
        """More spread-out means → higher epistemic component."""
        _, _, epi_tight = _crps_uq_decompose_gaussian_ensemble(
            means=[0.0, 0.1], stds=[1.0, 1.0],
        )
        _, _, epi_wide = _crps_uq_decompose_gaussian_ensemble(
            means=[0.0, 10.0], stds=[1.0, 1.0],
        )
        assert epi_wide > epi_tight, (
            f"epistemic should increase with diversity: tight={epi_tight}, wide={epi_wide}"
        )

    def test_non_negativity(self):
        """All decomposition components must be non-negative."""
        for means, stds in [
            ([0.0], [1.0]),
            ([0.0, 1.0], [0.5, 2.0]),
            ([-3.0, 0.0, 3.0], [0.1, 1.0, 5.0]),
        ]:
            total, alea, epi = _crps_uq_decompose_gaussian_ensemble(means, stds)
            assert total >= -1e-12, f"total must be ≥ 0, got {total}"
            assert alea >= -1e-12, f"aleatoric must be ≥ 0, got {alea}"
            assert epi >= -1e-12, f"epistemic must be ≥ 0, got {epi}"

    def test_total_equals_aleatoric_plus_epistemic(self):
        """total ≈ aleatoric + epistemic within floating-point tolerance."""
        for means, stds in [
            ([0.0, 2.0], [1.0, 1.5]),
            ([-1.0, 0.0, 1.0], [0.5, 1.0, 2.0]),
        ]:
            total, alea, epi = _crps_uq_decompose_gaussian_ensemble(means, stds)
            assert abs(total - (alea + epi)) < 1e-10, (
                f"total ({total}) ≠ alea ({alea}) + epi ({epi})"
            )

    def test_single_member_epistemic_zero(self):
        """Single-member ensemble → epistemic = 0."""
        total, alea, epi = _crps_uq_decompose_gaussian_ensemble(
            means=[3.0], stds=[2.0],
        )
        assert abs(epi) < 1e-10
        assert abs(total - alea) < 1e-10
        # For a single Gaussian N(mu, sigma^2), aleatoric = sigma / sqrt(pi)
        expected_alea = 2.0 / math.sqrt(math.pi)
        assert abs(alea - expected_alea) < 1e-10

    def test_uniform_vs_nonuniform_weights(self):
        """Non-uniform weights should shift the decomposition."""
        total_u, alea_u, epi_u = _crps_uq_decompose_gaussian_ensemble(
            means=[0.0, 10.0], stds=[1.0, 1.0], weights=[0.5, 0.5],
        )
        total_w, alea_w, epi_w = _crps_uq_decompose_gaussian_ensemble(
            means=[0.0, 10.0], stds=[1.0, 1.0], weights=[0.99, 0.01],
        )
        # Concentrating weight on one member should reduce epistemic
        assert epi_w < epi_u, (
            f"concentrated weights should reduce epistemic: uniform={epi_u}, concentrated={epi_w}"
        )

    def test_invalid_inputs_raise(self):
        """Empty or mismatched inputs should raise ValueError."""
        with pytest.raises(ValueError):
            _crps_uq_decompose_gaussian_ensemble(means=[], stds=[])
        with pytest.raises(ValueError):
            _crps_uq_decompose_gaussian_ensemble(means=[1.0], stds=[1.0, 2.0])

    def test_all_nan_stds_raise(self):
        """All-NaN stds → ValueError (no finite members)."""
        with pytest.raises(ValueError):
            _crps_uq_decompose_gaussian_ensemble(
                means=[1.0, 2.0], stds=[float("nan"), float("nan")],
            )


# ===================================================================
# Group 3: Flag Trigger Correctness
# ===================================================================


class TestFlagTriggerCorrectness:
    """Verify flags trigger on synthetic boundary cases."""

    def test_near_constant_triggers(self):
        """All-identical data → is_near_constant."""
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_near_constant())
        assert bool(audit.support.is_near_constant) is True

    def test_too_few_unique_triggers(self):
        """Only 3 distinct values among 300 samples → too_few_unique."""
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_too_few_unique())
        assert bool(audit.too_few_unique) is True

    def test_near_constant_rejects_fit(self):
        """Near-constant data should result in rejected DistributionFitSummary."""
        fitter = _make_fitter()
        summary = fitter.select_best_distribution(
            _gen_near_constant(), criterion="simple", feature_index=0,
        )
        assert summary.rejected is True
        assert summary.family is None

    def test_all_diagnostics_finite(self):
        """All numeric audit fields must be finite for clean data."""
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_normal())
        assert np.isfinite(audit.support.frac_zero)
        assert np.isfinite(audit.support.min_value)
        assert np.isfinite(audit.support.max_value)
        assert np.isfinite(audit.support.unique_ratio)
        assert np.isfinite(audit.outlier_fraction)
        assert np.isfinite(audit.frac_negative)
        assert int(audit.n_clean) > 0
        assert int(audit.n_missing) == 0
        assert int(audit.n_unique) > 0

    def test_heaping_delta_is_finite_when_detected(self):
        """When heaping is detected, delta must be finite and positive."""
        fitter = _make_fitter()
        audit = fitter.audit_data(_gen_heaped())
        if audit.has_heaping:
            assert audit.heaping_delta is not None
            assert np.isfinite(audit.heaping_delta)
            assert float(audit.heaping_delta) > 0.0

    def test_outlier_fraction_bounded(self):
        """Outlier fraction must be in [0, 1]."""
        for gen_func in [_gen_normal, _gen_skew, _gen_heavy_tail, _gen_zero_inflated]:
            fitter = _make_fitter()
            audit = fitter.audit_data(gen_func())
            assert 0.0 <= float(audit.outlier_fraction) <= 1.0

    def test_unique_ratio_bounded(self):
        """Unique ratio must be in (0, 1]."""
        for gen_func in [_gen_normal, _gen_skew, _gen_integer_like, _gen_too_few_unique]:
            fitter = _make_fitter()
            audit = fitter.audit_data(gen_func())
            assert 0.0 < float(audit.support.unique_ratio) <= 1.0


# ===================================================================
# Group 4: CRPS UQ Integration via DistributionFitter
# ===================================================================


class TestCRPSUQViaFitter:
    """Test CRPS UQ decomposition when invoked through DistributionFitter."""

    def test_uq_decomposition_emits_fields(self):
        """When opted in, CRPS UQ fields should be populated."""
        fitter = _make_fitter(
            compute_crps_uq_decomposition=True,
            use_support_filtering=False,
            confidence_margin=999.0,  # force large confidence set
        )
        summary = fitter.select_best_distribution(
            _gen_normal(seed=11, n=200), criterion="simple", feature_index=0,
        )
        # Only check if the family was successfully fitted
        if summary.family is not None and not summary.rejected:
            # At least the total should be populated
            assert summary.crps_uq_total is not None
            assert summary.crps_uq_aleatoric is not None
            assert summary.crps_uq_epistemic is not None
            assert float(summary.crps_uq_total) >= 0.0
            assert float(summary.crps_uq_aleatoric) >= 0.0
            assert float(summary.crps_uq_epistemic) >= -1e-10

    def test_uq_not_emitted_when_not_opted_in(self):
        """When not opted in, CRPS UQ fields should be None."""
        fitter = _make_fitter(compute_crps_uq_decomposition=False)
        summary = fitter.select_best_distribution(
            _gen_normal(seed=11, n=200), criterion="simple", feature_index=0,
        )
        assert summary.crps_uq_total is None
        assert summary.crps_uq_aleatoric is None
        assert summary.crps_uq_epistemic is None

    def test_crps_non_negative(self):
        """CRPS score must be non-negative."""
        fitter = _make_fitter(
            compute_crps=True,
            crps_mc_samples=64,
            crps_data_subsample=128,
        )
        summary = fitter.select_best_distribution(
            _gen_normal(seed=23, n=200), criterion="crps", feature_index=0,
        )
        if summary.crps is not None:
            assert float(summary.crps) >= 0.0, f"CRPS must be ≥ 0, got {summary.crps}"

    def test_crps_ordering_independent(self):
        """CRPS should be consistent regardless of observation ordering."""
        data = _gen_skew(seed=42, n=150)
        fitter = _make_fitter(
            compute_crps=True,
            crps_mc_samples=64,
            crps_data_subsample=128,
        )
        s1 = fitter.select_best_distribution(data, criterion="crps", feature_index=0)
        # Reverse order — same underlying values
        s2 = fitter.select_best_distribution(data[::-1].copy(), criterion="crps", feature_index=0)
        if s1.crps is not None and s2.crps is not None and s1.family == s2.family:
            # Should be very close (same data, same family, same MC seed)
            assert abs(float(s1.crps) - float(s2.crps)) < 0.5 * max(float(s1.crps), 1e-6), (
                f"CRPS should be order-independent: {s1.crps} vs {s2.crps}"
            )


# ===================================================================
# Group 5: Seed Stability Tests
# ===================================================================


class TestSeedStability:
    """Verify determinism and stability of diagnostics across seeds."""

    def test_same_seed_same_audit(self):
        """Identical seed → identical audit results."""
        fitter = _make_fitter(random_state=42)
        a1 = fitter.audit_data(_gen_normal(seed=42))
        a2 = fitter.audit_data(_gen_normal(seed=42))
        assert a1.n_clean == a2.n_clean
        assert a1.n_unique == a2.n_unique
        assert a1.support.is_near_constant == a2.support.is_near_constant
        assert a1.has_heaping == a2.has_heaping
        assert a1.is_integer_like == a2.is_integer_like
        assert a1.zero_inflated == a2.zero_inflated
        assert a1.too_few_unique == a2.too_few_unique
        assert abs(a1.outlier_fraction - a2.outlier_fraction) < 1e-12

    def test_same_seed_same_fit(self):
        """Same seed → same fitting result (family + params)."""
        for _ in range(2):
            fitter = _make_fitter(random_state=42)
            s = fitter.select_best_distribution(
                _gen_normal(seed=42, n=200), criterion="simple", feature_index=0,
            )
        fitter2 = _make_fitter(random_state=42)
        s2 = fitter2.select_best_distribution(
            _gen_normal(seed=42, n=200), criterion="simple", feature_index=0,
        )
        assert s.family == s2.family
        if s.params is not None and s2.params is not None:
            for p1, p2 in zip(s.params, s2.params):
                assert abs(p1 - p2) < 1e-10

    def test_different_seed_similar_flags(self):
        """Different data seeds for same distribution → same qualitative flags."""
        fitter = _make_fitter()
        # Zero-inflated data should trigger zero_inflated regardless of seed
        a11 = fitter.audit_data(_gen_zero_inflated(seed=11))
        a23 = fitter.audit_data(_gen_zero_inflated(seed=23))
        a37 = fitter.audit_data(_gen_zero_inflated(seed=37))
        assert bool(a11.zero_inflated) is True
        assert bool(a23.zero_inflated) is True
        assert bool(a37.zero_inflated) is True

    def test_different_seed_integer_like_stable(self):
        """is_integer_like stable across seeds for Poisson data."""
        fitter = _make_fitter()
        for seed in [11, 23, 37, 42, 99]:
            audit = fitter.audit_data(_gen_integer_like(seed=seed))
            assert bool(audit.is_integer_like) is True, (
                f"is_integer_like should be True for Poisson data with seed={seed}"
            )

    def test_different_seed_heaping_stable(self):
        """has_heaping stable across seeds for rounded data."""
        fitter = _make_fitter()
        for seed in [11, 23, 37, 42, 99]:
            audit = fitter.audit_data(_gen_heaped(seed=seed))
            assert bool(audit.has_heaping) is True, (
                f"has_heaping should be True for rounded data with seed={seed}"
            )

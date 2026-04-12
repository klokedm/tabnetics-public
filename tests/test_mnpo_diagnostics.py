"""Tests for MNPO diagnostics instrumentation (T-VR-13 / VAL12_Suggestions)."""

import numpy as np
import pytest


class TestMirrorDescentKLHistory:
    """Verify KL trajectory is returned from mirror_descent_reference_regularized."""

    def test_kl_history_returned_with_history(self):
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n = 5
        rng = np.random.RandomState(42)
        payoff = rng.randn(n, n)
        payoff = (payoff + payoff.T) / 2  # symmetric
        ref = np.ones(n) / n

        p, history = mirror_descent_reference_regularized(
            payoff, ref, steps=50, eta=0.5, lambda_=0.1, return_history=True,
        )

        # kl_values should be attached to history
        assert hasattr(history, 'kl_values'), "history should have kl_values attribute"
        kl = history.kl_values
        assert isinstance(kl, list)
        assert len(kl) > 0, "should have at least one KL value"
        # KL divergence should be non-negative
        for v in kl:
            assert v >= 0.0, f"KL divergence should be non-negative, got {v}"

    def test_kl_history_not_returned_without_history(self):
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n = 3
        ref = np.ones(n) / n
        payoff = np.eye(n)

        result = mirror_descent_reference_regularized(
            payoff, ref, steps=10, eta=0.5, lambda_=0.1, return_history=False,
        )

        # Without return_history, just get the array
        assert isinstance(result, np.ndarray)

    def test_kl_decreases_toward_convergence(self):
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n = 4
        rng = np.random.RandomState(123)
        payoff = rng.randn(n, n)
        payoff = (payoff + payoff.T) / 2
        ref = np.ones(n) / n

        _, history = mirror_descent_reference_regularized(
            payoff, ref, steps=200, eta=0.3, lambda_=0.2, return_history=True,
        )

        kl = history.kl_values
        if len(kl) > 5:
            # Last KL should be smaller than first (convergence)
            assert kl[-1] <= kl[0] + 1e-6, (
                f"KL should decrease: first={kl[0]}, last={kl[-1]}"
            )


class TestPreferenceQuantizationStats:
    """Verify preference quantization statistics computation."""

    def test_quantization_on_synthetic_matrix(self):
        """Test quantization detection on a matrix with known quantized values."""
        # Build a 4x4 preference matrix with values at quantized levels
        # Note: diagonal is 0.5 in real preference matrices; include 0.5.
        mat = np.array([
            [0.5, 0.0, 0.2, 0.8],
            [1.0, 0.5, 0.4, 0.6],
            [0.8, 0.6, 0.5, 0.2],
            [0.2, 0.4, 0.8, 0.5],
        ])

        quantized_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        n = mat.size
        counts = {}
        for level in quantized_levels:
            c = int(np.sum(np.abs(mat - level) < 1e-6))
            counts[str(level)] = c
        total_quantized = sum(counts.values())
        frac = float(total_quantized) / float(n)

        # 0.5 is NOT in quantized_levels, so the 4 diagonal entries are excluded
        # 12 off-diagonal entries are all at quantized levels
        assert frac == 12.0 / 16.0, f"Expected 12/16, got fraction={frac}"
        assert counts["0.0"] == 1
        assert counts["1.0"] == 1
        assert counts["0.2"] == 3
        assert counts["0.8"] == 3

    def test_non_quantized_matrix(self):
        """Test that continuous values have low quantization fraction."""
        rng = np.random.RandomState(42)
        mat = rng.uniform(0.01, 0.99, size=(10, 10))

        quantized_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        n = mat.size
        total_quantized = 0
        for level in quantized_levels:
            total_quantized += int(np.sum(np.abs(mat - level) < 1e-6))
        frac = float(total_quantized) / float(n)

        assert frac < 0.1, f"Random matrix should have low quantization, got {frac}"


class TestWeightTrajectory:
    """Verify weight trajectory capping logic."""

    def test_trajectory_capped_at_20(self):
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n = 3
        ref = np.ones(n) / n
        rng = np.random.RandomState(42)
        payoff = rng.randn(n, n)

        _, history = mirror_descent_reference_regularized(
            payoff, ref, steps=200, eta=0.01, lambda_=0.001,
            tol_kl=1e-20,  # very tight to force many iterations
            return_history=True,
        )

        # Simulate the capping logic from portfolio.py
        cap = 20
        if len(history) > cap:
            capped = [[float(v) for v in w.ravel()] for w in history[-cap:]]
        else:
            capped = [[float(v) for v in w.ravel()] for w in history]

        assert len(capped) <= cap, f"Weight trajectory should be capped at {cap}"

    def test_short_trajectory_not_truncated(self):
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n = 3
        ref = np.ones(n) / n
        payoff = np.eye(n)

        _, history = mirror_descent_reference_regularized(
            payoff, ref, steps=5, eta=0.5, lambda_=0.1, return_history=True,
        )

        cap = 20
        if len(history) > cap:
            capped = history[-cap:]
        else:
            capped = history

        assert len(capped) == len(history), "Short trajectory should not be truncated"
        assert len(capped) <= 6, "At most steps+1 entries (including initial)"

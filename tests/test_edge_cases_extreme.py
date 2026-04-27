"""Tests for extreme/degenerate inputs (VAL12_Suggestions §1.3 / Independent Review E.1)."""

import numpy as np
import pytest


class TestEmptyDataset:
    """Verify pipeline components handle n=0 gracefully."""

    def test_cv_splits_empty(self):
        from tabnetics.feature_selection.cv import get_inner_cv_splits

        X = np.empty((0, 10), dtype=float)
        y = np.array([], dtype=int)
        splits = get_inner_cv_splits(X, y, 'classification', 42, 5, 3)
        assert splits == [], "Empty dataset should produce no splits"

    def test_prefilter_empty(self):
        from tabnetics.feature_selection.base import FeatureSelector

        fs = FeatureSelector.__new__(FeatureSelector)
        # Minimal attributes needed by _prefilter_feature_pool
        fs.random_state = 42
        fs.prefilter_mi_n_neighbors = 5
        fs.prefilter_mi_weight = 0.60
        fs.prefilter_f_weight = 0.40
        X = np.empty((0, 5), dtype=float)
        y = np.array([], dtype=int)
        result = fs._prefilter_feature_pool(X, y, max_features=100)
        assert result.size == 0, "Empty dataset should return empty indices"

    def test_mirror_descent_empty_payoff(self):
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        payoff = np.zeros((3, 3), dtype=float)
        ref = np.ones(3) / 3
        p = mirror_descent_reference_regularized(
            payoff, ref, steps=10, eta=0.5, lambda_=0.1, return_history=False,
        )
        # With zero payoff, mirror descent should stay near reference prior
        assert p.shape == (3,)
        assert abs(float(np.sum(p)) - 1.0) < 1e-6, "Should sum to 1"
        np.testing.assert_allclose(p, ref, atol=0.1)


class TestSingleSample:
    """Verify pipeline components handle n=1 gracefully."""

    def test_cv_splits_single(self):
        from tabnetics.feature_selection.cv import get_inner_cv_splits

        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([0])
        splits = get_inner_cv_splits(X, y, 'classification', 42, 5, 3)
        assert splits == [], "Single sample should produce no splits"

    def test_prefilter_single(self):
        from tabnetics.feature_selection.base import FeatureSelector

        fs = FeatureSelector.__new__(FeatureSelector)
        fs.random_state = 42
        fs.prefilter_mi_n_neighbors = 5
        fs.prefilter_mi_weight = 0.60
        fs.prefilter_f_weight = 0.40
        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([0])
        result = fs._prefilter_feature_pool(X, y, max_features=100)
        assert result.size == 0, "Single sample should return empty indices"


class TestTwoSamples:
    """Verify pipeline handles n=2 (minimum for any meaningful split)."""

    def test_cv_splits_two_samples(self):
        from tabnetics.feature_selection.cv import get_inner_cv_splits

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        splits = get_inner_cv_splits(X, y, 'classification', 42, 5, 3)
        # n=2 < 4 triggers KFold(n_splits=2) → exactly 1 split (2-fold = 2 subsets)
        assert len(splits) == 2, "Two samples with KFold(2) should produce 2 splits"
        for train_idx, val_idx in splits:
            assert len(train_idx) == 1
            assert len(val_idx) == 1


class TestExtremePtoNRatio:
    """Verify pipeline handles p >> n (HDLSS regime)."""

    def test_mirror_descent_hdlss(self):
        """Mirror descent with many candidates should still converge."""
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n_cand = 50  # Many candidates
        rng = np.random.RandomState(42)
        payoff = rng.randn(n_cand, n_cand)
        payoff = (payoff + payoff.T) / 2
        ref = np.ones(n_cand) / n_cand

        p = mirror_descent_reference_regularized(
            payoff, ref, steps=200, eta=0.1, lambda_=0.1, return_history=False,
        )
        assert p.shape == (n_cand,)
        assert abs(float(np.sum(p)) - 1.0) < 1e-6
        assert np.all(p >= 0), "Weights should be non-negative"

    def test_prefilter_high_dim(self):
        """Prefilter should handle p=1000 features with n=10 samples."""
        from tabnetics.feature_selection.base import FeatureSelector

        fs = FeatureSelector.__new__(FeatureSelector)
        fs.random_state = 42
        fs.prefilter_mi_n_neighbors = 3
        fs.prefilter_mi_weight = 0.60
        fs.prefilter_f_weight = 0.40
        rng = np.random.RandomState(42)
        X = rng.randn(10, 1000)
        y = np.array([0] * 5 + [1] * 5)
        result = fs._prefilter_feature_pool(X, y, max_features=50)
        assert result.size <= 50, "Should respect max_features cap"
        assert result.size > 0, "Should select some features"

    def test_aggregate_payoff_single_candidate(self):
        """Aggregate payoff with 1 candidate should produce 1x1 matrix."""
        from tabnetics.core.mnpo import aggregate_payoff_matrix

        mat = np.array([[0.5]])
        oracle_matrices = {"perf": mat}
        oracle_weights = {"perf": 1.0}
        payoff = aggregate_payoff_matrix(oracle_matrices, oracle_weights)
        assert payoff.shape == (1, 1)
        assert payoff[0, 0] == 0.0, "Diagonal should be 0 for skew-symmetric payoff"


class TestDegeneratePayoff:
    """Verify mirror descent handles pathological payoff matrices."""

    def test_identity_payoff(self):
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n = 4
        payoff = np.eye(n)
        ref = np.ones(n) / n
        p = mirror_descent_reference_regularized(
            payoff, ref, steps=50, eta=0.1, lambda_=0.1, return_history=False,
        )
        assert p.shape == (n,)
        assert abs(float(np.sum(p)) - 1.0) < 1e-6

    def test_all_equal_payoff(self):
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n = 4
        payoff = np.full((n, n), 0.5)
        ref = np.ones(n) / n
        p = mirror_descent_reference_regularized(
            payoff, ref, steps=50, eta=0.1, lambda_=0.1, return_history=False,
        )
        assert p.shape == (n,)
        assert abs(float(np.sum(p)) - 1.0) < 1e-6

    def test_zero_reference_prior(self):
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n = 3
        rng = np.random.RandomState(42)
        payoff = rng.randn(n, n)
        payoff = (payoff + payoff.T) / 2
        ref = np.zeros(n)  # Degenerate prior — should be clipped to eps

        p = mirror_descent_reference_regularized(
            payoff, ref, steps=50, eta=0.1, lambda_=0.1, return_history=False,
        )
        assert p.shape == (n,)
        assert abs(float(np.sum(p)) - 1.0) < 1e-6
        assert np.all(np.isfinite(p)), "Should produce finite weights despite zero prior"

    def test_large_eta_no_divergence(self):
        """Large learning rate should not cause NaN or inf."""
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n = 5
        rng = np.random.RandomState(42)
        payoff = rng.randn(n, n) * 10
        ref = np.ones(n) / n

        p = mirror_descent_reference_regularized(
            payoff, ref, steps=100, eta=10.0, lambda_=0.01, return_history=False,
        )
        assert np.all(np.isfinite(p)), "Large eta should not cause NaN/inf"
        assert abs(float(np.sum(p)) - 1.0) < 1e-6


class TestFoldRegret:
    """Verify fold_regret_mean_max handles edge cases."""

    def test_fold_regret_empty(self):
        from tabnetics.core.mnpo import fold_regret_mean_max

        mat = np.empty((0, 0), dtype=float)
        mean_r, max_r = fold_regret_mean_max(mat)
        assert mean_r.size == 0
        assert max_r.size == 0

    def test_fold_regret_single_candidate(self):
        from tabnetics.core.mnpo import fold_regret_mean_max

        mat = np.array([[0.8, 0.9, 0.7]])  # 1 candidate, 3 folds
        mean_r, max_r = fold_regret_mean_max(mat)
        # Single candidate is always the best → zero regret
        np.testing.assert_allclose(mean_r, [0.0], atol=1e-12)
        np.testing.assert_allclose(max_r, [0.0], atol=1e-12)

    def test_fold_regret_1d_raises(self):
        from tabnetics.core.mnpo import fold_regret_mean_max

        with pytest.raises(ValueError, match="2D"):
            fold_regret_mean_max(np.array([1.0, 2.0, 3.0]))


class TestSpearmanCorrelationEdge:
    """Verify spearman_correlation handles edge cases."""

    def test_spearman_too_few(self):
        from tabnetics.core.mnpo import spearman_correlation

        assert spearman_correlation([1.0], [2.0]) == 0.0
        assert spearman_correlation([], []) == 0.0

    def test_spearman_constant(self):
        from tabnetics.core.mnpo import spearman_correlation

        result = spearman_correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])
        assert result == 0.0, "Constant input should give zero correlation"

    def test_spearman_nan_handling(self):
        from tabnetics.core.mnpo import spearman_correlation

        result = spearman_correlation(
            [1.0, float('nan'), 3.0, 4.0, 5.0],
            [2.0, float('nan'), 6.0, 8.0, 10.0],
        )
        assert np.isfinite(result), "NaN values should be filtered, not propagated"


class TestLowerTailCVaR:
    """Verify lower_tail_cvar handles edge cases."""

    def test_cvar_empty(self):
        from tabnetics.core.mnpo import lower_tail_cvar

        result = lower_tail_cvar([], alpha=0.33)
        assert np.isnan(result), "Empty input should return NaN"

    def test_cvar_single(self):
        from tabnetics.core.mnpo import lower_tail_cvar

        result = lower_tail_cvar([0.5], alpha=0.33)
        assert result == 0.5, "Single value CVaR should equal the value"

    def test_cvar_alpha_zero(self):
        from tabnetics.core.mnpo import lower_tail_cvar

        result = lower_tail_cvar([0.1, 0.5, 0.9], alpha=0.0)
        assert result == pytest.approx(0.1), "Alpha=0 should return the minimum"

    def test_cvar_all_nan(self):
        from tabnetics.core.mnpo import lower_tail_cvar

        result = lower_tail_cvar([float('nan'), float('nan')], alpha=0.33)
        assert np.isnan(result), "All-NaN input should return NaN"

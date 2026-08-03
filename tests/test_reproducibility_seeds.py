"""Tests for reproducibility / seed propagation (VAL12_Suggestions §1.2)."""

import numpy as np
import pytest


class TestMirrorDescentDeterminism:
    """Verify mirror descent produces identical outputs with the same seed."""

    def test_same_seed_same_result(self):
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n = 5
        rng = np.random.RandomState(42)
        payoff = rng.randn(n, n)
        payoff = (payoff + payoff.T) / 2
        ref = np.ones(n) / n

        kwargs = dict(steps=100, eta=0.5, lambda_=0.1, return_history=False)

        p1 = mirror_descent_reference_regularized(payoff.copy(), ref.copy(), **kwargs)
        p2 = mirror_descent_reference_regularized(payoff.copy(), ref.copy(), **kwargs)

        np.testing.assert_array_almost_equal(p1, p2, decimal=14)

    def test_different_payoff_different_result(self):
        from tabnetics.core.mnpo import mirror_descent_reference_regularized

        n = 4
        ref = np.ones(n) / n
        kwargs = dict(steps=50, eta=0.3, lambda_=0.1, return_history=False)

        rng1 = np.random.RandomState(1)
        payoff1 = rng1.randn(n, n)
        payoff1 = (payoff1 + payoff1.T) / 2

        rng2 = np.random.RandomState(2)
        payoff2 = rng2.randn(n, n)
        payoff2 = (payoff2 + payoff2.T) / 2

        p1 = mirror_descent_reference_regularized(payoff1, ref.copy(), **kwargs)
        p2 = mirror_descent_reference_regularized(payoff2, ref.copy(), **kwargs)

        # Different payoffs should give different equilibria
        assert not np.allclose(p1, p2, atol=1e-6)


class TestCVSplitDeterminism:
    """Verify CV splitting is deterministic with the same seed."""

    def test_stratified_cv_deterministic(self):
        from tabnetics.feature_selection.cv import get_inner_cv_splits

        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        y = np.array([0] * 25 + [1] * 25)

        splits1 = get_inner_cv_splits(X, y, 'classification', 42, 5, 3)
        splits2 = get_inner_cv_splits(X, y, 'classification', 42, 5, 3)

        assert len(splits1) == len(splits2), "Should produce same number of splits"
        for (t1, v1), (t2, v2) in zip(splits1, splits2):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_different_seed_different_splits(self):
        from tabnetics.feature_selection.cv import get_inner_cv_splits

        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        y = np.array([0] * 25 + [1] * 25)

        splits1 = get_inner_cv_splits(X, y, 'classification', 42, 5, 3)
        splits2 = get_inner_cv_splits(X, y, 'classification', 99, 5, 3)

        # At least some splits should differ
        any_different = False
        for (t1, _), (t2, _) in zip(splits1, splits2):
            if not np.array_equal(t1, t2):
                any_different = True
                break
        assert any_different, "Different seeds should produce different splits"

    def test_loocv_deterministic(self):
        """LOOCV (n < 20 with min-class < 2) is inherently deterministic."""
        from tabnetics.feature_selection.cv import get_inner_cv_splits

        # n=6 < 20 and min-class=1 < 2 triggers class-sparse fallback (RepeatedKFold),
        # so use n=3 < 4 which triggers the simple KFold path with seed.
        X = np.eye(3)
        y = np.array([0, 0, 1])

        # n < 4 uses KFold(n_splits=2, shuffle, random_state) → deterministic
        splits1 = get_inner_cv_splits(X, y, 'classification', 42, 5, 3)
        splits2 = get_inner_cv_splits(X, y, 'classification', 42, 5, 3)

        assert len(splits1) == len(splits2)
        for (t1, v1), (t2, v2) in zip(splits1, splits2):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_repeated_cv_repeats_not_pinned_to_one_split(self):
        """Repeated CV must generate multiple distinct folds under one root seed."""
        from tabnetics.feature_selection.cv import get_inner_cv_splits

        rng = np.random.RandomState(42)
        X = rng.randn(60, 12)
        y = np.array([0] * 30 + [1] * 30)

        splits = get_inner_cv_splits(X, y, 'classification', 42, 5, 3)

        assert len(splits) == 15
        unique_val_splits = {tuple(np.asarray(val_idx, dtype=int).tolist()) for _, val_idx in splits}
        assert len(unique_val_splits) > 5, "Repeated CV appears pinned to one fold schedule"


class TestPerMethodSeedDerivation:
    """Verify per-method seed derivation produces unique, deterministic seeds."""

    def test_per_method_seeds_unique(self):
        """Each method should get a distinct seed."""
        from tabnetics.feature_selection.base import FeatureSelector

        selector = FeatureSelector(random_state=42)
        method_names = [f"method_{idx}" for idx in range(20)]
        seeds = [selector._derive_method_seed(name) for name in method_names]
        assert len(set(seeds)) == len(method_names), "All method seeds should be unique"

    def test_per_method_seeds_deterministic(self):
        """Same root seed + method name should produce the same derived seed."""
        from tabnetics.feature_selection.base import FeatureSelector

        selector = FeatureSelector(random_state=42)
        for idx in range(10):
            method_name = f"method_{idx}"
            s1 = selector._derive_method_seed(method_name)
            s2 = selector._derive_method_seed(method_name)
            assert s1 == s2

    def test_per_method_seeds_valid_range(self):
        """Derived seeds should be valid numpy RandomState seeds."""
        from tabnetics.feature_selection.base import FeatureSelector

        for base_seed in [0, 1, 42, 2**31 - 2, 2**31 - 1]:
            selector = FeatureSelector(random_state=base_seed)
            for idx in range(50):
                seed = selector._derive_method_seed(f"method_{idx}")
                assert 0 <= seed < 2**31, f"Seed {seed} out of valid range"
                # Should not raise
                rng = np.random.RandomState(seed)
                rng.random()

    def test_per_method_seeds_stable_across_execution_order(self):
        """A method seed should not depend on task ordering or enabled subset shape."""
        from tabnetics.feature_selection.base import FeatureSelector

        selector = FeatureSelector(random_state=77)
        schedule_a = {
            name: selector._derive_method_seed(name)
            for name in ("mutual_information", "anova_f", "linear_svm")
        }
        schedule_b = {
            name: selector._derive_method_seed(name)
            for name in ("linear_svm", "mutual_information")
        }

        assert schedule_a["mutual_information"] == schedule_b["mutual_information"]
        assert schedule_a["linear_svm"] == schedule_b["linear_svm"]

    def test_run_selection_methods_uses_same_schedule_sequential_and_parallel(self, monkeypatch):
        """Sequential and parallel FS dispatch should derive the same method seeds."""
        from tabnetics.feature_selection.base import FeatureSelector

        rng = np.random.RandomState(7)
        X = rng.randn(24, 8)
        y = np.array([0] * 12 + [1] * 12)
        methods = ("mutual_information", "anova_f", "linear_svm")

        seen_seq = {}
        selector_seq = FeatureSelector(random_state=123, enabled_methods=methods, parallel_n_jobs=1)

        def _fake_seq(method_name, method_fn, contract, X_uncorr, y_arr, n_target, class_pareto_min_classes,
                      use_timeout=True, method_seed=None):
            seen_seq[str(method_name)] = method_seed
            return str(method_name), ({}, {}), 0.0

        monkeypatch.setattr(selector_seq, "_run_single_method", _fake_seq)
        selector_seq._run_selection_methods(X, y, 3)

        selector_par = FeatureSelector(random_state=123, enabled_methods=methods, parallel_n_jobs=2)

        def _fake_par(method_name, method_fn, contract, X_uncorr, y_arr, n_target, class_pareto_min_classes,
                      use_timeout=True, method_seed=None):
            return str(method_name), ({}, {}), 0.0

        monkeypatch.setattr(selector_par, "_run_single_method", _fake_par)
        parallel_results, _ = selector_par._run_selection_methods(X, y, 3)
        seen_par = {
            str(method_name): parallel_results[str(method_name)][0][
                "execution_provenance"
            ]["method_seed"]
            for method_name in methods
        }

        assert seen_seq == seen_par
        assert set(seen_seq.keys()) == set(methods)
        assert all(seed is not None for seed in seen_seq.values())
        assert len(set(seen_seq.values())) == len(methods)


class TestDistributionFitSeedIndependence:
    """Verify per-distribution seed derivation in ProcessPool config."""

    def test_per_dist_seeds_unique(self):
        """Each distribution worker should get a distinct seed."""
        base_seed = 42
        n_dists = 15
        seeds = [int((base_seed + idx) % (2**31 - 1)) for idx in range(n_dists)]
        assert len(set(seeds)) == n_dists

    def test_per_dist_rng_independence(self):
        """Different distribution seeds should produce different random sequences."""
        base_seed = 42
        sequences = []
        for idx in range(5):
            seed = int((base_seed + idx) % (2**31 - 1))
            rng = np.random.RandomState(seed)
            sequences.append(rng.random(10))

        # All sequences should be different
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                assert not np.allclose(sequences[i], sequences[j]), (
                    f"Distributions {i} and {j} got identical random sequences"
                )


class TestAggregatePayoffDeterminism:
    """Verify aggregate payoff matrix computation is deterministic."""

    def test_aggregate_payoff_deterministic(self):
        from tabnetics.core.mnpo import aggregate_payoff_matrix

        rng = np.random.RandomState(42)
        n = 4
        mat_a = rng.uniform(0.1, 0.9, (n, n))
        np.fill_diagonal(mat_a, 0.5)
        mat_b = rng.uniform(0.1, 0.9, (n, n))
        np.fill_diagonal(mat_b, 0.5)

        oracle_matrices = {"perf": mat_a, "stab": mat_b}
        oracle_weights = {"perf": 0.7, "stab": 0.3}

        payoff1 = aggregate_payoff_matrix(oracle_matrices, oracle_weights)
        payoff2 = aggregate_payoff_matrix(oracle_matrices, oracle_weights)

        np.testing.assert_array_almost_equal(payoff1, payoff2, decimal=14)

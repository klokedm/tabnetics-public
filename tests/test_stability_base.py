"""Unit tests for stability_base.py — Phase 5 extraction.

Tests that each variant class runs end-to-end with synthetic data and
produces the expected result-dict keys.  Does NOT test numerical equivalence
with the monolith (that's covered by the regression tests and the existing
test_mnpo_feature_selector.py tests which exercise the thin wrappers).
"""

import unittest
import numpy as np
from tabnetics.feature_selection.stability_base import (
    StabilitySelectionBase,
    SubsampleStability,
    TigressStability,
    DecorrelatedStability,
    ClusterStability,
    SubspaceStability,
    _normalize_vector_01,
)


def _dummy_prefilter(X, y, max_features):
    """Identity prefilter — returns all feature indices."""
    return np.arange(X.shape[1], dtype=int)


def _dummy_fit_score(X_train, y_train, X_val, y_val):
    """Dummy scorer: random balanced accuracy between 0.4 and 0.8."""
    return 0.6, {}


def _make_binary_data(n_samples=60, n_features=30, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    # Make first 5 features signal-bearing
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def _make_regression_data(n_samples=60, n_features=30, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.1, n_samples)
    return X, y


class TestNormalizeVector01(unittest.TestCase):
    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0])
        normed = _normalize_vector_01(arr)
        self.assertAlmostEqual(float(normed[0]), 0.0)
        self.assertAlmostEqual(float(normed[2]), 1.0)

    def test_constant(self):
        arr = np.array([5.0, 5.0, 5.0])
        normed = _normalize_vector_01(arr)
        np.testing.assert_allclose(normed, 0.5)

    def test_empty(self):
        arr = np.array([])
        normed = _normalize_vector_01(arr)
        self.assertEqual(normed.size, 0)


class TestSubsampleStability(unittest.TestCase):
    def test_classification_smoke(self):
        X, y = _make_binary_data()
        runner = SubsampleStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
        )
        results, scores = runner.run(X, y, 5, _dummy_prefilter)
        self.assertIn("selected_indices", results)
        self.assertEqual(len(results["selected_indices"]), 5)
        self.assertIn("stability_score", results)
        self.assertIn("selection_frequency", results)
        self.assertIn("n_fits", results)
        self.assertIn("loss_guided_validation_enabled", results)
        self.assertFalse(results["loss_guided_validation_enabled"])
        self.assertEqual(len(scores), X.shape[1])

    def test_regression_smoke(self):
        X, y = _make_regression_data()
        runner = SubsampleStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="regression",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
        )
        results, scores = runner.run(X, y, 5, _dummy_prefilter)
        self.assertIn("selected_indices", results)
        self.assertEqual(len(results["selected_indices"]), 5)

    def test_loss_guided_validation_metadata(self):
        X, y = _make_binary_data(n_samples=80)
        runner = SubsampleStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=3,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
            use_loss_guided_validation=True,
            validation_fraction=0.30,
            validation_quantile=0.50,
            validation_min_samples=5,
        )
        results, scores = runner.run(
            X, y, 5, _dummy_prefilter, fit_score_fn=_dummy_fit_score,
        )
        self.assertTrue(results["loss_guided_validation_enabled"])
        self.assertIn("loss_guided_validation_threshold", results)
        self.assertIn("loss_guided_validation_fallback", results)
        self.assertIn("n_fit_records", results)
        self.assertGreater(results["n_fit_records"], 0)

    def test_empty_features(self):
        X = np.zeros((10, 0))
        y = np.array([0] * 5 + [1] * 5)
        runner = SubsampleStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
        )
        results, scores = runner.run(X, y, 5, _dummy_prefilter)
        self.assertEqual(results, {})
        self.assertEqual(scores, {})


class TestTigressStability(unittest.TestCase):
    def test_classification_smoke(self):
        X, y = _make_binary_data()
        runner = TigressStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
            ipss_min_c=0.08,
            ipss_max_c=1.20,
            ipss_path_grid_size=3,
        )
        results, scores = runner.run(X, y, 5, _dummy_prefilter)
        self.assertIn("selected_indices", results)
        self.assertEqual(len(results["selected_indices"]), 5)
        self.assertIn("tigress_score", results)
        self.assertIn("tigress_path_grid", results)
        self.assertIn("tigress_path_fit_counts", results)
        self.assertIn("tigress_max_frequency", results)
        self.assertIn("tigress_random_weight_low", results)
        self.assertGreaterEqual(len(results["tigress_path_grid"]), 3)

    def test_regression_smoke(self):
        X, y = _make_regression_data()
        runner = TigressStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="regression",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
        )
        results, scores = runner.run(X, y, 5, _dummy_prefilter)
        self.assertIn("selected_indices", results)


class TestDecorrelatedStability(unittest.TestCase):
    def test_classification_smoke(self):
        X, y = _make_binary_data()
        runner = DecorrelatedStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
            decorrelated_stability_eps=1e-3,
        )
        results, scores = runner.run(X, y, 5, _dummy_prefilter)
        self.assertIn("selected_indices", results)
        self.assertGreater(len(results["selected_indices"]), 0)
        self.assertIn("decorrelation_condition_number", results)
        self.assertIn("decorrelation_rank", results)
        self.assertFalse(results.get("decorrelation_gated", True))

    def test_correlation_gate(self):
        """Near-orthogonal features should trigger the correlation gate."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 20))
        y = (X[:, 0] > 0).astype(int)
        runner = DecorrelatedStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
            decorrelated_stability_min_max_abs_corr=0.99,
        )
        results, scores = runner.run(X, y, 5, _dummy_prefilter)
        self.assertTrue(results.get("decorrelation_gated", False))
        self.assertEqual(len(results.get("selected_indices", [])), 0)

    def test_small_pool_returns_empty(self):
        """With only 1 feature, should return empty (caller fallback)."""
        X = np.random.default_rng(42).standard_normal((30, 1))
        y = (X[:, 0] > 0).astype(int)
        runner = DecorrelatedStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
        )
        results, scores = runner.run(X, y, 1, _dummy_prefilter)
        self.assertEqual(results, {})


class TestClusterStability(unittest.TestCase):
    def test_classification_smoke(self):
        X, y = _make_binary_data()
        runner = ClusterStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
            cluster_stability_corr_threshold=0.85,
            cluster_stability_max_per_cluster=2,
            cluster_stability_min_cluster_freq=0.55,
        )
        results, scores = runner.run(X, y, 5, _dummy_prefilter)
        self.assertIn("selected_indices", results)
        self.assertGreater(len(results["selected_indices"]), 0)
        self.assertIn("n_clusters", results)
        self.assertIn("cluster_sizes", results)
        self.assertIn("cluster_frequency", results)
        self.assertGreaterEqual(results["n_clusters"], 1)

    def test_build_correlation_clusters(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        runner = ClusterStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
        )
        clusters = runner._build_correlation_clusters(X)
        self.assertIsInstance(clusters, list)
        self.assertGreater(len(clusters), 0)
        # All features should be present across clusters
        all_members = set()
        for c in clusters:
            all_members.update(c)
        self.assertEqual(all_members, set(range(10)))


class TestSubspaceStability(unittest.TestCase):
    def test_classification_smoke(self):
        X, y = _make_binary_data()
        subsample_runner = SubsampleStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
        )
        runner = SubspaceStability(
            subsample_stability=subsample_runner,
            corr_threshold=0.85,
            selection_threshold=0.6,
        )
        results, scores = runner.run(X, y, 5, _dummy_prefilter)
        self.assertIn("selected_indices", results)
        self.assertGreater(len(results["selected_indices"]), 0)
        self.assertIn("equivalent_models", results)
        self.assertGreaterEqual(len(results["equivalent_models"]), 1)
        self.assertIn("n_subspace_groups", results)
        self.assertIn("subspace_corr_threshold", results)
        self.assertIn("subspace_group_sizes", results)


class TestBaseClassIsAbstract(unittest.TestCase):
    def test_cannot_instantiate_base(self):
        with self.assertRaises(TypeError):
            StabilitySelectionBase(
                subsample_fraction=0.5,
                selection_threshold=0.6,
                n_bootstrap_iterations=2,
                random_state=42,
                problem_type="classification",
                linear_svm_max_iter=2000,
                mrmr_max_features=200,
            )


class TestParallelBootstrap(unittest.TestCase):
    """Verify parallel_n_jobs > 1 produces identical results to sequential."""

    def _compare_results(self, cls, cls_kwargs, run_kwargs=None):
        X, y = _make_binary_data(n_samples=80, n_features=20)
        if run_kwargs is None:
            run_kwargs = {}

        runner_seq = cls(**cls_kwargs, parallel_n_jobs=1)
        res_seq, scores_seq = runner_seq.run(X, y, 5, _dummy_prefilter, **run_kwargs)

        runner_par = cls(**cls_kwargs, parallel_n_jobs=2)
        res_par, scores_par = runner_par.run(X, y, 5, _dummy_prefilter, **run_kwargs)

        # Selected indices must be identical
        np.testing.assert_array_equal(
            res_seq["selected_indices"], res_par["selected_indices"],
            err_msg=f"{cls.__name__}: selected_indices differ with parallel_n_jobs=2",
        )
        # Scores dict must match
        self.assertEqual(
            set(scores_seq.keys()), set(scores_par.keys()),
            f"{cls.__name__}: score keys differ",
        )
        for k in scores_seq:
            self.assertAlmostEqual(
                scores_seq[k], scores_par[k], places=6,
                msg=f"{cls.__name__}: score[{k}] differs",
            )

    def test_subsample_parallel_matches_sequential(self):
        self._compare_results(
            SubsampleStability,
            dict(
                subsample_fraction=0.5,
                selection_threshold=0.6,
                n_bootstrap_iterations=4,
                random_state=42,
                problem_type="classification",
                linear_svm_max_iter=2000,
                mrmr_max_features=200,
            ),
        )

    def test_subsample_loss_guided_parallel(self):
        self._compare_results(
            SubsampleStability,
            dict(
                subsample_fraction=0.5,
                selection_threshold=0.6,
                n_bootstrap_iterations=4,
                random_state=42,
                problem_type="classification",
                linear_svm_max_iter=2000,
                mrmr_max_features=200,
                use_loss_guided_validation=True,
                validation_fraction=0.30,
                validation_quantile=0.50,
                validation_min_samples=5,
            ),
            run_kwargs=dict(fit_score_fn=_dummy_fit_score),
        )

    def test_tigress_parallel_matches_sequential(self):
        self._compare_results(
            TigressStability,
            dict(
                subsample_fraction=0.5,
                selection_threshold=0.6,
                n_bootstrap_iterations=4,
                random_state=42,
                problem_type="classification",
                linear_svm_max_iter=2000,
                mrmr_max_features=200,
                ipss_min_c=0.08,
                ipss_max_c=1.20,
                ipss_path_grid_size=3,
            ),
        )

    def test_decorrelated_parallel_matches_sequential(self):
        self._compare_results(
            DecorrelatedStability,
            dict(
                subsample_fraction=0.5,
                selection_threshold=0.6,
                n_bootstrap_iterations=4,
                random_state=42,
                problem_type="classification",
                linear_svm_max_iter=2000,
                mrmr_max_features=200,
                decorrelated_stability_eps=1e-3,
            ),
        )

    def test_cluster_parallel_matches_sequential(self):
        self._compare_results(
            ClusterStability,
            dict(
                subsample_fraction=0.5,
                selection_threshold=0.6,
                n_bootstrap_iterations=4,
                random_state=42,
                problem_type="classification",
                linear_svm_max_iter=2000,
                mrmr_max_features=200,
                cluster_stability_corr_threshold=0.85,
                cluster_stability_max_per_cluster=2,
                cluster_stability_min_cluster_freq=0.55,
            ),
        )

    def test_parallel_n_jobs_minus1(self):
        """parallel_n_jobs=-1 should use all CPUs without error."""
        X, y = _make_binary_data()
        runner = SubsampleStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
            parallel_n_jobs=-1,
        )
        results, scores = runner.run(X, y, 5, _dummy_prefilter)
        self.assertIn("selected_indices", results)
        self.assertEqual(len(results["selected_indices"]), 5)

    def test_default_parallel_n_jobs_is_1(self):
        """Default parallel_n_jobs should be 1."""
        runner = SubsampleStability(
            subsample_fraction=0.5,
            selection_threshold=0.6,
            n_bootstrap_iterations=2,
            random_state=42,
            problem_type="classification",
            linear_svm_max_iter=2000,
            mrmr_max_features=200,
        )
        self.assertEqual(runner.parallel_n_jobs, 1)


if __name__ == "__main__":
    unittest.main()

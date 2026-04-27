"""Tests for T-R-249: Banzhaf value computation."""

import unittest
import numpy as np


class TestBanzhafValues(unittest.TestCase):
    """Unit tests for Banzhaf value computation in mnpo_core."""

    def _make_oracle_matrices(self, k=3, m=4):
        """Create k random oracle preference matrices of size m×m."""
        rng = np.random.RandomState(42)
        matrices = {}
        for i in range(k):
            mat = rng.uniform(0.1, 0.9, (m, m))
            np.fill_diagonal(mat, 0.5)
            matrices[f"oracle_{i}"] = mat
        return matrices

    def test_empty_oracles_returns_empty(self):
        from tabnetics.core.mnpo import compute_banzhaf_values
        weights, meta = compute_banzhaf_values({})
        self.assertEqual(weights, {})
        self.assertFalse(meta["applied"])

    def test_single_oracle_returns_unit_weight(self):
        from tabnetics.core.mnpo import compute_banzhaf_values
        mat = np.full((3, 3), 0.5)
        weights, meta = compute_banzhaf_values({"perf": mat})
        self.assertEqual(weights["perf"], 1.0)
        self.assertTrue(meta["applied"])

    def test_banzhaf_weights_sum_to_one(self):
        from tabnetics.core.mnpo import compute_banzhaf_values
        matrices = self._make_oracle_matrices(k=3, m=4)
        weights, meta = compute_banzhaf_values(matrices)
        self.assertTrue(meta["applied"])
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=8)

    def test_banzhaf_values_are_non_negative(self):
        from tabnetics.core.mnpo import compute_banzhaf_values
        matrices = self._make_oracle_matrices(k=4, m=5)
        weights, meta = compute_banzhaf_values(matrices)
        for w in weights.values():
            self.assertGreaterEqual(w, 0.0)

    def test_banzhaf_raw_values_in_meta(self):
        from tabnetics.core.mnpo import compute_banzhaf_values
        matrices = self._make_oracle_matrices(k=3, m=4)
        _, meta = compute_banzhaf_values(matrices)
        self.assertIn("raw_banzhaf", meta)
        self.assertEqual(len(meta["raw_banzhaf"]), 3)

    def test_kernel_banzhaf_fallback_for_many_oracles(self):
        from tabnetics.core.mnpo import compute_banzhaf_values
        matrices = self._make_oracle_matrices(k=14, m=4)
        weights, meta = compute_banzhaf_values(matrices, max_coalitions=256)
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["method"], "kernel_regression")
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_kernel_banzhaf_direct(self):
        from tabnetics.core.mnpo import kernel_banzhaf_values
        matrices = self._make_oracle_matrices(k=5, m=4)
        weights, meta = kernel_banzhaf_values(matrices, n_samples=512, seed=0)
        self.assertTrue(meta["applied"])
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        for w in weights.values():
            self.assertGreaterEqual(w, 0.0)

    def test_banzhaf_exact_vs_kernel_agreement(self):
        """Exact and kernel Banzhaf should roughly agree for small k."""
        from tabnetics.core.mnpo import compute_banzhaf_values, kernel_banzhaf_values
        matrices = self._make_oracle_matrices(k=3, m=6)
        exact_w, _ = compute_banzhaf_values(matrices)
        kernel_w, _ = kernel_banzhaf_values(matrices, n_samples=4096, seed=42)
        # Kernel estimator has sampling variance; 0.15 delta is expected
        # for 4096 samples with k=3 oracles.
        for name in exact_w:
            self.assertAlmostEqual(exact_w[name], kernel_w[name], delta=0.15)

    def test_identical_oracles_get_equal_banzhaf(self):
        """Two identical oracles should get equal Banzhaf values."""
        from tabnetics.core.mnpo import compute_banzhaf_values
        mat = np.random.RandomState(7).uniform(0.1, 0.9, (4, 4))
        np.fill_diagonal(mat, 0.5)
        matrices = {"a": mat.copy(), "b": mat.copy(), "ref": np.full((4, 4), 0.5)}
        weights, _ = compute_banzhaf_values(matrices, reference="ref")
        self.assertAlmostEqual(weights["a"], weights["b"], places=8)


if __name__ == "__main__":
    unittest.main()

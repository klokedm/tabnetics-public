import unittest

import numpy as np

from tabnetics.core.mnpo import (
    apply_oracle_redundancy_penalty,
    fold_regret_mean_max,
    lower_tail_cvar,
    matrix_from_scalar_scores,
    pairwise_pref_from_scalar,
    spearman_correlation,
    tremble_oracle_matrices,
)


class TestMNPOCoreOracles(unittest.TestCase):
    def test_pairwise_pref_tie_margin_returns_exact_tie(self):
        self.assertEqual(pairwise_pref_from_scalar(0.10, 0.11, tie_margin=0.02), 0.5)

    def test_pairwise_pref_temperature_smooths_preferences(self):
        # Same diff but different temperatures: higher temperature -> closer to 0.5.
        p_cold = pairwise_pref_from_scalar(1.0, 0.0, tie_margin=0.0, temperature=0.05)
        p_hot = pairwise_pref_from_scalar(1.0, 0.0, tie_margin=0.0, temperature=5.0)
        self.assertGreater(p_cold, p_hot)
        self.assertGreater(p_hot, 0.5)

    def test_lower_tail_cvar_matches_worst_k_mean(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        self.assertAlmostEqual(lower_tail_cvar(vals, alpha=0.50), 1.5)
        self.assertAlmostEqual(lower_tail_cvar(vals, alpha=0.0), 1.0)
        self.assertAlmostEqual(lower_tail_cvar(vals, alpha=1.0), 2.5)

    def test_fold_regret_mean_max_shapes(self):
        score_matrix = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.1, 0.3],
            ],
            dtype=float,
        )
        mean_regret, max_regret = fold_regret_mean_max(score_matrix)
        self.assertEqual(mean_regret.shape, (2,))
        self.assertEqual(max_regret.shape, (2,))
        self.assertTrue(np.all(mean_regret >= 0))
        self.assertTrue(np.all(max_regret >= 0))

    def test_spearman_correlation_basic(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [10.0, 20.0, 30.0, 40.0]
        c = [40.0, 30.0, 20.0, 10.0]
        self.assertAlmostEqual(spearman_correlation(a, b), 1.0, places=7)
        self.assertAlmostEqual(spearman_correlation(a, c), -1.0, places=7)
        self.assertEqual(spearman_correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]), 0.0)

    def test_apply_oracle_redundancy_penalty_zeroes_perfectly_redundant_weights(self):
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        scores = {
            "a": [0.1, 0.2, 0.3, 0.4],
            "b": [10.0, 20.0, 30.0, 40.0],  # perfectly rank-correlated with a
            "c": [0.2, 0.2, 0.2, 0.2],      # constant -> treated as rho=0 with others
        }
        new_weights, meta = apply_oracle_redundancy_penalty(weights, scores)
        self.assertIn("penalties", meta)
        self.assertAlmostEqual(float(new_weights["a"]), 0.0, places=7)
        self.assertAlmostEqual(float(new_weights["b"]), 0.0, places=7)
        self.assertAlmostEqual(float(new_weights["c"]), 1.0, places=7)

    def test_tremble_oracle_matrices_shrinks_toward_half(self):
        mat = np.array([[0.5, 1.0], [0.0, 0.5]], dtype=float)
        trembled = tremble_oracle_matrices({"x": mat}, epsilon=1.0)["x"]
        np.testing.assert_allclose(trembled, np.full((2, 2), 0.5), atol=0.0)

        trembled0 = tremble_oracle_matrices({"x": mat}, epsilon=0.0)["x"]
        np.testing.assert_allclose(trembled0, mat, atol=0.0)

    def test_matrix_from_scalar_scores_meta_temperature(self):
        scores = np.array([0.0, 1.0, 2.0], dtype=float)
        mat0, meta0 = matrix_from_scalar_scores(
            scores,
            tie_margin=0.01,
            use_qre_smoothing=False,
            qre_temperature_gamma=1.0,
        )
        self.assertIsNone(meta0.get("temperature"))
        self.assertEqual(mat0.shape, (3, 3))

        mat1, meta1 = matrix_from_scalar_scores(
            scores,
            tie_margin=0.01,
            use_qre_smoothing=True,
            qre_temperature_gamma=2.0,
        )
        self.assertIsInstance(meta1.get("temperature"), float)
        self.assertEqual(mat1.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()


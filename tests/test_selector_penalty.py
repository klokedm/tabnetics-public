"""Tests for T-R-243: MNPO selector penalty map."""

import unittest
import numpy as np


class TestSelectorPenaltyMap(unittest.TestCase):
    """Unit tests for selector penalty/exclusion logic."""

    def test_default_penalty_map_excludes_weak_selectors(self):
        """Default penalty map sets treeshap, dove_class_specific, sparse_multinomial to 0."""
        from tabnetics.feature_selection.config import MNPOConfig, DEFAULT_SELECTOR_PENALTY_MAP

        cfg = MNPOConfig()
        self.assertIsNotNone(cfg.selector_penalty_map)
        for name in ("treeshap", "dove_class_specific", "sparse_multinomial"):
            self.assertIn(name, cfg.selector_penalty_map)
            self.assertEqual(cfg.selector_penalty_map[name], 0.0)

    def test_custom_penalty_map_overrides_default(self):
        """User-supplied penalty map is used instead of default."""
        from tabnetics.feature_selection.config import MNPOConfig

        custom = {"treeshap": 0.5, "my_method": 0.0}
        cfg = MNPOConfig(selector_penalty_map=custom)
        self.assertEqual(cfg.selector_penalty_map, custom)
        self.assertNotIn("dove_class_specific", cfg.selector_penalty_map)

    def test_penalty_zero_excludes_from_weights(self):
        """A penalty of 0.0 zeros out the selector weight."""
        candidate_names = ["anova_f", "treeshap", "linear_svm"]
        p_star = np.array([0.4, 0.3, 0.3])
        penalty_map = {"treeshap": 0.0}

        for idx, name in enumerate(candidate_names):
            if name in penalty_map:
                p_star[idx] *= penalty_map[name]
        total = float(np.sum(p_star))
        if total > 0:
            p_star = p_star / total

        self.assertAlmostEqual(p_star[1], 0.0, places=10)
        self.assertAlmostEqual(float(np.sum(p_star)), 1.0, places=10)

    def test_penalty_partial_downweights(self):
        """A penalty < 1.0 reduces weight proportionally."""
        candidate_names = ["anova_f", "sparse_multinomial", "linear_svm"]
        p_star = np.array([0.4, 0.3, 0.3])
        penalty_map = {"sparse_multinomial": 0.5}

        original_weight = p_star[1]
        for idx, name in enumerate(candidate_names):
            if name in penalty_map:
                p_star[idx] *= penalty_map[name]
        total = float(np.sum(p_star))
        if total > 0:
            p_star = p_star / total

        # Weight should be reduced
        self.assertLess(p_star[1], original_weight)
        self.assertAlmostEqual(float(np.sum(p_star)), 1.0, places=10)

    def test_empty_penalty_map_no_effect(self):
        """Empty penalty map leaves weights unchanged."""
        candidate_names = ["anova_f", "linear_svm"]
        p_star = np.array([0.6, 0.4])
        penalty_map = {}
        original = p_star.copy()

        for idx, name in enumerate(candidate_names):
            if name in penalty_map:
                p_star[idx] *= penalty_map[name]

        np.testing.assert_allclose(p_star, original)

    def test_all_penalized_to_zero_falls_back_to_uniform(self):
        """If all candidates are zeroed out, fall back to uniform."""
        candidate_names = ["treeshap", "dove_class_specific"]
        p_star = np.array([0.5, 0.5])
        penalty_map = {"treeshap": 0.0, "dove_class_specific": 0.0}

        for idx, name in enumerate(candidate_names):
            if name in penalty_map:
                p_star[idx] *= penalty_map[name]
        total = float(np.sum(p_star))
        if total > 0:
            p_star = p_star / total
        else:
            p_star = np.full(len(p_star), 1.0 / max(1, len(p_star)))

        np.testing.assert_allclose(p_star, [0.5, 0.5])

    def test_default_penalty_map_constant_content(self):
        """DEFAULT_SELECTOR_PENALTY_MAP has exactly the expected keys."""
        from tabnetics.feature_selection.config import DEFAULT_SELECTOR_PENALTY_MAP

        expected_keys = {"treeshap", "dove_class_specific", "sparse_multinomial"}
        self.assertEqual(set(DEFAULT_SELECTOR_PENALTY_MAP.keys()), expected_keys)
        for v in DEFAULT_SELECTOR_PENALTY_MAP.values():
            self.assertEqual(v, 0.0)


if __name__ == "__main__":
    unittest.main()

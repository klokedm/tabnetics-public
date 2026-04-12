"""Tests for T-R-248: Oracle pruning presets."""

import unittest


class TestOraclePresets(unittest.TestCase):
    """Unit tests for OracleConfig.from_preset()."""

    def test_perf_only_disables_auxiliary_oracles(self):
        from tabnetics.feature_selection.config import OracleConfig
        cfg = OracleConfig.from_preset("perf_only")
        self.assertFalse(cfg.use_stability_oracle)
        self.assertFalse(cfg.use_complexity_oracle)
        self.assertFalse(cfg.use_robust_oracle)
        self.assertFalse(cfg.use_diversity_oracle)

    def test_perf_complexity_enables_only_two(self):
        from tabnetics.feature_selection.config import OracleConfig
        cfg = OracleConfig.from_preset("perf_complexity")
        self.assertTrue(cfg.use_complexity_oracle)
        self.assertFalse(cfg.use_stability_oracle)
        self.assertFalse(cfg.use_robust_oracle)
        self.assertFalse(cfg.use_diversity_oracle)

    def test_perf_complexity_stability_enables_three(self):
        from tabnetics.feature_selection.config import OracleConfig
        cfg = OracleConfig.from_preset("perf_complexity_stability")
        self.assertTrue(cfg.use_complexity_oracle)
        self.assertTrue(cfg.use_stability_oracle)
        self.assertFalse(cfg.use_robust_oracle)
        self.assertFalse(cfg.use_diversity_oracle)

    def test_full_preserves_defaults(self):
        from tabnetics.feature_selection.config import OracleConfig
        cfg = OracleConfig.from_preset("full")
        default = OracleConfig()
        self.assertEqual(cfg.use_stability_oracle, default.use_stability_oracle)
        self.assertEqual(cfg.use_complexity_oracle, default.use_complexity_oracle)
        self.assertEqual(cfg.use_robust_oracle, default.use_robust_oracle)
        self.assertEqual(cfg.use_diversity_oracle, default.use_diversity_oracle)

    def test_minimal_cvar_enables_cvar(self):
        from tabnetics.feature_selection.config import OracleConfig
        cfg = OracleConfig.from_preset("minimal_cvar")
        self.assertTrue(cfg.use_cvar)
        self.assertAlmostEqual(cfg.cvar_alpha, 0.33)
        self.assertFalse(cfg.use_stability_oracle)
        self.assertFalse(cfg.use_diversity_oracle)

    def test_overrides_applied(self):
        from tabnetics.feature_selection.config import OracleConfig
        cfg = OracleConfig.from_preset("perf_only", cvar_alpha=0.5)
        self.assertAlmostEqual(cfg.cvar_alpha, 0.5)

    def test_invalid_preset_raises(self):
        from tabnetics.feature_selection.config import OracleConfig
        with self.assertRaises(ValueError):
            OracleConfig.from_preset("nonexistent")

    def test_preset_names_case_insensitive(self):
        from tabnetics.feature_selection.config import OracleConfig
        cfg = OracleConfig.from_preset("PERF_ONLY")
        self.assertFalse(cfg.use_stability_oracle)


if __name__ == "__main__":
    unittest.main()

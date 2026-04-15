"""Tests for T-R-251: Banzhaf oracle weighting mode."""

import unittest
import warnings


class TestBanzhafWeightingConfig(unittest.TestCase):
    """Test Banzhaf weighting mode in OracleConfig."""

    def test_banzhaf_accepted_as_weighting_mode(self):
        from tabnetics.feature_selection.config import OracleConfig
        cfg = OracleConfig(weighting_mode="banzhaf")
        self.assertEqual(cfg.weighting_mode, "banzhaf")

    def test_banzhaf_mode_in_mnpo_config(self):
        from tabnetics.feature_selection.config import MNPOConfig
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = MNPOConfig(oracle_weighting_mode="banzhaf")
        self.assertEqual(cfg.oracle.weighting_mode, "banzhaf")

    def test_invalid_mode_falls_back_to_tritrust(self):
        from tabnetics.feature_selection.config import OracleConfig
        cfg = OracleConfig(weighting_mode="invalid_mode")
        self.assertEqual(cfg.weighting_mode, "tritrust")

    def test_all_valid_modes_accepted(self):
        from tabnetics.feature_selection.config import OracleConfig
        for mode in ("tritrust", "uniform", "shapley", "banzhaf"):
            cfg = OracleConfig(weighting_mode=mode)
            self.assertEqual(cfg.weighting_mode, mode)


if __name__ == "__main__":
    unittest.main()

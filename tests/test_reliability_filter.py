"""Tests for the reliability filter and bootstrap gate integration."""

import unittest

from tabnetics.validation.core.gates import GateConfig, GateResult, compute_gate, compute_bootstrap_gate


class TestReliabilityFilter(unittest.TestCase):
    """Verify that the reliability_max_seed_std filter works correctly."""

    TIER_MAP = {
        "e1": "easy",
        "e2": "easy",
        "m_stable": "medium",
        "m_noisy": "medium",
        "h1": "hard",
    }
    DELTAS = {
        "e1": [-0.01, -0.01, -0.01],
        "e2": [-0.01, -0.01, -0.01],
        "m_stable": [0.02, 0.02, 0.02],
        "m_noisy": [-0.08, -0.08, -0.08],  # fails medium gate
        "h1": [0.05, 0.05, 0.05],
    }
    BASELINE_STD = {
        "e1": 0.02,
        "e2": 0.03,
        "m_stable": 0.04,
        "m_noisy": 0.15,  # high variance → unreliable
        "h1": 0.06,
    }

    def test_strict_fails_without_filter(self) -> None:
        cfg = GateConfig(mode="strict")
        res = compute_gate(
            deltas_by_dataset=self.DELTAS,
            tier_by_dataset=self.TIER_MAP,
            config=cfg,
        )
        self.assertEqual(res.verdict, "FAIL")  # m_noisy causes failure
        self.assertEqual(res.reliability_excluded, ())

    def test_strict_passes_with_filter(self) -> None:
        cfg = GateConfig(mode="strict", reliability_max_seed_std=0.12)
        res = compute_gate(
            deltas_by_dataset=self.DELTAS,
            tier_by_dataset=self.TIER_MAP,
            config=cfg,
            baseline_seed_std_by_dataset=self.BASELINE_STD,
        )
        self.assertEqual(res.verdict, "PASS")
        self.assertIn("m_noisy", res.reliability_excluded)
        self.assertEqual(len(res.reliability_excluded), 1)

    def test_filter_disabled_when_threshold_zero(self) -> None:
        cfg = GateConfig(mode="strict", reliability_max_seed_std=0.0)
        res = compute_gate(
            deltas_by_dataset=self.DELTAS,
            tier_by_dataset=self.TIER_MAP,
            config=cfg,
            baseline_seed_std_by_dataset=self.BASELINE_STD,
        )
        self.assertEqual(res.verdict, "FAIL")
        self.assertEqual(res.reliability_excluded, ())

    def test_filter_disabled_when_no_std_data(self) -> None:
        cfg = GateConfig(mode="strict", reliability_max_seed_std=0.12)
        res = compute_gate(
            deltas_by_dataset=self.DELTAS,
            tier_by_dataset=self.TIER_MAP,
            config=cfg,
            baseline_seed_std_by_dataset=None,
        )
        self.assertEqual(res.verdict, "FAIL")
        self.assertEqual(res.reliability_excluded, ())

    def test_filter_high_threshold_excludes_nothing(self) -> None:
        cfg = GateConfig(mode="strict", reliability_max_seed_std=0.50)
        res = compute_gate(
            deltas_by_dataset=self.DELTAS,
            tier_by_dataset=self.TIER_MAP,
            config=cfg,
            baseline_seed_std_by_dataset=self.BASELINE_STD,
        )
        self.assertEqual(res.verdict, "FAIL")  # m_noisy still included
        self.assertEqual(res.reliability_excluded, ())


class TestBootstrapGateReliability(unittest.TestCase):
    """Verify that bootstrap gate respects the reliability filter."""

    def test_bootstrap_passes_when_noisy_dataset_excluded(self) -> None:
        tier_map = {
            "e1": "easy",
            "m1": "medium",
            "m_noisy": "medium",
            "h1": "hard",
            "h2": "hard",
        }
        deltas = {
            "e1": [0.02, 0.02, 0.02],
            "m1": [0.03, 0.03, 0.03],
            "m_noisy": [-0.08, -0.08, -0.08],
            "h1": [0.05, 0.05, 0.05],
            "h2": [0.05, 0.05, 0.05],
        }
        baseline_std = {
            "e1": 0.01,
            "m1": 0.03,
            "m_noisy": 0.20,
            "h1": 0.05,
            "h2": 0.04,
        }
        cfg = GateConfig(
            mode="bootstrap",
            reliability_max_seed_std=0.12,
            n_bootstrap=300,
            n_permutations=300,
            random_seed=42,
        )
        res = compute_bootstrap_gate(
            deltas_by_dataset=deltas,
            tier_by_dataset=tier_map,
            config=cfg,
            baseline_seed_std_by_dataset=baseline_std,
        )
        self.assertEqual(res.verdict, "PASS")
        self.assertIn("m_noisy", res.reliability_excluded)

    def test_bootstrap_without_filter_fails(self) -> None:
        tier_map = {
            "e1": "easy",
            "m1": "medium",
            "m_noisy": "medium",
            "h1": "hard",
            "h2": "hard",
        }
        deltas = {
            "e1": [0.02, 0.02, 0.02],
            "m1": [0.03, 0.03, 0.03],
            "m_noisy": [-0.08, -0.08, -0.08],
            "h1": [0.05, 0.05, 0.05],
            "h2": [0.05, 0.05, 0.05],
        }
        cfg = GateConfig(
            mode="bootstrap",
            reliability_max_seed_std=0.0,
            n_bootstrap=300,
            n_permutations=300,
            random_seed=42,
        )
        res = compute_bootstrap_gate(
            deltas_by_dataset=deltas,
            tier_by_dataset=tier_map,
            config=cfg,
        )
        self.assertEqual(res.verdict, "FAIL")
        self.assertEqual(res.reliability_excluded, ())


if __name__ == "__main__":
    unittest.main()

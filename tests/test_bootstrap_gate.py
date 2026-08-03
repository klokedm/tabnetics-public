import unittest

from tabnetics.validation.core.gates import GateConfig, compute_bootstrap_gate


class TestBootstrapGate(unittest.TestCase):
    def test_bootstrap_gate_pass_on_clear_signal(self) -> None:
        tier_by_dataset = {
            "e1": "easy",
            "e2": "easy",
            "m1": "medium",
            "m2": "medium",
            "h1": "hard",
            "h2": "hard",
        }
        deltas = {k: [0.20, 0.20, 0.20] for k in tier_by_dataset.keys()}
        cfg = GateConfig(
            mode="bootstrap",
            overall_threshold=0.0,
            catastrophic_veto_threshold=-0.10,
            n_bootstrap=300,
            n_permutations=300,
            random_seed=0,
        )
        res = compute_bootstrap_gate(deltas_by_dataset=deltas, tier_by_dataset=tier_by_dataset, config=cfg)
        self.assertEqual(res.verdict, "PASS")
        self.assertFalse(res.vetoed)
        self.assertTrue(res.effect_floor_passed)
        self.assertGreaterEqual(res.ci_worst_easy[0], -0.02)
        self.assertGreaterEqual(res.ci_worst_medium[0], -0.05)
        self.assertGreaterEqual(res.ci_hard_mean[0], 0.0)

    def test_bootstrap_gate_blocks_negligible_positive_effect(self) -> None:
        tier_by_dataset = {"e1": "easy", "m1": "medium", "h1": "hard"}
        deltas = {k: [0.0001, 0.0001, 0.0001] for k in tier_by_dataset.keys()}
        cfg = GateConfig(
            mode="bootstrap",
            hard_threshold=0.0,
            overall_threshold=0.0,
            min_practical_effect=0.01,
            catastrophic_veto_threshold=-0.10,
            n_bootstrap=200,
            n_permutations=200,
            random_seed=0,
        )
        res = compute_bootstrap_gate(deltas_by_dataset=deltas, tier_by_dataset=tier_by_dataset, config=cfg)
        self.assertEqual(res.verdict, "FAIL")
        self.assertFalse(res.effect_floor_passed)

    def test_bootstrap_gate_fails_on_catastrophic_veto(self) -> None:
        tier_by_dataset = {"e1": "easy", "m1": "medium", "h1": "hard"}
        deltas = {
            "e1": [-0.20, -0.20, -0.20],  # catastrophic
            "m1": [0.10, 0.10, 0.10],
            "h1": [0.10, 0.10, 0.10],
        }
        cfg = GateConfig(
            mode="bootstrap",
            catastrophic_veto_threshold=-0.10,
            overall_threshold=0.0,
            n_bootstrap=200,
            n_permutations=200,
            random_seed=0,
        )
        res = compute_bootstrap_gate(deltas_by_dataset=deltas, tier_by_dataset=tier_by_dataset, config=cfg)
        self.assertEqual(res.verdict, "FAIL")
        self.assertTrue(res.vetoed)

    def test_bootstrap_gate_requires_overall_nonnegative_ci(self) -> None:
        tier_by_dataset = {"e1": "easy", "m1": "medium", "h1": "hard", "h2": "hard"}
        deltas = {
            "e1": [-0.02, -0.02, -0.02],
            "m1": [-0.02, -0.02, -0.02],
            "h1": [0.00, 0.00, 0.00],
            "h2": [0.00, 0.00, 0.00],
        }
        # Overall mean = -0.01, so CI lower bound should also be < 0 -> FAIL.
        cfg = GateConfig(
            mode="bootstrap",
            overall_threshold=0.0,
            catastrophic_veto_threshold=-0.10,
            n_bootstrap=400,
            n_permutations=200,
            random_seed=0,
        )
        res = compute_bootstrap_gate(deltas_by_dataset=deltas, tier_by_dataset=tier_by_dataset, config=cfg)
        self.assertEqual(res.verdict, "FAIL")
        self.assertLess(res.ci_overall_mean[0], 0.0)


if __name__ == "__main__":
    unittest.main()

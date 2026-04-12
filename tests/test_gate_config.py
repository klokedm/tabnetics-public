import unittest

from tabnetics.validation.core.gates import GateConfig, compute_gate
from tabnetics.validation.core.report import build_side_by_side_gate_report


class TestGateConfig(unittest.TestCase):
    def test_strict_gate_pass(self) -> None:
        tier_by_dataset = {
            "e1": "easy",
            "m1": "medium",
            "h1": "hard",
        }
        deltas = {
            "e1": [-0.01, -0.01, -0.01],
            "m1": [-0.02, -0.01, -0.01],  # mean -0.0133 >= -0.05
            "h1": [0.00, 0.01, 0.00],     # mean 0.0033 >= 0
        }
        res = compute_gate(deltas_by_dataset=deltas, tier_by_dataset=tier_by_dataset, config=GateConfig(mode="strict"))
        self.assertEqual(res.verdict, "PASS")

    def test_strict_gate_fail_on_worst_easy(self) -> None:
        tier_by_dataset = {"e1": "easy", "m1": "medium", "h1": "hard"}
        deltas = {"e1": [-0.03], "m1": [0.0], "h1": [0.01]}
        res = compute_gate(deltas_by_dataset=deltas, tier_by_dataset=tier_by_dataset, config=GateConfig(mode="strict"))
        self.assertEqual(res.verdict, "FAIL")

    def test_quantile_gate_trims_outlier(self) -> None:
        tier_by_dataset = {"e_bad": "easy", "e_ok": "easy", "m1": "medium", "h1": "hard"}
        deltas = {
            "e_bad": [-0.03, -0.03, -0.03],
            "e_ok": [-0.01, -0.01, -0.01],
            "m1": [-0.01, -0.01, -0.01],
            "h1": [0.10, 0.10, 0.10],
        }
        cfg = GateConfig(mode="quantile", easy_trim=1, medium_trim=0, overall_threshold=0.0)
        res = compute_gate(deltas_by_dataset=deltas, tier_by_dataset=tier_by_dataset, config=cfg)
        self.assertEqual(res.verdict, "PASS")
        self.assertEqual(len(res.trimmed_easy), 1)
        self.assertEqual(res.trimmed_easy[0][0], "e_bad")

    def test_quantile_gate_catastrophic_veto(self) -> None:
        tier_by_dataset = {"e1": "easy", "m1": "medium", "h1": "hard"}
        deltas = {
            "e1": [-0.12, -0.11, -0.09],  # catastrophic_min = -0.12
            "m1": [0.0, 0.0, 0.0],
            "h1": [0.10, 0.10, 0.10],
        }
        cfg = GateConfig(mode="quantile", easy_trim=1, medium_trim=0, catastrophic_veto_threshold=-0.10, overall_threshold=0.0)
        res = compute_gate(deltas_by_dataset=deltas, tier_by_dataset=tier_by_dataset, config=cfg)
        self.assertEqual(res.verdict, "FAIL")
        self.assertTrue(res.vetoed)

    def test_quantile_gate_requires_overall_nonnegative(self) -> None:
        tier_by_dataset = {"e1": "easy", "e2": "easy", "m1": "medium", "h1": "hard"}
        deltas = {
            "e1": [-0.03, -0.03, -0.03],  # will be trimmed
            "e2": [-0.01, -0.01, -0.01],  # passes easy threshold post-trim
            "m1": [-0.01, -0.01, -0.01],
            "h1": [0.02, 0.02, 0.02],
        }
        # Overall mean is negative: (-0.03 + -0.01 + -0.01 + 0.02)/4 = -0.0075
        cfg = GateConfig(mode="quantile", easy_trim=1, medium_trim=0, overall_threshold=0.0)
        res = compute_gate(deltas_by_dataset=deltas, tier_by_dataset=tier_by_dataset, config=cfg)
        self.assertEqual(res.verdict, "FAIL")

    def test_side_by_side_report_keeps_strict_as_primary(self) -> None:
        tier_by_dataset = {"e1": "easy", "m1": "medium", "h1": "hard"}
        deltas = {
            "e1": [-0.01, -0.01, -0.01],
            "m1": [-0.02, -0.01, -0.01],
            "h1": [0.00, 0.01, 0.00],
        }
        report = build_side_by_side_gate_report(
            deltas_by_dataset=deltas,
            tier_by_dataset=tier_by_dataset,
        )
        self.assertEqual(report["primary_mode"], "strict")
        self.assertEqual(report["promotion_decision"]["mode_used"], "strict")
        self.assertEqual(report["promotion_decision"]["verdict"], "PASS")
        self.assertIn("strict", report)
        self.assertIn("quantile", report)

    def test_side_by_side_report_quantile_is_advisory(self) -> None:
        tier_by_dataset = {"e_bad": "easy", "e_ok": "easy", "m1": "medium", "h1": "hard"}
        deltas = {
            "e_bad": [-0.03, -0.03, -0.03],
            "e_ok": [-0.01, -0.01, -0.01],
            "m1": [-0.01, -0.01, -0.01],
            "h1": [0.10, 0.10, 0.10],
        }
        report = build_side_by_side_gate_report(
            deltas_by_dataset=deltas,
            tier_by_dataset=tier_by_dataset,
            quantile_config=GateConfig(mode="quantile", easy_trim=1, medium_trim=0, overall_threshold=0.0),
        )
        self.assertEqual(report["strict"]["verdict"], "FAIL")
        self.assertEqual(report["quantile"]["verdict"], "PASS")
        self.assertFalse(report["promotion_decision"]["promote"])


if __name__ == "__main__":
    unittest.main()

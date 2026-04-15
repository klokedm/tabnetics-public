"""Tests for IPSS benchmark evaluation (VAL12_Suggestions §5.3)."""

import numpy as np
import pytest

from tabnetics.feature_selection.methods.embedded import ipss_benchmark_evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ipss_result(n_selected=5, n_features=50):
    """Create a synthetic IPSS result dict for testing."""
    selected = np.arange(n_selected)
    sel_freq = np.zeros(n_features)
    sel_freq[:n_selected] = np.linspace(0.9, 0.6, n_selected)
    sel_freq[n_selected:n_selected + 5] = np.linspace(0.3, 0.1, 5)
    q_vals = np.ones(n_features)
    q_vals[:n_selected] = np.linspace(0.02, 0.10, n_selected)
    q_vals[n_selected:n_selected + 5] = np.linspace(0.3, 0.6, 5)

    return {
        "selected_indices": selected,
        "scores": {int(i): float(sel_freq[i]) for i in selected},
        "selection_frequency": sel_freq,
        "stability_score": sel_freq * 0.8,
        "selection_frequency_max": sel_freq,
        "q_values": q_vals,
        "n_fits": 100,
        "pool_size": 30,
        "path_grid": [0.1, 0.5, 1.0],
        "path_fit_counts": [50, 50, 50],
        "stable_threshold": 0.55,
        "target_fdr": 0.15,
        "n_null_scores": 20,
        "ipss_importance_model": "lasso",
        "ipss_use_eats_threshold": False,
        "ipss_gated": False,
        "ipss_gate_reason": "",
        "ipss_gate_min_classes": 0,
        "ipss_gate_min_p_over_n": 0.0,
        "ipss_gate_n_classes": 3,
        "ipss_gate_p_over_n": 5.0,
        "eats_exclusion_floor": float("nan"),
        "eats_elbow_threshold": float("nan"),
        "eats_n_threshold_candidates": 0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIPSSBenchmarkEvaluate:
    """Verify IPSS benchmark evaluation diagnostics."""

    def test_basic_report_structure(self):
        result = _make_ipss_result()
        report = ipss_benchmark_evaluate(result)
        assert "selected_count" in report
        assert "stable_threshold" in report
        assert "target_fdr" in report
        assert "pool_size" in report
        assert "n_fits" in report
        assert "selection_frequency_stats" in report
        assert "selected_q_value_stats" in report

    def test_selected_count(self):
        result = _make_ipss_result(n_selected=7)
        report = ipss_benchmark_evaluate(result)
        assert report["selected_count"] == 7

    def test_threshold_and_fdr(self):
        result = _make_ipss_result()
        report = ipss_benchmark_evaluate(result)
        assert report["stable_threshold"] == 0.55
        assert report["target_fdr"] == 0.15

    def test_selection_frequency_stats(self):
        result = _make_ipss_result(n_selected=5)
        report = ipss_benchmark_evaluate(result)
        stats = report["selection_frequency_stats"]
        assert "mean" in stats
        assert "median" in stats
        assert "max" in stats
        assert "n_nonzero" in stats
        assert stats["n_nonzero"] == 10  # 5 selected + 5 noise with freq > 0
        assert stats["max"] == pytest.approx(0.9, abs=0.01)

    def test_q_value_stats(self):
        result = _make_ipss_result(n_selected=5)
        report = ipss_benchmark_evaluate(result)
        q_stats = report["selected_q_value_stats"]
        assert "mean" in q_stats
        assert "fdr_estimate" in q_stats
        assert q_stats["fdr_estimate"] < 0.15  # selected features have low q

    def test_balanced_accuracy_with_labels(self):
        result = _make_ipss_result()
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1])
        report = ipss_benchmark_evaluate(result, y_true=y_true, y_pred=y_pred)
        assert "balanced_accuracy" in report
        assert 0.0 <= report["balanced_accuracy"] <= 1.0

    def test_no_labels_no_ba(self):
        result = _make_ipss_result()
        report = ipss_benchmark_evaluate(result)
        assert "balanced_accuracy" not in report

    def test_empty_result(self):
        report = ipss_benchmark_evaluate({})
        assert "error" in report

    def test_none_result(self):
        report = ipss_benchmark_evaluate(None)
        assert "error" in report

    def test_gated_result(self):
        result = {
            "selected_indices": np.array([], dtype=int),
            "scores": {},
            "method": "ipss",
            "ipss_gated": True,
            "ipss_gate_reason": "gate_not_satisfied",
            "ipss_gate_min_classes": 3,
            "ipss_gate_min_p_over_n": 0.0,
            "ipss_gate_n_classes": 2,
            "ipss_gate_p_over_n": 1.0,
        }
        report = ipss_benchmark_evaluate(result)
        assert report["selected_count"] == 0
        assert report["ipss_gated"] is True
        assert report["ipss_gate_reason"] == "gate_not_satisfied"

    def test_eats_metadata(self):
        result = _make_ipss_result()
        result["ipss_use_eats_threshold"] = True
        result["eats_elbow_threshold"] = 0.65
        report = ipss_benchmark_evaluate(result)
        assert report["ipss_use_eats_threshold"] is True
        assert report["eats_elbow_threshold"] == 0.65

    def test_no_selection_frequency(self):
        result = _make_ipss_result()
        del result["selection_frequency"]
        report = ipss_benchmark_evaluate(result)
        assert report["selection_frequency_stats"] == {}

    def test_no_q_values(self):
        result = _make_ipss_result()
        del result["q_values"]
        report = ipss_benchmark_evaluate(result)
        assert report["selected_q_value_stats"] == {}


# ---------------------------------------------------------------------------
# Config toggle test
# ---------------------------------------------------------------------------


class TestBenchmarkIPSSConfig:
    """Verify benchmark_ipss_enabled config field exists."""

    def test_config_field_default_false(self):
        from tabnetics.pipeline.pipeline import DFFSConfig
        cfg = DFFSConfig()
        assert hasattr(cfg, "benchmark_ipss_enabled")
        assert cfg.benchmark_ipss_enabled is False

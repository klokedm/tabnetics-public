from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from tabnetics.benchmarks import gaming_detectors as gd


def _base_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4"],
            "seed": [11, 23, 11, 23, 11, 23, 11, 23],
            "tier": ["easy", "easy", "medium", "medium", "hard", "hard", "hard", "hard"],
            "balanced_accuracy": [0.70, 0.71, 0.65, 0.66, 0.72, 0.73, 0.69, 0.68],
            "delta_balanced_accuracy": [-0.01, -0.01, -0.02, -0.01, 0.03, 0.03, 0.01, 0.01],
            "class_count": [2, 2, 3, 3, 6, 6, 8, 8],
        }
    )


def test_class_correlation_flags_strong_monotonic_pattern():
    rows = _base_rows().copy()
    rows["delta_balanced_accuracy"] = np.linspace(-0.2, 0.2, num=len(rows))
    rows["class_count"] = np.arange(2, 2 + len(rows))
    out = gd.detect_class_correlation(rows)
    assert out["status"] == "ok"
    assert out["flagged"] is True


def test_class_correlation_not_flagged_when_weak():
    rows = _base_rows().copy()
    rows["class_count"] = [2, 8, 3, 7, 4, 6, 5, 9]
    out = gd.detect_class_correlation(rows)
    assert out["status"] == "ok"
    assert out["flagged"] is False


def test_class_correlation_handles_missing_columns():
    out = gd.detect_class_correlation(pd.DataFrame({"delta_balanced_accuracy": [0.1, 0.2]}))
    assert out["status"] == "insufficient_columns"
    assert out["flagged"] is False


def test_tier_imbalance_flags_hard_only_uplift():
    rows = _base_rows().copy()
    rows.loc[rows["tier"] == "easy", "delta_balanced_accuracy"] = -0.04
    rows.loc[rows["tier"] == "medium", "delta_balanced_accuracy"] = -0.06
    rows.loc[rows["tier"] == "hard", "delta_balanced_accuracy"] = 0.05
    out = gd.detect_tier_imbalance(rows)
    assert out["flagged"] is True


def test_tier_imbalance_not_flagged_for_balanced_tiers():
    rows = _base_rows().copy()
    rows["delta_balanced_accuracy"] = 0.01
    out = gd.detect_tier_imbalance(rows)
    assert out["flagged"] is False


def test_tier_imbalance_handles_missing_columns():
    out = gd.detect_tier_imbalance(pd.DataFrame({"tier": ["easy", "hard"]}))
    assert out["status"] == "insufficient_columns"
    assert out["flagged"] is False


def test_threshold_hugging_flags_dense_gate_boundary_cluster():
    rows = pd.DataFrame(
        {
            "tier": ["easy"] * 10 + ["hard"] * 10,
            "delta_balanced_accuracy": [-0.02 + 0.0005 * ((i % 3) - 1) for i in range(10)] + [0.01] * 10,
        }
    )
    out = gd.detect_threshold_hugging(rows, min_count_per_tier=5)
    assert out["flagged"] is True


def test_threshold_hugging_not_flagged_when_spread():
    rows = pd.DataFrame(
        {
            "tier": ["easy"] * 10,
            "delta_balanced_accuracy": np.linspace(-0.10, 0.10, num=10),
        }
    )
    out = gd.detect_threshold_hugging(rows, min_count_per_tier=5)
    assert out["flagged"] is False


def test_threshold_hugging_handles_missing_columns():
    out = gd.detect_threshold_hugging(pd.DataFrame({"delta_balanced_accuracy": [0.1]}))
    assert out["status"] == "insufficient_columns"
    assert out["flagged"] is False


def test_seed_variance_flags_high_variance_dataset():
    rows = _base_rows().copy()
    rows.loc[rows["dataset_id"] == "d3", "balanced_accuracy"] = [0.40, 0.95]
    out = gd.detect_seed_variance(rows, std_threshold=0.10)
    assert out["flagged"] is True
    assert "d3" in out["high_variance_datasets"]


def test_seed_variance_not_flagged_when_stable():
    rows = _base_rows().copy()
    out = gd.detect_seed_variance(rows, std_threshold=0.20)
    assert out["flagged"] is False


def test_seed_variance_handles_missing_columns():
    out = gd.detect_seed_variance(pd.DataFrame({"dataset_id": ["d1"], "balanced_accuracy": [0.1]}))
    assert out["status"] == "insufficient_columns"
    assert out["flagged"] is False


def test_bellwether_concentration_flags_when_top_dataset_dominates():
    rows = _base_rows().copy()
    rows.loc[rows["dataset_id"] == "d1", "delta_balanced_accuracy"] = 0.20
    rows.loc[rows["dataset_id"] != "d1", "delta_balanced_accuracy"] = 0.01
    out = gd.detect_bellwether_concentration(rows, top_fraction=0.25, share_threshold=0.60)
    assert out["flagged"] is True
    assert out["top_share"] > 0.60


def test_bellwether_concentration_not_flagged_when_uplift_is_distributed():
    rows = _base_rows().copy()
    rows["delta_balanced_accuracy"] = 0.05
    out = gd.detect_bellwether_concentration(rows, top_fraction=0.25, share_threshold=0.90)
    assert out["flagged"] is False


def test_bellwether_concentration_handles_no_positive_uplift():
    rows = _base_rows().copy()
    rows["delta_balanced_accuracy"] = -0.01
    out = gd.detect_bellwether_concentration(rows)
    assert out["status"] == "no_positive_uplift"
    assert out["flagged"] is False


def test_run_gaming_detectors_returns_all_gt_sections():
    rows = _base_rows()
    out = gd.run_gaming_detectors(rows)
    assert "gt1_class_correlation" in out
    assert "gt2_tier_imbalance" in out
    assert "gt3_threshold_hugging" in out
    assert "gt4_seed_variance" in out
    assert "gt5_bellwether" in out
    assert isinstance(out["any_flagged"], bool)


def test_run_gaming_detectors_accepts_list_of_dict_rows():
    rows = _base_rows().to_dict(orient="records")
    out = gd.run_gaming_detectors(rows)
    assert isinstance(out, dict)
    assert "any_flagged" in out


def test_module_is_read_only_overlay_without_feature_selection_imports():
    src = inspect.getsource(gd)
    assert "feature_selection" not in src

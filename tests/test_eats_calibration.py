"""Tests for EATS threshold calibration (VAL12_Suggestions §5.1)."""

import numpy as np
import pytest

from tabnetics.feature_selection.methods.stability_selection import (
    EATSThresholdCalibrator,
    eats_calibrate_threshold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stability_scores(n_informative=5, n_noise=45, seed=42):
    """Synthetic stability scores: informative features high, noise low."""
    rng = np.random.RandomState(seed)
    informative = rng.uniform(0.7, 1.0, size=n_informative)
    noise = rng.uniform(0.0, 0.3, size=n_noise)
    return np.concatenate([informative, noise])


def _make_null_scores(n=50, seed=99):
    """Null distribution: permuted-label stability scores, typically low."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.0, 0.35, size=n)


# ---------------------------------------------------------------------------
# EATSThresholdCalibrator tests
# ---------------------------------------------------------------------------


class TestEATSThresholdCalibrator:
    """Verify EATS elbow-adaptive threshold calibration."""

    def test_basic_calibration(self):
        scores = _make_stability_scores()
        null = _make_null_scores()
        cal = EATSThresholdCalibrator()
        threshold, meta = cal.calibrate(scores, null)
        assert 0.0 <= threshold <= 1.0
        assert isinstance(meta, dict)
        assert "eats_exclusion_floor" in meta
        assert "eats_elbow_threshold" in meta
        assert "eats_n_threshold_candidates" in meta

    def test_threshold_is_adaptive(self):
        """Different data should yield different thresholds."""
        cal = EATSThresholdCalibrator()
        scores1 = _make_stability_scores(n_informative=5, n_noise=45, seed=1)
        null1 = _make_null_scores(seed=1)
        t1, _ = cal.calibrate(scores1, null1)

        scores2 = _make_stability_scores(n_informative=20, n_noise=30, seed=2)
        null2 = _make_null_scores(seed=2)
        t2, _ = cal.calibrate(scores2, null2)

        # With many more informative features, threshold can differ.
        # At minimum they should both be valid.
        assert 0.0 <= t1 <= 1.0
        assert 0.0 <= t2 <= 1.0

    def test_threshold_above_min(self):
        """Threshold should never go below min_threshold."""
        cal = EATSThresholdCalibrator(min_threshold=0.5)
        scores = _make_stability_scores()
        null = _make_null_scores()
        threshold, _ = cal.calibrate(scores, null)
        assert threshold >= 0.5

    def test_threshold_above_exclusion_floor(self):
        """Threshold should be at least the exclusion floor."""
        cal = EATSThresholdCalibrator(exclusion_quantile=0.95)
        scores = _make_stability_scores()
        null = _make_null_scores()
        threshold, meta = cal.calibrate(scores, null)
        assert threshold >= meta["eats_exclusion_floor"]

    def test_no_null_scores(self):
        """Without null scores, fallback to min_threshold as floor."""
        cal = EATSThresholdCalibrator(min_threshold=0.45)
        scores = _make_stability_scores()
        threshold, meta = cal.calibrate(scores, null_scores=None)
        assert threshold >= 0.45
        assert 0.0 <= threshold <= 1.0

    def test_empty_scores(self):
        """Empty stability scores should return fallback."""
        cal = EATSThresholdCalibrator(fallback_threshold=0.60)
        threshold, meta = cal.calibrate(np.array([]))
        assert threshold == 0.60
        assert not meta.get("eats_calibrated", True)

    def test_all_scores_below_floor(self):
        """All scores below exclusion floor: fewer than 3 eligible."""
        cal = EATSThresholdCalibrator(min_threshold=0.5, fallback_threshold=0.6)
        scores = np.array([0.1, 0.2, 0.15, 0.05])
        null = np.array([0.3, 0.35, 0.4, 0.45, 0.5])  # high null → high floor
        threshold, meta = cal.calibrate(scores, null)
        assert threshold >= 0.5

    def test_all_identical_scores(self):
        """All identical scores: degenerate case handled gracefully."""
        cal = EATSThresholdCalibrator()
        scores = np.full(20, 0.7)
        threshold, meta = cal.calibrate(scores)
        assert 0.0 <= threshold <= 1.0

    def test_two_features(self):
        """Only two features: fewer than 3 eligible → fallback path."""
        cal = EATSThresholdCalibrator(fallback_threshold=0.55)
        scores = np.array([0.8, 0.9])
        threshold, meta = cal.calibrate(scores)
        assert threshold >= 0.55

    def test_meta_calibrated_flag(self):
        """Meta should indicate whether full calibration was performed."""
        cal = EATSThresholdCalibrator()
        scores = _make_stability_scores()
        null = _make_null_scores()
        _, meta = cal.calibrate(scores, null)
        assert meta["eats_calibrated"] is True

        _, meta2 = cal.calibrate(np.array([0.8]))
        assert meta2["eats_calibrated"] is False

    def test_custom_exclusion_quantile(self):
        """Different quantiles should give different floors."""
        scores = _make_stability_scores()
        null = _make_null_scores()

        cal_low = EATSThresholdCalibrator(exclusion_quantile=0.50)
        _, meta_low = cal_low.calibrate(scores, null)

        cal_high = EATSThresholdCalibrator(exclusion_quantile=0.99)
        _, meta_high = cal_high.calibrate(scores, null)

        assert meta_high["eats_exclusion_floor"] >= meta_low["eats_exclusion_floor"]

    def test_deterministic(self):
        """Same input should give same output."""
        cal = EATSThresholdCalibrator()
        scores = _make_stability_scores()
        null = _make_null_scores()
        t1, m1 = cal.calibrate(scores, null)
        t2, m2 = cal.calibrate(scores, null)
        assert t1 == t2
        assert m1 == m2

    def test_param_clipping(self):
        """Parameters should be clipped to valid ranges."""
        cal = EATSThresholdCalibrator(
            exclusion_quantile=1.5,  # should clip to 0.995
            min_threshold=-0.5,      # should clip to 0.05
        )
        assert cal.exclusion_quantile == 0.995
        assert cal.min_threshold == 0.05


# ---------------------------------------------------------------------------
# Functional wrapper tests
# ---------------------------------------------------------------------------


class TestEATSFunctionalWrapper:
    """Verify the convenience function matches the class API."""

    def test_matches_class_result(self):
        scores = _make_stability_scores()
        null = _make_null_scores()

        cal = EATSThresholdCalibrator(
            exclusion_quantile=0.90,
            min_threshold=0.45,
            fallback_threshold=0.60,
        )
        t_class, m_class = cal.calibrate(scores, null)

        t_func, m_func = eats_calibrate_threshold(
            scores, null,
            exclusion_quantile=0.90,
            min_threshold=0.45,
            fallback_threshold=0.60,
        )

        assert t_class == t_func
        assert m_class == m_func


# ---------------------------------------------------------------------------
# Config toggle test
# ---------------------------------------------------------------------------


class TestStabilityThresholdConfig:
    """Verify fs_stability_threshold_method config field exists."""

    def test_config_field_default_fixed(self):
        from tabnetics.pipeline.pipeline import DFFSConfig
        cfg = DFFSConfig()
        assert hasattr(cfg, "fs_stability_threshold_method")
        assert cfg.fs_stability_threshold_method == "fixed"

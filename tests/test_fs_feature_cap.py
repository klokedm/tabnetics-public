"""Tests for the post-FS feature count safety cap (VAL12_Suggestions / Independent Review E.1)."""

import numpy as np
import pytest

from tabnetics.pipeline.pipeline import DFFSConfig


class TestFSFeatureCapConfig:
    """Verify config fields exist and have correct defaults."""

    def test_default_ratio(self):
        cfg = DFFSConfig()
        assert cfg.fs_max_selected_features_ratio == 0.5

    def test_default_cap(self):
        cfg = DFFSConfig()
        assert cfg.fs_max_selected_features_cap == 500

    def test_custom_values(self):
        cfg = DFFSConfig(fs_max_selected_features_ratio=0.3, fs_max_selected_features_cap=200)
        assert cfg.fs_max_selected_features_ratio == 0.3
        assert cfg.fs_max_selected_features_cap == 200


class TestFSFeatureCapLogic:
    """Test the safety cap arithmetic without running the full pipeline."""

    @staticmethod
    def _compute_fs_max(n_train: int, ratio: float, cap: int) -> int:
        """Replicate the cap computation from df_fs_pipeline.py."""
        return int(min(
            max(1, n_train * ratio),
            max(1, cap),
        ))

    def test_cap_calculation_small_dataset(self):
        # n_train=50, ratio=0.5 → n_train*ratio=25; cap=500 → min(25,500)=25
        assert self._compute_fs_max(50, 0.5, 500) == 25

    def test_cap_calculation_large_dataset(self):
        # n_train=2000, ratio=0.5 → n_train*ratio=1000; cap=500 → min(1000,500)=500
        assert self._compute_fs_max(2000, 0.5, 500) == 500

    def test_cap_calculation_custom_ratio(self):
        # n_train=100, ratio=0.3 → 30; cap=500 → min(30,500)=30
        assert self._compute_fs_max(100, 0.3, 500) == 30

    def test_cap_calculation_custom_cap(self):
        # n_train=1000, ratio=0.5 → 500; cap=200 → min(500,200)=200
        assert self._compute_fs_max(1000, 0.5, 200) == 200

    def test_cap_minimum_is_one(self):
        # Even with very small values, cap is at least 1
        assert self._compute_fs_max(1, 0.01, 1) >= 1
        assert self._compute_fs_max(0, 0.5, 500) >= 1

    def test_cap_trimming_by_importance(self):
        """Verify that when cap triggers, top-importance features are kept."""
        rng = np.random.RandomState(42)
        n_sel = 100
        fs_max = 25
        # Simulate feature importance: features 10,20,30,...,90 have highest importance
        scores = rng.rand(n_sel)
        top_indices = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        scores[top_indices] = 10.0  # boost these

        keep_cols = np.argsort(scores)[::-1][:fs_max]
        keep_cols = np.sort(keep_cols)

        # All top-importance indices should be kept
        for idx in top_indices:
            assert idx in keep_cols, f"High-importance feature {idx} should be kept"

    def test_cap_preserves_column_order(self):
        """Verify kept columns maintain their original order."""
        n_sel = 50
        fs_max = 10
        scores = np.arange(n_sel, dtype=float)  # 0,1,...,49
        keep_cols = np.argsort(scores)[::-1][:fs_max]
        keep_cols = np.sort(keep_cols)
        # Should be [40,41,...,49] (top-10 by ascending score)
        np.testing.assert_array_equal(keep_cols, np.arange(40, 50))
        # Verify sorted order
        assert all(keep_cols[i] < keep_cols[i + 1] for i in range(len(keep_cols) - 1))

    def test_cap_no_trigger_when_under_limit(self):
        """When n_selected <= fs_max, cap should NOT trigger."""
        n_train = 100
        ratio = 0.5
        cap = 500
        fs_max = self._compute_fs_max(n_train, ratio, cap)  # 50
        n_sel = 30  # under limit
        assert n_sel <= fs_max

    def test_variance_fallback(self):
        """When no importance is available, variance is used as proxy."""
        rng = np.random.RandomState(42)
        X = rng.randn(20, 50)
        # Make first 5 features constant (zero variance)
        X[:, :5] = 0.0
        # Make features 45-49 high variance
        X[:, 45:50] *= 100.0

        scores = np.var(X, axis=0)
        fs_max = 10
        keep_cols = np.argsort(scores)[::-1][:fs_max]
        keep_cols = np.sort(keep_cols)

        # High-variance features should be kept
        for idx in range(45, 50):
            assert idx in keep_cols, f"High-variance feature {idx} should be kept"
        # Zero-variance features should NOT be kept
        for idx in range(5):
            assert idx not in keep_cols, f"Zero-variance feature {idx} should be dropped"

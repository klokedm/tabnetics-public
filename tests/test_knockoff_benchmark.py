"""Tests for knockpy/GRIP2 benchmark selector (VAL12_Suggestions §5.2)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from tabnetics.feature_selection.methods.knockoff_benchmark import (
    KnockoffBenchmarkSelector,
    knockoff_benchmark_selection,
    KNOCKPY_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_data(n=100, p=20, n_informative=5, seed=42):
    """Create synthetic data with known informative features."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    beta = np.zeros(p)
    beta[:n_informative] = rng.uniform(1.0, 3.0, size=n_informative)
    logits = X @ beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (prob > 0.5).astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Fallback tests (no knockpy needed)
# ---------------------------------------------------------------------------


class TestKnockoffFallback:
    """Verify behavior when knockpy is unavailable."""

    def test_unavailable_returns_empty(self):
        X, y = _make_binary_data()
        with patch(
            "tabnetics.feature_selection.methods.knockoff_benchmark.KNOCKPY_AVAILABLE",
            False,
        ):
            selector = KnockoffBenchmarkSelector(fdr=0.10, random_state=42)
            with pytest.warns(RuntimeWarning, match="knockpy not installed"):
                result, meta = selector.select(X, y)
        assert result["selected_count"] == 0
        assert result["selected_indices"].size == 0
        assert meta["knockoff_backend"] == "unavailable"

    def test_empty_data(self):
        selector = KnockoffBenchmarkSelector(random_state=42)
        with patch(
            "tabnetics.feature_selection.methods.knockoff_benchmark.KNOCKPY_AVAILABLE",
            True,
        ):
            result, meta = selector.select(np.zeros((1, 0)), np.array([0]))
        assert result["selected_count"] == 0

    def test_single_sample(self):
        selector = KnockoffBenchmarkSelector(random_state=42)
        with patch(
            "tabnetics.feature_selection.methods.knockoff_benchmark.KNOCKPY_AVAILABLE",
            True,
        ):
            result, meta = selector.select(np.zeros((1, 5)), np.array([0]))
        assert result["selected_count"] == 0
        assert meta["knockoff_backend"] == "skipped"

    def test_functional_wrapper_unavailable(self):
        with patch(
            "tabnetics.feature_selection.methods.knockoff_benchmark.KNOCKPY_AVAILABLE",
            False,
        ):
            X, y = _make_binary_data(n=50, p=10)
            with pytest.warns(RuntimeWarning):
                result, meta = knockoff_benchmark_selection(
                    X, y, fdr=0.10, random_state=42,
                )
        assert result["selected_count"] == 0

    def test_config_default(self):
        selector = KnockoffBenchmarkSelector()
        assert selector.fdr == 0.10
        assert selector.knockoff_type == "gaussian"
        assert selector.statistic == "lasso"
        assert selector.random_state is None

    def test_result_structure(self):
        X, y = _make_binary_data()
        with patch(
            "tabnetics.feature_selection.methods.knockoff_benchmark.KNOCKPY_AVAILABLE",
            False,
        ):
            selector = KnockoffBenchmarkSelector(fdr=0.20, random_state=0)
            with pytest.warns(RuntimeWarning):
                result, meta = selector.select(X, y)
        assert "selected_indices" in result
        assert "selected_count" in result
        assert "fdr" in meta

    def test_error_handling(self):
        """Simulate knockpy raising an exception."""
        X, y = _make_binary_data(n=50, p=10)
        selector = KnockoffBenchmarkSelector(fdr=0.10, random_state=42)
        with patch(
            "tabnetics.feature_selection.methods.knockoff_benchmark.KNOCKPY_AVAILABLE",
            True,
        ):
            with patch.object(
                selector, "_run_knockpy", side_effect=RuntimeError("mock error"),
            ):
                with pytest.warns(RuntimeWarning, match="knockpy failed"):
                    result, meta = selector.select(X, y)
        assert result["selected_count"] == 0
        assert meta["knockoff_backend"] == "error"
        assert "mock error" in meta["error"]


# ---------------------------------------------------------------------------
# Real knockpy tests (skip if not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not KNOCKPY_AVAILABLE, reason="knockpy not installed")
class TestKnockoffReal:
    """Verify real knockpy integration."""

    def test_basic_selection(self):
        X, y = _make_binary_data(n=200, p=20, n_informative=5)
        selector = KnockoffBenchmarkSelector(fdr=0.20, random_state=42)
        result, meta = selector.select(X, y)
        assert "selected_indices" in result
        assert isinstance(result["selected_indices"], np.ndarray)
        assert meta["knockoff_backend"] == "knockpy"

    def test_high_fdr_selects_more(self):
        X, y = _make_binary_data(n=200, p=20, n_informative=5)
        sel_low = KnockoffBenchmarkSelector(fdr=0.05, random_state=42)
        sel_high = KnockoffBenchmarkSelector(fdr=0.50, random_state=42)
        r_low, _ = sel_low.select(X, y)
        r_high, _ = sel_high.select(X, y)
        # Higher FDR should generally select at least as many features.
        assert r_high["selected_count"] >= r_low["selected_count"]


# ---------------------------------------------------------------------------
# Config toggle test
# ---------------------------------------------------------------------------


class TestBenchmarkKnockoffConfig:
    """Verify benchmark_knockoff_enabled config field exists."""

    def test_config_field_default_false(self):
        from tabnetics.pipeline.pipeline import DFFSConfig
        cfg = DFFSConfig()
        assert hasattr(cfg, "benchmark_knockoff_enabled")
        assert cfg.benchmark_knockoff_enabled is False

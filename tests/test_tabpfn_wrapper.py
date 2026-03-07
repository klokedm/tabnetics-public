"""Tests for TabPFN benchmark classifier wrapper (VAL12_Suggestions §4.2)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from tabnetics.feature_selection.methods.tabpfn_classifier import (
    TabPFNBenchmarkClassifier,
    TABPFN_AVAILABLE,
    MAX_FEATURES,
    MAX_SAMPLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_data(n=80, p=10, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = np.repeat([0, 1], n // 2)
    X[:n // 2, :2] += 1.5
    return X, y


def _make_multiclass_data(n=90, p=8, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = np.tile([0, 1, 2], n // 3)[:n]
    X[y == 0, :2] += 2.0
    X[y == 1, 2:4] += 2.0
    return X, y


# ---------------------------------------------------------------------------
# Fallback tests (always run, no tabpfn needed)
# ---------------------------------------------------------------------------


class TestTabPFNFallback:
    """Verify fallback behavior when TabPFN is unavailable or limits exceeded."""

    def test_fallback_when_too_many_features(self):
        X, y = _make_binary_data(n=50, p=MAX_FEATURES + 10)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        with patch(
            "tabnetics.feature_selection.methods.tabpfn_classifier.TABPFN_AVAILABLE",
            True,
        ):
            with pytest.warns(RuntimeWarning, match="n_features="):
                clf.fit(X, y)
        assert clf.exceeded_limits_
        assert clf.fallback_reason_ is not None
        assert "n_features" in clf.fallback_reason_
        assert not clf.is_using_tabpfn

    def test_fallback_when_too_many_samples(self):
        X, y = _make_binary_data(n=MAX_SAMPLES + 100, p=5)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        with patch(
            "tabnetics.feature_selection.methods.tabpfn_classifier.TABPFN_AVAILABLE",
            True,
        ):
            with pytest.warns(RuntimeWarning, match="n_samples="):
                clf.fit(X, y)
        assert clf.exceeded_limits_
        assert "n_samples" in clf.fallback_reason_
        assert not clf.is_using_tabpfn

    def test_fallback_predict_produces_valid_labels(self):
        X, y = _make_binary_data(n=50, p=MAX_FEATURES + 1)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        with pytest.warns(RuntimeWarning):
            clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (50,)
        assert set(preds).issubset({0, 1})

    def test_fallback_predict_proba_shape(self):
        X, y = _make_binary_data(n=50, p=MAX_FEATURES + 1)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        with pytest.warns(RuntimeWarning):
            clf.fit(X, y)
        probs = clf.predict_proba(X)
        assert probs.shape == (50, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_custom_limits(self):
        X, y = _make_binary_data(n=50, p=20)
        clf = TabPFNBenchmarkClassifier(max_features=15, max_samples=40, random_state=0)
        with patch(
            "tabnetics.feature_selection.methods.tabpfn_classifier.TABPFN_AVAILABLE",
            True,
        ):
            with pytest.warns(RuntimeWarning, match="n_features="):
                clf.fit(X, y)
        assert clf.exceeded_limits_

    def test_custom_sample_limit(self):
        X, y = _make_binary_data(n=50, p=5)
        clf = TabPFNBenchmarkClassifier(max_samples=30, random_state=0)
        with patch(
            "tabnetics.feature_selection.methods.tabpfn_classifier.TABPFN_AVAILABLE",
            True,
        ):
            with pytest.warns(RuntimeWarning, match="n_samples="):
                clf.fit(X, y)
        assert clf.exceeded_limits_

    def test_unfitted_predict_raises(self):
        clf = TabPFNBenchmarkClassifier()
        with pytest.raises(Exception):  # NotFittedError or AttributeError
            clf.predict(np.zeros((5, 3)))

    def test_multiclass_fallback(self):
        X, y = _make_multiclass_data(n=60, p=MAX_FEATURES + 5)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        with pytest.warns(RuntimeWarning):
            clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (60,)
        probs = clf.predict_proba(X)
        assert probs.shape == (60, 3)

    def test_fallback_when_package_unavailable(self):
        """Simulate tabpfn not installed."""
        with patch(
            "tabnetics.feature_selection.methods.tabpfn_classifier.TABPFN_AVAILABLE",
            False,
        ):
            X, y = _make_binary_data(n=50, p=5)
            clf = TabPFNBenchmarkClassifier(random_state=42)
            with pytest.warns(RuntimeWarning, match="not installed"):
                clf.fit(X, y)
            assert clf.exceeded_limits_
            assert "not installed" in clf.fallback_reason_

    def test_classes_attribute_set(self):
        X, y = _make_multiclass_data(n=60, p=MAX_FEATURES + 1)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        with pytest.warns(RuntimeWarning):
            clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    def test_n_features_in_set(self):
        X, y = _make_binary_data(n=50, p=MAX_FEATURES + 1)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        with pytest.warns(RuntimeWarning):
            clf.fit(X, y)
        assert clf.n_features_in_ == MAX_FEATURES + 1


# ---------------------------------------------------------------------------
# Tests that use real TabPFN (skip if not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TABPFN_AVAILABLE, reason="tabpfn not installed")
class TestTabPFNReal:
    """Verify real TabPFN integration (requires tabpfn package)."""

    def test_fit_predict_binary(self):
        X, y = _make_binary_data(n=80, p=10)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        clf.fit(X[:60], y[:60])
        assert clf.is_using_tabpfn
        preds = clf.predict(X[60:])
        assert preds.shape == (20,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_binary(self):
        X, y = _make_binary_data(n=80, p=10)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        clf.fit(X[:60], y[:60])
        probs = clf.predict_proba(X[60:])
        assert probs.shape == (20, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_fit_predict_multiclass(self):
        X, y = _make_multiclass_data(n=90, p=8)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        clf.fit(X[:60], y[:60])
        preds = clf.predict(X[60:])
        assert preds.shape == (30,)
        probs = clf.predict_proba(X[60:])
        assert probs.shape == (30, 3)

    def test_within_limits_uses_tabpfn(self):
        X, y = _make_binary_data(n=50, p=10)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        clf.fit(X, y)
        assert clf.is_using_tabpfn
        assert not clf.exceeded_limits_

    def test_at_feature_boundary(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, MAX_FEATURES)
        y = np.repeat([0, 1], 25)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        clf.fit(X, y)
        assert clf.is_using_tabpfn

    def test_at_sample_boundary(self):
        rng = np.random.RandomState(42)
        X = rng.randn(MAX_SAMPLES, 10)
        y = np.repeat([0, 1], MAX_SAMPLES // 2)
        clf = TabPFNBenchmarkClassifier(random_state=42)
        clf.fit(X, y)
        assert clf.is_using_tabpfn


# ---------------------------------------------------------------------------
# Config toggle test
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    """Verify benchmark_tabpfn_enabled config field exists."""

    def test_config_field_default_false(self):
        from tabnetics.pipeline.pipeline import DFFSConfig
        cfg = DFFSConfig()
        assert hasattr(cfg, "benchmark_tabpfn_enabled")
        assert cfg.benchmark_tabpfn_enabled is False

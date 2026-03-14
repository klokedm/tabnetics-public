"""Tests for extract_meta_features (T-P3-INFRA-001)."""
import numpy as np
import pytest

from tabnetics.datasets.meta_features import extract_meta_features

EXPECTED_KEYS = {
    "n", "p", "p_over_n", "class_count",
    "class_balance_entropy", "correlation_spectrum_decay",
    "heaping_fraction",
}


def test_meta_features_basic_shape():
    """All 7 keys present and values are float."""
    rng = np.random.RandomState(0)
    X = rng.randn(100, 10)
    y = rng.choice([0, 1], size=100)
    mf = extract_meta_features(X, y)
    assert set(mf.keys()) == EXPECTED_KEYS
    for k, v in mf.items():
        assert isinstance(v, float), f"{k} is not float: {type(v)}"
    assert mf["n"] == 100.0
    assert mf["p"] == 10.0
    assert mf["p_over_n"] == pytest.approx(0.1)
    assert mf["class_count"] == 2.0


def test_meta_features_class_balance_entropy():
    """Balanced binary → entropy ~1.0; fully imbalanced → entropy ~0.0."""
    X = np.random.RandomState(1).randn(100, 5)

    # perfectly balanced binary
    y_balanced = np.array([0, 1] * 50)
    mf = extract_meta_features(X, y_balanced)
    assert mf["class_balance_entropy"] == pytest.approx(1.0, abs=1e-6)

    # completely imbalanced (single effective class)
    y_imbalanced = np.zeros(100)
    mf2 = extract_meta_features(X, y_imbalanced)
    assert mf2["class_balance_entropy"] == pytest.approx(0.0, abs=1e-6)


def test_meta_features_heaping_fraction():
    """All integer features → 1.0; all continuous → ~0.0."""
    rng = np.random.RandomState(2)

    # integer features
    X_int = rng.randint(0, 10, size=(50, 8)).astype(float)
    y = rng.choice([0, 1], size=50)
    mf = extract_meta_features(X_int, y)
    assert mf["heaping_fraction"] == pytest.approx(1.0)

    # continuous features (extremely unlikely to be integer-valued)
    X_cont = rng.randn(50, 8) * 0.1 + 0.3
    mf2 = extract_meta_features(X_cont, y)
    assert mf2["heaping_fraction"] == pytest.approx(0.0)


def test_meta_features_edge_single_feature():
    """Single feature should work correctly."""
    X = np.random.RandomState(3).randn(20, 1)
    y = np.zeros(20)
    mf = extract_meta_features(X, y)
    assert mf["p"] == 1.0
    assert mf["correlation_spectrum_decay"] == 0.0  # p < 3


def test_meta_features_edge_two_features():
    """Two features: correlation_spectrum_decay should be 0 (p < 3)."""
    X = np.random.RandomState(4).randn(30, 2)
    y = np.array([0] * 15 + [1] * 15)
    mf = extract_meta_features(X, y)
    assert mf["correlation_spectrum_decay"] == 0.0


def test_meta_features_multiclass():
    """Three classes: class_count == 3, entropy < 1.0 for imbalanced."""
    X = np.random.RandomState(5).randn(90, 5)
    # Imbalanced 3-class
    y = np.array([0]*60 + [1]*20 + [2]*10)
    mf = extract_meta_features(X, y)
    assert mf["class_count"] == 3.0
    assert 0.0 < mf["class_balance_entropy"] < 1.0

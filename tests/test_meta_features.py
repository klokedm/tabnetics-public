"""Tests for extract_meta_features (T-P3-INFRA-001, T-R-398)."""
import numpy as np
import pytest

import tabnetics.datasets.meta_features as meta_features_module
from tabnetics.datasets.meta_features import extract_meta_features

EXPECTED_KEYS = {
    "n", "p", "p_over_n", "class_count",
    "class_balance_entropy", "correlation_spectrum_decay",
    "heaping_fraction",
}

EXPANDED_KEYS = EXPECTED_KEYS | {
    "fisher_f1", "f2_overlap", "n1_borderline", "n2_nn_ratio",
    "lsc", "t4_pca_ratio", "intrinsic_dim", "correlation_alpha",
    "signal_eigenvalue_fraction",
}


# -------------------------------------------------------------------
# Shared fixtures
# -------------------------------------------------------------------

def _well_separated(rng, n_per_class=50, p=10, gap=10.0):
    """Two well-separated Gaussian clusters."""
    X = np.vstack([
        rng.randn(n_per_class, p) + gap,
        rng.randn(n_per_class, p) - gap,
    ])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def _overlapping(rng, n=100, p=10):
    """Random features with random labels (no class structure)."""
    X = rng.randn(n, p)
    y = rng.choice([0, 1], size=n)
    return X, y


# -------------------------------------------------------------------
# Base feature tests (unchanged)
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# Expanded meta-feature tests (T-R-398)
# -------------------------------------------------------------------

class TestExpandedKeySet:
    """expanded=True returns all 16 keys; default returns only 7."""

    def test_expanded_returns_all_keys(self):
        rng = np.random.RandomState(10)
        X, y = _well_separated(rng)
        mf = extract_meta_features(X, y, expanded=True)
        assert set(mf.keys()) == EXPANDED_KEYS
        for k, v in mf.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"

    def test_default_returns_base_keys_only(self):
        rng = np.random.RandomState(10)
        X, y = _well_separated(rng)
        mf = extract_meta_features(X, y)
        assert set(mf.keys()) == EXPECTED_KEYS

    def test_base_values_identical_with_expanded(self):
        """Base 7 features must be identical regardless of expanded flag."""
        rng = np.random.RandomState(11)
        X, y = _well_separated(rng)
        base = extract_meta_features(X, y, expanded=False)
        full = extract_meta_features(X, y, expanded=True)
        for k in EXPECTED_KEYS:
            assert base[k] == pytest.approx(full[k]), f"{k} differs"


class TestFisherF1:
    def test_high_for_separated(self):
        rng = np.random.RandomState(20)
        X, y = _well_separated(rng, gap=10.0)
        mf = extract_meta_features(X, y, expanded=True)
        assert mf["fisher_f1"] > 50.0  # very separated → large ratio

    def test_low_for_overlapping(self):
        rng = np.random.RandomState(21)
        X, y = _overlapping(rng)
        mf = extract_meta_features(X, y, expanded=True)
        assert mf["fisher_f1"] < 5.0  # random labels → small ratio


class TestF2Overlap:
    def test_low_for_separated(self):
        rng = np.random.RandomState(30)
        X, y = _well_separated(rng, gap=10.0)
        mf = extract_meta_features(X, y, expanded=True)
        assert mf["f2_overlap"] < 0.3

    def test_high_for_overlapping(self):
        rng = np.random.RandomState(31)
        X, y = _overlapping(rng)
        mf = extract_meta_features(X, y, expanded=True)
        assert mf["f2_overlap"] > 0.5


class TestN1Borderline:
    def test_low_for_separated(self):
        rng = np.random.RandomState(40)
        X, y = _well_separated(rng, gap=10.0)
        mf = extract_meta_features(X, y, expanded=True)
        # Only ~1 MST edge crosses the boundary for well-separated clusters
        assert mf["n1_borderline"] < 0.1

    def test_high_for_overlapping(self):
        rng = np.random.RandomState(41)
        X, y = _overlapping(rng)
        mf = extract_meta_features(X, y, expanded=True)
        assert mf["n1_borderline"] > 0.3


class TestN2NNRatio:
    def test_low_for_separated(self):
        rng = np.random.RandomState(50)
        X, y = _well_separated(rng, gap=10.0)
        mf = extract_meta_features(X, y, expanded=True)
        # Intra < inter → ratio < 1
        assert mf["n2_nn_ratio"] < 0.2

    def test_near_one_for_overlapping(self):
        rng = np.random.RandomState(51)
        X, y = _overlapping(rng)
        mf = extract_meta_features(X, y, expanded=True)
        # Random labels → intra ≈ inter → ratio near 1
        assert 0.5 < mf["n2_nn_ratio"] < 1.5


class TestLSC:
    def test_positive_for_separated(self):
        rng = np.random.RandomState(60)
        X, y = _well_separated(rng)
        mf = extract_meta_features(X, y, expanded=True)
        # Each cluster's points have many same-class neighbors closer than
        # the nearest enemy, so lsc should be large.
        assert mf["lsc"] > 5.0

    def test_small_for_overlapping(self):
        rng = np.random.RandomState(61)
        X, y = _overlapping(rng)
        mf = extract_meta_features(X, y, expanded=True)
        # Random labels → enemies often very close → small local sets
        assert mf["lsc"] < 5.0


class TestT4PcaRatio:
    def test_low_for_redundant_features(self):
        """High-dim data with 2 true components → small ratio."""
        rng = np.random.RandomState(70)
        base = rng.randn(100, 2)
        proj = rng.randn(2, 20)
        X = base @ proj + rng.randn(100, 20) * 0.01
        y = (base[:, 0] > 0).astype(int)
        mf = extract_meta_features(X, y, expanded=True)
        # 2 out of 20 needed for 95% variance
        assert mf["t4_pca_ratio"] < 0.3

    def test_high_for_independent_features(self):
        """iid features → need most components."""
        rng = np.random.RandomState(71)
        X = rng.randn(100, 5)
        y = rng.choice([0, 1], size=100)
        mf = extract_meta_features(X, y, expanded=True)
        assert mf["t4_pca_ratio"] > 0.6


class TestSignalEigenvalueFraction:
    def test_in_unit_interval(self):
        rng = np.random.RandomState(80)
        X, y = _well_separated(rng)
        mf = extract_meta_features(X, y, expanded=True)
        assert 0.0 <= mf["signal_eigenvalue_fraction"] <= 1.0

    def test_positive_for_structured_data(self):
        """Data with strong signal should have some eigenvalues above MP."""
        rng = np.random.RandomState(81)
        base = rng.randn(100, 2)
        proj = rng.randn(2, 20)
        X = base @ proj + rng.randn(100, 20) * 0.01
        y = (base[:, 0] > 0).astype(int)
        mf = extract_meta_features(X, y, expanded=True)
        assert mf["signal_eigenvalue_fraction"] > 0.0


class TestCorrelationAlpha:
    def test_default_for_few_features(self):
        """p < 3 → abs_corr_sorted=None → returns default 0.5."""
        rng = np.random.RandomState(90)
        X = rng.randn(50, 2)
        y = rng.choice([0, 1], size=50)
        mf = extract_meta_features(X, y, expanded=True)
        assert mf["correlation_alpha"] == pytest.approx(0.5)

    def test_finite_for_correlated_data(self):
        """Structured correlation matrix → non-default alpha."""
        rng = np.random.RandomState(91)
        base = rng.randn(100, 3)
        proj = rng.randn(3, 30)
        X = base @ proj + rng.randn(100, 30) * 0.1
        y = rng.choice([0, 1], size=100)
        mf = extract_meta_features(X, y, expanded=True)
        assert np.isfinite(mf["correlation_alpha"])


class TestIntrinsicDim:
    def test_reasonable_for_2d_manifold(self):
        """2D plane + noise embedded in 20D → intrinsic dim near 2."""
        rng = np.random.RandomState(100)
        base = rng.randn(200, 2)
        proj = rng.randn(2, 20)
        X = base @ proj + rng.randn(200, 20) * 0.001
        y = (base[:, 0] > 0).astype(int)
        mf = extract_meta_features(X, y, expanded=True)
        assert 1.0 < mf["intrinsic_dim"] < 5.0

    def test_positive(self):
        rng = np.random.RandomState(101)
        X, y = _well_separated(rng)
        mf = extract_meta_features(X, y, expanded=True)
        assert mf["intrinsic_dim"] > 0.0


class TestExpandedEdgeCases:
    def test_single_class(self):
        """Single class → overlap/border measures degenerate gracefully."""
        rng = np.random.RandomState(110)
        X = rng.randn(30, 5)
        y = np.zeros(30, dtype=int)
        mf = extract_meta_features(X, y, expanded=True)
        assert mf["fisher_f1"] == 0.0
        assert mf["f2_overlap"] == 0.0
        assert mf["n1_borderline"] == 0.0
        assert mf["lsc"] == 0.0

    def test_small_n(self):
        """n=6, p=4 → expanded should still work."""
        rng = np.random.RandomState(111)
        X = rng.randn(6, 4)
        y = np.array([0, 0, 0, 1, 1, 1])
        mf = extract_meta_features(X, y, expanded=True)
        assert set(mf.keys()) == EXPANDED_KEYS
        for v in mf.values():
            assert np.isfinite(v)

    def test_hdlss_regime(self):
        """p >> n (HDLSS) should not crash."""
        rng = np.random.RandomState(112)
        X = rng.randn(20, 200)
        y = np.array([0] * 10 + [1] * 10)
        mf = extract_meta_features(X, y, expanded=True)
        assert set(mf.keys()) == EXPANDED_KEYS
        for v in mf.values():
            assert np.isfinite(v)

    def test_eigen_solver_fallback_when_eigvalsh_fails(self, monkeypatch):
        """Expanded extractor should fall back cleanly when eigvalsh fails."""
        rng = np.random.RandomState(113)
        X, y = _well_separated(rng, n_per_class=20, p=12)

        def _boom(*_args, **_kwargs):
            raise np.linalg.LinAlgError("Eigenvalues did not converge")

        monkeypatch.setattr(meta_features_module.np.linalg, "eigvalsh", _boom)
        mf = extract_meta_features(X, y, expanded=True)
        assert set(mf.keys()) == EXPANDED_KEYS
        for v in mf.values():
            assert np.isfinite(v)

    def test_non_finite_inputs_are_sanitized(self):
        """Expanded extractor should return finite outputs for NaN/inf-heavy inputs."""
        rng = np.random.RandomState(114)
        X = rng.randn(40, 8)
        X[0, 0] = np.nan
        X[1, 1] = np.inf
        X[2, 2] = -np.inf
        y = np.array([0] * 20 + [1] * 20)
        mf = extract_meta_features(X, y, expanded=True)
        assert set(mf.keys()) == EXPANDED_KEYS
        for v in mf.values():
            assert np.isfinite(v)

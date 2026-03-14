"""Tests for Tier 2 interaction-aware screening (T-004).

Covers:
 1. STIR scores shape
 2. STIR scores are finite (no NaN/Inf)
 3. Determinism (same seed → same output)
 4. XOR synthetic test (STIR > univariate for interaction features)
 5. Single class → graceful zeros
 6. Empty features → empty array
 7. Small n (n < n_neighbors)
 8. screen_features_stir returns valid indices
 9. Keep fraction: correct count
10. Min features cap
11. Toggle OFF → no screening (returns None)
12. Toggle ON → screening applied
13. Config wiring via from_config
14. Pool cap: subsamples wide data
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tabnetics.feature_selection.methods.screening import (
    compute_stir_scores,
    screen_features_stir,
    _nearest_neighbors,
)
from tabnetics.feature_selection.config import (
    ScreeningConfig,
    FeatureSelectorConfig,
)


SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────
def _make_data(n=100, p=20, n_classes=3, seed=SEED):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = rng.randint(0, n_classes, size=n)
    return X, y


def _make_xor_data(n=200, seed=SEED):
    """Create XOR dataset where informative features interact.

    Features 0 and 1 jointly determine class via XOR.
    Features 2..9 are noise.
    Univariate methods cannot rank features 0/1 highly.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 10)
    # XOR: class = (sign(X0) == sign(X1))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


# ── 1. shape ─────────────────────────────────────────────────────────────
def test_stir_scores_shape():
    X, y = _make_data(n=60, p=15)
    scores = compute_stir_scores(X, y, n_neighbors=5, n_iter=30, random_state=SEED)
    assert scores.shape == (15,)


# ── 2. finite ────────────────────────────────────────────────────────────
def test_stir_scores_finite():
    X, y = _make_data(n=80, p=25)
    scores = compute_stir_scores(X, y, random_state=SEED)
    assert np.all(np.isfinite(scores))


# ── 3. determinism ───────────────────────────────────────────────────────
def test_stir_determinism():
    X, y = _make_data(n=80, p=20)
    s1 = compute_stir_scores(X, y, random_state=SEED)
    s2 = compute_stir_scores(X, y, random_state=SEED)
    np.testing.assert_array_equal(s1, s2)


# ── 4. XOR interaction test ──────────────────────────────────────────────
def test_stir_xor_interaction():
    """STIR should rank the two XOR features (0, 1) higher than noise."""
    X, y = _make_xor_data(n=400, seed=SEED)
    scores = compute_stir_scores(X, y, n_neighbors=10, n_iter=200, random_state=SEED)

    # Features 0 and 1 should be in the top 4
    top4 = set(np.argsort(scores)[::-1][:4].tolist())
    assert 0 in top4, f"Feature 0 not in top-4; scores={scores}"
    assert 1 in top4, f"Feature 1 not in top-4; scores={scores}"


# ── 5. single class ─────────────────────────────────────────────────────
def test_stir_single_class():
    X = np.random.RandomState(SEED).randn(40, 10)
    y = np.zeros(40, dtype=int)
    scores = compute_stir_scores(X, y, random_state=SEED)
    assert scores.shape == (10,)
    np.testing.assert_array_equal(scores, 0.0)


# ── 6. empty features ───────────────────────────────────────────────────
def test_stir_empty_features():
    X = np.empty((20, 0), dtype=float)
    y = np.zeros(20, dtype=int)
    scores = compute_stir_scores(X, y, random_state=SEED)
    assert scores.shape == (0,)


# ── 7. small n ───────────────────────────────────────────────────────────
def test_stir_small_n():
    """n < n_neighbors should not crash."""
    X, y = _make_data(n=3, p=5, n_classes=2, seed=SEED)
    scores = compute_stir_scores(X, y, n_neighbors=10, n_iter=50, random_state=SEED)
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))


# ── 8. screen_features_stir valid indices ────────────────────────────────
def test_screen_features_stir_returns_valid_indices():
    X, y = _make_data(n=60, p=30)
    idx = screen_features_stir(
        X, y,
        enabled=True, method="stir",
        stir_keep_fraction=0.5, stir_min_features=5,
        random_state=SEED,
    )
    assert idx is not None
    assert idx.ndim == 1
    assert np.all(idx >= 0)
    assert np.all(idx < 30)
    # no duplicates
    assert len(np.unique(idx)) == len(idx)


# ── 9. keep fraction ────────────────────────────────────────────────────
def test_screen_keep_fraction():
    X, y = _make_data(n=80, p=40)
    idx = screen_features_stir(
        X, y,
        enabled=True, method="stir",
        stir_keep_fraction=0.25, stir_min_features=1,
        random_state=SEED,
    )
    assert idx is not None
    expected = int(np.ceil(0.25 * 40))
    assert len(idx) == expected


# ── 10. min features cap ────────────────────────────────────────────────
def test_screen_min_features_cap():
    X, y = _make_data(n=60, p=30)
    idx = screen_features_stir(
        X, y,
        enabled=True, method="stir",
        stir_keep_fraction=0.01,   # would give ~1
        stir_min_features=15,
        random_state=SEED,
    )
    assert idx is not None
    assert len(idx) >= 15


# ── 11. toggle OFF ──────────────────────────────────────────────────────
def test_screen_toggle_off():
    X, y = _make_data()
    result = screen_features_stir(
        X, y, enabled=False, method="stir", random_state=SEED,
    )
    assert result is None


def test_screen_method_none():
    X, y = _make_data()
    result = screen_features_stir(
        X, y, enabled=True, method="none", random_state=SEED,
    )
    assert result is None


# ── 12. toggle ON ───────────────────────────────────────────────────────
def test_screen_toggle_on():
    X, y = _make_data(n=60, p=30)
    idx = screen_features_stir(
        X, y,
        enabled=True, method="stir",
        stir_keep_fraction=0.5, stir_min_features=5,
        random_state=SEED,
    )
    assert idx is not None
    assert len(idx) < 30


# ── 13. config wiring via from_config ────────────────────────────────────
def test_config_wiring():
    cfg = FeatureSelectorConfig(
        screening=ScreeningConfig(
            enabled=True,
            method="stir",
            pool_cap=500,
            stir_n_neighbors=7,
            stir_n_iter=40,
            stir_keep_fraction=0.3,
            stir_min_features=10,
        ),
    )
    from tabnetics.feature_selection.base import FeatureSelector
    fs = FeatureSelector.from_config(cfg)
    assert fs.screening_enabled is True
    assert fs.screening_method == "stir"
    assert fs.screening_pool_cap == 500
    assert fs.screening_stir_n_neighbors == 7
    assert fs.screening_stir_n_iter == 40
    assert fs.screening_stir_keep_fraction == pytest.approx(0.3)
    assert fs.screening_stir_min_features == 10


def test_config_defaults_off():
    """Default ScreeningConfig must leave screening disabled."""
    cfg = FeatureSelectorConfig()
    from tabnetics.feature_selection.base import FeatureSelector
    fs = FeatureSelector.from_config(cfg)
    assert fs.screening_enabled is False
    assert fs.screening_method == "none"


# ── 14. pool cap subsampling ────────────────────────────────────────────
def test_screen_pool_cap():
    """When p > pool_cap, columns should be subsampled."""
    X, y = _make_data(n=50, p=100)
    idx = screen_features_stir(
        X, y,
        enabled=True, method="stir",
        pool_cap=30,  # much smaller than p=100
        stir_keep_fraction=0.5, stir_min_features=5,
        random_state=SEED,
    )
    assert idx is not None
    assert len(idx) <= 30
    assert np.all(idx < 100)  # original column space


# ── 15. constant features get zero weight ────────────────────────────────
def test_stir_constant_features():
    """Constant features should get zero STIR score."""
    rng = np.random.RandomState(SEED)
    X = rng.randn(60, 10)
    X[:, 3] = 7.0  # constant
    X[:, 7] = -1.0  # constant
    y = rng.randint(0, 2, size=60)
    scores = compute_stir_scores(X, y, random_state=SEED)
    assert scores[3] == 0.0
    assert scores[7] == 0.0


# ── 16. unknown method returns None ──────────────────────────────────────
def test_screen_unknown_method():
    X, y = _make_data()
    result = screen_features_stir(
        X, y, enabled=True, method="unknown_method", random_state=SEED,
    )
    assert result is None


def test_stir_knn_matches_argsort_reference():
    rng = np.random.RandomState(SEED)
    X = rng.randn(40, 7)
    y = rng.randint(0, 3, size=40)
    idx = 5
    k = 6

    nn = _nearest_neighbors(X, idx, y, same_class=True, k=k)

    label = y[idx]
    mask = (y == label)
    mask[idx] = False
    cand = np.flatnonzero(mask)
    diffs = X[cand] - X[idx]
    dists = np.einsum("ij,ij->i", diffs, diffs)
    k_eff = min(k, cand.size)
    ref = cand[np.argsort(dists)[:k_eff]]

    assert set(nn.tolist()) == set(ref.tolist())


def test_boundary_tier2_routing_in_base_module_source():
    """FeatureSelector should route Tier 2 screening via methods/screening module."""
    import inspect
    import tabnetics.feature_selection.base as base_module

    source = inspect.getsource(base_module.FeatureSelector.fit_transform)
    assert "from .methods.screening import screen_features" in source

"""Smoke tests for Chi-Square and ReliefF filter methods."""
import numpy as np
import pytest

from tabnetics.feature_selection.methods.filter import (
    chi_square_selection,
    relieff_selection,
)
from tabnetics.feature_selection.registry import METHOD_REGISTRY


# ---------- helpers ----------
def _make_synth(n_samples=100, n_features=50, n_classes=2, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features)
    y = rng.randint(0, n_classes, size=n_samples)
    return X, y


# ---------- Chi-Square tests ----------
class TestChiSquare:
    def test_chi_square_basic(self):
        X, y = _make_synth()
        results, importance = chi_square_selection(X, y, n_target_features=10)
        assert len(results) > 0
        assert 'selected_indices' in results
        assert 'scores' in results
        assert 'all_scores' in results
        assert len(results['selected_indices']) <= 10
        assert len(importance) == X.shape[1]

    def test_chi_square_empty_input(self):
        X = np.empty((100, 0))
        y = np.zeros(100, dtype=int)
        results, importance = chi_square_selection(X, y, n_target_features=5)
        assert results == {}
        assert importance == {}

    def test_chi_square_registered(self):
        assert 'chi_square' in METHOD_REGISTRY
        spec = METHOD_REGISTRY['chi_square']
        assert spec.paradigm == 'filter'
        assert spec.fn_name == '_chi_square_selection'


# ---------- ReliefF tests ----------
class TestReliefF:
    def test_relieff_basic(self):
        X, y = _make_synth()
        results, importance = relieff_selection(X, y, n_target_features=10)
        assert len(results) > 0
        assert 'selected_indices' in results
        assert 'scores' in results
        assert 'all_scores' in results
        assert len(results['selected_indices']) <= 10
        assert len(importance) == X.shape[1]

    def test_relieff_multiclass(self):
        X, y = _make_synth(n_classes=4)
        results, importance = relieff_selection(X, y, n_target_features=10)
        assert len(results) > 0
        assert 'selected_indices' in results
        assert len(results['selected_indices']) <= 10
        assert len(importance) == X.shape[1]

    def test_relieff_registered(self):
        assert 'relieff' in METHOD_REGISTRY
        spec = METHOD_REGISTRY['relieff']
        assert spec.paradigm == 'filter'
        assert spec.fn_name == '_relieff_selection'

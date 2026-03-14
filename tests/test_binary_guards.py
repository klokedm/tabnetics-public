"""Tests for binary class guards in FeatureSelector (T-P3-GATE-004).

The guards (disable_redundancy_penalty_binary, disable_class_pareto_binary)
are implemented in experiments/feature_selection/base.py around lines 2355-2366.
Diagnostics for them are stored in selection_result_.config dict with keys:
  - binary_redundancy_penalty_disabled (int, 0 or 1)
  - binary_class_pareto_override (int, 0 or 1)
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from tabnetics.feature_selection.base import FeatureSelector


def _make_binary_dataset(random_state=42):
    """Create a small binary classification dataset."""
    X, y = make_classification(
        n_classes=2, n_features=50, n_samples=100,
        n_informative=10, random_state=random_state,
    )
    return X, y


def _make_multiclass_dataset(random_state=42):
    """Create a 4-class dataset."""
    X, y = make_classification(
        n_classes=4, n_features=50, n_samples=200,
        n_informative=20, n_clusters_per_class=1,
        random_state=random_state,
    )
    return X, y


def _fit_selector(X, y, **kwargs):
    """Fit a FeatureSelector with minimal methods for speed and return the result config."""
    defaults = dict(
        random_state=42,
        n_target_features=10,
        enabled_methods=["anova_f", "mutual_information"],
        portfolio_size=2,
        n_bootstrap_iterations=2,
        method_timeout_seconds=30,
    )
    defaults.update(kwargs)
    n_target = defaults.pop("n_target_features")
    sel = FeatureSelector(**defaults)
    _, result = sel.fit_transform(X, y, n_final_features=n_target, return_result_object=True)
    return result.config


class TestRedundancyPenaltyDisabledForBinary:
    """When disable_redundancy_penalty_binary=True and data is binary,
    oracle redundancy penalty should be disabled."""

    def test_redundancy_penalty_disabled_for_binary(self):
        X, y = _make_binary_dataset()
        cfg = _fit_selector(
            X, y,
            use_oracle_redundancy_penalty=True,
            disable_redundancy_penalty_binary=True,
        )
        assert cfg["binary_redundancy_penalty_disabled"] == 1, (
            "binary_redundancy_penalty_disabled should be 1 for binary data"
        )
        assert cfg["use_oracle_redundancy_penalty_effective"] is False, (
            "Oracle redundancy penalty should be effectively disabled for binary"
        )


class TestClassParetoSkippedForBinary:
    """When disable_class_pareto_binary=True and data is binary,
    class-Pareto min_classes gate should be raised to ≥3."""

    def test_class_pareto_skipped_for_binary(self):
        X, y = _make_binary_dataset()
        cfg = _fit_selector(
            X, y,
            disable_class_pareto_binary=True,
        )
        assert cfg["binary_class_pareto_override"] == 1, (
            "binary_class_pareto_override should be 1 for binary data"
        )
        assert cfg["class_pareto_min_classes_effective"] >= 3, (
            "class_pareto_min_classes_effective should be ≥3 for binary data"
        )


class TestGuardsInactiveForMulticlass:
    """For multiclass data (C≥3), neither binary guard should activate."""

    def test_guards_inactive_for_multiclass(self):
        X, y = _make_multiclass_dataset()
        cfg = _fit_selector(
            X, y,
            use_oracle_redundancy_penalty=True,
            disable_redundancy_penalty_binary=True,
            disable_class_pareto_binary=True,
        )
        assert cfg["binary_redundancy_penalty_disabled"] == 0, (
            "binary_redundancy_penalty_disabled should be 0 for multiclass"
        )
        assert cfg["binary_class_pareto_override"] == 0, (
            "binary_class_pareto_override should be 0 for multiclass"
        )

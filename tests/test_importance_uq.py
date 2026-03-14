import numpy as np
from sklearn.datasets import make_classification

from tabnetics.feature_selection.cv import compute_feature_importance_uq
from tabnetics.feature_selection import FeatureSelector


def _dataset(seed: int = 7):
    X, y = make_classification(
        n_samples=72,
        n_features=24,
        n_informative=8,
        n_redundant=4,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.02,
        random_state=seed,
    )
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def test_compute_feature_importance_uq_returns_variance_vectors():
    X, y = _dataset()
    uq = compute_feature_importance_uq(
        X,
        y,
        problem_type="classification",
        random_state=7,
        inner_cv_splits=3,
        inner_cv_repeats=1,
        min_cv_folds=2,
        max_folds=4,
    )

    assert bool(uq["importance_uq_enabled"]) is True
    assert bool(uq["importance_uq_computed"]) is True
    assert int(uq["importance_uq_n_folds"]) >= 2
    assert np.asarray(uq["importance_mean"]).shape[0] == X.shape[1]
    assert np.asarray(uq["importance_variance"]).shape[0] == X.shape[1]


def test_compute_feature_importance_uq_respects_min_fold_requirement():
    X, y = _dataset()
    uq = compute_feature_importance_uq(
        X,
        y,
        problem_type="classification",
        random_state=7,
        inner_cv_splits=2,
        inner_cv_repeats=1,
        min_cv_folds=8,
        max_folds=4,
    )

    assert bool(uq["importance_uq_enabled"]) is True
    assert bool(uq["importance_uq_computed"]) is False
    assert str(uq["importance_uq_reason"]) in {"insufficient_successful_folds", "no_cv_splits"}


def test_feature_selector_emits_importance_uq_when_enabled():
    X, y = _dataset()
    selector = FeatureSelector(
        random_state=7,
        selection_strategy="legacy_voting",
        enabled_methods=("mutual_information", "anova_f"),
        n_bootstrap_iterations=1,
        importance_uq_enabled=True,
        importance_uq_min_cv_folds=2,
        inner_cv_splits=3,
        inner_cv_repeats=1,
    )

    _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
    assert bool(result.importance_uq.get("importance_uq_enabled", False)) is True
    assert "importance_uq_computed" in result.importance_uq
    assert isinstance(result.feature_importance_variance, dict)
    if bool(result.importance_uq.get("importance_uq_computed", False)):
        assert len(result.feature_importance_variance) > 0
        assert len(result.unstable_feature_indices) <= len(result.feature_importance_variance)


def test_feature_selector_keeps_uq_disabled_by_default():
    X, y = _dataset()
    selector = FeatureSelector(
        random_state=7,
        selection_strategy="legacy_voting",
        enabled_methods=("mutual_information", "anova_f"),
        n_bootstrap_iterations=1,
        importance_uq_enabled=False,
    )

    _, result = selector.fit_transform(X, y, n_final_features=10, return_result_object=True)
    assert bool(result.importance_uq.get("importance_uq_enabled", True)) is False
    assert bool(result.importance_uq.get("importance_uq_computed", True)) is False

import numpy as np
from sklearn.datasets import make_classification

from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.mnpo.oracles import compute_rashomon_importance_bounds


def test_compute_rashomon_importance_bounds_returns_feature_bounds():
    # Use a larger, more clearly separable dataset to be robust across
    # sklearn versions (1.6→1.8 deprecation of penalty/multi_class params).
    X, y = make_classification(
        n_samples=200,
        n_features=24,
        n_informative=10,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=21,
    )
    feature_idx = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    out = compute_rashomon_importance_bounds(
        X,
        y,
        feature_idx,
        random_state=7,
        max_models=12,
        score_tolerance=0.05,
        cv_splits=3,
    )
    assert bool(out["rashomon_computed"]) is True
    assert int(out["rashomon_n_models_total"]) >= 1
    assert int(out["rashomon_n_models_kept"]) >= 1
    bounds = dict(out.get("importance_bounds", {}))
    assert set(bounds.keys()) == {0, 1, 2, 3, 4, 5}
    for payload in bounds.values():
        assert float(payload["min"]) <= float(payload["max"])


def test_compute_rashomon_importance_bounds_handles_single_class():
    X = np.random.RandomState(0).normal(size=(20, 6))
    y = np.zeros(20, dtype=int)
    out = compute_rashomon_importance_bounds(
        X,
        y,
        np.array([0, 1, 2], dtype=int),
        random_state=3,
        max_models=5,
        score_tolerance=0.01,
        cv_splits=3,
    )
    assert bool(out["rashomon_computed"]) is False
    assert str(out["rashomon_reason"]) == "single_class"


def test_feature_selector_emits_rashomon_metadata_when_enabled():
    X, y = make_classification(
        n_samples=84,
        n_features=80,
        n_informative=12,
        n_redundant=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=33,
    )
    sel = FeatureSelector(
        random_state=17,
        n_bootstrap_iterations=1,
        inner_cv_splits=3,
        inner_cv_repeats=1,
        mirror_descent_steps=40,
        portfolio_size=4,
        enabled_methods={"mutual_information", "anova_f", "linear_svm"},
        rashomon_enabled=True,
        rashomon_max_models=6,
        rashomon_score_tolerance=0.02,
    )

    _, result = sel.fit_transform(X, y, n_final_features=12, return_result_object=True)
    mnpo = dict(result.method_results.get("mnpo_portfolio", {}) or {})
    rash = dict(mnpo.get("rashomon_importance", {}) or {})

    assert bool(rash.get("rashomon_enabled", False)) is True
    assert "rashomon_computed" in rash
    assert int(rash.get("rashomon_n_features", 0)) == 12

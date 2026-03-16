import builtins

import numpy as np
from sklearn.datasets import make_classification

from tabnetics.feature_selection.methods.embedded import treeshap_selection


def test_treeshap_skips_when_n_below_gate():
    X, y = make_classification(
        n_samples=40,
        n_features=30,
        n_classes=2,
        n_informative=8,
        random_state=1,
    )
    results, scores = treeshap_selection(
        X,
        y,
        12,
        problem_type="classification",
        random_state=3,
        n_bootstrap_iterations=3,
        min_samples=50,
    )
    assert bool(results.get("skipped", False)) is True
    assert results.get("skip_reason") == "min_n_gate"
    assert scores == {}


def test_treeshap_binary_smoke():
    X, y = make_classification(
        n_samples=120,
        n_features=50,
        n_classes=2,
        n_informative=12,
        random_state=7,
    )
    results, scores = treeshap_selection(
        X,
        y,
        15,
        problem_type="classification",
        random_state=11,
        n_bootstrap_iterations=3,
        min_samples=50,
        multi_seed_runs=2,
    )
    selected = np.asarray(results.get("selected_indices", np.array([], dtype=int)), dtype=int)
    assert selected.size == 15
    assert len(scores) == X.shape[1]
    assert np.isfinite(np.asarray(list(scores.values()), dtype=float)).all()


def test_treeshap_multiclass_smoke():
    X, y = make_classification(
        n_samples=140,
        n_features=60,
        n_classes=4,
        n_clusters_per_class=1,
        n_informative=20,
        random_state=13,
    )
    results, _ = treeshap_selection(
        X,
        y,
        18,
        problem_type="classification",
        random_state=5,
        n_bootstrap_iterations=3,
        min_samples=50,
    )
    selected = np.asarray(results.get("selected_indices", np.array([], dtype=int)), dtype=int)
    assert selected.size == 18


def test_treeshap_falls_back_when_shap_missing(monkeypatch):
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("forced missing shap")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    X, y = make_classification(
        n_samples=110,
        n_features=40,
        n_classes=2,
        n_informative=10,
        random_state=19,
    )
    results, _ = treeshap_selection(
        X,
        y,
        12,
        problem_type="classification",
        random_state=9,
        n_bootstrap_iterations=3,
        min_samples=50,
    )
    assert bool(results.get("shap_available", True)) is False

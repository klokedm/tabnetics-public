import warnings

import numpy as np
from sklearn.datasets import make_classification

from tabnetics.feature_selection.methods.embedded import oaenet_adaptive_selection


def test_oaenet_skips_when_n_below_gate():
    X, y = make_classification(
        n_samples=30,
        n_features=25,
        n_classes=2,
        n_informative=8,
        random_state=2,
    )
    results, scores = oaenet_adaptive_selection(
        X,
        y,
        10,
        problem_type="classification",
        random_state=4,
        min_samples=40,
    )
    assert bool(results.get("skipped", False)) is True
    assert results.get("skip_reason") == "min_n_gate"
    assert scores == {}


def test_oaenet_binary_smoke():
    X, y = make_classification(
        n_samples=140,
        n_features=120,
        n_classes=2,
        n_informative=20,
        random_state=8,
    )
    results, scores = oaenet_adaptive_selection(
        X,
        y,
        16,
        problem_type="classification",
        random_state=3,
        min_samples=40,
        prescreen_max_features=64,
    )
    selected = np.asarray(results.get("selected_indices", np.array([], dtype=int)), dtype=int)
    assert selected.size == 16
    assert len(scores) == X.shape[1]
    assert int(results.get("prescreen_size", 0)) <= int(results.get("prescreen_cap", 0))


def test_oaenet_multiclass_smoke():
    X, y = make_classification(
        n_samples=160,
        n_features=90,
        n_classes=4,
        n_clusters_per_class=1,
        n_informative=25,
        random_state=15,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=".*multi_class.*deprecated.*",
            category=FutureWarning,
        )
        results, _ = oaenet_adaptive_selection(
            X,
            y,
            20,
            problem_type="classification",
            random_state=7,
            min_samples=40,
        )
    selected = np.asarray(results.get("selected_indices", np.array([], dtype=int)), dtype=int)
    assert selected.size == 20


def test_oaenet_handles_zero_correlation_regime():
    rng = np.random.default_rng(21)
    X = rng.normal(0.0, 1.0, size=(120, 80))
    y = np.array([0] * 60 + [1] * 60, dtype=int)
    results, scores = oaenet_adaptive_selection(
        X,
        y,
        12,
        problem_type="classification",
        random_state=5,
        min_samples=40,
        prescreen_max_features=40,
    )
    selected = np.asarray(results.get("selected_indices", np.array([], dtype=int)), dtype=int)
    assert selected.size == 12
    assert np.isfinite(np.asarray(list(scores.values()), dtype=float)).all()

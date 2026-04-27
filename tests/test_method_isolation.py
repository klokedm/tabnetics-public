import numpy as np
import pytest
from sklearn.datasets import make_classification

from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.registry import METHOD_REGISTRY


def _make_dataset_for_spec(spec, seed: int = 19):
    if bool(getattr(spec, "binary_only", False)):
        n_classes = 2
    elif bool(getattr(spec, "requires_multiclass", False)):
        n_classes = 3
    else:
        n_classes = 3

    X, y = make_classification(
        n_samples=54,
        n_features=42,
        n_informative=10,
        n_redundant=4,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


@pytest.mark.parametrize("method_key", sorted(METHOD_REGISTRY.keys()))
def test_method_registry_dispatch_isolation(method_key):
    spec = METHOD_REGISTRY[method_key]
    X, y = _make_dataset_for_spec(spec, seed=101)

    fs = FeatureSelector(
        problem_type="classification",
        random_state=17,
        n_bootstrap_iterations=1,
        inner_cv_splits=2,
        inner_cv_repeats=1,
        mirror_descent_steps=20,
        method_timeout_seconds=2.0,
        linear_svm_max_iter=500,
        copula_knockoff_draws=2,
        enabled_methods={method_key},
    )

    results, runtimes = fs._run_selection_methods(
        X,
        y,
        n_target=8,
        class_pareto_min_classes=3,
    )

    assert method_key in runtimes
    # GPU-only methods may be skipped when no GPU is available.
    if bool(getattr(spec, "requires_gpu", False)) and not bool(fs._gpu_available):
        assert method_key not in results
        return

    assert method_key in results
    payload = results[method_key]
    assert isinstance(payload, tuple) and len(payload) == 2
    method_result, all_scores = payload
    assert isinstance(method_result, dict)
    assert isinstance(all_scores, dict)

    if "selected_indices" in method_result:
        idx = np.asarray(method_result.get("selected_indices", []), dtype=int)
        if idx.size:
            assert bool(np.all(idx >= 0)) is True
            assert bool(np.all(idx < X.shape[1])) is True

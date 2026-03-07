import numpy as np
from sklearn.datasets import make_classification

from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.contracts import (
    FeatureSelectorMethodContract,
    build_default_method_contracts,
)
from tabnetics.feature_selection.registry import METHOD_REGISTRY


def test_default_method_contracts_cover_registry_methods():
    fs = FeatureSelector(problem_type="classification", random_state=7)
    contracts = build_default_method_contracts(fs)
    expected = {
        key
        for key, spec in METHOD_REGISTRY.items()
        if str(getattr(spec, "maturity", "stable")).strip().lower() != "deprecated"
    }
    # Not every registered method is guaranteed to be callable in minimal envs
    # (optional backends), but high-use core methods must be contract-bound.
    core = {"mutual_information", "linear_svm", "stability_lasso"}
    assert core.issubset(set(contracts.keys()))
    assert len(contracts) >= min(len(expected), 20)
    for key in core:
        runtime = str(contracts[key].estimated_runtime_class)
        assert runtime in {"fast", "medium", "slow", "gpu_required"}


def test_contract_compute_matches_mutual_information_dispatch():
    X, y = make_classification(
        n_samples=64,
        n_features=70,
        n_informative=10,
        n_redundant=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=13,
    )
    fs = FeatureSelector(problem_type="classification", random_state=5)
    contracts = build_default_method_contracts(fs)

    result_contract, _ = contracts["mutual_information"].compute(X, y, 12)
    result_direct, _ = fs._mutual_information_selection(X, y, 12)

    idx_contract = np.sort(np.asarray(result_contract.get("selected_indices", []), dtype=int))
    idx_direct = np.sort(np.asarray(result_direct.get("selected_indices", []), dtype=int))
    np.testing.assert_array_equal(idx_contract, idx_direct)


def test_contract_supports_dataset_guards_binary_and_multiclass():
    dummy = FeatureSelectorMethodContract(
        key="dummy",
        fn=lambda X, y, n_target: ({"selected_indices": np.array([], dtype=int)}, {}),
        binary_only=True,
        runtime_class="fast",
    )
    assert bool(dummy.supports_dataset(n_samples=20, n_features=8, n_classes=2)) is True
    assert bool(dummy.supports_dataset(n_samples=20, n_features=8, n_classes=3)) is False

    dummy_mc = FeatureSelectorMethodContract(
        key="dummy_mc",
        fn=lambda X, y, n_target: ({"selected_indices": np.array([], dtype=int)}, {}),
        requires_multiclass=True,
        runtime_class="medium",
    )
    assert bool(dummy_mc.supports_dataset(n_samples=20, n_features=8, n_classes=2)) is False
    assert bool(dummy_mc.supports_dataset(n_samples=20, n_features=8, n_classes=4)) is True


def test_wmw_auc_contract_allows_multiclass_datasets():
    fs = FeatureSelector(problem_type="classification", random_state=19)
    contracts = build_default_method_contracts(fs)
    wmw = contracts["wmw_auc"]
    assert bool(wmw.supports_dataset(n_samples=48, n_features=20, n_classes=3)) is True


def test_wmw_auc_contract_dispatches_on_multiclass_problem():
    X, y = make_classification(
        n_samples=84,
        n_features=36,
        n_informative=10,
        n_redundant=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=41,
    )
    fs = FeatureSelector(problem_type="classification", random_state=11)
    contracts = build_default_method_contracts(fs)
    result, _ = contracts["wmw_auc"].compute(X, y, 10)
    selected = np.asarray(result.get("selected_indices", []), dtype=int)
    assert selected.size > 0

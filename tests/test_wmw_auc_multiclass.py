import numpy as np
from sklearn.datasets import make_classification

from tabnetics.feature_selection.methods.filter import wmw_auc_selection


def _make_multiclass(seed: int = 0):
    X, y = make_classification(
        n_samples=120,
        n_features=24,
        n_informative=10,
        n_redundant=2,
        n_repeated=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return X, y


def test_wmw_auc_multiclass_returns_expected_schema():
    X, y = _make_multiclass(seed=11)
    results, all_scores = wmw_auc_selection(X, y, n_target_features=8, problem_type="classification")
    assert "selected_indices" in results
    assert "scores" in results
    assert "all_scores" in results
    assert "m_metric_scores" in results
    assert "n_classes" in results
    assert results["n_classes"] == 4
    assert len(results["selected_indices"]) == 8
    assert len(all_scores) == X.shape[1]


def test_wmw_auc_multiclass_is_label_permutation_invariant():
    X, y = _make_multiclass(seed=12)
    results_a, _ = wmw_auc_selection(X, y, n_target_features=10, problem_type="classification")
    # Permute labels: 0->2, 1->3, 2->1, 3->0
    mapping = {0: 2, 1: 3, 2: 1, 3: 0}
    y_perm = np.vectorize(mapping.get)(y)
    results_b, _ = wmw_auc_selection(X, y_perm, n_target_features=10, problem_type="classification")
    np.testing.assert_allclose(
        np.asarray(results_a["all_scores"], dtype=float),
        np.asarray(results_b["all_scores"], dtype=float),
        atol=1e-12,
    )


def test_wmw_auc_multiclass_constant_feature_gets_zero_score():
    X, y = _make_multiclass(seed=13)
    X = np.asarray(X, dtype=float)
    X[:, 0] = 1.2345
    results, _ = wmw_auc_selection(X, y, n_target_features=6, problem_type="classification")
    assert float(results["all_scores"][0]) == 0.0


def test_wmw_auc_multiclass_rejects_non_classification_problem_type():
    X, y = _make_multiclass(seed=14)
    results, all_scores = wmw_auc_selection(X, y, n_target_features=6, problem_type="regression")
    assert results == {}
    assert all_scores == {}

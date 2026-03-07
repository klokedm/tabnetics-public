import numpy as np
from sklearn.datasets import make_classification

from tabnetics.feature_selection import FeatureSelector


def test_joint_auc_l1_selection_runs_on_binary_and_emits_best_c():
    X, y = make_classification(
        n_samples=72,
        n_features=40,
        n_informative=12,
        n_redundant=6,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=41,
    )
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    selector = FeatureSelector(
        n_bootstrap_iterations=1,
        random_state=41,
        problem_type="classification",
        selection_strategy="legacy_voting",
        enabled_methods=["joint_auc_l1"],
    )

    n_final = 10
    _, result = selector.fit_transform(X, y, n_final_features=n_final, return_result_object=True)
    method_res = result.method_results.get("joint_auc_l1")
    assert method_res is not None
    assert "best_c" in method_res
    assert "selected_indices" in method_res
    # FeatureSelector runs methods with an internal n_target that can exceed n_final.
    n_target = min(2 * n_final, max(n_final, X.shape[1] // 2))
    assert len(np.asarray(method_res["selected_indices"], dtype=int).ravel()) <= n_target

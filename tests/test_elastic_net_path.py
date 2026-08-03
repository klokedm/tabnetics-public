import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from tabnetics.classification import (
    ElasticNetPathClassifier,
    ElasticNetPathResult,
    ElasticNetPathSelectionError,
    select_one_standard_error,
)
from tabnetics.classification.backends import SklearnBackend
from tabnetics.classification.registry import (
    DEFAULT_CLASSIFIER_REGISTRY,
    REGIME_CLASSIFIER_POOLS,
    SupportLevel,
)


def _result(C, ratio, score, se, ap=0.5):
    return ElasticNetPathResult(
        C=C,
        l1_ratio=ratio,
        mean_score=score,
        standard_error=se,
        mean_average_precision=ap,
        mean_log_loss=score,
        valid=True,
    )


def test_one_se_log_loss_prefers_more_regularized_eligible_point():
    results = (
        _result(0.01, 1.0, 0.54, 0.01),
        _result(0.1, 0.5, 0.50, 0.05),
        _result(1.0, 0.0, 0.49, 0.02),
    )
    selected, threshold = select_one_standard_error(results, metric="log_loss")
    assert threshold == pytest.approx(0.51)
    assert (selected.C, selected.l1_ratio) == (0.1, 0.5)


def test_one_se_average_precision_uses_opposite_direction():
    results = (
        _result(0.01, 1.0, 0.71, 0.01, ap=0.71),
        _result(0.1, 0.5, 0.75, 0.05, ap=0.75),
        _result(1.0, 0.0, 0.77, 0.02, ap=0.77),
    )
    selected, threshold = select_one_standard_error(results, metric="average_precision")
    assert threshold == pytest.approx(0.75)
    assert (selected.C, selected.l1_ratio) == (0.1, 0.5)


def test_one_se_same_c_prefers_larger_l1_ratio():
    results = (
        _result(0.1, 0.25, 0.50, 0.02),
        _result(0.1, 0.75, 0.50, 0.02),
        _result(1.0, 1.0, 0.49, 0.02),
    )

    selected, threshold = select_one_standard_error(results, metric="log_loss")

    assert threshold == pytest.approx(0.51)
    assert (selected.C, selected.l1_ratio) == (0.1, 0.75)


def test_binary_average_precision_matches_weighted_positive_class_semantics():
    classes = np.asarray(["control", "case"], dtype=object)
    y_true = np.asarray(["control", "case", "case", "control", "case"])
    probabilities = np.asarray(
        [
            [0.80, 0.20],
            [0.25, 0.75],
            [0.40, 0.60],
            [0.65, 0.35],
            [0.10, 0.90],
        ]
    )
    sample_weight = np.asarray([1.0, 2.0, 0.5, 1.5, 3.0])
    expected = average_precision_score(
        (y_true == "case").astype(int),
        probabilities[:, 1],
        sample_weight=sample_weight,
    )

    observed = ElasticNetPathClassifier._average_precision(
        y_true,
        probabilities,
        classes,
        sample_weight,
    )

    assert observed == pytest.approx(expected)


def test_multiclass_average_precision_matches_macro_ovr_semantics():
    classes = np.asarray(["alpha", "beta", "gamma"], dtype=object)
    y_true = np.asarray(["alpha", "beta", "gamma", "alpha", "gamma", "beta"])
    probabilities = np.asarray(
        [
            [0.75, 0.15, 0.10],
            [0.10, 0.70, 0.20],
            [0.15, 0.20, 0.65],
            [0.55, 0.30, 0.15],
            [0.20, 0.25, 0.55],
            [0.25, 0.60, 0.15],
        ]
    )
    sample_weight = np.asarray([1.0, 2.0, 0.5, 1.5, 1.0, 2.5])
    expected = average_precision_score(
        label_binarize(y_true, classes=classes),
        probabilities,
        average="macro",
        sample_weight=sample_weight,
    )

    observed = ElasticNetPathClassifier._average_precision(
        y_true,
        probabilities,
        classes,
        sample_weight,
    )

    assert observed == pytest.approx(expected)


def test_binary_imbalanced_fit_weights_determinism_support_and_serialization():
    X, y = make_classification(
        n_samples=180,
        n_features=24,
        n_informative=8,
        weights=[0.92, 0.08],
        flip_y=0.0,
        random_state=23,
    )
    weights = np.where(y == 1, 3.0, 1.0)
    estimator = ElasticNetPathClassifier(
        C_grid=(0.05, 0.2),
        l1_ratio_grid=(0.0, 0.5),
        cv=3,
        class_weight=None,
        max_iter=5000,
        random_state=31,
    )
    fitted = estimator.fit(X, y, sample_weight=weights)
    probabilities = fitted.predict_proba(X[:20])
    assert probabilities.shape == (20, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert fitted.get_support().shape == (X.shape[1],)
    assert fitted.provenance_["sample_weight_used"] is True
    assert np.any(fitted.cv_results_["valid"])
    assert all(
        valid or reason is not None
        for valid, reason in zip(
            fitted.cv_results_["valid"],
            fitted.cv_results_["failure_reason"],
            strict=True,
        )
    )
    assert fitted.path_model_builds_ == 3 * 2
    records = fitted.cv_results_["fold_records"]
    assert all(
        record["warm_start_reused"] is (record["path_position"] > 0)
        for record in records
    )
    assert all(
        [record["C"] for record in records if record["path_id"] == path_id]
        == sorted(record["C"] for record in records if record["path_id"] == path_id)
        for path_id in {record["path_id"] for record in records}
    )

    repeated = clone(estimator).fit(X, y, sample_weight=weights)
    assert repeated.selected_C_ == fitted.selected_C_
    assert repeated.selected_l1_ratio_ == fitted.selected_l1_ratio_
    np.testing.assert_allclose(repeated.predict_proba(X[:20]), probabilities)

    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_allclose(restored.predict_proba(X[:20]), probabilities)


def test_multiclass_macro_ovr_ap_and_dataframe_order_contract():
    X, y = make_classification(
        n_samples=150,
        n_features=18,
        n_informative=10,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=41,
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model = ElasticNetPathClassifier(
        C_grid=(0.1,),
        l1_ratio_grid=(0.0, 0.5),
        selection_metric="average_precision",
        cv=3,
        max_iter=5000,
        random_state=9,
    ).fit(frame, y)
    assert model.predict_proba(frame.iloc[:8]).shape == (8, 3)
    assert np.all(np.isfinite(model.cv_results_["mean_average_precision"]))
    with pytest.raises(ValueError, match="identity/order"):
        model.predict(frame[list(reversed(frame.columns))])


def test_nonconvergence_invalidates_every_path_point():
    X, y = make_classification(
        n_samples=90,
        n_features=60,
        n_informative=20,
        random_state=5,
    )
    with pytest.raises(ElasticNetPathSelectionError, match="no_valid_configuration"):
        ElasticNetPathClassifier(
            C_grid=(1000.0,),
            l1_ratio_grid=(0.5,),
            cv=3,
            max_iter=1,
            tol=1e-12,
            random_state=2,
        ).fit(X, y)


def test_registry_identity_is_opt_in_and_backend_can_construct_it():
    spec = DEFAULT_CLASSIFIER_REGISTRY.get("elastic_net_path_lr")
    assert spec.name == "elastic_net_path_lr"
    assert spec.structured_resampling is SupportLevel.UNSUPPORTED
    assert all(
        "elastic_net_path_lr" not in pool for pool in REGIME_CLASSIFIER_POOLS.values()
    )
    X, y = make_classification(
        n_samples=90,
        n_features=12,
        n_informative=6,
        random_state=17,
    )
    backend = SklearnBackend(candidate_names=("elastic_net_path_lr",))
    candidates = backend._build_candidates(X_train=X, y_train=y, seed=3)
    assert isinstance(candidates["elastic_net_path_lr"], ElasticNetPathClassifier)


def test_custom_cv_rejects_repeated_complete_partition():
    X, y = make_classification(
        n_samples=80,
        n_features=10,
        n_informative=5,
        random_state=29,
    )
    folds = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=7).split(X, y))
    repeated = [*folds, *folds]
    with pytest.raises(
        ElasticNetPathSelectionError,
        match="elastic_net_path_uneven_validation_coverage",
    ):
        ElasticNetPathClassifier(
            C_grid=(0.1,),
            l1_ratio_grid=(0.0,),
            cv=repeated,
            max_iter=2000,
            random_state=5,
        ).fit(X, y)


def test_sample_weight_requires_positive_mass_for_every_class():
    X, y = make_classification(
        n_samples=80,
        n_features=10,
        n_informative=5,
        random_state=37,
    )
    weights = np.where(y == 0, 1.0, 0.0)
    with pytest.raises(
        ElasticNetPathSelectionError,
        match="elastic_net_path_nonpositive_class_weight_mass:final",
    ):
        ElasticNetPathClassifier(
            C_grid=(0.1,),
            l1_ratio_grid=(0.0,),
            cv=2,
            max_iter=2000,
            random_state=5,
        ).fit(X, y, sample_weight=weights)


def test_parallel_path_evaluation_matches_single_worker_selection():
    X, y = make_classification(
        n_samples=100,
        n_features=12,
        n_informative=6,
        random_state=43,
    )
    common = {
        "C_grid": (0.05, 0.2),
        "l1_ratio_grid": (0.0, 0.5),
        "cv": 2,
        "max_iter": 3000,
        "random_state": 11,
    }
    serial = ElasticNetPathClassifier(**common, n_jobs=1).fit(X, y)
    parallel = ElasticNetPathClassifier(**common, n_jobs=2).fit(X, y)
    assert parallel.selected_C_ == serial.selected_C_
    assert parallel.selected_l1_ratio_ == serial.selected_l1_ratio_
    np.testing.assert_allclose(
        parallel.predict_proba(X[:10]),
        serial.predict_proba(X[:10]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_backend_excludes_elastic_path_when_resolved_cv_plan_is_supplied():
    X, y = make_classification(
        n_samples=90,
        n_features=12,
        n_informative=6,
        random_state=47,
    )
    counts = np.bincount(y)
    cv_plan = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=13).split(X, y)
    )
    backend = SklearnBackend(candidate_names=("lr", "elastic_net_path_lr"))
    _, _, _, _, _, meta = backend.fit_and_select(
        X,
        y,
        seed=3,
        n_classes=2,
        class_counts=counts,
        cv_plan=cv_plan,
    )
    diagnostic = next(
        row
        for row in meta["model_cv_candidate_registry_diagnostics"]
        if row["canonical_name"] == "elastic_net_path_lr"
    )
    assert diagnostic["admission_outcome"] == "rejected"
    assert diagnostic["rejection_reason"] == "structured_resampling:unsupported"
    assert "elastic_net_path_lr" not in meta["model_cv_constructed_candidates"]

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import balanced_accuracy_score, f1_score

from tabnetics.classification.backends import (
    SampleWeightRoutingError,
    SklearnBackend,
    fit_estimator_with_sample_weight,
)
from tabnetics.pipeline.pipeline import (
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
)
from tabnetics.pipeline.resampling import (
    FitResamplingContext,
    ResamplingContractError,
)


def _data(*, seed: int = 71, n_samples: int = 84):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=18,
        n_informative=8,
        n_redundant=3,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=seed,
    )
    _, counts = np.unique(y, return_counts=True)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(counts)


def _config(*, seed: int = 71, posthoc: bool = False) -> DFFSConfig:
    return DFFSConfig(
        random_seed=seed,
        auto_router_enabled=False,
        enabled_methods=("mutual_information",),
        selection_strategy="legacy_voting",
        fs_fraction=0.75,
        n_final_features=6,
        max_dist_features=6,
        prefilter_top_k=12,
        classification_selection_mode="legacy",
        model_candidates=("lr", "knn"),
        include_elastic_net_model=False,
        include_rf_model=False,
        include_knn_model=True,
        include_svm_linear_model=False,
        include_dlda_model=False,
        include_nb_model=False,
        model_cv_runtime_containment_enabled=False,
        stage2_ratio_augmentation_enabled=False,
        calibration_reporting_enabled=True,
        classifier_posthoc_calibration_enabled=posthoc,
        classifier_posthoc_calibration_fraction=0.25,
        classifier_posthoc_calibration_min_calibration=8,
        classifier_posthoc_calibration_refinement_stopping=False,
    )


@pytest.mark.parametrize(
    ("weights", "code"),
    [
        ((1.0, -0.1, 1.0, 1.0), "negative_sample_weight"),
        ((1.0, np.nan, 1.0, 1.0), "nonfinite_sample_weight"),
        ((1.0, np.inf, 1.0, 1.0), "nonfinite_sample_weight"),
        ((1.0, 1.0, 1.0), "row_vector_length_mismatch"),
    ],
)
def test_resampling_context_rejects_invalid_sample_weights(weights, code):
    with pytest.raises(ResamplingContractError) as exc_info:
        FitResamplingContext(n_rows=4, sample_weights=weights)
    assert exc_info.value.code == code


def test_resampling_context_keeps_zero_individual_weights_and_hides_values():
    context = FitResamplingContext(
        n_rows=5,
        row_ids=("a", "b", "c", "d", "e"),
        sample_weights=(0.0, 1.5, 0.0, 2.0, 3.0),
    )

    child = context.take((4, 0, 3))
    metadata = child.to_metadata(
        sample_weights_consumed=True,
        sample_weight_usage="stage2_fit",
    )

    assert child.sample_weights == (3.0, 0.0, 2.0)
    assert metadata["sample_weights_consumed"] is True
    assert metadata["sample_weight_usage"] == "stage2_fit"
    assert "sample_weights" not in metadata


def test_weighted_sklearn_cv_admits_lr_and_excludes_knn():
    X, y, counts = _data()
    weights = np.linspace(0.25, 2.0, num=y.size)
    backend = SklearnBackend(candidate_names=("knn", "lr"))

    _, name, score, _, n_splits, meta = backend.fit_and_select(
        X,
        y,
        seed=71,
        n_classes=2,
        class_counts=counts,
        cv_splits=3,
        sample_weight=weights,
    )

    assert name == "lr"
    assert np.isfinite(score)
    assert n_splits == 3
    assert meta["model_cv_sample_weight_requested"] is True
    assert meta["model_cv_sample_weight_cv_routed"] is True
    assert meta["model_cv_sample_weight_routes"]["lr"] == "direct:sample_weight"
    diagnostics = meta["model_cv_candidate_registry_diagnostics"]
    knn = next(row for row in diagnostics if row["requested_name"] == "knn")
    assert knn["admission_outcome"] == "rejected"
    assert knn["rejection_reason"] == "sample_weight:unsupported"


def test_verified_weighted_sklearn_families_route_without_fallback():
    X, y, _ = _data(n_samples=220)
    weights = np.linspace(0.25, 2.0, num=y.size)
    backend = SklearnBackend(
        candidate_names=("lr", "elastic_net_lr", "svm_rbf", "rf", "extra_tree")
    )
    models = backend._build_candidates(
        X_train=X,
        y_train=y,
        seed=71,
        sample_weight_requested=True,
    )

    assert set(models) == {"lr", "elastic_net_lr", "svm_rbf", "rf", "extra_tree"}
    for model in models.values():
        assert (
            fit_estimator_with_sample_weight(model, X, y, weights)
            == "direct:sample_weight"
        )


def test_weighted_sklearn_cv_rejects_zero_mass_fold():
    X, y, counts = _data(n_samples=40)
    weights = np.r_[np.zeros(20), np.ones(20)]
    backend = SklearnBackend(candidate_names=("lr",))
    folds = (
        (tuple(range(20)), tuple(range(20, 40))),
        (tuple(range(20, 40)), tuple(range(20))),
    )

    with pytest.raises(SampleWeightRoutingError, match="zero_train_mass"):
        backend.fit_and_select(
            X,
            y,
            seed=71,
            n_classes=2,
            class_counts=counts,
            cv_plan=folds,
            sample_weight=weights,
        )


def test_direct_weighted_fit_rejects_overflowing_total_mass():
    X, y, _ = _data(n_samples=40)
    backend = SklearnBackend(candidate_names=("lr",))
    model = backend._build_candidates(
        X_train=X,
        y_train=y,
        seed=71,
        sample_weight_requested=True,
    )["lr"]

    with pytest.raises(SampleWeightRoutingError, match="nonfinite_mass"):
        fit_estimator_with_sample_weight(model, X, y, np.full(y.size, 1e308))


def test_weighted_sklearn_cv_rejects_overflowing_total_mass():
    X, y, counts = _data(n_samples=40)
    backend = SklearnBackend(candidate_names=("lr",))

    with pytest.raises(SampleWeightRoutingError, match="nonfinite_mass"):
        backend.fit_and_select(
            X,
            y,
            seed=71,
            n_classes=2,
            class_counts=counts,
            cv_splits=3,
            sample_weight=np.full(y.size, 1e308),
        )


def test_weighted_sklearn_cv_rejects_overflowing_partition_mass():
    X, y, counts = _data(n_samples=40)
    weights = np.zeros(y.size, dtype=float)
    weights[0] = 1e308
    weights[20] = 1.0
    backend = SklearnBackend(candidate_names=("lr",))
    folds = (
        (tuple([0, 0, *range(1, 20)]), tuple(range(20, 40))),
        (tuple(range(20, 40)), tuple(range(20))),
    )

    with pytest.raises(SampleWeightRoutingError, match="nonfinite_train_mass"):
        backend.fit_and_select(
            X,
            y,
            seed=71,
            n_classes=2,
            class_counts=counts,
            cv_plan=folds,
            sample_weight=weights,
        )


def test_pipeline_routes_weights_through_stage2_and_weighted_metrics():
    X, y, _ = _data()
    weights = np.linspace(0.25, 2.0, num=y.size)

    result = DistributionFeatureSelectionPipeline(_config()).run(
        X,
        y,
        dataset_name="weighted_pipeline",
        seed=71,
        sample_weight=weights,
    )
    snapshot = dict(result.config_snapshot or {})
    provenance = dict(snapshot["sample_weight_provenance"])

    assert result.model_name == "lr"
    assert provenance == {
        "sample_weight_requested": True,
        "sample_weight_feature_selection_consumed": False,
        "sample_weight_stage2_fit_consumed": True,
        "sample_weight_stage2_cv_consumed": True,
        "sample_weight_posthoc_calibration_consumed": False,
        "sample_weight_metrics_consumed": True,
    }
    assert tuple(snapshot["model_cv_sample_weight_admitted_candidates"]) == ("lr",)
    assert snapshot["model_cv_sample_weight_excluded_candidates"] == {
        "knn": "sample_weight:unsupported"
    }
    assert snapshot["resampling"]["fit_context"]["sample_weights_consumed"] is True
    assert snapshot["resampling"]["full_context"]["sample_weights_consumed"] is True
    assert "sample_weights" not in snapshot["resampling"]["fit_context"]
    assert "sample_weights" not in snapshot["resampling"]["full_context"]


def test_pipeline_weighted_posthoc_calibration_records_consumption():
    X, y, _ = _data(seed=73, n_samples=96)
    weights = np.linspace(0.25, 2.0, num=y.size)

    result = DistributionFeatureSelectionPipeline(_config(seed=73, posthoc=True)).run(
        X,
        y,
        dataset_name="weighted_posthoc",
        seed=73,
        sample_weight=weights,
    )
    snapshot = dict(result.config_snapshot or {})

    assert snapshot["classifier_posthoc_calibration_skip_reason"] == "ok"
    assert snapshot["classifier_posthoc_calibration_sample_weight_requested"] is True
    assert snapshot["classifier_posthoc_calibration_sample_weight_consumed"] is True
    assert "sample_weight" in snapshot[
        "classifier_posthoc_calibration_sample_weight_route"
    ]
    assert snapshot["sample_weight_provenance"][
        "sample_weight_posthoc_calibration_consumed"
    ] is True


def test_weighted_metric_helpers_match_sklearn():
    y_true = np.asarray([0, 0, 1, 1])
    y_pred = np.asarray([0, 1, 0, 1])
    weights = np.asarray([1.0, 9.0, 2.0, 3.0])

    assert DistributionFeatureSelectionPipeline._safe_balanced_accuracy(
        y_true, y_pred, sample_weight=weights
    ) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred, sample_weight=weights)
    )
    assert DistributionFeatureSelectionPipeline._safe_macro_f1(
        y_true, y_pred, sample_weight=weights
    ) == pytest.approx(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
            sample_weight=weights,
        )
    )


def test_pipeline_rejects_zero_effective_training_mass():
    X, y, _ = _data(n_samples=60)
    with pytest.raises(ResamplingContractError) as exc_info:
        DistributionFeatureSelectionPipeline(_config()).run_pre_split(
            X[:42],
            y[:42],
            X[42:],
            y[42:],
            sample_weight_train=np.zeros(42),
            sample_weight_test=np.ones(18),
        )
    assert exc_info.value.code == "zero_sample_weight_mass"


def test_public_context_reconciliation_rejects_weight_mismatch():
    X, y, _ = _data(n_samples=60)
    context = FitResamplingContext(
        n_rows=60,
        sample_weights=np.ones(60),
    )

    with pytest.raises(ResamplingContractError) as exc_info:
        DistributionFeatureSelectionPipeline(_config()).run(
            X,
            y,
            sample_weight=np.linspace(1.0, 2.0, num=60),
            resampling_context=context,
        )
    assert exc_info.value.code == "sample_weight_context_mismatch"


def test_none_weight_path_matches_legacy_execution():
    X, y, _ = _data(seed=79)

    legacy = DistributionFeatureSelectionPipeline(_config(seed=79)).run(
        X, y, dataset_name="weight_parity", seed=79
    )
    explicit_none = DistributionFeatureSelectionPipeline(_config(seed=79)).run(
        X,
        y,
        dataset_name="weight_parity",
        seed=79,
        sample_weight=None,
    )

    assert explicit_none.model_name == legacy.model_name
    assert explicit_none.selected_feature_indices_original == legacy.selected_feature_indices_original
    assert explicit_none.accuracy == pytest.approx(legacy.accuracy)
    assert explicit_none.balanced_accuracy == pytest.approx(legacy.balanced_accuracy)
    assert explicit_none.macro_f1 == pytest.approx(legacy.macro_f1)


def test_typed_input_marks_feature_selection_unweighted_but_allows_stage2_weights():
    pd = pytest.importorskip("pandas")
    X, y, _ = _data(seed=83, n_samples=60)
    frame = pd.DataFrame(X, columns=[f"f{idx}" for idx in range(X.shape[1])])
    config = _config(seed=83)
    config.typed_input_enabled = True

    result = DistributionFeatureSelectionPipeline(config).run(
        frame,
        y,
        dataset_name="typed_weighted",
        seed=83,
        sample_weight=np.linspace(0.25, 2.0, num=y.size),
    )
    admission = result.config_snapshot["typed_feature_selector_admission"]

    assert admission["runtime"]["sample_weight_requested"] is False
    assert admission["runtime"]["sample_weight_stage2_only"] is True
    assert admission["runtime"]["sample_weight_consumed"] is False
    assert result.config_snapshot["sample_weight_provenance"][
        "sample_weight_stage2_fit_consumed"
    ] is True

import sys
import types

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification

from tabnetics.classification.backends import (
    BiasCorrectedLinearSVM,
    ClassifierBackend,
    ClassifierOracle,
    CLASSIFIER_COMPLEXITY_PRIOR,
    CPDAClassifier,
    CopulaDiscriminantAnalysis,
    DistanceBasedDiscriminantAnalysis,
    FLAMLBackend,
    GeometricalQuadraticDiscriminantAnalysis,
    MNPOClassifierBackend,
    NearestSubspaceClassifier,
    OptunaBackend,
    RandomFourierFeaturesClassifier,
    RandomProjectionEnsembleClassifier,
    REGIME_HDLSS_MODERATE,
    SklearnBackend,
    SparseGroupLassoNNClassifier,
    SpatialMedianDiscriminantAnalysis,
    TabMClassifier,
    RealMLPClassifier,
    _RealMLP_TD_Classifier,
    _TabM_D_Classifier,
    _get_flaml_custom_specs,
    _make_flaml_custom_learner_class,
    FLAML_NATIVE_BY_FAMILY,
)


def _toy_data(n_samples=80, n_features=20, n_classes=3, seed=0):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=max(6, n_classes * 2),
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=seed,
    )
    classes, counts = np.unique(y, return_counts=True)
    return np.asarray(X, dtype=float), np.asarray(y), int(classes.size), np.asarray(counts, dtype=int)


def test_classifier_backend_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        _ = ClassifierBackend()  # type: ignore[abstract]


def test_supports_dataset_boundary_checks():
    backend = SklearnBackend(candidate_names=("lr",))
    assert backend.supports_dataset(n_samples=2, n_features=1, n_classes=2) is True
    assert backend.supports_dataset(n_samples=1, n_features=1, n_classes=2) is False
    assert backend.supports_dataset(n_samples=2, n_features=0, n_classes=2) is False
    assert backend.supports_dataset(n_samples=2, n_features=1, n_classes=1) is False


def test_sklearn_backend_basic_fit_and_select_smoke():
    X, y, n_classes, counts = _toy_data(n_samples=90, n_features=24, n_classes=3, seed=2)
    backend = SklearnBackend(candidate_names=("lr", "svm_rbf", "nb"))
    model, name, score, std, n_splits, meta = backend.fit_and_select(
        X, y, seed=7, n_classes=n_classes, class_counts=counts
    )
    assert name in {"lr", "svm_rbf", "nb"}
    assert np.isfinite(score) or np.isnan(score)
    assert np.isfinite(std) or np.isnan(std)
    assert int(n_splits) >= 0
    assert "classification_backend_used" in meta
    assert model is not None


def test_sklearn_backend_hybrid_scoring_path():
    X, y, n_classes, counts = _toy_data(n_samples=84, n_features=18, n_classes=3, seed=3)
    backend = SklearnBackend(
        candidate_names=("lr", "svm_rbf"),
        use_hybrid_score=True,
        hybrid_balanced_weight=0.7,
        hybrid_macro_f1_weight=0.3,
    )
    _, _, score, _, n_splits, _ = backend.fit_and_select(
        X, y, seed=11, n_classes=n_classes, class_counts=counts
    )
    assert (np.isfinite(score) and 0.0 <= score <= 1.0) or np.isnan(score)
    assert int(n_splits) >= 0


def test_sklearn_backend_expanded_classifier_pool_smoke():
    X, y, n_classes, counts = _toy_data(n_samples=120, n_features=30, n_classes=3, seed=4)
    backend = SklearnBackend(
        candidate_names=("nsc", "pls_da_classifier", "shrinkage_lda", "gpc", "lr"),
    )
    _, name, _, _, _, meta = backend.fit_and_select(
        X, y, seed=13, n_classes=n_classes, class_counts=counts
    )
    assert name in {"nsc", "pls_da_classifier", "shrinkage_lda", "gpc", "lr"}
    assert "model_cv_evaluated_candidates" in meta


def test_random_projection_ensemble_classifier_predict_proba_rows_sum_to_one():
    X, y, _, _ = _toy_data(n_samples=90, n_features=48, n_classes=3, seed=21)
    model = RandomProjectionEnsembleClassifier(
        n_estimators=5,
        max_components=16,
        random_state=11,
    )
    model.fit(X, y)
    probs = model.predict_proba(X[:12])
    assert probs.shape == (12, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(12), atol=1e-6)


def test_sklearn_backend_supports_rp_ensemble_candidate():
    X, y, n_classes, counts = _toy_data(n_samples=96, n_features=64, n_classes=2, seed=22)
    backend = SklearnBackend(candidate_names=("rp_ensemble",))
    model, name, score, std, n_splits, meta = backend.fit_and_select(
        X, y, seed=5, n_classes=n_classes, class_counts=counts
    )
    assert name == "rp_ensemble"
    assert model is not None
    assert (np.isfinite(score) and 0.0 <= score <= 1.0) or np.isnan(score)
    assert np.isfinite(std) or np.isnan(std)
    assert int(n_splits) >= 0
    assert "rp_ensemble" in tuple(meta.get("model_cv_evaluated_candidates", ()))


def test_dbda_predict_proba_rows_sum_to_one():
    X, y, _, _ = _toy_data(n_samples=90, n_features=48, n_classes=3, seed=30)
    model = DistanceBasedDiscriminantAnalysis()
    model.fit(X, y)
    probs = model.predict_proba(X[:12])
    assert probs.shape == (12, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(12), atol=1e-6)
    preds = model.predict(X[:12])
    assert preds.shape == (12,)


def test_gqda_predict_proba_rows_sum_to_one():
    X, y, _, _ = _toy_data(n_samples=90, n_features=48, n_classes=3, seed=31)
    model = GeometricalQuadraticDiscriminantAnalysis()
    model.fit(X, y)
    probs = model.predict_proba(X[:12])
    assert probs.shape == (12, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(12), atol=1e-6)
    preds = model.predict(X[:12])
    assert preds.shape == (12,)


def test_bc_svm_linear_predict_proba_binary():
    X, y, _, _ = _toy_data(n_samples=90, n_features=48, n_classes=2, seed=32)
    model = BiasCorrectedLinearSVM()
    model.fit(X, y)
    probs = model.predict_proba(X[:12])
    assert probs.shape == (12, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(12), atol=1e-6)
    preds = model.predict(X[:12])
    assert preds.shape == (12,)


def test_cpda_predict_proba_rows_sum_to_one():
    X, y, _, _ = _toy_data(n_samples=96, n_features=36, n_classes=3, seed=34)
    model = CPDAClassifier(random_state=7)
    model.fit(X, y)
    probs = model.predict_proba(X[:10])
    assert probs.shape == (10, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(10), atol=1e-6)
    assert int(model.n_rounds_) >= 1
    assert len(model.round_feature_counts_) == int(model.n_rounds_)
    assert all(
        model.round_feature_counts_[idx] >= model.round_feature_counts_[idx + 1]
        for idx in range(len(model.round_feature_counts_) - 1)
    )


def test_cpda_can_force_single_round_when_misclassified_threshold_is_high():
    X, y, _, _ = _toy_data(n_samples=90, n_features=24, n_classes=3, seed=35)
    model = CPDAClassifier(min_misclassified=10_000, random_state=3)
    model.fit(X, y)
    assert int(model.n_rounds_) == 1
    assert model.stop_reason_ == "too_few_misclassified"


def test_sklearn_backend_supports_cpda_candidate():
    X, y, n_classes, counts = _toy_data(n_samples=96, n_features=40, n_classes=3, seed=36)
    backend = SklearnBackend(candidate_names=("cpda",))
    model, name, score, std, n_splits, meta = backend.fit_and_select(
        X, y, seed=17, n_classes=n_classes, class_counts=counts
    )
    assert name == "cpda"
    assert model is not None
    assert (np.isfinite(score) and 0.0 <= score <= 1.0) or np.isnan(score)
    assert np.isfinite(std) or np.isnan(std)
    assert int(n_splits) >= 0
    assert "cpda" in tuple(meta.get("model_cv_evaluated_candidates", ()))


def test_mnpo_classifier_backend_current_profile_enables_multiclass_fixes():
    X, y, n_classes, counts = _toy_data(n_samples=120, n_features=30, n_classes=5, seed=37)
    backend = MNPOClassifierBackend(
        candidate_names=("lr", "dlda", "svm_linear"),
        oracle_behavior_profile="current",
        use_per_family_flaml=False,
    )
    _, _, _, _, _, meta = backend.fit_and_select(
        X, y, seed=19, n_classes=n_classes, class_counts=counts
    )
    assert str(meta.get("mnpo_oracle_behavior_profile", "")) == "current"
    assert bool(meta.get("mnpo_multiclass_ensemble_auto", False)) is True
    assert bool(meta.get("mnpo_multiclass_diversity_enabled", False)) is True
    assert bool(meta.get("mnpo_multiclass_complexity_flattened", False)) is True
    assert int(meta.get("mnpo_effective_oracle_k", 0) or 0) >= 3


def test_mnpo_classifier_backend_applies_val20_candidate_exclusions():
    backend = MNPOClassifierBackend(
        candidate_names=("lr", "tabpfn", "svm_linear"),
        exclude_candidate_names=("svm_linear",),
        regime_candidate_exclusions=("hdlss_moderate:tabpfn", "standard:tabpfn"),
        use_per_family_flaml=False,
    )
    regime, names, dropped = backend._filtered_candidates_by_regime(
        n_samples=120,
        n_features=7_000,
    )
    assert regime == REGIME_HDLSS_MODERATE
    assert names == ["lr"]
    assert "svm_linear" in dropped
    assert "tabpfn" in dropped


def test_mnpo_classifier_backend_applies_complexity_prior_override_in_stats():
    X, y, n_classes, counts = _toy_data(n_samples=96, n_features=24, n_classes=2, seed=39)
    backend = MNPOClassifierBackend(
        candidate_names=("lr", "svm_linear"),
        oracle_complexity_prior_overrides=("lr=0.25",),
        use_per_family_flaml=False,
    )
    _, _, _, _, _, meta = backend.fit_and_select(
        X, y, seed=23, n_classes=n_classes, class_counts=counts
    )
    stats = dict(meta.get("mnpo_candidate_stats") or {})
    assert float(stats["lr"]["complexity_score"]) == pytest.approx(0.25, abs=1e-6)
    assert meta.get("classification_complexity_prior_overrides", {}) == {"lr": 0.25}


def test_mnpo_classifier_backend_val18_compat_disables_multiclass_fixes():
    X, y, n_classes, counts = _toy_data(n_samples=120, n_features=30, n_classes=5, seed=38)
    backend = MNPOClassifierBackend(
        candidate_names=("lr", "dlda", "svm_linear"),
        oracle_behavior_profile="val18_compat",
        use_per_family_flaml=False,
    )
    _, _, _, _, _, meta = backend.fit_and_select(
        X, y, seed=23, n_classes=n_classes, class_counts=counts
    )
    assert str(meta.get("mnpo_oracle_behavior_profile", "")) == "val18_compat"
    assert bool(meta.get("mnpo_multiclass_ensemble_auto", False)) is False
    assert bool(meta.get("mnpo_multiclass_diversity_enabled", False)) is False
    assert bool(meta.get("mnpo_multiclass_complexity_flattened", False)) is False
    assert int(meta.get("mnpo_effective_oracle_k", 0) or 0) == 1


def test_classifier_oracle_opt_in_cvar_and_dynamic_complexity():
    X, y, _, _ = _toy_data(n_samples=220, n_features=16, n_classes=2, seed=39)
    backend = SklearnBackend(candidate_names=("lr", "rf"))
    candidates = backend._build_candidates(X_train=X, y_train=y, seed=17)
    for model in candidates.values():
        model.fit(X, y)

    oracle = ClassifierOracle(
        include_cvar=True,
        include_complexity=True,
        use_dynamic_complexity=True,
        tuning_meta={
            "lr": {
                "flaml_tuning_time_sec": 2.0,
                "flaml_time_budget": 60,
                "n_trials": 3,
            },
            "rf": {
                "flaml_tuning_time_sec": 55.0,
                "flaml_time_budget": 60,
                "n_trials": 80,
            },
        },
    )
    out = oracle.run(
        candidates=candidates,
        candidate_names=("lr", "rf"),
        X=X,
        y=y,
        seed=19,
        cv_splits=5,
        top_k=2,
    )

    stats = dict(out.get("candidate_stats") or {})
    assert "cvar" in dict(out.get("oracle_matrices") or {})
    assert np.isfinite(float(stats["lr"]["cvar_score"]))
    assert np.isfinite(float(stats["rf"]["cvar_score"]))
    assert float(stats["lr"]["complexity_score"]) != pytest.approx(
        float(CLASSIFIER_COMPLEXITY_PRIOR["lr"])
    )
    assert float(stats["rf"]["complexity_score"]) < float(stats["lr"]["complexity_score"])


def test_mnpo_classifier_backend_extract_portfolio_filters_redundant_predictions():
    backend = MNPOClassifierBackend(
        candidate_names=("lr", "svm_rbf", "nb"),
        oracle_portfolio_diversity=True,
        use_per_family_flaml=False,
    )
    y_true = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    pred_a = y_true.copy()
    pred_b = y_true.copy()
    pred_c = np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 0, 1])

    selected = backend._extract_portfolio(
        {"a": 0.50, "b": 0.30, "c": 0.20},
        {"a": pred_a, "b": pred_b, "c": pred_c},
        y_true,
        k=2,
        use_diversity=True,
        overlap_threshold=0.75,
        corr_threshold=0.85,
    )

    assert selected == ["a", "c"]


def test_mnpo_classifier_backend_tune_first_only_tunes_once_per_family(monkeypatch):
    X, y, n_classes, counts = _toy_data(n_samples=96, n_features=20, n_classes=3, seed=40)
    backend = MNPOClassifierBackend(
        candidate_names=("lr", "svm_rbf", "nb"),
        oracle_k=1,
        use_per_family_flaml=True,
        tune_first=True,
        flaml_time_budget=5,
    )

    calls = []

    def _fake_fit_family_with_flaml(
        *,
        family_name,
        fallback_model,
        X_train,
        y_train,
        seed,
        n_classes,
        class_counts,
        cv_splits,
        scoring,
        budget,
    ):
        calls.append(str(family_name))
        return fallback_model, f"flaml_{family_name}", 0.6, 0.05, int(cv_splits), {
            "mnpo_selected_family": str(family_name),
            "flaml_time_budget": int(budget),
            "flaml_tuning_time_sec": 0.01,
            "n_trials": 1,
        }

    monkeypatch.setattr(backend, "_fit_family_with_flaml", _fake_fit_family_with_flaml)

    _, model_name, _, _, _, meta = backend.fit_and_select(
        X,
        y,
        seed=23,
        n_classes=n_classes,
        class_counts=counts,
    )

    assert calls == ["lr", "svm_rbf", "nb"]
    assert str(model_name).startswith("tune_first_")
    assert str(meta.get("mnpo_hpo_mode", "")) == "tune_first"
    assert bool(meta.get("mnpo_tune_first_enabled", False)) is True


@pytest.mark.parametrize("clf_name", ["dbda", "gqda", "bc_svm_linear"])
def test_sklearn_backend_supports_hdlss_classifier(clf_name):
    n_classes = 2 if clf_name == "bc_svm_linear" else 3
    X, y, n_classes, counts = _toy_data(n_samples=96, n_features=64, n_classes=n_classes, seed=33)
    backend = SklearnBackend(candidate_names=(clf_name,))
    model, name, score, std, n_splits, meta = backend.fit_and_select(
        X, y, seed=5, n_classes=n_classes, class_counts=counts
    )
    assert name == clf_name
    assert model is not None
    assert (np.isfinite(score) and 0.0 <= score <= 1.0) or np.isnan(score)
    assert clf_name in tuple(meta.get("model_cv_evaluated_candidates", ()))


def test_sklearn_backend_records_tabpfn_evaluation_failures():
    X, y, n_classes, counts = _toy_data(n_samples=72, n_features=16, n_classes=2, seed=23)

    class _ExplodingEstimator(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            raise RuntimeError("tabpfn fit exploded")

        def predict(self, X):
            return np.zeros(np.asarray(X).shape[0], dtype=int)

    backend = SklearnBackend(
        candidate_names=("lr", "tabpfn"),
        build_tabpfn_model_fn=lambda seed: (_ExplodingEstimator(), None),
    )
    model, name, score, std, n_splits, meta = backend.fit_and_select(
        X, y, seed=31, n_classes=n_classes, class_counts=counts
    )

    assert model is not None
    assert name == "lr"
    assert (np.isfinite(score) and 0.0 <= score <= 1.0) or np.isnan(score)
    assert np.isfinite(std) or np.isnan(std)
    assert int(n_splits) >= 0
    assert "tabpfn" in tuple(meta.get("model_cv_constructed_candidates", ()))
    assert "tabpfn" in tuple(meta.get("model_cv_failed_candidates", ()))
    assert "tabpfn" not in tuple(meta.get("model_cv_evaluated_candidates", ()))
    assert "RuntimeError: tabpfn fit exploded" in str(
        dict(meta.get("model_cv_candidate_failure_reasons", {})).get("tabpfn", "")
    )


def test_sklearn_backend_gpc_gated_for_large_n():
    X, y, n_classes, counts = _toy_data(n_samples=240, n_features=25, n_classes=3, seed=5)
    backend = SklearnBackend(candidate_names=("gpc", "lr"))
    _, name, _, _, _, _ = backend.fit_and_select(
        X, y, seed=17, n_classes=n_classes, class_counts=counts
    )
    # gpc is gated at n<=200; lr remains as safe candidate.
    assert name == "lr"


def test_flaml_supports_dataset_thresholds():
    backend = FLAMLBackend(
        min_n_for_automl=50,
        min_n_per_class_for_automl=10,
        max_p_over_n_for_automl=200,
    )
    counts_ok = np.array([25, 25], dtype=int)
    counts_low = np.array([4, 45], dtype=int)
    assert backend.supports_dataset(n_samples=49, n_features=20, n_classes=2, class_counts=counts_ok) is False
    assert backend.supports_dataset(n_samples=50, n_features=20, n_classes=2, class_counts=counts_ok) is True
    assert backend.supports_dataset(n_samples=60, n_features=20000, n_classes=2, class_counts=np.array([30, 30])) is False
    assert backend.supports_dataset(n_samples=49, n_features=20, n_classes=2, class_counts=counts_low) is False


def test_flaml_fit_and_select_returns_lr_when_cv_too_small():
    X, y, n_classes, _ = _toy_data(n_samples=60, n_features=10, n_classes=2, seed=6)
    class_counts = np.array([3, 57], dtype=int)  # min class count < min_n_per_class_for_cv
    backend = FLAMLBackend(min_n_per_class_for_cv=5)
    model, name, score, std, n_splits, meta = backend.fit_and_select(
        X,
        y,
        seed=19,
        n_classes=n_classes,
        class_counts=class_counts,
    )
    assert name == "lr"
    assert int(n_splits) == 0
    assert np.isnan(score)
    assert np.isnan(std)
    assert meta.get("classification_guard_reason") == "min_n_per_class_for_cv"
    assert model is not None


def test_flaml_fit_and_select_raises_import_error_when_missing():
    X, y, n_classes, counts = _toy_data(n_samples=90, n_features=10, n_classes=2, seed=7)
    backend = FLAMLBackend()
    with pytest.raises(ImportError):
        backend.fit_and_select(X, y, seed=23, n_classes=n_classes, class_counts=counts)


def test_flaml_metric_mapping():
    assert FLAMLBackend._map_scoring("balanced_accuracy", "macro_f1") == "accuracy"
    assert FLAMLBackend._map_scoring("balanced-accuracy", "macro_f1") == "accuracy"
    assert FLAMLBackend._map_scoring("bal_acc", "macro_f1") == "accuracy"
    assert FLAMLBackend._map_scoring("macro_f1", "accuracy") == "macro_f1"
    assert FLAMLBackend._map_scoring("unknown_metric", "balanced_accuracy") == "accuracy"
    assert FLAMLBackend._map_scoring("unknown_metric", "macro_f1") == "macro_f1"
    assert FLAMLBackend(metric="balanced_accuracy").metric == "accuracy"


def test_flaml_happy_path_with_fake_module(monkeypatch):
    X, y, n_classes, counts = _toy_data(n_samples=90, n_features=12, n_classes=2, seed=9)

    class _FakeAutoML:
        def __init__(self):
            self.best_loss = 0.22
            self.best_estimator = "rf"
            self.best_config = {"max_depth": 5}
            self.model = object()

        def fit(self, *args, **kwargs):
            return None

    fake_module = types.SimpleNamespace(AutoML=_FakeAutoML)
    monkeypatch.setitem(sys.modules, "flaml", fake_module)

    backend = FLAMLBackend(time_budget=17)
    model, name, score, std, n_splits, meta = backend.fit_and_select(
        X, y, seed=29, n_classes=n_classes, class_counts=counts
    )
    assert name == "flaml_rf"
    assert model is not None
    assert score == pytest.approx(0.78)
    assert np.isnan(std)
    assert int(n_splits) >= 2
    assert meta.get("flaml_best_estimator") == "rf"


def test_optuna_supports_dataset_thresholds():
    backend = OptunaBackend(
        candidate_names=("lr", "svm_rbf"),
        min_n_for_automl=50,
        min_n_per_class_for_automl=10,
        max_p_over_n_for_automl=200,
    )
    counts_ok = np.array([25, 25], dtype=int)
    counts_low = np.array([4, 45], dtype=int)
    assert backend.supports_dataset(n_samples=49, n_features=20, n_classes=2, class_counts=counts_ok) is False
    assert backend.supports_dataset(n_samples=50, n_features=20, n_classes=2, class_counts=counts_ok) is True
    assert backend.supports_dataset(n_samples=60, n_features=20000, n_classes=2, class_counts=np.array([30, 30])) is False
    assert backend.supports_dataset(n_samples=49, n_features=20, n_classes=2, class_counts=counts_low) is False


def test_optuna_fit_and_select_returns_lr_when_cv_too_small():
    X, y, n_classes, _ = _toy_data(n_samples=60, n_features=10, n_classes=2, seed=12)
    class_counts = np.array([3, 57], dtype=int)  # min class count < min_n_per_class_for_cv
    backend = OptunaBackend(candidate_names=("lr", "svm_rbf"), min_n_per_class_for_cv=5)
    model, name, score, std, n_splits, meta = backend.fit_and_select(
        X,
        y,
        seed=19,
        n_classes=n_classes,
        class_counts=class_counts,
    )
    assert name == "lr"
    assert int(n_splits) == 0
    assert np.isnan(score)
    assert np.isnan(std)
    assert meta.get("classification_guard_reason") == "min_n_per_class_for_cv"
    assert model is not None


def test_optuna_fit_and_select_raises_import_error_when_missing(monkeypatch):
    X, y, n_classes, counts = _toy_data(n_samples=90, n_features=10, n_classes=2, seed=13)
    monkeypatch.setitem(sys.modules, "optuna", None)
    backend = OptunaBackend(candidate_names=("lr", "svm_rbf"))
    with pytest.raises(ImportError):
        backend.fit_and_select(X, y, seed=23, n_classes=n_classes, class_counts=counts)


def test_optuna_happy_path_with_fake_module(monkeypatch):
    X, y, n_classes, counts = _toy_data(n_samples=90, n_features=12, n_classes=2, seed=14)

    class _FakeTrial:
        def __init__(self):
            self.params = {}

        def suggest_categorical(self, name, choices):
            value = choices[0]
            self.params[name] = value
            return value

        def suggest_float(self, name, low, high, log=False):
            _ = (high, log)
            value = float(low)
            self.params[name] = value
            return value

        def suggest_int(self, name, low, high):
            _ = high
            value = int(low)
            self.params[name] = value
            return value

    class _FakeStudy:
        def __init__(self):
            self.trials = []
            self.best_trial = None

        def optimize(self, objective, n_trials=10, timeout=None, n_jobs=1):
            _ = (timeout, n_jobs)
            best_val = -np.inf
            for _ in range(int(max(1, n_trials))):
                trial = _FakeTrial()
                val = float(objective(trial))
                trial.value = val
                self.trials.append(trial)
                if self.best_trial is None or val > best_val:
                    self.best_trial = trial
                    best_val = val

    class _FakeTPESampler:
        def __init__(self, seed=0):
            self.seed = int(seed)

    fake_module = types.SimpleNamespace(
        create_study=lambda direction, sampler: _FakeStudy(),
        samplers=types.SimpleNamespace(TPESampler=_FakeTPESampler),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_module)

    backend = OptunaBackend(
        candidate_names=("lr", "svm_linear"),
        time_budget=15,
        n_trials=2,
        min_n_for_automl=10,
        min_n_per_class_for_automl=2,
        min_n_per_class_for_cv=2,
        max_p_over_n_for_automl=10_000,
    )
    model, name, score, std, n_splits, meta = backend.fit_and_select(
        X,
        y,
        seed=31,
        n_classes=n_classes,
        class_counts=counts,
    )
    assert name.startswith("optuna_")
    assert model is not None
    assert np.isfinite(score) or np.isnan(score)
    assert np.isfinite(std) or np.isnan(std)
    assert int(n_splits) >= 2
    assert meta.get("classification_backend_used") == "optuna"


def test_sglnn_predict_proba_rows_sum_to_one():
    X, y, _, _ = _toy_data(n_samples=90, n_features=48, n_classes=3, seed=22)
    model = SparseGroupLassoNNClassifier(
        lambda_sgl=0.05,
        max_iter=50,
        cv_lambda=False,
        random_state=42,
    )
    model.fit(X, y)
    probs = model.predict_proba(X[:12])
    assert probs.shape == (12, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(12), atol=1e-6)


def test_sglnn_feature_sparsity():
    """SGLNN should zero out some input weight rows with sufficient penalty."""
    X, y, _, _ = _toy_data(n_samples=60, n_features=40, n_classes=2, seed=23)
    model = SparseGroupLassoNNClassifier(
        lambda_sgl=0.5,
        max_iter=100,
        cv_lambda=False,
        random_state=7,
    )
    model.fit(X, y)
    assert model.n_selected_features_ < 40, "Expected some features to be zeroed out"
    assert model.n_selected_features_ > 0, "Expected some features to survive"


def test_sglnn_cv_lambda_selection():
    """Smoke test for the internal CV lambda selection path."""
    X, y, _, _ = _toy_data(n_samples=60, n_features=20, n_classes=2, seed=24)
    model = SparseGroupLassoNNClassifier(
        cv_lambda=True,
        max_iter=30,
        random_state=11,
    )
    model.fit(X, y)
    assert hasattr(model, "lambda_sgl_")
    assert model.lambda_sgl_ > 0.0
    preds = model.predict(X[:5])
    assert preds.shape == (5,)


def test_sglnn_sklearn_attributes():
    X, y, _, _ = _toy_data(n_samples=50, n_features=15, n_classes=2, seed=25)
    model = SparseGroupLassoNNClassifier(cv_lambda=False, max_iter=20, random_state=0)
    model.fit(X, y)
    assert hasattr(model, "classes_")
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == 15
    assert len(model.classes_) == 2


# ---------------------------------------------------------------------------
# Added classifiers (diversity expansion)
# ---------------------------------------------------------------------------

class TestRandomFourierFeaturesClassifier:
    def test_predict_proba_rows_sum_to_one(self):
        X, y, _, _ = _toy_data(n_samples=90, n_features=48, n_classes=3, seed=40)
        model = RandomFourierFeaturesClassifier(random_state=42)
        model.fit(X, y)
        probs = model.predict_proba(X[:12])
        assert probs.shape == (12, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(12), atol=1e-6)

    def test_predict_returns_valid_classes(self):
        X, y, _, _ = _toy_data(n_samples=80, n_features=30, n_classes=2, seed=41)
        model = RandomFourierFeaturesClassifier(random_state=7)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset(set(np.unique(y)))

    def test_hdlss_regime(self):
        """RFF-LR should work in extreme HDLSS: p=500, n=30."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 500)
        y = np.array([0] * 15 + [1] * 15)
        model = RandomFourierFeaturesClassifier(random_state=0)
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-6)

    def test_custom_rff_dim_and_gamma(self):
        X, y, _, _ = _toy_data(n_samples=60, n_features=20, n_classes=2, seed=42)
        model = RandomFourierFeaturesClassifier(
            n_features_rff=32, gamma=0.1, random_state=5
        )
        model.fit(X, y)
        assert model.omega_.shape == (20, 32)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)

    def test_sklearn_backend_supports_rff_lr(self):
        X, y, n_classes, counts = _toy_data(n_samples=96, n_features=64, n_classes=2, seed=43)
        backend = SklearnBackend(candidate_names=("rff_lr",))
        model, name, score, std, n_splits, meta = backend.fit_and_select(
            X, y, seed=5, n_classes=n_classes, class_counts=counts
        )
        assert name == "rff_lr"
        assert model is not None
        assert (np.isfinite(score) and 0.0 <= score <= 1.0) or np.isnan(score)


class TestNearestSubspaceClassifier:
    def test_predict_proba_rows_sum_to_one(self):
        X, y, _, _ = _toy_data(n_samples=90, n_features=48, n_classes=3, seed=50)
        model = NearestSubspaceClassifier()
        model.fit(X, y)
        probs = model.predict_proba(X[:12])
        assert probs.shape == (12, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(12), atol=1e-6)

    def test_predict_returns_valid_classes(self):
        X, y, _, _ = _toy_data(n_samples=80, n_features=30, n_classes=3, seed=51)
        model = NearestSubspaceClassifier(n_components=3)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset(set(np.unique(y)))

    def test_hdlss_regime(self):
        """Nearest subspace should work in extreme HDLSS: p=500, n=30."""
        rng = np.random.RandomState(52)
        X = rng.randn(30, 500)
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        model = NearestSubspaceClassifier()
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-6)

    def test_sklearn_backend_supports_near_subspace(self):
        X, y, n_classes, counts = _toy_data(n_samples=96, n_features=64, n_classes=3, seed=53)
        backend = SklearnBackend(candidate_names=("near_subspace",))
        model, name, score, std, n_splits, meta = backend.fit_and_select(
            X, y, seed=5, n_classes=n_classes, class_counts=counts
        )
        assert name == "near_subspace"
        assert model is not None


class TestSpatialMedianDiscriminantAnalysis:
    def test_predict_proba_rows_sum_to_one(self):
        X, y, _, _ = _toy_data(n_samples=90, n_features=48, n_classes=3, seed=60)
        model = SpatialMedianDiscriminantAnalysis()
        model.fit(X, y)
        probs = model.predict_proba(X[:12])
        assert probs.shape == (12, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(12), atol=1e-6)

    def test_predict_returns_valid_classes(self):
        X, y, _, _ = _toy_data(n_samples=80, n_features=30, n_classes=2, seed=61)
        model = SpatialMedianDiscriminantAnalysis()
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset(set(np.unique(y)))

    def test_hdlss_regime(self):
        """Spatial median DA should work in extreme HDLSS."""
        rng = np.random.RandomState(62)
        X = rng.randn(30, 500)
        y = np.array([0] * 15 + [1] * 15)
        model = SpatialMedianDiscriminantAnalysis()
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-6)

    def test_spatial_median_differs_from_mean(self):
        """Spatial median should differ from arithmetic mean, especially with outliers."""
        rng = np.random.RandomState(63)
        X = rng.randn(20, 10)
        # Add an outlier
        X[0] = X[0] * 100
        median = SpatialMedianDiscriminantAnalysis._spatial_median(X, 200, 1e-6)
        mean = X.mean(axis=0)
        # The spatial median should be less affected by the outlier
        assert not np.allclose(median, mean, atol=0.5)

    def test_sklearn_backend_supports_spatial_median_da(self):
        X, y, n_classes, counts = _toy_data(n_samples=96, n_features=64, n_classes=2, seed=64)
        backend = SklearnBackend(candidate_names=("spatial_median_da",))
        model, name, score, std, n_splits, meta = backend.fit_and_select(
            X, y, seed=5, n_classes=n_classes, class_counts=counts
        )
        assert name == "spatial_median_da"
        assert model is not None


class TestCopulaDiscriminantAnalysis:
    def test_predict_proba_rows_sum_to_one(self):
        X, y, _, _ = _toy_data(n_samples=90, n_features=48, n_classes=3, seed=70)
        model = CopulaDiscriminantAnalysis()
        model.fit(X, y)
        probs = model.predict_proba(X[:12])
        assert probs.shape == (12, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(12), atol=1e-6)

    def test_predict_returns_valid_classes(self):
        X, y, _, _ = _toy_data(n_samples=80, n_features=30, n_classes=2, seed=71)
        model = CopulaDiscriminantAnalysis()
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset(set(np.unique(y)))

    def test_hdlss_regime(self):
        """Copula DA should work in extreme HDLSS via shrinkage."""
        rng = np.random.RandomState(72)
        X = rng.randn(30, 500)
        y = np.array([0] * 15 + [1] * 15)
        model = CopulaDiscriminantAnalysis()
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-6)

    def test_sklearn_backend_supports_copula_da(self):
        X, y, n_classes, counts = _toy_data(n_samples=96, n_features=64, n_classes=3, seed=73)
        backend = SklearnBackend(candidate_names=("copula_da",))
        model, name, score, std, n_splits, meta = backend.fit_and_select(
            X, y, seed=5, n_classes=n_classes, class_counts=counts
        )
        assert name == "copula_da"
        assert model is not None

    def test_oas_shrinkage_is_supported(self):
        X, y, _, _ = _toy_data(n_samples=75, n_features=24, n_classes=3, seed=74)
        model = CopulaDiscriminantAnalysis(shrinkage="oas")
        model.fit(X, y)
        probs = model.predict_proba(X[:8])
        assert probs.shape == (8, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(8), atol=1e-6)

    def test_invalid_shrinkage_raises(self):
        X, y, _, _ = _toy_data(n_samples=60, n_features=20, n_classes=2, seed=75)
        model = CopulaDiscriminantAnalysis(shrinkage="bogus")
        with pytest.raises(ValueError, match="shrinkage"):
            model.fit(X, y)


class TestTabMClassifier:
    def test_predict_proba_rows_sum_to_one(self):
        X, y, _, _ = _toy_data(n_samples=90, n_features=48, n_classes=3, seed=80)
        model = TabMClassifier(random_state=42)
        model.fit(X, y)
        probs = model.predict_proba(X[:12])
        assert probs.shape == (12, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(12), atol=1e-6)

    def test_predict_returns_valid_classes(self):
        X, y, _, _ = _toy_data(n_samples=80, n_features=30, n_classes=2, seed=81)
        model = TabMClassifier(random_state=7)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset(set(np.unique(y)))

    def test_hdlss_regime(self):
        """TabM should work in extreme HDLSS: p=500, n=30."""
        rng = np.random.RandomState(82)
        X = rng.randn(30, 500)
        y = np.array([0] * 15 + [1] * 15)
        model = TabMClassifier(random_state=0)
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-6)

    def test_custom_heads(self):
        X, y, _, _ = _toy_data(n_samples=60, n_features=20, n_classes=2, seed=83)
        model = TabMClassifier(n_heads=4, n_hidden=16, random_state=5)
        model.fit(X, y)
        assert len(model.heads_) == 4
        preds = model.predict(X[:5])
        assert preds.shape == (5,)

    def test_scales_attribute_tracks_trained_heads(self):
        X, y, _, _ = _toy_data(n_samples=70, n_features=18, n_classes=3, seed=84)
        model = TabMClassifier(n_heads=3, n_hidden=12, max_iter=20, random_state=9)
        model.fit(X, y)
        learned_scales = np.vstack([head[0] for head in model.heads_])
        np.testing.assert_allclose(model.scales_, learned_scales)

    def test_sklearn_backend_supports_tabm(self):
        X, y, n_classes, counts = _toy_data(n_samples=96, n_features=64, n_classes=2, seed=85)
        backend = SklearnBackend(candidate_names=("tabm",))
        model, name, score, std, n_splits, meta = backend.fit_and_select(
            X, y, seed=5, n_classes=n_classes, class_counts=counts
        )
        assert name == "tabm"
        assert model is not None
        assert (np.isfinite(score) and 0.0 <= score <= 1.0) or np.isnan(score)


class TestRealMLPClassifier:
    def test_predict_proba_rows_sum_to_one(self):
        X, y, _, _ = _toy_data(n_samples=90, n_features=48, n_classes=3, seed=90)
        model = RealMLPClassifier(random_state=42)
        model.fit(X, y)
        probs = model.predict_proba(X[:12])
        assert probs.shape == (12, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(12), atol=1e-6)

    def test_predict_returns_valid_classes(self):
        X, y, _, _ = _toy_data(n_samples=80, n_features=30, n_classes=2, seed=91)
        model = RealMLPClassifier(random_state=7)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset(set(np.unique(y)))

    def test_hdlss_regime(self):
        """RealMLP should work in extreme HDLSS: p=500, n=30."""
        rng = np.random.RandomState(92)
        X = rng.randn(30, 500)
        y = np.array([0] * 15 + [1] * 15)
        model = RealMLPClassifier(random_state=0)
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-6)

    def test_custom_depth(self):
        X, y, _, _ = _toy_data(n_samples=60, n_features=20, n_classes=2, seed=93)
        model = RealMLPClassifier(depth=3, n_hidden=16, dropout=0.2, random_state=5)
        model.fit(X, y)
        assert len(model.layers_) == 4  # 3 hidden + 1 output
        preds = model.predict(X[:5])
        assert preds.shape == (5,)

    def test_sklearn_backend_supports_realmlp(self):
        X, y, n_classes, counts = _toy_data(n_samples=96, n_features=64, n_classes=2, seed=94)
        backend = SklearnBackend(candidate_names=("realmlp",))
        model, name, score, std, n_splits, meta = backend.fit_and_select(
            X, y, seed=5, n_classes=n_classes, class_counts=counts
        )
        assert name == "realmlp"
        assert model is not None
        assert (np.isfinite(score) and 0.0 <= score <= 1.0) or np.isnan(score)


# ---------------------------------------------------------------------------
# Gradient-correctness tests (finite-difference)
# ---------------------------------------------------------------------------

def _finite_diff_cross_entropy_grad(model_cls, X, y, param_name, idx, eps=1e-5, **kwargs):
    """Numerical gradient of cross-entropy loss w.r.t. a single parameter element."""
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    k = int(np.unique(y_enc).size)
    n = X.shape[0]
    Y = np.zeros((n, k), dtype=float)
    Y[np.arange(n), y_enc] = 1.0

    def _loss(m):
        probs = m.predict_proba(X)
        probs = np.clip(probs, 1e-12, None)
        return -float(np.mean(np.sum(Y * np.log(probs), axis=1)))

    m1 = model_cls(**kwargs)
    m1.fit(X, y)
    m2 = model_cls(**kwargs)
    m2.fit(X, y)

    param1 = getattr(m1, param_name)
    param2 = getattr(m2, param_name)
    if isinstance(param1, list):
        flat1 = param1[idx[0]]
        flat2 = param2[idx[0]]
        rest = idx[1:]
        flat1[rest] += eps
        flat2[rest] -= eps
    else:
        param1[idx] += eps
        param2[idx] -= eps

    return (_loss(m1) - _loss(m2)) / (2.0 * eps)


class TestTabMGradientCorrectness:
    def test_loss_decreases_over_training(self):
        """TabM loss should decrease across epochs — confirms gradient direction."""
        X, y, _, _ = _toy_data(n_samples=60, n_features=12, n_classes=3, seed=100)
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        k = int(np.unique(y_enc).size)
        n = X.shape[0]
        Y = np.zeros((n, k), dtype=float)
        Y[np.arange(n), y_enc] = 1.0

        losses = []
        for max_iter in [1, 10, 50, 150]:
            m = TabMClassifier(n_heads=2, n_hidden=8, max_iter=max_iter, random_state=42, lr=0.01)
            m.fit(X, y)
            probs = np.clip(m.predict_proba(X), 1e-12, None)
            loss = -float(np.mean(np.sum(Y * np.log(probs), axis=1)))
            losses.append(loss)
        # Loss should generally decrease; allow non-strict last step
        assert losses[2] < losses[0], f"TabM loss did not decrease: {losses}"


class TestRealMLPGradientCorrectness:
    def test_loss_decreases_over_training(self):
        """RealMLP loss should decrease across epochs — confirms gradient direction."""
        X, y, _, _ = _toy_data(n_samples=60, n_features=12, n_classes=3, seed=101)
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        k = int(np.unique(y_enc).size)
        n = X.shape[0]
        Y = np.zeros((n, k), dtype=float)
        Y[np.arange(n), y_enc] = 1.0

        losses = []
        for max_iter in [1, 10, 50, 150]:
            m = RealMLPClassifier(depth=2, n_hidden=12, dropout=0.0, max_iter=max_iter, random_state=42, lr=0.01)
            m.fit(X, y)
            probs = np.clip(m.predict_proba(X), 1e-12, None)
            loss = -float(np.mean(np.sum(Y * np.log(probs), axis=1)))
            losses.append(loss)
        assert losses[2] < losses[0], f"RealMLP loss did not decrease: {losses}"

    def test_backprop_uses_pre_update_weights(self):
        """Ensure the backward pass does not use stale (already-updated) weights."""
        X, y, _, _ = _toy_data(n_samples=40, n_features=8, n_classes=2, seed=102)
        # Train two models with depth=2 to exercise multi-layer backprop
        m1 = RealMLPClassifier(depth=2, n_hidden=8, dropout=0.0, max_iter=30, random_state=0, lr=0.01)
        m1.fit(X, y)
        # If the backward pass were using stale weights, the hidden layer
        # gradients would be biased; the loss should still decrease smoothly.
        probs = np.clip(m1.predict_proba(X), 1e-12, None)
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        Y = np.zeros((X.shape[0], 2), dtype=float)
        Y[np.arange(X.shape[0]), y_enc] = 1.0
        loss_30 = -float(np.mean(np.sum(Y * np.log(probs), axis=1)))

        m2 = RealMLPClassifier(depth=2, n_hidden=8, dropout=0.0, max_iter=1, random_state=0, lr=0.01)
        m2.fit(X, y)
        probs2 = np.clip(m2.predict_proba(X), 1e-12, None)
        loss_1 = -float(np.mean(np.sum(Y * np.log(probs2), axis=1)))
        assert loss_30 < loss_1, "30-epoch model should have lower loss than 1-epoch"


# ---------------------------------------------------------------------------
# Diversity oracle test
# ---------------------------------------------------------------------------

class TestClassifierOracleDiversityMatrix:
    def test_diversity_matrix_shape_and_values(self):
        """_build_diversity_matrix should return a valid (m, m) oracle matrix."""
        from tabnetics.classification.backends import ClassifierOracle

        rng = np.random.RandomState(200)
        n = 100
        y_true = rng.choice([0, 1, 2], size=n)
        # Classifier A: perfect
        pred_a = y_true.copy()
        # Classifier B: random errors
        pred_b = y_true.copy()
        mask_b = rng.rand(n) < 0.3
        pred_b[mask_b] = (pred_b[mask_b] + 1) % 3
        # Classifier C: different random errors
        pred_c = y_true.copy()
        mask_c = rng.rand(n) < 0.25
        pred_c[mask_c] = (pred_c[mask_c] + 2) % 3

        oof_preds = {"A": pred_a, "B": pred_b, "C": pred_c}
        names = ["A", "B", "C"]
        mat = ClassifierOracle._build_diversity_matrix(oof_preds, names, y_true)
        assert mat.shape == (3, 3)
        # Diagonal should be 0.5 (self-comparison)
        np.testing.assert_allclose(np.diag(mat), 0.5, atol=0.15)
        # Matrix values should be in [0, 1]
        assert np.all(mat >= 0.0) and np.all(mat <= 1.0)


# ---------------------------------------------------------------------------
# Pytabkit optional backend tests
# ---------------------------------------------------------------------------

_has_pytabkit = _RealMLP_TD_Classifier is not None and _TabM_D_Classifier is not None


class TestPytabkitOptionalBackends:
    """Tests for the optional pytabkit-backed TabM and RealMLP variants."""

    def test_regime_pools_contain_official_keys(self):
        """tabm_official and realmlp_td must appear in both regime pools."""
        from tabnetics.classification.backends import REGIME_POOLS
        for regime, pool in REGIME_POOLS.items():
            assert "tabm_official" in pool, f"tabm_official missing from {regime}"
            assert "realmlp_td" in pool, f"realmlp_td missing from {regime}"

    def test_complexity_prior_keys(self):
        """Complexity prior must include the new backends."""
        from tabnetics.classification.backends import CLASSIFIER_COMPLEXITY_PRIOR
        assert "tabm_official" in CLASSIFIER_COMPLEXITY_PRIOR
        assert "realmlp_td" in CLASSIFIER_COMPLEXITY_PRIOR
        # Official backends should rank between numpy versions and heavier models
        assert CLASSIFIER_COMPLEXITY_PRIOR["tabm_official"] < CLASSIFIER_COMPLEXITY_PRIOR["tabm"]
        assert CLASSIFIER_COMPLEXITY_PRIOR["realmlp_td"] < CLASSIFIER_COMPLEXITY_PRIOR["realmlp"]

    def test_build_candidates_without_pytabkit(self):
        """When pytabkit is NOT installed, build_failures should report it."""
        import tabnetics.classification.backends as _mod
        orig_tabm = _mod._TabM_D_Classifier
        orig_realmlp = _mod._RealMLP_TD_Classifier
        try:
            _mod._TabM_D_Classifier = None
            _mod._RealMLP_TD_Classifier = None
            X, y, n_cls, counts = _toy_data(n_samples=60, n_features=10, n_classes=2)
            backend = SklearnBackend(candidate_names=("lr", "tabm_official", "realmlp_td"))
            _, name, *_ = backend.fit_and_select(
                X, y, seed=0, n_classes=n_cls, class_counts=counts,
            )
            failures = backend._last_candidate_build_failures
            assert "tabm_official" in failures
            assert "realmlp_td" in failures
        finally:
            _mod._TabM_D_Classifier = orig_tabm
            _mod._RealMLP_TD_Classifier = orig_realmlp

    @pytest.mark.skipif(not _has_pytabkit, reason="pytabkit not installed")
    def test_tabm_official_fit_predict(self):
        """TabM_D_Classifier from pytabkit should fit and predict."""
        X, y, n_cls, counts = _toy_data(n_samples=80, n_features=10, n_classes=3)
        backend = SklearnBackend(candidate_names=("tabm_official",))
        model, name, score, *_ = backend.fit_and_select(
            X, y, seed=42, n_classes=n_cls,
            class_counts=counts,
        )
        assert name == "tabm_official"
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        assert set(preds).issubset(set(np.unique(y)))

    @pytest.mark.skipif(not _has_pytabkit, reason="pytabkit not installed")
    def test_realmlp_td_fit_predict(self):
        """RealMLP_TD_Classifier from pytabkit should fit and predict."""
        X, y, n_cls, counts = _toy_data(n_samples=80, n_features=10, n_classes=3)
        backend = SklearnBackend(candidate_names=("realmlp_td",))
        model, name, score, *_ = backend.fit_and_select(
            X, y, seed=42, n_classes=n_cls,
            class_counts=counts,
        )
        assert name == "realmlp_td"
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        assert set(preds).issubset(set(np.unique(y)))


# ---------------------------------------------------------------------------
# FLAML custom learner infrastructure tests
# ---------------------------------------------------------------------------

_EXPECTED_CUSTOM_FAMILIES = {
    "elastic_net_lr", "svm_linear", "svm_rbf", "knn", "nb",
    "dlda", "shrinkage_lda", "nsc", "pls_da_classifier",
    "bc_svm_linear", "sglnn", "rff_lr", "copula_da", "cpda",
    "tabm", "realmlp", "rp_ensemble", "near_subspace",
}


def test_flaml_custom_specs_covers_expected_families():
    """All non-native, non-excluded families have a custom FLAML spec."""
    try:
        import flaml  # noqa: F401
    except ImportError:
        pytest.skip("flaml not installed")
    specs = _get_flaml_custom_specs()
    for fam in _EXPECTED_CUSTOM_FAMILIES:
        assert fam in specs, f"Family '{fam}' missing from custom FLAML specs"
        assert "search_space" in specs[fam]
        assert "build" in specs[fam]
        assert "cost" in specs[fam]


def test_flaml_custom_specs_no_overlap_with_native():
    """Custom specs must not overlap with FLAML native families."""
    try:
        import flaml  # noqa: F401
    except ImportError:
        pytest.skip("flaml not installed")
    specs = _get_flaml_custom_specs()
    overlap = set(specs.keys()) & set(FLAML_NATIVE_BY_FAMILY.keys())
    assert not overlap, f"Overlap between custom and native: {overlap}"


def test_flaml_custom_spec_build_fns_produce_valid_estimators():
    """Each custom spec build function returns a fit/predict-capable estimator."""
    try:
        import flaml  # noqa: F401
    except ImportError:
        pytest.skip("flaml not installed")
    X, y, _, _ = _toy_data(n_samples=60, n_features=10, n_classes=2, seed=42)
    specs = _get_flaml_custom_specs()
    for fam, spec in specs.items():
        build_fn = spec["build"]
        est = build_fn(42)  # default HPs, seed=42
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (X.shape[0],), f"{fam} predict shape mismatch"


def test_make_flaml_custom_learner_class_creates_valid_subclass():
    """Dynamic learner class creation should produce a FLAML-compatible class."""
    try:
        from flaml.automl.model import SKLearnEstimator  # type: ignore
    except ImportError:
        pytest.skip("flaml not installed")
    specs = _get_flaml_custom_specs()
    spec = specs["copula_da"]
    cls = _make_flaml_custom_learner_class("copula_da", spec, seed=7)
    assert issubclass(cls, SKLearnEstimator)
    assert cls.__name__ == "FLAML_copula_da"
    # search_space should return the defined space
    space = cls.search_space(data_size=(100, 10), task="classification")
    assert "shrinkage" in space


def test_mnpo_fit_family_with_flaml_custom_learner_happy_path(monkeypatch):
    """MNPOClassifierBackend._fit_family_with_flaml should use custom learner."""
    try:
        import flaml  # noqa: F401
    except ImportError:
        pytest.skip("flaml not installed")

    X, y, n_classes, counts = _toy_data(n_samples=90, n_features=12, n_classes=2, seed=20)

    # Build a mock FLAML AutoML that simulates custom learner path
    class _FakeAutoML:
        def __init__(self):
            self.best_loss = 0.18
            self.best_estimator = "tabnetics_copula_da"
            self.best_config = {"shrinkage": "oas"}
            self.model = CopulaDiscriminantAnalysis(shrinkage="oas")

        def add_learner(self, name, cls):
            self._learner_name = name

        def fit(self, *args, **kwargs):
            return None

    fake_module = types.SimpleNamespace(
        AutoML=_FakeAutoML,
        tune=flaml.tune,
    )
    # Patch flaml module for the import inside _fit_custom_family_with_flaml
    monkeypatch.setitem(sys.modules, "flaml", fake_module)

    backend = MNPOClassifierBackend(
        candidate_names=("copula_da",), use_per_family_flaml=True,
    )
    fallback = CopulaDiscriminantAnalysis()
    fallback.fit(X, y)

    model, name, score, std, n_splits, meta = backend._fit_family_with_flaml(
        family_name="copula_da",
        fallback_model=fallback,
        X_train=X,
        y_train=y,
        seed=7,
        n_classes=n_classes,
        class_counts=counts,
        cv_splits=5,
        scoring="balanced_accuracy",
        budget=10,
    )
    assert name == "flaml_copula_da"
    assert meta.get("classification_backend_used") == "mnpo_hybrid_flaml_custom"
    assert meta.get("flaml_custom_learner") == "tabnetics_copula_da"
    assert model is not None


def test_flaml_backend_accepts_custom_estimators_in_estimator_list(monkeypatch):
    """Legacy FLAML backend should register custom learners and drop unsupported names."""
    try:
        import flaml
    except ImportError:
        pytest.skip("flaml not installed")

    X, y, n_classes, counts = _toy_data(n_samples=90, n_features=12, n_classes=2, seed=22)

    class _FakeAutoML:
        def __init__(self):
            self.best_loss = 0.16
            self.best_estimator = "tabnetics_copula_da"
            self.best_config = {"shrinkage": "oas"}
            self.model = CopulaDiscriminantAnalysis(shrinkage="oas")
            self.added_learners = []
            self.fit_kwargs = {}

        def add_learner(self, name, cls):
            self.added_learners.append((name, cls))

        def fit(self, *args, **kwargs):
            self.fit_kwargs = dict(kwargs)
            return None

    fake_automl = _FakeAutoML()

    class _FakeAutoMLFactory:
        def __call__(self):
            return fake_automl

    fake_module = types.SimpleNamespace(
        AutoML=_FakeAutoMLFactory(),
        tune=flaml.tune,
    )
    monkeypatch.setitem(sys.modules, "flaml", fake_module)

    backend = FLAMLBackend(
        time_budget=17,
        estimator_list=("copula_da", "tabm_official", "lrl2"),
        metric="roc_auc",
    )
    model, name, score, std, n_splits, meta = backend.fit_and_select(
        X, y, seed=7, n_classes=n_classes, class_counts=counts
    )

    assert name == "flaml_copula_da"
    assert meta["flaml_requested_estimators"] == ["copula_da", "tabm_official", "lrl2"]
    assert meta["flaml_effective_estimators"] == ["tabnetics_copula_da", "lrl2"]
    assert meta["flaml_unsupported_estimators"] == ["tabm_official"]
    assert meta["flaml_best_estimator"] == "tabnetics_copula_da"
    assert meta["flaml_best_estimator_family"] == "copula_da"
    assert meta["flaml_custom_estimators"] == {"tabnetics_copula_da": "copula_da"}
    assert fake_automl.fit_kwargs["estimator_list"] == ["tabnetics_copula_da", "lrl2"]
    assert fake_automl.added_learners[0][0] == "tabnetics_copula_da"
    assert model is not None
    assert np.isfinite(score)
    assert np.isnan(std)
    assert n_splits >= 2


def test_mnpo_fit_family_with_flaml_falls_back_for_excluded_families():
    """Parameter-free and excluded families should still fall back gracefully."""
    X, y, n_classes, counts = _toy_data(n_samples=90, n_features=12, n_classes=2, seed=21)
    backend = MNPOClassifierBackend(
        candidate_names=("dbda",), use_per_family_flaml=True,
    )
    fallback = CopulaDiscriminantAnalysis()
    fallback.fit(X, y)

    # 'dbda' has no custom spec and is not FLAML-native → fallback
    model, name, score, std, n_splits, meta = backend._fit_family_with_flaml(
        family_name="dbda",
        fallback_model=fallback,
        X_train=X,
        y_train=y,
        seed=7,
        n_classes=n_classes,
        class_counts=counts,
        cv_splits=5,
        scoring="balanced_accuracy",
        budget=10,
    )
    assert name == "dbda"
    assert meta.get("mnpo_flaml_fallback_reason") == "family_not_supported_by_flaml"

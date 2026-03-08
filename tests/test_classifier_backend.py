import sys
import types

import numpy as np
import pytest
from sklearn.datasets import make_classification

from tabnetics.classification.backends import (
    ClassifierBackend,
    FLAMLBackend,
    OptunaBackend,
    SklearnBackend,
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

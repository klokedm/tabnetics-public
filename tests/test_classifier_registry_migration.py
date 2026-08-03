import hashlib
import json
import sys
import types
from dataclasses import replace

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import tabnetics.classification.backends as backend_module
from tabnetics.classification.backends import (
    CLASSIFIER_COMPLEXITY_PRIOR,
    FLAML_NATIVE_BY_FAMILY,
    REGIME_POOLS,
    ClassifierCandidateAdmissionError,
    FLAMLBackend,
    MNPOClassifierBackend,
    OptunaBackend,
    SklearnBackend,
)
from tabnetics.classification.registry import (
    CLASSIFIER_COMPLEXITY_PRIORS,
    CLASSIFIER_SPECS,
    FLAML_NATIVE_TUNING_KEYS,
    REGIME_CLASSIFIER_POOLS,
    get_classifier_spec,
)


def _toy_data(
    *, n_samples: int = 60, n_features: int = 8, n_classes: int = 2, seed: int = 0
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(3, n_classes),
        n_redundant=1,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    _, counts = np.unique(y, return_counts=True)
    return np.asarray(X, dtype=float), np.asarray(y), np.asarray(counts, dtype=int)


def _snapshot_digest(value) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def test_registry_compatibility_views_match_legacy_snapshots_and_are_immutable():
    pools = [(name, list(values)) for name, values in REGIME_POOLS.items()]
    priors = list(CLASSIFIER_COMPLEXITY_PRIOR.items())
    flaml = list(FLAML_NATIVE_BY_FAMILY.items())

    assert REGIME_POOLS is REGIME_CLASSIFIER_POOLS
    assert CLASSIFIER_COMPLEXITY_PRIOR is CLASSIFIER_COMPLEXITY_PRIORS
    assert FLAML_NATIVE_BY_FAMILY is FLAML_NATIVE_TUNING_KEYS
    assert _snapshot_digest(pools) == (
        "03f954d077ff7b62ce1389436eede7a38cf12e2a68393b43f023b6697119f2f6"
    )
    assert _snapshot_digest(priors) == (
        "2e4bdac0a81a20f45a08d367f3408c4f6b5b0b21780405916c8826c610cd7823"
    )
    assert _snapshot_digest(flaml) == (
        "a50f2fa51a458e005e4340b8f50d4422e09d729b3761d47ad710c3a5e74e5255"
    )
    assert tuple(FLAML_NATIVE_BY_FAMILY.items()) == (
        ("lr", "lrl2"),
        ("rf", "rf"),
        ("xgb", "xgboost"),
        ("lgbm", "lgbm"),
        ("extra_tree", "extra_tree"),
        ("catboost", "catboost"),
    )
    with pytest.raises(TypeError):
        REGIME_POOLS["standard"] = ("lr",)  # type: ignore[index]
    with pytest.raises(TypeError):
        CLASSIFIER_COMPLEXITY_PRIOR["lr"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        FLAML_NATIVE_BY_FAMILY["lr"] = "changed"  # type: ignore[index]


def test_tree_identity_is_explicit_and_exactly_registry_derived():
    expected = {"rf", "extra_tree", "xgb", "lgbm", "catboost"}
    assert {name for name, spec in CLASSIFIER_SPECS.items() if spec.tree_model} == expected
    assert backend_module._TREE_MODEL_NAMES == frozenset(expected)
    assert all(type(spec.tree_model) is bool for spec in CLASSIFIER_SPECS.values())
    with pytest.raises(TypeError, match="tree_model"):
        replace(get_classifier_spec("lr"), tree_model=1)


def test_sklearn_mixed_unknown_requests_are_ordered_visible_and_json_safe():
    X, y, _ = _toy_data()
    backend = SklearnBackend(
        candidate_names=("missing", "LR", "lr", "lr", " lr")
    )
    _, _, _, _, _, meta = backend.fit_and_select(
        X,
        y,
        seed=1,
        n_classes=2,
        class_counts=np.asarray([1, 1]),
    )
    diagnostics = meta["model_cv_candidate_registry_diagnostics"]

    assert [value["requested_name"] for value in diagnostics] == [
        "missing",
        "LR",
        "lr",
        "lr",
        " lr",
    ]
    assert [value["admission_outcome"] for value in diagnostics] == [
        "rejected",
        "rejected",
        "admitted",
        "rejected",
        "rejected",
    ]
    assert diagnostics[0]["rejection_reason"] == "registry:unknown_classifier"
    assert diagnostics[2]["static"]["builder_key"] == "lr"
    assert diagnostics[2]["static"]["estimator_sparse_input"] == "supported"
    assert diagnostics[2]["resolved"]["effective_sparse_input"] == "unsupported"
    assert diagnostics[2]["build_outcome"] == "constructed"
    assert diagnostics[2]["fit_outcome"] == "not_observed"
    json.dumps(diagnostics, allow_nan=False)


def test_sklearn_all_unknown_raises_with_same_persisted_diagnostics():
    X, y, counts = _toy_data()
    backend = SklearnBackend(candidate_names=("missing", "LR", "lr "))

    with pytest.raises(ClassifierCandidateAdmissionError) as caught:
        backend.fit_and_select(
            X, y, seed=2, n_classes=2, class_counts=counts, cv_splits=2
        )

    assert caught.value.diagnostics == backend.get_candidate_registry_diagnostics()
    assert [value["canonical_name"] for value in caught.value.diagnostics] == [
        None,
        None,
        None,
    ]
    json.dumps(caught.value.diagnostics, allow_nan=False)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (("dlda", "shrinkage_lda"), "dlda"),
        (("shrinkage_lda", "dlda"), "shrinkage_lda"),
        (("shrinkage_lda",), "shrinkage_lda"),
    ],
)
def test_alias_dispatch_uses_canonical_builder_and_preserves_first_spelling(
    requested, expected
):
    X, y, _ = _toy_data()
    backend = SklearnBackend(candidate_names=requested)
    models = backend._build_candidates(X_train=X, y_train=y, seed=3)
    diagnostics = backend.get_candidate_registry_diagnostics()

    assert list(models) == [expected]
    assert diagnostics[0]["canonical_name"] == "dlda"
    assert diagnostics[0]["static"]["builder_key"] == "dlda"
    if len(requested) == 2:
        assert diagnostics[1]["rejection_reason"] == (
            "registry:duplicate_canonical_request:dlda"
        )


def test_callback_absence_failure_and_success_have_distinct_resolved_build_state():
    X, y, _ = _toy_data()

    absent = SklearnBackend(candidate_names=("tabpfn",))
    assert absent._build_candidates(X_train=X, y_train=y, seed=4) == {}
    absent_diag = absent.get_candidate_registry_diagnostics()[0]
    assert absent_diag["resolved"]["dependency_status"] == "conditional"
    assert absent_diag["resolved"]["builder_status"] == "unsupported"
    assert absent_diag["rejection_reason"] == "builder:tabpfn:unavailable"

    failed = SklearnBackend(
        candidate_names=("tabpfn",),
        build_tabpfn_model_fn=lambda seed: (None, "checkpoint unavailable"),
    )
    assert failed._build_candidates(X_train=X, y_train=y, seed=4) == {}
    failed_diag = failed.get_candidate_registry_diagnostics()[0]
    assert failed_diag["resolved"]["availability"] == "conditional"
    assert failed_diag["build_outcome"] == "failed"
    assert failed_diag["build_reason"] == "checkpoint unavailable"

    success = SklearnBackend(
        candidate_names=("tabpfn",),
        build_tabpfn_model_fn=lambda seed: LogisticRegression(),
    )
    assert list(success._build_candidates(X_train=X, y_train=y, seed=4)) == [
        "tabpfn"
    ]
    success_diag = success.get_candidate_registry_diagnostics()[0]
    assert success_diag["resolved"]["availability"] == "conditional"
    assert success_diag["build_outcome"] == "constructed"


def test_missing_optional_packages_preserve_legacy_warning_and_failure_order(monkeypatch):
    X = np.zeros((220, 4), dtype=float)
    y = np.asarray([0, 1] * 110)
    warnings = []
    monkeypatch.setattr(backend_module, "LGBMClassifier", None)
    monkeypatch.setattr(backend_module, "CatBoostClassifier", None)
    monkeypatch.setattr(backend_module, "_TabM_D_Classifier", None)
    monkeypatch.setattr(backend_module, "_RealMLP_TD_Classifier", None)
    backend = SklearnBackend(
        candidate_names=(
            "realmlp_td",
            "tabm_official",
            "catboost",
            "lgbm",
            "lr",
        ),
        warn_missing_backend_fn=lambda name, package, reason: warnings.append(
            (name, package, reason)
        ),
    )

    models = backend._build_candidates(X_train=X, y_train=y, seed=5)

    assert list(models) == ["lr"]
    assert [value[0] for value in warnings] == [
        "lgbm",
        "catboost",
        "tabm_official",
        "realmlp_td",
    ]
    assert backend._last_candidate_build_failures == {
        "tabm_official": "pytabkit not installed",
        "realmlp_td": "pytabkit not installed",
    }


def test_sklearn_requested_candidate_and_tie_order_remains_unchanged():
    X, y, _ = _toy_data()
    backend = SklearnBackend(candidate_names=("nb", "lr", "svm_linear"))
    _, name, _, _, _, meta = backend.fit_and_select(
        X,
        y,
        seed=6,
        n_classes=2,
        class_counts=np.asarray([1, 1]),
    )
    assert meta["model_cv_constructed_candidates"] == ("nb", "lr", "svm_linear")
    assert name == "lr"


def test_flaml_alias_uses_canonical_custom_tuning_identity(monkeypatch):
    flaml = pytest.importorskip("flaml")
    pytest.importorskip("flaml.automl.model")
    X, y, counts = _toy_data(n_samples=80)

    class FakeAutoML:
        best_loss = 0.2
        best_estimator = "tabnetics_dlda"
        best_config = {}
        best_iter = 0
        model = LogisticRegression()

        def __init__(self):
            self.added = []
            self.fit_kwargs = {}

        def add_learner(self, name, learner):
            self.added.append((name, learner))

        def fit(self, *args, **kwargs):
            self.fit_kwargs = dict(kwargs)

    fake = FakeAutoML()
    monkeypatch.setattr(backend_module, "_flaml_custom_specs_cache", None)
    monkeypatch.setitem(
        sys.modules,
        "flaml",
        types.SimpleNamespace(AutoML=lambda: fake, tune=flaml.tune),
    )
    backend = FLAMLBackend(
        estimator_list=("shrinkage_lda",),
        min_n_for_automl=2,
        min_n_per_class_for_automl=2,
        min_n_per_class_for_cv=2,
    )

    _, name, _, _, _, meta = backend.fit_and_select(
        X, y, seed=7, n_classes=2, class_counts=counts, cv_splits=2
    )

    assert name == "flaml_shrinkage_lda"
    assert fake.added[0][0] == "tabnetics_dlda"
    assert fake.fit_kwargs["estimator_list"] == ["tabnetics_dlda"]
    diagnostic = meta["flaml_candidate_registry_diagnostics"][0]
    assert diagnostic["canonical_name"] == "dlda"
    assert diagnostic["static"]["tuning_key"] == "dlda"
    assert diagnostic["effective_tuning_name"] == "tabnetics_dlda"
    assert diagnostic["fit_outcome"] == "fitted_by_flaml"
    json.dumps(meta["flaml_candidate_registry_diagnostics"], allow_nan=False)


def test_native_only_flaml_does_not_poison_lazy_custom_spec_cache(monkeypatch):
    pytest.importorskip("flaml")
    X, y, counts = _toy_data(n_samples=80)

    class NativeOnlyAutoML:
        best_loss = 0.2
        best_estimator = "lrl2"
        best_config = {}
        model = LogisticRegression()

        def fit(self, *args, **kwargs):
            return None

    monkeypatch.setattr(backend_module, "_flaml_custom_specs_cache", None)
    with monkeypatch.context() as scoped:
        scoped.setitem(
            sys.modules,
            "flaml",
            types.SimpleNamespace(AutoML=NativeOnlyAutoML),
        )
        backend = FLAMLBackend(
            estimator_list=("lr",),
            min_n_for_automl=2,
            min_n_per_class_for_automl=2,
            min_n_per_class_for_cv=2,
        )
        backend.fit_and_select(
            X, y, seed=7, n_classes=2, class_counts=counts, cv_splits=2
        )

    assert backend_module._flaml_custom_specs_cache is None
    assert "dlda" in backend_module._get_flaml_custom_specs()


def test_flaml_unknown_no_tuning_and_unsupported_requests_fail_closed_before_import():
    X, y, counts = _toy_data()
    for requested, n_classes in (
        (("missing",), 2),
        (("tabm_official",), 2),
        (("bc_svm_linear",), 3),
    ):
        backend = FLAMLBackend(estimator_list=requested)
        with pytest.raises(ClassifierCandidateAdmissionError) as caught:
            backend.fit_and_select(
                X,
                y,
                seed=8,
                n_classes=n_classes,
                class_counts=counts,
                cv_splits=2,
            )
        assert caught.value.diagnostics[0]["admission_outcome"] == "rejected"
    assert caught.value.diagnostics[0]["resolved"]["availability"] == "unsupported"
    assert caught.value.diagnostics[0]["rejection_reason"] == "multiclass:unsupported"


def test_optuna_early_and_all_unknown_paths_preserve_registry_diagnostics():
    X, y, _ = _toy_data()
    mixed = OptunaBackend(candidate_names=("missing", "lr"))
    _, _, _, _, _, meta = mixed.fit_and_select(
        X,
        y,
        seed=9,
        n_classes=2,
        class_counts=np.asarray([1, 1]),
    )
    diagnostics = meta["optuna_candidate_registry_diagnostics"]
    assert [value["admission_outcome"] for value in diagnostics] == [
        "rejected",
        "admitted",
    ]
    json.dumps(diagnostics, allow_nan=False)

    unknown = OptunaBackend(candidate_names=("missing", "LR"))
    with pytest.raises(ClassifierCandidateAdmissionError):
        unknown.fit_and_select(
            X,
            y,
            seed=9,
            n_classes=2,
            class_counts=np.asarray([1, 1]),
        )


def test_optuna_normal_path_carries_delegated_sklearn_diagnostics(monkeypatch):
    X, y, counts = _toy_data(n_samples=50)

    class FakeStudy:
        trials = []
        best_trial = None

        def optimize(self, objective, **kwargs):
            return None

    fake_optuna = types.SimpleNamespace(
        samplers=types.SimpleNamespace(TPESampler=lambda seed: object()),
        create_study=lambda **kwargs: FakeStudy(),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    backend = OptunaBackend(
        candidate_names=("missing", "lr"),
        min_n_for_automl=2,
        min_n_per_class_for_automl=2,
        min_n_per_class_for_cv=2,
        n_trials=1,
    )

    _, _, _, _, _, meta = backend.fit_and_select(
        X, y, seed=10, n_classes=2, class_counts=counts, cv_splits=2
    )

    diagnostics = meta["optuna_candidate_registry_diagnostics"]
    assert [value["requested_name"] for value in diagnostics] == ["missing", "lr"]
    assert diagnostics[1]["build_outcome"] == "constructed"
    assert diagnostics[1]["evaluation_outcome"] == "evaluated"
    assert diagnostics[1]["fit_outcome"] == "fitted_on_full_training_data"


def test_optuna_alias_dispatch_preserves_outward_name_and_uses_canonical_builder():
    X, y, _ = _toy_data()
    backend = OptunaBackend(candidate_names=("shrinkage_lda",))
    models = backend._build_candidates(X_train=X, y_train=y, seed=10)
    tuned = backend._build_tuned_model(
        family="shrinkage_lda",
        params={"shrinkage": 0.4},
        fallback=models["shrinkage_lda"],
        seed=10,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_classes=2,
        y_train=y,
    )
    diagnostic = backend.get_candidate_registry_diagnostics()[0]
    assert list(models) == ["shrinkage_lda"]
    assert diagnostic["canonical_name"] == "dlda"
    assert diagnostic["static"]["builder_key"] == "dlda"
    assert tuned.steps[-1][0] == "lineardiscriminantanalysis"


@pytest.mark.parametrize(
    ("case", "n_samples", "expected_reason"),
    [
        ("missing_dependency", 220, "known_requested_candidates_not_constructible"),
        ("callback_absent", 80, "known_requested_candidates_not_constructible"),
        ("callback_failed", 80, "known_requested_candidates_not_constructible"),
        ("tree_gate", 80, "known_requested_candidates_not_constructible"),
    ],
)
def test_optuna_known_unconstructible_candidates_preserve_lr_fallback(
    monkeypatch, case, n_samples, expected_reason
):
    X, y, counts = _toy_data(n_samples=n_samples)

    class FakeStudy:
        trials = []
        best_trial = None

        def optimize(self, objective, **kwargs):
            return None

    monkeypatch.setitem(
        sys.modules,
        "optuna",
        types.SimpleNamespace(
            samplers=types.SimpleNamespace(TPESampler=lambda seed: object()),
            create_study=lambda **kwargs: FakeStudy(),
        ),
    )
    kwargs = {}
    if case == "missing_dependency":
        monkeypatch.setattr(backend_module, "LGBMClassifier", None)
        candidate_names = ("lgbm",)
    elif case == "callback_absent":
        candidate_names = ("tabpfn",)
    elif case == "callback_failed":
        candidate_names = ("tabpfn",)
        kwargs["build_tabpfn_model_fn"] = lambda seed: (
            None,
            "checkpoint unavailable",
        )
    else:
        candidate_names = ("rf",)

    backend = OptunaBackend(
        candidate_names=candidate_names,
        min_n_for_automl=2,
        min_n_per_class_for_automl=2,
        min_n_per_class_for_cv=2,
        n_trials=1,
        **kwargs,
    )
    _, name, _, _, _, meta = backend.fit_and_select(
        X, y, seed=10, n_classes=2, class_counts=counts, cv_splits=2
    )

    assert name == "optuna_lr"
    assert meta["optuna_registry_fallback_reason"] == expected_reason
    diagnostics = meta["optuna_candidate_registry_diagnostics"]
    assert diagnostics[0]["canonical_name"] in {"lgbm", "tabpfn", "rf"}
    assert diagnostics[0]["build_outcome"] in {
        "not_attempted",
        "failed",
    }


@pytest.mark.parametrize(
    ("candidate_names", "n_classes", "expected_reason"),
    [
        (("missing", "LR"), 2, "registry:unknown_classifier"),
        (("bc_svm_linear",), 3, "multiclass:unsupported"),
    ],
)
def test_optuna_unknown_or_semantically_incompatible_requests_still_fail_closed(
    candidate_names, n_classes, expected_reason
):
    X, y, counts = _toy_data()
    backend = OptunaBackend(candidate_names=candidate_names)
    with pytest.raises(ClassifierCandidateAdmissionError) as caught:
        backend.fit_and_select(
            X,
            y,
            seed=10,
            n_classes=n_classes,
            class_counts=counts,
            cv_splits=2,
        )
    assert any(
        value["rejection_reason"] == expected_reason
        for value in caught.value.diagnostics
    )


@pytest.mark.parametrize(
    ("candidate", "excluded"),
    [
        ("shrinkage_lda", ("dlda",)),
        ("shrinkage_lda", ("shrinkage_lda",)),
        ("dlda", ("shrinkage_lda",)),
        ("dlda", ("dlda",)),
    ],
)
def test_mnpo_global_alias_exclusion_removes_both_spellings(candidate, excluded):
    backend = MNPOClassifierBackend(
        candidate_names=(candidate, "lr"),
        exclude_candidate_names=excluded,
        use_per_family_flaml=False,
    )
    _, names, dropped = backend._filtered_candidates_by_regime(
        n_samples=40, n_features=8, n_classes=2
    )
    assert names == ["lr"]
    assert dropped == [candidate]
    assert backend.get_candidate_registry_diagnostics()[0]["rejection_reason"] == (
        "filter:global_exclusion:dlda"
    )


@pytest.mark.parametrize(
    ("candidate", "exclusion"),
    [("shrinkage_lda", "dlda"), ("dlda", "shrinkage_lda")],
)
def test_mnpo_regime_alias_exclusion_and_multiclass_support_are_canonical(
    candidate, exclusion
):
    excluded = MNPOClassifierBackend(
        candidate_names=(candidate, "lr"),
        regime_candidate_exclusions=(f"standard:{exclusion}",),
        use_per_family_flaml=False,
    )
    regime, names, _ = excluded._filtered_candidates_by_regime(
        n_samples=300, n_features=8, n_classes=2
    )
    assert regime == "standard"
    assert names == ["lr"]

    multiclass = MNPOClassifierBackend(
        candidate_names=("bc_svm_linear", "lr"), use_per_family_flaml=False
    )
    _, names, dropped = multiclass._filtered_candidates_by_regime(
        n_samples=40, n_features=8, n_classes=3
    )
    assert names == ["lr"]
    assert dropped == ["bc_svm_linear"]
    assert multiclass.get_candidate_registry_diagnostics()[0]["rejection_reason"] == (
        "multiclass:unsupported"
    )


@pytest.mark.parametrize(
    ("n_samples", "n_features", "expected_regime", "expected_names"),
    [
        (40, 8, "hdlss_extreme", ["lr"]),
        (100, 6_000, "hdlss_moderate", ["lr", "svm_rbf"]),
        (300, 8, "standard", ["lr", "svm_rbf"]),
    ],
)
def test_mnpo_empty_configuration_preserves_default_order_by_regime(
    n_samples, n_features, expected_regime, expected_names
):
    backend = MNPOClassifierBackend(
        candidate_names=(), use_per_family_flaml=False
    )
    regime, names, _ = backend._filtered_candidates_by_regime(
        n_samples=n_samples, n_features=n_features, n_classes=2
    )
    assert regime == expected_regime
    assert names == expected_names


def test_mnpo_per_family_flaml_alias_routes_through_registry_tuning_identity(
    monkeypatch,
):
    X, y, counts = _toy_data()
    fallback = LogisticRegression()
    backend = MNPOClassifierBackend(
        candidate_names=("shrinkage_lda",), use_per_family_flaml=True
    )
    captured = {}

    monkeypatch.setattr(
        backend_module,
        "_get_flaml_custom_specs",
        lambda: {"dlda": {"build": object()}},
    )

    def fake_custom_fit(**kwargs):
        captured.update(kwargs)
        return fallback, "flaml_dlda", 0.7, 0.1, 2, {}

    monkeypatch.setattr(backend, "_fit_custom_family_with_flaml", fake_custom_fit)
    _, name, _, _, _, meta = backend._fit_family_with_flaml(
        family_name="shrinkage_lda",
        fallback_model=fallback,
        X_train=X,
        y_train=y,
        seed=11,
        n_classes=2,
        class_counts=counts,
        cv_splits=2,
        scoring="balanced_accuracy",
        budget=2,
    )
    assert captured["family_name"] == "dlda"
    assert name == "flaml_shrinkage_lda"
    assert meta["mnpo_selected_family"] == "shrinkage_lda"
    assert meta["mnpo_registry_tuning_identity"] == "dlda"


def test_mnpo_all_unknown_raises_without_lr_injection_and_keeps_diagnostics():
    X, y, counts = _toy_data()
    backend = MNPOClassifierBackend(
        candidate_names=("missing", "LR"), use_per_family_flaml=False
    )
    with pytest.raises(ClassifierCandidateAdmissionError) as caught:
        backend.fit_and_select(
            X, y, seed=11, n_classes=2, class_counts=counts, cv_splits=2
        )
    assert [value["requested_name"] for value in caught.value.diagnostics] == [
        "missing",
        "LR",
    ]
    assert backend.get_candidates() == {}


def test_mnpo_normal_meta_merges_filter_build_and_oracle_diagnostics():
    X, y, counts = _toy_data()
    backend = MNPOClassifierBackend(
        candidate_names=("missing", "lr"),
        use_per_family_flaml=False,
        enable_bbc=False,
        enable_hoeffding_racing=False,
    )
    _, _, _, _, _, meta = backend.fit_and_select(
        X, y, seed=12, n_classes=2, class_counts=counts, cv_splits=2
    )
    diagnostics = meta["classification_candidate_registry_diagnostics"]
    assert [value["requested_name"] for value in diagnostics] == ["missing", "lr"]
    assert diagnostics[1]["build_outcome"] == "constructed"
    assert diagnostics[1]["evaluation_outcome"] == "evaluated"
    json.dumps(diagnostics, allow_nan=False)


def test_mnpo_optional_dependency_is_resolved_by_delegated_builder_with_warning(
    monkeypatch,
):
    X, y, counts = _toy_data(n_samples=220)
    warnings = []
    monkeypatch.setattr(backend_module, "LGBMClassifier", None)
    backend = MNPOClassifierBackend(
        candidate_names=("lr", "lgbm"),
        use_per_family_flaml=False,
        enable_bbc=False,
        enable_hoeffding_racing=False,
        warn_missing_backend_fn=lambda name, package, reason: warnings.append(
            (name, package, reason)
        ),
    )
    _, name, _, _, _, meta = backend.fit_and_select(
        X, y, seed=12, n_classes=2, class_counts=counts, cv_splits=2
    )

    assert name == "mnpo_lr"
    assert warnings == [("lgbm", "lightgbm", None)]
    assert "lgbm" not in meta["classification_regime_dropped_candidates"]
    diagnostics = meta["classification_candidate_registry_diagnostics"]
    lgbm = next(value for value in diagnostics if value["requested_name"] == "lgbm")
    assert lgbm["resolved"]["dependency_status"] == "unsupported"
    assert lgbm["admission_outcome"] == "rejected"
    assert lgbm["build_outcome"] == "failed"


def test_mnpo_regime_filtered_known_candidate_falls_back_but_unknown_fails_closed():
    X = np.zeros((60, 4_000), dtype=float)
    y = np.asarray([0, 1] * 30)
    known = MNPOClassifierBackend(
        candidate_names=("rf",), use_per_family_flaml=False
    )
    _, name, _, _, _, meta = known.fit_and_select(
        X,
        y,
        seed=12,
        n_classes=2,
        class_counts=np.asarray([1, 1]),
        cv_splits=2,
    )
    assert name == "lr"
    assert meta["classification_regime"] == "hdlss_moderate"
    assert meta["classification_registry_fallback_reason"] == (
        "known_candidates_filtered_by_policy"
    )
    diagnostic = meta["classification_candidate_registry_diagnostics"][0]
    assert diagnostic["canonical_name"] == "rf"
    assert diagnostic["rejection_reason"] == (
        "filter:regime_not_allowed:hdlss_moderate"
    )

    unknown = MNPOClassifierBackend(
        candidate_names=("missing",), use_per_family_flaml=False
    )
    with pytest.raises(ClassifierCandidateAdmissionError):
        unknown.fit_and_select(
            X,
            y,
            seed=12,
            n_classes=2,
            class_counts=np.asarray([1, 1]),
            cv_splits=2,
        )


def test_mnpo_alias_complexity_override_applies_to_outward_alias_candidate():
    X, y, counts = _toy_data()
    backend = MNPOClassifierBackend(
        candidate_names=("shrinkage_lda",),
        oracle_complexity_prior_overrides=("shrinkage_lda=0.25",),
        use_per_family_flaml=False,
        enable_bbc=False,
        enable_hoeffding_racing=False,
    )
    _, _, _, _, _, meta = backend.fit_and_select(
        X, y, seed=13, n_classes=2, class_counts=counts, cv_splits=2
    )
    assert meta["classification_complexity_prior_overrides"] == {"dlda": 0.25}
    assert meta["mnpo_candidate_stats"]["shrinkage_lda"][
        "complexity_score"
    ] == pytest.approx(0.25)


def test_tree_gate_uses_registry_flag_and_records_nonconstruction():
    X, y, _ = _toy_data(n_samples=80)
    backend = SklearnBackend(candidate_names=("rf", "sglnn"))
    models = backend._build_candidates(X_train=X, y_train=y, seed=14)
    diagnostics = backend.get_candidate_registry_diagnostics()
    assert "rf" not in models
    assert "sglnn" in models
    assert diagnostics[0]["build_reason"] == "tree_gate:dataset_or_config_disallowed"
    assert diagnostics[1]["static"]["tree_model"] is False

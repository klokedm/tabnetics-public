from __future__ import annotations

import json
import pickle
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tabnetics.classification.backends import (
    BiasCorrectedLinearSVM,
    ClassifierCandidateAdmissionError,
    NearestSubspaceClassifier,
    SklearnBackend,
    _HardVotingPredictAdapter,
    _LabelEncodedEstimator,
    _SoftVotingPredictAdapter,
    _ensemble_member_identity_records,
    _selection_identity_metadata,
)
from tabnetics.classification.conformance import (
    ProbabilityRequirement,
    check_probability_requirement,
    extract_probability_matrix,
    inspect_fitted_classifier,
    LabelValue,
)
from tabnetics.classification.registry import ProbabilityKind
from tabnetics.classification.registry import (
    DEFAULT_CLASSIFIER_REGISTRY,
    SupportLevel,
)
from tabnetics.feature_selection.conformal import compute_split_conformal_sets
from tabnetics.feature_selection.conformal import (
    _score_error_record,
    _strict_score_matrix_for_source,
    _typed_class_order,
)


def _binary(seed: int = 17):
    X, y = make_classification(
        n_samples=96,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        random_state=seed,
    )
    labels = np.where(y == 0, "control", "case")
    return np.asarray(X, dtype=float), labels


@pytest.mark.parametrize(
    ("kind", "matrix", "genuine", "calibrated"),
    [
        (ProbabilityKind.NATIVE, True, True, False),
        (ProbabilityKind.CALIBRATED, True, True, True),
        (ProbabilityKind.SCORE_DERIVED, True, False, False),
        (ProbabilityKind.NONE, False, False, False),
        (ProbabilityKind.HARD_LABEL_PROXY, False, False, False),
        (ProbabilityKind.UNKNOWN, False, False, False),
    ],
)
def test_probability_requirement_table(kind, matrix, genuine, calibrated):
    assert check_probability_requirement(kind, ProbabilityRequirement.MATRIX).admitted is matrix
    assert check_probability_requirement(kind, ProbabilityRequirement.GENUINE).admitted is genuine
    assert check_probability_requirement(kind, ProbabilityRequirement.CALIBRATED).admitted is calibrated


def test_pipeline_terminal_descriptor_is_json_safe_and_observed():
    X, y = _binary()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X, y)
    descriptor = inspect_fitted_classifier(
        model,
        canonical_name="lr",
        backend="sklearn",
        requested_name="lr",
        outward_name="lr",
        effective_model_name="lr",
        probe_X=X[:12],
        observe_pickle=True,
    )

    assert descriptor.fitted_probability_kind is ProbabilityKind.NATIVE
    assert descriptor.matrix_observation == "passed"
    assert descriptor.probability_matrix_available is True
    assert descriptor.argmax_observation == "passed"
    assert descriptor.clone_observation == "passed"
    assert descriptor.pickle_observation == "passed"
    assert [value.value for value in descriptor.class_order] == list(model.classes_)
    json.dumps(descriptor.to_dict(), allow_nan=False)
    clone(model)
    pickle.loads(pickle.dumps(model)).predict(X[:2])


def test_svc_probability_semantics_are_configuration_sensitive():
    X, y = _binary(19)
    disabled = make_pipeline(StandardScaler(), SVC(probability=False, random_state=19))
    enabled = make_pipeline(StandardScaler(), SVC(probability=True, random_state=19))
    disabled.fit(X, y)
    enabled.fit(X, y)

    none_descriptor = inspect_fitted_classifier(
        disabled,
        canonical_name="svm_rbf",
        backend="sklearn",
        config={"enable_svc_probability": False},
        probe_X=X[:10],
    )
    calibrated_descriptor = inspect_fitted_classifier(
        enabled,
        canonical_name="svm_rbf",
        backend="sklearn",
        config={"enable_svc_probability": True},
        probe_X=X[:10],
    )

    assert none_descriptor.fitted_probability_kind is ProbabilityKind.NONE
    assert none_descriptor.matrix_observation == "not_applicable"
    assert calibrated_descriptor.fitted_probability_kind is ProbabilityKind.CALIBRATED
    assert calibrated_descriptor.probability_source == "svc_internal_platt"
    assert calibrated_descriptor.argmax_contract == "not_promised"


class _HardProxyClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._label = self.classes_[0]
        return self

    def predict(self, X):
        return np.repeat(self._label, len(X))

    def predict_proba(self, X):
        out = np.zeros((len(X), len(self.classes_)), dtype=float)
        out[:, 0] = 1.0
        return out

    def tabnetics_probability_protocol(self):
        return {
            "mode": "declared",
            "probability_kind": "hard_label_proxy",
            "probability_source": "predict_one_hot",
        }


class _RevisionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, revision="sha256:test-revision", device="cpu"):
        self.revision = revision
        self.device = device

    def fit(self, X, y):
        self._model = LogisticRegression(max_iter=2000).fit(X, y)
        self.classes_ = np.asarray(self._model.classes_)
        self.checkpoint_metadata_ = {"checkpoint_sha256": self.revision}
        self.device_ = self.device
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class _LazyRevisionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        checkpoint="/configured/native-diakrino.pt",
        device="auto",
    ):
        self.checkpoint = checkpoint
        self.device = device

    def fit(self, X, y):
        self._model = LogisticRegression(max_iter=2000).fit(X, y)
        self.classes_ = np.asarray(self._model.classes_)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        self.native_diakrino_meta_ = {
            "native_diakrino_checkpoint": "/resolved/native-diakrino.pt",
            "native_diakrino_device": "cuda:3",
        }
        return self._model.predict_proba(X)


def test_hard_label_proxy_is_rejected_by_strict_matrix_consumer():
    X, y = _binary(23)
    model = _HardProxyClassifier().fit(X, y)
    descriptor = inspect_fitted_classifier(
        model,
        canonical_name="pls_da_classifier",
        backend="test",
        probe_X=X[:8],
    )
    result = extract_probability_matrix(
        model,
        X[:8],
        descriptor,
        requirement=ProbabilityRequirement.MATRIX,
        target_classes=model.classes_,
    )

    assert descriptor.fitted_probability_kind is ProbabilityKind.HARD_LABEL_PROXY
    assert result.available is False
    assert "hard_label_proxy" in result.reason


def test_revision_and_device_propagate_through_known_fitted_wrappers():
    from sklearn.calibration import CalibratedClassifierCV

    X, y = _binary(27)
    pipeline = make_pipeline(StandardScaler(), _RevisionClassifier()).fit(X, y)
    calibrated = CalibratedClassifierCV(
        estimator=make_pipeline(StandardScaler(), _RevisionClassifier()),
        method="sigmoid",
        cv=2,
    ).fit(X, y)
    delegate = _LabelEncodedEstimator(_RevisionClassifier()).fit(X, y)

    for model in (pipeline, calibrated, delegate):
        descriptor = inspect_fitted_classifier(
            model,
            canonical_name="tabentics_diakrino",
            backend="test",
            probe_X=X[:8],
        )
        assert descriptor.model_revision_source == (
            "checkpoint_metadata_.checkpoint_sha256"
        )
        assert descriptor.model_revision_value == "sha256:test-revision"
        assert descriptor.observed_device == "cpu"


def _diakrino_member_identity(name):
    return {
        "outward_name": name,
        "canonical_name": "tabentics_diakrino",
        "configured_probability_kind": "native",
        "registry_tuning_key": None,
        "executed_tuning_identity": None,
        "selected_flaml_family": None,
    }


def test_soft_vote_refreshes_lazy_member_revision_and_actual_device():
    X, y = _binary(28)
    model = VotingClassifier(
        estimators=[
            (
                "diakrino_a",
                _SoftVotingPredictAdapter(_LazyRevisionClassifier()),
            ),
            (
                "diakrino_b",
                _SoftVotingPredictAdapter(_LazyRevisionClassifier()),
            ),
        ],
        voting="soft",
    ).fit(X, y)
    identity = {
        "requested_name": "mnpo_ensemble",
        "outward_name": "mnpo_ensemble",
        "canonical_name": None,
        "composite_identity": "mnpo_ensemble",
        "members": [_diakrino_member_identity("diakrino_a"), _diakrino_member_identity("diakrino_b")],
    }
    descriptor = inspect_fitted_classifier(
        model,
        canonical_name=None,
        backend="mnpo_hybrid",
        selection_identity=identity,
        probe_X=X[:8],
    )
    assert descriptor.model_revision_source is None
    assert descriptor.model_revision_value is None
    assert descriptor.requested_device is None
    assert descriptor.requested_device_aggregation == "member_specific"
    assert descriptor.observed_device is None
    assert all(
        member.model_revision_source
        == "native_diakrino_meta_.native_diakrino_checkpoint"
        for member in descriptor.members
    )
    assert all(
        member.model_revision_value == "/resolved/native-diakrino.pt"
        for member in descriptor.members
    )
    assert all(member.observed_device == "cuda:3" for member in descriptor.members)
    assert all(member.requested_device == "auto" for member in descriptor.members)


def test_hard_vote_records_configured_member_checkpoint_without_inference():
    X, y = _binary(29)
    model = VotingClassifier(
        estimators=[
            (
                "diakrino_a",
                _HardVotingPredictAdapter(_LazyRevisionClassifier()),
            ),
            (
                "diakrino_b",
                _HardVotingPredictAdapter(_LazyRevisionClassifier()),
            ),
        ],
        voting="hard",
    ).fit(X, y)
    identity = {
        "requested_name": "mnpo_ensemble",
        "outward_name": "mnpo_ensemble",
        "canonical_name": None,
        "composite_identity": "mnpo_ensemble",
        "members": [_diakrino_member_identity("diakrino_a"), _diakrino_member_identity("diakrino_b")],
    }
    descriptor = inspect_fitted_classifier(
        model,
        canonical_name=None,
        backend="mnpo_hybrid",
        selection_identity=identity,
        probe_X=X[:8],
    )
    assert all(member.model_revision_source == "checkpoint" for member in descriptor.members)
    assert all(
        member.model_revision_value == "/configured/native-diakrino.pt"
        for member in descriptor.members
    )
    assert descriptor.requested_device is None
    assert descriptor.requested_device_aggregation == "member_specific"
    assert all(member.requested_device == "auto" for member in descriptor.members)
    assert all(member.observed_device is None for member in descriptor.members)


def test_static_registry_tuning_key_is_not_an_executed_tuning_claim():
    X, y = _binary(33)
    model = LogisticRegression(max_iter=2000).fit(X, y)
    direct_meta = _selection_identity_metadata(
        backend="sklearn",
        requested_name="lr",
        outward_name="lr",
        effective_model_name="lr",
        canonical_name="lr",
    )
    identity = direct_meta["classification_selected_identity"]
    assert identity["registry_tuning_key"] == "lrl2"
    assert identity["executed_tuning_identity"] is None
    assert identity["tuning_identity"] is None
    assert identity["selected_flaml_family"] is None

    descriptor = inspect_fitted_classifier(
        model,
        canonical_name="lr",
        backend="sklearn",
        selection_identity=identity,
        probe_X=X[:8],
    )
    assert descriptor.registry_tuning_key == "lrl2"
    assert descriptor.executed_tuning_identity is None
    assert descriptor.tuning_identity is None
    assert descriptor.selected_flaml_family is None


def test_direct_vote_vs_executed_flaml_member_tuning_provenance():
    direct = _ensemble_member_identity_records(
        ("lr",),
        tuned_names={"lr": "lr"},
    )[0]
    assert direct["registry_tuning_key"] == "lrl2"
    assert direct["executed_tuning_identity"] is None
    assert direct["selected_flaml_family"] is None
    assert direct["tuned_family"] is None
    assert direct["tuned_model"] is None

    tuned = _ensemble_member_identity_records(
        ("lr",),
        tuned_names={"lr": "flaml_lr"},
        tuning_meta_by_family={
            "lr": {
                "classification_backend_used": "flaml",
                "flaml_best_estimator": "lrl2",
            }
        },
    )[0]
    assert tuned["registry_tuning_key"] == "lrl2"
    assert tuned["executed_tuning_identity"] == "lrl2"
    assert tuned["selected_flaml_family"] == "lr"
    assert tuned["tuned_family"] == "lr"
    assert tuned["tuned_model"] == "flaml_lr"


def test_soft_vote_composite_keeps_member_identity_and_observes_matrices():
    X, y = _binary(29)
    members = [
        {
            "outward_name": "lr",
            "canonical_name": "lr",
            "tuned_family": None,
            "tuned_model": None,
            "registry_tuning_key": "lrl2",
            "executed_tuning_identity": None,
            "selected_flaml_family": None,
            "duplicate_position": 0,
            "weight": 0.7,
            "configured_probability_kind": "native",
        },
        {
            "outward_name": "nb",
            "canonical_name": "nb",
            "tuned_family": None,
            "tuned_model": None,
            "registry_tuning_key": "nb",
            "executed_tuning_identity": None,
            "selected_flaml_family": None,
            "duplicate_position": 0,
            "weight": 0.3,
            "configured_probability_kind": "native",
        },
    ]
    model = VotingClassifier(
        estimators=[
            ("lr", _SoftVotingPredictAdapter(LogisticRegression(max_iter=2000))),
            ("nb", _SoftVotingPredictAdapter(GaussianNB())),
        ],
        voting="soft",
        weights=[0.7, 0.3],
    ).fit(X, y)
    identity = {
        "requested_name": "mnpo_ensemble",
        "outward_name": "mnpo_ensemble",
        "canonical_name": None,
        "registry_anchor_name": None,
        "effective_model_name": "mnpo_ensemble_lr__nb",
        "composite_identity": "mnpo_ensemble",
        "members": members,
    }
    descriptor = inspect_fitted_classifier(
        model,
        canonical_name=None,
        backend="mnpo_hybrid",
        selection_identity=identity,
        probe_X=X[:10],
    )

    assert descriptor.canonical_name is None
    assert descriptor.composite_identity == "mnpo_ensemble"
    assert descriptor.fitted_probability_kind is ProbabilityKind.SCORE_DERIVED
    assert descriptor.matrix_observation == "passed"
    assert [member.canonical_name for member in descriptor.members] == ["lr", "nb"]
    assert all(member.reason == "observed_matrix_passed" for member in descriptor.members)
    assert [member.registry_tuning_key for member in descriptor.members] == [
        "lrl2",
        "nb",
    ]
    assert all(member.executed_tuning_identity is None for member in descriptor.members)
    assert all(member.tuned_family is None for member in descriptor.members)
    assert all(member.tuned_model is None for member in descriptor.members)


def test_soft_vote_member_failure_reason_does_not_persist_third_party_message(
    monkeypatch,
):
    X, y = _binary(30)
    model = VotingClassifier(
        estimators=[
            ("lr", _SoftVotingPredictAdapter(LogisticRegression(max_iter=2000))),
            ("nb", _SoftVotingPredictAdapter(GaussianNB())),
        ],
        voting="soft",
    ).fit(X, y)

    def _raise(_):
        raise ValueError("/private/path/checkpoint-specific-message")

    monkeypatch.setattr(model.estimators_[0], "predict_proba", _raise)
    identity = {
        "requested_name": "mnpo_ensemble",
        "outward_name": "mnpo_ensemble",
        "canonical_name": None,
        "composite_identity": "mnpo_ensemble",
        "members": [
            {
                "outward_name": "lr",
                "canonical_name": "lr",
                "configured_probability_kind": "native",
            },
            {
                "outward_name": "nb",
                "canonical_name": "nb",
                "configured_probability_kind": "native",
            },
        ],
    }
    descriptor = inspect_fitted_classifier(
        model,
        canonical_name=None,
        backend="mnpo_hybrid",
        selection_identity=identity,
        probe_X=X[:8],
    )
    assert descriptor.members[0].reason == "member_probability:ValueError"
    assert "/private/path" not in json.dumps(descriptor.to_dict())


def test_soft_vote_member_rejects_tiny_negative_probability(monkeypatch):
    X, y = _binary(32)
    model = VotingClassifier(
        estimators=[
            ("lr", _SoftVotingPredictAdapter(LogisticRegression(max_iter=2000))),
            ("nb", _SoftVotingPredictAdapter(GaussianNB())),
        ],
        voting="soft",
    ).fit(X, y)

    def _tiny_negative(X_eval):
        return np.tile(
            np.asarray([[-1e-15, 1.0 + 1e-15]], dtype=float),
            (len(X_eval), 1),
        )

    monkeypatch.setattr(model.estimators_[0], "predict_proba", _tiny_negative)
    identity = {
        "requested_name": "mnpo_ensemble",
        "outward_name": "mnpo_ensemble",
        "canonical_name": None,
        "composite_identity": "mnpo_ensemble",
        "members": [
            {
                "outward_name": "lr",
                "canonical_name": "lr",
                "configured_probability_kind": "native",
            },
            {
                "outward_name": "nb",
                "canonical_name": "nb",
                "configured_probability_kind": "native",
            },
        ],
    }
    descriptor = inspect_fitted_classifier(
        model,
        canonical_name=None,
        backend="mnpo_hybrid",
        selection_identity=identity,
        probe_X=X[:8],
    )
    assert descriptor.members[0].reason == "negative_probability"
    assert descriptor.fitted_probability_kind is ProbabilityKind.UNKNOWN


def test_hard_vote_string_labels_uses_weighted_vote_fraction_conformity():
    X, y = _binary(31)
    model = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000)),
            ("nb", GaussianNB()),
        ],
        voting="hard",
        weights=[3.0, 1.0],
    )
    out = compute_split_conformal_sets(
        model=model,
        X_train=X,
        y_train=y,
        X_eval=X[:12],
        y_eval=y[:12],
        min_calibration=12,
        seed=31,
    )

    assert out["classifier_conformal_applied"] is True
    assert out["classifier_conformal_conformity_kind"] == "hard_vote_fraction"
    assert out["classifier_conformal_score_source"] == "hard_vote_fraction"
    assert out["classifier_conformal_probability_claim"] is False
    assert out["classifier_conformal_source_consistent"] is True


def test_near_subspace_rounding_collapse_preserves_argmax_agreement():
    rng = np.random.default_rng(37)
    X = rng.normal(size=(45, 2))
    y = np.asarray(["a", "b", "c"] * 15)
    model = NearestSubspaceClassifier().fit(X, y)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    assert np.array_equal(pred, model.classes_[np.argmax(proba, axis=1)])


def test_bc_svm_zero_tie_selects_first_class_for_predict_and_probability(monkeypatch):
    X, y = _binary(41)
    model = BiasCorrectedLinearSVM(random_state=41).fit(X, y)
    monkeypatch.setattr(
        model,
        "decision_function",
        lambda X_eval: np.zeros(len(X_eval), dtype=float),
    )
    assert np.all(model.predict(X[:5]) == model.classes_[0])
    assert np.all(np.argmax(model.predict_proba(X[:5]), axis=1) == 0)


def test_bc_svm_multiclass_singleton_request_fails_before_cv():
    X, y_int = make_classification(
        n_samples=90,
        n_features=10,
        n_informative=7,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=43,
    )
    backend = SklearnBackend(candidate_names=("bc_svm_linear",))
    with pytest.raises(ClassifierCandidateAdmissionError) as caught:
        backend.fit_and_select(
            X,
            y_int,
            seed=43,
            n_classes=3,
            class_counts=np.bincount(y_int),
        )
    assert caught.value.diagnostics[0]["rejection_reason"] == "multiclass:unsupported"


class _MatrixStub:
    def __init__(self, classes, matrix, predictions=None):
        self.classes_ = np.asarray(classes)
        self._matrix = np.asarray(matrix)
        self._predictions = predictions

    def predict_proba(self, X):
        return np.asarray(self._matrix)

    def predict(self, X):
        if self._predictions is not None:
            return np.asarray(self._predictions)
        return self.classes_[np.argmax(np.asarray(self._matrix), axis=1)]


def _native_descriptor_for_matrix_tests():
    X, y = _binary(47)
    model = LogisticRegression(max_iter=2000).fit(X, y)
    descriptor = inspect_fitted_classifier(
        model,
        canonical_name="lr",
        backend="test",
        probe_X=X[:4],
    )
    return X[:4], model, descriptor


@pytest.mark.parametrize(
    ("matrix", "reason"),
    [
        (np.asarray([0.2, 0.8, 0.4, 0.6]), "matrix:invalid_rank"),
        (np.full((4, 3), 1.0 / 3.0), "matrix:column_count_mismatch"),
        (
            np.asarray(
                [[np.nan, 0.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
            ),
            "matrix:nonfinite",
        ),
        (
            np.asarray(
                [[-0.1, 1.1], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
            ),
            "matrix:negative",
        ),
        (
            np.asarray(
                [
                    [-1e-15, 1.0 + 1e-15],
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.5],
                ]
            ),
            "matrix:negative",
        ),
        (np.full((4, 2), 0.4), "matrix:not_simplex"),
    ],
)
def test_strict_probability_extraction_rejects_malformed_matrices(matrix, reason):
    X, model, descriptor = _native_descriptor_for_matrix_tests()
    stub = _MatrixStub(model.classes_, matrix)
    result = extract_probability_matrix(stub, X, descriptor)
    assert result.available is False
    assert result.reason == reason


def test_probability_extraction_rejects_stale_descriptor_class_order():
    X, model, descriptor = _native_descriptor_for_matrix_tests()
    stale = replace(descriptor, class_order=tuple(reversed(descriptor.class_order)))
    result = extract_probability_matrix(model, X, stale)
    assert result.available is False
    assert result.reason == "descriptor:model_class_order_mismatch"


def test_probability_extraction_reorders_exact_typed_target_classes():
    X, model, descriptor = _native_descriptor_for_matrix_tests()
    source = np.asarray(model.predict_proba(X), dtype=float)
    target = np.asarray(model.classes_)[::-1]
    result = extract_probability_matrix(
        model,
        X,
        descriptor,
        target_classes=target,
    )
    assert result.available is True
    assert result.aligned is True
    assert np.allclose(result.matrix, source[:, ::-1])
    assert [record.value for record in result.class_order] == list(target)


def test_probability_extraction_preserves_mixed_typed_target_labels():
    X, _, descriptor = _native_descriptor_for_matrix_tests()
    classes = np.empty(2, dtype=object)
    classes[:] = [1, "1"]
    matrix = np.tile(np.asarray([[0.75, 0.25]], dtype=float), (len(X), 1))
    model = _MatrixStub(classes, matrix)
    records = (
        LabelValue(
            value=1,
            python_type="builtins.int",
            numpy_dtype=str(np.asarray(1).dtype),
        ),
        LabelValue(
            value="1",
            python_type="builtins.str",
            numpy_dtype=str(np.asarray("1").dtype),
        ),
    )
    mixed_descriptor = replace(
        descriptor,
        class_order=records,
        probability_column_order=records,
    )

    reordered = extract_probability_matrix(
        model,
        X,
        mixed_descriptor,
        target_classes=["1", 1],
    )
    assert reordered.available is True
    assert np.allclose(reordered.matrix, matrix[:, ::-1])
    assert [record.python_type for record in reordered.class_order] == [
        "builtins.str",
        "builtins.int",
    ]

    duplicate = extract_probability_matrix(
        model,
        X,
        mixed_descriptor,
        target_classes=[1, 1],
    )
    assert duplicate.available is False
    assert duplicate.reason == "target_classes:duplicate"


@pytest.mark.parametrize(
    ("target", "reason"),
    [
        (["case", "unknown"], "classes:alignment_failed"),
        (["case", "case"], "target_classes:duplicate"),
    ],
)
def test_probability_extraction_rejects_unknown_or_duplicate_targets(target, reason):
    X, model, descriptor = _native_descriptor_for_matrix_tests()
    result = extract_probability_matrix(
        model,
        X,
        descriptor,
        target_classes=target,
    )
    assert result.available is False
    assert result.reason == reason


def test_required_argmax_failure_invalidates_probability_admission():
    X, model, _ = _native_descriptor_for_matrix_tests()
    matrix = np.asarray(model.predict_proba(X), dtype=float)
    wrong = model.classes_[1 - np.argmax(matrix, axis=1)]
    stub = _MatrixStub(model.classes_, matrix, predictions=wrong)
    descriptor = inspect_fitted_classifier(
        stub,
        canonical_name="lr",
        backend="test",
        probe_X=X,
    )
    assert descriptor.argmax_contract == "required"
    assert descriptor.argmax_observation == "failed"
    assert descriptor.matrix_observation == "failed"
    assert descriptor.matrix_reason == "argmax:agreement_mismatch"
    assert descriptor.fitted_probability_kind is ProbabilityKind.UNKNOWN
    assert descriptor.probability_matrix_available is False


def test_conformal_typed_class_order_is_json_safe_for_object_labels():
    records = _typed_class_order(
        np.asarray([Decimal("1.25"), Decimal("2.50")], dtype=object)
    )
    assert [record["value"] for record in records] == ["1.25", "2.50"]
    json.dumps(records, allow_nan=False)


def test_score_error_record_whitelists_only_canonical_reasons():
    assert _score_error_record(ValueError("invalid_probability_shape")) == {
        "reason": "invalid_probability_shape",
        "exception_type": "ValueError",
    }
    assert _score_error_record(ValueError("/private/model/message")) == {
        "reason": "score_source_failed",
        "exception_type": "ValueError",
    }


def test_strict_conformal_probability_source_rejects_tiny_negative_value():
    X, model, _ = _native_descriptor_for_matrix_tests()
    matrix = np.asarray(model.predict_proba(X), dtype=float)
    matrix[0] = np.asarray([-1e-15, 1.0 + 1e-15])
    stub = _MatrixStub(model.classes_, matrix)
    with pytest.raises(ValueError, match="^negative_probability$"):
        _strict_score_matrix_for_source(
            stub,
            X,
            np.asarray(model.classes_),
            source="predict_proba",
        )


def _matrix_data(n_classes: int, *, tree_gate: bool, seed: int):
    n_samples = 210 if tree_gate else 60
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=seed,
    )
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _matrix_xgb_builder(y, seed):
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None, "optional_dependency_unavailable"
    n_classes = int(np.unique(y).size)
    params = {
        "n_estimators": 8,
        "max_depth": 2,
        "learning_rate": 0.1,
        "n_jobs": 1,
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": int(seed),
        "objective": "binary:logistic" if n_classes == 2 else "multi:softprob",
        "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
    }
    if n_classes > 2:
        params["num_class"] = n_classes
    return XGBClassifier(**params), None


def _configure_matrix_model(model, canonical_name):
    requested = {}
    if canonical_name in {"rf", "extra_tree", "lgbm", "catboost"}:
        requested["n_estimators"] = 8
    if canonical_name == "lgbm":
        requested["verbosity"] = -1
    if canonical_name == "catboost":
        requested["depth"] = 2
    if canonical_name == "sglnn":
        requested.update(
            {
                "sparsegrouplassonnclassifier__max_iter": 12,
                "sparsegrouplassonnclassifier__cv_lambda": False,
                "sparsegrouplassonnclassifier__n_hidden": 6,
            }
        )
    if canonical_name == "tabm":
        requested.update({"max_iter": 12, "n_heads": 2, "n_hidden": 8})
    if canonical_name == "realmlp":
        requested.update({"max_iter": 12, "depth": 1, "n_hidden": 8})
    if canonical_name in {"tabm_official", "realmlp_td"}:
        requested["n_epochs"] = 4
    if canonical_name == "cpda":
        requested.update({"max_rounds": 2, "internal_cv": 2})
    if canonical_name == "spls_da_classifier":
        requested.update({"max_components": 2, "n_cv_folds": 2})
    available = model.get_params(deep=True)
    model.set_params(**{key: value for key, value in requested.items() if key in available})
    return model


def _construct_matrix_model(canonical_name, X, y, seed):
    backend = SklearnBackend(
        candidate_names=(canonical_name,),
        allow_tree_models=True,
        n_jobs=1,
        build_xgb_model_fn=_matrix_xgb_builder,
    )
    models = backend._build_candidates(X_train=X, y_train=y, seed=seed)
    diagnostics = backend.get_candidate_registry_diagnostics()
    diagnostic = next(
        value
        for value in diagnostics
        if value.get("canonical_name") == canonical_name
    )
    model = models.get(canonical_name)
    if model is not None:
        model = _configure_matrix_model(model, canonical_name)
    return model, diagnostic


def _outcome(values):
    observed = [str(value) for value in values if value is not None]
    if not observed:
        return "skipped"
    for value in observed:
        if value.startswith("failed"):
            return value
    if all(value == "not_applicable" for value in observed):
        return "not_applicable"
    if any(value == "passed" for value in observed):
        return "passed"
    return observed[0]


def _matrix_skip_evidence(diagnostic):
    resolved = dict(diagnostic.get("resolved") or {})
    unsupported = any(
        str(resolved.get(field) or "") == "unsupported"
        for field in ("dependency_status", "builder_status", "gpu_status")
    )
    if unsupported:
        return True, "unavailable"
    build_reason = str(
        diagnostic.get("build_reason")
        or diagnostic.get("rejection_reason")
        or ""
    ).strip()
    canonical_missing_optional = (
        build_reason
        in {
            "optional_dependency_unavailable",
            "optional dependency unavailable",
            "pytabkit not installed",
        }
        or (
            build_reason.startswith("dependency:")
            and build_reason.endswith(":unavailable")
        )
    )
    if (
        str(resolved.get("dependency_status") or "") == "conditional"
        and canonical_missing_optional
    ):
        return True, "unavailable"
    if str(resolved.get("gpu_status") or "") == "conditional":
        return True, "conditional"
    return False, None


def _matrix_availability(constructed, skip_permitted, skip_state):
    if constructed or not skip_permitted:
        return "available"
    return str(skip_state or "conditional")


def _matrix_execution(fit_values, *, constructed, skip_permitted):
    if any(str(value).startswith("failed") for value in fit_values):
        return "fit-failed"
    if any(value == "passed" for value in fit_values):
        return "fit-passed"
    if not constructed and not skip_permitted:
        return "fit-failed"
    return "constructed" if constructed else "not-constructed"


def _run_matrix_scenario(canonical_name, n_classes, seed):
    spec = DEFAULT_CLASSIFIER_REGISTRY.get(canonical_name)
    tree_gate = bool(spec.tree_model)
    X, y = _matrix_data(n_classes, tree_gate=tree_gate, seed=seed)
    model, diagnostic = _construct_matrix_model(canonical_name, X, y, seed)
    if model is None:
        reason = str(
            diagnostic.get("build_reason")
            or diagnostic.get("rejection_reason")
            or "builder:returned_no_model"
        )
        skip_permitted, skip_state = _matrix_skip_evidence(diagnostic)
        return {
            "fit": "skipped",
            "probability": "skipped",
            "clone": "skipped",
            "pickle": "skipped",
            "row_permutation": "skipped",
            "label_permutation": "skipped",
            "repeat_seed": "skipped",
            "reason": reason,
            "construction": str(diagnostic.get("build_outcome") or "not_attempted"),
            "class_order": [],
            "dependency_status": str(
                (diagnostic.get("resolved") or {}).get(
                    "dependency_status", "unknown"
                )
            ),
            "skip_permitted": skip_permitted,
            "skip_availability": skip_state,
        }

    probe = X[:12]
    try:
        model.fit(X, y)
    except Exception as exc:
        reason = f"fit:{type(exc).__name__}"
        return {
            "fit": f"failed:{type(exc).__name__}",
            "probability": "skipped",
            "clone": "skipped",
            "pickle": "skipped",
            "row_permutation": "skipped",
            "label_permutation": "skipped",
            "repeat_seed": "skipped",
            "reason": reason,
            "construction": "constructed",
            "class_order": [],
            "dependency_status": str(
                (diagnostic.get("resolved") or {}).get(
                    "dependency_status", "unknown"
                )
            ),
            "skip_permitted": False,
            "skip_availability": None,
        }

    descriptor = inspect_fitted_classifier(
        model,
        canonical_name=canonical_name,
        backend="conformance_matrix",
        requested_name=canonical_name,
        outward_name=canonical_name,
        effective_model_name=canonical_name,
        config={"enable_svc_probability": False},
        probe_X=probe,
        observe_pickle=True,
    )
    probability = (
        "passed"
        if descriptor.matrix_observation == "passed"
        else "not_applicable"
        if descriptor.matrix_observation == "not_applicable"
        else f"failed:{descriptor.matrix_reason}"
    )
    base_prediction = np.asarray(model.predict(probe)).ravel()

    try:
        restored = pickle.loads(pickle.dumps(model))
        pickle_outcome = (
            "passed"
            if np.array_equal(np.asarray(restored.predict(probe)).ravel(), base_prediction)
            else "failed:prediction_mismatch"
        )
    except Exception as exc:
        pickle_outcome = f"failed:{type(exc).__name__}"

    try:
        repeated, _ = _construct_matrix_model(canonical_name, X, y, seed)
        assert repeated is not None
        repeated.fit(X, y)
        repeat_outcome = (
            "passed"
            if np.array_equal(np.asarray(repeated.predict(probe)).ravel(), base_prediction)
            else "failed:prediction_mismatch"
        )
    except Exception as exc:
        repeat_outcome = f"failed:{type(exc).__name__}"

    order = np.random.default_rng(seed + 1000).permutation(len(y))
    try:
        row_model, _ = _construct_matrix_model(canonical_name, X, y, seed)
        assert row_model is not None
        row_model.fit(X[order], y[order])
        row_outcome = (
            "passed"
            if np.array_equal(np.asarray(row_model.predict(probe)).ravel(), base_prediction)
            else "failed:prediction_mismatch"
        )
    except Exception as exc:
        row_outcome = f"failed:{type(exc).__name__}"

    label_map = np.roll(np.arange(n_classes, dtype=int), 1)
    inverse_map = np.argsort(label_map)
    y_permuted = label_map[y]
    try:
        label_model, _ = _construct_matrix_model(
            canonical_name, X, y_permuted, seed
        )
        assert label_model is not None
        label_model.fit(X, y_permuted)
        renamed = np.asarray(label_model.predict(probe), dtype=int).ravel()
        label_outcome = (
            "passed"
            if np.array_equal(inverse_map[renamed], base_prediction)
            else "failed:prediction_mismatch"
        )
    except Exception as exc:
        label_outcome = f"failed:{type(exc).__name__}"

    return {
        "fit": "passed",
        "probability": probability,
        "clone": descriptor.clone_observation,
        "pickle": pickle_outcome,
        "row_permutation": row_outcome,
        "label_permutation": label_outcome,
        "repeat_seed": repeat_outcome,
        "reason": "ok",
        "construction": "constructed",
        "class_order": [record.to_dict() for record in descriptor.class_order],
        "dependency_status": descriptor.dependency_status,
        "skip_permitted": False,
        "skip_availability": None,
    }


def build_canonical_classifier_conformance_matrix(seed=53):
    rows = []
    for canonical_name in DEFAULT_CLASSIFIER_REGISTRY.names():
        spec = DEFAULT_CLASSIFIER_REGISTRY.get(canonical_name)
        binary = _run_matrix_scenario(canonical_name, 2, seed)
        if spec.multiclass is SupportLevel.UNSUPPORTED:
            multiclass = {
                "fit": "skipped:multiclass:unsupported",
                "probability": "skipped",
                "clone": "skipped",
                "pickle": "skipped",
                "row_permutation": "skipped",
                "label_permutation": "skipped",
                "repeat_seed": "skipped",
                "reason": "multiclass:unsupported",
                "construction": "not_attempted",
                "class_order": [],
                "dependency_status": "supported",
                "skip_permitted": True,
                "skip_availability": "conditional",
            }
        else:
            multiclass = _run_matrix_scenario(canonical_name, 3, seed + 1)
        constructed = binary["construction"] == "constructed"
        availability = _matrix_availability(
            constructed,
            binary["skip_permitted"],
            binary["skip_availability"],
        )
        fit_values = [binary["fit"], multiclass["fit"]]
        execution = _matrix_execution(
            fit_values,
            constructed=constructed,
            skip_permitted=bool(binary["skip_permitted"]),
        )
        row_permutation_contract = (
            "not_promised:numerical_svd_order"
            if canonical_name == "near_subspace"
            else "not_promised:conditional_backend_determinism"
            if spec.deterministic is not SupportLevel.SUPPORTED
            else "required"
        )
        label_permutation_contract = (
            "not_promised:label_indexed_random_output_initialization"
            if canonical_name in {"sglnn", "tabm", "realmlp"}
            else "not_promised:conditional_backend_determinism"
            if spec.deterministic is not SupportLevel.SUPPORTED
            else "required"
        )
        row = {
            "canonical_name": canonical_name,
            "availability": availability,
            "construction": _outcome(
                [binary["construction"], multiclass["construction"]]
            ),
            "execution": execution,
            "probability": _outcome(
                [binary["probability"], multiclass["probability"]]
            ),
            "clone": _outcome([binary["clone"], multiclass["clone"]]),
            "pickle": _outcome([binary["pickle"], multiclass["pickle"]]),
            "row_permutation": _outcome(
                [binary["row_permutation"], multiclass["row_permutation"]]
            ),
            "label_permutation": _outcome(
                [binary["label_permutation"], multiclass["label_permutation"]]
            ),
            "repeat_seed": _outcome(
                [binary["repeat_seed"], multiclass["repeat_seed"]]
            ),
            "row_permutation_contract": row_permutation_contract,
            "label_permutation_contract": label_permutation_contract,
            "repeat_seed_contract": (
                "required"
                if spec.deterministic is SupportLevel.SUPPORTED
                else "observe_only:conditional_determinism"
            ),
            "binary": binary["fit"],
            "multiclass": multiclass["fit"],
            "class_order": {
                "binary": binary["class_order"],
                "multiclass": multiclass["class_order"],
            },
            "resource_class": spec.resource_class.value,
            "declared_dependencies": list(spec.dependencies),
            "dependency_status": _outcome(
                [binary["dependency_status"], multiclass["dependency_status"]]
            ),
            "determinism_declared": spec.deterministic.value,
            "serialization_declared": spec.serialization.value,
            "observations": {"binary": binary, "multiclass": multiclass},
        }
        observed_failures = [
            f"{field}:{row[field]}"
            for field in (
                "probability",
                "clone",
                "pickle",
                "row_permutation",
                "label_permutation",
                "repeat_seed",
                "binary",
                "multiclass",
            )
            if str(row[field]).startswith("failed")
        ]
        if observed_failures:
            row["reason"] = "|".join(observed_failures)
        elif binary["reason"] == "ok" and multiclass["reason"] in {
            "ok",
            "multiclass:unsupported",
        }:
            row["reason"] = "ok"
        else:
            row["reason"] = (
                f"binary:{binary['reason']}|multiclass:{multiclass['reason']}"
            )
        rows.append(row)
    return rows


def test_ordered_one_row_per_canonical_family_conformance_matrix():
    rows = build_canonical_classifier_conformance_matrix()
    expected = DEFAULT_CLASSIFIER_REGISTRY.names()
    assert tuple(row["canonical_name"] for row in rows) == expected
    assert len(rows) == len(set(expected)) == 42
    required = {
        "availability",
        "construction",
        "execution",
        "probability",
        "clone",
        "pickle",
        "row_permutation",
        "label_permutation",
        "repeat_seed",
        "binary",
        "multiclass",
        "reason",
        "class_order",
        "resource_class",
        "declared_dependencies",
        "dependency_status",
        "determinism_declared",
        "serialization_declared",
        "row_permutation_contract",
        "label_permutation_contract",
        "repeat_seed_contract",
    }
    assert all(required <= set(row) for row in rows)

    for row in rows:
        spec = DEFAULT_CLASSIFIER_REGISTRY.get(row["canonical_name"])
        assert row["resource_class"] == spec.resource_class.value
        assert row["declared_dependencies"] == list(spec.dependencies)
        assert row["determinism_declared"] == spec.deterministic.value
        assert row["serialization_declared"] == spec.serialization.value
        assert row["dependency_status"] in {
            "supported",
            "unsupported",
            "conditional",
            "unknown",
        }
        if row["availability"] != "available":
            assert row["reason"] != "ok"
            assert row["class_order"] == {"binary": [], "multiclass": []}
            continue
        assert row["execution"] == "fit-passed", row
        assert row["binary"] == "passed", row
        if spec.multiclass is SupportLevel.SUPPORTED:
            assert row["multiclass"] == "passed", row
        if spec.probability_kind in {
            ProbabilityKind.NATIVE,
            ProbabilityKind.CALIBRATED,
            ProbabilityKind.SCORE_DERIVED,
        }:
            assert row["probability"] == "passed", row
        else:
            assert row["probability"] == "not_applicable", row
        assert row["clone"] == "passed", row
        if spec.serialization is SupportLevel.SUPPORTED:
            assert row["pickle"] == "passed", row
        if spec.deterministic is SupportLevel.SUPPORTED:
            assert row["repeat_seed"] == "passed", row
        if row["row_permutation_contract"] == "required":
            assert row["row_permutation"] == "passed", row
        else:
            assert row["row_permutation"] in {
                "passed",
                "failed:prediction_mismatch",
            }, row
        if row["label_permutation_contract"] == "required":
            assert row["label_permutation"] == "passed", row
        else:
            assert row["label_permutation"] in {
                "passed",
                "failed:prediction_mismatch",
            }, row
        assert len(row["class_order"]["binary"]) == 2
        if spec.multiclass is SupportLevel.SUPPORTED:
            assert len(row["class_order"]["multiclass"]) == 3

    json.dumps(rows, allow_nan=False, sort_keys=True)


def test_matrix_core_or_installed_optional_construction_failure_is_blocking():
    supported_diagnostic = {
        "resolved": {
            "dependency_status": "supported",
            "builder_status": "supported",
            "gpu_status": "supported",
        },
        "build_outcome": "failed",
        "build_reason": "builder:returned_no_model",
    }
    skip_permitted, skip_state = _matrix_skip_evidence(supported_diagnostic)
    assert skip_permitted is False
    assert skip_state is None
    availability = _matrix_availability(False, skip_permitted, skip_state)
    execution = _matrix_execution(
        ["skipped", "skipped"],
        constructed=False,
        skip_permitted=skip_permitted,
    )
    assert availability == "available"
    assert execution == "fit-failed"


def test_matrix_skip_requires_diagnostic_missing_resource_evidence():
    missing_builder = {
        "resolved": {
            "dependency_status": "supported",
            "builder_status": "unsupported",
            "gpu_status": "supported",
        }
    }
    assert _matrix_skip_evidence(missing_builder) == (True, "unavailable")

    unresolved_dependency = {
        "resolved": {
            "dependency_status": "conditional",
            "builder_status": "supported",
            "gpu_status": "supported",
        }
    }
    assert _matrix_skip_evidence(unresolved_dependency) == (False, None)

    unresolved_but_proven_missing = {
        **unresolved_dependency,
        "build_reason": "optional_dependency_unavailable",
    }
    assert _matrix_skip_evidence(unresolved_but_proven_missing) == (
        True,
        "unavailable",
    )

    generic_optional_failure = {
        **unresolved_dependency,
        "build_reason": "builder:returned_no_model",
    }
    assert _matrix_skip_evidence(generic_optional_failure) == (False, None)


def test_alias_identity_has_one_explicit_canonical_mapping_record():
    X, y = _matrix_data(2, tree_gate=False, seed=59)
    backend = SklearnBackend(candidate_names=("shrinkage_lda", "dlda"))
    models = backend._build_candidates(X_train=X, y_train=y, seed=59)
    diagnostics = backend.get_candidate_registry_diagnostics()
    alias = diagnostics[0]
    duplicate = diagnostics[1]
    record = {
        "requested_name": alias["requested_name"],
        "canonical_name": alias["canonical_name"],
        "alias_identity": "explicit",
        "duplicate_reason": duplicate["rejection_reason"],
    }
    assert tuple(models) == ("shrinkage_lda",)
    assert record == {
        "requested_name": "shrinkage_lda",
        "canonical_name": "dlda",
        "alias_identity": "explicit",
        "duplicate_reason": "registry:duplicate_canonical_request:dlda",
    }
    json.dumps(record, allow_nan=False, sort_keys=True)

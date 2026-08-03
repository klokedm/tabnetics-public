from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from tabnetics.classification.conformance import inspect_fitted_classifier
from tabnetics.classification.decision_policy import (
    DecisionPolicy,
    DecisionPolicyRequest,
    evaluate_binary_decision_policy,
    fit_binary_decision_policy,
)


def _request(*, score_source: str = "genuine_probability") -> DecisionPolicyRequest:
    X, y = make_classification(
        n_samples=360, n_features=12, n_informative=8, n_redundant=2,
        weights=[0.78, 0.22], class_sep=1.3, random_state=7,
    )
    model = LogisticRegression(max_iter=4000, random_state=7).fit(X[:160], y[:160])
    descriptor = inspect_fitted_classifier(
        model, canonical_name="lr", backend="sklearn", probe_X=X[:16]
    )
    scores = model.predict_proba(X[160:260])[:, 1]
    return DecisionPolicyRequest(
        scores=scores, y_true=y[160:260], class_order=(0, 1), positive_class=1,
        score_class=1,
        score_source=score_source, descriptor=descriptor,
        calibration_rows_sha256="a" * 64, split_fingerprint="b" * 64,
        context_fingerprint="c" * 64, objective="balanced_accuracy",
        threshold_grid=(0.30, 0.50, 0.70), abstention_grid=(0.0, 0.08),
        cost_matrix=(0.0, 1.0, 4.0, 0.0), abstain_costs=(0.5, 2.0),
        minimum_accepted_per_class=2,
    )


def test_binary_policy_is_deterministic_and_outer_labels_do_not_fit_it() -> None:
    request = _request()
    first = fit_binary_decision_policy(request)
    second = fit_binary_decision_policy(request)
    assert first.to_dict() == second.to_dict()
    assert first.score_source == "genuine_probability"
    assert first.selection_curve

    outer_scores = np.linspace(0.01, 0.99, 30)
    probe = evaluate_binary_decision_policy(first, outer_scores)
    clean_labels = np.asarray(probe.predicted, dtype=int)
    assert set(clean_labels.tolist()) == {0, 1}
    clean = evaluate_binary_decision_policy(first, outer_scores, clean_labels)
    mutated = evaluate_binary_decision_policy(first, outer_scores, 1 - clean_labels)
    assert clean.predicted == mutated.predicted
    assert clean.abstained == mutated.abstained
    assert clean.metrics["selective_risk"] != mutated.metrics["selective_risk"]


def test_binary_policy_keeps_abstention_separate_and_round_trips() -> None:
    policy = fit_binary_decision_policy(_request())
    loaded = DecisionPolicy.from_dict(policy.to_dict())
    assert loaded.to_dict() == policy.to_dict()
    result = evaluate_binary_decision_policy(loaded, [0.49, 0.90], [0, 1])
    assert len(result.predicted) == 2
    assert len(result.abstained) == 2
    assert all(isinstance(value, bool) for value in result.abstained)
    assert 0.0 <= result.metrics["coverage"] <= 1.0


def test_genuine_probability_rejects_score_derived_descriptor() -> None:
    request = _request()
    X = np.arange(400, dtype=float).reshape(100, 4)
    y = np.asarray([0, 1] * 50)
    model = SVC(kernel="linear", probability=False, random_state=3).fit(X, y)
    descriptor = inspect_fitted_classifier(
        model, canonical_name="svm_linear", backend="sklearn", probe_X=X[:16]
    )
    request = replace(request, descriptor=descriptor, scores=model.decision_function(X), y_true=y)
    with pytest.raises(ValueError, match="genuine_probability_required"):
        fit_binary_decision_policy(request)


def test_decision_score_mode_is_explicit_and_cost_validation_fails_closed() -> None:
    request = replace(_request(), score_source="decision_score", scores=np.linspace(-2.0, 2.0, 100))
    policy = fit_binary_decision_policy(request)
    assert policy.score_source == "decision_score"
    with pytest.raises(ValueError, match="invalid_decision_cost_matrix"):
        fit_binary_decision_policy(replace(request, cost_matrix=(0.0, -1.0, 1.0, 0.0)))


def test_exact_metric_ties_use_predeclared_lower_numeric_threshold() -> None:
    request = _request()
    scores = np.asarray([0.01, 0.99] * 50)
    labels = np.asarray([0, 1] * 50)
    policy = fit_binary_decision_policy(
        replace(
            request, scores=scores, y_true=labels,
            threshold_grid=(0.25, 0.75), abstention_grid=(0.0,),
        )
    )
    assert policy.threshold == 0.25


def test_probability_score_column_and_descriptor_order_are_bound() -> None:
    request = _request()
    with pytest.raises(ValueError, match="score_class_must_match"):
        fit_binary_decision_policy(replace(request, score_class=0))
    reversed_descriptor = replace(
        request.descriptor,
        probability_column_order=tuple(reversed(request.descriptor.probability_column_order)),
    )
    with pytest.raises(ValueError, match="probability_class_order_mismatch"):
        fit_binary_decision_policy(replace(request, descriptor=reversed_descriptor))


def test_curve_keeps_ineligible_points_and_costs_include_abstentions() -> None:
    request = _request()
    policy = fit_binary_decision_policy(
        replace(
            request, objective="expected_cost", threshold_grid=(0.25,),
            abstention_grid=(0.0, 0.49),
        )
    )
    assert len(policy.selection_curve) == 2
    assert any(not bool(row["eligible"]) for row in policy.selection_curve)
    assert all(np.isfinite(float(row["expected_cost"])) for row in policy.selection_curve)
    with pytest.raises(ValueError, match="expected_cost_requires_abstain_costs"):
        fit_binary_decision_policy(
            replace(request, objective="expected_cost", abstain_costs=None)
        )


def test_one_class_accepted_evaluation_reports_unavailable_class_metrics() -> None:
    policy = fit_binary_decision_policy(_request())
    result = evaluate_binary_decision_policy(policy, np.full(12, 0.99), np.ones(12, dtype=int))
    assert result.metrics["class_metrics_available"] == 0.0
    assert np.isnan(result.metrics["balanced_accuracy"])
    assert np.isnan(result.metrics["worst_class_recall"])


def test_tampered_serialized_policy_is_rejected() -> None:
    policy = fit_binary_decision_policy(_request())
    payload = policy.to_dict()
    payload["descriptor_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="invalid_policy_provenance"):
        DecisionPolicy.from_dict(payload)
    payload = policy.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="invalid_decision_policy_fields"):
        DecisionPolicy.from_dict(payload)


def test_abstention_costs_follow_negative_positive_contract_with_reversed_class_order() -> None:
    request = replace(
        _request(), score_source="decision_score", class_order=(1, 0),
        positive_class=1, score_class=1,
        scores=np.linspace(-2.0, 2.0, 100),
        y_true=np.asarray([1] * 80 + [0] * 20), objective="expected_cost",
        threshold_grid=(0.0,), abstention_grid=(0.0, 10.0),
        abstain_costs=(5.0, 1.0),
    )
    policy = fit_binary_decision_policy(request)
    all_abstained = next(row for row in policy.selection_curve if row["abstention_margin"] == 10.0)
    assert all_abstained["expected_cost"] == pytest.approx(1.8)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda payload: payload["selection_curve"][0].update({"raw_scores": [0.1, 0.9]}), "invalid_policy_curve"),
        (lambda payload: payload["selection_curve"][0].update({"eligible": 1}), "invalid_policy_curve"),
        (lambda payload: payload["selection_curve"][0].update({"coverage": float("inf")}), "invalid_policy_metrics"),
        (lambda payload: payload["selected_metrics"].update({"raw_labels": [0, 1]}), "invalid_policy_metrics"),
    ],
)
def test_serialized_policy_rejects_nonaggregate_or_malformed_metrics(mutate, error: str) -> None:
    payload = fit_binary_decision_policy(_request()).to_dict()
    mutate(payload)
    with pytest.raises(ValueError, match=error):
        DecisionPolicy.from_dict(payload)

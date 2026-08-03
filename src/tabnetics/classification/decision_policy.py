"""Fold-local, evaluator-only binary decision policies.

Policies in this module are deliberately detached from the production pipeline.
They may be fitted only from already-isolated calibration/OOF scores; callers
must not use outer-test labels to choose their operating point.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from tabnetics.classification.conformance import (
    FittedClassifierDescriptor,
    ProbabilityRequirement,
    check_probability_requirement,
)


POLICY_SCHEMA_VERSION = "tabnetics_binary_decision_policy_v1"
_SHA256 = frozenset("0123456789abcdef")
_OBJECTIVES = frozenset({"balanced_accuracy", "macro_f1", "worst_class_recall", "expected_cost"})
_SOURCES = frozenset({"genuine_probability", "decision_score"})
_METRIC_FIELDS = frozenset({
    "coverage", "accepted_count", "accepted_class_0_count", "accepted_class_1_count",
    "class_metrics_available", "selective_risk", "balanced_accuracy", "macro_f1",
    "worst_class_recall", "expected_cost",
})
_CURVE_FIELDS = frozenset({
    "threshold", "abstention_margin", "eligible", "ineligibility_code", *_METRIC_FIELDS,
})
_INELIGIBLE_CODE = "insufficient_accepted_class_coverage"


def _typed_key(value: object) -> tuple[str, str, str]:
    scalar = value.item() if isinstance(value, np.generic) else value
    return type(scalar).__module__, type(scalar).__qualname__, repr(scalar)


def _label_record(value: object) -> dict[str, object]:
    scalar = value.item() if isinstance(value, np.generic) else value
    if not isinstance(scalar, (str, int, float, bool)) or isinstance(scalar, float) and not np.isfinite(scalar):
        raise ValueError("unsupported_policy_label")
    return {"value": scalar, "python_type": f"{type(scalar).__module__}.{type(scalar).__qualname__}"}


def _decode_label(record: object) -> object:
    if not isinstance(record, dict) or "value" not in record or "python_type" not in record:
        raise ValueError("invalid_policy_label_record")
    value, kind = record["value"], str(record["python_type"])
    expected = f"{type(value).__module__}.{type(value).__qualname__}"
    if expected != kind:
        raise ValueError("policy_label_type_mismatch")
    return value


def _digest(record: object) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _sha(value: str) -> bool:
    return len(value) == 64 and set(value) <= _SHA256


@dataclass(frozen=True, slots=True)
class DecisionPolicyRequest:
    """Provenance-bound OOF calibration scores for one binary policy."""

    scores: Sequence[float]
    y_true: Sequence[Any]
    class_order: tuple[Any, Any]
    positive_class: Any
    score_class: Any
    score_source: str
    descriptor: FittedClassifierDescriptor
    calibration_rows_sha256: str
    split_fingerprint: str
    context_fingerprint: str
    objective: str = "balanced_accuracy"
    threshold_grid: tuple[float, ...] = (0.25, 0.50, 0.75)
    abstention_grid: tuple[float, ...] = (0.0,)
    cost_matrix: tuple[float, float, float, float] | None = None  # tn, fp, fn, tp
    abstain_costs: tuple[float, float] | None = None  # negative, positive
    minimum_accepted_per_class: int = 1


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """JSON-safe selected point policy; no calibration rows or scores are retained."""

    class_order: tuple[Any, Any]
    positive_class: Any
    score_class: Any
    score_source: str
    objective: str
    threshold: float
    abstention_margin: float
    cost_matrix: tuple[float, float, float, float] | None
    abstain_costs: tuple[float, float] | None
    calibration_rows_sha256: str
    split_fingerprint: str
    context_fingerprint: str
    descriptor_sha256: str
    selection_curve: tuple[dict[str, Any], ...]
    selected_metrics: dict[str, float]
    schema_version: str = POLICY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "class_order": [_label_record(value) for value in self.class_order],
            "positive_class": _label_record(self.positive_class),
            "score_class": _label_record(self.score_class),
            "score_source": self.score_source,
            "objective": self.objective,
            "threshold": self.threshold,
            "abstention_margin": self.abstention_margin,
            "cost_matrix": None if self.cost_matrix is None else list(self.cost_matrix),
            "abstain_costs": None if self.abstain_costs is None else list(self.abstain_costs),
            "calibration_rows_sha256": self.calibration_rows_sha256,
            "split_fingerprint": self.split_fingerprint,
            "context_fingerprint": self.context_fingerprint,
            "descriptor_sha256": self.descriptor_sha256,
            "selection_curve": [dict(row) for row in self.selection_curve],
            "selected_metrics": dict(self.selected_metrics),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DecisionPolicy":
        required = {
            "schema_version", "class_order", "positive_class", "score_class",
            "score_source", "objective", "threshold", "abstention_margin",
            "cost_matrix", "abstain_costs", "calibration_rows_sha256",
            "split_fingerprint", "context_fingerprint", "descriptor_sha256",
            "selection_curve", "selected_metrics",
        }
        if set(payload) != required:
            raise ValueError("invalid_decision_policy_fields")
        if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported_decision_policy_schema")
        raw_order = payload.get("class_order")
        if not isinstance(raw_order, list) or len(raw_order) != 2:
            raise ValueError("invalid_policy_class_order")
        cost = payload.get("cost_matrix")
        abstain_costs = payload.get("abstain_costs")
        if cost is not None and (not isinstance(cost, list) or len(cost) != 4):
            raise ValueError("invalid_policy_cost_matrix")
        if abstain_costs is not None and (
            not isinstance(abstain_costs, list) or len(abstain_costs) != 2
        ):
            raise ValueError("invalid_policy_abstain_costs")
        curve = payload.get("selection_curve")
        metrics = payload.get("selected_metrics")
        if not isinstance(curve, list) or not curve or not isinstance(metrics, dict):
            raise ValueError("invalid_policy_metrics")
        policy = cls(
            class_order=tuple(_decode_label(value) for value in raw_order),  # type: ignore[arg-type]
            positive_class=_decode_label(payload.get("positive_class")),
            score_class=_decode_label(payload.get("score_class")),
            score_source=str(payload.get("score_source") or ""),
            objective=str(payload.get("objective") or ""),
            threshold=float(payload.get("threshold")),
            abstention_margin=float(payload.get("abstention_margin")),
            cost_matrix=None if cost is None else tuple(float(value) for value in cost),
            abstain_costs=(
                None if abstain_costs is None
                else tuple(float(value) for value in abstain_costs)
            ),
            calibration_rows_sha256=str(payload.get("calibration_rows_sha256") or ""),
            split_fingerprint=str(payload.get("split_fingerprint") or ""),
            context_fingerprint=str(payload.get("context_fingerprint") or ""),
            descriptor_sha256=str(payload.get("descriptor_sha256") or ""),
            selection_curve=tuple(dict(row) for row in curve),
            selected_metrics=dict(metrics),
        )
        _validate_policy(policy)
        return policy


@dataclass(frozen=True, slots=True)
class DecisionPolicyEvaluation:
    """Separate point prediction and abstention state with aggregate metrics."""

    predicted: tuple[Any, ...]
    abstained: tuple[bool, ...]
    metrics: dict[str, float]


def _validate_request(request: DecisionPolicyRequest) -> tuple[np.ndarray, np.ndarray, tuple[Any, Any], int, tuple[float, ...], tuple[float, ...]]:
    source = str(request.score_source).strip().lower()
    if source not in _SOURCES:
        raise ValueError("unknown_decision_score_source")
    objective = str(request.objective).strip().lower()
    if objective not in _OBJECTIVES:
        raise ValueError("unknown_decision_objective")
    if not all(_sha(str(value)) for value in (request.calibration_rows_sha256, request.split_fingerprint, request.context_fingerprint)):
        raise ValueError("invalid_decision_provenance")
    classes = tuple(request.class_order)
    if len(classes) != 2 or len({_typed_key(value) for value in classes}) != 2:
        raise ValueError("decision_policy_requires_two_typed_classes")
    positive_key = _typed_key(request.positive_class)
    if positive_key not in {_typed_key(value) for value in classes}:
        raise ValueError("positive_class_not_in_class_order")
    if _typed_key(request.score_class) != positive_key:
        raise ValueError("score_class_must_match_positive_class")
    positive_index = next(index for index, value in enumerate(classes) if _typed_key(value) == positive_key)
    scores = np.asarray(request.scores, dtype=float).ravel()
    labels = np.asarray(request.y_true, dtype=object).ravel()
    if not scores.size or scores.size != labels.size or not np.all(np.isfinite(scores)):
        raise ValueError("invalid_calibration_scores")
    label_keys = {_typed_key(value) for value in labels.tolist()}
    if label_keys != {_typed_key(value) for value in classes}:
        raise ValueError("calibration_labels_not_binary_class_order")
    if source == "genuine_probability":
        admission = check_probability_requirement(request.descriptor.fitted_probability_kind, ProbabilityRequirement.GENUINE)
        if not admission.admitted or not request.descriptor.probability_matrix_available:
            raise ValueError("genuine_probability_required")
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("invalid_probability_score")
        descriptor_classes = tuple(
            (_typed_key(value.value), str(value.python_type))
            for value in request.descriptor.class_order
        )
        descriptor_columns = tuple(
            (_typed_key(value.value), str(value.python_type))
            for value in request.descriptor.probability_column_order
        )
        requested_classes = tuple(
            (_typed_key(value), f"{type(value).__module__}.{type(value).__qualname__}")
            for value in classes
        )
        if descriptor_classes != requested_classes or descriptor_columns != requested_classes:
            raise ValueError("probability_class_order_mismatch")
    thresholds = tuple(sorted({float(value) for value in request.threshold_grid}))
    margins = tuple(sorted({float(value) for value in request.abstention_grid}))
    if not thresholds or not margins or any(not np.isfinite(value) for value in (*thresholds, *margins)):
        raise ValueError("invalid_decision_grid")
    if any(value < 0.0 for value in margins):
        raise ValueError("negative_abstention_margin")
    if source == "genuine_probability" and (thresholds[0] < 0.0 or thresholds[-1] > 1.0 or margins[-1] > 0.5):
        raise ValueError("probability_decision_grid_out_of_range")
    if request.cost_matrix is not None and (len(request.cost_matrix) != 4 or any(not np.isfinite(float(value)) or float(value) < 0.0 for value in request.cost_matrix)):
        raise ValueError("invalid_decision_cost_matrix")
    if request.abstain_costs is not None and (
        len(request.abstain_costs) != 2
        or any(not np.isfinite(float(value)) or float(value) < 0.0 for value in request.abstain_costs)
    ):
        raise ValueError("invalid_decision_abstain_costs")
    if objective == "expected_cost" and (
        request.cost_matrix is None or request.abstain_costs is None
    ):
        raise ValueError("expected_cost_requires_abstain_costs")
    return scores, labels, classes, positive_index, thresholds, margins


def _validate_policy(policy: DecisionPolicy) -> None:
    if policy.schema_version != POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported_decision_policy_schema")
    if policy.score_source not in _SOURCES or policy.objective not in _OBJECTIVES:
        raise ValueError("invalid_policy_source_or_objective")
    if len(policy.class_order) != 2 or len({_typed_key(value) for value in policy.class_order}) != 2:
        raise ValueError("invalid_policy_class_order")
    if _typed_key(policy.positive_class) not in {_typed_key(value) for value in policy.class_order}:
        raise ValueError("invalid_policy_positive_class")
    if _typed_key(policy.score_class) != _typed_key(policy.positive_class):
        raise ValueError("invalid_policy_score_class")
    if not all(_sha(str(value)) for value in (
        policy.calibration_rows_sha256, policy.split_fingerprint,
        policy.context_fingerprint, policy.descriptor_sha256,
    )):
        raise ValueError("invalid_policy_provenance")
    if not np.isfinite(policy.threshold) or not np.isfinite(policy.abstention_margin) or policy.abstention_margin < 0.0:
        raise ValueError("invalid_policy_threshold")
    if policy.score_source == "genuine_probability" and (
        not 0.0 <= policy.threshold <= 1.0 or policy.abstention_margin > 0.5
    ):
        raise ValueError("invalid_policy_probability_threshold")
    if policy.cost_matrix is not None and (
        len(policy.cost_matrix) != 4 or any(not np.isfinite(value) or value < 0.0 for value in policy.cost_matrix)
    ):
        raise ValueError("invalid_policy_cost_matrix")
    if policy.abstain_costs is not None and (
        len(policy.abstain_costs) != 2 or any(not np.isfinite(value) or value < 0.0 for value in policy.abstain_costs)
    ):
        raise ValueError("invalid_policy_abstain_costs")
    if policy.objective == "expected_cost" and (
        policy.cost_matrix is None or policy.abstain_costs is None
    ):
        raise ValueError("expected_cost_requires_abstain_costs")
    if not isinstance(policy.selection_curve, tuple) or not policy.selection_curve:
        raise ValueError("invalid_policy_curve")
    for row in policy.selection_curve:
        _validate_curve_row(row)
    _validate_metrics(policy.selected_metrics)


def _validate_metrics(metrics: object) -> None:
    """Enforce the aggregate-only policy artifact boundary."""
    if not isinstance(metrics, dict) or set(metrics) != _METRIC_FIELDS:
        raise ValueError("invalid_policy_metrics")
    values: dict[str, float] = {}
    for key in _METRIC_FIELDS:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError("invalid_policy_metrics")
        numeric = float(value)
        if np.isinf(numeric):
            raise ValueError("invalid_policy_metrics")
        values[key] = numeric
    if not 0.0 <= values["coverage"] <= 1.0:
        raise ValueError("invalid_policy_metrics")
    if values["class_metrics_available"] not in {0.0, 1.0}:
        raise ValueError("invalid_policy_metrics")
    for key in ("accepted_count", "accepted_class_0_count", "accepted_class_1_count"):
        if not np.isfinite(values[key]) or values[key] < 0.0 or not values[key].is_integer():
            raise ValueError("invalid_policy_metrics")
    if not np.isfinite(values["selective_risk"]) and not np.isnan(values["selective_risk"]):
        raise ValueError("invalid_policy_metrics")
    if np.isfinite(values["selective_risk"]) and not 0.0 <= values["selective_risk"] <= 1.0:
        raise ValueError("invalid_policy_metrics")
    if values["class_metrics_available"] == 1.0:
        for key in ("balanced_accuracy", "macro_f1", "worst_class_recall"):
            if not np.isfinite(values[key]) or not 0.0 <= values[key] <= 1.0:
                raise ValueError("invalid_policy_metrics")
    else:
        for key in ("balanced_accuracy", "macro_f1", "worst_class_recall"):
            if not np.isnan(values[key]):
                raise ValueError("invalid_policy_metrics")
    if not np.isnan(values["expected_cost"]) and (
        not np.isfinite(values["expected_cost"]) or values["expected_cost"] < 0.0
    ):
        raise ValueError("invalid_policy_metrics")


def _validate_curve_row(row: object) -> None:
    if not isinstance(row, dict) or set(row) != _CURVE_FIELDS:
        raise ValueError("invalid_policy_curve")
    if type(row["eligible"]) is not bool or not isinstance(row["ineligibility_code"], str):
        raise ValueError("invalid_policy_curve")
    if row["eligible"] and row["ineligibility_code"]:
        raise ValueError("invalid_policy_curve")
    if not row["eligible"] and row["ineligibility_code"] != _INELIGIBLE_CODE:
        raise ValueError("invalid_policy_curve")
    for key in ("threshold", "abstention_margin"):
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(float(value)):
            raise ValueError("invalid_policy_curve")
    _validate_metrics({key: row[key] for key in _METRIC_FIELDS})


def apply_binary_decision_policy(policy: DecisionPolicy, scores: Sequence[float]) -> tuple[tuple[Any, ...], tuple[bool, ...]]:
    _validate_policy(policy)
    values = np.asarray(scores, dtype=float).ravel()
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("invalid_evaluation_scores")
    if policy.score_source == "genuine_probability" and (np.any(values < 0.0) or np.any(values > 1.0)):
        raise ValueError("invalid_probability_score")
    positive_idx = next(index for index, value in enumerate(policy.class_order) if _typed_key(value) == _typed_key(policy.positive_class))
    negative = policy.class_order[1 - positive_idx]
    predicted = tuple(policy.positive_class if score >= policy.threshold else negative for score in values.tolist())
    abstained = tuple(abs(float(score) - policy.threshold) <= policy.abstention_margin for score in values.tolist())
    return predicted, abstained


def _metrics(y_true: np.ndarray, predicted: Sequence[Any], abstained: np.ndarray, classes: tuple[Any, Any], positive_class: Any, cost: tuple[float, float, float, float] | None, abstain_costs: tuple[float, float] | None, *, minimum_accepted_per_class: int) -> dict[str, float]:
    accepted = ~abstained
    coverage = float(np.mean(accepted)) if accepted.size else float("nan")
    result = {"coverage": coverage, "accepted_count": float(np.sum(accepted)), "accepted_class_0_count": 0.0, "accepted_class_1_count": 0.0, "class_metrics_available": 0.0, "selective_risk": float("nan"), "balanced_accuracy": float("nan"), "macro_f1": float("nan"), "worst_class_recall": float("nan"), "expected_cost": float("nan")}
    class_positions = {_typed_key(value): index for index, value in enumerate(classes)}
    if cost is not None and abstain_costs is not None:
        positive = positive_class
        negative = next(value for value in classes if _typed_key(value) != _typed_key(positive))
        tn, fp, fn, tp = cost
        all_predicted = np.asarray(predicted, dtype=object)
        charges = [
            abstain_costs[1] if abstained[index] and _typed_key(t) == _typed_key(positive)
            else abstain_costs[0] if abstained[index]
            else tp if _typed_key(t) == _typed_key(positive) and _typed_key(p) == _typed_key(positive)
            else tn if _typed_key(t) == _typed_key(negative) and _typed_key(p) == _typed_key(negative)
            else fp if _typed_key(t) == _typed_key(negative) else fn
            for index, (t, p) in enumerate(zip(y_true.tolist(), all_predicted.tolist()))
        ]
        result["expected_cost"] = float(np.mean(charges))
    if not np.any(accepted):
        return result
    truth = y_true[accepted]
    pred = np.asarray(predicted, dtype=object)[accepted]
    truth_codes = np.asarray([class_positions[_typed_key(value)] for value in truth.tolist()], dtype=int)
    pred_codes = np.asarray([class_positions[_typed_key(value)] for value in pred.tolist()], dtype=int)
    accepted_counts = np.bincount(truth_codes, minlength=2)
    result["accepted_class_0_count"] = float(accepted_counts[0])
    result["accepted_class_1_count"] = float(accepted_counts[1])
    if np.all(accepted_counts >= int(minimum_accepted_per_class)):
        result["class_metrics_available"] = 1.0
        result["balanced_accuracy"] = float(balanced_accuracy_score(truth_codes, pred_codes))
        result["macro_f1"] = float(f1_score(truth_codes, pred_codes, average="macro", zero_division=0))
        recalls = recall_score(truth_codes, pred_codes, labels=[0, 1], average=None, zero_division=0)
        result["worst_class_recall"] = float(np.min(recalls))
    result["selective_risk"] = float(np.mean(pred_codes != truth_codes))
    return result


def fit_binary_decision_policy(request: DecisionPolicyRequest) -> DecisionPolicy:
    """Select a deterministic threshold/abstention policy from calibration only."""
    scores, labels, classes, _positive_index, thresholds, margins = _validate_request(request)
    objective = str(request.objective).strip().lower()
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[tuple[float, ...], float, float, dict[str, float]]] = []
    for threshold in thresholds:
        for margin in margins:
            negative = next(
                value for value in classes
                if _typed_key(value) != _typed_key(request.positive_class)
            )
            predicted = tuple(
                request.positive_class if score >= threshold else negative
                for score in scores.tolist()
            )
            abstained = tuple(
                abs(float(score) - threshold) <= margin for score in scores.tolist()
            )
            metrics = _metrics(
                labels, predicted, np.asarray(abstained, dtype=bool), classes,
                request.positive_class, request.cost_matrix, request.abstain_costs,
                minimum_accepted_per_class=int(max(1, request.minimum_accepted_per_class)),
            )
            eligible = bool(metrics["class_metrics_available"])
            row = {
                "threshold": float(threshold), "abstention_margin": float(margin),
                "eligible": eligible,
                "ineligibility_code": "" if eligible else _INELIGIBLE_CODE,
                **metrics,
            }
            rows.append(row)
            if not eligible:
                continue
            secondary_cost = float(metrics["expected_cost"])
            if not np.isfinite(secondary_cost):
                secondary_cost = 0.0
            primary = -metrics[objective] if objective != "expected_cost" else metrics[objective]
            rank = (primary, -metrics["worst_class_recall"], secondary_cost, -metrics["coverage"], float(threshold), float(margin))
            candidates.append((rank, float(threshold), float(margin), metrics))
    if not candidates:
        raise ValueError("no_valid_decision_policy_candidate")
    _rank, threshold, margin, metrics = min(candidates, key=lambda item: item[0])
    return DecisionPolicy(
        class_order=classes, positive_class=request.positive_class,
        score_class=request.score_class,
        score_source=str(request.score_source).strip().lower(), objective=objective,
        threshold=threshold, abstention_margin=margin, cost_matrix=request.cost_matrix,
        abstain_costs=request.abstain_costs,
        calibration_rows_sha256=request.calibration_rows_sha256, split_fingerprint=request.split_fingerprint,
        context_fingerprint=request.context_fingerprint, descriptor_sha256=_digest(request.descriptor.to_dict()),
        selection_curve=tuple(rows), selected_metrics={str(key): float(value) for key, value in metrics.items()},
    )


def evaluate_binary_decision_policy(policy: DecisionPolicy, scores: Sequence[float], y_true: Sequence[Any] | None = None) -> DecisionPolicyEvaluation:
    """Apply a frozen policy; labels affect metrics only, never policy selection."""
    predicted, abstained = apply_binary_decision_policy(policy, scores)
    if y_true is None:
        return DecisionPolicyEvaluation(predicted, abstained, {"coverage": float(np.mean(~np.asarray(abstained, dtype=bool)))})
    labels = np.asarray(y_true, dtype=object).ravel()
    if labels.size != len(predicted):
        raise ValueError("evaluation_label_length_mismatch")
    if {_typed_key(value) for value in labels.tolist()} - {_typed_key(value) for value in policy.class_order}:
        raise ValueError("evaluation_labels_not_policy_classes")
    return DecisionPolicyEvaluation(
        predicted,
        abstained,
        _metrics(
            labels, predicted, np.asarray(abstained, dtype=bool), policy.class_order,
            policy.positive_class, policy.cost_matrix, policy.abstain_costs,
            minimum_accepted_per_class=1,
        ),
    )


__all__ = [
    "DecisionPolicy", "DecisionPolicyEvaluation", "DecisionPolicyRequest",
    "POLICY_SCHEMA_VERSION", "apply_binary_decision_policy", "evaluate_binary_decision_policy",
    "fit_binary_decision_policy",
]

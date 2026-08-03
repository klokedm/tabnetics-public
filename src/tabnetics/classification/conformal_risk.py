"""Default-off, provenance-bound conformal risk-control primitives.

This module deliberately does not alter the existing split/APS/RAPS helpers.
It accepts only fitted, genuinely probabilistic classifiers and returns
aggregate/hash-only diagnostics so callers cannot turn score proxies or raw
participant identities into an unsupported conditional guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

from tabnetics.classification.conformance import (
    FittedClassifierDescriptor,
    ProbabilityRequirement,
    extract_probability_matrix,
)


_SHA256_LENGTH = 64
_GROUP_AXES = frozenset({"", "groups", "patient_ids", "site_ids", "batch_ids"})
_IID_POLICIES = frozenset({"iid", "stratified"})


def _sha256(record: object) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _typed_key(value: object) -> tuple[str, str, str]:
    scalar = value.item() if isinstance(value, np.generic) else value
    return type(scalar).__module__, type(scalar).__qualname__, repr(scalar)


def _typed_equal(left: object, right: object) -> bool:
    return _typed_key(left) == _typed_key(right)


def _valid_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(char in "0123456789abcdef" for char in value)


def _model_classes(model: Any) -> tuple[Any, ...]:
    classes = getattr(model, "classes_", None)
    if classes is None and getattr(model, "steps", None):
        classes = getattr(model.steps[-1][1], "classes_", None)
    return tuple(np.asarray(classes if classes is not None else ()).ravel().tolist())


def _wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return float("nan")
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / float(total)
    centre = p + z * z / (2.0 * float(total))
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * float(total))) / float(total))
    return float(max(0.0, (centre - radius) / denominator))


def _hoeffding_upper_risk(errors: int, total: int, *, delta: float) -> float:
    if total <= 0:
        return float("inf")
    empirical = float(errors) / float(total)
    radius = math.sqrt(math.log(1.0 / float(delta)) / (2.0 * float(total)))
    return float(min(1.0, empirical + radius))


@dataclass(frozen=True, slots=True)
class DensityRatioProvenance:
    """Fixed calibration-only density ratios for an external estimator."""

    ratios: tuple[float, ...]
    ratio_sha256: str
    estimator_sha256: str
    feature_schema_sha256: str
    training_rows_sha256: str
    calibration_rows_sha256: str
    context_fingerprint: str
    clip_min: float = 0.05
    clip_max: float = 20.0
    minimum_effective_sample_size: float = 20.0
    minimum_overlap: float = 0.05
    trained_without_labels: bool = True
    trained_without_evaluation_rows: bool = True

    def validate(self, *, n_calibration: int, calibration_rows_sha256: str, context_fingerprint: str) -> tuple[np.ndarray, dict[str, float]]:
        if len(self.ratios) != int(n_calibration):
            raise ValueError("density_ratio_length_mismatch")
        if self.calibration_rows_sha256 != calibration_rows_sha256 or self.context_fingerprint != context_fingerprint:
            raise ValueError("density_ratio_provenance_mismatch")
        if not all(_valid_sha256(value) for value in (
            self.ratio_sha256, self.estimator_sha256, self.feature_schema_sha256,
            self.training_rows_sha256, self.calibration_rows_sha256, self.context_fingerprint,
        )):
            raise ValueError("density_ratio_invalid_sha256")
        raw = np.asarray(self.ratios, dtype=float)
        if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
            raise ValueError("density_ratio_nonpositive_or_nonfinite")
        if _sha256([float(value) for value in raw.tolist()]) != self.ratio_sha256:
            raise ValueError("density_ratio_digest_mismatch")
        if not self.trained_without_labels or not self.trained_without_evaluation_rows:
            raise ValueError("density_ratio_training_leakage")
        if not (0.0 < self.clip_min <= self.clip_max and self.minimum_effective_sample_size > 0.0):
            raise ValueError("density_ratio_invalid_clipping")
        clipped = np.clip(raw, self.clip_min, self.clip_max)
        mass = float(clipped.sum())
        ess = float(mass * mass / float(np.square(clipped).sum()))
        overlap = float(np.mean((raw >= self.clip_min) & (raw <= self.clip_max)))
        if not np.isfinite(mass) or mass <= 0.0 or ess < self.minimum_effective_sample_size or overlap < self.minimum_overlap:
            raise ValueError("density_ratio_overlap_failure")
        return clipped, {
            "effective_sample_size": ess,
            "overlap_fraction": overlap,
            "clipped_fraction": float(np.mean(clipped != raw)),
        }


@dataclass(frozen=True, slots=True)
class ConformalRiskControlRequest:
    """Bound calibration/evaluation inputs for selective risk control."""

    model: Any
    descriptor: FittedClassifierDescriptor
    X_calibration: np.ndarray
    y_calibration: np.ndarray
    X_evaluation: np.ndarray
    y_evaluation: np.ndarray | None
    calibration_rows_sha256: str
    evaluation_rows_sha256: str
    context_fingerprint: str
    split_fingerprint: str
    resampling_policy: str = "iid"
    group_axis: str = ""
    calibration_groups: Sequence[Any] | None = None
    evaluation_groups: Sequence[Any] | None = None
    target_risk: float = 0.10
    confidence_delta: float = 0.05
    min_stratum_size: int = 20
    threshold_grid: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90)
    density_ratio: DensityRatioProvenance | None = None


def _base_result(request: ConformalRiskControlRequest) -> dict[str, Any]:
    return {
        "classifier_conformal_risk_enabled": True,
        "classifier_conformal_risk_applied": False,
        "classifier_conformal_risk_fallback_reason": "",
        "classifier_conformal_risk_guarantee_status": "not_available",
        "classifier_conformal_risk_target_risk": float(request.target_risk),
        "classifier_conformal_risk_confidence_delta": float(request.confidence_delta),
        "classifier_conformal_risk_group_axis": str(request.group_axis),
        "classifier_conformal_risk_calibration_rows_sha256": str(request.calibration_rows_sha256),
        "classifier_conformal_risk_evaluation_rows_sha256": str(request.evaluation_rows_sha256),
        "classifier_conformal_risk_context_fingerprint": str(request.context_fingerprint),
        "classifier_conformal_risk_split_fingerprint": str(request.split_fingerprint),
        "classifier_conformal_risk_score_source": "predict_proba",
        "classifier_conformal_risk_probability_kind": request.descriptor.fitted_probability_kind.value,
        "classifier_conformal_risk_stratum_class_role": "predicted_class",
        "classifier_conformal_risk_strata": [],
        "classifier_conformal_risk_thresholds": [],
        "classifier_conformal_risk_abstention_rate": float("nan"),
        "classifier_conformal_risk_selective_risk": float("nan"),
        "classifier_conformal_risk_selective_risk_upper": float("nan"),
        "classifier_conformal_risk_coverage": float("nan"),
        "classifier_conformal_risk_coverage_lcb": float("nan"),
        "classifier_conformal_risk_density_ratio": {},
    }


def fit_mondrian_risk_control(request: ConformalRiskControlRequest) -> dict[str, Any]:
    """Fit class/(optional group)-conditional selective risk thresholds.

    The policy chooses the lowest predeclared confidence threshold whose
    Hoeffding upper error bound meets ``target_risk`` in every stratum. This
    standalone evaluator is empirical-only because its opaque hashes cannot
    attest calibration/model/evaluation separation. A future #236/#237
    procedure-bound adapter may expose a guarantee-bearing status; weighted
    covariate-shift results remain empirical-only in either case.
    """

    out = _base_result(request)
    if request.group_axis not in _GROUP_AXES:
        out["classifier_conformal_risk_fallback_reason"] = "unknown_group_axis"
        return out
    if str(request.resampling_policy).strip().lower() not in _IID_POLICIES:
        out["classifier_conformal_risk_fallback_reason"] = "unsupported_resampling_policy"
        return out
    if not all(_valid_sha256(str(value)) for value in (
        request.calibration_rows_sha256, request.evaluation_rows_sha256,
        request.context_fingerprint, request.split_fingerprint,
    )):
        out["classifier_conformal_risk_fallback_reason"] = "invalid_provenance_sha256"
        return out
    if not (0.0 < float(request.target_risk) < 1.0 and 0.0 < float(request.confidence_delta) < 1.0):
        out["classifier_conformal_risk_fallback_reason"] = "invalid_risk_parameters"
        return out
    try:
        thresholds = tuple(sorted({float(value) for value in request.threshold_grid}))
    except (TypeError, ValueError):
        out["classifier_conformal_risk_fallback_reason"] = "invalid_threshold_grid"
        return out
    minimum_stratum_size = request.min_stratum_size
    try:
        validated_minimum_stratum_size = int(minimum_stratum_size)
    except (TypeError, ValueError, OverflowError):
        out["classifier_conformal_risk_fallback_reason"] = "invalid_min_stratum_size"
        return out
    if (
        isinstance(minimum_stratum_size, bool)
        or validated_minimum_stratum_size != minimum_stratum_size
        or validated_minimum_stratum_size < 1
    ):
        out["classifier_conformal_risk_fallback_reason"] = "invalid_min_stratum_size"
        return out
    minimum_stratum_size = validated_minimum_stratum_size
    if (
        not thresholds
        or not all(np.isfinite(value) for value in thresholds)
        or thresholds[0] < 0.0
        or thresholds[-1] > 1.0
    ):
        out["classifier_conformal_risk_fallback_reason"] = "invalid_threshold_grid"
        return out
    x_cal = np.asarray(request.X_calibration, dtype=float)
    y_cal = np.asarray(request.y_calibration, dtype=object).ravel()
    x_eval = np.asarray(request.X_evaluation, dtype=float)
    y_eval = None if request.y_evaluation is None else np.asarray(request.y_evaluation, dtype=object).ravel()
    if x_cal.ndim != 2 or x_eval.ndim != 2 or x_cal.shape[1] != x_eval.shape[1] or y_cal.size != x_cal.shape[0]:
        out["classifier_conformal_risk_fallback_reason"] = "invalid_input_alignment"
        return out
    if y_eval is not None and y_eval.size != x_eval.shape[0]:
        out["classifier_conformal_risk_fallback_reason"] = "evaluation_label_mismatch"
        return out
    if not np.all(np.isfinite(x_cal)) or not np.all(np.isfinite(x_eval)):
        out["classifier_conformal_risk_fallback_reason"] = "nonfinite_features"
        return out
    if request.group_axis:
        cal_groups = tuple(request.calibration_groups or ())
        eval_groups = tuple(request.evaluation_groups or ())
        if len(cal_groups) != y_cal.size or len(eval_groups) != x_eval.shape[0] or any(value is None for value in (*cal_groups, *eval_groups)):
            out["classifier_conformal_risk_fallback_reason"] = "group_alignment_failure"
            return out
    else:
        cal_groups = tuple("__all__" for _ in range(y_cal.size))
        eval_groups = tuple("__all__" for _ in range(x_eval.shape[0]))
    result_cal = extract_probability_matrix(request.model, x_cal, request.descriptor, requirement=ProbabilityRequirement.GENUINE)
    result_eval = extract_probability_matrix(request.model, x_eval, request.descriptor, requirement=ProbabilityRequirement.GENUINE)
    if not result_cal.available or not result_eval.available or result_cal.matrix is None or result_eval.matrix is None:
        out["classifier_conformal_risk_fallback_reason"] = "genuine_probability_required"
        return out
    classes = _model_classes(request.model)
    if (
        len(classes) != int(np.asarray(result_cal.matrix).shape[1])
        or tuple(result_eval.class_order) != tuple(result_cal.class_order)
        or len({_typed_key(value) for value in classes}) != len(classes)
    ):
        out["classifier_conformal_risk_fallback_reason"] = "probability_class_order_mismatch"
        return out
    class_positions = {_typed_key(value): index for index, value in enumerate(classes)}
    true_positions = np.asarray([class_positions.get(_typed_key(value), -1) for value in y_cal], dtype=int)
    if np.any(true_positions < 0):
        out["classifier_conformal_risk_fallback_reason"] = "calibration_label_outside_class_order"
        return out
    weights = np.ones(y_cal.size, dtype=float)
    if request.density_ratio is not None:
        try:
            weights, ratio_meta = request.density_ratio.validate(
                n_calibration=y_cal.size,
                calibration_rows_sha256=request.calibration_rows_sha256,
                context_fingerprint=request.context_fingerprint,
            )
        except ValueError as exc:
            out["classifier_conformal_risk_fallback_reason"] = str(exc)
            return out
        out["classifier_conformal_risk_density_ratio"] = {
            "ratio_sha256": request.density_ratio.ratio_sha256,
            "estimator_sha256": request.density_ratio.estimator_sha256,
            "feature_schema_sha256": request.density_ratio.feature_schema_sha256,
            **ratio_meta,
        }
    probabilities_cal = np.asarray(result_cal.matrix, dtype=float)
    calibration_predicted = np.argmax(probabilities_cal, axis=1)
    strata: dict[tuple[tuple[str, str, str], tuple[str, str, str]], list[int]] = {}
    for index, (class_index, group) in enumerate(zip(calibration_predicted.tolist(), cal_groups)):
        strata.setdefault((_typed_key(classes[int(class_index)]), _typed_key(group)), []).append(index)
    if not strata or any(len(indices) < minimum_stratum_size for indices in strata.values()):
        out["classifier_conformal_risk_fallback_reason"] = "insufficient_mondrian_stratum"
        return out
    selected: dict[tuple[tuple[str, str, str], tuple[str, str, str]], float] = {}
    stratum_rows: list[dict[str, Any]] = []
    stratum_rows_by_key: dict[
        tuple[tuple[str, str, str], tuple[str, str, str]], dict[str, Any]
    ] = {}
    for key, indices in sorted(strata.items(), key=lambda item: repr(item[0])):
        idx = np.asarray(indices, dtype=int)
        predicted_position = calibration_predicted[idx]
        confidence = probabilities_cal[idx, predicted_position]
        errors_by_row = predicted_position != true_positions[idx]
        chosen: float | None = None
        bound = float("inf")
        for candidate in thresholds:
            accepted = confidence >= candidate
            n_accepted = int(np.sum(accepted))
            if n_accepted < minimum_stratum_size:
                continue
            errors = int(np.sum(errors_by_row[accepted]))
            if request.density_ratio is None:
                candidate_bound = _hoeffding_upper_risk(
                    errors, n_accepted,
                    delta=(
                        float(request.confidence_delta)
                        / float(len(strata) * len(thresholds))
                    ),
                )
            else:
                accepted_weights = weights[idx][accepted]
                weighted_risk = float(np.sum(accepted_weights * errors_by_row[accepted]) / np.sum(accepted_weights))
                candidate_bound = weighted_risk
            if candidate_bound <= float(request.target_risk):
                chosen, bound = candidate, candidate_bound
                break
        if chosen is None:
            out["classifier_conformal_risk_fallback_reason"] = "no_risk_qualified_threshold"
            return out
        selected[key] = chosen
        row = {
            "class_sha256": _sha256(key[0]), "group_sha256": _sha256(key[1]),
            "calibration_count": int(idx.size), "threshold": float(chosen),
            "risk_upper": float(bound), "weighted_mass": float(weights[idx].sum()),
            "evaluation_count": 0, "evaluation_accepted_count": 0,
            "evaluation_selective_risk": float("nan"),
            "evaluation_coverage_lcb": float("nan"),
        }
        stratum_rows.append(row)
        stratum_rows_by_key[key] = row
    probabilities_eval = np.asarray(result_eval.matrix, dtype=float)
    predicted = np.argmax(probabilities_eval, axis=1)
    accepted = np.zeros(x_eval.shape[0], dtype=bool)
    for index, (class_index, group) in enumerate(zip(predicted.tolist(), eval_groups)):
        key = (_typed_key(classes[int(class_index)]), _typed_key(group))
        threshold = selected.get(key)
        if threshold is None:
            out["classifier_conformal_risk_fallback_reason"] = "unknown_evaluation_stratum"
            return out
        accepted[index] = bool(probabilities_eval[index, int(class_index)] >= threshold)
    accepted_count = int(np.sum(accepted))
    out.update({
        "classifier_conformal_risk_applied": True,
        # A standalone request carries opaque hashes, not a procedure-bound
        # proof that model fit, calibration, and evaluation rows are disjoint.
        "classifier_conformal_risk_guarantee_status": "empirical_only",
        "classifier_conformal_risk_strata": stratum_rows,
        "classifier_conformal_risk_thresholds": sorted({float(value) for value in selected.values()}),
        "classifier_conformal_risk_abstention_rate": float(1.0 - accepted.mean()) if accepted.size else float("nan"),
        "classifier_conformal_risk_coverage": float(accepted.mean()) if accepted.size else float("nan"),
        "classifier_conformal_risk_coverage_lcb": _wilson_lower_bound(accepted_count, int(accepted.size)),
    })
    if y_eval is not None and accepted_count:
        predicted_labels = np.asarray([classes[int(value)] for value in predicted], dtype=object)
        errors = int(np.sum([
            not _typed_equal(predicted_label, true_label)
            for predicted_label, true_label in zip(
                predicted_labels[accepted].tolist(), y_eval[accepted].tolist()
            )
        ]))
        out["classifier_conformal_risk_selective_risk"] = float(errors / accepted_count)
        out["classifier_conformal_risk_selective_risk_upper"] = _hoeffding_upper_risk(errors, accepted_count, delta=float(request.confidence_delta))
        for key, row in stratum_rows_by_key.items():
            class_key, group_key = key
            positions = np.asarray([
                index
                for index, (class_index, group) in enumerate(zip(predicted.tolist(), eval_groups))
                if _typed_key(classes[int(class_index)]) == class_key
                and _typed_key(group) == group_key
            ], dtype=int)
            if not positions.size:
                continue
            accepted_positions = positions[accepted[positions]]
            row["evaluation_count"] = int(positions.size)
            row["evaluation_accepted_count"] = int(accepted_positions.size)
            row["evaluation_coverage_lcb"] = _wilson_lower_bound(
                int(accepted_positions.size), int(positions.size)
            )
            if accepted_positions.size:
                stratum_errors = int(np.sum([
                    not _typed_equal(predicted_label, true_label)
                    for predicted_label, true_label in zip(
                        predicted_labels[accepted_positions].tolist(),
                        y_eval[accepted_positions].tolist(),
                    )
                ]))
                row["evaluation_selective_risk"] = float(
                    stratum_errors / int(accepted_positions.size)
                )
    return out


__all__ = [
    "ConformalRiskControlRequest", "DensityRatioProvenance", "fit_mondrian_risk_control",
]

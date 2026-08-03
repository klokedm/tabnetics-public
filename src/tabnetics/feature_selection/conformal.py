"""Conformal prediction-set utilities for Stage-2 classifiers.

This module provides:
1. Split-conformal prediction sets (default, lightweight, model-agnostic).
2. MAPIE-style APS/RAPS integration (opt-in, requires ``mapie`` package).

Both modes use a train/calibration split from the provided training set and
return diagnostics that can be attached to benchmark artifacts without changing
default behavior.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from tabnetics.classification.conformance import (
    FittedClassifierDescriptor,
    ProbabilityRequirement,
    check_probability_requirement,
    extract_probability_matrix,
    inspect_fitted_classifier,
)
from tabnetics.classification.registry import ProbabilityKind


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    den = np.sum(e, axis=1, keepdims=True)
    den[~np.isfinite(den) | (den <= 0.0)] = 1.0
    return e / den


def _label_key(value: Any) -> Tuple[str, Any]:
    scalar = value.item() if isinstance(value, np.generic) else value
    return f"{type(scalar).__module__}.{type(scalar).__qualname__}", scalar


def _typed_class_order(classes: np.ndarray) -> List[Dict[str, Any]]:
    values = list(classes.ravel()) if isinstance(classes, np.ndarray) else list(classes)
    return [
        {
            "value": _json_label_value(value),
            "python_type": _label_key(value)[0],
            "numpy_dtype": str(np.asarray(value).dtype),
        }
        for value in values
    ]


def _json_label_value(value: Any) -> Any:
    scalar = value.item() if isinstance(value, np.generic) else value
    if scalar is None or isinstance(scalar, (str, bool, int, float)):
        if isinstance(scalar, float) and not np.isfinite(scalar):
            return None
        return scalar
    return str(scalar)


def _strict_score_matrix_for_source(
    model: Any,
    X: np.ndarray,
    classes: np.ndarray,
    *,
    source: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(X, dtype=float)
    cls = np.asarray(classes).ravel()
    if x.ndim != 2 or cls.size < 2:
        raise ValueError("invalid_score_input")
    n = int(x.shape[0])
    c = int(cls.size)
    class_keys = [_label_key(value) for value in cls]
    if len(set(class_keys)) != c:
        raise ValueError("duplicate_class_order")
    meta: Dict[str, Any] = {
        "score_source": str(source),
        "used_predict_proba": source == "predict_proba",
        "model_supports_predict_proba": callable(
            getattr(model, "predict_proba", None)
        ),
        "model_class_count": c,
    }

    if source == "predict_proba":
        method = getattr(model, "predict_proba", None)
        if not callable(method):
            raise ValueError("predict_proba_unavailable")
        raw = np.asarray(method(x), dtype=float)
        model_classes = np.asarray(getattr(model, "classes_", ())).ravel()
        model_keys = [_label_key(value) for value in model_classes]
        if raw.ndim != 2 or raw.shape != (n, model_classes.size):
            raise ValueError("invalid_probability_shape")
        if len(set(model_keys)) != len(model_keys) or set(model_keys) != set(class_keys):
            raise ValueError("probability_class_alignment_failed")
        raw = raw[:, np.asarray([model_keys.index(key) for key in class_keys], dtype=int)]
        if not np.all(np.isfinite(raw)):
            raise ValueError("nonfinite_probability")
        if np.any(raw < 0.0):
            raise ValueError("negative_probability")
        row_sums = np.sum(raw, axis=1)
        if np.any(row_sums <= 1e-12) or not np.allclose(
            row_sums, 1.0, atol=1e-6, rtol=1e-6
        ):
            raise ValueError("invalid_probability_simplex")
        return raw / row_sums[:, None], meta

    if source in {"decision_function_sigmoid", "decision_function_softmax"}:
        method = getattr(model, "decision_function", None)
        if not callable(method):
            raise ValueError("decision_function_unavailable")
        decision = np.asarray(method(x), dtype=float)
        if not np.all(np.isfinite(decision)):
            raise ValueError("nonfinite_decision_score")
        if source == "decision_function_sigmoid":
            if c != 2 or decision.ndim != 1 or decision.size != n:
                raise ValueError("invalid_binary_decision_shape")
            clipped = np.clip(decision, -700.0, 700.0)
            p1 = 1.0 / (1.0 + np.exp(-clipped))
            return np.column_stack([1.0 - p1, p1]), meta
        if decision.ndim != 2 or decision.shape != (n, c):
            raise ValueError("invalid_multiclass_decision_shape")
        return _softmax(decision), meta

    if source == "hard_vote_fraction":
        if not isinstance(model, VotingClassifier) or str(model.voting).lower() != "hard":
            raise ValueError("hard_vote_unavailable")
        estimators = list(getattr(model, "estimators_", ()) or ())
        if not estimators:
            raise ValueError("hard_vote_members_unavailable")
        votes = np.zeros((n, c), dtype=float)
        class_to_idx = {key: idx for idx, key in enumerate(class_keys)}
        raw_weights = list(getattr(model, "weights", ()) or ())
        if raw_weights and len(raw_weights) != len(estimators):
            raise ValueError("hard_vote_weight_count_mismatch")
        vote_weights = (
            np.asarray(raw_weights, dtype=float)
            if raw_weights
            else np.ones(len(estimators), dtype=float)
        )
        if (
            not np.all(np.isfinite(vote_weights))
            or np.any(vote_weights < 0.0)
            or float(np.sum(vote_weights)) <= 1e-12
        ):
            raise ValueError("hard_vote_invalid_weights")
        for estimator_idx, estimator in enumerate(estimators):
            pred = np.asarray(estimator.predict(x)).ravel()
            if pred.size != n:
                raise ValueError("hard_vote_prediction_size_mismatch")
            if any(_label_key(label) not in class_to_idx for label in pred):
                encoder = getattr(model, "le_", None)
                if encoder is None:
                    raise ValueError("hard_vote_label_encoder_unavailable")
                try:
                    encoded = np.asarray(pred, dtype=float)
                    if not np.all(np.isfinite(encoded)) or not np.allclose(
                        encoded, np.rint(encoded), atol=0.0, rtol=0.0
                    ):
                        raise ValueError("hard_vote_noninteger_code")
                    encoded_int = np.asarray(np.rint(encoded), dtype=int)
                    if np.any(encoded_int < 0) or np.any(encoded_int >= c):
                        raise ValueError("hard_vote_code_out_of_range")
                    pred = np.asarray(
                        encoder.inverse_transform(encoded_int)
                    ).ravel()
                except ValueError:
                    raise
                except Exception as exc:
                    raise ValueError("hard_vote_decode_failed") from exc
            for row_idx, label in enumerate(pred):
                key = _label_key(label)
                if key not in class_to_idx:
                    raise ValueError("hard_vote_unknown_label")
                votes[row_idx, class_to_idx[key]] += float(
                    vote_weights[estimator_idx]
                )
        return votes / float(np.sum(vote_weights)), meta

    if source != "predict_one_hot":
        raise ValueError("unknown_score_source")
    pred = np.asarray(model.predict(x)).ravel()
    if pred.size != n:
        raise ValueError("prediction_size_mismatch")
    class_to_idx = {key: idx for idx, key in enumerate(class_keys)}
    out = np.zeros((n, c), dtype=float)
    for row_idx, label in enumerate(pred):
        key = _label_key(label)
        if key not in class_to_idx:
            raise ValueError("predict_unknown_label")
        out[row_idx, class_to_idx[key]] = 1.0
    return out, meta


def _choose_conformity_source(
    model: Any,
    X_probe: np.ndarray,
    classes: np.ndarray,
    *,
    allow_probability_matrix: Optional[bool] = None,
) -> Tuple[str, str, Dict[str, Dict[str, str]]]:
    errors: Dict[str, Dict[str, str]] = {}
    candidates: List[Tuple[str, str]] = []
    allow_proba = bool(allow_probability_matrix)
    if allow_probability_matrix is None and callable(
        getattr(model, "predict_proba", None)
    ):
        try:
            observed, _ = _strict_score_matrix_for_source(
                model, X_probe, classes, source="predict_proba"
            )
            predicted = np.asarray(model.predict(np.asarray(X_probe, dtype=float))).ravel()
            expected = np.asarray(classes)[np.argmax(observed, axis=1)]
            is_one_hot = bool(
                np.all((observed == 0.0) | (observed == 1.0))
                and np.all(np.sum(observed == 1.0, axis=1) == 1)
            )
            is_predict_proxy = bool(
                predicted.size == expected.size
                and all(
                    _label_key(left) == _label_key(right)
                    for left, right in zip(predicted, expected)
                )
            )
            allow_proba = not (is_one_hot and is_predict_proxy)
            if not allow_proba:
                errors["predict_proba"] = {
                    "reason": "hard_label_proxy_observed",
                    "exception_type": "none",
                }
        except Exception as exc:
            errors["predict_proba"] = _score_error_record(exc)
    if allow_proba and callable(getattr(model, "predict_proba", None)):
        candidates.append(("predict_proba", "probability_matrix"))
    if callable(getattr(model, "decision_function", None)):
        candidates.append(
            (
                "decision_function_sigmoid"
                if int(np.asarray(classes).size) == 2
                else "decision_function_softmax",
                "decision_score",
            )
        )
    if isinstance(model, VotingClassifier) and str(model.voting).lower() == "hard":
        candidates.append(("hard_vote_fraction", "hard_vote_fraction"))
    candidates.append(("predict_one_hot", "hard_label"))
    for source, kind in candidates:
        try:
            _strict_score_matrix_for_source(
                model, X_probe, classes, source=source
            )
            return source, kind, errors
        except Exception as exc:
            errors[source] = _score_error_record(exc)
    raise ValueError("no_valid_conformity_source")


def _score_error_record(exc: Exception) -> Dict[str, str]:
    canonical_reasons = {
        "invalid_score_input",
        "duplicate_class_order",
        "predict_proba_unavailable",
        "invalid_probability_shape",
        "probability_class_alignment_failed",
        "nonfinite_probability",
        "negative_probability",
        "invalid_probability_simplex",
        "decision_function_unavailable",
        "nonfinite_decision_score",
        "invalid_binary_decision_shape",
        "invalid_multiclass_decision_shape",
        "hard_vote_unavailable",
        "hard_vote_members_unavailable",
        "hard_vote_weight_count_mismatch",
        "hard_vote_invalid_weights",
        "hard_vote_prediction_size_mismatch",
        "hard_vote_label_encoder_unavailable",
        "hard_vote_noninteger_code",
        "hard_vote_code_out_of_range",
        "hard_vote_decode_failed",
        "hard_vote_unknown_label",
        "unknown_score_source",
        "prediction_size_mismatch",
        "predict_unknown_label",
        "no_valid_conformity_source",
    }
    reason = "score_source_failed"
    if isinstance(exc, ValueError) and exc.args and isinstance(exc.args[0], str):
        candidate = str(exc.args[0])
        if candidate in canonical_reasons:
            reason = candidate
    return {"reason": reason, "exception_type": type(exc).__name__}


def _safe_score_matrix_with_meta(
    model: Any,
    X: np.ndarray,
    classes: np.ndarray,
    *,
    source: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    selected = source
    if selected is None:
        selected, _, _ = _choose_conformity_source(model, X, classes)
    return _strict_score_matrix_for_source(
        model, X, classes, source=str(selected)
    )


def _safe_score_matrix(model: Any, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return normalized class-score rows, preserving the historical API."""

    scores, _ = _safe_score_matrix_with_meta(model, X, classes)
    return scores


def _quantile_higher(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    qq = float(np.clip(q, 0.0, 1.0))
    try:
        return float(np.quantile(arr, qq, method="higher"))
    except TypeError:  # numpy<1.22
        return float(np.quantile(arr, qq, interpolation="higher"))


def compute_split_conformal_sets(
    *,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: Optional[np.ndarray] = None,
    alpha: float = 0.10,
    calibration_fraction: float = 0.25,
    min_calibration: int = 20,
    seed: int = 0,
    include_prediction_sets: bool = False,
    include_score_source: bool = False,
    classifier_identity: Optional[Mapping[str, Any]] = None,
    classifier_backend: str = "standalone",
    calibration_indices: Optional[
        Tuple[Sequence[int], Sequence[int]]
    ] = None,
) -> Dict[str, Any]:
    """Compute split-conformal prediction sets and summary diagnostics."""
    x_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train).ravel()
    x_ev = np.asarray(X_eval, dtype=float)
    y_ev = None if y_eval is None else np.asarray(y_eval).ravel()

    alpha_val = float(np.clip(alpha, 1e-4, 0.49))
    cal_frac = float(np.clip(calibration_fraction, 0.05, 0.95))
    min_cal = int(max(2, min_calibration))

    out: Dict[str, Any] = {
        "classifier_conformal_enabled": True,
        "classifier_conformal_applied": False,
        "classifier_conformal_skip_reason": "",
        "classifier_conformal_alpha": float(alpha_val),
        "classifier_conformal_calibration_fraction": float(cal_frac),
        "classifier_conformal_min_calibration": int(min_cal),
        "classifier_conformal_calibration_size": 0,
        "classifier_conformal_fit_size": 0,
        "classifier_conformal_qhat": float("nan"),
        "classifier_conformal_threshold": float("nan"),
        "classifier_conformal_set_size_mean": float("nan"),
        "classifier_conformal_set_size_median": float("nan"),
        "classifier_conformal_singleton_rate": float("nan"),
        "classifier_conformal_empty_set_rate": float("nan"),
        "classifier_conformal_coverage": float("nan"),
        "classifier_conformal_classes": [],
        "classifier_conformal_prediction_sets": [],
        "classifier_conformal_model_probability_kind": ProbabilityKind.UNKNOWN.value,
        "classifier_conformal_conformity_kind": "unavailable",
        "classifier_conformal_score_source": "unavailable",
        "classifier_conformal_calibration_score_source": "unavailable",
        "classifier_conformal_evaluation_score_source": "unavailable",
        "classifier_conformal_class_order": [],
        "classifier_conformal_source_consistent": False,
        "classifier_conformal_probability_required": False,
        "classifier_conformal_probability_claim": False,
        "classifier_conformal_used_predict_proba": False,
        "classifier_conformal_model_supports_predict_proba": False,
        "classifier_conformal_source_errors": {},
    }
    _ = include_score_source  # Compatibility flag; provenance is now mandatory.

    if x_tr.ndim != 2 or x_ev.ndim != 2:
        out["classifier_conformal_skip_reason"] = "invalid_input_rank"
        return out
    if x_tr.shape[1] != x_ev.shape[1]:
        out["classifier_conformal_skip_reason"] = "feature_dim_mismatch"
        return out
    if y_tr.size != x_tr.shape[0]:
        out["classifier_conformal_skip_reason"] = "train_label_mismatch"
        return out

    classes = np.unique(y_tr)
    n_classes = int(classes.size)
    if n_classes < 2:
        out["classifier_conformal_skip_reason"] = "single_class"
        return out
    out["classifier_conformal_classes"] = [str(c) for c in classes.tolist()]
    out["classifier_conformal_class_order"] = _typed_class_order(classes)

    n_train = int(x_tr.shape[0])
    min_required = int(max(min_cal + n_classes, 2 * n_classes + 2))
    if n_train < min_required:
        out["classifier_conformal_skip_reason"] = "insufficient_train_samples"
        return out

    cal_size = int(max(min_cal, round(cal_frac * float(n_train))))
    cal_size = int(min(cal_size, n_train - max(2, n_classes)))
    if cal_size < min_cal or cal_size < n_classes:
        out["classifier_conformal_skip_reason"] = "insufficient_calibration_size"
        return out

    if calibration_indices is not None:
        tr_idx = np.asarray(calibration_indices[0], dtype=int).ravel()
        cal_idx = np.asarray(calibration_indices[1], dtype=int).ravel()
        out["classifier_conformal_resampling_source"] = "resolved_plan"
    else:
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=int(cal_size),
                random_state=int(seed),
            )
            tr_idx, cal_idx = next(splitter.split(np.zeros((n_train, 1)), y_tr))
        except Exception:
            out["classifier_conformal_skip_reason"] = "stratified_split_failed"
            return out
        out["classifier_conformal_resampling_source"] = (
            "legacy_stratified_shuffle_split"
        )

    x_fit = x_tr[np.asarray(tr_idx, dtype=int)]
    y_fit = y_tr[np.asarray(tr_idx, dtype=int)]
    x_cal = x_tr[np.asarray(cal_idx, dtype=int)]
    y_cal = y_tr[np.asarray(cal_idx, dtype=int)]
    if np.unique(y_fit).size < 2:
        out["classifier_conformal_skip_reason"] = "fit_split_single_class"
        return out

    try:
        est = clone(model)
        est.fit(x_fit, y_fit)
    except Exception:
        out["classifier_conformal_skip_reason"] = "model_refit_failed"
        return out

    classes_fit = np.asarray(getattr(est, "classes_", np.unique(y_fit))).ravel()
    if classes_fit.size < 2:
        out["classifier_conformal_skip_reason"] = "fitted_model_single_class"
        return out

    identity = dict(classifier_identity or {})
    fitted_descriptor: Optional[FittedClassifierDescriptor] = None
    if identity:
        try:
            fitted_descriptor = inspect_fitted_classifier(
                est,
                canonical_name=identity.get("canonical_name"),
                registry_anchor_name=identity.get("registry_anchor_name"),
                backend=str(classifier_backend),
                requested_name=identity.get("requested_name"),
                outward_name=identity.get("outward_name"),
                effective_model_name=identity.get("effective_model_name"),
                selection_identity=identity,
                probe_X=np.asarray(x_fit, dtype=float)[: min(16, len(x_fit))],
            )
            out["classifier_conformal_model_probability_kind"] = (
                fitted_descriptor.fitted_probability_kind.value
            )
        except Exception as exc:
            out["classifier_conformal_skip_reason"] = (
                f"fitted_descriptor_{type(exc).__name__}"
            )
            return out

    try:
        selected_source, conformity_kind, source_errors = _choose_conformity_source(
            est,
            np.asarray(x_fit, dtype=float)[: min(16, len(x_fit))],
            classes_fit,
            allow_probability_matrix=(
                fitted_descriptor.probability_matrix_available
                if fitted_descriptor is not None
                else None
            ),
        )
        score_cal, score_meta_cal = _safe_score_matrix_with_meta(
            est, x_cal, classes_fit, source=selected_source
        )
    except Exception as exc:
        out["classifier_conformal_skip_reason"] = (
            f"calibration_score_source_{type(exc).__name__}"
        )
        return out
    if fitted_descriptor is None:
        out["classifier_conformal_model_probability_kind"] = (
            ProbabilityKind.SCORE_DERIVED.value
            if selected_source == "predict_proba"
            else ProbabilityKind.NONE.value
        )
    out.update(
        {
            "classifier_conformal_conformity_kind": str(conformity_kind),
            "classifier_conformal_score_source": str(selected_source),
            "classifier_conformal_calibration_score_source": str(selected_source),
            "classifier_conformal_class_order": _typed_class_order(classes_fit),
            "classifier_conformal_used_predict_proba": bool(
                score_meta_cal.get("used_predict_proba", False)
            ),
            "classifier_conformal_model_supports_predict_proba": bool(
                score_meta_cal.get("model_supports_predict_proba", False)
            ),
            "classifier_conformal_source_errors": dict(source_errors),
            "classifier_conformal_probability_claim": bool(
                selected_source == "predict_proba"
                and out["classifier_conformal_model_probability_kind"]
                in {
                    ProbabilityKind.NATIVE.value,
                    ProbabilityKind.CALIBRATED.value,
                    ProbabilityKind.SCORE_DERIVED.value,
                }
            ),
        }
    )
    class_to_idx = {
        _label_key(value): idx for idx, value in enumerate(classes_fit.tolist())
    }
    idx = np.asarray(
        [class_to_idx.get(_label_key(value), -1) for value in y_cal], dtype=int
    )
    valid = idx >= 0
    if int(np.sum(valid)) < min_cal:
        out["classifier_conformal_skip_reason"] = "insufficient_valid_calibration_points"
        return out
    rows = np.arange(idx.size, dtype=int)[valid]
    true_prob = score_cal[rows, idx[valid]]
    nonconformity = 1.0 - np.asarray(true_prob, dtype=float)
    n_cal = int(nonconformity.size)
    q_level = float(math.ceil((n_cal + 1) * (1.0 - alpha_val)) / float(max(1, n_cal)))
    q_hat = _quantile_higher(nonconformity, q_level)
    if not np.isfinite(q_hat):
        out["classifier_conformal_skip_reason"] = "invalid_qhat"
        return out
    threshold = float(1.0 - q_hat)

    try:
        score_eval, score_meta_eval = _safe_score_matrix_with_meta(
            est, x_ev, classes_fit, source=selected_source
        )
    except Exception as exc:
        out["classifier_conformal_skip_reason"] = (
            f"evaluation_score_source_{type(exc).__name__}"
        )
        out["classifier_conformal_evaluation_score_source"] = str(selected_source)
        out["classifier_conformal_source_consistent"] = False
        return out
    eval_source = str(score_meta_eval.get("score_source", "") or "")
    out["classifier_conformal_evaluation_score_source"] = eval_source
    out["classifier_conformal_source_consistent"] = bool(
        eval_source == str(selected_source)
    )
    if not out["classifier_conformal_source_consistent"]:
        out["classifier_conformal_skip_reason"] = "score_source_mismatch"
        return out
    pred_sets_idx: List[np.ndarray] = []
    for row in np.asarray(score_eval, dtype=float):
        keep = np.where(np.asarray(row, dtype=float) >= threshold - 1e-12)[0]
        if keep.size == 0:
            keep = np.asarray([int(np.argmax(row))], dtype=int)
        pred_sets_idx.append(np.asarray(keep, dtype=int))

    set_sizes = np.asarray([int(v.size) for v in pred_sets_idx], dtype=float)
    coverage = float("nan")
    if y_ev is not None and y_ev.size == len(pred_sets_idx):
        hit = []
        for label, keep_idx in zip(y_ev, pred_sets_idx):
            labels = {classes_fit[int(i)] for i in keep_idx.tolist()}
            hit.append(1.0 if label in labels else 0.0)
        coverage = float(np.mean(np.asarray(hit, dtype=float))) if hit else float("nan")

    out["classifier_conformal_applied"] = True
    out["classifier_conformal_calibration_size"] = int(n_cal)
    out["classifier_conformal_fit_size"] = int(x_fit.shape[0])
    out["classifier_conformal_qhat"] = float(q_hat)
    out["classifier_conformal_threshold"] = float(threshold)
    out["classifier_conformal_set_size_mean"] = float(np.mean(set_sizes)) if set_sizes.size else float("nan")
    out["classifier_conformal_set_size_median"] = float(np.median(set_sizes)) if set_sizes.size else float("nan")
    out["classifier_conformal_singleton_rate"] = (
        float(np.mean(set_sizes <= 1.0)) if set_sizes.size else float("nan")
    )
    out["classifier_conformal_empty_set_rate"] = (
        float(np.mean(set_sizes == 0.0)) if set_sizes.size else float("nan")
    )
    out["classifier_conformal_coverage"] = float(coverage)
    out["classifier_conformal_used_predict_proba"] = bool(
        score_meta_cal.get("used_predict_proba", False)
        and score_meta_eval.get("used_predict_proba", False)
    )
    out["classifier_conformal_model_supports_predict_proba"] = bool(
        score_meta_cal.get("model_supports_predict_proba", False)
        or score_meta_eval.get("model_supports_predict_proba", False)
    )

    if include_prediction_sets:
        serialised: List[List[str]] = []
        for keep_idx in pred_sets_idx:
            serialised.append([str(classes_fit[int(i)]) for i in keep_idx.tolist()])
        out["classifier_conformal_prediction_sets"] = serialised

    return out


# ---------------------------------------------------------------------------
# MAPIE APS/RAPS integration (VAL12_Suggestions §2.3)
# ---------------------------------------------------------------------------

# MAPIE >=1.0 uses SplitConformalClassifier / CrossConformalClassifier.
# MAPIE <1.0 used MapieClassifier.  Keep both imports inside the opt-in path:
# loading this module is part of the normal split-conformal pipeline and must
# not pull external callable objects into its canonical execution closure.
def _load_mapie_api() -> tuple[bool, Any, Any, Any, Any] | None:
    """Return the installed MAPIE API lazily, without retaining global aliases."""

    try:
        from mapie.classification import (  # type: ignore
            CrossConformalClassifier,
            RAPSConformityScore,
            SplitConformalClassifier,
        )

        return (
            False,
            SplitConformalClassifier,
            CrossConformalClassifier,
            RAPSConformityScore,
            None,
        )
    except Exception:
        try:
            from mapie.classification import MapieClassifier  # type: ignore

            return (True, None, None, None, MapieClassifier)
        except Exception:  # pragma: no cover - optional dependency unavailable.
            return None


def mapie_available() -> bool:
    """Return True if the optional MAPIE API can be imported on demand."""

    return _load_mapie_api() is not None


def _mapie_fitted_estimators(mapie_model: Any) -> List[Any]:
    """Return fitted estimator objects exposed by supported MAPIE versions."""

    found: List[Any] = []
    seen: set[int] = set()
    pending: List[Any] = [mapie_model]
    while pending:
        node = pending.pop(0)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        children: List[Any] = []
        for attr in (
            "_mapie_classifier",
            "estimator_",
            "estimators_",
            "single_estimator_",
            "estimator",
        ):
            value = getattr(node, attr, None)
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            children.extend(candidate for candidate in values if candidate is not None)
        pending.extend(children)
        if (
            np.asarray(getattr(node, "classes_", ())).size >= 2
            and callable(getattr(node, "predict_proba", None))
        ):
            found.append(node)
    return found


def compute_mapie_conformal_sets(
    *,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: Optional[np.ndarray] = None,
    alpha: float = 0.10,
    method: str = "aps",
    seed: int = 0,
    cv_folds: int = 5,
    include_prediction_sets: bool = False,
    classifier_descriptor: Optional[FittedClassifierDescriptor] = None,
    classifier_identity: Optional[Mapping[str, Any]] = None,
    classifier_backend: str = "standalone",
    calibration_indices: Optional[
        Tuple[Sequence[int], Sequence[int]]
    ] = None,
    structured_resampling: bool = False,
) -> Dict[str, Any]:
    """Compute MAPIE-style conformal prediction sets.

    Parameters
    ----------
    method : str
        ``"aps"`` for Adaptive Prediction Sets, ``"raps"`` for Regularized
        APS, ``"cross"`` for cross-conformal via MAPIE's internal CV.
    cv_folds : int
        Number of CV folds for cross-conformal (``method="cross"``).  Ignored
        for ``"aps"`` and ``"raps"`` which use ``cv="prefit"``.
    """
    x_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train).ravel()
    x_ev = np.asarray(X_eval, dtype=float)
    y_ev = None if y_eval is None else np.asarray(y_eval).ravel()

    alpha_val = float(np.clip(alpha, 1e-4, 0.49))
    method_key = str(method or "aps").strip().lower()
    if method_key not in {"aps", "raps", "cross"}:
        method_key = "aps"

    prefix = "classifier_conformal_mapie_"
    out: Dict[str, Any] = {
        f"{prefix}enabled": True,
        f"{prefix}applied": False,
        f"{prefix}skip_reason": "",
        f"{prefix}method": str(method_key),
        f"{prefix}alpha": float(alpha_val),
        f"{prefix}set_size_mean": float("nan"),
        f"{prefix}set_size_median": float("nan"),
        f"{prefix}singleton_rate": float("nan"),
        f"{prefix}empty_set_rate": float("nan"),
        f"{prefix}coverage": float("nan"),
        f"{prefix}classes": [],
        f"{prefix}prediction_sets": [],
        f"{prefix}probability_requirement": ProbabilityRequirement.MATRIX.value,
        f"{prefix}probability_kind": ProbabilityKind.UNKNOWN.value,
        f"{prefix}probability_source": "unavailable",
        f"{prefix}probability_admitted": False,
        f"{prefix}probability_reason": "unobserved",
        f"{prefix}class_order": [],
        f"{prefix}external_callable_identity": "not_loaded",
    }

    mapie_api = _load_mapie_api()
    if mapie_api is None:
        out[f"{prefix}skip_reason"] = "mapie_not_installed"
        return out
    (
        mapie_legacy,
        split_conformal_classifier,
        cross_conformal_classifier,
        raps_conformity_score,
        mapie_classifier,
    ) = mapie_api
    out[f"{prefix}external_callable_identity"] = "unattested:mapie"

    if x_tr.ndim != 2 or x_ev.ndim != 2:
        out[f"{prefix}skip_reason"] = "invalid_input_rank"
        return out
    if x_tr.shape[1] != x_ev.shape[1]:
        out[f"{prefix}skip_reason"] = "feature_dim_mismatch"
        return out
    if y_tr.size != x_tr.shape[0]:
        out[f"{prefix}skip_reason"] = "train_label_mismatch"
        return out

    classes = np.unique(y_tr)
    n_classes = int(classes.size)
    if n_classes < 2:
        out[f"{prefix}skip_reason"] = "single_class"
        return out
    out[f"{prefix}classes"] = [str(c) for c in classes.tolist()]
    out[f"{prefix}class_order"] = _typed_class_order(classes)

    identity = dict(classifier_identity or {})
    semantic_model = model
    semantic_descriptor = classifier_descriptor
    try:
        if semantic_descriptor is None:
            semantic_model = clone(model)
            semantic_model.fit(x_tr, y_tr)
            if identity:
                semantic_descriptor = inspect_fitted_classifier(
                    semantic_model,
                    canonical_name=identity.get("canonical_name"),
                    registry_anchor_name=identity.get("registry_anchor_name"),
                    backend=str(classifier_backend),
                    requested_name=identity.get("requested_name"),
                    outward_name=identity.get("outward_name"),
                    effective_model_name=identity.get("effective_model_name"),
                    selection_identity=identity,
                    probe_X=x_tr[: min(16, len(x_tr))],
                )
        if semantic_descriptor is not None:
            admission = check_probability_requirement(
                semantic_descriptor.fitted_probability_kind,
                ProbabilityRequirement.MATRIX,
            )
            out[f"{prefix}probability_kind"] = (
                semantic_descriptor.fitted_probability_kind.value
            )
            out[f"{prefix}probability_source"] = str(
                semantic_descriptor.probability_source
            )
            out[f"{prefix}probability_admitted"] = bool(admission.admitted)
            out[f"{prefix}probability_reason"] = str(admission.reason)
            if not admission.admitted:
                out[f"{prefix}skip_reason"] = str(admission.reason)
                return out
            observed = extract_probability_matrix(
                semantic_model,
                x_tr[: min(16, len(x_tr))],
                semantic_descriptor,
                requirement=ProbabilityRequirement.MATRIX,
                target_classes=np.asarray(
                    getattr(semantic_model, "classes_", classes)
                ).ravel(),
            )
            out[f"{prefix}probability_admitted"] = bool(observed.available)
            out[f"{prefix}probability_reason"] = str(observed.reason)
            if not observed.available:
                out[f"{prefix}skip_reason"] = str(observed.reason)
                return out
        else:
            observed, _ = _strict_score_matrix_for_source(
                semantic_model,
                x_tr[: min(16, len(x_tr))],
                np.asarray(getattr(semantic_model, "classes_", classes)).ravel(),
                source="predict_proba",
            )
            predicted = np.asarray(
                semantic_model.predict(x_tr[: min(16, len(x_tr))])
            ).ravel()
            observed_classes = np.asarray(
                getattr(semantic_model, "classes_", classes)
            ).ravel()
            expected = observed_classes[np.argmax(observed, axis=1)]
            hard_proxy = bool(
                np.all((observed == 0.0) | (observed == 1.0))
                and np.all(np.sum(observed == 1.0, axis=1) == 1)
                and predicted.size == expected.size
                and all(
                    _label_key(left) == _label_key(right)
                    for left, right in zip(predicted, expected)
                )
            )
            observed_kind = (
                ProbabilityKind.HARD_LABEL_PROXY
                if hard_proxy
                else ProbabilityKind.SCORE_DERIVED
            )
            admission = check_probability_requirement(
                observed_kind, ProbabilityRequirement.MATRIX
            )
            out[f"{prefix}probability_kind"] = observed_kind.value
            out[f"{prefix}probability_source"] = (
                "observed_hard_label_proxy"
                if hard_proxy
                else "validated_predict_proba_unknown_provenance"
            )
            out[f"{prefix}probability_admitted"] = bool(admission.admitted)
            out[f"{prefix}probability_reason"] = str(admission.reason)
            if not admission.admitted:
                out[f"{prefix}skip_reason"] = str(admission.reason)
                return out
    except Exception as exc:
        out[f"{prefix}probability_reason"] = f"probability_probe:{type(exc).__name__}"
        out[f"{prefix}skip_reason"] = out[f"{prefix}probability_reason"]
        return out

    try:
        from sklearn.base import clone as sk_clone
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_enc = le.fit_transform(y_tr)
        fitted_clone_candidates: List[Any] = []

        if method_key == "cross":
            # Cross-conformal: MAPIE handles internal CV.
            if structured_resampling:
                out[f"{prefix}skip_reason"] = (
                    "non_iid_internal_resampling_unsupported:mapie_cross"
                )
                return out
            n_folds = int(max(2, min(int(cv_folds), int(np.min(np.bincount(y_enc))))))
            if mapie_legacy:
                if mapie_classifier is None:
                    raise RuntimeError("mapie_legacy_api_incomplete")
                mapie_clf = mapie_classifier(
                    estimator=sk_clone(model),
                    cv=int(n_folds),
                    method="score",
                    random_state=int(seed),
                )
                mapie_clf.fit(x_tr, y_enc)
            else:
                # MAPIE >=1.0 enforces "lac" for binary targets.
                cross_score = "lac" if n_classes <= 2 else "aps"
                if cross_conformal_classifier is None:
                    raise RuntimeError("mapie_cross_api_incomplete")
                mapie_clf = cross_conformal_classifier(
                    estimator=sk_clone(model),
                    confidence_level=1.0 - float(alpha_val),
                    conformity_score=cross_score,
                    cv=int(n_folds),
                    random_state=int(seed),
                )
                mapie_clf.fit_conformalize(x_tr, y_enc)
            fitted_clone_candidates = _mapie_fitted_estimators(mapie_clf)
        else:
            # APS / RAPS: prefit on a train/calibration split.
            n_train = int(x_tr.shape[0])
            cal_size = int(max(20, round(0.25 * float(n_train))))
            cal_size = int(min(cal_size, n_train - max(2, n_classes)))
            if cal_size < n_classes or cal_size < 10:
                out[f"{prefix}skip_reason"] = "insufficient_calibration_size"
                return out

            if calibration_indices is not None:
                tr_idx = np.asarray(calibration_indices[0], dtype=int).ravel()
                cal_idx = np.asarray(calibration_indices[1], dtype=int).ravel()
                out[f"{prefix}resampling_source"] = "resolved_plan"
            else:
                from sklearn.model_selection import StratifiedShuffleSplit as SSS

                spl = SSS(
                    n_splits=1,
                    test_size=cal_size,
                    random_state=int(seed),
                )
                tr_idx, cal_idx = next(
                    spl.split(np.zeros((n_train, 1)), y_enc)
                )
                out[f"{prefix}resampling_source"] = (
                    "legacy_stratified_shuffle_split"
                )

            est = sk_clone(model)
            est.fit(x_tr[tr_idx], y_enc[tr_idx])
            fitted_clone_candidates = [est]

            if identity:
                fitted_clone_descriptor = inspect_fitted_classifier(
                    est,
                    canonical_name=identity.get("canonical_name"),
                    registry_anchor_name=identity.get("registry_anchor_name"),
                    backend=str(classifier_backend),
                    requested_name=identity.get("requested_name"),
                    outward_name=identity.get("outward_name"),
                    effective_model_name=identity.get("effective_model_name"),
                    selection_identity=identity,
                    probe_X=x_tr[tr_idx][: min(16, len(tr_idx))],
                )
                clone_matrix = extract_probability_matrix(
                    est,
                    x_tr[cal_idx][: min(16, len(cal_idx))],
                    fitted_clone_descriptor,
                    requirement=ProbabilityRequirement.MATRIX,
                    target_classes=np.asarray(est.classes_).ravel(),
                )
                if not clone_matrix.available:
                    out[f"{prefix}skip_reason"] = (
                        f"fitted_clone:{clone_matrix.reason}"
                    )
                    out[f"{prefix}probability_reason"] = str(
                        clone_matrix.reason
                    )
                    return out

            if mapie_legacy:
                if mapie_classifier is None:
                    raise RuntimeError("mapie_legacy_api_incomplete")
                mapie_method = "cumulated_score" if method_key == "aps" else "raps"
                mapie_clf = mapie_classifier(
                    estimator=est,
                    cv="prefit",
                    method=mapie_method,
                )
                mapie_clf.fit(x_tr[cal_idx], y_enc[cal_idx])
            else:
                # MAPIE >=1.0 enforces "lac" for binary targets.
                if n_classes <= 2:
                    score = "lac"
                elif method_key == "aps":
                    score = "aps"
                else:
                    if raps_conformity_score is None:
                        raise RuntimeError("mapie_raps_api_incomplete")
                    score = raps_conformity_score()
                if split_conformal_classifier is None:
                    raise RuntimeError("mapie_split_api_incomplete")
                mapie_clf = split_conformal_classifier(
                    estimator=est,
                    confidence_level=1.0 - float(alpha_val),
                    conformity_score=score,
                    prefit=True,
                    random_state=int(seed),
                )
                mapie_clf.conformalize(x_tr[cal_idx], y_enc[cal_idx])

        if not fitted_clone_candidates:
            out[f"{prefix}skip_reason"] = "fitted_clone_unavailable"
            out[f"{prefix}probability_reason"] = "fitted_clone_unavailable"
            return out
        for clone_index, fitted_estimator in enumerate(fitted_clone_candidates):
            clone_probe = x_tr[: min(16, len(x_tr))]
            clone_classes = np.asarray(
                getattr(fitted_estimator, "classes_", ())
            ).ravel()
            if identity:
                descriptor = inspect_fitted_classifier(
                    fitted_estimator,
                    canonical_name=identity.get("canonical_name"),
                    registry_anchor_name=identity.get("registry_anchor_name"),
                    backend=str(classifier_backend),
                    requested_name=identity.get("requested_name"),
                    outward_name=identity.get("outward_name"),
                    effective_model_name=identity.get("effective_model_name"),
                    selection_identity=identity,
                    probe_X=clone_probe,
                )
                clone_matrix = extract_probability_matrix(
                    fitted_estimator,
                    clone_probe,
                    descriptor,
                    requirement=ProbabilityRequirement.MATRIX,
                    target_classes=clone_classes,
                )
                if not clone_matrix.available:
                    reason = f"fitted_clone_{clone_index}:{clone_matrix.reason}"
                    out[f"{prefix}skip_reason"] = reason
                    out[f"{prefix}probability_reason"] = reason
                    return out
            else:
                try:
                    _strict_score_matrix_for_source(
                        fitted_estimator,
                        clone_probe,
                        clone_classes,
                        source="predict_proba",
                    )
                except Exception as exc:
                    record = _score_error_record(exc)
                    reason = (
                        f"fitted_clone_{clone_index}:"
                        f"{record['exception_type']}:{record['reason']}"
                    )
                    out[f"{prefix}skip_reason"] = reason
                    out[f"{prefix}probability_reason"] = reason
                    return out

        # Predict on evaluation set.
        if mapie_legacy:
            _, pred_sets = mapie_clf.predict(x_ev, alpha=float(alpha_val))
        else:
            _, pred_sets = mapie_clf.predict_set(x_ev)
        # pred_sets shape: (n_eval, n_classes, 1) — boolean mask
        if pred_sets.ndim == 3:
            pred_sets = pred_sets[:, :, 0]

        set_sizes = np.sum(pred_sets.astype(int), axis=1).astype(float)

        coverage = float("nan")
        if y_ev is not None and y_ev.size == pred_sets.shape[0]:
            y_ev_enc = le.transform(np.asarray(y_ev).ravel())
            hits = []
            for i, lab in enumerate(y_ev_enc):
                hits.append(1.0 if bool(pred_sets[i, int(lab)]) else 0.0)
            coverage = float(np.mean(hits)) if hits else float("nan")

        out[f"{prefix}applied"] = True
        out[f"{prefix}set_size_mean"] = float(np.mean(set_sizes)) if set_sizes.size else float("nan")
        out[f"{prefix}set_size_median"] = float(np.median(set_sizes)) if set_sizes.size else float("nan")
        out[f"{prefix}singleton_rate"] = (
            float(np.mean(set_sizes <= 1.0)) if set_sizes.size else float("nan")
        )
        out[f"{prefix}empty_set_rate"] = (
            float(np.mean(set_sizes == 0.0)) if set_sizes.size else float("nan")
        )
        out[f"{prefix}coverage"] = float(coverage)

        if include_prediction_sets:
            serialised_mapie: List[List[str]] = []
            inv_classes = le.classes_
            for row in pred_sets:
                included = [str(inv_classes[j]) for j in range(len(inv_classes)) if bool(row[j])]
                serialised_mapie.append(included)
            out[f"{prefix}prediction_sets"] = serialised_mapie

    except Exception as exc:
        out[f"{prefix}applied"] = False
        out[f"{prefix}skip_reason"] = str(type(exc).__name__)

    return out


def compute_conformal_singleton_rate(
    *,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: Optional[np.ndarray] = None,
    alpha: float = 0.10,
    method: str = "split",
    seed: int = 0,
) -> Dict[str, Any]:
    """Dispatch to a conformal backend and return singleton-rate diagnostics."""
    method_key = str(method or "split").strip().lower()
    if method_key not in {"split", "aps"}:
        method_key = "split"

    out: Dict[str, Any] = {
        "conformal_efficiency_method": str(method_key),
        "conformal_efficiency_applied": False,
        "conformal_efficiency_skip_reason": "",
        "conformal_singleton_rate": float("nan"),
        "conformal_set_size_mean": float("nan"),
        "conformal_set_size_median": float("nan"),
        "conformal_coverage": float("nan"),
    }

    if method_key == "aps":
        result = compute_mapie_conformal_sets(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            alpha=float(alpha),
            method="aps",
            seed=int(seed),
            include_prediction_sets=False,
        )
        applied = bool(result.get("classifier_conformal_mapie_applied", False))
        out.update(
            {
                "conformal_efficiency_applied": bool(applied),
                "conformal_efficiency_skip_reason": str(
                    result.get("classifier_conformal_mapie_skip_reason", "") or ""
                ),
                "conformal_singleton_rate": float(
                    result.get("classifier_conformal_mapie_singleton_rate", float("nan"))
                ),
                "conformal_set_size_mean": float(
                    result.get("classifier_conformal_mapie_set_size_mean", float("nan"))
                ),
                "conformal_set_size_median": float(
                    result.get("classifier_conformal_mapie_set_size_median", float("nan"))
                ),
                "conformal_coverage": float(
                    result.get("classifier_conformal_mapie_coverage", float("nan"))
                ),
            }
        )
        return out

    result = compute_split_conformal_sets(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        alpha=float(alpha),
        seed=int(seed),
        include_prediction_sets=False,
    )
    applied = bool(result.get("classifier_conformal_applied", False))
    out.update(
        {
            "conformal_efficiency_applied": bool(applied),
            "conformal_efficiency_skip_reason": str(
                result.get("classifier_conformal_skip_reason", "") or ""
            ),
            "conformal_singleton_rate": float(
                result.get("classifier_conformal_singleton_rate", float("nan"))
            ),
            "conformal_set_size_mean": float(
                result.get("classifier_conformal_set_size_mean", float("nan"))
            ),
            "conformal_set_size_median": float(
                result.get("classifier_conformal_set_size_median", float("nan"))
            ),
            "conformal_coverage": float(
                result.get("classifier_conformal_coverage", float("nan"))
            ),
        }
    )
    return out

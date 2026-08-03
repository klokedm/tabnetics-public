from __future__ import annotations

import hashlib
import json
from dataclasses import replace
import math

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from tabnetics.classification.conformance import inspect_fitted_classifier
from tabnetics.classification.conformal_risk import (
    ConformalRiskControlRequest,
    DensityRatioProvenance,
    fit_mondrian_risk_control,
)


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _wilson_lower_bound(successes: int, total: int) -> float:
    z = 1.96
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = probability + z * z / (2.0 * total)
    radius = z * math.sqrt((probability * (1.0 - probability) + z * z / (4.0 * total)) / total)
    return max(0.0, (centre - radius) / denominator)


def _request(*, groups: bool = False, ratio: DensityRatioProvenance | None = None):
    X, y = make_classification(
        n_samples=360, n_features=10, n_informative=7, n_redundant=1,
        n_classes=2, class_sep=2.0, random_state=19,
    )
    model = LogisticRegression(max_iter=4000, random_state=19).fit(X[:180], y[:180])
    descriptor = inspect_fitted_classifier(
        model, canonical_name="lr", backend="sklearn", probe_X=X[:16]
    )
    cal = slice(180, 270)
    evaluation = slice(270, None)
    kwargs = {}
    if groups:
        kwargs = {
            "group_axis": "site_ids",
            "calibration_groups": tuple("site_a" for _ in y[cal]),
            "evaluation_groups": tuple("site_a" for _ in y[evaluation]),
        }
    return ConformalRiskControlRequest(
        model=model,
        descriptor=descriptor,
        X_calibration=X[cal], y_calibration=y[cal],
        X_evaluation=X[evaluation], y_evaluation=y[evaluation],
        calibration_rows_sha256="a" * 64, evaluation_rows_sha256="b" * 64,
        context_fingerprint="c" * 64, split_fingerprint="d" * 64,
        min_stratum_size=12, threshold_grid=(0.0, 0.5, 0.75),
        target_risk=0.80, confidence_delta=0.20,
        density_ratio=ratio, **kwargs,
    )


def test_risk_control_uses_genuine_probabilities_and_emits_hash_only_strata() -> None:
    out = fit_mondrian_risk_control(_request(groups=True))
    assert out["classifier_conformal_risk_applied"] is True
    assert out["classifier_conformal_risk_guarantee_status"] == "empirical_only"
    assert out["classifier_conformal_risk_stratum_class_role"] == "predicted_class"
    assert out["classifier_conformal_risk_group_axis"] == "site_ids"
    assert 0.0 <= out["classifier_conformal_risk_abstention_rate"] <= 1.0
    assert out["classifier_conformal_risk_strata"]
    assert "site_a" not in json.dumps(out)
    assert all(len(row["class_sha256"]) == 64 for row in out["classifier_conformal_risk_strata"])
    assert all("evaluation_coverage_lcb" in row for row in out["classifier_conformal_risk_strata"])
    assert all(int(row["evaluation_count"]) > 0 for row in out["classifier_conformal_risk_strata"])
    for row in out["classifier_conformal_risk_strata"]:
        assert row["evaluation_coverage_lcb"] == _wilson_lower_bound(
            int(row["evaluation_accepted_count"]), int(row["evaluation_count"])
        )


def test_risk_control_fails_closed_for_unknown_evaluation_stratum() -> None:
    request = _request(groups=True)
    request = replace(
        request, evaluation_groups=tuple("site_b" for _ in request.y_evaluation)
    )
    out = fit_mondrian_risk_control(request)
    assert out["classifier_conformal_risk_applied"] is False
    assert out["classifier_conformal_risk_fallback_reason"] == "unknown_evaluation_stratum"


def test_risk_control_rejects_score_derived_probability_sources() -> None:
    request = _request()
    model = SVC(kernel="linear", probability=False, random_state=11).fit(
        request.X_calibration, request.y_calibration
    )
    descriptor = inspect_fitted_classifier(
        model, canonical_name="svm_linear", backend="sklearn",
        probe_X=request.X_calibration[:16],
    )
    request = replace(request, model=model, descriptor=descriptor)
    out = fit_mondrian_risk_control(request)
    assert out["classifier_conformal_risk_applied"] is False
    assert out["classifier_conformal_risk_fallback_reason"] == "genuine_probability_required"


def test_risk_control_ratio_provenance_is_empirical_only_and_digest_bound() -> None:
    ratios = tuple(1.0 for _ in range(90))
    ratio = DensityRatioProvenance(
        ratios=ratios, ratio_sha256=_sha256(list(ratios)), estimator_sha256="e" * 64,
        feature_schema_sha256="f" * 64, training_rows_sha256="1" * 64,
        calibration_rows_sha256="a" * 64, context_fingerprint="c" * 64,
        minimum_effective_sample_size=20.0,
    )
    out = fit_mondrian_risk_control(_request(ratio=ratio))
    assert out["classifier_conformal_risk_applied"] is True
    assert out["classifier_conformal_risk_guarantee_status"] == "empirical_only"
    assert out["classifier_conformal_risk_density_ratio"]["ratio_sha256"] == ratio.ratio_sha256

    invalid = replace(ratio, ratio_sha256="0" * 64)
    out = fit_mondrian_risk_control(_request(ratio=invalid))
    assert out["classifier_conformal_risk_applied"] is False
    assert out["classifier_conformal_risk_fallback_reason"] == "density_ratio_digest_mismatch"


def test_risk_control_uses_grid_and_stratum_familywise_delta() -> None:
    out = fit_mondrian_risk_control(_request(groups=True))
    assert out["classifier_conformal_risk_applied"] is True
    # The chosen lower threshold is evaluated against every threshold in the
    # predeclared grid and every predicted-class/group stratum.
    expected_delta = 0.20 / (len(out["classifier_conformal_risk_strata"]) * 3)
    for row in out["classifier_conformal_risk_strata"]:
        accepted = int(row["calibration_count"])
        risk_upper = float(row["risk_upper"])
        # The selected threshold in this fixture is 0.0, so all stratum rows
        # are accepted and the bound must include the grid multiplicity.
        assert risk_upper >= math.sqrt(math.log(1.0 / expected_delta) / (2.0 * accepted))


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("threshold_grid", (0.5, float("nan")), "invalid_threshold_grid"),
        ("min_stratum_size", 0, "invalid_min_stratum_size"),
        ("min_stratum_size", 1.5, "invalid_min_stratum_size"),
    ],
)
def test_risk_control_rejects_invalid_grid_and_stratum_size(field, value, reason: str) -> None:
    out = fit_mondrian_risk_control(replace(_request(), **{field: value}))
    assert out["classifier_conformal_risk_applied"] is False
    assert out["classifier_conformal_risk_fallback_reason"] == reason


def test_risk_control_evaluation_metrics_preserve_typed_labels() -> None:
    request = _request()
    model = DummyClassifier(strategy="constant", constant=1).fit(
        request.X_calibration, request.y_calibration
    )
    descriptor = inspect_fitted_classifier(
        model, canonical_name="lr", backend="sklearn", probe_X=request.X_calibration[:16]
    )
    request = replace(
        request,
        model=model,
        descriptor=descriptor,
        y_evaluation=np.ones_like(request.y_evaluation, dtype=bool),
    )
    out = fit_mondrian_risk_control(request)
    assert out["classifier_conformal_risk_applied"] is True
    # `True == 1` in Python, but they are different declared label categories.
    assert out["classifier_conformal_risk_selective_risk"] == 1.0

from __future__ import annotations

import numpy as np
import pytest

from tabnetics.feature_selection.mnpo.portfolio import (
    DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT,
    DIAKRINO_SELECTOR_PRIOR_CURRENT_MAX_BLEND,
    build_diakrino_selector_reference_prior,
    calibrate_diakrino_selector_prior,
)
from tabnetics.feature_selection.diakrino_trust import default_diakrino_sidecar_trust_record


def test_diakrino_selector_reference_prior_disabled_returns_base_exactly():
    base = np.array([0.2, 0.3, 0.5], dtype=float)

    out, meta = build_diakrino_selector_reference_prior(
        base,
        ["gradient_boosting", "boruta", "stability_lasso"],
        {"boruta": 1.0},
        enabled=False,
        blend_weight=1.0,
    )

    assert out is base
    assert meta["applied"] is False
    assert meta["reason"] == "disabled"


def test_diakrino_selector_reference_prior_maps_selector_pool_to_core_methods():
    base = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    names = ["gradient_boosting", "boruta", "stability_lasso"]

    out, meta = build_diakrino_selector_reference_prior(
        base,
        names,
        {"boruta": 1.0},
        enabled=True,
        blend_weight=1.0,
        calibration="none",
    )

    assert meta["applied"] is True
    assert meta["reason"] == "applied"
    assert meta["mapped_pool_weights"] == {"boruta": pytest.approx(1.0)}
    assert np.isclose(out.sum(), 1.0)
    assert out[names.index("boruta")] > out[names.index("gradient_boosting")]
    assert out[names.index("boruta")] > out[names.index("stability_lasso")]
    assert meta["candidate_prior"]["boruta"] == pytest.approx(out[names.index("boruta")])


def test_diakrino_selector_reference_prior_keeps_uniform_floor_for_unmapped_candidates():
    base = np.array([0.5, 0.5], dtype=float)
    names = ["gradient_boosting", "boruta"]

    out, meta = build_diakrino_selector_reference_prior(
        base,
        names,
        {"strict_plus_mrmr": 1.0},
        enabled=True,
        blend_weight=1.0,
        calibration="none",
    )

    assert meta["applied"] is True
    assert out[names.index("gradient_boosting")] > out[names.index("boruta")]
    assert out[names.index("boruta")] > 0.0


def test_diakrino_selector_reference_prior_no_mapped_candidates_returns_base_exactly():
    base = np.array([0.4, 0.6], dtype=float)

    out, meta = build_diakrino_selector_reference_prior(
        base,
        ["unknown_a", "unknown_b"],
        {"boruta": 1.0},
        enabled=True,
        blend_weight=1.0,
        calibration="none",
    )

    assert out is base
    assert meta["applied"] is False
    assert meta["reason"] == "no_mapped_candidates"


def test_diakrino_selector_reference_prior_zero_blend_returns_base_exactly():
    base = np.array([0.4, 0.6], dtype=float)

    out, meta = build_diakrino_selector_reference_prior(
        base,
        ["gradient_boosting", "boruta"],
        {"boruta": 1.0},
        enabled=True,
        blend_weight=0.0,
        calibration="none",
    )

    assert out is base
    assert meta["applied"] is False
    assert meta["reason"] == "zero_blend_weight"


def test_diakrino_selector_prior_current_checkpoint_calibration_shrinks_raw_weights():
    raw, meta = calibrate_diakrino_selector_prior({"boruta": 1.0})

    assert raw is not None
    assert meta["mode"] == DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT
    assert meta["applied"] is True
    assert meta["raw_weight"] == pytest.approx(0.25)
    assert meta["max_blend_weight"] == pytest.approx(DIAKRINO_SELECTOR_PRIOR_CURRENT_MAX_BLEND)
    assert raw["strict_plus_mrmr"] > raw["boruta"]
    assert raw["boruta"] > meta["anchor_weights"]["boruta"]
    assert sum(raw.values()) == pytest.approx(1.0)


def test_diakrino_selector_reference_prior_current_checkpoint_caps_effective_blend():
    base = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    names = ["gradient_boosting", "boruta", "stability_lasso"]

    out, meta = build_diakrino_selector_reference_prior(
        base,
        names,
        {"boruta": 1.0},
        enabled=True,
        blend_weight=1.0,
    )

    assert meta["applied"] is True
    assert meta["requested_blend_weight"] == pytest.approx(1.0)
    assert meta["effective_blend_weight"] == pytest.approx(DIAKRINO_SELECTOR_PRIOR_CURRENT_MAX_BLEND)
    assert meta["calibration"]["reason"] == "current_checkpoint_validation_shrinkage"
    assert np.isclose(out.sum(), 1.0)
    assert np.max(np.abs(out - base)) < 0.10


def test_diakrino_selector_prior_reads_calibration_from_trust_record():
    trust = default_diakrino_sidecar_trust_record(checkpoint_sha256="abc")
    trust["selector_prior_calibration"]["anchor_weights"] = {
        "mnpo_broad_stable": 0.0,
        "strict_plus_mrmr": 0.0,
        "boruta": 1.0,
        "copula_knockoff": 0.0,
        "stability_lasso": 0.0,
    }
    trust["selector_prior_calibration"]["raw_weight"] = 0.0
    trust["selector_prior_calibration"]["max_blend_weight"] = 0.10

    calibrated, meta = calibrate_diakrino_selector_prior(
        {"stability_lasso": 1.0},
        trust_record=trust,
    )

    assert calibrated is not None
    assert calibrated["boruta"] == pytest.approx(1.0)
    assert meta["anchor_weights"]["boruta"] == pytest.approx(1.0)
    assert meta["raw_weight"] == pytest.approx(0.0)
    assert meta["max_blend_weight"] == pytest.approx(0.10)


def test_diakrino_selector_prior_rejects_trust_record_mode_disagreement():
    trust = default_diakrino_sidecar_trust_record(checkpoint_sha256="abc")

    with pytest.raises(ValueError, match="disagrees"):
        calibrate_diakrino_selector_prior(
            {"boruta": 1.0},
            calibration="none",
            trust_record=trust,
        )

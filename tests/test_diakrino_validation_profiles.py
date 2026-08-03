from __future__ import annotations

from tabnetics.benchmarks.profiles import (
    FS_METHOD_SETS,
    DIAKRINO_REPLAY_CALIBRATION_OUTPUT_COLUMNS,
    DIAKRINO_VALIDATION_PROFILES,
    DIAKRINO_VALIDATION_PROFILE_ORDER,
    diakrino_validation_profile_inventory,
)


def test_diakrino_validation_profiles_cover_deferred_campaign_arms() -> None:
    expected = {
        "diakrino_baseline_strict_pinned_lr",
        "diakrino_fusion_w075_zscore_nn_union_pinned_lr",
        "diakrino_distribution_gates_strict_pinned_lr",
        "diakrino_selector_prior_qualified_strict_pinned_lr",
        "diakrino_svc_probability_pool_baseline",
        "diakrino_svc_probability_candidate_pool",
    }

    assert set(DIAKRINO_VALIDATION_PROFILES) == expected
    assert tuple(DIAKRINO_VALIDATION_PROFILE_ORDER) == tuple(
        profile["profile_id"] for profile in diakrino_validation_profile_inventory()
    )


def test_diakrino_validation_profiles_are_deferred_opt_in_and_reference_method_sets() -> None:
    for profile in DIAKRINO_VALIDATION_PROFILES.values():
        assert profile.execution_status == "deferred"
        assert profile.fs_method_set in FS_METHOD_SETS

    baseline = DIAKRINO_VALIDATION_PROFILES["diakrino_baseline_strict_pinned_lr"]
    assert baseline.requires_sidecar is False
    assert baseline.requires_qualification_record is False
    assert not baseline.config_toggles
    assert baseline.pinned_classifier == "lr"

    for profile_id, profile in DIAKRINO_VALIDATION_PROFILES.items():
        if profile_id in {"diakrino_baseline_strict_pinned_lr", "diakrino_svc_probability_pool_baseline"}:
            continue
        if profile_id == "diakrino_svc_probability_candidate_pool":
            assert profile.comparison_baseline == "diakrino_svc_probability_pool_baseline"
        else:
            assert profile.comparison_baseline == "diakrino_baseline_strict_pinned_lr"
        assert profile.cli_args or profile.config_toggles


def test_diakrino_fusion_profile_preserves_chunk_zscore_output_contract() -> None:
    profile = DIAKRINO_VALIDATION_PROFILES["diakrino_fusion_w075_zscore_nn_union_pinned_lr"]

    assert profile.requires_sidecar is True
    assert profile.normalization == "chunk_zscore"
    assert profile.calibration == "within_chunk_mean_std_then_global_rank01"
    assert "--probe-normalize" in profile.cli_args
    assert "chunk_zscore" in profile.cli_args
    assert "--probe-fusion-prior-weight" in profile.cli_args
    assert "0.75" in profile.cli_args
    assert "--nn-modes" in profile.cli_args
    assert "nn_union" in profile.cli_args
    assert profile.expected_output_columns == DIAKRINO_REPLAY_CALIBRATION_OUTPUT_COLUMNS
    assert "normalization_mode" in profile.expected_output_columns
    assert "calibration_mode" in profile.expected_output_columns
    assert "zscore_applied" in profile.expected_output_columns
    assert "nn_probe_zscore_applied" in profile.expected_output_columns
    assert "nn_probe_calibration_summary" in profile.expected_output_columns
    assert "nn_probe_chunk_zscore_drift_shrink_ratio" in profile.expected_output_columns
    assert "nn_probe_chunk_logit_mean_drift_json" in profile.expected_output_columns


def test_diakrino_selector_prior_profile_is_qualification_gated() -> None:
    profile = DIAKRINO_VALIDATION_PROFILES["diakrino_selector_prior_qualified_strict_pinned_lr"]

    assert profile.requires_sidecar is True
    assert profile.requires_qualification_record is True
    assert "fs_config.mnpo.oracle.use_diakrino_selector_prior" in profile.config_toggles
    assert "fs_config.mnpo.oracle.diakrino_selector_prior_qualification_record" in profile.config_toggles
    assert any("qualification_record" in artifact for artifact in profile.required_artifacts)


def test_diakrino_distribution_and_svc_profiles_name_existing_opt_in_toggles() -> None:
    distribution = DIAKRINO_VALIDATION_PROFILES["diakrino_distribution_gates_strict_pinned_lr"]
    svc_baseline = DIAKRINO_VALIDATION_PROFILES["diakrino_svc_probability_pool_baseline"]
    svc = DIAKRINO_VALIDATION_PROFILES["diakrino_svc_probability_candidate_pool"]

    assert distribution.requires_sidecar is True
    assert distribution.pinned_classifier == "lr"
    assert "dist_config.diakrino_family_prescreen_enabled" in distribution.config_toggles
    assert "dist_config.diakrino_skip_fit_discrete_enabled" in distribution.config_toggles
    assert "diakrino_cdf_trust_gate_enabled" in distribution.config_toggles
    assert "diakrino_stability_surrogate_enabled" in distribution.config_toggles
    assert "--diakrino-family-prescreen" in distribution.cli_args
    assert "--diakrino-skip-fit-discrete" in distribution.cli_args

    assert svc_baseline.requires_sidecar is False
    assert svc_baseline.config_toggles == tuple()
    assert "candidate_pool" in svc_baseline.pinned_classifier

    assert svc.requires_sidecar is False
    assert svc.config_toggles == ("model_cv_enable_svc_probability",)
    assert svc.comparison_baseline == "diakrino_svc_probability_pool_baseline"
    assert "candidate_pool" in svc.pinned_classifier
    assert "model_cv_candidate_wall_seconds" in svc.notes

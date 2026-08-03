"""S0c: every DIAKRINO native-integration config toggle defaults to a strict no-op.

Pins the default-off / no-op-value contract across all five config surfaces so a
default-constructed pipeline is byte-identical to pre-integration behaviour.
"""

from __future__ import annotations

import pytest

from tabnetics.feature_selection import FeatureSelector
from tabnetics.feature_selection.config import (
    FeatureSelectorConfig,
    MethodConfig,
    MNPOConfig,
    OracleConfig,
)
from tabnetics.pipeline.pipeline import (
    ClassificationConfig,
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
    DistributionFitterConfig,
)


def test_oracle_config_diakrino_relevance_default_off():
    oc = OracleConfig()
    assert oc.use_diakrino_relevance_oracle is False
    assert oc.diakrino_relevance_score_column == "prior_logit"  # calibration-safe surface
    assert oc.use_diakrino_selector_prior is False
    assert oc.diakrino_selector_prior_weight == 1.0
    assert oc.diakrino_selector_prior_calibration == "current_checkpoint_20260628"
    assert oc.diakrino_selector_prior_qualification_record == ""


def test_method_config_diakrino_candidate_defaults_disabled():
    mc = MethodConfig()
    assert mc.diakrino_prior_sidecar_path == ""          # "" => method skips (no sidecar)
    assert mc.diakrino_prior_score_column == "prior_logit"
    assert mc.diakrino_screening_score_column == "screening_logit"
    assert mc.diakrino_prior_top_k == 0
    assert mc.diakrino_conformal_selection_enabled is False
    assert mc.diakrino_conformal_target_fdp == pytest.approx(0.20)
    assert mc.diakrino_conformal_calibrate == "chunk_zscore"
    assert mc.diakrino_conformal_null_fraction == pytest.approx(0.50)
    assert mc.diakrino_conformal_min_null_scores == 4
    assert mc.diakrino_conformal_max_features == 0
    assert mc.diakrino_conformal_qualification_record == ""


def test_distribution_fitter_diakrino_priors_default_off():
    dfc = DistributionFitterConfig()
    assert dfc.diakrino_family_prescreen_enabled is False
    assert dfc.diakrino_skip_fit_discrete_enabled is False
    assert dfc.diakrino_warm_start_enabled is False       # §3.2 BLOCKED
    assert dfc.diakrino_sidecar_path == ""
    assert dfc.diakrino_family_prescreen_keep_mandatory is True  # never empty the set
    assert dfc.diakrino_family_prior_lambda == 0.0        # λ=0 == baseline


def test_dffs_diakrino_toggles_default_to_noop_values():
    c = DFFSConfig()
    assert c.diakrino_sidecar_path == ""
    assert c.diakrino_prefilter_enabled is False
    assert c.diakrino_prefilter_lambda == 0.0             # λ=0 == baseline
    assert c.diakrino_cdf_trust_gate_enabled is False
    assert c.diakrino_cdf_trust_entropy_threshold >= 1.0  # >=1.0 can never fire
    assert c.diakrino_cdf_trust_fallback == "rank_gaussian"
    assert c.diakrino_stability_surrogate_enabled is False
    assert c.diakrino_regime_conditional is False
    assert c.diakrino_prior_score_column == "prior_logit"
    assert c.diakrino_screening_score_column == "screening_logit"
    assert c.diakrino_prior_calibrate == "chunk_zscore"
    assert c.diakrino_prior_top_k == 0
    assert c.diakrino_conformal_selection_enabled is False
    assert c.diakrino_conformal_target_fdp == pytest.approx(0.20)
    assert c.diakrino_conformal_calibrate == "chunk_zscore"
    assert c.diakrino_conformal_null_fraction == pytest.approx(0.50)
    assert c.diakrino_conformal_min_null_scores == 4
    assert c.diakrino_conformal_max_features == 0
    assert c.diakrino_conformal_qualification_record == ""
    assert c.fs_use_diakrino_relevance_oracle is False
    assert c.fs_diakrino_relevance_score_column == "prior_logit"
    assert c.classifier_oracle_include_diakrino_family_meta is False
    assert c.diakrino_router_dispersion_descriptor_enabled is False


def test_classification_diakrino_family_meta_default_off():
    c = ClassificationConfig()
    assert c.oracle_include_diakrino_family_meta is False
    assert c.tabentics_diakrino_support_joint_serving_cache is False
    assert c.tabentics_diakrino_retry_cuda_oom_microbatch is False


def test_dffs_diakrino_calibration_modes_keep_experiment_outputs_selectable():
    for mode in (
        "chunk_ecdf",
        "chunk_minmax",
        "chunk_robust_iqr",
        "chunk_softmax_temp",
        "blend",
    ):
        assert DFFSConfig(diakrino_prior_calibrate=mode).diakrino_prior_calibrate == mode


def test_feature_selector_config_constructs_with_diakrino_fields():
    # FeatureSelectorConfig embeds MethodConfig; default construction must not raise
    # and must leave the DIAKRINO method fields disabled.
    fsc = FeatureSelectorConfig()
    assert fsc.methods.diakrino_prior_sidecar_path == ""


def test_diakrino_selector_prior_config_propagates_from_nested_oracle():
    cfg = FeatureSelectorConfig(
        mnpo=MNPOConfig(
            oracle=OracleConfig(
                use_diakrino_selector_prior=True,
                diakrino_selector_prior_weight=0.25,
                diakrino_selector_prior_calibration="none",
                diakrino_selector_prior_qualification_record="/tmp/diakrino_qual.json",
            )
        )
    )

    selector = FeatureSelector.from_config(cfg)

    assert selector.use_diakrino_selector_prior is True
    assert selector.diakrino_selector_prior_weight == pytest.approx(0.25)
    assert selector.diakrino_selector_prior_calibration == "none"
    assert selector.diakrino_selector_prior_qualification_record == "/tmp/diakrino_qual.json"
    assert selector.oracle.use_diakrino_selector_prior is True
    assert selector.oracle.diakrino_selector_prior_weight == pytest.approx(0.25)
    assert selector.oracle.diakrino_selector_prior_calibration == "none"
    assert selector.oracle.diakrino_selector_prior_qualification_record == "/tmp/diakrino_qual.json"


def test_dffs_diakrino_sidecar_bridge_reaches_legacy_feature_selector_mapping():
    cfg = DFFSConfig(
        enabled_methods=("diakrino_prior", "diakrino_screening_prior"),
        diakrino_sidecar_path="/tmp/ds.parquet",
        diakrino_prior_score_column="prior_logit",
        diakrino_screening_score_column="screening_logit",
        diakrino_prior_top_k=17,
        diakrino_conformal_selection_enabled=True,
        diakrino_conformal_target_fdp=0.15,
        diakrino_conformal_calibrate="chunk_zscore",
        diakrino_conformal_null_fraction=0.40,
        diakrino_conformal_min_null_scores=6,
        diakrino_conformal_max_features=11,
        diakrino_conformal_qualification_record="/tmp/diakrino_conformal_qual.json",
        fs_use_diakrino_relevance_oracle=True,
        fs_diakrino_relevance_min_n_train=12,
        fs_diakrino_relevance_score_column="screening_logit",
    )

    selector = DistributionFeatureSelectionPipeline(cfg)._build_feature_selector(
        seed=5,
        enabled_methods=cfg.enabled_methods,
        dataset_name="dataset_alpha__nestedcv",
    )

    assert selector.diakrino_prior_sidecar_path == "/tmp/ds.parquet"
    assert selector.diakrino_prior_score_column == "prior_logit"
    assert selector.diakrino_screening_score_column == "screening_logit"
    assert selector.diakrino_prior_top_k == 17
    assert selector.diakrino_conformal_selection_enabled is True
    assert selector.diakrino_conformal_target_fdp == pytest.approx(0.15)
    assert selector.diakrino_conformal_calibrate == "chunk_zscore"
    assert selector.diakrino_conformal_null_fraction == pytest.approx(0.40)
    assert selector.diakrino_conformal_min_null_scores == 6
    assert selector.diakrino_conformal_max_features == 11
    assert selector.diakrino_conformal_qualification_record == "/tmp/diakrino_conformal_qual.json"
    assert selector.oracle.use_diakrino_relevance_oracle is True
    assert selector.oracle.diakrino_relevance_min_n_train == 12
    assert selector.oracle.diakrino_relevance_score_column == "screening_logit"
    assert selector.diakrino_prior_dataset_id == "dataset_alpha"


def test_dffs_diakrino_sidecar_dataset_id_prefers_explicit_config():
    cfg = DFFSConfig(
        enabled_methods=("diakrino_prior",),
        diakrino_sidecar_path="/tmp/root",
        diakrino_sidecar_dataset_id="explicit_dataset",
    )

    selector = DistributionFeatureSelectionPipeline(cfg)._build_feature_selector(
        seed=5,
        enabled_methods=cfg.enabled_methods,
        dataset_name="ignored_dataset__nestedcv",
    )

    assert selector.diakrino_prior_sidecar_path == "/tmp/root"
    assert selector.diakrino_prior_dataset_id == "explicit_dataset"


def test_dffs_diakrino_sidecar_bridge_reaches_nested_feature_selector_config():
    cfg = DFFSConfig(
        enabled_methods=("diakrino_prior",),
        fs_config=FeatureSelectorConfig(),
        diakrino_sidecar_path="/tmp/nested.parquet",
        diakrino_prior_top_k=9,
        fs_use_diakrino_relevance_oracle=True,
    )

    selector = DistributionFeatureSelectionPipeline(cfg)._build_feature_selector(
        seed=5,
        enabled_methods=cfg.enabled_methods,
    )

    assert selector.diakrino_prior_sidecar_path == "/tmp/nested.parquet"
    assert selector.diakrino_prior_top_k == 9
    assert selector.oracle.use_diakrino_relevance_oracle is True

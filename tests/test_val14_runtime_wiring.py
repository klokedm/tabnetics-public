from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import tabnetics.benchmarks.runner as benchmark
from tabnetics.pipeline.pipeline import (
    ClassificationConfig,
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
)
from tabnetics.validation.generate_plan import build_jobs_validation14


def test_dffsconfig_low_p_mode_and_threshold_normalization():
    cfg = DFFSConfig(regime_gating_low_p_over_n_mode="fast_univariate_filter")
    assert cfg.regime_gating_low_p_over_n_mode == "fast_univariate_filter"
    assert float(cfg.regime_gating_low_p_over_n_threshold) == pytest.approx(0.0)

    cfg_invalid = DFFSConfig(regime_gating_low_p_over_n_mode="unknown_mode")
    assert cfg_invalid.regime_gating_low_p_over_n_mode == "fast_univariate_filter"


def test_dffsconfig_df_stage_position_normalization():
    cfg_default = DFFSConfig()
    assert cfg_default.df_stage_position == "after_fs"

    cfg = DFFSConfig(df_stage_position="after_fs")
    assert cfg.df_stage_position == "after_fs"

    cfg_alias = DFFSConfig(df_stage_position="after")
    assert cfg_alias.df_stage_position == "after_fs"

    cfg_invalid = DFFSConfig(df_stage_position="invalid")
    assert cfg_invalid.df_stage_position == "after_fs"


def test_dffsconfig_fs_cap_and_stability_threshold_normalization():
    cfg = DFFSConfig(
        fs_max_selected_features_ratio=-3.0,
        fs_max_selected_features_cap=0,
        fs_stability_threshold_method="bad_value",
    )
    assert cfg.fs_max_selected_features_ratio == pytest.approx(1e-3)
    assert cfg.fs_max_selected_features_cap == 1
    assert cfg.fs_stability_threshold_method == "fixed"


@pytest.mark.parametrize("method", ["split", "aps", "raps", "cross", "invalid"])
def test_classification_conformal_method_normalization(method: str):
    cfg = ClassificationConfig(conformal_method=method)
    if method in {"split", "aps", "raps", "cross"}:
        assert cfg.conformal_method == method
    else:
        assert cfg.conformal_method == "split"


def test_val14_parser_defaults_for_copula_and_multiclass_spc():
    args = benchmark.build_arg_parser().parse_args([
        "--datasets",
        "synthetic_medium_mixed",
    ])
    assert str(args.df_stage_position) == "after_fs"
    assert int(args.fs_copula_derandomize_runs) == 5
    assert float(args.regime_gating_extreme_multiclass_min_samples_per_class) == pytest.approx(11.0)
    assert float(args.regime_gating_low_p_over_n_threshold) == pytest.approx(0.0)
    assert str(args.classifier_conformal_method) == "split"


def test_val14_parser_accepts_new_val14_flags():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--df-stage-position",
            "after_fs",
            "--classifier-conformal-method",
            "raps",
            "--batch-label-policy",
            "kmeans2",
            "--multiomics-adapter",
            "split_halves",
            "--multiomics-integrator",
            "mb_plsda",
            "--multiomics-n-components",
            "3",
            "--fs-stability-threshold-method",
            "eats",
            "--meta-learning-selector",
            "decision_tree",
            "--meta-learning-confidence-threshold",
            "0.61",
            "--fs-fold-preference-mode",
            "logistic",
            "--fs-use-conformal-efficiency",
            "--fs-conformal-efficiency-method",
            "aps",
            "--fs-oracle-weight-js-shrinkage",
            "--fs-payoff-shrinkage-kappa",
            "0.15",
        ]
    )

    assert str(args.df_stage_position) == "after_fs"
    assert str(args.classifier_conformal_method) == "raps"
    assert str(args.batch_label_policy) == "kmeans2"
    assert str(args.multiomics_adapter) == "split_halves"
    assert str(args.multiomics_integrator) == "mb_plsda"
    assert int(args.multiomics_n_components) == 3
    assert str(args.fs_stability_threshold_method) == "eats"
    assert str(args.meta_learning_selector) == "decision_tree"
    assert float(args.meta_learning_confidence_threshold) == pytest.approx(0.61)
    assert str(args.fs_fold_preference_mode) == "logistic"
    assert bool(args.fs_use_conformal_efficiency) is True
    assert str(args.fs_conformal_efficiency_method) == "aps"
    assert bool(args.fs_oracle_weight_js_shrinkage) is True
    assert float(args.fs_payoff_shrinkage_kappa) == pytest.approx(0.15)


def test_build_base_config_maps_val14_runtime_fields():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--df-stage-position",
            "after_fs",
            "--classifier-conformal-method",
            "cross",
            "--regime-gating-low-p-over-n-mode",
            "fast_univariate_filter",
            "--fs-stability-threshold-method",
            "eats",
            "--meta-learning-selector",
            "decision_tree",
            "--meta-learning-confidence-threshold",
            "0.6",
            "--fs-fold-preference-mode",
            "logistic",
            "--fs-use-conformal-efficiency",
            "--fs-conformal-efficiency-method",
            "aps",
            "--fs-oracle-weight-js-shrinkage",
            "--fs-payoff-shrinkage-kappa",
            "0.15",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)

    assert str(cfg.df_stage_position) == "after_fs"
    assert str(cfg.classification.conformal_method) == "cross"
    assert str(cfg.regime_gating_low_p_over_n_mode) == "fast_univariate_filter"
    assert str(cfg.fs_stability_threshold_method) == "eats"
    assert str(cfg.meta_learning_selector_mode) == "decision_tree"
    assert float(cfg.meta_learning_confidence_threshold) == pytest.approx(0.6)
    assert str(cfg.fs_fold_preference_mode) == "logistic"
    assert bool(cfg.fs_use_conformal_efficiency) is True
    assert str(cfg.fs_conformal_efficiency_method) == "aps"
    assert bool(cfg.fs_oracle_weight_js_shrinkage) is True
    assert float(cfg.fs_payoff_shrinkage_kappa) == pytest.approx(0.15)


def test_pipeline_snapshot_includes_classifier_conformal_method():
    cfg = DFFSConfig(
        df_stage_position="after_fs",
        classification=ClassificationConfig(
            conformal_enabled=True,
            conformal_method="raps",
        ),
        meta_learning_selector_mode="decision_tree",
        meta_learning_confidence_threshold=0.6,
        fs_fold_preference_mode="logistic",
        fs_use_conformal_efficiency=True,
        fs_conformal_efficiency_method="aps",
        fs_oracle_weight_js_shrinkage=True,
        fs_payoff_shrinkage_kappa=0.15,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    snap = pipe._config_snapshot()
    assert str(snap.get("df_stage_position", "")) == "after_fs"
    assert str(snap.get("classifier_conformal_method", "")) == "raps"
    assert str(snap.get("meta_learning_selector_mode", "")) == "decision_tree"
    assert float(snap.get("meta_learning_confidence_threshold", 0.0)) == pytest.approx(0.6)
    assert str(snap.get("fs_fold_preference_mode", "")) == "logistic"
    assert bool(snap.get("fs_use_conformal_efficiency", False)) is True
    assert str(snap.get("fs_conformal_efficiency_method", "")) == "aps"
    assert bool(snap.get("fs_oracle_weight_js_shrinkage", False)) is True
    assert float(snap.get("fs_payoff_shrinkage_kappa", 0.0)) == pytest.approx(0.15)


def test_validation14_runtime_no_prefilter_flags_change_effective_config(tmp_path):
    jobs = build_jobs_validation14(dataset_shards=2, val13_root=Path(tmp_path))
    by_profile = defaultdict(list)
    for job in jobs:
        profile_id = str(job.job_id).split("/")[1]
        by_profile[profile_id].append(job)

    def _cfg_for(profile_id: str):
        job = by_profile[profile_id][0]
        args_list = [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-method-set",
            str(job.params.get("fs_method_set")),
            *list(job.params.get("extra_args") or []),
        ]
        args = benchmark.build_arg_parser().parse_args(args_list)
        spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
        return benchmark._build_base_config(args, spec, seed=11)

    cfg_ref = _cfg_for("v14_ref")
    cfg_no_bh = _cfg_for("v14_no_bh")
    cfg_no_var = _cfg_for("v14_no_varfloor")
    cfg_gates = _cfg_for("v14_gates_only")

    assert bool(cfg_ref.prefilter_bh_ttest_enabled) is True
    assert bool(cfg_ref.prefilter_variance_floor_enabled) is True
    assert "bh_fdr" in set(cfg_ref.prefilter_strategies)

    assert bool(cfg_no_bh.prefilter_bh_ttest_enabled) is False
    assert bool(cfg_no_bh.prefilter_variance_floor_enabled) is True
    assert "bh_fdr" not in set(cfg_no_bh.prefilter_strategies)

    assert bool(cfg_no_var.prefilter_bh_ttest_enabled) is True
    assert bool(cfg_no_var.prefilter_variance_floor_enabled) is False
    assert "bh_fdr" in set(cfg_no_var.prefilter_strategies)

    assert bool(cfg_gates.prefilter_bh_ttest_enabled) is False
    assert bool(cfg_gates.prefilter_variance_floor_enabled) is False
    assert "bh_fdr" not in set(cfg_gates.prefilter_strategies)


def test_validation14_gate3_profile_pair_diverges_on_extreme_multiclass_geometry(tmp_path):
    jobs = build_jobs_validation14(dataset_shards=2, val13_root=Path(tmp_path))
    by_profile = defaultdict(list)
    for job in jobs:
        profile_id = str(job.job_id).split("/")[1]
        by_profile[profile_id].append(job)

    def _cfg_for(profile_id: str):
        job = by_profile[profile_id][0]
        args_list = [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-method-set",
            str(job.params.get("fs_method_set")),
            *list(job.params.get("extra_args") or []),
        ]
        args = benchmark.build_arg_parser().parse_args(args_list)
        spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
        return benchmark._build_base_config(args, spec, seed=11)

    # Approximate post-split geometry for 11-class datasets in the Val-14 catalog.
    n_samples = 122
    n_classes = 11
    n_features = 7000
    y = np.arange(n_samples, dtype=int) % n_classes
    X = np.random.default_rng(11).normal(size=(n_samples, n_features))

    def _mode(cfg: DFFSConfig) -> str:
        pipe = DistributionFeatureSelectionPipeline(cfg)
        pipe._resolve_dataset_catalog_context = lambda _: {
            "dataset_id": "mock_11class",
            "display_name": "mock_11class",
            "domain": "",
            "tier": "hard",
            "is_face_domain": False,
            "found_in_catalog": True,
        }
        policy = pipe._resolve_method_policy("mock_11class", X, y)
        return str(policy.get("regime_policy_mode", ""))

    ref_mode = _mode(_cfg_for("v14_ref"))
    no_extreme_mode = _mode(_cfg_for("v14_no_extreme_multiclass"))

    assert ref_mode == "extreme_multiclass_recovery"
    assert no_extreme_mode != "extreme_multiclass_recovery"

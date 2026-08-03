"""T-DIAKRINO-IMP-01: MNPO DIAKRINO pick-frequency logging in the NN validation harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

nnfs = pytest.importorskip("run_nn_fs_tabnetics_validation")


def _entry() -> nnfs.PlanEntry:
    return nnfs.PlanEntry(
        dataset_id="ds",
        seed=11,
        role_id="probe",
        profile_id="mnpo_diakrino_probe",
        nn_mode="none",
        planned_action="run",
    )


def _pipeline_result() -> SimpleNamespace:
    return SimpleNamespace(
        balanced_accuracy=0.75,
        macro_f1=0.72,
        hybrid_score=0.73,
        accuracy=0.76,
        model_name="svm_linear",
        selected_features_count=8,
        n_features_total=100,
        n_samples_total=40,
        selected_feature_indices_original=(1, 3, 5),
        run_diagnostics={
            "pipeline_stages": {
                "feature_selection": {
                    "detailed": {
                        "mnpo_diagnostics": {
                            "candidate_names": (
                                "gradient_boosting",
                                "diakrino_prior",
                                "diakrino_screening_prior",
                            ),
                            "candidate_weights": {
                                "gradient_boosting": 0.625,
                                "diakrino_prior": 0.25,
                                "diakrino_screening_prior": 0.125,
                            },
                            "portfolio_candidates": ("diakrino_prior", "linear_svm"),
                            "portfolio_weights": {"diakrino_prior": 0.8, "linear_svm": 0.2},
                            "oracle_weights": {"diakrino_relevance": 0.4, "performance": 0.6},
                        }
                    }
                }
            }
        },
    )


def test_pipeline_result_row_logs_diakrino_mnpo_diagnostics():
    row = nnfs._result_row_from_pipeline(_entry(), _pipeline_result(), status="ran")

    assert row["diakrino_prior_weight"] == pytest.approx(0.25)
    assert row["diakrino_screening_weight"] == pytest.approx(0.125)
    assert row["diakrino_prior_in_portfolio"] is True
    assert row["diakrino_screening_prior_in_portfolio"] is False
    assert row["diakrino_prior_portfolio_weight"] == pytest.approx(0.8)
    assert row["diakrino_screening_prior_portfolio_weight"] == pytest.approx(0.0)
    assert row["diakrino_relevance_oracle_weight"] == pytest.approx(0.4)
    assert json.loads(row["mnpo_candidate_names"]) == [
        "gradient_boosting",
        "diakrino_prior",
        "diakrino_screening_prior",
    ]
    assert json.loads(row["mnpo_diagnostics"])["candidate_weights"]["diakrino_prior"] == pytest.approx(0.25)


def test_history_result_row_defaults_diakrino_mnpo_diagnostics():
    hist = nnfs.HistoryRow(
        dataset_id="ds",
        seed=11,
        profile_id="strict_plus_mrmr",
        csv_path="history.csv",
        balanced_accuracy=0.5,
        macro_f1=0.4,
        hybrid_score=0.45,
        row={
            "accuracy": 0.55,
            "model": "lr",
            "selected_features": 5,
            "n_features_total": 20,
            "n_samples_total": 30,
        },
    )
    row = nnfs._result_row_from_history(_entry(), hist)

    assert row["diakrino_prior_weight"] == 0.0
    assert row["diakrino_screening_weight"] == 0.0
    assert row["diakrino_prior_in_portfolio"] is False
    assert row["diakrino_screening_prior_in_portfolio"] is False
    assert row["diakrino_prior_portfolio_weight"] == 0.0
    assert row["diakrino_screening_prior_portfolio_weight"] == 0.0
    assert row["diakrino_relevance_oracle_weight"] == 0.0
    assert row["mnpo_candidate_names"] == ""
    assert row["mnpo_diagnostics"] == ""


def test_mnpo_diakrino_probe_profile_is_available_to_harness():
    expected = (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
        "mrmr_jmi",
        "diakrino_prior",
        "diakrino_screening_prior",
    )
    assert nnfs.FS_METHOD_SETS["mnpo_diakrino_probe"] == expected


def test_explicit_profile_ids_build_pairable_diakrino_probe_plan():
    dataset_id = next(iter(nnfs.BENCHMARK_DATASETS))

    entries = nnfs.build_plan(
        [dataset_id],
        [11],
        history_index={},
        dataset_winners={},
        tabnetics_default_reuse_profile="",
        role_ids=["none"],
        profile_ids=["strict_plus_mrmr", "mnpo_diakrino_probe"],
        nn_modes=["none"],
        disable_historical_baseline_reuse=True,
        nn_candidate_budget=1024,
        nn_prefilter_k=0,
        nn_only_k=0,
        nn_prefilter_k_values=["default"],
        nn_only_k_values=["default"],
        df_variants=["df_on"],
    )

    assert [(entry.role_id, entry.profile_id, entry.nn_mode) for entry in entries] == [
        ("profile:strict_plus_mrmr", "strict_plus_mrmr", "none"),
        ("profile:mnpo_diakrino_probe", "mnpo_diakrino_probe", "none"),
    ]
    assert all(entry.planned_action == "run" for entry in entries)


def test_diakrino_sidecar_replay_config_resolves_dataset_specific_file(tmp_path):
    root = tmp_path / "sidecars"
    sidecar = root / "feature_logits" / "ds.parquet"
    sidecar.parent.mkdir(parents=True)
    sidecar.touch()
    cfg = nnfs.DFFSConfig(enabled_methods=nnfs.FS_METHOD_SETS["mnpo_diakrino_probe"])
    args = SimpleNamespace(
        diakrino_sidecar_root=root,
        probe_logits_root=None,
        diakrino_prior_score_column="prior_logit",
        diakrino_screening_score_column="screening_logit",
        diakrino_prior_calibrate="chunk_rank01",
        diakrino_prior_top_k=13,
        use_diakrino_relevance_oracle=True,
        diakrino_relevance_min_n_train=17,
        diakrino_relevance_score_column="screening_logit",
        diakrino_regime_conditional=True,
        diakrino_oracle_complexity_conditioning=True,
    )

    out = nnfs._apply_diakrino_sidecar_replay_config(cfg, args=args, dataset_id="ds")

    assert out.diakrino_sidecar_path == str(sidecar)
    assert out.diakrino_prior_calibrate == "chunk_rank01"
    assert out.diakrino_prior_top_k == 13
    assert out.fs_use_diakrino_relevance_oracle is True
    assert out.fs_diakrino_relevance_min_n_train == 17
    assert out.fs_diakrino_relevance_score_column == "screening_logit"
    assert out.diakrino_regime_conditional is True
    assert out.fs_oracle_complexity_conditioning is True


def test_diakrino_sidecar_replay_config_uses_probe_logits_root_fallback(tmp_path):
    root = tmp_path / "feature_logits"
    sidecar = root / "ds.parquet"
    sidecar.parent.mkdir(parents=True)
    sidecar.touch()
    cfg = nnfs.DFFSConfig(enabled_methods=("diakrino_prior",))
    args = SimpleNamespace(
        diakrino_sidecar_root=None,
        probe_logits_root=root,
        diakrino_prior_score_column="prior_logit",
        diakrino_screening_score_column="screening_logit",
        diakrino_prior_calibrate="chunk_zscore",
        diakrino_prior_top_k=0,
        use_diakrino_relevance_oracle=False,
        diakrino_relevance_min_n_train=100,
        diakrino_relevance_score_column="prior_logit",
        diakrino_oracle_complexity_conditioning=False,
    )

    out = nnfs._apply_diakrino_sidecar_replay_config(cfg, args=args, dataset_id="ds")

    assert out.diakrino_sidecar_path == str(sidecar)
    assert out.fs_use_diakrino_relevance_oracle is False
    assert out.diakrino_regime_conditional is False


def test_diakrino_sidecar_replay_config_leaves_non_diakrino_profiles_untouched(tmp_path):
    cfg = nnfs.DFFSConfig(enabled_methods=nnfs.FS_METHOD_SETS["strict_plus_mrmr"])
    args = SimpleNamespace(
        diakrino_sidecar_root=tmp_path,
        probe_logits_root=None,
        diakrino_prior_score_column="screening_logit",
        diakrino_screening_score_column="prior_logit",
        diakrino_prior_calibrate="chunk_rank01",
        diakrino_prior_top_k=99,
        use_diakrino_relevance_oracle=True,
        diakrino_relevance_min_n_train=2,
        diakrino_relevance_score_column="screening_logit",
        diakrino_regime_conditional=True,
        diakrino_oracle_complexity_conditioning=True,
    )

    out = nnfs._apply_diakrino_sidecar_replay_config(cfg, args=args, dataset_id="ds")

    assert out.diakrino_sidecar_path == ""
    assert out.diakrino_prior_score_column == "prior_logit"
    assert out.fs_use_diakrino_relevance_oracle is False
    assert out.diakrino_regime_conditional is False
    assert out.fs_oracle_complexity_conditioning is False


def test_diakrino_distribution_replay_config_attaches_sidecar_to_baseline_profile(tmp_path):
    root = tmp_path / "feature_logits"
    sidecar = root / "ds.parquet"
    sidecar.parent.mkdir(parents=True)
    sidecar.touch()
    cfg = nnfs.DFFSConfig(enabled_methods=nnfs.FS_METHOD_SETS["strict_plus_mrmr"])
    args = SimpleNamespace(
        diakrino_sidecar_root=root,
        probe_logits_root=None,
        diakrino_prior_score_column="screening_logit",
        diakrino_screening_score_column="prior_logit",
        diakrino_prior_calibrate="chunk_rank01",
        diakrino_prior_top_k=99,
        use_diakrino_relevance_oracle=True,
        diakrino_relevance_min_n_train=2,
        diakrino_relevance_score_column="screening_logit",
        diakrino_regime_conditional=True,
        diakrino_oracle_complexity_conditioning=True,
        diakrino_family_prescreen=True,
        diakrino_family_prescreen_top_k=3,
        diakrino_family_prescreen_disable_mandatory=False,
        diakrino_family_prior_lambda=0.25,
        diakrino_skip_fit_discrete=True,
        diakrino_cdf_trust_gate=True,
        diakrino_cdf_trust_entropy_threshold=0.62,
        diakrino_cdf_trust_fallback="drop",
        diakrino_stability_surrogate=True,
        use_distribution_stability_weight=False,
        stability_bootstrap=7,
    )

    out = nnfs._apply_diakrino_sidecar_replay_config(cfg, args=args, dataset_id="ds")

    assert out.diakrino_sidecar_path == str(sidecar)
    assert out.dist_config.diakrino_sidecar_path == str(sidecar)
    assert out.dist_config.diakrino_family_prescreen_enabled is True
    assert out.dist_config.diakrino_family_prescreen_top_k == 3
    assert out.dist_config.diakrino_family_prescreen_keep_mandatory is True
    assert out.dist_config.diakrino_family_prior_lambda == pytest.approx(0.25)
    assert out.dist_config.diakrino_skip_fit_discrete_enabled is True
    assert out.diakrino_cdf_trust_gate_enabled is True
    assert out.diakrino_cdf_trust_entropy_threshold == pytest.approx(0.62)
    assert out.diakrino_cdf_trust_fallback == "drop"
    assert out.diakrino_stability_surrogate_enabled is True
    assert out.use_distribution_stability_weight is True
    assert out.stability_bootstrap == 7
    assert out.diakrino_prior_score_column == "prior_logit"
    assert out.fs_use_diakrino_relevance_oracle is False
    assert out.diakrino_regime_conditional is False


def test_parse_args_exposes_diakrino_regime_conditional_flag():
    args = nnfs.parse_args(["--diakrino-regime-conditional"])

    assert args.diakrino_regime_conditional is True


def test_parse_args_exposes_diakrino_distribution_scaffold_flags():
    args = nnfs.parse_args(
        [
            "--diakrino-family-prescreen",
            "--diakrino-family-prescreen-top-k",
            "3",
            "--diakrino-family-prior-lambda",
            "0.15",
            "--diakrino-skip-fit-discrete",
            "--diakrino-cdf-trust-gate",
            "--diakrino-cdf-trust-entropy-threshold",
            "0.7",
            "--diakrino-cdf-trust-fallback",
            "drop",
            "--diakrino-stability-surrogate",
            "--use-distribution-stability-weight",
            "--stability-bootstrap",
            "5",
        ]
    )

    assert args.diakrino_family_prescreen is True
    assert args.diakrino_family_prescreen_top_k == 3
    assert args.diakrino_family_prescreen_disable_mandatory is False
    assert args.diakrino_family_prior_lambda == pytest.approx(0.15)
    assert args.diakrino_skip_fit_discrete is True
    assert args.diakrino_cdf_trust_gate is True
    assert args.diakrino_cdf_trust_entropy_threshold == pytest.approx(0.7)
    assert args.diakrino_cdf_trust_fallback == "drop"
    assert args.diakrino_stability_surrogate is True
    assert args.use_distribution_stability_weight is True
    assert args.stability_bootstrap == 5

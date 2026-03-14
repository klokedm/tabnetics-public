from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.benchmarking.tabarena_benchmark import (  # noqa: E402
    LiveResultsWriter,
    _GENERAL_FULL_MODEL_CANDIDATES,
    _GENERAL_TABULAR_MODEL_CANDIDATES,
    _configure_openml_cache,
    _build_tasks,
    _build_tabarena_config,
    _monitor_parallel_progress,
    build_arg_parser,
)
from experiments.benchmarking.tabarena_datasets import (  # noqa: E402
    TABARENA_DATASETS,
    TABARENA_DATASET_SETS,
)


def test_general_full_profile_uses_post_val17_defaults_and_disables_tabpfn() -> None:
    args = build_arg_parser().parse_args([])
    spec = next(iter(TABARENA_DATASETS.values()))

    cfg = _build_tabarena_config(spec, seed=42, args=args)

    assert args.profile == "general_full"
    assert args.protocol == "openml_task"
    assert cfg.classification.model_candidates == _GENERAL_FULL_MODEL_CANDIDATES
    assert "tabpfn" not in cfg.classification.model_candidates
    assert cfg.classification.include_tabpfn_model is False
    assert cfg.classification.selection_mode == "mnpo_hybrid"
    assert cfg.classification.backend == "flaml"
    assert cfg.classification.use_hybrid_score is True
    assert cfg.classification.runtime_containment_enabled is False
    assert cfg.fs_fold_preference_mode == "logistic"
    assert cfg.fs_use_conformal_efficiency is True
    assert cfg.fs_conformal_efficiency_method == "aps"
    assert cfg.fs_oracle_weight_js_shrinkage is True
    assert cfg.fs_payoff_shrinkage_kappa == 0.15


def test_general_profile_uses_flaml_backed_mnpo_selector() -> None:
    args = build_arg_parser().parse_args(["--profile", "general"])
    spec = next(iter(TABARENA_DATASETS.values()))

    cfg = _build_tabarena_config(spec, seed=42, args=args)

    assert cfg.classification.selection_mode == "mnpo_hybrid"
    assert cfg.classification.backend == "flaml"
    assert cfg.classification.runtime_max_candidates == 10


def test_general_tabular_profile_uses_tree_weighted_pool_and_skips_hdlss() -> None:
    args = build_arg_parser().parse_args(["--profile", "general_tabular"])
    spec = next(iter(TABARENA_DATASETS.values()))

    cfg = _build_tabarena_config(spec, seed=42, args=args)

    # Tree-weighted candidate pool
    assert cfg.classification.model_candidates == _GENERAL_TABULAR_MODEL_CANDIDATES
    assert "copula_da" in cfg.classification.model_candidates
    assert "tabpfn" not in cfg.classification.model_candidates
    assert cfg.classification.include_tabpfn_model is False
    # Tree models enabled
    assert cfg.classification.include_rf_model is True
    assert cfg.classification.include_extra_tree_model is True
    assert cfg.classification.include_xgb_model is True
    assert cfg.classification.include_lgbm_model is True
    assert cfg.classification.include_catboost_model is True
    # HDLSS-oriented models disabled
    assert cfg.classification.include_gpc_model is False
    assert cfg.classification.include_pls_da_model is False
    assert cfg.classification.include_nsc_model is False
    assert cfg.classification.include_vote_ensemble_model is False
    # Combined Val-18/19/extensions review: keep general-tabular on legacy
    # selection by default; the HDLSS-specific MNPO collapse story does not
    # transfer cleanly to N>>p data.
    assert cfg.classification.selection_mode == "legacy"
    assert cfg.classification.backend == "flaml"
    assert cfg.classification.use_hybrid_score is True
    assert cfg.classification.runtime_containment_enabled is False
    # No CDF transform / screening / folding
    assert cfg.apply_cdf_transform is False
    assert cfg.screening_enabled is False
    assert cfg.folding_method == "none"
    # Val-18 evidence: BH prefilter off (P03 +0.027), conformal APS > split (C07)
    assert cfg.prefilter_bh_ttest_enabled is False
    assert cfg.classification.conformal_method == "aps"
    # Post-Val-17 FS defaults still applied
    assert cfg.fs_fold_preference_mode == "logistic"
    assert cfg.fs_use_conformal_efficiency is True
    assert cfg.fs_oracle_weighting_mode == "banzhaf"


def test_general_tabular_profile_adapts_fs_fraction_by_ratio() -> None:
    """FS fraction should be generous when sample-to-feature ratio is high."""
    from dataclasses import replace as dc_replace

    args = build_arg_parser().parse_args(["--profile", "general_tabular"])
    base_spec = next(iter(TABARENA_DATASETS.values()))

    # High ratio (N>>p): fs_fraction should be 1.0
    spec_high = dc_replace(base_spec, n_samples=10000, n_features=10)
    cfg_high = _build_tabarena_config(spec_high, seed=42, args=args)
    assert cfg_high.fs_fraction == 1.0

    # Moderate ratio: fs_fraction should be 0.90
    spec_mod = dc_replace(base_spec, n_samples=500, n_features=40)
    cfg_mod = _build_tabarena_config(spec_mod, seed=42, args=args)
    assert cfg_mod.fs_fraction == 0.90

    # Low ratio (approaching HDLSS): fs_fraction should be 0.50
    spec_low = dc_replace(base_spec, n_samples=100, n_features=200)
    cfg_low = _build_tabarena_config(spec_low, seed=42, args=args)
    assert cfg_low.fs_fraction == 0.50


def test_general_tabular_profile_probe_overrides_support_mnpo_and_keepall() -> None:
    args = build_arg_parser().parse_args(
        [
            "--profile", "general_tabular",
            "--classifier-selection-mode", "mnpo_hybrid",
            "--classifier-oracle-behavior-profile", "val18_compat",
            "--classifier-oracle-weighting-mode", "tritrust",
            "--classifier-oracle-k", "1",
            "--disable-classifier-oracle-robustness",
            "--disable-classifier-oracle-calibration",
            "--disable-classifier-oracle-james-stein",
            "--disable-classifier-oracle-hoeffding-racing",
            "--disable-classifier-oracle-bbc",
            "--df-stage-position-override", "before_fs",
            "--keep-all-features-through-fs",
        ]
    )
    spec = next(iter(TABARENA_DATASETS.values()))

    cfg = _build_tabarena_config(spec, seed=42, args=args)

    assert cfg.classification.selection_mode == "mnpo_hybrid"
    assert cfg.classification.oracle_behavior_profile == "val18_compat"
    assert cfg.classification.oracle_weighting_mode == "tritrust"
    assert cfg.classification.oracle_k == 1
    assert cfg.classification.oracle_include_robustness is False
    assert cfg.classification.oracle_include_calibration is False
    assert cfg.classification.oracle_include_james_stein is False
    assert cfg.classification.oracle_enable_hoeffding_racing is False
    assert cfg.classification.oracle_enable_bbc is False
    assert cfg.df_stage_position == "before_fs"
    assert cfg.fs_fraction == 1.0
    assert cfg.n_final_features == spec.n_features
    assert cfg.use_rank_prefilter is False
    assert cfg.prefilter_top_k is None


def test_general_tabular_probe_dataset_set_targets_collapses_and_fast_sanity_cases() -> None:
    ds = TABARENA_DATASET_SETS["general_tabular_probe"]

    assert ds == [
        "MIC",
        "hiva_agnostic",
        "anneal",
        "students_dropout_and_academic_success",
        "Marketing_Campaign",
        "Bank_Customer_Churn",
        "polish_companies_bankruptcy",
    ]


def test_classifier_oracle_accepts_all_weighting_modes() -> None:
    """ClassificationConfig and ClassifierOracle should accept tritrust, uniform, shapley, banzhaf."""
    from dataclasses import replace as dc_replace
    from tabnetics.pipeline.pipeline import ClassificationConfig

    for mode in ("tritrust", "uniform", "shapley", "banzhaf"):
        cfg = ClassificationConfig(oracle_weighting_mode=mode)
        assert cfg.oracle_weighting_mode == mode, f"ClassificationConfig rejected {mode}"

    # Unknown modes should fall back to tritrust
    cfg_bad = ClassificationConfig(oracle_weighting_mode="unknown_mode")
    assert cfg_bad.oracle_weighting_mode == "tritrust"


def test_build_tasks_applies_deterministic_global_sharding() -> None:
    args = build_arg_parser().parse_args(
        [
            "--protocol", "holdout",
            "--seeds", "11", "23",
            "--task-shard-count", "3",
            "--task-shard-index", "1",
        ]
    )
    selected = list(TABARENA_DATASETS.keys())[:4]

    shard_tasks, total_tasks = _build_tasks(selected, args)

    assert total_tasks == 8
    assert [task.task_id for task in shard_tasks] == [1, 4, 7]
    assert [task.shard_task_index for task in shard_tasks] == [0, 1, 2]
    assert all(task.task_id % 3 == 1 for task in shard_tasks)


def test_configure_openml_cache_sets_shard_local_root(tmp_path: Path) -> None:
    args = build_arg_parser().parse_args(
        ["--openml-cache-dir", str(tmp_path / "openml-cache")]
    )
    old_value = os.environ.get("OPENML_CACHE_DIR")
    try:
        cache_root = _configure_openml_cache(args)
        assert cache_root == (tmp_path / "openml-cache").resolve()
        assert cache_root.exists()
        assert os.environ["OPENML_CACHE_DIR"] == str(cache_root)
    finally:
        if old_value is None:
            os.environ.pop("OPENML_CACHE_DIR", None)
        else:
            os.environ["OPENML_CACHE_DIR"] = old_value


def test_live_results_writer_appends_rows_without_rewriting_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "tabarena_results.csv"
    writer = LiveResultsWriter(csv_path)
    writer.write_row(
        {
            "task_id": 0,
            "dataset_id": "d1",
            "problem_type": "binary",
            "metric": "roc_auc",
            "seed": 11,
            "task_protocol": "holdout",
            "task_shard_index": 0,
            "task_shard_count": 1,
            "profile": "general_full",
            "status": "ok",
            "metric_error": 0.12,
        }
    )
    writer.write_row(
        {
            "task_id": 1,
            "dataset_id": "d2",
            "problem_type": "multiclass",
            "metric": "log_loss",
            "seed": 23,
            "task_protocol": "holdout",
            "task_shard_index": 0,
            "task_shard_count": 1,
            "profile": "general_full",
            "status": "error",
            "error": "boom",
        }
    )
    writer.close()

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    df = pd.read_csv(csv_path)
    assert df["dataset_id"].tolist() == ["d1", "d2"]
    assert df["status"].tolist() == ["ok", "error"]


def test_monitor_parallel_progress_writes_heartbeat_snapshot(tmp_path: Path) -> None:
    progress_queue: queue.Queue[dict[str, object]] = queue.Queue()
    stop_event = threading.Event()
    heartbeat_path = tmp_path / "heartbeat.json"
    thread = threading.Thread(
        target=_monitor_parallel_progress,
        args=(progress_queue, stop_event, 1, 30.0, 0.0, 0.0, heartbeat_path),
        daemon=True,
    )
    thread.start()

    now = time.time()
    progress_queue.put(
        {
            "event": "task_start",
            "task_id": 0,
            "dataset_id": "demo",
            "seed": 42,
            "fold": 0,
            "ts": now,
            "pid": 1234,
        }
    )
    progress_queue.put(
        {
            "event": "task_step",
            "task_id": 0,
            "dataset_id": "demo",
            "seed": 42,
            "fold": 0,
            "stage": "dataset_load",
            "message": "loading OpenML dataset demo",
            "stage_enter_ts": now + 0.01,
            "rss_mib": 128.5,
            "details": {"openml_dataset_id": 123},
            "ts": now + 0.01,
            "pid": 1234,
        }
    )
    progress_queue.put(
        {
            "event": "task_done",
            "task_id": 0,
            "dataset_id": "demo",
            "seed": 42,
            "fold": 0,
            "status": "ok",
            "elapsed_sec": 1.5,
            "ts": now + 0.1,
        }
    )
    time.sleep(0.2)
    stop_event.set()
    thread.join(timeout=3.5)

    assert not thread.is_alive()
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["completed"] == 1
    assert payload["running"] == 0
    assert payload["total_tasks"] == 1
    assert payload["stage_counts"] == {}
    assert payload["last_completed_task"]["dataset_id"] == "demo"
    assert payload["last_completed_task"]["last_stage"] == "dataset_load"


def test_monitor_parallel_progress_tracks_active_task_stage_details(tmp_path: Path) -> None:
    progress_queue: queue.Queue[dict[str, object]] = queue.Queue()
    stop_event = threading.Event()
    heartbeat_path = tmp_path / "heartbeat.json"
    thread = threading.Thread(
        target=_monitor_parallel_progress,
        args=(progress_queue, stop_event, 2, 0.1, 0.0, 0.0, heartbeat_path),
        daemon=True,
    )
    thread.start()

    now = time.time()
    progress_queue.put(
        {
            "event": "task_start",
            "task_id": 1,
            "dataset_id": "APSFailure",
            "seed": 42,
            "fold": 3,
            "task_shard_position": 2,
            "ts": now,
            "pid": 4321,
        }
    )
    progress_queue.put(
        {
            "event": "task_step",
            "task_id": 1,
            "dataset_id": "APSFailure",
            "seed": 42,
            "fold": 3,
            "task_shard_position": 2,
            "stage": "pipeline_run",
            "message": "running official fold with n_train=1000 n_test=500",
            "stage_enter_ts": now + 0.02,
            "rss_mib": 256.0,
            "details": {"n_train": 1000, "n_test": 500},
            "ts": now + 0.02,
            "pid": 4321,
        }
    )

    time.sleep(0.2)
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload["running"] == 1
    assert payload["stage_counts"] == {"pipeline_run": 1}
    assert payload["active_tasks"][0]["dataset_id"] == "APSFailure"
    assert payload["active_tasks"][0]["current_stage"] == "pipeline_run"
    assert payload["active_tasks"][0]["details"] == {"n_train": 1000, "n_test": 500}
    assert payload["active_tasks"][0]["rss_mib"] == 256.0

    progress_queue.put(
        {
            "event": "task_done",
            "task_id": 1,
            "dataset_id": "APSFailure",
            "seed": 42,
            "fold": 3,
            "task_shard_position": 2,
            "status": "ok",
            "elapsed_sec": 3.0,
            "ts": now + 0.3,
        }
    )
    time.sleep(0.1)
    stop_event.set()
    thread.join(timeout=3.5)
    assert not thread.is_alive()

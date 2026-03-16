import json
import multiprocessing as mp
import os
import pickle
import queue
import signal
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from tabnetics.benchmarks import runner as benchmark


def _base_row(dataset_id: str, seed: int) -> dict:
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_id,
        "tier": "easy",
        "effective_tier": "easy",
        "domain": "synthetic",
        "platform": "synthetic",
        "seed": seed,
        "config": "baseline",
        "protocol": "holdout",
        "data_source": "synthetic_enhanced",
        "validation_pipeline": "",
        "validation_scenario": "",
        "n_samples_total": 100,
        "n_features_total": 50,
        "n_train": 80,
        "n_test": 20,
        "n_fs_subset": 32,
        "accuracy": 0.9,
        "balanced_accuracy": 0.9,
        "macro_f1": 0.9,
        "hybrid_score": 0.9,
        "selected_features": 10,
        "model": "lr",
        "fs_time_sec": 1.0,
        "dist_time_sec": 1.0,
        "transform_time_sec": 1.0,
        "n_dist_features_fitted": 10,
        "n_dist_features_transformed": 8,
        "n_dist_rejected": 1,
        "n_dist_skipped_unreliable": 1,
        "n_dist_skipped_block_cv": 0,
        "n_low_gof_downweighted": 0,
        "mean_dist_stability_weight": 1.0,
        "cdf_block_gating_time_sec": 0.0,
        "cdf_block_gating_budget_hit": 0,
        "cdf_block_gating_blocks_evaluated": 0,
        "cdf_block_gating_blocks_applied": 0,
        "sota_holdout_bal_acc_low": 0.7,
        "sota_holdout_bal_acc_high": 1.0,
        "sota_inflated_bal_acc_low": 0.8,
        "sota_inflated_bal_acc_high": 1.0,
        "sota_holdout_status": "within",
        "sota_inflated_status": "within",
        "sota_bal_acc_low": 0.7,
        "sota_bal_acc_high": 1.0,
        "sota_status": "within",
        "protocol_gap_note": "",
        "sanity_ok": 1,
    }


def test_parallel_task_path_emits_outputs(tmp_path, monkeypatch):
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--seeds",
            "11",
            "23",
            "--ablation-profile",
            "none",
            "--max-workers",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    captured = {"n_jobs": None}

    class FakeParallel:
        def __init__(self, n_jobs, prefer, verbose):
            captured["n_jobs"] = n_jobs

        def __call__(self, jobs):
            out = []
            for fn, fn_args, fn_kwargs in jobs:
                out.append(fn(*fn_args, **fn_kwargs))
            return out

    def fake_delayed(fn):
        def _wrapper(*fn_args, **fn_kwargs):
            return fn, fn_args, fn_kwargs

        return _wrapper

    def fake_task(ds_id, seed, _args):
        base = _base_row(ds_id, seed)
        nested = dict(base)
        nested["protocol"] = "nestedcv"
        nested["dataset_name"] = f"{ds_id} (NestedCV Audit)"
        nested["sota_holdout_status"] = "audit"
        nested["sota_inflated_status"] = "audit"
        nested["sota_status"] = "audit"
        bundle = {
            "schema_version": "1.0",
            "artifact_type": "df_fs_model_bundle",
            "dataset_id": ds_id,
            "seed": int(seed),
            "config": "baseline",
            "protocol": "holdout",
            "run_key": f"{ds_id}__seed{int(seed)}",
        }
        diag = {
            "schema_version": "1.0",
            "artifact_type": "df_fs_run_diagnostics",
            "dataset_id": ds_id,
            "seed": int(seed),
            "config": "baseline",
            "protocol": "holdout",
            "run_key": f"{ds_id}__seed{int(seed)}",
            "pipeline_stages": {"feature_selection": {}, "classifier_selection": {}},
        }
        return {
            "rows": [base, nested],
            "failures": [],
            "model_bundles": [bundle],
            "run_diagnostics": [diag],
        }

    monkeypatch.setattr(benchmark, "Parallel", FakeParallel)
    monkeypatch.setattr(benchmark, "delayed", fake_delayed)
    monkeypatch.setattr(benchmark, "_run_dataset_seed_task", fake_task)

    run_dir = benchmark.run_benchmark(args)

    runs_df = pd.read_csv(Path(run_dir) / "df_fs_runs.csv")
    metadata = json.loads((Path(run_dir) / "df_fs_metadata.json").read_text(encoding="utf-8"))
    model_bundles = json.loads((Path(run_dir) / "df_fs_model_bundles.json").read_text(encoding="utf-8"))
    run_diags = json.loads((Path(run_dir) / "df_fs_run_diagnostics.json").read_text(encoding="utf-8"))

    assert captured["n_jobs"] == 2
    assert set(runs_df["seed"].tolist()) == {11, 23}
    assert "domain" in runs_df.columns
    assert "platform" in runs_df.columns
    assert metadata["max_workers"] == 2
    assert "enable_model_cv_runtime_containment" in metadata["config_flags"]
    assert model_bundles["artifact_type"] == "df_fs_model_bundle_collection"
    assert run_diags["artifact_type"] == "df_fs_run_diagnostics_collection"
    assert "items" in model_bundles
    assert "items" in run_diags
    assert int(model_bundles.get("n_items", 0)) == 2
    assert int(run_diags.get("n_items", 0)) == 2


def test_pick_safe_nestedcv_splits_respects_min_train_per_class():
    y = [0] * 3 + [1] * 3
    n_splits, reason = benchmark._pick_safe_nestedcv_splits(
        y,
        requested_splits=5,
        min_train_per_class=2,
    )
    assert n_splits == 3
    assert reason == ""

    y_small = [0] * 2 + [1] * 2
    n_splits, reason = benchmark._pick_safe_nestedcv_splits(
        y_small,
        requested_splits=5,
        min_train_per_class=2,
    )
    assert n_splits is None
    assert "no_safe_n_splits" in reason


def test_task_timeout_context_raises_on_overrun():
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("SIGALRM not supported on this platform")

    with pytest.raises(TimeoutError):
        with benchmark._task_timeout(0.05):
            time.sleep(0.2)


def test_hard_timeout_worker_kills_overrun_process(monkeypatch):
    if "fork" not in set(mp.get_all_start_methods()):
        pytest.skip("fork start method not available")

    def slow_worker(result_queue, cfg, X, y, dataset_name, seed, quiet_worker_logs):
        time.sleep(0.2)

    monkeypatch.setattr(benchmark, "_pipeline_run_worker", slow_worker)

    X = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.5, -0.2]]
    y = [0, 1, 0, 1]
    cfg = benchmark.DFFSConfig(
        random_seed=11,
        fs_fraction=0.5,
        n_final_features=2,
        max_dist_features=2,
        prefilter_top_k=2,
        enabled_methods=("mutual_information", "anova_f"),
    )

    with pytest.raises(TimeoutError):
        benchmark._run_pipeline_with_hard_timeout(
            cfg=cfg,
            X=X,
            y=y,
            dataset_name="tiny",
            seed=11,
            timeout_sec=0.05,
            quiet_worker_logs=True,
        )


def test_hard_timeout_worker_result_queue_no_false_empty(monkeypatch):
    if "fork" not in set(mp.get_all_start_methods()):
        pytest.skip("fork start method not available")

    def fast_worker(result_queue, cfg, X, y, dataset_name, seed, quiet_worker_logs):
        result_queue.put({"ok": True, "result": {"status": "ok", "seed": int(seed)}})

    monkeypatch.setattr(benchmark, "_pipeline_run_worker", fast_worker)

    import multiprocessing.queues as mpq

    original_empty = mpq.Queue.empty

    def _fail_if_empty_called(self):
        raise AssertionError("Queue.empty() must not be used in hard-timeout payload path")

    monkeypatch.setattr(mpq.Queue, "empty", _fail_if_empty_called)

    X = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.5, -0.2]]
    y = [0, 1, 0, 1]
    cfg = benchmark.DFFSConfig(
        random_seed=11,
        fs_fraction=0.5,
        n_final_features=2,
        max_dist_features=2,
        prefilter_top_k=2,
        enabled_methods=("mutual_information", "anova_f"),
    )

    result = benchmark._run_pipeline_with_hard_timeout(
        cfg=cfg,
        X=X,
        y=y,
        dataset_name="tiny",
        seed=11,
        timeout_sec=1.0,
        quiet_worker_logs=True,
    )
    assert result["status"] == "ok"
    assert result["seed"] == 11

    # Keep local reference to avoid lint false-positive if test body changes.
    assert callable(original_empty)


def test_hard_timeout_reads_result_queue_before_join(monkeypatch):
    monkeypatch.setattr(benchmark.mp, "get_all_start_methods", lambda: ["fork"])
    monkeypatch.setattr(
        benchmark.mp,
        "current_process",
        lambda: type("P", (), {"name": "MainProcess"})(),
    )

    state = {"got_payload": False}

    class _Queue:
        def __init__(self):
            self._item = None

        def put(self, item, timeout=None):
            self._item = item

        def get(self, timeout=None):
            state["got_payload"] = True
            if self._item is None:
                raise queue.Empty()
            return self._item

        def close(self):
            return None

        def join_thread(self):
            return None

    class _Process:
        def __init__(self, target, args):
            self._target = target
            self._args = args
            self._alive = True
            self.exitcode = 0
            self.daemon = False

        def start(self):
            self._target(*self._args)

        def join(self, timeout=None):
            assert state["got_payload"], "join() called before draining result_queue payload"
            self._alive = False

        def is_alive(self):
            return bool(self._alive)

        def terminate(self):
            self._alive = False

        def kill(self):
            self._alive = False

    class _Context:
        def Queue(self, maxsize=0):
            return _Queue()

        def Process(self, target, args):
            return _Process(target, args)

    monkeypatch.setattr(benchmark.mp, "get_context", lambda method: _Context())

    def fast_worker(result_queue, cfg, X, y, dataset_name, seed, quiet_worker_logs):
        result_queue.put({"ok": True, "result": {"status": "ok"}})

    monkeypatch.setattr(benchmark, "_pipeline_run_worker", fast_worker)

    cfg = benchmark.DFFSConfig(
        random_seed=11,
        fs_fraction=0.5,
        n_final_features=2,
        max_dist_features=2,
        prefilter_top_k=2,
        enabled_methods=("mutual_information", "anova_f"),
    )
    result = benchmark._run_pipeline_with_hard_timeout(
        cfg=cfg,
        X=[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.5, -0.2]],
        y=[0, 1, 0, 1],
        dataset_name="tiny",
        seed=11,
        timeout_sec=1.0,
        quiet_worker_logs=True,
    )
    assert result["status"] == "ok"
    assert state["got_payload"] is True


def test_hard_timeout_loads_and_cleans_spilled_result_path(monkeypatch, tmp_path):
    if "fork" not in set(mp.get_all_start_methods()):
        pytest.skip("fork start method not available")

    payload_obj = {"status": "ok", "seed": 11, "metrics": {"ba": 0.82}}

    def spilled_worker(result_queue, cfg, X, y, dataset_name, seed, quiet_worker_logs):
        path = tmp_path / f"payload_{int(seed)}.pkl"
        with path.open("wb") as fh:
            pickle.dump(payload_obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        result_queue.put({"ok": True, "result_path": str(path)})

    monkeypatch.setattr(benchmark, "_pipeline_run_worker", spilled_worker)

    cfg = benchmark.DFFSConfig(
        random_seed=11,
        fs_fraction=0.5,
        n_final_features=2,
        max_dist_features=2,
        prefilter_top_k=2,
        enabled_methods=("mutual_information", "anova_f"),
    )
    payload_path = tmp_path / "payload_11.pkl"
    result = benchmark._run_pipeline_with_hard_timeout(
        cfg=cfg,
        X=[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.5, -0.2]],
        y=[0, 1, 0, 1],
        dataset_name="tiny",
        seed=11,
        timeout_sec=1.0,
        quiet_worker_logs=True,
    )
    assert result == payload_obj
    assert payload_path.exists() is False


class _FakeQueue:
    def __init__(self):
        self._items = []

    def put(self, item, timeout=None):
        self._items.append(item)

    def put_nowait(self, item):
        self._items.append(item)

    def get(self, timeout=None):
        if not self._items:
            raise queue.Empty()
        return self._items.pop(0)

    def close(self):
        return None

    def join_thread(self):
        return None


class _FakeProcess:
    def __init__(self, target, args, *, start_exc=None):
        self._target = target
        self._args = args
        self._start_exc = start_exc
        self._alive = False
        self.exitcode = 0
        self.daemon = False

    def start(self):
        if self._start_exc is not None:
            raise self._start_exc
        self._alive = True
        self._target(*self._args)
        self._alive = False
        self.exitcode = 0

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return bool(self._alive)

    def kill(self):
        self._alive = False
        self.exitcode = -9


class _FakeContext:
    def __init__(self, *, start_exc=None):
        self._start_exc = start_exc

    def Queue(self, maxsize=0):
        return _FakeQueue()

    def Process(self, target, args):
        return _FakeProcess(target, args, start_exc=self._start_exc)


def test_monitor_parallel_progress_emits_stall_watchdog_logs(monkeypatch, capsys):
    q = _FakeQueue()
    q.put(
        {
            "event": "task_start",
            "dataset_id": "synthetic_easy_dfshift",
            "seed": 11,
            "tier": "easy",
            "pid": 1234,
            "ts": time.time(),
        }
    )
    monkeypatch.setattr(
        benchmark,
        "_collect_descendant_wait_snapshot",
        lambda max_items=12: ["pid=1234 ppid=1 state=S wchan=futex_wait_queue_me cmd=loky-worker"],
    )

    stop_event = threading.Event()
    t = threading.Thread(
        target=benchmark._monitor_parallel_progress,
        args=(q, stop_event, 1, 5.0, 0.0, 0.05),
        daemon=True,
    )
    t.start()
    time.sleep(0.12)
    stop_event.set()
    t.join(timeout=2.0)

    out = capsys.readouterr().out
    assert "[stall-watchdog]" in out
    assert "no task completion" in out
    assert "futex_wait_queue_me" in out


def test_hard_timeout_prefers_spawn_inside_loky_worker(monkeypatch):
    monkeypatch.delenv("TABNETICS_HARD_TIMEOUT_START_METHOD", raising=False)
    monkeypatch.setattr(benchmark.mp, "get_all_start_methods", lambda: ["fork", "spawn"])
    monkeypatch.setattr(
        benchmark.mp,
        "current_process",
        lambda: type("P", (), {"name": "LokyProcess-5"})(),
    )
    context_calls = []

    def _get_context(method):
        context_calls.append(str(method))
        return _FakeContext()

    monkeypatch.setattr(benchmark.mp, "get_context", _get_context)

    def fast_worker(result_queue, cfg, X, y, dataset_name, seed, quiet_worker_logs):
        result_queue.put({"ok": True, "result": {"status": "ok", "seed": int(seed)}})

    monkeypatch.setattr(benchmark, "_pipeline_run_worker", fast_worker)

    cfg = benchmark.DFFSConfig(
        random_seed=11,
        fs_fraction=0.5,
        n_final_features=2,
        max_dist_features=2,
        prefilter_top_k=2,
        enabled_methods=("mutual_information", "anova_f"),
    )
    result = benchmark._run_pipeline_with_hard_timeout(
        cfg=cfg,
        X=[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.5, -0.2]],
        y=[0, 1, 0, 1],
        dataset_name="tiny",
        seed=11,
        timeout_sec=1.0,
        quiet_worker_logs=True,
    )
    assert result["status"] == "ok"
    assert result["seed"] == 11
    assert context_calls
    assert context_calls[0] == "spawn"


def test_hard_timeout_spawn_launch_failure_falls_back_to_fork(monkeypatch):
    monkeypatch.delenv("TABNETICS_HARD_TIMEOUT_START_METHOD", raising=False)
    monkeypatch.setattr(benchmark.mp, "get_all_start_methods", lambda: ["fork", "spawn"])
    monkeypatch.setattr(
        benchmark.mp,
        "current_process",
        lambda: type("P", (), {"name": "LokyProcess-7"})(),
    )
    context_calls = []

    def _get_context(method):
        m = str(method)
        context_calls.append(m)
        if m == "spawn":
            return _FakeContext(start_exc=RuntimeError("spawn launch failure"))
        if m == "fork":
            return _FakeContext()
        raise AssertionError(f"Unexpected start method: {m}")

    monkeypatch.setattr(benchmark.mp, "get_context", _get_context)

    def fast_worker(result_queue, cfg, X, y, dataset_name, seed, quiet_worker_logs):
        result_queue.put({"ok": True, "result": {"status": "ok"}})

    monkeypatch.setattr(benchmark, "_pipeline_run_worker", fast_worker)

    cfg = benchmark.DFFSConfig(
        random_seed=11,
        fs_fraction=0.5,
        n_final_features=2,
        max_dist_features=2,
        prefilter_top_k=2,
        enabled_methods=("mutual_information", "anova_f"),
    )
    with pytest.warns(RuntimeWarning, match="falling back to 'fork'"):
        result = benchmark._run_pipeline_with_hard_timeout(
            cfg=cfg,
            X=[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.5, -0.2]],
            y=[0, 1, 0, 1],
            dataset_name="tiny",
            seed=11,
            timeout_sec=1.0,
            quiet_worker_logs=True,
        )
    assert result["status"] == "ok"
    assert context_calls[:2] == ["spawn", "fork"]


def test_subspace_method_set_and_ablation_toggle_present():
    assert "mnpo_subspace_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_copula_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_tigress_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_rankagg_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_ova_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_ecoc_class_aware_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_iterative_pruning_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_joint_multiclass_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_dove_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_sparse_multinomial_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_nsc_extended" in benchmark.FS_METHOD_SETS
    assert "a22_nsc" in benchmark.FS_METHOD_SETS
    assert "mnpo_nsc_threshold_variants_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_class_pareto_extended" in benchmark.FS_METHOD_SETS
    assert "a28_class_pareto" in benchmark.FS_METHOD_SETS
    assert "mnpo_hsic_lasso_extended" in benchmark.FS_METHOD_SETS
    assert "a25_hsic_lasso" in benchmark.FS_METHOD_SETS
    assert "mnpo_slce_extended" in benchmark.FS_METHOD_SETS
    assert "a29_slce" in benchmark.FS_METHOD_SETS
    assert "mnpo_dove_sparse_multinomial_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_iterative_pruning_bounded_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_iterative_pruning_bounded_cpss_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_iterative_pruning_bounded_pareto_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_iterative_pruning_bounded_pareto_stability_extended" in benchmark.FS_METHOD_SETS
    assert "mnpo_broad_stable" in benchmark.FS_METHOD_SETS


def test_build_base_config_wires_classifier_hybrid_flags():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--seeds",
            "11",
            "--ablation-profile",
            "none",
            "--classifier-selection-mode",
            "mnpo_hybrid",
            "--classification-backend",
            "optuna",
            "--optuna-time-budget",
            "45",
            "--optuna-n-trials",
            "7",
            "--classifier-oracle-k",
            "2",
            "--classifier-oracle-weighting-mode",
            "tritrust",
            "--include-nsc-model",
            "--include-pls-da-model",
            "--include-gpc-model",
            "--include-lgbm-model",
            "--include-extra-tree-model",
            "--include-catboost-model",
            "--fs-use-conformal-uq",
            "--fs-conformal-uq-alpha",
            "0.12",
            "--fs-conformal-uq-min-folds",
            "6",
            "--enable-stage2-ratio-augmentation",
            "--stage2-ratio-max-features",
            "12",
            "--stage2-ratio-selection-method",
            "correlation",
            "--enable-classifier-conformal",
            "--classifier-conformal-alpha",
            "0.13",
            "--classifier-conformal-calibration-fraction",
            "0.30",
            "--classifier-conformal-min-calibration",
            "12",
            "--classifier-conformal-output-sets",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]
    cfg = benchmark._build_base_config(args, spec, seed=11)

    assert cfg.classification.selection_mode == "mnpo_hybrid"
    assert cfg.classification.oracle_k == 2
    assert cfg.classification.oracle_weighting_mode == "tritrust"
    assert cfg.classification.include_nsc_model is True
    assert cfg.classification.include_pls_da_model is True
    assert cfg.classification.include_gpc_model is True
    assert cfg.classification.include_lgbm_model is True
    assert cfg.classification.include_extra_tree_model is True
    assert cfg.classification.include_catboost_model is True
    assert cfg.classification.backend == "optuna"
    assert cfg.classification.optuna_time_budget == 45
    assert cfg.classification.optuna_n_trials == 7
    assert bool(cfg.fs_use_conformal_uq) is True
    assert float(cfg.fs_conformal_uq_alpha) == pytest.approx(0.12)
    assert int(cfg.fs_conformal_uq_min_folds) == 6
    assert cfg.classification.stage2_ratio_augmentation_enabled is True
    assert cfg.classification.stage2_ratio_max_features == 12
    assert cfg.classification.stage2_ratio_selection_method == "correlation"
    assert cfg.classification.conformal_enabled is True
    assert cfg.classification.conformal_alpha == pytest.approx(0.13)
    assert cfg.classification.conformal_calibration_fraction == pytest.approx(0.30)
    assert cfg.classification.conformal_min_calibration == 12
    assert cfg.classification.conformal_output_sets is True
    assert "mnpo_broad_all" in benchmark.FS_METHOD_SETS
    assert "mnpo_broad_bundle_a" in benchmark.FS_METHOD_SETS
    assert "mnpo_broad_bundle_b" in benchmark.FS_METHOD_SETS
    assert "mnpo_broad_bundle_c" in benchmark.FS_METHOD_SETS

    base = benchmark.DFFSConfig(
        enabled_methods=("subspace_stability", "linear_svm", "mutual_information", "anova_f")
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_subspace_stability" in names


def test_broad_profiles_have_expected_method_coverage():
    stable = benchmark.FS_METHOD_SETS["mnpo_broad_stable"]
    broad_all = benchmark.FS_METHOD_SETS["mnpo_broad_all"]

    assert len(stable) == 14
    assert "class_pareto_front" in stable
    assert "copula_knockoff" in stable
    assert "relieff" in stable
    assert "hsic_lasso" in stable

    assert len(broad_all) >= 25
    assert "iterative_redundancy_pruning_bounded" in broad_all
    assert "joint_auc_l1" in broad_all
    assert "ktsp" in broad_all
    assert "fcbf" in broad_all
    assert "cmim" in broad_all
    assert "treeshap" in broad_all
    assert "oaenet" in broad_all


def test_broad_bundle_overlays_apply_expected_toggles():
    parser = benchmark.build_arg_parser()
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]

    args_a = parser.parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-method-set",
            "mnpo_broad_bundle_a",
        ]
    )
    cfg_a = benchmark._build_base_config(args_a, spec, seed=11)
    assert bool(cfg_a.fs_adaptive_portfolio_sizing_enabled) is True
    assert bool(cfg_a.use_qre_smoothing) is True
    assert bool(cfg_a.use_oracle_redundancy_penalty) is True

    args_b = parser.parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-method-set",
            "mnpo_broad_bundle_b",
        ]
    )
    cfg_b = benchmark._build_base_config(args_b, spec, seed=11)
    assert bool(cfg_b.fs_wrapper_refine_enabled) is True
    assert bool(cfg_b.fs_rashomon_enabled) is True

    args_c = parser.parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-method-set",
            "mnpo_broad_bundle_c",
        ]
    )
    cfg_c = benchmark._build_base_config(args_c, spec, seed=11)
    assert bool(cfg_c.screening_enabled) is True
    assert str(cfg_c.screening_method) == "stir"


def test_shadow_evaluator_helper_concordance_table():
    rows = [
        {
            **_base_row("synthetic_easy_dfshift", 11),
            "protocol": "holdout",
            "sota_holdout_status": "within",
            "balanced_accuracy": 0.92,
            "hybrid_score": 0.91,
        },
        {
            **_base_row("synthetic_easy_dfshift", 23),
            "protocol": "holdout",
            "sota_holdout_status": "within",
            "balanced_accuracy": 0.93,
            "hybrid_score": 0.20,
        },
    ]
    shadow_df, meta = benchmark._build_shadow_evaluator_pilot(
        rows,
        frozen_dataset_ids=["synthetic_easy_dfshift"],
    )
    assert not shadow_df.empty
    assert int(meta["n_compared"]) == 2
    assert int(meta["disagreement_count"]) >= 1


def test_parallel_task_path_emits_shadow_artifact(tmp_path, monkeypatch):
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--seeds",
            "11",
            "--ablation-profile",
            "none",
            "--max-workers",
            "1",
            "--enable-shadow-evaluator",
            "--shadow-frozen-datasets",
            "synthetic_easy_dfshift",
            "--output-dir",
            str(tmp_path),
        ]
    )

    def fake_task(ds_id, seed, _args):
        base = _base_row(ds_id, seed)
        base["hybrid_score"] = 0.95
        return {"rows": [base], "failures": []}

    monkeypatch.setattr(benchmark, "_run_dataset_seed_task", fake_task)

    run_dir = benchmark.run_benchmark(args)
    metadata = json.loads((Path(run_dir) / "df_fs_metadata.json").read_text(encoding="utf-8"))
    shadow_path = Path(run_dir) / "shadow_evaluator_pilot.csv"

    assert metadata["shadow_evaluator"]["enabled"] is True
    assert shadow_path.exists()


def test_rank_aggregation_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("stability_subsample", "linear_svm", "mutual_information", "anova_f"),
        fs_rank_aggregation_mode="rra",
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_disable_rank_aggregation" in names


def test_wrapper_refine_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("stability_subsample", "linear_svm", "mutual_information", "anova_f"),
        fs_wrapper_refine_enabled=True,
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_disable_wrapper_refine" in names


def test_ova_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("ova_ensemble", "linear_svm", "mutual_information", "anova_f"),
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_ova_ensemble" in names


def test_ecoc_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("ecoc_class_aware", "linear_svm", "mutual_information", "anova_f"),
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_ecoc_class_aware" in names


def test_joint_multiclass_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("joint_multiclass_support", "linear_svm", "mutual_information", "anova_f"),
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_joint_multiclass_support" in names


def test_dove_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("dove_class_specific", "linear_svm", "mutual_information", "anova_f"),
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_dove_class_specific" in names


def test_sparse_multinomial_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("sparse_multinomial", "linear_svm", "mutual_information", "anova_f"),
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_sparse_multinomial" in names


def test_nsc_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("nearest_shrunken_centroid", "linear_svm", "mutual_information", "anova_f"),
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_nearest_shrunken_centroid" in names


def test_class_pareto_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("class_pareto_front", "linear_svm", "mutual_information", "anova_f"),
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_class_pareto_front" in names


def test_hsic_lasso_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("hsic_lasso", "linear_svm", "mutual_information", "anova_f"),
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_hsic_lasso" in names


def test_iterative_pruning_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("iterative_redundancy_pruning", "linear_svm", "mutual_information", "anova_f"),
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_iterative_redundancy_pruning" in names


def test_iterative_pruning_bounded_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("iterative_redundancy_pruning_bounded", "linear_svm", "mutual_information", "anova_f"),
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_iterative_redundancy_pruning_bounded" in names
    assert "fs_enable_iterative_pruning_bounded_cpss_overlay" in names
    assert "fs_enable_iterative_pruning_class_pareto_prefilter" in names
    assert "fs_enable_iterative_pruning_class_pareto_stability_gate" in names


def test_tigress_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("tigress_stability", "linear_svm", "mutual_information", "anova_f")
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_tigress_stability" in names


def test_copula_stabilizer_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("copula_knockoff", "linear_svm", "mutual_information", "anova_f"),
        fs_copula_stabilizer_runs=3,
        fs_copula_stabilizer_use_ebh=True,
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_disable_copula_stabilizer" in names


def test_runtime_racing_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
        fs_runtime_racing_enabled=True,
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_disable_runtime_racing" in names


def test_model_harness_flags_expand_candidates_in_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--include-elastic-net-model",
            "--include-rf-model",
            "--include-knn-model",
            "--include-rp-ensemble-model",
            "--include-xgb-model",
            "--include-tabpfn-model",
        ]
    )

    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]
    cfg = benchmark._build_base_config(args, spec, seed=11)

    assert cfg.model_candidates == ("lr", "svm_rbf", "elastic_net_lr", "rf", "knn", "rp_ensemble", "xgb", "tabpfn")
    assert cfg.include_elastic_net_model is True
    assert cfg.include_rf_model is True
    assert cfg.include_knn_model is True
    assert cfg.include_rp_ensemble_model is True
    assert cfg.include_xgb_model is True
    assert cfg.include_tabpfn_model is True


def test_benchmark_cli_defaults_are_strict():
    args = benchmark.build_arg_parser().parse_args(["--datasets", "synthetic_easy_dfshift"])
    assert args.allow_synthetic_fallback is False
    assert str(args.dataset_integrity_policy) == "error"


def test_benchmark_requires_hf_bundle_for_validation_catalog(monkeypatch, tmp_path):
    monkeypatch.delenv("TABNETICS_HF_ORG", raising=False)
    monkeypatch.delenv("TABNETICS_HF_REPO_ID", raising=False)

    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "cumida_brain_gse50161",
            "--seeds",
            "11",
            "--max-workers",
            "1",
            "--ablation-profile",
            "none",
            "--output-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(RuntimeError, match="HuggingFace bundle"):
        benchmark.run_benchmark(args)


def test_df_multimodal_defaults_are_enabled_in_benchmark_base_config():
    args = benchmark.build_arg_parser().parse_args(["--datasets", "synthetic_easy_dfshift"])
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]
    cfg = benchmark._build_base_config(args, spec, seed=11)

    assert bool(args.df_compute_dip) is True
    assert str(args.df_multimodal_fallback) == "gmm"
    assert bool(cfg.dist_config.compute_dip) is True
    assert str(cfg.multimodal_fallback) == "gmm"


def test_df_multimodal_flags_allow_opt_out_and_rank_fallback():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--no-df-compute-dip",
            "--df-multimodal-fallback",
            "rank_transform",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]
    cfg = benchmark._build_base_config(args, spec, seed=11)

    assert bool(args.df_compute_dip) is False
    assert bool(cfg.dist_config.compute_dip) is False
    assert str(cfg.multimodal_fallback) == "rank_transform"


def test_clone_config_preserves_df_multimodal_settings():
    base = benchmark.DFFSConfig(
        multimodal_fallback="rank_transform",
        dist_config=benchmark.DistributionFitterConfig(compute_dip=False),
    )
    cloned = benchmark.clone_config(base)
    assert bool(cloned.dist_config.compute_dip) is False
    assert str(cloned.multimodal_fallback) == "rank_transform"


@pytest.mark.skipif(
    not os.environ.get("TABNETICS_HF_ORG") or not os.environ.get("HF_TOKEN"),
    reason="TABNETICS_HF_ORG and HF_TOKEN must be set for integration tests",
)
def test_cumida_brain_succeeds_when_synthetic_fallback_disabled():
    """Integration test: verify that a real dataset loads successfully when synthetic fallback is disabled."""
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "cumida_brain_gse50161",
            "--seeds",
            "11",
            "--ablation-profile",
            "none",
        ]
    )

    result = benchmark._run_dataset_seed_task("cumida_brain_gse50161", 11, args)
    assert result["rows"] != []
    assert result["failures"] == []


def test_model_candidates_cli_override_controls_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--model-candidates",
            "xgb",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]
    cfg = benchmark._build_base_config(args, spec, seed=11)

    assert cfg.model_candidates == ("xgb",)
    assert cfg.include_xgb_model is True


def test_model_candidate_profile_a11_controls_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--model-candidate-profile",
            "a11_medium_mismatch",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]
    cfg = benchmark._build_base_config(args, spec, seed=11)

    assert cfg.model_candidates == ("lr", "svm_rbf", "svm_linear", "dlda", "knn", "nb", "vote_ensemble")
    assert cfg.include_svm_linear_model is True
    assert cfg.include_dlda_model is True
    assert cfg.include_knn_model is True
    assert cfg.include_nb_model is True
    assert cfg.include_vote_ensemble_model is True
    assert cfg.include_rf_model is False


def test_model_cv_runtime_containment_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--enable-model-cv-runtime-containment",
            "--model-cv-runtime-max-candidates",
            "3",
            "--model-cv-runtime-high-p-over-n-threshold",
            "25",
            "--model-cv-runtime-high-class-threshold",
            "5",
            "--model-cv-runtime-min-class-count-threshold",
            "10",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]
    cfg = benchmark._build_base_config(args, spec, seed=11)

    assert cfg.model_cv_runtime_containment_enabled is True
    assert cfg.model_cv_runtime_max_candidates == 3
    assert cfg.model_cv_runtime_high_p_over_n_threshold == 25.0
    assert cfg.model_cv_runtime_high_class_threshold == 5
    assert cfg.model_cv_runtime_min_class_count_threshold == 10


def test_cumida_brain_benchmark_metadata_is_promotable():
    """CuMiDa brain is real-data; benchmark promotion metadata should not mark it fallback-only."""
    spec = benchmark.BENCHMARK_DATASETS["cumida_brain_gse50161"]
    meta = benchmark._benchmark_dataset_promotion_metadata(spec)

    assert int(meta["promotion_eligible"]) == 1
    assert str(meta["promotion_blocker"]) == ""
    assert str(meta["source_policy"]) in {"standard", "real_only"}


def test_wmw_auc_ablation_toggle_present():
    base = benchmark.DFFSConfig(
        enabled_methods=("wmw_auc", "linear_svm", "mutual_information", "anova_f"),
    )
    cfgs = benchmark._build_ablation_configs(base, profile="core")
    names = [name for name, _ in cfgs]

    assert "fs_no_wmw_auc" in names


def test_maqc_pairing_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--enable-maqc-pairing",
            "--maqc-fs-method-sets",
            "strict_plus_mrmr",
            "mnpo_ova_extended",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]
    cfg = benchmark._build_base_config(args, spec, seed=11)

    assert cfg.enable_maqc_pairing is True
    assert cfg.maqc_pairing_method_set_names == ("strict_plus_mrmr", "mnpo_ova_extended")
    assert cfg.maqc_pairing_method_sets[0] == benchmark.FS_METHOD_SETS["strict_plus_mrmr"]
    assert cfg.maqc_pairing_method_sets[1] == benchmark.FS_METHOD_SETS["mnpo_ova_extended"]


def test_ova_linear_backend_flag_populates_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-ova-linear-backend",
            "elastic_net_lr",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert cfg.fs_ova_linear_backend == "elastic_net_lr"


def test_ova_min_classes_flag_populates_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-ova-min-classes",
            "5",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert int(cfg.fs_ova_min_classes) == 5


def test_ecoc_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-ecoc-min-classes",
            "5",
            "--fs-ecoc-max-ovo-pairs",
            "9",
            "--fs-ecoc-random-code-bits",
            "3",
            "--fs-ecoc-class-complexity-weight",
            "1.3",
            "--fs-ecoc-negative-ratio",
            "1.6",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert int(cfg.fs_ecoc_min_classes) == 5
    assert int(cfg.fs_ecoc_max_ovo_pairs) == 9
    assert int(cfg.fs_ecoc_random_code_bits) == 3
    assert float(cfg.fs_ecoc_class_complexity_weight) == pytest.approx(1.3)
    assert float(cfg.fs_ecoc_negative_ratio) == pytest.approx(1.6)


def test_joint_multiclass_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-joint-multiclass-min-classes",
            "4",
            "--fs-joint-multiclass-max-features",
            "180",
            "--fs-joint-multiclass-path-grid-size",
            "5",
            "--fs-joint-multiclass-min-c",
            "0.07",
            "--fs-joint-multiclass-max-c",
            "1.5",
            "--fs-joint-multiclass-l1-ratio",
            "0.6",
            "--fs-joint-multiclass-univariate-blend",
            "0.3",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert int(cfg.fs_joint_multiclass_min_classes) == 4
    assert int(cfg.fs_joint_multiclass_max_features) == 180
    assert int(cfg.fs_joint_multiclass_path_grid_size) == 5
    assert float(cfg.fs_joint_multiclass_min_c) == pytest.approx(0.07)
    assert float(cfg.fs_joint_multiclass_max_c) == pytest.approx(1.5)
    assert float(cfg.fs_joint_multiclass_l1_ratio) == pytest.approx(0.6)
    assert float(cfg.fs_joint_multiclass_univariate_blend) == pytest.approx(0.3)


def test_dove_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-dove-min-classes",
            "4",
            "--fs-dove-max-pairs-per-class",
            "5",
            "--fs-dove-path-grid-size",
            "6",
            "--fs-dove-specificity-weight",
            "0.4",
            "--fs-dove-minority-boost",
            "0.7",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert int(cfg.fs_dove_min_classes) == 4
    assert int(cfg.fs_dove_max_pairs_per_class) == 5
    assert int(cfg.fs_dove_path_grid_size) == 6
    assert float(cfg.fs_dove_specificity_weight) == pytest.approx(0.4)
    assert float(cfg.fs_dove_minority_boost) == pytest.approx(0.7)


def test_sparse_multinomial_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-sparse-multinomial-min-classes",
            "4",
            "--fs-sparse-multinomial-max-features",
            "180",
            "--fs-sparse-multinomial-path-grid-size",
            "5",
            "--fs-sparse-multinomial-min-c",
            "0.07",
            "--fs-sparse-multinomial-max-c",
            "1.4",
            "--fs-sparse-multinomial-backend",
            "elasticnet",
            "--fs-sparse-multinomial-l1-ratio",
            "0.6",
            "--fs-sparse-multinomial-univariate-blend",
            "0.3",
            "--fs-sparse-multinomial-max-iter",
            "4000",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert int(cfg.fs_sparse_multinomial_min_classes) == 4
    assert int(cfg.fs_sparse_multinomial_max_features) == 180
    assert int(cfg.fs_sparse_multinomial_path_grid_size) == 5
    assert float(cfg.fs_sparse_multinomial_min_c) == pytest.approx(0.07)
    assert float(cfg.fs_sparse_multinomial_max_c) == pytest.approx(1.4)
    assert str(cfg.fs_sparse_multinomial_backend) == "elasticnet"
    assert float(cfg.fs_sparse_multinomial_l1_ratio) == pytest.approx(0.6)
    assert float(cfg.fs_sparse_multinomial_univariate_blend) == pytest.approx(0.3)
    assert int(cfg.fs_sparse_multinomial_max_iter) == 4000


def test_nsc_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-nsc-shrinkage-grid-size",
            "8",
            "--fs-nsc-min-classes",
            "4",
            "--fs-nsc-thresholding-mode",
            "auto",
            "--fs-nsc-order-quantile",
            "0.8",
            "--enable-fs-nsc-deep-shrinkage-search",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert int(cfg.fs_nsc_shrinkage_grid_size) == 8
    assert int(cfg.fs_nsc_min_classes) == 4
    assert str(cfg.fs_nsc_thresholding_mode) == "auto"
    assert float(cfg.fs_nsc_order_quantile) == pytest.approx(0.8)
    assert bool(cfg.fs_nsc_deep_shrinkage_search) is True


def test_class_pareto_and_hsic_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-class-pareto-min-classes",
            "4",
            "--fs-class-pareto-top-per-class",
            "36",
            "--fs-class-pareto-global-fraction",
            "0.3",
            "--fs-class-pareto-minority-boost",
            "0.7",
            "--fs-class-pareto-kw-weight",
            "0.2",
            "--fs-hsic-lasso-alpha",
            "0.02",
            "--fs-hsic-lasso-prefilter-max-features",
            "96",
            "--fs-hsic-lasso-feature-sigma",
            "0.5",
            "--fs-hsic-lasso-target-sigma",
            "0.8",
            "--fs-hsic-lasso-relevance-blend",
            "0.3",
            "--fs-hsic-lasso-max-iter",
            "3000",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert int(cfg.fs_class_pareto_min_classes) == 4
    assert int(cfg.fs_class_pareto_top_per_class) == 36
    assert float(cfg.fs_class_pareto_global_fraction) == pytest.approx(0.3)
    assert float(cfg.fs_class_pareto_minority_boost) == pytest.approx(0.7)
    assert float(cfg.fs_class_pareto_kw_weight) == pytest.approx(0.2)
    assert float(cfg.fs_hsic_lasso_alpha) == pytest.approx(0.02)
    assert int(cfg.fs_hsic_lasso_prefilter_max_features) == 96
    assert float(cfg.fs_hsic_lasso_feature_sigma) == pytest.approx(0.5)
    assert float(cfg.fs_hsic_lasso_target_sigma) == pytest.approx(0.8)
    assert float(cfg.fs_hsic_lasso_relevance_blend) == pytest.approx(0.3)
    assert int(cfg.fs_hsic_lasso_max_iter) == 3000


def test_folding_and_face_projection_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--folding-method",
            "tensor_sketch",
            "--folding-n-components",
            "256",
            "--folding-rff-gamma",
            "0.25",
            "--folding-prefilter-k",
            "180",
            "--enable-face-domain-projection",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert str(cfg.folding_method) == "tensor_sketch"
    assert int(cfg.folding_n_components) == 256
    assert float(cfg.folding_rff_gamma) == pytest.approx(0.25)
    assert int(cfg.folding_prefilter_k) == 180
    assert bool(cfg.enable_face_domain_projection) is True


def test_runtime_racing_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--enable-fs-runtime-racing",
            "--fs-runtime-racing-proxy-splits",
            "2",
            "--fs-runtime-racing-keep-fraction",
            "0.5",
            "--fs-runtime-racing-min-candidates",
            "3",
            "--fs-runtime-racing-runtime-weight",
            "0.2",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert bool(cfg.fs_runtime_racing_enabled) is True
    assert int(cfg.fs_runtime_racing_proxy_splits) == 2
    assert float(cfg.fs_runtime_racing_keep_fraction) == pytest.approx(0.5)
    assert int(cfg.fs_runtime_racing_min_candidates) == 3
    assert float(cfg.fs_runtime_racing_runtime_weight) == pytest.approx(0.2)


def test_runtime_racing_op13_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--enable-fs-runtime-racing",
            "--fs-runtime-racing-mode",
            "successive_halving",
            "--fs-runtime-racing-stages",
            "3",
            "--fs-runtime-racing-confidence-bound",
            "bernstein",
            "--fs-runtime-racing-delta",
            "0.12",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert bool(cfg.fs_runtime_racing_enabled) is True
    assert str(cfg.fs_runtime_racing_mode) == "successive_halving"
    assert int(cfg.fs_runtime_racing_stages) == 3
    assert str(cfg.fs_runtime_racing_confidence_bound) == "bernstein"
    assert float(cfg.fs_runtime_racing_delta) == pytest.approx(0.12)


def test_pls_ova_sparse_quota_and_new_model_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--folding-method",
            "pls_da",
            "--folding-pls-components",
            "20",
            "--disable-folding-pls-scale",
            "--enable-fs-ova-calibration",
            "--fs-ova-calibration-cv",
            "4",
            "--fs-sparse-multinomial-screening-mode",
            "prefilter_aggressive",
            "--fs-sparse-multinomial-screening-keep-fraction",
            "0.6",
            "--fs-sparse-multinomial-screening-min-features",
            "72",
            "--disable-fs-sparse-multinomial-screening-fallback-on-failure",
            "--enable-fs-per-class-quota",
            "--fs-per-class-quota-min-per-class",
            "2",
            "--fs-per-class-quota-max-fraction",
            "0.5",
            "--include-nb-model",
            "--include-vote-ensemble-model",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert str(cfg.folding_method) == "pls_da"
    assert int(cfg.folding_pls_components) == 20
    assert bool(cfg.folding_pls_scale) is False
    assert bool(cfg.fs_ova_enable_calibration) is True
    assert int(cfg.fs_ova_calibration_cv) == 4
    assert str(cfg.fs_sparse_multinomial_screening_mode) == "prefilter_aggressive"
    assert float(cfg.fs_sparse_multinomial_screening_keep_fraction) == pytest.approx(0.6)
    assert int(cfg.fs_sparse_multinomial_screening_min_features) == 72
    assert bool(cfg.fs_sparse_multinomial_screening_fallback_on_failure) is False
    assert bool(cfg.fs_per_class_quota_enabled) is True
    assert int(cfg.fs_per_class_quota_min_per_class) == 2
    assert float(cfg.fs_per_class_quota_max_fraction) == pytest.approx(0.5)
    assert bool(cfg.include_nb_model) is True
    assert bool(cfg.include_vote_ensemble_model) is True


def test_tier_lockout_and_routing_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--tier-lockout-enabled",
            "--tier-lockout-tier",
            "easy",
            "--tier-lockout-difficulty-source",
            "historical",
            "--tier-lockout-fallback-fs-method-set",
            "strict_plus_mrmr",
            "--tier-routing-enabled",
            "--tier-routing-difficulty-classifier",
            "meta_features",
            "--tier-routing-table",
            "easy=strict_plus_mrmr;hard=linear_svm,anova_f",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert bool(cfg.tier_lockout_enabled) is True
    assert str(cfg.tier_lockout_tier) == "easy"
    assert str(cfg.tier_lockout_difficulty_source) == "historical"
    assert tuple(cfg.tier_lockout_fallback_methods) == tuple(benchmark.FS_METHOD_SETS["strict_plus_mrmr"])
    assert bool(cfg.tier_routing_enabled) is True
    assert str(cfg.tier_routing_difficulty_classifier) == "meta_features"
    assert tuple(cfg.tier_routing_table["easy"]) == tuple(benchmark.FS_METHOD_SETS["strict_plus_mrmr"])
    assert tuple(cfg.tier_routing_table["hard"]) == ("linear_svm", "anova_f")


def test_regime_gating_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--regime-gating-enabled",
            "--regime-gating-difficulty-source",
            "historical",
            "--regime-gating-target-tier",
            "very_hard",
            "--regime-gating-min-samples-per-class",
            "15",
            "--regime-gating-low-p-over-n-threshold",
            "2.0",
            "--regime-gating-simple-fs-method-set",
            "strict_plus_mrmr",
            "--regime-gating-very-hard-portfolio-max-methods",
            "4",
            "--regime-gating-very-hard-copula-derandomize-runs",
            "5",
            "--regime-gating-low-p-over-n-mode",
            "all_features",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert bool(cfg.regime_gating_enabled) is True
    assert str(cfg.regime_gating_difficulty_source) == "historical"
    assert str(cfg.regime_gating_target_tier) == "very_hard"
    assert float(cfg.regime_gating_min_samples_per_class) == pytest.approx(15.0)
    assert float(cfg.regime_gating_low_p_over_n_threshold) == pytest.approx(2.0)
    assert tuple(cfg.regime_gating_simple_methods) == tuple(benchmark.FS_METHOD_SETS["strict_plus_mrmr"])
    assert int(cfg.regime_gating_very_hard_portfolio_max_methods) == 4
    assert int(cfg.regime_gating_very_hard_copula_derandomize_runs) == 5
    assert str(cfg.regime_gating_low_p_over_n_mode) == "all_features"


def test_importance_uq_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-importance-uq-enabled",
            "--fs-importance-uq-min-cv-folds",
            "5",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert bool(cfg.fs_importance_uq_enabled) is True
    assert int(cfg.fs_importance_uq_min_cv_folds) == 5


def test_copula_deepdrk_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-copula-generator",
            "deepdrk",
            "--fs-copula-deepdrk-latent-fraction",
            "0.4",
            "--fs-copula-deepdrk-noise-scale",
            "1.2",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert str(cfg.fs_copula_generator) == "deepdrk"
    assert float(cfg.fs_copula_deepdrk_latent_fraction) == pytest.approx(0.4)
    assert float(cfg.fs_copula_deepdrk_noise_scale) == pytest.approx(1.2)


def test_adaptive_portfolio_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-portfolio-size",
            "6",
            "--enable-fs-adaptive-portfolio-sizing",
            "--fs-adaptive-size-min",
            "4",
            "--fs-adaptive-size-max",
            "8",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert int(cfg.fs_portfolio_size) == 6
    assert bool(cfg.fs_adaptive_portfolio_sizing_enabled) is True
    assert int(cfg.fs_adaptive_size_min) == 4
    assert int(cfg.fs_adaptive_size_max) == 8


def test_val7_shapley_and_adaptive_penalty_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--enable-fs-adaptive-portfolio-sizing",
            "--adaptive-sizing-variance-penalty",
            "--adaptive-sizing-variance-penalty-strength",
            "0.7",
            "--fs-oracle-weighting-mode",
            "shapley",
            "--shapley-bayesian-shrinkage",
            "--shapley-bayesian-prior-strength",
            "9.5",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert bool(cfg.fs_adaptive_portfolio_sizing_enabled) is True
    assert bool(cfg.fs_adaptive_sizing_variance_penalty) is True
    assert float(cfg.fs_adaptive_sizing_variance_penalty_strength) == pytest.approx(0.7)
    assert str(cfg.fs_oracle_weighting_mode) == "shapley"
    assert bool(cfg.fs_shapley_bayesian_shrinkage) is True
    assert float(cfg.fs_shapley_bayesian_prior_strength) == pytest.approx(9.5)


def test_fs_portfolio_size_default_is_six():
    args = benchmark.build_arg_parser().parse_args(["--datasets", "synthetic_medium_mixed"])
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert int(cfg.fs_portfolio_size) == 6


def test_adaptive_portfolio_defaults_to_plus_minus_two_when_enabled_without_bounds():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-portfolio-size",
            "7",
            "--enable-fs-adaptive-portfolio-sizing",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert bool(cfg.fs_adaptive_portfolio_sizing_enabled) is True
    assert int(cfg.fs_adaptive_size_min) == 5
    assert int(cfg.fs_adaptive_size_max) == 9


def test_rashomon_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--enable-fs-rashomon",
            "--fs-rashomon-max-models",
            "14",
            "--fs-rashomon-score-tolerance",
            "0.02",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert bool(cfg.fs_rashomon_enabled) is True
    assert int(cfg.fs_rashomon_max_models) == 14
    assert float(cfg.fs_rashomon_score_tolerance) == pytest.approx(0.02)


def test_iterative_pruning_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-iterative-pruning-pool-factor",
            "2.2",
            "--fs-iterative-pruning-max-rounds",
            "15",
            "--fs-iterative-pruning-min-improvement",
            "-0.01",
            "--fs-iterative-pruning-max-cumulative-loss",
            "0.015",
            "--fs-iterative-pruning-redundancy-weight",
            "0.7",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert float(cfg.fs_iterative_pruning_pool_factor) == pytest.approx(2.2)
    assert int(cfg.fs_iterative_pruning_max_rounds) == 15
    assert float(cfg.fs_iterative_pruning_min_improvement) == pytest.approx(-0.01)
    assert float(cfg.fs_iterative_pruning_max_cumulative_loss) == pytest.approx(0.015)
    assert float(cfg.fs_iterative_pruning_redundancy_weight) == pytest.approx(0.7)


def test_iterative_pruning_bounded_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-iterative-pruning-bounded-prefilter-cap",
            "180",
            "--fs-iterative-pruning-bounded-candidate-fraction",
            "0.28",
            "--fs-iterative-pruning-bounded-min-candidates",
            "3",
            "--fs-iterative-pruning-bounded-max-evaluations",
            "16",
            "--fs-iterative-pruning-bounded-max-runtime-seconds",
            "12.0",
            "--fs-iterative-pruning-bounded-multiclass-scale",
            "0.6",
            "--fs-iterative-pruning-bounded-imbalance-trigger",
            "2.1",
            "--fs-iterative-pruning-bounded-imbalance-scale",
            "0.68",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert int(cfg.fs_iterative_pruning_bounded_prefilter_cap) == 180
    assert float(cfg.fs_iterative_pruning_bounded_candidate_fraction) == pytest.approx(0.28)
    assert int(cfg.fs_iterative_pruning_bounded_min_candidates) == 3
    assert int(cfg.fs_iterative_pruning_bounded_max_evaluations) == 16
    assert float(cfg.fs_iterative_pruning_bounded_max_runtime_seconds) == pytest.approx(12.0)
    assert float(cfg.fs_iterative_pruning_bounded_multiclass_scale) == pytest.approx(0.6)
    assert float(cfg.fs_iterative_pruning_bounded_imbalance_trigger) == pytest.approx(2.1)
    assert float(cfg.fs_iterative_pruning_bounded_imbalance_scale) == pytest.approx(0.68)
    assert bool(cfg.fs_iterative_pruning_bounded_enable_class_gating) is True


def test_iterative_pruning_bounded_cpss_and_pareto_flags_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--enable-fs-iterative-pruning-bounded-cpss-overlay",
            "--fs-iterative-pruning-bounded-cpss-pairs",
            "3",
            "--fs-iterative-pruning-bounded-cpss-stability-threshold",
            "0.58",
            "--fs-iterative-pruning-bounded-cpss-min-stable-features",
            "2",
            "--fs-iterative-pruning-bounded-cpss-min-jaccard",
            "0.22",
            "--fs-iterative-pruning-bounded-cpss-max-score-drop",
            "0.015",
            "--enable-fs-iterative-pruning-class-pareto-prefilter",
            "--fs-iterative-pruning-class-pareto-min-classes",
            "3",
            "--fs-iterative-pruning-class-pareto-top-per-class",
            "20",
            "--fs-iterative-pruning-class-pareto-global-fraction",
            "0.33",
            "--fs-iterative-pruning-class-pareto-minority-boost",
            "0.7",
            "--enable-fs-iterative-pruning-class-pareto-stability-gate",
            "--fs-iterative-pruning-class-pareto-stability-subsamples",
            "5",
            "--fs-iterative-pruning-class-pareto-stability-fraction",
            "0.72",
            "--fs-iterative-pruning-class-pareto-stability-threshold",
            "0.56",
            "--fs-iterative-pruning-class-pareto-stability-min-overlap",
            "0.45",
            "--fs-iterative-pruning-class-pareto-stability-min-stable-features",
            "2",
            "--disable-fs-iterative-pruning-class-pareto-stability-fallback-on-failure",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert bool(cfg.fs_iterative_pruning_bounded_use_cpss_overlay) is True
    assert int(cfg.fs_iterative_pruning_bounded_cpss_pairs) == 3
    assert float(cfg.fs_iterative_pruning_bounded_cpss_stability_threshold) == pytest.approx(0.58)
    assert float(cfg.fs_iterative_pruning_bounded_cpss_min_jaccard) == pytest.approx(0.22)
    assert float(cfg.fs_iterative_pruning_bounded_cpss_max_score_drop) == pytest.approx(0.015)
    assert bool(cfg.fs_iterative_pruning_class_pareto_prefilter_enabled) is True
    assert int(cfg.fs_iterative_pruning_class_pareto_min_classes) == 3
    assert int(cfg.fs_iterative_pruning_class_pareto_top_per_class) == 20
    assert float(cfg.fs_iterative_pruning_class_pareto_global_fraction) == pytest.approx(0.33)
    assert float(cfg.fs_iterative_pruning_class_pareto_minority_boost) == pytest.approx(0.7)
    assert bool(cfg.fs_iterative_pruning_class_pareto_stability_gate_enabled) is True
    assert int(cfg.fs_iterative_pruning_class_pareto_stability_subsamples) == 5
    assert float(cfg.fs_iterative_pruning_class_pareto_stability_fraction) == pytest.approx(0.72)
    assert float(cfg.fs_iterative_pruning_class_pareto_stability_threshold) == pytest.approx(0.56)
    assert float(cfg.fs_iterative_pruning_class_pareto_stability_min_overlap) == pytest.approx(0.45)
    assert int(cfg.fs_iterative_pruning_class_pareto_stability_min_stable_features) == 2
    assert bool(cfg.fs_iterative_pruning_class_pareto_stability_fallback_on_failure) is False


def test_iterative_pruning_variant_method_sets_enable_expected_overlays_by_default():
    parser = benchmark.build_arg_parser()
    spec = benchmark.BENCHMARK_DATASETS["synthetic_medium_mixed"]

    args_base = parser.parse_args(
        ["--datasets", "synthetic_medium_mixed", "--fs-method-set", "mnpo_iterative_pruning_bounded_extended"]
    )
    cfg_base = benchmark._build_base_config(args_base, spec, seed=11)
    assert bool(cfg_base.fs_iterative_pruning_bounded_use_cpss_overlay) is False
    assert bool(cfg_base.fs_iterative_pruning_class_pareto_prefilter_enabled) is False
    assert bool(cfg_base.fs_iterative_pruning_class_pareto_stability_gate_enabled) is False

    args_cpss = parser.parse_args(
        ["--datasets", "synthetic_medium_mixed", "--fs-method-set", "mnpo_iterative_pruning_bounded_cpss_extended"]
    )
    cfg_cpss = benchmark._build_base_config(args_cpss, spec, seed=11)
    assert bool(cfg_cpss.fs_iterative_pruning_bounded_use_cpss_overlay) is True
    assert bool(cfg_cpss.fs_iterative_pruning_class_pareto_prefilter_enabled) is False
    assert bool(cfg_cpss.fs_iterative_pruning_class_pareto_stability_gate_enabled) is False

    args_pareto = parser.parse_args(
        ["--datasets", "synthetic_medium_mixed", "--fs-method-set", "mnpo_iterative_pruning_bounded_pareto_extended"]
    )
    cfg_pareto = benchmark._build_base_config(args_pareto, spec, seed=11)
    assert bool(cfg_pareto.fs_iterative_pruning_bounded_use_cpss_overlay) is False
    assert bool(cfg_pareto.fs_iterative_pruning_class_pareto_prefilter_enabled) is True
    assert bool(cfg_pareto.fs_iterative_pruning_class_pareto_stability_gate_enabled) is False

    args_stability = parser.parse_args(
        [
            "--datasets",
            "synthetic_medium_mixed",
            "--fs-method-set",
            "mnpo_iterative_pruning_bounded_pareto_stability_extended",
        ]
    )
    cfg_stability = benchmark._build_base_config(args_stability, spec, seed=11)
    assert bool(cfg_stability.fs_iterative_pruning_bounded_use_cpss_overlay) is False
    assert bool(cfg_stability.fs_iterative_pruning_class_pareto_prefilter_enabled) is True
    assert bool(cfg_stability.fs_iterative_pruning_class_pareto_stability_gate_enabled) is True

def test_df_fastpath_flags_are_noop_after_cleanup():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "int_low_gof_downweighting",
            "--enable-df-fastpath",
            "--df-fastpath-scope",
            "fs_only",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["int_low_gof_downweighting"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert cfg.df_fastpath_enabled is False

    args_fs = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "lung_gordon",
            "--enable-df-fastpath",
            "--df-fastpath-scope",
            "fs_only",
        ]
    )
    spec_fs = benchmark.BENCHMARK_DATASETS["lung_gordon"]
    cfg_fs = benchmark._build_base_config(args_fs, spec_fs, seed=11)
    assert cfg_fs.df_fastpath_enabled is False


def test_df_fastpath_cli_flags_emit_deprecation_warnings():
    benchmark._DEPRECATED_TOGGLE_WARNED.clear()
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--enable-df-fastpath",
            "--df-fastpath-trigger",
            "low_unique",
            "--df-fastpath-small-n-threshold",
            "128",
            "--df-fastpath-unique-ratio-threshold",
            "0.10",
            "--df-fastpath-n-unique-threshold",
            "3",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]
    with pytest.warns(DeprecationWarning):
        _ = benchmark._build_base_config(args, spec, seed=11)


def test_maqc_pairing_improvement_thresholds_populate_base_config():
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--enable-maqc-pairing",
            "--maqc-fs-method-sets",
            "strict_plus_mrmr",
            "mnpo_ova_extended",
            "--maqc-pairing-min-improvement",
            "0.02",
            "--maqc-pairing-min-improvement-se-mult",
            "1.0",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]
    cfg = benchmark._build_base_config(args, spec, seed=11)
    assert cfg.enable_maqc_pairing is True
    assert float(cfg.maqc_pairing_min_improvement) == pytest.approx(0.02)
    assert float(cfg.maqc_pairing_min_improvement_se_mult) == pytest.approx(1.0)

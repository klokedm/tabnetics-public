"""Fault-injection coverage for process-isolated feature-selection dispatch."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import types
from pathlib import Path

import numpy as np
import pytest
import tabnetics.feature_selection.method_execution as execution_module
from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.method_execution import (
    MethodExecutionTask,
    execute_isolated_method_tasks,
    process_isolation_available,
)
from tabnetics.feature_selection.mnpo.portfolio import selector_result_eligibility

pytestmark = pytest.mark.skipif(
    not process_isolation_available(),
    reason="feature-selection process isolation requires a POSIX fork context",
)


def _xy(*, n_samples: int = 32, n_features: int = 12, seed: int = 7):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = np.asarray([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    return X, y


def _bind_method(selector: FeatureSelector, method_name: str, handler) -> None:
    attr_by_method = {
        "mutual_information": "_mutual_information_selection",
        "anova_f": "_anova_f_selection",
    }
    setattr(selector, attr_by_method[method_name], types.MethodType(handler, selector))


def _seed_echo_handler(self, X, y, n_target_features):
    del y
    index = int(self.random_state) % int(X.shape[1])
    return (
        {
            "selected_indices": np.asarray([index], dtype=int),
            "scores": {index: float(self.random_state)},
            "observed_seed": int(self.random_state),
            "observed_parallel_n_jobs": int(self.parallel_n_jobs),
        },
        {index: float(self.random_state)},
    )


def _sleep_handler(self, X, y, n_target_features):
    del self, X, y, n_target_features
    time.sleep(1.0)
    return {"selected_indices": np.asarray([0], dtype=int), "scores": {0: 1.0}}, {0: 1.0}


def _infinite_handler(self, X, y, n_target_features):
    del self, X, y, n_target_features
    while True:
        time.sleep(0.01)


def _crash_handler(self, X, y, n_target_features):
    del self, X, y, n_target_features
    os._exit(73)


def _read_only_handler(self, X, y, n_target_features):
    del self, n_target_features
    assert not bool(X.flags.writeable)
    assert not bool(y.flags.writeable)
    index = int(X.shape[1] // 2)
    return (
        {
            "selected_indices": np.asarray([index], dtype=int),
            "scores": {index: float(X.shape[0])},
            "worker_pid": int(os.getpid()),
        },
        {index: float(X.shape[0])},
    )


def _scheduler_infinite_worker(task):
    del task
    while True:
        time.sleep(0.01)


def _spawn_probe_worker(task):
    return {
        "method_name": str(task.method_name),
        "start_method": mp.get_start_method(),
        "worker_pid": int(os.getpid()),
    }


def _orphaning_worker(task):
    del task
    child_pid = os.fork()
    if child_pid == 0:
        try:
            while True:
                time.sleep(0.01)
        finally:
            os._exit(0)
    return {"child_pid": int(child_pid)}


def _cleanup_assertion() -> None:
    time.sleep(0.03)
    remaining = [
        process
        for process in mp.active_children()
        if str(process.name).startswith("tabnetics-fs-")
    ]
    assert not remaining


def _assert_fail_closed_exclusion(payload) -> None:
    eligibility = selector_result_eligibility(payload)
    assert eligibility["status"] == "incomplete_excluded"
    assert bool(eligibility["legacy_vote_eligible"]) is False
    assert bool(eligibility["mnpo_candidate_eligible"]) is False
    assert bool(eligibility["mnpo_consensus_eligible"]) is False


def test_multithreaded_interactive_parent_fails_closed_before_spawn(monkeypatch):
    monkeypatch.setattr(execution_module, "_parent_has_multiple_threads", lambda: True)
    monkeypatch.setattr(execution_module, "_spawn_bootstrap_available", lambda: False)
    assert execution_module._select_start_method() is None


def test_multithreaded_parent_uses_spawn_and_reaps_serializable_worker(monkeypatch):
    if "spawn" not in mp.get_all_start_methods():
        pytest.skip("multiprocessing spawn context is unavailable")
    monkeypatch.setattr(execution_module, "_parent_has_multiple_threads", lambda: True)
    assert execution_module._select_start_method() == "spawn"

    task = MethodExecutionTask(
        ordinal=0,
        method_name="spawn_fixture",
        method_seed=1,
        timeout_seconds=2.0,
        max_rss_bytes=0,
    )
    outcomes = execute_isolated_method_tasks(
        (task,),
        worker=_spawn_probe_worker,
        max_workers=1,
    )

    outcome = outcomes[0]
    assert outcome.status == "completed"
    assert outcome.payload["method_name"] == "spawn_fixture"
    assert outcome.payload["start_method"] == "spawn"
    assert outcome.payload["worker_pid"] != os.getpid()
    _cleanup_assertion()


def test_parallel_workers_use_independent_seeds_and_preserve_parent_state():
    X, y = _xy()
    methods = ("mutual_information", "anova_f")
    sequential = FeatureSelector(
        random_state=41,
        enabled_methods=methods,
        parallel_n_jobs=1,
    )
    parallel = FeatureSelector(
        random_state=41,
        enabled_methods=methods,
        parallel_n_jobs=2,
    )
    for selector in (sequential, parallel):
        for method_name in methods:
            _bind_method(selector, method_name, _seed_echo_handler)

    sequential_results, _ = sequential._run_selection_methods(X, y, n_target=3)
    parallel_results, _ = parallel._run_selection_methods(X, y, n_target=3)

    assert sequential.random_state == 41
    assert parallel.random_state == 41
    assert list(parallel_results) == list(methods)
    for method_name in methods:
        sequential_payload = sequential_results[method_name][0]
        parallel_payload = parallel_results[method_name][0]
        expected_seed = parallel._derive_method_seed(method_name)
        assert sequential_payload["observed_seed"] == expected_seed
        assert parallel_payload["observed_seed"] == expected_seed
        assert sequential_payload["observed_parallel_n_jobs"] == 1
        assert parallel_payload["observed_parallel_n_jobs"] == 1
        np.testing.assert_array_equal(
            sequential_payload["selected_indices"], parallel_payload["selected_indices"]
        )
        assert sequential_payload["scores"] == parallel_payload["scores"]
        assert parallel_payload["execution_status"] == "completed"
        assert parallel_payload["execution_provenance"]["backend"] == "process"

    _cleanup_assertion()


@pytest.mark.parametrize("handler", (_sleep_handler, _infinite_handler))
def test_hard_timeout_terminates_sleep_and_infinite_workers(handler):
    X, y = _xy()
    selector = FeatureSelector(
        random_state=17,
        enabled_methods={"mutual_information"},
        method_timeout_seconds=0.10,
    )
    _bind_method(selector, "mutual_information", handler)

    started_at = time.monotonic()
    results, runtimes = selector._run_selection_methods(X, y, n_target=3)
    elapsed = time.monotonic() - started_at

    payload = results["mutual_information"][0]
    # Spawn startup is outside the configured method-work budget on threaded
    # parents; the recorded worker runtime remains the enforced bound.
    assert elapsed < 3.0
    assert runtimes["mutual_information"] < 0.75
    assert payload["execution_status"] == "timed_out"
    assert payload["stop_reason"] == "method_timeout"
    assert bool(payload["timed_out"]) is True
    assert bool(payload["incomplete"]) is True
    assert payload["execution_provenance"]["worker_exit_code"] is not None
    _assert_fail_closed_exclusion(payload)
    _cleanup_assertion()


def test_crashed_worker_is_fail_closed_and_reaped():
    X, y = _xy()
    selector = FeatureSelector(
        random_state=19,
        enabled_methods={"mutual_information"},
        method_timeout_seconds=1.0,
    )
    _bind_method(selector, "mutual_information", _crash_handler)

    results, _ = selector._run_selection_methods(X, y, n_target=3)
    payload = results["mutual_information"][0]

    assert payload["execution_status"] == "crashed"
    assert payload["stop_reason"] == "worker_crash"
    assert bool(payload["incomplete"]) is True
    assert payload["execution_provenance"]["worker_exit_code"] == 73
    _assert_fail_closed_exclusion(payload)
    _cleanup_assertion()


def test_rss_cap_is_fail_closed_and_reaped():
    X, y = _xy()
    selector = FeatureSelector(
        random_state=23,
        enabled_methods={"mutual_information"},
        method_max_rss_mb=1.0,
    )
    _bind_method(selector, "mutual_information", _sleep_handler)

    results, _ = selector._run_selection_methods(X, y, n_target=3)
    payload = results["mutual_information"][0]

    assert payload["execution_status"] == "rss_limit_exceeded"
    assert payload["stop_reason"] == "method_rss_limit_exceeded"
    assert bool(payload["resource_exhausted"]) is True
    assert int(payload["execution_provenance"]["peak_rss_bytes"]) > 0
    _assert_fail_closed_exclusion(payload)
    _cleanup_assertion()


def test_cancellation_reaps_active_worker(monkeypatch):
    task = MethodExecutionTask(
        ordinal=0,
        method_name="cancellation_fixture",
        method_seed=1,
        timeout_seconds=0.0,
        max_rss_bytes=0,
    )

    def _interrupt_scheduler(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(execution_module, "sleep", _interrupt_scheduler)
    with pytest.raises(KeyboardInterrupt):
        execute_isolated_method_tasks(
            (task,),
            worker=_scheduler_infinite_worker,
            max_workers=1,
        )
    _cleanup_assertion()


def test_normal_worker_exit_cleans_orphaned_descendant():
    task = MethodExecutionTask(
        ordinal=0,
        method_name="orphan_fixture",
        method_seed=1,
        timeout_seconds=2.0,
        max_rss_bytes=0,
    )
    outcomes = execute_isolated_method_tasks(
        (task,),
        worker=_orphaning_worker,
        max_workers=1,
    )

    outcome = outcomes[0]
    child_pid = int(outcome.payload["child_pid"])
    assert outcome.status == "orphaned_descendant"
    assert outcome.exception_type == "OrphanedWorkerDescendant"
    assert not Path(f"/proc/{child_pid}").exists()
    _cleanup_assertion()


def test_large_read_only_inputs_are_inherited_without_residual_workers():
    X, y = _xy(n_samples=512, n_features=2048, seed=29)
    X.setflags(write=False)
    y.setflags(write=False)
    selector = FeatureSelector(
        random_state=29,
        enabled_methods={"mutual_information", "anova_f"},
        parallel_n_jobs=2,
    )
    for method_name in ("mutual_information", "anova_f"):
        _bind_method(selector, method_name, _read_only_handler)

    results, _ = selector._run_selection_methods(X, y, n_target=3)

    for method_name, (payload, _) in results.items():
        assert payload["execution_status"] == "completed"
        worker_pid = int(payload["worker_pid"])
        assert not Path(f"/proc/{worker_pid}").exists(), method_name
    _cleanup_assertion()


def test_default_single_worker_path_remains_in_process():
    X, y = _xy()
    observed_pids = []

    def _in_process_handler(self, X_data, y_data, n_target_features):
        del self, X_data, y_data, n_target_features
        observed_pids.append(int(os.getpid()))
        return {"selected_indices": np.asarray([0], dtype=int), "scores": {0: 1.0}}, {0: 1.0}

    selector = FeatureSelector(random_state=31, enabled_methods={"mutual_information"})
    _bind_method(selector, "mutual_information", _in_process_handler)
    results, _ = selector._run_selection_methods(X, y, n_target=3)

    assert observed_pids == [os.getpid()]
    assert "execution_provenance" not in results["mutual_information"][0]
    _cleanup_assertion()

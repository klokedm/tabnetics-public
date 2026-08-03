"""POSIX process isolation for feature-selection method dispatch.

This module is deliberately narrow: it executes one already-resolved feature
selection method per short-lived process.  It is not a general task runner.
The parent owns scheduling, wall-clock/RSS limits, and process-group cleanup so
that a timed out method cannot keep consuming CPU or leave joblib descendants
behind.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
import signal
import sys
import threading
from time import perf_counter, sleep
import traceback
from typing import Any, Callable, Iterable, Optional


_POLL_INTERVAL_SECONDS = 0.01
_TERMINATE_GRACE_SECONDS = 0.20
_STARTUP_TIMEOUT_SECONDS = 30.0
_ERROR_MESSAGE_LIMIT = 512


class ProcessIsolationUnavailable(RuntimeError):
    """Raised when this platform cannot create an independently killable worker."""


@dataclass(frozen=True)
class MethodExecutionTask:
    """Immutable parent-side execution contract for one selector method."""

    ordinal: int
    method_name: str
    method_seed: Optional[int]
    timeout_seconds: float
    max_rss_bytes: int
    inner_n_jobs: int = 1


@dataclass(frozen=True)
class MethodExecutionOutcome:
    """Serializable terminal state produced by the isolated method runner."""

    task: MethodExecutionTask
    status: str
    runtime_seconds: float
    peak_rss_bytes: int
    worker_exit_code: Optional[int]
    payload: Any = None
    exception_type: str = ""
    exception_message: str = ""


@dataclass
class _ActiveMethod:
    task: MethodExecutionTask
    process: Any
    receiver: Connection
    launched_at: float
    started_at: Optional[float] = None
    peak_rss_bytes: int = 0
    message: Optional[dict[str, Any]] = None


def _cloudpickle_available() -> bool:
    try:
        import cloudpickle  # noqa: F401
    except ImportError:
        return False
    return True


def _parent_has_multiple_threads() -> bool:
    """Detect native as well as Python threads before selecting fork()."""

    try:
        return sum(1 for _ in os.scandir("/proc/self/task")) > 1
    except OSError:
        return threading.active_count() > 1


def _spawn_bootstrap_available() -> bool:
    """Return whether multiprocessing spawn can import the parent main module."""

    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", "") if main_module else ""
    return bool(main_file and os.path.isfile(str(main_file)))


def _select_start_method() -> Optional[str]:
    """Choose COW fork only when Python-level fork safety is available.

    ``fork`` retains the input arrays copy-on-write on the normal single-thread
    CPU-host path.  Forking a Python process that already owns threads is not
    safe, so use a cloudpickle-backed ``spawn`` child in that case.  Both paths
    preserve an isolated selector snapshot; neither reintroduces a thread pool.
    """

    if os.name != "posix":
        return None
    available = set(mp.get_all_start_methods())
    if "fork" in available and not _parent_has_multiple_threads():
        return "fork"
    if (
        "spawn" in available
        and _cloudpickle_available()
        and _spawn_bootstrap_available()
    ):
        return "spawn"
    return None


def process_isolation_available() -> bool:
    """Return whether a safe process-isolated feature-selection worker exists."""

    return _select_start_method() is not None


def _read_rss_bytes(pid: Optional[int]) -> Optional[int]:
    """Read Linux resident memory without adding a runtime psutil dependency."""

    if pid is None or pid <= 0:
        return None
    try:
        with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    if len(fields) >= 2:
                        return int(fields[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def _read_process_group_rss_bytes(pid: Optional[int]) -> Optional[int]:
    """Return aggregate RSS for a worker process group when it is isolated."""

    if pid is None or pid <= 0 or os.name != "posix":
        return _read_rss_bytes(pid)
    try:
        process_group = int(os.getpgid(int(pid)))
    except (ProcessLookupError, OSError):
        return None
    # Before the child calls setsid(), scanning its inherited parent group would
    # incorrectly charge unrelated work to this method.  Use leader RSS during
    # that brief startup window and aggregate only after the private group exists.
    if process_group != int(pid):
        return _read_rss_bytes(pid)

    total = 0
    found = False
    try:
        entries = tuple(os.scandir("/proc"))
    except OSError:
        return _read_rss_bytes(pid)
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            with open(
                f"/proc/{entry.name}/stat", "r", encoding="utf-8"
            ) as handle:
                stat_tail = handle.read().rsplit(")", 1)[1].split()
            # /proc/<pid>/stat fields after the executable name: state, ppid,
            # pgrp, ... so index 2 is the process group identifier.
            if len(stat_tail) < 3 or int(stat_tail[2]) != process_group:
                continue
            rss_bytes = _read_rss_bytes(int(entry.name))
            if rss_bytes is not None:
                total += int(rss_bytes)
                found = True
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
    return int(total) if found else _read_rss_bytes(pid)


def _process_group_members(process_group: int) -> tuple[int, ...]:
    """Return current members of a private worker process group on Linux."""

    if os.name != "posix" or int(process_group) <= 0:
        return tuple()
    members = []
    try:
        entries = tuple(os.scandir("/proc"))
    except OSError:
        return tuple()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            with open(
                f"/proc/{entry.name}/stat", "r", encoding="utf-8"
            ) as handle:
                stat_tail = handle.read().rsplit(")", 1)[1].split()
            if len(stat_tail) >= 3 and int(stat_tail[2]) == int(process_group):
                members.append(int(entry.name))
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
    return tuple(sorted(members))


def _become_process_group_leader() -> None:
    """Put the worker and any descendants in a killable session/process group."""

    try:
        os.setsid()
    except (AttributeError, OSError):
        # The parent verifies group leadership before using killpg().  A direct
        # terminate/kill fallback remains safe if a platform rejects setsid().
        return


def _send_worker_message(sender: Connection, message: dict[str, Any]) -> None:
    try:
        sender.send(message)
    except BaseException as exc:  # pragma: no cover - requires unpickleable result
        fallback = {
            "kind": "serialization_error",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:_ERROR_MESSAGE_LIMIT],
        }
        try:
            sender.send(fallback)
        except BaseException:
            pass


def _method_worker_entry(
    sender: Connection,
    task: MethodExecutionTask,
    worker: Callable[[MethodExecutionTask], Any],
) -> None:
    """Run one inherited selector callable and send one terminal message."""

    try:
        _become_process_group_leader()
        _send_worker_message(sender, {"kind": "ready"})
        started_at = perf_counter()
        payload = worker(task)
        _send_worker_message(
            sender,
            {
                "kind": "ok",
                "payload": payload,
                "worker_runtime_seconds": float(perf_counter() - started_at),
            },
        )
    except BaseException as exc:
        _send_worker_message(
            sender,
            {
                "kind": "exception",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:_ERROR_MESSAGE_LIMIT],
                "traceback": traceback.format_exc(limit=5)[-_ERROR_MESSAGE_LIMIT:],
            },
        )
    finally:
        try:
            sender.close()
        except BaseException:
            pass


def _serialized_method_worker_entry(
    sender: Connection,
    task: MethodExecutionTask,
    serialized_worker: bytes,
) -> None:
    """Load a trusted parent-side worker snapshot for the spawn fallback."""

    try:
        import cloudpickle

        worker = cloudpickle.loads(serialized_worker)
    except BaseException as exc:
        _send_worker_message(
            sender,
            {
                "kind": "exception",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:_ERROR_MESSAGE_LIMIT],
            },
        )
        try:
            sender.close()
        except BaseException:
            pass
        return
    _method_worker_entry(sender, task, worker)


def _terminate_process_tree(active: _ActiveMethod) -> None:
    """Terminate one worker and inherited descendants, then reap it."""

    process = active.process
    pid = process.pid
    group_terminated = False
    if pid is not None and os.name == "posix":
        try:
            if os.getpgid(int(pid)) == int(pid):
                os.killpg(int(pid), signal.SIGTERM)
                group_terminated = True
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if not group_terminated:
        try:
            process.terminate()
        except (AttributeError, OSError):
            pass

    process.join(timeout=_TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        group_killed = False
        if pid is not None and os.name == "posix":
            try:
                if os.getpgid(int(pid)) == int(pid):
                    os.killpg(int(pid), signal.SIGKILL)
                    group_killed = True
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if not group_killed:
            try:
                process.kill()
            except (AttributeError, OSError):
                pass
        process.join(timeout=_TERMINATE_GRACE_SECONDS)


def _terminate_orphaned_process_group(active: _ActiveMethod) -> tuple[int, ...]:
    """Kill descendants left after a nominally successful worker exit.

    A worker calls ``setsid()``, so its PID is also the process-group ID. Once
    the direct worker has exited and been reaped, any remaining member of that
    private group is an orphaned descendant. Treat that as a fail-closed worker
    fault rather than returning a successful method result with leaked work.
    """

    leader_pid = active.process.pid
    if leader_pid is None or os.name != "posix":
        return tuple()
    members = _process_group_members(int(leader_pid))
    if not members:
        return tuple()
    try:
        os.killpg(int(leader_pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return members

    deadline = perf_counter() + _TERMINATE_GRACE_SECONDS
    while perf_counter() < deadline:
        if not _process_group_members(int(leader_pid)):
            return members
        sleep(_POLL_INTERVAL_SECONDS)
    if _process_group_members(int(leader_pid)):
        try:
            os.killpg(int(leader_pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    deadline = perf_counter() + _TERMINATE_GRACE_SECONDS
    while perf_counter() < deadline:
        if not _process_group_members(int(leader_pid)):
            break
        sleep(_POLL_INTERVAL_SECONDS)
    return members


def _close_active(active: _ActiveMethod) -> None:
    try:
        active.receiver.close()
    except (OSError, ValueError):
        pass
    try:
        active.process.close()
    except (AttributeError, ValueError):
        pass


def _drain_worker_message(active: _ActiveMethod) -> None:
    """Receive any completed terminal message without blocking the scheduler."""

    try:
        while active.receiver.poll():
            message = active.receiver.recv()
            if str(message.get("kind", "")) == "ready":
                if active.started_at is None:
                    active.started_at = perf_counter()
                continue
            if active.message is None:
                active.message = message
    except EOFError:
        pass


def _active_runtime_seconds(active: _ActiveMethod) -> float:
    started_at = active.started_at
    if started_at is None:
        return 0.0
    return float(max(0.0, perf_counter() - started_at))


def _outcome_from_terminal_worker(active: _ActiveMethod) -> MethodExecutionOutcome:
    process = active.process
    process.join(timeout=0)
    exit_code = process.exitcode
    message = active.message or {}
    runtime = _active_runtime_seconds(active)
    kind = str(message.get("kind", ""))
    if kind == "ok" and exit_code == 0:
        orphaned_members = _terminate_orphaned_process_group(active)
        if orphaned_members:
            return MethodExecutionOutcome(
                task=active.task,
                status="orphaned_descendant",
                runtime_seconds=runtime,
                peak_rss_bytes=int(active.peak_rss_bytes),
                worker_exit_code=exit_code,
                payload=message.get("payload"),
                exception_type="OrphanedWorkerDescendant",
                exception_message=(
                    "terminated worker descendants after normal worker exit: "
                    + ",".join(str(pid) for pid in orphaned_members)
                ),
            )
        return MethodExecutionOutcome(
            task=active.task,
            status="completed",
            runtime_seconds=runtime,
            peak_rss_bytes=int(active.peak_rss_bytes),
            worker_exit_code=exit_code,
            payload=message.get("payload"),
        )
    if kind in {"exception", "serialization_error"}:
        return MethodExecutionOutcome(
            task=active.task,
            status="failed",
            runtime_seconds=runtime,
            peak_rss_bytes=int(active.peak_rss_bytes),
            worker_exit_code=exit_code,
            exception_type=str(message.get("exception_type", "WorkerError")),
            exception_message=str(message.get("exception_message", ""))[:_ERROR_MESSAGE_LIMIT],
        )
    return MethodExecutionOutcome(
        task=active.task,
        status="crashed",
        runtime_seconds=runtime,
        peak_rss_bytes=int(active.peak_rss_bytes),
        worker_exit_code=exit_code,
        exception_type="WorkerCrash",
        exception_message="worker exited without a terminal result",
    )


def _forced_stop_outcome(active: _ActiveMethod, status: str) -> MethodExecutionOutcome:
    _terminate_process_tree(active)
    return MethodExecutionOutcome(
        task=active.task,
        status=str(status),
        runtime_seconds=_active_runtime_seconds(active),
        peak_rss_bytes=int(active.peak_rss_bytes),
        worker_exit_code=active.process.exitcode,
        exception_type="",
        exception_message="",
    )


def execute_isolated_method_tasks(
    tasks: Iterable[MethodExecutionTask],
    *,
    worker: Callable[[MethodExecutionTask], Any],
    max_workers: int,
) -> tuple[MethodExecutionOutcome, ...]:
    """Execute resolved selector methods with hard wall-clock and RSS stops.

    A single-threaded parent uses the POSIX fork context, preserving a fully
    configured selector and large read-only arrays copy-on-write.  If the parent
    already owns Python threads, a trusted cloudpickle snapshot is loaded in a
    spawn child instead.  Every terminal return reaps its direct child; forced
    stops additionally kill the worker process group.
    """

    start_method = _select_start_method()
    if start_method is None:
        raise ProcessIsolationUnavailable(
            "feature-selection process isolation requires POSIX fork or cloudpickle spawn"
        )

    ordered_tasks = tuple(tasks)
    ordinals = [int(task.ordinal) for task in ordered_tasks]
    names = [str(task.method_name) for task in ordered_tasks]
    if len(set(ordinals)) != len(ordinals) or len(set(names)) != len(names):
        raise ValueError("isolated method tasks require unique ordinals and method names")
    if not ordered_tasks:
        return tuple()

    serialized_worker: Optional[bytes] = None
    if start_method == "spawn":
        try:
            import cloudpickle

            serialized_worker = cloudpickle.dumps(worker)
        except BaseException as exc:
            raise ProcessIsolationUnavailable(
                "could not serialize the isolated feature-selection worker"
            ) from exc

    context = mp.get_context(start_method)
    worker_cap = max(1, int(max_workers))
    pending = list(ordered_tasks)
    active: dict[int, _ActiveMethod] = {}
    outcomes: dict[int, MethodExecutionOutcome] = {}

    try:
        while pending or active:
            while pending and len(active) < worker_cap:
                task = pending.pop(0)
                receiver, sender = context.Pipe(duplex=False)
                process = None
                try:
                    if start_method == "fork":
                        target = _method_worker_entry
                        args = (sender, task, worker)
                    else:
                        target = _serialized_method_worker_entry
                        args = (sender, task, serialized_worker)
                    process = context.Process(
                        target=target,
                        args=args,
                        name=f"tabnetics-fs-{task.method_name}",
                    )
                    process.daemon = False
                    process.start()
                    sender.close()
                    active[task.ordinal] = _ActiveMethod(
                        task=task,
                        process=process,
                        receiver=receiver,
                        launched_at=perf_counter(),
                    )
                except BaseException as exc:
                    if process is not None and process.pid is not None:
                        active_item = _ActiveMethod(
                            task=task,
                            process=process,
                            receiver=receiver,
                            launched_at=perf_counter(),
                        )
                        _terminate_process_tree(active_item)
                        _close_active(active_item)
                    else:
                        try:
                            receiver.close()
                        except (OSError, ValueError):
                            pass
                    try:
                        sender.close()
                    except (OSError, ValueError):
                        pass
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    outcomes[task.ordinal] = MethodExecutionOutcome(
                        task=task,
                        status="launch_failed",
                        runtime_seconds=0.0,
                        peak_rss_bytes=0,
                        worker_exit_code=None,
                        exception_type=type(exc).__name__,
                        exception_message=str(exc)[:_ERROR_MESSAGE_LIMIT],
                    )
                    continue

            made_progress = False
            for ordinal in sorted(tuple(active)):
                item = active.get(ordinal)
                if item is None:
                    continue
                rss_bytes = (
                    _read_process_group_rss_bytes(item.process.pid)
                    if int(item.task.max_rss_bytes) > 0
                    else _read_rss_bytes(item.process.pid)
                )
                if rss_bytes is not None:
                    item.peak_rss_bytes = max(int(item.peak_rss_bytes), int(rss_bytes))

                if (
                    int(item.task.max_rss_bytes) > 0
                    and int(item.peak_rss_bytes) > int(item.task.max_rss_bytes)
                ):
                    outcomes[ordinal] = _forced_stop_outcome(item, "rss_limit_exceeded")
                    _close_active(item)
                    del active[ordinal]
                    made_progress = True
                    continue

                _drain_worker_message(item)
                if item.started_at is None:
                    if not item.process.is_alive():
                        outcomes[ordinal] = _outcome_from_terminal_worker(item)
                        _close_active(item)
                        del active[ordinal]
                        made_progress = True
                        continue
                    startup_elapsed = float(
                        max(0.0, perf_counter() - item.launched_at)
                    )
                    if startup_elapsed >= _STARTUP_TIMEOUT_SECONDS:
                        outcomes[ordinal] = _forced_stop_outcome(
                            item, "launch_failed"
                        )
                        _close_active(item)
                        del active[ordinal]
                        made_progress = True
                    continue

                elapsed = _active_runtime_seconds(item)
                if (
                    float(item.task.timeout_seconds) > 0.0
                    and elapsed >= float(item.task.timeout_seconds)
                ):
                    outcomes[ordinal] = _forced_stop_outcome(item, "timed_out")
                    _close_active(item)
                    del active[ordinal]
                    made_progress = True
                    continue

                if not item.process.is_alive():
                    _drain_worker_message(item)
                    outcomes[ordinal] = _outcome_from_terminal_worker(item)
                    _close_active(item)
                    del active[ordinal]
                    made_progress = True

            if active and not made_progress:
                sleep(_POLL_INTERVAL_SECONDS)
    finally:
        # Cancellation/KeyboardInterrupt must not leave a child or joblib
        # descendant running after the caller regains control.
        for item in tuple(active.values()):
            _terminate_process_tree(item)
            _close_active(item)

    return tuple(outcomes[task.ordinal] for task in ordered_tasks)

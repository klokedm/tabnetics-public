"""Shared runtime bootstrap helpers for stable package entrypoints."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from collections.abc import Iterator

_THREADING_VARS = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

_thread_local = threading.local()
__tabnetics_execution_ephemeral_globals__ = ("_thread_local",)


def has_nvidia_gpu() -> bool:
    """Fast, import-free check for an NVIDIA GPU driver."""
    try:
        return os.path.isdir("/proc/driver/nvidia") or any(
            os.path.exists(f"/dev/nvidia{i}") for i in range(4)
        )
    except Exception:
        return False


def configure_runtime_environment() -> None:
    """Apply BLAS and CUDA safety defaults before heavy imports."""
    for key, value in _THREADING_VARS.items():
        os.environ.setdefault(key, value)
    if not has_nvidia_gpu():
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def set_sklearn_n_jobs(n_jobs: int = 1) -> None:
    """Set the thread-local sklearn n_jobs value used across the package."""
    _thread_local.sklearn_n_jobs = resolve_sklearn_n_jobs(n_jobs)


def get_sklearn_n_jobs() -> int:
    """Get the thread-local sklearn n_jobs value."""
    return int(getattr(_thread_local, "sklearn_n_jobs", 1))


def resolve_sklearn_n_jobs(n_jobs: int = 1) -> int:
    """Normalize a configured sklearn worker count to a concrete positive value."""

    try:
        requested = int(n_jobs)
    except (TypeError, ValueError):
        requested = 1
    if requested == -1:
        requested = int(os.cpu_count() or 1)
    return int(max(1, requested))


@contextmanager
def sklearn_n_jobs_scope(n_jobs: int = 1) -> Iterator[int]:
    """Bind sklearn worker count for one operation and restore prior thread state."""

    previous = getattr(_thread_local, "sklearn_n_jobs", None)
    effective = resolve_sklearn_n_jobs(n_jobs)
    _thread_local.sklearn_n_jobs = effective
    try:
        yield effective
    finally:
        if previous is None:
            try:
                delattr(_thread_local, "sklearn_n_jobs")
            except AttributeError:
                pass
        else:
            _thread_local.sklearn_n_jobs = int(previous)

"""Shared runtime bootstrap helpers for stable package entrypoints."""

from __future__ import annotations

import os
import threading

_THREADING_VARS = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

_thread_local = threading.local()


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
    _thread_local.sklearn_n_jobs = int(max(1, n_jobs))


def get_sklearn_n_jobs() -> int:
    """Get the thread-local sklearn n_jobs value."""
    return int(getattr(_thread_local, "sklearn_n_jobs", 1))


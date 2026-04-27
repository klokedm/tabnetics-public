"""Benchmark runner and profile surface backing ``tabnetics-benchmark`` / ``python -m tabnetics.benchmarks.cli``, exposing the profile registry for systematic paired comparisons and the runner that enforces the validation-catalog data policy: evidence-bearing runs use the HuggingFace mirror of public upstream sources and do not silently fall back to synthetic proxies."""

from .profiles import FS_METHOD_SETS
from .runner import *  # noqa: F401,F403

__all__ = ["FS_METHOD_SETS"]

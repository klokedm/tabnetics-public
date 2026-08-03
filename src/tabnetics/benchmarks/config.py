"""Benchmark config helpers.

Thin bridge module extracted for T-A3-011 so config helpers can be imported
without pulling the full runner entrypoint call sites.
"""

from __future__ import annotations

from typing import Any


def clone_config(cfg: Any):
    from tabnetics.benchmarks.runner import clone_config as _clone_config

    return _clone_config(cfg)


def apply_config_override(cfg: Any, key: str, value: Any) -> None:
    from tabnetics.benchmarks.runner import apply_config_override as _apply_config_override

    _apply_config_override(cfg, key, value)


def build_base_config(args: Any, spec: Any, seed: int):
    from tabnetics.benchmarks.runner import _build_base_config as _build

    return _build(args, spec, seed)


__all__ = [
    "clone_config",
    "apply_config_override",
    "build_base_config",
]

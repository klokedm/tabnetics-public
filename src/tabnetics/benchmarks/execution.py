"""Benchmark execution helpers.

Extracted compatibility module for T-A3-011.
"""

from __future__ import annotations

from typing import Any


def run_benchmark(args: Any):
    from tabnetics.benchmarks.runner import run_benchmark as _run_benchmark

    return _run_benchmark(args)


def build_arg_parser():
    from tabnetics.benchmarks.runner import build_arg_parser as _build_arg_parser

    return _build_arg_parser()


def main() -> None:
    from tabnetics.benchmarks.runner import main as _main

    _main()


__all__ = [
    "run_benchmark",
    "build_arg_parser",
    "main",
]

"""Installed benchmark CLI entrypoint backing ``tabnetics-benchmark`` / ``python -m tabnetics.benchmarks.cli``; the packaged surface defaults to ``df_stage_position="after_fs"`` and enforces the validation-catalog data policy: the HuggingFace bundle is the operational mirror of public upstream sources for evidence-bearing runs, and synthetic fallback is forbidden there."""

from .runner import build_arg_parser, main, run_benchmark

__all__ = [
    "build_arg_parser",
    "main",
    "run_benchmark",
]

if __name__ == "__main__":
    main()

"""Installed benchmark CLI entrypoint."""

from .runner import build_arg_parser, main, run_benchmark

__all__ = [
    "build_arg_parser",
    "main",
    "run_benchmark",
]

if __name__ == "__main__":
    main()


"""Filesystem path helpers for packaged tabnetics modules."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(anchor: str | Path) -> Path:
    """Resolve the repository root by walking upward until `pyproject.toml` is found."""
    current = Path(anchor).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from anchor={anchor!r}")


__all__ = ["find_repo_root"]

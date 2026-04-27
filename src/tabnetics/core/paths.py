"""Filesystem path helpers for packaged tabnetics modules."""

from __future__ import annotations

from pathlib import Path


def _walk_up(anchor: str | Path) -> tuple[Path, ...]:
    current = Path(anchor).resolve()
    if current.is_file():
        current = current.parent
    return (current, *current.parents)


def find_project_root(anchor: str | Path) -> Path:
    """Resolve the nearest tabnetics project root via ``pyproject.toml``."""
    for candidate in _walk_up(anchor):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate project root from anchor={anchor!r}")


def find_repo_root(anchor: str | Path) -> Path:
    """Resolve the monorepo root when available, else fall back to the project root."""
    project_root = find_project_root(anchor)
    for candidate in (project_root, *project_root.parents):
        if (candidate / ".git").exists():
            return candidate
        if (candidate / "core" / "pyproject.toml").exists() and (candidate / "ui" / "pyproject.toml").exists():
            return candidate
    return project_root


def find_repo_root_or_none(anchor: str | Path) -> Path | None:
    """Like :func:`find_repo_root` but returns ``None`` when called from an
    installed package (e.g. inside ``site-packages``) where no project root or
    repo root can be located.  Safe for module-level use."""
    try:
        return find_repo_root(anchor)
    except RuntimeError:
        return None


__all__ = ["find_project_root", "find_repo_root", "find_repo_root_or_none"]

"""Helpers for run-specific artifact directories."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip().lower())
    cleaned = cleaned.strip("._-")
    return cleaned or "run"


def create_timestamped_run_dir(
    base_dir: Union[str, Path] = "run_artifacts",
    run_name: Optional[str] = None,
) -> Path:
    """
    Create a unique timestamped directory for a run.

    Args:
        base_dir: Parent directory that contains all run folders.
        run_name: Optional suffix to identify the runner.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _sanitize_name(run_name) if run_name else ""
    if suffix:
        candidate = base_path / f"{stamp}_{suffix}"
    else:
        candidate = base_path / stamp

    counter = 1
    while candidate.exists():
        if suffix:
            candidate = base_path / f"{stamp}_{suffix}_{counter:02d}"
        else:
            candidate = base_path / f"{stamp}_{counter:02d}"
        counter += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate

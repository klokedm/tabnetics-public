"""Materialize planned BeyondArena DataFoundry artifacts from Hugging Face.

This module bridges the metadata-only planner and the local runner.  It reads a
``beyondarena_task_manifest.csv`` and materializes exactly the referenced
DataFoundry artifact directories.  Parquet payloads and text caches are
explicit opt-ins so CI and planning commands do not accidentally download the
full BeyondArena corpus.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import pandas as pd

from tabnetics.datasets.beyondarena import (
    BEYONDARENA_HF_REPO_ID,
    CONTAINER_METADATA,
    DATASET_METADATA,
    DATASET_PARQUET,
    DTYPES_METADATA,
    SPLIT_METADATA,
    TASK_METADATA,
    TEXT_CACHE_BASENAME,
    BeyondArenaUnavailableError,
)


DownloadFn = Callable[[str], Path]

_REQUIRED_ARTIFACT_FILES = (
    CONTAINER_METADATA,
    DATASET_METADATA,
    TASK_METADATA,
    SPLIT_METADATA,
    DTYPES_METADATA,
)
_LOCAL_READY_STATUSES = {"already_present", "downloaded"}
_PLAN_READY_STATUSES = _LOCAL_READY_STATUSES | {"planned"}
_OPTIONAL_MISSING_STATUSES = {"error", "missing_remote", "skipped_optional_missing_remote"}
__tabnetics_execution_isolated_state__ = {
    "_PLAN_READY_STATUSES": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "_REQUIRED_ARTIFACT_FILES": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}
_READINESS_COLUMNS = (
    "artifact_plan_ready",
    "artifact_materialization_ready",
    "artifact_local_runner_ready",
    "artifact_required_file_count",
    "artifact_required_ready_count",
    "artifact_required_pending_count",
    "artifact_required_blocked_count",
    "artifact_optional_missing_count",
    "artifact_missing_required_files",
    "artifact_pending_required_files",
    "artifact_optional_missing_files",
    "artifact_blocker",
    "artifact_resume_hint",
)


@dataclass(frozen=True)
class BeyondArenaMaterializeConfig:
    """Controls for materializing planned BeyondArena artifacts."""

    task_manifest_csv: Path
    out_dir: Path
    repo_id: str = BEYONDARENA_HF_REPO_ID
    revision: str = "main"
    include_dataset: bool = False
    include_text_cache: bool = False
    dry_run: bool = False
    force: bool = False
    on_error: str = "row"  # row | raise
    plan_csv: Optional[Path] = None
    require_local_runner_ready: bool = False
    fetch_size_metadata: bool = False


def _norm_uri(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").strip("/")


def _read_manifest(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    missing = sorted({"data_foundry_uri"}.difference(frame.columns))
    if missing:
        raise ValueError(f"BeyondArena task manifest missing required columns: {missing}")
    frame = frame.copy()
    frame["data_foundry_uri"] = frame["data_foundry_uri"].map(_norm_uri)
    frame = frame[frame["data_foundry_uri"].astype(str).str.len().gt(0)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("BeyondArena task manifest contains no data_foundry_uri values")
    return frame


def _artifact_summary_rows(manifest: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for uri, group in manifest.groupby("data_foundry_uri", sort=False, dropna=False):
        first = group.iloc[0]
        records.append(
            {
                "data_foundry_uri": str(uri),
                "dataset_id": str(first.get("dataset_id", "")),
                "dataset_name": str(first.get("dataset_name", "")),
                "problem_type": str(first.get("problem_type", "")),
                "task_type": str(first.get("task_type", "")),
                "task_count": int(len(group)),
            }
        )
    return tuple(records)


def _planned_filenames(
    *,
    include_dataset: bool = False,
    include_text_cache: bool = False,
) -> tuple[tuple[str, bool], ...]:
    files: list[tuple[str, bool]] = [(name, True) for name in _REQUIRED_ARTIFACT_FILES]
    if bool(include_dataset):
        files.append((DATASET_PARQUET, True))
    if bool(include_text_cache):
        files.append((TEXT_CACHE_BASENAME, False))
    return tuple(files)


def _remote_status(remote_path: str, *, required: bool, available_paths: Optional[set[str]]) -> str:
    if available_paths is None:
        return "planned"
    if remote_path in available_paths:
        return "planned"
    return "missing_remote" if required else "skipped_optional_missing_remote"


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _filename_list(frame: pd.DataFrame) -> str:
    filenames = frame.get("filename", pd.Series(dtype=str)).dropna()
    return ";".join(sorted({str(name) for name in filenames}))


def _artifact_readiness_summary(group: pd.DataFrame) -> dict[str, Any]:
    required_mask = (
        group["required"].map(_is_truthy)
        if "required" in group.columns
        else pd.Series(False, index=group.index)
    )
    required = group[required_mask]
    optional = group[~required_mask]
    required_status = required["status"].astype(str) if "status" in required.columns else pd.Series(dtype=str)
    required_ready = required_status.isin(_LOCAL_READY_STATUSES)
    required_pending = required_status.eq("planned")
    required_blocked = ~(required_status.isin(_PLAN_READY_STATUSES))

    metadata = (
        group[group["filename"].astype(str).isin(_REQUIRED_ARTIFACT_FILES)]
        if "filename" in group.columns
        else group.iloc[0:0]
    )
    metadata_status = metadata["status"].astype(str) if "status" in metadata.columns else pd.Series(dtype=str)
    metadata_ready = len(metadata) == len(_REQUIRED_ARTIFACT_FILES) and bool(
        metadata_status.isin(_LOCAL_READY_STATUSES).all()
    )
    dataset = (
        group[group["filename"].astype(str).eq(DATASET_PARQUET)]
        if "filename" in group.columns
        else group.iloc[0:0]
    )
    dataset_status = dataset["status"].astype(str) if "status" in dataset.columns else pd.Series(dtype=str)
    dataset_planned = not dataset.empty
    dataset_ready = bool((not dataset.empty) and dataset_status.isin(_LOCAL_READY_STATUSES).all())

    blocked_required_files = _filename_list(required[required_blocked.to_numpy()])
    pending_required_files = _filename_list(required[required_pending.to_numpy()])
    optional_missing = (
        optional[optional["status"].astype(str).isin(_OPTIONAL_MISSING_STATUSES)]
        if "status" in optional.columns
        else optional.iloc[0:0]
    )
    optional_missing_files = _filename_list(optional_missing)

    artifact_blocker = ""
    artifact_resume_hint = ""
    if blocked_required_files:
        artifact_blocker = f"required files missing or errored: {blocked_required_files}"
        artifact_resume_hint = (
            "Check DataFoundry URI/revision and rerun materialization after fixing remote access."
        )
    elif pending_required_files:
        artifact_blocker = f"required files planned but not materialized: {pending_required_files}"
        artifact_resume_hint = (
            "Run materialization without --dry-run; include --include-dataset before local execution."
        )
    elif not dataset_planned:
        artifact_blocker = f"{DATASET_PARQUET} not planned"
        artifact_resume_hint = "Rerun materialization with --include-dataset before local execution."
    elif not (metadata_ready and dataset_ready):
        artifact_blocker = "required local runner files are incomplete"
        artifact_resume_hint = "Rerun materialization and verify metadata files plus dataset.parquet are present."

    return {
        "artifact_plan_ready": int(required_blocked.sum()) == 0,
        "artifact_materialization_ready": int(required_pending.sum()) == 0 and int(required_blocked.sum()) == 0,
        "artifact_local_runner_ready": bool(metadata_ready and dataset_ready),
        "artifact_required_file_count": int(len(required)),
        "artifact_required_ready_count": int(required_ready.sum()),
        "artifact_required_pending_count": int(required_pending.sum()),
        "artifact_required_blocked_count": int(required_blocked.sum()),
        "artifact_optional_missing_count": int(len(optional_missing)),
        "artifact_missing_required_files": blocked_required_files,
        "artifact_pending_required_files": pending_required_files,
        "artifact_optional_missing_files": optional_missing_files,
        "artifact_blocker": artifact_blocker,
        "artifact_resume_hint": artifact_resume_hint,
    }


def annotate_beyondarena_materialization_readiness(plan: pd.DataFrame) -> pd.DataFrame:
    """Add artifact-level readiness columns to a file-grain materialization plan."""

    if plan.empty or "data_foundry_uri" not in plan.columns:
        return plan.copy()
    annotations: list[dict[str, Any]] = []
    for uri, group in plan.groupby("data_foundry_uri", sort=False, dropna=False):
        annotations.append({"data_foundry_uri": uri, **_artifact_readiness_summary(group)})
    annotated = plan.drop(columns=[col for col in _READINESS_COLUMNS if col in plan.columns]).copy()
    readiness = pd.DataFrame.from_records(annotations)
    return annotated.merge(readiness, on="data_foundry_uri", how="left")


def summarize_beyondarena_materialization(plan: pd.DataFrame) -> dict[str, Any]:
    """Summarize artifact readiness for CLI and issue handoff records."""

    annotated = annotate_beyondarena_materialization_readiness(plan)
    status_counts = _status_counts(annotated)
    if annotated.empty or "data_foundry_uri" not in annotated.columns:
        return {
            "row_count": int(len(annotated)),
            "artifact_count": 0,
            "status_counts": status_counts,
            "artifact_plan_ready_count": 0,
            "artifact_materialization_ready_count": 0,
            "artifact_local_runner_ready_count": 0,
            "artifact_blocked_count": 0,
            "artifact_pending_count": 0,
            "artifact_dataset_not_planned_count": 0,
            "materialization_ready": False,
            "local_runner_ready": False,
            "materialization_blocker": "no materialization rows",
        }
    artifacts = annotated.drop_duplicates("data_foundry_uri")
    blockers = [str(value) for value in artifacts["artifact_blocker"].fillna("") if str(value)]
    artifact_count = int(len(artifacts))
    materialization_ready_count = int(artifacts["artifact_materialization_ready"].map(_is_truthy).sum())
    local_runner_ready_count = int(artifacts["artifact_local_runner_ready"].map(_is_truthy).sum())
    pending_count = int(artifacts["artifact_required_pending_count"].fillna(0).astype(int).gt(0).sum())
    blocked_count = int(artifacts["artifact_required_blocked_count"].fillna(0).astype(int).gt(0).sum())
    dataset_not_planned_count = int(
        artifacts["artifact_blocker"].fillna("").astype(str).eq(f"{DATASET_PARQUET} not planned").sum()
    )
    return {
        "row_count": int(len(annotated)),
        "artifact_count": artifact_count,
        "status_counts": status_counts,
        "artifact_plan_ready_count": int(artifacts["artifact_plan_ready"].map(_is_truthy).sum()),
        "artifact_materialization_ready_count": materialization_ready_count,
        "artifact_local_runner_ready_count": local_runner_ready_count,
        "artifact_blocked_count": blocked_count,
        "artifact_pending_count": pending_count,
        "artifact_dataset_not_planned_count": dataset_not_planned_count,
        "materialization_ready": materialization_ready_count == artifact_count,
        "local_runner_ready": local_runner_ready_count == artifact_count,
        "materialization_blocker": blockers[0] if blockers else "",
    }


def build_beyondarena_materialization_plan(
    manifest: str | Path | pd.DataFrame,
    *,
    out_dir: str | Path,
    include_dataset: bool = False,
    include_text_cache: bool = False,
    available_paths: Optional[Iterable[str]] = None,
    size_by_remote_path: Optional[Mapping[str, int]] = None,
) -> pd.DataFrame:
    """Build a file-grain materialization plan from a task manifest.

    ``available_paths`` is an optional offline/test hook containing repo-relative
    Hugging Face paths.  When omitted, rows are marked ``planned`` and existence
    is checked during download.
    """

    manifest_frame = _read_manifest(manifest)
    available = None if available_paths is None else {_norm_uri(path) for path in available_paths}
    sizes = {
        _norm_uri(path): int(size)
        for path, size in (size_by_remote_path or {}).items()
        if size is not None
    }
    out_root = Path(out_dir)
    records: list[dict[str, Any]] = []
    for artifact in _artifact_summary_rows(manifest_frame):
        uri = str(artifact["data_foundry_uri"])
        local_artifact_dir = out_root / uri
        for filename, required in _planned_filenames(
            include_dataset=include_dataset,
            include_text_cache=include_text_cache,
        ):
            remote_path = f"{uri}/{filename}"
            records.append(
                {
                    **artifact,
                    "filename": filename,
                    "required": bool(required),
                    "remote_path": remote_path,
                    "local_artifact_dir": str(local_artifact_dir),
                    "local_path": str(local_artifact_dir / filename),
                    "status": _remote_status(remote_path, required=required, available_paths=available),
                    "size_bytes": sizes.get(remote_path, pd.NA),
                    "error": "",
                }
            )
    return annotate_beyondarena_materialization_readiness(pd.DataFrame.from_records(records))


def fetch_beyondarena_hf_file_sizes(
    data_foundry_uris: Iterable[str],
    *,
    repo_id: str = BEYONDARENA_HF_REPO_ID,
    revision: str = "main",
) -> dict[str, int]:
    """Fetch remote file sizes for planned DataFoundry artifact directories."""

    try:
        from huggingface_hub import HfApi  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise BeyondArenaUnavailableError(
            "huggingface_hub is required to fetch BeyondArena file-size metadata"
        ) from exc

    api = HfApi()
    sizes: dict[str, int] = {}
    for uri in sorted({_norm_uri(value) for value in data_foundry_uris if _norm_uri(value)}):
        for item in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            path_in_repo=uri,
            recursive=False,
            expand=True,
        ):
            path = _norm_uri(getattr(item, "path", ""))
            size = getattr(item, "size", None)
            if path and size is not None:
                sizes[path] = int(size)
    return sizes


def _copy_downloaded_file(source: Path, target: Path, *, force: bool = False) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not bool(force):
        return "already_present", int(target.stat().st_size)
    try:
        if target.exists() and source.exists() and source.resolve() == target.resolve():
            return "already_present", int(target.stat().st_size)
    except Exception:
        pass
    shutil.copy2(source, target)
    return "downloaded", int(target.stat().st_size)


def materialize_beyondarena_hf_artifacts(
    config: BeyondArenaMaterializeConfig,
    *,
    download_fn: Optional[DownloadFn] = None,
    available_paths: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Materialize manifest-referenced HF files and return file-grain statuses."""

    manifest_frame = _read_manifest(config.task_manifest_csv)
    size_by_remote_path = (
        fetch_beyondarena_hf_file_sizes(
            manifest_frame["data_foundry_uri"].astype(str).tolist(),
            repo_id=config.repo_id,
            revision=config.revision,
        )
        if bool(config.fetch_size_metadata)
        else None
    )
    plan = build_beyondarena_materialization_plan(
        manifest_frame,
        out_dir=config.out_dir,
        include_dataset=bool(config.include_dataset),
        include_text_cache=bool(config.include_text_cache),
        available_paths=available_paths,
        size_by_remote_path=size_by_remote_path,
    )
    if bool(config.dry_run):
        return plan

    if download_fn is None:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency branch
            raise BeyondArenaUnavailableError(
                "huggingface_hub is required to materialize BeyondArena artifacts"
            ) from exc

        def _download(remote_path: str) -> Path:
            return Path(
                hf_hub_download(
                    repo_id=config.repo_id,
                    repo_type="dataset",
                    revision=config.revision,
                    filename=remote_path,
                )
            )

        download_fn = _download

    rows: list[dict[str, Any]] = []
    for record in plan.to_dict(orient="records"):
        status = str(record.get("status", "planned"))
        if status in {"missing_remote", "skipped_optional_missing_remote"}:
            rows.append(record)
            continue
        remote_path = str(record["remote_path"])
        local_path = Path(str(record["local_path"]))
        try:
            source = Path(download_fn(remote_path))
            copied_status, size_bytes = _copy_downloaded_file(source, local_path, force=bool(config.force))
            record["status"] = copied_status
            record["size_bytes"] = int(size_bytes)
        except Exception as exc:
            if bool(record.get("required", True)) and config.on_error == "raise":
                raise
            record["status"] = "error" if bool(record.get("required", True)) else "skipped_optional_missing_remote"
            record["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)
    return annotate_beyondarena_materialization_readiness(pd.DataFrame.from_records(rows))


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "status" not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame["status"].value_counts(dropna=False).to_dict().items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=BEYONDARENA_HF_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--include-dataset",
        action="store_true",
        help="Also materialize dataset.parquet payloads. Without this flag only metadata files are copied.",
    )
    parser.add_argument(
        "--include-text-cache",
        action="store_true",
        help="Also attempt optional Qwen3 text-cache parquet files; missing caches are skipped.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the plan without downloading or copying files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing local files.")
    parser.add_argument("--on-error", choices=("row", "raise"), default="row")
    parser.add_argument(
        "--require-local-runner-ready",
        action="store_true",
        help=(
            "Return a nonzero exit code unless every planned artifact has required metadata "
            "and dataset.parquet staged for local execution."
        ),
    )
    parser.add_argument(
        "--fetch-size-metadata",
        action="store_true",
        help=(
            "Fetch Hugging Face file-size metadata for planned remote paths before download; "
            "useful with --dry-run to estimate dataset.parquet payload size."
        ),
    )
    parser.add_argument(
        "--plan-csv",
        type=Path,
        help="Output CSV for file-grain materialization statuses; defaults to OUT_DIR/materialization_plan.csv.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = BeyondArenaMaterializeConfig(
        task_manifest_csv=args.task_manifest_csv,
        out_dir=args.out_dir,
        repo_id=str(args.repo_id),
        revision=str(args.revision),
        include_dataset=bool(args.include_dataset),
        include_text_cache=bool(args.include_text_cache),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        on_error=str(args.on_error),
        plan_csv=args.plan_csv,
        require_local_runner_ready=bool(args.require_local_runner_ready),
        fetch_size_metadata=bool(args.fetch_size_metadata),
    )
    rows = materialize_beyondarena_hf_artifacts(config)
    out_csv = config.plan_csv or (config.out_dir / "materialization_plan.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_csv, index=False)
    summary = summarize_beyondarena_materialization(rows)
    payload = {
        **asdict(config),
        "task_manifest_csv": str(config.task_manifest_csv),
        "out_dir": str(config.out_dir),
        "plan_csv": str(out_csv),
        **summary,
        "planned_size_bytes": int(pd.to_numeric(rows.get("size_bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
    }
    print(json.dumps(payload, sort_keys=True))
    if bool(config.require_local_runner_ready) and not bool(summary.get("local_runner_ready", False)):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BeyondArenaMaterializeConfig",
    "annotate_beyondarena_materialization_readiness",
    "build_beyondarena_materialization_plan",
    "fetch_beyondarena_hf_file_sizes",
    "materialize_beyondarena_hf_artifacts",
    "summarize_beyondarena_materialization",
    "main",
]

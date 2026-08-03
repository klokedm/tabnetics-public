"""BeyondArena task manifests and comparison-plan artifacts.

This module is the bridge between metadata-only BeyondArena discovery and the
deferred lab validation run: it writes exact smoke/core task manifests, records
which model profiles are expected, and optionally materializes joined comparison
tables when official and local result rows are available.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from urllib.request import urlopen

import io

import numpy as np
import pandas as pd

from tabnetics.datasets.beyondarena import (
    BEYONDARENA_CORE_TASKS_CSV_URL,
    BEYONDARENA_TASK_METADATA_CSV_URL,
    BeyondArenaTaskMetadataRow,
    build_beyondarena_current_feasibility_task_rows,
    build_beyondarena_smoke_task_rows,
    load_beyondarena_core_tasks_csv,
    load_beyondarena_task_metadata_csv,
    select_beyondarena_core_dataset_split_rows,
    select_beyondarena_core_task_rows,
)

from .beyondarena_compare import (
    BEYONDARENA_METRIC_CONTRACT_VERSION,
    BeyondArenaOfficialResultsStatus,
    build_beyondarena_comparison_artifacts,
    load_official_beyondarena_results,
    metric_lower_is_better,
    normalize_beyondarena_metric_name,
    normalize_beyondarena_result_table,
)
from .beyondarena_materialize import summarize_beyondarena_materialization
from .profiles import beyondarena_parity_inventory


@dataclass(frozen=True)
class BeyondArenaRunProfile:
    """One model/profile row expected in a BeyondArena validation plan."""

    profile_id: str
    method: str
    profile_type: str
    status: str
    device: str
    backend: str = ""
    local_backend: str = ""
    skip_reason: str = ""
    sample_limit: str = ""
    feature_limit: str = ""
    tuning_mode: str = ""
    dependency: str = ""
    install_hint: str = ""
    compatibility_scope: str = ""
    execution_guard: str = ""
    fallback_status: str = ""


@dataclass(frozen=True)
class BeyondArenaPlanArtifacts:
    """File outputs from a manifest/comparison plan write."""

    out_dir: Path
    task_manifest: Path
    model_manifest: Path
    comparison_status: Path
    readiness_audit: Path
    official_normalized: Optional[Path] = None
    local_normalized: Optional[Path] = None
    local_execution_plan: Optional[Path] = None
    pending_local: Optional[Path] = None
    run_spec: Optional[Path] = None
    joined: Optional[Path] = None
    summary: Optional[Path] = None


def _read_table(source: str | Path) -> pd.DataFrame:
    if isinstance(source, Path) or (isinstance(source, str) and "://" not in source):
        path = Path(source)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    with urlopen(str(source)) as response:
        payload = response.read()
    if str(source).lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(payload))
    return pd.read_csv(io.BytesIO(payload))


def _size_tier(n_train: Optional[int]) -> str:
    if n_train is None:
        return "unknown"
    n = int(n_train)
    if n <= 1_000:
        return "tiny"
    if n <= 10_000:
        return "small"
    if n <= 100_000:
        return "medium"
    return "large"


def task_rows_to_manifest_frame(rows: Iterable[BeyondArenaTaskMetadataRow]) -> pd.DataFrame:
    """Convert official metadata rows into the stable run-manifest schema."""

    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "dataset_id": row.tabarena_task_name,
                "dataset_name": row.dataset_name,
                "data_foundry_uri": row.data_foundry_uri,
                "task_id_str": row.task_id_str,
                "split": int(row.split),
                "official_split_id": str(row.split),
                "split_id": row.split_index,
                "repeat": int(row.repeat),
                "fold": int(row.fold),
                "task_type": row.normalized_task_type,
                "official_task_type": row.task_type,
                "problem_type": row.normalized_problem_type,
                "official_problem_type": row.problem_type,
                "metric": normalize_beyondarena_metric_name(row.eval_metric),
                "official_metric": row.eval_metric,
                "lower_is_better": bool(row.metric_lower_is_better),
                "target_column": row.target_name,
                "size_tier": _size_tier(row.num_instances_train),
                "dimensionality": "high" if row.is_high_dimensional else "low",
                "has_text": bool(row.has_text_features),
                "high_cardinality": bool(row.has_high_cardinality_features),
                "num_instances": row.num_instances,
                "num_features": row.num_features,
                "num_cols_after_preprocessing": row.num_cols_after_preprocessing,
                "num_instances_train": row.num_instances_train,
                "num_instances_test": row.num_instances_test,
                "domain": row.domain,
                "source": row.source,
            }
        )
    return pd.DataFrame.from_records(records)


def _manifest_alias_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    aliases: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        dataset_id = str(row["dataset_id"])
        official_split_id = str(row["official_split_id"])
        alias_values = {
            official_split_id,
            str(row["split"]),
            str(row["split_id"]),
            f"{row['repeat']}:{row['fold']}",
            f"r{row['repeat']}f{row['fold']}",
        }
        for alias in alias_values:
            aliases.append(
                {
                    "dataset_id": dataset_id,
                    "split_alias": str(alias),
                    "official_split_id": official_split_id,
                    "manifest_task_type": row["task_type"],
                    "manifest_problem_type": row["problem_type"],
                    "manifest_size_tier": row["size_tier"],
                    "manifest_dimensionality": row["dimensionality"],
                    "manifest_has_text": row["has_text"],
                    "manifest_high_cardinality": row["high_cardinality"],
                    "manifest_artifact_revision": row["data_foundry_uri"],
                }
            )
    return pd.DataFrame.from_records(aliases).drop_duplicates(["dataset_id", "split_alias"])


def align_beyondarena_results_to_manifest(
    results: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Filter normalized result rows to a manifest and canonicalize split ids."""

    if results.empty or manifest.empty:
        return results.iloc[0:0].copy()
    aliases = _manifest_alias_frame(manifest)
    source = results.copy()
    source["source_split_id"] = source["split_id"].astype(str)
    merged = source.merge(
        aliases,
        left_on=["dataset_id", "source_split_id"],
        right_on=["dataset_id", "split_alias"],
        how="inner",
    )
    if merged.empty:
        return source.iloc[0:0].copy()
    merged["split_id"] = merged["official_split_id"].astype(str)
    for target, source_col in (
        ("task_type", "manifest_task_type"),
        ("problem_type", "manifest_problem_type"),
        ("size_tier", "manifest_size_tier"),
        ("dimensionality", "manifest_dimensionality"),
        ("has_text", "manifest_has_text"),
        ("high_cardinality", "manifest_high_cardinality"),
        ("artifact_revision", "manifest_artifact_revision"),
    ):
        if target not in merged.columns:
            merged[target] = merged[source_col]
        else:
            merged[target] = merged[target].where(merged[target].notna(), merged[source_col])
    drop_cols = [
        "split_alias",
        "official_split_id",
        "manifest_task_type",
        "manifest_problem_type",
        "manifest_size_tier",
        "manifest_dimensionality",
        "manifest_has_text",
        "manifest_high_cardinality",
        "manifest_artifact_revision",
    ]
    return merged.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)


def resolve_beyondarena_task_rows(
    *,
    task_metadata_source: str | Path | pd.DataFrame = BEYONDARENA_TASK_METADATA_CSV_URL,
    core_tasks_source: str | Path | pd.DataFrame = BEYONDARENA_CORE_TASKS_CSV_URL,
    subset: str = "smoke",
    max_smoke_items: int = 6,
) -> tuple[BeyondArenaTaskMetadataRow, ...]:
    """Resolve official rows for staged BeyondArena plan scopes."""

    metadata = load_beyondarena_task_metadata_csv(task_metadata_source)
    normalized_subset = str(subset).strip().lower().replace("_", "-")
    allowed = {
        "smoke",
        "current-feasibility",
        "core",
        "core-all-splits",
        "core-classification-all-splits",
        "all",
    }
    if normalized_subset not in allowed:
        raise ValueError(
            "Unknown BeyondArena subset "
            f"{subset!r}; expected smoke, current-feasibility, core, "
            "core-all-splits, core-classification-all-splits, or all"
        )
    if normalized_subset == "all":
        return metadata
    if normalized_subset == "current-feasibility":
        return build_beyondarena_current_feasibility_task_rows(metadata, max_items=1)
    core_tasks = load_beyondarena_core_tasks_csv(core_tasks_source)
    if normalized_subset in {"core-all-splits", "core-classification-all-splits"}:
        rows = select_beyondarena_core_dataset_split_rows(metadata, core_tasks)
        if normalized_subset == "core-classification-all-splits":
            return tuple(
                row
                for row in rows
                if row.normalized_problem_type == "classification" and row.is_classification
            )
        return rows
    core_rows = select_beyondarena_core_task_rows(metadata, core_tasks)
    if normalized_subset == "core":
        return core_rows
    return build_beyondarena_smoke_task_rows(core_rows, max_items=int(max_smoke_items))


def default_beyondarena_run_profiles() -> tuple[BeyondArenaRunProfile, ...]:
    """Return expected model/profile rows for staged BeyondArena comparisons."""

    profiles: list[BeyondArenaRunProfile] = [
        BeyondArenaRunProfile(
            profile_id="tabnetics_current",
            method="tabnetics-current",
            profile_type="local_portfolio",
            status="pending_external_run",
            device="cpu",
            skip_reason="requires materialized BeyondArena artifacts and lab execution",
        ),
        BeyondArenaRunProfile(
            profile_id="tabentics_diakrino_experimental",
            method="TabenticsDiakrino",
            profile_type="experimental_portfolio",
            status="pending_gpu_run",
            device="gpu",
            skip_reason="requires public-gpu-host GPU validation before paired claims",
        ),
    ]
    for row in beyondarena_parity_inventory():
        status = str(row.get("availability", "unknown"))
        profiles.append(
            BeyondArenaRunProfile(
                profile_id=f"beyondarena_parity:{row['normalized_name']}",
                method=str(row["paper_name"]),
                profile_type="paper_parity",
                status=status,
                device=str(row.get("device", "")),
                backend=str(row.get("tabnetics_backend", "")),
                local_backend=str(row.get("local_backend", "")),
                skip_reason=str(row.get("skip_reason", "")) if status != "available" else "",
                sample_limit=str(row.get("sample_limit", "")),
                feature_limit=str(row.get("feature_limit", "")),
                tuning_mode=str(row.get("tuning_mode", "")),
                dependency=str(row.get("dependency", "")),
                install_hint=str(row.get("install_hint", "")),
                compatibility_scope=str(row.get("compatibility_scope", "")),
                execution_guard=str(row.get("execution_guard", "")),
                fallback_status=str(row.get("fallback_status", "")),
            )
        )
    return tuple(profiles)


def run_profiles_to_frame(profiles: Iterable[BeyondArenaRunProfile]) -> pd.DataFrame:
    """Convert run-profile dataclasses to a stable model manifest."""

    return pd.DataFrame.from_records([asdict(profile) for profile in profiles])


def _local_run_profiles(
    profiles: Optional[Iterable[BeyondArenaRunProfile]] = None,
) -> list[BeyondArenaRunProfile]:
    return [
        profile
        for profile in (profiles or default_beyondarena_run_profiles())
        if profile.profile_type in {"local_portfolio", "experimental_portfolio"}
    ]


def _execution_profiles_for_subset(
    profiles: Iterable[BeyondArenaRunProfile],
    *,
    subset: str,
) -> tuple[BeyondArenaRunProfile, ...]:
    normalized_subset = str(subset).strip().lower().replace("_", "-")
    if normalized_subset == "current-feasibility":
        return tuple(profile for profile in profiles if profile.profile_id == "tabnetics_current")
    return tuple(profiles)


def _local_execution_state(
    profile: BeyondArenaRunProfile,
    problem_type: Any,
    *,
    arch_ml_revalidated: bool = False,
    tabentics_diakrino_checkpoint_ready: bool = False,
) -> tuple[str, bool, str, str]:
    problem = str(problem_type or "").strip().lower()
    if problem != "classification":
        target_host = (
            "public-gpu-host"
            if profile.profile_type == "experimental_portfolio" or "gpu" in profile.device.strip().lower()
            else "public_cpu_host_1/public_cpu_host_2"
        )
        return (
            "skipped_unsupported_regression",
            False,
            (
                "current BeyondArena local execution path is classification-only; "
                "regression rows require a regression-capable tabnetics baseline before paired claims"
            ),
            target_host,
        )
    if profile.profile_type == "experimental_portfolio" or "gpu" in profile.device.strip().lower():
        if bool(arch_ml_revalidated) and bool(tabentics_diakrino_checkpoint_ready):
            return ("ready_after_artifact_materialization", True, "", "public-gpu-host")
        if bool(arch_ml_revalidated):
            return (
                "deferred_tabentics_diakrino_checkpoint",
                False,
                (
                    "requires a trained native Tabnetics Diakrino FS-classifier checkpoint "
                    "from #193/#192 before local Tabnetics Diakrino execution"
                ),
                "public-gpu-host",
            )
        return (
            "deferred_gpu_revalidation",
            False,
            "requires public-gpu-host GPU access/capacity revalidation before local Tabnetics Diakrino execution",
            "public-gpu-host",
        )
    return ("ready_after_artifact_materialization", True, "", "public_cpu_host_1/public_cpu_host_2")


def build_beyondarena_local_execution_plan(
    manifest: pd.DataFrame,
    *,
    profiles: Optional[Iterable[BeyondArenaRunProfile]] = None,
    arch_ml_revalidated: bool = False,
    tabentics_diakrino_checkpoint_ready: bool = False,
) -> pd.DataFrame:
    """Build explicit local execution/defer rows for tabnetics profiles.

    The current integrated tabnetics benchmark path is classification-only.  This
    plan therefore marks regression tasks as unsupported instead of allowing
    them to appear runnable in a local BeyondArena run handoff.
    """

    records: list[dict[str, Any]] = []
    for _, task in manifest.iterrows():
        for profile in _local_run_profiles(profiles):
            status, runnable, skip_reason, target_host = _local_execution_state(
                profile,
                task.get("problem_type"),
                arch_ml_revalidated=arch_ml_revalidated,
                tabentics_diakrino_checkpoint_ready=tabentics_diakrino_checkpoint_ready,
            )
            records.append(
                {
                    "dataset_id": task["dataset_id"],
                    "datafoundry_split_id": task["split_id"],
                    "official_split_id": task["official_split_id"],
                    "split": task["official_split_id"],
                    "task_type": task["task_type"],
                    "problem_type": task["problem_type"],
                    "metric": task["metric"],
                    "method": profile.method,
                    "model_profile": profile.profile_id,
                    "profile_type": profile.profile_type,
                    "execution_status": status,
                    "runnable": bool(runnable),
                    "target_host": target_host,
                    "required_device": profile.device,
                    "requires_materialized_artifact": True,
                    "artifact_revision": task["data_foundry_uri"],
                    "preprocessing_profile": "beyondarena_local_fallback",
                    "expected_result_status": "ok" if runnable else "skipped",
                    "skip_reason": skip_reason,
                }
            )
    return pd.DataFrame.from_records(records)


def build_pending_local_result_rows(
    manifest: pd.DataFrame,
    *,
    profiles: Optional[Iterable[BeyondArenaRunProfile]] = None,
    arch_ml_revalidated: bool = False,
    tabentics_diakrino_checkpoint_ready: bool = False,
) -> pd.DataFrame:
    """Build explicit skipped rows for deferred local BeyondArena profiles."""

    execution_plan = build_beyondarena_local_execution_plan(
        manifest,
        profiles=profiles,
        arch_ml_revalidated=bool(arch_ml_revalidated),
        tabentics_diakrino_checkpoint_ready=bool(tabentics_diakrino_checkpoint_ready),
    )
    records: list[dict[str, Any]] = []
    for _, row in execution_plan.iterrows():
        skip_reason = str(row.get("skip_reason", "") or "").strip()
        if not skip_reason:
            skip_reason = "not executed; requires materialized BeyondArena artifacts and lab execution"
        records.append(
            {
                "dataset_id": row["dataset_id"],
                "split_id": row["official_split_id"],
                "local_split_id": row["datafoundry_split_id"],
                "split": row["official_split_id"],
                "method": row["method"],
                "metric": row["metric"],
                "metric_value": pd.NA,
                "status": "skipped",
                "skip_reason": skip_reason,
                "execution_status": row["execution_status"],
                "model_profile": row["model_profile"],
                "device": row["required_device"],
                "execution_host": row["target_host"],
                "execution_lane": "gpu" if str(row["required_device"]).strip().lower() == "gpu" else "cpu",
                "allow_gpu_execution": bool(
                    arch_ml_revalidated and row["required_device"] == "gpu" and bool(row["runnable"])
                ),
                "origin": "tabnetics_local_beyondarena_pending",
                "lower_is_better": metric_lower_is_better(row["metric"]),
                "artifact_revision": row["artifact_revision"],
                "preprocessing_profile": row["preprocessing_profile"],
            }
        )
    return pd.DataFrame.from_records(records)


def _require_positive_int(value: Optional[int], name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required when emitting a BeyondArena run spec")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1 when emitting a BeyondArena run spec")
    return parsed


def _local_current_cpu_shard_plan(
    manifest: pd.DataFrame,
    *,
    host_allocation: Sequence[dict[str, Any]],
    shard_count: int,
    shard_base: Path,
) -> list[dict[str, Any]]:
    """Map local-current manifest shards onto the configured CPU host pods."""

    cpu_hosts = [row for row in host_allocation if str(row.get("role") or "") == "cpu"]
    slots: list[dict[str, Any]] = []
    for host in cpu_hosts:
        pods = int(host.get("PODS_PER_HOST", 0) or 0)
        max_workers = int(host.get("MAX_WORKERS", 0) or 0)
        workers_per_pod = int(max(1, max_workers // max(1, pods))) if pods > 0 and max_workers > 0 else 0
        for pod_index in range(pods):
            slots.append(
                {
                    "host": str(host.get("host") or ""),
                    "pod_index_on_host": int(pod_index),
                    "host_max_workers": int(max_workers),
                    "target_workers_per_pod": int(workers_per_pod),
                    "host_status": str(host.get("status") or ""),
                }
            )
    rows: list[dict[str, Any]] = []
    manifest_rows = manifest.reset_index(drop=True).copy()
    problem_values = (
        manifest_rows["problem_type"].fillna("").astype(str).str.strip().str.lower()
        if "problem_type" in manifest_rows.columns
        else pd.Series("", index=manifest_rows.index)
    )
    for shard_index in range(int(shard_count)):
        shard = manifest_rows.iloc[
            [idx for idx in range(len(manifest_rows)) if idx % int(shard_count) == int(shard_index)]
        ]
        shard_problem_values = problem_values.iloc[
            [idx for idx in range(len(manifest_rows)) if idx % int(shard_count) == int(shard_index)]
        ]
        slot = slots[shard_index % len(slots)] if slots else {}
        rows.append(
            {
                "backend": "tabnetics-current",
                "execution_lane": "cpu",
                "shard_index": int(shard_index),
                "shard_count": int(shard_count),
                "assigned_host": str(slot.get("host", "")),
                "pod_index_on_host": int(slot.get("pod_index_on_host", 0)),
                "host_max_workers": int(slot.get("host_max_workers", 0)),
                "target_workers_per_pod": int(slot.get("target_workers_per_pod", 0)),
                "host_status": str(slot.get("host_status", "")),
                "planned_manifest_rows": int(len(shard)),
                "planned_classification_rows": int(shard_problem_values.eq("classification").sum()),
                "out_csv": f"{shard_base}.shard_{shard_index}.csv",
            }
        )
    return rows


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    values = frame[column].fillna("").astype(str).str.strip()
    values = values[values.ne("")]
    return {str(k): int(v) for k, v in values.value_counts(dropna=False).to_dict().items()}


def _truthy_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _schema_only_local_result_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify local rows that are schema/accounting rows, not performance evidence."""

    mask = pd.Series(False, index=frame.index)
    if frame.empty:
        return mask
    if "execution_backend" in frame.columns:
        backend = frame["execution_backend"].fillna("").astype(str).str.strip().str.lower()
        mask = mask | backend.eq("sklearn-smoke")
    if "model_profile" in frame.columns:
        profile = frame["model_profile"].fillna("").astype(str).str.strip().str.lower()
        mask = mask | profile.eq("sklearn_smoke")
    if "method" in frame.columns:
        method = frame["method"].fillna("").astype(str).str.strip().str.lower()
        mask = mask | method.str.contains("smoke", regex=False)
    return mask.fillna(False).astype(bool)


def _manifest_split_stability_payload(manifest: pd.DataFrame) -> dict[str, Any]:
    if manifest.empty or "dataset_id" not in manifest.columns:
        return {
            "split_id_source": "official_split_id",
            "split_rows": int(len(manifest)),
            "unique_dataset_count": 0,
            "dataset_split_counts": {},
            "dataset_repeat_counts": {},
            "dataset_fold_counts": {},
            "datasets_with_multiple_splits": [],
            "datasets_with_single_split": [],
            "datasets_eligible_for_split_stability_claims": [],
            "datasets_excluded_from_split_stability_claims": [],
            "split_stability_claim_eligible_dataset_count": 0,
            "split_stability_claim_excluded_dataset_count": 0,
            "split_stability_claims_ready": False,
            "per_dataset_split_stability_ready": False,
            "split_stability_blocker": "manifest has no datasets",
            "split_stability_limit": "",
        }

    dataset_ids = manifest["dataset_id"].fillna("").astype(str)
    split_col = "official_split_id" if "official_split_id" in manifest.columns else "split_id"
    split_counts = (
        manifest.assign(_dataset_id=dataset_ids)
        .groupby("_dataset_id")[split_col]
        .nunique(dropna=True)
        .astype(int)
        .to_dict()
    )
    repeat_counts = (
        manifest.assign(_dataset_id=dataset_ids)
        .groupby("_dataset_id")["repeat"]
        .nunique(dropna=True)
        .astype(int)
        .to_dict()
        if "repeat" in manifest.columns
        else {}
    )
    fold_counts = (
        manifest.assign(_dataset_id=dataset_ids)
        .groupby("_dataset_id")["fold"]
        .nunique(dropna=True)
        .astype(int)
        .to_dict()
        if "fold" in manifest.columns
        else {}
    )
    split_counts = {str(key): int(value) for key, value in split_counts.items() if str(key)}
    repeat_counts = {str(key): int(value) for key, value in repeat_counts.items() if str(key)}
    fold_counts = {str(key): int(value) for key, value in fold_counts.items() if str(key)}
    multiple = sorted([dataset for dataset, count in split_counts.items() if int(count) > 1])
    single = sorted([dataset for dataset, count in split_counts.items() if int(count) <= 1])
    claim_ready = bool(multiple)
    all_dataset_ready = bool(split_counts) and not single
    return {
        "split_id_source": str(split_col),
        "split_rows": int(len(manifest)),
        "unique_dataset_count": int(len(split_counts)),
        "dataset_split_counts": dict(sorted(split_counts.items())),
        "dataset_repeat_counts": dict(sorted(repeat_counts.items())),
        "dataset_fold_counts": dict(sorted(fold_counts.items())),
        "datasets_with_multiple_splits": multiple,
        "datasets_with_single_split": single,
        "datasets_eligible_for_split_stability_claims": multiple,
        "datasets_excluded_from_split_stability_claims": single,
        "split_stability_claim_eligible_dataset_count": int(len(multiple)),
        "split_stability_claim_excluded_dataset_count": int(len(single)),
        "split_stability_claims_ready": claim_ready,
        "per_dataset_split_stability_ready": all_dataset_ready,
        "split_stability_blocker": ""
        if claim_ready
        else "per-dataset split-stability claims require more than one planned split per dataset",
        "split_stability_limit": (
            "single-split datasets must be excluded from per-dataset split-stability claims"
            if single
            else ""
        ),
    }


def _manifest_coverage_payload(manifest: pd.DataFrame, *, subset: str) -> dict[str, Any]:
    task_type = (
        manifest["task_type"].fillna("").astype(str).str.lower()
        if "task_type" in manifest
        else pd.Series(dtype=str)
    )
    dimensionality = (
        manifest["dimensionality"].fillna("").astype(str).str.lower()
        if "dimensionality" in manifest
        else pd.Series(dtype=str)
    )
    coverage = {
        "iid": bool(task_type.eq("iid").any()),
        "grouped": bool(task_type.eq("grouped").any()),
        "temporal": bool(task_type.eq("temporal").any()),
        "text": bool(_truthy_series(manifest, "has_text").any()),
        "high_cardinality": bool(_truthy_series(manifest, "high_cardinality").any()),
        "high_dimensional": bool(dimensionality.eq("high").any()),
    }
    missing = sorted([name for name, covered in coverage.items() if not covered])
    return {
        "subset": str(subset),
        "task_count": int(len(manifest)),
        "unique_dataset_count": int(manifest["dataset_id"].nunique()) if "dataset_id" in manifest.columns else 0,
        "task_type_counts": _value_counts(manifest, "task_type"),
        "problem_type_counts": _value_counts(manifest, "problem_type"),
        "size_tier_counts": _value_counts(manifest, "size_tier"),
        "dimensionality_counts": _value_counts(manifest, "dimensionality"),
        "text_task_count": int(_truthy_series(manifest, "has_text").sum()),
        "high_cardinality_task_count": int(_truthy_series(manifest, "high_cardinality").sum()),
        "smoke_coverage": coverage,
        "smoke_coverage_ready": bool(not missing) if str(subset).strip().lower() == "smoke" else False,
        "smoke_missing_facets": missing if str(subset).strip().lower() == "smoke" else [],
        "split_stability": _manifest_split_stability_payload(manifest),
    }


def _local_result_status_payload(raw: pd.DataFrame, aligned: pd.DataFrame) -> dict[str, Any]:
    aligned_status = (
        aligned["status"].fillna("ok").astype(str).str.lower()
        if "status" in aligned.columns
        else pd.Series(["ok"] * len(aligned), index=aligned.index)
    )
    schema_only = _schema_only_local_result_mask(aligned)
    ok_mask = aligned_status.eq("ok")
    manifest_key_cols = ["dataset_id", "split_id", "metric"]
    manifest_key_rows = (
        int(len(aligned[manifest_key_cols].drop_duplicates()))
        if all(col in aligned.columns for col in manifest_key_cols)
        else 0
    )
    claim_eligible_ok_mask = (~schema_only) & ok_mask
    claim_eligible_ok_key_rows = (
        int(len(aligned.loc[claim_eligible_ok_mask, manifest_key_cols].drop_duplicates()))
        if all(col in aligned.columns for col in manifest_key_cols)
        else 0
    )
    payload: dict[str, Any] = {
        "local_result_rows": int(len(raw)),
        "local_manifest_aligned_rows": int(len(aligned)),
        "local_manifest_key_rows": manifest_key_rows,
        "local_ok_rows": int(ok_mask.sum()),
        "local_schema_only_rows": int(schema_only.sum()),
        "local_schema_only_ok_rows": int((schema_only & ok_mask).sum()),
        "local_claim_eligible_ok_rows": int(claim_eligible_ok_mask.sum()),
        "local_claim_eligible_ok_key_rows": claim_eligible_ok_key_rows,
        "local_status_counts": _value_counts(aligned, "status"),
        "local_method_counts": _value_counts(aligned, "method"),
        "local_execution_status_counts": _value_counts(aligned, "execution_status"),
        "local_execution_backend_counts": _value_counts(aligned, "execution_backend"),
        "local_skip_reason_counts": _value_counts(aligned, "skip_reason"),
    }
    if "dataset_id" in aligned.columns:
        payload["local_unique_dataset_count"] = int(aligned["dataset_id"].nunique())
    return payload


def _status_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    try:
        if pd.isna(value):
            return 0
    except TypeError:
        pass
    return int(value or 0)


def _materialization_audit_payload(run_spec_payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    if run_spec_payload is None:
        return {
            "configured": False,
            "plan_csv": "",
            "artifact_count": 0,
            "materialization_ready": False,
            "local_runner_ready": False,
            "materialization_blocker": "run_spec.json was not emitted",
        }
    outputs = dict(run_spec_payload.get("outputs") or {})
    raw_plan = str(outputs.get("materialization_plan", "") or "").strip()
    if not raw_plan:
        return {
            "configured": False,
            "plan_csv": "",
            "artifact_count": 0,
            "materialization_ready": False,
            "local_runner_ready": False,
            "materialization_blocker": "run_spec does not name a materialization_plan output",
        }
    plan_csv = Path(raw_plan)
    if not plan_csv.exists():
        return {
            "configured": False,
            "plan_csv": str(plan_csv),
            "artifact_count": 0,
            "materialization_ready": False,
            "local_runner_ready": False,
            "materialization_blocker": "materialization_plan.csv not found; run materialize_dry_run first",
        }
    try:
        plan = pd.read_csv(plan_csv)
        summary = summarize_beyondarena_materialization(plan)
    except Exception as exc:
        return {
            "configured": True,
            "plan_csv": str(plan_csv),
            "artifact_count": 0,
            "materialization_ready": False,
            "local_runner_ready": False,
            "materialization_blocker": f"materialization_plan.csv unreadable: {type(exc).__name__}: {exc}",
        }
    return {
        "configured": True,
        "plan_csv": str(plan_csv),
        **summary,
    }


def _readiness_audit_payload(
    status_payload: dict[str, Any],
    manifest: pd.DataFrame,
    *,
    execution_plan: Optional[pd.DataFrame] = None,
    run_spec_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a compact machine-readable acceptance/readiness audit."""

    coverage = status_payload.get("manifest_coverage", {})
    split_stability = coverage.get("split_stability", {}) if isinstance(coverage, dict) else {}
    split_stability_claims_ready = bool(
        split_stability.get(
            "split_stability_claims_ready",
            split_stability.get("per_dataset_split_stability_ready", False),
        )
    )
    subset = str(status_payload.get("subset", ""))
    local_results_source = str(status_payload.get("local_results_source", "") or "")
    local_results_configured = local_results_source not in {"", "not configured"}
    exact_official_ready = bool(status_payload.get("exact_paired_rows_available"))
    joined_rows = _status_int(status_payload, "joined_rows")
    manifest_rows = int(len(manifest))
    local_manifest_aligned_rows = _status_int(status_payload, "local_manifest_aligned_rows")
    local_manifest_key_rows = _status_int(status_payload, "local_manifest_key_rows")
    local_ok_rows = _status_int(status_payload, "local_ok_rows")
    local_schema_only_ok_rows = _status_int(status_payload, "local_schema_only_ok_rows")
    local_claim_eligible_ok_rows = _status_int(status_payload, "local_claim_eligible_ok_rows")
    local_claim_eligible_ok_key_rows = _status_int(
        status_payload,
        "local_claim_eligible_ok_key_rows",
    )
    metric_contract_valid = bool(status_payload.get("metric_contract_valid", False))
    comparison_ready = (
        bool(status_payload.get("comparison_ready"))
        and joined_rows > 0
        and metric_contract_valid
    )
    materialization_status = _materialization_audit_payload(run_spec_payload)
    materialization_artifact_count = _status_int(materialization_status, "artifact_count")
    artifact_plan_ready = bool(
        materialization_artifact_count > 0
        and _status_int(materialization_status, "artifact_plan_ready_count") == materialization_artifact_count
    )

    execution_status_counts = (
        _value_counts(execution_plan, "execution_status") if execution_plan is not None else {}
    )
    target_host_counts = _value_counts(execution_plan, "target_host") if execution_plan is not None else {}
    runnable_rows = (
        int(_truthy_series(execution_plan, "runnable").sum()) if execution_plan is not None else 0
    )
    unsupported_regression_rows = int(execution_status_counts.get("skipped_unsupported_regression", 0))
    deferred_gpu_rows = int(execution_status_counts.get("deferred_gpu_revalidation", 0))
    deferred_tabentics_diakrino_checkpoint_rows = int(
        execution_status_counts.get("deferred_tabentics_diakrino_checkpoint", 0)
    )
    ready_after_materialization_rows = int(
        execution_status_counts.get("ready_after_artifact_materialization", 0)
    )

    host_allocation = []
    if run_spec_payload is not None:
        host_allocation = list(run_spec_payload.get("host_allocation", []))
    host_status_counts: dict[str, int] = {}
    for row in host_allocation:
        status = str(row.get("status", "") or "").strip()
        if status:
            host_status_counts[status] = host_status_counts.get(status, 0) + 1
    live_host_capacity_ready = bool(host_allocation) and not any(
        str(row.get("status", "") or "").startswith("requires_live")
        for row in host_allocation
        if row.get("role") == "cpu"
    )
    gpu_revalidation_ready = deferred_gpu_rows == 0 and not any(
        str(row.get("status", "") or "") == "deferred_gpu_revalidation" for row in host_allocation
    )
    prelaunch_gates = list(run_spec_payload.get("prelaunch_gates", [])) if run_spec_payload is not None else []
    tabentics_diakrino_checkpoint_ready = deferred_tabentics_diakrino_checkpoint_rows == 0 and not any(
        str(row.get("status", "") or "") == "deferred_tabentics_diakrino_checkpoint"
        for row in prelaunch_gates
    )

    performance_claim_blockers: list[str] = []
    if not exact_official_ready:
        performance_claim_blockers.append(
            "exact official per-dataset/per-split result rows are not configured"
        )
    if not local_results_configured:
        performance_claim_blockers.append("local result rows are not configured/materialized")
    elif local_manifest_key_rows < manifest_rows:
        performance_claim_blockers.append(
            f"local result rows cover only {local_manifest_key_rows}/{manifest_rows} manifest rows"
        )
    elif local_ok_rows <= 0:
        performance_claim_blockers.append("local result rows contain no manifest-aligned ok rows")
    elif local_schema_only_ok_rows > 0:
        performance_claim_blockers.append(
            "local ok rows include schema-only smoke/accounting rows; filter to claim-eligible local rows before performance claims"
        )
    elif local_claim_eligible_ok_rows <= 0:
        performance_claim_blockers.append("local result rows contain no claim-eligible ok rows")
    elif local_claim_eligible_ok_key_rows < manifest_rows:
        performance_claim_blockers.append(
            "claim-eligible ok local result rows cover only "
            f"{local_claim_eligible_ok_key_rows}/{manifest_rows} manifest rows"
        )
    if not comparison_ready:
        comparison_blocker = str(status_payload.get("comparison_blocker", "") or "").strip()
        if joined_rows > 0 and not metric_contract_valid:
            comparison_blocker = (
                "paired rows exist but do not satisfy the explicit BeyondArena metric contract"
            )
        performance_claim_blockers.append(
            comparison_blocker
            or "no exact paired comparison rows are available for local-vs-official claims"
        )
    if subset == "current-feasibility":
        performance_claim_blockers.append(
            "current-feasibility subset is runner/provenance evidence only; "
            "it does not satisfy Stage-1 smoke, Stage-2 core, or portfolio-level performance claims"
        )
    elif subset == "smoke":
        performance_claim_blockers.append(
            "smoke subset is a Stage-1 correctness/feasibility check only; "
            "it does not satisfy Stage-2 full-catalog or portfolio-level performance claims"
        )

    split_claim_blockers: list[str] = []
    if not split_stability_claims_ready:
        blocker = str(split_stability.get("split_stability_blocker", "") or "").strip()
        split_claim_blockers.append(
            blocker
            or "per-dataset split-stability claims require multiple planned split rows per dataset"
        )

    launch_blockers: list[str] = []
    if run_spec_payload is None:
        launch_blockers.append("run_spec.json was not emitted for lab handoff")
    if execution_plan is None:
        launch_blockers.append("local_execution_plan.csv was not emitted for lab handoff")
    if ready_after_materialization_rows > 0:
        if not bool(materialization_status.get("configured", False)):
            launch_blockers.append("materialization_plan.csv was not emitted for current CPU rows")
        elif not bool(materialization_status.get("local_runner_ready", False)):
            launch_blockers.append(
                "current tabnetics rows require materialized BeyondArena artifacts before local execution"
            )
    if host_allocation and not live_host_capacity_ready:
        launch_blockers.append("live lab CPU capacity checks are not recorded")
    if deferred_gpu_rows > 0 or any(
        str(row.get("status", "") or "") == "deferred_gpu_revalidation" for row in host_allocation
    ):
        launch_blockers.append("GPU-required DIAKRINO rows remain deferred pending public-gpu-host revalidation")
    if deferred_tabentics_diakrino_checkpoint_rows > 0:
        launch_blockers.append(
            "native Tabnetics Diakrino rows remain deferred pending trained FS-classifier checkpoint from #193/#192"
        )
    if unsupported_regression_rows > 0:
        launch_blockers.append(
            "BeyondArena regression rows are unsupported by the tabnetics/DIAKRINO local execution plan"
        )

    performance_claims_ready = bool(comparison_ready and not performance_claim_blockers)
    return {
        "task_id": "T-BEYONDARENA-EPIC",
        "epic_issue": "#170",
        "validation_issue": "#178",
        "subset": subset,
        "overall_status": "ready_for_paired_claim_review" if performance_claims_ready else "blocked",
        "acceptance_checks": {
            "manifest_rows_ready": bool(len(manifest) > 0),
            "smoke_facets_ready": bool(coverage.get("smoke_coverage_ready", False))
            if subset == "smoke"
            else None,
            "official_exact_results_ready": exact_official_ready,
            "local_result_rows_ready": bool(local_ok_rows > 0),
            "paired_comparison_ready": comparison_ready,
            "metric_contract_valid": metric_contract_valid,
            "split_stability_claims_ready": split_stability_claims_ready,
            "local_execution_plan_emitted": execution_plan is not None,
            "run_spec_emitted": run_spec_payload is not None,
            "artifact_plan_ready": artifact_plan_ready,
            "artifact_materialization_ready": bool(materialization_status.get("materialization_ready", False)),
            "artifact_local_runner_ready": bool(materialization_status.get("local_runner_ready", False)),
            "live_host_capacity_ready": live_host_capacity_ready,
            "gpu_revalidation_ready": gpu_revalidation_ready,
            "tabentics_diakrino_checkpoint_ready": tabentics_diakrino_checkpoint_ready,
        },
        "manifest": {
            "task_count": int(len(manifest)),
            "unique_dataset_count": int(manifest["dataset_id"].nunique())
            if "dataset_id" in manifest.columns
            else 0,
            "coverage": coverage,
        },
        "official_results": {
            "source": str(status_payload.get("source", "")),
            "available": bool(status_payload.get("available")),
            "exact_paired_rows_available": exact_official_ready,
            "row_count": _status_int(status_payload, "row_count"),
        },
        "local_results": {
            "source": local_results_source or "not configured",
            "configured": local_results_configured,
            "row_count": _status_int(status_payload, "local_result_rows"),
            "manifest_aligned_rows": local_manifest_aligned_rows,
            "manifest_key_rows": local_manifest_key_rows,
            "ok_rows": local_ok_rows,
            "schema_only_rows": _status_int(status_payload, "local_schema_only_rows"),
            "schema_only_ok_rows": local_schema_only_ok_rows,
            "claim_eligible_ok_rows": local_claim_eligible_ok_rows,
            "claim_eligible_ok_key_rows": local_claim_eligible_ok_key_rows,
            "status_counts": status_payload.get("local_status_counts", {}),
            "execution_status_counts": status_payload.get("local_execution_status_counts", {}),
            "backend_counts": status_payload.get("local_execution_backend_counts", {}),
            "skip_reason_counts": status_payload.get("local_skip_reason_counts", {}),
        },
        "local_execution_plan": {
            "emitted": execution_plan is not None,
            "rows": int(len(execution_plan)) if execution_plan is not None else 0,
            "runnable_rows_after_materialization": runnable_rows,
            "ready_after_artifact_materialization_rows": ready_after_materialization_rows,
            "unsupported_regression_rows": unsupported_regression_rows,
            "deferred_gpu_revalidation_rows": deferred_gpu_rows,
            "deferred_tabentics_diakrino_checkpoint_rows": deferred_tabentics_diakrino_checkpoint_rows,
            "execution_status_counts": execution_status_counts,
            "target_host_counts": target_host_counts,
        },
        "artifact_materialization": materialization_status,
        "run_spec": {
            "emitted": run_spec_payload is not None,
            "host_status_counts": host_status_counts,
            "live_host_capacity_ready": live_host_capacity_ready,
            "gpu_revalidation_ready": gpu_revalidation_ready,
            "tabentics_diakrino_checkpoint_ready": tabentics_diakrino_checkpoint_ready,
            "command_sequence": list(run_spec_payload.get("command_sequence", []))
            if run_spec_payload is not None
            else [],
        },
        "claims": {
            "performance_claims_ready": performance_claims_ready,
            "split_stability_claims_ready": split_stability_claims_ready,
            "claims_policy": (
                "No production default promotion or BeyondArena performance claim is valid without "
                "exact paired local-vs-official rows and portfolio-level validation evidence."
            ),
        },
        "blockers": {
            "performance_claims": performance_claim_blockers,
            "split_stability_claims": split_claim_blockers,
            "lab_launch": launch_blockers,
        },
    }


def build_beyondarena_run_spec(
    manifest: pd.DataFrame,
    execution_plan: pd.DataFrame,
    *,
    subset: str,
    out_dir: str | Path,
    artifact_root: str | Path,
    local_results_source: Optional[str | Path] = None,
    comparison_out_dir: Optional[str | Path] = None,
    arch_ml_revalidated: bool = False,
    public_cpu_host_1_max_workers: Optional[int] = None,
    public_cpu_host_1_pods_per_host: Optional[int] = None,
    public_cpu_host_2_max_workers: Optional[int] = None,
    public_cpu_host_2_pods_per_host: Optional[int] = None,
    cpu_capacity_revalidated: bool = False,
    arch_ml_max_workers: Optional[int] = None,
    arch_ml_pods_per_host: Optional[int] = None,
    tabentics_diakrino_checkpoint_ready: bool = False,
) -> dict[str, Any]:
    """Build a concrete, non-executing BeyondArena validation run spec."""

    root = Path(out_dir)
    artifact_root = Path(artifact_root)
    normalized_subset = str(subset).strip().lower().replace("_", "-")
    include_diakrino_lanes = normalized_subset != "current-feasibility"
    local_current_results = (
        Path(local_results_source)
        if local_results_source is not None
        else root.parent / "local_current_results.csv"
    )
    local_tabpfn_candidate_rows = root.parent / "local_tabpfn_candidate_rows.csv"
    local_tabentics_diakrino_rows = root.parent / "local_tabentics_diakrino_rows.csv"
    local_current_shard_base = root.parent / "local_current_results"
    local_tabpfn_candidate_shard_base = root.parent / "local_tabpfn_candidate_rows"
    local_tabentics_diakrino_shard_base = root.parent / "local_tabentics_diakrino_rows"
    compare_current_out = (
        Path(comparison_out_dir)
        if comparison_out_dir is not None
        else root.parent / f"{subset}_compare_current"
    )
    compare_tabpfn_candidate_out = root.parent / f"{subset}_compare_tabpfn_candidate"
    compare_tabentics_diakrino_out = root.parent / f"{subset}_compare_tabentics_diakrino"
    cpu_host_status = (
        "live_capacity_recorded"
        if bool(cpu_capacity_revalidated)
        else "requires_live_capacity_check_before_launch"
    )
    host_allocation: list[dict[str, Any]] = [
        {
            "host": "host1.example.com",
            "role": "cpu",
            "target": "CPU shards for current tabnetics classification rows",
            "MAX_WORKERS": _require_positive_int(public_cpu_host_1_max_workers, "public_cpu_host_1_max_workers"),
            "PODS_PER_HOST": _require_positive_int(public_cpu_host_1_pods_per_host, "public_cpu_host_1_pods_per_host"),
            "status": cpu_host_status,
        },
        {
            "host": "host2.example.com",
            "role": "cpu",
            "target": "CPU shards for current tabnetics classification rows",
            "MAX_WORKERS": _require_positive_int(public_cpu_host_2_max_workers, "public_cpu_host_2_max_workers"),
            "PODS_PER_HOST": _require_positive_int(public_cpu_host_2_pods_per_host, "public_cpu_host_2_pods_per_host"),
            "status": cpu_host_status,
        },
    ]
    if include_diakrino_lanes and bool(arch_ml_revalidated):
        host_allocation.append(
            {
                "host": "public-gpu-host",
                "role": "gpu",
                "target": "GPU-required TabPFN/Tabnetics Diakrino shards",
                "MAX_WORKERS": _require_positive_int(arch_ml_max_workers, "arch_ml_max_workers"),
                "PODS_PER_HOST": _require_positive_int(arch_ml_pods_per_host, "arch_ml_pods_per_host"),
                "status": "live_gpu_env_recorded",
            }
        )
    elif include_diakrino_lanes:
        host_allocation.append(
            {
                "host": "public-gpu-host",
                "role": "gpu",
                "target": "GPU-required TabPFN/Tabnetics Diakrino shards",
                "MAX_WORKERS": 0,
                "PODS_PER_HOST": 0,
                "status": "deferred_gpu_revalidation",
            }
        )

    runnable_counts = (
        execution_plan.groupby(["method", "execution_status"], dropna=False)
        .size()
        .reset_index(name="rows")
        .to_dict(orient="records")
        if not execution_plan.empty
        else []
    )
    def quote(value: object) -> str:
        return shlex.quote(str(value))

    plan_parts = [
        "python",
        "-m",
        "tabnetics.benchmarks.beyondarena_plan",
        "--subset",
        str(subset),
        "--official-results",
        "public-r2",
        "--emit-local-execution-plan",
        "--emit-run-spec",
        "--artifact-root",
        str(artifact_root),
        "--public_cpu_host_1-max-workers",
        str(host_allocation[0]["MAX_WORKERS"]),
        "--public_cpu_host_1-pods-per-host",
        str(host_allocation[0]["PODS_PER_HOST"]),
        "--public_cpu_host_2-max-workers",
        str(host_allocation[1]["MAX_WORKERS"]),
        "--public_cpu_host_2-pods-per-host",
        str(host_allocation[1]["PODS_PER_HOST"]),
        "--out-dir",
        str(root),
    ]
    if local_results_source is not None:
        plan_parts.extend(["--local-results", str(local_current_results)])
    if bool(cpu_capacity_revalidated):
        plan_parts.append("--cpu-capacity-revalidated")
    if include_diakrino_lanes and bool(arch_ml_revalidated):
        plan_parts.extend(
            [
                "--public-gpu-host-revalidated",
                "--public-gpu-host-max-workers",
                str(host_allocation[2]["MAX_WORKERS"]),
                "--public-gpu-host-pods-per-host",
                str(host_allocation[2]["PODS_PER_HOST"]),
            ]
        )
    if include_diakrino_lanes and bool(tabentics_diakrino_checkpoint_ready):
        plan_parts.append("--tabnetics-diakrino-checkpoint-ready")
    gpu_allow_flag = " --allow-gpu-execution" if bool(arch_ml_revalidated) else ""
    tabentics_diakrino_allow_flag = (
        " --allow-gpu-execution"
        if bool(arch_ml_revalidated and tabentics_diakrino_checkpoint_ready)
        else ""
    )
    cpu_shard_count = int(host_allocation[0]["PODS_PER_HOST"]) + int(host_allocation[1]["PODS_PER_HOST"])
    gpu_shard_count = (
        int(host_allocation[2]["PODS_PER_HOST"])
        if include_diakrino_lanes and len(host_allocation) > 2 and int(host_allocation[2]["PODS_PER_HOST"]) > 0
        else 1
    )
    local_current_shard_plan = _local_current_cpu_shard_plan(
        manifest,
        host_allocation=host_allocation,
        shard_count=cpu_shard_count,
        shard_base=local_current_shard_base,
    )
    current_shard_target_workers = (
        int(local_current_shard_plan[0]["target_workers_per_pod"])
        if local_current_shard_plan
        else 1
    )
    commands = {
        "plan": " ".join(quote(part) for part in plan_parts),
        "materialize_dry_run": (
            "python -m tabnetics.benchmarks.beyondarena_materialize "
            f"--task-manifest-csv {quote(root / 'beyondarena_task_manifest.csv')} "
            f"--out-dir {quote(artifact_root)} --dry-run --include-dataset --fetch-size-metadata"
        ),
        "materialize_dataset": (
            "python -m tabnetics.benchmarks.beyondarena_materialize "
            f"--task-manifest-csv {quote(root / 'beyondarena_task_manifest.csv')} "
            f"--out-dir {quote(artifact_root)} --include-dataset --require-local-runner-ready"
        ),
        "local_current": (
            "python -m tabnetics.benchmarks.beyondarena_local "
            f"--artifact-root {quote(artifact_root)} "
            f"--task-manifest-csv {quote(root / 'beyondarena_task_manifest.csv')} "
            "--backend tabnetics-current --device cpu --execution-host public_cpu_host_1/public_cpu_host_2 "
            f"--execution-lane cpu --max-workers {current_shard_target_workers} "
            f"--out-csv {quote(local_current_results)}"
        ),
        "local_current_sharded": (
            "python -m tabnetics.benchmarks.beyondarena_local "
            f"--artifact-root {quote(artifact_root)} "
            f"--task-manifest-csv {quote(root / 'beyondarena_task_manifest.csv')} "
            "--backend tabnetics-current --device cpu --execution-host public_cpu_host_1/public_cpu_host_2 "
            f"--max-workers \"${{BEYONDARENA_MAX_WORKERS:-{current_shard_target_workers}}}\" "
            "--execution-lane cpu --manifest-shard-index \"${BEYONDARENA_SHARD_INDEX}\" "
            f"--manifest-shard-count {cpu_shard_count} "
            f"--out-csv {quote(local_current_shard_base)}.shard_\"${{BEYONDARENA_SHARD_INDEX}}\".csv"
        ),
        "merge_current_shards": (
            "python -m tabnetics.benchmarks.beyondarena_local "
            f"--merge-shard-glob {quote(root.parent / 'local_current_results.shard_*.csv')} "
            f"--out-csv {quote(local_current_results)}"
        ),
        "compare_current": (
            "python -m tabnetics.benchmarks.beyondarena_plan "
            f"--subset {subset} --official-results public-r2 "
            f"--local-results {quote(local_current_results)} --out-dir {quote(compare_current_out)}"
        ),
    }
    if include_diakrino_lanes:
        commands.update(
            {
                "local_tabpfn_candidate": (
                    "python -m tabnetics.benchmarks.beyondarena_local "
                    f"--artifact-root {quote(artifact_root)} "
                    f"--task-manifest-csv {quote(root / 'beyondarena_task_manifest.csv')} "
                    f"--backend tabpfn-candidate{gpu_allow_flag} --device gpu --execution-host public-gpu-host "
                    f"--execution-lane gpu --out-csv {quote(local_tabpfn_candidate_rows)}"
                ),
                "local_tabpfn_candidate_sharded": (
                    "python -m tabnetics.benchmarks.beyondarena_local "
                    f"--artifact-root {quote(artifact_root)} "
                    f"--task-manifest-csv {quote(root / 'beyondarena_task_manifest.csv')} "
                    f"--backend tabpfn-candidate{gpu_allow_flag} --device gpu --execution-host public-gpu-host "
                    "--execution-lane gpu --manifest-shard-index \"${BEYONDARENA_SHARD_INDEX}\" "
                    f"--manifest-shard-count {gpu_shard_count} "
                    f"--out-csv {quote(local_tabpfn_candidate_shard_base)}.shard_\"${{BEYONDARENA_SHARD_INDEX}}\".csv"
                ),
                "merge_tabpfn_candidate_shards": (
                    "python -m tabnetics.benchmarks.beyondarena_local "
                    f"--merge-shard-glob {quote(root.parent / 'local_tabpfn_candidate_rows.shard_*.csv')} "
                    f"--out-csv {quote(local_tabpfn_candidate_rows)}"
                ),
                "local_tabentics_diakrino": (
                    "python -m tabnetics.benchmarks.beyondarena_local "
                    f"--artifact-root {quote(artifact_root)} "
                    f"--task-manifest-csv {quote(root / 'beyondarena_task_manifest.csv')} "
                    f"--backend tabnetics-diakrino{tabentics_diakrino_allow_flag} --device gpu --execution-host public-gpu-host "
                    "--tabnetics-diakrino-checkpoint ${TABENTICS_DIAKRINO_CHECKPOINT} "
                    "--tabnetics-diakrino-max-features 1024 --tabnetics-diakrino-batch-size 128 "
                    f"--execution-lane gpu --out-csv {quote(local_tabentics_diakrino_rows)}"
                ),
                "local_tabentics_diakrino_sharded": (
                    "python -m tabnetics.benchmarks.beyondarena_local "
                    f"--artifact-root {quote(artifact_root)} "
                    f"--task-manifest-csv {quote(root / 'beyondarena_task_manifest.csv')} "
                    f"--backend tabnetics-diakrino{tabentics_diakrino_allow_flag} --device gpu --execution-host public-gpu-host "
                    "--tabnetics-diakrino-checkpoint ${TABENTICS_DIAKRINO_CHECKPOINT} "
                    "--tabnetics-diakrino-max-features 1024 --tabnetics-diakrino-batch-size 128 "
                    "--execution-lane gpu --manifest-shard-index \"${BEYONDARENA_SHARD_INDEX}\" "
                    f"--manifest-shard-count {gpu_shard_count} "
                    f"--out-csv {quote(local_tabentics_diakrino_shard_base)}.shard_\"${{BEYONDARENA_SHARD_INDEX}}\".csv"
                ),
                "merge_tabentics_diakrino_shards": (
                    "python -m tabnetics.benchmarks.beyondarena_local "
                    f"--merge-shard-glob {quote(root.parent / 'local_tabentics_diakrino_rows.shard_*.csv')} "
                    f"--out-csv {quote(local_tabentics_diakrino_rows)}"
                ),
                "compare_tabpfn_candidate": (
                    "python -m tabnetics.benchmarks.beyondarena_plan "
                    f"--subset {subset} --official-results public-r2 "
                    f"--local-results {quote(local_tabpfn_candidate_rows)} "
                    f"--out-dir {quote(compare_tabpfn_candidate_out)}"
                ),
                "compare_tabentics_diakrino": (
                    "python -m tabnetics.benchmarks.beyondarena_plan "
                    f"--subset {subset} --official-results public-r2 "
                    f"--local-results {quote(local_tabentics_diakrino_rows)} "
                    f"--out-dir {quote(compare_tabentics_diakrino_out)}"
                ),
            }
        )
    comparison_dirs = {"current": str(compare_current_out)}
    comparison_joined_pairs = {"current": str(compare_current_out / "joined_pairs.csv")}
    comparison_summaries = {"current": str(compare_current_out / "summary.csv")}
    if include_diakrino_lanes:
        comparison_dirs.update(
            {
                "tabpfn_candidate": str(compare_tabpfn_candidate_out),
                "tabentics_diakrino": str(compare_tabentics_diakrino_out),
            }
        )
        comparison_joined_pairs.update(
            {
                "tabpfn_candidate": str(compare_tabpfn_candidate_out / "joined_pairs.csv"),
                "tabentics_diakrino": str(compare_tabentics_diakrino_out / "joined_pairs.csv"),
            }
        )
        comparison_summaries.update(
            {
                "tabpfn_candidate": str(compare_tabpfn_candidate_out / "summary.csv"),
                "tabentics_diakrino": str(compare_tabentics_diakrino_out / "summary.csv"),
            }
        )
    command_sequence = [
        "plan",
        "materialize_dry_run",
        "materialize_dataset",
        "local_current",
        "compare_current",
    ]
    sharded_command_sequence = [
        "plan",
        "materialize_dry_run",
        "materialize_dataset",
        "local_current_sharded",
        "merge_current_shards",
        "compare_current",
    ]
    if include_diakrino_lanes:
        command_sequence.extend(
            [
                "local_tabpfn_candidate",
                "compare_tabpfn_candidate",
                "local_tabentics_diakrino",
                "compare_tabentics_diakrino",
            ]
        )
        sharded_command_sequence.extend(
            [
                "local_tabpfn_candidate_sharded",
                "merge_tabpfn_candidate_shards",
                "compare_tabpfn_candidate",
                "local_tabentics_diakrino_sharded",
                "merge_tabentics_diakrino_shards",
                "compare_tabentics_diakrino",
            ]
        )
    prelaunch_gates = [
        {
            "gate": "materialization_local_runner_ready",
            "command": "materialize_dataset",
            "required": True,
            "ready_field": "artifact_local_runner_ready",
            "status": "must_pass_before_local_result_rows",
        },
        {
            "gate": "live_cpu_capacity",
            "command": "plan",
            "required": True,
            "ready_field": "host_allocation.MAX_WORKERS/PODS_PER_HOST",
            "status": "recorded" if bool(cpu_capacity_revalidated) else "must_be_recorded_before_lab_execution",
        },
        {
            "gate": "sharded_result_claiming",
            "command": "local_current/local_tabpfn_candidate/local_tabentics_diakrino",
            "required": True,
            "ready_field": "manifest_shard_index/manifest_shard_count plus per-output CSV locks",
            "status": "supported_by_manifest_shard_flags_and_output_locks",
        },
    ]
    if include_diakrino_lanes:
        prelaunch_gates.append(
            {
                "gate": "gpu_revalidation",
                "command": "local_tabpfn_candidate/local_tabentics_diakrino",
                "required": bool(arch_ml_revalidated),
                "ready_field": "arch_ml_revalidated",
                "status": "recorded" if bool(arch_ml_revalidated) else "deferred_unless_arch_ml_revalidated",
            }
        )
        prelaunch_gates.append(
            {
                "gate": "tabentics_diakrino_checkpoint",
                "command": "local_tabentics_diakrino",
                "required": bool(tabentics_diakrino_checkpoint_ready),
                "ready_field": "TABENTICS_DIAKRINO_CHECKPOINT and trained classifier provenance from #193/#192",
                "status": "recorded" if bool(tabentics_diakrino_checkpoint_ready) else "deferred_tabentics_diakrino_checkpoint",
            }
        )
    outputs = {
        "materialization_plan": str(artifact_root / "materialization_plan.csv"),
        "local_current_results": str(local_current_results),
        "local_current_shard_glob": str(root.parent / "local_current_results.shard_*.csv"),
        "local_current_shard_template": str(
            root.parent / "local_current_results.shard_${BEYONDARENA_SHARD_INDEX}.csv"
        ),
        "comparison_dir": str(compare_current_out),
        "joined_pairs": str(compare_current_out / "joined_pairs.csv"),
        "summary": str(compare_current_out / "summary.csv"),
        "comparison_dirs": comparison_dirs,
        "comparison_joined_pairs": comparison_joined_pairs,
        "comparison_summaries": comparison_summaries,
    }
    if include_diakrino_lanes:
        outputs.update(
            {
                "local_tabpfn_candidate_rows": str(local_tabpfn_candidate_rows),
                "local_tabentics_diakrino_rows": str(local_tabentics_diakrino_rows),
                "local_tabpfn_candidate_shard_glob": str(root.parent / "local_tabpfn_candidate_rows.shard_*.csv"),
                "local_tabpfn_candidate_shard_template": str(
                    root.parent / "local_tabpfn_candidate_rows.shard_${BEYONDARENA_SHARD_INDEX}.csv"
                ),
                "local_tabentics_diakrino_shard_glob": str(root.parent / "local_tabentics_diakrino_rows.shard_*.csv"),
                "local_tabentics_diakrino_shard_template": str(
                    root.parent / "local_tabentics_diakrino_rows.shard_${BEYONDARENA_SHARD_INDEX}.csv"
                ),
            }
        )
    spec_payload = {
        "task_id": "T-BEYONDARENA-VALRUN",
        "subset": str(subset),
        "task_count": int(len(manifest)),
        "unique_dataset_count": int(manifest["dataset_id"].nunique()) if "dataset_id" in manifest.columns else 0,
        "manifest_coverage": _manifest_coverage_payload(manifest, subset=str(subset)),
        "classification_task_count": int(manifest["problem_type"].astype(str).str.lower().eq("classification").sum())
        if "problem_type" in manifest.columns
        else 0,
        "regression_task_count": int(manifest["problem_type"].astype(str).str.lower().ne("classification").sum())
        if "problem_type" in manifest.columns
        else 0,
        "inputs": {
            "task_manifest": str(root / "beyondarena_task_manifest.csv"),
            "model_manifest": str(root / "beyondarena_model_manifest.csv"),
            "artifact_root": str(artifact_root),
            "official_results": "public-r2",
            "local_results": str(local_current_results),
        },
        "outputs": outputs,
        "host_allocation": host_allocation,
        "local_current_shard_plan": local_current_shard_plan,
        "cpu_capacity_revalidated": bool(cpu_capacity_revalidated),
        "tabentics_diakrino_checkpoint_ready": bool(tabentics_diakrino_checkpoint_ready),
        "execution_plan_counts": runnable_counts,
        "commands": commands,
        "command_sequence": command_sequence,
        "sharded_command_sequence": sharded_command_sequence,
        "prelaunch_gates": prelaunch_gates,
        "metrics": [
            "official BeyondArena metric per task",
            "metric_value",
            "paired W/T/L",
            "mean_delta",
            "median_delta",
            "Wilcoxon signed-rank p-value where pair count permits",
            (
                "subgroup breakdowns by "
                "task_type/problem_type/size_tier/dimensionality/text/high_cardinality"
            ),
        ],
        "stop_conditions": [
            "missing required DataFoundry metadata or dataset.parquet for a planned materialized task",
            "split leakage guard failure",
            "metric direction mismatch between official and local rows",
            "no exact paired rows after manifest alignment",
            "GPU-required DIAKRINO backend requested before public-gpu-host revalidation",
            "native Tabnetics Diakrino backend requested before trained FS-classifier checkpoint is recorded",
            "concurrent pods share a result directory without disjoint queues or work-claim locks",
            (
                "sustained host utilization below target with queued jobs, "
                "requiring pod split/rebalance before continuation"
            ),
        ],
        "resume_plan": [
            (
                "Materialization is file-grain and idempotent; rerun with the same artifact_root "
                "to skip already-present files."
            ),
            (
                "Local result CSVs are manifest-aligned; rerun failed backend/subset rows and "
                "concatenate before comparison."
            ),
            "Comparison is reproducible from task_manifest, public-r2 official rows, and local result CSVs.",
        ],
        "expected_runtime": (
            "Stage-1 smoke is expected to be short after artifacts are staged; "
            "BeyondArena-Core/full runtime must be estimated from materialized row counts "
            "and live host telemetry before launch."
        ),
        "claims_policy": (
            "Rows from sklearn-smoke, deferred DIAKRINO backends, or skipped regression tasks are "
            "schema/accounting rows only. "
            "No production default promotion or performance claim is valid without paired portfolio-level evidence."
        ),
    }
    if include_diakrino_lanes:
        spec_payload["inputs"]["tabentics_diakrino_checkpoint_env"] = "TABENTICS_DIAKRINO_CHECKPOINT"
    spec_payload["readiness_audit"] = _readiness_audit_payload(
        {
            "subset": str(subset),
            "source": "not configured",
            "available": False,
            "exact_paired_rows_available": False,
            "manifest_coverage": spec_payload["manifest_coverage"],
            "comparison_ready": False,
            "local_results_source": "not configured",
        },
        manifest,
        execution_plan=execution_plan,
        run_spec_payload=spec_payload,
    )
    return spec_payload


def write_beyondarena_plan_artifacts(
    *,
    out_dir: str | Path,
    subset: str = "smoke",
    task_metadata_source: str | Path | pd.DataFrame = BEYONDARENA_TASK_METADATA_CSV_URL,
    core_tasks_source: str | Path | pd.DataFrame = BEYONDARENA_CORE_TASKS_CSV_URL,
    max_smoke_items: int = 6,
    official_results: Optional[str | Path] = None,
    local_results: Optional[str | Path] = None,
    emit_local_execution_plan: bool = False,
    emit_pending_local_rows: bool = False,
    emit_run_spec: bool = False,
    arch_ml_revalidated: bool = False,
    artifact_root: Optional[str | Path] = None,
    public_cpu_host_1_max_workers: Optional[int] = None,
    public_cpu_host_1_pods_per_host: Optional[int] = None,
    public_cpu_host_2_max_workers: Optional[int] = None,
    public_cpu_host_2_pods_per_host: Optional[int] = None,
    cpu_capacity_revalidated: bool = False,
    arch_ml_max_workers: Optional[int] = None,
    arch_ml_pods_per_host: Optional[int] = None,
    tabentics_diakrino_checkpoint_ready: bool = False,
) -> BeyondArenaPlanArtifacts:
    """Write task/model manifests and optional exact comparison artifacts."""

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    rows = resolve_beyondarena_task_rows(
        task_metadata_source=task_metadata_source,
        core_tasks_source=core_tasks_source,
        subset=subset,
        max_smoke_items=max_smoke_items,
    )
    task_manifest = root / "beyondarena_task_manifest.csv"
    manifest = task_rows_to_manifest_frame(rows)
    manifest.to_csv(task_manifest, index=False)
    manifest_coverage = _manifest_coverage_payload(manifest, subset=str(subset))

    profiles = default_beyondarena_run_profiles()
    execution_profiles = _execution_profiles_for_subset(profiles, subset=str(subset))
    model_manifest = root / "beyondarena_model_manifest.csv"
    run_profiles_to_frame(profiles).to_csv(model_manifest, index=False)

    official_loaded: Optional[pd.DataFrame] = None
    if official_results is None:
        status = BeyondArenaOfficialResultsStatus(
            source="not configured",
            available=False,
            exact_paired_rows_available=False,
            reason=(
                "No official BeyondArena per-dataset/per-split result artifact was configured. "
                "Paper-level aggregate values can be cited, but exact paired parity claims are blocked."
            ),
        )
    else:
        try:
            official_loaded = load_official_beyondarena_results(official_results)
            required = {"dataset_id", "split_id", "method", "metric", "metric_value"}
            exact = required.issubset(official_loaded.columns) and not official_loaded.empty
            status = BeyondArenaOfficialResultsStatus(
                source=str(official_results),
                available=not official_loaded.empty,
                exact_paired_rows_available=bool(exact),
                reason="exact paired rows available" if exact else "loaded table lacks exact paired rows",
                columns=tuple(str(col) for col in official_loaded.columns),
                row_count=int(len(official_loaded)),
            )
        except Exception as exc:
            status = BeyondArenaOfficialResultsStatus(
                source=str(official_results),
                available=False,
                exact_paired_rows_available=False,
                reason=f"Unable to load official BeyondArena results: {exc}",
            )
    status_payload: dict[str, Any] = {
        **asdict(status),
        "subset": str(subset),
        "task_count": int(len(rows)),
        "manifest_coverage": manifest_coverage,
        "metric_contract_version": BEYONDARENA_METRIC_CONTRACT_VERSION,
        "metric_contract_valid": False,
        "comparison_ready": False,
        "local_results_source": str(local_results) if local_results is not None else "not configured",
    }
    official_normalized = local_normalized = local_execution_plan = None
    pending_local = run_spec = joined_path = summary_path = None
    execution_plan: Optional[pd.DataFrame] = None
    spec_payload: Optional[dict[str, Any]] = None
    if emit_local_execution_plan or emit_run_spec:
        execution_plan = build_beyondarena_local_execution_plan(
            manifest,
            profiles=execution_profiles,
            arch_ml_revalidated=bool(arch_ml_revalidated),
            tabentics_diakrino_checkpoint_ready=bool(tabentics_diakrino_checkpoint_ready),
        )
    if emit_local_execution_plan:
        local_execution_plan = root / "local_execution_plan.csv"
        assert execution_plan is not None
        execution_plan.to_csv(local_execution_plan, index=False)
        status_payload["local_execution_plan_rows"] = int(len(execution_plan))
    if emit_run_spec:
        assert execution_plan is not None
        run_spec = root / "run_spec.json"
        spec_payload = build_beyondarena_run_spec(
            manifest,
            execution_plan,
            subset=str(subset),
            out_dir=root,
            artifact_root=artifact_root or (root.parent / "artifacts"),
            arch_ml_revalidated=bool(arch_ml_revalidated),
            public_cpu_host_1_max_workers=public_cpu_host_1_max_workers,
            public_cpu_host_1_pods_per_host=public_cpu_host_1_pods_per_host,
            public_cpu_host_2_max_workers=public_cpu_host_2_max_workers,
            public_cpu_host_2_pods_per_host=public_cpu_host_2_pods_per_host,
            cpu_capacity_revalidated=bool(cpu_capacity_revalidated),
            arch_ml_max_workers=arch_ml_max_workers,
            arch_ml_pods_per_host=arch_ml_pods_per_host,
            tabentics_diakrino_checkpoint_ready=bool(tabentics_diakrino_checkpoint_ready),
            local_results_source=local_results,
            comparison_out_dir=root if local_results is not None else None,
        )
        status_payload["run_spec"] = str(run_spec)
    if official_results is not None and local_results is not None and status.exact_paired_rows_available:
        assert official_loaded is not None
        official = align_beyondarena_results_to_manifest(
            official_loaded,
            manifest,
        )
        local_raw = normalize_beyondarena_result_table(_read_table(local_results), origin="tabnetics")
        local = align_beyondarena_results_to_manifest(
            local_raw,
            manifest,
        )
        status_payload.update(_local_result_status_payload(local_raw, local))
        artifacts = build_beyondarena_comparison_artifacts(official, local)
        official_normalized = root / "official_normalized.csv"
        local_normalized = root / "local_normalized.csv"
        joined_path = root / "joined_pairs.csv"
        summary_path = root / "summary.csv"
        artifacts.official.to_csv(official_normalized, index=False)
        artifacts.local.to_csv(local_normalized, index=False)
        artifacts.joined.to_csv(joined_path, index=False)
        artifacts.summary.to_csv(summary_path, index=False)
        joined_rows = int(len(artifacts.joined))
        summary_rows = int(len(artifacts.summary))
        required_contract_columns = {
            "comparison_value_semantics",
            "comparison_lower_is_better",
            "comparison_value_official",
            "comparison_value_local",
        }
        metric_contract_valid = bool(
            joined_rows > 0
            and required_contract_columns.issubset(artifacts.joined.columns)
            and artifacts.joined["comparison_value_semantics"].isin({"metric", "error"}).all()
            and np.isfinite(
                pd.to_numeric(artifacts.joined["comparison_value_official"], errors="coerce")
            ).all()
            and np.isfinite(
                pd.to_numeric(artifacts.joined["comparison_value_local"], errors="coerce")
            ).all()
            and np.isfinite(
                pd.to_numeric(artifacts.joined["comparison_delta"], errors="coerce")
            ).all()
            and (
                ~artifacts.joined["comparison_value_semantics"].eq("error")
                | artifacts.joined["comparison_lower_is_better"].astype(bool)
            ).all()
        )
        status_payload.update(
            {
                "comparison_ready": metric_contract_valid,
                "metric_contract_valid": metric_contract_valid,
                "comparison_value_semantics_counts": _value_counts(
                    artifacts.joined,
                    "comparison_value_semantics",
                ),
                "joined_rows": joined_rows,
                "summary_rows": summary_rows,
                "comparison_blocker": (
                    ""
                    if metric_contract_valid
                    else (
                        "Paired rows were emitted but do not satisfy the explicit BeyondArena metric contract."
                        if joined_rows > 0
                        else (
                            "No exact ok local rows joined official results after manifest alignment. "
                            "Check skipped rows, dataset_id/split_id/metric keys, and materialized local artifacts."
                        )
                    )
                ),
            }
        )
    elif local_results is not None:
        local_normalized = root / "local_normalized.csv"
        local_raw = normalize_beyondarena_result_table(_read_table(local_results), origin="tabnetics")
        local = align_beyondarena_results_to_manifest(
            local_raw,
            manifest,
        )
        status_payload.update(_local_result_status_payload(local_raw, local))
        local.to_csv(local_normalized, index=False)
    elif emit_pending_local_rows:
        pending_local = root / "pending_local_rows.csv"
        pending = build_pending_local_result_rows(
            manifest,
            profiles=execution_profiles,
            arch_ml_revalidated=bool(arch_ml_revalidated),
            tabentics_diakrino_checkpoint_ready=bool(tabentics_diakrino_checkpoint_ready),
        )
        pending.to_csv(pending_local, index=False)
        status_payload["pending_local_rows"] = int(len(pending))

    readiness_audit = root / "readiness_audit.json"
    readiness_payload = _readiness_audit_payload(
        status_payload,
        manifest,
        execution_plan=execution_plan,
        run_spec_payload=spec_payload,
    )
    status_payload["readiness_audit"] = readiness_payload
    if spec_payload is not None:
        spec_payload["readiness_audit"] = readiness_payload
        assert run_spec is not None
        run_spec.write_text(json.dumps(spec_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readiness_audit.write_text(json.dumps(readiness_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    comparison_status = root / "comparison_status.json"
    comparison_status.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return BeyondArenaPlanArtifacts(
        out_dir=root,
        task_manifest=task_manifest,
        model_manifest=model_manifest,
        comparison_status=comparison_status,
        readiness_audit=readiness_audit,
        official_normalized=official_normalized,
        local_normalized=local_normalized,
        local_execution_plan=local_execution_plan,
        pending_local=pending_local,
        run_spec=run_spec,
        joined=joined_path,
        summary=summary_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--subset",
        choices=(
            "smoke",
            "current-feasibility",
            "core",
            "core-all-splits",
            "core-classification-all-splits",
            "all",
        ),
        default="smoke",
    )
    parser.add_argument("--task-metadata-csv", default=BEYONDARENA_TASK_METADATA_CSV_URL)
    parser.add_argument("--core-tasks-csv", default=BEYONDARENA_CORE_TASKS_CSV_URL)
    parser.add_argument("--max-smoke-items", type=int, default=6)
    parser.add_argument("--official-results", help="Optional official CSV/parquet path or URL")
    parser.add_argument("--local-results", help="Optional local CSV/parquet path or URL")
    parser.add_argument(
        "--emit-pending-local-rows",
        action="store_true",
        help="Write pending_local_rows.csv with skipped Tabnetics Diakrino/current rows when --local-results is absent.",
    )
    parser.add_argument(
        "--emit-local-execution-plan",
        action="store_true",
        help="Write local_execution_plan.csv with runnable/deferred/unsupported local profile rows.",
    )
    parser.add_argument(
        "--emit-run-spec",
        action="store_true",
        help="Write run_spec.json with commands, host allocation, stop conditions, and resume plan.",
    )
    parser.add_argument(
        "--public-gpu-host-revalidated",
        action="store_true",
        help="Mark GPU-required classification rows ready instead of deferred in the local execution plan.",
    )
    parser.add_argument("--artifact-root", type=Path, help="Artifact root to record in run_spec.json.")
    parser.add_argument("--public_cpu_host_1-max-workers", type=int)
    parser.add_argument("--public_cpu_host_1-pods-per-host", type=int)
    parser.add_argument("--public_cpu_host_2-max-workers", type=int)
    parser.add_argument("--public_cpu_host_2-pods-per-host", type=int)
    parser.add_argument(
        "--cpu-capacity-revalidated",
        action="store_true",
        help=(
            "Mark public_cpu_host_1/public_cpu_host_2 MAX_WORKERS and PODS_PER_HOST as derived from live capacity "
            "checks recorded in the active issue/run log."
        ),
    )
    parser.add_argument("--public-gpu-host-max-workers", type=int)
    parser.add_argument("--public-gpu-host-pods-per-host", type=int)
    parser.add_argument(
        "--tabnetics-diakrino-checkpoint-ready",
        dest="tabentics_diakrino_checkpoint_ready",
        action="store_true",
        help=(
            "Mark the native Tabnetics Diakrino FS-classifier checkpoint from #193/#192 as recorded "
            "before allowing tabnetics-diakrino GPU execution."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifacts = write_beyondarena_plan_artifacts(
        out_dir=args.out_dir,
        subset=args.subset,
        task_metadata_source=args.task_metadata_csv,
        core_tasks_source=args.core_tasks_csv,
        max_smoke_items=args.max_smoke_items,
        official_results=args.official_results,
        local_results=args.local_results,
        emit_local_execution_plan=args.emit_local_execution_plan,
        emit_pending_local_rows=args.emit_pending_local_rows,
        emit_run_spec=args.emit_run_spec,
        arch_ml_revalidated=args.arch_ml_revalidated,
        artifact_root=args.artifact_root,
        public_cpu_host_1_max_workers=args.public_cpu_host_1_max_workers,
        public_cpu_host_1_pods_per_host=args.public_cpu_host_1_pods_per_host,
        public_cpu_host_2_max_workers=args.public_cpu_host_2_max_workers,
        public_cpu_host_2_pods_per_host=args.public_cpu_host_2_pods_per_host,
        cpu_capacity_revalidated=args.cpu_capacity_revalidated,
        arch_ml_max_workers=args.arch_ml_max_workers,
        arch_ml_pods_per_host=args.arch_ml_pods_per_host,
        tabentics_diakrino_checkpoint_ready=args.tabentics_diakrino_checkpoint_ready,
    )
    print(f"task_manifest={artifacts.task_manifest}")
    print(f"model_manifest={artifacts.model_manifest}")
    print(f"comparison_status={artifacts.comparison_status}")
    print(f"readiness_audit={artifacts.readiness_audit}")
    if artifacts.joined is not None:
        print(f"joined={artifacts.joined}")
        print(f"summary={artifacts.summary}")
    if artifacts.pending_local is not None:
        print(f"pending_local={artifacts.pending_local}")
    if artifacts.local_execution_plan is not None:
        print(f"local_execution_plan={artifacts.local_execution_plan}")
    if artifacts.run_spec is not None:
        print(f"run_spec={artifacts.run_spec}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BeyondArenaPlanArtifacts",
    "BeyondArenaRunProfile",
    "align_beyondarena_results_to_manifest",
    "build_beyondarena_local_execution_plan",
    "build_beyondarena_run_spec",
    "build_pending_local_result_rows",
    "default_beyondarena_run_profiles",
    "resolve_beyondarena_task_rows",
    "run_profiles_to_frame",
    "task_rows_to_manifest_frame",
    "write_beyondarena_plan_artifacts",
]

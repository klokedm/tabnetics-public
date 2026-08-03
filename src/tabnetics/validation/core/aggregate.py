#!/usr/bin/env python3
"""Aggregate per-job outputs from a pod_validation run directory.

This is intentionally lightweight: it concatenates summary CSVs so you can
quickly inspect coverage and deltas without manually digging through shards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from tabnetics.core.paths import find_repo_root_or_none
from tabnetics.validation.core.host_calibration import (
    append_records,
    collect_records_from_root,
    default_host_calibration_path,
    resolve_host_calibration_path,
)
from tabnetics.validation.core.provenance import (
    CANONICAL_BENCHMARK_ARTIFACT_PROVENANCE_FILENAME,
    canonical_benchmark_artifact_provenance_consistency,
    canonical_execution_contract_consistency,
    canonical_execution_eligibility,
    execution_row_fields,
)

REPO_ROOT = find_repo_root_or_none(__file__)
__tabnetics_execution_isolated_state__ = {
    "REPO_ROOT": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}


def _require_repo_root() -> Path:
    if REPO_ROOT is None:
        raise RuntimeError(
            "This command requires a tabnetics repo checkout but no project root "
            "was found (running from an installed package?)"
        )
    return REPO_ROOT


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_job_id(job_id: str) -> str:
    return job_id.replace("/", "__")


def _load_plan(plan_path: Path) -> List[Dict[str, Any]]:
    data = _read_json(plan_path)
    jobs = data.get("jobs") or []
    if not isinstance(jobs, list):
        raise ValueError(f"Invalid plan format: jobs is not a list in {plan_path}")
    return jobs


def _maybe_load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        print(f"[aggregate] WARNING: failed to read {path}: {exc}")
        return None


def _find_latest_run_dir(job_dir: Path, *, suffix: str) -> Optional[Path]:
    """Return the newest timestamped run dir under a job directory."""
    if not job_dir.exists():
        return None
    candidates = [p for p in job_dir.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    if not candidates:
        return None
    # Timestamped dirs sort lexicographically by design: YYYYMMDD_HHMMSS_...
    return sorted(candidates, key=lambda p: p.name)[-1]


_CANONICAL_BENCHMARK_CSV_ARTIFACTS: tuple[str, ...] = (
    "df_fs_runs.csv",
    "df_fs_summary.csv",
    "df_fs_sota_comparison.csv",
    "df_fs_ablation_deltas.csv",
)
_CANONICAL_EXECUTION_ROW_FIELDS: tuple[str, ...] = (
    "implementation_stack",
    "evidence_status",
    "canonical_scorecard_eligible",
    "execution_provenance_schema",
    "execution_provenance_sha256",
    "resolved_cli_config_sha256",
    "input_data_identity_sha256",
    "materialized_input_set_sha256",
    "package_identity_sha256",
    "source_revision_git_sha",
    "source_revision_tabnetics_version",
    "source_revision_module_hashes_sha256",
    "loaded_package_modules_sha256",
    "loaded_package_symbols_sha256",
    "pipeline_import_origin",
)


def _benchmark_execution_contract(
    run_dir: Path,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Fail closed for benchmark artifacts without the core execution contract."""

    metadata_path = run_dir / "df_fs_metadata.json"
    if not metadata_path.exists():
        return False, "df_fs_metadata_missing", None
    try:
        metadata = _read_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"df_fs_metadata_unreadable:{type(exc).__name__}", None
    eligible, reason = canonical_execution_eligibility(metadata)
    execution = metadata.get("execution_provenance")
    if not isinstance(execution, dict):
        execution = None
    return eligible, reason, execution


def _execution_value_matches(
    actual: Any,
    expected: Any,
    *,
    field: str | None = None,
) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual is expected
        normalized = str(actual).strip().lower()
        return normalized in ({"true", "1"} if expected else {"false", "0"})
    if pd.isna(actual):
        # CSV readers represent an intentionally empty provenance field as
        # NaN.  Source archives legitimately have no Git SHA, but no other
        # missing execution identity is accepted by this canonical contract.
        return (
            field == "source_revision_git_sha"
            and (expected is None or not str(expected).strip())
        )
    return str(actual) == str(expected)


def _frame_execution_contract_reason(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    execution: Mapping[str, Any],
) -> str:
    if frame.empty:
        return f"{artifact_name}_execution_rows_missing"
    expected_fields = execution_row_fields(execution)
    for field in _CANONICAL_EXECUTION_ROW_FIELDS:
        if field not in frame.columns:
            return f"{artifact_name}_execution_field_missing:{field}"
        expected = expected_fields[field]
        for row_index, actual in frame[field].items():
            if not _execution_value_matches(actual, expected, field=field):
                return f"{artifact_name}_execution_field_mismatch:{field}:row={row_index}"
    if artifact_name == "df_fs_runs.csv":
        for field in ("dataset_id", "seed", "materialized_input_identity_sha256"):
            if field not in frame.columns:
                return f"{artifact_name}_materialized_input_field_missing:{field}"
        input_identity = execution.get("input_data_identity")
        records = (
            input_identity.get("materialized_inputs")
            if isinstance(input_identity, Mapping)
            else None
        )
        if not isinstance(records, list):
            return f"{artifact_name}_materialized_input_records_missing"
        by_task: dict[tuple[str, int], str] = {}
        for record in records:
            if not isinstance(record, Mapping):
                return f"{artifact_name}_materialized_input_record_invalid"
            try:
                key = (str(record["dataset_id"]), int(record["seed"]))
            except (KeyError, TypeError, ValueError):
                return f"{artifact_name}_materialized_input_record_invalid"
            digest = str(record.get("materialized_input_sha256", "") or "")
            if not digest or key in by_task:
                return f"{artifact_name}_materialized_input_record_invalid"
            by_task[key] = digest
        for row_index, row in frame.iterrows():
            try:
                key = (str(row["dataset_id"]), int(row["seed"]))
            except (TypeError, ValueError):
                return f"{artifact_name}_materialized_input_task_invalid:row={row_index}"
            expected_digest = by_task.get(key)
            if not expected_digest:
                return f"{artifact_name}_materialized_input_task_missing:row={row_index}"
            if not _execution_value_matches(
                row["materialized_input_identity_sha256"], expected_digest
            ):
                return f"{artifact_name}_materialized_input_digest_mismatch:row={row_index}"
    return ""


def _read_required_csv(path: Path) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        return None, f"{path.name}_missing"
    try:
        return pd.read_csv(path), ""
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        return None, f"{path.name}_unreadable:{type(exc).__name__}"


def _benchmark_artifact_contract_reason(
    run_dir: Path,
    execution: Mapping[str, Any],
) -> str:
    manifest_path = run_dir / "df_fs_execution_provenance.json"
    if not manifest_path.exists():
        return "df_fs_execution_provenance_missing"
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"df_fs_execution_provenance_unreadable:{type(exc).__name__}"
    consistent, reason = canonical_execution_contract_consistency(execution, manifest)
    if not consistent:
        return f"df_fs_execution_provenance_{reason}"

    for artifact_name in _CANONICAL_BENCHMARK_CSV_ARTIFACTS:
        frame, read_reason = _read_required_csv(run_dir / artifact_name)
        if frame is None:
            return read_reason
        row_reason = _frame_execution_contract_reason(
            frame,
            artifact_name=artifact_name,
            execution=execution,
        )
        if row_reason:
            return row_reason

    artifact_provenance_path = (
        run_dir / CANONICAL_BENCHMARK_ARTIFACT_PROVENANCE_FILENAME
    )
    if not artifact_provenance_path.exists():
        return "df_fs_artifact_provenance_missing"
    try:
        artifact_provenance = _read_json(artifact_provenance_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"df_fs_artifact_provenance_unreadable:{type(exc).__name__}"
    artifact_consistent, artifact_reason = (
        canonical_benchmark_artifact_provenance_consistency(
            execution,
            artifact_provenance,
            artifact_paths={
                "df_fs_runs.csv": run_dir / "df_fs_runs.csv",
                "df_fs_summary.csv": run_dir / "df_fs_summary.csv",
                "df_fs_sota_comparison.csv": run_dir / "df_fs_sota_comparison.csv",
                "df_fs_ablation_deltas.csv": run_dir / "df_fs_ablation_deltas.csv",
                "df_fs_metadata.json": run_dir / "df_fs_metadata.json",
                "df_fs_execution_provenance.json": manifest_path,
            },
        )
    )
    if not artifact_consistent:
        return f"df_fs_artifact_provenance_{artifact_reason}"
    return ""


def aggregate(
    root_out: Path,
    *,
    plan_path: Path,
    out_dir: Optional[Path] = None,
    host_calibration_path: Path | None = None,
) -> None:
    jobs = _load_plan(plan_path)
    if out_dir is None:
        out_dir = root_out / "_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_rows: List[pd.DataFrame] = []
    bench_sota_rows: List[pd.DataFrame] = []
    bench_runs_rows: List[pd.DataFrame] = []
    suite_by_dataset_rows: List[pd.DataFrame] = []
    suite_runs_rows: List[pd.DataFrame] = []

    missing: List[Tuple[str, str]] = []
    excluded_noncanonical: List[Dict[str, str]] = []

    for job in jobs:
        job_id = str(job.get("job_id") or "")
        kind = str(job.get("kind") or "")
        job_dir = root_out / job_id
        run_dir = None

        if kind == "run_df_fs_sota_benchmark":
            run_dir = _find_latest_run_dir(job_dir, suffix="df_fs_sota_benchmark")
            if run_dir is None:
                missing.append((job_id, "<run_dir>/*df_fs_sota_benchmark"))
                continue
            eligible, exclusion_reason, execution = _benchmark_execution_contract(run_dir)
            if eligible and execution is not None:
                exclusion_reason = _benchmark_artifact_contract_reason(run_dir, execution)
                eligible = not bool(exclusion_reason)
            if not eligible:
                excluded_noncanonical.append(
                    {
                        "job_id": job_id,
                        "job_kind": kind,
                        "run_dir": str(run_dir),
                        "exclusion_reason": exclusion_reason,
                    }
                )
                print(
                    "[aggregate] excluded noncanonical benchmark artifact "
                    f"job={job_id} reason={exclusion_reason} run_dir={run_dir}"
                )
                continue
            summary = _maybe_load_csv(run_dir / "df_fs_summary.csv")
            if summary is not None:
                summary.insert(0, "run_dir", str(run_dir))
                summary.insert(0, "job_id", job_id)
                summary.insert(1, "job_kind", kind)
                bench_rows.append(summary)
            else:
                missing.append((job_id, f"{run_dir.name}/df_fs_summary.csv"))

            sota = _maybe_load_csv(run_dir / "df_fs_sota_comparison.csv")
            if sota is not None:
                sota.insert(0, "run_dir", str(run_dir))
                sota.insert(0, "job_id", job_id)
                sota.insert(1, "job_kind", kind)
                bench_sota_rows.append(sota)
            else:
                missing.append((job_id, f"{run_dir.name}/df_fs_sota_comparison.csv"))

            runs = _maybe_load_csv(run_dir / "df_fs_runs.csv")
            if runs is not None:
                runs.insert(0, "run_dir", str(run_dir))
                runs.insert(0, "job_id", job_id)
                runs.insert(1, "job_kind", kind)
                bench_runs_rows.append(runs)
            else:
                missing.append((job_id, f"{run_dir.name}/df_fs_runs.csv"))
            continue

        if kind == "validation_suite":
            run_dir = _find_latest_run_dir(job_dir, suffix="validation_suite")
            if run_dir is None:
                missing.append((job_id, "<run_dir>/*validation_suite"))
                continue

            by_ds = _maybe_load_csv(run_dir / "validation_summary_by_dataset.csv")
            runs = _maybe_load_csv(run_dir / "validation_runs.csv")

            if by_ds is not None:
                by_ds.insert(0, "run_dir", str(run_dir))
                by_ds.insert(0, "job_id", job_id)
                by_ds.insert(1, "job_kind", kind)
                suite_by_dataset_rows.append(by_ds)
            else:
                missing.append((job_id, f"{run_dir.name}/validation_summary_by_dataset.csv"))

            if runs is not None:
                runs.insert(0, "run_dir", str(run_dir))
                runs.insert(0, "job_id", job_id)
                runs.insert(1, "job_kind", kind)
                suite_runs_rows.append(runs)
            else:
                missing.append((job_id, f"{run_dir.name}/validation_runs.csv"))
            continue

    if bench_rows:
        df_bench = pd.concat(bench_rows, axis=0, ignore_index=True, sort=False)
        out_path = out_dir / "benchmark_df_fs_summary__all_jobs.csv"
        df_bench.to_csv(out_path, index=False)
        print(f"Wrote: {out_path} rows={len(df_bench)}")

    if bench_sota_rows:
        df_sota = pd.concat(bench_sota_rows, axis=0, ignore_index=True, sort=False)
        out_path = out_dir / "benchmark_df_fs_sota_comparison__all_jobs.csv"
        df_sota.to_csv(out_path, index=False)
        print(f"Wrote: {out_path} rows={len(df_sota)}")

    if bench_runs_rows:
        df_runs = pd.concat(bench_runs_rows, axis=0, ignore_index=True, sort=False)
        out_path = out_dir / "benchmark_df_fs_runs__all_jobs.csv"
        df_runs.to_csv(out_path, index=False)
        print(f"Wrote: {out_path} rows={len(df_runs)}")

    if suite_by_dataset_rows:
        df_suite = pd.concat(suite_by_dataset_rows, axis=0, ignore_index=True, sort=False)
        out_path = out_dir / "validation_suite_summary_by_dataset__all_jobs.csv"
        df_suite.to_csv(out_path, index=False)
        print(f"Wrote: {out_path} rows={len(df_suite)}")

    if suite_runs_rows:
        df_runs = pd.concat(suite_runs_rows, axis=0, ignore_index=True, sort=False)
        out_path = out_dir / "validation_suite_runs__all_jobs.csv"
        df_runs.to_csv(out_path, index=False)
        print(f"Wrote: {out_path} rows={len(df_runs)}")

    if missing:
        miss_path = out_dir / "missing_artifacts.csv"
        pd.DataFrame(missing, columns=["job_id", "missing_file"]).to_csv(miss_path, index=False)
        print(f"Wrote: {miss_path} rows={len(missing)}")

    if excluded_noncanonical:
        excluded_path = out_dir / "excluded_noncanonical_artifacts.csv"
        pd.DataFrame(excluded_noncanonical).to_csv(excluded_path, index=False)
        print(f"Wrote: {excluded_path} rows={len(excluded_noncanonical)}")

    calibration_records = collect_records_from_root(root_out)
    if calibration_records:
        catalog_path = host_calibration_path or default_host_calibration_path(_require_repo_root())
        append_records(catalog_path, calibration_records)
        print(f"Wrote: {catalog_path} host_calibration_records={len(calibration_records)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate pod_validation job outputs into combined CSVs.")
    parser.add_argument("--root-out-dir", type=str, required=True)
    parser.add_argument("--num-pods", type=int, default=20)
    parser.add_argument("--plan", type=str, default="", help="Override plan JSON path.")
    parser.add_argument("--out-dir", type=str, default="", help="Override aggregate output directory.")
    parser.add_argument(
        "--host-calibration-catalog",
        type=str,
        default="",
        help=(
            "Path to versioned host calibration JSON. Defaults to "
            "run_artifacts/meta/host_calibration.json under the repo root."
        ),
    )
    args = parser.parse_args()

    root_out = Path(args.root_out_dir).expanduser().resolve()
    num_pods = int(max(1, min(int(args.num_pods), 20)))
    plan_path = Path(args.plan) if args.plan else (_require_repo_root() / "pod_validation" / f"plan_{num_pods}.json")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if not plan_path.exists():
        raise FileNotFoundError(
            f"Missing plan file: {plan_path} "
            "(run `python -m tabnetics.validation.generate_plan ...`)"
        )
    host_calibration_path = (
        resolve_host_calibration_path(args.host_calibration_catalog, repo_root=_require_repo_root())
        if str(args.host_calibration_catalog or "").strip()
        else None
    )
    aggregate(
        root_out,
        plan_path=plan_path,
        out_dir=out_dir,
        host_calibration_path=host_calibration_path,
    )


if __name__ == "__main__":
    main()

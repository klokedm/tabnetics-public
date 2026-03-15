#!/usr/bin/env python3
"""Aggregate per-job outputs from a pod_validation run directory.

This is intentionally lightweight: it concatenates summary CSVs so you can
quickly inspect coverage and deltas without manually digging through shards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from tabnetics.core.paths import find_repo_root

REPO_ROOT = find_repo_root(__file__)


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


def aggregate(root_out: Path, *, plan_path: Path, out_dir: Optional[Path] = None) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate pod_validation job outputs into combined CSVs.")
    parser.add_argument("--root-out-dir", type=str, required=True)
    parser.add_argument("--num-pods", type=int, default=20)
    parser.add_argument("--plan", type=str, default="", help="Override plan JSON path.")
    parser.add_argument("--out-dir", type=str, default="", help="Override aggregate output directory.")
    args = parser.parse_args()

    root_out = Path(args.root_out_dir).expanduser().resolve()
    num_pods = int(max(1, min(int(args.num_pods), 20)))
    plan_path = Path(args.plan) if args.plan else (REPO_ROOT / "pod_validation" / f"plan_{num_pods}.json")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if not plan_path.exists():
        raise FileNotFoundError(
            f"Missing plan file: {plan_path} "
            "(run `python -m tabnetics.validation.generate_plan ...`)"
        )
    aggregate(root_out, plan_path=plan_path, out_dir=out_dir)


if __name__ == "__main__":
    main()

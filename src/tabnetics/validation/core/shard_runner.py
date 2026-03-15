#!/usr/bin/env python3
"""Run a single shard from a generated validation plan, backing ``tabnetics-validation-shard`` and consuming the ``plan_*.json`` / ``shards_*.json`` artifacts emitted by ``tabnetics-validation-plan``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tabnetics.core.paths import find_repo_root

REPO_ROOT = find_repo_root(__file__)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_job_id(job_id: str) -> str:
    return job_id.replace("/", "__")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_path_list(name: str) -> List[Path]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    out: List[Path] = []
    seen: set[str] = set()
    for item in raw.split(os.pathsep):
        item = str(item).strip()
        if not item:
            continue
        key = os.path.normpath(os.path.expanduser(item))
        if key in seen:
            continue
        seen.add(key)
        out.append(Path(key))
    return out


def _skip_done_status_dirs(root_out: Path) -> List[Path]:
    status_dir = root_out / "_status"
    out: List[Path] = [status_dir]
    seen = {str(status_dir)}
    for extra_dir in _env_path_list("SKIP_DONE_STATUS_DIRS"):
        key = str(extra_dir)
        if key in seen:
            continue
        seen.add(key)
        out.append(extra_dir)
    return out


def _find_done_marker(status_dirs: Sequence[Path], safe_id: str) -> Optional[Path]:
    marker_name = f"{safe_id}.DONE.ok"
    for status_dir in status_dirs:
        marker = status_dir / marker_name
        if marker.exists():
            return marker
    return None


def _build_benchmark_cmd(job: Dict[str, Any], *, out_dir: Path, max_workers: int) -> List[str]:
    p = dict(job.get("params") or {})
    dataset_sets = list(p.get("dataset_sets") or [])
    datasets = list(p.get("datasets") or [])
    seeds = list(p.get("seeds") or [11])
    fs_method_set = str(p.get("fs_method_set") or "").strip()
    if not fs_method_set:
        raise ValueError(f"Missing fs_method_set for job_id={job.get('job_id')}")

    cmd: List[str] = [sys.executable, "-m", "tabnetics.benchmarks.cli"]
    if dataset_sets:
        cmd += ["--dataset-sets", *[str(x) for x in dataset_sets]]
    if datasets:
        cmd += ["--datasets", *[str(x) for x in datasets]]
    cmd += ["--seeds", *[str(int(s)) for s in seeds]]
    cmd += ["--max-workers", str(int(max_workers))]

    task_timeout_sec = _env_float("TASK_TIMEOUT_SEC_OVERRIDE", float(p.get("task_timeout_sec", 0.0)))
    fs_method_timeout_sec = _env_float("FS_METHOD_TIMEOUT_SEC_OVERRIDE", float(p.get("fs_method_timeout_sec", 0.0)))
    cmd += ["--task-timeout-sec", str(float(task_timeout_sec))]
    cmd += ["--fs-method-timeout-sec", str(float(fs_method_timeout_sec))]

    if bool(p.get("quiet_worker_logs", False)):
        cmd.append("--quiet-worker-logs")
    cmd += ["--progress-heartbeat-sec", str(float(p.get("progress_heartbeat_sec", 60.0)))]
    cmd += ["--progress-watchdog-sec", str(float(p.get("progress_watchdog_sec", 900.0)))]
    cmd += ["--progress-stall-watchdog-sec", str(float(p.get("progress_stall_watchdog_sec", 1800.0)))]

    if bool(p.get("allow_synthetic_fallback", False)):
        raise ValueError(
            f"job_id={job.get('job_id')} requested synthetic fallback, which is forbidden for pod validation runs"
        )
    dataset_integrity_policy = str(p.get("dataset_integrity_policy", "") or "").strip()
    if dataset_integrity_policy:
        cmd += ["--dataset-integrity-policy", dataset_integrity_policy]

    ablation_profile = str(p.get("ablation_profile", "none") or "none")
    cmd += ["--ablation-profile", ablation_profile]

    # Optional global override: disable DF fast-path (benchmark default is enabled).
    if _env_bool("DISABLE_DF_FASTPATH", default=False):
        cmd.append("--disable-df-fastpath")

    extra_args = list(p.get("extra_args") or [])
    cmd += [str(x) for x in extra_args]

    cmd += ["--fs-method-set", fs_method_set]
    cmd += ["--output-dir", str(out_dir)]
    return cmd


def _build_validation_suite_cmd(job: Dict[str, Any], *, out_dir: Path) -> List[str]:
    p = dict(job.get("params") or {})
    pipelines = list(p.get("pipelines") or [])
    dataset_sets = list(p.get("dataset_sets") or [])
    seeds = list(p.get("seeds") or [11])
    df_criterion = str(p.get("df_criterion") or "").strip()
    integrated_dist_criterion = str(p.get("integrated_dist_criterion") or "").strip()

    cmd: List[str] = [sys.executable, "-m", "tabnetics.validation.suite"]
    if dataset_sets:
        cmd += ["--dataset-sets", *[str(x) for x in dataset_sets]]
    if pipelines:
        cmd += ["--pipelines", *[str(x) for x in pipelines]]
    cmd += ["--seeds", *[str(int(s)) for s in seeds]]

    if df_criterion:
        cmd += ["--df-criterion", df_criterion]
    if integrated_dist_criterion:
        cmd += ["--integrated-dist-criterion", integrated_dist_criterion]

    ablation_profile = str(p.get("ablation_profile", "none") or "none")
    cmd += ["--ablation-profile", ablation_profile]

    if bool(p.get("allow_synthetic_fallback", False)):
        raise ValueError(
            f"job_id={job.get('job_id')} requested synthetic fallback, which is forbidden for pod validation runs"
        )
    dataset_integrity_policy = str(p.get("dataset_integrity_policy", "") or "").strip()
    if dataset_integrity_policy:
        cmd += ["--dataset-integrity-policy", dataset_integrity_policy]

    cmd += ["--output-dir", str(out_dir)]
    return cmd


def _run_with_tee(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] cmd: {' '.join(cmd)}\n")
        log.flush()

        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return int(proc.wait())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one shard of the pod validation plan.")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--root-out-dir", type=str, required=True)
    parser.add_argument("--num-pods", type=int, default=20, help="Shard count; selects plan_<N>.json/shards_<N>.json by default.")
    parser.add_argument("--plan", type=str, default="", help="Override plan JSON path.")
    parser.add_argument("--shards", type=str, default="", help="Override shards JSON path.")
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    num_pods = int(max(1, min(int(args.num_pods), 20)))
    plan_path = Path(args.plan) if args.plan else (REPO_ROOT / "pod_validation" / f"plan_{num_pods}.json")
    shards_path = Path(args.shards) if args.shards else (REPO_ROOT / "pod_validation" / f"shards_{num_pods}.json")

    if not plan_path.exists():
        raise FileNotFoundError(
            f"Missing plan file: {plan_path} "
            "(run `python -m tabnetics.validation.generate_plan ...`)"
        )
    if not shards_path.exists():
        raise FileNotFoundError(
            f"Missing shards file: {shards_path} "
            "(run `python -m tabnetics.validation.generate_plan ...`)"
        )

    plan = _read_json(plan_path)
    shards = _read_json(shards_path).get("shards") or {}

    shard_id = int(args.shard_id)
    shard_jobs = shards.get(str(shard_id))
    if shard_jobs is None:
        known = sorted(int(k) for k in shards.keys()) if shards else []
        raise ValueError(f"Unknown shard_id={shard_id}. Known shard IDs: {known}")

    jobs_by_id: Dict[str, Dict[str, Any]] = {j["job_id"]: j for j in (plan.get("jobs") or [])}
    missing = [jid for jid in shard_jobs if jid not in jobs_by_id]
    if missing:
        raise ValueError(f"Shard {shard_id} references missing job IDs in plan: {missing}")

    root_out = Path(args.root_out_dir).expanduser().resolve()
    logs_dir = root_out / "_logs"
    status_dir = root_out / "_status"
    shard_status_ok = status_dir / f"shard{shard_id:02d}.DONE.ok"
    shard_status_fail = status_dir / f"shard{shard_id:02d}.DONE.fail"
    status_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    max_workers_env_raw = os.environ.get("MAX_WORKERS")
    max_workers_env_set = bool(max_workers_env_raw is not None and max_workers_env_raw.strip())
    max_workers_default = _env_int("MAX_WORKERS", 4)
    skip_done = bool(args.skip_done) or _env_bool("SKIP_DONE", False)
    skip_done_status_dirs = _skip_done_status_dirs(root_out)

    overall_ok = True
    (status_dir / f"shard{shard_id:02d}.STARTED").write_text(
        f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n",
        encoding="utf-8",
    )
    shard_status_fail.touch()
    if shard_status_ok.exists():
        shard_status_ok.unlink()

    print(
        f"[pod_validation] shard={shard_id} jobs={len(shard_jobs)} "
        f"max_workers_default={max_workers_default} "
        f"max_workers_env_cap={int(max_workers_env_set)} "
        f"skip_done={int(skip_done)}"
    )
    print(f"[pod_validation] plan={plan_path}")
    print(f"[pod_validation] shards={shards_path}")
    print(f"[pod_validation] out={root_out}")
    if len(skip_done_status_dirs) > 1:
        extra_dirs = ", ".join(str(p) for p in skip_done_status_dirs[1:])
        print(f"[pod_validation] skip_done_extra_status_dirs={extra_dirs}")

    for job_id in shard_jobs:
        job = jobs_by_id[job_id]
        kind = str(job.get("kind") or "").strip()
        params = dict(job.get("params") or {})

        job_out = root_out / job_id
        job_out.mkdir(parents=True, exist_ok=True)

        safe_id = _sanitize_job_id(job_id)
        job_ok = status_dir / f"{safe_id}.DONE.ok"
        job_fail = status_dir / f"{safe_id}.DONE.fail"
        job_log = logs_dir / f"{safe_id}.log"

        existing_done = _find_done_marker(skip_done_status_dirs, safe_id) if skip_done else None
        if existing_done is not None:
            if existing_done != job_ok:
                if job_fail.exists():
                    job_fail.unlink()
                job_ok.touch()
                print(
                    f"[pod_validation] reconcile job={job_id} "
                    f"current_status={job_ok} source={existing_done}"
                )
            print(f"[pod_validation] skip job={job_id} (DONE.ok exists at {existing_done})")
            continue

        if job_ok.exists():
            job_ok.unlink()
        if job_fail.exists():
            job_fail.unlink()

        job_max_workers = int(max_workers_default)
        if kind == "run_df_fs_sota_benchmark":
            raw_job_workers = params.get("max_workers")
            if raw_job_workers is not None:
                try:
                    parsed = int(raw_job_workers)
                    if parsed > 0:
                        if max_workers_env_set:
                            job_max_workers = max(1, min(int(max_workers_default), int(parsed)))
                        else:
                            job_max_workers = parsed
                except Exception:
                    pass

        print(
            f"[pod_validation] start job={job_id} kind={kind}"
            + (
                f" max_workers={job_max_workers}"
                if kind == "run_df_fs_sota_benchmark"
                else ""
            )
        )
        t0 = time.perf_counter()

        if kind == "run_df_fs_sota_benchmark":
            cmd = _build_benchmark_cmd(job, out_dir=job_out, max_workers=job_max_workers)
        elif kind == "validation_suite":
            cmd = _build_validation_suite_cmd(job, out_dir=job_out)
        else:
            raise ValueError(f"Unknown job kind: {kind} (job_id={job_id})")

        if bool(args.dry_run):
            print(f"[pod_validation] dry_run cmd: {' '.join(cmd)}")
            # NOTE: do NOT write job_ok marker during dry-run;
            # that would poison subsequent --skip-done real runs.
            continue

        rc = _run_with_tee(cmd, cwd=REPO_ROOT, log_path=job_log)
        dt = time.perf_counter() - t0
        if rc == 0:
            job_ok.touch()
            print(f"[pod_validation] ok job={job_id} sec={dt:.1f}")
        else:
            overall_ok = False
            job_fail.touch()
            print(f"[pod_validation] FAIL job={job_id} rc={rc} sec={dt:.1f}")

    if overall_ok:
        if shard_status_fail.exists():
            shard_status_fail.unlink()
        shard_status_ok.touch()
    print(f"[pod_validation] completed shard={shard_id} overall_ok={int(overall_ok)}")


if __name__ == "__main__":
    main()

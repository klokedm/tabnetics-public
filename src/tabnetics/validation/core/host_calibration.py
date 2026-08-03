"""Host calibration catalog for validation campaign planning.

The catalog is an advisory prior.  It records observed throughput and memory
envelopes from completed jobs so plan generation can start from measured caps,
while the live nproc/memory checks required by the operator guide remain authoritative.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tabnetics.validation.core.provenance import utc_now_iso, write_json


HOST_CALIBRATION_SCHEMA_VERSION = "tabnetics_host_calibration_v1"
HOST_CALIBRATION_ENV = "TABNETICS_HOST_CALIBRATION_PATH"
HOST_LABEL_ENV_KEYS: tuple[str, ...] = ("TABNETICS_HOST_LABEL", "TABNETICS_RUN_HOST", "HOSTNAME")


def default_host_calibration_path(repo_root: str | Path | None = None) -> Path:
    raw = os.environ.get(HOST_CALIBRATION_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd()
    return root / "run_artifacts" / "meta" / "host_calibration.json"


def resolve_host_calibration_path(raw_path: str | Path | None, *, repo_root: str | Path | None = None) -> Path:
    raw = str(raw_path or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return default_host_calibration_path(repo_root)


def load_catalog(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {
            "schema_version": HOST_CALIBRATION_SCHEMA_VERSION,
            "records": [],
        }
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid host calibration catalog: {p}")
    if str(data.get("schema_version") or "") != HOST_CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported host calibration schema in {p}: {data.get('schema_version')!r}"
        )
    records = data.get("records")
    if records is None:
        data["records"] = []
    elif not isinstance(records, list):
        raise ValueError(f"Invalid host calibration catalog records in {p}")
    return data


def append_records(path: str | Path, records: Sequence[Mapping[str, Any]]) -> Path:
    p = Path(path)
    payload = load_catalog(p)
    existing = list(payload.get("records") or [])
    new_records = [dict(record) for record in records]
    if not new_records:
        return p
    payload.update(
        {
            "schema_version": HOST_CALIBRATION_SCHEMA_VERSION,
            "updated_at": utc_now_iso(),
            "records": [*existing, *new_records],
        }
    )
    return write_json(p, payload)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_from_provenance(payload: Mapping[str, Any]) -> dict[str, str]:
    env = payload.get("environment") or {}
    return {str(k): str(v) for k, v in dict(env).items()}


def _normalize_host_label(value: str) -> str:
    host = str(value or "").strip()
    if host in {"public_cpu_host_1", "public_cpu_host_2", "public_cpu_host_3", "public_cpu_host_4"}:
        return f"{host}.example.com"
    return host or "unknown"


def host_label_from_provenance(payload: Mapping[str, Any]) -> str:
    env = _env_from_provenance(payload)
    for key in HOST_LABEL_ENV_KEYS:
        value = str(env.get(key, "") or "").strip()
        if value:
            return _normalize_host_label(value)
    host = dict(payload.get("host") or {})
    return _normalize_host_label(str(host.get("hostname") or host.get("platform_node") or "unknown"))


def infer_method_family(job: Mapping[str, Any]) -> str:
    params = dict(job.get("params") or {})
    explicit = str(params.get("calibration_method_family") or "").strip().lower()
    if explicit:
        return explicit
    kind = str(job.get("kind") or "").strip().lower()
    lane = str(params.get("execution_lane") or "").strip().lower()
    if lane in {"tabpfn", "gpu"}:
        return "tabpfn"
    if kind == "tabarena_benchmark":
        return "tabarena"
    if kind == "validation_suite":
        return "validation_suite"
    return "cpu"


def _dataset_tier_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    data_identity = dict(payload.get("data_identity") or {})
    counts: dict[str, int] = {}
    for record in list(data_identity.get("datasets") or []):
        tier = str(dict(record).get("tier") or "unknown").strip().lower() or "unknown"
        counts[tier] = counts.get(tier, 0) + 1
    if counts:
        return counts
    dataset_ids = list(data_identity.get("dataset_ids") or [])
    if dataset_ids:
        return {"unknown": len(dataset_ids)}
    return {"unknown": 1}


def dataset_tier_key(tier_counts: Mapping[str, int]) -> str:
    tiers = sorted(str(tier) for tier, count in dict(tier_counts).items() if int(count) > 0)
    if len(tiers) == 1:
        return tiers[0]
    return "mixed"


def build_job_telemetry(
    *,
    base: Mapping[str, Any],
    job: Mapping[str, Any],
    status: str,
    wall_time_sec: float,
    exit_code: int | None,
    max_workers: int,
    usage_delta: Mapping[str, Any],
    peak_rss_source: str = "job_child_process",
) -> dict[str, Any]:
    env = _env_from_provenance(base)
    pods_per_host = _safe_int(env.get("PODS_PER_HOST"), 1)
    tier_counts = _dataset_tier_counts(base)
    dataset_count = int(sum(int(v) for v in tier_counts.values()))
    wall = max(0.0, _safe_float(wall_time_sec))
    peak_rss_kb = _safe_int(usage_delta.get("peak_rss_kb"), 0)
    status_txt = str(status or "")
    oom = bool(exit_code in {137, -9} or "oom" in status_txt.lower())
    throughput = float(dataset_count * 3600.0 / wall) if wall > 0.0 else 0.0
    return {
        "schema_version": HOST_CALIBRATION_SCHEMA_VERSION,
        "observed_at": utc_now_iso(),
        "host": host_label_from_provenance(base),
        "method_family": infer_method_family(job),
        "dataset_tier": dataset_tier_key(tier_counts),
        "dataset_tier_counts": {str(k): int(v) for k, v in sorted(tier_counts.items())},
        "job_id": str(job.get("job_id") or ""),
        "job_kind": str(job.get("kind") or ""),
        "status": status_txt,
        "exit_code": None if exit_code is None else int(exit_code),
        "oom": oom,
        "wall_time_sec": float(wall),
        "dataset_count": int(dataset_count),
        "throughput_datasets_per_hour": float(throughput),
        "max_workers_per_pod": int(max_workers),
        "pods_per_host": int(max(1, pods_per_host)),
        "target_total_workers": int(max(1, pods_per_host) * max(0, int(max_workers))),
        "peak_rss_kb": int(peak_rss_kb),
        "peak_rss_mb": float(peak_rss_kb / 1024.0) if peak_rss_kb > 0 else 0.0,
        "peak_rss_source": str(peak_rss_source or ""),
    }


def records_from_job_provenance(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    telemetry = dict(payload.get("job_telemetry") or {})
    if not telemetry:
        return []
    if str(telemetry.get("schema_version") or "") != HOST_CALIBRATION_SCHEMA_VERSION:
        return []
    if str(telemetry.get("status") or "") in {"dry_run", "skipped_done", "not_run"}:
        return []
    base_record = dict(telemetry)
    counts = dict(base_record.get("dataset_tier_counts") or {})
    tiers = [str(tier) for tier, count in counts.items() if _safe_int(count) > 0]
    if not tiers:
        tiers = [str(base_record.get("dataset_tier") or "unknown")]
    if "mixed" not in tiers:
        tiers.append("mixed")
    out: list[dict[str, Any]] = []
    for tier in sorted(set(tiers)):
        record = dict(base_record)
        record["dataset_tier"] = str(tier)
        record["calibration_key"] = {
            "host": str(record.get("host") or ""),
            "method_family": str(record.get("method_family") or ""),
            "dataset_tier": str(tier),
        }
        out.append(record)
    return out


def collect_records_from_root(root_out: str | Path) -> list[dict[str, Any]]:
    root = Path(root_out)
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for path in sorted(root.rglob("provenance.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.extend(records_from_job_provenance(payload))
    return records


def latest_record(
    catalog: Mapping[str, Any],
    *,
    host: str,
    method_family: str,
    dataset_tier: str = "mixed",
) -> dict[str, Any] | None:
    wanted = (str(host), str(method_family), str(dataset_tier))
    fallbacks = [
        wanted,
        (wanted[0], wanted[1], "mixed"),
        (wanted[0], wanted[1], "*"),
    ]
    records = [dict(record) for record in list(catalog.get("records") or [])]
    for host_key, method_key, tier_key in fallbacks:
        matches = [
            record
            for record in records
            if str(record.get("host") or "") == host_key
            and str(record.get("method_family") or "") == method_key
            and str(record.get("dataset_tier") or "") == tier_key
        ]
        if matches:
            return sorted(matches, key=lambda r: str(r.get("observed_at") or ""))[-1]
    return None


def merge_worker_targets_with_catalog(
    *,
    defaults: Mapping[str, Mapping[str, Any]],
    catalog_path: str | Path | None,
    cpu_method_family: str = "cpu",
    gpu_hosts: Iterable[str] = (),
    gpu_method_family: str = "tabpfn",
    dataset_tier: str = "mixed",
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, Any]]]:
    targets = {
        str(host): {
            "pods_per_host": _safe_int(values.get("pods_per_host"), 0),
            "max_workers_per_pod": _safe_int(values.get("max_workers_per_pod"), 0),
            "target_total_workers": _safe_int(values.get("target_total_workers"), 0),
        }
        for host, values in dict(defaults).items()
    }
    sources: dict[str, dict[str, Any]] = {
        str(host): {"source": "fallback_static"} for host in targets
    }
    if catalog_path is None:
        return targets, sources
    p = Path(catalog_path)
    if not p.exists():
        return targets, sources

    catalog = load_catalog(p)
    gpu_set = {str(host) for host in gpu_hosts}
    for host in list(targets.keys()):
        method = gpu_method_family if host in gpu_set else cpu_method_family
        record = latest_record(catalog, host=host, method_family=method, dataset_tier=dataset_tier)
        if record is None:
            continue
        pods = max(1, _safe_int(record.get("pods_per_host"), targets[host]["pods_per_host"] or 1))
        workers = max(1, _safe_int(record.get("max_workers_per_pod"), targets[host]["max_workers_per_pod"] or 1))
        if bool(record.get("oom", False)) and workers > 1:
            workers -= 1
        targets[host] = {
            "pods_per_host": int(pods),
            "max_workers_per_pod": int(workers),
            "target_total_workers": int(pods * workers),
        }
        sources[host] = {
            "source": "host_calibration_catalog",
            "catalog_path": str(p),
            "observed_at": str(record.get("observed_at") or ""),
            "method_family": str(record.get("method_family") or method),
            "dataset_tier": str(record.get("dataset_tier") or dataset_tier),
            "throughput_datasets_per_hour": _safe_float(record.get("throughput_datasets_per_hour"), 0.0),
            "peak_rss_mb": _safe_float(record.get("peak_rss_mb"), 0.0),
            "oom": bool(record.get("oom", False)),
        }
    return targets, sources

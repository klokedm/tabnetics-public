from __future__ import annotations

import json
from pathlib import Path

from tabnetics.validation.core.aggregate import aggregate
from tabnetics.validation.core.host_calibration import (
    HOST_CALIBRATION_SCHEMA_VERSION,
    append_records,
    build_job_telemetry,
    latest_record,
    load_catalog,
    merge_worker_targets_with_catalog,
    records_from_job_provenance,
)
from tabnetics.validation.generate_plan import (
    Job,
    VAL19_HOST_WORKER_TARGETS,
    _validation19_recommended_host_assignment,
)


def _base_provenance() -> dict:
    return {
        "host": {"hostname": "public_cpu_host_1"},
        "environment": {"PODS_PER_HOST": "2"},
        "data_identity": {
            "dataset_ids": ["leukemia_golub"],
            "datasets": [
                {
                    "dataset_id": "leukemia_golub",
                    "tier": "hard",
                    "registered": True,
                }
            ],
        },
    }


def _job() -> dict:
    return {
        "job_id": "val19_cpu/V19_C01/ds01",
        "kind": "run_df_fs_sota_benchmark",
        "params": {
            "datasets": ["leukemia_golub"],
            "fs_method_set": "strict_plus_mrmr",
        },
    }


def _telemetry_payload() -> dict:
    telemetry = build_job_telemetry(
        base=_base_provenance(),
        job=_job(),
        status="ok",
        wall_time_sec=1800.0,
        exit_code=0,
        max_workers=4,
        usage_delta={"peak_rss_kb": 2_048_000},
    )
    return {"job_telemetry": telemetry}


def test_job_telemetry_records_throughput_memory_and_tier() -> None:
    records = records_from_job_provenance(_telemetry_payload())

    assert {rec["dataset_tier"] for rec in records} == {"hard", "mixed"}
    rec = next(rec for rec in records if rec["dataset_tier"] == "hard")
    assert rec["schema_version"] == HOST_CALIBRATION_SCHEMA_VERSION
    assert rec["host"] == "host1.example.com"
    assert rec["method_family"] == "cpu"
    assert rec["dataset_tier"] == "hard"
    assert rec["throughput_datasets_per_hour"] == 2.0
    assert rec["peak_rss_mb"] == 2000.0
    assert rec["peak_rss_source"] == "job_child_process"
    assert rec["max_workers_per_pod"] == 4
    assert rec["pods_per_host"] == 2
    assert rec["target_total_workers"] == 8
    assert rec["oom"] is False


def test_catalog_append_latest_and_worker_target_merge(tmp_path: Path) -> None:
    catalog_path = tmp_path / "host_calibration.json"
    records = records_from_job_provenance(_telemetry_payload())

    append_records(catalog_path, records)
    catalog = load_catalog(catalog_path)
    latest = latest_record(
        catalog,
        host="host1.example.com",
        method_family="cpu",
        dataset_tier="hard",
    )

    assert latest is not None
    assert latest["max_workers_per_pod"] == 4

    targets, sources = merge_worker_targets_with_catalog(
        defaults={
            "host1.example.com": {
                "pods_per_host": 1,
                "max_workers_per_pod": 1,
                "target_total_workers": 1,
            }
        },
        catalog_path=catalog_path,
        dataset_tier="hard",
    )

    assert targets["host1.example.com"] == {
        "pods_per_host": 2,
        "max_workers_per_pod": 4,
        "target_total_workers": 8,
    }
    assert sources["host1.example.com"]["source"] == "host_calibration_catalog"
    assert sources["host1.example.com"]["peak_rss_mb"] == 2000.0


def test_oom_catalog_record_steps_down_worker_target(tmp_path: Path) -> None:
    catalog_path = tmp_path / "host_calibration.json"
    record = records_from_job_provenance(_telemetry_payload())[0]
    record["oom"] = True
    append_records(catalog_path, [record])

    targets, sources = merge_worker_targets_with_catalog(
        defaults={
            "host1.example.com": {
                "pods_per_host": 2,
                "max_workers_per_pod": 4,
                "target_total_workers": 8,
            }
        },
        catalog_path=catalog_path,
        dataset_tier="hard",
    )

    assert targets["host1.example.com"]["max_workers_per_pod"] == 3
    assert targets["host1.example.com"]["target_total_workers"] == 6
    assert sources["host1.example.com"]["oom"] is True


def test_aggregate_appends_host_calibration_records(tmp_path: Path) -> None:
    root_out = tmp_path / "run"
    job_dir = root_out / "val19_cpu/V19_C01/ds01"
    job_dir.mkdir(parents=True)
    (job_dir / "provenance.json").write_text(json.dumps(_telemetry_payload()), encoding="utf-8")

    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "val19_cpu/V19_C01/ds01",
                        "kind": "run_df_fs_sota_benchmark",
                        "params": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    catalog_path = tmp_path / "meta" / "host_calibration.json"

    aggregate(
        root_out,
        plan_path=plan,
        out_dir=tmp_path / "aggregate",
        host_calibration_path=catalog_path,
    )

    catalog = load_catalog(catalog_path)
    assert len(catalog["records"]) == 2
    assert catalog["records"][0]["host"] == "host1.example.com"


def test_generate_plan_host_assignment_uses_catalog_with_static_fallback(tmp_path: Path) -> None:
    catalog_path = tmp_path / "host_calibration.json"
    append_records(catalog_path, records_from_job_provenance(_telemetry_payload()))

    job = Job(
        job_id="val19_cpu/V19_C01/ds01",
        kind="run_df_fs_sota_benchmark",
        params={"datasets": ["leukemia_golub"]},
        weight=1.0,
    )
    assignment = _validation19_recommended_host_assignment(
        {1: [job.job_id]},
        {job.job_id: job},
        host_calibration_path=catalog_path,
    )
    host_summary = dict(assignment["host_summary"])

    assert host_summary["host1.example.com"]["worker_target"] == {
        "pods_per_host": 2,
        "max_workers_per_pod": 4,
        "target_total_workers": 8,
    }
    assert host_summary["host1.example.com"]["worker_target_source"]["source"] == "host_calibration_catalog"
    assert host_summary["host2.example.com"]["worker_target"] == VAL19_HOST_WORKER_TARGETS["host2.example.com"]

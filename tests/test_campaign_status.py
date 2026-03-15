from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/analysis/campaign_status.py"
MODULE_SPEC = importlib.util.spec_from_file_location("campaign_status", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
campaign_status = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = campaign_status
MODULE_SPEC.loader.exec_module(campaign_status)

CampaignSpec = campaign_status.CampaignSpec
JobPlan = campaign_status.JobPlan
build_benchmark_snapshot = campaign_status.build_benchmark_snapshot
build_tabarena_snapshot = campaign_status.build_tabarena_snapshot
parse_status_filename = campaign_status.parse_status_filename


def _dummy_campaign(slug: str, kind: str = "benchmark") -> CampaignSpec:
    return CampaignSpec(
        slug=slug,
        display_name=slug,
        kind=kind,
        plan_files=(),
        shard_files=(),
        remote_patterns={},
        local_roots={},
        tmux_patterns=(re.compile(r".*"),),
        service_patterns=(),
    )


def test_parse_status_filename_handles_job_and_shard_markers() -> None:
    job = parse_status_filename("val19_classifiers__C_ONLY_cpda__ds03.DONE.ok")
    assert job == {
        "kind": "job",
        "job_id": "val19_classifiers/C_ONLY_cpda/ds03",
        "result": "ok",
    }

    shard = parse_status_filename("shard503.DONE.fail")
    assert shard == {
        "kind": "shard",
        "shard_id": 503,
        "state": "DONE",
        "result": "fail",
    }

    assert parse_status_filename("README.md") is None


def test_build_benchmark_snapshot_rolls_up_status_and_eta() -> None:
    campaign = _dummy_campaign("val19")
    plan_jobs = {
        "val19_classifiers/C_ONLY_cpda/ds01": JobPlan(
            job_id="val19_classifiers/C_ONLY_cpda/ds01",
            family="classifiers",
            profile="C_ONLY_cpda",
            dataset_tag="ds01",
            dataset_name="xena_tcga_brca",
            expected_runs=5,
            weight=10.0,
            source_files=["plan.json"],
        ),
        "val19_classifiers/C_ONLY_realmlp/ds01": JobPlan(
            job_id="val19_classifiers/C_ONLY_realmlp/ds01",
            family="classifiers",
            profile="C_ONLY_realmlp",
            dataset_tag="ds01",
            dataset_name="xena_tcga_brca",
            expected_runs=5,
            weight=20.0,
            source_files=["plan.json"],
        ),
        "val19_classifiers/V19_C01_old_regime_legacy_full64/ds01": JobPlan(
            job_id="val19_classifiers/V19_C01_old_regime_legacy_full64/ds01",
            family="classifiers",
            profile="V19_C01_old_regime_legacy_full64",
            dataset_tag="ds01",
            dataset_name="xena_tcga_brca",
            expected_runs=5,
            weight=30.0,
            source_files=["plan.json"],
        ),
    }
    shard_jobs = {
        ("classifiers", 1): [
            "val19_classifiers/C_ONLY_cpda/ds01",
            "val19_classifiers/C_ONLY_realmlp/ds01",
        ],
        ("classifiers", 2): [
            "val19_classifiers/V19_C01_old_regime_legacy_full64/ds01",
        ],
    }
    now_epoch = 28_800.0
    markers = [
        {
            "type": "shard",
            "host": "lab01",
            "source": "remote",
            "root": "/tmp/classifiers",
            "rel_path": "_status/shard01.STARTED",
            "timestamp": 7_200.0,
            "family": "classifiers",
            "shard_id": 1,
            "state": "STARTED",
            "result": None,
        },
        {
            "type": "shard",
            "host": "lab02",
            "source": "remote",
            "root": "/tmp/classifiers",
            "rel_path": "_status/shard02.STARTED",
            "timestamp": 7_200.0,
            "family": "classifiers",
            "shard_id": 2,
            "state": "STARTED",
            "result": None,
        },
        {
            "type": "job",
            "host": "lab01",
            "source": "remote",
            "root": "/tmp/classifiers",
            "rel_path": "_status/val19_classifiers__C_ONLY_cpda__ds01.DONE.ok",
            "timestamp": 25_200.0,
            "family": "classifiers",
            "job_id": "val19_classifiers/C_ONLY_cpda/ds01",
            "result": "ok",
        },
        {
            "type": "job",
            "host": "lab01",
            "source": "remote",
            "root": "/tmp/classifiers",
            "rel_path": "_status/val19_classifiers__C_ONLY_realmlp__ds01.DONE.fail",
            "timestamp": 20_000.0,
            "family": "classifiers",
            "job_id": "val19_classifiers/C_ONLY_realmlp/ds01",
            "result": "fail",
        },
    ]
    handle_matches = {"lab01": {"tmux": ["val19_l01_p1"], "services": []}}

    snapshot = build_benchmark_snapshot(
        campaign=campaign,
        plan_jobs=plan_jobs,
        shard_jobs=shard_jobs,
        markers=markers,
        handle_matches=handle_matches,
        now_epoch=now_epoch,
        eta_window_hours=2.0,
        warnings=[],
        timestamp_source="remote",
    )

    summary = snapshot["summary"]
    assert summary["total_jobs"] == 3
    assert summary["done_jobs"] == 1
    assert summary["fail_marked_jobs"] == 1
    assert summary["open_jobs"] == 2
    assert summary["progress_count_pct"] == pytest.approx(100.0 / 3.0)
    assert summary["progress_weight_pct"] == pytest.approx((10.0 / 60.0) * 100.0)
    assert summary["live_tmux_handles"] == 1
    assert summary["live_service_handles"] == 0
    assert summary["start_timestamp"] is not None
    assert summary["eta_weight_hours"] == pytest.approx(10.0)

    job_status = {job["job_id"]: job["status"] for job in snapshot["jobs"]}
    assert job_status["val19_classifiers/C_ONLY_cpda/ds01"] == "done"
    assert job_status["val19_classifiers/C_ONLY_realmlp/ds01"] == "fail_marked"
    assert job_status["val19_classifiers/V19_C01_old_regime_legacy_full64/ds01"] == "pending"

    shard_open = {(shard["family"], shard["shard_id"]): shard["open_jobs"] for shard in snapshot["shards"]}
    assert shard_open[("classifiers", 1)] == 1
    assert shard_open[("classifiers", 2)] == 1


def test_build_tabarena_snapshot_rolls_up_tasks() -> None:
    campaign = _dummy_campaign("tabarena", kind="tabarena")
    shards = [
        {
            "host": "lab01",
            "timestamp": 1_000.0,
            "completed": 25,
            "total_tasks": 100,
            "running": 1,
            "queue_depth": "0",
            "status": "running",
            "no_done_for_sec": 30.0,
            "oldest_running": "task1",
            "updated_at_epoch": 8_000.0,
            "task_shard_index": 0,
            "task_shard_count": 2,
            "max_workers": 1,
            "profile": "general_tabular",
            "dataset_sets": ["all"],
            "protocol": "openml_task",
            "stage_counts": {"pipeline_run": 1},
            "active_tasks": [{"dataset_id": "APSFailure", "current_stage": "pipeline_run"}],
            "last_completed_task": {},
        },
        {
            "host": "lab02",
            "timestamp": 1_000.0,
            "completed": 50,
            "total_tasks": 100,
            "running": 2,
            "queue_depth": "0",
            "status": "running",
            "no_done_for_sec": 45.0,
            "oldest_running": "task2",
            "updated_at_epoch": 8_050.0,
            "task_shard_index": 1,
            "task_shard_count": 2,
            "max_workers": 2,
            "profile": "general_tabular",
            "dataset_sets": ["all"],
            "protocol": "openml_task",
            "stage_counts": {"dataset_load": 2},
            "active_tasks": [{"dataset_id": "Bioresponse", "current_stage": "dataset_load"}],
            "last_completed_task": {"dataset_id": "adult", "status": "ok"},
        },
    ]
    handle_matches = {"lab01": {"tmux": ["tabarena_general_tabular_official"], "services": []}}

    snapshot = build_tabarena_snapshot(
        campaign=campaign,
        shard_heartbeats=shards,
        handle_matches=handle_matches,
        now_epoch=8_200.0,
        warnings=[],
        timestamp_source="remote",
    )

    summary = snapshot["summary"]
    assert summary["total_tasks"] == 200
    assert summary["completed_tasks"] == 75
    assert summary["open_tasks"] == 125
    assert summary["progress_pct"] == pytest.approx(37.5)
    assert summary["live_tmux_handles"] == 1
    assert summary["eta_hours"] == pytest.approx(10.0 / 3.0)
    assert snapshot["shards"][0]["stage_counts"] == {"pipeline_run": 1}
    assert snapshot["shards"][1]["last_completed_task"]["dataset_id"] == "adult"

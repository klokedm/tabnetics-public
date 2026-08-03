from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import tabnetics.benchmarks.beyondarena_compare as beyondarena_compare
from tabnetics.benchmarks.beyondarena_compare import OFFICIAL_BEST_TFM_METHOD
from tabnetics.benchmarks.beyondarena_materialize import build_beyondarena_materialization_plan
from tabnetics.benchmarks.beyondarena_plan import (
    align_beyondarena_results_to_manifest,
    build_beyondarena_local_execution_plan,
    build_beyondarena_run_spec,
    build_pending_local_result_rows,
    default_beyondarena_run_profiles,
    main,
    resolve_beyondarena_task_rows,
    task_rows_to_manifest_frame,
    write_beyondarena_plan_artifacts,
)


FIXTURES = Path(__file__).parent / "fixtures" / "beyondarena" / "official_metadata"
TASKS = FIXTURES / "BeyondArena_tasks_metadata.csv.fixture"
CORE = FIXTURES / "BeyondArena_core_tasks.csv.fixture"


def test_resolve_task_rows_builds_core_and_smoke_scopes() -> None:
    core_rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="core",
    )
    core_all_splits_rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="core-all-splits",
    )
    core_classification_all_splits_rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="core-classification-all-splits",
    )
    smoke_rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    feasibility_rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="current-feasibility",
    )

    assert len(core_rows) == 6
    assert len(core_all_splits_rows) == 10
    assert len(core_classification_all_splits_rows) == 7
    assert {row.tabarena_task_name for row in core_all_splits_rows} == {
        row.tabarena_task_name for row in core_rows
    }
    assert {row.normalized_problem_type for row in core_classification_all_splits_rows} == {
        "classification"
    }
    assert {row.tabarena_task_name for row in core_classification_all_splits_rows} == {
        "grouped_energy-bbbb",
        "temporal_sales-cccc",
        "text_reviews-dddd",
        "merchant_high_card-eeee",
    }
    assert len(smoke_rows) == 6
    assert len({row.tabarena_task_name for row in smoke_rows}) == 6
    assert {row.normalized_task_type for row in smoke_rows} == {"iid", "grouped", "temporal"}
    assert len(feasibility_rows) == 1
    assert feasibility_rows[0].normalized_problem_type == "classification"
    assert feasibility_rows[0].normalized_task_type == "grouped"

    manifest = task_rows_to_manifest_frame(smoke_rows)
    assert manifest["dataset_id"].is_unique
    assert set(manifest["dimensionality"]) == {"low", "high"}
    assert manifest["has_text"].any()
    assert manifest["high_cardinality"].any()


def test_plan_writer_records_pending_status_without_result_tables(tmp_path: Path) -> None:
    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path,
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )

    tasks = pd.read_csv(artifacts.task_manifest)
    models = pd.read_csv(artifacts.model_manifest)
    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    audit = json.loads(artifacts.readiness_audit.read_text(encoding="utf-8"))

    assert len(tasks) == 6
    assert {"tabnetics-current", "TabenticsDiakrino", "TabICLv2"}.issubset(set(models["method"]))
    expected_model_manifest_columns = {
        "sample_limit",
        "feature_limit",
        "compatibility_scope",
        "execution_guard",
        "fallback_status",
    }
    assert expected_model_manifest_columns.issubset(set(models.columns))
    tabpfn = models[models["method"].eq("TabPFN-2.6")].iloc[0]
    assert "installed optional package" in str(tabpfn["compatibility_scope"])
    assert "deferred_gpu_revalidation" in str(tabpfn["fallback_status"])
    assert status["comparison_ready"] is False
    assert status["exact_paired_rows_available"] is False
    assert status["task_count"] == 6
    coverage = status["manifest_coverage"]
    assert coverage["smoke_coverage_ready"] is True
    assert set(coverage["smoke_coverage"]) == {
        "iid",
        "grouped",
        "temporal",
        "text",
        "high_cardinality",
        "high_dimensional",
    }
    assert all(coverage["smoke_coverage"].values())
    assert coverage["smoke_missing_facets"] == []
    assert coverage["task_type_counts"] == {"iid": 4, "grouped": 1, "temporal": 1}
    assert coverage["text_task_count"] >= 1
    split_stability = coverage["split_stability"]
    assert split_stability["split_id_source"] == "official_split_id"
    assert split_stability["per_dataset_split_stability_ready"] is False
    assert split_stability["split_stability_claims_ready"] is False
    assert split_stability["split_stability_claim_eligible_dataset_count"] == 0
    assert len(split_stability["datasets_with_single_split"]) == 6
    assert "split-stability claims require more than one planned split" in split_stability["split_stability_blocker"]
    assert status["readiness_audit"] == audit
    assert audit["overall_status"] == "blocked"
    assert audit["acceptance_checks"]["manifest_rows_ready"] is True
    assert audit["acceptance_checks"]["paired_comparison_ready"] is False
    assert audit["claims"]["performance_claims_ready"] is False
    assert "local result rows are not configured/materialized" in audit["blockers"]["performance_claims"]
    assert (
        "exact official per-dataset/per-split result rows are not configured"
        in audit["blockers"]["performance_claims"]
    )
    assert "run_spec.json was not emitted for lab handoff" in audit["blockers"]["lab_launch"]


def test_plan_writer_records_split_stability_counts_for_all_metadata(tmp_path: Path) -> None:
    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path,
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="all",
    )

    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    split_stability = status["manifest_coverage"]["split_stability"]

    assert split_stability["dataset_split_counts"]["grouped_energy-bbbb"] == 3
    assert split_stability["dataset_split_counts"]["temporal_sales-cccc"] == 2
    assert "grouped_energy-bbbb" in split_stability["datasets_with_multiple_splits"]
    assert "wide_genomics-ffff" in split_stability["datasets_with_single_split"]
    assert split_stability["per_dataset_split_stability_ready"] is False
    assert split_stability["split_stability_claims_ready"] is True
    assert set(split_stability["datasets_eligible_for_split_stability_claims"]) == {
        "grouped_energy-bbbb",
        "simple_iid_regression-aaaa",
        "temporal_sales-cccc",
    }
    assert "wide_genomics-ffff" in split_stability["datasets_excluded_from_split_stability_claims"]
    assert "single-split datasets must be excluded" in split_stability["split_stability_limit"]


def test_plan_writer_core_all_splits_expands_core_datasets_for_stage2(tmp_path: Path) -> None:
    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path,
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="core-all-splits",
    )

    manifest = pd.read_csv(artifacts.task_manifest)
    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    split_stability = status["manifest_coverage"]["split_stability"]

    assert len(manifest) == 10
    assert status["subset"] == "core-all-splits"
    assert split_stability["dataset_split_counts"]["grouped_energy-bbbb"] == 3
    assert split_stability["dataset_split_counts"]["temporal_sales-cccc"] == 2
    assert "grouped_energy-bbbb" in split_stability["datasets_with_multiple_splits"]
    assert "wide_genomics-ffff" in split_stability["datasets_with_single_split"]
    assert split_stability["split_stability_claims_ready"] is True


def test_plan_writer_core_classification_all_splits_filters_stage2_regression_rows(
    tmp_path: Path,
) -> None:
    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path,
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="core-classification-all-splits",
        emit_local_execution_plan=True,
        emit_pending_local_rows=True,
        emit_run_spec=True,
        public_cpu_host_1_max_workers=29,
        public_cpu_host_1_pods_per_host=2,
        public_cpu_host_2_max_workers=21,
        public_cpu_host_2_pods_per_host=2,
    )

    manifest = pd.read_csv(artifacts.task_manifest)
    execution_plan = pd.read_csv(artifacts.local_execution_plan)
    pending = pd.read_csv(artifacts.pending_local)
    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    audit = json.loads(artifacts.readiness_audit.read_text(encoding="utf-8"))
    run_spec = json.loads(artifacts.run_spec.read_text(encoding="utf-8"))

    assert len(manifest) == 7
    assert status["subset"] == "core-classification-all-splits"
    assert set(manifest["problem_type"]) == {"classification"}
    assert set(manifest["dataset_id"]) == {
        "grouped_energy-bbbb",
        "temporal_sales-cccc",
        "text_reviews-dddd",
        "merchant_high_card-eeee",
    }
    assert status["manifest_coverage"]["problem_type_counts"] == {"classification": 7}
    split_stability = status["manifest_coverage"]["split_stability"]
    assert split_stability["dataset_split_counts"]["grouped_energy-bbbb"] == 3
    assert split_stability["dataset_split_counts"]["temporal_sales-cccc"] == 2
    assert split_stability["split_stability_claims_ready"] is True
    assert split_stability["per_dataset_split_stability_ready"] is False
    assert audit["acceptance_checks"]["split_stability_claims_ready"] is True
    assert audit["claims"]["split_stability_claims_ready"] is True
    assert audit["blockers"]["split_stability_claims"] == []

    assert len(execution_plan) == 14
    assert "skipped_unsupported_regression" not in set(execution_plan["execution_status"])
    current = execution_plan[execution_plan["method"].eq("tabnetics-current")]
    diakrino = execution_plan[execution_plan["method"].eq("TabenticsDiakrino")]
    assert len(current) == 7
    assert set(current["execution_status"]) == {"ready_after_artifact_materialization"}
    assert current["runnable"].eq(True).all()
    assert len(diakrino) == 7
    assert set(diakrino["execution_status"]) == {"deferred_gpu_revalidation"}
    assert diakrino["runnable"].eq(False).all()

    assert len(pending) == 14
    assert "skipped_unsupported_regression" not in set(pending["execution_status"])
    assert audit["local_execution_plan"]["unsupported_regression_rows"] == 0
    assert audit["local_execution_plan"]["ready_after_artifact_materialization_rows"] == 7
    assert audit["local_execution_plan"]["deferred_gpu_revalidation_rows"] == 7
    assert (
        "BeyondArena regression rows are unsupported by the tabnetics/DIAKRINO local execution plan"
        not in audit["blockers"]["lab_launch"]
    )
    assert (
        "GPU-required DIAKRINO rows remain deferred pending public-gpu-host revalidation"
        in audit["blockers"]["lab_launch"]
    )
    assert run_spec["subset"] == "core-classification-all-splits"
    assert run_spec["classification_task_count"] == 7
    assert run_spec["regression_task_count"] == 0
    shard_plan = run_spec["local_current_shard_plan"]
    assert len(shard_plan) == 4
    assert [row["planned_manifest_rows"] for row in shard_plan] == [2, 2, 2, 1]
    assert [row["planned_classification_rows"] for row in shard_plan] == [2, 2, 2, 1]
    assert [row["assigned_host"] for row in shard_plan] == [
        "host1.example.com",
        "host1.example.com",
        "host2.example.com",
        "host2.example.com",
    ]


def test_plan_writer_reports_missing_smoke_facets_when_subset_is_limited(tmp_path: Path) -> None:
    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path,
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
        max_smoke_items=3,
    )

    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    coverage = status["manifest_coverage"]

    assert coverage["task_count"] == 3
    assert coverage["smoke_coverage_ready"] is False
    assert {"text", "high_cardinality", "high_dimensional"}.issubset(set(coverage["smoke_missing_facets"]))


def test_local_execution_plan_separates_classification_gpu_and_regression() -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)

    plan = build_beyondarena_local_execution_plan(manifest)

    assert len(plan) == 12
    assert set(plan["method"]) == {"tabnetics-current", "TabenticsDiakrino"}

    regression = plan[plan["problem_type"].eq("regression")]
    assert not regression.empty
    assert set(regression["execution_status"]) == {"skipped_unsupported_regression"}
    assert regression["runnable"].eq(False).all()
    assert regression["skip_reason"].str.contains("classification-only").all()
    assert set(regression["target_host"]) == {"public_cpu_host_1/public_cpu_host_2", "public-gpu-host"}

    current_cls = plan[
        plan["problem_type"].eq("classification") & plan["method"].eq("tabnetics-current")
    ]
    assert not current_cls.empty
    assert set(current_cls["execution_status"]) == {"ready_after_artifact_materialization"}
    assert current_cls["runnable"].eq(True).all()
    assert set(current_cls["target_host"]) == {"public_cpu_host_1/public_cpu_host_2"}

    diakrino_cls = plan[
        plan["problem_type"].eq("classification") & plan["method"].eq("TabenticsDiakrino")
    ]
    assert not diakrino_cls.empty
    assert set(diakrino_cls["execution_status"]) == {"deferred_gpu_revalidation"}
    assert diakrino_cls["runnable"].eq(False).all()
    assert set(diakrino_cls["target_host"]) == {"public-gpu-host"}

    revalidated = build_beyondarena_local_execution_plan(
        manifest,
        arch_ml_revalidated=True,
        tabentics_diakrino_checkpoint_ready=True,
    )
    revalidated_diakrino_cls = revalidated[
        revalidated["problem_type"].eq("classification")
        & revalidated["method"].eq("TabenticsDiakrino")
    ]
    assert set(revalidated_diakrino_cls["execution_status"]) == {"ready_after_artifact_materialization"}
    assert revalidated_diakrino_cls["runnable"].eq(True).all()


def test_pending_local_rows_represent_deferred_tabnetics_profiles() -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)

    pending = build_pending_local_result_rows(manifest)

    assert len(pending) == 12
    assert set(pending["method"]) == {"tabnetics-current", "TabenticsDiakrino"}
    assert set(pending["status"]) == {"skipped"}
    assert "execution_status" in pending.columns
    assert pending["skip_reason"].str.len().gt(0).all()
    assert pending["skip_reason"].str.contains("classification-only").any()
    assert pending["skip_reason"].str.contains("public-gpu-host").any()
    assert set(pending["split_id"]).issubset(set(manifest["official_split_id"].astype(str)))
    assert set(pending["local_split_id"]).issubset(set(manifest["split_id"]))
    assert set(pending["execution_host"]) == {"public_cpu_host_1/public_cpu_host_2", "public-gpu-host"}
    assert set(pending["execution_lane"]) == {"cpu", "gpu"}
    assert set(pending["origin"]) == {"tabnetics_local_beyondarena_pending"}
    assert "lower_is_better" in pending.columns


def test_pending_local_rows_respect_arch_ml_revalidation() -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)

    pending = build_pending_local_result_rows(
        manifest,
        arch_ml_revalidated=True,
        tabentics_diakrino_checkpoint_ready=True,
    )
    diakrino_classification = pending[
        pending["method"].eq("TabenticsDiakrino")
        & pending["execution_status"].eq("ready_after_artifact_materialization")
    ]

    assert not diakrino_classification.empty
    assert diakrino_classification["skip_reason"].eq(
        "not executed; requires materialized BeyondArena artifacts and lab execution"
    ).all()
    assert diakrino_classification["execution_host"].eq("public-gpu-host").all()
    assert diakrino_classification["execution_lane"].eq("gpu").all()
    assert diakrino_classification["allow_gpu_execution"].eq(True).all()


def test_run_spec_records_commands_host_caps_and_stop_conditions(tmp_path: Path) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)
    execution_plan = build_beyondarena_local_execution_plan(manifest)

    spec = build_beyondarena_run_spec(
        manifest,
        execution_plan,
        subset="smoke",
        out_dir=tmp_path / "smoke_plan",
        artifact_root=tmp_path / "artifact root",
        public_cpu_host_1_max_workers=29,
        public_cpu_host_1_pods_per_host=2,
        public_cpu_host_2_max_workers=21,
        public_cpu_host_2_pods_per_host=2,
    )

    hosts = {row["host"]: row for row in spec["host_allocation"]}
    assert hosts["host1.example.com"]["MAX_WORKERS"] == 29
    assert hosts["host1.example.com"]["PODS_PER_HOST"] == 2
    assert hosts["host2.example.com"]["MAX_WORKERS"] == 21
    assert hosts["public-gpu-host"]["status"] == "deferred_gpu_revalidation"
    assert hosts["public-gpu-host"]["MAX_WORKERS"] == 0
    assert "materialize_dataset" in spec["commands"]
    assert "compare_current" in spec["commands"]
    assert "compare_tabpfn_candidate" in spec["commands"]
    assert "compare_tabentics_diakrino" in spec["commands"]
    assert "--require-local-runner-ready" in spec["commands"]["materialize_dataset"]
    assert "--execution-host public_cpu_host_1/public_cpu_host_2" in spec["commands"]["local_current"]
    assert "--device cpu" in spec["commands"]["local_current"]
    assert "--max-workers 14" in spec["commands"]["local_current"]
    assert '--max-workers "${BEYONDARENA_MAX_WORKERS:-14}"' in spec["commands"]["local_current_sharded"]
    assert "--manifest-shard-index \"${BEYONDARENA_SHARD_INDEX}\"" in spec["commands"]["local_current_sharded"]
    assert "--manifest-shard-count 4" in spec["commands"]["local_current_sharded"]
    assert "local_current_results.shard_\"${BEYONDARENA_SHARD_INDEX}\".csv" in spec["commands"][
        "local_current_sharded"
    ]
    assert "--merge-shard-glob" in spec["commands"]["merge_current_shards"]
    assert "local_current_results.shard_*.csv" in spec["commands"]["merge_current_shards"]
    assert "--allow-gpu-execution" not in spec["commands"]["local_tabpfn_candidate"]
    assert "--allow-gpu-execution" not in spec["commands"]["local_tabentics_diakrino"]
    assert "--execution-host public-gpu-host" in spec["commands"]["local_tabpfn_candidate"]
    assert "--execution-lane gpu" in spec["commands"]["local_tabentics_diakrino"]
    assert "--manifest-shard-count 1" in spec["commands"]["local_tabpfn_candidate_sharded"]
    assert "--manifest-shard-count 1" in spec["commands"]["local_tabentics_diakrino_sharded"]
    assert "--merge-shard-glob" in spec["commands"]["merge_tabpfn_candidate_shards"]
    assert "--merge-shard-glob" in spec["commands"]["merge_tabentics_diakrino_shards"]
    assert "--tabnetics-diakrino-checkpoint ${TABENTICS_DIAKRINO_CHECKPOINT}" in spec["commands"]["local_tabentics_diakrino"]
    assert "--tabnetics-diakrino-max-features 1024" in spec["commands"]["local_tabentics_diakrino"]
    assert spec["inputs"]["tabentics_diakrino_checkpoint_env"] == "TABENTICS_DIAKRINO_CHECKPOINT"
    assert "artifact root" in spec["commands"]["materialize_dataset"]
    assert "'" in spec["commands"]["materialize_dataset"]
    assert "--official-results public-r2" in spec["commands"]["plan"]
    assert "--include-dataset --fetch-size-metadata" in spec["commands"]["materialize_dry_run"]
    assert spec["command_sequence"].index("materialize_dataset") < spec["command_sequence"].index("local_current")
    assert spec["command_sequence"].index("local_tabentics_diakrino") < spec["command_sequence"].index(
        "compare_tabentics_diakrino"
    )
    assert spec["sharded_command_sequence"].index("local_current_sharded") < spec["sharded_command_sequence"].index(
        "merge_current_shards"
    )
    assert spec["sharded_command_sequence"].index("merge_current_shards") < spec["sharded_command_sequence"].index(
        "compare_current"
    )
    gates = {row["gate"]: row for row in spec["prelaunch_gates"]}
    assert gates["materialization_local_runner_ready"]["ready_field"] == "artifact_local_runner_ready"
    assert gates["materialization_local_runner_ready"]["status"] == "must_pass_before_local_result_rows"
    assert gates["sharded_result_claiming"]["ready_field"] == (
        "manifest_shard_index/manifest_shard_count plus per-output CSV locks"
    )
    assert gates["sharded_result_claiming"]["status"] == (
        "supported_by_manifest_shard_flags_and_output_locks"
    )
    assert any("split leakage" in item for item in spec["stop_conditions"])
    assert any("work-claim locks" in item for item in spec["stop_conditions"])
    assert spec["outputs"]["joined_pairs"].endswith("joined_pairs.csv")
    assert spec["outputs"]["local_current_shard_glob"].endswith("local_current_results.shard_*.csv")
    assert spec["outputs"]["comparison_dirs"]["current"].endswith("smoke_compare_current")
    assert spec["outputs"]["comparison_dirs"]["tabpfn_candidate"].endswith("smoke_compare_tabpfn_candidate")
    assert spec["outputs"]["comparison_dirs"]["tabentics_diakrino"].endswith("smoke_compare_tabentics_diakrino")
    assert "local_tabentics_diakrino_rows.csv" in spec["commands"]["compare_tabentics_diakrino"]
    shard_plan = spec["local_current_shard_plan"]
    assert len(shard_plan) == 4
    assert [row["shard_index"] for row in shard_plan] == [0, 1, 2, 3]
    assert [row["shard_count"] for row in shard_plan] == [4, 4, 4, 4]
    assert [row["planned_manifest_rows"] for row in shard_plan] == [2, 2, 1, 1]
    assert [row["assigned_host"] for row in shard_plan] == [
        "host1.example.com",
        "host1.example.com",
        "host2.example.com",
        "host2.example.com",
    ]
    assert [row["pod_index_on_host"] for row in shard_plan] == [0, 1, 0, 1]
    assert shard_plan[0]["target_workers_per_pod"] == 14
    assert shard_plan[2]["target_workers_per_pod"] == 10
    assert shard_plan[0]["out_csv"].endswith("local_current_results.shard_0.csv")
    audit = spec["readiness_audit"]
    assert audit["acceptance_checks"]["run_spec_emitted"] is True
    assert audit["acceptance_checks"]["local_execution_plan_emitted"] is True
    assert audit["acceptance_checks"]["live_host_capacity_ready"] is False
    assert audit["acceptance_checks"]["gpu_revalidation_ready"] is False
    assert audit["local_execution_plan"]["unsupported_regression_rows"] > 0
    assert audit["local_execution_plan"]["deferred_gpu_revalidation_rows"] > 0
    assert "live lab CPU capacity checks are not recorded" in audit["blockers"]["lab_launch"]
    assert (
        "GPU-required DIAKRINO rows remain deferred pending public-gpu-host revalidation"
        in audit["blockers"]["lab_launch"]
    )


def test_run_spec_readiness_audit_summarizes_materialization_plan(tmp_path: Path) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)
    execution_plan = build_beyondarena_local_execution_plan(manifest)
    artifact_root = tmp_path / "artifacts"
    materialization_plan = build_beyondarena_materialization_plan(
        manifest,
        out_dir=artifact_root,
        include_dataset=True,
    )
    artifact_root.mkdir(parents=True)
    materialization_plan.to_csv(artifact_root / "materialization_plan.csv", index=False)

    spec = build_beyondarena_run_spec(
        manifest,
        execution_plan,
        subset="smoke",
        out_dir=tmp_path / "smoke_plan",
        artifact_root=artifact_root,
        public_cpu_host_1_max_workers=29,
        public_cpu_host_1_pods_per_host=2,
        public_cpu_host_2_max_workers=21,
        public_cpu_host_2_pods_per_host=2,
        cpu_capacity_revalidated=True,
    )

    audit = spec["readiness_audit"]
    materialization = audit["artifact_materialization"]
    assert materialization["configured"] is True
    assert materialization["artifact_count"] == len(manifest)
    assert materialization["artifact_plan_ready_count"] == len(manifest)
    assert materialization["artifact_pending_count"] == len(manifest)
    assert materialization["materialization_ready"] is False
    assert materialization["local_runner_ready"] is False
    assert audit["acceptance_checks"]["artifact_plan_ready"] is True
    assert audit["acceptance_checks"]["artifact_materialization_ready"] is False
    assert audit["acceptance_checks"]["artifact_local_runner_ready"] is False
    assert (
        "current tabnetics rows require materialized BeyondArena artifacts before local execution"
        in audit["blockers"]["lab_launch"]
    )


def test_run_spec_marks_cpu_capacity_ready_only_after_explicit_revalidation(
    tmp_path: Path,
) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)
    execution_plan = build_beyondarena_local_execution_plan(manifest)

    spec = build_beyondarena_run_spec(
        manifest,
        execution_plan,
        subset="smoke",
        out_dir=tmp_path / "smoke_plan",
        artifact_root=tmp_path / "artifacts",
        public_cpu_host_1_max_workers=29,
        public_cpu_host_1_pods_per_host=2,
        public_cpu_host_2_max_workers=21,
        public_cpu_host_2_pods_per_host=2,
        cpu_capacity_revalidated=True,
    )

    hosts = {row["host"]: row for row in spec["host_allocation"]}
    assert hosts["host1.example.com"]["status"] == "live_capacity_recorded"
    assert hosts["host2.example.com"]["status"] == "live_capacity_recorded"
    assert spec["cpu_capacity_revalidated"] is True
    assert "--cpu-capacity-revalidated" in spec["commands"]["plan"]
    gates = {row["gate"]: row for row in spec["prelaunch_gates"]}
    assert gates["live_cpu_capacity"]["status"] == "recorded"
    audit = spec["readiness_audit"]
    assert audit["acceptance_checks"]["live_host_capacity_ready"] is True
    assert "live lab CPU capacity checks are not recorded" not in audit["blockers"]["lab_launch"]
    assert (
        "GPU-required DIAKRINO rows remain deferred pending public-gpu-host revalidation"
        in audit["blockers"]["lab_launch"]
    )


def test_run_spec_keeps_tabentics_diakrino_deferred_after_gpu_revalidation_without_checkpoint(tmp_path: Path) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)
    execution_plan = build_beyondarena_local_execution_plan(manifest, arch_ml_revalidated=True)

    spec = build_beyondarena_run_spec(
        manifest,
        execution_plan,
        subset="smoke",
        out_dir=tmp_path / "smoke_plan",
        artifact_root=tmp_path / "artifacts",
        arch_ml_revalidated=True,
        public_cpu_host_1_max_workers=29,
        public_cpu_host_1_pods_per_host=2,
        public_cpu_host_2_max_workers=21,
        public_cpu_host_2_pods_per_host=2,
        arch_ml_max_workers=1,
        arch_ml_pods_per_host=1,
    )

    hosts = {row["host"]: row for row in spec["host_allocation"]}
    assert hosts["public-gpu-host"]["MAX_WORKERS"] == 1
    assert hosts["public-gpu-host"]["PODS_PER_HOST"] == 1
    assert hosts["public-gpu-host"]["status"] == "live_gpu_env_recorded"
    gates = {row["gate"]: row for row in spec["prelaunch_gates"]}
    assert gates["gpu_revalidation"]["status"] == "recorded"
    assert gates["tabentics_diakrino_checkpoint"]["status"] == "deferred_tabentics_diakrino_checkpoint"
    assert "--allow-gpu-execution" in spec["commands"]["local_tabpfn_candidate"]
    assert "--allow-gpu-execution" not in spec["commands"]["local_tabentics_diakrino"]
    assert spec["readiness_audit"]["acceptance_checks"]["gpu_revalidation_ready"] is True
    assert spec["readiness_audit"]["acceptance_checks"]["tabentics_diakrino_checkpoint_ready"] is False
    assert spec["readiness_audit"]["local_execution_plan"]["deferred_tabentics_diakrino_checkpoint_rows"] > 0


def test_run_spec_enables_native_diakrino_only_after_checkpoint_ready(tmp_path: Path) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)
    execution_plan = build_beyondarena_local_execution_plan(
        manifest,
        arch_ml_revalidated=True,
        tabentics_diakrino_checkpoint_ready=True,
    )

    spec = build_beyondarena_run_spec(
        manifest,
        execution_plan,
        subset="smoke",
        out_dir=tmp_path / "smoke_plan",
        artifact_root=tmp_path / "artifacts",
        arch_ml_revalidated=True,
        tabentics_diakrino_checkpoint_ready=True,
        public_cpu_host_1_max_workers=29,
        public_cpu_host_1_pods_per_host=2,
        public_cpu_host_2_max_workers=21,
        public_cpu_host_2_pods_per_host=2,
        arch_ml_max_workers=1,
        arch_ml_pods_per_host=1,
    )

    gates = {row["gate"]: row for row in spec["prelaunch_gates"]}
    assert gates["tabentics_diakrino_checkpoint"]["status"] == "recorded"
    assert "--tabnetics-diakrino-checkpoint-ready" in spec["commands"]["plan"]
    assert "--allow-gpu-execution" in spec["commands"]["local_tabentics_diakrino"]
    assert spec["tabentics_diakrino_checkpoint_ready"] is True
    assert spec["readiness_audit"]["acceptance_checks"]["tabentics_diakrino_checkpoint_ready"] is True
    assert spec["readiness_audit"]["local_execution_plan"]["deferred_tabentics_diakrino_checkpoint_rows"] == 0


def test_current_feasibility_run_spec_is_current_cpu_only(tmp_path: Path) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="current-feasibility",
    )
    manifest = task_rows_to_manifest_frame(rows)
    execution_plan = build_beyondarena_local_execution_plan(
        manifest,
        profiles=[
            profile
            for profile in default_beyondarena_run_profiles()
            if profile.profile_id == "tabnetics_current"
        ],
    )

    spec = build_beyondarena_run_spec(
        manifest,
        execution_plan,
        subset="current-feasibility",
        out_dir=tmp_path / "current_feasibility_plan",
        artifact_root=tmp_path / "artifacts",
        public_cpu_host_1_max_workers=29,
        public_cpu_host_1_pods_per_host=2,
        public_cpu_host_2_max_workers=21,
        public_cpu_host_2_pods_per_host=2,
    )

    assert spec["command_sequence"] == [
        "plan",
        "materialize_dry_run",
        "materialize_dataset",
        "local_current",
        "compare_current",
    ]
    assert "local_tabpfn_candidate" not in spec["commands"]
    assert "local_tabentics_diakrino" not in spec["commands"]
    assert "local_tabpfn_candidate_rows" not in spec["outputs"]
    assert "local_tabentics_diakrino_rows" not in spec["outputs"]
    assert [row["host"] for row in spec["host_allocation"]] == ["host1.example.com", "host2.example.com"]
    assert spec["readiness_audit"]["local_execution_plan"]["deferred_gpu_revalidation_rows"] == 0
    assert spec["readiness_audit"]["acceptance_checks"]["gpu_revalidation_ready"] is True


def test_model_manifest_preserves_isolated_tabiclv2_local_backend() -> None:
    profiles = default_beyondarena_run_profiles()
    tabiclv2 = next(profile for profile in profiles if profile.method == "TabICLv2")

    assert tabiclv2.backend == ""
    assert tabiclv2.local_backend == "tabiclv2-candidate"
    assert tabiclv2.status == "optional"


def test_plan_writer_emits_joined_comparison_when_exact_rows_exist(tmp_path: Path) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)
    official_csv = tmp_path / "official.csv"
    local_csv = tmp_path / "local.csv"
    official_records = []
    local_records = []
    for pos, (_, row) in enumerate(manifest.iterrows()):
        official_score = 0.80
        if bool(row["lower_is_better"]):
            local_score = 0.75 if pos % 2 == 0 else 0.85
        else:
            local_score = 0.85 if pos % 2 == 0 else 0.75
        official_records.append(
            {
                "dataset_id": row["dataset_id"],
                "fold": row["official_split_id"],
                "method": "TabPFN-2.6",
                "metric": row["metric"],
                "metric_error": official_score,
                "task_type": row["task_type"],
                "size_tier": row["size_tier"],
            }
        )
        local_records.append(
            {
                "dataset_id": row["dataset_id"],
                "split_id": row["split_id"],
                "method": "TabenticsDiakrino",
                "metric": row["metric"],
                "metric_error": local_score,
                "model_profile": "tabentics_diakrino_experimental",
                "execution_status": "ok",
                "execution_backend": "tabnetics-diakrino",
            }
        )
    pd.DataFrame(official_records).to_csv(official_csv, index=False)
    pd.DataFrame(local_records).to_csv(local_csv, index=False)

    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path / "plan",
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
        official_results=official_csv,
        local_results=local_csv,
    )

    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    joined = pd.read_csv(artifacts.joined)
    summary = pd.read_csv(artifacts.summary)

    assert status["comparison_ready"] is True
    assert status["joined_rows"] == 2 * len(manifest)
    assert status["local_result_rows"] == len(manifest)
    assert status["local_manifest_aligned_rows"] == len(manifest)
    assert status["local_manifest_key_rows"] == len(manifest)
    assert status["local_ok_rows"] == len(manifest)
    assert status["local_status_counts"] == {"ok": len(manifest)}
    assert status["local_execution_status_counts"] == {"ok": len(manifest)}
    assert status["local_execution_backend_counts"] == {"tabnetics-diakrino": len(manifest)}
    assert status["local_schema_only_ok_rows"] == 0
    assert status["local_claim_eligible_ok_rows"] == len(manifest)
    assert status["local_claim_eligible_ok_key_rows"] == len(manifest)
    assert status["metric_contract_valid"] is True
    assert status["comparison_value_semantics_counts"] == {"error": 2 * len(manifest)}
    assert status["readiness_audit"]["overall_status"] == "blocked"
    assert status["readiness_audit"]["acceptance_checks"]["paired_comparison_ready"] is True
    assert status["readiness_audit"]["acceptance_checks"]["metric_contract_valid"] is True
    assert status["readiness_audit"]["claims"]["performance_claims_ready"] is False
    assert status["readiness_audit"]["local_results"]["manifest_key_rows"] == len(manifest)
    assert status["readiness_audit"]["local_results"]["claim_eligible_ok_key_rows"] == len(manifest)
    assert any(
        "smoke subset is a Stage-1 correctness/feasibility check only" in blocker
        for blocker in status["readiness_audit"]["blockers"]["performance_claims"]
    )
    assert len(joined) == 2 * len(manifest)
    assert OFFICIAL_BEST_TFM_METHOD in set(joined["method_official"])
    assert {"win", "loss"}.issubset(set(joined["outcome"]))
    assert not summary.empty


def test_public_r2_load_survives_plan_writer_and_joins_through_metric_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)

    method_error_offsets = {
        "TA-TabDPT": 0.03,
        "TA-TabPFN-2.6": 0.02,
        "TA-TabICLv2": 0.01,
    }

    def fake_read_public_r2(method_path: str) -> pd.DataFrame:
        offset = method_error_offsets.get(method_path, 0.04)
        records = []
        for _, row in manifest.iterrows():
            base_error = 0.18 if row["metric"] == "roc_auc" else 0.38
            records.append(
                {
                    "dataset": row["dataset_id"],
                    "fold": row["official_split_id"],
                    "metric": row["metric"],
                    "metric_error": base_error + offset,
                    "problem_type": row["problem_type"],
                }
            )
        return pd.DataFrame.from_records(records)

    monkeypatch.setattr(
        beyondarena_compare,
        "_read_public_r2_parquet",
        fake_read_public_r2,
    )

    local_csv = tmp_path / "local.csv"
    local_records = []
    for _, row in manifest.iterrows():
        higher_is_better = not bool(row["lower_is_better"])
        local_records.append(
            {
                "dataset_id": row["dataset_id"],
                "split_id": row["official_split_id"],
                "method": "tabnetics-current",
                "metric": row["metric"],
                "metric_value": 0.82 if higher_is_better else 0.35,
                "metric_error": 0.18 if higher_is_better else 0.35,
                "model_profile": "tabnetics_current",
                "execution_status": "ok",
                "execution_backend": "tabnetics-current",
                "status": "ok",
            }
        )
    pd.DataFrame.from_records(local_records).to_csv(local_csv, index=False)

    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path / "prior" / "metric_contract_v2",
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
        official_results="public-r2",
        local_results=local_csv,
        emit_local_execution_plan=True,
        emit_run_spec=True,
        public_cpu_host_1_max_workers=1,
        public_cpu_host_1_pods_per_host=1,
        public_cpu_host_2_max_workers=1,
        public_cpu_host_2_pods_per_host=1,
    )

    official = pd.read_csv(artifacts.official_normalized)
    joined = pd.read_csv(artifacts.joined)
    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    run_spec = json.loads(artifacts.run_spec.read_text(encoding="utf-8"))

    assert official["lower_is_better"].eq(True).all()
    assert official["metric_value_semantics"].eq("error").all()
    assert joined["comparison_value_semantics"].eq("error").all()
    assert joined["comparison_lower_is_better"].eq(True).all()
    assert status["metric_contract_valid"] is True
    assert status["comparison_value_semantics_counts"] == {"error": len(joined)}
    assert status["readiness_audit"]["acceptance_checks"]["paired_comparison_ready"] is True
    assert status["readiness_audit"]["claims"]["performance_claims_ready"] is False
    assert run_spec["inputs"]["local_results"] == str(local_csv)
    assert run_spec["outputs"]["local_current_results"] == str(local_csv)
    assert run_spec["outputs"]["comparison_dir"] == str(artifacts.out_dir)
    assert f"--local-results {local_csv}" in run_spec["commands"]["plan"]
    assert f"--out-dir {artifacts.out_dir}" in run_spec["commands"]["compare_current"]

    roc_dataset = manifest.loc[manifest["metric"].eq("roc_auc"), "dataset_id"].iloc[0]
    diakrino_pair = joined[
        joined["dataset_id"].eq(roc_dataset)
        & joined["method_official"].eq("TabPFN-2.6")
    ].iloc[0]
    assert diakrino_pair["comparison_value_official"] == pytest.approx(0.20)
    assert diakrino_pair["comparison_value_local"] == pytest.approx(0.18)
    assert diakrino_pair["comparison_delta"] == pytest.approx(0.02)
    best_tfm = joined[
        joined["dataset_id"].eq(roc_dataset)
        & joined["method_official"].eq(OFFICIAL_BEST_TFM_METHOD)
    ].iloc[0]
    assert best_tfm["best_tfm_source_method"] == "TabICLv2"


def test_plan_writer_blocks_claim_ready_for_partial_local_coverage(tmp_path: Path) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)
    first = manifest.iloc[0]
    official_csv = tmp_path / "official.csv"
    local_csv = tmp_path / "local.csv"
    local_score = 0.75 if bool(first["lower_is_better"]) else 0.85
    pd.DataFrame(
        [
            {
                "dataset_id": first["dataset_id"],
                "fold": first["official_split_id"],
                "method": "TabPFN-2.6",
                "metric": first["metric"],
                "metric_error": 0.80,
                "task_type": first["task_type"],
                "size_tier": first["size_tier"],
            }
        ]
    ).to_csv(official_csv, index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": first["dataset_id"],
                "split_id": first["split_id"],
                "method": "TabenticsDiakrino",
                "metric": first["metric"],
                "metric_error": local_score,
                "model_profile": "tabentics_diakrino_experimental",
                "execution_status": "ok",
                "execution_backend": "tabnetics-diakrino",
            }
            for _ in range(len(manifest))
        ]
    ).to_csv(local_csv, index=False)

    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path / "plan",
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
        official_results=official_csv,
        local_results=local_csv,
    )

    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    audit = status["readiness_audit"]

    assert status["comparison_ready"] is True
    assert status["joined_rows"] == 2 * len(manifest)
    assert status["local_manifest_aligned_rows"] == len(manifest)
    assert status["local_manifest_key_rows"] == 1
    assert status["local_ok_rows"] == len(manifest)
    assert status["local_claim_eligible_ok_key_rows"] == 1
    assert audit["overall_status"] == "blocked"
    assert audit["claims"]["performance_claims_ready"] is False
    assert audit["local_results"]["manifest_aligned_rows"] == len(manifest)
    assert audit["local_results"]["manifest_key_rows"] == 1
    assert audit["local_results"]["claim_eligible_ok_key_rows"] == 1
    assert (
        f"local result rows cover only 1/{len(manifest)} manifest rows"
        in audit["blockers"]["performance_claims"]
    )


def test_plan_writer_blocks_claim_ready_for_error_rows_even_with_full_local_coverage(
    tmp_path: Path,
) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)
    official_csv = tmp_path / "official.csv"
    local_csv = tmp_path / "local.csv"
    official_records = []
    local_records = []
    for pos, (_, row) in enumerate(manifest.iterrows()):
        official_records.append(
            {
                "dataset_id": row["dataset_id"],
                "fold": row["official_split_id"],
                "method": "TabPFN-2.6",
                "metric": row["metric"],
                "metric_error": 0.80,
            }
        )
        local_status = "error" if pos == len(manifest) - 1 else "ok"
        local_records.append(
            {
                "dataset_id": row["dataset_id"],
                "split_id": row["split_id"],
                "method": "TabenticsDiakrino",
                "metric": row["metric"],
                "metric_error": pd.NA if local_status == "error" else 0.75,
                "status": local_status,
                "model_profile": "tabentics_diakrino_experimental",
                "execution_status": "metric_unavailable" if local_status == "error" else "ok",
                "execution_backend": "tabnetics-diakrino",
                "skip_reason": "metric unavailable" if local_status == "error" else "",
            }
        )
    pd.DataFrame(official_records).to_csv(official_csv, index=False)
    pd.DataFrame(local_records).to_csv(local_csv, index=False)

    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path / "plan",
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
        official_results=official_csv,
        local_results=local_csv,
    )

    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    audit = status["readiness_audit"]

    assert status["comparison_ready"] is True
    assert status["joined_rows"] == 2 * (len(manifest) - 1)
    assert status["local_manifest_key_rows"] == len(manifest)
    assert status["local_ok_rows"] == len(manifest) - 1
    assert status["local_claim_eligible_ok_rows"] == len(manifest) - 1
    assert status["local_claim_eligible_ok_key_rows"] == len(manifest) - 1
    assert status["local_status_counts"] == {"ok": len(manifest) - 1, "error": 1}
    assert audit["overall_status"] == "blocked"
    assert audit["claims"]["performance_claims_ready"] is False
    assert audit["local_results"]["manifest_key_rows"] == len(manifest)
    assert audit["local_results"]["claim_eligible_ok_key_rows"] == len(manifest) - 1
    assert (
        f"claim-eligible ok local result rows cover only {len(manifest) - 1}/{len(manifest)} manifest rows"
        in audit["blockers"]["performance_claims"]
    )


def test_plan_writer_blocks_claim_ready_for_schema_only_smoke_rows(tmp_path: Path) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)
    official_csv = tmp_path / "official.csv"
    local_csv = tmp_path / "local.csv"
    official_records = []
    local_records = []
    for _, row in manifest.iterrows():
        official_records.append(
            {
                "dataset_id": row["dataset_id"],
                "fold": row["official_split_id"],
                "method": "TabPFN-2.6",
                "metric": row["metric"],
                "metric_error": 0.80,
            }
        )
        local_records.append(
            {
                "dataset_id": row["dataset_id"],
                "split_id": row["split_id"],
                "method": "tabnetics-current-smoke",
                "metric": row["metric"],
                "metric_error": 0.75,
                "model_profile": "sklearn_smoke",
                "execution_status": "ok",
                "execution_backend": "sklearn-smoke",
            }
        )
    pd.DataFrame(official_records).to_csv(official_csv, index=False)
    pd.DataFrame(local_records).to_csv(local_csv, index=False)

    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path / "plan",
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
        official_results=official_csv,
        local_results=local_csv,
    )

    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    joined = pd.read_csv(artifacts.joined)
    audit = status["readiness_audit"]

    assert status["comparison_ready"] is True
    assert status["joined_rows"] == 2 * len(manifest)
    assert status["local_manifest_key_rows"] == len(manifest)
    assert status["local_ok_rows"] == len(manifest)
    assert status["local_schema_only_ok_rows"] == len(manifest)
    assert status["local_claim_eligible_ok_rows"] == 0
    assert status["local_claim_eligible_ok_key_rows"] == 0
    assert status["local_execution_backend_counts"] == {"sklearn-smoke": len(manifest)}
    assert audit["overall_status"] == "blocked"
    assert audit["claims"]["performance_claims_ready"] is False
    assert audit["local_results"]["schema_only_ok_rows"] == len(manifest)
    assert audit["local_results"]["claim_eligible_ok_rows"] == 0
    assert audit["local_results"]["claim_eligible_ok_key_rows"] == 0
    assert audit["local_results"]["manifest_key_rows"] == len(manifest)
    assert any("schema-only smoke/accounting rows" in item for item in audit["blockers"]["performance_claims"])
    assert len(joined) == 2 * len(manifest)
    assert OFFICIAL_BEST_TFM_METHOD in set(joined["method_official"])


def test_plan_writer_current_feasibility_is_runner_proof_not_claim_gate(tmp_path: Path) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="current-feasibility",
    )
    manifest = task_rows_to_manifest_frame(rows)
    first = manifest.iloc[0]
    official_csv = tmp_path / "official.csv"
    local_csv = tmp_path / "local.csv"
    pd.DataFrame(
        [
            {
                "dataset_id": first["dataset_id"],
                "fold": first["official_split_id"],
                "method": "LightGBM",
                "metric": first["metric"],
                "metric_error": 0.80,
            }
        ]
    ).to_csv(official_csv, index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": first["dataset_id"],
                "split_id": first["split_id"],
                "method": "tabnetics-current",
                "metric": first["metric"],
                "metric_error": 0.75,
                "model_profile": "current_default",
                "execution_status": "ok",
                "execution_backend": "tabnetics-current",
            }
        ]
    ).to_csv(local_csv, index=False)

    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path / "plan",
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="current-feasibility",
        official_results=official_csv,
        local_results=local_csv,
    )

    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    joined = pd.read_csv(artifacts.joined)
    audit = status["readiness_audit"]

    assert status["subset"] == "current-feasibility"
    assert status["comparison_ready"] is True
    assert status["joined_rows"] == 1
    assert status["local_claim_eligible_ok_rows"] == 1
    assert not joined.empty
    assert audit["overall_status"] == "blocked"
    assert audit["claims"]["performance_claims_ready"] is False
    assert any("runner/provenance evidence only" in item for item in audit["blockers"]["performance_claims"])


def test_plan_writer_blocks_comparison_ready_when_no_ok_rows_join(tmp_path: Path) -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)
    official_csv = tmp_path / "official.csv"
    local_csv = tmp_path / "local.csv"
    official_records = []
    local_records = []
    for _, row in manifest.iterrows():
        official_records.append(
            {
                "dataset_id": row["dataset_id"],
                "fold": row["official_split_id"],
                "method": "TabPFN-2.6",
                "metric": row["metric"],
                "metric_error": 0.80,
            }
        )
        local_records.append(
            {
                "dataset_id": row["dataset_id"],
                "split_id": row["split_id"],
                "method": "TabenticsDiakrino",
                "metric": row["metric"],
                "metric_error": pd.NA,
                "status": "skipped",
                "execution_status": "skipped_missing_artifact",
                "execution_backend": "tabnetics-diakrino",
                "skip_reason": "no materialized local DataFoundry artifact matched task manifest row",
            }
        )
    pd.DataFrame(official_records).to_csv(official_csv, index=False)
    pd.DataFrame(local_records).to_csv(local_csv, index=False)

    artifacts = write_beyondarena_plan_artifacts(
        out_dir=tmp_path / "plan",
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
        official_results=official_csv,
        local_results=local_csv,
    )

    status = json.loads(artifacts.comparison_status.read_text(encoding="utf-8"))
    joined = pd.read_csv(artifacts.joined)

    assert status["exact_paired_rows_available"] is True
    assert status["comparison_ready"] is False
    assert status["joined_rows"] == 0
    assert status["local_result_rows"] == len(manifest)
    assert status["local_manifest_aligned_rows"] == len(manifest)
    assert status["local_manifest_key_rows"] == len(manifest)
    assert status["local_ok_rows"] == 0
    assert status["local_claim_eligible_ok_key_rows"] == 0
    assert status["local_status_counts"] == {"skipped": len(manifest)}
    assert status["local_execution_status_counts"] == {"skipped_missing_artifact": len(manifest)}
    assert status["local_execution_backend_counts"] == {"tabnetics-diakrino": len(manifest)}
    assert status["local_method_counts"] == {"TabenticsDiakrino": len(manifest)}
    assert "No exact ok local rows joined" in status["comparison_blocker"]
    audit = status["readiness_audit"]
    assert audit["overall_status"] == "blocked"
    assert audit["local_results"]["ok_rows"] == 0
    assert audit["local_results"]["manifest_key_rows"] == len(manifest)
    assert audit["local_results"]["claim_eligible_ok_key_rows"] == 0
    assert "local result rows contain no manifest-aligned ok rows" in audit["blockers"]["performance_claims"]
    assert joined.empty


def test_align_results_to_manifest_accepts_official_and_datafoundry_split_aliases() -> None:
    rows = resolve_beyondarena_task_rows(
        task_metadata_source=TASKS,
        core_tasks_source=CORE,
        subset="smoke",
    )
    manifest = task_rows_to_manifest_frame(rows)
    first = manifest.iloc[0]
    raw = pd.DataFrame(
        [
            {
                "dataset_id": first["dataset_id"],
                "split_id": first["split_id"],
                "method": "TabenticsDiakrino",
                "metric": first["metric"],
                "metric_value": 1.2,
                "origin": "tabnetics",
                "lower_is_better": True,
                "status": "ok",
            },
            {
                "dataset_id": first["dataset_id"],
                "split_id": first["official_split_id"],
                "method": "tabnetics-current",
                "metric": first["metric"],
                "metric_value": 1.1,
                "origin": "tabnetics",
                "lower_is_better": True,
                "status": "ok",
            },
            {
                "dataset_id": "not-in-manifest",
                "split_id": "0",
                "method": "tabnetics-current",
                "metric": first["metric"],
                "metric_value": 1.1,
                "origin": "tabnetics",
                "lower_is_better": True,
                "status": "ok",
            },
        ]
    )

    aligned = align_beyondarena_results_to_manifest(raw, manifest)

    assert len(aligned) == 2
    assert set(aligned["split_id"]) == {str(first["official_split_id"])}
    assert set(aligned["source_split_id"]) == {str(first["split_id"]), str(first["official_split_id"])}
    assert set(aligned["task_type"]) == {first["task_type"]}


def test_beyondarena_plan_cli_writes_artifacts(tmp_path: Path) -> None:
    rc = main(
        [
            "--out-dir",
            str(tmp_path),
            "--subset",
            "smoke",
            "--task-metadata-csv",
            str(TASKS),
            "--core-tasks-csv",
            str(CORE),
            "--emit-pending-local-rows",
            "--emit-local-execution-plan",
            "--emit-run-spec",
            "--cpu-capacity-revalidated",
            "--public_cpu_host_1-max-workers",
            "29",
            "--public_cpu_host_1-pods-per-host",
            "2",
            "--public_cpu_host_2-max-workers",
            "21",
            "--public_cpu_host_2-pods-per-host",
            "2",
        ]
    )

    assert rc == 0
    assert (tmp_path / "beyondarena_task_manifest.csv").exists()
    assert (tmp_path / "beyondarena_model_manifest.csv").exists()
    assert (tmp_path / "comparison_status.json").exists()
    assert (tmp_path / "readiness_audit.json").exists()
    pending = pd.read_csv(tmp_path / "pending_local_rows.csv")
    execution_plan = pd.read_csv(tmp_path / "local_execution_plan.csv")
    status = json.loads((tmp_path / "comparison_status.json").read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "readiness_audit.json").read_text(encoding="utf-8"))
    run_spec = json.loads((tmp_path / "run_spec.json").read_text(encoding="utf-8"))
    assert len(pending) == 12
    assert len(execution_plan) == 12
    assert {"execution_host", "execution_lane", "local_split_id"}.issubset(set(pending.columns))
    assert "skipped_unsupported_regression" in set(execution_plan["execution_status"])
    assert run_spec["host_allocation"][0]["MAX_WORKERS"] == 29
    assert run_spec["host_allocation"][0]["status"] == "live_capacity_recorded"
    assert run_spec["host_allocation"][1]["PODS_PER_HOST"] == 2
    assert run_spec["host_allocation"][1]["status"] == "live_capacity_recorded"
    assert run_spec["host_allocation"][2]["status"] == "deferred_gpu_revalidation"
    assert run_spec["cpu_capacity_revalidated"] is True
    assert run_spec["manifest_coverage"]["smoke_coverage_ready"] is True
    assert run_spec["manifest_coverage"]["smoke_missing_facets"] == []
    assert run_spec["manifest_coverage"]["split_stability"]["per_dataset_split_stability_ready"] is False
    assert "prelaunch_gates" in run_spec
    assert status["readiness_audit"] == audit
    assert run_spec["readiness_audit"] == audit
    assert audit["acceptance_checks"]["live_host_capacity_ready"] is True
    assert "--official-results public-r2" in run_spec["commands"]["plan"]
    assert "--cpu-capacity-revalidated" in run_spec["commands"]["plan"]
    assert "--include-dataset --fetch-size-metadata" in run_spec["commands"]["materialize_dry_run"]
    assert audit["local_execution_plan"]["rows"] == 12
    assert audit["run_spec"]["emitted"] is True
    assert "--require-local-runner-ready" in run_spec["commands"]["materialize_dataset"]
    assert "compare_tabentics_diakrino" in run_spec["commands"]
    assert run_spec["outputs"]["comparison_joined_pairs"]["tabentics_diakrino"].endswith("joined_pairs.csv")

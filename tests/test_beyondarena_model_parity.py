from __future__ import annotations

from tabnetics.benchmarks.profiles import (
    BEYONDARENA_MODEL_PARITY,
    BEYONDARENA_PARITY_BACKENDS,
    beyondarena_parity_inventory,
)
from tabnetics.benchmarks.runner import MODEL_CANDIDATE_PROFILES
from tabnetics.classification.backends import CLASSIFIER_COMPLEXITY_PRIOR


def test_beyondarena_parity_inventory_covers_paper_methods() -> None:
    inventory = {row["normalized_name"]: row for row in beyondarena_parity_inventory()}

    assert len(BEYONDARENA_MODEL_PARITY) == 11
    assert set(inventory) == {
        "Linear/Logistic Regression",
        "Random Forest",
        "ExtraTrees",
        "CatBoost",
        "LightGBM",
        "XGBoost",
        "RealMLP",
        "TabM",
        "TabDPT",
        "TabPFN-2.6",
        "TabICLv2",
    }
    assert inventory["TabDPT"]["availability"] == "unavailable"
    assert inventory["TabICLv2"]["availability"] == "optional"
    assert inventory["TabDPT"]["skip_reason"]
    assert inventory["TabICLv2"]["skip_reason"]
    assert inventory["TabDPT"]["compatibility_scope"] == "result-schema skip stub only"
    assert inventory["TabICLv2"]["tabnetics_backend"] == ""
    assert inventory["TabICLv2"]["local_backend"] == "tabiclv2-candidate"
    assert "skipped_tabiclv2_outside_published_regime" in inventory["TabICLv2"]["fallback_status"]


def test_tabiclv2_is_local_only_and_does_not_change_stage2_backends() -> None:
    inventory = {row["normalized_name"]: row for row in beyondarena_parity_inventory()}

    assert inventory["TabICLv2"]["local_backend"] == "tabiclv2-candidate"
    assert "tabiclv2-candidate" not in BEYONDARENA_PARITY_BACKENDS
    assert "tabiclv2-candidate" not in MODEL_CANDIDATE_PROFILES["beyondarena_parity"]


def test_tabpfn_parity_inventory_records_audit_boundaries_and_fallbacks() -> None:
    inventory = {row["normalized_name"]: row for row in beyondarena_parity_inventory()}
    tabpfn = inventory["TabPFN-2.6"]

    assert tabpfn["tabnetics_backend"] == "tabpfn"
    assert tabpfn["availability"] == "optional"
    assert tabpfn["dependency"] == "tabpfn"
    assert tabpfn["sample_limit"]
    assert tabpfn["feature_limit"]
    assert "public-r2" in tabpfn["tuning_mode"]
    assert "installed optional package" in tabpfn["compatibility_scope"]
    assert "not treated as native Tabnetics Diakrino" in tabpfn["compatibility_scope"]
    assert "--allow-gpu-execution" in tabpfn["execution_guard"]
    assert "deferred_gpu_revalidation" in tabpfn["fallback_status"]
    assert "skipped_gpu_unavailable" in tabpfn["fallback_status"]
    assert "skipped_optional_dependency_unavailable" in tabpfn["fallback_status"]


def test_pytabkit_parity_entries_record_official_wrapper_scope() -> None:
    inventory = {row["normalized_name"]: row for row in beyondarena_parity_inventory()}

    for method in ("RealMLP", "TabM"):
        row = inventory[method]
        assert row["dependency"] == "pytabkit"
        assert "official pytabkit wrapper" in row["compatibility_scope"]
        assert row["fallback_status"] == "skipped_optional_dependency_unavailable"


def test_beyondarena_executable_backends_are_registered() -> None:
    assert MODEL_CANDIDATE_PROFILES["beyondarena_parity"] == BEYONDARENA_PARITY_BACKENDS
    for backend in BEYONDARENA_PARITY_BACKENDS:
        assert backend in CLASSIFIER_COMPLEXITY_PRIOR

    assert "realmlp_td" in BEYONDARENA_PARITY_BACKENDS
    assert "tabm_official" in BEYONDARENA_PARITY_BACKENDS
    assert "tabpfn" in BEYONDARENA_PARITY_BACKENDS


def test_beyondarena_profile_wires_benchmark_base_config() -> None:
    from tabnetics.benchmarks import runner as benchmark

    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--model-candidate-profile",
            "beyondarena_parity",
        ]
    )
    spec = benchmark.BENCHMARK_DATASETS["synthetic_easy_dfshift"]
    cfg = benchmark._build_base_config(args, spec, seed=11)

    assert cfg.model_candidates == BEYONDARENA_PARITY_BACKENDS
    assert cfg.include_tabpfn_model is True
    assert cfg.include_rf_model is True
    assert cfg.include_xgb_model is True


def test_beyondarena_public_package_exports_are_discoverable() -> None:
    import tabnetics.benchmarks as benchmarks
    import tabnetics.datasets as datasets

    benchmark_exports = set(benchmarks.__all__)
    dataset_exports = set(datasets.__all__)

    assert "load_public_beyondarena_r2_results" in benchmark_exports
    assert "BeyondArenaLocalRunConfig" in benchmark_exports
    assert "BeyondArenaMaterializeConfig" in benchmark_exports
    assert "write_beyondarena_plan_artifacts" in benchmark_exports
    assert "main" not in benchmark_exports

    assert "BeyondArenaDatasetSpec" in dataset_exports
    assert "load_beyondarena_task_metadata_csv" in dataset_exports
    assert "select_beyondarena_core_dataset_split_rows" in dataset_exports
    assert "discover_hf_beyondarena_specs" in dataset_exports

from __future__ import annotations

import copy
import ast
from collections.abc import Mapping
import builtins
import importlib
from importlib.machinery import ModuleSpec, SourceFileLoader
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from tabnetics.benchmarks import runner as benchmark
import tabnetics.pipeline.pipeline as pipeline_module
from tabnetics.validation.core.aggregate import _execution_value_matches, aggregate
from tabnetics.validation.core import provenance as provenance_module
from tabnetics.validation.core.provenance import (
    CanonicalExecutionExternalDependencyError,
    CanonicalExecutionInputIdentityError,
    CanonicalExecutionOriginError,
    EXTERNAL_CALLABLE_UNATTESTED_EVIDENCE_STATUS,
    EXECUTION_PROVENANCE_SCHEMA_VERSION,
    build_canonical_benchmark_artifact_provenance,
    build_canonical_execution_provenance,
    build_materialized_dataset_input_identity,
    capture_loaded_tabnetics_module_closure,
    canonical_json_sha256,
    canonical_execution_eligibility,
    execution_fingerprint_sha256,
    execution_row_fields,
    external_callable_identity_unattested_reason,
)


def _row(dataset_id: str, seed: int) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_id,
        "tier": "easy",
        "domain": "synthetic",
        "platform": "synthetic",
        "seed": seed,
        "config": "baseline",
        "protocol": "holdout",
        "accuracy": 0.9,
        "balanced_accuracy": 0.9,
        "macro_f1": 0.9,
        "hybrid_score": 0.9,
        "selected_features": 10,
        "fs_time_sec": 0.1,
        "dist_time_sec": 0.1,
        "transform_time_sec": 0.1,
        "n_dist_features_transformed": 10,
        "n_dist_rejected": 0,
        "n_dist_skipped_unreliable": 0,
        "n_dist_skipped_block_cv": 0,
        "n_low_gof_downweighted": 0,
        "cdf_block_gating_time_sec": 0.0,
        "cdf_block_gating_budget_hit": 0,
        "cdf_block_gating_blocks_evaluated": 0,
        "cdf_block_gating_blocks_applied": 0,
    }


def _canonical_contract() -> dict[str, object]:
    args = benchmark.build_arg_parser().parse_args(
        ["--datasets", "synthetic_easy_dfshift", "--seeds", "11"]
    )
    return benchmark._build_canonical_execution_contract(
        args,
        ["synthetic_easy_dfshift"],
        [_materialized_input_identity()],
    )


def _materialized_input_identity() -> dict[str, object]:
    return build_materialized_dataset_input_identity(
        dataset_id="synthetic_easy_dfshift",
        seed=11,
        data_source="synthetic:fixture",
        source_identity={
            "benchmark_source_kind": "synthetic",
            "source_policy": "fixture",
        },
        X=np.asarray([[0.0, 1.0], [2.0, np.nan]], dtype=float),
        y=np.asarray([0, 1], dtype=int),
        split_fingerprints=["fixture-split"],
    )


def _clean_reference_runtime_request() -> tuple[dict[str, object], dict[str, object]]:
    """Build a real clean-child request for the small core runtime module."""

    package_identity = provenance_module._verified_tabnetics_package_identity()
    runtime_module = importlib.import_module("tabnetics.core.runtime")
    prepared = provenance_module._prepare_loaded_package_module(
        "tabnetics.core.runtime",
        runtime_module,
        package_identity=package_identity,
    )
    return provenance_module._clean_reference_request(
        (prepared,),
        package_root=Path(str(package_identity["package_root"])),
    )


def _parallel_worker_dynamic_filter_task(
    dataset_id: str,
    seed: int,
    _args: object,
    progress_queue: object | None = None,
) -> dict[str, object]:
    """Small joblib-serializable task used to attest worker-only imports."""

    importlib.import_module("tabnetics.feature_selection.methods.filter")
    materialized = build_materialized_dataset_input_identity(
        dataset_id=dataset_id,
        seed=int(seed),
        data_source="synthetic:parallel-worker-fixture",
        source_identity={
            "benchmark_source_kind": "synthetic",
            "source_policy": "fixture",
        },
        X=np.asarray([[0.0, 1.0], [2.0, np.nan]], dtype=float),
        y=np.asarray([0, 1], dtype=int),
        split_fingerprints=[f"parallel-worker-split-{int(seed)}"],
    )
    return {
        "rows": [_row(dataset_id, int(seed))],
        "failures": [],
        "model_bundles": [],
        "run_diagnostics": [],
        "materialized_input_identity": materialized,
        "execution_module_closure": capture_loaded_tabnetics_module_closure(),
        "nested_execution_module_closures": [],
        "execution_module_closure_complete": True,
    }


def _write_canonical_aggregate_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    root_out = tmp_path / "out"
    job_id = "smoke/canonical/ds01"
    run_dir = root_out / job_id / "20260711_000000_df_fs_sota_benchmark"
    run_dir.mkdir(parents=True)
    execution = _canonical_contract()
    row_fields = execution_row_fields(execution)
    (run_dir / "df_fs_metadata.json").write_text(
        json.dumps({"execution_provenance": execution}),
        encoding="utf-8",
    )
    (run_dir / "df_fs_execution_provenance.json").write_text(
        json.dumps(execution),
        encoding="utf-8",
    )

    run_row = _row("synthetic_easy_dfshift", 11)
    run_row.update(row_fields)
    run_row["materialized_input_identity_sha256"] = execution["input_data_identity"][
        "materialized_inputs"
    ][0]["materialized_input_sha256"]
    summary_row = {"dataset_id": "synthetic_easy_dfshift", **row_fields}
    pd.DataFrame([run_row]).to_csv(run_dir / "df_fs_runs.csv", index=False)
    pd.DataFrame([summary_row]).to_csv(run_dir / "df_fs_summary.csv", index=False)
    pd.DataFrame([summary_row]).to_csv(
        run_dir / "df_fs_sota_comparison.csv",
        index=False,
    )
    pd.DataFrame([summary_row]).to_csv(
        run_dir / "df_fs_ablation_deltas.csv",
        index=False,
    )
    artifact_provenance = build_canonical_benchmark_artifact_provenance(
        execution_provenance=execution,
        artifact_paths={
            "df_fs_runs.csv": run_dir / "df_fs_runs.csv",
            "df_fs_summary.csv": run_dir / "df_fs_summary.csv",
            "df_fs_sota_comparison.csv": run_dir / "df_fs_sota_comparison.csv",
            "df_fs_ablation_deltas.csv": run_dir / "df_fs_ablation_deltas.csv",
            "df_fs_metadata.json": run_dir / "df_fs_metadata.json",
            "df_fs_execution_provenance.json": run_dir
            / "df_fs_execution_provenance.json",
        },
    )
    (run_dir / "df_fs_artifact_provenance.json").write_text(
        json.dumps(artifact_provenance),
        encoding="utf-8",
    )

    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": job_id,
                        "kind": "run_df_fs_sota_benchmark",
                        "params": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return root_out, plan, run_dir, execution


def test_canonical_contract_writes_core_execution_identity() -> None:
    execution = _canonical_contract()
    row_fields = execution_row_fields(execution)

    assert execution["implementation_stack"] == "tabnetics_core"
    assert execution["evidence_status"] == "canonical"
    assert execution["canonical_scorecard_eligible"] is True
    assert execution["resolved_cli_config_sha256"]
    assert execution["input_data_identity_sha256"]
    assert (
        execution["input_data_identity"]["dataset_registry"]["registry_available"]
        is True
    )
    assert execution["input_data_identity"]["materialized_inputs"][0]["x"]["sha256"]
    assert execution["fingerprint_sha256"]
    assert (
        execution["import_origins"]["pipeline"]["module"]
        == "tabnetics.pipeline.pipeline"
    )
    assert execution["import_origins"]["feature_selector"]["module"] == (
        "tabnetics.feature_selection.base"
    )
    assert execution["source_revision"]["module_sha256"]["pipeline"]
    loaded_modules = execution["loaded_package_modules"]
    loaded_records = loaded_modules["modules"]
    loaded_names = [record["module"] for record in loaded_records]
    assert loaded_names == sorted(loaded_names)
    assert len(loaded_names) == len(set(loaded_names))
    assert "tabnetics.distribution.selector" in loaded_names
    assert "tabnetics.feature_selection.base" in loaded_names
    assert loaded_modules["modules_sha256"] == canonical_json_sha256(loaded_records)
    assert (
        execution["loaded_package_modules_sha256"] == loaded_modules["modules_sha256"]
    )
    assert (
        execution["source_revision"]["loaded_package_modules_sha256"]
        == loaded_modules["modules_sha256"]
    )
    assert (
        execution["loaded_package_symbols_sha256"] == loaded_modules["symbols_sha256"]
    )
    assert (
        execution["source_revision"]["loaded_package_symbols_sha256"]
        == loaded_modules["symbols_sha256"]
    )
    assert "/research/" not in execution["import_origins"]["pipeline"]["path"]
    assert canonical_execution_eligibility(execution) == (True, "")
    assert row_fields["implementation_stack"] == "tabnetics_core"
    assert row_fields["evidence_status"] == "canonical"
    assert (
        row_fields["loaded_package_modules_sha256"]
        == execution["loaded_package_modules_sha256"]
    )
    assert (
        row_fields["loaded_package_symbols_sha256"]
        == execution["loaded_package_symbols_sha256"]
    )


def _source_identical_function_with_defaults(
    original: types.FunctionType,
    *,
    positional_defaults: tuple[object, ...] | None = None,
    keyword_defaults: dict[str, object] | None = None,
) -> types.FunctionType:
    """Clone code/globals while changing only mutable callable default state."""

    replacement = types.FunctionType(
        original.__code__,
        original.__globals__,
        original.__name__,
        original.__defaults__ if positional_defaults is None else positional_defaults,
        original.__closure__,
    )
    replacement.__kwdefaults__ = (
        original.__kwdefaults__ if keyword_defaults is None else keyword_defaults
    )
    replacement.__module__ = original.__module__
    replacement.__qualname__ = original.__qualname__
    return replacement


def test_closure_rejects_source_identical_function_default_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tabnetics.feature_selection.methods.filter")
    original = module.relieff_selection
    assert type(original) is types.FunctionType
    replacement = _source_identical_function_with_defaults(
        original,
        positional_defaults=(999,),
    )
    monkeypatch.setattr(module, "relieff_selection", replacement)

    with pytest.raises(CanonicalExecutionOriginError, match="callable state/defaults"):
        capture_loaded_tabnetics_module_closure()


def test_closure_rejects_class_method_positional_and_keyword_default_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = importlib.import_module("tabnetics.distribution.selector")
    original_init = selector.UnifiedDistributionSelectorV6.__init__
    assert type(original_init) is types.FunctionType
    defaults = list(original_init.__defaults__ or ())
    assert defaults[9] == 250
    defaults[9] = 1
    monkeypatch.setattr(
        selector.UnifiedDistributionSelectorV6,
        "__init__",
        _source_identical_function_with_defaults(
            original_init,
            positional_defaults=tuple(defaults),
        ),
    )

    with pytest.raises(CanonicalExecutionOriginError, match="callable state/defaults"):
        capture_loaded_tabnetics_module_closure()

    monkeypatch.undo()
    original_fit = selector.UnifiedDistributionSelectorV6._fit_single_distribution
    assert type(original_fit) is types.FunctionType
    keyword_defaults = dict(original_fit.__kwdefaults__ or {})
    assert keyword_defaults["compute_gof"] is True
    keyword_defaults["compute_gof"] = False
    monkeypatch.setattr(
        selector.UnifiedDistributionSelectorV6,
        "_fit_single_distribution",
        _source_identical_function_with_defaults(
            original_fit,
            keyword_defaults=keyword_defaults,
        ),
    )
    with pytest.raises(CanonicalExecutionOriginError, match="callable state/defaults"):
        capture_loaded_tabnetics_module_closure()


def test_closure_rejects_generated_dataclass_factory_closure_mutation() -> None:
    constructor = pipeline_module.DFFSConfig.__dict__["__init__"]
    assert type(constructor) is types.FunctionType
    closure_by_name = dict(
        zip(constructor.__code__.co_freevars, constructor.__closure__ or ())
    )
    factory_cell = closure_by_name["__dataclass_dflt_classification__"]
    original_factory = factory_cell.cell_contents

    def altered_factory() -> object:
        return pipeline_module.ClassificationConfig(include_tabpfn_model=True)

    factory_cell.cell_contents = altered_factory
    try:
        with pytest.raises(
            CanonicalExecutionOriginError,
            match="DFFSConfig.__init__.*generated dataclass",
        ):
            capture_loaded_tabnetics_module_closure()
        assert pipeline_module.DFFSConfig().classification.include_tabpfn_model is True
    finally:
        factory_cell.cell_contents = original_factory
    assert pipeline_module.DFFSConfig().classification.include_tabpfn_model is False


def test_closure_rejects_mutated_literal_global_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = importlib.import_module("tabnetics.distribution.selector")
    original = set(selector._POSITIVE_ONLY_FAMILIES)
    selector._POSITIVE_ONLY_FAMILIES.clear()
    try:
        with pytest.raises(
            CanonicalExecutionOriginError, match="_POSITIVE_ONLY_FAMILIES"
        ):
            capture_loaded_tabnetics_module_closure()
    finally:
        selector._POSITIVE_ONLY_FAMILIES.update(original)

    monkeypatch.setattr(
        builtins,
        "_POSITIVE_ONLY_FAMILIES",
        object(),
        raising=False,
    )
    selector._POSITIVE_ONLY_FAMILIES.clear()
    try:
        with pytest.raises(
            CanonicalExecutionOriginError, match="_POSITIVE_ONLY_FAMILIES"
        ):
            capture_loaded_tabnetics_module_closure()
    finally:
        selector._POSITIVE_ONLY_FAMILIES.update(original)


def test_closure_rejects_algorithmic_state_replaced_by_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = importlib.import_module("tabnetics.distribution.selector")
    monkeypatch.setattr(
        selector,
        "_POSITIVE_ONLY_FAMILIES",
        logging.getLogger("tabnetics-attestation-bypass"),
    )

    with pytest.raises(CanonicalExecutionOriginError, match="_POSITIVE_ONLY_FAMILIES"):
        capture_loaded_tabnetics_module_closure()


def test_source_assignment_index_marks_only_real_container_mutation() -> None:
    records = provenance_module._source_direct_assignment_bindings(
        ast.parse(
            "\n".join(
                (
                    "STATIC = 1",
                    "STATE = {}",
                    "STATE['enabled'] = True",
                    "STATE.update({'other': False})",
                    "if True:",
                    "    GUARDED = (STATIC,)",
                )
            )
        )
    )
    by_name: dict[str, list[object]] = {}
    by_conditional: dict[str, list[bool]] = {}
    for record in records:
        by_name.setdefault(str(record["name"]), []).append(record["value"])
        by_conditional.setdefault(str(record["name"]), []).append(
            bool(record["conditional"])
        )
    assert len(by_name["STATIC"]) == 1
    assert isinstance(by_name["STATIC"][0], ast.Constant)
    assert by_name["STATIC"][0].value == 1
    assert by_name["STATE"].count(None) == 2
    assert by_conditional["STATIC"] == [False]
    assert by_conditional["GUARDED"] == [True]


def _load_module_scope_definition_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_text: str,
    *,
    suffix: str,
) -> tuple[types.ModuleType, Path, dict[str, object]]:
    """Load source once and resolve its source-only definition alternatives."""

    module_name = f"tabnetics._provenance_{suffix}"
    source = tmp_path / f"_provenance_{suffix}.py"
    source.write_text(textwrap.dedent(source_text), encoding="utf-8")
    loader = SourceFileLoader(module_name, str(source))
    spec = ModuleSpec(module_name, loader, origin=str(source))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    loader.exec_module(module)
    source_sha256 = provenance_module.sha256_file(source)
    raw_reference = provenance_module._source_symbol_reference(
        str(source),
        source_sha256,
        module_name,
        False,
    )
    reference = provenance_module._resolve_source_symbol_reference(
        raw_reference,
        module_name=module_name,
        module=module,
        source=source,
    )
    return module, source, reference


def test_internal_import_derived_state_is_clean_sealed_and_substitution_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _owner_source, _owner_reference = _load_module_scope_definition_fixture(
        tmp_path,
        monkeypatch,
        """
        CURRENT = "current"
        NONE = "none"
        """,
        suffix="internal_state_owner",
    )
    module, source, reference = _load_module_scope_definition_fixture(
        tmp_path,
        monkeypatch,
        f"""
        from {owner.__name__} import CURRENT, NONE

        MODES = (CURRENT, NONE)

        if True:
            def ConditionalExport():
                return MODES

        try:
            raise RuntimeError("fixture")
        except RuntimeError:
            def TryExport():
                return MODES
        """,
        suffix="internal_state_consumer",
    )
    state_plan = provenance_module._source_static_state_plan(
        reference,
        module_name=module.__name__,
    )
    spec = state_plan.internal_import_state_specs["MODES"]
    assert spec["provider"] == "clean_isolated_internal_import_expression_v1"
    assert [binding["local_name"] for binding in spec["import_bindings"]] == [
        "CURRENT",
        "NONE",
    ]
    prepared = provenance_module._PreparedLoadedModule(
        module_name=module.__name__,
        module=module,
        source=source,
        source_sha256=provenance_module.sha256_file(source),
        is_package=False,
        reference=reference,
        state_plan=state_plan,
    )
    states, _defaults, dependencies = provenance_module._source_state_request(prepared)
    assert states == ["MODES"]
    assert dependencies == []

    clean_reference = provenance_module._CleanIsolatedReference(
        module_payloads={
            module.__name__: {
                "states": {
                    "MODES": provenance_module._state_value_payload(module.MODES),
                }
            }
        },
        source_tree_sha256="0" * 64,
        environment={},
        isolated_paths={},
        python={},
        platform={},
        dependencies={},
    )
    records = provenance_module._attest_module_runtime_state(
        module_name=module.__name__,
        module=module,
        reference=reference,
        state_plan=state_plan,
        state_context=provenance_module._EMPTY_STATE_SERIALIZATION_CONTEXT,
        clean_reference=clean_reference,
    )
    assert [(record["name"], record["provider"]) for record in records] == [
        ("MODES", "clean_isolated_internal_import_expression_v1")
    ]

    module.CURRENT = "runtime-import-substitution"
    with pytest.raises(CanonicalExecutionOriginError, match="CURRENT"):
        provenance_module._loaded_module_import_records(
            module.__name__,
            module,
            reference=reference,
            state_plan=state_plan,
        )
    module.CURRENT = owner.CURRENT

    module.MODES = ("runtime-substitution",)
    with pytest.raises(CanonicalExecutionOriginError, match="MODES"):
        provenance_module._attest_module_runtime_state(
            module_name=module.__name__,
            module=module,
            reference=reference,
            state_plan=state_plan,
            state_context=provenance_module._EMPTY_STATE_SERIALIZATION_CONTEXT,
            clean_reference=clean_reference,
        )


def test_diakrino_selector_calibration_modes_have_internal_import_state_contract() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/tabnetics/feature_selection/mnpo/portfolio.py"
    )
    tree = ast.parse(source.read_bytes(), filename=str(source))
    reference = {
        "module_state_bindings": provenance_module._source_direct_assignment_bindings(
            tree
        ),
        "imports": provenance_module._source_tabnetics_import_bindings(
            tree,
            module_name="tabnetics.feature_selection.mnpo.portfolio",
            is_package=False,
        ),
    }
    state_plan = provenance_module._source_static_state_plan(
        reference,
        module_name="tabnetics.feature_selection.mnpo.portfolio",
    )
    spec = state_plan.internal_import_state_specs[
        "DIAKRINO_SELECTOR_PRIOR_CALIBRATION_MODES"
    ]
    assert [binding["local_name"] for binding in spec["import_bindings"]] == [
        "DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT",
        "DIAKRINO_SELECTOR_PRIOR_CALIBRATION_NONE",
    ]


@pytest.mark.parametrize(
    "source_text",
    [
        """
        if True:
            from tabnetics._provenance_owner import CURRENT
        MODES = (CURRENT,)
        """,
        """
        from tabnetics._provenance_owner import CURRENT
        if True:
            MODES = (CURRENT,)
        """,
    ],
)
def test_internal_import_state_contract_excludes_guarded_imports_and_targets(
    source_text: str,
) -> None:
    tree = ast.parse(textwrap.dedent(source_text))
    reference = {
        "module_state_bindings": provenance_module._source_direct_assignment_bindings(
            tree
        ),
        "imports": provenance_module._source_tabnetics_import_bindings(
            tree,
            module_name="tabnetics._provenance_consumer",
            is_package=False,
        ),
    }
    state_plan = provenance_module._source_static_state_plan(
        reference,
        module_name="tabnetics._provenance_consumer",
    )
    assert "MODES" not in state_plan.internal_import_state_specs


def test_source_reference_attests_module_scope_conditional_and_try_definitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, source, reference = _load_module_scope_definition_fixture(
        tmp_path,
        monkeypatch,
        """
        MODE = "enabled"
        if MODE == "enabled":
            class ConditionalExport:
                def branch(self):
                    return "enabled"
        else:
            class ConditionalExport:
                def branch(self):
                    return "disabled"

        try:
            raise RuntimeError("fixture")
        except RuntimeError:
            def TryExport(value=3):
                return value
        """,
        suffix="module_scope_positive",
    )

    groups = {str(group["name"]): group for group in reference["definition_groups"]}
    assert len(groups["ConditionalExport"]["variants"]) == 2
    assert groups["ConditionalExport"]["conditional"] is True
    assert module.ConditionalExport().branch() == "enabled"
    assert module.TryExport() == 3
    assert [record["name"] for record in reference["definitions"]] == [
        "ConditionalExport",
        "TryExport",
    ]

    prepared = provenance_module._PreparedLoadedModule(
        module_name=module.__name__,
        module=module,
        source=source,
        source_sha256=provenance_module.sha256_file(source),
        is_package=False,
        reference=reference,
        state_plan=provenance_module._source_static_state_plan(
            reference,
            module_name=module.__name__,
        ),
    )
    provenance_module._validate_loaded_module_code_origin(prepared)


def test_source_reference_rejects_injected_nested_module_symbol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, source, _reference = _load_module_scope_definition_fixture(
        tmp_path,
        monkeypatch,
        """
        def factory():
            def nested():
                return "nested"
            return nested

        if True:
            def ConditionalExport():
                return "source"
        """,
        suffix="module_scope_negative",
    )
    raw_reference = provenance_module._source_symbol_reference(
        str(source),
        provenance_module.sha256_file(source),
        module.__name__,
        False,
    )
    module.leaked_nested = module.factory()
    with pytest.raises(
        CanonicalExecutionOriginError, match="unrecognized local symbol"
    ):
        provenance_module._resolve_source_symbol_reference(
            raw_reference,
            module_name=module.__name__,
            module=module,
            source=source,
        )

    del module.leaked_nested
    module.ConditionalExport = module.factory()
    with pytest.raises(
        CanonicalExecutionOriginError,
        match="unexpected qualified name",
    ):
        provenance_module._resolve_source_symbol_reference(
            raw_reference,
            module_name=module.__name__,
            module=module,
            source=source,
        )


def test_clean_reference_rejects_inactive_source_branch_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source-authored but inactive branch is not a clean-runtime substitute."""

    live_package_root = Path(provenance_module.__file__).resolve().parents[2]
    clean_package_root = tmp_path / "clean-package" / "tabnetics"
    for relative in (
        "__init__.py",
        "core/paths.py",
        "validation/core/provenance.py",
    ):
        destination = clean_package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live_package_root / relative, destination)
    for relative in (
        "core/__init__.py",
        "validation/__init__.py",
        "validation/core/__init__.py",
    ):
        destination = clean_package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("", encoding="utf-8")

    module, source, active_reference = _load_module_scope_definition_fixture(
        clean_package_root,
        monkeypatch,
        """
        if globals().get("_BRANCH_OVERRIDE") == "inactive":
            def BranchExport():
                return "inactive"
        else:
            def BranchExport():
                return "clean"

        try:
            raise RuntimeError("fixture")
        except RuntimeError:
            def TryExport():
                return "clean-try"
        """,
        suffix="clean_branch_selection",
    )
    assert module.BranchExport() == "clean"
    assert module.TryExport() == "clean-try"
    active_selection = provenance_module._selected_definition_variants_sha256(
        active_reference
    )
    active_prepared = provenance_module._PreparedLoadedModule(
        module_name=module.__name__,
        module=module,
        source=source,
        source_sha256=provenance_module.sha256_file(source),
        is_package=False,
        reference=active_reference,
        state_plan=provenance_module._source_static_state_plan(
            active_reference,
            module_name=module.__name__,
        ),
    )
    active_request, active_expected = provenance_module._clean_reference_request(
        (active_prepared,),
        package_root=clean_package_root,
    )
    assert (
        provenance_module._run_clean_isolated_reference(
            active_request,
            active_expected,
        )
        is not None
    )

    module._BRANCH_OVERRIDE = "inactive"
    loader = module.__loader__
    assert isinstance(loader, SourceFileLoader)
    loader.exec_module(module)
    assert module.BranchExport() == "inactive"
    raw_reference = provenance_module._source_symbol_reference(
        str(source),
        provenance_module.sha256_file(source),
        module.__name__,
        False,
    )
    inactive_reference = provenance_module._resolve_source_symbol_reference(
        raw_reference,
        module_name=module.__name__,
        module=module,
        source=source,
    )
    inactive_selection = provenance_module._selected_definition_variants_sha256(
        inactive_reference
    )
    assert inactive_selection != active_selection

    prepared = provenance_module._PreparedLoadedModule(
        module_name=module.__name__,
        module=module,
        source=source,
        source_sha256=provenance_module.sha256_file(source),
        is_package=False,
        reference=inactive_reference,
        state_plan=provenance_module._source_static_state_plan(
            inactive_reference,
            module_name=module.__name__,
        ),
    )
    request, expected = provenance_module._clean_reference_request(
        (prepared,),
        package_root=clean_package_root,
    )
    assert request["modules"][0]["definition_selection_sha256"] == inactive_selection
    with pytest.raises(
        CanonicalExecutionOriginError,
        match="definition selection differs from the live module",
    ):
        provenance_module._run_clean_isolated_reference(request, expected)


def test_guarded_internal_import_annotation_fallback_is_inactive_but_runtime_use_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation_module, annotation_source, annotation_reference = (
        _load_module_scope_definition_fixture(
            tmp_path,
            monkeypatch,
            """
            from __future__ import annotations

            try:
                from tabnetics._missing_guarded_owner import Missing as GuardedAlias
            except ImportError:
                GuardedAlias = object

            def annotation_only(value: GuardedAlias) -> str:
                return "ok"
            """,
            suffix="guarded_annotation_fallback",
        )
    )
    annotation_state_plan = provenance_module._source_static_state_plan(
        annotation_reference,
        module_name=annotation_module.__name__,
    )
    assert "GuardedAlias" not in annotation_reference["module_state_references"]
    assert (
        provenance_module._loaded_module_import_records(
            annotation_module.__name__,
            annotation_module,
            reference=annotation_reference,
            state_plan=annotation_state_plan,
        )
        == []
    )

    runtime_module, _runtime_source, runtime_reference = (
        _load_module_scope_definition_fixture(
            tmp_path,
            monkeypatch,
            """
            from __future__ import annotations

            try:
                from tabnetics._missing_guarded_owner import Missing as GuardedAlias
            except ImportError:
                GuardedAlias = object

            def runtime_use():
                return GuardedAlias
            """,
            suffix="guarded_runtime_fallback",
        )
    )
    runtime_state_plan = provenance_module._source_static_state_plan(
        runtime_reference,
        module_name=runtime_module.__name__,
    )
    assert "GuardedAlias" in runtime_reference["module_state_references"]
    runtime_module.GuardedAlias = object()
    with pytest.raises(
        CanonicalExecutionOriginError,
        match="has no loaded source-declared internal owner",
    ):
        provenance_module._loaded_module_import_records(
            runtime_module.__name__,
            runtime_module,
            reference=runtime_reference,
            state_plan=runtime_state_plan,
        )


def test_closure_rejects_custom_metaclass_without_dispatch() -> None:
    marker = {"triggered": False}

    class MarkerMeta(type):
        def __getattribute__(cls, name: str):
            marker["triggered"] = True
            return super().__getattribute__(name)

    class Replacement(metaclass=MarkerMeta):
        pass

    Replacement.__module__ = pipeline_module.__name__
    Replacement.__qualname__ = "DFFSConfig"
    marker["triggered"] = False
    original = pipeline_module.__dict__["DFFSConfig"]
    pipeline_module.__dict__["DFFSConfig"] = Replacement
    assert marker["triggered"] is False
    try:
        with pytest.raises(CanonicalExecutionOriginError, match="metaclass"):
            capture_loaded_tabnetics_module_closure()
        assert marker["triggered"] is False
    finally:
        pipeline_module.__dict__["DFFSConfig"] = original


def test_state_and_builtin_proxy_rejection_do_not_dispatch_metaclass() -> None:
    marker = {"triggered": False}

    class MarkerMeta(type):
        def __getattribute__(cls, name: str):
            marker["triggered"] = True
            return super().__getattribute__(name)

    class MarkerValue(metaclass=MarkerMeta):
        pass

    with pytest.raises(provenance_module._ExecutionStateUnsupported):
        provenance_module._state_value_payload(MarkerValue())
    assert marker["triggered"] is False

    class BuiltinProxy(metaclass=MarkerMeta):
        pass

    with pytest.raises(CanonicalExecutionOriginError, match="untrusted runtime type"):
        provenance_module._builtin_identity_payload(BuiltinProxy())
    assert marker["triggered"] is False


def test_rebound_provenance_builtins_rejects_without_proxy_dispatch() -> None:
    marker = {"triggered": False}

    class BuiltinsProxy:
        def __getattribute__(self, name: str):
            marker["triggered"] = True
            raise AssertionError(f"unexpected proxy access: {name}")

    original = provenance_module.__dict__["builtins"]
    provenance_module.__dict__["builtins"] = BuiltinsProxy()
    try:
        with pytest.raises(
            CanonicalExecutionOriginError, match="builtin module binding"
        ):
            capture_loaded_tabnetics_module_closure()
        assert marker["triggered"] is False
    finally:
        provenance_module.__dict__["builtins"] = original


def test_rebound_provenance_logging_rejects_without_proxy_dispatch() -> None:
    marker = {"triggered": False}

    class LoggingProxy:
        def __getattribute__(self, name: str):
            marker["triggered"] = True
            raise AssertionError(f"unexpected proxy access: {name}")

    original = provenance_module.__dict__["logging"]
    provenance_module.__dict__["logging"] = LoggingProxy()
    try:
        with pytest.raises(
            CanonicalExecutionOriginError, match="logging module binding"
        ):
            capture_loaded_tabnetics_module_closure()
        assert marker["triggered"] is False
    finally:
        provenance_module.__dict__["logging"] = original


@pytest.mark.parametrize("binding_name", ("dataclasses", "inspect", "importlib"))
def test_rebound_verifier_import_rejects_without_proxy_dispatch(
    binding_name: str,
) -> None:
    marker = {"triggered": False}

    class ImportProxy:
        def __getattribute__(self, name: str):
            marker["triggered"] = True
            raise AssertionError(f"unexpected proxy access: {name}")

    original = provenance_module.__dict__[binding_name]
    provenance_module.__dict__[binding_name] = ImportProxy()
    try:
        with pytest.raises(
            CanonicalExecutionOriginError,
            match=rf"verifier import binding '{binding_name}'",
        ):
            capture_loaded_tabnetics_module_closure()
        assert marker["triggered"] is False
    finally:
        provenance_module.__dict__[binding_name] = original


def test_closure_rejects_factory_replacement_without_executing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = importlib.import_module("tabnetics.benchmarks.runner")
    monkeypatch.setitem(runner.__dict__, "__audit_evaluator_executed__", False)

    def body() -> dict[str, str]:
        globals()["__audit_evaluator_executed__"] = True
        return {}

    original = runner._build_integrated_parent_map
    replacement = types.FunctionType(
        body.__code__,
        runner.__dict__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    replacement.__module__ = original.__module__
    replacement.__qualname__ = original.__qualname__
    monkeypatch.setattr(runner, "_build_integrated_parent_map", replacement)

    with pytest.raises(CanonicalExecutionOriginError):
        capture_loaded_tabnetics_module_closure()
    assert runner.__audit_evaluator_executed__ is False


def test_closure_rejects_custom_mapping_without_dispatching_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = importlib.import_module("tabnetics.benchmarks.runner")
    marker = {"items_called": False}

    class MarkerMapping(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):
            marker["items_called"] = True
            return ()

    monkeypatch.setattr(runner, "_INTEGRATED_PARENT_MAP", MarkerMapping())
    with pytest.raises(CanonicalExecutionOriginError, match="_INTEGRATED_PARENT_MAP"):
        capture_loaded_tabnetics_module_closure()
    assert marker["items_called"] is False


def test_conformal_module_does_not_eagerly_bind_mapie_callables() -> None:
    conformal = importlib.import_module("tabnetics.feature_selection.conformal")
    assert not hasattr(conformal, "_SplitConformal")
    assert not hasattr(conformal, "_CrossConformal")
    assert not hasattr(conformal, "_RAPSScore")
    capture_loaded_tabnetics_module_closure()


def test_conformal_import_is_mapie_lazy_in_a_fresh_process() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_root = repo_root / "core" / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import tabnetics.feature_selection.conformal as conformal; "
                "assert 'mapie.classification' not in sys.modules; "
                "assert not hasattr(conformal, '_SplitConformal')"
            ),
        ],
        cwd=repo_root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                part
                for part in (str(source_root), os.environ.get("PYTHONPATH", ""))
                if part
            ),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_external_binding_fingerprint_rejects_same_nominal_mapie_class_spoof() -> None:
    """The structural check remains defense-in-depth for simple MAPIE spoofs."""

    conformal = importlib.import_module("tabnetics.feature_selection.conformal")
    api = conformal._load_mapie_api()
    if api is None or api[0]:
        pytest.skip(
            "MAPIE SplitConformalClassifier is not available in this environment"
        )
    original = api[1]
    assert original is not None

    class Spoof:
        pass

    Spoof.__module__ = original.__module__
    Spoof.__qualname__ = original.__qualname__
    assert provenance_module._external_binding_identity(Spoof) != (
        provenance_module._external_binding_identity(original)
    )


def test_closure_rejects_builtin_shadow_and_function_builtin_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tabnetics.feature_selection.methods.filter")
    monkeypatch.setattr(module, "len", lambda value: 0, raising=False)
    with pytest.raises(CanonicalExecutionOriginError, match="shadows builtin 'len'"):
        capture_loaded_tabnetics_module_closure()
    monkeypatch.undo()

    original = module.relieff_selection
    original_builtins = module.__dict__["__builtins__"]
    builtins_mapping = dict(
        original_builtins
        if isinstance(original_builtins, dict)
        else vars(original_builtins)
    )
    builtins_mapping["len"] = lambda value: 0
    monkeypatch.setitem(module.__dict__, "__builtins__", builtins_mapping)
    monkeypatch.setattr(
        module,
        "relieff_selection",
        _source_identical_function_with_defaults(original),
    )
    with pytest.raises(CanonicalExecutionOriginError, match="builtin 'len'"):
        capture_loaded_tabnetics_module_closure()


def test_fresh_process_rejects_default_global_mapping_dependency_and_builtin_mutations() -> (
    None
):
    repo_root = Path(__file__).resolve().parents[2]
    source_root = repo_root / "core" / "src"
    script = textwrap.dedent(
        """
        import types
        from collections.abc import Mapping

        from tabnetics.validation.core.provenance import (
            CanonicalExecutionOriginError,
            capture_loaded_tabnetics_module_closure,
        )
        from tabnetics.feature_selection.methods import filter as filter_module
        from tabnetics.distribution import selector
        from tabnetics.benchmarks import runner

        def reject(label):
            try:
                capture_loaded_tabnetics_module_closure()
            except CanonicalExecutionOriginError:
                print("rejected:" + label)
                return
            raise SystemExit("closure accepted " + label)

        original = filter_module.relieff_selection
        changed = types.FunctionType(
            original.__code__, original.__globals__, original.__name__, (999,), original.__closure__
        )
        changed.__kwdefaults__ = original.__kwdefaults__
        changed.__module__ = original.__module__
        changed.__qualname__ = original.__qualname__
        filter_module.relieff_selection = changed
        reject("default")
        filter_module.relieff_selection = original

        positive = set(selector._POSITIVE_ONLY_FAMILIES)
        selector._POSITIVE_ONLY_FAMILIES.clear()
        reject("global")
        selector._POSITIVE_ONLY_FAMILIES.update(positive)

        marker = {"items": False}
        class MarkerMapping(Mapping):
            def __getitem__(self, key): raise KeyError(key)
            def __iter__(self): return iter(())
            def __len__(self): return 0
            def items(self):
                marker["items"] = True
                return ()
        original_map = runner._INTEGRATED_PARENT_MAP
        runner._INTEGRATED_PARENT_MAP = MarkerMapping()
        reject("mapping")
        if marker["items"]:
            raise SystemExit("mapping items dispatched")
        runner._INTEGRATED_PARENT_MAP = original_map

        original_len = getattr(filter_module, "len", None)
        filter_module.len = lambda value: 0
        reject("builtin")
        if original_len is None:
            del filter_module.len
        else:
            filter_module.len = original_len
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for label in ("default", "global", "mapping", "builtin"):
        assert f"rejected:{label}" in result.stdout


def test_clean_reference_timeout_kills_process_group_and_drains_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("process-group cleanup is POSIX-specific")

    class FakeProcess:
        pid = 424242

        def __init__(self) -> None:
            self.returncode = None
            self.calls: list[float | None] = []

        def poll(self):
            return None

        def communicate(self, _payload=None, timeout=None):
            self.calls.append(timeout)
            if len(self.calls) == 1:
                raise subprocess.TimeoutExpired("clean-reference", timeout)
            return b"", b""

    process = FakeProcess()
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        provenance_module.subprocess, "Popen", lambda *_a, **_kw: process
    )
    monkeypatch.setattr(
        provenance_module.os,
        "killpg",
        lambda pid, signal_value: killed.append((pid, signal_value)),
    )
    request, expected = _clean_reference_runtime_request()

    with pytest.raises(
        CanonicalExecutionOriginError, match="timed out and was terminated"
    ):
        provenance_module._run_clean_isolated_reference(request, expected)
    assert killed == [(process.pid, provenance_module.signal.SIGKILL)]
    assert len(process.calls) == 2
    assert process.calls[1] == 5.0


def test_clean_reference_uses_staged_source_and_child_owned_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source swap after staging cannot change the clean child's imported code."""

    request, expected = _clean_reference_runtime_request()
    live_root = Path(str(request["package_root"])).resolve()
    staged_live_root = tmp_path / "mutable-live" / "tabnetics"
    shutil.copytree(live_root, staged_live_root)
    request = dict(request)
    manifest = provenance_module._source_tree_manifest(staged_live_root)
    request.update(
        {
            "package_root": str(staged_live_root),
            "source_manifest": manifest,
            "source_manifest_sha256": canonical_json_sha256(manifest),
        }
    )
    runtime_source = staged_live_root / "core" / "runtime.py"
    original_runtime_source = runtime_source.read_bytes()
    original_popen = provenance_module.subprocess.Popen
    observed: dict[str, object] = {}

    def staged_popen(command: object, **kwargs: object):
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        observed["environment"] = dict(kwargs["env"])
        # This is deliberately after the parent has copied and verified the
        # requested sources, but before the child process starts importing.
        runtime_source.write_bytes(
            original_runtime_source + b"\n# source swapped after staging\n"
        )
        return original_popen(command, **kwargs)

    monkeypatch.setenv("HOME", "parent-home-sentinel")
    monkeypatch.setenv("HF_HOME", "parent-hf-sentinel")
    monkeypatch.setenv("MPLCONFIGDIR", "parent-mpl-sentinel")
    monkeypatch.setattr(provenance_module.subprocess, "Popen", staged_popen)

    reference = provenance_module._run_clean_isolated_reference(request, expected)

    assert reference is not None
    assert reference.source_tree_sha256 == request["source_manifest_sha256"]
    assert reference.environment["HOME"] == "child_owned:home"
    assert reference.environment["HF_HOME"] == "child_owned:hf_home"
    assert reference.environment["MPLCONFIGDIR"] == "child_owned:matplotlib_config_dir"
    assert (
        reference.isolated_paths["source_package_root"] == "verified_temp_source_copy"
    )
    assert reference.isolated_paths["xdg_cache_home"] == "child_owned:xdg_cache_home"

    command = list(observed["command"])
    assert command[1:3] == ["-I", "-B"]
    work_dir = Path(str(observed["cwd"])).resolve()
    child_environment = observed["environment"]
    assert child_environment["HOME"] != "parent-home-sentinel"
    assert child_environment["HF_HOME"] != "parent-hf-sentinel"
    assert child_environment["MPLCONFIGDIR"] != "parent-mpl-sentinel"
    for key in (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "MPLCONFIGDIR",
        "TMPDIR",
    ):
        Path(str(child_environment[key])).resolve().relative_to(work_dir)
    assert provenance_module._source_tree_manifest(staged_live_root) != manifest


def test_generate_plan_profile_manifests_are_clean_reference_sealed() -> None:
    """Import-time profile construction must remain admissible to closure capture."""

    module_name = "tabnetics.validation.generate_plan"
    plan = importlib.import_module(module_name)
    assert plan.VALIDATION18_SINGLETONS_PROFILE_IDS
    closure = capture_loaded_tabnetics_module_closure()
    recorded_names = {record["module"] for record in closure["modules"]}
    assert module_name in recorded_names


def test_final_closure_captures_module_loaded_during_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "tabnetics.feature_selection.methods.filter"
    methods_package = importlib.import_module("tabnetics.feature_selection.methods")
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delattr(methods_package, "filter", raising=False)
    assert module_name not in sys.modules

    benchmark.validate_loaded_tabnetics_module_closure()
    importlib.import_module(module_name)
    execution = {"loaded_package_modules": capture_loaded_tabnetics_module_closure()}
    recorded_names = {
        record["module"] for record in execution["loaded_package_modules"]["modules"]
    }
    assert module_name in recorded_names


def test_parallel_worker_closure_is_merged_into_final_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "tabnetics.feature_selection.methods.filter"
    methods_package = importlib.import_module("tabnetics.feature_selection.methods")
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delattr(methods_package, "filter", raising=False)
    assert module_name not in sys.modules

    source_root = Path(__file__).resolve().parents[2] / "core" / "src"
    script = (
        "import importlib, json; "
        "from tabnetics.validation.core.provenance import capture_loaded_tabnetics_module_closure; "
        f"importlib.import_module({module_name!r}); "
        "print(json.dumps(capture_loaded_tabnetics_module_closure()))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    worker_closure = json.loads(result.stdout)
    assert module_name not in sys.modules
    args = benchmark.build_arg_parser().parse_args(
        ["--datasets", "synthetic_easy_dfshift", "--seeds", "11"]
    )
    execution = benchmark._build_canonical_execution_contract(
        args,
        ["synthetic_easy_dfshift"],
        [_materialized_input_identity()],
        worker_module_closures=[worker_closure],
    )
    recorded_names = {
        record["module"] for record in execution["loaded_package_modules"]["modules"]
    }
    assert execution["worker_module_closures_complete"] is True
    assert module_name in recorded_names


def test_parallel_runner_rejects_missing_worker_closure() -> None:
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--seeds",
            "11",
        ]
    )
    closures: list[dict[str, object]] = []
    issues: list[str] = []
    benchmark._collect_task_execution_module_closures(
        {"execution_module_closure_complete": False},
        dataset_id="synthetic_easy_dfshift",
        seed=11,
        closures=closures,
        issues=issues,
    )
    assert not closures
    assert issues
    with pytest.raises(
        CanonicalExecutionOriginError,
        match="missing one or more execution-worker module closures",
    ):
        benchmark._build_canonical_execution_contract(
            args,
            ["synthetic_easy_dfshift"],
            [_materialized_input_identity()],
            worker_module_closures_complete=False,
        )


def test_hard_timeout_worker_returns_its_module_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "fork" not in set(benchmark.mp.get_all_start_methods()):
        pytest.skip("fork start method is unavailable")

    module_name = "tabnetics.feature_selection.methods.filter"
    methods_package = importlib.import_module("tabnetics.feature_selection.methods")
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delattr(methods_package, "filter", raising=False)
    monkeypatch.setenv("TABNETICS_HARD_TIMEOUT_START_METHOD", "fork")

    class _FakePipeline:
        def __init__(self, _cfg: object) -> None:
            pass

        def run(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            importlib.import_module(module_name)
            return {"status": "ok"}

    monkeypatch.setattr(
        benchmark, "DistributionFeatureSelectionPipeline", _FakePipeline
    )
    with pytest.raises(
        CanonicalExecutionOriginError,
        match="runtime import 'DistributionFeatureSelectionPipeline'",
    ):
        benchmark._run_pipeline_with_hard_timeout(
            cfg=benchmark.DFFSConfig(
                random_seed=11,
                fs_fraction=0.5,
                n_final_features=2,
                max_dist_features=2,
                prefilter_top_k=2,
                enabled_methods=("mutual_information",),
            ),
            X=[[0.0, 1.0], [1.0, 0.0]],
            y=[0, 1],
            dataset_name="closure-fixture",
            seed=11,
            timeout_sec=5.0,
            quiet_worker_logs=True,
            use_hard_kill=True,
        )


def test_canonical_runner_rejects_legacy_pipeline_import_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = benchmark.build_arg_parser().parse_args(
        ["--datasets", "synthetic_easy_dfshift"]
    )
    legacy_pipeline = types.ModuleType("experiments.df_fs_pipeline")
    legacy_pipeline.__file__ = "/tmp/research/df_fs_pipeline.py"
    monkeypatch.setattr(benchmark, "_CANONICAL_PIPELINE_IMPORT_TARGET", legacy_pipeline)

    with pytest.raises(CanonicalExecutionOriginError, match="unexpected module"):
        benchmark._build_canonical_execution_contract(
            args,
            ["synthetic_easy_dfshift"],
            [_materialized_input_identity()],
        )


def test_canonical_contract_rejects_missing_bootstrap_import_labels() -> None:
    with pytest.raises(
        CanonicalExecutionOriginError, match="bootstrap_import_labels_missing"
    ):
        build_canonical_execution_provenance(
            args={"fixture": True},
            selected_dataset_ids=["synthetic_easy_dfshift"],
            import_targets={"pipeline": benchmark._CANONICAL_PIPELINE_IMPORT_TARGET},
        )


def test_canonical_contract_rejects_incomplete_worker_module_closures() -> None:
    args = benchmark.build_arg_parser().parse_args(
        ["--datasets", "synthetic_easy_dfshift", "--seeds", "11"]
    )
    with pytest.raises(
        CanonicalExecutionOriginError,
        match="missing one or more execution-worker module closures",
    ):
        benchmark._build_canonical_execution_contract(
            args,
            ["synthetic_easy_dfshift"],
            [_materialized_input_identity()],
            worker_module_closures_complete=False,
        )


def test_canonical_contract_rejects_shadow_pipeline_module_and_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pipeline = sys.modules["tabnetics.pipeline.pipeline"]
    shadow = types.ModuleType("tabnetics.pipeline.pipeline")
    shadow.__file__ = str(real_pipeline.__file__)
    shadow.ClassificationConfig = (
        benchmark._CANONICAL_CLASSIFICATION_CONFIG_IMPORT_TARGET
    )
    shadow.DFFSConfig = benchmark._CANONICAL_DFFS_CONFIG_IMPORT_TARGET
    shadow.DistributionFitterConfig = benchmark._CANONICAL_DIST_CONFIG_IMPORT_TARGET
    fake_pipeline = type("DistributionFeatureSelectionPipeline", (), {})
    fake_pipeline.__module__ = "tabnetics.pipeline.pipeline"
    shadow.DistributionFeatureSelectionPipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "tabnetics.pipeline.pipeline", shadow)
    monkeypatch.setattr(benchmark, "_CANONICAL_PIPELINE_IMPORT_TARGET", fake_pipeline)

    args = benchmark.build_arg_parser().parse_args(
        ["--datasets", "synthetic_easy_dfshift", "--seeds", "11"]
    )
    with pytest.raises(
        CanonicalExecutionOriginError, match="captured import-time identity"
    ):
        benchmark._build_canonical_execution_contract(
            args,
            ["synthetic_easy_dfshift"],
            [_materialized_input_identity()],
        )


def test_canonical_contract_rejects_shadow_feature_selection_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = types.ModuleType("tabnetics.feature_selection")
    shadow.FeatureSelector = type("FeatureSelector", (), {})
    monkeypatch.setitem(sys.modules, "tabnetics.feature_selection", shadow)

    args = benchmark.build_arg_parser().parse_args(
        ["--datasets", "synthetic_easy_dfshift", "--seeds", "11"]
    )
    with pytest.raises(CanonicalExecutionOriginError, match="Loaded tabnetics module"):
        benchmark._build_canonical_execution_contract(
            args,
            ["synthetic_easy_dfshift"],
            [_materialized_input_identity()],
        )


def test_pipeline_selector_lookup_is_sealed_against_feature_selection_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = types.ModuleType("tabnetics.feature_selection")
    shadow.FeatureSelector = type("FeatureSelector", (), {})
    monkeypatch.setitem(sys.modules, "tabnetics.feature_selection", shadow)

    assert (
        pipeline_module._load_feature_selector_cls()
        is benchmark._CANONICAL_FEATURE_SELECTOR_IMPORT_TARGET
    )


def test_canonical_contract_rejects_shadow_lazy_filter_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = types.ModuleType("tabnetics.feature_selection.methods.filter")
    monkeypatch.setitem(
        sys.modules, "tabnetics.feature_selection.methods.filter", shadow
    )

    args = benchmark.build_arg_parser().parse_args(
        ["--datasets", "synthetic_easy_dfshift", "--seeds", "11"]
    )
    with pytest.raises(CanonicalExecutionOriginError, match="Loaded tabnetics module"):
        benchmark._build_canonical_execution_contract(
            args,
            ["synthetic_easy_dfshift"],
            [_materialized_input_identity()],
        )


def test_runner_rejects_lazy_filter_shadow_before_task_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--seeds",
            "11",
            "--output-dir",
            str(tmp_path),
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "tabnetics.feature_selection.methods.filter",
        types.ModuleType("tabnetics.feature_selection.methods.filter"),
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("benchmark task executed despite an invalid import seal")

    monkeypatch.setattr(benchmark, "_run_dataset_seed_task", fail_if_called)
    with pytest.raises(CanonicalExecutionOriginError, match="Loaded tabnetics module"):
        benchmark.run_benchmark(args)


def test_fresh_process_rejects_preimport_distribution_selector_shadow() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_root = repo_root / "core" / "src"
    script = textwrap.dedent(
        """
        import sys
        import types
        from importlib.machinery import ModuleSpec, SourceFileLoader

        import tabnetics.distribution as distribution_package
        import tabnetics.distribution.selector as real_selector

        module_name = "tabnetics.distribution.selector"
        source = str(real_selector.__file__)
        loader = SourceFileLoader(module_name, source)
        shadow = types.ModuleType(module_name)
        shadow.__dict__.update(vars(real_selector))
        shadow.__file__ = source
        shadow.__loader__ = loader
        shadow.__spec__ = ModuleSpec(module_name, loader, origin=source)
        shadow.__package__ = "tabnetics.distribution"

        class FakeSelector(real_selector.UnifiedDistributionSelectorV6):
            pass

        FakeSelector.__name__ = "UnifiedDistributionSelectorV6"
        FakeSelector.__qualname__ = "UnifiedDistributionSelectorV6"
        FakeSelector.__module__ = module_name
        shadow.UnifiedDistributionSelectorV6 = FakeSelector
        sys.modules[module_name] = shadow
        distribution_package.selector = shadow

        from tabnetics.benchmarks import runner as benchmark
        from tabnetics.validation.core.provenance import CanonicalExecutionOriginError

        try:
            benchmark._assert_canonical_execution_bootstrap_seal()
            benchmark.validate_loaded_tabnetics_module_closure()
        except CanonicalExecutionOriginError as exc:
            print("preflight_rejected", type(exc).__name__)
        else:
            raise SystemExit("canonical benchmark accepted a metadata-cloned selector shadow")
        """
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), existing_pythonpath) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "preflight_rejected CanonicalExecutionOriginError" in result.stdout


def test_fresh_process_rejects_copied_selector_descriptors() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_root = repo_root / "core" / "src"
    script = textwrap.dedent(
        """
        import sys
        import types
        from importlib.machinery import ModuleSpec, SourceFileLoader

        import tabnetics.distribution as distribution_package
        import tabnetics.distribution.selector as real_selector

        module_name = "tabnetics.distribution.selector"
        source = str(real_selector.__file__)
        loader = SourceFileLoader(module_name, source)
        shadow = types.ModuleType(module_name)
        shadow.__dict__.update(vars(real_selector))
        shadow.__file__ = source
        shadow.__loader__ = loader
        shadow.__spec__ = ModuleSpec(module_name, loader, origin=source)
        shadow.__package__ = "tabnetics.distribution"

        real_class = real_selector.UnifiedDistributionSelectorV6
        copied_members = {
            name: value
            for name, value in vars(real_class).items()
            if name not in {"__dict__", "__weakref__"}
        }
        FakeSelector = type(
            "UnifiedDistributionSelectorV6",
            real_class.__bases__,
            copied_members,
        )
        FakeSelector.__qualname__ = "UnifiedDistributionSelectorV6"
        FakeSelector.__module__ = module_name
        shadow.UnifiedDistributionSelectorV6 = FakeSelector
        sys.modules[module_name] = shadow
        distribution_package.selector = shadow

        from tabnetics.benchmarks import runner as benchmark
        from tabnetics.validation.core.provenance import CanonicalExecutionOriginError

        try:
            benchmark.validate_loaded_tabnetics_module_closure()
        except CanonicalExecutionOriginError as exc:
            if "live verified module globals" not in str(exc):
                raise SystemExit(f"wrong preflight rejection: {exc}")
            print("preflight_rejected", type(exc).__name__)
        else:
            raise SystemExit("canonical benchmark accepted copied selector descriptors")
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "preflight_rejected CanonicalExecutionOriginError" in result.stdout


def test_runtime_import_binding_rejects_substituted_pipeline() -> None:
    class FakePipeline:
        pass

    original = benchmark.DistributionFeatureSelectionPipeline
    try:
        benchmark.DistributionFeatureSelectionPipeline = FakePipeline
        with pytest.raises(
            CanonicalExecutionOriginError,
            match="runtime import 'DistributionFeatureSelectionPipeline'",
        ):
            benchmark.validate_loaded_tabnetics_module_closure()
    finally:
        benchmark.DistributionFeatureSelectionPipeline = original


def test_canonical_finalization_rejects_task_time_metadata_cloned_filter_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "tabnetics.feature_selection.methods.filter"
    methods_package = importlib.import_module("tabnetics.feature_selection.methods")
    real_module = importlib.import_module(module_name)
    source = str(real_module.__file__)
    loader = SourceFileLoader(module_name, source)
    shadow = types.ModuleType(module_name)
    shadow.__name__ = module_name
    shadow.__file__ = source
    shadow.__loader__ = loader
    shadow.__spec__ = ModuleSpec(module_name, loader, origin=source)
    shadow.__package__ = "tabnetics.feature_selection.methods"
    loader.exec_module(shadow)

    def fake_mrmr_jmi_selection(*_args: object, **_kwargs: object) -> list[int]:
        return [0]

    fake_mrmr_jmi_selection = types.FunctionType(
        fake_mrmr_jmi_selection.__code__.replace(co_filename=source),
        shadow.__dict__,
        "mrmr_jmi_selection",
    )
    fake_mrmr_jmi_selection.__qualname__ = "mrmr_jmi_selection"
    fake_mrmr_jmi_selection.__module__ = module_name
    shadow.mrmr_jmi_selection = fake_mrmr_jmi_selection

    # This simulates a task-time import after a successful controller preflight.
    benchmark.validate_loaded_tabnetics_module_closure()
    monkeypatch.setitem(sys.modules, module_name, shadow)
    monkeypatch.setattr(methods_package, "filter", shadow, raising=False)
    args = benchmark.build_arg_parser().parse_args(
        ["--datasets", "synthetic_easy_dfshift", "--seeds", "11"]
    )
    with pytest.raises(
        CanonicalExecutionOriginError,
        match="mrmr_jmi_selection.*independently compiled verified source",
    ):
        benchmark._build_canonical_execution_contract(
            args,
            ["synthetic_easy_dfshift"],
            [_materialized_input_identity()],
        )


def test_canonical_contract_rejects_digest_and_import_origin_tampering() -> None:
    contract = _canonical_contract()
    assert canonical_execution_eligibility(contract) == (True, "")

    created_at_only = copy.deepcopy(contract)
    created_at_only["created_at"] = "2099-01-01T00:00:00+00:00"
    assert canonical_execution_eligibility(created_at_only) == (True, "")

    cli_tampered = copy.deepcopy(contract)
    cli_tampered["resolved_cli_config"]["arguments"]["max_workers"] = 99
    assert canonical_execution_eligibility(cli_tampered) == (
        False,
        "resolved_cli_config_digest_mismatch",
    )

    input_tampered = copy.deepcopy(contract)
    input_tampered["input_data_identity"]["selected_dataset_ids"] = ["leukemia_golub"]
    assert canonical_execution_eligibility(input_tampered) == (
        False,
        "input_data_identity_digest_mismatch",
    )

    import_tampered = copy.deepcopy(contract)
    import_tampered["import_origins"]["pipeline"]["path"] = str(
        Path(json.__file__).resolve()
    )
    assert canonical_execution_eligibility(import_tampered) == (
        False,
        "execution_provenance_fingerprint_mismatch",
    )

    closure_tampered = copy.deepcopy(contract)
    closure = closure_tampered["loaded_package_modules"]
    closure["modules"][0]["path"] = "/tmp/noncanonical-tabnetics-module.py"
    closure["modules"][0]["spec_origin"] = "/tmp/noncanonical-tabnetics-module.py"
    closure_digest = canonical_json_sha256(closure["modules"])
    closure["modules_sha256"] = closure_digest
    closure_tampered["loaded_package_modules_sha256"] = closure_digest
    closure_tampered["source_revision"]["loaded_package_modules_sha256"] = (
        closure_digest
    )
    closure_tampered["fingerprint_sha256"] = execution_fingerprint_sha256(
        closure_tampered
    )
    assert canonical_execution_eligibility(closure_tampered) == (
        False,
        "loaded_package_module_path_outside_package_root",
    )

    fingerprint_tampered = copy.deepcopy(contract)
    fingerprint_tampered["fingerprint_sha256"] = "0" * 64
    assert canonical_execution_eligibility(fingerprint_tampered) == (
        False,
        "execution_provenance_fingerprint_mismatch",
    )


def test_canonical_contract_requires_and_validates_materialized_xy_identity() -> None:
    args = benchmark.build_arg_parser().parse_args(
        ["--datasets", "synthetic_easy_dfshift", "--seeds", "11"]
    )
    with pytest.raises(
        CanonicalExecutionInputIdentityError,
        match="materialized-input coverage is incomplete",
    ):
        benchmark._build_canonical_execution_contract(
            args,
            ["synthetic_easy_dfshift"],
            [],
        )

    contract = _canonical_contract()
    tampered = copy.deepcopy(contract)
    input_identity = tampered["input_data_identity"]
    input_identity["materialized_inputs"][0]["x"]["sha256"] = "0" * 64
    input_identity["materialized_inputs_sha256"] = canonical_json_sha256(
        input_identity["materialized_inputs"]
    )
    tampered["input_data_identity_sha256"] = canonical_json_sha256(input_identity)
    tampered["fingerprint_sha256"] = execution_fingerprint_sha256(tampered)
    eligible, reason = canonical_execution_eligibility(tampered)
    assert eligible is False
    assert reason.startswith("input_data_identity_materialized_invalid:")


def test_mapie_aps_default_is_explicitly_noncanonical_with_preserved_inputs() -> None:
    """The APS FS default is operationally valid but lacks external identity proof."""

    default_config = pipeline_module.DFFSConfig()
    assert default_config.fs_use_conformal_efficiency is True
    assert default_config.fs_conformal_efficiency_method == "aps"
    reason = external_callable_identity_unattested_reason(default_config)
    assert reason == "external_callable_identity_unattested:mapie"

    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--seeds",
            "11",
            "--fs-use-conformal-efficiency",
            "--fs-conformal-efficiency-method",
            "aps",
        ]
    )
    assert external_callable_identity_unattested_reason(args) == reason
    with pytest.raises(CanonicalExecutionExternalDependencyError, match=reason):
        benchmark._build_canonical_execution_contract(
            args,
            ["synthetic_easy_dfshift"],
            [_materialized_input_identity()],
        )

    execution = benchmark._build_noncanonical_execution_contract(
        args,
        ["synthetic_easy_dfshift"],
        [_materialized_input_identity()],
        reason=reason,
        evidence_status=EXTERNAL_CALLABLE_UNATTESTED_EVIDENCE_STATUS,
        preserve_materialized_input_identity=True,
    )
    assert execution["evidence_status"] == EXTERNAL_CALLABLE_UNATTESTED_EVIDENCE_STATUS
    assert execution["canonical_scorecard_eligible"] is False
    assert execution["noncanonical_reason"] == reason
    assert execution["input_data_identity"]["materialized_inputs"]
    assert execution_row_fields(execution)["noncanonical_reason"] == reason
    assert canonical_execution_eligibility(execution) == (
        False,
        "evidence_status_not_canonical",
    )


def test_runner_marks_mapie_execution_noncanonical_and_preserves_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmark boundary must retain inputs when MAPIE attestation fails."""

    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--seeds",
            "11",
            "--output-dir",
            str(tmp_path),
            "--fs-use-conformal-efficiency",
            "--fs-conformal-efficiency-method",
            "aps",
        ]
    )
    materialized_input = _materialized_input_identity()
    reason = "external_callable_identity_unattested:mapie"
    assert external_callable_identity_unattested_reason(args) == reason
    captured: dict[str, object] = {}

    def fake_task(
        dataset_id: str,
        seed: int,
        _args: object,
    ) -> dict[str, object]:
        return {
            "rows": [_row(dataset_id, seed)],
            "failures": [],
            "model_bundles": [],
            "run_diagnostics": [],
            "materialized_input_identity": materialized_input,
            "execution_module_closure_complete": True,
        }

    def fake_canonical_contract(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise CanonicalExecutionExternalDependencyError(reason)

    def fake_noncanonical_contract(
        _args: object,
        selected_dataset_ids: object,
        materialized_input_records: object,
        _worker_module_closures: object = (),
        _worker_module_closures_complete: bool = True,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["selected_dataset_ids"] = selected_dataset_ids
        captured["materialized_input_records"] = materialized_input_records
        captured.update(kwargs)
        return {
            "schema_version": EXECUTION_PROVENANCE_SCHEMA_VERSION,
            "implementation_stack": "tabnetics_core",
            "evidence_status": EXTERNAL_CALLABLE_UNATTESTED_EVIDENCE_STATUS,
            "canonical_scorecard_eligible": False,
            "noncanonical_reason": reason,
            "input_data_identity": {
                "materialization_status": "complete",
                "materialized_inputs": list(materialized_input_records),
                "materialized_inputs_sha256": "fixture-materialized-input-set",
            },
            "fingerprint_sha256": "fixture-execution-fingerprint",
        }

    # The runner's normal seal rejects substituted functions before it can reach
    # the catch path.  Stub only those independent provenance boundaries here;
    # the contract helper tests above exercise their concrete implementations.
    monkeypatch.setattr(
        benchmark, "_assert_canonical_execution_bootstrap_seal", lambda: None
    )
    monkeypatch.setattr(
        benchmark, "validate_loaded_tabnetics_module_closure", lambda: None
    )
    monkeypatch.setattr(benchmark, "_run_dataset_seed_task", fake_task)
    monkeypatch.setattr(
        benchmark, "_build_canonical_execution_contract", fake_canonical_contract
    )
    monkeypatch.setattr(
        benchmark,
        "_build_noncanonical_execution_contract",
        fake_noncanonical_contract,
    )
    run_dir = benchmark.run_benchmark(args)

    execution = json.loads(
        (run_dir / "df_fs_execution_provenance.json").read_text(encoding="utf-8")
    )
    assert execution["evidence_status"] == EXTERNAL_CALLABLE_UNATTESTED_EVIDENCE_STATUS
    assert execution["canonical_scorecard_eligible"] is False
    assert execution["noncanonical_reason"] == (
        "external_callable_identity_unattested:mapie"
    )
    assert execution["input_data_identity"]["materialization_status"] == "complete"
    assert execution["input_data_identity"]["materialized_inputs"] == [
        materialized_input
    ]
    assert captured["selected_dataset_ids"] == ["synthetic_easy_dfshift"]
    assert captured["materialized_input_records"] == [materialized_input]
    assert captured["reason"] == reason
    assert captured["evidence_status"] == EXTERNAL_CALLABLE_UNATTESTED_EVIDENCE_STATUS
    assert captured["preserve_materialized_input_identity"] is True

    runs = pd.read_csv(run_dir / "df_fs_runs.csv")
    assert (
        runs.loc[0, "evidence_status"] == EXTERNAL_CALLABLE_UNATTESTED_EVIDENCE_STATUS
    )
    assert bool(runs.loc[0, "canonical_scorecard_eligible"]) is False
    assert runs.loc[0, "noncanonical_reason"] == (
        "external_callable_identity_unattested:mapie"
    )
    assert (
        runs.loc[0, "materialized_input_identity_sha256"]
        == materialized_input["materialized_input_sha256"]
    )
    assert not (run_dir / "df_fs_artifact_provenance.json").exists()


def test_contract_marks_missing_materialized_identity_noncanonical() -> None:
    args = benchmark.build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_easy_dfshift",
            "--seeds",
            "11",
        ]
    )
    execution = benchmark._build_noncanonical_execution_contract(
        args,
        ["synthetic_easy_dfshift"],
        [],
        reason="materialized_input_identity_missing",
    )

    assert execution["canonical_scorecard_eligible"] is False
    assert execution["evidence_status"] == "noncanonical_input_identity"
    assert canonical_execution_eligibility(execution) == (
        False,
        "evidence_status_not_canonical",
    )


def test_aggregate_excludes_legacy_noncanonical_benchmark_artifacts(
    tmp_path: Path,
) -> None:
    root_out = tmp_path / "out"
    job_id = "smoke/legacy/ds01"
    run_dir = root_out / job_id / "20260710_000000_df_fs_sota_benchmark"
    run_dir.mkdir(parents=True)
    execution = {
        "schema_version": EXECUTION_PROVENANCE_SCHEMA_VERSION,
        "implementation_stack": "experiments_legacy",
        "evidence_status": "legacy_noncanonical",
        "canonical_scorecard_eligible": False,
        "fingerprint_sha256": "legacy-fixture",
    }
    (run_dir / "df_fs_metadata.json").write_text(
        json.dumps({"execution_provenance": execution}),
        encoding="utf-8",
    )
    pd.DataFrame([_row("synthetic_easy_dfshift", 11)]).to_csv(
        run_dir / "df_fs_runs.csv",
        index=False,
    )
    pd.DataFrame([{"dataset_id": "synthetic_easy_dfshift"}]).to_csv(
        run_dir / "df_fs_summary.csv",
        index=False,
    )
    pd.DataFrame([{"dataset_id": "synthetic_easy_dfshift"}]).to_csv(
        run_dir / "df_fs_sota_comparison.csv",
        index=False,
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": job_id,
                        "kind": "run_df_fs_sota_benchmark",
                        "params": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "aggregate"

    aggregate(root_out, plan_path=plan, out_dir=out_dir)

    excluded = pd.read_csv(out_dir / "excluded_noncanonical_artifacts.csv")
    assert excluded.loc[0, "job_id"] == job_id
    assert (
        excluded.loc[0, "exclusion_reason"] == "implementation_stack_not_tabnetics_core"
    )
    assert not (out_dir / "benchmark_df_fs_runs__all_jobs.csv").exists()


def test_aggregate_accepts_cross_artifact_consistent_canonical_contract(
    tmp_path: Path,
) -> None:
    root_out, plan, _run_dir, _execution = _write_canonical_aggregate_fixture(tmp_path)
    out_dir = tmp_path / "aggregate"

    aggregate(root_out, plan_path=plan, out_dir=out_dir)

    assert (out_dir / "benchmark_df_fs_runs__all_jobs.csv").exists()
    assert (out_dir / "benchmark_df_fs_summary__all_jobs.csv").exists()
    assert not (out_dir / "excluded_noncanonical_artifacts.csv").exists()


def test_aggregate_excludes_canonical_metadata_with_mismatched_run_row(
    tmp_path: Path,
) -> None:
    root_out, plan, run_dir, _execution = _write_canonical_aggregate_fixture(tmp_path)
    runs_path = run_dir / "df_fs_runs.csv"
    runs = pd.read_csv(runs_path)
    runs.loc[0, "execution_provenance_sha256"] = "mismatched"
    runs.to_csv(runs_path, index=False)
    out_dir = tmp_path / "aggregate"

    aggregate(root_out, plan_path=plan, out_dir=out_dir)

    excluded = pd.read_csv(out_dir / "excluded_noncanonical_artifacts.csv")
    assert (
        excluded.loc[0, "exclusion_reason"]
        == "df_fs_runs.csv_execution_field_mismatch:execution_provenance_sha256:row=0"
    )
    assert not (out_dir / "benchmark_df_fs_runs__all_jobs.csv").exists()


def test_aggregate_excludes_metric_csv_tampering_even_when_identity_columns_match(
    tmp_path: Path,
) -> None:
    root_out, plan, run_dir, _execution = _write_canonical_aggregate_fixture(tmp_path)
    runs_path = run_dir / "df_fs_runs.csv"
    runs = pd.read_csv(runs_path)
    runs.loc[0, "balanced_accuracy"] = 0.01
    runs.to_csv(runs_path, index=False)
    out_dir = tmp_path / "aggregate"

    aggregate(root_out, plan_path=plan, out_dir=out_dir)

    excluded = pd.read_csv(out_dir / "excluded_noncanonical_artifacts.csv")
    assert (
        excluded.loc[0, "exclusion_reason"]
        == "df_fs_artifact_provenance_artifact_sha256_mismatch:df_fs_runs.csv"
    )


def test_aggregate_compares_source_revision_identity_columns(
    tmp_path: Path,
) -> None:
    root_out, plan, run_dir, _execution = _write_canonical_aggregate_fixture(tmp_path)
    runs_path = run_dir / "df_fs_runs.csv"
    runs = pd.read_csv(runs_path)
    runs.loc[0, "source_revision_git_sha"] = "different-source-revision"
    runs.to_csv(runs_path, index=False)
    out_dir = tmp_path / "aggregate"

    aggregate(root_out, plan_path=plan, out_dir=out_dir)

    excluded = pd.read_csv(out_dir / "excluded_noncanonical_artifacts.csv")
    assert (
        excluded.loc[0, "exclusion_reason"]
        == "df_fs_runs.csv_execution_field_mismatch:source_revision_git_sha:row=0"
    )


def test_execution_contract_accepts_csv_missing_value_for_expected_blank_field() -> None:
    """An archive deployment has no Git SHA; pandas reloads its CSV cell as NaN."""

    assert _execution_value_matches(
        float("nan"), "", field="source_revision_git_sha"
    )
    assert not _execution_value_matches(
        float("nan"), "pinned-source-revision", field="source_revision_git_sha"
    )
    assert not _execution_value_matches(float("nan"), "", field="package_identity_sha256")


def test_aggregate_compares_loaded_package_module_closure_column(
    tmp_path: Path,
) -> None:
    root_out, plan, run_dir, _execution = _write_canonical_aggregate_fixture(tmp_path)
    runs_path = run_dir / "df_fs_runs.csv"
    runs = pd.read_csv(runs_path)
    runs.loc[0, "loaded_package_modules_sha256"] = "different-loaded-module-closure"
    runs.to_csv(runs_path, index=False)
    out_dir = tmp_path / "aggregate"

    aggregate(root_out, plan_path=plan, out_dir=out_dir)

    excluded = pd.read_csv(out_dir / "excluded_noncanonical_artifacts.csv")
    assert (
        excluded.loc[0, "exclusion_reason"]
        == "df_fs_runs.csv_execution_field_mismatch:loaded_package_modules_sha256:row=0"
    )


def test_aggregate_compares_per_row_materialized_input_identity(
    tmp_path: Path,
) -> None:
    root_out, plan, run_dir, _execution = _write_canonical_aggregate_fixture(tmp_path)
    runs_path = run_dir / "df_fs_runs.csv"
    runs = pd.read_csv(runs_path)
    runs.loc[0, "materialized_input_identity_sha256"] = "wrong-input"
    runs.to_csv(runs_path, index=False)
    out_dir = tmp_path / "aggregate"

    aggregate(root_out, plan_path=plan, out_dir=out_dir)

    excluded = pd.read_csv(out_dir / "excluded_noncanonical_artifacts.csv")
    assert (
        excluded.loc[0, "exclusion_reason"]
        == "df_fs_runs.csv_materialized_input_digest_mismatch:row=0"
    )


def test_aggregate_excludes_missing_or_mismatched_artifact_hash(
    tmp_path: Path,
) -> None:
    root_out, plan, run_dir, _execution = _write_canonical_aggregate_fixture(tmp_path)
    artifact_path = run_dir / "df_fs_artifact_provenance.json"
    artifact_provenance = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_provenance["artifacts"]["df_fs_runs.csv"]["sha256"] = ""
    artifact_path.write_text(json.dumps(artifact_provenance), encoding="utf-8")
    out_dir = tmp_path / "aggregate_missing_hash"

    aggregate(root_out, plan_path=plan, out_dir=out_dir)

    excluded = pd.read_csv(out_dir / "excluded_noncanonical_artifacts.csv")
    assert (
        excluded.loc[0, "exclusion_reason"]
        == "df_fs_artifact_provenance_artifact_sha256_missing:df_fs_runs.csv"
    )

    root_out, plan, run_dir, _execution = _write_canonical_aggregate_fixture(
        tmp_path / "mismatch"
    )
    artifact_path = run_dir / "df_fs_artifact_provenance.json"
    artifact_provenance = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_provenance["artifacts"]["df_fs_runs.csv"]["sha256"] = "0" * 64
    artifact_path.write_text(json.dumps(artifact_provenance), encoding="utf-8")
    out_dir = tmp_path / "aggregate_mismatched_hash"

    aggregate(root_out, plan_path=plan, out_dir=out_dir)

    excluded = pd.read_csv(out_dir / "excluded_noncanonical_artifacts.csv")
    assert (
        excluded.loc[0, "exclusion_reason"]
        == "df_fs_artifact_provenance_artifact_sha256_mismatch:df_fs_runs.csv"
    )


def test_aggregate_excludes_canonical_metadata_with_missing_summary_identity_field(
    tmp_path: Path,
) -> None:
    root_out, plan, run_dir, _execution = _write_canonical_aggregate_fixture(tmp_path)
    summary_path = run_dir / "df_fs_summary.csv"
    summary = pd.read_csv(summary_path)
    summary.drop(columns=["input_data_identity_sha256"]).to_csv(
        summary_path, index=False
    )
    out_dir = tmp_path / "aggregate"

    aggregate(root_out, plan_path=plan, out_dir=out_dir)

    excluded = pd.read_csv(out_dir / "excluded_noncanonical_artifacts.csv")
    assert (
        excluded.loc[0, "exclusion_reason"]
        == "df_fs_summary.csv_execution_field_missing:input_data_identity_sha256"
    )

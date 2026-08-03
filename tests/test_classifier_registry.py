import ast
import inspect
import textwrap
from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

import tabnetics.classification as classification
from tabnetics.classification.backends import (
    CLASSIFIER_COMPLEXITY_PRIOR,
    FLAML_NATIVE_BY_FAMILY,
    REGIME_POOLS,
    SklearnBackend,
    _flaml_custom_learner_specs,
)
from tabnetics.classification.registry import (
    CLASSIFIER_ALIASES,
    CLASSIFIER_COMPLEXITY_PRIORS,
    CLASSIFIER_NAMES,
    CLASSIFIER_SPECS,
    DEFAULT_CLASSIFIER_REGISTRY,
    REGIME_CLASSIFIER_POOLS,
    REGIME_HDLSS_EXTREME,
    REGIME_STANDARD,
    BuilderKind,
    CalibrationObservation,
    ClassifierCapabilityOverrides,
    ClassifierRegistry,
    ClassifierRuntimeFacts,
    ClassifierSpec,
    ClassifierTask,
    ProbabilityKind,
    ResourceClass,
    SupportLevel,
    TuningKind,
    UnknownClassifierError,
    canonical_classifier_name,
    get_classifier_spec,
    resolve_classifier_capabilities,
)


def _sklearn_builder_implementation_keys() -> list[str]:
    source = textwrap.dedent(inspect.getsource(SklearnBackend._build_candidates))
    tree = ast.parse(source)
    implementation_keys: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or not isinstance(node.ops[0], ast.In):
                continue
            if len(node.comparators) != 1:
                continue
            comparator = node.comparators[0]
            if not isinstance(comparator, ast.Name) or comparator.id not in {
                "direct_requested",
                "callback_requested",
            }:
                continue
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                implementation_keys.append(node.left.value)
    return implementation_keys


def _flaml_custom_learner_names() -> set[str]:
    source = textwrap.dedent(inspect.getsource(_flaml_custom_learner_specs))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name) or node.target.id != "specs":
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        return {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    raise AssertionError("Could not locate FLAML custom learner specification table.")


def _identity_fields(name: str) -> dict[str, object]:
    return {
        "builder_kind": BuilderKind.DIRECT,
        "builder_key": name,
        "tuning_kind": TuningKind.NO_TUNING,
        "tuning_key": None,
    }


def _minimal_spec(
    name: str, *, aliases: tuple[str, ...] = (), builder_key: str | None = None
) -> ClassifierSpec:
    return ClassifierSpec(
        name=name,
        aliases=aliases,
        complexity_prior=0.5,
        probability_kind=ProbabilityKind.NONE,
        relative_cost=1.0,
        **_identity_fields(builder_key or name),
    )


def test_registry_inventory_matches_every_constructible_backend_name():
    implementation_keys = _sklearn_builder_implementation_keys()
    registry_builder_keys = [spec.builder_key for spec in CLASSIFIER_SPECS.values()]

    assert len(implementation_keys) == 42
    assert Counter(implementation_keys) == Counter(registry_builder_keys)
    assert set(CLASSIFIER_NAMES) == set(CLASSIFIER_COMPLEXITY_PRIOR)
    assert DEFAULT_CLASSIFIER_REGISTRY.names(include_aliases=True) == tuple(
        CLASSIFIER_COMPLEXITY_PRIOR
    )
    assert len(CLASSIFIER_SPECS) == 42
    assert len(CLASSIFIER_ALIASES) == 1


def test_registry_builder_identities_match_current_construction_families():
    for name in CLASSIFIER_NAMES:
        spec = get_classifier_spec(name)
        assert spec.builder_key == canonical_classifier_name(name)

    callback_specs = {
        spec.name: spec.builder_key
        for spec in CLASSIFIER_SPECS.values()
        if spec.builder_kind is BuilderKind.CALLBACK
    }
    assert callback_specs == {
        "tabpfn": "tabpfn",
        "tabentics_diakrino": "tabentics_diakrino",
        "xgb": "xgb",
    }
    for name, builder_key in callback_specs.items():
        assert get_classifier_spec(name).required_builders == (builder_key,)


def test_registry_tuning_identities_match_flaml_sources_and_explicit_none():
    native = {
        name: get_classifier_spec(name).tuning_key
        for name in CLASSIFIER_NAMES
        if get_classifier_spec(name).tuning_kind is TuningKind.FLAML_NATIVE
    }
    custom = {
        str(spec.tuning_key)
        for spec in CLASSIFIER_SPECS.values()
        if spec.tuning_kind is TuningKind.FLAML_CUSTOM
    }
    no_tuning = {
        spec.name
        for spec in CLASSIFIER_SPECS.values()
        if spec.tuning_kind is TuningKind.NO_TUNING
    }

    assert native == FLAML_NATIVE_BY_FAMILY
    assert custom == _flaml_custom_learner_names()
    assert set(CLASSIFIER_SPECS) == set(native) | custom | no_tuning
    assert not (set(native) & custom or set(native) & no_tuning or custom & no_tuning)
    assert all(get_classifier_spec(name).tuning_key is None for name in no_tuning)


def test_registry_complexity_priors_match_backend_order_and_values():
    assert tuple(CLASSIFIER_COMPLEXITY_PRIORS.items()) == tuple(
        CLASSIFIER_COMPLEXITY_PRIOR.items()
    )
    for name, expected in CLASSIFIER_COMPLEXITY_PRIOR.items():
        assert get_classifier_spec(name).complexity_prior == pytest.approx(expected)


def test_registry_regime_pools_match_backend_order_and_membership():
    assert tuple(REGIME_CLASSIFIER_POOLS) == tuple(REGIME_POOLS)
    for regime, expected_pool in REGIME_POOLS.items():
        assert REGIME_CLASSIFIER_POOLS[regime] == expected_pool
        assert DEFAULT_CLASSIFIER_REGISTRY.names_for_regime(regime) == expected_pool
        for name in expected_pool:
            assert regime in get_classifier_spec(name).regimes

    assert DEFAULT_CLASSIFIER_REGISTRY.names_for_regime(REGIME_STANDARD) == (
        CLASSIFIER_NAMES
    )


def test_every_static_spec_has_typed_complete_conservative_metadata():
    for name in CLASSIFIER_NAMES:
        spec = get_classifier_spec(name)
        assert spec.task is ClassifierTask.CLASSIFICATION
        assert isinstance(spec.probability_kind, ProbabilityKind)
        assert isinstance(spec.builder_kind, BuilderKind)
        assert isinstance(spec.builder_key, str)
        assert isinstance(spec.tuning_kind, TuningKind)
        assert isinstance(spec.resource_class, ResourceClass)
        assert isinstance(spec.regimes, frozenset)
        assert REGIME_STANDARD in spec.regimes
        assert 0.0 <= spec.complexity_prior <= 1.0
        assert spec.relative_cost > 0.0
        assert isinstance(spec.dependencies, tuple)
        assert isinstance(spec.required_builders, tuple)
        for capability in (
            spec.multiclass,
            spec.estimator_sample_weight,
            spec.effective_sample_weight,
            spec.estimator_sparse_input,
            spec.effective_sparse_input,
            spec.nan_input,
            spec.categorical_input,
            spec.deterministic,
            spec.serialization,
        ):
            assert isinstance(capability, SupportLevel)


def test_dlda_alias_semantics_are_explicit_and_identity_preserving():
    dlda = get_classifier_spec("dlda")
    shrinkage = get_classifier_spec("shrinkage_lda")

    assert dlda is shrinkage
    assert dlda.name == "dlda"
    assert dlda.aliases == ("shrinkage_lda",)
    assert dlda.equivalence_group == "lda_shrink"
    assert CLASSIFIER_ALIASES == {"shrinkage_lda": "dlda"}
    assert canonical_classifier_name("shrinkage_lda") == "dlda"

    resolved = resolve_classifier_capabilities("shrinkage_lda")
    assert resolved.requested_name == "shrinkage_lda"
    assert resolved.canonical_name == "dlda"


def test_static_registry_objects_are_immutable():
    with pytest.raises(TypeError):
        CLASSIFIER_SPECS["new"] = get_classifier_spec("lr")  # type: ignore[index]
    with pytest.raises(TypeError):
        CLASSIFIER_ALIASES["new"] = "lr"  # type: ignore[index]
    with pytest.raises(TypeError):
        REGIME_CLASSIFIER_POOLS[REGIME_STANDARD] = ("lr",)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        get_classifier_spec("lr").relative_cost = 99.0  # type: ignore[misc]


@pytest.mark.parametrize("name", ["", "LR", " lr", "lr ", "missing"])
def test_unknown_or_noncanonical_names_fail_closed(name):
    with pytest.raises(UnknownClassifierError):
        get_classifier_spec(name)
    with pytest.raises(UnknownClassifierError):
        resolve_classifier_capabilities(name)


def test_non_string_classifier_name_fails_closed():
    with pytest.raises(TypeError):
        DEFAULT_CLASSIFIER_REGISTRY.get(1)  # type: ignore[arg-type]


def test_registry_rejects_canonical_and_alias_collisions():
    with pytest.raises(ValueError, match="Duplicate canonical"):
        ClassifierRegistry((_minimal_spec("one"), _minimal_spec("one")))

    with pytest.raises(ValueError, match="collides with a canonical"):
        ClassifierRegistry(
            (_minimal_spec("one", aliases=("two",)), _minimal_spec("two"))
        )

    with pytest.raises(ValueError, match="multiple families"):
        ClassifierRegistry(
            (
                _minimal_spec("one", aliases=("shared",)),
                _minimal_spec("two", aliases=("shared",)),
            )
        )

    with pytest.raises(ValueError, match="cannot alias itself"):
        _minimal_spec("one", aliases=("one",))

    with pytest.raises(ValueError, match="duplicates builder identity"):
        ClassifierRegistry(
            (_minimal_spec("one"), _minimal_spec("two", builder_key="one"))
        )


def test_registry_rejects_unknown_or_inconsistent_pool_members():
    with pytest.raises(UnknownClassifierError):
        ClassifierRegistry(
            (_minimal_spec("one"),),
            regime_pools={REGIME_STANDARD: ("missing",)},
        )

    with pytest.raises(ValueError, match="does not declare that regime"):
        ClassifierRegistry(
            (_minimal_spec("one"),),
            regime_pools={REGIME_HDLSS_EXTREME: ("one",)},
        )


def test_classifier_spec_rejects_invalid_static_metadata():
    with pytest.raises(ValueError, match="complexity_prior"):
        ClassifierSpec(
            name="bad",
            complexity_prior=1.1,
            probability_kind=ProbabilityKind.NONE,
            relative_cost=1.0,
            **_identity_fields("bad"),
        )
    with pytest.raises(ValueError, match="relative_cost"):
        ClassifierSpec(
            name="bad",
            complexity_prior=0.5,
            probability_kind=ProbabilityKind.NONE,
            relative_cost=0.0,
            **_identity_fields("bad"),
        )
    with pytest.raises(ValueError, match="requires probability_config_key"):
        ClassifierSpec(
            name="bad",
            complexity_prior=0.5,
            probability_kind=ProbabilityKind.NONE,
            probability_when_enabled=ProbabilityKind.NATIVE,
            relative_cost=1.0,
            **_identity_fields("bad"),
        )
    with pytest.raises(TypeError, match="probability_kind"):
        ClassifierSpec(
            name="bad",
            complexity_prior=0.5,
            probability_kind="native",  # type: ignore[arg-type]
            relative_cost=1.0,
            **_identity_fields("bad"),
        )


@pytest.mark.parametrize("name", ["svm_linear", "svm_rbf"])
def test_svc_probability_kind_resolves_from_config_without_static_mutation(name):
    static = get_classifier_spec(name)
    assert static.probability_kind is ProbabilityKind.NONE

    disabled = resolve_classifier_capabilities(
        name, config={"enable_svc_probability": False}
    )
    enabled = resolve_classifier_capabilities(
        name, config={"enable_svc_probability": True}
    )

    assert disabled.probability_kind is ProbabilityKind.NONE
    assert disabled.probability_matrix_available is False
    assert disabled.genuine_probability_eligible is False
    assert disabled.calibrated_probability_eligible is False
    assert enabled.probability_kind is ProbabilityKind.CALIBRATED
    assert enabled.probability_matrix_available is True
    assert enabled.genuine_probability_eligible is True
    assert enabled.calibrated_probability_eligible is True
    assert static.probability_kind is ProbabilityKind.NONE


def test_vote_probability_resolves_only_for_soft_voting_config():
    hard = resolve_classifier_capabilities("vote_ensemble")
    soft = resolve_classifier_capabilities(
        "vote_ensemble", config={"enable_svc_probability": True}
    )

    assert hard.probability_kind is ProbabilityKind.NONE
    assert soft.probability_kind is ProbabilityKind.NATIVE


def test_probability_admission_separates_matrix_genuine_and_calibrated_lanes():
    score_derived = resolve_classifier_capabilities("dwd_classifier")
    native = resolve_classifier_capabilities("lr")
    hard_proxy = resolve_classifier_capabilities(
        "pls_da_classifier",
        overrides=ClassifierCapabilityOverrides(
            probability_kind=ProbabilityKind.HARD_LABEL_PROXY
        ),
    )

    assert score_derived.probability_matrix_available is True
    assert score_derived.genuine_probability_eligible is False
    assert score_derived.calibrated_probability_eligible is False
    assert native.probability_matrix_available is True
    assert native.genuine_probability_eligible is True
    assert native.calibrated_probability_eligible is False
    assert hard_proxy.probability_matrix_available is False
    assert hard_proxy.genuine_probability_eligible is False
    assert hard_proxy.calibrated_probability_eligible is False


def test_tabentics_diakrino_calibration_requires_positive_fitted_evidence():
    config_key = "tabentics_diakrino_calibrate_probabilities"
    disabled = resolve_classifier_capabilities(
        "tabentics_diakrino", config={config_key: False}
    )
    configured_unobserved = resolve_classifier_capabilities(
        "tabentics_diakrino", config={config_key: True}
    )
    skipped = resolve_classifier_capabilities(
        "tabentics_diakrino",
        config={config_key: True},
        overrides=ClassifierCapabilityOverrides(
            calibration_observation=CalibrationObservation.SKIPPED
        ),
    )
    failed = resolve_classifier_capabilities(
        "tabentics_diakrino",
        config={config_key: True},
        overrides=ClassifierCapabilityOverrides(
            calibration_observation=CalibrationObservation.FAILED
        ),
    )
    calibrated = resolve_classifier_capabilities(
        "tabentics_diakrino",
        config={config_key: True},
        overrides=ClassifierCapabilityOverrides(
            calibration_observation=CalibrationObservation.TEMPERATURE_HOLDOUT
        ),
    )

    assert disabled.calibration_requested is False
    assert disabled.calibration_observation is CalibrationObservation.DISABLED
    assert configured_unobserved.calibration_requested is True
    assert (
        configured_unobserved.calibration_observation
        is CalibrationObservation.UNOBSERVED
    )
    for unresolved in (disabled, configured_unobserved, skipped, failed):
        assert unresolved.probability_kind is ProbabilityKind.NATIVE
        assert unresolved.calibrated_probability_eligible is False
    assert skipped.calibration_observation is CalibrationObservation.SKIPPED
    assert failed.calibration_observation is CalibrationObservation.FAILED
    assert calibrated.calibration_requested is True
    assert (
        calibrated.calibration_observation
        is CalibrationObservation.TEMPERATURE_HOLDOUT
    )
    assert calibrated.probability_kind is ProbabilityKind.CALIBRATED
    assert calibrated.calibrated_probability_eligible is True
    assert get_classifier_spec("tabentics_diakrino").probability_kind is ProbabilityKind.NATIVE


def test_tabentics_diakrino_calibration_rejects_contradictory_observations():
    with pytest.raises(ValueError, match="Disabled calibration"):
        resolve_classifier_capabilities(
            "tabentics_diakrino",
            config={"tabentics_diakrino_calibrate_probabilities": False},
            overrides=ClassifierCapabilityOverrides(
                calibration_observation=CalibrationObservation.TEMPERATURE_HOLDOUT
            ),
        )
    with pytest.raises(ValueError, match="calibration-aware"):
        resolve_classifier_capabilities(
            "lr",
            overrides=ClassifierCapabilityOverrides(
                calibration_observation=CalibrationObservation.TEMPERATURE_HOLDOUT
            ),
        )
    with pytest.raises(ValueError, match="positive fitted evidence"):
        resolve_classifier_capabilities(
            "tabentics_diakrino",
            config={"tabentics_diakrino_calibrate_probabilities": True},
            overrides=ClassifierCapabilityOverrides(
                probability_kind=ProbabilityKind.CALIBRATED
            ),
        )


def test_invalid_config_and_environment_fact_types_fail_closed():
    with pytest.raises(TypeError, match="must be bool"):
        resolve_classifier_capabilities(
            "svm_rbf", config={"enable_svc_probability": 1}
        )
    with pytest.raises(TypeError, match="dependency_facts"):
        resolve_classifier_capabilities(
            "xgb", dependency_facts={"xgboost": "yes"}  # type: ignore[dict-item]
        )
    with pytest.raises(TypeError, match="builder_facts"):
        resolve_classifier_capabilities(
            "xgb", builder_facts={"xgb": 1}  # type: ignore[dict-item]
        )


def test_optional_dependency_and_builder_resolution_is_fail_closed():
    unresolved = resolve_classifier_capabilities("xgb")
    assert unresolved.dependency_status is SupportLevel.CONDITIONAL
    assert unresolved.builder_status is SupportLevel.CONDITIONAL
    assert unresolved.availability is SupportLevel.CONDITIONAL
    assert unresolved.is_available is False
    assert "dependency:xgboost:unresolved" in unresolved.availability_reasons
    assert "builder:xgb:unresolved" in unresolved.availability_reasons

    resolved = resolve_classifier_capabilities(
        "xgb",
        dependency_facts={"xgboost": True},
        builder_facts={"xgb": True},
    )
    assert resolved.availability is SupportLevel.SUPPORTED
    assert resolved.is_available is True
    assert resolved.availability_reasons == ()

    unavailable = resolve_classifier_capabilities(
        "xgb",
        dependency_facts={"xgboost": False},
        builder_facts={"xgb": True},
    )
    assert unavailable.availability is SupportLevel.UNSUPPORTED
    assert unavailable.is_available is False
    assert "dependency:xgboost:unavailable" in unavailable.availability_reasons


def test_runtime_requirements_use_effective_not_estimator_support():
    binary_bc = resolve_classifier_capabilities(
        "bc_svm_linear", runtime=ClassifierRuntimeFacts(n_classes=2)
    )
    multiclass_bc = resolve_classifier_capabilities(
        "bc_svm_linear", runtime=ClassifierRuntimeFacts(n_classes=3)
    )
    assert binary_bc.is_available is True
    assert multiclass_bc.availability is SupportLevel.UNSUPPORTED
    assert "multiclass:unsupported" in multiclass_bc.availability_reasons

    weighted_sparse_lr = resolve_classifier_capabilities(
        "lr",
        runtime=ClassifierRuntimeFacts(
            n_classes=3,
            sample_weight_requested=True,
            input_is_sparse=True,
        ),
    )
    assert weighted_sparse_lr.estimator_sample_weight is SupportLevel.SUPPORTED
    assert weighted_sparse_lr.estimator_sparse_input is SupportLevel.SUPPORTED
    assert weighted_sparse_lr.effective_sample_weight is SupportLevel.SUPPORTED
    assert weighted_sparse_lr.effective_sparse_input is SupportLevel.UNSUPPORTED
    assert weighted_sparse_lr.availability is SupportLevel.UNSUPPORTED
    assert "sample_weight:unsupported" not in weighted_sparse_lr.availability_reasons
    assert "sparse_input:unsupported" in weighted_sparse_lr.availability_reasons

    integrated_lr = resolve_classifier_capabilities(
        "lr",
        runtime=ClassifierRuntimeFacts(
            n_classes=3,
            sample_weight_requested=True,
            input_is_sparse=True,
        ),
        overrides=ClassifierCapabilityOverrides(
            effective_sample_weight=SupportLevel.SUPPORTED,
            effective_sparse_input=SupportLevel.SUPPORTED,
        ),
    )
    assert integrated_lr.is_available is True

    sparse_dlda = resolve_classifier_capabilities(
        "dlda", runtime=ClassifierRuntimeFacts(input_is_sparse=True)
    )
    assert sparse_dlda.availability is SupportLevel.UNSUPPORTED
    assert "sparse_input:unsupported" in sparse_dlda.availability_reasons


def test_conditional_input_support_requires_an_observed_instance_override():
    unresolved = resolve_classifier_capabilities(
        "catboost",
        runtime=ClassifierRuntimeFacts(input_has_categorical=True),
        dependency_facts={"catboost": True},
    )
    assert unresolved.availability is SupportLevel.CONDITIONAL
    assert unresolved.is_available is False

    proven = resolve_classifier_capabilities(
        "catboost",
        runtime=ClassifierRuntimeFacts(input_has_categorical=True),
        dependency_facts={"catboost": True},
        overrides=ClassifierCapabilityOverrides(
            categorical_input=SupportLevel.SUPPORTED
        ),
    )
    assert proven.is_available is True
    assert get_classifier_spec("catboost").categorical_input is SupportLevel.CONDITIONAL


def test_instance_probability_override_identifies_hard_label_proxy():
    static = get_classifier_spec("pls_da_classifier")
    resolved = resolve_classifier_capabilities(
        "pls_da_classifier",
        overrides=ClassifierCapabilityOverrides(
            probability_kind=ProbabilityKind.HARD_LABEL_PROXY
        ),
    )

    assert static.probability_kind is ProbabilityKind.NONE
    assert resolved.probability_kind is ProbabilityKind.HARD_LABEL_PROXY
    assert resolved.probability_matrix_available is False
    assert resolved.genuine_probability_eligible is False
    assert resolved.calibrated_probability_eligible is False
    assert get_classifier_spec("pls_da_classifier") is static


def test_gpu_requirement_resolves_from_host_facts_without_changing_spec():
    override = ClassifierCapabilityOverrides(
        resource_class=ResourceClass.GPU_REQUIRED
    )
    missing_gpu = resolve_classifier_capabilities(
        "lr",
        runtime=ClassifierRuntimeFacts(gpu_available=False),
        overrides=override,
    )
    available_gpu = resolve_classifier_capabilities(
        "lr",
        runtime=ClassifierRuntimeFacts(gpu_available=True),
        overrides=override,
    )

    assert missing_gpu.gpu_required is True
    assert missing_gpu.availability is SupportLevel.UNSUPPORTED
    assert "gpu:unavailable" in missing_gpu.availability_reasons
    assert available_gpu.is_available is True
    assert get_classifier_spec("lr").resource_class is ResourceClass.CPU_LIGHT


def test_registry_types_and_helpers_are_publicly_exported():
    assert classification.DEFAULT_CLASSIFIER_REGISTRY is DEFAULT_CLASSIFIER_REGISTRY
    assert classification.BuilderKind is BuilderKind
    assert classification.CalibrationObservation is CalibrationObservation
    assert classification.ClassifierSpec is ClassifierSpec
    assert classification.ProbabilityKind is ProbabilityKind
    assert classification.TuningKind is TuningKind
    assert classification.resolve_classifier_capabilities is (
        resolve_classifier_capabilities
    )

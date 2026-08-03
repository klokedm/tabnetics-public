"""Fitted classifier probability semantics and strict consumer admission.

The registry describes invariant and configured capabilities.  This module adds
the fitted layer: it inspects an already fitted estimator without selecting,
tuning, calibrating, or mutating it, and exposes one strict probability-matrix
extractor shared by downstream consumers.
"""

from __future__ import annotations

import importlib.metadata
import pickle
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .registry import (
    CalibrationObservation,
    ClassifierRuntimeFacts,
    ProbabilityKind,
    ResolvedClassifierCapabilities,
    ResourceClass,
    SupportLevel,
    get_classifier_spec,
    resolve_classifier_capabilities,
)


FITTED_CLASSIFIER_DESCRIPTOR_SCHEMA_VERSION = "1.0"


class ProbabilityRequirement(str, Enum):
    """Typed requirements used by probability-consuming operations."""

    MATRIX = "matrix"
    GENUINE = "genuine_probability"
    CALIBRATED = "calibrated_probability"


@dataclass(frozen=True, slots=True)
class ProbabilityAdmission:
    admitted: bool
    requirement: ProbabilityRequirement
    probability_kind: ProbabilityKind
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": bool(self.admitted),
            "requirement": self.requirement.value,
            "probability_kind": self.probability_kind.value,
            "reason": str(self.reason),
        }


@dataclass(frozen=True, slots=True)
class LabelValue:
    """JSON-safe label record that does not collapse differently typed labels."""

    value: Any
    python_type: str
    numpy_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _json_scalar(self.value),
            "python_type": str(self.python_type),
            "numpy_dtype": str(self.numpy_dtype),
        }


@dataclass(frozen=True, slots=True)
class DependencyVersion:
    name: str
    declared_status: str
    installed: bool | None
    version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "declared_status": str(self.declared_status),
            "installed": self.installed,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class FittedMemberRecord:
    outward_name: str
    canonical_name: str | None
    tuned_family: str | None
    tuned_model: str | None
    registry_tuning_key: str | None
    executed_tuning_identity: str | None
    selected_flaml_family: str | None
    duplicate_position: int
    weight: float | None
    estimator_type: str
    probability_kind: ProbabilityKind
    probability_source: str
    model_revision_source: str | None
    model_revision_value: Any
    requested_device: str | None
    observed_device: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "outward_name": str(self.outward_name),
            "canonical_name": self.canonical_name,
            "tuned_family": self.tuned_family,
            "tuned_model": self.tuned_model,
            "registry_tuning_key": self.registry_tuning_key,
            "executed_tuning_identity": self.executed_tuning_identity,
            "selected_flaml_family": self.selected_flaml_family,
            "duplicate_position": int(self.duplicate_position),
            "weight": self.weight,
            "estimator_type": str(self.estimator_type),
            "probability_kind": self.probability_kind.value,
            "probability_source": str(self.probability_source),
            "model_revision_source": self.model_revision_source,
            "model_revision_value": _json_scalar(self.model_revision_value),
            "requested_device": self.requested_device,
            "observed_device": self.observed_device,
            "reason": str(self.reason),
        }


@dataclass(frozen=True, slots=True)
class FittedClassifierDescriptor:
    """Immutable, JSON-safe observation of one fitted classifier graph."""

    schema_version: str
    backend: str
    requested_name: str
    outward_name: str
    canonical_name: str | None
    registry_anchor_name: str | None
    effective_model_name: str
    composite_identity: str | None
    registry_tuning_key: str | None
    executed_tuning_identity: str | None
    tuning_identity: str | None
    selected_flaml_family: str | None
    fallback_reason: str | None
    estimator_type: str
    estimator_library: str
    estimator_library_version: str | None
    wrapper_path: tuple[str, ...]
    terminal_estimator_type: str
    configured_probability_kind: ProbabilityKind
    fitted_probability_kind: ProbabilityKind
    probability_source: str
    calibration_method: str | None
    calibration_observation: str
    class_order: tuple[LabelValue, ...]
    probability_column_order: tuple[LabelValue, ...]
    class_alignment: str
    matrix_observation: str
    matrix_shape: tuple[int, int] | None
    matrix_finite: bool | None
    matrix_simplex: bool | None
    matrix_reason: str
    argmax_contract: str
    argmax_observation: str
    argmax_agreement_rate: float | None
    estimator_sample_weight: str
    effective_sample_weight: str
    sample_weight_requested: bool
    sample_weight_routed_observation: str
    dependency_status: str
    dependencies: tuple[DependencyVersion, ...]
    resource_class: str
    requested_device: str | None
    requested_device_aggregation: str
    observed_device: str | None
    serialization_declared: str
    clone_observation: str
    pickle_observation: str
    model_revision_source: str | None
    model_revision_value: Any
    members: tuple[FittedMemberRecord, ...]
    reasons: tuple[str, ...]

    @property
    def probability_matrix_available(self) -> bool:
        return check_probability_requirement(
            self.fitted_probability_kind, ProbabilityRequirement.MATRIX
        ).admitted and self.matrix_observation == "passed"

    @property
    def probability_matrix_kind_admissible(self) -> bool:
        """Whether the resolved kind can enter a matrix probe, not proof it passed."""

        return check_probability_requirement(
            self.fitted_probability_kind, ProbabilityRequirement.MATRIX
        ).admitted

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "backend": str(self.backend),
            "requested_name": str(self.requested_name),
            "outward_name": str(self.outward_name),
            "canonical_name": self.canonical_name,
            "registry_anchor_name": self.registry_anchor_name,
            "effective_model_name": str(self.effective_model_name),
            "composite_identity": self.composite_identity,
            "registry_tuning_key": self.registry_tuning_key,
            "executed_tuning_identity": self.executed_tuning_identity,
            "tuning_identity": self.tuning_identity,
            "selected_flaml_family": self.selected_flaml_family,
            "fallback_reason": self.fallback_reason,
            "estimator_type": str(self.estimator_type),
            "estimator_library": str(self.estimator_library),
            "estimator_library_version": self.estimator_library_version,
            "wrapper_path": list(self.wrapper_path),
            "terminal_estimator_type": str(self.terminal_estimator_type),
            "configured_probability_kind": self.configured_probability_kind.value,
            "fitted_probability_kind": self.fitted_probability_kind.value,
            "probability_source": str(self.probability_source),
            "calibration_method": self.calibration_method,
            "calibration_observation": str(self.calibration_observation),
            "class_order": [value.to_dict() for value in self.class_order],
            "probability_column_order": [
                value.to_dict() for value in self.probability_column_order
            ],
            "class_alignment": str(self.class_alignment),
            "matrix_observation": str(self.matrix_observation),
            "matrix_shape": (
                None if self.matrix_shape is None else list(self.matrix_shape)
            ),
            "matrix_finite": self.matrix_finite,
            "matrix_simplex": self.matrix_simplex,
            "matrix_reason": str(self.matrix_reason),
            "argmax_contract": str(self.argmax_contract),
            "argmax_observation": str(self.argmax_observation),
            "argmax_agreement_rate": self.argmax_agreement_rate,
            "estimator_sample_weight": str(self.estimator_sample_weight),
            "effective_sample_weight": str(self.effective_sample_weight),
            "sample_weight_requested": bool(self.sample_weight_requested),
            "sample_weight_routed_observation": str(
                self.sample_weight_routed_observation
            ),
            "dependency_status": str(self.dependency_status),
            "dependencies": [value.to_dict() for value in self.dependencies],
            "resource_class": str(self.resource_class),
            "requested_device": self.requested_device,
            "requested_device_aggregation": str(
                self.requested_device_aggregation
            ),
            "observed_device": self.observed_device,
            "serialization_declared": str(self.serialization_declared),
            "clone_observation": str(self.clone_observation),
            "pickle_observation": str(self.pickle_observation),
            "model_revision_source": self.model_revision_source,
            "model_revision_value": _json_scalar(self.model_revision_value),
            "members": [value.to_dict() for value in self.members],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ProbabilityMatrixResult:
    available: bool
    matrix: np.ndarray | None
    reason: str
    requirement: ProbabilityRequirement
    probability_kind: ProbabilityKind
    probability_source: str
    class_order: tuple[LabelValue, ...]
    aligned: bool

    def metadata(self, *, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_available": bool(self.available),
            f"{prefix}_reason": str(self.reason),
            f"{prefix}_requirement": self.requirement.value,
            f"{prefix}_probability_kind": self.probability_kind.value,
            f"{prefix}_probability_source": str(self.probability_source),
            f"{prefix}_class_order": [value.to_dict() for value in self.class_order],
            f"{prefix}_class_alignment": "aligned" if self.aligned else "failed",
        }


@dataclass(frozen=True, slots=True)
class _SemanticNode:
    probability_kind: ProbabilityKind
    source: str
    path: tuple[str, ...]
    terminal_type: str
    calibration_method: str | None = None
    calibration_observation: str = "not_applicable"
    argmax_contract: str = "required"
    members: tuple[FittedMemberRecord, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _CompositeSpec:
    name: str | None
    tuning_key: str | None
    dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CompositeConfigured:
    probability_kind: ProbabilityKind
    estimator_sample_weight: SupportLevel
    effective_sample_weight: SupportLevel
    dependency_status: SupportLevel
    resource_class: ResourceClass
    serialization: SupportLevel


def check_probability_requirement(
    probability_kind: ProbabilityKind,
    requirement: ProbabilityRequirement,
) -> ProbabilityAdmission:
    """Apply the central probability-kind admission table."""

    if not isinstance(probability_kind, ProbabilityKind):
        raise TypeError("probability_kind must be a ProbabilityKind")
    if not isinstance(requirement, ProbabilityRequirement):
        raise TypeError("requirement must be a ProbabilityRequirement")
    allowed = {
        ProbabilityRequirement.MATRIX: {
            ProbabilityKind.NATIVE,
            ProbabilityKind.CALIBRATED,
            ProbabilityKind.SCORE_DERIVED,
        },
        ProbabilityRequirement.GENUINE: {
            ProbabilityKind.NATIVE,
            ProbabilityKind.CALIBRATED,
        },
        ProbabilityRequirement.CALIBRATED: {ProbabilityKind.CALIBRATED},
    }[requirement]
    if probability_kind in allowed:
        return ProbabilityAdmission(
            True, requirement, probability_kind, "admitted"
        )
    return ProbabilityAdmission(
        False,
        requirement,
        probability_kind,
        f"probability_kind:{probability_kind.value}:does_not_satisfy:{requirement.value}",
    )


def extract_probability_matrix(
    model: Any,
    X: np.ndarray,
    descriptor: FittedClassifierDescriptor,
    *,
    requirement: ProbabilityRequirement = ProbabilityRequirement.MATRIX,
    target_classes: Sequence[Any] | None = None,
) -> ProbabilityMatrixResult:
    """Return a strictly validated and class-aligned probability-like matrix."""

    admission = check_probability_requirement(
        descriptor.fitted_probability_kind, requirement
    )
    descriptor_classes = _classes_from_model(model)
    class_records = _label_records(descriptor_classes)
    if not admission.admitted:
        return ProbabilityMatrixResult(
            False,
            None,
            admission.reason,
            requirement,
            descriptor.fitted_probability_kind,
            descriptor.probability_source,
            class_records,
            False,
        )
    if not descriptor.class_order:
        return _matrix_failure(
            descriptor, requirement, class_records, "descriptor:class_order_unavailable"
        )
    if tuple(descriptor.class_order) != class_records:
        return _matrix_failure(
            descriptor,
            requirement,
            class_records,
            "descriptor:model_class_order_mismatch",
        )
    method = getattr(model, "predict_proba", None)
    if not callable(method):
        return _matrix_failure(
            descriptor, requirement, class_records, "predict_proba:unavailable"
        )
    try:
        raw = np.asarray(method(np.asarray(X, dtype=float)), dtype=float)
    except Exception as exc:
        return _matrix_failure(
            descriptor,
            requirement,
            class_records,
            f"predict_proba:raised:{type(exc).__name__}",
        )
    if raw.ndim != 2:
        return _matrix_failure(
            descriptor, requirement, class_records, "matrix:invalid_rank"
        )
    if raw.shape[0] != int(np.asarray(X).shape[0]):
        return _matrix_failure(
            descriptor, requirement, class_records, "matrix:row_count_mismatch"
        )
    if descriptor_classes.size < 2:
        return _matrix_failure(
            descriptor, requirement, class_records, "classes:unavailable"
        )
    if raw.shape[1] != descriptor_classes.size:
        return _matrix_failure(
            descriptor, requirement, class_records, "matrix:column_count_mismatch"
        )
    if _has_duplicate_labels(descriptor_classes):
        return _matrix_failure(
            descriptor, requirement, class_records, "classes:duplicate_columns"
        )

    matrix = np.asarray(raw, dtype=float)
    aligned_classes = descriptor_classes
    aligned = True
    if target_classes is not None:
        target_values = list(target_classes)
        target = np.empty(len(target_values), dtype=object)
        target[:] = target_values
        if _has_duplicate_labels(target):
            return _matrix_failure(
                descriptor, requirement, class_records, "target_classes:duplicate"
            )
        source_keys = [_label_key(value) for value in descriptor_classes]
        target_keys = [_label_key(value) for value in target]
        if len(source_keys) != len(target_keys) or set(source_keys) != set(target_keys):
            return _matrix_failure(
                descriptor, requirement, class_records, "classes:alignment_failed"
            )
        order = [source_keys.index(key) for key in target_keys]
        matrix = matrix[:, np.asarray(order, dtype=int)]
        aligned_classes = target

    if not np.all(np.isfinite(matrix)):
        return _matrix_failure(
            descriptor, requirement, class_records, "matrix:nonfinite"
        )
    if np.any(matrix < 0.0):
        return _matrix_failure(
            descriptor, requirement, class_records, "matrix:negative"
        )
    row_sums = np.sum(matrix, axis=1)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 1e-12):
        return _matrix_failure(
            descriptor, requirement, class_records, "matrix:invalid_row_sum"
        )
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        return _matrix_failure(
            descriptor, requirement, class_records, "matrix:not_simplex"
        )
    matrix = matrix / row_sums[:, None]
    return ProbabilityMatrixResult(
        True,
        matrix,
        "ok",
        requirement,
        descriptor.fitted_probability_kind,
        descriptor.probability_source,
        _label_records(aligned_classes),
        aligned,
    )


def inspect_fitted_classifier(
    model: Any,
    *,
    canonical_name: str | None,
    backend: str,
    requested_name: str | None = None,
    outward_name: str | None = None,
    effective_model_name: str | None = None,
    configured: ResolvedClassifierCapabilities | None = None,
    config: Mapping[str, Any] | None = None,
    selection_identity: Mapping[str, Any] | None = None,
    registry_anchor_name: str | None = None,
    probe_X: np.ndarray | None = None,
    sample_weight_requested: bool = False,
    sample_weight_routed_observation: str = "not_requested",
    requested_device: str | None = None,
    observe_pickle: bool = False,
) -> FittedClassifierDescriptor:
    """Inspect an already fitted classifier without changing fitted state."""

    identity = dict(selection_identity or {})
    if canonical_name is None:
        spec, composite_configured = _composite_views(
            tuple(identity.get("members") or ()), model=model
        )
        configured = configured or composite_configured  # type: ignore[assignment]
    else:
        spec = get_classifier_spec(canonical_name)
        configured = configured or resolve_classifier_capabilities(
            canonical_name,
            runtime=ClassifierRuntimeFacts(
                sample_weight_requested=bool(sample_weight_requested)
            ),
            config=config,
            dependency_facts={name: None for name in spec.dependencies},
            builder_facts={name: None for name in spec.required_builders},
        )
    assert configured is not None
    requested = str(
        requested_name
        or identity.get("requested_name")
        or canonical_name
        or identity.get("composite_identity")
        or "composite"
    )
    outward = str(outward_name or identity.get("outward_name") or requested)
    effective = str(
        effective_model_name
        or identity.get("effective_model_name")
        or outward
    )
    node = _inspect_semantic_node(
        model,
        configured_kind=configured.probability_kind,
        canonical_name=spec.name,
        selection_members=tuple(identity.get("members") or ()),
    )
    if isinstance(model, VotingClassifier) and probe_X is not None:
        node = _observe_voting_members(
            model,
            node,
            np.asarray(probe_X, dtype=float)[: min(16, len(probe_X))],
        )
    classes = _classes_from_model(model)
    class_records = _label_records(classes)
    matrix_observation = "not_applicable"
    matrix_shape: tuple[int, int] | None = None
    matrix_finite: bool | None = None
    matrix_simplex: bool | None = None
    matrix_reason = "probability_not_applicable"
    fitted_kind = node.probability_kind
    reasons = list(node.reasons)
    argmax_observation = "not_applicable"
    argmax_rate: float | None = None

    provisional = _descriptor_stub(
        model=model,
        spec=spec,
        configured=configured,
        node=node,
        fitted_kind=fitted_kind,
        backend=backend,
        requested=requested,
        outward=outward,
        effective=effective,
        identity=identity,
        class_records=class_records,
        matrix_observation="unobserved",
        matrix_shape=None,
        matrix_finite=None,
        matrix_simplex=None,
        matrix_reason="probe_not_provided",
        argmax_observation="unobserved",
        argmax_rate=None,
        sample_weight_requested=sample_weight_requested,
        sample_weight_routed_observation=sample_weight_routed_observation,
        requested_device=requested_device,
        registry_anchor_name=(
            registry_anchor_name
            or _optional_string(identity.get("registry_anchor_name"))
        ),
        clone_observation="unobserved",
        pickle_observation="unobserved",
        reasons=tuple(reasons),
    )

    if probe_X is None:
        if check_probability_requirement(
            fitted_kind, ProbabilityRequirement.MATRIX
        ).admitted:
            matrix_observation = "unobserved"
            matrix_reason = "probe_not_provided"
            argmax_observation = "unobserved"
    elif check_probability_requirement(
        fitted_kind, ProbabilityRequirement.MATRIX
    ).admitted:
        probe = np.asarray(probe_X, dtype=float)
        result = extract_probability_matrix(
            model,
            probe,
            provisional,
            requirement=ProbabilityRequirement.MATRIX,
            target_classes=classes,
        )
        if result.available and result.matrix is not None:
            matrix_observation = "passed"
            matrix_shape = tuple(int(value) for value in result.matrix.shape)
            matrix_finite = True
            matrix_simplex = True
            matrix_reason = "ok"
            argmax_observation, argmax_rate, argmax_reason = _observe_argmax(
                model, probe, result.matrix, classes, node.argmax_contract
            )
            if argmax_reason:
                reasons.append(argmax_reason)
                if node.argmax_contract == "required":
                    matrix_observation = "failed"
                    matrix_reason = str(argmax_reason)
                    fitted_kind = ProbabilityKind.UNKNOWN
        else:
            matrix_observation = "failed"
            matrix_reason = str(result.reason)
            matrix_finite = False if result.reason == "matrix:nonfinite" else None
            matrix_simplex = False if result.reason.startswith("matrix:") else None
            reasons.append(str(result.reason))
            fitted_kind = ProbabilityKind.UNKNOWN
            argmax_observation = "not_applicable"

    clone_observation = "passed"
    try:
        clone(model)
    except Exception as exc:
        clone_observation = f"failed:{type(exc).__name__}"
        reasons.append(f"clone:{type(exc).__name__}")
    pickle_observation = "unobserved"
    if observe_pickle:
        try:
            pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_observation = "passed"
        except Exception as exc:
            pickle_observation = f"failed:{type(exc).__name__}"
            reasons.append(f"pickle:{type(exc).__name__}")

    return _descriptor_stub(
        model=model,
        spec=spec,
        configured=configured,
        node=node,
        fitted_kind=fitted_kind,
        backend=backend,
        requested=requested,
        outward=outward,
        effective=effective,
        identity=identity,
        class_records=class_records,
        matrix_observation=matrix_observation,
        matrix_shape=matrix_shape,
        matrix_finite=matrix_finite,
        matrix_simplex=matrix_simplex,
        matrix_reason=matrix_reason,
        argmax_observation=argmax_observation,
        argmax_rate=argmax_rate,
        sample_weight_requested=sample_weight_requested,
        sample_weight_routed_observation=sample_weight_routed_observation,
        requested_device=requested_device,
        registry_anchor_name=(
            registry_anchor_name
            or _optional_string(identity.get("registry_anchor_name"))
        ),
        clone_observation=clone_observation,
        pickle_observation=pickle_observation,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _descriptor_stub(
    *,
    model: Any,
    spec: Any,
    configured: Any,
    node: _SemanticNode,
    fitted_kind: ProbabilityKind,
    backend: str,
    requested: str,
    outward: str,
    effective: str,
    identity: Mapping[str, Any],
    class_records: tuple[LabelValue, ...],
    matrix_observation: str,
    matrix_shape: tuple[int, int] | None,
    matrix_finite: bool | None,
    matrix_simplex: bool | None,
    matrix_reason: str,
    argmax_observation: str,
    argmax_rate: float | None,
    sample_weight_requested: bool,
    sample_weight_routed_observation: str,
    requested_device: str | None,
    registry_anchor_name: str | None,
    clone_observation: str,
    pickle_observation: str,
    reasons: tuple[str, ...],
) -> FittedClassifierDescriptor:
    estimator_type = _qualified_type(model)
    library, version = _estimator_library_version(model)
    revision_source, revision_value = _model_revision(model)
    observed_device = _observed_device(model)
    member_requested_devices = tuple(
        member.requested_device
        for member in node.members
        if member.requested_device is not None
    )
    requested_device_aggregation = (
        "member_specific"
        if member_requested_devices
        else "direct"
        if requested_device is not None
        else "unavailable"
    )
    dependency_records = tuple(
        _dependency_version(name, configured.dependency_status.value)
        for name in spec.dependencies
    )
    return FittedClassifierDescriptor(
        schema_version=FITTED_CLASSIFIER_DESCRIPTOR_SCHEMA_VERSION,
        backend=str(backend),
        requested_name=str(requested),
        outward_name=str(outward),
        canonical_name=(None if spec.name is None else str(spec.name)),
        registry_anchor_name=_optional_string(registry_anchor_name),
        effective_model_name=str(effective),
        composite_identity=_optional_string(identity.get("composite_identity")),
        registry_tuning_key=_optional_string(
            identity.get("registry_tuning_key") or spec.tuning_key
        ),
        executed_tuning_identity=_optional_string(
            identity.get("executed_tuning_identity")
            or identity.get("tuning_identity")
        ),
        tuning_identity=_optional_string(
            identity.get("executed_tuning_identity")
            or identity.get("tuning_identity")
        ),
        selected_flaml_family=_optional_string(
            identity.get("selected_flaml_family")
        ),
        fallback_reason=_optional_string(identity.get("fallback_reason")),
        estimator_type=estimator_type,
        estimator_library=library,
        estimator_library_version=version,
        wrapper_path=node.path,
        terminal_estimator_type=node.terminal_type,
        configured_probability_kind=configured.probability_kind,
        fitted_probability_kind=fitted_kind,
        probability_source=node.source,
        calibration_method=node.calibration_method,
        calibration_observation=node.calibration_observation,
        class_order=class_records,
        probability_column_order=class_records,
        class_alignment=("aligned" if class_records else "unobserved"),
        matrix_observation=str(matrix_observation),
        matrix_shape=matrix_shape,
        matrix_finite=matrix_finite,
        matrix_simplex=matrix_simplex,
        matrix_reason=str(matrix_reason),
        argmax_contract=node.argmax_contract,
        argmax_observation=str(argmax_observation),
        argmax_agreement_rate=argmax_rate,
        estimator_sample_weight=configured.estimator_sample_weight.value,
        effective_sample_weight=configured.effective_sample_weight.value,
        sample_weight_requested=bool(sample_weight_requested),
        sample_weight_routed_observation=str(sample_weight_routed_observation),
        dependency_status=configured.dependency_status.value,
        dependencies=dependency_records,
        resource_class=configured.resource_class.value,
        requested_device=_optional_string(requested_device),
        requested_device_aggregation=requested_device_aggregation,
        observed_device=observed_device,
        serialization_declared=configured.serialization.value,
        clone_observation=str(clone_observation),
        pickle_observation=str(pickle_observation),
        model_revision_source=revision_source,
        model_revision_value=revision_value,
        members=node.members,
        reasons=reasons,
    )


def _inspect_semantic_node(
    model: Any,
    *,
    configured_kind: ProbabilityKind,
    canonical_name: str | None,
    selection_members: Sequence[Mapping[str, Any]],
) -> _SemanticNode:
    path_head = (_qualified_type(model),)
    if isinstance(model, Pipeline):
        if not model.steps:
            return _SemanticNode(
                ProbabilityKind.UNKNOWN,
                "pipeline_without_terminal",
                path_head,
                _qualified_type(model),
                reasons=("pipeline:missing_terminal",),
            )
        terminal = model.steps[-1][1]
        child = _inspect_semantic_node(
            terminal,
            configured_kind=configured_kind,
            canonical_name=canonical_name,
            selection_members=selection_members,
        )
        step_path = tuple(
            f"{name}:{_qualified_type(estimator)}"
            for name, estimator in model.steps
        )
        return _SemanticNode(
            child.probability_kind,
            child.source,
            (*path_head, *step_path, *child.path[1:]),
            child.terminal_type,
            child.calibration_method,
            child.calibration_observation,
            child.argmax_contract,
            child.members,
            child.reasons,
        )

    if isinstance(model, CalibratedClassifierCV):
        fitted = bool(getattr(model, "calibrated_classifiers_", None)) and bool(
            _classes_from_model(model).size
        )
        method = str(getattr(model, "method", "sigmoid") or "sigmoid")
        fitted_calibrators = list(
            getattr(model, "calibrated_classifiers_", ()) or ()
        )
        base = (
            getattr(fitted_calibrators[0], "estimator", None)
            if fitted_calibrators
            else None
        ) or getattr(model, "estimator", None)
        if type(base).__name__ == "FrozenEstimator":
            base = getattr(base, "estimator", base)
        child = (
            _inspect_semantic_node(
                base,
                configured_kind=configured_kind,
                canonical_name=canonical_name,
                selection_members=selection_members,
            )
            if base is not None
            else None
        )
        base_path = () if child is None else child.path
        return _SemanticNode(
            ProbabilityKind.CALIBRATED if fitted else ProbabilityKind.UNKNOWN,
            f"calibrated_classifier_cv:{method}" if fitted else "calibrator_unfitted",
            (*path_head, *base_path),
            (
                child.terminal_type
                if child is not None
                else _qualified_type(model)
            ),
            calibration_method=method,
            calibration_observation=(
                f"posthoc_{method}_holdout" if fitted else "unobserved"
            ),
            argmax_contract="required",
            members=() if child is None else child.members,
            reasons=(
                *(child.reasons if child is not None else ()),
                *(("calibrator:not_fitted",) if not fitted else ()),
            ),
        )

    if isinstance(model, SVC):
        if bool(getattr(model, "probability", False)):
            fitted = bool(_classes_from_model(model).size)
            return _SemanticNode(
                ProbabilityKind.CALIBRATED if fitted else ProbabilityKind.UNKNOWN,
                "svc_internal_platt" if fitted else "svc_probability_unfitted",
                path_head,
                _qualified_type(model),
                calibration_method="platt",
                calibration_observation=(
                    "svc_internal_platt" if fitted else "unobserved"
                ),
                argmax_contract="not_promised",
            )
        return _SemanticNode(
            ProbabilityKind.NONE,
            "decision_function",
            path_head,
            _qualified_type(model),
            argmax_contract="not_applicable",
        )

    if isinstance(model, VotingClassifier):
        voting = str(getattr(model, "voting", "hard") or "hard").lower()
        fitted_members = list(getattr(model, "estimators_", ()) or ())
        names = [name for name, _ in list(getattr(model, "estimators", ()) or ())]
        weights = list(getattr(model, "weights", ()) or ())
        member_records: list[FittedMemberRecord] = []
        member_failures: list[str] = []
        for position, estimator in enumerate(fitted_members):
            supplied = (
                dict(selection_members[position])
                if position < len(selection_members)
                else {}
            )
            member_kind = _configured_kind_from_member(supplied)
            child = _inspect_semantic_node(
                estimator,
                configured_kind=member_kind,
                canonical_name=str(supplied.get("canonical_name") or ""),
                selection_members=(),
            )
            admission = check_probability_requirement(
                child.probability_kind, ProbabilityRequirement.MATRIX
            )
            reason = (
                "not_required_hard_vote"
                if voting != "soft"
                else "ok"
                if admission.admitted
                else admission.reason
            )
            if voting == "soft" and not admission.admitted:
                member_failures.append(f"soft_vote:member:{position}:{reason}")
            revision_source, revision_value = _model_revision(estimator)
            member_records.append(
                FittedMemberRecord(
                    outward_name=str(
                        supplied.get("outward_name")
                        or (names[position] if position < len(names) else f"member_{position}")
                    ),
                    canonical_name=_optional_string(supplied.get("canonical_name")),
                    tuned_family=_optional_string(supplied.get("tuned_family")),
                    tuned_model=_optional_string(supplied.get("tuned_model")),
                    registry_tuning_key=_optional_string(
                        supplied.get("registry_tuning_key")
                    ),
                    executed_tuning_identity=_optional_string(
                        supplied.get("executed_tuning_identity")
                    ),
                    selected_flaml_family=_optional_string(
                        supplied.get("selected_flaml_family")
                    ),
                    duplicate_position=int(
                        supplied.get("duplicate_position", position) or 0
                    ),
                    weight=(
                        float(supplied["weight"])
                        if supplied.get("weight") is not None
                        else (
                            float(weights[position])
                            if position < len(weights)
                            else None
                        )
                    ),
                    estimator_type=_qualified_type(estimator),
                    probability_kind=child.probability_kind,
                    probability_source=child.source,
                    model_revision_source=revision_source,
                    model_revision_value=revision_value,
                    requested_device=_requested_device_from_estimator(
                        estimator,
                        supplied.get("requested_device"),
                    ),
                    observed_device=_observed_device(estimator),
                    reason=reason,
                )
            )
        if not fitted_members and voting == "soft":
            member_failures.append("soft_vote:members_unfitted")
        if voting != "soft":
            return _SemanticNode(
                ProbabilityKind.NONE,
                "hard_vote_fraction",
                path_head,
                _qualified_type(model),
                argmax_contract="not_applicable",
                members=tuple(member_records),
            )
        return _SemanticNode(
            (
                ProbabilityKind.SCORE_DERIVED
                if not member_failures
                else ProbabilityKind.UNKNOWN
            ),
            "soft_vote",
            path_head,
            _qualified_type(model),
            argmax_contract="required",
            members=tuple(member_records),
            reasons=tuple(member_failures),
        )

    protocol = _probability_protocol(model)
    if protocol:
        mode = str(protocol.get("mode") or "declared")
        if mode == "delegate":
            attr = str(protocol.get("estimator_attr") or "estimator")
            inner = getattr(model, attr, None)
            if inner is None:
                return _SemanticNode(
                    ProbabilityKind.UNKNOWN,
                    "delegate_missing",
                    path_head,
                    _qualified_type(model),
                    reasons=(f"protocol:missing_delegate:{attr}",),
                )
            child = _inspect_semantic_node(
                inner,
                configured_kind=configured_kind,
                canonical_name=canonical_name,
                selection_members=selection_members,
            )
            return _SemanticNode(
                child.probability_kind,
                child.source,
                (*path_head, *child.path),
                child.terminal_type,
                child.calibration_method,
                child.calibration_observation,
                str(protocol.get("argmax_contract") or child.argmax_contract),
                child.members,
                child.reasons,
            )
        kind_value = protocol.get("probability_kind")
        try:
            kind = ProbabilityKind(str(kind_value))
        except ValueError:
            kind = ProbabilityKind.UNKNOWN
        return _SemanticNode(
            kind,
            str(protocol.get("probability_source") or "protocol_declared"),
            path_head,
            _qualified_type(model),
            calibration_method=_optional_string(protocol.get("calibration_method")),
            calibration_observation=str(
                protocol.get("calibration_observation") or "not_applicable"
            ),
            argmax_contract=str(protocol.get("argmax_contract") or "required"),
        )

    if canonical_name == "tabentics_diakrino":
        calibration_meta = dict(getattr(model, "calibration_meta_", {}) or {})
        observation = str(
            calibration_meta.get("native_diakrino_probability_calibration") or "unobserved"
        )
        calibrated = observation == "temperature_holdout"
        return _SemanticNode(
            ProbabilityKind.CALIBRATED if calibrated else ProbabilityKind.NATIVE,
            "native_diakrino_temperature" if calibrated else "native_diakrino",
            path_head,
            _qualified_type(model),
            calibration_method="temperature" if calibrated else None,
            calibration_observation=observation,
            argmax_contract="required",
        )

    if not callable(getattr(model, "predict_proba", None)):
        return _SemanticNode(
            ProbabilityKind.NONE,
            (
                "decision_function"
                if callable(getattr(model, "decision_function", None))
                else "predict"
            ),
            path_head,
            _qualified_type(model),
            argmax_contract="not_applicable",
        )
    source = {
        ProbabilityKind.NATIVE: "predict_proba_native",
        ProbabilityKind.CALIBRATED: "predict_proba_calibrated",
        ProbabilityKind.SCORE_DERIVED: "predict_proba_score_derived",
        ProbabilityKind.HARD_LABEL_PROXY: "predict_proba_hard_label_proxy",
    }.get(configured_kind, "predict_proba_unknown")
    return _SemanticNode(
        configured_kind,
        source,
        path_head,
        _qualified_type(model),
        calibration_observation=(
            CalibrationObservation.NOT_APPLICABLE.value
            if configured_kind is not ProbabilityKind.CALIBRATED
            else CalibrationObservation.UNOBSERVED.value
        ),
        argmax_contract="required",
    )


def _observe_argmax(
    model: Any,
    X: np.ndarray,
    matrix: np.ndarray,
    classes: np.ndarray,
    contract: str,
) -> tuple[str, float | None, str | None]:
    if contract == "not_applicable":
        return "not_applicable", None, None
    try:
        pred = np.asarray(model.predict(np.asarray(X, dtype=float))).ravel()
    except Exception as exc:
        return f"failed:{type(exc).__name__}", None, f"argmax:predict:{type(exc).__name__}"
    if pred.size != matrix.shape[0]:
        return "failed", None, "argmax:prediction_size_mismatch"
    expected = np.asarray(classes)[np.argmax(matrix, axis=1)]
    pred_keys = [_label_key(value) for value in pred]
    expected_keys = [_label_key(value) for value in expected]
    known = {_label_key(value) for value in classes}
    if any(key not in known for key in pred_keys):
        return "failed", None, "argmax:unknown_predicted_label"
    agreement = float(
        np.mean(
            np.asarray(
                [left == right for left, right in zip(pred_keys, expected_keys)],
                dtype=float,
            )
        )
    )
    if agreement >= 1.0 - 1e-12:
        return "passed", agreement, None
    if contract == "not_promised":
        return "observed_not_promised", agreement, None
    return "failed", agreement, "argmax:agreement_mismatch"


def _observe_voting_members(
    model: VotingClassifier,
    node: _SemanticNode,
    probe_X: np.ndarray,
) -> _SemanticNode:
    if str(getattr(model, "voting", "hard") or "hard").lower() != "soft":
        return node
    estimators = list(getattr(model, "estimators_", ()) or ())
    observed: list[FittedMemberRecord] = []
    failures: list[str] = []
    for position, record in enumerate(node.members):
        if position >= len(estimators):
            reason = "member_estimator_missing"
        else:
            admission = check_probability_requirement(
                record.probability_kind, ProbabilityRequirement.MATRIX
            )
            if not admission.admitted:
                reason = admission.reason
            else:
                try:
                    estimator = estimators[position]
                    method = getattr(estimator, "predict_proba", None)
                    if not callable(method):
                        raise ValueError("predict_proba_unavailable")
                    matrix = np.asarray(method(probe_X), dtype=float)
                    classes = _classes_from_model(estimator)
                    if matrix.ndim != 2 or matrix.shape != (
                        probe_X.shape[0],
                        classes.size,
                    ):
                        raise ValueError("invalid_probability_shape")
                    if not np.all(np.isfinite(matrix)):
                        raise ValueError("nonfinite_probability")
                    if np.any(matrix < 0.0):
                        raise ValueError("negative_probability")
                    row_sums = np.sum(matrix, axis=1)
                    if np.any(row_sums <= 1e-12) or not np.allclose(
                        row_sums, 1.0, atol=1e-6, rtol=1e-6
                    ):
                        raise ValueError("invalid_probability_simplex")
                    reason = "observed_matrix_passed"
                except Exception as exc:
                    canonical_reasons = {
                        "predict_proba_unavailable",
                        "invalid_probability_shape",
                        "nonfinite_probability",
                        "negative_probability",
                        "invalid_probability_simplex",
                    }
                    candidate = (
                        str(exc.args[0])
                        if isinstance(exc, ValueError)
                        and exc.args
                        and isinstance(exc.args[0], str)
                        else ""
                    )
                    reason = (
                        candidate
                        if candidate in canonical_reasons
                        else f"member_probability:{type(exc).__name__}"
                    )
        if reason != "observed_matrix_passed":
            failures.append(f"soft_vote:member:{position}:{reason}")
        if position < len(estimators):
            revision_source, revision_value = _model_revision(
                estimators[position]
            )
            observed_device = _observed_device(estimators[position])
        else:
            revision_source = record.model_revision_source
            revision_value = record.model_revision_value
            observed_device = record.observed_device
        observed.append(
            replace(
                record,
                reason=reason,
                model_revision_source=revision_source,
                model_revision_value=revision_value,
                observed_device=observed_device,
            )
        )
    if len(estimators) != len(node.members):
        failures.append("soft_vote:member_count_mismatch")
    return replace(
        node,
        probability_kind=(
            ProbabilityKind.SCORE_DERIVED
            if not failures
            else ProbabilityKind.UNKNOWN
        ),
        members=tuple(observed),
        reasons=tuple(dict.fromkeys((*node.reasons, *failures))),
    )


def _matrix_failure(
    descriptor: FittedClassifierDescriptor,
    requirement: ProbabilityRequirement,
    class_order: tuple[LabelValue, ...],
    reason: str,
) -> ProbabilityMatrixResult:
    return ProbabilityMatrixResult(
        False,
        None,
        str(reason),
        requirement,
        descriptor.fitted_probability_kind,
        descriptor.probability_source,
        class_order,
        False,
    )


def _probability_protocol(model: Any) -> Mapping[str, Any]:
    protocol = getattr(model, "tabnetics_probability_protocol", None)
    if not callable(protocol):
        return {}
    try:
        value = protocol()
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _configured_kind_from_member(member: Mapping[str, Any]) -> ProbabilityKind:
    raw = member.get("configured_probability_kind") or member.get("probability_kind")
    try:
        return ProbabilityKind(str(raw))
    except ValueError:
        return ProbabilityKind.UNKNOWN


def _composite_views(
    members: Sequence[Mapping[str, Any]],
    *,
    model: Any,
) -> tuple[_CompositeSpec, _CompositeConfigured]:
    specs = []
    dependencies: list[str] = []
    for member in members:
        canonical = member.get("canonical_name")
        if canonical is None:
            continue
        try:
            spec = get_classifier_spec(str(canonical))
        except Exception:
            continue
        specs.append(spec)
        for dependency in spec.dependencies:
            if dependency not in dependencies:
                dependencies.append(str(dependency))

    voting = str(getattr(model, "voting", "") or "").lower()
    probability_kind = (
        ProbabilityKind.SCORE_DERIVED
        if voting == "soft"
        else ProbabilityKind.NONE
        if voting == "hard"
        else ProbabilityKind.UNKNOWN
    )
    resource_order = {
        ResourceClass.CPU_LIGHT: 0,
        ResourceClass.CPU_STANDARD: 1,
        ResourceClass.CPU_HEAVY: 2,
        ResourceClass.GPU_PREFERRED: 3,
        ResourceClass.GPU_REQUIRED: 4,
    }
    resource_class = max(
        (spec.resource_class for spec in specs),
        key=lambda value: resource_order[value],
        default=ResourceClass.CPU_STANDARD,
    )
    return (
        _CompositeSpec(None, None, tuple(dependencies)),
        _CompositeConfigured(
            probability_kind=probability_kind,
            estimator_sample_weight=_aggregate_support(
                [spec.estimator_sample_weight for spec in specs]
            ),
            effective_sample_weight=_aggregate_support(
                [spec.effective_sample_weight for spec in specs]
            ),
            dependency_status=(
                SupportLevel.CONDITIONAL
                if dependencies
                else SupportLevel.SUPPORTED
            ),
            resource_class=resource_class,
            serialization=_aggregate_support(
                [spec.serialization for spec in specs]
            ),
        ),
    )


def _aggregate_support(values: Sequence[SupportLevel]) -> SupportLevel:
    levels = tuple(values)
    if not levels:
        return SupportLevel.UNKNOWN
    if any(value is SupportLevel.UNSUPPORTED for value in levels):
        return SupportLevel.UNSUPPORTED
    if any(value is SupportLevel.UNKNOWN for value in levels):
        return SupportLevel.UNKNOWN
    if any(value is SupportLevel.CONDITIONAL for value in levels):
        return SupportLevel.CONDITIONAL
    return SupportLevel.SUPPORTED


def _classes_from_model(model: Any) -> np.ndarray:
    classes = getattr(model, "classes_", None)
    if classes is None and isinstance(model, Pipeline) and model.steps:
        classes = getattr(model.steps[-1][1], "classes_", None)
    if classes is None:
        return np.asarray([], dtype=object)
    return np.asarray(classes).ravel()


def _label_records(values: Sequence[Any]) -> tuple[LabelValue, ...]:
    raw_values = (
        list(values.ravel()) if isinstance(values, np.ndarray) else list(values)
    )
    records: list[LabelValue] = []
    for value in raw_values:
        scalar = value.item() if isinstance(value, np.generic) else value
        records.append(
            LabelValue(
                value=_json_scalar(scalar),
                python_type=_qualified_type(scalar),
                numpy_dtype=str(np.asarray(value).dtype),
            )
        )
    return tuple(records)


def _label_key(value: Any) -> tuple[str, Any]:
    scalar = value.item() if isinstance(value, np.generic) else value
    try:
        hash(scalar)
        stable = scalar
    except TypeError:
        stable = str(scalar)
    return _qualified_type(scalar), stable


def _has_duplicate_labels(values: Sequence[Any]) -> bool:
    raw_values = (
        list(values.ravel()) if isinstance(values, np.ndarray) else list(values)
    )
    keys = [_label_key(value) for value in raw_values]
    return len(keys) != len(set(keys))


def _qualified_type(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    return str(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _dependency_version(name: str, declared_status: str) -> DependencyVersion:
    try:
        version = importlib.metadata.version(str(name))
        return DependencyVersion(str(name), str(declared_status), True, str(version))
    except importlib.metadata.PackageNotFoundError:
        return DependencyVersion(str(name), str(declared_status), False, None)
    except Exception:
        return DependencyVersion(str(name), str(declared_status), None, None)


def _estimator_library_version(model: Any) -> tuple[str, str | None]:
    root = type(model).__module__.split(".", 1)[0]
    package = {"sklearn": "scikit-learn"}.get(root, root)
    try:
        return package, str(importlib.metadata.version(package))
    except Exception:
        return package, None


def _model_revision(model: Any) -> tuple[str | None, Any]:
    keys = (
        "model_revision",
        "checkpoint_revision",
        "checkpoint_sha256",
        "checkpoint_id",
        "native_diakrino_checkpoint",
        "checkpoint",
    )
    for node in _known_wrapper_nodes(model):
        mappings = (
            ("native_diakrino_meta_", getattr(node, "native_diakrino_meta_", None)),
            ("checkpoint_metadata_", getattr(node, "checkpoint_metadata_", None)),
            ("model_meta_", getattr(node, "model_meta_", None)),
        )
        for attr_name, value in mappings:
            if not isinstance(value, Mapping):
                continue
            for key in keys:
                if key in value and value[key] is not None:
                    return f"{attr_name}.{key}", _json_scalar(value[key])
        for key in keys:
            value = getattr(node, key, None)
            if value is not None:
                return key, _json_scalar(value)
    return None, None


def _observed_device(model: Any) -> str | None:
    for node in _known_wrapper_nodes(model):
        native_meta = getattr(node, "native_diakrino_meta_", None)
        if isinstance(native_meta, Mapping):
            for key in ("native_diakrino_device", "observed_device"):
                value = native_meta.get(key)
                if value is not None:
                    return str(value)
        for key in ("device_", "inference_device_"):
            value = getattr(node, key, None)
            if value is None:
                continue
            if isinstance(value, str):
                return value
            device_type = getattr(value, "type", None)
            if isinstance(device_type, str):
                index = getattr(value, "index", None)
                return device_type if index is None else f"{device_type}:{int(index)}"
    return None


def _requested_device_from_estimator(
    model: Any,
    supplied: Any = None,
) -> str | None:
    if supplied is not None and str(supplied):
        return str(supplied)
    for node in _known_wrapper_nodes(model):
        get_params = getattr(node, "get_params", None)
        if not callable(get_params):
            continue
        try:
            params = dict(get_params(deep=False) or {})
        except Exception:
            continue
        for key in ("device", "inference_device"):
            value = params.get(key)
            if value is not None and str(value):
                return str(value)
    return None


def _known_wrapper_nodes(model: Any) -> tuple[Any, ...]:
    """Return a bounded outer-to-inner traversal of supported wrapper edges."""

    nodes: list[Any] = []
    pending: list[Any] = [model]
    seen: set[int] = set()
    while pending and len(nodes) < 64:
        node = pending.pop(0)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        nodes.append(node)

        if isinstance(node, Pipeline) and node.steps:
            pending.append(node.steps[-1][1])
            continue
        if isinstance(node, CalibratedClassifierCV):
            fitted = list(getattr(node, "calibrated_classifiers_", ()) or ())
            pending.extend(fitted)
            pending.append(getattr(node, "estimator", None))
            continue

        protocol = _probability_protocol(node)
        if str(protocol.get("mode") or "") == "delegate":
            attr = str(protocol.get("estimator_attr") or "estimator")
            pending.append(getattr(node, attr, None))
            continue

        if type(node).__name__ in {"FrozenEstimator", "_CalibratedClassifier"}:
            pending.append(getattr(node, "estimator", None))
    return tuple(nodes)


__all__ = (
    "FITTED_CLASSIFIER_DESCRIPTOR_SCHEMA_VERSION",
    "ProbabilityRequirement",
    "ProbabilityAdmission",
    "LabelValue",
    "DependencyVersion",
    "FittedMemberRecord",
    "FittedClassifierDescriptor",
    "ProbabilityMatrixResult",
    "check_probability_requirement",
    "extract_probability_matrix",
    "inspect_fitted_classifier",
)

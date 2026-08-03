"""Immutable classifier inventory and capability contracts.

This module is the authoritative source for classifier identity and capability
metadata. Runtime construction callables remain in
:mod:`tabnetics.classification.backends` and consume the immutable builder and
tuning identities declared here.

Static specifications intentionally describe candidates as directly constructed
by the backend.  Wrappers such as FLAML may change a resolved capability (for
example, by exposing hard-label one-hot values through ``predict_proba``), which
must be represented with :class:`ClassifierCapabilityOverrides` rather than by
mutating a static specification.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


REGIME_HDLSS_EXTREME = "hdlss_extreme"
REGIME_HDLSS_MODERATE = "hdlss_moderate"
REGIME_STANDARD = "standard"

_VALID_REGIMES = frozenset(
    {REGIME_HDLSS_EXTREME, REGIME_HDLSS_MODERATE, REGIME_STANDARD}
)
_VALID_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ProbabilityKind(str, Enum):
    """Provenance of a candidate's probability-like output."""

    NONE = "none"
    NATIVE = "native"
    CALIBRATED = "calibrated"
    SCORE_DERIVED = "score_derived"
    HARD_LABEL_PROXY = "hard_label_proxy"
    UNKNOWN = "unknown"


class SupportLevel(str, Enum):
    """Four-state support declaration used instead of lossy booleans."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


class ResourceClass(str, Enum):
    """Coarse execution class for admission and scheduling."""

    CPU_LIGHT = "cpu_light"
    CPU_STANDARD = "cpu_standard"
    CPU_HEAVY = "cpu_heavy"
    GPU_PREFERRED = "gpu_preferred"
    GPU_REQUIRED = "gpu_required"


class ClassifierTask(str, Enum):
    """Task family implemented by a registry entry."""

    CLASSIFICATION = "classification"


class BuilderKind(str, Enum):
    """How the active sklearn backend obtains a classifier instance."""

    DIRECT = "direct"
    CALLBACK = "callback"


class TuningKind(str, Enum):
    """Canonical tuning path currently associated with a classifier family."""

    NO_TUNING = "no_tuning"
    FLAML_NATIVE = "flaml_native"
    FLAML_CUSTOM = "flaml_custom"


class CalibrationObservation(str, Enum):
    """Observed fitted state for configuration-dependent probability calibration."""

    NOT_APPLICABLE = "not_applicable"
    DISABLED = "disabled"
    UNOBSERVED = "unobserved"
    SKIPPED = "skipped"
    FAILED = "failed"
    TEMPERATURE_HOLDOUT = "temperature_holdout"


class UnknownClassifierError(KeyError):
    """Raised when a classifier name is not present in a registry."""


@dataclass(frozen=True, slots=True)
class ClassifierSpec:
    """Invariant metadata for one canonical classifier family."""

    name: str
    complexity_prior: float
    probability_kind: ProbabilityKind
    relative_cost: float
    builder_kind: BuilderKind
    builder_key: str
    tuning_kind: TuningKind
    tuning_key: str | None
    aliases: tuple[str, ...] = ()
    equivalence_group: str | None = None
    regimes: frozenset[str] = frozenset({REGIME_STANDARD})
    task: ClassifierTask = ClassifierTask.CLASSIFICATION
    multiclass: SupportLevel = SupportLevel.SUPPORTED
    estimator_sample_weight: SupportLevel = SupportLevel.UNSUPPORTED
    effective_sample_weight: SupportLevel = SupportLevel.UNSUPPORTED
    estimator_sparse_input: SupportLevel = SupportLevel.UNSUPPORTED
    effective_sparse_input: SupportLevel = SupportLevel.UNSUPPORTED
    nan_input: SupportLevel = SupportLevel.UNSUPPORTED
    categorical_input: SupportLevel = SupportLevel.UNSUPPORTED
    deterministic: SupportLevel = SupportLevel.SUPPORTED
    structured_resampling: SupportLevel = SupportLevel.SUPPORTED
    dependencies: tuple[str, ...] = ()
    required_builders: tuple[str, ...] = ()
    resource_class: ResourceClass = ResourceClass.CPU_STANDARD
    tree_model: bool = False
    serialization: SupportLevel = SupportLevel.SUPPORTED
    probability_config_key: str | None = None
    probability_when_enabled: ProbabilityKind | None = None
    calibration_config_key: str | None = None
    calibration_requested_by_default: bool | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, field_name="name")

        aliases = tuple(self.aliases)
        for alias in aliases:
            _validate_name(alias, field_name="alias")
        if self.name in aliases:
            raise ValueError(f"Classifier {self.name!r} cannot alias itself.")
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"Classifier {self.name!r} contains duplicate aliases.")
        object.__setattr__(self, "aliases", aliases)

        regimes = frozenset(self.regimes)
        unknown_regimes = regimes - _VALID_REGIMES
        if unknown_regimes:
            raise ValueError(
                f"Classifier {self.name!r} has unknown regimes: {sorted(unknown_regimes)!r}."
            )
        if not regimes:
            raise ValueError(f"Classifier {self.name!r} must support at least one regime.")
        object.__setattr__(self, "regimes", regimes)

        dependencies = _validated_requirement_names(
            self.dependencies, field_name="dependencies", classifier_name=self.name
        )
        builders = _validated_requirement_names(
            self.required_builders,
            field_name="required_builders",
            classifier_name=self.name,
        )
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "required_builders", builders)

        if not isinstance(self.probability_kind, ProbabilityKind):
            raise TypeError("probability_kind must be a ProbabilityKind.")
        if not isinstance(self.task, ClassifierTask):
            raise TypeError("task must be a ClassifierTask.")
        if not isinstance(self.builder_kind, BuilderKind):
            raise TypeError("builder_kind must be a BuilderKind.")
        _validate_name(self.builder_key, field_name="builder_key")
        if not isinstance(self.tuning_kind, TuningKind):
            raise TypeError("tuning_kind must be a TuningKind.")
        if self.tuning_kind is TuningKind.NO_TUNING:
            if self.tuning_key is not None:
                raise ValueError("NO_TUNING identities cannot define tuning_key.")
        elif self.tuning_key is None:
            raise ValueError("Tuned identities require tuning_key.")
        else:
            _validate_name(self.tuning_key, field_name="tuning_key")
        for field_name in (
            "multiclass",
            "estimator_sample_weight",
            "effective_sample_weight",
            "estimator_sparse_input",
            "effective_sparse_input",
            "nan_input",
            "categorical_input",
            "deterministic",
            "structured_resampling",
            "serialization",
        ):
            if not isinstance(getattr(self, field_name), SupportLevel):
                raise TypeError(f"{field_name} must be a SupportLevel.")
        if not isinstance(self.resource_class, ResourceClass):
            raise TypeError("resource_class must be a ResourceClass.")
        if type(self.tree_model) is not bool:
            raise TypeError("tree_model must be bool.")

        prior = float(self.complexity_prior)
        if not math.isfinite(prior) or not 0.0 <= prior <= 1.0:
            raise ValueError("complexity_prior must be finite and in [0, 1].")
        object.__setattr__(self, "complexity_prior", prior)

        cost = float(self.relative_cost)
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("relative_cost must be finite and greater than zero.")
        object.__setattr__(self, "relative_cost", cost)

        group = self.equivalence_group
        if group is not None:
            group = str(group)
            _validate_name(group, field_name="equivalence_group")
            object.__setattr__(self, "equivalence_group", group)

        config_key = self.probability_config_key
        configured_kind = self.probability_when_enabled
        if config_key is None and configured_kind is not None:
            raise ValueError(
                "probability_when_enabled requires probability_config_key."
            )
        if config_key is not None:
            _validate_name(config_key, field_name="probability_config_key")
            if not isinstance(configured_kind, ProbabilityKind):
                raise TypeError(
                    "probability_when_enabled must be a ProbabilityKind when a "
                    "probability_config_key is set."
                )

        calibration_key = self.calibration_config_key
        calibration_default = self.calibration_requested_by_default
        if calibration_key is None:
            if calibration_default is not None:
                raise ValueError(
                    "calibration_requested_by_default requires calibration_config_key."
                )
        else:
            _validate_name(calibration_key, field_name="calibration_config_key")
            if type(calibration_default) is not bool:
                raise TypeError(
                    "calibration_requested_by_default must be bool when a "
                    "calibration_config_key is set."
                )

    @property
    def gpu_required(self) -> bool:
        """Whether this static resource class has no supported CPU lane."""

        return self.resource_class is ResourceClass.GPU_REQUIRED


@dataclass(frozen=True, slots=True)
class ClassifierRuntimeFacts:
    """Per-dataset and per-host facts used during capability resolution."""

    n_classes: int | None = None
    sample_weight_requested: bool = False
    input_is_sparse: bool = False
    input_has_nan: bool = False
    input_has_categorical: bool = False
    structured_resampling_requested: bool = False
    gpu_available: bool | None = None

    def __post_init__(self) -> None:
        if self.n_classes is not None and int(self.n_classes) < 2:
            raise ValueError("n_classes must be at least 2 when provided.")
        if self.n_classes is not None:
            object.__setattr__(self, "n_classes", int(self.n_classes))
        for field_name in (
            "sample_weight_requested",
            "input_is_sparse",
            "input_has_nan",
            "input_has_categorical",
            "structured_resampling_requested",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool.")
        if self.gpu_available is not None and type(self.gpu_available) is not bool:
            raise TypeError("gpu_available must be bool or None.")


@dataclass(frozen=True, slots=True)
class ClassifierCapabilityOverrides:
    """Observed instance-level capabilities that supersede static metadata."""

    probability_kind: ProbabilityKind | None = None
    calibration_observation: CalibrationObservation | None = None
    multiclass: SupportLevel | None = None
    estimator_sample_weight: SupportLevel | None = None
    effective_sample_weight: SupportLevel | None = None
    estimator_sparse_input: SupportLevel | None = None
    effective_sparse_input: SupportLevel | None = None
    nan_input: SupportLevel | None = None
    categorical_input: SupportLevel | None = None
    deterministic: SupportLevel | None = None
    structured_resampling: SupportLevel | None = None
    serialization: SupportLevel | None = None
    resource_class: ResourceClass | None = None

    def __post_init__(self) -> None:
        if self.probability_kind is not None and not isinstance(
            self.probability_kind, ProbabilityKind
        ):
            raise TypeError("probability_kind override must be a ProbabilityKind.")
        if self.calibration_observation is not None and not isinstance(
            self.calibration_observation, CalibrationObservation
        ):
            raise TypeError(
                "calibration_observation override must be a CalibrationObservation."
            )
        for field_name in (
            "multiclass",
            "estimator_sample_weight",
            "effective_sample_weight",
            "estimator_sparse_input",
            "effective_sparse_input",
            "nan_input",
            "categorical_input",
            "deterministic",
            "structured_resampling",
            "serialization",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, SupportLevel):
                raise TypeError(f"{field_name} override must be a SupportLevel.")
        if self.resource_class is not None and not isinstance(
            self.resource_class, ResourceClass
        ):
            raise TypeError("resource_class override must be a ResourceClass.")


@dataclass(frozen=True, slots=True)
class ResolvedClassifierCapabilities:
    """Capabilities after applying config, environment, and instance facts."""

    requested_name: str
    canonical_name: str
    probability_kind: ProbabilityKind
    calibration_requested: bool | None
    calibration_observation: CalibrationObservation
    multiclass: SupportLevel
    estimator_sample_weight: SupportLevel
    effective_sample_weight: SupportLevel
    estimator_sparse_input: SupportLevel
    effective_sparse_input: SupportLevel
    nan_input: SupportLevel
    categorical_input: SupportLevel
    deterministic: SupportLevel
    structured_resampling: SupportLevel
    serialization: SupportLevel
    resource_class: ResourceClass
    dependency_status: SupportLevel
    builder_status: SupportLevel
    gpu_status: SupportLevel
    availability: SupportLevel
    availability_reasons: tuple[str, ...] = ()

    @property
    def is_available(self) -> bool:
        """Return true only for fully proven, admissible candidates."""

        return self.availability is SupportLevel.SUPPORTED

    @property
    def probability_matrix_available(self) -> bool:
        """Whether output provides a simplex matrix beyond hard-label proxies."""

        return self.probability_kind in {
            ProbabilityKind.NATIVE,
            ProbabilityKind.CALIBRATED,
            ProbabilityKind.SCORE_DERIVED,
        }

    @property
    def genuine_probability_eligible(self) -> bool:
        """Whether output is a native or explicitly calibrated probability."""

        return self.probability_kind in {
            ProbabilityKind.NATIVE,
            ProbabilityKind.CALIBRATED,
        }

    @property
    def calibrated_probability_eligible(self) -> bool:
        """Whether positive calibration evidence supports calibrated admission."""

        return self.probability_kind is ProbabilityKind.CALIBRATED

    @property
    def gpu_required(self) -> bool:
        return self.resource_class is ResourceClass.GPU_REQUIRED


@dataclass(frozen=True, slots=True)
class ClassifierRegistry:
    """Validated immutable lookup for canonical specifications and aliases."""

    specs: tuple[ClassifierSpec, ...]
    regime_pools: Mapping[str, Sequence[str]] = field(default_factory=dict)
    _canonical: Mapping[str, ClassifierSpec] = field(
        init=False, repr=False, compare=False
    )
    _aliases: Mapping[str, str] = field(init=False, repr=False, compare=False)
    _all_names: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        specs = tuple(self.specs)
        if not specs:
            raise ValueError("ClassifierRegistry requires at least one specification.")
        canonical: dict[str, ClassifierSpec] = {}
        builder_identities: dict[tuple[BuilderKind, str], str] = {}
        for spec in specs:
            if not isinstance(spec, ClassifierSpec):
                raise TypeError("ClassifierRegistry entries must be ClassifierSpec objects.")
            if spec.name in canonical:
                raise ValueError(f"Duplicate canonical classifier name: {spec.name!r}.")
            builder_identity = (spec.builder_kind, spec.builder_key)
            existing_builder = builder_identities.get(builder_identity)
            if existing_builder is not None:
                raise ValueError(
                    f"Classifier {spec.name!r} duplicates builder identity "
                    f"owned by {existing_builder!r}."
                )
            canonical[spec.name] = spec
            builder_identities[builder_identity] = spec.name

        aliases: dict[str, str] = {}
        all_names: list[str] = []
        for spec in specs:
            all_names.append(spec.name)
            for alias in spec.aliases:
                if alias in canonical:
                    raise ValueError(
                        f"Classifier alias {alias!r} collides with a canonical name."
                    )
                if alias in aliases:
                    raise ValueError(
                        f"Classifier alias {alias!r} is assigned to multiple families."
                    )
                aliases[alias] = spec.name
                all_names.append(alias)

        normalized_pools: dict[str, tuple[str, ...]] = {}
        for raw_regime, raw_names in dict(self.regime_pools).items():
            regime = str(raw_regime)
            if regime not in _VALID_REGIMES:
                raise ValueError(f"Unknown classifier regime: {regime!r}.")
            names = tuple(raw_names)
            if len(set(names)) != len(names):
                raise ValueError(f"Classifier pool {regime!r} contains duplicates.")
            for name in names:
                resolved_name = canonical.get(name)
                if resolved_name is None and name in aliases:
                    resolved_name = canonical[aliases[name]]
                if resolved_name is None:
                    raise UnknownClassifierError(name)
                if regime not in resolved_name.regimes:
                    raise ValueError(
                        f"Classifier {name!r} is in pool {regime!r} but its spec "
                        "does not declare that regime."
                    )
            normalized_pools[regime] = names

        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "regime_pools", MappingProxyType(normalized_pools))
        object.__setattr__(self, "_canonical", MappingProxyType(canonical))
        object.__setattr__(self, "_aliases", MappingProxyType(aliases))
        object.__setattr__(self, "_all_names", tuple(all_names))

    @property
    def canonical_specs(self) -> Mapping[str, ClassifierSpec]:
        return self._canonical

    @property
    def aliases(self) -> Mapping[str, str]:
        return self._aliases

    def canonical_name(self, name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("Classifier name must be a string.")
        if name in self._canonical:
            return name
        if name in self._aliases:
            return self._aliases[name]
        raise UnknownClassifierError(name)

    def get(self, name: str) -> ClassifierSpec:
        return self._canonical[self.canonical_name(name)]

    def names(self, *, include_aliases: bool = False) -> tuple[str, ...]:
        if include_aliases:
            return self._all_names
        return tuple(spec.name for spec in self.specs)

    def names_for_regime(
        self, regime: str, *, include_aliases: bool = True
    ) -> tuple[str, ...]:
        if regime not in _VALID_REGIMES:
            raise ValueError(f"Unknown classifier regime: {regime!r}.")
        if regime in self.regime_pools:
            names = tuple(self.regime_pools[regime])
            if include_aliases:
                return names
            return tuple(name for name in names if name in self._canonical)
        source = self._all_names if include_aliases else self.names()
        return tuple(name for name in source if regime in self.get(name).regimes)


def _validate_name(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not _VALID_NAME.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {_VALID_NAME.pattern!r}; got {value!r}."
        )


def _validated_requirement_names(
    values: Sequence[str], *, field_name: str, classifier_name: str
) -> tuple[str, ...]:
    normalized = tuple(values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"Classifier {classifier_name!r} contains duplicate {field_name}."
        )
    for value in normalized:
        _validate_name(value, field_name=field_name)
    return normalized


# These ordered values are intentionally duplicated from backends.py during the
# migration slice.  Tests compare both directions so additions cannot land in
# construction, pools, or priors without being represented here.
REGIME_CLASSIFIER_POOLS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        REGIME_HDLSS_EXTREME: (
            "lr",
            "elastic_net_lr",
            "svm_linear",
            "bc_svm_linear",
            "rp_ensemble",
            "dlda",
            "shrinkage_lda",
            "nsc",
            "pls_da_classifier",
            "nb",
            "tabentics_diakrino",
            "dbda",
            "gqda",
            "sglnn",
            "rff_lr",
            "near_subspace",
            "spatial_median_da",
            "copula_da",
            "cpda",
            "tabm",
            "realmlp",
            "tabm_official",
            "realmlp_td",
            "hdrda",
            "dwd_classifier",
            "spls_da_classifier",
            "ecoc_hdrda",
            "ecoc_dwd",
            "ecoc_svm_linear",
        ),
        REGIME_HDLSS_MODERATE: (
            "lr",
            "elastic_net_lr",
            "svm_linear",
            "bc_svm_linear",
            "rp_ensemble",
            "svm_rbf",
            "dlda",
            "shrinkage_lda",
            "nsc",
            "pls_da_classifier",
            "gpc",
            "nb",
            "knn",
            "vote_ensemble",
            "tabpfn",
            "tabentics_diakrino",
            "dbda",
            "gqda",
            "sglnn",
            "rff_lr",
            "near_subspace",
            "spatial_median_da",
            "copula_da",
            "cpda",
            "tabm",
            "realmlp",
            "tabm_official",
            "realmlp_td",
            "hdrda",
            "spls_da_classifier",
        ),
    }
)


CLASSIFIER_COMPLEXITY_PRIORS: Mapping[str, float] = MappingProxyType(
    {
        "lr": 1.00,
        "elastic_net_lr": 0.98,
        "elastic_net_path_lr": 0.97,
        "svm_linear": 0.96,
        "bc_svm_linear": 0.95,
        "dbda": 0.93,
        "rp_ensemble": 0.94,
        "dlda": 0.92,
        "shrinkage_lda": 0.92,
        "gqda": 0.91,
        "sglnn": 0.82,
        "rff_lr": 0.84,
        "near_subspace": 0.89,
        "spatial_median_da": 0.91,
        "copula_da": 0.87,
        "cpda": 0.85,
        "cada_tent1": 0.86,
        "cada_tent2": 0.84,
        "cada_hinge2": 0.87,
        "tabm": 0.78,
        "realmlp": 0.76,
        "tabm_official": 0.72,
        "realmlp_td": 0.70,
        "nsc": 0.90,
        "pls_da_classifier": 0.88,
        "nb": 0.86,
        "svm_rbf": 0.74,
        "gpc": 0.68,
        "knn": 0.62,
        "vote_ensemble": 0.58,
        "tabpfn": 0.54,
        "tabentics_diakrino": 0.52,
        "rf": 0.42,
        "extra_tree": 0.36,
        "xgb": 0.34,
        "lgbm": 0.34,
        "catboost": 0.34,
        "hdrda": 0.91,
        "dwd_classifier": 0.88,
        "spls_da_classifier": 0.89,
        "ecoc_hdrda": 0.83,
        "ecoc_dwd": 0.83,
        "ecoc_svm_linear": 0.83,
    }
)


_CALLBACK_BUILDER_NAMES = frozenset({"tabpfn", "tabentics_diakrino", "xgb"})
FLAML_NATIVE_TUNING_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "lr": "lrl2",
        "rf": "rf",
        "xgb": "xgboost",
        "lgbm": "lgbm",
        "extra_tree": "extra_tree",
        "catboost": "catboost",
    }
)
_FLAML_CUSTOM_TUNING_NAMES = frozenset(
    {
        "elastic_net_lr",
        "svm_linear",
        "svm_rbf",
        "knn",
        "nb",
        "dlda",
        "nsc",
        "pls_da_classifier",
        "bc_svm_linear",
        "sglnn",
        "rff_lr",
        "copula_da",
        "cpda",
        "hdrda",
        "dwd_classifier",
        "spls_da_classifier",
        "tabm",
        "realmlp",
        "rp_ensemble",
        "near_subspace",
    }
)


def _regimes_for(name: str, aliases: Sequence[str] = ()) -> frozenset[str]:
    names = {name, *aliases}
    regimes = {REGIME_STANDARD}
    for regime, pool in REGIME_CLASSIFIER_POOLS.items():
        if names.intersection(pool):
            regimes.add(regime)
    return frozenset(regimes)


def _spec(
    name: str,
    *,
    probability_kind: ProbabilityKind,
    relative_cost: float,
    aliases: tuple[str, ...] = (),
    equivalence_group: str | None = None,
    multiclass: SupportLevel = SupportLevel.SUPPORTED,
    estimator_sample_weight: SupportLevel = SupportLevel.UNSUPPORTED,
    effective_sample_weight: SupportLevel = SupportLevel.UNSUPPORTED,
    estimator_sparse_input: SupportLevel = SupportLevel.UNSUPPORTED,
    effective_sparse_input: SupportLevel = SupportLevel.UNSUPPORTED,
    nan_input: SupportLevel = SupportLevel.UNSUPPORTED,
    categorical_input: SupportLevel = SupportLevel.UNSUPPORTED,
    deterministic: SupportLevel = SupportLevel.SUPPORTED,
    structured_resampling: SupportLevel = SupportLevel.SUPPORTED,
    dependencies: tuple[str, ...] = (),
    required_builders: tuple[str, ...] = (),
    resource_class: ResourceClass = ResourceClass.CPU_STANDARD,
    tree_model: bool = False,
    serialization: SupportLevel = SupportLevel.SUPPORTED,
    probability_config_key: str | None = None,
    probability_when_enabled: ProbabilityKind | None = None,
    calibration_config_key: str | None = None,
    calibration_requested_by_default: bool | None = None,
) -> ClassifierSpec:
    if name in FLAML_NATIVE_TUNING_KEYS:
        tuning_kind = TuningKind.FLAML_NATIVE
        tuning_key = FLAML_NATIVE_TUNING_KEYS[name]
    elif name in _FLAML_CUSTOM_TUNING_NAMES:
        tuning_kind = TuningKind.FLAML_CUSTOM
        tuning_key = name
    else:
        tuning_kind = TuningKind.NO_TUNING
        tuning_key = None
    return ClassifierSpec(
        name=name,
        aliases=aliases,
        equivalence_group=equivalence_group,
        regimes=_regimes_for(name, aliases),
        complexity_prior=CLASSIFIER_COMPLEXITY_PRIORS[name],
        probability_kind=probability_kind,
        builder_kind=(
            BuilderKind.CALLBACK
            if name in _CALLBACK_BUILDER_NAMES
            else BuilderKind.DIRECT
        ),
        builder_key=name,
        tuning_kind=tuning_kind,
        tuning_key=tuning_key,
        multiclass=multiclass,
        estimator_sample_weight=estimator_sample_weight,
        effective_sample_weight=effective_sample_weight,
        estimator_sparse_input=estimator_sparse_input,
        effective_sparse_input=effective_sparse_input,
        nan_input=nan_input,
        categorical_input=categorical_input,
        deterministic=deterministic,
        structured_resampling=structured_resampling,
        dependencies=dependencies,
        required_builders=required_builders,
        resource_class=resource_class,
        tree_model=tree_model,
        serialization=serialization,
        relative_cost=relative_cost,
        probability_config_key=probability_config_key,
        probability_when_enabled=probability_when_enabled,
        calibration_config_key=calibration_config_key,
        calibration_requested_by_default=calibration_requested_by_default,
    )


_CLASSIFIER_SPECS = (
    _spec(
        "lr",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=0.2,
        estimator_sample_weight=SupportLevel.SUPPORTED,
        effective_sample_weight=SupportLevel.SUPPORTED,
        estimator_sparse_input=SupportLevel.SUPPORTED,
        resource_class=ResourceClass.CPU_LIGHT,
    ),
    _spec(
        "elastic_net_lr",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=0.5,
        estimator_sample_weight=SupportLevel.SUPPORTED,
        effective_sample_weight=SupportLevel.SUPPORTED,
        estimator_sparse_input=SupportLevel.SUPPORTED,
        resource_class=ResourceClass.CPU_LIGHT,
    ),
    _spec(
        "elastic_net_path_lr",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=1.2,
        estimator_sample_weight=SupportLevel.SUPPORTED,
        effective_sample_weight=SupportLevel.SUPPORTED,
        structured_resampling=SupportLevel.UNSUPPORTED,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "svm_linear",
        probability_kind=ProbabilityKind.NONE,
        probability_config_key="enable_svc_probability",
        probability_when_enabled=ProbabilityKind.CALIBRATED,
        relative_cost=0.3,
        estimator_sample_weight=SupportLevel.CONDITIONAL,
        structured_resampling=SupportLevel.CONDITIONAL,
        resource_class=ResourceClass.CPU_LIGHT,
    ),
    _spec(
        "bc_svm_linear",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=0.4,
        multiclass=SupportLevel.UNSUPPORTED,
        resource_class=ResourceClass.CPU_LIGHT,
    ),
    _spec(
        "dbda",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=0.2,
        resource_class=ResourceClass.CPU_LIGHT,
    ),
    _spec(
        "rp_ensemble",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=0.8,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "dlda",
        aliases=("shrinkage_lda",),
        equivalence_group="lda_shrink",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=0.3,
        resource_class=ResourceClass.CPU_LIGHT,
    ),
    _spec(
        "gqda",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=0.3,
        resource_class=ResourceClass.CPU_LIGHT,
    ),
    _spec(
        "sglnn",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=1.5,
        structured_resampling=SupportLevel.UNSUPPORTED,
        resource_class=ResourceClass.CPU_HEAVY,
    ),
    _spec(
        "rff_lr",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=0.4,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "near_subspace",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=0.2,
        resource_class=ResourceClass.CPU_LIGHT,
    ),
    _spec(
        "spatial_median_da",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=0.3,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "copula_da",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=0.3,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "cpda",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=0.8,
        structured_resampling=SupportLevel.UNSUPPORTED,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "cada_tent1",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=0.8,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "cada_tent2",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=1.0,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "cada_hinge2",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=1.0,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "tabm",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=2.0,
        structured_resampling=SupportLevel.UNKNOWN,
        resource_class=ResourceClass.CPU_HEAVY,
    ),
    _spec(
        "realmlp",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=2.0,
        structured_resampling=SupportLevel.UNKNOWN,
        resource_class=ResourceClass.CPU_HEAVY,
    ),
    _spec(
        "tabm_official",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=3.0,
        estimator_sample_weight=SupportLevel.UNKNOWN,
        estimator_sparse_input=SupportLevel.UNKNOWN,
        nan_input=SupportLevel.CONDITIONAL,
        deterministic=SupportLevel.CONDITIONAL,
        structured_resampling=SupportLevel.UNKNOWN,
        dependencies=("pytabkit",),
        resource_class=ResourceClass.CPU_HEAVY,
        serialization=SupportLevel.CONDITIONAL,
    ),
    _spec(
        "realmlp_td",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=3.0,
        estimator_sample_weight=SupportLevel.UNKNOWN,
        estimator_sparse_input=SupportLevel.UNKNOWN,
        nan_input=SupportLevel.CONDITIONAL,
        deterministic=SupportLevel.CONDITIONAL,
        structured_resampling=SupportLevel.UNKNOWN,
        dependencies=("pytabkit",),
        resource_class=ResourceClass.CPU_HEAVY,
        serialization=SupportLevel.CONDITIONAL,
    ),
    _spec(
        "nsc",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=0.1,
        resource_class=ResourceClass.CPU_LIGHT,
    ),
    _spec(
        "pls_da_classifier",
        probability_kind=ProbabilityKind.NONE,
        relative_cost=0.2,
        resource_class=ResourceClass.CPU_LIGHT,
    ),
    _spec(
        "nb",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=0.1,
        estimator_sample_weight=SupportLevel.CONDITIONAL,
        resource_class=ResourceClass.CPU_LIGHT,
    ),
    _spec(
        "svm_rbf",
        probability_kind=ProbabilityKind.NONE,
        probability_config_key="enable_svc_probability",
        probability_when_enabled=ProbabilityKind.CALIBRATED,
        relative_cost=0.5,
        estimator_sample_weight=SupportLevel.SUPPORTED,
        effective_sample_weight=SupportLevel.SUPPORTED,
        estimator_sparse_input=SupportLevel.SUPPORTED,
        structured_resampling=SupportLevel.CONDITIONAL,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "gpc",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=3.0,
        resource_class=ResourceClass.CPU_HEAVY,
    ),
    _spec(
        "knn",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=0.2,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "vote_ensemble",
        probability_kind=ProbabilityKind.NONE,
        probability_config_key="enable_svc_probability",
        probability_when_enabled=ProbabilityKind.NATIVE,
        relative_cost=0.8,
        estimator_sample_weight=SupportLevel.CONDITIONAL,
        structured_resampling=SupportLevel.CONDITIONAL,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "tabpfn",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=5.0,
        estimator_sample_weight=SupportLevel.UNKNOWN,
        estimator_sparse_input=SupportLevel.UNKNOWN,
        nan_input=SupportLevel.CONDITIONAL,
        deterministic=SupportLevel.CONDITIONAL,
        dependencies=("tabpfn",),
        required_builders=("tabpfn",),
        resource_class=ResourceClass.GPU_PREFERRED,
        serialization=SupportLevel.CONDITIONAL,
    ),
    _spec(
        "tabentics_diakrino",
        probability_kind=ProbabilityKind.NATIVE,
        calibration_config_key="tabentics_diakrino_calibrate_probabilities",
        calibration_requested_by_default=True,
        structured_resampling=SupportLevel.CONDITIONAL,
        relative_cost=5.0,
        nan_input=SupportLevel.UNKNOWN,
        deterministic=SupportLevel.CONDITIONAL,
        dependencies=("torch",),
        required_builders=("tabentics_diakrino",),
        resource_class=ResourceClass.GPU_PREFERRED,
        serialization=SupportLevel.CONDITIONAL,
    ),
    _spec(
        "rf",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=1.0,
        estimator_sample_weight=SupportLevel.SUPPORTED,
        effective_sample_weight=SupportLevel.SUPPORTED,
        estimator_sparse_input=SupportLevel.SUPPORTED,
        nan_input=SupportLevel.SUPPORTED,
        resource_class=ResourceClass.CPU_HEAVY,
        tree_model=True,
        structured_resampling=SupportLevel.CONDITIONAL,
    ),
    _spec(
        "extra_tree",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=1.0,
        estimator_sample_weight=SupportLevel.SUPPORTED,
        effective_sample_weight=SupportLevel.SUPPORTED,
        estimator_sparse_input=SupportLevel.SUPPORTED,
        nan_input=SupportLevel.SUPPORTED,
        resource_class=ResourceClass.CPU_HEAVY,
        tree_model=True,
        structured_resampling=SupportLevel.CONDITIONAL,
    ),
    _spec(
        "xgb",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=2.0,
        estimator_sample_weight=SupportLevel.SUPPORTED,
        estimator_sparse_input=SupportLevel.SUPPORTED,
        nan_input=SupportLevel.SUPPORTED,
        deterministic=SupportLevel.CONDITIONAL,
        dependencies=("xgboost",),
        required_builders=("xgb",),
        resource_class=ResourceClass.CPU_HEAVY,
        tree_model=True,
        structured_resampling=SupportLevel.CONDITIONAL,
        serialization=SupportLevel.CONDITIONAL,
    ),
    _spec(
        "lgbm",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=1.0,
        estimator_sample_weight=SupportLevel.SUPPORTED,
        estimator_sparse_input=SupportLevel.SUPPORTED,
        nan_input=SupportLevel.SUPPORTED,
        categorical_input=SupportLevel.CONDITIONAL,
        deterministic=SupportLevel.CONDITIONAL,
        dependencies=("lightgbm",),
        resource_class=ResourceClass.CPU_HEAVY,
        tree_model=True,
        structured_resampling=SupportLevel.CONDITIONAL,
        serialization=SupportLevel.CONDITIONAL,
    ),
    _spec(
        "catboost",
        probability_kind=ProbabilityKind.NATIVE,
        relative_cost=1.5,
        estimator_sample_weight=SupportLevel.SUPPORTED,
        estimator_sparse_input=SupportLevel.CONDITIONAL,
        nan_input=SupportLevel.SUPPORTED,
        categorical_input=SupportLevel.CONDITIONAL,
        deterministic=SupportLevel.CONDITIONAL,
        dependencies=("catboost",),
        resource_class=ResourceClass.CPU_HEAVY,
        tree_model=True,
        structured_resampling=SupportLevel.CONDITIONAL,
        serialization=SupportLevel.CONDITIONAL,
    ),
    _spec(
        "hdrda",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=0.3,
        resource_class=ResourceClass.CPU_STANDARD,
    ),
    _spec(
        "dwd_classifier",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=0.6,
        resource_class=ResourceClass.CPU_HEAVY,
    ),
    _spec(
        "spls_da_classifier",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=0.5,
        structured_resampling=SupportLevel.UNSUPPORTED,
        resource_class=ResourceClass.CPU_HEAVY,
    ),
    _spec(
        "ecoc_hdrda",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=0.8,
        resource_class=ResourceClass.CPU_HEAVY,
    ),
    _spec(
        "ecoc_dwd",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=1.2,
        resource_class=ResourceClass.CPU_HEAVY,
    ),
    _spec(
        "ecoc_svm_linear",
        probability_kind=ProbabilityKind.SCORE_DERIVED,
        relative_cost=0.8,
        resource_class=ResourceClass.CPU_HEAVY,
    ),
)


DEFAULT_CLASSIFIER_REGISTRY = ClassifierRegistry(
    specs=_CLASSIFIER_SPECS,
    regime_pools=REGIME_CLASSIFIER_POOLS,
)
__tabnetics_execution_isolated_state__ = {
    "CLASSIFIER_COMPLEXITY_PRIORS": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "DEFAULT_CLASSIFIER_REGISTRY": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "FLAML_NATIVE_TUNING_KEYS": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "REGIME_CLASSIFIER_POOLS": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "_VALID_NAME": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}

CLASSIFIER_SPECS: Mapping[str, ClassifierSpec] = (
    DEFAULT_CLASSIFIER_REGISTRY.canonical_specs
)
CLASSIFIER_ALIASES: Mapping[str, str] = DEFAULT_CLASSIFIER_REGISTRY.aliases
CLASSIFIER_NAMES: tuple[str, ...] = DEFAULT_CLASSIFIER_REGISTRY.names(
    include_aliases=True
)


def get_classifier_spec(name: str) -> ClassifierSpec:
    """Return the canonical static specification for ``name`` or fail closed."""

    return DEFAULT_CLASSIFIER_REGISTRY.get(name)


def canonical_classifier_name(name: str) -> str:
    """Resolve a canonical name without accepting case or whitespace variants."""

    return DEFAULT_CLASSIFIER_REGISTRY.canonical_name(name)


def resolve_classifier_capabilities(
    name: str,
    *,
    runtime: ClassifierRuntimeFacts | None = None,
    config: Mapping[str, Any] | None = None,
    dependency_facts: Mapping[str, bool | None] | None = None,
    builder_facts: Mapping[str, bool | None] | None = None,
    overrides: ClassifierCapabilityOverrides | None = None,
    registry: ClassifierRegistry = DEFAULT_CLASSIFIER_REGISTRY,
) -> ResolvedClassifierCapabilities:
    """Resolve a classifier against dataset, config, and environment facts.

    Missing optional-dependency or builder facts remain ``CONDITIONAL`` and are
    therefore not reported as available.  Explicit false facts and incompatible
    dataset requirements resolve to ``UNSUPPORTED``.  The function never edits
    the registry or its static specifications.
    """

    requested_name = name
    spec = registry.get(name)
    runtime = runtime or ClassifierRuntimeFacts()
    if not isinstance(runtime, ClassifierRuntimeFacts):
        raise TypeError("runtime must be ClassifierRuntimeFacts or None.")
    overrides = overrides or ClassifierCapabilityOverrides()
    if not isinstance(overrides, ClassifierCapabilityOverrides):
        raise TypeError("overrides must be ClassifierCapabilityOverrides or None.")

    config_values = _normalized_config(config)
    dependency_values = _normalized_bool_facts(
        dependency_facts, field_name="dependency_facts"
    )
    builder_values = _normalized_bool_facts(
        builder_facts, field_name="builder_facts"
    )

    probability_kind = spec.probability_kind
    if spec.probability_config_key is not None:
        key = spec.probability_config_key
        if key in config_values:
            enabled = config_values[key]
            if type(enabled) is not bool:
                raise TypeError(f"Config value {key!r} must be bool.")
            if enabled:
                assert spec.probability_when_enabled is not None
                probability_kind = spec.probability_when_enabled

    calibration_requested: bool | None = None
    calibration_observation = CalibrationObservation.NOT_APPLICABLE
    if spec.calibration_config_key is not None:
        calibration_requested = spec.calibration_requested_by_default
        key = spec.calibration_config_key
        if key in config_values:
            configured = config_values[key]
            if type(configured) is not bool:
                raise TypeError(f"Config value {key!r} must be bool.")
            calibration_requested = configured
        assert calibration_requested is not None
        calibration_observation = (
            CalibrationObservation.UNOBSERVED
            if calibration_requested
            else CalibrationObservation.DISABLED
        )
        if overrides.calibration_observation is not None:
            calibration_observation = overrides.calibration_observation
        if calibration_requested:
            if calibration_observation not in {
                CalibrationObservation.UNOBSERVED,
                CalibrationObservation.SKIPPED,
                CalibrationObservation.FAILED,
                CalibrationObservation.TEMPERATURE_HOLDOUT,
            }:
                raise ValueError(
                    "Requested calibration requires an unobserved, skipped, failed, "
                    "or successful fitted observation."
                )
        elif calibration_observation is not CalibrationObservation.DISABLED:
            raise ValueError(
                "Disabled calibration cannot have a fitted calibration observation."
            )
        if calibration_observation is CalibrationObservation.TEMPERATURE_HOLDOUT:
            probability_kind = ProbabilityKind.CALIBRATED
    elif overrides.calibration_observation is not None:
        raise ValueError(
            "calibration_observation is only valid for a calibration-aware spec."
        )

    if (
        spec.calibration_config_key is not None
        and overrides.probability_kind is ProbabilityKind.CALIBRATED
        and calibration_observation is not CalibrationObservation.TEMPERATURE_HOLDOUT
    ):
        raise ValueError(
            "A calibration-aware classifier requires positive fitted evidence "
            "before a CALIBRATED probability override."
        )
    probability_kind = overrides.probability_kind or probability_kind
    multiclass = overrides.multiclass or spec.multiclass
    estimator_sample_weight = (
        overrides.estimator_sample_weight or spec.estimator_sample_weight
    )
    effective_sample_weight = (
        overrides.effective_sample_weight or spec.effective_sample_weight
    )
    estimator_sparse_input = (
        overrides.estimator_sparse_input or spec.estimator_sparse_input
    )
    effective_sparse_input = (
        overrides.effective_sparse_input or spec.effective_sparse_input
    )
    nan_input = overrides.nan_input or spec.nan_input
    categorical_input = overrides.categorical_input or spec.categorical_input
    deterministic = overrides.deterministic or spec.deterministic
    structured_resampling = (
        overrides.structured_resampling or spec.structured_resampling
    )
    serialization = overrides.serialization or spec.serialization
    resource_class = overrides.resource_class or spec.resource_class

    dependency_status, dependency_reasons = _resolve_named_requirements(
        spec.dependencies, dependency_values, requirement_kind="dependency"
    )
    builder_status, builder_reasons = _resolve_named_requirements(
        spec.required_builders, builder_values, requirement_kind="builder"
    )

    if resource_class is ResourceClass.GPU_REQUIRED:
        if runtime.gpu_available is True:
            gpu_status = SupportLevel.SUPPORTED
            gpu_reasons: tuple[str, ...] = ()
        elif runtime.gpu_available is False:
            gpu_status = SupportLevel.UNSUPPORTED
            gpu_reasons = ("gpu:unavailable",)
        else:
            gpu_status = SupportLevel.CONDITIONAL
            gpu_reasons = ("gpu:unresolved",)
    else:
        gpu_status = SupportLevel.SUPPORTED
        gpu_reasons = ()

    requested_statuses: list[SupportLevel] = []
    requested_reasons: list[str] = []
    if runtime.n_classes is not None and runtime.n_classes > 2:
        requested_statuses.append(multiclass)
        requested_reasons.extend(_support_reasons("multiclass", multiclass))
    if runtime.sample_weight_requested:
        requested_statuses.append(effective_sample_weight)
        requested_reasons.extend(
            _support_reasons("sample_weight", effective_sample_weight)
        )
    if runtime.input_is_sparse:
        requested_statuses.append(effective_sparse_input)
        requested_reasons.extend(
            _support_reasons("sparse_input", effective_sparse_input)
        )
    if runtime.input_has_nan:
        requested_statuses.append(nan_input)
        requested_reasons.extend(_support_reasons("nan_input", nan_input))
    if runtime.input_has_categorical:
        requested_statuses.append(categorical_input)
        requested_reasons.extend(
            _support_reasons("categorical_input", categorical_input)
        )
    if runtime.structured_resampling_requested:
        requested_statuses.append(structured_resampling)
        requested_reasons.extend(
            _support_reasons("structured_resampling", structured_resampling)
        )

    availability = _combine_support_levels(
        [dependency_status, builder_status, gpu_status, *requested_statuses]
    )
    reasons = (
        *dependency_reasons,
        *builder_reasons,
        *gpu_reasons,
        *requested_reasons,
    )
    return ResolvedClassifierCapabilities(
        requested_name=requested_name,
        canonical_name=spec.name,
        probability_kind=probability_kind,
        calibration_requested=calibration_requested,
        calibration_observation=calibration_observation,
        multiclass=multiclass,
        estimator_sample_weight=estimator_sample_weight,
        effective_sample_weight=effective_sample_weight,
        estimator_sparse_input=estimator_sparse_input,
        effective_sparse_input=effective_sparse_input,
        nan_input=nan_input,
        categorical_input=categorical_input,
        deterministic=deterministic,
        structured_resampling=structured_resampling,
        serialization=serialization,
        resource_class=resource_class,
        dependency_status=dependency_status,
        builder_status=builder_status,
        gpu_status=gpu_status,
        availability=availability,
        availability_reasons=tuple(reasons),
    )


def _normalized_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping or None.")
    normalized: dict[str, Any] = {}
    for key, value in config.items():
        if not isinstance(key, str):
            raise TypeError("config keys must be strings.")
        normalized[key] = value
    return normalized


def _normalized_bool_facts(
    facts: Mapping[str, bool | None] | None, *, field_name: str
) -> dict[str, bool | None]:
    if facts is None:
        return {}
    if not isinstance(facts, Mapping):
        raise TypeError(f"{field_name} must be a mapping or None.")
    normalized: dict[str, bool | None] = {}
    for key, value in facts.items():
        _validate_name(key, field_name=field_name)
        if value is not None and type(value) is not bool:
            raise TypeError(f"{field_name}[{key!r}] must be bool or None.")
        normalized[key] = value
    return normalized


def _resolve_named_requirements(
    requirements: Sequence[str],
    facts: Mapping[str, bool | None],
    *,
    requirement_kind: str,
) -> tuple[SupportLevel, tuple[str, ...]]:
    if not requirements:
        return SupportLevel.SUPPORTED, ()
    statuses: list[SupportLevel] = []
    reasons: list[str] = []
    for requirement in requirements:
        value = facts.get(requirement)
        if value is True:
            statuses.append(SupportLevel.SUPPORTED)
        elif value is False:
            statuses.append(SupportLevel.UNSUPPORTED)
            reasons.append(f"{requirement_kind}:{requirement}:unavailable")
        else:
            statuses.append(SupportLevel.CONDITIONAL)
            reasons.append(f"{requirement_kind}:{requirement}:unresolved")
    return _combine_support_levels(statuses), tuple(reasons)


def _support_reasons(name: str, level: SupportLevel) -> tuple[str, ...]:
    if level is SupportLevel.SUPPORTED:
        return ()
    return (f"{name}:{level.value}",)


def _combine_support_levels(levels: Sequence[SupportLevel]) -> SupportLevel:
    if any(level is SupportLevel.UNSUPPORTED for level in levels):
        return SupportLevel.UNSUPPORTED
    if any(level is SupportLevel.UNKNOWN for level in levels):
        return SupportLevel.UNKNOWN
    if any(level is SupportLevel.CONDITIONAL for level in levels):
        return SupportLevel.CONDITIONAL
    return SupportLevel.SUPPORTED


__all__ = [
    "BuilderKind",
    "CalibrationObservation",
    "CLASSIFIER_ALIASES",
    "CLASSIFIER_COMPLEXITY_PRIORS",
    "CLASSIFIER_NAMES",
    "CLASSIFIER_SPECS",
    "DEFAULT_CLASSIFIER_REGISTRY",
    "FLAML_NATIVE_TUNING_KEYS",
    "REGIME_CLASSIFIER_POOLS",
    "REGIME_HDLSS_EXTREME",
    "REGIME_HDLSS_MODERATE",
    "REGIME_STANDARD",
    "ClassifierCapabilityOverrides",
    "ClassifierRegistry",
    "ClassifierRuntimeFacts",
    "ClassifierSpec",
    "ClassifierTask",
    "ProbabilityKind",
    "ResolvedClassifierCapabilities",
    "ResourceClass",
    "SupportLevel",
    "TuningKind",
    "UnknownClassifierError",
    "canonical_classifier_name",
    "get_classifier_spec",
    "resolve_classifier_capabilities",
]

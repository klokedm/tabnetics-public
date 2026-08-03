"""Immutable typed-input schema and feature-lineage contracts.

The evaluation pipeline historically accepted only dense numeric ``ndarray``
inputs.  This module deliberately keeps data *out* of the value objects: a
schema describes a matrix, while a fold-local pipeline component owns fitted
encoding or imputation state.  Keeping these records immutable makes them safe
to fingerprint, persist in diagnostics, and pass between nested folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # pandas is a core dependency, but retain an explicit failure boundary.
    import pandas as pd
except Exception:  # pragma: no cover - exercised only in minimal installations
    pd = None  # type: ignore[assignment]

try:
    from scipy import sparse as sp
except Exception:  # pragma: no cover - scipy is a core dependency
    sp = None  # type: ignore[assignment]


class SchemaContractError(ValueError):
    """Raised when an input does not satisfy an immutable schema contract."""


class SchemaAlignmentMode(str, Enum):
    """Reviewed input-order policy for fitted estimator inference.

    ``STRICT`` is the default contract: names, order, and semantic dtype
    families must agree with the fitted schema. ``REORDER`` is intentionally
    narrow. It only permits a DataFrame containing the same uniquely named,
    type-compatible columns in a different order; it never fills, drops,
    renames, or coerces columns.
    """

    STRICT = "strict"
    REORDER = "reorder"


@dataclass(frozen=True, slots=True)
class InferenceSchemaCompatibilityReport:
    """Immutable record of an inference schema compatibility decision.

    Positional numeric inputs remain supported for legacy numeric schemas, but
    cannot establish name/role/dtype equivalence. Callers that persist or log a
    prediction should retain this record alongside it, particularly when
    ``alignment_applied`` is true or ``typed_semantics_verified`` is false.
    """

    schema_fingerprint: str
    alignment_mode: SchemaAlignmentMode
    input_kind: str
    typed_semantics_verified: bool
    alignment_applied: bool
    received_feature_names: tuple[str, ...] = tuple()
    output_feature_names: tuple[str, ...] = tuple()
    received_dtypes: tuple[tuple[str, str], ...] = tuple()

    def __post_init__(self) -> None:
        try:
            mode = (
                self.alignment_mode
                if isinstance(self.alignment_mode, SchemaAlignmentMode)
                else SchemaAlignmentMode(str(self.alignment_mode).strip().lower())
            )
        except ValueError as exc:
            raise SchemaContractError(
                f"Unknown inference schema alignment mode: {self.alignment_mode!r}."
            ) from exc
        object.__setattr__(self, "alignment_mode", mode)
        object.__setattr__(self, "schema_fingerprint", str(self.schema_fingerprint))
        object.__setattr__(self, "input_kind", _clean_text(self.input_kind, field_name="input kind"))
        object.__setattr__(
            self,
            "received_feature_names",
            tuple(str(value) for value in self.received_feature_names),
        )
        object.__setattr__(
            self,
            "output_feature_names",
            tuple(str(value) for value in self.output_feature_names),
        )
        object.__setattr__(
            self,
            "received_dtypes",
            tuple((str(name), str(dtype)) for name, dtype in self.received_dtypes),
        )

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-safe provenance record for an inference decision."""

        return {
            "schema_fingerprint": self.schema_fingerprint,
            "alignment_mode": self.alignment_mode.value,
            "input_kind": self.input_kind,
            "typed_semantics_verified": bool(self.typed_semantics_verified),
            "alignment_applied": bool(self.alignment_applied),
            "received_feature_names": list(self.received_feature_names),
            "output_feature_names": list(self.output_feature_names),
            "received_dtypes": {
                name: dtype for name, dtype in self.received_dtypes
            },
        }


class FeatureRole(str, Enum):
    """Semantics of an input or derived feature.

    ``GROUP`` and ``TIME`` retain their meaning in the schema as resampling
    metadata. The core predictor view excludes both, so they cannot silently
    become ordinary numeric predictors.
    """

    CONTINUOUS = "continuous"
    COUNT = "count"
    BINARY = "binary"
    ORDINAL = "ordinal"
    CATEGORICAL = "categorical"
    TEXT = "text"
    GROUP = "group"
    TIME = "time"
    IGNORED = "ignored"
    DERIVED = "derived"


def _clean_text(value: Any, *, field_name: str, allow_empty: bool = False) -> str:
    text = str(value if value is not None else "").strip()
    if not text and not allow_empty:
        raise SchemaContractError(f"{field_name} must be a non-empty string.")
    return text


def _canonical_json(value: Any) -> str:
    """Return stable JSON for a hashable metadata item."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError as exc:  # pragma: no cover - ``default=str`` is defensive
        raise SchemaContractError(f"Unable to serialize schema metadata: {exc}") from exc


def _metadata_items(values: Mapping[str, Any] | Iterable[tuple[str, Any]] | None) -> tuple[tuple[str, str], ...]:
    if values is None:
        return tuple()
    raw_items = values.items() if isinstance(values, Mapping) else values
    items: list[tuple[str, str]] = []
    for key, value in raw_items:
        text_key = _clean_text(key, field_name="metadata key")
        items.append((text_key, _canonical_json(value)))
    items.sort(key=lambda item: item[0])
    if len({key for key, _ in items}) != len(items):
        raise SchemaContractError("Schema metadata keys must be unique.")
    return tuple(items)


@dataclass(frozen=True, slots=True)
class FeatureAnnotation:
    """Immutable real-world feature-group/pathway membership.

    ``source_hash`` and ``version_hash`` make annotations attributable to an
    external resource.  Algorithmic correlation clusters are intentionally not
    represented here: callers should use a distinct diagnostic field for those
    proxy groups rather than presenting them as pathway annotation.
    """

    source: str
    version: str
    identifier: str
    kind: str = "group"
    source_hash: str = ""
    version_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _clean_text(self.source, field_name="annotation source"))
        object.__setattr__(self, "version", _clean_text(self.version, field_name="annotation version"))
        object.__setattr__(self, "identifier", _clean_text(self.identifier, field_name="annotation identifier"))
        object.__setattr__(self, "kind", _clean_text(self.kind, field_name="annotation kind"))
        object.__setattr__(self, "source_hash", str(self.source_hash or "").strip())
        object.__setattr__(self, "version_hash", str(self.version_hash or "").strip())

    def to_record(self) -> dict[str, str]:
        return {
            "source": self.source,
            "version": self.version,
            "identifier": self.identifier,
            "kind": self.kind,
            "source_hash": self.source_hash,
            "version_hash": self.version_hash,
        }


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One immutable input-column definition."""

    name: str
    role: FeatureRole
    dtype: str = "unknown"
    source_name: str | None = None
    annotations: tuple[FeatureAnnotation, ...] = tuple()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_text(self.name, field_name="feature name"))
        try:
            role = self.role if isinstance(self.role, FeatureRole) else FeatureRole(str(self.role))
        except ValueError as exc:
            raise SchemaContractError(f"Unknown feature role: {self.role!r}.") from exc
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "dtype", _clean_text(self.dtype, field_name="feature dtype", allow_empty=True) or "unknown")
        source_name = self.name if self.source_name is None else _clean_text(
            self.source_name, field_name="feature source_name"
        )
        object.__setattr__(self, "source_name", source_name)
        annotations = tuple(self.annotations or tuple())
        if not all(isinstance(value, FeatureAnnotation) for value in annotations):
            raise TypeError("Feature annotations must be FeatureAnnotation values.")
        # Stable ordering makes equality/fingerprints independent of caller order.
        annotations = tuple(sorted(annotations, key=lambda value: (
            value.kind,
            value.source,
            value.version,
            value.identifier,
            value.source_hash,
            value.version_hash,
        )))
        object.__setattr__(self, "annotations", annotations)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "dtype": self.dtype,
            "source_name": self.source_name,
            "annotations": [annotation.to_record() for annotation in self.annotations],
        }


def _dtype_family(dtype: Any) -> str:
    """Classify a pandas/numpy dtype without treating object as text or category.

    The classification is deliberately coarser than exact dtype equality: a
    fitted ``float64`` feature can safely receive ``float32`` values, while a
    categorical, object, integer, or boolean replacement is a semantic drift.
    """

    if pd is None:
        return "unknown"
    try:
        normalized = pd.api.types.pandas_dtype(dtype)
    except (TypeError, ValueError):
        return "unknown"
    if isinstance(normalized, pd.SparseDtype):
        return _dtype_family(normalized.subtype)
    if isinstance(normalized, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_bool_dtype(normalized):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(normalized):
        return "datetime"
    if pd.api.types.is_timedelta64_dtype(normalized):
        return "timedelta"
    if pd.api.types.is_integer_dtype(normalized):
        return "integer"
    if pd.api.types.is_float_dtype(normalized):
        return "floating"
    if pd.api.types.is_complex_dtype(normalized):
        return "complex"
    if pd.api.types.is_object_dtype(normalized):
        return "object"
    if pd.api.types.is_string_dtype(normalized):
        return "string"
    return "unknown"


def _feature_role_allows_dtype_family(feature: FeatureSpec, family: str) -> bool:
    """Return whether a dtype family can represent the declared feature role."""

    if feature.role is FeatureRole.CONTINUOUS:
        return family == "floating"
    if feature.role is FeatureRole.COUNT:
        return family == "integer"
    if feature.role is FeatureRole.BINARY:
        return family == "boolean"
    if feature.role is FeatureRole.ORDINAL:
        return family in {"integer", "floating", "categorical"}
    if feature.role is FeatureRole.CATEGORICAL:
        # Explicitly labelled integer codes are allowed, but a float column is
        # not silently reinterpreted as categorical at inference time.
        return family in {"categorical", "object", "string", "integer"}
    if feature.role is FeatureRole.TEXT:
        return family in {"object", "string"}
    if feature.role is FeatureRole.GROUP:
        return family in {"categorical", "object", "string", "integer"}
    if feature.role is FeatureRole.TIME:
        return family in {"datetime", "timedelta"}
    if feature.role is FeatureRole.DERIVED:
        return family in {"boolean", "integer", "floating"}
    # IGNORED fields remain schema-bound but need no predictor semantic claim.
    return family != "unknown"


def _inference_dtype_compatible(feature: FeatureSpec, observed_dtype: Any) -> tuple[bool, str, str]:
    """Compare an observed DataFrame dtype with a feature's strict contract."""

    expected_family = _dtype_family(feature.dtype)
    observed_family = _dtype_family(observed_dtype)
    if observed_family == "unknown":
        return False, expected_family, observed_family
    if expected_family != "unknown" and observed_family != expected_family:
        return False, expected_family, observed_family
    if not _feature_role_allows_dtype_family(feature, observed_family):
        return False, expected_family, observed_family
    return True, expected_family, observed_family


def _coerce_schema_alignment_mode(value: SchemaAlignmentMode | str) -> SchemaAlignmentMode:
    try:
        return (
            value
            if isinstance(value, SchemaAlignmentMode)
            else SchemaAlignmentMode(str(value).strip().lower())
        )
    except ValueError as exc:
        raise SchemaContractError(
            "alignment_mode must be 'strict' or the explicit reviewed mode 'reorder'."
        ) from exc


@dataclass(frozen=True, slots=True)
class FeatureLineage:
    """One output feature's immutable parent/transform record."""

    output_name: str
    operation: str
    input_names: tuple[str, ...]
    source_schema_hash: str = ""
    parameters: tuple[tuple[str, str], ...] = tuple()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_name", _clean_text(self.output_name, field_name="lineage output_name"))
        object.__setattr__(self, "operation", _clean_text(self.operation, field_name="lineage operation"))
        inputs = tuple(_clean_text(value, field_name="lineage input_name") for value in self.input_names)
        if not inputs:
            raise SchemaContractError("Feature lineage requires at least one input feature.")
        object.__setattr__(self, "input_names", inputs)
        object.__setattr__(self, "source_schema_hash", str(self.source_schema_hash or "").strip())
        object.__setattr__(self, "parameters", _metadata_items(self.parameters))

    @classmethod
    def from_parameters(
        cls,
        *,
        output_name: str,
        operation: str,
        input_names: Sequence[str],
        source_schema_hash: str = "",
        parameters: Mapping[str, Any] | None = None,
    ) -> "FeatureLineage":
        return cls(
            output_name=str(output_name),
            operation=str(operation),
            input_names=tuple(str(value) for value in input_names),
            source_schema_hash=str(source_schema_hash or ""),
            parameters={} if parameters is None else parameters,
        )

    @property
    def parameters_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                key: json.loads(value)
                for key, value in self.parameters
            }
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "output_name": self.output_name,
            "operation": self.operation,
            "input_names": list(self.input_names),
            "source_schema_hash": self.source_schema_hash,
            "parameters": dict(self.parameters_dict),
        }


def _infer_role(name: str, series: Any) -> FeatureRole:
    """Infer only safe dtype/name-level defaults; callers may override roles.

    No cardinality/value statistics are used here, so the schema itself cannot
    learn from held-out labels or category frequency.  Richer distinctions such
    as ordinal vs categorical should be supplied explicitly by the caller.
    """

    if pd is None:
        return FeatureRole.CONTINUOUS
    name_key = str(name).strip().lower()
    dtype = getattr(series, "dtype", None)
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        return FeatureRole.TIME
    if pd.api.types.is_bool_dtype(dtype):
        return FeatureRole.BINARY
    if isinstance(dtype, pd.CategoricalDtype):
        return FeatureRole.CATEGORICAL
    if pd.api.types.is_integer_dtype(dtype):
        return FeatureRole.COUNT
    if pd.api.types.is_float_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        return FeatureRole.CONTINUOUS
    if any(token in name_key for token in ("text", "comment", "review", "description", "body")):
        return FeatureRole.TEXT
    return FeatureRole.CATEGORICAL


def _coerce_annotations(values: Sequence[FeatureAnnotation | Mapping[str, Any]] | None) -> tuple[FeatureAnnotation, ...]:
    annotations: list[FeatureAnnotation] = []
    for value in values or tuple():
        if isinstance(value, FeatureAnnotation):
            annotations.append(value)
        elif isinstance(value, Mapping):
            annotations.append(FeatureAnnotation(**dict(value)))
        else:
            raise TypeError("Annotations must be FeatureAnnotation or mapping values.")
    return tuple(annotations)


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Frozen schema plus feature-level lineage for one matrix view."""

    features: tuple[FeatureSpec, ...]
    lineage: tuple[FeatureLineage, ...] = tuple()
    version: str = "1"
    metadata: tuple[tuple[str, str], ...] = tuple()

    def __post_init__(self) -> None:
        features = tuple(self.features or tuple())
        if not features:
            raise SchemaContractError("DatasetSchema requires at least one feature.")
        if not all(isinstance(value, FeatureSpec) for value in features):
            raise TypeError("DatasetSchema.features must contain FeatureSpec values.")
        names = tuple(value.name for value in features)
        if len(set(names)) != len(names):
            raise SchemaContractError("DatasetSchema feature names must be unique.")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "version", _clean_text(self.version, field_name="schema version"))
        object.__setattr__(self, "metadata", _metadata_items(self.metadata))
        lineage = tuple(self.lineage or tuple())
        if not lineage:
            lineage = tuple(
                FeatureLineage(
                    output_name=feature.name,
                    operation="identity",
                    input_names=(str(feature.source_name),),
                )
                for feature in features
            )
        if not all(isinstance(value, FeatureLineage) for value in lineage):
            raise TypeError("DatasetSchema.lineage must contain FeatureLineage values.")
        lineage_names = tuple(value.output_name for value in lineage)
        if len(set(lineage_names)) != len(lineage_names):
            raise SchemaContractError("DatasetSchema lineage output names must be unique.")
        feature_name_set = set(names)
        lineage_name_set = set(lineage_names)
        if feature_name_set != lineage_name_set:
            missing_lineage = sorted(feature_name_set - lineage_name_set)
            unexpected_lineage = sorted(lineage_name_set - feature_name_set)
            raise SchemaContractError(
                "DatasetSchema lineage outputs must match feature names exactly; "
                f"missing={missing_lineage!r}, unexpected={unexpected_lineage!r}."
            )
        object.__setattr__(self, "lineage", lineage)

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(value.name for value in self.features)

    @property
    def feature_roles(self) -> tuple[FeatureRole, ...]:
        return tuple(value.role for value in self.features)

    @property
    def metadata_dict(self) -> Mapping[str, Any]:
        return MappingProxyType({key: json.loads(value) for key, value in self.metadata})

    @property
    def fingerprint(self) -> str:
        payload = _canonical_json(self.to_record(include_fingerprint=False))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_record(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.version,
            "features": [feature.to_record() for feature in self.features],
            "lineage": [record.to_record() for record in self.lineage],
            "metadata": dict(self.metadata_dict),
        }
        if include_fingerprint:
            out["fingerprint"] = self.fingerprint
        return out

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "DatasetSchema":
        raw_features = record.get("features")
        if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
            raise SchemaContractError("Dataset schema record requires a feature sequence.")
        features = []
        for value in raw_features:
            if not isinstance(value, Mapping):
                raise SchemaContractError("Dataset schema feature record must be a mapping.")
            payload = dict(value)
            payload["annotations"] = _coerce_annotations(payload.get("annotations"))
            features.append(FeatureSpec(**payload))
        raw_lineage = record.get("lineage") or tuple()
        lineage = []
        for value in raw_lineage:
            if not isinstance(value, Mapping):
                raise SchemaContractError("Dataset schema lineage record must be a mapping.")
            payload = dict(value)
            payload["parameters"] = payload.get("parameters") or {}
            payload["input_names"] = tuple(payload.get("input_names") or tuple())
            lineage.append(FeatureLineage(**payload))
        schema = cls(
            features=tuple(features),
            lineage=tuple(lineage),
            version=str(record.get("schema_version", record.get("version", "1"))),
            metadata=record.get("metadata") or {},
        )
        expected = str(record.get("fingerprint", "") or "").strip()
        if expected and expected != schema.fingerprint:
            raise SchemaContractError("Dataset schema fingerprint does not match its content.")
        return schema

    @classmethod
    def from_dataframe(
        cls,
        frame: Any,
        *,
        roles: Mapping[str, FeatureRole | str] | None = None,
        annotations: Mapping[str, Sequence[FeatureAnnotation | Mapping[str, Any]]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "DatasetSchema":
        if pd is None or not isinstance(frame, pd.DataFrame):
            raise TypeError("from_dataframe requires a pandas DataFrame.")
        if frame.ndim != 2 or frame.shape[1] <= 0:
            raise SchemaContractError("Input DataFrame must have at least one column.")
        name_values = [str(value) for value in frame.columns]
        if len(set(name_values)) != len(name_values):
            raise SchemaContractError("Input DataFrame columns must be unique.")
        role_map = dict(roles or {})
        annotations = dict(annotations or {})
        unknown_roles = set(role_map) - set(name_values)
        if unknown_roles:
            raise SchemaContractError(
                f"Role overrides reference unknown columns: {sorted(unknown_roles)!r}."
            )
        unknown_annotations = set(annotations) - set(name_values)
        if unknown_annotations:
            raise SchemaContractError(
                "Annotation mappings reference unknown columns: "
                f"{sorted(unknown_annotations)!r}."
            )
        features: list[FeatureSpec] = []
        for raw_name, name in zip(frame.columns, name_values, strict=True):
            raw_role = role_map.get(name, _infer_role(name, frame[raw_name]))
            try:
                role = raw_role if isinstance(raw_role, FeatureRole) else FeatureRole(str(raw_role))
            except ValueError as exc:
                raise SchemaContractError(f"Unknown role override for {name!r}: {raw_role!r}.") from exc
            features.append(
                FeatureSpec(
                    name=name,
                    role=role,
                    dtype=str(frame[raw_name].dtype),
                    source_name=name,
                    annotations=_coerce_annotations(annotations.get(name)),
                )
            )
        input_kind = (
            "pandas_sparse_dataframe"
            if any(isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes)
            else "dataframe"
        )
        return cls(
            features=tuple(features),
            version="1",
            metadata={**dict(metadata or {}), "input_kind": input_kind},
        )

    @classmethod
    def from_input(
        cls,
        X: Any,
        *,
        roles: Mapping[str, FeatureRole | str] | None = None,
        annotations: Mapping[str, Sequence[FeatureAnnotation | Mapping[str, Any]]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "DatasetSchema":
        if pd is not None and isinstance(X, pd.DataFrame):
            return cls.from_dataframe(
                X,
                roles=roles,
                annotations=annotations,
                metadata=metadata,
            )
        is_sparse = bool(sp is not None and sp.issparse(X))
        if is_sparse:
            shape = X.shape
            dtype = str(getattr(X, "dtype", "float"))
            input_kind = "sparse"
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise SchemaContractError(f"Expected a 2-D input matrix, got shape {array.shape}.")
            shape = array.shape
            dtype = str(array.dtype)
            input_kind = "ndarray"
        if len(shape) != 2 or int(shape[1]) <= 0:
            raise SchemaContractError("Input matrix must have at least one feature column.")
        names = (
            tuple(str(value) for value in feature_names)
            if feature_names is not None
            else tuple(f"x{index}" for index in range(int(shape[1])))
        )
        if len(names) != int(shape[1]) or len(set(names)) != len(names):
            raise SchemaContractError("feature_names must be unique and match input width.")
        role_map = dict(roles or {})
        annotations = dict(annotations or {})
        unknown = (set(role_map) | set(annotations)) - set(names)
        if unknown:
            raise SchemaContractError(f"Schema mappings reference unknown columns: {sorted(unknown)!r}.")
        non_numeric_dense_input = bool(
            not is_sparse
            and not np.issubdtype(array.dtype, np.number)
            and not np.issubdtype(array.dtype, np.bool_)
        )
        if non_numeric_dense_input and set(role_map) != set(names):
            missing_roles = sorted(set(names) - set(role_map))
            raise SchemaContractError(
                "Non-numeric ndarray inputs require an explicit role for every "
                f"feature; missing roles for {missing_roles!r}. Pass a DatasetSchema "
                "or a complete roles mapping."
            )
        default_role = FeatureRole.CONTINUOUS
        features = tuple(
            FeatureSpec(
                name=name,
                role=(
                    role_map.get(name, default_role)
                    if isinstance(role_map.get(name, default_role), FeatureRole)
                    else FeatureRole(str(role_map.get(name, default_role)))
                ),
                dtype=dtype,
                source_name=name,
                annotations=_coerce_annotations(annotations.get(name)),
            )
            for name in names
        )
        return cls(
            features=features,
            version="1",
            metadata={**dict(metadata or {}), "input_kind": input_kind},
        )

    def validate_input(self, X: Any) -> None:
        """Fail closed on column identity/order or feature-width drift."""

        if pd is not None and isinstance(X, pd.DataFrame):
            names = tuple(str(value) for value in X.columns)
            if names != self.feature_names:
                raise SchemaContractError(
                    "DataFrame columns do not match schema order/identity; expected "
                    f"{self.feature_names!r}, got {names!r}."
                )
            return
        if sp is not None and sp.issparse(X):
            width = int(X.shape[1])
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise SchemaContractError(f"Expected a 2-D input matrix, got shape {array.shape}.")
            width = int(array.shape[1])
        if width != self.n_features:
            raise SchemaContractError(
                f"Input width {width} does not match schema width {self.n_features}."
            )

    def validate_inference_input(
        self,
        X: Any,
        *,
        alignment_mode: SchemaAlignmentMode | str = SchemaAlignmentMode.STRICT,
    ) -> InferenceSchemaCompatibilityReport:
        """Validate a fitted-estimator inference input without silently adapting it.

        This is intentionally stricter than :meth:`validate_input`, which is a
        legacy pipeline shape/name guard and is still used while fitting
        fold-local preprocessors.  For a DataFrame, this contract validates
        column identity/order and semantic dtype families against every
        ``FeatureSpec``.  ``alignment_mode='reorder'`` is the sole adaptation:
        it can handle an otherwise identical DataFrame in a different order and
        returns a report that records the requested reordering.

        Dense ndarrays and SciPy sparse matrices preserve legacy positional
        behavior only for schemas that were not fitted from a DataFrame. They
        receive ``typed_semantics_verified=False`` because a positional matrix
        cannot establish feature-name, role, or dtype equivalence.
        """

        mode = _coerce_schema_alignment_mode(alignment_mode)
        if pd is not None and isinstance(X, pd.DataFrame):
            return self._validate_inference_dataframe(X, alignment_mode=mode)

        if mode is SchemaAlignmentMode.REORDER:
            raise SchemaContractError(
                "alignment_mode='reorder' is only available for pandas DataFrame "
                "inputs with explicitly named columns."
            )

        if sp is not None and sp.issparse(X):
            width = int(X.shape[1])
            input_kind = "sparse"
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise SchemaContractError(
                    f"Expected a 2-D input matrix, got shape {array.shape}."
                )
            width = int(array.shape[1])
            input_kind = "ndarray"
        if width != self.n_features:
            raise SchemaContractError(
                "Inference input width does not match the fitted schema; "
                f"expected={self.n_features}, received={width}."
            )

        if not self._allows_legacy_positional_inference():
            raise SchemaContractError(
                "The fitted schema requires a pandas DataFrame so feature names, "
                "roles, and dtypes can be verified; a positional "
                f"{input_kind} cannot establish typed semantic equivalence."
            )

        return InferenceSchemaCompatibilityReport(
            schema_fingerprint=self.fingerprint,
            alignment_mode=mode,
            input_kind=input_kind,
            typed_semantics_verified=False,
            alignment_applied=False,
            output_feature_names=self.feature_names,
        )

    def _allows_legacy_positional_inference(self) -> bool:
        """Return whether the schema has an explicit legacy numeric matrix contract."""

        fitted_input_kind = str(self.metadata_dict.get("input_kind", "")).strip().lower()
        if fitted_input_kind in {"ndarray", "sparse"}:
            return True
        if fitted_input_kind:
            return False
        # Older numeric schemas did not persist ``input_kind``. Preserve only
        # their unmistakable x0/x1/... all-continuous form; a manually declared
        # typed schema must not silently downgrade to positional inference.
        return (
            self.feature_names == tuple(f"x{index}" for index in range(self.n_features))
            and all(feature.role is FeatureRole.CONTINUOUS for feature in self.features)
        )

    def align_inference_input(
        self,
        X: Any,
        *,
        alignment_mode: SchemaAlignmentMode | str = SchemaAlignmentMode.STRICT,
    ) -> tuple[Any, InferenceSchemaCompatibilityReport]:
        """Return a checked input plus its compatibility provenance record.

        In strict mode the original object is returned. In reviewed reorder mode
        a DataFrame is column-reindexed only after full name and dtype/role
        validation. Missing, extra, renamed, and type-incompatible fields are
        always rejected rather than filled, dropped, or coerced.
        """

        report = self.validate_inference_input(X, alignment_mode=alignment_mode)
        if not report.alignment_applied:
            return X, report
        if pd is None or not isinstance(X, pd.DataFrame):  # pragma: no cover - guarded above
            raise RuntimeError("DataFrame reorder alignment was reported for a non-DataFrame input.")
        received_positions = {
            str(name): position for position, name in enumerate(X.columns)
        }
        return X.iloc[:, [received_positions[name] for name in self.feature_names]], report

    def _validate_inference_dataframe(
        self,
        frame: Any,
        *,
        alignment_mode: SchemaAlignmentMode,
    ) -> InferenceSchemaCompatibilityReport:
        """Validate strict DataFrame inference identity and semantic dtypes."""

        received_names = tuple(str(value) for value in frame.columns)
        if len(set(received_names)) != len(received_names):
            raise SchemaContractError(
                "Inference DataFrame columns must be unique after schema name "
                f"normalization; received={received_names!r}."
            )
        expected_names = self.feature_names
        received_set = set(received_names)
        expected_set = set(expected_names)
        missing = tuple(name for name in expected_names if name not in received_set)
        unexpected = tuple(name for name in received_names if name not in expected_set)
        reordered = not missing and not unexpected and received_names != expected_names
        if missing or unexpected:
            raise SchemaContractError(
                "Inference DataFrame columns do not match the fitted schema; "
                f"missing={list(missing)!r}, unexpected={list(unexpected)!r}, "
                f"expected_order={expected_names!r}, received_order={received_names!r}. "
                "Reviewed reorder alignment never fills or drops columns."
            )
        if reordered and alignment_mode is SchemaAlignmentMode.STRICT:
            raise SchemaContractError(
                "Inference DataFrame columns are reordered relative to the fitted "
                f"schema; expected_order={expected_names!r}, "
                f"received_order={received_names!r}. Use "
                "alignment_mode='reorder' only after reviewing this input."
            )

        positions_by_name = {
            name: position for position, name in enumerate(received_names)
        }
        received_dtypes = tuple(
            (name, str(frame.iloc[:, position].dtype))
            for position, name in enumerate(received_names)
        )
        for feature in self.features:
            observed_dtype = frame.iloc[:, positions_by_name[feature.name]].dtype
            compatible, expected_family, observed_family = _inference_dtype_compatible(
                feature,
                observed_dtype,
            )
            if not compatible:
                raise SchemaContractError(
                    "Inference dtype/role mismatch for feature "
                    f"{feature.name!r}; expected role={feature.role.value!r}, "
                    f"stored_dtype={feature.dtype!r}, "
                    f"semantic_family={expected_family!r}; received "
                    f"dtype={str(observed_dtype)!r}, "
                    f"semantic_family={observed_family!r}. Strict inference does "
                    "not coerce categorical, object, numeric, or boolean values."
                )

        return InferenceSchemaCompatibilityReport(
            schema_fingerprint=self.fingerprint,
            alignment_mode=alignment_mode,
            input_kind=(
                "pandas_sparse_dataframe"
                if any(isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes)
                else "dataframe"
            ),
            typed_semantics_verified=True,
            alignment_applied=bool(reordered),
            received_feature_names=received_names,
            output_feature_names=expected_names,
            received_dtypes=received_dtypes,
        )

    def select(self, indices: Sequence[int], *, operation: str = "selector_output") -> "DatasetSchema":
        selected = tuple(int(value) for value in indices)
        if not selected:
            raise SchemaContractError("Selector output cannot contain zero features.")
        if len(set(selected)) != len(selected):
            raise SchemaContractError("Selector output indices must be unique.")
        if any(value < 0 or value >= self.n_features for value in selected):
            raise SchemaContractError("Selector output index is out of schema bounds.")
        features = tuple(self.features[index] for index in selected)
        parent_lineage = {record.output_name: record for record in self.lineage}
        lineage = tuple(
            FeatureLineage.from_parameters(
                output_name=feature.name,
                operation=str(operation),
                input_names=(feature.name,),
                source_schema_hash=self.fingerprint,
                parameters={
                    "selected_index": int(index),
                    "parent_operation": parent_lineage[feature.name].operation,
                },
            )
            for index, feature in zip(selected, features, strict=True)
        )
        return DatasetSchema(
            features=features,
            lineage=lineage,
            version=self.version,
            metadata={
                **dict(self.metadata_dict),
                "parent_schema_fingerprint": self.fingerprint,
                "selection_operation": str(operation),
            },
        )


def infer_dataset_schema(
    X: Any,
    *,
    schema: DatasetSchema | None = None,
    roles: Mapping[str, FeatureRole | str] | None = None,
    annotations: Mapping[str, Sequence[FeatureAnnotation | Mapping[str, Any]]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    feature_names: Sequence[str] | None = None,
) -> DatasetSchema:
    """Return a supplied schema after validation or infer a frozen schema."""

    if schema is not None:
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be a DatasetSchema or None.")
        schema.validate_input(X)
        return schema
    return DatasetSchema.from_input(
        X,
        roles=roles,
        annotations=annotations,
        metadata=metadata,
        feature_names=feature_names,
    )


__all__ = [
    "DatasetSchema",
    "FeatureAnnotation",
    "FeatureLineage",
    "FeatureRole",
    "FeatureSpec",
    "InferenceSchemaCompatibilityReport",
    "SchemaAlignmentMode",
    "SchemaContractError",
    "infer_dataset_schema",
]

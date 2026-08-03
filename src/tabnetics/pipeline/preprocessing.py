"""Fold-local typed-input preprocessing and selector input admission.

The existing DF+FS implementation is numeric.  This module forms the explicit
boundary for DataFrames and sparse matrices: learned encoder/imputer state is
fit on a training fold only and every output carries immutable schema/lineage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer

from tabnetics.classification.registry import (
    ClassifierCapabilityOverrides,
    ClassifierRuntimeFacts,
    ResourceClass,
    SupportLevel,
    resolve_classifier_capabilities,
)
from tabnetics.datasets.schema import (
    DatasetSchema,
    FeatureLineage,
    FeatureRole,
    FeatureSpec,
    SchemaContractError,
    infer_dataset_schema,
)

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    from scipy import sparse as sp
except Exception:  # pragma: no cover
    sp = None  # type: ignore[assignment]


_TEXT_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_NUMERIC_ROLES = frozenset(
    {
        FeatureRole.CONTINUOUS,
        FeatureRole.COUNT,
        FeatureRole.BINARY,
        FeatureRole.ORDINAL,
        FeatureRole.DERIVED,
    }
)
_EXCLUDED_ROLES = frozenset(
    {FeatureRole.GROUP, FeatureRole.TIME, FeatureRole.IGNORED}
)
_TEXT_VALUE_HASH_BUCKETS = 1024
__tabnetics_execution_isolated_state__ = {
    "_EXCLUDED_ROLES": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "_NUMERIC_ROLES": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "_TEXT_TOKEN_PATTERN": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}


class TypedInputCapabilityError(RuntimeError):
    """Fail-closed typed input error with deterministic provenance."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.diagnostics = MappingProxyType(dict(diagnostics or {}))


def _record_items(values: Mapping[str, Any] | Sequence[tuple[str, str]] | None) -> tuple[tuple[str, str], ...]:
    if values is None:
        return tuple()
    if isinstance(values, Mapping):
        raw = values.items()
        return tuple(
            sorted(
                (
                    str(key),
                    json.dumps(value, sort_keys=True, separators=(",", ":"), default=str),
                )
                for key, value in raw
            )
        )
    return tuple((str(key), str(value)) for key, value in values)


@dataclass(frozen=True, slots=True)
class TransformedInput:
    """Matrix plus frozen output schema/lineage provenance."""

    X: Any
    schema: DatasetSchema
    source_schema: DatasetSchema
    output_mode: str
    metadata: tuple[tuple[str, str], ...] = tuple()

    def __post_init__(self) -> None:
        mode = str(self.output_mode).strip().lower()
        if mode not in {"numeric", "native_categorical", "sparse"}:
            raise ValueError(f"Unknown transformed input mode: {self.output_mode!r}.")
        object.__setattr__(self, "output_mode", mode)
        object.__setattr__(self, "metadata", _record_items(self.metadata))

    @property
    def metadata_dict(self) -> Mapping[str, Any]:
        return MappingProxyType({key: json.loads(value) for key, value in self.metadata})

    def to_record(self) -> dict[str, Any]:
        return {
            "output_mode": self.output_mode,
            "shape": [int(value) for value in getattr(self.X, "shape", tuple())],
            "source_schema_fingerprint": self.source_schema.fingerprint,
            "schema": self.schema.to_record(),
            "metadata": dict(self.metadata_dict),
        }


@dataclass(frozen=True, slots=True)
class NativeCategoricalStage2Bridge:
    """Validated correspondence between numeric FS and a native categorical view.

    The numeric output remains the authoritative feature-selection space.  This
    bridge exists only to map an already-selected numeric position to its exact
    paired native DataFrame column without inferring relationships from late
    result metadata.
    """

    source_schema_fingerprint: str
    numeric_schema_fingerprint: str
    native_schema_fingerprint: str
    numeric_feature_names: tuple[str, ...]
    native_feature_names: tuple[str, ...]
    native_position_by_numeric_position: tuple[int, ...]
    categorical_native_columns: tuple[str, ...]
    category_vocabularies: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        n_numeric = len(self.numeric_feature_names)
        if n_numeric <= 0:
            raise ValueError("A native Stage-2 bridge requires numeric features.")
        if len(self.native_position_by_numeric_position) != n_numeric:
            raise ValueError("Native Stage-2 position map must cover every numeric feature.")
        if len(set(self.native_feature_names)) != len(self.native_feature_names):
            raise ValueError("Native Stage-2 schema contains duplicate feature names.")
        for position in self.native_position_by_numeric_position:
            if not 0 <= int(position) < len(self.native_feature_names):
                raise ValueError("Native Stage-2 position map contains an out-of-range value.")
        vocab_names = tuple(name for name, _ in self.category_vocabularies)
        if tuple(self.categorical_native_columns) != vocab_names:
            raise ValueError("Native Stage-2 categorical vocabulary records are misaligned.")

    @property
    def category_vocabulary_map(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {str(name): tuple(str(value) for value in values)
             for name, values in self.category_vocabularies}
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "source_schema_fingerprint": str(self.source_schema_fingerprint),
            "numeric_schema_fingerprint": str(self.numeric_schema_fingerprint),
            "native_schema_fingerprint": str(self.native_schema_fingerprint),
            "numeric_feature_names": list(self.numeric_feature_names),
            "native_feature_names": list(self.native_feature_names),
            "native_position_by_numeric_position": [
                int(value) for value in self.native_position_by_numeric_position
            ],
            "categorical_native_columns": list(self.categorical_native_columns),
            "category_vocabularies": {
                str(name): list(values) for name, values in self.category_vocabularies
            },
        }


def _is_pandas_sparse_dataframe(X: Any) -> bool:
    """Return whether a DataFrame carries one or more SparseDtype columns."""

    return bool(
        pd is not None
        and isinstance(X, pd.DataFrame)
        and any(isinstance(dtype, pd.SparseDtype) for dtype in X.dtypes)
    )


def is_sparse_input(X: Any) -> bool:
    """Return true for SciPy sparse matrices and pandas sparse DataFrames."""

    return bool(sp is not None and sp.issparse(X)) or _is_pandas_sparse_dataframe(X)


def _as_csr_sparse(X: Any) -> Any:
    """Convert a supported sparse input to CSR without dense materialization."""

    if sp is None:
        raise TypedInputCapabilityError(
            "sparse_backend_unavailable",
            "Sparse typed input requires scipy.sparse.",
        )
    if sp.issparse(X):
        return X.tocsr(copy=True)
    if not _is_pandas_sparse_dataframe(X):
        raise TypeError("Expected a SciPy sparse matrix or pandas sparse DataFrame.")

    assert pd is not None and isinstance(X, pd.DataFrame)
    sparse_columns = [isinstance(dtype, pd.SparseDtype) for dtype in X.dtypes]
    if not all(sparse_columns):
        raise TypedInputCapabilityError(
            "mixed_pandas_sparse_dataframe",
            "A pandas sparse DataFrame must use SparseDtype for every model column; "
            "mixed sparse/dense frames are rejected before dense materialization.",
            diagnostics={
                "sparse_columns": int(sum(sparse_columns)),
                "n_columns": int(X.shape[1]),
            },
        )
    try:
        return X.sparse.to_coo().tocsr(copy=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypedInputCapabilityError(
            "pandas_sparse_conversion_unsupported",
            "The pandas sparse DataFrame cannot be converted to CSR without a "
            "dense materialization.",
            diagnostics={
                "n_rows": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "dtypes": [str(dtype) for dtype in X.dtypes],
                "error_type": type(exc).__name__,
            },
        ) from exc


def is_typed_input(X: Any) -> bool:
    """Return true when input cannot use legacy dense-numeric ndarray flow."""

    if pd is not None and isinstance(X, pd.DataFrame):
        return True
    if is_sparse_input(X):
        return True
    array = np.asarray(X)
    return bool(array.ndim == 2 and not np.issubdtype(array.dtype, np.number))


def guarded_sparse_to_dense(X: Any, *, max_elements: int, callsite: str) -> np.ndarray:
    """Materialize sparse data only under an explicit, deterministic cap."""

    if not is_sparse_input(X):
        return np.asarray(X)
    matrix = _as_csr_sparse(X)
    rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
    elements = rows * columns
    limit = int(max(0, max_elements))
    if limit <= 0 or elements > limit:
        raise TypedInputCapabilityError(
            "sparse_to_dense_unsafe",
            "The numeric DF+FS core has no effective sparse selector route and the "
            "requested sparse-to-dense bridge exceeds its configured safety cap.",
            diagnostics={
                "callsite": str(callsite),
                "n_rows": rows,
                "n_features": columns,
                "dense_elements": elements,
                "max_dense_elements": limit,
                "reason": "sparse_input_unsupported_by_numeric_fs_core",
            },
        )
    return np.asarray(matrix.toarray())


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value) if pd is not None else np.isnan(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _category_key(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _tokenize(value: Any) -> tuple[str, ...]:
    if _is_missing(value):
        return tuple()
    return tuple(_TEXT_TOKEN_PATTERN.findall(str(value).lower()))


def _bucket(value: str, buckets: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % max(1, int(buckets))


@dataclass(frozen=True, slots=True)
class FeatureSelectorRuntimeFacts:
    """Raw input and fold-local adapter facts for selector admission."""

    input_is_sparse: bool = False
    input_has_categorical: bool = False
    input_has_missing: bool = False
    sample_weight_requested: bool = False
    structured_resampling_requested: bool = False
    fold_local_adapter: str = "none"
    structured_output_required: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "input_is_sparse",
            "input_has_categorical",
            "input_has_missing",
            "sample_weight_requested",
            "structured_resampling_requested",
            "structured_output_required",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool.")
        adapter = str(self.fold_local_adapter or "none").strip().lower()
        if adapter not in {"none", "numeric", "bounded_dense", "sparse_native"}:
            raise ValueError(f"Unknown fold_local_adapter: {self.fold_local_adapter!r}.")
        object.__setattr__(self, "fold_local_adapter", adapter)


@dataclass(frozen=True, slots=True)
class ResolvedFeatureSelectorCapabilities:
    """Resolved capability state for one canonical selector method."""

    method_name: str
    estimator_sparse_input: SupportLevel
    effective_sparse_input: SupportLevel
    categorical_input: SupportLevel
    missing_values: SupportLevel
    sample_weight: SupportLevel
    structured_resampling: SupportLevel
    deterministic: SupportLevel
    resource_class: ResourceClass
    structured_output: SupportLevel
    state_serialization: SupportLevel
    availability: SupportLevel
    adapter: str
    availability_reasons: tuple[str, ...] = tuple()

    @property
    def is_available(self) -> bool:
        return self.availability is SupportLevel.SUPPORTED

    def to_record(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "estimator_sparse_input": self.estimator_sparse_input.value,
            "effective_sparse_input": self.effective_sparse_input.value,
            "categorical_input": self.categorical_input.value,
            "missing_values": self.missing_values.value,
            "sample_weight": self.sample_weight.value,
            "structured_resampling": self.structured_resampling.value,
            "deterministic": self.deterministic.value,
            "resource_class": self.resource_class.value,
            "structured_output": self.structured_output.value,
            "state_serialization": self.state_serialization.value,
            "availability": self.availability.value,
            "adapter": self.adapter,
            "availability_reasons": list(self.availability_reasons),
        }


def _combine_support(values: Sequence[SupportLevel]) -> SupportLevel:
    if any(value is SupportLevel.UNSUPPORTED for value in values):
        return SupportLevel.UNSUPPORTED
    if any(value is SupportLevel.UNKNOWN for value in values):
        return SupportLevel.UNKNOWN
    if any(value is SupportLevel.CONDITIONAL for value in values):
        return SupportLevel.CONDITIONAL
    return SupportLevel.SUPPORTED


def resolve_feature_selector_capabilities(
    method_name: str,
    *,
    runtime: FeatureSelectorRuntimeFacts | None = None,
) -> ResolvedFeatureSelectorCapabilities:
    """Resolve every registered selector's typed-input admission contract.

    The selector core currently receives numeric data. Categorical, missing,
    and bounded sparse inputs are admitted only after an explicit fold-local
    adapter. Feature selection itself does not consume sample weights.
    """

    from tabnetics.feature_selection.registry import METHOD_REGISTRY

    name = str(method_name).strip()
    spec = METHOD_REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"Unknown feature selector method: {method_name!r}.")
    runtime = runtime or FeatureSelectorRuntimeFacts()
    if not isinstance(runtime, FeatureSelectorRuntimeFacts):
        raise TypeError("runtime must be FeatureSelectorRuntimeFacts or None.")
    dense_adapter_honored = runtime.fold_local_adapter in {"numeric", "bounded_dense"}
    sparse_bridge_honored = runtime.fold_local_adapter == "bounded_dense"
    estimator_sparse = SupportLevel.UNSUPPORTED
    effective_sparse = (
        SupportLevel.SUPPORTED
        if not runtime.input_is_sparse or sparse_bridge_honored
        else SupportLevel.UNSUPPORTED
    )
    categorical = (
        SupportLevel.SUPPORTED
        if not runtime.input_has_categorical or dense_adapter_honored
        else SupportLevel.UNSUPPORTED
    )
    missing = (
        SupportLevel.SUPPORTED
        if not runtime.input_has_missing or dense_adapter_honored
        else SupportLevel.UNSUPPORTED
    )
    sample_weight = (
        SupportLevel.UNSUPPORTED if runtime.sample_weight_requested else SupportLevel.SUPPORTED
    )
    raw_structured = str(getattr(spec, "structured_resampling", "unsupported") or "unsupported")
    try:
        structured = SupportLevel(raw_structured)
    except ValueError:
        structured = SupportLevel.UNKNOWN
    if not runtime.structured_resampling_requested:
        structured = SupportLevel.SUPPORTED
    paradigm = str(getattr(spec, "paradigm", "filter") or "filter").strip().lower()
    deterministic = (
        SupportLevel.CONDITIONAL
        if paradigm in {"stability", "wrapper", "knockoff"} or name == "random"
        else SupportLevel.SUPPORTED
    )
    resource = (
        ResourceClass.GPU_REQUIRED
        if bool(getattr(spec, "requires_gpu", False))
        else ResourceClass.CPU_HEAVY
        if paradigm in {"stability", "wrapper", "knockoff"}
        else ResourceClass.CPU_STANDARD
    )
    needs_structured_state = name in {"ktsp"}
    structured_output = (
        SupportLevel.UNSUPPORTED
        if runtime.structured_output_required and needs_structured_state
        else SupportLevel.SUPPORTED
    )
    state_serialization = (
        SupportLevel.CONDITIONAL if needs_structured_state else SupportLevel.SUPPORTED
    )
    availability = _combine_support(
        [effective_sparse, categorical, missing, sample_weight, structured, structured_output]
    )
    reasons: list[str] = []
    if runtime.input_is_sparse and estimator_sparse is SupportLevel.UNSUPPORTED:
        if sparse_bridge_honored:
            reasons.append("sparse_input:bounded_dense_adapter")
        elif runtime.fold_local_adapter == "sparse_native":
            reasons.append("sparse_input:sparse_native_unimplemented")
        else:
            reasons.append("sparse_input:bounded_dense_adapter_required")
    if runtime.input_is_sparse and not sparse_bridge_honored:
        reasons.append("sparse_input:unsupported")
    if runtime.input_has_categorical and not dense_adapter_honored:
        reasons.append("categorical_input:fold_local_adapter_required")
    if runtime.input_has_missing and not dense_adapter_honored:
        reasons.append("missing_values:fold_local_adapter_required")
    if runtime.sample_weight_requested:
        reasons.append("sample_weight:unsupported_until_235")
    if runtime.structured_resampling_requested and structured is not SupportLevel.SUPPORTED:
        reasons.append(f"structured_resampling:{structured.value}")
    if runtime.structured_output_required and needs_structured_state:
        reasons.append("structured_output:unsupported_until_241_236")
    return ResolvedFeatureSelectorCapabilities(
        method_name=name,
        estimator_sparse_input=estimator_sparse,
        effective_sparse_input=effective_sparse,
        categorical_input=categorical,
        missing_values=missing,
        sample_weight=sample_weight,
        structured_resampling=structured,
        deterministic=deterministic,
        resource_class=resource,
        structured_output=structured_output,
        state_serialization=state_serialization,
        availability=availability,
        adapter=runtime.fold_local_adapter,
        availability_reasons=tuple(reasons),
    )


def admit_feature_selector_methods(
    method_names: Sequence[str],
    *,
    runtime: FeatureSelectorRuntimeFacts,
) -> tuple[tuple[str, ...], Mapping[str, str], tuple[ResolvedFeatureSelectorCapabilities, ...]]:
    """Return admitted methods, deterministic rejections, and full records."""

    admitted: list[str] = []
    rejected: dict[str, str] = {}
    records: list[ResolvedFeatureSelectorCapabilities] = []
    for name in method_names:
        record = resolve_feature_selector_capabilities(name, runtime=runtime)
        records.append(record)
        if record.is_available:
            admitted.append(str(name))
        else:
            rejected[str(name)] = (
                record.availability_reasons[0]
                if record.availability_reasons
                else f"availability:{record.availability.value}"
            )
    return tuple(admitted), MappingProxyType(rejected), tuple(records)


class FoldLocalPreprocessor(BaseEstimator):
    """Cloneable train-fold-only typed preprocessor.

    ``fit`` consumes one train partition. Category levels, median fills, and
    text IDF are then reused by ``transform``; test rows never update learned
    state. Group, time, and ignored columns remain out of predictor views.
    """

    def __init__(
        self,
        *,
        categorical_encoding: str = "ordinal",
        text_encoding: str = "tfidf_hash",
        text_hash_buckets: int = 16,
        unknown_category_code: float = 0.0,
        missing_category_code: float = -1.0,
        unknown_category_token: str = "__tabnetics_unknown__",
        missing_category_token: str = "__tabnetics_missing__",
    ) -> None:
        self.categorical_encoding = categorical_encoding
        self.text_encoding = text_encoding
        self.text_hash_buckets = text_hash_buckets
        self.unknown_category_code = unknown_category_code
        self.missing_category_code = missing_category_code
        self.unknown_category_token = unknown_category_token
        self.missing_category_token = missing_category_token

    def _validate_parameters(self) -> None:
        if str(self.categorical_encoding).strip().lower() != "ordinal":
            raise ValueError("categorical_encoding must currently be 'ordinal'.")
        if str(self.text_encoding).strip().lower() not in {"tfidf_hash", "length_hash", "drop"}:
            raise ValueError("text_encoding must be tfidf_hash, length_hash, or drop.")
        if int(self.text_hash_buckets) <= 0:
            raise ValueError("text_hash_buckets must be positive.")
        if not math.isfinite(float(self.unknown_category_code)):
            raise ValueError("unknown_category_code must be finite.")
        if not math.isfinite(float(self.missing_category_code)):
            raise ValueError("missing_category_code must be finite.")
        if not str(self.unknown_category_token) or not str(self.missing_category_token):
            raise ValueError("native category tokens must be non-empty.")

    def _frame(self, X: Any) -> Any:
        if pd is None:
            raise ImportError("pandas is required for typed dense preprocessing.")
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            frame.columns = [str(value) for value in frame.columns]
            return frame.loc[:, list(self.input_schema_.feature_names)].copy()
        values = np.asarray(X)
        if values.ndim != 2:
            raise SchemaContractError(f"Expected 2-D X, got shape {values.shape}.")
        return pd.DataFrame(values, columns=list(self.input_schema_.feature_names))

    def _lineage(self, name: str, operation: str, parameters: Mapping[str, Any]) -> FeatureLineage:
        return FeatureLineage.from_parameters(
            output_name=name,
            operation=operation,
            input_names=(name.split("__", 1)[0],),
            source_schema_hash=self.input_schema_.fingerprint,
            parameters=parameters,
        )

    def _fit_sparse(self, X: Any) -> None:
        invalid = [
            feature.name
            for feature in self.input_schema_.features
            if feature.role not in _NUMERIC_ROLES
        ]
        if invalid:
            raise TypedInputCapabilityError(
                "sparse_non_numeric_schema",
                "Sparse matrices require numeric roles; use a DataFrame for categorical, text, group, or time columns.",
                diagnostics={"invalid_features": invalid},
            )
        matrix = _as_csr_sparse(X)
        try:
            matrix = matrix.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypedInputCapabilityError(
                "sparse_non_numeric_values",
                "Sparse typed input must contain values convertible to float64.",
                diagnostics={"dtype": str(getattr(matrix, "dtype", "unknown"))},
            ) from exc
        has_nan = bool(matrix.data.size and np.isnan(matrix.data).any())
        self.sparse_imputer_ = SimpleImputer(strategy="median") if has_nan else None
        if self.sparse_imputer_ is not None:
            self.sparse_imputer_.fit(matrix)
        features = tuple(
            FeatureSpec(
                name=feature.name,
                role=feature.role,
                dtype="float64",
                source_name=feature.source_name,
                annotations=feature.annotations,
            )
            for feature in self.input_schema_.features
        )
        lineage = tuple(
            FeatureLineage.from_parameters(
                output_name=feature.name,
                operation="sparse_numeric_impute" if has_nan else "sparse_passthrough",
                input_names=(feature.name,),
                source_schema_hash=self.input_schema_.fingerprint,
                parameters={"fit_rows": int(matrix.shape[0])},
            )
            for feature in features
        )
        self.numeric_schema_ = DatasetSchema(
            features=features,
            lineage=lineage,
            metadata={
                "parent_schema_fingerprint": self.input_schema_.fingerprint,
                "preprocessor": "fold_local",
                "output_mode": "sparse",
            },
        )
        self.native_schema_ = self.numeric_schema_
        self._output_feature_names = self.numeric_schema_.feature_names
        self._sparse_input_ = True
        self.input_has_missing_ = has_nan

    def _fit_dense(self, frame: Any) -> None:
        assert pd is not None
        numeric_fill: dict[str, float] = {}
        category_levels: dict[str, tuple[str, ...]] = {}
        native_levels: dict[str, tuple[str, ...]] = {}
        text_idf: dict[str, np.ndarray] = {}
        numeric_features: list[FeatureSpec] = []
        native_features: list[FeatureSpec] = []
        numeric_lineage: list[FeatureLineage] = []
        native_lineage: list[FeatureLineage] = []
        excluded: list[str] = []
        has_missing = False

        for feature in self.input_schema_.features:
            name = feature.name
            series = frame[name]
            if feature.role in _EXCLUDED_ROLES:
                excluded.append(name)
                continue
            if feature.role in _NUMERIC_ROLES:
                values = pd.to_numeric(series, errors="coerce").astype(float)
                values = values.replace([np.inf, -np.inf], np.nan)
                has_missing = has_missing or bool(values.isna().any())
                finite = values.dropna().to_numpy(dtype=float)
                fill = float(np.median(finite)) if finite.size else 0.0
                numeric_fill[name] = fill
                spec = FeatureSpec(name=name, role=feature.role, dtype="float64", source_name=feature.source_name, annotations=feature.annotations)
                numeric_features.append(spec)
                native_features.append(spec)
                record = FeatureLineage.from_parameters(
                    output_name=name,
                    operation="numeric_median_impute",
                    input_names=(name,),
                    source_schema_hash=self.input_schema_.fingerprint,
                    parameters={"fill_value": fill, "fit_rows": int(frame.shape[0])},
                )
                numeric_lineage.append(record)
                native_lineage.append(record)
                continue
            if feature.role is FeatureRole.CATEGORICAL:
                raw = list(series.tolist())
                # Category keys include the Python type so values such as ``1``
                # and ``"1"`` cannot collapse into one learned level.  They are
                # also stable across a pickle round trip and safe for a native
                # categorical backend to consume as opaque labels.
                category_levels[name] = tuple(
                    sorted(
                        {
                            _category_key(value)
                            for value in raw
                            if not _is_missing(value)
                        }
                    )
                )
                native_levels[name] = category_levels[name]
                has_missing = has_missing or any(_is_missing(value) for value in raw)
                numeric_features.append(FeatureSpec(name=name, role=FeatureRole.DERIVED, dtype="float64", source_name=feature.source_name, annotations=feature.annotations))
                native_features.append(FeatureSpec(name=name, role=FeatureRole.CATEGORICAL, dtype="category", source_name=feature.source_name, annotations=feature.annotations))
                numeric_lineage.append(
                    FeatureLineage.from_parameters(
                        output_name=name,
                        operation="ordinal_encode_train_only",
                        input_names=(name,),
                        source_schema_hash=self.input_schema_.fingerprint,
                        parameters={"n_train_categories": len(category_levels[name]), "unknown_code": float(self.unknown_category_code), "missing_code": float(self.missing_category_code)},
                    )
                )
                native_lineage.append(
                    FeatureLineage.from_parameters(
                        output_name=name,
                        operation="native_categorical_train_only",
                        input_names=(name,),
                        source_schema_hash=self.input_schema_.fingerprint,
                        parameters={"n_train_categories": len(native_levels[name]), "unknown_token": str(self.unknown_category_token), "missing_token": str(self.missing_category_token)},
                    )
                )
                continue
            if feature.role is FeatureRole.TEXT:
                raw = list(series.tolist())
                has_missing = has_missing or any(_is_missing(value) for value in raw)
                buckets = int(self.text_hash_buckets)
                document_frequency = np.zeros(buckets, dtype=float)
                for value in raw:
                    for bucket in {_bucket(token, buckets) for token in _tokenize(value)}:
                        document_frequency[bucket] += 1.0
                text_idf[name] = np.log((1.0 + float(max(1, len(raw)))) / (1.0 + document_frequency)) + 1.0
                if str(self.text_encoding).strip().lower() == "drop":
                    excluded.append(name)
                    continue
                names = [f"{name}__text_len", f"{name}__text_hash"]
                if str(self.text_encoding).strip().lower() == "tfidf_hash":
                    names.extend(f"{name}__tfidf_hash_{i:02d}" for i in range(buckets))
                for derived in names:
                    spec = FeatureSpec(name=derived, role=FeatureRole.DERIVED, dtype="float64", source_name=feature.source_name, annotations=feature.annotations)
                    numeric_features.append(spec)
                    native_features.append(spec)
                    if derived == f"{name}__text_len":
                        operation = "text_character_length"
                        parameters: Mapping[str, Any] = {}
                    elif derived == f"{name}__text_hash":
                        operation = "text_value_hash"
                        parameters = {
                            "hash_algorithm": "blake2b-64",
                            "hash_buckets": _TEXT_VALUE_HASH_BUCKETS,
                        }
                    else:
                        operation = "text_tfidf_hash_train_only"
                        parameters = {
                            "hash_algorithm": "blake2b-64",
                            "hash_buckets": buckets,
                            "fit_rows": int(frame.shape[0]),
                        }
                    record = FeatureLineage.from_parameters(
                        output_name=derived,
                        operation=operation,
                        input_names=(name,),
                        source_schema_hash=self.input_schema_.fingerprint,
                        parameters=parameters,
                    )
                    numeric_lineage.append(record)
                    native_lineage.append(record)
                continue
            raise TypedInputCapabilityError("unsupported_feature_role", f"Unsupported role {feature.role.value!r}.", diagnostics={"feature": name})

        if not numeric_features:
            raise TypedInputCapabilityError(
                "no_model_features_after_schema_roles",
                "The schema excludes every column from model input.",
                diagnostics={"excluded_features": excluded},
            )
        metadata = {
            "parent_schema_fingerprint": self.input_schema_.fingerprint,
            "preprocessor": "fold_local",
            "fit_rows": int(frame.shape[0]),
            "excluded_features": tuple(excluded),
            "text_encoding": str(self.text_encoding).strip().lower(),
        }
        self.numeric_schema_ = DatasetSchema(features=tuple(numeric_features), lineage=tuple(numeric_lineage), metadata={**metadata, "output_mode": "numeric"})
        self.native_schema_ = DatasetSchema(features=tuple(native_features), lineage=tuple(native_lineage), metadata={**metadata, "output_mode": "native_categorical"})
        # Fitted estimators must remain pickleable.  The schema/lineage records
        # are immutable; sklearn-style fitted state is intentionally stored in
        # ordinary private dictionaries so joblib/pickle can replay it.
        self.numeric_fill_values_ = dict(numeric_fill)
        self.category_levels_ = dict(category_levels)
        self.native_category_levels_ = dict(native_levels)
        self.text_idf_ = dict(text_idf)
        self.excluded_features_ = tuple(excluded)
        self.input_has_missing_ = bool(has_missing)
        self._output_feature_names = self.numeric_schema_.feature_names
        self._sparse_input_ = False

    def fit(self, X: Any, y: Any = None, *, schema: DatasetSchema | None = None) -> "FoldLocalPreprocessor":
        """Fit all learned preprocessing state on the supplied train partition."""

        del y
        self._validate_parameters()
        self.input_schema_ = infer_dataset_schema(X, schema=schema)
        self.input_schema_.validate_input(X)
        if is_sparse_input(X):
            self._fit_sparse(X)
        else:
            self._fit_dense(self._frame(X))
        self.n_features_in_ = int(self.input_schema_.n_features)
        self.fit_row_count_ = int(X.shape[0])
        self.fit_schema_fingerprint_ = self.input_schema_.fingerprint
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "input_schema_"):
            raise RuntimeError("FoldLocalPreprocessor must be fit before transform.")

    def _transform_sparse(self, X: Any) -> TransformedInput:
        matrix = _as_csr_sparse(X)
        try:
            matrix = matrix.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypedInputCapabilityError(
                "sparse_non_numeric_values",
                "Sparse typed input must contain values convertible to float64.",
                diagnostics={"dtype": str(getattr(matrix, "dtype", "unknown"))},
            ) from exc
        if self.sparse_imputer_ is not None:
            matrix = self.sparse_imputer_.transform(matrix)
            if not is_sparse_input(matrix):
                raise TypedInputCapabilityError(
                    "sparse_transform_densified",
                    "Sparse imputation unexpectedly materialized a dense matrix.",
                    diagnostics={"shape": tuple(int(value) for value in X.shape)},
                )
            matrix = matrix.tocsr()
        return TransformedInput(
            X=matrix,
            schema=self.numeric_schema_,
            source_schema=self.input_schema_,
            output_mode="sparse",
            metadata=_record_items({"fit_rows": int(self.fit_row_count_), "sparse_preserved": True}),
        )

    def _numeric_values(self, frame: Any, feature: FeatureSpec) -> np.ndarray:
        assert pd is not None
        name = feature.name
        if feature.role in _NUMERIC_ROLES:
            values = pd.to_numeric(frame[name], errors="coerce").astype(float)
            values = values.replace([np.inf, -np.inf], np.nan)
            return values.fillna(float(self.numeric_fill_values_[name])).to_numpy(dtype=float)
        if feature.role is FeatureRole.CATEGORICAL:
            mapping = {key: index + 1 for index, key in enumerate(self.category_levels_[name])}
            out = np.empty(frame.shape[0], dtype=float)
            for index, value in enumerate(frame[name].tolist()):
                if _is_missing(value):
                    out[index] = float(self.missing_category_code)
                else:
                    out[index] = float(mapping.get(_category_key(value), self.unknown_category_code))
            return out
        raise KeyError(name)

    def _text_values(self, frame: Any, name: str) -> list[tuple[str, np.ndarray]]:
        raw = list(frame[name].tolist())
        values: list[tuple[str, np.ndarray]] = [
            (f"{name}__text_len", np.asarray([float(len(str(value))) if not _is_missing(value) else 0.0 for value in raw])),
            (f"{name}__text_hash", np.asarray([float(_bucket(str(value), _TEXT_VALUE_HASH_BUCKETS)) if not _is_missing(value) else 0.0 for value in raw])),
        ]
        if str(self.text_encoding).strip().lower() != "tfidf_hash":
            return values
        buckets = int(self.text_hash_buckets)
        matrix = np.zeros((len(raw), buckets), dtype=float)
        idf = self.text_idf_[name]
        for row, value in enumerate(raw):
            tokens = _tokenize(value)
            if not tokens:
                continue
            counts: dict[int, int] = {}
            for token in tokens:
                bucket = _bucket(token, buckets)
                counts[bucket] = counts.get(bucket, 0) + 1
            for bucket, count in counts.items():
                matrix[row, bucket] = float(count) / float(len(tokens)) * float(idf[bucket])
        values.extend((f"{name}__tfidf_hash_{bucket:02d}", matrix[:, bucket]) for bucket in range(buckets))
        return values

    def _native_categories(self, series: Any, name: str) -> Any:
        assert pd is not None
        known = set(self.native_category_levels_[name])
        values = []
        for value in series.tolist():
            if _is_missing(value):
                values.append(str(self.missing_category_token))
            else:
                candidate = _category_key(value)
                values.append(candidate if candidate in known else str(self.unknown_category_token))
        categories = tuple(dict.fromkeys([*self.native_category_levels_[name], str(self.unknown_category_token), str(self.missing_category_token)]))
        return pd.Categorical(values, categories=categories)

    def transform_with_schema(
        self,
        X: Any,
        *,
        schema: DatasetSchema | None = None,
        output_mode: str = "numeric",
    ) -> TransformedInput:
        """Transform with only train-fold state and emit immutable provenance."""

        self._require_fitted()
        if schema is not None:
            checked = infer_dataset_schema(X, schema=schema)
            if checked.fingerprint != self.input_schema_.fingerprint:
                raise SchemaContractError("Transform schema differs from the fitted input schema.")
        self.input_schema_.validate_input(X)
        mode = str(output_mode).strip().lower()
        if is_sparse_input(X):
            if mode not in {"numeric", "sparse"}:
                raise TypedInputCapabilityError("sparse_native_categorical_unavailable", "Sparse matrices cannot expose native categorical values.")
            return self._transform_sparse(X)
        if mode not in {"numeric", "native_categorical"}:
            raise ValueError("output_mode must be numeric, sparse, or native_categorical.")
        frame = self._frame(X)
        numeric_columns: list[np.ndarray] = []
        names: list[str] = []
        native_columns: dict[str, Any] = {}
        for feature in self.input_schema_.features:
            if feature.role in _EXCLUDED_ROLES:
                continue
            if feature.role in _NUMERIC_ROLES:
                values = self._numeric_values(frame, feature)
                numeric_columns.append(values)
                names.append(feature.name)
                native_columns[feature.name] = values
            elif feature.role is FeatureRole.CATEGORICAL:
                values = self._numeric_values(frame, feature)
                numeric_columns.append(values)
                names.append(feature.name)
                native_columns[feature.name] = self._native_categories(frame[feature.name], feature.name)
            elif feature.role is FeatureRole.TEXT:
                if str(self.text_encoding).strip().lower() == "drop":
                    continue
                for derived, values in self._text_values(frame, feature.name):
                    numeric_columns.append(values)
                    names.append(derived)
                    native_columns[derived] = values
        if tuple(names) != self.numeric_schema_.feature_names:
            raise RuntimeError("Transformed columns diverge from the fitted schema.")
        numeric = np.column_stack(numeric_columns).astype(float, copy=False)
        if mode == "numeric":
            return TransformedInput(
                X=numeric,
                schema=self.numeric_schema_,
                source_schema=self.input_schema_,
                output_mode="numeric",
                metadata=_record_items({"fit_rows": int(self.fit_row_count_), "unknown_category_code": float(self.unknown_category_code), "missing_category_code": float(self.missing_category_code)}),
            )
        assert pd is not None
        native = pd.DataFrame(native_columns, index=frame.index).loc[:, list(self.native_schema_.feature_names)]
        return TransformedInput(
            X=native,
            schema=self.native_schema_,
            source_schema=self.input_schema_,
            output_mode="native_categorical",
            metadata=_record_items({
                "fit_rows": int(self.fit_row_count_),
                "unknown_category_token": str(self.unknown_category_token),
                "missing_category_token": str(self.missing_category_token),
                "native_categorical_columns": tuple(feature.name for feature in self.native_schema_.features if feature.role is FeatureRole.CATEGORICAL),
            }),
        )

    def transform(self, X: Any) -> Any:
        return self.transform_with_schema(X, output_mode="numeric").X

    def fit_transform_with_schema(
        self,
        X: Any,
        y: Any = None,
        *,
        schema: DatasetSchema | None = None,
        output_mode: str = "numeric",
    ) -> TransformedInput:
        return self.fit(X, y=y, schema=schema).transform_with_schema(X, schema=schema, output_mode=output_mode)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        self._require_fitted()
        if input_features is not None and tuple(str(value) for value in input_features) != self.input_schema_.feature_names:
            raise SchemaContractError("input_features do not match the fitted input schema.")
        return np.asarray(self._output_feature_names, dtype=object)

    def get_output_schema(self, *, output_mode: str = "numeric") -> DatasetSchema:
        self._require_fitted()
        mode = str(output_mode).strip().lower()
        if mode in {"numeric", "sparse"}:
            return self.numeric_schema_
        if mode == "native_categorical":
            return self.native_schema_
        raise ValueError("output_mode must be numeric, sparse, or native_categorical.")

    def native_stage2_bridge(self) -> NativeCategoricalStage2Bridge:
        """Return a fail-closed numeric-to-native column correspondence.

        Numeric feature selection can only hand off positions through this
        bridge.  Names alone are not sufficient: source names and lineage must
        agree, and the only permitted lineage divergence is ordinal encoding
        paired with the matching train-only native categorical representation.
        """

        self._require_fitted()
        if bool(self._sparse_input_):
            raise TypedInputCapabilityError(
                "native_stage2_sparse_unavailable",
                "Native categorical Stage-2 routing cannot be built for sparse input.",
                diagnostics={
                    "source_schema_fingerprint": self.input_schema_.fingerprint,
                    "numeric_schema_fingerprint": self.numeric_schema_.fingerprint,
                },
            )

        numeric_names = tuple(self.numeric_schema_.feature_names)
        native_names = tuple(self.native_schema_.feature_names)
        if len(set(numeric_names)) != len(numeric_names) or len(set(native_names)) != len(native_names):
            raise TypedInputCapabilityError(
                "native_stage2_duplicate_schema_columns",
                "Native categorical Stage-2 routing requires unique numeric and native schema columns.",
                diagnostics={
                    "numeric_feature_names": list(numeric_names),
                    "native_feature_names": list(native_names),
                },
            )
        native_positions = {name: index for index, name in enumerate(native_names)}
        numeric_lineage = {
            record.output_name: record for record in self.numeric_schema_.lineage
        }
        native_lineage = {
            record.output_name: record for record in self.native_schema_.lineage
        }
        mapping: list[int] = []
        categorical_columns: list[str] = []
        vocabularies: list[tuple[str, tuple[str, ...]]] = []
        for numeric_position, numeric_feature in enumerate(self.numeric_schema_.features):
            name = str(numeric_feature.name)
            native_position = native_positions.get(name)
            if native_position is None:
                raise TypedInputCapabilityError(
                    "native_stage2_schema_column_missing",
                    "A numeric feature-selection column has no paired native column.",
                    diagnostics={
                        "numeric_position": int(numeric_position),
                        "numeric_name": name,
                        "numeric_schema_fingerprint": self.numeric_schema_.fingerprint,
                        "native_schema_fingerprint": self.native_schema_.fingerprint,
                    },
                )
            native_feature = self.native_schema_.features[int(native_position)]
            numeric_record = numeric_lineage.get(name)
            native_record = native_lineage.get(name)
            if numeric_record is None or native_record is None:
                raise TypedInputCapabilityError(
                    "native_stage2_lineage_missing",
                    "A numeric/native Stage-2 column is missing immutable lineage.",
                    diagnostics={"feature": name},
                )
            lineage_identity_matches = bool(
                tuple(numeric_record.input_names) == tuple(native_record.input_names)
                and str(numeric_record.source_schema_hash)
                == str(native_record.source_schema_hash)
                and str(numeric_record.output_name) == str(native_record.output_name)
                and str(numeric_feature.source_name) == str(native_feature.source_name)
            )
            native_category_pair = bool(
                native_feature.role is FeatureRole.CATEGORICAL
                and str(numeric_record.operation) == "ordinal_encode_train_only"
                and str(native_record.operation) == "native_categorical_train_only"
            )
            same_lineage = bool(
                str(numeric_record.operation) == str(native_record.operation)
                and tuple(numeric_record.parameters) == tuple(native_record.parameters)
            )
            if not lineage_identity_matches or not (same_lineage or native_category_pair):
                raise TypedInputCapabilityError(
                    "native_stage2_lineage_mismatch",
                    "Numeric and native Stage-2 columns do not have an approved shared lineage.",
                    diagnostics={
                        "feature": name,
                        "numeric_operation": str(numeric_record.operation),
                        "native_operation": str(native_record.operation),
                        "numeric_source_name": str(numeric_feature.source_name),
                        "native_source_name": str(native_feature.source_name),
                    },
                )
            mapping.append(int(native_position))
            if native_feature.role is FeatureRole.CATEGORICAL:
                levels = tuple(
                    str(value)
                    for value in (
                        *self.native_category_levels_.get(name, tuple()),
                        str(self.unknown_category_token),
                        str(self.missing_category_token),
                    )
                )
                if len(set(levels)) != len(levels):
                    raise TypedInputCapabilityError(
                        "native_stage2_category_vocabulary_duplicate",
                        "Native categorical vocabulary contains duplicate category tokens.",
                        diagnostics={"feature": name, "categories": list(levels)},
                    )
                categorical_columns.append(name)
                vocabularies.append((name, levels))

        return NativeCategoricalStage2Bridge(
            source_schema_fingerprint=str(self.input_schema_.fingerprint),
            numeric_schema_fingerprint=str(self.numeric_schema_.fingerprint),
            native_schema_fingerprint=str(self.native_schema_.fingerprint),
            numeric_feature_names=numeric_names,
            native_feature_names=native_names,
            native_position_by_numeric_position=tuple(mapping),
            categorical_native_columns=tuple(categorical_columns),
            category_vocabularies=tuple(vocabularies),
        )

    def select_native_stage2_view(
        self,
        transformed: TransformedInput,
        *,
        bridge: NativeCategoricalStage2Bridge,
        selected_numeric_positions: Sequence[int],
    ) -> tuple[Any, Mapping[str, Any]]:
        """Select a native DataFrame using numeric-FS positions and validate it.

        The resulting DataFrame intentionally preserves pandas categorical
        dtypes.  It is suitable only for the explicitly admitted native Stage-2
        adapters; callers must not coerce it back into the numeric FS core.
        """

        self._require_fitted()
        if not isinstance(bridge, NativeCategoricalStage2Bridge):
            raise TypeError("bridge must be a NativeCategoricalStage2Bridge.")
        if str(transformed.output_mode) != "native_categorical":
            raise TypedInputCapabilityError(
                "native_stage2_view_mode_mismatch",
                "Native Stage-2 selection requires a native categorical transformed view.",
                diagnostics={"output_mode": str(transformed.output_mode)},
            )
        if str(transformed.schema.fingerprint) != str(bridge.native_schema_fingerprint):
            raise TypedInputCapabilityError(
                "native_stage2_view_schema_mismatch",
                "Native Stage-2 transformed data does not match the admitted native schema.",
                diagnostics={
                    "expected_native_schema_fingerprint": bridge.native_schema_fingerprint,
                    "observed_native_schema_fingerprint": transformed.schema.fingerprint,
                },
            )
        if str(transformed.source_schema.fingerprint) != str(
            bridge.source_schema_fingerprint
        ):
            raise TypedInputCapabilityError(
                "native_stage2_view_source_schema_mismatch",
                "Native Stage-2 transformed data does not match the admitted source schema.",
                diagnostics={
                    "expected_source_schema_fingerprint": bridge.source_schema_fingerprint,
                    "observed_source_schema_fingerprint": transformed.source_schema.fingerprint,
                },
            )
        if pd is None or not isinstance(transformed.X, pd.DataFrame):
            raise TypedInputCapabilityError(
                "native_stage2_view_not_dataframe",
                "Native Stage-2 routing requires a pandas DataFrame.",
                diagnostics={"observed_type": type(transformed.X).__name__},
            )
        actual_columns = tuple(str(value) for value in transformed.X.columns)
        if actual_columns != bridge.native_feature_names:
            raise TypedInputCapabilityError(
                "native_stage2_view_column_order_mismatch",
                "Native Stage-2 DataFrame columns differ from the admitted schema order.",
                diagnostics={
                    "expected_columns": list(bridge.native_feature_names),
                    "observed_columns": list(actual_columns),
                },
            )

        positions: list[int] = []
        for raw_position in selected_numeric_positions:
            if isinstance(raw_position, (bool, np.bool_)):
                raise TypedInputCapabilityError(
                    "native_stage2_selection_non_integer",
                    "Native Stage-2 selection positions must be integer indices.",
                    diagnostics={"position": repr(raw_position)},
                )
            try:
                position = int(raw_position)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypedInputCapabilityError(
                    "native_stage2_selection_non_integer",
                    "Native Stage-2 selection positions must be integer indices.",
                    diagnostics={"position": repr(raw_position)},
                ) from exc
            if position != raw_position:
                raise TypedInputCapabilityError(
                    "native_stage2_selection_non_integer",
                    "Native Stage-2 selection positions must be integer indices.",
                    diagnostics={"position": repr(raw_position)},
                )
            positions.append(position)
        if not positions:
            raise TypedInputCapabilityError(
                "native_stage2_selection_empty",
                "Native Stage-2 routing requires at least one selected numeric feature.",
            )
        if len(set(positions)) != len(positions):
            raise TypedInputCapabilityError(
                "native_stage2_selection_duplicate",
                "Native Stage-2 routing refuses duplicate selected numeric positions.",
                diagnostics={"selected_numeric_positions": list(positions)},
            )
        invalid = [
            int(position)
            for position in positions
            if not 0 <= int(position) < len(bridge.numeric_feature_names)
        ]
        if invalid:
            raise TypedInputCapabilityError(
                "native_stage2_selection_out_of_range",
                "Native Stage-2 routing received an out-of-range numeric selection position.",
                diagnostics={
                    "invalid_positions": invalid,
                    "n_numeric_features": len(bridge.numeric_feature_names),
                },
            )
        native_positions = tuple(
            int(bridge.native_position_by_numeric_position[position])
            for position in positions
        )
        selected_columns = tuple(
            bridge.native_feature_names[position] for position in native_positions
        )
        if len(set(selected_columns)) != len(selected_columns):
            raise TypedInputCapabilityError(
                "native_stage2_selection_native_duplicate",
                "Native Stage-2 position mapping produced duplicate native columns.",
                diagnostics={"selected_native_columns": list(selected_columns)},
            )
        view = transformed.X.loc[:, list(selected_columns)].copy()
        selected_categories = tuple(
            name for name in bridge.categorical_native_columns if name in selected_columns
        )
        vocabularies = bridge.category_vocabulary_map
        for name in selected_categories:
            dtype = view[name].dtype
            if not isinstance(dtype, pd.CategoricalDtype):
                raise TypedInputCapabilityError(
                    "native_stage2_category_dtype_mismatch",
                    "Native Stage-2 categorical columns must retain pandas CategoricalDtype.",
                    diagnostics={"feature": name, "dtype": str(dtype)},
                )
            observed = tuple(str(value) for value in view[name].cat.categories.tolist())
            expected = tuple(vocabularies[name])
            if observed != expected:
                raise TypedInputCapabilityError(
                    "native_stage2_category_vocabulary_mismatch",
                    "Native Stage-2 categorical vocabulary differs from the fitted train-fold vocabulary.",
                    diagnostics={
                        "feature": name,
                        "expected_categories": list(expected),
                        "observed_categories": list(observed),
                    },
                )
        record = {
            "source_schema_fingerprint": bridge.source_schema_fingerprint,
            "numeric_schema_fingerprint": bridge.numeric_schema_fingerprint,
            "native_schema_fingerprint": bridge.native_schema_fingerprint,
            "selected_numeric_positions": [int(value) for value in positions],
            "selected_numeric_columns": [
                bridge.numeric_feature_names[position] for position in positions
            ],
            "selected_native_positions": [int(value) for value in native_positions],
            "selected_native_columns": list(selected_columns),
            "selected_categorical_columns": list(selected_categories),
            "selected_category_vocabularies": {
                name: list(vocabularies[name]) for name in selected_categories
            },
        }
        return view, MappingProxyType(record)

    def transform_for_classifier(
        self,
        X: Any,
        *,
        classifier_name: str,
        dependency_facts: Mapping[str, bool | None] | None = None,
        builder_facts: Mapping[str, bool | None] | None = None,
        capability_overrides: ClassifierCapabilityOverrides | None = None,
    ) -> TransformedInput:
        """Return an estimator-specific view only after concrete registry admission.

        Conditional native routes remain unavailable unless the caller supplies
        a concrete observed adapter override.  The pipeline uses that narrow
        contract for the explicit singleton Stage-2 route only.
        """

        self._require_fitted()
        has_categorical = any(feature.role is FeatureRole.CATEGORICAL for feature in self.input_schema_.features)
        try:
            resolved = resolve_classifier_capabilities(
                str(classifier_name),
                runtime=ClassifierRuntimeFacts(
                    input_is_sparse=bool(self._sparse_input_),
                    input_has_nan=False,
                    input_has_categorical=bool(has_categorical),
                ),
                dependency_facts=dependency_facts,
                builder_facts=builder_facts,
                overrides=capability_overrides,
            )
        except KeyError as exc:
            raise TypedInputCapabilityError(
                "native_categorical_classifier_unknown",
                "Native categorical transformation requires a registered classifier.",
                diagnostics={"classifier_name": str(classifier_name)},
            ) from exc
        if not resolved.is_available:
            raise TypedInputCapabilityError(
                "native_categorical_classifier_unavailable",
                "Native categorical transformation requires a concretely admitted "
                "classifier route.",
                diagnostics={
                    "classifier_name": str(classifier_name),
                    "canonical_name": str(resolved.canonical_name),
                    "availability": resolved.availability.value,
                    "categorical_input": resolved.categorical_input.value,
                    "availability_reasons": list(resolved.availability_reasons),
                },
            )
        native_allowed = bool(
            has_categorical
            and not self._sparse_input_
            and resolved.categorical_input is SupportLevel.SUPPORTED
        )
        if has_categorical and not native_allowed:
            raise TypedInputCapabilityError(
                "native_categorical_classifier_unavailable",
                "The admitted classifier does not provide a concrete native "
                "categorical input contract.",
                diagnostics={
                    "classifier_name": str(classifier_name),
                    "canonical_name": str(resolved.canonical_name),
                    "categorical_input": resolved.categorical_input.value,
                },
            )
        transformed = self.transform_with_schema(X, output_mode="native_categorical" if native_allowed else "numeric")
        metadata = dict(transformed.metadata_dict)
        metadata.update({
            "classifier_name": str(classifier_name),
            "classifier_categorical_capability": resolved.categorical_input.value,
            "classifier_input_route": "native_categorical" if native_allowed else "numeric_adapter",
            "classifier_availability": resolved.availability.value,
            "classifier_availability_reasons": tuple(resolved.availability_reasons),
        })
        return TransformedInput(
            X=transformed.X,
            schema=transformed.schema,
            source_schema=transformed.source_schema,
            output_mode=transformed.output_mode,
            metadata=_record_items(metadata),
        )


__all__ = [
    "FeatureSelectorRuntimeFacts",
    "FoldLocalPreprocessor",
    "NativeCategoricalStage2Bridge",
    "ResolvedFeatureSelectorCapabilities",
    "TransformedInput",
    "TypedInputCapabilityError",
    "admit_feature_selector_methods",
    "guarded_sparse_to_dense",
    "is_sparse_input",
    "is_typed_input",
    "resolve_feature_selector_capabilities",
]

"""Immutable row and resampling contracts for leakage-safe pipeline fitting.

The objects in this module deliberately retain only immutable Python values.
Caller-provided splitters are materialized immediately; mutable splitter objects
and NumPy arrays are never stored on a context or split plan.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Any, Mapping, NoReturn, Optional, Sequence

import numpy as np
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


RESAMPLING_SCHEMA_VERSION = "1.0"

_IDENTITY_FIELDS = ("groups", "patient_ids", "site_ids", "batch_ids")
_POLICY_KINDS = {
    "iid",
    "stratified",
    "group",
    "stratified_group",
    "blocked_temporal",
    "supplied",
}


class ResamplingContractError(ValueError):
    """Raised when a declared resampling contract cannot be honored."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.diagnostics = _json_safe_mapping(diagnostics or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": str(self),
            "diagnostics": dict(self.diagnostics),
        }


def _raise_contract(
    code: str,
    message: str,
    **diagnostics: Any,
) -> NoReturn:
    raise ResamplingContractError(
        message,
        code=code,
        diagnostics=diagnostics,
    )


def _json_safe_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or isinstance(value, (str, int, bool)):
            out[str(key)] = value
        elif isinstance(value, float):
            out[str(key)] = value if math.isfinite(value) else str(value)
        elif isinstance(value, Mapping):
            out[str(key)] = _json_safe_mapping(value)
        elif isinstance(value, (tuple, list)):
            out[str(key)] = [
                _json_safe_mapping(v) if isinstance(v, Mapping) else v
                for v in value
            ]
        else:
            out[str(key)] = str(value)
    return out


def _python_scalar(value: Any, *, field_name: str) -> Any:
    if isinstance(value, np.generic):
        if isinstance(value, np.datetime64):
            if np.isnat(value):
                return None
            return value.astype("datetime64[us]").item()
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, bytes, bool, int, float, Decimal, datetime, date, time)):
        return value
    _raise_contract(
        "unsupported_scalar",
        f"{field_name} contains unsupported value type {type(value).__module__}.{type(value).__qualname__}.",
        field=field_name,
        value_type=f"{type(value).__module__}.{type(value).__qualname__}",
    )


def _freeze_vector(
    values: Optional[Sequence[Any]],
    *,
    field_name: str,
    n_rows: int,
    allow_empty: bool = True,
) -> tuple[Any, ...]:
    if values is None:
        return tuple()
    if isinstance(values, np.ndarray):
        raw = np.asarray(values, dtype=object).ravel().tolist()
    else:
        raw = list(values)
    if not raw and allow_empty:
        return tuple()
    if len(raw) != int(n_rows):
        _raise_contract(
            "row_vector_length_mismatch",
            f"{field_name} has {len(raw)} values but the context has {n_rows} rows.",
            field=field_name,
            expected_rows=int(n_rows),
            observed_rows=int(len(raw)),
        )
    return tuple(_python_scalar(value, field_name=field_name) for value in raw)


def coerce_sample_weights(
    values: Optional[Sequence[Any]],
    *,
    n_rows: int,
    field_name: str = "sample_weights",
    require_positive_mass: bool = False,
) -> tuple[float, ...]:
    """Freeze an aligned non-negative finite sample-weight vector.

    Empty/``None`` input means weights were not requested.  Individual zero
    weights are valid, but callers fitting or scoring a partition can require
    positive total mass explicitly.
    """

    if values is None:
        return tuple()
    try:
        raw = (
            np.asarray(values, dtype=object).ravel().tolist()
            if isinstance(values, np.ndarray)
            else list(values)
        )
    except TypeError as exc:
        _raise_contract(
            "invalid_sample_weight",
            f"{field_name} must be a one-dimensional collection of numeric scalars.",
            field=field_name,
            n_rows=int(n_rows),
            reason=type(exc).__name__,
        )
    if not raw:
        return tuple()
    if len(raw) != int(n_rows):
        _raise_contract(
            "row_vector_length_mismatch",
            f"{field_name} has {len(raw)} values but the context has {n_rows} rows.",
            field=field_name,
            expected_rows=int(n_rows),
            observed_rows=int(len(raw)),
        )
    try:
        weights = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        _raise_contract(
            "invalid_sample_weight",
            f"{field_name} must contain numeric scalars.",
            field=field_name,
            n_rows=int(n_rows),
            reason=type(exc).__name__,
        )
    nonfinite = sum(not math.isfinite(value) for value in weights)
    if nonfinite:
        _raise_contract(
            "nonfinite_sample_weight",
            f"{field_name} must contain only finite values.",
            field=field_name,
            n_rows=int(n_rows),
            nonfinite_count=int(nonfinite),
        )
    negative = sum(value < 0.0 for value in weights)
    if negative:
        _raise_contract(
            "negative_sample_weight",
            f"{field_name} must be non-negative.",
            field=field_name,
            n_rows=int(n_rows),
            negative_count=int(negative),
        )
    try:
        total_mass = math.fsum(weights)
    except OverflowError:
        total_mass = float("inf")
    if not math.isfinite(total_mass):
        _raise_contract(
            "nonfinite_sample_weight_mass",
            f"{field_name} has non-finite total weight mass.",
            field=field_name,
            n_rows=int(n_rows),
        )
    if require_positive_mass and not total_mass > 0.0:
        _raise_contract(
            "zero_sample_weight_mass",
            f"{field_name} must have positive total weight mass for this operation.",
            field=field_name,
            n_rows=int(n_rows),
        )
    return weights


def _type_name(value: Any) -> tuple[str, str]:
    value_type = type(value)
    return value_type.__module__, value_type.__qualname__


def _typed_record(value: Any) -> dict[str, Any]:
    """Return a canonical, type-preserving JSON record for one scalar."""

    value = _python_scalar(value, field_name="canonical_scalar")
    if value is None:
        return {"module": "builtins", "qualname": "NoneType", "value": None}
    module, qualname = _type_name(value)
    if isinstance(value, bytes):
        encoded: Any = base64.b64encode(value).decode("ascii")
    elif isinstance(value, float):
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "+inf" if value > 0 else "-inf"
        else:
            encoded = value.hex()
    elif isinstance(value, Decimal):
        encoded = format(value, "f")
    elif isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        encoded = value.isoformat()
    elif isinstance(value, (date, time)):
        encoded = value.isoformat()
    else:
        encoded = value
    return {"module": module, "qualname": qualname, "value": encoded}


def typed_scalar_key(value: Any) -> tuple[str, str, str]:
    """Return a stable key that never collapses differently typed scalar values."""

    record = _typed_record(value)
    payload = json.dumps(
        record["value"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return str(record["module"]), str(record["qualname"]), payload


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _vector_record(values: Sequence[Any]) -> list[dict[str, Any]]:
    return [_typed_record(value) for value in values]


def _typed_counts(values: Sequence[Any]) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    for value in values:
        key = typed_scalar_key(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _homogeneous_sklearn_labels(values: Sequence[Any]) -> Optional[np.ndarray]:
    """Return legacy-compatible labels when sklearn can preserve their types."""

    frozen = tuple(_python_scalar(v, field_name="y") for v in values)
    nonmissing = [v for v in frozen if v is not None]
    if len(nonmissing) != len(frozen) or not nonmissing:
        return None
    first_type = type(nonmissing[0])
    if not all(type(v) is first_type for v in nonmissing):
        return None
    if first_type not in {bool, int, float, str, bytes}:
        return None
    return np.asarray(frozen)


def _stratification_labels(values: Sequence[Any]) -> np.ndarray:
    """Encode mixed typed labels without merging equal-looking Python values."""

    legacy = _homogeneous_sklearn_labels(values)
    if legacy is not None:
        return legacy
    keys = [typed_scalar_key(value) for value in values]
    ordered = {key: index for index, key in enumerate(sorted(set(keys)))}
    return np.asarray([ordered[key] for key in keys], dtype=int)


def _normalise_indices(values: Sequence[int], *, field_name: str) -> tuple[int, ...]:
    if isinstance(values, np.ndarray):
        raw = np.asarray(values).ravel().tolist()
    else:
        raw = list(values)
    out: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            _raise_contract(
                "invalid_split_index",
                f"{field_name} contains a boolean index.",
                field=field_name,
            )
        try:
            index = int(value)
        except (TypeError, ValueError):
            _raise_contract(
                "invalid_split_index",
                f"{field_name} contains a non-integer index.",
                field=field_name,
                value_type=f"{type(value).__module__}.{type(value).__qualname__}",
            )
        if isinstance(value, (float, np.floating)) and float(value) != float(index):
            _raise_contract(
                "invalid_split_index",
                f"{field_name} contains non-integral index {value!r}.",
                field=field_name,
            )
        out.append(index)
    return tuple(out)


@dataclass(frozen=True)
class ResamplingPolicy:
    """Declared policy governing every score-bearing split in one fit."""

    kind: str = "iid"
    enforced_boundaries: tuple[str, ...] = field(default_factory=tuple)
    time_field: str = "timestamps"
    require_class_coverage: Optional[bool] = None
    require_full_coverage: bool = True
    temporal_gap: int = 0
    enforce_temporal_order: bool = False

    def __post_init__(self) -> None:
        kind = str(self.kind or "iid").strip().lower()
        if kind not in _POLICY_KINDS:
            _raise_contract(
                "unknown_resampling_policy",
                f"Unknown resampling policy {self.kind!r}.",
                policy=str(self.kind),
                supported=tuple(sorted(_POLICY_KINDS)),
            )
        boundaries = tuple(str(value).strip() for value in self.enforced_boundaries)
        invalid = tuple(value for value in boundaries if value not in _IDENTITY_FIELDS)
        if invalid:
            _raise_contract(
                "unknown_identity_boundary",
                f"Unknown identity boundary fields: {invalid!r}.",
                invalid_fields=invalid,
                supported_fields=_IDENTITY_FIELDS,
            )
        if len(set(boundaries)) != len(boundaries):
            _raise_contract(
                "duplicate_identity_boundary",
                "enforced_boundaries contains duplicate field names.",
                fields=boundaries,
            )
        if kind in {"group", "stratified_group"} and not boundaries:
            _raise_contract(
                "group_boundary_required",
                f"Policy {kind!r} requires at least one enforced identity boundary.",
                policy=kind,
            )
        gap = int(self.temporal_gap)
        if gap < 0:
            _raise_contract(
                "invalid_temporal_gap",
                "temporal_gap must be non-negative.",
                temporal_gap=gap,
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "enforced_boundaries", boundaries)
        object.__setattr__(self, "time_field", str(self.time_field or "timestamps"))
        object.__setattr__(self, "temporal_gap", gap)

    @property
    def is_structured(self) -> bool:
        return self.kind in {"group", "stratified_group", "blocked_temporal", "supplied"}

    @property
    def stratified(self) -> bool:
        return self.kind in {"stratified", "stratified_group"}

    @property
    def class_coverage_required(self) -> bool:
        if self.require_class_coverage is not None:
            return bool(self.require_class_coverage)
        return self.kind in {"stratified", "stratified_group"}

    @property
    def temporal_order_required(self) -> bool:
        return self.kind == "blocked_temporal" or bool(self.enforce_temporal_order)

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "enforced_boundaries": list(self.enforced_boundaries),
            "time_field": self.time_field,
            "require_class_coverage": self.require_class_coverage,
            "require_full_coverage": bool(self.require_full_coverage),
            "temporal_gap": int(self.temporal_gap),
            "enforce_temporal_order": bool(self.enforce_temporal_order),
        }


@dataclass(frozen=True)
class SplitAssignment:
    """One immutable, positional train/test assignment."""

    scope: str
    split_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    source: str = "resolved"
    allow_unassigned: bool = False
    parent_context_fingerprint: Optional[str] = None
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", str(self.scope or "split"))
        object.__setattr__(self, "split_id", str(self.split_id or "split-0"))
        object.__setattr__(
            self,
            "train_indices",
            _normalise_indices(self.train_indices, field_name="train_indices"),
        )
        object.__setattr__(
            self,
            "test_indices",
            _normalise_indices(self.test_indices, field_name="test_indices"),
        )
        object.__setattr__(self, "source", str(self.source or "resolved"))
        if self.parent_context_fingerprint is not None:
            object.__setattr__(
                self,
                "parent_context_fingerprint",
                str(self.parent_context_fingerprint),
            )
        frozen_meta: list[tuple[str, Any]] = []
        raw_metadata = (
            self.metadata.items()
            if isinstance(self.metadata, Mapping)
            else self.metadata
        )
        for key, value in raw_metadata:
            frozen_meta.append(
                (str(key), _python_scalar(value, field_name=f"metadata.{key}"))
            )
        object.__setattr__(self, "metadata", tuple(sorted(frozen_meta)))

    def to_record(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "split_id": self.split_id,
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
            "source": self.source,
            "allow_unassigned": bool(self.allow_unassigned),
            "parent_context_fingerprint": self.parent_context_fingerprint,
            "metadata": [
                {"key": key, "value": _typed_record(value)}
                for key, value in self.metadata
            ],
        }


@dataclass(frozen=True)
class LeakageAudit:
    """PII-free result of validating one split against its row contract."""

    ok: bool
    reason: str
    n_rows: int
    n_train: int
    n_test: int
    n_unassigned: int
    row_overlap_count: int
    identity_overlap_counts: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    temporal_order_ok: Optional[bool] = None
    class_support_ok: Optional[bool] = None
    train_class_count: int = 0
    test_class_count: int = 0
    full_class_count: int = 0
    diagnostics: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "reason": str(self.reason),
            "n_rows": int(self.n_rows),
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "n_unassigned": int(self.n_unassigned),
            "row_overlap_count": int(self.row_overlap_count),
            "identity_overlap_counts": {
                field_name: int(count)
                for field_name, count in self.identity_overlap_counts
            },
            "temporal_order_ok": self.temporal_order_ok,
            "class_support_ok": self.class_support_ok,
            "train_class_count": int(self.train_class_count),
            "test_class_count": int(self.test_class_count),
            "full_class_count": int(self.full_class_count),
            "diagnostics": {
                key: value for key, value in self.diagnostics
            },
        }


@dataclass(frozen=True)
class ResolvedSplit:
    """A validated assignment with immutable provenance."""

    assignment: SplitAssignment
    fingerprint: str
    audit: LeakageAudit

    @property
    def train_indices(self) -> tuple[int, ...]:
        return self.assignment.train_indices

    @property
    def test_indices(self) -> tuple[int, ...]:
        return self.assignment.test_indices

    def to_metadata(self) -> dict[str, Any]:
        return {
            "split_id": self.assignment.split_id,
            "scope": self.assignment.scope,
            "source": self.assignment.source,
            "fingerprint": self.fingerprint,
            "audit": self.audit.to_dict(),
        }


@dataclass(frozen=True)
class ResolvedSplitPlan:
    """One or more resolver-produced splits for a named purpose."""

    purpose: str
    policy: ResamplingPolicy
    context_fingerprint: str
    splits: tuple[ResolvedSplit, ...]
    fingerprint: str

    @property
    def primary(self) -> ResolvedSplit:
        if not self.splits:
            _raise_contract(
                "empty_split_plan",
                f"Split plan {self.purpose!r} contains no assignments.",
                purpose=self.purpose,
            )
        return self.splits[0]

    def index_pairs(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        return tuple(
            (split.train_indices, split.test_indices) for split in self.splits
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": RESAMPLING_SCHEMA_VERSION,
            "purpose": self.purpose,
            "policy": self.policy.to_record(),
            "context_fingerprint": self.context_fingerprint,
            "plan_fingerprint": self.fingerprint,
            "n_splits": len(self.splits),
            "splits": [split.to_metadata() for split in self.splits],
        }


@dataclass(frozen=True)
class FitResamplingContext:
    """Immutable row metadata and supplied assignments for one row universe."""

    n_rows: int
    row_ids: tuple[Any, ...] = field(default_factory=tuple)
    groups: tuple[Any, ...] = field(default_factory=tuple)
    patient_ids: tuple[Any, ...] = field(default_factory=tuple)
    site_ids: tuple[Any, ...] = field(default_factory=tuple)
    batch_ids: tuple[Any, ...] = field(default_factory=tuple)
    timestamps: tuple[Any, ...] = field(default_factory=tuple)
    sample_weights: tuple[float, ...] = field(default_factory=tuple)
    policy: ResamplingPolicy = field(default_factory=ResamplingPolicy)
    supplied_splits: tuple[SplitAssignment, ...] = field(default_factory=tuple)
    parent_split_fingerprint: Optional[str] = None
    schema_version: str = RESAMPLING_SCHEMA_VERSION
    _base_fingerprint: str = field(init=False, repr=False)
    _fingerprint: str = field(init=False, repr=False)
    _row_ids_fingerprint: str = field(init=False, repr=False)
    _policy_fingerprint: str = field(init=False, repr=False)
    _sample_weights_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n_rows = int(self.n_rows)
        if n_rows < 2:
            _raise_contract(
                "insufficient_rows",
                "A fit/resampling context requires at least two rows.",
                n_rows=n_rows,
            )
        object.__setattr__(self, "n_rows", n_rows)
        if not isinstance(self.policy, ResamplingPolicy):
            if isinstance(self.policy, Mapping):
                object.__setattr__(self, "policy", ResamplingPolicy(**dict(self.policy)))
            else:
                _raise_contract(
                    "invalid_resampling_policy",
                    "policy must be a ResamplingPolicy or mapping.",
                    policy_type=f"{type(self.policy).__module__}.{type(self.policy).__qualname__}",
                )

        row_ids = self.row_ids
        if len(row_ids) == 0:
            row_ids = tuple(range(n_rows))
        row_ids = _freeze_vector(
            row_ids,
            field_name="row_ids",
            n_rows=n_rows,
            allow_empty=False,
        )
        row_keys = [typed_scalar_key(value) for value in row_ids]
        if len(set(row_keys)) != n_rows:
            _raise_contract(
                "duplicate_row_ids",
                "row_ids must identify rows uniquely using type-preserving equality.",
                n_rows=n_rows,
                unique_row_ids=len(set(row_keys)),
            )
        object.__setattr__(self, "row_ids", row_ids)

        for field_name in (*_IDENTITY_FIELDS, "timestamps"):
            frozen = _freeze_vector(
                getattr(self, field_name),
                field_name=field_name,
                n_rows=n_rows,
            )
            object.__setattr__(self, field_name, frozen)

        weights = coerce_sample_weights(
            self.sample_weights,
            n_rows=n_rows,
            field_name="sample_weights",
        )
        object.__setattr__(self, "sample_weights", weights)

        for field_name in self.policy.enforced_boundaries:
            values = getattr(self, field_name)
            if not values:
                _raise_contract(
                    "missing_identity_boundary",
                    f"Policy requires {field_name!r}, but the context has no values for it.",
                    field=field_name,
                    policy=self.policy.kind,
                )
            missing = sum(value is None for value in values)
            if missing:
                _raise_contract(
                    "missing_identity_value",
                    f"Enforced identity field {field_name!r} contains missing values.",
                    field=field_name,
                    missing_count=missing,
                )
            nonfinite = sum(
                isinstance(value, float) and not math.isfinite(value)
                for value in values
            )
            if nonfinite:
                _raise_contract(
                    "nonfinite_identity_value",
                    f"Enforced identity field {field_name!r} contains non-finite values.",
                    field=field_name,
                    nonfinite_count=nonfinite,
                )
        if self.policy.temporal_order_required:
            if self.policy.time_field != "timestamps":
                _raise_contract(
                    "unknown_time_field",
                    "The current context exposes temporal values through 'timestamps'.",
                    requested_time_field=self.policy.time_field,
                )
            if not self.timestamps:
                _raise_contract(
                    "missing_timestamps",
                    "blocked_temporal policy requires a timestamp for every row.",
                    n_rows=n_rows,
                )
            missing = sum(value is None for value in self.timestamps)
            if missing:
                _raise_contract(
                    "missing_timestamps",
                    "blocked_temporal policy does not permit missing timestamps.",
                    missing_count=missing,
                    n_rows=n_rows,
                )

        supplied = tuple(self.supplied_splits or tuple())
        for split in supplied:
            if not isinstance(split, SplitAssignment):
                _raise_contract(
                    "invalid_supplied_split",
                    "supplied_splits must contain SplitAssignment objects.",
                    split_type=f"{type(split).__module__}.{type(split).__qualname__}",
                )
        object.__setattr__(self, "supplied_splits", supplied)
        object.__setattr__(self, "schema_version", str(self.schema_version))
        if self.parent_split_fingerprint is not None:
            object.__setattr__(
                self,
                "parent_split_fingerprint",
                str(self.parent_split_fingerprint),
            )

        base_record = self._record(include_supplied=False)
        base_fingerprint = _sha256(base_record)
        object.__setattr__(self, "_base_fingerprint", base_fingerprint)
        object.__setattr__(
            self,
            "_fingerprint",
            _sha256(self._record(include_supplied=True)),
        )
        object.__setattr__(
            self,
            "_row_ids_fingerprint",
            _sha256(
                {
                    "schema_version": self.schema_version,
                    "row_ids": _vector_record(self.row_ids),
                }
            ),
        )
        object.__setattr__(
            self,
            "_policy_fingerprint",
            _sha256(
                {
                    "schema_version": self.schema_version,
                    "policy": self.policy.to_record(),
                }
            ),
        )
        object.__setattr__(
            self,
            "_sample_weights_fingerprint",
            _sha256(
                {
                    "schema_version": self.schema_version,
                    "sample_weights": [
                        _typed_record(value) for value in self.sample_weights
                    ],
                }
            ),
        )

    @classmethod
    def iid(
        cls,
        n_rows: int,
        *,
        row_ids: Optional[Sequence[Any]] = None,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> "FitResamplingContext":
        return cls(
            n_rows=int(n_rows),
            row_ids=tuple() if row_ids is None else tuple(row_ids),
            sample_weights=(
                tuple() if sample_weights is None else tuple(sample_weights)
            ),
            policy=ResamplingPolicy(kind="iid"),
        )

    @property
    def base_fingerprint(self) -> str:
        return self._base_fingerprint

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def row_ids_fingerprint(self) -> str:
        """SHA-256 identity of the ordered row universe without raw IDs."""

        return self._row_ids_fingerprint

    @property
    def policy_fingerprint(self) -> str:
        """SHA-256 identity of the declared resampling policy."""

        return self._policy_fingerprint

    @property
    def sample_weights_fingerprint(self) -> str:
        """SHA-256 identity of the aligned weight vector without raw weights."""

        return self._sample_weights_fingerprint

    @property
    def is_structured(self) -> bool:
        return self.policy.is_structured

    def _record(self, *, include_supplied: bool) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "n_rows": int(self.n_rows),
            "row_ids": _vector_record(self.row_ids),
            "groups": _vector_record(self.groups),
            "patient_ids": _vector_record(self.patient_ids),
            "site_ids": _vector_record(self.site_ids),
            "batch_ids": _vector_record(self.batch_ids),
            "timestamps": _vector_record(self.timestamps),
            "sample_weights": [_typed_record(value) for value in self.sample_weights],
            "policy": self.policy.to_record(),
            "parent_split_fingerprint": self.parent_split_fingerprint,
        }
        if include_supplied:
            record["supplied_splits"] = [
                split.to_record() for split in self.supplied_splits
            ]
        return record

    def to_metadata(
        self,
        *,
        sample_weights_consumed: bool = False,
        sample_weight_usage: str = "not_consumed",
    ) -> dict[str, Any]:
        weights_present = bool(self.sample_weights)
        usage = str(sample_weight_usage or "not_consumed")
        if not weights_present:
            usage = "not_requested"
        return {
            "schema_version": self.schema_version,
            "n_rows": int(self.n_rows),
            "policy": self.policy.to_record(),
            "context_fingerprint": self.fingerprint,
            "base_context_fingerprint": self.base_fingerprint,
            "row_ids_fingerprint": self.row_ids_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "sample_weights_fingerprint": self.sample_weights_fingerprint,
            "parent_split_fingerprint": self.parent_split_fingerprint,
            "row_ids_present": bool(self.row_ids),
            "identity_fields_present": [
                field_name for field_name in _IDENTITY_FIELDS if getattr(self, field_name)
            ],
            "timestamps_present": bool(self.timestamps),
            "sample_weights_present": weights_present,
            "sample_weights_nonzero_count": int(
                sum(value > 0.0 for value in self.sample_weights)
            ),
            "sample_weights_total_mass": float(math.fsum(self.sample_weights)),
            "sample_weights_consumed": bool(
                weights_present and sample_weights_consumed
            ),
            "sample_weight_usage": usage,
            "n_supplied_splits": len(self.supplied_splits),
        }

    def take(
        self,
        indices: Sequence[int],
        *,
        parent_split_fingerprint: Optional[str] = None,
        policy: Optional[ResamplingPolicy] = None,
        supplied_splits: Optional[Sequence[SplitAssignment]] = None,
    ) -> "FitResamplingContext":
        positions = _normalise_indices(indices, field_name="context_indices")
        if not positions:
            _raise_contract(
                "empty_context_subset",
                "Cannot create a fit context from an empty row subset.",
            )
        if len(set(positions)) != len(positions):
            _raise_contract(
                "duplicate_context_index",
                "Context subset indices must be unique.",
                n_indices=len(positions),
                n_unique=len(set(positions)),
            )
        invalid = [index for index in positions if index < 0 or index >= self.n_rows]
        if invalid:
            _raise_contract(
                "context_index_out_of_bounds",
                "Context subset contains out-of-bounds row positions.",
                n_rows=self.n_rows,
                invalid_count=len(invalid),
                invalid_min=min(invalid),
                invalid_max=max(invalid),
            )

        def subset(values: tuple[Any, ...]) -> tuple[Any, ...]:
            return tuple(values[index] for index in positions) if values else tuple()

        resolved_policy = self.policy if policy is None else policy
        if supplied_splits is None:
            position_map = {
                source_position: child_position
                for child_position, source_position in enumerate(positions)
            }
            remapped: list[SplitAssignment] = []
            for assignment in self.supplied_splits:
                assignment_positions = set(assignment.train_indices).union(
                    assignment.test_indices
                )
                if not assignment_positions.issubset(position_map):
                    continue
                remapped.append(
                    replace(
                        assignment,
                        train_indices=tuple(
                            position_map[index]
                            for index in assignment.train_indices
                        ),
                        test_indices=tuple(
                            position_map[index]
                            for index in assignment.test_indices
                        ),
                        parent_context_fingerprint=None,
                    )
                )
            child_supplied = tuple(remapped)
        else:
            child_supplied = tuple(supplied_splits)

        unbound_child = FitResamplingContext(
            n_rows=len(positions),
            row_ids=subset(self.row_ids),
            groups=subset(self.groups),
            patient_ids=subset(self.patient_ids),
            site_ids=subset(self.site_ids),
            batch_ids=subset(self.batch_ids),
            timestamps=subset(self.timestamps),
            sample_weights=subset(self.sample_weights),
            policy=resolved_policy,
            supplied_splits=tuple(),
            parent_split_fingerprint=(
                self.parent_split_fingerprint
                if parent_split_fingerprint is None
                else str(parent_split_fingerprint)
            ),
        )
        bound_splits = tuple(
            replace(
                assignment,
                parent_context_fingerprint=unbound_child.base_fingerprint,
            )
            for assignment in child_supplied
        )
        if not bound_splits:
            return unbound_child
        return FitResamplingContext(
            n_rows=unbound_child.n_rows,
            row_ids=unbound_child.row_ids,
            groups=unbound_child.groups,
            patient_ids=unbound_child.patient_ids,
            site_ids=unbound_child.site_ids,
            batch_ids=unbound_child.batch_ids,
            timestamps=unbound_child.timestamps,
            sample_weights=unbound_child.sample_weights,
            policy=unbound_child.policy,
            supplied_splits=bound_splits,
            parent_split_fingerprint=unbound_child.parent_split_fingerprint,
        )

    def with_supplied_splits(
        self,
        assignments: Sequence[SplitAssignment],
        *,
        policy: Optional[ResamplingPolicy] = None,
    ) -> "FitResamplingContext":
        resolved_policy = (
            policy
            if policy is not None
            else replace(
                self.policy,
                kind="supplied",
            )
        )
        unbound_context = FitResamplingContext(
            n_rows=self.n_rows,
            row_ids=self.row_ids,
            groups=self.groups,
            patient_ids=self.patient_ids,
            site_ids=self.site_ids,
            batch_ids=self.batch_ids,
            timestamps=self.timestamps,
            sample_weights=self.sample_weights,
            policy=resolved_policy,
            supplied_splits=tuple(),
            parent_split_fingerprint=self.parent_split_fingerprint,
        )
        bound_assignments = tuple(
            replace(
                assignment,
                parent_context_fingerprint=(
                    unbound_context.base_fingerprint
                    if assignment.parent_context_fingerprint in {None, self.base_fingerprint}
                    else assignment.parent_context_fingerprint
                ),
            )
            for assignment in assignments
        )
        return FitResamplingContext(
            n_rows=self.n_rows,
            row_ids=self.row_ids,
            groups=self.groups,
            patient_ids=self.patient_ids,
            site_ids=self.site_ids,
            batch_ids=self.batch_ids,
            timestamps=self.timestamps,
            sample_weights=self.sample_weights,
            policy=resolved_policy,
            supplied_splits=bound_assignments,
            parent_split_fingerprint=self.parent_split_fingerprint,
        )

    def materialize_splitter(
        self,
        splitter: Any,
        *,
        y: Optional[Sequence[Any]] = None,
        scope: str = "inner_cv",
        source: str = "caller_splitter",
        allow_unassigned: bool = False,
    ) -> "FitResamplingContext":
        if not hasattr(splitter, "split") or not callable(splitter.split):
            _raise_contract(
                "invalid_supplied_splitter",
                "Supplied splitter must expose a callable split() method.",
                splitter_type=f"{type(splitter).__module__}.{type(splitter).__qualname__}",
            )
        if y is not None and len(y) != self.n_rows:
            _raise_contract(
                "label_length_mismatch",
                f"y has {len(y)} values but the context has {self.n_rows} rows.",
                n_rows=self.n_rows,
                y_rows=len(y),
            )
        x_dummy = np.zeros((self.n_rows, 1), dtype=np.uint8)
        y_values = None if y is None else _stratification_labels(y)
        groups = _component_labels(self) if self.policy.enforced_boundaries else None
        try:
            iterator = splitter.split(x_dummy, y_values, groups)
            assignments = tuple(
                SplitAssignment(
                    scope=scope,
                    split_id=f"{scope}-{index}",
                    train_indices=tuple(int(v) for v in np.asarray(train).ravel()),
                    test_indices=tuple(int(v) for v in np.asarray(test).ravel()),
                    source=source,
                    allow_unassigned=bool(allow_unassigned),
                    parent_context_fingerprint=self.base_fingerprint,
                )
                for index, (train, test) in enumerate(iterator)
            )
        except ResamplingContractError:
            raise
        except Exception as exc:
            _raise_contract(
                "supplied_splitter_failed",
                f"Supplied splitter failed with {type(exc).__name__}: {exc}",
                splitter_type=f"{type(splitter).__module__}.{type(splitter).__qualname__}",
                exception_type=type(exc).__name__,
            )
        if not assignments:
            _raise_contract(
                "empty_supplied_splitter",
                "Supplied splitter produced no assignments.",
                scope=scope,
            )
        return self.with_supplied_splits(assignments)


def ensure_fit_resampling_context(
    context: Optional[FitResamplingContext],
    *,
    n_rows: int,
) -> FitResamplingContext:
    if context is None:
        return FitResamplingContext.iid(int(n_rows))
    if not isinstance(context, FitResamplingContext):
        _raise_contract(
            "invalid_resampling_context",
            "resampling_context must be a FitResamplingContext or None.",
            context_type=f"{type(context).__module__}.{type(context).__qualname__}",
        )
    if context.n_rows != int(n_rows):
        _raise_contract(
            "context_row_mismatch",
            f"Resampling context has {context.n_rows} rows but the data has {n_rows} rows.",
            context_rows=context.n_rows,
            data_rows=int(n_rows),
        )
    return context


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _component_labels(context: FitResamplingContext) -> np.ndarray:
    boundaries = context.policy.enforced_boundaries
    if not boundaries:
        return np.arange(context.n_rows, dtype=int)
    union_find = _UnionFind(context.n_rows)
    for field_name in boundaries:
        first_by_key: dict[tuple[str, str, str], int] = {}
        values = getattr(context, field_name)
        for index, value in enumerate(values):
            if value is None:
                _raise_contract(
                    "missing_identity_value",
                    f"Enforced identity field {field_name!r} contains a missing value.",
                    field=field_name,
                    row_index=index,
                )
            key = typed_scalar_key(value)
            previous = first_by_key.setdefault(key, index)
            union_find.union(previous, index)
    roots = [union_find.find(index) for index in range(context.n_rows)]
    root_order: dict[int, int] = {}
    labels: list[int] = []
    for root in roots:
        labels.append(root_order.setdefault(root, len(root_order)))
    return np.asarray(labels, dtype=int)


def _time_sort_key(value: Any) -> tuple[int, Any]:
    value = _python_scalar(value, field_name="timestamps")
    if value is None:
        _raise_contract(
            "missing_timestamps",
            "Temporal splitting does not permit missing timestamps.",
        )
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return 0, value.astimezone(timezone.utc).timestamp()
    if isinstance(value, date):
        return 0, datetime.combine(value, time.min, tzinfo=timezone.utc).timestamp()
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            _raise_contract(
                "nonfinite_timestamp",
                "Temporal splitting requires finite numeric timestamps.",
            )
        return 0, numeric
    if isinstance(value, str):
        text = value.strip()
        if not text:
            _raise_contract(
                "missing_timestamps",
                "Temporal splitting does not permit empty timestamp strings.",
            )
        normalised = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalised)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return 0, parsed.astimezone(timezone.utc).timestamp()
        except ValueError:
            return 1, text
    _raise_contract(
        "unsupported_timestamp",
        f"Unsupported timestamp type {type(value).__module__}.{type(value).__qualname__}.",
        value_type=f"{type(value).__module__}.{type(value).__qualname__}",
    )


def _time_blocks(context: FitResamplingContext) -> tuple[tuple[int, ...], ...]:
    indexed = sorted(
        range(context.n_rows),
        key=lambda index: (_time_sort_key(context.timestamps[index]), index),
    )
    blocks: list[list[int]] = []
    previous_key: Optional[tuple[int, Any]] = None
    for index in indexed:
        key = _time_sort_key(context.timestamps[index])
        if previous_key is None or key != previous_key:
            blocks.append([])
            previous_key = key
        blocks[-1].append(index)
    return tuple(tuple(block) for block in blocks)


def _validate_assignment(
    context: FitResamplingContext,
    assignment: SplitAssignment,
    *,
    y: Optional[Sequence[Any]],
) -> LeakageAudit:
    train = assignment.train_indices
    test = assignment.test_indices
    n_rows = context.n_rows
    if not train or not test:
        _raise_contract(
            "empty_split_partition",
            f"Split {assignment.split_id!r} must have non-empty train and test partitions.",
            split_id=assignment.split_id,
            n_train=len(train),
            n_test=len(test),
        )
    if len(set(train)) != len(train):
        _raise_contract(
            "duplicate_train_index",
            f"Split {assignment.split_id!r} contains duplicate training indices.",
            split_id=assignment.split_id,
            n_train=len(train),
            n_unique_train=len(set(train)),
        )
    if len(set(test)) != len(test):
        _raise_contract(
            "duplicate_test_index",
            f"Split {assignment.split_id!r} contains duplicate test indices.",
            split_id=assignment.split_id,
            n_test=len(test),
            n_unique_test=len(set(test)),
        )
    invalid = [index for index in (*train, *test) if index < 0 or index >= n_rows]
    if invalid:
        _raise_contract(
            "split_index_out_of_bounds",
            f"Split {assignment.split_id!r} contains out-of-bounds indices.",
            split_id=assignment.split_id,
            n_rows=n_rows,
            invalid_count=len(invalid),
            invalid_min=min(invalid),
            invalid_max=max(invalid),
        )
    overlap = set(train).intersection(test)
    if overlap:
        _raise_contract(
            "train_test_overlap",
            f"Split {assignment.split_id!r} has rows in both train and test partitions.",
            split_id=assignment.split_id,
            overlap_count=len(overlap),
        )
    assigned = set(train).union(test)
    n_unassigned = n_rows - len(assigned)
    coverage_required = bool(
        context.policy.require_full_coverage and not assignment.allow_unassigned
    )
    if coverage_required and n_unassigned:
        _raise_contract(
            "incomplete_split_coverage",
            f"Split {assignment.split_id!r} leaves {n_unassigned} rows unassigned.",
            split_id=assignment.split_id,
            n_rows=n_rows,
            n_unassigned=n_unassigned,
        )
    if (
        assignment.parent_context_fingerprint is not None
        and assignment.parent_context_fingerprint != context.base_fingerprint
    ):
        _raise_contract(
            "split_parent_scope_mismatch",
            f"Split {assignment.split_id!r} was materialized for a different row context.",
            split_id=assignment.split_id,
            expected_parent=context.base_fingerprint,
            observed_parent=assignment.parent_context_fingerprint,
        )

    identity_overlaps: list[tuple[str, int]] = []
    for field_name in context.policy.enforced_boundaries:
        values = getattr(context, field_name)
        train_values = {typed_scalar_key(values[index]) for index in train}
        test_values = {typed_scalar_key(values[index]) for index in test}
        count = len(train_values.intersection(test_values))
        identity_overlaps.append((field_name, count))
        if count:
            _raise_contract(
                "identity_boundary_overlap",
                f"Split {assignment.split_id!r} crosses enforced identity field {field_name!r}.",
                split_id=assignment.split_id,
                field=field_name,
                overlap_count=count,
            )

    temporal_order_ok: Optional[bool] = None
    if context.policy.temporal_order_required:
        train_times = [_time_sort_key(context.timestamps[index]) for index in train]
        test_times = [_time_sort_key(context.timestamps[index]) for index in test]
        temporal_order_ok = max(train_times) < min(test_times)
        if not temporal_order_ok:
            _raise_contract(
                "temporal_order_violation",
                f"Split {assignment.split_id!r} is not a strict forward-time split.",
                split_id=assignment.split_id,
                n_train=len(train),
                n_test=len(test),
            )

    train_class_count = 0
    test_class_count = 0
    full_class_count = 0
    class_support_ok: Optional[bool] = None
    if y is not None:
        if len(y) != n_rows:
            _raise_contract(
                "label_length_mismatch",
                f"y has {len(y)} values but the context has {n_rows} rows.",
                n_rows=n_rows,
                y_rows=len(y),
            )
        frozen_y = tuple(_python_scalar(value, field_name="y") for value in y)
        full_keys = set(_typed_counts(frozen_y))
        train_keys = set(_typed_counts(tuple(frozen_y[index] for index in train)))
        test_keys = set(_typed_counts(tuple(frozen_y[index] for index in test)))
        full_class_count = len(full_keys)
        train_class_count = len(train_keys)
        test_class_count = len(test_keys)
        class_support_ok = train_keys == full_keys and test_keys == full_keys
        if context.policy.class_coverage_required and not class_support_ok:
            _raise_contract(
                "split_class_support_missing",
                f"Split {assignment.split_id!r} does not preserve every class in train and test.",
                split_id=assignment.split_id,
                full_class_count=full_class_count,
                train_class_count=train_class_count,
                test_class_count=test_class_count,
                train_missing_class_count=len(full_keys - train_keys),
                test_missing_class_count=len(full_keys - test_keys),
            )

    return LeakageAudit(
        ok=True,
        reason="ok",
        n_rows=n_rows,
        n_train=len(train),
        n_test=len(test),
        n_unassigned=n_unassigned,
        row_overlap_count=0,
        identity_overlap_counts=tuple(identity_overlaps),
        temporal_order_ok=temporal_order_ok,
        class_support_ok=class_support_ok,
        train_class_count=train_class_count,
        test_class_count=test_class_count,
        full_class_count=full_class_count,
    )


def _resolve_assignments(
    context: FitResamplingContext,
    assignments: Sequence[SplitAssignment],
    *,
    y: Optional[Sequence[Any]],
    purpose: str,
) -> ResolvedSplitPlan:
    resolved: list[ResolvedSplit] = []
    for assignment in assignments:
        audit = _validate_assignment(context, assignment, y=y)
        fingerprint = _sha256(
            {
                "schema_version": RESAMPLING_SCHEMA_VERSION,
                "base_context_fingerprint": context.base_fingerprint,
                "policy": context.policy.to_record(),
                "assignment": assignment.to_record(),
                "train_row_ids": [
                    _typed_record(context.row_ids[index])
                    for index in assignment.train_indices
                ],
                "test_row_ids": [
                    _typed_record(context.row_ids[index])
                    for index in assignment.test_indices
                ],
            }
        )
        resolved.append(
            ResolvedSplit(
                assignment=assignment,
                fingerprint=fingerprint,
                audit=audit,
            )
        )
    plan_record = {
        "schema_version": RESAMPLING_SCHEMA_VERSION,
        "purpose": str(purpose),
        "context_fingerprint": context.fingerprint,
        "policy": context.policy.to_record(),
        "splits": [split.fingerprint for split in resolved],
    }
    return ResolvedSplitPlan(
        purpose=str(purpose),
        policy=context.policy,
        context_fingerprint=context.fingerprint,
        splits=tuple(resolved),
        fingerprint=_sha256(plan_record),
    )


def validate_supplied_assignments(
    context: FitResamplingContext,
    *,
    y: Optional[Sequence[Any]] = None,
    scope: Optional[str] = None,
) -> ResolvedSplitPlan:
    assignments = tuple(
        assignment
        for assignment in context.supplied_splits
        if scope is None or assignment.scope == str(scope)
    )
    if not assignments:
        _raise_contract(
            "supplied_split_not_found",
            "No supplied split assignments match the requested scope.",
            scope=scope,
            n_supplied_splits=len(context.supplied_splits),
        )
    return _resolve_assignments(
        context,
        assignments,
        y=y,
        purpose=str(scope or "supplied"),
    )


def resolve_assignment(
    context: FitResamplingContext,
    assignment: SplitAssignment,
    *,
    y: Optional[Sequence[Any]] = None,
    purpose: Optional[str] = None,
) -> ResolvedSplitPlan:
    """Validate one caller-materialized assignment without changing its policy."""

    context = ensure_fit_resampling_context(context, n_rows=context.n_rows)
    return _resolve_assignments(
        context,
        (assignment,),
        y=y,
        purpose=str(purpose or assignment.scope),
    )


def resolve_leave_one_source_out(
    context: FitResamplingContext,
    y: Sequence[Any],
    *,
    source_ids: Sequence[Any],
    axis: str,
    purpose: str = "outer_external",
) -> ResolvedSplitPlan:
    """Resolve exhaustive source-held-out assignments without IID fallback."""

    context = ensure_fit_resampling_context(context, n_rows=context.n_rows)
    if (
        context.policy.is_structured
        or context.groups
        or context.patient_ids
        or context.site_ids
        or context.batch_ids
        or context.timestamps
        or context.supplied_splits
    ):
        _raise_contract(
            "external_source_context_composition_unsupported",
            "Leave-one-source-out cannot compose a source boundary with existing structured resampling constraints.",
            policy=context.policy.kind,
        )
    sources = _freeze_vector(source_ids, field_name="source_ids", n_rows=context.n_rows, allow_empty=False)
    if any(value is None for value in sources):
        _raise_contract("missing_source_id", "Leave-one-source-out splitting does not permit missing source ids.")
    source_keys = tuple(typed_scalar_key(value) for value in sources)
    unique_sources = sorted(set(source_keys))
    if len(unique_sources) < 2:
        _raise_contract("insufficient_sources", "Leave-one-source-out splitting requires at least two sources.", n_sources=len(unique_sources))
    axis_text = str(axis or "").strip()
    if axis_text not in {"study_id", "cohort_id", "site_id"}:
        _raise_contract("invalid_external_holdout_axis", "External holdout axis must be study_id, cohort_id, or site_id.", axis=axis_text)
    frozen_y = tuple(_python_scalar(value, field_name="y") for value in y)
    if len(frozen_y) != context.n_rows:
        _raise_contract("label_length_mismatch", f"y has {len(frozen_y)} values but the context has {context.n_rows} rows.", n_rows=context.n_rows, y_rows=len(frozen_y))
    source_context = FitResamplingContext(
        n_rows=context.n_rows, row_ids=context.row_ids, groups=sources,
        sample_weights=context.sample_weights,
        policy=ResamplingPolicy(kind="group", enforced_boundaries=("groups",)),
        parent_split_fingerprint=context.fingerprint,
    )
    source_vector_sha256 = _sha256({"axis": axis_text, "source_ids": [_typed_record(value) for value in sources]})
    assignments: list[SplitAssignment] = []
    for ordinal, source_key in enumerate(unique_sources):
        test = tuple(index for index, key in enumerate(source_keys) if key == source_key)
        train = tuple(index for index, key in enumerate(source_keys) if key != source_key)
        train_labels = {typed_scalar_key(frozen_y[index]) for index in train}
        test_labels = {typed_scalar_key(frozen_y[index]) for index in test}
        if not test_labels.issubset(train_labels):
            _raise_contract("external_unseen_test_label", "A held-out source contains a label absent from all training sources.", axis=axis_text, held_out_source_ordinal=ordinal, train_class_count=len(train_labels), test_class_count=len(test_labels))
        assignments.append(SplitAssignment(
            scope="outer_external", split_id=f"{purpose}-{axis_text}-{ordinal}",
            train_indices=train, test_indices=test, source="resolver_leave_one_source_out",
            parent_context_fingerprint=source_context.base_fingerprint,
            metadata={"external_axis": axis_text, "held_out_source_ordinal": ordinal,
                      "source_membership_sha256": source_vector_sha256,
                      "parent_context_fingerprint": context.fingerprint},
        ))
    return _resolve_assignments(source_context, assignments, y=frozen_y, purpose=str(purpose))


def _target_partition_sizes(
    n_rows: int,
    *,
    test_size: float,
    train_size: Optional[int],
) -> tuple[int, int]:
    if train_size is not None:
        target_train = int(train_size)
        if target_train <= 0 or target_train >= n_rows:
            _raise_contract(
                "invalid_train_size",
                f"train_size={target_train} is invalid for {n_rows} rows.",
                n_rows=n_rows,
                train_size=target_train,
            )
        return target_train, n_rows - target_train
    fraction = float(test_size)
    if not 0.0 < fraction < 1.0:
        _raise_contract(
            "invalid_test_size",
            "test_size must be strictly between zero and one.",
            test_size=fraction,
        )
    target_test = int(math.ceil(fraction * n_rows))
    target_test = min(max(1, target_test), n_rows - 1)
    return n_rows - target_test, target_test


def _legacy_iid_holdout(
    context: FitResamplingContext,
    y: Sequence[Any],
    *,
    seed: int,
    test_size: float,
    train_size: Optional[int],
) -> SplitAssignment:
    counts = _typed_counts(tuple(y))
    if not counts:
        _raise_contract(
            "empty_labels",
            "Cannot split an empty label vector.",
        )
    declared_stratified = context.policy.kind == "stratified"
    auto_stratified = context.policy.kind == "iid" and len(counts) >= 2 and min(counts.values()) >= 2
    use_stratified = declared_stratified or auto_stratified
    if declared_stratified and (len(counts) < 2 or min(counts.values()) < 2):
        _raise_contract(
            "stratification_impossible",
            "Declared stratification requires at least two classes and two rows per class.",
            n_classes=len(counts),
            min_class_count=min(counts.values()),
            n_rows=context.n_rows,
        )
    kwargs: dict[str, Any] = {"random_state": int(seed)}
    if train_size is None:
        kwargs["test_size"] = float(test_size)
    else:
        kwargs["train_size"] = int(train_size)
    indices = np.arange(context.n_rows, dtype=int)
    try:
        train, test = train_test_split(
            indices,
            stratify=(_stratification_labels(y) if use_stratified else None),
            **kwargs,
        )
    except ValueError as exc:
        _raise_contract(
            "stratification_impossible" if use_stratified else "iid_split_impossible",
            f"Unable to materialize the declared holdout: {exc}",
            n_rows=context.n_rows,
            n_classes=len(counts),
            min_class_count=min(counts.values()),
            test_size=float(test_size),
            train_size=train_size,
            exception_type=type(exc).__name__,
        )
    return SplitAssignment(
        scope="outer",
        split_id=f"outer-seed-{int(seed)}",
        train_indices=tuple(int(v) for v in np.asarray(train).ravel()),
        test_indices=tuple(int(v) for v in np.asarray(test).ravel()),
        source="resolver_iid_stratified" if use_stratified else "resolver_iid",
        parent_context_fingerprint=context.base_fingerprint,
    )


def _group_holdout(
    context: FitResamplingContext,
    y: Sequence[Any],
    *,
    seed: int,
    test_size: float,
    train_size: Optional[int],
) -> SplitAssignment:
    components = _component_labels(context)
    n_components = len(set(int(v) for v in components.tolist()))
    if n_components < 2:
        _raise_contract(
            "group_split_impossible",
            "Grouped holdout requires at least two disconnected identity components.",
            n_components=n_components,
            n_rows=context.n_rows,
            enforced_boundaries=context.policy.enforced_boundaries,
        )
    target_train, target_test = _target_partition_sizes(
        context.n_rows,
        test_size=test_size,
        train_size=train_size,
    )
    x_dummy = np.zeros((context.n_rows, 1), dtype=np.uint8)
    candidates: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
    if context.policy.kind == "stratified_group":
        approximate_folds = max(2, int(round(context.n_rows / max(1, target_test))))
        n_splits = min(n_components, approximate_folds)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(seed),
        )
        try:
            iterator = splitter.split(
                x_dummy,
                _stratification_labels(y),
                components,
            )
            for train, test in iterator:
                train_tuple = tuple(int(v) for v in np.asarray(train).ravel())
                test_tuple = tuple(int(v) for v in np.asarray(test).ravel())
                size_cost = abs(len(test_tuple) - target_test) / max(1, context.n_rows)
                candidates.append((size_cost, train_tuple, test_tuple))
        except ValueError as exc:
            _raise_contract(
                "group_stratification_impossible",
                f"Unable to materialize a stratified-group holdout: {exc}",
                n_rows=context.n_rows,
                n_components=n_components,
                n_classes=len(_typed_counts(tuple(y))),
                exception_type=type(exc).__name__,
            )
    else:
        splitter = GroupShuffleSplit(
            n_splits=max(16, min(128, n_components * 4)),
            test_size=float(target_test / context.n_rows),
            random_state=int(seed),
        )
        try:
            for train, test in splitter.split(x_dummy, groups=components):
                train_tuple = tuple(int(v) for v in np.asarray(train).ravel())
                test_tuple = tuple(int(v) for v in np.asarray(test).ravel())
                size_cost = abs(len(test_tuple) - target_test) / max(1, context.n_rows)
                candidates.append((size_cost, train_tuple, test_tuple))
        except ValueError as exc:
            _raise_contract(
                "group_split_impossible",
                f"Unable to materialize a grouped holdout: {exc}",
                n_rows=context.n_rows,
                n_components=n_components,
                exception_type=type(exc).__name__,
            )
    if not candidates:
        _raise_contract(
            "group_split_impossible",
            "Grouped holdout resolver produced no assignments.",
            n_components=n_components,
            target_test=target_test,
        )
    if context.policy.class_coverage_required:
        full_classes = set(_typed_counts(tuple(y)))
        feasible: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for candidate in candidates:
            _, train_candidate, test_candidate = candidate
            train_classes = {
                typed_scalar_key(y[index]) for index in train_candidate
            }
            test_classes = {
                typed_scalar_key(y[index]) for index in test_candidate
            }
            if train_classes == full_classes and test_classes == full_classes:
                feasible.append(candidate)
        if not feasible:
            _raise_contract(
                "group_stratification_impossible",
                "No grouped holdout preserves every class in train and test.",
                n_components=n_components,
                n_classes=len(full_classes),
                target_test=target_test,
            )
        candidates = feasible
    candidates.sort(key=lambda item: (item[0], item[2], item[1]))
    _, train, test = candidates[0]
    return SplitAssignment(
        scope="outer",
        split_id=f"outer-group-seed-{int(seed)}",
        train_indices=train,
        test_indices=test,
        source=(
            "resolver_stratified_group"
            if context.policy.kind == "stratified_group"
            else "resolver_group"
        ),
        parent_context_fingerprint=context.base_fingerprint,
    )


def _temporal_holdout(
    context: FitResamplingContext,
    *,
    seed: int,
    test_size: float,
    train_size: Optional[int],
) -> SplitAssignment:
    del seed  # Forward blocked splitting is deterministic for a fixed row contract.
    blocks = _time_blocks(context)
    if len(blocks) < 2 + context.policy.temporal_gap:
        _raise_contract(
            "temporal_split_impossible",
            "Not enough distinct time blocks for train, gap, and test partitions.",
            n_time_blocks=len(blocks),
            temporal_gap=context.policy.temporal_gap,
        )
    _, target_test = _target_partition_sizes(
        context.n_rows,
        test_size=test_size,
        train_size=train_size,
    )
    candidates: list[tuple[int, int]] = []
    for boundary in range(1, len(blocks)):
        gap_start = boundary
        train_end = gap_start - context.policy.temporal_gap
        if train_end <= 0:
            continue
        test_count = sum(len(block) for block in blocks[boundary:])
        if test_count <= 0:
            continue
        candidates.append((abs(test_count - target_test), boundary))
    if not candidates:
        _raise_contract(
            "temporal_split_impossible",
            "No valid forward-time boundary satisfies the declared gap.",
            n_time_blocks=len(blocks),
            temporal_gap=context.policy.temporal_gap,
        )
    _, boundary = min(candidates, key=lambda item: (item[0], item[1]))
    train_end = boundary - context.policy.temporal_gap
    train = tuple(index for block in blocks[:train_end] for index in block)
    test = tuple(index for block in blocks[boundary:] for index in block)
    return SplitAssignment(
        scope="outer",
        split_id="outer-temporal",
        train_indices=train,
        test_indices=test,
        source="resolver_blocked_temporal",
        allow_unassigned=bool(context.policy.temporal_gap),
        parent_context_fingerprint=context.base_fingerprint,
        metadata=(("temporal_boundary_block", boundary),),
    )


def resolve_holdout(
    context: FitResamplingContext,
    y: Sequence[Any],
    *,
    seed: int,
    test_size: float = 0.2,
    train_size: Optional[int] = None,
    purpose: str = "outer",
    supplied_split_id: Optional[str] = None,
) -> ResolvedSplitPlan:
    """Resolve and validate one outer/holdout assignment."""

    context = ensure_fit_resampling_context(context, n_rows=len(y))
    y_values = tuple(_python_scalar(value, field_name="y") for value in y)
    if supplied_split_id is not None:
        candidates = tuple(
            split
            for split in context.supplied_splits
            if split.scope == str(purpose)
            and split.split_id == str(supplied_split_id)
        )
        if len(candidates) != 1:
            _raise_contract(
                "supplied_split_not_found",
                "Exactly one supplied holdout must match the requested scope and id.",
                purpose=purpose,
                supplied_split_id=supplied_split_id,
                n_matches=len(candidates),
            )
        assignments = candidates
    elif context.policy.kind == "supplied":
        candidates = tuple(
            split
            for split in context.supplied_splits
            if split.scope == str(purpose)
        )
        if len(candidates) > 1:
            _raise_contract(
                "ambiguous_supplied_holdout",
                "Multiple supplied holdouts match; select one by supplied_split_id.",
                purpose=purpose,
                matching_split_ids=tuple(split.split_id for split in candidates),
            )
        if len(candidates) != 1:
            _raise_contract(
                "supplied_split_not_found",
                "Exactly one supplied holdout must match the requested scope and id.",
                purpose=purpose,
                supplied_split_id=supplied_split_id,
                n_matches=len(candidates),
            )
        assignments = candidates
    elif context.policy.kind in {"iid", "stratified"}:
        assignments = (
            replace(
                _legacy_iid_holdout(
                    context,
                    y_values,
                    seed=int(seed),
                    test_size=float(test_size),
                    train_size=train_size,
                ),
                scope=str(purpose),
            ),
        )
    elif context.policy.kind in {"group", "stratified_group"}:
        assignments = (
            replace(
                _group_holdout(
                    context,
                    y_values,
                    seed=int(seed),
                    test_size=float(test_size),
                    train_size=train_size,
                ),
                scope=str(purpose),
            ),
        )
    elif context.policy.kind == "blocked_temporal":
        assignments = (
            replace(
                _temporal_holdout(
                    context,
                    seed=int(seed),
                    test_size=float(test_size),
                    train_size=train_size,
                ),
                scope=str(purpose),
            ),
        )
    else:
        _raise_contract(
            "unsupported_resampling_policy",
            f"Holdout resolution does not support policy {context.policy.kind!r}.",
            policy=context.policy.kind,
            purpose=purpose,
        )
    return _resolve_assignments(
        context,
        assignments,
        y=y_values,
        purpose=str(purpose),
    )


def _iid_cv_assignments(
    context: FitResamplingContext,
    y: Sequence[Any],
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
    stratified: bool,
    shuffle: bool,
    purpose: str,
) -> tuple[SplitAssignment, ...]:
    counts = _typed_counts(tuple(y))
    if n_splits < 2:
        _raise_contract(
            "invalid_cv_splits",
            "n_splits must be at least two.",
            n_splits=n_splits,
        )
    if n_splits > context.n_rows:
        _raise_contract(
            "cv_split_impossible",
            "n_splits cannot exceed the number of rows.",
            n_splits=n_splits,
            n_rows=context.n_rows,
        )
    if stratified and (
        len(counts) < 2 or min(counts.values()) < int(n_splits)
    ):
        _raise_contract(
            "stratification_impossible",
            "Stratified CV requires at least n_splits rows in every class.",
            n_splits=n_splits,
            n_classes=len(counts),
            min_class_count=min(counts.values()),
        )
    x_dummy = np.zeros((context.n_rows, 1), dtype=np.uint8)
    try:
        if stratified and n_repeats > 1:
            splitter: Any = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=int(seed),
            )
            iterator = splitter.split(x_dummy, _stratification_labels(y))
        elif stratified:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=bool(shuffle),
                random_state=(int(seed) if shuffle else None),
            )
            iterator = splitter.split(x_dummy, _stratification_labels(y))
        elif n_repeats > 1:
            splitter = RepeatedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=int(seed),
            )
            iterator = splitter.split(x_dummy)
        else:
            splitter = KFold(
                n_splits=n_splits,
                shuffle=bool(shuffle),
                random_state=(int(seed) if shuffle else None),
            )
            iterator = splitter.split(x_dummy)
        return tuple(
            SplitAssignment(
                scope=purpose,
                split_id=f"{purpose}-{index}",
                train_indices=tuple(int(v) for v in np.asarray(train).ravel()),
                test_indices=tuple(int(v) for v in np.asarray(test).ravel()),
                source="resolver_iid_stratified_cv" if stratified else "resolver_iid_cv",
                allow_unassigned=True,
                parent_context_fingerprint=context.base_fingerprint,
            )
            for index, (train, test) in enumerate(iterator)
        )
    except ValueError as exc:
        _raise_contract(
            "stratification_impossible" if stratified else "cv_split_impossible",
            f"Unable to materialize CV plan: {exc}",
            n_rows=context.n_rows,
            n_splits=n_splits,
            n_repeats=n_repeats,
            stratified=bool(stratified),
            exception_type=type(exc).__name__,
        )


def _group_cv_assignments(
    context: FitResamplingContext,
    y: Sequence[Any],
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
    stratified: bool,
    purpose: str,
) -> tuple[SplitAssignment, ...]:
    components = _component_labels(context)
    unique_components = tuple(sorted(set(int(v) for v in components.tolist())))
    if len(unique_components) < n_splits:
        _raise_contract(
            "group_cv_impossible",
            "Grouped CV requires at least n_splits disconnected identity components.",
            n_splits=n_splits,
            n_components=len(unique_components),
            enforced_boundaries=context.policy.enforced_boundaries,
        )
    x_dummy = np.zeros((context.n_rows, 1), dtype=np.uint8)
    assignments: list[SplitAssignment] = []
    for repeat_index in range(n_repeats):
        repeat_seed = int(seed) + repeat_index
        if stratified:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=repeat_seed,
            )
            try:
                iterator = splitter.split(
                    x_dummy,
                    _stratification_labels(y),
                    components,
                )
                pairs = list(iterator)
            except ValueError as exc:
                _raise_contract(
                    "group_stratification_impossible",
                    f"Unable to materialize stratified-group CV: {exc}",
                    n_splits=n_splits,
                    n_components=len(unique_components),
                    exception_type=type(exc).__name__,
                )
        else:
            rng = np.random.default_rng(repeat_seed)
            shuffled = list(unique_components)
            rng.shuffle(shuffled)
            folds: list[list[int]] = [[] for _ in range(n_splits)]
            fold_sizes = [0] * n_splits
            component_rows = {
                component: tuple(
                    index
                    for index, observed in enumerate(components.tolist())
                    if int(observed) == component
                )
                for component in shuffled
            }
            for component in sorted(
                shuffled,
                key=lambda value: (-len(component_rows[value]), shuffled.index(value)),
            ):
                fold_index = min(
                    range(n_splits),
                    key=lambda index: (fold_sizes[index], index),
                )
                folds[fold_index].append(component)
                fold_sizes[fold_index] += len(component_rows[component])
            pairs = []
            all_indices = set(range(context.n_rows))
            for fold in folds:
                test_set = {
                    index for component in fold for index in component_rows[component]
                }
                train = np.asarray(sorted(all_indices - test_set), dtype=int)
                test = np.asarray(sorted(test_set), dtype=int)
                pairs.append((train, test))
        for fold_index, (train, test) in enumerate(pairs):
            assignments.append(
                SplitAssignment(
                    scope=purpose,
                    split_id=f"{purpose}-repeat-{repeat_index}-fold-{fold_index}",
                    train_indices=tuple(int(v) for v in np.asarray(train).ravel()),
                    test_indices=tuple(int(v) for v in np.asarray(test).ravel()),
                    source=(
                        "resolver_stratified_group_cv"
                        if stratified
                        else "resolver_group_cv"
                    ),
                    allow_unassigned=True,
                    parent_context_fingerprint=context.base_fingerprint,
                )
            )
    return tuple(assignments)


def _temporal_cv_assignments(
    context: FitResamplingContext,
    *,
    n_splits: int,
    purpose: str,
) -> tuple[SplitAssignment, ...]:
    blocks = _time_blocks(context)
    required_blocks = n_splits + 1 + context.policy.temporal_gap
    if len(blocks) < required_blocks:
        _raise_contract(
            "temporal_cv_impossible",
            "Forward CV requires at least n_splits + 1 + gap distinct time blocks.",
            n_splits=n_splits,
            n_time_blocks=len(blocks),
            temporal_gap=context.policy.temporal_gap,
            required_time_blocks=required_blocks,
        )
    chunks = tuple(
        tuple(int(v) for v in chunk.tolist())
        for chunk in np.array_split(np.arange(len(blocks), dtype=int), n_splits + 1)
    )
    assignments: list[SplitAssignment] = []
    for fold_index in range(1, len(chunks)):
        test_block_ids = chunks[fold_index]
        if not test_block_ids:
            continue
        test_start = min(test_block_ids)
        train_end = test_start - context.policy.temporal_gap
        if train_end <= 0:
            continue
        train = tuple(index for block in blocks[:train_end] for index in block)
        test = tuple(index for block_id in test_block_ids for index in blocks[block_id])
        assignments.append(
            SplitAssignment(
                scope=purpose,
                split_id=f"{purpose}-fold-{fold_index - 1}",
                train_indices=train,
                test_indices=test,
                source="resolver_expanding_temporal_cv",
                allow_unassigned=True,
                parent_context_fingerprint=context.base_fingerprint,
                metadata=(("temporal_test_start_block", test_start),),
            )
        )
    if len(assignments) != n_splits:
        _raise_contract(
            "temporal_cv_impossible",
            "Unable to materialize the requested number of forward CV folds.",
            requested_splits=n_splits,
            resolved_splits=len(assignments),
            n_time_blocks=len(blocks),
        )
    return tuple(assignments)


def resolve_cv(
    context: FitResamplingContext,
    y: Sequence[Any],
    *,
    n_splits: int,
    seed: int,
    purpose: str,
    n_repeats: int = 1,
    stratified: Optional[bool] = None,
    shuffle: bool = True,
) -> ResolvedSplitPlan:
    """Resolve every fold for one score-bearing inner-CV purpose."""

    context = ensure_fit_resampling_context(context, n_rows=len(y))
    y_values = tuple(_python_scalar(value, field_name="y") for value in y)
    n_splits = int(n_splits)
    n_repeats = int(n_repeats)
    if n_repeats < 1:
        _raise_contract(
            "invalid_cv_repeats",
            "n_repeats must be at least one.",
            n_repeats=n_repeats,
        )
    supplied_assignments = tuple(
        split for split in context.supplied_splits if split.scope == str(purpose)
    )
    if supplied_assignments:
        assignments = supplied_assignments
    elif context.policy.kind == "supplied":
        if not supplied_assignments:
            _raise_contract(
                "supplied_split_not_found",
                "No supplied inner splits match the requested purpose.",
                purpose=purpose,
                available_scopes=tuple(sorted({s.scope for s in context.supplied_splits})),
            )
    elif context.policy.kind == "blocked_temporal":
        if n_repeats != 1:
            _raise_contract(
                "temporal_repeats_unsupported",
                "Repeated random CV is undefined for a forward temporal policy.",
                n_repeats=n_repeats,
                purpose=purpose,
            )
        assignments = _temporal_cv_assignments(
            context,
            n_splits=n_splits,
            purpose=str(purpose),
        )
    elif context.policy.kind in {"group", "stratified_group"}:
        use_stratified = (
            context.policy.kind == "stratified_group"
            if stratified is None
            else bool(stratified)
        )
        assignments = _group_cv_assignments(
            context,
            y_values,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=int(seed),
            stratified=use_stratified,
            purpose=str(purpose),
        )
    else:
        use_stratified = (
            context.policy.kind == "stratified"
            if stratified is None
            else bool(stratified)
        )
        assignments = _iid_cv_assignments(
            context,
            y_values,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=int(seed),
            stratified=use_stratified,
            shuffle=bool(shuffle),
            purpose=str(purpose),
        )
    return _resolve_assignments(
        context,
        assignments,
        y=y_values,
        purpose=str(purpose),
    )


def resolve_fit_subsample(
    context: FitResamplingContext,
    y: Sequence[Any],
    *,
    fraction: float,
    seed: int,
    balanced: bool = True,
    min_per_class: int = 2,
) -> tuple[int, ...]:
    """Resolve a fit-only row subset while preserving declared atomic boundaries."""

    context = ensure_fit_resampling_context(context, n_rows=len(y))
    fraction = float(max(0.05, min(1.0, fraction)))
    if fraction >= 0.999:
        return tuple(range(context.n_rows))
    target = int(max(2, round(fraction * context.n_rows)))
    if context.policy.kind == "blocked_temporal":
        blocks = _time_blocks(context)
        selected: list[int] = []
        for block in blocks:
            selected.extend(block)
            if len(selected) >= target:
                break
        return tuple(selected)
    if context.policy.enforced_boundaries:
        components = _component_labels(context)
        component_rows: dict[int, tuple[int, ...]] = {}
        for component in sorted(set(int(v) for v in components.tolist())):
            component_rows[component] = tuple(
                index
                for index, observed in enumerate(components.tolist())
                if int(observed) == component
            )
        rng = np.random.default_rng(int(seed))
        order = list(component_rows)
        rng.shuffle(order)
        selected_components: list[int] = []
        selected_count = 0
        for component in order:
            candidate_count = selected_count + len(component_rows[component])
            if selected_components and abs(selected_count - target) < abs(candidate_count - target):
                continue
            selected_components.append(component)
            selected_count = candidate_count
            if selected_count >= target:
                break
        selected = tuple(
            sorted(
                index
                for component in selected_components
                for index in component_rows[component]
            )
        )
        if not selected:
            _raise_contract(
                "group_subsample_impossible",
                "No identity-atomic fit subset could be selected.",
                target_rows=target,
                n_components=len(component_rows),
            )
        if balanced:
            selected_keys = set(
                typed_scalar_key(y[index]) for index in selected
            )
            full_keys = set(_typed_counts(tuple(y)))
            if selected_keys != full_keys:
                _raise_contract(
                    "group_subsample_class_support_missing",
                    "Identity-atomic fit subset does not contain every class.",
                    target_rows=target,
                    selected_rows=len(selected),
                    missing_class_count=len(full_keys - selected_keys),
                )
        return selected

    # This mirrors the legacy pipeline sampler for feasible IID paths.
    counts = _typed_counts(tuple(y))
    rng = np.random.default_rng(int(seed))
    class_keys = sorted(counts)
    if balanced and len(class_keys) >= 2:
        min_per_class = int(max(1, min_per_class))
        grouped: dict[tuple[str, str, str], list[int]] = {key: [] for key in class_keys}
        for index, value in enumerate(y):
            grouped[typed_scalar_key(value)].append(index)
        for values in grouped.values():
            rng.shuffle(values)
        chosen: list[int] = []
        for key in class_keys:
            if grouped[key]:
                chosen.append(grouped[key].pop(0))
        if min_per_class > 1:
            for key in class_keys:
                current_count = 1
                while current_count < min_per_class and grouped[key] and len(chosen) < target:
                    chosen.append(grouped[key].pop(0))
                    current_count += 1
        pool = [index for key in class_keys for index in grouped[key]]
        remaining = max(0, target - len(chosen))
        if remaining and pool:
            rng.shuffle(pool)
            chosen.extend(pool[:remaining])
        if len(chosen) >= 2:
            return tuple(sorted(set(chosen)))
    if len(counts) >= 2 and min(counts.values()) >= 2:
        try:
            train, _ = train_test_split(
                np.arange(context.n_rows, dtype=int),
                train_size=fraction,
                stratify=_stratification_labels(y),
                random_state=int(seed),
            )
            return tuple(int(v) for v in np.asarray(train).ravel())
        except ValueError:
            pass
    selected = rng.choice(np.arange(context.n_rows), size=target, replace=False)
    return tuple(int(v) for v in np.asarray(selected).ravel())


def require_supported_resampling(
    context: Optional[FitResamplingContext],
    *,
    callsite: str,
    supported_policies: Sequence[str] = ("iid", "stratified"),
) -> None:
    """Fail closed when a private/opaque splitter cannot honor the row policy."""

    if context is None:
        return
    supported = tuple(str(value) for value in supported_policies)
    if context.policy.kind not in supported:
        _raise_contract(
            "non_iid_internal_resampling_unsupported",
            f"{callsite} cannot honor resampling policy {context.policy.kind!r}.",
            callsite=str(callsite),
            policy=context.policy.kind,
            supported_policies=supported,
            deterministic_reason=(
                f"non_iid_internal_resampling_unsupported:{callsite}"
            ),
        )


__all__ = [
    "FitResamplingContext",
    "LeakageAudit",
    "RESAMPLING_SCHEMA_VERSION",
    "ResolvedSplit",
    "ResolvedSplitPlan",
    "ResamplingContractError",
    "ResamplingPolicy",
    "SplitAssignment",
    "coerce_sample_weights",
    "ensure_fit_resampling_context",
    "require_supported_resampling",
    "resolve_assignment",
    "resolve_cv",
    "resolve_fit_subsample",
    "resolve_holdout",
    "resolve_leave_one_source_out",
    "typed_scalar_key",
    "validate_supplied_assignments",
]

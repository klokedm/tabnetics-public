"""Deterministic, fold-local training balance adapters.

Balancing is a fitting concern only.  The helpers in this module consume one
already-transformed training partition and never retain an executable sampler
for inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, NoReturn, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .resampling import FitResamplingContext, coerce_sample_weights, typed_scalar_key


TRAINING_BALANCE_SCHEMA_VERSION = "2.0"
TRAINING_BALANCE_METHODS = (
    "none",
    "smote",
    "propensity_match",
    "random_over",
    "random_under",
)
_METHODS = frozenset(TRAINING_BALANCE_METHODS)
_SUPPORTED_POLICIES = {"iid", "stratified"}


class TrainingBalanceContractError(ValueError):
    """Fail-closed error for an unsupported training-balance composition."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.diagnostics = _json_safe(diagnostics or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": str(self),
            "diagnostics": dict(self.diagnostics),
        }


def _raise(code: str, message: str, **diagnostics: Any) -> NoReturn:
    raise TrainingBalanceContractError(message, code=code, diagnostics=diagnostics)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_safe(item) for item in list(value)]
    return str(value)


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze strict provenance data without changing its meaning."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(_deep_freeze(item) for item in list(value))
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(TRAINING_BALANCE_SCHEMA_VERSION.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    if array.dtype.hasobject:
        digest.update(
            json.dumps(
                [list(typed_scalar_key(item)) for item in array.ravel().tolist()],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
    else:
        digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _label_fingerprint(values: Sequence[Any]) -> str:
    return _sha256_json([list(typed_scalar_key(value)) for value in values])


@dataclass(frozen=True, slots=True)
class TrainingBalanceConfig:
    """Immutable, default-off training balance configuration."""

    method: str = "none"
    smote_sampling_strategy: str = "auto"
    smote_k_neighbors: int = 5
    propensity_n_splits: int = 5
    propensity_probability_clip: float = 1e-6
    propensity_caliper_sd: float = 0.2
    random_state: int | None = None
    schema_version: str = TRAINING_BALANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        method = str(self.method or "none").strip().lower()
        if method not in _METHODS:
            raise ValueError(
                "training_balance.method must be one of: "
                + ", ".join(TRAINING_BALANCE_METHODS)
            )
        object.__setattr__(self, "method", method)
        if str(self.schema_version) != TRAINING_BALANCE_SCHEMA_VERSION:
            raise ValueError(
                f"training_balance.schema_version must be {TRAINING_BALANCE_SCHEMA_VERSION!r}"
            )
        object.__setattr__(self, "schema_version", TRAINING_BALANCE_SCHEMA_VERSION)
        strategy = str(self.smote_sampling_strategy or "auto").strip().lower()
        if strategy != "auto":
            raise ValueError("training_balance.smote_sampling_strategy supports only 'auto' in v1")
        object.__setattr__(self, "smote_sampling_strategy", strategy)
        if isinstance(self.smote_k_neighbors, bool) or int(self.smote_k_neighbors) < 1:
            raise ValueError("training_balance.smote_k_neighbors must be >= 1")
        object.__setattr__(self, "smote_k_neighbors", int(self.smote_k_neighbors))
        if isinstance(self.propensity_n_splits, bool) or int(self.propensity_n_splits) < 2:
            raise ValueError("training_balance.propensity_n_splits must be >= 2")
        object.__setattr__(self, "propensity_n_splits", int(self.propensity_n_splits))
        clip = float(self.propensity_probability_clip)
        if not math.isfinite(clip) or not 0.0 < clip < 0.5:
            raise ValueError(
                "training_balance.propensity_probability_clip must be finite and in (0, 0.5)"
            )
        object.__setattr__(self, "propensity_probability_clip", clip)
        caliper = float(self.propensity_caliper_sd)
        if not math.isfinite(caliper) or caliper <= 0.0:
            raise ValueError("training_balance.propensity_caliper_sd must be finite and > 0")
        object.__setattr__(self, "propensity_caliper_sd", caliper)
        if self.random_state is not None:
            if isinstance(self.random_state, bool):
                raise ValueError("training_balance.random_state must be an integer or None")
            object.__setattr__(self, "random_state", int(self.random_state))

    @property
    def enabled(self) -> bool:
        return self.method != "none"

    def effective_seed(self, pipeline_seed: int) -> int:
        return int(pipeline_seed if self.random_state is None else self.random_state)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingBalanceConfig":
        config = cls(**dict(value))
        if config.to_dict() != _json_safe(dict(value)):
            raise ValueError("training balance config mapping is not canonical")
        return config

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class TrainingBalanceProvenance:
    """Aggregate, non-sensitive provenance for one training-only application."""

    schema_version: str
    method: str
    seed: int
    config: Mapping[str, Any]
    config_fingerprint: str
    input_fingerprint: str
    output_fingerprint: str
    input_lineage_fingerprint: str
    output_lineage_fingerprint: str
    input_rows: int
    output_rows: int
    input_class_counts: tuple[int, ...]
    output_class_counts: tuple[int, ...]
    synthetic_rows: int = 0
    matched_pairs: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    provenance_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _deep_freeze(self.config))
        object.__setattr__(self, "diagnostics", _deep_freeze(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        record = _json_safe(
            {
                "schema_version": self.schema_version,
                "method": self.method,
                "seed": self.seed,
                "config": self.config,
                "config_fingerprint": self.config_fingerprint,
                "input_fingerprint": self.input_fingerprint,
                "output_fingerprint": self.output_fingerprint,
                "input_lineage_fingerprint": self.input_lineage_fingerprint,
                "output_lineage_fingerprint": self.output_lineage_fingerprint,
                "input_rows": self.input_rows,
                "output_rows": self.output_rows,
                "input_class_counts": self.input_class_counts,
                "output_class_counts": self.output_class_counts,
                "synthetic_rows": self.synthetic_rows,
                "matched_pairs": self.matched_pairs,
                "diagnostics": self.diagnostics,
                "provenance_fingerprint": self.provenance_fingerprint,
            }
        )
        if not record.get("provenance_fingerprint"):
            unsigned = dict(record)
            unsigned.pop("provenance_fingerprint", None)
            record["provenance_fingerprint"] = _sha256_json(unsigned)
        return record


@dataclass(frozen=True, slots=True)
class TrainingBalanceResult:
    """Balanced fitting arrays plus aligned original-row context when available."""

    X: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray | None
    context: FitResamplingContext | None
    provenance: TrainingBalanceProvenance


def _coerce_config(value: TrainingBalanceConfig | Mapping[str, Any] | None) -> TrainingBalanceConfig:
    if value is None:
        return TrainingBalanceConfig()
    if isinstance(value, TrainingBalanceConfig):
        return value
    if isinstance(value, Mapping):
        return TrainingBalanceConfig.from_mapping(value)
    raise TypeError("training balance config must be TrainingBalanceConfig, mapping, or None")


def _validate_inputs(
    X: Any,
    y: Sequence[Any],
    *,
    context: FitResamplingContext | None,
    require_continuous: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(X):
        _raise("sparse_input_unsupported", "Training balancing v1 requires a dense matrix.")
    raw = np.asarray(X)
    if raw.ndim != 2:
        _raise("invalid_matrix_shape", "Training balancing requires a two-dimensional matrix.")
    if require_continuous and raw.dtype.kind != "f":
        _raise(
            "continuous_numeric_input_required",
            "Training balancing v1 accepts only continuous floating-point matrices.",
            dtype=str(raw.dtype),
        )
    x_arr = np.asarray(raw, dtype=float) if require_continuous else raw
    if require_continuous and not np.all(np.isfinite(x_arr)):
        _raise("nonfinite_input", "Training balancing requires only finite feature values.")
    y_arr = np.asarray(y).ravel()
    if y_arr.ndim != 1 or int(y_arr.size) != int(x_arr.shape[0]):
        _raise(
            "label_row_mismatch",
            "Training labels must be one-dimensional and row-aligned with X.",
            x_rows=int(x_arr.shape[0]),
            y_rows=int(y_arr.size),
        )
    if context is not None:
        if context.n_rows != int(y_arr.size):
            _raise(
                "context_row_mismatch",
                "Training balance context is not row-aligned with X and y.",
                context_rows=int(context.n_rows),
                data_rows=int(y_arr.size),
            )
        if require_continuous and context.policy.kind not in _SUPPORTED_POLICIES:
            _raise(
                "unsupported_resampling_policy",
                "Training balancing v1 supports only iid and stratified policies.",
                policy=str(context.policy.kind),
            )
    return x_arr, y_arr


def _class_encoding(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    invalid = sum(
        value is None
        or isinstance(value, (float, np.floating))
        and not math.isfinite(float(value))
        for value in y.tolist()
    )
    if invalid:
        _raise(
            "invalid_predictive_labels",
            "Training balancing does not permit missing or non-finite predictive labels.",
            invalid_count=int(invalid),
        )
    try:
        classes, encoded, counts = np.unique(y, return_inverse=True, return_counts=True)
    except TypeError as exc:
        _raise(
            "unsupported_label_types",
            "Training balancing requires mutually comparable predictive class labels.",
            reason=type(exc).__name__,
        )
    if classes.size < 2:
        _raise("insufficient_classes", "Training balancing requires at least two classes.")
    return classes, np.asarray(encoded, dtype=int), np.asarray(counts, dtype=int)


def _make_provenance(
    *,
    config: TrainingBalanceConfig,
    seed: int,
    X_input: np.ndarray,
    y_input: np.ndarray,
    X_output: np.ndarray,
    y_output: np.ndarray,
    context_input: FitResamplingContext | None,
    context_output: FitResamplingContext | None,
    synthetic_rows: int = 0,
    matched_pairs: int = 0,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> TrainingBalanceProvenance:
    _, _, input_counts = _class_encoding(y_input)
    _, _, output_counts = _class_encoding(y_output)
    input_lineage = (
        context_input.row_ids_fingerprint
        if context_input is not None
        else _sha256_json({"positions": list(range(int(y_input.size)))})
    )
    output_lineage = (
        context_output.row_ids_fingerprint
        if context_output is not None
        else _sha256_json(
            {
                "input_lineage": input_lineage,
                "method": config.method,
                "output_rows": int(y_output.size),
                "synthetic_rows": int(synthetic_rows),
            }
        )
    )
    config_record = config.to_dict()
    record = {
        "schema_version": TRAINING_BALANCE_SCHEMA_VERSION,
        "method": config.method,
        "seed": int(seed),
        "config": config_record,
        "config_fingerprint": config.fingerprint,
        "input_fingerprint": _sha256_json(
            {"X": _array_fingerprint(X_input), "y": _label_fingerprint(y_input.tolist())}
        ),
        "output_fingerprint": _sha256_json(
            {"X": _array_fingerprint(X_output), "y": _label_fingerprint(y_output.tolist())}
        ),
        "input_lineage_fingerprint": input_lineage,
        "output_lineage_fingerprint": output_lineage,
        "input_rows": int(y_input.size),
        "output_rows": int(y_output.size),
        "input_class_counts": tuple(int(value) for value in input_counts.tolist()),
        "output_class_counts": tuple(int(value) for value in output_counts.tolist()),
        "synthetic_rows": int(synthetic_rows),
        "matched_pairs": int(matched_pairs),
        "diagnostics": _json_safe(diagnostics or {}),
    }
    return TrainingBalanceProvenance(
        **record,
        provenance_fingerprint=_sha256_json(record),
    )


def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: TrainingBalanceConfig,
    seed: int,
    context: FitResamplingContext | None,
    sample_weight: np.ndarray | None,
) -> TrainingBalanceResult:
    if sample_weight is not None:
        _raise(
            "smote_sample_weight_unsupported",
            "SMOTE plus caller-provided sample weights is unsupported in v1.",
        )
    _, _, counts = _class_encoding(y)
    maximum = int(np.max(counts))
    synthesised_counts = counts[counts < maximum]
    if synthesised_counts.size == 0:
        _raise("smote_balanced_input", "SMOTE requires at least one underrepresented class.")
    minimum_required = int(config.smote_k_neighbors + 1)
    if np.any(synthesised_counts < minimum_required):
        _raise(
            "smote_class_too_small",
            "Every class synthesized by SMOTE must contain at least k_neighbors + 1 rows.",
            minimum_required=minimum_required,
            observed_minimum=int(np.min(synthesised_counts)),
        )
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise TrainingBalanceContractError(
            "SMOTE requires the optional compatibility-bounded dependency; "
            "install it with `pip install 'tabnetics[balancing]'`.",
            code="smote_dependency_unavailable",
            diagnostics={"extra": "balancing"},
        ) from exc
    sampler = SMOTE(
        sampling_strategy=config.smote_sampling_strategy,
        k_neighbors=int(config.smote_k_neighbors),
        random_state=int(seed),
    )
    try:
        X_out, y_out = sampler.fit_resample(X, y)
    except Exception as exc:
        raise TrainingBalanceContractError(
            "SMOTE rejected the validated training partition.",
            code="smote_fit_resample_failed",
            diagnostics={"reason": type(exc).__name__},
        ) from exc
    X_out = np.asarray(X_out, dtype=float)
    y_out = np.asarray(y_out).ravel()
    synthetic_rows = int(y_out.size - y.size)
    provenance = _make_provenance(
        config=config,
        seed=seed,
        X_input=X,
        y_input=y,
        X_output=X_out,
        y_output=y_out,
        context_input=context,
        context_output=None,
        synthetic_rows=synthetic_rows,
        diagnostics={
            "sampling_strategy": config.smote_sampling_strategy,
            "k_neighbors": int(config.smote_k_neighbors),
            "synthetic_context_rows_persisted": 0,
        },
    )
    return TrainingBalanceResult(X_out, y_out, None, None, provenance)


def _pooled_sd(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(first.size + second.size - 2)
    if denominator <= 0:
        return 0.0
    variance = (
        (first.size - 1) * float(np.var(first, ddof=1))
        + (second.size - 1) * float(np.var(second, ddof=1))
    ) / float(denominator)
    return float(math.sqrt(max(0.0, variance)))


def _lineage_sort_keys(
    context: FitResamplingContext | None,
    *,
    n_rows: int,
) -> tuple[tuple[str, ...], ...]:
    if context is None:
        return tuple((f"position:{index:020d}",) for index in range(int(n_rows)))
    keys: list[tuple[str, ...]] = []
    for row_id in context.row_ids:
        typed = tuple(typed_scalar_key(row_id))
        keys.append((_sha256_json(list(typed)), *typed))
    return tuple(keys)


def _validate_random_sampler_partition(
    y: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    """Return class counts after enforcing the bounded v2 contract."""

    _, _, counts = _class_encoding(y)
    if np.all(counts == counts[0]):
        _raise(
            f"{method}_balanced_input",
            f"{method} requires at least one underrepresented predictive class.",
        )
    if int(np.min(counts)) < 2:
        _raise(
            f"{method}_class_too_small",
            f"{method} requires at least two original rows in every class.",
            minimum_required=2,
            observed_minimum=int(np.min(counts)),
        )
    return counts


def _coerce_sampler_indices(
    sampler: Any,
    *,
    method: str,
    input_rows: int,
    output_rows: int,
    require_unique: bool,
) -> np.ndarray:
    raw = getattr(sampler, "sample_indices_", None)
    if raw is None:
        _raise(
            f"{method}_sample_indices_unavailable",
            f"{method} did not expose the required source-row lineage.",
        )
    indices = np.asarray(raw)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        _raise(
            f"{method}_sample_indices_invalid",
            f"{method} returned invalid source-row lineage.",
        )
    indices = np.asarray(indices, dtype=int)
    if int(indices.size) != int(output_rows):
        _raise(
            f"{method}_sample_indices_invalid",
            f"{method} returned source-row lineage with the wrong length.",
            expected_rows=int(output_rows),
            observed_rows=int(indices.size),
        )
    if np.any(indices < 0) or np.any(indices >= int(input_rows)):
        _raise(
            f"{method}_sample_indices_invalid",
            f"{method} returned out-of-bounds source-row lineage.",
        )
    if require_unique and int(np.unique(indices).size) != int(indices.size):
        _raise(
            f"{method}_sample_indices_not_unique",
            f"{method} must select source rows without replacement.",
        )
    return indices


def _coerce_sampler_output(
    X_output: Any,
    y_output: Any,
    *,
    method: str,
    input_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        X_out = np.asarray(X_output, dtype=float)
        y_out = np.asarray(y_output)
    except (TypeError, ValueError) as exc:
        _raise(
            f"{method}_output_invalid",
            f"{method} returned output that cannot be represented as numeric arrays.",
            reason=type(exc).__name__,
        )
    if (
        X_out.ndim != 2
        or int(X_out.shape[1]) != int(input_features)
        or y_out.ndim != 1
        or int(y_out.size) != int(X_out.shape[0])
    ):
        _raise(
            f"{method}_output_invalid",
            f"{method} returned arrays with an invalid shape.",
            x_shape=list(X_out.shape),
            y_shape=list(y_out.shape),
            expected_features=int(input_features),
        )
    if not np.all(np.isfinite(X_out)):
        _raise(
            f"{method}_output_invalid",
            f"{method} returned non-finite feature values.",
        )
    return X_out, np.asarray(y_out).ravel()


def _apply_random_over(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: TrainingBalanceConfig,
    seed: int,
    context: FitResamplingContext | None,
    sample_weight: np.ndarray | None,
) -> TrainingBalanceResult:
    if sample_weight is not None:
        _raise(
            "random_over_sample_weight_unsupported",
            "random_over plus caller-provided sample weights is unsupported in v2.",
        )
    input_counts = _validate_random_sampler_partition(y, method="random_over")
    try:
        from imblearn.over_sampling import RandomOverSampler
    except ImportError as exc:
        raise TrainingBalanceContractError(
            "random_over requires the optional compatibility-bounded dependency; "
            "install it with `pip install 'tabnetics[balancing]'`.",
            code="random_over_dependency_unavailable",
            diagnostics={"extra": "balancing"},
        ) from exc
    sampler = RandomOverSampler(sampling_strategy="auto", random_state=int(seed))
    try:
        X_out, y_out = sampler.fit_resample(X, y)
    except Exception as exc:
        raise TrainingBalanceContractError(
            "random_over rejected the validated training partition.",
            code="random_over_fit_resample_failed",
            diagnostics={"reason": type(exc).__name__},
        ) from exc
    X_out, y_out = _coerce_sampler_output(
        X_out,
        y_out,
        method="random_over",
        input_features=int(X.shape[1]),
    )
    _, _, output_counts = _class_encoding(y_out)
    expected_counts = np.full(input_counts.size, int(np.max(input_counts)), dtype=int)
    if not np.array_equal(output_counts, expected_counts):
        _raise(
            "random_over_auto_counts_invalid",
            "random_over did not produce the class counts required by sampling_strategy='auto'.",
            expected_class_counts=expected_counts,
            observed_class_counts=output_counts,
        )
    indices = _coerce_sampler_indices(
        sampler,
        method="random_over",
        input_rows=int(y.size),
        output_rows=int(y_out.size),
        require_unique=False,
    )
    if not np.array_equal(X_out, X[indices]) or not np.array_equal(y_out, y[indices]):
        _raise(
            "random_over_sample_indices_misaligned",
            "random_over source-row lineage is not aligned with its output arrays.",
        )
    reuse_counts = np.bincount(indices, minlength=int(y.size))
    reused = reuse_counts[reuse_counts > 1]
    reuse_histogram = {
        str(int(reuse)): int(np.sum(reuse_counts == reuse))
        for reuse in np.unique(reuse_counts[reuse_counts > 0]).tolist()
    }
    duplicated_by_class = tuple(
        int(output - original)
        for original, output in zip(input_counts.tolist(), output_counts.tolist())
    )
    provenance = _make_provenance(
        config=config,
        seed=seed,
        X_input=X,
        y_input=y,
        X_output=X_out,
        y_output=y_out,
        context_input=context,
        context_output=None,
        diagnostics={
            "sampling_strategy": "auto",
            "source_indices_fingerprint": _array_fingerprint(indices),
            "source_rows_reused": int(reused.size),
            "source_row_reuse_total": int(np.sum(np.maximum(reuse_counts - 1, 0))),
            "max_source_row_reuse": int(np.max(reuse_counts)),
            "source_reuse_histogram": reuse_histogram,
            "duplicated_rows": int(y_out.size - y.size),
            "duplicated_class_counts": duplicated_by_class,
            "synthetic_rows": 0,
            "output_context_persisted": False,
        },
    )
    return TrainingBalanceResult(X_out, y_out, None, None, provenance)


def _apply_random_under(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: TrainingBalanceConfig,
    seed: int,
    context: FitResamplingContext | None,
    sample_weight: np.ndarray | None,
) -> TrainingBalanceResult:
    input_counts = _validate_random_sampler_partition(y, method="random_under")
    try:
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError as exc:
        raise TrainingBalanceContractError(
            "random_under requires the optional compatibility-bounded dependency; "
            "install it with `pip install 'tabnetics[balancing]'`.",
            code="random_under_dependency_unavailable",
            diagnostics={"extra": "balancing"},
        ) from exc
    sampler = RandomUnderSampler(
        sampling_strategy="auto",
        random_state=int(seed),
        replacement=False,
    )
    try:
        X_out, y_out = sampler.fit_resample(X, y)
    except Exception as exc:
        raise TrainingBalanceContractError(
            "random_under rejected the validated training partition.",
            code="random_under_fit_resample_failed",
            diagnostics={"reason": type(exc).__name__},
        ) from exc
    X_out, y_out = _coerce_sampler_output(
        X_out,
        y_out,
        method="random_under",
        input_features=int(X.shape[1]),
    )
    _, _, output_counts = _class_encoding(y_out)
    expected_counts = np.full(input_counts.size, int(np.min(input_counts)), dtype=int)
    if not np.array_equal(output_counts, expected_counts):
        _raise(
            "random_under_auto_counts_invalid",
            "random_under did not produce the class counts required by sampling_strategy='auto'.",
            expected_class_counts=expected_counts,
            observed_class_counts=output_counts,
        )
    indices = _coerce_sampler_indices(
        sampler,
        method="random_under",
        input_rows=int(y.size),
        output_rows=int(y_out.size),
        require_unique=True,
    )
    if not np.array_equal(X_out, X[indices]) or not np.array_equal(y_out, y[indices]):
        _raise(
            "random_under_sample_indices_misaligned",
            "random_under source-row lineage is not aligned with its output arrays.",
        )
    weight_out = None if sample_weight is None else np.asarray(sample_weight[indices], dtype=float)
    context_out = None if context is None else context.take(
        indices,
        parent_split_fingerprint=context.fingerprint,
    )
    retained_by_class = tuple(int(value) for value in output_counts.tolist())
    dropped_by_class = tuple(
        int(original - retained)
        for original, retained in zip(input_counts.tolist(), output_counts.tolist())
    )
    lineage_keys = _lineage_sort_keys(context, n_rows=int(y.size))
    selected_lineage = [lineage_keys[int(index)] for index in indices.tolist()]
    provenance = _make_provenance(
        config=config,
        seed=seed,
        X_input=X,
        y_input=y,
        X_output=X_out,
        y_output=y_out,
        context_input=context,
        context_output=context_out,
        diagnostics={
            "sampling_strategy": "auto",
            "replacement": False,
            "source_indices_fingerprint": _array_fingerprint(indices),
            "selected_lineage_fingerprint": _sha256_json(selected_lineage),
            "retained_class_counts": retained_by_class,
            "dropped_class_counts": dropped_by_class,
            "dropped_rows": int(y.size - y_out.size),
            "sample_weights_subset": sample_weight is not None,
            "output_context_persisted": context_out is not None,
        },
    )
    return TrainingBalanceResult(X_out, y_out, weight_out, context_out, provenance)


def _max_abs_smd(X: np.ndarray, target: np.ndarray) -> float:
    first = X[target == 1]
    second = X[target == 0]
    pooled = np.sqrt((np.var(first, axis=0, ddof=1) + np.var(second, axis=0, ddof=1)) / 2.0)
    differences = np.abs(np.mean(first, axis=0) - np.mean(second, axis=0))
    smd = np.divide(differences, pooled, out=np.zeros_like(differences), where=pooled > 0.0)
    return float(np.max(smd)) if smd.size else 0.0


def _apply_propensity_match(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: TrainingBalanceConfig,
    seed: int,
    context: FitResamplingContext | None,
    sample_weight: np.ndarray | None,
) -> TrainingBalanceResult:
    _, encoded, counts = _class_encoding(y)
    if counts.size != 2:
        _raise(
            "propensity_binary_only",
            "Propensity matching v1 is limited to binary predictive classification.",
            n_classes=int(counts.size),
        )
    if int(counts[0]) == int(counts[1]):
        _raise(
            "propensity_balanced_input",
            "Propensity matching requires one strictly smaller predictive class.",
        )
    minority_code = int(np.argmin(counts))
    target = np.asarray(encoded == minority_code, dtype=int)
    minimum = int(config.propensity_n_splits)
    if int(np.min(counts)) < minimum:
        _raise(
            "propensity_class_too_small",
            "Each class must contain at least propensity_n_splits rows for cross-fitting.",
            minimum_required=minimum,
            observed_minimum=int(np.min(counts)),
        )
    splitter = StratifiedKFold(
        n_splits=int(config.propensity_n_splits),
        shuffle=True,
        random_state=int(seed),
    )
    estimator = LogisticRegression(max_iter=2000, random_state=int(seed))
    try:
        probabilities = cross_val_predict(
            estimator,
            X,
            target,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
    except Exception as exc:
        raise TrainingBalanceContractError(
            "The fold-local propensity model could not produce complete OOF scores.",
            code="propensity_crossfit_failed",
            diagnostics={"reason": type(exc).__name__},
        ) from exc
    clip = float(config.propensity_probability_clip)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), clip, 1.0 - clip)
    logits = np.log(probabilities / (1.0 - probabilities))
    minority_logits = logits[target == 1]
    majority_logits = logits[target == 0]
    support_low = float(max(np.min(minority_logits), np.min(majority_logits)))
    support_high = float(min(np.max(minority_logits), np.max(majority_logits)))
    if not support_low <= support_high:
        _raise(
            "propensity_empty_common_support",
            "The OOF propensity score ranges have no common support.",
        )
    within_support = (logits >= support_low) & (logits <= support_high)
    lineage_keys = _lineage_sort_keys(context, n_rows=int(y.size))
    minority_indices = np.flatnonzero((target == 1) & within_support)
    majority_indices = np.flatnonzero((target == 0) & within_support)
    if minority_indices.size < 2 or majority_indices.size < 2:
        _raise(
            "propensity_insufficient_common_support",
            "Common support must retain at least two rows from each class.",
            minority_within_support=int(minority_indices.size),
            majority_within_support=int(majority_indices.size),
        )
    pooled_sd = _pooled_sd(minority_logits, majority_logits)
    caliper = float(config.propensity_caliper_sd * pooled_sd)
    if not math.isfinite(caliper) or caliper <= 0.0:
        _raise(
            "propensity_degenerate_caliper",
            "The pooled OOF logit variation does not define a positive caliper.",
        )
    ordered_minority = sorted(
        (int(value) for value in minority_indices.tolist()),
        key=lambda index: lineage_keys[index],
    )
    unused = set(int(value) for value in majority_indices.tolist())
    pairs: list[tuple[int, int]] = []
    for minority_index in ordered_minority:
        candidates = sorted(
            unused,
            key=lambda majority_index: (
                abs(float(logits[minority_index] - logits[majority_index])),
                lineage_keys[int(majority_index)],
            ),
        )
        if not candidates:
            _raise(
                "propensity_incomplete_matching",
                "A minority row surviving support trimming has no unused majority match.",
                matched_pairs=len(pairs),
                minority_within_support=int(minority_indices.size),
            )
        majority_index = int(candidates[0])
        distance = abs(float(logits[minority_index] - logits[majority_index]))
        if distance > caliper:
            _raise(
                "propensity_caliper_match_unavailable",
                "A minority row surviving support trimming has no caliper-valid unused majority match.",
                matched_pairs=len(pairs),
                minority_within_support=int(minority_indices.size),
            )
        pairs.append((int(minority_index), majority_index))
        unused.remove(majority_index)
    if len(pairs) < 2:
        _raise(
            "propensity_too_few_pairs",
            "Propensity matching must retain at least two complete pairs.",
            matched_pairs=len(pairs),
        )
    selected = np.asarray(
        sorted(
            (index for pair in pairs for index in pair),
            key=lambda index: lineage_keys[int(index)],
        ),
        dtype=int,
    )
    X_out = np.asarray(X[selected], dtype=float)
    y_out = np.asarray(y[selected]).ravel()
    target_out = np.asarray(target[selected], dtype=int)
    before_smd = _max_abs_smd(X, target)
    after_smd = _max_abs_smd(X_out, target_out)
    if after_smd > before_smd + 1e-12:
        _raise(
            "propensity_balance_worsened",
            "Propensity matching worsened maximum standardized predictor imbalance.",
            before_max_abs_smd=before_smd,
            after_max_abs_smd=after_smd,
        )
    weight_out = None if sample_weight is None else np.asarray(sample_weight[selected], dtype=float)
    context_out = None if context is None else context.take(
        selected,
        parent_split_fingerprint=context.fingerprint,
    )
    provenance = _make_provenance(
        config=config,
        seed=seed,
        X_input=X,
        y_input=y,
        X_output=X_out,
        y_output=y_out,
        context_input=context,
        context_output=context_out,
        matched_pairs=len(pairs),
        diagnostics={
            "predictive_balancing_only": True,
            "causal_estimand": None,
            "propensity_n_splits": int(config.propensity_n_splits),
            "probability_clip": clip,
            "support_logit_low": support_low,
            "support_logit_high": support_high,
            "minority_support_trimmed": int(np.sum((target == 1) & ~within_support)),
            "majority_support_trimmed": int(np.sum((target == 0) & ~within_support)),
            "minority_match_drops": 0,
            "majority_unmatched_within_support": int(len(unused)),
            "matching_with_replacement": False,
            "max_source_row_reuse": 1,
            "pooled_oof_logit_sd": pooled_sd,
            "caliper": caliper,
            "before_max_abs_smd": before_smd,
            "after_max_abs_smd": after_smd,
        },
    )
    return TrainingBalanceResult(X_out, y_out, weight_out, context_out, provenance)


def apply_training_balance(
    X: Any,
    y: Sequence[Any],
    *,
    config: TrainingBalanceConfig | Mapping[str, Any] | None = None,
    pipeline_seed: int = 42,
    context: FitResamplingContext | None = None,
    sample_weight: Sequence[float] | None = None,
) -> TrainingBalanceResult:
    """Apply one deterministic adapter to one already-transformed train partition."""

    resolved = _coerce_config(config)
    X_arr, y_arr = _validate_inputs(
        X,
        y,
        context=context,
        require_continuous=resolved.enabled,
    )
    effective_weights = sample_weight
    if effective_weights is None and context is not None and context.sample_weights:
        effective_weights = context.sample_weights
    weights = None
    if effective_weights is not None:
        weights = np.asarray(
            coerce_sample_weights(
                effective_weights,
                n_rows=int(y_arr.size),
                field_name="training_balance_sample_weight",
                require_positive_mass=True,
            ),
            dtype=float,
        )
        if context is not None and context.sample_weights and not np.array_equal(
            weights,
            np.asarray(context.sample_weights, dtype=float),
        ):
            _raise(
                "sample_weight_context_mismatch",
                "Caller weights disagree with context.sample_weights.",
            )
    seed = resolved.effective_seed(int(pipeline_seed))
    if resolved.method == "none":
        provenance = _make_provenance(
            config=resolved,
            seed=seed,
            X_input=X_arr,
            y_input=y_arr,
            X_output=X_arr,
            y_output=y_arr,
            context_input=context,
            context_output=context,
            diagnostics={"status": "disabled"},
        )
        return TrainingBalanceResult(X_arr, y_arr, weights, context, provenance)
    if resolved.method == "smote":
        return _apply_smote(
            X_arr,
            y_arr,
            config=resolved,
            seed=seed,
            context=context,
            sample_weight=weights,
        )
    if resolved.method == "propensity_match":
        return _apply_propensity_match(
            X_arr,
            y_arr,
            config=resolved,
            seed=seed,
            context=context,
            sample_weight=weights,
        )
    if resolved.method == "random_over":
        return _apply_random_over(
            X_arr,
            y_arr,
            config=resolved,
            seed=seed,
            context=context,
            sample_weight=weights,
        )
    return _apply_random_under(
        X_arr,
        y_arr,
        config=resolved,
        seed=seed,
        context=context,
        sample_weight=weights,
    )


__all__ = [
    "TRAINING_BALANCE_METHODS",
    "TRAINING_BALANCE_SCHEMA_VERSION",
    "TrainingBalanceConfig",
    "TrainingBalanceContractError",
    "TrainingBalanceProvenance",
    "TrainingBalanceResult",
    "apply_training_balance",
]

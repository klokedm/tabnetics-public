"""Restricted non-executable inference bundles for allowlisted DFFS routes.

The legacy ``DFFSReproducibleModel`` format stores arbitrary pickle payloads.
This module deliberately does not extend that format.  It encodes only a
small, inspectable route whose inference semantics fit in JSON:

``SimpleImputer(strategy="median") -> StandardScaler -> positional columns
-> sklearn LogisticRegression coefficients``.

The loader never invokes pickle/joblib/cloudpickle, imports a class named by a
bundle, or executes a serialized callable.  A SHA-256 digest catches accidental
or unauthorised manifest changes, while the allowlisted JSON schema remains the
security boundary when a party can recompute that digest.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from tabnetics.datasets.schema import (
    DatasetSchema,
    FeatureRole,
    SchemaContractError,
)


SAFE_DFFS_BUNDLE_ARTIFACT = "tabnetics_dffs_safe_inference_bundle"
SAFE_DFFS_BUNDLE_SCHEMA_VERSION = "3"
SAFE_DFFS_BUNDLE_TRUST_MODE = "non_executable_json_allowlist"
SAFE_DFFS_ROUTE_ID = "dffs_numeric_median_standard_lr_positional_v3"
_LEGACY_SAFE_DFFS_BUNDLE_SCHEMA_VERSION = "2"
_LEGACY_SAFE_DFFS_ROUTE_ID = "dffs_numeric_median_standard_lr_positional_v2"

_MAX_FEATURES = 1_000_000
_MAX_CLASSES = 100_000
_RUNTIME_TYPE = ("tabnetics.pipeline.pipeline", "DFFSReproducibleModel")
_COMPONENTS_TYPE = ("tabnetics.pipeline.pipeline", "FittedPipelineComponents")
_SAFE_SELECTOR_TYPES = {
    ("tabnetics.feature_selection.base", "FeatureSelector"),
    ("tabnetics.pipeline.pipeline", "_FixedIndexFeatureSelector"),
    ("tabnetics.pipeline.pipeline", "_IdentityFeatureSelector"),
}


class DFFSSafeBundleError(ValueError):
    """Base error for the restricted non-executable DFFS bundle codec."""


class UnsupportedSafeBundleStateError(DFFSSafeBundleError):
    """Raised when a fitted route cannot be represented without executable state."""


class SafeBundleIntegrityError(DFFSSafeBundleError):
    """Raised when a bundle digest or its bound content hashes do not verify."""


class SafeBundleSchemaError(DFFSSafeBundleError):
    """Raised when a bundle or inference input violates the safe schema contract."""


def _type_id(value: Any) -> tuple[str, str]:
    value_type = type(value)
    return (str(value_type.__module__), str(value_type.__qualname__))


def _canonical_json(value: Any) -> str:
    """Canonical, finite JSON used for every digest in this codec."""

    return json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _plain_json(value: Any) -> Any:
    """Return strict JSON data without stringifying arbitrary runtime objects."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        out = float(value)
        if not np.isfinite(out):
            raise SafeBundleSchemaError("Safe bundle JSON cannot contain NaN or infinity.")
        return out
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _plain_json(value.tolist())
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SafeBundleSchemaError(
                    "Safe bundle JSON mappings require string keys; arbitrary keys are not supported."
                )
            out[key] = _plain_json(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    raise SafeBundleSchemaError(
        "Safe bundle JSON accepts only null, booleans, strings, finite numbers, "
        f"lists, and string-keyed mappings; got {type(value).__name__}."
    )


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SafeBundleSchemaError(f"{field} must be a JSON object.")
    return {str(key): item for key, item in value.items()}


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    field: str,
    required: set[str],
) -> None:
    received = set(value)
    if received != required:
        raise SafeBundleSchemaError(
            f"{field} has an unsupported shape; expected keys={sorted(required)!r}, "
            f"received keys={sorted(received)!r}."
        )


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SafeBundleSchemaError(f"{field} must be a non-empty string.")
    return value


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SafeBundleSchemaError(f"{field} must be an integer.")
    out = int(value)
    if out < minimum:
        raise SafeBundleSchemaError(f"{field} must be >= {minimum}.")
    return out


def _finite_vector(value: Any, *, field: str, length: int | None = None) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SafeBundleSchemaError(f"{field} must be a finite numeric vector.") from exc
    if out.ndim != 1:
        raise SafeBundleSchemaError(f"{field} must be one-dimensional.")
    if length is not None and int(out.size) != int(length):
        raise SafeBundleSchemaError(
            f"{field} width mismatch; expected={length}, received={out.size}."
        )
    if not np.all(np.isfinite(out)):
        raise SafeBundleSchemaError(f"{field} must contain only finite values.")
    return np.asarray(out, dtype=float)


def _finite_matrix(
    value: Any,
    *,
    field: str,
    rows: int | None = None,
    columns: int | None = None,
) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SafeBundleSchemaError(f"{field} must be a finite numeric matrix.") from exc
    if out.ndim != 2:
        raise SafeBundleSchemaError(f"{field} must be two-dimensional.")
    if rows is not None and int(out.shape[0]) != int(rows):
        raise SafeBundleSchemaError(
            f"{field} row mismatch; expected={rows}, received={out.shape[0]}."
        )
    if columns is not None and int(out.shape[1]) != int(columns):
        raise SafeBundleSchemaError(
            f"{field} column mismatch; expected={columns}, received={out.shape[1]}."
        )
    if not np.all(np.isfinite(out)):
        raise SafeBundleSchemaError(f"{field} must contain only finite values.")
    return np.asarray(out, dtype=float)


def _indices(value: Any, *, field: str, width: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise UnsupportedSafeBundleStateError(
            f"{field} must be a one-dimensional integer positional index array."
        )
    out = np.asarray(raw, dtype=int)
    if out.size == 0:
        raise UnsupportedSafeBundleStateError(f"{field} cannot be empty for a safe route.")
    if np.any(out < 0) or np.any(out >= int(width)):
        raise UnsupportedSafeBundleStateError(
            f"{field} contains an index outside [0, {int(width)})."
        )
    if np.unique(out).size != out.size:
        raise UnsupportedSafeBundleStateError(f"{field} contains duplicate positions.")
    return out


def _optional_indices_or_identity(value: Any, *, field: str, width: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.size == 0:
        return np.arange(int(width), dtype=int)
    return _indices(raw, field=field, width=width)


def _encode_label(value: Any) -> dict[str, Any]:
    if isinstance(value, (str, np.str_)):
        return {"type": "str", "value": str(value)}
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return {"type": "int", "value": str(int(value))}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise UnsupportedSafeBundleStateError("LogisticRegression class labels must be finite.")
        return {"type": "float", "value": number}
    raise UnsupportedSafeBundleStateError(
        "Safe bundles support only string, boolean, integer, and finite floating "
        f"LogisticRegression labels; got {type(value).__name__}."
    )


def _encode_labels(values: Sequence[Any]) -> list[dict[str, Any]]:
    out = [_encode_label(value) for value in values]
    if len(out) < 2:
        raise UnsupportedSafeBundleStateError("A safe LogisticRegression route requires at least two classes.")
    if len({_canonical_json(value) for value in out}) != len(out):
        raise UnsupportedSafeBundleStateError("LogisticRegression class order contains duplicate labels.")
    return out


def _decode_label(value: Any, *, field: str) -> Any:
    record = _require_mapping(value, field=field)
    _require_exact_keys(record, field=field, required={"type", "value"})
    kind = _require_string(record.get("type"), field=f"{field}.type")
    raw = record.get("value")
    if kind == "str":
        if not isinstance(raw, str):
            raise SafeBundleSchemaError(f"{field}.value must be a string.")
        return raw
    if kind == "bool":
        if not isinstance(raw, bool):
            raise SafeBundleSchemaError(f"{field}.value must be a boolean.")
        return raw
    if kind == "int":
        if not isinstance(raw, str):
            raise SafeBundleSchemaError(f"{field}.value must be a decimal string.")
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise SafeBundleSchemaError(f"{field}.value is not a valid integer.") from exc
        if str(parsed) != raw:
            raise SafeBundleSchemaError(f"{field}.value is not canonical integer text.")
        return parsed
    if kind == "float":
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise SafeBundleSchemaError(f"{field}.value must be a finite number.")
        parsed = float(raw)
        if not np.isfinite(parsed):
            raise SafeBundleSchemaError(f"{field}.value must be finite.")
        return parsed
    raise SafeBundleSchemaError(f"{field}.type {kind!r} is not supported.")


def _decode_labels(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise SafeBundleSchemaError("class_order must be a JSON array.")
    if len(value) < 2 or len(value) > _MAX_CLASSES:
        raise SafeBundleSchemaError("class_order must contain a supported number of classes.")
    out = tuple(_decode_label(item, field=f"class_order[{index}]") for index, item in enumerate(value))
    encoded = [_encode_label(item) for item in out]
    if len({_canonical_json(item) for item in encoded}) != len(encoded):
        raise SafeBundleSchemaError("class_order contains duplicate labels.")
    return out


def _schema_record(schema: DatasetSchema) -> dict[str, Any]:
    return _plain_json(schema.to_record())


def _generated_numeric_schema(n_features: int) -> DatasetSchema:
    if n_features <= 0 or n_features > _MAX_FEATURES:
        raise UnsupportedSafeBundleStateError(
            f"Safe numeric route requires 1..{_MAX_FEATURES} input features."
        )
    return DatasetSchema.from_input(
        np.zeros((1, int(n_features)), dtype=float),
        metadata={
            "safe_bundle_schema_origin": "generated_positional_numeric",
            "safe_bundle_route": SAFE_DFFS_ROUTE_ID,
        },
    )


def _validate_numeric_schema(schema: DatasetSchema, *, field: str, width: int) -> None:
    if type(schema) is not DatasetSchema:
        raise UnsupportedSafeBundleStateError(f"{field} must be an immutable DatasetSchema.")
    if int(schema.n_features) != int(width):
        raise UnsupportedSafeBundleStateError(
            f"{field} width mismatch; expected={width}, received={schema.n_features}."
        )
    numeric_roles = {
        FeatureRole.CONTINUOUS,
        FeatureRole.COUNT,
        FeatureRole.BINARY,
        FeatureRole.ORDINAL,
        FeatureRole.DERIVED,
    }
    for feature in schema.features:
        if feature.role not in numeric_roles:
            raise UnsupportedSafeBundleStateError(
                "The v2 safe route is numeric-only; "
                f"feature {feature.name!r} has unsupported role {feature.role.value!r}."
            )
        try:
            dtype = np.dtype(feature.dtype)
        except TypeError as exc:
            raise UnsupportedSafeBundleStateError(
                f"Feature {feature.name!r} has unsupported non-numeric dtype {feature.dtype!r}."
            ) from exc
        if dtype.kind not in {"b", "i", "u", "f"}:
            raise UnsupportedSafeBundleStateError(
                f"Feature {feature.name!r} has unsupported non-numeric dtype {feature.dtype!r}."
            )


def _schema_from_record(value: Any, *, field: str, width: int) -> DatasetSchema:
    record = _require_mapping(value, field=field)
    try:
        schema = DatasetSchema.from_record(record)
    except (SchemaContractError, TypeError, ValueError) as exc:
        raise SafeBundleSchemaError(f"{field} is not a valid bound DatasetSchema.") from exc
    try:
        _validate_numeric_schema(schema, field=field, width=width)
    except UnsupportedSafeBundleStateError as exc:
        raise SafeBundleSchemaError(str(exc)) from exc
    return schema


def _same_feature_contract(left: DatasetSchema, right: DatasetSchema) -> bool:
    return tuple(feature.to_record() for feature in left.features) == tuple(
        feature.to_record() for feature in right.features
    )


def _assert_runtime_disabled_state(runtime: Any) -> None:
    batch_model = getattr(runtime, "batch_model", None)
    if batch_model is not None:
        if not isinstance(batch_model, Mapping):
            raise UnsupportedSafeBundleStateError("Safe route rejects non-mapping batch state.")
        mode = str(batch_model.get("mode", "none") or "none").strip().lower()
        labels = batch_model.get("label_to_code", {})
        if mode != "none" or bool(labels):
            raise UnsupportedSafeBundleStateError(
                "Safe route does not support batch-correction state."
            )

    face_meta = getattr(runtime, "face_meta", {})
    if not isinstance(face_meta, Mapping):
        raise UnsupportedSafeBundleStateError("Safe route rejects malformed face-projection state.")
    if bool(face_meta.get("face_projection_applied", False)) or any(
        getattr(runtime, field, None) is not None for field in ("face_pca", "face_lda")
    ):
        raise UnsupportedSafeBundleStateError("Safe route does not support face-projection state.")

    ratio_meta = getattr(runtime, "ratio_meta", {})
    if not isinstance(ratio_meta, Mapping):
        raise UnsupportedSafeBundleStateError("Safe route rejects malformed ratio state.")
    if bool(ratio_meta.get("ratio_features_applied", False)) or bool(ratio_meta.get("ratio_pairs", [])):
        raise UnsupportedSafeBundleStateError("Safe route does not support Stage-1 ratio features.")

    distribution_plan = getattr(runtime, "distribution_plan", {})
    if not isinstance(distribution_plan, Mapping):
        raise UnsupportedSafeBundleStateError("Safe route rejects malformed distribution-fit state.")
    if bool(distribution_plan.get("apply_cdf_transform", False)) or bool(
        distribution_plan.get("feature_plans", [])
    ) or bool(distribution_plan.get("dist_feature_indices", [])):
        raise UnsupportedSafeBundleStateError(
            "Safe route does not support distribution-fit transform state."
        )

    folding_meta = getattr(runtime, "folding_meta", {})
    if not isinstance(folding_meta, Mapping):
        raise UnsupportedSafeBundleStateError("Safe route rejects malformed folding state.")
    if bool(folding_meta.get("folding_applied", False)) or getattr(
        runtime, "folding_transformer", None
    ) is not None:
        raise UnsupportedSafeBundleStateError("Safe route does not support folding state.")
    for field in ("folding_standardize_mean", "folding_standardize_scale"):
        value = getattr(runtime, field, None)
        if value is not None and np.asarray(value).size:
            raise UnsupportedSafeBundleStateError("Safe route does not support folding standardisation state.")

    stage2_ratio = getattr(runtime, "stage2_ratio_meta", {})
    if not isinstance(stage2_ratio, Mapping):
        raise UnsupportedSafeBundleStateError("Safe route rejects malformed Stage-2 ratio state.")
    if bool(stage2_ratio.get("stage2_ratio_features_applied", False)) or bool(
        stage2_ratio.get("stage2_ratio_pairs", [])
    ):
        raise UnsupportedSafeBundleStateError("Safe route does not support Stage-2 ratio features.")


def _extract_positional_selection(runtime: Any, *, n_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefilter = _optional_indices_or_identity(
        getattr(runtime, "prefilter_indices", tuple()),
        field="prefilter_indices",
        width=n_features,
    )
    variance = _optional_indices_or_identity(
        getattr(runtime, "variance_keep_indices", tuple()),
        field="variance_keep_indices",
        width=int(prefilter.size),
    )
    selector = getattr(runtime, "selector", None)
    if _type_id(selector) not in _SAFE_SELECTOR_TYPES:
        raise UnsupportedSafeBundleStateError(
            "Safe route only supports the canonical positional FeatureSelector or "
            "the pipeline's fixed/identity positional selector."
        )
    get_indices = getattr(selector, "get_selected_features_indices", None)
    if not callable(get_indices):
        raise UnsupportedSafeBundleStateError("Safe selector does not expose positional selected indices.")
    try:
        selector_indices = _indices(
            get_indices(),
            field="selector_indices",
            width=int(variance.size),
        )
    except (TypeError, ValueError) as exc:
        raise UnsupportedSafeBundleStateError(
            "Safe selector failed to provide deterministic positional indices."
        ) from exc
    selected = prefilter[variance[selector_indices]]
    if np.unique(selected).size != selected.size:
        raise UnsupportedSafeBundleStateError(
            "Composed safe selection route contains duplicate model-input positions."
        )
    return prefilter, variance, selector_indices, np.asarray(selected, dtype=int)


def _extract_preprocessing(runtime: Any, *, n_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imputer = getattr(runtime, "imputer", None)
    if type(imputer) is not SimpleImputer:
        raise UnsupportedSafeBundleStateError(
            "Safe route requires an exact sklearn SimpleImputer instance."
        )
    if str(getattr(imputer, "strategy", "") or "").strip().lower() != "median":
        raise UnsupportedSafeBundleStateError("Safe route supports only SimpleImputer(strategy='median').")
    missing_value = getattr(imputer, "missing_values", np.nan)
    if not (isinstance(missing_value, (float, np.floating)) and np.isnan(missing_value)):
        raise UnsupportedSafeBundleStateError("Safe route supports only NaN missing-value markers.")
    if bool(getattr(imputer, "add_indicator", False)) or getattr(imputer, "indicator_", None) is not None:
        raise UnsupportedSafeBundleStateError("Safe route does not support imputer indicator features.")
    if bool(getattr(imputer, "keep_empty_features", False)):
        raise UnsupportedSafeBundleStateError("Safe route does not support kept all-missing features.")
    statistics = _finite_vector(
        getattr(imputer, "statistics_", None),
        field="SimpleImputer.statistics_",
        length=n_features,
    )
    fitted_width = getattr(imputer, "n_features_in_", n_features)
    if int(fitted_width) != int(n_features):
        raise UnsupportedSafeBundleStateError("SimpleImputer fitted input width does not match runtime state.")

    scaler = getattr(runtime, "scaler_base", None)
    if type(scaler) is not StandardScaler:
        raise UnsupportedSafeBundleStateError(
            "Safe route requires an exact sklearn StandardScaler instance."
        )
    if not bool(getattr(scaler, "with_mean", False)) or not bool(
        getattr(scaler, "with_std", False)
    ):
        raise UnsupportedSafeBundleStateError(
            "Safe route requires StandardScaler(with_mean=True, with_std=True)."
        )
    mean = _finite_vector(getattr(scaler, "mean_", None), field="StandardScaler.mean_", length=n_features)
    scale = _finite_vector(getattr(scaler, "scale_", None), field="StandardScaler.scale_", length=n_features)
    if np.any(scale <= 0.0):
        raise UnsupportedSafeBundleStateError("StandardScaler.scale_ must be strictly positive.")
    fitted_width = getattr(scaler, "n_features_in_", n_features)
    if int(fitted_width) != int(n_features):
        raise UnsupportedSafeBundleStateError("StandardScaler fitted input width does not match runtime state.")
    return statistics, mean, scale


def _resolved_probability_mode(model: Any, *, n_classes: int) -> str:
    if n_classes == 2:
        return "binary"
    requested = getattr(model, "multi_class", None)
    if requested is None:
        # sklearn >= 1.8 removed the public multi_class parameter and uses
        # multinomial probabilities for multiclass LogisticRegression.
        return "multinomial"
    requested_text = str(requested).strip().lower()
    solver = str(getattr(model, "solver", "") or "").strip().lower()
    if requested_text == "ovr" or (requested_text == "auto" and solver == "liblinear"):
        return "ovr"
    if requested_text in {"auto", "multinomial"}:
        return "multinomial"
    raise UnsupportedSafeBundleStateError(
        f"Safe route does not support LogisticRegression multi_class={requested!r}."
    )


def _extract_classifier(model: Any, *, n_selected: int) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, str, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    if type(model) is not LogisticRegression:
        raise UnsupportedSafeBundleStateError(
            "Safe route supports only an exact fitted sklearn LogisticRegression; "
            "generic sklearn, custom, native, and DIAKRINO classifiers are not safe bundle state."
        )
    try:
        classes_raw = np.asarray(model.classes_).ravel()
        coef = _finite_matrix(getattr(model, "coef_", None), field="LogisticRegression.coef_")
        intercept = _finite_vector(getattr(model, "intercept_", None), field="LogisticRegression.intercept_")
    except AttributeError as exc:
        raise UnsupportedSafeBundleStateError("LogisticRegression is not fitted.") from exc
    classes = _encode_labels(classes_raw.tolist())
    n_classes = len(classes)
    expected_rows = 1 if n_classes == 2 else n_classes
    if coef.shape != (expected_rows, int(n_selected)):
        raise UnsupportedSafeBundleStateError(
            "LogisticRegression coefficient shape does not match safe selected features/classes."
        )
    if intercept.shape != (expected_rows,):
        raise UnsupportedSafeBundleStateError(
            "LogisticRegression intercept shape does not match safe selected features/classes."
        )
    fitted_width = getattr(model, "n_features_in_", n_selected)
    if int(fitted_width) != int(n_selected):
        raise UnsupportedSafeBundleStateError(
            "LogisticRegression fitted input width does not match safe selected features."
        )
    probability_mode = _resolved_probability_mode(model, n_classes=n_classes)
    descriptor = {
        "kind": "sklearn_logistic_regression_coefficients",
        "solver": str(getattr(model, "solver", "") or ""),
        "probability_mode": probability_mode,
        "n_features_in": int(n_selected),
    }
    return classes, coef, intercept, probability_mode, descriptor


def _safe_package_record() -> dict[str, str]:
    try:
        from tabnetics import __version__

        version = str(__version__)
    except Exception:  # pragma: no cover - package import is available in supported installs
        version = "unknown"
    return {
        "name": "tabnetics",
        "version": version,
        "codec": SAFE_DFFS_ROUTE_ID,
    }


def _normalise_config_snapshot(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise UnsupportedSafeBundleStateError("Fitted component config_snapshot must be a JSON mapping.")
    try:
        return _plain_json(value)
    except SafeBundleSchemaError as exc:
        raise UnsupportedSafeBundleStateError(
            "Fitted component config_snapshot is not strict JSON metadata."
        ) from exc


def _validate_training_balance_metadata(
    config: Mapping[str, Any],
    *,
    require_metadata: bool,
    legacy_v2: bool,
    error_type: type[Exception],
) -> None:
    """Validate non-executable balance config/provenance stored in a safe bundle."""

    raw_config = config.get("training_balance")
    raw_provenance = config.get("training_balance_provenance")
    if legacy_v2:
        if raw_config is not None or raw_provenance is not None:
            raise error_type(
                "Legacy safe-v2 bundles cannot declare training-balance metadata."
            )
        return
    if raw_config is None:
        if require_metadata:
            raise error_type("Safe-v3 bundle is missing training_balance metadata.")
        return
    if "training_balance_provenance" not in config:
        raise error_type(
            "Safe-v3 bundle is missing training_balance_provenance metadata."
        )
    try:
        balance_record = _require_mapping(raw_config, field="config.training_balance")
        _require_exact_keys(
            balance_record,
            field="config.training_balance",
            required={
                "method",
                "smote_sampling_strategy",
                "smote_k_neighbors",
                "propensity_n_splits",
                "propensity_probability_clip",
                "propensity_caliper_sd",
                "random_state",
                "schema_version",
            },
        )
        balance_schema = balance_record.get("schema_version")
        if balance_schema not in {"1.0", "2.0"}:
            raise SafeBundleSchemaError("training_balance schema_version is unsupported.")
        from .balancing import TrainingBalanceConfig

        if balance_schema == "1.0":
            if balance_record.get("method") not in {
                "none",
                "smote",
                "propensity_match",
            }:
                raise SafeBundleSchemaError(
                    "Legacy training_balance metadata declares an unsupported method."
                )
            migrated_record = dict(balance_record)
            migrated_record["schema_version"] = "2.0"
            balance = TrainingBalanceConfig.from_mapping(migrated_record)
            canonical_record = balance.to_dict()
            canonical_record["schema_version"] = "1.0"
            expected_config_fingerprint = _sha256_json(balance_record)
        else:
            balance = TrainingBalanceConfig.from_mapping(balance_record)
            canonical_record = balance.to_dict()
            expected_config_fingerprint = balance.fingerprint
        if canonical_record != balance_record:
            raise SafeBundleSchemaError("training_balance metadata is not canonical.")
        provenance = _require_mapping(
            {} if raw_provenance is None else raw_provenance,
            field="config.training_balance_provenance",
        )
        if balance.enabled and not provenance:
            raise SafeBundleSchemaError(
                "Enabled training balancing requires aggregate final-fit provenance."
            )
        for callsite, raw_record in provenance.items():
            record = _require_mapping(
                raw_record,
                field=f"config.training_balance_provenance.{callsite}",
            )
            _require_exact_keys(
                record,
                field=f"config.training_balance_provenance.{callsite}",
                required={
                    "schema_version",
                    "method",
                    "seed",
                    "config",
                    "config_fingerprint",
                    "input_fingerprint",
                    "output_fingerprint",
                    "input_lineage_fingerprint",
                    "output_lineage_fingerprint",
                    "input_rows",
                    "output_rows",
                    "input_class_counts",
                    "output_class_counts",
                    "synthetic_rows",
                    "matched_pairs",
                    "diagnostics",
                    "provenance_fingerprint",
                },
            )
            if (
                record.get("schema_version") != balance_schema
                or record.get("method") != balance.method
            ):
                raise SafeBundleSchemaError("Training-balance provenance disagrees with its config.")
            if record.get("config") != balance_record:
                raise SafeBundleSchemaError("Training-balance provenance embeds a different config.")
            config_fingerprint = _require_string(
                record.get("config_fingerprint"),
                field=f"config.training_balance_provenance.{callsite}.config_fingerprint",
            )
            if not hmac.compare_digest(
                config_fingerprint,
                expected_config_fingerprint,
            ):
                raise SafeBundleIntegrityError(
                    "Training-balance provenance config fingerprint did not verify."
                )
            unsigned = dict(record)
            received = _require_string(
                unsigned.pop("provenance_fingerprint", None),
                field=f"config.training_balance_provenance.{callsite}.provenance_fingerprint",
            )
            if not hmac.compare_digest(received, _sha256_json(unsigned)):
                raise SafeBundleIntegrityError(
                    "Training-balance aggregate provenance fingerprint did not verify."
                )
            for digest_field in (
                "input_fingerprint",
                "output_fingerprint",
                "input_lineage_fingerprint",
                "output_lineage_fingerprint",
            ):
                digest = _require_string(
                    record.get(digest_field),
                    field=f"config.training_balance_provenance.{callsite}.{digest_field}",
                )
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise SafeBundleSchemaError(
                        f"Training-balance provenance {digest_field} is not SHA-256."
                    )
    except (SafeBundleSchemaError, SafeBundleIntegrityError, ValueError, TypeError) as exc:
        if isinstance(exc, SafeBundleIntegrityError):
            raise
        if isinstance(exc, error_type):
            raise
        raise error_type(str(exc)) from exc


def _config_has_enabled_forbidden_state(config: Mapping[str, Any]) -> str | None:
    """Detect actual DIAKRINO/native execution state, not inactive tuning defaults.

    A config snapshot intentionally records values such as
    ``diakrino_prefilter_max_extras=50`` and a default score-column name even while
    every DIAKRINO switch is off.  Treating every nonzero ``diakrino_*`` field as active
    would make the documented numeric route impossible to export.  This list
    therefore contains only execution toggles, sidecar/checkpoint paths, and
    priors whose nonzero value changes the fitted route.
    """

    active_boolean_keys = {
        "diakrino_family_prescreen_enabled",
        "dist_config__diakrino_family_prescreen_enabled",
        "diakrino_skip_fit_discrete_enabled",
        "dist_config__diakrino_skip_fit_discrete_enabled",
        "diakrino_warm_start_enabled",
        "dist_config__diakrino_warm_start_enabled",
        "diakrino_prefilter_enabled",
        "diakrino_cdf_trust_gate_enabled",
        "diakrino_stability_surrogate_enabled",
        "diakrino_regime_conditional",
        "diakrino_regime_conditional_enabled",
        "diakrino_conformal_selection_enabled",
        "fs_use_diakrino_relevance_oracle",
        "diakrino_router_dispersion_descriptor_enabled",
        "include_tabpfn_model",
        "include_tabentics_diakrino_model",
        "benchmark_tabpfn_enabled",
        "native_categorical_stage2_enabled",
    }
    active_nonzero_keys = {
        "diakrino_family_prior_lambda",
        "dist_config__diakrino_family_prior_lambda",
    }
    active_path_keys = {
        "diakrino_sidecar_path",
        "dist_config__diakrino_sidecar_path",
        "tabentics_diakrino_checkpoint",
    }

    def nonempty_path(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def active_bool(value: Any) -> bool:
        return value is True or isinstance(value, np.bool_) and bool(value)

    def active_nonzero(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        return bool(np.isfinite(float(value)) and float(value) != 0.0)

    for key, value in config.items():
        key_text = str(key).strip().lower()
        if key_text in active_boolean_keys and active_bool(value):
            return str(key)
        if key_text in active_nonzero_keys and active_nonzero(value):
            return str(key)
        if key_text in active_path_keys and nonempty_path(value):
            return str(key)

        # Runtime snapshots add these only when an active route materialised.
        if key_text in {"diakrino_prefilter", "diakrino_protected_augmentation", "native_categorical_stage2_route"}:
            if isinstance(value, Mapping) and any(
                bool(value.get(flag, False)) for flag in ("enabled", "configured", "applied")
            ):
                return str(key)
    return None


def _bundle_without_integrity(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in bundle.items() if str(key) != "integrity"}


def _add_integrity(bundle: Mapping[str, Any]) -> dict[str, Any]:
    plain = _plain_json(bundle)
    if "integrity" in plain:
        raise SafeBundleSchemaError("Cannot add integrity to a bundle that already has integrity data.")
    plain["integrity"] = {
        "algorithm": "sha256",
        "scope": "bundle_without_integrity",
        "sha256": _sha256_json(plain),
    }
    return plain


def _verify_integrity(bundle: Mapping[str, Any]) -> dict[str, Any]:
    plain = _plain_json(bundle)
    integrity = _require_mapping(plain.get("integrity"), field="integrity")
    _require_exact_keys(
        integrity,
        field="integrity",
        required={"algorithm", "scope", "sha256"},
    )
    if integrity.get("algorithm") != "sha256" or integrity.get("scope") != "bundle_without_integrity":
        raise SafeBundleIntegrityError("Safe bundle integrity metadata is not supported.")
    received = _require_string(integrity.get("sha256"), field="integrity.sha256")
    if len(received) != 64 or any(character not in "0123456789abcdef" for character in received):
        raise SafeBundleIntegrityError("integrity.sha256 must be a lowercase SHA-256 digest.")
    expected = _sha256_json(_bundle_without_integrity(plain))
    if not hmac.compare_digest(expected, received):
        raise SafeBundleIntegrityError("Safe bundle SHA-256 integrity check failed.")
    return plain


@dataclass(frozen=True)
class SafeDFFSInferenceModel:
    """Array-only inference reconstructed from an allowlisted v2/v3 bundle."""

    source_schema: DatasetSchema
    model_input_schema: DatasetSchema
    selected_schema: DatasetSchema
    imputer_statistics_: np.ndarray
    scaler_mean_: np.ndarray
    scaler_scale_: np.ndarray
    selected_model_input_indices_: np.ndarray
    classes_: np.ndarray
    coef_: np.ndarray
    intercept_: np.ndarray
    probability_mode_: str
    bundle_sha256_: str

    @property
    def n_features_in_(self) -> int:
        return int(self.source_schema.n_features)

    def _checked_matrix(self, X: Any, *, alignment: str) -> np.ndarray:
        try:
            checked, _ = self.source_schema.align_inference_input(
                X,
                alignment_mode=alignment,
            )
        except SchemaContractError as exc:
            raise SafeBundleSchemaError(f"Safe bundle input schema mismatch: {exc}") from exc
        try:
            matrix = np.asarray(checked, dtype=float)
        except (TypeError, ValueError) as exc:
            raise SafeBundleSchemaError("Safe bundle inference input must be numeric.") from exc
        if matrix.ndim != 2 or int(matrix.shape[1]) != self.n_features_in_:
            raise SafeBundleSchemaError(
                "Safe bundle inference input has an incompatible numeric matrix shape."
            )
        if np.any(np.isinf(matrix)):
            raise SafeBundleSchemaError("Safe bundle inference input may contain NaN but not infinity.")
        return np.asarray(matrix, dtype=float)

    def transform(self, X: Any, *, alignment: str = "strict") -> np.ndarray:
        """Apply the bound median-impute, scale, and positional-selection route."""

        matrix = self._checked_matrix(X, alignment=alignment)
        imputed = np.where(np.isnan(matrix), self.imputer_statistics_[None, :], matrix)
        scaled = (imputed - self.scaler_mean_[None, :]) / self.scaler_scale_[None, :]
        return np.asarray(scaled[:, self.selected_model_input_indices_], dtype=float)

    def decision_function(self, X: Any, *, alignment: str = "strict") -> np.ndarray:
        transformed = self.transform(X, alignment=alignment)
        scores = transformed @ self.coef_.T + self.intercept_[None, :]
        if self.probability_mode_ == "binary":
            return np.asarray(scores[:, 0], dtype=float)
        return np.asarray(scores, dtype=float)

    @staticmethod
    def _expit(values: np.ndarray) -> np.ndarray:
        out = np.empty_like(values, dtype=float)
        positive = values >= 0.0
        out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        negative_exp = np.exp(values[~positive])
        out[~positive] = negative_exp / (1.0 + negative_exp)
        return out

    def predict_proba(self, X: Any, *, alignment: str = "strict") -> np.ndarray:
        scores = self.decision_function(X, alignment=alignment)
        if self.probability_mode_ == "binary":
            positive = self._expit(np.asarray(scores, dtype=float))
            return np.column_stack((1.0 - positive, positive))
        scores_2d = np.asarray(scores, dtype=float)
        if self.probability_mode_ == "ovr":
            probabilities = self._expit(scores_2d)
            normalizer = probabilities.sum(axis=1, keepdims=True)
            if np.any(normalizer <= 0.0):
                raise SafeBundleSchemaError(
                    "Safe LogisticRegression OvR probability normalisation underflowed."
                )
            return probabilities / normalizer
        shifted = scores_2d - np.max(scores_2d, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, X: Any, *, alignment: str = "strict") -> np.ndarray:
        scores = self.decision_function(X, alignment=alignment)
        if self.probability_mode_ == "binary":
            positions = np.asarray(scores > 0.0, dtype=int)
        else:
            positions = np.argmax(np.asarray(scores, dtype=float), axis=1)
        return np.asarray(self.classes_)[positions]

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(self.selected_schema.feature_names, dtype=object)


def create_safe_dffs_bundle(fitted: Any) -> dict[str, Any]:
    """Encode one fitted numeric DFFS route as a non-executable v3 JSON manifest.

    ``fitted`` may be a current ``FittedPipelineComponents`` instance or its
    current ``DFFSReproducibleModel.runtime_model``.  This is intentionally not
    a generic model serializer: unsupported state is rejected before a bundle
    is produced.
    """

    fitted_type = _type_id(fitted)
    if fitted_type == _COMPONENTS_TYPE:
        components = fitted
        runtime = getattr(components, "runtime_model", None)
        source_schema = getattr(components, "source_schema", None)
        model_input_schema = getattr(components, "model_input_schema", None)
        selected_schema_record = getattr(components, "selected_feature_schema", {})
        config_snapshot = _normalise_config_snapshot(
            getattr(components, "config_snapshot", {})
        )
        if getattr(components, "typed_preprocessor", None) is not None:
            raise UnsupportedSafeBundleStateError(
                "Safe v3 route does not support typed-preprocessor state; use a future typed codec."
            )
        declared_selected = tuple(
            getattr(components, "selected_model_input_indices", tuple()) or tuple()
        )
        source_kind = "fitted_pipeline_components"
    elif fitted_type == _RUNTIME_TYPE:
        components = None
        runtime = fitted
        source_schema = None
        model_input_schema = None
        selected_schema_record = {}
        config_snapshot = {}
        declared_selected = tuple()
        source_kind = "runtime_model_only_generated_positional_schema"
    else:
        raise UnsupportedSafeBundleStateError(
            "Safe v3 exporter accepts only the current FittedPipelineComponents "
            "or DFFSReproducibleModel runtime object."
        )
    if _type_id(runtime) != _RUNTIME_TYPE:
        raise UnsupportedSafeBundleStateError(
            "Fitted components do not carry the current DFFSReproducibleModel runtime state."
        )

    if "training_balance" not in config_snapshot and "training_balance_provenance" not in config_snapshot:
        from .balancing import TrainingBalanceConfig

        config_snapshot = dict(config_snapshot)
        config_snapshot["training_balance"] = TrainingBalanceConfig().to_dict()
        config_snapshot["training_balance_provenance"] = {}

    forbidden = _config_has_enabled_forbidden_state(config_snapshot)
    if forbidden is not None:
        raise UnsupportedSafeBundleStateError(
            f"Safe v3 route rejects enabled DIAKRINO/native state in config field {forbidden!r}."
        )
    _validate_training_balance_metadata(
        config_snapshot,
        require_metadata=True,
        legacy_v2=False,
        error_type=UnsupportedSafeBundleStateError,
    )

    try:
        n_features = int(getattr(runtime, "n_input_features"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise UnsupportedSafeBundleStateError("Runtime model has no valid input feature width.") from exc
    if n_features <= 0 or n_features > _MAX_FEATURES:
        raise UnsupportedSafeBundleStateError(
            f"Safe numeric route requires 1..{_MAX_FEATURES} input features."
        )

    if source_schema is None:
        source = _generated_numeric_schema(n_features)
    else:
        _validate_numeric_schema(source_schema, field="source_schema", width=n_features)
        source = source_schema
    if model_input_schema is None:
        model_schema = source
    else:
        _validate_numeric_schema(model_input_schema, field="model_input_schema", width=n_features)
        model_schema = model_input_schema
    if not _same_feature_contract(source, model_schema):
        raise UnsupportedSafeBundleStateError(
            "Safe route permits no feature-generating/converting preprocessor; source and "
            "model-input schemas must have the same feature contract."
        )

    _assert_runtime_disabled_state(runtime)
    runtime_metadata = getattr(runtime, "metadata", {})
    if not isinstance(runtime_metadata, Mapping):
        raise UnsupportedSafeBundleStateError("Safe route rejects malformed runtime metadata.")
    metadata_forbidden = _config_has_enabled_forbidden_state(runtime_metadata)
    if metadata_forbidden is not None:
        raise UnsupportedSafeBundleStateError(
            f"Safe v3 route rejects enabled DIAKRINO/native runtime metadata field {metadata_forbidden!r}."
        )
    statistics, scaler_mean, scaler_scale = _extract_preprocessing(runtime, n_features=n_features)
    prefilter, variance, selector_indices, selected_positions = _extract_positional_selection(
        runtime,
        n_features=n_features,
    )
    if declared_selected:
        try:
            declared = _indices(
                np.asarray(declared_selected),
                field="selected_model_input_indices",
                width=n_features,
            )
        except UnsupportedSafeBundleStateError:
            raise
        if not np.array_equal(declared, selected_positions):
            raise UnsupportedSafeBundleStateError(
                "Fitted component selected_model_input_indices disagree with its replayable selector route."
            )
    expected_selected_schema = model_schema.select(
        selected_positions.tolist(),
        operation="safe_bundle_positional_selection",
    )
    selected_schema = expected_selected_schema
    if selected_schema_record:
        if not isinstance(selected_schema_record, Mapping):
            raise UnsupportedSafeBundleStateError("selected_feature_schema must be a schema record when present.")
        if "features" in selected_schema_record:
            try:
                provided_selected_schema = DatasetSchema.from_record(selected_schema_record)
            except (SchemaContractError, TypeError, ValueError) as exc:
                raise UnsupportedSafeBundleStateError(
                    "selected_feature_schema is not a valid immutable schema record."
                ) from exc
            if not _same_feature_contract(expected_selected_schema, provided_selected_schema):
                raise UnsupportedSafeBundleStateError(
                    "selected_feature_schema disagrees with the positional selected-feature route."
                )
            selected_schema = provided_selected_schema

    classes, coef, intercept, probability_mode, classifier_descriptor = _extract_classifier(
        getattr(runtime, "classifier_model", None),
        n_selected=int(selected_positions.size),
    )
    route = {
        "id": SAFE_DFFS_ROUTE_ID,
        "input_kind": "numeric_only",
        "schema_alignment_default": "strict",
        "preprocessing": [
            "simple_imputer_median",
            "standard_scaler",
            "positional_selected_features",
        ],
        "unsupported_state_policy": "fail_closed",
        "n_input_features": int(n_features),
        "n_selected_features": int(selected_positions.size),
    }
    model_record = {
        "imputer": {
            "kind": "sklearn_simple_imputer_median",
            "statistics": statistics.tolist(),
        },
        "scaler": {
            "kind": "sklearn_standard_scaler",
            "mean": scaler_mean.tolist(),
            "scale": scaler_scale.tolist(),
        },
        "selection": {
            "kind": "positional_indices",
            "prefilter_indices": prefilter.tolist(),
            "variance_keep_indices": variance.tolist(),
            "selector_indices": selector_indices.tolist(),
            "selected_model_input_indices": selected_positions.tolist(),
        },
        "classifier": {
            **classifier_descriptor,
            "class_order": classes,
            "coef": coef.tolist(),
            "intercept": intercept.tolist(),
        },
    }
    package = _safe_package_record()
    schemas = {
        "source": _schema_record(source),
        "model_input": _schema_record(model_schema),
        "selected": _schema_record(selected_schema),
    }
    lineage = {
        name: list(record.get("lineage") or [])
        for name, record in schemas.items()
    }
    bundle = {
        "artifact_type": SAFE_DFFS_BUNDLE_ARTIFACT,
        "schema_version": SAFE_DFFS_BUNDLE_SCHEMA_VERSION,
        "trust_mode": SAFE_DFFS_BUNDLE_TRUST_MODE,
        "source_kind": source_kind,
        "route": route,
        "schemas": schemas,
        "lineage": lineage,
        "class_order": classes,
        "config": config_snapshot,
        "model": model_record,
        "package": package,
        "hashes": {
            "config_sha256": _sha256_json(config_snapshot),
            "model_sha256": _sha256_json(model_record),
            "package_sha256": _sha256_json(package),
            "source_schema_sha256": _sha256_json(schemas["source"]),
            "model_input_schema_sha256": _sha256_json(schemas["model_input"]),
            "selected_schema_sha256": _sha256_json(schemas["selected"]),
        },
    }
    sealed = _add_integrity(bundle)
    # Export and import share one strict validator, so an encoder regression cannot
    # emit a manifest that the non-executable loader interprets differently.
    load_safe_dffs_bundle(sealed)
    return sealed


def _parse_payload(payload: Mapping[str, Any] | str | bytes | bytearray) -> dict[str, Any]:
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            raw = json.loads(payload)
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise SafeBundleSchemaError("Safe bundle input is not valid JSON.") from exc
    else:
        raw = payload
    return _require_mapping(raw, field="bundle")


def load_safe_dffs_bundle(payload: Mapping[str, Any] | str | bytes | bytearray) -> SafeDFFSInferenceModel:
    """Validate and load a v2/v3 JSON-only bundle without executable deserialization."""

    bundle = _verify_integrity(_parse_payload(payload))
    _require_exact_keys(
        bundle,
        field="bundle",
        required={
            "artifact_type",
            "schema_version",
            "trust_mode",
            "source_kind",
            "route",
            "schemas",
            "lineage",
            "class_order",
            "config",
            "model",
            "package",
            "hashes",
            "integrity",
        },
    )
    if bundle.get("artifact_type") != SAFE_DFFS_BUNDLE_ARTIFACT:
        raise SafeBundleSchemaError("Bundle artifact_type is not a safe DFFS artifact.")
    schema_version = bundle.get("schema_version")
    if schema_version not in {
        SAFE_DFFS_BUNDLE_SCHEMA_VERSION,
        _LEGACY_SAFE_DFFS_BUNDLE_SCHEMA_VERSION,
    }:
        raise SafeBundleSchemaError("Bundle schema_version is not supported by the safe loader.")
    legacy_v2 = schema_version == _LEGACY_SAFE_DFFS_BUNDLE_SCHEMA_VERSION
    expected_route_id = _LEGACY_SAFE_DFFS_ROUTE_ID if legacy_v2 else SAFE_DFFS_ROUTE_ID
    if bundle.get("trust_mode") != SAFE_DFFS_BUNDLE_TRUST_MODE:
        raise SafeBundleSchemaError("Bundle trust_mode is not the non-executable allowlist mode.")
    source_kind = _require_string(bundle.get("source_kind"), field="source_kind")
    if source_kind not in {"fitted_pipeline_components", "runtime_model_only_generated_positional_schema"}:
        raise SafeBundleSchemaError("Bundle source_kind is not supported.")

    route = _require_mapping(bundle.get("route"), field="route")
    _require_exact_keys(
        route,
        field="route",
        required={
            "id",
            "input_kind",
            "schema_alignment_default",
            "preprocessing",
            "unsupported_state_policy",
            "n_input_features",
            "n_selected_features",
        },
    )
    if route.get("id") != expected_route_id or route.get("input_kind") != "numeric_only":
        raise SafeBundleSchemaError("Bundle route is not the supported numeric safe route.")
    if route.get("schema_alignment_default") != "strict" or route.get("unsupported_state_policy") != "fail_closed":
        raise SafeBundleSchemaError("Bundle route has an unsupported inference policy.")
    if route.get("preprocessing") != [
        "simple_imputer_median",
        "standard_scaler",
        "positional_selected_features",
    ]:
        raise SafeBundleSchemaError("Bundle preprocessing route is not allowlisted.")
    n_features = _require_int(route.get("n_input_features"), field="route.n_input_features", minimum=1)
    n_selected = _require_int(route.get("n_selected_features"), field="route.n_selected_features", minimum=1)
    if n_features > _MAX_FEATURES or n_selected > n_features:
        raise SafeBundleSchemaError("Bundle route dimensions are outside safe bounds.")

    config = _require_mapping(bundle.get("config"), field="config")
    forbidden = _config_has_enabled_forbidden_state(config)
    if forbidden is not None:
        raise SafeBundleSchemaError(
            f"Bundle contains enabled DIAKRINO/native state in config field {forbidden!r}."
        )
    _validate_training_balance_metadata(
        config,
        require_metadata=not legacy_v2,
        legacy_v2=legacy_v2,
        error_type=SafeBundleSchemaError,
    )
    package = _require_mapping(bundle.get("package"), field="package")
    _require_exact_keys(package, field="package", required={"name", "version", "codec"})
    if package.get("name") != "tabnetics" or package.get("codec") != expected_route_id:
        raise SafeBundleSchemaError("Bundle package identity is not supported.")
    _require_string(package.get("version"), field="package.version")

    schemas = _require_mapping(bundle.get("schemas"), field="schemas")
    _require_exact_keys(schemas, field="schemas", required={"source", "model_input", "selected"})
    source_schema = _schema_from_record(schemas["source"], field="schemas.source", width=n_features)
    model_input_schema = _schema_from_record(
        schemas["model_input"], field="schemas.model_input", width=n_features
    )
    selected_schema = _schema_from_record(
        schemas["selected"], field="schemas.selected", width=n_selected
    )
    if not _same_feature_contract(source_schema, model_input_schema):
        raise SafeBundleSchemaError(
            "Safe route source and model-input schemas must have identical feature contracts."
        )

    lineage = _require_mapping(bundle.get("lineage"), field="lineage")
    _require_exact_keys(lineage, field="lineage", required={"source", "model_input", "selected"})
    for name in ("source", "model_input", "selected"):
        expected = list(schemas[name].get("lineage") or [])
        if lineage[name] != expected:
            raise SafeBundleSchemaError(
                f"Bundle lineage.{name} does not match the schema-bound lineage record."
            )

    model = _require_mapping(bundle.get("model"), field="model")
    _require_exact_keys(model, field="model", required={"imputer", "scaler", "selection", "classifier"})
    imputer = _require_mapping(model.get("imputer"), field="model.imputer")
    _require_exact_keys(imputer, field="model.imputer", required={"kind", "statistics"})
    if imputer.get("kind") != "sklearn_simple_imputer_median":
        raise SafeBundleSchemaError("Bundle imputer is not the supported median-imputer state.")
    statistics = _finite_vector(imputer.get("statistics"), field="model.imputer.statistics", length=n_features)

    scaler = _require_mapping(model.get("scaler"), field="model.scaler")
    _require_exact_keys(scaler, field="model.scaler", required={"kind", "mean", "scale"})
    if scaler.get("kind") != "sklearn_standard_scaler":
        raise SafeBundleSchemaError("Bundle scaler is not the supported StandardScaler state.")
    scaler_mean = _finite_vector(scaler.get("mean"), field="model.scaler.mean", length=n_features)
    scaler_scale = _finite_vector(scaler.get("scale"), field="model.scaler.scale", length=n_features)
    if np.any(scaler_scale <= 0.0):
        raise SafeBundleSchemaError("Bundle scaler.scale must be strictly positive.")

    selection = _require_mapping(model.get("selection"), field="model.selection")
    _require_exact_keys(
        selection,
        field="model.selection",
        required={
            "kind",
            "prefilter_indices",
            "variance_keep_indices",
            "selector_indices",
            "selected_model_input_indices",
        },
    )
    if selection.get("kind") != "positional_indices":
        raise SafeBundleSchemaError("Bundle selection is not the supported positional route.")
    prefilter = _indices(
        np.asarray(selection.get("prefilter_indices"), dtype=int),
        field="model.selection.prefilter_indices",
        width=n_features,
    )
    variance = _indices(
        np.asarray(selection.get("variance_keep_indices"), dtype=int),
        field="model.selection.variance_keep_indices",
        width=int(prefilter.size),
    )
    selector_indices = _indices(
        np.asarray(selection.get("selector_indices"), dtype=int),
        field="model.selection.selector_indices",
        width=int(variance.size),
    )
    selected_positions = _indices(
        np.asarray(selection.get("selected_model_input_indices"), dtype=int),
        field="model.selection.selected_model_input_indices",
        width=n_features,
    )
    composed = prefilter[variance[selector_indices]]
    if not np.array_equal(composed, selected_positions) or int(selected_positions.size) != n_selected:
        raise SafeBundleSchemaError("Bundle positional selection route does not compose to its selected positions.")
    expected_selected = model_input_schema.select(
        selected_positions.tolist(),
        operation="safe_bundle_positional_selection",
    )
    if not _same_feature_contract(expected_selected, selected_schema):
        raise SafeBundleSchemaError(
            "Bundle selected schema does not match the positional selected-feature route."
        )

    classes_encoded = bundle.get("class_order")
    classes = _decode_labels(classes_encoded)
    classifier = _require_mapping(model.get("classifier"), field="model.classifier")
    _require_exact_keys(
        classifier,
        field="model.classifier",
        required={"kind", "solver", "probability_mode", "n_features_in", "class_order", "coef", "intercept"},
    )
    if classifier.get("kind") != "sklearn_logistic_regression_coefficients":
        raise SafeBundleSchemaError("Bundle classifier is not allowlisted LogisticRegression coefficient state.")
    _require_string(classifier.get("solver"), field="model.classifier.solver")
    probability_mode = classifier.get("probability_mode")
    if probability_mode not in {"binary", "ovr", "multinomial"}:
        raise SafeBundleSchemaError("Bundle LogisticRegression probability mode is not supported.")
    if probability_mode == "binary" and len(classes) != 2:
        raise SafeBundleSchemaError("Binary LogisticRegression state requires exactly two classes.")
    if probability_mode != "binary" and len(classes) <= 2:
        raise SafeBundleSchemaError("Multiclass LogisticRegression state requires more than two classes.")
    if _require_int(classifier.get("n_features_in"), field="model.classifier.n_features_in", minimum=1) != n_selected:
        raise SafeBundleSchemaError("Bundle LogisticRegression n_features_in disagrees with selected route width.")
    if classifier.get("class_order") != classes_encoded:
        raise SafeBundleSchemaError("Bundle classifier class order does not match top-level class_order.")
    coefficient_rows = 1 if probability_mode == "binary" else len(classes)
    coef = _finite_matrix(
        classifier.get("coef"),
        field="model.classifier.coef",
        rows=coefficient_rows,
        columns=n_selected,
    )
    intercept = _finite_vector(
        classifier.get("intercept"),
        field="model.classifier.intercept",
        length=coefficient_rows,
    )

    hashes = _require_mapping(bundle.get("hashes"), field="hashes")
    _require_exact_keys(
        hashes,
        field="hashes",
        required={
            "config_sha256",
            "model_sha256",
            "package_sha256",
            "source_schema_sha256",
            "model_input_schema_sha256",
            "selected_schema_sha256",
        },
    )
    bound_values = {
        "config_sha256": config,
        "model_sha256": model,
        "package_sha256": package,
        "source_schema_sha256": schemas["source"],
        "model_input_schema_sha256": schemas["model_input"],
        "selected_schema_sha256": schemas["selected"],
    }
    for key, value in bound_values.items():
        received = _require_string(hashes.get(key), field=f"hashes.{key}")
        expected = _sha256_json(value)
        if not hmac.compare_digest(received, expected):
            raise SafeBundleIntegrityError(f"Bundle bound hash {key} did not verify.")

    for array in (statistics, scaler_mean, scaler_scale, selected_positions, coef, intercept):
        array.setflags(write=False)
    classes_array = np.asarray(classes)
    classes_array.setflags(write=False)
    return SafeDFFSInferenceModel(
        source_schema=source_schema,
        model_input_schema=model_input_schema,
        selected_schema=selected_schema,
        imputer_statistics_=statistics,
        scaler_mean_=scaler_mean,
        scaler_scale_=scaler_scale,
        selected_model_input_indices_=selected_positions,
        classes_=classes_array,
        coef_=coef,
        intercept_=intercept,
        probability_mode_=str(probability_mode),
        bundle_sha256_=_require_string(bundle["integrity"].get("sha256"), field="integrity.sha256"),
    )


__all__ = [
    "DFFSSafeBundleError",
    "SAFE_DFFS_BUNDLE_ARTIFACT",
    "SAFE_DFFS_BUNDLE_SCHEMA_VERSION",
    "SAFE_DFFS_BUNDLE_TRUST_MODE",
    "SAFE_DFFS_ROUTE_ID",
    "SafeBundleIntegrityError",
    "SafeBundleSchemaError",
    "SafeDFFSInferenceModel",
    "UnsupportedSafeBundleStateError",
    "create_safe_dffs_bundle",
    "load_safe_dffs_bundle",
]

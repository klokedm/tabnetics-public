"""Small provenance payloads for validation and benchmark artifacts."""

from __future__ import annotations

import ast
import builtins
from collections import Counter
import contextlib
import datetime as dt
import dataclasses
from enum import Enum
import functools
import gc
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import logging
import math
import os
import platform
import re
import resource
import signal
import socket
import subprocess
import sys
import symtable
import tempfile
from importlib import metadata as importlib_metadata
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path, PosixPath, PurePosixPath, WindowsPath
from types import (
    BuiltinFunctionType,
    CodeType,
    FunctionType,
    MappingProxyType,
    ModuleType,
)
from typing import Any, Mapping, Sequence

from tabnetics.core.paths import find_repo_root_or_none


def _trusted_builtin_value(
    name: str,
    default: Any,
    _stdlib_builtins: ModuleType = builtins,
) -> Any:
    """Read the import-time stdlib builtin table without a rebound global.

    The verifier must not resolve a source name through the mutable
    ``provenance.builtins`` import binding.  The exact module default is sealed
    with the other callable defaults against the clean child.
    """

    if type(_stdlib_builtins) is not ModuleType:
        raise CanonicalExecutionOriginError(
            "Trusted builtin module has an invalid type."
        )
    module_dict = object.__getattribute__(_stdlib_builtins, "__dict__")
    if type(module_dict) is not dict:
        raise CanonicalExecutionOriginError(
            "Trusted builtin module has an invalid dictionary."
        )
    return dict.get(module_dict, name, default)


def _assert_trusted_builtins_binding(
    _stdlib_builtins: ModuleType = builtins,
    _self_module: ModuleType = sys.modules[__name__],
) -> None:
    """Reject a rebound ``builtins`` import before any source resolution."""

    module_values = object.__getattribute__(_self_module, "__dict__")
    live = dict.get(module_values, "builtins") if type(module_values) is dict else None
    if live is not _stdlib_builtins:
        raise CanonicalExecutionOriginError(
            "Provenance builtin module binding no longer matches its import-time identity."
        )


def _assert_trusted_logging_binding(
    _stdlib_logging: ModuleType = logging,
    _self_module: ModuleType = sys.modules[__name__],
) -> None:
    """Reject a rebound ``logging`` import before logger-state inspection."""

    module_values = object.__getattribute__(_self_module, "__dict__")
    live = dict.get(module_values, "logging") if type(module_values) is dict else None
    if live is not _stdlib_logging:
        raise CanonicalExecutionOriginError(
            "Provenance logging module binding no longer matches its import-time identity."
        )


def _assert_trusted_verifier_import_bindings(
    _self_module: ModuleType = sys.modules[__name__],
    _bindings: tuple[tuple[str, ModuleType], ...] = (
        ("ast", ast),
        ("contextlib", contextlib),
        ("dataclasses", dataclasses),
        ("functools", functools),
        ("gc", gc),
        ("hashlib", hashlib),
        ("importlib", importlib),
        ("inspect", inspect),
        ("io", io),
        ("json", json),
        ("math", math),
        ("os", os),
        ("platform", platform),
        ("re", re),
        ("resource", resource),
        ("signal", signal),
        ("subprocess", subprocess),
        ("symtable", symtable),
        ("sys", sys),
        ("tempfile", tempfile),
    ),
) -> None:
    """Reject a rebound stdlib helper before verifier code can dispatch through it.

    The closure verifier imports a broad standard-library surface.  Checking
    only ``builtins`` and ``logging`` leaves a proxy placed in (for example)
    ``provenance.dataclasses`` or ``provenance.inspect`` able to run while the
    verifier decides whether to reject it.  Retain import-time object references
    in defaults and compare them through the captured module dictionary.
    """

    module_values = object.__getattribute__(_self_module, "__dict__")
    if type(module_values) is not dict:
        raise CanonicalExecutionOriginError(
            "Provenance verifier module has an invalid dictionary."
        )
    for name, expected in _bindings:
        if dict.get(module_values, name, _MISSING_RUNTIME_BINDING) is not expected:
            raise CanonicalExecutionOriginError(
                f"Provenance verifier import binding {name!r} no longer matches its import-time identity."
            )


PROVENANCE_SCHEMA_VERSION = "tabnetics_provenance_v1"
EXECUTION_PROVENANCE_SCHEMA_VERSION = "tabnetics_execution_provenance_v4"
CANONICAL_IMPLEMENTATION_STACK = "tabnetics_core"
CANONICAL_EVIDENCE_STATUS = "canonical"
LEGACY_NONCANONICAL_EVIDENCE_STATUS = "legacy_noncanonical"
EXTERNAL_CALLABLE_UNATTESTED_EVIDENCE_STATUS = "noncanonical_external_callable_identity"
EXTERNAL_CALLABLE_UNATTESTED_REASON_PREFIX = "external_callable_identity_unattested:"
# These direct objects are the bootstrap boundary of the benchmark runner.
# They are intentionally *not* treated as a complete execution closure.  The
# complete closure is captured at finalization from all loaded ``tabnetics.*``
# modules; see ``_loaded_package_module_closure`` below.
CANONICAL_BOOTSTRAP_IMPORT_SPECS: dict[str, tuple[str, str | None]] = {
    "benchmark_runner": ("tabnetics.benchmarks.runner", None),
    "classification_config": ("tabnetics.pipeline.pipeline", "ClassificationConfig"),
    "dffs_config": ("tabnetics.pipeline.pipeline", "DFFSConfig"),
    "distribution_fitter_config": (
        "tabnetics.pipeline.pipeline",
        "DistributionFitterConfig",
    ),
    "pipeline": ("tabnetics.pipeline.pipeline", "DistributionFeatureSelectionPipeline"),
    # ``FeatureSelector`` is re-exported from the package but authored in
    # ``base.py``.
    "feature_selector": ("tabnetics.feature_selection.base", "FeatureSelector"),
}
CANONICAL_BOOTSTRAP_IMPORT_LABELS: tuple[str, ...] = tuple(
    CANONICAL_BOOTSTRAP_IMPORT_SPECS
)
CANONICAL_BENCHMARK_ARTIFACT_PROVENANCE_SCHEMA_VERSION = (
    "tabnetics_benchmark_artifact_provenance_v4"
)
CANONICAL_BENCHMARK_ARTIFACT_PROVENANCE_FILENAME = "df_fs_artifact_provenance.json"
CANONICAL_BENCHMARK_ARTIFACT_NAMES: tuple[str, ...] = (
    "df_fs_runs.csv",
    "df_fs_summary.csv",
    "df_fs_sota_comparison.csv",
    "df_fs_ablation_deltas.csv",
    "df_fs_metadata.json",
    "df_fs_execution_provenance.json",
)
MATERIALIZED_INPUT_IDENTITY_SCHEMA_VERSION = "tabnetics_materialized_input_v1"
INPUT_DATA_IDENTITY_SCHEMA_VERSION = "tabnetics_input_data_identity_v2"
LOADED_PACKAGE_MODULE_CLOSURE_SCHEMA_VERSION = "tabnetics_loaded_package_modules_v4"
LOADED_PACKAGE_MODULE_PREFIX = "tabnetics"
# ``created_at`` describes when a contract was written, not what was executed.
# It is intentionally excluded from the execution fingerprint so a manifest can
# be copied or reserialized without changing its execution identity.
EXECUTION_FINGERPRINT_EXCLUDED_FIELDS = frozenset({"created_at", "fingerprint_sha256"})
DEFAULT_ENV_KEYS: tuple[str, ...] = (
    "MAX_WORKERS",
    "PODS_PER_HOST",
    "TABNETICS_HOST_LABEL",
    "TABNETICS_RUN_HOST",
    "TABNETICS_HOST_CALIBRATION_PATH",
    "SKIP_DONE",
    "SKIP_DONE_STATUS_DIRS",
    "TASK_TIMEOUT_SEC_OVERRIDE",
    "FS_METHOD_TIMEOUT_SEC_OVERRIDE",
    "FS_METHOD_MAX_RSS_MB_OVERRIDE",
    "DISABLE_DF_FASTPATH",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "TABNETICS_HF_ORG",
    "TABNETICS_HF_REPO_NAME",
    "TABNETICS_HF_REPO_ID",
)
MAX_PROVENANCE_HASH_BYTES = 256 * 1024 * 1024


class CanonicalExecutionOriginError(RuntimeError):
    """Raised when a claimed core validation path resolves outside tabnetics."""


class CanonicalExecutionInputIdentityError(RuntimeError):
    """Raised when a benchmark input cannot be bound to the dataset registry."""


class CanonicalExecutionExternalDependencyError(CanonicalExecutionOriginError):
    """Raised when a run uses an external callable without identity attestation."""


def utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _stable_json_value(value: Any) -> Any:
    """Normalize values before hashing an execution contract.

    The regular artifact serializer permits a few convenient Python values.  A
    contract digest needs stronger determinism, especially for sets and non-
    finite floats that may appear in a programmatically built ``Namespace``.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _stable_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_stable_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return _stable_json_value(value.tolist())
        if isinstance(value, np.generic):
            return _stable_json_value(value.item())
    except Exception:
        pass
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def canonical_json_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest for an execution-contract payload."""

    encoded = json.dumps(
        _stable_json_value(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execution_fingerprint_payload(
    execution_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable portion of an execution contract.

    ``created_at`` and the fingerprint itself are intentionally excluded. Every
    other top-level field is part of the hash, including future fields, so a
    consumer fails closed when a newer producer changes the contract shape.
    """

    return {
        str(key): value
        for key, value in execution_provenance.items()
        if str(key) not in EXECUTION_FINGERPRINT_EXCLUDED_FIELDS
    }


def execution_fingerprint_sha256(execution_provenance: Mapping[str, Any]) -> str:
    """Return the deterministic execution identity excluding timestamp metadata."""

    return canonical_json_sha256(execution_fingerprint_payload(execution_provenance))


def _namespace_mapping(args: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(args, Mapping):
        return args
    try:
        return vars(args)
    except TypeError as exc:
        raise TypeError(
            "execution provenance args must be a mapping or namespace"
        ) from exc


def _nested_config_mapping(value: Any) -> Mapping[str, Any]:
    """Return a trusted config object's direct fields without invoking properties."""

    if isinstance(value, Mapping):
        return value
    try:
        return vars(value)
    except TypeError:
        return {}


def external_callable_identity_unattested_reason(
    args: Mapping[str, Any] | Any,
) -> str:
    """Return the noncanonical reason for configured unanchored callables.

    A clean child can compare external source and structure but cannot establish
    that a parent-process class object is the original imported object.  MAPIE
    APS/RAPS/cross execution is therefore useful operationally but excluded
    from canonical scorecards until it is backed by signed or out-of-process
    external attestation.  The ordinary split implementation remains internal
    and canonical.
    """

    values = _namespace_mapping(args)
    fs_enabled = bool(values.get("fs_use_conformal_efficiency", False))
    fs_method = (
        str(values.get("fs_conformal_efficiency_method", "split") or "split")
        .strip()
        .lower()
    )
    if fs_enabled and fs_method == "aps":
        return EXTERNAL_CALLABLE_UNATTESTED_REASON_PREFIX + "mapie"

    classification_values = _nested_config_mapping(values.get("classification"))
    classifier_enabled = bool(
        values.get(
            "enable_classifier_conformal",
            values.get(
                "classifier_conformal_enabled",
                classification_values.get("conformal_enabled", False),
            ),
        )
    )
    classifier_method = (
        str(
            values.get(
                "classifier_conformal_method",
                classification_values.get("conformal_method", "split"),
            )
            or "split"
        )
        .strip()
        .lower()
    )
    if classifier_enabled and classifier_method in {"aps", "raps", "cross"}:
        return EXTERNAL_CALLABLE_UNATTESTED_REASON_PREFIX + "mapie"
    return ""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _verified_tabnetics_package_identity(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, str]:
    """Bind execution imports to either this checkout or an installed wheel.

    A module name is not a trustworthy origin by itself: ``sys.modules`` can
    hold an in-memory module with a tabnetics-looking name and an arbitrary
    ``__file__``. We first verify the loaded top-level package root against the
    active checkout's ``core/src/tabnetics`` root or a file owned by the
    installed ``tabnetics`` distribution.
    """

    # This function imports the package before the closure builder gets a
    # chance to run its own guard.  Reject rebound verifier helpers first so a
    # proxy cannot execute through ``importlib`` during package identity setup.
    _assert_trusted_builtins_binding()
    _assert_trusted_logging_binding()
    _assert_trusted_verifier_import_bindings()

    try:
        package = importlib.import_module("tabnetics")
    except Exception as exc:
        raise CanonicalExecutionOriginError(
            "Canonical execution cannot import the tabnetics package."
        ) from exc
    package_file_raw = str(getattr(package, "__file__", "") or "").strip()
    if not package_file_raw:
        raise CanonicalExecutionOriginError(
            "Canonical execution tabnetics package has no source origin."
        )
    package_file = Path(package_file_raw).resolve()
    package_root = package_file.parent
    if package_file.name != "__init__.py" or not package_file.is_file():
        raise CanonicalExecutionOriginError(
            f"Canonical tabnetics package origin is invalid: {package_file}."
        )

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else find_repo_root_or_none(__file__)
    )
    if root is not None:
        editable_root = (root / "core" / "src" / "tabnetics").resolve()
        if (editable_root / "__init__.py").is_file() and package_root == editable_root:
            return {
                "kind": "editable_checkout",
                "package_root": str(package_root),
                "distribution_name": "",
                "distribution_version": "",
            }

    try:
        distribution = importlib_metadata.distribution("tabnetics")
    except importlib_metadata.PackageNotFoundError as exc:
        raise CanonicalExecutionOriginError(
            "Canonical tabnetics package is neither this checkout nor an installed distribution."
        ) from exc
    distribution_roots: set[Path] = set()
    for relative_path in distribution.files or ():
        if str(relative_path).replace("\\", "/") != "tabnetics/__init__.py":
            continue
        candidate = Path(distribution.locate_file(relative_path)).resolve()
        if candidate.is_file():
            distribution_roots.add(candidate.parent)
    if package_root not in distribution_roots:
        raise CanonicalExecutionOriginError(
            "Canonical tabnetics package root is not owned by the installed tabnetics distribution: "
            f"{package_root}."
        )
    return {
        "kind": "installed_distribution",
        "package_root": str(package_root),
        "distribution_name": str(
            distribution.metadata.get("Name", "tabnetics") or "tabnetics"
        ),
        "distribution_version": str(distribution.version or ""),
    }


def _expected_module_paths(module_name: str, package_root: Path) -> set[Path]:
    parts = tuple(str(part) for part in module_name.split(".")[1:] if str(part))
    if not parts:
        return {(package_root / "__init__.py").resolve()}
    module_base = package_root.joinpath(*parts)
    return {
        module_base.with_suffix(".py").resolve(),
        (module_base / "__init__.py").resolve(),
    }


def _bootstrap_import_labels_reason(import_targets: Mapping[str, Any]) -> str:
    labels = {str(label) for label in import_targets}
    required = set(CANONICAL_BOOTSTRAP_IMPORT_LABELS)
    missing = sorted(required - labels)
    if missing:
        return "bootstrap_import_labels_missing:" + ",".join(missing)
    unexpected = sorted(labels - required)
    if unexpected:
        return "bootstrap_import_labels_unexpected:" + ",".join(unexpected)
    return ""


def _target_code_origin(
    label: str,
    target: Any,
    *,
    source: Path,
) -> dict[str, Any]:
    """Return a compact code-origin record for a captured import target.

    Source-path validation alone is insufficient for classes: a synthetic class
    can claim a canonical ``__module__`` while the module file on disk remains
    genuine. The runner additionally supplies a sealed import-time identity
    map, and this inspection catches the common synthetic-object case before a
    contract is emitted. This is integrity evidence, not a substitute for a
    signed external attestation against a process that can rewrite every Python
    object and every output artifact.
    """

    if isinstance(target, ModuleType):
        target_kind = "module"
        target_qualname = str(getattr(target, "__name__", "") or "")
    elif inspect.isclass(target):
        target_kind = "class"
        target_qualname = str(getattr(target, "__qualname__", "") or "")
    else:
        raise CanonicalExecutionOriginError(
            f"Canonical execution target {label!r} is not a module or class."
        )

    try:
        inspected_path_raw = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError) as exc:
        raise CanonicalExecutionOriginError(
            f"Canonical execution target {label!r} has no inspectable source origin."
        ) from exc
    if not inspected_path_raw:
        raise CanonicalExecutionOriginError(
            f"Canonical execution target {label!r} has no inspectable source origin."
        )
    inspected_path = Path(str(inspected_path_raw)).resolve()
    if inspected_path != source:
        raise CanonicalExecutionOriginError(
            f"Canonical execution target {label!r} inspect origin does not match its module source: "
            f"got {inspected_path}; expected {source}."
        )

    code_records: list[dict[str, Any]] = []
    if isinstance(target, ModuleType):
        spec = getattr(target, "__spec__", None)
        spec_origin_raw = str(getattr(spec, "origin", "") or "").strip()
        if not spec_origin_raw:
            raise CanonicalExecutionOriginError(
                f"Canonical execution module {label!r} has no import-spec origin."
            )
        spec_origin = Path(spec_origin_raw).resolve()
        if spec_origin != source:
            raise CanonicalExecutionOriginError(
                f"Canonical execution module {label!r} import-spec origin does not match its source: "
                f"got {spec_origin}; expected {source}."
            )
        if getattr(spec, "loader", None) is None:
            raise CanonicalExecutionOriginError(
                f"Canonical execution module {label!r} has no import-spec loader."
            )
    else:
        for member_name, descriptor in vars(target).items():
            member = descriptor
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            if not inspect.isfunction(member):
                continue
            # Decorators such as ``contextlib.contextmanager`` expose a wrapper
            # implemented outside the class's module.  The wrapped callable is
            # the authored implementation whose origin must be attested.
            member = inspect.unwrap(member)
            code = getattr(member, "__code__", None)
            if code is None:
                continue
            code_path_raw = str(getattr(code, "co_filename", "") or "")
            generated_dataclass_method = bool(
                dataclasses.is_dataclass(target)
                and member_name
                in {
                    "__init__",
                    "__repr__",
                    "__eq__",
                    "__hash__",
                    "__lt__",
                    "__le__",
                    "__gt__",
                    "__ge__",
                    "__setattr__",
                    "__delattr__",
                    "__replace__",
                }
            )
            if (
                code_path_raw
                and not code_path_raw.startswith("<")
                and not generated_dataclass_method
            ):
                code_path = Path(code_path_raw).resolve()
                if not code_path.exists() or code_path != source:
                    raise CanonicalExecutionOriginError(
                        f"Canonical execution target {label!r} method {member_name!r} "
                        "does not originate from the verified module source."
                    )
            code_hash = hashlib.sha256()
            code_hash.update(str(member_name).encode("utf-8"))
            code_hash.update(bytes(code.co_code))
            code_hash.update(repr(code.co_consts).encode("utf-8"))
            code_records.append(
                {
                    "name": str(member_name),
                    "filename": code_path_raw,
                    "firstlineno": int(code.co_firstlineno),
                    "sha256": code_hash.hexdigest(),
                }
            )

    return {
        "kind": target_kind,
        "qualname": target_qualname,
        "inspect_source": str(inspected_path),
        "code_sha256": canonical_json_sha256(code_records),
    }


def _module_origin_record(
    label: str,
    target: Any,
    *,
    package_identity: Mapping[str, str],
    expected_target: Any | None = None,
) -> dict[str, Any]:
    """Resolve and validate one imported implementation target.

    The benchmark runner keeps import-time references to its bootstrap classes.
    Looking at those references rather than mutable module globals catches a
    future accidental ``experiments.df_fs_pipeline`` import while preserving
    ordinary test monkeypatching of runtime collaborators.
    """

    expected_spec = CANONICAL_BOOTSTRAP_IMPORT_SPECS.get(str(label))
    if expected_spec is None:
        raise CanonicalExecutionOriginError(
            f"Canonical execution target {label!r} is not a recognized bootstrap import."
        )
    expected_module_name, expected_attribute_name = expected_spec

    if isinstance(target, ModuleType):
        module = target
        target_module_name = str(getattr(target, "__name__", "") or "").strip()
    else:
        target_module_name = str(getattr(target, "__module__", "") or "").strip()
        if not target_module_name:
            raise CanonicalExecutionOriginError(
                f"Canonical execution target {label!r} has no import module."
            )
        try:
            module = importlib.import_module(target_module_name)
        except Exception as exc:
            raise CanonicalExecutionOriginError(
                f"Could not resolve canonical execution target {label!r} from {target_module_name!r}."
            ) from exc

    module_name = str(getattr(module, "__name__", "") or "").strip()
    if (
        target_module_name != expected_module_name
        or module_name != expected_module_name
    ):
        raise CanonicalExecutionOriginError(
            f"Canonical execution target {label!r} resolved to unexpected module "
            f"{module_name!r}; expected {expected_module_name!r}."
        )
    if expected_attribute_name is None:
        if not isinstance(target, ModuleType) or module is not target:
            raise CanonicalExecutionOriginError(
                f"Canonical execution target {label!r} is not the expected module object."
            )
    else:
        if not inspect.isclass(target):
            raise CanonicalExecutionOriginError(
                f"Canonical execution target {label!r} is not the expected class object."
            )
        if str(getattr(target, "__qualname__", "") or "") != expected_attribute_name:
            raise CanonicalExecutionOriginError(
                f"Canonical execution target {label!r} has unexpected qualified name."
            )
        module_attribute = getattr(module, expected_attribute_name, None)
        if module_attribute is not target:
            raise CanonicalExecutionOriginError(
                f"Canonical execution target {label!r} does not match the canonical module attribute."
            )
    if expected_target is not None and target is not expected_target:
        raise CanonicalExecutionOriginError(
            f"Canonical execution target {label!r} does not match the captured import-time identity."
        )

    source_path = str(getattr(module, "__file__", "") or "").strip()
    if not source_path:
        raise CanonicalExecutionOriginError(
            f"Canonical execution target {label!r} has no source origin."
        )
    source = Path(source_path).resolve()
    if not source.exists() or not source.is_file():
        raise CanonicalExecutionOriginError(
            f"Canonical execution target {label!r} source is unavailable: {source}."
        )
    package_root = Path(str(package_identity.get("package_root", "") or "")).resolve()
    if not _is_within(source, package_root):
        raise CanonicalExecutionOriginError(
            f"Canonical execution target {label!r} source is outside the verified tabnetics "
            f"package root: {source}."
        )
    expected_paths = _expected_module_paths(module_name, package_root)
    if source not in expected_paths:
        expected_text = ", ".join(str(path) for path in sorted(expected_paths))
        raise CanonicalExecutionOriginError(
            f"Canonical execution target {label!r} source does not match module {module_name!r}: "
            f"got {source}; expected one of {expected_text}."
        )
    target_origin = _target_code_origin(label, target, source=source)
    return {
        "module": module_name,
        "path": str(source),
        "sha256": sha256_file(source),
        "package_root": str(package_root),
        "package_origin_kind": str(package_identity.get("kind", "") or ""),
        "target_kind": str(target_origin["kind"]),
        "target_qualname": str(target_origin["qualname"]),
        "target_code_sha256": str(target_origin["code_sha256"]),
    }


def _loader_type_name(loader: Any) -> str:
    return f"{type(loader).__module__}.{type(loader).__qualname__}"


_DATACLASS_GENERATED_METHOD_NAMES = frozenset(
    {
        "__init__",
        "__repr__",
        "__eq__",
        "__hash__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__setattr__",
        "__delattr__",
        "__replace__",
        "__getstate__",
        "__setstate__",
    }
)
_LRU_CACHE_WRAPPER_TYPE = type(functools.lru_cache(maxsize=1)(lambda: None))


def _is_trusted_dataclass_missing(
    value: Any,
    _missing: Any = dataclasses.MISSING,
) -> bool:
    """Recognize only the import-time dataclasses missing sentinel."""

    return value is _missing


def _is_trusted_dataclass_default_factory_sentinel(
    value: Any,
    _sentinel: Any = dataclasses._HAS_DEFAULT_FACTORY,
) -> bool:
    """Recognize only the import-time dataclass factory sentinel."""

    return value is _sentinel


class _TabneticsProvenanceSlotsProbe:
    """Private slot carrier used only to obtain the built-in descriptor type."""

    __slots__ = ("value",)


def _is_trusted_member_descriptor(
    value: Any,
) -> bool:
    """Return whether ``value`` is an exact built-in slot descriptor."""

    return type(value) is type(_TabneticsProvenanceSlotsProbe.value)


def _is_overload_definition(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether an AST definition is a typing-only overload stub."""

    for decorator in node.decorator_list:
        candidate = decorator
        if isinstance(candidate, ast.Call):
            candidate = candidate.func
        if isinstance(candidate, ast.Name) and candidate.id == "overload":
            return True
        if isinstance(candidate, ast.Attribute) and candidate.attr == "overload":
            return True
    return False


def _code_sha256(code: CodeType) -> str:
    """Fingerprint the semantic fields of a code object.

    The reference and live code objects are compiled by the same interpreter in
    the same process.  Unlike a source filename or a module attribute, this
    payload binds bytecode, constants, and nested code objects to the
    independently compiled verified source.  We deliberately avoid raw
    ``marshal.dumps(code)`` here: marshal's reference-table encoding can vary
    between equivalent independently allocated code objects.  Source line
    tables are intentionally excluded because stale-but-semantically-identical
    bytecode caches can retain older debug offsets after a source-only edit.
    """

    return canonical_json_sha256(_code_fingerprint_payload(code))


def _code_constant_payload(value: Any) -> Any:
    """Normalize code constants without relying on object identity."""

    if isinstance(value, CodeType):
        return {"code": _code_fingerprint_payload(value)}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return {"tuple": [_code_constant_payload(item) for item in value]}
    if isinstance(value, frozenset):
        normalized = [_code_constant_payload(item) for item in value]
        return {
            "frozenset": sorted(
                normalized,
                key=lambda item: canonical_json_sha256(item),
            )
        }
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, complex):
        return {"complex": [float(value.real), float(value.imag)]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _code_fingerprint_payload(code: CodeType) -> dict[str, Any]:
    """Return stable code fields including nested code constants."""

    return {
        "argcount": int(code.co_argcount),
        "posonlyargcount": int(code.co_posonlyargcount),
        "kwonlyargcount": int(code.co_kwonlyargcount),
        "nlocals": int(code.co_nlocals),
        "stacksize": int(code.co_stacksize),
        "flags": int(code.co_flags),
        "code": bytes(code.co_code).hex(),
        "consts": [_code_constant_payload(value) for value in code.co_consts],
        "names": [str(value) for value in code.co_names],
        "varnames": [str(value) for value in code.co_varnames],
        "freevars": [str(value) for value in code.co_freevars],
        "cellvars": [str(value) for value in code.co_cellvars],
        "name": str(code.co_name),
        "qualname": str(code.co_qualname),
        "exceptiontable": bytes(code.co_exceptiontable).hex(),
    }


def _walk_code_objects(code: CodeType) -> Sequence[CodeType]:
    """Return one code object and every nested code object in source order."""

    records: list[CodeType] = [code]
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            records.extend(_walk_code_objects(constant))
    return records


class _ExecutionStateUnsupported(TypeError):
    """Raised when a source-owned runtime state value has no safe semantic seal."""


def _safe_class_text_attribute(target: Any, attribute: str) -> str:
    """Read a class metadata string without invoking a custom metaclass.

    Provenance failures must be non-dispatching: an untrusted class can install
    ``__getattribute__`` on its metaclass solely to run code while it is being
    rejected.  Calling ``type.__getattribute__`` explicitly bypasses that
    override and is safe for a class object after the exact type gate.
    """

    if not isinstance(target, type):
        raise _ExecutionStateUnsupported("runtime metadata target is not a class")
    try:
        value = type.__getattribute__(target, attribute)
    except (AttributeError, TypeError) as exc:
        raise _ExecutionStateUnsupported(
            f"runtime class metadata {attribute!r} is unavailable"
        ) from exc
    if type(value) is not str:
        raise _ExecutionStateUnsupported(
            f"runtime class metadata {attribute!r} is not a string"
        )
    return value


def _safe_runtime_type_label(value: Any) -> str:
    """Return a diagnostic type name without invoking an untrusted metaclass."""

    value_type = type(value)
    try:
        module_name = _safe_class_text_attribute(value_type, "__module__")
        qualname = _safe_class_text_attribute(value_type, "__qualname__")
    except _ExecutionStateUnsupported:
        return "<untrusted-runtime-type>"
    return f"{module_name}.{qualname}"


def _safe_class_dict(target: Any) -> MappingProxyType:
    """Return an exact class mapping proxy without metaclass dispatch."""

    if not isinstance(target, type):
        raise _ExecutionStateUnsupported("runtime metadata target is not a class")
    try:
        value = type.__getattribute__(target, "__dict__")
    except (AttributeError, TypeError) as exc:
        raise _ExecutionStateUnsupported(
            "runtime class dictionary is unavailable"
        ) from exc
    if type(value) is not MappingProxyType:
        raise _ExecutionStateUnsupported("runtime class dictionary is invalid")
    return value


@dataclasses.dataclass(frozen=True)
class _IsolatedStatePayload:
    """A clean-reference semantic payload used without rehydrating live objects."""

    payload: Any


@dataclasses.dataclass(frozen=True)
class _StateSerializationContext:
    """Exact runtime identities admitted after the code/origin validation phase.

    Attestation runs in a process that may contain substituted state.  It must
    never discover an object's capabilities by invoking ``items()``,
    ``__iter__``, ``repr()``, dataclass helpers, NumPy coercion, or arbitrary
    attributes.  The code phase constructs this context from independently
    verified source symbols first; serialization then permits only exact
    identities held here or exact built-in containers.
    """

    functions: Mapping[int, tuple[Any, Mapping[str, str]]]
    dataclass_factory_functions: Mapping[int, tuple[FunctionType, Mapping[str, str]]]
    classes: Mapping[int, tuple[Any, Mapping[str, str]]]
    modules: Mapping[int, tuple[ModuleType, Mapping[str, str]]]
    dataclass_fields: Mapping[int, tuple[type, tuple[str, ...], Mapping[str, str]]]
    enum_types: Mapping[int, tuple[type, Mapping[str, str]]]
    package_root: Path | None = None
    repository_root: Path | None = None
    pandas_na_type: type | None = None
    ndarray_type: type | None = None
    regex_type: type | None = None


_EMPTY_STATE_SERIALIZATION_CONTEXT = _StateSerializationContext(
    functions={},
    dataclass_factory_functions={},
    classes={},
    modules={},
    dataclass_fields={},
    enum_types={},
)


def _context_identity_payload(
    records: Mapping[int, tuple[Any, Mapping[str, str]]],
    value: Any,
) -> Mapping[str, str] | None:
    """Look up an exact object identity without invoking its hash/equality."""

    record = records.get(id(value))
    if record is None or record[0] is not value:
        return None
    return record[1]


def _dataclass_factory_global_payload(
    value: Any,
    *,
    context: _StateSerializationContext,
) -> Any:
    """Serialize a source-verified factory global without executing it.

    Dataclass factory lambdas are not direct module symbols, so their global
    reads are not covered by the regular callable-default contract.  Preserve
    ordinary internal/literal values through the state serializer, and give
    external import bindings a structural identity instead of accepting a
    nominal module/qualname match.
    """

    try:
        return {"kind": "state", "value": _state_value_payload(value, context=context)}
    except _ExecutionStateUnsupported:
        value_type = type(value)
        if value_type in {ModuleType, FunctionType, BuiltinFunctionType} or isinstance(
            value, type
        ):
            return {
                "kind": "external_binding",
                "value": _external_binding_identity(value),
            }
        raise


def _dataclass_factory_function_payload(
    target: FunctionType,
    *,
    identity: Mapping[str, str],
    context: _StateSerializationContext,
) -> dict[str, Any]:
    """Seal an exact, source-verified dataclass default factory.

    The code digest was admitted while constructing ``context``.  Defaults,
    closure cells, and globals reachable through the factory's code object are
    still mutable at runtime, so include each in the clean-child-comparable
    payload.  Attribute names in ``co_names`` are ignored unless they resolve
    as a module global or a builtin; this avoids treating ``np.zeros``'s
    ``zeros`` member name as a global while catching a rebound ``np`` binding.
    """

    if type(target) is not FunctionType:
        raise _ExecutionStateUnsupported("dataclass factory is not an exact function")
    code = target.__code__
    if type(code) is not CodeType:
        raise _ExecutionStateUnsupported("dataclass factory has no exact code object")
    module_globals = target.__globals__
    builtin_mapping = target.__builtins__
    if type(module_globals) is not dict or type(builtin_mapping) is not dict:
        raise _ExecutionStateUnsupported(
            "dataclass factory has non-plain globals or builtins"
        )
    raw_names = code.co_names
    if type(raw_names) is not tuple or any(type(name) is not str for name in raw_names):
        raise _ExecutionStateUnsupported("dataclass factory has invalid global names")
    global_records: list[dict[str, Any]] = []
    for name in sorted(set(raw_names)):
        value = dict.get(module_globals, name, _MISSING_RUNTIME_BINDING)
        if value is not _MISSING_RUNTIME_BINDING:
            global_records.append(
                {
                    "name": name,
                    "scope": "module",
                    "value": _dataclass_factory_global_payload(value, context=context),
                }
            )
            continue
        trusted_builtin = _trusted_builtin_value(name, _MISSING_RUNTIME_BINDING)
        if trusted_builtin is _MISSING_RUNTIME_BINDING:
            continue
        actual_builtin = dict.get(builtin_mapping, name, _MISSING_RUNTIME_BINDING)
        if actual_builtin is not trusted_builtin:
            raise _ExecutionStateUnsupported(
                "dataclass factory builtin does not match the trusted builtin table"
            )
        global_records.append(
            {
                "name": name,
                "scope": "builtin",
                "value": _builtin_identity_payload(trusted_builtin),
            }
        )
    return {
        "module": str(identity["module"]),
        "qualname": str(identity["qualname"]),
        "code_sha256": str(identity["code_sha256"]),
        "defaults": _exact_function_default_payload(target, state_context=context),
        "closure": _exact_function_closure_payload(target, state_context=context),
        "globals": global_records,
    }


def _state_exact_dict_items_payload(
    value: dict[Any, Any],
    *,
    context: _StateSerializationContext,
    seen: set[int],
) -> list[dict[str, Any]]:
    """Serialize an exact ``dict`` using unbound built-in operations only."""

    items = [
        {
            "key": _state_value_payload(key, context=context, _seen=seen),
            "value": _state_value_payload(item, context=context, _seen=seen),
        }
        for key, item in dict.items(value)
    ]
    return sorted(items, key=canonical_json_sha256)


def _mappingproxy_backing_dict(value: Any) -> dict[Any, Any]:
    """Return a mapping-proxy backing dict without protocol dispatch.

    ``mappingproxy`` can wrap an arbitrary ``Mapping``.  Calling ``items`` on
    it would dispatch into that mapping, so CPython's GC referent API is used
    only for an exact mappingproxy backed by an exact dict.  Other runtimes and
    custom mapping backings fail closed.
    """

    if sys.implementation.name != "cpython":
        raise _ExecutionStateUnsupported("mappingproxy sealing requires CPython")
    referents = gc.get_referents(value)
    if len(referents) != 1 or type(referents[0]) is not dict:
        raise _ExecutionStateUnsupported("mappingproxy has a non-exact-dict backing")
    return referents[0]


def _state_value_payload(
    value: Any,
    *,
    context: _StateSerializationContext = _EMPTY_STATE_SERIALIZATION_CONTEXT,
    _seen: set[int] | None = None,
) -> Any:
    """Return a fail-closed semantic payload without executing live state.

    Every container branch below first requires an exact built-in type.  This
    ordering is deliberate: a custom ``Mapping`` or dataclass-like object is
    rejected before attribute access, iteration, hashing, or third-party
    conversion can run user-controlled code.
    """

    value_type = type(value)
    if value_type is _IsolatedStatePayload:
        return value.payload
    if value is None:
        return {"kind": "none"}
    if value_type is bool:
        return {"kind": "bool", "value": value}
    if value_type is int:
        return {"kind": "int", "value": value}
    if value_type is float:
        if math.isnan(value):
            rendered = "nan"
        elif math.isinf(value):
            rendered = "inf" if value > 0 else "-inf"
        else:
            rendered = float.hex(value)
        return {"kind": "float", "value": rendered}
    if value_type is complex:
        return {
            "kind": "complex",
            "real": _state_value_payload(float(value.real), context=context),
            "imag": _state_value_payload(float(value.imag), context=context),
        }
    if value_type is str:
        return {"kind": "str", "value": value}
    if value_type is bytes:
        return {"kind": "bytes", "value": bytes.hex(value)}
    if value_type in {PosixPath, WindowsPath}:
        package_root = context.package_root
        if package_root is not None:
            try:
                resolved_value = value.resolve()
                relative = resolved_value.relative_to(package_root.resolve())
            except (OSError, RuntimeError, ValueError):
                pass
            else:
                # A clean child intentionally imports a verified staged copy of
                # the package. Source-owned paths therefore have distinct
                # absolute roots across the two processes; bind them to their
                # package-relative location instead. External paths retain the
                # exact absolute identity below.
                return {
                    "kind": "package_path",
                    "path": PurePosixPath(relative).as_posix(),
                }
        repository_root = context.repository_root
        if repository_root is not None:
            try:
                resolved_value = value.resolve()
                relative = resolved_value.relative_to(repository_root.resolve())
            except (OSError, RuntimeError, ValueError):
                pass
            else:
                return {
                    "kind": "repository_path",
                    "path": PurePosixPath(relative).as_posix(),
                }
        return {"kind": "path", "value": str(value)}
    if _is_trusted_dataclass_missing(value):
        return {"kind": "dataclass_missing"}
    if _is_trusted_dataclass_default_factory_sentinel(value):
        return {"kind": "dataclass_default_factory_sentinel"}
    if value_type is type:
        try:
            builtin_class_module = _safe_class_text_attribute(value, "__module__")
            builtin_class_qualname = _safe_class_text_attribute(value, "__qualname__")
        except _ExecutionStateUnsupported:
            builtin_class_module = ""
            builtin_class_qualname = ""
        trusted_builtin_class = _trusted_builtin_value(
            builtin_class_qualname,
            _MISSING_RUNTIME_BINDING,
        )
        if (
            builtin_class_module == "builtins"
            and builtin_class_qualname
            and trusted_builtin_class is value
        ):
            return {
                "kind": "builtin_class",
                "module": builtin_class_module,
                "qualname": builtin_class_qualname,
            }
        if builtin_class_module == "logging" and builtin_class_qualname == "Logger":
            return {"kind": "stdlib_logging_logger_class"}
    if value_type is ModuleType:
        module_dict = object.__getattribute__(value, "__dict__")
        if (
            type(module_dict) is dict
            and dict.get(module_dict, "__name__") == "builtins"
        ):
            spec = dict.get(module_dict, "__spec__")
            if (
                type(spec) is ModuleSpec
                and str(spec.name or "") == "builtins"
                and str(spec.origin or "") == "built-in"
            ):
                return {"kind": "stdlib_builtins_module"}
        if dict.get(module_dict, "__name__") == "logging":
            spec = dict.get(module_dict, "__spec__")
            source_raw = dict.get(module_dict, "__file__")
            if (
                type(spec) is ModuleSpec
                and str(spec.name or "") == "logging"
                and type(source_raw) is str
                and source_raw
            ):
                source = Path(source_raw).resolve()
                if (
                    source.is_file()
                    and Path(str(spec.origin or "")).resolve() == source
                ):
                    return {
                        "kind": "stdlib_logging_module",
                        "path": str(source),
                        "sha256": sha256_file(source),
                    }

    module_payload = _context_identity_payload(context.modules, value)
    if module_payload is not None:
        return {"kind": "module", **dict(module_payload)}
    if value_type is ModuleType:
        module_values = object.__getattribute__(value, "__dict__")
        module_name = (
            dict.get(module_values, "__name__", "")
            if type(module_values) is dict
            else ""
        )
        if type(module_name) is not str or not module_name:
            raise _ExecutionStateUnsupported(
                "runtime module state has invalid metadata"
            )
        return {
            "kind": "external_module",
            "identity": _external_owner_module_payload(module_name),
        }
    class_payload = _context_identity_payload(context.classes, value)
    if class_payload is not None:
        return {"kind": "class", **dict(class_payload)}
    function_payload = _context_identity_payload(context.functions, value)
    if function_payload is not None:
        return {"kind": "function", **dict(function_payload)}
    dataclass_factory_payload = _context_identity_payload(
        context.dataclass_factory_functions,
        value,
    )
    if dataclass_factory_payload is not None:
        return {
            "kind": "dataclass_factory_function",
            "value": _dataclass_factory_function_payload(
                value,
                identity=dataclass_factory_payload,
                context=context,
            ),
        }

    if context.pandas_na_type is not None and value_type is context.pandas_na_type:
        return {"kind": "pandas_na"}

    enum_record = context.enum_types.get(id(value_type))
    if enum_record is not None and enum_record[0] is value_type:
        enum_name = object.__getattribute__(value, "_name_")
        enum_value = object.__getattribute__(value, "_value_")
        if type(enum_name) is not str:
            raise _ExecutionStateUnsupported("trusted enum has a non-string name")
        return {
            "kind": "enum",
            "class": dict(enum_record[1]),
            "name": enum_name,
            "value": _state_value_payload(enum_value, context=context, _seen=_seen),
        }

    seen = set() if _seen is None else _seen
    object_id = id(value)
    if object_id in seen:
        raise _ExecutionStateUnsupported("cyclic runtime state")
    seen.add(object_id)
    try:
        if value_type is dict:
            return {
                "kind": "dict",
                "items": _state_exact_dict_items_payload(
                    value,
                    context=context,
                    seen=seen,
                ),
            }
        if value_type is MappingProxyType:
            return {
                "kind": "mappingproxy",
                "items": _state_exact_dict_items_payload(
                    _mappingproxy_backing_dict(value),
                    context=context,
                    seen=seen,
                ),
            }
        if value_type is tuple:
            return {
                "kind": "tuple",
                "items": [
                    _state_value_payload(item, context=context, _seen=seen)
                    for item in tuple.__iter__(value)
                ],
            }
        if value_type is list:
            return {
                "kind": "list",
                "items": [
                    _state_value_payload(item, context=context, _seen=seen)
                    for item in list.__iter__(value)
                ],
            }
        if value_type is set:
            items = [
                _state_value_payload(item, context=context, _seen=seen)
                for item in set.__iter__(value)
            ]
            return {"kind": "set", "items": sorted(items, key=canonical_json_sha256)}
        if value_type is frozenset:
            items = [
                _state_value_payload(item, context=context, _seen=seen)
                for item in frozenset.__iter__(value)
            ]
            return {
                "kind": "frozenset",
                "items": sorted(items, key=canonical_json_sha256),
            }

        dataclass_record = context.dataclass_fields.get(id(value_type))
        if dataclass_record is not None and dataclass_record[0] is value_type:
            _, field_names, class_payload = dataclass_record
            return {
                "kind": "dataclass",
                "class": dict(class_payload),
                "fields": [
                    {
                        "name": field_name,
                        "value": _state_value_payload(
                            object.__getattribute__(value, field_name),
                            context=context,
                            _seen=seen,
                        ),
                    }
                    for field_name in field_names
                ],
            }

        if context.ndarray_type is not None and value_type is context.ndarray_type:
            dtype = object.__getattribute__(value, "dtype")
            if bool(dtype.hasobject):
                raise _ExecutionStateUnsupported("object ndarray state is unsupported")
            shape = object.__getattribute__(value, "shape")
            if type(shape) is not tuple or any(type(size) is not int for size in shape):
                raise _ExecutionStateUnsupported("ndarray shape is invalid")
            raw = context.ndarray_type.tobytes(value, order="C")
            if type(raw) is not bytes:
                raise _ExecutionStateUnsupported("ndarray bytes payload is invalid")
            return {
                "kind": "ndarray",
                "dtype": str(dtype),
                "shape": list(shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }

        if context.regex_type is not None and value_type is context.regex_type:
            pattern = object.__getattribute__(value, "pattern")
            flags = object.__getattribute__(value, "flags")
            if type(pattern) is not str or type(flags) is not int:
                raise _ExecutionStateUnsupported("regex state is invalid")
            return {"kind": "regex", "pattern": pattern, "flags": flags}
    finally:
        seen.remove(object_id)
    raise _ExecutionStateUnsupported(
        "unsupported runtime state type " + _safe_runtime_type_label(value)
    )


def _static_source_expression_value(
    node: ast.AST,
    known_values: Mapping[str, Any],
) -> Any:
    """Evaluate the literal-only subset used for source state/default seals."""

    if isinstance(node, ast.Constant):
        if (
            isinstance(node.value, (str, bytes, int, float, complex, bool))
            or node.value is None
        ):
            return node.value
    elif isinstance(node, ast.Name) and node.id in known_values:
        return known_values[node.id]
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [
            _static_source_expression_value(item, known_values) for item in node.elts
        ]
        if isinstance(node, ast.Tuple):
            return tuple(values)
        if isinstance(node, ast.List):
            return list(values)
        return set(values)
    elif isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise _ExecutionStateUnsupported("dictionary unpacking is not static state")
        return {
            _static_source_expression_value(
                key, known_values
            ): _static_source_expression_value(
                value,
                known_values,
            )
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _static_source_expression_value(node.operand, known_values)
        if not isinstance(value, (int, float, complex)) or isinstance(value, bool):
            raise _ExecutionStateUnsupported("unary state expression is not numeric")
        return +value if isinstance(node.op, ast.UAdd) else -value
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozenset"
        and len(node.args) == 1
        and not node.keywords
    ):
        return frozenset(_static_source_expression_value(node.args[0], known_values))
    raise _ExecutionStateUnsupported(
        f"unsupported static source expression {type(node).__name__}"
    )


def _class_direct_callable_qualnames(
    node: ast.ClassDef,
    *,
    class_qualname: str,
) -> tuple[str, ...]:
    """Return source-authored direct callable qualnames for one class body."""

    names: list[str] = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_overload_definition(child):
                names.append(f"{class_qualname}.{child.name}")
    return tuple(names)


def _source_class_base_contract(node: ast.ClassDef) -> dict[str, Any]:
    """Capture source-derived base and metaclass expressions for one class."""

    metaclass_node = next(
        (keyword.value for keyword in node.keywords if keyword.arg == "metaclass"),
        None,
    )
    return {
        "base_nodes": tuple(node.bases),
        "metaclass_node": metaclass_node,
        "source_sha256": canonical_json_sha256(
            {
                "bases": [
                    ast.dump(base, annotate_fields=True, include_attributes=False)
                    for base in node.bases
                ],
                "metaclass": (
                    ""
                    if metaclass_node is None
                    else ast.dump(
                        metaclass_node,
                        annotate_fields=True,
                        include_attributes=False,
                    )
                ),
            }
        ),
    }


def _source_tabnetics_import_bindings(
    tree: ast.AST,
    *,
    module_name: str,
    is_package: bool,
) -> tuple[dict[str, Any], ...]:
    """Return source-declared internal import bindings for one module.

    The binding list is derived from verified source, not the live module
    namespace.  It closes the gap between a source-attested owner module and a
    consumer whose imported global was rebound after import, for example the
    runner's ``DistributionFeatureSelectionPipeline`` dispatch target.
    """

    package_context = module_name if is_package else module_name.rpartition(".")[0]
    import_nodes: list[tuple[ast.Import | ast.ImportFrom, bool]] = []

    def _collect_module_scope_imports(node: ast.AST, *, guarded: bool) -> None:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append((node, guarded))
            return
        for child in ast.iter_child_nodes(node):
            _collect_module_scope_imports(
                child,
                guarded=guarded or not isinstance(node, ast.Module),
            )

    _collect_module_scope_imports(tree, guarded=False)
    bindings: list[dict[str, Any]] = []
    for node, guarded in import_nodes:
        if isinstance(node, ast.ImportFrom):
            raw_module = str(node.module or "")
            if node.level:
                relative_name = "." * int(node.level) + raw_module
                try:
                    imported_module = importlib.util.resolve_name(
                        relative_name,
                        package_context,
                    )
                except (ImportError, ValueError):
                    continue
            else:
                imported_module = raw_module
            if not (
                imported_module == LOADED_PACKAGE_MODULE_PREFIX
                or imported_module.startswith(LOADED_PACKAGE_MODULE_PREFIX + ".")
            ):
                continue
            for alias in node.names:
                if alias.name == "*":
                    # Star imports do not give a static, unambiguous binding
                    # set. The exporting internal module is still attested
                    # independently if it is loaded.
                    continue
                bindings.append(
                    {
                        "local_name": str(alias.asname or alias.name),
                        "module": imported_module,
                        "attribute": str(alias.name),
                        "conditional": guarded,
                    }
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_module = str(alias.name)
                if not (
                    imported_module == LOADED_PACKAGE_MODULE_PREFIX
                    or imported_module.startswith(LOADED_PACKAGE_MODULE_PREFIX + ".")
                ):
                    continue
                # ``import tabnetics.foo`` without an alias binds only the
                # root package name in Python's local namespace.
                bindings.append(
                    {
                        "local_name": str(
                            alias.asname or imported_module.split(".", 1)[0]
                        ),
                        "module": imported_module
                        if alias.asname
                        else LOADED_PACKAGE_MODULE_PREFIX,
                        "attribute": "",
                        "conditional": guarded,
                    }
                )
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in bindings:
        key = (record["local_name"], record["module"], record["attribute"])
        existing = unique.get(key)
        if existing is None:
            unique[key] = record
        else:
            # One unconditional declaration makes the binding unconditional
            # even if another guarded declaration uses the same import.
            existing["conditional"] = bool(
                existing["conditional"] and record["conditional"]
            )
    return tuple(unique[key] for key in sorted(unique))


def _source_external_import_bindings(tree: ast.AST) -> tuple[dict[str, str], ...]:
    """Index top-level external imports without importing or resolving them."""

    if not isinstance(tree, ast.Module):
        raise CanonicalExecutionOriginError("Loaded source does not have a module AST.")
    bindings: list[dict[str, str]] = []

    class _Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module_name = str(node.module or "")
            if not module_name or module_name.startswith(LOADED_PACKAGE_MODULE_PREFIX):
                return
            for alias in node.names:
                if alias.name == "*":
                    continue
                bindings.append(
                    {
                        "local_name": str(alias.asname or alias.name),
                        "module": module_name,
                        "attribute": str(alias.name),
                    }
                )

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                module_name = str(alias.name)
                if module_name.startswith(LOADED_PACKAGE_MODULE_PREFIX):
                    continue
                bindings.append(
                    {
                        "local_name": str(alias.asname or module_name.split(".", 1)[0]),
                        "module": module_name,
                        "attribute": "",
                    }
                )

    _Collector().visit(tree)
    unique = {
        (record["local_name"], record["module"], record["attribute"]): record
        for record in bindings
    }
    return tuple(unique[key] for key in sorted(unique))


def _source_direct_assignment_bindings(tree: ast.AST) -> tuple[dict[str, Any], ...]:
    """Index module-scope assignments, including guarded optional-state branches."""

    if not isinstance(tree, ast.Module):
        raise CanonicalExecutionOriginError("Loaded source does not have a module AST.")
    records: list[dict[str, Any]] = []

    def _visit_statements(
        statements: Sequence[ast.stmt],
        *,
        guarded: bool,
    ) -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            _record_assignment(node, guarded=guarded)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    _visit_statements((child,), guarded=True)
                elif isinstance(child, ast.ExceptHandler):
                    _visit_statements(child.body, guarded=True)
                elif isinstance(child, ast.match_case):
                    _visit_statements(child.body, guarded=True)

    def _target_names(target: ast.AST) -> tuple[str, ...]:
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, ast.Starred):
            return _target_names(target.value)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(name for item in target.elts for name in _target_names(item))
        return ()

    def _root_name(target: ast.AST) -> tuple[str, ...]:
        current = target
        while isinstance(current, (ast.Subscript, ast.Attribute)):
            current = current.value
        return (current.id,) if isinstance(current, ast.Name) else ()

    def _mutated_root_names(target: ast.AST) -> tuple[str, ...]:
        """Return module globals mutated through a subscript/attribute target.

        A literal assignment followed by ``state[key] = ...`` or
        ``state.update(...)`` is not a static value.  Recording the mutation as
        an ambiguous binding forces an explicit clean-reference declaration
        instead of sealing the stale initial literal.
        """

        if not isinstance(target, (ast.Subscript, ast.Attribute)):
            return ()
        return _root_name(target)

    def _record_assignment(node: ast.stmt, *, guarded: bool) -> None:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr
            in {
                "add",
                "append",
                "clear",
                "discard",
                "extend",
                "insert",
                "pop",
                "remove",
                "setdefault",
                "update",
            }
        ):
            for name in _root_name(node.value.func.value):
                records.append({"name": name, "value": None, "conditional": guarded})
            return
        targets: Sequence[ast.expr]
        value: ast.AST | None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
            value = node.value
        elif isinstance(node, ast.AugAssign):
            targets = (node.target,)
            value = None
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = (node.target,)
            value = None
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets = tuple(
                item.optional_vars
                for item in node.items
                if item.optional_vars is not None
            )
            value = None
        else:
            return
        for target in targets:
            for name in _target_names(target):
                records.append({"name": name, "value": value, "conditional": guarded})
            for name in _mutated_root_names(target):
                records.append({"name": name, "value": None, "conditional": guarded})

    _visit_statements(tree.body, guarded=False)
    return tuple(records)


def _source_class_static_bindings(node: ast.ClassDef) -> tuple[dict[str, Any], ...]:
    """Index direct class assignments for literal class-state sealing."""

    records: list[dict[str, Any]] = []
    for child in node.body:
        targets: Sequence[ast.expr]
        value: ast.AST | None
        if isinstance(child, ast.Assign):
            targets = child.targets
            value = child.value
        elif isinstance(child, ast.AnnAssign):
            targets = (child.target,)
            value = child.value
        else:
            continue
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                records.append({"name": target.id, "value": value})
    return tuple(records)


def _source_imported_binding_names(tree: ast.AST) -> frozenset[str]:
    """Return import-local names visible at module scope, including guarded imports."""

    names: set[str] = set()

    class _ImportCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                names.add(str(alias.asname or alias.name.split(".", 1)[0]))

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if alias.name != "*":
                    names.add(str(alias.asname or alias.name))

    _ImportCollector().visit(tree)
    return frozenset(names)


def _callable_default_contract(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, Any]:
    """Return source AST nodes for callable defaults without executing source."""

    keyword_defaults = tuple(
        (str(argument.arg), default)
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
        if default is not None
    )
    return {
        "positional": tuple(node.args.defaults),
        "keyword": keyword_defaults,
    }


def _source_module_scope_definition_nodes(
    tree: ast.AST,
) -> tuple[tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, bool], ...]:
    """Return definitions authored in module-scope executable suites.

    Python binds definitions inside module-level ``if``/``try``/``with`` and
    ``match`` suites directly into the module namespace.  They are therefore
    source-authored exports even though they are not direct ``Module.body``
    children.  Function and class bodies are deliberate traversal barriers:
    nested definitions are not module exports and must remain unrecognized if
    somebody later injects them into the module namespace.

    The boolean records whether the definition is an unconditional direct
    ``Module.body`` child.  Guarded definitions are reconciled with a clean
    isolated import before canonical evidence is emitted.
    """

    if not isinstance(tree, ast.Module):
        raise CanonicalExecutionOriginError("Loaded source does not have a module AST.")
    records: list[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, bool]
    ] = []

    def _visit_suite(statements: Sequence[ast.stmt], *, direct: bool) -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _is_overload_definition(node):
                    records.append((node, direct))
                continue
            if isinstance(node, ast.ClassDef):
                records.append((node, direct))
                continue
            if isinstance(node, ast.If):
                _visit_suite(node.body, direct=False)
                _visit_suite(node.orelse, direct=False)
                continue
            if isinstance(node, (ast.Try, ast.TryStar)):
                _visit_suite(node.body, direct=False)
                for handler in node.handlers:
                    _visit_suite(handler.body, direct=False)
                _visit_suite(node.orelse, direct=False)
                _visit_suite(node.finalbody, direct=False)
                continue
            if isinstance(node, (ast.With, ast.AsyncWith)):
                _visit_suite(node.body, direct=False)
                continue
            if isinstance(node, ast.Match):
                for case in node.cases:
                    _visit_suite(case.body, direct=False)

    _visit_suite(tree.body, direct=True)
    return tuple(records)


def _definition_direct_callable_nodes(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> tuple[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef], ...]:
    """Return one module definition's directly authored callable nodes."""

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ((str(node.name), node),)
    return tuple(
        (f"{node.name}.{child.name}", child)
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not _is_overload_definition(child)
    )


def _source_direct_callable_nodes(
    tree: ast.AST,
) -> tuple[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef], ...]:
    """Return module-scope functions and direct members of module classes."""

    return tuple(
        callable_record
        for definition, _direct in _source_module_scope_definition_nodes(tree)
        for callable_record in _definition_direct_callable_nodes(definition)
    )


def _is_source_dataclass_decorator(decorator: ast.expr) -> bool:
    """Return whether a verified class decorator generates an ``__init__``."""

    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        is_dataclass = target.id == "dataclass"
    elif isinstance(target, ast.Attribute):
        is_dataclass = target.attr == "dataclass"
    else:
        is_dataclass = False
    if not is_dataclass:
        return False
    if not isinstance(decorator, ast.Call):
        return True
    for keyword in decorator.keywords:
        if keyword.arg != "init":
            continue
        return not (
            isinstance(keyword.value, ast.Constant) and keyword.value.value is False
        )
    return True


def _definition_callable_default_contracts(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> dict[str, Mapping[str, Any]]:
    """Return defaults generated or authored by one module definition.

    ``dataclasses`` emits ``__init__`` using dynamic code, so it cannot be
    checked against the module's AST code hash.  The generated constructor is
    still source-derived; request its clean-child defaults/closure payload by a
    synthetic, source-determined label.
    """

    contracts: dict[str, Mapping[str, Any]] = {
        qualname: _callable_default_contract(callable_node)
        for qualname, callable_node in _definition_direct_callable_nodes(node)
    }
    if not isinstance(node, ast.ClassDef):
        return contracts
    if any(
        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and child.name == "__init__"
        for child in node.body
    ):
        return contracts
    if any(_is_source_dataclass_decorator(item) for item in node.decorator_list):
        generated_name = f"{node.name}.__init__"
        if generated_name in contracts:
            raise CanonicalExecutionOriginError(
                "Loaded source has conflicting direct/generated dataclass init "
                f"contracts for {node.name!r}."
            )
        contracts[generated_name] = {"generated_dataclass_init": True}
    return contracts


def _source_callable_global_names_by_definition(
    source_text: str,
    source_path: str,
    tree: ast.AST,
) -> dict[int, tuple[str, ...]]:
    """Resolve global/free names separately for each module definition.

    ``symtable`` supplies Python's lexical resolution rather than treating every
    identifier in a method body as a module global.  Nested functions are part
    of an authored callable's runtime behavior, so their global references are
    folded into the enclosing direct callable as well.
    """

    try:
        table_root = symtable.symtable(source_text, source_path, "exec")
    except (SyntaxError, TypeError, ValueError) as exc:
        raise CanonicalExecutionOriginError(
            f"Cannot inspect source state references for {source_path}."
        ) from exc

    tables_by_line: dict[tuple[int, str], list[Any]] = {}

    def _index(table: Any) -> None:
        if str(table.get_type()) == "function":
            tables_by_line.setdefault(
                (int(table.get_lineno()), str(table.get_name())),
                [],
            ).append(table)
        for child in table.get_children():
            _index(child)

    def _table_globals(table: Any) -> set[str]:
        names = {
            str(symbol.get_name())
            for symbol in table.get_symbols()
            if bool(symbol.is_global()) and bool(symbol.is_referenced())
        }
        for child in table.get_children():
            names.update(_table_globals(child))
        return names

    _index(table_root)
    names_by_definition: dict[int, tuple[str, ...]] = {}
    for definition, _direct in _source_module_scope_definition_nodes(tree):
        names: set[str] = set()
        for _qualname, node in _definition_direct_callable_nodes(definition):
            candidates = tables_by_line.get((int(node.lineno), str(node.name)), ())
            if len(candidates) != 1:
                raise CanonicalExecutionOriginError(
                    "Cannot uniquely resolve source callable state references for "
                    f"{source_path}:{node.lineno}:{node.name}."
                )
            names.update(_table_globals(candidates[0]))
            for default in tuple(node.args.defaults) + tuple(
                item for item in node.args.kw_defaults if item is not None
            ):
                names.update(
                    child.id
                    for child in ast.walk(default)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                )
        names_by_definition[id(definition)] = tuple(sorted(names))
    return names_by_definition


def _definition_code_hashes(
    module_code: CodeType,
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    *,
    source_path: str,
) -> dict[str, tuple[str, ...]]:
    """Return independently compiled hashes belonging to one definition variant."""

    first_line = min(
        (int(node.lineno), *(int(item.lineno) for item in node.decorator_list)),
    )
    roots = [
        value
        for value in module_code.co_consts
        if isinstance(value, CodeType)
        and str(value.co_name) == str(node.name)
        and int(value.co_firstlineno) == first_line
    ]
    if len(roots) != 1:
        raise CanonicalExecutionOriginError(
            "Cannot uniquely resolve compiled module definition for "
            f"{source_path}:{node.lineno}:{node.name}."
        )
    hashes: dict[str, list[str]] = {}
    for code in _walk_code_objects(roots[0]):
        qualname = str(getattr(code, "co_qualname", "") or "")
        if qualname:
            hashes.setdefault(qualname, []).append(_code_sha256(code))
    return {
        qualname: tuple(sorted(values)) for qualname, values in sorted(hashes.items())
    }


@functools.lru_cache(maxsize=512)
def _source_symbol_reference(
    source_path: str,
    source_sha256: str,
    module_name: str,
    is_package: bool,
) -> dict[str, Any]:
    """Compile verified source without executing it and index authored symbols.

    This is intentionally not an import of the live module under another name:
    imports can inherit a contaminated ``sys.modules`` graph.  Compiling the
    verified source bytes provides an independent code reference without
    executing module side effects or trusting live module globals.
    """

    source = Path(source_path)
    try:
        source_bytes = source.read_bytes()
    except OSError as exc:
        raise CanonicalExecutionOriginError(
            f"Cannot read loaded tabnetics module source for symbol attestation: {source}."
        ) from exc
    observed_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if observed_sha256 != source_sha256:
        raise CanonicalExecutionOriginError(
            "Loaded tabnetics module source changed while its symbol reference was being "
            f"compiled: {source}."
        )
    try:
        tree = ast.parse(source_bytes, filename=source_path)
        module_code = compile(source_bytes, source_path, "exec", dont_inherit=True)
        source_text = source_bytes.decode("utf-8")
    except (SyntaxError, TypeError, ValueError) as exc:
        raise CanonicalExecutionOriginError(
            f"Cannot compile loaded tabnetics module source for symbol attestation: {source}."
        ) from exc

    definition_nodes = _source_module_scope_definition_nodes(tree)
    global_names_by_definition = _source_callable_global_names_by_definition(
        source_text,
        source_path,
        tree,
    )
    groups: dict[str, dict[str, Any]] = {}
    for node, direct in definition_nodes:
        name = str(node.name)
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        variant_sha256 = canonical_json_sha256(
            {
                "kind": kind,
                "definition": ast.dump(
                    node,
                    annotate_fields=True,
                    include_attributes=False,
                ),
            }
        )
        variant = {
            "name": name,
            "kind": kind,
            "qualname": name,
            "member_qualnames": (
                _class_direct_callable_qualnames(node, class_qualname=name)
                if isinstance(node, ast.ClassDef)
                else ()
            ),
            "class_base_contract": (
                _source_class_base_contract(node)
                if isinstance(node, ast.ClassDef)
                else None
            ),
            "class_static_bindings": (
                _source_class_static_bindings(node)
                if isinstance(node, ast.ClassDef)
                else ()
            ),
            "variant_sha256": variant_sha256,
            "code_hashes_by_qualname": _definition_code_hashes(
                module_code,
                node,
                source_path=source_path,
            ),
            "callable_defaults": _definition_callable_default_contracts(node),
            "module_state_references": global_names_by_definition[id(node)],
        }
        group = groups.setdefault(
            name,
            {
                "name": name,
                "required": False,
                "conditional": False,
                "variants": {},
            },
        )
        group["required"] = bool(group["required"] or direct)
        group["conditional"] = bool(group["conditional"] or not direct)
        variants = group["variants"]
        if not isinstance(variants, dict):
            raise CanonicalExecutionOriginError(
                f"Loaded source has an invalid definition group for {name!r}: {source}."
            )
        existing = variants.get(variant_sha256)
        if existing is None:
            variants[variant_sha256] = variant

    definition_groups = tuple(
        {
            "name": str(group["name"]),
            "required": bool(group["required"]),
            "conditional": bool(group["conditional"]),
            "variants": tuple(
                group["variants"][variant_sha256]
                for variant_sha256 in sorted(group["variants"])
            ),
        }
        for _name, group in sorted(groups.items())
    )
    requires_clean_selection = any(
        bool(group["conditional"]) or len(group["variants"]) > 1
        for group in definition_groups
    )
    return {
        # These four fields are populated only after exact live code-origin
        # matching chooses one source variant per bound module name.
        "definitions": (),
        "code_hashes_by_qualname": {},
        "module_state_references": (),
        "callable_defaults": {},
        "definition_groups": definition_groups,
        "requires_clean_definition_selection": requires_clean_selection,
        "imports": _source_tabnetics_import_bindings(
            tree,
            module_name=module_name,
            is_package=is_package,
        ),
        "external_imports": _source_external_import_bindings(tree),
        "module_state_bindings": _source_direct_assignment_bindings(tree),
        "imported_binding_names": _source_imported_binding_names(tree),
    }


def _reference_code_hashes(
    reference: Mapping[str, Any],
    qualname: str,
    *,
    module_name: str,
) -> tuple[str, ...]:
    raw_hashes = reference.get("code_hashes_by_qualname")
    if not isinstance(raw_hashes, Mapping):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid source symbol reference."
        )
    values = raw_hashes.get(qualname)
    if not isinstance(values, tuple) or not values:
        raise CanonicalExecutionOriginError(
            "Loaded tabnetics module "
            f"{module_name!r} source has no independently compiled code for {qualname!r}."
        )
    if any(not _is_sha256_digest(value) for value in values):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} source symbol digest is invalid."
        )
    return tuple(str(value) for value in values)


@dataclasses.dataclass(frozen=True)
class _SourceStaticStatePlan:
    """Source-only static values and explicit exceptions for one module."""

    values: Mapping[str, Any]
    assigned_names: frozenset[str]
    ambiguous_names: frozenset[str]
    unsupported_names: frozenset[str]
    ephemeral_names: frozenset[str]
    isolated_specs: Mapping[str, Mapping[str, Any]]
    internal_import_state_specs: Mapping[str, Mapping[str, Any]]


def _source_static_state_plan(
    reference: Mapping[str, Any],
    *,
    module_name: str,
) -> _SourceStaticStatePlan:
    """Build a source-AST-only state plan without reading live module values."""

    raw_bindings = reference.get("module_state_bindings")
    if not isinstance(raw_bindings, tuple):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid source state index."
        )
    candidates: dict[str, list[ast.AST | None]] = {}
    conditional_candidates: dict[str, list[bool]] = {}
    for binding in raw_bindings:
        if not isinstance(binding, Mapping):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid source state binding."
            )
        name = str(binding.get("name", "") or "")
        value = binding.get("value")
        conditional = binding.get("conditional")
        if (
            not name
            or (value is not None and not isinstance(value, ast.AST))
            or type(conditional) is not bool
        ):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid source state value."
            )
        candidates.setdefault(name, []).append(value)
        conditional_candidates.setdefault(name, []).append(conditional)
    ambiguous = frozenset(
        name for name, values in candidates.items() if len(values) != 1
    )
    unsupported = frozenset(
        name
        for name, values in candidates.items()
        if len(values) == 1 and values[0] is None
    )
    values: dict[str, Any] = {}
    for name, candidates_for_name in candidates.items():
        if len(candidates_for_name) != 1 or not isinstance(
            candidates_for_name[0], ast.AST
        ):
            continue
        try:
            values[name] = _static_source_expression_value(
                candidates_for_name[0],
                values,
            )
        except _ExecutionStateUnsupported:
            continue

    raw_imports = reference.get("imports")
    if not isinstance(raw_imports, tuple):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid source import index."
        )
    imports_by_local_name: dict[str, list[dict[str, Any]]] = {}
    for binding in raw_imports:
        if not isinstance(binding, Mapping):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid source import binding."
            )
        local_name = binding.get("local_name")
        owner_name = binding.get("module")
        attribute = binding.get("attribute")
        conditional = binding.get("conditional")
        if (
            type(local_name) is not str
            or not local_name
            or type(owner_name) is not str
            or not (
                owner_name == LOADED_PACKAGE_MODULE_PREFIX
                or owner_name.startswith(LOADED_PACKAGE_MODULE_PREFIX + ".")
            )
            or type(attribute) is not str
            or type(conditional) is not bool
        ):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid internal import binding."
            )
        imports_by_local_name.setdefault(local_name, []).append(
            {
                "local_name": local_name,
                "module": owner_name,
                "attribute": attribute,
                "conditional": conditional,
            }
        )

    # A narrow class of nonliteral module state is nevertheless fully
    # source-determined: an exact built-in collection composed from one or
    # more unconditional internal imports plus already resolved literals.
    # Request those values from the existing source-pinned clean child.  Do
    # not admit guarded/fallback aliases, multiply-declared aliases, external
    # imports, arbitrary calls, attributes, operators, or
    # mutated/conditional targets.
    eligible_imports: dict[str, dict[str, Any]] = {}
    for local_name, bindings in imports_by_local_name.items():
        if (
            len(bindings) == 1
            and bindings[0]["conditional"] is False
            and local_name not in candidates
        ):
            eligible_imports[local_name] = bindings[0]
    imported_markers = {name: object() for name in eligible_imports}
    internal_import_state_specs: dict[str, Mapping[str, Any]] = {}
    for name, candidates_for_name in candidates.items():
        if (
            name in values
            or len(candidates_for_name) != 1
            or not isinstance(candidates_for_name[0], ast.AST)
            or conditional_candidates.get(name) != [False]
        ):
            continue
        expression = candidates_for_name[0]
        loaded_names = {
            node.id
            for node in ast.walk(expression)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        dependency_names = tuple(sorted(loaded_names.intersection(eligible_imports)))
        if not dependency_names:
            continue
        try:
            _static_source_expression_value(
                expression,
                {**values, **imported_markers},
            )
        except _ExecutionStateUnsupported:
            continue
        import_bindings = tuple(
            {
                "local_name": eligible_imports[dependency]["local_name"],
                "module": eligible_imports[dependency]["module"],
                "attribute": eligible_imports[dependency]["attribute"],
            }
            for dependency in dependency_names
        )
        internal_import_state_specs[name] = {
            "provider": "clean_isolated_internal_import_expression_v1",
            "source_expression_sha256": canonical_json_sha256(
                {
                    "expression": ast.dump(
                        expression,
                        annotate_fields=True,
                        include_attributes=False,
                    ),
                    "import_bindings": import_bindings,
                }
            ),
            "import_bindings": import_bindings,
        }

    def _declared_names(declaration_name: str) -> frozenset[str]:
        declaration = values.get(declaration_name)
        if declaration is None:
            return frozenset()
        if not isinstance(declaration, (tuple, list, set, frozenset)) or not all(
            isinstance(item, str) and item for item in declaration
        ):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid {declaration_name} "
                "declaration."
            )
        return frozenset(str(item) for item in declaration)

    isolated_raw = values.get("__tabnetics_execution_isolated_state__")
    isolated_specs: dict[str, Mapping[str, Any]] = {}
    if isolated_raw is not None:
        if not isinstance(isolated_raw, dict):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid "
                "__tabnetics_execution_isolated_state__ declaration."
            )
        for name, spec in isolated_raw.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(spec, dict)
                or set(spec) != {"provider", "dependencies"}
                or spec.get("provider") != "clean_isolated_reference_v1"
                or not isinstance(spec.get("dependencies"), (tuple, list))
                or not all(
                    isinstance(dependency, str) and dependency
                    for dependency in spec["dependencies"]
                )
            ):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} has an invalid isolated "
                    "state declaration."
                )
            isolated_specs[str(name)] = {
                "provider": "clean_isolated_reference_v1",
                "dependencies": tuple(str(item) for item in spec["dependencies"]),
            }
    return _SourceStaticStatePlan(
        values=values,
        assigned_names=frozenset(candidates),
        ambiguous_names=ambiguous,
        unsupported_names=unsupported,
        ephemeral_names=_declared_names("__tabnetics_execution_ephemeral_globals__"),
        isolated_specs=isolated_specs,
        internal_import_state_specs=internal_import_state_specs,
    )


def _source_declares_process_logger(reference: Mapping[str, Any]) -> bool:
    """Return whether verified source declares the one exempt diagnostic logger.

    A logger is the only process-local global allowed to avoid a semantic state
    seal.  The exemption belongs to the source binding, not to every object
    whose runtime type happens to be ``logging.Logger``.
    """

    raw_bindings = reference.get("module_state_bindings")
    if not isinstance(raw_bindings, tuple):
        return False
    for binding in raw_bindings:
        if not isinstance(binding, Mapping) or binding.get("name") != "logger":
            continue
        expression = binding.get("value")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "getLogger"
            and isinstance(expression.func.value, ast.Name)
            and expression.func.value.id == "logging"
        ):
            continue
        return True
    return False


def _validated_internal_import_state_spec(
    spec: Any,
    *,
    module_name: str,
    state_name: str,
) -> tuple[str, tuple[dict[str, str], ...]]:
    """Validate one source-derived internal-import state contract."""

    if not isinstance(spec, Mapping) or set(spec) != {
        "provider",
        "source_expression_sha256",
        "import_bindings",
    }:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid internal-import "
            f"state contract for {state_name!r}."
        )
    source_expression_sha256 = spec.get("source_expression_sha256")
    raw_bindings = spec.get("import_bindings")
    if (
        spec.get("provider") != "clean_isolated_internal_import_expression_v1"
        or not _is_sha256_digest(source_expression_sha256)
        or not isinstance(raw_bindings, tuple)
        or not raw_bindings
    ):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid internal-import "
            f"state contract for {state_name!r}."
        )
    bindings: list[dict[str, str]] = []
    for binding in raw_bindings:
        if not isinstance(binding, Mapping) or set(binding) != {
            "local_name",
            "module",
            "attribute",
        }:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid internal-import "
                f"state contract for {state_name!r}."
            )
        local_name = binding.get("local_name")
        owner_name = binding.get("module")
        attribute = binding.get("attribute")
        if (
            type(local_name) is not str
            or not local_name
            or type(owner_name) is not str
            or not (
                owner_name == LOADED_PACKAGE_MODULE_PREFIX
                or owner_name.startswith(LOADED_PACKAGE_MODULE_PREFIX + ".")
            )
            or type(attribute) is not str
        ):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid internal-import "
                f"state contract for {state_name!r}."
            )
        bindings.append(
            {
                "local_name": local_name,
                "module": owner_name,
                "attribute": attribute,
            }
        )
    canonical_bindings = sorted(
        bindings,
        key=lambda item: (item["local_name"], item["module"], item["attribute"]),
    )
    if bindings != canonical_bindings or len(
        {(item["local_name"], item["module"], item["attribute"]) for item in bindings}
    ) != len(bindings):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has a noncanonical internal-import "
            f"state contract for {state_name!r}."
        )
    return str(source_expression_sha256), tuple(bindings)


_CLEAN_ISOLATED_REFERENCE_SCHEMA_VERSION = "tabnetics_clean_isolated_reference_v2"
_CLEAN_ISOLATED_REFERENCE_TIMEOUT_SEC = 120.0
_CLEAN_ISOLATED_REFERENCE_MAX_RSS_MB = 16384
_CLEAN_ISOLATED_REFERENCE_ENV_KEYS: tuple[str, ...] = (
    "LANG",
    "LC_ALL",
    "TZ",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
)
_CLEAN_ISOLATED_REFERENCE_FIXED_ENV: Mapping[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONNOUSERSITE": "1",
}
_CLEAN_ISOLATED_REFERENCE_PATH_KEYS: tuple[str, ...] = (
    "work_dir",
    "source_package_root",
    "home",
    "xdg_cache_home",
    "xdg_config_home",
    "xdg_data_home",
    "xdg_state_home",
    "hf_home",
    "huggingface_hub_cache",
    "hf_datasets_cache",
    "transformers_cache",
    "torch_home",
    "matplotlib_config_dir",
    "python_pycache_prefix",
    "temporary_dir",
    "joblib_temp_folder",
)
_CLEAN_ISOLATED_REFERENCE_PATH_ENV_BINDINGS: tuple[tuple[str, str], ...] = (
    ("HOME", "home"),
    ("XDG_CACHE_HOME", "xdg_cache_home"),
    ("XDG_CONFIG_HOME", "xdg_config_home"),
    ("XDG_DATA_HOME", "xdg_data_home"),
    ("XDG_STATE_HOME", "xdg_state_home"),
    ("HF_HOME", "hf_home"),
    ("HUGGINGFACE_HUB_CACHE", "huggingface_hub_cache"),
    ("HF_DATASETS_CACHE", "hf_datasets_cache"),
    ("TRANSFORMERS_CACHE", "transformers_cache"),
    ("TORCH_HOME", "torch_home"),
    ("MPLCONFIGDIR", "matplotlib_config_dir"),
    ("PYTHONPYCACHEPREFIX", "python_pycache_prefix"),
    ("TMPDIR", "temporary_dir"),
    ("TEMP", "temporary_dir"),
    ("TMP", "temporary_dir"),
    ("JOBLIB_TEMP_FOLDER", "joblib_temp_folder"),
)


@dataclasses.dataclass(frozen=True)
class _PreparedLoadedModule:
    """Verified source metadata retained across the closure's attestation phases."""

    module_name: str
    module: ModuleType
    source: Path
    source_sha256: str
    is_package: bool
    reference: Mapping[str, Any]
    state_plan: _SourceStaticStatePlan


@dataclasses.dataclass(frozen=True)
class _CleanIsolatedReference:
    """Validated clean-child payloads indexed by module and requested symbol."""

    module_payloads: Mapping[str, Mapping[str, Any]]
    source_tree_sha256: str
    environment: Mapping[str, str]
    isolated_paths: Mapping[str, str]
    python: Mapping[str, str]
    platform: Mapping[str, str]
    dependencies: Mapping[str, Mapping[str, str]]

    def state_payload(self, module_name: str, state_name: str) -> Any:
        module = self.module_payloads.get(module_name)
        if not isinstance(module, Mapping):
            raise CanonicalExecutionOriginError(
                f"Clean isolated reference has no module payload for {module_name!r}."
            )
        states = module.get("states")
        if not isinstance(states, Mapping) or state_name not in states:
            raise CanonicalExecutionOriginError(
                "Clean isolated reference is missing requested state "
                f"{module_name!r}:{state_name!r}."
            )
        return states[state_name]

    def default_payload(self, module_name: str, qualname: str) -> Mapping[str, Any]:
        module = self.module_payloads.get(module_name)
        if not isinstance(module, Mapping):
            raise CanonicalExecutionOriginError(
                f"Clean isolated reference has no module payload for {module_name!r}."
            )
        defaults = module.get("defaults")
        value = defaults.get(qualname) if isinstance(defaults, Mapping) else None
        if not isinstance(value, Mapping):
            raise CanonicalExecutionOriginError(
                "Clean isolated reference is missing requested callable defaults "
                f"{module_name!r}:{qualname!r}."
            )
        return value

    def builtin_payload(self, module_name: str, builtin_name: str) -> Mapping[str, str]:
        module = self.module_payloads.get(module_name)
        if not isinstance(module, Mapping):
            raise CanonicalExecutionOriginError(
                f"Clean isolated reference has no module payload for {module_name!r}."
            )
        builtins_payload = module.get("builtins")
        value = (
            builtins_payload.get(builtin_name)
            if isinstance(builtins_payload, Mapping)
            else None
        )
        if not isinstance(value, Mapping):
            raise CanonicalExecutionOriginError(
                "Clean isolated reference is missing requested builtin "
                f"{module_name!r}:{builtin_name!r}."
            )
        return value


def _source_tree_manifest(package_root: Path) -> list[dict[str, str]]:
    """Return a complete, deterministic manifest for the canonical source tree."""

    package_root = package_root.resolve()
    if not package_root.is_dir():
        raise CanonicalExecutionOriginError(
            f"Canonical tabnetics source root is unavailable: {package_root}."
        )
    records: list[dict[str, str]] = []
    for source in sorted(package_root.rglob("*.py")):
        resolved_source = source.resolve()
        if (
            source.is_symlink()
            or not source.is_file()
            or not _is_within(resolved_source, package_root)
        ):
            raise CanonicalExecutionOriginError(
                f"Canonical tabnetics source manifest found an unsafe source entry: {source}."
            )
        records.append(
            {
                "path": source.relative_to(package_root).as_posix(),
                "sha256": sha256_file(source),
            }
        )
    if not records or records[0]["path"] != "__init__.py":
        raise CanonicalExecutionOriginError(
            "Canonical tabnetics source manifest does not contain the package root."
        )
    return records


def _editable_checkout_repository_root(package_root: Path) -> Path | None:
    """Return the monorepo root for the editable package layout, when present."""

    package_root = package_root.resolve()
    try:
        repository_root = package_root.parents[2]
    except IndexError:
        return None
    if (
        (repository_root / "core" / "src" / "tabnetics").resolve() != package_root
        or not (repository_root / "core" / "pyproject.toml").is_file()
        or not (repository_root / "ui" / "pyproject.toml").is_file()
    ):
        return None
    return repository_root


def _staged_clean_reference_repository_root(package_root: Path) -> Path | None:
    """Return the generated checkout root used only by a clean child."""

    repository_root = package_root.resolve().parent
    if (
        not (repository_root / "pyproject.toml").is_file()
        or not (repository_root / "core" / "pyproject.toml").is_file()
        or not (repository_root / "ui" / "pyproject.toml").is_file()
    ):
        return None
    return repository_root


def _write_clean_reference_checkout_markers(package_root: Path) -> None:
    """Make source-relative ``REPO_ROOT`` declarations reproducible in the child."""

    repository_root = package_root.resolve().parent
    markers = (
        repository_root / "pyproject.toml",
        repository_root / "core" / "pyproject.toml",
        repository_root / "ui" / "pyproject.toml",
    )
    try:
        for marker in markers:
            marker.parent.mkdir(parents=True, exist_ok=True)
            with marker.open("xb") as handle:
                handle.write(b"# Clean isolated-reference checkout marker.\n")
    except OSError as exc:
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference source copy could not create checkout markers."
        ) from exc
    if _staged_clean_reference_repository_root(package_root) != repository_root:
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference checkout markers are invalid."
        )


def _clean_reference_environment() -> dict[str, str]:
    """Return the small recorded environment exposed to the clean child."""

    environment = {
        key: str(os.environ[key])
        for key in _CLEAN_ISOLATED_REFERENCE_ENV_KEYS
        if key in os.environ
    }
    environment.update(_CLEAN_ISOLATED_REFERENCE_FIXED_ENV)
    return dict(sorted(environment.items()))


def _manifest_relative_source_path(value: Any) -> PurePosixPath:
    """Validate one manifest path before it is used to materialize a child copy."""

    if type(value) is not str or not value or "\\" in value:
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference manifest path is invalid."
        )
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference manifest path is unsafe."
        )
    if relative.suffix != ".py":
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference manifest contains a non-Python source."
        )
    return relative


def _materialize_verified_clean_source_copy(
    request: Mapping[str, Any],
    *,
    destination_root: Path,
) -> Path:
    """Copy the manifest-pinned package into a private child-only source tree."""

    package_root_raw = request.get("package_root")
    manifest = request.get("source_manifest")
    manifest_sha256 = request.get("source_manifest_sha256")
    if (
        type(package_root_raw) is not str
        or not isinstance(manifest, list)
        or not _is_sha256_digest(manifest_sha256)
        or canonical_json_sha256(manifest) != manifest_sha256
    ):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference source manifest is invalid."
        )

    package_root = Path(package_root_raw).resolve()
    if not package_root.is_dir():
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference package root is unavailable."
        )
    child_root = destination_root.resolve() / "source" / "tabnetics"
    seen_paths: set[str] = set()
    expected_manifest: list[dict[str, str]] = []
    for record in manifest:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference manifest record is invalid."
            )
        relative = _manifest_relative_source_path(record.get("path"))
        expected_sha256 = record.get("sha256")
        if not _is_sha256_digest(expected_sha256) or relative.as_posix() in seen_paths:
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference manifest record is invalid."
            )
        seen_paths.add(relative.as_posix())
        source_path = package_root / Path(*relative.parts)
        source = source_path.resolve()
        if (
            not _is_within(source, package_root)
            or not source.is_file()
            or source_path.is_symlink()
        ):
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference source copy encountered an unsafe source path."
            )
        try:
            source_bytes = source.read_bytes()
        except OSError as exc:
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference source copy could not read a manifest source."
            ) from exc
        if hashlib.sha256(source_bytes).hexdigest() != expected_sha256:
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference source changed before private-copy materialization."
            )
        destination = child_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as handle:
                handle.write(source_bytes)
        except OSError as exc:
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference source copy could not write a private source file."
            ) from exc
        expected_manifest.append(
            {"path": relative.as_posix(), "sha256": str(expected_sha256)}
        )

    if expected_manifest != manifest or _source_tree_manifest(child_root) != manifest:
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference private source copy does not match the verified manifest."
        )
    if _editable_checkout_repository_root(package_root) is not None:
        _write_clean_reference_checkout_markers(child_root)
    return child_root


def _clean_reference_isolated_environment(
    *,
    work_dir: Path,
    source_package_root: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Create the child-owned paths plus runtime and stable evidence environments."""

    work_dir = work_dir.resolve()
    paths = {
        "work_dir": work_dir,
        "source_package_root": source_package_root.resolve(),
        "home": work_dir / "home",
        "xdg_cache_home": work_dir / "xdg" / "cache",
        "xdg_config_home": work_dir / "xdg" / "config",
        "xdg_data_home": work_dir / "xdg" / "data",
        "xdg_state_home": work_dir / "xdg" / "state",
        "hf_home": work_dir / "hf",
        "huggingface_hub_cache": work_dir / "hf" / "hub",
        "hf_datasets_cache": work_dir / "hf" / "datasets",
        "transformers_cache": work_dir / "hf" / "transformers",
        "torch_home": work_dir / "torch",
        "matplotlib_config_dir": work_dir / "matplotlib",
        "python_pycache_prefix": work_dir / "pycache",
        "temporary_dir": work_dir / "tmp",
        "joblib_temp_folder": work_dir / "joblib",
    }
    if set(paths) != set(_CLEAN_ISOLATED_REFERENCE_PATH_KEYS) or any(
        not _is_within(path.resolve(), work_dir) for path in paths.values()
    ):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference path layout is invalid."
        )
    for path in paths.values():
        if path != work_dir:
            path.mkdir(parents=True, exist_ok=True)
    child_environment = _clean_reference_environment()
    child_environment.update(
        {
            name: str(paths[path_name])
            for name, path_name in _CLEAN_ISOLATED_REFERENCE_PATH_ENV_BINDINGS
        }
    )
    evidence_environment = _clean_reference_environment()
    evidence_environment.update(
        {
            name: f"child_owned:{path_name}"
            for name, path_name in _CLEAN_ISOLATED_REFERENCE_PATH_ENV_BINDINGS
        }
    )
    evidence_paths = {
        key: (
            "verified_temp_source_copy"
            if key == "source_package_root"
            else f"child_owned:{key}"
        )
        for key in _CLEAN_ISOLATED_REFERENCE_PATH_KEYS
    }
    return (
        dict(sorted(child_environment.items())),
        dict(sorted(evidence_environment.items())),
        dict(sorted(evidence_paths.items())),
    )


def _source_state_request(
    prepared: _PreparedLoadedModule,
) -> tuple[list[str], list[str], list[str]]:
    """Return declared dynamic globals, callable defaults, and dependencies.

    A source-level declaration is mandatory for every referenced module global
    that is neither a single unambiguous literal AST value nor a constrained
    expression over unconditional, source-declared internal imports.  The
    latter is evaluated only by the source-pinned clean child.  This blocks a
    live process from choosing its own evaluator or provider for a mutated flag.
    """

    raw_references = prepared.reference.get("module_state_references")
    raw_definitions = prepared.reference.get("definitions")
    imported_names = prepared.reference.get("imported_binding_names")
    raw_defaults = prepared.reference.get("callable_defaults")
    if (
        not isinstance(raw_references, tuple)
        or not all(type(name) is str for name in raw_references)
        or not isinstance(raw_definitions, tuple)
        or not isinstance(imported_names, frozenset)
        or not isinstance(raw_defaults, Mapping)
    ):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {prepared.module_name!r} has an invalid source state index."
        )
    definition_names = {
        str(record.get("name", "") or "")
        for record in raw_definitions
        if isinstance(record, Mapping)
    }
    logger_binding = _source_declares_process_logger(prepared.reference)
    state_names: list[str] = []
    dependencies: set[str] = set()
    for name in raw_references:
        if name in definition_names or name in imported_names:
            continue
        if name in {"__name__", "__file__", "__package__"}:
            continue
        if (
            name not in prepared.state_plan.assigned_names
            and _trusted_builtin_value(name, _MISSING_RUNTIME_BINDING)
            is not _MISSING_RUNTIME_BINDING
        ):
            continue
        if name == "__tabnetics_execution_ephemeral_globals__":
            continue
        if name == "logger" and logger_binding:
            continue
        if name in prepared.state_plan.ephemeral_names:
            continue
        spec = prepared.state_plan.isolated_specs.get(name)
        if isinstance(spec, Mapping):
            state_names.append(name)
            raw_dependencies = spec.get("dependencies")
            if not isinstance(raw_dependencies, tuple):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {prepared.module_name!r} has an invalid isolated state declaration."
                )
            dependencies.update(str(item) for item in raw_dependencies)
            continue
        internal_import_spec = prepared.state_plan.internal_import_state_specs.get(name)
        if isinstance(internal_import_spec, Mapping):
            _validated_internal_import_state_spec(
                internal_import_spec,
                module_name=prepared.module_name,
                state_name=name,
            )
            state_names.append(name)
            continue
        if (
            name in prepared.state_plan.values
            and name not in prepared.state_plan.ambiguous_names
            and name not in prepared.state_plan.unsupported_names
        ):
            continue
        if not isinstance(spec, Mapping):
            raise CanonicalExecutionOriginError(
                "Loaded tabnetics module "
                f"{prepared.module_name!r} has referenced nonliteral/conditional state {name!r} "
                "without a clean isolated-reference declaration."
            )
    default_names = sorted(str(name) for name in raw_defaults)
    return sorted(set(state_names)), default_names, sorted(dependencies)


def _source_builtin_names(prepared: _PreparedLoadedModule) -> list[str]:
    """Return source lexical builtin references not shadowed by authored globals."""

    raw_references = prepared.reference.get("module_state_references")
    raw_definitions = prepared.reference.get("definitions")
    imported_names = prepared.reference.get("imported_binding_names")
    if (
        not isinstance(raw_references, tuple)
        or not isinstance(raw_definitions, tuple)
        or not isinstance(imported_names, frozenset)
    ):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {prepared.module_name!r} has an invalid builtin index."
        )
    definition_names = {
        str(record.get("name", "") or "")
        for record in raw_definitions
        if isinstance(record, Mapping)
    }
    names: set[str] = set()
    for name in raw_references:
        if (
            name in definition_names
            or name in imported_names
            or name in prepared.state_plan.assigned_names
            or name == "__builtins__"
            or name.startswith("__")
        ):
            continue
        candidate = _trusted_builtin_value(name, _MISSING_RUNTIME_BINDING)
        if candidate is not _MISSING_RUNTIME_BINDING:
            names.add(name)
    return sorted(names)


def _declared_dependency_import_bindings(
    prepared: _PreparedLoadedModule,
    dependencies: Sequence[str],
) -> list[dict[str, str]]:
    """Return source-declared external bindings covered by a state dependency."""

    declared_roots = {str(name).replace("-", "_") for name in dependencies}
    raw_bindings = prepared.reference.get("external_imports")
    if not isinstance(raw_bindings, tuple):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {prepared.module_name!r} has an invalid external import index."
        )
    bindings: list[dict[str, str]] = []
    for binding in raw_bindings:
        if not isinstance(binding, Mapping):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} has an invalid external import binding."
            )
        local_name = str(binding.get("local_name", "") or "")
        owner = str(binding.get("module", "") or "")
        attribute = str(binding.get("attribute", "") or "")
        root = owner.split(".", 1)[0].replace("-", "_")
        if root not in declared_roots:
            continue
        if not local_name or not owner:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} has an invalid external import binding."
            )
        bindings.append(
            {
                "local_name": local_name,
                "module": owner,
                "attribute": attribute,
            }
        )
    return sorted(
        bindings,
        key=lambda record: (
            record["local_name"],
            record["module"],
            record["attribute"],
        ),
    )


def _clean_reference_request(
    prepared_modules: Sequence[_PreparedLoadedModule],
    *,
    package_root: Path,
) -> tuple[
    dict[str, Any],
    dict[
        str,
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[str, str, str], ...],
        ],
    ],
]:
    """Build one exact batch request for all loaded modules in this closure."""

    manifest = _source_tree_manifest(package_root)
    module_requests: list[dict[str, Any]] = []
    expected: dict[
        str,
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[str, str, str], ...],
        ],
    ] = {}
    dependencies: set[str] = set()
    for prepared in prepared_modules:
        states, defaults, state_dependencies = _source_state_request(prepared)
        builtin_names = _source_builtin_names(prepared)
        external_bindings = _declared_dependency_import_bindings(
            prepared,
            state_dependencies,
        )
        requires_definition_selection = prepared.reference.get(
            "requires_clean_definition_selection"
        )
        if type(requires_definition_selection) is not bool:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} has an invalid "
                "definition-selection policy."
            )
        if (
            not states
            and not defaults
            and not builtin_names
            and not external_bindings
            and not requires_definition_selection
        ):
            continue
        if prepared.module_name in expected:
            raise CanonicalExecutionOriginError(
                f"Duplicate clean-reference request for {prepared.module_name!r}."
            )
        expected[prepared.module_name] = (
            tuple(states),
            tuple(defaults),
            tuple(builtin_names),
            tuple(
                (item["local_name"], item["module"], item["attribute"])
                for item in external_bindings
            ),
        )
        dependencies.update(state_dependencies)
        module_requests.append(
            {
                "module": prepared.module_name,
                "source_sha256": prepared.source_sha256,
                "definition_selection_sha256": _selected_definition_variants_sha256(
                    prepared.reference
                ),
                "states": states,
                "defaults": defaults,
                "builtins": builtin_names,
                "external_bindings": external_bindings,
            }
        )
    payload = {
        "schema_version": _CLEAN_ISOLATED_REFERENCE_SCHEMA_VERSION,
        "package_root": str(package_root),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_json_sha256(manifest),
        "environment": _clean_reference_environment(),
        "dependencies": sorted(dependencies),
        "modules": module_requests,
    }
    return payload, expected


def _clean_reference_preexec() -> None:
    """Apply bounded child resources before importing optional model packages."""

    max_bytes = int(_CLEAN_ISOLATED_REFERENCE_MAX_RSS_MB) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    except (AttributeError, OSError, ValueError):
        pass
    try:
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (int(_CLEAN_ISOLATED_REFERENCE_TIMEOUT_SEC) + 10,) * 2,
        )
    except (AttributeError, OSError, ValueError):
        pass


_CLEAN_ISOLATED_REFERENCE_CHILD = "\n".join(
    (
        "import hashlib",
        "import io",
        "import json",
        "import os",
        "from pathlib import Path",
        "import sys",
        "",
        "payload = json.loads(sys.stdin.read())",
        "if payload.get('schema_version') != 'tabnetics_clean_isolated_reference_v2':",
        "    raise RuntimeError('unsupported clean isolated-reference schema')",
        "root = Path(str(payload['package_root'])).resolve()",
        "manifest = payload.get('source_manifest')",
        "child_environment = payload.get('child_environment')",
        "isolated_paths = payload.get('isolated_paths')",
        "if not isinstance(manifest, list):",
        "    raise RuntimeError('clean isolated-reference manifest is invalid')",
        "if not isinstance(child_environment, dict) or not isinstance(isolated_paths, dict):",
        "    raise RuntimeError('clean isolated-reference isolation request is invalid')",
        "if any(type(key) is not str or type(value) is not str for key, value in child_environment.items()):",
        "    raise RuntimeError('clean isolated-reference child environment is invalid')",
        "if any(type(key) is not str or type(value) is not str for key, value in isolated_paths.items()):",
        "    raise RuntimeError('clean isolated-reference isolated paths are invalid')",
        "if isolated_paths.get('source_package_root') != 'verified_temp_source_copy':",
        "    raise RuntimeError('clean isolated-reference source copy policy is invalid')",
        "if any(value != f'child_owned:{key}' for key, value in isolated_paths.items() if key != 'source_package_root'):",
        "    raise RuntimeError('clean isolated-reference path policy is invalid')",
        "if any(os.environ.get(key, '') != value for key, value in child_environment.items()):",
        "    raise RuntimeError('clean isolated-reference child environment was not applied')",
        "observed = []",
        "for source in sorted(root.rglob('*.py')):",
        "    if not source.is_file():",
        "        raise RuntimeError('clean isolated-reference source entry is not a file')",
        "    observed.append({'path': source.relative_to(root).as_posix(), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})",
        "encoded = json.dumps(manifest, ensure_ascii=True, separators=(',', ':'), sort_keys=True)",
        "if observed != manifest or hashlib.sha256(encoded.encode('utf-8')).hexdigest() != payload.get('source_manifest_sha256'):",
        "    raise RuntimeError('clean isolated-reference source manifest mismatch')",
        "sys.path.insert(0, str(root.parent))",
        "captured_stdout = io.StringIO()",
        "with __import__('contextlib').redirect_stdout(captured_stdout):",
        "    from tabnetics.validation.core.provenance import _clean_isolated_reference_child_payload",
        "    result = _clean_isolated_reference_child_payload(payload)",
        "if captured_stdout.getvalue():",
        "    raise RuntimeError('clean isolated-reference import wrote to stdout')",
        "for module_name, module in tuple(sys.modules.items()):",
        "    if module_name != 'tabnetics' and not module_name.startswith('tabnetics.'):",
        "        continue",
        "    source_path = Path(str(getattr(module, '__file__', '') or '')).resolve()",
        "    try:",
        "        source_path.relative_to(root)",
        "    except ValueError:",
        "        raise RuntimeError(f'clean isolated-reference imported outside staged source: {module_name}')",
        "sys.__stdout__.write(json.dumps(result, ensure_ascii=True, separators=(',', ':'), sort_keys=True))",
    )
)


def _terminate_clean_reference_process(process: subprocess.Popen[bytes]) -> None:
    """Kill the child process group so timed-out imports cannot outlive attestation."""

    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:  # pragma: no cover - Windows is not exercised in the lab fleet.
        process.kill()


def _run_clean_isolated_reference(
    request: Mapping[str, Any],
    expected: Mapping[
        str,
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[str, str, str], ...],
        ],
    ],
) -> _CleanIsolatedReference | None:
    """Run one sanitized, bounded, source-pinned clean reference subprocess."""

    if not expected:
        return None
    environment = request.get("environment")
    if not isinstance(environment, Mapping) or any(
        type(key) is not str or type(value) is not str
        for key, value in environment.items()
    ):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference environment is invalid."
        )
    with tempfile.TemporaryDirectory(prefix="tabnetics-clean-reference-") as cwd:
        work_dir = Path(cwd).resolve()
        child_package_root = _materialize_verified_clean_source_copy(
            request,
            destination_root=work_dir,
        )
        child_environment, evidence_environment, isolated_paths = (
            _clean_reference_isolated_environment(
                work_dir=work_dir,
                source_package_root=child_package_root,
            )
        )
        child_request = dict(request)
        child_request.update(
            {
                "package_root": str(child_package_root),
                "environment": evidence_environment,
                "child_environment": child_environment,
                "isolated_paths": isolated_paths,
            }
        )
        try:
            serialized_request = json.dumps(
                child_request,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference request is not JSON serializable."
            ) from exc
        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": child_environment,
            "cwd": cwd,
            "start_new_session": True,
        }
        if os.name == "posix":
            popen_kwargs["preexec_fn"] = _clean_reference_preexec
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", _CLEAN_ISOLATED_REFERENCE_CHILD],
            **popen_kwargs,
        )
        try:
            stdout, stderr = process.communicate(
                serialized_request,
                timeout=float(_CLEAN_ISOLATED_REFERENCE_TIMEOUT_SEC),
            )
        except subprocess.TimeoutExpired as exc:
            _terminate_clean_reference_process(process)
            try:
                process.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
                raise CanonicalExecutionOriginError(
                    "Clean isolated-reference subprocess timed out; its process group was killed "
                    "but pipe drain did not complete within 5 seconds."
                ) from exc
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference subprocess timed out and was terminated."
            ) from exc
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference subprocess failed"
            + (f": {detail[-1000:]}" if detail else ".")
        )
    try:
        response = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference subprocess did not emit one valid JSON payload."
        ) from exc
    return _validate_clean_isolated_reference_response(
        child_request, expected, response
    )


def _qualified_runtime_type(value: Any) -> str:
    """Return a stable qualified type identity for class contracts."""

    return (
        f"{_safe_class_text_attribute(value, '__module__')}."
        f"{_safe_class_text_attribute(value, '__qualname__')}"
    )


def _resolve_source_class_expression(
    node: ast.AST,
    module_values: Mapping[str, Any],
) -> Any:
    """Resolve a non-executing class-header expression from verified AST."""

    if isinstance(node, ast.Subscript):
        return _resolve_source_class_expression(node.value, module_values)
    if isinstance(node, ast.Name):
        value = module_values.get(node.id, _MISSING_RUNTIME_BINDING)
        if value is not _MISSING_RUNTIME_BINDING:
            return value
        return _trusted_builtin_value(node.id, _MISSING_RUNTIME_BINDING)
    if isinstance(node, ast.Attribute):
        base = _resolve_source_class_expression(node.value, module_values)
        if base is _MISSING_RUNTIME_BINDING:
            return base
        return getattr(base, node.attr, _MISSING_RUNTIME_BINDING)
    return _MISSING_RUNTIME_BINDING


def _derived_metaclass(bases: Sequence[type]) -> type:
    """Mirror Python's compatible-metaclass selection for a class header."""

    candidate: type = type
    for base in bases:
        base_metaclass = type(base)
        if issubclass(candidate, base_metaclass):
            continue
        if issubclass(base_metaclass, candidate):
            candidate = base_metaclass
            continue
        raise CanonicalExecutionOriginError(
            "Source-derived class bases have incompatible metaclasses."
        )
    return candidate


_MISSING_RUNTIME_BINDING = object()
__tabnetics_execution_isolated_state__ = {
    "CANONICAL_BOOTSTRAP_IMPORT_LABELS": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "MAX_PROVENANCE_HASH_BYTES": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "_CLEAN_ISOLATED_REFERENCE_CHILD": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "_EMPTY_STATE_SERIALIZATION_CONTEXT": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}
__tabnetics_execution_ephemeral_globals__ = (
    "_LRU_CACHE_WRAPPER_TYPE",
    "_MISSING_RUNTIME_BINDING",
)


def _attest_class_structure(
    *,
    target: type,
    name: str,
    module_name: str,
    module_values: Mapping[str, Any],
    source_contract: Mapping[str, Any],
) -> tuple[list[str], str, str]:
    """Match a live class's bases and metaclass to its verified source header."""

    if not isinstance(target, type):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} is not a class object."
        )
    raw_base_nodes = source_contract.get("base_nodes")
    if not isinstance(raw_base_nodes, tuple):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has an invalid "
            "source base contract."
        )
    expected_bases: list[type] = []
    for base_node in raw_base_nodes:
        if not isinstance(base_node, ast.AST):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} class {name!r} has an invalid "
                "source base expression."
            )
        resolved = _resolve_source_class_expression(base_node, module_values)
        if not inspect.isclass(resolved):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} class {name!r} has an "
                "unresolvable source-derived base expression."
            )
        expected_bases.append(resolved)
    if not expected_bases:
        expected_bases = [object]
    # Do this only through ``type.__getattribute__``.  A replacement class can
    # carry a metaclass whose normal attribute access executes attacker code.
    try:
        actual_bases_raw = type.__getattribute__(target, "__bases__")
    except (AttributeError, TypeError) as exc:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has invalid bases."
        ) from exc
    if type(actual_bases_raw) is not tuple or not all(
        isinstance(base, type) for base in actual_bases_raw
    ):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has invalid bases."
        )
    actual_bases = actual_bases_raw
    if actual_bases != tuple(expected_bases):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} bases do not match "
            "the verified source header."
        )

    metaclass_node = source_contract.get("metaclass_node")
    if metaclass_node is None:
        expected_metaclass = _derived_metaclass(expected_bases)
    elif isinstance(metaclass_node, ast.AST):
        expected_metaclass = _resolve_source_class_expression(
            metaclass_node,
            module_values,
        )
        if not inspect.isclass(expected_metaclass):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} class {name!r} has an "
                "unresolvable source-derived metaclass expression."
            )
    else:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has an invalid "
            "source metaclass contract."
        )
    actual_metaclass = type(target)
    if actual_metaclass is not expected_metaclass:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} metaclass does not "
            "match the verified source header."
        )
    source_contract_sha256 = str(source_contract.get("source_sha256", "") or "")
    if not _is_sha256_digest(source_contract_sha256):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has an invalid "
            "source base contract digest."
        )
    return (
        [_qualified_runtime_type(base) for base in actual_bases],
        _qualified_runtime_type(actual_metaclass),
        source_contract_sha256,
    )


def _attest_callable_default_state(
    target: Any,
    *,
    module_name: str,
    label: str,
    source_contract: Mapping[str, Any],
    state_context: _StateSerializationContext,
    clean_reference: _CleanIsolatedReference,
    class_owner: type | None,
) -> str:
    """Bind defaults to a clean source-pinned process without evaluating source."""

    raw_positional = source_contract.get("positional")
    raw_keyword = source_contract.get("keyword")
    if not isinstance(raw_positional, tuple) or not isinstance(raw_keyword, tuple):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has an invalid "
            "source callable default contract."
        )
    if any(not isinstance(expression, ast.AST) for expression in raw_positional):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has an invalid "
            "source positional default expression."
        )
    expected_keyword_names: set[str] = set()
    for item in raw_keyword:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or type(item[0]) is not str
            or not isinstance(item[1], ast.AST)
        ):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} symbol {label!r} has an "
                "invalid source keyword default expression."
            )
        expected_keyword_names.add(item[0])
    if type(target) is not FunctionType:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} is not an exact function."
        )
    actual_positional_raw = target.__defaults__
    actual_positional = () if actual_positional_raw is None else actual_positional_raw
    actual_keyword_raw = target.__kwdefaults__
    actual_keyword = {} if actual_keyword_raw is None else actual_keyword_raw
    if type(actual_positional) is not tuple or type(actual_keyword) is not dict:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has invalid "
            "runtime keyword defaults."
        )
    actual_keyword_names = tuple(dict.keys(actual_keyword))
    if (
        len(actual_positional) != len(raw_positional)
        or any(type(key) is not str for key in actual_keyword_names)
        or set(actual_keyword_names) != expected_keyword_names
    ):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} runtime "
            "callable state/defaults do not match verified source."
        )
    try:
        actual_positional_payload = [
            _state_value_payload(value, context=state_context)
            for value in tuple.__iter__(actual_positional)
        ]
        actual_keyword_payload = {
            key: _state_value_payload(value, context=state_context)
            for key, value in sorted(
                dict.items(actual_keyword), key=lambda item: item[0]
            )
        }
    except _ExecutionStateUnsupported as exc:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has unsupported "
            "runtime callable default state."
        ) from exc
    expected_payload = clean_reference.default_payload(module_name, label)
    if set(expected_payload) != {"positional", "keyword"}:
        raise CanonicalExecutionOriginError(
            "Clean isolated reference has an invalid callable-default payload for "
            f"{module_name!r}:{label!r}."
        )
    expected_positional_payload = expected_payload["positional"]
    expected_keyword_payload = expected_payload["keyword"]
    if (
        not isinstance(expected_positional_payload, list)
        or not isinstance(expected_keyword_payload, dict)
        or expected_positional_payload != actual_positional_payload
        or expected_keyword_payload != actual_keyword_payload
    ):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} runtime "
            "callable state/defaults do not match verified source."
        )

    freevars = tuple(str(name) for name in target.__code__.co_freevars)
    closure = tuple(getattr(target, "__closure__", None) or ())
    if len(freevars) != len(closure):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has invalid "
            "runtime closure state."
        )
    closure_payload: list[dict[str, Any]] = []
    for freevar, cell in zip(freevars, closure):
        try:
            value = cell.cell_contents
        except ValueError as exc:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} symbol {label!r} has an empty "
                "runtime closure cell."
            ) from exc
        if freevar != "__class__" or class_owner is None or value is not class_owner:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} symbol {label!r} has an "
                "unverified runtime closure cell."
            )
        closure_payload.append(
            {
                "name": freevar,
                "value": {"kind": "class", "value": _qualified_runtime_type(value)},
            }
        )
    return canonical_json_sha256(
        {
            "positional": actual_positional_payload,
            "keyword": actual_keyword_payload,
            "closure": closure_payload,
        }
    )


def _unwrap_exact_function(value: Any, *, module_name: str, label: str) -> FunctionType:
    """Unwrap only exact functions without invoking arbitrary descriptors."""

    target = value
    seen: set[int] = set()
    while True:
        if type(target) is _LRU_CACHE_WRAPPER_TYPE:
            target = target.__wrapped__
            continue
        if type(target) is not FunctionType:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} symbol {label!r} is not an exact function."
            )
        target_id = id(target)
        if target_id in seen:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} symbol {label!r} has cyclic wrapping."
            )
        seen.add(target_id)
        target_dict = target.__dict__
        if type(target_dict) is not dict:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} symbol {label!r} has invalid function state."
            )
        wrapped = dict.get(target_dict, "__wrapped__", _MISSING_RUNTIME_BINDING)
        if wrapped is _MISSING_RUNTIME_BINDING:
            return target
        target = wrapped


def _attested_callable_code_origin(
    callable_obj: Any,
    *,
    module_name: str,
    source: Path,
    module_globals: Mapping[str, Any],
    qualname: str,
    expected_hashes: Sequence[str],
    label: str,
) -> tuple[FunctionType, str]:
    """Validate code/origin before any runtime state is serialized."""

    target = _unwrap_exact_function(callable_obj, module_name=module_name, label=label)
    target_module = target.__module__
    if type(target_module) is not str or target_module != module_name:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has a foreign owner."
        )
    if target.__globals__ is not module_globals:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} does not use "
            "the live verified module globals."
        )
    target_qualname = target.__qualname__
    if type(target_qualname) is not str or target_qualname != qualname:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has an unexpected "
            "qualified name."
        )
    code = target.__code__
    if type(code) is not CodeType:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has no code object."
        )
    code_path_raw = code.co_filename
    if not code_path_raw or code_path_raw.startswith("<"):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has a synthetic code origin."
        )
    if Path(code_path_raw).resolve() != source:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} code originates "
            "outside its verified source."
        )
    digest = _code_sha256(code)
    if digest not in set(expected_hashes):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} code does not match "
            "the independently compiled verified source."
        )
    return target, digest


def _attested_callable_code_sha256(
    callable_obj: Any,
    *,
    module_name: str,
    source: Path,
    module_globals: Mapping[str, Any],
    qualname: str,
    expected_hashes: Sequence[str],
    label: str,
    source_default_contract: Mapping[str, Any],
    state_context: _StateSerializationContext,
    clean_reference: _CleanIsolatedReference,
    class_owner: type | None = None,
) -> tuple[str, str]:
    """Validate one callable's origin, code, and clean-reference defaults."""

    target, digest = _attested_callable_code_origin(
        callable_obj,
        module_name=module_name,
        source=source,
        module_globals=module_globals,
        qualname=qualname,
        expected_hashes=expected_hashes,
        label=label,
    )
    default_state_sha256 = _attest_callable_default_state(
        target,
        module_name=module_name,
        label=label,
        source_contract=source_default_contract,
        state_context=state_context,
        clean_reference=clean_reference,
        class_owner=class_owner,
    )
    return digest, default_state_sha256


def _descriptor_callables(descriptor: Any) -> Sequence[Any]:
    """Return authored callables exposed by common class descriptors."""

    descriptor_type = type(descriptor)
    if descriptor_type is staticmethod or descriptor_type is classmethod:
        return (descriptor.__func__,)
    if descriptor_type is property:
        return tuple(
            member
            for member in (descriptor.fget, descriptor.fset, descriptor.fdel)
            if member is not None
        )
    if descriptor_type is functools.cached_property:
        return (descriptor.func,)
    if descriptor_type is FunctionType:
        return (descriptor,)
    return ()


def _validate_class_code_origin(
    *,
    name: str,
    target: type,
    module_name: str,
    source: Path,
    module_values: Mapping[str, Any],
    reference: Mapping[str, Any],
    expected_member_qualnames: Sequence[str],
    source_base_contract: Mapping[str, Any],
) -> None:
    """Validate class bases and authored member code without touching state."""

    _attest_class_structure(
        target=target,
        name=name,
        module_name=module_name,
        module_values=module_values,
        source_contract=source_base_contract,
    )
    try:
        owner = _safe_class_text_attribute(target, "__module__")
        qualname = _safe_class_text_attribute(target, "__qualname__")
    except _ExecutionStateUnsupported as exc:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has invalid metadata."
        ) from exc
    if owner != module_name:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has a foreign owner."
        )
    if qualname != name:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has an unexpected "
            "qualified name."
        )
    expected_counts = Counter(str(item) for item in expected_member_qualnames)
    expected_members = set(expected_counts)
    observed_counts: Counter[str] = Counter()
    try:
        class_dict = _safe_class_dict(target)
    except _ExecutionStateUnsupported as exc:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has invalid class state."
        ) from exc
    for member_name, descriptor in sorted(
        dict.items(_mappingproxy_backing_dict(class_dict)),
        key=lambda item: str(item[0]),
    ):
        for member in _descriptor_callables(descriptor):
            unwrapped = _unwrap_exact_function(
                member,
                module_name=module_name,
                label=f"{name}.{member_name}",
            )
            member_qualname = unwrapped.__qualname__
            owner = unwrapped.__module__
            generated_dataclass_method = bool(
                dataclasses.is_dataclass(target)
                and str(member_name) in _DATACLASS_GENERATED_METHOD_NAMES
                and member_qualname not in expected_members
            )
            generated_enum_member = bool(
                issubclass(target, Enum)
                and owner == "enum"
                and member_qualname.startswith("Enum.")
            )
            if generated_dataclass_method or generated_enum_member:
                continue
            if owner != module_name or member_qualname not in expected_members:
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} class {name!r} has an "
                    f"unrecognized local member {member_name!r}."
                )
            _attested_callable_code_origin(
                unwrapped,
                module_name=module_name,
                source=source,
                module_globals=module_values,
                qualname=member_qualname,
                expected_hashes=_reference_code_hashes(
                    reference,
                    member_qualname,
                    module_name=module_name,
                ),
                label=f"{name}.{member_name}",
            )
            observed_counts[member_qualname] += 1
    if observed_counts != expected_counts:
        missing = sorted((expected_counts - observed_counts).elements())
        unexpected = sorted((observed_counts - expected_counts).elements())
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} direct member set "
            f"does not match independently compiled source: missing={missing!r} "
            f"unexpected={unexpected!r}."
        )


def _selected_definition_variants_sha256(reference: Mapping[str, Any]) -> str:
    """Hash the exact source variants selected for one live module import."""

    raw_selection = reference.get("selected_definition_variants")
    if not isinstance(raw_selection, tuple) or not all(
        isinstance(record, Mapping)
        and set(record) == {"name", "kind", "variant_sha256"}
        and type(record.get("name")) is str
        and record.get("kind") in {"function", "class"}
        and _is_sha256_digest(record.get("variant_sha256"))
        for record in raw_selection
    ):
        raise CanonicalExecutionOriginError(
            "Loaded source has an invalid selected definition-variant index."
        )
    return canonical_json_sha256(raw_selection)


def _resolve_source_symbol_reference(
    reference: Mapping[str, Any],
    *,
    module_name: str,
    module: ModuleType,
    source: Path,
) -> dict[str, Any]:
    """Select exact module-scope source variants matching a live import.

    Selection is code/origin-only.  It never executes a candidate callable or
    reads arbitrary module state.  Guarded definitions are subsequently bound
    to the variant set chosen by a clean isolated import, preventing a caller
    from substituting an inactive-but-source-authored branch.
    """

    raw_groups = reference.get("definition_groups")
    imported_binding_names = reference.get("imported_binding_names")
    if not isinstance(raw_groups, tuple) or not isinstance(
        imported_binding_names,
        frozenset,
    ):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid source definition group index."
        )
    module_values = vars(module)
    selected_definitions: list[dict[str, Any]] = []
    selected_code_hashes: dict[str, tuple[str, ...]] = {}
    selected_defaults: dict[str, Mapping[str, Any]] = {}
    selected_state_references: set[str] = set()
    selected_variants: list[dict[str, str]] = []
    source_names: set[str] = set()
    missing = _MISSING_RUNTIME_BINDING

    for group in raw_groups:
        if not isinstance(group, Mapping) or set(group) != {
            "name",
            "required",
            "conditional",
            "variants",
        }:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid source definition group."
            )
        name = group.get("name")
        required = group.get("required")
        conditional = group.get("conditional")
        variants = group.get("variants")
        if (
            type(name) is not str
            or not name
            or type(required) is not bool
            or type(conditional) is not bool
            or not isinstance(variants, tuple)
            or not variants
        ):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has a malformed source definition group."
            )
        source_names.add(name)
        target = module_values.get(name, missing)
        if target is missing:
            if required:
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} is missing source symbol {name!r}."
                )
            continue
        if (
            not required
            and name in imported_binding_names
            and str(getattr(target, "__module__", "") or "") != module_name
        ):
            # A guarded fallback definition shares the imported alias.  When
            # the import succeeded, its exact owner is attested by the import
            # and dependency phases; no local source variant is active.
            continue

        matches: list[Mapping[str, Any]] = []
        candidate_errors: list[CanonicalExecutionOriginError] = []
        for variant in variants:
            if not isinstance(variant, Mapping):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} has an invalid source definition variant."
                )
            kind = variant.get("kind")
            variant_name = variant.get("name")
            qualname = variant.get("qualname")
            variant_sha256 = variant.get("variant_sha256")
            raw_hashes = variant.get("code_hashes_by_qualname")
            if (
                variant_name != name
                or kind not in {"function", "class"}
                or qualname != name
                or not _is_sha256_digest(variant_sha256)
                or not isinstance(raw_hashes, Mapping)
            ):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} has a malformed source definition variant."
                )
            candidate_reference = dict(reference)
            candidate_reference["code_hashes_by_qualname"] = raw_hashes
            try:
                if kind == "function":
                    _attested_callable_code_origin(
                        target,
                        module_name=module_name,
                        source=source,
                        module_globals=module_values,
                        qualname=name,
                        expected_hashes=_reference_code_hashes(
                            candidate_reference,
                            name,
                            module_name=module_name,
                        ),
                        label=name,
                    )
                else:
                    if not isinstance(target, type):
                        continue
                    raw_members = variant.get("member_qualnames")
                    raw_base_contract = variant.get("class_base_contract")
                    if not isinstance(raw_members, tuple) or not isinstance(
                        raw_base_contract,
                        Mapping,
                    ):
                        raise CanonicalExecutionOriginError(
                            f"Loaded tabnetics module {module_name!r} class {name!r} has "
                            "an invalid source variant contract."
                        )
                    _validate_class_code_origin(
                        name=name,
                        target=target,
                        module_name=module_name,
                        source=source,
                        module_values=module_values,
                        reference=candidate_reference,
                        expected_member_qualnames=raw_members,
                        source_base_contract=raw_base_contract,
                    )
            except CanonicalExecutionOriginError as exc:
                candidate_errors.append(exc)
                continue
            matches.append(variant)

        if len(matches) != 1:
            if not matches and len(variants) == 1 and len(candidate_errors) == 1:
                # Preserve the established fail-closed diagnostic for ordinary
                # (non-alternative) symbols and its no-dispatch test coverage.
                raise candidate_errors[0]
            reason = "no" if not matches else "multiple"
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} symbol {name!r} matches {reason} "
                "module-scope source definition variant."
            )
        selected = matches[0]
        selected_kind = str(selected["kind"])
        selected_variant_sha256 = str(selected["variant_sha256"])
        selected_definitions.append(
            {
                "name": name,
                "kind": selected_kind,
                "qualname": name,
                "member_qualnames": selected.get("member_qualnames"),
                "class_base_contract": selected.get("class_base_contract"),
                "class_static_bindings": selected.get("class_static_bindings"),
                "variant_sha256": selected_variant_sha256,
            }
        )
        selected_variants.append(
            {
                "name": name,
                "kind": selected_kind,
                "variant_sha256": selected_variant_sha256,
            }
        )
        raw_hashes = selected.get("code_hashes_by_qualname")
        if not isinstance(raw_hashes, Mapping):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} selected source hashes are invalid."
            )
        for selected_qualname, hashes in raw_hashes.items():
            if type(selected_qualname) is not str or not isinstance(hashes, tuple):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} selected source hashes are malformed."
                )
            existing_hashes = selected_code_hashes.get(selected_qualname)
            if existing_hashes is not None and existing_hashes != hashes:
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} selected source code is ambiguous."
                )
            selected_code_hashes[selected_qualname] = hashes
        raw_defaults = selected.get("callable_defaults")
        raw_references = selected.get("module_state_references")
        if not isinstance(raw_defaults, Mapping) or not isinstance(
            raw_references, tuple
        ):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} selected source state is invalid."
            )
        for default_qualname, contract in raw_defaults.items():
            if type(default_qualname) is not str or not isinstance(contract, Mapping):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} selected callable defaults are invalid."
                )
            if default_qualname in selected_defaults:
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} selected callable defaults are ambiguous."
                )
            selected_defaults[default_qualname] = contract
        if any(type(item) is not str for item in raw_references):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} selected state references are invalid."
            )
        selected_state_references.update(raw_references)

    # Preserve the existing fail-closed rule for dynamically exposed locals.
    # A nested source function is not a module-scope definition and therefore
    # remains an injected symbol if rebound into the module namespace.
    for name, target in sorted(module_values.items(), key=lambda item: str(item[0])):
        if str(name) in source_names:
            continue
        if not (inspect.isfunction(target) or inspect.isclass(target)):
            continue
        if str(getattr(target, "__module__", "") or "") != module_name:
            continue
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} exposes an unrecognized local "
            f"symbol {str(name)!r}."
        )

    resolved = dict(reference)
    resolved["definitions"] = tuple(
        sorted(
            selected_definitions,
            key=lambda record: (str(record["name"]), str(record["kind"])),
        )
    )
    resolved["code_hashes_by_qualname"] = {
        qualname: selected_code_hashes[qualname]
        for qualname in sorted(selected_code_hashes)
    }
    resolved["callable_defaults"] = {
        qualname: selected_defaults[qualname] for qualname in sorted(selected_defaults)
    }
    resolved["module_state_references"] = tuple(sorted(selected_state_references))
    resolved["selected_definition_variants"] = tuple(
        sorted(
            selected_variants,
            key=lambda record: (
                record["name"],
                record["kind"],
                record["variant_sha256"],
            ),
        )
    )
    _selected_definition_variants_sha256(resolved)
    return resolved


def _validate_loaded_module_code_origin(prepared: _PreparedLoadedModule) -> None:
    """Perform the mandatory code/origin-only phase for one loaded module."""

    raw_definitions = prepared.reference.get("definitions")
    if not isinstance(raw_definitions, tuple):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {prepared.module_name!r} has an invalid source definition index."
        )
    module_values = vars(prepared.module)
    missing = _MISSING_RUNTIME_BINDING
    for definition in raw_definitions:
        if not isinstance(definition, Mapping):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} has an invalid source definition."
            )
        name = str(definition.get("name", "") or "")
        kind = str(definition.get("kind", "") or "")
        if not name or kind not in {"function", "class"}:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} has an invalid source symbol."
            )
        target = module_values.get(name, missing)
        if target is missing:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} is missing source symbol {name!r}."
            )
        if kind == "function":
            _attested_callable_code_origin(
                target,
                module_name=prepared.module_name,
                source=prepared.source,
                module_globals=module_values,
                qualname=name,
                expected_hashes=_reference_code_hashes(
                    prepared.reference,
                    name,
                    module_name=prepared.module_name,
                ),
                label=name,
            )
            continue
        if not isinstance(target, type):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} symbol {name!r} is not a class."
            )
        raw_members = definition.get("member_qualnames")
        raw_base_contract = definition.get("class_base_contract")
        if not isinstance(raw_members, tuple) or not isinstance(
            raw_base_contract, Mapping
        ):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} class {name!r} has an invalid source contract."
            )
        _validate_class_code_origin(
            name=name,
            target=target,
            module_name=prepared.module_name,
            source=prepared.source,
            module_values=module_values,
            reference=prepared.reference,
            expected_member_qualnames=raw_members,
            source_base_contract=raw_base_contract,
        )


def _state_identity_payload(value: Any) -> dict[str, str]:
    """Construct a source-symbol identity after code/origin validation."""

    return {
        "module": str(getattr(value, "__module__", "") or ""),
        "qualname": str(getattr(value, "__qualname__", "") or ""),
    }


def _verified_dataclass_factory_functions(
    prepared: _PreparedLoadedModule,
    *,
    owner_name: str,
    owner: type,
    module_values: Mapping[str, Any],
) -> list[tuple[FunctionType, Mapping[str, str]]]:
    """Return source-verified exact function default factories for one class."""

    if not dataclasses.is_dataclass(owner):
        return []
    try:
        class_values = _mappingproxy_backing_dict(_safe_class_dict(owner))
        raw_fields = dict.get(class_values, "__dataclass_fields__")
    except _ExecutionStateUnsupported as exc:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {prepared.module_name!r} dataclass {owner_name!r} "
            "has invalid field state."
        ) from exc
    if type(raw_fields) is not dict:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {prepared.module_name!r} dataclass {owner_name!r} "
            "has an invalid field table."
        )
    records: list[tuple[FunctionType, Mapping[str, str]]] = []
    for field_name, field_value in sorted(
        dict.items(raw_fields), key=lambda item: str(item[0])
    ):
        if type(field_name) is not str or type(field_value) is not dataclasses.Field:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} dataclass {owner_name!r} "
                "has a malformed field table."
            )
        factory = object.__getattribute__(field_value, "default_factory")
        if type(factory) is not FunctionType:
            continue
        factory_qualname = factory.__qualname__
        if type(factory_qualname) is not str or not factory_qualname:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} dataclass {owner_name!r} "
                f"field {field_name!r} has an invalid function factory."
            )
        _, digest = _attested_callable_code_origin(
            factory,
            module_name=prepared.module_name,
            source=prepared.source,
            module_globals=module_values,
            qualname=factory_qualname,
            expected_hashes=_reference_code_hashes(
                prepared.reference,
                factory_qualname,
                module_name=prepared.module_name,
            ),
            label=f"{owner_name}.{field_name}.default_factory",
        )
        records.append(
            (
                factory,
                {
                    "module": prepared.module_name,
                    "qualname": factory_qualname,
                    "code_sha256": digest,
                },
            )
        )
    return records


def _build_state_serialization_context(
    prepared_modules: Sequence[_PreparedLoadedModule],
    *,
    package_root: Path | None = None,
    repository_root: Path | None = None,
) -> _StateSerializationContext:
    """Admit exact source-defined identities after the global code phase."""

    functions: dict[int, tuple[Any, Mapping[str, str]]] = {}
    dataclass_factory_functions: dict[int, tuple[FunctionType, Mapping[str, str]]] = {}
    classes: dict[int, tuple[Any, Mapping[str, str]]] = {}
    modules: dict[int, tuple[ModuleType, Mapping[str, str]]] = {}
    dataclass_fields: dict[int, tuple[type, tuple[str, ...], Mapping[str, str]]] = {}
    enum_types: dict[int, tuple[type, Mapping[str, str]]] = {}
    for prepared in prepared_modules:
        modules[id(prepared.module)] = (
            prepared.module,
            # The isolated reference imports a verified staging copy, so an
            # absolute source path is intentionally different in the parent
            # and child. Bind the module identity to its logical name and the
            # source bytes that passed the code-origin phase instead.
            {
                "name": prepared.module_name,
                "source_sha256": prepared.source_sha256,
            },
        )
        raw_definitions = prepared.reference.get("definitions")
        if not isinstance(raw_definitions, tuple):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} has an invalid source definition index."
            )
        module_values = vars(prepared.module)
        for definition in raw_definitions:
            if not isinstance(definition, Mapping):
                continue
            name = str(definition.get("name", "") or "")
            kind = str(definition.get("kind", "") or "")
            value = module_values.get(name, _MISSING_RUNTIME_BINDING)
            if value is _MISSING_RUNTIME_BINDING:
                continue
            if kind == "function" and type(value) is FunctionType:
                functions[id(value)] = (
                    value,
                    {
                        **_state_identity_payload(value),
                        "code_sha256": _code_sha256(value.__code__),
                    },
                )
            elif kind == "class" and inspect.isclass(value):
                payload = _state_identity_payload(value)
                classes[id(value)] = (value, payload)
                if dataclasses.is_dataclass(value):
                    fields = tuple(
                        str(field.name) for field in dataclasses.fields(value)
                    )
                    dataclass_fields[id(value)] = (value, fields, payload)
                if issubclass(value, Enum):
                    enum_types[id(value)] = (value, payload)
    for prepared in prepared_modules:
        raw_definitions = prepared.reference.get("definitions")
        if not isinstance(raw_definitions, tuple):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} has an invalid source definition index."
            )
        module_values = vars(prepared.module)
        for definition in raw_definitions:
            if not isinstance(definition, Mapping) or definition.get("kind") != "class":
                continue
            name = str(definition.get("name", "") or "")
            owner = module_values.get(name, _MISSING_RUNTIME_BINDING)
            if not isinstance(owner, type):
                continue
            for factory, payload in _verified_dataclass_factory_functions(
                prepared,
                owner_name=name,
                owner=owner,
                module_values=module_values,
            ):
                existing = dataclass_factory_functions.get(id(factory))
                if existing is not None and existing[0] is not factory:
                    raise CanonicalExecutionOriginError(
                        "Dataclass default-factory identity collision."
                    )
                dataclass_factory_functions[id(factory)] = (factory, payload)
    pandas_na_type: type | None = None
    pandas_module = sys.modules.get("pandas")
    if type(pandas_module) is ModuleType:
        pandas_na = vars(pandas_module).get("NA", _MISSING_RUNTIME_BINDING)
        if pandas_na is not _MISSING_RUNTIME_BINDING:
            pandas_na_type = type(pandas_na)
    ndarray_type: type | None = None
    numpy_module = sys.modules.get("numpy")
    if type(numpy_module) is ModuleType:
        candidate = vars(numpy_module).get("ndarray", _MISSING_RUNTIME_BINDING)
        if inspect.isclass(candidate):
            ndarray_type = candidate
    return _StateSerializationContext(
        functions=functions,
        dataclass_factory_functions=dataclass_factory_functions,
        classes=classes,
        modules=modules,
        dataclass_fields=dataclass_fields,
        enum_types=enum_types,
        package_root=None if package_root is None else package_root.resolve(),
        repository_root=(
            None if repository_root is None else repository_root.resolve()
        ),
        pandas_na_type=pandas_na_type,
        ndarray_type=ndarray_type,
        regex_type=type(re.compile("")),
    )


def _attest_callable_builtins(
    target: FunctionType,
    *,
    module_name: str,
    label: str,
    module_values: Mapping[str, Any],
    builtin_names: Sequence[str],
    clean_reference: _CleanIsolatedReference,
) -> None:
    """Seal lexical builtins against both a clean child and local builtins."""

    builtin_mapping = target.__builtins__
    if type(builtin_mapping) is not dict:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has non-plain builtins."
        )
    for builtin_name in builtin_names:
        # Any post-import module global with a source-lexical builtin name
        # changes Python lookup precedence and is therefore a hard rejection.
        if builtin_name in module_values:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} shadows builtin {builtin_name!r}."
            )
        expected = _trusted_builtin_value(builtin_name, _MISSING_RUNTIME_BINDING)
        actual = dict.get(builtin_mapping, builtin_name, _MISSING_RUNTIME_BINDING)
        if expected is _MISSING_RUNTIME_BINDING or actual is not expected:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} symbol {label!r} builtin "
                f"{builtin_name!r} does not match the process builtin table."
            )
        if _builtin_identity_payload(expected) != clean_reference.builtin_payload(
            module_name,
            builtin_name,
        ):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} builtin {builtin_name!r} "
                "does not match the clean isolated reference."
            )


def _attest_module_builtin_state(
    prepared: _PreparedLoadedModule,
    *,
    clean_reference: _CleanIsolatedReference,
) -> None:
    """Apply the bounded builtin seal to all direct authored callables."""

    builtin_names = _source_builtin_names(prepared)
    if not builtin_names:
        return
    raw_definitions = prepared.reference.get("definitions")
    if not isinstance(raw_definitions, tuple):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {prepared.module_name!r} has an invalid source definition index."
        )
    module_values = vars(prepared.module)
    for definition in raw_definitions:
        if not isinstance(definition, Mapping):
            continue
        name = str(definition.get("name", "") or "")
        kind = str(definition.get("kind", "") or "")
        target = module_values.get(name, _MISSING_RUNTIME_BINDING)
        if target is _MISSING_RUNTIME_BINDING:
            continue
        if kind == "function":
            _attest_callable_builtins(
                _unwrap_exact_function(
                    target,
                    module_name=prepared.module_name,
                    label=name,
                ),
                module_name=prepared.module_name,
                label=name,
                module_values=module_values,
                builtin_names=builtin_names,
                clean_reference=clean_reference,
            )
            continue
        if kind != "class" or not inspect.isclass(target):
            continue
        for member_name, descriptor in vars(target).items():
            for member in _descriptor_callables(descriptor):
                _attest_callable_builtins(
                    _unwrap_exact_function(
                        member,
                        module_name=prepared.module_name,
                        label=f"{name}.{member_name}",
                    ),
                    module_name=prepared.module_name,
                    label=f"{name}.{member_name}",
                    module_values=module_values,
                    builtin_names=builtin_names,
                    clean_reference=clean_reference,
                )


def _attest_module_declared_dependency_bindings(
    prepared: _PreparedLoadedModule,
    *,
    clean_reference: _CleanIsolatedReference,
) -> None:
    """Bind declared optional-dependency aliases to clean and live owners."""

    _, _, dependencies = _source_state_request(prepared)
    bindings = _declared_dependency_import_bindings(prepared, dependencies)
    if not bindings:
        return
    clean_module = clean_reference.module_payloads.get(prepared.module_name)
    external_payloads = (
        clean_module.get("external_bindings")
        if isinstance(clean_module, Mapping)
        else None
    )
    if not isinstance(external_payloads, list):
        raise CanonicalExecutionOriginError(
            f"Clean isolated reference is missing external bindings for {prepared.module_name!r}."
        )
    payload_by_binding: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for payload in external_payloads:
        if not isinstance(payload, Mapping):
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference external binding is invalid."
            )
        key = (
            str(payload.get("local_name", "") or ""),
            str(payload.get("owner_module", "") or ""),
            str(payload.get("attribute", "") or ""),
        )
        payload_by_binding[key] = payload  # Shape was validated before this phase.
    module_values = vars(prepared.module)
    for binding in bindings:
        local_name = binding["local_name"]
        owner_name = binding["module"]
        attribute = binding["attribute"]
        key = (local_name, owner_name, attribute)
        clean_payload = payload_by_binding.get(key)
        if clean_payload is None:
            raise CanonicalExecutionOriginError(
                f"Clean isolated reference is missing dependency binding {key!r}."
            )
        actual = module_values.get(local_name, _MISSING_RUNTIME_BINDING)
        if clean_payload.get("kind") == "missing":
            if actual is not _MISSING_RUNTIME_BINDING:
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {prepared.module_name!r} unexpectedly binds dependency {local_name!r}."
                )
            continue
        if actual is _MISSING_RUNTIME_BINDING:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} is missing dependency binding {local_name!r}."
            )
        if clean_payload.get("kind") == "none":
            if actual is not None:
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {prepared.module_name!r} dependency binding {local_name!r} "
                    "does not match the clean optional-import result."
                )
            continue
        try:
            owner_module = importlib.import_module(owner_name)
        except Exception as exc:
            raise CanonicalExecutionOriginError(
                f"Loaded dependency owner {owner_name!r} cannot be imported for {local_name!r}."
            ) from exc
        expected = (
            owner_module
            if not attribute
            else vars(owner_module).get(attribute, _MISSING_RUNTIME_BINDING)
        )
        if expected is _MISSING_RUNTIME_BINDING or actual is not expected:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} dependency binding {local_name!r} "
                "does not match its live declared owner."
            )
        expected_identity = {
            "kind": str(clean_payload.get("kind", "") or ""),
            "module": str(clean_payload.get("target_module", "") or ""),
            "qualname": str(clean_payload.get("qualname", "") or ""),
            "structure_sha256": str(clean_payload.get("structure_sha256", "") or ""),
        }
        if _external_binding_identity(actual) != expected_identity:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {prepared.module_name!r} dependency binding {local_name!r} "
                "does not match the clean isolated reference."
            )


def _exact_function_default_payload(
    target: FunctionType,
    *,
    state_context: _StateSerializationContext,
) -> dict[str, Any]:
    """Serialize one exact function's defaults after type admission."""

    positional_raw = target.__defaults__
    keyword_raw = target.__kwdefaults__
    positional = () if positional_raw is None else positional_raw
    keyword = {} if keyword_raw is None else keyword_raw
    if type(positional) is not tuple or type(keyword) is not dict:
        raise _ExecutionStateUnsupported("function defaults are not plain built-ins")
    keys = tuple(dict.keys(keyword))
    if any(type(key) is not str for key in keys):
        raise _ExecutionStateUnsupported(
            "function keyword defaults have non-string names"
        )
    return {
        "positional": [
            _state_value_payload(value, context=state_context)
            for value in tuple.__iter__(positional)
        ],
        "keyword": {
            key: _state_value_payload(value, context=state_context)
            for key, value in sorted(dict.items(keyword), key=lambda item: item[0])
        },
    }


def _exact_function_closure_payload(
    target: FunctionType,
    *,
    state_context: _StateSerializationContext,
) -> list[dict[str, Any]]:
    """Serialize exact function closure cells without invoking their contents."""

    if type(target) is not FunctionType:
        raise _ExecutionStateUnsupported("generated callable is not an exact function")
    code = target.__code__
    if type(code) is not CodeType:
        raise _ExecutionStateUnsupported("generated callable has no exact code object")
    freevars = tuple(str(name) for name in code.co_freevars)
    closure = target.__closure__
    cells = () if closure is None else closure
    if type(cells) is not tuple or len(cells) != len(freevars):
        raise _ExecutionStateUnsupported("generated callable closure shape is invalid")
    payload: list[dict[str, Any]] = []
    for freevar, cell in zip(freevars, cells):
        try:
            value = cell.cell_contents
        except ValueError as exc:
            raise _ExecutionStateUnsupported(
                "generated callable has an empty closure cell"
            ) from exc
        payload.append(
            {
                "name": freevar,
                "value": _state_value_payload(value, context=state_context),
            }
        )
    return payload


def _dataclass_constructor_fields_payload(
    owner: type,
    *,
    state_context: _StateSerializationContext,
) -> list[dict[str, Any]]:
    """Serialize the constructor-relevant exact dataclass field schema."""

    if not dataclasses.is_dataclass(owner):
        raise _ExecutionStateUnsupported(
            "generated constructor owner is not a dataclass"
        )
    try:
        raw_fields = type.__getattribute__(owner, "__dataclass_fields__")
    except (AttributeError, TypeError) as exc:
        raise _ExecutionStateUnsupported(
            "dataclass field table is unavailable"
        ) from exc
    if type(raw_fields) is not dict:
        raise _ExecutionStateUnsupported("dataclass field table is not an exact dict")
    records: list[dict[str, Any]] = []
    for field_name, field_value in sorted(
        dict.items(raw_fields), key=lambda item: str(item[0])
    ):
        if type(field_name) is not str or type(field_value) is not dataclasses.Field:
            raise _ExecutionStateUnsupported("dataclass field table is malformed")
        actual_name = object.__getattribute__(field_value, "name")
        init = object.__getattribute__(field_value, "init")
        kw_only = object.__getattribute__(field_value, "kw_only")
        if (
            actual_name != field_name
            or type(init) is not bool
            or type(kw_only) is not bool
        ):
            raise _ExecutionStateUnsupported("dataclass field contract is malformed")
        records.append(
            {
                "name": field_name,
                "init": init,
                "kw_only": kw_only,
                "default": _state_value_payload(
                    object.__getattribute__(field_value, "default"),
                    context=state_context,
                ),
                "default_factory": _state_value_payload(
                    object.__getattribute__(field_value, "default_factory"),
                    context=state_context,
                ),
            }
        )
    return records


def _generated_dataclass_init_payload(
    target: FunctionType,
    *,
    owner: type,
    state_context: _StateSerializationContext,
) -> dict[str, Any]:
    """Return a clean-comparable payload for a generated dataclass ``__init__``."""

    if type(target) is not FunctionType:
        raise _ExecutionStateUnsupported(
            "generated dataclass init is not an exact function"
        )
    code = target.__code__
    if type(code) is not CodeType:
        raise _ExecutionStateUnsupported(
            "generated dataclass init has no exact code object"
        )
    return {
        "code_sha256": _code_sha256(code),
        "defaults": _exact_function_default_payload(
            target,
            state_context=state_context,
        ),
        "closure": _exact_function_closure_payload(
            target,
            state_context=state_context,
        ),
        "fields": _dataclass_constructor_fields_payload(
            owner,
            state_context=state_context,
        ),
    }


def _attest_generated_dataclass_init_state(
    target: Any,
    *,
    owner: type,
    module_name: str,
    label: str,
    source_contract: Mapping[str, Any],
    state_context: _StateSerializationContext,
    clean_reference: _CleanIsolatedReference,
) -> str:
    """Compare generated constructor state against the clean-child equivalent."""

    if (
        set(source_contract) != {"generated_dataclass_init"}
        or source_contract.get("generated_dataclass_init") is not True
    ):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has an invalid "
            "generated dataclass contract."
        )
    if type(target) is not FunctionType:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} is not an exact "
            "generated dataclass function."
        )
    try:
        actual_payload = _generated_dataclass_init_payload(
            target,
            owner=owner,
            state_context=state_context,
        )
    except _ExecutionStateUnsupported as exc:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} has unsupported "
            "generated dataclass constructor state."
        ) from exc
    expected_payload = clean_reference.default_payload(module_name, label)
    if expected_payload != actual_payload:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} symbol {label!r} generated dataclass "
            "constructor state does not match the clean isolated reference."
        )
    return canonical_json_sha256(actual_payload)


def _direct_callable_by_qualname(module: ModuleType, qualname: str) -> FunctionType:
    """Resolve a direct source callable using exact module/class dictionaries."""

    parts = qualname.split(".")
    if len(parts) == 1:
        value = vars(module).get(parts[0], _MISSING_RUNTIME_BINDING)
        return _unwrap_exact_function(
            value,
            module_name=module.__name__,
            label=qualname,
        )
    if len(parts) != 2:
        raise CanonicalExecutionOriginError(
            f"Clean isolated reference has an unsupported callable qualname {qualname!r}."
        )
    owner = vars(module).get(parts[0], _MISSING_RUNTIME_BINDING)
    if not inspect.isclass(owner):
        raise CanonicalExecutionOriginError(
            f"Clean isolated reference is missing class owner for {qualname!r}."
        )
    descriptor = vars(owner).get(parts[1], _MISSING_RUNTIME_BINDING)
    candidates = _descriptor_callables(descriptor)
    if len(candidates) != 1:
        raise CanonicalExecutionOriginError(
            f"Clean isolated reference has an ambiguous callable descriptor {qualname!r}."
        )
    return _unwrap_exact_function(
        candidates[0],
        module_name=module.__name__,
        label=qualname,
    )


def _dependency_evidence(dependency: str) -> dict[str, str]:
    """Record source origin, package version, and import API for one dependency."""

    if type(dependency) is not str or not dependency:
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference dependency name is invalid."
        )
    spec = importlib.util.find_spec(dependency)
    if spec is None:
        return {
            "name": dependency,
            "origin": "",
            "origin_sha256": "",
            "version": "",
            "api_sha256": canonical_json_sha256({"available": False}),
        }
    origin = str(spec.origin or "")
    origin_sha256 = ""
    if origin and origin not in {"built-in", "frozen", "namespace"}:
        origin_path = Path(origin)
        if origin_path.is_file():
            origin_sha256 = sha256_file(origin_path)
    try:
        version = str(importlib_metadata.version(dependency))
    except importlib_metadata.PackageNotFoundError:
        version = ""
    api = {
        "available": True,
        "name": str(spec.name or ""),
        "origin": origin,
        "loader": _loader_type_name(spec.loader),
        "is_package": bool(spec.submodule_search_locations is not None),
    }
    return {
        "name": dependency,
        "origin": origin,
        "origin_sha256": origin_sha256,
        "version": version,
        "api_sha256": canonical_json_sha256(api),
    }


def _builtin_identity_payload(value: Any) -> dict[str, str]:
    """Describe a trusted builtin without serializing arbitrary implementation state."""

    value_type = type(value)
    if value is Ellipsis:
        return {
            "type": "builtins.ellipsis",
            "module": "builtins",
            "qualname": "Ellipsis",
        }
    if value_type is type:
        try:
            module_name = _safe_class_text_attribute(value, "__module__")
            qualname = _safe_class_text_attribute(value, "__qualname__")
        except _ExecutionStateUnsupported as exc:
            raise CanonicalExecutionOriginError(
                "Builtin identity payload is incomplete."
            ) from exc
    elif value_type in {BuiltinFunctionType, FunctionType}:
        # Both exact callable implementations use object-level metadata.  Do
        # not call ``getattr`` here: a proxy injected into ``builtins`` can use
        # an attribute hook to execute during the rejection path.
        try:
            module_name = object.__getattribute__(value, "__module__")
            qualname = object.__getattribute__(value, "__qualname__")
        except AttributeError as exc:
            raise CanonicalExecutionOriginError(
                "Builtin identity payload is incomplete."
            ) from exc
        if type(module_name) is not str or type(qualname) is not str:
            raise CanonicalExecutionOriginError(
                "Builtin identity payload is incomplete."
            )
    else:
        raise CanonicalExecutionOriginError(
            "Builtin identity has an untrusted runtime type "
            + _safe_runtime_type_label(value)
        )
    return {
        "type": _safe_runtime_type_label(value),
        "module": module_name,
        "qualname": qualname,
    }


def _external_owner_module_payload(module_name: str) -> dict[str, str]:
    """Return a source-pinned record for a loaded external owner module."""

    if type(module_name) is not str or not module_name:
        raise CanonicalExecutionOriginError(
            "External binding has an invalid owner module."
        )
    module = sys.modules.get(module_name)
    if type(module) is not ModuleType:
        raise CanonicalExecutionOriginError(
            f"External binding owner module {module_name!r} is not an exact module."
        )
    module_values = object.__getattribute__(module, "__dict__")
    if type(module_values) is not dict:
        raise CanonicalExecutionOriginError(
            f"External binding owner module {module_name!r} has an invalid dictionary."
        )
    recorded_name = dict.get(module_values, "__name__", "")
    source_raw = dict.get(module_values, "__file__", "")
    if type(recorded_name) is not str or recorded_name != module_name:
        raise CanonicalExecutionOriginError(
            f"External binding owner module {module_name!r} has an invalid name."
        )
    if type(source_raw) is not str:
        raise CanonicalExecutionOriginError(
            f"External binding owner module {module_name!r} has an invalid source path."
        )
    source_path = ""
    source_sha256 = ""
    if source_raw:
        source = Path(source_raw).resolve()
        if not source.is_file():
            raise CanonicalExecutionOriginError(
                f"External binding owner module {module_name!r} source is unavailable."
            )
        source_path = str(source)
        source_sha256 = sha256_file(source)
    return {
        "module": module_name,
        "path": source_path,
        "sha256": source_sha256,
    }


def _external_exact_function_structure(
    value: FunctionType,
    *,
    owner_module: str | None = None,
) -> dict[str, str]:
    """Fingerprint an external Python function without invoking user hooks."""

    if type(value) is not FunctionType:
        raise CanonicalExecutionOriginError("External function binding is not exact.")
    module_name = object.__getattribute__(value, "__module__")
    qualname = object.__getattribute__(value, "__qualname__")
    code = object.__getattribute__(value, "__code__")
    globals_mapping = object.__getattribute__(value, "__globals__")
    if (
        type(module_name) is not str
        or not module_name
        or type(qualname) is not str
        or not qualname
        or type(code) is not CodeType
        or type(globals_mapping) is not dict
    ):
        raise CanonicalExecutionOriginError(
            "External function binding has invalid metadata."
        )
    if owner_module is not None and module_name != owner_module:
        raise CanonicalExecutionOriginError(
            f"External function binding has owner {module_name!r}, expected {owner_module!r}."
        )
    owner = _external_owner_module_payload(module_name)
    live_owner = sys.modules.get(module_name)
    if (
        type(live_owner) is not ModuleType
        or object.__getattribute__(live_owner, "__dict__") is not globals_mapping
    ):
        raise CanonicalExecutionOriginError(
            f"External function binding {module_name}.{qualname} has foreign globals."
        )
    return {
        "module": module_name,
        "qualname": qualname,
        "code_sha256": _code_sha256(code),
        "owner_source_sha256": owner["sha256"],
    }


def _external_descriptor_function_records(
    descriptor: Any,
    *,
    owner_module: str,
) -> list[dict[str, str]]:
    """Return code records for exact built-in descriptor forms only."""

    descriptor_type = type(descriptor)
    candidates: tuple[Any, ...]
    if descriptor_type is FunctionType:
        candidates = (descriptor,)
    elif descriptor_type in {staticmethod, classmethod}:
        candidates = (object.__getattribute__(descriptor, "__func__"),)
    elif descriptor_type is property:
        candidates = tuple(
            candidate
            for candidate in (
                object.__getattribute__(descriptor, "fget"),
                object.__getattribute__(descriptor, "fset"),
                object.__getattribute__(descriptor, "fdel"),
            )
            if candidate is not None
        )
    elif descriptor_type is functools.cached_property:
        candidates = (object.__getattribute__(descriptor, "func"),)
    else:
        return []
    records: list[dict[str, str]] = []
    for candidate in candidates:
        if type(candidate) is not FunctionType:
            raise CanonicalExecutionOriginError(
                "External class has a malformed callable descriptor."
            )
        records.append(
            _external_exact_function_structure(candidate, owner_module=owner_module)
        )
    return records


def _external_class_structure(value: type) -> dict[str, str]:
    """Fingerprint external class bases and direct callable code structurally."""

    if not isinstance(value, type):
        raise CanonicalExecutionOriginError("External class binding is not a class.")
    try:
        module_name = _safe_class_text_attribute(value, "__module__")
        qualname = _safe_class_text_attribute(value, "__qualname__")
        class_values = _mappingproxy_backing_dict(_safe_class_dict(value))
        bases = type.__getattribute__(value, "__bases__")
        metaclass = type(value)
    except _ExecutionStateUnsupported as exc:
        raise CanonicalExecutionOriginError(
            "External class binding has invalid metadata."
        ) from exc
    if type(bases) is not tuple or any(not isinstance(base, type) for base in bases):
        raise CanonicalExecutionOriginError("External class binding has invalid bases.")
    owner = _external_owner_module_payload(module_name)
    base_records = [
        {
            "module": _safe_class_text_attribute(base, "__module__"),
            "qualname": _safe_class_text_attribute(base, "__qualname__"),
        }
        for base in tuple.__iter__(bases)
    ]
    member_records: list[dict[str, Any]] = []
    for member_name, descriptor in sorted(
        dict.items(class_values),
        key=lambda item: str(item[0]),
    ):
        if type(member_name) is not str:
            raise CanonicalExecutionOriginError(
                "External class binding has a non-string member name."
            )
        member_records.append(
            {
                "name": member_name,
                "descriptor_type": _safe_runtime_type_label(descriptor),
                "callables": _external_descriptor_function_records(
                    descriptor,
                    owner_module=module_name,
                ),
            }
        )
    structure_sha256 = canonical_json_sha256(
        {
            "owner_source": owner,
            "bases": base_records,
            "metaclass": _safe_runtime_type_label(metaclass),
            "members": member_records,
        }
    )
    return {
        "module": module_name,
        "qualname": qualname,
        "structure_sha256": structure_sha256,
    }


def _external_binding_identity(value: Any) -> dict[str, str]:
    """Describe a dependency binding with code/structure, not nominal identity.

    A matching module and qualified name is insufficient: an in-process caller
    can replace both a consumer alias and its owner attribute.  Exact Python
    functions and classes therefore carry clean-child-comparable code or class
    structure fingerprints in addition to their nominal owner.
    """

    if value is None:
        return {"kind": "none", "module": "", "qualname": "", "structure_sha256": ""}
    if type(value) is ModuleType:
        module_values = object.__getattribute__(value, "__dict__")
        if type(module_values) is not dict:
            raise CanonicalExecutionOriginError(
                "External module binding has an invalid dictionary."
            )
        module_name = dict.get(module_values, "__name__", "")
        if type(module_name) is not str or not module_name:
            raise CanonicalExecutionOriginError(
                "External module binding has an invalid name."
            )
        return {
            "kind": "module",
            "module": module_name,
            "qualname": "",
            "structure_sha256": canonical_json_sha256(
                _external_owner_module_payload(module_name)
            ),
        }
    if isinstance(value, type):
        class_payload = _external_class_structure(value)
        return {"kind": "class", **class_payload}
    if type(value) is FunctionType:
        function_payload = _external_exact_function_structure(value)
        return {
            "kind": "function",
            "module": function_payload["module"],
            "qualname": function_payload["qualname"],
            "structure_sha256": canonical_json_sha256(function_payload),
        }
    if type(value) is BuiltinFunctionType:
        module_name = object.__getattribute__(value, "__module__")
        qualname = object.__getattribute__(value, "__qualname__")
        if type(module_name) is not str or type(qualname) is not str:
            raise CanonicalExecutionOriginError(
                "External builtin binding has invalid metadata."
            )
        return {
            "kind": "builtin_function",
            "module": module_name,
            "qualname": qualname,
            "structure_sha256": canonical_json_sha256(
                _external_owner_module_payload(module_name)
            ),
        }
    value_type = type(value)
    return {
        "kind": f"object:{_safe_runtime_type_label(value)}",
        "module": _safe_class_text_attribute(value_type, "__module__"),
        "qualname": _safe_class_text_attribute(value_type, "__qualname__"),
        "structure_sha256": "",
    }


def _clean_isolated_reference_child_payload(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce clean source/default payloads in the isolated child process only."""

    package_root_raw = request.get("package_root")
    raw_modules = request.get("modules")
    raw_dependencies = request.get("dependencies")
    raw_environment = request.get("environment")
    raw_child_environment = request.get("child_environment")
    raw_isolated_paths = request.get("isolated_paths")
    if (
        type(package_root_raw) is not str
        or not isinstance(raw_modules, list)
        or not isinstance(raw_dependencies, list)
        or not isinstance(raw_environment, Mapping)
        or not isinstance(raw_child_environment, Mapping)
        or not isinstance(raw_isolated_paths, Mapping)
        or any(
            type(key) is not str or type(value) is not str
            for key, value in raw_environment.items()
        )
        or any(
            type(key) is not str or type(value) is not str
            for key, value in raw_child_environment.items()
        )
        or set(raw_isolated_paths) != set(_CLEAN_ISOLATED_REFERENCE_PATH_KEYS)
        or any(
            type(key) is not str or type(value) is not str
            for key, value in raw_isolated_paths.items()
        )
    ):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference request has invalid fields."
        )
    if raw_isolated_paths.get(
        "source_package_root"
    ) != "verified_temp_source_copy" or any(
        raw_isolated_paths.get(name) != f"child_owned:{name}"
        for name in _CLEAN_ISOLATED_REFERENCE_PATH_KEYS
        if name != "source_package_root"
    ):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference path policy is invalid."
        )
    if any(
        os.environ.get(key, "") != value for key, value in raw_child_environment.items()
    ):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference child environment was not applied."
        )
    package_root = Path(package_root_raw).resolve()
    prepared_modules: list[_PreparedLoadedModule] = []
    requested_by_name: dict[
        str,
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[str, str, str], ...],
        ],
    ] = {}
    for raw_module in raw_modules:
        if not isinstance(raw_module, Mapping) or set(raw_module) != {
            "module",
            "source_sha256",
            "definition_selection_sha256",
            "states",
            "defaults",
            "builtins",
            "external_bindings",
        }:
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference module request is invalid."
            )
        module_name = raw_module.get("module")
        source_sha256 = raw_module.get("source_sha256")
        definition_selection_sha256 = raw_module.get("definition_selection_sha256")
        states = raw_module.get("states")
        defaults = raw_module.get("defaults")
        builtin_names = raw_module.get("builtins")
        external_bindings = raw_module.get("external_bindings")
        if (
            type(module_name) is not str
            or not _is_sha256_digest(source_sha256)
            or not _is_sha256_digest(definition_selection_sha256)
            or not isinstance(states, list)
            or not isinstance(defaults, list)
            or not isinstance(builtin_names, list)
            or not isinstance(external_bindings, list)
            or any(
                type(value) is not str for value in states + defaults + builtin_names
            )
            or states != sorted(set(states))
            or defaults != sorted(set(defaults))
            or builtin_names != sorted(set(builtin_names))
            or module_name in requested_by_name
        ):
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference module request is invalid."
            )
        normalized_external_bindings: list[tuple[str, str, str]] = []
        for binding in external_bindings:
            if not isinstance(binding, Mapping) or set(binding) != {
                "local_name",
                "module",
                "attribute",
            }:
                raise CanonicalExecutionOriginError(
                    "Clean isolated-reference external binding is invalid."
                )
            values = (
                binding.get("local_name"),
                binding.get("module"),
                binding.get("attribute"),
            )
            if any(type(value) is not str for value in values):
                raise CanonicalExecutionOriginError(
                    "Clean isolated-reference external binding is invalid."
                )
            normalized_external_bindings.append(values)
        if normalized_external_bindings != sorted(set(normalized_external_bindings)):
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference external bindings are not canonical."
            )
        module = importlib.import_module(module_name)
        if type(module) is not ModuleType:
            raise CanonicalExecutionOriginError(
                f"Clean isolated reference did not import a module for {module_name!r}."
            )
        source_raw = str(getattr(module, "__file__", "") or "")
        source = Path(source_raw).resolve()
        if not _is_within(source, package_root) or sha256_file(source) != source_sha256:
            raise CanonicalExecutionOriginError(
                f"Clean isolated reference module source does not match request: {module_name!r}."
            )
        reference = _source_symbol_reference(
            str(source),
            str(source_sha256),
            module_name,
            source.name == "__init__.py",
        )
        reference = _resolve_source_symbol_reference(
            reference,
            module_name=module_name,
            module=module,
            source=source,
        )
        prepared = _PreparedLoadedModule(
            module_name=module_name,
            module=module,
            source=source,
            source_sha256=str(source_sha256),
            is_package=source.name == "__init__.py",
            reference=reference,
            state_plan=_source_static_state_plan(reference, module_name=module_name),
        )
        if (
            _selected_definition_variants_sha256(reference)
            != definition_selection_sha256
        ):
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference definition selection differs from the live "
                f"module for {module_name!r}."
            )
        expected_states, expected_defaults, _ = _source_state_request(prepared)
        expected_builtins = _source_builtin_names(prepared)
        expected_external_bindings = _declared_dependency_import_bindings(
            prepared,
            _source_state_request(prepared)[2],
        )
        expected_external_tuples = tuple(
            (item["local_name"], item["module"], item["attribute"])
            for item in expected_external_bindings
        )
        if (
            tuple(states) != tuple(expected_states)
            or tuple(defaults) != tuple(expected_defaults)
            or tuple(builtin_names) != tuple(expected_builtins)
            or tuple(normalized_external_bindings) != expected_external_tuples
        ):
            raise CanonicalExecutionOriginError(
                f"Clean isolated-reference request does not exactly match source state/defaults for {module_name!r}."
            )
        requested_by_name[module_name] = (
            tuple(states),
            tuple(defaults),
            tuple(builtin_names),
            tuple(normalized_external_bindings),
        )
        prepared_modules.append(prepared)
    # State payloads can contain classes/functions authored by an imported
    # sibling module (for example a runner's sealed pipeline class).  Admit
    # identities from the complete clean loaded source closure, not merely the
    # requested modules, while retaining the request's exact output boundary.
    context_modules = list(prepared_modules)
    context_names = {prepared.module_name for prepared in context_modules}
    for loaded_name, loaded_module in tuple(sys.modules.items()):
        module_name = str(loaded_name)
        if module_name in context_names or not (
            module_name == LOADED_PACKAGE_MODULE_PREFIX
            or module_name.startswith(LOADED_PACKAGE_MODULE_PREFIX + ".")
        ):
            continue
        if type(loaded_module) is not ModuleType:
            continue
        source_raw = str(getattr(loaded_module, "__file__", "") or "")
        if not source_raw:
            continue
        source = Path(source_raw).resolve()
        if not source.is_file() or not _is_within(source, package_root):
            continue
        source_sha256 = sha256_file(source)
        reference = _source_symbol_reference(
            str(source),
            source_sha256,
            module_name,
            source.name == "__init__.py",
        )
        reference = _resolve_source_symbol_reference(
            reference,
            module_name=module_name,
            module=loaded_module,
            source=source,
        )
        context_modules.append(
            _PreparedLoadedModule(
                module_name=module_name,
                module=loaded_module,
                source=source,
                source_sha256=source_sha256,
                is_package=source.name == "__init__.py",
                reference=reference,
                state_plan=_source_static_state_plan(
                    reference, module_name=module_name
                ),
            )
        )
        context_names.add(module_name)
    context = _build_state_serialization_context(
        context_modules,
        package_root=package_root,
        repository_root=_staged_clean_reference_repository_root(package_root),
    )
    module_payloads: dict[str, dict[str, Any]] = {}
    for prepared in prepared_modules:
        states, defaults, builtin_names, external_bindings = requested_by_name[
            prepared.module_name
        ]
        module_values = vars(prepared.module)
        state_payloads: dict[str, Any] = {}
        for state_name in states:
            value = module_values.get(state_name, _MISSING_RUNTIME_BINDING)
            if value is _MISSING_RUNTIME_BINDING:
                raise CanonicalExecutionOriginError(
                    f"Clean isolated reference is missing state {prepared.module_name!r}:{state_name!r}."
                )
            state_payloads[state_name] = _state_value_payload(value, context=context)
        default_payloads: dict[str, Any] = {}
        callable_contracts = prepared.reference.get("callable_defaults")
        if not isinstance(callable_contracts, Mapping):
            raise CanonicalExecutionOriginError(
                "Clean isolated reference has an invalid callable-default index."
            )
        for qualname in defaults:
            target = _direct_callable_by_qualname(prepared.module, qualname)
            source_contract = callable_contracts.get(qualname)
            if not isinstance(source_contract, Mapping):
                raise CanonicalExecutionOriginError(
                    f"Clean isolated reference has no callable contract for {qualname!r}."
                )
            if source_contract.get("generated_dataclass_init") is True:
                owner_name, separator, method_name = qualname.partition(".")
                owner = vars(prepared.module).get(owner_name, _MISSING_RUNTIME_BINDING)
                if (
                    separator != "."
                    or method_name != "__init__"
                    or not isinstance(owner, type)
                ):
                    raise CanonicalExecutionOriginError(
                        f"Clean isolated reference has an invalid generated dataclass owner for {qualname!r}."
                    )
                default_payloads[qualname] = _generated_dataclass_init_payload(
                    target,
                    owner=owner,
                    state_context=context,
                )
            else:
                default_payloads[qualname] = _exact_function_default_payload(
                    target,
                    state_context=context,
                )
        builtin_payloads: dict[str, Any] = {}
        for builtin_name in builtin_names:
            builtin_value = _trusted_builtin_value(
                builtin_name,
                _MISSING_RUNTIME_BINDING,
            )
            if builtin_value is _MISSING_RUNTIME_BINDING:
                raise CanonicalExecutionOriginError(
                    f"Clean isolated reference is missing builtin {builtin_name!r}."
                )
            builtin_payloads[builtin_name] = _builtin_identity_payload(builtin_value)
        external_payloads: list[dict[str, str]] = []
        for local_name, owner_name, attribute in external_bindings:
            value = module_values.get(local_name, _MISSING_RUNTIME_BINDING)
            identity = (
                {
                    "kind": "missing",
                    "module": "",
                    "qualname": "",
                    "structure_sha256": "",
                }
                if value is _MISSING_RUNTIME_BINDING
                else _external_binding_identity(value)
            )
            external_payloads.append(
                {
                    "local_name": local_name,
                    "owner_module": owner_name,
                    "attribute": attribute,
                    "kind": identity["kind"],
                    "target_module": identity["module"],
                    "qualname": identity["qualname"],
                    "structure_sha256": identity["structure_sha256"],
                }
            )
        module_payloads[prepared.module_name] = {
            "states": state_payloads,
            "defaults": default_payloads,
            "builtins": builtin_payloads,
            "external_bindings": external_payloads,
        }
    dependencies = sorted(set(raw_dependencies))
    if any(type(value) is not str or not value for value in dependencies):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference dependency request is invalid."
        )
    post_manifest = _source_tree_manifest(package_root)
    return {
        "schema_version": _CLEAN_ISOLATED_REFERENCE_SCHEMA_VERSION,
        "source_manifest_sha256": canonical_json_sha256(post_manifest),
        "environment": {
            str(key): str(value) for key, value in sorted(raw_environment.items())
        },
        "isolated_paths": {
            str(key): str(value) for key, value in sorted(raw_isolated_paths.items())
        },
        "python": {
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "dependencies": {
            dependency: _dependency_evidence(dependency) for dependency in dependencies
        },
        "modules": module_payloads,
    }


def _validate_clean_isolated_reference_response(
    request: Mapping[str, Any],
    expected: Mapping[
        str,
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[str, str, str], ...],
        ],
    ],
    response: Any,
) -> _CleanIsolatedReference:
    """Reject malformed, extra, missing, or source-racy child responses."""

    expected_keys = {
        "schema_version",
        "source_manifest_sha256",
        "environment",
        "isolated_paths",
        "python",
        "platform",
        "dependencies",
        "modules",
    }
    if not isinstance(response, dict) or set(response) != expected_keys:
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference response has an invalid shape."
        )
    if response.get("schema_version") != _CLEAN_ISOLATED_REFERENCE_SCHEMA_VERSION:
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference response schema is invalid."
        )
    if response.get("source_manifest_sha256") != request.get("source_manifest_sha256"):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference source tree changed during import/evaluation."
        )
    environment = response.get("environment")
    requested_environment = request.get("environment")
    if not isinstance(environment, dict) or environment != requested_environment:
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference environment mismatch."
        )
    isolated_paths = response.get("isolated_paths")
    requested_isolated_paths = request.get("isolated_paths")
    if (
        not isinstance(isolated_paths, dict)
        or isolated_paths != requested_isolated_paths
    ):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference path policy mismatch."
        )
    python_record = response.get("python")
    platform_record = response.get("platform")
    if (
        not isinstance(python_record, dict)
        or set(python_record) != {"implementation", "version", "executable"}
        or not isinstance(platform_record, dict)
        or set(platform_record) != {"system", "machine"}
        or any(type(value) is not str for value in python_record.values())
        or any(type(value) is not str for value in platform_record.values())
    ):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference runtime evidence is invalid."
        )
    if (
        python_record["implementation"] != sys.implementation.name
        or python_record["version"] != platform.python_version()
        or Path(python_record["executable"]).resolve() != Path(sys.executable).resolve()
        or platform_record["system"] != platform.system()
        or platform_record["machine"] != platform.machine()
    ):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference runtime evidence does not match the invoking host."
        )
    raw_dependencies = response.get("dependencies")
    requested_dependencies = request.get("dependencies")
    if not isinstance(raw_dependencies, dict) or not isinstance(
        requested_dependencies, list
    ):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference dependency evidence is invalid."
        )
    if set(raw_dependencies) != set(requested_dependencies):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference dependency set mismatch."
        )
    dependencies: dict[str, Mapping[str, str]] = {}
    for dependency in sorted(raw_dependencies):
        record = raw_dependencies[dependency]
        if (
            not isinstance(record, dict)
            or set(record)
            != {"name", "origin", "origin_sha256", "version", "api_sha256"}
            or record.get("name") != dependency
            or any(type(value) is not str for value in record.values())
        ):
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference dependency record is invalid."
            )
        live_record = _dependency_evidence(dependency)
        if record != live_record:
            raise CanonicalExecutionOriginError(
                f"Clean isolated-reference dependency evidence differs from the live environment: {dependency!r}."
            )
        dependencies[dependency] = record
    raw_modules = response.get("modules")
    if not isinstance(raw_modules, dict) or set(raw_modules) != set(expected):
        raise CanonicalExecutionOriginError(
            "Clean isolated-reference module set mismatch."
        )
    modules: dict[str, Mapping[str, Any]] = {}
    for module_name in sorted(expected):
        record = raw_modules[module_name]
        if not isinstance(record, dict) or set(record) != {
            "states",
            "defaults",
            "builtins",
            "external_bindings",
        }:
            raise CanonicalExecutionOriginError(
                "Clean isolated-reference module payload is invalid."
            )
        states = record.get("states")
        defaults = record.get("defaults")
        builtin_payloads = record.get("builtins")
        external_payloads = record.get("external_bindings")
        (
            expected_states,
            expected_defaults,
            expected_builtins,
            expected_external_bindings,
        ) = expected[module_name]
        if (
            not isinstance(states, dict)
            or not isinstance(defaults, dict)
            or not isinstance(builtin_payloads, dict)
            or not isinstance(external_payloads, list)
            or set(states) != set(expected_states)
            or set(defaults) != set(expected_defaults)
            or set(builtin_payloads) != set(expected_builtins)
        ):
            raise CanonicalExecutionOriginError(
                f"Clean isolated-reference symbols do not exactly match request for {module_name!r}."
            )
        normalized_external_bindings: list[tuple[str, str, str]] = []
        for payload in external_payloads:
            if not isinstance(payload, dict) or set(payload) != {
                "local_name",
                "owner_module",
                "attribute",
                "kind",
                "target_module",
                "qualname",
                "structure_sha256",
            }:
                raise CanonicalExecutionOriginError(
                    "Clean isolated-reference external binding payload is invalid."
                )
            if any(type(value) is not str for value in payload.values()):
                raise CanonicalExecutionOriginError(
                    "Clean isolated-reference external binding payload is invalid."
                )
            normalized_external_bindings.append(
                (
                    payload["local_name"],
                    payload["owner_module"],
                    payload["attribute"],
                )
            )
        if tuple(normalized_external_bindings) != expected_external_bindings:
            raise CanonicalExecutionOriginError(
                f"Clean isolated-reference external bindings do not match request for {module_name!r}."
            )
        modules[module_name] = record
    return _CleanIsolatedReference(
        module_payloads=modules,
        source_tree_sha256=str(response["source_manifest_sha256"]),
        environment=environment,
        isolated_paths=isolated_paths,
        python=python_record,
        platform=platform_record,
        dependencies=dependencies,
    )


def _clean_reference_record(reference: _CleanIsolatedReference) -> dict[str, Any]:
    """Persist the child evidence that selected dynamic/default baselines."""

    return {
        "schema_version": _CLEAN_ISOLATED_REFERENCE_SCHEMA_VERSION,
        "source_tree_sha256": reference.source_tree_sha256,
        "environment": dict(reference.environment),
        "isolated_paths": dict(reference.isolated_paths),
        "python": dict(reference.python),
        "platform": dict(reference.platform),
        "dependencies": {
            name: dict(record)
            for name, record in sorted(reference.dependencies.items())
        },
    }


def _is_logging_logger(
    value: Any,
    _logger_type: type = logging.Logger,
) -> bool:
    """Allow the narrow logger exclusion without treating arbitrary objects as caches."""

    return type(value) is _logger_type


def _attest_module_runtime_state(
    *,
    module_name: str,
    module: ModuleType,
    reference: Mapping[str, Any],
    state_plan: _SourceStaticStatePlan,
    state_context: _StateSerializationContext,
    clean_reference: _CleanIsolatedReference | None,
) -> list[dict[str, str]]:
    """Seal reachable source globals using AST literals or a clean child only."""

    raw_references = reference.get("module_state_references")
    raw_definitions = reference.get("definitions")
    imported_names = reference.get("imported_binding_names")
    if (
        not isinstance(raw_references, tuple)
        or not all(isinstance(name, str) for name in raw_references)
        or not isinstance(raw_definitions, tuple)
        or not isinstance(imported_names, frozenset)
    ):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has invalid source state references."
        )
    definitions = {
        str(record.get("name", "") or "")
        for record in raw_definitions
        if isinstance(record, Mapping)
    }
    records: list[dict[str, str]] = []
    for name in raw_references:
        # Imports and source functions/classes are covered by their code and
        # internal-import contracts, which ran before this state phase.
        if name in definitions or name in imported_names:
            continue
        if name in {"__name__", "__file__", "__package__"}:
            continue
        if (
            name not in state_plan.assigned_names
            and _trusted_builtin_value(name, _MISSING_RUNTIME_BINDING)
            is not _MISSING_RUNTIME_BINDING
        ):
            continue
        if name == "__tabnetics_execution_ephemeral_globals__":
            continue
        if name in state_plan.ephemeral_names:
            continue
        actual = vars(module).get(name, _MISSING_RUNTIME_BINDING)
        if actual is _MISSING_RUNTIME_BINDING:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} is missing referenced source "
                f"state {name!r}."
            )
        # A source-authored process logger is diagnostic-only.  Do not make
        # this a type-only exemption: an attacker could replace an algorithmic
        # global with an exact ``logging.Logger`` and bypass state sealing.
        if name == "logger" and _source_declares_process_logger(reference):
            if not _is_logging_logger(actual):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} source logger binding is invalid."
                )
            continue
        provider_name = ""
        provider_payload: Mapping[str, Any] | None = None
        try:
            spec = state_plan.isolated_specs.get(name)
            if isinstance(spec, Mapping):
                if clean_reference is None:
                    raise _ExecutionStateUnsupported(
                        "clean isolated reference is unavailable"
                    )
                provider_name = "clean_isolated_reference_v1"
                provider_payload = {
                    "dependencies": list(spec.get("dependencies", ())),
                    "source_tree_sha256": clean_reference.source_tree_sha256,
                }
                expected_payload = clean_reference.state_payload(module_name, name)
            elif isinstance(
                internal_import_spec := state_plan.internal_import_state_specs.get(
                    name
                ),
                Mapping,
            ):
                if clean_reference is None:
                    raise _ExecutionStateUnsupported(
                        "clean isolated reference is unavailable"
                    )
                source_expression_sha256, import_bindings = (
                    _validated_internal_import_state_spec(
                        internal_import_spec,
                        module_name=module_name,
                        state_name=name,
                    )
                )
                provider_name = "clean_isolated_internal_import_expression_v1"
                provider_payload = {
                    "source_expression_sha256": source_expression_sha256,
                    "import_bindings": list(import_bindings),
                    "source_tree_sha256": clean_reference.source_tree_sha256,
                }
                expected_payload = clean_reference.state_payload(module_name, name)
            elif (
                name in state_plan.values
                and name not in state_plan.ambiguous_names
                and name not in state_plan.unsupported_names
            ):
                expected_payload = _state_value_payload(
                    state_plan.values[name],
                    context=state_context,
                )
            else:
                raise _ExecutionStateUnsupported(
                    "nonliteral/conditional state has no clean isolated reference"
                )
            actual_payload = _state_value_payload(actual, context=state_context)
        except _ExecutionStateUnsupported as exc:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} referenced source state {name!r} "
                "cannot be independently sealed; declare a justified source-owned "
                "ephemeral cache or add a clean isolated-reference declaration."
            ) from exc
        if expected_payload != actual_payload:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} referenced source state {name!r} "
                "does not match its verified source-derived value."
            )
        records.append(
            {
                "owner": "module",
                "name": name,
                "type": str(actual_payload.get("kind", "") or ""),
                "semantic_sha256": canonical_json_sha256(actual_payload),
                "provider": provider_name,
                "provider_sha256": (
                    ""
                    if provider_payload is None
                    else canonical_json_sha256(provider_payload)
                ),
            }
        )
    records.sort(key=lambda record: (record["owner"], record["name"]))
    return records


def _attest_class_static_state(
    *,
    target: type,
    name: str,
    module_name: str,
    source_bindings: Sequence[Mapping[str, Any]],
    state_plan: _SourceStaticStatePlan,
    state_context: _StateSerializationContext,
) -> tuple[list[dict[str, str]], str]:
    """Seal direct literal class state while leaving generated descriptors alone."""

    known_values = dict(state_plan.values)
    records: list[dict[str, str]] = []
    try:
        class_dict = _safe_class_dict(target)
        class_values = _mappingproxy_backing_dict(class_dict)
    except _ExecutionStateUnsupported as exc:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has invalid class state."
        ) from exc
    for binding in source_bindings:
        raw_name = binding.get("name") if isinstance(binding, Mapping) else None
        expression = binding.get("value") if isinstance(binding, Mapping) else None
        state_name = str(raw_name or "")
        if not state_name or not isinstance(expression, ast.AST):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} class {name!r} has an invalid "
                "source class-state binding."
            )
        try:
            expected = _static_source_expression_value(expression, known_values)
        except _ExecutionStateUnsupported:
            # Class decorators and dataclass ``field(...)`` entries are not
            # literal runtime state. Their behavior is covered by callable
            # defaults/methods and is intentionally not repr-hashed here.
            continue
        known_values[state_name] = expected
        actual = dict.get(class_values, state_name, _MISSING_RUNTIME_BINDING)
        if dataclasses.is_dataclass(target) and _is_trusted_member_descriptor(actual):
            dataclass_fields = type.__getattribute__(target, "__dataclass_fields__")
            dataclass_field = (
                dict.get(dataclass_fields, state_name)
                if type(dataclass_fields) is dict
                else None
            )
            if dataclass_field is not None:
                actual = dataclass_field.default
        if actual is _MISSING_RUNTIME_BINDING:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} class {name!r} is missing "
                f"source class state {state_name!r}."
            )
        if issubclass(target, Enum) and state_name in target.__members__:
            actual = target.__members__[state_name].value
        try:
            expected_payload = _state_value_payload(expected, context=state_context)
            actual_payload = _state_value_payload(actual, context=state_context)
        except _ExecutionStateUnsupported as exc:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} class {name!r} source state "
                f"{state_name!r} cannot be sealed."
            ) from exc
        if expected_payload != actual_payload:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} class {name!r} source state "
                f"{state_name!r} does not match its verified literal value."
            )
        records.append(
            {
                "owner": f"class:{name}",
                "name": state_name,
                "type": str(actual_payload.get("kind", "") or ""),
                "semantic_sha256": canonical_json_sha256(actual_payload),
            }
        )
    records.sort(key=lambda record: (record["owner"], record["name"]))
    return records, canonical_json_sha256(records)


def _class_symbol_record(
    *,
    name: str,
    target: type,
    module_name: str,
    source: Path,
    module_values: Mapping[str, Any],
    reference: Mapping[str, Any],
    expected_member_qualnames: Sequence[str],
    source_base_contract: Mapping[str, Any],
    source_static_bindings: Sequence[Mapping[str, Any]],
    state_plan: _SourceStaticStatePlan,
    state_context: _StateSerializationContext,
    clean_reference: _CleanIsolatedReference,
) -> dict[str, Any]:
    """Attest direct source-authored class methods while allowing dataclass codegen."""

    expected_class_hashes = _reference_code_hashes(
        reference,
        name,
        module_name=module_name,
    )
    base_classes, metaclass, source_base_contract_sha256 = _attest_class_structure(
        target=target,
        name=name,
        module_name=module_name,
        module_values=module_values,
        source_contract=source_base_contract,
    )
    try:
        qualname = _safe_class_text_attribute(target, "__qualname__")
    except _ExecutionStateUnsupported as exc:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has invalid metadata."
        ) from exc
    if qualname != name:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has an unexpected "
            "qualified name."
        )
    state_records, state_sha256 = _attest_class_static_state(
        target=target,
        name=name,
        module_name=module_name,
        source_bindings=source_static_bindings,
        state_plan=state_plan,
        state_context=state_context,
    )
    raw_default_contracts = reference.get("callable_defaults")
    if not isinstance(raw_default_contracts, Mapping):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid source callable "
            "default index."
        )
    expected_member_counts = Counter(str(item) for item in expected_member_qualnames)
    expected_members = set(expected_member_counts)
    observed_member_counts: Counter[str] = Counter()
    member_records: list[dict[str, str]] = []
    try:
        class_dict = _safe_class_dict(target)
    except _ExecutionStateUnsupported as exc:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} has invalid class state."
        ) from exc
    for member_name, descriptor in sorted(
        dict.items(_mappingproxy_backing_dict(class_dict)),
        key=lambda item: str(item[0]),
    ):
        for member in _descriptor_callables(descriptor):
            unwrapped = _unwrap_exact_function(
                member,
                module_name=module_name,
                label=f"{name}.{member_name}",
            )
            member_qualname = unwrapped.__qualname__
            owner = unwrapped.__module__
            generated_dataclass_method = bool(
                dataclasses.is_dataclass(target)
                and str(member_name) in _DATACLASS_GENERATED_METHOD_NAMES
                and member_qualname not in expected_members
            )
            generated_enum_member = bool(
                issubclass(target, Enum)
                and owner == "enum"
                and member_qualname.startswith("Enum.")
            )
            if generated_dataclass_method:
                if str(member_name) == "__init__":
                    generated_label = f"{name}.__init__"
                    source_contract = raw_default_contracts.get(generated_label)
                    if not isinstance(source_contract, Mapping):
                        raise CanonicalExecutionOriginError(
                            f"Loaded tabnetics module {module_name!r} class {name!r} generated "
                            "constructor has no source contract."
                        )
                    generated_state_sha256 = _attest_generated_dataclass_init_state(
                        unwrapped,
                        owner=target,
                        module_name=module_name,
                        label=generated_label,
                        source_contract=source_contract,
                        state_context=state_context,
                        clean_reference=clean_reference,
                    )
                    member_records.append(
                        {
                            "name": str(member_name),
                            "qualname": generated_label,
                            "code_sha256": _code_sha256(unwrapped.__code__),
                            "default_state_sha256": generated_state_sha256,
                        }
                    )
                continue
            if generated_enum_member:
                continue
            if owner != module_name:
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} class {name!r} has a foreign "
                    f"direct member {member_name!r}."
                )
            if member_qualname not in expected_members:
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} class {name!r} has an "
                    f"unrecognized local member {member_name!r}."
                )
            source_default_contract = raw_default_contracts.get(member_qualname)
            if not isinstance(source_default_contract, Mapping):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} class {name!r} member "
                    f"{member_name!r} has no source default contract."
                )
            member_digest, member_default_state_sha256 = _attested_callable_code_sha256(
                unwrapped,
                module_name=module_name,
                source=source,
                module_globals=module_values,
                qualname=member_qualname,
                expected_hashes=_reference_code_hashes(
                    reference,
                    member_qualname,
                    module_name=module_name,
                ),
                label=f"{name}.{member_name}",
                source_default_contract=source_default_contract,
                state_context=state_context,
                clean_reference=clean_reference,
                class_owner=target,
            )
            member_records.append(
                {
                    "name": str(member_name),
                    "qualname": member_qualname,
                    "code_sha256": member_digest,
                    "default_state_sha256": member_default_state_sha256,
                }
            )
            observed_member_counts[member_qualname] += 1
    if observed_member_counts != expected_member_counts:
        missing = sorted((expected_member_counts - observed_member_counts).elements())
        unexpected = sorted(
            (observed_member_counts - expected_member_counts).elements()
        )
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} class {name!r} direct member set "
            f"does not match independently compiled source: missing={missing!r} "
            f"unexpected={unexpected!r}."
        )
    member_records.sort(
        key=lambda record: (
            record["name"],
            record["qualname"],
            record["code_sha256"],
            record["default_state_sha256"],
        )
    )
    return {
        "name": name,
        "kind": "class",
        "qualname": qualname,
        "source_code_sha256": canonical_json_sha256(expected_class_hashes),
        "base_classes": base_classes,
        "metaclass": metaclass,
        "source_base_contract_sha256": source_base_contract_sha256,
        "state": state_records,
        "state_sha256": state_sha256,
        "members": member_records,
        "member_code_sha256": canonical_json_sha256(member_records),
    }


def _loaded_module_symbol_records(
    module_name: str,
    module: ModuleType,
    *,
    source: Path,
    source_sha256: str,
    reference: Mapping[str, Any],
    state_plan: _SourceStaticStatePlan,
    state_context: _StateSerializationContext,
    clean_reference: _CleanIsolatedReference,
) -> list[dict[str, Any]]:
    """Attest source-authored runtime symbols in a loaded internal module.

    Module metadata can be fabricated while a module's global bindings point to
    substituted classes or functions.  The source is compiled independently
    and every live locally authored callable is matched to that reference.
    Imported third-party objects are intentionally outside this local internal
    code closure; imported ``tabnetics`` modules are separately attested when
    they are present in ``sys.modules``.
    """

    raw_definitions = reference.get("definitions")
    if not isinstance(raw_definitions, tuple):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid source definition index."
        )
    definitions: dict[str, Mapping[str, Any]] = {}
    for definition in raw_definitions:
        if not isinstance(definition, Mapping):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid source definition."
            )
        name = str(definition.get("name", "") or "")
        kind = str(definition.get("kind", "") or "")
        qualname = str(definition.get("qualname", "") or "")
        if not name or kind not in {"function", "class"} or qualname != name:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid source symbol name."
            )
        definitions[name] = definition

    module_values = vars(module)
    raw_default_contracts = reference.get("callable_defaults")
    if not isinstance(raw_default_contracts, Mapping):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid source callable "
            "default index."
        )
    records: list[dict[str, Any]] = []
    missing = object()
    for name, definition in sorted(definitions.items()):
        target = module_values.get(name, missing)
        # Definitions indexed here occur directly in the module body, so a
        # loaded source module must retain the binding.  Treat deletion or a
        # replacement with ``None`` as a divergence rather than an optional
        # import branch.
        if target is missing:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} is missing source symbol {name!r}."
            )
        kind = str(definition["kind"])
        if kind == "function":
            _unwrap_exact_function(target, module_name=module_name, label=name)
            source_default_contract = raw_default_contracts.get(name)
            if not isinstance(source_default_contract, Mapping):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} symbol {name!r} has no "
                    "source default contract."
                )
            digest, default_state_sha256 = _attested_callable_code_sha256(
                target,
                module_name=module_name,
                source=source,
                module_globals=module_values,
                qualname=name,
                expected_hashes=_reference_code_hashes(
                    reference,
                    name,
                    module_name=module_name,
                ),
                label=name,
                source_default_contract=source_default_contract,
                state_context=state_context,
                clean_reference=clean_reference,
            )
            records.append(
                {
                    "name": name,
                    "kind": "function",
                    "qualname": name,
                    "code_sha256": digest,
                    "default_state_sha256": default_state_sha256,
                }
            )
        else:
            if not isinstance(target, type):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} symbol {name!r} is not a class."
                )
            raw_members = definition.get("member_qualnames")
            if not isinstance(raw_members, tuple):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} class {name!r} has an invalid "
                    "source member index."
                )
            raw_base_contract = definition.get("class_base_contract")
            if not isinstance(raw_base_contract, Mapping):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} class {name!r} has an invalid "
                    "source base contract."
                )
            raw_static_bindings = definition.get("class_static_bindings")
            if not isinstance(raw_static_bindings, tuple) or not all(
                isinstance(binding, Mapping) for binding in raw_static_bindings
            ):
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} class {name!r} has an invalid "
                    "source class-state index."
                )
            records.append(
                _class_symbol_record(
                    name=name,
                    target=target,
                    module_name=module_name,
                    source=source,
                    module_values=module_values,
                    reference=reference,
                    expected_member_qualnames=raw_members,
                    source_base_contract=raw_base_contract,
                    source_static_bindings=raw_static_bindings,
                    state_plan=state_plan,
                    state_context=state_context,
                    clean_reference=clean_reference,
                )
            )

    source_names = set(definitions)
    for name, target in sorted(module_values.items(), key=lambda item: str(item[0])):
        if str(name) in source_names:
            continue
        if not (inspect.isfunction(target) or inspect.isclass(target)):
            continue
        if str(getattr(target, "__module__", "") or "") != module_name:
            continue
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} exposes an unrecognized local "
            f"symbol {str(name)!r}."
        )

    records.sort(key=lambda record: (str(record["name"]), str(record["kind"])))
    return records


def _runtime_binding_kind(value: Any) -> str:
    """Describe a bound internal import without serializing object identity."""

    if isinstance(value, ModuleType):
        return "module"
    if inspect.isclass(value):
        return "class"
    try:
        if inspect.isfunction(inspect.unwrap(value)):
            return "function"
    except Exception:
        pass
    return f"object:{type(value).__module__}.{type(value).__qualname__}"


def _loaded_module_import_records(
    module_name: str,
    module: ModuleType,
    *,
    reference: Mapping[str, Any],
    state_plan: _SourceStaticStatePlan,
) -> list[dict[str, str]]:
    """Verify that live internal imports still equal source-declared owners."""

    raw_bindings = reference.get("imports")
    raw_state_references = reference.get("module_state_references")
    raw_definitions = reference.get("definitions")
    if not isinstance(raw_bindings, tuple):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid source import index."
        )
    missing = object()
    module_values = vars(module)
    records: list[dict[str, str]] = []
    if (
        not isinstance(raw_state_references, tuple)
        or any(type(name) is not str for name in raw_state_references)
        or not isinstance(raw_definitions, tuple)
    ):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid source state index."
        )
    selected_definition_names = {
        str(record.get("name", "") or "")
        for record in raw_definitions
        if isinstance(record, Mapping)
    }
    bindings_by_local_name: dict[str, list[dict[str, Any]]] = {}
    for binding in raw_bindings:
        if not isinstance(binding, Mapping):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid source import binding."
            )
        local_name = str(binding.get("local_name", "") or "")
        owner_name = str(binding.get("module", "") or "")
        attribute = str(binding.get("attribute", "") or "")
        conditional = binding.get("conditional")
        if (
            not local_name
            or not owner_name
            or type(conditional) is not bool
            or not (
                owner_name == LOADED_PACKAGE_MODULE_PREFIX
                or owner_name.startswith(LOADED_PACKAGE_MODULE_PREFIX + ".")
            )
        ):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} has an invalid internal import binding."
            )
        bindings_by_local_name.setdefault(local_name, []).append(
            {
                "local_name": local_name,
                "module": owner_name,
                "attribute": attribute,
                "conditional": conditional,
            }
        )

    for local_name, candidates in sorted(bindings_by_local_name.items()):
        actual = module_values.get(local_name, missing)
        # A branch-local optional import may legitimately not have produced a
        # binding. If it did, it must still be the exact owner attribute.
        if actual is missing:
            continue
        matching_candidates: list[tuple[dict[str, Any], Any]] = []
        resolved_candidate = False
        for candidate in candidates:
            owner_name = candidate["module"]
            attribute = candidate["attribute"]
            owner_module = sys.modules.get(owner_name)
            if not isinstance(owner_module, ModuleType):
                continue
            expected = (
                owner_module
                if not attribute
                else vars(owner_module).get(attribute, missing)
            )
            if expected is missing:
                continue
            resolved_candidate = True
            if actual is expected:
                matching_candidates.append((candidate, actual))
        if not matching_candidates:
            if all(bool(candidate["conditional"]) for candidate in candidates) and (
                local_name in selected_definition_names
                or (
                    local_name in state_plan.assigned_names
                    and local_name not in raw_state_references
                )
            ):
                # A failed guarded import may be replaced by a source fallback
                # that is either itself a fully attested local definition or
                # is used only for postponed annotations.  It is not an active
                # internal dependency; clean branch reconciliation binds the
                # selected fallback before this phase.
                continue
            candidate_text = ", ".join(
                f"{candidate['module']}:{candidate['attribute']}"
                for candidate in candidates
            )
            if not resolved_candidate:
                raise CanonicalExecutionOriginError(
                    f"Loaded tabnetics module {module_name!r} import {local_name!r} has no "
                    f"loaded source-declared internal owner ({candidate_text})."
                )
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics module {module_name!r} runtime import {local_name!r} no "
                "longer matches its source-declared internal owner."
            )
        candidate, _ = sorted(
            matching_candidates,
            key=lambda item: (item[0]["module"], item[0]["attribute"]),
        )[0]
        records.append(
            {
                "local_name": local_name,
                "module": candidate["module"],
                "attribute": candidate["attribute"],
                "target_kind": _runtime_binding_kind(actual),
            }
        )
    records.sort(
        key=lambda record: (
            record["local_name"],
            record["module"],
            record["attribute"],
        )
    )
    return records


def _prepare_loaded_package_module(
    module_name: str,
    module: Any,
    *,
    package_identity: Mapping[str, str],
) -> _PreparedLoadedModule:
    """Read source-only data needed before the global code/origin phase."""

    if type(module) is not ModuleType:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} is not an exact module object."
        )
    declared_name = str(getattr(module, "__name__", "") or "").strip()
    if declared_name != module_name:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module key/name mismatch: key={module_name!r} "
            f"module={declared_name!r}."
        )
    package_root = Path(str(package_identity.get("package_root", "") or "")).resolve()
    source_path_raw = str(getattr(module, "__file__", "") or "").strip()
    if not source_path_raw:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has no source origin."
        )
    source = Path(source_path_raw).resolve()
    if not source.is_file() or not _is_within(source, package_root):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} source is outside the verified package root."
        )
    if source not in _expected_module_paths(module_name, package_root):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} source does not match its module name."
        )
    source_sha256 = sha256_file(source)
    is_package = source.name == "__init__.py"
    reference = _source_symbol_reference(
        str(source),
        source_sha256,
        module_name,
        is_package,
    )
    reference = _resolve_source_symbol_reference(
        reference,
        module_name=module_name,
        module=module,
        source=source,
    )
    return _PreparedLoadedModule(
        module_name=module_name,
        module=module,
        source=source,
        source_sha256=source_sha256,
        is_package=is_package,
        reference=reference,
        state_plan=_source_static_state_plan(reference, module_name=module_name),
    )


def _loaded_package_module_record(
    module_name: str,
    module: Any,
    *,
    package_identity: Mapping[str, str],
    prepared: _PreparedLoadedModule,
    state_context: _StateSerializationContext,
    clean_reference: _CleanIsolatedReference | None,
) -> dict[str, Any]:
    """Validate one loaded internal module without trusting mutable metadata.

    Canonical benchmark runs attest the complete loaded ``tabnetics.*`` graph
    at finalization.  A hand-maintained list of lazy imports cannot cover code
    that was already contaminated before the runner itself was imported.
    """

    if not isinstance(module, ModuleType):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} is not a module object."
        )
    declared_name = str(getattr(module, "__name__", "") or "").strip()
    if declared_name != module_name:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module key/name mismatch: key={module_name!r} "
            f"module={declared_name!r}."
        )

    package_root = Path(str(package_identity.get("package_root", "") or "")).resolve()
    source_path_raw = str(getattr(module, "__file__", "") or "").strip()
    if not source_path_raw:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has no source origin."
        )
    source = Path(source_path_raw).resolve()
    if not source.is_file():
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} source is unavailable: {source}."
        )
    if not _is_within(source, package_root):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} source is outside the verified "
            f"package root: {source}."
        )
    expected_paths = _expected_module_paths(module_name, package_root)
    if source not in expected_paths:
        expected_text = ", ".join(str(path) for path in sorted(expected_paths))
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} source does not match its module "
            f"name: got {source}; expected one of {expected_text}."
        )

    spec = getattr(module, "__spec__", None)
    if not isinstance(spec, ModuleSpec):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has a missing or synthetic import spec."
        )
    if str(spec.name or "") != module_name:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} import-spec name does not match."
        )
    spec_origin_raw = str(spec.origin or "").strip()
    if not spec_origin_raw or spec_origin_raw in {"built-in", "frozen", "namespace"}:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has a synthetic import-spec origin."
        )
    spec_origin = Path(spec_origin_raw).resolve()
    if spec_origin != source:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} import-spec origin does not match "
            f"its source: got {spec_origin}; expected {source}."
        )
    loader = spec.loader
    if not isinstance(loader, SourceFileLoader):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has a non-source import loader."
        )
    if getattr(module, "__loader__", None) is not loader:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} loader does not match its import spec."
        )
    if str(getattr(loader, "name", "") or "") != module_name:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} source loader name does not match."
        )
    try:
        loader_path = Path(str(loader.get_filename(module_name))).resolve()
    except Exception as exc:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} source loader has no valid filename."
        ) from exc
    if loader_path != source:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} loader filename does not match its "
            f"source: got {loader_path}; expected {source}."
        )

    is_package = source.name == "__init__.py"
    locations = spec.submodule_search_locations
    if is_package:
        if locations is None:
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics package {module_name!r} has no submodule search path."
            )
        resolved_locations = tuple(Path(str(item)).resolve() for item in locations)
        if resolved_locations != (source.parent,):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics package {module_name!r} has an invalid submodule search path."
            )
        module_paths = getattr(module, "__path__", None)
        if (
            module_paths is None
            or tuple(Path(str(item)).resolve() for item in module_paths)
            != resolved_locations
        ):
            raise CanonicalExecutionOriginError(
                f"Loaded tabnetics package {module_name!r} path does not match its import spec."
            )
    elif locations is not None:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} unexpectedly declares package paths."
        )
    expected_package = module_name if is_package else module_name.rpartition(".")[0]
    if str(getattr(module, "__package__", "") or "") != expected_package:
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} has an invalid package declaration."
        )

    source_sha256 = sha256_file(source)
    if (
        prepared.module_name != module_name
        or prepared.module is not module
        or prepared.source != source
        or prepared.source_sha256 != source_sha256
        or prepared.is_package is not is_package
    ):
        raise CanonicalExecutionOriginError(
            f"Loaded tabnetics module {module_name!r} changed before state attestation."
        )
    reference = prepared.reference
    state = _attest_module_runtime_state(
        module_name=module_name,
        module=module,
        reference=reference,
        state_plan=prepared.state_plan,
        state_context=state_context,
        clean_reference=clean_reference,
    )
    if clean_reference is None:
        raise CanonicalExecutionOriginError(
            "Canonical execution is missing clean isolated-reference defaults."
        )
    symbols = _loaded_module_symbol_records(
        module_name,
        module,
        source=source,
        source_sha256=source_sha256,
        reference=reference,
        state_plan=prepared.state_plan,
        state_context=state_context,
        clean_reference=clean_reference,
    )
    imports = _loaded_module_import_records(
        module_name,
        module,
        reference=reference,
        state_plan=prepared.state_plan,
    )
    return {
        "module": module_name,
        "path": str(source),
        "sha256": source_sha256,
        "spec_origin": str(spec_origin),
        "loader": _loader_type_name(loader),
        "is_package": is_package,
        "state": state,
        "state_sha256": canonical_json_sha256(state),
        "symbols": symbols,
        "symbols_sha256": canonical_json_sha256(symbols),
        "imports": imports,
        "imports_sha256": canonical_json_sha256(imports),
    }


def build_loaded_tabnetics_module_closure(
    *,
    package_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Capture the complete loaded internal execution closure at finalization.

    This deliberately covers only loaded ``tabnetics`` package code.  Python
    packages outside the library are not treated as trusted implementation
    targets by this local artifact contract. A principal able to forge module
    objects, import specs/loaders, and every output artifact in one process can
    still create a self-consistent record; that stronger threat model requires
    signed external attestation rather than local provenance fields.
    """

    _assert_trusted_builtins_binding()
    _assert_trusted_logging_binding()
    _assert_trusted_verifier_import_bindings()
    prefix = LOADED_PACKAGE_MODULE_PREFIX
    snapshot = tuple(sys.modules.items())
    entries = sorted(
        (
            (str(module_name), module)
            for module_name, module in snapshot
            if str(module_name) == prefix or str(module_name).startswith(prefix + ".")
        ),
        key=lambda item: item[0],
    )
    if not entries or entries[0][0] != prefix:
        raise CanonicalExecutionOriginError(
            "Loaded tabnetics module closure does not contain the root package."
        )
    prepared_modules = tuple(
        _prepare_loaded_package_module(
            module_name,
            module,
            package_identity=package_identity,
        )
        for module_name, module in entries
    )
    # No runtime state is read until every loaded source symbol has passed a
    # code/origin-only phase.  This prevents a hostile registry/table value
    # from executing while the verifier is still deciding what it trusts.
    for prepared in prepared_modules:
        _validate_loaded_module_code_origin(prepared)
    package_root = Path(str(package_identity.get("package_root", "") or "")).resolve()
    state_context = _build_state_serialization_context(
        prepared_modules,
        package_root=package_root,
        repository_root=_editable_checkout_repository_root(package_root),
    )
    clean_request, clean_expected = _clean_reference_request(
        prepared_modules,
        package_root=package_root,
    )
    clean_reference = _run_clean_isolated_reference(clean_request, clean_expected)
    if clean_reference is None:
        raise CanonicalExecutionOriginError(
            "Canonical execution is missing clean isolated-reference evidence."
        )
    for prepared in prepared_modules:
        _attest_module_declared_dependency_bindings(
            prepared,
            clean_reference=clean_reference,
        )
        _attest_module_builtin_state(prepared, clean_reference=clean_reference)
    clean_reference_record = _clean_reference_record(clean_reference)
    records = [
        _loaded_package_module_record(
            prepared.module_name,
            prepared.module,
            package_identity=package_identity,
            prepared=prepared,
            state_context=state_context,
            clean_reference=clean_reference,
        )
        for prepared in prepared_modules
    ]
    # Do not serialize a mixed graph if another thread mutates ``sys.modules``
    # while the closure is being hashed.
    current_entries = {
        str(module_name): module
        for module_name, module in tuple(sys.modules.items())
        if str(module_name) == prefix or str(module_name).startswith(prefix + ".")
    }
    expected_entries = dict(entries)
    if set(current_entries) != set(expected_entries) or any(
        current_entries.get(module_name) is not module
        for module_name, module in expected_entries.items()
    ):
        raise CanonicalExecutionOriginError(
            "Loaded tabnetics module closure changed while it was being captured."
        )
    symbol_digest_records = [
        {
            "module": str(record["module"]),
            "symbols_sha256": str(record["symbols_sha256"]),
        }
        for record in records
    ]
    return {
        "schema_version": LOADED_PACKAGE_MODULE_CLOSURE_SCHEMA_VERSION,
        "package_prefix": prefix,
        "modules": records,
        "modules_sha256": canonical_json_sha256(records),
        "symbols_sha256": canonical_json_sha256(symbol_digest_records),
        "clean_reference": clean_reference_record,
        "clean_reference_sha256": canonical_json_sha256(clean_reference_record),
    }


def capture_loaded_tabnetics_module_closure(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and snapshot the current process's loaded internal modules."""

    package_identity = _verified_tabnetics_package_identity(repo_root=repo_root)
    return build_loaded_tabnetics_module_closure(package_identity=package_identity)


def _merge_clean_reference_records(
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge worker clean-reference evidence without losing exercised dependencies."""

    shared_keys = (
        "schema_version",
        "source_tree_sha256",
        "environment",
        "isolated_paths",
        "python",
        "platform",
    )
    if any(current.get(key) != candidate.get(key) for key in shared_keys):
        raise CanonicalExecutionOriginError(
            "Execution-worker clean isolated-reference evidence conflicts across closures."
        )
    current_dependencies = current.get("dependencies")
    candidate_dependencies = candidate.get("dependencies")
    if not isinstance(current_dependencies, Mapping) or not isinstance(
        candidate_dependencies, Mapping
    ):
        raise CanonicalExecutionOriginError(
            "Execution-worker clean isolated-reference dependency evidence is invalid."
        )
    dependencies: dict[str, Any] = {}
    for name, value in current_dependencies.items():
        if type(name) is not str:
            raise CanonicalExecutionOriginError(
                "Execution-worker clean isolated-reference dependency evidence is invalid."
            )
        dependencies[name] = value
    for name, value in candidate_dependencies.items():
        if type(name) is not str:
            raise CanonicalExecutionOriginError(
                "Execution-worker clean isolated-reference dependency evidence is invalid."
            )
        existing = dependencies.get(name, _MISSING_RUNTIME_BINDING)
        if existing is not _MISSING_RUNTIME_BINDING and existing != value:
            raise CanonicalExecutionOriginError(
                "Execution-worker clean isolated-reference dependency evidence conflicts "
                f"for {name!r}."
            )
        dependencies[name] = value
    return {
        **{key: current[key] for key in shared_keys},
        "dependencies": {name: dependencies[name] for name in sorted(dependencies)},
    }


def merge_loaded_tabnetics_module_closures(
    *,
    package_identity: Mapping[str, str],
    worker_closures: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge the controller and every execution-worker module closure.

    Joblib and hard-timeout workers have independent ``sys.modules`` tables.
    Their individually validated snapshots therefore need to be unioned before
    a controller can attest the code that actually executed a benchmark row.
    """

    package_root = Path(str(package_identity.get("package_root", "") or "")).resolve()
    provided_closures = tuple(worker_closures or ())
    closures: list[Mapping[str, Any]] = [
        build_loaded_tabnetics_module_closure(package_identity=package_identity)
    ]
    closures.extend(
        closure for closure in provided_closures if isinstance(closure, Mapping)
    )
    if len(closures) != 1 + len(provided_closures):
        raise CanonicalExecutionOriginError(
            "Execution-worker module closure is not a mapping."
        )

    records_by_module: dict[str, dict[str, Any]] = {}
    clean_reference_record: Mapping[str, Any] | None = None
    for closure in closures:
        closure_digest = str(closure.get("modules_sha256", "") or "")
        symbol_digest = str(closure.get("symbols_sha256", "") or "")
        validation_payload = {
            "loaded_package_modules": closure,
            "loaded_package_modules_sha256": closure_digest,
            "loaded_package_symbols_sha256": symbol_digest,
        }
        reason = _loaded_package_module_closure_reason(
            validation_payload,
            package_root=package_root,
            source_revision={
                "loaded_package_modules_sha256": closure_digest,
                "loaded_package_symbols_sha256": symbol_digest,
            },
        )
        if reason:
            raise CanonicalExecutionOriginError(
                "Execution-worker loaded tabnetics module closure is invalid: "
                f"{reason}."
            )
        candidate_clean_reference = closure.get("clean_reference")
        if not isinstance(candidate_clean_reference, Mapping):
            raise CanonicalExecutionOriginError(
                "Execution-worker clean isolated-reference evidence is missing."
            )
        if clean_reference_record is None:
            clean_reference_record = dict(candidate_clean_reference)
        else:
            clean_reference_record = _merge_clean_reference_records(
                clean_reference_record,
                candidate_clean_reference,
            )
        for raw_record in closure["modules"]:
            assert isinstance(raw_record, Mapping)  # Validated above.
            record = {str(key): value for key, value in raw_record.items()}
            module_name = str(record["module"])
            existing = records_by_module.get(module_name)
            if existing is not None and existing != record:
                raise CanonicalExecutionOriginError(
                    "Execution-worker loaded tabnetics module closure conflicts for "
                    f"{module_name!r}."
                )
            records_by_module[module_name] = record
    records = [records_by_module[name] for name in sorted(records_by_module)]
    symbol_digest_records = [
        {
            "module": str(record["module"]),
            "symbols_sha256": str(record["symbols_sha256"]),
        }
        for record in records
    ]
    return {
        "schema_version": LOADED_PACKAGE_MODULE_CLOSURE_SCHEMA_VERSION,
        "package_prefix": LOADED_PACKAGE_MODULE_PREFIX,
        "modules": records,
        "modules_sha256": canonical_json_sha256(records),
        "symbols_sha256": canonical_json_sha256(symbol_digest_records),
        "clean_reference": dict(clean_reference_record or {}),
        "clean_reference_sha256": canonical_json_sha256(
            dict(clean_reference_record or {})
        ),
    }


def validate_loaded_tabnetics_module_closure(
    *,
    repo_root: str | Path | None = None,
) -> None:
    """Fail fast on invalid already-loaded internal code before tasks start.

    This preflight makes no canonical-evidence claim and records no artifact.
    The finalization-time closure is rebuilt after materialization and is the
    only closure bound into output rows and artifacts.
    """

    capture_loaded_tabnetics_module_closure(repo_root=repo_root)


def _array_content_identity(values: Any) -> dict[str, Any]:
    """Hash an array's values and layout without retaining raw dataset values."""

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - canonical runner requires numpy.
        raise CanonicalExecutionInputIdentityError(
            "Canonical materialized-input identity requires numpy."
        ) from exc

    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    header = {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
    }
    kind = str(array.dtype.kind)
    if kind == "f":
        canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        nan_mask = np.isnan(canonical)
        if bool(np.any(nan_mask)):
            canonical = canonical.copy()
            canonical[nan_mask] = 0.0
        digest = hashlib.sha256()
        digest.update(canonical_json_sha256(header).encode("ascii"))
        digest.update(np.ascontiguousarray(nan_mask, dtype=np.uint8).tobytes())
        digest.update(canonical.tobytes())
        return {
            **header,
            "encoding": "float64_with_nan_mask",
            "sha256": digest.hexdigest(),
        }
    if kind in {"i", "u", "b"}:
        if kind == "u":
            canonical = np.ascontiguousarray(np.asarray(array, dtype="<u8"))
        elif kind == "b":
            canonical = np.ascontiguousarray(np.asarray(array, dtype=np.uint8))
        else:
            canonical = np.ascontiguousarray(np.asarray(array, dtype="<i8"))
        digest = hashlib.sha256()
        digest.update(canonical_json_sha256(header).encode("ascii"))
        digest.update(canonical.tobytes())
        return {**header, "encoding": "canonical_numeric", "sha256": digest.hexdigest()}
    return {
        **header,
        "encoding": "stable_json",
        "sha256": canonical_json_sha256(
            {
                **header,
                "values": _stable_json_value(array.tolist()),
            }
        ),
    }


def _materialized_input_digest_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in record.items()
        if str(key) != "materialized_input_sha256"
    }


def build_materialized_dataset_input_identity(
    *,
    dataset_id: str,
    seed: int,
    data_source: str,
    source_identity: Mapping[str, Any],
    X: Any,
    y: Any,
    feature_names: Sequence[Any] | None = None,
    batch_labels: Any | None = None,
    split_fingerprints: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a deterministic identity for the exact matrix/labels a task used."""

    x_identity = _array_content_identity(X)
    y_identity = _array_content_identity(y)
    if int(x_identity["shape"][0]) != int(y_identity["shape"][0]):
        raise CanonicalExecutionInputIdentityError(
            "Materialized input X/y rows do not match."
        )
    n_features = int(x_identity["shape"][1]) if len(x_identity["shape"]) >= 2 else 0
    if feature_names is None:
        feature_order = {
            "kind": "matrix_column_positions",
            "n_features": n_features,
            "sha256": canonical_json_sha256(
                {"kind": "matrix_column_positions", "n_features": n_features}
            ),
        }
    else:
        names = [str(value) for value in feature_names]
        if len(names) != n_features or len(set(names)) != len(names):
            raise CanonicalExecutionInputIdentityError(
                "Materialized input feature names do not uniquely match X columns."
            )
        feature_order = {
            "kind": "feature_names",
            "n_features": n_features,
            "sha256": canonical_json_sha256(names),
        }
    normalized_splits = sorted(
        {str(value) for value in split_fingerprints if str(value)}
    )
    record: dict[str, Any] = {
        "schema_version": MATERIALIZED_INPUT_IDENTITY_SCHEMA_VERSION,
        "dataset_id": str(dataset_id),
        "seed": int(seed),
        "data_source": str(data_source),
        "source_identity": _stable_json_value(dict(source_identity)),
        "x": x_identity,
        "y": y_identity,
        "feature_order": feature_order,
        "batch_labels": (
            None if batch_labels is None else _array_content_identity(batch_labels)
        ),
        "split_fingerprints": normalized_splits,
        "split_fingerprints_sha256": canonical_json_sha256(normalized_splits),
    }
    record["materialized_input_sha256"] = canonical_json_sha256(
        _materialized_input_digest_payload(record)
    )
    return record


def bind_materialized_input_split_fingerprints(
    identity: Mapping[str, Any],
    split_fingerprints: Sequence[str],
) -> dict[str, Any]:
    """Return a materialized-input record bound to the task's resolved splits."""

    record = {
        str(key): value
        for key, value in identity.items()
        if str(key) != "materialized_input_sha256"
    }
    if str(record.get("schema_version", "") or "") != (
        MATERIALIZED_INPUT_IDENTITY_SCHEMA_VERSION
    ):
        raise CanonicalExecutionInputIdentityError(
            "Cannot bind split fingerprints to an unsupported materialized-input identity."
        )
    normalized_splits = sorted(
        {str(value) for value in split_fingerprints if str(value)}
    )
    record["split_fingerprints"] = normalized_splits
    record["split_fingerprints_sha256"] = canonical_json_sha256(normalized_splits)
    record["materialized_input_sha256"] = canonical_json_sha256(
        _materialized_input_digest_payload(record)
    )
    return record


def _validate_materialized_input_records(
    *,
    records: Sequence[Mapping[str, Any]] | None,
    selected_dataset_ids: Sequence[str],
    expected_seeds: Sequence[Any],
) -> list[dict[str, Any]]:
    if records is None:
        raise CanonicalExecutionInputIdentityError(
            "Canonical execution requires materialized X/y input identity records."
        )
    try:
        expected_keys = {
            (str(dataset_id), int(seed))
            for dataset_id in selected_dataset_ids
            for seed in expected_seeds
        }
    except (TypeError, ValueError) as exc:
        raise CanonicalExecutionInputIdentityError(
            "Canonical execution seed configuration is invalid."
        ) from exc
    if not expected_keys:
        raise CanonicalExecutionInputIdentityError(
            "Canonical execution requires at least one dataset/seed materialization key."
        )
    normalized: list[dict[str, Any]] = []
    observed_keys: set[tuple[str, int]] = set()
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input record is malformed."
            )
        record = {str(key): value for key, value in raw_record.items()}
        if str(record.get("schema_version", "") or "") != (
            MATERIALIZED_INPUT_IDENTITY_SCHEMA_VERSION
        ):
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input record has an unsupported schema."
            )
        try:
            key = (str(record["dataset_id"]), int(record["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input record lacks dataset/seed identity."
            ) from exc
        if key in observed_keys:
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution has duplicate materialized-input records."
            )
        observed_keys.add(key)
        for required_key in (
            "data_source",
            "source_identity",
            "x",
            "y",
            "feature_order",
        ):
            if required_key not in record:
                raise CanonicalExecutionInputIdentityError(
                    "Canonical execution materialized-input record is incomplete: "
                    f"missing {required_key!r}."
                )
        if not str(record.get("data_source", "") or ""):
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input record lacks a data source."
            )
        if not isinstance(record.get("source_identity"), Mapping):
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input source identity is invalid."
            )
        x_identity = record.get("x")
        y_identity = record.get("y")
        if not isinstance(x_identity, Mapping) or not isinstance(y_identity, Mapping):
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input X/y identities are invalid."
            )
        x_shape = x_identity.get("shape")
        y_shape = y_identity.get("shape")
        if (
            not isinstance(x_shape, list)
            or not isinstance(y_shape, list)
            or len(x_shape) < 2
            or len(y_shape) < 1
            or not str(x_identity.get("sha256", "") or "")
            or not str(y_identity.get("sha256", "") or "")
        ):
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input X/y identity fields are incomplete."
            )
        try:
            n_rows = int(x_shape[0])
            n_features = int(x_shape[1])
            y_rows = int(y_shape[0])
        except (TypeError, ValueError) as exc:
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input X/y shapes are invalid."
            ) from exc
        if n_rows != y_rows or n_rows < 1 or n_features < 1:
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input X/y shapes are inconsistent."
            )
        feature_order = record.get("feature_order")
        if (
            not isinstance(feature_order, Mapping)
            or int(feature_order.get("n_features", -1) or -1) != n_features
            or not str(feature_order.get("sha256", "") or "")
        ):
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input feature-order identity is invalid."
            )
        batch_identity = record.get("batch_labels")
        if batch_identity is not None and (
            not isinstance(batch_identity, Mapping)
            or not str(batch_identity.get("sha256", "") or "")
        ):
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input batch-label identity is invalid."
            )
        splits = record.get("split_fingerprints")
        if not isinstance(splits, list) or str(
            record.get("split_fingerprints_sha256", "") or ""
        ) != canonical_json_sha256(splits):
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input split identity is invalid."
            )
        recorded_digest = str(record.get("materialized_input_sha256", "") or "")
        if not recorded_digest or recorded_digest != canonical_json_sha256(
            _materialized_input_digest_payload(record)
        ):
            raise CanonicalExecutionInputIdentityError(
                "Canonical execution materialized-input record digest is invalid."
            )
        normalized.append(record)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise CanonicalExecutionInputIdentityError(
            "Canonical execution materialized-input coverage is incomplete: "
            f"missing={missing!r} unexpected={unexpected!r}."
        )
    return sorted(
        normalized, key=lambda record: (str(record["dataset_id"]), int(record["seed"]))
    )


def _input_data_identity(
    selected_dataset_ids: Sequence[str],
    *,
    materialized_input_records: Sequence[Mapping[str, Any]] | None,
    expected_seeds: Sequence[Any],
) -> dict[str, Any]:
    selected = [str(item) for item in selected_dataset_ids]
    if len(selected) != len(set(selected)):
        raise CanonicalExecutionInputIdentityError(
            "Canonical execution input dataset ids must be unique."
        )
    registry_identity = dataset_registry_identity(selected)
    if not bool(registry_identity.get("registry_available", False)):
        raise CanonicalExecutionInputIdentityError(
            "Canonical execution cannot resolve the tabnetics dataset registry."
        )
    records = list(registry_identity.get("datasets") or ())
    if len(records) != len(selected) or any(
        not isinstance(record, Mapping) or not bool(record.get("registered", False))
        for record in records
    ):
        raise CanonicalExecutionInputIdentityError(
            "Canonical execution has an unregistered benchmark input dataset."
        )
    materialized_records = _validate_materialized_input_records(
        records=materialized_input_records,
        selected_dataset_ids=selected,
        expected_seeds=expected_seeds,
    )
    return {
        "schema_version": INPUT_DATA_IDENTITY_SCHEMA_VERSION,
        "selected_dataset_ids": selected,
        "dataset_registry": registry_identity,
        "materialized_inputs": materialized_records,
        "materialized_inputs_sha256": canonical_json_sha256(materialized_records),
    }


def build_canonical_execution_provenance(
    *,
    args: Mapping[str, Any] | Any,
    selected_dataset_ids: Sequence[str],
    import_targets: Mapping[str, Any],
    expected_import_targets: Mapping[str, Any] | None = None,
    materialized_input_records: Sequence[Mapping[str, Any]] | None = None,
    worker_module_closures: Sequence[Mapping[str, Any]] | None = None,
    worker_module_closures_complete: bool = True,
    command: Sequence[str] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed identity contract for a core benchmark execution.

    This is deliberately separate from the broader host/input provenance:
    scorecard consumers need a compact, row-level assertion that the evaluated
    code came from the packaged ``tabnetics`` stack rather than a similarly
    named historical runner under ``experiments``.
    """

    label_reason = _bootstrap_import_labels_reason(import_targets)
    if label_reason:
        raise CanonicalExecutionOriginError(
            f"Canonical execution bootstrap contract is incomplete: {label_reason}."
        )
    if expected_import_targets is not None:
        expected_label_reason = _bootstrap_import_labels_reason(expected_import_targets)
        if expected_label_reason:
            raise CanonicalExecutionOriginError(
                "Canonical execution captured bootstrap contract is incomplete: "
                f"{expected_label_reason}."
            )

    resolved_args = {
        str(key): _stable_json_value(value)
        for key, value in sorted(
            _namespace_mapping(args).items(), key=lambda item: str(item[0])
        )
        if not str(key).startswith("_")
    }
    resolved_cli_config = {
        "arguments": resolved_args,
        "selected_dataset_ids": [str(item) for item in selected_dataset_ids],
        "command": [str(item) for item in list(command or ())],
    }
    external_callable_reason = external_callable_identity_unattested_reason(args)
    if external_callable_reason:
        raise CanonicalExecutionExternalDependencyError(external_callable_reason)
    package_identity = _verified_tabnetics_package_identity(repo_root=repo_root)
    import_origins = {
        str(label): _module_origin_record(
            str(label),
            target,
            package_identity=package_identity,
            expected_target=(
                expected_import_targets.get(str(label))
                if expected_import_targets is not None
                else None
            ),
        )
        for label, target in sorted(
            import_targets.items(), key=lambda item: str(item[0])
        )
    }
    seeds = resolved_args.get("seeds")
    if not isinstance(seeds, (list, tuple)):
        raise CanonicalExecutionInputIdentityError(
            "Canonical execution resolved CLI config does not contain a seed list."
        )
    input_data_identity = _input_data_identity(
        selected_dataset_ids,
        materialized_input_records=materialized_input_records,
        expected_seeds=seeds,
    )
    if worker_module_closures_complete is not True:
        raise CanonicalExecutionOriginError(
            "Canonical execution is missing one or more execution-worker module closures."
        )
    git = git_provenance(repo_root)
    packages = package_versions()
    # This is deliberately the final tabnetics import/origin operation before
    # emitting the contract: it captures modules loaded while materializing the
    # completed benchmark rather than an import-time approximation.
    loaded_package_modules = merge_loaded_tabnetics_module_closures(
        package_identity=package_identity,
        worker_closures=worker_module_closures,
    )
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_PROVENANCE_SCHEMA_VERSION,
        "implementation_stack": CANONICAL_IMPLEMENTATION_STACK,
        "evidence_status": CANONICAL_EVIDENCE_STATUS,
        "canonical_scorecard_eligible": True,
        "worker_module_closures_complete": True,
        "resolved_cli_config": resolved_cli_config,
        "resolved_cli_config_sha256": canonical_json_sha256(resolved_cli_config),
        "input_data_identity": input_data_identity,
        "input_data_identity_sha256": canonical_json_sha256(input_data_identity),
        "package_identity": package_identity,
        "package_identity_sha256": canonical_json_sha256(package_identity),
        "import_origins": import_origins,
        "loaded_package_modules": loaded_package_modules,
        "loaded_package_modules_sha256": str(loaded_package_modules["modules_sha256"]),
        "loaded_package_symbols_sha256": str(loaded_package_modules["symbols_sha256"]),
        "source_revision": {
            "git": git,
            "tabnetics_package_version": str(packages.get("tabnetics", "") or ""),
            "module_sha256": {
                label: str(record["sha256"]) for label, record in import_origins.items()
            },
            "module_hashes_sha256": canonical_json_sha256(
                {
                    label: str(record["sha256"])
                    for label, record in import_origins.items()
                }
            ),
            "loaded_package_modules_sha256": str(
                loaded_package_modules["modules_sha256"]
            ),
            "loaded_package_symbols_sha256": str(
                loaded_package_modules["symbols_sha256"]
            ),
        },
    }
    payload["created_at"] = utc_now_iso()
    payload["fingerprint_sha256"] = execution_fingerprint_sha256(payload)
    return payload


def build_noncanonical_execution_provenance(
    *,
    args: Mapping[str, Any] | Any,
    selected_dataset_ids: Sequence[str],
    import_targets: Mapping[str, Any],
    expected_import_targets: Mapping[str, Any] | None = None,
    materialized_input_records: Sequence[Mapping[str, Any]] | None = None,
    worker_module_closures: Sequence[Mapping[str, Any]] | None = None,
    worker_module_closures_complete: bool = True,
    noncanonical_reason: str,
    evidence_status: str = "noncanonical_input_identity",
    preserve_materialized_input_identity: bool = False,
    command: Sequence[str] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Record a noncanonical run without allowing it into canonical scorecards."""

    allowed_evidence_statuses = {
        "noncanonical_input_identity",
        EXTERNAL_CALLABLE_UNATTESTED_EVIDENCE_STATUS,
    }
    if evidence_status not in allowed_evidence_statuses:
        raise CanonicalExecutionOriginError("Unsupported noncanonical evidence status.")

    label_reason = _bootstrap_import_labels_reason(import_targets)
    if label_reason:
        raise CanonicalExecutionOriginError(
            f"Execution bootstrap contract is incomplete: {label_reason}."
        )
    if expected_import_targets is not None:
        expected_label_reason = _bootstrap_import_labels_reason(expected_import_targets)
        if expected_label_reason:
            raise CanonicalExecutionOriginError(
                "Captured execution bootstrap contract is incomplete: "
                f"{expected_label_reason}."
            )
    resolved_args = {
        str(key): _stable_json_value(value)
        for key, value in sorted(
            _namespace_mapping(args).items(), key=lambda item: str(item[0])
        )
        if not str(key).startswith("_")
    }
    resolved_cli_config = {
        "arguments": resolved_args,
        "selected_dataset_ids": [str(item) for item in selected_dataset_ids],
        "command": [str(item) for item in list(command or ())],
    }
    package_identity = _verified_tabnetics_package_identity(repo_root=repo_root)
    import_origins = {
        str(label): _module_origin_record(
            str(label),
            target,
            package_identity=package_identity,
            expected_target=(
                expected_import_targets.get(str(label))
                if expected_import_targets is not None
                else None
            ),
        )
        for label, target in sorted(
            import_targets.items(), key=lambda item: str(item[0])
        )
    }
    if preserve_materialized_input_identity:
        seeds = resolved_args.get("seeds")
        if not isinstance(seeds, (list, tuple)):
            raise CanonicalExecutionInputIdentityError(
                "Noncanonical external-callable execution requires a seed list."
            )
        input_data_identity = _input_data_identity(
            selected_dataset_ids,
            materialized_input_records=materialized_input_records,
            expected_seeds=seeds,
        )
    else:
        partial_records = [
            _stable_json_value(dict(record))
            for record in list(materialized_input_records or ())
            if isinstance(record, Mapping)
        ]
        partial_records.sort(
            key=lambda record: (
                str(record.get("dataset_id", "")),
                str(record.get("seed", "")),
            )
        )
        input_data_identity = {
            "schema_version": INPUT_DATA_IDENTITY_SCHEMA_VERSION,
            "selected_dataset_ids": [str(item) for item in selected_dataset_ids],
            "dataset_registry": dataset_registry_identity(selected_dataset_ids),
            "materialized_inputs": partial_records,
            "materialized_inputs_sha256": canonical_json_sha256(partial_records),
            "materialization_status": "incomplete_noncanonical",
        }
    git = git_provenance(repo_root)
    packages = package_versions()
    loaded_package_modules = merge_loaded_tabnetics_module_closures(
        package_identity=package_identity,
        worker_closures=worker_module_closures,
    )
    module_hashes = {
        label: str(record["sha256"]) for label, record in import_origins.items()
    }
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_PROVENANCE_SCHEMA_VERSION,
        "implementation_stack": CANONICAL_IMPLEMENTATION_STACK,
        "evidence_status": evidence_status,
        "canonical_scorecard_eligible": False,
        "noncanonical_reason": str(noncanonical_reason),
        "worker_module_closures_complete": bool(worker_module_closures_complete),
        "resolved_cli_config": resolved_cli_config,
        "resolved_cli_config_sha256": canonical_json_sha256(resolved_cli_config),
        "input_data_identity": input_data_identity,
        "input_data_identity_sha256": canonical_json_sha256(input_data_identity),
        "package_identity": package_identity,
        "package_identity_sha256": canonical_json_sha256(package_identity),
        "import_origins": import_origins,
        "loaded_package_modules": loaded_package_modules,
        "loaded_package_modules_sha256": str(loaded_package_modules["modules_sha256"]),
        "loaded_package_symbols_sha256": str(loaded_package_modules["symbols_sha256"]),
        "source_revision": {
            "git": git,
            "tabnetics_package_version": str(packages.get("tabnetics", "") or ""),
            "module_sha256": module_hashes,
            "module_hashes_sha256": canonical_json_sha256(module_hashes),
            "loaded_package_modules_sha256": str(
                loaded_package_modules["modules_sha256"]
            ),
            "loaded_package_symbols_sha256": str(
                loaded_package_modules["symbols_sha256"]
            ),
        },
    }
    payload["created_at"] = utc_now_iso()
    payload["fingerprint_sha256"] = execution_fingerprint_sha256(payload)
    return payload


def execution_row_fields(execution_provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact execution identity copied into materialized rows."""

    input_data_identity = execution_provenance.get("input_data_identity")
    if not isinstance(input_data_identity, Mapping):
        input_data_identity = {}
    source_revision = execution_provenance.get("source_revision")
    if not isinstance(source_revision, Mapping):
        source_revision = {}
    git = source_revision.get("git")
    if not isinstance(git, Mapping):
        git = {}
    import_origins = execution_provenance.get("import_origins")
    if not isinstance(import_origins, Mapping):
        import_origins = {}
    pipeline = import_origins.get("pipeline")
    if not isinstance(pipeline, Mapping):
        pipeline = {}
    return {
        "implementation_stack": str(
            execution_provenance.get("implementation_stack", "")
        ),
        "evidence_status": str(execution_provenance.get("evidence_status", "")),
        "canonical_scorecard_eligible": bool(
            execution_provenance.get("canonical_scorecard_eligible", False)
        ),
        "noncanonical_reason": str(
            execution_provenance.get("noncanonical_reason", "") or ""
        ),
        "execution_provenance_schema": str(
            execution_provenance.get("schema_version", "")
        ),
        "execution_provenance_sha256": str(
            execution_provenance.get("fingerprint_sha256", "")
        ),
        "resolved_cli_config_sha256": str(
            execution_provenance.get("resolved_cli_config_sha256", "")
        ),
        "input_data_identity_sha256": str(
            execution_provenance.get("input_data_identity_sha256", "")
        ),
        "materialized_input_set_sha256": str(
            input_data_identity.get("materialized_inputs_sha256", "")
        ),
        "package_identity_sha256": str(
            execution_provenance.get("package_identity_sha256", "")
        ),
        "source_revision_git_sha": str(git.get("sha", "") or ""),
        "source_revision_tabnetics_version": str(
            source_revision.get("tabnetics_package_version", "") or ""
        ),
        "source_revision_module_hashes_sha256": str(
            source_revision.get("module_hashes_sha256", "") or ""
        ),
        "loaded_package_modules_sha256": str(
            execution_provenance.get("loaded_package_modules_sha256", "") or ""
        ),
        "loaded_package_symbols_sha256": str(
            execution_provenance.get("loaded_package_symbols_sha256", "") or ""
        ),
        "pipeline_import_origin": str(pipeline.get("path", "") or ""),
    }


def _is_sha256_digest(value: Any) -> bool:
    digest = str(value or "")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _loaded_package_module_closure_reason(
    execution: Mapping[str, Any],
    *,
    package_root: Path,
    source_revision: Mapping[str, Any],
) -> str:
    closure = execution.get("loaded_package_modules")
    if not isinstance(closure, Mapping):
        return "loaded_package_modules_missing"
    if str(closure.get("schema_version", "") or "") != (
        LOADED_PACKAGE_MODULE_CLOSURE_SCHEMA_VERSION
    ):
        return "loaded_package_modules_schema_missing_or_unsupported"
    if str(closure.get("package_prefix", "") or "") != LOADED_PACKAGE_MODULE_PREFIX:
        return "loaded_package_modules_prefix_mismatch"
    records = closure.get("modules")
    if not isinstance(records, list) or not records:
        return "loaded_package_modules_records_missing"
    record_digest = str(closure.get("modules_sha256", "") or "")
    if not _is_sha256_digest(record_digest):
        return "loaded_package_modules_digest_missing_or_invalid"
    if record_digest != canonical_json_sha256(records):
        return "loaded_package_modules_digest_mismatch"
    if str(execution.get("loaded_package_modules_sha256", "") or "") != record_digest:
        return "loaded_package_modules_execution_digest_mismatch"
    if (
        str(source_revision.get("loaded_package_modules_sha256", "") or "")
        != record_digest
    ):
        return "loaded_package_modules_source_revision_digest_mismatch"
    symbol_digest = str(closure.get("symbols_sha256", "") or "")
    if not _is_sha256_digest(symbol_digest):
        return "loaded_package_symbols_digest_missing_or_invalid"
    if str(execution.get("loaded_package_symbols_sha256", "") or "") != symbol_digest:
        return "loaded_package_symbols_execution_digest_mismatch"
    if (
        str(source_revision.get("loaded_package_symbols_sha256", "") or "")
        != symbol_digest
    ):
        return "loaded_package_symbols_source_revision_digest_mismatch"
    clean_reference = closure.get("clean_reference")
    clean_reference_digest = str(closure.get("clean_reference_sha256", "") or "")
    if not isinstance(clean_reference, Mapping) or not _is_sha256_digest(
        clean_reference_digest
    ):
        return "loaded_package_clean_reference_missing_or_invalid"
    if clean_reference_digest != canonical_json_sha256(clean_reference):
        return "loaded_package_clean_reference_digest_mismatch"
    expected_clean_reference_keys = {
        "schema_version",
        "source_tree_sha256",
        "environment",
        "isolated_paths",
        "python",
        "platform",
        "dependencies",
    }
    if set(clean_reference) != expected_clean_reference_keys:
        return "loaded_package_clean_reference_shape_invalid"
    if (
        clean_reference.get("schema_version")
        != _CLEAN_ISOLATED_REFERENCE_SCHEMA_VERSION
    ):
        return "loaded_package_clean_reference_schema_invalid"
    if not _is_sha256_digest(clean_reference.get("source_tree_sha256")):
        return "loaded_package_clean_reference_source_tree_invalid"
    environment = clean_reference.get("environment")
    isolated_paths = clean_reference.get("isolated_paths")
    python_record = clean_reference.get("python")
    platform_record = clean_reference.get("platform")
    dependencies = clean_reference.get("dependencies")
    if (
        not isinstance(environment, Mapping)
        or any(
            type(key) is not str or type(value) is not str
            for key, value in environment.items()
        )
        or not isinstance(isolated_paths, Mapping)
        or set(isolated_paths) != set(_CLEAN_ISOLATED_REFERENCE_PATH_KEYS)
        or any(
            type(key) is not str or type(value) is not str
            for key, value in isolated_paths.items()
        )
        or isolated_paths.get("source_package_root") != "verified_temp_source_copy"
        or any(
            isolated_paths.get(name) != f"child_owned:{name}"
            for name in _CLEAN_ISOLATED_REFERENCE_PATH_KEYS
            if name != "source_package_root"
        )
        or not isinstance(python_record, Mapping)
        or set(python_record) != {"implementation", "version", "executable"}
        or not isinstance(platform_record, Mapping)
        or set(platform_record) != {"system", "machine"}
        or not isinstance(dependencies, Mapping)
    ):
        return "loaded_package_clean_reference_runtime_invalid"
    for dependency_name, dependency_record in dependencies.items():
        if (
            type(dependency_name) is not str
            or not isinstance(dependency_record, Mapping)
            or set(dependency_record)
            != {"name", "origin", "origin_sha256", "version", "api_sha256"}
            or dependency_record.get("name") != dependency_name
            or any(type(value) is not str for value in dependency_record.values())
        ):
            return "loaded_package_clean_reference_dependency_invalid"

    expected_record_keys = {
        "module",
        "path",
        "sha256",
        "spec_origin",
        "loader",
        "is_package",
        "state",
        "state_sha256",
        "symbols",
        "symbols_sha256",
        "imports",
        "imports_sha256",
    }
    module_names: list[str] = []
    symbol_digest_records: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_record_keys:
            return "loaded_package_module_record_invalid"
        module_name = str(record.get("module", "") or "")
        if not (
            module_name == LOADED_PACKAGE_MODULE_PREFIX
            or module_name.startswith(LOADED_PACKAGE_MODULE_PREFIX + ".")
        ):
            return "loaded_package_module_name_outside_prefix"
        module_names.append(module_name)
        path_raw = str(record.get("path", "") or "").strip()
        spec_origin_raw = str(record.get("spec_origin", "") or "").strip()
        if not path_raw or not spec_origin_raw:
            return "loaded_package_module_path_or_spec_missing"
        path = Path(path_raw).resolve()
        spec_origin = Path(spec_origin_raw).resolve()
        if path != spec_origin:
            return "loaded_package_module_spec_origin_mismatch"
        if not _is_within(path, package_root):
            return "loaded_package_module_path_outside_package_root"
        if path not in _expected_module_paths(module_name, package_root):
            return "loaded_package_module_path_not_canonical"
        if not _is_sha256_digest(record.get("sha256")):
            return "loaded_package_module_source_hash_missing_or_invalid"
        loader = str(record.get("loader", "") or "")
        if not loader.endswith(".SourceFileLoader"):
            return "loaded_package_module_loader_not_source_file_loader"
        is_package = record.get("is_package")
        if not isinstance(is_package, bool) or is_package != (
            path.name == "__init__.py"
        ):
            return "loaded_package_module_package_flag_mismatch"
        state = record.get("state")
        if not isinstance(state, list):
            return "loaded_package_module_state_missing"
        state_digest = str(record.get("state_sha256", "") or "")
        if not _is_sha256_digest(state_digest):
            return "loaded_package_module_state_digest_missing_or_invalid"
        if state_digest != canonical_json_sha256(state):
            return "loaded_package_module_state_digest_mismatch"
        previous_state_key: tuple[str, str] | None = None
        for state_record in state:
            if not isinstance(state_record, Mapping) or set(state_record) != {
                "owner",
                "name",
                "type",
                "semantic_sha256",
                "provider",
                "provider_sha256",
            }:
                return "loaded_package_module_state_record_invalid"
            owner = str(state_record.get("owner", "") or "")
            state_name = str(state_record.get("name", "") or "")
            state_type = str(state_record.get("type", "") or "")
            semantic_digest = str(state_record.get("semantic_sha256", "") or "")
            provider = str(state_record.get("provider", "") or "")
            provider_digest = str(state_record.get("provider_sha256", "") or "")
            if (
                owner != "module"
                or not state_name
                or not state_type
                or not _is_sha256_digest(semantic_digest)
            ):
                return "loaded_package_module_state_identity_invalid"
            if bool(provider) != bool(provider_digest):
                return "loaded_package_module_state_provider_missing_or_invalid"
            if provider and not _is_sha256_digest(provider_digest):
                return "loaded_package_module_state_provider_digest_invalid"
            state_key = (owner, state_name)
            if previous_state_key is not None and state_key <= previous_state_key:
                return "loaded_package_module_state_not_deterministically_sorted"
            previous_state_key = state_key
        symbols = record.get("symbols")
        if not isinstance(symbols, list):
            return "loaded_package_module_symbols_missing"
        record_symbol_digest = str(record.get("symbols_sha256", "") or "")
        if not _is_sha256_digest(record_symbol_digest):
            return "loaded_package_module_symbols_digest_missing_or_invalid"
        if record_symbol_digest != canonical_json_sha256(symbols):
            return "loaded_package_module_symbols_digest_mismatch"
        previous_symbol_key: tuple[str, str] | None = None
        for symbol in symbols:
            if not isinstance(symbol, Mapping):
                return "loaded_package_module_symbol_record_invalid"
            name = str(symbol.get("name", "") or "")
            kind = str(symbol.get("kind", "") or "")
            qualname = str(symbol.get("qualname", "") or "")
            if not name or kind not in {"function", "class"} or qualname != name:
                return "loaded_package_module_symbol_identity_invalid"
            symbol_key = (name, kind)
            if previous_symbol_key is not None and symbol_key <= previous_symbol_key:
                return "loaded_package_module_symbols_not_deterministically_sorted"
            previous_symbol_key = symbol_key
            if kind == "function":
                if set(symbol) != {
                    "name",
                    "kind",
                    "qualname",
                    "code_sha256",
                    "default_state_sha256",
                }:
                    return "loaded_package_module_function_symbol_shape_invalid"
                if not _is_sha256_digest(symbol.get("code_sha256")):
                    return "loaded_package_module_function_symbol_digest_invalid"
                if not _is_sha256_digest(symbol.get("default_state_sha256")):
                    return "loaded_package_module_function_default_state_digest_invalid"
            else:
                if set(symbol) != {
                    "name",
                    "kind",
                    "qualname",
                    "source_code_sha256",
                    "base_classes",
                    "metaclass",
                    "source_base_contract_sha256",
                    "state",
                    "state_sha256",
                    "members",
                    "member_code_sha256",
                }:
                    return "loaded_package_module_class_symbol_shape_invalid"
                if not _is_sha256_digest(symbol.get("source_code_sha256")):
                    return "loaded_package_module_class_source_digest_invalid"
                base_classes = symbol.get("base_classes")
                if not isinstance(base_classes, list) or not all(
                    isinstance(base_class, str) and base_class
                    for base_class in base_classes
                ):
                    return "loaded_package_module_class_bases_invalid"
                if not str(symbol.get("metaclass", "") or ""):
                    return "loaded_package_module_class_metaclass_missing"
                if not _is_sha256_digest(symbol.get("source_base_contract_sha256")):
                    return "loaded_package_module_class_base_contract_digest_invalid"
                class_state = symbol.get("state")
                if not isinstance(class_state, list):
                    return "loaded_package_module_class_state_missing"
                if str(symbol.get("state_sha256", "") or "") != canonical_json_sha256(
                    class_state
                ):
                    return "loaded_package_module_class_state_digest_mismatch"
                previous_class_state_key: tuple[str, str] | None = None
                for state_record in class_state:
                    if not isinstance(state_record, Mapping) or set(state_record) != {
                        "owner",
                        "name",
                        "type",
                        "semantic_sha256",
                    }:
                        return "loaded_package_module_class_state_record_invalid"
                    owner = str(state_record.get("owner", "") or "")
                    state_name = str(state_record.get("name", "") or "")
                    state_type = str(state_record.get("type", "") or "")
                    semantic_digest = str(state_record.get("semantic_sha256", "") or "")
                    if (
                        not owner.startswith("class:")
                        or not state_name
                        or not state_type
                        or not _is_sha256_digest(semantic_digest)
                    ):
                        return "loaded_package_module_class_state_identity_invalid"
                    state_key = (owner, state_name)
                    if (
                        previous_class_state_key is not None
                        and state_key <= previous_class_state_key
                    ):
                        return "loaded_package_module_class_state_not_deterministically_sorted"
                    previous_class_state_key = state_key
                members = symbol.get("members")
                if not isinstance(members, list):
                    return "loaded_package_module_class_members_missing"
                if str(
                    symbol.get("member_code_sha256", "") or ""
                ) != canonical_json_sha256(members):
                    return "loaded_package_module_class_members_digest_mismatch"
                previous_member_key: tuple[str, str, str, str] | None = None
                for member in members:
                    if not isinstance(member, Mapping) or set(member) != {
                        "name",
                        "qualname",
                        "code_sha256",
                        "default_state_sha256",
                    }:
                        return "loaded_package_module_class_member_shape_invalid"
                    member_name = str(member.get("name", "") or "")
                    member_qualname = str(member.get("qualname", "") or "")
                    member_digest = str(member.get("code_sha256", "") or "")
                    if (
                        not member_name
                        or not member_qualname.startswith(qualname + ".")
                        or not _is_sha256_digest(member_digest)
                        or not _is_sha256_digest(member.get("default_state_sha256"))
                    ):
                        return "loaded_package_module_class_member_identity_invalid"
                    member_key = (
                        member_name,
                        member_qualname,
                        member_digest,
                        str(member.get("default_state_sha256", "") or ""),
                    )
                    if (
                        previous_member_key is not None
                        and member_key < previous_member_key
                    ):
                        return "loaded_package_module_class_members_not_deterministically_sorted"
                    previous_member_key = member_key
        symbol_digest_records.append(
            {
                "module": module_name,
                "symbols_sha256": record_symbol_digest,
            }
        )
        imports = record.get("imports")
        if not isinstance(imports, list):
            return "loaded_package_module_imports_missing"
        import_digest = str(record.get("imports_sha256", "") or "")
        if not _is_sha256_digest(import_digest):
            return "loaded_package_module_imports_digest_missing_or_invalid"
        if import_digest != canonical_json_sha256(imports):
            return "loaded_package_module_imports_digest_mismatch"
        previous_import_key: tuple[str, str, str] | None = None
        for binding in imports:
            if not isinstance(binding, Mapping) or set(binding) != {
                "local_name",
                "module",
                "attribute",
                "target_kind",
            }:
                return "loaded_package_module_import_record_invalid"
            local_name = str(binding.get("local_name", "") or "")
            owner_name = str(binding.get("module", "") or "")
            attribute = str(binding.get("attribute", "") or "")
            target_kind = str(binding.get("target_kind", "") or "")
            if (
                not local_name
                or not owner_name
                or not target_kind
                or not (
                    owner_name == LOADED_PACKAGE_MODULE_PREFIX
                    or owner_name.startswith(LOADED_PACKAGE_MODULE_PREFIX + ".")
                )
            ):
                return "loaded_package_module_import_identity_invalid"
            import_key = (local_name, owner_name, attribute)
            if previous_import_key is not None and import_key <= previous_import_key:
                return "loaded_package_module_imports_not_deterministically_sorted"
            previous_import_key = import_key
    if module_names != sorted(module_names) or len(module_names) != len(
        set(module_names)
    ):
        return "loaded_package_modules_not_deterministically_sorted"
    if module_names[0] != LOADED_PACKAGE_MODULE_PREFIX:
        return "loaded_package_modules_root_package_missing"
    if symbol_digest != canonical_json_sha256(symbol_digest_records):
        return "loaded_package_symbols_digest_mismatch"
    return ""


def _canonical_import_origins_reason(execution: Mapping[str, Any]) -> str:
    package_identity = execution.get("package_identity")
    if not isinstance(package_identity, Mapping):
        return "package_identity_missing"
    package_root_raw = str(package_identity.get("package_root", "") or "").strip()
    package_kind = str(package_identity.get("kind", "") or "").strip()
    if not package_root_raw:
        return "package_identity_root_missing"
    if package_kind not in {"editable_checkout", "installed_distribution"}:
        return "package_identity_kind_unsupported"
    if str(execution.get("package_identity_sha256", "") or "") != canonical_json_sha256(
        package_identity
    ):
        return "package_identity_digest_mismatch"

    import_origins = execution.get("import_origins")
    if not isinstance(import_origins, Mapping):
        return "import_origins_missing"
    label_reason = _bootstrap_import_labels_reason(import_origins)
    if label_reason:
        return label_reason

    package_root = Path(package_root_raw).resolve()
    source_revision = execution.get("source_revision")
    if not isinstance(source_revision, Mapping):
        return "source_revision_missing"
    module_sha256 = source_revision.get("module_sha256")
    if not isinstance(module_sha256, Mapping):
        return "source_revision_module_hashes_missing"
    module_hash_label_reason = _bootstrap_import_labels_reason(module_sha256)
    if module_hash_label_reason:
        return "source_revision_" + module_hash_label_reason

    expected_module_hashes = {
        label: str(module_sha256.get(label, "") or "")
        for label in CANONICAL_BOOTSTRAP_IMPORT_LABELS
    }
    if any(not value for value in expected_module_hashes.values()):
        return "source_revision_module_hash_missing"
    if str(
        source_revision.get("module_hashes_sha256", "") or ""
    ) != canonical_json_sha256(expected_module_hashes):
        return "source_revision_module_hashes_digest_mismatch"

    for label in CANONICAL_BOOTSTRAP_IMPORT_LABELS:
        record = import_origins.get(label)
        if not isinstance(record, Mapping):
            return f"bootstrap_import_origin_missing:{label}"
        expected_module, expected_attribute = CANONICAL_BOOTSTRAP_IMPORT_SPECS[label]
        module_name = str(record.get("module", "") or "")
        path_raw = str(record.get("path", "") or "").strip()
        if module_name != expected_module:
            return f"bootstrap_import_module_not_canonical:{label}"
        if not path_raw:
            return f"bootstrap_import_path_missing:{label}"
        path = Path(path_raw).resolve()
        if not _is_within(path, package_root):
            return f"bootstrap_import_path_outside_package_root:{label}"
        if path not in _expected_module_paths(module_name, package_root):
            return f"bootstrap_import_path_not_canonical:{label}"
        if str(record.get("package_root", "") or "") != str(package_root):
            return f"bootstrap_import_package_root_mismatch:{label}"
        if str(record.get("package_origin_kind", "") or "") != package_kind:
            return f"bootstrap_import_package_kind_mismatch:{label}"
        expected_kind = "module" if expected_attribute is None else "class"
        if str(record.get("target_kind", "") or "") != expected_kind:
            return f"bootstrap_import_target_kind_mismatch:{label}"
        expected_qualname = (
            expected_module if expected_attribute is None else expected_attribute
        )
        if str(record.get("target_qualname", "") or "") != expected_qualname:
            return f"bootstrap_import_target_qualname_mismatch:{label}"
        if not str(record.get("target_code_sha256", "") or ""):
            return f"bootstrap_import_target_code_digest_missing:{label}"
        record_sha = str(record.get("sha256", "") or "")
        if not record_sha:
            return f"bootstrap_import_source_hash_missing:{label}"
        if expected_module_hashes[label] != record_sha:
            return f"bootstrap_import_source_hash_mismatch:{label}"
    closure_reason = _loaded_package_module_closure_reason(
        execution,
        package_root=package_root,
        source_revision=source_revision,
    )
    if closure_reason:
        return closure_reason
    return ""


def canonical_execution_eligibility(
    payload: Mapping[str, Any] | Any,
) -> tuple[bool, str]:
    """Return whether an artifact is eligible for canonical scorecards.

    Missing execution identity is intentionally treated as noncanonical.  This
    prevents pre-cutover and manually copied legacy CSVs from being promoted by
    a newer aggregation command. The fingerprint excludes only ``created_at``;
    it covers every other top-level contract field.
    """

    if not isinstance(payload, Mapping):
        return False, "execution_provenance_missing"
    execution = payload.get("execution_provenance", payload)
    if not isinstance(execution, Mapping):
        return False, "execution_provenance_missing"
    if str(execution.get("schema_version", "")) != EXECUTION_PROVENANCE_SCHEMA_VERSION:
        return False, "execution_provenance_schema_missing_or_unsupported"
    if str(execution.get("implementation_stack", "")) != CANONICAL_IMPLEMENTATION_STACK:
        return False, "implementation_stack_not_tabnetics_core"
    if str(execution.get("evidence_status", "")) != CANONICAL_EVIDENCE_STATUS:
        return False, "evidence_status_not_canonical"
    if execution.get("canonical_scorecard_eligible") is not True:
        return False, "canonical_scorecard_eligible_not_true"
    if execution.get("worker_module_closures_complete") is not True:
        return False, "worker_module_closures_incomplete"
    resolved_cli_config = execution.get("resolved_cli_config")
    if not isinstance(resolved_cli_config, Mapping):
        return False, "resolved_cli_config_missing"
    resolved_cli_digest = str(execution.get("resolved_cli_config_sha256", "") or "")
    if not resolved_cli_digest:
        return False, "resolved_cli_config_digest_missing"
    if resolved_cli_digest != canonical_json_sha256(resolved_cli_config):
        return False, "resolved_cli_config_digest_mismatch"

    input_data_identity = execution.get("input_data_identity")
    if not isinstance(input_data_identity, Mapping):
        return False, "input_data_identity_missing"
    input_data_digest = str(execution.get("input_data_identity_sha256", "") or "")
    if not input_data_digest:
        return False, "input_data_identity_digest_missing"
    if input_data_digest != canonical_json_sha256(input_data_identity):
        return False, "input_data_identity_digest_mismatch"
    if (
        str(input_data_identity.get("schema_version", ""))
        != INPUT_DATA_IDENTITY_SCHEMA_VERSION
    ):
        return False, "input_data_identity_schema_missing_or_unsupported"
    config_dataset_ids = [
        str(item)
        for item in list(resolved_cli_config.get("selected_dataset_ids") or ())
    ]
    input_dataset_ids = [
        str(item)
        for item in list(input_data_identity.get("selected_dataset_ids") or ())
    ]
    if config_dataset_ids != input_dataset_ids:
        return False, "input_data_identity_dataset_ids_mismatch"
    arguments = resolved_cli_config.get("arguments")
    if not isinstance(arguments, Mapping) or not isinstance(
        arguments.get("seeds"), (list, tuple)
    ):
        return False, "input_data_identity_seed_config_missing"
    materialized_records = input_data_identity.get("materialized_inputs")
    try:
        normalized_records = _validate_materialized_input_records(
            records=(
                materialized_records
                if isinstance(materialized_records, (list, tuple))
                else None
            ),
            selected_dataset_ids=config_dataset_ids,
            expected_seeds=arguments["seeds"],
        )
    except CanonicalExecutionInputIdentityError as exc:
        return False, f"input_data_identity_materialized_invalid:{exc}"
    if str(input_data_identity.get("materialized_inputs_sha256", "") or "") != (
        canonical_json_sha256(normalized_records)
    ):
        return False, "input_data_identity_materialized_digest_mismatch"

    fingerprint = str(execution.get("fingerprint_sha256", "") or "")
    if not fingerprint:
        return False, "execution_provenance_fingerprint_missing"
    if fingerprint != execution_fingerprint_sha256(execution):
        return False, "execution_provenance_fingerprint_mismatch"

    origin_reason = _canonical_import_origins_reason(execution)
    if origin_reason:
        return False, origin_reason
    return True, ""


def canonical_execution_contract_consistency(
    reference: Mapping[str, Any] | Any,
    candidate: Mapping[str, Any] | Any,
) -> tuple[bool, str]:
    """Check that two canonical contracts describe the same execution.

    ``created_at`` may differ because it is explicitly outside the execution
    fingerprint. Every immutable identity field must otherwise match.
    """

    reference_ok, reference_reason = canonical_execution_eligibility(reference)
    if not reference_ok:
        return False, f"reference_{reference_reason}"
    candidate_ok, candidate_reason = canonical_execution_eligibility(candidate)
    if not candidate_ok:
        return False, f"candidate_{candidate_reason}"
    reference_contract = reference.get("execution_provenance", reference)
    candidate_contract = candidate.get("execution_provenance", candidate)
    assert isinstance(reference_contract, Mapping)
    assert isinstance(candidate_contract, Mapping)
    for field in (
        "schema_version",
        "implementation_stack",
        "evidence_status",
        "canonical_scorecard_eligible",
        "worker_module_closures_complete",
        "resolved_cli_config_sha256",
        "input_data_identity_sha256",
        "package_identity_sha256",
        "loaded_package_modules_sha256",
        "loaded_package_symbols_sha256",
        "fingerprint_sha256",
    ):
        if reference_contract.get(field) != candidate_contract.get(field):
            return False, f"execution_contract_field_mismatch:{field}"
    return True, ""


def _artifact_labels_reason(artifacts: Mapping[str, Any]) -> str:
    labels = {str(label) for label in artifacts}
    required = set(CANONICAL_BENCHMARK_ARTIFACT_NAMES)
    missing = sorted(required - labels)
    if missing:
        return "artifact_labels_missing:" + ",".join(missing)
    unexpected = sorted(labels - required)
    if unexpected:
        return "artifact_labels_unexpected:" + ",".join(unexpected)
    return ""


def _artifact_identity(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Canonical artifact is missing or not a file: {artifact_path}"
        )
    stat = artifact_path.stat()
    return {
        "sha256": sha256_file(artifact_path),
        "size_bytes": int(stat.st_size),
    }


def build_canonical_benchmark_artifact_provenance(
    *,
    execution_provenance: Mapping[str, Any],
    artifact_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Bind canonical benchmark result bytes to one execution contract.

    The artifact record intentionally excludes itself: hashing a manifest that
    contains its own hash would be cyclic. It binds every CSV consumed by the
    aggregator plus the metadata and execution-contract files that describe
    those CSVs.
    """

    eligible, reason = canonical_execution_eligibility(execution_provenance)
    if not eligible:
        raise ValueError(
            "Cannot create canonical benchmark artifact provenance for an "
            f"ineligible execution contract: {reason}."
        )
    label_reason = _artifact_labels_reason(artifact_paths)
    if label_reason:
        raise ValueError(
            f"Canonical benchmark artifact set is incomplete: {label_reason}."
        )
    artifacts: dict[str, dict[str, Any]] = {}
    for name in CANONICAL_BENCHMARK_ARTIFACT_NAMES:
        path = Path(artifact_paths[name])
        if path.name != name:
            raise ValueError(
                f"Canonical artifact {name!r} has an unexpected path name: {path.name!r}."
            )
        artifacts[name] = _artifact_identity(path)
    return {
        "schema_version": CANONICAL_BENCHMARK_ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_type": "df_fs_canonical_artifact_bundle",
        "execution_provenance": dict(execution_provenance),
        "execution_provenance_sha256": str(
            execution_provenance.get("fingerprint_sha256", "") or ""
        ),
        "loaded_package_modules_sha256": str(
            execution_provenance.get("loaded_package_modules_sha256", "") or ""
        ),
        "loaded_package_symbols_sha256": str(
            execution_provenance.get("loaded_package_symbols_sha256", "") or ""
        ),
        "artifacts": artifacts,
    }


def canonical_benchmark_artifact_provenance_consistency(
    reference_execution: Mapping[str, Any],
    artifact_provenance: Mapping[str, Any] | Any,
    *,
    artifact_paths: Mapping[str, str | Path],
) -> tuple[bool, str]:
    """Verify an artifact manifest against its contract and on-disk bytes.

    This is tamper-evident for partial/copy/accidental changes. A principal
    able to replace the executing interpreter and every manifest/artifact can
    forge a self-consistent record; that stronger adversary needs signed remote
    attestation outside the local file-format contract.
    """

    if not isinstance(artifact_provenance, Mapping):
        return False, "artifact_provenance_missing"
    if str(artifact_provenance.get("schema_version", "") or "") != (
        CANONICAL_BENCHMARK_ARTIFACT_PROVENANCE_SCHEMA_VERSION
    ):
        return False, "artifact_provenance_schema_missing_or_unsupported"
    if str(artifact_provenance.get("artifact_type", "") or "") != (
        "df_fs_canonical_artifact_bundle"
    ):
        return False, "artifact_provenance_type_mismatch"
    embedded_execution = artifact_provenance.get("execution_provenance")
    if not isinstance(embedded_execution, Mapping):
        return False, "artifact_execution_provenance_missing"
    consistent, reason = canonical_execution_contract_consistency(
        reference_execution,
        embedded_execution,
    )
    if not consistent:
        return False, "artifact_execution_contract_" + reason
    expected_fingerprint = str(reference_execution.get("fingerprint_sha256", "") or "")
    if str(artifact_provenance.get("execution_provenance_sha256", "") or "") != (
        expected_fingerprint
    ):
        return False, "artifact_execution_fingerprint_mismatch"
    expected_loaded_module_digest = str(
        reference_execution.get("loaded_package_modules_sha256", "") or ""
    )
    if str(artifact_provenance.get("loaded_package_modules_sha256", "") or "") != (
        expected_loaded_module_digest
    ):
        return False, "artifact_loaded_package_modules_digest_mismatch"
    expected_loaded_symbol_digest = str(
        reference_execution.get("loaded_package_symbols_sha256", "") or ""
    )
    if str(artifact_provenance.get("loaded_package_symbols_sha256", "") or "") != (
        expected_loaded_symbol_digest
    ):
        return False, "artifact_loaded_package_symbols_digest_mismatch"

    artifacts = artifact_provenance.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return False, "artifact_identities_missing"
    label_reason = _artifact_labels_reason(artifacts)
    if label_reason:
        return False, label_reason
    path_label_reason = _artifact_labels_reason(artifact_paths)
    if path_label_reason:
        return False, "expected_" + path_label_reason
    for name in CANONICAL_BENCHMARK_ARTIFACT_NAMES:
        expected_identity = artifacts.get(name)
        if not isinstance(expected_identity, Mapping):
            return False, f"artifact_identity_missing:{name}"
        expected_hash = str(expected_identity.get("sha256", "") or "")
        expected_size = expected_identity.get("size_bytes")
        if not expected_hash:
            return False, f"artifact_sha256_missing:{name}"
        try:
            expected_size_int = int(expected_size)
        except (TypeError, ValueError):
            return False, f"artifact_size_missing_or_invalid:{name}"
        path = Path(artifact_paths[name])
        if not path.is_file():
            return False, f"artifact_file_missing:{name}"
        actual_size = int(path.stat().st_size)
        if sha256_file(path) != expected_hash:
            return False, f"artifact_sha256_mismatch:{name}"
        if actual_size != expected_size_int:
            return False, f"artifact_size_mismatch:{name}"
    return True, ""


def _run_git(repo_root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def git_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = (
        Path(repo_root) if repo_root is not None else find_repo_root_or_none(__file__)
    )
    if root is None:
        return {"repo_root": "", "sha": "", "branch": "", "dirty": None}
    status = _run_git(root, ["status", "--porcelain"])
    return {
        "repo_root": str(root),
        "sha": _run_git(root, ["rev-parse", "HEAD"]),
        "branch": _run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_porcelain_count": len(
            [line for line in status.splitlines() if line.strip()]
        ),
    }


def _version_for(import_name: str, package_name: str | None = None) -> str:
    try:
        return importlib_metadata.version(package_name or import_name)
    except Exception:
        try:
            module = __import__(import_name)
            return str(getattr(module, "__version__", ""))
        except Exception:
            return ""


def package_versions() -> dict[str, str]:
    return {
        "numpy": _version_for("numpy"),
        "pandas": _version_for("pandas"),
        "scipy": _version_for("scipy"),
        "sklearn": _version_for("sklearn", "scikit-learn"),
        "torch": _version_for("torch"),
        "tabnetics": _version_for("tabnetics"),
    }


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path_identity(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"path": "", "exists": False}
    p = Path(path).expanduser()
    out: dict[str, Any] = {"path": str(p), "exists": bool(p.exists())}
    if p.exists() and p.is_file():
        stat = p.stat()
        out.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime_unix": float(stat.st_mtime),
            }
        )
        if int(stat.st_size) <= MAX_PROVENANCE_HASH_BYTES:
            out["sha256"] = sha256_file(p)
        else:
            out["sha256"] = ""
            out["sha256_skipped_reason"] = (
                f"file_exceeds_{MAX_PROVENANCE_HASH_BYTES}_bytes"
            )
    return out


def _dataset_source_hints(spec: Any) -> dict[str, Any]:
    params = dict(getattr(spec, "params", {}) or {})
    hints: dict[str, Any] = {}
    for key in (
        "openml_options",
        "mat_url_options",
        "tab_url_options",
        "default_local_path",
        "default_local_paths",
        "local_path_env",
        "source_policy",
        "hf_repo_id",
        "hf_revision",
        "data_foundry_uri",
    ):
        if key in params:
            hints[key] = params.get(key)
    for key, value in params.items():
        if "revision" in str(key).lower() and key not in hints:
            hints[str(key)] = value
    return _json_safe(hints)


def dataset_registry_identity(
    dataset_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    try:
        from tabnetics.datasets.registry import DATASET_REGISTRY
    except Exception:
        return {"dataset_ids": list(dataset_ids or []), "registry_available": False}

    wanted = [str(item) for item in list(dataset_ids or []) if str(item).strip()]
    all_ids = sorted(str(key) for key in DATASET_REGISTRY.keys())
    if not wanted:
        encoded_all = json.dumps(all_ids, sort_keys=True).encode("utf-8")
        return {
            "registry_available": True,
            "dataset_count": 0,
            "dataset_ids": [],
            "registry_total_dataset_count": int(len(all_ids)),
            "registry_ids_fingerprint_sha256": hashlib.sha256(encoded_all).hexdigest(),
            "datasets": [],
        }
    records: list[dict[str, Any]] = []
    for ds_id in wanted:
        spec = DATASET_REGISTRY.get(ds_id)
        if spec is None:
            records.append({"dataset_id": ds_id, "registered": False})
            continue
        records.append(
            {
                "dataset_id": str(getattr(spec, "dataset_id", ds_id)),
                "registered": True,
                "pipeline": str(getattr(spec, "pipeline", "")),
                "tier": str(getattr(spec, "tier", "")),
                "loader_kind": str(getattr(spec, "loader_kind", "")),
                "source_kind": str(getattr(spec, "source_kind", "")),
                "domain": str(getattr(spec, "domain", "")),
                "platform": str(getattr(spec, "platform", "")),
                "source_hints": _dataset_source_hints(spec),
            }
        )
    encoded = json.dumps(records, sort_keys=True, default=str).encode("utf-8")
    return {
        "registry_available": True,
        "dataset_count": int(len(records)),
        "dataset_ids": [record["dataset_id"] for record in records],
        "registry_fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
        "datasets": records,
    }


def env_snapshot(extra_env: Sequence[str] | None = None) -> dict[str, str]:
    keys = [*DEFAULT_ENV_KEYS, *list(extra_env or ())]
    out: dict[str, str] = {}
    for key in keys:
        if key in os.environ:
            out[str(key)] = str(os.environ.get(key, ""))
    return out


def resource_usage_children() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "user_cpu_sec": float(usage.ru_utime),
        "system_cpu_sec": float(usage.ru_stime),
        "max_rss_kb": int(usage.ru_maxrss),
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
    }


def resource_usage_delta(
    start: Mapping[str, Any], end: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "user_cpu_sec": float(end.get("user_cpu_sec", 0.0))
        - float(start.get("user_cpu_sec", 0.0)),
        "system_cpu_sec": float(end.get("system_cpu_sec", 0.0))
        - float(start.get("system_cpu_sec", 0.0)),
        "peak_rss_kb": int(end.get("max_rss_kb", 0)),
        "minor_faults": int(end.get("minor_faults", 0))
        - int(start.get("minor_faults", 0)),
        "major_faults": int(end.get("major_faults", 0))
        - int(start.get("major_faults", 0)),
        "voluntary_context_switches": int(end.get("voluntary_context_switches", 0))
        - int(start.get("voluntary_context_switches", 0)),
        "involuntary_context_switches": int(end.get("involuntary_context_switches", 0))
        - int(start.get("involuntary_context_switches", 0)),
    }


def build_code_provenance(
    *,
    repo_root: str | Path | None = None,
    dataset_ids: Sequence[str] | None = None,
    command: Sequence[str] | None = None,
    plan_path: str | Path | None = None,
    shards_path: str | Path | None = None,
    extra_paths: Mapping[str, str | Path] | None = None,
    extra_env: Sequence[str] | None = None,
) -> dict[str, Any]:
    paths = {
        str(name): path_identity(path) for name, path in dict(extra_paths or {}).items()
    }
    if plan_path:
        paths["plan"] = path_identity(plan_path)
    if shards_path:
        paths["shards"] = path_identity(shards_path)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "host": {
            "hostname": socket.gethostname(),
            "platform_node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
        },
        "packages": package_versions(),
        "git": git_provenance(repo_root),
        "environment": env_snapshot(extra_env),
        "command": [str(item) for item in list(command or [])],
        "inputs": paths,
        "data_identity": dataset_registry_identity(dataset_ids),
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return p

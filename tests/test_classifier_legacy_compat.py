import ast
import importlib
import importlib.util
import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import pytest


EXPECTED_EXPORTS = (
    "REGIME_HDLSS_EXTREME",
    "REGIME_HDLSS_MODERATE",
    "REGIME_STANDARD",
    "REGIME_POOLS",
    "CLASSIFIER_COMPLEXITY_PRIOR",
    "FLAML_NATIVE_BY_FAMILY",
    "classify_regime",
    "OracleCandidateStats",
    "PLSDAClassifier",
    "ClassifierBackend",
    "SklearnBackend",
    "FLAMLBackend",
    "OptunaBackend",
    "ClassifierOracle",
    "MNPOClassifierBackend",
)
LEGACY_MODULE = "tabnetics.feature_selection.classification"
CANONICAL_MODULE = "tabnetics.classification"
BACKEND_MODULE = "tabnetics.classification.backends"
CORE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = CORE_ROOT / "src"
SHIM_PATH = SOURCE_ROOT / "tabnetics" / "feature_selection" / "classification.py"


def _run_clean_python(source: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SOURCE_ROOT) + (os.pathsep + current if current else "")
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=CORE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


@pytest.fixture(scope="module")
def legacy_module():
    parent = importlib.import_module("tabnetics.feature_selection")
    sys.modules.pop(LEGACY_MODULE, None)
    parent.__dict__.pop("classification", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(LEGACY_MODULE)
    compatibility = [
        item for item in caught if LEGACY_MODULE in str(item.message)
    ]
    assert len(compatibility) == 1
    assert compatibility[0].category is DeprecationWarning
    assert CANONICAL_MODULE in str(compatibility[0].message)
    return module


def test_legacy_exports_are_exact_ordered_and_identical(legacy_module):
    canonical = importlib.import_module(CANONICAL_MODULE)
    backends = importlib.import_module(BACKEND_MODULE)

    assert legacy_module.__all__ == EXPECTED_EXPORTS
    for name in EXPECTED_EXPORTS:
        assert getattr(legacy_module, name) is getattr(backends, name)
        assert getattr(legacy_module, name) is getattr(canonical, name)


def test_legacy_star_import_contains_only_supported_names(legacy_module):
    namespace = {}
    exec(f"from {LEGACY_MODULE} import *", namespace)
    exported = {name for name in namespace if name != "__builtins__"}
    assert exported == set(EXPECTED_EXPORTS)
    assert all(namespace[name] is getattr(legacy_module, name) for name in exported)


def test_legacy_class_aliases_retain_canonical_pickle_identity(legacy_module):
    backends = importlib.import_module(BACKEND_MODULE)
    instance = legacy_module.PLSDAClassifier()

    assert type(instance) is backends.PLSDAClassifier
    assert type(instance).__module__ == BACKEND_MODULE
    assert BACKEND_MODULE.encode() in pickle.dumps(instance)


@pytest.mark.parametrize(
    "imports",
    [
        (LEGACY_MODULE, CANONICAL_MODULE, "tabnetics.pipeline.pipeline"),
        (CANONICAL_MODULE, "tabnetics.pipeline.pipeline", LEGACY_MODULE),
    ],
)
def test_cold_import_orders_have_no_partial_or_circular_state(imports):
    source = f"""
import importlib
import sys
import warnings

expected = {EXPECTED_EXPORTS!r}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    modules = [importlib.import_module(name) for name in {imports!r}]
legacy = importlib.import_module({LEGACY_MODULE!r})
canonical = importlib.import_module({CANONICAL_MODULE!r})
backends = importlib.import_module({BACKEND_MODULE!r})
pipeline = importlib.import_module("tabnetics.pipeline.pipeline")
assert tuple(legacy.__all__) == expected
for name in expected:
    assert getattr(legacy, name) is getattr(canonical, name)
    assert getattr(legacy, name) is getattr(backends, name)
for module in (legacy, canonical, backends, pipeline, *modules):
    assert module.__spec__ is not None
    assert not getattr(module.__spec__, "_initializing", False)
    assert sys.modules[module.__name__] is module
assert pipeline.ClassifierBackend is backends.ClassifierBackend
assert pipeline.SklearnBackend is backends.SklearnBackend
"""
    _run_clean_python(source)


@pytest.mark.parametrize(
    "imports",
    [
        (CANONICAL_MODULE, LEGACY_MODULE, "tabnetics.pipeline.pipeline"),
        (LEGACY_MODULE, CANONICAL_MODULE, "tabnetics.pipeline.pipeline"),
    ],
)
def test_cold_imports_succeed_with_runtime_optional_packages_blocked(imports):
    source = f"""
import importlib
import sys
import warnings

blocked = {{"flaml", "optuna", "tabpfn", "xgboost", "pytabkit", "catboost", "lightgbm"}}

class BlockOptionalPackages:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise ModuleNotFoundError(f"blocked optional package: {{fullname}}")
        return None

sys.meta_path.insert(0, BlockOptionalPackages())
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    [importlib.import_module(name) for name in {imports!r}]
legacy = importlib.import_module({LEGACY_MODULE!r})
canonical = importlib.import_module({CANONICAL_MODULE!r})
backends = importlib.import_module({BACKEND_MODULE!r})
for name in {EXPECTED_EXPORTS!r}:
    assert getattr(legacy, name) is getattr(canonical, name)
    assert getattr(legacy, name) is getattr(backends, name)
"""
    _run_clean_python(source)


def test_warning_is_isolated_to_explicit_legacy_import_in_fresh_interpreter():
    source = f"""
import importlib
import warnings

legacy_path = {LEGACY_MODULE!r}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    importlib.import_module("tabnetics.feature_selection")
    importlib.import_module({CANONICAL_MODULE!r})
    importlib.import_module("tabnetics.pipeline.pipeline")
compat = [item for item in caught if legacy_path in str(item.message)]
assert compat == []

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    importlib.import_module(legacy_path)
compat = [item for item in caught if legacy_path in str(item.message)]
assert len(compat) == 1
assert compat[0].category is DeprecationWarning
assert {CANONICAL_MODULE!r} in str(compat[0].message)
"""
    _run_clean_python(source)


def test_shim_ast_contains_only_explicit_reexports_warning_and_all():
    tree = ast.parse(SHIM_PATH.read_text(encoding="utf-8"), filename=str(SHIM_PATH))

    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(tree)
    )
    assert not any(isinstance(node, (ast.Dict, ast.DictComp)) for node in ast.walk(tree))
    assert not any(
        isinstance(node, ast.Name) and node.id == "__getattr__"
        for node in ast.walk(tree)
    )

    imports = [node for node in tree.body if isinstance(node, ast.Import)]
    assert len(imports) == 1
    assert [(alias.name, alias.asname) for alias in imports[0].names] == [
        ("warnings", "_warnings")
    ]
    from_imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert len(from_imports) == 1
    assert from_imports[0].module == BACKEND_MODULE
    assert tuple(alias.name for alias in from_imports[0].names) == EXPECTED_EXPORTS
    assert all(alias.name != "*" for alias in from_imports[0].names)

    assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
    assert len(assignments) == 1
    assert [target.id for target in assignments[0].targets] == ["__all__"]
    assert ast.literal_eval(assignments[0].value) == EXPECTED_EXPORTS

    warning_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_warnings"
        and node.func.attr == "warn"
    ]
    assert len(warning_calls) == 1
    assert isinstance(warning_calls[0].args[1], ast.Name)
    assert warning_calls[0].args[1].id == "DeprecationWarning"
    stacklevel = next(
        keyword.value
        for keyword in warning_calls[0].keywords
        if keyword.arg == "stacklevel"
    )
    assert ast.literal_eval(stacklevel) == 2


def _module_and_package(path: Path) -> tuple[str, str]:
    parts = path.relative_to(SOURCE_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        module = ".".join(parts[:-1])
        return module, module
    module = ".".join(parts)
    return module, ".".join(parts[:-1])


def test_production_sources_do_not_import_or_reference_legacy_module():
    violations = []
    for path in sorted((SOURCE_ROOT / "tabnetics").rglob("*.py")):
        if path == SHIM_PATH:
            continue
        source = path.read_text(encoding="utf-8")
        if LEGACY_MODULE in source:
            violations.append((path, "absolute/dynamic reference"))
            continue
        tree = ast.parse(source, filename=str(path))
        _, package = _module_and_package(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == LEGACY_MODULE or alias.name.startswith(
                        LEGACY_MODULE + "."
                    ):
                        violations.append((path, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative = "." * node.level + (node.module or "")
                    imported_module = importlib.util.resolve_name(relative, package)
                else:
                    imported_module = node.module or ""
                imported_names = {
                    imported_module,
                    *(f"{imported_module}.{alias.name}" for alias in node.names),
                }
                if any(
                    name == LEGACY_MODULE or name.startswith(LEGACY_MODULE + ".")
                    for name in imported_names
                ):
                    violations.append((path, f"from {relative if node.level else imported_module}"))

    assert violations == []

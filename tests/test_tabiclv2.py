from __future__ import annotations

import hashlib
import importlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tabnetics.classification import tabiclv2


def _training_data(
    n_rows: int = 300, n_features: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(n_rows * n_features, dtype=np.float64).reshape(
        n_rows, n_features
    )
    labels = np.resize(np.array(["case", "control"], dtype=str), n_rows)
    return values, labels


def _checkpoint(tmp_path: Path, content: bytes = b"pinned-tabiclv2-checkpoint") -> Path:
    path = tmp_path / tabiclv2.TABICLV2_CHECKPOINT
    path.write_bytes(content)
    return path


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "imports": [],
        "constructor_calls": [],
        "fit_calls": [],
        "predict_calls": [],
        "predict_output": None,
        "classes_override": None,
    }

    def upstream_init(self: Any, **kwargs: Any) -> None:
        state["constructor_calls"].append(dict(kwargs))

    def upstream_fit(self: Any, X: np.ndarray, y: np.ndarray) -> Any:
        state["fit_calls"].append((np.array(X, copy=True), np.array(y, copy=True)))
        override = state["classes_override"]
        self.classes_ = np.array(np.unique(y) if override is None else override)
        return self

    def upstream_predict_proba(self: Any, X: np.ndarray) -> np.ndarray:
        state["predict_calls"].append(np.array(X, copy=True))
        configured = state["predict_output"]
        if callable(configured):
            return np.asarray(configured(X))
        if configured is not None:
            return np.asarray(configured)
        return np.tile(np.array([[2.0, 1.0]]), (len(X), 1))

    upstream_class = type(
        tabiclv2.TABICLV2_CLASS_NAME,
        (),
        {
            "__module__": tabiclv2.TABICLV2_CLASS_MODULE,
            "__init__": upstream_init,
            "fit": upstream_fit,
            "predict_proba": upstream_predict_proba,
        },
    )
    upstream_module = SimpleNamespace(TabICLClassifier=upstream_class)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            current_device=lambda: 0,
        )
    )
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> Any:
        state["imports"].append(name)
        if name == "torch":
            return fake_torch
        if name == tabiclv2.TABICLV2_PACKAGE:
            return upstream_module
        return real_import(name, package)

    monkeypatch.setattr(tabiclv2.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        tabiclv2, "_distribution_version", lambda: tabiclv2.TABICLV2_PACKAGE_VERSION
    )
    checkpoint_content = b"pinned-tabiclv2-checkpoint"
    monkeypatch.setattr(
        tabiclv2, "TABICLV2_CHECKPOINT_SIZE_BYTES", len(checkpoint_content)
    )
    monkeypatch.setattr(
        tabiclv2,
        "TABICLV2_CHECKPOINT_SHA256",
        hashlib.sha256(checkpoint_content).hexdigest(),
    )
    monkeypatch.setattr(
        tabiclv2,
        "_pinned_cache_checkpoint_identity",
        lambda: {
            "path": "/mock/huggingface/cache/pinned-tabiclv2.ckpt",
            "size_bytes": tabiclv2.TABICLV2_CHECKPOINT_SIZE_BYTES,
            "sha256": tabiclv2.TABICLV2_CHECKPOINT_SHA256,
        },
    )
    state.update(
        {
            "upstream_class": upstream_class,
            "upstream_module": upstream_module,
            "torch": fake_torch,
        }
    )
    return state


def test_import_is_lazy_for_upstream_and_torch() -> None:
    sys.modules.pop("tabicl", None)
    reloaded = importlib.reload(tabiclv2)

    assert "tabicl" not in sys.modules
    identity = reloaded.tabiclv2_contract_identity()
    assert identity["package"] == {"name": "tabicl", "version": "2.1.1"}


def test_contract_identity_is_fixed_fresh_and_import_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tabiclv2.importlib,
        "import_module",
        lambda name: pytest.fail(f"identity unexpectedly imported {name}"),
    )

    first = tabiclv2.tabiclv2_contract_identity()
    first["checkpoint"]["revision"] = "mutated"  # type: ignore[index]
    second = tabiclv2.tabiclv2_contract_identity()

    assert second["checkpoint"] == {
        "repo_id": "jingang/TabICL",
        "revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
        "filename": "tabicl-classifier-v2-20260212.ckpt",
        "size_bytes": 110_368_038,
        "sha256": "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
        "identity_semantics": "exact Hugging Face LFS object at the pinned revision",
    }
    assert second["license"] == "BSD-3-Clause"
    assert second["published_limits"] == {
        "min_train_rows": 300,
        "max_train_rows": 100_000,
        "max_features": 2_000,
        "semantics": "training-context limits documented for TabICLv2 evaluation",
    }


def test_sklearn_clone_preserves_only_constructor_parameters(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    classifier = tabiclv2.TabICLv2Classifier(
        checkpoint,
        device="cuda:1",
        min_train_rows=400,
        max_train_rows=900,
        max_features=50,
        random_state=7,
    )

    cloned = clone(classifier)

    assert cloned.get_params(deep=True) == classifier.get_params(deep=True)
    assert not hasattr(cloned, "metadata_")


def test_missing_package_has_typed_machine_readable_skip(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _training_data()
    checkpoint = _checkpoint(tmp_path)

    def missing_version() -> str:
        raise tabiclv2._availability(
            "missing", "skipped_tabiclv2_dependency_unavailable"
        )

    monkeypatch.setattr(tabiclv2, "_distribution_version", missing_version)
    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as caught:
        tabiclv2.TabICLv2Classifier(checkpoint).fit(X, y)

    assert caught.value.status == "skipped_tabiclv2_dependency_unavailable"
    assert "tabicl" not in fake_runtime["imports"]


def test_distribution_version_reports_uninstalled_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> str:
        raise importlib_metadata.PackageNotFoundError("tabicl")

    monkeypatch.setattr(tabiclv2.importlib_metadata, "version", missing)
    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as caught:
        tabiclv2._distribution_version()
    assert caught.value.status == "skipped_tabiclv2_dependency_unavailable"


def test_version_mismatch_fails_before_upstream_import(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _training_data()
    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(tabiclv2, "_distribution_version", lambda: "2.1.0")

    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as caught:
        tabiclv2.TabICLv2Classifier(checkpoint).fit(X, y)

    assert caught.value.status == "skipped_tabiclv2_version_mismatch"
    assert "tabicl" not in fake_runtime["imports"]


@pytest.mark.parametrize("identity_failure", ["missing", "wrong_module", "wrong_name"])
def test_upstream_class_identity_mismatch_fails_closed(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
    identity_failure: str,
) -> None:
    X, y = _training_data()
    checkpoint = _checkpoint(tmp_path)
    if identity_failure == "missing":
        fake_runtime["upstream_module"].TabICLClassifier = None
    elif identity_failure == "wrong_module":
        fake_runtime["upstream_class"].__module__ = "tabicl"
    else:
        fake_runtime["upstream_class"].__name__ = "CompatibleClassifier"

    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as caught:
        tabiclv2.TabICLv2Classifier(checkpoint).fit(X, y)

    assert caught.value.status == "skipped_tabiclv2_api_mismatch"


def test_constructor_api_mismatch_fails_closed(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    X, y = _training_data()
    checkpoint = _checkpoint(tmp_path)

    def incompatible_init(self: Any, renamed_path: str) -> None:
        del self, renamed_path

    incompatible = type(
        tabiclv2.TABICLV2_CLASS_NAME,
        (),
        {
            "__module__": tabiclv2.TABICLV2_CLASS_MODULE,
            "__init__": incompatible_init,
        },
    )
    fake_runtime["upstream_module"].TabICLClassifier = incompatible

    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as caught:
        tabiclv2.TabICLv2Classifier(checkpoint).fit(X, y)

    assert caught.value.status == "skipped_tabiclv2_api_mismatch"


@pytest.mark.parametrize("checkpoint_value", ["", "missing.ckpt"])
def test_checkpoint_must_be_explicit_and_existing_before_cuda_or_upstream_import(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
    checkpoint_value: str,
) -> None:
    X, y = _training_data()
    value = (
        checkpoint_value if checkpoint_value == "" else str(tmp_path / checkpoint_value)
    )

    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as caught:
        tabiclv2.TabICLv2Classifier(value).fit(X, y)

    assert caught.value.status in {
        "skipped_tabiclv2_checkpoint_not_configured",
        "skipped_tabiclv2_checkpoint_unavailable",
    }
    assert fake_runtime["imports"] == []


def test_empty_checkpoint_fails_before_cuda_or_upstream_import(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    X, y = _training_data()
    checkpoint = _checkpoint(tmp_path, b"")

    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as caught:
        tabiclv2.TabICLv2Classifier(checkpoint).fit(X, y)

    assert caught.value.status == "skipped_tabiclv2_checkpoint_unavailable"
    assert fake_runtime["imports"] == []


def test_pinned_cache_resolution_is_local_only_and_revision_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"pinned-cache-fixture"
    cache_path = tmp_path / tabiclv2.TABICLV2_CHECKPOINT
    cache_path.write_bytes(content)
    calls: list[dict[str, object]] = []

    def hf_hub_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(cache_path)

    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "huggingface_hub":
            return SimpleNamespace(hf_hub_download=hf_hub_download)
        return real_import(name, package)

    monkeypatch.setattr(tabiclv2.importlib, "import_module", fake_import)
    monkeypatch.setattr(tabiclv2, "TABICLV2_CHECKPOINT_SIZE_BYTES", len(content))
    monkeypatch.setattr(
        tabiclv2, "TABICLV2_CHECKPOINT_SHA256", hashlib.sha256(content).hexdigest()
    )

    identity = tabiclv2._pinned_cache_checkpoint_identity()

    assert calls == [
        {
            "repo_id": "jingang/TabICL",
            "filename": "tabicl-classifier-v2-20260212.ckpt",
            "revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
            "local_files_only": True,
        }
    ]
    assert identity == {
        "path": str(cache_path.resolve()),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_missing_pinned_cache_is_typed_provenance_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(**_: object) -> str:
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(
        tabiclv2.importlib,
        "import_module",
        lambda name: SimpleNamespace(hf_hub_download=unavailable),
    )

    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as caught:
        tabiclv2._pinned_cache_checkpoint_identity()

    assert caught.value.status == "skipped_tabiclv2_checkpoint_provenance_unavailable"


def test_nonempty_wrong_checkpoint_identity_fails_before_cuda_or_upstream_import(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    X, y = _training_data()
    checkpoint = _checkpoint(tmp_path, b"x" * len(b"pinned-tabiclv2-checkpoint"))

    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as caught:
        tabiclv2.TabICLv2Classifier(checkpoint).fit(X, y)

    assert caught.value.status == "skipped_tabiclv2_checkpoint_identity_mismatch"
    assert fake_runtime["imports"] == []


@pytest.mark.parametrize("device", ["cpu", "mps", "", 0])
def test_non_cuda_device_fails_before_upstream_import(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
    device: Any,
) -> None:
    X, y = _training_data()
    checkpoint = _checkpoint(tmp_path)

    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as caught:
        tabiclv2.TabICLv2Classifier(checkpoint, device=device).fit(X, y)

    assert caught.value.status == "skipped_tabiclv2_cuda_required"
    assert "tabicl" not in fake_runtime["imports"]


def test_unavailable_and_out_of_range_cuda_fail_before_upstream_import(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    X, y = _training_data()
    checkpoint = _checkpoint(tmp_path)
    fake_runtime["torch"].cuda.is_available = lambda: False

    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as unavailable:
        tabiclv2.TabICLv2Classifier(checkpoint).fit(X, y)
    assert unavailable.value.status == "skipped_tabiclv2_cuda_unavailable"
    assert "tabicl" not in fake_runtime["imports"]

    fake_runtime["torch"].cuda.is_available = lambda: True
    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as out_of_range:
        tabiclv2.TabICLv2Classifier(checkpoint, device="cuda:2").fit(X, y)
    assert out_of_range.value.status == "skipped_tabiclv2_cuda_unavailable"
    assert "tabicl" not in fake_runtime["imports"]


@pytest.mark.parametrize(
    ("n_rows", "n_features"),
    [(299, 4), (100_001, 1), (300, 2_001)],
)
def test_published_size_failures_are_deterministic_skips(
    fake_runtime: dict[str, Any],
    n_rows: int,
    n_features: int,
) -> None:
    X, y = _training_data(n_rows=n_rows, n_features=n_features)

    with pytest.raises(tabiclv2.TabICLv2AvailabilityError) as caught:
        tabiclv2.TabICLv2Classifier("not-reached.ckpt").fit(X, y)

    assert caught.value.status == "skipped_tabiclv2_outside_published_regime"
    assert fake_runtime["imports"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_train_rows": 299},
        {"min_train_rows": 500, "max_train_rows": 499},
        {"max_train_rows": 100_001},
        {"max_features": 2_001},
        {"max_features": 0},
        {"min_train_rows": 300.0},
    ],
)
def test_configured_limits_cannot_widen_or_invert_published_regime(
    fake_runtime: dict[str, Any], kwargs: dict[str, Any]
) -> None:
    X, y = _training_data()
    with pytest.raises(tabiclv2.TabICLv2ContractError):
        tabiclv2.TabICLv2Classifier("not-reached.ckpt", **kwargs).fit(X, y)
    assert fake_runtime["imports"] == []


@pytest.mark.parametrize(
    "invalid_X",
    [
        np.ones(4),
        np.empty((0, 4)),
        np.empty((3, 0)),
        np.array([["1", "2"]]),
        np.array([[True, False]]),
        np.array([[1.0, np.nan]]),
        np.array([[1.0, np.inf]]),
        np.array([[1.0 + 2.0j]]),
    ],
)
def test_fit_rejects_malformed_input_before_environment_checks(
    fake_runtime: dict[str, Any], invalid_X: np.ndarray
) -> None:
    y = np.zeros(len(invalid_X)) if invalid_X.ndim else np.array([])
    with pytest.raises(tabiclv2.TabICLv2ContractError):
        tabiclv2.TabICLv2Classifier("not-reached.ckpt").fit(invalid_X, y)
    assert fake_runtime["imports"] == []


@pytest.mark.parametrize(
    "invalid_y",
    [
        None,
        np.array([[0], [1]] * 150),
        np.zeros(299),
        np.zeros(300),
        np.resize(np.array([0.0, np.nan]), 300),
        np.resize(np.array([0, None], dtype=object), 300),
    ],
)
def test_fit_rejects_invalid_or_single_class_training_labels(
    fake_runtime: dict[str, Any], invalid_y: Any
) -> None:
    X, _ = _training_data()
    with pytest.raises(tabiclv2.TabICLv2ContractError):
        tabiclv2.TabICLv2Classifier("not-reached.ckpt").fit(X, invalid_y)
    assert fake_runtime["imports"] == []


def test_exact_constructor_kwargs_and_training_only_label_encoding(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    X, y = _training_data()
    y = np.where(y == "case", "z-last", "a-first")
    checkpoint = _checkpoint(tmp_path)
    classifier = tabiclv2.TabICLv2Classifier(
        checkpoint, device="cuda:1", random_state=9
    )

    returned = classifier.fit(X, y)
    probabilities = classifier.predict_proba(X[:3] + 0.25)

    assert returned is classifier
    assert fake_runtime["constructor_calls"] == [
        {
            "model_path": str(checkpoint.resolve()),
            "allow_auto_download": False,
            "checkpoint_version": tabiclv2.TABICLV2_CHECKPOINT,
            "device": "cuda:1",
            "random_state": 9,
        }
    ]
    np.testing.assert_array_equal(
        fake_runtime["fit_calls"][0][1], np.resize([1, 0], 300)
    )
    assert len(fake_runtime["predict_calls"]) == 1
    assert fake_runtime["predict_calls"][0].shape == (3, 4)
    np.testing.assert_array_equal(classifier.classes_, ["a-first", "z-last"])
    np.testing.assert_allclose(probabilities, np.tile([[2 / 3, 1 / 3]], (3, 1)))


def test_success_metadata_records_checkpoint_device_limits_classes_and_probability_semantics(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    X, y = _training_data()
    content = b"pinned-tabiclv2-checkpoint"
    checkpoint = _checkpoint(tmp_path)

    classifier = tabiclv2.TabICLv2Classifier(checkpoint).fit(X, y)
    metadata = classifier.metadata_

    assert metadata["package"] == {"name": "tabicl", "version": "2.1.1"}
    assert metadata["upstream_class"] == {
        "module": "tabicl._sklearn.classifier",
        "name": "TabICLClassifier",
    }
    assert metadata["checkpoint"] == {
        "path": str(checkpoint.resolve()),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "repo_id": "jingang/TabICL",
        "revision": tabiclv2.TABICLV2_REVISION,
        "filename": tabiclv2.TABICLV2_CHECKPOINT,
        "pinned_cache_path": "/mock/huggingface/cache/pinned-tabiclv2.ckpt",
    }
    assert metadata["selected_cuda_device"] == "cuda:0"
    assert metadata["class_order"] == ["case", "control"]
    assert metadata["probability"]["columns"] == "class_order"
    assert metadata["effective_limits"] == {
        "min_train_rows": 300,
        "max_train_rows": 100_000,
        "max_features": 2_000,
    }
    assert "upstream" in metadata["preprocessing"]


def test_unfitted_predict_methods_raise() -> None:
    classifier = tabiclv2.TabICLv2Classifier("unused.ckpt")
    with pytest.raises(NotFittedError):
        classifier.predict_proba(np.ones((2, 4)))
    with pytest.raises(NotFittedError):
        classifier.predict(np.ones((2, 4)))


@pytest.mark.parametrize(
    "invalid_X",
    [
        np.empty((0, 4)),
        np.ones((2, 3)),
        np.array([[1.0, 2.0, 3.0, np.nan]]),
        np.array([["1", "2", "3", "4"]]),
    ],
)
def test_predict_validates_nonempty_finite_numeric_feature_contract(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
    invalid_X: np.ndarray,
) -> None:
    X, y = _training_data()
    classifier = tabiclv2.TabICLv2Classifier(_checkpoint(tmp_path)).fit(X, y)
    with pytest.raises(tabiclv2.TabICLv2ContractError):
        classifier.predict_proba(invalid_X)


def test_upstream_fit_class_order_is_validated(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    X, y = _training_data()
    fake_runtime["classes_override"] = [1, 0]

    with pytest.raises(tabiclv2.TabICLv2ContractError, match="classes_"):
        tabiclv2.TabICLv2Classifier(_checkpoint(tmp_path)).fit(X, y)


@pytest.mark.parametrize(
    "output",
    [
        np.ones((3, 3)),
        np.array([[np.nan, 1.0]] * 3),
        np.array([[-0.1, 1.1]] * 3),
        np.zeros((3, 2)),
        np.array([[np.inf, 1.0]] * 3),
        np.array(["not", "a", "matrix"]),
    ],
)
def test_malformed_probabilities_fail_before_normalization(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
    output: np.ndarray,
) -> None:
    X, y = _training_data()
    classifier = tabiclv2.TabICLv2Classifier(_checkpoint(tmp_path)).fit(X, y)
    fake_runtime["predict_output"] = output

    with pytest.raises(tabiclv2.TabICLv2ContractError):
        classifier.predict_proba(X[:3])


def test_mutated_upstream_class_order_fails_before_prediction(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    X, y = _training_data()
    classifier = tabiclv2.TabICLv2Classifier(_checkpoint(tmp_path)).fit(X, y)
    classifier._estimator_.classes_ = np.array([1, 0])

    with pytest.raises(tabiclv2.TabICLv2ContractError, match="classes_"):
        classifier.predict_proba(X[:2])
    assert fake_runtime["predict_calls"] == []


def test_valid_probabilities_normalize_and_predict_in_original_class_order(
    tmp_path: Path,
    fake_runtime: dict[str, Any],
) -> None:
    X, y = _training_data()
    classifier = tabiclv2.TabICLv2Classifier(_checkpoint(tmp_path)).fit(X, y)
    fake_runtime["predict_output"] = np.array([[9.0, 1.0], [2.0, 8.0]])

    probabilities = classifier.predict_proba(X[:2])
    predictions = classifier.predict(X[:2])

    np.testing.assert_allclose(probabilities, [[0.9, 0.1], [0.2, 0.8]])
    np.testing.assert_array_equal(predictions, ["case", "control"])

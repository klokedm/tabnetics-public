from __future__ import annotations

import builtins
import inspect
import sys
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification

import tabnetics.pipeline.balancing as balancing_module
from tabnetics.classification.backends import SklearnBackend
from tabnetics.pipeline.balancing import (
    TRAINING_BALANCE_METHODS,
    TRAINING_BALANCE_SCHEMA_VERSION,
    TrainingBalanceConfig,
    TrainingBalanceContractError,
    apply_training_balance,
)
from tabnetics.pipeline.pipeline import (
    ClassificationConfig,
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
)
from tabnetics.pipeline.estimator import DFFSClassifier
from tabnetics.pipeline.resampling import FitResamplingContext, ResamplingPolicy


def _propensity_partition() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    majority_logits = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    minority_logits = np.asarray([-0.5, 0.0, 0.5, 1.0])
    logits = np.concatenate([majority_logits, minority_logits])
    X = np.column_stack([logits, logits * 0.5]).astype(float)
    y = np.asarray([0] * majority_logits.size + [1] * minority_logits.size)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return X, y, probabilities


def _fake_crossfit(probabilities: np.ndarray):
    def predict(_estimator, X, y, **_kwargs):
        assert X.shape[0] == probabilities.size
        assert set(np.unique(y)) == {0, 1}
        return np.column_stack([1.0 - probabilities, probabilities])

    return predict


def _install_fake_random_samplers(monkeypatch):
    calls: dict[str, list[dict[str, object]]] = {"over": [], "under": []}

    class FakeRandomOverSampler:
        def __init__(self, **kwargs):
            calls["over"].append(dict(kwargs))
            self.random_state = int(kwargs["random_state"])

        def fit_resample(self, X, y):
            y_arr = np.asarray(y)
            rng = np.random.RandomState(self.random_state)
            _, counts = np.unique(y_arr, return_counts=True)
            target = int(np.max(counts))
            selected = list(range(int(y_arr.size)))
            for label in np.unique(y_arr):
                positions = np.flatnonzero(y_arr == label)
                selected.extend(
                    rng.choice(
                        positions,
                        size=target - int(positions.size),
                        replace=True,
                    ).tolist()
                )
            self.sample_indices_ = np.asarray(selected, dtype=int)
            return np.asarray(X)[self.sample_indices_], y_arr[self.sample_indices_]

    class FakeRandomUnderSampler:
        def __init__(self, **kwargs):
            calls["under"].append(dict(kwargs))
            self.random_state = int(kwargs["random_state"])

        def fit_resample(self, X, y):
            y_arr = np.asarray(y)
            rng = np.random.RandomState(self.random_state)
            _, counts = np.unique(y_arr, return_counts=True)
            target = int(np.min(counts))
            selected: list[int] = []
            for label in np.unique(y_arr):
                positions = np.flatnonzero(y_arr == label)
                selected.extend(
                    rng.choice(positions, size=target, replace=False).tolist()
                )
            self.sample_indices_ = np.asarray(selected, dtype=int)
            return np.asarray(X)[self.sample_indices_], y_arr[self.sample_indices_]

    package = ModuleType("imblearn")
    over_sampling = ModuleType("imblearn.over_sampling")
    under_sampling = ModuleType("imblearn.under_sampling")
    over_sampling.RandomOverSampler = FakeRandomOverSampler  # type: ignore[attr-defined]
    under_sampling.RandomUnderSampler = FakeRandomUnderSampler  # type: ignore[attr-defined]
    package.over_sampling = over_sampling  # type: ignore[attr-defined]
    package.under_sampling = under_sampling  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imblearn", package)
    monkeypatch.setitem(sys.modules, "imblearn.over_sampling", over_sampling)
    monkeypatch.setitem(sys.modules, "imblearn.under_sampling", under_sampling)
    return calls


def test_training_balance_config_is_immutable_canonical_and_fingerprinted() -> None:
    config = TrainingBalanceConfig(method=" PROPENSITY_MATCH ")
    assert config.method == "propensity_match"
    assert config.propensity_n_splits == 5
    assert config.propensity_caliper_sd == pytest.approx(0.2)
    assert len(config.fingerprint) == 64
    assert config.fingerprint == TrainingBalanceConfig(method="propensity_match").fingerprint
    with pytest.raises(FrozenInstanceError):
        config.method = "none"  # type: ignore[misc]
    with pytest.raises(ValueError):
        TrainingBalanceConfig(method="unknown")

    record = config.to_dict()
    assert TrainingBalanceConfig.from_mapping(record) == config
    assert TrainingBalanceConfig(**record) == config
    assert DFFSConfig(training_balance=record).training_balance == config
    assert TRAINING_BALANCE_SCHEMA_VERSION == "2.0"
    assert TRAINING_BALANCE_METHODS == (
        "none",
        "smote",
        "propensity_match",
        "random_over",
        "random_under",
    )
    for method in TRAINING_BALANCE_METHODS:
        assert TrainingBalanceConfig(method=method).method == method


def test_none_is_default_off_and_preserves_values_dtype_and_context() -> None:
    X = np.arange(12, dtype=np.int64).reshape(6, 2)
    y = np.asarray([0, 1, 0, 1, 0, 1])
    context = FitResamplingContext.iid(6)
    result = apply_training_balance(X, y, context=context)
    assert result.X is X
    assert result.X.dtype == X.dtype
    np.testing.assert_array_equal(result.y, y)
    assert result.context is context
    assert result.provenance.method == "none"
    assert result.provenance.input_fingerprint == result.provenance.output_fingerprint


@pytest.mark.parametrize(
    ("X", "code"),
    [
        (sparse.csr_matrix(np.eye(8)), "sparse_input_unsupported"),
        (np.ones((8, 2), dtype=object), "continuous_numeric_input_required"),
        (np.ones((8, 2), dtype=np.int64), "continuous_numeric_input_required"),
        (np.asarray([[0.0, np.inf]] * 8), "nonfinite_input"),
    ],
)
def test_enabled_balancing_rejects_noncontinuous_or_nonfinite_matrices(X, code) -> None:
    with pytest.raises(TrainingBalanceContractError) as exc_info:
        apply_training_balance(
            X,
            np.asarray([0] * 6 + [1] * 2),
            config=TrainingBalanceConfig(method="smote", smote_k_neighbors=1),
        )
    assert exc_info.value.code == code


def test_enabled_balancing_rejects_structured_policy() -> None:
    context = FitResamplingContext(
        n_rows=8,
        groups=tuple(range(8)),
        policy=ResamplingPolicy(kind="group", enforced_boundaries=("groups",)),
    )
    with pytest.raises(TrainingBalanceContractError) as exc_info:
        apply_training_balance(
            np.arange(16, dtype=float).reshape(8, 2),
            np.asarray([0] * 6 + [1] * 2),
            config=TrainingBalanceConfig(method="propensity_match", propensity_n_splits=2),
            context=context,
        )
    assert exc_info.value.code == "unsupported_resampling_policy"


def test_smote_fails_closed_for_weights_small_classes_and_missing_extra(monkeypatch) -> None:
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0] * 8 + [1] * 4)
    with pytest.raises(TrainingBalanceContractError) as weighted:
        apply_training_balance(
            X,
            y,
            config=TrainingBalanceConfig(method="smote", smote_k_neighbors=2),
            sample_weight=np.ones(y.size),
        )
    assert weighted.value.code == "smote_sample_weight_unsupported"

    with pytest.raises(TrainingBalanceContractError) as small:
        apply_training_balance(
            X,
            y,
            config=TrainingBalanceConfig(method="smote", smote_k_neighbors=5),
        )
    assert small.value.code == "smote_class_too_small"

    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("imblearn"):
            raise ImportError("blocked for contract test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(TrainingBalanceContractError) as dependency:
        apply_training_balance(
            X,
            y,
            config=TrainingBalanceConfig(method="smote", smote_k_neighbors=2),
        )
    assert dependency.value.code == "smote_dependency_unavailable"
    assert "tabnetics[balancing]" in str(dependency.value)


def test_smote_wrapper_is_deterministic_and_never_synthesizes_context(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeSMOTE:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))

        def fit_resample(self, X, y):
            minority = np.flatnonzero(np.asarray(y) == 1)
            needed = int(np.sum(np.asarray(y) == 0) - minority.size)
            added = minority[np.arange(needed) % minority.size]
            return np.vstack([X, X[added]]), np.concatenate([y, y[added]])

    package = ModuleType("imblearn")
    over_sampling = ModuleType("imblearn.over_sampling")
    over_sampling.SMOTE = FakeSMOTE  # type: ignore[attr-defined]
    package.over_sampling = over_sampling  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imblearn", package)
    monkeypatch.setitem(sys.modules, "imblearn.over_sampling", over_sampling)

    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0] * 8 + [1] * 4)
    context = FitResamplingContext.iid(y.size)
    config = TrainingBalanceConfig(method="smote", smote_k_neighbors=2)
    result = apply_training_balance(X, y, config=config, pipeline_seed=91, context=context)
    assert calls == [
        {"sampling_strategy": "auto", "k_neighbors": 2, "random_state": 91}
    ]
    assert result.X.shape == (16, 2)
    assert result.provenance.synthetic_rows == 4
    assert result.context is None
    assert result.provenance.diagnostics["synthetic_context_rows_persisted"] == 0


def test_random_over_is_seeded_multiclass_and_drops_context(monkeypatch) -> None:
    calls = _install_fake_random_samplers(monkeypatch)
    X = np.arange(33, dtype=float).reshape(11, 3)
    y = np.asarray([0] * 6 + [1] * 3 + [2] * 2)
    context = FitResamplingContext.iid(
        y.size,
        row_ids=tuple(f"private-row-{index}" for index in range(y.size)),
    )
    config = TrainingBalanceConfig(method="random_over", random_state=71)
    first = apply_training_balance(X, y, config=config, context=context)
    second = apply_training_balance(X, y, config=config, context=context)
    different = apply_training_balance(
        X,
        y,
        config=TrainingBalanceConfig(method="random_over", random_state=72),
        context=context,
    )

    assert calls["over"] == [
        {"sampling_strategy": "auto", "random_state": 71},
        {"sampling_strategy": "auto", "random_state": 71},
        {"sampling_strategy": "auto", "random_state": 72},
    ]
    np.testing.assert_array_equal(np.unique(first.y, return_counts=True)[1], [6, 6, 6])
    np.testing.assert_array_equal(first.X, second.X)
    assert not np.array_equal(first.X, different.X)
    assert first.context is None
    assert first.sample_weight is None
    assert first.provenance.synthetic_rows == 0
    diagnostics = first.provenance.to_dict()["diagnostics"]
    assert diagnostics["duplicated_rows"] == 7
    assert diagnostics["duplicated_class_counts"] == [0, 3, 4]
    assert diagnostics["source_row_reuse_total"] == 7
    assert diagnostics["source_rows_reused"] >= 2
    assert diagnostics["max_source_row_reuse"] >= 2
    assert len(diagnostics["source_indices_fingerprint"]) == 64
    assert "private-row" not in str(diagnostics)


def test_random_over_rejects_weights(monkeypatch) -> None:
    _install_fake_random_samplers(monkeypatch)
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0] * 8 + [1] * 4)
    context = FitResamplingContext.iid(
        y.size,
        sample_weights=np.linspace(0.5, 1.5, num=y.size),
    )
    with pytest.raises(TrainingBalanceContractError) as exc_info:
        apply_training_balance(
            X,
            y,
            config=TrainingBalanceConfig(method="random_over"),
            context=context,
        )
    assert exc_info.value.code == "random_over_sample_weight_unsupported"


def test_random_under_subsets_context_and_weights_without_replacement(monkeypatch) -> None:
    calls = _install_fake_random_samplers(monkeypatch)
    X = np.arange(33, dtype=float).reshape(11, 3)
    y = np.asarray([0] * 6 + [1] * 3 + [2] * 2)
    weights = np.linspace(0.25, 1.25, num=y.size)
    context = FitResamplingContext.iid(
        y.size,
        row_ids=tuple(f"row-{index}" for index in range(y.size)),
        sample_weights=weights,
    )
    config = TrainingBalanceConfig(method="random_under", random_state=81)
    first = apply_training_balance(X, y, config=config, context=context)
    second = apply_training_balance(X, y, config=config, context=context)

    assert calls["under"] == [
        {
            "sampling_strategy": "auto",
            "random_state": 81,
            "replacement": False,
        },
        {
            "sampling_strategy": "auto",
            "random_state": 81,
            "replacement": False,
        },
    ]
    assert first.context is not None
    assert len(set(first.context.row_ids)) == first.y.size == 6
    np.testing.assert_array_equal(np.unique(first.y, return_counts=True)[1], [2, 2, 2])
    np.testing.assert_array_equal(first.X, second.X)
    assert first.context.row_ids == second.context.row_ids
    for output_index, row_id in enumerate(first.context.row_ids):
        source_index = int(str(row_id).split("-")[1])
        np.testing.assert_array_equal(first.X[output_index], X[source_index])
        assert first.y[output_index] == y[source_index]
        assert first.sample_weight is not None
        assert first.sample_weight[output_index] == weights[source_index]
    diagnostics = first.provenance.to_dict()["diagnostics"]
    assert diagnostics["retained_class_counts"] == [2, 2, 2]
    assert diagnostics["dropped_class_counts"] == [4, 1, 0]
    assert diagnostics["dropped_rows"] == 5
    assert diagnostics["replacement"] is False
    assert len(diagnostics["selected_lineage_fingerprint"]) == 64


@pytest.mark.parametrize("method", ["random_over", "random_under"])
def test_random_samplers_fail_closed_for_balanced_rare_and_missing_dependency(
    monkeypatch,
    method,
) -> None:
    balanced_X = np.arange(24, dtype=float).reshape(12, 2)
    with pytest.raises(TrainingBalanceContractError) as balanced:
        apply_training_balance(
            balanced_X,
            np.asarray([0] * 6 + [1] * 6),
            config=TrainingBalanceConfig(method=method),
        )
    assert balanced.value.code == f"{method}_balanced_input"

    with pytest.raises(TrainingBalanceContractError) as rare:
        apply_training_balance(
            balanced_X,
            np.asarray([0] * 8 + [1] * 3 + [2]),
            config=TrainingBalanceConfig(method=method),
        )
    assert rare.value.code == f"{method}_class_too_small"

    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("imblearn"):
            raise ImportError("blocked for contract test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(TrainingBalanceContractError) as dependency:
        apply_training_balance(
            balanced_X,
            np.asarray([0] * 8 + [1] * 4),
            config=TrainingBalanceConfig(method=method),
        )
    assert dependency.value.code == f"{method}_dependency_unavailable"
    assert "tabnetics[balancing]" in str(dependency.value)


@pytest.mark.parametrize("method", ["random_over", "random_under"])
@pytest.mark.parametrize(
    ("malformation", "expected_suffix"),
    [
        ("feature_width", "output_invalid"),
        ("nonfinite", "output_invalid"),
        ("auto_counts", "auto_counts_invalid"),
    ],
)
def test_random_samplers_reject_malformed_backend_outputs(
    monkeypatch,
    method,
    malformation,
    expected_suffix,
) -> None:
    class MalformedSampler:
        def __init__(self, **_kwargs):
            pass

        def fit_resample(self, X, y):
            self.sample_indices_ = np.arange(len(y), dtype=int)
            X_out = np.asarray(X).copy()
            if malformation == "feature_width":
                X_out = X_out[:, :-1]
            elif malformation == "nonfinite":
                X_out[0, 0] = np.inf
            return X_out, np.asarray(y).copy()

    package = ModuleType("imblearn")
    over_sampling = ModuleType("imblearn.over_sampling")
    under_sampling = ModuleType("imblearn.under_sampling")
    over_sampling.RandomOverSampler = MalformedSampler  # type: ignore[attr-defined]
    under_sampling.RandomUnderSampler = MalformedSampler  # type: ignore[attr-defined]
    package.over_sampling = over_sampling  # type: ignore[attr-defined]
    package.under_sampling = under_sampling  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imblearn", package)
    monkeypatch.setitem(sys.modules, "imblearn.over_sampling", over_sampling)
    monkeypatch.setitem(sys.modules, "imblearn.under_sampling", under_sampling)

    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0] * 8 + [1] * 4)
    with pytest.raises(TrainingBalanceContractError) as exc_info:
        apply_training_balance(
            X,
            y,
            config=TrainingBalanceConfig(method=method),
        )
    assert exc_info.value.code == f"{method}_{expected_suffix}"


def test_propensity_match_is_deterministic_and_subsets_context_and_weights(monkeypatch) -> None:
    X, y, probabilities = _propensity_partition()
    monkeypatch.setattr(
        balancing_module,
        "cross_val_predict",
        _fake_crossfit(probabilities),
    )
    weights = np.linspace(0.5, 1.4, num=y.size)
    context = FitResamplingContext.iid(
        y.size,
        row_ids=tuple(f"row-{index}" for index in range(y.size)),
        sample_weights=weights,
    )
    config = TrainingBalanceConfig(method="propensity_match", propensity_n_splits=2)
    first = apply_training_balance(X, y, config=config, context=context)
    second = apply_training_balance(X, y, config=config, context=context)

    assert first.context is not None
    expected = {f"row-{index}" for index in [1, 2, 3, 4, 6, 7, 8, 9]}
    assert set(first.context.row_ids) == expected
    for output_index, row_id in enumerate(first.context.row_ids):
        source_index = int(str(row_id).split("-")[1])
        np.testing.assert_array_equal(first.X[output_index], X[source_index])
        assert first.y[output_index] == y[source_index]
        assert first.sample_weight is not None
        assert first.sample_weight[output_index] == weights[source_index]
    assert first.provenance.matched_pairs == 4
    assert first.provenance.provenance_fingerprint == second.provenance.provenance_fingerprint
    assert first.provenance.diagnostics["matching_with_replacement"] is False
    assert first.provenance.diagnostics["minority_match_drops"] == 0


def test_propensity_lineage_order_and_ties_are_permutation_invariant(monkeypatch) -> None:
    row_ids = tuple(f"lineage-{index}" for index in range(8))
    logits_by_id = {
        "lineage-0": 0.10,
        "lineage-1": 0.20,
        "lineage-2": 0.05,
        "lineage-3": 0.20,
        "lineage-4": 0.00,
        "lineage-5": 0.30,
        "lineage-6": 0.15,
        "lineage-7": 0.25,
    }
    label_by_id = {row_id: int(index >= 6) for index, row_id in enumerate(row_ids)}

    def run(order):
        ordered_ids = tuple(row_ids[index] for index in order)
        logits = np.asarray([logits_by_id[row_id] for row_id in ordered_ids])
        X = np.column_stack([logits, logits * 2.0])
        y = np.asarray([label_by_id[row_id] for row_id in ordered_ids])
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        monkeypatch.setattr(
            balancing_module,
            "cross_val_predict",
            _fake_crossfit(probabilities),
        )
        monkeypatch.setattr(balancing_module, "_max_abs_smd", lambda *_args: 0.0)
        return apply_training_balance(
            X,
            y,
            config=TrainingBalanceConfig(
                method="propensity_match",
                propensity_n_splits=2,
                propensity_caliper_sd=10.0,
            ),
            context=FitResamplingContext.iid(y.size, row_ids=ordered_ids),
        )

    original = run(np.arange(8))
    permuted = run(np.asarray([7, 2, 5, 0, 6, 3, 1, 4]))
    assert original.context is not None and permuted.context is not None
    assert original.context.row_ids == permuted.context.row_ids
    assert original.provenance.output_fingerprint == permuted.provenance.output_fingerprint
    assert (
        original.provenance.output_lineage_fingerprint
        == permuted.provenance.output_lineage_fingerprint
    )


def test_provenance_nested_mappings_are_deeply_immutable(monkeypatch) -> None:
    X, y, probabilities = _propensity_partition()
    monkeypatch.setattr(
        balancing_module,
        "cross_val_predict",
        _fake_crossfit(probabilities),
    )
    result = apply_training_balance(
        X,
        y,
        config=TrainingBalanceConfig(method="propensity_match", propensity_n_splits=2),
    )
    with pytest.raises(TypeError):
        result.provenance.config["method"] = "none"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.provenance.diagnostics["caliper"] = 99.0  # type: ignore[index]
    exported = result.provenance.to_dict()
    exported["diagnostics"]["caliper"] = 99.0
    assert result.provenance.to_dict()["diagnostics"]["caliper"] != 99.0


def test_propensity_match_fails_closed_for_multiclass_overlap_and_caliper(monkeypatch) -> None:
    X = np.arange(30, dtype=float).reshape(15, 2)
    with pytest.raises(TrainingBalanceContractError) as multiclass:
        apply_training_balance(
            X,
            np.asarray([0] * 6 + [1] * 5 + [2] * 4),
            config=TrainingBalanceConfig(method="propensity_match", propensity_n_splits=2),
        )
    assert multiclass.value.code == "propensity_binary_only"

    y = np.asarray([0] * 9 + [1] * 6)
    separated = np.asarray([0.01] * 9 + [0.99] * 6)
    monkeypatch.setattr(balancing_module, "cross_val_predict", _fake_crossfit(separated))
    with pytest.raises(TrainingBalanceContractError) as overlap:
        apply_training_balance(
            X,
            y,
            config=TrainingBalanceConfig(method="propensity_match", propensity_n_splits=2),
        )
    assert overlap.value.code == "propensity_empty_common_support"


@pytest.mark.parametrize(
    "method",
    ["propensity_match", "random_over", "random_under"],
)
def test_sklearn_backend_balances_only_explicit_cv_training_folds(
    monkeypatch,
    method,
) -> None:
    X = np.arange(48, dtype=float).reshape(24, 2)
    y = np.asarray([0, 1] * 12)
    folds = [
        (np.arange(0, 12), np.arange(12, 24)),
        (np.arange(12, 24), np.arange(0, 12)),
    ]
    seen: list[np.ndarray] = []

    class Provenance:
        def __init__(self, fingerprint: str):
            self.fingerprint = fingerprint

        def to_dict(self):
            return {"input_fingerprint": self.fingerprint}

    def spy(X_train, y_train, **_kwargs):
        seen.append(np.asarray(X_train).copy())
        fingerprint = str(float(np.sum(X_train)))
        return SimpleNamespace(
            X=np.asarray(X_train),
            y=np.asarray(y_train),
            sample_weight=None,
            context=None,
            provenance=Provenance(fingerprint),
        )

    monkeypatch.setattr(balancing_module, "apply_training_balance", spy)
    backend = SklearnBackend(candidate_names=("lr",), max_train_test_gap=0.0)
    _, _, _, _, _, metadata = backend.fit_and_select(
        X,
        y,
        seed=17,
        n_classes=2,
        class_counts=np.asarray([12, 12]),
        cv_plan=folds,
        training_balance=TrainingBalanceConfig(method=method, propensity_n_splits=2),
        balance_context=FitResamplingContext.iid(y.size),
    )
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], X[folds[0][0]])
    np.testing.assert_array_equal(seen[1], X[folds[1][0]])
    assert metadata["training_balance_enabled"] is True
    assert [record["validation_rows"] for record in metadata["training_balance_cv_provenance"]] == [12, 12]

    heldout_changed_X = X.copy()
    heldout_changed_X[folds[0][1]] += 100_000.0
    heldout_changed_y = y.copy()
    heldout_changed_y[folds[0][1]] = 1 - heldout_changed_y[folds[0][1]]
    _, _, _, _, _, changed_metadata = backend.fit_and_select(
        heldout_changed_X,
        heldout_changed_y,
        seed=17,
        n_classes=2,
        class_counts=np.asarray([12, 12]),
        cv_plan=folds,
        training_balance=TrainingBalanceConfig(method=method, propensity_n_splits=2),
        balance_context=FitResamplingContext.iid(y.size),
    )
    np.testing.assert_array_equal(seen[0], seen[2])
    assert (
        metadata["training_balance_cv_provenance"][0]["input_fingerprint"]
        == changed_metadata["training_balance_cv_provenance"][0]["input_fingerprint"]
    )


def test_pipeline_rejects_backend_and_internal_refit_compositions() -> None:
    balance = TrainingBalanceConfig(method="propensity_match")
    for classification, expected in (
        (ClassificationConfig(backend="flaml", conformal_enabled=False), "backend:flaml"),
        (ClassificationConfig(posthoc_calibration_enabled=True, conformal_enabled=False), "posthoc_calibration_internal_refit"),
        (ClassificationConfig(conformal_enabled=True), "conformal_internal_refit"),
    ):
        pipeline = DistributionFeatureSelectionPipeline(
            DFFSConfig(training_balance=balance, classification=classification)
        )
        with pytest.raises(TrainingBalanceContractError) as exc_info:
            pipeline._validate_training_balance_composition(
                fit_context=FitResamplingContext.iid(20),
                callsite="test",
            )
        assert expected in exc_info.value.diagnostics["unsupported"]

    typed_pipeline = DistributionFeatureSelectionPipeline(
        DFFSConfig(
            training_balance=balance,
            typed_input_enabled=True,
            classification=ClassificationConfig(conformal_enabled=False),
        )
    )
    with pytest.raises(TrainingBalanceContractError) as typed_error:
        typed_pipeline._validate_training_balance_composition(
            fit_context=FitResamplingContext.iid(20),
            callsite="test",
        )
    assert "typed_or_mixed_input" in typed_error.value.diagnostics["unsupported"]


def test_both_final_classifier_fit_sites_apply_training_balance_contract() -> None:
    fit_components = inspect.getsource(
        DistributionFeatureSelectionPipeline._fit_components
    )
    evaluation_fit = inspect.getsource(
        DistributionFeatureSelectionPipeline._run_feature_selection
    )
    assert "fit_components_final_fit" in fit_components
    assert "_apply_final_training_balance" in fit_components
    assert "run_feature_selection_final_fit" in evaluation_fit
    assert "_apply_final_training_balance" in evaluation_fit


def _fast_pipeline_config(
    *,
    seed: int,
    training_balance: TrainingBalanceConfig | None = None,
) -> DFFSConfig:
    kwargs = {}
    if training_balance is not None:
        kwargs["training_balance"] = training_balance
    return DFFSConfig(
        random_seed=seed,
        fs_fraction=1.0,
        n_final_features=4,
        enabled_methods=("anova_f",),
        selection_strategy="legacy_voting",
        use_rank_prefilter=False,
        apply_cdf_transform=False,
        folding_method="none",
        classification=ClassificationConfig(
            model_candidates=("lr",),
            include_elastic_net_model=False,
            include_rf_model=False,
            include_knn_model=False,
            include_svm_linear_model=False,
            include_dlda_model=False,
            include_nb_model=False,
            runtime_containment_enabled=False,
            stage2_max_train_test_gap=0.0,
            stage2_ratio_augmentation_enabled=False,
            posthoc_calibration_enabled=False,
            conformal_enabled=False,
        ),
        **kwargs,
    )


def test_pipeline_omitted_balance_and_explicit_none_are_equivalent() -> None:
    X, y = make_classification(
        n_samples=64,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=801,
    )
    omitted = DFFSClassifier(
        config=_fast_pipeline_config(seed=801),
        random_state=801,
    ).fit(X, y)
    explicit = DFFSClassifier(
        config=_fast_pipeline_config(
            seed=801,
            training_balance=TrainingBalanceConfig(method="none"),
        ),
        random_state=801,
    ).fit(X, y)
    np.testing.assert_array_equal(omitted.predict(X), explicit.predict(X))
    np.testing.assert_allclose(
        omitted.predict_proba(X),
        explicit.predict_proba(X),
        rtol=0.0,
        atol=0.0,
    )
    assert omitted.config_snapshot_ == explicit.config_snapshot_


@pytest.mark.parametrize(
    "balance",
    [
        TrainingBalanceConfig(method="smote", smote_k_neighbors=3),
        TrainingBalanceConfig(method="random_over"),
        TrainingBalanceConfig(method="random_under"),
    ],
    ids=lambda config: config.method,
)
def test_enabled_samplers_run_through_both_dynamic_final_fit_routes(balance) -> None:
    pytest.importorskip("imblearn")
    X, y = make_classification(
        n_samples=96,
        n_features=10,
        n_informative=6,
        n_redundant=1,
        weights=[0.75, 0.25],
        random_state=811,
    )
    config = _fast_pipeline_config(seed=811, training_balance=balance)
    estimator = DFFSClassifier(config=config, random_state=811).fit(X[:72], y[:72])
    assert estimator.config_snapshot_["training_balance"]["method"] == balance.method
    assert "fit_components_final_fit" in estimator.config_snapshot_[
        "training_balance_provenance"
    ]
    cv_records = estimator.fit_provenance_["classifier_selection"][
        "training_balance_cv_provenance"
    ]
    assert cv_records and all(record["validation_rows"] > 0 for record in cv_records)

    evaluation = DistributionFeatureSelectionPipeline(config).run_pre_split(
        X[:72],
        y[:72],
        X[72:],
        y[72:],
        dataset_name="training_balance_dynamic_contract",
        seed=811,
    )
    assert "run_feature_selection_final_fit" in evaluation.config_snapshot[
        "training_balance_provenance"
    ]

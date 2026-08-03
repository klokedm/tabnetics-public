from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from sklearn.model_selection import KFold, train_test_split

from tabnetics.pipeline.resampling import (
    FitResamplingContext,
    ResamplingContractError,
    ResamplingPolicy,
    SplitAssignment,
    resolve_cv,
    resolve_holdout,
    typed_scalar_key,
    validate_supplied_assignments,
)


def test_context_copies_sources_and_preserves_typed_identity() -> None:
    row_ids = np.asarray([1, "1", True, 1.0], dtype=object)
    groups = np.asarray(["a", "b", "c", "d"], dtype=object)
    weights = np.asarray([1.0, 2.0, 3.0, 4.0])

    context = FitResamplingContext(
        n_rows=4,
        row_ids=row_ids,
        groups=groups,
        sample_weights=weights,
    )
    original_fingerprint = context.fingerprint

    row_ids[:] = "mutated"
    groups[:] = "mutated"
    weights[:] = -1.0

    assert context.row_ids == (1, "1", True, 1.0)
    assert context.groups == ("a", "b", "c", "d")
    assert context.sample_weights == (1.0, 2.0, 3.0, 4.0)
    assert context.fingerprint == original_fingerprint
    assert len({typed_scalar_key(value) for value in context.row_ids}) == 4
    assert isinstance(context.row_ids, tuple)
    with pytest.raises(FrozenInstanceError):
        context.n_rows = 5  # type: ignore[misc]


def test_context_fingerprint_binds_row_order_types_weights_and_policy() -> None:
    first = FitResamplingContext(
        n_rows=4,
        row_ids=(1, "1", True, 1.0),
        sample_weights=(1.0, 2.0, 3.0, 4.0),
    )
    same = FitResamplingContext(
        n_rows=4,
        row_ids=(1, "1", True, 1.0),
        sample_weights=(1.0, 2.0, 3.0, 4.0),
    )
    reordered = FitResamplingContext(
        n_rows=4,
        row_ids=("1", 1, True, 1.0),
        sample_weights=(1.0, 2.0, 3.0, 4.0),
    )
    reweighted = FitResamplingContext(
        n_rows=4,
        row_ids=(1, "1", True, 1.0),
        sample_weights=(1.0, 2.0, 4.0, 3.0),
    )

    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != reordered.fingerprint
    assert first.fingerprint != reweighted.fingerprint
    assert first.row_ids_fingerprint == same.row_ids_fingerprint
    assert first.row_ids_fingerprint != reordered.row_ids_fingerprint
    assert first.sample_weights_fingerprint == same.sample_weights_fingerprint
    assert first.sample_weights_fingerprint != reweighted.sample_weights_fingerprint
    assert first.policy_fingerprint == same.policy_fingerprint
    metadata = first.to_metadata()
    assert metadata["row_ids_fingerprint"] == first.row_ids_fingerprint
    assert metadata["policy_fingerprint"] == first.policy_fingerprint
    assert metadata["sample_weights_fingerprint"] == first.sample_weights_fingerprint


@pytest.mark.parametrize(
    ("seed", "test_size", "train_size"),
    [(3, 0.2, None), (19, 0.35, None), (7, 0.2, 14)],
)
def test_default_iid_holdout_matches_legacy_sklearn_indices(
    seed: int,
    test_size: float,
    train_size: int | None,
) -> None:
    y = np.asarray([0, 1] * 12)
    context = FitResamplingContext.iid(len(y))
    kwargs: dict[str, object] = {"random_state": seed, "stratify": y}
    if train_size is None:
        kwargs["test_size"] = test_size
    else:
        kwargs["train_size"] = train_size
    expected_train, expected_test = train_test_split(np.arange(len(y)), **kwargs)

    split = resolve_holdout(
        context,
        y,
        seed=seed,
        test_size=test_size,
        train_size=train_size,
    ).primary

    assert split.train_indices == tuple(int(value) for value in expected_train)
    assert split.test_indices == tuple(int(value) for value in expected_test)
    assert split.audit.ok


def test_declared_impossible_stratification_fails_with_diagnostics() -> None:
    context = FitResamplingContext(
        n_rows=6,
        policy=ResamplingPolicy(kind="stratified"),
    )

    with pytest.raises(ResamplingContractError) as error:
        resolve_holdout(context, [0, 0, 0, 0, 0, 1], seed=4)

    assert error.value.code == "stratification_impossible"
    assert error.value.diagnostics["min_class_count"] == 1


def _connected_group_context() -> tuple[FitResamplingContext, np.ndarray]:
    patient_ids: list[str] = []
    site_ids: list[str] = []
    batch_ids: list[str] = []
    groups: list[str] = []
    labels: list[int] = []
    for component in range(8):
        patient_ids.extend([f"p{component}-a", f"p{component}-a", f"p{component}-b", f"p{component}-b"])
        site_ids.extend([f"s{component}-a", f"s{component}-link", f"s{component}-link", f"s{component}-b"])
        batch_ids.extend([f"b{component}"] * 4)
        groups.extend([f"g{component}"] * 4)
        labels.extend([0, 1, 0, 1])
    context = FitResamplingContext(
        n_rows=len(labels),
        patient_ids=patient_ids,
        site_ids=site_ids,
        batch_ids=batch_ids,
        groups=groups,
        policy=ResamplingPolicy(
            kind="stratified_group",
            enforced_boundaries=("patient_ids", "site_ids", "batch_ids", "groups"),
        ),
    )
    return context, np.asarray(labels)


def _assert_no_identity_overlap(
    context: FitResamplingContext,
    train: tuple[int, ...],
    test: tuple[int, ...],
) -> None:
    for field_name in context.policy.enforced_boundaries:
        values = getattr(context, field_name)
        train_values = {typed_scalar_key(values[index]) for index in train}
        test_values = {typed_scalar_key(values[index]) for index in test}
        assert train_values.isdisjoint(test_values), field_name


def test_connected_identity_components_remain_atomic_in_outer_and_inner_splits() -> None:
    context, y = _connected_group_context()

    outer = resolve_holdout(context, y, seed=11, test_size=0.25).primary
    _assert_no_identity_overlap(context, outer.train_indices, outer.test_indices)
    assert dict(outer.audit.identity_overlap_counts) == {
        "patient_ids": 0,
        "site_ids": 0,
        "batch_ids": 0,
        "groups": 0,
    }

    inner = resolve_cv(
        context,
        y,
        n_splits=4,
        seed=13,
        purpose="classifier_cv",
        stratified=True,
    )
    assert len(inner.splits) == 4
    for split in inner.splits:
        _assert_no_identity_overlap(context, split.train_indices, split.test_indices)
        assert split.audit.class_support_ok


def test_group_stratification_fails_when_class_is_confined_to_one_component() -> None:
    context = FitResamplingContext(
        n_rows=12,
        groups=tuple(group for group in range(6) for _ in range(2)),
        policy=ResamplingPolicy(
            kind="stratified_group",
            enforced_boundaries=("groups",),
        ),
    )
    y = np.asarray([1, 1] + [0] * 10)

    with pytest.raises(ResamplingContractError) as error:
        resolve_holdout(context, y, seed=2, test_size=0.25)

    assert error.value.code in {
        "group_stratification_impossible",
        "split_class_support_missing",
    }


def test_temporal_outer_and_expanding_cv_keep_ties_and_gap_atomic() -> None:
    timestamps = tuple(value for value in range(10) for _ in range(2))
    y = np.asarray([0, 1] * 10)
    context = FitResamplingContext(
        n_rows=len(y),
        timestamps=timestamps,
        policy=ResamplingPolicy(kind="blocked_temporal", temporal_gap=1),
    )

    outer = resolve_holdout(context, y, seed=99, test_size=0.2).primary
    train_times = {timestamps[index] for index in outer.train_indices}
    test_times = {timestamps[index] for index in outer.test_indices}
    assert max(train_times) < min(test_times)
    assert train_times.isdisjoint(test_times)
    assert outer.audit.n_unassigned == 2

    inner = resolve_cv(
        context,
        y,
        n_splits=3,
        seed=1,
        purpose="inner_cv",
        stratified=True,
    )
    assert len(inner.splits) == 3
    for split in inner.splits:
        fold_train_times = {timestamps[index] for index in split.train_indices}
        fold_test_times = {timestamps[index] for index in split.test_indices}
        assert max(fold_train_times) < min(fold_test_times)
        assert fold_train_times.isdisjoint(fold_test_times)


def test_temporal_context_rejects_missing_time() -> None:
    with pytest.raises(ResamplingContractError) as error:
        FitResamplingContext(
            n_rows=4,
            timestamps=(0, 1, None, 3),
            policy=ResamplingPolicy(kind="blocked_temporal"),
        )

    assert error.value.code == "missing_timestamps"


@pytest.mark.parametrize(
    ("assignment", "expected_code"),
    [
        (
            SplitAssignment("outer", "overlap", (0, 1, 2), (2, 3, 4), allow_unassigned=True),
            "train_test_overlap",
        ),
        (
            SplitAssignment("outer", "duplicate", (0, 0, 1), (2, 3, 4), allow_unassigned=True),
            "duplicate_train_index",
        ),
        (
            SplitAssignment("outer", "bounds", (0, 1, 2), (3, 4, 9), allow_unassigned=True),
            "split_index_out_of_bounds",
        ),
        (
            SplitAssignment("outer", "coverage", (0, 1), (2, 3)),
            "incomplete_split_coverage",
        ),
    ],
)
def test_supplied_assignment_validation_is_fail_closed(
    assignment: SplitAssignment,
    expected_code: str,
) -> None:
    base = FitResamplingContext.iid(6)
    context = base.with_supplied_splits((assignment,))

    with pytest.raises(ResamplingContractError) as error:
        validate_supplied_assignments(context, y=[0, 1, 0, 1, 0, 1], scope="outer")

    assert error.value.code == expected_code


def test_materialized_splitter_is_discarded_and_bound_to_context() -> None:
    base = FitResamplingContext.iid(12)
    splitter = KFold(n_splits=3, shuffle=True, random_state=5)
    context = base.materialize_splitter(splitter, y=[0, 1] * 6, scope="inner_cv")

    assert context.policy.kind == "supplied"
    assert len(context.supplied_splits) == 3
    assert all(
        split.parent_context_fingerprint == context.base_fingerprint
        for split in context.supplied_splits
    )
    assert not any(value is splitter for value in context.__dict__.values())
    plan = validate_supplied_assignments(
        context,
        y=[0, 1] * 6,
        scope="inner_cv",
    )
    assert len(plan.splits) == 3
    assert isinstance(plan.splits, tuple)


def test_context_take_slices_weights_and_binds_parent_split() -> None:
    context = FitResamplingContext(
        n_rows=6,
        row_ids=("a", "b", "c", "d", "e", "f"),
        groups=(0, 0, 1, 1, 2, 2),
        sample_weights=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    )

    child = context.take((4, 1, 3), parent_split_fingerprint="split-sha")

    assert child.row_ids == ("e", "b", "d")
    assert child.groups == (2, 0, 1)
    assert child.sample_weights == (5.0, 2.0, 4.0)
    assert child.parent_split_fingerprint == "split-sha"
    assert child.fingerprint != context.fingerprint

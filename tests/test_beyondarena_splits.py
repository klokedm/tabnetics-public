from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from tabnetics.datasets.beyondarena import (
    BeyondArenaUnavailableError,
    build_beyondarena_fallback_split,
    build_beyondarena_inner_validation_policy,
    build_beyondarena_resampling_context,
    load_beyondarena_spec,
    load_beyondarena_splits,
    validate_beyondarena_split_leakage,
)
from tabnetics.pipeline.resampling import (
    ResamplingContractError,
    resolve_cv,
    resolve_holdout,
)


FIXTURES = Path(__file__).parent / "fixtures" / "beyondarena"


def test_grouped_split_guard_rejects_group_overlap() -> None:
    artifact = FIXTURES / "grouped_small" / "v1"
    spec = load_beyondarena_spec(artifact)
    split = load_beyondarena_splits(artifact).splits[0]
    clean = pd.DataFrame(
        {
            "group_id": ["A", "A", "B", "B", "C", "C"],
            "event_date": pd.date_range("2026-01-01", periods=6),
            "x_num": range(6),
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    leaky = clean.copy()
    leaky.loc[4, "group_id"] = "A"

    assert validate_beyondarena_split_leakage(clean, spec, split)["ok"] is True
    result = validate_beyondarena_split_leakage(leaky, spec, split)
    assert result["ok"] is False
    assert result["reason"] == "group labels cross train/test split"


def test_temporal_split_guard_checks_train_precedes_test() -> None:
    artifact = FIXTURES / "temporal_small" / "v1"
    spec = load_beyondarena_spec(artifact)
    split = load_beyondarena_splits(artifact).splits[0]
    clean = pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=6),
            "x_num": range(6),
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    leaky = clean.copy()
    leaky.loc[5, "event_date"] = "2025-12-01"

    assert validate_beyondarena_split_leakage(clean, spec, split)["ok"] is True
    result = validate_beyondarena_split_leakage(leaky, spec, split)
    assert result["ok"] is False
    assert result["reason"] == "temporal train timestamps exceed test horizon"


def test_temporal_split_guard_can_accept_prediction_point_splits() -> None:
    artifact = FIXTURES / "temporal_small" / "v1"
    spec = load_beyondarena_spec(artifact)
    split = replace(
        load_beyondarena_splits(artifact).splits[0],
        allow_temporal_train_after_test=True,
    )
    frame = pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=6),
            "x_num": range(6),
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    frame.loc[5, "event_date"] = "2025-12-01"

    result = validate_beyondarena_split_leakage(frame, spec, split)

    assert result["ok"] is True
    assert result["temporal_order_ok"] is False
    assert result["temporal_train_after_test_allowed"] is True


def test_iid_fallback_is_not_silent_for_non_iid_datasets() -> None:
    spec = load_beyondarena_spec(FIXTURES / "grouped_small" / "v1")

    with pytest.raises(BeyondArenaUnavailableError, match="IID fallback split refused"):
        build_beyondarena_fallback_split(spec, n_samples=10)


def test_inner_validation_policy_matches_beyondarena_row_thresholds() -> None:
    grouped = load_beyondarena_spec(FIXTURES / "grouped_small" / "v1")
    temporal = load_beyondarena_spec(FIXTURES / "temporal_small" / "v1")

    small = build_beyondarena_inner_validation_policy(grouped, n_train_rows=499)
    large = build_beyondarena_inner_validation_policy(temporal, n_train_rows=500)

    assert small.repeats == 5
    assert small.folds == 5
    assert small.stratified is True
    assert small.group_column == "group_id"
    assert large.repeats == 1
    assert large.folds == 8
    assert large.time_column == "event_date"


def test_grouped_official_split_adapts_to_grouped_inner_contract() -> None:
    artifact = FIXTURES / "grouped_small" / "v1"
    spec = load_beyondarena_spec(artifact)
    split = load_beyondarena_splits(artifact).splits[0]
    frame = pd.DataFrame(
        {
            "group_id": ["A", "A", "B", "B", "C", "C"],
            "event_date": pd.date_range("2026-01-01", periods=6),
            "x_num": range(6),
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    context = build_beyondarena_resampling_context(frame, spec, (split,))
    outer = resolve_holdout(
        context,
        frame["target"],
        seed=7,
        purpose="outer",
        supplied_split_id=split.split_id,
    )
    fit_context = context.take(
        outer.primary.train_indices,
        parent_split_fingerprint=outer.primary.fingerprint,
    )
    inner = resolve_cv(
        fit_context,
        frame.iloc[list(split.train_indices)]["target"],
        n_splits=2,
        seed=7,
        purpose="classifier_selection_cv",
        stratified=True,
    )

    assert context.policy.kind == "stratified_group"
    assert outer.primary.audit.identity_overlap_counts == (("groups", 0),)
    assert all(
        fold.audit.identity_overlap_counts == (("groups", 0),)
        for fold in inner.splits
    )


def test_core_temporal_adapter_rejects_future_train_exception() -> None:
    artifact = FIXTURES / "temporal_small" / "v1"
    spec = load_beyondarena_spec(artifact)
    split = replace(
        load_beyondarena_splits(artifact).splits[0],
        allow_temporal_train_after_test=True,
    )
    frame = pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=6),
            "x_num": range(6),
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    frame.loc[5, "event_date"] = "2025-12-01"
    context = build_beyondarena_resampling_context(frame, spec, (split,))

    with pytest.raises(ResamplingContractError) as exc_info:
        resolve_holdout(
            context,
            frame["target"],
            seed=7,
            purpose="outer",
            supplied_split_id=split.split_id,
        )

    assert exc_info.value.code == "temporal_order_violation"

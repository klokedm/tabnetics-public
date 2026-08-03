from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tabnetics.feature_selection.diakrino_identity import build_producer_source_manifest
from tabnetics.feature_selection.diakrino_views import (
    DIAKRINO_FROZEN_VIEW_IDS,
    DIAKRINO_VIEW_SCORE_SOURCE,
    DiakrinoInferenceView,
    DiakrinoPairedPanelChunk,
    DiakrinoViewError,
    build_view_artifact,
    frozen_inference_views,
    paired_panel_chunks,
    uniform_rank_aggregate,
    validate_view_artifact,
)


BINDING = "a" * 64


def _score_inputs(n_features: int) -> dict[str, dict[str, np.ndarray]]:
    chunks = np.arange(n_features, dtype=np.int64) // 2
    return {
        view_id: {
            "prior_logit": np.linspace(-1.0, 1.0, n_features) + index * 0.1,
            "screening_logit": np.linspace(0.5, -0.5, n_features) - index * 0.05,
            "chunk_id": chunks.copy(),
        }
        for index, view_id in enumerate(DIAKRINO_FROZEN_VIEW_IDS)
    }


def test_frozen_views_are_golden_deterministic_reversible_and_support_only() -> None:
    first = frozen_inference_views(
        binding_sha256=BINDING, n_features=7, n_support=5, n_classes=3
    )
    second = frozen_inference_views(
        binding_sha256=BINDING, n_features=7, n_support=5, n_classes=3
    )
    assert first == second
    assert tuple(view.view_id for view in first) == DIAKRINO_FROZEN_VIEW_IDS
    assert tuple(view.seed for view in first) == (
        8908761277098936196,
        1967036739245154414,
        6959474356636290573,
        999190510007342597,
        7533427822143701227,
    )
    assert tuple(view.feature_permutation for view in first) == (
        (0, 1, 2, 3, 4, 5, 6),
        (2, 5, 4, 0, 3, 6, 1),
        (0, 1, 2, 3, 4, 5, 6),
        (0, 1, 2, 3, 4, 5, 6),
        (2, 1, 6, 3, 5, 0, 4),
    )
    assert tuple(view.support_permutation for view in first) == (
        (0, 1, 2, 3, 4),
        (0, 1, 2, 3, 4),
        (3, 0, 1, 4, 2),
        (0, 1, 2, 3, 4),
        (4, 3, 0, 2, 1),
    )
    assert tuple(view.class_rotation for view in first) == (0, 0, 0, 2, 2)

    X = np.arange(35, dtype=float).reshape(5, 7)
    y = np.asarray([0, 1, 2, 0, 1], dtype=np.int64)
    for view in first:
        X_view, y_view, original_feature_ids = view.transform_support(X, y, n_classes=3)
        assert X_view.shape == X.shape
        assert y_view.shape == y.shape
        assert np.array_equal(np.sort(original_feature_ids), np.arange(7))
        observed = np.arange(7, dtype=float)
        assert np.array_equal(
            view.remap_feature_vector(observed)[original_feature_ids], observed
        )


def test_paired_panels_preserve_real_shadow_adjacency_under_every_view() -> None:
    X = np.arange(35, dtype=np.float64).reshape(5, 7)
    shadow = X + 1000.0
    y = np.asarray([0, 1, 2, 0, 1], dtype=np.int64)
    chunks = paired_panel_chunks(
        original_feature_indices=np.arange(X.shape[1], dtype=np.int64),
        candidate_budget=6,
    )
    assert chunks == (
        DiakrinoPairedPanelChunk(0, (0, 1, 2), (0, 2, 4), (1, 3, 5)),
        DiakrinoPairedPanelChunk(1, (3, 4, 5), (0, 2, 4), (1, 3, 5)),
        DiakrinoPairedPanelChunk(2, (6,), (0,), (1,)),
    )
    assert [chunk.manifest_record() for chunk in chunks][0] == {
        "chunk_id": 0,
        "original_feature_indices": [0, 1, 2],
        "real_slots": [0, 2, 4],
        "shadow_slots": [1, 3, 5],
    }

    for view in frozen_inference_views(
        binding_sha256=BINDING, n_features=7, n_support=5, n_classes=3
    ):
        panel, labels, original_ids = view.transform_paired_support(
            X, shadow, y, n_classes=3
        )
        assert panel.shape == (5, 14)
        assert labels.shape == y.shape
        assert np.all(panel[:, 1::2] - panel[:, 0::2] == 1000.0)
        assert np.array_equal(np.sort(original_ids), np.arange(7))

    with pytest.raises(DiakrinoViewError, match="same shape"):
        frozen_inference_views(
            binding_sha256=BINDING, n_features=7, n_support=5, n_classes=3
        )[0].transform_paired_support(X, shadow[:, :-1], y, n_classes=3)
    with pytest.raises(DiakrinoViewError, match="candidate_budget"):
        paired_panel_chunks(original_feature_indices=[0, 1], candidate_budget=1)


def test_uniform_rank_aggregation_is_equal_weight_and_rejects_bad_views() -> None:
    aggregate, dispersion = uniform_rank_aggregate(
        {view_id: np.asarray([1.0, 2.0, 3.0]) for view_id in DIAKRINO_FROZEN_VIEW_IDS}
    )
    assert np.allclose(aggregate, [0.0, 0.5, 1.0])
    assert np.allclose(dispersion, 0.0)

    with pytest.raises(DiakrinoViewError, match="exactly the frozen"):
        uniform_rank_aggregate({"identity": np.asarray([1.0, 2.0])})
    with pytest.raises(DiakrinoViewError, match="not finite"):
        uniform_rank_aggregate(
            {view_id: np.asarray([1.0, np.nan]) for view_id in DIAKRINO_FROZEN_VIEW_IDS}
        )


def test_view_artifact_rederives_raw_chain_and_rejects_semantic_tampering() -> None:
    artifact = build_view_artifact(
        binding_sha256=BINDING,
        n_features=7,
        n_support=5,
        n_classes=3,
        score_inputs=_score_inputs(7),
    )
    validated = validate_view_artifact(
        artifact,
        binding_sha256=BINDING,
        n_features=7,
        n_support=5,
        n_classes=3,
    )
    assert validated.view_ids == DIAKRINO_FROZEN_VIEW_IDS
    assert validated.score_source == DIAKRINO_VIEW_SCORE_SOURCE
    assert len(validated.uniform_rank_std) == 7

    artifact["views"][0]["score_inputs"]["prior_logit"][0] += 0.25  # type: ignore[index]
    with pytest.raises(DiakrinoViewError, match="derived scores"):
        validate_view_artifact(
            artifact,
            binding_sha256=BINDING,
            n_features=7,
            n_support=5,
            n_classes=3,
        )


def test_view_artifact_rejects_permutation_and_missing_raw_input_tampering() -> None:
    artifact = build_view_artifact(
        binding_sha256=BINDING,
        n_features=7,
        n_support=5,
        n_classes=3,
        score_inputs=_score_inputs(7),
    )
    permutation = artifact["views"][1]["feature_permutation"]  # type: ignore[index]
    permutation[0], permutation[1] = permutation[1], permutation[0]
    with pytest.raises(DiakrinoViewError, match="frozen view contract"):
        validate_view_artifact(
            artifact,
            binding_sha256=BINDING,
            n_features=7,
            n_support=5,
            n_classes=3,
        )

    missing = build_view_artifact(
        binding_sha256=BINDING,
        n_features=3,
        n_support=3,
        n_classes=2,
        score_inputs=_score_inputs(3),
    )
    missing["views"][0]["score_inputs"].pop("chunk_id")  # type: ignore[index]
    with pytest.raises(DiakrinoViewError, match="missing or malformed"):
        validate_view_artifact(
            missing,
            binding_sha256=BINDING,
            n_features=3,
            n_support=3,
            n_classes=2,
        )


@pytest.mark.parametrize("value", [True, "7", 7.5, 2**63])
def test_view_dimensions_reject_non_exact_or_overflow_values(value) -> None:
    with pytest.raises(DiakrinoViewError):
        frozen_inference_views(
            binding_sha256=BINDING,
            n_features=value,
            n_support=5,
            n_classes=3,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("seed", "1"),
        ("seed", 1.5),
        ("seed", 2**63),
        ("class_rotation", True),
        ("class_rotation", "1"),
        ("class_rotation", 0.5),
        ("class_rotation", 2**63),
    ],
)
def test_direct_view_dataclass_rejects_lossy_seed_and_rotation(
    field: str, value
) -> None:
    base = DiakrinoInferenceView(
        view_id="identity",
        seed=1,
        feature_permutation=(0, 1),
        support_permutation=(0, 1),
        class_rotation=0,
    )
    with pytest.raises(DiakrinoViewError):
        replace(base, **{field: value}).validate(
            n_features=2,
            n_support=2,
            n_classes=2,
        )


@pytest.mark.parametrize("value", [True, "0", 0.5, 2**63])
def test_direct_view_dataclass_rejects_lossy_permutation_entries(value) -> None:
    view = DiakrinoInferenceView(
        view_id="identity",
        seed=1,
        feature_permutation=(value, 1),
        support_permutation=(0, 1),
        class_rotation=0,
    )
    with pytest.raises(DiakrinoViewError):
        view.validate(n_features=2, n_support=2, n_classes=2)


def test_direct_view_dataclass_rejects_transform_semantics_cross_wire() -> None:
    view = DiakrinoInferenceView(
        view_id="identity",
        seed=1,
        feature_permutation=(1, 0),
        support_permutation=(0, 1),
        class_rotation=0,
    )
    with pytest.raises(DiakrinoViewError, match="inconsistent with the frozen view id"):
        view.validate(n_features=2, n_support=2, n_classes=2)


def test_view_artifact_rejects_lossy_json_integer_fields() -> None:
    for field, value in (
        ("n_features", True),
        ("n_support", "5"),
        ("n_classes", 3.5),
    ):
        artifact = build_view_artifact(
            binding_sha256=BINDING,
            n_features=7,
            n_support=5,
            n_classes=3,
            score_inputs=_score_inputs(7),
        )
        artifact[field] = value
        with pytest.raises(DiakrinoViewError):
            validate_view_artifact(
                artifact,
                binding_sha256=BINDING,
                n_features=7,
                n_support=5,
                n_classes=3,
            )


def test_view_source_bytes_are_in_producer_source_closure(tmp_path: Path) -> None:
    view_path = (
        tmp_path / "core" / "src" / "tabnetics" / "feature_selection" / "diakrino_views.py"
    )
    view_path.parent.mkdir(parents=True)
    view_path.write_text("VIEW_CONTRACT = 'first'\n", encoding="utf-8")
    first = build_producer_source_manifest(tmp_path)
    assert [item["path"] for item in first["files"]] == [
        "core/src/tabnetics/feature_selection/diakrino_views.py"
    ]

    view_path.write_text("VIEW_CONTRACT = 'other'\n", encoding="utf-8")
    second = build_producer_source_manifest(tmp_path)
    assert first["files"][0]["sha256"] != second["files"][0]["sha256"]
    assert first["manifest_sha256"] != second["manifest_sha256"]

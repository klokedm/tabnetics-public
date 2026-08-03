from __future__ import annotations

import copy
import hashlib
import json
import numpy as np
import pytest

from tabnetics.feature_selection.diakrino_closeout import (
    DiakrinoCloseoutConfig,
    DiakrinoCloseoutError,
    build_support_only_null_design,
    build_native_null_artifact,
    build_paired_native_null_artifact,
    dynamic_addition_budget,
    finalize_closeout_decision,
    jmi_admit,
    matched_control_additions,
    native_null_proposals,
    nogueira_selected_set_stability,
    support_only_classical_scores,
    validate_native_null_artifact,
    validate_paired_native_null_artifact,
)
from tabnetics.feature_selection.diakrino_views import paired_panel_chunks
from tabnetics.pipeline.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline


def test_support_only_nulls_are_deterministic_and_preserve_marginals() -> None:
    X = np.arange(48, dtype=float).reshape(12, 4)
    y = np.asarray([0] * 6 + [1] * 4 + [2] * 2)

    first = build_support_only_null_design(X, y, seed=29)
    second = build_support_only_null_design(X, y, seed=29)

    assert np.array_equal(first.shadow_support, second.shadow_support)
    assert np.array_equal(first.label_null, second.label_null)
    assert sorted(first.label_null.tolist()) == sorted(y.tolist())
    assert not np.array_equal(first.label_null, y)
    for column, permutation in enumerate(first.shadow_permutations):
        assert np.array_equal(first.shadow_support[:, column], X[list(permutation), column])
        assert sorted(first.shadow_support[:, column].tolist()) == sorted(X[:, column].tolist())


def test_support_only_nulls_do_not_accept_query_arrays_or_degenerate_support() -> None:
    with pytest.raises(TypeError):
        build_support_only_null_design(  # type: ignore[call-arg]
            np.ones((4, 2)), np.asarray([0, 0, 1, 1]), seed=1, X_query=np.ones((2, 2))
        )
    with pytest.raises(DiakrinoCloseoutError, match="two rows and classes"):
        build_support_only_null_design(np.ones((4, 2)), np.zeros(4), seed=1)


def _native_null_artifact_fixture() -> tuple[dict[str, object], dict[str, object]]:
    X = np.arange(40, dtype=float).reshape(10, 4)
    y = np.tile([0, 1], 5)
    ranks_by_view = np.asarray(
        [[0.9, 0.6, 0.3, 0.0], [0.8, 0.7, 0.2, 0.1]], dtype=float
    )
    real = ranks_by_view.mean(axis=0)
    rank_std = ranks_by_view.std(axis=0)
    kwargs: dict[str, object] = {
        "binding_sha256": "a" * 64,
        "X_support": X,
        "y_support": y,
        "seed": 29,
        "expected_real_rank": real,
        "expected_rank_std": rank_std,
        "expected_view_ids": ["identity", "combined"],
        "expected_ranks_by_view": ranks_by_view,
    }
    artifact = build_native_null_artifact(
        binding_sha256="a" * 64,
        X_support=X,
        y_support=y,
        seed=29,
        real_rank=real,
        shadow_rank=[0.2, 0.2, 0.1, 0.0],
        label_null_rank=[0.1, 0.3, 0.2, 0.0],
        rank_std=rank_std,
        view_ids=["identity", "combined"],
        ranks_by_view=ranks_by_view,
    )
    return artifact, kwargs


def test_native_null_artifact_roundtrip_binds_support_and_real_views() -> None:
    artifact, kwargs = _native_null_artifact_fixture()
    validated = validate_native_null_artifact(artifact, **kwargs)

    assert validated.binding_sha256 == "a" * 64
    assert validated.shadow_rank == pytest.approx((0.2, 0.2, 0.1, 0.0))
    assert validated.view_ids == ("identity", "combined")


@pytest.mark.parametrize("dimension", [10.0, True, "10"])
def test_native_null_artifact_rejects_non_exact_integer_dimensions(
    dimension: object,
) -> None:
    artifact, kwargs = _native_null_artifact_fixture()
    artifact["n_support"] = dimension
    with pytest.raises(DiakrinoCloseoutError, match="identity is cross-wired"):
        validate_native_null_artifact(artifact, **kwargs)


def test_native_null_artifact_rejects_tampering_and_cross_wires() -> None:
    artifact, kwargs = _native_null_artifact_fixture()
    mutations: list[tuple[dict[str, object], str]] = []

    transformed = copy.deepcopy(artifact)
    transformed["transformations"]["label_permutation"][0] = 9  # type: ignore[index]
    mutations.append((transformed, "transformations are not support-only"))

    semantics = copy.deepcopy(artifact)
    semantics["scores"]["score_source"]["calibration"] = "none"  # type: ignore[index]
    mutations.append((semantics, "score-source identity"))

    out_of_range = copy.deepcopy(artifact)
    out_of_range["scores"]["shadow_uniform_rank01"][0] = 1.1  # type: ignore[index]
    # Recompute is intentionally omitted: either the semantic digest or range
    # check must reject the mutated bytes.
    mutations.append((out_of_range, "semantic SHA-256"))

    for mutated, message in mutations:
        with pytest.raises(DiakrinoCloseoutError, match=message):
            validate_native_null_artifact(mutated, **kwargs)


def _rehash_native_null_artifact(artifact: dict[str, object]) -> None:
    semantic = dict(artifact)
    semantic.pop("semantic_sha256")
    artifact["semantic_sha256"] = hashlib.sha256(
        json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_native_null_artifact_rejects_rehashed_invalid_rank_and_view_crosswire() -> None:
    artifact, kwargs = _native_null_artifact_fixture()
    invalid_rank = copy.deepcopy(artifact)
    invalid_rank["scores"]["shadow_uniform_rank01"][0] = 1.1  # type: ignore[index]
    _rehash_native_null_artifact(invalid_rank)
    with pytest.raises(DiakrinoCloseoutError, match="out of range"):
        validate_native_null_artifact(invalid_rank, **kwargs)

    crosswired = copy.deepcopy(artifact)
    crosswired["scores"]["real_uniform_rank01"] = [0.0, 0.3, 0.6, 0.9]  # type: ignore[index]
    crosswired["scores"]["real_rank01_by_view"] = [  # type: ignore[index]
        [0.0, 0.2, 0.6, 0.9],
        [0.1, 0.4, 0.5, 0.8],
    ]
    _rehash_native_null_artifact(crosswired)
    with pytest.raises(DiakrinoCloseoutError, match="real-view evidence is cross-wired"):
        validate_native_null_artifact(crosswired, **kwargs)


def test_paired_native_null_artifact_binds_same_panel_ledger_and_layout() -> None:
    X = np.arange(40, dtype=float).reshape(10, 4)
    y = np.tile([0, 1], 5)
    panels = [item.manifest_record() for item in paired_panel_chunks(
        original_feature_indices=np.arange(4, dtype=np.int64), candidate_budget=4
    )]
    layouts = {"identity": panels, "combined": panels}
    real = np.asarray([[0.1, 0.4, 0.7, 1.0], [0.2, 0.5, 0.8, 0.9]])
    artifact = build_paired_native_null_artifact(
        binding_sha256="a" * 64,
        X_support=X,
        y_support=y,
        seed=29,
        view_ids=["identity", "combined"],
        paired_view_artifact_sha256="b" * 64,
        panel_chunks_by_view=layouts,
        real_rank01_by_view=real,
        shadow_rank01_by_view=real[::-1],
        label_null_rank01_by_view=np.flip(real, axis=1),
    )
    validated = validate_paired_native_null_artifact(
        artifact,
        binding_sha256="a" * 64,
        X_support=X,
        y_support=y,
        seed=29,
        expected_paired_view_artifact_sha256="b" * 64,
        expected_panel_chunks_by_view=layouts,
        expected_real_rank01_by_view=real,
    )
    assert validated.real_rank == pytest.approx((0.15, 0.45, 0.75, 0.95))

    tampered = copy.deepcopy(artifact)
    tampered["panel"]["layouts_by_view"]["identity"][0]["shadow_slots"] = [0, 1]  # type: ignore[index]
    _rehash_native_null_artifact(tampered)
    with pytest.raises(DiakrinoCloseoutError, match="slot map"):
        validate_paired_native_null_artifact(
            tampered,
            binding_sha256="a" * 64,
            X_support=X,
            y_support=y,
            seed=29,
            expected_paired_view_artifact_sha256="b" * 64,
            expected_panel_chunks_by_view=layouts,
            expected_real_rank01_by_view=real,
        )


def test_nogueira_uses_full_feature_width_not_selected_union() -> None:
    selected = ({0, 1}, {0, 2}, {0, 1})
    observed = nogueira_selected_set_stability(selected, n_features=10)
    wrong_union_width = nogueira_selected_set_stability(selected, n_features=3)

    assert observed == pytest.approx(0.5833333333333333)
    assert observed != pytest.approx(wrong_union_width)
    assert nogueira_selected_set_stability(({0},), n_features=10) is None


def test_native_null_proposals_gate_each_signal_and_truthfully_abstain() -> None:
    config = DiakrinoCloseoutConfig(
        shadow_margin_min=0.1,
        label_null_margin_min=0.1,
        rank_std_max=0.2,
        selected_set_stability_min=0.0,
        proposal_pool_multiplier=2,
    )
    views = np.asarray(
        [
            [0.95, 0.85, 0.75, 0.35, 0.25, 0.15],
            [0.94, 0.84, 0.74, 0.36, 0.24, 0.14],
            [0.93, 0.83, 0.73, 0.37, 0.23, 0.13],
        ]
    )
    proposals, diagnostics = native_null_proposals(
        real_rank=views.mean(axis=0),
        shadow_rank=[0.1, 0.1, 0.7, 0.1, 0.1, 0.1],
        label_null_rank=[0.1, 0.1, 0.1, 0.3, 0.1, 0.1],
        rank_std=views.std(axis=0),
        ranks_by_view=views,
        protected_core=[0],
        config=config,
    )

    assert proposals == (1,)
    assert diagnostics["selected_set_stability_feature_width"] == 6
    assert diagnostics["selected_set_stability_pass"] is True

    abstained, unstable = native_null_proposals(
        real_rank=views.mean(axis=0),
        shadow_rank=[0.1] * 6,
        label_null_rank=[0.1] * 6,
        rank_std=views.std(axis=0),
        ranks_by_view=np.asarray([views[0], views[0][::-1], views[1]]),
        protected_core=[0, 1],
        config=DiakrinoCloseoutConfig(selected_set_stability_min=0.9),
    )
    assert abstained == ()
    assert unstable["selected_set_stability_pass"] is False


def test_native_null_proposals_reject_cross_wired_widths_and_nonfinite_values() -> None:
    kwargs = {
        "real_rank": [0.9, 0.8, 0.7],
        "shadow_rank": [0.2, 0.2, 0.2],
        "label_null_rank": [0.1, 0.1, 0.1],
        "rank_std": [0.1, 0.1, 0.1],
        "ranks_by_view": [[0.9, 0.8, 0.7], [0.8, 0.9, 0.7]],
        "protected_core": [0],
        "config": DiakrinoCloseoutConfig(),
    }
    with pytest.raises(DiakrinoCloseoutError, match="shadow_rank"):
        native_null_proposals(**{**kwargs, "shadow_rank": [0.2, 0.2]})
    with pytest.raises(DiakrinoCloseoutError, match="label_null_rank"):
        native_null_proposals(**{**kwargs, "label_null_rank": [0.1, np.nan, 0.1]})


def test_native_null_proposal_pool_excludes_gated_tail_features() -> None:
    proposals, diagnostics = native_null_proposals(
        real_rank=[0.9, 0.8, 0.7, 0.6, 0.5],
        shadow_rank=[0.0] * 5,
        label_null_rank=[0.0] * 5,
        rank_std=[0.0] * 5,
        ranks_by_view=[
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [0.9, 0.8, 0.7, 0.6, 0.5],
        ],
        protected_core=[0],
        config=DiakrinoCloseoutConfig(proposal_pool_multiplier=2),
    )

    assert proposals == (1,)
    assert diagnostics["feature_gate_pass_count"] == 5
    assert diagnostics["proposal_pool_size"] == 2


def test_jmi_admission_is_support_only_bounded_and_never_evicts_core() -> None:
    rng = np.random.default_rng(47)
    y = np.repeat([0, 1], 20)
    signal = y + rng.normal(0.0, 0.05, y.size)
    duplicate = signal + rng.normal(0.0, 0.01, y.size)
    interaction = (np.arange(y.size) % 3) + y
    noise = rng.normal(size=y.size)
    X = np.column_stack([signal, duplicate, interaction, noise])

    admitted, ledger = jmi_admit(
        X,
        y,
        protected_core=[0],
        proposals=[1, 2, 3],
        real_rank=[1.0, 0.95, 0.9, 0.2],
        budget=2,
    )

    assert len(admitted) == 2
    assert 0 not in admitted
    assert len(ledger) == 2
    assert set(ledger[0]) == {
        "feature_index",
        "criterion",
        "diakrino_relevance_rank01",
        "diakrino_relevance_nats",
        "label_entropy_nats",
        "mean_redundancy_mi",
        "mean_conditional_complementarity_mi",
    }
    entropy = ledger[0]["label_entropy_nats"]
    assert entropy == pytest.approx(np.log(2.0))
    assert ledger[0]["diakrino_relevance_nats"] <= entropy


def test_c1_classical_scores_are_dense_support_only_and_fail_closed() -> None:
    X = np.column_stack(
        [np.arange(12, dtype=float), np.tile([0.0, 1.0], 6), np.ones(12)]
    )
    y = np.tile([0, 1], 6)
    scores = support_only_classical_scores(X, y)

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert scores[1] > scores[2]
    tied = support_only_classical_scores(np.ones((6, 3)), np.asarray([0, 0, 0, 1, 1, 1]))
    assert np.array_equal(tied, np.asarray([0.5, 0.5, 0.5]))
    with pytest.raises(DiakrinoCloseoutError, match="must be finite"):
        support_only_classical_scores(
            np.asarray([[0.0, np.nan], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
            np.asarray([0, 0, 1, 1]),
        )


@pytest.mark.parametrize(
    "arm", ["classical_next_best", "random_extras", "permuted_diakrino_ranks"]
)
def test_controls_match_p4_realized_budget_exactly_and_are_deterministic(arm: str) -> None:
    kwargs = {
        "protected_core": [0, 2, 4, 6],
        "realized_additions": 1,
        "n_features": 8,
        "seed": 83,
        "classical_scores": np.linspace(0.0, 1.0, 8),
        "diakrino_ranks": np.linspace(1.0, 0.0, 8),
    }
    first = matched_control_additions(arm, **kwargs)
    second = matched_control_additions(arm, **kwargs)

    assert first == second
    assert len(first) == 1
    assert not set(first) & {0, 2, 4, 6}


def test_exact_fallback_identity_and_dynamic_budget() -> None:
    assert dynamic_addition_budget(1) == 1
    assert dynamic_addition_budget(8) == 2
    assert dynamic_addition_budget(100) == 10

    fallback = finalize_closeout_decision(
        "protected_native_null_abstain",
        protected_core=[4, 1],
        additions=[],
        n_features=8,
        reason="native_null_abstain",
    )
    assert fallback.final == fallback.protected_core == (4, 1)
    assert fallback.abstained is True
    assert fallback.fallback_exact is True

    with pytest.raises(DiakrinoCloseoutError, match="overlap"):
        finalize_closeout_decision(
            "protected_native_null_jmi",
            protected_core=[0],
            additions=[0],
            n_features=4,
            reason="invalid",
        )


def _pipeline_closeout_context(width: int) -> dict[str, object]:
    indices = np.arange(width, dtype=int)
    return {
        "active_original_indices": indices,
        "initial_local_by_original": {int(index): int(index) for index in indices},
    }


def _pipeline_closeout_evidence(arm: str, width: int) -> dict[str, object]:
    ranks = np.linspace(1.0, 0.1, width)
    return {
        "arm": arm,
        "n_features": width,
        "real_rank": ranks,
        "shadow_rank": np.zeros(width),
        "label_null_rank": np.zeros(width),
        "rank_std": np.zeros(width),
        "ranks_by_view": np.vstack([ranks, ranks, ranks]),
        "thresholds": {
            "shadow_margin_min": 0.0,
            "label_null_margin_min": 0.0,
            "rank_std_max": 0.25,
            "selected_set_stability_min": 0.0,
            "proposal_pool_multiplier": 4,
            "discretization_bins": 5,
        },
    }


def test_pipeline_closeout_hook_preserves_core_and_returns_exact_fallback() -> None:
    pipeline = DistributionFeatureSelectionPipeline(DFFSConfig())
    width = 6
    pipeline._diakrino_closeout_evidence = _pipeline_closeout_evidence(  # noqa: SLF001
        "protected_native_null_abstain", width
    )
    pipeline._diakrino_closeout_evidence["shadow_rank"] = np.ones(width)  # type: ignore[index]
    X = np.arange(48, dtype=float).reshape(8, width)
    pairs, diagnostics = pipeline._diakrino_closeout_admitted_pairs(  # noqa: SLF001
        X_support=X,
        y_support=np.tile([0, 1], 4),
        seed=29,
        protected_ctx=_pipeline_closeout_context(width),
        protected_core_indices=np.asarray([0, 1]),
    )

    assert pairs == []
    assert diagnostics["abstained"] is True
    assert diagnostics["fallback_exact"] is True
    assert diagnostics["addition_budget"] == 1


@pytest.mark.parametrize(
    "arm", ["classical_next_best", "random_extras", "permuted_diakrino_ranks"]
)
def test_pipeline_closeout_controls_use_exact_p4_realized_count(arm: str) -> None:
    pipeline = DistributionFeatureSelectionPipeline(DFFSConfig())
    width = 8
    evidence = _pipeline_closeout_evidence(arm, width)
    evidence["p4_realized_additions"] = 1
    pipeline._diakrino_closeout_evidence = evidence  # noqa: SLF001
    X = np.arange(96, dtype=float).reshape(12, width)
    pairs, diagnostics = pipeline._diakrino_closeout_admitted_pairs(  # noqa: SLF001
        X_support=X,
        y_support=np.tile([0, 1], 6),
        seed=47,
        protected_ctx=_pipeline_closeout_context(width),
        protected_core_indices=np.asarray([0, 2, 4, 6]),
    )

    assert len(pairs) == 1
    assert diagnostics["control_budget_matched"] is True
    assert diagnostics["realized_additions"] == 1
    assert not {original for original, _ in pairs} & {0, 2, 4, 6}

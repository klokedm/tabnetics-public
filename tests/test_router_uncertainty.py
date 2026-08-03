from __future__ import annotations

import hashlib

import pytest

from tabnetics.auto_router.uncertainty import (
    CrossFitRouterUncertaintyArtifact,
    RouterOutcomeRow,
    RouterUncertaintyError,
    fit_crossfit_router_uncertainty,
    router_candidate_schema_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _rows() -> list[RouterOutcomeRow]:
    rows = []
    for source_index, source_id in enumerate(("source_a", "source_b", "source_c", "source_d")):
        for seed in (11, 29, 47):
            rows.append(
                RouterOutcomeRow(
                    dataset_id=f"dataset_{source_index}", source_id=source_id, seed=seed,
                    split_fingerprint=_sha(f"split-{source_id}-{seed}"),
                    descriptor_sha256=_sha(f"descriptor-{source_id}"), candidate_id="candidate",
                    fold_id="fold_0" if source_index % 2 == 0 else "fold_1",
                    predicted_delta_utility=0.20,
                    realized_delta_utility=0.10 if seed != 47 else 0.05,
                    beats_default_probability=0.90,
                )
            )
    return rows


def _artifact() -> CrossFitRouterUncertaintyArtifact:
    return fit_crossfit_router_uncertainty(
        _rows(), base_router_sha256=_sha("router"), descriptor_schema_sha256=_sha("descriptor-schema"),
        frozen_source_ids=("frozen_source",), minimum_support=3,
    )


def test_crossfit_artifact_is_hash_only_and_runtime_bound() -> None:
    artifact = _artifact()
    payload = artifact.to_dict()
    assert "source_a" not in str(payload)
    assert artifact.from_dict(payload).to_dict() == payload
    artifact.validate_runtime(
        base_router_sha256=_sha("router"), candidate_ids=("candidate",),
        descriptor_schema_sha256=_sha("descriptor-schema"),
    )
    with pytest.raises(RouterUncertaintyError, match="candidate schema"):
        artifact.validate_runtime(
            base_router_sha256=_sha("router"), candidate_ids=("other",),
            descriptor_schema_sha256=_sha("descriptor-schema"),
        )


def test_crossfit_artifact_defaults_when_lower_bound_or_probability_fails() -> None:
    artifact = _artifact()
    selected, reason, lower = artifact.decide(
        candidate_id="candidate", predicted_delta_utility=0.01, beats_default_probability=0.99
    )
    assert selected is False
    assert reason == "nonpositive_delta_lower_bound"
    assert lower <= 0.0
    selected, reason, _lower = artifact.decide(
        candidate_id="candidate", predicted_delta_utility=0.30, beats_default_probability=0.0
    )
    assert selected is False
    assert reason == "calibrated_probability_below_threshold"


def test_crossfit_rejects_source_leakage_frozen_sources_and_duplicate_rows() -> None:
    rows = _rows()
    leaked = list(rows)
    leaked[1] = RouterOutcomeRow(
        **{**leaked[1].ledger_record(), "fold_id": "fold_other"}
    )
    with pytest.raises(RouterUncertaintyError, match="source group spans"):
        fit_crossfit_router_uncertainty(
            leaked, base_router_sha256=_sha("router"), descriptor_schema_sha256=_sha("descriptor"),
            frozen_source_ids=(), minimum_support=2,
        )
    with pytest.raises(RouterUncertaintyError, match="frozen source"):
        fit_crossfit_router_uncertainty(
            rows, base_router_sha256=_sha("router"), descriptor_schema_sha256=_sha("descriptor"),
            frozen_source_ids=("source_a",), minimum_support=2,
        )
    with pytest.raises(RouterUncertaintyError, match="duplicate"):
        fit_crossfit_router_uncertainty(
            [*rows, rows[0]], base_router_sha256=_sha("router"), descriptor_schema_sha256=_sha("descriptor"),
            frozen_source_ids=(), minimum_support=2,
        )


def test_candidate_schema_is_order_independent() -> None:
    assert router_candidate_schema_sha256(("b", "a", "b")) == router_candidate_schema_sha256(("a", "b"))


def test_crossfit_policy_requires_leave_fold_out_calibration_support() -> None:
    rows = _rows()
    one_fold_candidate = [
        RouterOutcomeRow(**{**row.ledger_record(), "fold_id": "fold_0"})
        for row in rows
    ]
    with pytest.raises(RouterUncertaintyError, match="at least two source folds"):
        fit_crossfit_router_uncertainty(
            one_fold_candidate, base_router_sha256=_sha("router"),
            descriptor_schema_sha256=_sha("descriptor"), frozen_source_ids=(), minimum_support=2,
        )

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tabnetics.datasets.external_validation import (
    DeclaredFeatureMapping,
    DeclaredLabelMapping,
    ExternalCohortContractError,
    ExternalCohortFamily,
    ExternalCohortManifest,
    ExternalMappingError,
    evaluate_external_source,
    map_external_cohort,
    skipped_external_source,
    summarize_external_sources,
)
from tabnetics.pipeline.resampling import (
    FitResamplingContext,
    ResamplingPolicy,
    ResamplingContractError,
    resolve_leave_one_source_out,
)


def _manifest(
    cohort_id: str,
    *,
    study_id: str | None = None,
    evidence_kind: str = "real_public",
    feature_namespace_version: str = "v1",
    output_feature_order_sha256: str | None = None,
) -> ExternalCohortManifest:
    digest = "a" * 64
    return ExternalCohortManifest(
        family_id="brca_pam50",
        cohort_id=cohort_id,
        study_id=study_id or cohort_id,
        site_id=f"site_{cohort_id}",
        assay="expression",
        platform="microarray",
        preprocessing_id="rma_v1",
        public_source_id=f"gdc_{cohort_id}",
        public_source_revision="2026_07",
        data_artifact_sha256=digest,
        label_namespace_id="pam50_intrinsic",
        label_namespace_sha256="b" * 64,
        feature_namespace_id="hgnc",
        feature_namespace_version=feature_namespace_version,
        feature_mapping_source="hgnc_release",
        feature_mapping_sha256="c" * 64,
        mapping_code_sha256="d" * 64,
        output_feature_order_sha256=(
            output_feature_order_sha256
            or hashlib.sha256(
                json.dumps(["GENE_A", "GENE_B"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        ),
        evidence_kind=evidence_kind,
    )


def test_external_manifest_fingerprint_binds_mapping_and_synthetic_is_nonclaim() -> None:
    real = _manifest("tcga")
    changed = _manifest("tcga", feature_namespace_version="v2")
    synthetic = _manifest("synthetic", evidence_kind="synthetic_contract_only")

    assert real.claim_eligible is True
    assert real.fingerprint != changed.fingerprint
    assert synthetic.claim_eligible is False
    assert "participant" not in " ".join(real.to_record()).lower()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cohort_id": "patient_123"},
        {"study_id": "study with whitespace"},
    ],
)
def test_external_manifest_rejects_noncohort_safe_tokens(kwargs: dict[str, str]) -> None:
    values = {"cohort_id": "tcga", "study_id": "tcga"}
    values.update(kwargs)
    with pytest.raises(ExternalCohortContractError):
        _manifest(**values)


def test_external_family_requires_compatible_namespaces_and_unique_holdout_values() -> None:
    family = ExternalCohortFamily((_manifest("tcga"), _manifest("metabric")))
    assert family.claim_eligible is True
    assert family.holdout_axis == "study_id"

    with pytest.raises(ExternalCohortContractError, match="feature namespace"):
        ExternalCohortFamily((_manifest("tcga"), _manifest("metabric", feature_namespace_version="v2")))
    with pytest.raises(ExternalCohortContractError, match="unique"):
        ExternalCohortFamily((_manifest("tcga"), _manifest("tcga")))
    with pytest.raises(ExternalCohortContractError, match="output feature order"):
        ExternalCohortFamily((_manifest("tcga"), _manifest("metabric", output_feature_order_sha256="e" * 64)))


def test_leave_one_source_out_enforces_every_source_boundary_and_parent_identity() -> None:
    y = [0, 1, 0, 1, 0, 1]
    source_ids = ["study_a", "study_a", "study_b", "study_b", "study_c", "study_c"]
    context = FitResamplingContext(n_rows=len(y), row_ids=tuple(f"row_{i}" for i in range(len(y))))

    plan = resolve_leave_one_source_out(
        context, y, source_ids=source_ids, axis="study_id"
    )

    assert len(plan.splits) == 3
    assert plan.policy.kind == "group"
    assert plan.context_fingerprint != context.fingerprint
    for split in plan.splits:
        train_sources = {source_ids[index] for index in split.train_indices}
        test_sources = {source_ids[index] for index in split.test_indices}
        assert train_sources.isdisjoint(test_sources)
        assert len(test_sources) == 1
        assert split.audit.ok is True
        metadata = dict(split.assignment.metadata)
        assert metadata["parent_context_fingerprint"] == context.fingerprint
        assert len(metadata["source_membership_sha256"]) == 64


def test_leave_one_source_out_fails_closed_for_insufficient_unseen_or_mutated_sources() -> None:
    y = [0, 1, 0, 1]
    context = FitResamplingContext(n_rows=len(y))
    with pytest.raises(ResamplingContractError, match="at least two"):
        resolve_leave_one_source_out(context, y, source_ids=["a"] * 4, axis="study_id")
    with pytest.raises(ResamplingContractError) as error:
        resolve_leave_one_source_out(
            context, [0, 0, 1, 1], source_ids=["a", "a", "b", "b"], axis="study_id"
        )
    assert error.value.code == "external_unseen_test_label"

    sources = ["a", "a", "b", "b"]
    first = resolve_leave_one_source_out(context, y, source_ids=sources, axis="study_id")
    permuted = copy.copy(sources)
    permuted[0], permuted[2] = permuted[2], permuted[0]
    second = resolve_leave_one_source_out(context, y, source_ids=permuted, axis="study_id")
    assert first.fingerprint != second.fingerprint


def test_leave_one_source_out_rejects_uncomposed_structured_context() -> None:
    y = [0, 1, 0, 1]
    context = FitResamplingContext(
        n_rows=len(y),
        groups=("patient_a", "patient_a", "patient_b", "patient_b"),
        policy=ResamplingPolicy(kind="group", enforced_boundaries=("groups",)),
    )
    with pytest.raises(ResamplingContractError) as error:
        resolve_leave_one_source_out(
            context, y, source_ids=["study_a", "study_b", "study_a", "study_b"], axis="study_id"
        )
    assert error.value.code == "external_source_context_composition_unsupported"


def _feature_mapping() -> DeclaredFeatureMapping:
    return DeclaredFeatureMapping(
        mapping_id="probe_to_hgnc_v1",
        source_namespace_id="probe",
        target_namespace_id="hgnc",
        mapping_artifact_sha256="c" * 64,
        mapping_code_sha256="d" * 64,
        output_feature_order=("GENE_A", "GENE_B"),
        source_to_target=(("probe_1", "GENE_A"), ("probe_2", "GENE_A"), ("probe_3", "GENE_B")),
    )


def _label_mapping() -> DeclaredLabelMapping:
    return DeclaredLabelMapping(
        namespace_id="pam50_intrinsic",
        namespace_sha256="b" * 64,
        labels=(("0", "luminal_a"), ("1", "basal")),
    )


def test_declared_mapping_is_deterministic_and_many_to_one_without_pooling() -> None:
    manifest = _manifest("tcga")
    frame = pd.DataFrame(
        {"probe_3": [3.0, 9.0], "probe_1": [1.0, 5.0], "probe_2": [2.0, 7.0]}
    )

    mapped = map_external_cohort(
        frame, ["0", "1"], manifest=manifest,
        feature_mapping=_feature_mapping(), label_mapping=_label_mapping(),
    )

    assert mapped.feature_order == ("GENE_A", "GENE_B")
    assert mapped.X.tolist() == [[1.5, 3.0], [6.0, 9.0]]
    assert mapped.y.tolist() == ["luminal_a", "basal"]
    assert set(mapped.provenance) == {
        "external_manifest_fingerprint", "external_feature_mapping_fingerprint",
        "external_label_mapping_sha256", "external_feature_order_sha256",
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.rename(columns={"probe_1": "unknown"}), "features"),
        (lambda frame: frame.assign(probe_4=[1.0, 2.0]), "features"),
    ],
)
def test_declared_mapping_fails_closed_on_feature_drift(
    mutator, message: str
) -> None:
    frame = pd.DataFrame({"probe_1": [1.0, 5.0], "probe_2": [2.0, 7.0], "probe_3": [3.0, 9.0]})
    with pytest.raises(ExternalMappingError, match=message):
        map_external_cohort(
            mutator(frame), ["0", "1"], manifest=_manifest("tcga"),
            feature_mapping=_feature_mapping(), label_mapping=_label_mapping(),
        )


def test_declared_mapping_rejects_namespace_and_label_drift() -> None:
    frame = pd.DataFrame({"probe_1": [1.0, 5.0], "probe_2": [2.0, 7.0], "probe_3": [3.0, 9.0]})
    wrong_manifest = _manifest("tcga")
    wrong_mapping = DeclaredFeatureMapping(
        mapping_id="probe_to_wrong", source_namespace_id="probe", target_namespace_id="wrong",
        mapping_artifact_sha256="c" * 64, mapping_code_sha256="d" * 64,
        output_feature_order=("GENE_A",), source_to_target=(("probe_1", "GENE_A"), ("probe_2", "GENE_A"), ("probe_3", "GENE_A")),
    )
    with pytest.raises(ExternalMappingError, match="target namespace"):
        map_external_cohort(frame, ["0", "1"], manifest=wrong_manifest, feature_mapping=wrong_mapping, label_mapping=_label_mapping())
    with pytest.raises(ExternalMappingError, match="labels"):
        map_external_cohort(frame, ["0", "unknown"], manifest=wrong_manifest, feature_mapping=_feature_mapping(), label_mapping=_label_mapping())


def test_declared_mapping_rejects_output_order_drift() -> None:
    frame = pd.DataFrame({"probe_1": [1.0], "probe_2": [2.0], "probe_3": [3.0]})
    with pytest.raises(ExternalMappingError, match="output order"):
        map_external_cohort(
            frame, ["0"], manifest=_manifest("tcga", output_feature_order_sha256="e" * 64),
            feature_mapping=_feature_mapping(), label_mapping=_label_mapping(),
        )


def test_external_source_metrics_are_sourcewise_and_synthetic_rows_cannot_claim() -> None:
    result = evaluate_external_source(
        manifest=_manifest("tcga"),
        holdout_axis="study_id",
        held_out_source="tcga",
        y_true=["luminal_a", "basal", "luminal_a", "basal"],
        y_pred=["luminal_a", "basal", "basal", "basal"],
        proba=np.asarray([[0.9, 0.1], [0.1, 0.9], [0.4, 0.6], [0.2, 0.8]]),
        classes=["luminal_a", "basal"],
        n_train_sources=2,
        n_train_rows=20,
        bootstrap_rounds=20,
        bootstrap_seed=17,
        provenance={"outer_split_fingerprint": "f" * 64},
    )
    synthetic = evaluate_external_source(
        manifest=_manifest("synthetic", evidence_kind="synthetic_contract_only"),
        holdout_axis="study_id",
        held_out_source="synthetic",
        y_true=["luminal_a", "basal"], y_pred=["luminal_a", "basal"],
        n_train_sources=1, n_train_rows=4,
    )
    summary = summarize_external_sources([result, synthetic], expected_sources=["tcga", "synthetic"])

    assert result.claim_eligible is True
    assert result.balanced_accuracy_ci is not None
    assert np.isfinite(result.log_loss)
    assert np.isfinite(result.brier)
    assert np.isfinite(result.ece)
    assert synthetic.claim_eligible is False
    assert summary["pooled_test_metrics_used"] is False
    assert summary["claim_eligible"] is False
    assert summary["balanced_accuracy_worst_source"] == pytest.approx(
        min(result.balanced_accuracy, synthetic.balanced_accuracy)
    )


@pytest.mark.parametrize(
    "rows, expected_sources, expected_complete, message",
    [
        (
            [
                evaluate_external_source(
                    manifest=_manifest("tcga"), holdout_axis="study_id", held_out_source="tcga",
                    y_true=["luminal_a", "basal"], y_pred=["luminal_a", "basal"],
                    n_train_sources=1, n_train_rows=4,
                ),
                skipped_external_source(
                    manifest=_manifest("metabric"), holdout_axis="study_id", held_out_source="metabric",
                    reason="feature_namespace_incompatible",
                ),
            ],
            ["tcga", "metabric"],
            True,
            "skipped",
        ),
        (
            [
                evaluate_external_source(
                    manifest=_manifest("tcga"), holdout_axis="study_id", held_out_source="tcga",
                    y_true=["luminal_a", "basal"], y_pred=["luminal_a", "basal"],
                    n_train_sources=1, n_train_rows=4,
                ),
                evaluate_external_source(
                    manifest=_manifest("tcga"), holdout_axis="study_id", held_out_source="tcga",
                    y_true=["luminal_a", "basal"], y_pred=["luminal_a", "basal"],
                    n_train_sources=1, n_train_rows=4,
                ),
            ],
            ["tcga", "metabric"],
            False,
            "duplicate",
        ),
    ],
)
def test_external_summary_cannot_claim_partial_or_duplicate_source_rows(rows, expected_sources, expected_complete: bool, message: str) -> None:
    summary = summarize_external_sources(rows, expected_sources=expected_sources)
    assert summary["claim_eligible"] is False, message
    assert summary["complete_expected_sources"] is expected_complete, message


def test_external_summary_rejects_mixed_axis_and_requires_explicit_complete_sources() -> None:
    study = evaluate_external_source(
        manifest=_manifest("tcga"), holdout_axis="study_id", held_out_source="tcga",
        y_true=["luminal_a", "basal"], y_pred=["luminal_a", "basal"], n_train_sources=1, n_train_rows=4,
    )
    site = evaluate_external_source(
        manifest=_manifest("metabric"), holdout_axis="site_id", held_out_source="metabric",
        y_true=["luminal_a", "basal"], y_pred=["luminal_a", "basal"], n_train_sources=1, n_train_rows=4,
    )
    with pytest.raises(ExternalCohortContractError, match="axes"):
        summarize_external_sources([study, site], expected_sources=["tcga", "metabric"])
    with pytest.raises(ExternalCohortContractError, match="expected_sources"):
        summarize_external_sources([study], expected_sources=[])


def test_external_result_provenance_is_hash_only() -> None:
    with pytest.raises(ExternalCohortContractError, match="unsupported"):
        evaluate_external_source(
            manifest=_manifest("tcga"), holdout_axis="study_id", held_out_source="tcga",
            y_true=["luminal_a", "basal"], y_pred=["luminal_a", "basal"],
            n_train_sources=1, n_train_rows=4, provenance={"pred": "f" * 64},
        )
    with pytest.raises(ExternalCohortContractError, match="SHA-256"):
        evaluate_external_source(
            manifest=_manifest("tcga"), holdout_axis="study_id", held_out_source="tcga",
            y_true=["luminal_a", "basal"], y_pred=["luminal_a", "basal"],
            n_train_sources=1, n_train_rows=4, provenance={"outer_split_fingerprint": "UPPER"},
        )


def test_external_probability_metrics_and_skips_are_truthful_when_unavailable() -> None:
    result = evaluate_external_source(
        manifest=_manifest("tcga"), holdout_axis="study_id", held_out_source="tcga",
        y_true=["luminal_a", "basal"], y_pred=["luminal_a", "basal"],
        n_train_sources=1, n_train_rows=4,
        proba=np.asarray([[0.5, 0.5], [0.5, 0.5]]), classes=["luminal_a", "other"],
    )
    skipped = skipped_external_source(
        manifest=_manifest("metabric"), holdout_axis="study_id",
        held_out_source="metabric", reason="feature_namespace_incompatible",
    )

    assert np.isnan(result.log_loss)
    assert skipped.status == "skipped"
    assert skipped.claim_eligible is False
    assert skipped.to_record()["skip_reason"] == "feature_namespace_incompatible"

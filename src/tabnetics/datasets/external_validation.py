"""Immutable contracts for evidence-bearing external cohort validation.

These records intentionally describe only public cohort-level provenance. They
must never contain participant identifiers or a data-derived feature mapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

try:
    import pandas as pd
except Exception:  # pragma: no cover - explicit installation boundary
    pd = None  # type: ignore[assignment]


EXTERNAL_COHORT_MANIFEST_SCHEMA_VERSION = "tabnetics_external_cohort_manifest_v1"
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROVENANCE_SHA256_FIELDS = frozenset({
    "external_manifest_fingerprint",
    "external_feature_mapping_fingerprint",
    "external_label_mapping_sha256",
    "external_feature_order_sha256",
    "outer_split_fingerprint",
    "source_membership_sha256",
    "input_data_identity_sha256",
    "source_schema_sha256",
})

__tabnetics_execution_isolated_state__ = {
    "_SAFE_TOKEN": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": ("re",),
    },
    "_HEX_SHA256": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": ("re",),
    },
}


class ExternalCohortContractError(ValueError):
    """Raised when a cohort cannot support the external-evidence contract."""


class ExternalEvidenceKind(str, Enum):
    REAL_PUBLIC = "real_public"
    SYNTHETIC_CONTRACT_ONLY = "synthetic_contract_only"


class ExternalMappingError(ExternalCohortContractError):
    """Raised when a declared feature or label mapping cannot be honored."""


def _token(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not _SAFE_TOKEN.fullmatch(text) or text.lower().startswith(("patient", "participant")):
        raise ExternalCohortContractError(
            f"{field} must be a non-empty safe cohort-level token."
        )
    return text


def _sha256(value: object, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if not _HEX_SHA256.fullmatch(digest):
        raise ExternalCohortContractError(f"{field} must be a lowercase SHA-256.")
    return digest


def _canonical_sha256(record: object) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _provenance(value: Mapping[str, str] | None) -> dict[str, str]:
    """Accept only fixed hash-only provenance, never row-level artifacts."""

    record = dict(value or {})
    unexpected = set(record) - _PROVENANCE_SHA256_FIELDS
    if unexpected:
        raise ExternalCohortContractError(
            "External provenance contains unsupported or non-aggregate keys."
        )
    return {
        str(key): _sha256(digest, field=f"provenance.{key}")
        for key, digest in record.items()
    }


@dataclass(frozen=True, slots=True)
class ExternalCohortManifest:
    """Public, versioned provenance for one independently held-out cohort."""

    family_id: str
    cohort_id: str
    study_id: str
    site_id: str
    assay: str
    platform: str
    preprocessing_id: str
    public_source_id: str
    public_source_revision: str
    data_artifact_sha256: str
    label_namespace_id: str
    label_namespace_sha256: str
    feature_namespace_id: str
    feature_namespace_version: str
    feature_mapping_source: str
    feature_mapping_sha256: str
    mapping_code_sha256: str
    output_feature_order_sha256: str
    evidence_kind: ExternalEvidenceKind | str = ExternalEvidenceKind.REAL_PUBLIC
    schema_version: str = EXTERNAL_COHORT_MANIFEST_SCHEMA_VERSION
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if str(self.schema_version) != EXTERNAL_COHORT_MANIFEST_SCHEMA_VERSION:
            raise ExternalCohortContractError(
                "Unsupported external cohort manifest schema version."
            )
        for name in (
            "family_id", "cohort_id", "study_id", "site_id", "assay", "platform",
            "preprocessing_id", "public_source_id", "public_source_revision",
            "label_namespace_id", "feature_namespace_id",
            "feature_namespace_version", "feature_mapping_source",
        ):
            object.__setattr__(self, name, _token(getattr(self, name), field=name))
        try:
            kind = self.evidence_kind if isinstance(self.evidence_kind, ExternalEvidenceKind) else ExternalEvidenceKind(str(self.evidence_kind))
        except ValueError as exc:
            raise ExternalCohortContractError(
                f"Unknown external evidence kind: {self.evidence_kind!r}."
            ) from exc
        object.__setattr__(self, "evidence_kind", kind)
        for name in (
            "data_artifact_sha256", "label_namespace_sha256", "feature_mapping_sha256",
            "mapping_code_sha256", "output_feature_order_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), field=name))
        object.__setattr__(self, "_fingerprint", _canonical_sha256(self.to_record()))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def claim_eligible(self) -> bool:
        return self.evidence_kind is ExternalEvidenceKind.REAL_PUBLIC

    def to_record(self) -> dict[str, str]:
        return {
            "schema_version": EXTERNAL_COHORT_MANIFEST_SCHEMA_VERSION,
            "family_id": self.family_id, "cohort_id": self.cohort_id,
            "study_id": self.study_id, "site_id": self.site_id, "assay": self.assay,
            "platform": self.platform, "preprocessing_id": self.preprocessing_id,
            "public_source_id": self.public_source_id,
            "public_source_revision": self.public_source_revision,
            "data_artifact_sha256": self.data_artifact_sha256,
            "label_namespace_id": self.label_namespace_id,
            "label_namespace_sha256": self.label_namespace_sha256,
            "feature_namespace_id": self.feature_namespace_id,
            "feature_namespace_version": self.feature_namespace_version,
            "feature_mapping_source": self.feature_mapping_source,
            "feature_mapping_sha256": self.feature_mapping_sha256,
            "mapping_code_sha256": self.mapping_code_sha256,
            "output_feature_order_sha256": self.output_feature_order_sha256,
            "evidence_kind": self.evidence_kind.value,
        }


@dataclass(frozen=True, slots=True)
class ExternalCohortFamily:
    """A compatible cohort family eligible for leave-one-source-out planning."""

    manifests: tuple[ExternalCohortManifest, ...]
    holdout_axis: str = "study_id"
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        manifests = tuple(self.manifests or tuple())
        if len(manifests) < 2 or not all(isinstance(item, ExternalCohortManifest) for item in manifests):
            raise ExternalCohortContractError("An external cohort family requires at least two manifests.")
        if len({item.cohort_id for item in manifests}) != len(manifests):
            raise ExternalCohortContractError("External cohort ids must be unique.")
        if len({item.family_id for item in manifests}) != 1:
            raise ExternalCohortContractError("External manifests must share one family_id.")
        axis = _token(self.holdout_axis, field="holdout_axis")
        if axis not in {"study_id", "cohort_id", "site_id"}:
            raise ExternalCohortContractError("holdout_axis must be study_id, cohort_id, or site_id.")
        if len({getattr(item, axis) for item in manifests}) < 2:
            raise ExternalCohortContractError("External cohort family needs at least two holdout-axis values.")
        if len({(item.label_namespace_id, item.label_namespace_sha256) for item in manifests}) != 1:
            raise ExternalCohortContractError("External cohorts require one declared label namespace.")
        if len({(item.feature_namespace_id, item.feature_namespace_version) for item in manifests}) != 1:
            raise ExternalCohortContractError("External cohorts require one declared feature namespace.")
        if len({item.output_feature_order_sha256 for item in manifests}) != 1:
            raise ExternalCohortContractError("External cohorts require one declared output feature order.")
        object.__setattr__(self, "manifests", tuple(sorted(manifests, key=lambda item: item.cohort_id)))
        object.__setattr__(self, "holdout_axis", axis)
        object.__setattr__(self, "_fingerprint", _canonical_sha256(self.to_record()))

    @property
    def claim_eligible(self) -> bool:
        return all(item.claim_eligible for item in self.manifests)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": EXTERNAL_COHORT_MANIFEST_SCHEMA_VERSION,
            "holdout_axis": self.holdout_axis,
            "manifests": [item.to_record() for item in self.manifests],
        }


@dataclass(frozen=True, slots=True)
class DeclaredFeatureMapping:
    """Versioned source-to-canonical mapping fixed before cohort values are read.

    A target may receive more than one source feature only through the declared
    arithmetic mean.  This handles a documented probe-to-gene aggregation while
    prohibiting data- or label-derived feature matching.
    """

    mapping_id: str
    source_namespace_id: str
    target_namespace_id: str
    mapping_artifact_sha256: str
    mapping_code_sha256: str
    output_feature_order: tuple[str, ...]
    source_to_target: tuple[tuple[str, str], ...]
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("mapping_id", "source_namespace_id", "target_namespace_id"):
            object.__setattr__(self, name, _token(getattr(self, name), field=name))
        object.__setattr__(self, "mapping_artifact_sha256", _sha256(self.mapping_artifact_sha256, field="mapping_artifact_sha256"))
        object.__setattr__(self, "mapping_code_sha256", _sha256(self.mapping_code_sha256, field="mapping_code_sha256"))
        order = tuple(_token(value, field="output_feature_order") for value in self.output_feature_order)
        if not order or len(set(order)) != len(order):
            raise ExternalMappingError("output_feature_order must be non-empty and unique.")
        pairs = tuple(
            (_token(source, field="source feature"), _token(target, field="target feature"))
            for source, target in self.source_to_target
        )
        if not pairs or len({source for source, _ in pairs}) != len(pairs):
            raise ExternalMappingError("Every declared source feature must map exactly once.")
        if {target for _, target in pairs} != set(order):
            raise ExternalMappingError("Mapping targets must exactly match output_feature_order.")
        object.__setattr__(self, "output_feature_order", order)
        object.__setattr__(self, "source_to_target", tuple(sorted(pairs)))
        object.__setattr__(self, "_fingerprint", _canonical_sha256(self.to_record()))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def to_record(self) -> dict[str, object]:
        return {
            "mapping_id": self.mapping_id,
            "source_namespace_id": self.source_namespace_id,
            "target_namespace_id": self.target_namespace_id,
            "mapping_artifact_sha256": self.mapping_artifact_sha256,
            "mapping_code_sha256": self.mapping_code_sha256,
            "output_feature_order": list(self.output_feature_order),
            "source_to_target": [list(pair) for pair in self.source_to_target],
            "aggregation": "mean",
        }


@dataclass(frozen=True, slots=True)
class DeclaredLabelMapping:
    """A fixed cohort-label to canonical-label mapping."""

    namespace_id: str
    namespace_sha256: str
    labels: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace_id", _token(self.namespace_id, field="namespace_id"))
        object.__setattr__(self, "namespace_sha256", _sha256(self.namespace_sha256, field="namespace_sha256"))
        labels = tuple(
            (_token(source, field="source label"), _token(target, field="target label"))
            for source, target in self.labels
        )
        if not labels or len({source for source, _ in labels}) != len(labels):
            raise ExternalMappingError("Every declared source label must map exactly once.")
        object.__setattr__(self, "labels", tuple(sorted(labels)))

    def to_record(self) -> dict[str, object]:
        return {
            "namespace_id": self.namespace_id,
            "namespace_sha256": self.namespace_sha256,
            "labels": [list(pair) for pair in self.labels],
        }


@dataclass(frozen=True, slots=True)
class MappedExternalCohort:
    """Mapped cohort arrays and only the provenance needed by a runner."""

    manifest: ExternalCohortManifest
    X: np.ndarray
    y: np.ndarray
    feature_order: tuple[str, ...]
    provenance: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ExternalSourceResult:
    """Aggregate-only outcome for one held-out external source."""

    family_id: str
    holdout_axis: str
    held_out_source: str
    status: str
    claim_eligible: bool
    n_train_sources: int
    n_test_sources: int
    n_train_rows: int
    n_test_rows: int
    balanced_accuracy: float = float("nan")
    macro_f1: float = float("nan")
    accuracy: float = float("nan")
    log_loss: float = float("nan")
    brier: float = float("nan")
    ece: float = float("nan")
    balanced_accuracy_ci: tuple[float, float] | None = None
    skip_reason: str = ""
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _token(self.family_id, field="family_id"))
        axis = _token(self.holdout_axis, field="holdout_axis")
        if axis not in {"study_id", "cohort_id", "site_id"}:
            raise ExternalCohortContractError("holdout_axis must be study_id, cohort_id, or site_id.")
        object.__setattr__(self, "holdout_axis", axis)
        object.__setattr__(self, "held_out_source", _token(self.held_out_source, field="held_out_source"))
        if self.status not in {"ran", "skipped"}:
            raise ExternalCohortContractError("External source status must be ran or skipped.")
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "family_id": self.family_id,
            "holdout_axis": self.holdout_axis,
            "held_out_source": self.held_out_source,
            "status": self.status,
            "claim_eligible": bool(self.claim_eligible),
            "n_train_sources": int(self.n_train_sources),
            "n_test_sources": int(self.n_test_sources),
            "n_train_rows": int(self.n_train_rows),
            "n_test_rows": int(self.n_test_rows),
            "balanced_accuracy": float(self.balanced_accuracy),
            "macro_f1": float(self.macro_f1),
            "accuracy": float(self.accuracy),
            "log_loss": float(self.log_loss),
            "brier": float(self.brier),
            "ece": float(self.ece),
            "skip_reason": self.skip_reason,
            "provenance": dict(self.provenance),
        }
        if self.balanced_accuracy_ci is not None:
            record["balanced_accuracy_ci_low"] = float(self.balanced_accuracy_ci[0])
            record["balanced_accuracy_ci_high"] = float(self.balanced_accuracy_ci[1])
        return record


def _top_label_ece(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    predicted = classes[np.argmax(proba, axis=1)]
    confidence = np.max(proba, axis=1)
    correct = predicted == y_true
    total = float(y_true.size)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & ((confidence < upper) if upper < 1.0 else (confidence <= upper))
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return float(ece) if total else float("nan")


def _probability_metrics(
    y_true: np.ndarray,
    proba: np.ndarray | None,
    classes: Sequence[Any] | None,
) -> tuple[float, float, float]:
    if proba is None or classes is None:
        return float("nan"), float("nan"), float("nan")
    values = np.asarray(proba, dtype=float)
    labels = np.asarray(classes, dtype=object).ravel()
    if values.ndim != 2 or values.shape[0] != y_true.size or values.shape[1] != labels.size:
        return float("nan"), float("nan"), float("nan")
    if labels.size < 2 or len(set(labels.tolist())) != labels.size or not np.isfinite(values).all():
        return float("nan"), float("nan"), float("nan")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-7, atol=1e-7):
        return float("nan"), float("nan"), float("nan")
    if not set(y_true.tolist()).issubset(set(labels.tolist())):
        return float("nan"), float("nan"), float("nan")
    # sklearn's log-loss API sorts class labels. Reorder the declared columns
    # explicitly so a valid non-lexicographic model class order stays silent.
    ordered_labels = sorted(labels.tolist(), key=str)
    ordered_columns = [labels.tolist().index(label) for label in ordered_labels]
    loss = float(log_loss(y_true, values[:, ordered_columns], labels=ordered_labels))
    indicator = np.zeros_like(values)
    class_index = {value: index for index, value in enumerate(labels.tolist())}
    indicator[np.arange(y_true.size), [class_index[value] for value in y_true.tolist()]] = 1.0
    brier = float(np.mean(np.sum((values - indicator) ** 2, axis=1)))
    return loss, brier, _top_label_ece(y_true, values, labels)


def evaluate_external_source(
    *,
    manifest: ExternalCohortManifest,
    holdout_axis: str,
    held_out_source: str,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    n_train_sources: int,
    n_train_rows: int,
    proba: np.ndarray | None = None,
    classes: Sequence[Any] | None = None,
    bootstrap_rounds: int = 0,
    bootstrap_seed: int = 0,
    provenance: Mapping[str, str] | None = None,
) -> ExternalSourceResult:
    """Return aggregate source metrics without retaining predictions or labels."""

    axis = _token(holdout_axis, field="holdout_axis")
    held_out = _token(held_out_source, field="held_out_source")
    truth = np.asarray(y_true, dtype=object).ravel()
    predicted = np.asarray(y_pred, dtype=object).ravel()
    if truth.size == 0 or truth.size != predicted.size:
        raise ExternalCohortContractError("External source predictions must align to a non-empty test label vector.")
    if n_train_sources < 1 or n_train_rows < 1:
        raise ExternalCohortContractError("External source evaluation requires non-empty training sources and rows.")
    ba = float(balanced_accuracy_score(truth, predicted))
    macro_f1 = float(f1_score(truth, predicted, average="macro", zero_division=0))
    accuracy = float(accuracy_score(truth, predicted))
    loss, brier, ece = _probability_metrics(truth, proba, classes)
    ci: tuple[float, float] | None = None
    rounds = int(bootstrap_rounds)
    if rounds > 0:
        rng = np.random.default_rng(int(bootstrap_seed))
        values: list[float] = []
        full_labels = set(truth.tolist())
        for _ in range(max(rounds * 20, rounds)):
            if len(values) >= rounds:
                break
            sampled = rng.integers(0, truth.size, size=truth.size)
            if set(truth[sampled].tolist()) != full_labels:
                continue
            values.append(float(balanced_accuracy_score(truth[sampled], predicted[sampled])))
        if len(values) == rounds:
            draws = np.asarray(values, dtype=float)
            ci = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
    return ExternalSourceResult(
        family_id=manifest.family_id,
        holdout_axis=axis,
        held_out_source=held_out,
        status="ran",
        claim_eligible=manifest.claim_eligible,
        n_train_sources=int(n_train_sources),
        n_test_sources=1,
        n_train_rows=int(n_train_rows),
        n_test_rows=int(truth.size),
        balanced_accuracy=ba,
        macro_f1=macro_f1,
        accuracy=accuracy,
        log_loss=loss,
        brier=brier,
        ece=ece,
        balanced_accuracy_ci=ci,
        provenance=_provenance(provenance),
    )


def skipped_external_source(
    *,
    manifest: ExternalCohortManifest,
    holdout_axis: str,
    held_out_source: str,
    reason: str,
    provenance: Mapping[str, str] | None = None,
) -> ExternalSourceResult:
    """Emit a truthful non-claim-bearing incompatibility/viability row."""

    return ExternalSourceResult(
        family_id=manifest.family_id,
        holdout_axis=_token(holdout_axis, field="holdout_axis"),
        held_out_source=_token(held_out_source, field="held_out_source"),
        status="skipped",
        claim_eligible=False,
        n_train_sources=0,
        n_test_sources=1,
        n_train_rows=0,
        n_test_rows=0,
        skip_reason=_token(reason, field="skip_reason"),
        provenance=_provenance(provenance),
    )


def summarize_external_sources(
    rows: Sequence[ExternalSourceResult],
    *,
    expected_sources: Sequence[str],
) -> dict[str, object]:
    """Summarize by source, never by pooled test observations."""

    results = tuple(rows)
    if not results:
        raise ExternalCohortContractError("External source summary requires at least one row.")
    families = {row.family_id for row in results}
    if len(families) != 1:
        raise ExternalCohortContractError("External source summaries cannot mix cohort families.")
    axes = {row.holdout_axis for row in results}
    if len(axes) != 1:
        raise ExternalCohortContractError("External source summaries cannot mix holdout axes.")
    expected = tuple(_token(source, field="expected_source") for source in expected_sources)
    if not expected or len(set(expected)) != len(expected):
        raise ExternalCohortContractError("expected_sources must be non-empty and unique.")
    observed = tuple(row.held_out_source for row in results)
    complete = len(observed) == len(expected) and set(observed) == set(expected) and len(set(observed)) == len(observed)
    ran = [row for row in results if row.status == "ran"]
    ba = np.asarray([row.balanced_accuracy for row in ran], dtype=float)
    finite = ba[np.isfinite(ba)]
    finite_claim_metrics = all(
        np.isfinite(row.balanced_accuracy) and np.isfinite(row.macro_f1)
        for row in results
    )
    eligible = (
        complete
        and all(row.status == "ran" and row.claim_eligible for row in results)
        and finite_claim_metrics
    )
    return {
        "family_id": next(iter(families)),
        "source_rows": len(results),
        "valid_source_rows": len(ran),
        "skipped_source_rows": len(results) - len(ran),
        "holdout_axis": next(iter(axes)),
        "expected_source_rows": len(expected),
        "complete_expected_sources": complete,
        "claim_eligible": eligible,
        "balanced_accuracy_macro_source_mean": float(finite.mean()) if finite.size else float("nan"),
        "balanced_accuracy_worst_source": float(finite.min()) if finite.size else float("nan"),
        "pooled_test_metrics_used": False,
    }


def map_external_cohort(
    X: Any,
    y: Sequence[Any],
    *,
    manifest: ExternalCohortManifest,
    feature_mapping: DeclaredFeatureMapping,
    label_mapping: DeclaredLabelMapping,
) -> MappedExternalCohort:
    """Apply only fully declared mappings and reject every source/schema drift."""

    if pd is None or not isinstance(X, pd.DataFrame):
        raise ExternalMappingError("External cohort mapping requires a named pandas DataFrame.")
    if len(X) != len(y):
        raise ExternalMappingError("External cohort X/y row counts do not match.")
    if feature_mapping.target_namespace_id != manifest.feature_namespace_id:
        raise ExternalMappingError("Feature mapping target namespace does not match cohort manifest.")
    if feature_mapping.mapping_artifact_sha256 != manifest.feature_mapping_sha256:
        raise ExternalMappingError("Feature mapping artifact digest does not match cohort manifest.")
    if feature_mapping.mapping_code_sha256 != manifest.mapping_code_sha256:
        raise ExternalMappingError("Feature mapping code digest does not match cohort manifest.")
    if _canonical_sha256(list(feature_mapping.output_feature_order)) != manifest.output_feature_order_sha256:
        raise ExternalMappingError("Feature mapping output order digest does not match cohort manifest.")
    if label_mapping.namespace_id != manifest.label_namespace_id or label_mapping.namespace_sha256 != manifest.label_namespace_sha256:
        raise ExternalMappingError("Label mapping namespace does not match cohort manifest.")
    source_columns = tuple(str(column) for column in X.columns)
    if len(set(source_columns)) != len(source_columns):
        raise ExternalMappingError("External cohort source feature names must be unique.")
    declared_sources = tuple(source for source, _ in feature_mapping.source_to_target)
    if set(source_columns) != set(declared_sources):
        raise ExternalMappingError("External cohort source features do not exactly match the declared mapping.")
    try:
        numeric = X.loc[:, list(declared_sources)].astype(float)
    except (TypeError, ValueError) as exc:
        raise ExternalMappingError("External cohort mapped features must be finite numeric values.") from exc
    values = numeric.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ExternalMappingError("External cohort mapped features contain non-finite values.")
    by_target: dict[str, list[str]] = {target: [] for target in feature_mapping.output_feature_order}
    for source, target in feature_mapping.source_to_target:
        by_target[target].append(source)
    mapped = np.column_stack(
        [numeric.loc[:, by_target[target]].mean(axis=1).to_numpy(dtype=float) for target in feature_mapping.output_feature_order]
    )
    label_lookup = dict(label_mapping.labels)
    source_labels = [str(value) for value in y]
    unknown = sorted({value for value in source_labels if value not in label_lookup})
    if unknown:
        raise ExternalMappingError("External cohort contains labels absent from the declared mapping.")
    mapped_labels = np.asarray([label_lookup[value] for value in source_labels], dtype=object)
    provenance = {
        "external_manifest_fingerprint": manifest.fingerprint,
        "external_feature_mapping_fingerprint": feature_mapping.fingerprint,
        "external_label_mapping_sha256": _canonical_sha256(label_mapping.to_record()),
        "external_feature_order_sha256": _canonical_sha256(list(feature_mapping.output_feature_order)),
    }
    return MappedExternalCohort(
        manifest=manifest,
        X=np.asarray(mapped, dtype=float),
        y=mapped_labels,
        feature_order=feature_mapping.output_feature_order,
        provenance=provenance,
    )


__all__ = [
    "EXTERNAL_COHORT_MANIFEST_SCHEMA_VERSION", "ExternalCohortContractError",
    "DeclaredFeatureMapping", "DeclaredLabelMapping", "ExternalCohortFamily",
    "ExternalCohortManifest", "ExternalEvidenceKind", "ExternalMappingError",
    "ExternalSourceResult", "MappedExternalCohort", "evaluate_external_source",
    "map_external_cohort", "skipped_external_source", "summarize_external_sources",
]

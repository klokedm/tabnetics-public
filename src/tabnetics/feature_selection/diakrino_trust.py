"""DIAKRINO sidecar artifact trust-record helpers.

The trust record is intentionally small and JSON-native.  It lets sidecar
artifacts carry the calibration/head-trust facts that otherwise live in issue
comments and docs, while preserving legacy behavior when no record is present.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .diakrino_identity import canonical_json_sha256, sha256_file


DIAKRINO_SIDECAR_TRUST_SCHEMA_VERSION = "diakrino_sidecar_trust_v1"
DIAKRINO_CHECKPOINT_QUALIFICATION_SCHEMA_VERSION = "diakrino_checkpoint_qualification_v1"
DIAKRINO_QUALIFICATION_ATTESTATION_SCHEMA_VERSION = "diakrino_qualification_attestation_v2"
DIAKRINO_QUALIFICATION_ATTESTATION_TOOL = (
    "tabnetics.scripts.analysis.diakrino_checkpoint_qualification"
)
DIAKRINO_QUALIFICATION_ATTESTATION_EVALUATORS: dict[str, str] = {
    "qualification_script": "scripts/analysis/diakrino_checkpoint_qualification.py",
    "diakrino_identity_module": "core/src/tabnetics/feature_selection/diakrino_identity.py",
    "diakrino_trust_module": "core/src/tabnetics/feature_selection/diakrino_trust.py",
}
DIAKRINO_QUALIFICATION_ATTESTED_RECORD_FIELDS: tuple[str, ...] = (
    "schema_version",
    "task_id",
    "record_id",
    "dry_run",
    "qualification_scope",
    "checkpoint",
    "overall_pass",
    "qualified_sidecar_identities",
    "sidecar_identity_contract",
    "gates",
    "metric_binding",
    "consumer_classes",
    "generated_metric_sources",
)
DIAKRINO_FEATURE_SELECTION_REQUIRED_GATES: tuple[str, ...] = (
    "checkpoint_geometry_load",
    "head_trust_gradient",
    "sidecar_reemit",
    "selector_weights_candidate_probe",
    "s1_chunk_calibration_replay",
    "cross_chunk_logit_mean_drift",
)
DIAKRINO_SOURCE_EMISSION_QUALIFICATION_SCOPE = "source_emission"
DIAKRINO_SOURCE_EMISSION_BINDING_CONSUMER = "source_emission_binding"
DIAKRINO_SOURCE_EMISSION_REQUIRED_GATES: tuple[str, ...] = (
    "checkpoint_geometry_load",
    "head_trust_gradient",
    "sidecar_reemit",
    "selector_weights_candidate_probe",
)
DIAKRINO_SOURCE_ATTESTATION_IDENTITY_FIELDS = frozenset(
    {
        "dataset_id",
        "binding_sha256",
        "checkpoint_sha256",
        "source_manifest_file_sha256",
        "feature_logits_sha256",
    }
)
DIAKRINO_REQUIRED_CALIBRATION_MODE = "chunk_zscore"
DIAKRINO_DISCRETE_SKIP_MIN_FAMILY_ID = 31

DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT = "current_checkpoint_20260628"
DIAKRINO_SELECTOR_PRIOR_CALIBRATION_NONE = "none"
DIAKRINO_SELECTOR_PRIOR_CURRENT_ANCHOR: dict[str, float] = {
    "mnpo_broad_stable": 0.25,
    "strict_plus_mrmr": 0.45,
    "boruta": 0.075,
    "copula_knockoff": 0.075,
    "stability_lasso": 0.15,
}
DIAKRINO_SELECTOR_PRIOR_CURRENT_RAW_WEIGHT = 0.25
DIAKRINO_SELECTOR_PRIOR_CURRENT_MAX_BLEND = 0.25

DIAKRINO_SELECTOR_PRIOR_EVIDENCE: dict[str, Any] = {
    "pick_frequency_mean_delta_ba": -0.009523,
    "pick_frequency_wilcoxon_p": 0.347484,
    "regime_conditional_mean_delta_ba": 0.0002,
    "structural_verdict": "chance_or_unproven_candidate",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np  # type: ignore

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return value


def _load_json_object(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _sha256_file(path: str | Path | None) -> str:
    if not path:
        return ""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_binding(value: Mapping[str, Any] | Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("binding_sha256") or "").strip().lower()
    return str(getattr(value, "binding_sha256", "") or "").strip().lower()


def _exact_nonnegative_int(value: Any) -> int | None:
    """Return a JSON integer without accepting bool, strings, or floats."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _attestation_identity_tuple(item: Mapping[str, Any] | Any) -> tuple[str, str, str, str, str] | None:
    if not isinstance(item, Mapping) or set(item) != DIAKRINO_SOURCE_ATTESTATION_IDENTITY_FIELDS:
        return None
    dataset_id = str(item.get("dataset_id") or "").strip()
    binding = str(item.get("binding_sha256") or "").strip().lower()
    checkpoint = str(item.get("checkpoint_sha256") or "").strip().lower()
    source_manifest = str(item.get("source_manifest_file_sha256") or "").strip().lower()
    feature_logits = str(item.get("feature_logits_sha256") or "").strip().lower()
    if not dataset_id or not all(
        _is_sha256(value) for value in (binding, checkpoint, source_manifest, feature_logits)
    ):
        return None
    return dataset_id, binding, checkpoint, source_manifest, feature_logits


def _attestation_digest_is_valid(attestation: Mapping[str, Any]) -> bool:
    declared = str(attestation.get("attestation_sha256") or "").strip().lower()
    if not _is_sha256(declared):
        return False
    unsigned = dict(attestation)
    unsigned.pop("attestation_sha256", None)
    return declared == canonical_json_sha256(
        unsigned,
        payload_schema_version=DIAKRINO_QUALIFICATION_ATTESTATION_SCHEMA_VERSION,
    )


def qualification_record_attested_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the qualification facts covered by the v2 attestation digest."""

    return {
        name: record.get(name)
        for name in DIAKRINO_QUALIFICATION_ATTESTED_RECORD_FIELDS
    }


def qualification_record_attested_payload_sha256(record: Mapping[str, Any]) -> str:
    """Hash gate and identity facts independently of mutable record metadata."""

    return canonical_json_sha256(
        qualification_record_attested_payload(record),
        payload_schema_version="diakrino_qualification_attested_record_v2",
    )


def _attestation_evaluator_sources_are_current(
    record: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> bool:
    sources = attestation.get("evaluator_sources")
    provenance = record.get("code_provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, Mapping) else None
    if (
        not isinstance(sources, Mapping)
        or not isinstance(inputs, Mapping)
        or str(provenance.get("schema_version") or "") != "tabnetics_provenance_v1"
    ):
        return False
    if str(attestation.get("record_payload_sha256") or "").strip().lower() != (
        qualification_record_attested_payload_sha256(record)
    ):
        return False
    root = Path(__file__).resolve().parents[4]
    for name, relative_path in DIAKRINO_QUALIFICATION_ATTESTATION_EVALUATORS.items():
        path = root / relative_path
        source = sources.get(name)
        provenance_input = inputs.get(name)
        if not path.is_file() or not isinstance(source, Mapping) or not isinstance(
            provenance_input, Mapping
        ):
            return False
        expected_sha256 = sha256_file(path)
        expected_size = int(path.stat().st_size)
        source_size = _exact_nonnegative_int(source.get("size_bytes"))
        provenance_size = _exact_nonnegative_int(provenance_input.get("size_bytes"))
        if (
            str(source.get("relative_path") or "") != relative_path
            or str(source.get("sha256") or "").strip().lower() != expected_sha256
            or source_size != expected_size
            or str(provenance_input.get("sha256") or "").strip().lower()
            != expected_sha256
            or provenance_size != expected_size
            or provenance_input.get("exists") is not True
        ):
            return False
    return True


def qualification_record_attestation_covers_identity(
    record: Mapping[str, Any],
    identity: Mapping[str, Any] | Any,
    *,
    consumer_class: str,
    source_manifest_sha256: str | None,
) -> bool:
    """Validate the generated-record attestation before trusting qualification gates.

    This is an anti-accidental/generic-JSON boundary: it binds an evaluator version,
    checkpoint, verified source manifest, and replay tuple identities. A hostile
    principal able to replace the evaluator and every local bundle can still forge a
    self-consistent record; that stronger threat model needs detached attestation.
    """

    attestation = record.get("qualification_attestation")
    if (
        not isinstance(attestation, Mapping)
        or str(attestation.get("schema_version") or "")
        != DIAKRINO_QUALIFICATION_ATTESTATION_SCHEMA_VERSION
        or str(attestation.get("tool") or "") != DIAKRINO_QUALIFICATION_ATTESTATION_TOOL
        or not _attestation_digest_is_valid(attestation)
        or not _attestation_evaluator_sources_are_current(record, attestation)
    ):
        return False
    binding = _identity_binding(identity)
    checkpoint = (
        str(identity.get("checkpoint_sha256") or "").strip().lower()
        if isinstance(identity, Mapping)
        else str(getattr(identity, "checkpoint_sha256", "") or "").strip().lower()
    )
    dataset_id = (
        str(identity.get("dataset_id") or "").strip()
        if isinstance(identity, Mapping)
        else str(getattr(identity, "dataset_id", "") or "").strip()
    )
    checkpoint_artifact = attestation.get("checkpoint_artifact")
    if (
        not isinstance(checkpoint_artifact, Mapping)
        or str(checkpoint_artifact.get("sha256") or "").strip().lower() != checkpoint
        or _exact_nonnegative_int(checkpoint_artifact.get("size_bytes")) is None
    ):
        return False
    source_entries = attestation.get("source_manifest_artifacts")
    if not isinstance(source_entries, list) or not source_entries:
        return False
    target_sources: set[tuple[str, str, str, str, str]] = set()
    source_match = False
    for entry in source_entries:
        if not isinstance(entry, Mapping):
            return False
        source_sha = str(entry.get("sha256") or "").strip().lower()
        if not _is_sha256(source_sha) or _exact_nonnegative_int(entry.get("size_bytes")) is None:
            return False
        pairs = entry.get("identity_bindings")
        if not isinstance(pairs, list) or not pairs:
            return False
        for pair in pairs:
            parsed = _attestation_identity_tuple(pair)
            if parsed is None or parsed[3] != source_sha:
                return False
            target_sources.add(parsed)
            if (
                parsed[0] == dataset_id
                and parsed[1] == binding
                and parsed[2] == checkpoint
                and (source_manifest_sha256 is None or parsed[3] == source_manifest_sha256)
            ):
                source_match = True
    if not source_match:
        return False
    if str(consumer_class) == DIAKRINO_SOURCE_EMISSION_BINDING_CONSUMER:
        replays = attestation.get("source_replay_artifacts", [])
    elif str(consumer_class) == "feature_selection":
        replays = attestation.get("canonical_replay_artifacts")
        if not isinstance(replays, list) or not replays:
            return False
    else:
        return True
    if not isinstance(replays, list):
        return False
    for replay in replays:
        if not isinstance(replay, Mapping):
            return False
        if (
            not _is_sha256(replay.get("sha256"))
            or _exact_nonnegative_int(replay.get("size_bytes")) is None
        ):
            return False
        pairs = replay.get("identity_bindings")
        if not isinstance(pairs, list) or not pairs:
            return False
        for pair in pairs:
            parsed = _attestation_identity_tuple(pair)
            if parsed is None or parsed not in target_sources:
                return False
    return True


def _record_has_exact_identity_contract(record: Mapping[str, Any]) -> bool:
    qualified = record.get("qualified_sidecar_identities")
    contract = record.get("sidecar_identity_contract")
    if not isinstance(qualified, list) or not isinstance(contract, Mapping):
        return False
    qualified_count = _exact_nonnegative_int(contract.get("qualified_identity_count"))
    manifest_count = _exact_nonnegative_int(contract.get("manifest_count"))
    return bool(
        contract.get("required_for_claims") is True
        and contract.get("valid") is True
        and qualified_count == len(qualified)
        and qualified_count > 0
        and manifest_count is not None
        and manifest_count > 0
        and not bool(list(contract.get("errors") or []))
    )


def _record_identity_matches(
    record: Mapping[str, Any],
    identity: Mapping[str, Any] | Any,
    *,
    source_manifest_sha256: str | None,
) -> bool:
    binding = _identity_binding(identity)
    if not _is_sha256(binding):
        return False
    checkpoint = record.get("checkpoint")
    checkpoint_sha = (
        str(checkpoint.get("sha256") or "").strip().lower()
        if isinstance(checkpoint, Mapping)
        else ""
    )
    identity_checkpoint = (
        str(identity.get("checkpoint_sha256") or "").strip().lower()
        if isinstance(identity, Mapping)
        else str(getattr(identity, "checkpoint_sha256", "") or "").strip().lower()
    )
    identity_dataset = (
        str(identity.get("dataset_id") or "")
        if isinstance(identity, Mapping)
        else str(getattr(identity, "dataset_id", "") or "")
    )
    source_sha = str(source_manifest_sha256 or "").strip().lower()
    if (
        not identity_dataset
        or not _is_sha256(checkpoint_sha)
        or identity_checkpoint != checkpoint_sha
        or not _is_sha256(source_sha)
    ):
        return False
    for item in list(record.get("qualified_sidecar_identities") or []):
        if not isinstance(item, Mapping):
            continue
        if (
            str(item.get("binding_sha256") or "").strip().lower() == binding
            and str(item.get("checkpoint_sha256") or "").strip().lower() == checkpoint_sha
            and str(item.get("dataset_id") or "") == identity_dataset
            and str(item.get("source_manifest_file_sha256") or "").strip().lower()
            == source_sha
        ):
            return True
    return False


def source_emission_qualification_covers_identity(
    record: Mapping[str, Any] | None,
    identity: Mapping[str, Any] | Any,
    *,
    source_manifest_sha256: str | None,
) -> bool:
    """Return whether an immutable source qualification can authorize binding.

    This is deliberately narrower than feature-selection promotion.  It binds
    source emission bytes and the gates available before a claim/campaign exists;
    canonical replay/closeout evidence belongs to the later campaign authority.
    """

    if not isinstance(record, Mapping):
        return False
    if (
        str(record.get("schema_version") or "")
        != DIAKRINO_CHECKPOINT_QUALIFICATION_SCHEMA_VERSION
        or record.get("dry_run") is not False
        or record.get("qualification_scope")
        != DIAKRINO_SOURCE_EMISSION_QUALIFICATION_SCOPE
        or not _record_has_exact_identity_contract(record)
    ):
        return False
    attestation = record.get("qualification_attestation")
    if (
        not isinstance(attestation, Mapping)
        or "canonical_replay_artifacts" in attestation
    ):
        return False
    classes = record.get("consumer_classes")
    consumer = (
        classes.get(DIAKRINO_SOURCE_EMISSION_BINDING_CONSUMER)
        if isinstance(classes, Mapping)
        else None
    )
    if (
        not isinstance(consumer, Mapping)
        or consumer.get("allowed") is not True
        or tuple(consumer.get("required_gates") or ())
        != DIAKRINO_SOURCE_EMISSION_REQUIRED_GATES
        or list(consumer.get("failed_gates") or []) != []
    ):
        return False
    gate_lookup = {
        str(gate.get("name") or ""): gate
        for gate in list(record.get("gates") or [])
        if isinstance(gate, Mapping) and str(gate.get("name") or "")
    }
    for gate_name in DIAKRINO_SOURCE_EMISSION_REQUIRED_GATES:
        gate = gate_lookup.get(gate_name)
        if (
            not isinstance(gate, Mapping)
            or gate.get("pass") is not True
            or gate.get("required") is not True
            or str(gate.get("status") or "") != "pass"
        ):
            return False
    if not _record_identity_matches(
        record,
        identity,
        source_manifest_sha256=source_manifest_sha256,
    ):
        return False
    return qualification_record_attestation_covers_identity(
        record,
        identity,
        consumer_class=DIAKRINO_SOURCE_EMISSION_BINDING_CONSUMER,
        source_manifest_sha256=source_manifest_sha256,
    )


def qualification_record_covers_identity(
    record: Mapping[str, Any] | None,
    identity: Mapping[str, Any] | Any,
    *,
    consumer_class: str = "feature_selection",
    source_manifest_sha256: str | None = None,
) -> bool:
    """Return whether qualification explicitly covers one exact sidecar binding."""

    if not isinstance(record, Mapping):
        return False
    if str(record.get("schema_version") or "") != DIAKRINO_CHECKPOINT_QUALIFICATION_SCHEMA_VERSION:
        return False
    if record.get("dry_run") is not False:
        return False
    binding = _identity_binding(identity)
    if len(binding) != 64:
        return False
    classes = record.get("consumer_classes")
    if not isinstance(classes, Mapping):
        return False
    consumer = classes.get(str(consumer_class))
    if (
        not isinstance(consumer, Mapping)
        or consumer.get("allowed") is not True
        or bool(list(consumer.get("failed_gates") or []))
        or not bool(list(consumer.get("required_gates") or []))
    ):
        return False
    if str(consumer_class) == "feature_selection":
        required_gates = tuple(str(name) for name in list(consumer.get("required_gates") or []))
        if required_gates != DIAKRINO_FEATURE_SELECTION_REQUIRED_GATES:
            return False
        gate_lookup = {
            str(gate.get("name") or ""): gate
            for gate in list(record.get("gates") or [])
            if isinstance(gate, Mapping) and str(gate.get("name") or "")
        }
        for gate_name in DIAKRINO_FEATURE_SELECTION_REQUIRED_GATES:
            gate = gate_lookup.get(gate_name)
            if (
                not isinstance(gate, Mapping)
                or gate.get("pass") is not True
                or str(gate.get("status") or "") != "pass"
                or gate.get("required") is not True
            ):
                return False
    qualified = record.get("qualified_sidecar_identities")
    if not isinstance(qualified, list):
        return False
    contract = record.get("sidecar_identity_contract")
    if not isinstance(contract, Mapping):
        return False
    qualified_count = _exact_nonnegative_int(contract.get("qualified_identity_count"))
    manifest_count = _exact_nonnegative_int(contract.get("manifest_count"))
    if qualified_count is None or manifest_count is None:
        return False
    if (
        contract.get("required_for_claims") is not True
        or contract.get("valid") is not True
        or qualified_count != len(qualified)
        or qualified_count <= 0
        or manifest_count <= 0
        or bool(list(contract.get("errors") or []))
    ):
        return False
    checkpoint = record.get("checkpoint")
    checkpoint_sha = (
        str(checkpoint.get("sha256") or "").strip().lower()
        if isinstance(checkpoint, Mapping)
        else ""
    )
    identity_checkpoint = (
        str(identity.get("checkpoint_sha256") or "").strip().lower()
        if isinstance(identity, Mapping)
        else str(getattr(identity, "checkpoint_sha256", "") or "").strip().lower()
    )
    identity_dataset = (
        str(identity.get("dataset_id") or "")
        if isinstance(identity, Mapping)
        else str(getattr(identity, "dataset_id", "") or "")
    )
    if len(checkpoint_sha) != 64 or identity_checkpoint != checkpoint_sha:
        return False
    if not qualification_record_attestation_covers_identity(
        record,
        identity,
        consumer_class=str(consumer_class),
        source_manifest_sha256=source_manifest_sha256,
    ):
        return False
    for item in qualified:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("binding_sha256") or "").strip().lower() != binding:
            continue
        if str(item.get("checkpoint_sha256") or "").strip().lower() != checkpoint_sha:
            continue
        if str(item.get("dataset_id") or "") != identity_dataset:
            continue
        if source_manifest_sha256 is not None and str(
            item.get("source_manifest_file_sha256") or ""
        ).strip().lower() != str(source_manifest_sha256).strip().lower():
            continue
        return True
    return False


def default_diakrino_sidecar_trust_record(
    *,
    checkpoint_sha256: str | None = None,
    qualification_record_path: str | Path | None = None,
    qualification_record: Mapping[str, Any] | None = None,
    required_sidecar_identities: Sequence[Mapping[str, Any] | Any] | None = None,
    source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the default sidecar trust/calibration contract for current DIAKRINO artifacts."""

    qual = dict(qualification_record or {})
    if not qual and qualification_record_path:
        qual = dict(_load_json_object(qualification_record_path) or {})
    gates = [
        {
            "name": str(gate.get("name") or ""),
            "status": str(gate.get("status") or ""),
            "pass": bool(gate.get("pass", False)),
            "required": bool(gate.get("required", False)),
        }
        for gate in list(qual.get("gates") or [])
        if isinstance(gate, Mapping)
    ]
    required_bindings = [
        binding
        for binding in (_identity_binding(item) for item in list(required_sidecar_identities or []))
        if binding
    ]
    qualified_bindings = [
        str(item.get("binding_sha256") or "").strip().lower()
        for item in list(qual.get("qualified_sidecar_identities") or [])
        if isinstance(item, Mapping) and str(item.get("binding_sha256") or "").strip()
    ]
    source_qualification_allowed = bool(required_bindings) and all(
        source_emission_qualification_covers_identity(
            qual,
            item,
            source_manifest_sha256=source_manifest_sha256,
        )
        for item in list(required_sidecar_identities or [])
    )
    claim_scope = "evaluation" if source_qualification_allowed else "diagnostic"
    return {
        "schema_version": DIAKRINO_SIDECAR_TRUST_SCHEMA_VERSION,
        "checkpoint_sha256": str(checkpoint_sha256 or ""),
        "qualification_record_path": str(qualification_record_path or ""),
        "qualification_record_sha256": _sha256_file(qualification_record_path),
        "qualification_record_id": str(qual.get("record_id") or ""),
        "qualification_schema_version": str(qual.get("schema_version") or ""),
        "qualification_overall_pass": bool(qual.get("overall_pass", False)) if qual else False,
        "qualification_gates": gates,
        "qualified_identity_bindings": sorted(set(qualified_bindings)),
        "required_identity_bindings": list(required_bindings),
        "source_manifest_sha256": str(source_manifest_sha256 or "").strip().lower(),
        "qualification_scope": str(qual.get("qualification_scope") or ""),
        "claim_scope": claim_scope,
        "exact_identity_coverage": bool(source_qualification_allowed),
        "required_calibration_mode": DIAKRINO_REQUIRED_CALIBRATION_MODE,
        "allow_explicit_ablation_override": True,
        "discrete_family_filter": {
            "enabled": True,
            "skip_family_id_min": DIAKRINO_DISCRETE_SKIP_MIN_FAMILY_ID,
        },
        "heads": {
            "feature_selection": {
                "status": (
                    "evaluation_qualified"
                    if source_qualification_allowed
                    else "missing_exact_source_qualification"
                ),
                "consumer_allowed": bool(source_qualification_allowed),
                "evidence": {
                    "verdict": (
                        "source_emission_qualification_for_evaluation"
                        if source_qualification_allowed
                        else "blocked_without_exact_source_emission_checkpoint_dataset_split_input_feature_binding"
                    ),
                    "required_identity_bindings": list(required_bindings),
                },
            },
            "population_family": {
                "status": "trusted",
                "consumer_allowed": True,
                "evidence": {
                    "verdict": "learned",
                    "source": "S0/S1 DIAKRINO native integration verdict",
                },
            },
            "population_param": {
                "status": "degenerate",
                "consumer_allowed": False,
                "evidence": {"verdict": "blocked_param_warm_start"},
            },
            "support_type": {
                "status": "chance",
                "consumer_allowed": False,
                "evidence": {"verdict": "chance_head_do_not_consume"},
            },
            "task_family": {
                "status": "chance",
                "consumer_allowed": False,
                "evidence": {"verdict": "chance_head_do_not_consume"},
            },
            "legacy_query_heads": {
                "status": "chance",
                "consumer_allowed": False,
                "evidence": {"verdict": "dropped_legacy_query_consumers"},
            },
            "selector_weights": {
                "status": "missing_or_unqualified",
                "consumer_allowed": False,
                "evidence": {"verdict": "blocked_missing_dataset_level_selector_weights"},
            },
            "selector_weights_candidate": {
                "status": "candidate_report_only",
                "consumer_allowed": False,
                "evidence": {
                    "verdict": "predeclared_candidate_requires_nat12_probe_before_promotion",
                },
            },
        },
        "selector_prior_calibration": {
            "mode": DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT,
            "anchor_weights": dict(DIAKRINO_SELECTOR_PRIOR_CURRENT_ANCHOR),
            "raw_weight": float(DIAKRINO_SELECTOR_PRIOR_CURRENT_RAW_WEIGHT),
            "max_blend_weight": float(DIAKRINO_SELECTOR_PRIOR_CURRENT_MAX_BLEND),
            "evidence": dict(DIAKRINO_SELECTOR_PRIOR_EVIDENCE),
        },
    }


def normalize_diakrino_sidecar_trust_record(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(record, Mapping):
        return None
    out = dict(_json_safe(record))
    if str(out.get("schema_version") or "") != DIAKRINO_SIDECAR_TRUST_SCHEMA_VERSION:
        return None
    return out


def trust_record_checkpoint_sha256(record: Mapping[str, Any] | None) -> str:
    rec = normalize_diakrino_sidecar_trust_record(record)
    return "" if rec is None else str(rec.get("checkpoint_sha256") or "")


def trust_record_required_calibration(record: Mapping[str, Any] | None) -> str:
    rec = normalize_diakrino_sidecar_trust_record(record)
    if rec is None:
        return DIAKRINO_REQUIRED_CALIBRATION_MODE
    mode = str(rec.get("required_calibration_mode") or DIAKRINO_REQUIRED_CALIBRATION_MODE).strip().lower()
    return mode or DIAKRINO_REQUIRED_CALIBRATION_MODE


def trust_record_discrete_filter_ok(record: Mapping[str, Any] | None) -> bool:
    rec = normalize_diakrino_sidecar_trust_record(record)
    if rec is None:
        return True
    filt = rec.get("discrete_family_filter")
    if not isinstance(filt, Mapping):
        return False
    min_id = _exact_nonnegative_int(filt.get("skip_family_id_min"))
    return (
        filt.get("enabled") is True
        and min_id == DIAKRINO_DISCRETE_SKIP_MIN_FAMILY_ID
    )


def trust_record_head_allowed(record: Mapping[str, Any] | None, head: str) -> bool:
    rec = normalize_diakrino_sidecar_trust_record(record)
    if rec is None:
        return True
    heads = rec.get("heads")
    if not isinstance(heads, Mapping):
        return False
    item = heads.get(str(head))
    if not isinstance(item, Mapping):
        return False
    return item.get("consumer_allowed") is True


def selector_prior_calibration_from_trust_record(
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    rec = normalize_diakrino_sidecar_trust_record(record)
    if rec is None:
        return {
            "mode": DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT,
            "anchor_weights": dict(DIAKRINO_SELECTOR_PRIOR_CURRENT_ANCHOR),
            "raw_weight": float(DIAKRINO_SELECTOR_PRIOR_CURRENT_RAW_WEIGHT),
            "max_blend_weight": float(DIAKRINO_SELECTOR_PRIOR_CURRENT_MAX_BLEND),
            "evidence": dict(DIAKRINO_SELECTOR_PRIOR_EVIDENCE),
        }
    raw = rec.get("selector_prior_calibration")
    if not isinstance(raw, Mapping):
        raise ValueError("DIAKRINO sidecar trust record is missing selector_prior_calibration")
    out = dict(raw)
    mode = str(out.get("mode") or "").strip().lower()
    if mode not in {DIAKRINO_SELECTOR_PRIOR_CALIBRATION_CURRENT, DIAKRINO_SELECTOR_PRIOR_CALIBRATION_NONE}:
        raise ValueError(f"unknown DIAKRINO selector-prior calibration mode in trust record: {mode!r}")
    return {
        "mode": mode,
        "anchor_weights": dict(out.get("anchor_weights") or {}),
        "raw_weight": float(out.get("raw_weight", DIAKRINO_SELECTOR_PRIOR_CURRENT_RAW_WEIGHT)),
        "max_blend_weight": float(out.get("max_blend_weight", DIAKRINO_SELECTOR_PRIOR_CURRENT_MAX_BLEND)),
        "evidence": dict(out.get("evidence") or {}),
    }

"""Core-side DIAKRINO checkpoint qualification-record gate helpers.

The qualification record is emitted by ``scripts/analysis/diakrino_checkpoint_qualification.py``.
Core runtime consumers intentionally keep a tiny independent reader so production
paths do not import experiment scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "diakrino_checkpoint_qualification_v1"

DEFAULT_REQUIRED_GATES: tuple[str, ...] = (
    "checkpoint_geometry_load",
    "head_trust_gradient",
    "population_param_non_degenerate",
    "structural_redundancy_probe",
    "query_icl_accuracy",
    "sidecar_reemit",
    "selector_weights_candidate_probe",
    "s1_chunk_calibration_replay",
    "cross_chunk_logit_mean_drift",
)

# Per-consumer-class gate subsets (T-DIAKRINO-NAT-12 granular gating). Must stay in
# sync with ``scripts/analysis/diakrino_checkpoint_qualification.py``. Naming a class
# lets a checkpoint that clears the feature-selection gates enable FS consumers
# even when its classifier / warm-start / redundancy heads are dead (and so the
# whole-checkpoint ``overall_pass`` is false). The default (no class) keeps the
# strict all-nine policy, so existing callers are unchanged.
CONSUMER_CLASS_REQUIRED_GATES: dict[str, tuple[str, ...]] = {
    "feature_selection": (
        "checkpoint_geometry_load",
        "head_trust_gradient",
        "sidecar_reemit",
        "selector_weights_candidate_probe",
        "s1_chunk_calibration_replay",
        "cross_chunk_logit_mean_drift",
    ),
    "classifier": (
        "checkpoint_geometry_load",
        "head_trust_gradient",
        "query_icl_accuracy",
        "sidecar_reemit",
    ),
    "warm_start": (
        "checkpoint_geometry_load",
        "population_param_non_degenerate",
    ),
    "redundancy": (
        "checkpoint_geometry_load",
        "structural_redundancy_probe",
    ),
    "all": DEFAULT_REQUIRED_GATES,
}


def _resolve_required_gates(
    consumer_class: str | None,
    required_gates: Iterable[str] | None,
) -> tuple[str, ...] | None:
    """Resolve the gate subset for a consumer. ``None`` means 'unknown class'."""

    if required_gates is not None:
        return tuple(str(name) for name in required_gates)
    if consumer_class is not None:
        subset = CONSUMER_CLASS_REQUIRED_GATES.get(str(consumer_class))
        return None if subset is None else subset
    return DEFAULT_REQUIRED_GATES


def load_diakrino_qualification_record(path: str | Path) -> dict[str, Any] | None:
    """Load a qualification record, returning ``None`` on absent/unreadable files."""

    text = str(path or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(Path(text).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _gate_map(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for item in list(record.get("gates") or []):
        if isinstance(item, Mapping) and str(item.get("name") or ""):
            out[str(item["name"])] = item
    return out


def record_allows_gated_consumers(
    record: Mapping[str, Any] | None,
    *,
    consumer_class: str | None = None,
    required_gates: Iterable[str] | None = None,
) -> bool:
    """Fail-closed qualification policy for any core DIAKRINO consumer.

    With no ``consumer_class`` / ``required_gates`` this is the strict
    whole-checkpoint policy (schema + ``overall_pass`` + all nine gates), so
    every existing caller is unchanged. Naming a ``consumer_class`` (or an
    explicit ``required_gates`` subset) checks only that subset — a checkpoint
    that clears the feature-selection gates but fails the dead classifier /
    warm-start / redundancy gates can still enable FS consumers. Missing records,
    schema mismatches, unknown classes, absent gates, and ``not_run`` gates all
    fail closed.
    """

    if not isinstance(record, Mapping):
        return False
    if str(record.get("schema_version")) != SCHEMA_VERSION:
        return False
    strict_default = consumer_class is None and required_gates is None
    if strict_default and not bool(record.get("overall_pass", False)):
        return False
    required = _resolve_required_gates(consumer_class, required_gates)
    if required is None:  # unknown consumer_class
        return False
    gates = _gate_map(record)
    for name in required:
        gate = gates.get(str(name))
        if gate is None or not bool(gate.get("pass", False)) or str(gate.get("status", "")) != "pass":
            return False
    return True


def qualification_gate_status(
    path: str | Path,
    *,
    consumer_class: str | None = None,
    required_gates: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return JSON-safe diagnostic metadata for a qualification-record check.

    ``consumer_class`` selects a per-class gate subset (see
    ``CONSUMER_CLASS_REQUIRED_GATES``); omit it for the strict all-nine policy.
    """

    path_text = str(path or "").strip()
    strict_default = consumer_class is None and required_gates is None
    resolved = _resolve_required_gates(consumer_class, required_gates)
    unknown_class = resolved is None
    required = tuple(resolved or ())
    meta: dict[str, Any] = {
        "path": path_text,
        "consumer_class": str(consumer_class) if consumer_class is not None else None,
        "required_gates": list(required),
        "record_loaded": False,
        "allowed": False,
        "reason": "unknown_consumer_class" if unknown_class else "missing_qualification_record",
        "failed_gates": list(required),
    }
    if unknown_class:
        return meta
    record = load_diakrino_qualification_record(path_text)
    if record is None:
        if path_text:
            meta["reason"] = "record_unreadable"
        return meta
    meta["record_loaded"] = True
    meta["schema_version"] = str(record.get("schema_version") or "")
    meta["overall_pass"] = bool(record.get("overall_pass", False))
    if str(record.get("schema_version")) != SCHEMA_VERSION:
        meta["reason"] = "schema_mismatch"
        return meta
    if strict_default and not bool(record.get("overall_pass", False)):
        meta["reason"] = "overall_pass_false"
    gates = _gate_map(record)
    failed: list[str] = []
    for name in required:
        gate = gates.get(name)
        if gate is None or not bool(gate.get("pass", False)) or str(gate.get("status", "")) != "pass":
            failed.append(name)
    meta["failed_gates"] = failed
    if failed:
        if meta["reason"] in ("missing_qualification_record",):
            meta["reason"] = "required_gate_not_passed"
        return meta
    meta["allowed"] = True
    meta["reason"] = "allowed"
    return meta

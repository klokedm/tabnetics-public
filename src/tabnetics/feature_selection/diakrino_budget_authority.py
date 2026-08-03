"""Immutable P4-to-control budget authority for canonical DIAKRINO closeout runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .diakrino_identity import DiakrinoSidecarIdentityError, canonical_json_sha256, sha256_file


SCHEMA_VERSION = "diakrino_p4_budget_authority_v1"
ARM = "protected_native_null_jmi"
_SHA_FIELDS = (
    "diakrino_campaign_contract_sha256",
    "diakrino_identity_binding_sha256",
    "diakrino_native_nulls_sha256",
    "diakrino_paired_inference_views_sha256",
)


def _sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise DiakrinoSidecarIdentityError(f"P4 authority has no valid {field}")
    return digest


def _index_list(value: Any, *, field: str) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise DiakrinoSidecarIdentityError(f"P4 authority has malformed {field}") from exc
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise DiakrinoSidecarIdentityError(f"P4 authority has malformed {field}")
    try:
        indices = [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise DiakrinoSidecarIdentityError(f"P4 authority has malformed {field}") from exc
    if any(item < 0 for item in indices) or len(set(indices)) != len(indices):
        raise DiakrinoSidecarIdentityError(f"P4 authority has invalid {field}")
    return indices


def canonical_p4_result_row_sha256(row: Mapping[str, Any]) -> str:
    """Hash the exact finalized result row, excluding no mutable fields."""

    return canonical_json_sha256(
        dict(row), payload_schema_version="diakrino_canonical_p4_result_row_v1"
    )


def build_p4_budget_authority(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build a complete authority payload from one finalized P4 result row."""

    dataset_id = str(row.get("dataset_id") or "").strip()
    try:
        seed = int(row.get("seed"))
        additions = int(row.get("diakrino_additions"))
    except (TypeError, ValueError) as exc:
        raise DiakrinoSidecarIdentityError("P4 authority has malformed task or additions") from exc
    if not dataset_id or seed < 0 or additions < 0:
        raise DiakrinoSidecarIdentityError("P4 authority has invalid task or additions")
    if str(row.get("diakrino_feature_selection_arm") or "").strip() != ARM:
        raise DiakrinoSidecarIdentityError("P4 authority requires a protected_native_null_jmi result")
    protected_core = _index_list(
        row.get("diakrino_protected_core_original_indices_json"), field="protected core"
    )
    extras = _index_list(row.get("diakrino_extra_original_indices_json"), field="extra indices")
    if len(extras) != additions or set(protected_core) & set(extras):
        raise DiakrinoSidecarIdentityError("P4 authority additions do not match the protected result")
    authority = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "seed": seed,
        "p4_feature_selection_arm": ARM,
        "campaign_contract_sha256": _sha256(
            row.get("diakrino_campaign_contract_sha256"), field="diakrino_campaign_contract_sha256"
        ),
        "binding_sha256": _sha256(
            row.get("diakrino_identity_binding_sha256"), field="diakrino_identity_binding_sha256"
        ),
        "native_nulls_sha256": _sha256(
            row.get("diakrino_native_nulls_sha256"), field="diakrino_native_nulls_sha256"
        ),
        "paired_inference_views_sha256": _sha256(
            row.get("diakrino_paired_inference_views_sha256"),
            field="diakrino_paired_inference_views_sha256",
        ),
        "protected_core_indices": protected_core,
        "protected_core_sha256": canonical_json_sha256(
            protected_core, payload_schema_version="diakrino_protected_core_indices_v1"
        ),
        "p4_extra_indices": extras,
        "p4_realized_additions": additions,
        "p4_result_row_sha256": canonical_p4_result_row_sha256(row),
    }
    return authority


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DiakrinoSidecarIdentityError(f"refusing to overwrite existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise DiakrinoSidecarIdentityError(
                f"refusing to overwrite existing artifact: {path}"
            ) from exc
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def write_p4_budget_authority(path: str | Path, row: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    payload = build_p4_budget_authority(row)
    _write_once(output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return output


def read_p4_budget_authority(
    path: str | Path,
    *,
    dataset_id: str,
    seed: int,
    campaign_contract_sha256: str,
    binding_sha256: str,
    native_nulls_sha256: str,
    paired_inference_views_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify an authority is for the exact control task and return its digest."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiakrinoSidecarIdentityError("P4 budget authority is unreadable") from exc
    if not isinstance(payload, Mapping) or str(payload.get("schema_version") or "") != SCHEMA_VERSION:
        raise DiakrinoSidecarIdentityError("P4 budget authority schema is invalid")
    expected = {
        "dataset_id": str(dataset_id),
        "seed": int(seed),
        "campaign_contract_sha256": _sha256(campaign_contract_sha256, field="campaign contract"),
        "binding_sha256": _sha256(binding_sha256, field="binding"),
        "native_nulls_sha256": _sha256(native_nulls_sha256, field="native nulls"),
        "paired_inference_views_sha256": _sha256(
            paired_inference_views_sha256, field="paired views"
        ),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise DiakrinoSidecarIdentityError(f"P4 budget authority {field} does not match control task")
    if int(payload.get("p4_realized_additions", -1)) < 0:
        raise DiakrinoSidecarIdentityError("P4 budget authority has invalid realized additions")
    protected = _index_list(payload.get("protected_core_indices"), field="protected core")
    expected_core_sha = canonical_json_sha256(
        protected, payload_schema_version="diakrino_protected_core_indices_v1"
    )
    if payload.get("protected_core_sha256") != expected_core_sha:
        raise DiakrinoSidecarIdentityError("P4 budget authority protected core digest is invalid")
    _sha256(payload.get("p4_result_row_sha256"), field="P4 result row")
    return dict(payload), sha256_file(source)


__all__ = [
    "ARM",
    "SCHEMA_VERSION",
    "build_p4_budget_authority",
    "canonical_p4_result_row_sha256",
    "read_p4_budget_authority",
    "write_p4_budget_authority",
]

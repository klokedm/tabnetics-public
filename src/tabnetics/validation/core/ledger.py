"""Append-only promotion-attempt ledger helpers.

The ledger is a queryable scorecard artifact, not canonical project history.
GitHub issues and comments remain the durable work log; rows here capture the
structured evidence needed to compare intervention classes across campaigns.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LEDGER_SCHEMA_VERSION = "tabnetics_promotion_ledger_v1"


def _now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def infer_intervention_class(profile_id: str, comparison_id: str = "") -> str:
    text = f"{profile_id} {comparison_id}".lower()
    if any(token in text for token in ("prefilter", "screening", "filter")):
        return "prefilter"
    if any(token in text for token in ("oracle", "router", "selector", "mnpo")):
        return "oracle"
    if any(token in text for token in ("calibration", "calibrat", "zscore", "trust", "gate")):
        return "calibration"
    if any(token in text for token in ("method", "model", "classifier", "pls")):
        return "method_pool"
    return "other"


def _tier_summary(tier_breakdown: Mapping[str, Any]) -> tuple[dict[str, Any], int, int, int, int]:
    out: dict[str, Any] = {}
    total_n = wins = ties = losses = 0
    for tier, raw in dict(tier_breakdown or {}).items():
        item = dict(raw or {})
        n = _int_or_zero(item.get("n"))
        improved = _int_or_zero(item.get("improved"))
        same = _int_or_zero(item.get("same"))
        regressed = _int_or_zero(item.get("regressed"))
        total_n += n
        wins += improved
        ties += same
        losses += regressed
        out[str(tier)] = {
            "n": n,
            "mean_delta_balanced_accuracy": _float_or_none(item.get("mean_delta")),
            "worst_delta_balanced_accuracy": _float_or_none(item.get("worst_delta")),
            "best_delta_balanced_accuracy": _float_or_none(item.get("best_delta")),
            "wins": improved,
            "ties": same,
            "losses": regressed,
        }
    return out, total_n, wins, ties, losses


def row_from_pair_report(
    *,
    campaign: str,
    comparison_id: str,
    pair_report: Mapping[str, Any],
    campaign_promotion_decision: Mapping[str, Any] | None = None,
    intervention_class: str | None = None,
    evidence_links: Sequence[str] = (),
    source_artifacts: Mapping[str, str | Path] | None = None,
    compute_cost: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = str(pair_report.get("candidate") or "")
    baseline = str(pair_report.get("baseline") or "")
    tiers, tier_n, wins, ties, losses = _tier_summary(dict(pair_report.get("tier_breakdown") or {}))
    n_datasets = _int_or_zero(pair_report.get("observed_dataset_overlap")) or tier_n
    gate_report = dict(pair_report.get("gate") or {})
    strict_gate = dict(gate_report.get("strict") or {})
    pair_decision = dict(gate_report.get("promotion_decision") or {})
    row = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "campaign": str(campaign),
        "comparison_id": str(comparison_id),
        "profile_id": candidate,
        "parent_baseline": baseline,
        "intervention_class": str(intervention_class or infer_intervention_class(candidate, comparison_id)),
        "n_datasets": int(n_datasets),
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "mean_delta_balanced_accuracy": _float_or_none(pair_report.get("mean_delta")),
        "wilcoxon_p": _float_or_none(pair_report.get("wilcoxon_p")),
        "per_tier_deltas": tiers,
        "anti_gaming_telemetry": dict(pair_report.get("gaming_detectors") or {}),
        "compute_cost": dict(compute_cost or pair_report.get("compute_cost") or {}),
        "gate_result": strict_gate,
        "promotion_decision": pair_decision,
        "campaign_promotion_decision": dict(campaign_promotion_decision or {}),
        "evidence_links": [str(link) for link in evidence_links if str(link).strip()],
        "source_artifacts": {str(k): str(v) for k, v in dict(source_artifacts or {}).items()},
    }
    return _json_safe(row)


def rows_from_val4_report(
    report: Mapping[str, Any],
    *,
    campaign: str = "val4",
    evidence_links: Sequence[str] = (),
    source_artifacts: Mapping[str, str | Path] | None = None,
) -> list[dict[str, Any]]:
    decision = dict(report.get("promotion_decision") or {})
    pair_reports = dict(report.get("pair_reports") or {})
    rows: list[dict[str, Any]] = []
    for comparison_id, pair_report in pair_reports.items():
        if not isinstance(pair_report, Mapping):
            continue
        rows.append(
            row_from_pair_report(
                campaign=campaign,
                comparison_id=str(comparison_id),
                pair_report=pair_report,
                campaign_promotion_decision=decision,
                evidence_links=evidence_links,
                source_artifacts=source_artifacts,
            )
        )
    return rows


def append_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("a", encoding="utf-8") as fh:
        for raw in rows:
            row = dict(raw)
            row.setdefault("schema_version", LEDGER_SCHEMA_VERSION)
            row.setdefault("created_at", _now_iso())
            fh.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
            count += 1
    return count


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            out.append(payload)
    return out


def report(rows_or_path: Iterable[Mapping[str, Any]] | str | Path) -> dict[str, Any]:
    rows = load_rows(rows_or_path) if isinstance(rows_or_path, (str, Path)) else [dict(row) for row in rows_or_path]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("intervention_class") or "other"), []).append(row)

    by_class: dict[str, Any] = {}
    for klass, items in sorted(groups.items()):
        attempts = len(items)
        total_n = sum(_int_or_zero(item.get("n_datasets")) for item in items)
        wins = sum(_int_or_zero(item.get("wins")) for item in items)
        ties = sum(_int_or_zero(item.get("ties")) for item in items)
        losses = sum(_int_or_zero(item.get("losses")) for item in items)
        verified = [
            item
            for item in items
            if bool(dict(item.get("promotion_decision") or {}).get("promote", False))
            or str(dict(item.get("gate_result") or {}).get("verdict", "")).upper() == "PASS"
        ]
        deltas = [_float_or_none(item.get("mean_delta_balanced_accuracy")) for item in items]
        verified_deltas = [_float_or_none(item.get("mean_delta_balanced_accuracy")) for item in verified]
        deltas = [value for value in deltas if value is not None]
        verified_deltas = [value for value in verified_deltas if value is not None]
        denom = wins + ties + losses
        by_class[klass] = {
            "attempt_count": attempts,
            "verified_count": len(verified),
            "dataset_count": int(total_n),
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "dataset_win_rate": (float(wins) / float(denom)) if denom else None,
            "mean_delta_balanced_accuracy": (sum(deltas) / len(deltas)) if deltas else None,
            "mean_verified_delta_balanced_accuracy": (
                sum(verified_deltas) / len(verified_deltas)
            )
            if verified_deltas
            else None,
            "evidence_links": sorted(
                {
                    str(link)
                    for item in items
                    for link in list(item.get("evidence_links") or [])
                    if str(link).strip()
                }
            ),
        }
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "row_count": len(rows),
        "intervention_class_count": len(by_class),
        "by_intervention_class": by_class,
    }


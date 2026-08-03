"""DIAKRINO conformal FDP-controlled feature-selection scaffold.

This module is distinct from :mod:`tabnetics.feature_selection.conformal`, which
contains Stage-2 classifier prediction-set diagnostics.  The functions here
consume DIAKRINO FS-teacher schema-v2 ``conformal_score`` sidecar columns and apply a
Benjamini-Hochberg conformal p-value gate over per-feature scores.

The sidecar-facing entry point is intentionally opt-in and fail-closed: missing
qualification records, absent sidecars, missing conformal scores, and alignment
problems all return an empty selection plus diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


DEFAULT_DIAKRINO_CONFORMAL_CALIBRATION = "chunk_zscore"
DEFAULT_DIAKRINO_CONFORMAL_TARGET_FDP = 0.20
DEFAULT_DIAKRINO_CONFORMAL_NULL_FRACTION = 0.50
DEFAULT_DIAKRINO_CONFORMAL_MIN_NULL_SCORES = 4
DIAKRINO_CONFORMAL_SCORE_COLUMN = "conformal_score"


@dataclass(frozen=True)
class DiakrinoConformalSelectionResult:
    """Result from a DIAKRINO conformal selection pass."""

    selected_indices: np.ndarray
    scores: np.ndarray
    pvalues: np.ndarray
    selected_mask: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _calibration_diagnostics(calibrate: str) -> dict[str, Any]:
    mode = str(calibrate or DEFAULT_DIAKRINO_CONFORMAL_CALIBRATION).strip().lower()
    family = "chunk_zscore" if mode == "chunk_zscore" else mode
    calibration = "within_chunk_mean_std_then_split_conformal_bh" if mode == "chunk_zscore" else "split_conformal_bh"
    return {
        "score_column": DIAKRINO_CONFORMAL_SCORE_COLUMN,
        "normalize": mode,
        "normalization_mode": mode,
        "normalization_family": family,
        "calibration": calibration,
        "calibration_mode": calibration,
        "zscore_applied": bool(mode == "chunk_zscore"),
        # Output-compatible aliases used by the DIAKRINO replay qualification surface.
        "nn_probe_normalize": mode,
        "nn_probe_normalization_family": family,
        "nn_probe_calibration": calibration,
        "nn_probe_zscore_applied": bool(mode == "chunk_zscore"),
    }


def bh_reject(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg step-up rejection mask at target FDR/FDP level ``q``."""

    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    p = np.where(np.isfinite(p), p, 1.0)
    level = float(np.clip(float(q), 0.0, 1.0))
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    thresh = level * (np.arange(1, p.size + 1, dtype=np.float64) / float(p.size))
    below = sorted_p <= thresh
    if not bool(np.any(below)):
        return np.zeros(p.size, dtype=bool)
    cutoff = float(sorted_p[np.nonzero(below)[0].max()])
    return p <= cutoff


def conformal_pvalues(scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    """Conformal p-values for larger-is-more-relevant per-feature scores.

    ``p_j = (1 + #{null_score >= score_j}) / (1 + n_null)`` matches the Phase-0
    probe in ``bsc-run/phase0_probe/conformal_selection.py``.
    """

    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    null_arr = np.asarray(null_scores, dtype=np.float64).reshape(-1)
    null_arr = null_arr[np.isfinite(null_arr)]
    p = np.ones(score_arr.size, dtype=np.float64)
    if null_arr.size == 0:
        return p
    valid = np.isfinite(score_arr)
    if not bool(np.any(valid)):
        return p
    sorted_null = np.sort(null_arr, kind="mergesort")
    ge = null_arr.size - np.searchsorted(sorted_null, score_arr[valid], side="left")
    p[valid] = (1.0 + ge.astype(np.float64)) / (1.0 + float(null_arr.size))
    return p


def lower_tail_null_scores(
    scores: np.ndarray,
    *,
    null_fraction: float = DEFAULT_DIAKRINO_CONFORMAL_NULL_FRACTION,
    min_null_scores: int = DEFAULT_DIAKRINO_CONFORMAL_MIN_NULL_SCORES,
) -> np.ndarray:
    """Use the lower score tail as a conservative null proxy for sidecar replay.

    The pure conformal functions accept explicit null scores and are used for the
    reference/parity tests.  The sidecar scaffold has no oracle null labels, so it
    estimates a calibration-null pool from the lower tail and reports that source
    in diagnostics.  This remains default-off pending portfolio-level validation.
    """

    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    min_null = int(max(1, int(min_null_scores)))
    if finite.size <= min_null:
        return np.asarray([], dtype=np.float64)
    frac = float(np.clip(float(null_fraction), 0.0, 1.0))
    take = int(np.ceil(frac * float(finite.size)))
    take = max(min_null, take)
    take = min(take, int(finite.size) - 1)
    if take < min_null:
        return np.asarray([], dtype=np.float64)
    return np.sort(finite, kind="mergesort")[:take]


def select_with_conformal_fdp(
    scores: np.ndarray,
    null_scores: np.ndarray,
    *,
    target_fdp: float = DEFAULT_DIAKRINO_CONFORMAL_TARGET_FDP,
    max_features: int = 0,
    diagnostics: Mapping[str, Any] | None = None,
) -> DiakrinoConformalSelectionResult:
    """Select features by conformal p-values plus BH at ``target_fdp``."""

    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    pvalues = conformal_pvalues(score_arr, null_scores)
    selected_mask = bh_reject(pvalues, float(target_fdp))
    selected_mask &= np.isfinite(score_arr)

    cap = int(max(0, int(max_features)))
    if cap > 0 and int(np.count_nonzero(selected_mask)) > cap:
        selected = np.flatnonzero(selected_mask)
        keep_order = np.argsort(-score_arr[selected], kind="mergesort")[:cap]
        capped = np.zeros(score_arr.size, dtype=bool)
        capped[selected[keep_order]] = True
        selected_mask = capped

    selected_indices = np.flatnonzero(selected_mask).astype(np.int64)
    meta = dict(diagnostics or {})
    meta.update(
        {
            "target_fdp": float(np.clip(float(target_fdp), 0.0, 1.0)),
            "max_features": int(cap),
            "selected_count": int(selected_indices.size),
            "score_count": int(score_arr.size),
            "finite_score_count": int(np.count_nonzero(np.isfinite(score_arr))),
            "null_score_count": int(np.count_nonzero(np.isfinite(np.asarray(null_scores, dtype=np.float64)))),
        }
    )
    return DiakrinoConformalSelectionResult(
        selected_indices=selected_indices,
        scores=score_arr,
        pvalues=pvalues,
        selected_mask=selected_mask,
        diagnostics=meta,
    )


def _empty_result(n_features: int, diagnostics: Mapping[str, Any]) -> DiakrinoConformalSelectionResult:
    n = int(max(0, n_features))
    return DiakrinoConformalSelectionResult(
        selected_indices=np.zeros(0, dtype=np.int64),
        scores=np.full(n, np.nan, dtype=np.float64),
        pvalues=np.ones(n, dtype=np.float64),
        selected_mask=np.zeros(n, dtype=bool),
        diagnostics=dict(diagnostics),
    )


def _align_original_scores(
    scores: np.ndarray,
    *,
    n_columns: int,
    feature_mapping: np.ndarray | None,
) -> np.ndarray | None:
    orig = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_cols = int(n_columns)
    if feature_mapping is None:
        return orig.copy() if orig.shape[0] == n_cols else None
    fm = np.asarray(feature_mapping, dtype=np.int64).reshape(-1)
    if fm.shape[0] != n_cols or fm.size == 0:
        return None
    if int(np.min(fm)) < 0 or int(np.max(fm)) >= orig.shape[0]:
        return None
    return orig[fm]


def load_sidecar_conformal_selection(
    *,
    sidecar_path: str,
    dataset_id: str | None = None,
    n_columns: int,
    feature_mapping: np.ndarray | None = None,
    qualification_record: str = "",
    enabled: bool = False,
    target_fdp: float = DEFAULT_DIAKRINO_CONFORMAL_TARGET_FDP,
    calibrate: str = DEFAULT_DIAKRINO_CONFORMAL_CALIBRATION,
    null_fraction: float = DEFAULT_DIAKRINO_CONFORMAL_NULL_FRACTION,
    min_null_scores: int = DEFAULT_DIAKRINO_CONFORMAL_MIN_NULL_SCORES,
    max_features: int = 0,
) -> DiakrinoConformalSelectionResult:
    """Load DIAKRINO schema-v2 conformal scores and apply the FDP gate.

    The wrapper performs the qualification-record gate before reading the sidecar
    signal and returns an empty result on any unavailable or unqualified input.
    """

    n_cols = int(max(0, n_columns))
    meta: dict[str, Any] = {
        "applied": False,
        "reason": "disabled",
        "sidecar_path": str(sidecar_path or ""),
        "dataset_id": str(dataset_id or ""),
        "qualification_record": str(qualification_record or ""),
        "null_source": "lower_tail_score_proxy",
        "null_fraction": float(np.clip(float(null_fraction), 0.0, 1.0)),
        "min_null_scores": int(max(1, int(min_null_scores))),
    }
    meta.update(_calibration_diagnostics(calibrate))
    if not bool(enabled):
        return _empty_result(n_cols, meta)
    if not str(sidecar_path or "").strip():
        meta["reason"] = "missing_sidecar_path"
        return _empty_result(n_cols, meta)

    try:
        from .diakrino_qualification import qualification_gate_status

        # Conformal FDP-controlled selection is a feature-selection consumer:
        # gate on the FS-class subset (T-DIAKRINO-NAT-12 granular gating).
        gate = qualification_gate_status(
            str(qualification_record or ""), consumer_class="feature_selection"
        )
        meta["qualification_gate"] = dict(gate)
        if not bool(gate.get("allowed", False)):
            meta["reason"] = str(gate.get("reason") or "qualification_gate_failed")
            return _empty_result(n_cols, meta)
    except Exception:
        meta["reason"] = "qualification_gate_error"
        return _empty_result(n_cols, meta)

    try:
        from .diakrino_sidecar import DiakrinoSidecar

        sidecar = DiakrinoSidecar.load(str(sidecar_path), dataset_id=dataset_id)
        if sidecar is None:
            meta["reason"] = "sidecar_unreadable"
            return _empty_result(n_cols, meta)
        meta["sidecar"] = sidecar.resolution_diagnostics()
        original_scores = sidecar.conformal_scores(calibrate=str(calibrate or DEFAULT_DIAKRINO_CONFORMAL_CALIBRATION))
    except Exception:
        meta["reason"] = "sidecar_conformal_score_error"
        return _empty_result(n_cols, meta)

    if original_scores is None:
        meta["reason"] = "missing_conformal_score"
        return _empty_result(n_cols, meta)
    col_scores = _align_original_scores(
        original_scores,
        n_columns=n_cols,
        feature_mapping=feature_mapping,
    )
    if col_scores is None:
        meta["reason"] = "feature_mapping_misaligned"
        return _empty_result(n_cols, meta)
    if not bool(np.any(np.isfinite(col_scores))):
        meta["reason"] = "no_finite_conformal_scores"
        return _empty_result(n_cols, meta)

    null_scores = lower_tail_null_scores(
        col_scores,
        null_fraction=float(null_fraction),
        min_null_scores=int(min_null_scores),
    )
    if null_scores.size < int(max(1, int(min_null_scores))):
        meta["reason"] = "insufficient_null_scores"
        return _empty_result(n_cols, meta)

    meta["applied"] = True
    meta["reason"] = "applied"
    return select_with_conformal_fdp(
        col_scores,
        null_scores,
        target_fdp=float(target_fdp),
        max_features=int(max_features),
        diagnostics=meta,
    )


def diakrino_conformal_selection_method(
    X: np.ndarray,
    y: np.ndarray,
    n_target_features: int,
    *,
    sidecar_path: str,
    dataset_id: str | None = None,
    feature_mapping: np.ndarray | None = None,
    qualification_record: str = "",
    enabled: bool = False,
    target_fdp: float = DEFAULT_DIAKRINO_CONFORMAL_TARGET_FDP,
    calibrate: str = DEFAULT_DIAKRINO_CONFORMAL_CALIBRATION,
    null_fraction: float = DEFAULT_DIAKRINO_CONFORMAL_NULL_FRACTION,
    min_null_scores: int = DEFAULT_DIAKRINO_CONFORMAL_MIN_NULL_SCORES,
    max_features: int = 0,
) -> tuple[dict, dict, dict[str, Any]]:
    """FeatureSelector method-contract adapter for the DIAKRINO conformal scaffold."""

    X_arr = np.asarray(X)
    n_cols = int(X_arr.shape[1]) if X_arr.ndim == 2 else 0
    cap = int(max_features)
    if cap <= 0:
        cap = 0
    result = load_sidecar_conformal_selection(
        sidecar_path=str(sidecar_path or ""),
        dataset_id=dataset_id,
        n_columns=n_cols,
        feature_mapping=feature_mapping,
        qualification_record=str(qualification_record or ""),
        enabled=bool(enabled),
        target_fdp=float(target_fdp),
        calibrate=str(calibrate or DEFAULT_DIAKRINO_CONFORMAL_CALIBRATION),
        null_fraction=float(null_fraction),
        min_null_scores=int(min_null_scores),
        max_features=cap,
    )
    finite = np.isfinite(result.scores)
    all_scores = {int(i): float(result.scores[i]) for i in np.flatnonzero(finite)}
    if result.selected_indices.size == 0:
        return {}, all_scores, dict(result.diagnostics)
    selected = result.selected_indices.astype(int)
    method_result = {
        "selected_indices": selected,
        "scores": {int(i): float(result.scores[i]) for i in selected},
        "all_scores": result.scores.astype(float),
        "conformal_pvalues": result.pvalues.astype(float),
        "diagnostics": dict(result.diagnostics),
    }
    return method_result, all_scores, dict(result.diagnostics)

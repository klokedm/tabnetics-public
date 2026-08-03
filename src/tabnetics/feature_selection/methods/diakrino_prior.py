"""DIAKRINO prior / screening candidate methods (native integration §2.1).

These are opt-in MNPO candidate selectors that read a persisted DIAKRINO FS-teacher sidecar
parquet (no model inference) and emit a calibrated per-feature relevance score, competing
in the portfolio exactly like ``mutual_information_selection``.  They never select alone
(``default_enabled=False`` in the registry) and degrade to an empty result — a graceful
skip — whenever the sidecar is missing, misaligned, or pandas is unavailable.

Alignment: the FS methods operate on the ``X_uncorr`` column space, while the sidecar
scores are indexed by ORIGINAL feature.  The selector stages
``self._current_feature_mapping`` (X_uncorr column -> original index) so the score vector
can be gathered onto the current column space.  Any width/index mismatch falls back to the
empty (skip) result rather than misaligning.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _empty() -> tuple[dict, dict]:
    return {}, {}


def diakrino_prior_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_target_features: int,
    *,
    sidecar_path: str,
    score_column: str,
    calibrate: str,
    feature_mapping: np.ndarray | None,
    top_k: int = 0,
    dataset_id: str | None = None,
) -> tuple[dict, dict]:
    """Select features by a calibrated DIAKRINO per-feature relevance score from the sidecar.

    Returns the standard ``(results, all_scores)`` method contract, or ``({}, {})`` to
    skip when the signal is unavailable/misaligned.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < 1:
        return _empty()
    n_cols = int(X.shape[1])
    if not sidecar_path:
        return _empty()

    try:
        from ..diakrino_sidecar import DiakrinoSidecar
    except Exception:
        return _empty()

    sc = DiakrinoSidecar.load(sidecar_path, dataset_id=dataset_id)
    if sc is None:
        return _empty()
    orig = sc.scalar_scores(str(score_column), calibrate=str(calibrate or "chunk_zscore"))
    if orig is None:
        return _empty()
    orig = np.asarray(orig, dtype=float).ravel()

    # Map original-indexed sidecar scores onto the current X_uncorr column space.
    if feature_mapping is not None:
        fm = np.asarray(feature_mapping, dtype=np.int64).ravel()
        if fm.shape[0] != n_cols or fm.size == 0 or int(fm.max()) >= orig.shape[0]:
            return _empty()
        col_scores = orig[fm]
    elif orig.shape[0] == n_cols:
        col_scores = orig
    else:
        return _empty()

    if not np.any(np.isfinite(col_scores)):
        return _empty()
    col_scores = np.where(np.isfinite(col_scores), col_scores, np.nanmin(col_scores))

    k = int(top_k) if int(top_k) > 0 else int(n_target_features)
    k = max(1, min(k, n_cols))
    selected_indices = np.argsort(col_scores)[::-1][:k].astype(int)

    results = {
        "selected_indices": selected_indices,
        "scores": {int(idx): float(col_scores[idx]) for idx in selected_indices},
        "all_scores": col_scores.astype(float),
    }
    return results, {int(i): float(col_scores[i]) for i in range(n_cols)}

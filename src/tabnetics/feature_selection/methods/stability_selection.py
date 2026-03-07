"""EATS threshold calibration for stability selection (VAL12_Suggestions §5.1).

Provides a standalone ``EATSThresholdCalibrator`` that replaces fixed
stability thresholds (e.g. 0.6, 0.8) with a data-driven elbow-adaptive
cutoff.  The core algorithm is the same as used internally by IPSS —
this module makes it a first-class, reusable component.

Algorithm (simplified):
1. Receive feature stability scores (from any selection method).
2. Optionally receive null scores (from label-permutation runs).
3. Compute an exclusion floor from the null distribution.
4. Find the *elbow* in the threshold-vs-count curve above the floor.
5. Return ``max(elbow, min_threshold, exclusion_floor)`` as the
   calibrated threshold.

Usage::

    from tabnetics.feature_selection.methods.stability_selection import (
        EATSThresholdCalibrator,
    )

    calibrator = EATSThresholdCalibrator(
        exclusion_quantile=0.90,
        min_threshold=0.45,
    )
    threshold, meta = calibrator.calibrate(
        stability_scores=selection_frequencies,
        null_scores=permuted_frequencies,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class EATSThresholdCalibrator:
    """Elbow-Adaptive Threshold Selection for stability methods.

    Parameters
    ----------
    exclusion_quantile : float
        Quantile of the null-score distribution used to set the
        exclusion floor (default 0.90).  Scores below this floor are
        not considered as threshold candidates.
    min_threshold : float
        Hard lower bound on the returned threshold (default 0.45).
    fallback_threshold : float
        Threshold returned when calibration is impossible (e.g. fewer
        than 3 eligible scores).  Defaults to 0.60.
    """

    exclusion_quantile: float = 0.90
    min_threshold: float = 0.45
    fallback_threshold: float = 0.60

    def __post_init__(self) -> None:
        self.exclusion_quantile = float(
            np.clip(self.exclusion_quantile, 0.5, 0.995)
        )
        self.min_threshold = float(
            np.clip(self.min_threshold, 0.05, 0.95)
        )
        self.fallback_threshold = float(
            np.clip(self.fallback_threshold, 0.05, 0.99)
        )

    def calibrate(
        self,
        stability_scores: np.ndarray,
        null_scores: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute a data-driven stability threshold.

        Parameters
        ----------
        stability_scores : (p,) array
            Stability frequency (or score) for each feature, typically
            in [0, 1].
        null_scores : (m,) array or None
            Stability scores obtained under label permutation (null
            distribution).  If ``None``, ``exclusion_floor`` is set to
            ``min_threshold``.

        Returns
        -------
        threshold : float
            Calibrated threshold.
        meta : dict
            Diagnostic metadata including ``exclusion_floor``,
            ``elbow_threshold``, and ``n_threshold_candidates``.
        """
        scores = np.asarray(stability_scores, dtype=float).ravel()
        if scores.size == 0:
            return float(self.fallback_threshold), {
                "eats_exclusion_floor": float("nan"),
                "eats_elbow_threshold": float("nan"),
                "eats_n_threshold_candidates": 0,
                "eats_calibrated": False,
            }

        # Exclusion floor from null distribution.
        if null_scores is not None and np.asarray(null_scores).size > 0:
            null = np.asarray(null_scores, dtype=float).ravel()
            exclusion_floor = float(
                np.quantile(null, self.exclusion_quantile)
            )
        else:
            exclusion_floor = float(self.min_threshold)
        exclusion_floor = float(np.clip(exclusion_floor, 0.0, 1.0))

        eligible = scores[scores >= exclusion_floor]
        if eligible.size < 3:
            threshold = max(
                float(self.min_threshold),
                exclusion_floor,
                float(self.fallback_threshold),
            )
            return threshold, {
                "eats_exclusion_floor": exclusion_floor,
                "eats_elbow_threshold": threshold,
                "eats_n_threshold_candidates": int(eligible.size),
                "eats_calibrated": False,
            }

        # Elbow detection on threshold-vs-count curve.
        thresholds = np.unique(np.round(eligible, decimals=6))
        thresholds = np.sort(thresholds)[::-1]  # descending
        selected_counts = np.asarray(
            [(scores >= t).sum() for t in thresholds], dtype=float
        )

        if (
            thresholds.size < 3
            or np.max(selected_counts) <= np.min(selected_counts) + 1e-9
        ):
            elbow_threshold = float(thresholds[0])
        else:
            # Normalise to [0, 1] interval and find max distance from
            # the diagonal baseline (classic Kneedle / elbow method).
            y = (selected_counts - selected_counts.min()) / (
                selected_counts.max() - selected_counts.min() + 1e-12
            )
            baseline = np.linspace(y[0], y[-1], num=y.size)
            distances = np.abs(y - baseline)
            elbow_idx = int(np.argmax(distances))
            elbow_threshold = float(thresholds[elbow_idx])

        threshold = max(
            float(self.min_threshold),
            exclusion_floor,
            elbow_threshold,
        )
        threshold = float(np.clip(threshold, 0.0, 1.0))
        return threshold, {
            "eats_exclusion_floor": exclusion_floor,
            "eats_elbow_threshold": elbow_threshold,
            "eats_n_threshold_candidates": int(thresholds.size),
            "eats_calibrated": True,
        }


def eats_calibrate_threshold(
    stability_scores: np.ndarray,
    null_scores: Optional[np.ndarray] = None,
    *,
    exclusion_quantile: float = 0.90,
    min_threshold: float = 0.45,
    fallback_threshold: float = 0.60,
) -> Tuple[float, Dict[str, float]]:
    """Functional convenience wrapper around ``EATSThresholdCalibrator``.

    Parameters and return values are identical to
    ``EATSThresholdCalibrator.calibrate``.
    """
    cal = EATSThresholdCalibrator(
        exclusion_quantile=exclusion_quantile,
        min_threshold=min_threshold,
        fallback_threshold=fallback_threshold,
    )
    return cal.calibrate(stability_scores, null_scores)

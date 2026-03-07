"""Feature score caching and normalization utilities.

Extracted from ``tabnetics.feature_selection`` during Phase 6 module decomposition.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)


class _FeatureScoreCache:
    """
    Cache for univariate feature scores computed once per fit_transform() call.

    Eliminates redundant MI/F-test computations across methods. Per audit
    (Claude-Audit-2026-02-14.md §8.1), scores are recomputed 3-4 times
    independently in:
    - _prefilter_feature_pool()
    - _iterative_redundancy_pruning_core()
    - _class_dominance_pareto_prefilter()
    - _binary_class_prefilter_scores()

    This cache eliminates that redundancy.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, random_state: int, problem_type: str = 'classification'):
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y)
        self._random_state = random_state
        self._problem_type = problem_type
        self._cache: dict = {}

    @property
    def mi_scores(self) -> np.ndarray:
        """Mutual information scores (classification) or f_regression (regression)."""
        if 'mi' not in self._cache:
            if self._problem_type == 'classification':
                self._cache['mi'] = mutual_info_classif(
                    self._X, self._y, random_state=self._random_state
                )
            else:
                # For regression, use f_regression as proxy
                f_vals, _ = f_regression(self._X, self._y)
                self._cache['mi'] = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
        return self._cache['mi']

    @property
    def f_scores(self) -> tuple:
        """F-test scores and p-values."""
        if 'f' not in self._cache:
            if self._problem_type == 'classification':
                f_vals, p_vals = f_classif(self._X, self._y)
            else:
                f_vals, p_vals = f_regression(self._X, self._y)
            self._cache['f'] = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
            self._cache['f_pvals'] = np.nan_to_num(p_vals, nan=1.0, posinf=1.0, neginf=1.0)
        return self._cache['f'], self._cache['f_pvals']

    @property
    def blended_prefilter(self) -> np.ndarray:
        """
        MI + F-test blended scores with default 60/40 weights.

        Note: These weights (0.60/0.40) are currently hard-coded per audit §6.4.
        Future enhancement: make configurable via PrefilterConfig.
        """
        if 'blended' not in self._cache:
            mi = self._safe_normalize(self.mi_scores)
            f, _ = self.f_scores
            f_norm = self._safe_normalize(f)
            # Default blend: 60% MI, 40% F-test (see audit §6.4)
            self._cache['blended'] = 0.60 * mi + 0.40 * f_norm
        return self._cache['blended']

    @staticmethod
    def _safe_normalize(v: np.ndarray) -> np.ndarray:
        """Min-max normalization with zero-range protection."""
        v = np.asarray(v, dtype=float).ravel()
        vmin, vmax = np.min(v), np.max(v)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)

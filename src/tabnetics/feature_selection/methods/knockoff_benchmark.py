"""knockpy/GRIP2 targeted FDR benchmark (VAL12_Suggestions §5.2).

Provides a benchmark-lane wrapper around the ``knockpy`` package for
model-X knockoff feature selection with FDR control.  This is a
**benchmark reference** — it is not integrated into the default
portfolio or MNPO evaluation.

Usage::

    from tabnetics.feature_selection.methods.knockoff_benchmark import (
        KnockoffBenchmarkSelector,
        KNOCKPY_AVAILABLE,
    )

    if KNOCKPY_AVAILABLE:
        selector = KnockoffBenchmarkSelector(fdr=0.10, random_state=42)
        result, meta = selector.select(X, y, n_target_features=10)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------
# Optional dependency
# ------------------------------------------------------------------

try:
    import knockpy
    from knockpy import KnockoffFilter

    KNOCKPY_AVAILABLE = True
except Exception:  # pragma: no cover
    knockpy = None  # type: ignore[assignment]
    KnockoffFilter = None  # type: ignore[assignment,misc]
    KNOCKPY_AVAILABLE = False


@dataclass
class KnockoffBenchmarkSelector:
    """Model-X knockoff benchmark selector (wraps ``knockpy``).

    Parameters
    ----------
    fdr : float
        Target false discovery rate (default 0.10).
    knockoff_type : str
        Knockoff construction method (default ``"gaussian"``).
        Options depend on the ``knockpy`` version installed.
    statistic : str
        Feature importance statistic (default ``"lasso"``).
    random_state : int or None
        Random seed for reproducibility.
    """

    fdr: float = 0.10
    knockoff_type: str = "gaussian"
    statistic: str = "lasso"
    random_state: Optional[int] = None

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_target_features: int = 10,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run knockoff selection and return results.

        Parameters
        ----------
        X : (n, p) array
        y : (n,) array
        n_target_features : int
            Ignored for knockoff (FDR controls selection size), but
            included for API compatibility with other selectors.

        Returns
        -------
        result : dict
            ``"selected_indices"`` (array of int), ``"selected_count"``
            (int), ``"w_statistics"`` (array of knockoff W-statistics).
        meta : dict
            Diagnostic metadata including FDR, method, and whether
            the real knockpy backend was used.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        if not KNOCKPY_AVAILABLE:
            warnings.warn(
                "knockpy not installed — returning empty selection.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._empty_result(n_features), {
                "knockoff_backend": "unavailable",
                "fdr": self.fdr,
            }

        if n_samples < 2 or n_features == 0:
            return self._empty_result(n_features), {
                "knockoff_backend": "skipped",
                "reason": "insufficient data",
                "fdr": self.fdr,
            }

        try:
            selected = self._run_knockpy(X, y)
        except Exception as exc:
            warnings.warn(
                f"knockpy failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._empty_result(n_features), {
                "knockoff_backend": "error",
                "error": str(exc),
                "fdr": self.fdr,
            }

        selected_idx = np.sort(np.asarray(selected, dtype=int))
        return {
            "selected_indices": selected_idx,
            "selected_count": int(selected_idx.size),
        }, {
            "knockoff_backend": "knockpy",
            "fdr": self.fdr,
            "knockoff_type": self.knockoff_type,
            "statistic": self.statistic,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_knockpy(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Execute knockpy KnockoffFilter."""
        kfilter = KnockoffFilter(
            ksampler=self.knockoff_type,
            fstat=self.statistic,
        )
        rejections = kfilter.forward(
            X=X,
            y=y,
            fdr=self.fdr,
            seed=self.random_state if self.random_state is not None else 0,
        )
        # knockpy returns a boolean array or index array depending on version.
        rej = np.asarray(rejections).ravel()
        if rej.dtype == bool:
            return np.where(rej)[0]
        return rej.astype(int)

    @staticmethod
    def _empty_result(n_features: int) -> Dict[str, Any]:
        return {
            "selected_indices": np.array([], dtype=int),
            "selected_count": 0,
        }


def knockoff_benchmark_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_target_features: int = 10,
    *,
    fdr: float = 0.10,
    knockoff_type: str = "gaussian",
    statistic: str = "lasso",
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Functional convenience wrapper for ``KnockoffBenchmarkSelector``."""
    sel = KnockoffBenchmarkSelector(
        fdr=fdr,
        knockoff_type=knockoff_type,
        statistic=statistic,
        random_state=random_state,
    )
    return sel.select(X, y, n_target_features)

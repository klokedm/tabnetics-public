"""Dataset meta-feature extraction helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import entropy as _shannon_entropy


def extract_meta_features(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Extract dataset meta-features useful for tier assignment and analysis."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    n, p = X.shape if X.ndim == 2 else (X.shape[0], 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    classes, counts = np.unique(y, return_counts=True)
    class_count = float(len(classes))
    if class_count > 1:
        proportions = counts / counts.sum()
        raw_entropy = _shannon_entropy(proportions, base=np.e)
        class_balance_entropy = float(raw_entropy / np.log(class_count))
    else:
        class_balance_entropy = 0.0

    p_over_n = float(p) / float(n) if n > 0 else 0.0

    if p >= 3 and n >= 2:
        rng = np.random.RandomState(42)
        max_cols = 200
        if p > max_cols:
            col_idx = rng.choice(p, max_cols, replace=False)
            X_sub = X[:, col_idx]
        else:
            X_sub = X
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(X_sub, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        tri_idx = np.triu_indices_from(corr, k=1)
        abs_corr = np.sort(np.abs(corr[tri_idx]))[::-1]
        xs = np.arange(len(abs_corr), dtype=float)
        try:
            def _exp_decay(x, a, b):
                return a * np.exp(-b * x)

            popt, _ = curve_fit(
                _exp_decay,
                xs,
                abs_corr,
                p0=[abs_corr[0] if len(abs_corr) else 1.0, 0.01],
                maxfev=2000,
            )
            correlation_spectrum_decay = float(max(popt[1], 0.0))
        except Exception:
            correlation_spectrum_decay = 0.0
    else:
        correlation_spectrum_decay = 0.0

    if p > 0:
        heaping_count = 0
        for j in range(p):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0 and np.all(valid == np.round(valid)):
                heaping_count += 1
        heaping_fraction = float(heaping_count) / float(p)
    else:
        heaping_fraction = 0.0

    return {
        "n": float(n),
        "p": float(p),
        "p_over_n": float(p_over_n),
        "class_count": float(class_count),
        "class_balance_entropy": float(class_balance_entropy),
        "correlation_spectrum_decay": float(correlation_spectrum_decay),
        "heaping_fraction": float(heaping_fraction),
    }


__all__ = ["extract_meta_features"]

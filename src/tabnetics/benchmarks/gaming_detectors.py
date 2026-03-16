"""Read-only anti-gaming diagnostics for benchmark result analysis.

Detectors are intentionally decoupled from the feature-selection runtime path.
They consume flat result rows (DataFrame-compatible mappings) and return
diagnostics only; they do not mutate inputs or gate promotions directly.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd


def _to_frame(rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame(list(rows))


def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.asarray([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def detect_class_correlation(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    metric_col: str = "delta_balanced_accuracy",
    class_count_col: str = "class_count",
    min_points: int = 6,
    rho_threshold: float = 0.60,
) -> Dict[str, Any]:
    """GT-1: flag strong monotonic relationship between class-count and uplift."""
    df = _to_frame(rows)
    if metric_col not in df.columns or class_count_col not in df.columns:
        return {
            "detector": "class_correlation",
            "status": "insufficient_columns",
            "flagged": False,
            "n_points": 0,
            "spearman_rho": float("nan"),
        }

    use = df[[metric_col, class_count_col]].copy()
    use[metric_col] = pd.to_numeric(use[metric_col], errors="coerce")
    use[class_count_col] = pd.to_numeric(use[class_count_col], errors="coerce")
    use = use.dropna()
    n = int(use.shape[0])
    if n < int(min_points):
        return {
            "detector": "class_correlation",
            "status": "insufficient_points",
            "flagged": False,
            "n_points": n,
            "spearman_rho": float("nan"),
        }

    metric_rank = use[metric_col].rank(method="average").to_numpy(dtype=float)
    class_rank = use[class_count_col].rank(method="average").to_numpy(dtype=float)
    rho = float(np.corrcoef(metric_rank, class_rank)[0, 1]) if n > 1 else float("nan")
    if not np.isfinite(rho):
        rho = 0.0
    flagged = bool(abs(rho) >= float(rho_threshold))
    return {
        "detector": "class_correlation",
        "status": "ok",
        "flagged": flagged,
        "n_points": n,
        "spearman_rho": float(rho),
        "rho_threshold": float(rho_threshold),
    }


def detect_tier_imbalance(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    delta_col: str = "delta_balanced_accuracy",
    tier_col: str = "tier",
    gap_threshold: float = 0.03,
) -> Dict[str, Any]:
    """GT-2: hard-tier gains with easy/medium regressions."""
    df = _to_frame(rows)
    if delta_col not in df.columns or tier_col not in df.columns:
        return {
            "detector": "tier_imbalance",
            "status": "insufficient_columns",
            "flagged": False,
        }
    use = df[[delta_col, tier_col]].copy()
    use[delta_col] = pd.to_numeric(use[delta_col], errors="coerce")
    use[tier_col] = use[tier_col].astype(str).str.strip().str.lower()
    use = use.dropna(subset=[delta_col, tier_col])

    means = use.groupby(tier_col)[delta_col].mean().to_dict()
    easy_mean = float(means.get("easy", np.nan))
    medium_mean = float(means.get("medium", np.nan))
    hard_mean = float(means.get("hard", np.nan))
    baseline = np.nanmean(np.asarray([easy_mean, medium_mean], dtype=float))
    gap = float(hard_mean - baseline) if np.isfinite(hard_mean) and np.isfinite(baseline) else float("nan")

    flagged = bool(
        np.isfinite(hard_mean)
        and hard_mean > 0.0
        and (
            (np.isfinite(easy_mean) and easy_mean < 0.0)
            or (np.isfinite(medium_mean) and medium_mean < 0.0)
        )
        and np.isfinite(gap)
        and gap >= float(gap_threshold)
    )
    return {
        "detector": "tier_imbalance",
        "status": "ok",
        "flagged": flagged,
        "easy_mean": easy_mean,
        "medium_mean": medium_mean,
        "hard_mean": hard_mean,
        "hard_vs_easy_medium_gap": gap,
        "gap_threshold": float(gap_threshold),
    }


def detect_threshold_hugging(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    delta_col: str = "delta_balanced_accuracy",
    tier_col: str = "tier",
    epsilon: float = 0.002,
    min_count_per_tier: int = 5,
    fraction_threshold: float = 0.50,
) -> Dict[str, Any]:
    """GT-3: suspicious clustering right around gate boundaries."""
    df = _to_frame(rows)
    if delta_col not in df.columns or tier_col not in df.columns:
        return {
            "detector": "threshold_hugging",
            "status": "insufficient_columns",
            "flagged": False,
        }
    thresholds = {"easy": -0.02, "medium": -0.05, "hard": 0.0}
    use = df[[delta_col, tier_col]].copy()
    use[delta_col] = pd.to_numeric(use[delta_col], errors="coerce")
    use[tier_col] = use[tier_col].astype(str).str.strip().str.lower()
    use = use.dropna(subset=[delta_col, tier_col])

    shares: Dict[str, float] = {}
    flagged = False
    for tier, threshold in thresholds.items():
        tier_vals = pd.to_numeric(
            use.loc[use[tier_col] == tier, delta_col],
            errors="coerce",
        ).dropna()
        n = int(tier_vals.shape[0])
        if n < int(min_count_per_tier):
            shares[tier] = float("nan")
            continue
        share = float(np.mean(np.abs(tier_vals.to_numpy(dtype=float) - float(threshold)) <= float(epsilon)))
        shares[tier] = share
        if share >= float(fraction_threshold):
            flagged = True
    return {
        "detector": "threshold_hugging",
        "status": "ok",
        "flagged": bool(flagged),
        "epsilon": float(epsilon),
        "fraction_threshold": float(fraction_threshold),
        "share_by_tier": shares,
    }


def detect_seed_variance(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    metric_col: str = "balanced_accuracy",
    dataset_col: str = "dataset_id",
    seed_col: str = "seed",
    std_threshold: float = 0.03,
) -> Dict[str, Any]:
    """GT-4: high across-seed volatility indicates fragile gains."""
    df = _to_frame(rows)
    needed = {metric_col, dataset_col, seed_col}
    if not needed.issubset(set(df.columns)):
        return {
            "detector": "seed_variance",
            "status": "insufficient_columns",
            "flagged": False,
            "high_variance_datasets": [],
        }

    use = df[[dataset_col, seed_col, metric_col]].copy()
    use[dataset_col] = use[dataset_col].astype(str)
    use[metric_col] = pd.to_numeric(use[metric_col], errors="coerce")
    use = use.dropna(subset=[dataset_col, metric_col])
    grouped = use.groupby(dataset_col)[metric_col]
    std_by_dataset = grouped.std(ddof=0).fillna(0.0)
    high = std_by_dataset[std_by_dataset > float(std_threshold)]
    return {
        "detector": "seed_variance",
        "status": "ok",
        "flagged": bool(not high.empty),
        "std_threshold": float(std_threshold),
        "n_datasets": int(std_by_dataset.shape[0]),
        "median_std": float(std_by_dataset.median()) if std_by_dataset.shape[0] else float("nan"),
        "high_variance_datasets": [str(k) for k in high.sort_values(ascending=False).index.tolist()],
        "std_by_dataset": {str(k): float(v) for k, v in std_by_dataset.items()},
    }


def detect_bellwether_concentration(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    delta_col: str = "delta_balanced_accuracy",
    dataset_col: str = "dataset_id",
    top_fraction: float = 0.20,
    share_threshold: float = 0.70,
) -> Dict[str, Any]:
    """GT-5: flag when a tiny bellwether subset drives most uplift."""
    df = _to_frame(rows)
    if delta_col not in df.columns or dataset_col not in df.columns:
        return {
            "detector": "bellwether_concentration",
            "status": "insufficient_columns",
            "flagged": False,
            "top_share": float("nan"),
        }
    use = df[[dataset_col, delta_col]].copy()
    use[dataset_col] = use[dataset_col].astype(str)
    use[delta_col] = pd.to_numeric(use[delta_col], errors="coerce")
    use = use.dropna(subset=[dataset_col, delta_col])
    means = use.groupby(dataset_col)[delta_col].mean()
    positive = means[means > 0.0].sort_values(ascending=False)
    total = float(positive.sum())
    if positive.empty or total <= 1e-12:
        return {
            "detector": "bellwether_concentration",
            "status": "no_positive_uplift",
            "flagged": False,
            "top_share": float("nan"),
        }

    n_top = int(max(1, np.ceil(float(top_fraction) * float(positive.shape[0]))))
    top_share = float(positive.iloc[:n_top].sum() / total)
    flagged = bool(top_share >= float(share_threshold))
    return {
        "detector": "bellwether_concentration",
        "status": "ok",
        "flagged": flagged,
        "top_fraction": float(top_fraction),
        "share_threshold": float(share_threshold),
        "top_share": float(top_share),
        "top_datasets": [str(k) for k in positive.iloc[:n_top].index.tolist()],
    }


def run_gaming_detectors(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> Dict[str, Any]:
    """Run the GT-1..GT-5 detector suite over paired result rows."""
    gt1 = detect_class_correlation(rows)
    gt2 = detect_tier_imbalance(rows)
    gt3 = detect_threshold_hugging(rows)
    gt4 = detect_seed_variance(rows)
    gt5 = detect_bellwether_concentration(rows)
    detectors = {
        "gt1_class_correlation": gt1,
        "gt2_tier_imbalance": gt2,
        "gt3_threshold_hugging": gt3,
        "gt4_seed_variance": gt4,
        "gt5_bellwether": gt5,
    }
    detectors["any_flagged"] = bool(
        any(bool((payload or {}).get("flagged", False)) for payload in detectors.values() if isinstance(payload, dict))
    )
    return detectors


__all__ = [
    "detect_bellwether_concentration",
    "detect_class_correlation",
    "detect_seed_variance",
    "detect_threshold_hugging",
    "detect_tier_imbalance",
    "run_gaming_detectors",
]

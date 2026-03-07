"""Unified validation suite for FS, DF, and integrated pipeline benchmarks.

This module implements the datasets listed in VALIDATION.md and exposes:
1. Dataset-set selection (`--dataset-sets`, `--datasets`, `--exclude-datasets`)
2. Component ablation profiles (`--ablation-profile`, enable/disable overrides)
3. Run artifact recording (run-level CSV, summaries, metadata)

Notes on dataset availability:
- Some real-world datasets in VALIDATION.md are not uniformly available via a
  stable public API. For those entries, this suite attempts a public loader when
  configured and falls back to synthetic profile-matched proxies when enabled.
- The fallback status is recorded per run row via `data_source`.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.io import loadmat
import scipy.stats as sps
from sklearn.datasets import fetch_openml, make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from tabnetics.core.compat import make_logistic_regression
except Exception as exc:
    from tabnetics.core.compat import make_logistic_regression  # type: ignore

try:
    from tabnetics.benchmarks.artifacts import create_timestamped_run_dir
    from tabnetics.distribution.selector import UnifiedDistributionSelectorV6
    from tabnetics.domains.face import load_face_proxy_dataset
except Exception as exc:
    from tabnetics.benchmarks.artifacts import create_timestamped_run_dir
    from tabnetics.distribution.selector import UnifiedDistributionSelectorV6
    from tabnetics.domains.face import load_face_proxy_dataset  # type: ignore


logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _retry_with_backoff(
    fn: Callable[[], _T],
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    label: str = "network_call",
) -> _T:
    """Execute *fn* with exponential-backoff retry on transient failures.

    Retries on ``OSError``, ``urllib.error.URLError``, ``TimeoutError``, and
    ``ConnectionError``.  HTTP 4xx errors (client errors) are NOT retried
    because they indicate a permanent problem (e.g. 404 Not Found).
    Other exceptions propagate immediately.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (OSError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            # Don't retry HTTP client errors (4xx) — they are permanent.
            _resp = getattr(exc, "response", None)
            if _resp is not None:
                _status = getattr(_resp, "status_code", None) or getattr(_resp, "status", None)
                if _status is not None and 400 <= int(_status) < 500:
                    raise
            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    "[%s] attempt %d/%d failed (%s); retrying in %.1fs…",
                    label, attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "[%s] all %d attempts failed; last error: %s",
                    label, max_retries + 1, exc,
                )
    assert last_exc is not None
    raise last_exc


FS_BASE_METHODS: List[str] = [
    "gradient_boosting",
    "linear_svm",
    "mutual_information",
    "anova_f",
]


try:
    from tabnetics.datasets.registry import DATASET_REGISTRY, DatasetSpec as ValidationDatasetSpec
except Exception as exc:
    from tabnetics.datasets.registry import DATASET_REGISTRY, DatasetSpec as ValidationDatasetSpec


@dataclass
class LoadedTabularDataset:
    X: np.ndarray
    y: np.ndarray
    data_source: str
    notes: str = ""


@dataclass
class DistributionCase:
    case_id: str
    data: np.ndarray
    true_family: Optional[str] = None
    true_params: Optional[Tuple[float, ...]] = None
    acceptable_families: Optional[Tuple[str, ...]] = None
    expect_rejection: bool = False
    notes: str = ""


@dataclass
class AblationConfig:
    name: str
    components: Dict[str, bool]

try:
    from tabnetics.core.errors import DatasetIntegrityPolicyError, DatasetIntegritySkipError
except Exception as exc:
    from tabnetics.core.errors import DatasetIntegrityPolicyError, DatasetIntegritySkipError


def _load_feature_selector_cls():
    try:
        from tabnetics.feature_selection import FeatureSelector
    except Exception as exc:
        from tabnetics.feature_selection import FeatureSelector
    return FeatureSelector


def _build_catalog() -> Dict[str, ValidationDatasetSpec]:
    # Validation-suite catalog is a view into the single source-of-truth registry.
    # Exclude synthetic benchmark-only datasets (these are generated in the benchmark runner).
    return {
        ds_id: spec
        for ds_id, spec in DATASET_REGISTRY.items()
        if str(getattr(spec, "source_kind", "")).strip().lower() != "synthetic"
    }


def _build_dataset_sets(catalog: Dict[str, ValidationDatasetSpec]) -> Dict[str, List[str]]:
    by_pipeline = {
        "fs": [k for k, v in catalog.items() if v.pipeline == "fs"],
        "df": [k for k, v in catalog.items() if v.pipeline == "df"],
        "integrated": [k for k, v in catalog.items() if v.pipeline == "integrated"],
    }

    sets: Dict[str, List[str]] = {
        "all": list(catalog.keys()),
        "smoke": ["leukemia_golub", "df_synthetic_parametric", "int_cdf_leukemia"],
        "fs_all": by_pipeline["fs"],
        "df_all": by_pipeline["df"],
        "integrated_all": by_pipeline["integrated"],
    }

    # "extended" includes all datasets; "core" excludes extended_only datasets.
    extended_only_ids = {
        k for k, v in catalog.items()
        if getattr(v, "extended_only", False)
    }
    sets["extended"] = list(catalog.keys())
    sets["core"] = [k for k in catalog if k not in extended_only_ids]

    for pipeline in ("fs", "df", "integrated"):
        for tier in ("easy", "medium", "hard", "very_hard"):
            key = f"{pipeline}_{tier}"
            sets[key] = [
                ds_id
                for ds_id, spec in catalog.items()
                if spec.pipeline == pipeline and spec.tier == tier
            ]

    return sets


# ---------------------------------------------------------------------------
# Meta-feature extractor (T-P3-INFRA-001)
# ---------------------------------------------------------------------------

def extract_meta_features(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Extract dataset meta-features useful for tier assignment / analysis.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector (class labels for classification).

    Returns
    -------
    dict with keys:
        n, p, p_over_n, class_count, class_balance_entropy,
        correlation_spectrum_decay, heaping_fraction
    """
    from scipy.stats import entropy as _shannon_entropy
    from scipy.optimize import curve_fit

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    n, p = X.shape if X.ndim == 2 else (X.shape[0], 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # --- class statistics ---
    classes, counts = np.unique(y, return_counts=True)
    class_count = float(len(classes))
    if class_count > 1:
        proportions = counts / counts.sum()
        raw_entropy = _shannon_entropy(proportions, base=np.e)
        class_balance_entropy = float(raw_entropy / np.log(class_count))
    else:
        class_balance_entropy = 0.0

    # --- p / n ratio ---
    p_over_n = float(p) / float(n) if n > 0 else 0.0

    # --- correlation spectrum decay ---
    if p >= 3 and n >= 2:
        rng = np.random.RandomState(42)
        max_cols = 200
        if p > max_cols:
            col_idx = rng.choice(p, max_cols, replace=False)
            X_sub = X[:, col_idx]
        else:
            X_sub = X
        # compute correlation matrix (handle constant columns)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(X_sub, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        # extract upper triangle absolute values
        tri_idx = np.triu_indices_from(corr, k=1)
        abs_corr = np.sort(np.abs(corr[tri_idx]))[::-1]
        # fit a * exp(-b * i)
        xs = np.arange(len(abs_corr), dtype=float)
        try:
            def _exp_decay(x, a, b):
                return a * np.exp(-b * x)
            popt, _ = curve_fit(
                _exp_decay, xs, abs_corr,
                p0=[abs_corr[0] if len(abs_corr) else 1.0, 0.01],
                maxfev=2000,
            )
            correlation_spectrum_decay = float(max(popt[1], 0.0))
        except Exception as exc:
            correlation_spectrum_decay = 0.0
    else:
        correlation_spectrum_decay = 0.0

    # --- heaping fraction ---
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


CATALOG = _build_catalog()
DATASET_SETS = _build_dataset_sets(CATALOG)


COMPONENT_DEFAULTS: Dict[str, bool] = {
    "fs.tritrust": True,
    "fs.stability_oracle": True,
    "fs.complexity_oracle": True,
    "fs.robust_oracle": True,
    "fs.diversity_oracle": False,
    "fs.method_mrmr_jmi": True,
    "fs.method_ktsp": True,
    "fs.method_stability_subsample": True,
    "df.robust_mode": True,
    "df.use_adaptive_strategy": True,
    "df.use_lrt": True,
    "df.use_cv": False,
    "df.rejection_gate": True,
    # Listed in VALIDATION.md protocol. `df.interval_likelihood` and
    # `df.contamination_em` remain placeholders until implemented upstream.
    "df.interval_likelihood": False,
    "df.contamination_em": False,
    "df.tritrust": False,
    "df.mnpo_aggregation": False,
    "integrated.cdf_transform": True,
    "integrated.low_gof_downweighting": False,
    "integrated.dist_stability_signal": False,
}


NOOP_COMPONENTS = {
    "df.contamination_em",
}


def _dataset_to_json(spec: ValidationDatasetSpec) -> Dict[str, Any]:
    data = asdict(spec)
    data["params"] = dict(spec.params)
    return data


def _dataset_promotion_metadata(spec: ValidationDatasetSpec) -> Dict[str, Any]:
    params = dict(spec.params or {})
    eligible = bool(params.get("promotion_eligible", True))
    blocker = str(params.get("promotion_blocker", "") or "").strip()
    source_policy = str(params.get("source_policy", "standard") or "standard").strip().lower()
    if eligible:
        blocker = ""
    return {
        "promotion_eligible": int(bool(eligible)),
        "promotion_blocker": blocker,
        "source_policy": source_policy,
    }


def _safe_label_encode(y_raw: np.ndarray) -> np.ndarray:
    if y_raw.dtype.kind in {"i", "u"}:
        return y_raw.astype(int)
    if y_raw.dtype.kind == "f":
        if np.all(np.isfinite(y_raw)):
            uniq = np.unique(y_raw)
            if np.allclose(uniq, np.round(uniq)):
                return y_raw.astype(int)
    return LabelEncoder().fit_transform(y_raw.astype(str))


def _class_diversity_summary(y: np.ndarray) -> Dict[str, Any]:
    y_arr = np.asarray(y).ravel()
    classes, counts = np.unique(y_arr, return_counts=True)
    return {
        "n_classes": int(classes.size),
        "min_class_count": int(counts.min()) if counts.size else 0,
        "class_counts": {str(cls): int(cnt) for cls, cnt in zip(classes.tolist(), counts.tolist())},
    }


def _check_class_diversity(
    y: np.ndarray,
    min_classes: int,
    min_class_count: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    stats = _class_diversity_summary(y)
    n_classes = int(stats["n_classes"])
    min_count = int(stats["min_class_count"])

    req_classes = int(max(1, min_classes))
    req_min_count = int(max(1, min_class_count))

    if n_classes < req_classes:
        reason = f"n_classes={n_classes}<required={req_classes}"
        return False, reason, stats
    if min_count < req_min_count:
        reason = f"min_class_count={min_count}<required={req_min_count}"
        return False, reason, stats
    return True, "", stats


def _enforce_loaded_dataset_integrity_policy(
    loaded: LoadedTabularDataset,
    *,
    spec: ValidationDatasetSpec,
    seed: int,
    source_policy: str,
    allow_synthetic_fallback: bool,
    sample_cap: int,
    feature_cap: int,
    class_integrity_policy: str,
    class_min_classes: int,
    class_min_class_count: int,
) -> LoadedTabularDataset:
    policy = str(class_integrity_policy).strip().lower()
    if policy not in {"fallback", "skip", "error"}:
        raise ValueError(f"Unknown class_integrity_policy: {class_integrity_policy}")

    source_policy_norm = str(source_policy).strip().lower() if source_policy is not None else "standard"

    ok, reason, stats = _check_class_diversity(
        loaded.y,
        min_classes=class_min_classes,
        min_class_count=class_min_class_count,
    )
    if ok:
        return loaded

    details = (
        f"dataset={spec.dataset_id} data_source={loaded.data_source} reason={reason} "
        f"class_counts={stats.get('class_counts', {})}"
    )

    if policy == "skip":
        raise DatasetIntegritySkipError(f"class_diversity_failed ({details})")
    if source_policy_norm == "real_only":
        raise DatasetIntegrityPolicyError(
            f"class_diversity_failed ({details}); source_policy=real_only forbids synthetic fallback"
        )
    if policy == "error":
        raise DatasetIntegrityPolicyError(f"class_diversity_failed ({details})")

    # policy == "fallback"
    if not allow_synthetic_fallback:
        raise DatasetIntegrityPolicyError(
            f"class_diversity_failed ({details}); synthetic fallback disabled"
        )

    fallback = _generate_synthetic_fs_dataset(
        spec,
        seed=seed,
        sample_cap=sample_cap,
        feature_cap=feature_cap,
        reason=f"class_diversity_failed:{reason}",
    )
    fb_ok, fb_reason, _ = _check_class_diversity(
        fallback.y,
        min_classes=class_min_classes,
        min_class_count=class_min_class_count,
    )
    if not fb_ok:
        raise DatasetIntegrityPolicyError(
            f"Synthetic fallback failed class-diversity checks: {fb_reason} (dataset={spec.dataset_id})"
        )

    original_notes = str(fallback.notes or "").strip()
    tag = f"integrity_fallback_from:{loaded.data_source}"
    fallback.notes = f"{original_notes}; {tag}" if original_notes else tag
    return fallback


def _stratified_subsample_indices(y: np.ndarray, n_keep: int, seed: int) -> np.ndarray:
    if n_keep <= 0 or n_keep >= y.shape[0]:
        return np.arange(y.shape[0], dtype=int)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=seed)
    (keep_idx, _) = next(splitter.split(np.zeros((y.shape[0], 1)), y))
    return np.asarray(keep_idx, dtype=int)


def _cap_sparse_features(X: sp.spmatrix, feature_cap: int) -> sp.spmatrix:
    if feature_cap <= 0 or X.shape[1] <= feature_cap:
        return X
    X_csc = X.tocsc(copy=False)
    nnz_per_col = np.diff(X_csc.indptr)
    # Keep the densest columns (most non-zeros) to preserve signal under cap.
    keep_cols = np.argsort(-nnz_per_col, kind="mergesort")[:feature_cap]
    keep_cols = np.sort(keep_cols)
    return X_csc[:, keep_cols].tocsr(copy=False)


def _cap_dense_features_by_variance(X: np.ndarray, feature_cap: int) -> np.ndarray:
    if feature_cap <= 0 or X.shape[1] <= feature_cap:
        return X
    var = np.nanvar(X, axis=0)
    if var.ndim != 1 or var.size != X.shape[1]:
        raise RuntimeError("Dense feature variance computation failed.")
    keep_cols = np.argsort(-var, kind="mergesort")[:feature_cap]
    keep_cols = np.sort(keep_cols)
    return X[:, keep_cols]


def _download_url_bytes(url: str, *, timeout_sec: int = 60, max_retries: int = 3) -> bytes:
    def _do_download() -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "tabnetics-validation/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = resp.read()
        if not payload:
            raise RuntimeError(f"Downloaded empty payload from {url}")
        return payload

    return _retry_with_backoff(_do_download, max_retries=max_retries, label=f"download:{url[:80]}")


def _extract_mat_array(mat_obj: Dict[str, Any], keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in mat_obj:
            return np.asarray(mat_obj[key])
    return None


def _load_mat_dataset_from_url_options(
    options: Sequence[Dict[str, Any]],
    *,
    seed: int,
    sample_cap: Optional[int] = None,
    feature_cap: Optional[int] = None,
) -> LoadedTabularDataset:
    failures: List[str] = []
    for option in options:
        try:
            url = str(option.get("url", "")).strip()
            if not url:
                raise RuntimeError("Missing `url` in mat_url_options entry.")

            payload = _download_url_bytes(url, timeout_sec=int(option.get("timeout_sec", 120)))
            mat_obj = loadmat(io.BytesIO(payload))

            x_key = str(option.get("x_key", "X")).strip()
            y_key = str(option.get("y_key", "Y")).strip()
            X_raw = _extract_mat_array(mat_obj, (x_key, "X", "x", "data", "Data"))
            y_raw = _extract_mat_array(mat_obj, (y_key, "Y", "y", "labels", "label", "target"))
            if X_raw is None or y_raw is None:
                raise RuntimeError(f"Unable to locate X/Y arrays in .mat payload (keys={list(mat_obj.keys())}).")

            X = np.asarray(X_raw, dtype=float)
            if X.ndim != 2:
                X = np.asarray(X).reshape(X.shape[0], -1)
            y = _safe_label_encode(np.asarray(y_raw).ravel())

            notes: List[str] = []
            if X.shape[0] != y.shape[0] and X.shape[1] == y.shape[0]:
                X = X.T
                notes.append("transposed_x_to_match_y")
            if X.shape[0] != y.shape[0]:
                raise RuntimeError(
                    f"X/y row mismatch after normalization: X.shape={X.shape}, y.shape={y.shape}."
                )

            eff_feature_cap = int(feature_cap or 0)
            if eff_feature_cap and X.shape[1] > eff_feature_cap:
                before = X.shape[1]
                X = _cap_dense_features_by_variance(X, eff_feature_cap)
                notes.append(f"dense_feature_cap:{before}->{X.shape[1]}")

            eff_sample_cap = int(sample_cap or 0)
            if eff_sample_cap and X.shape[0] > eff_sample_cap:
                keep = _stratified_subsample_indices(y, eff_sample_cap, seed)
                before = X.shape[0]
                X = X[keep]
                y = y[keep]
                notes.append(f"sample_cap:{before}->{X.shape[0]}")

            return LoadedTabularDataset(
                X=X.astype(float, copy=False),
                y=y,
                data_source=f"mat_url:{url}",
                notes="; ".join(notes),
            )
        except Exception as exc:
            failures.append(f"{option}: {exc}")

    raise RuntimeError("All MAT URL options failed. " + " | ".join(failures))


def _load_tab_url_dataset(
    options: Sequence[Dict[str, Any]],
    *,
    seed: int,
    sample_cap: Optional[int] = None,
    feature_cap: Optional[int] = None,
) -> LoadedTabularDataset:
    """Load a dataset from a biolab.si (or compatible) Orange .tab URL.

    The Orange .tab format is tab-delimited with three header rows:
      1. Column names (features + class column)
      2. Column types (continuous, discrete, string, etc.)
      3. Column roles (meta, class, ignore, or blank for feature)

    Data rows follow. The class column is identified by the "class" role
    in the third header row.
    """
    failures: List[str] = []
    for option in options:
        try:
            url = str(option.get("url", "")).strip()
            if not url:
                raise RuntimeError("Missing `url` in tab_url_options entry.")

            payload = _download_url_bytes(url, timeout_sec=int(option.get("timeout_sec", 120)))
            text = payload.decode("utf-8", errors="replace")
            lines = text.split("\n")

            # Parse the three header rows.
            if len(lines) < 4:
                raise RuntimeError(f"Tab file has fewer than 4 lines (need 3 headers + data): {len(lines)}")

            col_names = lines[0].rstrip("\r").split("\t")
            col_types = lines[1].rstrip("\r").split("\t")
            col_roles = lines[2].rstrip("\r").split("\t")

            # Pad roles to match col_names length (blank = feature).
            while len(col_roles) < len(col_names):
                col_roles.append("")

            # Identify class column and feature columns.
            class_col_idx: Optional[int] = None
            feature_col_indices: List[int] = []
            for i, role in enumerate(col_roles):
                role_clean = role.strip().lower()
                if role_clean == "class":
                    class_col_idx = i
                elif role_clean in ("", "feature"):
                    # Only keep continuous/numeric features.
                    ctype = col_types[i].strip().lower() if i < len(col_types) else ""
                    if ctype in ("continuous", "c", ""):
                        feature_col_indices.append(i)
                # Skip "meta", "ignore", "string", "weight" columns.

            if class_col_idx is None:
                raise RuntimeError(f"No 'class' column found in .tab header roles: {col_roles[:10]}")
            if not feature_col_indices:
                raise RuntimeError("No feature columns found in .tab file.")

            # Parse data rows.
            data_rows: List[List[str]] = []
            for line in lines[3:]:
                stripped = line.rstrip("\r").strip()
                if not stripped:
                    continue
                data_rows.append(stripped.split("\t"))

            if not data_rows:
                raise RuntimeError("No data rows found in .tab file.")

            n_cols = len(col_names)
            X_list: List[List[float]] = []
            y_list: List[str] = []
            for row in data_rows:
                if len(row) < n_cols:
                    row.extend([""] * (n_cols - len(row)))
                y_list.append(row[class_col_idx].strip())
                feat_vals: List[float] = []
                for fi in feature_col_indices:
                    val_str = row[fi].strip()
                    if val_str in ("", "?", "NA", "nan", "NaN"):
                        feat_vals.append(float("nan"))
                    else:
                        try:
                            feat_vals.append(float(val_str))
                        except ValueError:
                            feat_vals.append(float("nan"))
                X_list.append(feat_vals)

            X = np.array(X_list, dtype=float)
            y = _safe_label_encode(np.array(y_list))

            notes: List[str] = []

            # Handle NaN: replace with column median.
            nan_mask = np.isnan(X)
            if nan_mask.any():
                col_medians = np.nanmedian(X, axis=0)
                for ci in range(X.shape[1]):
                    col_nans = nan_mask[:, ci]
                    if col_nans.any():
                        X[col_nans, ci] = col_medians[ci]
                notes.append(f"nan_imputed_median:{int(nan_mask.sum())}_values")

            eff_feature_cap = int(feature_cap or 0)
            if eff_feature_cap and X.shape[1] > eff_feature_cap:
                before = X.shape[1]
                X = _cap_dense_features_by_variance(X, eff_feature_cap)
                notes.append(f"dense_feature_cap:{before}->{X.shape[1]}")

            eff_sample_cap = int(sample_cap or 0)
            if eff_sample_cap and X.shape[0] > eff_sample_cap:
                keep = _stratified_subsample_indices(y, eff_sample_cap, seed)
                before = X.shape[0]
                X = X[keep]
                y = y[keep]
                notes.append(f"sample_cap:{before}->{X.shape[0]}")

            return LoadedTabularDataset(
                X=X.astype(float, copy=False),
                y=y,
                data_source=f"tab_url:{url}",
                notes="; ".join(notes),
            )
        except Exception as exc:
            failures.append(f"{option}: {exc}")

    raise RuntimeError("All tab URL options failed. " + " | ".join(failures))


def _fetch_openml_dataset(
    options: Sequence[Dict[str, Any]],
    *,
    seed: int,
    sample_cap: Optional[int] = None,
    feature_cap: Optional[int] = None,
    max_retries: int = 3,
) -> LoadedTabularDataset:
    failures: List[str] = []
    for option in options:
        try:
            # Use as_frame='auto' to support:
            # - string attributes (DataFrame path)
            # - sparse ARFF datasets (scipy sparse matrix path)
            ds = _retry_with_backoff(
                lambda opt=option: fetch_openml(as_frame="auto", **opt),
                max_retries=max_retries,
                label=f"openml:{option}",
            )

            y_raw = ds.target
            if isinstance(y_raw, (pd.DataFrame, pd.Series)):
                y_arr = np.asarray(y_raw.to_numpy()).ravel()
            else:
                y_arr = np.asarray(y_raw).ravel()
            y = _safe_label_encode(y_arr)

            X_raw = ds.data
            notes: List[str] = []

            if sp.issparse(X_raw):
                X_sp = X_raw.tocsr(copy=False)

                eff_feature_cap = int(feature_cap or 0)
                X_sp = _cap_sparse_features(X_sp, eff_feature_cap)
                if eff_feature_cap and X_raw.shape[1] > eff_feature_cap:
                    notes.append(f"sparse_feature_cap:{X_raw.shape[1]}->{X_sp.shape[1]}")

                eff_sample_cap = int(sample_cap or 0)
                if eff_sample_cap and X_sp.shape[0] > eff_sample_cap:
                    keep = _stratified_subsample_indices(y, eff_sample_cap, seed)
                    X_sp = X_sp[keep]
                    y = y[keep]
                    notes.append(f"sample_cap:{X_raw.shape[0]}->{X_sp.shape[0]}")

                X = X_sp.toarray().astype(float, copy=False)
                notes.append("sparse_densified")
            else:
                if isinstance(X_raw, pd.DataFrame):
                    X_df = X_raw.apply(pd.to_numeric, errors="coerce")
                    keep_cols = [
                        c for c in X_df.columns if np.isfinite(X_df[c].to_numpy(dtype=float)).any()
                    ]
                    if keep_cols and len(keep_cols) != X_df.shape[1]:
                        notes.append(f"dropped_all_nan_cols:{X_df.shape[1]-len(keep_cols)}")
                    X = X_df[keep_cols].to_numpy(dtype=float)
                else:
                    X = np.asarray(X_raw)
                    if not np.issubdtype(X.dtype, np.number):
                        X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                    else:
                        X = X.astype(float, copy=False)

            return LoadedTabularDataset(
                X=X,
                y=y,
                data_source=f"openml:{option}",
                notes="; ".join(notes),
            )
        except Exception as exc:
            failures.append(f"{option}: {exc}")
    raise RuntimeError("All OpenML options failed. " + " | ".join(failures))


def _load_nci60_from_url(*, max_retries: int = 3) -> LoadedTabularDataset:
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/NCI60.csv"
    df = _retry_with_backoff(
        lambda: pd.read_csv(url),
        max_retries=max_retries,
        label="nci60:rdatasets",
    )

    if df.shape[1] < 3:
        raise RuntimeError(f"Unexpected NCI60 shape: {df.shape}")

    y_raw = df.iloc[:, -1].to_numpy()
    first_col = df.iloc[:, 0]
    first_col_is_index = False
    try:
        if (
            (first_col.dtype == "int64" and first_col.nunique() == len(first_col))
            or (isinstance(first_col.iloc[0], str) and first_col.nunique() == len(first_col))
        ):
            first_col_is_index = True
    except Exception as exc:
        first_col_is_index = False

    start_col = 1 if first_col_is_index else 0
    X_df = df.iloc[:, start_col:-1]
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    keep_cols = [c for c in X_df.columns if np.isfinite(X_df[c].to_numpy(dtype=float)).any()]
    if not keep_cols:
        raise RuntimeError("NCI60 numeric feature extraction failed.")

    X = X_df[keep_cols].to_numpy(dtype=float)
    y = _safe_label_encode(np.asarray(y_raw))
    return LoadedTabularDataset(X=X, y=y, data_source="rdatasets:ISLR:NCI60")


def _load_nci60_8class_proxy_dataset(*, target_features: int = 5244, max_retries: int = 3) -> LoadedTabularDataset:
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/NCI60.csv"
    df = _retry_with_backoff(
        lambda: pd.read_csv(url),
        max_retries=max_retries,
        label="nci60_proxy:rdatasets",
    )
    if df.shape[1] < 3:
        raise RuntimeError(f"Unexpected NCI60 shape: {df.shape}")

    y_raw = df.iloc[:, -1].astype(str).str.strip()
    first_col = df.iloc[:, 0]
    first_col_is_index = False
    try:
        if (
            (first_col.dtype == "int64" and first_col.nunique() == len(first_col))
            or (isinstance(first_col.iloc[0], str) and first_col.nunique() == len(first_col))
        ):
            first_col_is_index = True
    except Exception as exc:
        first_col_is_index = False

    start_col = 1 if first_col_is_index else 0
    X_df = df.iloc[:, start_col:-1].apply(pd.to_numeric, errors="coerce")
    keep_cols = [c for c in X_df.columns if np.isfinite(X_df[c].to_numpy(dtype=float)).any()]
    if not keep_cols:
        raise RuntimeError("NCI60 numeric feature extraction failed.")
    X = X_df[keep_cols].to_numpy(dtype=float)

    y_norm = y_raw.replace(
        {
            "K562A-repro": "LEUKEMIA",
            "K562B-repro": "LEUKEMIA",
            "MCF7A-repro": "BREAST",
            "MCF7D-repro": "BREAST",
        }
    )
    drop_labels = {"UNKNOWN", "PROSTATE"}
    keep_mask = ~y_norm.isin(drop_labels)
    X = X[keep_mask.to_numpy()]
    y_norm = y_norm[keep_mask]

    expected_labels = {"BREAST", "CNS", "COLON", "LEUKEMIA", "MELANOMA", "NSCLC", "OVARIAN", "RENAL"}
    observed_labels = set(np.unique(y_norm.to_numpy(dtype=object)).tolist())
    if observed_labels != expected_labels:
        raise RuntimeError(
            f"NCI60 proxy class-set mismatch: expected={sorted(expected_labels)}, observed={sorted(observed_labels)}"
        )
    if X.shape[0] != 61:
        raise RuntimeError(f"NCI60 proxy sample count mismatch: expected 61, got {X.shape[0]}")

    target_features = int(max(8, target_features))
    if X.shape[1] < target_features:
        raise RuntimeError(f"NCI60 proxy feature count too small: {X.shape[1]} < requested {target_features}")
    if X.shape[1] > target_features:
        X = _cap_dense_features_by_variance(X, target_features)

    y = _safe_label_encode(y_norm.to_numpy(dtype=object))
    notes = (
        "nci60_proxy_transform;"
        " merged_labels=[K562A-repro,K562B-repro->LEUKEMIA; MCF7A-repro,MCF7D-repro->BREAST];"
        " dropped_labels=[UNKNOWN,PROSTATE];"
        f" top_variance_features={target_features}"
    )
    return LoadedTabularDataset(
        X=X.astype(float, copy=False),
        y=y,
        data_source="rdatasets:ISLR:NCI60:proxy_61x5244_8class",
        notes=notes,
    )


def _balanced_weights(n_classes: int) -> List[float]:
    if n_classes <= 0:
        return [1.0]
    return [1.0 / n_classes] * n_classes


def _normalize_weights(weights: Sequence[float], n_classes: int) -> List[float]:
    vals = np.asarray(weights, dtype=float)
    if vals.size != n_classes:
        return _balanced_weights(n_classes)
    vals = np.clip(vals, 1e-6, np.inf)
    vals = vals / vals.sum()
    return vals.tolist()


def _generate_synthetic_fs_dataset(
    spec: ValidationDatasetSpec,
    seed: int,
    sample_cap: int,
    feature_cap: int,
    reason: str,
) -> LoadedTabularDataset:
    profile = dict(spec.params.get("synthetic_profile", {}))

    n_samples_orig = int(profile.get("n_samples", 120))
    n_features_orig = int(profile.get("n_features", 2000))

    n_samples = int(max(24, min(sample_cap, n_samples_orig)))
    n_features = int(max(40, min(feature_cap, n_features_orig)))
    n_classes = int(max(2, profile.get("n_classes", 2)))

    difficulty = str(profile.get("difficulty", spec.tier)).lower()
    class_sep_map = {
        "easy": 2.0,
        "medium": 1.25,
        "hard": 0.9,
        "very_hard": 0.65,
    }
    noise_map = {
        "easy": 0.01,
        "medium": 0.03,
        "hard": 0.05,
        "very_hard": 0.08,
    }

    class_sep = class_sep_map.get(difficulty, 1.1)
    flip_y = noise_map.get(difficulty, 0.04)

    weights = _normalize_weights(profile.get("weights", _balanced_weights(n_classes)), n_classes)

    n_informative = max(8, min(n_features // 3, n_classes * 8 + 8))
    n_redundant = max(4, min(n_features - n_informative - 1, n_informative // 2))
    if n_redundant < 0:
        n_redundant = 0

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=weights,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=seed,
    )

    # Keep high-dimensional features numerically stable and mildly heavy-tailed.
    if spec.tier in {"hard", "very_hard"}:
        rng = np.random.default_rng(seed + 19)
        heavy_mask = rng.random(X.shape) < 0.03
        X = X.copy()
        X[heavy_mask] += rng.standard_t(df=3, size=int(heavy_mask.sum()))

    notes = (
        f"Synthetic fallback ({reason}); target profile n={n_samples_orig}, p={n_features_orig}, "
        f"c={n_classes}; capped to n={n_samples}, p={n_features}."
    )
    return LoadedTabularDataset(
        X=X.astype(float),
        y=y.astype(int),
        data_source="synthetic_fallback",
        notes=notes,
    )


def _load_hf_parquet_dataset(
    dataset_id: str,
    hf_org: str = "tabnetics",
    *,
    bundle_repo_name: str = "tabnetics-validation",
    repo_id: Optional[str] = None,
) -> LoadedTabularDataset:
    """Load a dataset from the HuggingFace validation bundle repo.

    Expects a single HF dataset repo ``<hf_org>/<bundle_repo_name>``
    with config-per-dataset_id structure. Each config exposes a 'train' split
    with columns: ``features`` (list[float32]) and ``label`` (int64).

    This loader uses the authoritative HF bundle and RAISES when the dataset/config
    is missing. No fallback to OpenML/URL/local or synthetic is performed. 
    The calling code should only invoke this loader when HF_TOKEN + TABNETICS_HF_ORG 
    are configured and the intention is to use HF as the single source of truth.

    Requires the ``datasets`` package and ``HF_TOKEN`` env var for private repos.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "datasets library not installed; run `pip install datasets` "
            "to enable HF bundle dataset loading."
        )

    repo_id = str(repo_id or "").strip() or f"{hf_org}/{bundle_repo_name}"
    
    try:
        ds_dict = load_dataset(repo_id, name=dataset_id, split="train")
    except Exception as exc:
        # Fail fast — no fallback to other loaders or synthetic
        raise RuntimeError(
            f"Failed to load dataset '{dataset_id}' from HF bundle repo '{repo_id}': {exc}. "
            f"Ensure the dataset exists as a config in the HF repo and HF_TOKEN is valid. "
            f"No fallback to OpenML/URL/local or synthetic is performed."
        ) from exc

    # Extract features and label from the HF bundle format
    if "features" not in ds_dict.features or "label" not in ds_dict.features:
        raise RuntimeError(
            f"HF bundle dataset '{dataset_id}' missing expected columns 'features' and 'label'. "
            f"Found columns: {list(ds_dict.features.keys())}"
        )

    # Convert to numpy arrays
    # features is list[float32] per row, label is int64
    import numpy as np
    X = np.array(ds_dict["features"], dtype=np.float32)
    y = np.array(ds_dict["label"], dtype=np.int64)

    # Label encode to ensure 0-based contiguous labels
    y = _safe_label_encode(y)

    return LoadedTabularDataset(
        X=X,
        y=y,
        data_source=f"hf_bundle:{repo_id}/{dataset_id}",
        notes=f"loaded_from_hf_bundle:config={dataset_id},split=train",
    )


_HF_AUTHORITATIVE_SKIP_KINDS = {"synthetic_only", "dist_benchmark"}


def _hf_bundle_is_configured() -> bool:
    import os as _os

    return bool(_os.environ.get("TABNETICS_HF_ORG", "").strip() or _os.environ.get("TABNETICS_HF_REPO_ID", "").strip())


def _require_hf_bundle_configuration(*, dataset_id: str, loader_kind: str) -> None:
    if loader_kind in _HF_AUTHORITATIVE_SKIP_KINDS:
        return
    if _hf_bundle_is_configured():
        return
    raise RuntimeError(
        f"dataset={dataset_id} requires authoritative HuggingFace bundle loading for validation/testing runs. "
        "Set TABNETICS_HF_ORG or TABNETICS_HF_REPO_ID and ensure the dataset exists in the configured HF bundle."
    )


def _load_manual_tabular_dataset(path: Path) -> LoadedTabularDataset:
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".arff":
        # Git LFS pointer stubs can appear in bundle clones. Fail fast with a
        # clear message so callers can fall back (or fetch LFS objects).
        if _is_git_lfs_pointer_file(path):
            raise RuntimeError(
                f"ARFF path looks like a Git LFS pointer file (not real data): {path}. "
                "Fetch LFS objects (git-lfs) or enable synthetic fallback."
            )
        return _load_dense_arff_dataset(path)
    if suffix == ".csv":
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise RuntimeError("CSV must include at least one feature column and one target column.")
        first_col = df.iloc[:, 0]
        first_col_name = str(df.columns[0]).strip().lower()
        first_col_is_index = False
        try:
            if first_col_name in {"", "index"} or first_col_name.startswith("unnamed"):
                if first_col.nunique(dropna=False) == len(first_col):
                    first_col_is_index = True
            elif first_col.dtype.kind in {"i", "u"} and first_col.nunique(dropna=False) == len(first_col):
                vals = first_col.to_numpy()
                if vals.size:
                    min_val = int(np.nanmin(vals))
                    max_val = int(np.nanmax(vals))
                    if min_val in {0, 1} and max_val - min_val == len(vals) - 1:
                        first_col_is_index = True
        except Exception as exc:
            first_col_is_index = False

        start_col = 1 if first_col_is_index else 0
        X_df = df.iloc[:, start_col:-1].apply(pd.to_numeric, errors="coerce")
        keep_cols = [c for c in X_df.columns if np.isfinite(X_df[c].to_numpy(dtype=float)).any()]
        if not keep_cols:
            raise RuntimeError("CSV numeric feature extraction failed (all columns NaN).")
        X = X_df[keep_cols].to_numpy(dtype=float)
        y = _safe_label_encode(df.iloc[:, -1].to_numpy())
        return LoadedTabularDataset(X=X, y=y, data_source=f"manual_csv:{path}")

    if suffix in {".npz", ".npy"}:
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.lib.npyio.NpzFile):
            if "X" not in obj or "y" not in obj:
                raise RuntimeError("NPZ file must contain `X` and `y` arrays.")
            X = np.asarray(obj["X"], dtype=float)
            y = _safe_label_encode(np.asarray(obj["y"]))
        else:
            raise RuntimeError("NPY is unsupported for manual tabular loading; use NPZ with X/y.")
        return LoadedTabularDataset(X=X, y=y, data_source=f"manual_npz:{path}")

    raise RuntimeError(f"Unsupported manual dataset format: {path.suffix}")


def _is_git_lfs_pointer_file(path: Path) -> bool:
    """Return True when `path` appears to be a Git LFS pointer stub, not real data."""
    try:
        size = int(path.stat().st_size)
    except Exception as exc:
        return False
    # Pointer files are tiny; real ARFF/CSV datasets are typically much larger.
    if size <= 0 or size > 8192:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:512]
    except Exception as exc:
        return False
    head = head.strip()
    return bool(head.startswith("version https://git-lfs.github.com/spec/v1") and "oid sha256:" in head)


def _load_dense_arff_dataset(path: Path) -> LoadedTabularDataset:
    """Load a dense ARFF dataset with numeric features and a final class label.

    Notes:
    - We intentionally implement a lightweight parser here rather than relying
      on `scipy.io.arff.loadarff`, because some CuMiDa ARFF exports contain
      malformed/truncated rows that SciPy rejects (IndexError).
    - The parser assumes the target is the final attribute named "class"
      (case-insensitive). This matches the CuMiDa files used in this repo.
    """

    # First pass: count attributes and locate the class attribute.
    attr_names: List[str] = []
    class_labels_declared: Optional[Tuple[str, ...]] = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if s.upper().startswith("@ATTRIBUTE"):
                rest = s[len("@ATTRIBUTE") :].strip()
                if not rest:
                    continue
                # ARFF attribute names can be quoted or bare tokens.
                if rest[0] in {"'", '"'}:
                    quote = rest[0]
                    end = rest.find(quote, 1)
                    if end <= 0:
                        continue
                    name = rest[1:end]
                    attr_type = rest[end + 1 :].strip()
                else:
                    parts = rest.split(None, 1)
                    name = parts[0].strip().strip("'\"")
                    attr_type = parts[1].strip() if len(parts) > 1 else ""

                attr_names.append(name)
                if name.lower() == "class":
                    attr_type_clean = attr_type.strip()
                    if attr_type_clean.startswith("{") and attr_type_clean.endswith("}"):
                        raw_vals = attr_type_clean[1:-1]
                        vals = [v.strip().strip("'\"") for v in raw_vals.split(",")]
                        vals = [v for v in vals if v]
                        if vals:
                            class_labels_declared = tuple(vals)
                continue
            if s.upper() == "@DATA":
                break

    if not attr_names:
        raise RuntimeError(f"ARFF parse failed (no attributes): {path}")

    n_attr = int(len(attr_names))
    # Prefer an explicitly named "class" attribute; otherwise assume last column.
    target_idx = next((i for i in range(n_attr - 1, -1, -1) if attr_names[i].lower() == "class"), n_attr - 1)
    if target_idx != n_attr - 1:
        raise RuntimeError(f"ARFF loader expects class attribute last (found at index {target_idx}): {path}")

    n_features = int(n_attr - 1)
    declared_exact = set(class_labels_declared or ())
    declared_folded = {v.casefold() for v in declared_exact}

    X_rows: List[np.ndarray] = []
    y_rows: List[str] = []
    bad_rows = 0
    unknown_class_rows = 0

    # Second pass: parse data rows.
    with path.open("r", encoding="utf-8", errors="replace") as f:
        in_data = False
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if not in_data:
                if s.upper() == "@DATA":
                    in_data = True
                continue

            try:
                row_tokens = next(csv.reader([s], skipinitialspace=True))
            except Exception as exc:
                bad_rows += 1
                continue

            if len(row_tokens) != n_attr:
                bad_rows += 1
                continue

            numeric_str = ",".join(tok.strip() for tok in row_tokens[:n_features])
            # ARFF missing values are represented as '?'.
            if "?" in numeric_str:
                numeric_str = numeric_str.replace("?", "nan")
            vals = np.fromstring(numeric_str, sep=",", dtype=float)
            if vals.size != n_features:
                bad_rows += 1
                continue

            label = row_tokens[target_idx].strip().strip("'\"")
            if declared_exact and label not in declared_exact and label.casefold() not in declared_folded:
                unknown_class_rows += 1
                continue

            X_rows.append(vals)
            y_rows.append(label)

    if not X_rows:
        raise RuntimeError(f"ARFF parse failed (no valid rows): {path}")

    X = np.vstack(X_rows).astype(float, copy=False)
    y = _safe_label_encode(np.asarray(y_rows, dtype=object))
    note_parts: List[str] = []
    if bad_rows:
        note_parts.append(f"bad_rows_dropped:{bad_rows}")
    if unknown_class_rows:
        note_parts.append(f"unknown_class_rows_dropped:{unknown_class_rows}")
    notes = "; ".join(note_parts)
    return LoadedTabularDataset(X=X, y=y, data_source=f"manual_arff:{path}", notes=notes)


def load_feature_selection_dataset(
    spec: ValidationDatasetSpec,
    seed: int,
    allow_synthetic_fallback: bool,
    sample_cap: int,
    feature_cap: int,
    source_policy: Optional[str] = None,
    class_integrity_policy: str = "error",
    class_min_classes: int = 2,
    class_min_class_count: int = 1,
    require_hf_source: bool = False,
) -> LoadedTabularDataset:
    if spec.pipeline not in {"fs", "integrated"}:
        raise ValueError(f"Unsupported pipeline for FS loader: {spec.pipeline}")

    source_policy = str(source_policy).strip().lower() if source_policy is not None else str(
        spec.params.get("source_policy", "standard")
    ).strip().lower()
    if source_policy not in {"standard", "real_only", "fallback_only"}:
        raise ValueError(f"Unknown source_policy: {source_policy}")
    force_fallback_only = source_policy == "fallback_only"
    kind = spec.loader_kind
    if require_hf_source and allow_synthetic_fallback:
        raise DatasetIntegrityPolicyError(
            f"dataset={spec.dataset_id} requires HF-authoritative loading; synthetic fallback is forbidden"
        )
    if require_hf_source:
        _require_hf_bundle_configuration(dataset_id=str(spec.dataset_id), loader_kind=str(kind))
    try:
        loaded: Optional[LoadedTabularDataset] = None

        # --- HF bundle: AUTHORITATIVE source when configured (fail-fast, no fallback) ---
        # When TABNETICS_HF_ORG is set, the HF validation bundle is the single source
        # of truth for all real datasets. No fallback to OpenML/URL/local or synthetic
        # is performed. This ensures data integrity and reproducibility.
        import os as _os
        hf_org = _os.environ.get("TABNETICS_HF_ORG", "").strip()
        hf_repo_name = _os.environ.get("TABNETICS_HF_REPO_NAME", "").strip() or "tabnetics-validation"
        hf_repo_id = _os.environ.get("TABNETICS_HF_REPO_ID", "").strip()
        if (hf_org or hf_repo_id) and kind not in _HF_AUTHORITATIVE_SKIP_KINDS:
            # HF is configured — use bundle repo as authoritative source
            # This will RAISE if the dataset/config is missing (no fallback)
            loaded = _load_hf_parquet_dataset(
                spec.dataset_id,
                hf_org,
                bundle_repo_name=hf_repo_name,
                repo_id=hf_repo_id or None,
            )
            resolved_repo = hf_repo_id or f"{hf_org}/{hf_repo_name}"
            logger.info("Loaded %s from HF bundle (%s, config=%s)", spec.dataset_id, resolved_repo, spec.dataset_id)
        elif loaded is None and kind == "openml_or_synth":
            options = spec.params.get("openml_options", [])
            if options:
                loaded = _fetch_openml_dataset(
                    options,
                    seed=seed,
                    sample_cap=spec.params.get("openml_sample_cap"),
                    feature_cap=spec.params.get("openml_feature_cap"),
                )
            else:
                raise RuntimeError("No OpenML options configured.")

        elif kind == "mat_url_or_synth":
            options = spec.params.get("mat_url_options", [])
            if not options:
                raise RuntimeError("No MAT URL options configured.")
            loaded = _load_mat_dataset_from_url_options(
                options,
                seed=seed,
                sample_cap=spec.params.get("mat_sample_cap"),
                feature_cap=spec.params.get("mat_feature_cap"),
            )

        elif kind == "tab_url_or_synth":
            options = spec.params.get("tab_url_options", [])
            if not options:
                raise RuntimeError("No tab URL options configured.")
            loaded = _load_tab_url_dataset(
                options,
                seed=seed,
                sample_cap=spec.params.get("tab_sample_cap"),
                feature_cap=spec.params.get("tab_feature_cap"),
            )

        elif kind == "nci60_url_or_synth":
            loaded = _load_nci60_from_url()

        elif kind == "nci60_proxy_or_synth":
            loaded = _load_nci60_8class_proxy_dataset(
                target_features=int(spec.params.get("proxy_target_features", 5244))
            )

        elif kind == "face_proxy_or_synth":
            face_proxy = load_face_proxy_dataset(spec, seed=seed)
            loaded = LoadedTabularDataset(
                X=np.asarray(face_proxy.X, dtype=float),
                y=np.asarray(face_proxy.y),
                data_source=str(face_proxy.data_source),
                notes=str(face_proxy.notes or ""),
            )

        elif kind == "manual_or_synth":
            if bool(spec.params.get("force_synthetic_fallback", False)):
                if source_policy == "real_only":
                    raise DatasetIntegrityPolicyError(
                        f"dataset={spec.dataset_id} source_policy=real_only; synthetic fallback disabled"
                    )
                if not allow_synthetic_fallback:
                    raise DatasetIntegrityPolicyError(
                        f"dataset={spec.dataset_id} requires synthetic fallback "
                        "(source_policy=fallback_only); synthetic fallback disabled"
                    )
                loaded = _generate_synthetic_fs_dataset(
                    spec,
                    seed=seed,
                    sample_cap=sample_cap,
                    feature_cap=feature_cap,
                    reason="policy_forced_fallback_only",
                )
                original_notes = str(loaded.notes or "").strip()
                policy_tag = "source_policy:fallback_only"
                loaded.notes = f"{original_notes}; {policy_tag}" if original_notes else policy_tag
            else:
                env_name = spec.params.get("local_path_env")
                if env_name:
                    import os

                    env_raw = os.environ.get(str(env_name), "").strip()
                    if env_raw:
                        env_value = Path(env_raw)
                        loaded = _load_manual_tabular_dataset(env_value)
                # Optional: repo-local fallback for convenience in CI/dev (e.g. CuMiDa).
                if loaded is None:
                    repo_root = Path(__file__).resolve().parents[1]
                    default_candidates: List[str] = []
                    default_one = spec.params.get("default_local_path")
                    if default_one:
                        default_candidates.append(str(default_one))
                    for extra in list(spec.params.get("default_local_paths", []) or ()):
                        default_candidates.append(str(extra))

                    for cand in default_candidates:
                        p = Path(cand)
                        if not p.is_absolute():
                            p = repo_root / p
                        if p.exists():
                            loaded = _load_manual_tabular_dataset(p)
                            break

                if loaded is None:
                    if source_policy == "real_only":
                        raise DatasetIntegrityPolicyError(
                            f"dataset={spec.dataset_id} source_policy=real_only; no real data path available"
                        )
                    if force_fallback_only and not allow_synthetic_fallback:
                        raise DatasetIntegrityPolicyError(
                            f"dataset={spec.dataset_id} source_policy=fallback_only; synthetic fallback disabled"
                        )
                    raise RuntimeError(
                        f"No data source for dataset={spec.dataset_id}. "
                        f"Set TABNETICS_HF_ORG and upload via migrate_datasets_to_hf.py, "
                        f"or configure local path via env var or default_local_path."
                    )

        elif kind == "synthetic_only":
            loaded = _generate_synthetic_fs_dataset(
                spec, seed=seed, sample_cap=sample_cap, feature_cap=feature_cap, reason="synthetic_only"
            )

        elif kind == "integrated":
            base_id = str(spec.params["base_dataset"])
            base_spec = CATALOG[base_id]
            loaded = load_feature_selection_dataset(
                base_spec,
                seed=seed,
                allow_synthetic_fallback=allow_synthetic_fallback,
                sample_cap=sample_cap,
                feature_cap=feature_cap,
                source_policy=source_policy,
                class_integrity_policy=class_integrity_policy,
                class_min_classes=class_min_classes,
                class_min_class_count=class_min_class_count,
                require_hf_source=require_hf_source,
            )

        else:
            raise ValueError(f"Unknown loader kind: {kind}")

        if loaded is None:
            raise RuntimeError(f"Loader returned no dataset for kind={kind}")
        if require_hf_source and kind not in _HF_AUTHORITATIVE_SKIP_KINDS:
            resolved_source = str(getattr(loaded, "data_source", "") or "")
            if not resolved_source.startswith("hf_bundle:"):
                raise RuntimeError(
                    f"dataset={spec.dataset_id} loaded from non-HF source '{resolved_source}' "
                    "while HF-authoritative loading is required"
                )

        return _enforce_loaded_dataset_integrity_policy(
            loaded,
            spec=spec,
            seed=seed,
            source_policy=source_policy,
            allow_synthetic_fallback=allow_synthetic_fallback,
            sample_cap=sample_cap,
            feature_cap=feature_cap,
            class_integrity_policy=class_integrity_policy,
            class_min_classes=class_min_classes,
            class_min_class_count=class_min_class_count,
        )

    except (DatasetIntegritySkipError, DatasetIntegrityPolicyError):
        raise
    except Exception as exc:
        if require_hf_source:
            raise
        if source_policy == "real_only":
            raise
        if not allow_synthetic_fallback:
            raise
        if force_fallback_only:
            return _generate_synthetic_fs_dataset(
                spec,
                seed=seed,
                sample_cap=sample_cap,
                feature_cap=feature_cap,
                reason="policy_forced_fallback_only",
            )
        return _generate_synthetic_fs_dataset(
            spec,
            seed=seed,
            sample_cap=sample_cap,
            feature_cap=feature_cap,
            reason=f"loader_failed:{exc}",
        )


def _safe_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
    max_train_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    forced_train_n: Optional[int] = None
    n_total = int(X.shape[0])
    if max_train_samples is not None and int(max_train_samples) > 0 and n_total >= 4:
        train_n = int(max(2, int(max_train_samples)))
        # Cap-mode must still preserve a minimum 80/20 split.
        min_test_n = int(max(2, np.ceil(0.20 * float(n_total))))
        train_n = int(min(train_n, n_total - min_test_n))
        forced_train_n = int(train_n)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) >= 2 and counts.min() >= 2:
        try:
            if forced_train_n is not None:
                return train_test_split(X, y, train_size=int(forced_train_n), stratify=y, random_state=seed)
            return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
        except ValueError:
            # Can fail when train/test quotas are too tight for class counts.
            pass
    if forced_train_n is not None:
        return train_test_split(X, y, train_size=int(forced_train_n), random_state=seed)
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def _select_model_via_cv(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    models = {
        "lr": make_logistic_regression(
            random_state=seed,
            max_iter=3000,
            solver="lbfgs",
            penalty="l2",
            class_weight="balanced",
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            random_state=seed,
        ),
    }

    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return models["lr"], "lr"

    n_splits = int(max(2, min(5, counts.min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    best_name = "lr"
    best_score = -np.inf
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="balanced_accuracy")
            score = float(np.mean(scores))
            if score > best_score:
                best_name = name
                best_score = score
        except Exception as exc:
            continue
    return models[best_name], best_name


def _select_fs_subsample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fs_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    fs_fraction = float(max(0.05, min(1.0, fs_fraction)))
    if fs_fraction >= 0.999:
        return X_train, y_train

    classes, counts = np.unique(y_train, return_counts=True)
    n_samples = X_train.shape[0]
    n_fs = int(max(2, round(fs_fraction * n_samples)))

    if len(classes) < 2 or counts.min() < 2:
        rng = np.random.default_rng(seed)
        fs_idx = rng.choice(np.arange(n_samples), size=n_fs, replace=False)
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=fs_fraction, random_state=seed)
        fs_idx, _ = next(splitter.split(X_train, y_train))

    return X_train[fs_idx], y_train[fs_idx]


def _build_fs_enabled_methods(components: Dict[str, bool]) -> List[str]:
    methods = list(FS_BASE_METHODS)
    if components.get("fs.method_mrmr_jmi", False):
        methods.append("mrmr_jmi")
    if components.get("fs.method_ktsp", False):
        methods.append("ktsp")
    if components.get("fs.method_stability_subsample", False):
        methods.append("stability_subsample")
    # Preserve order + unique
    seen = set()
    uniq = []
    for m in methods:
        if m not in seen:
            uniq.append(m)
            seen.add(m)
    return uniq


def _run_fs_core(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    components: Dict[str, bool],
    fs_fraction: float,
    n_final_features: int,
    seed: int,
    already_preprocessed: bool = False,
) -> Dict[str, Any]:
    FeatureSelector = _load_feature_selector_cls()

    if already_preprocessed:
        X_train_proc = X_train
        X_test_proc = X_test
    else:
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train_imp)
        X_test_proc = scaler.transform(X_test_imp)

    X_fs, y_fs = _select_fs_subsample(X_train_proc, y_train, fs_fraction=fs_fraction, seed=seed)

    selector = FeatureSelector(
        n_bootstrap_iterations=3,
        random_state=seed,
        problem_type="classification",
        selection_strategy="mnpo_portfolio",
        inner_cv_splits=3,
        inner_cv_repeats=1,
        mirror_descent_steps=100,
        portfolio_size=5,
        enabled_methods=_build_fs_enabled_methods(components),
        use_tritrust=bool(components.get("fs.tritrust", True)),
        use_stability_oracle=bool(components.get("fs.stability_oracle", True)),
        use_complexity_oracle=bool(components.get("fs.complexity_oracle", True)),
        use_robust_oracle=bool(components.get("fs.robust_oracle", True)),
        use_diversity_oracle=bool(components.get("fs.diversity_oracle", False)),
        ktsp_k_pairs=16,
        mrmr_redundancy_weight=0.55,
        stability_selection_threshold=0.6,
        stability_subsample_fraction=0.5,
    )

    t0 = time.perf_counter()
    selector.fit_transform(X_fs, y_fs, n_final_features=n_final_features, return_result_object=True)
    fs_time = time.perf_counter() - t0

    X_train_sel = selector.transform(X_train_proc)
    X_test_sel = selector.transform(X_test_proc)

    model, model_name = _select_model_via_cv(X_train_sel, y_train, seed)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)

    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "hybrid_score": float(0.6 * bal_acc + 0.4 * macro_f1),
        "selected_features": int(X_train_sel.shape[1]),
        "model": model_name,
        "fs_time_sec": float(fs_time),
    }


def _selector_for_distribution(components: Dict[str, bool], *, criterion: str = "simple") -> UnifiedDistributionSelectorV6:
    crit = str(criterion).strip().lower()
    return UnifiedDistributionSelectorV6(
        robust_mode=bool(components.get("df.robust_mode", True)),
        use_adaptive_strategy=bool(components.get("df.use_adaptive_strategy", True)),
        use_lrt=bool(components.get("df.use_lrt", True)),
        use_cv=bool(components.get("df.use_cv", False)),
        n_jobs=1,
        compute_crps=bool(crit == "crps"),
        interval_likelihood=bool(components.get("df.interval_likelihood", False)),
        mnpo_use_tritrust=bool(components.get("df.tritrust", False)),
    )


def _maybe_reject(
    best_name: Optional[str],
    best_result: Any,
    components: Dict[str, bool],
    threshold: float,
) -> bool:
    if not components.get("df.rejection_gate", True):
        return False
    if best_name is None or best_result is None:
        return True

    cvm_p = getattr(best_result, "cvm_p", None)
    ks_p = getattr(best_result, "ks_p", None)

    if cvm_p is None and ks_p is None:
        return False

    cvm_bad = (cvm_p is not None) and (not np.isfinite(cvm_p) or cvm_p < threshold)
    ks_bad = (ks_p is not None) and (not np.isfinite(ks_p) or ks_p < threshold)
    return bool(cvm_bad and ks_bad)


def _parameter_rel_error(true_params: Sequence[float], est_params: Sequence[float]) -> float:
    true_arr = np.asarray(true_params, dtype=float).ravel()
    est_arr = np.asarray(est_params, dtype=float).ravel()
    k = int(min(true_arr.size, est_arr.size))
    if k == 0:
        return float("nan")
    denom = np.abs(true_arr[:k]) + 1e-8
    return float(np.mean(np.abs(est_arr[:k] - true_arr[:k]) / denom))


def _random_params_for_family(family: str, rng: np.random.Generator) -> Tuple[Tuple[float, ...], np.ndarray]:
    n = 1
    if family == "norm":
        loc = float(rng.normal(0.0, 2.0))
        scale = float(rng.uniform(0.8, 3.0))
        params = (loc, scale)
        data = sps.norm.rvs(*params, size=n)
    elif family == "expon":
        loc = 0.0
        scale = float(rng.uniform(0.4, 3.0))
        params = (loc, scale)
        data = sps.expon.rvs(*params, size=n)
    elif family == "gamma":
        a = float(rng.uniform(1.2, 6.0))
        loc = 0.0
        scale = float(rng.uniform(0.5, 3.0))
        params = (a, loc, scale)
        data = sps.gamma.rvs(*params, size=n)
    elif family == "weibull_min":
        c = float(rng.uniform(0.8, 3.5))
        loc = 0.0
        scale = float(rng.uniform(0.6, 3.0))
        params = (c, loc, scale)
        data = sps.weibull_min.rvs(*params, size=n)
    elif family == "lognorm":
        s = float(rng.uniform(0.35, 1.2))
        loc = 0.0
        scale = float(rng.uniform(0.8, 2.5))
        params = (s, loc, scale)
        data = sps.lognorm.rvs(*params, size=n)
    elif family == "beta":
        a = float(rng.uniform(0.8, 4.0))
        b = float(rng.uniform(0.8, 4.0))
        loc = 0.0
        scale = 1.0
        params = (a, b, loc, scale)
        data = sps.beta.rvs(*params, size=n)
    elif family == "t":
        df = float(rng.uniform(2.5, 12.0))
        loc = float(rng.normal(0.0, 0.25))
        scale = float(rng.uniform(0.9, 2.0))
        params = (df, loc, scale)
        data = sps.t.rvs(*params, size=n)
    elif family == "uniform":
        loc = float(rng.uniform(-3.0, 2.0))
        scale = float(rng.uniform(1.0, 6.0))
        params = (loc, scale)
        data = sps.uniform.rvs(*params, size=n)
    elif family == "laplace":
        loc = float(rng.normal(0.0, 0.2))
        scale = float(rng.uniform(0.8, 1.6))
        params = (loc, scale)
        data = sps.laplace.rvs(*params, size=n)
    elif family == "pareto":
        b = float(rng.uniform(1.2, 3.5))
        loc = 0.0
        scale = float(rng.uniform(0.8, 3.0))
        params = (b, loc, scale)
        data = sps.pareto.rvs(*params, size=n)
    else:
        raise ValueError(f"Unsupported family for synthetic sampling: {family}")

    return params, np.asarray(data, dtype=float)


def _generate_distribution_cases(spec: ValidationDatasetSpec, seed: int) -> List[DistributionCase]:
    rng = np.random.default_rng(seed)
    profile = str(spec.params.get("profile", ""))

    def sample_family_case(case_id: str, family: str, n: int, notes: str = "") -> DistributionCase:
        params, _ = _random_params_for_family(family, rng)
        data = np.asarray(getattr(sps, family).rvs(*params, size=n, random_state=rng), dtype=float)
        return DistributionCase(
            case_id=case_id,
            data=data,
            true_family=family,
            true_params=params,
            notes=notes,
        )

    cases: List[DistributionCase] = []

    if profile == "synthetic_parametric":
        families = ["norm", "expon", "gamma", "weibull_min", "lognorm", "beta", "t", "uniform"]
        for n in (50, 100, 500):
            for family in families:
                params, _ = _random_params_for_family(family, rng)
                data = np.asarray(getattr(sps, family).rvs(*params, size=n, random_state=rng), dtype=float)
                cases.append(
                    DistributionCase(
                        case_id=f"{family}_n{n}",
                        data=data,
                        true_family=family,
                        true_params=params,
                    )
                )
        return cases

    if profile == "actuarial":
        families = ["lognorm", "pareto", "gamma", "weibull_min"]
        for family in families:
            for n in (300, 1200):
                params, _ = _random_params_for_family(family, rng)
                data = np.asarray(getattr(sps, family).rvs(*params, size=n, random_state=rng), dtype=float)
                cases.append(
                    DistributionCase(
                        case_id=f"{family}_loss_n{n}",
                        data=data,
                        true_family=family,
                        true_params=params,
                    )
                )
        return cases

    if profile == "reliability":
        for family, n in (("weibull_min", 45), ("expon", 90), ("lognorm", 180)):
            params, _ = _random_params_for_family(family, rng)
            data = np.asarray(getattr(sps, family).rvs(*params, size=n, random_state=rng), dtype=float)
            cases.append(
                DistributionCase(
                    case_id=f"{family}_reliability_n{n}",
                    data=data,
                    true_family=family,
                    true_params=params,
                )
            )
        return cases

    if profile == "financial":
        for n, df in ((500, 8.0), (1000, 5.0), (2000, 3.5)):
            data = sps.t.rvs(df=df, loc=0.0, scale=1.0, size=n, random_state=rng)
            # Mild volatility clustering proxy.
            vol = np.abs(rng.normal(1.0, 0.18, size=n))
            data = data * vol
            cases.append(
                DistributionCase(
                    case_id=f"returns_t_df{df:.1f}_n{n}",
                    data=np.asarray(data, dtype=float),
                    true_family="t",
                    true_params=(df, 0.0, 1.0),
                    notes="Synthetic heavy-tailed returns proxy",
                )
            )
        return cases

    if profile == "hydrology":
        for n in (120, 365 * 4):
            gamma_part = sps.gamma.rvs(2.0, 0.0, 4.0, size=n, random_state=rng)
            zero_mask = rng.random(n) < 0.18
            gamma_part = np.asarray(gamma_part, dtype=float)
            gamma_part[zero_mask] = 0.0
            cases.append(
                DistributionCase(
                    case_id=f"rain_gamma_zero_inflated_n{n}",
                    data=gamma_part,
                    true_family="gamma",
                    true_params=(2.0, 0.0, 4.0),
                    notes="Zero-inflated rainfall proxy",
                )
            )
        return cases

    if profile == "internet":
        for n in (2000, 6000):
            mix = rng.random(n)
            pareto = sps.pareto.rvs(1.8, 0.0, 1.2, size=n, random_state=rng)
            logn = sps.lognorm.rvs(1.0, 0.0, 1.8, size=n, random_state=rng)
            data = np.where(mix < 0.65, pareto, logn)
            cases.append(
                DistributionCase(
                    case_id=f"traffic_heavy_tail_n{n}",
                    data=np.asarray(data, dtype=float),
                    acceptable_families=("pareto", "lognorm", "t"),
                    notes="Heavy-tailed traffic/file-size proxy",
                )
            )
        return cases

    if profile == "contaminated":
        base_families = ["norm", "gamma", "weibull_min"]
        for family in base_families:
            params, _ = _random_params_for_family(family, rng)
            base = np.asarray(getattr(sps, family).rvs(*params, size=500, random_state=rng), dtype=float)
            for eps in (0.05, 0.10, 0.20):
                n = base.size
                contam_mask = rng.random(n) < eps
                contam_vals = sps.t.rvs(df=1.0, loc=0.0, scale=max(1.0, np.std(base) * 4), size=n, random_state=rng)
                mixed = base.copy()
                mixed[contam_mask] = contam_vals[contam_mask]
                cases.append(
                    DistributionCase(
                        case_id=f"{family}_eps{int(eps*100)}",
                        data=mixed,
                        true_family=family,
                        true_params=params,
                        notes=f"{family} contaminated with heavy-tailed outliers ({eps:.0%})",
                    )
                )
        return cases

    if profile == "heaped":
        for family in ("norm", "expon", "lognorm"):
            params, _ = _random_params_for_family(family, rng)
            raw = np.asarray(getattr(sps, family).rvs(*params, size=800, random_state=rng), dtype=float)
            for delta in (1.0, 5.0, 10.0):
                rounded = np.round(raw / delta) * delta
                cases.append(
                    DistributionCase(
                        case_id=f"{family}_rounded_{delta:g}",
                        data=rounded,
                        true_family=family,
                        true_params=params,
                        notes=f"Rounded to nearest {delta:g}",
                    )
                )
        return cases

    if profile == "tail_discrimination":
        for n in (50, 100, 200):
            for family in ("gamma", "lognorm", "weibull_min"):
                params, _ = _random_params_for_family(family, rng)
                data = np.asarray(getattr(sps, family).rvs(*params, size=n, random_state=rng), dtype=float)
                cases.append(
                    DistributionCase(
                        case_id=f"tail_{family}_n{n}",
                        data=data,
                        true_family=family,
                        true_params=params,
                    )
                )
        return cases

    if profile == "near_symmetric":
        for n in (30, 50, 75):
            # Logistic is not in the default selector library; include near-equivalent acceptable families.
            logistic_data = sps.logistic.rvs(loc=0.0, scale=1.0, size=n, random_state=rng)
            cases.append(
                DistributionCase(
                    case_id=f"near_symmetric_logistic_n{n}",
                    data=np.asarray(logistic_data, dtype=float),
                    acceptable_families=("norm", "t", "laplace", "johnsonsu"),
                    notes="True logistic; accepted nearest in candidate library",
                )
            )
            for family in ("norm", "t", "laplace"):
                params, _ = _random_params_for_family(family, rng)
                data = np.asarray(getattr(sps, family).rvs(*params, size=n, random_state=rng), dtype=float)
                cases.append(
                    DistributionCase(
                        case_id=f"near_symmetric_{family}_n{n}",
                        data=data,
                        true_family=family,
                        true_params=params,
                    )
                )
        return cases

    if profile == "out_of_library":
        for n in (120, 500, 1000):
            # Stable distribution (alpha=1.5) is out of default selector library.
            stable = sps.levy_stable.rvs(1.5, 0.1, loc=0.0, scale=1.0, size=n, random_state=rng)
            cases.append(
                DistributionCase(
                    case_id=f"stable_alpha1.5_n{n}",
                    data=np.asarray(stable, dtype=float),
                    expect_rejection=True,
                    notes="Out-of-library stable distribution",
                )
            )
            # Sinh-arcsinh transform proxy (also out-of-library functional form).
            z = rng.normal(0.0, 1.0, size=n)
            sas = np.sinh(np.arcsinh(z) + 0.8)
            cases.append(
                DistributionCase(
                    case_id=f"sinh_arcsinh_n{n}",
                    data=np.asarray(sas, dtype=float),
                    expect_rejection=True,
                    notes="Out-of-library sinh-arcsinh style sample",
                )
            )
        return cases

    if profile == "mixtures":
        for n in (100, 300):
            for separation in (1.0, 2.0, 3.0):
                mix = rng.random(n)
                comp_a = rng.normal(loc=-0.5 * separation, scale=1.0, size=n)
                comp_b = rng.normal(loc=0.5 * separation, scale=1.0, size=n)
                data = np.where(mix < 0.5, comp_a, comp_b)
                cases.append(
                    DistributionCase(
                        case_id=f"gaussian_mix_sep{separation:.1f}_n{n}",
                        data=np.asarray(data, dtype=float),
                        expect_rejection=True,
                        notes="Two-component Gaussian mixture",
                    )
                )
        return cases

    raise ValueError(f"Unsupported distribution benchmark profile: {profile}")


def _fit_single_distribution_case(
    case: DistributionCase,
    components: Dict[str, bool],
    criterion: str,
    rejection_threshold: float,
) -> Dict[str, Any]:
    effective_criterion = str(criterion).strip()
    if bool(components.get("df.mnpo_aggregation", False)) and effective_criterion.lower() != "mnpo_oracle":
        effective_criterion = "mnpo_oracle"

    selector = _selector_for_distribution(components, criterion=effective_criterion)

    t0 = time.perf_counter()
    best_name, best_result, all_results = selector.select_best_distribution(
        case.data,
        criterion=effective_criterion,
        verbose=False,
    )
    fit_time = time.perf_counter() - t0

    rejected = _maybe_reject(best_name, best_result, components, rejection_threshold)

    if case.expect_rejection:
        correct = float(rejected)
    else:
        if rejected:
            correct = 0.0
        elif case.true_family is not None:
            correct = float(best_name == case.true_family)
        elif case.acceptable_families is not None:
            correct = float(best_name in set(case.acceptable_families))
        else:
            correct = float("nan")

    rel_error = float("nan")
    if (
        case.true_params is not None
        and best_name is not None
        and best_result is not None
        and best_name == case.true_family
        and getattr(best_result, "params", None) is not None
    ):
        try:
            rel_error = _parameter_rel_error(case.true_params, best_result.params)
        except Exception as exc:
            rel_error = float("nan")

    cvm_p = float(getattr(best_result, "cvm_p", np.nan)) if best_result is not None else float("nan")
    ks_p = float(getattr(best_result, "ks_p", np.nan)) if best_result is not None else float("nan")
    score_simple = float(getattr(best_result, "simple_score", np.nan)) if best_result is not None else float("nan")

    return {
        "case_id": case.case_id,
        "predicted_family": best_name,
        "true_family": case.true_family,
        "expect_rejection": int(case.expect_rejection),
        "rejected": int(rejected),
        "family_correct": correct,
        "parameter_rel_error": rel_error,
        "best_cvm_p": cvm_p,
        "best_ks_p": ks_p,
        "simple_score": score_simple,
        "n_candidates": int(len(all_results) if all_results is not None else 0),
        "fit_time_sec": float(fit_time),
        "notes": case.notes,
    }


def _transform_feature_with_distribution(
    x_train: np.ndarray,
    x_test: np.ndarray,
    selector: UnifiedDistributionSelectorV6,
    criterion: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)

    train_clean = x_train[np.isfinite(x_train)]
    if train_clean.size < 12 or np.nanstd(train_clean) < 1e-12:
        return x_train, x_test, {"status": "skipped", "reason": "insufficient_variation"}

    best_name, best_result, _ = selector.select_best_distribution(
        train_clean,
        criterion=criterion,
        verbose=False,
    )
    if best_name is None or best_result is None or getattr(best_result, "params", None) is None:
        return x_train, x_test, {"status": "skipped", "reason": "fit_failed"}

    dist_obj = selector.distributions.get(best_name)
    if dist_obj is None:
        return x_train, x_test, {"status": "skipped", "reason": "dist_missing"}

    params = best_result.params

    def to_gauss(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        out = arr.copy()
        mask = np.isfinite(arr)
        if np.any(mask):
            cdf_vals = dist_obj.cdf(arr[mask], *params)
            cdf_vals = np.clip(cdf_vals, 1e-8, 1 - 1e-8)
            out[mask] = sps.norm.ppf(cdf_vals)
        return out

    transformed_train = to_gauss(x_train)
    transformed_test = to_gauss(x_test)

    meta = {
        "status": "ok",
        "best_name": best_name,
        "cvm_p": float(getattr(best_result, "cvm_p", np.nan)),
        "ks_p": float(getattr(best_result, "ks_p", np.nan)),
    }
    return transformed_train, transformed_test, meta


def _estimate_family_stability(
    values: np.ndarray,
    components: Dict[str, bool],
    criterion: str,
    seed: int,
    n_bootstrap: int = 3,
) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 20:
        return 1.0

    rng = np.random.default_rng(seed)
    votes: List[str] = []
    for _ in range(max(2, n_bootstrap)):
        idx = rng.choice(np.arange(arr.size), size=arr.size, replace=True)
        boot = arr[idx]
        selector = _selector_for_distribution(components, criterion=criterion)
        name, _, _ = selector.select_best_distribution(boot, criterion=criterion, verbose=False)
        votes.append(str(name))

    mode_count = max(votes.count(v) for v in set(votes)) if votes else 0
    return float(mode_count / max(1, len(votes)))


def _run_integrated_preprocessing(
    X_train: np.ndarray,
    X_test: np.ndarray,
    components: Dict[str, bool],
    criterion: str,
    max_dist_features: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_imp)
    X_test_std = scaler.transform(X_test_imp)

    if not components.get("integrated.cdf_transform", True):
        return X_train_std, X_test_std, {
            "n_transformed_features": 0,
            "n_downweighted_low_gof": 0,
            "mean_stability_weight": 1.0,
            "transform_time_sec": 0.0,
        }

    effective_criterion = str(criterion).strip()
    if bool(components.get("df.mnpo_aggregation", False)) and effective_criterion.lower() != "mnpo_oracle":
        effective_criterion = "mnpo_oracle"

    n_features = X_train_std.shape[1]
    if n_features <= 0:
        return X_train_std, X_test_std, {
            "n_transformed_features": 0,
            "n_downweighted_low_gof": 0,
            "mean_stability_weight": 1.0,
            "transform_time_sec": 0.0,
        }

    max_dist_features = int(max(1, min(max_dist_features, n_features)))
    variances = np.nanvar(X_train_imp, axis=0)
    transform_idx = np.argsort(variances)[::-1][:max_dist_features]

    selector = _selector_for_distribution(components, criterion=effective_criterion)
    X_train_out = X_train_std.copy()
    X_test_out = X_test_std.copy()

    t0 = time.perf_counter()
    transformed = 0
    downweighted = 0
    stability_weights: List[float] = []

    for idx in transform_idx:
        tr_col, te_col, meta = _transform_feature_with_distribution(
            X_train_imp[:, idx],
            X_test_imp[:, idx],
            selector=selector,
            criterion=effective_criterion,
        )
        if meta.get("status") != "ok":
            continue

        weight = 1.0
        cvm_p = meta.get("cvm_p", np.nan)

        if components.get("integrated.low_gof_downweighting", False):
            if np.isfinite(cvm_p) and cvm_p < 0.01:
                weight *= 0.5
                downweighted += 1

        if components.get("integrated.dist_stability_signal", False):
            stability = _estimate_family_stability(
                X_train_imp[:, idx],
                components=components,
                criterion=effective_criterion,
                seed=seed + int(idx),
                n_bootstrap=3,
            )
            weight *= 0.5 + 0.5 * stability
            stability_weights.append(0.5 + 0.5 * stability)

        X_train_out[:, idx] = tr_col * weight
        X_test_out[:, idx] = te_col * weight
        transformed += 1

    transform_time = time.perf_counter() - t0
    mean_weight = float(np.mean(stability_weights)) if stability_weights else 1.0

    return X_train_out, X_test_out, {
        "n_transformed_features": int(transformed),
        "n_downweighted_low_gof": int(downweighted),
        "mean_stability_weight": mean_weight,
        "transform_time_sec": float(transform_time),
    }


def _component_relevant_to_pipeline(component: str, pipeline: str) -> bool:
    if component.startswith("fs."):
        return pipeline in {"fs", "integrated"}
    if component.startswith("df."):
        return pipeline in {"df", "integrated"}
    if component.startswith("integrated."):
        return pipeline == "integrated"
    return False


def _project_components_for_pipeline(components: Dict[str, bool], pipeline: str) -> Dict[str, bool]:
    return {
        k: v
        for k, v in components.items()
        if _component_relevant_to_pipeline(k, pipeline)
    }


def resolve_dataset_ids(
    catalog: Dict[str, ValidationDatasetSpec],
    dataset_sets: Sequence[str],
    explicit_ids: Sequence[str],
    exclude_ids: Sequence[str],
    pipelines: Sequence[str],
    max_datasets: Optional[int],
) -> List[str]:
    requested: List[str] = []

    for set_name in dataset_sets:
        if set_name not in DATASET_SETS:
            raise ValueError(f"Unknown dataset set: {set_name}")
        requested.extend(DATASET_SETS[set_name])

    for ds_id in explicit_ids:
        if ds_id not in catalog:
            raise ValueError(f"Unknown dataset id: {ds_id}")
        requested.append(ds_id)

    if not requested:
        requested = [ds_id for ds_id, spec in catalog.items() if spec.pipeline in set(pipelines)]

    # Preserve order, unique.
    seen = set()
    selected: List[str] = []
    pipeline_set = set(pipelines)
    exclude_set = set(exclude_ids)
    for ds_id in requested:
        if ds_id in seen:
            continue
        seen.add(ds_id)

        spec = catalog[ds_id]
        if spec.pipeline not in pipeline_set:
            continue
        if ds_id in exclude_set:
            continue
        selected.append(ds_id)

    if max_datasets is not None and max_datasets > 0:
        selected = selected[:max_datasets]

    if not selected:
        raise ValueError("Dataset selection is empty after filtering.")

    return selected


def build_ablation_configs(
    profile: str,
    base_components: Dict[str, bool],
    pipelines: Sequence[str],
    constrained_components: Optional[Sequence[str]] = None,
) -> List[AblationConfig]:
    pipeline_set = set(pipelines)

    if constrained_components is not None:
        for comp in constrained_components:
            if comp not in base_components:
                raise ValueError(f"Unknown component in --ablation-components: {comp}")

    relevant = [
        comp
        for comp in base_components.keys()
        if any(_component_relevant_to_pipeline(comp, p) for p in pipeline_set)
    ]

    if constrained_components is not None:
        constrained = set(constrained_components)
        relevant = [comp for comp in relevant if comp in constrained]

    baseline = AblationConfig(name="baseline", components=dict(base_components))

    if profile == "none":
        return [baseline]

    configs = [baseline]
    for comp in relevant:
        if profile == "single_off":
            if not base_components[comp]:
                continue
            modified = dict(base_components)
            modified[comp] = False
            configs.append(
                AblationConfig(name=f"disable_{comp.replace('.', '_')}", components=modified)
            )
        elif profile == "single_toggle":
            modified = dict(base_components)
            modified[comp] = not bool(base_components[comp])
            configs.append(
                AblationConfig(name=f"toggle_{comp.replace('.', '_')}", components=modified)
            )
        else:
            raise ValueError(f"Unknown ablation profile: {profile}")

    return configs


def _dedupe_pipeline_configs(
    configs: Sequence[AblationConfig],
    pipeline: str,
) -> List[AblationConfig]:
    deduped: Dict[Tuple[Tuple[str, bool], ...], AblationConfig] = {}
    for cfg in configs:
        projection = _project_components_for_pipeline(cfg.components, pipeline)
        key = tuple(sorted(projection.items()))
        if key not in deduped:
            deduped[key] = AblationConfig(name=cfg.name, components=projection)
    return list(deduped.values())


def _integrated_scenario_defaults(spec: ValidationDatasetSpec) -> Dict[str, bool]:
    scenario = str(spec.params.get("scenario", "")).strip().lower()
    if scenario == "low_gof_downweighting":
        return {"integrated.low_gof_downweighting": True}
    if scenario == "stability_signal":
        return {"integrated.dist_stability_signal": True}
    if scenario == "cdf_transform":
        return {"integrated.cdf_transform": True}
    return {}


def _apply_scenario_defaults_to_configs(
    configs: Sequence[AblationConfig],
    base_projection: Dict[str, bool],
    scenario_defaults: Dict[str, bool],
) -> List[AblationConfig]:
    if not scenario_defaults:
        return list(configs)

    updated: List[AblationConfig] = []
    for cfg in configs:
        merged = dict(cfg.components)
        # Scenario defaults define baseline behavior for scenario-specific tests.
        merged.update(scenario_defaults)

        # Preserve ablation deltas relative to the global base projection.
        for key, value in cfg.components.items():
            if key in base_projection and value != base_projection[key]:
                merged[key] = value

        updated.append(AblationConfig(name=cfg.name, components=merged))
    return updated


def _summarize_runs(runs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["pipeline", "dataset_id", "config_name"]
    if "domain" in runs_df.columns:
        group_cols.append("domain")
    if "platform" in runs_df.columns:
        group_cols.append("platform")
    metric_cols = [
        c
        for c in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "hybrid_score",
            "selected_features",
            "family_correct",
            "parameter_rel_error",
            "rejected",
            "best_cvm_p",
            "best_ks_p",
            "simple_score",
            "fit_time_sec",
            "fs_time_sec",
            "transform_time_sec",
            "n_transformed_features",
            "n_downweighted_low_gof",
            "mean_stability_weight",
        ]
        if c in runs_df.columns
    ]

    if not metric_cols:
        return pd.DataFrame(), pd.DataFrame()

    summary = runs_df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]

    by_config = runs_df.groupby(["pipeline", "config_name"], dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    by_config.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in by_config.columns
    ]

    return summary, by_config


def _print_dataset_listing(catalog: Dict[str, ValidationDatasetSpec]) -> None:
    rows = []
    for ds_id, spec in catalog.items():
        rows.append(
            {
                "dataset_id": ds_id,
                "pipeline": spec.pipeline,
                "tier": spec.tier,
                "loader_kind": spec.loader_kind,
                "display_name": spec.display_name,
            }
        )
    df = pd.DataFrame(rows).sort_values(["pipeline", "tier", "dataset_id"])  # type: ignore[arg-type]
    print(df.to_string(index=False))
    print("\nAvailable dataset sets:")
    for set_name in sorted(DATASET_SETS.keys()):
        print(f"- {set_name}: {len(DATASET_SETS[set_name])} datasets")


def run_validation_suite(args: argparse.Namespace) -> Path:
    if args.list_datasets:
        _print_dataset_listing(CATALOG)
        raise SystemExit(0)
    if bool(args.allow_synthetic_fallback):
        raise ValueError(
            "Synthetic fallback is forbidden for validation-suite runs. "
            "Use the HuggingFace bundle via TABNETICS_HF_ORG or TABNETICS_HF_REPO_ID."
        )

    selected_ids = resolve_dataset_ids(
        catalog=CATALOG,
        dataset_sets=args.dataset_sets,
        explicit_ids=args.datasets,
        exclude_ids=args.exclude_datasets,
        pipelines=args.pipelines,
        max_datasets=args.max_datasets,
    )
    for ds_id in selected_ids:
        spec = CATALOG[ds_id]
        if spec.loader_kind not in _HF_AUTHORITATIVE_SKIP_KINDS:
            _require_hf_bundle_configuration(dataset_id=str(ds_id), loader_kind=str(spec.loader_kind))

    base_components = dict(COMPONENT_DEFAULTS)
    for name in args.enable_components:
        if name not in base_components:
            raise ValueError(f"Unknown component in --enable-components: {name}")
        base_components[name] = True
    for name in args.disable_components:
        if name not in base_components:
            raise ValueError(f"Unknown component in --disable-components: {name}")
        base_components[name] = False

    ablation_configs_all = build_ablation_configs(
        profile=args.ablation_profile,
        base_components=base_components,
        pipelines=args.pipelines,
        constrained_components=args.ablation_components,
    )

    pipeline_configs = {
        "fs": _dedupe_pipeline_configs(ablation_configs_all, "fs"),
        "df": _dedupe_pipeline_configs(ablation_configs_all, "df"),
        "integrated": _dedupe_pipeline_configs(ablation_configs_all, "integrated"),
    }

    run_output_dir = create_timestamped_run_dir(args.output_dir, "validation_suite")

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for ds_id in selected_ids:
        spec = CATALOG[ds_id]
        promotion_meta = _dataset_promotion_metadata(spec)
        print(f"[dataset] {ds_id} ({spec.pipeline}/{spec.tier})", flush=True)

        if spec.pipeline == "fs":
            cfgs = pipeline_configs["fs"]
            for seed in args.seeds:
                try:
                    loaded = load_feature_selection_dataset(
                        spec,
                        seed=seed,
                        allow_synthetic_fallback=False,
                        sample_cap=args.synthetic_sample_cap,
                        feature_cap=args.synthetic_feature_cap,
                        class_integrity_policy=args.dataset_integrity_policy,
                        class_min_classes=args.dataset_min_classes,
                        class_min_class_count=args.dataset_min_class_count,
                        source_policy="real_only",
                        require_hf_source=True,
                    )
                    X_train, X_test, y_train, y_test = _safe_train_test_split(
                        loaded.X,
                        loaded.y,
                        test_size=0.20,
                        seed=seed,
                        max_train_samples=spec.params.get("max_train_samples"),
                    )
                except DatasetIntegritySkipError as exc:
                    failures.append(
                        {
                            "dataset_id": ds_id,
                            "seed": seed,
                            "status": "skipped_dataset_integrity",
                            "error": str(exc),
                        }
                    )
                    continue
                except Exception as exc:
                    failures.append({"dataset_id": ds_id, "seed": seed, "error": str(exc)})
                    continue

                fs_fraction = float(spec.params.get("fs_fraction", args.fs_fraction))
                n_final = int(spec.params.get("n_final_features", args.n_final_features))

                for cfg in cfgs:
                    try:
                        metrics = _run_fs_core(
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            components=cfg.components,
                            fs_fraction=fs_fraction,
                            n_final_features=n_final,
                            seed=seed,
                            already_preprocessed=False,
                        )
                        metrics.update(
                            {
                                "pipeline": "fs",
                                "dataset_id": ds_id,
                                "dataset_name": spec.display_name,
                                "tier": spec.tier,
                                "domain": str(getattr(spec, "domain", "genomics") or "genomics"),
                                "platform": str(getattr(spec, "platform", "cDNA") or "cDNA"),
                                "seed": seed,
                                "config_name": cfg.name,
                                "n_samples_total": int(loaded.X.shape[0]),
                                "n_features_total": int(loaded.X.shape[1]),
                                "data_source": loaded.data_source,
                                "data_notes": loaded.notes,
                                "promotion_eligible": int(promotion_meta["promotion_eligible"]),
                                "promotion_blocker": str(promotion_meta["promotion_blocker"]),
                                "source_policy": str(promotion_meta["source_policy"]),
                            }
                        )
                        rows.append(metrics)
                    except Exception as exc:
                        failures.append(
                            {
                                "dataset_id": ds_id,
                                "seed": seed,
                                "config": cfg.name,
                                "error": str(exc),
                            }
                        )

        elif spec.pipeline == "df":
            cfgs = pipeline_configs["df"]
            for seed in args.seeds:
                try:
                    cases = _generate_distribution_cases(spec, seed)
                except Exception as exc:
                    failures.append({"dataset_id": ds_id, "seed": seed, "error": str(exc)})
                    continue

                for cfg in cfgs:
                    for case in cases:
                        try:
                            case_metrics = _fit_single_distribution_case(
                                case,
                                components=cfg.components,
                                criterion=args.df_criterion,
                                rejection_threshold=args.df_rejection_threshold,
                            )
                            case_metrics.update(
                                {
                                    "pipeline": "df",
                                    "dataset_id": ds_id,
                                    "dataset_name": spec.display_name,
                                    "tier": spec.tier,
                                    "domain": str(getattr(spec, "domain", "genomics") or "genomics"),
                                    "platform": str(getattr(spec, "platform", "cDNA") or "cDNA"),
                                    "seed": seed,
                                    "config_name": cfg.name,
                                    "n_samples_total": int(case.data.size),
                                    "n_features_total": 1,
                                    "data_source": "benchmark_synthetic",
                                    "data_notes": case.notes,
                                    "promotion_eligible": int(promotion_meta["promotion_eligible"]),
                                    "promotion_blocker": str(promotion_meta["promotion_blocker"]),
                                    "source_policy": str(promotion_meta["source_policy"]),
                                }
                            )
                            rows.append(case_metrics)
                        except Exception as exc:
                            failures.append(
                                {
                                    "dataset_id": ds_id,
                                    "seed": seed,
                                    "config": cfg.name,
                                    "case_id": case.case_id,
                                    "error": str(exc),
                                }
                            )

        elif spec.pipeline == "integrated":
            scenario_defaults = _integrated_scenario_defaults(spec)
            integrated_base_projection = _project_components_for_pipeline(base_components, "integrated")
            cfgs = _apply_scenario_defaults_to_configs(
                pipeline_configs["integrated"],
                base_projection=integrated_base_projection,
                scenario_defaults=scenario_defaults,
            )
            for seed in args.seeds:
                try:
                    loaded = load_feature_selection_dataset(
                        spec,
                        seed=seed,
                        allow_synthetic_fallback=False,
                        sample_cap=args.synthetic_sample_cap,
                        feature_cap=args.synthetic_feature_cap,
                        class_integrity_policy=args.dataset_integrity_policy,
                        class_min_classes=args.dataset_min_classes,
                        class_min_class_count=args.dataset_min_class_count,
                        source_policy="real_only",
                        require_hf_source=True,
                    )
                    X_train, X_test, y_train, y_test = _safe_train_test_split(
                        loaded.X,
                        loaded.y,
                        test_size=0.20,
                        seed=seed,
                        max_train_samples=spec.params.get("max_train_samples"),
                    )
                except DatasetIntegritySkipError as exc:
                    failures.append(
                        {
                            "dataset_id": ds_id,
                            "seed": seed,
                            "status": "skipped_dataset_integrity",
                            "error": str(exc),
                        }
                    )
                    continue
                except Exception as exc:
                    failures.append({"dataset_id": ds_id, "seed": seed, "error": str(exc)})
                    continue

                fs_fraction = float(args.fs_fraction)
                n_final = int(args.n_final_features)

                for cfg in cfgs:
                    try:
                        X_train_int, X_test_int, integ_meta = _run_integrated_preprocessing(
                            X_train,
                            X_test,
                            components=cfg.components,
                            criterion=args.integrated_dist_criterion,
                            max_dist_features=args.integrated_max_dist_features,
                            seed=seed,
                        )

                        metrics = _run_fs_core(
                            X_train_int,
                            y_train,
                            X_test_int,
                            y_test,
                            components=cfg.components,
                            fs_fraction=fs_fraction,
                            n_final_features=n_final,
                            seed=seed,
                            already_preprocessed=True,
                        )

                        metrics.update(integ_meta)
                        metrics.update(
                            {
                                "pipeline": "integrated",
                                "dataset_id": ds_id,
                                "dataset_name": spec.display_name,
                                "tier": spec.tier,
                                "domain": str(getattr(spec, "domain", "genomics") or "genomics"),
                                "platform": str(getattr(spec, "platform", "cDNA") or "cDNA"),
                                "seed": seed,
                                "config_name": cfg.name,
                                "n_samples_total": int(loaded.X.shape[0]),
                                "n_features_total": int(loaded.X.shape[1]),
                                "data_source": loaded.data_source,
                                "data_notes": loaded.notes,
                                "promotion_eligible": int(promotion_meta["promotion_eligible"]),
                                "promotion_blocker": str(promotion_meta["promotion_blocker"]),
                                "source_policy": str(promotion_meta["source_policy"]),
                            }
                        )
                        rows.append(metrics)
                    except Exception as exc:
                        failures.append(
                            {
                                "dataset_id": ds_id,
                                "seed": seed,
                                "config": cfg.name,
                                "error": str(exc),
                            }
                        )

        else:
            failures.append({"dataset_id": ds_id, "error": f"Unsupported pipeline: {spec.pipeline}"})

    if not rows:
        raise RuntimeError("Validation run produced no rows. Check failures metadata.")

    runs_df = pd.DataFrame(rows)
    summary_df, by_config_df = _summarize_runs(runs_df)
    domain_summary_df = pd.DataFrame()
    if "domain" in runs_df.columns:
        metric_cols = [
            c for c in ("accuracy", "balanced_accuracy", "macro_f1", "hybrid_score")
            if c in runs_df.columns
        ]
        if metric_cols:
            domain_summary_df = (
                runs_df.groupby(["pipeline", "domain", "config_name"], dropna=False)[metric_cols]
                .agg(["mean", "std"])
                .reset_index()
            )
            domain_summary_df.columns = [
                "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
                for col in domain_summary_df.columns
            ]

    runs_path = run_output_dir / "validation_runs.csv"
    summary_path = run_output_dir / "validation_summary_by_dataset.csv"
    by_config_path = run_output_dir / "validation_summary_by_config.csv"
    domain_summary_path = run_output_dir / "validation_summary_by_domain.csv"
    failures_path = run_output_dir / "validation_failures.json"
    metadata_path = run_output_dir / "validation_metadata.json"
    datasets_path = run_output_dir / "validation_dataset_manifest.json"

    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    by_config_df.to_csv(by_config_path, index=False)
    if not domain_summary_df.empty:
        domain_summary_df.to_csv(domain_summary_path, index=False)

    metadata = {
        "selected_datasets": selected_ids,
        "pipelines": args.pipelines,
        "dataset_sets": args.dataset_sets,
        "seeds": args.seeds,
        "ablation_profile": args.ablation_profile,
        "ablation_component_overrides": {
            "enable": args.enable_components,
            "disable": args.disable_components,
            "subset": args.ablation_components,
        },
        "component_defaults": base_components,
        "dataset_integrity": {
            "policy": str(args.dataset_integrity_policy),
            "min_classes": int(args.dataset_min_classes),
            "min_class_count": int(args.dataset_min_class_count),
            "allow_synthetic_fallback": bool(args.allow_synthetic_fallback),
        },
        "noop_components": sorted(NOOP_COMPONENTS),
        "pipeline_config_counts": {k: len(v) for k, v in pipeline_configs.items()},
        "output_files": {
            "runs": str(runs_path),
            "summary_by_dataset": str(summary_path),
            "summary_by_config": str(by_config_path),
            "summary_by_domain": str(domain_summary_path) if not domain_summary_df.empty else "",
            "failures": str(failures_path),
        },
    }

    with failures_path.open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    dataset_manifest = {ds_id: _dataset_to_json(CATALOG[ds_id]) for ds_id in selected_ids}
    with datasets_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    print(f"Run output directory: {run_output_dir}")
    print(f"Saved run-level results to: {runs_path}")
    print(f"Saved dataset summary to: {summary_path}")
    print(f"Saved config summary to: {by_config_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved failures to: {failures_path}")

    return run_output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified validation suite for distribution fitting, feature selection, and integrated pipeline tests"
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["fs", "df", "integrated"],
        choices=["fs", "df", "integrated"],
        help="Pipelines to run",
    )
    parser.add_argument(
        "--dataset-sets",
        nargs="+",
        default=[],
        help="Named dataset sets (e.g., all, fs_all, df_hard, integrated_all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Explicit dataset ids to include",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=[],
        help="Dataset ids to exclude",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Optional cap after selection",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print available dataset ids and exit",
    )

    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--fs-fraction", type=float, default=0.4)
    parser.add_argument("--n-final-features", type=int, default=50)

    parser.add_argument(
        "--df-criterion",
        type=str,
        default="simple",
        choices=["simple", "cvm_p", "ks_p", "bic", "aic", "aicc", "cv", "cv_loglik", "crps", "mnpo_oracle"],
    )
    parser.add_argument("--df-rejection-threshold", type=float, default=0.01)

    parser.add_argument(
        "--integrated-dist-criterion",
        type=str,
        default="simple",
        choices=["simple", "cvm_p", "ks_p", "bic", "aic", "aicc", "cv", "cv_loglik", "crps", "mnpo_oracle"],
    )
    parser.add_argument("--integrated-max-dist-features", type=int, default=128)

    parser.add_argument(
        "--ablation-profile",
        type=str,
        default="single_off",
        choices=["none", "single_off", "single_toggle"],
    )
    parser.add_argument(
        "--ablation-components",
        nargs="*",
        default=None,
        help="Optional subset of component keys to ablate",
    )
    parser.add_argument("--enable-components", nargs="*", default=[])
    parser.add_argument("--disable-components", nargs="*", default=[])

    parser.add_argument(
        "--allow-synthetic-fallback",
        dest="allow_synthetic_fallback",
        action="store_true",
        help="Enable synthetic fallback when real dataset loading fails",
    )
    parser.add_argument(
        "--no-synthetic-fallback",
        dest="allow_synthetic_fallback",
        action="store_false",
        help="Fail when synthetic fallback is disabled",
    )
    parser.set_defaults(allow_synthetic_fallback=False)
    parser.add_argument(
        "--dataset-integrity-policy",
        type=str,
        default="error",
        choices=["fallback", "skip", "error"],
        help=(
            "Policy when loaded dataset fails class-diversity sanity checks "
            "(n_classes/min class count)."
        ),
    )
    parser.add_argument(
        "--dataset-min-classes",
        type=int,
        default=2,
        help="Minimum required number of classes after loading.",
    )
    parser.add_argument(
        "--dataset-min-class-count",
        type=int,
        default=1,
        help="Minimum required samples per class after loading.",
    )

    parser.add_argument("--synthetic-sample-cap", type=int, default=2500)
    parser.add_argument("--synthetic-feature-cap", type=int, default=10000)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="run_artifacts/validation/unified_suite",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_validation_suite(args)


if __name__ == "__main__":
    main()

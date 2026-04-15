"""Run integrated DF+FS benchmarking with SOTA comparison and ablations."""

from __future__ import annotations

import os as _os  # noqa: E402  (used only for env setup, re-imported below)

from tabnetics.core.runtime import configure_runtime_environment  # noqa: E402

configure_runtime_environment()

import argparse
import inspect
import json
import os
import pickle
import re
import queue
import signal
import sys
import math
import tempfile
import warnings
import multiprocessing as mp
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as sps
try:
    from tabnetics.core.compat import Parallel, delayed
except Exception as exc:
    from tabnetics.core.compat import Parallel, delayed  # type: ignore
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

try:
    from tabnetics.benchmarks.artifacts import create_timestamped_run_dir
    from tabnetics.core.errors import DatasetIntegritySkipError
    from tabnetics.datasets.loaders import _require_hf_bundle_configuration, load_feature_selection_dataset
    from tabnetics.datasets.validation_catalog import CATALOG
    from tabnetics.domains.bio import infer_prefilter_data_domain
    from tabnetics.pipeline.pipeline import (
        ClassificationConfig,
        DFFSConfig,
        DistributionFeatureSelectionPipeline,
        DistributionFitterConfig,
    )
except Exception as exc:
    from tabnetics.benchmarks.artifacts import create_timestamped_run_dir
    from tabnetics.core.errors import DatasetIntegritySkipError  # type: ignore
    from tabnetics.datasets.loaders import _require_hf_bundle_configuration, load_feature_selection_dataset  # type: ignore
    from tabnetics.datasets.validation_catalog import CATALOG  # type: ignore
    from tabnetics.domains.bio import infer_prefilter_data_domain  # type: ignore
    from tabnetics.pipeline.pipeline import ClassificationConfig, DFFSConfig, DistributionFeatureSelectionPipeline, DistributionFitterConfig


try:
    from tabnetics.datasets.benchmark_catalog import (
        BENCHMARK_DATASETS,
        DATASET_SETS,
        BenchmarkDatasetSpec,
        KNOWN_SOTA_PROTOCOL_RANGES,
        TIER_SOTA_DEFAULTS_INFLATED,
        TIER_SOTA_DEFAULTS_STRICT,
        _build_benchmark_datasets,
        _build_dataset_sets,
        _sota_ranges_for_dataset,
        _synthetic_specs,
        _validation_specs,
    )
except Exception as exc:
    from tabnetics.datasets.benchmark_catalog import (  # type: ignore
        BENCHMARK_DATASETS,
        DATASET_SETS,
        BenchmarkDatasetSpec,
        KNOWN_SOTA_PROTOCOL_RANGES,
        TIER_SOTA_DEFAULTS_INFLATED,
        TIER_SOTA_DEFAULTS_STRICT,
        _build_benchmark_datasets,
        _build_dataset_sets,
        _sota_ranges_for_dataset,
        _synthetic_specs,
        _validation_specs,
    )

from tabnetics.benchmarks.profiles import FS_METHOD_SETS


def _build_integrated_parent_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ds_id, spec in CATALOG.items():
        if str(getattr(spec, "pipeline", "")).strip().lower() != "integrated":
            continue
        base = None
        try:
            base = spec.params.get("base_dataset")
        except Exception as exc:
            base = None
        if base:
            out[str(ds_id)] = str(base)
    return out


# Map integrated datasets to their parent FS dataset for SOTA-matched evaluation.
_INTEGRATED_PARENT_MAP: Dict[str, str] = _build_integrated_parent_map()


MODEL_CANDIDATE_PROFILES: Dict[str, Tuple[str, ...]] = {
    "default": tuple(),
    # A11: medium-tier classifier mismatch follow-through pool.
    "a11_medium_mismatch": ("lr", "svm_rbf", "svm_linear", "dlda", "knn", "nb", "vote_ensemble"),
}


_SPARSE_SCREENING_MODE_ALIASES: Dict[str, str] = {
    "strong": "prefilter_aggressive",
    "gap_safe": "prefilter_balanced",
    "slores": "prefilter_conservative",
}
_SPARSE_SCREENING_MODE_VALID = {
    "none",
    "prefilter_aggressive",
    "prefilter_balanced",
    "prefilter_conservative",
}
_DEPRECATED_TOGGLE_WARNED: set[str] = set()


def _warn_deprecated_toggle_once(key: str, message: str) -> None:
    if key in _DEPRECATED_TOGGLE_WARNED:
        return
    _DEPRECATED_TOGGLE_WARNED.add(str(key))
    warnings.warn(message, DeprecationWarning)


def _canonicalize_sparse_screening_mode(mode: Any, *, warn_deprecated: bool = False) -> str:
    raw = str(mode if mode is not None else "none").strip().lower()
    if raw in _SPARSE_SCREENING_MODE_ALIASES:
        canonical = _SPARSE_SCREENING_MODE_ALIASES[raw]
        if warn_deprecated:
            warnings.warn(
                f"--fs-sparse-multinomial-screening-mode={raw!r} is deprecated; "
                f"use {canonical!r} instead.",
                DeprecationWarning,
            )
        raw = canonical
    if raw not in _SPARSE_SCREENING_MODE_VALID:
        raw = "none"
    return raw


def _warn_deprecated_df_fastpath_args(args: argparse.Namespace) -> None:
    """Emit one-time warnings for legacy no-op DF fast-path CLI flags."""
    if bool(getattr(args, "enable_df_fastpath", False)):
        _warn_deprecated_toggle_once(
            "enable_df_fastpath",
            "--enable-df-fastpath is deprecated and ignored; the DF fast-path was removed.",
        )
    if str(getattr(args, "df_fastpath_trigger", "small_n_or_low_unique") or "small_n_or_low_unique") != "small_n_or_low_unique":
        _warn_deprecated_toggle_once(
            "df_fastpath_trigger",
            "--df-fastpath-trigger is deprecated and ignored; the DF fast-path was removed.",
        )
    if int(getattr(args, "df_fastpath_small_n_threshold", 250) or 250) != 250:
        _warn_deprecated_toggle_once(
            "df_fastpath_small_n_threshold",
            "--df-fastpath-small-n-threshold is deprecated and ignored; the DF fast-path was removed.",
        )
    if float(getattr(args, "df_fastpath_unique_ratio_threshold", 0.05) or 0.05) != 0.05:
        _warn_deprecated_toggle_once(
            "df_fastpath_unique_ratio_threshold",
            "--df-fastpath-unique-ratio-threshold is deprecated and ignored; the DF fast-path was removed.",
        )
    if int(getattr(args, "df_fastpath_n_unique_threshold", 12) or 12) != 12:
        _warn_deprecated_toggle_once(
            "df_fastpath_n_unique_threshold",
            "--df-fastpath-n-unique-threshold is deprecated and ignored; the DF fast-path was removed.",
        )


def _parse_csv_or_space_list(raw: Any) -> Tuple[str, ...]:
    """Parse a comma/space-separated list argument into a stable tuple."""
    if raw is None:
        return tuple()
    if isinstance(raw, (list, tuple)):
        toks = [str(x) for x in raw]
    else:
        toks = re.split(r"[,\s]+", str(raw))
    cleaned = [t.strip().lower() for t in toks if str(t).strip()]
    return tuple(cleaned)


def _resolve_fs_method_stack(raw: Any) -> Tuple[str, ...]:
    """Resolve a method stack from either a preset name or explicit list."""
    if raw is None:
        return tuple()
    token = str(raw).strip()
    if not token:
        return tuple()
    if token in FS_METHOD_SETS:
        return tuple(str(m) for m in FS_METHOD_SETS[token])
    return tuple(str(m) for m in _parse_csv_or_space_list(token))


def _resolve_tier_lockout_fallback_methods(
    args: argparse.Namespace,
    default_methods: Sequence[str],
) -> Tuple[str, ...]:
    explicit = _resolve_fs_method_stack(getattr(args, "tier_lockout_fallback_methods", ""))
    if explicit:
        return explicit
    preset = _resolve_fs_method_stack(getattr(args, "tier_lockout_fallback_fs_method_set", ""))
    if preset:
        return preset
    return tuple(str(m) for m in default_methods)


def _resolve_regime_gating_simple_methods(
    args: argparse.Namespace,
    default_methods: Sequence[str],
) -> Tuple[str, ...]:
    explicit = _resolve_fs_method_stack(getattr(args, "regime_gating_simple_methods", ""))
    if explicit:
        return explicit
    preset = _resolve_fs_method_stack(getattr(args, "regime_gating_simple_fs_method_set", ""))
    if preset:
        return preset
    return tuple(str(m) for m in default_methods)


def _parse_tier_routing_table(raw: Any) -> Dict[str, Tuple[str, ...]]:
    """Parse tier-routing spec.

    Supports:
    - JSON dict: {"easy":"strict_plus_mrmr","hard":"m1,m2"}
    - ';'-separated k=v pairs: easy=strict_plus_mrmr;hard=m1,m2
    """
    out: Dict[str, Tuple[str, ...]] = {}
    if raw is None:
        return out
    text = str(raw).strip()
    if not text:
        return out

    parsed: Any = None
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except Exception as exc:
            parsed = None
    if isinstance(parsed, dict):
        for tier_raw, methods_raw in parsed.items():
            tier = str(tier_raw).strip().lower()
            if tier not in {"easy", "medium", "hard", "very_hard"}:
                continue
            methods = _resolve_fs_method_stack(methods_raw)
            if methods:
                out[tier] = methods
        return out

    for chunk in text.split(";"):
        entry = str(chunk).strip()
        if not entry or "=" not in entry:
            continue
        tier_raw, methods_raw = entry.split("=", 1)
        tier = str(tier_raw).strip().lower()
        if tier not in {"easy", "medium", "hard", "very_hard"}:
            continue
        methods = _resolve_fs_method_stack(methods_raw)
        if methods:
            out[tier] = methods
    return out


def _get_sota_classifiers_for_dataset(dataset_id: str) -> Optional[Tuple[str, ...]]:
    ds = str(dataset_id).strip()
    if ds in DATASET_SOTA_CLASSIFIERS:
        return DATASET_SOTA_CLASSIFIERS[ds]
    parent = _INTEGRATED_PARENT_MAP.get(ds)
    if parent and parent in DATASET_SOTA_CLASSIFIERS:
        return DATASET_SOTA_CLASSIFIERS[parent]
    return None


def _benchmark_dataset_promotion_metadata(spec: BenchmarkDatasetSpec) -> Dict[str, Any]:
    source_policy = "standard"
    promotion_eligible = True
    promotion_blocker = ""
    if spec.source_kind == "validation_catalog" and spec.validation_dataset_id in CATALOG:
        val_spec = CATALOG[spec.validation_dataset_id]
        params = dict(getattr(val_spec, "params", {}) or {})
        source_policy = str(params.get("source_policy", "standard") or "standard").strip().lower()
        promotion_eligible = bool(params.get("promotion_eligible", True))
        promotion_blocker = str(params.get("promotion_blocker", "") or "").strip()
        if promotion_eligible:
            promotion_blocker = ""
    return {
        "promotion_eligible": int(bool(promotion_eligible)),
        "promotion_blocker": promotion_blocker,
        "source_policy": source_policy,
    }


def _apply_exact_model_candidate_set(cfg: DFFSConfig, candidates: Sequence[str]) -> None:
    # Ensure the config's candidate list is *exactly* the requested set: include
    # flags would otherwise expand the set beyond cfg.model_candidates.
    cand_set = set(str(c) for c in candidates)
    cfg.model_candidates = tuple(str(c) for c in candidates)
    if hasattr(cfg, "classification") and cfg.classification is not None:
        cfg.classification.model_candidates = tuple(str(c) for c in candidates)
    cfg.include_elastic_net_model = "elastic_net_lr" in cand_set
    cfg.include_rf_model = "rf" in cand_set
    cfg.include_knn_model = "knn" in cand_set
    cfg.include_svm_linear_model = "svm_linear" in cand_set
    cfg.include_dlda_model = "dlda" in cand_set
    cfg.include_nsc_model = "nsc" in cand_set
    cfg.include_pls_da_model = "pls_da_classifier" in cand_set
    cfg.include_gpc_model = "gpc" in cand_set
    cfg.include_nb_model = "nb" in cand_set
    cfg.include_vote_ensemble_model = "vote_ensemble" in cand_set
    cfg.include_rp_ensemble_model = "rp_ensemble" in cand_set
    cfg.include_dbda_model = "dbda" in cand_set
    cfg.include_gqda_model = "gqda" in cand_set
    cfg.include_bc_svm_linear_model = "bc_svm_linear" in cand_set
    cfg.include_sglnn_model = "sglnn" in cand_set
    cfg.include_xgb_model = "xgb" in cand_set
    cfg.include_lgbm_model = "lgbm" in cand_set
    cfg.include_extra_tree_model = "extra_tree" in cand_set
    cfg.include_catboost_model = "catboost" in cand_set
    cfg.include_tabpfn_model = "tabpfn" in cand_set
    if hasattr(cfg, "classification") and cfg.classification is not None:
        cfg.classification.include_elastic_net_model = bool(cfg.include_elastic_net_model)
        cfg.classification.include_rf_model = bool(cfg.include_rf_model)
        cfg.classification.include_knn_model = bool(cfg.include_knn_model)
        cfg.classification.include_svm_linear_model = bool(cfg.include_svm_linear_model)
        cfg.classification.include_dlda_model = bool(cfg.include_dlda_model)
        cfg.classification.include_nsc_model = bool(cfg.include_nsc_model)
        cfg.classification.include_pls_da_model = bool(cfg.include_pls_da_model)
        cfg.classification.include_gpc_model = bool(cfg.include_gpc_model)
        cfg.classification.include_nb_model = bool(cfg.include_nb_model)
        cfg.classification.include_vote_ensemble_model = bool(cfg.include_vote_ensemble_model)
        cfg.classification.include_rp_ensemble_model = bool(cfg.include_rp_ensemble_model)
        cfg.classification.include_dbda_model = bool(cfg.include_dbda_model)
        cfg.classification.include_gqda_model = bool(cfg.include_gqda_model)
        cfg.classification.include_bc_svm_linear_model = bool(cfg.include_bc_svm_linear_model)
        cfg.classification.include_sglnn_model = bool(cfg.include_sglnn_model)
        cfg.classification.include_xgb_model = bool(cfg.include_xgb_model)
        cfg.classification.include_lgbm_model = bool(cfg.include_lgbm_model)
        cfg.classification.include_extra_tree_model = bool(cfg.include_extra_tree_model)
        cfg.classification.include_catboost_model = bool(cfg.include_catboost_model)
        cfg.classification.include_tabpfn_model = bool(cfg.include_tabpfn_model)


def _synthesize_dataset(dataset_id: str, seed: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
    rng = np.random.default_rng(seed)

    if dataset_id == "synthetic_easy_dfshift":
        X, y = make_classification(
            n_samples=260,
            n_features=700,
            n_informative=55,
            n_redundant=35,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=2.2,
            flip_y=0.01,
            weights=[0.52, 0.48],
            random_state=seed,
        )
        # Add class-conditional skew and heavy-tail shifts on a subset.
        for j in range(60):
            col = X[:, j]
            if j % 3 == 0:
                X[:, j] = np.exp(0.35 * col)
            elif j % 3 == 1:
                X[:, j] = np.sign(col) * np.abs(col) ** 1.5
            else:
                X[:, j] = col + 0.25 * sps_student_t_sample(rng, size=col.size, df=5)
        X[y == 1, :40] += 0.3
        return X.astype(float), y.astype(int), "synthetic_enhanced", "easy"

    if dataset_id == "synthetic_medium_mixed":
        X, y = make_classification(
            n_samples=320,
            n_features=1200,
            n_informative=80,
            n_redundant=45,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=1.1,
            flip_y=0.035,
            weights=[0.52, 0.30, 0.18],
            random_state=seed,
        )
        for j in range(120):
            col = X[:, j]
            if j % 4 == 0:
                X[:, j] = np.log1p(np.abs(col)) * np.sign(col)
            elif j % 4 == 1:
                X[:, j] = np.exp(0.2 * col)
            elif j % 4 == 2:
                X[:, j] = col + 0.35 * sps_student_t_sample(rng, size=col.size, df=4)
            else:
                X[:, j] = np.tanh(col) + 0.05 * rng.normal(size=col.size)

        # Inject mild contamination and rounding to stress DF.
        contam_mask = rng.random(X.shape) < 0.005
        X[contam_mask] += 6.0 * rng.normal(size=int(np.sum(contam_mask)))
        for j in range(20):
            X[:, j] = np.round(X[:, j] / 0.25) * 0.25

        return X.astype(float), y.astype(int), "synthetic_enhanced", "medium"

    if dataset_id == "synthetic_very_hard_sparse":
        X, y = make_classification(
            n_samples=180,
            n_features=1800,
            n_informative=45,
            n_redundant=30,
            n_classes=6,
            n_clusters_per_class=1,
            class_sep=0.75,
            flip_y=0.08,
            weights=[0.24, 0.20, 0.18, 0.15, 0.13, 0.10],
            random_state=seed,
        )
        for j in range(150):
            col = X[:, j]
            if j % 5 == 0:
                X[:, j] = np.exp(0.25 * col)
            elif j % 5 == 1:
                X[:, j] = np.sign(col) * np.sqrt(np.abs(col) + 1e-8)
            elif j % 5 == 2:
                X[:, j] = col + 0.45 * sps_student_t_sample(rng, size=col.size, df=3)
            elif j % 5 == 3:
                X[:, j] = np.round(col / 0.5) * 0.5
            else:
                X[:, j] = np.tanh(col) + 0.08 * rng.normal(size=col.size)

        # Heavier contamination in very hard synthetic.
        contam_mask = rng.random(X.shape) < 0.012
        X[contam_mask] += 10.0 * rng.normal(size=int(np.sum(contam_mask)))

        return X.astype(float), y.astype(int), "synthetic_enhanced", "very_hard"

    raise ValueError(f"Unknown synthetic dataset id: {dataset_id}")


def sps_student_t_sample(rng: np.random.Generator, size: int, df: int) -> np.ndarray:
    return rng.standard_t(df=df, size=size)


def _extract_source_batch_labels(loaded: Any) -> Tuple[Optional[np.ndarray], str]:
    """Best-effort extraction of source-provided batch labels from a loaded dataset object."""
    candidates = (
        "batch_labels",
        "batch",
        "batch_ids",
        "study_labels",
        "source_batches",
    )
    for attr in candidates:
        if not hasattr(loaded, attr):
            continue
        try:
            raw = getattr(loaded, attr)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=object).ravel()
            if arr.size <= 1:
                continue
            return arr, f"attr:{attr}"
        except Exception:
            continue
    return None, "source_batch_labels_unavailable"


def _extract_source_modality_membership(
    loaded: Any,
) -> Tuple[Optional[Dict[str, Tuple[int, ...]]], str]:
    """Best-effort extraction of source-provided feature-block metadata."""
    raw = getattr(loaded, "modality_membership", None)
    if not isinstance(raw, dict):
        return None, "source_modality_membership_unavailable"
    block_map: Dict[str, Tuple[int, ...]] = {}
    for raw_name, raw_indices in raw.items():
        name = str(raw_name).strip()
        if not name:
            continue
        try:
            idx_arr = np.asarray(list(raw_indices), dtype=int).ravel()
        except Exception:
            continue
        cleaned = tuple(sorted({int(idx) for idx in idx_arr.tolist() if int(idx) >= 0}))
        if cleaned:
            block_map[name] = cleaned
    if len(block_map) < 2:
        return None, "source_modality_membership_insufficient"
    return block_map, "source_modality_membership_ok"


def _derive_kmeans2_batch_labels(
    X: np.ndarray,
    *,
    seed: int,
) -> Tuple[Optional[np.ndarray], str]:
    """Derive pseudo-batch labels via 2-means clustering for ComBat policy testing."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        return None, "invalid_matrix_shape"
    n_samples = int(X_arr.shape[0])
    n_features = int(X_arr.shape[1]) if X_arr.ndim == 2 else 0
    if n_samples < 4:
        return None, "too_few_samples"
    if n_features == 0:
        return None, "no_features"

    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Keep clustering lightweight in very high-dimensional regimes.
    if X_arr.shape[1] > 256:
        X_work = X_arr[:, :256]
    else:
        X_work = X_arr

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(X_work)
        labels = KMeans(
            n_clusters=2,
            random_state=int(seed),
            n_init=10,
        ).fit_predict(X_scaled)
        labels_arr = np.asarray(labels, dtype=object).ravel()
        if int(np.unique(labels_arr).size) < 2:
            return None, "kmeans_single_cluster"
        return labels_arr, "kmeans2_ok"
    except Exception as exc:
        return None, f"kmeans2_error:{type(exc).__name__}"


def _load_dataset(
    spec: BenchmarkDatasetSpec,
    seed: int,
    allow_synthetic_fallback: bool,
    sample_cap: int,
    feature_cap: int,
    source_policy: Optional[str] = None,
    batch_label_policy: str = "none",
    dataset_integrity_policy: str = "error",
    dataset_min_classes: int = 2,
    dataset_min_class_count: int = 1,
) -> Tuple[np.ndarray, np.ndarray, str, str, Optional[np.ndarray], Dict[str, Any]]:
    policy = str(batch_label_policy or "none").strip().lower()
    if policy not in {"none", "source", "kmeans2"}:
        policy = "none"

    source_batch_labels: Optional[np.ndarray] = None
    source_batch_reason = "not_attempted"
    source_modality_membership: Optional[Dict[str, Tuple[int, ...]]] = None
    source_modality_reason = "not_attempted"
    if spec.source_kind == "synthetic":
        X, y, data_source, tier = _synthesize_dataset(spec.dataset_id, seed)
    elif spec.source_kind == "validation_catalog":
        if spec.validation_dataset_id is None:
            raise ValueError(f"Missing validation dataset id for {spec.dataset_id}")
        validation_spec = CATALOG[spec.validation_dataset_id]
        _require_hf_bundle_configuration(
            dataset_id=str(spec.validation_dataset_id),
            loader_kind=str(validation_spec.loader_kind),
        )
        if allow_synthetic_fallback:
            raise ValueError(
                f"dataset={spec.dataset_id} forbids synthetic fallback in benchmark/validation runs"
            )
        loaded = load_feature_selection_dataset(
            validation_spec,
            seed=seed,
            allow_synthetic_fallback=False,
            sample_cap=sample_cap,
            feature_cap=feature_cap,
            source_policy="real_only",
            class_integrity_policy=dataset_integrity_policy,
            class_min_classes=dataset_min_classes,
            class_min_class_count=dataset_min_class_count,
            require_hf_source=True,
        )
        data_source = str(loaded.data_source)
        tier = validation_spec.tier
        X = np.asarray(loaded.X, dtype=float)
        y = np.asarray(loaded.y).ravel()
        source_batch_labels, source_batch_reason = _extract_source_batch_labels(loaded)
        source_modality_membership, source_modality_reason = _extract_source_modality_membership(loaded)
    else:
        raise ValueError(f"Unknown source kind: {spec.source_kind}")

    selected_batch_labels: Optional[np.ndarray] = None
    selected_reason = "policy_none"
    if policy == "source":
        if source_batch_labels is None:
            selected_reason = source_batch_reason
        else:
            batch_arr = np.asarray(source_batch_labels, dtype=object).ravel()
            if int(batch_arr.size) != int(np.asarray(X).shape[0]):
                selected_reason = "source_size_mismatch"
            elif int(np.unique(batch_arr).size) < 2:
                selected_reason = "source_single_batch"
            else:
                selected_batch_labels = batch_arr
                selected_reason = "source_ok"
    elif policy == "kmeans2":
        selected_batch_labels, selected_reason = _derive_kmeans2_batch_labels(
            np.asarray(X, dtype=float),
            seed=int(seed),
        )

    batch_meta: Dict[str, Any] = {
        "batch_label_policy": str(policy),
        "batch_label_policy_reason": str(selected_reason),
        "batch_label_source_reason": str(source_batch_reason),
        "batch_labels_available": bool(selected_batch_labels is not None),
        "batch_labels_n_unique": int(
            np.unique(np.asarray(selected_batch_labels, dtype=object)).size
        )
        if selected_batch_labels is not None
        else 0,
        "multiomics_feature_blocks_available": bool(source_modality_membership is not None),
        "multiomics_feature_blocks_source_reason": str(source_modality_reason),
        "multiomics_feature_blocks": dict(source_modality_membership or {}),
    }
    return np.asarray(X, dtype=float), np.asarray(y).ravel(), str(data_source), str(tier), selected_batch_labels, batch_meta


def _resolve_dataset_list(
    dataset_sets: Sequence[str],
    datasets: Sequence[str],
    exclude: Sequence[str],
) -> List[str]:
    selected: List[str] = []
    for set_name in dataset_sets:
        if set_name not in DATASET_SETS:
            raise ValueError(f"Unknown dataset set: {set_name}")
        selected.extend(DATASET_SETS[set_name])

    selected.extend(datasets)
    if not selected:
        selected = list(DATASET_SETS["smoke"])

    seen = set()
    out: List[str] = []
    for ds_id in selected:
        if ds_id not in BENCHMARK_DATASETS:
            raise ValueError(f"Unknown dataset id: {ds_id}")
        if ds_id in seen or ds_id in set(exclude):
            continue
        seen.add(ds_id)
        out.append(ds_id)
    if not out:
        raise ValueError("No datasets selected after filtering.")
    return out


def _build_ablation_configs(base: DFFSConfig, profile: str) -> List[Tuple[str, DFFSConfig]]:
    if profile == "none":
        return [("baseline", base)]

    cfgs: List[Tuple[str, DFFSConfig]] = [("baseline", base)]

    def mk(name: str, **kwargs: Any) -> None:
        cfg = clone_config(base)
        for key, value in kwargs.items():
            apply_config_override(cfg, key, value)
        cfgs.append((name, cfg))

    mk("no_cdf_transform", apply_cdf_transform=False)
    mk("no_cdf_reliability_gate", cdf_reliability_gate=False)
    mk("no_low_gof_downweight", low_gof_downweighting=False)
    mk("df_no_support_filter", dist_config__use_support_filtering=False)
    mk("df_no_robust", dist_config__robust_mode=False)
    mk("df_no_lrt", dist_config__use_lrt=False)
    mk("fs_no_rank_prefilter", use_rank_prefilter=False)
    mk("fs_baseline_core", enabled_methods=("gradient_boosting", "linear_svm", "mutual_information", "anova_f"))
    if "wmw_auc" in base.enabled_methods:
        mk(
            "fs_no_wmw_auc",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "wmw_auc"),
        )
    else:
        mk(
            "fs_add_wmw_auc",
            enabled_methods=tuple(list(base.enabled_methods) + ["wmw_auc"]),
        )
    if "ktsp" in base.enabled_methods:
        mk("fs_no_ktsp", enabled_methods=tuple(m for m in base.enabled_methods if m != "ktsp"))
    else:
        mk("fs_add_ktsp", enabled_methods=tuple(list(base.enabled_methods) + ["ktsp"]))
    if "stability_subsample" in base.enabled_methods:
        mk(
            "fs_no_stability_subsample",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "stability_subsample"),
        )
        if base.fs_stability_use_loss_guided_validation:
            mk(
                "fs_disable_loss_guided_stability",
                fs_stability_use_loss_guided_validation=False,
            )
        else:
            mk(
                "fs_enable_loss_guided_stability",
                fs_stability_use_loss_guided_validation=True,
            )
    else:
        mk(
            "fs_add_stability_subsample",
            enabled_methods=tuple(list(base.enabled_methods) + ["stability_subsample"]),
        )
    rank_mode = str(getattr(base, "fs_rank_aggregation_mode", "none")).strip().lower()
    if rank_mode in {"borda", "rra"}:
        mk("fs_disable_rank_aggregation", fs_rank_aggregation_mode="none")
        alt_mode = "rra" if rank_mode == "borda" else "borda"
        mk(f"fs_switch_rank_aggregation_{alt_mode}", fs_rank_aggregation_mode=alt_mode)
    else:
        mk("fs_enable_rank_aggregation_borda", fs_rank_aggregation_mode="borda")
    if bool(base.fs_wrapper_refine_enabled):
        mk("fs_disable_wrapper_refine", fs_wrapper_refine_enabled=False)
    else:
        mk("fs_enable_wrapper_refine", fs_wrapper_refine_enabled=True)
    if "ova_ensemble" in base.enabled_methods:
        mk(
            "fs_no_ova_ensemble",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "ova_ensemble"),
        )
    else:
        mk(
            "fs_add_ova_ensemble",
            enabled_methods=tuple(list(base.enabled_methods) + ["ova_ensemble"]),
        )
    if "ecoc_class_aware" in base.enabled_methods:
        mk(
            "fs_no_ecoc_class_aware",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "ecoc_class_aware"),
        )
    else:
        mk(
            "fs_add_ecoc_class_aware",
            enabled_methods=tuple(list(base.enabled_methods) + ["ecoc_class_aware"]),
        )
    if "joint_multiclass_support" in base.enabled_methods:
        mk(
            "fs_no_joint_multiclass_support",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "joint_multiclass_support"),
        )
    else:
        mk(
            "fs_add_joint_multiclass_support",
            enabled_methods=tuple(list(base.enabled_methods) + ["joint_multiclass_support"]),
        )
    if "dove_class_specific" in base.enabled_methods:
        mk(
            "fs_no_dove_class_specific",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "dove_class_specific"),
        )
    else:
        mk(
            "fs_add_dove_class_specific",
            enabled_methods=tuple(list(base.enabled_methods) + ["dove_class_specific"]),
        )
    if "sparse_multinomial" in base.enabled_methods:
        mk(
            "fs_no_sparse_multinomial",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "sparse_multinomial"),
        )
    else:
        mk(
            "fs_add_sparse_multinomial",
            enabled_methods=tuple(list(base.enabled_methods) + ["sparse_multinomial"]),
        )
    if "nearest_shrunken_centroid" in base.enabled_methods:
        mk(
            "fs_no_nearest_shrunken_centroid",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "nearest_shrunken_centroid"),
        )
    else:
        mk(
            "fs_add_nearest_shrunken_centroid",
            enabled_methods=tuple(list(base.enabled_methods) + ["nearest_shrunken_centroid"]),
        )
    if "class_pareto_front" in base.enabled_methods:
        mk(
            "fs_no_class_pareto_front",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "class_pareto_front"),
        )
    else:
        mk(
            "fs_add_class_pareto_front",
            enabled_methods=tuple(list(base.enabled_methods) + ["class_pareto_front"]),
        )
    if "hsic_lasso" in base.enabled_methods:
        mk(
            "fs_no_hsic_lasso",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "hsic_lasso"),
        )
    else:
        mk(
            "fs_add_hsic_lasso",
            enabled_methods=tuple(list(base.enabled_methods) + ["hsic_lasso"]),
        )
    if "tigress_stability" in base.enabled_methods:
        mk(
            "fs_no_tigress_stability",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "tigress_stability"),
        )
    else:
        mk(
            "fs_add_tigress_stability",
            enabled_methods=tuple(list(base.enabled_methods) + ["tigress_stability"]),
        )
    if "ipss" in base.enabled_methods:
        mk(
            "fs_no_ipss",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "ipss"),
        )
    else:
        mk(
            "fs_add_ipss",
            enabled_methods=tuple(list(base.enabled_methods) + ["ipss"]),
        )
    if "cluster_stability" in base.enabled_methods:
        mk(
            "fs_no_cluster_stability",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "cluster_stability"),
        )
    else:
        mk(
            "fs_add_cluster_stability",
            enabled_methods=tuple(list(base.enabled_methods) + ["cluster_stability"]),
        )
    if "decorrelated_stability" in base.enabled_methods:
        mk(
            "fs_no_decorrelated_stability",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "decorrelated_stability"),
        )
    else:
        mk(
            "fs_add_decorrelated_stability",
            enabled_methods=tuple(list(base.enabled_methods) + ["decorrelated_stability"]),
        )
    if "subspace_stability" in base.enabled_methods:
        mk(
            "fs_no_subspace_stability",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "subspace_stability"),
        )
    else:
        mk(
            "fs_add_subspace_stability",
            enabled_methods=tuple(list(base.enabled_methods) + ["subspace_stability"]),
        )
    if "iterative_redundancy_pruning" in base.enabled_methods:
        mk(
            "fs_no_iterative_redundancy_pruning",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "iterative_redundancy_pruning"),
        )
    else:
        mk(
            "fs_add_iterative_redundancy_pruning",
            enabled_methods=tuple(list(base.enabled_methods) + ["iterative_redundancy_pruning"]),
        )
    if "iterative_redundancy_pruning_bounded" in base.enabled_methods:
        mk(
            "fs_no_iterative_redundancy_pruning_bounded",
            enabled_methods=tuple(m for m in base.enabled_methods if m != "iterative_redundancy_pruning_bounded"),
        )
        if bool(getattr(base, "fs_iterative_pruning_bounded_use_cpss_overlay", False)):
            mk(
                "fs_disable_iterative_pruning_bounded_cpss_overlay",
                fs_iterative_pruning_bounded_use_cpss_overlay=False,
            )
        else:
            mk(
                "fs_enable_iterative_pruning_bounded_cpss_overlay",
                fs_iterative_pruning_bounded_use_cpss_overlay=True,
            )
        if bool(getattr(base, "fs_iterative_pruning_class_pareto_prefilter_enabled", False)):
            mk(
                "fs_disable_iterative_pruning_class_pareto_prefilter",
                fs_iterative_pruning_class_pareto_prefilter_enabled=False,
            )
        else:
            mk(
                "fs_enable_iterative_pruning_class_pareto_prefilter",
                fs_iterative_pruning_class_pareto_prefilter_enabled=True,
            )
        if bool(getattr(base, "fs_iterative_pruning_class_pareto_stability_gate_enabled", False)):
            mk(
                "fs_disable_iterative_pruning_class_pareto_stability_gate",
                fs_iterative_pruning_class_pareto_stability_gate_enabled=False,
            )
        else:
            mk(
                "fs_enable_iterative_pruning_class_pareto_stability_gate",
                fs_iterative_pruning_class_pareto_stability_gate_enabled=True,
            )
    else:
        mk(
            "fs_add_iterative_redundancy_pruning_bounded",
            enabled_methods=tuple(list(base.enabled_methods) + ["iterative_redundancy_pruning_bounded"]),
        )
    if "copula_knockoff" in base.enabled_methods:
        if base.fs_copula_stabilizer_runs > 1 or base.fs_copula_stabilizer_use_ebh:
            mk(
                "fs_disable_copula_stabilizer",
                fs_copula_stabilizer_runs=1,
                fs_copula_stabilizer_use_ebh=False,
            )
        else:
            mk(
                "fs_enable_copula_stabilizer",
                fs_copula_stabilizer_runs=3,
                fs_copula_stabilizer_use_ebh=True,
            )
    if bool(getattr(base, "fs_runtime_racing_enabled", False)):
        mk("fs_disable_runtime_racing", fs_runtime_racing_enabled=False)
    else:
        mk("fs_enable_runtime_racing", fs_runtime_racing_enabled=True)
    mk("fs_no_tritrust", use_tritrust=False)

    if profile == "full":
        mk("no_df_block", apply_cdf_transform=False, low_gof_downweighting=False)
        mk(
            "minimal_stack",
            apply_cdf_transform=False,
            use_rank_prefilter=False,
            enabled_methods=("gradient_boosting", "linear_svm", "mutual_information", "anova_f"),
            use_tritrust=False,
        )

    return cfgs


def clone_config(cfg: DFFSConfig) -> DFFSConfig:
    dist = DistributionFitterConfig(
        robust_mode=cfg.dist_config.robust_mode,
        use_adaptive_strategy=cfg.dist_config.use_adaptive_strategy,
        use_lrt=cfg.dist_config.use_lrt,
        use_cv=cfg.dist_config.use_cv,
        compute_budget=cfg.dist_config.compute_budget,
        use_support_filtering=cfg.dist_config.use_support_filtering,
        rejection_gate=cfg.dist_config.rejection_gate,
        rejection_p_threshold=cfg.dist_config.rejection_p_threshold,
        confidence_margin=cfg.dist_config.confidence_margin,
        family_set=str(getattr(cfg.dist_config, "family_set", "v6") or "v6"),
        compute_ad=bool(getattr(cfg.dist_config, "compute_ad", False)),
        ad_bootstrap_samples=int(getattr(cfg.dist_config, "ad_bootstrap_samples", 0) or 0),
        compute_qq_pp=bool(getattr(cfg.dist_config, "compute_qq_pp", False)),
        compute_dip=bool(getattr(cfg.dist_config, "compute_dip", True)),
        dip_hist_bins=int(getattr(cfg.dist_config, "dip_hist_bins", 40) or 40),
        interval_likelihood=bool(getattr(cfg.dist_config, "interval_likelihood", False)),
        interval_delta_override=float(getattr(cfg.dist_config, "interval_delta_override", 0.0) or 0.0),
        use_lmoment_prescreen=bool(getattr(cfg.dist_config, "use_lmoment_prescreen", False)),
        lmoment_prescreen_max_candidates=int(
            getattr(cfg.dist_config, "lmoment_prescreen_max_candidates", 0) or 0
        ),
        estimator=str(getattr(cfg.dist_config, "estimator", "mle") or "mle"),
        mps_maxiter=int(getattr(cfg.dist_config, "mps_maxiter", 250) or 250),
        mps_tol=float(getattr(cfg.dist_config, "mps_tol", 1e-6) or 1e-6),
        compute_crps=bool(getattr(cfg.dist_config, "compute_crps", False)),
        crps_mc_samples=int(getattr(cfg.dist_config, "crps_mc_samples", 96) or 96),
        crps_data_subsample=int(getattr(cfg.dist_config, "crps_data_subsample", 256) or 256),
        compute_crps_uq_decomposition=bool(getattr(cfg.dist_config, "compute_crps_uq_decomposition", False)),
        mnpo_use_tritrust=bool(getattr(cfg.dist_config, "mnpo_use_tritrust", True)),
        mnpo_include_crps=bool(getattr(cfg.dist_config, "mnpo_include_crps", False)),
        mnpo_include_preq=bool(getattr(cfg.dist_config, "mnpo_include_preq", False)),
        mnpo_use_tail_risk_oracle=False,
        mnpo_tail_risk_alpha=float(getattr(cfg.dist_config, "mnpo_tail_risk_alpha", 0.33) or 0.33),
        mnpo_use_qre_smoothing=bool(getattr(cfg.dist_config, "mnpo_use_qre_smoothing", False)),
        mnpo_qre_temperature_gamma=float(getattr(cfg.dist_config, "mnpo_qre_temperature_gamma", 1.0) or 1.0),
        mnpo_use_oracle_redundancy_penalty=bool(getattr(cfg.dist_config, "mnpo_use_oracle_redundancy_penalty", False)),
        mnpo_compute_tremble_sensitivity=bool(getattr(cfg.dist_config, "mnpo_compute_tremble_sensitivity", False)),
        preq_holdout_fraction=float(getattr(cfg.dist_config, "preq_holdout_fraction", 0.20) or 0.20),
        preq_min_train=int(getattr(cfg.dist_config, "preq_min_train", 20) or 20),
        preq_max_test_points=int(getattr(cfg.dist_config, "preq_max_test_points", 128) or 128),
        random_state=getattr(cfg.dist_config, "random_state", None),
    )
    return DFFSConfig(
        random_seed=cfg.random_seed,
        test_size=cfg.test_size,
        max_train_samples=cfg.max_train_samples,
        fs_fraction=cfg.fs_fraction,
        n_final_features=cfg.n_final_features,
        classification=ClassificationConfig(**vars(cfg.classification)),
        enable_ratio_features=bool(getattr(cfg, "enable_ratio_features", False)),
        ratio_pool_size=int(getattr(cfg, "ratio_pool_size", 0) or 0),
        ratio_selection_method=str(getattr(cfg, "ratio_selection_method", "ktsp") or "ktsp"),
        ratio_max_pairs=int(getattr(cfg, "ratio_max_pairs", 0) or 0),
        max_ratio_features=int(getattr(cfg, "max_ratio_features", 0) or 0),
        ratio_epsilon=float(getattr(cfg, "ratio_epsilon", 1e-6) or 1e-6),
        ratio_include_originals=bool(getattr(cfg, "ratio_include_originals", True)),
        ratio_abs_value=bool(getattr(cfg, "ratio_abs_value", False)),
        ratio_require_positive=bool(getattr(cfg, "ratio_require_positive", True)),
        multiomics_adapter=str(getattr(cfg, "multiomics_adapter", "none") or "none"),
        multiomics_integrator=str(
            getattr(cfg, "multiomics_integrator", "mb_plsda") or "mb_plsda"
        ),
        multiomics_n_components=int(getattr(cfg, "multiomics_n_components", 2) or 2),
        dist_config=dist,
        dist_criterion=cfg.dist_criterion,
        apply_cdf_transform=cfg.apply_cdf_transform,
        df_stage_position=str(getattr(cfg, "df_stage_position", "after_fs") or "after_fs"),
        cdf_reliability_gate=cfg.cdf_reliability_gate,
        cdf_min_gof_p=cfg.cdf_min_gof_p,
        cdf_max_confidence_set=cfg.cdf_max_confidence_set,
        cdf_skip_heaped_features=cfg.cdf_skip_heaped_features,
        cdf_block_gating_cv=cfg.cdf_block_gating_cv,
        cdf_block_gating_n_blocks=cfg.cdf_block_gating_n_blocks,
        cdf_block_gating_min_block_size=cfg.cdf_block_gating_min_block_size,
        cdf_block_gating_cv_splits=cfg.cdf_block_gating_cv_splits,
        cdf_block_gating_max_blocks=cfg.cdf_block_gating_max_blocks,
        cdf_block_gating_time_budget_sec=cfg.cdf_block_gating_time_budget_sec,
        cdf_block_gating_min_improvement=cfg.cdf_block_gating_min_improvement,
        multimodal_fallback=str(getattr(cfg, "multimodal_fallback", "gmm") or "gmm"),
        max_dist_features=cfg.max_dist_features,
        low_gof_downweighting=cfg.low_gof_downweighting,
        low_gof_threshold=cfg.low_gof_threshold,
        low_gof_weight=cfg.low_gof_weight,
        use_distribution_stability_weight=cfg.use_distribution_stability_weight,
        stability_bootstrap=cfg.stability_bootstrap,
        use_rank_prefilter=cfg.use_rank_prefilter,
        prefilter_top_k=cfg.prefilter_top_k,
        prefilter_mi_weight=float(getattr(cfg, "prefilter_mi_weight", 0.60) or 0.60),
        prefilter_f_weight=float(getattr(cfg, "prefilter_f_weight", 0.40) or 0.40),
        prefilter_union_enabled=bool(getattr(cfg, "prefilter_union_enabled", False)),
        prefilter_strategies=tuple(
            getattr(cfg, "prefilter_strategies", ("mi_ftest_blend",)) or ("mi_ftest_blend",)
        ),
        prefilter_nondefault_budget_fraction=float(
            getattr(cfg, "prefilter_nondefault_budget_fraction", 0.10) or 0.10
        ),
        prefilter_wsnr_enabled=bool(getattr(cfg, "prefilter_wsnr_enabled", False)),
        prefilter_bh_ttest_enabled=bool(getattr(cfg, "prefilter_bh_ttest_enabled", True)),
        prefilter_bh_ttest_alpha=float(getattr(cfg, "prefilter_bh_ttest_alpha", 0.05) or 0.05),
        prefilter_data_domain=str(getattr(cfg, "prefilter_data_domain", "auto") or "auto"),
        prefilter_rnaseq_transform_enabled=bool(
            getattr(cfg, "prefilter_rnaseq_transform_enabled", True)
        ),
        prefilter_rnaseq_transform_force=bool(
            getattr(cfg, "prefilter_rnaseq_transform_force", False)
        ),
        prefilter_rnaseq_nb_lrt_enabled=bool(
            getattr(cfg, "prefilter_rnaseq_nb_lrt_enabled", False)
        ),
        prefilter_rnaseq_nb_lrt_alpha=float(
            getattr(cfg, "prefilter_rnaseq_nb_lrt_alpha", 0.10) or 0.10
        ),
        batch_correction=str(getattr(cfg, "batch_correction", "none") or "none"),
        batch_correction_combat_prior_strength=float(
            getattr(cfg, "batch_correction_combat_prior_strength", 8.0) or 8.0
        ),
        batch_correction_cdf_n_quantiles=int(
            getattr(cfg, "batch_correction_cdf_n_quantiles", 33) or 33
        ),
        batch_correction_cdf_clip_low=float(
            getattr(cfg, "batch_correction_cdf_clip_low", 0.01) or 0.01
        ),
        batch_correction_cdf_clip_high=float(
            getattr(cfg, "batch_correction_cdf_clip_high", 0.99) or 0.99
        ),
        screening_enabled=bool(getattr(cfg, "screening_enabled", False)),
        screening_method=str(getattr(cfg, "screening_method", "none") or "none"),
        screening_pool_cap=int(getattr(cfg, "screening_pool_cap", 2000) or 2000),
        screening_stir_n_neighbors=int(getattr(cfg, "screening_stir_n_neighbors", 10) or 10),
        screening_stir_n_iter=int(getattr(cfg, "screening_stir_n_iter", 50) or 50),
        screening_stir_keep_fraction=float(
            getattr(cfg, "screening_stir_keep_fraction", 0.5) or 0.5
        ),
        screening_stir_min_features=int(getattr(cfg, "screening_stir_min_features", 20) or 20),
        screening_evalue_alpha=float(getattr(cfg, "screening_evalue_alpha", 0.20) or 0.20),
        screening_evalue_min_features=int(
            getattr(cfg, "screening_evalue_min_features", 20) or 20
        ),
        folding_method=cfg.folding_method,
        folding_n_components=cfg.folding_n_components,
        folding_rff_gamma=cfg.folding_rff_gamma,
        folding_pls_components=cfg.folding_pls_components,
        folding_pls_scale=cfg.folding_pls_scale,
        folding_pls_min_classes=cfg.folding_pls_min_classes,
        folding_pls_min_n_per_class=int(getattr(cfg, "folding_pls_min_n_per_class", 3) or 3),
        folding_pls_max_imbalance_ratio=float(
            getattr(cfg, "folding_pls_max_imbalance_ratio", 6.0) or 6.0
        ),
        folding_prefilter_k=cfg.folding_prefilter_k,
        enable_face_domain_projection=cfg.enable_face_domain_projection,
        selection_strategy=str(
            getattr(cfg, "selection_strategy", "mnpo_portfolio") or "mnpo_portfolio"
        ),
        use_balanced_fs_subsample=cfg.use_balanced_fs_subsample,
        fs_min_per_class=cfg.fs_min_per_class,
        fs_method_timeout_seconds=cfg.fs_method_timeout_seconds,
        fs_linear_svm_max_iter=cfg.fs_linear_svm_max_iter,
        fs_runtime_racing_enabled=cfg.fs_runtime_racing_enabled,
        fs_runtime_racing_proxy_splits=cfg.fs_runtime_racing_proxy_splits,
        fs_runtime_racing_keep_fraction=cfg.fs_runtime_racing_keep_fraction,
        fs_runtime_racing_min_candidates=cfg.fs_runtime_racing_min_candidates,
        fs_runtime_racing_runtime_weight=cfg.fs_runtime_racing_runtime_weight,
        fs_runtime_racing_mode=cfg.fs_runtime_racing_mode,
        fs_runtime_racing_stages=cfg.fs_runtime_racing_stages,
        fs_runtime_racing_confidence_bound=cfg.fs_runtime_racing_confidence_bound,
        fs_runtime_racing_delta=cfg.fs_runtime_racing_delta,
        fs_portfolio_size=int(getattr(cfg, "fs_portfolio_size", 6) or 6),
        fs_portfolio_size_guard=str(getattr(cfg, "fs_portfolio_size_guard", "none") or "none"),
        fs_adaptive_portfolio_sizing_enabled=bool(
            getattr(cfg, "fs_adaptive_portfolio_sizing_enabled", False)
        ),
        fs_adaptive_size_min=getattr(cfg, "fs_adaptive_size_min", None),
        fs_adaptive_size_max=getattr(cfg, "fs_adaptive_size_max", None),
        fs_adaptive_sizing_variance_penalty=bool(
            getattr(cfg, "fs_adaptive_sizing_variance_penalty", False)
        ),
        fs_adaptive_sizing_variance_penalty_strength=float(
            getattr(cfg, "fs_adaptive_sizing_variance_penalty_strength", 0.5) or 0.5
        ),
        fs_mnpo_paradigm_aware_prior_enabled=bool(
            getattr(cfg, "fs_mnpo_paradigm_aware_prior_enabled", False)
        ),
        fs_mnpo_interaction_floor=float(getattr(cfg, "fs_mnpo_interaction_floor", 0.12) or 0.12),
        fs_rashomon_enabled=bool(getattr(cfg, "fs_rashomon_enabled", False)),
        fs_rashomon_max_models=int(getattr(cfg, "fs_rashomon_max_models", 12) or 12),
        fs_rashomon_score_tolerance=float(
            getattr(cfg, "fs_rashomon_score_tolerance", 0.01) or 0.01
        ),
        enabled_methods=tuple(cfg.enabled_methods),
        fs_inner_cv_splits=int(getattr(cfg, "fs_inner_cv_splits", 3) or 3),
        fs_inner_cv_repeats=int(getattr(cfg, "fs_inner_cv_repeats", 1) or 1),
        use_tritrust=cfg.use_tritrust,
        use_stability_oracle=cfg.use_stability_oracle,
        use_complexity_oracle=cfg.use_complexity_oracle,
        use_robust_oracle=cfg.use_robust_oracle,
        use_diversity_oracle=cfg.use_diversity_oracle,
        fs_use_cvar_oracle=bool(getattr(cfg, "fs_use_cvar_oracle", False)),
        fs_cvar_alpha=float(getattr(cfg, "fs_cvar_alpha", 0.33) or 0.33),
        fs_oracle_weighting_mode=str(
            getattr(cfg, "fs_oracle_weighting_mode", "tritrust") or "tritrust"
        ),
        fs_shapley_n_coalitions_max=int(
            getattr(cfg, "fs_shapley_n_coalitions_max", 4096) or 4096
        ),
        fs_shapley_bayesian_shrinkage=bool(
            getattr(cfg, "fs_shapley_bayesian_shrinkage", False)
        ),
        fs_shapley_bayesian_prior_strength=float(
            getattr(cfg, "fs_shapley_bayesian_prior_strength", 8.0) or 8.0
        ),
        fs_use_interaction_oracle=bool(getattr(cfg, "fs_use_interaction_oracle", False)),
        fs_interaction_oracle_min_n_train=int(
            getattr(cfg, "fs_interaction_oracle_min_n_train", 150) or 150
        ),
        fs_interaction_oracle_pool_size_cap=int(
            getattr(cfg, "fs_interaction_oracle_pool_size_cap", 64) or 64
        ),
        fs_interaction_oracle_pair_cap=int(
            getattr(cfg, "fs_interaction_oracle_pair_cap", 20000) or 20000
        ),
        fs_use_ubayfs_oracle=bool(getattr(cfg, "fs_use_ubayfs_oracle", False)),
        fs_ubayfs_n_bootstrap=int(getattr(cfg, "fs_ubayfs_n_bootstrap", 32) or 32),
        fs_ubayfs_min_n=int(getattr(cfg, "fs_ubayfs_min_n", 100) or 100),
        fs_ubayfs_prior_weight=float(getattr(cfg, "fs_ubayfs_prior_weight", 0.0) or 0.0),
        fs_use_conformal_uq=bool(getattr(cfg, "fs_use_conformal_uq", False)),
        fs_conformal_uq_alpha=float(getattr(cfg, "fs_conformal_uq_alpha", 0.10) or 0.10),
        fs_conformal_uq_min_folds=int(getattr(cfg, "fs_conformal_uq_min_folds", 5) or 5),
        fs_fold_preference_mode=str(
            getattr(cfg, "fs_fold_preference_mode", "vote") or "vote"
        ),
        fs_use_conformal_efficiency=bool(
            getattr(cfg, "fs_use_conformal_efficiency", False)
        ),
        fs_conformal_efficiency_method=str(
            getattr(cfg, "fs_conformal_efficiency_method", "split") or "split"
        ),
        fs_oracle_weight_js_shrinkage=bool(
            getattr(cfg, "fs_oracle_weight_js_shrinkage", False)
        ),
        fs_payoff_shrinkage_kappa=float(
            getattr(cfg, "fs_payoff_shrinkage_kappa", 0.0) or 0.0
        ),
        use_tail_risk_oracle=False,
        tail_risk_alpha=float(getattr(cfg, "tail_risk_alpha", 0.33) or 0.33),
        use_regret_oracle=False,
        use_qre_smoothing=bool(getattr(cfg, "use_qre_smoothing", False)),
        qre_temperature_gamma=float(getattr(cfg, "qre_temperature_gamma", 1.0) or 1.0),
        use_oracle_redundancy_penalty=bool(getattr(cfg, "use_oracle_redundancy_penalty", False)),
        compute_tremble_sensitivity=bool(getattr(cfg, "compute_tremble_sensitivity", False)),
        fs_diversity_oracle_mode=cfg.fs_diversity_oracle_mode,
        fs_diversity_redundancy_weight=cfg.fs_diversity_redundancy_weight,
        fs_diversity_complementarity_weight=cfg.fs_diversity_complementarity_weight,
        fs_performance_balanced_weight=cfg.fs_performance_balanced_weight,
        fs_performance_macro_f1_weight=cfg.fs_performance_macro_f1_weight,
        fs_performance_use_adaptive_imbalance=cfg.fs_performance_use_adaptive_imbalance,
        fs_performance_imbalance_ratio_trigger=cfg.fs_performance_imbalance_ratio_trigger,
        fs_performance_min_classes_for_adaptive=cfg.fs_performance_min_classes_for_adaptive,
        eval_models_enabled=bool(getattr(cfg, "eval_models_enabled", False)),
        eval_models=tuple(getattr(cfg, "eval_models", ("lr_l2", "linear_svc", "rf_small")) or ()),
        eval_aggregate=str(getattr(cfg, "eval_aggregate", "mean") or "mean"),
        eval_cvar_alpha=float(getattr(cfg, "eval_cvar_alpha", 0.33) or 0.33),
        tier_lockout_enabled=bool(getattr(cfg, "tier_lockout_enabled", False)),
        tier_classifier_mode=str(
            getattr(cfg, "tier_classifier_mode", "heuristic") or "heuristic"
        ).strip().lower(),
        tier_classifier_model_path=str(
            getattr(cfg, "tier_classifier_model_path", "") or ""
        ).strip(),
        tier_lockout_tier=str(getattr(cfg, "tier_lockout_tier", "easy") or "easy").strip().lower(),
        tier_lockout_difficulty_source=str(
            getattr(cfg, "tier_lockout_difficulty_source", "historical") or "historical"
        ).strip().lower(),
        tier_lockout_fallback_methods=tuple(
            getattr(cfg, "tier_lockout_fallback_methods", tuple()) or tuple()
        ),
        tier_routing_enabled=bool(getattr(cfg, "tier_routing_enabled", False)),
        tier_routing_difficulty_classifier=str(
            getattr(cfg, "tier_routing_difficulty_classifier", "meta_features") or "meta_features"
        ).strip().lower(),
        tier_routing_table={
            str(k): tuple(v) if isinstance(v, (list, tuple, set)) else tuple(_parse_csv_or_space_list(v))
            for k, v in (getattr(cfg, "tier_routing_table", {}) or {}).items()
        },
        mnpo_performance_oracle_mode=str(
            getattr(cfg, "mnpo_performance_oracle_mode", "single") or "single"
        ).strip().lower(),
        fs_rank_aggregation_mode=cfg.fs_rank_aggregation_mode,
        fs_wrapper_refine_enabled=cfg.fs_wrapper_refine_enabled,
        fs_wrapper_refine_top_k=cfg.fs_wrapper_refine_top_k,
        fs_wrapper_refine_max_add=cfg.fs_wrapper_refine_max_add,
        fs_wrapper_refine_min_gain=cfg.fs_wrapper_refine_min_gain,
        fs_ova_negative_ratio=cfg.fs_ova_negative_ratio,
        fs_ova_min_classes=cfg.fs_ova_min_classes,
        fs_ova_min_pos_samples=cfg.fs_ova_min_pos_samples,
        fs_ova_class_weight_mode=cfg.fs_ova_class_weight_mode,
        fs_ova_aggregation_mode=cfg.fs_ova_aggregation_mode,
        fs_ova_aggregation_p=cfg.fs_ova_aggregation_p,
        fs_ova_linear_backend=cfg.fs_ova_linear_backend,
        fs_ova_enable_calibration=cfg.fs_ova_enable_calibration,
        fs_ova_calibration_cv=cfg.fs_ova_calibration_cv,
        fs_ecoc_min_classes=cfg.fs_ecoc_min_classes,
        fs_ecoc_max_ovo_pairs=cfg.fs_ecoc_max_ovo_pairs,
        fs_ecoc_random_code_bits=cfg.fs_ecoc_random_code_bits,
        fs_ecoc_class_complexity_weight=cfg.fs_ecoc_class_complexity_weight,
        fs_ecoc_include_ova_tasks=cfg.fs_ecoc_include_ova_tasks,
        fs_ecoc_negative_ratio=cfg.fs_ecoc_negative_ratio,
        fs_joint_multiclass_min_classes=cfg.fs_joint_multiclass_min_classes,
        fs_joint_multiclass_max_features=cfg.fs_joint_multiclass_max_features,
        fs_joint_multiclass_path_grid_size=cfg.fs_joint_multiclass_path_grid_size,
        fs_joint_multiclass_min_c=cfg.fs_joint_multiclass_min_c,
        fs_joint_multiclass_max_c=cfg.fs_joint_multiclass_max_c,
        fs_joint_multiclass_l1_ratio=cfg.fs_joint_multiclass_l1_ratio,
        fs_joint_multiclass_univariate_blend=cfg.fs_joint_multiclass_univariate_blend,
        fs_dove_min_classes=cfg.fs_dove_min_classes,
        fs_dove_max_pairs_per_class=cfg.fs_dove_max_pairs_per_class,
        fs_dove_path_grid_size=cfg.fs_dove_path_grid_size,
        fs_dove_specificity_weight=cfg.fs_dove_specificity_weight,
        fs_dove_minority_boost=cfg.fs_dove_minority_boost,
        fs_sparse_multinomial_min_classes=cfg.fs_sparse_multinomial_min_classes,
        fs_sparse_multinomial_max_features=cfg.fs_sparse_multinomial_max_features,
        fs_sparse_multinomial_path_grid_size=cfg.fs_sparse_multinomial_path_grid_size,
        fs_sparse_multinomial_min_c=cfg.fs_sparse_multinomial_min_c,
        fs_sparse_multinomial_max_c=cfg.fs_sparse_multinomial_max_c,
        fs_sparse_multinomial_backend=cfg.fs_sparse_multinomial_backend,
        fs_sparse_multinomial_l1_ratio=cfg.fs_sparse_multinomial_l1_ratio,
        fs_sparse_multinomial_univariate_blend=cfg.fs_sparse_multinomial_univariate_blend,
        fs_sparse_multinomial_max_iter=cfg.fs_sparse_multinomial_max_iter,
        fs_sparse_multinomial_screening_mode=cfg.fs_sparse_multinomial_screening_mode,
        fs_sparse_multinomial_screening_keep_fraction=cfg.fs_sparse_multinomial_screening_keep_fraction,
        fs_sparse_multinomial_screening_min_features=cfg.fs_sparse_multinomial_screening_min_features,
        fs_sparse_multinomial_screening_fallback_on_failure=cfg.fs_sparse_multinomial_screening_fallback_on_failure,
        fs_nsc_shrinkage_grid_size=cfg.fs_nsc_shrinkage_grid_size,
        fs_nsc_min_classes=cfg.fs_nsc_min_classes,
        fs_nsc_thresholding_mode=cfg.fs_nsc_thresholding_mode,
        fs_nsc_order_quantile=cfg.fs_nsc_order_quantile,
        fs_nsc_deep_shrinkage_search=cfg.fs_nsc_deep_shrinkage_search,
        fs_class_pareto_min_classes=cfg.fs_class_pareto_min_classes,
        fs_class_pareto_top_per_class=cfg.fs_class_pareto_top_per_class,
        fs_class_pareto_global_fraction=cfg.fs_class_pareto_global_fraction,
        fs_class_pareto_minority_boost=cfg.fs_class_pareto_minority_boost,
        fs_class_pareto_kw_weight=cfg.fs_class_pareto_kw_weight,
        fs_sdr_min_classes=int(getattr(cfg, "fs_sdr_min_classes", 3) or 3),
        fs_sdr_prefilter_max_features=int(
            getattr(cfg, "fs_sdr_prefilter_max_features", 512) or 512
        ),
        fs_sdr_n_components=int(getattr(cfg, "fs_sdr_n_components", 3) or 3),
        fs_sdr_covariance_ridge=float(
            getattr(cfg, "fs_sdr_covariance_ridge", 1e-3) or 1e-3
        ),
        fs_per_class_quota_enabled=cfg.fs_per_class_quota_enabled,
        fs_per_class_quota_min_per_class=cfg.fs_per_class_quota_min_per_class,
        fs_per_class_quota_max_fraction=cfg.fs_per_class_quota_max_fraction,
        fs_hsic_lasso_alpha=cfg.fs_hsic_lasso_alpha,
        fs_hsic_lasso_prefilter_max_features=cfg.fs_hsic_lasso_prefilter_max_features,
        fs_hsic_lasso_feature_sigma=cfg.fs_hsic_lasso_feature_sigma,
        fs_hsic_lasso_target_sigma=cfg.fs_hsic_lasso_target_sigma,
        fs_hsic_lasso_relevance_blend=cfg.fs_hsic_lasso_relevance_blend,
        fs_hsic_lasso_max_iter=cfg.fs_hsic_lasso_max_iter,
        fs_mrmr_mi_redundancy_enabled=bool(
            getattr(cfg, "fs_mrmr_mi_redundancy_enabled", False)
        ),
        fs_mrmr_mi_n_bins=int(getattr(cfg, "fs_mrmr_mi_n_bins", 8) or 8),
        fs_cmim_min_samples=int(getattr(cfg, "fs_cmim_min_samples", 60) or 60),
        fs_cmim_n_bins=int(getattr(cfg, "fs_cmim_n_bins", 8) or 8),
        fs_fcbf_n_bins=int(getattr(cfg, "fs_fcbf_n_bins", 8) or 8),
        fs_ipss_path_grid_size=cfg.fs_ipss_path_grid_size,
        fs_ipss_min_c=cfg.fs_ipss_min_c,
        fs_ipss_max_c=cfg.fs_ipss_max_c,
        fs_ipss_target_fdr=cfg.fs_ipss_target_fdr,
        fs_ipss_null_shuffle_rounds=cfg.fs_ipss_null_shuffle_rounds,
        fs_ipss_use_eats_threshold=cfg.fs_ipss_use_eats_threshold,
        fs_ipss_eats_exclusion_quantile=cfg.fs_ipss_eats_exclusion_quantile,
        fs_ipss_eats_min_threshold=cfg.fs_ipss_eats_min_threshold,
        fs_ipss_importance_model=cfg.fs_ipss_importance_model,
        fs_cluster_stability_corr_threshold=cfg.fs_cluster_stability_corr_threshold,
        fs_cluster_stability_max_per_cluster=cfg.fs_cluster_stability_max_per_cluster,
        fs_cluster_stability_min_cluster_freq=cfg.fs_cluster_stability_min_cluster_freq,
        fs_stability_threshold_method=str(
            getattr(cfg, "fs_stability_threshold_method", "fixed") or "fixed"
        ),
        fs_stability_target_pfer=float(
            getattr(cfg, "fs_stability_target_pfer", 1.0) or 1.0
        ),
        fs_stability_use_loss_guided_validation=cfg.fs_stability_use_loss_guided_validation,
        fs_stability_validation_fraction=cfg.fs_stability_validation_fraction,
        fs_stability_validation_quantile=cfg.fs_stability_validation_quantile,
        fs_stability_validation_min_samples=cfg.fs_stability_validation_min_samples,
        fs_copula_knockoff_draws=cfg.fs_copula_knockoff_draws,
        fs_copula_alpha_kn=cfg.fs_copula_alpha_kn,
        fs_copula_alpha_ebh=cfg.fs_copula_alpha_ebh,
        fs_copula_truncation_level=cfg.fs_copula_truncation_level,
        fs_copula_generator=str(getattr(cfg, "fs_copula_generator", "copula") or "copula"),
        fs_copula_deepdrk_latent_fraction=float(
            getattr(cfg, "fs_copula_deepdrk_latent_fraction", 0.35) or 0.35
        ),
        fs_copula_deepdrk_noise_scale=float(
            getattr(cfg, "fs_copula_deepdrk_noise_scale", 1.0) or 1.0
        ),
        fs_copula_derandomize_runs=int(getattr(cfg, "fs_copula_derandomize_runs", 5) or 5),
        fs_copula_stabilizer_runs=cfg.fs_copula_stabilizer_runs,
        fs_copula_stabilizer_use_ebh=cfg.fs_copula_stabilizer_use_ebh,
        fs_copula_stabilizer_seed_stride=cfg.fs_copula_stabilizer_seed_stride,
        fs_importance_uq_enabled=bool(getattr(cfg, "fs_importance_uq_enabled", False)),
        fs_importance_uq_min_cv_folds=int(getattr(cfg, "fs_importance_uq_min_cv_folds", 3) or 3),
        fs_decorrelated_stability_eps=cfg.fs_decorrelated_stability_eps,
        fs_decorrelated_stability_min_max_abs_corr=cfg.fs_decorrelated_stability_min_max_abs_corr,
        fs_iterative_pruning_pool_factor=cfg.fs_iterative_pruning_pool_factor,
        fs_iterative_pruning_max_rounds=cfg.fs_iterative_pruning_max_rounds,
        fs_iterative_pruning_min_improvement=cfg.fs_iterative_pruning_min_improvement,
        fs_iterative_pruning_max_cumulative_loss=getattr(cfg, "fs_iterative_pruning_max_cumulative_loss", 0.02),
        fs_iterative_pruning_redundancy_weight=cfg.fs_iterative_pruning_redundancy_weight,
        fs_iterative_pruning_bounded_prefilter_cap=cfg.fs_iterative_pruning_bounded_prefilter_cap,
        fs_iterative_pruning_bounded_candidate_fraction=cfg.fs_iterative_pruning_bounded_candidate_fraction,
        fs_iterative_pruning_bounded_min_candidates=cfg.fs_iterative_pruning_bounded_min_candidates,
        fs_iterative_pruning_bounded_max_evaluations=cfg.fs_iterative_pruning_bounded_max_evaluations,
        fs_iterative_pruning_bounded_max_runtime_seconds=cfg.fs_iterative_pruning_bounded_max_runtime_seconds,
        fs_iterative_pruning_bounded_enable_class_gating=cfg.fs_iterative_pruning_bounded_enable_class_gating,
        fs_iterative_pruning_bounded_multiclass_scale=cfg.fs_iterative_pruning_bounded_multiclass_scale,
        fs_iterative_pruning_bounded_imbalance_trigger=cfg.fs_iterative_pruning_bounded_imbalance_trigger,
        fs_iterative_pruning_bounded_imbalance_scale=cfg.fs_iterative_pruning_bounded_imbalance_scale,
        fs_iterative_pruning_bounded_use_cpss_overlay=cfg.fs_iterative_pruning_bounded_use_cpss_overlay,
        fs_iterative_pruning_bounded_cpss_pairs=cfg.fs_iterative_pruning_bounded_cpss_pairs,
        fs_iterative_pruning_bounded_cpss_stability_threshold=cfg.fs_iterative_pruning_bounded_cpss_stability_threshold,
        fs_iterative_pruning_bounded_cpss_min_stable_features=cfg.fs_iterative_pruning_bounded_cpss_min_stable_features,
        fs_iterative_pruning_bounded_cpss_min_jaccard=cfg.fs_iterative_pruning_bounded_cpss_min_jaccard,
        fs_iterative_pruning_bounded_cpss_max_score_drop=cfg.fs_iterative_pruning_bounded_cpss_max_score_drop,
        fs_iterative_pruning_class_pareto_prefilter_enabled=cfg.fs_iterative_pruning_class_pareto_prefilter_enabled,
        fs_iterative_pruning_class_pareto_min_classes=cfg.fs_iterative_pruning_class_pareto_min_classes,
        fs_iterative_pruning_class_pareto_top_per_class=cfg.fs_iterative_pruning_class_pareto_top_per_class,
        fs_iterative_pruning_class_pareto_global_fraction=cfg.fs_iterative_pruning_class_pareto_global_fraction,
        fs_iterative_pruning_class_pareto_minority_boost=cfg.fs_iterative_pruning_class_pareto_minority_boost,
        fs_iterative_pruning_class_pareto_stability_gate_enabled=cfg.fs_iterative_pruning_class_pareto_stability_gate_enabled,
        fs_iterative_pruning_class_pareto_stability_subsamples=cfg.fs_iterative_pruning_class_pareto_stability_subsamples,
        fs_iterative_pruning_class_pareto_stability_fraction=cfg.fs_iterative_pruning_class_pareto_stability_fraction,
        fs_iterative_pruning_class_pareto_stability_threshold=cfg.fs_iterative_pruning_class_pareto_stability_threshold,
        fs_iterative_pruning_class_pareto_stability_min_overlap=cfg.fs_iterative_pruning_class_pareto_stability_min_overlap,
        fs_iterative_pruning_class_pareto_stability_min_stable_features=cfg.fs_iterative_pruning_class_pareto_stability_min_stable_features,
        fs_iterative_pruning_class_pareto_stability_fallback_on_failure=cfg.fs_iterative_pruning_class_pareto_stability_fallback_on_failure,
        classification_selection_mode=str(getattr(cfg, "classification_selection_mode", "legacy") or "legacy"),
        model_candidates=tuple(cfg.model_candidates),
        exclude_model_candidates=tuple(getattr(cfg, "exclude_model_candidates", tuple()) or tuple()),
        classifier_regime_candidate_exclusions=tuple(
            getattr(cfg, "classifier_regime_candidate_exclusions", tuple()) or tuple()
        ),
        classifier_oracle_complexity_prior_overrides=tuple(
            getattr(cfg, "classifier_oracle_complexity_prior_overrides", tuple()) or tuple()
        ),
        include_elastic_net_model=cfg.include_elastic_net_model,
        include_rf_model=cfg.include_rf_model,
        include_knn_model=cfg.include_knn_model,
        include_svm_linear_model=cfg.include_svm_linear_model,
        include_dlda_model=cfg.include_dlda_model,
        include_nsc_model=bool(getattr(cfg, "include_nsc_model", False)),
        include_pls_da_model=bool(getattr(cfg, "include_pls_da_model", False)),
        include_gpc_model=bool(getattr(cfg, "include_gpc_model", False)),
        include_nb_model=cfg.include_nb_model,
        include_vote_ensemble_model=cfg.include_vote_ensemble_model,
        include_rp_ensemble_model=bool(getattr(cfg, "include_rp_ensemble_model", False)),
        include_dbda_model=bool(getattr(cfg, "include_dbda_model", False)),
        include_gqda_model=bool(getattr(cfg, "include_gqda_model", False)),
        include_bc_svm_linear_model=bool(getattr(cfg, "include_bc_svm_linear_model", False)),
        include_sglnn_model=bool(getattr(cfg, "include_sglnn_model", False)),
        include_xgb_model=cfg.include_xgb_model,
        include_lgbm_model=bool(getattr(cfg, "include_lgbm_model", False)),
        include_extra_tree_model=bool(getattr(cfg, "include_extra_tree_model", False)),
        include_catboost_model=bool(getattr(cfg, "include_catboost_model", False)),
        include_tabpfn_model=cfg.include_tabpfn_model,
        model_cv_lr_max_iter=cfg.model_cv_lr_max_iter,
        model_cv_use_hybrid_score=cfg.model_cv_use_hybrid_score,
        model_cv_balanced_weight=cfg.model_cv_balanced_weight,
        model_cv_macro_f1_weight=cfg.model_cv_macro_f1_weight,
        model_cv_runtime_containment_enabled=cfg.model_cv_runtime_containment_enabled,
        model_cv_runtime_max_candidates=cfg.model_cv_runtime_max_candidates,
        model_cv_runtime_high_p_over_n_threshold=cfg.model_cv_runtime_high_p_over_n_threshold,
        model_cv_runtime_high_class_threshold=cfg.model_cv_runtime_high_class_threshold,
        model_cv_runtime_min_class_count_threshold=cfg.model_cv_runtime_min_class_count_threshold,
        classifier_oracle_k=int(getattr(cfg, "classifier_oracle_k", 1) or 1),
        classifier_oracle_weighting_mode=str(
            getattr(cfg, "classifier_oracle_weighting_mode", "tritrust") or "tritrust"
        ),
        classifier_oracle_include_calibration=bool(
            getattr(cfg, "classifier_oracle_include_calibration", True)
        ),
        classifier_oracle_include_james_stein=bool(
            getattr(cfg, "classifier_oracle_include_james_stein", True)
        ),
        classifier_oracle_complexity_shrinkage=bool(
            getattr(cfg, "classifier_oracle_complexity_shrinkage", False)
        ),
        classifier_oracle_include_cvar=bool(
            getattr(cfg, "classifier_oracle_include_cvar", False)
        ),
        classifier_oracle_cvar_alpha=float(
            getattr(cfg, "classifier_oracle_cvar_alpha", 0.33) or 0.33
        ),
        classifier_oracle_use_dynamic_complexity=bool(
            getattr(cfg, "classifier_oracle_use_dynamic_complexity", False)
        ),
        classifier_oracle_portfolio_diversity=bool(
            getattr(cfg, "classifier_oracle_portfolio_diversity", False)
        ),
        classifier_oracle_portfolio_overlap_threshold=float(
            getattr(cfg, "classifier_oracle_portfolio_overlap_threshold", 0.75) or 0.75
        ),
        classifier_oracle_portfolio_corr_threshold=float(
            getattr(cfg, "classifier_oracle_portfolio_corr_threshold", 0.85) or 0.85
        ),
        classifier_oracle_enable_hoeffding_racing=bool(
            getattr(cfg, "classifier_oracle_enable_hoeffding_racing", True)
        ),
        classifier_oracle_hoeffding_delta=float(
            getattr(cfg, "classifier_oracle_hoeffding_delta", 0.10) or 0.10
        ),
        classifier_oracle_enable_bbc=bool(getattr(cfg, "classifier_oracle_enable_bbc", True)),
        classifier_oracle_bbc_bootstrap_rounds=int(
            getattr(cfg, "classifier_oracle_bbc_bootstrap_rounds", 200) or 200
        ),
        classifier_oracle_bbc_ci_level=float(
            getattr(cfg, "classifier_oracle_bbc_ci_level", 0.90) or 0.90
        ),
        classifier_oracle_enable_ensemble=bool(
            getattr(cfg, "classifier_oracle_enable_ensemble", False)
        ),
        classifier_oracle_ensemble_voting_mode=str(
            getattr(cfg, "classifier_oracle_ensemble_voting_mode", "hard") or "hard"
        ),
        classifier_oracle_greedy_ensemble=bool(
            getattr(cfg, "classifier_oracle_greedy_ensemble", False)
        ),
        classifier_oracle_greedy_ensemble_rounds=int(
            getattr(cfg, "classifier_oracle_greedy_ensemble_rounds", 10) or 10
        ),
        classifier_oracle_candidate_pruning=bool(
            getattr(cfg, "classifier_oracle_candidate_pruning", False)
        ),
        classifier_oracle_candidate_pruning_threshold=float(
            getattr(cfg, "classifier_oracle_candidate_pruning_threshold", 0.0) or 0.0
        ),
        classifier_oracle_incumbent_early_stopping=bool(
            getattr(cfg, "classifier_oracle_incumbent_early_stopping", False)
        ),
        classifier_oracle_behavior_profile=str(
            getattr(cfg, "classifier_oracle_behavior_profile", "current") or "current"
        ).strip().lower(),
        classifier_oracle_use_per_family_flaml=bool(
            getattr(cfg, "classifier_oracle_use_per_family_flaml", True)
        ),
        stage2_ratio_augmentation_enabled=bool(
            getattr(cfg, "stage2_ratio_augmentation_enabled", False)
        ),
        stage2_ratio_max_features=int(getattr(cfg, "stage2_ratio_max_features", 16) or 16),
        stage2_ratio_selection_method=str(
            getattr(cfg, "stage2_ratio_selection_method", "correlation") or "correlation"
        ),
        stage2_ratio_epsilon=float(getattr(cfg, "stage2_ratio_epsilon", 1e-6) or 1e-6),
        enable_maqc_pairing=cfg.enable_maqc_pairing,
        maqc_pairing_method_sets=tuple(tuple(methods) for methods in (cfg.maqc_pairing_method_sets or ())),
        maqc_pairing_method_set_names=tuple(cfg.maqc_pairing_method_set_names or ()),
        meta_learning_selector_mode=str(
            getattr(cfg, "meta_learning_selector_mode", "none") or "none"
        ),
        meta_learning_confidence_threshold=float(
            getattr(cfg, "meta_learning_confidence_threshold", 0.55) or 0.55
        ),
        meta_learning_records_path=str(
            getattr(cfg, "meta_learning_records_path", "") or ""
        ).strip(),
    )


def apply_config_override(cfg: DFFSConfig, key: str, value: Any) -> None:
    if "__" in key:
        root, child = key.split("__", 1)
        if root == "dist_config":
            setattr(cfg.dist_config, child, value)
            return
    setattr(cfg, key, value)


def _compare_to_sota(metric: float, sota_low: float, sota_high: float) -> str:
    if not np.isfinite(metric):
        return "nan"
    if metric > sota_high:
        return "above"
    if metric >= sota_low:
        return "within"
    return "below"


def _protocol_gap_note(spec: BenchmarkDatasetSpec) -> str:
    if spec.source_kind == "validation_catalog":
        return (
            "Published SOTA often uses LOOCV / repeated CV / best-of-N protocols. "
            "This benchmark reports both strict-holdout and inflated-reference ranges, "
            "with strict holdout used for promotion decisions."
        )
    return ""


def _build_shadow_evaluator_pilot(
    rows: List[Dict[str, Any]],
    *,
    frozen_dataset_ids: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute a shadow-evaluator concordance table on a frozen subset.

    Primary evaluator uses existing strict-holdout balanced-accuracy status.
    Shadow evaluator re-scores against the same strict ranges using hybrid score.
    """
    frozen = {str(ds).strip() for ds in (frozen_dataset_ids or []) if str(ds).strip()}
    if not frozen:
        return pd.DataFrame(), {
            "enabled": False,
            "frozen_subset_id": "",
            "n_compared": 0,
            "concordance_rate": 0.0,
            "disagreement_count": 0,
        }

    entries: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("protocol", "holdout")) != "holdout":
            continue
        dataset_id = str(row.get("dataset_id", ""))
        if dataset_id not in frozen:
            continue
        spec = BENCHMARK_DATASETS.get(dataset_id)
        if spec is None:
            continue
        primary_status = str(row.get("sota_holdout_status", ""))
        shadow_status = _compare_to_sota(
            float(row.get("hybrid_score", float("nan"))),
            *spec.sota_holdout_bal_acc,
        )
        concordant = int(primary_status == shadow_status)
        entries.append(
            {
                "dataset_id": dataset_id,
                "seed": int(row.get("seed", 0)),
                "config": str(row.get("config", "")),
                "primary_status": primary_status,
                "shadow_status": str(shadow_status),
                "concordant": concordant,
            }
        )

    if not entries:
        return pd.DataFrame(), {
            "enabled": True,
            "frozen_subset_id": "shadow_frozen_empty",
            "n_compared": 0,
            "concordance_rate": 0.0,
            "disagreement_count": 0,
        }

    shadow_df = pd.DataFrame(entries)
    n_compared = int(len(shadow_df))
    n_concordant = int(shadow_df["concordant"].sum())
    return shadow_df, {
        "enabled": True,
        "frozen_subset_id": "shadow_frozen_" + "_".join(sorted(frozen)),
        "n_compared": n_compared,
        "concordance_rate": float(n_concordant / max(1, n_compared)),
        "disagreement_count": int(n_compared - n_concordant),
    }


def _apply_integrated_scenario_defaults(cfg: DFFSConfig, spec: BenchmarkDatasetSpec) -> None:
    if spec.validation_pipeline != "integrated":
        return
    scenario = str(spec.validation_scenario or "").strip()
    if scenario == "low_gof_downweighting":
        cfg.low_gof_downweighting = True
    elif scenario == "stability_signal":
        cfg.use_distribution_stability_weight = True
    elif scenario == "cdf_transform":
        cfg.apply_cdf_transform = True


def _resolve_model_candidates_from_args(args: argparse.Namespace) -> Tuple[str, ...]:
    explicit = tuple(getattr(args, "model_candidates", []) or ())
    if explicit:
        return explicit

    profile = str(getattr(args, "model_candidate_profile", "default") or "default").strip().lower()
    profiled = MODEL_CANDIDATE_PROFILES.get(profile, tuple())
    if profiled:
        return tuple(profiled)

    model_candidates: List[str] = ["lr", "svm_rbf"]
    if args.include_elastic_net_model:
        model_candidates.append("elastic_net_lr")
    if args.include_rf_model:
        model_candidates.append("rf")
    if args.include_knn_model:
        model_candidates.append("knn")
    if getattr(args, "include_svm_linear_model", False):
        model_candidates.append("svm_linear")
    if getattr(args, "include_dlda_model", False):
        model_candidates.append("dlda")
    if bool(getattr(args, "include_nsc_model", False)):
        model_candidates.append("nsc")
    if bool(getattr(args, "include_pls_da_model", False)):
        model_candidates.append("pls_da_classifier")
    if bool(getattr(args, "include_gpc_model", False)):
        model_candidates.append("gpc")
    if getattr(args, "include_nb_model", False):
        model_candidates.append("nb")
    if getattr(args, "include_vote_ensemble_model", False):
        model_candidates.append("vote_ensemble")
    if bool(getattr(args, "include_rp_ensemble_model", False)):
        model_candidates.append("rp_ensemble")
    if bool(getattr(args, "include_dbda_model", False)):
        model_candidates.append("dbda")
    if bool(getattr(args, "include_gqda_model", False)):
        model_candidates.append("gqda")
    if bool(getattr(args, "include_bc_svm_linear_model", False)):
        model_candidates.append("bc_svm_linear")
    if bool(getattr(args, "include_sglnn_model", False)):
        model_candidates.append("sglnn")
    if args.include_xgb_model:
        model_candidates.append("xgb")
    if bool(getattr(args, "include_lgbm_model", False)):
        model_candidates.append("lgbm")
    if bool(getattr(args, "include_extra_tree_model", False)):
        model_candidates.append("extra_tree")
    if bool(getattr(args, "include_catboost_model", False)):
        model_candidates.append("catboost")
    if args.include_tabpfn_model:
        model_candidates.append("tabpfn")
    return tuple(model_candidates)


def _flatten_cli_values(raw_values: Any) -> Tuple[str, ...]:
    if raw_values is None:
        return tuple()
    if isinstance(raw_values, (str, bytes)):
        raw_iter = [raw_values]
    else:
        raw_iter = list(raw_values)
    flattened: List[str] = []
    for raw in raw_iter:
        if isinstance(raw, (list, tuple, set)):
            flattened.extend(_flatten_cli_values(list(raw)))
            continue
        tokens = [
            str(token).strip()
            for token in re.split(r"[,\s]+", str(raw))
            if str(token).strip()
        ]
        flattened.extend(tokens)
    out: List[str] = []
    seen: set[str] = set()
    for token in flattened:
        if token in seen:
            continue
        out.append(str(token))
        seen.add(str(token))
    return tuple(out)


def _resolve_maqc_pairing_method_sets_from_args(args: argparse.Namespace) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
    if not bool(getattr(args, "enable_maqc_pairing", False)):
        return tuple(), tuple()

    names = tuple(getattr(args, "maqc_fs_method_sets", []) or ())
    if not names:
        # Conservative default: baseline strict selector plus two selectors that
        # directly target measured multiclass failure modes.
        names = tuple(
            name
            for name in (
                "strict_plus_mrmr",
                "mnpo_rankagg_extended",
                "mnpo_ova_extended",
            )
            if name in FS_METHOD_SETS
        )

    method_sets: List[Tuple[str, ...]] = []
    resolved_names: List[str] = []
    for name in names:
        if name in FS_METHOD_SETS:
            resolved_names.append(str(name))
            method_sets.append(FS_METHOD_SETS[name])

    return tuple(resolved_names), tuple(method_sets)


def _resolve_inner_n_jobs(args: argparse.Namespace) -> int:
    """Resolve adaptive inner parallelism from --inner-n-jobs and --max-workers.

    Strategy (TOP-1):
    - "auto" + max_workers > 1 → 1  (no inner parallelism to avoid over-subscription)
    - "auto" + max_workers == 1 → max(1, cpu_count // 2)
    - explicit integer → use as-is (-1 maps to cpu_count)
    """
    raw = str(getattr(args, "inner_n_jobs", "auto") or "auto").strip().lower()
    max_workers = int(getattr(args, "max_workers", 1) or 1)
    if raw == "auto":
        if max_workers > 1:
            return 1
        return max(1, (os.cpu_count() or 1) // 2)
    n = int(raw)
    if n == -1:
        return os.cpu_count() or 1
    return max(1, n)


def _build_base_config(args: argparse.Namespace, spec: BenchmarkDatasetSpec, seed: int) -> DFFSConfig:
    # Some method-set names are aliases for a shared method stack with additional
    # opt-in overlays enabled by default. Keep these presets local to config
    # construction so programmatic callers that use `_build_base_config()` (e.g.
    # tests) see consistent behavior.
    fs_method_set = str(getattr(args, "fs_method_set", "") or "").strip()
    preset_iter_pruning_bounded_cpss = fs_method_set == "mnpo_iterative_pruning_bounded_cpss_extended"
    preset_iter_pruning_bounded_pareto = fs_method_set in {
        "mnpo_iterative_pruning_bounded_pareto_extended",
        "mnpo_iterative_pruning_bounded_pareto_stability_extended",
    }
    preset_iter_pruning_bounded_pareto_stability = (
        fs_method_set == "mnpo_iterative_pruning_bounded_pareto_stability_extended"
    )
    preset_broad_bundle_a = fs_method_set == "mnpo_broad_bundle_a"
    preset_broad_bundle_b = fs_method_set == "mnpo_broad_bundle_b"
    preset_broad_bundle_c = fs_method_set == "mnpo_broad_bundle_c"

    model_candidates = _resolve_model_candidates_from_args(args)
    exclude_model_candidates = _flatten_cli_values(getattr(args, "exclude_classifiers", []))
    regime_candidate_exclusions = _flatten_cli_values(
        getattr(args, "classifier_regime_candidate_exclusions", [])
    )
    complexity_prior_overrides = _flatten_cli_values(
        getattr(args, "classifier_complexity_prior_override", [])
    )
    include_elastic = bool(args.include_elastic_net_model) or ("elastic_net_lr" in set(model_candidates))
    include_rf = bool(args.include_rf_model) or ("rf" in set(model_candidates))
    include_knn = bool(args.include_knn_model) or ("knn" in set(model_candidates))
    include_svm_linear = bool(getattr(args, "include_svm_linear_model", False)) or ("svm_linear" in set(model_candidates))
    include_dlda = bool(getattr(args, "include_dlda_model", False)) or ("dlda" in set(model_candidates))
    include_nsc = bool(getattr(args, "include_nsc_model", False)) or ("nsc" in set(model_candidates))
    include_pls_da = bool(getattr(args, "include_pls_da_model", False)) or ("pls_da_classifier" in set(model_candidates))
    include_gpc = bool(getattr(args, "include_gpc_model", False)) or ("gpc" in set(model_candidates))
    include_nb = bool(getattr(args, "include_nb_model", False)) or ("nb" in set(model_candidates))
    include_vote_ensemble = bool(getattr(args, "include_vote_ensemble_model", False)) or ("vote_ensemble" in set(model_candidates))
    include_rp_ensemble = bool(getattr(args, "include_rp_ensemble_model", False)) or ("rp_ensemble" in set(model_candidates))
    include_dbda = bool(getattr(args, "include_dbda_model", False)) or ("dbda" in set(model_candidates))
    include_gqda = bool(getattr(args, "include_gqda_model", False)) or ("gqda" in set(model_candidates))
    include_bc_svm_linear = bool(getattr(args, "include_bc_svm_linear_model", False)) or ("bc_svm_linear" in set(model_candidates))
    include_sglnn = bool(getattr(args, "include_sglnn_model", False)) or ("sglnn" in set(model_candidates))
    include_xgb = bool(args.include_xgb_model) or ("xgb" in set(model_candidates))
    include_lgbm = bool(getattr(args, "include_lgbm_model", False)) or ("lgbm" in set(model_candidates))
    include_extra_tree = bool(getattr(args, "include_extra_tree_model", False)) or ("extra_tree" in set(model_candidates))
    include_catboost = bool(getattr(args, "include_catboost_model", False)) or ("catboost" in set(model_candidates))
    include_tabpfn = bool(args.include_tabpfn_model) or ("tabpfn" in set(model_candidates))
    # TOP-1: Resolve adaptive inner parallelism.
    inner_n_jobs = _resolve_inner_n_jobs(args)
    classification_cfg = ClassificationConfig(
        selection_mode=str(getattr(args, "classifier_selection_mode", "legacy") or "legacy"),
        backend=str(getattr(args, "classification_backend", "sklearn") or "sklearn"),
        flaml_time_budget=int(getattr(args, "flaml_time_budget", 60) or 60),
        optuna_time_budget=int(getattr(args, "optuna_time_budget", 120) or 120),
        optuna_n_trials=int(getattr(args, "optuna_n_trials", 25) or 25),
        model_candidates=tuple(model_candidates),
        exclude_model_candidates=tuple(exclude_model_candidates),
        regime_candidate_exclusions=tuple(regime_candidate_exclusions),
        oracle_complexity_prior_overrides=tuple(complexity_prior_overrides),
        include_elastic_net_model=bool(include_elastic),
        include_rf_model=bool(include_rf),
        include_knn_model=bool(include_knn),
        include_svm_linear_model=bool(include_svm_linear),
        include_dlda_model=bool(include_dlda),
        include_nsc_model=bool(include_nsc),
        include_pls_da_model=bool(include_pls_da),
        include_gpc_model=bool(include_gpc),
        include_nb_model=bool(include_nb),
        include_vote_ensemble_model=bool(include_vote_ensemble),
        include_rp_ensemble_model=bool(include_rp_ensemble),
        include_dbda_model=bool(include_dbda),
        include_gqda_model=bool(include_gqda),
        include_bc_svm_linear_model=bool(include_bc_svm_linear),
        include_sglnn_model=bool(include_sglnn),
        include_xgb_model=bool(include_xgb),
        include_lgbm_model=bool(include_lgbm),
        include_extra_tree_model=bool(include_extra_tree),
        include_catboost_model=bool(include_catboost),
        include_tabpfn_model=bool(include_tabpfn),
        use_hybrid_score=bool(args.enable_hybrid_model_cv),
        hybrid_balanced_weight=float(getattr(args, "model_cv_balanced_weight", 0.6) or 0.6),
        hybrid_macro_f1_weight=float(getattr(args, "model_cv_macro_f1_weight", 0.4) or 0.4),
        runtime_containment_enabled=bool(getattr(args, "enable_model_cv_runtime_containment", False)),
        runtime_max_candidates=int(getattr(args, "model_cv_runtime_max_candidates", 0) or 0),
        runtime_high_p_over_n_threshold=float(
            getattr(args, "model_cv_runtime_high_p_over_n_threshold", 40.0) or 40.0
        ),
        runtime_high_class_threshold=int(getattr(args, "model_cv_runtime_high_class_threshold", 6) or 6),
        runtime_min_class_count_threshold=int(
            getattr(args, "model_cv_runtime_min_class_count_threshold", 12) or 12
        ),
        lr_max_iter=int(getattr(args, "model_cv_lr_max_iter", 10000) or 10000),
        oracle_k=int(getattr(args, "classifier_oracle_k", 1) or 1),
        oracle_weighting_mode=str(
            getattr(args, "classifier_oracle_weighting_mode", "tritrust") or "tritrust"
        ),
        oracle_include_calibration=not bool(
            getattr(args, "disable_classifier_oracle_calibration", False)
        ),
        oracle_include_james_stein=not bool(
            getattr(args, "disable_classifier_oracle_james_stein", False)
        ),
        oracle_complexity_shrinkage=bool(
            getattr(args, "classifier_oracle_complexity_shrinkage", False)
        ),
        oracle_include_cvar=bool(
            getattr(args, "enable_classifier_oracle_cvar", False)
        ),
        oracle_cvar_alpha=float(
            getattr(args, "classifier_oracle_cvar_alpha", 0.33) or 0.33
        ),
        oracle_use_dynamic_complexity=bool(
            getattr(args, "enable_classifier_oracle_dynamic_complexity", False)
        ),
        oracle_portfolio_diversity=bool(
            getattr(args, "enable_classifier_oracle_portfolio_diversity", False)
        ),
        oracle_portfolio_overlap_threshold=float(
            getattr(args, "classifier_oracle_portfolio_overlap_threshold", 0.75) or 0.75
        ),
        oracle_portfolio_corr_threshold=float(
            getattr(args, "classifier_oracle_portfolio_corr_threshold", 0.85) or 0.85
        ),
        oracle_enable_hoeffding_racing=not bool(
            getattr(args, "disable_classifier_oracle_hoeffding_racing", False)
        ),
        oracle_hoeffding_delta=float(
            getattr(args, "classifier_oracle_hoeffding_delta", 0.10) or 0.10
        ),
        oracle_enable_bbc=not bool(getattr(args, "disable_classifier_oracle_bbc", False)),
        oracle_bbc_bootstrap_rounds=int(
            getattr(args, "classifier_oracle_bbc_bootstrap_rounds", 200) or 200
        ),
        oracle_bbc_ci_level=float(
            getattr(args, "classifier_oracle_bbc_ci_level", 0.90) or 0.90
        ),
        oracle_enable_ensemble=bool(getattr(args, "enable_classifier_oracle_ensemble", False)),
        oracle_ensemble_voting_mode=str(
            getattr(args, "classifier_oracle_ensemble_voting_mode", "hard") or "hard"
        ),
        oracle_greedy_ensemble=bool(
            getattr(args, "enable_classifier_oracle_greedy_ensemble", False)
        ),
        oracle_greedy_ensemble_rounds=int(
            getattr(args, "classifier_oracle_greedy_ensemble_rounds", 10) or 10
        ),
        oracle_candidate_pruning=bool(
            getattr(args, "enable_classifier_oracle_candidate_pruning", False)
        ),
        oracle_candidate_pruning_threshold=float(
            getattr(args, "classifier_oracle_candidate_pruning_threshold", 0.0) or 0.0
        ),
        oracle_incumbent_early_stopping=bool(
            getattr(args, "enable_classifier_oracle_incumbent_early_stopping", False)
        ),
        oracle_behavior_profile=str(
            getattr(args, "classifier_oracle_behavior_profile", "current") or "current"
        ).strip().lower(),
        oracle_use_per_family_flaml=not bool(
            getattr(args, "disable_classifier_oracle_per_family_flaml", False)
        ),
        stage2_ratio_augmentation_enabled=bool(
            getattr(args, "enable_stage2_ratio_augmentation", False)
        ),
        stage2_ratio_max_features=int(
            getattr(args, "stage2_ratio_max_features", 16) or 16
        ),
        stage2_ratio_selection_method=str(
            getattr(args, "stage2_ratio_selection_method", "correlation") or "correlation"
        ),
        stage2_ratio_epsilon=float(
            getattr(args, "stage2_ratio_epsilon", 1e-6) or 1e-6
        ),
        conformal_enabled=bool(getattr(args, "enable_classifier_conformal", False)),
        conformal_alpha=float(getattr(args, "classifier_conformal_alpha", 0.10) or 0.10),
        conformal_calibration_fraction=float(
            getattr(args, "classifier_conformal_calibration_fraction", 0.25) or 0.25
        ),
        conformal_min_calibration=int(
            getattr(args, "classifier_conformal_min_calibration", 20) or 20
        ),
        conformal_output_sets=bool(
            getattr(args, "classifier_conformal_output_sets", False)
        ),
        conformal_method=str(
            getattr(args, "classifier_conformal_method", "split") or "split"
        ).strip().lower(),
        stage2_max_train_test_gap=float(
            getattr(args, "stage2_max_train_test_gap", 0.0) or 0.0
        ),
        stage2_tree_complexity_penalty_enabled=bool(
            getattr(args, "stage2_tree_complexity_penalty_enabled", False)
        ),
        stage2_tree_complexity_penalty_strength=float(
            getattr(args, "stage2_tree_complexity_penalty_strength", 0.1) or 0.1
        ),
        n_jobs=inner_n_jobs,
    )
    maqc_names, maqc_method_sets = _resolve_maqc_pairing_method_sets_from_args(args)
    # T-DS2: DF fast-path removed; retain CLI/config compatibility as no-op.
    _warn_deprecated_df_fastpath_args(args)
    df_fastpath_enabled = False
    forced_train_cap = int(getattr(args, "max_train_samples", 0) or 0)
    effective_max_train_samples: Optional[int] = None
    if forced_train_cap > 0:
        effective_max_train_samples = int(forced_train_cap)
    elif spec.max_train_samples is not None:
        effective_max_train_samples = int(spec.max_train_samples)

    explicit_iter_pruning_bounded_cpss = bool(
        getattr(args, "enable_fs_iterative_pruning_bounded_cpss_overlay", False)
    )
    explicit_iter_pruning_pareto_prefilter = bool(
        getattr(args, "enable_fs_iterative_pruning_class_pareto_prefilter", False)
    )
    explicit_iter_pruning_pareto_stability_gate = bool(
        getattr(args, "enable_fs_iterative_pruning_class_pareto_stability_gate", False)
    )
    effective_iter_pruning_bounded_cpss = explicit_iter_pruning_bounded_cpss or preset_iter_pruning_bounded_cpss
    effective_iter_pruning_pareto_stability_gate = (
        explicit_iter_pruning_pareto_stability_gate or preset_iter_pruning_bounded_pareto_stability
    )
    # Pareto stability gate requires the Pareto prefilter; enable prefilter when the gate is enabled.
    effective_iter_pruning_pareto_prefilter = (
        explicit_iter_pruning_pareto_prefilter
        or preset_iter_pruning_bounded_pareto
        or effective_iter_pruning_pareto_stability_gate
    )
    effective_wrapper_refine_enabled = bool(getattr(args, "enable_fs_wrapper_refine", False)) or preset_broad_bundle_b
    effective_rashomon_enabled = bool(getattr(args, "enable_fs_rashomon", False)) or preset_broad_bundle_b
    effective_screening_enabled = bool(getattr(args, "screening_enabled", False)) or preset_broad_bundle_c
    effective_screening_method = str(getattr(args, "screening_method", "none") or "none")
    if preset_broad_bundle_c and effective_screening_method == "none":
        effective_screening_method = "stir"
    enabled_method_stack = tuple(FS_METHOD_SETS[args.fs_method_set])
    tier_lockout_fallback_methods = _resolve_tier_lockout_fallback_methods(
        args,
        enabled_method_stack,
    )
    tier_routing_table = _parse_tier_routing_table(getattr(args, "tier_routing_table", ""))
    regime_default_methods = tuple(FS_METHOD_SETS.get("strict_plus_mrmr", enabled_method_stack))
    regime_gating_simple_methods = _resolve_regime_gating_simple_methods(
        args,
        regime_default_methods,
    )
    fs_portfolio_size = int(getattr(args, "fs_portfolio_size", 6) or 6)
    adaptive_enabled = bool(getattr(args, "enable_fs_adaptive_portfolio_sizing", False)) or preset_broad_bundle_a
    adaptive_size_min_raw = getattr(args, "fs_adaptive_size_min", None)
    adaptive_size_max_raw = getattr(args, "fs_adaptive_size_max", None)
    if adaptive_enabled and (adaptive_size_min_raw is None or adaptive_size_max_raw is None):
        adaptive_size_min_raw = int(max(1, fs_portfolio_size - 2))
        adaptive_size_max_raw = int(max(adaptive_size_min_raw, fs_portfolio_size + 2))
    effective_fs_qre = bool(getattr(args, "fs_use_qre_smoothing", False)) or preset_broad_bundle_a
    effective_fs_oracle_redundancy = bool(getattr(args, "fs_use_oracle_redundancy_penalty", False)) or preset_broad_bundle_a
    parsed_prefilter_strategies = tuple(
        _parse_csv_or_space_list(getattr(args, "prefilter_strategies", ""))
        or ("mi_ftest_blend",)
    )
    effective_prefilter_wsnr_enabled = bool(getattr(args, "prefilter_wsnr_enabled", False))
    if effective_prefilter_wsnr_enabled and "wsnr" not in set(parsed_prefilter_strategies):
        parsed_prefilter_strategies = tuple(list(parsed_prefilter_strategies) + ["wsnr"])
    # T-R-265: auto-add bh_fdr strategy when prefilter_bh_ttest_enabled flag is on.
    effective_prefilter_bh_ttest_enabled = bool(getattr(args, "prefilter_bh_ttest_enabled", True))
    if effective_prefilter_bh_ttest_enabled and "bh_fdr" not in set(parsed_prefilter_strategies):
        parsed_prefilter_strategies = tuple(list(parsed_prefilter_strategies) + ["bh_fdr"])
    effective_prefilter_union_enabled = bool(
        bool(getattr(args, "prefilter_union_enabled", False)) or effective_prefilter_wsnr_enabled
    )

    fs_fraction_override = float(getattr(args, "fs_fraction", 0.0) or 0.0)
    if fs_fraction_override > 0.0:
        fs_fraction_value = float(np.clip(fs_fraction_override, 0.05, 1.0))
    else:
        fs_fraction_value = float(spec.fs_fraction)

    cfg = DFFSConfig(
        random_seed=seed,
        test_size=args.test_size,
        max_train_samples=effective_max_train_samples,
        fs_fraction=float(fs_fraction_value),
        n_final_features=int(spec.n_final_features),
        n_jobs=inner_n_jobs,
        classification=classification_cfg,
        dist_config=DistributionFitterConfig(
            robust_mode=not args.disable_df_robust,
            use_adaptive_strategy=True,
            use_lrt=not args.disable_df_lrt,
            use_cv=args.enable_df_cv,
            compute_budget=args.compute_budget,
            use_support_filtering=not args.disable_support_filter,
            rejection_gate=True,
            rejection_p_threshold=args.df_rejection_threshold,
            confidence_margin=args.df_confidence_margin,
            family_set=str(getattr(args, "df_family_set", "v6") or "v6"),
            compute_ad=bool(getattr(args, "df_compute_ad", False)),
            ad_bootstrap_samples=int(getattr(args, "df_ad_bootstrap_samples", 0) or 0),
            compute_qq_pp=bool(getattr(args, "df_compute_qq_pp", False)),
            compute_dip=bool(getattr(args, "df_compute_dip", True)),
            dip_hist_bins=int(getattr(args, "df_dip_hist_bins", 40) or 40),
            interval_likelihood=bool(getattr(args, "df_interval_likelihood", False)),
            interval_delta_override=float(getattr(args, "df_interval_delta_override", 0.0) or 0.0),
            use_lmoment_prescreen=bool(getattr(args, "df_lmoment_prescreen", False)),
            lmoment_prescreen_max_candidates=(
                int(getattr(args, "df_lmoment_prescreen_max_candidates", 12) or 12)
                if bool(getattr(args, "df_lmoment_prescreen", False))
                else 0
            ),
            estimator=str(getattr(args, "df_estimator", "mle") or "mle"),
            mps_maxiter=int(getattr(args, "df_mps_maxiter", 250) or 250),
            mps_tol=float(getattr(args, "df_mps_tol", 1e-6) or 1e-6),
            compute_crps=bool(getattr(args, "df_compute_crps", False)),
            crps_mc_samples=int(getattr(args, "df_crps_mc_samples", 96) or 96),
            crps_data_subsample=int(getattr(args, "df_crps_data_subsample", 256) or 256),
            compute_crps_uq_decomposition=bool(getattr(args, "df_crps_uq_decomposition", False)),
            mnpo_use_tritrust=not bool(getattr(args, "df_mnpo_disable_tritrust", False)),
            mnpo_include_crps=bool(getattr(args, "df_mnpo_include_crps", False)),
            mnpo_include_preq=bool(getattr(args, "df_mnpo_include_preq", False)),
            mnpo_use_tail_risk_oracle=False,
            mnpo_tail_risk_alpha=float(getattr(args, "df_tail_risk_alpha", 0.33) or 0.33),
            mnpo_use_qre_smoothing=bool(getattr(args, "df_use_qre_smoothing", False)),
            mnpo_qre_temperature_gamma=float(getattr(args, "df_qre_temperature_gamma", 1.0) or 1.0),
            mnpo_use_oracle_redundancy_penalty=bool(getattr(args, "df_use_oracle_redundancy_penalty", False)),
            mnpo_compute_tremble_sensitivity=bool(getattr(args, "df_compute_tremble_sensitivity", False)),
            preq_holdout_fraction=float(getattr(args, "df_preq_holdout_fraction", 0.20) or 0.20),
            preq_min_train=int(getattr(args, "df_preq_min_train", 20) or 20),
            preq_max_test_points=int(getattr(args, "df_preq_max_test_points", 128) or 128),
            random_state=int(seed),
        ),
        dist_criterion=args.dist_criterion,
        apply_cdf_transform=not args.disable_cdf_transform,
        df_stage_position=str(getattr(args, "df_stage_position", "after_fs") or "after_fs"),
        df_fastpath_enabled=df_fastpath_enabled,
        df_fastpath_trigger=str(getattr(args, "df_fastpath_trigger", "small_n_or_low_unique") or "small_n_or_low_unique"),
        df_fastpath_small_n_threshold=int(getattr(args, "df_fastpath_small_n_threshold", 250) or 250),
        df_fastpath_unique_ratio_threshold=float(getattr(args, "df_fastpath_unique_ratio_threshold", 0.05) or 0.05),
        df_fastpath_n_unique_threshold=int(getattr(args, "df_fastpath_n_unique_threshold", 12) or 12),
        multimodal_fallback=str(getattr(args, "df_multimodal_fallback", "gmm") or "gmm"),
        cdf_reliability_gate=not args.disable_cdf_reliability_gate,
        cdf_min_gof_p=args.cdf_min_gof_p,
        cdf_max_confidence_set=args.cdf_max_confidence_set,
        cdf_skip_heaped_features=args.cdf_skip_heaped_features,
        cdf_block_gating_cv=args.enable_cdf_block_gating_cv,
        cdf_block_gating_n_blocks=args.cdf_block_gating_n_blocks,
        cdf_block_gating_min_block_size=args.cdf_block_gating_min_block_size,
        cdf_block_gating_cv_splits=args.cdf_block_gating_cv_splits,
        cdf_block_gating_max_blocks=args.cdf_block_gating_max_blocks,
        cdf_block_gating_time_budget_sec=args.cdf_block_gating_time_budget_sec,
        cdf_block_gating_min_improvement=args.cdf_block_gating_min_improvement,
        max_dist_features=args.max_dist_features,
        low_gof_downweighting=not args.disable_low_gof_downweight,
        low_gof_threshold=args.low_gof_threshold,
        low_gof_weight=args.low_gof_weight,
        use_distribution_stability_weight=args.enable_dist_stability_weight,
        stability_bootstrap=args.stability_bootstrap,
        use_rank_prefilter=not args.disable_rank_prefilter,
        prefilter_top_k=args.prefilter_top_k,
        prefilter_adaptive_top_k=bool(getattr(args, "prefilter_adaptive_top_k", False)),
        prefilter_adaptive_top_k_scaling=float(
            getattr(args, "prefilter_adaptive_top_k_scaling", 0.5) or 0.5
        ),
        prefilter_mi_weight=float(getattr(args, "prefilter_mi_weight", 0.60) or 0.60),
        prefilter_f_weight=float(getattr(args, "prefilter_f_weight", 0.40) or 0.40),
        prefilter_union_enabled=bool(effective_prefilter_union_enabled),
        prefilter_strategies=tuple(parsed_prefilter_strategies),
        prefilter_nondefault_budget_fraction=float(
            getattr(args, "prefilter_nondefault_budget_fraction", 0.10) or 0.10
        ),
        prefilter_wsnr_enabled=bool(effective_prefilter_wsnr_enabled),
        prefilter_bh_ttest_enabled=bool(getattr(args, "prefilter_bh_ttest_enabled", True)),
        prefilter_bh_ttest_alpha=float(getattr(args, "prefilter_bh_ttest_alpha", 0.05) or 0.05),
        # T-R-272: variance floor config.
        prefilter_variance_floor_enabled=bool(getattr(args, "prefilter_variance_floor_enabled", True)),
        prefilter_variance_floor_threshold=float(getattr(args, "prefilter_variance_floor_threshold", 1e-6) or 1e-6),
        prefilter_variance_floor_mode_freq=float(getattr(args, "prefilter_variance_floor_mode_freq", 0.99) or 0.99),
        prefilter_data_domain=infer_prefilter_data_domain(getattr(spec, "platform", "")),
        prefilter_rnaseq_transform_enabled=bool(
            not getattr(args, "disable_prefilter_rnaseq_transform", False)
        ),
        prefilter_rnaseq_transform_force=bool(
            getattr(args, "force_prefilter_rnaseq_transform", False)
        ),
        prefilter_rnaseq_nb_lrt_enabled=bool(
            getattr(args, "enable_prefilter_rnaseq_nb_lrt", False)
        ),
        prefilter_rnaseq_nb_lrt_alpha=float(
            getattr(args, "prefilter_rnaseq_nb_lrt_alpha", 0.10) or 0.10
        ),
        batch_correction=str(getattr(args, "batch_correction", "none") or "none"),
        batch_correction_combat_prior_strength=float(
            getattr(args, "batch_correction_combat_prior_strength", 8.0) or 8.0
        ),
        batch_correction_cdf_n_quantiles=int(
            getattr(args, "batch_correction_cdf_n_quantiles", 33) or 33
        ),
        batch_correction_cdf_clip_low=float(
            getattr(args, "batch_correction_cdf_clip_low", 0.01) or 0.01
        ),
        batch_correction_cdf_clip_high=float(
            getattr(args, "batch_correction_cdf_clip_high", 0.99) or 0.99
        ),
        screening_enabled=bool(effective_screening_enabled),
        screening_method=str(effective_screening_method),
        screening_pool_cap=int(getattr(args, "screening_pool_cap", 2000) or 2000),
        screening_stir_n_neighbors=int(getattr(args, "screening_stir_n_neighbors", 10) or 10),
        screening_stir_n_iter=int(getattr(args, "screening_stir_n_iter", 50) or 50),
        screening_stir_keep_fraction=float(
            getattr(args, "screening_stir_keep_fraction", 0.5) or 0.5
        ),
        screening_stir_min_features=int(getattr(args, "screening_stir_min_features", 20) or 20),
        screening_evalue_alpha=float(getattr(args, "screening_evalue_alpha", 0.20) or 0.20),
        screening_evalue_min_features=int(
            getattr(args, "screening_evalue_min_features", 20) or 20
        ),
        eval_models_enabled=bool(getattr(args, "eval_models_enabled", False)),
        eval_models=(
            _parse_csv_or_space_list(getattr(args, "eval_models", ""))
            or ("lr_l2", "linear_svc", "rf_small")
        ),
        eval_aggregate=str(getattr(args, "eval_aggregate", "mean") or "mean"),
        eval_cvar_alpha=float(getattr(args, "eval_cvar_alpha", 0.33) or 0.33),
        tier_lockout_enabled=bool(getattr(args, "tier_lockout_enabled", False)),
        tier_classifier_mode=str(
            getattr(args, "tier_classifier_mode", "heuristic") or "heuristic"
        ).strip().lower(),
        tier_classifier_model_path=str(
            getattr(args, "tier_classifier_model_path", "") or ""
        ).strip(),
        tier_lockout_tier=str(getattr(args, "tier_lockout_tier", "easy") or "easy").strip().lower(),
        tier_lockout_difficulty_source=str(
            getattr(args, "tier_lockout_difficulty_source", "historical") or "historical"
        ).strip().lower(),
        tier_lockout_fallback_methods=tier_lockout_fallback_methods,
        tier_routing_enabled=bool(getattr(args, "tier_routing_enabled", False)),
        tier_routing_difficulty_classifier=str(
            getattr(args, "tier_routing_difficulty_classifier", "meta_features") or "meta_features"
        ).strip().lower(),
        tier_routing_table=tier_routing_table,
        regime_gating_enabled=bool(getattr(args, "regime_gating_enabled", False)),
        regime_gating_difficulty_source=str(
            getattr(args, "regime_gating_difficulty_source", "historical") or "historical"
        ).strip().lower(),
        regime_gating_target_tier=str(
            getattr(args, "regime_gating_target_tier", "very_hard") or "very_hard"
        ).strip().lower(),
        regime_gating_min_samples_per_class=float(
            getattr(args, "regime_gating_min_samples_per_class", 7.0) or 7.0
        ),
        regime_gating_use_expanded_features=bool(
            getattr(args, "regime_gating_use_expanded_features", False)
        ),
        regime_gating_min_fisher_f1=float(
            getattr(args, "regime_gating_min_fisher_f1", 0.10) or 0.10
        ),
        regime_gating_max_n1_borderline=float(
            getattr(args, "regime_gating_max_n1_borderline", 0.40) or 0.40
        ),
        regime_gating_low_p_over_n_threshold=float(
            getattr(args, "regime_gating_low_p_over_n_threshold", 0.0) or 0.0
        ),
        regime_gating_simple_methods=tuple(regime_gating_simple_methods),
        regime_gating_very_hard_portfolio_max_methods=int(
            getattr(args, "regime_gating_very_hard_portfolio_max_methods", 4) or 4
        ),
        regime_gating_very_hard_copula_derandomize_runs=int(
            getattr(args, "regime_gating_very_hard_copula_derandomize_runs", 5) or 5
        ),
        regime_gating_low_p_over_n_mode=str(
            getattr(args, "regime_gating_low_p_over_n_mode", "fast_univariate_filter") or "fast_univariate_filter"
        ).strip().lower(),
        regime_gating_low_p_over_n_filter_max_k=int(
            getattr(args, "regime_gating_low_p_over_n_filter_max_k", 200) or 200
        ),
        regime_gating_very_hard_min_classes=int(
            getattr(args, "regime_gating_very_hard_min_classes", 5) or 5
        ),
        # T-R-268: extreme multiclass gate config.
        regime_gating_extreme_multiclass_enabled=bool(
            getattr(args, "regime_gating_extreme_multiclass_enabled", True)
        ),
        regime_gating_extreme_multiclass_threshold=int(
            getattr(args, "regime_gating_extreme_multiclass_threshold", 8) or 8
        ),
        regime_gating_extreme_multiclass_min_samples_per_class=float(
            getattr(args, "regime_gating_extreme_multiclass_min_samples_per_class", 11.0) or 11.0
        ),
        mnpo_performance_oracle_mode=str(
            getattr(args, "mnpo_performance_oracle_mode", "single") or "single"
        ).strip().lower(),
        selection_strategy=str(
            getattr(args, "selection_strategy", "mnpo_portfolio") or "mnpo_portfolio"
        ).strip().lower(),
        folding_method=str(getattr(args, "folding_method", "pls_da") or "pls_da"),
        folding_n_components=int(getattr(args, "folding_n_components", 512) or 512),
        folding_rff_gamma=(
            None
            if getattr(args, "folding_rff_gamma", None) is None
            else float(getattr(args, "folding_rff_gamma"))
        ),
        folding_pls_components=int(getattr(args, "folding_pls_components", 32) or 32),
        folding_pls_scale=not bool(getattr(args, "disable_folding_pls_scale", False)),
        folding_pls_min_classes=int(getattr(args, "folding_pls_min_classes", 5) or 5),
        folding_pls_min_n_per_class=int(
            getattr(args, "folding_pls_min_n_per_class", 3) or 3
        ),
        folding_pls_max_imbalance_ratio=float(
            getattr(args, "folding_pls_max_imbalance_ratio", 6.0) or 6.0
        ),
        folding_prefilter_k=(
            None
            if int(getattr(args, "folding_prefilter_k", 0) or 0) <= 0
            else int(getattr(args, "folding_prefilter_k", 0) or 0)
        ),
        enable_face_domain_projection=bool(getattr(args, "enable_face_domain_projection", False)),
        enable_ratio_features=bool(getattr(args, "enable_ratio_features", False)),
        ratio_pool_size=int(getattr(args, "ratio_pool_size", 80) or 80),
        ratio_selection_method=str(getattr(args, "ratio_selection_method", "ktsp") or "ktsp"),
        ratio_max_pairs=int(getattr(args, "ratio_max_pairs", 12000) or 12000),
        max_ratio_features=int(getattr(args, "max_ratio_features", 30) or 30),
        ratio_epsilon=float(getattr(args, "ratio_epsilon", 1e-6) or 1e-6),
        ratio_abs_value=bool(getattr(args, "ratio_abs_value", False)),
        ratio_require_positive=not bool(getattr(args, "ratio_allow_nonpositive", False)),
        multiomics_adapter=str(getattr(args, "multiomics_adapter", "none") or "none"),
        multiomics_integrator=str(
            getattr(args, "multiomics_integrator", "mb_plsda") or "mb_plsda"
        ),
        multiomics_n_components=int(getattr(args, "multiomics_n_components", 2) or 2),
        use_balanced_fs_subsample=args.enable_balanced_fs_subsample,
        fs_min_per_class=args.fs_min_per_class,
        fs_method_timeout_seconds=float(getattr(args, "fs_method_timeout_sec", 0.0) or 0.0),
        fs_linear_svm_max_iter=int(getattr(args, "fs_linear_svm_max_iter", 10000) or 10000),
        fs_runtime_racing_enabled=bool(getattr(args, "enable_fs_runtime_racing", False)),
        fs_runtime_racing_proxy_splits=int(getattr(args, "fs_runtime_racing_proxy_splits", 1) or 1),
        fs_runtime_racing_keep_fraction=float(
            getattr(args, "fs_runtime_racing_keep_fraction", 0.60) or 0.60
        ),
        fs_runtime_racing_min_candidates=int(
            getattr(args, "fs_runtime_racing_min_candidates", 4) or 4
        ),
        fs_runtime_racing_runtime_weight=float(
            getattr(args, "fs_runtime_racing_runtime_weight", 0.15) or 0.15
        ),
        fs_runtime_racing_mode=str(getattr(args, "fs_runtime_racing_mode", "single_stage") or "single_stage"),
        fs_runtime_racing_stages=int(getattr(args, "fs_runtime_racing_stages", 2) or 2),
        fs_runtime_racing_confidence_bound=str(
            getattr(args, "fs_runtime_racing_confidence_bound", "none") or "none"
        ),
        fs_runtime_racing_delta=float(getattr(args, "fs_runtime_racing_delta", 0.10) or 0.10),
        enabled_methods=enabled_method_stack,
        fs_portfolio_size=fs_portfolio_size,
        fs_portfolio_size_guard=str(getattr(args, "fs_portfolio_size_guard", "none") or "none"),
        fs_adaptive_portfolio_sizing_enabled=bool(adaptive_enabled),
        fs_adaptive_size_min=(
            None
            if adaptive_size_min_raw is None
            else int(adaptive_size_min_raw)
        ),
        fs_adaptive_size_max=(
            None
            if adaptive_size_max_raw is None
            else int(adaptive_size_max_raw)
        ),
        fs_adaptive_sizing_variance_penalty=bool(
            getattr(args, "fs_adaptive_sizing_variance_penalty", False)
        ),
        fs_adaptive_sizing_variance_penalty_strength=float(
            getattr(args, "fs_adaptive_sizing_variance_penalty_strength", 0.5) or 0.5
        ),
        # T-R-266: Pareto portfolio sizing.
        fs_pareto_portfolio_sizing_enabled=bool(
            getattr(args, "fs_pareto_portfolio_sizing_enabled", False)
        ),
        # T-R-271: stability-weighted portfolio aggregation.
        fs_stability_weighted_aggregation_enabled=bool(
            getattr(args, "fs_stability_weighted_aggregation_enabled", False)
        ),
        # VAL12_Suggestions: Post-FS feature count safety cap.
        **({
            "fs_max_selected_features_ratio": float(args.fs_max_selected_features_ratio),
        } if getattr(args, "fs_max_selected_features_ratio", None) is not None else {}),
        **({
            "fs_max_selected_features_cap": int(args.fs_max_selected_features_cap),
        } if getattr(args, "fs_max_selected_features_cap", None) is not None else {}),
        fs_mnpo_paradigm_aware_prior_enabled=bool(
            getattr(args, "enable_fs_mnpo_paradigm_aware_prior", False)
        ),
        fs_mnpo_interaction_floor=float(
            getattr(args, "fs_mnpo_interaction_floor", 0.12) or 0.12
        ),
        fs_rashomon_enabled=bool(effective_rashomon_enabled),
        fs_rashomon_max_models=int(getattr(args, "fs_rashomon_max_models", 12) or 12),
        fs_rashomon_score_tolerance=float(
            getattr(args, "fs_rashomon_score_tolerance", 0.01) or 0.01
        ),
        fs_mnpo_consensus_exclude_methods=tuple(getattr(args, "fs_mnpo_consensus_exclude_methods", ()) or ()),
        fs_mnpo_consensus_exclude_protect_top_k=int(
            getattr(args, "fs_mnpo_consensus_exclude_protect_top_k", 0) or 0
        ),
        fs_mnpo_include_legacy_consensus=not bool(getattr(args, "disable_fs_mnpo_legacy_consensus", False)),
        fs_mnpo_include_majority_consensus=not bool(getattr(args, "disable_fs_mnpo_majority_consensus", False)),
        fs_inner_cv_splits=int(getattr(args, "fs_inner_cv_splits", 3) or 3),
        fs_inner_cv_repeats=int(getattr(args, "fs_inner_cv_repeats", 1) or 1),
        use_stability_oracle=not bool(getattr(args, "disable_fs_stability_oracle", False)),
        use_complexity_oracle=not bool(getattr(args, "disable_fs_complexity_oracle", False)),
        use_robust_oracle=not bool(getattr(args, "disable_fs_robust_oracle", False)),
        fs_diversity_oracle_mode=args.fs_diversity_oracle_mode,
        fs_diversity_redundancy_weight=args.fs_diversity_redundancy_weight,
        fs_diversity_complementarity_weight=args.fs_diversity_complementarity_weight,
        fs_use_cvar_oracle=bool(getattr(args, "fs_use_cvar_oracle", False)),
        fs_cvar_alpha=float(getattr(args, "fs_cvar_alpha", 0.33) or 0.33),
        fs_oracle_weighting_mode=str(
            getattr(args, "fs_oracle_weighting_mode", "tritrust") or "tritrust"
        ),
        fs_shapley_n_coalitions_max=int(
            getattr(args, "fs_shapley_n_coalitions_max", 4096) or 4096
        ),
        fs_shapley_bayesian_shrinkage=bool(
            getattr(args, "fs_shapley_bayesian_shrinkage", False)
        ),
        fs_shapley_bayesian_prior_strength=float(
            getattr(args, "fs_shapley_bayesian_prior_strength", 8.0) or 8.0
        ),
        fs_use_interaction_oracle=bool(getattr(args, "fs_use_interaction_oracle", False)),
        fs_interaction_oracle_min_n_train=int(
            getattr(args, "fs_interaction_oracle_min_n_train", 150) or 150
        ),
        fs_interaction_oracle_pool_size_cap=int(
            getattr(args, "fs_interaction_oracle_pool_size_cap", 64) or 64
        ),
        fs_interaction_oracle_pair_cap=int(
            getattr(args, "fs_interaction_oracle_pair_cap", 20000) or 20000
        ),
        fs_use_ubayfs_oracle=bool(getattr(args, "fs_use_ubayfs_oracle", False)),
        fs_ubayfs_n_bootstrap=int(getattr(args, "fs_ubayfs_n_bootstrap", 32) or 32),
        fs_ubayfs_min_n=int(getattr(args, "fs_ubayfs_min_n", 100) or 100),
        fs_ubayfs_prior_weight=float(getattr(args, "fs_ubayfs_prior_weight", 0.0) or 0.0),
        fs_use_conformal_uq=bool(getattr(args, "fs_use_conformal_uq", False)),
        fs_conformal_uq_alpha=float(getattr(args, "fs_conformal_uq_alpha", 0.10) or 0.10),
        fs_conformal_uq_min_folds=int(getattr(args, "fs_conformal_uq_min_folds", 5) or 5),
        fs_fold_preference_mode=str(
            getattr(args, "fs_fold_preference_mode", "vote") or "vote"
        ),
        fs_use_conformal_efficiency=bool(
            getattr(args, "fs_use_conformal_efficiency", False)
        ),
        fs_conformal_efficiency_method=str(
            getattr(args, "fs_conformal_efficiency_method", "split") or "split"
        ),
        fs_oracle_weight_js_shrinkage=bool(
            getattr(args, "fs_oracle_weight_js_shrinkage", False)
        ),
        fs_oracle_complexity_conditioning=bool(
            getattr(args, "fs_oracle_complexity_conditioning", False)
        ),
        fs_payoff_shrinkage_kappa=float(
            getattr(args, "fs_payoff_shrinkage_kappa", 0.0) or 0.0
        ),
        fs_performance_balanced_weight=args.fs_performance_balanced_weight,
        fs_performance_macro_f1_weight=args.fs_performance_macro_f1_weight,
        fs_performance_use_adaptive_imbalance=args.enable_fs_adaptive_imbalance_score,
        fs_performance_imbalance_ratio_trigger=args.fs_imbalance_ratio_trigger,
        fs_performance_min_classes_for_adaptive=args.fs_imbalance_min_classes,
        fs_rank_aggregation_mode=args.fs_rank_aggregation_mode,
        fs_wrapper_refine_enabled=bool(effective_wrapper_refine_enabled),
        fs_wrapper_refine_top_k=args.fs_wrapper_refine_top_k,
        fs_wrapper_refine_max_add=args.fs_wrapper_refine_max_add,
        fs_wrapper_refine_min_gain=args.fs_wrapper_refine_min_gain,
        fs_ova_negative_ratio=args.fs_ova_negative_ratio,
        fs_ova_min_classes=args.fs_ova_min_classes,
        fs_ova_min_pos_samples=args.fs_ova_min_pos_samples,
        fs_ova_class_weight_mode=args.fs_ova_class_weight_mode,
        fs_ova_aggregation_mode=args.fs_ova_aggregation_mode,
        fs_ova_aggregation_p=args.fs_ova_aggregation_p,
        fs_ova_linear_backend=args.fs_ova_linear_backend,
        fs_ova_enable_calibration=bool(getattr(args, "enable_fs_ova_calibration", False)),
        fs_ova_calibration_cv=int(getattr(args, "fs_ova_calibration_cv", 3) or 3),
        fs_ecoc_min_classes=int(getattr(args, "fs_ecoc_min_classes", 4) or 4),
        fs_ecoc_max_ovo_pairs=int(getattr(args, "fs_ecoc_max_ovo_pairs", 8) or 8),
        fs_ecoc_random_code_bits=int(getattr(args, "fs_ecoc_random_code_bits", 4) or 4),
        fs_ecoc_class_complexity_weight=float(
            getattr(args, "fs_ecoc_class_complexity_weight", 1.0) or 1.0
        ),
        fs_ecoc_include_ova_tasks=not bool(getattr(args, "disable_fs_ecoc_include_ova_tasks", False)),
        fs_ecoc_negative_ratio=float(getattr(args, "fs_ecoc_negative_ratio", 2.0) or 2.0),
        fs_joint_multiclass_min_classes=int(getattr(args, "fs_joint_multiclass_min_classes", 3) or 3),
        fs_joint_multiclass_max_features=int(getattr(args, "fs_joint_multiclass_max_features", 256) or 256),
        fs_joint_multiclass_path_grid_size=int(getattr(args, "fs_joint_multiclass_path_grid_size", 6) or 6),
        fs_joint_multiclass_min_c=float(getattr(args, "fs_joint_multiclass_min_c", 0.05) or 0.05),
        fs_joint_multiclass_max_c=float(getattr(args, "fs_joint_multiclass_max_c", 1.6) or 1.6),
        fs_joint_multiclass_l1_ratio=float(getattr(args, "fs_joint_multiclass_l1_ratio", 0.55) or 0.55),
        fs_joint_multiclass_univariate_blend=float(
            getattr(args, "fs_joint_multiclass_univariate_blend", 0.20) or 0.20
        ),
        fs_dove_min_classes=int(getattr(args, "fs_dove_min_classes", 3) or 3),
        fs_dove_max_pairs_per_class=int(getattr(args, "fs_dove_max_pairs_per_class", 4) or 4),
        fs_dove_path_grid_size=int(getattr(args, "fs_dove_path_grid_size", 5) or 5),
        fs_dove_specificity_weight=float(getattr(args, "fs_dove_specificity_weight", 0.35) or 0.35),
        fs_dove_minority_boost=float(getattr(args, "fs_dove_minority_boost", 0.50) or 0.50),
        fs_sparse_multinomial_min_classes=int(
            getattr(args, "fs_sparse_multinomial_min_classes", 3) or 3
        ),
        fs_sparse_multinomial_max_features=int(
            getattr(args, "fs_sparse_multinomial_max_features", 320) or 320
        ),
        fs_sparse_multinomial_path_grid_size=int(
            getattr(args, "fs_sparse_multinomial_path_grid_size", 6) or 6
        ),
        fs_sparse_multinomial_min_c=float(getattr(args, "fs_sparse_multinomial_min_c", 0.05) or 0.05),
        fs_sparse_multinomial_max_c=float(getattr(args, "fs_sparse_multinomial_max_c", 1.6) or 1.6),
        fs_sparse_multinomial_backend=str(
            getattr(args, "fs_sparse_multinomial_backend", "mixed") or "mixed"
        ),
        fs_sparse_multinomial_l1_ratio=float(
            getattr(args, "fs_sparse_multinomial_l1_ratio", 0.70) or 0.70
        ),
        fs_sparse_multinomial_univariate_blend=float(
            getattr(args, "fs_sparse_multinomial_univariate_blend", 0.20) or 0.20
        ),
        fs_sparse_multinomial_max_iter=int(getattr(args, "fs_sparse_multinomial_max_iter", 5000) or 5000),
        fs_sparse_multinomial_screening_mode=_canonicalize_sparse_screening_mode(
            getattr(args, "fs_sparse_multinomial_screening_mode", "none"),
            warn_deprecated=True,
        ),
        fs_sparse_multinomial_screening_keep_fraction=float(
            getattr(args, "fs_sparse_multinomial_screening_keep_fraction", 1.0) or 1.0
        ),
        fs_sparse_multinomial_screening_min_features=int(
            getattr(args, "fs_sparse_multinomial_screening_min_features", 64) or 64
        ),
        fs_sparse_multinomial_screening_fallback_on_failure=not bool(
            getattr(args, "disable_fs_sparse_multinomial_screening_fallback_on_failure", False)
        ),
        fs_nsc_shrinkage_grid_size=int(getattr(args, "fs_nsc_shrinkage_grid_size", 6) or 6),
        fs_nsc_min_classes=int(getattr(args, "fs_nsc_min_classes", 3) or 3),
        fs_nsc_thresholding_mode=str(getattr(args, "fs_nsc_thresholding_mode", "soft") or "soft"),
        fs_nsc_order_quantile=float(getattr(args, "fs_nsc_order_quantile", 0.75) or 0.75),
        fs_nsc_deep_shrinkage_search=bool(getattr(args, "enable_fs_nsc_deep_shrinkage_search", False)),
        fs_class_pareto_min_classes=int(getattr(args, "fs_class_pareto_min_classes", 3) or 3),
        fs_class_pareto_top_per_class=int(getattr(args, "fs_class_pareto_top_per_class", 64) or 64),
        fs_class_pareto_global_fraction=float(
            getattr(args, "fs_class_pareto_global_fraction", 0.40) or 0.40
        ),
        fs_class_pareto_minority_boost=float(
            getattr(args, "fs_class_pareto_minority_boost", 0.50) or 0.50
        ),
        fs_class_pareto_kw_weight=float(getattr(args, "fs_class_pareto_kw_weight", 0.25) or 0.25),
        fs_sdr_min_classes=int(getattr(args, "fs_sdr_min_classes", 3) or 3),
        fs_sdr_prefilter_max_features=int(
            getattr(args, "fs_sdr_prefilter_max_features", 512) or 512
        ),
        fs_sdr_n_components=int(getattr(args, "fs_sdr_n_components", 3) or 3),
        fs_sdr_covariance_ridge=float(
            getattr(args, "fs_sdr_covariance_ridge", 1e-3) or 1e-3
        ),
        fs_per_class_quota_enabled=bool(getattr(args, "enable_fs_per_class_quota", False)),
        fs_per_class_quota_min_per_class=int(getattr(args, "fs_per_class_quota_min_per_class", 1) or 1),
        fs_per_class_quota_max_fraction=float(
            getattr(args, "fs_per_class_quota_max_fraction", 0.60) or 0.60
        ),
        fs_hsic_lasso_alpha=float(getattr(args, "fs_hsic_lasso_alpha", 0.01) or 0.01),
        fs_hsic_lasso_prefilter_max_features=int(
            getattr(args, "fs_hsic_lasso_prefilter_max_features", 128) or 128
        ),
        fs_hsic_lasso_feature_sigma=float(getattr(args, "fs_hsic_lasso_feature_sigma", 0.0) or 0.0),
        fs_hsic_lasso_target_sigma=float(getattr(args, "fs_hsic_lasso_target_sigma", 0.0) or 0.0),
        fs_hsic_lasso_relevance_blend=float(
            getattr(args, "fs_hsic_lasso_relevance_blend", 0.20) or 0.20
        ),
        fs_hsic_lasso_max_iter=int(getattr(args, "fs_hsic_lasso_max_iter", 4000) or 4000),
        fs_mrmr_mi_redundancy_enabled=bool(
            getattr(args, "enable_fs_mrmr_mi_redundancy", False)
        ),
        fs_mrmr_mi_n_bins=int(getattr(args, "fs_mrmr_mi_n_bins", 8) or 8),
        fs_cmim_min_samples=int(getattr(args, "fs_cmim_min_samples", 60) or 60),
        fs_cmim_n_bins=int(getattr(args, "fs_cmim_n_bins", 8) or 8),
        fs_fcbf_n_bins=int(getattr(args, "fs_fcbf_n_bins", 8) or 8),
        fs_ipss_path_grid_size=args.fs_ipss_path_grid_size,
        fs_ipss_min_c=args.fs_ipss_min_c,
        fs_ipss_max_c=args.fs_ipss_max_c,
        fs_ipss_target_fdr=args.fs_ipss_target_fdr,
        fs_ipss_null_shuffle_rounds=args.fs_ipss_null_shuffle_rounds,
        fs_ipss_use_eats_threshold=args.enable_fs_ipss_eats_threshold,
        fs_ipss_eats_exclusion_quantile=args.fs_ipss_eats_exclusion_quantile,
        fs_ipss_eats_min_threshold=args.fs_ipss_eats_min_threshold,
        fs_ipss_importance_model=args.fs_ipss_importance_model,
        fs_ipss_gate_min_classes=int(getattr(args, "fs_ipss_gate_min_classes", 0) or 0),
        fs_ipss_gate_min_p_over_n=float(getattr(args, "fs_ipss_gate_min_p_over_n", 0.0) or 0.0),
        fs_cluster_stability_corr_threshold=args.fs_cluster_corr_threshold,
        fs_cluster_stability_max_per_cluster=args.fs_cluster_max_per_cluster,
        fs_cluster_stability_min_cluster_freq=args.fs_cluster_min_freq,
        fs_stability_threshold_method=str(
            getattr(args, "fs_stability_threshold_method", "fixed") or "fixed"
        ).strip().lower(),
        fs_stability_target_pfer=float(
            getattr(args, "fs_stability_target_pfer", 1.0) or 1.0
        ),
        fs_stability_use_loss_guided_validation=args.enable_fs_stability_loss_guided_validation,
        fs_stability_validation_fraction=args.fs_stability_validation_fraction,
        fs_stability_validation_quantile=args.fs_stability_validation_quantile,
        fs_stability_validation_min_samples=args.fs_stability_validation_min_samples,
        fs_copula_knockoff_draws=args.fs_copula_knockoff_draws,
        fs_copula_alpha_kn=args.fs_copula_alpha_kn,
        fs_copula_alpha_ebh=args.fs_copula_alpha_ebh,
        fs_copula_truncation_level=args.fs_copula_truncation_level,
        fs_copula_generator=str(getattr(args, "fs_copula_generator", "copula") or "copula"),
        fs_copula_deepdrk_latent_fraction=float(
            getattr(args, "fs_copula_deepdrk_latent_fraction", 0.35) or 0.35
        ),
        fs_copula_deepdrk_noise_scale=float(
            getattr(args, "fs_copula_deepdrk_noise_scale", 1.0) or 1.0
        ),
        fs_copula_derandomize_runs=int(
            getattr(args, "fs_copula_derandomize_runs", 5) or 5
        ),
        fs_copula_stabilizer_runs=args.fs_copula_stabilizer_runs,
        fs_copula_stabilizer_use_ebh=args.enable_fs_copula_stabilizer_ebh,
        fs_copula_stabilizer_seed_stride=args.fs_copula_stabilizer_seed_stride,
        fs_importance_uq_enabled=bool(getattr(args, "fs_importance_uq_enabled", False)),
        fs_importance_uq_min_cv_folds=int(
            max(1, int(getattr(args, "fs_importance_uq_min_cv_folds", 3) or 3))
        ),
        fs_decorrelated_stability_eps=args.fs_decorrelated_stability_eps,
        fs_decorrelated_stability_min_max_abs_corr=float(
            getattr(args, "fs_decorrelated_stability_min_max_abs_corr", 0.0) or 0.0
        ),
        fs_iterative_pruning_pool_factor=float(
            getattr(args, "fs_iterative_pruning_pool_factor", 2.5) or 2.5
        ),
        fs_iterative_pruning_max_rounds=int(
            getattr(args, "fs_iterative_pruning_max_rounds", 32) or 32
        ),
        fs_iterative_pruning_min_improvement=float(
            getattr(args, "fs_iterative_pruning_min_improvement", -0.002) or -0.002
        ),
        fs_iterative_pruning_max_cumulative_loss=float(
            getattr(args, "fs_iterative_pruning_max_cumulative_loss", 0.02) or 0.02
        ),
        fs_iterative_pruning_redundancy_weight=float(
            getattr(args, "fs_iterative_pruning_redundancy_weight", 0.65) or 0.65
        ),
        fs_iterative_pruning_bounded_prefilter_cap=int(
            getattr(args, "fs_iterative_pruning_bounded_prefilter_cap", 220) or 220
        ),
        fs_iterative_pruning_bounded_candidate_fraction=float(
            getattr(args, "fs_iterative_pruning_bounded_candidate_fraction", 0.35) or 0.35
        ),
        fs_iterative_pruning_bounded_min_candidates=int(
            getattr(args, "fs_iterative_pruning_bounded_min_candidates", 4) or 4
        ),
        fs_iterative_pruning_bounded_max_evaluations=int(
            getattr(args, "fs_iterative_pruning_bounded_max_evaluations", 48) or 48
        ),
        fs_iterative_pruning_bounded_max_runtime_seconds=float(
            getattr(args, "fs_iterative_pruning_bounded_max_runtime_seconds", 30.0) or 30.0
        ),
        fs_iterative_pruning_bounded_enable_class_gating=not bool(
            getattr(args, "disable_fs_iterative_pruning_bounded_class_gating", False)
        ),
        fs_iterative_pruning_bounded_multiclass_scale=float(
            getattr(args, "fs_iterative_pruning_bounded_multiclass_scale", 0.70) or 0.70
        ),
        fs_iterative_pruning_bounded_imbalance_trigger=float(
            getattr(args, "fs_iterative_pruning_bounded_imbalance_trigger", 2.5) or 2.5
        ),
        fs_iterative_pruning_bounded_imbalance_scale=float(
            getattr(args, "fs_iterative_pruning_bounded_imbalance_scale", 0.75) or 0.75
        ),
        fs_iterative_pruning_bounded_use_cpss_overlay=bool(effective_iter_pruning_bounded_cpss),
        fs_iterative_pruning_bounded_cpss_pairs=int(
            getattr(args, "fs_iterative_pruning_bounded_cpss_pairs", 4) or 4
        ),
        fs_iterative_pruning_bounded_cpss_stability_threshold=float(
            getattr(args, "fs_iterative_pruning_bounded_cpss_stability_threshold", 0.60) or 0.60
        ),
        fs_iterative_pruning_bounded_cpss_min_stable_features=int(
            getattr(args, "fs_iterative_pruning_bounded_cpss_min_stable_features", 2) or 2
        ),
        fs_iterative_pruning_bounded_cpss_min_jaccard=float(
            getattr(args, "fs_iterative_pruning_bounded_cpss_min_jaccard", 0.35) or 0.35
        ),
        fs_iterative_pruning_bounded_cpss_max_score_drop=float(
            getattr(args, "fs_iterative_pruning_bounded_cpss_max_score_drop", 0.005) or 0.005
        ),
        fs_iterative_pruning_class_pareto_prefilter_enabled=bool(effective_iter_pruning_pareto_prefilter),
        fs_iterative_pruning_class_pareto_min_classes=int(
            getattr(args, "fs_iterative_pruning_class_pareto_min_classes", 3) or 3
        ),
        fs_iterative_pruning_class_pareto_top_per_class=int(
            getattr(args, "fs_iterative_pruning_class_pareto_top_per_class", 64) or 64
        ),
        fs_iterative_pruning_class_pareto_global_fraction=float(
            getattr(args, "fs_iterative_pruning_class_pareto_global_fraction", 0.40) or 0.40
        ),
        fs_iterative_pruning_class_pareto_minority_boost=float(
            getattr(args, "fs_iterative_pruning_class_pareto_minority_boost", 0.50) or 0.50
        ),
        fs_iterative_pruning_class_pareto_stability_gate_enabled=bool(effective_iter_pruning_pareto_stability_gate),
        fs_iterative_pruning_class_pareto_stability_subsamples=int(
            getattr(args, "fs_iterative_pruning_class_pareto_stability_subsamples", 6) or 6
        ),
        fs_iterative_pruning_class_pareto_stability_fraction=float(
            getattr(args, "fs_iterative_pruning_class_pareto_stability_fraction", 0.70) or 0.70
        ),
        fs_iterative_pruning_class_pareto_stability_threshold=float(
            getattr(args, "fs_iterative_pruning_class_pareto_stability_threshold", 0.55) or 0.55
        ),
        fs_iterative_pruning_class_pareto_stability_min_overlap=float(
            getattr(args, "fs_iterative_pruning_class_pareto_stability_min_overlap", 0.50) or 0.50
        ),
        fs_iterative_pruning_class_pareto_stability_min_stable_features=int(
            getattr(args, "fs_iterative_pruning_class_pareto_stability_min_stable_features", 4) or 4
        ),
        fs_iterative_pruning_class_pareto_stability_fallback_on_failure=not bool(
            getattr(args, "disable_fs_iterative_pruning_class_pareto_stability_fallback_on_failure", False)
        ),
        use_diversity_oracle=args.enable_diversity_oracle,
        use_tail_risk_oracle=False,
        tail_risk_alpha=float(getattr(args, "fs_tail_risk_alpha", 0.33) or 0.33),
        use_regret_oracle=False,
        use_qre_smoothing=bool(effective_fs_qre),
        qre_temperature_gamma=float(getattr(args, "fs_qre_temperature_gamma", 1.0) or 1.0),
        use_oracle_redundancy_penalty=bool(effective_fs_oracle_redundancy),
        compute_tremble_sensitivity=bool(getattr(args, "fs_compute_tremble_sensitivity", False)),
        classification_selection_mode=str(
            getattr(args, "classifier_selection_mode", "legacy") or "legacy"
        ),
        model_candidates=model_candidates,
        exclude_model_candidates=tuple(exclude_model_candidates),
        classifier_regime_candidate_exclusions=tuple(regime_candidate_exclusions),
        classifier_oracle_complexity_prior_overrides=tuple(complexity_prior_overrides),
        include_elastic_net_model=include_elastic,
        include_rf_model=include_rf,
        include_knn_model=include_knn,
        include_svm_linear_model=include_svm_linear,
        include_dlda_model=include_dlda,
        include_nsc_model=include_nsc,
        include_pls_da_model=include_pls_da,
        include_gpc_model=include_gpc,
        include_nb_model=include_nb,
        include_vote_ensemble_model=include_vote_ensemble,
        include_rp_ensemble_model=include_rp_ensemble,
        include_dbda_model=include_dbda,
        include_gqda_model=include_gqda,
        include_bc_svm_linear_model=include_bc_svm_linear,
        include_sglnn_model=include_sglnn,
        include_xgb_model=include_xgb,
        include_lgbm_model=include_lgbm,
        include_extra_tree_model=include_extra_tree,
        include_catboost_model=include_catboost,
        include_tabpfn_model=include_tabpfn,
        model_cv_lr_max_iter=int(getattr(args, "model_cv_lr_max_iter", 10000) or 10000),
        model_cv_use_hybrid_score=args.enable_hybrid_model_cv,
        model_cv_balanced_weight=args.model_cv_balanced_weight,
        model_cv_macro_f1_weight=args.model_cv_macro_f1_weight,
        model_cv_runtime_containment_enabled=bool(
            getattr(args, "enable_model_cv_runtime_containment", False)
        ),
        model_cv_runtime_max_candidates=int(
            getattr(args, "model_cv_runtime_max_candidates", 0) or 0
        ),
        model_cv_runtime_high_p_over_n_threshold=float(
            getattr(args, "model_cv_runtime_high_p_over_n_threshold", 40.0) or 40.0
        ),
        model_cv_runtime_high_class_threshold=int(
            getattr(args, "model_cv_runtime_high_class_threshold", 6) or 6
        ),
        model_cv_runtime_min_class_count_threshold=int(
            getattr(args, "model_cv_runtime_min_class_count_threshold", 12) or 12
        ),
        classifier_oracle_k=int(getattr(args, "classifier_oracle_k", 1) or 1),
        classifier_oracle_weighting_mode=str(
            getattr(args, "classifier_oracle_weighting_mode", "tritrust") or "tritrust"
        ),
        classifier_oracle_include_calibration=not bool(
            getattr(args, "disable_classifier_oracle_calibration", False)
        ),
        classifier_oracle_include_james_stein=not bool(
            getattr(args, "disable_classifier_oracle_james_stein", False)
        ),
        classifier_oracle_complexity_shrinkage=bool(
            getattr(args, "classifier_oracle_complexity_shrinkage", False)
        ),
        classifier_oracle_include_cvar=bool(
            getattr(args, "enable_classifier_oracle_cvar", False)
        ),
        classifier_oracle_cvar_alpha=float(
            getattr(args, "classifier_oracle_cvar_alpha", 0.33) or 0.33
        ),
        classifier_oracle_use_dynamic_complexity=bool(
            getattr(args, "enable_classifier_oracle_dynamic_complexity", False)
        ),
        classifier_oracle_portfolio_diversity=bool(
            getattr(args, "enable_classifier_oracle_portfolio_diversity", False)
        ),
        classifier_oracle_portfolio_overlap_threshold=float(
            getattr(args, "classifier_oracle_portfolio_overlap_threshold", 0.75) or 0.75
        ),
        classifier_oracle_portfolio_corr_threshold=float(
            getattr(args, "classifier_oracle_portfolio_corr_threshold", 0.85) or 0.85
        ),
        classifier_oracle_enable_hoeffding_racing=not bool(
            getattr(args, "disable_classifier_oracle_hoeffding_racing", False)
        ),
        classifier_oracle_hoeffding_delta=float(
            getattr(args, "classifier_oracle_hoeffding_delta", 0.10) or 0.10
        ),
        classifier_oracle_enable_bbc=not bool(
            getattr(args, "disable_classifier_oracle_bbc", False)
        ),
        classifier_oracle_bbc_bootstrap_rounds=int(
            getattr(args, "classifier_oracle_bbc_bootstrap_rounds", 200) or 200
        ),
        classifier_oracle_bbc_ci_level=float(
            getattr(args, "classifier_oracle_bbc_ci_level", 0.90) or 0.90
        ),
        classifier_oracle_enable_ensemble=bool(
            getattr(args, "enable_classifier_oracle_ensemble", False)
        ),
        classifier_oracle_ensemble_voting_mode=str(
            getattr(args, "classifier_oracle_ensemble_voting_mode", "hard") or "hard"
        ),
        classifier_oracle_greedy_ensemble=bool(
            getattr(args, "enable_classifier_oracle_greedy_ensemble", False)
        ),
        classifier_oracle_greedy_ensemble_rounds=int(
            getattr(args, "classifier_oracle_greedy_ensemble_rounds", 10) or 10
        ),
        classifier_oracle_candidate_pruning=bool(
            getattr(args, "enable_classifier_oracle_candidate_pruning", False)
        ),
        classifier_oracle_candidate_pruning_threshold=float(
            getattr(args, "classifier_oracle_candidate_pruning_threshold", 0.0) or 0.0
        ),
        classifier_oracle_incumbent_early_stopping=bool(
            getattr(args, "enable_classifier_oracle_incumbent_early_stopping", False)
        ),
        classifier_oracle_behavior_profile=str(
            getattr(args, "classifier_oracle_behavior_profile", "current") or "current"
        ).strip().lower(),
        classifier_oracle_use_per_family_flaml=not bool(
            getattr(args, "disable_classifier_oracle_per_family_flaml", False)
        ),
        stage2_ratio_augmentation_enabled=bool(
            getattr(args, "enable_stage2_ratio_augmentation", False)
        ),
        stage2_ratio_max_features=int(
            getattr(args, "stage2_ratio_max_features", 16) or 16
        ),
        stage2_ratio_selection_method=str(
            getattr(args, "stage2_ratio_selection_method", "correlation") or "correlation"
        ),
        stage2_ratio_epsilon=float(
            getattr(args, "stage2_ratio_epsilon", 1e-6) or 1e-6
        ),
        enable_maqc_pairing=bool(getattr(args, "enable_maqc_pairing", False)),
        maqc_pairing_method_sets=maqc_method_sets,
        maqc_pairing_method_set_names=maqc_names,
        maqc_pairing_min_improvement=float(getattr(args, "maqc_pairing_min_improvement", 0.0) or 0.0),
        maqc_pairing_min_improvement_se_mult=float(getattr(args, "maqc_pairing_min_improvement_se_mult", 0.0) or 0.0),
        meta_learning_selector_mode=str(
            getattr(args, "meta_learning_selector", "none") or "none"
        ),
        meta_learning_confidence_threshold=float(
            getattr(args, "meta_learning_confidence_threshold", 0.55) or 0.55
        ),
        meta_learning_records_path=str(
            getattr(args, "meta_learning_records_path", "") or ""
        ).strip(),
    )
    _apply_integrated_scenario_defaults(cfg, spec)
    return cfg


@contextmanager
def _task_timeout(seconds: float) -> Any:
    if seconds <= 0:
        yield
        return
    # SIGALRM is POSIX-only; on unsupported platforms we run without timeout.
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    # FIX T-A3-FIX-005: SIGALRM only works in the main thread.
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Timed out after {seconds:.1f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


@contextmanager
def _maybe_suppress_worker_logs(enabled: bool) -> Any:
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _pipeline_run_worker(
    result_queue: Any,
    cfg: DFFSConfig,
    X: np.ndarray,
    y: np.ndarray,
    batch_labels: Optional[np.ndarray],
    dataset_name: str,
    seed: int,
    quiet_worker_logs: bool,
    capture_artifacts: bool,
    capture_diagnostics: bool,
) -> None:
    # Defence-in-depth: re-apply threading limits inside spawned worker.
    # The env-var guard at module top covers the common case, but spawn
    # children re-import the module and threadpoolctl ensures runtime
    # enforcement even if the env vars were not inherited.
    for _k, _v in {
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }.items():
        os.environ[_k] = _v
    # Block CUDA on CPU-only hosts to avoid cuBLAS deadlock.
    if not (os.path.isdir("/proc/driver/nvidia") or any(
        os.path.exists(f"/dev/nvidia{i}") for i in range(4)
    )):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        from threadpoolctl import threadpool_limits as _tpl
        _tpl_ctx = _tpl(limits=1)
        _tpl_ctx.__enter__()
    except Exception:
        _tpl_ctx = None
    # Constrain torch internal threads to prevent cuBLAS contention.
    try:
        import torch as _torch_worker
        _torch_worker.set_num_threads(1)
        _torch_worker.set_num_interop_threads(1)
        del _torch_worker
    except Exception:
        pass

    result_path: Optional[str] = None
    try:
        with _maybe_suppress_worker_logs(bool(quiet_worker_logs)):
            pipeline = DistributionFeatureSelectionPipeline(cfg)
            result = pipeline.run(
                X,
                y,
                dataset_name=dataset_name,
                seed=seed,
                batch_labels=batch_labels,
                capture_artifacts=bool(capture_artifacts),
                capture_diagnostics=bool(capture_diagnostics),
            )
        fd, tmp_path = tempfile.mkstemp(prefix="tabnetics_hard_timeout_result_", suffix=".pkl")
        result_path = str(tmp_path)
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        result_queue.put({"ok": True, "result_path": result_path})
        result_path = None
    except Exception as exc:
        if result_path:
            try:
                os.remove(result_path)
            except Exception:
                pass
        try:
            result_queue.put({"ok": False, "error": str(exc)})
        except Exception:
            pass


def _run_pipeline_with_hard_timeout(
    cfg: DFFSConfig,
    X: np.ndarray,
    y: np.ndarray,
    batch_labels: Optional[np.ndarray] = None,
    dataset_name: str = "",
    seed: int = 0,
    timeout_sec: float = 0.0,
    quiet_worker_logs: bool = False,
    use_hard_kill: bool = True,
    capture_artifacts: bool = False,
    capture_diagnostics: bool = False,
):
    timeout = float(timeout_sec)
    if timeout <= 0.0:
        with _maybe_suppress_worker_logs(bool(quiet_worker_logs)):
            pipeline = DistributionFeatureSelectionPipeline(cfg)
            return pipeline.run(
                X,
                y,
                dataset_name=dataset_name,
                seed=seed,
                batch_labels=batch_labels,
                capture_artifacts=bool(capture_artifacts),
                capture_diagnostics=bool(capture_diagnostics),
            )

    if not bool(use_hard_kill):
        with _task_timeout(timeout):
            with _maybe_suppress_worker_logs(bool(quiet_worker_logs)):
                pipeline = DistributionFeatureSelectionPipeline(cfg)
                return pipeline.run(
                    X,
                    y,
                    dataset_name=dataset_name,
                    seed=seed,
                    batch_labels=batch_labels,
                    capture_artifacts=bool(capture_artifacts),
                    capture_diagnostics=bool(capture_diagnostics),
                )

    def _is_running_in_loky_worker() -> bool:
        # Heuristic: when this process itself is a joblib loky worker,
        # prefer spawn to avoid fork-from-worker lock inheritance.
        try:
            pname = str(mp.current_process().name or "").strip().lower()
        except Exception as exc:
            pname = ""
        if "loky" in pname:
            return True
        try:
            with open("/proc/self/cmdline", "rb") as fh:
                cmdline = fh.read().decode("utf-8", errors="ignore").replace("\x00", " ").lower()
            if "joblib.externals.loky.backend.popen_loky_posix" in cmdline:
                return True
        except Exception as exc:
            pass
        return False

    def _resolve_hard_timeout_start_method() -> Optional[str]:
        methods = set(mp.get_all_start_methods())
        override = str(os.environ.get("TABNETICS_HARD_TIMEOUT_START_METHOD", "auto") or "auto").strip().lower()
        if override in {"none", "signal"}:
            return None
        if override in {"fork", "spawn", "forkserver"}:
            if override in methods:
                return override
            warnings.warn(
                f"TABNETICS_HARD_TIMEOUT_START_METHOD={override!r} unavailable; falling back to auto.",
                RuntimeWarning,
            )
        # Auto policy:
        # 1) inside loky worker -> spawn (safer than fork in nested process pools)
        # 2) otherwise prefer fork for lower serialization overhead
        # 3) fallback to other available methods
        if _is_running_in_loky_worker() and "spawn" in methods:
            return "spawn"
        if "fork" in methods:
            return "fork"
        if "spawn" in methods:
            return "spawn"
        if "forkserver" in methods:
            return "forkserver"
        return None

    start_method = _resolve_hard_timeout_start_method()
    if start_method is None:
        with _task_timeout(timeout):
            with _maybe_suppress_worker_logs(bool(quiet_worker_logs)):
                pipeline = DistributionFeatureSelectionPipeline(cfg)
                return pipeline.run(
                    X,
                    y,
                    dataset_name=dataset_name,
                    seed=seed,
                    batch_labels=batch_labels,
                    capture_artifacts=bool(capture_artifacts),
                    capture_diagnostics=bool(capture_diagnostics),
                )

    ctx = mp.get_context(start_method)
    worker_fn = _pipeline_run_worker
    worker_args: List[Any] = [
        None,  # filled below to keep queue context-local.
        cfg,
        X,
        y,
        batch_labels,
        dataset_name,
        int(seed),
        bool(quiet_worker_logs),
    ]
    try:
        worker_params = inspect.signature(worker_fn).parameters
    except Exception as exc:
        worker_params = {}
    worker_args: List[Any] = [None, cfg, X, y]
    # Keep the worker-launch path backward compatible with older test helpers
    # and monkeypatched workers that predate batch-label propagation.
    if "batch_labels" in worker_params:
        worker_args.append(batch_labels)
    worker_args.extend(
        [
            dataset_name,
            int(seed),
            bool(quiet_worker_logs),
        ]
    )
    if "capture_artifacts" in worker_params:
        worker_args.append(bool(capture_artifacts))
    if "capture_diagnostics" in worker_params:
        worker_args.append(bool(capture_diagnostics))

    def _launch_with_context(context: mp.context.BaseContext) -> Tuple[Any, Any]:
        q = context.Queue(maxsize=1)
        args_local = list(worker_args)
        args_local[0] = q
        p = context.Process(
            target=worker_fn,
            args=tuple(args_local),
        )
        p.daemon = True
        p.start()
        return p, q

    try:
        proc, result_queue = _launch_with_context(ctx)
    except Exception as exc:
        start_methods = set(mp.get_all_start_methods())
        # Fallback: keep legacy fork path if spawn/forkserver launch fails
        # (e.g., test monkeypatches with local callables).
        if start_method != "fork" and "fork" in start_methods:
            warnings.warn(
                f"hard-timeout worker start_method={start_method!r} failed ({exc}); falling back to 'fork'.",
                RuntimeWarning,
            )
            ctx = mp.get_context("fork")
            proc, result_queue = _launch_with_context(ctx)
        else:
            raise

    payload: Optional[Dict[str, Any]] = None
    deadline = float(time.monotonic() + timeout)
    poll_sec = float(min(0.25, max(0.05, timeout * 0.01)))

    # Read from the queue while the worker is still running. Waiting on join()
    # first can deadlock when the worker is blocked flushing a large payload.
    while payload is None:
        remaining = float(deadline - time.monotonic())
        if remaining <= 0.0:
            break
        try:
            payload = result_queue.get(timeout=min(poll_sec, remaining))
            break
        except queue.Empty:
            if not proc.is_alive():
                break

    if payload is None:
        if proc.is_alive():
            proc.kill()
            proc.join()
            try:
                result_queue.close()
                result_queue.join_thread()
            except Exception as exc:
                pass
            raise TimeoutError(f"Timed out after {timeout:.1f}s (hard-killed worker process)")

        result_wait_sec = float(min(1.0, max(0.05, timeout * 0.1)))
        try:
            payload = result_queue.get(timeout=result_wait_sec)
        except queue.Empty:
            payload = {
                "ok": False,
                "error": f"Worker exited without result payload (exit_code={proc.exitcode})",
            }

    if proc.is_alive():
        proc.join(timeout=0.2)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
    if proc.is_alive():
        proc.kill()
        proc.join()

    try:
        result_queue.close()
        result_queue.join_thread()
    except Exception as exc:
        pass

    payload_ok = bool(payload.get("ok", False))
    payload_result_path = str(payload.get("result_path", "") or "").strip()
    try:
        if payload_ok:
            if payload_result_path:
                with open(payload_result_path, "rb") as fh:
                    return pickle.load(fh)
            return payload.get("result")
        raise RuntimeError(str(payload.get("error", "Unknown worker failure")))
    finally:
        if payload_result_path:
            try:
                os.remove(payload_result_path)
            except Exception:
                pass


def _mean_std_ci(values: Sequence[float], ci_level: float = 0.95) -> Tuple[float, float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, 0.0, mean, mean
    std = float(np.std(arr, ddof=1))
    se = std / float(np.sqrt(arr.size))
    if not np.isfinite(se) or se <= 0.0:
        return mean, std, mean, mean
    alpha = float(max(0.0, min(1.0, ci_level)))
    q = 0.5 * (1.0 + alpha)
    try:
        tcrit = float(sps.t.ppf(q, df=int(arr.size - 1)))
    except Exception as exc:
        tcrit = 1.96
    half = float(tcrit * se)
    return mean, std, float(mean - half), float(mean + half)


def _is_timeout_error_message(message: str) -> bool:
    text = str(message).strip().lower()
    if not text:
        return False
    return ("timed out after" in text) or ("hard-killed worker process" in text)


def _count_timeout_failures(failures: Sequence[Dict[str, Any]]) -> int:
    count = 0
    for item in failures:
        if _is_timeout_error_message(str(item.get("error", ""))):
            count += 1
    return int(count)


def _emit_progress_event(progress_queue: Optional[Any], payload: Dict[str, Any]) -> None:
    if progress_queue is None:
        return
    try:
        progress_queue.put_nowait(payload)
        return
    except Exception as exc:
        pass
    try:
        progress_queue.put(payload, timeout=0.05)
    except Exception as exc:
        # Progress telemetry must never break benchmark execution.
        pass


def _safe_progress_queue_depth(progress_queue: Any) -> str:
    try:
        depth = progress_queue.qsize()
    except Exception as exc:
        return "na"
    try:
        depth_int = int(depth)
    except Exception as exc:
        return "na"
    if depth_int < 0:
        return "na"
    return str(depth_int)


def _collect_descendant_wait_snapshot(max_items: int = 12) -> List[str]:
    """Capture a compact /proc-based wait-state snapshot for descendant processes."""
    try:
        max_items_int = max(1, int(max_items))
    except Exception as exc:
        max_items_int = 12

    root_pid = int(os.getpid())
    parent_by_pid: Dict[int, int] = {}
    state_by_pid: Dict[int, str] = {}
    comm_by_pid: Dict[int, str] = {}

    try:
        proc_entries = list(os.listdir("/proc"))
    except Exception as exc:
        return []

    for entry in proc_entries:
        if not str(entry).isdigit():
            continue
        pid = int(entry)
        stat_path = f"/proc/{pid}/stat"
        try:
            with open(stat_path, "r", encoding="utf-8", errors="replace") as fh:
                stat_line = fh.read().strip()
        except Exception as exc:
            continue
        match = re.match(r"^(\d+)\s+\((.+)\)\s+([A-Za-z])\s+(\d+)\s+", stat_line)
        if match is None:
            continue
        try:
            ppid = int(match.group(4))
        except Exception as exc:
            continue
        parent_by_pid[pid] = ppid
        state_by_pid[pid] = str(match.group(3))
        comm_by_pid[pid] = str(match.group(2))

    children: Dict[int, List[int]] = {}
    for pid, ppid in parent_by_pid.items():
        children.setdefault(int(ppid), []).append(int(pid))

    descendants: List[int] = []
    stack: List[int] = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = int(stack.pop())
        if pid in seen:
            continue
        seen.add(pid)
        descendants.append(pid)
        for child in children.get(pid, []):
            stack.append(int(child))

    if not descendants:
        return []

    rows: List[Tuple[int, int, str]] = []
    for pid in descendants:
        ppid = int(parent_by_pid.get(pid, -1))
        state = str(state_by_pid.get(pid, "?") or "?")
        comm = str(comm_by_pid.get(pid, "?") or "?")

        wchan = "-"
        try:
            with open(f"/proc/{pid}/wchan", "r", encoding="utf-8", errors="replace") as fh:
                wchan_raw = fh.read().strip()
            if wchan_raw:
                wchan = str(wchan_raw)
        except Exception as exc:
            pass

        cmd = comm
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                raw = fh.read()
            if raw:
                txt = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
                if txt:
                    cmd = txt
        except Exception as exc:
            pass

        if len(cmd) > 180:
            cmd = cmd[:177] + "..."

        cmd_lc = cmd.lower()
        wchan_lc = wchan.lower()
        score = 0
        if "loky" in cmd_lc or "popen_loky_posix" in cmd_lc:
            score += 4
        if "futex" in wchan_lc or "poll" in wchan_lc:
            score += 3
        if state in {"D", "S"}:
            score += 2
        if pid == root_pid:
            score -= 1

        row = f"pid={pid} ppid={ppid} state={state} wchan={wchan} cmd={cmd}"
        rows.append((score, pid, row))

    rows.sort(key=lambda item: (-int(item[0]), int(item[1])))
    return [row for _, _, row in rows[:max_items_int]]


def _build_dataset_task_job(
    ds_id: str,
    seed: int,
    args: argparse.Namespace,
    progress_queue: Optional[Any] = None,
) -> Any:
    task_fn = _run_dataset_seed_task
    try:
        params = inspect.signature(task_fn).parameters
    except Exception as exc:
        params = {}
    if progress_queue is not None and "progress_queue" in params:
        return delayed(task_fn)(ds_id, seed, args, progress_queue=progress_queue)
    return delayed(task_fn)(ds_id, seed, args)


def _monitor_parallel_progress(
    progress_queue: Any,
    stop_event: threading.Event,
    total_tasks: int,
    heartbeat_sec: float,
    watchdog_sec: float,
    stall_watchdog_sec: float,
) -> None:
    heartbeat = float(heartbeat_sec)
    if not np.isfinite(heartbeat) or heartbeat <= 0.0:
        heartbeat = 60.0
    heartbeat = max(5.0, heartbeat)

    watchdog = float(watchdog_sec)
    if not np.isfinite(watchdog) or watchdog < 0.0:
        watchdog = 0.0

    stall_watchdog = float(stall_watchdog_sec)
    if not np.isfinite(stall_watchdog) or stall_watchdog < 0.0:
        stall_watchdog = 0.0

    running: Dict[Tuple[str, int], Dict[str, Any]] = {}
    warned_stages: Dict[Tuple[str, int], int] = {}
    completed = 0
    last_heartbeat = time.time()
    last_completion_ts = last_heartbeat
    stall_warn_stage = 0
    idle_cycles = 0

    def _task_key(evt: Dict[str, Any]) -> Tuple[str, int]:
        ds_id = str(evt.get("dataset_id", "unknown"))
        try:
            seed_val = int(evt.get("seed", -1))
        except Exception as exc:
            seed_val = -1
        return (ds_id, seed_val)

    def _elapsed(start_ts: float, now_ts: float) -> float:
        try:
            return max(0.0, float(now_ts) - float(start_ts))
        except Exception as exc:
            return 0.0

    def _oldest_running(now_ts: float, limit: int = 3) -> str:
        if not running:
            return "none"
        ranked = sorted(
            running.items(),
            key=lambda item: _elapsed(float(item[1].get("start_ts", now_ts)), now_ts),
            reverse=True,
        )
        out: List[str] = []
        for (ds_id, seed_val), state in ranked[: max(1, int(limit))]:
            tier = str(state.get("tier", "?"))
            elapsed_sec = _elapsed(float(state.get("start_ts", now_ts)), now_ts)
            out.append(f"{ds_id}[{tier}] seed={seed_val} {elapsed_sec:.0f}s")
        return "; ".join(out)

    def _handle_event(evt: Dict[str, Any], now_ts: float) -> None:
        nonlocal completed, last_completion_ts, stall_warn_stage
        event_name = str(evt.get("event", "")).strip().lower()
        ds_id, seed_val = _task_key(evt)
        key = (ds_id, seed_val)

        if event_name == "task_start":
            tier = str(evt.get("tier", "?"))
            pid = evt.get("pid")
            start_ts = float(evt.get("ts", now_ts) or now_ts)
            running[key] = {"start_ts": start_ts, "tier": tier}
            warned_stages[key] = 0
            if pid is None:
                print(f"[task-start] dataset={ds_id} tier={tier} seed={seed_val}", flush=True)
            else:
                print(
                    f"[task-start] dataset={ds_id} tier={tier} seed={seed_val} pid={pid}",
                    flush=True,
                )
            return

        if event_name == "task_done":
            state = running.pop(key, None)
            warned_stages.pop(key, None)
            completed += 1
            tier = str(evt.get("tier", (state or {}).get("tier", "?")))
            rows_count = int(evt.get("rows_count", 0) or 0)
            failures_count = int(evt.get("failures_count", 0) or 0)
            timeout_failures = int(evt.get("timeout_failures", 0) or 0)
            status = str(evt.get("status", "ok") or "ok")
            if "elapsed_sec" in evt and evt.get("elapsed_sec") is not None:
                elapsed_sec = float(evt.get("elapsed_sec", 0.0) or 0.0)
            elif state is not None:
                elapsed_sec = _elapsed(float(state.get("start_ts", now_ts)), now_ts)
            else:
                elapsed_sec = 0.0
            last_completion_ts = float(now_ts)
            stall_warn_stage = 0
            print(
                f"[task-done] dataset={ds_id} tier={tier} seed={seed_val} "
                f"status={status} elapsed={elapsed_sec:.1f}s rows={rows_count} "
                f"failures={failures_count} timeout_failures={timeout_failures}",
                flush=True,
            )
            return

        if event_name in {"config_timeout", "config_error"}:
            cfg = str(evt.get("config", "unknown"))
            err = str(evt.get("error", "")).replace("\n", " ").strip()
            if len(err) > 220:
                err = err[:217] + "..."
            print(
                f"[{event_name}] dataset={ds_id} seed={seed_val} config={cfg} error={err}",
                flush=True,
            )

    while True:
        now = time.time()
        remaining = heartbeat - (now - last_heartbeat)
        poll_timeout = min(1.0, max(0.05, remaining))
        evt: Optional[Dict[str, Any]] = None
        try:
            raw = progress_queue.get(timeout=poll_timeout)
            if isinstance(raw, dict):
                evt = raw
            idle_cycles = 0
        except queue.Empty:
            idle_cycles += 1
        except Exception as exc:
            idle_cycles += 1

        now = time.time()
        if evt is not None:
            _handle_event(evt, now)

        if now - last_heartbeat >= heartbeat:
            no_completion_sec = max(0.0, float(now - last_completion_ts))
            print(
                f"[heartbeat] completed={completed}/{total_tasks} running={len(running)} "
                f"oldest={_oldest_running(now)} no_done_for={no_completion_sec:.0f}s",
                flush=True,
            )
            last_heartbeat = now

        if watchdog > 0.0 and running:
            for key, state in list(running.items()):
                elapsed_sec = _elapsed(float(state.get("start_ts", now)), now)
                stage = int(elapsed_sec // watchdog)
                prev = int(warned_stages.get(key, 0) or 0)
                if stage >= 1 and stage > prev:
                    warned_stages[key] = stage
                    ds_id, seed_val = key
                    tier = str(state.get("tier", "?"))
                    print(
                        f"[watchdog] dataset={ds_id} tier={tier} seed={seed_val} "
                        f"running_for={elapsed_sec:.1f}s threshold={watchdog:.1f}s stage={stage}",
                        flush=True,
                    )

        if stall_watchdog > 0.0 and running:
            no_completion_sec = max(0.0, float(now - last_completion_ts))
            stage = int(no_completion_sec // stall_watchdog)
            if stage >= 1 and stage > stall_warn_stage:
                stall_warn_stage = int(stage)
                print(
                    f"[stall-watchdog] no task completion for {no_completion_sec:.1f}s "
                    f"completed={completed}/{total_tasks} running={len(running)} "
                    f"queue_depth={_safe_progress_queue_depth(progress_queue)} "
                    f"oldest={_oldest_running(now)} stage={stage}",
                    flush=True,
                )
                snapshot = _collect_descendant_wait_snapshot(max_items=12)
                if snapshot:
                    for line in snapshot:
                        print(f"[stall-watchdog] process {line}", flush=True)
                else:
                    print("[stall-watchdog] process snapshot unavailable", flush=True)
        elif not running:
            stall_warn_stage = 0
            last_completion_ts = float(now)

        if stop_event.is_set():
            drained_any = False
            while True:
                try:
                    raw = progress_queue.get_nowait()
                except queue.Empty:
                    break
                except Exception as exc:
                    break
                if isinstance(raw, dict):
                    _handle_event(raw, time.time())
                drained_any = True
            if drained_any:
                idle_cycles = 0
                continue
            if not running and idle_cycles >= 2:
                break
            if idle_cycles >= 10:
                for (ds_id, seed_val), state in list(running.items()):
                    elapsed_sec = _elapsed(float(state.get("start_ts", time.time())), time.time())
                    tier = str(state.get("tier", "?"))
                    print(
                        f"[task-unknown] dataset={ds_id} tier={tier} seed={seed_val} "
                        f"no completion event observed; elapsed={elapsed_sec:.1f}s",
                        flush=True,
                    )
                break


def _pick_safe_nestedcv_splits(
    y: np.ndarray,
    requested_splits: int,
    min_train_per_class: int,
) -> Tuple[Optional[int], str]:
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.size < 2:
        return None, "single_class"

    max_splits = int(counts.min()) if counts.size else 0
    if max_splits < 2:
        return None, "min_class_count<2"

    desired = int(max(2, requested_splits))
    desired = int(min(desired, max_splits))
    min_train_per_class = int(max(1, min_train_per_class))

    for n_splits in range(desired, 1, -1):
        ok = True
        for m in counts:
            m_int = int(m)
            if m_int < n_splits:
                ok = False
                break
            worst_test = int(math.ceil(m_int / n_splits))
            worst_train = int(m_int - worst_test)
            if worst_train < min_train_per_class:
                ok = False
                break
        if ok:
            return int(n_splits), ""

    return None, (
        f"no_safe_n_splits(requested={requested_splits}, "
        f"min_train_per_class={min_train_per_class}, "
        f"min_class_count={max_splits})"
    )


def _protocol_audit_note_nestedcv() -> str:
    return (
        "Protocol audit: repeated nested CV overlay on the strict-holdout training split. "
        "Strict holdout remains the primary benchmark row."
    )


def _run_nestedcv_audit(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    cfg: DFFSConfig,
    ds_id: str,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    # Returns a dict with nested-CV aggregate metrics + CI. Any failures are reported in the row.
    out: Dict[str, Any] = {
        "n_samples_total": int(np.asarray(X_pool).shape[0]),
        "n_features_total": int(np.asarray(X_pool).shape[1]) if np.asarray(X_pool).ndim == 2 else 0,
        "n_train": 0,
        "n_test": 0,
        "n_fs_subset": 0,
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "hybrid_score": float("nan"),
        "roc_auc": float("nan"),
        "roc_curve_type": "nestedcv_not_available",
        "roc_auc_source": "nestedcv_not_available",
        "roc_curve_points": [],
        "roc_curves_by_method": {},
        "selected_features": 0,
        "model": "",
        "fs_time_sec": float("nan"),
        "dist_time_sec": float("nan"),
        "transform_time_sec": float("nan"),
        "n_dist_features_fitted": 0,
        "n_dist_features_transformed": 0,
        "n_dist_rejected": 0,
        "n_dist_skipped_unreliable": 0,
        "n_dist_skipped_block_cv": 0,
        "n_low_gof_downweighted": 0,
        "mean_dist_stability_weight": float("nan"),
        "cdf_block_gating_time_sec": float("nan"),
        "cdf_block_gating_budget_hit": 0,
        "cdf_block_gating_blocks_evaluated": 0,
        "cdf_block_gating_blocks_applied": 0,
        "nestedcv_skipped": 0,
        "nestedcv_skip_reason": "",
        "nestedcv_failures": 0,
        "nestedcv_failure_example": "",
        "nestedcv_outer_splits_requested": int(args.nestedcv_outer_splits),
        "nestedcv_outer_repeats": int(args.nestedcv_outer_repeats),
        "nestedcv_min_train_per_class": int(args.nestedcv_min_train_per_class),
        "nestedcv_seed_stride": int(args.nestedcv_seed_stride),
        "balanced_accuracy_ci_low": float("nan"),
        "balanced_accuracy_ci_high": float("nan"),
        "macro_f1_ci_low": float("nan"),
        "macro_f1_ci_high": float("nan"),
        "hybrid_score_ci_low": float("nan"),
        "hybrid_score_ci_high": float("nan"),
    }

    y_arr = np.asarray(y_pool)
    n_splits, reason = _pick_safe_nestedcv_splits(
        y_arr,
        requested_splits=int(args.nestedcv_outer_splits),
        min_train_per_class=int(args.nestedcv_min_train_per_class),
    )
    if n_splits is None:
        out["nestedcv_skipped"] = 1
        out["nestedcv_skip_reason"] = reason
        out["nestedcv_outer_splits_used"] = 0
        out["nestedcv_n_evals"] = 0
        return out

    out["nestedcv_outer_splits_used"] = int(n_splits)

    bal_scores: List[float] = []
    f1_scores: List[float] = []
    hybrid_scores: List[float] = []
    acc_scores: List[float] = []
    roc_auc_scores: List[float] = []
    n_train_sizes: List[int] = []
    n_test_sizes: List[int] = []
    n_fs_sizes: List[int] = []
    selected_features: List[int] = []
    fs_times: List[float] = []
    dist_times: List[float] = []
    transform_times: List[float] = []
    dist_fit_counts: List[int] = []
    dist_transform_counts: List[int] = []
    dist_reject_counts: List[int] = []
    dist_skip_unreliable_counts: List[int] = []
    dist_skip_block_counts: List[int] = []
    low_gof_downweighted_counts: List[int] = []
    mean_stability_weights: List[float] = []
    cdf_block_time_secs: List[float] = []
    cdf_block_budget_hits: List[int] = []
    cdf_block_blocks_evaluated: List[int] = []
    cdf_block_blocks_applied: List[int] = []
    model_names: List[str] = []

    pipeline = DistributionFeatureSelectionPipeline(cfg)
    X_pool_arr = np.asarray(X_pool, dtype=float)

    eval_idx = 0
    for rep in range(int(max(1, args.nestedcv_outer_repeats))):
        cv_seed = int(seed + rep * int(args.nestedcv_seed_stride))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
        try:
            split_iter = list(cv.split(np.zeros((y_arr.size, 1)), y_arr))
        except Exception as exc:
            out["nestedcv_failures"] += 1
            out["nestedcv_failure_example"] = str(exc)
            continue

        for fold_idx, (tr_idx, te_idx) in enumerate(split_iter):
            fold_seed = int(seed + rep * int(args.nestedcv_seed_stride) + 17 * fold_idx)
            eval_idx += 1
            try:
                res = pipeline.run_pre_split(
                    X_train=X_pool_arr[tr_idx],
                    y_train=y_arr[tr_idx],
                    X_test=X_pool_arr[te_idx],
                    y_test=y_arr[te_idx],
                    dataset_name=f"{ds_id}__nestedcv",
                    seed=fold_seed,
                    split_indices_train=tr_idx,
                    split_indices_test=te_idx,
                )
            except Exception as exc:
                out["nestedcv_failures"] += 1
                if not out["nestedcv_failure_example"]:
                    out["nestedcv_failure_example"] = str(exc)
                continue

            bal_scores.append(float(res.balanced_accuracy))
            f1_scores.append(float(res.macro_f1))
            hybrid_scores.append(float(res.hybrid_score))
            acc_scores.append(float(res.accuracy))
            roc_auc_scores.append(float(res.roc_auc))
            n_train_sizes.append(int(res.n_train))
            n_test_sizes.append(int(res.n_test))
            n_fs_sizes.append(int(res.n_fs_subset))
            selected_features.append(int(res.selected_features_count))
            fs_times.append(float(res.fs_time_sec))
            dist_times.append(float(res.dist_time_sec))
            transform_times.append(float(res.transform_time_sec))
            dist_fit_counts.append(int(res.n_dist_features_fitted))
            dist_transform_counts.append(int(res.n_dist_features_transformed))
            dist_reject_counts.append(int(res.n_dist_rejected))
            dist_skip_unreliable_counts.append(int(res.n_dist_skipped_unreliable))
            dist_skip_block_counts.append(int(res.n_dist_skipped_block_cv))
            low_gof_downweighted_counts.append(int(res.n_low_gof_downweighted))
            mean_stability_weights.append(float(res.mean_dist_stability_weight))
            cdf_block_time_secs.append(float(res.cdf_block_gating_time_sec))
            cdf_block_budget_hits.append(int(bool(res.cdf_block_gating_budget_hit)))
            cdf_block_blocks_evaluated.append(int(res.cdf_block_gating_blocks_evaluated))
            cdf_block_blocks_applied.append(int(res.cdf_block_gating_blocks_applied))
            model_names.append(str(res.model_name))

    out["nestedcv_n_evals"] = int(len(bal_scores))
    if not bal_scores:
        out["nestedcv_skipped"] = 1
        out["nestedcv_skip_reason"] = out["nestedcv_skip_reason"] or "no_successful_folds"
        return out

    bal_mean, bal_std, bal_lo, bal_hi = _mean_std_ci(bal_scores, ci_level=float(args.nestedcv_ci_level))
    f1_mean, f1_std, f1_lo, f1_hi = _mean_std_ci(f1_scores, ci_level=float(args.nestedcv_ci_level))
    hyb_mean, hyb_std, hyb_lo, hyb_hi = _mean_std_ci(hybrid_scores, ci_level=float(args.nestedcv_ci_level))
    acc_mean, _, _, _ = _mean_std_ci(acc_scores, ci_level=float(args.nestedcv_ci_level))

    out.update(
        {
            "nestedcv_balanced_accuracy_std": float(bal_std),
            "nestedcv_macro_f1_std": float(f1_std),
            "nestedcv_hybrid_score_std": float(hyb_std),
            "balanced_accuracy": float(bal_mean),
            "macro_f1": float(f1_mean),
            "hybrid_score": float(hyb_mean),
            "accuracy": float(acc_mean),
            "roc_auc": float(np.nanmean(np.asarray(roc_auc_scores, dtype=float))) if roc_auc_scores else float("nan"),
            "roc_curve_type": "nestedcv_fold_mean_auc_only",
            "roc_auc_source": "nestedcv_fold_mean",
            "roc_curve_points": [],
            "roc_curves_by_method": {},
            "balanced_accuracy_ci_low": float(bal_lo),
            "balanced_accuracy_ci_high": float(bal_hi),
            "macro_f1_ci_low": float(f1_lo),
            "macro_f1_ci_high": float(f1_hi),
            "hybrid_score_ci_low": float(hyb_lo),
            "hybrid_score_ci_high": float(hyb_hi),
            "n_train": int(round(float(np.mean(n_train_sizes)))) if n_train_sizes else 0,
            "n_test": int(round(float(np.mean(n_test_sizes)))) if n_test_sizes else 0,
            "n_fs_subset": int(round(float(np.mean(n_fs_sizes)))) if n_fs_sizes else 0,
            "selected_features": int(round(float(np.mean(selected_features)))) if selected_features else 0,
            "fs_time_sec": float(np.mean(fs_times)) if fs_times else float("nan"),
            "dist_time_sec": float(np.mean(dist_times)) if dist_times else float("nan"),
            "transform_time_sec": float(np.mean(transform_times)) if transform_times else float("nan"),
            "n_dist_features_fitted": int(round(float(np.mean(dist_fit_counts)))) if dist_fit_counts else 0,
            "n_dist_features_transformed": int(round(float(np.mean(dist_transform_counts)))) if dist_transform_counts else 0,
            "n_dist_rejected": int(round(float(np.mean(dist_reject_counts)))) if dist_reject_counts else 0,
            "n_dist_skipped_unreliable": int(round(float(np.mean(dist_skip_unreliable_counts)))) if dist_skip_unreliable_counts else 0,
            "n_dist_skipped_block_cv": int(round(float(np.mean(dist_skip_block_counts)))) if dist_skip_block_counts else 0,
            "n_low_gof_downweighted": int(round(float(np.mean(low_gof_downweighted_counts)))) if low_gof_downweighted_counts else 0,
            "mean_dist_stability_weight": float(np.mean(mean_stability_weights)) if mean_stability_weights else float("nan"),
            "cdf_block_gating_time_sec": float(np.mean(cdf_block_time_secs)) if cdf_block_time_secs else float("nan"),
            "cdf_block_gating_budget_hit": int(bool(np.any(np.asarray(cdf_block_budget_hits, dtype=int) > 0))) if cdf_block_budget_hits else 0,
            "cdf_block_gating_blocks_evaluated": int(round(float(np.mean(cdf_block_blocks_evaluated)))) if cdf_block_blocks_evaluated else 0,
            "cdf_block_gating_blocks_applied": int(round(float(np.mean(cdf_block_blocks_applied)))) if cdf_block_blocks_applied else 0,
        }
    )

    if cdf_block_budget_hits:
        out["nestedcv_cdf_block_budget_hit_frac"] = float(np.mean(np.asarray(cdf_block_budget_hits, dtype=float)))

    # Report the most common selected model as a compact diagnostic.
    if model_names:
        values, counts = np.unique(np.asarray(model_names, dtype=str), return_counts=True)
        if values.size:
            best = int(np.argmax(counts))
            out["model"] = str(values[best])
            out["nestedcv_model_mode_frac"] = float(counts[best] / max(1, int(np.sum(counts))))

    return out


def _run_dataset_seed_task(
    ds_id: str,
    seed: int,
    args: argparse.Namespace,
    progress_queue: Optional[Any] = None,
) -> Dict[str, Any]:
    spec = BENCHMARK_DATASETS[ds_id]
    promotion_meta = _benchmark_dataset_promotion_metadata(spec)
    task_start_ts = time.time()
    failures: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    model_bundles: List[Dict[str, Any]] = []
    run_diagnostics: List[Dict[str, Any]] = []
    _emit_progress_event(
        progress_queue,
        {
            "event": "task_start",
            "dataset_id": ds_id,
            "tier": str(spec.tier),
            "seed": int(seed),
            "pid": int(os.getpid()),
            "ts": float(task_start_ts),
        },
    )

    def _emit_task_done(status: str) -> None:
        _emit_progress_event(
            progress_queue,
            {
                "event": "task_done",
                "dataset_id": ds_id,
                "tier": str(spec.tier),
                "seed": int(seed),
                "status": str(status),
                "rows_count": int(len(rows)),
                "failures_count": int(len(failures)),
                "timeout_failures": int(_count_timeout_failures(failures)),
                "elapsed_sec": float(max(0.0, time.time() - task_start_ts)),
            },
        )

    try:
        X, y, data_source, effective_tier, batch_labels, batch_label_meta = _load_dataset(
            spec,
            seed=seed,
            allow_synthetic_fallback=args.allow_synthetic_fallback,
            sample_cap=args.synthetic_sample_cap,
            feature_cap=args.synthetic_feature_cap,
            dataset_integrity_policy=str(getattr(args, "dataset_integrity_policy", "error")),
            dataset_min_classes=int(getattr(args, "dataset_min_classes", 2) or 2),
            dataset_min_class_count=int(getattr(args, "dataset_min_class_count", 1) or 1),
            source_policy=str(promotion_meta["source_policy"]),
            batch_label_policy=str(getattr(args, "batch_label_policy", "none") or "none"),
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
        _emit_task_done("skipped_dataset_integrity")
        return {
            "rows": rows,
            "failures": failures,
            "model_bundles": model_bundles,
            "run_diagnostics": run_diagnostics,
        }
    except Exception as exc:
        failures.append({"dataset_id": ds_id, "seed": seed, "error": str(exc)})
        _emit_task_done("dataset_load_error")
        return {
            "rows": rows,
            "failures": failures,
            "model_bundles": model_bundles,
            "run_diagnostics": run_diagnostics,
        }

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    base_cfg = _build_base_config(args=args, spec=spec, seed=seed)
    configs = _build_ablation_configs(base_cfg, profile=args.ablation_profile)

    # Optional: evaluate the baseline stack with per-dataset SOTA-matched
    # downstream classifier candidates to avoid classifier-mismatch confounds.
    if bool(getattr(args, "use_sota_matched_classifiers", False)):
        sota_candidates = _get_sota_classifiers_for_dataset(ds_id)
        if sota_candidates:
            sota_cfg = clone_config(base_cfg)
            _apply_exact_model_candidate_set(sota_cfg, sota_candidates)
            configs.append(("sota_matched", sota_cfg))

    maqc_names, maqc_method_sets = _resolve_maqc_pairing_method_sets_from_args(args)
    enable_maqc = bool(getattr(args, "enable_maqc_pairing", False))
    maqc_all_configs = bool(getattr(args, "maqc_pairing_all_configs", False))

    for config_name, cfg in configs:
        cfg.multiomics_feature_blocks = dict(
            batch_label_meta.get("multiomics_feature_blocks", {}) or {}
        ) or None
        if enable_maqc and (config_name == "baseline" or maqc_all_configs):
            cfg.enable_maqc_pairing = True
            cfg.maqc_pairing_method_sets = maqc_method_sets
            cfg.maqc_pairing_method_set_names = maqc_names
        else:
            cfg.enable_maqc_pairing = False
            cfg.maqc_pairing_method_sets = tuple()
            cfg.maqc_pairing_method_set_names = tuple()
        try:
            result = _run_pipeline_with_hard_timeout(
                cfg=cfg,
                X=X,
                y=y,
                batch_labels=batch_labels,
                dataset_name=ds_id,
                seed=seed,
                timeout_sec=float(args.task_timeout_sec),
                quiet_worker_logs=bool(args.quiet_worker_logs),
                use_hard_kill=bool(getattr(args, "max_workers", 1) > 1),
                capture_artifacts=True,
                capture_diagnostics=True,
            )
        except Exception as exc:
            err_txt = str(exc)
            failures.append(
                {
                    "dataset_id": ds_id,
                    "seed": seed,
                    "config": config_name,
                    "error": err_txt,
                }
            )
            _emit_progress_event(
                progress_queue,
                {
                    "event": "config_timeout" if _is_timeout_error_message(err_txt) else "config_error",
                    "dataset_id": ds_id,
                    "seed": int(seed),
                    "config": str(config_name),
                    "error": err_txt,
                },
            )
            continue

        sanity_ok = (
            result.selected_features_count > 0
            and np.isfinite(result.balanced_accuracy)
            and np.isfinite(result.macro_f1)
        )

        effective_enabled_methods = list(result.config_snapshot.get("effective_enabled_methods", []) or [])
        if not effective_enabled_methods:
            effective_enabled_methods = list(result.config_snapshot.get("enabled_methods", []) or [])
        conformal_method = str(
            result.config_snapshot.get("classifier_conformal_method", "split") or "split"
        ).strip().lower()
        mapie_requested = bool(result.config_snapshot.get("classifier_conformal_enabled", False)) and (
            conformal_method in {"aps", "raps", "cross"}
        )
        mapie_applied = bool(result.config_snapshot.get("classifier_conformal_mapie_applied", False))

        row = {
            "dataset_id": ds_id,
            "dataset_name": spec.display_name,
            "tier": spec.tier,
            "effective_tier": effective_tier,
            "domain": str(getattr(spec, "domain", "genomics") or "genomics"),
            "platform": str(getattr(spec, "platform", "cDNA") or "cDNA"),
            "seed": seed,
            "config": config_name,
            "protocol": "holdout",
            "data_source": data_source,
            "batch_label_policy": str(batch_label_meta.get("batch_label_policy", "none")),
            "batch_label_policy_reason": str(
                batch_label_meta.get("batch_label_policy_reason", "unknown")
            ),
            "batch_labels_available": int(bool(batch_label_meta.get("batch_labels_available", False))),
            "batch_labels_n_unique": int(batch_label_meta.get("batch_labels_n_unique", 0) or 0),
            "multiomics_feature_blocks_available": int(
                bool(batch_label_meta.get("multiomics_feature_blocks_available", False))
            ),
            "multiomics_feature_blocks_source_reason": str(
                batch_label_meta.get("multiomics_feature_blocks_source_reason", "unknown")
            ),
            "multiomics_adapter_mode": str(
                result.config_snapshot.get(
                    "multiomics_adapter_mode",
                    result.config_snapshot.get("multiomics_adapter", "none"),
                )
                or "none"
            ),
            "multiomics_integrator": str(
                result.config_snapshot.get("multiomics_integrator", "mb_plsda") or "mb_plsda"
            ),
            "multiomics_adapter_applied": int(
                bool(result.config_snapshot.get("multiomics_adapter_applied", False))
            ),
            "multiomics_adapter_reason": str(
                result.config_snapshot.get("multiomics_adapter_reason", "disabled") or "disabled"
            ),
            "multiomics_n_blocks": int(
                result.config_snapshot.get("multiomics_n_blocks", 0) or 0
            ),
            "multiomics_latent_dim": int(
                result.config_snapshot.get("multiomics_latent_dim", 0) or 0
            ),
            "validation_pipeline": str(spec.validation_pipeline or ""),
            "validation_scenario": str(spec.validation_scenario or ""),
            "promotion_eligible": int(promotion_meta["promotion_eligible"]),
            "promotion_blocker": str(promotion_meta["promotion_blocker"]),
            "source_policy": str(promotion_meta["source_policy"]),
            "n_samples_total": result.n_samples_total,
            "n_features_total": result.n_features_total,
            "n_train": result.n_train,
            "n_test": result.n_test,
            "max_train_samples": result.config_snapshot.get("max_train_samples"),
            "n_fs_subset": result.n_fs_subset,
            "accuracy": result.accuracy,
            "balanced_accuracy": result.balanced_accuracy,
            "macro_f1": result.macro_f1,
            "hybrid_score": result.hybrid_score,
            "roc_auc": result.roc_auc,
            "roc_curve_type": result.roc_curve_type,
            "roc_auc_source": result.roc_auc_source,
            "roc_curve_points": [[float(p[0]), float(p[1])] for p in result.roc_curve_points],
            "roc_curves_by_method": dict(result.roc_curves_by_method or {}),
            "selected_features": result.selected_features_count,
            "selected_feature_indices_original": [
                int(i) for i in tuple(result.selected_feature_indices_original or tuple())
            ],
            "model": result.model_name,
            "fs_time_sec": result.fs_time_sec,
            "dist_time_sec": result.dist_time_sec,
            "transform_time_sec": result.transform_time_sec,
            "n_dist_features_fitted": result.n_dist_features_fitted,
            "n_dist_features_transformed": result.n_dist_features_transformed,
            "n_dist_rejected": result.n_dist_rejected,
            "n_dist_skipped_unreliable": result.n_dist_skipped_unreliable,
            "n_dist_skipped_block_cv": result.n_dist_skipped_block_cv,
            "n_low_gof_downweighted": result.n_low_gof_downweighted,
            "mean_dist_stability_weight": result.mean_dist_stability_weight,
            "cdf_block_gating_time_sec": result.cdf_block_gating_time_sec,
            "cdf_block_gating_budget_hit": int(bool(result.cdf_block_gating_budget_hit)),
            "cdf_block_gating_blocks_evaluated": result.cdf_block_gating_blocks_evaluated,
            "cdf_block_gating_blocks_applied": result.cdf_block_gating_blocks_applied,
            "sota_holdout_bal_acc_low": spec.sota_holdout_bal_acc[0],
            "sota_holdout_bal_acc_high": spec.sota_holdout_bal_acc[1],
            "sota_inflated_bal_acc_low": spec.sota_inflated_bal_acc[0],
            "sota_inflated_bal_acc_high": spec.sota_inflated_bal_acc[1],
            "sota_source_confidence": str(getattr(spec, "sota_source_confidence", "proxy") or "proxy"),
            "sota_claim_scope": str(getattr(spec, "sota_claim_scope", "positioning_only") or "positioning_only"),
            "sota_holdout_status": _compare_to_sota(result.balanced_accuracy, *spec.sota_holdout_bal_acc),
            "sota_inflated_status": _compare_to_sota(result.balanced_accuracy, *spec.sota_inflated_bal_acc),
            # Backward-compatible aliases: keep existing downstream tooling working.
            "sota_bal_acc_low": spec.sota_holdout_bal_acc[0],
            "sota_bal_acc_high": spec.sota_holdout_bal_acc[1],
            "sota_status": _compare_to_sota(result.balanced_accuracy, *spec.sota_holdout_bal_acc),
            "protocol_gap_note": _protocol_gap_note(spec),
            "sanity_ok": int(bool(sanity_ok)),
            "enabled_methods_source": str(result.config_snapshot.get("enabled_methods_source", "")),
            "maqc_pairing_enabled": int(bool(result.config_snapshot.get("maqc_pairing_enabled", False))),
            "maqc_pairing_selected_fs_name": str(result.config_snapshot.get("maqc_pairing_selected_fs_name", "")),
            "maqc_pairing_selected_cv_score": (
                float(result.config_snapshot.get("maqc_pairing_selected_cv_score"))
                if isinstance(result.config_snapshot.get("maqc_pairing_selected_cv_score", None), (int, float, np.floating))
                else float("nan")
            ),
            "maqc_pairing_candidate_count": int(result.config_snapshot.get("maqc_pairing_candidate_count", 0) or 0),
            "maqc_pairing_evaluated_count": int(result.config_snapshot.get("maqc_pairing_evaluated_count", 0) or 0),
            "maqc_pairing_failed_count": int(result.config_snapshot.get("maqc_pairing_failed_count", 0) or 0),
            "classifier_conformal_enabled": int(
                bool(result.config_snapshot.get("classifier_conformal_enabled", False))
            ),
            "classifier_conformal_applied": int(
                bool(result.config_snapshot.get("classifier_conformal_applied", False))
            ),
            "df_stage_position": str(
                result.config_snapshot.get("df_stage_position_effective", result.config_snapshot.get("df_stage_position", "before_fs"))
            ),
            "df_stage_source_space": str(
                result.config_snapshot.get("df_stage_source_space", "model_input")
            ),
            "classifier_conformal_skip_reason": str(
                result.config_snapshot.get("classifier_conformal_skip_reason", "")
            ),
            "classifier_conformal_coverage": (
                float(result.config_snapshot.get("classifier_conformal_coverage"))
                if isinstance(result.config_snapshot.get("classifier_conformal_coverage", None), (int, float, np.floating))
                else float("nan")
            ),
            "classifier_conformal_set_size_mean": (
                float(result.config_snapshot.get("classifier_conformal_set_size_mean"))
                if isinstance(result.config_snapshot.get("classifier_conformal_set_size_mean", None), (int, float, np.floating))
                else float("nan")
            ),
            "classifier_conformal_singleton_rate": (
                float(result.config_snapshot.get("classifier_conformal_singleton_rate"))
                if isinstance(result.config_snapshot.get("classifier_conformal_singleton_rate", None), (int, float, np.floating))
                else float("nan")
            ),
            "classifier_conformal_prediction_sets": list(
                result.config_snapshot.get("classifier_conformal_prediction_sets", []) or []
            ),
            "classifier_conformal_method": str(conformal_method),
            "classifier_conformal_mapie_applied": int(
                bool(mapie_applied)
            ),
            "classifier_conformal_mapie_enabled": int(
                bool(result.config_snapshot.get("classifier_conformal_mapie_enabled", False))
            ),
            "classifier_conformal_mapie_method": str(
                result.config_snapshot.get("classifier_conformal_mapie_method", "")
            ),
            "classifier_conformal_mapie_skip_reason": str(
                result.config_snapshot.get("classifier_conformal_mapie_skip_reason", "")
            ),
            "regime_policy_applied": int(
                bool(result.config_snapshot.get("regime_policy_applied", False))
            ),
            "regime_policy_mode": str(
                result.config_snapshot.get("regime_policy_mode", "")
            ),
            "regime_policy_bypass_mode": str(
                result.config_snapshot.get("regime_policy_bypass_mode", "")
            ),
            "regime_policy_trigger_low_p_over_n": int(
                bool(result.config_snapshot.get("regime_policy_trigger_low_p_over_n", False))
            ),
            "regime_policy_trigger_very_hard": int(
                bool(result.config_snapshot.get("regime_policy_trigger_very_hard", False))
            ),
            "regime_policy_p_over_n": (
                float(result.config_snapshot.get("regime_policy_p_over_n"))
                if isinstance(result.config_snapshot.get("regime_policy_p_over_n"), (int, float, np.floating))
                else float("nan")
            ),
            "fs_cap_applied": int(bool(result.config_snapshot.get("fs_cap_applied", False))),
            "fs_cap_max_allowed": int(
                (result.config_snapshot.get("fs_cap_meta", {}) or {}).get("max_allowed", 0) or 0
            ),
            "fs_ipss_use_eats_threshold": int(
                bool(result.config_snapshot.get("fs_ipss_use_eats_threshold", False))
            ),
            "selection_strategy": str(
                result.config_snapshot.get("selection_strategy", "mnpo_portfolio")
                or "mnpo_portfolio"
            ),
            "fs_stability_threshold_method": str(
                result.config_snapshot.get("fs_stability_threshold_method", "fixed")
            ),
            "fs_stability_target_pfer": float(
                result.config_snapshot.get("fs_stability_target_pfer", 1.0) or 1.0
            ),
            "effective_enabled_methods": list(effective_enabled_methods),
        }
        row["telemetry_effective_methods_populated"] = int(len(effective_enabled_methods) > 0)
        row["telemetry_effective_methods_missing"] = int(len(effective_enabled_methods) == 0)
        row["telemetry_mapie_requested"] = int(bool(mapie_requested))
        row["telemetry_mapie_activation_mismatch"] = int(bool(mapie_requested and not mapie_applied))
        _batch_apply_meta = {}
        if isinstance(result.run_diagnostics, dict):
            _batch_apply_meta = (
                ((result.run_diagnostics.get("pipeline_stages") or {}).get("batch_correction_apply") or {})
                if isinstance(result.run_diagnostics.get("pipeline_stages"), dict)
                else {}
            )
        row["batch_correction_mode_applied"] = str(
            _batch_apply_meta.get("batch_correction_mode_applied", "")
        )
        row["batch_correction_apply_reason"] = str(
            _batch_apply_meta.get("batch_correction_apply_reason", "")
        )
        rows.append(row)
        run_key = f"{ds_id}__seed{int(seed)}__config_{str(config_name)}__protocol_holdout"
        if isinstance(result.model_bundle, dict) and result.model_bundle:
            bundle = dict(result.model_bundle)
            bundle["dataset_id"] = str(ds_id)
            bundle["dataset_name"] = str(spec.display_name)
            bundle["seed"] = int(seed)
            bundle["config"] = str(config_name)
            bundle["protocol"] = "holdout"
            bundle["run_key"] = str(run_key)
            model_bundles.append(bundle)
        if isinstance(result.run_diagnostics, dict) and result.run_diagnostics:
            diag = dict(result.run_diagnostics)
            diag["dataset_id"] = str(ds_id)
            diag["dataset_name"] = str(spec.display_name)
            diag["seed"] = int(seed)
            diag["config"] = str(config_name)
            diag["protocol"] = "holdout"
            diag["run_key"] = str(run_key)
            run_diagnostics.append(diag)

        # Optional protocol audit overlay: repeated nested CV on the strict-holdout training split.
        if (
            bool(getattr(args, "enable_nestedcv_audit", False))
            and (bool(getattr(args, "nestedcv_audit_all_configs", False)) or config_name == "baseline")
            and int(result.n_train) < int(getattr(args, "nestedcv_min_n_train", 0))
        ):
            train_idx = np.asarray(result.split_indices_train, dtype=int)
            if train_idx.size:
                X_pool = np.asarray(X, dtype=float)[train_idx]
                y_pool = np.asarray(y)[train_idx]
            else:
                X_pool = np.asarray(X, dtype=float)
                y_pool = np.asarray(y)

            nested_row = dict(row)
            nested_row["protocol"] = "nestedcv"
            nested_row["dataset_name"] = f"{spec.display_name} (NestedCV Audit)"
            nested_row["protocol_gap_note"] = _protocol_audit_note_nestedcv()
            nested_row["sota_holdout_status"] = "audit"
            nested_row["sota_inflated_status"] = "audit"
            nested_row["sota_status"] = "audit"

            nested_metrics = _run_nestedcv_audit(
                X_pool=X_pool,
                y_pool=y_pool,
                cfg=cfg,
                ds_id=ds_id,
                seed=seed,
                args=args,
            )
            nested_row.update(nested_metrics)

            sanity_nested = (
                int(nested_row.get("nestedcv_skipped", 0)) == 0
                and np.isfinite(float(nested_row.get("balanced_accuracy", float("nan"))))
                and np.isfinite(float(nested_row.get("macro_f1", float("nan"))))
            )
            nested_row["sanity_ok"] = int(bool(sanity_nested))
            rows.append(nested_row)
    _emit_task_done("ok" if rows else "no_successful_rows")
    return {
        "rows": rows,
        "failures": failures,
        "model_bundles": model_bundles,
        "run_diagnostics": run_diagnostics,
    }


def _build_run_summary(
    rows: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    run_dir: Any,
) -> Dict[str, Any]:
    """Build a compact, deterministic per-run JSON summary.

    Contains FS method preset, portfolio candidates, oracle weights,
    DF selected families, key diagnostics and runtime breakdowns.
    Includes ``schema_version`` per ArchitectureRefactor.md §14.3 OBS-1.
    """
    import datetime as _dt

    # Aggregate DF family distribution across all rows
    df_family_counts: Dict[str, int] = {}
    for row in rows:
        # distribution_summaries aren't in flat rows; extract from
        # config snapshot if present.
        pass

    # Collect per-config/dataset/seed results
    per_run_results: List[Dict[str, Any]] = []
    for row in rows:
        entry: Dict[str, Any] = {
            "dataset_id": row.get("dataset_id", ""),
            "seed": row.get("seed", 0),
            "config": row.get("config", ""),
            "protocol": row.get("protocol", "holdout"),
            "domain": row.get("domain", ""),
            "platform": row.get("platform", ""),
            "balanced_accuracy": row.get("balanced_accuracy", float("nan")),
            "macro_f1": row.get("macro_f1", float("nan")),
            "hybrid_score": row.get("hybrid_score", float("nan")),
            "roc_auc": row.get("roc_auc", float("nan")),
            "roc_curve_type": row.get("roc_curve_type", "unavailable"),
            "roc_auc_source": row.get("roc_auc_source", "unavailable"),
            "roc_curve_points": row.get("roc_curve_points", []),
            "roc_curves_by_method": row.get("roc_curves_by_method", {}),
            "classifier_conformal_applied": int(row.get("classifier_conformal_applied", 0) or 0),
            "classifier_conformal_coverage": row.get("classifier_conformal_coverage", float("nan")),
            "classifier_conformal_set_size_mean": row.get("classifier_conformal_set_size_mean", float("nan")),
            "classifier_conformal_singleton_rate": row.get("classifier_conformal_singleton_rate", float("nan")),
            "classifier_conformal_skip_reason": row.get("classifier_conformal_skip_reason", ""),
            "selected_features": row.get("selected_features", 0),
            "model": row.get("model", ""),
            "tier": row.get("tier", ""),
        }
        # Runtime breakdowns
        entry["runtime"] = {
            "fs_time_sec": row.get("fs_time_sec", 0.0),
            "dist_time_sec": row.get("dist_time_sec", 0.0),
            "transform_time_sec": row.get("transform_time_sec", 0.0),
        }
        # DF diagnostics
        entry["df_diagnostics"] = {
            "n_dist_features_fitted": row.get("n_dist_features_fitted", 0),
            "n_dist_features_transformed": row.get("n_dist_features_transformed", 0),
            "n_dist_rejected": row.get("n_dist_rejected", 0),
            "n_dist_skipped_unreliable": row.get("n_dist_skipped_unreliable", 0),
            "n_dist_skipped_block_cv": row.get("n_dist_skipped_block_cv", 0),
            "n_low_gof_downweighted": row.get("n_low_gof_downweighted", 0),
        }
        # FS method source
        entry["enabled_methods_source"] = row.get("enabled_methods_source", "")
        entry["oracle_diagnostics"] = {
            "oracle_stability_mean_rank_correlation": row.get("oracle_stability_mean_rank_correlation", None),
            "evaluation_failures_total": row.get("evaluation_failures_total", 0),
            "copula_low_information_reason": row.get("copula_low_information_reason", ""),
        }
        per_run_results.append(entry)

    # Config snapshot from metadata
    config_flags = metadata.get("config_flags", {})

    # Anti-gaming telemetry (measurement-only, no promotion logic changes).
    hard_rows = [r for r in rows if str(r.get("tier", "")) == "hard"]
    hard_by_dataset: Dict[str, float] = {}
    for r in hard_rows:
        ds = str(r.get("dataset_id", ""))
        if not ds:
            continue
        hard_by_dataset[ds] = hard_by_dataset.get(ds, 0.0) + float(r.get("balanced_accuracy", 0.0))
    hard_vals = np.asarray(list(hard_by_dataset.values()), dtype=float)
    t1_top_share = 0.0
    if hard_vals.size > 0 and np.sum(np.abs(hard_vals)) > 1e-12:
        k = max(1, int(np.ceil(0.2 * hard_vals.size)))
        top = np.sort(np.abs(hard_vals))[::-1][:k]
        t1_top_share = float(np.sum(top) / np.sum(np.abs(hard_vals)))

    rerun_counts = metadata.get("candidate_rerun_counts", {})
    t3_asymmetry_ratio = 0.0
    if isinstance(rerun_counts, dict) and rerun_counts:
        vals = np.asarray([float(v) for v in rerun_counts.values()], dtype=float)
        med = float(np.median(vals)) if vals.size else 0.0
        mx = float(np.max(vals)) if vals.size else 0.0
        t3_asymmetry_ratio = float(mx / med) if med > 1e-12 else 0.0

    shadow_meta = metadata.get("shadow_evaluator", {})
    if not isinstance(shadow_meta, dict):
        shadow_meta = {}

    # T-R-274: seed variance guard — flag datasets with BA std > 0.10 across seeds.
    seed_variance_guard: Dict[str, Any] = {}
    _ds_ba_map: Dict[str, List[float]] = {}
    for r in rows:
        ds = str(r.get("dataset_id", ""))
        ba = r.get("balanced_accuracy")
        if ds and ba is not None:
            _ds_ba_map.setdefault(ds, []).append(float(ba))
    flagged_datasets = []
    for ds, ba_vals in _ds_ba_map.items():
        if len(ba_vals) >= 2:
            std_val = float(np.std(ba_vals, ddof=1))
            if std_val > 0.10:
                flagged_datasets.append({
                    "dataset_id": ds,
                    "ba_std": round(std_val, 5),
                    "n_seeds": len(ba_vals),
                    "ba_values": [round(v, 5) for v in ba_vals],
                })
    seed_variance_guard = {
        "seed_variance_flag": len(flagged_datasets) > 0,
        "threshold": 0.10,
        "n_flagged": len(flagged_datasets),
        "flagged_datasets": flagged_datasets,
    }

    telemetry_missing_methods = int(
        np.sum([int(r.get("telemetry_effective_methods_missing", 0) or 0) for r in rows])
    )
    telemetry_mapie_requested = int(
        np.sum([int(r.get("telemetry_mapie_requested", 0) or 0) for r in rows])
    )
    telemetry_mapie_mismatch = int(
        np.sum([int(r.get("telemetry_mapie_activation_mismatch", 0) or 0) for r in rows])
    )

    summary: Dict[str, Any] = {
        "schema_version": "1.0",
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "fs_method_preset": metadata.get("fs_method_set", "unknown"),
        "datasets": metadata.get("datasets", []),
        "seeds": metadata.get("seeds", []),
        "compute_budget": metadata.get("compute_budget", "standard"),
        "dist_criterion": metadata.get("dist_criterion", "simple"),
        "n_total_runs": len(rows),
        "n_failures": len(failures),
        "results": per_run_results,
        "config_snapshot": config_flags,
        "anti_gaming_telemetry": {
            "t1_selection_concentration_share": float(t1_top_share),
            "t2_negative_drift_count_easy_medium": int(
                np.sum([
                    1
                    for r in rows
                    if str(r.get("tier", "")) in {"easy", "medium"}
                    and float(r.get("balanced_accuracy", 0.0)) < 0.0
                ])
            ),
            "t3_rerun_asymmetry_ratio": float(t3_asymmetry_ratio),
        },
        "shadow_evaluator": {
            "enabled": bool(shadow_meta.get("enabled", False)),
            "frozen_subset_id": str(shadow_meta.get("frozen_subset_id", "")),
            "concordance_rate": float(shadow_meta.get("concordance_rate", 0.0)),
            "n_compared": int(shadow_meta.get("n_compared", 0)),
            "disagreement_count": int(shadow_meta.get("disagreement_count", 0)),
        },
        # T-R-274: seed variance guard.
        "seed_variance_guard": seed_variance_guard,
        "telemetry_hardening": {
            "effective_methods_missing_rows": telemetry_missing_methods,
            "effective_methods_missing_rate": (
                float(telemetry_missing_methods / max(1, len(rows)))
            ),
            "mapie_requested_rows": telemetry_mapie_requested,
            "mapie_activation_mismatch_rows": telemetry_mapie_mismatch,
            "mapie_activation_mismatch_rate": (
                float(telemetry_mapie_mismatch / max(1, telemetry_mapie_requested))
                if telemetry_mapie_requested > 0
                else float("nan")
            ),
        },
    }
    return summary


def run_benchmark(args: argparse.Namespace) -> Any:
    dataset_sets = list(args.dataset_sets)
    if getattr(args, "extended", False) and "extended" not in dataset_sets:
        dataset_sets.append("extended")
    selected = _resolve_dataset_list(dataset_sets, args.datasets, args.exclude_datasets)
    selected_validation_specs = [
        BENCHMARK_DATASETS[ds_id]
        for ds_id in selected
        if str(getattr(BENCHMARK_DATASETS[ds_id], "source_kind", "")) == "validation_catalog"
    ]
    if selected_validation_specs and bool(args.allow_synthetic_fallback):
        raise ValueError(
            "Synthetic fallback is forbidden for benchmark validation datasets. "
            "Use the HuggingFace bundle via TABNETICS_HF_ORG or TABNETICS_HF_REPO_ID."
        )
    for spec in selected_validation_specs:
        validation_dataset_id = str(spec.validation_dataset_id or "").strip()
        if not validation_dataset_id:
            raise ValueError(f"Missing validation dataset id for benchmark dataset {spec.dataset_id}")
        validation_spec = CATALOG[validation_dataset_id]
        _require_hf_bundle_configuration(
            dataset_id=validation_dataset_id,
            loader_kind=str(validation_spec.loader_kind),
        )
    run_dir = create_timestamped_run_dir(args.output_dir, "df_fs_sota_benchmark")

    all_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    all_model_bundles: List[Dict[str, Any]] = []
    all_run_diagnostics: List[Dict[str, Any]] = []
    tasks: List[Tuple[str, int]] = [(ds_id, seed) for ds_id in selected for seed in args.seeds]
    heartbeat_sec = float(getattr(args, "progress_heartbeat_sec", 60.0) or 60.0)
    if not np.isfinite(heartbeat_sec) or heartbeat_sec <= 0.0:
        heartbeat_sec = 60.0
    watchdog_sec = float(getattr(args, "progress_watchdog_sec", 900.0) or 0.0)
    if not np.isfinite(watchdog_sec) or watchdog_sec < 0.0:
        watchdog_sec = 0.0
    stall_watchdog_sec = float(getattr(args, "progress_stall_watchdog_sec", 1800.0) or 0.0)
    if not np.isfinite(stall_watchdog_sec) or stall_watchdog_sec < 0.0:
        stall_watchdog_sec = 0.0

    print(
        f"[run] tasks={len(tasks)} datasets={len(selected)} seeds={len(args.seeds)} "
        f"workers={args.max_workers} heartbeat_sec={heartbeat_sec:.1f} "
        f"watchdog_sec={watchdog_sec:.1f} stall_watchdog_sec={stall_watchdog_sec:.1f}",
        flush=True,
    )
    if args.max_workers <= 1:
        for idx, (ds_id, seed) in enumerate(tasks, start=1):
            spec = BENCHMARK_DATASETS[ds_id]
            started = time.time()
            print(
                f"[task-start] index={idx}/{len(tasks)} dataset={ds_id} tier={spec.tier} seed={seed}",
                flush=True,
            )
            task_fn = _run_dataset_seed_task
            try:
                params = inspect.signature(task_fn).parameters
            except Exception as exc:
                params = {}
            if "progress_queue" in params:
                task_result = task_fn(ds_id, seed, args, progress_queue=None)
            else:
                task_result = task_fn(ds_id, seed, args)
            all_rows.extend(task_result["rows"])
            failures.extend(task_result["failures"])
            all_model_bundles.extend(task_result.get("model_bundles", []))
            all_run_diagnostics.extend(task_result.get("run_diagnostics", []))
            timeout_failures = _count_timeout_failures(task_result.get("failures", []))
            print(
                f"[task-done] index={idx}/{len(tasks)} dataset={ds_id} tier={spec.tier} seed={seed} "
                f"elapsed={max(0.0, time.time() - started):.1f}s rows={len(task_result.get('rows', []))} "
                f"failures={len(task_result.get('failures', []))} timeout_failures={timeout_failures}",
                flush=True,
            )
    else:
        manager = mp.Manager()
        progress_queue = manager.Queue(maxsize=max(128, int(len(tasks) * 4)))
        monitor_stop = threading.Event()
        monitor_thread = threading.Thread(
            target=_monitor_parallel_progress,
            args=(progress_queue, monitor_stop, len(tasks), heartbeat_sec, watchdog_sec, stall_watchdog_sec),
            daemon=True,
        )
        monitor_thread.start()
        task_results: List[Dict[str, Any]] = []
        parallel_error: Optional[Exception] = None
        try:
            jobs = [_build_dataset_task_job(ds_id, seed, args, progress_queue=progress_queue) for ds_id, seed in tasks]
            task_results = Parallel(n_jobs=args.max_workers, prefer="processes", verbose=10)(jobs)
        except Exception as exc:
            parallel_error = exc
        finally:
            monitor_stop.set()
            monitor_thread.join(timeout=max(6.0, heartbeat_sec + 1.0))
            try:
                manager.shutdown()
            except Exception as exc:
                pass
        if parallel_error is not None:
            raise parallel_error
        for task_result in task_results:
            all_rows.extend(task_result["rows"])
            failures.extend(task_result["failures"])
            all_model_bundles.extend(task_result.get("model_bundles", []))
            all_run_diagnostics.extend(task_result.get("run_diagnostics", []))

    if not all_rows:
        raise RuntimeError("No successful runs were produced.")

    runs_df = pd.DataFrame(all_rows)
    if "protocol" not in runs_df.columns:
        runs_df["protocol"] = "holdout"
    if "domain" not in runs_df.columns:
        runs_df["domain"] = runs_df["dataset_id"].map(
            lambda ds: str(getattr(BENCHMARK_DATASETS.get(str(ds), None), "domain", "unknown") or "unknown")
        )
    if "platform" not in runs_df.columns:
        runs_df["platform"] = runs_df["dataset_id"].map(
            lambda ds: str(getattr(BENCHMARK_DATASETS.get(str(ds), None), "platform", "unknown") or "unknown")
        )

    holdout_df = runs_df[runs_df["protocol"] == "holdout"].copy()

    summary_df = (
        runs_df.groupby(
            ["dataset_id", "dataset_name", "tier", "domain", "platform", "config", "protocol"]
        )[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "hybrid_score",
                "selected_features",
                "fs_time_sec",
                "dist_time_sec",
                "transform_time_sec",
                "n_dist_features_transformed",
                "n_dist_rejected",
                "n_dist_skipped_unreliable",
                "n_dist_skipped_block_cv",
                "n_low_gof_downweighted",
                "cdf_block_gating_time_sec",
                "cdf_block_gating_budget_hit",
                "cdf_block_gating_blocks_evaluated",
                "cdf_block_gating_blocks_applied",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in summary_df.columns
    ]

    sota_summary_rows: List[Dict[str, Any]] = []
    for ds_id, ds_rows in holdout_df.groupby("dataset_id"):
        spec = BENCHMARK_DATASETS[ds_id]
        baseline_rows = ds_rows[ds_rows["config"] == "baseline"]
        if baseline_rows.empty:
            baseline_rows = ds_rows
        bal = float(baseline_rows["balanced_accuracy"].mean())
        holdout_status = _compare_to_sota(bal, *spec.sota_holdout_bal_acc)
        inflated_status = _compare_to_sota(bal, *spec.sota_inflated_bal_acc)
        sota_summary_rows.append(
            {
                "dataset_id": ds_id,
                "dataset_name": spec.display_name,
                "tier": spec.tier,
                "domain": str(getattr(spec, "domain", "genomics") or "genomics"),
                "platform": str(getattr(spec, "platform", "cDNA") or "cDNA"),
                "sota_holdout_bal_acc_low": spec.sota_holdout_bal_acc[0],
                "sota_holdout_bal_acc_high": spec.sota_holdout_bal_acc[1],
                "sota_inflated_bal_acc_low": spec.sota_inflated_bal_acc[0],
                "sota_inflated_bal_acc_high": spec.sota_inflated_bal_acc[1],
                "sota_source_confidence": str(getattr(spec, "sota_source_confidence", "proxy") or "proxy"),
                "sota_claim_scope": str(getattr(spec, "sota_claim_scope", "positioning_only") or "positioning_only"),
                # Backward-compatible aliases.
                "sota_bal_acc_low": spec.sota_holdout_bal_acc[0],
                "sota_bal_acc_high": spec.sota_holdout_bal_acc[1],
                "observed_bal_acc_mean": bal,
                "holdout_status": holdout_status,
                "inflated_status": inflated_status,
                "status": holdout_status,
                "protocol_gap_note": _protocol_gap_note(spec),
            }
        )
    sota_df = pd.DataFrame(sota_summary_rows)

    # Delta vs baseline per dataset.
    holdout_summary_df = summary_df[summary_df["protocol"] == "holdout"].copy()
    baseline = holdout_summary_df[holdout_summary_df["config"] == "baseline"][
        ["dataset_id", "balanced_accuracy_mean", "macro_f1_mean", "hybrid_score_mean"]
    ].rename(
        columns={
            "balanced_accuracy_mean": "baseline_balanced_accuracy",
            "macro_f1_mean": "baseline_macro_f1",
            "hybrid_score_mean": "baseline_hybrid",
        }
    )
    ablation_df = holdout_summary_df.merge(baseline, on="dataset_id", how="left")
    ablation_df["delta_balanced_accuracy"] = ablation_df["balanced_accuracy_mean"] - ablation_df["baseline_balanced_accuracy"]
    ablation_df["delta_macro_f1"] = ablation_df["macro_f1_mean"] - ablation_df["baseline_macro_f1"]
    ablation_df["delta_hybrid"] = ablation_df["hybrid_score_mean"] - ablation_df["baseline_hybrid"]

    runs_path = run_dir / "df_fs_runs.csv"
    summary_path = run_dir / "df_fs_summary.csv"
    sota_path = run_dir / "df_fs_sota_comparison.csv"
    ablation_path = run_dir / "df_fs_ablation_deltas.csv"
    failures_path = run_dir / "df_fs_failures.json"
    metadata_path = run_dir / "df_fs_metadata.json"
    model_bundles_path = run_dir / "df_fs_model_bundles.json"
    run_diagnostics_path = run_dir / "df_fs_run_diagnostics.json"

    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    sota_df.to_csv(sota_path, index=False)
    ablation_df.to_csv(ablation_path, index=False)

    shadow_path = run_dir / "shadow_evaluator_pilot.csv"
    shadow_meta: Dict[str, Any] = {
        "enabled": False,
        "frozen_subset_id": "",
        "n_compared": 0,
        "concordance_rate": 0.0,
        "disagreement_count": 0,
    }
    if bool(getattr(args, "enable_shadow_evaluator", False)):
        frozen_subset = tuple(getattr(args, "shadow_frozen_datasets", []) or ())
        shadow_df, shadow_meta = _build_shadow_evaluator_pilot(
            all_rows,
            frozen_dataset_ids=frozen_subset,
        )
        if not shadow_df.empty:
            shadow_df.to_csv(shadow_path, index=False)

    with model_bundles_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "1.0",
                "artifact_type": "df_fs_model_bundle_collection",
                "n_items": int(len(all_model_bundles)),
                "items": all_model_bundles,
            },
            f,
            indent=2,
        )
    with run_diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "1.0",
                "artifact_type": "df_fs_run_diagnostics_collection",
                "n_items": int(len(all_run_diagnostics)),
                "items": all_run_diagnostics,
            },
            f,
            indent=2,
        )

    metadata = {
        "datasets": selected,
        "seeds": args.seeds,
        "ablation_profile": args.ablation_profile,
        "fs_method_set": args.fs_method_set,
        "max_workers": args.max_workers,
        "dataset_set_sizes": {k: len(v) for k, v in DATASET_SETS.items()},
        "compute_budget": args.compute_budget,
        "dist_criterion": args.dist_criterion,
        "config_flags": {
            "allow_synthetic_fallback": args.allow_synthetic_fallback,
            "dataset_integrity_policy": str(getattr(args, "dataset_integrity_policy", "error")),
            "dataset_min_classes": int(getattr(args, "dataset_min_classes", 2) or 2),
            "dataset_min_class_count": int(getattr(args, "dataset_min_class_count", 1) or 1),
            "synthetic_sample_cap": args.synthetic_sample_cap,
            "synthetic_feature_cap": args.synthetic_feature_cap,
            "progress_heartbeat_sec": float(getattr(args, "progress_heartbeat_sec", 60.0) or 60.0),
            "progress_watchdog_sec": float(getattr(args, "progress_watchdog_sec", 900.0) or 0.0),
            "test_size": args.test_size,
            "max_train_samples": int(getattr(args, "max_train_samples", 0) or 0),
            "disable_df_robust": args.disable_df_robust,
            "disable_df_lrt": args.disable_df_lrt,
            "disable_support_filter": args.disable_support_filter,
            "df_stage_position": str(getattr(args, "df_stage_position", "after_fs") or "after_fs"),
            "df_family_set": str(getattr(args, "df_family_set", "v6") or "v6"),
            "df_compute_ad": bool(getattr(args, "df_compute_ad", False)),
            "df_ad_bootstrap_samples": int(getattr(args, "df_ad_bootstrap_samples", 0) or 0),
            "df_compute_qq_pp": bool(getattr(args, "df_compute_qq_pp", False)),
            "df_compute_dip": bool(getattr(args, "df_compute_dip", True)),
            "df_dip_hist_bins": int(getattr(args, "df_dip_hist_bins", 40) or 40),
            "df_multimodal_fallback": str(getattr(args, "df_multimodal_fallback", "gmm") or "gmm"),
            "disable_cdf_transform": args.disable_cdf_transform,
            "disable_cdf_reliability_gate": args.disable_cdf_reliability_gate,
            "disable_low_gof_downweight": args.disable_low_gof_downweight,
            "enable_df_fastpath": False,
            "df_fastpath_scope": "none",
            "df_fastpath_trigger": str(getattr(args, "df_fastpath_trigger", "small_n_or_low_unique") or "small_n_or_low_unique"),
            "df_fastpath_small_n_threshold": int(getattr(args, "df_fastpath_small_n_threshold", 250) or 250),
            "df_fastpath_unique_ratio_threshold": float(getattr(args, "df_fastpath_unique_ratio_threshold", 0.05) or 0.05),
            "df_fastpath_n_unique_threshold": int(getattr(args, "df_fastpath_n_unique_threshold", 12) or 12),
            "enable_cdf_block_gating_cv": args.enable_cdf_block_gating_cv,
            "cdf_block_gating_n_blocks": args.cdf_block_gating_n_blocks,
            "cdf_block_gating_min_block_size": args.cdf_block_gating_min_block_size,
            "cdf_block_gating_cv_splits": args.cdf_block_gating_cv_splits,
            "cdf_block_gating_max_blocks": args.cdf_block_gating_max_blocks,
            "cdf_block_gating_time_budget_sec": args.cdf_block_gating_time_budget_sec,
            "cdf_block_gating_min_improvement": args.cdf_block_gating_min_improvement,
            "max_dist_features": args.max_dist_features,
            "disable_rank_prefilter": args.disable_rank_prefilter,
            "prefilter_top_k": args.prefilter_top_k,
            "prefilter_mi_weight": float(getattr(args, "prefilter_mi_weight", 0.60) or 0.60),
            "prefilter_f_weight": float(getattr(args, "prefilter_f_weight", 0.40) or 0.40),
            "prefilter_union_enabled": bool(getattr(args, "prefilter_union_enabled", False)),
            "prefilter_strategies": list(_parse_csv_or_space_list(getattr(args, "prefilter_strategies", ""))),
            "prefilter_nondefault_budget_fraction": float(
                getattr(args, "prefilter_nondefault_budget_fraction", 0.10) or 0.10
            ),
            "prefilter_wsnr_enabled": bool(getattr(args, "prefilter_wsnr_enabled", False)),
            "prefilter_bh_ttest_enabled": bool(getattr(args, "prefilter_bh_ttest_enabled", True)),
            "prefilter_bh_ttest_alpha": float(getattr(args, "prefilter_bh_ttest_alpha", 0.05) or 0.05),
            # T-R-272: variance floor snapshot.
            "prefilter_variance_floor_enabled": bool(getattr(args, "prefilter_variance_floor_enabled", True)),
            "prefilter_variance_floor_threshold": float(getattr(args, "prefilter_variance_floor_threshold", 1e-6) or 1e-6),
            "prefilter_variance_floor_mode_freq": float(getattr(args, "prefilter_variance_floor_mode_freq", 0.99) or 0.99),
            "disable_prefilter_rnaseq_transform": bool(
                getattr(args, "disable_prefilter_rnaseq_transform", False)
            ),
            "force_prefilter_rnaseq_transform": bool(
                getattr(args, "force_prefilter_rnaseq_transform", False)
            ),
            "enable_prefilter_rnaseq_nb_lrt": bool(
                getattr(args, "enable_prefilter_rnaseq_nb_lrt", False)
            ),
            "prefilter_rnaseq_nb_lrt_alpha": float(
                getattr(args, "prefilter_rnaseq_nb_lrt_alpha", 0.10) or 0.10
            ),
            "batch_correction": str(getattr(args, "batch_correction", "none") or "none"),
            "batch_correction_combat_prior_strength": float(
                getattr(args, "batch_correction_combat_prior_strength", 8.0) or 8.0
            ),
            "batch_correction_cdf_n_quantiles": int(
                getattr(args, "batch_correction_cdf_n_quantiles", 33) or 33
            ),
            "batch_correction_cdf_clip_low": float(
                getattr(args, "batch_correction_cdf_clip_low", 0.01) or 0.01
            ),
            "batch_correction_cdf_clip_high": float(
                getattr(args, "batch_correction_cdf_clip_high", 0.99) or 0.99
            ),
            "batch_label_policy": str(getattr(args, "batch_label_policy", "none") or "none"),
            "multiomics_adapter": str(getattr(args, "multiomics_adapter", "none") or "none"),
            "multiomics_integrator": str(getattr(args, "multiomics_integrator", "mb_plsda") or "mb_plsda"),
            "multiomics_n_components": int(getattr(args, "multiomics_n_components", 2) or 2),
            "selection_strategy": str(
                getattr(args, "selection_strategy", "mnpo_portfolio") or "mnpo_portfolio"
            ),
            "meta_learning_selector": str(
                getattr(args, "meta_learning_selector", "none") or "none"
            ),
            "meta_learning_confidence_threshold": float(
                getattr(args, "meta_learning_confidence_threshold", 0.55) or 0.55
            ),
            "meta_learning_records_path": str(
                getattr(args, "meta_learning_records_path", "") or ""
            ),
            "screening_enabled": bool(getattr(args, "screening_enabled", False)),
            "screening_method": str(getattr(args, "screening_method", "none") or "none"),
            "screening_pool_cap": int(getattr(args, "screening_pool_cap", 2000) or 2000),
            "screening_stir_n_neighbors": int(getattr(args, "screening_stir_n_neighbors", 10) or 10),
            "screening_stir_n_iter": int(getattr(args, "screening_stir_n_iter", 50) or 50),
            "screening_stir_keep_fraction": float(
                getattr(args, "screening_stir_keep_fraction", 0.5) or 0.5
            ),
            "screening_stir_min_features": int(getattr(args, "screening_stir_min_features", 20) or 20),
            "eval_models_enabled": bool(getattr(args, "eval_models_enabled", False)),
            "eval_models": list(_parse_csv_or_space_list(getattr(args, "eval_models", ""))),
            "eval_aggregate": str(getattr(args, "eval_aggregate", "mean") or "mean"),
            "eval_cvar_alpha": float(getattr(args, "eval_cvar_alpha", 0.33) or 0.33),
            "tier_lockout_enabled": bool(getattr(args, "tier_lockout_enabled", False)),
            "tier_classifier_mode": str(
                getattr(args, "tier_classifier_mode", "heuristic") or "heuristic"
            ).strip().lower(),
            "tier_classifier_model_path": str(
                getattr(args, "tier_classifier_model_path", "") or ""
            ).strip(),
            "tier_lockout_tier": str(getattr(args, "tier_lockout_tier", "easy") or "easy").strip().lower(),
            "tier_lockout_difficulty_source": str(
                getattr(args, "tier_lockout_difficulty_source", "historical") or "historical"
            ).strip().lower(),
            "tier_lockout_fallback_methods": list(
                _resolve_tier_lockout_fallback_methods(
                    args,
                    tuple(FS_METHOD_SETS.get(str(getattr(args, "fs_method_set", "")), tuple())),
                )
            ),
            "tier_routing_enabled": bool(getattr(args, "tier_routing_enabled", False)),
            "tier_routing_difficulty_classifier": str(
                getattr(args, "tier_routing_difficulty_classifier", "meta_features") or "meta_features"
            ).strip().lower(),
            "tier_routing_table": {
                str(k): list(v) for k, v in _parse_tier_routing_table(getattr(args, "tier_routing_table", "")).items()
            },
            "regime_gating_enabled": bool(getattr(args, "regime_gating_enabled", False)),
            "regime_gating_difficulty_source": str(
                getattr(args, "regime_gating_difficulty_source", "historical") or "historical"
            ).strip().lower(),
            "regime_gating_target_tier": str(
                getattr(args, "regime_gating_target_tier", "very_hard") or "very_hard"
            ).strip().lower(),
            "regime_gating_min_samples_per_class": float(
                getattr(args, "regime_gating_min_samples_per_class", 7.0) or 7.0
            ),
            "regime_gating_use_expanded_features": bool(
                getattr(args, "regime_gating_use_expanded_features", False)
            ),
            "regime_gating_min_fisher_f1": float(
                getattr(args, "regime_gating_min_fisher_f1", 0.10) or 0.10
            ),
            "regime_gating_max_n1_borderline": float(
                getattr(args, "regime_gating_max_n1_borderline", 0.40) or 0.40
            ),
            "regime_gating_low_p_over_n_threshold": float(
                getattr(args, "regime_gating_low_p_over_n_threshold", 0.0) or 0.0
            ),
            "regime_gating_simple_methods": list(
                _resolve_regime_gating_simple_methods(
                    args,
                    tuple(FS_METHOD_SETS.get("strict_plus_mrmr", tuple())),
                )
            ),
            "regime_gating_very_hard_portfolio_max_methods": int(
                getattr(args, "regime_gating_very_hard_portfolio_max_methods", 4) or 4
            ),
            "regime_gating_very_hard_copula_derandomize_runs": int(
                getattr(args, "regime_gating_very_hard_copula_derandomize_runs", 5) or 5
            ),
            "regime_gating_low_p_over_n_mode": str(
                getattr(args, "regime_gating_low_p_over_n_mode", "fast_univariate_filter") or "fast_univariate_filter"
            ).strip().lower(),
            "regime_gating_low_p_over_n_filter_max_k": int(
                getattr(args, "regime_gating_low_p_over_n_filter_max_k", 200) or 200
            ),
            "regime_gating_very_hard_min_classes": int(
                getattr(args, "regime_gating_very_hard_min_classes", 5) or 5
            ),
            # T-R-268: extreme multiclass gate snapshot.
            "regime_gating_extreme_multiclass_enabled": bool(
                getattr(args, "regime_gating_extreme_multiclass_enabled", True)
            ),
            "regime_gating_extreme_multiclass_threshold": int(
                getattr(args, "regime_gating_extreme_multiclass_threshold", 8) or 8
            ),
            "regime_gating_extreme_multiclass_min_samples_per_class": float(
                getattr(args, "regime_gating_extreme_multiclass_min_samples_per_class", 11.0) or 11.0
            ),
            "mnpo_performance_oracle_mode": str(
                getattr(args, "mnpo_performance_oracle_mode", "single") or "single"
            ).strip().lower(),
            "folding_method": str(getattr(args, "folding_method", "pls_da") or "pls_da"),
            "folding_n_components": int(getattr(args, "folding_n_components", 512) or 512),
            "folding_rff_gamma": (
                None
                if getattr(args, "folding_rff_gamma", None) is None
                else float(getattr(args, "folding_rff_gamma"))
            ),
            "folding_pls_components": int(getattr(args, "folding_pls_components", 32) or 32),
            "folding_pls_scale": not bool(getattr(args, "disable_folding_pls_scale", False)),
            "folding_pls_min_classes": int(getattr(args, "folding_pls_min_classes", 5) or 5),
            "folding_pls_min_n_per_class": int(
                getattr(args, "folding_pls_min_n_per_class", 3) or 3
            ),
            "folding_pls_max_imbalance_ratio": float(
                getattr(args, "folding_pls_max_imbalance_ratio", 6.0) or 6.0
            ),
            "folding_prefilter_k": int(getattr(args, "folding_prefilter_k", 0) or 0),
            "enable_face_domain_projection": bool(
                getattr(args, "enable_face_domain_projection", False)
            ),
            "enable_dist_stability_weight": args.enable_dist_stability_weight,
            "enable_balanced_fs_subsample": args.enable_balanced_fs_subsample,
            "fs_min_per_class": args.fs_min_per_class,
            "fs_method_timeout_sec": float(getattr(args, "fs_method_timeout_sec", 0.0) or 0.0),
            "fs_linear_svm_max_iter": int(getattr(args, "fs_linear_svm_max_iter", 10000) or 10000),
            "enable_fs_runtime_racing": bool(getattr(args, "enable_fs_runtime_racing", False)),
            "fs_runtime_racing_proxy_splits": int(
                getattr(args, "fs_runtime_racing_proxy_splits", 1) or 1
            ),
            "fs_runtime_racing_keep_fraction": float(
                getattr(args, "fs_runtime_racing_keep_fraction", 0.60) or 0.60
            ),
            "fs_runtime_racing_min_candidates": int(
                getattr(args, "fs_runtime_racing_min_candidates", 4) or 4
            ),
            "fs_runtime_racing_runtime_weight": float(
                getattr(args, "fs_runtime_racing_runtime_weight", 0.15) or 0.15
            ),
            "fs_runtime_racing_mode": str(
                getattr(args, "fs_runtime_racing_mode", "single_stage") or "single_stage"
            ),
            "fs_runtime_racing_stages": int(getattr(args, "fs_runtime_racing_stages", 2) or 2),
            "fs_runtime_racing_confidence_bound": str(
                getattr(args, "fs_runtime_racing_confidence_bound", "none") or "none"
            ),
            "fs_runtime_racing_delta": float(getattr(args, "fs_runtime_racing_delta", 0.10) or 0.10),
            "fs_portfolio_size": int(getattr(args, "fs_portfolio_size", 6) or 6),
            "fs_portfolio_size_guard": str(getattr(args, "fs_portfolio_size_guard", "none") or "none"),
            "enable_fs_adaptive_portfolio_sizing": bool(
                getattr(args, "enable_fs_adaptive_portfolio_sizing", False)
            ),
            "fs_adaptive_size_min": getattr(args, "fs_adaptive_size_min", None),
            "fs_adaptive_size_max": getattr(args, "fs_adaptive_size_max", None),
            "fs_adaptive_sizing_variance_penalty": bool(
                getattr(args, "fs_adaptive_sizing_variance_penalty", False)
            ),
            "fs_adaptive_sizing_variance_penalty_strength": float(
                getattr(args, "fs_adaptive_sizing_variance_penalty_strength", 0.5) or 0.5
            ),
            # T-R-266: Pareto portfolio sizing snapshot.
            "fs_pareto_portfolio_sizing_enabled": bool(
                getattr(args, "fs_pareto_portfolio_sizing_enabled", False)
            ),
            # T-R-271: stability-weighted portfolio aggregation snapshot.
            "fs_stability_weighted_aggregation_enabled": bool(
                getattr(args, "fs_stability_weighted_aggregation_enabled", False)
            ),
            "enable_fs_rashomon": bool(getattr(args, "enable_fs_rashomon", False)),
            "fs_rashomon_max_models": int(getattr(args, "fs_rashomon_max_models", 12) or 12),
            "fs_rashomon_score_tolerance": float(
                getattr(args, "fs_rashomon_score_tolerance", 0.01) or 0.01
            ),
            "fs_mnpo_consensus_exclude_methods": list(getattr(args, "fs_mnpo_consensus_exclude_methods", ()) or ()),
            "fs_mnpo_consensus_exclude_protect_top_k": int(
                getattr(args, "fs_mnpo_consensus_exclude_protect_top_k", 0) or 0
            ),
            "fs_mnpo_include_legacy_consensus": not bool(getattr(args, "disable_fs_mnpo_legacy_consensus", False)),
            "fs_mnpo_include_majority_consensus": not bool(getattr(args, "disable_fs_mnpo_majority_consensus", False)),
            "fs_inner_cv_splits": int(getattr(args, "fs_inner_cv_splits", 3) or 3),
            "fs_inner_cv_repeats": int(getattr(args, "fs_inner_cv_repeats", 1) or 1),
            "use_stability_oracle": not bool(getattr(args, "disable_fs_stability_oracle", False)),
            "use_complexity_oracle": not bool(getattr(args, "disable_fs_complexity_oracle", False)),
            "use_robust_oracle": not bool(getattr(args, "disable_fs_robust_oracle", False)),
            "fs_diversity_oracle_mode": args.fs_diversity_oracle_mode,
            "fs_diversity_redundancy_weight": args.fs_diversity_redundancy_weight,
            "fs_diversity_complementarity_weight": args.fs_diversity_complementarity_weight,
            "fs_use_cvar_oracle": bool(getattr(args, "fs_use_cvar_oracle", False)),
            "fs_cvar_alpha": float(getattr(args, "fs_cvar_alpha", 0.33) or 0.33),
            "fs_oracle_weighting_mode": str(
                getattr(args, "fs_oracle_weighting_mode", "tritrust") or "tritrust"
            ),
            "fs_shapley_n_coalitions_max": int(
                getattr(args, "fs_shapley_n_coalitions_max", 4096) or 4096
            ),
            "fs_shapley_bayesian_shrinkage": bool(
                getattr(args, "fs_shapley_bayesian_shrinkage", False)
            ),
            "fs_shapley_bayesian_prior_strength": float(
                getattr(args, "fs_shapley_bayesian_prior_strength", 8.0) or 8.0
            ),
            "fs_use_interaction_oracle": bool(getattr(args, "fs_use_interaction_oracle", False)),
            "fs_interaction_oracle_min_n_train": int(
                getattr(args, "fs_interaction_oracle_min_n_train", 150) or 150
            ),
            "fs_interaction_oracle_pool_size_cap": int(
                getattr(args, "fs_interaction_oracle_pool_size_cap", 64) or 64
            ),
            "fs_interaction_oracle_pair_cap": int(
                getattr(args, "fs_interaction_oracle_pair_cap", 20000) or 20000
            ),
            "fs_use_ubayfs_oracle": bool(getattr(args, "fs_use_ubayfs_oracle", False)),
            "fs_ubayfs_n_bootstrap": int(getattr(args, "fs_ubayfs_n_bootstrap", 32) or 32),
            "fs_ubayfs_min_n": int(getattr(args, "fs_ubayfs_min_n", 100) or 100),
            "fs_ubayfs_prior_weight": float(getattr(args, "fs_ubayfs_prior_weight", 0.0) or 0.0),
            "fs_use_conformal_uq": bool(getattr(args, "fs_use_conformal_uq", False)),
            "fs_conformal_uq_alpha": float(getattr(args, "fs_conformal_uq_alpha", 0.10) or 0.10),
            "fs_conformal_uq_min_folds": int(getattr(args, "fs_conformal_uq_min_folds", 5) or 5),
            "fs_fold_preference_mode": str(
                getattr(args, "fs_fold_preference_mode", "vote") or "vote"
            ),
            "fs_use_conformal_efficiency": bool(
                getattr(args, "fs_use_conformal_efficiency", False)
            ),
            "fs_conformal_efficiency_method": str(
                getattr(args, "fs_conformal_efficiency_method", "split") or "split"
            ),
            "fs_oracle_weight_js_shrinkage": bool(
                getattr(args, "fs_oracle_weight_js_shrinkage", False)
            ),
            "fs_payoff_shrinkage_kappa": float(
                getattr(args, "fs_payoff_shrinkage_kappa", 0.0) or 0.0
            ),
            "fs_performance_balanced_weight": args.fs_performance_balanced_weight,
            "fs_performance_macro_f1_weight": args.fs_performance_macro_f1_weight,
            "enable_fs_adaptive_imbalance_score": args.enable_fs_adaptive_imbalance_score,
            "fs_imbalance_ratio_trigger": args.fs_imbalance_ratio_trigger,
            "fs_imbalance_min_classes": args.fs_imbalance_min_classes,
            "fs_rank_aggregation_mode": args.fs_rank_aggregation_mode,
            "enable_fs_wrapper_refine": args.enable_fs_wrapper_refine,
            "fs_wrapper_refine_top_k": args.fs_wrapper_refine_top_k,
            "fs_wrapper_refine_max_add": args.fs_wrapper_refine_max_add,
            "fs_wrapper_refine_min_gain": args.fs_wrapper_refine_min_gain,
            "fs_ova_negative_ratio": args.fs_ova_negative_ratio,
            "fs_ova_min_classes": args.fs_ova_min_classes,
            "fs_ova_min_pos_samples": args.fs_ova_min_pos_samples,
            "fs_ova_class_weight_mode": args.fs_ova_class_weight_mode,
            "fs_ova_aggregation_mode": args.fs_ova_aggregation_mode,
            "fs_ova_aggregation_p": args.fs_ova_aggregation_p,
            "fs_ova_linear_backend": args.fs_ova_linear_backend,
            "enable_fs_ova_calibration": bool(getattr(args, "enable_fs_ova_calibration", False)),
            "fs_ova_calibration_cv": int(getattr(args, "fs_ova_calibration_cv", 3) or 3),
            "fs_ecoc_min_classes": int(getattr(args, "fs_ecoc_min_classes", 4) or 4),
            "fs_ecoc_max_ovo_pairs": int(getattr(args, "fs_ecoc_max_ovo_pairs", 8) or 8),
            "fs_ecoc_random_code_bits": int(getattr(args, "fs_ecoc_random_code_bits", 4) or 4),
            "fs_ecoc_class_complexity_weight": float(
                getattr(args, "fs_ecoc_class_complexity_weight", 1.0) or 1.0
            ),
            "fs_ecoc_include_ova_tasks": not bool(getattr(args, "disable_fs_ecoc_include_ova_tasks", False)),
            "fs_ecoc_negative_ratio": float(getattr(args, "fs_ecoc_negative_ratio", 2.0) or 2.0),
            "fs_joint_multiclass_min_classes": int(getattr(args, "fs_joint_multiclass_min_classes", 3) or 3),
            "fs_joint_multiclass_max_features": int(getattr(args, "fs_joint_multiclass_max_features", 256) or 256),
            "fs_joint_multiclass_path_grid_size": int(
                getattr(args, "fs_joint_multiclass_path_grid_size", 6) or 6
            ),
            "fs_joint_multiclass_min_c": float(getattr(args, "fs_joint_multiclass_min_c", 0.05) or 0.05),
            "fs_joint_multiclass_max_c": float(getattr(args, "fs_joint_multiclass_max_c", 1.6) or 1.6),
            "fs_joint_multiclass_l1_ratio": float(getattr(args, "fs_joint_multiclass_l1_ratio", 0.55) or 0.55),
            "fs_joint_multiclass_univariate_blend": float(
                getattr(args, "fs_joint_multiclass_univariate_blend", 0.20) or 0.20
            ),
            "fs_dove_min_classes": int(getattr(args, "fs_dove_min_classes", 3) or 3),
            "fs_dove_max_pairs_per_class": int(getattr(args, "fs_dove_max_pairs_per_class", 4) or 4),
            "fs_dove_path_grid_size": int(getattr(args, "fs_dove_path_grid_size", 5) or 5),
            "fs_dove_specificity_weight": float(getattr(args, "fs_dove_specificity_weight", 0.35) or 0.35),
            "fs_dove_minority_boost": float(getattr(args, "fs_dove_minority_boost", 0.50) or 0.50),
            "fs_sparse_multinomial_min_classes": int(
                getattr(args, "fs_sparse_multinomial_min_classes", 3) or 3
            ),
            "fs_sparse_multinomial_max_features": int(
                getattr(args, "fs_sparse_multinomial_max_features", 320) or 320
            ),
            "fs_sparse_multinomial_path_grid_size": int(
                getattr(args, "fs_sparse_multinomial_path_grid_size", 6) or 6
            ),
            "fs_sparse_multinomial_min_c": float(
                getattr(args, "fs_sparse_multinomial_min_c", 0.05) or 0.05
            ),
            "fs_sparse_multinomial_max_c": float(
                getattr(args, "fs_sparse_multinomial_max_c", 1.6) or 1.6
            ),
            "fs_sparse_multinomial_backend": str(
                getattr(args, "fs_sparse_multinomial_backend", "mixed") or "mixed"
            ),
            "fs_sparse_multinomial_l1_ratio": float(
                getattr(args, "fs_sparse_multinomial_l1_ratio", 0.70) or 0.70
            ),
            "fs_sparse_multinomial_univariate_blend": float(
                getattr(args, "fs_sparse_multinomial_univariate_blend", 0.20) or 0.20
            ),
            "fs_sparse_multinomial_max_iter": int(
                getattr(args, "fs_sparse_multinomial_max_iter", 5000) or 5000
            ),
            "fs_sparse_multinomial_screening_mode": _canonicalize_sparse_screening_mode(
                getattr(args, "fs_sparse_multinomial_screening_mode", "none"),
                warn_deprecated=False,
            ),
            "fs_sparse_multinomial_screening_keep_fraction": float(
                getattr(args, "fs_sparse_multinomial_screening_keep_fraction", 1.0) or 1.0
            ),
            "fs_sparse_multinomial_screening_min_features": int(
                getattr(args, "fs_sparse_multinomial_screening_min_features", 64) or 64
            ),
            "fs_sparse_multinomial_screening_fallback_on_failure": not bool(
                getattr(args, "disable_fs_sparse_multinomial_screening_fallback_on_failure", False)
            ),
            "fs_nsc_shrinkage_grid_size": int(
                getattr(args, "fs_nsc_shrinkage_grid_size", 6) or 6
            ),
            "fs_nsc_min_classes": int(
                getattr(args, "fs_nsc_min_classes", 3) or 3
            ),
            "fs_nsc_thresholding_mode": str(
                getattr(args, "fs_nsc_thresholding_mode", "soft") or "soft"
            ),
            "fs_nsc_order_quantile": float(
                getattr(args, "fs_nsc_order_quantile", 0.75) or 0.75
            ),
            "enable_fs_nsc_deep_shrinkage_search": bool(
                getattr(args, "enable_fs_nsc_deep_shrinkage_search", False)
            ),
            "fs_class_pareto_min_classes": int(
                getattr(args, "fs_class_pareto_min_classes", 3) or 3
            ),
            "fs_class_pareto_top_per_class": int(
                getattr(args, "fs_class_pareto_top_per_class", 64) or 64
            ),
            "fs_class_pareto_global_fraction": float(
                getattr(args, "fs_class_pareto_global_fraction", 0.40) or 0.40
            ),
            "fs_class_pareto_minority_boost": float(
                getattr(args, "fs_class_pareto_minority_boost", 0.50) or 0.50
            ),
            "fs_class_pareto_kw_weight": float(
                getattr(args, "fs_class_pareto_kw_weight", 0.25) or 0.25
            ),
            "fs_sdr_min_classes": int(getattr(args, "fs_sdr_min_classes", 3) or 3),
            "fs_sdr_prefilter_max_features": int(
                getattr(args, "fs_sdr_prefilter_max_features", 512) or 512
            ),
            "fs_sdr_n_components": int(getattr(args, "fs_sdr_n_components", 3) or 3),
            "fs_sdr_covariance_ridge": float(
                getattr(args, "fs_sdr_covariance_ridge", 1e-3) or 1e-3
            ),
            "enable_fs_per_class_quota": bool(getattr(args, "enable_fs_per_class_quota", False)),
            "fs_per_class_quota_min_per_class": int(
                getattr(args, "fs_per_class_quota_min_per_class", 1) or 1
            ),
            "fs_per_class_quota_max_fraction": float(
                getattr(args, "fs_per_class_quota_max_fraction", 0.60) or 0.60
            ),
            "fs_hsic_lasso_alpha": float(
                getattr(args, "fs_hsic_lasso_alpha", 0.01) or 0.01
            ),
            "fs_hsic_lasso_prefilter_max_features": int(
                getattr(args, "fs_hsic_lasso_prefilter_max_features", 128) or 128
            ),
            "fs_hsic_lasso_feature_sigma": float(
                getattr(args, "fs_hsic_lasso_feature_sigma", 0.0) or 0.0
            ),
            "fs_hsic_lasso_target_sigma": float(
                getattr(args, "fs_hsic_lasso_target_sigma", 0.0) or 0.0
            ),
            "fs_hsic_lasso_relevance_blend": float(
                getattr(args, "fs_hsic_lasso_relevance_blend", 0.20) or 0.20
            ),
            "fs_hsic_lasso_max_iter": int(
                getattr(args, "fs_hsic_lasso_max_iter", 4000) or 4000
            ),
            "fs_ipss_path_grid_size": args.fs_ipss_path_grid_size,
            "fs_ipss_min_c": args.fs_ipss_min_c,
            "fs_ipss_max_c": args.fs_ipss_max_c,
            "fs_ipss_target_fdr": args.fs_ipss_target_fdr,
            "fs_ipss_null_shuffle_rounds": args.fs_ipss_null_shuffle_rounds,
            "enable_fs_ipss_eats_threshold": args.enable_fs_ipss_eats_threshold,
            "fs_ipss_eats_exclusion_quantile": args.fs_ipss_eats_exclusion_quantile,
            "fs_ipss_eats_min_threshold": args.fs_ipss_eats_min_threshold,
            "fs_ipss_importance_model": args.fs_ipss_importance_model,
            "fs_ipss_gate_min_classes": int(getattr(args, "fs_ipss_gate_min_classes", 0) or 0),
            "fs_ipss_gate_min_p_over_n": float(getattr(args, "fs_ipss_gate_min_p_over_n", 0.0) or 0.0),
            "fs_cluster_corr_threshold": args.fs_cluster_corr_threshold,
            "fs_cluster_max_per_cluster": args.fs_cluster_max_per_cluster,
            "fs_cluster_min_freq": args.fs_cluster_min_freq,
            "enable_fs_stability_loss_guided_validation": args.enable_fs_stability_loss_guided_validation,
            "fs_stability_validation_fraction": args.fs_stability_validation_fraction,
            "fs_stability_validation_quantile": args.fs_stability_validation_quantile,
            "fs_stability_validation_min_samples": args.fs_stability_validation_min_samples,
            "fs_copula_knockoff_draws": args.fs_copula_knockoff_draws,
            "fs_copula_alpha_kn": args.fs_copula_alpha_kn,
            "fs_copula_alpha_ebh": args.fs_copula_alpha_ebh,
            "fs_copula_truncation_level": args.fs_copula_truncation_level,
            "fs_copula_generator": str(getattr(args, "fs_copula_generator", "copula") or "copula"),
            "fs_copula_deepdrk_latent_fraction": float(
                getattr(args, "fs_copula_deepdrk_latent_fraction", 0.35) or 0.35
            ),
            "fs_copula_deepdrk_noise_scale": float(
                getattr(args, "fs_copula_deepdrk_noise_scale", 1.0) or 1.0
            ),
            "fs_copula_stabilizer_runs": args.fs_copula_stabilizer_runs,
            "enable_fs_copula_stabilizer_ebh": args.enable_fs_copula_stabilizer_ebh,
            "fs_copula_stabilizer_seed_stride": args.fs_copula_stabilizer_seed_stride,
            "fs_importance_uq_enabled": bool(getattr(args, "fs_importance_uq_enabled", False)),
            "fs_importance_uq_min_cv_folds": int(getattr(args, "fs_importance_uq_min_cv_folds", 3) or 3),
            "fs_decorrelated_stability_eps": args.fs_decorrelated_stability_eps,
            "fs_decorrelated_stability_min_max_abs_corr": float(
                getattr(args, "fs_decorrelated_stability_min_max_abs_corr", 0.0) or 0.0
            ),
            "fs_iterative_pruning_pool_factor": float(
                getattr(args, "fs_iterative_pruning_pool_factor", 2.5) or 2.5
            ),
            "fs_iterative_pruning_max_rounds": int(
                getattr(args, "fs_iterative_pruning_max_rounds", 32) or 32
            ),
            "fs_iterative_pruning_min_improvement": float(
                getattr(args, "fs_iterative_pruning_min_improvement", -0.002) or -0.002
            ),
            "fs_iterative_pruning_max_cumulative_loss": float(
                getattr(args, "fs_iterative_pruning_max_cumulative_loss", 0.02) or 0.02
            ),
            "fs_iterative_pruning_redundancy_weight": float(
                getattr(args, "fs_iterative_pruning_redundancy_weight", 0.65) or 0.65
            ),
            "fs_iterative_pruning_bounded_prefilter_cap": int(
                getattr(args, "fs_iterative_pruning_bounded_prefilter_cap", 220) or 220
            ),
            "fs_iterative_pruning_bounded_candidate_fraction": float(
                getattr(args, "fs_iterative_pruning_bounded_candidate_fraction", 0.35) or 0.35
            ),
            "fs_iterative_pruning_bounded_min_candidates": int(
                getattr(args, "fs_iterative_pruning_bounded_min_candidates", 4) or 4
            ),
            "fs_iterative_pruning_bounded_max_evaluations": int(
                getattr(args, "fs_iterative_pruning_bounded_max_evaluations", 48) or 48
            ),
            "fs_iterative_pruning_bounded_max_runtime_seconds": float(
                getattr(args, "fs_iterative_pruning_bounded_max_runtime_seconds", 30.0) or 30.0
            ),
            "fs_iterative_pruning_bounded_enable_class_gating": not bool(
                getattr(args, "disable_fs_iterative_pruning_bounded_class_gating", False)
            ),
            "fs_iterative_pruning_bounded_multiclass_scale": float(
                getattr(args, "fs_iterative_pruning_bounded_multiclass_scale", 0.70) or 0.70
            ),
            "fs_iterative_pruning_bounded_imbalance_trigger": float(
                getattr(args, "fs_iterative_pruning_bounded_imbalance_trigger", 2.5) or 2.5
            ),
            "fs_iterative_pruning_bounded_imbalance_scale": float(
                getattr(args, "fs_iterative_pruning_bounded_imbalance_scale", 0.75) or 0.75
            ),
            "enable_fs_iterative_pruning_bounded_cpss_overlay": bool(
                getattr(args, "enable_fs_iterative_pruning_bounded_cpss_overlay", False)
            ),
            "fs_iterative_pruning_bounded_cpss_pairs": int(
                getattr(args, "fs_iterative_pruning_bounded_cpss_pairs", 4) or 4
            ),
            "fs_iterative_pruning_bounded_cpss_stability_threshold": float(
                getattr(args, "fs_iterative_pruning_bounded_cpss_stability_threshold", 0.60) or 0.60
            ),
            "fs_iterative_pruning_bounded_cpss_min_stable_features": int(
                getattr(args, "fs_iterative_pruning_bounded_cpss_min_stable_features", 2) or 2
            ),
            "fs_iterative_pruning_bounded_cpss_min_jaccard": float(
                getattr(args, "fs_iterative_pruning_bounded_cpss_min_jaccard", 0.35) or 0.35
            ),
            "fs_iterative_pruning_bounded_cpss_max_score_drop": float(
                getattr(args, "fs_iterative_pruning_bounded_cpss_max_score_drop", 0.005) or 0.005
            ),
            "enable_fs_iterative_pruning_class_pareto_prefilter": bool(
                getattr(args, "enable_fs_iterative_pruning_class_pareto_prefilter", False)
            ),
            "fs_iterative_pruning_class_pareto_min_classes": int(
                getattr(args, "fs_iterative_pruning_class_pareto_min_classes", 3) or 3
            ),
            "fs_iterative_pruning_class_pareto_top_per_class": int(
                getattr(args, "fs_iterative_pruning_class_pareto_top_per_class", 64) or 64
            ),
            "fs_iterative_pruning_class_pareto_global_fraction": float(
                getattr(args, "fs_iterative_pruning_class_pareto_global_fraction", 0.40) or 0.40
            ),
            "fs_iterative_pruning_class_pareto_minority_boost": float(
                getattr(args, "fs_iterative_pruning_class_pareto_minority_boost", 0.50) or 0.50
            ),
            "enable_fs_iterative_pruning_class_pareto_stability_gate": bool(
                getattr(args, "enable_fs_iterative_pruning_class_pareto_stability_gate", False)
            ),
            "fs_iterative_pruning_class_pareto_stability_subsamples": int(
                getattr(args, "fs_iterative_pruning_class_pareto_stability_subsamples", 6) or 6
            ),
            "fs_iterative_pruning_class_pareto_stability_fraction": float(
                getattr(args, "fs_iterative_pruning_class_pareto_stability_fraction", 0.70) or 0.70
            ),
            "fs_iterative_pruning_class_pareto_stability_threshold": float(
                getattr(args, "fs_iterative_pruning_class_pareto_stability_threshold", 0.55) or 0.55
            ),
            "fs_iterative_pruning_class_pareto_stability_min_overlap": float(
                getattr(args, "fs_iterative_pruning_class_pareto_stability_min_overlap", 0.50) or 0.50
            ),
            "fs_iterative_pruning_class_pareto_stability_min_stable_features": int(
                getattr(args, "fs_iterative_pruning_class_pareto_stability_min_stable_features", 4) or 4
            ),
            "fs_iterative_pruning_class_pareto_stability_fallback_on_failure": not bool(
                getattr(args, "disable_fs_iterative_pruning_class_pareto_stability_fallback_on_failure", False)
            ),
            "enable_diversity_oracle": args.enable_diversity_oracle,
            "enable_hybrid_model_cv": args.enable_hybrid_model_cv,
            "model_cv_lr_max_iter": int(getattr(args, "model_cv_lr_max_iter", 10000) or 10000),
            "model_cv_balanced_weight": args.model_cv_balanced_weight,
            "model_cv_macro_f1_weight": args.model_cv_macro_f1_weight,
            "enable_model_cv_runtime_containment": bool(
                getattr(args, "enable_model_cv_runtime_containment", False)
            ),
            "model_cv_runtime_max_candidates": int(
                getattr(args, "model_cv_runtime_max_candidates", 0) or 0
            ),
            "model_cv_runtime_high_p_over_n_threshold": float(
                getattr(args, "model_cv_runtime_high_p_over_n_threshold", 40.0) or 40.0
            ),
            "model_cv_runtime_high_class_threshold": int(
                getattr(args, "model_cv_runtime_high_class_threshold", 6) or 6
            ),
            "model_cv_runtime_min_class_count_threshold": int(
                getattr(args, "model_cv_runtime_min_class_count_threshold", 12) or 12
            ),
            "classifier_selection_mode": str(
                getattr(args, "classifier_selection_mode", "legacy") or "legacy"
            ),
            "classification_backend": str(
                getattr(args, "classification_backend", "sklearn") or "sklearn"
            ),
            "flaml_time_budget": int(getattr(args, "flaml_time_budget", 60) or 60),
            "optuna_time_budget": int(getattr(args, "optuna_time_budget", 120) or 120),
            "optuna_n_trials": int(getattr(args, "optuna_n_trials", 25) or 25),
            "classifier_oracle_k": int(getattr(args, "classifier_oracle_k", 1) or 1),
            "classifier_oracle_weighting_mode": str(
                getattr(args, "classifier_oracle_weighting_mode", "tritrust") or "tritrust"
            ),
            "disable_classifier_oracle_calibration": bool(
                getattr(args, "disable_classifier_oracle_calibration", False)
            ),
            "disable_classifier_oracle_james_stein": bool(
                getattr(args, "disable_classifier_oracle_james_stein", False)
            ),
            "enable_classifier_oracle_cvar": bool(
                getattr(args, "enable_classifier_oracle_cvar", False)
            ),
            "classifier_oracle_cvar_alpha": float(
                getattr(args, "classifier_oracle_cvar_alpha", 0.33) or 0.33
            ),
            "enable_classifier_oracle_dynamic_complexity": bool(
                getattr(args, "enable_classifier_oracle_dynamic_complexity", False)
            ),
            "enable_classifier_oracle_portfolio_diversity": bool(
                getattr(args, "enable_classifier_oracle_portfolio_diversity", False)
            ),
            "classifier_oracle_portfolio_overlap_threshold": float(
                getattr(args, "classifier_oracle_portfolio_overlap_threshold", 0.75) or 0.75
            ),
            "classifier_oracle_portfolio_corr_threshold": float(
                getattr(args, "classifier_oracle_portfolio_corr_threshold", 0.85) or 0.85
            ),
            "disable_classifier_oracle_hoeffding_racing": bool(
                getattr(args, "disable_classifier_oracle_hoeffding_racing", False)
            ),
            "classifier_oracle_hoeffding_delta": float(
                getattr(args, "classifier_oracle_hoeffding_delta", 0.10) or 0.10
            ),
            "disable_classifier_oracle_bbc": bool(
                getattr(args, "disable_classifier_oracle_bbc", False)
            ),
            "classifier_oracle_bbc_bootstrap_rounds": int(
                getattr(args, "classifier_oracle_bbc_bootstrap_rounds", 200) or 200
            ),
            "classifier_oracle_bbc_ci_level": float(
                getattr(args, "classifier_oracle_bbc_ci_level", 0.90) or 0.90
            ),
            "enable_classifier_oracle_ensemble": bool(
                getattr(args, "enable_classifier_oracle_ensemble", False)
            ),
            "classifier_oracle_ensemble_voting_mode": str(
                getattr(args, "classifier_oracle_ensemble_voting_mode", "hard") or "hard"
            ),
            "enable_classifier_oracle_greedy_ensemble": bool(
                getattr(args, "enable_classifier_oracle_greedy_ensemble", False)
            ),
            "classifier_oracle_greedy_ensemble_rounds": int(
                getattr(args, "classifier_oracle_greedy_ensemble_rounds", 10) or 10
            ),
            "enable_classifier_oracle_candidate_pruning": bool(
                getattr(args, "enable_classifier_oracle_candidate_pruning", False)
            ),
            "classifier_oracle_candidate_pruning_threshold": float(
                getattr(args, "classifier_oracle_candidate_pruning_threshold", 0.0) or 0.0
            ),
            "enable_classifier_oracle_incumbent_early_stopping": bool(
                getattr(args, "enable_classifier_oracle_incumbent_early_stopping", False)
            ),
            "classifier_oracle_behavior_profile": str(
                getattr(args, "classifier_oracle_behavior_profile", "current") or "current"
            ).strip().lower(),
            "exclude_classifiers": list(_flatten_cli_values(getattr(args, "exclude_classifiers", []))),
            "classifier_regime_candidate_exclusions": list(
                _flatten_cli_values(getattr(args, "classifier_regime_candidate_exclusions", []))
            ),
            "classifier_complexity_prior_override": list(
                _flatten_cli_values(getattr(args, "classifier_complexity_prior_override", []))
            ),
            "disable_classifier_oracle_per_family_flaml": bool(
                getattr(args, "disable_classifier_oracle_per_family_flaml", False)
            ),
            "enable_classifier_conformal": bool(
                getattr(args, "enable_classifier_conformal", False)
            ),
            "classifier_conformal_alpha": float(
                getattr(args, "classifier_conformal_alpha", 0.10) or 0.10
            ),
            "classifier_conformal_calibration_fraction": float(
                getattr(args, "classifier_conformal_calibration_fraction", 0.25) or 0.25
            ),
            "classifier_conformal_min_calibration": int(
                getattr(args, "classifier_conformal_min_calibration", 20) or 20
            ),
            "classifier_conformal_output_sets": bool(
                getattr(args, "classifier_conformal_output_sets", False)
            ),
            "classifier_conformal_method": str(
                getattr(args, "classifier_conformal_method", "split") or "split"
            ).strip().lower(),
            "model_candidate_profile": str(getattr(args, "model_candidate_profile", "default") or "default"),
            "include_elastic_net_model": args.include_elastic_net_model,
            "include_rf_model": args.include_rf_model,
            "include_knn_model": args.include_knn_model,
            "include_svm_linear_model": getattr(args, "include_svm_linear_model", False),
            "include_dlda_model": getattr(args, "include_dlda_model", False),
            "include_nsc_model": bool(getattr(args, "include_nsc_model", False)),
            "include_pls_da_model": bool(getattr(args, "include_pls_da_model", False)),
            "include_gpc_model": bool(getattr(args, "include_gpc_model", False)),
            "include_nb_model": getattr(args, "include_nb_model", False),
            "include_vote_ensemble_model": getattr(args, "include_vote_ensemble_model", False),
            "include_rp_ensemble_model": bool(getattr(args, "include_rp_ensemble_model", False)),
            "include_dbda_model": bool(getattr(args, "include_dbda_model", False)),
            "include_gqda_model": bool(getattr(args, "include_gqda_model", False)),
            "include_bc_svm_linear_model": bool(getattr(args, "include_bc_svm_linear_model", False)),
            "include_sglnn_model": bool(getattr(args, "include_sglnn_model", False)),
            "include_xgb_model": args.include_xgb_model,
            "include_lgbm_model": bool(getattr(args, "include_lgbm_model", False)),
            "include_extra_tree_model": bool(getattr(args, "include_extra_tree_model", False)),
            "include_catboost_model": bool(getattr(args, "include_catboost_model", False)),
            "include_tabpfn_model": args.include_tabpfn_model,
            "enable_stage2_ratio_augmentation": bool(
                getattr(args, "enable_stage2_ratio_augmentation", False)
            ),
            "stage2_ratio_max_features": int(
                getattr(args, "stage2_ratio_max_features", 16) or 16
            ),
            "stage2_ratio_selection_method": str(
                getattr(args, "stage2_ratio_selection_method", "correlation") or "correlation"
            ),
            "stage2_ratio_epsilon": float(
                getattr(args, "stage2_ratio_epsilon", 1e-6) or 1e-6
            ),
            "enable_maqc_pairing": getattr(args, "enable_maqc_pairing", False),
            "maqc_fs_method_sets": list(getattr(args, "maqc_fs_method_sets", []) or ()),
            "maqc_pairing_all_configs": getattr(args, "maqc_pairing_all_configs", False),
            "maqc_pairing_min_improvement": getattr(args, "maqc_pairing_min_improvement", 0.0),
            "maqc_pairing_min_improvement_se_mult": getattr(args, "maqc_pairing_min_improvement_se_mult", 0.0),
            "enable_nestedcv_audit": getattr(args, "enable_nestedcv_audit", False),
            "nestedcv_audit_all_configs": getattr(args, "nestedcv_audit_all_configs", False),
            "nestedcv_min_n_train": getattr(args, "nestedcv_min_n_train", 80),
            "nestedcv_outer_splits": getattr(args, "nestedcv_outer_splits", 3),
            "nestedcv_outer_repeats": getattr(args, "nestedcv_outer_repeats", 2),
            "nestedcv_min_train_per_class": getattr(args, "nestedcv_min_train_per_class", 2),
            "nestedcv_seed_stride": getattr(args, "nestedcv_seed_stride", 997),
            "nestedcv_ci_level": getattr(args, "nestedcv_ci_level", 0.95),
            "task_timeout_sec": args.task_timeout_sec,
            "progress_heartbeat_sec": float(getattr(args, "progress_heartbeat_sec", 60.0) or 60.0),
            "progress_watchdog_sec": float(getattr(args, "progress_watchdog_sec", 900.0) or 0.0),
            "progress_stall_watchdog_sec": float(
                getattr(args, "progress_stall_watchdog_sec", 1800.0) or 0.0
            ),
            "quiet_worker_logs": args.quiet_worker_logs,
            "enable_shadow_evaluator": bool(getattr(args, "enable_shadow_evaluator", False)),
            "shadow_frozen_datasets": list(getattr(args, "shadow_frozen_datasets", []) or ()),
        },
        "shadow_evaluator": dict(shadow_meta),
        "artifact_counts": {
            "model_bundles": int(len(all_model_bundles)),
            "run_diagnostics": int(len(all_run_diagnostics)),
        },
        "notes": (
            "SOTA comparisons are reported as protocol families "
            "(strict_holdout vs inflated literature references) with protocol-gap disclaimers."
        ),
        "outputs": {
            "runs": str(runs_path),
            "summary": str(summary_path),
            "sota_comparison": str(sota_path),
            "ablation_deltas": str(ablation_path),
            "failures": str(failures_path),
            "model_bundles": str(model_bundles_path),
            "run_diagnostics": str(run_diagnostics_path),
            "shadow_evaluator": str(shadow_path) if bool(shadow_meta.get("enabled", False)) else "",
        },
    }

    with failures_path.open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Run output directory: {run_dir}")
    print(f"Saved run-level results to: {runs_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved SOTA comparison to: {sota_path}")
    print(f"Saved ablation deltas to: {ablation_path}")
    print(f"Saved model bundles to: {model_bundles_path}")
    print(f"Saved run diagnostics to: {run_diagnostics_path}")
    print(f"Saved metadata to: {metadata_path}")

    # Emit compact per-run summary (opt-in via --emit-summary).
    if getattr(args, "emit_summary", False):
        from pathlib import Path as _Path
        summary_obj = _build_run_summary(
            rows=all_rows,
            failures=failures,
            metadata=metadata,
            run_dir=run_dir,
        )
        summary_dir_str = str(getattr(args, "summary_dir", "") or "").strip()
        if summary_dir_str:
            summary_out_dir = _Path(summary_dir_str)
            summary_out_dir.mkdir(parents=True, exist_ok=True)
        else:
            summary_out_dir = run_dir
        run_summary_path = summary_out_dir / "run_summary_v1.json"
        with run_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_obj, f, indent=2, default=str)
        print(f"Saved per-run summary to: {run_summary_path}")

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrated DF+FS benchmark with SOTA comparison and ablations")

    parser.add_argument("--dataset-sets", nargs="+", default=[], help=f"Dataset sets: {sorted(DATASET_SETS.keys())}")
    parser.add_argument("--datasets", nargs="*", default=[], help="Explicit dataset IDs to include")
    parser.add_argument("--exclude-datasets", nargs="*", default=[], help="Dataset IDs to exclude")
    parser.add_argument(
        "--extended",
        action="store_true",
        default=False,
        help="Include extended datasets (CuMiDa expansion + UCSC Xena/TCGA). "
             "Equivalent to adding 'extended' to --dataset-sets.",
    )

    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel workers across dataset/seed tasks.")
    parser.add_argument(
        "--inner-n-jobs",
        type=str,
        default="auto",
        help=(
            "Inner parallelism (per-feature DF, per-method FS, classifier candidates). "
            "'auto' sets n_jobs=cpu_count//2 when max_workers==1, else 1. "
            "Use an integer to override (e.g. '4'). Use '-1' for all cores."
        ),
    )
    parser.add_argument(
        "--task-timeout-sec",
        type=float,
        default=300.0,
        help="Per dataset/seed/config timeout in seconds (hard-kills the worker process; 0 disables timeout).",
    )
    parser.add_argument(
        "--quiet-worker-logs",
        action="store_true",
        help="Suppress stdout/stderr from worker tasks to keep long runs monitorable.",
    )
    parser.add_argument(
        "--progress-heartbeat-sec",
        type=float,
        default=60.0,
        help="Heartbeat cadence in seconds for benchmark progress logs.",
    )
    parser.add_argument(
        "--progress-watchdog-sec",
        type=float,
        default=900.0,
        help="Warn when a dataset/seed task runs longer than this threshold (0 disables).",
    )
    parser.add_argument(
        "--progress-stall-watchdog-sec",
        type=float,
        default=1800.0,
        help=(
            "Warn and emit process wait-state diagnostics when tasks are running but no task "
            "completes for this threshold (0 disables)."
        ),
    )
    parser.add_argument(
        "--enable-shadow-evaluator",
        action="store_true",
        help="Run shadow evaluator pilot on a frozen dataset subset and emit concordance artifact.",
    )
    parser.add_argument(
        "--shadow-frozen-datasets",
        nargs="*",
        default=[],
        help="Frozen dataset IDs used by the shadow evaluator pilot.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional absolute cap on the train split size (Artificial HDLSS). 0 disables.",
    )
    parser.add_argument(
        "--fs-fraction",
        type=float,
        default=0.0,
        help="Override FS train-subset fraction in (0,1]; <=0 keeps dataset-profile default.",
    )

    parser.add_argument(
        "--enable-nestedcv-audit",
        action="store_true",
        help="Emit a protocol-audit overlay row with repeated nested CV on the strict-holdout training split when n_train is small.",
    )
    parser.add_argument(
        "--nestedcv-audit-all-configs",
        action="store_true",
        help="Run nested-CV audit for every ablation config (default: baseline config only).",
    )
    parser.add_argument(
        "--nestedcv-min-n-train",
        type=int,
        default=80,
        help="Trigger nested-CV audit when the strict-holdout n_train is below this threshold.",
    )
    parser.add_argument(
        "--nestedcv-outer-splits",
        type=int,
        default=3,
        help="Outer CV splits for nested-CV audit (auto-reduced when class counts are small).",
    )
    parser.add_argument(
        "--nestedcv-outer-repeats",
        type=int,
        default=2,
        help="Outer CV repeats for nested-CV audit (different shuffle seeds).",
    )
    parser.add_argument(
        "--nestedcv-min-train-per-class",
        type=int,
        default=2,
        help="Minimum per-class samples required in each outer-training fold; otherwise audit is skipped.",
    )
    parser.add_argument("--nestedcv-seed-stride", type=int, default=997, help="Seed stride applied per outer repeat.")
    parser.add_argument("--nestedcv-ci-level", type=float, default=0.95, help="Confidence interval level for nested-CV audit metrics.")

    parser.add_argument("--compute-budget", type=str, default="standard", choices=["fast", "standard", "thorough"])
    parser.add_argument(
        "--dist-criterion",
        type=str,
        default="simple",
        choices=["simple", "cvm_p", "ks_p", "aic", "bic", "aicc", "cv", "cv_loglik", "crps", "mnpo_oracle"],
    )
    parser.add_argument(
        "--df-family-set",
        type=str,
        default="v6",
        choices=["v6", "extended", "flex"],
        help="Distribution-family candidate library (opt-in).",
    )
    parser.add_argument(
        "--df-compute-ad",
        action="store_true",
        help="Compute Anderson-Darling statistic for DF fits (opt-in).",
    )
    parser.add_argument(
        "--df-ad-bootstrap-samples",
        type=int,
        default=0,
        help="If >0 and --df-compute-ad, estimate AD p-values via parametric bootstrap (per family).",
    )
    parser.add_argument(
        "--df-compute-qq-pp",
        action="store_true",
        help="Compute lightweight Q-Q / P-P diagnostics for DF fits (opt-in).",
    )
    parser.add_argument(
        "--df-compute-dip",
        dest="df_compute_dip",
        action="store_true",
        help="Enable cheap multimodality diagnostics (dip_stat + mode_count) during DF auditing (default: enabled).",
    )
    parser.add_argument(
        "--no-df-compute-dip",
        dest="df_compute_dip",
        action="store_false",
        help="Disable multimodality diagnostics and multimodal fallback routing in DF.",
    )
    parser.set_defaults(df_compute_dip=True)
    parser.add_argument(
        "--df-dip-hist-bins",
        type=int,
        default=40,
        help="Histogram bins used for mode_count when multimodality diagnostics are enabled.",
    )
    parser.add_argument(
        "--df-multimodal-fallback",
        type=str,
        default="gmm",
        choices=["none", "gmm", "rank_transform"],
        help="Fallback transform for multimodal DF features (default: gmm).",
    )
    parser.add_argument(
        "--df-interval-likelihood",
        action="store_true",
        help="Use randomized-PIT interval likelihood for heaped/integer-like features (opt-in).",
    )
    parser.add_argument(
        "--df-interval-delta-override",
        type=float,
        default=0.0,
        help="Optional override for the heaping delta used by interval likelihood (0 disables override).",
    )
    parser.add_argument(
        "--df-lmoment-prescreen",
        action="store_true",
        help="Limit fitted distribution candidates using L-moment ratio heuristics (opt-in).",
    )
    parser.add_argument(
        "--df-lmoment-prescreen-max-candidates",
        type=int,
        default=12,
        help="Max candidate families to fit when --df-lmoment-prescreen is enabled (0 disables cap).",
    )
    parser.add_argument(
        "--df-estimator",
        type=str,
        default="mle",
        choices=["mle", "mps"],
        help="Parameter estimator for DF fitting (opt-in).",
    )
    parser.add_argument(
        "--df-mps-maxiter",
        type=int,
        default=250,
        help="Max iterations for MPS optimizer (used only when --df-estimator mps).",
    )
    parser.add_argument(
        "--df-mps-tol",
        type=float,
        default=1e-6,
        help="Optimizer tolerance for MPS (used only when --df-estimator mps).",
    )
    parser.add_argument(
        "--df-compute-crps",
        action="store_true",
        help="Compute MC CRPS diagnostic for DF fits (opt-in).",
    )
    parser.add_argument(
        "--df-crps-uq-decomposition",
        action="store_true",
        help="Compute CRPS-based aleatoric/epistemic uncertainty decomposition (opt-in).",
    )
    parser.add_argument(
        "--df-crps-mc-samples",
        type=int,
        default=96,
        help="Monte Carlo sample size per family for CRPS diagnostics.",
    )
    parser.add_argument(
        "--df-crps-data-subsample",
        type=int,
        default=256,
        help="Max n used from data per family for CRPS diagnostics.",
    )
    parser.add_argument(
        "--df-mnpo-include-crps",
        action="store_true",
        help="When --dist-criterion mnpo_oracle, include CRPS as an additional oracle (opt-in).",
    )
    parser.add_argument(
        "--df-mnpo-disable-tritrust",
        action="store_true",
        help="When --dist-criterion mnpo_oracle, disable TriTrust oracle weighting (opt-in).",
    )
    parser.add_argument(
        "--df-mnpo-include-preq",
        action="store_true",
        help="When --dist-criterion mnpo_oracle, include a cheap prequential/holdout predictive oracle (opt-in).",
    )
    parser.add_argument(
        "--df-preq-holdout-fraction",
        type=float,
        default=0.20,
        help="Holdout fraction used for the prequential predictive oracle (used only when --df-mnpo-include-preq is set).",
    )
    parser.add_argument(
        "--df-preq-min-train",
        type=int,
        default=20,
        help="Minimum train size for the prequential predictive oracle (used only when --df-mnpo-include-preq is set).",
    )
    parser.add_argument(
        "--df-preq-max-test-points",
        type=int,
        default=128,
        help="Max holdout points for the prequential predictive oracle (used only when --df-mnpo-include-preq is set).",
    )
    parser.add_argument(
        "--df-use-tail-risk-oracle",
        action="store_true",
        help="Deprecated no-op (T-DS3): DF tail-risk oracle was removed.",
    )
    parser.add_argument(
        "--df-tail-risk-alpha",
        type=float,
        default=0.33,
        help="Deprecated no-op (T-DS3).",
    )
    parser.add_argument(
        "--df-use-qre-smoothing",
        action="store_true",
        help="When --dist-criterion mnpo_oracle, enable QRE-style smoothing for scalar-oracle preferences (opt-in).",
    )
    parser.add_argument(
        "--df-qre-temperature-gamma",
        type=float,
        default=1.0,
        help="Gamma multiplier for DF QRE temperature (higher = smoother preferences).",
    )
    parser.add_argument(
        "--df-use-oracle-redundancy-penalty",
        action="store_true",
        help="When --dist-criterion mnpo_oracle, apply oracle-redundancy penalty to MNPO oracle weights (opt-in).",
    )
    parser.add_argument(
        "--df-compute-tremble-sensitivity",
        action="store_true",
        help="When --dist-criterion mnpo_oracle, compute tremble-sensitivity diagnostic (opt-in).",
    )
    parser.add_argument("--df-rejection-threshold", type=float, default=0.01)
    parser.add_argument("--df-confidence-margin", type=float, default=0.05)

    parser.add_argument("--max-dist-features", type=int, default=256)
    parser.add_argument(
        "--df-stage-position",
        type=str,
        default="after_fs",
        choices=["before_fs", "after_fs"],
        help="Apply DF before feature selection or, by default, only after FS selects the feature subset.",
    )
    parser.add_argument(
        "--enable-df-fastpath",
        action="store_true",
        help="Deprecated no-op (T-DS2): DF fast-path heuristic was removed.",
    )
    parser.add_argument(
        "--disable-df-fastpath",
        dest="enable_df_fastpath",
        action="store_false",
        help="Deprecated no-op (T-DS2).",
    )
    # Promotion (A6): revert fast-path default to disabled (Val-3 finding: fitting always wins).
    # Explicit --enable-df-fastpath required to opt-in.
    parser.set_defaults(enable_df_fastpath=False)
    parser.add_argument(
        "--df-fastpath-scope",
        type=str,
        default="none",
        choices=["all", "fs_only", "none"],
        help="Deprecated no-op (T-DS2).",
    )
    parser.add_argument(
        "--df-fastpath-trigger",
        type=str,
        default="small_n_or_low_unique",
        choices=[
            "small_n",
            "low_unique",
            "small_n_or_low_unique",
            "small_n_and_low_unique",
        ],
        help="Deprecated no-op (T-DS2).",
    )
    parser.add_argument("--df-fastpath-small-n-threshold", type=int, default=150, help="Deprecated no-op (T-DS2).")
    parser.add_argument(
        "--df-fastpath-unique-ratio-threshold",
        type=float,
        default=0.05,
        help="Deprecated no-op (T-DS2).",
    )
    parser.add_argument("--df-fastpath-n-unique-threshold", type=int, default=12, help="Deprecated no-op (T-DS2).")
    parser.add_argument("--cdf-min-gof-p", type=float, default=0.005)
    parser.add_argument("--cdf-max-confidence-set", type=int, default=8)
    parser.add_argument("--cdf-skip-heaped-features", action="store_true")
    parser.add_argument("--enable-cdf-block-gating-cv", action="store_true")
    parser.add_argument("--cdf-block-gating-n-blocks", type=int, default=4)
    parser.add_argument("--cdf-block-gating-min-block-size", type=int, default=8)
    parser.add_argument("--cdf-block-gating-cv-splits", type=int, default=2)
    parser.add_argument("--cdf-block-gating-max-blocks", type=int, default=6)
    parser.add_argument("--cdf-block-gating-time-budget-sec", type=float, default=12.0)
    parser.add_argument("--cdf-block-gating-min-improvement", type=float, default=0.0)
    parser.add_argument("--low-gof-threshold", type=float, default=0.01)
    parser.add_argument("--low-gof-weight", type=float, default=0.5)
    parser.add_argument("--stability-bootstrap", type=int, default=3)
    parser.add_argument("--prefilter-top-k", type=int, default=600)
    parser.add_argument(
        "--prefilter-adaptive-top-k",
        action="store_true",
        help="Scale prefilter top-k from expanded meta-feature complexity (T-R-405).",
    )
    parser.add_argument(
        "--prefilter-adaptive-top-k-scaling",
        type=float,
        default=0.5,
        help="Scaling factor used by adaptive prefilter top-k when enabled.",
    )
    parser.add_argument(
        "--prefilter-mi-weight",
        type=float,
        default=0.60,
        help="Tier-1 prefilter blend weight for mutual information (T-003).",
    )
    parser.add_argument(
        "--prefilter-f-weight",
        type=float,
        default=0.40,
        help="Tier-1 prefilter blend weight for ANOVA F-test (T-003).",
    )
    parser.add_argument(
        "--prefilter-union-enabled",
        action="store_true",
        help="Enable multi-strategy prefilter union (T-R-127; opt-in).",
    )
    parser.add_argument(
        "--prefilter-strategies",
        type=str,
        default="mi_ftest_blend",
        help=(
            "Comma/space-separated prefilter strategies used when union is enabled "
            "(supported: mi_ftest_blend, rf_importance, relieff_scores, wsnr, bh_fdr)."
        ),
    )
    parser.add_argument(
        "--prefilter-nondefault-budget-fraction",
        type=float,
        default=0.10,
        help="Per non-default strategy budget fraction under prefilter union (T-R-127).",
    )
    parser.add_argument(
        "--prefilter-wsnr-enabled",
        action="store_true",
        help="Enable WSNR binary prefilter strategy (T-R-172). Auto-enables prefilter union.",
    )
    # T-R-265: BH-adjusted t-test/ANOVA prefilter flag.
    parser.add_argument(
        "--prefilter-bh-ttest-enabled",
        dest="prefilter_bh_ttest_enabled",
        action="store_true",
        default=True,
        help="Enable BH-adjusted parametric filter in prefilter union (T-R-265; default=on).",
    )
    parser.add_argument(
        "--no-prefilter-bh-ttest",
        dest="prefilter_bh_ttest_enabled",
        action="store_false",
        help="Disable BH-adjusted parametric filter in prefilter union.",
    )
    parser.add_argument(
        "--prefilter-bh-ttest-alpha",
        "--bh-prefilter-alpha",
        dest="prefilter_bh_ttest_alpha",
        type=float,
        default=0.05,
        help="Alpha used by the BH-adjusted parametric prefilter (default: 0.05).",
    )
    # T-R-272: variance floor prefilter flags.
    parser.add_argument(
        "--prefilter-variance-floor-enabled",
        dest="prefilter_variance_floor_enabled",
        action="store_true",
        default=True,
        help="Enable near-constant feature removal before FS pipeline (T-R-272; default=on).",
    )
    parser.add_argument(
        "--no-prefilter-variance-floor",
        dest="prefilter_variance_floor_enabled",
        action="store_false",
        help="Disable variance floor prefilter.",
    )
    parser.add_argument(
        "--prefilter-variance-floor-threshold",
        dest="prefilter_variance_floor_threshold",
        type=float,
        default=1e-6,
        help="Variance threshold for near-constant removal (T-R-272; default 1e-6).",
    )
    parser.add_argument(
        "--prefilter-variance-floor-mode-freq",
        dest="prefilter_variance_floor_mode_freq",
        type=float,
        default=0.99,
        help="Mode frequency threshold for near-constant removal (T-R-272; default 0.99).",
    )
    parser.add_argument(
        "--disable-prefilter-rnaseq-transform",
        action="store_true",
        help="Disable RNA-seq log2(CPM+1)+TMM stabilization in prefilter scoring.",
    )
    parser.add_argument(
        "--force-prefilter-rnaseq-transform",
        action="store_true",
        help="Force RNA-seq transform regardless of auto-detection.",
    )
    parser.add_argument(
        "--enable-prefilter-rnaseq-nb-lrt",
        action="store_true",
        help="Enable RNA-seq Negative-Binomial LRT prefilter signal (domain-gated).",
    )
    parser.add_argument(
        "--prefilter-rnaseq-nb-lrt-alpha",
        type=float,
        default=0.10,
        help="BH-FDR alpha for RNA-seq NB-LRT prefilter signal.",
    )
    parser.add_argument(
        "--batch-correction",
        type=str,
        default="none",
        choices=[
            "none",
            "combat",
            "combat_seq",
            "combat-seq",
            "combatseq",
            "cdf_center",
            "center_scale",
            "center-scale",
            "centerscale",
        ],
        help="Optional train-fold-only batch correction mode (T-R-145/T-R-149).",
    )
    parser.add_argument(
        "--batch-correction-combat-prior-strength",
        type=float,
        default=8.0,
        help="Empirical-Bayes shrinkage prior strength for ComBat mode.",
    )
    parser.add_argument(
        "--batch-correction-cdf-n-quantiles",
        type=int,
        default=33,
        help="Number of quantiles used by cdf_center research mapping.",
    )
    parser.add_argument(
        "--batch-correction-cdf-clip-low",
        type=float,
        default=0.01,
        help="Lower quantile clip for cdf_center mapping.",
    )
    parser.add_argument(
        "--batch-correction-cdf-clip-high",
        type=float,
        default=0.99,
        help="Upper quantile clip for cdf_center mapping.",
    )
    parser.add_argument(
        "--batch-label-policy",
        type=str,
        default="none",
        choices=["none", "source", "kmeans2"],
        help=(
            "Batch-label source for train-fold batch correction: "
            "none (no labels), source (dataset-provided labels when available), "
            "kmeans2 (derive pseudo-batches via 2-means clustering)."
        ),
    )
    parser.add_argument(
        "--multiomics-adapter",
        type=str,
        default="none",
        choices=[
            "none",
            "split_halves",
            "metadata_blocks",
            "metadata",
            "metadata_block",
            "feature_metadata",
            "diablo_blocks",
        ],
        help=(
            "Optional benchmark-time multi-omics adapter that derives feature blocks "
            "from the same dataset before FS/classification."
        ),
    )
    parser.add_argument(
        "--multiomics-integrator",
        type=str,
        default="mb_plsda",
        choices=["mb_plsda", "mint"],
        help="Integrator used when --multiomics-adapter is enabled.",
    )
    parser.add_argument(
        "--multiomics-n-components",
        type=int,
        default=2,
        help="Latent component count for multi-omics integrators.",
    )
    parser.add_argument(
        "--meta-learning-selector",
        type=str,
        default="none",
        choices=["none", "decision_tree", "logistic"],
        help=(
            "Optional runtime profile selector trained on the records payload at "
            "--meta-learning-records-path (legacy Val-14/15 by default; Val-21 composite payload later)."
        ),
    )
    parser.add_argument(
        "--meta-learning-confidence-threshold",
        type=float,
        default=0.55,
        help="Fallback to v16_ref when meta-learning confidence is below this threshold.",
    )
    parser.add_argument(
        "--meta-learning-records-path",
        type=str,
        default="",
        help=(
            "Optional records payload for runtime meta-learning. "
            "Use this to point Val-21 placeholder profiles at the post-Val-20 composite selector artifact."
        ),
    )
    parser.add_argument(
        "--tier-classifier-mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "learned"],
        help="Tier classifier used for meta-feature tier inference.",
    )
    parser.add_argument(
        "--tier-classifier-model-path",
        type=str,
        default="",
        help=(
            "Optional learned tier-classifier artifact path. "
            "Leave empty to use the default bundled model path once that artifact exists."
        ),
    )
    parser.add_argument(
        "--screening-enabled",
        action="store_true",
        help="Enable Tier-2 interaction-aware screening between prefilter and methods (T-004; opt-in).",
    )
    parser.add_argument(
        "--screening-method",
        type=str,
        default="none",
        choices=["none", "stir", "evalue"],
        help="Tier-2 screening method (used only when --screening-enabled is set).",
    )
    parser.add_argument(
        "--screening-pool-cap",
        type=int,
        default=2000,
        help="Max feature count screened by Tier-2 screening (safety cap).",
    )
    parser.add_argument("--screening-stir-n-neighbors", type=int, default=10)
    parser.add_argument("--screening-stir-n-iter", type=int, default=50)
    parser.add_argument("--screening-stir-keep-fraction", type=float, default=0.5)
    parser.add_argument("--screening-stir-min-features", type=int, default=20)
    parser.add_argument("--screening-evalue-alpha", type=float, default=0.20)
    parser.add_argument("--screening-evalue-min-features", type=int, default=20)
    parser.add_argument(
        "--eval-models-enabled",
        action="store_true",
        help="Enable multi-classifier evaluation proxy during fold scoring (T-001; opt-in).",
    )
    parser.add_argument(
        "--eval-models",
        type=str,
        default="lr_l2,linear_svc,rf_small",
        help="Comma/space-separated model keys used by the evaluation proxy (T-001).",
    )
    parser.add_argument(
        "--eval-aggregate",
        type=str,
        default="mean",
        choices=["mean", "min", "cvar"],
        help="Aggregation for multi-model fold scores (T-001).",
    )
    parser.add_argument(
        "--eval-cvar-alpha",
        type=float,
        default=0.33,
        help="CVaR alpha for --eval-aggregate cvar (T-001).",
    )
    parser.add_argument(
        "--tier-lockout-enabled",
        action="store_true",
        help="Enable easy-tier lockout policy (T-P3-002, opt-in).",
    )
    parser.add_argument(
        "--tier-lockout-tier",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard", "very_hard"],
        help="Tier where lockout activates and routes to fallback methods.",
    )
    parser.add_argument(
        "--tier-lockout-difficulty-source",
        type=str,
        default="historical",
        choices=["historical", "meta_features"],
        help="Tier source used by lockout policy.",
    )
    parser.add_argument(
        "--tier-lockout-fallback-methods",
        type=str,
        default="",
        help="Fallback method stack for lockout; accepts FS preset name or comma/space-separated methods.",
    )
    parser.add_argument(
        "--tier-lockout-fallback-fs-method-set",
        type=str,
        default="",
        help="Fallback FS method-set name for lockout (used when --tier-lockout-fallback-methods is unset).",
    )
    parser.add_argument(
        "--tier-routing-enabled",
        action="store_true",
        help="Enable tier-conditional FS method routing (T-P3-005, opt-in).",
    )
    parser.add_argument(
        "--tier-routing-difficulty-classifier",
        type=str,
        default="meta_features",
        choices=["historical", "meta_features"],
        help="Tier source used by routing policy.",
    )
    parser.add_argument(
        "--tier-routing-table",
        type=str,
        default="",
        help=(
            "Tier routing table as JSON dict or ';'-separated pairs, e.g. "
            "'easy=strict_plus_mrmr;hard=mnpo_dove_sparse_multinomial_extended'."
        ),
    )
    parser.add_argument(
        "--regime-gating-enabled",
        action="store_true",
        help="Enable Val-12 regime-conditional FS gating (T-R-257/T-R-258/T-R-259).",
    )
    parser.add_argument(
        "--regime-gating-difficulty-source",
        type=str,
        default="historical",
        choices=["historical", "meta_features"],
        help="Tier source used by regime gate.",
    )
    parser.add_argument(
        "--regime-gating-target-tier",
        type=str,
        default="very_hard",
        choices=["easy", "medium", "hard", "very_hard"],
        help="Tier considered high-risk by regime gate.",
    )
    parser.add_argument(
        "--regime-gating-min-samples-per-class",
        type=float,
        default=7.0,
        help="Very-hard safeguard trigger: apply fallback when n/classes < threshold.",
    )
    parser.add_argument(
        "--regime-gating-use-expanded-features",
        action="store_true",
        help="Add Fisher F1 / N1 borderline checks to regime gating (T-R-404).",
    )
    parser.add_argument(
        "--regime-gating-min-fisher-f1",
        type=float,
        default=0.10,
        help="Expanded-feature regime gate threshold: fallback when Fisher F1 is below this value.",
    )
    parser.add_argument(
        "--regime-gating-max-n1-borderline",
        type=float,
        default=0.40,
        help="Expanded-feature regime gate threshold: fallback when N1 borderline is above this value.",
    )
    parser.add_argument(
        "--regime-gating-low-p-over-n-threshold",
        type=float,
        default=0.0,
        help=(
            "Low p/n trigger: bypass FS when p/n < threshold. "
            "Default is disabled (0.0); set >0 to re-enable."
        ),
    )
    parser.add_argument(
        "--regime-gating-simple-methods",
        type=str,
        default="",
        help="Simple fallback method stack (preset name or comma/space-separated list).",
    )
    parser.add_argument(
        "--regime-gating-simple-fs-method-set",
        type=str,
        default="strict_plus_mrmr",
        help="Fallback FS method-set used when --regime-gating-simple-methods is unset.",
    )
    parser.add_argument(
        "--regime-gating-very-hard-portfolio-max-methods",
        type=int,
        default=4,
        help="Very-hard safeguard cap for adaptive FS portfolio max methods.",
    )
    parser.add_argument(
        "--regime-gating-very-hard-copula-derandomize-runs",
        type=int,
        default=5,
        help="Very-hard safeguard override for copula derandomization runs.",
    )
    parser.add_argument(
        "--regime-gating-low-p-over-n-mode",
        type=str,
        default="fast_univariate_filter",
        choices=["all_features", "fast_univariate_filter"],
        help="Low p/n fallback behavior. 'fast_univariate_filter' (Val-13 default) applies "
             "ANOVA F-test and keeps top min(200, p/2) features. 'all_features' keeps all features.",
    )
    parser.add_argument(
        "--regime-gating-low-p-over-n-filter-max-k",
        type=int,
        default=200,
        help="Max features to keep when fast_univariate_filter mode is active (T-R-269).",
    )
    parser.add_argument(
        "--regime-gating-very-hard-min-classes",
        type=int,
        default=5,
        help="Gate 1 class-count qualifier: only trigger very-hard fallback when c >= this (T-R-270).",
    )
    # T-R-268: extreme multiclass gate CLI flags.
    parser.add_argument(
        "--regime-gating-extreme-multiclass-enabled",
        action="store_true",
        default=True,
        help="Enable Gate 3: extreme multiclass classifier recovery (T-R-268).",
    )
    parser.add_argument(
        "--no-regime-gating-extreme-multiclass",
        action="store_false",
        dest="regime_gating_extreme_multiclass_enabled",
        help="Disable Gate 3: extreme multiclass classifier recovery.",
    )
    parser.add_argument(
        "--regime-gating-extreme-multiclass-threshold",
        type=int,
        default=8,
        help="Gate 3 fires when class_count >= this threshold (T-R-268).",
    )
    parser.add_argument(
        "--regime-gating-extreme-multiclass-min-samples-per-class",
        type=float,
        default=11.0,
        help="Gate 3 guard: samples per class must be >= this for Gate 3 to fire (T-R-268).",
    )
    parser.add_argument(
        "--mnpo-performance-oracle-mode",
        type=str,
        default="single",
        choices=["single", "multi_model_oracles"],
        help="MNPO performance oracle construction mode (T-002; requires --eval-models-enabled for per-model oracles).",
    )
    parser.add_argument(
        "--folding-method",
        type=str,
        default="pls_da",
        choices=["none", "rff", "tensor_sketch", "pls_da"],
        help="A24/A23 folding stage applied after rank prefilter (default: pls_da with class-count guard).",
    )
    parser.add_argument(
        "--folding-n-components",
        type=int,
        default=512,
        help="Output dimensionality for A24 folding stage.",
    )
    parser.add_argument(
        "--folding-rff-gamma",
        type=float,
        default=None,
        help="RBF gamma for A24 RFF folding (used only when --folding-method rff). If unset, uses 1/n_features.",
    )
    parser.add_argument(
        "--folding-pls-components",
        type=int,
        default=32,
        help="A23 PLS-DA folding components (used only when --folding-method pls_da).",
    )
    parser.add_argument(
        "--disable-folding-pls-scale",
        action="store_true",
        help="Disable internal scaling in A23 PLS-DA folding.",
    )
    parser.add_argument(
        "--folding-pls-min-classes",
        type=int,
        default=5,
        help="Val-3 guardrail: minimum number of classes required to enable PLS-DA folding. "
             "Regressions observed at C<5 (e.g. -0.08 on TOX at C=4). Default: 5.",
    )
    parser.add_argument(
        "--folding-pls-min-n-per-class",
        type=int,
        default=3,
        help="Minimum samples per class required for PLS-DA folding.",
    )
    parser.add_argument(
        "--folding-pls-max-imbalance-ratio",
        type=float,
        default=6.0,
        help="Maximum class-count imbalance ratio allowed for PLS-DA folding.",
    )
    parser.add_argument(
        "--folding-prefilter-k",
        type=int,
        default=0,
        help="Optional rank-prefilter cap applied when folding is enabled (0 keeps --prefilter-top-k behavior).",
    )
    parser.add_argument(
        "--enable-face-domain-projection",
        action="store_true",
        help="Enable A21 Fisherfaces-style PCA->LDA projection on catalog-tagged face-domain datasets.",
    )
    parser.add_argument(
        "--enable-ratio-features",
        action="store_true",
        help="Enable RP-1 log-ratio feature generation stage (opt-in).",
    )
    parser.add_argument(
        "--ratio-pool-size",
        type=int,
        default=80,
        help="Max feature count in the screened pool used for ratio pairing (RP-1).",
    )
    parser.add_argument(
        "--ratio-selection-method",
        type=str,
        default="ktsp",
        choices=["ktsp", "correlation"],
        help="Pair selection heuristic used by the RP-1 ratio stage.",
    )
    parser.add_argument(
        "--ratio-max-pairs",
        type=int,
        default=12000,
        help="Hard cap on the number of candidate feature pairs scored during ratio pairing.",
    )
    parser.add_argument(
        "--max-ratio-features",
        type=int,
        default=30,
        help="Maximum number of ratio/log-ratio features appended by RP-1.",
    )
    parser.add_argument(
        "--ratio-epsilon",
        type=float,
        default=1e-6,
        help="Numerical stability constant added to numerator/denominator in log-ratio features.",
    )
    parser.add_argument(
        "--ratio-abs-value",
        action="store_true",
        help="Apply abs() to log-ratio features after construction (sign-invariant).",
    )
    parser.add_argument(
        "--ratio-allow-nonpositive",
        action="store_true",
        help="Allow ratio construction from non-positive features (may produce unstable ratios).",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        default="mnpo_portfolio",
        choices=["mnpo_portfolio", "legacy_voting"],
        help="FS aggregation regime used inside FeatureSelector.",
    )
    parser.add_argument(
        "--fs-portfolio-size",
        type=int,
        default=6,
        help="MNPO portfolio size (number of selector candidates participating in portfolio weighting).",
    )
    parser.add_argument(
        "--fs-portfolio-size-guard",
        type=str,
        default="none",
        choices=["none", "warn", "raise"],
        help="Guard against fs_portfolio_size < enabled method count (helps catch 'no-effect' additions).",
    )
    parser.add_argument(
        "--enable-fs-adaptive-portfolio-sizing",
        action="store_true",
        help="Enable bounded adaptive portfolio sizing (T-P3-009).",
    )
    parser.add_argument(
        "--fs-adaptive-size-min",
        type=int,
        default=None,
        help="Lower bound for adaptive portfolio size (used only when adaptive sizing is enabled).",
    )
    parser.add_argument(
        "--fs-adaptive-size-max",
        type=int,
        default=None,
        help="Upper bound for adaptive portfolio size (used only when adaptive sizing is enabled).",
    )
    parser.add_argument(
        "--adaptive-sizing-variance-penalty",
        "--fs-adaptive-sizing-variance-penalty",
        dest="fs_adaptive_sizing_variance_penalty",
        action="store_true",
        help="Enable variance penalty in adaptive MNPO portfolio sizing (Val-7 candidate toggle).",
    )
    parser.add_argument(
        "--adaptive-sizing-variance-penalty-strength",
        "--fs-adaptive-sizing-variance-penalty-strength",
        dest="fs_adaptive_sizing_variance_penalty_strength",
        type=float,
        default=0.5,
        help="Strength of adaptive portfolio-size variance penalty (Val-7 candidate toggle).",
    )
    # T-R-266: Pareto-front portfolio sizing CLI flag.
    parser.add_argument(
        "--enable-fs-pareto-portfolio-sizing",
        "--fs-pareto-portfolio-sizing-enabled",
        dest="fs_pareto_portfolio_sizing_enabled",
        action="store_true",
        help="Enable Pareto-front adaptive portfolio sizing (T-R-266; opt-in, replaces fixed min/max bounds).",
    )
    # T-R-271: stability-weighted portfolio aggregation CLI flag.
    parser.add_argument(
        "--fs-stability-weighted-aggregation-enabled",
        dest="fs_stability_weighted_aggregation_enabled",
        action="store_true",
        help="Enable stability-weighted portfolio aggregation (T-R-271; Kuncheva stability × Banzhaf weight).",
    )
    # VAL12_Suggestions: Post-FS feature count safety cap.
    parser.add_argument(
        "--fs-max-selected-features-ratio",
        type=float,
        default=None,
        help="Post-FS safety cap: max selected features = n_train * ratio (default: 0.5).",
    )
    parser.add_argument(
        "--fs-max-selected-features-cap",
        type=int,
        default=None,
        help="Post-FS safety cap: absolute max selected features (default: 500).",
    )
    parser.add_argument(
        "--enable-fs-mnpo-paradigm-aware-prior",
        action="store_true",
        help="Enable paradigm-aware MNPO reference prior with interaction floor (T-R-128; opt-in).",
    )
    parser.add_argument(
        "--fs-mnpo-interaction-floor",
        type=float,
        default=0.12,
        help="Minimum reference-prior mass reserved for interaction-capable methods (T-R-128).",
    )
    parser.add_argument(
        "--enable-fs-rashomon",
        action="store_true",
        help="Enable post-selection Rashomon importance bounds (T-P3-011).",
    )
    parser.add_argument(
        "--fs-rashomon-max-models",
        type=int,
        default=12,
        help="Maximum candidate models used to estimate Rashomon importance bounds.",
    )
    parser.add_argument(
        "--fs-rashomon-score-tolerance",
        type=float,
        default=0.01,
        help="Score tolerance from best model to include in Rashomon set.",
    )
    parser.add_argument(
        "--fs-mnpo-consensus-exclude-methods",
        nargs="*",
        default=(),
        help="Optional method names to exclude from the MNPO candidate library when synthetic consensus candidates "
        "are present (helps avoid double-counting in portfolio-size experiments).",
    )
    parser.add_argument(
        "--fs-mnpo-consensus-exclude-protect-top-k",
        type=int,
        default=0,
        help="When >0, protect the top-k candidates (by MNPO equilibrium weight) from consensus-exclusion filtering.",
    )
    parser.add_argument(
        "--disable-fs-mnpo-legacy-consensus",
        action="store_true",
        help="Disable the synthetic `legacy_consensus` candidate in the MNPO candidate library (A8.1 diagnostic).",
    )
    parser.add_argument(
        "--disable-fs-mnpo-majority-consensus",
        action="store_true",
        help="Disable the synthetic `majority_consensus` candidate in the MNPO candidate library (A8.1 diagnostic).",
    )
    parser.add_argument("--enable-balanced-fs-subsample", action="store_true")
    parser.add_argument("--fs-min-per-class", type=int, default=2)
    parser.add_argument(
        "--fs-method-timeout-sec",
        type=float,
        default=0.0,
        help="Per feature-selector method timeout in seconds (0 disables).",
    )
    parser.add_argument(
        "--fs-linear-svm-max-iter",
        type=int,
        default=10000,
        help="LinearSVC max_iter for feature-selector methods (liblinear convergence guard).",
    )
    parser.add_argument(
        "--fs-inner-cv-splits",
        type=int,
        default=3,
        help="Inner-CV split count used by the benchmark/runtime FS path (default benchmark setting: 3).",
    )
    parser.add_argument(
        "--fs-inner-cv-repeats",
        type=int,
        default=1,
        help="Inner-CV repeat count used by the benchmark/runtime FS path (default benchmark setting: 1).",
    )
    parser.add_argument(
        "--enable-fs-runtime-racing",
        action="store_true",
        help="Enable OP1.2 runtime-aware candidate racing before full MNPO candidate evaluation.",
    )
    parser.add_argument(
        "--fs-runtime-racing-proxy-splits",
        type=int,
        default=1,
        help="Number of inner-CV splits used for runtime-racing proxy scoring.",
    )
    parser.add_argument(
        "--fs-runtime-racing-keep-fraction",
        type=float,
        default=0.60,
        help="Fraction of candidates retained after runtime-racing proxy scoring.",
    )
    parser.add_argument(
        "--fs-runtime-racing-min-candidates",
        type=int,
        default=4,
        help="Minimum candidate count retained by runtime-racing.",
    )
    parser.add_argument(
        "--fs-runtime-racing-runtime-weight",
        type=float,
        default=0.15,
        help="Runtime penalty weight in runtime-racing proxy objective.",
    )
    parser.add_argument(
        "--fs-runtime-racing-mode",
        type=str,
        default="single_stage",
        choices=["single_stage", "successive_halving"],
        help="Racing policy: legacy single-stage or OP1.3 successive-halving elimination.",
    )
    parser.add_argument(
        "--fs-runtime-racing-stages",
        type=int,
        default=2,
        help="Number of elimination stages for OP1.3 successive-halving racing.",
    )
    parser.add_argument(
        "--fs-runtime-racing-confidence-bound",
        type=str,
        default="none",
        choices=["none", "hoeffding", "bernstein"],
        help="Confidence bound used for OP1.3 racing eliminations.",
    )
    parser.add_argument(
        "--fs-runtime-racing-delta",
        type=float,
        default=0.10,
        help="Tail probability (delta) used in OP1.3 confidence bounds.",
    )
    parser.add_argument(
        "--fs-diversity-oracle-mode",
        type=str,
        default="legacy_jaccard",
        choices=["legacy_jaccard", "mi_redundancy", "pid_mi", "complementarity"],
    )
    parser.add_argument("--fs-diversity-redundancy-weight", type=float, default=0.6)
    parser.add_argument("--fs-diversity-complementarity-weight", type=float, default=0.35)
    parser.add_argument("--fs-performance-balanced-weight", type=float, default=0.6)
    parser.add_argument("--fs-performance-macro-f1-weight", type=float, default=0.4)
    parser.add_argument("--enable-fs-adaptive-imbalance-score", action="store_true")
    parser.add_argument("--fs-imbalance-ratio-trigger", type=float, default=1.75)
    parser.add_argument("--fs-imbalance-min-classes", type=int, default=3)
    parser.add_argument(
        "--fs-use-tail-risk-oracle",
        action="store_true",
        help="Deprecated no-op (T-DS3): FS tail-risk oracle was removed.",
    )
    parser.add_argument(
        "--fs-tail-risk-alpha",
        type=float,
        default=0.33,
        help="Deprecated no-op (T-DS3).",
    )
    parser.add_argument(
        "--fs-use-regret-oracle",
        action="store_true",
        help="Deprecated no-op (T-DS3): FS regret oracle was removed.",
    )
    parser.add_argument(
        "--fs-use-qre-smoothing",
        action="store_true",
        help="Enable QRE-style smoothing for scalar-oracle pairwise preferences in FS MNPO (opt-in).",
    )
    parser.add_argument(
        "--fs-use-cvar-oracle",
        action="store_true",
        help="Enable CVaR oracle over fold scores (T-R-181).",
    )
    parser.add_argument(
        "--fs-cvar-alpha",
        type=float,
        default=0.33,
        help="Tail mass alpha for the CVaR oracle (T-R-181).",
    )
    parser.add_argument(
        "--fs-qre-temperature-gamma",
        type=float,
        default=1.0,
        help="Gamma multiplier for FS QRE temperature (higher = smoother preferences).",
    )
    parser.add_argument(
        "--fs-use-oracle-redundancy-penalty",
        action="store_true",
        help="Enable oracle-redundancy penalty for FS MNPO oracle weights (opt-in).",
    )
    parser.add_argument(
        "--fs-compute-tremble-sensitivity",
        action="store_true",
        help="Compute tremble-sensitivity diagnostic for FS MNPO equilibrium (opt-in).",
    )
    parser.add_argument(
        "--fs-oracle-weighting-mode",
        type=str,
        default="tritrust",
        choices=["tritrust", "uniform", "shapley", "banzhaf"],
        help="Oracle weighting mode for MNPO payoff aggregation (T-R-184, T-R-251).",
    )
    parser.add_argument(
        "--fs-shapley-n-coalitions-max",
        type=int,
        default=4096,
        help="Maximum coalition count for exact Shapley oracle weights (T-R-184).",
    )
    parser.add_argument(
        "--shapley-bayesian-shrinkage",
        "--fs-shapley-bayesian-shrinkage",
        dest="fs_shapley_bayesian_shrinkage",
        action="store_true",
        help="Enable Bayesian shrinkage on Shapley oracle weights (Val-7 candidate toggle).",
    )
    parser.add_argument(
        "--shapley-bayesian-prior-strength",
        "--fs-shapley-bayesian-prior-strength",
        dest="fs_shapley_bayesian_prior_strength",
        type=float,
        default=8.0,
        help="Prior strength used by Shapley Bayesian shrinkage.",
    )
    parser.add_argument(
        "--fs-use-interaction-oracle",
        action="store_true",
        help="Enable interaction-density oracle in FS MNPO (T-R-142, opt-in).",
    )
    parser.add_argument(
        "--fs-interaction-oracle-min-n-train",
        type=int,
        default=150,
        help="Minimum n_train gate for interaction oracle activation.",
    )
    parser.add_argument(
        "--fs-interaction-oracle-pool-size-cap",
        type=int,
        default=64,
        help="Maximum per-candidate feature pool size used by interaction oracle.",
    )
    parser.add_argument(
        "--fs-interaction-oracle-pair-cap",
        type=int,
        default=20000,
        help="Maximum pair evaluations per candidate in interaction oracle.",
    )
    parser.add_argument(
        "--fs-use-ubayfs-oracle",
        action="store_true",
        help="Enable UBayFS Bayesian ensemble oracle (T-R-186).",
    )
    parser.add_argument(
        "--fs-ubayfs-n-bootstrap",
        type=int,
        default=32,
        help="Bootstrap rounds for UBayFS oracle posterior estimation.",
    )
    parser.add_argument(
        "--fs-ubayfs-min-n",
        type=int,
        default=100,
        help="Minimum sample size gate for UBayFS oracle activation.",
    )
    parser.add_argument(
        "--fs-ubayfs-prior-weight",
        type=float,
        default=0.0,
        help="Prior weight for UBayFS posterior blending (0=uninformative empirical only).",
    )
    parser.add_argument(
        "--fs-use-conformal-uq",
        action="store_true",
        help="Enable conformal-reliability oracle in FS MNPO (opt-in).",
    )
    parser.add_argument(
        "--fs-conformal-uq-alpha",
        type=float,
        default=0.10,
        help="Conformal alpha for FS conformal-reliability oracle.",
    )
    parser.add_argument(
        "--fs-conformal-uq-min-folds",
        type=int,
        default=5,
        help="Minimum successful folds required to apply FS conformal-reliability oracle.",
    )
    parser.add_argument(
        "--fs-fold-preference-mode",
        type=str,
        default="vote",
        choices=["vote", "logistic"],
        help="Pairwise fold comparison mode for repeated-CV FS oracles.",
    )
    parser.add_argument(
        "--fs-use-conformal-efficiency",
        action="store_true",
        help="Enable conformal-efficiency singleton-rate oracle in FS MNPO.",
    )
    parser.add_argument(
        "--fs-conformal-efficiency-method",
        type=str,
        default="split",
        choices=["split", "aps"],
        help="Conformal backend used by the FS efficiency oracle.",
    )
    parser.add_argument(
        "--fs-oracle-weight-js-shrinkage",
        action="store_true",
        help="Apply James-Stein shrinkage to Banzhaf FS oracle weights.",
    )
    parser.add_argument(
        "--fs-oracle-complexity-conditioning",
        action="store_true",
        help="Condition FS oracle weights on dataset complexity profile (T-R-406).",
    )
    parser.add_argument(
        "--fs-payoff-shrinkage-kappa",
        type=float,
        default=0.0,
        help="Shrink aggregated FS payoff matrices toward zero by variance-adaptive strength.",
    )
    parser.add_argument(
        "--fs-rank-aggregation-mode",
        type=str,
        default="none",
        choices=["none", "borda", "rra"],
    )
    parser.add_argument("--enable-fs-wrapper-refine", action="store_true")
    parser.add_argument("--fs-wrapper-refine-top-k", type=int, default=24)
    parser.add_argument("--fs-wrapper-refine-max-add", type=int, default=12)
    parser.add_argument("--fs-wrapper-refine-min-gain", type=float, default=1e-4)
    parser.add_argument("--fs-ova-negative-ratio", type=float, default=2.0)
    parser.add_argument(
        "--fs-ova-min-classes",
        type=int,
        default=5,
        help="Minimum number of classes required to run the OVA candidate generator (multiclass-only gating).",
    )
    parser.add_argument(
        "--fs-ova-min-pos-samples",
        type=int,
        default=2,
        help="Minimum number of positive samples required to include a class in OVA aggregation.",
    )
    parser.add_argument(
        "--fs-ova-class-weight-mode",
        type=str,
        default="uniform",
        choices=["uniform", "sqrt_pos", "pos", "log_pos", "inv_pos", "inv_sqrt_pos", "inv_log_pos"],
        help="Optional OVA aggregation weighting (opt-in).",
    )
    parser.add_argument(
        "--fs-ova-aggregation-mode",
        type=str,
        default="mean",
        choices=["mean", "p_norm"],
        help="OVA class-score aggregation across classes (opt-in).",
    )
    parser.add_argument(
        "--fs-ova-aggregation-p",
        type=float,
        default=4.0,
        help="p for OVA 'p_norm' aggregation (p>1 emphasizes class-specific peaks vs mean).",
    )
    parser.add_argument(
        "--fs-ova-linear-backend",
        type=str,
        default="linear_svm_l1",
        choices=["linear_svm_l1", "elastic_net_lr"],
        help="Linear model used inside the OVA selector for class-specific scoring.",
    )
    parser.add_argument(
        "--enable-fs-ova-calibration",
        action="store_true",
        help="Enable A29 probabilistic calibration weighting in OVA class-specific scoring.",
    )
    parser.add_argument(
        "--fs-ova-calibration-cv",
        type=int,
        default=3,
        help="Cross-validation folds used by OVA calibration (A29).",
    )
    parser.add_argument(
        "--fs-ecoc-min-classes",
        type=int,
        default=4,
        help="Minimum class count required to run ECOC class-aware selector.",
    )
    parser.add_argument(
        "--fs-ecoc-max-ovo-pairs",
        type=int,
        default=8,
        help="Maximum number of confusable class-pair (OVO) ECOC tasks.",
    )
    parser.add_argument(
        "--fs-ecoc-random-code-bits",
        type=int,
        default=4,
        help="Number of random ECOC dichotomy tasks.",
    )
    parser.add_argument(
        "--fs-ecoc-class-complexity-weight",
        type=float,
        default=1.0,
        help="Class-complexity weighting strength for ECOC task aggregation.",
    )
    parser.add_argument(
        "--disable-fs-ecoc-include-ova-tasks",
        action="store_true",
        help="Disable OVA tasks inside ECOC decomposition (OVO + random ECOC only).",
    )
    parser.add_argument(
        "--fs-ecoc-negative-ratio",
        type=float,
        default=2.0,
        help="Negative:positive cap ratio for ECOC one-vs-all tasks.",
    )
    parser.add_argument(
        "--fs-joint-multiclass-min-classes",
        type=int,
        default=3,
        help="Minimum class count required to run the joint multinomial shared-support selector.",
    )
    parser.add_argument(
        "--fs-joint-multiclass-max-features",
        type=int,
        default=256,
        help="Maximum prefiltered feature pool size for joint multinomial selector fitting.",
    )
    parser.add_argument(
        "--fs-joint-multiclass-path-grid-size",
        type=int,
        default=6,
        help="Number of C values in the multinomial elastic-net path.",
    )
    parser.add_argument(
        "--fs-joint-multiclass-min-c",
        type=float,
        default=0.05,
        help="Minimum C in the multinomial elastic-net path.",
    )
    parser.add_argument(
        "--fs-joint-multiclass-max-c",
        type=float,
        default=1.6,
        help="Maximum C in the multinomial elastic-net path.",
    )
    parser.add_argument(
        "--fs-joint-multiclass-l1-ratio",
        type=float,
        default=0.55,
        help="Elastic-net L1 ratio for the joint multinomial selector.",
    )
    parser.add_argument(
        "--fs-joint-multiclass-univariate-blend",
        type=float,
        default=0.20,
        help="Blend weight for MI/F-score relevance in final joint selector scores.",
    )
    parser.add_argument(
        "--fs-dove-min-classes",
        type=int,
        default=3,
        help="Minimum class count required to run A19 DOvE-style class-specific selector.",
    )
    parser.add_argument(
        "--fs-dove-max-pairs-per-class",
        type=int,
        default=4,
        help="Maximum one-vs-each class pairs evaluated per class in A19 selector.",
    )
    parser.add_argument(
        "--fs-dove-path-grid-size",
        type=int,
        default=5,
        help="Number of support-size scaling steps for A19 DOvE path search.",
    )
    parser.add_argument(
        "--fs-dove-specificity-weight",
        type=float,
        default=0.35,
        help="Class-specificity blend weight in A19 DOvE scoring.",
    )
    parser.add_argument(
        "--fs-dove-minority-boost",
        type=float,
        default=0.50,
        help="Inverse class-count weighting exponent for A19 DOvE class weighting.",
    )
    parser.add_argument(
        "--fs-sparse-multinomial-min-classes",
        type=int,
        default=3,
        help="Minimum class count required for A20 sparse multinomial selector.",
    )
    parser.add_argument(
        "--fs-sparse-multinomial-max-features",
        type=int,
        default=320,
        help="Maximum prefiltered pool size for A20 sparse multinomial fitting.",
    )
    parser.add_argument(
        "--fs-sparse-multinomial-path-grid-size",
        type=int,
        default=6,
        help="Number of C values in A20 sparse multinomial path.",
    )
    parser.add_argument(
        "--fs-sparse-multinomial-min-c",
        type=float,
        default=0.05,
        help="Minimum C in A20 sparse multinomial regularization path.",
    )
    parser.add_argument(
        "--fs-sparse-multinomial-max-c",
        type=float,
        default=1.6,
        help="Maximum C in A20 sparse multinomial regularization path.",
    )
    parser.add_argument(
        "--fs-sparse-multinomial-backend",
        type=str,
        default="mixed",
        choices=["l1", "elasticnet", "mixed"],
        help="Sparse multinomial backend mode for A20.",
    )
    parser.add_argument(
        "--fs-sparse-multinomial-l1-ratio",
        type=float,
        default=0.70,
        help="Elastic-net l1_ratio when A20 backend includes elastic-net.",
    )
    parser.add_argument(
        "--fs-sparse-multinomial-univariate-blend",
        type=float,
        default=0.20,
        help="Blend weight for MI/F relevance in A20 sparse multinomial scores.",
    )
    parser.add_argument(
        "--fs-sparse-multinomial-max-iter",
        type=int,
        default=5000,
        help="max_iter for A20 sparse multinomial logistic fits.",
    )
    parser.add_argument(
        "--fs-sparse-multinomial-screening-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "prefilter_aggressive",
            "prefilter_balanced",
            "prefilter_conservative",
            "strong",
            "gap_safe",
            "slores",
        ],
        help=(
            "A20-R1 heuristic prefilter mode for sparse multinomial runtime containment. "
            "Use prefilter_aggressive/prefilter_balanced/prefilter_conservative. "
            "Legacy aliases strong/gap_safe/slores remain accepted (deprecated)."
        ),
    )
    parser.add_argument(
        "--fs-sparse-multinomial-screening-keep-fraction",
        type=float,
        default=1.0,
        help="Fraction of prefiltered features retained by A20-R1 screening.",
    )
    parser.add_argument(
        "--fs-sparse-multinomial-screening-min-features",
        type=int,
        default=64,
        help="Minimum retained features under A20-R1 screening.",
    )
    parser.add_argument(
        "--disable-fs-sparse-multinomial-screening-fallback-on-failure",
        action="store_true",
        help="Disable fallback to the unscreened pool when A20-R1 screened sparse path fails.",
    )
    parser.add_argument(
        "--fs-nsc-shrinkage-grid-size",
        type=int,
        default=6,
        help="Number of shrinkage deltas evaluated by A22 nearest shrunken centroids selector.",
    )
    parser.add_argument(
        "--fs-nsc-min-classes",
        type=int,
        default=3,
        help="Minimum class count required to run A22 nearest shrunken centroids selector.",
    )
    parser.add_argument(
        "--fs-nsc-thresholding-mode",
        type=str,
        default="soft",
        choices=["soft", "hard", "order", "auto"],
        help="A27 NSC thresholding variant: soft/hard/order or auto-search across all.",
    )
    parser.add_argument(
        "--fs-nsc-order-quantile",
        type=float,
        default=0.75,
        help="A27 NSC order-threshold quantile (used when mode is order or auto).",
    )
    parser.add_argument(
        "--enable-fs-nsc-deep-shrinkage-search",
        action="store_true",
        help="Enable A27 deep NSC shrinkage grid expansion with quantile-derived deltas.",
    )
    parser.add_argument(
        "--fs-class-pareto-min-classes",
        type=int,
        default=3,
        help="Minimum class count required for A28 class-specific Pareto selector.",
    )
    parser.add_argument(
        "--fs-class-pareto-top-per-class",
        type=int,
        default=64,
        help="Per-class candidate budget before A28 Pareto dominance filtering.",
    )
    parser.add_argument(
        "--fs-class-pareto-global-fraction",
        type=float,
        default=0.40,
        help="Additional global candidate fraction mixed into A28 candidate pool.",
    )
    parser.add_argument(
        "--fs-class-pareto-minority-boost",
        type=float,
        default=0.50,
        help="Minority-class weighting exponent used by A28 class-specific scoring.",
    )
    parser.add_argument(
        "--fs-class-pareto-kw-weight",
        type=float,
        default=0.25,
        help="Rank-based (KW-family) score weight in A28 class-specific relevance.",
    )
    parser.add_argument(
        "--fs-sdr-min-classes",
        type=int,
        default=3,
        help="Minimum class count required to run SDR selectors (SIR/SAVE/PFC).",
    )
    parser.add_argument(
        "--fs-sdr-prefilter-max-features",
        type=int,
        default=512,
        help="Max prefiltered candidate pool size used by SDR selectors.",
    )
    parser.add_argument(
        "--fs-sdr-n-components",
        type=int,
        default=3,
        help="Number of SDR directions used for feature scoring.",
    )
    parser.add_argument(
        "--fs-sdr-covariance-ridge",
        type=float,
        default=1e-3,
        help="Ridge regularization added to SDR covariance estimates.",
    )
    parser.add_argument(
        "--enable-fs-per-class-quota",
        action="store_true",
        help="Enable A26 per-class quota feature allocation overlay for class-specific selectors.",
    )
    parser.add_argument(
        "--disable-fs-per-class-quota",
        dest="enable_fs_per_class_quota",
        action="store_false",
        help="Disable A26 per-class quota feature allocation overlay.",
    )
    # Val-3 Promotion: enable per-class quota by default.
    parser.set_defaults(enable_fs_per_class_quota=True)
    parser.add_argument(
        "--fs-per-class-quota-min-per-class",
        type=int,
        default=1,
        help="Minimum selected features requested per class under A26 quota overlay.",
    )
    parser.add_argument(
        "--fs-per-class-quota-max-fraction",
        type=float,
        default=0.60,
        help="Maximum fraction of final selected features that can be quota-forced by A26.",
    )
    parser.add_argument(
        "--fs-hsic-lasso-alpha",
        type=float,
        default=0.01,
        help="A25 HSIC Lasso non-negative L1 regularization strength.",
    )
    parser.add_argument(
        "--fs-hsic-lasso-prefilter-max-features",
        type=int,
        default=128,
        help="Maximum prefiltered candidate count for A25 HSIC Lasso fitting.",
    )
    parser.add_argument(
        "--fs-hsic-lasso-feature-sigma",
        type=float,
        default=0.0,
        help="RBF sigma for A25 feature kernels (<=0 uses per-feature median heuristic).",
    )
    parser.add_argument(
        "--fs-hsic-lasso-target-sigma",
        type=float,
        default=0.0,
        help="RBF sigma for A25 regression target kernel (classification uses delta kernel).",
    )
    parser.add_argument(
        "--fs-hsic-lasso-relevance-blend",
        type=float,
        default=0.20,
        help="Blend weight for HSIC relevance in A25 combined ranking.",
    )
    parser.add_argument(
        "--fs-hsic-lasso-max-iter",
        type=int,
        default=4000,
        help="Maximum solver iterations for A25 HSIC Lasso fit.",
    )
    parser.add_argument(
        "--enable-fs-mrmr-mi-redundancy",
        action="store_true",
        help="Enable MI-based mRMR redundancy penalty tiers (T-R-126; default uses legacy Pearson).",
    )
    parser.add_argument(
        "--fs-mrmr-mi-n-bins",
        type=int,
        default=8,
        help="Bin count for binned-MI redundancy estimation in mRMR/CMIM/FCBF.",
    )
    parser.add_argument(
        "--fs-cmim-min-samples",
        type=int,
        default=60,
        help="Minimum samples required to activate CMIM selector.",
    )
    parser.add_argument(
        "--fs-cmim-n-bins",
        type=int,
        default=8,
        help="Bin count for CMIM conditional-MI estimation.",
    )
    parser.add_argument(
        "--fs-fcbf-n-bins",
        type=int,
        default=8,
        help="Bin count for FCBF symmetric-uncertainty estimation.",
    )
    parser.add_argument("--fs-ipss-path-grid-size", type=int, default=7)
    parser.add_argument("--fs-ipss-min-c", type=float, default=0.08)
    parser.add_argument("--fs-ipss-max-c", type=float, default=1.20)
    parser.add_argument("--fs-ipss-target-fdr", type=float, default=0.15)
    parser.add_argument("--fs-ipss-null-shuffle-rounds", type=int, default=1)
    parser.add_argument("--enable-fs-ipss-eats-threshold", action="store_true")
    parser.add_argument("--fs-ipss-eats-exclusion-quantile", type=float, default=0.90)
    parser.add_argument("--fs-ipss-eats-min-threshold", type=float, default=0.45)
    parser.add_argument(
        "--fs-ipss-importance-model",
        type=str,
        default="linear_svm",
        choices=["linear_svm", "gradient_boosting", "random_forest"],
    )
    parser.add_argument(
        "--fs-ipss-gate-min-classes",
        type=int,
        default=0,
        help="If >0, IPSS only runs when n_classes >= threshold (0 disables this gate).",
    )
    parser.add_argument(
        "--fs-ipss-gate-min-p-over-n",
        type=float,
        default=0.0,
        help="If >0, IPSS only runs when (p/n) >= threshold (0 disables this gate).",
    )
    parser.add_argument("--fs-cluster-corr-threshold", type=float, default=0.85)
    parser.add_argument("--fs-cluster-max-per-cluster", type=int, default=2)
    parser.add_argument("--fs-cluster-min-freq", type=float, default=0.55)
    parser.add_argument("--enable-fs-stability-loss-guided-validation", action="store_true")
    parser.add_argument("--fs-stability-validation-fraction", type=float, default=0.25)
    parser.add_argument("--fs-stability-validation-quantile", type=float, default=0.40)
    parser.add_argument("--fs-stability-validation-min-samples", type=int, default=6)
    parser.add_argument(
        "--fs-stability-threshold-method",
        type=str,
        default="fixed",
        choices=["fixed", "eats", "cpss"],
        help="Thresholding mode for stability selectors (fixed vs EATS vs CPSS-calibrated threshold).",
    )
    parser.add_argument(
        "--fs-stability-target-pfer",
        type=float,
        default=1.0,
        help="Target PFER budget used when --fs-stability-threshold-method cpss is active.",
    )
    parser.add_argument("--fs-copula-knockoff-draws", type=int, default=30)
    parser.add_argument("--fs-copula-alpha-kn", type=float, default=0.10)
    parser.add_argument("--fs-copula-alpha-ebh", type=float, default=0.20)
    parser.add_argument("--fs-copula-truncation-level", type=int, default=5)
    parser.add_argument(
        "--fs-copula-generator",
        type=str,
        default="copula",
        choices=["copula", "deepdrk"],
        help="Knockoff generator backend: copula (DTDCKe) or deepdrk (CPU-only low-rank residual sampler).",
    )
    parser.add_argument(
        "--fs-copula-deepdrk-latent-fraction",
        type=float,
        default=0.35,
        help="Latent rank fraction used by deepdrk knockoff generator.",
    )
    parser.add_argument(
        "--fs-copula-deepdrk-noise-scale",
        type=float,
        default=1.0,
        help="Residual noise scale used by deepdrk knockoff generator.",
    )
    parser.add_argument(
        "--fs-copula-derandomize-runs",
        type=int,
        default=5,
        help="Derandomized knockoff aggregation runs (K). Use 20 for T-R-110 experiments.",
    )
    parser.add_argument("--fs-copula-stabilizer-runs", type=int, default=1)
    parser.add_argument("--enable-fs-copula-stabilizer-ebh", action="store_true")
    parser.add_argument("--fs-copula-stabilizer-seed-stride", type=int, default=997)
    parser.add_argument(
        "--fs-importance-uq-enabled",
        action="store_true",
        help="Enable reporting-only feature-importance fold-variance diagnostics (T-P3-006).",
    )
    parser.add_argument(
        "--fs-importance-uq-min-cv-folds",
        type=int,
        default=3,
        help="Minimum successful folds required for importance-UQ reporting.",
    )
    parser.add_argument("--fs-decorrelated-stability-eps", type=float, default=1e-3)
    parser.add_argument(
        "--fs-decorrelated-stability-min-max-abs-corr",
        type=float,
        default=0.0,
        help="If >0, decorrelated stability is skipped when the prefiltered pool's max abs corr is below this threshold.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-pool-factor",
        type=float,
        default=2.5,
        help="Initial pool multiplier for iterative redundancy-pruning wrapper.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-max-rounds",
        type=int,
        default=32,
        help="Maximum feature-removal rounds for iterative redundancy-pruning wrapper.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-min-improvement",
        type=float,
        default=-0.002,
        help="Minimum accepted inner-CV score delta per pruning step.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-max-cumulative-loss",
        type=float,
        default=0.02,
        help="Stop iterative pruning once the cumulative score delta drops below -budget (sum of per-round deltas).",
    )
    parser.add_argument(
        "--fs-iterative-pruning-redundancy-weight",
        type=float,
        default=0.65,
        help="Redundancy-vs-relevance tradeoff in pruning priority (0..1).",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-prefilter-cap",
        type=int,
        default=220,
        help="Max prefilter pool size for runtime-bounded iterative pruning.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-candidate-fraction",
        type=float,
        default=0.35,
        help="Per-round fraction of removal candidates evaluated in bounded iterative pruning.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-min-candidates",
        type=int,
        default=4,
        help="Minimum per-round candidate removals evaluated in bounded iterative pruning.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-max-evaluations",
        type=int,
        default=48,
        help="Maximum wrapper score evaluations in bounded iterative pruning.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-max-runtime-seconds",
        type=float,
        default=30.0,
        help="Wall-clock time budget (seconds) for bounded iterative pruning.",
    )
    parser.add_argument(
        "--disable-fs-iterative-pruning-bounded-class-gating",
        action="store_true",
        help="Disable class-aware candidate-budget gating in bounded iterative pruning.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-multiclass-scale",
        type=float,
        default=0.70,
        help="Candidate-budget multiplier when classes > 2 in bounded iterative pruning.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-imbalance-trigger",
        type=float,
        default=2.5,
        help="Imbalance ratio threshold for additional candidate-budget downscaling in bounded iterative pruning.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-imbalance-scale",
        type=float,
        default=0.75,
        help="Candidate-budget multiplier when imbalance threshold is exceeded in bounded iterative pruning.",
    )
    parser.add_argument(
        "--enable-fs-iterative-pruning-bounded-cpss-overlay",
        action="store_true",
        help="Enable A16 CPSS-style complementary-pairs stability overlay for bounded iterative pruning.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-cpss-pairs",
        type=int,
        default=4,
        help="Number of complementary pairs used by CPSS overlay.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-cpss-stability-threshold",
        type=float,
        default=0.60,
        help="Minimum CPSS support frequency for stable features.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-cpss-min-stable-features",
        type=int,
        default=2,
        help="Minimum stable-support size required before CPSS overlay can switch.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-cpss-min-jaccard",
        type=float,
        default=0.35,
        help="Minimum Jaccard overlap with base bounded support required for CPSS switch.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-bounded-cpss-max-score-drop",
        type=float,
        default=0.005,
        help="Maximum allowed wrapper-score drop when CPSS overlay switches support.",
    )
    parser.add_argument(
        "--enable-fs-iterative-pruning-class-pareto-prefilter",
        action="store_true",
        help="Enable A17 class-dominance-aware Pareto prefilter before iterative pruning wrappers.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-class-pareto-min-classes",
        type=int,
        default=3,
        help="Minimum number of classes required to activate A17 Pareto prefilter.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-class-pareto-top-per-class",
        type=int,
        default=64,
        help="Per-class candidate budget used by A17 Pareto prefilter before dominance ranking.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-class-pareto-global-fraction",
        type=float,
        default=0.40,
        help="Additional global candidate fraction mixed into A17 Pareto prefilter candidate pool.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-class-pareto-minority-boost",
        type=float,
        default=0.50,
        help="Minority-class weighting exponent for A17 Pareto prefilter class-specific scores.",
    )
    parser.add_argument(
        "--enable-fs-iterative-pruning-class-pareto-stability-gate",
        action="store_true",
        help="Enable A18 conservative stability gate on top of A17 class-Pareto prefilter.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-class-pareto-stability-subsamples",
        type=int,
        default=6,
        help="Number of stratified subsample rounds used by the A18 stability gate.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-class-pareto-stability-fraction",
        type=float,
        default=0.70,
        help="Subsample train fraction used by the A18 stability gate.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-class-pareto-stability-threshold",
        type=float,
        default=0.55,
        help="Support-frequency threshold for stable candidates in the A18 gate.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-class-pareto-stability-min-overlap",
        type=float,
        default=0.50,
        help="Minimum stable-overlap recall required to keep raw Pareto support in A18.",
    )
    parser.add_argument(
        "--fs-iterative-pruning-class-pareto-stability-min-stable-features",
        type=int,
        default=4,
        help="Minimum number of stable candidates required by the A18 gate.",
    )
    parser.add_argument(
        "--disable-fs-iterative-pruning-class-pareto-stability-fallback-on-failure",
        action="store_true",
        help="Disable A18 fallback-to-global-prefilter when the stability gate fails hard.",
    )
    parser.add_argument("--enable-diversity-oracle", action="store_true")
    parser.add_argument(
        "--disable-fs-stability-oracle",
        action="store_true",
        help="Disable the MNPO stability oracle for the benchmark/runtime FS path.",
    )
    parser.add_argument(
        "--disable-fs-complexity-oracle",
        action="store_true",
        help="Disable the MNPO complexity oracle for the benchmark/runtime FS path.",
    )
    parser.add_argument(
        "--disable-fs-robust-oracle",
        action="store_true",
        help="Disable the MNPO robust oracle for the benchmark/runtime FS path.",
    )
    parser.add_argument("--enable-hybrid-model-cv", action="store_true")
    parser.add_argument(
        "--model-cv-lr-max-iter",
        type=int,
        default=10000,
        help="max_iter for LogisticRegression model-candidate CV (lbfgs/saga).",
    )
    parser.add_argument("--model-cv-balanced-weight", type=float, default=0.6)
    parser.add_argument("--model-cv-macro-f1-weight", type=float, default=0.4)
    parser.add_argument(
        "--enable-model-cv-runtime-containment",
        action="store_true",
        help=(
            "Enable OP1.1 runtime-aware model-candidate containment for model-CV "
            "selection on high-p/multiclass/sparse-class regimes."
        ),
    )
    parser.add_argument(
        "--model-cv-runtime-max-candidates",
        type=int,
        default=0,
        help=(
            "Hard cap on model-CV candidates after containment prioritization "
            "(0 enables auto-cap based on data regime)."
        ),
    )
    parser.add_argument(
        "--model-cv-runtime-high-p-over-n-threshold",
        type=float,
        default=40.0,
        help="Auto-containment trigger: treat tasks with p/n >= threshold as high-p.",
    )
    parser.add_argument(
        "--model-cv-runtime-high-class-threshold",
        type=int,
        default=6,
        help="Auto-containment trigger: treat tasks with n_classes >= threshold as high-class-count.",
    )
    parser.add_argument(
        "--model-cv-runtime-min-class-count-threshold",
        type=int,
        default=12,
        help="Auto-containment trigger: treat tasks with min class count <= threshold as sparse-class.",
    )
    parser.add_argument(
        "--model-candidate-profile",
        type=str,
        default="default",
        choices=sorted(MODEL_CANDIDATE_PROFILES.keys()),
        help=(
            "Optional named downstream model-candidate profile. "
            "'a11_medium_mismatch' sets exactly: lr svm_rbf svm_linear dlda knn nb vote_ensemble."
        ),
    )
    parser.add_argument(
        "--model-candidates",
        nargs="+",
        default=[],
        choices=[
            "lr",
            "svm_rbf",
            "svm_linear",
            "dlda",
            "shrinkage_lda",
            "nsc",
            "pls_da_classifier",
            "gpc",
            "nb",
            "vote_ensemble",
            "rp_ensemble",
            "dbda",
            "gqda",
            "bc_svm_linear",
            "sglnn",
            "rff_lr",
            "near_subspace",
            "spatial_median_da",
            "copula_da",
            "cada_tent1",
            "cada_tent2",
            "cada_hinge2",
            "tabm",
            "realmlp",
            "cpda",
            "elastic_net_lr",
            "rf",
            "knn",
            "xgb",
            "lgbm",
            "extra_tree",
            "catboost",
            "tabpfn",
        ],
        help="Explicit model candidate set for CV selection (overrides include-* flags).",
    )
    parser.add_argument(
        "--classifier-selection-mode",
        type=str,
        default="legacy",
        choices=["legacy", "mnpo_hybrid", "tune_first"],
        help="Classifier selection regime: legacy model-CV, MNPO hybrid oracle selection, or tune-first MNPO.",
    )
    parser.add_argument(
        "--classifier-oracle-k",
        type=int,
        default=1,
        help="Top-k classifier families retained by MNPO classifier oracle (mnpo_hybrid mode).",
    )
    parser.add_argument(
        "--classifier-oracle-weighting-mode",
        type=str,
        default="tritrust",
        choices=["tritrust", "uniform", "banzhaf", "shapley"],
        help="Oracle weighting mode for classifier MNPO (mnpo_hybrid mode).",
    )
    parser.add_argument(
        "--disable-classifier-oracle-calibration",
        action="store_true",
        help="Disable Brier-based calibration oracle in mnpo_hybrid classifier selection.",
    )
    parser.add_argument(
        "--disable-classifier-oracle-james-stein",
        action="store_true",
        help="Disable James-Stein shrinkage over classifier-oracle weights in mnpo_hybrid mode.",
    )
    parser.add_argument(
        "--enable-classifier-oracle-cvar",
        action="store_true",
        help="Add a CVaR fold-tail oracle to classifier MNPO scoring (opt-in).",
    )
    parser.add_argument(
        "--classifier-oracle-cvar-alpha",
        type=float,
        default=0.33,
        help="Tail mass alpha used by the classifier CVaR oracle.",
    )
    parser.add_argument(
        "--enable-classifier-oracle-dynamic-complexity",
        action="store_true",
        help="Use tuned-model complexity metrics instead of the static classifier prior.",
    )
    parser.add_argument(
        "--enable-classifier-oracle-portfolio-diversity",
        action="store_true",
        help="Apply greedy diversity filtering when extracting the top-k classifier portfolio.",
    )
    parser.add_argument(
        "--classifier-oracle-portfolio-overlap-threshold",
        type=float,
        default=0.75,
        help="OOF prediction-overlap threshold used by classifier portfolio diversity filtering.",
    )
    parser.add_argument(
        "--classifier-oracle-portfolio-corr-threshold",
        type=float,
        default=0.85,
        help="Correctness-correlation threshold used by classifier portfolio diversity filtering.",
    )
    parser.add_argument(
        "--disable-classifier-oracle-hoeffding-racing",
        action="store_true",
        help="Disable Hoeffding racing for classifier elimination in mnpo_hybrid mode.",
    )
    parser.add_argument(
        "--classifier-oracle-hoeffding-delta",
        type=float,
        default=0.10,
        help="Delta used by Hoeffding racing in mnpo_hybrid classifier selection.",
    )
    parser.add_argument(
        "--disable-classifier-oracle-bbc",
        action="store_true",
        help="Disable BBC bootstrap bias correction in classifier oracle scoring.",
    )
    parser.add_argument(
        "--classifier-oracle-bbc-bootstrap-rounds",
        type=int,
        default=200,
        help="Bootstrap rounds for BBC correction in classifier oracle scoring.",
    )
    parser.add_argument(
        "--classifier-oracle-bbc-ci-level",
        type=float,
        default=0.90,
        help="Confidence level for BBC interval reporting in classifier oracle scoring.",
    )
    parser.add_argument(
        "--enable-classifier-oracle-ensemble",
        action="store_true",
        help="Enable ensemble over MNPO top-k classifier families (mnpo_hybrid mode).",
    )
    parser.add_argument(
        "--classifier-oracle-ensemble-voting-mode",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help="Voting mode for MNPO classifier ensemble. 'soft' uses predict_proba with Nash weights (B2).",
    )
    parser.add_argument(
        "--enable-classifier-oracle-greedy-ensemble",
        action="store_true",
        help="Enable greedy ensemble selection with replacement (B1, Caruana 2004).",
    )
    parser.add_argument(
        "--classifier-oracle-greedy-ensemble-rounds",
        type=int,
        default=10,
        help="Maximum rounds of greedy model addition in ensemble selection (B1).",
    )
    parser.add_argument(
        "--enable-classifier-oracle-candidate-pruning",
        action="store_true",
        help="Enable leave-one-out marginal-contribution pruning of candidates before oracle game (B3).",
    )
    parser.add_argument(
        "--classifier-oracle-candidate-pruning-threshold",
        type=float,
        default=0.0,
        help="Marginal contribution threshold for candidate pruning; candidates below this are dropped (B3).",
    )
    parser.add_argument(
        "--enable-classifier-oracle-incumbent-early-stopping",
        action="store_true",
        help="Enable incumbent-based early stopping in Hoeffding racing (B8).",
    )
    parser.add_argument(
        "--classifier-oracle-behavior-profile",
        type=str,
        default="current",
        choices=["current", "val18_compat"],
        help=(
            "Classifier-oracle behavior profile. "
            "'current' enables the post-val-18 multiclass fixes; "
            "'val18_compat' emulates the val-18 tagged classifier-oracle behavior."
        ),
    )
    parser.add_argument(
        "--classifier-oracle-complexity-shrinkage",
        action="store_true",
        help="Scale classifier-oracle James-Stein shrinkage by dataset complexity (T-R-406).",
    )
    parser.add_argument(
        "--exclude-classifiers",
        nargs="+",
        default=[],
        help=(
            "Classifier families to remove from the candidate pool before regime gating. "
            "Accepts a space/comma-separated list."
        ),
    )
    parser.add_argument(
        "--classifier-regime-candidate-exclusions",
        nargs="+",
        default=[],
        help=(
            "Regime-specific classifier exclusions encoded as regime:family "
            "(for example: hdlss_moderate:tabpfn standard:tabpfn)."
        ),
    )
    parser.add_argument(
        "--classifier-complexity-prior-override",
        nargs="+",
        action="append",
        default=[],
        help=(
            "Override classifier complexity priors with family=value entries "
            "(for example: lr=0.75 svm_linear=0.85)."
        ),
    )
    parser.add_argument(
        "--disable-classifier-oracle-per-family-flaml",
        action="store_true",
        help="Disable per-family FLAML HPO dispatch in mnpo_hybrid mode.",
    )
    parser.add_argument(
        "--classification-backend",
        type=str,
        default="sklearn",
        choices=["sklearn", "flaml", "optuna"],
        help="Stage-2 classifier backend used after FS feature selection.",
    )
    parser.add_argument(
        "--flaml-time-budget",
        type=int,
        default=60,
        help="FLAML time budget in seconds (used only when --classification-backend flaml).",
    )
    parser.add_argument(
        "--optuna-time-budget",
        type=int,
        default=120,
        help="Optuna optimization timeout in seconds (used only when --classification-backend optuna).",
    )
    parser.add_argument(
        "--optuna-n-trials",
        type=int,
        default=25,
        help="Maximum Optuna trials (used only when --classification-backend optuna).",
    )
    parser.add_argument("--include-elastic-net-model", action="store_true")
    parser.add_argument("--include-rf-model", action="store_true")
    parser.add_argument("--include-knn-model", action="store_true")
    parser.add_argument("--include-svm-linear-model", action="store_true")
    parser.add_argument("--include-dlda-model", action="store_true")
    parser.add_argument("--include-nsc-model", action="store_true")
    parser.add_argument("--include-pls-da-model", action="store_true")
    parser.add_argument("--include-gpc-model", action="store_true")
    parser.add_argument("--include-nb-model", action="store_true")
    parser.add_argument("--include-vote-ensemble-model", action="store_true")
    parser.add_argument("--include-rp-ensemble-model", action="store_true")
    parser.add_argument("--include-dbda-model", action="store_true")
    parser.add_argument("--include-gqda-model", action="store_true")
    parser.add_argument("--include-bc-svm-linear-model", action="store_true")
    parser.add_argument("--include-sglnn-model", action="store_true")
    parser.add_argument("--include-xgb-model", action="store_true")
    parser.add_argument("--include-lgbm-model", action="store_true")
    parser.add_argument("--include-extra-tree-model", action="store_true")
    parser.add_argument("--include-catboost-model", action="store_true")
    parser.add_argument("--include-tabpfn-model", action="store_true")
    parser.add_argument(
        "--enable-stage2-ratio-augmentation",
        action="store_true",
        help="Enable post-FS ratio feature augmentation for final classifier training.",
    )
    parser.add_argument(
        "--stage2-ratio-max-features",
        type=int,
        default=16,
        help="Maximum number of post-FS ratio features appended in stage-2 augmentation.",
    )
    parser.add_argument(
        "--stage2-ratio-selection-method",
        type=str,
        default="correlation",
        choices=["correlation", "ktsp"],
        help="Pair scoring heuristic for stage-2 ratio augmentation.",
    )
    parser.add_argument(
        "--stage2-ratio-epsilon",
        type=float,
        default=1e-6,
        help="Numerical epsilon used in stage-2 ratio construction.",
    )
    parser.add_argument(
        "--enable-classifier-conformal",
        action="store_true",
        help="Enable split-conformal prediction-set diagnostics for the selected Stage-2 classifier.",
    )
    parser.add_argument(
        "--classifier-conformal-alpha",
        type=float,
        default=0.10,
        help="Conformal alpha (target miscoverage) for classifier prediction sets.",
    )
    parser.add_argument(
        "--classifier-conformal-calibration-fraction",
        type=float,
        default=0.25,
        help="Calibration split fraction used by split-conformal diagnostics.",
    )
    parser.add_argument(
        "--classifier-conformal-min-calibration",
        type=int,
        default=20,
        help="Minimum calibration samples required to apply classifier conformal diagnostics.",
    )
    parser.add_argument(
        "--classifier-conformal-output-sets",
        action="store_true",
        help="Emit per-sample conformal prediction sets in run artifacts (can increase artifact size).",
    )
    parser.add_argument(
        "--classifier-conformal-method",
        type=str,
        default="split",
        choices=["split", "aps", "raps", "cross"],
        help="Conformal method for classifier diagnostics (split baseline or MAPIE-style aps/raps/cross).",
    )
    parser.add_argument(
        "--stage2-max-train-test-gap",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed train-test balanced-accuracy gap for classifier candidates. "
            "Candidates exceeding this gap are rejected. 0.0 = disabled (T-R-246)."
        ),
    )
    parser.add_argument(
        "--stage2-tree-complexity-penalty-enabled",
        action="store_true",
        help="Enable soft score penalty for tree-based classifier candidates proportional to their train-test gap (T-R-246).",
    )
    parser.add_argument(
        "--stage2-tree-complexity-penalty-strength",
        type=float,
        default=0.1,
        help="Strength multiplier for the tree-model complexity penalty (T-R-246).",
    )
    parser.add_argument(
        "--use-sota-matched-classifiers",
        action="store_true",
        help=(
            "For each dataset, run an additional 'sota_matched' config using "
            "exactly the evaluation classifier(s) from the published SOTA "
            "paper (see DATASET_SOTA_CLASSIFIERS).  The normal baseline config "
            "is always run as well, enabling direct comparison."
        ),
    )

    parser.add_argument(
        "--enable-maqc-pairing",
        action="store_true",
        help="Enable MAQC-II style selector+classifier pairing search (baseline config only unless --maqc-pairing-all-configs is set).",
    )
    parser.add_argument(
        "--maqc-fs-method-sets",
        nargs="+",
        default=[],
        choices=sorted(FS_METHOD_SETS.keys()),
        help="FS method presets to consider when --enable-maqc-pairing is active (default: strict_plus_mrmr, mnpo_rankagg_extended, mnpo_ova_extended).",
    )
    parser.add_argument(
        "--maqc-pairing-all-configs",
        action="store_true",
        help="Apply MAQC pairing search to every ablation config (default: baseline config only).",
    )
    parser.add_argument(
        "--maqc-pairing-min-improvement",
        type=float,
        default=0.0,
        help="Minimum absolute model-CV score improvement required to switch away from the configured selector.",
    )
    parser.add_argument(
        "--maqc-pairing-min-improvement-se-mult",
        type=float,
        default=0.0,
        help="If >0, also require improvement >= (mult * combined CV standard error) before switching selectors.",
    )

    parser.add_argument("--ablation-profile", type=str, default="core", choices=["none", "core", "full"])

    parser.add_argument(
        "--emit-summary",
        action="store_true",
        help="Emit a compact per-run JSON summary capturing FS method preset, portfolio, oracle weights, DF families, diagnostics and runtime breakdowns.",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default="",
        help="Directory for per-run summary JSONs (default: alongside other outputs in the run directory).",
    )

    parser.add_argument("--disable-df-robust", action="store_true")
    parser.add_argument("--disable-df-lrt", action="store_true")
    parser.add_argument("--disable-support-filter", action="store_true")
    parser.add_argument("--disable-cdf-transform", action="store_true")
    parser.add_argument("--disable-cdf-reliability-gate", action="store_true")
    parser.add_argument("--disable-low-gof-downweight", action="store_true")
    parser.add_argument("--disable-rank-prefilter", action="store_true")
    parser.add_argument(
        "--disable-prefilter",
        dest="disable_rank_prefilter",
        action="store_true",
        help="Disable the rank prefilter stage (alias for --disable-rank-prefilter).",
    )
    parser.add_argument(
        "--enable-prefilter",
        dest="disable_rank_prefilter",
        action="store_false",
        help="Force-enable the rank prefilter stage.",
    )
    parser.add_argument("--enable-df-cv", action="store_true")
    parser.add_argument("--enable-dist-stability-weight", action="store_true")
    parser.add_argument(
        "--fs-method-set",
        type=str,
        default="mnpo_class_pareto_extended",
        choices=sorted(FS_METHOD_SETS.keys()),
        help="Preset FS stack for the baseline configuration.",
    )

    parser.add_argument(
        "--allow-synthetic-fallback",
        dest="allow_synthetic_fallback",
        action="store_true",
        help="Enable synthetic fallback when real dataset loading fails.",
    )
    parser.add_argument(
        "--no-synthetic-fallback",
        dest="allow_synthetic_fallback",
        action="store_false",
        help="Fail when synthetic fallback is disabled.",
    )
    parser.set_defaults(allow_synthetic_fallback=False)
    parser.add_argument(
        "--dataset-integrity-policy",
        type=str,
        default="error",
        choices=["fallback", "skip", "error"],
        help=(
            "Policy when a loaded dataset fails class-diversity sanity checks "
            "(n_classes/min class count)."
        ),
    )
    parser.add_argument(
        "--dataset-min-classes",
        type=int,
        default=2,
        help="Minimum required number of classes after dataset loading.",
    )
    parser.add_argument(
        "--dataset-min-class-count",
        type=int,
        default=1,
        help="Minimum required samples per class after dataset loading.",
    )

    parser.add_argument("--synthetic-sample-cap", type=int, default=2500)
    parser.add_argument("--synthetic-feature-cap", type=int, default=10000)

    parser.add_argument("--output-dir", type=str, default="run_artifacts/validation/df_fs_benchmark")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if bool(args.allow_synthetic_fallback):
        parser.error(
            "--allow-synthetic-fallback is not supported for benchmark validation runs; "
            "use the HuggingFace bundle instead."
        )
    run_benchmark(args)


if __name__ == "__main__":
    main()

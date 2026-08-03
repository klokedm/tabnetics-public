"""Bounded dependence descriptors for auto-router dataset summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from tabnetics.feature_selection.methods.filter import (
    _mutual_information_discrete,
    _safe_rank_bin,
)

DEPENDENCE_DESCRIPTOR_POLICY = "bounded_binned_mi_v1"
DEPENDENCE_DESCRIPTOR_KEYS: tuple[str, ...] = (
    "mutual_info_mean",
    "mutual_info_std",
    "pairwise_redundancy_ratio",
)
DEPENDENCE_DESCRIPTOR_CAPS: dict[str, int] = {
    "target_mi_max_features": 5000,
    "pairwise_max_features": 128,
    "pairwise_max_pairs": 512,
    "max_bins": 8,
}
DEPENDENCE_DESCRIPTOR_ARTIFACT_STATUS_SCHEMA_VERSION = "dependence_descriptor_artifact_status_v1"


def dependence_descriptor_model_input_enabled(
    metadata: Mapping[str, Any] | None,
    feature_names: Sequence[str] | None = None,
) -> bool:
    """Return true only for artifacts trained with the current descriptor contract."""

    payload = dict(metadata or {})
    if str(payload.get("dependence_descriptor_policy", "") or "") != DEPENDENCE_DESCRIPTOR_POLICY:
        return False
    keys = tuple(str(key) for key in (payload.get("dependence_descriptor_keys", []) or []))
    if keys != tuple(DEPENDENCE_DESCRIPTOR_KEYS):
        return False
    caps = dict(payload.get("dependence_descriptor_caps", {}) or {})
    if caps != dict(DEPENDENCE_DESCRIPTOR_CAPS):
        return False
    if feature_names is None:
        return True
    feature_set = {str(name) for name in feature_names}
    return set(DEPENDENCE_DESCRIPTOR_KEYS).issubset(feature_set)


def dependence_descriptor_artifact_status(
    manifest_or_metadata: Mapping[str, Any] | None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Explain whether a router artifact may consume dependence descriptors."""

    payload = dict(manifest_or_metadata or {})
    if "training_metadata" in payload or "feature_names" in payload:
        metadata = dict(payload.get("training_metadata", {}) or {})
        names = tuple(str(name) for name in (payload.get("feature_names", feature_names or []) or []))
        artifact_type = str(payload.get("artifact_type", "") or "")
    else:
        metadata = payload
        names = tuple(str(name) for name in (feature_names or []) if str(name))
        artifact_type = ""

    keys = tuple(str(key) for key in (metadata.get("dependence_descriptor_keys", []) or []))
    caps = dict(metadata.get("dependence_descriptor_caps", {}) or {})
    policy = str(metadata.get("dependence_descriptor_policy", "") or "")
    feature_set = {str(name) for name in names}
    missing_features = [key for key in DEPENDENCE_DESCRIPTOR_KEYS if key not in feature_set]

    checks = {
        "policy_matches": policy == DEPENDENCE_DESCRIPTOR_POLICY,
        "keys_match": keys == tuple(DEPENDENCE_DESCRIPTOR_KEYS),
        "caps_match": caps == dict(DEPENDENCE_DESCRIPTOR_CAPS),
        "feature_names_cover_keys": not missing_features,
    }
    enabled = dependence_descriptor_model_input_enabled(metadata, names)
    if enabled:
        reasons: list[str] = []
    else:
        reasons = [name for name, ok in checks.items() if not bool(ok)]

    return {
        "schema_version": DEPENDENCE_DESCRIPTOR_ARTIFACT_STATUS_SCHEMA_VERSION,
        "policy": DEPENDENCE_DESCRIPTOR_POLICY,
        "expected_keys": list(DEPENDENCE_DESCRIPTOR_KEYS),
        "expected_caps": dict(DEPENDENCE_DESCRIPTOR_CAPS),
        "artifact_type": artifact_type,
        "artifact_feature_count": int(len(names)),
        "artifact_dependence_feature_names_present": sorted(feature_set & set(DEPENDENCE_DESCRIPTOR_KEYS)),
        "artifact_missing_dependence_feature_names": missing_features,
        "artifact_training_metadata": {
            "dependence_descriptor_policy": policy,
            "dependence_descriptor_keys": list(keys),
            "dependence_descriptor_caps": caps,
        },
        "checks": checks,
        "model_input_enabled": bool(enabled),
        "status": "enabled" if enabled else "legacy_or_incomplete",
        "fail_reasons": reasons,
    }


def load_dependence_descriptor_artifact_status(manifest_path: str | Path) -> dict[str, Any]:
    """Load a router manifest and return the dependence-descriptor readiness status."""

    path = Path(manifest_path).expanduser()
    if path.is_dir():
        path = path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"router manifest must be a JSON object: {path}")
    status = dependence_descriptor_artifact_status(payload)
    status["manifest_path"] = str(path)
    return status


def _as_2d_float(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    if arr.size == 0:
        return arr.astype(float, copy=False)
    finite = np.isfinite(arr)
    if finite.all():
        return arr.astype(float, copy=False)
    cleaned = arr.astype(float, copy=True)
    masked = np.where(finite, cleaned, np.nan)
    with np.errstate(all="ignore"):
        fill = np.nanmedian(masked, axis=0)
    fill = np.nan_to_num(fill, nan=0.0, posinf=0.0, neginf=0.0)
    rows, cols = np.where(~finite)
    cleaned[rows, cols] = fill[cols]
    return np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)


def _label_codes(y: Any, n: int) -> np.ndarray:
    labels = np.asarray(y, dtype=object).reshape(-1)[:n]
    if labels.size == 0:
        return np.zeros(0, dtype=int)
    keys = np.asarray([repr(value) for value in labels], dtype=object)
    _, inv = np.unique(keys, return_inverse=True)
    return np.asarray(inv, dtype=int)


def _labels_contain_missing_or_nonfinite(y: np.ndarray) -> bool:
    for value in np.asarray(y, dtype=object).reshape(-1).tolist():
        if value is None:
            return True
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return True
    return False


def _feature_variances(X: np.ndarray) -> np.ndarray:
    if X.size == 0 or X.shape[1] == 0:
        return np.zeros(0, dtype=float)
    finite = np.where(np.isfinite(X), X, np.nan)
    with np.errstate(all="ignore"):
        variances = np.nanvar(finite, axis=0)
    return np.nan_to_num(variances, nan=0.0, posinf=0.0, neginf=0.0)


def select_descriptor_feature_indices(
    X: Any,
    *,
    max_features: int = DEPENDENCE_DESCRIPTOR_CAPS["target_mi_max_features"],
    random_state: int = 42,
) -> np.ndarray:
    """Select a deterministic variance-stratified descriptor feature subset."""

    X_arr = _as_2d_float(X)
    p = int(X_arr.shape[1]) if X_arr.ndim == 2 else 0
    cap = int(max(0, max_features))
    if p == 0 or cap == 0:
        return np.zeros(0, dtype=int)
    if p <= cap:
        return np.arange(p, dtype=int)

    variances = _feature_variances(X_arr)
    order = np.argsort(variances, kind="mergesort")
    top_count = int(min(max(1, cap // 10), min(512, cap), p))
    top = order[-top_count:]
    selected: set[int] = {int(idx) for idx in top.tolist()}
    remaining = np.asarray([idx for idx in order[:-top_count] if int(idx) not in selected], dtype=int)
    sample_count = int(cap - len(selected))
    if sample_count <= 0 or remaining.size == 0:
        return np.asarray(sorted(selected), dtype=int)

    rng = np.random.default_rng(int(random_state))
    strata_count = int(min(8, sample_count, remaining.size))
    ranks = np.empty(remaining.size, dtype=int)
    ranks[np.argsort(variances[remaining], kind="mergesort")] = np.arange(remaining.size)
    strata = np.minimum((ranks * strata_count) // max(remaining.size, 1), strata_count - 1)
    quota = int(np.ceil(sample_count / max(strata_count, 1)))
    picked: list[int] = []
    for stratum in range(strata_count):
        group = remaining[strata == stratum]
        if group.size == 0:
            continue
        take = int(min(group.size, quota, sample_count - len(picked)))
        if take <= 0:
            break
        chosen = rng.choice(group, size=take, replace=False)
        picked.extend(int(idx) for idx in np.asarray(chosen, dtype=int).tolist())

    if len(picked) < sample_count:
        already = selected | set(picked)
        leftovers = np.asarray([idx for idx in remaining.tolist() if int(idx) not in already], dtype=int)
        if leftovers.size:
            take = int(min(leftovers.size, sample_count - len(picked)))
            chosen = rng.choice(leftovers, size=take, replace=False)
            picked.extend(int(idx) for idx in np.asarray(chosen, dtype=int).tolist())

    selected.update(int(idx) for idx in picked[:sample_count])
    return np.asarray(sorted(selected), dtype=int)


def _feature_target_mi_values(
    X: np.ndarray,
    y_codes: np.ndarray,
    indices: np.ndarray,
    *,
    max_bins: int,
) -> np.ndarray:
    values: list[float] = []
    for idx in indices.astype(int, copy=False).tolist():
        codes = _safe_rank_bin(X[:, int(idx)], int(max_bins))
        score = _mutual_information_discrete(codes, y_codes)
        if np.isfinite(score):
            values.append(float(max(0.0, score)))
    return np.asarray(values, dtype=float)


def _sample_pair_indices(n_features: int, *, max_pairs: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    if n_features < 2 or max_pairs <= 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    i, j = np.triu_indices(int(n_features), k=1)
    if i.size <= int(max_pairs):
        return i.astype(int, copy=False), j.astype(int, copy=False)
    rng = np.random.default_rng(int(random_state))
    budget = int(max_pairs)
    spans = j - i
    strata_count = int(min(8, budget, np.unique(spans).size))
    if strata_count <= 1:
        chosen = np.sort(rng.choice(i.size, size=budget, replace=False))
        return i[chosen].astype(int, copy=False), j[chosen].astype(int, copy=False)

    span_order = np.argsort(spans, kind="mergesort")
    span_rank = np.empty(i.size, dtype=int)
    span_rank[span_order] = np.arange(i.size, dtype=int)
    strata = np.minimum((span_rank * strata_count) // max(i.size, 1), strata_count - 1)
    quota = int(np.ceil(budget / max(strata_count, 1)))
    picked: list[int] = []
    for stratum in range(strata_count):
        group = np.flatnonzero(strata == stratum)
        if group.size == 0:
            continue
        take = int(min(group.size, quota, budget - len(picked)))
        if take <= 0:
            break
        picked.extend(int(idx) for idx in rng.choice(group, size=take, replace=False).tolist())
    if len(picked) < budget:
        used = set(picked)
        leftovers = np.asarray([idx for idx in range(i.size) if idx not in used], dtype=int)
        if leftovers.size:
            take = int(min(leftovers.size, budget - len(picked)))
            picked.extend(int(idx) for idx in rng.choice(leftovers, size=take, replace=False).tolist())
    chosen = np.sort(np.asarray(picked[:budget], dtype=int))
    return i[chosen].astype(int, copy=False), j[chosen].astype(int, copy=False)


def _maximum_dependency_forest_weight(n_nodes: int, edges: list[tuple[float, int, int]]) -> float:
    if n_nodes < 2 or not edges:
        return 0.0
    parent = list(range(int(n_nodes)))
    rank = [0 for _ in range(int(n_nodes))]

    def _find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def _union(left: int, right: int) -> bool:
        root_left = _find(left)
        root_right = _find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        return True

    total = 0.0
    edge_count = 0
    for weight, left, right in sorted(edges, key=lambda item: (-item[0], item[1], item[2])):
        if _union(int(left), int(right)):
            total += float(max(0.0, weight))
            edge_count += 1
            if edge_count >= int(n_nodes) - 1:
                break
    return float(total)


def _pairwise_redundancy_ratio(
    X: np.ndarray,
    y_codes: np.ndarray,
    *,
    max_features: int,
    max_pairs: int,
    max_bins: int,
    random_state: int,
) -> float:
    indices = select_descriptor_feature_indices(
        X,
        max_features=int(max_features),
        random_state=int(random_state) + 17,
    )
    if indices.size < 2:
        return 0.0
    target_mi = _feature_target_mi_values(X, y_codes, indices, max_bins=int(max_bins))
    mean_target_mi = float(np.mean(target_mi)) if target_mi.size else 0.0

    codes = [
        _safe_rank_bin(X[:, int(idx)], int(max_bins))
        for idx in indices.astype(int, copy=False).tolist()
    ]
    left, right = _sample_pair_indices(
        len(codes),
        max_pairs=int(max_pairs),
        random_state=int(random_state) + 31,
    )
    edges: list[tuple[float, int, int]] = []
    for i, j in zip(left.tolist(), right.tolist()):
        score = _mutual_information_discrete(codes[int(i)], codes[int(j)])
        if np.isfinite(score):
            weight = float(max(0.0, score))
            edges.append((weight, int(i), int(j)))
    forest_weight = _maximum_dependency_forest_weight(len(codes), edges)
    target_weight = float(mean_target_mi * len(codes))
    denom = forest_weight + target_weight
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(forest_weight / denom, 0.0, 1.0))


def compute_dependence_descriptors(
    X: Any,
    y: Any,
    *,
    target_mi_max_features: int = DEPENDENCE_DESCRIPTOR_CAPS["target_mi_max_features"],
    pairwise_max_features: int = DEPENDENCE_DESCRIPTOR_CAPS["pairwise_max_features"],
    pairwise_max_pairs: int = DEPENDENCE_DESCRIPTOR_CAPS["pairwise_max_pairs"],
    max_bins: int = DEPENDENCE_DESCRIPTOR_CAPS["max_bins"],
    random_state: int = 42,
) -> dict[str, float]:
    """Compute bounded binned MI descriptors for router training and runtime."""

    raw_X = np.asarray(X)
    if raw_X.ndim != 2:
        return {key: 0.0 for key in DEPENDENCE_DESCRIPTOR_KEYS}
    X_arr = _as_2d_float(raw_X)
    raw_y = np.asarray(y, dtype=object)
    if raw_y.ndim != 1:
        return {key: 0.0 for key in DEPENDENCE_DESCRIPTOR_KEYS}
    if _labels_contain_missing_or_nonfinite(raw_y):
        return {key: 0.0 for key in DEPENDENCE_DESCRIPTOR_KEYS}
    y_arr = raw_y
    if int(X_arr.shape[0]) != int(y_arr.size):
        return {key: 0.0 for key in DEPENDENCE_DESCRIPTOR_KEYS}
    n = int(X_arr.shape[0])
    if n <= 1 or X_arr.shape[1] == 0:
        return {key: 0.0 for key in DEPENDENCE_DESCRIPTOR_KEYS}
    X_arr = X_arr[:n, :]
    y_codes = _label_codes(y_arr, n)
    if np.unique(y_codes).size <= 1:
        return {key: 0.0 for key in DEPENDENCE_DESCRIPTOR_KEYS}

    target_indices = select_descriptor_feature_indices(
        X_arr,
        max_features=int(target_mi_max_features),
        random_state=int(random_state),
    )
    mi_values = _feature_target_mi_values(
        X_arr,
        y_codes,
        target_indices,
        max_bins=int(max_bins),
    )
    mean_mi = float(np.mean(mi_values)) if mi_values.size else 0.0
    std_mi = float(np.std(mi_values)) if mi_values.size else 0.0
    redundancy = _pairwise_redundancy_ratio(
        X_arr,
        y_codes,
        max_features=int(pairwise_max_features),
        max_pairs=int(pairwise_max_pairs),
        max_bins=int(max_bins),
        random_state=int(random_state),
    )
    return {
        "mutual_info_mean": float(max(0.0, mean_mi)),
        "mutual_info_std": float(max(0.0, std_mi)),
        "pairwise_redundancy_ratio": float(np.clip(redundancy, 0.0, 1.0)),
    }

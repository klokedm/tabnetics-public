"""Support-only missingness descriptors for auto-router decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


MISSINGNESS_DESCRIPTOR_POLICY = "support_mask_summary_v1"
MISSINGNESS_DESCRIPTOR_KEYS: tuple[str, ...] = (
    "mean_missing_fraction",
    "missingness_column_concentration",
    "missingness_monotone_flag",
)
MISSINGNESS_DESCRIPTOR_ARTIFACT_STATUS_SCHEMA_VERSION = (
    "missingness_descriptor_artifact_status_v1"
)


def missingness_descriptor_model_input_enabled(
    metadata: Mapping[str, Any] | None,
    feature_names: Sequence[str] | None = None,
) -> bool:
    """Return true only for artifacts trained with the exact mask contract."""

    payload = dict(metadata or {})
    if str(payload.get("missingness_descriptor_policy", "") or "") != MISSINGNESS_DESCRIPTOR_POLICY:
        return False
    keys = tuple(str(key) for key in (payload.get("missingness_descriptor_keys", []) or []))
    if keys != MISSINGNESS_DESCRIPTOR_KEYS:
        return False
    if feature_names is None:
        return True
    return set(MISSINGNESS_DESCRIPTOR_KEYS).issubset(str(name) for name in feature_names)


def missingness_descriptor_artifact_status(
    manifest_or_metadata: Mapping[str, Any] | None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Explain whether a router artifact may consume missingness descriptors."""

    payload = dict(manifest_or_metadata or {})
    if "training_metadata" in payload or "feature_names" in payload:
        metadata = dict(payload.get("training_metadata", {}) or {})
        names = tuple(
            str(name)
            for name in (payload.get("feature_names", feature_names or []) or [])
        )
    else:
        metadata = payload
        names = tuple(str(name) for name in (feature_names or []) if str(name))
    policy = str(metadata.get("missingness_descriptor_policy", "") or "")
    keys = tuple(
        str(key) for key in (metadata.get("missingness_descriptor_keys", []) or [])
    )
    feature_set = set(names)
    missing_features = [key for key in MISSINGNESS_DESCRIPTOR_KEYS if key not in feature_set]
    checks = {
        "policy_matches": policy == MISSINGNESS_DESCRIPTOR_POLICY,
        "keys_match": keys == MISSINGNESS_DESCRIPTOR_KEYS,
        "feature_names_cover_keys": not missing_features,
    }
    enabled = missingness_descriptor_model_input_enabled(metadata, names)
    return {
        "schema_version": MISSINGNESS_DESCRIPTOR_ARTIFACT_STATUS_SCHEMA_VERSION,
        "policy": MISSINGNESS_DESCRIPTOR_POLICY,
        "expected_keys": list(MISSINGNESS_DESCRIPTOR_KEYS),
        "artifact_feature_count": len(names),
        "artifact_missing_missingness_feature_names": missing_features,
        "checks": checks,
        "model_input_enabled": bool(enabled),
        "status": "enabled" if enabled else "legacy_or_incomplete",
        "fail_reasons": [name for name, ok in checks.items() if not ok],
    }


def compute_missingness_descriptors(X: Any) -> dict[str, float]:
    """Summarize missing support values without inspecting labels or query rows."""

    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.size == 0:
        return {key: 0.0 for key in MISSINGNESS_DESCRIPTOR_KEYS}

    mask = ~np.isfinite(values)
    n_rows, n_features = mask.shape
    missing_count = int(mask.sum())
    mean_fraction = float(missing_count / max(n_rows * n_features, 1))

    per_feature = mask.sum(axis=0, dtype=np.float64)
    if missing_count <= 0 or n_features <= 1:
        column_concentration = 0.0
    else:
        shares = per_feature / float(missing_count)
        hhi = float(np.sum(np.square(shares)))
        uniform_hhi = 1.0 / float(n_features)
        column_concentration = float(
            np.clip((hhi - uniform_hhi) / max(1.0 - uniform_hhi, 1e-12), 0.0, 1.0)
        )

    incomplete_masks = mask[mask.any(axis=1)]
    if incomplete_masks.shape[0] == 0:
        monotone_flag = 0.0
    else:
        patterns = np.unique(incomplete_masks, axis=0)
        order = np.argsort(patterns.sum(axis=1), kind="stable")
        ordered = patterns[order]
        monotone_flag = float(
            all(
                bool(np.all(lower <= upper))
                for lower, upper in zip(ordered[:-1], ordered[1:])
            )
        )

    return {
        "mean_missing_fraction": mean_fraction,
        "missingness_column_concentration": column_concentration,
        "missingness_monotone_flag": monotone_flag,
    }

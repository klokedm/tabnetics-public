"""Frozen vnext-C DIAKRINO feature-selection closeout mechanics.

This module deliberately contains no DIAKRINO model invocation.  Producers pass the
rank vectors emitted from support-only real, shuffled-shadow, and label-null
views; consumers rederive admission, abstention, JMI arbitration, and matched
controls from those sealed vectors.  All behavior is opt-in through an explicit
``DiakrinoCloseoutConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


DIAKRINO_CLOSEOUT_ARMS = frozenset(
    {
        "protected_native_null_abstain",
        "protected_native_null_jmi",
        "classical_next_best",
        "random_extras",
        "permuted_diakrino_ranks",
    }
)
DIAKRINO_NATIVE_NULL_SCHEMA_VERSION = "diakrino_native_null_artifact_v1"
DIAKRINO_NATIVE_NULL_PAIRED_SCHEMA_VERSION = "diakrino_native_null_artifact_v2"
DIAKRINO_NATIVE_NULL_SCORE_SOURCE = "uniform_five_view_rank01_native_null_v1"
DIAKRINO_NATIVE_NULL_PAIRED_SCORE_SOURCE = "paired_real_shadow_uniform_rank01_v2"


class DiakrinoCloseoutError(ValueError):
    """Raised when closeout evidence is missing, cross-wired, or malformed."""


@dataclass(frozen=True)
class DiakrinoCloseoutConfig:
    """Mechanics-only thresholds frozen before downstream outcomes are read."""

    shadow_margin_min: float = 0.0
    label_null_margin_min: float = 0.0
    rank_std_max: float = 0.25
    selected_set_stability_min: float = 0.0
    proposal_pool_multiplier: int = 4
    discretization_bins: int = 5

    def validate(self) -> None:
        for name in (
            "shadow_margin_min",
            "label_null_margin_min",
            "rank_std_max",
            "selected_set_stability_min",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise DiakrinoCloseoutError(f"{name} must be finite")
        if not 0.0 <= float(self.rank_std_max) <= 1.0:
            raise DiakrinoCloseoutError("rank_std_max must be within [0, 1]")
        if not -1.0 <= float(self.selected_set_stability_min) <= 1.0:
            raise DiakrinoCloseoutError(
                "selected_set_stability_min must be within [-1, 1]"
            )
        if int(self.proposal_pool_multiplier) < 1:
            raise DiakrinoCloseoutError("proposal_pool_multiplier must be positive")
        if int(self.discretization_bins) < 2:
            raise DiakrinoCloseoutError("discretization_bins must be at least two")


@dataclass(frozen=True)
class NativeNullDesign:
    """Support-only transformations for one deterministic producer emission."""

    shadow_support: np.ndarray
    label_null: np.ndarray
    shadow_permutations: tuple[tuple[int, ...], ...]
    label_permutation: tuple[int, ...]


@dataclass(frozen=True)
class DiakrinoCloseoutDecision:
    arm: str
    protected_core: tuple[int, ...]
    additions: tuple[int, ...]
    final: tuple[int, ...]
    addition_budget: int
    realized_additions: int
    abstained: bool
    fallback_exact: bool
    reason: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class ValidatedNativeNullArtifact:
    binding_sha256: str
    seed: int
    real_rank: tuple[float, ...]
    shadow_rank: tuple[float, ...]
    label_null_rank: tuple[float, ...]
    rank_std: tuple[float, ...]
    view_ids: tuple[str, ...]
    ranks_by_view: tuple[tuple[float, ...], ...]


def dynamic_addition_budget(protected_core_size: int) -> int:
    size = int(protected_core_size)
    if size < 1:
        raise DiakrinoCloseoutError("protected core must contain at least one feature")
    return min(10, max(1, int(math.ceil(0.25 * size))))


def _seed(base_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(
        f"diakrino-closeout-v1:{int(base_seed)}:{namespace}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _permutation(size: int, *, seed: int, namespace: str) -> np.ndarray:
    if size < 2:
        raise DiakrinoCloseoutError("null diagnostics require at least two support rows")
    rng = np.random.Generator(np.random.PCG64(_seed(seed, namespace)))
    permutation = rng.permutation(size).astype(np.int64, copy=False)
    if np.array_equal(permutation, np.arange(size, dtype=np.int64)):
        permutation = np.roll(permutation, 1)
    return permutation


def build_support_only_null_design(
    X_support: np.ndarray,
    y_support: np.ndarray,
    *,
    seed: int,
) -> NativeNullDesign:
    """Construct independent row-shuffled shadows and a class-count null.

    The API accepts support arrays only.  Each shadow column receives its own
    deterministic row permutation.  The label null is a permutation of support
    labels, so its class multiset is exactly preserved.
    """

    X = np.asarray(X_support)
    y = np.asarray(y_support).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[1] < 1:
        raise DiakrinoCloseoutError("support X/y dimensions are inconsistent")
    if X.shape[0] < 2 or np.unique(y).size < 2:
        raise DiakrinoCloseoutError("native nulls require at least two rows and classes")
    shadow = np.empty_like(X)
    permutations: list[tuple[int, ...]] = []
    for feature_index in range(X.shape[1]):
        permutation = _permutation(
            X.shape[0], seed=seed, namespace=f"shadow:{feature_index}"
        )
        shadow[:, feature_index] = X[permutation, feature_index]
        permutations.append(tuple(int(value) for value in permutation.tolist()))
    label_permutation = _permutation(X.shape[0], seed=seed, namespace="label-null")
    label_null = y[label_permutation]
    if np.array_equal(label_null, y):
        for attempt in range(1, 17):
            candidate = _permutation(
                X.shape[0], seed=seed, namespace=f"label-null:{attempt}"
            )
            if not np.array_equal(y[candidate], y):
                label_permutation = candidate
                label_null = y[candidate]
                break
        else:
            # With at least two classes some cyclic row shift changes at least
            # one assignment.  This deterministic fallback avoids a false null.
            for shift in range(1, X.shape[0]):
                candidate = np.roll(np.arange(X.shape[0], dtype=np.int64), shift)
                if not np.array_equal(y[candidate], y):
                    label_permutation = candidate
                    label_null = y[candidate]
                    break
    if not np.array_equal(np.sort(label_null), np.sort(y)):
        raise AssertionError("label permutation did not preserve class counts")
    return NativeNullDesign(
        shadow_support=shadow,
        label_null=label_null,
        shadow_permutations=tuple(permutations),
        label_permutation=tuple(int(value) for value in label_permutation.tolist()),
    )


def _binding_sha256(value: str) -> str:
    rendered = str(value or "").strip().lower()
    if len(rendered) != 64 or any(char not in "0123456789abcdef" for char in rendered):
        raise DiakrinoCloseoutError("binding_sha256 must be lowercase SHA-256")
    return rendered


def build_native_null_artifact(
    *,
    binding_sha256: str,
    X_support: np.ndarray,
    y_support: np.ndarray,
    seed: int,
    real_rank: Sequence[float],
    shadow_rank: Sequence[float],
    label_null_rank: Sequence[float],
    rank_std: Sequence[float],
    view_ids: Sequence[str],
    ranks_by_view: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Seal producer-owned native-null scores and support-only recipes."""

    binding = _binding_sha256(binding_sha256)
    X = np.asarray(X_support)
    y = np.asarray(y_support).reshape(-1)
    design = build_support_only_null_design(X, y, seed=int(seed))
    width = int(X.shape[1])
    real = _vector(real_rank, name="real_rank", width=width)
    shadow = _vector(shadow_rank, name="shadow_rank", width=width)
    label_null = _vector(label_null_rank, name="label_null_rank", width=width)
    dispersion = _vector(rank_std, name="rank_std", width=width)
    ids = tuple(str(value) for value in view_ids)
    matrix = np.asarray(ranks_by_view, dtype=np.float64)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or matrix.shape != (len(ids), width)
        or not np.all(np.isfinite(matrix))
    ):
        raise DiakrinoCloseoutError("native-null view ranks are missing or malformed")
    if (
        np.any((real < 0.0) | (real > 1.0))
        or np.any((shadow < 0.0) | (shadow > 1.0))
        or np.any((label_null < 0.0) | (label_null > 1.0))
        or np.any(dispersion < 0.0)
        or np.any((matrix < 0.0) | (matrix > 1.0))
    ):
        raise DiakrinoCloseoutError("native-null ranks or dispersion are out of range")
    artifact: dict[str, object] = {
        "schema_version": DIAKRINO_NATIVE_NULL_SCHEMA_VERSION,
        "binding_sha256": binding,
        "seed": int(seed),
        "n_support": int(X.shape[0]),
        "n_features": width,
        "transformations": {
            "shadow_kind": "independent_support_row_permutation_per_feature",
            "label_null_kind": "support_label_permutation_class_counts_preserved",
            "shadow_permutations": [list(row) for row in design.shadow_permutations],
            "label_permutation": list(design.label_permutation),
        },
        "scores": {
            "score_source": {
                "id": DIAKRINO_NATIVE_NULL_SCORE_SOURCE,
                "calibration": "chunk_zscore_then_rank01_per_view",
                "aggregation": "uniform_mean_five_frozen_views",
                "shadow_pairing": "real_and_shadow_in_same_panel_view",
            },
            "real_uniform_rank01": real.tolist(),
            "shadow_uniform_rank01": shadow.tolist(),
            "label_null_uniform_rank01": label_null.tolist(),
            "real_uniform_rank_std": dispersion.tolist(),
            "view_ids": list(ids),
            "real_rank01_by_view": matrix.tolist(),
        },
    }
    artifact["semantic_sha256"] = hashlib.sha256(
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return artifact


def validate_native_null_artifact(
    payload: Mapping[str, object],
    *,
    binding_sha256: str,
    X_support: np.ndarray,
    y_support: np.ndarray,
    seed: int,
    expected_real_rank: Sequence[float],
    expected_rank_std: Sequence[float],
    expected_view_ids: Sequence[str],
    expected_ranks_by_view: Sequence[Sequence[float]],
) -> ValidatedNativeNullArtifact:
    """Validate a native-null artifact against canonical support and views."""

    expected_keys = {
        "schema_version",
        "binding_sha256",
        "seed",
        "n_support",
        "n_features",
        "transformations",
        "scores",
        "semantic_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise DiakrinoCloseoutError("native-null artifact fields are missing or malformed")
    binding = _binding_sha256(binding_sha256)
    X = np.asarray(X_support)
    y = np.asarray(y_support).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise DiakrinoCloseoutError("canonical support X/y dimensions are invalid")
    if (
        payload["schema_version"] != DIAKRINO_NATIVE_NULL_SCHEMA_VERSION
        or payload["binding_sha256"] != binding
        or type(payload["seed"]) is not int
        or int(payload["seed"]) != int(seed)
        or type(payload["n_support"]) is not int
        or type(payload["n_features"]) is not int
        or payload["n_support"] != int(X.shape[0])
        or payload["n_features"] != int(X.shape[1])
    ):
        raise DiakrinoCloseoutError("native-null artifact identity is cross-wired")
    expected_design = build_support_only_null_design(X, y, seed=int(seed))
    transformations = payload["transformations"]
    if not isinstance(transformations, Mapping) or set(transformations) != {
        "shadow_kind",
        "label_null_kind",
        "shadow_permutations",
        "label_permutation",
    }:
        raise DiakrinoCloseoutError("native-null transformation ledger is malformed")
    if (
        transformations["shadow_kind"]
        != "independent_support_row_permutation_per_feature"
        or transformations["label_null_kind"]
        != "support_label_permutation_class_counts_preserved"
        or transformations["shadow_permutations"]
        != [list(row) for row in expected_design.shadow_permutations]
        or transformations["label_permutation"]
        != list(expected_design.label_permutation)
    ):
        raise DiakrinoCloseoutError("native-null transformations are not support-only")
    scores = payload["scores"]
    if not isinstance(scores, Mapping) or set(scores) != {
        "score_source",
        "real_uniform_rank01",
        "shadow_uniform_rank01",
        "label_null_uniform_rank01",
        "real_uniform_rank_std",
        "view_ids",
        "real_rank01_by_view",
    }:
        raise DiakrinoCloseoutError("native-null score ledger is malformed")
    if scores["score_source"] != {
        "id": DIAKRINO_NATIVE_NULL_SCORE_SOURCE,
        "calibration": "chunk_zscore_then_rank01_per_view",
        "aggregation": "uniform_mean_five_frozen_views",
        "shadow_pairing": "real_and_shadow_in_same_panel_view",
    }:
        raise DiakrinoCloseoutError("native-null score-source identity is invalid")
    semantic_payload = dict(payload)
    declared_semantic_sha256 = semantic_payload.pop("semantic_sha256")
    expected_semantic_sha256 = hashlib.sha256(
        json.dumps(
            semantic_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        not isinstance(declared_semantic_sha256, str)
        or declared_semantic_sha256 != expected_semantic_sha256
    ):
        raise DiakrinoCloseoutError("native-null semantic SHA-256 is invalid")
    width = int(X.shape[1])
    real = _vector(scores["real_uniform_rank01"], name="real_rank", width=width)
    shadow = _vector(scores["shadow_uniform_rank01"], name="shadow_rank", width=width)
    label_null = _vector(
        scores["label_null_uniform_rank01"], name="label_null_rank", width=width
    )
    dispersion = _vector(
        scores["real_uniform_rank_std"], name="rank_std", width=width
    )
    expected_real = _vector(
        expected_real_rank, name="expected_real_rank", width=width
    )
    expected_dispersion = _vector(
        expected_rank_std, name="expected_rank_std", width=width
    )
    ids = tuple(str(value) for value in scores["view_ids"])
    expected_ids = tuple(str(value) for value in expected_view_ids)
    matrix = np.asarray(scores["real_rank01_by_view"], dtype=np.float64)
    expected_matrix = np.asarray(expected_ranks_by_view, dtype=np.float64)
    if (
        np.any((real < 0.0) | (real > 1.0))
        or np.any((shadow < 0.0) | (shadow > 1.0))
        or np.any((label_null < 0.0) | (label_null > 1.0))
        or np.any(dispersion < 0.0)
        or np.any((matrix < 0.0) | (matrix > 1.0))
    ):
        raise DiakrinoCloseoutError("native-null ranks or dispersion are out of range")
    if not (
        np.array_equal(real, expected_real)
        and np.array_equal(dispersion, expected_dispersion)
        and ids == expected_ids
        and matrix.shape == (len(ids), width)
        and np.all(np.isfinite(matrix))
        and np.array_equal(matrix, expected_matrix)
    ):
        raise DiakrinoCloseoutError("native-null real-view evidence is cross-wired")
    return ValidatedNativeNullArtifact(
        binding_sha256=binding,
        seed=int(seed),
        real_rank=tuple(float(value) for value in real.tolist()),
        shadow_rank=tuple(float(value) for value in shadow.tolist()),
        label_null_rank=tuple(float(value) for value in label_null.tolist()),
        rank_std=tuple(float(value) for value in dispersion.tolist()),
        view_ids=ids,
        ranks_by_view=tuple(
            tuple(float(value) for value in row.tolist()) for row in matrix
        ),
    )


def _paired_rank_matrix(
    values: Sequence[Sequence[float]], *, name: str, n_views: int, width: int
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (n_views, width) or not np.all(np.isfinite(matrix)):
        raise DiakrinoCloseoutError(f"{name} is missing, non-finite, or misaligned")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise DiakrinoCloseoutError(f"{name} is out of range")
    return matrix


def _paired_rank_statistics(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0, dtype=np.float64)
    std = np.sqrt(np.mean((matrix - mean[None, :]) ** 2, axis=0, dtype=np.float64))
    return mean, std


def _paired_panel_records(value: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    covered: list[int] = []
    for expected_chunk, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {
            "chunk_id", "original_feature_indices", "real_slots", "shadow_slots"
        }:
            raise DiakrinoCloseoutError("paired panel layout is malformed")
        if type(raw["chunk_id"]) is not int or int(raw["chunk_id"]) != expected_chunk:
            raise DiakrinoCloseoutError("paired panel chunk ids are malformed")
        original = [int(item) for item in list(raw["original_feature_indices"])]
        real = [int(item) for item in list(raw["real_slots"])]
        shadow = [int(item) for item in list(raw["shadow_slots"])]
        if (
            not original
            or len(original) != len(set(original))
            or any(item < 0 for item in original)
            or real != list(range(0, 2 * len(original), 2))
            or shadow != list(range(1, 2 * len(original), 2))
        ):
            raise DiakrinoCloseoutError("paired panel slot map is malformed")
        covered.extend(original)
        records.append(
            {
                "chunk_id": expected_chunk,
                "original_feature_indices": original,
                "real_slots": real,
                "shadow_slots": shadow,
            }
        )
    if sorted(covered) != list(range(len(covered))):
        raise DiakrinoCloseoutError("paired panel layout does not cover original features")
    return records


def build_paired_native_null_artifact(
    *,
    binding_sha256: str,
    X_support: np.ndarray,
    y_support: np.ndarray,
    seed: int,
    view_ids: Sequence[str],
    paired_view_artifact_sha256: str,
    panel_chunks_by_view: Mapping[str, Sequence[Mapping[str, object]]],
    real_rank01_by_view: Sequence[Sequence[float]],
    shadow_rank01_by_view: Sequence[Sequence[float]],
    label_null_rank01_by_view: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Seal the paired P2/P3/P4 null ledger from same-panel DIAKRINO outputs."""

    binding = _binding_sha256(binding_sha256)
    X = np.asarray(X_support)
    y = np.asarray(y_support).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise DiakrinoCloseoutError("canonical support X/y dimensions are invalid")
    view_names = tuple(str(item) for item in view_ids)
    if not view_names or len(view_names) != len(set(view_names)):
        raise DiakrinoCloseoutError("paired native-null view ids are malformed")
    paired_digest = _binding_sha256(paired_view_artifact_sha256)
    width = int(X.shape[1])
    if tuple(panel_chunks_by_view) != view_names:
        raise DiakrinoCloseoutError("paired panel layouts must match frozen view order")
    panels_by_view = {
        view_id: _paired_panel_records(panel_chunks_by_view[view_id])
        for view_id in view_names
    }
    if any(
        sum(len(item["original_feature_indices"]) for item in panels) != width
        for panels in panels_by_view.values()
    ):
        raise DiakrinoCloseoutError("paired panel width is inconsistent with support")
    real = _paired_rank_matrix(real_rank01_by_view, name="paired real ranks", n_views=len(view_names), width=width)
    shadow = _paired_rank_matrix(shadow_rank01_by_view, name="paired shadow ranks", n_views=len(view_names), width=width)
    label_null = _paired_rank_matrix(label_null_rank01_by_view, name="paired label-null ranks", n_views=len(view_names), width=width)
    design = build_support_only_null_design(X, y, seed=int(seed))
    aggregate, dispersion = _paired_rank_statistics(real)
    artifact: dict[str, object] = {
        "schema_version": DIAKRINO_NATIVE_NULL_PAIRED_SCHEMA_VERSION,
        "binding_sha256": binding,
        "seed": int(seed),
        "n_support": int(X.shape[0]),
        "n_features": width,
        "panel": {
            "mode": "paired_real_shadow_v1",
            "calibration_population": "joint_real_shadow_slots_per_chunk",
            "ranking_population": "joint_real_shadow_slots_per_view",
            "paired_view_artifact_sha256": paired_digest,
            "layouts_by_view": panels_by_view,
        },
        "transformations": {
            "shadow_kind": "independent_support_row_permutation_per_feature",
            "label_null_kind": "support_label_permutation_class_counts_preserved",
            "shadow_permutations": [list(row) for row in design.shadow_permutations],
            "label_permutation": list(design.label_permutation),
        },
        "scores": {
            "score_source": DIAKRINO_NATIVE_NULL_PAIRED_SCORE_SOURCE,
            "view_ids": list(view_names),
            "real_rank01_by_view": real.tolist(),
            "shadow_rank01_by_view": shadow.tolist(),
            "label_null_rank01_by_view": label_null.tolist(),
            "real_uniform_rank01": aggregate.tolist(),
            "real_uniform_rank_std": dispersion.tolist(),
        },
    }
    artifact["semantic_sha256"] = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return artifact


def validate_paired_native_null_artifact(
    payload: Mapping[str, object],
    *,
    binding_sha256: str,
    X_support: np.ndarray,
    y_support: np.ndarray,
    seed: int,
    expected_paired_view_artifact_sha256: str,
    expected_panel_chunks_by_view: Mapping[str, Sequence[Mapping[str, object]]],
    expected_real_rank01_by_view: Sequence[Sequence[float]],
) -> ValidatedNativeNullArtifact:
    """Fail closed unless v2 paired evidence exactly reproduces the P2 ledger."""

    required = {"schema_version", "binding_sha256", "seed", "n_support", "n_features", "panel", "transformations", "scores", "semantic_sha256"}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise DiakrinoCloseoutError("paired native-null artifact fields are malformed")
    X = np.asarray(X_support)
    y = np.asarray(y_support).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise DiakrinoCloseoutError("canonical support X/y dimensions are invalid")
    if (
        payload["schema_version"] != DIAKRINO_NATIVE_NULL_PAIRED_SCHEMA_VERSION
        or payload["binding_sha256"] != _binding_sha256(binding_sha256)
        or type(payload["seed"]) is not int or int(payload["seed"]) != int(seed)
        or payload["n_support"] != int(X.shape[0]) or payload["n_features"] != int(X.shape[1])
    ):
        raise DiakrinoCloseoutError("paired native-null artifact identity is cross-wired")
    semantic = dict(payload)
    declared = semantic.pop("semantic_sha256")
    digest = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
    if not isinstance(declared, str) or declared != digest:
        raise DiakrinoCloseoutError("paired native-null semantic SHA-256 is invalid")
    panel = payload["panel"]
    if not isinstance(panel, Mapping) or set(panel) != {"mode", "calibration_population", "ranking_population", "paired_view_artifact_sha256", "layouts_by_view"}:
        raise DiakrinoCloseoutError("paired native-null panel is malformed")
    if panel["mode"] != "paired_real_shadow_v1" or panel["calibration_population"] != "joint_real_shadow_slots_per_chunk" or panel["ranking_population"] != "joint_real_shadow_slots_per_view" or panel["paired_view_artifact_sha256"] != _binding_sha256(expected_paired_view_artifact_sha256):
        raise DiakrinoCloseoutError("paired native-null panel identity is invalid")
    raw_layouts = panel["layouts_by_view"]
    if not isinstance(raw_layouts, Mapping) or tuple(raw_layouts) != tuple(expected_panel_chunks_by_view):
        raise DiakrinoCloseoutError("paired native-null panel is cross-wired")
    if any(
        _paired_panel_records(raw_layouts[view_id])
        != _paired_panel_records(expected_panel_chunks_by_view[view_id])
        for view_id in raw_layouts
    ):
        raise DiakrinoCloseoutError("paired native-null panel is cross-wired")
    design = build_support_only_null_design(X, y, seed=int(seed))
    expected_transformations = {"shadow_kind": "independent_support_row_permutation_per_feature", "label_null_kind": "support_label_permutation_class_counts_preserved", "shadow_permutations": [list(row) for row in design.shadow_permutations], "label_permutation": list(design.label_permutation)}
    if payload["transformations"] != expected_transformations:
        raise DiakrinoCloseoutError("paired native-null transformations are not support-only")
    scores = payload["scores"]
    keys = {"score_source", "view_ids", "real_rank01_by_view", "shadow_rank01_by_view", "label_null_rank01_by_view", "real_uniform_rank01", "real_uniform_rank_std"}
    if not isinstance(scores, Mapping) or set(scores) != keys or scores["score_source"] != DIAKRINO_NATIVE_NULL_PAIRED_SCORE_SOURCE:
        raise DiakrinoCloseoutError("paired native-null score ledger is malformed")
    views = tuple(str(item) for item in scores["view_ids"])
    width = int(X.shape[1])
    real = _paired_rank_matrix(scores["real_rank01_by_view"], name="paired real ranks", n_views=len(views), width=width)
    shadow = _paired_rank_matrix(scores["shadow_rank01_by_view"], name="paired shadow ranks", n_views=len(views), width=width)
    label = _paired_rank_matrix(scores["label_null_rank01_by_view"], name="paired label-null ranks", n_views=len(views), width=width)
    expected_real = _paired_rank_matrix(expected_real_rank01_by_view, name="expected paired real ranks", n_views=len(views), width=width)
    aggregate, dispersion = _paired_rank_statistics(real)
    if not (np.array_equal(real, expected_real) and np.array_equal(_vector(scores["real_uniform_rank01"], name="paired aggregate", width=width), aggregate) and np.array_equal(_vector(scores["real_uniform_rank_std"], name="paired dispersion", width=width), dispersion)):
        raise DiakrinoCloseoutError("paired native-null real-view evidence is cross-wired")
    return ValidatedNativeNullArtifact(binding_sha256=_binding_sha256(binding_sha256), seed=int(seed), real_rank=tuple(aggregate.tolist()), shadow_rank=tuple(np.mean(shadow, axis=0).tolist()), label_null_rank=tuple(np.mean(label, axis=0).tolist()), rank_std=tuple(dispersion.tolist()), view_ids=views, ranks_by_view=tuple(tuple(row.tolist()) for row in real))


def nogueira_selected_set_stability(
    selected_sets: Sequence[Iterable[int]], *, n_features: int
) -> float | None:
    """Chance-corrected Nogueira stability using the full feature width.

    The feature denominator is ``n_features``, never the selected-set union.
    Returns ``None`` when fewer than two sets or a degenerate all/none selection
    makes the chance correction undefined.
    """

    width = int(n_features)
    if width < 1:
        raise DiakrinoCloseoutError("n_features must be positive")
    sets = [set(int(value) for value in values) for values in selected_sets]
    if len(sets) < 2:
        return None
    if any(value < 0 or value >= width for values in sets for value in values):
        raise DiakrinoCloseoutError("selected set contains an out-of-range feature")
    matrix = np.zeros((len(sets), width), dtype=np.float64)
    for row_index, values in enumerate(sets):
        if values:
            matrix[row_index, sorted(values)] = 1.0
    mean_size = float(np.mean(np.sum(matrix, axis=1)))
    expected = (mean_size / width) * (1.0 - mean_size / width)
    if expected <= 0.0:
        return None
    feature_variances = np.var(matrix, axis=0, ddof=1)
    return float(1.0 - float(np.mean(feature_variances)) / expected)


def _vector(values: Sequence[float], *, name: str, width: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (width,) or not np.all(np.isfinite(vector)):
        raise DiakrinoCloseoutError(f"{name} is missing, non-finite, or misaligned")
    return vector


def native_null_proposals(
    *,
    real_rank: Sequence[float],
    shadow_rank: Sequence[float],
    label_null_rank: Sequence[float],
    rank_std: Sequence[float],
    ranks_by_view: Sequence[Sequence[float]],
    protected_core: Sequence[int],
    config: DiakrinoCloseoutConfig,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Apply predeclared real/null/dispersion/stability proposal gates."""

    config.validate()
    width = len(real_rank)
    if width < 1:
        raise DiakrinoCloseoutError("real_rank must not be empty")
    real = _vector(real_rank, name="real_rank", width=width)
    shadow = _vector(shadow_rank, name="shadow_rank", width=width)
    label_null = _vector(label_null_rank, name="label_null_rank", width=width)
    dispersion = _vector(rank_std, name="rank_std", width=width)
    views = np.asarray(ranks_by_view, dtype=np.float64)
    if (
        views.ndim != 2
        or views.shape[1] != width
        or views.shape[0] < 2
        or not np.all(np.isfinite(views))
    ):
        raise DiakrinoCloseoutError("ranks_by_view is missing, non-finite, or misaligned")
    core = {int(value) for value in protected_core}
    if not core or any(value < 0 or value >= width for value in core):
        raise DiakrinoCloseoutError("protected_core is empty or out of range")
    budget = dynamic_addition_budget(len(core))
    pool_size = min(width, budget * int(config.proposal_pool_multiplier))
    ordered = np.argsort(-real, kind="mergesort")
    view_sets = [
        set(np.argsort(-views[row], kind="mergesort")[:pool_size].tolist())
        for row in range(views.shape[0])
    ]
    stability = nogueira_selected_set_stability(view_sets, n_features=width)
    stability_pass = stability is not None and stability >= float(
        config.selected_set_stability_min
    )
    feature_mask = (
        (real - shadow >= float(config.shadow_margin_min))
        & (real - label_null >= float(config.label_null_margin_min))
        & (dispersion <= float(config.rank_std_max))
    )
    proposal_pool = set(int(index) for index in ordered[:pool_size].tolist())
    proposals = tuple(
        int(index)
        for index in ordered.tolist()
        if int(index) in proposal_pool
        and int(index) not in core
        and bool(feature_mask[int(index)])
    )
    if not stability_pass:
        proposals = tuple()
    diagnostics: dict[str, object] = {
        "shadow_separation": float(np.mean(real - shadow)),
        "label_null_separation": float(np.mean(real - label_null)),
        "selected_set_stability": stability,
        "selected_set_stability_feature_width": width,
        "selected_set_stability_pass": bool(stability_pass),
        "proposal_count": len(proposals),
        "feature_gate_pass_count": int(np.sum(feature_mask)),
        "proposal_pool_size": int(pool_size),
    }
    return proposals, diagnostics


def _discretize(values: np.ndarray, bins: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.where(np.isfinite(vector), vector, np.nanmedian(vector))
    if np.ptp(finite) <= 1e-12:
        return np.zeros(vector.size, dtype=np.int64)
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1))[1:-1])
    return np.searchsorted(edges, finite, side="right").astype(np.int64)


def _mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    left_values, left_inverse = np.unique(left, return_inverse=True)
    right_values, right_inverse = np.unique(right, return_inverse=True)
    counts = np.zeros((left_values.size, right_values.size), dtype=np.float64)
    np.add.at(counts, (left_inverse, right_inverse), 1.0)
    joint = counts / float(left.size)
    px = np.sum(joint, axis=1, keepdims=True)
    py = np.sum(joint, axis=0, keepdims=True)
    mask = joint > 0.0
    return float(np.sum(joint[mask] * np.log(joint[mask] / (px @ py)[mask])))


def _conditional_mutual_information(
    left: np.ndarray, right: np.ndarray, labels: np.ndarray
) -> float:
    total = 0.0
    for label in np.unique(labels):
        mask = labels == label
        total += float(np.mean(mask)) * _mutual_information(left[mask], right[mask])
    return total


def support_only_classical_scores(
    X_support: np.ndarray,
    y_support: np.ndarray,
    *,
    discretization_bins: int = 5,
) -> np.ndarray:
    """Dense deterministic C1 score from support-only MI plus ANOVA F ranks."""

    X = np.asarray(X_support, dtype=np.float64)
    y = np.asarray(y_support).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size or np.unique(y).size < 2:
        raise DiakrinoCloseoutError("classical score support X/y dimensions are invalid")
    if not np.all(np.isfinite(X)):
        raise DiakrinoCloseoutError("classical score support X must be finite")
    mi = np.asarray(
        [
            _mutual_information(_discretize(X[:, index], discretization_bins), y)
            for index in range(X.shape[1])
        ],
        dtype=np.float64,
    )
    overall = np.mean(X, axis=0)
    numerator = np.zeros(X.shape[1], dtype=np.float64)
    denominator = np.zeros(X.shape[1], dtype=np.float64)
    for label in np.unique(y):
        group = X[y == label]
        mean = np.mean(group, axis=0)
        numerator += group.shape[0] * (mean - overall) ** 2
        denominator += np.sum((group - mean) ** 2, axis=0)
    f_score = numerator / np.maximum(denominator, np.finfo(np.float64).eps)

    def rank01(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < order.size:
            stop = start + 1
            while stop < order.size and values[order[stop]] == values[order[start]]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * float(start + stop - 1)
            start = stop
        return ranks / max(1, values.size - 1)

    return 0.5 * rank01(mi) + 0.5 * rank01(f_score)


def jmi_admit(
    X_support: np.ndarray,
    y_support: np.ndarray,
    *,
    protected_core: Sequence[int],
    proposals: Sequence[int],
    real_rank: Sequence[float],
    budget: int,
    discretization_bins: int = 5,
) -> tuple[tuple[int, ...], tuple[dict[str, float | int], ...]]:
    """Greedy support-only three-term JMI admission over DIAKRINO proposals."""

    X = np.asarray(X_support, dtype=np.float64)
    y = np.asarray(y_support).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size or np.unique(y).size < 2:
        raise DiakrinoCloseoutError("JMI support X/y dimensions are invalid")
    ranks = _vector(real_rank, name="real_rank", width=X.shape[1])
    _, class_counts = np.unique(y, return_counts=True)
    class_probabilities = class_counts.astype(np.float64) / float(y.size)
    label_entropy_nats = float(
        -np.sum(class_probabilities * np.log(class_probabilities))
    )
    relevance_nats = ranks * label_entropy_nats
    core = tuple(dict.fromkeys(int(value) for value in protected_core))
    candidates = tuple(
        value
        for value in dict.fromkeys(int(value) for value in proposals)
        if value not in set(core)
    )
    if any(value < 0 or value >= X.shape[1] for value in (*core, *candidates)):
        raise DiakrinoCloseoutError("JMI feature index is out of range")
    count = min(max(0, int(budget)), len(candidates))
    discrete = [_discretize(X[:, index], int(discretization_bins)) for index in range(X.shape[1])]
    admitted: list[int] = []
    ledger: list[dict[str, float | int]] = []
    remaining = list(candidates)
    while remaining and len(admitted) < count:
        conditioning = [*core, *admitted]
        scored: list[tuple[float, int, float, float]] = []
        for candidate in remaining:
            if conditioning:
                redundancy = float(
                    np.mean(
                        [
                            _mutual_information(discrete[candidate], discrete[selected])
                            for selected in conditioning
                        ]
                    )
                )
                complementarity = float(
                    np.mean(
                        [
                            _conditional_mutual_information(
                                discrete[candidate], discrete[selected], y
                            )
                            for selected in conditioning
                        ]
                    )
                )
            else:
                redundancy = 0.0
                complementarity = 0.0
            score = float(relevance_nats[candidate] - redundancy + complementarity)
            scored.append((score, candidate, redundancy, complementarity))
        score, chosen, redundancy, complementarity = max(
            scored, key=lambda item: (item[0], -item[1])
        )
        admitted.append(chosen)
        remaining.remove(chosen)
        ledger.append(
            {
                "feature_index": chosen,
                "criterion": score,
                "diakrino_relevance_rank01": float(ranks[chosen]),
                "diakrino_relevance_nats": float(relevance_nats[chosen]),
                "label_entropy_nats": label_entropy_nats,
                "mean_redundancy_mi": redundancy,
                "mean_conditional_complementarity_mi": complementarity,
            }
        )
    return tuple(admitted), tuple(ledger)


def matched_control_additions(
    arm: str,
    *,
    protected_core: Sequence[int],
    realized_additions: int,
    n_features: int,
    seed: int,
    classical_scores: Sequence[float],
    diakrino_ranks: Sequence[float],
) -> tuple[int, ...]:
    """Return C1/C2/C3 additions at exactly P4's realized budget."""

    if arm not in {"classical_next_best", "random_extras", "permuted_diakrino_ranks"}:
        raise DiakrinoCloseoutError(f"unsupported matched control arm: {arm!r}")
    width = int(n_features)
    core = {int(value) for value in protected_core}
    count = int(realized_additions)
    available = np.asarray([value for value in range(width) if value not in core], dtype=int)
    if count < 0 or count > available.size:
        raise DiakrinoCloseoutError("matched control budget is impossible")
    classical = _vector(classical_scores, name="classical_scores", width=width)
    diakrino = _vector(diakrino_ranks, name="diakrino_ranks", width=width)
    if arm == "classical_next_best":
        ordered = available[np.argsort(-classical[available], kind="mergesort")]
    elif arm == "random_extras":
        rng = np.random.Generator(np.random.PCG64(_seed(seed, arm)))
        ordered = rng.permutation(available)
    else:
        rng = np.random.Generator(np.random.PCG64(_seed(seed, arm)))
        permuted = diakrino[rng.permutation(width)]
        ordered = available[np.argsort(-permuted[available], kind="mergesort")]
    return tuple(int(value) for value in ordered[:count].tolist())


def finalize_closeout_decision(
    arm: str,
    *,
    protected_core: Sequence[int],
    additions: Sequence[int],
    n_features: int,
    reason: str,
    diagnostics: Mapping[str, object] | None = None,
) -> DiakrinoCloseoutDecision:
    """Seal retention, fallback, realized-budget, and output identity."""

    if arm not in DIAKRINO_CLOSEOUT_ARMS:
        raise DiakrinoCloseoutError(f"unsupported closeout arm: {arm!r}")
    core = tuple(dict.fromkeys(int(value) for value in protected_core))
    extras = tuple(dict.fromkeys(int(value) for value in additions))
    width = int(n_features)
    if not core or any(value < 0 or value >= width for value in (*core, *extras)):
        raise DiakrinoCloseoutError("decision indices are empty, invalid, or out of range")
    if set(core) & set(extras):
        raise DiakrinoCloseoutError("closeout additions overlap the protected core")
    budget = dynamic_addition_budget(len(core))
    if len(extras) > budget:
        raise DiakrinoCloseoutError("closeout additions exceed the dynamic budget")
    final = (*core, *extras)
    fallback = len(extras) == 0 and final == core
    return DiakrinoCloseoutDecision(
        arm=arm,
        protected_core=core,
        additions=extras,
        final=final,
        addition_budget=budget,
        realized_additions=len(extras),
        abstained=len(extras) == 0,
        fallback_exact=fallback,
        reason=str(reason),
        diagnostics=dict(diagnostics or {}),
    )

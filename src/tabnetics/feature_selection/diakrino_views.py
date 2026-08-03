"""Frozen deterministic views for the vnext-C DIAKRINO feature-selection campaign.

The views transform only the support-side DIAKRINO episode.  They do not create a
new train/test split, and every feature permutation has an exact inverse back
to original feature positions.  The persisted artifact carries enough raw
inputs to rederive the complete frozen score chain without trusting producer
computed intermediates.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DIAKRINO_VIEW_SCHEMA_VERSION = "diakrino_uniform_views_v2"
DIAKRINO_VIEW_SCORE_SOURCE = "prior_screening_fusion_w075_chunk_zscore"
DIAKRINO_VIEW_AGGREGATION = "uniform_rank_mean_v1"
DIAKRINO_VIEW_RANKING = "rank01_average_ties_v1"
DIAKRINO_VIEW_CALIBRATION = "chunk_zscore_v1"
DIAKRINO_VIEW_PRIOR_WEIGHT = 0.75
DIAKRINO_VIEW_SCREENING_WEIGHT = 0.25
DIAKRINO_VIEW_PAD_SENTINEL = -30.0
DIAKRINO_FROZEN_VIEW_IDS = (
    "identity",
    "feature_panel_permutation",
    "support_row_permutation",
    "class_id_rotation",
    "combined",
)


class DiakrinoViewError(ValueError):
    """Raised for malformed or non-reversible DIAKRINO view inputs."""


def _exact_int64(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise DiakrinoViewError(f"{label} must be an exact integer")
    integer = int(value)
    if integer < np.iinfo(np.int64).min or integer > np.iinfo(np.int64).max:
        raise DiakrinoViewError(f"{label} is outside int64 range")
    if minimum is not None and integer < minimum:
        raise DiakrinoViewError(f"{label} must be at least {minimum}")
    return integer


def _exact_dimension(value: Any, *, label: str, minimum: int) -> int:
    return _exact_int64(value, label=label, minimum=minimum)


def _exact_int_vector(
    values: Sequence[int] | np.ndarray,
    *,
    label: str,
    size: int,
) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise DiakrinoViewError(f"{label} must be an exact integer vector")
    try:
        items = list(values)
    except TypeError as exc:
        raise DiakrinoViewError(f"{label} must be an exact integer vector") from exc
    if len(items) != size:
        raise DiakrinoViewError(f"{label} has the wrong width")
    return np.asarray(
        [
            _exact_int64(item, label=f"{label}[{index}]")
            for index, item in enumerate(items)
        ],
        dtype=np.int64,
    )


def _finite_float_vector(values: Any, *, label: str, size: int) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise DiakrinoViewError(f"{label} must be a finite numeric vector")
    try:
        items = list(values)
    except TypeError as exc:
        raise DiakrinoViewError(f"{label} must be a finite numeric vector") from exc
    if len(items) != size:
        raise DiakrinoViewError(f"{label} has the wrong width")
    parsed: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, float, np.integer, np.floating)
        ):
            raise DiakrinoViewError(f"{label}[{index}] is not numeric")
        value = float(item)
        if not np.isfinite(value):
            raise DiakrinoViewError(f"{label}[{index}] is not finite")
        parsed.append(value)
    return np.asarray(parsed, dtype=np.float64)


def _validate_binding_sha256(binding_sha256: Any) -> str:
    if not isinstance(binding_sha256, str):
        raise DiakrinoViewError("DIAKRINO view construction requires a lowercase binding SHA-256")
    binding = binding_sha256.strip()
    if (
        binding != binding_sha256
        or len(binding) != 64
        or any(char not in "0123456789abcdef" for char in binding)
    ):
        raise DiakrinoViewError("DIAKRINO view construction requires a lowercase binding SHA-256")
    return binding


def _seed(binding_sha256: str, view_id: str) -> int:
    binding = _validate_binding_sha256(binding_sha256)
    digest = hashlib.sha256(
        f"{DIAKRINO_VIEW_SCHEMA_VERSION}:{binding}:{view_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & np.iinfo(np.int64).max


def _permutation(
    size: int,
    *,
    seed: int,
    namespace: str,
    enabled: bool,
) -> np.ndarray:
    """Return a version-independent SHA-256-keyed permutation."""

    width = _exact_dimension(size, label="permutation size", minimum=1)
    exact_seed = _exact_int64(seed, label="permutation seed", minimum=0)
    if not enabled:
        return np.arange(width, dtype=np.int64)
    if width < 2:
        raise DiakrinoViewError("a nonidentity DIAKRINO view requires at least two items")
    seed_bytes = exact_seed.to_bytes(8, "big", signed=False)
    keyed = []
    for index in range(width):
        digest = hashlib.sha256(
            b"diakrino-view-permutation-v1\0"
            + namespace.encode("utf-8")
            + b"\0"
            + seed_bytes
            + index.to_bytes(8, "big", signed=False)
        ).digest()
        keyed.append((digest, index))
    out = np.asarray([index for _digest, index in sorted(keyed)], dtype=np.int64)
    if np.array_equal(out, np.arange(width, dtype=np.int64)):
        # An enabled view must not become an accidental second identity vote.
        out = np.roll(out, 1)
    return out


def _validate_permutation(
    values: Sequence[int] | np.ndarray,
    *,
    size: int,
    label: str,
) -> np.ndarray:
    width = _exact_dimension(size, label=f"{label} width", minimum=1)
    array = _exact_int_vector(values, label=label, size=width)
    if np.any(array < 0) or not np.array_equal(np.sort(array), np.arange(width)):
        raise DiakrinoViewError(f"{label} is not an exact permutation")
    return array


@dataclass(frozen=True)
class DiakrinoInferenceView:
    """One reproducible support-only DIAKRINO inference view."""

    view_id: str
    seed: int
    feature_permutation: tuple[int, ...]
    support_permutation: tuple[int, ...]
    class_rotation: int

    def validate(self, *, n_features: int, n_support: int, n_classes: int) -> None:
        feature_count = _exact_dimension(n_features, label="n_features", minimum=2)
        support_count = _exact_dimension(n_support, label="n_support", minimum=2)
        class_count = _exact_dimension(n_classes, label="n_classes", minimum=2)
        if not isinstance(self.view_id, str) or self.view_id not in DIAKRINO_FROZEN_VIEW_IDS:
            raise DiakrinoViewError(f"unsupported frozen DIAKRINO view {self.view_id!r}")
        _exact_int64(self.seed, label="view seed", minimum=0)
        feature = _validate_permutation(
            self.feature_permutation,
            size=feature_count,
            label="feature_permutation",
        )
        support = _validate_permutation(
            self.support_permutation,
            size=support_count,
            label="support_permutation",
        )
        rotation = _exact_int64(self.class_rotation, label="class_rotation", minimum=0)
        if rotation >= class_count:
            raise DiakrinoViewError("class_rotation is outside the observed class range")
        feature_enabled = self.view_id in {"feature_panel_permutation", "combined"}
        support_enabled = self.view_id in {"support_row_permutation", "combined"}
        rotation_enabled = self.view_id in {"class_id_rotation", "combined"}
        if (not np.array_equal(feature, np.arange(feature_count))) != feature_enabled:
            raise DiakrinoViewError(
                "feature permutation is inconsistent with the frozen view id"
            )
        if (not np.array_equal(support, np.arange(support_count))) != support_enabled:
            raise DiakrinoViewError(
                "support permutation is inconsistent with the frozen view id"
            )
        if (rotation != 0) != rotation_enabled:
            raise DiakrinoViewError("class rotation is inconsistent with the frozen view id")

    def transform_support(
        self,
        X_support: np.ndarray,
        y_support: np.ndarray,
        *,
        n_classes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return feature-/row-transformed support arrays and original feature ids."""

        X = np.asarray(X_support)
        y = np.asarray(y_support).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise DiakrinoViewError("support X/y shapes are incompatible")
        class_count = _exact_dimension(n_classes, label="n_classes", minimum=2)
        self.validate(
            n_features=int(X.shape[1]),
            n_support=int(X.shape[0]),
            n_classes=class_count,
        )
        labels = _exact_int_vector(y, label="support labels", size=int(y.shape[0]))
        feature = _validate_permutation(
            self.feature_permutation,
            size=int(X.shape[1]),
            label="feature_permutation",
        )
        rows = _validate_permutation(
            self.support_permutation,
            size=int(X.shape[0]),
            label="support_permutation",
        )
        labels = labels[rows]
        if np.any(labels < 0) or np.any(labels >= class_count):
            raise DiakrinoViewError("support labels are outside the canonical class range")
        labels = (labels + self.class_rotation) % class_count
        return np.asarray(X[rows][:, feature]), labels, feature

    def remap_feature_vector(
        self, view_values: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        values = np.asarray(view_values, dtype=np.float64).reshape(-1)
        feature = _validate_permutation(
            self.feature_permutation,
            size=values.shape[0],
            label="feature_permutation",
        )
        out = np.empty_like(values)
        out[feature] = values
        return out

    def transform_paired_support(
        self,
        X_support: np.ndarray,
        X_shadow_support: np.ndarray,
        y_support: np.ndarray,
        *,
        n_classes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply one frozen view while keeping real/shadow slots adjacent.

        The caller supplies a deterministic support-only shadow matrix with the
        same shape as ``X_support``.  Row and feature transforms are applied to
        both matrices before interleaving, so each even/odd slot pair always
        maps to the same original feature identity.
        """

        X = np.asarray(X_support)
        shadow = np.asarray(X_shadow_support)
        if X.ndim != 2 or shadow.shape != X.shape:
            raise DiakrinoViewError("paired support matrices must have the same shape")
        transformed, labels, feature = self.transform_support(
            X, y_support, n_classes=n_classes
        )
        rows = _validate_permutation(
            self.support_permutation,
            size=int(X.shape[0]),
            label="support_permutation",
        )
        shadow_transformed = np.asarray(shadow[rows][:, feature])
        panel = np.empty(
            (int(X.shape[0]), int(X.shape[1]) * 2), dtype=transformed.dtype
        )
        panel[:, 0::2] = transformed
        panel[:, 1::2] = shadow_transformed
        return panel, labels, feature

    def manifest_record(self) -> dict[str, object]:
        seed = _exact_int64(self.seed, label="view seed", minimum=0)
        rotation = _exact_int64(self.class_rotation, label="class_rotation", minimum=0)
        return {
            "view_id": self.view_id,
            "seed": seed,
            "feature_permutation": list(self.feature_permutation),
            "support_permutation": list(self.support_permutation),
            "class_rotation": rotation,
        }


@dataclass(frozen=True)
class DiakrinoPairedPanelChunk:
    """One bounded real/shadow panel with exact original-feature slot mapping."""

    chunk_id: int
    original_feature_indices: tuple[int, ...]
    real_slots: tuple[int, ...]
    shadow_slots: tuple[int, ...]

    def manifest_record(self) -> dict[str, object]:
        return {
            "chunk_id": int(self.chunk_id),
            "original_feature_indices": list(self.original_feature_indices),
            "real_slots": list(self.real_slots),
            "shadow_slots": list(self.shadow_slots),
        }


def paired_panel_chunks(
    *,
    original_feature_indices: Sequence[int] | np.ndarray,
    candidate_budget: int,
) -> tuple[DiakrinoPairedPanelChunk, ...]:
    """Partition ordered original features into bounded adjacent-slot panels."""

    original = _exact_int_vector(
        original_feature_indices,
        label="paired original_feature_indices",
        size=len(original_feature_indices),
    )
    if original.size < 1 or np.any(original < 0) or len(set(original.tolist())) != original.size:
        raise DiakrinoViewError("paired panel features must be unique non-negative integers")
    budget = _exact_dimension(candidate_budget, label="paired candidate_budget", minimum=2)
    capacity = budget // 2
    if capacity < 1:
        raise DiakrinoViewError("paired candidate_budget cannot hold one feature pair")
    chunks: list[DiakrinoPairedPanelChunk] = []
    for chunk_id, start in enumerate(range(0, int(original.size), capacity)):
        feature_ids = tuple(int(value) for value in original[start : start + capacity])
        slots = tuple(range(len(feature_ids)))
        chunks.append(
            DiakrinoPairedPanelChunk(
                chunk_id=int(chunk_id),
                original_feature_indices=feature_ids,
                real_slots=tuple(2 * slot for slot in slots),
                shadow_slots=tuple((2 * slot) + 1 for slot in slots),
            )
        )
    return tuple(chunks)


@dataclass(frozen=True)
class ValidatedDiakrinoViewArtifact:
    """Strictly rederived view artifact values used by consumers."""

    views: tuple[DiakrinoInferenceView, ...]
    score_source: str
    uniform_rank: tuple[float, ...]
    uniform_rank_std: tuple[float, ...]
    rank01_by_view: tuple[tuple[float, ...], ...]

    @property
    def view_ids(self) -> tuple[str, ...]:
        return tuple(view.view_id for view in self.views)

    def diagnostics(self) -> dict[str, object]:
        dispersion = [float(value) for value in self.uniform_rank_std]
        ordered = sorted(dispersion)
        midpoint = len(ordered) // 2
        median = (
            ordered[midpoint]
            if len(ordered) % 2
            else 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
        )
        return {
            "schema_version": DIAKRINO_VIEW_SCHEMA_VERSION,
            "view_ids": list(self.view_ids),
            "score_source": self.score_source,
            "uniform_rank_std_mean": math.fsum(dispersion) / float(len(dispersion)),
            "uniform_rank_std_median": median,
            "uniform_rank_std_max": max(dispersion),
            "uniform_rank_std_nonzero_fraction": sum(
                value > 0.0 for value in dispersion
            )
            / float(len(dispersion)),
        }


def frozen_inference_views(
    *, binding_sha256: str, n_features: int, n_support: int, n_classes: int
) -> tuple[DiakrinoInferenceView, ...]:
    """Build the five predeclared views from one immutable sidecar binding."""

    binding = _validate_binding_sha256(binding_sha256)
    feature_count = _exact_dimension(n_features, label="n_features", minimum=2)
    support_count = _exact_dimension(n_support, label="n_support", minimum=2)
    class_count = _exact_dimension(n_classes, label="n_classes", minimum=2)
    specs: list[DiakrinoInferenceView] = []
    for view_id in DIAKRINO_FROZEN_VIEW_IDS:
        seed = _seed(binding, view_id)
        feature_enabled = view_id in {"feature_panel_permutation", "combined"}
        support_enabled = view_id in {"support_row_permutation", "combined"}
        rotate = (
            1 + (seed % (class_count - 1))
            if view_id in {"class_id_rotation", "combined"}
            else 0
        )
        view = DiakrinoInferenceView(
            view_id=view_id,
            seed=seed,
            feature_permutation=tuple(
                _permutation(
                    feature_count,
                    seed=seed,
                    namespace=f"{view_id}:feature",
                    enabled=feature_enabled,
                ).tolist()
            ),
            support_permutation=tuple(
                _permutation(
                    support_count,
                    seed=seed,
                    namespace=f"{view_id}:support",
                    enabled=support_enabled,
                ).tolist()
            ),
            class_rotation=int(rotate),
        )
        view.validate(
            n_features=feature_count,
            n_support=support_count,
            n_classes=class_count,
        )
        specs.append(view)
    return tuple(specs)


def rank01_average_ties(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Rank finite values in [0, 1], preserving NaNs and averaging ties."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(array.shape, np.nan, dtype=np.float64)
    indices = np.flatnonzero(np.isfinite(array))
    if indices.size <= 1:
        out[indices] = 0.0
        return out
    order = indices[np.argsort(array[indices], kind="mergesort")]
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and array[order[stop]] == array[order[start]]:
            stop += 1
        out[order[start:stop]] = 0.5 * float(start + stop - 1) / float(order.size - 1)
        start = stop
    return out


def uniform_rank_aggregate(
    values_by_view: Mapping[str, Sequence[float] | np.ndarray],
    *,
    expected_view_ids: Iterable[str] = DIAKRINO_FROZEN_VIEW_IDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uniform mean ranks and per-feature rank standard deviations."""

    expected = tuple(expected_view_ids)
    if tuple(values_by_view) != expected:
        raise DiakrinoViewError(
            "DIAKRINO uniform aggregation requires exactly the frozen view ids"
        )
    widths = {
        np.asarray(values_by_view[view_id]).reshape(-1).shape[0] for view_id in expected
    }
    if len(widths) != 1:
        raise DiakrinoViewError("DIAKRINO view vectors have inconsistent feature widths")
    width = widths.pop()
    ranks = [
        rank01_average_ties(
            _finite_float_vector(
                values_by_view[view_id],
                label=f"{view_id} calibrated scores",
                size=width,
            )
        )
        for view_id in expected
    ]
    matrix = np.vstack(ranks)
    if not np.all(np.isfinite(matrix)):
        raise DiakrinoViewError("DIAKRINO uniform aggregation rejects non-finite view scores")
    return _uniform_rank_statistics(matrix)


def _uniform_rank_statistics(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact-order float64 moments without version-dependent reductions."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise DiakrinoViewError("DIAKRINO uniform rank matrix has invalid dimensions")
    means = np.empty(values.shape[1], dtype=np.float64)
    stds = np.empty(values.shape[1], dtype=np.float64)
    denominator = float(values.shape[0])
    for feature_index in range(values.shape[1]):
        column = [float(value) for value in values[:, feature_index].tolist()]
        mean = math.fsum(column) / denominator
        variance = math.fsum((value - mean) ** 2 for value in column) / denominator
        means[feature_index] = mean
        stds[feature_index] = math.sqrt(variance)
    return means, stds


def _chunk_zscore(values: np.ndarray, chunk_ids: np.ndarray) -> np.ndarray:
    """Mirror the frozen sidecar chunk-zscore semantics without an import cycle."""

    out = np.asarray(values, dtype=np.float64).copy()
    chunks = np.asarray(chunk_ids, dtype=np.int64)
    valid = np.isfinite(out) & (out > DIAKRINO_VIEW_PAD_SENTINEL + 1e-6)
    result = np.full(out.shape[0], np.nan, dtype=np.float64)
    for chunk_id in np.unique(chunks):
        mask = (chunks == chunk_id) & valid
        if not np.any(mask):
            continue
        local = [float(value) for value in out[mask].tolist()]
        mean = math.fsum(local) / float(len(local))
        variance = math.fsum((value - mean) ** 2 for value in local) / float(len(local))
        std = math.sqrt(variance)
        result[mask] = (
            np.asarray([(value - mean) / std for value in local], dtype=np.float64)
            if std > 1e-12
            else 0.0
        )
    if np.any(~np.isfinite(result)):
        fill = float(np.nanmin(result)) if np.any(np.isfinite(result)) else 0.0
        result = np.where(np.isfinite(result), result, fill)
    return result


def _score_source_record() -> dict[str, object]:
    return {
        "id": DIAKRINO_VIEW_SCORE_SOURCE,
        "prior_weight": DIAKRINO_VIEW_PRIOR_WEIGHT,
        "screening_weight": DIAKRINO_VIEW_SCREENING_WEIGHT,
        "calibration": DIAKRINO_VIEW_CALIBRATION,
        "ranking": DIAKRINO_VIEW_RANKING,
        "pad_sentinel": DIAKRINO_VIEW_PAD_SENTINEL,
    }


def _validate_score_source_record(raw: Any) -> None:
    expected = _score_source_record()
    if not isinstance(raw, Mapping) or set(raw) != set(expected):
        raise DiakrinoViewError("DIAKRINO inference-view score-source identity is invalid")
    for field in ("id", "calibration", "ranking"):
        if not isinstance(raw[field], str) or raw[field] != expected[field]:
            raise DiakrinoViewError("DIAKRINO inference-view score-source identity is invalid")
    for field in (
        "prior_weight",
        "screening_weight",
        "pad_sentinel",
    ):
        if not isinstance(raw[field], float) or raw[field] != expected[field]:
            raise DiakrinoViewError("DIAKRINO inference-view score-source identity is invalid")


def _derive_view_scores(
    raw: Mapping[str, Any],
    *,
    view_id: str,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "prior_logit",
        "screening_logit",
        "chunk_id",
    }:
        raise DiakrinoViewError(f"{view_id} score inputs are missing or malformed")
    prior = _finite_float_vector(
        raw["prior_logit"], label=f"{view_id} prior_logit", size=n_features
    )
    screening = _finite_float_vector(
        raw["screening_logit"],
        label=f"{view_id} screening_logit",
        size=n_features,
    )
    chunks = _exact_int_vector(
        raw["chunk_id"], label=f"{view_id} chunk_id", size=n_features
    )
    if np.any(chunks < 0) or not np.array_equal(
        np.unique(chunks), np.arange(int(np.max(chunks)) + 1, dtype=np.int64)
    ):
        raise DiakrinoViewError(f"{view_id} chunk ids are negative or gapped")
    fused = np.asarray(
        [
            (DIAKRINO_VIEW_PRIOR_WEIGHT * float(prior_value))
            + (DIAKRINO_VIEW_SCREENING_WEIGHT * float(screening_value))
            for prior_value, screening_value in zip(prior, screening, strict=True)
        ],
        dtype=np.float64,
    )
    calibrated = _chunk_zscore(fused, chunks)
    ranks = rank01_average_ties(calibrated)
    if not np.all(np.isfinite(ranks)):
        raise DiakrinoViewError(f"{view_id} derived ranks are non-finite")
    return prior, screening, chunks, fused, calibrated, ranks


def build_view_artifact(
    *,
    binding_sha256: str,
    n_features: int,
    n_support: int,
    n_classes: int,
    score_inputs: Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
) -> dict[str, object]:
    """Build a raw-input-sealed JSON payload for a multi-view sidecar."""

    binding = _validate_binding_sha256(binding_sha256)
    feature_count = _exact_dimension(n_features, label="n_features", minimum=2)
    support_count = _exact_dimension(n_support, label="n_support", minimum=2)
    class_count = _exact_dimension(n_classes, label="n_classes", minimum=2)
    if (
        not isinstance(score_inputs, Mapping)
        or tuple(score_inputs) != DIAKRINO_FROZEN_VIEW_IDS
    ):
        raise DiakrinoViewError("DIAKRINO view score inputs require exactly the frozen view ids")
    views = frozen_inference_views(
        binding_sha256=binding,
        n_features=feature_count,
        n_support=support_count,
        n_classes=class_count,
    )
    records: list[dict[str, object]] = []
    ranks_by_view: dict[str, np.ndarray] = {}
    for view in views:
        prior, screening, chunks, fused, calibrated, ranks = _derive_view_scores(
            score_inputs[view.view_id],
            view_id=view.view_id,
            n_features=feature_count,
        )
        record = view.manifest_record()
        record["score_inputs"] = {
            "prior_logit": prior.tolist(),
            "screening_logit": screening.tolist(),
            "chunk_id": chunks.tolist(),
        }
        record["derived_scores"] = {
            "fused_logit": fused.tolist(),
            "calibrated_score": calibrated.tolist(),
            "rank01": ranks.tolist(),
        }
        records.append(record)
        ranks_by_view[view.view_id] = calibrated
    aggregate, dispersion = uniform_rank_aggregate(ranks_by_view)
    return {
        "schema_version": DIAKRINO_VIEW_SCHEMA_VERSION,
        "binding_sha256": binding,
        "n_features": feature_count,
        "n_support": support_count,
        "n_classes": class_count,
        "score_source": _score_source_record(),
        "aggregation": DIAKRINO_VIEW_AGGREGATION,
        "views": records,
        "uniform_rank": aggregate.tolist(),
        "uniform_rank_std": dispersion.tolist(),
    }


def _parse_view_record(
    raw: Any,
    *,
    reference: DiakrinoInferenceView,
    n_features: int,
    n_support: int,
    n_classes: int,
) -> tuple[DiakrinoInferenceView, np.ndarray]:
    expected_keys = {
        "view_id",
        "seed",
        "feature_permutation",
        "support_permutation",
        "class_rotation",
        "score_inputs",
        "derived_scores",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise DiakrinoViewError("DIAKRINO inference-view artifact has a malformed view")
    view_id = raw["view_id"]
    if not isinstance(view_id, str):
        raise DiakrinoViewError("DIAKRINO inference-view artifact has a malformed view id")
    candidate = DiakrinoInferenceView(
        view_id=view_id,
        seed=_exact_int64(raw["seed"], label=f"{view_id} seed", minimum=0),
        feature_permutation=tuple(
            _exact_int_vector(
                raw["feature_permutation"],
                label=f"{view_id} feature_permutation",
                size=n_features,
            ).tolist()
        ),
        support_permutation=tuple(
            _exact_int_vector(
                raw["support_permutation"],
                label=f"{view_id} support_permutation",
                size=n_support,
            ).tolist()
        ),
        class_rotation=_exact_int64(
            raw["class_rotation"], label=f"{view_id} class_rotation", minimum=0
        ),
    )
    candidate.validate(
        n_features=n_features,
        n_support=n_support,
        n_classes=n_classes,
    )
    if candidate != reference:
        raise DiakrinoViewError(
            "DIAKRINO inference-view artifact differs from the frozen view contract"
        )
    _prior, _screening, _chunks, fused, calibrated, ranks = _derive_view_scores(
        raw["score_inputs"], view_id=view_id, n_features=n_features
    )
    derived = raw["derived_scores"]
    if not isinstance(derived, Mapping) or set(derived) != {
        "fused_logit",
        "calibrated_score",
        "rank01",
    }:
        raise DiakrinoViewError(f"{view_id} derived scores are missing or malformed")
    declared_fused = _finite_float_vector(
        derived["fused_logit"], label=f"{view_id} fused_logit", size=n_features
    )
    declared_calibrated = _finite_float_vector(
        derived["calibrated_score"],
        label=f"{view_id} calibrated_score",
        size=n_features,
    )
    declared_ranks = _finite_float_vector(
        derived["rank01"], label=f"{view_id} rank01", size=n_features
    )
    if not (
        np.array_equal(declared_fused, fused)
        and np.array_equal(declared_calibrated, calibrated)
        and np.array_equal(declared_ranks, ranks)
    ):
        raise DiakrinoViewError(
            "DIAKRINO inference-view derived scores do not match their sealed raw inputs"
        )
    return candidate, ranks


def validate_view_artifact(
    payload: Mapping[str, Any],
    *,
    binding_sha256: str,
    n_features: int,
    n_support: int,
    n_classes: int,
) -> ValidatedDiakrinoViewArtifact:
    """Strictly rederive and validate the frozen uniform-view contract."""

    try:
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "binding_sha256",
            "n_features",
            "n_support",
            "n_classes",
            "score_source",
            "aggregation",
            "views",
            "uniform_rank",
            "uniform_rank_std",
        }:
            raise DiakrinoViewError(
                "DIAKRINO inference-view artifact fields are missing or malformed"
            )
        if payload["schema_version"] != DIAKRINO_VIEW_SCHEMA_VERSION:
            raise DiakrinoViewError("unsupported DIAKRINO inference-view artifact schema")
        binding = _validate_binding_sha256(binding_sha256)
        feature_count = _exact_dimension(n_features, label="n_features", minimum=2)
        support_count = _exact_dimension(n_support, label="n_support", minimum=2)
        class_count = _exact_dimension(n_classes, label="n_classes", minimum=2)
        if payload["binding_sha256"] != binding:
            raise DiakrinoViewError(
                "DIAKRINO inference-view artifact binding does not match the sidecar"
            )
        for field, expected in (
            ("n_features", feature_count),
            ("n_support", support_count),
            ("n_classes", class_count),
        ):
            if (
                _exact_int64(payload[field], label=f"artifact {field}", minimum=2)
                != expected
            ):
                raise DiakrinoViewError(
                    "DIAKRINO inference-view artifact dimensions do not match the sidecar"
                )
        _validate_score_source_record(payload["score_source"])
        if payload["aggregation"] != DIAKRINO_VIEW_AGGREGATION:
            raise DiakrinoViewError("DIAKRINO inference-view aggregation identity is invalid")
        raw_views = payload["views"]
        if not isinstance(raw_views, list) or len(raw_views) != len(
            DIAKRINO_FROZEN_VIEW_IDS
        ):
            raise DiakrinoViewError("DIAKRINO inference-view artifact has an invalid view ledger")
        expected_views = frozen_inference_views(
            binding_sha256=binding,
            n_features=feature_count,
            n_support=support_count,
            n_classes=class_count,
        )
        observed_views: list[DiakrinoInferenceView] = []
        ranks_by_view: dict[str, np.ndarray] = {}
        for raw, reference in zip(raw_views, expected_views, strict=True):
            candidate, ranks = _parse_view_record(
                raw,
                reference=reference,
                n_features=feature_count,
                n_support=support_count,
                n_classes=class_count,
            )
            if candidate.view_id in ranks_by_view:
                raise DiakrinoViewError(
                    "DIAKRINO inference-view artifact contains duplicate view ids"
                )
            observed_views.append(candidate)
            # uniform_rank_aggregate ranks this calibrated-like vector again.  Using
            # the already derived ranks preserves those ranks exactly, including ties.
            ranks_by_view[candidate.view_id] = ranks
        rank_matrix = np.vstack(
            [ranks_by_view[view_id] for view_id in DIAKRINO_FROZEN_VIEW_IDS]
        )
        aggregate, dispersion = _uniform_rank_statistics(rank_matrix)
        declared_aggregate = _finite_float_vector(
            payload["uniform_rank"], label="uniform_rank", size=feature_count
        )
        declared_dispersion = _finite_float_vector(
            payload["uniform_rank_std"],
            label="uniform_rank_std",
            size=feature_count,
        )
        if not (
            np.array_equal(aggregate, declared_aggregate)
            and np.array_equal(dispersion, declared_dispersion)
        ):
            raise DiakrinoViewError(
                "DIAKRINO inference-view aggregate does not match its rederived ranks"
            )
        return ValidatedDiakrinoViewArtifact(
            views=tuple(observed_views),
            score_source=DIAKRINO_VIEW_SCORE_SOURCE,
            uniform_rank=tuple(float(value) for value in aggregate.tolist()),
            uniform_rank_std=tuple(float(value) for value in dispersion.tolist()),
            rank01_by_view=tuple(
                tuple(float(value) for value in ranks_by_view[view_id].tolist())
                for view_id in DIAKRINO_FROZEN_VIEW_IDS
            ),
        )
    except DiakrinoViewError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DiakrinoViewError("DIAKRINO inference-view artifact is malformed") from exc

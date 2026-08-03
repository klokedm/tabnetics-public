"""Cross-fitted, fail-closed router uncertainty artifacts.

This module deliberately does not reinterpret legacy score-router offsets as
statistical uncertainty.  It creates a hash-only artifact from source-group
OOF outcomes; a runtime adapter may select a non-default route only after the
artifact validates its candidate, descriptor, and base-router identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


ROUTER_UNCERTAINTY_SCHEMA_VERSION = "tabnetics_router_crossfit_uncertainty_v1"
_HEX = frozenset("0123456789abcdef")


class RouterUncertaintyError(ValueError):
    """Raised when a router uncertainty artifact is malformed or unbound."""


def _sha256(record: object) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _sha(value: object, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or set(text) - _HEX:
        raise RouterUncertaintyError(f"{field} must be a lowercase SHA-256")
    return text


def _token(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text or any(char.isspace() for char in text):
        raise RouterUncertaintyError(f"{field} must be a non-empty token")
    return text


def _finite(value: object, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RouterUncertaintyError(f"{field} must be finite") from exc
    if not np.isfinite(out):
        raise RouterUncertaintyError(f"{field} must be finite")
    return out


def _wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    probability = float(successes) / float(total)
    denominator = 1.0 + z * z / float(total)
    centre = probability + z * z / (2.0 * float(total))
    radius = z * math.sqrt(
        (probability * (1.0 - probability) + z * z / (4.0 * float(total))) / float(total)
    )
    return float(max(0.0, (centre - radius) / denominator))


def router_candidate_schema_sha256(candidate_ids: Sequence[str]) -> str:
    values = tuple(sorted({_token(value, field="candidate_id") for value in candidate_ids}))
    if not values:
        raise RouterUncertaintyError("candidate schema cannot be empty")
    return _sha256({"candidate_ids": values})


def router_descriptor_schema_sha256(feature_names: Sequence[str]) -> str:
    values = tuple(_token(value, field="descriptor_feature") for value in feature_names)
    if len(set(values)) != len(values):
        raise RouterUncertaintyError("descriptor feature names must be unique")
    return _sha256({"feature_names": values})


@dataclass(frozen=True, slots=True)
class RouterOutcomeRow:
    """One source-group OOF candidate-versus-default outcome (training input)."""

    dataset_id: str
    source_id: str
    seed: int
    split_fingerprint: str
    descriptor_sha256: str
    candidate_id: str
    fold_id: str
    predicted_delta_utility: float
    realized_delta_utility: float
    beats_default_probability: float

    def __post_init__(self) -> None:
        for field in ("dataset_id", "source_id", "candidate_id", "fold_id"):
            object.__setattr__(self, field, _token(getattr(self, field), field=field))
        for field in ("split_fingerprint", "descriptor_sha256"):
            object.__setattr__(self, field, _sha(getattr(self, field), field=field))
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise RouterUncertaintyError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))
        for field in ("predicted_delta_utility", "realized_delta_utility"):
            object.__setattr__(self, field, _finite(getattr(self, field), field=field))
        probability = _finite(self.beats_default_probability, field="beats_default_probability")
        if not 0.0 <= probability <= 1.0:
            raise RouterUncertaintyError("beats_default_probability must be in [0, 1]")
        object.__setattr__(self, "beats_default_probability", probability)

    @property
    def identity(self) -> tuple[str, str, int, str, str]:
        return (
            self.dataset_id, self.source_id, self.seed, self.candidate_id,
            self.split_fingerprint,
        )

    def ledger_record(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "source_id": self.source_id,
            "seed": self.seed,
            "split_fingerprint": self.split_fingerprint,
            "descriptor_sha256": self.descriptor_sha256,
            "candidate_id": self.candidate_id,
            "fold_id": self.fold_id,
            "predicted_delta_utility": self.predicted_delta_utility,
            "realized_delta_utility": self.realized_delta_utility,
            "beats_default_probability": self.beats_default_probability,
        }


@dataclass(frozen=True, slots=True)
class CandidateUncertaintyPolicy:
    """Aggregate-only calibration state for one non-default candidate."""

    candidate_id: str
    support_count: int
    residual_quantile: float
    beats_default_probability_threshold: float
    positive_rate_lcb: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _token(self.candidate_id, field="candidate_id"))
        if isinstance(self.support_count, bool) or int(self.support_count) != self.support_count or int(self.support_count) < 1:
            raise RouterUncertaintyError("support_count must be a positive integer")
        object.__setattr__(self, "support_count", int(self.support_count))
        object.__setattr__(self, "residual_quantile", _finite(self.residual_quantile, field="residual_quantile"))
        for field in ("beats_default_probability_threshold", "positive_rate_lcb"):
            value = _finite(getattr(self, field), field=field)
            if not 0.0 <= value <= 1.0:
                raise RouterUncertaintyError(f"{field} must be in [0, 1]")
            object.__setattr__(self, field, value)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "support_count": self.support_count,
            "residual_quantile": self.residual_quantile,
            "beats_default_probability_threshold": self.beats_default_probability_threshold,
            "positive_rate_lcb": self.positive_rate_lcb,
        }


@dataclass(frozen=True, slots=True)
class CrossFitRouterUncertaintyArtifact:
    """Validated hash-only artifact consumed by the default-off runtime guard."""

    base_router_sha256: str
    candidate_schema_sha256: str
    descriptor_schema_sha256: str
    outcome_ledger_sha256: str
    fold_ledger_sha256: str
    frozen_holdout_sha256: str
    residual_quantile_level: float
    minimum_support: int
    policies: tuple[CandidateUncertaintyPolicy, ...]
    schema_version: str = ROUTER_UNCERTAINTY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROUTER_UNCERTAINTY_SCHEMA_VERSION:
            raise RouterUncertaintyError("unsupported router uncertainty schema")
        for field in (
            "base_router_sha256", "candidate_schema_sha256", "descriptor_schema_sha256",
            "outcome_ledger_sha256", "fold_ledger_sha256", "frozen_holdout_sha256",
        ):
            object.__setattr__(self, field, _sha(getattr(self, field), field=field))
        level = _finite(self.residual_quantile_level, field="residual_quantile_level")
        if not 0.0 < level < 1.0:
            raise RouterUncertaintyError("residual_quantile_level must be in (0, 1)")
        object.__setattr__(self, "residual_quantile_level", level)
        if isinstance(self.minimum_support, bool) or int(self.minimum_support) != self.minimum_support or int(self.minimum_support) < 1:
            raise RouterUncertaintyError("minimum_support must be a positive integer")
        object.__setattr__(self, "minimum_support", int(self.minimum_support))
        policies = tuple(self.policies)
        if not policies or not all(isinstance(policy, CandidateUncertaintyPolicy) for policy in policies):
            raise RouterUncertaintyError("router uncertainty artifact requires candidate policies")
        if len({policy.candidate_id for policy in policies}) != len(policies):
            raise RouterUncertaintyError("router uncertainty policy candidate ids must be unique")
        object.__setattr__(self, "policies", tuple(sorted(policies, key=lambda policy: policy.candidate_id)))

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "base_router_sha256": self.base_router_sha256,
            "candidate_schema_sha256": self.candidate_schema_sha256,
            "descriptor_schema_sha256": self.descriptor_schema_sha256,
            "outcome_ledger_sha256": self.outcome_ledger_sha256,
            "fold_ledger_sha256": self.fold_ledger_sha256,
            "frozen_holdout_sha256": self.frozen_holdout_sha256,
            "residual_quantile_level": self.residual_quantile_level,
            "minimum_support": self.minimum_support,
            "policies": [policy.to_dict() for policy in self.policies],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CrossFitRouterUncertaintyArtifact":
        required = {
            "schema_version", "base_router_sha256", "candidate_schema_sha256",
            "descriptor_schema_sha256", "outcome_ledger_sha256", "fold_ledger_sha256",
            "frozen_holdout_sha256", "residual_quantile_level", "minimum_support", "policies",
        }
        if set(payload) != required or not isinstance(payload.get("policies"), list):
            raise RouterUncertaintyError("invalid router uncertainty artifact fields")
        return cls(
            schema_version=str(payload["schema_version"]),
            base_router_sha256=str(payload["base_router_sha256"]),
            candidate_schema_sha256=str(payload["candidate_schema_sha256"]),
            descriptor_schema_sha256=str(payload["descriptor_schema_sha256"]),
            outcome_ledger_sha256=str(payload["outcome_ledger_sha256"]),
            fold_ledger_sha256=str(payload["fold_ledger_sha256"]),
            frozen_holdout_sha256=str(payload["frozen_holdout_sha256"]),
            residual_quantile_level=float(payload["residual_quantile_level"]),
            minimum_support=int(payload["minimum_support"]),
            policies=tuple(
                CandidateUncertaintyPolicy(**dict(item))
                for item in payload["policies"]
                if isinstance(item, Mapping)
            ),
        )

    def validate_runtime(
        self, *, base_router_sha256: str, candidate_ids: Sequence[str], descriptor_schema_sha256: str
    ) -> None:
        if self.base_router_sha256 != _sha(base_router_sha256, field="base_router_sha256"):
            raise RouterUncertaintyError("router uncertainty base artifact mismatch")
        if self.candidate_schema_sha256 != router_candidate_schema_sha256(candidate_ids):
            raise RouterUncertaintyError("router uncertainty candidate schema mismatch")
        if self.descriptor_schema_sha256 != _sha(descriptor_schema_sha256, field="descriptor_schema_sha256"):
            raise RouterUncertaintyError("router uncertainty descriptor schema mismatch")

    def decide(
        self, *, candidate_id: str, predicted_delta_utility: float, beats_default_probability: float
    ) -> tuple[bool, str, float]:
        candidate = _token(candidate_id, field="candidate_id")
        policy = next((item for item in self.policies if item.candidate_id == candidate), None)
        if policy is None:
            return False, "candidate_unsupported", float("nan")
        if policy.support_count < self.minimum_support:
            return False, "insufficient_support", float("nan")
        predicted = _finite(predicted_delta_utility, field="predicted_delta_utility")
        probability = _finite(beats_default_probability, field="beats_default_probability")
        if not 0.0 <= probability <= 1.0:
            raise RouterUncertaintyError("beats_default_probability must be in [0, 1]")
        lower_bound = float(predicted - policy.residual_quantile)
        if lower_bound <= 0.0:
            return False, "nonpositive_delta_lower_bound", lower_bound
        if probability < policy.beats_default_probability_threshold:
            return False, "calibrated_probability_below_threshold", lower_bound
        return True, "", lower_bound


def fit_crossfit_router_uncertainty(
    rows: Iterable[RouterOutcomeRow],
    *,
    base_router_sha256: str,
    descriptor_schema_sha256: str,
    frozen_source_ids: Sequence[str],
    residual_quantile_level: float = 0.90,
    minimum_support: int = 20,
    beats_probability_grid: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90),
) -> CrossFitRouterUncertaintyArtifact:
    """Fit aggregate policies from source-group OOF rows, never frozen sources."""
    outcomes = tuple(rows)
    if not outcomes or not all(isinstance(row, RouterOutcomeRow) for row in outcomes):
        raise RouterUncertaintyError("cross-fit router training requires outcome rows")
    level = _finite(residual_quantile_level, field="residual_quantile_level")
    if not 0.0 < level < 1.0:
        raise RouterUncertaintyError("residual_quantile_level must be in (0, 1)")
    if isinstance(minimum_support, bool) or int(minimum_support) != minimum_support or int(minimum_support) < 2:
        raise RouterUncertaintyError("minimum_support must be an integer >= 2")
    minimum_support = int(minimum_support)
    frozen = {_token(value, field="frozen_source_id") for value in frozen_source_ids}
    if {row.source_id for row in outcomes} & frozen:
        raise RouterUncertaintyError("frozen source appears in cross-fit router training")
    if len({row.identity for row in outcomes}) != len(outcomes):
        raise RouterUncertaintyError("duplicate router outcome identity")
    source_folds: dict[str, str] = {}
    for row in outcomes:
        previous = source_folds.setdefault(row.source_id, row.fold_id)
        if previous != row.fold_id:
            raise RouterUncertaintyError("source group spans cross-fit folds")
    if len(set(source_folds.values())) < 2:
        raise RouterUncertaintyError("cross-fit router training requires at least two source folds")
    grid = tuple(sorted({_finite(value, field="beats_probability_grid") for value in beats_probability_grid}))
    if not grid or grid[0] < 0.0 or grid[-1] > 1.0:
        raise RouterUncertaintyError("invalid beats probability grid")
    candidates = tuple(sorted({row.candidate_id for row in outcomes}))
    policies: list[CandidateUncertaintyPolicy] = []
    for candidate_id in candidates:
        candidate_rows = [row for row in outcomes if row.candidate_id == candidate_id]
        if len(candidate_rows) < minimum_support:
            continue
        folds = tuple(sorted({row.fold_id for row in candidate_rows}))
        if len(folds) < 2:
            raise RouterUncertaintyError("candidate lacks leave-fold-out calibration rows")
        heldout_residuals: list[float] = []
        heldout_thresholds: list[float] = []
        heldout_successes = 0
        heldout_accepted = 0
        for heldout_fold in folds:
            calibration = [row for row in candidate_rows if row.fold_id != heldout_fold]
            evaluation = [row for row in candidate_rows if row.fold_id == heldout_fold]
            if len(calibration) < minimum_support or not evaluation:
                raise RouterUncertaintyError("insufficient leave-fold-out calibration support")
            calibration_residuals = np.asarray(
                [row.predicted_delta_utility - row.realized_delta_utility for row in calibration],
                dtype=float,
            )
            residual_quantile = float(np.quantile(calibration_residuals, level, method="higher"))
            predicted = np.asarray([row.predicted_delta_utility for row in calibration], dtype=float)
            actual = np.asarray([row.realized_delta_utility for row in calibration], dtype=float)
            probabilities = np.asarray([row.beats_default_probability for row in calibration], dtype=float)
            best: tuple[float, float, int] | None = None
            for threshold in grid:
                accepted = (predicted - residual_quantile > 0.0) & (probabilities >= threshold)
                count = int(np.sum(accepted))
                if count < minimum_support:
                    continue
                lower = _wilson_lower_bound(int(np.sum(actual[accepted] > 0.0)), count)
                candidate_key = (lower, threshold, count)
                if best is None or candidate_key > best:
                    best = candidate_key
            if best is None:
                heldout_residuals.append(residual_quantile)
                heldout_thresholds.append(1.0)
                continue
            _lower, threshold, _count = best
            heldout_residuals.append(residual_quantile)
            heldout_thresholds.append(threshold)
            for row in evaluation:
                accepted = (
                    row.predicted_delta_utility - residual_quantile > 0.0
                    and row.beats_default_probability >= threshold
                )
                if accepted:
                    heldout_accepted += 1
                    heldout_successes += int(row.realized_delta_utility > 0.0)
        policies.append(
            CandidateUncertaintyPolicy(
                candidate_id,
                len(candidate_rows),
                max(heldout_residuals),
                max(heldout_thresholds),
                _wilson_lower_bound(heldout_successes, heldout_accepted),
            )
        )
    if not policies:
        raise RouterUncertaintyError("no candidate meets cross-fit minimum support")
    return CrossFitRouterUncertaintyArtifact(
        base_router_sha256=base_router_sha256,
        candidate_schema_sha256=router_candidate_schema_sha256(candidates),
        descriptor_schema_sha256=descriptor_schema_sha256,
        outcome_ledger_sha256=_sha256([row.ledger_record() for row in sorted(outcomes, key=lambda row: row.identity)]),
        fold_ledger_sha256=_sha256(sorted(source_folds.items())),
        frozen_holdout_sha256=_sha256(sorted(frozen)),
        residual_quantile_level=level,
        minimum_support=minimum_support,
        policies=tuple(policies),
    )


__all__ = [
    "CandidateUncertaintyPolicy", "CrossFitRouterUncertaintyArtifact",
    "ROUTER_UNCERTAINTY_SCHEMA_VERSION", "RouterOutcomeRow", "RouterUncertaintyError",
    "fit_crossfit_router_uncertainty", "router_candidate_schema_sha256",
    "router_descriptor_schema_sha256",
]

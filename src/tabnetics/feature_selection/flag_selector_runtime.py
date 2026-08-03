"""Runtime model and artifact loader for the flag-selector pathway."""

from __future__ import annotations

import ast
import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

try:  # pragma: no cover - torch availability depends on environment
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


DEFAULT_FLAG_SELECTOR_THRESHOLD = 0.55
DEFAULT_FLAG_SELECTOR_FALLBACK = "v16_ref"
ARTIFACT_SCHEMA_VERSION = 1

_CLASSIFICATION_ALIAS_MAP: Dict[str, str] = {
    "selection_mode": "classification_selection_mode",
    "backend": "classification_backend",
    "model_candidates": "model_candidates",
    "exclude_model_candidates": "exclude_model_candidates",
    "oracle_k": "classifier_oracle_k",
    "oracle_weighting_mode": "classifier_oracle_weighting_mode",
    "runtime_containment_enabled": "model_cv_runtime_containment_enabled",
    "runtime_max_candidates": "model_cv_runtime_max_candidates",
    "runtime_high_p_over_n_threshold": "model_cv_runtime_high_p_over_n_threshold",
    "runtime_high_class_threshold": "model_cv_runtime_high_class_threshold",
    "runtime_min_class_count_threshold": "model_cv_runtime_min_class_count_threshold",
    "lr_max_iter": "model_cv_lr_max_iter",
    "use_hybrid_score": "model_cv_use_hybrid_score",
    "hybrid_balanced_weight": "model_cv_balanced_weight",
    "hybrid_macro_f1_weight": "model_cv_macro_f1_weight",
    "conformal_enabled": "classifier_conformal_enabled",
    "conformal_alpha": "classifier_conformal_alpha",
    "conformal_calibration_fraction": "classifier_conformal_calibration_fraction",
    "conformal_min_calibration": "classifier_conformal_min_calibration",
    "conformal_method": "classifier_conformal_method",
}


def _require_torch() -> None:
    if torch is None:
        raise ImportError("flag_selector_v1 requires torch to be installed")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if not np.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _parse_structured_text(text: str) -> Any:
    candidate = str(text or "").strip()
    if not candidate:
        return candidate
    if candidate.startswith("{") or candidate.startswith("[") or candidate.startswith("("):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except Exception:
                continue
    return candidate


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        rounded = round(float(value))
        if abs(float(value) - rounded) < 1e-12:
            return int(rounded)
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        low = text.lower()
        if low in {"true", "false"}:
            return low == "true"
        if low in {"none", "null", "nan"}:
            return None
        parsed = _parse_structured_text(text)
        if parsed is not text:
            return _clean_scalar(parsed)
        return text
    if isinstance(value, (list, tuple)):
        return [_clean_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _clean_scalar(v) for k, v in value.items()}
    return value


def _feature_vector_from_descriptor(descriptor: Any, feature_names: Sequence[str]) -> Dict[str, float]:
    if hasattr(descriptor, "to_feature_dict"):
        data = dict(getattr(descriptor, "to_feature_dict")())
    elif isinstance(descriptor, Mapping):
        maybe_nested = descriptor.get("feature_vector") if isinstance(descriptor.get("feature_vector"), Mapping) else None
        data = dict(maybe_nested or descriptor)
    else:
        raise TypeError("descriptor must be a mapping or expose to_feature_dict()")
    return {str(name): _safe_float(data.get(name, 0.0), 0.0) for name in feature_names}


def _descriptor_metadata(descriptor: Any, key: str, default: str) -> str:
    if hasattr(descriptor, key):
        value = getattr(descriptor, key)
        return str(value if value is not None else default) or default
    if isinstance(descriptor, Mapping):
        return str(descriptor.get(key, default) or default)
    return str(default)


def _softmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    shifted = arr - float(np.max(arr))
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    if denom <= 0.0:
        return np.full(arr.shape, 1.0 / max(arr.size, 1), dtype=float)
    return exp / denom


@dataclass(frozen=True)
class CalibrationTable:
    """Piecewise-linear confidence calibration table."""

    x_thresholds: Tuple[float, ...] = tuple()
    y_thresholds: Tuple[float, ...] = tuple()

    def transform(self, value: float) -> float:
        if not self.x_thresholds or not self.y_thresholds:
            return float(np.clip(value, 0.0, 1.0))
        return float(
            np.interp(
                float(np.clip(value, 0.0, 1.0)),
                np.asarray(self.x_thresholds, dtype=float),
                np.asarray(self.y_thresholds, dtype=float),
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_thresholds": list(self.x_thresholds),
            "y_thresholds": list(self.y_thresholds),
        }

    @classmethod
    def from_regressor(cls, reg: IsotonicRegression) -> "CalibrationTable":
        return cls(
            x_thresholds=tuple(float(v) for v in getattr(reg, "X_thresholds_", [])),
            y_thresholds=tuple(float(v) for v in getattr(reg, "y_thresholds_", [])),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationTable":
        return cls(
            x_thresholds=tuple(float(v) for v in list(payload.get("x_thresholds", []) or [])),
            y_thresholds=tuple(float(v) for v in list(payload.get("y_thresholds", []) or [])),
        )


@dataclass
class RankedProfile:
    profile: str
    predicted_regret: float
    probability: float
    override_dict: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectorOutput:
    profile: str
    override_dict: Dict[str, Any]
    confidence: float
    ranked_topk: List[RankedProfile]
    raw_profile: str
    raw_confidence: float
    fallback_profile: str
    selector_used: bool
    abstained: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_snapshot(self) -> Dict[str, Any]:
        topk = [item.profile for item in self.ranked_topk]
        return {
            "selector_predicted_profile": str(self.profile),
            "selector_raw_profile": str(self.raw_profile),
            "selector_confidence": float(self.confidence),
            "selector_raw_confidence": float(self.raw_confidence),
            "selector_used": bool(self.selector_used),
            "selector_abstain": bool(self.abstained),
            "selector_fallback_profile": str(self.fallback_profile),
            "selector_ranked_topk": list(topk),
            "selector_override_count": int(len(self.override_dict or {})),
            "meta_learning_profile_selected": str(self.profile),
            "meta_learning_profile_raw": str(self.raw_profile),
            "meta_learning_confidence": float(self.confidence),
            "meta_learning_fallback_applied": bool(self.abstained),
            "meta_learning_candidate_profiles": list(topk),
            "meta_learning_fallback_profile": str(self.fallback_profile),
        }


@dataclass
class FlagSelectorConfig:
    hidden_dim: int = 128
    n_res_blocks: int = 3
    dropout: float = 0.10
    profile_embedding_dim: int = 32
    context_embedding_dim: int = 16
    domain_embedding_dropout: float = 0.50
    tier_embedding_dropout: float = 0.50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    margin: float = 0.01
    regret_lambda: float = 1.0
    wall_time_lambda: float = 0.10
    aux_cv_lambda: float = 0.10
    aux_n_features_lambda: float = 0.05
    aux_tier_lambda: float = 0.05
    confidence_threshold: float = DEFAULT_FLAG_SELECTOR_THRESHOLD
    fallback_profile: str = DEFAULT_FLAG_SELECTOR_FALLBACK
    max_negative_pairs_per_group: int = 4
    max_epochs: int = 32
    batch_size: int = 256
    top_k: int = 5


if torch is not None:

    class _ResBlock(nn.Module):
        def __init__(self, dim: int, dropout: float) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            h = F.gelu(self.fc1(h))
            h = self.drop(h)
            h = self.fc2(h)
            h = self.drop(h)
            return x + h


    class FlagSelectorNet(nn.Module):
        """Pairwise regret model over (descriptor, profile, domain, tier)."""

        def __init__(
            self,
            n_features: int,
            n_profiles: int,
            n_domains: int,
            n_tiers: int,
            config: Optional[FlagSelectorConfig] = None,
        ) -> None:
            _require_torch()
            super().__init__()
            cfg = config or FlagSelectorConfig()
            self.config = cfg
            self.profile_embedding = nn.Embedding(max(n_profiles, 1), cfg.profile_embedding_dim)
            self.domain_embedding = nn.Embedding(max(n_domains, 1), cfg.context_embedding_dim)
            self.tier_embedding = nn.Embedding(max(n_tiers, 1), cfg.context_embedding_dim)
            input_dim = (
                int(n_features)
                + cfg.profile_embedding_dim
                + cfg.context_embedding_dim
                + cfg.context_embedding_dim
            )
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            self.trunk = nn.Sequential(*[_ResBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.n_res_blocks)])
            self.trunk_norm = nn.LayerNorm(cfg.hidden_dim)
            self.head_regret = nn.Linear(cfg.hidden_dim, 1)
            self.head_wall_time = nn.Linear(cfg.hidden_dim, 1)
            self.head_cv_score = nn.Linear(cfg.hidden_dim, 1)
            self.head_n_features = nn.Linear(cfg.hidden_dim, 1)
            self.head_tier = nn.Linear(cfg.hidden_dim, max(n_tiers, 1))

        def _drop_context(self, emb: torch.Tensor, p: float) -> torch.Tensor:
            if not self.training or p <= 0.0:
                return emb
            mask = (torch.rand(emb.shape[0], 1, device=emb.device) > p).float()
            scale = 1.0 / max(1.0 - p, 1e-6)
            return emb * mask * scale

        def forward(
            self,
            descriptor: torch.Tensor,
            profile_idx: torch.Tensor,
            domain_idx: torch.Tensor,
            tier_idx: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            prof = self.profile_embedding(profile_idx)
            dom = self._drop_context(self.domain_embedding(domain_idx), self.config.domain_embedding_dropout)
            tier = self._drop_context(self.tier_embedding(tier_idx), self.config.tier_embedding_dropout)
            h = torch.cat([descriptor, prof, dom, tier], dim=-1)
            h = self.input_proj(h)
            h = self.trunk(h)
            h = self.trunk_norm(h)
            return {
                "regret": self.head_regret(h).squeeze(-1),
                "wall_time": self.head_wall_time(h).squeeze(-1),
                "cv_score": self.head_cv_score(h).squeeze(-1),
                "n_features": self.head_n_features(h).squeeze(-1),
                "tier_logits": self.head_tier(h),
            }

else:  # pragma: no cover - import guard path

    class FlagSelectorNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("flag_selector_v1 requires torch to be installed")


@dataclass
class FlagSelector:
    """Artifact-backed runtime predictor with an optional local training path."""

    config: FlagSelectorConfig = field(default_factory=FlagSelectorConfig)
    random_state: int = 42
    model_: Any = field(default=None, init=False, repr=False)
    feature_names_: Tuple[str, ...] = field(default_factory=tuple, init=False)
    profile_labels_: Tuple[str, ...] = field(default_factory=tuple, init=False)
    domain_labels_: Tuple[str, ...] = field(default_factory=tuple, init=False)
    tier_labels_: Tuple[str, ...] = field(default_factory=tuple, init=False)
    feature_mean_: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32), init=False, repr=False)
    feature_scale_: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=np.float32), init=False, repr=False)
    calibration_: Optional[CalibrationTable] = field(default=None, init=False, repr=False)
    profile_overrides_: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    training_metadata_: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    fitted_: bool = field(default=False, init=False)

    def _device(self) -> Any:
        _require_torch()
        return torch.device("cpu")

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        arr = np.asarray(matrix, dtype=np.float32)
        if self.feature_mean_.size == 0 or self.feature_scale_.size == 0:
            return arr
        return (arr - self.feature_mean_) / self.feature_scale_

    def _profile_override(self, profile_id: str) -> Dict[str, Any]:
        payload = self.profile_overrides_.get(str(profile_id), {})
        return {
            "enabled_methods": list(payload.get("enabled_methods", []) or []),
            "config_overrides": dict(payload.get("config_overrides", {}) or {}),
            "classification_overrides": dict(payload.get("classification_overrides", {}) or {}),
        }

    def _predict_scores(self, descriptor: Any, *, apply_calibration: bool = True) -> Tuple[np.ndarray, np.ndarray, float, float]:
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("FlagSelector is not fitted.")
        feature_map = _feature_vector_from_descriptor(descriptor, self.feature_names_)
        x_row = np.asarray(
            [[float(feature_map.get(name, 0.0) or 0.0) for name in self.feature_names_]],
            dtype=np.float32,
        )
        x_row = self._normalize_matrix(x_row)
        domain = _descriptor_metadata(descriptor, "domain", "unknown")
        tier = _descriptor_metadata(descriptor, "effective_tier", "unknown")
        domain_idx = self.domain_labels_.index(domain) if domain in self.domain_labels_ else 0
        tier_idx = self.tier_labels_.index(tier) if tier in self.tier_labels_ else 0
        profile_idx = np.arange(len(self.profile_labels_), dtype=np.int64)

        self.model_.eval()
        with torch.no_grad():
            descriptor_t = torch.from_numpy(np.repeat(x_row, len(profile_idx), axis=0)).float().to(self._device())
            profile_t = torch.from_numpy(profile_idx).long().to(self._device())
            domain_t = torch.full((len(profile_idx),), int(domain_idx), dtype=torch.long, device=self._device())
            tier_t = torch.full((len(profile_idx),), int(tier_idx), dtype=torch.long, device=self._device())
            out = self.model_(descriptor_t, profile_t, domain_t, tier_t)
            regrets = out["regret"].detach().cpu().numpy().astype(float)
        raw_probs = _softmax(-regrets)
        raw_conf = float(np.max(raw_probs)) if raw_probs.size else 0.0
        conf = float(self.calibration_.transform(raw_conf)) if (apply_calibration and self.calibration_ is not None) else raw_conf
        return regrets, raw_probs, raw_conf, conf

    def predict(self, descriptor: Any) -> SelectorOutput:
        regrets, probs, raw_conf, conf = self._predict_scores(descriptor)
        if probs.size == 0:
            raise RuntimeError("FlagSelector has no candidate profiles.")
        order = np.argsort(regrets)
        raw_idx = int(order[0])
        raw_profile = str(self.profile_labels_[raw_idx])
        fallback_profile = str(self.config.fallback_profile or DEFAULT_FLAG_SELECTOR_FALLBACK)
        abstained = bool(conf < float(self.config.confidence_threshold))
        final_profile = fallback_profile if abstained else raw_profile
        ranked = [
            RankedProfile(
                profile=str(self.profile_labels_[int(idx)]),
                predicted_regret=float(regrets[int(idx)]),
                probability=float(probs[int(idx)]),
                override_dict=self._profile_override(str(self.profile_labels_[int(idx)])),
            )
            for idx in order[: max(1, int(self.config.top_k))]
        ]
        selected_override = self._profile_override(final_profile)
        if abstained and final_profile == fallback_profile and final_profile not in self.profile_overrides_:
            selected_override = {}
        return SelectorOutput(
            profile=str(final_profile),
            override_dict=selected_override,
            confidence=float(conf),
            ranked_topk=ranked,
            raw_profile=str(raw_profile),
            raw_confidence=float(raw_conf),
            fallback_profile=str(fallback_profile),
            selector_used=not abstained,
            abstained=bool(abstained),
            metadata={
                "domain": _descriptor_metadata(descriptor, "domain", "unknown"),
                "effective_tier": _descriptor_metadata(descriptor, "effective_tier", "unknown"),
            },
        )

    def fit(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        feature_names: Optional[Sequence[str]] = None,
        calibration_rows: Optional[Sequence[Mapping[str, Any]]] = None,
        profile_overrides: Optional[Mapping[str, Dict[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "FlagSelector":
        _require_torch()
        prepared = [dict(row) for row in rows]
        if len(prepared) < 2:
            raise ValueError("FlagSelector.fit requires at least two training rows.")
        if feature_names is None:
            nested = prepared[0].get("feature_vector", {})
            feature_names = tuple(sorted(str(key) for key in dict(nested).keys()))
        self.feature_names_ = tuple(str(name) for name in feature_names or ())
        if not self.feature_names_:
            raise ValueError("FlagSelector.fit requires a non-empty feature schema.")

        profiles = sorted({str(row.get("profile", "") or "") for row in prepared if str(row.get("profile", "") or "")})
        domains = sorted({str(row.get("domain", "unknown") or "unknown") for row in prepared}) or ["unknown"]
        tiers = sorted({str(row.get("effective_tier", "unknown") or "unknown") for row in prepared}) or ["unknown"]
        self.profile_labels_ = tuple(profiles)
        self.domain_labels_ = tuple(domains)
        self.tier_labels_ = tuple(tiers)
        fallback_profile = str(self.config.fallback_profile or DEFAULT_FLAG_SELECTOR_FALLBACK)
        if fallback_profile not in self.profile_labels_:
            best_counts: Dict[str, int] = {}
            for row in prepared:
                best = str(row.get("best_profile", "") or "")
                if best:
                    best_counts[best] = best_counts.get(best, 0) + 1
            if fallback_profile not in best_counts and best_counts:
                fallback_profile = max(best_counts.items(), key=lambda item: item[1])[0]
            elif not best_counts and self.profile_labels_:
                fallback_profile = self.profile_labels_[0]
            self.config.fallback_profile = fallback_profile
        self.profile_overrides_ = {
            str(key): {
                "enabled_methods": list(dict(value).get("enabled_methods", []) or []),
                "config_overrides": dict(dict(value).get("config_overrides", {}) or {}),
                "classification_overrides": dict(dict(value).get("classification_overrides", {}) or {}),
            }
            for key, value in dict(profile_overrides or {}).items()
            if str(key).strip()
        }

        profile_index = {name: idx for idx, name in enumerate(self.profile_labels_)}
        domain_index = {name: idx for idx, name in enumerate(self.domain_labels_)}
        tier_index = {name: idx for idx, name in enumerate(self.tier_labels_)}

        X = np.asarray(
            [
                [float(dict(row.get("feature_vector", {})).get(name, 0.0) or 0.0) for name in self.feature_names_]
                for row in prepared
            ],
            dtype=np.float32,
        )
        self.feature_mean_ = np.mean(X, axis=0).astype(np.float32)
        self.feature_scale_ = np.std(X, axis=0).astype(np.float32)
        self.feature_scale_[self.feature_scale_ < 1e-6] = 1.0
        Xn = self._normalize_matrix(X)
        profile_idx = np.asarray([profile_index[str(row["profile"])] for row in prepared], dtype=np.int64)
        domain_idx = np.asarray([domain_index[str(row.get("domain", "unknown") or "unknown")] for row in prepared], dtype=np.int64)
        tier_idx = np.asarray([tier_index[str(row.get("effective_tier", "unknown") or "unknown")] for row in prepared], dtype=np.int64)
        regret = np.asarray([_safe_float(row.get("regret", 0.0), 0.0) for row in prepared], dtype=np.float32)
        wall_time = np.asarray([np.log1p(_safe_float(row.get("wall_time_sec", 0.0), 0.0)) for row in prepared], dtype=np.float32)
        cv_score = np.asarray([_safe_float(row.get("model_cv_score", 0.0), 0.0) for row in prepared], dtype=np.float32)
        n_features = np.asarray([np.log1p(max(0.0, _safe_float(row.get("n_features_selected", 0.0), 0.0))) for row in prepared], dtype=np.float32)
        tier_target = tier_idx.copy()

        group_to_rows: Dict[str, List[int]] = {}
        for idx, row in enumerate(prepared):
            group_key = str(row.get("group_key", f"{row.get('dataset_id', '')}::{row.get('seed', 0)}"))
            group_to_rows.setdefault(group_key, []).append(idx)
        pair_pos: List[int] = []
        pair_neg: List[int] = []
        pair_weight: List[float] = []
        rng = np.random.default_rng(int(self.random_state))
        for indices in group_to_rows.values():
            if len(indices) < 2:
                continue
            ordered = sorted(indices, key=lambda idx: (float(regret[idx]), str(prepared[idx].get("profile", ""))))
            pos = int(ordered[0])
            negatives = ordered[1:]
            if len(negatives) > int(self.config.max_negative_pairs_per_group):
                negatives = list(rng.choice(negatives, size=int(self.config.max_negative_pairs_per_group), replace=False))
            for neg in negatives:
                pair_pos.append(pos)
                pair_neg.append(int(neg))
                pair_weight.append(max(1e-4, float(regret[int(neg)] - regret[pos] + self.config.margin)))

        self.model_ = FlagSelectorNet(
            n_features=len(self.feature_names_),
            n_profiles=max(len(self.profile_labels_), 1),
            n_domains=max(len(self.domain_labels_), 1),
            n_tiers=max(len(self.tier_labels_), 1),
            config=self.config,
        ).to(self._device())
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        tensor_X = torch.from_numpy(Xn).float().to(self._device())
        tensor_profile = torch.from_numpy(profile_idx).long().to(self._device())
        tensor_domain = torch.from_numpy(domain_idx).long().to(self._device())
        tensor_tier = torch.from_numpy(tier_idx).long().to(self._device())
        tensor_regret = torch.from_numpy(regret).float().to(self._device())
        tensor_wall = torch.from_numpy(wall_time).float().to(self._device())
        tensor_cv = torch.from_numpy(cv_score).float().to(self._device())
        tensor_nfeat = torch.from_numpy(n_features).float().to(self._device())
        tensor_tier_target = torch.from_numpy(tier_target).long().to(self._device())
        batch_size = max(16, int(self.config.batch_size))

        for _epoch in range(max(1, int(self.config.max_epochs))):
            self.model_.train()
            perm = rng.permutation(len(prepared))
            for start in range(0, len(prepared), batch_size):
                batch_idx = np.asarray(perm[start : start + batch_size], dtype=np.int64)
                optimizer.zero_grad(set_to_none=True)
                out = self.model_(
                    tensor_X[batch_idx],
                    tensor_profile[batch_idx],
                    tensor_domain[batch_idx],
                    tensor_tier[batch_idx],
                )
                loss = (
                    float(self.config.regret_lambda) * F.mse_loss(out["regret"], tensor_regret[batch_idx])
                    + float(self.config.wall_time_lambda) * F.mse_loss(out["wall_time"], tensor_wall[batch_idx])
                    + float(self.config.aux_cv_lambda) * F.mse_loss(out["cv_score"], tensor_cv[batch_idx])
                    + float(self.config.aux_n_features_lambda) * F.mse_loss(out["n_features"], tensor_nfeat[batch_idx])
                    + float(self.config.aux_tier_lambda) * F.cross_entropy(out["tier_logits"], tensor_tier_target[batch_idx])
                )
                loss.backward()
                optimizer.step()
            if pair_pos:
                sampled = np.arange(len(pair_pos), dtype=np.int64)
                if sampled.size > len(prepared) * 4:
                    sampled = rng.choice(sampled, size=len(prepared) * 4, replace=False)
                for start in range(0, sampled.size, batch_size):
                    pair_batch = sampled[start : start + batch_size]
                    pos_idx = torch.from_numpy(np.asarray([pair_pos[int(i)] for i in pair_batch], dtype=np.int64)).long().to(self._device())
                    neg_idx = torch.from_numpy(np.asarray([pair_neg[int(i)] for i in pair_batch], dtype=np.int64)).long().to(self._device())
                    pair_w = torch.from_numpy(np.asarray([pair_weight[int(i)] for i in pair_batch], dtype=np.float32)).float().to(self._device())
                    optimizer.zero_grad(set_to_none=True)
                    out_pos = self.model_(tensor_X[pos_idx], tensor_profile[pos_idx], tensor_domain[pos_idx], tensor_tier[pos_idx])
                    out_neg = self.model_(tensor_X[neg_idx], tensor_profile[neg_idx], tensor_domain[neg_idx], tensor_tier[neg_idx])
                    pair_loss = F.relu(float(self.config.margin) + out_pos["regret"] - out_neg["regret"]) * pair_w
                    pair_loss = torch.mean(pair_loss)
                    pair_loss.backward()
                    optimizer.step()

        self.fitted_ = True
        self.training_metadata_ = dict(metadata or {})
        calib_rows = [dict(row) for row in list(calibration_rows or prepared)]
        calibration = self._fit_calibration(calib_rows)
        if calibration is not None:
            self.calibration_ = calibration
        self._tune_threshold(calib_rows)
        return self

    def _fit_calibration(self, rows: Sequence[Mapping[str, Any]]) -> Optional[CalibrationTable]:
        if not rows:
            return None
        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            group_key = str(row.get("group_key", f"{row.get('dataset_id', '')}::{row.get('seed', 0)}"))
            grouped.setdefault(group_key, []).append(row)
        xs: List[float] = []
        ys: List[float] = []
        for group_rows in grouped.values():
            exemplar = group_rows[0]
            try:
                output = self.predict(exemplar)
            except Exception:
                continue
            actual_best = min(
                group_rows,
                key=lambda item: (float(item.get("regret", 0.0) or 0.0), str(item.get("profile", "") or "")),
            )
            xs.append(float(output.raw_confidence))
            ys.append(1.0 if str(actual_best.get("profile", "") or "") == str(output.raw_profile) else 0.0)
        if len(xs) < 4 or len(set(round(v, 6) for v in ys)) < 2:
            return None
        reg = IsotonicRegression(out_of_bounds="clip")
        reg.fit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
        return CalibrationTable.from_regressor(reg)

    def _tune_threshold(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            group_key = str(row.get("group_key", f"{row.get('dataset_id', '')}::{row.get('seed', 0)}"))
            grouped.setdefault(group_key, []).append(row)
        candidates = sorted(
            {
                round(float(self.predict(group_rows[0]).confidence), 4)
                for group_rows in grouped.values()
            }
            | {round(float(self.config.confidence_threshold), 4)}
        )
        if not candidates:
            return
        best_threshold = float(self.config.confidence_threshold)
        best_regret = float("inf")
        for threshold in candidates:
            total_regret = 0.0
            count = 0
            for group_rows in grouped.values():
                exemplar = group_rows[0]
                output = self.predict(exemplar)
                chosen_profile = str(output.raw_profile)
                if float(output.confidence) < float(threshold):
                    chosen_profile = str(self.config.fallback_profile or DEFAULT_FLAG_SELECTOR_FALLBACK)
                chosen_row = next(
                    (row for row in group_rows if str(row.get("profile", "") or "") == chosen_profile),
                    None,
                )
                if chosen_row is None:
                    continue
                total_regret += float(chosen_row.get("regret", 0.0) or 0.0)
                count += 1
            if count <= 0:
                continue
            mean_regret = total_regret / float(count)
            if mean_regret < best_regret:
                best_regret = mean_regret
                best_threshold = float(threshold)
        self.config.confidence_threshold = float(np.clip(best_threshold, 0.0, 1.0))

    def save(self, model_path: Path | str, *, extra_metadata: Optional[Mapping[str, Any]] = None) -> Path:
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("FlagSelector must be fitted before save().")
        path = Path(model_path)
        path.mkdir(parents=True, exist_ok=True)
        weights_path = path / "model.pt"
        calibration_path = path / "calibration.json"
        manifest_path = path / "manifest.json"
        overrides_path = path / "profile_overrides.json"
        torch.save(self.model_.state_dict(), weights_path)
        calibration_payload = self.calibration_.to_dict() if self.calibration_ is not None else {"x_thresholds": [], "y_thresholds": []}
        calibration_path.write_text(json.dumps(calibration_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        overrides_path.write_text(json.dumps(self.profile_overrides_, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "feature_names": list(self.feature_names_),
            "profile_labels": list(self.profile_labels_),
            "domain_labels": list(self.domain_labels_),
            "tier_labels": list(self.tier_labels_),
            "feature_mean": self.feature_mean_.tolist(),
            "feature_scale": self.feature_scale_.tolist(),
            "fallback_profile": str(self.config.fallback_profile or DEFAULT_FLAG_SELECTOR_FALLBACK),
            "confidence_threshold": float(self.config.confidence_threshold),
            "config": {
                "hidden_dim": int(self.config.hidden_dim),
                "n_res_blocks": int(self.config.n_res_blocks),
                "dropout": float(self.config.dropout),
                "profile_embedding_dim": int(self.config.profile_embedding_dim),
                "context_embedding_dim": int(self.config.context_embedding_dim),
                "domain_embedding_dropout": float(self.config.domain_embedding_dropout),
                "tier_embedding_dropout": float(self.config.tier_embedding_dropout),
                "learning_rate": float(self.config.learning_rate),
                "weight_decay": float(self.config.weight_decay),
                "margin": float(self.config.margin),
                "regret_lambda": float(self.config.regret_lambda),
                "wall_time_lambda": float(self.config.wall_time_lambda),
                "aux_cv_lambda": float(self.config.aux_cv_lambda),
                "aux_n_features_lambda": float(self.config.aux_n_features_lambda),
                "aux_tier_lambda": float(self.config.aux_tier_lambda),
                "top_k": int(self.config.top_k),
            },
            "training_metadata": {**self.training_metadata_, **dict(extra_metadata or {})},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, model_path: Path | str) -> "FlagSelector":
        _require_torch()
        path = Path(model_path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        config = FlagSelectorConfig(**dict(manifest.get("config", {}) or {}))
        config.fallback_profile = str(manifest.get("fallback_profile", config.fallback_profile) or config.fallback_profile)
        config.confidence_threshold = float(
            manifest.get("confidence_threshold", config.confidence_threshold) or config.confidence_threshold
        )
        selector = cls(config=config)
        selector.feature_names_ = tuple(str(v) for v in list(manifest.get("feature_names", []) or []))
        selector.profile_labels_ = tuple(str(v) for v in list(manifest.get("profile_labels", []) or []))
        selector.domain_labels_ = tuple(str(v) for v in list(manifest.get("domain_labels", []) or [])) or ("unknown",)
        selector.tier_labels_ = tuple(str(v) for v in list(manifest.get("tier_labels", []) or [])) or ("unknown",)
        selector.feature_mean_ = np.asarray(list(manifest.get("feature_mean", []) or []), dtype=np.float32)
        selector.feature_scale_ = np.asarray(list(manifest.get("feature_scale", []) or []), dtype=np.float32)
        selector.feature_scale_[selector.feature_scale_ < 1e-6] = 1.0
        selector.training_metadata_ = dict(manifest.get("training_metadata", {}) or {})
        selector.model_ = FlagSelectorNet(
            n_features=max(len(selector.feature_names_), 1),
            n_profiles=max(len(selector.profile_labels_), 1),
            n_domains=max(len(selector.domain_labels_), 1),
            n_tiers=max(len(selector.tier_labels_), 1),
            config=config,
        ).to(selector._device())
        state = torch.load(path / "model.pt", map_location=selector._device())
        selector.model_.load_state_dict(state)
        calibration_payload = json.loads((path / "calibration.json").read_text(encoding="utf-8"))
        calibration = CalibrationTable.from_dict(calibration_payload)
        selector.calibration_ = calibration if calibration.x_thresholds else None
        if (path / "profile_overrides.json").exists():
            selector.profile_overrides_ = json.loads((path / "profile_overrides.json").read_text(encoding="utf-8"))
        selector.fitted_ = True
        return selector


def _set_classification_attr(config: Any, key: str, value: Any) -> None:
    cls_cfg = getattr(config, "classification", None)
    if cls_cfg is not None:
        try:
            setattr(cls_cfg, key, value)
        except Exception:
            pass
    flat_key = _CLASSIFICATION_ALIAS_MAP.get(str(key))
    if flat_key is not None:
        setattr(config, flat_key, value)
        return
    setattr(config, str(key), value)


def apply_selector_output(config: Any, output: SelectorOutput) -> Any:
    """Apply a selector decision onto a DFFSConfig-like object."""

    payload = dict(output.override_dict or {})
    enabled_methods = list(payload.get("enabled_methods", []) or [])
    if enabled_methods:
        setattr(config, "enabled_methods", tuple(str(m) for m in enabled_methods if str(m).strip()))
    for key, value in dict(payload.get("config_overrides", {}) or {}).items():
        setattr(config, str(key), _clean_scalar(value))
    for key, value in dict(payload.get("classification_overrides", {}) or {}).items():
        _set_classification_attr(config, str(key), _clean_scalar(value))
    return config

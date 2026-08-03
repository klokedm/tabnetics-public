"""Feature-selector-style classifier for one unlabeled query record.

The model reuses the current FS teacher trunk for feature tokens, then adds a
classification head that scores one raw query record against per-class support
statistics.  Population-side targets may still be used to train the selector
checkpoint, but this classifier consumes only support/sample-derived statistics
plus the query record at inference time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - non-torch import environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from .tabentics_diakrino_fs_teacher import (
    TabenticsDiakrinoFSTeacher,
    TabenticsDiakrinoFSTeacherBatch,
    TabenticsDiakrinoFSTeacherConfig,
    TabenticsDiakrinoFSTeacherOutputs,
    _BalancedCosineInteraction,
    _SwiGLUContextEncoder,
    _SwiGLUResidualBlock,
    _SwiGLUScoringHead,
    _apply_feature_rope,
    _fourier_position_features,
    _match_last_dim,
    compute_fs_screening_features,
)


JsonDict = dict[str, Any]


def _ensure_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("tabentics_diakrino_fs_classifier requires torch to be installed.")


def _summarize_prefixes(keys: list[str]) -> JsonDict:
    counts: dict[str, int] = {}
    for key in keys:
        prefix = key.split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _checkpoint_config_dict(checkpoint: JsonDict) -> JsonDict:
    raw = checkpoint.get("model_config") or checkpoint.get("config") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _teacher_config_from_checkpoint(checkpoint: JsonDict) -> tuple[TabenticsDiakrinoFSTeacherConfig | None, JsonDict]:
    raw = _checkpoint_config_dict(checkpoint)
    if not raw:
        return None, {"matched": False, "reason": "checkpoint_config_missing"}
    if "refiner_steps" not in raw and "fs_refiner_steps" in raw:
        raw["refiner_steps"] = raw["fs_refiner_steps"]
    valid = {field.name for field in fields(TabenticsDiakrinoFSTeacherConfig)}
    used = {key: value for key, value in raw.items() if key in valid}
    if not used:
        return None, {"matched": False, "reason": "checkpoint_config_has_no_teacher_fields"}
    teacher_cfg = TabenticsDiakrinoFSTeacherConfig(**used)
    return teacher_cfg, {
        "matched": True,
        "used_checkpoint_config_keys": sorted(used),
        "ignored_checkpoint_config_keys": sorted(set(raw) - set(used)),
    }


def _classifier_config_matching_teacher(
    base: TabenticsDiakrinoFSClassifierConfig | None,
    teacher_cfg: TabenticsDiakrinoFSTeacherConfig | None,
) -> tuple[TabenticsDiakrinoFSClassifierConfig, JsonDict]:
    cfg = base or TabenticsDiakrinoFSClassifierConfig()
    if teacher_cfg is None:
        return cfg, {"matched": False, "reason": "teacher_config_unavailable"}
    overrides: JsonDict = {
        "architecture_variant": str(teacher_cfg.architecture_variant),
        "d_model": int(teacher_cfg.d_model),
        "n_heads": int(teacher_cfg.n_heads),
        "context_layers": int(teacher_cfg.context_layers),
        "ffn_expansion": int(teacher_cfg.ffn_expansion),
        "dropout": float(teacher_cfg.dropout),
        "max_classes": int(teacher_cfg.max_classes),
        "feature_stats_dim": int(teacher_cfg.feature_stats_dim),
        "screening_feature_dim": int(teacher_cfg.screening_feature_dim),
        "prior_scale_init": float(teacher_cfg.prior_scale_init),
        "screening_scale_init": float(teacher_cfg.screening_scale_init),
        "residual_scale_init": float(teacher_cfg.residual_scale_init),
        "series_scale_init": float(teacher_cfg.series_scale_init),
        "fusion_scale_init": float(teacher_cfg.fusion_scale_init),
        "calibration_bias_init": float(teacher_cfg.calibration_bias_init),
        "use_distribution_series": bool(teacher_cfg.use_distribution_series),
        "series_samples": int(teacher_cfg.series_samples),
        "fs_refiner_steps": int(teacher_cfg.refiner_steps),
        "refiner_scale_init": float(teacher_cfg.refiner_scale_init),
        "attention_backend": str(teacher_cfg.attention_backend),
        "attn_qk_norm": bool(teacher_cfg.attn_qk_norm),
        "interaction_mode": str(teacher_cfg.interaction_mode),
        "interaction_heads": int(teacher_cfg.interaction_heads),
        "interaction_temp_init": float(teacher_cfg.interaction_temp_init),
        "interaction_temp_learnable": bool(teacher_cfg.interaction_temp_learnable),
        "interaction_scale_channel": bool(teacher_cfg.interaction_scale_channel),
        "context_candidate_topk": int(teacher_cfg.context_candidate_topk),
        "feature_position_encoding": str(teacher_cfg.feature_position_encoding),
        "position_encoding_scale_init": float(teacher_cfg.position_encoding_scale_init),
        "position_frequency_bands": int(teacher_cfg.position_frequency_bands),
        "feature_metadata_dim": int(teacher_cfg.feature_metadata_dim),
        "feature_metadata_scale_init": float(teacher_cfg.feature_metadata_scale_init),
        "sample_class_feature_dim": int(teacher_cfg.sample_class_feature_dim),
        "class_extras_scale_init": float(teacher_cfg.class_extras_scale_init),
        "class_extras_logit_scale_init": float(teacher_cfg.class_extras_logit_scale_init),
        "local_residual_scale_init": float(teacher_cfg.local_residual_scale_init),
        "max_feature_tokens": int(teacher_cfg.max_feature_tokens),
        "clip_value": float(teacher_cfg.clip_value),
        "eps": float(teacher_cfg.eps),
    }
    return replace(cfg, **overrides), {
        "matched": True,
        "overridden_classifier_fields": sorted(overrides),
        "teacher_geometry": {
            "d_model": int(teacher_cfg.d_model),
            "n_heads": int(teacher_cfg.n_heads),
            "context_layers": int(teacher_cfg.context_layers),
            "max_classes": int(teacher_cfg.max_classes),
            "max_feature_tokens": int(teacher_cfg.max_feature_tokens),
            "joint_sample_mode": str(teacher_cfg.joint_sample_mode),
            "query_icl_mode": str(teacher_cfg.query_icl_mode),
            "redundancy_head_mode": str(teacher_cfg.redundancy_head_mode),
            "conformal_head_mode": str(teacher_cfg.conformal_head_mode),
        },
    }


def _pad_class_stats(values: torch.Tensor, target_classes: int) -> torch.Tensor:
    target = max(1, int(target_classes))
    current = int(values.shape[2])
    if current == target:
        return values
    if current > target:
        return values[:, :, :target, :]
    return F.pad(values, (0, 0, 0, target - current))


def _pad_feature_class_mask(values: torch.Tensor, target_classes: int) -> torch.Tensor:
    target = max(1, int(target_classes))
    current = int(values.shape[-1])
    if current == target:
        return values
    if current > target:
        return values[..., :target]
    return F.pad(values, (0, target - current))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, *, dim: int, eps: float) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=dim).clamp(min=float(eps))
    return (values * mask_f).sum(dim=dim) / denom


def _probability_logit(value: float, *, eps: float) -> float:
    prob = min(1.0 - float(eps), max(float(eps), float(value)))
    return math.log(prob / (1.0 - prob))


def compute_fs_classifier_feature_stats(
    marginal_stats: torch.Tensor,
    class_stats: torch.Tensor,
    class_stats_valid: torch.Tensor,
    feature_valid_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build the FS teacher's 10-channel feature stats from class-ready inputs."""

    _ensure_torch()
    marginal = _match_last_dim(torch.nan_to_num(marginal_stats, nan=0.0, posinf=0.0, neginf=0.0), 5)
    stats = torch.nan_to_num(class_stats, nan=0.0, posinf=0.0, neginf=0.0)
    valid = class_stats_valid.to(dtype=torch.bool) & feature_valid_mask.to(dtype=torch.bool).unsqueeze(-1)
    zero = torch.zeros_like(stats[..., 0])
    fisher = torch.where(valid, stats[..., 17].clamp(min=0.0), zero).amax(dim=2)
    shifts = torch.where(valid, stats[..., 18].abs(), zero)
    max_shift = shifts.amax(dim=2)
    mean_shift = shifts.sum(dim=2) / valid.to(dtype=stats.dtype).sum(dim=2).clamp(min=1.0)
    log_std_ratio = torch.where(valid, stats[..., 19].abs(), zero).amax(dim=2)
    priors = torch.where(valid, stats[..., 16].clamp(min=0.0), zero)
    priors = priors / priors.sum(dim=2, keepdim=True).clamp(min=float(eps))
    entropy = -(priors * torch.log(priors.clamp(min=float(eps)))).sum(dim=2)
    class_balance = entropy / math.log(max(2, int(stats.shape[2])))
    class_summary = torch.stack(
        [
            fisher,
            max_shift,
            mean_shift,
            log_std_ratio,
            class_balance.clamp(min=0.0, max=1.0),
        ],
        dim=-1,
    )
    out = torch.cat([marginal, class_summary], dim=-1)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(feature_valid_mask.unsqueeze(-1), out, torch.zeros_like(out))


@dataclass(frozen=True)
class TabenticsDiakrinoFSClassifierConfig:
    """Configuration for the FS-trunk single-query classifier."""

    architecture_variant: str = "swiglu_fusion_v2"
    d_model: int = 576
    n_heads: int = 8
    context_layers: int = 3
    query_layers: int = 1
    class_layers: int = 1
    ffn_expansion: int = 2
    dropout: float = 0.1
    max_classes: int = 25
    feature_stats_dim: int = 10
    class_stats_dim: int = 24
    screening_feature_dim: int = 18
    query_value_dim: int = 4
    relative_feature_dim: int = 8
    max_feature_tokens: int = 1024
    clip_value: float = 8.0
    eps: float = 1e-6
    use_distribution_series: bool = False
    series_samples: int = 16
    fs_refiner_steps: int = 4
    attention_backend: str = "auto"
    # --- KVarN-inspired interaction preconditioning (opt-in; no-ops at defaults).
    # See KVARN_INTERACTION_IMPLEMENTATION_PLAN.md. ---
    attn_qk_norm: bool = False
    interaction_mode: str = "bilinear"  # "bilinear" (current) | "cosine"
    interaction_heads: int = 0  # 0 -> use n_heads
    interaction_temp_init: float = 0.07
    interaction_temp_learnable: bool = True
    interaction_scale_channel: bool = True
    context_candidate_topk: int = 0
    feature_position_encoding: str = "rope_fourier"
    position_encoding_scale_init: float = 0.0
    position_frequency_bands: int = 16
    classifier_adapter_layers: int = 0
    classifier_adapter_scale_init: float = 0.05
    classifier_adapter_use_query: bool = True
    classifier_adapter_use_class_summary: bool = True
    feature_metadata_dim: int = 0
    feature_metadata_scale_init: float = 0.0
    sample_class_feature_dim: int = 0
    class_extras_scale_init: float = 0.10
    class_extras_logit_scale_init: float = 0.05
    prior_scale_init: float = 1.0
    screening_scale_init: float = 0.25
    residual_scale_init: float = 0.10
    series_scale_init: float = 0.10
    fusion_scale_init: float = 0.25
    calibration_bias_init: float = 0.0
    refiner_scale_init: float = 0.10
    selector_logit_scale_init: float = 0.0
    selector_temperature: float = 1.0
    selector_stochastic: bool = False
    local_residual_scale_init: float = 0.0
    feature_gate_scale_init: float = 0.25
    evidence_scale_init: float = 1.0
    class_prior_scale_init: float = 0.20
    label_smoothing: float = 0.02
    classification_weight: float = 1.0
    evidence_auxiliary_weight: float = 0.0
    evidence_auxiliary_detach_gates: bool = True
    selector_signal_weight: float = 0.0
    selector_signal_listwise_weight: float = 0.0
    selector_signal_top_fraction: float = 0.10
    selector_signal_min_targets: int = 8
    selector_relevance_weight: float = 0.0
    selector_relevance_listwise_weight: float = 0.0
    gate_cardinality_weight: float = 0.0
    gate_target_fraction: float = 0.02
    gate_entropy_weight: float = 0.0
    feature_class_gate_bias_init: float = 0.0
    feature_class_gate_bias_from_target: bool = False
    # --- v-next-C trunk consumption (default ON by owner direction, 2026-07-05).
    # When enabled AND the batch carries raw support rows, the FS-teacher selector
    # receives the real support set + query row so its joint sample channel and
    # query-ICL head (#179) run; the ICL logits are blended into the class logits
    # with a learned scale. Graceful fallback: batches without support rows (or
    # trunks without a joint encoder, e.g. step-32000-era checkpoints) take the
    # legacy zero-support path, so enabling this on old callers is a no-op.
    # Set False to force the legacy path. Init 0.1, NOT 0.0: a zero-init blend is
    # gradient-dead for the CE-through-blend path (joint17 lesson, #111/#141). ---
    use_support_joint_channel: bool = True
    query_icl_blend_scale_init: float = 0.10

    def fs_teacher_config(self) -> TabenticsDiakrinoFSTeacherConfig:
        return TabenticsDiakrinoFSTeacherConfig(
            architecture_variant=str(self.architecture_variant),
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            context_layers=int(self.context_layers),
            ffn_expansion=int(self.ffn_expansion),
            dropout=float(self.dropout),
            max_classes=int(self.max_classes),
            feature_stats_dim=int(self.feature_stats_dim),
            screening_feature_dim=int(self.screening_feature_dim),
            prior_scale_init=float(self.prior_scale_init),
            screening_scale_init=float(self.screening_scale_init),
            residual_scale_init=float(self.residual_scale_init),
            series_scale_init=float(self.series_scale_init),
            fusion_scale_init=float(self.fusion_scale_init),
            calibration_bias_init=float(self.calibration_bias_init),
            use_distribution_series=bool(self.use_distribution_series),
            series_samples=int(self.series_samples),
            refiner_steps=int(self.fs_refiner_steps),
            refiner_scale_init=float(self.refiner_scale_init),
            attention_backend=str(self.attention_backend),
            attn_qk_norm=bool(self.attn_qk_norm),
            max_feature_tokens=int(self.max_feature_tokens),
            clip_value=float(self.clip_value),
            eps=float(self.eps),
            selector_logit_scale_init=float(self.selector_logit_scale_init),
            selector_temperature=float(self.selector_temperature),
            selector_stochastic=bool(self.selector_stochastic),
            context_candidate_topk=int(self.context_candidate_topk),
            feature_position_encoding=str(self.feature_position_encoding),
            position_encoding_scale_init=float(self.position_encoding_scale_init),
            position_frequency_bands=int(self.position_frequency_bands),
            feature_metadata_dim=int(self.feature_metadata_dim),
            feature_metadata_scale_init=float(self.feature_metadata_scale_init),
            sample_class_feature_dim=int(self.sample_class_feature_dim),
            class_extras_scale_init=float(self.class_extras_scale_init),
            class_extras_logit_scale_init=float(self.class_extras_logit_scale_init),
            local_residual_scale_init=float(self.local_residual_scale_init),
            teacher_loss_weight=0.0,
            support_prediction_weight=0.0,
            reconstruction_weight=0.0,
            population_reconstruction_weight=0.0,
            population_class_reconstruction_weight=0.0,
        )


@dataclass(frozen=True)
class TabenticsDiakrinoFSClassifierBatch:
    query_values: torch.Tensor
    query_mask: torch.Tensor
    marginal_stats: torch.Tensor
    class_stats: torch.Tensor
    class_stats_valid: torch.Tensor
    feature_valid_mask: torch.Tensor
    query_labels: torch.Tensor | None = None
    class_valid: torch.Tensor | None = None
    feature_indices: torch.Tensor | None = None
    feature_positions: torch.Tensor | None = None
    feature_metadata: torch.Tensor | None = None
    feature_stats_input: torch.Tensor | None = None
    screening_features_input: torch.Tensor | None = None
    sample_class_features_input: torch.Tensor | None = None
    distribution_series_input: torch.Tensor | None = None
    distribution_series_valid: torch.Tensor | None = None
    feature_relevance_targets: torch.Tensor | None = None
    strict_feature_targets: torch.Tensor | None = None
    # Raw scaled support rows for the opt-in v-next-C joint/query-ICL path.
    # Ignored (and safe to omit) unless config.use_support_joint_channel is set.
    support_values: torch.Tensor | None = None  # [B, S, F]
    support_missing: torch.Tensor | None = None  # [B, S, F] bool, True = missing
    support_labels: torch.Tensor | None = None  # [B, S] long, -1 = pad
    support_row_valid: torch.Tensor | None = None  # [B, S] bool


@dataclass(frozen=True)
class TabenticsDiakrinoFSClassifierOutputs:
    class_logits: torch.Tensor
    feature_class_evidence: torch.Tensor
    feature_class_gates: torch.Tensor
    feature_embeddings: torch.Tensor
    class_feature_embeddings: torch.Tensor
    query_feature_embeddings: torch.Tensor
    selector_outputs: TabenticsDiakrinoFSTeacherOutputs
    feature_valid_mask: torch.Tensor
    class_valid: torch.Tensor


@dataclass(frozen=True)
class TabenticsDiakrinoFSClassifierSupportContext:
    """Compact, immutable support state for eval-only native inference."""

    model_identity: int
    selector_outputs: TabenticsDiakrinoFSTeacherOutputs
    feature_tokens: torch.Tensor
    class_stats: torch.Tensor
    class_stats_tokens: torch.Tensor
    class_stats_valid: torch.Tensor
    feature_valid: torch.Tensor
    class_valid: torch.Tensor
    class_prior: torch.Tensor
    joint_support_summary: torch.Tensor | None
    joint_support_row_embeddings: torch.Tensor | None
    joint_support_row_valid: torch.Tensor | None
    joint_support_row_labels: torch.Tensor | None
    joint_feature_tokens: torch.Tensor | None


class TabenticsDiakrinoFSClassifier(nn.Module):
    """FS-teacher trunk plus a 25-way single-record classification head."""

    def __init__(self, config: TabenticsDiakrinoFSClassifierConfig | None = None) -> None:
        _ensure_torch()
        super().__init__()
        self.config = config or TabenticsDiakrinoFSClassifierConfig()
        d_model = int(self.config.d_model)
        hidden_dim = max(d_model, d_model * int(self.config.ffn_expansion))
        self.feature_selector = TabenticsDiakrinoFSTeacher(self.config.fs_teacher_config())
        use_swiglu = str(self.config.architecture_variant).lower() in {"swiglu_fusion_v2", "swiglu"}
        if use_swiglu:
            self.class_stats_encoder = nn.Sequential(
                nn.Linear(int(self.config.class_stats_dim), d_model),
                nn.RMSNorm(d_model),
                _SwiGLUResidualBlock(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout)),
            )
            self.query_value_encoder = nn.Sequential(
                nn.Linear(int(self.config.query_value_dim), d_model),
                nn.RMSNorm(d_model),
                _SwiGLUResidualBlock(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout)),
            )
        else:
            self.class_stats_encoder = nn.Sequential(
                nn.Linear(int(self.config.class_stats_dim), d_model),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, d_model),
                nn.RMSNorm(d_model),
            )
            self.query_value_encoder = nn.Sequential(
                nn.Linear(int(self.config.query_value_dim), d_model),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, d_model),
                nn.RMSNorm(d_model),
            )
        n_heads = max(1, min(int(self.config.n_heads), d_model))
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.query_context: nn.Module | None
        if int(self.config.query_layers) > 0:
            self.query_context = _SwiGLUContextEncoder(
                d_model=d_model,
                n_heads=n_heads,
                hidden_dim=hidden_dim,
                dropout=float(self.config.dropout),
                num_layers=int(self.config.query_layers),
                attention_backend=str(self.config.attention_backend),
                qk_norm=bool(self.config.attn_qk_norm),
            )
        else:
            self.query_context = None
        self.class_context: nn.Module | None
        if int(self.config.class_layers) > 0:
            self.class_context = _SwiGLUContextEncoder(
                d_model=d_model,
                n_heads=n_heads,
                hidden_dim=hidden_dim,
                dropout=float(self.config.dropout),
                num_layers=int(self.config.class_layers),
                attention_backend=str(self.config.attention_backend),
                qk_norm=bool(self.config.attn_qk_norm),
            )
        else:
            self.class_context = None
        self.classifier_adapter_input: nn.Module | None
        self.classifier_adapter: nn.Module | None
        self.classifier_adapter_output_norm: nn.Module | None
        if int(self.config.classifier_adapter_layers) > 0:
            self.classifier_adapter_input = nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.RMSNorm(d_model),
                _SwiGLUResidualBlock(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout)),
            )
            self.classifier_adapter = _SwiGLUContextEncoder(
                d_model=d_model,
                n_heads=n_heads,
                hidden_dim=hidden_dim,
                dropout=float(self.config.dropout),
                num_layers=int(self.config.classifier_adapter_layers),
                attention_backend=str(self.config.attention_backend),
                qk_norm=bool(self.config.attn_qk_norm),
            )
            self.classifier_adapter_output_norm = nn.RMSNorm(d_model)
            self.classifier_adapter_scale = nn.Parameter(torch.tensor(float(self.config.classifier_adapter_scale_init)))
        else:
            self.classifier_adapter_input = None
            self.classifier_adapter = None
            self.classifier_adapter_output_norm = None
            self.classifier_adapter_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.class_projection = nn.Linear(d_model, d_model, bias=False)
        if str(self.config.interaction_mode).lower() == "cosine":
            self.classifier_interaction: nn.Module | None = _BalancedCosineInteraction(
                d_model=d_model,
                heads=int(self.config.interaction_heads) or n_heads,
                temp_init=float(self.config.interaction_temp_init),
                temp_learnable=bool(self.config.interaction_temp_learnable),
                scale_channel=bool(self.config.interaction_scale_channel),
                eps=float(self.config.eps),
            )
        else:
            self.classifier_interaction = None
        self.relative_evidence = nn.Sequential(
            nn.Linear(int(self.config.relative_feature_dim), max(16, d_model // 4)),
            nn.SiLU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(max(16, d_model // 4), 1),
        )
        self.feature_class_gate_head = _SwiGLUScoringHead(
            d_model,
            hidden_dim=hidden_dim,
            dropout=float(self.config.dropout),
        )
        gate_bias = float(self.config.feature_class_gate_bias_init)
        if bool(self.config.feature_class_gate_bias_from_target):
            gate_bias += _probability_logit(float(self.config.gate_target_fraction), eps=float(self.config.eps))
        nn.init.constant_(self.feature_class_gate_head.output.bias, gate_bias)
        self.class_hidden_projection = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, d_model))
        self.query_global_projection = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, d_model))
        self.class_logit_head = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, 1))
        self.feature_gate_scale = nn.Parameter(torch.tensor(float(self.config.feature_gate_scale_init)))
        self.evidence_scale = nn.Parameter(torch.tensor(float(self.config.evidence_scale_init)))
        self.class_prior_scale = nn.Parameter(torch.tensor(float(self.config.class_prior_scale_init)))
        if bool(self.config.use_support_joint_channel):
            self.query_icl_blend_scale = nn.Parameter(torch.tensor(float(self.config.query_icl_blend_scale_init)))
        else:
            self.query_icl_blend_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def _selector_batch(
        self,
        batch: TabenticsDiakrinoFSClassifierBatch,
        *,
        feature_stats: torch.Tensor,
        screening_features: torch.Tensor,
        feature_valid: torch.Tensor,
    ) -> TabenticsDiakrinoFSTeacherBatch:
        batch_size, feature_count = feature_valid.shape
        dtype = batch.query_values.dtype
        device = batch.query_values.device
        use_joint = (
            bool(self.config.use_support_joint_channel)
            and batch.support_values is not None
            and batch.support_labels is not None
        )
        if use_joint:
            support_raw = batch.support_values
            labels_raw = batch.support_labels
            if support_raw.ndim != 3:
                raise ValueError(f"support_values must be 3D [batch, rows, features], got shape {tuple(support_raw.shape)}")
            if int(support_raw.shape[0]) != int(batch_size):
                raise ValueError(
                    "support_values batch size must match query batch size, "
                    f"got {int(support_raw.shape[0])} and {int(batch_size)}"
                )
            if int(support_raw.shape[2]) != int(feature_count):
                raise ValueError(
                    "support_values feature count must match feature_valid_mask, "
                    f"got {int(support_raw.shape[2])} and {int(feature_count)}"
                )
            if labels_raw.ndim != 2:
                raise ValueError(f"support_labels must be 2D [batch, rows], got shape {tuple(labels_raw.shape)}")
            if int(labels_raw.shape[0]) != int(batch_size) or int(labels_raw.shape[1]) != int(support_raw.shape[1]):
                raise ValueError(
                    "support_labels shape must match support_values batch and row dimensions, "
                    f"got {tuple(labels_raw.shape)} for support_values {tuple(support_raw.shape)}"
                )
            if batch.support_missing is not None:
                missing_raw = batch.support_missing
                if missing_raw.ndim != 3:
                    raise ValueError(f"support_missing must be 3D [batch, rows, features], got shape {tuple(missing_raw.shape)}")
                if tuple(missing_raw.shape) != tuple(support_raw.shape):
                    raise ValueError(
                        "support_missing shape must match support_values shape, "
                        f"got {tuple(missing_raw.shape)} and {tuple(support_raw.shape)}"
                    )
            if batch.support_row_valid is not None:
                row_valid_raw = batch.support_row_valid
                if row_valid_raw.ndim != 2:
                    raise ValueError(f"support_row_valid must be 2D [batch, rows], got shape {tuple(row_valid_raw.shape)}")
                if tuple(row_valid_raw.shape) != tuple(labels_raw.shape):
                    raise ValueError(
                        "support_row_valid shape must match support_labels shape, "
                        f"got {tuple(row_valid_raw.shape)} and {tuple(labels_raw.shape)}"
                    )
            support = torch.nan_to_num(
                support_raw.to(dtype=dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0
            )
            support_mask = (
                batch.support_missing.to(dtype=torch.bool, device=device)
                if batch.support_missing is not None
                else torch.zeros_like(support, dtype=torch.bool)
            )
            support_labels = labels_raw.to(dtype=torch.long, device=device)
            support_valid = (
                batch.support_row_valid.to(dtype=torch.bool, device=device)
                if batch.support_row_valid is not None
                else support_labels >= 0
            )
            joint_query_values = torch.nan_to_num(batch.query_values, nan=0.0, posinf=0.0, neginf=0.0)
            joint_query_mask = batch.query_mask.to(dtype=torch.bool)
        else:
            support = torch.zeros((batch_size, 1, feature_count), dtype=dtype, device=device)
            support_mask = torch.ones((batch_size, 1, feature_count), dtype=torch.bool, device=device)
            support_valid = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
            support_labels = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
            joint_query_values = None
            joint_query_mask = None
        teacher_targets = torch.zeros((batch_size, feature_count), dtype=dtype, device=device)
        series_input = batch.distribution_series_input
        series_valid = batch.distribution_series_valid
        if bool(self.config.use_distribution_series) and (series_input is None or series_valid is None):
            slot_count = int(self.config.max_classes) + 1
            series_dim = 2 * max(1, int(self.config.series_samples)) + 2
            series_input = torch.zeros((batch_size, feature_count, slot_count, series_dim), dtype=dtype, device=device)
            series_valid = torch.zeros((batch_size, feature_count, slot_count), dtype=torch.bool, device=device)
        return TabenticsDiakrinoFSTeacherBatch(
            support=support,
            support_mask=support_mask,
            support_valid=support_valid,
            support_labels=support_labels,
            feature_valid_mask=feature_valid,
            teacher_targets=teacher_targets,
            feature_indices=batch.feature_indices,
            feature_positions=batch.feature_positions,
            feature_metadata=batch.feature_metadata,
            feature_stats_input=feature_stats,
            screening_features_input=screening_features,
            sample_class_features_input=batch.sample_class_features_input,
            distribution_series_input=series_input,
            distribution_series_valid=series_valid,
            query_values=joint_query_values,
            query_mask=joint_query_mask,
        )

    def _relative_channels(self, batch: TabenticsDiakrinoFSClassifierBatch, class_stats: torch.Tensor) -> torch.Tensor:
        eps = float(self.config.eps)
        values = torch.nan_to_num(batch.query_values, nan=0.0, posinf=0.0, neginf=0.0)
        observed = (~batch.query_mask.to(dtype=torch.bool)).to(dtype=values.dtype)
        query = values.unsqueeze(-1).expand(-1, -1, class_stats.shape[2])
        obs = observed.unsqueeze(-1).expand_as(query)
        mean = class_stats[..., 1]
        std = class_stats[..., 2].abs().clamp(min=eps)
        min_value = class_stats[..., 3]
        max_value = class_stats[..., 4]
        median = class_stats[..., 21] if class_stats.shape[-1] > 21 else mean
        iqr = class_stats[..., 22].abs().clamp(min=eps) if class_stats.shape[-1] > 22 else std
        z = torch.clamp((query - mean) / std, -float(self.config.clip_value), float(self.config.clip_value))
        robust_z = torch.clamp((query - median) / iqr, -float(self.config.clip_value), float(self.config.clip_value))
        in_range = ((query >= min_value) & (query <= max_value)).to(dtype=values.dtype)
        channels = torch.stack(
            [
                torch.clamp(query, -float(self.config.clip_value), float(self.config.clip_value)),
                obs,
                z,
                z.abs(),
                robust_z,
                robust_z.abs(),
                in_range,
                1.0 - obs,
            ],
            dim=-1,
        )
        return torch.nan_to_num(_match_last_dim(channels, int(self.config.relative_feature_dim)), nan=0.0, posinf=0.0, neginf=0.0)

    def _position_encode_query_values(
        self,
        query_value_tokens: torch.Tensor,
        *,
        feature_positions: torch.Tensor | None,
        feature_valid: torch.Tensor,
    ) -> torch.Tensor:
        if feature_positions is None:
            return torch.where(feature_valid.unsqueeze(-1), query_value_tokens, torch.zeros_like(query_value_tokens))
        mode = str(self.config.feature_position_encoding).lower()
        tokens = query_value_tokens
        if self.feature_selector.position_encoder is not None and mode in {"fourier", "rope", "rope_fourier"}:
            position_features = _fourier_position_features(
                feature_positions.to(dtype=tokens.dtype),
                bands=int(self.config.position_frequency_bands),
            ).to(dtype=tokens.dtype)
            position_latent = self.feature_selector.position_encoder(position_features)
            position_latent = torch.where(feature_valid.unsqueeze(-1), position_latent, torch.zeros_like(position_latent))
            tokens = tokens + self.feature_selector.position_encoding_scale.to(dtype=tokens.dtype) * position_latent
        if mode in {"rope", "rope_fourier"}:
            tokens = _apply_feature_rope(tokens, feature_positions.to(dtype=tokens.dtype))
        return torch.where(feature_valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))

    def _classifier_adapter_features(
        self,
        feature_tokens: torch.Tensor,
        *,
        query_value_tokens: torch.Tensor,
        class_stats_tokens: torch.Tensor,
        class_stats_valid: torch.Tensor,
        feature_valid: torch.Tensor,
    ) -> torch.Tensor:
        if self.classifier_adapter is None or self.classifier_adapter_input is None or self.classifier_adapter_output_norm is None:
            return feature_tokens
        zeros = torch.zeros_like(feature_tokens)
        query_context = query_value_tokens if bool(self.config.classifier_adapter_use_query) else zeros
        if bool(self.config.classifier_adapter_use_class_summary):
            class_summary = _masked_mean(class_stats_tokens, class_stats_valid, dim=2, eps=float(self.config.eps))
        else:
            class_summary = zeros
        adapter_input = torch.cat([feature_tokens, query_context, class_summary], dim=-1)
        adapter_tokens = self.classifier_adapter_input(adapter_input)
        adapter_tokens = self.classifier_adapter(adapter_tokens, valid_mask=feature_valid)
        adapter_tokens = self.classifier_adapter_output_norm(adapter_tokens)
        fused = feature_tokens + self.classifier_adapter_scale.to(dtype=feature_tokens.dtype) * adapter_tokens
        return torch.where(feature_valid.unsqueeze(-1), fused, zeros)

    def _class_prior_logits(
        self,
        class_stats: torch.Tensor,
        *,
        class_stats_valid: torch.Tensor,
        feature_valid: torch.Tensor,
        class_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = class_stats_valid & feature_valid.unsqueeze(-1) & class_valid.unsqueeze(1)
        counts = torch.where(valid, class_stats[..., 0].clamp(min=0.0), torch.zeros_like(class_stats[..., 0]))
        denom = valid.to(dtype=counts.dtype).sum(dim=1).clamp(min=1.0)
        class_counts = counts.sum(dim=1) / denom
        class_counts = torch.where(class_valid, class_counts.clamp(min=float(self.config.eps)), torch.zeros_like(class_counts))
        log_counts = torch.log(class_counts.clamp(min=float(self.config.eps)))
        normalizer = torch.logsumexp(torch.where(class_valid, log_counts, torch.full_like(log_counts, -30.0)), dim=1, keepdim=True)
        return torch.where(class_valid, log_counts - normalizer, torch.zeros_like(log_counts))

    @staticmethod
    def _expand_cached_value(value: Any, batch_size: int) -> Any:
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == 1:
            return value.expand(int(batch_size), *value.shape[1:])
        if isinstance(value, tuple):
            return tuple(TabenticsDiakrinoFSClassifier._expand_cached_value(item, batch_size) for item in value)
        return value

    def _expanded_selector_outputs(
        self,
        outputs: TabenticsDiakrinoFSTeacherOutputs,
        *,
        batch_size: int,
        query_icl_logits: torch.Tensor | None,
    ) -> TabenticsDiakrinoFSTeacherOutputs:
        updates = {
            field.name: self._expand_cached_value(getattr(outputs, field.name), int(batch_size))
            for field in fields(TabenticsDiakrinoFSTeacherOutputs)
        }
        updates["query_icl_logits"] = query_icl_logits
        for name in (
            "query_class_logits",
            "query_labels",
            "query_class_valid",
            "query_feature_class_evidence",
            "query_feature_class_gates",
        ):
            updates[name] = None
        return replace(outputs, **updates)

    def _forward_query_from_static(
        self,
        batch: TabenticsDiakrinoFSClassifierBatch,
        *,
        selector_outputs: TabenticsDiakrinoFSTeacherOutputs,
        feature_tokens: torch.Tensor,
        class_stats: torch.Tensor,
        class_stats_tokens: torch.Tensor,
        class_stats_valid: torch.Tensor,
        feature_valid: torch.Tensor,
        class_valid: torch.Tensor,
        class_prior: torch.Tensor | None = None,
    ) -> TabenticsDiakrinoFSClassifierOutputs:
        values = torch.nan_to_num(batch.query_values, nan=0.0, posinf=0.0, neginf=0.0)
        observed = (~batch.query_mask.to(dtype=torch.bool)).to(dtype=values.dtype)
        query_features = torch.stack(
            [
                torch.clamp(values, -float(self.config.clip_value), float(self.config.clip_value)),
                observed,
                values.abs().clamp(max=float(self.config.clip_value)),
                1.0 - observed,
            ],
            dim=-1,
        )
        query_features = _match_last_dim(query_features, int(self.config.query_value_dim))
        query_value_tokens = self.query_value_encoder(query_features)
        query_value_tokens = self._position_encode_query_values(
            query_value_tokens,
            feature_positions=selector_outputs.feature_positions,
            feature_valid=feature_valid,
        )
        feature_tokens = self._classifier_adapter_features(
            feature_tokens,
            query_value_tokens=query_value_tokens,
            class_stats_tokens=class_stats_tokens,
            class_stats_valid=class_stats_valid,
            feature_valid=feature_valid,
        )

        class_tokens = class_stats_tokens + feature_tokens.unsqueeze(2)
        class_tokens = torch.where(class_stats_valid.unsqueeze(-1), class_tokens, torch.zeros_like(class_tokens))
        query_tokens = query_value_tokens + feature_tokens
        query_tokens = torch.where(feature_valid.unsqueeze(-1), query_tokens, torch.zeros_like(query_tokens))
        if self.query_context is not None:
            query_tokens = self.query_context(query_tokens, valid_mask=feature_valid)

        query_projected = self.query_projection(query_tokens)
        class_projected = self.class_projection(class_tokens)
        if self.classifier_interaction is not None:
            bilinear = self.classifier_interaction(query_projected, class_projected)
        else:
            bilinear = torch.einsum("bfd,bfkd->bfk", query_projected, class_projected)
            bilinear = bilinear / math.sqrt(max(1, int(self.config.d_model)))
        relative_logits = self.relative_evidence(self._relative_channels(batch, class_stats)).squeeze(-1)
        feature_class_evidence = self.evidence_scale * bilinear + relative_logits

        fs_gate_logits = selector_outputs.logits.unsqueeze(-1)
        gate_logits = self.feature_class_gate_head(class_tokens).squeeze(-1) + self.feature_gate_scale * fs_gate_logits
        gates = torch.sigmoid(gate_logits)
        valid_fk = class_stats_valid & feature_valid.unsqueeze(-1) & class_valid.unsqueeze(1)
        gates = torch.where(valid_fk, gates, torch.zeros_like(gates))
        feature_class_evidence = torch.where(valid_fk, feature_class_evidence, torch.zeros_like(feature_class_evidence))
        weighted_sum = (feature_class_evidence * gates).sum(dim=1)
        gate_mass = gates.sum(dim=1).clamp(min=1.0)
        pooled_evidence = weighted_sum / torch.sqrt(gate_mass)

        class_hidden = (class_tokens * gates.unsqueeze(-1)).sum(dim=1) / gate_mass.unsqueeze(-1)
        class_hidden = self.class_hidden_projection(class_hidden)
        query_global = _masked_mean(query_tokens, feature_valid, dim=1, eps=float(self.config.eps))
        query_global = self.query_global_projection(query_global)
        class_query_hidden = class_hidden + query_global.unsqueeze(1)
        if self.class_context is not None:
            class_query_hidden = self.class_context(class_query_hidden, valid_mask=class_valid)

        prior = class_prior
        if prior is None:
            prior = self._class_prior_logits(
                class_stats,
                class_stats_valid=class_stats_valid,
                feature_valid=feature_valid,
                class_valid=class_valid,
            )
        logits = pooled_evidence + self.class_logit_head(class_query_hidden).squeeze(-1) + self.class_prior_scale * prior
        icl_logits = getattr(selector_outputs, "query_icl_logits", None)
        if icl_logits is not None and bool(self.config.use_support_joint_channel):
            icl = torch.nan_to_num(icl_logits.to(dtype=logits.dtype), nan=0.0, posinf=-30.0, neginf=-30.0)
            icl = _pad_feature_class_mask(icl, int(logits.shape[-1]))
            icl = torch.where(class_valid, icl.clamp(min=-30.0, max=30.0), torch.zeros_like(icl))
            logits = logits + self.query_icl_blend_scale.to(dtype=logits.dtype) * icl
        logits = torch.where(class_valid, logits, torch.full_like(logits, -30.0))

        return TabenticsDiakrinoFSClassifierOutputs(
            class_logits=logits,
            feature_class_evidence=feature_class_evidence,
            feature_class_gates=gates,
            feature_embeddings=feature_tokens,
            class_feature_embeddings=class_tokens,
            query_feature_embeddings=query_tokens,
            selector_outputs=selector_outputs,
            feature_valid_mask=feature_valid,
            class_valid=class_valid,
        )

    def forward(self, batch: TabenticsDiakrinoFSClassifierBatch) -> TabenticsDiakrinoFSClassifierOutputs:
        feature_valid = batch.feature_valid_mask.to(dtype=torch.bool)
        class_stats = _match_last_dim(
            torch.nan_to_num(batch.class_stats, nan=0.0, posinf=0.0, neginf=0.0),
            int(self.config.class_stats_dim),
        )
        class_stats = _pad_class_stats(class_stats, int(self.config.max_classes))
        class_stats_valid = _pad_feature_class_mask(batch.class_stats_valid.to(dtype=torch.bool), int(self.config.max_classes))
        class_valid = (
            _pad_feature_class_mask(batch.class_valid.to(dtype=torch.bool), int(self.config.max_classes))
            if batch.class_valid is not None
            else class_stats_valid.any(dim=1)
        )
        class_stats_valid = class_stats_valid & feature_valid.unsqueeze(-1) & class_valid.unsqueeze(1)
        if batch.feature_stats_input is not None:
            feature_stats = _match_last_dim(
                torch.nan_to_num(batch.feature_stats_input, nan=0.0, posinf=0.0, neginf=0.0),
                int(self.config.feature_stats_dim),
            )
            feature_stats = torch.where(feature_valid.unsqueeze(-1), feature_stats, torch.zeros_like(feature_stats))
        else:
            feature_stats = compute_fs_classifier_feature_stats(
                batch.marginal_stats,
                class_stats,
                class_stats_valid,
                feature_valid,
                eps=float(self.config.eps),
            )
            feature_stats = _match_last_dim(feature_stats, int(self.config.feature_stats_dim))
        if batch.screening_features_input is not None:
            screening_features = _match_last_dim(
                torch.nan_to_num(batch.screening_features_input, nan=0.0, posinf=0.0, neginf=0.0),
                int(self.config.screening_feature_dim),
            )
            screening_features = torch.where(feature_valid.unsqueeze(-1), screening_features, torch.zeros_like(screening_features))
        else:
            screening_features = compute_fs_screening_features(
                _match_last_dim(feature_stats, 10),
                feature_valid_mask=feature_valid,
                eps=float(self.config.eps),
            )
            screening_features = _match_last_dim(screening_features, int(self.config.screening_feature_dim))

        selector_batch = self._selector_batch(
            batch,
            feature_stats=feature_stats,
            screening_features=screening_features,
            feature_valid=feature_valid,
        )
        selector_outputs = self.feature_selector(selector_batch)
        feature_tokens = selector_outputs.feature_embeddings
        feature_tokens = torch.where(feature_valid.unsqueeze(-1), feature_tokens, torch.zeros_like(feature_tokens))

        class_stats_tokens = self.class_stats_encoder(class_stats)
        class_stats_tokens = torch.where(class_stats_valid.unsqueeze(-1), class_stats_tokens, torch.zeros_like(class_stats_tokens))
        return self._forward_query_from_static(
            batch,
            selector_outputs=selector_outputs,
            feature_tokens=feature_tokens,
            class_stats=class_stats,
            class_stats_tokens=class_stats_tokens,
            class_stats_valid=class_stats_valid,
            feature_valid=feature_valid,
            class_valid=class_valid,
        )

    def prepare_support_context(
        self,
        batch: TabenticsDiakrinoFSClassifierBatch,
    ) -> TabenticsDiakrinoFSClassifierSupportContext:
        """Encode static support once for the native eval-only serving path."""

        if self.training:
            raise RuntimeError("support context preparation requires model.eval()")
        if torch.is_grad_enabled():
            raise RuntimeError("support context preparation requires gradients to be disabled")
        if int(batch.query_values.shape[0]) != 1:
            raise ValueError("support context preparation requires a single representative query row")
        outputs = self.forward(batch)
        selector_outputs = outputs.selector_outputs
        feature_valid = batch.feature_valid_mask.to(dtype=torch.bool)
        class_stats = _pad_class_stats(
            _match_last_dim(
                torch.nan_to_num(batch.class_stats, nan=0.0, posinf=0.0, neginf=0.0),
                int(self.config.class_stats_dim),
            ),
            int(self.config.max_classes),
        )
        class_stats_valid = _pad_feature_class_mask(
            batch.class_stats_valid.to(dtype=torch.bool), int(self.config.max_classes)
        )
        class_valid = (
            _pad_feature_class_mask(batch.class_valid.to(dtype=torch.bool), int(self.config.max_classes))
            if batch.class_valid is not None
            else class_stats_valid.any(dim=1)
        )
        class_stats_valid = class_stats_valid & feature_valid.unsqueeze(-1) & class_valid.unsqueeze(1)
        class_stats_tokens = self.class_stats_encoder(class_stats)
        class_stats_tokens = torch.where(
            class_stats_valid.unsqueeze(-1), class_stats_tokens, torch.zeros_like(class_stats_tokens)
        )
        class_prior = self._class_prior_logits(
            class_stats,
            class_stats_valid=class_stats_valid,
            feature_valid=feature_valid,
            class_valid=class_valid,
        )
        return TabenticsDiakrinoFSClassifierSupportContext(
            model_identity=id(self),
            selector_outputs=selector_outputs,
            feature_tokens=selector_outputs.feature_embeddings,
            class_stats=class_stats,
            class_stats_tokens=class_stats_tokens,
            class_stats_valid=class_stats_valid,
            feature_valid=feature_valid,
            class_valid=class_valid,
            class_prior=class_prior,
            joint_support_summary=selector_outputs.joint_support_summary,
            joint_support_row_embeddings=selector_outputs.joint_support_row_embeddings,
            joint_support_row_valid=selector_outputs.joint_support_row_valid,
            joint_support_row_labels=selector_outputs.joint_support_row_labels,
            joint_feature_tokens=selector_outputs.joint_feature_tokens,
        )

    def forward_from_support_context(
        self,
        query_values: torch.Tensor,
        query_mask: torch.Tensor,
        context: TabenticsDiakrinoFSClassifierSupportContext,
    ) -> TabenticsDiakrinoFSClassifierOutputs:
        """Score query rows without rerunning the support cell encoder."""

        if self.training:
            raise RuntimeError("cached support inference requires model.eval()")
        if torch.is_grad_enabled():
            raise RuntimeError("cached support inference requires gradients to be disabled")
        if int(context.model_identity) != id(self):
            raise ValueError("cached support context belongs to a different classifier instance")
        if query_values.ndim != 2 or query_mask.ndim != 2 or tuple(query_values.shape) != tuple(query_mask.shape):
            raise ValueError("query_values and query_mask must have matching 2D [batch, features] shapes")
        batch_size, feature_count = query_values.shape
        if int(feature_count) != int(context.feature_valid.shape[1]):
            raise ValueError("query feature count does not match cached support context")

        def expand(value: torch.Tensor) -> torch.Tensor:
            return value.expand(int(batch_size), *value.shape[1:])

        feature_valid = expand(context.feature_valid)
        class_valid = expand(context.class_valid)
        class_stats = expand(context.class_stats)
        class_stats_valid = expand(context.class_stats_valid)
        class_stats_tokens = expand(context.class_stats_tokens)
        feature_tokens = expand(context.feature_tokens)
        class_prior = expand(context.class_prior)
        icl_logits: torch.Tensor | None = None
        joint_encoder = self.feature_selector.joint_sample_encoder
        query_icl_head = self.feature_selector.query_icl_head
        if (
            bool(self.config.use_support_joint_channel)
            and joint_encoder is not None
            and query_icl_head is not None
            and context.joint_support_row_embeddings is not None
            and context.joint_support_row_valid is not None
            and context.joint_support_row_labels is not None
            and context.joint_feature_tokens is not None
        ):
            joint_feature_tokens = expand(context.joint_feature_tokens)
            query_embedding = joint_encoder.query_row_embedding(
                query_values=query_values,
                query_mask=query_mask,
                feature_tokens=joint_feature_tokens,
                feature_valid=feature_valid,
            )
            icl_logits = query_icl_head(
                row_embeddings=expand(context.joint_support_row_embeddings),
                row_valid=expand(context.joint_support_row_valid),
                row_labels=expand(context.joint_support_row_labels),
                query_embedding=query_embedding,
            )
            icl_logits = torch.where(class_valid, icl_logits, torch.full_like(icl_logits, -30.0))
        selector_outputs = self._expanded_selector_outputs(
            context.selector_outputs,
            batch_size=int(batch_size),
            query_icl_logits=icl_logits,
        )
        query_batch = TabenticsDiakrinoFSClassifierBatch(
            query_values=query_values,
            query_mask=query_mask,
            marginal_stats=expand(context.selector_outputs.feature_stats[..., :5]),
            class_stats=class_stats,
            class_stats_valid=class_stats_valid,
            feature_valid_mask=feature_valid,
            class_valid=class_valid,
        )
        return self._forward_query_from_static(
            query_batch,
            selector_outputs=selector_outputs,
            feature_tokens=feature_tokens,
            class_stats=class_stats,
            class_stats_tokens=class_stats_tokens,
            class_stats_valid=class_stats_valid,
            feature_valid=feature_valid,
            class_valid=class_valid,
            class_prior=class_prior,
        )

    @classmethod
    def from_fs_teacher_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config: TabenticsDiakrinoFSClassifierConfig | None = None,
        map_location: str | torch.device = "cpu",
        match_fs_teacher_config: bool = False,
        teacher_config_overrides: JsonDict | None = None,
    ) -> tuple["TabenticsDiakrinoFSClassifier", JsonDict]:
        """Create a classifier and load the reusable FS teacher trunk weights.

        ``teacher_config_overrides`` patches runtime-only fields (for example
        ``joint_sample_checkpoint``) onto the checkpoint-derived teacher config;
        unknown keys are ignored so weight-shape-relevant geometry stays intact.
        """

        _ensure_torch()
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
        source_state: JsonDict = checkpoint.get("model_state_dict", checkpoint)
        teacher_cfg, teacher_config_report = (
            _teacher_config_from_checkpoint(checkpoint) if isinstance(checkpoint, dict) else (None, {"matched": False})
        )
        if teacher_cfg is not None and teacher_config_overrides:
            valid_fields = {field.name for field in fields(TabenticsDiakrinoFSTeacherConfig)}
            applied = {key: value for key, value in dict(teacher_config_overrides).items() if key in valid_fields}
            if applied:
                teacher_cfg = replace(teacher_cfg, **applied)
                teacher_config_report = {**teacher_config_report, "runtime_overrides": sorted(applied)}
        config_report: JsonDict = {"matched": False, "reason": "not_requested"}
        effective_config = config
        if bool(match_fs_teacher_config) or config is None:
            effective_config, config_report = _classifier_config_matching_teacher(config, teacher_cfg)
        model = cls(effective_config)
        if bool(match_fs_teacher_config) and teacher_cfg is not None:
            model.feature_selector = TabenticsDiakrinoFSTeacher(teacher_cfg)
        target_state = model.state_dict()
        mapped: dict[str, torch.Tensor] = {}
        skipped_shape: list[str] = []
        loaded_source_keys: list[str] = []
        for source_key, value in source_state.items():
            target_key = source_key if source_key.startswith("feature_selector.") else f"feature_selector.{source_key}"
            if target_key not in target_state:
                continue
            if tuple(target_state[target_key].shape) != tuple(value.shape):
                skipped_shape.append(
                    f"{source_key} -> {target_key}: {tuple(value.shape)} vs {tuple(target_state[target_key].shape)}"
                )
                continue
            mapped[target_key] = value
            loaded_source_keys.append(source_key)
        load_result = model.load_state_dict(mapped, strict=False)
        discarded = sorted(set(source_state) - set(loaded_source_keys))
        report: JsonDict = {
            "checkpoint_path": str(checkpoint_path),
            "loaded_source_keys": sorted(loaded_source_keys),
            "loaded_count": len(loaded_source_keys),
            "discarded_count": len(discarded),
            "discarded_prefixes": _summarize_prefixes(discarded),
            "skipped_shape": skipped_shape,
            "missing_after_partial_load": list(load_result.missing_keys),
            "unexpected_after_partial_load": list(load_result.unexpected_keys),
            "source_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "source_step": checkpoint.get("step") if isinstance(checkpoint, dict) else None,
            "match_fs_teacher_config": bool(match_fs_teacher_config),
            "teacher_config_report": teacher_config_report,
            "classifier_config_match_report": config_report,
            "effective_config": model.config.__dict__,
        }
        return model, report


def _classifier_signal_scores(
    batch: TabenticsDiakrinoFSClassifierBatch,
    outputs: TabenticsDiakrinoFSClassifierOutputs,
    cfg: TabenticsDiakrinoFSClassifierConfig,
) -> torch.Tensor:
    feature_valid = outputs.feature_valid_mask.to(dtype=torch.bool)
    score = outputs.class_logits.new_zeros(feature_valid.shape)
    if batch.screening_features_input is not None:
        screening = _match_last_dim(
            torch.nan_to_num(batch.screening_features_input, nan=0.0, posinf=0.0, neginf=0.0),
            int(cfg.screening_feature_dim),
        )
        fisher = screening[..., 0].clamp(min=0.0)
        fisher_rank = screening[..., 14].clamp(min=0.0, max=1.0) if screening.shape[-1] > 14 else fisher
        shift_rank = screening[..., 15].clamp(min=0.0, max=1.0) if screening.shape[-1] > 15 else score
        score = torch.maximum(score, fisher_rank + 0.25 * shift_rank + 0.05 * fisher)
    class_stats_valid = _pad_feature_class_mask(batch.class_stats_valid.to(dtype=torch.bool), int(cfg.max_classes))
    class_stats = _pad_class_stats(
        _match_last_dim(torch.nan_to_num(batch.class_stats, nan=0.0, posinf=0.0, neginf=0.0), int(cfg.class_stats_dim)),
        int(cfg.max_classes),
    )
    fisher = torch.where(class_stats_valid, class_stats[..., 17].clamp(min=0.0), torch.zeros_like(class_stats[..., 17])).amax(dim=2)
    shift = torch.where(class_stats_valid, class_stats[..., 18].abs(), torch.zeros_like(class_stats[..., 18])).amax(dim=2)
    score = torch.maximum(score, fisher + 0.25 * shift)
    score = torch.where(feature_valid, score, torch.zeros_like(score))
    row_max = score.amax(dim=1, keepdim=True).clamp(min=float(cfg.eps))
    return torch.where(feature_valid, score / row_max, torch.zeros_like(score))


def _classifier_signal_targets(
    batch: TabenticsDiakrinoFSClassifierBatch,
    outputs: TabenticsDiakrinoFSClassifierOutputs,
    cfg: TabenticsDiakrinoFSClassifierConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    score = _classifier_signal_scores(batch, outputs, cfg)
    mask = outputs.feature_valid_mask.to(dtype=torch.bool)
    targets = torch.zeros_like(score)
    top_fraction = max(0.0, min(1.0, float(cfg.selector_signal_top_fraction)))
    min_targets = max(1, int(cfg.selector_signal_min_targets))
    for row in range(int(score.shape[0])):
        valid_idx = torch.nonzero(mask[row], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        valid_scores = score[row, valid_idx]
        positive_count = int((valid_scores > 0.0).sum().detach().cpu())
        if positive_count <= 0:
            continue
        target_count = min(
            int(valid_idx.numel()),
            positive_count,
            max(min_targets, int(math.ceil(float(valid_idx.numel()) * top_fraction))),
        )
        if target_count <= 0:
            continue
        selected = torch.topk(valid_scores, k=target_count, largest=True, sorted=False).indices
        targets[row, valid_idx[selected]] = 1.0
    return targets, mask


def _feature_relevance_targets(
    batch: TabenticsDiakrinoFSClassifierBatch,
    outputs: TabenticsDiakrinoFSClassifierOutputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = outputs.selector_outputs.logits
    if batch.feature_relevance_targets is None:
        return torch.zeros_like(logits), torch.zeros_like(outputs.feature_valid_mask, dtype=torch.bool)
    targets = batch.feature_relevance_targets.to(device=logits.device, dtype=logits.dtype)
    if targets.shape != logits.shape:
        if targets.shape[1] > logits.shape[1]:
            targets = targets[:, : logits.shape[1]]
        else:
            targets = F.pad(targets, (0, logits.shape[1] - targets.shape[1]))
    finite = torch.isfinite(targets)
    targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
    mask = outputs.feature_valid_mask.to(dtype=torch.bool) & finite
    targets = torch.where(mask, targets, torch.zeros_like(targets))
    return targets, mask


def fs_classifier_loss(
    outputs: TabenticsDiakrinoFSClassifierOutputs,
    batch: TabenticsDiakrinoFSClassifierBatch,
    *,
    config: TabenticsDiakrinoFSClassifierConfig | None = None,
) -> tuple[torch.Tensor, JsonDict]:
    """Cross-entropy loss for the single-query FS classifier."""

    _ensure_torch()
    cfg = config or TabenticsDiakrinoFSClassifierConfig()
    logits = outputs.class_logits
    labels = batch.query_labels
    total = logits.new_tensor(0.0)
    classification = logits.new_tensor(0.0)
    evidence_auxiliary = logits.new_tensor(0.0)
    selector_signal = logits.new_tensor(0.0)
    selector_signal_listwise = logits.new_tensor(0.0)
    selector_relevance = logits.new_tensor(0.0)
    selector_relevance_listwise = logits.new_tensor(0.0)
    gate_cardinality = logits.new_tensor(0.0)
    gate_entropy = logits.new_tensor(0.0)
    accuracy = logits.new_tensor(0.0)
    valid_count = logits.new_tensor(0.0)
    signal_target_rate = logits.new_tensor(0.0)
    relevance_target_mean = logits.new_tensor(0.0)
    strict_target_rate = logits.new_tensor(0.0)
    class_count = int(logits.shape[-1])
    label_valid = None
    if labels is not None:
        labels = labels.to(device=logits.device, dtype=torch.long)
        label_valid = (labels >= 0) & (labels < class_count)
        valid_count = label_valid.to(dtype=logits.dtype).sum()
    if labels is not None and label_valid is not None and float(cfg.classification_weight) > 0.0:
        if bool(torch.any(label_valid).detach().cpu()):
            classification = F.cross_entropy(
                logits[label_valid],
                labels[label_valid],
                label_smoothing=float(cfg.label_smoothing),
            )
            total = total + float(cfg.classification_weight) * classification
            predictions = logits[label_valid].argmax(dim=-1)
            accuracy = (predictions == labels[label_valid]).to(dtype=logits.dtype).mean()
    if labels is not None and label_valid is not None and float(cfg.evidence_auxiliary_weight) > 0.0:
        feature_class_valid = outputs.feature_valid_mask.unsqueeze(-1) & outputs.class_valid.unsqueeze(1)
        evidence = torch.where(feature_class_valid, outputs.feature_class_evidence, torch.zeros_like(outputs.feature_class_evidence))
        gates = torch.where(feature_class_valid, outputs.feature_class_gates, torch.zeros_like(outputs.feature_class_gates))
        if bool(cfg.evidence_auxiliary_detach_gates):
            gates = gates.detach()
        gate_mass = gates.sum(dim=1).clamp(min=1.0)
        evidence_logits = (evidence * gates).sum(dim=1) / torch.sqrt(gate_mass)
        evidence_logits = torch.where(outputs.class_valid, evidence_logits, torch.full_like(evidence_logits, -30.0))
        if bool(torch.any(label_valid).detach().cpu()):
            evidence_auxiliary = F.cross_entropy(
                evidence_logits[label_valid],
                labels[label_valid],
                label_smoothing=float(cfg.label_smoothing),
            )
            total = total + float(cfg.evidence_auxiliary_weight) * evidence_auxiliary
    if float(cfg.selector_signal_weight) > 0.0 or float(cfg.selector_signal_listwise_weight) > 0.0:
        signal_targets, signal_mask = _classifier_signal_targets(batch, outputs, cfg)
        signal_target_rate = (
            signal_targets[signal_mask].mean()
            if bool(torch.any(signal_mask).detach().cpu())
            else logits.new_tensor(0.0)
        )
        if bool(torch.any(signal_targets > 0.0).detach().cpu()):
            selector_logits = outputs.selector_outputs.logits
            if float(cfg.selector_signal_weight) > 0.0:
                flat_logits = selector_logits[signal_mask]
                flat_targets = signal_targets[signal_mask].to(dtype=flat_logits.dtype)
                positives = flat_targets.sum().clamp(min=1.0)
                negatives = (flat_targets.numel() - flat_targets.sum()).clamp(min=1.0)
                pos_weight = (negatives / positives).clamp(min=1.0, max=32.0)
                selector_signal = F.binary_cross_entropy_with_logits(
                    flat_logits,
                    flat_targets,
                    pos_weight=pos_weight,
                )
                total = total + float(cfg.selector_signal_weight) * selector_signal
            if float(cfg.selector_signal_listwise_weight) > 0.0:
                masked_logits = torch.where(signal_mask, selector_logits, torch.full_like(selector_logits, -30.0))
                target_mass = signal_targets.sum(dim=1, keepdim=True)
                row_valid = target_mass.squeeze(-1) > 0.0
                if bool(torch.any(row_valid).detach().cpu()):
                    target_dist = signal_targets / target_mass.clamp(min=float(cfg.eps))
                    log_probs = F.log_softmax(masked_logits, dim=1)
                    selector_signal_listwise = -(target_dist[row_valid] * log_probs[row_valid]).sum(dim=1).mean()
                    total = total + float(cfg.selector_signal_listwise_weight) * selector_signal_listwise
    if float(cfg.selector_relevance_weight) > 0.0 or float(cfg.selector_relevance_listwise_weight) > 0.0:
        relevance_targets, relevance_mask = _feature_relevance_targets(batch, outputs)
        relevance_target_mean = (
            relevance_targets[relevance_mask].mean()
            if bool(torch.any(relevance_mask).detach().cpu())
            else logits.new_tensor(0.0)
        )
        if batch.strict_feature_targets is not None:
            strict_targets = batch.strict_feature_targets.to(device=logits.device, dtype=logits.dtype)
            if strict_targets.shape != relevance_targets.shape:
                if strict_targets.shape[1] > relevance_targets.shape[1]:
                    strict_targets = strict_targets[:, : relevance_targets.shape[1]]
                else:
                    strict_targets = F.pad(strict_targets, (0, relevance_targets.shape[1] - strict_targets.shape[1]))
            strict_targets = torch.nan_to_num(strict_targets, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
            strict_target_rate = (
                strict_targets[relevance_mask].mean()
                if bool(torch.any(relevance_mask).detach().cpu())
                else logits.new_tensor(0.0)
            )
        if bool(torch.any((relevance_targets > 0.0) & relevance_mask).detach().cpu()):
            selector_logits = outputs.selector_outputs.logits
            if float(cfg.selector_relevance_weight) > 0.0:
                flat_logits = selector_logits[relevance_mask]
                flat_targets = relevance_targets[relevance_mask].to(dtype=flat_logits.dtype)
                positives = flat_targets.sum().clamp(min=1.0)
                negatives = (flat_targets.numel() - flat_targets.sum()).clamp(min=1.0)
                pos_weight = (negatives / positives).clamp(min=1.0, max=32.0)
                weights = 1.0 + (pos_weight - 1.0) * flat_targets
                per_feature = F.binary_cross_entropy_with_logits(flat_logits, flat_targets, reduction="none")
                selector_relevance = (per_feature * weights).sum() / weights.sum().clamp(min=1.0)
                total = total + float(cfg.selector_relevance_weight) * selector_relevance
            if float(cfg.selector_relevance_listwise_weight) > 0.0:
                masked_logits = torch.where(relevance_mask, selector_logits, torch.full_like(selector_logits, -30.0))
                target_mass = relevance_targets.sum(dim=1, keepdim=True)
                row_valid = target_mass.squeeze(-1) > 0.0
                if bool(torch.any(row_valid).detach().cpu()):
                    target_dist = relevance_targets / target_mass.clamp(min=float(cfg.eps))
                    log_probs = F.log_softmax(masked_logits, dim=1)
                    selector_relevance_listwise = -(target_dist[row_valid] * log_probs[row_valid]).sum(dim=1).mean()
                    total = total + float(cfg.selector_relevance_listwise_weight) * selector_relevance_listwise
    gate_mask = outputs.feature_valid_mask.unsqueeze(-1) & outputs.class_valid.unsqueeze(1)
    if float(cfg.gate_cardinality_weight) > 0.0 and bool(torch.any(gate_mask).detach().cpu()):
        gate_valid = gate_mask.to(dtype=outputs.feature_class_gates.dtype)
        per_class_counts = gate_valid.sum(dim=1).clamp(min=1.0)
        gate_fraction = (outputs.feature_class_gates * gate_valid).sum(dim=1) / per_class_counts
        class_valid = outputs.class_valid.to(dtype=torch.bool)
        target = max(float(cfg.eps), float(cfg.gate_target_fraction))
        gate_cardinality = torch.square((gate_fraction[class_valid] - target) / target).mean()
        total = total + float(cfg.gate_cardinality_weight) * gate_cardinality
    if float(cfg.gate_entropy_weight) > 0.0 and bool(torch.any(gate_mask).detach().cpu()):
        p = outputs.feature_class_gates.clamp(min=float(cfg.eps), max=1.0 - float(cfg.eps))
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        gate_entropy = (entropy * gate_mask.to(dtype=entropy.dtype)).sum() / gate_mask.to(dtype=entropy.dtype).sum().clamp(min=1.0)
        total = total + float(cfg.gate_entropy_weight) * gate_entropy
    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=-1)
        confidence = probabilities.max(dim=-1).values.mean()
        gate_mean = (
            outputs.feature_class_gates[gate_mask].mean()
            if bool(torch.any(gate_mask).detach().cpu())
            else logits.new_tensor(0.0)
        )
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_classification": float(classification.detach().cpu()),
        "loss_evidence_auxiliary": float(evidence_auxiliary.detach().cpu()),
        "loss_selector_signal": float(selector_signal.detach().cpu()),
        "loss_selector_signal_listwise": float(selector_signal_listwise.detach().cpu()),
        "loss_selector_relevance": float(selector_relevance.detach().cpu()),
        "loss_selector_relevance_listwise": float(selector_relevance_listwise.detach().cpu()),
        "loss_gate_cardinality": float(gate_cardinality.detach().cpu()),
        "loss_gate_entropy": float(gate_entropy.detach().cpu()),
        "accuracy": float(accuracy.detach().cpu()),
        "valid_examples": float(valid_count.detach().cpu()),
        "mean_confidence": float(confidence.detach().cpu()),
        "feature_class_gate_mean": float(gate_mean.detach().cpu()),
        "signal_feature_target_rate": float(signal_target_rate.detach().cpu()),
        "relevance_feature_target_mean": float(relevance_target_mean.detach().cpu()),
        "strict_feature_target_rate": float(strict_target_rate.detach().cpu()),
    }


__all__ = [
    "TabenticsDiakrinoFSClassifier",
    "TabenticsDiakrinoFSClassifierBatch",
    "TabenticsDiakrinoFSClassifierConfig",
    "TabenticsDiakrinoFSClassifierOutputs",
    "compute_fs_classifier_feature_stats",
    "fs_classifier_loss",
]

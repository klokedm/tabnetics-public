"""Minimal axial DIAKRINO smoke implementation for the experimental world-backed corpus."""

from __future__ import annotations

import copy
import math
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:  # pragma: no cover - older torch builds
        SDPBackend = None  # type: ignore[assignment]
        sdpa_kernel = None  # type: ignore[assignment]
    # torch<2.4 (e.g. the MN5 ACC base module's torch 2.3.0) lacks nn.RMSNorm.
    # Install a faithful fallback before any model is constructed so existing
    # checkpoints load and run identically. No-op on torch>=2.4.
    try:
        from components.torch_compat import install_torch_compat

        install_torch_compat()
    except Exception:  # pragma: no cover - compat is best-effort
        pass
except Exception:  # pragma: no cover - guarded for non-torch environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]

try:
    from components.feature_id_rope import FeatureIDCoordinateGenerator, FeatureIDRoPE  # type: ignore
except ImportError:  # pragma: no cover
    FeatureIDCoordinateGenerator = None  # type: ignore[assignment,misc]
    FeatureIDRoPE = None  # type: ignore[assignment,misc]
try:
    from ..data import EpisodeBatch, build_episode_loader
except Exception:  # pragma: no cover - core inference path does not ship experimental corpus loaders
    EpisodeBatch = Any  # type: ignore[misc,assignment]

    def build_episode_loader(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError(
            "Tabnetics Diakrino synthetic episode training loaders are not packaged in "
            "tabnetics.classification; use the experimental training package for "
            "world-backed DIAKRINO training."
        )

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class TabenticsDiakrinoConfig:
    fourier_features_k: int = 16
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    ffn_expansion: int = 2
    dropout: float = 0.1
    max_classes: int = 25
    max_feature_tokens: int | None = None
    ce_weight: float = 1.0
    salience_weight: float = 0.5
    # v8c: weight for gate-side focal BCE on feature_gate_head (untied from salience_head)
    gate_weight: float = 0.25
    mae_weight: float = 0.0
    chaotic_weight: float = 0.10
    harmonic_fusion: bool = True
    enable_chaotic_head: bool = True
    jepa_weight: float = 0.25
    sigreg_weight: float = 0.05
    sigreg_projections: int = 128
    ema_decay: float = 0.996
    use_meta_features: bool = False
    meta_feature_dim: int = 16
    gradient_checkpointing: bool = False
    clip_value: float = 6.0
    eps: float = 1e-6
    iqr_to_std: float = 1.349
    # HRM (Hierarchical Reasoning Model) parameters
    hrm_inner_steps: int = 4
    hrm_outer_cycles: int = 2
    hrm_segments: int = 3
    # TRM-inspired: full backprop through all recursions in last HRM step
    hrm_full_backprop_last: bool = True
    # Random cycle depth: sample N ~ Uniform{1..hrm_outer_cycles} per step
    hrm_random_cycles: bool = False
    # Loss: use stable-max (Prieto et al. 2025) instead of softmax+CE
    use_stable_max: bool = True
    # Class masking: mask logits for inactive classes to -inf before loss
    class_masking: bool = True


@dataclass(frozen=True)
class TabenticsDiakrinoTrainerConfig:
    epochs: int = 3
    steps_per_epoch: int = 8
    batch_size: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    shuffle: bool = True
    loader_seed: int = 0
    prefetch_batches: int = 2
    validation_batches: int = 2
    step_log_interval: int = 4
    step_checkpoint_interval: int = 0
    warmup_steps: int = 50
    lr_min_fraction: float = 0.01
    pretrain_clean: bool = False
    # LR schedule: 'constant' (HRM/TRM) or 'cosine'
    lr_schedule: str = "constant"
    # TRM: higher LR multiplier for embedding layers
    embedding_lr_multiplier: float = 1.0
    # SIREN warmup mode (v8): freeze backbone, train only harmonic_branch + fusion
    siren_warmup: bool = False
    siren_warmup_gate_value: float = 0.25
    siren_warmup_gate_freeze_steps: int = 25
    siren_warmup_siren_lr: float = 3e-4
    # v8f: supervise salience/gate against the synthetic generation ground
    # truth (active_feature_targets from world spec) instead of the Fisher-
    # derived per-episode proxy. The Fisher signal is still computed and
    # fed as an INPUT feature via class-conditional stats, but the
    # supervision target becomes the true causal feature set.
    use_gt_salience_target: bool = False


@dataclass(frozen=True)
class TabenticsDiakrinoPreparedBatch:
    support: torch.Tensor
    support_mask: torch.Tensor
    support_valid: torch.Tensor
    support_labels: torch.Tensor
    query: torch.Tensor
    query_mask: torch.Tensor
    query_valid: torch.Tensor
    query_labels: torch.Tensor
    feature_valid_mask: torch.Tensor
    active_feature_targets: torch.Tensor
    per_episode_target: torch.Tensor
    class_counts: torch.Tensor
    mae_mask: torch.Tensor
    mae_reconstruction_target: torch.Tensor
    support_clean: torch.Tensor
    chaotic_targets: torch.Tensor
    meta_vectors: torch.Tensor


@dataclass(frozen=True)
class TabenticsDiakrinoOutputs:
    logits: torch.Tensor
    importance_logits: torch.Tensor
    # v8d Stage 2: pre-attention gate logits = salience_head(feature_bias) + gate_bias_offset (TIED)
    gate_logits: torch.Tensor
    mae_reconstruction: torch.Tensor
    chaotic_logit: torch.Tensor
    latent_prediction: torch.Tensor
    latent_target: torch.Tensor
    sigreg_samples: torch.Tensor
    harmonic_gate_value: float = 0.0


@dataclass(frozen=True)
class TabenticsDiakrinoLossReport:
    total: float
    components: JsonDict

    def to_dict(self) -> JsonDict:
        return {
            "total": float(self.total),
            "components": dict(self.components),
        }


def _class_count_for_example(example: Any) -> int:
    head = next(item for item in example.world_spec.task_heads if item.head_id == example.head_id)
    return max(1, int(head.class_count))


def _nan_quantile(values: torch.Tensor, q: float, dim: int) -> torch.Tensor:
    if hasattr(torch, "nanquantile"):
        return torch.nanquantile(values, q=q, dim=dim)
    return torch.quantile(torch.nan_to_num(values, nan=0.0), q=q, dim=dim)


def _fit_scaler(
    values: torch.Tensor,
    *,
    missing_mask: torch.Tensor,
    valid_rows: torch.Tensor,
    config: TabenticsDiakrinoConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_mask = (~valid_rows).unsqueeze(-1).expand_as(values)
    masked = values.masked_fill(missing_mask | row_mask, float("nan"))
    center = torch.nanmedian(masked, dim=1).values
    q25 = _nan_quantile(masked, q=0.25, dim=1)
    q75 = _nan_quantile(masked, q=0.75, dim=1)
    scale = torch.clamp((q75 - q25) / max(config.iqr_to_std, config.eps), min=config.eps)
    center = torch.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    return center, scale


def _apply_scaler(
    values: torch.Tensor,
    *,
    center: torch.Tensor,
    scale: torch.Tensor,
    missing_mask: torch.Tensor,
    valid_rows: torch.Tensor,
    config: TabenticsDiakrinoConfig,
) -> torch.Tensor:
    scaled = (values - center.unsqueeze(1)) / scale.unsqueeze(1)
    scaled = torch.clamp(scaled, min=-config.clip_value, max=config.clip_value)
    row_mask = (~valid_rows).unsqueeze(-1)
    scaled = torch.where(missing_mask | row_mask, torch.zeros_like(scaled), scaled)
    return torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def _select_feature_indices(
    values: torch.Tensor,
    *,
    missing_mask: torch.Tensor,
    valid_rows: torch.Tensor,
    feature_valid_mask: torch.Tensor,
    budget: int,
    required_per_batch: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select up to `budget` feature columns per batch item by variance×coverage score.

    When `required_per_batch` is given (list of active feature index tensors, one per batch
    item), those indices are guaranteed to be included first as long as their count does not
    exceed the budget.  Remaining slots are filled by the highest-scoring non-required
    features.  If required count > budget (e.g. head_poly with 256 active at budget=128),
    falls back to pure variance-based selection.
    """
    batch_size, _, feature_count = values.shape
    selected_indices = torch.zeros((batch_size, budget), dtype=torch.long, device=values.device)
    selected_valid_mask = torch.zeros((batch_size, budget), dtype=torch.bool, device=values.device)
    for batch_index in range(batch_size):
        available = torch.nonzero(feature_valid_mask[batch_index], as_tuple=False).squeeze(-1)
        if available.numel() == 0:
            continue
        chosen = available
        if available.numel() > budget:
            feature_values = values[batch_index, :, available]
            observed = (
                valid_rows[batch_index].unsqueeze(-1)
                & (~missing_mask[batch_index, :, available])
            ).to(dtype=feature_values.dtype)
            counts = observed.sum(dim=0)
            mean = (feature_values * observed).sum(dim=0) / counts.clamp(min=1.0)
            centered = (feature_values - mean.unsqueeze(0)) * observed
            variance = (centered * centered).sum(dim=0) / counts.clamp(min=1.0)
            coverage = counts / counts.clamp(min=1.0).max()
            score = variance * coverage
            # Tiny tie-breaker so index ordering is stable when scores are equal.
            score = score + torch.linspace(0.0, 1e-6, steps=available.numel(), device=values.device)

            # Priority-include required (active) features when they fit in budget.
            req = required_per_batch[batch_index] if required_per_batch is not None else None
            if req is not None and req.numel() > 0:
                req_valid = req[(req >= 0) & (req < feature_count)]
                req_mask = torch.isin(available, req_valid)       # positions in `available`
                n_req = int(req_mask.sum().item())
                if n_req <= budget:
                    # Guarantee all required; fill remaining slots by score.
                    req_positions = req_mask.nonzero(as_tuple=False).squeeze(-1)
                    free_positions = (~req_mask).nonzero(as_tuple=False).squeeze(-1)
                    n_fill = budget - n_req
                    if n_fill > 0 and free_positions.numel() > 0:
                        fill_k = min(n_fill, int(free_positions.numel()))
                        free_scores = score[free_positions]
                        topk_fill = torch.topk(free_scores, k=fill_k, largest=True).indices
                        fill_chosen = available[free_positions[topk_fill]]
                    else:
                        fill_chosen = available.new_empty(0)
                    chosen = torch.cat([available[req_positions], fill_chosen]).sort().values
                else:
                    # More required features than budget (e.g. head_poly at budget=128).
                    # Fall back to variance-based selection; can't fit all active features.
                    topk = torch.topk(score, k=budget, largest=True).indices
                    chosen = available[topk].sort().values
            else:
                topk = torch.topk(score, k=budget, largest=True).indices
                chosen = available[topk].sort().values
        count = min(int(chosen.numel()), budget)
        selected_indices[batch_index, :count] = chosen[:count]
        selected_valid_mask[batch_index, :count] = True
    return selected_indices, selected_valid_mask


def _compute_feature_stats(
    support: torch.Tensor,
    *,
    support_mask: torch.Tensor,
    support_valid: torch.Tensor,
) -> torch.Tensor:
    """Per-feature statistics: mean, std, skew, kurtosis, observed fraction. Returns (B, d, 5)."""
    row_mask = (~support_valid).unsqueeze(-1).expand_as(support)
    observed = (~(support_mask | row_mask)).to(dtype=support.dtype)
    count = torch.clamp(observed.sum(dim=1), min=1.0)
    mean = (support * observed).sum(dim=1) / count
    centered = (support - mean.unsqueeze(1)) * observed
    var = (centered * centered).sum(dim=1) / count
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    standardized = centered / std.unsqueeze(1)
    skew = (standardized.pow(3) * observed).sum(dim=1) / count
    kurt = (standardized.pow(4) * observed).sum(dim=1) / count
    observed_fraction = count / max(1, int(support.shape[1]))
    stats = torch.stack([mean, std, skew, kurt, observed_fraction], dim=-1)
    return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_fisher_signal(
    support: torch.Tensor,
    *,
    support_mask: torch.Tensor,
    support_valid: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes_max: int,
) -> torch.Tensor:
    """Per-feature log1p(between-class variance / within-class variance).

    Returns [B, F] tensor measuring how class-discriminative each support
    column is in this episode (i.e. how useful it is for predicting y_support).
    Used both as a class-conditional stat for the gate's input encoder and
    as the source of per-episode usefulness targets.
    """
    B, S, feat = support.shape
    K = max(2, int(num_classes_max))
    eps = 1e-6
    dtype = support.dtype
    row_valid = support_valid.unsqueeze(-1).expand(-1, -1, feat)
    value_valid = (~support_mask).to(dtype=torch.bool)
    cell_valid = (row_valid & value_valid).to(dtype=dtype)
    valid_label = (support_labels >= 0).to(dtype=dtype)
    labels_clamped = support_labels.clamp(min=0, max=K - 1)
    class_onehot = F.one_hot(labels_clamped, num_classes=K).to(dtype=dtype)
    class_onehot = class_onehot * valid_label.unsqueeze(-1)
    n_kf = torch.einsum("bsk,bsf->bkf", class_onehot, cell_valid)
    sum_kf = torch.einsum("bsk,bsf->bkf", class_onehot, support * cell_valid)
    sum_x2_kf = torch.einsum("bsk,bsf->bkf", class_onehot, (support * support) * cell_valid)
    n_safe = n_kf.clamp(min=1.0)
    mean_kf = sum_kf / n_safe
    var_kf = (sum_x2_kf / n_safe) - (mean_kf * mean_kf)
    var_kf = var_kf.clamp(min=0.0)
    # Singleton-class robustness: a class with n_k<2 has degenerate variance
    # (always 0), which inflates between/within ratios for noise columns when
    # the support contains rare classes. Restrict aggregation to classes with
    # n_k>=2, and require >=2 such usable classes per feature for a meaningful
    # Fisher signal — otherwise emit 0 for that feature.
    usable = (n_kf >= 2.0).to(dtype=dtype)  # [B, K, F]
    used_n_f = (n_kf * usable).sum(dim=1)  # [B, F]
    used_n_safe = used_n_f.clamp(min=1.0)
    within_var_f = (n_kf * var_kf * usable).sum(dim=1) / used_n_safe
    global_mean_f = (n_kf * usable * mean_kf).sum(dim=1) / used_n_safe
    diff = mean_kf - global_mean_f.unsqueeze(1)
    between_var_f = (n_kf * usable * diff * diff).sum(dim=1) / used_n_safe
    fisher = between_var_f / (within_var_f + eps)
    fisher_log = torch.log1p(fisher.clamp(min=0.0))
    n_usable_classes_f = usable.sum(dim=1)  # [B, F]
    fisher_log = torch.where(
        n_usable_classes_f >= 2.0, fisher_log, torch.zeros_like(fisher_log)
    )
    return torch.nan_to_num(fisher_log, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_class_conditional_stats(
    support: torch.Tensor,
    *,
    support_mask: torch.Tensor,
    support_valid: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes_max: int,
) -> torch.Tensor:
    """Per-feature class-conditional summary stats. Returns [B, F, 5].

    Stats per feature:
      0. fisher_log: log1p(between-class variance / within-class variance)
      1. max_class_shift: max_k |mean_k - global_mean| / global_std (sigma units)
      2. mean_class_shift: mean_k |mean_k - global_mean| / global_std
      3. log_std_ratio: log1p((max_k std_k - min_k std_k) / min_k std_k)
      4. class_balance_entropy: H(class_counts) / log(K_active)

    These let the feature gate see WHETHER each column has class-discriminative
    signal in this episode (uses y_support), not just marginal statistics.
    """
    B, S, feat = support.shape
    K = max(2, int(num_classes_max))
    eps = 1e-6
    dtype = support.dtype
    row_valid = support_valid.unsqueeze(-1).expand(-1, -1, feat)
    value_valid = (~support_mask).to(dtype=torch.bool)
    cell_valid = (row_valid & value_valid).to(dtype=dtype)
    valid_label = (support_labels >= 0).to(dtype=dtype)
    labels_clamped = support_labels.clamp(min=0, max=K - 1)
    class_onehot = F.one_hot(labels_clamped, num_classes=K).to(dtype=dtype)
    class_onehot = class_onehot * valid_label.unsqueeze(-1)
    n_kf = torch.einsum("bsk,bsf->bkf", class_onehot, cell_valid)
    sum_kf = torch.einsum("bsk,bsf->bkf", class_onehot, support * cell_valid)
    sum_x2_kf = torch.einsum("bsk,bsf->bkf", class_onehot, (support * support) * cell_valid)
    n_safe = n_kf.clamp(min=1.0)
    mean_kf = sum_kf / n_safe
    var_kf = (sum_x2_kf / n_safe) - (mean_kf * mean_kf)
    var_kf = var_kf.clamp(min=0.0)
    std_kf = torch.sqrt(var_kf + eps)
    present_kf = (n_kf > 0).to(dtype=dtype)
    n_classes_per_f = present_kf.sum(dim=1).clamp(min=1.0)
    total_n_f = n_kf.sum(dim=1).clamp(min=1.0)
    global_mean_f = (n_kf * mean_kf).sum(dim=1) / total_n_f
    within_var_f = (n_kf * var_kf).sum(dim=1) / total_n_f
    diff = mean_kf - global_mean_f.unsqueeze(1)
    between_var_f = (n_kf * diff * diff).sum(dim=1) / total_n_f
    global_std_f = torch.sqrt(within_var_f + between_var_f + eps)
    fisher = between_var_f / (within_var_f + eps)
    fisher_log = torch.log1p(fisher.clamp(min=0.0))
    abs_shift = diff.abs() * present_kf
    max_class_shift = abs_shift.amax(dim=1) / (global_std_f + eps)
    mean_class_shift = abs_shift.sum(dim=1) / (n_classes_per_f * (global_std_f + eps))
    LARGE = 1e6
    std_for_max = std_kf * present_kf
    max_std = std_for_max.amax(dim=1)
    std_for_min = std_kf * present_kf + (1.0 - present_kf) * LARGE
    min_std = std_for_min.amin(dim=1)
    log_std_ratio = torch.log1p((max_std - min_std).clamp(min=0.0) / (min_std + eps))
    p_kf = n_kf / (total_n_f.unsqueeze(1) + eps)
    p_log = torch.where(p_kf > 0, p_kf * torch.log(p_kf + eps), torch.zeros_like(p_kf))
    entropy_f = -p_log.sum(dim=1)
    log_k = torch.log(n_classes_per_f + eps)
    class_balance = entropy_f / (log_k + eps)
    stats = torch.stack(
        [fisher_log, max_class_shift, mean_class_shift, log_std_ratio, class_balance],
        dim=-1,
    )
    return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)


def _flatten_valid_feature_latents(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected values with shape (batch, features, dim); got {tuple(values.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask with shape (batch, features); got {tuple(mask.shape)}")
    flat = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    if torch.any(flat_mask):
        return flat[flat_mask]
    return flat.new_zeros((0, values.shape[-1]))


def _validated_support_auxiliary_target(
    values: Any,
    *,
    name: str,
    batch_index: int,
    expected_rows: int,
    expected_features: int,
    selected_indices: torch.Tensor | None,
    feature_valid_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    target = torch.as_tensor(values, dtype=torch.float32, device=device)
    expected_shape = (int(expected_rows), int(expected_features))
    if target.ndim != 2:
        raise ValueError(
            f"{name} for batch item {int(batch_index)} must be 2D with shape "
            f"{expected_shape}; got {tuple(target.shape)}"
        )
    if tuple(target.shape) != expected_shape:
        raise ValueError(
            f"{name} for batch item {int(batch_index)} must have shape "
            f"{expected_shape}; got {tuple(target.shape)}"
        )
    if selected_indices is None:
        return target

    selected = selected_indices[int(batch_index)]
    valid_slots = feature_valid_mask[int(batch_index)]
    if torch.any((selected[valid_slots] < 0) | (selected[valid_slots] >= int(expected_features))):
        raise ValueError(
            f"{name} selected feature indices for batch item {int(batch_index)} "
            f"are outside the supplied target width {int(expected_features)}"
        )
    safe_selected = selected.clamp(min=0, max=max(0, int(expected_features) - 1))
    target = torch.index_select(target, 1, safe_selected)
    return torch.where(valid_slots.unsqueeze(0), target, torch.zeros_like(target))


def _validated_label_tensor(
    values: Any,
    *,
    name: str,
    expected_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    try:
        labels = torch.as_tensor(values, device=device)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be numeric integer labels with shape {expected_shape}"
        ) from exc
    if labels.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape {expected_shape}; got {tuple(labels.shape)}")
    if tuple(labels.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {tuple(labels.shape)}")
    if torch.is_complex(labels):
        raise ValueError(f"{name} must be integer-encoded labels, not complex values")
    if torch.is_floating_point(labels):
        if not bool(torch.isfinite(labels).all().detach().cpu()):
            raise ValueError(f"{name} must contain finite integer-encoded labels")
        if not bool((labels == labels.round()).all().detach().cpu()):
            raise ValueError(f"{name} must be integer-encoded labels")
    return labels.to(dtype=torch.long, device=device)


def prepare_tabentics_diakrino_batch(
    batch: EpisodeBatch,
    *,
    device: str | torch.device = "cpu",
    config: TabenticsDiakrinoConfig | None = None,
    permute_columns: bool = True,
) -> TabenticsDiakrinoPreparedBatch:
    if torch is None:
        raise ImportError("prepare_tabentics_diakrino_batch requires torch to be installed.")

    cfg = config or TabenticsDiakrinoConfig()
    resolved_device = torch.device(device)

    support = torch.as_tensor(batch.support, dtype=torch.float32, device=resolved_device)
    query = torch.as_tensor(batch.query, dtype=torch.float32, device=resolved_device)
    support_clean_raw = torch.as_tensor(batch.support_clean, dtype=torch.float32, device=resolved_device)
    support_mask = torch.as_tensor(batch.support_missing_mask, dtype=torch.bool, device=resolved_device)
    query_mask = torch.as_tensor(batch.query_missing_mask, dtype=torch.bool, device=resolved_device)
    support_labels = _validated_label_tensor(
        batch.support_labels,
        name="support_labels",
        expected_shape=(int(support.shape[0]), int(support.shape[1])),
        device=resolved_device,
    )
    query_labels = _validated_label_tensor(
        batch.query_labels,
        name="query_labels",
        expected_shape=(int(query.shape[0]), int(query.shape[1])),
        device=resolved_device,
    )

    support_valid = support_labels >= 0
    query_valid = query_labels >= 0

    center, scale = _fit_scaler(
        support,
        missing_mask=support_mask | torch.isnan(support),
        valid_rows=support_valid,
        config=cfg,
    )
    support_scaled = _apply_scaler(
        support,
        center=center,
        scale=scale,
        missing_mask=support_mask | torch.isnan(support),
        valid_rows=support_valid,
        config=cfg,
    )
    query_scaled = _apply_scaler(
        query,
        center=center,
        scale=scale,
        missing_mask=query_mask | torch.isnan(query),
        valid_rows=query_valid,
        config=cfg,
    )

    batch_size = len(batch.examples)
    original_feature_count = int(support.shape[-1])
    feature_valid_mask = torch.zeros((batch_size, original_feature_count), dtype=torch.bool, device=resolved_device)
    class_counts = torch.zeros(batch_size, dtype=torch.long, device=resolved_device)
    for batch_index, example in enumerate(batch.examples):
        width = int(example.episode.support.shape[1])
        feature_valid_mask[batch_index, :width] = True
        class_counts[batch_index] = _class_count_for_example(example)

    selected_indices: torch.Tensor | None = None
    feature_budget = cfg.max_feature_tokens
    if feature_budget is not None:
        feature_budget = max(1, min(int(feature_budget), original_feature_count))
        if feature_budget < original_feature_count:
            selected_indices, feature_valid_mask = _select_feature_indices(
                support_scaled,
                missing_mask=support_mask,
                valid_rows=support_valid,
                feature_valid_mask=feature_valid_mask,
                budget=feature_budget,
            )
            gather_index = selected_indices.unsqueeze(1).expand(-1, support_scaled.shape[1], -1)
            support_scaled = torch.gather(support_scaled, 2, gather_index)
            support_mask = torch.gather(support_mask, 2, gather_index)
            query_gather_index = selected_indices.unsqueeze(1).expand(-1, query_scaled.shape[1], -1)
            query_scaled = torch.gather(query_scaled, 2, query_gather_index)
            query_mask = torch.gather(query_mask, 2, query_gather_index)
            invalid_slots = ~feature_valid_mask.unsqueeze(1)
            support_scaled = torch.where(invalid_slots, torch.zeros_like(support_scaled), support_scaled)
            support_mask = torch.where(invalid_slots, torch.ones_like(support_mask), support_mask)
            query_scaled = torch.where(invalid_slots, torch.zeros_like(query_scaled), query_scaled)
            query_mask = torch.where(invalid_slots, torch.ones_like(query_mask), query_mask)

    feature_count = int(support_scaled.shape[-1])
    if selected_indices is None:
        selected_indices = torch.arange(feature_count, device=resolved_device).unsqueeze(0).expand(batch_size, -1)

    active_feature_targets = torch.zeros((batch_size, feature_count), dtype=torch.float32, device=resolved_device)
    for batch_index, example in enumerate(batch.examples):
        head = next(item for item in example.world_spec.task_heads if item.head_id == example.head_id)
        active = torch.as_tensor(head.active_features, dtype=torch.long, device=resolved_device)
        active = active[(active >= 0) & (active < original_feature_count)]
        if active.numel() == 0:
            continue
        selected = selected_indices[batch_index]
        match_mask = torch.isin(selected, active) & feature_valid_mask[batch_index]
        if torch.any(match_mask):
            active_feature_targets[batch_index, match_mask] = 1.0

    # v8d: per-episode usefulness target via Fisher discriminant on (X_support, y_support).
    # A feature is "useful in this episode" iff its support distribution is class-discriminative.
    # This honors the corruption transform (duplicates/transformations of active features pass
    # the threshold and are correctly labeled positive; injected noise columns or active features
    # whose signal was destroyed by corruption are correctly labeled negative).
    _fisher_log = _compute_fisher_signal(
        support_scaled,
        support_mask=support_mask,
        support_valid=support_valid,
        support_labels=support_labels,
        num_classes_max=int(cfg.max_classes),
    )
    # Threshold 0.05 corresponds to between/within variance >= ~5%; non-trivial signal.
    per_episode_target = (_fisher_log > 0.05).to(dtype=torch.float32) * feature_valid_mask.to(dtype=torch.float32)

    # Extract MAE mask, reconstruction target, clean values, and chaotic targets
    support_rows_count = int(support.shape[1])
    mae_mask = torch.zeros((batch_size, support_rows_count, feature_count), dtype=torch.float32, device=resolved_device)
    mae_reconstruction_target = torch.zeros((batch_size, support_rows_count, feature_count), dtype=torch.float32, device=resolved_device)
    support_clean = _apply_scaler(
        support_clean_raw,
        center=center,
        scale=scale,
        missing_mask=torch.isnan(support_clean_raw),
        valid_rows=support_valid,
        config=cfg,
    )
    if selected_indices is not None:
        gather_index = selected_indices.unsqueeze(1).expand(-1, support_clean.shape[1], -1)
        support_clean = torch.gather(support_clean, 2, gather_index)
        invalid_slots = ~feature_valid_mask.unsqueeze(1)
        support_clean = torch.where(invalid_slots, torch.zeros_like(support_clean), support_clean)
    chaotic_targets = torch.zeros(batch_size, dtype=torch.float32, device=resolved_device)
    for batch_index, example in enumerate(batch.examples):
        expected_rows = int(example.episode.support.shape[0])
        expected_features = int(example.episode.support.shape[1])
        _mae = example.episode.support_auxiliary_targets.get("mae_mask")
        if _mae is not None:
            _t = _validated_support_auxiliary_target(
                _mae,
                name="mae_mask",
                batch_index=batch_index,
                expected_rows=expected_rows,
                expected_features=expected_features,
                selected_indices=selected_indices,
                feature_valid_mask=feature_valid_mask,
                device=resolved_device,
            )
            mae_mask[batch_index, :expected_rows, : int(_t.shape[1])] = _t
        _target = example.episode.support_auxiliary_targets.get("mae_reconstruction_target")
        if _target is not None:
            _t2 = _validated_support_auxiliary_target(
                _target,
                name="mae_reconstruction_target",
                batch_index=batch_index,
                expected_rows=expected_rows,
                expected_features=expected_features,
                selected_indices=selected_indices,
                feature_valid_mask=feature_valid_mask,
                device=resolved_device,
            )
            mae_reconstruction_target[batch_index, :expected_rows, : int(_t2.shape[1])] = _t2
        # Chaotic target from task head metadata
        head = next(item for item in example.world_spec.task_heads if item.head_id == example.head_id)
        composite = {str(tag) for tag in (head.metadata.get("composite_family_tags") or [])}
        chaotic_targets[batch_index] = 1.0 if str(head.family_tag) in {"chaotic", "periodic"} or bool({"chaotic", "periodic"} & composite) else 0.0

    # Random column permutation for permutation-invariance training
    if permute_columns:
        perm = torch.randperm(feature_count, device=resolved_device)
        support_scaled = support_scaled[:, :, perm]
        support_mask = support_mask[:, :, perm]
        query_scaled = query_scaled[:, :, perm]
        query_mask = query_mask[:, :, perm]
        feature_valid_mask = feature_valid_mask[:, perm]
        active_feature_targets = active_feature_targets[:, perm]
        per_episode_target = per_episode_target[:, perm]
        mae_mask = mae_mask[:, :, perm]
        mae_reconstruction_target = mae_reconstruction_target[:, :, perm]
        support_clean = support_clean[:, :, perm]

    # Extract meta-feature vectors
    meta_vectors = torch.as_tensor(batch.support_meta_vectors, dtype=torch.float32, device=resolved_device)

    return TabenticsDiakrinoPreparedBatch(
        support=support_scaled,
        support_mask=support_mask,
        support_valid=support_valid,
        support_labels=support_labels,
        query=query_scaled,
        query_mask=query_mask,
        query_valid=query_valid,
        query_labels=query_labels,
        feature_valid_mask=feature_valid_mask,
        active_feature_targets=active_feature_targets,
        per_episode_target=per_episode_target,
        class_counts=class_counts,
        mae_mask=mae_mask,
        mae_reconstruction_target=mae_reconstruction_target,
        support_clean=support_clean,
        chaotic_targets=chaotic_targets,
        meta_vectors=meta_vectors,
    )


if nn is not None:

    class _SDPASelfAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int, *, dropout: float) -> None:
            super().__init__()
            self.d_model = int(d_model)
            self.n_heads = int(n_heads)
            self.head_dim = self.d_model // self.n_heads
            self.dropout = float(dropout)
            self.q_proj = nn.Linear(self.d_model, self.d_model)
            self.k_proj = nn.Linear(self.d_model, self.d_model)
            self.v_proj = nn.Linear(self.d_model, self.d_model)
            self.out_proj = nn.Linear(self.d_model, self.d_model)

        def _backend_context(self, *, values: torch.Tensor):
            if values.is_cuda and sdpa_kernel is not None and SDPBackend is not None:
                return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
            return nullcontext()

        def _amp_context(self, *, values: torch.Tensor):
            if values.is_cuda:
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            return nullcontext()

        def forward(
            self,
            values: torch.Tensor,
            *,
            key_padding_mask: torch.Tensor | None = None,
            causal_mask: torch.Tensor | None = None,
            rope: "FeatureIDRoPE | None" = None,
            feature_coordinates: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch_size, sequence_length, _ = values.shape
            with self._amp_context(values=values):
                q = self.q_proj(values).reshape(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(values).reshape(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(values).reshape(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

                if rope is not None and feature_coordinates is not None:
                    q, k = rope(q, k, feature_coordinates=feature_coordinates)

                attn_mask = None
                if causal_mask is not None:
                    attn_mask = causal_mask.to(dtype=q.dtype, device=q.device)
                if key_padding_mask is not None and torch.any(key_padding_mask):
                    padding_mask = torch.zeros(
                        (batch_size, 1, 1, sequence_length),
                        dtype=q.dtype,
                        device=values.device,
                    )
                    padding_mask = padding_mask.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
                    attn_mask = (attn_mask + padding_mask) if attn_mask is not None else padding_mask

                dropout_p = self.dropout if self.training else 0.0
                with self._backend_context(values=values):
                    attended = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                    )
                attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)
                attended = self.out_proj(attended)
            return attended.to(dtype=values.dtype)

    class _FourierTokenizer(nn.Module):
        """Multi-scale Fourier feature tokenizer.

        Initializes K learnable log-frequencies spanning ~0.1 to ~100 (log-spaced),
        giving the model access to both low-frequency trends and fine-grained
        periodic structure.  A learnable per-frequency phase offset adds expressiveness.
        """

        def __init__(self, config: TabenticsDiakrinoConfig) -> None:
            super().__init__()
            self.config = config
            k = max(1, int(config.fourier_features_k))
            # Multi-scale initialization: log-spaced from ~0.1 to ~100
            import math
            base = torch.linspace(math.log(0.1), math.log(100.0), steps=k)
            self.log_frequencies = nn.Parameter(base)
            self.phase_offsets = nn.Parameter(torch.zeros(k))
            # Input: sin(K) + cos(K) + missing_indicator(1) = 2K+1
            self.proj = nn.Linear((2 * k) + 1, int(config.d_model))

        def forward(self, values: torch.Tensor, *, missing_mask: torch.Tensor) -> torch.Tensor:
            freqs = torch.exp(self.log_frequencies).view(1, 1, 1, -1)
            phases = self.phase_offsets.view(1, 1, 1, -1)
            angles = values.unsqueeze(-1) * freqs + phases
            features = torch.cat(
                [
                    torch.sin(angles),
                    torch.cos(angles),
                    missing_mask.to(dtype=values.dtype).unsqueeze(-1),
                ],
                dim=-1,
            )
            return self.proj(features)


    class _FeatureStatsEncoder(nn.Module):
        def __init__(self, config: TabenticsDiakrinoConfig) -> None:
            super().__init__()
            hidden = max(16, int(config.d_model))
            # v8d: input dim 10 = 5 marginal stats + 5 class-conditional stats
            # (fisher_log, max_class_shift, mean_class_shift, log_std_ratio, class_balance)
            self.net = nn.Sequential(
                nn.Linear(10, hidden),
                nn.SiLU(),
                nn.Linear(hidden, int(config.d_model)),
                nn.RMSNorm(int(config.d_model)),
            )

        def forward(self, stats: torch.Tensor) -> torch.Tensor:
            """Encode pre-computed feature statistics. stats: (B, d, 10)."""
            return self.net(stats)


    class _HarmonicBranch(nn.Module):
        """SIREN-inspired branch for detecting chaotic/periodic structure in feature statistics."""

        def __init__(self, d_model: int, omega_0: float = 0.3) -> None:
            super().__init__()
            self.omega_0 = omega_0
            # v8d: 10 input stats (5 marginal + 5 class-conditional)
            self.proj_in = nn.Linear(10, d_model)
            self.norm_in = nn.RMSNorm(d_model)
            self.hidden1 = nn.Linear(d_model, d_model)
            self.norm1 = nn.RMSNorm(d_model)
            self.hidden2 = nn.Linear(d_model, d_model)
            self.norm2 = nn.RMSNorm(d_model)
            self.hidden3 = nn.Linear(d_model, d_model)
            self.norm_out = nn.RMSNorm(d_model)
            # SIREN first-layer init: U(-1/fan_in, 1/fan_in)
            _fan_in = 10
            nn.init.uniform_(self.proj_in.weight, -1.0 / _fan_in, 1.0 / _fan_in)
            nn.init.zeros_(self.proj_in.bias)
            # SIREN subsequent-layer init: U(-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0)
            _bound = math.sqrt(6.0 / d_model) / omega_0
            for layer in (self.hidden1, self.hidden2, self.hidden3):
                nn.init.uniform_(layer.weight, -_bound, _bound)
                nn.init.zeros_(layer.bias)

        def forward(self, stats: torch.Tensor) -> torch.Tensor:
            """stats: (B, d, 5) -> (B, d, d_model)"""
            h = torch.sin(self.omega_0 * self.proj_in(stats))
            h = self.norm_in(h)
            h = torch.sin(self.hidden1(h))
            h = self.norm1(h)
            h = torch.sin(self.hidden2(h))
            h = self.norm2(h)
            h = torch.sin(self.hidden3(h))
            return self.norm_out(h)


    class _AxialAttentionBlock(nn.Module):
        def __init__(self, config: TabenticsDiakrinoConfig) -> None:
            super().__init__()
            d_model = int(config.d_model)
            n_heads = max(1, min(int(config.n_heads), d_model))
            while d_model % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            hidden = max(d_model, d_model * int(config.ffn_expansion))
            self.feature_norm1 = nn.RMSNorm(d_model)
            self.feature_attn = _SDPASelfAttention(d_model, n_heads, dropout=float(config.dropout))
            self.feature_norm2 = nn.RMSNorm(d_model)
            self.feature_ffn = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.SiLU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(hidden, d_model),
            )
            self.sample_norm1 = nn.RMSNorm(d_model)
            self.sample_attn = _SDPASelfAttention(d_model, n_heads, dropout=float(config.dropout))
            self.sample_norm2 = nn.RMSNorm(d_model)
            self.sample_ffn = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.SiLU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(hidden, d_model),
            )
            self.dropout = nn.Dropout(float(config.dropout))

        def forward(
            self,
            tokens: torch.Tensor,
            *,
            sample_valid: torch.Tensor,
            feature_valid_mask: torch.Tensor,
            rope: "FeatureIDRoPE | None" = None,
            feature_coordinates: torch.Tensor | None = None,
            support_count: int | None = None,
        ) -> torch.Tensor:
            batch_size, sample_count, feature_count, d_model = tokens.shape

            # Expand feature coordinates for the batched feature-attention call
            rope_coords = None
            if rope is not None and feature_coordinates is not None:
                # feature_coordinates: (batch, n_features) -> repeat per sample
                rope_coords = feature_coordinates.unsqueeze(1).expand(batch_size, sample_count, feature_count).reshape(batch_size * sample_count, feature_count)

            feature_padding = (~feature_valid_mask).unsqueeze(1).expand(batch_size, sample_count, feature_count).reshape(batch_size * sample_count, feature_count)
            feature_tokens = tokens.reshape(batch_size * sample_count, feature_count, d_model)
            feature_norm = self.feature_norm1(feature_tokens)
            feature_attended = self.feature_attn(
                feature_norm,
                key_padding_mask=feature_padding,
                rope=rope,
                feature_coordinates=rope_coords,
            )
            feature_tokens = feature_tokens + self.dropout(feature_attended)
            feature_tokens = feature_tokens + self.dropout(self.feature_ffn(self.feature_norm2(feature_tokens)))
            tokens = feature_tokens.reshape(batch_size, sample_count, feature_count, d_model)

            # Asymmetric sample attention mask: query tokens attend only to support keys
            sample_causal_mask = None
            if support_count is not None and 0 < support_count < sample_count:
                sample_causal_mask = tokens.new_zeros((1, 1, sample_count, sample_count))
                # Block all attention to query keys: support→query and query→query
                sample_causal_mask[:, :, :, support_count:] = float("-inf")

            sample_padding = (~sample_valid).unsqueeze(1).expand(batch_size, feature_count, sample_count).reshape(batch_size * feature_count, sample_count)
            sample_tokens = tokens.transpose(1, 2).reshape(batch_size * feature_count, sample_count, d_model)
            sample_norm = self.sample_norm1(sample_tokens)
            sample_attended = self.sample_attn(
                sample_norm,
                key_padding_mask=sample_padding,
                causal_mask=sample_causal_mask,
            )
            sample_tokens = sample_tokens + self.dropout(sample_attended)
            sample_tokens = sample_tokens + self.dropout(self.sample_ffn(self.sample_norm2(sample_tokens)))
            return sample_tokens.reshape(batch_size, feature_count, sample_count, d_model).transpose(1, 2)


    class _HModule(nn.Module):
        """HRM high-level module: slow planning state update."""

        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.norm_h = nn.RMSNorm(d_model)
            self.norm_l = nn.RMSNorm(d_model)
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model, bias=False),
                nn.GELU(),
                nn.Linear(d_model, d_model, bias=False),
            )
            # Zero-init so H-module starts as identity
            nn.init.zeros_(self.gate[2].weight)

        def forward(self, z_h: torch.Tensor, z_l_pooled: torch.Tensor) -> torch.Tensor:
            combined = torch.cat([self.norm_h(z_h), self.norm_l(z_l_pooled)], dim=-1)
            return z_h + self.gate(combined)


    class TabenticsDiakrino(nn.Module):
        """Axial DIAKRINO with ICL decoder and HRM-style hierarchical recurrent reasoning."""

        def __init__(self, config: TabenticsDiakrinoConfig | None = None) -> None:
            super().__init__()
            self.config = config or TabenticsDiakrinoConfig()
            _d = int(self.config.d_model)
            self.tokenizer = _FourierTokenizer(self.config)
            self.feature_stats_encoder = _FeatureStatsEncoder(self.config)
            # ICL label encoder: index 0 = unknown (query), 1..max_classes = class labels
            self.label_encoder = nn.Embedding(int(self.config.max_classes) + 1, _d)
            # L-module: shared axial attention blocks (weight-tied across HRM steps)
            self.l_blocks = nn.ModuleList([_AxialAttentionBlock(self.config) for _ in range(int(self.config.n_layers))])
            # H-module: slow planning state update
            self.h_module = _HModule(_d)
            # Initial H-state (broadcast to batch)
            self.z_h_init = nn.Parameter(torch.zeros(_d))
            nn.init.normal_(self.z_h_init, mean=0.0, std=0.02)
            # Feature-ID RoPE: content-based rotary position embedding for feature attention
            if FeatureIDCoordinateGenerator is not None and FeatureIDRoPE is not None:
                n_heads = max(1, min(int(self.config.n_heads), int(self.config.d_model)))
                while int(self.config.d_model) % n_heads != 0 and n_heads > 1:
                    n_heads -= 1
                head_dim = int(self.config.d_model) // n_heads
                self.rope_coord_gen = FeatureIDCoordinateGenerator()
                self.rope = FeatureIDRoPE(head_dim)
            else:
                self.rope_coord_gen = None
                self.rope = None
            # ICL decoder head: extracts predictions from label column of test rows
            self.decoder_head = nn.Sequential(
                nn.Linear(_d, _d * 2),
                nn.GELU(),
                nn.Linear(_d * 2, int(self.config.max_classes)),
            )
            self.salience_head = nn.Linear(_d, 1)
            # v8d Stage 2: TIED gate. The same `salience_head` is queried at
            # two points: pre-attention on `feature_bias` (used as the
            # multiplicative feature gate) and post-attention on the pooled
            # support hidden state (used as the auxiliary salience prediction).
            # A learnable scalar offset keeps the gate near-identity at init
            # (~+2.0 -> sigmoid 0.88) without disturbing the salience side.
            self.gate_bias_offset = nn.Parameter(torch.tensor(2.0))
            # MAE reconstruction head: per-feature linear from d_model to scalar
            self.mae_head = nn.Sequential(
                nn.Linear(_d, _d),
                nn.SiLU(),
                nn.Linear(_d, 1),
            )
            self.latent_projector = nn.Sequential(
                nn.Linear(_d, _d),
                nn.GELU(),
                nn.Linear(_d, _d),
            )
            self.latent_ln = nn.RMSNorm(_d)
            self.latent_predictor = nn.Sequential(
                nn.Linear(_d, _d),
                nn.GELU(),
                nn.Linear(_d, _d),
            )
            # SIREN-inspired harmonic branch + chaotic gate for periodic/chaotic detection
            self.harmonic_branch = _HarmonicBranch(_d)
            if self.config.enable_chaotic_head:
                self.chaotic_gate = nn.Sequential(
                    nn.Linear(_d * 3, _d),
                    nn.GELU(),
                    nn.Linear(_d, 1),
                )
            # Harmonic fusion: gated residual of harmonic features into feature_bias
            # Gate initialized at 0 so harmonic contribution starts as zero perturbation
            if self.config.harmonic_fusion:
                self.harmonic_fusion_proj = nn.Linear(_d, _d)
                self.harmonic_fusion_gate = nn.Parameter(torch.zeros(1))
            projection_count = max(1, int(self.config.sigreg_projections))
            directions = torch.randn(projection_count, _d, dtype=torch.float32)
            directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            self.register_buffer("sigreg_directions", directions, persistent=True)
            self.output_norm = nn.RMSNorm(_d)
            self.dropout = nn.Dropout(float(self.config.dropout))
            # Optional meta-feature encoder for dataset-level conditioning
            if self.config.use_meta_features:
                _meta_dim = max(1, int(self.config.meta_feature_dim))
                self.meta_encoder = nn.Sequential(
                    nn.Linear(_meta_dim, _d),
                    nn.SiLU(),
                    nn.Linear(_d, _d),
                    nn.RMSNorm(_d),
                )
            else:
                self.meta_encoder = None
            # EMA teacher: shadow copy of encoder path (no gradients)
            self.ema_tokenizer = copy.deepcopy(self.tokenizer)
            self.ema_feature_stats_encoder = copy.deepcopy(self.feature_stats_encoder)
            self.ema_label_encoder = copy.deepcopy(self.label_encoder)
            self.ema_l_blocks = copy.deepcopy(self.l_blocks)
            self.ema_h_module = copy.deepcopy(self.h_module)
            self.ema_output_norm = copy.deepcopy(self.output_norm)
            self.register_buffer("ema_z_h_init", self.z_h_init.data.clone())
            if self.rope_coord_gen is not None:
                self.ema_rope_coord_gen = copy.deepcopy(self.rope_coord_gen)
            else:
                self.ema_rope_coord_gen = None
            if self.meta_encoder is not None:
                self.ema_meta_encoder = copy.deepcopy(self.meta_encoder)
            else:
                self.ema_meta_encoder = None
            # EMA copies of harmonic fusion modules
            if self.config.harmonic_fusion:
                self.ema_harmonic_branch = copy.deepcopy(self.harmonic_branch)
                self.ema_harmonic_fusion_proj = copy.deepcopy(self.harmonic_fusion_proj)
                self.register_buffer("ema_harmonic_fusion_gate", self.harmonic_fusion_gate.data.clone())
            _ema_modules: list[nn.Module] = [
                self.ema_tokenizer,
                self.ema_feature_stats_encoder,
                self.ema_label_encoder,
                self.ema_l_blocks,
                self.ema_h_module,
                self.ema_output_norm,
            ]
            if self.ema_rope_coord_gen is not None:
                _ema_modules.append(self.ema_rope_coord_gen)
            if self.ema_meta_encoder is not None:
                _ema_modules.append(self.ema_meta_encoder)
            if self.config.harmonic_fusion:
                _ema_modules.append(self.ema_harmonic_branch)
                _ema_modules.append(self.ema_harmonic_fusion_proj)
            for mod in _ema_modules:
                for p in mod.parameters():
                    p.requires_grad_(False)
                mod.eval()

        def train(self, mode: bool = True) -> "TabenticsDiakrino":
            """Override to keep EMA modules permanently in eval mode."""
            super().train(mode)
            # EMA teacher modules must stay in eval so their dropout is disabled
            for mod in (
                self.ema_tokenizer,
                self.ema_feature_stats_encoder,
                self.ema_label_encoder,
                self.ema_l_blocks,
                self.ema_h_module,
                self.ema_output_norm,
            ):
                mod.eval()
            if self.ema_rope_coord_gen is not None:
                self.ema_rope_coord_gen.eval()
            if self.ema_meta_encoder is not None:
                self.ema_meta_encoder.eval()
            if self.config.harmonic_fusion:
                self.ema_harmonic_branch.eval()
                self.ema_harmonic_fusion_proj.eval()
            return self

        @torch.no_grad()
        def _update_ema(self) -> None:
            tau = float(self.config.ema_decay)
            pairs: list[tuple[nn.Module, nn.Module]] = [
                (self.tokenizer, self.ema_tokenizer),
                (self.feature_stats_encoder, self.ema_feature_stats_encoder),
                (self.label_encoder, self.ema_label_encoder),
                (self.l_blocks, self.ema_l_blocks),
                (self.h_module, self.ema_h_module),
                (self.output_norm, self.ema_output_norm),
            ]
            if self.rope_coord_gen is not None and self.ema_rope_coord_gen is not None:
                pairs.append((self.rope_coord_gen, self.ema_rope_coord_gen))
            if self.meta_encoder is not None and self.ema_meta_encoder is not None:
                pairs.append((self.meta_encoder, self.ema_meta_encoder))
            if self.config.harmonic_fusion:
                pairs.append((self.harmonic_branch, self.ema_harmonic_branch))
                pairs.append((self.harmonic_fusion_proj, self.ema_harmonic_fusion_proj))
            for student, teacher in pairs:
                for sp, tp in zip(student.parameters(), teacher.parameters()):
                    tp.data.mul_(tau).add_(sp.data, alpha=1.0 - tau)
            # EMA update for z_h_init parameter
            self.ema_z_h_init.mul_(tau).add_(self.z_h_init.data, alpha=1.0 - tau)
            # EMA update for harmonic fusion gate (buffer, not a module parameter)
            if self.config.harmonic_fusion:
                self.ema_harmonic_fusion_gate.mul_(tau).add_(self.harmonic_fusion_gate.data, alpha=1.0 - tau)

        def _project_latent(self, x: torch.Tensor) -> torch.Tensor:
            """Apply latent projector + LayerNorm."""
            return self.latent_ln(self.latent_projector(x))

        def _encode_joint_tokens_ema(
            self,
            *,
            support: torch.Tensor,
            support_mask: torch.Tensor,
            support_valid: torch.Tensor,
            support_labels: torch.Tensor,
            feature_valid_mask: torch.Tensor,
            meta_vectors: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Encode support-only pass through the EMA teacher (ICL-style label column, no gradients)."""
            support_tokens = self.ema_tokenizer(support, missing_mask=support_mask)
            feature_stats_marginal = _compute_feature_stats(
                support,
                support_mask=support_mask,
                support_valid=support_valid,
            )
            # v8d: include class-conditional stats in EMA path too (must match
            # student dim 10).
            feature_stats_class = _compute_class_conditional_stats(
                support,
                support_mask=support_mask,
                support_valid=support_valid,
                support_labels=support_labels,
                num_classes_max=int(self.config.max_classes),
            )
            feature_stats_raw = torch.cat([feature_stats_marginal, feature_stats_class], dim=-1)
            feature_bias = self.ema_feature_stats_encoder(feature_stats_raw)
            # Harmonic fusion (EMA path): use EMA copies of harmonic branch + proj
            if self.config.harmonic_fusion:
                harmonic_features = self.ema_harmonic_branch(feature_stats_raw)
                harmonic_bias = self.ema_harmonic_fusion_proj(harmonic_features)
                gate = torch.tanh(self.ema_harmonic_fusion_gate)
                feature_bias = feature_bias + gate * harmonic_bias
            support_tokens = support_tokens + feature_bias.unsqueeze(1)

            # Encode labels as extra column (ICL)
            max_cls = max(0, int(self.config.max_classes) - 1)
            clamped_labels = torch.clamp(support_labels, min=0, max=max_cls)
            support_label_tokens = self.ema_label_encoder(clamped_labels + 1)
            support_label_tokens = support_label_tokens * support_valid.unsqueeze(-1).to(dtype=support_label_tokens.dtype)
            support_label_col = support_label_tokens.unsqueeze(2)
            joint_tokens = torch.cat([support_tokens, support_label_col], dim=2)  # [B, S, F+1, D]

            batch_size = support.shape[0]
            label_col_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=support.device)
            extended_feature_mask = torch.cat([feature_valid_mask, label_col_valid], dim=1)

            joint_valid = support_valid
            # Meta-feature conditioning in EMA path
            if self.ema_meta_encoder is not None and meta_vectors is not None:
                meta_bias = self.ema_meta_encoder(meta_vectors)
                joint_tokens = joint_tokens + meta_bias[:, None, None, :]
            # EMA modules stay in eval mode so dropout is disabled
            feature_coords = None
            if self.ema_rope_coord_gen is not None and self.rope is not None:
                feature_coords = self.ema_rope_coord_gen(
                    support,
                    support_mask=support_mask,
                    support_valid=support_valid,
                )
                label_coord = feature_coords.new_zeros((batch_size, 1))
                feature_coords = torch.cat([feature_coords, label_coord], dim=1)

            # HRM-style recurrence through EMA L-blocks + H-module
            z_H = self.ema_z_h_init.expand(batch_size, -1)
            N, T = self.config.hrm_outer_cycles, self.config.hrm_inner_steps
            for step in range(N * T):
                z_cond = joint_tokens + z_H[:, None, None, :]
                joint_tokens = self._run_l_blocks(
                    z_cond, self.ema_l_blocks, joint_valid, extended_feature_mask, feature_coords,
                    support_count=None,
                )
                if (step + 1) % T == 0:
                    z_L_pooled = self._masked_mean(joint_tokens, joint_valid, dim=1).mean(dim=1)
                    z_H = self.ema_h_module(z_H, z_L_pooled)
            return self.ema_output_norm(joint_tokens), feature_stats_raw, extended_feature_mask

        @staticmethod
        def _masked_mean(values: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
            weight = mask.to(dtype=values.dtype)
            while weight.ndim < values.ndim:
                weight = weight.unsqueeze(-1)
            total = (values * weight).sum(dim=dim)
            denom = torch.clamp(weight.sum(dim=dim), min=1.0)
            return total / denom

        def _run_l_blocks(
            self,
            z: torch.Tensor,
            blocks: nn.ModuleList,
            sample_valid: torch.Tensor,
            feature_valid_mask: torch.Tensor,
            feature_coords: torch.Tensor | None,
            *,
            support_count: int | None,
        ) -> torch.Tensor:
            """Run a single pass through all L-module blocks."""
            for block in blocks:
                if self.config.gradient_checkpointing and self.training:
                    z = torch.utils.checkpoint.checkpoint(
                        block,
                        z,
                        sample_valid=sample_valid,
                        feature_valid_mask=feature_valid_mask,
                        rope=self.rope,
                        feature_coordinates=feature_coords,
                        support_count=support_count,
                        use_reentrant=False,
                    )
                else:
                    z = block(
                        z,
                        sample_valid=sample_valid,
                        feature_valid_mask=feature_valid_mask,
                        rope=self.rope,
                        feature_coordinates=feature_coords,
                        support_count=support_count,
                    )
            return z

        def embed_and_init_state(
            self,
            batch: TabenticsDiakrinoPreparedBatch,
        ) -> tuple[torch.Tensor, torch.Tensor, dict]:
            """Tokenize input and initialize HRM state.

            Returns:
                z_L:  [B, S, F+1, D] initial token embeddings (not yet refined)
                z_H:  [B, D] initial H-module state
                ctx:  dict with keys needed by hrm_step and decode_outputs
            """
            joint_tokens, feature_stats_raw, extended_feature_mask = self._encode_joint_tokens(
                support=batch.support,
                support_mask=batch.support_mask,
                support_valid=batch.support_valid,
                support_labels=batch.support_labels,
                feature_valid_mask=batch.feature_valid_mask,
                query=batch.query,
                query_mask=batch.query_mask,
                query_valid=batch.query_valid,
                meta_vectors=batch.meta_vectors,
            )
            batch_size = joint_tokens.shape[0]
            z_H = self.z_h_init.expand(batch_size, -1)

            # Precompute RoPE coordinates
            feature_coords = None
            if self.rope_coord_gen is not None and self.rope is not None:
                feature_coords = self.rope_coord_gen(
                    batch.support,
                    support_mask=batch.support_mask,
                    support_valid=batch.support_valid,
                )
                label_coord = feature_coords.new_zeros((batch_size, 1))
                feature_coords = torch.cat([feature_coords, label_coord], dim=1)

            support_count = int(batch.support.shape[1])
            sample_valid = torch.cat([batch.support_valid, batch.query_valid], dim=1) if batch.query is not None else batch.support_valid

            ctx = {
                "feature_stats_raw": feature_stats_raw,
                "extended_feature_mask": extended_feature_mask,
                "feature_coords": feature_coords,
                "support_count": support_count,
                "sample_valid": sample_valid,
                "x_embed": joint_tokens,  # kept for grad flow to embedding layers
            }
            return joint_tokens, z_H, ctx

        def hrm_cycle(
            self,
            z_L: torch.Tensor,
            z_H: torch.Tensor,
            ctx: dict,
            *,
            inject_embed: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """One HRM outer cycle: T inner L-block steps + one H-module update.

            This is the atomic unit for deep supervision — the training loop
            calls this once per cycle and applies decode→loss→backward→step
            at each cycle boundary.

            Args:
                inject_embed: if True, add x_embed (original token embedding)
                    to the conditioning signal on the first inner step, keeping
                    gradient flow to the embedding layers.
            """
            T = int(self.config.hrm_inner_steps)

            ext_mask = ctx["extended_feature_mask"]
            f_coords = ctx["feature_coords"]
            s_count = ctx["support_count"]
            s_valid = ctx["sample_valid"]
            x_embed = ctx.get("x_embed") if inject_embed else None

            for inner in range(T):
                z_cond = z_L + z_H[:, None, None, :]
                if inner == 0 and x_embed is not None:
                    z_cond = z_cond + x_embed
                z_L = self._run_l_blocks(
                    z_cond, self.l_blocks, s_valid, ext_mask, f_coords,
                    support_count=s_count,
                )
            z_L_pooled = self._masked_mean(z_L, s_valid, dim=1).mean(dim=1)
            z_H = self.h_module(z_H, z_L_pooled)
            return z_L, z_H

        def hrm_step(
            self,
            z_L: torch.Tensor,
            z_H: torch.Tensor,
            ctx: dict,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """One HRM segment: N outer cycles × T inner steps.

            Used for inference / validation where we don't need per-cycle
            gradients.  Runs the first N-1 cycles without gradients when
            hrm_full_backprop_last=True.

            For training with deep supervision, the fit() loop calls
            hrm_cycle() directly instead.
            """
            N = int(self.config.hrm_outer_cycles)
            full_bp = bool(self.config.hrm_full_backprop_last)

            if full_bp:
                # N-1 cycles without grad, last cycle with grad
                with torch.no_grad():
                    for _outer in range(N - 1):
                        z_L, z_H = self.hrm_cycle(z_L, z_H, ctx, inject_embed=False)
                z_L, z_H = self.hrm_cycle(z_L, z_H, ctx, inject_embed=True)
            else:
                # Legacy: all cycles without grad except the last T steps
                T = int(self.config.hrm_inner_steps)
                total_steps = N * T
                ext_mask = ctx["extended_feature_mask"]
                f_coords = ctx["feature_coords"]
                s_count = ctx["support_count"]
                s_valid = ctx["sample_valid"]
                x_embed = ctx.get("x_embed")

                with torch.no_grad():
                    for step in range(total_steps - 1):
                        z_cond = z_L + z_H[:, None, None, :]
                        z_L = self._run_l_blocks(
                            z_cond, self.l_blocks, s_valid, ext_mask, f_coords,
                            support_count=s_count,
                        )
                        if (step + 1) % T == 0:
                            z_L_pooled = self._masked_mean(z_L, s_valid, dim=1).mean(dim=1)
                            z_H = self.h_module(z_H, z_L_pooled)

                z_cond = z_L + z_H[:, None, None, :]
                if x_embed is not None:
                    z_cond = z_cond + x_embed
                z_L = self._run_l_blocks(
                    z_cond, self.l_blocks, s_valid, ext_mask, f_coords,
                    support_count=s_count,
                )
                z_L_pooled = self._masked_mean(z_L, s_valid, dim=1).mean(dim=1)
                z_H = self.h_module(z_H, z_L_pooled)

            return z_L, z_H

        def decode_outputs(
            self,
            z_L: torch.Tensor,
            z_H: torch.Tensor,
            batch: TabenticsDiakrinoPreparedBatch,
            ctx: dict,
            *,
            skip_teacher: bool = False,
        ) -> TabenticsDiakrinoOutputs:
            """Decode refined HRM state into classification logits + auxiliary heads.

            Args:
                skip_teacher: If True, skip EMA teacher path and return zeros for
                    JEPA latent prediction/target and SIGReg samples. Used in
                    pretrain_clean mode.
            """
            support_count = ctx["support_count"]
            feature_stats_raw = ctx["feature_stats_raw"]
            ext_mask = ctx["extended_feature_mask"]

            z_normed = self.output_norm(z_L)
            support_hidden = z_normed[:, :support_count]
            query_hidden = z_normed[:, support_count:]

            # ICL decode: extract label column from query rows → decoder head
            # Label column is the last feature column (index -1)
            # Condition on z_H (global planning state from H-module)
            query_label_embeds = query_hidden[:, :, -1] + z_H.unsqueeze(1)  # [B, Q, D]
            logits = self.decoder_head(query_label_embeds)  # [B, Q, max_classes]

            # Masked means for auxiliary heads
            feature_mask_ext = ext_mask.unsqueeze(1).expand(-1, support_hidden.shape[1], -1)
            support_sample_mask = batch.support_valid.unsqueeze(-1).expand(-1, -1, support_hidden.shape[2]) & feature_mask_ext

            support_pooled = self._masked_mean(support_hidden, support_sample_mask, dim=2)

            # Salience head (on original feature columns only, excluding label column)
            support_feature_mean = self._masked_mean(
                support_hidden[:, :, :-1],  # exclude label column
                batch.support_valid.unsqueeze(-1).expand(-1, -1, support_hidden.shape[2] - 1) & batch.feature_valid_mask.unsqueeze(1).expand(-1, support_hidden.shape[1], -1),
                dim=1,
            )
            importance_logits = self.salience_head(support_feature_mean).squeeze(-1)
            mae_reconstruction = self.mae_head(support_hidden[:, :, :-1]).squeeze(-1)

            global_state = self._masked_mean(support_pooled, batch.support_valid, dim=1)

            if skip_teacher:
                # No EMA teacher path — return zeros for JEPA/SIGReg
                _d = int(self.config.d_model)
                latent_target = global_state.new_zeros(global_state.shape[0], _d)
                latent_prediction = latent_target
                sigreg_samples = global_state.new_zeros(1, _d)
            else:
                # EMA teacher for JEPA
                clean_support_mask = torch.zeros_like(batch.support_mask)
                with torch.no_grad():
                    clean_joint_tokens, _clean_fs_raw, clean_ext_mask = self._encode_joint_tokens_ema(
                        support=batch.support_clean,
                        support_mask=clean_support_mask,
                        support_valid=batch.support_valid,
                        support_labels=batch.support_labels,
                        feature_valid_mask=batch.feature_valid_mask,
                        meta_vectors=batch.meta_vectors,
                    )
                clean_support_hidden = clean_joint_tokens

                clean_support_sample_mask = (
                    batch.support_valid.unsqueeze(-1).expand(-1, -1, clean_support_hidden.shape[2]) & feature_mask_ext
                )
                clean_support_pooled = self._masked_mean(clean_support_hidden, clean_support_sample_mask, dim=2)

                clean_support_feature_mean = self._masked_mean(
                    clean_support_hidden[:, :, :-1],
                    batch.support_valid.unsqueeze(-1).expand(-1, -1, clean_support_hidden.shape[2] - 1) & batch.feature_valid_mask.unsqueeze(1).expand(-1, clean_support_hidden.shape[1], -1),
                    dim=1,
                )
                clean_global_state = self._masked_mean(clean_support_pooled, batch.support_valid, dim=1)
                latent_target = self._project_latent(clean_global_state).detach()
                latent_prediction = self.latent_predictor(self._project_latent(global_state))

                projected_feature_latents = torch.cat(
                    [
                        self._project_latent(support_feature_mean),
                        self._project_latent(clean_support_feature_mean),
                    ],
                    dim=0,
                )
                feature_masks = torch.cat([batch.feature_valid_mask, batch.feature_valid_mask], dim=0)
                sigreg_samples = _flatten_valid_feature_latents(projected_feature_latents, feature_masks)

            # Chaotic gate (only when chaotic head is enabled)
            if self.config.enable_chaotic_head:
                harmonic_features = self.harmonic_branch(feature_stats_raw)
                harmonic_pooled = self._masked_mean(harmonic_features, batch.feature_valid_mask, dim=1)
                chaotic_combined = torch.cat(
                    [global_state, harmonic_pooled, global_state * harmonic_pooled],
                    dim=-1,
                )
                chaotic_logit = self.chaotic_gate(chaotic_combined).squeeze(-1)
            else:
                chaotic_logit = global_state.new_zeros(global_state.shape[0])

            harmonic_gate_value = float(torch.tanh(self.harmonic_fusion_gate).item()) if self.config.harmonic_fusion else 0.0

            # v8d Stage 2: gate logits = salience_head(feature_bias) + gate_bias_offset.
            # Recomputed fresh from the detached stash each decode call so the
            # computation graph is new for every loss.backward() in the HRM
            # deep-supervision loop (M×N backward passes per step). Gradients
            # flow into salience_head (tied) and into the offset scalar.
            _fb_det = getattr(self, "_last_feature_bias_detached", None)
            if _fb_det is not None:
                _gate_logits_out = self.salience_head(_fb_det).squeeze(-1) + self.gate_bias_offset
            else:
                _gate_logits_out = importance_logits.new_zeros(importance_logits.shape)
            return TabenticsDiakrinoOutputs(
                logits=logits,
                importance_logits=importance_logits,
                gate_logits=_gate_logits_out,
                mae_reconstruction=mae_reconstruction,
                chaotic_logit=chaotic_logit,
                harmonic_gate_value=harmonic_gate_value,
                latent_prediction=latent_prediction,
                latent_target=latent_target,
                sigreg_samples=sigreg_samples,
            )

        def _encode_joint_tokens(
            self,
            *,
            support: torch.Tensor,
            support_mask: torch.Tensor,
            support_valid: torch.Tensor,
            support_labels: torch.Tensor,
            feature_valid_mask: torch.Tensor,
            query: torch.Tensor | None = None,
            query_mask: torch.Tensor | None = None,
            query_valid: torch.Tensor | None = None,
            meta_vectors: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Tokenize input with ICL-style label column.

            Returns (joint_tokens [B, S, F+1, D], feature_stats_raw, extended_feature_mask).
            The last column (F) is the label column.
            """
            support_tokens = self.tokenizer(support, missing_mask=support_mask)
            feature_stats_marginal = _compute_feature_stats(
                support,
                support_mask=support_mask,
                support_valid=support_valid,
            )
            # v8d: class-conditional stats (uses y_support) so the gate's input
            # encoder sees label-aware signal, not just marginals over X_support.
            feature_stats_class = _compute_class_conditional_stats(
                support,
                support_mask=support_mask,
                support_valid=support_valid,
                support_labels=support_labels,
                num_classes_max=int(self.config.max_classes),
            )
            feature_stats_raw = torch.cat([feature_stats_marginal, feature_stats_class], dim=-1)  # [B, F, 10]
            feature_bias = self.feature_stats_encoder(feature_stats_raw)
            # Harmonic fusion: add gated SIREN-encoded features into feature_bias
            if self.config.harmonic_fusion:
                harmonic_features = self.harmonic_branch(feature_stats_raw)  # [B, F, D]
                harmonic_bias = self.harmonic_fusion_proj(harmonic_features)  # [B, F, D]
                gate = torch.tanh(self.harmonic_fusion_gate)
                feature_bias = feature_bias + gate * harmonic_bias
            # v8d Stage 2: TIED gate. Use the same `salience_head` as both the
            # pre-attention feature gate (here) and the post-attention salience
            # predictor (in decode_outputs on pooled support hidden state).
            # Both query points are supervised against the chosen salience
            # target (GT active features when use_gt_salience_target=True,
            # or Fisher-derived per_episode_target otherwise). The label-aware
            # information reaches the gate via the class-conditional stats
            # inside `feature_stats_raw` (Stage 1).
            # side because `support_hidden` evolves across cycles; the gate
            # side is computed once per step from cheap stats.
            #
            # The detached feature_bias is also stashed so decode_outputs can
            # recompute gate logits fresh each HRM cycle (avoids "backward
            # through the graph a second time" with deep supervision).
            _fb_detached = feature_bias.detach()
            self._last_feature_bias_detached = _fb_detached
            _gate_logits = self.salience_head(_fb_detached).squeeze(-1) + self.gate_bias_offset  # [B, F]
            _sal_gate = torch.sigmoid(_gate_logits)  # [B, F]
            # Invalid features → gate=1.0 (no-op)
            _fvm = feature_valid_mask.float()
            _sal_gate = _sal_gate * _fvm + (1.0 - _fvm)
            # Use detached gate for the multiplicative path (main CE loss
            # must not backprop through salience_head's gate role at all).
            feature_bias = feature_bias * _sal_gate.detach().unsqueeze(-1)  # [B, F, D] * [B, F, 1]
            # Stash gate values for logging (detached, no grad needed)
            self._last_sal_gate = _sal_gate.detach()
            self._last_sal_gate_mask = feature_valid_mask
            support_tokens = support_tokens + feature_bias.unsqueeze(1)

            # Encode labels as extra column: known labels for support, unknown for query
            max_cls = max(0, int(self.config.max_classes) - 1)
            clamped_labels = torch.clamp(support_labels, min=0, max=max_cls)
            # Shift labels by 1: index 0 = unknown, 1..max_classes = class labels
            support_label_tokens = self.label_encoder(clamped_labels + 1)  # [B, S_sup, D]
            # Mask invalid support rows
            support_label_tokens = support_label_tokens * support_valid.unsqueeze(-1).to(dtype=support_label_tokens.dtype)
            support_label_col = support_label_tokens.unsqueeze(2)  # [B, S_sup, 1, D]

            # Concatenate label as extra feature column for support
            support_with_label = torch.cat([support_tokens, support_label_col], dim=2)  # [B, S_sup, F+1, D]

            # Extend feature_valid_mask with label column (always valid)
            batch_size = support.shape[0]
            label_col_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=support.device)
            extended_feature_mask = torch.cat([feature_valid_mask, label_col_valid], dim=1)

            if query is not None and query_mask is not None and query_valid is not None:
                query_tokens = self.tokenizer(query, missing_mask=query_mask)
                query_tokens = query_tokens + feature_bias.unsqueeze(1)
                # Query labels: unknown token (index 0)
                unknown_indices = torch.zeros(
                    (batch_size, query.shape[1]), dtype=torch.long, device=query.device,
                )
                query_label_tokens = self.label_encoder(unknown_indices)  # [B, S_query, D]
                query_label_col = query_label_tokens.unsqueeze(2)  # [B, S_query, 1, D]
                query_with_label = torch.cat([query_tokens, query_label_col], dim=2)  # [B, S_query, F+1, D]
                joint_tokens = torch.cat([support_with_label, query_with_label], dim=1)
                joint_valid = torch.cat([support_valid, query_valid], dim=1)
            else:
                joint_tokens = support_with_label
                joint_valid = support_valid

            # Meta-feature conditioning: global dataset-level bias
            if self.meta_encoder is not None and meta_vectors is not None:
                meta_bias = self.meta_encoder(meta_vectors)
                joint_tokens = joint_tokens + meta_bias[:, None, None, :]

            joint_tokens = self.dropout(joint_tokens)
            feature_coords = None
            if self.rope_coord_gen is not None and self.rope is not None:
                feature_coords = self.rope_coord_gen(
                    support,
                    support_mask=support_mask,
                    support_valid=support_valid,
                )
                # Extend feature coordinates for the label column
                label_coord = feature_coords.new_zeros((batch_size, 1))
                feature_coords = torch.cat([feature_coords, label_coord], dim=1)
            return joint_tokens, feature_stats_raw, extended_feature_mask

        def _sigreg_loss(self, samples: torch.Tensor) -> torch.Tensor:
            """Epps-Pulley normality test on random 1-D projections (SIGReg).

            For each random direction u, project samples h = Z @ u, standardize,
            then compute the Epps-Pulley test statistic:
                T = ∫ |φ_N(t) - φ_0(t)|² w(t) dt
            where φ_N is the empirical characteristic function, φ_0 = exp(-t²/2),
            and w(t) = exp(-t²/(2λ²)) is a Gaussian weight (λ = 1.0).
            The integral is approximated via trapezoidal quadrature on [0.2, 4.0].
            """
            if samples.ndim != 2:
                raise ValueError(f"Expected 2D latent samples for SIGReg, got {tuple(samples.shape)}")
            N = samples.shape[0]
            if N < 2:
                return samples.new_zeros(())
            directions = self.sigreg_directions.to(device=samples.device, dtype=samples.dtype)
            # projections: (N, M)
            projections = samples @ directions.transpose(0, 1)
            # standardize each projection to zero mean and unit variance
            mean = projections.mean(dim=0, keepdim=True)
            centered = projections - mean
            std = torch.sqrt(torch.clamp(centered.pow(2).mean(dim=0, keepdim=True), min=self.config.eps))
            h = centered / std  # (N, M)

            # Trapezoidal quadrature knots on [0.2, 4.0]
            n_knots = 32
            t = torch.linspace(0.2, 4.0, n_knots, device=samples.device, dtype=samples.dtype)  # (K,)
            dt = t[1] - t[0]

            # Empirical CF: φ_N(t) = (1/N) Σ_n exp(i·t·h_n)
            # th shape: (N, M, K) = h[:,:,None] * t[None,None,:]
            th = h.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0)
            ecf_real = th.cos().mean(dim=0)   # (M, K)
            ecf_imag = th.sin().mean(dim=0)   # (M, K)

            # Gaussian CF: φ_0(t) = exp(-t²/2)
            gcf = torch.exp(-0.5 * t.pow(2))  # (K,)

            # |φ_N - φ_0|²
            diff_real = ecf_real - gcf.unsqueeze(0)
            diff_imag = ecf_imag  # φ_0 is real
            integrand = diff_real.pow(2) + diff_imag.pow(2)  # (M, K)

            # Gaussian weight w(t) = exp(-t²/2) (λ=1)
            weight = torch.exp(-0.5 * t.pow(2))  # (K,)
            integrand = integrand * weight.unsqueeze(0)

            # Trapezoidal rule over t for each projection, then average over M
            per_direction = torch.trapezoid(integrand, dx=float(dt.item()), dim=-1)  # (M,)
            return per_direction.mean()

        def forward(self, batch: TabenticsDiakrinoPreparedBatch) -> TabenticsDiakrinoOutputs:
            """Full HRM forward pass (all segments) for evaluation/inference."""
            z_L, z_H, ctx = self.embed_and_init_state(batch)
            M = int(self.config.hrm_segments)
            for _ in range(M):
                z_L, z_H = self.hrm_step(z_L, z_H, ctx)
            return self.decode_outputs(z_L, z_H, batch, ctx)

        @staticmethod
        def _stable_max_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """Stable-max cross-entropy (Prieto et al. 2025, used in HRM & TRM).

            Uses softplus instead of relu for activations so that gradient
            flows to non-argmax classes (relu kills gradients for classes that
            are not the current winner, making early training impossible).
            Log-domain computation for numerical stability.
            """
            # x: [N, C], targets: [N]
            C = logits.shape[-1]
            # For each class i, compute max over all other classes
            expanded = logits.unsqueeze(-1).expand(-1, C, C)  # [N, C, C]
            mask_diag = torch.eye(C, device=logits.device, dtype=torch.bool).unsqueeze(0)
            filled = expanded.masked_fill(mask_diag, float("-inf"))
            max_others = filled.max(dim=-2).values  # [N, C]
            # softplus activations: always positive, always has gradient
            activations = F.softplus(logits - max_others)  # [N, C]
            log_activations = torch.log(activations.clamp(min=1e-10))
            log_denom = torch.logsumexp(log_activations, dim=-1, keepdim=True)
            log_probs = log_activations - log_denom  # [N, C]
            return F.nll_loss(log_probs, targets)

        def compute_losses(self, outputs: TabenticsDiakrinoOutputs, batch: TabenticsDiakrinoPreparedBatch) -> tuple[torch.Tensor, JsonDict]:
            components: JsonDict = {}
            valid_queries = batch.query_valid & (batch.query_labels >= 0)
            label_limit = max(0, int(self.config.max_classes) - 1)
            clamped_query_labels = torch.clamp(batch.query_labels, min=0, max=label_limit)
            logits = outputs.logits
            # Fix 6: class masking — mask logits for inactive classes to -inf
            if self.config.class_masking:
                # class_counts: [B] — number of active classes per sample
                # Active classes are indices 0..class_counts[b]-1
                n_classes = logits.shape[-1]
                class_range = torch.arange(n_classes, device=logits.device)  # [C]
                counts = batch.class_counts  # [B]
                # Build per-sample active mask: [B, C]
                active = class_range.unsqueeze(0) < counts.unsqueeze(-1)  # [B, C]
                # Broadcast to logits shape [B, Q, C]
                if logits.dim() == 3:
                    mask = ~active.unsqueeze(1).expand_as(logits)
                else:
                    mask = ~active.expand_as(logits)
                logits = logits.masked_fill(mask, float("-inf"))
            if torch.any(valid_queries):
                # Fix 5: stable-max loss (Prieto et al. 2025)
                if self.config.use_stable_max:
                    ce = self._stable_max_loss(logits[valid_queries], clamped_query_labels[valid_queries])
                else:
                    ce = F.cross_entropy(logits[valid_queries], clamped_query_labels[valid_queries])
            else:
                ce = outputs.logits.new_zeros(())
            feature_mask = batch.feature_valid_mask
            if torch.any(feature_mask):
                # v8f: choose supervision target based on config.
                # use_gt_salience_target=True → active_feature_targets (true
                # causal features from synthetic generation). The Fisher
                # signal remains as an input feature via class-conditional
                # stats but no longer serves as the supervision target.
                # use_gt_salience_target=False → per_episode_target (Fisher-
                # derived proxy, legacy v8d behavior).
                if getattr(self, '_use_gt_salience_target', False):
                    _sal_supervision = batch.active_feature_targets
                else:
                    _sal_supervision = batch.per_episode_target
                sal_targets_masked = _sal_supervision[feature_mask]
                # Sparse-target reweighting: predicting all-zero would otherwise
                # score ~95% accuracy because typically only ~5-10% of feature
                # columns are "active" per head. Use pos_weight = neg/pos
                # computed per batch so the BCE gradient on positives is
                # rescaled to match the (much larger) negative population.
                # This is the standard imbalanced-BCE recipe and is exactly
                # what `BCEWithLogitsLoss(pos_weight=...)` is designed for.
                # Focal loss (Lin et al. 2017) replaces BCE+pos_weight: handles
                # extreme class imbalance (~1-6%% positives) by down-weighting
                # easy negatives via (1-p_t)^gamma so the gradient is dominated
                # by hard examples (especially rare true positives). Escapes the
                # predict all-zero saturation trap that pos_weight clamping
                # cannot. alpha=0.75 upweights the rare positive class.
                _focal_alpha = 0.75
                _focal_gamma = 2.0
                _sal_logits = outputs.importance_logits[feature_mask]
                _sal_targets = sal_targets_masked
                # v8f2: dynamic pos_weight for GT targets. GT active features
                # are extremely sparse (~2-9% positive). Without pos_weight the
                # BCE gradient on the rare positives is swamped by the mass of
                # easy negatives, causing predict-all-zero collapse. Compute
                # neg/pos ratio per batch and inject via pos_weight so the raw
                # BCE on positive elements is amplified before focal weighting.
                _num_pos = _sal_targets.sum().clamp(min=1.0)
                _num_neg = (_sal_targets.numel() - _sal_targets.sum()).clamp(min=1.0)
                _pos_weight = (_num_neg / _num_pos).clamp(max=50.0)
                _pw_tensor = _sal_targets.new_tensor([_pos_weight])
                _ce_terms = F.binary_cross_entropy_with_logits(
                    _sal_logits, _sal_targets, reduction="none",
                    pos_weight=_pw_tensor,
                )
                _p = torch.sigmoid(_sal_logits)
                _p_t = _p * _sal_targets + (1.0 - _p) * (1.0 - _sal_targets)
                _alpha_t = _focal_alpha * _sal_targets + (1.0 - _focal_alpha) * (1.0 - _sal_targets)
                _focal_weight = _alpha_t * (1.0 - _p_t).pow(_focal_gamma)
                # RetinaNet normalization: divide by number of positive anchors
                # (not total elements) so absolute gradient magnitude on positives
                # does not vanish when negatives dominate the batch.
                salience = (_focal_weight * _ce_terms).sum() / _num_pos
                # v8c: gate-side focal BCE. Same recipe as classification
                # focal loss but on outputs.gate_logits, providing the gate
                # head with an independent supervision signal so it cannot be
                # corrupted by the post-attention classification gradient.
                _gl = outputs.gate_logits[feature_mask]
                _ce_gate = F.binary_cross_entropy_with_logits(
                    _gl, _sal_targets, reduction="none",
                    pos_weight=_pw_tensor,
                )
                _pg = torch.sigmoid(_gl)
                _pt_g = _pg * _sal_targets + (1.0 - _pg) * (1.0 - _sal_targets)
                _focal_w_g = _alpha_t * (1.0 - _pt_g).pow(_focal_gamma)
                # v8f2: switch gate to RetinaNet /num_pos normalization.
                # The old mean reduction was tuned for Fisher targets (~10-50%
                # positive) where the gate's failure mode was predict-all-positive
                # from bias=+2.0 init. With GT targets (~2-9%) and
                # gate_bias_offset now at -0.2, the failure mode flipped to
                # predict-all-zero — same as salience head. /num_pos + pos_weight
                # correctly amplifies the rare positive gradient.
                gate_loss = (_focal_w_g * _ce_gate).sum() / _num_pos
                with torch.no_grad():
                    _gate_pred = (_gl > 0.0).long()
                    gate_pred_pos_rate = float(_gate_pred.float().mean().item())
                    gate_tp = int(((_gate_pred == 1) & (_sal_targets.long() == 1)).sum().item())
                    gate_fp = int(((_gate_pred == 1) & (_sal_targets.long() == 0)).sum().item())
                    gate_fn = int(((_gate_pred == 0) & (_sal_targets.long() == 1)).sum().item())
                    _gp = float(gate_tp / max(1, gate_tp + gate_fp))
                    _gr = float(gate_tp / max(1, gate_tp + gate_fn))
                    gate_f1 = float(2 * _gp * _gr / max(1e-12, _gp + _gr))
            else:
                salience = outputs.logits.new_zeros(())
                gate_loss = outputs.logits.new_zeros(())
                gate_pred_pos_rate = 0.0
                gate_f1 = 0.0
            total = (float(self.config.ce_weight) * ce) + (float(self.config.salience_weight) * salience)
            total = total + (float(self.config.gate_weight) * gate_loss)
            # MAE reconstruction loss: MSE on explicitly masked entries
            mae_entry_mask = batch.mae_mask > 0.0
            support_entry_valid = batch.support_valid.unsqueeze(-1) & batch.feature_valid_mask.unsqueeze(1)
            mae_valid = mae_entry_mask & support_entry_valid
            if torch.any(mae_valid):
                mae_loss = F.mse_loss(outputs.mae_reconstruction[mae_valid], batch.mae_reconstruction_target[mae_valid])
            else:
                mae_loss = outputs.logits.new_zeros(())
            total = total + (float(self.config.mae_weight) * mae_loss)
            jepa_loss = F.mse_loss(outputs.latent_prediction, outputs.latent_target)
            total = total + (float(self.config.jepa_weight) * jepa_loss)
            sigreg_loss = self._sigreg_loss(outputs.sigreg_samples)
            total = total + (float(self.config.sigreg_weight) * sigreg_loss)
            # Chaotic/periodic detection BCE loss (skipped when head disabled)
            if self.config.enable_chaotic_head:
                chaotic_loss = F.binary_cross_entropy_with_logits(outputs.chaotic_logit, batch.chaotic_targets)
            else:
                chaotic_loss = outputs.logits.new_zeros(())
            if self.config.enable_chaotic_head:
                total = total + (float(self.config.chaotic_weight) * chaotic_loss)
            predictions = logits.argmax(dim=-1)
            valid_query_count = int(valid_queries.sum().item())
            correct_query_count = (
                int(((predictions == batch.query_labels) & valid_queries).sum().item())
                if valid_query_count > 0
                else 0
            )
            if torch.any(feature_mask):
                sal_scores = outputs.importance_logits[feature_mask].detach().float()
                sal_pred = (sal_scores > 0.0).long()
                # Primary metrics use whichever target the loss trains against.
                sal_target = _sal_supervision[feature_mask].long()
                sal_correct = int((sal_pred == sal_target).sum().item())
                sal_total = int(feature_mask.sum().item())
                # Sparse-target metrics: accuracy is misleading because the
                # majority class dominates. Precision/recall/F1 measure the
                # model's ability to actually recover the active features.
                tp = int(((sal_pred == 1) & (sal_target == 1)).sum().item())
                fp = int(((sal_pred == 1) & (sal_target == 0)).sum().item())
                fn = int(((sal_pred == 0) & (sal_target == 1)).sum().item())
                sal_pos_rate = float(sal_target.sum().item()) / max(1, sal_total)
                sal_pred_pos_rate = float(sal_pred.sum().item()) / max(1, sal_total)
                # v8d: auxiliary agreement metric vs synthetic ground truth
                # (active_feature_targets). Lets us see how often the per-episode
                # Fisher target matches the world-spec active features.
                sal_target_orig = batch.active_feature_targets[feature_mask].long()
                target_agree_rate = float(((sal_target == sal_target_orig).float().mean().item()))
                orig_pos_rate = float(sal_target_orig.sum().item()) / max(1, sal_total)
                # Salience F1 against synthetic ground truth (independent eval)
                tp_o = int(((sal_pred == 1) & (sal_target_orig == 1)).sum().item())
                fp_o = int(((sal_pred == 1) & (sal_target_orig == 0)).sum().item())
                fn_o = int(((sal_pred == 0) & (sal_target_orig == 1)).sum().item())
                sal_p_orig = float(tp_o / max(1, tp_o + fp_o))
                sal_r_orig = float(tp_o / max(1, tp_o + fn_o))
                sal_f1_orig = float(2 * sal_p_orig * sal_r_orig / max(1e-12, sal_p_orig + sal_r_orig))
                # AUROC against GT targets (ranking quality, threshold-free)
                _n_pos_gt = float(sal_target_orig.sum().item())
                _n_neg_gt = float(sal_target_orig.numel() - _n_pos_gt)
                if _n_pos_gt > 0 and _n_neg_gt > 0:
                    _sorted_idx = torch.argsort(sal_scores, descending=True)
                    _sorted_gt = sal_target_orig.float()[_sorted_idx]
                    _tpr_curve = torch.cumsum(_sorted_gt, dim=0) / _n_pos_gt
                    _fpr_curve = torch.cumsum(1.0 - _sorted_gt, dim=0) / _n_neg_gt
                    _fpr_curve = torch.cat([sal_scores.new_zeros(1), _fpr_curve])
                    _tpr_curve = torch.cat([sal_scores.new_zeros(1), _tpr_curve])
                    sal_auroc = float(torch.trapezoid(_tpr_curve, _fpr_curve).item())
                else:
                    sal_auroc = 0.5
            else:
                sal_correct = 0
                sal_total = 0
                tp = fp = fn = 0
                sal_pos_rate = 0.0
                sal_pred_pos_rate = 0.0
                target_agree_rate = 0.0
                orig_pos_rate = 0.0
                sal_f1_orig = 0.0
                sal_auroc = 0.5
            components["ce"] = float(ce.detach().cpu().item())
            components["salience"] = float(salience.detach().cpu().item())
            components["mae_reconstruction"] = float(mae_loss.detach().cpu().item())
            components["jepa"] = float(jepa_loss.detach().cpu().item())
            components["sigreg"] = float(sigreg_loss.detach().cpu().item())
            components["chaotic"] = float(chaotic_loss.detach().cpu().item())
            # Chaotic detection accuracy + harmonic gate stats
            with torch.no_grad():
                if self.config.enable_chaotic_head:
                    chaotic_pred = (outputs.chaotic_logit.detach() > 0.0).float()
                    chaotic_target = batch.chaotic_targets
                    chaotic_correct = int((chaotic_pred == chaotic_target).sum().item())
                    chaotic_total = int(chaotic_target.numel())
                    chaotic_pos_rate = float(chaotic_target.sum().item()) / max(1, chaotic_total)
                    chaotic_pred_pos_rate = float(chaotic_pred.sum().item()) / max(1, chaotic_total)
                else:
                    chaotic_correct = 0
                    chaotic_total = 0
                    chaotic_pos_rate = 0.0
                    chaotic_pred_pos_rate = 0.0
            components["chaotic_acc"] = float(chaotic_correct / max(1, chaotic_total))
            components["chaotic_pos_rate"] = chaotic_pos_rate
            components["chaotic_pred_pos_rate"] = chaotic_pred_pos_rate
            components["harmonic_gate"] = outputs.harmonic_gate_value
            # Salience gate statistics (for monitoring gate collapse)
            if hasattr(self, "_last_sal_gate") and self._last_sal_gate is not None:
                _sg = self._last_sal_gate
                _sgm = self._last_sal_gate_mask
                if torch.any(_sgm):
                    _sg_valid = _sg[_sgm]
                    components["sal_gate_mean"] = float(_sg_valid.mean().item())
                    components["sal_gate_min"] = float(_sg_valid.min().item())
                else:
                    components["sal_gate_mean"] = 0.0
                    components["sal_gate_min"] = 0.0
            if self.config.harmonic_fusion:
                components["h_gate_raw"] = float(self.harmonic_fusion_gate.item())
            components["total"] = float(total.detach().cpu().item())
            components["query_valid_count"] = valid_query_count
            components["query_correct_count"] = correct_query_count
            components["query_accuracy"] = (
                float(correct_query_count / valid_query_count) if valid_query_count > 0 else 0.0
            )
            components["salience_valid_count"] = sal_total
            components["salience_correct_count"] = sal_correct
            components["salience_accuracy"] = float(sal_correct / sal_total) if sal_total > 0 else 0.0
            sal_precision = float(tp / max(1, tp + fp))
            sal_recall = float(tp / max(1, tp + fn))
            sal_f1 = float(2 * sal_precision * sal_recall / max(1e-12, sal_precision + sal_recall))
            components["salience_tp"] = tp
            components["salience_fp"] = fp
            components["salience_fn"] = fn
            components["salience_precision"] = sal_precision
            components["salience_recall"] = sal_recall
            components["salience_f1"] = sal_f1
            components["salience_pos_rate"] = sal_pos_rate
            components["salience_pred_pos_rate"] = sal_pred_pos_rate
            # v8c: gate-side metrics (untied feature_gate_head)
            components["gate_loss"] = float(gate_loss.detach().cpu().item())
            components["gate_pred_pos_rate"] = gate_pred_pos_rate
            components["gate_f1"] = gate_f1
            # v8d: per-episode vs synthetic-ground-truth metrics
            components["target_agree_rate"] = target_agree_rate
            components["orig_pos_rate"] = orig_pos_rate
            components["salience_f1_orig"] = sal_f1_orig
            components["salience_auroc"] = sal_auroc
            # v8c: random-guess anchors for sanity-checking CE/q_acc per step.
            # K = mean active class count across the batch (clamped >=2).
            _cc = batch.class_counts.float().clamp(min=2.0)
            _mean_K = float(_cc.mean().item())
            import math as _math_log
            components["num_classes_mean"] = _mean_K
            components["random_ce"] = float(_math_log.log(max(2.0, _mean_K)))
            components["random_q_acc"] = float(1.0 / max(2.0, _mean_K))
            return total, components

        def activation_summary(self) -> str:
            return "silu+sdpa+siren+jepa+ema"


    class TabenticsDiakrinoTrainer:
        def __init__(self, config: TabenticsDiakrinoConfig | None = None, trainer_config: TabenticsDiakrinoTrainerConfig | None = None) -> None:
            self.config = config or TabenticsDiakrinoConfig()
            self.trainer_config = trainer_config or TabenticsDiakrinoTrainerConfig()
            self.model = TabenticsDiakrino(self.config)
            # Propagate trainer-level flag to model so compute_losses can access it.
            self.model._use_gt_salience_target = bool(self.trainer_config.use_gt_salience_target)
            self.history: list[JsonDict] = []
            self.last_components: JsonDict = {}

        def _resolve_device(self) -> torch.device:
            requested = str(self.trainer_config.device).strip().lower()
            if requested == "auto":
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.device(self.trainer_config.device)

        @staticmethod
        def _ddp_env() -> tuple[bool, int, int, int]:
            """Detect torchrun DDP env. Returns (enabled, rank, local_rank, world_size)."""
            import os
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
                return (
                    True,
                    int(os.environ["RANK"]),
                    int(os.environ.get("LOCAL_RANK", "0")),
                    int(os.environ["WORLD_SIZE"]),
                )
            return (False, 0, 0, 1)

        @staticmethod
        def _maybe_init_pg(local_rank: int) -> None:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

        def _ddp_reduce_scalars(
            self,
            values: Sequence[float | int],
            *,
            device: torch.device,
            op: str = "sum",
        ) -> tuple[float, ...]:
            tensor = torch.as_tensor(list(values), dtype=torch.float64, device=device)
            if getattr(self, "_ddp", False):
                import torch.distributed as dist
                if dist.is_initialized():
                    reduce_op = dist.ReduceOp.MAX if str(op).lower() == "max" else dist.ReduceOp.SUM
                    dist.all_reduce(tensor, op=reduce_op)
            return tuple(float(value) for value in tensor.detach().cpu().tolist())

        def _ddp_average_gradients(self) -> None:
            if not getattr(self, "_ddp", False):
                return
            import torch.distributed as dist
            if not dist.is_initialized():
                return
            params = [p for p in self.model.parameters() if p.requires_grad]
            if not params:
                return
            flags = torch.tensor(
                [1.0 if p.grad is not None else 0.0 for p in params],
                dtype=torch.float32,
                device=params[0].device,
            )
            dist.all_reduce(flags, op=dist.ReduceOp.SUM)
            world = float(self._world_size)
            active_grads_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
            for p, flag in zip(params, flags):
                if float(flag.item()) <= 0.0:
                    continue
                if p.grad is None:
                    p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
                active_grads_by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
            for grads in active_grads_by_dtype.values():
                if len(grads) == 1:
                    dist.all_reduce(grads[0], op=dist.ReduceOp.SUM)
                    grads[0].div_(world)
                    continue
                flat = torch.cat([grad.reshape(-1) for grad in grads], dim=0)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(world)
                offset = 0
                for grad in grads:
                    numel = grad.numel()
                    grad.copy_(flat[offset : offset + numel].view_as(grad))
                    offset += numel

        def fit(
            self,
            train_dataset: Any,
            validation_dataset: Any | None = None,
            checkpoint_dir: str | None = None,
            checkpoint_keep: int = 2,
            resume_from: str | None = None,
        ) -> TabenticsDiakrino:
            ddp_enabled, rank, local_rank, world_size = self._ddp_env()
            if ddp_enabled:
                self._maybe_init_pg(local_rank)
                device = torch.device(f"cuda:{local_rank}")
                # Per-rank seed for independent episode sampling
                torch.manual_seed(int(torch.initial_seed()) + rank)
            else:
                device = self._resolve_device()
            self.model.to(device)
            self._is_main = (rank == 0)
            self._rank = rank
            self._local_rank = local_rank
            self._world_size = world_size
            self._ddp = ddp_enabled
            if ddp_enabled:
                # Manual gradient allreduce strategy (no DDP wrapper).
                #
                # Rationale: the trainer drives the model through a sequence of
                # method calls — embed_and_init_state -> hrm_step -> decode_outputs
                # -> compute_losses — and never invokes a single `forward()`.
                # DDP's autograd reducer is installed by the wrapper's forward,
                # so wrapping the model would NOT actually synchronize grads here.
                # Instead, each rank computes local grads independently, and we
                # average them across ranks before optimizer.step() in each of the
                # M deep-supervision segments. EMA modules need no special
                # handling: each rank's optimizer.step() consumes the same
                # averaged grads, so student parameters stay in lock-step, and
                # `_update_ema()` (driven only by local student state) yields
                # identical EMA state on every rank.
                #
                # Broadcast initial parameters from rank 0 so all ranks start
                # from the same weights (independent of any per-rank random init).
                import torch.distributed as dist
                for p in self.model.parameters():
                    dist.broadcast(p.data, src=0)
                for b in self.model.buffers():
                    dist.broadcast(b.data, src=0)
                if self._is_main:
                    logger.info(
                        "Tabnetics Diakrino DDP enabled: world_size=%d backend=nccl data_sharding=rank",
                        world_size,
                    )
            # Fix 8+9: parameter groups — separate embedding LR, exclude RMSNorm from WD
            base_lr = float(self.trainer_config.learning_rate)
            base_wd = float(self.trainer_config.weight_decay)
            emb_mult = float(self.trainer_config.embedding_lr_multiplier)
            embedding_modules = {self.model.tokenizer, self.model.label_encoder, self.model.feature_stats_encoder}
            if hasattr(self.model, 'meta_encoder') and self.model.meta_encoder is not None:
                embedding_modules.add(self.model.meta_encoder)
            embedding_ids = {id(p) for m in embedding_modules for p in m.parameters()}
            no_wd_ids: set[int] = set()
            for mod in self.model.modules():
                if isinstance(mod, nn.RMSNorm):
                    for p in mod.parameters():
                        no_wd_ids.add(id(p))
            param_groups = []
            seen: set[int] = set()
            for p in self.model.parameters():
                pid = id(p)
                if pid in seen or not p.requires_grad:
                    continue
                seen.add(pid)
                lr = base_lr * emb_mult if pid in embedding_ids else base_lr
                wd = 0.0 if pid in no_wd_ids else base_wd
                param_groups.append({"params": [p], "lr": lr, "weight_decay": wd})
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=base_wd,
            )
            total_steps = int(self.trainer_config.epochs) * int(self.trainer_config.steps_per_epoch)
            warmup_steps = min(int(self.trainer_config.warmup_steps), max(1, total_steps // 2))
            lr_min_frac = float(self.trainer_config.lr_min_fraction)

            # Fix 4: support both constant and cosine LR schedules
            schedule = str(self.trainer_config.lr_schedule).lower()

            def _lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return max(lr_min_frac, current_step / max(1, warmup_steps))
                if schedule == "constant":
                    return 1.0
                progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
                return lr_min_frac + 0.5 * (1.0 - lr_min_frac) * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
            _resume_epoch = 0
            _resume_step = 0
            if resume_from is not None:
                import pathlib as _pathlib
                _ckpt_path = _pathlib.Path(resume_from)
                _ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
                # v8c: always allow strict=False so missing/extra heads (chaotic, gate) load cleanly.
                _strict = False
                # v8d: filter out checkpoint keys whose shapes no longer match
                # the current model (e.g. feature_stats_encoder/harmonic_branch
                # input dim changed 5→10 when class-conditional stats were added).
                _model_state = self.model.state_dict()
                _src_state = _ckpt["model_state_dict"]
                _filtered_state = {}
                _shape_mismatch = []
                for _k, _v in _src_state.items():
                    if _k in _model_state and tuple(_model_state[_k].shape) != tuple(_v.shape):
                        _shape_mismatch.append(
                            f"{_k}: ckpt {tuple(_v.shape)} vs model {tuple(_model_state[_k].shape)}"
                        )
                        continue
                    _filtered_state[_k] = _v
                if self._is_main and _shape_mismatch:
                    logger.info(
                        "Checkpoint load: dropping %d keys with shape mismatch: %s",
                        len(_shape_mismatch), _shape_mismatch,
                    )
                _load_result = self.model.load_state_dict(_filtered_state, strict=_strict)
                if self._is_main and hasattr(_load_result, "unexpected_keys") and _load_result.unexpected_keys:
                    logger.info("Checkpoint load (strict=False): dropped keys: %s", _load_result.unexpected_keys)
                # v8d Stage 2: gate is tied to salience_head with a learned
                # additive scalar `gate_bias_offset`. If the checkpoint
                # contains it (v8d+), the standard state_dict load above has
                # already restored it. If it does NOT contain it (legacy v8c
                # or earlier), the trained `salience_head.bias` already
                # encodes the gate calibration on its own — adding the +2.0
                # init would systematically over-open the gate. Force the
                # offset to 0.0 in that case.
                _missing_keys = set(getattr(_load_result, "missing_keys", []) or [])
                if "gate_bias_offset" in _missing_keys:
                    with torch.no_grad():
                        self.model.gate_bias_offset.data.zero_()
                    if self._is_main:
                        logger.info(
                            "v8d Stage 2: gate_bias_offset missing in checkpoint -> set to 0.0 "
                            "(preserves trained salience_head.bias as gate calibration)"
                        )
                if self._is_main:
                    _go = float(self.model.gate_bias_offset.detach().cpu().item())
                    logger.info("v8d Stage 2: tied gate active (gate_bias_offset=%.3f)", _go)
                # v8d: optimizer state is keyed by param-group index/order, so
                # any param whose shape changed has stale Adam moments
                # (exp_avg / exp_avg_sq sized for the OLD shape). Skip the
                # optimizer-state load entirely when shape mismatches were
                # detected, so we start with fresh moments matched to the
                # current parameter shapes. This is safer than trying to patch
                # individual slot tensors.
                if "optimizer_state_dict" in _ckpt:
                    if _shape_mismatch:
                        if self._is_main:
                            logger.warning(
                                "Skipping optimizer state load: %d shape mismatches detected; "
                                "stale Adam moments would corrupt updates for resized params.",
                                len(_shape_mismatch),
                            )
                    else:
                        try:
                            optimizer.load_state_dict(_ckpt["optimizer_state_dict"])
                        except ValueError as _e:
                            if self._is_main:
                                logger.warning(
                                    "Skipping optimizer state load (param group mismatch — likely resuming from warmup checkpoint): %s", _e
                                )
                if "scheduler_state_dict" in _ckpt:
                    try:
                        scheduler.load_state_dict(_ckpt["scheduler_state_dict"])
                    except (ValueError, KeyError) as _e:
                        if self._is_main:
                            logger.warning("Skipping scheduler state load: %s", _e)
                _resume_epoch = int(_ckpt.get("epoch", 1)) - 1
                _resume_step = int(_ckpt.get("step", 0))
                if self._is_main:
                    logger.info(
                        "Resumed from %s (epoch=%d, step=%d)",
                        resume_from, _resume_epoch + 1, _resume_step,
                    )
                del _ckpt
            # ─── SIREN warmup mode: freeze backbone, setup fresh optimizer ───
            _siren_warmup = bool(self.trainer_config.siren_warmup)
            if _siren_warmup:
                import math as _math
                # Freeze everything first
                for p in self.model.parameters():
                    p.requires_grad_(False)
                # Unfreeze SIREN modules: harmonic_branch + harmonic_fusion_proj
                _siren_params = []
                for p in self.model.harmonic_branch.parameters():
                    p.requires_grad_(True)
                    _siren_params.append(p)
                if self.config.harmonic_fusion:
                    for p in self.model.harmonic_fusion_proj.parameters():
                        p.requires_grad_(True)
                        _siren_params.append(p)
                # Set gate to target value (frozen initially)
                _gate_target = float(self.trainer_config.siren_warmup_gate_value)
                _gate_raw = _math.atanh(max(-0.99, min(0.99, _gate_target)))
                self.model.harmonic_fusion_gate.data.fill_(_gate_raw)
                self.model.harmonic_fusion_gate.requires_grad_(False)
                _gate_freeze_steps = int(self.trainer_config.siren_warmup_gate_freeze_steps)
                # Build fresh optimizer for SIREN params only
                _siren_lr = float(self.trainer_config.siren_warmup_siren_lr)
                _fusion_lr = float(self.trainer_config.learning_rate)
                optimizer = torch.optim.AdamW(
                    [
                        {"params": list(self.model.harmonic_branch.parameters()), "lr": _siren_lr},
                        {"params": list(self.model.harmonic_fusion_proj.parameters()), "lr": _fusion_lr},
                    ],
                    lr=_siren_lr,
                    weight_decay=float(self.trainer_config.weight_decay),
                )
                # Constant LR for short warmup (no schedule)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                _n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                _n_total = sum(p.numel() for p in self.model.parameters())
                if self._is_main:
                    logger.info(
                        "SIREN warmup: %d/%d params trainable, gate=%.4f (frozen for %d steps), siren_lr=%.1e",
                        _n_trainable, _n_total, _gate_target, _gate_freeze_steps, _siren_lr,
                    )
                # ── Salience head: re-init + unfreeze (tied gate) ──
                import torch.nn.init as _nn_init
                _nn_init.xavier_uniform_(self.model.salience_head.weight)
                _nn_init.zeros_(self.model.salience_head.bias)
                for p in self.model.salience_head.parameters():
                    p.requires_grad_(True)
                    _siren_params.append(p)
                optimizer.add_param_group({
                    "params": list(self.model.salience_head.parameters()),
                    "lr": float(self.trainer_config.learning_rate),
                    "weight_decay": float(self.trainer_config.weight_decay),
                })
                # v8d Stage 2: gate offset (tied gate). Init to +2.0 so the
                # gate starts near-identity even when salience logits are 0.
                self.model.gate_bias_offset.data.fill_(2.0)
                self.model.gate_bias_offset.requires_grad_(True)
                _siren_params.append(self.model.gate_bias_offset)
                optimizer.add_param_group({
                    "params": [self.model.gate_bias_offset],
                    "lr": float(self.trainer_config.learning_rate),
                    "weight_decay": 0.0,
                })
                if self._is_main:
                    logger.info("SIREN warmup: salience head re-init (tied gate, offset=+2.0)")
                # Override loss weights: CE + salience (from CLI), no chaotic
                from dataclasses import replace as _dc_replace
                self.config = _dc_replace(
                    self.config,
                    chaotic_weight=0.0,
                    enable_chaotic_head=False,
                )
                # Update model's config reference
                object.__setattr__(self.model, 'config', self.config)
                # Reset resume counters: warmup is a fresh run
                _resume_epoch = 0
                _resume_step = 0
            else:
                _siren_params = []
                _gate_freeze_steps = 0
            _siren_warmup = _siren_warmup  # ensure variable exists for training loop
            train_loader = build_episode_loader(
                train_dataset,
                batch_size=int(self.trainer_config.batch_size),
                shuffle=bool(self.trainer_config.shuffle),
                seed=int(self.trainer_config.loader_seed),
                rank=rank if ddp_enabled else 0,
                world_size=world_size if ddp_enabled else 1,
                endless=True,
                prefetch_batches=int(self.trainer_config.prefetch_batches),
            )
            M = int(self.model.config.hrm_segments)
            _skip_teacher = bool(self.trainer_config.pretrain_clean)
            try:
                for epoch in range(int(self.trainer_config.epochs)):
                    if epoch < _resume_epoch:
                        continue
                    start = time.time()
                    self.model.train()
                    epoch_losses: list[float] = []
                    epoch_valid_queries = 0
                    epoch_correct_queries = 0
                    epoch_sal_valid = 0
                    epoch_sal_correct = 0
                    for step in range(int(self.trainer_config.steps_per_epoch)):
                        if epoch == _resume_epoch and step < _resume_step:
                            continue
                        batch = next(train_loader)

                        prepared = prepare_tabentics_diakrino_batch(batch, device=device, config=self.config)
                        # Pretrain clean: feed un-corrupted data, skip teacher
                        if _skip_teacher:
                            from dataclasses import replace as _dc_replace
                            _replace_kwargs: dict = dict(
                                support=prepared.support_clean,
                                support_mask=torch.zeros_like(prepared.support_mask),
                                mae_mask=torch.zeros_like(prepared.mae_mask),
                            )
                            # When using GT salience targets, no need to
                            # recompute Fisher — active_feature_targets are
                            # already correct regardless of corruption.
                            # Only recompute per_episode_target (Fisher) when
                            # it is the actual supervision signal.
                            if not self.trainer_config.use_gt_salience_target:
                                with torch.no_grad():
                                    _fisher_clean = _compute_fisher_signal(
                                        prepared.support_clean,
                                        support_mask=torch.zeros_like(prepared.support_mask),
                                        support_valid=prepared.support_valid,
                                        support_labels=prepared.support_labels,
                                        num_classes_max=int(self.config.max_classes),
                                    )
                                    _per_episode_clean = (_fisher_clean > 0.05).to(
                                        dtype=torch.float32
                                    ) * prepared.feature_valid_mask.to(dtype=torch.float32)
                                _replace_kwargs["per_episode_target"] = _per_episode_clean
                            prepared = _dc_replace(prepared, **_replace_kwargs)
                        optimizer.zero_grad(set_to_none=True)

                        # Deep supervision: M segments × N cycles, with
                        # decode→loss→backward→step at every cycle boundary.
                        # Each cycle is T inner L-block steps + one H-module
                        # update. State is detached between cycles so memory
                        # cost is O(1 cycle) regardless of total depth.
                        N_max = int(self.model.config.hrm_outer_cycles)
                        if self.model.config.hrm_random_cycles:
                            # Deterministic per-step seed so all DDP ranks
                            # draw the same N (avoids NCCL all-reduce mismatch).
                            import random as _rng
                            _step_rng = _rng.Random(epoch * 100_000 + step)
                            N = _step_rng.randint(1, N_max)
                        else:
                            N = N_max
                        z_L, z_H, ctx = self.model.embed_and_init_state(prepared)
                        # Record sampled cycle/segment counts for step logging
                        _hrm_n_logged = N
                        _hrm_m_logged = M
                        for seg in range(M):
                            for cyc in range(N):
                                # Inject embedding on the first cycle of each segment
                                z_L, z_H = self.model.hrm_cycle(
                                    z_L, z_H, ctx, inject_embed=(cyc == 0),
                                )
                                outputs = self.model.decode_outputs(z_L, z_H, prepared, ctx, skip_teacher=_skip_teacher)
                                loss, components = self.model.compute_losses(outputs, prepared)
                                loss.backward()
                                self._ddp_average_gradients()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.trainer_config.grad_clip_norm))
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                                # SIREN warmup: unfreeze gate after N steps
                                if _siren_warmup and not self.model.harmonic_fusion_gate.requires_grad:
                                    _warmup_step = step - _resume_step
                                    if _warmup_step >= _gate_freeze_steps:
                                        self.model.harmonic_fusion_gate.requires_grad_(True)
                                        optimizer.add_param_group({"params": [self.model.harmonic_fusion_gate], "lr": float(self.trainer_config.learning_rate)})
                                        if self._is_main:
                                            logger.info("SIREN warmup: gate unfrozen at step %d", step)
                                # Detach state so next cycle doesn't reuse freed graph
                                z_L = z_L.detach()
                                z_H = z_H.detach()
                                if "x_embed" in ctx:
                                    ctx["x_embed"] = ctx["x_embed"].detach()
                                if ctx.get("feature_coords") is not None:
                                    ctx["feature_coords"] = ctx["feature_coords"].detach()

                        # Scheduler steps once per batch (not per segment)
                        scheduler.step()
                        self.model._update_ema()
                        # SIREN warmup: log gradient norm for SIREN params
                        if _siren_warmup and _siren_params:
                            _sg_norm = 0.0
                            for _sp in _siren_params:
                                if _sp.grad is not None:
                                    _sg_norm += float(_sp.grad.data.norm(2).item() ** 2)
                            components["siren_grad_norm"] = _sg_norm ** 0.5
                        epoch_losses.append(float(loss.detach().cpu().item()))
                        self.last_components = components
                        epoch_valid_queries += int(components.get("query_valid_count", 0))
                        epoch_correct_queries += int(components.get("query_correct_count", 0))
                        epoch_sal_valid += int(components.get("salience_valid_count", 0))
                        epoch_sal_correct += int(components.get("salience_correct_count", 0))
                        components["hrm_n"] = _hrm_n_logged
                        components["hrm_m"] = _hrm_m_logged
                        step_log_interval = max(0, int(self.trainer_config.step_log_interval))
                        if self._is_main and step_log_interval > 0 and ((step + 1) % step_log_interval == 0 or (step + 1) == int(self.trainer_config.steps_per_epoch)):
                            logger.info(
                                "Tabnetics Diakrino epoch %d step %d/%d loss=%.4f ce=%.4f sal=%.4f jepa=%.4f sigreg=%.4f chaotic=%.4f q_acc=%.4f sal_f1=%.4f sal_p=%.4f sal_r=%.4f sal_pos=%.3f sal_pred_pos=%.3f ch_acc=%.3f ch_pos=%.3f ch_ppos=%.3f h_gate=%.4f h_gate_raw=%.4f siren_gn=%.4f sal_gm=%.4f sal_gmin=%.4f K=%.1f rand_ce=%.4f rand_q=%.4f gate_loss=%.4f gate_f1=%.4f gate_ppos=%.3f tgt_agree=%.3f orig_pos=%.3f sal_f1_o=%.4f sal_auroc=%.4f hrm_n=%d hrm_m=%d",
                                epoch + 1,
                                step + 1,
                                int(self.trainer_config.steps_per_epoch),
                                float(components["total"]),
                                float(components["ce"]),
                                float(components["salience"]),
                                float(components.get("jepa", 0.0)),
                                float(components.get("sigreg", 0.0)),
                                float(components.get("chaotic", 0.0)),
                                float(components.get("query_accuracy", 0.0)),
                                float(components.get("salience_f1", 0.0)),
                                float(components.get("salience_precision", 0.0)),
                                float(components.get("salience_recall", 0.0)),
                                float(components.get("salience_pos_rate", 0.0)),
                                float(components.get("salience_pred_pos_rate", 0.0)),
                                float(components.get("chaotic_acc", 0.0)),
                                float(components.get("chaotic_pos_rate", 0.0)),
                                float(components.get("chaotic_pred_pos_rate", 0.0)),
                                float(components.get("harmonic_gate", 0.0)),
                                float(components.get("h_gate_raw", 0.0)),
                                float(components.get("siren_grad_norm", 0.0)),
                                float(components.get("sal_gate_mean", 0.0)),
                                float(components.get("sal_gate_min", 0.0)),
                                float(components.get("num_classes_mean", 0.0)),
                                float(components.get("random_ce", 0.0)),
                                float(components.get("random_q_acc", 0.0)),
                                float(components.get("gate_loss", 0.0)),
                                float(components.get("gate_f1", 0.0)),
                                float(components.get("gate_pred_pos_rate", 0.0)),
                                float(components.get("target_agree_rate", 0.0)),
                                float(components.get("orig_pos_rate", 0.0)),
                                float(components.get("salience_f1_orig", 0.0)),
                                float(components.get("salience_auroc", 0.5)),
                                int(components.get("hrm_n", 0)),
                                int(components.get("hrm_m", 0)),
                            )
                        # Mid-epoch step checkpoint (rank 0 only). The v8f2 run also
                        # keeps an explicit 450-step checkpoint for diagnosis.
                        step_ckpt_interval = int(self.trainer_config.step_checkpoint_interval)
                        step_index = step + 1
                        extra_step_checkpoints = {450}
                        save_step_checkpoint = (
                            self._is_main
                            and step_ckpt_interval > 0
                            and checkpoint_dir is not None
                            and (
                                step_index % step_ckpt_interval == 0
                                or step_index in extra_step_checkpoints
                            )
                        )
                        if save_step_checkpoint:
                            try:
                                import pathlib
                                ckpt_path = pathlib.Path(checkpoint_dir)
                                ckpt_path.mkdir(parents=True, exist_ok=True)
                                step_ckpt_file = ckpt_path / f"checkpoint_e{epoch + 1}_s{step_index}.pt"
                                torch.save(
                                    {
                                        "epoch": epoch + 1,
                                        "step": step_index,
                                        "model_state_dict": self.model.state_dict(),
                                        "optimizer_state_dict": optimizer.state_dict(),
                                        "scheduler_state_dict": scheduler.state_dict(),
                                        "history": list(self.history),
                                        "model_config": self.config.__dict__,
                                        "trainer_config": self.trainer_config.__dict__,
                                    },
                                    step_ckpt_file,
                                )
                                # Remove older non-protected step checkpoints.
                                protected_step_suffixes = ("_s450.pt",)
                                for old in sorted(ckpt_path.glob("checkpoint_e*_s*.pt")):
                                    if old != step_ckpt_file and not old.name.endswith(protected_step_suffixes):
                                        old.unlink(missing_ok=True)
                                logger.info("Step checkpoint saved: %s", step_ckpt_file)
                            except Exception as step_ckpt_exc:
                                logger.warning("Failed to save step checkpoint: %s", step_ckpt_exc)
                        # v8e fragmentation mitigation: periodic empty_cache.
                        # HRM's random_cycles (N=1..4) creates per-step shape
                        # variance that fragments the CUDA allocator even
                        # with expandable_segments=True. Releasing cached
                        # blocks every 25 steps keeps reserved memory bounded
                        # without measurable throughput cost.
                        if torch.cuda.is_available() and (step + 1) % 25 == 0:
                            torch.cuda.empty_cache()
                    elapsed_seconds = float(time.time() - start)
                    (
                        train_loss_sum,
                        train_loss_count,
                        train_correct_queries,
                        train_valid_queries,
                        train_salience_correct,
                        train_salience_valid,
                    ) = self._ddp_reduce_scalars(
                        [
                            float(sum(epoch_losses)),
                            float(len(epoch_losses)),
                            float(epoch_correct_queries),
                            float(epoch_valid_queries),
                            float(epoch_sal_correct),
                            float(epoch_sal_valid),
                        ],
                        device=device,
                    )
                    (elapsed_seconds_max,) = self._ddp_reduce_scalars(
                        [elapsed_seconds],
                        device=device,
                        op="max",
                    )
                    record: JsonDict = {
                        "epoch": epoch + 1,
                        "train_loss": float(train_loss_sum / max(1.0, train_loss_count)),
                        "elapsed_seconds": float(elapsed_seconds_max),
                        "train_query_accuracy": (
                            float(train_correct_queries / train_valid_queries) if train_valid_queries > 0 else None
                        ),
                        "train_salience_accuracy": (
                            float(train_salience_correct / train_salience_valid) if train_salience_valid > 0 else None
                        ),
                    }
                    if validation_dataset is not None:
                        validation_metrics = self.evaluate_metrics(validation_dataset, skip_teacher=_skip_teacher)
                        record["validation_loss"] = float(validation_metrics["loss"])
                        record["validation_query_accuracy"] = validation_metrics["query_accuracy"]
                        record["validation_salience_accuracy"] = validation_metrics["salience_accuracy"]
                    self.history.append(record)
                    val_acc_str = "n/a" if record.get("validation_query_accuracy") is None else f"{float(record['validation_query_accuracy']):.4f}"
                    val_sal_str = "n/a" if record.get("validation_salience_accuracy") is None else f"{float(record['validation_salience_accuracy']):.4f}"
                    val_suffix = (
                        ""
                        if "validation_loss" not in record
                        else f" val_loss={float(record['validation_loss']):.4f} val_q_acc={val_acc_str} val_sal_acc={val_sal_str}"
                    )
                    train_sal_str = "n/a" if record["train_salience_accuracy"] is None else f"{float(record['train_salience_accuracy']):.4f}"
                    if self._is_main:
                        logger.info(
                            "Tabnetics Diakrino epoch %d complete train_loss=%.4f train_q_acc=%s train_sal_acc=%s%s elapsed=%.2fs",
                            epoch + 1,
                            float(record["train_loss"]),
                            "n/a" if record["train_query_accuracy"] is None else f"{float(record['train_query_accuracy']):.4f}",
                            train_sal_str,
                            val_suffix,
                            float(record["elapsed_seconds"]),
                        )
                    # Per-epoch checkpoint with rotation (rank 0 only)
                    if self._is_main and checkpoint_dir is not None:
                        try:
                            import pathlib
                            ckpt_path = pathlib.Path(checkpoint_dir)
                            ckpt_path.mkdir(parents=True, exist_ok=True)
                            ckpt_file = ckpt_path / f"checkpoint_epoch{epoch + 1}.pt"
                            torch.save(
                                {
                                    "epoch": epoch + 1,
                                    "model_state_dict": self.model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "scheduler_state_dict": scheduler.state_dict(),
                                    "history": list(self.history),
                                    "model_config": self.config.__dict__,
                                    "trainer_config": self.trainer_config.__dict__,
                                },
                                ckpt_file,
                            )
                            # Remove old checkpoints, keep last checkpoint_keep
                            existing = sorted(ckpt_path.glob("checkpoint_epoch*.pt"))
                            for old in existing[:-max(1, int(checkpoint_keep))]:
                                old.unlink(missing_ok=True)
                            logger.info("Checkpoint saved: %s", ckpt_file)
                        except Exception as ckpt_exc:
                            logger.warning("Failed to save epoch checkpoint: %s", ckpt_exc)
            finally:
                close_loader = getattr(train_loader, "close", None)
                if callable(close_loader):
                    close_loader()
                if self._ddp:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.barrier()
                        dist.destroy_process_group()
            return self.model

        def evaluate_metrics(self, dataset: Any, *, skip_teacher: bool = False) -> JsonDict:
            device = next(self.model.parameters()).device
            ddp_enabled = bool(getattr(self, "_ddp", False))
            loader = build_episode_loader(
                dataset,
                batch_size=int(self.trainer_config.batch_size),
                shuffle=False,
                rank=int(getattr(self, "_rank", 0)) if ddp_enabled else 0,
                world_size=int(getattr(self, "_world_size", 1)) if ddp_enabled else 1,
                endless=False,
                prefetch_batches=0,
            )
            losses: list[float] = []
            valid_queries = 0
            correct_queries = 0
            sal_valid = 0
            sal_correct = 0
            self.model.eval()
            with torch.no_grad():
                for batch_index, batch in enumerate(loader):
                    if batch_index >= int(self.trainer_config.validation_batches):
                        break
                    prepared = prepare_tabentics_diakrino_batch(batch, device=device, config=self.config)
                    if skip_teacher:
                        from dataclasses import replace as _dc_replace
                        prepared = _dc_replace(
                            prepared,
                            support=prepared.support_clean,
                            support_mask=torch.zeros_like(prepared.support_mask),
                            mae_mask=torch.zeros_like(prepared.mae_mask),
                        )
                    # Use full forward (all segments) for validation
                    z_L, z_H, ctx = self.model.embed_and_init_state(prepared)
                    M = int(self.model.config.hrm_segments)
                    for _ in range(M):
                        z_L, z_H = self.model.hrm_step(z_L, z_H, ctx)
                    outputs = self.model.decode_outputs(z_L, z_H, prepared, ctx, skip_teacher=skip_teacher)
                    loss, components = self.model.compute_losses(outputs, prepared)
                    losses.append(float(loss.detach().cpu().item()))
                    valid_queries += int(components.get("query_valid_count", 0))
                    correct_queries += int(components.get("query_correct_count", 0))
                    sal_valid += int(components.get("salience_valid_count", 0))
                    sal_correct += int(components.get("salience_correct_count", 0))
            (
                loss_sum,
                loss_count,
                correct_queries_sum,
                valid_queries_sum,
                sal_correct_sum,
                sal_valid_sum,
            ) = self._ddp_reduce_scalars(
                [
                    float(sum(losses)),
                    float(len(losses)),
                    float(correct_queries),
                    float(valid_queries),
                    float(sal_correct),
                    float(sal_valid),
                ],
                device=device,
            )
            return {
                "loss": float(loss_sum / max(1.0, loss_count)),
                "query_accuracy": float(correct_queries_sum / valid_queries_sum) if valid_queries_sum > 0 else None,
                "salience_accuracy": float(sal_correct_sum / sal_valid_sum) if sal_valid_sum > 0 else None,
            }

        def evaluate(self, dataset: Any) -> float:
            return float(self.evaluate_metrics(dataset)["loss"])

        def latest_report(self) -> TabenticsDiakrinoLossReport:
            total = float(self.last_components.get("total", 0.0))
            return TabenticsDiakrinoLossReport(total=total, components=dict(self.last_components))


else:

    class TabenticsDiakrino:  # pragma: no cover - only used when torch is absent
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Tabnetics Diakrino requires torch to be installed in the active environment.")


    class TabenticsDiakrinoTrainer:  # pragma: no cover - only used when torch is absent
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("TabenticsDiakrinoTrainer requires torch to be installed in the active environment.")


__all__ = [
    "TabenticsDiakrino",
    "TabenticsDiakrinoConfig",
    "TabenticsDiakrinoOutputs",
    "TabenticsDiakrinoLossReport",
    "TabenticsDiakrinoPreparedBatch",
    "TabenticsDiakrinoTrainer",
    "TabenticsDiakrinoTrainerConfig",
    "prepare_tabentics_diakrino_batch",
]

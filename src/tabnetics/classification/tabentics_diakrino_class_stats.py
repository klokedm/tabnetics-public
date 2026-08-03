"""Class-statistics DIAKRINO-style classifier for HDLSS experiments.

The model is intentionally feature-centric.  It consumes support-derived
per-class statistics and raw/scaled query values, then pools feature evidence
into class logits.  Population statistics may be attached as reconstruction
targets, but they are never inference inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - non-torch import environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class TabenticsDiakrinoClassStatsConfig:
    """Configuration for the experimental class-statistics classifier."""

    d_model: int = 576
    n_heads: int = 8
    feature_layers: int = 3
    class_layers: int = 2
    query_layers: int = 1
    refiner_steps: int = 3
    ffn_expansion: int = 2
    dropout: float = 0.1
    max_classes: int = 25
    max_feature_tokens: int = 2048
    class_stats_dim: int = 24
    marginal_stats_dim: int = 5
    screening_feature_dim: int = 18
    relative_feature_dim: int = 8
    population_class_reconstruction_dim: int = 0
    clip_value: float = 8.0
    eps: float = 1e-6
    label_smoothing: float = 0.02
    class_weight_min: float = 0.25
    class_weight_max: float = 4.0
    classification_weight: float = 1.0
    reconstruction_weight: float = 0.0
    mask_estimation_weight: float = 0.0
    population_class_reconstruction_weight: float = 0.0
    evidence_scale_init: float = 1.0
    class_prior_scale_init: float = 0.20


@dataclass(frozen=True)
class TabenticsDiakrinoClassStatsBatch:
    """Batch for ``TabenticsDiakrinoClassStatsClassifier``.

    ``query_mask`` is true where the query value is unavailable or deliberately
    masked for reconstruction pretraining.
    """

    query_values: torch.Tensor
    query_mask: torch.Tensor
    class_stats: torch.Tensor
    class_stats_valid: torch.Tensor
    marginal_stats: torch.Tensor
    screening_features: torch.Tensor
    feature_valid: torch.Tensor
    class_valid: torch.Tensor
    query_labels: torch.Tensor
    query_reconstruction_targets: torch.Tensor | None = None
    query_reconstruction_valid: torch.Tensor | None = None
    mask_estimation_targets: torch.Tensor | None = None
    mask_estimation_valid: torch.Tensor | None = None
    population_class_reconstruction_targets: torch.Tensor | None = None
    population_class_reconstruction_valid: torch.Tensor | None = None
    feature_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class TabenticsDiakrinoClassStatsOutputs:
    class_logits: torch.Tensor
    feature_class_evidence: torch.Tensor
    feature_class_gates: torch.Tensor
    feature_embeddings: torch.Tensor
    class_feature_embeddings: torch.Tensor
    query_feature_embeddings: torch.Tensor
    query_reconstruction_predictions: torch.Tensor | None = None
    mask_estimation_logits: torch.Tensor | None = None
    population_class_reconstruction_predictions: torch.Tensor | None = None
    feature_valid: torch.Tensor | None = None
    class_valid: torch.Tensor | None = None


def _ensure_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("tabentics_diakrino_class_stats requires torch to be installed.")


def _match_last_dim(values: torch.Tensor, target_dim: int) -> torch.Tensor:
    target = max(1, int(target_dim))
    current = int(values.shape[-1])
    if current == target:
        return values
    if current > target:
        return values[..., :target]
    return F.pad(values, (0, target - current))


def _safe_key_padding_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    padding = ~valid_mask.to(dtype=torch.bool)
    if padding.ndim != 2 or padding.shape[1] == 0:
        return padding
    all_padded = padding.all(dim=1)
    if bool(torch.any(all_padded).detach().cpu()):
        padding = padding.clone()
        padding[all_padded, 0] = False
    return padding


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights.unsqueeze(-1)).sum(dim=dim) / weights.sum(dim=dim, keepdim=True).clamp(min=float(eps))


def _valid_feature_zscore(values: torch.Tensor, valid_mask: torch.Tensor, eps: float) -> torch.Tensor:
    mask = valid_mask.to(dtype=values.dtype)
    count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (values * mask).sum(dim=1, keepdim=True) / count
    centered = (values - mean) * mask
    var = (centered * centered).sum(dim=1, keepdim=True) / count
    z = (values - mean) / torch.sqrt(var + float(eps))
    return torch.where(valid_mask, torch.clamp(z, -6.0, 6.0), torch.zeros_like(z))


def _valid_feature_rank01(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros_like(values)
    large = torch.finfo(values.dtype).max
    masked_values = torch.where(valid_mask, values, torch.full_like(values, large))
    order = torch.argsort(masked_values, dim=1, stable=True)
    positions = torch.arange(values.shape[1], device=values.device, dtype=values.dtype).unsqueeze(0)
    ranks = torch.zeros_like(values)
    ranks.scatter_(1, order, positions.expand_as(values))
    denom = (valid_mask.to(dtype=values.dtype).sum(dim=1, keepdim=True) - 1.0).clamp(min=1.0)
    return torch.where(valid_mask, ranks / denom, torch.zeros_like(values))


def compute_class_stats_screening_features(
    marginal_stats: torch.Tensor,
    class_stats: torch.Tensor,
    *,
    class_stats_valid: torch.Tensor,
    feature_valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build the same 18 cheap screening channels expected by the model.

    The first five marginal channels are interpreted as mean, std, skew, kurt,
    observed fraction.  Class-stat channels use the precompute schema defined in
    ``precompute_diakrino_class_ready_episodes.py``.
    """

    _ensure_torch()
    marginal = _match_last_dim(marginal_stats, 5)
    class_values = _match_last_dim(class_stats, 24)
    valid = feature_valid.to(dtype=torch.bool)
    class_valid = class_stats_valid.to(dtype=torch.bool) & valid.unsqueeze(-1)
    mean = marginal[..., 0]
    std = marginal[..., 1].clamp(min=0.0)
    skew = marginal[..., 2]
    kurt = marginal[..., 3]
    observed = marginal[..., 4].clamp(min=0.0, max=1.0)
    fisher = torch.where(class_valid, class_values[..., 17].clamp(min=0.0), torch.zeros_like(class_values[..., 17])).amax(dim=2)
    max_shift = torch.where(class_valid, class_values[..., 18].abs(), torch.zeros_like(class_values[..., 18])).amax(dim=2)
    mean_shift = torch.where(class_valid, class_values[..., 18].abs(), torch.zeros_like(class_values[..., 18])).sum(dim=2)
    mean_shift = mean_shift / class_valid.to(dtype=mean_shift.dtype).sum(dim=2).clamp(min=1.0)
    log_std_ratio = torch.where(class_valid, class_values[..., 19].abs(), torch.zeros_like(class_values[..., 19])).amax(dim=2)
    priors = torch.where(class_valid, class_values[..., 16].clamp(min=0.0), torch.zeros_like(class_values[..., 16]))
    priors = priors / priors.sum(dim=2, keepdim=True).clamp(min=float(eps))
    class_balance = -(priors * torch.log(priors.clamp(min=float(eps)))).sum(dim=2)
    class_balance = class_balance / math.log(max(2, int(class_values.shape[2])))
    log_std = torch.log1p(std)
    channels = [
        fisher,
        max_shift,
        mean_shift,
        log_std_ratio,
        class_balance.clamp(min=0.0, max=1.0),
        log_std,
        mean.abs(),
        observed,
        skew.abs(),
        torch.log1p(kurt.abs()),
        _valid_feature_zscore(fisher, valid, eps),
        _valid_feature_zscore(max_shift, valid, eps),
        _valid_feature_zscore(mean_shift, valid, eps),
        _valid_feature_zscore(log_std, valid, eps),
        _valid_feature_rank01(fisher, valid),
        _valid_feature_rank01(max_shift, valid),
        _valid_feature_rank01(log_std, valid),
        1.0 - observed,
    ]
    features = torch.stack(channels, dim=-1)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
            nn.RMSNorm(int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ContextLayer(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(int(d_model))
        self.attn = nn.MultiheadAttention(int(d_model), max(1, int(n_heads)), dropout=float(dropout), batch_first=True)
        self.ffn_norm = nn.RMSNorm(int(d_model))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(d_model)),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        padding = _safe_key_padding_mask(valid_mask)
        normed = self.attn_norm(tokens)
        attended, _ = self.attn(normed, normed, normed, key_padding_mask=padding, need_weights=False)
        tokens = tokens + self.dropout(attended)
        tokens = tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))
        return torch.where(valid_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))


class _ContextStack(nn.Module):
    def __init__(self, *, layers: int, d_model: int, n_heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _ContextLayer(d_model=int(d_model), n_heads=int(n_heads), hidden_dim=int(hidden_dim), dropout=float(dropout))
                for _ in range(max(0, int(layers)))
            ]
        )

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens, valid_mask)
        return tokens


class TabenticsDiakrinoClassStatsClassifier(nn.Module):
    """Direct classifier over query values and support-derived class statistics."""

    def __init__(self, config: TabenticsDiakrinoClassStatsConfig | None = None) -> None:
        _ensure_torch()
        super().__init__()
        self.config = config or TabenticsDiakrinoClassStatsConfig()
        d_model = int(self.config.d_model)
        hidden = max(d_model, int(self.config.ffn_expansion) * d_model)
        self.class_stats_encoder = _MLP(int(self.config.class_stats_dim), hidden, d_model, dropout=float(self.config.dropout))
        self.marginal_encoder = _MLP(int(self.config.marginal_stats_dim), hidden, d_model, dropout=float(self.config.dropout))
        self.screening_encoder = _MLP(int(self.config.screening_feature_dim), hidden, d_model, dropout=float(self.config.dropout))
        self.feature_context = _ContextStack(
            layers=int(self.config.feature_layers),
            d_model=d_model,
            n_heads=int(self.config.n_heads),
            hidden_dim=hidden,
            dropout=float(self.config.dropout),
        )
        self.query_value_encoder = _MLP(4, hidden, d_model, dropout=float(self.config.dropout))
        self.query_context = _ContextStack(
            layers=int(self.config.query_layers),
            d_model=d_model,
            n_heads=int(self.config.n_heads),
            hidden_dim=hidden,
            dropout=float(self.config.dropout),
        )
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.class_projection = nn.Linear(d_model, d_model, bias=False)
        self.relative_evidence = nn.Sequential(
            nn.Linear(int(self.config.relative_feature_dim), max(16, d_model // 4)),
            nn.SiLU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(max(16, d_model // 4), 1),
        )
        self.gate_head = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, 1))
        self.class_hidden_projection = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, d_model))
        self.query_global_projection = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, d_model))
        self.class_context = _ContextStack(
            layers=int(self.config.class_layers),
            d_model=d_model,
            n_heads=int(self.config.n_heads),
            hidden_dim=hidden,
            dropout=float(self.config.dropout),
        )
        self.class_logit_head = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, 1))
        self.query_reconstruction_head = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, 1))
        self.mask_estimation_head = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, 1))
        pop_dim = max(0, int(self.config.population_class_reconstruction_dim))
        if pop_dim > 0:
            self.population_class_reconstruction_head: nn.Module | None = nn.Sequential(
                nn.RMSNorm(d_model),
                nn.Linear(d_model, max(16, d_model // 2)),
                nn.SiLU(),
                nn.Linear(max(16, d_model // 2), pop_dim),
            )
        else:
            self.population_class_reconstruction_head = None
        self.evidence_scale = nn.Parameter(torch.tensor(float(self.config.evidence_scale_init)))
        self.class_prior_scale = nn.Parameter(torch.tensor(float(self.config.class_prior_scale_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.evidence_scale.fill_(float(self.config.evidence_scale_init))
            self.class_prior_scale.fill_(float(self.config.class_prior_scale_init))

    def _relative_channels(self, batch: TabenticsDiakrinoClassStatsBatch, class_stats: torch.Tensor) -> torch.Tensor:
        eps = float(self.config.eps)
        values = torch.nan_to_num(batch.query_values, nan=0.0, posinf=0.0, neginf=0.0)
        observed = (~batch.query_mask.to(dtype=torch.bool)).to(dtype=values.dtype)
        mean = class_stats[..., 1]
        std = class_stats[..., 2].abs().clamp(min=eps)
        min_value = class_stats[..., 3]
        max_value = class_stats[..., 4]
        median = class_stats[..., 21] if class_stats.shape[-1] > 21 else mean
        iqr = class_stats[..., 22].abs().clamp(min=eps) if class_stats.shape[-1] > 22 else std
        query = values.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, class_stats.shape[2], -1)
        obs = observed.unsqueeze(-1).unsqueeze(-1).expand_as(query)
        z = torch.clamp(
            (query - mean.unsqueeze(1).unsqueeze(-1)) / std.unsqueeze(1).unsqueeze(-1),
            -float(self.config.clip_value),
            float(self.config.clip_value),
        )
        robust_z = torch.clamp(
            (query - median.unsqueeze(1).unsqueeze(-1)) / iqr.unsqueeze(1).unsqueeze(-1),
            -float(self.config.clip_value),
            float(self.config.clip_value),
        )
        in_range = (
            (query >= min_value.unsqueeze(1).unsqueeze(-1))
            & (query <= max_value.unsqueeze(1).unsqueeze(-1))
        ).to(dtype=values.dtype)
        channels = [
            torch.clamp(query, -float(self.config.clip_value), float(self.config.clip_value)),
            obs,
            z,
            z.abs(),
            robust_z,
            robust_z.abs(),
            in_range,
            1.0 - obs,
        ]
        relative = torch.cat(channels, dim=-1)
        relative = _match_last_dim(relative, int(self.config.relative_feature_dim))
        return torch.nan_to_num(relative, nan=0.0, posinf=0.0, neginf=0.0)

    def _class_prior_logits(
        self,
        class_stats: torch.Tensor,
        *,
        class_stats_valid: torch.Tensor,
        feature_valid: torch.Tensor,
        class_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = class_stats_valid.to(dtype=torch.bool) & feature_valid.unsqueeze(-1) & class_valid.unsqueeze(1)
        counts = torch.where(valid, class_stats[..., 0].clamp(min=0.0), torch.zeros_like(class_stats[..., 0]))
        denom = valid.to(dtype=counts.dtype).sum(dim=1).clamp(min=1.0)
        class_counts = counts.sum(dim=1) / denom
        class_counts = torch.where(class_valid, class_counts.clamp(min=float(self.config.eps)), torch.zeros_like(class_counts))
        log_counts = torch.log(class_counts.clamp(min=float(self.config.eps)))
        normalizer = torch.logsumexp(torch.where(class_valid, log_counts, torch.full_like(log_counts, -30.0)), dim=1, keepdim=True)
        return torch.where(class_valid, log_counts - normalizer, torch.zeros_like(log_counts))

    def forward(self, batch: TabenticsDiakrinoClassStatsBatch) -> TabenticsDiakrinoClassStatsOutputs:
        class_stats = _match_last_dim(
            torch.nan_to_num(batch.class_stats, nan=0.0, posinf=0.0, neginf=0.0),
            int(self.config.class_stats_dim),
        )
        marginal_stats = _match_last_dim(
            torch.nan_to_num(batch.marginal_stats, nan=0.0, posinf=0.0, neginf=0.0),
            int(self.config.marginal_stats_dim),
        )
        screening = _match_last_dim(
            torch.nan_to_num(batch.screening_features, nan=0.0, posinf=0.0, neginf=0.0),
            int(self.config.screening_feature_dim),
        )
        feature_valid = batch.feature_valid.to(dtype=torch.bool)
        class_valid = batch.class_valid.to(dtype=torch.bool)
        class_stats_valid = batch.class_stats_valid.to(dtype=torch.bool) & feature_valid.unsqueeze(-1) & class_valid.unsqueeze(1)
        batch_size, feature_count, class_count, _ = class_stats.shape
        query_count = int(batch.query_values.shape[1])

        class_tokens = self.class_stats_encoder(class_stats)
        class_tokens = torch.where(class_stats_valid.unsqueeze(-1), class_tokens, torch.zeros_like(class_tokens))
        class_summary = _masked_mean(class_tokens, class_stats_valid, dim=2, eps=float(self.config.eps))
        feature_tokens = self.marginal_encoder(marginal_stats) + self.screening_encoder(screening) + class_summary
        feature_tokens = torch.where(feature_valid.unsqueeze(-1), feature_tokens, torch.zeros_like(feature_tokens))
        feature_tokens = self.feature_context(feature_tokens, feature_valid)

        class_tokens = class_tokens + feature_tokens.unsqueeze(2)
        class_tokens = torch.where(class_stats_valid.unsqueeze(-1), class_tokens, torch.zeros_like(class_tokens))

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
        query_tokens = self.query_value_encoder(query_features) + feature_tokens.unsqueeze(1)
        query_tokens = torch.where(feature_valid[:, None, :, None], query_tokens, torch.zeros_like(query_tokens))
        flat_query = query_tokens.reshape(batch_size * query_count, feature_count, int(self.config.d_model))
        flat_valid = feature_valid.unsqueeze(1).expand(-1, query_count, -1).reshape(batch_size * query_count, feature_count)
        flat_query = self.query_context(flat_query, flat_valid)
        query_tokens = flat_query.reshape(batch_size, query_count, feature_count, int(self.config.d_model))

        query_projected = self.query_projection(query_tokens)
        class_projected = self.class_projection(class_tokens)
        bilinear = torch.einsum("bqfd,bfkd->bqfk", query_projected, class_projected)
        bilinear = bilinear / math.sqrt(max(1, int(self.config.d_model)))
        relative = self._relative_channels(batch, class_stats)
        relative_logits = self.relative_evidence(relative).squeeze(-1)
        feature_class_evidence = self.evidence_scale * bilinear + relative_logits

        gate_logits = self.gate_head(class_tokens).squeeze(-1)
        gates = torch.sigmoid(gate_logits)
        valid_fk = class_stats_valid & feature_valid.unsqueeze(-1) & class_valid.unsqueeze(1)
        gates = torch.where(valid_fk, gates, torch.zeros_like(gates))
        feature_class_evidence = torch.where(
            valid_fk.unsqueeze(1),
            feature_class_evidence,
            torch.zeros_like(feature_class_evidence),
        )
        weighted_sum = (feature_class_evidence * gates.unsqueeze(1)).sum(dim=2)
        gate_mass = gates.sum(dim=1).clamp(min=1.0)
        pooled_evidence = weighted_sum / torch.sqrt(gate_mass.unsqueeze(1))

        class_hidden = (class_tokens * gates.unsqueeze(-1)).sum(dim=1) / gate_mass.unsqueeze(-1)
        class_hidden = self.class_hidden_projection(class_hidden)
        query_mass = feature_valid.to(dtype=query_tokens.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
        query_global = (query_tokens * feature_valid[:, None, :, None].to(dtype=query_tokens.dtype)).sum(dim=2) / query_mass.unsqueeze(1)
        query_global = self.query_global_projection(query_global)
        class_query_hidden = class_hidden.unsqueeze(1) + query_global.unsqueeze(2)
        flat_class_hidden = class_query_hidden.reshape(batch_size * query_count, class_count, int(self.config.d_model))
        flat_class_valid = class_valid.unsqueeze(1).expand(-1, query_count, -1).reshape(batch_size * query_count, class_count)
        flat_class_hidden = self.class_context(flat_class_hidden, flat_class_valid)
        class_query_hidden = flat_class_hidden.reshape(batch_size, query_count, class_count, int(self.config.d_model))

        prior = self._class_prior_logits(
            class_stats,
            class_stats_valid=class_stats_valid,
            feature_valid=feature_valid,
            class_valid=class_valid,
        )
        class_logits = pooled_evidence + self.class_logit_head(class_query_hidden).squeeze(-1) + self.class_prior_scale * prior.unsqueeze(1)
        class_logits = torch.where(class_valid.unsqueeze(1), class_logits, torch.full_like(class_logits, -30.0))

        query_reconstruction = self.query_reconstruction_head(query_tokens).squeeze(-1)
        mask_logits = self.mask_estimation_head(query_tokens).squeeze(-1)
        pop_predictions = None
        if self.population_class_reconstruction_head is not None:
            pop_predictions = self.population_class_reconstruction_head(class_tokens)
            pop_predictions = torch.where(class_stats_valid.unsqueeze(-1), pop_predictions, torch.zeros_like(pop_predictions))

        return TabenticsDiakrinoClassStatsOutputs(
            class_logits=class_logits,
            feature_class_evidence=feature_class_evidence,
            feature_class_gates=gates,
            feature_embeddings=feature_tokens,
            class_feature_embeddings=class_tokens,
            query_feature_embeddings=query_tokens,
            query_reconstruction_predictions=query_reconstruction,
            mask_estimation_logits=mask_logits,
            population_class_reconstruction_predictions=pop_predictions,
            feature_valid=feature_valid,
            class_valid=class_valid,
        )


def _balanced_class_weights(labels: torch.Tensor, class_valid: torch.Tensor, *, cfg: TabenticsDiakrinoClassStatsConfig) -> torch.Tensor:
    class_count = int(class_valid.shape[1])
    valid_labels = labels[(labels >= 0) & (labels < class_count)]
    if valid_labels.numel() == 0:
        return torch.ones(class_count, dtype=torch.float32, device=labels.device)
    counts = torch.bincount(valid_labels, minlength=class_count).to(dtype=torch.float32)
    weights = torch.rsqrt(counts.clamp(min=1.0))
    weights = weights / weights[class_valid.any(dim=0)].mean().clamp(min=float(cfg.eps))
    weights = weights.clamp(min=float(cfg.class_weight_min), max=float(cfg.class_weight_max))
    return weights.to(device=labels.device)


def class_stats_classifier_loss(
    outputs: TabenticsDiakrinoClassStatsOutputs,
    batch: TabenticsDiakrinoClassStatsBatch,
    *,
    config: TabenticsDiakrinoClassStatsConfig | None = None,
) -> tuple[torch.Tensor, JsonDict]:
    """Return total loss and scalar diagnostics for the class-stat classifier."""

    _ensure_torch()
    cfg = config or TabenticsDiakrinoClassStatsConfig()
    logits = outputs.class_logits
    class_count = int(logits.shape[-1])
    labels = batch.query_labels.to(device=logits.device, dtype=torch.long)
    valid_query = (labels >= 0) & (labels < class_count)
    total = logits.new_tensor(0.0)
    classification = logits.new_tensor(0.0)
    if bool(torch.any(valid_query).detach().cpu()) and float(cfg.classification_weight) > 0.0:
        weights = _balanced_class_weights(labels, batch.class_valid.to(device=logits.device, dtype=torch.bool), cfg=cfg)
        classification = F.cross_entropy(
            logits[valid_query],
            labels[valid_query],
            weight=weights.to(dtype=logits.dtype),
            label_smoothing=float(cfg.label_smoothing),
        )
        total = total + float(cfg.classification_weight) * classification

    reconstruction = logits.new_tensor(0.0)
    if (
        float(cfg.reconstruction_weight) > 0.0
        and outputs.query_reconstruction_predictions is not None
        and batch.query_reconstruction_targets is not None
        and batch.query_reconstruction_valid is not None
    ):
        valid = batch.query_reconstruction_valid.to(device=logits.device, dtype=torch.bool)
        valid = valid & batch.feature_valid.to(device=logits.device, dtype=torch.bool).unsqueeze(1)
        if bool(torch.any(valid).detach().cpu()):
            targets = batch.query_reconstruction_targets.to(device=logits.device, dtype=logits.dtype)
            reconstruction = F.huber_loss(outputs.query_reconstruction_predictions[valid], targets[valid], delta=1.0)
            total = total + float(cfg.reconstruction_weight) * reconstruction

    mask_estimation = logits.new_tensor(0.0)
    if (
        float(cfg.mask_estimation_weight) > 0.0
        and outputs.mask_estimation_logits is not None
        and batch.mask_estimation_targets is not None
        and batch.mask_estimation_valid is not None
    ):
        valid = batch.mask_estimation_valid.to(device=logits.device, dtype=torch.bool)
        valid = valid & batch.feature_valid.to(device=logits.device, dtype=torch.bool).unsqueeze(1)
        if bool(torch.any(valid).detach().cpu()):
            targets = batch.mask_estimation_targets.to(device=logits.device, dtype=logits.dtype)
            mask_estimation = F.binary_cross_entropy_with_logits(outputs.mask_estimation_logits[valid], targets[valid])
            total = total + float(cfg.mask_estimation_weight) * mask_estimation

    population = logits.new_tensor(0.0)
    if (
        float(cfg.population_class_reconstruction_weight) > 0.0
        and outputs.population_class_reconstruction_predictions is not None
        and batch.population_class_reconstruction_targets is not None
        and batch.population_class_reconstruction_valid is not None
    ):
        valid = batch.population_class_reconstruction_valid.to(device=logits.device, dtype=torch.bool)
        if valid.ndim == outputs.population_class_reconstruction_predictions.ndim:
            valid = valid.any(dim=-1)
        valid = valid & batch.class_stats_valid.to(device=logits.device, dtype=torch.bool)
        if bool(torch.any(valid).detach().cpu()):
            targets = batch.population_class_reconstruction_targets.to(device=logits.device, dtype=logits.dtype)
            preds = outputs.population_class_reconstruction_predictions
            population = F.mse_loss(preds[valid], targets[valid])
            total = total + float(cfg.population_class_reconstruction_weight) * population

    with torch.no_grad():
        valid_count = int(valid_query.sum().detach().cpu())
        pred = torch.argmax(logits, dim=-1)
        accuracy = (
            float((pred[valid_query] == labels[valid_query]).to(dtype=torch.float32).mean().detach().cpu())
            if valid_count > 0
            else 0.0
        )
        gate_valid = outputs.feature_class_gates > 0.0
        gate_mean = (
            float(outputs.feature_class_gates[gate_valid].mean().detach().cpu())
            if bool(torch.any(gate_valid).detach().cpu())
            else 0.0
        )
    report: JsonDict = {
        "loss_total": float(total.detach().cpu()),
        "loss_classification": float(classification.detach().cpu()),
        "loss_reconstruction": float(reconstruction.detach().cpu()),
        "loss_mask_estimation": float(mask_estimation.detach().cpu()),
        "loss_population_class_reconstruction": float(population.detach().cpu()),
        "query_valid_count": valid_count,
        "accuracy": accuracy,
        "gate_mean": gate_mean,
    }
    return total, report


__all__ = [
    "TabenticsDiakrinoClassStatsBatch",
    "TabenticsDiakrinoClassStatsClassifier",
    "TabenticsDiakrinoClassStatsConfig",
    "TabenticsDiakrinoClassStatsOutputs",
    "class_stats_classifier_loss",
    "compute_class_stats_screening_features",
]

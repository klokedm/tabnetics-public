"""Feature-selection teacher initialized from the Tabnetics Diakrino checkpoint.

This module is intentionally additive.  The pretrained DIAKRINO remains available for
classification diagnostics, while this model reuses only the feature-statistics
encoder and salience prior that showed useful real-data signal.  The weak SIREN,
query decoder, tokenizer, HRM blocks, and auxiliary heads are discarded.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint
except Exception:  # pragma: no cover - non-torch import environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from .tabentics_diakrino import TabenticsDiakrinoConfig, _compute_class_conditional_stats, _compute_feature_stats


JsonDict = dict[str, Any]
QUERY_CLASSIFICATION_CLASS_BALANCE_MODES = ("none", "inverse_frequency", "inverse_sqrt_frequency")


def _normalize_query_classification_class_balance(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in QUERY_CLASSIFICATION_CLASS_BALANCE_MODES:
        raise ValueError(
            f"Unsupported query classification class balance {mode!r}; "
            f"expected one of {QUERY_CLASSIFICATION_CLASS_BALANCE_MODES}."
        )
    return value


def query_classification_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float,
    class_balance: str = "none",
) -> torch.Tensor:
    balance = _normalize_query_classification_class_balance(class_balance)
    per_example = F.cross_entropy(
        logits,
        labels,
        label_smoothing=float(label_smoothing),
        reduction="none",
    )
    if balance == "none" or labels.numel() <= 1:
        return per_example.mean()
    class_count = int(logits.shape[-1])
    counts = torch.bincount(
        labels.clamp(min=0, max=max(0, class_count - 1)),
        minlength=class_count,
    ).to(dtype=logits.dtype, device=logits.device)
    counts = counts.clamp(min=1.0)
    if balance == "inverse_sqrt_frequency":
        weights = torch.rsqrt(counts[labels])
    else:
        weights = torch.reciprocal(counts[labels])
    weights = weights / weights.mean().clamp(min=1e-6)
    return (per_example * weights).sum() / weights.sum().clamp(min=1e-6)


def _sdpa_kernel_context(*, backend: str, tokens: torch.Tensor, has_attn_mask: bool) -> Any:
    backend = str(backend).lower()
    if backend == "auto" or not bool(tokens.is_cuda):
        return nullcontext()
    if backend == "flash" and bool(has_attn_mask):
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:  # pragma: no cover - version dependent
        raise RuntimeError(f"Requested SDPA backend {backend!r}, but torch.nn.attention is unavailable.") from exc
    backend_names = {
        "flash": "FLASH_ATTENTION",
        "mem_efficient": "EFFICIENT_ATTENTION",
        "math": "MATH",
    }
    if backend == "mem_efficient" and not hasattr(SDPBackend, "EFFICIENT_ATTENTION"):
        backend_names["mem_efficient"] = "MEM_EFFICIENT_ATTENTION"
    backend_name = backend_names.get(backend)
    if backend_name is None or not hasattr(SDPBackend, backend_name):
        raise RuntimeError(f"Unsupported SDPA backend {backend!r}.")
    return sdpa_kernel(backends=[getattr(SDPBackend, backend_name)])


def _shared_valid_prefix_length(valid_mask: torch.Tensor) -> int | None:
    if valid_mask.ndim != 2 or valid_mask.shape[0] == 0:
        return None
    counts = valid_mask.sum(dim=1)
    if not bool(torch.all(counts == counts[0]).detach().cpu()):
        return None
    prefix_length = int(counts[0].detach().cpu())
    feature_count = int(valid_mask.shape[1])
    if prefix_length <= 0 or prefix_length > feature_count:
        return None
    if prefix_length < feature_count:
        prefix_ok = bool(torch.all(valid_mask[:, :prefix_length]).detach().cpu())
        suffix_ok = bool(torch.all(~valid_mask[:, prefix_length:]).detach().cpu())
        if not prefix_ok or not suffix_ok:
            return None
    return prefix_length


@dataclass(frozen=True)
class TabenticsDiakrinoFSTeacherConfig:
    """Configuration for the real-data feature-selection teacher."""

    architecture_variant: str = "baseline"
    d_model: int = 576
    n_heads: int = 8
    context_layers: int = 3
    ffn_expansion: int = 2
    dropout: float = 0.1
    max_classes: int = 25
    feature_stats_dim: int = 10
    screening_feature_dim: int = 18
    prior_scale_init: float = 1.0
    screening_scale_init: float = 0.25
    residual_scale_init: float = 0.10
    series_scale_init: float = 0.10
    fusion_scale_init: float = 0.25
    calibration_bias_init: float = 0.0
    use_distribution_series: bool = True
    series_samples: int = 16
    refiner_steps: int = 4
    refiner_scale_init: float = 0.10
    refiner_deep_supervision_weight: float = 0.25
    refiner_equilibrium_weight: float = 0.05
    attention_backend: str = "auto"
    # --- KVarN-inspired interaction preconditioning (opt-in; all no-ops at
    # defaults so existing checkpoints load and run identically). See
    # KVARN_INTERACTION_IMPLEMENTATION_PLAN.md. ---
    attn_qk_norm: bool = False
    interaction_mode: str = "bilinear"  # "bilinear" (current) | "cosine"
    interaction_heads: int = 0  # 0 -> use n_heads
    interaction_temp_init: float = 0.07
    interaction_temp_learnable: bool = True
    interaction_scale_channel: bool = True
    # --- Joint micro-sample interaction channel (opt-in; all no-ops at defaults
    # so existing checkpoints load and run identically — the output projection
    # and residual scale are zero-init, so warm-start is bit-identical). Reads a
    # fixed-size set of labeled support rows through induced-point attention to
    # inject cross-feature (interaction) evidence the per-feature stats destroy. ---
    joint_sample_mode: str = "off"  # "off" | "induced"
    joint_sample_size: int = 20
    joint_sample_layers: int = 3
    joint_sample_width: int = 512
    joint_sample_induced_points: int = 64
    joint_sample_heads: int = 8
    joint_sample_scale_init: float = 0.0
    # Optional: also model the QUERY record's own feature interactions (its cells
    # self-attend over features, conditioned on the support summary) and feed
    # that into the query classifier. Zero-init -> no-op until learned. Only
    # active when the query-classification head is enabled.
    joint_sample_include_query: bool = False
    joint_sample_query_scale_init: float = 0.0
    # Recompute the joint encoder's activations during backward instead of
    # retaining them (training-only; no effect on the math or the state dict).
    # The S*F cell tensor plus the ISAB/query blocks dominate the channel's
    # training-step memory at large F.
    joint_sample_checkpoint: bool = False
    # --- v-next full-row joint trunk (T-DIAKRINO-VNEXT-C-01 / #179). ---
    # joint_sample_size <= 0 promotes the channel to a FULL-ROW trunk: ALL
    # support rows are read (TabFM/TabICL-style learned aggregation over raw
    # cells) instead of a strided micro-sample. Positive values keep the
    # historical micro-sample semantics, so existing configs are unchanged.
    # joint_cell_fourier_bands adds a ZERO-INIT Fourier embedding of each cell
    # value alongside the existing linear value projection (TabFM numeric
    # encoding). 0 => module absent => bit-identical warm-start of the promoted
    # joint weights; >0 adds a fresh zero-init projection => still a no-op at
    # init, it only grows a path if it earns gradient.
    joint_cell_fourier_bands: int = 0
    # --- v-next query ICL head (T-DIAKRINO-VNEXT-C-01 / #179): compressed-row causal
    # in-context classifier (TabICL/TabFM ICL transformer). Requires the joint
    # channel (it reuses the joint cell embeddings' per-row means). All outputs
    # are new tensors; the blend into the legacy query logits is gated by the
    # zero-init query_icl_scale, so warm-start stays bit-identical. The CE loss
    # is applied DIRECTLY on query_icl_logits (weight query_icl_weight), so the
    # head has a live gradient even at zero blend scale — the corrected #111/#141
    # lesson (never rely on a zero-init gate for gradient flow).
    query_icl_mode: str = "off"  # "off" | "causal"
    query_icl_layers: int = 2
    query_icl_heads: int = 8
    query_icl_weight: float = 0.0
    query_icl_scale_init: float = 0.0
    query_icl_label_smoothing: float = 0.02
    query_icl_max_rows: int = 0  # 0 -> all available support rows
    # --- v-next pairwise redundancy head (T-DIAKRINO-VNEXT-C-01 / #179; targets the
    # #164 head-trust probe to unblock the #165 redundancy oracle). Low-rank
    # bilinear head over feature-token pairs supervised from SCM ground truth
    # (feature_block_id co-membership + world block_links strength). Off =>
    # modules absent => bit-identical.
    redundancy_head_mode: str = "off"  # "off" | "bilinear"
    redundancy_head_rank: int = 128
    redundancy_pair_weight: float = 0.0
    # --- v-next R1 conformal/FDR selection head (T-DIAKRINO-VNEXT-C-01 / #179;
    # FS_TEACHER_VNEXT_NEXT_STEPS R1, mCS-learn direction). A trained conformity
    # score optimized for the selection frontier: maximize soft recall subject to
    # a soft-FDP-at-q penalty, with a learnable global threshold. The honest
    # conformal calibration stays post-hoc (bsc-run/phase0_probe/
    # conformal_selection.py); this head's job is a score better separated than
    # the FS logit. Off => modules absent => bit-identical.
    conformal_head_mode: str = "off"  # "off" | "scores"
    conformal_selection_weight: float = 0.0
    conformal_target_fdr: float = 0.10
    conformal_fdp_penalty: float = 4.0
    conformal_temperature: float = 0.5
    conformal_relevance_threshold: float = 0.05
    # --- v-next trainer-side losses (T-DIAKRINO-VNEXT-C-01 / #179). Kept on the
    # config (like selector_decorr/null_suppression) so the phase scheduler and
    # the bal3obj rebalancer see one uniform weight surface. All 0.0 =>
    # bit-identical to the promoted base. ---
    # Cross-chunk consistency over multi-panel episodes (train-time fix for the
    # S1 chunk-calibration failure; see research/cross_chunk.py).
    cross_chunk_listwise_weight: float = 0.0
    chunk_mean_offset_weight: float = 0.0
    # V-REx invariance over SCM environments (see research/vrex.py).
    vrex_weight: float = 0.0
    max_feature_tokens: int = 1024
    clip_value: float = 6.0
    eps: float = 1e-6
    focal_alpha: float = 0.85
    focal_gamma: float = 2.0
    listwise_weight: float = 0.35
    tversky_weight: float = 0.20
    tversky_alpha: float = 0.30
    tversky_beta: float = 0.70
    tversky_gamma: float = 1.0
    bce_weight: float = 1.0
    teacher_loss_weight: float = 1.0
    positive_normalized_bce: bool = False
    dynamic_pos_weight: bool = False
    pairwise_rank_weight: float = 0.0
    pairwise_rank_margin: float = 0.0
    pairwise_rank_negatives: int = 64
    selector_gate_weight: float = 0.0
    selector_cardinality_weight: float = 0.0
    # Loss-side feature decorrelation (issue #155). Weight 0.0 => bit-identical to
    # the promoted base. mechanism: "barlow" (off-diagonal redundancy on feature
    # embeddings, the identifiable survivor) or "knockoff" (model-X knockoff margin,
    # staged pending the Phase-0c re-probe greenlight).
    selector_decorr_weight: float = 0.0
    selector_decorr_mechanism: str = "barlow"
    selector_decorr_margin: float = 0.1
    selector_decorr_gate_weighted: bool = True
    # Marginal null-suppression (#154 C2 reframe lever). Weight 0.0 => bit-identical.
    # Pushes selector logits down on pure-null features (proxy_relevance <= threshold);
    # the identifiable linked-vs-null axis the decorrelation mechanisms could not move.
    null_suppression_weight: float = 0.0
    null_suppression_threshold: float = 0.05
    null_suppression_mode: str = "hard"
    selector_cardinality_normalizer: str = "valid"
    # Clamp the per-row (expected - target)/normalizer ratio to [-clip, clip]
    # before squaring. 0.0 -> no clamp (bit-identical). A finite clip (e.g. 3.0)
    # caps the rare saturation spikes that drive the late val-loss blow-ups while
    # leaving the healthy-regime term untouched.
    selector_cardinality_ratio_clip: float = 0.0
    selector_entropy_weight: float = 0.0
    selector_logit_scale_init: float = 0.0
    selector_temperature: float = 1.0
    selector_stochastic: bool = True
    context_candidate_topk: int = 0
    feature_position_encoding: str = "none"
    position_encoding_scale_init: float = 0.0
    position_frequency_bands: int = 16
    feature_metadata_dim: int = 0
    feature_metadata_scale_init: float = 0.0
    sample_class_feature_dim: int = 0
    class_extras_scale_init: float = 0.10
    class_extras_logit_scale_init: float = 0.05
    query_classification_weight: float = 0.0
    query_classification_label_smoothing: float = 0.02
    query_classification_class_balance: str = "none"
    query_class_stats_dim: int = 24
    query_value_dim: int = 4
    query_relative_feature_dim: int = 8
    query_evidence_auxiliary_weight: float = 0.0
    query_evidence_auxiliary_detach_gates: bool = True
    query_selector_relevance_weight: float = 0.0
    query_selector_relevance_listwise_weight: float = 0.0
    query_gate_cardinality_weight: float = 0.0
    query_gate_target_fraction: float = 0.03
    query_gate_entropy_weight: float = 0.0
    query_feature_gate_scale_init: float = 0.25
    query_evidence_scale_init: float = 1.0
    query_class_prior_scale_init: float = 0.20
    query_gate_bias_init: float = 0.0
    query_gate_bias_from_target: bool = False
    local_residual_scale_init: float = 0.0
    support_prediction_weight: float = 0.0
    reconstruction_weight: float = 0.0
    reconstruction_hidden_multiplier: int = 1
    population_reconstruction_weight: float = 0.0
    population_reconstruction_dim: int = 0
    population_class_reconstruction_weight: float = 0.0
    population_class_reconstruction_dim: int = 0
    population_family_weight: float = 0.0
    population_family_classes: int = 0
    population_support_type_weight: float = 0.0
    population_support_type_classes: int = 0
    population_param_weight: float = 0.0
    population_param_nll_weight: float = 0.0
    population_param_dim: int = 0
    # Upper clamp on the population-param logvar head (epistemic spread ceiling).
    # Default 4.0 preserves prior behaviour; the §3.2 param-learnability probe
    # relaxes it (~7.0) so the head can express larger uncertainty instead of
    # collapsing to a constant mean.  Opt-in; see TABENTICS_DIAKRINO_NATIVE_INTEGRATION.md.
    population_param_logvar_max: float = 4.0
    population_dependency_weight: float = 0.0
    population_dependency_dim: int = 0
    population_dependence_type_weight: float = 0.0
    population_dependence_type_classes: int = 0
    population_task_family_weight: float = 0.0
    population_task_family_classes: int = 0
    population_task_variant_weight: float = 0.0
    population_task_variant_classes: int = 0
    population_coeff_weight: float = 0.0
    population_coeff_dim: int = 0
    population_conditioning_mode: str = "prediction"
    population_conditioning_scale_init: float = 0.02
    population_conditioning_detach: bool = True
    population_decision_layers: int = 0
    population_decision_scale_init: float = 0.05
    proxy_listwise_weight: float = 0.0
    proxy_pairwise_rank_weight: float = 0.0
    proxy_pairwise_rank_margin: float = 0.0
    proxy_pairwise_rank_negatives: int = 64
    logit_spread_weight: float = 0.0
    logit_spread_target: float = 2.0
    branch_logit_spread_weight: float = 0.0
    branch_logit_spread_max: float = 4.0
    # z-loss: masked mean-square penalty on the FS relevance logits and the
    # selector-gate logits. A small weight (~1e-4) bounds logit-norm drift under
    # bf16 (the documented late-training instability) without the antagonism that
    # logit_spread_weight has (spread inflates logit std, z-loss shrinks it -
    # enable at most one). 0.0 -> no-op / bit-identical.
    z_loss_weight: float = 0.0
    rate_calibration_weight: float = 0.0
    selector_rate_calibration_weight: float = 0.0

    @classmethod
    def from_diakrino_config(cls, config: JsonDict | TabenticsDiakrinoConfig | None) -> "TabenticsDiakrinoFSTeacherConfig":
        if config is None:
            return cls()
        if isinstance(config, TabenticsDiakrinoConfig):
            source = config.__dict__
        else:
            source = dict(config)
        return cls(
            architecture_variant=str(source.get("architecture_variant", cls.architecture_variant)),
            d_model=int(source.get("d_model", cls.d_model)),
            n_heads=max(1, int(source.get("n_heads", cls.n_heads))),
            context_layers=int(source.get("context_layers", cls.context_layers)),
            ffn_expansion=int(source.get("ffn_expansion", cls.ffn_expansion)),
            dropout=float(source.get("dropout", cls.dropout)),
            max_classes=int(source.get("max_classes", cls.max_classes)),
            max_feature_tokens=int(source.get("max_feature_tokens") or cls.max_feature_tokens),
        )

    def diakrino_encoder_config(self) -> TabenticsDiakrinoConfig:
        return TabenticsDiakrinoConfig(
            d_model=int(self.d_model),
            n_heads=max(1, int(self.n_heads)),
            dropout=float(self.dropout),
            max_classes=int(self.max_classes),
            max_feature_tokens=int(self.max_feature_tokens),
            harmonic_fusion=False,
            enable_chaotic_head=False,
            jepa_weight=0.0,
            sigreg_weight=0.0,
        )


@dataclass(frozen=True)
class TabenticsDiakrinoFSTeacherBatch:
    support: torch.Tensor
    support_mask: torch.Tensor
    support_valid: torch.Tensor
    support_labels: torch.Tensor
    feature_valid_mask: torch.Tensor
    teacher_targets: torch.Tensor
    feature_indices: torch.Tensor | None = None
    feature_positions: torch.Tensor | None = None
    feature_metadata: torch.Tensor | None = None
    feature_stats_input: torch.Tensor | None = None
    screening_features_input: torch.Tensor | None = None
    sample_class_features_input: torch.Tensor | None = None
    proxy_relevance_targets: torch.Tensor | None = None
    distribution_series_input: torch.Tensor | None = None
    distribution_series_valid: torch.Tensor | None = None
    reconstruction_row_indices: torch.Tensor | None = None
    reconstruction_feature_indices: torch.Tensor | None = None
    reconstruction_targets: torch.Tensor | None = None
    reconstruction_valid: torch.Tensor | None = None
    population_reconstruction_targets: torch.Tensor | None = None
    population_reconstruction_valid: torch.Tensor | None = None
    population_class_reconstruction_targets: torch.Tensor | None = None
    population_class_reconstruction_valid: torch.Tensor | None = None
    population_family_targets: torch.Tensor | None = None
    population_family_valid: torch.Tensor | None = None
    population_support_type_targets: torch.Tensor | None = None
    population_support_type_valid: torch.Tensor | None = None
    population_param_targets: torch.Tensor | None = None
    population_param_valid: torch.Tensor | None = None
    population_dependency_targets: torch.Tensor | None = None
    population_dependency_valid: torch.Tensor | None = None
    population_coeff_targets: torch.Tensor | None = None
    population_coeff_valid: torch.Tensor | None = None
    population_dependence_type_targets: torch.Tensor | None = None
    population_dependence_type_valid: torch.Tensor | None = None
    population_task_family_targets: torch.Tensor | None = None
    population_task_family_valid: torch.Tensor | None = None
    population_task_variant_targets: torch.Tensor | None = None
    population_task_variant_valid: torch.Tensor | None = None
    query_values: torch.Tensor | None = None
    query_mask: torch.Tensor | None = None
    query_labels: torch.Tensor | None = None
    query_class_stats: torch.Tensor | None = None
    query_class_stats_valid: torch.Tensor | None = None
    query_class_valid: torch.Tensor | None = None
    # v-next (#179): sampled feature pairs for the redundancy head. Indices are
    # LOCAL token positions [B, K, 2]; targets are graded link strengths in
    # [0, 1] from SCM ground truth; valid masks padded/unusable pairs.
    redundancy_pair_indices: torch.Tensor | None = None
    redundancy_pair_targets: torch.Tensor | None = None
    redundancy_pair_valid: torch.Tensor | None = None
    # v-next (#179): trainer-side loss grouping ids ([B] long, -1 = ungrouped).
    # panel_group_id groups multi-panel views of the same wide episode for the
    # cross-chunk consistency losses; environment_id tags the SCM environment
    # for the V-REx invariance penalty. The model forward ignores both.
    panel_group_id: torch.Tensor | None = None
    environment_id: torch.Tensor | None = None


@dataclass(frozen=True)
class TabenticsDiakrinoFSTeacherOutputs:
    logits: torch.Tensor
    base_logits: torch.Tensor
    prior_logits: torch.Tensor
    screening_logits: torch.Tensor
    series_logits: torch.Tensor
    residual_logits: torch.Tensor
    selector_gate_logits: torch.Tensor
    selector_gate_values: torch.Tensor
    class_extras_logits: torch.Tensor
    refiner_logits: torch.Tensor
    refiner_step_logits: tuple[torch.Tensor, ...]
    feature_embeddings: torch.Tensor
    feature_stats: torch.Tensor
    screening_features: torch.Tensor
    series_embeddings: torch.Tensor
    feature_valid_mask: torch.Tensor
    feature_positions: torch.Tensor | None = None
    feature_metadata: torch.Tensor | None = None
    support_class_logits: torch.Tensor | None = None
    support_labels: torch.Tensor | None = None
    support_valid: torch.Tensor | None = None
    reconstruction_predictions: torch.Tensor | None = None
    reconstruction_targets: torch.Tensor | None = None
    reconstruction_valid: torch.Tensor | None = None
    population_reconstruction_predictions: torch.Tensor | None = None
    population_reconstruction_targets: torch.Tensor | None = None
    population_reconstruction_valid: torch.Tensor | None = None
    population_class_reconstruction_predictions: torch.Tensor | None = None
    population_class_reconstruction_targets: torch.Tensor | None = None
    population_class_reconstruction_valid: torch.Tensor | None = None
    population_family_logits: torch.Tensor | None = None
    population_family_targets: torch.Tensor | None = None
    population_family_valid: torch.Tensor | None = None
    population_support_type_logits: torch.Tensor | None = None
    population_support_type_targets: torch.Tensor | None = None
    population_support_type_valid: torch.Tensor | None = None
    population_param_predictions: torch.Tensor | None = None
    population_param_logvar_predictions: torch.Tensor | None = None
    population_param_targets: torch.Tensor | None = None
    population_param_valid: torch.Tensor | None = None
    population_dependency_predictions: torch.Tensor | None = None
    population_dependency_targets: torch.Tensor | None = None
    population_dependency_valid: torch.Tensor | None = None
    population_coeff_predictions: torch.Tensor | None = None
    population_coeff_targets: torch.Tensor | None = None
    population_coeff_valid: torch.Tensor | None = None
    population_dependence_type_logits: torch.Tensor | None = None
    population_dependence_type_targets: torch.Tensor | None = None
    population_dependence_type_valid: torch.Tensor | None = None
    population_task_family_logits: torch.Tensor | None = None
    population_task_family_targets: torch.Tensor | None = None
    population_task_family_valid: torch.Tensor | None = None
    population_task_variant_logits: torch.Tensor | None = None
    population_task_variant_targets: torch.Tensor | None = None
    population_task_variant_valid: torch.Tensor | None = None
    proxy_relevance_targets: torch.Tensor | None = None
    query_class_logits: torch.Tensor | None = None
    query_labels: torch.Tensor | None = None
    query_class_valid: torch.Tensor | None = None
    query_feature_class_evidence: torch.Tensor | None = None
    query_feature_class_gates: torch.Tensor | None = None
    # v-next (#179) surfaces — all None unless the corresponding opt-in head is
    # enabled, so default-off runs produce byte-identical output structs.
    query_icl_logits: torch.Tensor | None = None
    # Eval-only serving cache inputs. These expose the already-computed joint
    # support context without changing the generic training forward contract.
    joint_support_summary: torch.Tensor | None = None
    joint_support_row_embeddings: torch.Tensor | None = None
    joint_support_row_valid: torch.Tensor | None = None
    joint_support_row_labels: torch.Tensor | None = None
    joint_feature_tokens: torch.Tensor | None = None
    redundancy_pair_logits: torch.Tensor | None = None
    redundancy_pair_targets: torch.Tensor | None = None
    redundancy_pair_valid: torch.Tensor | None = None
    conformal_scores: torch.Tensor | None = None
    conformal_selection_probs: torch.Tensor | None = None


def _ensure_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("tabentics_diakrino_fs_teacher requires torch to be installed.")


def _valid_feature_zscore(values: torch.Tensor, valid_mask: torch.Tensor, eps: float) -> torch.Tensor:
    mask = valid_mask.to(dtype=values.dtype)
    count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (values * mask).sum(dim=1, keepdim=True) / count
    centered = (values - mean) * mask
    var = (centered * centered).sum(dim=1, keepdim=True) / count
    z = (values - mean) / torch.sqrt(var + eps)
    return torch.where(valid_mask, torch.clamp(z, -6.0, 6.0), torch.zeros_like(z))


def _valid_feature_rank01(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Per-row rank feature in [0, 1], ignoring padded feature tokens."""

    if values.numel() == 0:
        return torch.zeros_like(values)
    large = torch.finfo(values.dtype).max
    masked_values = torch.where(valid_mask, values, torch.full_like(values, large))
    order = torch.argsort(masked_values, dim=1, stable=True)
    rank_positions = torch.arange(values.shape[1], device=values.device, dtype=values.dtype).unsqueeze(0)
    ranks = torch.zeros_like(values)
    ranks.scatter_(1, order, rank_positions.expand_as(values))
    denom = (valid_mask.to(dtype=values.dtype).sum(dim=1, keepdim=True) - 1.0).clamp(min=1.0)
    return torch.where(valid_mask, ranks / denom, torch.zeros_like(values))


def _default_feature_positions(valid_mask: torch.Tensor) -> torch.Tensor:
    feature_count = int(valid_mask.shape[1])
    if feature_count <= 1:
        return torch.zeros(valid_mask.shape, dtype=torch.float32, device=valid_mask.device)
    positions = torch.arange(feature_count, device=valid_mask.device, dtype=torch.float32)
    positions = positions.unsqueeze(0).expand(valid_mask.shape[0], -1)
    counts = (valid_mask.sum(dim=1, keepdim=True).to(dtype=torch.float32) - 1.0).clamp(min=1.0)
    positions = positions / counts
    return torch.where(valid_mask, positions, torch.zeros_like(positions))


def _fourier_position_features(positions: torch.Tensor, *, bands: int) -> torch.Tensor:
    band_count = max(1, int(bands))
    dtype = positions.dtype
    device = positions.device
    frequencies = torch.pow(
        torch.tensor(2.0, dtype=dtype, device=device),
        torch.arange(band_count, dtype=dtype, device=device),
    )
    angles = (2.0 * math.pi) * positions.unsqueeze(-1) * frequencies.view(1, 1, -1)
    return torch.cat([positions.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)


def _fourier_cell_features(values: torch.Tensor, *, bands: int) -> torch.Tensor:
    """NeRF/TabFM-style Fourier features of (robust-scaled) cell values (#179).

    ``values`` is any shape; returns ``[..., 2 * bands]`` as
    ``[sin(2^0·π·x), …, sin(2^{B−1}·π·x), cos(2^0·π·x), …]``. Inputs are already
    clipped by the episode robust scaler, so the base frequency π keeps the
    first band smooth across the clipped range.
    """
    band_count = max(1, int(bands))
    frequencies = torch.pow(
        torch.tensor(2.0, dtype=values.dtype, device=values.device),
        torch.arange(band_count, dtype=values.dtype, device=values.device),
    )
    angles = math.pi * values.unsqueeze(-1) * frequencies
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _apply_feature_rope(tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rotary_dim = int(tokens.shape[-1]) - (int(tokens.shape[-1]) % 2)
    if rotary_dim <= 0:
        return tokens
    half_dim = rotary_dim // 2
    dtype = tokens.dtype
    device = tokens.device
    inv_freq = torch.pow(
        torch.tensor(10000.0, dtype=dtype, device=device),
        -torch.arange(half_dim, dtype=dtype, device=device) / max(1, half_dim),
    )
    angles = (2.0 * math.pi) * positions.to(dtype=dtype).unsqueeze(-1) * inv_freq.view(1, 1, -1)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    rotary = tokens[..., :rotary_dim]
    even = rotary[..., 0::2]
    odd = rotary[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    if rotary_dim == int(tokens.shape[-1]):
        return rotated
    return torch.cat([rotated, tokens[..., rotary_dim:]], dim=-1)


def compute_fs_screening_features(
    feature_stats: torch.Tensor,
    *,
    feature_valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build cheap supervised screening channels from marginal/class stats."""

    _ensure_torch()
    mean = feature_stats[..., 0]
    std = feature_stats[..., 1].clamp(min=0.0)
    skew = feature_stats[..., 2]
    kurt = feature_stats[..., 3]
    observed = feature_stats[..., 4].clamp(min=0.0, max=1.0)
    fisher = feature_stats[..., 5].clamp(min=0.0)
    max_shift = feature_stats[..., 6].clamp(min=0.0)
    mean_shift = feature_stats[..., 7].clamp(min=0.0)
    log_std_ratio = feature_stats[..., 8].clamp(min=0.0)
    class_balance = feature_stats[..., 9].clamp(min=0.0, max=1.0)
    log_std = torch.log1p(std)
    abs_mean = mean.abs()
    abs_skew = skew.abs()
    log_kurt = torch.log1p(kurt.abs())
    channels = [
        fisher,
        max_shift,
        mean_shift,
        log_std_ratio,
        class_balance,
        log_std,
        abs_mean,
        observed,
        abs_skew,
        log_kurt,
        _valid_feature_zscore(fisher, feature_valid_mask, eps),
        _valid_feature_zscore(max_shift, feature_valid_mask, eps),
        _valid_feature_zscore(mean_shift, feature_valid_mask, eps),
        _valid_feature_zscore(log_std, feature_valid_mask, eps),
        _valid_feature_rank01(fisher, feature_valid_mask),
        _valid_feature_rank01(max_shift, feature_valid_mask),
        _valid_feature_rank01(log_std, feature_valid_mask),
        1.0 - observed,
    ]
    features = torch.stack(channels, dim=-1)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(feature_valid_mask.unsqueeze(-1), features, torch.zeros_like(features))


def compute_distribution_series(
    support: torch.Tensor,
    *,
    support_mask: torch.Tensor,
    support_valid: torch.Tensor,
    support_labels: torch.Tensor,
    feature_valid_mask: torch.Tensor,
    max_classes: int,
    series_samples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return evenly spaced sorted-value sketches for global and class distributions.

    Output shape:
      series_input: [B, F, K + 1, 2 * Q + 2]
      series_valid: [B, F, K + 1]

    Distribution slot 0 is the global support distribution.  Slots 1..K are
    class-conditional distributions.  For each slot we concatenate Q sampled
    sorted values, Q observed flags, the row fraction for that distribution,
    and an is_global flag.
    """

    _ensure_torch()
    batch_size, _support_count, feature_count = support.shape
    q_count = max(1, int(series_samples))
    class_count = max(1, int(max_classes))
    dist_count = class_count + 1
    dtype = support.dtype
    device = support.device
    series_values = torch.zeros((batch_size, feature_count, dist_count, q_count), dtype=dtype, device=device)
    series_observed = torch.zeros_like(series_values)
    class_fraction = torch.zeros((batch_size, feature_count, dist_count), dtype=dtype, device=device)
    is_global = torch.zeros_like(class_fraction)
    series_valid = torch.zeros((batch_size, feature_count, dist_count), dtype=torch.bool, device=device)
    grid = torch.linspace(0.0, 1.0, steps=q_count, device=device, dtype=dtype)

    for batch_index in range(batch_size):
        valid_rows = torch.nonzero(support_valid[batch_index], as_tuple=False).squeeze(-1)
        support_row_count = max(1, int(valid_rows.numel()))
        for dist_index in range(dist_count):
            if dist_index == 0:
                rows = valid_rows
                row_fraction = 1.0
                is_global[batch_index, :, dist_index] = 1.0
            else:
                class_id = dist_index - 1
                row_mask = support_valid[batch_index] & (support_labels[batch_index] == class_id)
                rows = torch.nonzero(row_mask, as_tuple=False).squeeze(-1)
                row_fraction = float(rows.numel()) / float(support_row_count)
            if rows.numel() == 0:
                continue
            values = support[batch_index, rows, :]
            observed = ~support_mask[batch_index, rows, :]
            observed_count = observed.sum(dim=0)
            dist_valid = (observed_count > 0) & feature_valid_mask[batch_index]
            if not torch.any(dist_valid):
                continue
            sorted_values = torch.sort(values.masked_fill(~observed, float("inf")), dim=0).values
            safe_count = observed_count.clamp(min=1)
            gather_positions = torch.round((safe_count.to(dtype=dtype).unsqueeze(-1) - 1.0) * grid.unsqueeze(0)).to(dtype=torch.long)
            gathered = sorted_values.gather(0, gather_positions.transpose(0, 1)).transpose(0, 1)
            gathered = torch.nan_to_num(gathered, nan=0.0, posinf=0.0, neginf=0.0)
            series_values[batch_index, :, dist_index, :] = torch.where(
                dist_valid.unsqueeze(-1),
                gathered,
                torch.zeros_like(gathered),
            )
            series_observed[batch_index, :, dist_index, :] = dist_valid.to(dtype=dtype).unsqueeze(-1)
            class_fraction[batch_index, :, dist_index] = float(row_fraction)
            series_valid[batch_index, :, dist_index] = dist_valid

    meta = torch.stack([class_fraction, is_global], dim=-1)
    series_input = torch.cat([series_values, series_observed, meta], dim=-1)
    series_input = torch.nan_to_num(series_input, nan=0.0, posinf=0.0, neginf=0.0)
    return series_input, series_valid


class _SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward projection with dropout on the gated hidden state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        output_dim = int(input_dim) if output_dim is None else int(output_dim)
        self.input_proj = nn.Linear(int(input_dim), 2 * int(hidden_dim))
        self.output_proj = nn.Linear(int(hidden_dim), output_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input_proj(x).chunk(2, dim=-1)
        return self.output_proj(self.dropout(value * F.silu(gate)))


class _SwiGLUResidualBlock(nn.Module):
    """Pre-norm residual MLP block used by the opt-in FS-teacher variant."""

    def __init__(self, d_model: int, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(int(d_model))
        self.ffn = _SwiGLUFFN(int(d_model), int(hidden_dim), dropout=float(dropout))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.ffn(self.norm(x)))


class _FlexibleFeatureStatsEncoder(nn.Module):
    """Feature-stat encoder with the same state names as the DIAKRINO 10-stat encoder."""

    def __init__(self, *, input_dim: int, d_model: int) -> None:
        super().__init__()
        hidden = max(16, int(d_model))
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(d_model)),
            nn.RMSNorm(int(d_model)),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.net(stats)


def _match_last_dim(values: torch.Tensor, target_dim: int) -> torch.Tensor:
    target = max(1, int(target_dim))
    current = int(values.shape[-1])
    if current == target:
        return values
    if current > target:
        return values[..., :target]
    return F.pad(values, (0, target - current))


def _masked_token_mean(tokens: torch.Tensor, valid_mask: torch.Tensor, *, eps: float) -> torch.Tensor:
    mask = valid_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=float(eps))
    return (tokens * mask).sum(dim=1) / denom


def _probability_logit(value: float, *, eps: float) -> float:
    prob = min(1.0 - float(eps), max(float(eps), float(value)))
    return math.log(prob / (1.0 - prob))


class _SwiGLUScoringHead(nn.Module):
    """Small gated scoring head for feature-wise salience logits."""

    def __init__(self, d_model: int, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(int(d_model))
        self.ffn = _SwiGLUFFN(int(d_model), int(hidden_dim), int(d_model), dropout=float(dropout))
        self.output = nn.Linear(int(d_model), 1)
        nn.init.xavier_uniform_(self.output.weight, gain=0.5)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.ffn(self.norm(x)))


def _l2_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Direction-only L2 normalization over the last dim (fp32-accumulated).

    Zero rows map to zero (not NaN): the squared-norm is floored at ``eps**2``
    before the reciprocal sqrt. torch-2.3 safe (no torch>=2.4 ops).
    """
    x_fp = x.float()
    inv = x_fp.pow(2).sum(dim=-1, keepdim=True).clamp_min(float(eps) ** 2).rsqrt()
    return (x_fp * inv).to(x.dtype)


class _BalancedCosineInteraction(nn.Module):
    """KVarN-style interaction preconditioning for query<->class evidence.

    Replaces a raw bilinear dot product (whose magnitude is contaminated by
    token norm) with **per-head scaled cosine** evidence, then re-injects the
    removed magnitude through a **zero-initialised** learned side-channel so the
    layer starts as pure direction and only grows a magnitude path if it earns
    gradient. This is the Weight-Norm / CLIP / Swin-V2 instantiation of the
    "separate direction from magnitude, then give magnitude back" principle.

    Opt-in and torch-2.3 safe: only einsum / softmax / clamp / log / Linear.
    """

    _MAX_INV_TEMP = 100.0  # Swin-V2 logit-scale cap; avoids degenerate sharpening.

    def __init__(
        self,
        *,
        d_model: int,
        heads: int,
        temp_init: float,
        temp_learnable: bool,
        scale_channel: bool,
        eps: float,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        heads = max(1, int(heads))
        while d_model % heads != 0 and heads > 1:
            heads -= 1
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.eps = float(eps)
        inv_temp = math.log(1.0 / max(1e-4, float(temp_init)))
        init = torch.full((heads,), float(inv_temp))
        if bool(temp_learnable):
            self.log_inv_temp = nn.Parameter(init)
        else:
            self.register_buffer("log_inv_temp", init, persistent=True)
        self.scale_channel = bool(scale_channel)
        if self.scale_channel:
            self.scale_mlp: nn.Module | None = nn.Sequential(
                nn.Linear(2, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.scale_mlp[-1].weight)
            nn.init.zeros_(self.scale_mlp[-1].bias)
        else:
            self.scale_mlp = None

    def forward(self, query_proj: torch.Tensor, class_proj: torch.Tensor) -> torch.Tensor:
        # query_proj: [B, F, D]; class_proj: [B, F, K, D] -> evidence [B, F, K]
        batch, fields, classes, d_model = class_proj.shape
        heads, head_dim = self.heads, self.head_dim
        q_heads = query_proj.reshape(batch, fields, heads, head_dim)
        k_heads = class_proj.reshape(batch, fields, classes, heads, head_dim)
        cos = torch.einsum(
            "bfhd,bfkhd->bfkh",
            _l2_normalize(q_heads, self.eps),
            _l2_normalize(k_heads, self.eps),
        )
        inv_temp = self.log_inv_temp.float().exp().clamp(max=self._MAX_INV_TEMP)
        evidence = (cos.float() * inv_temp).mean(dim=-1).to(query_proj.dtype)
        if self.scale_mlp is not None:
            log_rq = query_proj.float().pow(2).sum(-1).clamp_min(self.eps).sqrt().log()
            log_rk = class_proj.float().pow(2).sum(-1).clamp_min(self.eps).sqrt().log()
            scale_feats = torch.stack(
                [log_rq.unsqueeze(-1).expand(-1, -1, classes), log_rk], dim=-1
            )
            evidence = evidence + self.scale_mlp(scale_feats).squeeze(-1).to(query_proj.dtype)
        return evidence


def _apply_qk_norm(
    query: torch.Tensor,
    key: torch.Tensor,
    q_norm: nn.Module | None,
    k_norm: nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMS QK-normalization before SDPA (ViT-22B / Gemma-2 recipe).

    Bounds attention-logit growth (the σReparam entropy-collapse failure mode)
    without touching values. No-op when the norms are absent (opt-in default).
    """
    if q_norm is None or k_norm is None:
        return query, key
    return q_norm(query), k_norm(key)


class _SwiGLUContextLayer(nn.Module):
    """Self-attention plus SwiGLU FFN for the opt-in context encoder."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        hidden_dim: int,
        dropout: float,
        attention_backend: str,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.attention_backend = str(attention_backend)
        self.token_norm_attn = nn.RMSNorm(int(d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=max(1, int(n_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        head_dim = int(d_model) // max(1, int(self.attn.num_heads))
        if bool(qk_norm):
            self.q_norm: nn.Module | None = nn.RMSNorm(head_dim)
            self.k_norm: nn.Module | None = nn.RMSNorm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        self.token_norm_ffn = nn.RMSNorm(int(d_model))
        self.ffn = _SwiGLUFFN(int(d_model), int(hidden_dim), dropout=float(dropout))
        self.attn_residual_scale = nn.Parameter(torch.tensor(1.0))
        self.ffn_residual_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(float(dropout))

    def _attention_kernel_context(self, tokens: torch.Tensor, *, has_attn_mask: bool) -> Any:
        return _sdpa_kernel_context(
            backend=str(self.attention_backend),
            tokens=tokens,
            has_attn_mask=bool(has_attn_mask),
        )

    def _fused_self_attention(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, d_model = tokens.shape
        head_count = int(self.attn.num_heads)
        head_dim = int(d_model) // max(1, head_count)
        qkv = F.linear(tokens, self.attn.in_proj_weight, self.attn.in_proj_bias)
        qkv = qkv.view(batch_size, token_count, 3, head_count, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        query, key = _apply_qk_norm(query, key, self.q_norm, self.k_norm)
        all_tokens_valid = bool(torch.all(valid_mask).detach().cpu())
        attn_mask = None if all_tokens_valid else valid_mask[:, None, None, :]
        with self._attention_kernel_context(tokens, has_attn_mask=attn_mask is not None):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=float(self.attn.dropout) if self.training else 0.0,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, token_count, d_model)
        return self.attn.out_proj(attended)

    def forward(self, tokens: torch.Tensor, *, valid_mask: torch.Tensor) -> torch.Tensor:
        normed = self.token_norm_attn(tokens)
        tokens = tokens + self.attn_residual_scale * self.dropout(self._fused_self_attention(normed, valid_mask))
        tokens = tokens + self.ffn_residual_scale * self.dropout(self.ffn(self.token_norm_ffn(tokens)))
        return torch.where(valid_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))


class _SwiGLUContextEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
        attention_backend: str,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _SwiGLUContextLayer(
                    d_model=int(d_model),
                    n_heads=int(n_heads),
                    hidden_dim=int(hidden_dim),
                    dropout=float(dropout),
                    attention_backend=str(attention_backend),
                    qk_norm=bool(qk_norm),
                )
                for _ in range(max(0, int(num_layers)))
            ]
        )

    def forward(self, tokens: torch.Tensor, *, valid_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens, valid_mask=valid_mask)
        return tokens


class _TinySalienceRefiner(nn.Module):
    """Weight-shared recurrent feature-token refiner.

    This is a small TRM/HRM-inspired module for salience, not a resurrection of
    the old classification HRM path.  It iteratively updates feature tokens and
    a global state token until logits stabilize.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dropout: float,
        ffn_expansion: int,
        attention_backend: str = "auto",
        use_swiglu: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        hidden = max(int(d_model), int(d_model) * int(ffn_expansion))
        self.attention_backend = str(attention_backend)
        head_dim = int(d_model) // max(1, int(n_heads))
        if bool(qk_norm):
            self.q_norm: nn.Module | None = nn.RMSNorm(head_dim)
            self.k_norm: nn.Module | None = nn.RMSNorm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        state_activation: nn.Module = nn.SiLU() if bool(use_swiglu) else nn.GELU()
        self.state_init = nn.Sequential(
            nn.Linear(int(d_model), int(d_model)),
            state_activation,
            nn.RMSNorm(int(d_model)),
        )
        self.state_to_token = nn.Linear(int(d_model), int(d_model))
        self.token_norm_attn = nn.RMSNorm(int(d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=max(1, int(n_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.token_norm_ffn = nn.RMSNorm(int(d_model))
        if bool(use_swiglu):
            self.ffn: nn.Module = _SwiGLUFFN(int(d_model), hidden, dropout=float(dropout))
        else:
            self.ffn = nn.Sequential(
                nn.Linear(int(d_model), hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, int(d_model)),
            )
        self.state_update = nn.GRUCell(int(d_model), int(d_model))
        self.output_norm = nn.RMSNorm(int(d_model))
        self.logit_head = nn.Linear(int(d_model), 1)
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return (tokens * mask).sum(dim=1) / count

    def _attention_kernel_context(self, tokens: torch.Tensor, *, has_attn_mask: bool) -> Any:
        return _sdpa_kernel_context(
            backend=str(self.attention_backend),
            tokens=tokens,
            has_attn_mask=bool(has_attn_mask),
        )

    def _fused_self_attention(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, d_model = tokens.shape
        head_count = int(self.attn.num_heads)
        head_dim = int(d_model) // max(1, head_count)
        qkv = F.linear(tokens, self.attn.in_proj_weight, self.attn.in_proj_bias)
        qkv = qkv.view(batch_size, token_count, 3, head_count, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        query, key = _apply_qk_norm(query, key, self.q_norm, self.k_norm)
        all_tokens_valid = bool(torch.all(valid_mask).detach().cpu())
        attn_mask = None if all_tokens_valid else valid_mask[:, None, None, :]
        with self._attention_kernel_context(tokens, has_attn_mask=attn_mask is not None):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=float(self.attn.dropout) if self.training else 0.0,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, token_count, d_model)
        return self.attn.out_proj(attended)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        steps: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        step_count = max(0, int(steps))
        if step_count <= 0:
            return tokens, ()
        state = self.state_init(self._masked_mean(tokens, valid_mask))
        step_logits: list[torch.Tensor] = []
        for _ in range(step_count):
            conditioned = tokens + self.state_to_token(state).unsqueeze(1)
            normed = self.token_norm_attn(conditioned)
            attn_out = self._fused_self_attention(normed, valid_mask)
            tokens = tokens + self.dropout(attn_out)
            tokens = tokens + self.dropout(self.ffn(self.token_norm_ffn(tokens)))
            tokens = torch.where(valid_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
            pooled = self._masked_mean(tokens, valid_mask)
            state = self.state_update(pooled, state)
            raw_logits = self.logit_head(self.output_norm(tokens)).squeeze(-1)
            step_logits.append(raw_logits)
        return tokens, tuple(step_logits)


class _JointMAB(nn.Module):
    """Pre-norm multihead attention block: a query set attends to a key/value
    set, then a SwiGLU FFN. Supports an optional key-validity mask and per-head
    QK-norm. torch-2.3 safe (manual SDPA, same idiom as ``_SwiGLUContextLayer``).
    Used as the cross/self-attention primitive of ``_JointSampleEncoder``.
    """

    def __init__(
        self,
        d_model: int,
        *,
        heads: int,
        hidden_dim: int,
        dropout: float,
        attention_backend: str,
        qk_norm: bool,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        heads = max(1, int(heads))
        while d_model % heads != 0 and heads > 1:
            heads -= 1
        self.heads = heads
        self.head_dim = d_model // heads
        self.attention_backend = str(attention_backend)
        self.q_norm_in = nn.RMSNorm(d_model)
        self.kv_norm_in = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        if bool(qk_norm):
            self.q_head_norm: nn.Module | None = nn.RMSNorm(self.head_dim)
            self.k_head_norm: nn.Module | None = nn.RMSNorm(self.head_dim)
        else:
            self.q_head_norm = None
            self.k_head_norm = None
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = _SwiGLUFFN(d_model, int(hidden_dim), dropout=float(dropout))
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        key_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, q_len, d_model = query.shape
        c_len = context.shape[1]
        heads, head_dim = self.heads, self.head_dim
        q = self.q_proj(self.q_norm_in(query)).view(batch, q_len, heads, head_dim).transpose(1, 2)
        ctx = self.kv_norm_in(context)
        k = self.k_proj(ctx).view(batch, c_len, heads, head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(batch, c_len, heads, head_dim).transpose(1, 2)
        q, k = _apply_qk_norm(q, k, self.q_head_norm, self.k_head_norm)
        attn_mask = None
        if key_valid is not None:
            # Guard against an all-invalid key row (softmax over all -inf -> NaN):
            # if a batch element has zero valid keys, unmask all (cells are zero,
            # so the attended value is a benign zero-ish vector).
            safe = key_valid | (~key_valid.any(dim=1, keepdim=True))
            attn_mask = safe[:, None, None, :]
        with _sdpa_kernel_context(
            backend=self.attention_backend,
            tokens=query,
            has_attn_mask=attn_mask is not None,
        ):
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=float(self.dropout.p) if self.training else 0.0,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch, q_len, d_model)
        tokens = query + self.dropout(self.out_proj(attended))
        tokens = tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))
        return tokens


class _JointSampleEncoder(nn.Module):
    """Axial micro-sample encoder (Set-Transformer / ISAB style; opt-in).

    Reads a fixed-size set of S labeled support rows JOINTLY so the model can
    pick up cross-feature (interaction) relevance that the per-feature marginal /
    class-conditional statistics structurally destroy (e.g. XOR/parity-type
    signal). See the design review and NPT (arXiv:2106.02584), SAINT
    (arXiv:2106.01342), Set Transformer (arXiv:1810.00825).

    Each cell (row x feature) is embedded from (value, missing-flag, row-label,
    per-feature CONTENT identity = the trunk feature token projected down — so
    feature exchangeability is preserved, no feature-index encoding). All S*F
    cells are pooled through M learnable INDUCED points (cost linear in F, not
    F^2), which form the interaction bottleneck; the F feature tokens then read
    that summary by cross-attention. The output projection is ZERO-initialised,
    so with a zero residual scale the channel is a perfect no-op and a
    warm-started checkpoint is reproduced bit-for-bit; it only grows a path if it
    earns gradient. Runs at a reduced inner width ``enc_dim`` to keep memory
    small. torch-2.3 safe.
    """

    def __init__(
        self,
        *,
        d_model: int,
        enc_dim: int,
        n_heads: int,
        n_layers: int,
        induced_points: int,
        num_classes: int,
        dropout: float,
        attention_backend: str,
        qk_norm: bool,
        eps: float,
        include_query: bool = False,
        fourier_bands: int = 0,
    ) -> None:
        super().__init__()
        enc_dim = int(enc_dim)
        hidden = max(enc_dim, enc_dim * 2)
        self.include_query = bool(include_query)
        self.enc_dim = enc_dim
        self.eps = float(eps)
        self.num_classes = max(1, int(num_classes))
        self.value_proj = nn.Linear(1, enc_dim)
        self.fourier_bands = max(0, int(fourier_bands))
        if self.fourier_bands > 0:
            # TabFM-style Fourier numeric cell encoding (#179). ZERO-INIT so the
            # promoted linear value path warm-starts bit-identically; the Fourier
            # path only grows if it earns gradient.
            self.fourier_proj: nn.Module | None = nn.Linear(2 * self.fourier_bands, enc_dim)
            nn.init.zeros_(self.fourier_proj.weight)
            nn.init.zeros_(self.fourier_proj.bias)
        else:
            self.fourier_proj = None
        self.missing_embed = nn.Parameter(torch.zeros(enc_dim))
        # +1 row-label slot for invalid/padded rows.
        self.label_embed = nn.Embedding(self.num_classes + 1, enc_dim)
        self.feature_id_proj = nn.Linear(int(d_model), enc_dim)
        self.feature_query_proj = nn.Linear(int(d_model), enc_dim)
        self.cell_norm = nn.RMSNorm(enc_dim)
        n_layers = max(1, int(n_layers))
        self.induced = nn.Parameter(torch.randn(max(1, int(induced_points)), enc_dim) * 0.02)
        common = dict(
            heads=max(1, int(n_heads)),
            hidden_dim=hidden,
            dropout=float(dropout),
            attention_backend=str(attention_backend),
            qk_norm=bool(qk_norm),
        )
        # ISAB block i: induced points pool over cells, then self-attend.
        self.pool_blocks = nn.ModuleList([_JointMAB(enc_dim, **common) for _ in range(n_layers)])
        self.self_blocks = nn.ModuleList([_JointMAB(enc_dim, **common) for _ in range(n_layers)])
        # Feature tokens read the pooled summary.
        self.read_block = _JointMAB(enc_dim, **common)
        self.out_proj = nn.Linear(enc_dim, int(d_model))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Optional QUERY-row path: the query record's own cells self-attend over
        # features (F x F: query feature A sees query feature B -> forms the
        # query's own A^B interaction), conditioned on the support-derived
        # induced summary, producing a per-feature interaction read that feeds the
        # query classifier. Zero-init output so it is a no-op until it earns it.
        if self.include_query:
            self.query_self_blocks: nn.ModuleList | None = nn.ModuleList(
                [_JointMAB(enc_dim, **common) for _ in range(n_layers)]
            )
            self.query_cross_blocks: nn.ModuleList | None = nn.ModuleList(
                [_JointMAB(enc_dim, **common) for _ in range(n_layers)]
            )
            self.query_out_proj: nn.Module | None = nn.Linear(enc_dim, int(d_model))
            nn.init.zeros_(self.query_out_proj.weight)
            nn.init.zeros_(self.query_out_proj.bias)
        else:
            self.query_self_blocks = None
            self.query_cross_blocks = None
            self.query_out_proj = None

    def _embed_values(self, values: torch.Tensor) -> torch.Tensor:
        """Shared numeric cell embedding: linear value path + optional zero-init
        Fourier path (#179). ``values`` is [..., ] (no trailing channel dim)."""
        cell = self.value_proj(values.unsqueeze(-1))
        if self.fourier_proj is not None:
            cell = cell + self.fourier_proj(_fourier_cell_features(values, bands=self.fourier_bands))
        return cell

    def _support_summary_and_selection(
        self,
        sample_values: torch.Tensor,
        sample_missing: torch.Tensor,
        sample_row_valid: torch.Tensor,
        sample_labels: torch.Tensor,
        feature_tokens: torch.Tensor,
        feature_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = feature_tokens.dtype
        batch, n_rows, n_feat = sample_values.shape
        cell = self._embed_values(sample_values.to(dtype))             # [B,S,F,enc]
        missing = self.missing_embed.to(dtype).view(1, 1, 1, -1)
        cell = torch.where(sample_missing.unsqueeze(-1), missing.expand_as(cell), cell)
        feat_id = self.feature_id_proj(feature_tokens).to(dtype)       # [B,F,enc]
        cell = cell + feat_id.unsqueeze(1)
        labels = sample_labels.clamp(min=0, max=self.num_classes - 1)
        labels = torch.where(sample_row_valid, labels, torch.full_like(labels, self.num_classes))
        cell = cell + self.label_embed(labels).to(dtype).unsqueeze(2)  # [B,S,1,enc]
        cell = self.cell_norm(cell)
        cell_valid = sample_row_valid.unsqueeze(-1) & feature_valid.unsqueeze(1)  # [B,S,F]
        cell = torch.where(cell_valid.unsqueeze(-1), cell, torch.zeros_like(cell))
        # Per-row bag-of-cells embeddings for the query-ICL head (#179): masked
        # mean over the row's valid cells. Cheap (reuses the cell tensor) and
        # parameter-free, so the zero-init/no-op contract is unaffected.
        row_counts = cell_valid.to(dtype=cell.dtype).sum(dim=2, keepdim=True).clamp(min=1.0)  # [B,S,1]
        row_means = cell.sum(dim=2) / row_counts                                              # [B,S,enc]
        cells = cell.reshape(batch, n_rows * n_feat, self.enc_dim)
        cells_valid = cell_valid.reshape(batch, n_rows * n_feat)
        summary = self.induced.to(dtype).unsqueeze(0).expand(batch, -1, -1).contiguous()
        for pool, slf in zip(self.pool_blocks, self.self_blocks):
            summary = pool(summary, cells, key_valid=cells_valid)
            summary = slf(summary, summary, key_valid=None)
        feature_query = self.feature_query_proj(feature_tokens).to(dtype)  # [B,F,enc]
        read = self.read_block(feature_query, summary, key_valid=None)     # [B,F,enc]
        read = torch.where(feature_valid.unsqueeze(-1), read, torch.zeros_like(read))
        return self.out_proj(read), summary, row_means

    def query_row_embedding(
        self,
        *,
        query_values: torch.Tensor,
        query_mask: torch.Tensor | None,
        feature_tokens: torch.Tensor,
        feature_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Bag-of-cells embedding of the (unlabeled) query row for the query-ICL
        head (#179). Shares the cell embedding parameters with the support path
        (value/Fourier projections, feature-content identity, cell norm); the
        label slot is deliberately absent — the ICL head adds its own
        "unknown"-label embedding."""
        dtype = feature_tokens.dtype
        q_cell = self._embed_values(query_values.to(dtype))            # [B,F,enc]
        if query_mask is not None:
            q_missing = query_mask.to(dtype=torch.bool).unsqueeze(-1)
            q_cell = torch.where(
                q_missing, self.missing_embed.to(dtype).view(1, 1, -1).expand_as(q_cell), q_cell
            )
        q_cell = q_cell + self.feature_id_proj(feature_tokens).to(dtype)
        q_cell = self.cell_norm(q_cell)
        q_cell = torch.where(feature_valid.unsqueeze(-1), q_cell, torch.zeros_like(q_cell))
        denom = feature_valid.to(dtype=q_cell.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
        return q_cell.sum(dim=1) / denom                               # [B,enc]

    def _query_interaction(
        self,
        summary: torch.Tensor,
        feature_tokens: torch.Tensor,
        feature_valid: torch.Tensor,
        query_values: torch.Tensor,
        query_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        dtype = feature_tokens.dtype
        batch, n_feat = query_values.shape
        q_cell = self._embed_values(query_values.to(dtype))             # [B,F,enc]
        if query_mask is not None:
            q_missing = query_mask.to(dtype=torch.bool).unsqueeze(-1)
            q_cell = torch.where(
                q_missing, self.missing_embed.to(dtype).view(1, 1, -1).expand_as(q_cell), q_cell
            )
        # Same per-feature content identity as the support cells; recomputed here
        # (deterministic linear) so the two checkpoint regions stay independent.
        q_cell = q_cell + self.feature_id_proj(feature_tokens).to(dtype)
        # The query row is unlabeled -> the dedicated "unknown" label slot.
        unknown = torch.full((batch, n_feat), self.num_classes, dtype=torch.long, device=q_cell.device)
        q_cell = q_cell + self.label_embed(unknown).to(dtype)
        q_cell = self.cell_norm(q_cell)
        q_cell = torch.where(feature_valid.unsqueeze(-1), q_cell, torch.zeros_like(q_cell))
        assert self.query_self_blocks is not None and self.query_cross_blocks is not None
        for self_blk, cross_blk in zip(self.query_self_blocks, self.query_cross_blocks):
            # F x F: the query's own feature A attends to feature B -> A^B.
            q_cell = self_blk(q_cell, q_cell, key_valid=feature_valid)
            # condition on the task's joint structure distilled from support.
            q_cell = cross_blk(q_cell, summary, key_valid=None)
        q_cell = torch.where(feature_valid.unsqueeze(-1), q_cell, torch.zeros_like(q_cell))
        assert self.query_out_proj is not None
        return self.query_out_proj(q_cell)

    def forward(
        self,
        *,
        sample_values: torch.Tensor,
        sample_missing: torch.Tensor,
        sample_row_valid: torch.Tensor,
        sample_labels: torch.Tensor,
        feature_tokens: torch.Tensor,
        feature_valid: torch.Tensor,
        query_values: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        use_checkpoint: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        # sample_values [B,S,F]; sample_missing [B,S,F] (True=missing);
        # sample_row_valid [B,S]; sample_labels [B,S]; feature_tokens [B,F,d];
        # feature_valid [B,F]. query_values/query_mask [B,F] (optional).
        # Returns (selection_residual [B,F,d], query_residual [B,F,d] | None,
        # row_means [B,S,enc], support_summary [B,I,enc]); the residuals are
        # zero at init and the compact support tensors are reusable at serving.
        #
        # use_checkpoint: the support branch and the query branch MUST be two
        # SEPARATE checkpoint regions. The query residual is consumed next to
        # the loss, so its gradient arrives at the START of backward, while the
        # trunk's activations are still fully live — a single region would
        # recompute the big support branch (S*F cells + ISAB, multi-GiB) at the
        # worst possible moment and not reduce the peak at all (measured: MN5
        # job 41726894 OOM'd inside the recompute). Split, the small query
        # region recomputes early (cheap) and the big support region recomputes
        # at the END of backward, after the trunk's activations are freed.
        ckpt = bool(use_checkpoint) and torch.is_grad_enabled()
        if ckpt:
            selection_out, summary, row_means = torch.utils.checkpoint.checkpoint(
                self._support_summary_and_selection,
                sample_values,
                sample_missing,
                sample_row_valid,
                sample_labels,
                feature_tokens,
                feature_valid,
                use_reentrant=False,
            )
        else:
            selection_out, summary, row_means = self._support_summary_and_selection(
                sample_values, sample_missing, sample_row_valid, sample_labels, feature_tokens, feature_valid
            )
        query_out: torch.Tensor | None = None
        if (
            self.include_query
            and self.query_out_proj is not None
            and self.query_self_blocks is not None
            and self.query_cross_blocks is not None
            and query_values is not None
        ):
            if ckpt:
                query_out = torch.utils.checkpoint.checkpoint(
                    self._query_interaction,
                    summary,
                    feature_tokens,
                    feature_valid,
                    query_values,
                    query_mask,
                    use_reentrant=False,
                )
            else:
                query_out = self._query_interaction(summary, feature_tokens, feature_valid, query_values, query_mask)
        return selection_out, query_out, row_means, summary


def _apply_sequence_rope(heads: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding over a [B, H, T, D] head tensor (#179).

    Standard RoPE (base 10000) over integer sequence positions; used by the
    query-ICL causal blocks so the query token can address "how far back" a
    support row sits while keeping relative structure. torch-2.3 safe.
    """
    head_dim = int(heads.shape[-1])
    rotary_dim = head_dim - (head_dim % 2)
    if rotary_dim <= 0:
        return heads
    half_dim = rotary_dim // 2
    dtype = heads.dtype
    device = heads.device
    inv_freq = torch.pow(
        torch.tensor(10000.0, dtype=dtype, device=device),
        -torch.arange(half_dim, dtype=dtype, device=device) / max(1, half_dim),
    )
    angles = positions.to(dtype=dtype).unsqueeze(-1) * inv_freq.view(1, -1)  # [T, half]
    sin = torch.sin(angles).view(1, 1, -1, half_dim)
    cos = torch.cos(angles).view(1, 1, -1, half_dim)
    rotary = heads[..., :rotary_dim]
    even = rotary[..., 0::2]
    odd = rotary[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    if rotary_dim == head_dim:
        return rotated
    return torch.cat([rotated, heads[..., rotary_dim:]], dim=-1)


class _CausalICLBlock(nn.Module):
    """Pre-norm causal self-attention + SwiGLU FFN over a row sequence (#179).

    The in-context-learning primitive of ``_QueryICLHead``: compressed support
    rows form the causal prefix and the query row sits last (TabFM's "24-block
    causal transformer over compressed row vectors", at v-next scale). RoPE on
    q/k, optional per-head QK-norm, explicit causal∧valid key mask (the diagonal
    stays open so padded rows cannot produce an all-masked softmax).
    torch-2.3 safe (same manual-SDPA idiom as ``_JointMAB``).
    """

    def __init__(
        self,
        d_model: int,
        *,
        heads: int,
        hidden_dim: int,
        dropout: float,
        attention_backend: str,
        qk_norm: bool,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        heads = max(1, int(heads))
        while d_model % heads != 0 and heads > 1:
            heads -= 1
        self.heads = heads
        self.head_dim = d_model // heads
        self.attention_backend = str(attention_backend)
        self.norm_attn = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        if bool(qk_norm):
            self.q_head_norm: nn.Module | None = nn.RMSNorm(self.head_dim)
            self.k_head_norm: nn.Module | None = nn.RMSNorm(self.head_dim)
        else:
            self.q_head_norm = None
            self.k_head_norm = None
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = _SwiGLUFFN(d_model, int(hidden_dim), dropout=float(dropout))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor, *, key_valid: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = tokens.shape
        heads, head_dim = self.heads, self.head_dim
        normed = self.norm_attn(tokens)
        q = self.q_proj(normed).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        q, k = _apply_qk_norm(q, k, self.q_head_norm, self.k_head_norm)
        positions = torch.arange(seq_len, device=tokens.device)
        q = _apply_sequence_rope(q, positions)
        k = _apply_sequence_rope(k, positions)
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=tokens.device))
        allowed = causal.view(1, 1, seq_len, seq_len) & key_valid.to(dtype=torch.bool)[:, None, None, :]
        # Keep the diagonal open: a padded row attending to nothing would turn
        # its softmax row into NaN; its output is never consumed, but NaN would
        # still poison the backward pass.
        eye = torch.eye(seq_len, dtype=torch.bool, device=tokens.device)
        allowed = allowed | eye.view(1, 1, seq_len, seq_len)
        with _sdpa_kernel_context(
            backend=self.attention_backend,
            tokens=tokens,
            has_attn_mask=True,
        ):
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed,
                dropout_p=float(self.dropout.p) if self.training else 0.0,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        tokens = tokens + self.dropout(self.out_proj(attended))
        tokens = tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))
        return tokens


class _QueryICLHead(nn.Module):
    """Compressed-row causal in-context classifier (#179; opt-in).

    The TabICL/TabFM repair path for the chance-level cosine query branch: each
    support row is compressed to one embedding (by the joint channel's cell
    encoder — parameter-free row means, see ``_JointSampleEncoder``), a label
    embedding is added, the unlabeled query row is appended, and a small causal
    transformer with RoPE reads the sequence; the query position's output is
    decoded to class logits. Trained with its OWN cross-entropy
    (``query_icl_weight``) so it stays gradient-live even while the zero-init
    ``query_icl_scale`` blend into the legacy logits is still zero.
    """

    def __init__(
        self,
        *,
        enc_dim: int,
        n_layers: int,
        n_heads: int,
        num_classes: int,
        dropout: float,
        attention_backend: str,
        qk_norm: bool,
        max_rows: int = 0,
    ) -> None:
        super().__init__()
        enc_dim = int(enc_dim)
        self.num_classes = max(1, int(num_classes))
        self.max_rows = max(0, int(max_rows))
        # +1 label slot: the "unknown" label carried by the query token (and by
        # any padded row).
        self.label_embed = nn.Embedding(self.num_classes + 1, enc_dim)
        self.input_norm = nn.RMSNorm(enc_dim)
        hidden = max(enc_dim, enc_dim * 2)
        self.blocks = nn.ModuleList(
            [
                _CausalICLBlock(
                    enc_dim,
                    heads=max(1, int(n_heads)),
                    hidden_dim=hidden,
                    dropout=float(dropout),
                    attention_backend=str(attention_backend),
                    qk_norm=bool(qk_norm),
                )
                for _ in range(max(1, int(n_layers)))
            ]
        )
        self.class_head = nn.Sequential(nn.RMSNorm(enc_dim), nn.Linear(enc_dim, self.num_classes))

    def forward(
        self,
        *,
        row_embeddings: torch.Tensor,
        row_valid: torch.Tensor,
        row_labels: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # row_embeddings [B,S,enc]; row_valid [B,S]; row_labels [B,S];
        # query_embedding [B,enc]. Returns query class logits [B, num_classes].
        batch = int(row_embeddings.shape[0])
        rows_avail = int(row_embeddings.shape[1])
        if self.max_rows > 0 and rows_avail > self.max_rows:
            index = torch.linspace(
                0, rows_avail - 1, steps=self.max_rows, device=row_embeddings.device
            ).round().long()
            row_embeddings = row_embeddings.index_select(1, index)
            row_valid = row_valid.index_select(1, index)
            row_labels = row_labels.index_select(1, index)
        row_valid = row_valid.to(dtype=torch.bool)
        labels = row_labels.clamp(min=0, max=self.num_classes - 1)
        labels = torch.where(row_valid, labels, torch.full_like(labels, self.num_classes))
        dtype = row_embeddings.dtype
        tokens = row_embeddings + self.label_embed(labels).to(dtype)
        unknown = torch.full((batch, 1), self.num_classes, dtype=torch.long, device=tokens.device)
        query_token = query_embedding.unsqueeze(1) + self.label_embed(unknown).to(dtype)
        sequence = torch.cat([tokens, query_token], dim=1)
        sequence = self.input_norm(sequence)
        key_valid = torch.cat(
            [row_valid, torch.ones((batch, 1), dtype=torch.bool, device=tokens.device)], dim=1
        )
        for block in self.blocks:
            sequence = block(sequence, key_valid=key_valid)
        return self.class_head(sequence[:, -1])


class TabenticsDiakrinoFSTeacher(nn.Module):
    """Feature-selection teacher that starts from useful DIAKRINO weights only."""

    def __init__(self, config: TabenticsDiakrinoFSTeacherConfig | None = None) -> None:
        _ensure_torch()
        super().__init__()
        self.config = config or TabenticsDiakrinoFSTeacherConfig()
        d_model = int(self.config.d_model)
        architecture_variant = str(self.config.architecture_variant).lower()
        self._use_swiglu_fusion = architecture_variant in {"swiglu_fusion_v2", "swiglu"}
        hidden_dim = max(d_model, d_model * int(self.config.ffn_expansion))
        self.stats_encoder = _FlexibleFeatureStatsEncoder(
            input_dim=max(1, int(self.config.feature_stats_dim)),
            d_model=d_model,
        )
        self.salience_prior = nn.Linear(d_model, 1)
        if self._use_swiglu_fusion:
            self.screening_encoder = nn.Sequential(
                nn.Linear(int(self.config.screening_feature_dim), d_model),
                nn.RMSNorm(d_model),
                _SwiGLUResidualBlock(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout)),
                _SwiGLUResidualBlock(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout)),
            )
        else:
            self.screening_encoder = nn.Sequential(
                nn.Linear(int(self.config.screening_feature_dim), d_model),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, d_model),
                nn.RMSNorm(d_model),
            )
        series_input_dim = (2 * max(1, int(self.config.series_samples))) + 2
        if self._use_swiglu_fusion:
            self.series_distribution_encoder = nn.Sequential(
                nn.Linear(series_input_dim, d_model),
                nn.RMSNorm(d_model),
                _SwiGLUResidualBlock(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout)),
            )
        else:
            self.series_distribution_encoder = nn.Sequential(
                nn.Linear(series_input_dim, d_model),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, d_model),
                nn.RMSNorm(d_model),
            )
        self.series_distribution_gate = nn.Linear(d_model, 1)
        self.input_norm = nn.RMSNorm(d_model)
        if self._use_swiglu_fusion:
            self.fusion_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.RMSNorm(d_model),
                _SwiGLUResidualBlock(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout)),
                _SwiGLUResidualBlock(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout)),
            )
        else:
            self.fusion_encoder = None
        self.feature_position_mode = str(self.config.feature_position_encoding).lower()
        self.position_feature_dim = 1 + 2 * max(1, int(self.config.position_frequency_bands))
        if self.feature_position_mode in {"fourier", "rope", "rope_fourier"}:
            self.position_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(self.position_feature_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.RMSNorm(d_model),
            )
            self.position_encoding_scale = nn.Parameter(torch.tensor(float(self.config.position_encoding_scale_init)))
        else:
            self.position_encoder = None
            self.position_encoding_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        if int(self.config.feature_metadata_dim) > 0:
            self.feature_metadata_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(int(self.config.feature_metadata_dim), d_model),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, d_model),
                nn.RMSNorm(d_model),
            )
            self.feature_metadata_head: nn.Module | None = nn.Linear(d_model, 1)
            self.feature_metadata_scale = nn.Parameter(torch.tensor(float(self.config.feature_metadata_scale_init)))
        else:
            self.feature_metadata_encoder = None
            self.feature_metadata_head = None
            self.feature_metadata_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        class_dim = max(0, int(self.config.sample_class_feature_dim))
        if class_dim > 0:
            self.class_extras_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(class_dim, d_model),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, d_model),
                nn.RMSNorm(d_model),
            )
            self.class_extras_head: nn.Module | None = nn.Linear(d_model, 1)
            self.class_extras_scale = nn.Parameter(torch.tensor(float(self.config.class_extras_scale_init)))
            self.class_extras_logit_scale = nn.Parameter(torch.tensor(float(self.config.class_extras_logit_scale_init)))
        else:
            self.class_extras_encoder = None
            self.class_extras_head = None
            self.class_extras_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.class_extras_logit_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.query_classification_enabled = bool(
            float(self.config.query_classification_weight) > 0.0
            or float(self.config.query_evidence_auxiliary_weight) > 0.0
            or float(self.config.query_selector_relevance_weight) > 0.0
            or float(self.config.query_selector_relevance_listwise_weight) > 0.0
            or float(self.config.query_gate_cardinality_weight) > 0.0
            or float(self.config.query_gate_entropy_weight) > 0.0
            # v-next (#179): the query-ICL head rides the same query-row batch
            # machinery, so its weight keeps the query branch (and its inputs)
            # active even if every legacy query loss is zeroed.
            or float(self.config.query_icl_weight) > 0.0
        )
        n_heads = max(1, min(int(self.config.n_heads), d_model))
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        if int(self.config.context_layers) > 0:
            if self._use_swiglu_fusion:
                self.context_encoder = _SwiGLUContextEncoder(
                    d_model=d_model,
                    n_heads=n_heads,
                    hidden_dim=hidden_dim,
                    dropout=float(self.config.dropout),
                    num_layers=int(self.config.context_layers),
                    attention_backend=str(self.config.attention_backend),
                    qk_norm=bool(self.config.attn_qk_norm),
                )
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=max(d_model, d_model * int(self.config.ffn_expansion)),
                    dropout=float(self.config.dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.context_encoder: nn.Module | None = nn.TransformerEncoder(
                    layer,
                    num_layers=int(self.config.context_layers),
                )
        else:
            self.context_encoder = None
        if self._use_swiglu_fusion:
            self.screening_head = _SwiGLUScoringHead(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout))
            self.series_head = _SwiGLUScoringHead(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout))
            self.residual_head = _SwiGLUScoringHead(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout))
            self.selector_gate_head = _SwiGLUScoringHead(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout))
        else:
            self.screening_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, 1),
            )
            self.series_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, 1),
            )
            self.residual_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, 1),
            )
            self.selector_gate_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, 1),
            )
        if self._use_swiglu_fusion:
            self.local_residual_head = _SwiGLUScoringHead(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout))
        else:
            self.local_residual_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, 1),
            )
        if int(self.config.refiner_steps) > 0:
            self.refiner: nn.Module | None = _TinySalienceRefiner(
                d_model=d_model,
                n_heads=n_heads,
                dropout=float(self.config.dropout),
                ffn_expansion=int(self.config.ffn_expansion),
                attention_backend=str(self.config.attention_backend),
                use_swiglu=bool(self._use_swiglu_fusion),
                qk_norm=bool(self.config.attn_qk_norm),
            )
        else:
            self.refiner = None
        # v-next (#179): joint_sample_size <= 0 now means "ALL support rows"
        # (full-row trunk), so the size no longer gates construction — only the
        # mode does. Historical configs always carry mode="off" or size>0, so
        # nothing existing changes.
        if str(self.config.joint_sample_mode).lower() != "off":
            # The query path is only meaningful (and only DDP-safe under
            # find_unused_parameters=False) when the query head is also enabled,
            # since its params are then used iff a query row is present — the same
            # condition as the existing query-classification head.
            joint_include_query = bool(self.config.joint_sample_include_query) and self.query_classification_enabled
            self.joint_sample_encoder: nn.Module | None = _JointSampleEncoder(
                d_model=d_model,
                enc_dim=max(1, int(self.config.joint_sample_width)),
                n_heads=max(1, int(self.config.joint_sample_heads)),
                n_layers=max(1, int(self.config.joint_sample_layers)),
                induced_points=max(1, int(self.config.joint_sample_induced_points)),
                num_classes=int(self.config.max_classes),
                dropout=float(self.config.dropout),
                attention_backend=str(self.config.attention_backend),
                qk_norm=bool(self.config.attn_qk_norm),
                eps=float(self.config.eps),
                include_query=joint_include_query,
                fourier_bands=max(0, int(self.config.joint_cell_fourier_bands)),
            )
            self.joint_sample_scale = nn.Parameter(torch.tensor(float(self.config.joint_sample_scale_init)))
            if joint_include_query:
                self.joint_sample_query_scale = nn.Parameter(torch.tensor(float(self.config.joint_sample_query_scale_init)))
            else:
                self.joint_sample_query_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        else:
            self.joint_sample_encoder = None
            self.joint_sample_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.joint_sample_query_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        # --- v-next query ICL head (#179): compressed-row causal in-context
        # classifier. Gated on the SAME condition as the legacy query branch so
        # its parameters are exercised on exactly the same steps (DDP
        # static-graph safety), and it structurally requires the joint channel
        # (it consumes the joint cell encoder's row embeddings). ---
        query_icl_mode = str(self.config.query_icl_mode).lower()
        if query_icl_mode != "off" and self.joint_sample_encoder is None:
            raise ValueError(
                "query_icl_mode requires joint_sample_mode != 'off': the ICL head reads the joint cell encoder's per-row embeddings."
            )
        if query_icl_mode != "off" and self.query_classification_enabled:
            self.query_icl_head: nn.Module | None = _QueryICLHead(
                enc_dim=max(1, int(self.config.joint_sample_width)),
                n_layers=max(1, int(self.config.query_icl_layers)),
                n_heads=max(1, int(self.config.query_icl_heads)),
                num_classes=int(self.config.max_classes),
                dropout=float(self.config.dropout),
                attention_backend=str(self.config.attention_backend),
                qk_norm=bool(self.config.attn_qk_norm),
                max_rows=max(0, int(self.config.query_icl_max_rows)),
            )
            self.query_icl_scale = nn.Parameter(torch.tensor(float(self.config.query_icl_scale_init)))
        else:
            self.query_icl_head = None
            self.query_icl_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        # --- v-next pairwise redundancy head (#179 → #164/#165): low-rank
        # bilinear pair scorer over feature tokens. ---
        if str(self.config.redundancy_head_mode).lower() != "off":
            redundancy_rank = max(8, int(self.config.redundancy_head_rank))
            self.redundancy_query_proj: nn.Module | None = nn.Linear(d_model, redundancy_rank)
            self.redundancy_key_proj: nn.Module | None = nn.Linear(d_model, redundancy_rank)
            self.redundancy_bias = nn.Parameter(torch.tensor(0.0))
            self.redundancy_rank = redundancy_rank
        else:
            self.redundancy_query_proj = None
            self.redundancy_key_proj = None
            self.redundancy_bias = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.redundancy_rank = 0
        # --- v-next R1 conformal selection head (#179): trained conformity
        # score + learnable global threshold. ---
        if str(self.config.conformal_head_mode).lower() != "off":
            if self._use_swiglu_fusion:
                self.conformal_head: nn.Module | None = _SwiGLUScoringHead(
                    d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout)
                )
            else:
                self.conformal_head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    nn.Dropout(float(self.config.dropout)),
                    nn.Linear(d_model, 1),
                )
            self.conformal_threshold = nn.Parameter(torch.tensor(0.0))
        else:
            self.conformal_head = None
            self.conformal_threshold = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.support_class_head = nn.Linear(d_model, int(self.config.max_classes))
        self.support_class_bias = nn.Parameter(torch.zeros(int(self.config.max_classes)))
        if self.query_classification_enabled:
            query_class_dim = max(1, int(self.config.query_class_stats_dim))
            query_value_dim = max(1, int(self.config.query_value_dim))
            relative_dim = max(1, int(self.config.query_relative_feature_dim))
            if self._use_swiglu_fusion:
                self.query_class_stats_encoder: nn.Module | None = nn.Sequential(
                    nn.Linear(query_class_dim, d_model),
                    nn.RMSNorm(d_model),
                    _SwiGLUResidualBlock(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout)),
                )
                self.query_value_encoder: nn.Module | None = nn.Sequential(
                    nn.Linear(query_value_dim, d_model),
                    nn.RMSNorm(d_model),
                    _SwiGLUResidualBlock(d_model, hidden_dim=hidden_dim, dropout=float(self.config.dropout)),
                )
            else:
                self.query_class_stats_encoder = nn.Sequential(
                    nn.Linear(query_class_dim, d_model),
                    nn.SiLU(),
                    nn.Dropout(float(self.config.dropout)),
                    nn.Linear(d_model, d_model),
                    nn.RMSNorm(d_model),
                )
                self.query_value_encoder = nn.Sequential(
                    nn.Linear(query_value_dim, d_model),
                    nn.SiLU(),
                    nn.Dropout(float(self.config.dropout)),
                    nn.Linear(d_model, d_model),
                    nn.RMSNorm(d_model),
                )
            self.query_projection: nn.Module | None = nn.Linear(d_model, d_model, bias=False)
            self.query_class_projection: nn.Module | None = nn.Linear(d_model, d_model, bias=False)
            if str(self.config.interaction_mode).lower() == "cosine":
                self.query_interaction: nn.Module | None = _BalancedCosineInteraction(
                    d_model=d_model,
                    heads=int(self.config.interaction_heads) or n_heads,
                    temp_init=float(self.config.interaction_temp_init),
                    temp_learnable=bool(self.config.interaction_temp_learnable),
                    scale_channel=bool(self.config.interaction_scale_channel),
                    eps=float(self.config.eps),
                )
            else:
                self.query_interaction = None
            self.query_relative_evidence: nn.Module | None = nn.Sequential(
                nn.Linear(relative_dim, max(16, d_model // 4)),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(max(16, d_model // 4), 1),
            )
            if self._use_swiglu_fusion:
                self.query_feature_class_gate_head: nn.Module | None = _SwiGLUScoringHead(
                    d_model,
                    hidden_dim=hidden_dim,
                    dropout=float(self.config.dropout),
                )
            else:
                self.query_feature_class_gate_head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    nn.Dropout(float(self.config.dropout)),
                    nn.Linear(d_model, 1),
                )
            gate_bias = float(self.config.query_gate_bias_init)
            if bool(self.config.query_gate_bias_from_target):
                gate_bias += _probability_logit(
                    float(self.config.query_gate_target_fraction),
                    eps=float(self.config.eps),
                )
            output = getattr(self.query_feature_class_gate_head, "output", None)
            if output is not None and getattr(output, "bias", None) is not None:
                nn.init.constant_(output.bias, gate_bias)
            elif isinstance(self.query_feature_class_gate_head, nn.Sequential):
                last = self.query_feature_class_gate_head[-1]
                if isinstance(last, nn.Linear):
                    nn.init.constant_(last.bias, gate_bias)
            self.query_class_hidden_projection: nn.Module | None = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, d_model))
            self.query_global_projection: nn.Module | None = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, d_model))
            self.query_class_logit_head: nn.Module | None = nn.Sequential(nn.RMSNorm(d_model), nn.Linear(d_model, 1))
            self.query_feature_gate_scale = nn.Parameter(torch.tensor(float(self.config.query_feature_gate_scale_init)))
            self.query_evidence_scale = nn.Parameter(torch.tensor(float(self.config.query_evidence_scale_init)))
            self.query_class_prior_scale = nn.Parameter(torch.tensor(float(self.config.query_class_prior_scale_init)))
        else:
            self.query_class_stats_encoder = None
            self.query_value_encoder = None
            self.query_projection = None
            self.query_class_projection = None
            self.query_interaction = None
            self.query_relative_evidence = None
            self.query_feature_class_gate_head = None
            self.query_class_hidden_projection = None
            self.query_global_projection = None
            self.query_class_logit_head = None
            self.query_feature_gate_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.query_evidence_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.query_class_prior_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        recon_hidden = max(d_model, d_model * max(1, int(self.config.reconstruction_hidden_multiplier)))
        self.reconstruction_head = nn.Sequential(
            nn.Linear(3 * d_model, recon_hidden),
            nn.SiLU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(recon_hidden, 1),
        )
        population_dim = max(0, int(self.config.population_reconstruction_dim))
        if population_dim > 0:
            self.population_reconstruction_head: nn.Module | None = nn.Sequential(
                nn.Linear(d_model, recon_hidden),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(recon_hidden, population_dim),
            )
        else:
            self.population_reconstruction_head = None
        population_class_dim = max(0, int(self.config.population_class_reconstruction_dim))
        if population_class_dim > 0:
            self.population_class_reconstruction_head: nn.Module | None = nn.Sequential(
                nn.Linear(d_model, recon_hidden),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(recon_hidden, population_class_dim),
            )
        else:
            self.population_class_reconstruction_head = None
        family_classes = max(0, int(self.config.population_family_classes))
        if family_classes > 0:
            self.population_family_head: nn.Module | None = nn.Sequential(
                nn.Linear(d_model, recon_hidden),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(recon_hidden, family_classes),
            )
        else:
            self.population_family_head = None
        support_type_classes = max(0, int(self.config.population_support_type_classes))
        if support_type_classes > 0:
            self.population_support_type_head: nn.Module | None = nn.Sequential(
                nn.Linear(d_model, recon_hidden),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(recon_hidden, support_type_classes),
            )
        else:
            self.population_support_type_head = None
        population_param_dim = max(0, int(self.config.population_param_dim))
        if population_param_dim > 0:
            self.population_param_head: nn.Module | None = nn.Sequential(
                nn.Linear(d_model, recon_hidden),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(recon_hidden, population_param_dim),
            )
            self.population_param_logvar_head: nn.Module | None = nn.Sequential(
                nn.Linear(d_model, recon_hidden),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(recon_hidden, population_param_dim),
            )
        else:
            self.population_param_head = None
            self.population_param_logvar_head = None
        population_dependency_dim = max(0, int(self.config.population_dependency_dim))
        if population_dependency_dim > 0:
            self.population_dependency_head: nn.Module | None = nn.Sequential(
                nn.Linear(d_model, recon_hidden),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(recon_hidden, population_dependency_dim),
            )
        else:
            self.population_dependency_head = None
        population_coeff_dim = max(0, int(self.config.population_coeff_dim))
        if population_coeff_dim > 0:
            self.population_coeff_head: nn.Module | None = nn.Sequential(
                nn.Linear(d_model, recon_hidden),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(recon_hidden, population_coeff_dim),
            )
        else:
            self.population_coeff_head = None
        dependence_type_classes = max(0, int(self.config.population_dependence_type_classes))
        self.population_dependence_type_head = (
            nn.Sequential(nn.Linear(d_model, recon_hidden), nn.SiLU(), nn.Dropout(float(self.config.dropout)), nn.Linear(recon_hidden, dependence_type_classes))
            if dependence_type_classes > 0
            else None
        )
        task_family_classes = max(0, int(self.config.population_task_family_classes))
        self.population_task_family_head = (
            nn.Sequential(nn.Linear(d_model, recon_hidden), nn.SiLU(), nn.Dropout(float(self.config.dropout)), nn.Linear(recon_hidden, task_family_classes))
            if task_family_classes > 0
            else None
        )
        task_variant_classes = max(0, int(self.config.population_task_variant_classes))
        self.population_task_variant_head = (
            nn.Sequential(nn.Linear(d_model, recon_hidden), nn.SiLU(), nn.Dropout(float(self.config.dropout)), nn.Linear(recon_hidden, task_variant_classes))
            if task_variant_classes > 0
            else None
        )
        population_conditioning_mode = str(self.config.population_conditioning_mode).lower()
        population_conditioning_dim = 0
        if population_conditioning_mode not in {"off", "none", "disabled"}:
            population_conditioning_dim = (
                population_dim
                + population_class_dim
                + family_classes
                + support_type_classes
                + population_param_dim
                + population_dependency_dim
                + population_coeff_dim
                + dependence_type_classes
                + task_family_classes
                + task_variant_classes
            )
        self.population_conditioning_dim = int(population_conditioning_dim)
        if self.population_conditioning_dim > 0:
            self.population_conditioning_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(self.population_conditioning_dim, d_model),
                nn.SiLU(),
                nn.Dropout(float(self.config.dropout)),
                nn.Linear(d_model, d_model),
                nn.RMSNorm(d_model),
            )
            self.population_conditioning_token_head: nn.Module | None = nn.Linear(d_model, d_model)
            self.population_conditioning_logit_head: nn.Module | None = nn.Linear(d_model, 1)
            self.population_conditioning_selector_head: nn.Module | None = nn.Linear(d_model, 1)
            for head in (
                self.population_conditioning_token_head,
                self.population_conditioning_logit_head,
                self.population_conditioning_selector_head,
            ):
                if isinstance(head, nn.Linear):
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
            self.population_conditioning_scale = nn.Parameter(
                torch.tensor(float(self.config.population_conditioning_scale_init))
            )
        else:
            self.population_conditioning_encoder = None
            self.population_conditioning_token_head = None
            self.population_conditioning_logit_head = None
            self.population_conditioning_selector_head = None
            self.population_conditioning_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        population_decision_layers = max(0, int(self.config.population_decision_layers))
        if self.population_conditioning_dim > 0 and population_decision_layers > 0:
            if self._use_swiglu_fusion:
                self.population_decision_encoder: nn.Module | None = _SwiGLUContextEncoder(
                    d_model=d_model,
                    n_heads=n_heads,
                    hidden_dim=hidden_dim,
                    dropout=float(self.config.dropout),
                    num_layers=population_decision_layers,
                    attention_backend=str(self.config.attention_backend),
                    qk_norm=bool(self.config.attn_qk_norm),
                )
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=max(d_model, d_model * int(self.config.ffn_expansion)),
                    dropout=float(self.config.dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.population_decision_encoder = nn.TransformerEncoder(layer, num_layers=population_decision_layers)
            self.population_decision_scale = nn.Parameter(torch.tensor(float(self.config.population_decision_scale_init)))
        else:
            self.population_decision_encoder = None
            self.population_decision_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.prior_scale = nn.Parameter(torch.tensor(float(self.config.prior_scale_init)))
        self.screening_scale = nn.Parameter(torch.tensor(float(self.config.screening_scale_init)))
        self.series_scale = nn.Parameter(torch.tensor(float(self.config.series_scale_init)))
        self.residual_scale = nn.Parameter(torch.tensor(float(self.config.residual_scale_init)))
        self.fusion_scale = nn.Parameter(torch.tensor(float(self.config.fusion_scale_init)))
        self.refiner_scale = nn.Parameter(torch.tensor(float(self.config.refiner_scale_init)))
        self.selector_logit_scale = nn.Parameter(torch.tensor(float(self.config.selector_logit_scale_init)))
        self.local_residual_scale = nn.Parameter(torch.tensor(float(self.config.local_residual_scale_init)))
        self.calibration_bias = nn.Parameter(torch.tensor(float(self.config.calibration_bias_init)))
        # v-next (#179): joint-contribution warmup ramp. NON-PERSISTENT buffer
        # (absent from the state dict, so checkpoints/warm-starts are
        # unaffected); defaults to 1.0 so inference and non-ramped training are
        # bit-identical. The trainer drives it via
        # ``set_joint_contribution_progress`` — the corrected #111/#141
        # mitigation: scale_init>0 unblocks out_proj at a fixed gain, and this
        # ramp grows that fixed-gain injection gently into the warm-started
        # minimum (FS_TEACHER_VNEXT_PLAN.md §3.1).
        self.register_buffer("joint_contribution_progress", torch.tensor(1.0), persistent=False)
        self.load_report: JsonDict = {}

    def set_joint_contribution_progress(self, progress: float) -> None:
        """Trainer hook (#179): set the 0→1 warmup ramp multiplying the joint
        channel's injections (selection residual and query-row residual)."""
        with torch.no_grad():
            self.joint_contribution_progress.fill_(float(min(1.0, max(0.0, float(progress)))))

    def _query_relative_channels(
        self,
        batch: TabenticsDiakrinoFSTeacherBatch,
        class_stats: torch.Tensor,
    ) -> torch.Tensor:
        assert batch.query_values is not None
        assert batch.query_mask is not None
        eps = float(self.config.eps)
        values = torch.nan_to_num(batch.query_values, nan=0.0, posinf=0.0, neginf=0.0)
        observed = (~batch.query_mask.to(dtype=torch.bool)).to(dtype=values.dtype)
        query = values.unsqueeze(-1).expand(-1, -1, int(class_stats.shape[2]))
        obs = observed.unsqueeze(-1).expand_as(query)
        mean = class_stats[..., 1] if int(class_stats.shape[-1]) > 1 else torch.zeros_like(query)
        std = class_stats[..., 2].abs().clamp(min=eps) if int(class_stats.shape[-1]) > 2 else torch.ones_like(query)
        min_value = class_stats[..., 3] if int(class_stats.shape[-1]) > 3 else mean - std
        max_value = class_stats[..., 4] if int(class_stats.shape[-1]) > 4 else mean + std
        median = class_stats[..., 21] if int(class_stats.shape[-1]) > 21 else mean
        iqr = class_stats[..., 22].abs().clamp(min=eps) if int(class_stats.shape[-1]) > 22 else std
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
        return torch.nan_to_num(
            _match_last_dim(channels, int(self.config.query_relative_feature_dim)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _query_class_prior_logits(
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
        masked = torch.where(class_valid, log_counts, torch.full_like(log_counts, -30.0))
        return torch.where(class_valid, log_counts - torch.logsumexp(masked, dim=1, keepdim=True), torch.zeros_like(log_counts))

    def _position_encode_query_tokens(
        self,
        query_tokens: torch.Tensor,
        *,
        feature_positions: torch.Tensor | None,
        feature_valid: torch.Tensor,
    ) -> torch.Tensor:
        if feature_positions is None:
            return torch.where(feature_valid.unsqueeze(-1), query_tokens, torch.zeros_like(query_tokens))
        mode = str(self.config.feature_position_encoding).lower()
        tokens = query_tokens
        if self.position_encoder is not None and mode in {"fourier", "rope", "rope_fourier"}:
            position_features = _fourier_position_features(
                feature_positions.to(dtype=tokens.dtype),
                bands=int(self.config.position_frequency_bands),
            ).to(dtype=tokens.dtype)
            position_latent = self.position_encoder(position_features)
            position_latent = torch.where(feature_valid.unsqueeze(-1), position_latent, torch.zeros_like(position_latent))
            tokens = tokens + self.position_encoding_scale.to(dtype=tokens.dtype) * position_latent
        if mode in {"rope", "rope_fourier"}:
            tokens = _apply_feature_rope(tokens, feature_positions.to(dtype=tokens.dtype))
        return torch.where(feature_valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))

    def _selector_gate_values(self, selector_gate_logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        selector_temperature = max(float(self.config.selector_temperature), float(self.config.eps))
        if bool(self.training) and bool(self.config.selector_stochastic) and selector_temperature > 0.0:
            uniform = torch.rand_like(selector_gate_logits).clamp(
                min=float(self.config.eps),
                max=1.0 - float(self.config.eps),
            )
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
            selector_gate_values = torch.sigmoid((selector_gate_logits + logistic_noise) / selector_temperature)
        else:
            selector_gate_values = torch.sigmoid(selector_gate_logits / selector_temperature)
        return torch.where(valid_mask, selector_gate_values, torch.zeros_like(selector_gate_values))

    def _population_conditioning_latent(
        self,
        *,
        template_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        population_reconstruction_predictions: torch.Tensor | None,
        population_class_reconstruction_predictions: torch.Tensor | None,
        population_family_logits: torch.Tensor | None,
        population_support_type_logits: torch.Tensor | None,
        population_param_predictions: torch.Tensor | None,
        population_dependency_predictions: torch.Tensor | None,
        population_coeff_predictions: torch.Tensor | None,
        population_dependence_type_logits: torch.Tensor | None,
        population_task_family_logits: torch.Tensor | None,
        population_task_variant_logits: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.population_conditioning_encoder is None or self.population_conditioning_dim <= 0:
            return None
        batch_size, feature_count = int(template_tokens.shape[0]), int(template_tokens.shape[1])
        dtype = template_tokens.dtype
        device = template_tokens.device
        clip_value = float(self.config.clip_value)
        parts: list[torch.Tensor] = []

        def per_feature_part(value: torch.Tensor | None, dim: int) -> None:
            if dim <= 0:
                return
            if value is None:
                part = torch.zeros((batch_size, feature_count, dim), dtype=dtype, device=device)
            else:
                part = _match_last_dim(value.to(dtype=dtype, device=device), dim)
                part = torch.nan_to_num(part, nan=0.0, posinf=clip_value, neginf=-clip_value)
                part = part.clamp(min=-clip_value, max=clip_value)
            parts.append(part)

        def pooled_part(value: torch.Tensor | None, dim: int) -> None:
            if dim <= 0:
                return
            if value is None:
                part = torch.zeros((batch_size, feature_count, dim), dtype=dtype, device=device)
            else:
                pooled = _match_last_dim(value.to(dtype=dtype, device=device), dim)
                pooled = torch.nan_to_num(pooled, nan=0.0, posinf=clip_value, neginf=-clip_value)
                pooled = pooled.clamp(min=-clip_value, max=clip_value)
                part = pooled.unsqueeze(1).expand(-1, feature_count, -1)
            parts.append(part)

        per_feature_part(population_reconstruction_predictions, max(0, int(self.config.population_reconstruction_dim)))
        per_feature_part(
            population_class_reconstruction_predictions,
            max(0, int(self.config.population_class_reconstruction_dim)),
        )
        per_feature_part(population_family_logits, max(0, int(self.config.population_family_classes)))
        per_feature_part(population_support_type_logits, max(0, int(self.config.population_support_type_classes)))
        per_feature_part(population_param_predictions, max(0, int(self.config.population_param_dim)))
        per_feature_part(population_dependency_predictions, max(0, int(self.config.population_dependency_dim)))
        per_feature_part(population_coeff_predictions, max(0, int(self.config.population_coeff_dim)))
        pooled_part(population_dependence_type_logits, max(0, int(self.config.population_dependence_type_classes)))
        pooled_part(population_task_family_logits, max(0, int(self.config.population_task_family_classes)))
        pooled_part(population_task_variant_logits, max(0, int(self.config.population_task_variant_classes)))
        if not parts:
            return None
        conditioning_features = torch.cat(parts, dim=-1)
        conditioning_features = _match_last_dim(conditioning_features, int(self.population_conditioning_dim))
        conditioning_features = torch.where(
            valid_mask.unsqueeze(-1),
            conditioning_features,
            torch.zeros_like(conditioning_features),
        )
        if bool(self.config.population_conditioning_detach):
            conditioning_features = conditioning_features.detach()
        latent = self.population_conditioning_encoder(conditioning_features)
        return torch.where(valid_mask.unsqueeze(-1), latent, torch.zeros_like(latent))

    def _population_decision_tokens(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if self.population_decision_encoder is None:
            return tokens
        if isinstance(self.population_decision_encoder, _SwiGLUContextEncoder):
            encoded = self.population_decision_encoder(tokens, valid_mask=valid_mask)
        else:
            with _sdpa_kernel_context(
                backend=str(self.config.attention_backend),
                tokens=tokens,
                has_attn_mask=not bool(torch.all(valid_mask).detach().cpu()),
            ):
                encoded = self.population_decision_encoder(tokens, src_key_padding_mask=~valid_mask)
        scale = self.population_decision_scale.to(dtype=tokens.dtype)
        mixed = tokens + scale * (encoded - tokens)
        return torch.where(valid_mask.unsqueeze(-1), mixed, torch.zeros_like(mixed))

    def _query_classification_outputs(
        self,
        *,
        tokens: torch.Tensor,
        selector_logits: torch.Tensor,
        batch: TabenticsDiakrinoFSTeacherBatch,
        feature_valid: torch.Tensor,
        feature_positions: torch.Tensor | None,
        joint_query_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if (
            not self.query_classification_enabled
            or self.query_class_stats_encoder is None
            or self.query_value_encoder is None
            or self.query_projection is None
            or self.query_class_projection is None
            or self.query_relative_evidence is None
            or self.query_feature_class_gate_head is None
            or self.query_class_hidden_projection is None
            or self.query_global_projection is None
            or self.query_class_logit_head is None
            or batch.query_values is None
            or batch.query_mask is None
            or batch.query_class_stats is None
            or batch.query_class_stats_valid is None
        ):
            return None, None, None, None
        max_classes = max(1, int(self.config.max_classes))
        class_stats = _match_last_dim(
            torch.nan_to_num(batch.query_class_stats.to(dtype=tokens.dtype), nan=0.0, posinf=0.0, neginf=0.0),
            int(self.config.query_class_stats_dim),
        )
        class_count = int(class_stats.shape[2])
        if class_count > max_classes:
            class_stats = class_stats[:, :, :max_classes]
        elif class_count < max_classes:
            class_stats = F.pad(class_stats, (0, 0, 0, max_classes - class_count))
        class_stats_valid = batch.query_class_stats_valid.to(dtype=torch.bool)
        if int(class_stats_valid.shape[-1]) > max_classes:
            class_stats_valid = class_stats_valid[..., :max_classes]
        elif int(class_stats_valid.shape[-1]) < max_classes:
            class_stats_valid = F.pad(class_stats_valid, (0, max_classes - int(class_stats_valid.shape[-1])))
        class_valid = (
            batch.query_class_valid.to(dtype=torch.bool)
            if batch.query_class_valid is not None
            else class_stats_valid.any(dim=1)
        )
        if int(class_valid.shape[-1]) > max_classes:
            class_valid = class_valid[..., :max_classes]
        elif int(class_valid.shape[-1]) < max_classes:
            class_valid = F.pad(class_valid, (0, max_classes - int(class_valid.shape[-1])))
        class_stats_valid = class_stats_valid & feature_valid.unsqueeze(-1) & class_valid.unsqueeze(1)

        class_tokens = self.query_class_stats_encoder(class_stats)
        class_tokens = torch.where(class_stats_valid.unsqueeze(-1), class_tokens, torch.zeros_like(class_tokens))
        values = torch.nan_to_num(batch.query_values.to(dtype=tokens.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        observed = (~batch.query_mask.to(dtype=torch.bool)).to(dtype=tokens.dtype)
        query_features = torch.stack(
            [
                torch.clamp(values, -float(self.config.clip_value), float(self.config.clip_value)),
                observed,
                values.abs().clamp(max=float(self.config.clip_value)),
                1.0 - observed,
            ],
            dim=-1,
        )
        query_tokens = self.query_value_encoder(_match_last_dim(query_features, int(self.config.query_value_dim)))
        query_tokens = self._position_encode_query_tokens(
            query_tokens,
            feature_positions=feature_positions,
            feature_valid=feature_valid,
        )
        class_feature_tokens = class_tokens + tokens.unsqueeze(2)
        class_feature_tokens = torch.where(class_stats_valid.unsqueeze(-1), class_feature_tokens, torch.zeros_like(class_feature_tokens))
        query_feature_tokens = torch.where(feature_valid.unsqueeze(-1), query_tokens + tokens, torch.zeros_like(tokens))
        if joint_query_residual is not None:
            # Per-feature query-row interaction read (zero-init scale -> no-op
            # until learned); already masked to valid features in the encoder.
            query_feature_tokens = query_feature_tokens + self.joint_sample_query_scale.to(dtype=tokens.dtype) * joint_query_residual.to(dtype=tokens.dtype)
        query_feature_tokens = self._population_decision_tokens(query_feature_tokens, feature_valid)

        query_projected = self.query_projection(query_feature_tokens)
        class_projected = self.query_class_projection(class_feature_tokens)
        if self.query_interaction is not None:
            bilinear = self.query_interaction(query_projected, class_projected)
        else:
            bilinear = torch.einsum("bfd,bfkd->bfk", query_projected, class_projected)
            bilinear = bilinear / math.sqrt(max(1, int(self.config.d_model)))
        relative_logits = self.query_relative_evidence(self._query_relative_channels(batch, class_stats)).squeeze(-1)
        feature_class_evidence = self.query_evidence_scale.to(dtype=tokens.dtype) * bilinear + relative_logits
        gate_logits = (
            self.query_feature_class_gate_head(class_feature_tokens).squeeze(-1)
            + self.query_feature_gate_scale.to(dtype=tokens.dtype) * selector_logits.unsqueeze(-1)
        )
        valid_fk = class_stats_valid & feature_valid.unsqueeze(-1) & class_valid.unsqueeze(1)
        gates = torch.where(valid_fk, torch.sigmoid(gate_logits), torch.zeros_like(gate_logits))
        feature_class_evidence = torch.where(valid_fk, feature_class_evidence, torch.zeros_like(feature_class_evidence))

        gate_mass = gates.sum(dim=1).clamp(min=1.0)
        pooled_evidence = (feature_class_evidence * gates).sum(dim=1) / torch.sqrt(gate_mass)
        class_hidden = (class_feature_tokens * gates.unsqueeze(-1)).sum(dim=1) / gate_mass.unsqueeze(-1)
        class_hidden = self.query_class_hidden_projection(class_hidden)
        query_global = _masked_token_mean(query_feature_tokens, feature_valid, eps=float(self.config.eps))
        query_global = self.query_global_projection(query_global)
        logits = pooled_evidence + self.query_class_logit_head(class_hidden + query_global.unsqueeze(1)).squeeze(-1)
        prior = self._query_class_prior_logits(
            class_stats,
            class_stats_valid=class_stats_valid,
            feature_valid=feature_valid,
            class_valid=class_valid,
        )
        logits = logits + self.query_class_prior_scale.to(dtype=tokens.dtype) * prior
        logits = torch.where(class_valid, logits, torch.full_like(logits, -30.0))
        return logits, class_valid, feature_class_evidence, gates

    def forward(
        self,
        batch: TabenticsDiakrinoFSTeacherBatch,
        *,
        refiner_steps: int | None = None,
    ) -> TabenticsDiakrinoFSTeacherOutputs:
        valid_mask = batch.feature_valid_mask.to(dtype=torch.bool)
        if batch.feature_stats_input is not None:
            all_stats = batch.feature_stats_input.to(dtype=batch.support.dtype)
        else:
            feature_stats = _compute_feature_stats(
                batch.support,
                support_mask=batch.support_mask,
                support_valid=batch.support_valid,
            )
            class_stats = _compute_class_conditional_stats(
                batch.support,
                support_mask=batch.support_mask,
                support_valid=batch.support_valid,
                support_labels=batch.support_labels,
                num_classes_max=int(self.config.max_classes),
            )
            all_stats = torch.cat([feature_stats, class_stats], dim=-1)
        all_stats = _match_last_dim(all_stats, int(self.config.feature_stats_dim))
        all_stats = torch.where(valid_mask.unsqueeze(-1), all_stats, torch.zeros_like(all_stats))
        if batch.screening_features_input is not None:
            screening_features = batch.screening_features_input.to(dtype=batch.support.dtype)
            screening_features = _match_last_dim(screening_features, int(self.config.screening_feature_dim))
            screening_features = torch.where(
                valid_mask.unsqueeze(-1),
                screening_features,
                torch.zeros_like(screening_features),
            )
        else:
            screening_features = compute_fs_screening_features(
                _match_last_dim(all_stats, 10),
                feature_valid_mask=valid_mask,
                eps=float(self.config.eps),
            )
            screening_features = _match_last_dim(screening_features, int(self.config.screening_feature_dim))
        stats_latent = self.stats_encoder(all_stats)
        screening_latent = self.screening_encoder(screening_features)
        class_extras_latent = torch.zeros_like(stats_latent)
        class_extras_logits = torch.zeros(valid_mask.shape, dtype=stats_latent.dtype, device=stats_latent.device)
        if self.class_extras_encoder is not None:
            class_dim = max(1, int(self.config.sample_class_feature_dim))
            if batch.sample_class_features_input is None:
                class_input = torch.zeros(
                    (*valid_mask.shape, class_dim),
                    dtype=batch.support.dtype,
                    device=batch.support.device,
                )
            else:
                class_input = batch.sample_class_features_input.to(dtype=batch.support.dtype)
            class_input = _match_last_dim(class_input, class_dim)
            class_input = torch.where(valid_mask.unsqueeze(-1), class_input, torch.zeros_like(class_input))
            class_extras_latent = self.class_extras_encoder(class_input)
            class_extras_latent = torch.where(valid_mask.unsqueeze(-1), class_extras_latent, torch.zeros_like(class_extras_latent))
            if self.class_extras_head is not None:
                class_extras_logits = self.class_extras_head(class_extras_latent).squeeze(-1)
        series_latent = torch.zeros_like(stats_latent)
        if bool(self.config.use_distribution_series) and int(self.config.series_samples) > 0:
            if batch.distribution_series_input is not None and batch.distribution_series_valid is not None:
                series_input = batch.distribution_series_input
                series_valid = batch.distribution_series_valid
            else:
                series_input, series_valid = compute_distribution_series(
                    batch.support,
                    support_mask=batch.support_mask,
                    support_valid=batch.support_valid,
                    support_labels=batch.support_labels,
                    feature_valid_mask=valid_mask,
                    max_classes=int(self.config.max_classes),
                    series_samples=int(self.config.series_samples),
                )
            distribution_latents = self.series_distribution_encoder(series_input)
            series_valid = series_valid & valid_mask.unsqueeze(-1)
            gate_logits = self.series_distribution_gate(distribution_latents).squeeze(-1)
            gate_logits = gate_logits.masked_fill(~series_valid, -30.0)
            weights = torch.softmax(gate_logits, dim=2)
            weights = torch.where(series_valid, weights, torch.zeros_like(weights))
            weight_sum = weights.sum(dim=2, keepdim=True).clamp(min=float(self.config.eps))
            weights = weights / weight_sum
            series_latent = (distribution_latents * weights.unsqueeze(-1)).sum(dim=2)
            series_latent = torch.where(valid_mask.unsqueeze(-1), series_latent, torch.zeros_like(series_latent))
        token_sum = stats_latent + screening_latent + series_latent + self.class_extras_scale * class_extras_latent
        if self.fusion_encoder is not None:
            fused = self.fusion_encoder(torch.cat([stats_latent, screening_latent, series_latent], dim=-1))
            token_sum = token_sum + self.fusion_scale * fused
        joint_query_residual: torch.Tensor | None = None
        joint_row_means: torch.Tensor | None = None
        joint_row_valid: torch.Tensor | None = None
        joint_row_labels: torch.Tensor | None = None
        joint_support_summary: torch.Tensor | None = None
        joint_feature_tokens: torch.Tensor | None = None
        if self.joint_sample_encoder is not None:
            support = batch.support
            rows_avail = int(support.shape[1])
            # v-next (#179): size <= 0 => FULL-ROW trunk (all support rows).
            configured_rows = int(self.config.joint_sample_size)
            take = rows_avail if configured_rows <= 0 else max(1, min(configured_rows, rows_avail))
            if take >= rows_avail:
                row_index = torch.arange(rows_avail, device=support.device)
            else:
                # Evenly spaced stride across the support set: deterministic
                # (resume-stable) and spreads across class blocks regardless of
                # row ordering, without host-side selection logic.
                row_index = (
                    torch.linspace(0, rows_avail - 1, steps=take, device=support.device).round().long()
                )
            joint_row_valid = batch.support_valid.index_select(1, row_index).to(dtype=torch.bool)
            joint_row_labels = batch.support_labels.index_select(1, row_index).to(dtype=torch.long)
            # Keep a reference to the exact feature tokens the support cells are
            # embedded against, so the query-ICL row embedding (computed later)
            # uses the SAME per-feature content identities.
            joint_feature_tokens = token_sum
            joint_latent, joint_query_residual, joint_row_means, joint_support_summary = self.joint_sample_encoder(
                sample_values=support.index_select(1, row_index).to(token_sum.dtype),
                sample_missing=batch.support_mask.index_select(1, row_index).to(dtype=torch.bool),
                sample_row_valid=joint_row_valid,
                sample_labels=joint_row_labels,
                feature_tokens=token_sum,
                feature_valid=valid_mask,
                query_values=batch.query_values,
                query_mask=batch.query_mask,
                use_checkpoint=bool(self.config.joint_sample_checkpoint) and self.training,
            )
            # v-next (#179): contribution warmup ramp (1.0 unless the trainer is
            # actively ramping a warm-started joint channel in).
            # Snapshot the schedule buffer OUT of the autograd graph with a
            # clone: `joint_contribution_progress` is a registered buffer that
            # DDP broadcast_buffers (and set_joint_contribution_progress) mutate
            # IN-PLACE before the next forward. A bare `.to()` is a no-op that
            # returns the SAME storage when token_sum is fp32, so aliasing it
            # into the mul below lets that in-place bump corrupt the tensor saved
            # for MulBackward0 -> "modified by an inplace operation ... version N"
            # on the first DDP training step (#179). The ramp is non-
            # differentiable, so the clone is bit-identical in value and grad.
            joint_ramp = self.joint_contribution_progress.detach().clone().to(dtype=token_sum.dtype)
            token_sum = token_sum + joint_ramp * self.joint_sample_scale * joint_latent
            if joint_query_residual is not None:
                joint_query_residual = joint_ramp * joint_query_residual
        feature_positions = (
            batch.feature_positions.to(dtype=token_sum.dtype)
            if batch.feature_positions is not None
            else _default_feature_positions(valid_mask).to(dtype=token_sum.dtype)
        )
        feature_positions = torch.where(valid_mask, feature_positions, torch.zeros_like(feature_positions))
        metadata_logits = torch.zeros(token_sum.shape[:2], dtype=token_sum.dtype, device=token_sum.device)
        feature_metadata = batch.feature_metadata
        if (
            self.feature_metadata_encoder is not None
            and self.feature_metadata_head is not None
            and feature_metadata is not None
        ):
            feature_metadata = feature_metadata.to(dtype=token_sum.dtype)
            metadata_latent = self.feature_metadata_encoder(feature_metadata)
            metadata_latent = torch.where(valid_mask.unsqueeze(-1), metadata_latent, torch.zeros_like(metadata_latent))
            metadata_logits = self.feature_metadata_head(metadata_latent).squeeze(-1)
            token_sum = token_sum + self.feature_metadata_scale * metadata_latent
        if self.position_encoder is not None:
            position_features = _fourier_position_features(
                feature_positions,
                bands=int(self.config.position_frequency_bands),
            ).to(dtype=token_sum.dtype)
            position_latent = self.position_encoder(position_features)
            position_latent = torch.where(valid_mask.unsqueeze(-1), position_latent, torch.zeros_like(position_latent))
            token_sum = token_sum + self.position_encoding_scale * position_latent
        tokens = self.input_norm(token_sum)
        if self.feature_position_mode in {"rope", "rope_fourier"}:
            tokens = _apply_feature_rope(tokens, feature_positions)
            tokens = torch.where(valid_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        local_tokens = tokens
        prior_logits = self.salience_prior(stats_latent).squeeze(-1)
        screening_logits = self.screening_head(screening_latent).squeeze(-1)
        series_logits = self.series_head(series_latent).squeeze(-1)
        shared_prefix_length = _shared_valid_prefix_length(valid_mask)
        if self.context_encoder is not None:
            context_topk = int(self.config.context_candidate_topk)
            if context_topk > 0 and context_topk < int(tokens.shape[1]):
                context_scores = prior_logits.detach() + screening_logits.detach() + series_logits.detach()
                mask_value = -1.0e4 if context_scores.dtype in {torch.float16, torch.bfloat16} else -1.0e30
                context_scores = context_scores.masked_fill(~valid_mask, mask_value)
                take = max(1, min(context_topk, int(tokens.shape[1])))
                context_indices = torch.topk(context_scores, k=take, dim=1, largest=True).indices
                context_valid = valid_mask.gather(1, context_indices)
                gather_index = context_indices.unsqueeze(-1).expand(-1, -1, int(tokens.shape[-1]))
                context_tokens = tokens.gather(1, gather_index)
                if isinstance(self.context_encoder, _SwiGLUContextEncoder):
                    context_tokens = self.context_encoder(context_tokens, valid_mask=context_valid)
                else:
                    with _sdpa_kernel_context(
                        backend=str(self.config.attention_backend),
                        tokens=context_tokens,
                        has_attn_mask=True,
                    ):
                        context_tokens = self.context_encoder(context_tokens, src_key_padding_mask=~context_valid)
                tokens = tokens.scatter(1, gather_index, context_tokens)
                tokens = torch.where(valid_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
            elif shared_prefix_length is not None:
                prefix_tokens = tokens[:, :shared_prefix_length, :]
                if isinstance(self.context_encoder, _SwiGLUContextEncoder):
                    prefix_valid = torch.ones(
                        (prefix_tokens.shape[0], prefix_tokens.shape[1]),
                        dtype=torch.bool,
                        device=prefix_tokens.device,
                    )
                    prefix_tokens = self.context_encoder(prefix_tokens, valid_mask=prefix_valid)
                else:
                    with _sdpa_kernel_context(
                        backend=str(self.config.attention_backend),
                        tokens=prefix_tokens,
                        has_attn_mask=False,
                    ):
                        prefix_tokens = self.context_encoder(prefix_tokens, src_key_padding_mask=None)
                if shared_prefix_length == int(tokens.shape[1]):
                    tokens = prefix_tokens
                else:
                    full_tokens = torch.zeros_like(tokens)
                    full_tokens[:, :shared_prefix_length, :] = prefix_tokens
                    tokens = full_tokens
            else:
                key_padding_mask = ~valid_mask
                with _sdpa_kernel_context(
                    backend=str(self.config.attention_backend),
                    tokens=tokens,
                    has_attn_mask=True,
                ):
                    if isinstance(self.context_encoder, _SwiGLUContextEncoder):
                        tokens = self.context_encoder(tokens, valid_mask=valid_mask)
                    else:
                        tokens = self.context_encoder(tokens, src_key_padding_mask=key_padding_mask)
        scoring_tokens = tokens
        residual_logits = self.residual_head(scoring_tokens).squeeze(-1)
        local_residual_logits = self.local_residual_head(local_tokens).squeeze(-1)
        residual_logits = residual_logits + self.local_residual_scale * local_residual_logits
        selector_gate_logits = self.selector_gate_head(scoring_tokens).squeeze(-1)
        selector_gate_values = self._selector_gate_values(selector_gate_logits, valid_mask)
        # v-next R1 conformal selection head (#179): a dedicated conformity
        # score with a learnable global threshold; probs feed the soft-FDP loss,
        # the raw scores are the bake/e-BH surface for the post-hoc calibrator.
        conformal_scores: torch.Tensor | None = None
        conformal_selection_probs: torch.Tensor | None = None
        if self.conformal_head is not None:
            conformal_raw = self.conformal_head(scoring_tokens).squeeze(-1)
            conformal_temperature = max(float(self.config.conformal_temperature), float(self.config.eps))
            conformal_probs = torch.sigmoid(
                (conformal_raw - self.conformal_threshold.to(dtype=conformal_raw.dtype)) / conformal_temperature
            )
            conformal_selection_probs = torch.where(valid_mask, conformal_probs, torch.zeros_like(conformal_probs))
            conformal_scores = torch.where(valid_mask, conformal_raw, torch.full_like(conformal_raw, -30.0))
        logits = (
            self.prior_scale * prior_logits
            + self.screening_scale * screening_logits
            + self.series_scale * series_logits
            + self.residual_scale * residual_logits
            + self.selector_logit_scale * selector_gate_logits
            + self.feature_metadata_scale * metadata_logits
            + self.class_extras_logit_scale * class_extras_logits
        )
        if self._use_swiglu_fusion:
            logits = logits + self.calibration_bias
        base_logits = logits
        refiner_raw_logits = torch.zeros_like(base_logits)
        full_raw_steps: tuple[torch.Tensor, ...] = ()
        refiner_step_logits: tuple[torch.Tensor, ...] = ()
        if self.refiner is not None:
            steps = int(self.config.refiner_steps if refiner_steps is None else refiner_steps)
            if shared_prefix_length is not None:
                prefix_valid = torch.ones(
                    (tokens.shape[0], shared_prefix_length),
                    dtype=torch.bool,
                    device=tokens.device,
                )
                prefix_tokens, raw_steps = self.refiner(
                    tokens[:, :shared_prefix_length, :],
                    valid_mask=prefix_valid,
                    steps=steps,
                )
                if shared_prefix_length == int(tokens.shape[1]):
                    tokens = prefix_tokens
                    full_raw_steps = raw_steps
                else:
                    full_tokens = torch.zeros_like(tokens)
                    full_tokens[:, :shared_prefix_length, :] = prefix_tokens
                    tokens = full_tokens
                    expanded_steps: list[torch.Tensor] = []
                    for raw_step in raw_steps:
                        expanded = torch.zeros_like(base_logits)
                        expanded[:, :shared_prefix_length] = raw_step
                        expanded_steps.append(expanded)
                    full_raw_steps = tuple(expanded_steps)
            else:
                tokens, full_raw_steps = self.refiner(tokens, valid_mask=valid_mask, steps=steps)
            if full_raw_steps:
                full_steps = tuple(base_logits + self.refiner_scale * raw for raw in full_raw_steps)
                refiner_step_logits = tuple(
                    torch.where(valid_mask, step_logits, torch.full_like(step_logits, -30.0))
                    for step_logits in full_steps
                )
                refiner_raw_logits = full_raw_steps[-1]
                logits = full_steps[-1]
        logits = torch.where(valid_mask, logits, torch.full_like(logits, -30.0))
        base_logits = torch.where(valid_mask, base_logits, torch.full_like(base_logits, -30.0))
        selector_gate_logits = torch.where(valid_mask, selector_gate_logits, torch.full_like(selector_gate_logits, -30.0))
        support_class_logits: torch.Tensor | None = None
        reconstruction_predictions: torch.Tensor | None = None
        if (
            float(self.config.reconstruction_weight) > 0.0
            and batch.reconstruction_row_indices is not None
            and batch.reconstruction_feature_indices is not None
            and batch.reconstruction_valid is not None
        ):
            observed = (~batch.support_mask) & batch.support_valid.unsqueeze(-1) & valid_mask.unsqueeze(1)
            observed_f = observed.to(dtype=tokens.dtype)
            support_values = torch.where(observed, batch.support, torch.zeros_like(batch.support))
            row_context = torch.bmm(support_values, tokens)
            row_count = observed_f.sum(dim=2, keepdim=True).clamp(min=1.0)
            row_context = row_context / torch.sqrt(row_count)
            row_indices = batch.reconstruction_row_indices.clamp(min=0, max=max(0, int(batch.support.shape[1]) - 1))
            feature_indices = batch.reconstruction_feature_indices.clamp(min=0, max=max(0, int(tokens.shape[1]) - 1))
            row_gather = row_indices.unsqueeze(-1).expand(-1, -1, int(tokens.shape[-1]))
            feature_gather = feature_indices.unsqueeze(-1).expand(-1, -1, int(tokens.shape[-1]))
            gathered_rows = row_context.gather(1, row_gather)
            gathered_features = tokens.gather(1, feature_gather)
            pair_tokens = torch.cat([gathered_rows, gathered_features, gathered_rows * gathered_features], dim=-1)
            reconstruction_predictions = self.reconstruction_head(pair_tokens).squeeze(-1)
        population_reconstruction_predictions: torch.Tensor | None = None
        if (
            float(self.config.population_reconstruction_weight) > 0.0
            and self.population_reconstruction_head is not None
            and batch.population_reconstruction_targets is not None
            and batch.population_reconstruction_valid is not None
        ):
            population_reconstruction_predictions = self.population_reconstruction_head(tokens)
            population_reconstruction_predictions = torch.where(
                valid_mask.unsqueeze(-1),
                population_reconstruction_predictions,
                torch.zeros_like(population_reconstruction_predictions),
            )
        population_class_reconstruction_predictions: torch.Tensor | None = None
        if (
            float(self.config.population_class_reconstruction_weight) > 0.0
            and self.population_class_reconstruction_head is not None
            and batch.population_class_reconstruction_targets is not None
            and batch.population_class_reconstruction_valid is not None
        ):
            population_class_reconstruction_predictions = self.population_class_reconstruction_head(tokens)
            population_class_reconstruction_predictions = torch.where(
                valid_mask.unsqueeze(-1),
                population_class_reconstruction_predictions,
                torch.zeros_like(population_class_reconstruction_predictions),
            )
        population_family_logits: torch.Tensor | None = None
        if (
            float(self.config.population_family_weight) > 0.0
            and self.population_family_head is not None
            and batch.population_family_targets is not None
            and batch.population_family_valid is not None
        ):
            population_family_logits = self.population_family_head(tokens)
            population_family_logits = torch.where(
                valid_mask.unsqueeze(-1),
                population_family_logits,
                torch.zeros_like(population_family_logits),
            )
        population_support_type_logits: torch.Tensor | None = None
        if (
            float(self.config.population_support_type_weight) > 0.0
            and self.population_support_type_head is not None
            and batch.population_support_type_targets is not None
            and batch.population_support_type_valid is not None
        ):
            population_support_type_logits = self.population_support_type_head(tokens)
            population_support_type_logits = torch.where(
                valid_mask.unsqueeze(-1),
                population_support_type_logits,
                torch.zeros_like(population_support_type_logits),
            )
        population_param_predictions: torch.Tensor | None = None
        population_param_logvar_predictions: torch.Tensor | None = None
        if (
            (float(self.config.population_param_weight) > 0.0 or float(self.config.population_param_nll_weight) > 0.0)
            and self.population_param_head is not None
            and batch.population_param_targets is not None
            and batch.population_param_valid is not None
        ):
            population_param_predictions = self.population_param_head(tokens)
            population_param_predictions = torch.where(
                valid_mask.unsqueeze(-1),
                population_param_predictions,
                torch.zeros_like(population_param_predictions),
            )
            if float(self.config.population_param_nll_weight) > 0.0 and self.population_param_logvar_head is not None:
                population_param_logvar_predictions = self.population_param_logvar_head(tokens).clamp(
                    min=-6.0, max=float(self.config.population_param_logvar_max)
                )
                population_param_logvar_predictions = torch.where(
                    valid_mask.unsqueeze(-1),
                    population_param_logvar_predictions,
                    torch.zeros_like(population_param_logvar_predictions),
                )
        population_dependency_predictions: torch.Tensor | None = None
        if (
            float(self.config.population_dependency_weight) > 0.0
            and self.population_dependency_head is not None
            and batch.population_dependency_targets is not None
            and batch.population_dependency_valid is not None
        ):
            population_dependency_predictions = self.population_dependency_head(tokens)
            population_dependency_predictions = torch.where(
                valid_mask.unsqueeze(-1),
                population_dependency_predictions,
                torch.zeros_like(population_dependency_predictions),
            )
        population_coeff_predictions: torch.Tensor | None = None
        if (
            float(self.config.population_coeff_weight) > 0.0
            and self.population_coeff_head is not None
            and batch.population_coeff_targets is not None
            and batch.population_coeff_valid is not None
        ):
            population_coeff_predictions = self.population_coeff_head(tokens)
            population_coeff_predictions = torch.where(
                valid_mask.unsqueeze(-1),
                population_coeff_predictions,
                torch.zeros_like(population_coeff_predictions),
            )
        pooled_tokens = (tokens * valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)).sum(dim=1)
        pooled_tokens = pooled_tokens / valid_mask.to(dtype=tokens.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
        population_dependence_type_logits: torch.Tensor | None = None
        if (
            float(self.config.population_dependence_type_weight) > 0.0
            and self.population_dependence_type_head is not None
            and batch.population_dependence_type_targets is not None
            and batch.population_dependence_type_valid is not None
        ):
            population_dependence_type_logits = self.population_dependence_type_head(pooled_tokens)
        population_task_family_logits: torch.Tensor | None = None
        if (
            float(self.config.population_task_family_weight) > 0.0
            and self.population_task_family_head is not None
            and batch.population_task_family_targets is not None
            and batch.population_task_family_valid is not None
        ):
            population_task_family_logits = self.population_task_family_head(pooled_tokens)
        population_task_variant_logits: torch.Tensor | None = None
        if (
            float(self.config.population_task_variant_weight) > 0.0
            and self.population_task_variant_head is not None
            and batch.population_task_variant_targets is not None
            and batch.population_task_variant_valid is not None
        ):
            population_task_variant_logits = self.population_task_variant_head(pooled_tokens)
        support_query_tokens = tokens
        population_conditioning_latent = self._population_conditioning_latent(
            template_tokens=tokens,
            valid_mask=valid_mask,
            population_reconstruction_predictions=population_reconstruction_predictions,
            population_class_reconstruction_predictions=population_class_reconstruction_predictions,
            population_family_logits=population_family_logits,
            population_support_type_logits=population_support_type_logits,
            population_param_predictions=population_param_predictions,
            population_dependency_predictions=population_dependency_predictions,
            population_coeff_predictions=population_coeff_predictions,
            population_dependence_type_logits=population_dependence_type_logits,
            population_task_family_logits=population_task_family_logits,
            population_task_variant_logits=population_task_variant_logits,
        )
        if (
            population_conditioning_latent is not None
            and self.population_conditioning_token_head is not None
            and self.population_conditioning_logit_head is not None
            and self.population_conditioning_selector_head is not None
        ):
            population_conditioning_scale = self.population_conditioning_scale.to(dtype=tokens.dtype)
            token_delta = population_conditioning_scale * self.population_conditioning_token_head(
                population_conditioning_latent
            )
            support_query_tokens = torch.where(
                valid_mask.unsqueeze(-1),
                support_query_tokens + token_delta,
                torch.zeros_like(support_query_tokens),
            )
            support_query_tokens = self._population_decision_tokens(support_query_tokens, valid_mask)
            population_logit_delta = (
                population_conditioning_scale
                * self.population_conditioning_logit_head(support_query_tokens).squeeze(-1)
            )
            population_selector_delta = (
                population_conditioning_scale
                * self.population_conditioning_selector_head(support_query_tokens).squeeze(-1)
            )
            population_logit_delta = torch.where(
                valid_mask,
                population_logit_delta,
                torch.zeros_like(population_logit_delta),
            )
            population_selector_delta = torch.where(
                valid_mask,
                population_selector_delta,
                torch.zeros_like(population_selector_delta),
            )
            selector_gate_logits = selector_gate_logits + population_selector_delta
            selector_gate_values = self._selector_gate_values(selector_gate_logits, valid_mask)
            population_rank_delta = (
                population_logit_delta + self.selector_logit_scale.to(dtype=tokens.dtype) * population_selector_delta
            )
            base_logits = base_logits + population_rank_delta
            logits = logits + population_rank_delta
            if refiner_step_logits:
                refiner_step_logits = tuple(
                    torch.where(valid_mask, step_logits + population_rank_delta, torch.full_like(step_logits, -30.0))
                    for step_logits in refiner_step_logits
                )
            logits = torch.where(valid_mask, logits, torch.full_like(logits, -30.0))
            base_logits = torch.where(valid_mask, base_logits, torch.full_like(base_logits, -30.0))
            selector_gate_logits = torch.where(valid_mask, selector_gate_logits, torch.full_like(selector_gate_logits, -30.0))
        if float(self.config.support_prediction_weight) > 0.0:
            observed = (~batch.support_mask) & batch.support_valid.unsqueeze(-1) & valid_mask.unsqueeze(1)
            support_values = torch.where(observed, batch.support, torch.zeros_like(batch.support))
            gated_values = support_values * selector_gate_values.unsqueeze(1)
            feature_class_weights = self.support_class_head(support_query_tokens)
            denom = torch.sqrt(
                (observed.to(dtype=tokens.dtype) * selector_gate_values.unsqueeze(1).pow(2)).sum(dim=2).clamp(min=1.0)
            )
            support_class_logits = torch.einsum("bsf,bfc->bsc", gated_values, feature_class_weights)
            support_class_logits = support_class_logits / denom.unsqueeze(-1) + self.support_class_bias
        # v-next pairwise redundancy head (#179): score sampled feature pairs
        # from the final (context + population-conditioned) tokens.
        redundancy_pair_logits: torch.Tensor | None = None
        if (
            self.redundancy_query_proj is not None
            and self.redundancy_key_proj is not None
            and batch.redundancy_pair_indices is not None
        ):
            pair_indices = batch.redundancy_pair_indices.to(dtype=torch.long)
            max_token_index = max(0, int(support_query_tokens.shape[1]) - 1)
            left_index = pair_indices[..., 0].clamp(min=0, max=max_token_index)
            right_index = pair_indices[..., 1].clamp(min=0, max=max_token_index)
            redundancy_q = self.redundancy_query_proj(support_query_tokens)
            redundancy_k = self.redundancy_key_proj(support_query_tokens)

            def _gather_pairs(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
                return values.gather(1, index.unsqueeze(-1).expand(-1, -1, int(values.shape[-1])))

            q_left = _gather_pairs(redundancy_q, left_index)
            k_left = _gather_pairs(redundancy_k, left_index)
            q_right = _gather_pairs(redundancy_q, right_index)
            k_right = _gather_pairs(redundancy_k, right_index)
            redundancy_scale = 1.0 / math.sqrt(max(1, int(self.redundancy_rank)))
            # Symmetrized low-rank bilinear form: link(i,j) == link(j,i).
            redundancy_pair_logits = (
                0.5
                * redundancy_scale
                * ((q_left * k_right).sum(dim=-1) + (q_right * k_left).sum(dim=-1))
                + self.redundancy_bias.to(dtype=support_query_tokens.dtype)
            )
        (
            query_class_logits,
            query_class_valid,
            query_feature_class_evidence,
            query_feature_class_gates,
        ) = self._query_classification_outputs(
            tokens=support_query_tokens,
            selector_logits=selector_gate_logits,
            batch=batch,
            feature_valid=valid_mask,
            feature_positions=feature_positions,
            joint_query_residual=joint_query_residual,
        )
        # v-next query ICL head (#179): causal in-context classification over
        # compressed support rows + the query row. Own logits (own CE loss);
        # the zero-init query_icl_scale blend keeps warm-start bit-identical.
        query_icl_logits: torch.Tensor | None = None
        if (
            self.query_icl_head is not None
            and self.joint_sample_encoder is not None
            and joint_row_means is not None
            and joint_row_valid is not None
            and joint_row_labels is not None
            and batch.query_values is not None
        ):
            icl_query_embedding = self.joint_sample_encoder.query_row_embedding(
                query_values=batch.query_values,
                query_mask=batch.query_mask,
                feature_tokens=joint_feature_tokens,
                feature_valid=valid_mask,
            )
            query_icl_logits = self.query_icl_head(
                row_embeddings=joint_row_means,
                row_valid=joint_row_valid,
                row_labels=joint_row_labels,
                query_embedding=icl_query_embedding,
            )
            if query_class_valid is not None:
                query_icl_logits = torch.where(
                    query_class_valid, query_icl_logits, torch.full_like(query_icl_logits, -30.0)
                )
            if query_class_logits is not None and query_class_valid is not None:
                blended = query_class_logits + self.query_icl_scale.to(dtype=query_class_logits.dtype) * query_icl_logits
                query_class_logits = torch.where(
                    query_class_valid, blended, torch.full_like(blended, -30.0)
                )
        return TabenticsDiakrinoFSTeacherOutputs(
            logits=logits,
            base_logits=base_logits,
            prior_logits=torch.where(valid_mask, prior_logits, torch.full_like(prior_logits, -30.0)),
            screening_logits=torch.where(valid_mask, screening_logits, torch.full_like(screening_logits, -30.0)),
            series_logits=torch.where(valid_mask, series_logits, torch.full_like(series_logits, -30.0)),
            residual_logits=torch.where(valid_mask, residual_logits, torch.full_like(residual_logits, -30.0)),
            selector_gate_logits=selector_gate_logits,
            selector_gate_values=selector_gate_values,
            class_extras_logits=torch.where(valid_mask, class_extras_logits, torch.full_like(class_extras_logits, -30.0)),
            refiner_logits=torch.where(valid_mask, refiner_raw_logits, torch.full_like(refiner_raw_logits, -30.0)),
            refiner_step_logits=refiner_step_logits,
            feature_embeddings=support_query_tokens,
            feature_stats=all_stats,
            screening_features=screening_features,
            series_embeddings=series_latent,
            feature_valid_mask=valid_mask,
            feature_positions=feature_positions,
            feature_metadata=feature_metadata,
            support_class_logits=support_class_logits,
            support_labels=batch.support_labels,
            support_valid=batch.support_valid,
            reconstruction_predictions=reconstruction_predictions,
            reconstruction_targets=batch.reconstruction_targets,
            reconstruction_valid=batch.reconstruction_valid,
            population_reconstruction_predictions=population_reconstruction_predictions,
            population_reconstruction_targets=batch.population_reconstruction_targets,
            population_reconstruction_valid=batch.population_reconstruction_valid,
            population_class_reconstruction_predictions=population_class_reconstruction_predictions,
            population_class_reconstruction_targets=batch.population_class_reconstruction_targets,
            population_class_reconstruction_valid=batch.population_class_reconstruction_valid,
            population_family_logits=population_family_logits,
            population_family_targets=batch.population_family_targets,
            population_family_valid=batch.population_family_valid,
            population_support_type_logits=population_support_type_logits,
            population_support_type_targets=batch.population_support_type_targets,
            population_support_type_valid=batch.population_support_type_valid,
            population_param_predictions=population_param_predictions,
            population_param_logvar_predictions=population_param_logvar_predictions,
            population_param_targets=batch.population_param_targets,
            population_param_valid=batch.population_param_valid,
            population_dependency_predictions=population_dependency_predictions,
            population_dependency_targets=batch.population_dependency_targets,
            population_dependency_valid=batch.population_dependency_valid,
            population_coeff_predictions=population_coeff_predictions,
            population_coeff_targets=batch.population_coeff_targets,
            population_coeff_valid=batch.population_coeff_valid,
            population_dependence_type_logits=population_dependence_type_logits,
            population_dependence_type_targets=batch.population_dependence_type_targets,
            population_dependence_type_valid=batch.population_dependence_type_valid,
            population_task_family_logits=population_task_family_logits,
            population_task_family_targets=batch.population_task_family_targets,
            population_task_family_valid=batch.population_task_family_valid,
            population_task_variant_logits=population_task_variant_logits,
            population_task_variant_targets=batch.population_task_variant_targets,
            population_task_variant_valid=batch.population_task_variant_valid,
            proxy_relevance_targets=batch.proxy_relevance_targets,
            query_class_logits=query_class_logits,
            query_labels=batch.query_labels,
            query_class_valid=query_class_valid,
            query_feature_class_evidence=query_feature_class_evidence,
            query_feature_class_gates=query_feature_class_gates,
            query_icl_logits=query_icl_logits,
            joint_support_summary=joint_support_summary,
            joint_support_row_embeddings=joint_row_means,
            joint_support_row_valid=joint_row_valid,
            joint_support_row_labels=joint_row_labels,
            joint_feature_tokens=joint_feature_tokens,
            redundancy_pair_logits=redundancy_pair_logits,
            redundancy_pair_targets=batch.redundancy_pair_targets,
            redundancy_pair_valid=batch.redundancy_pair_valid,
            conformal_scores=conformal_scores,
            conformal_selection_probs=conformal_selection_probs,
        )

    @classmethod
    def from_tabentics_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config: TabenticsDiakrinoFSTeacherConfig | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple["TabenticsDiakrinoFSTeacher", JsonDict]:
        """Create an FS teacher and load reusable DIAKRINO weights.

        Loaded:
          * ``feature_stats_encoder.*`` -> ``stats_encoder.*``
          * ``salience_head.*`` -> ``salience_prior.*``

        Deliberately discarded:
          tokenizer, classifier decoder, label embeddings, HRM/axial blocks,
          EMA state, MAE/JEP/SigReg heads, chaotic head, and SIREN harmonic
          branch/fusion weights.
        """

        _ensure_torch()
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
        source_state: JsonDict = checkpoint.get("model_state_dict", checkpoint)
        if config is None:
            config = TabenticsDiakrinoFSTeacherConfig.from_diakrino_config(checkpoint.get("model_config"))
        model = cls(config)
        target_state = model.state_dict()
        mapped: dict[str, torch.Tensor] = {}
        skipped_shape: list[str] = []
        loaded_source_keys: list[str] = []
        mapping_prefixes = (
            ("feature_stats_encoder.", "stats_encoder."),
            ("salience_head.", "salience_prior."),
        )
        for source_key, value in source_state.items():
            target_key = None
            for source_prefix, target_prefix in mapping_prefixes:
                if source_key.startswith(source_prefix):
                    target_key = target_prefix + source_key[len(source_prefix) :]
                    break
            if target_key is None:
                continue
            if target_key not in target_state:
                skipped_shape.append(f"{source_key} -> {target_key}: target missing")
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
            "source_epoch": checkpoint.get("epoch"),
            "source_step": checkpoint.get("step"),
        }
        model.load_report = report
        return model, report


def _summarize_prefixes(keys: list[str]) -> JsonDict:
    counts: dict[str, int] = {}
    for key in keys:
        prefix = key.split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _soft_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor,
    config: TabenticsDiakrinoFSTeacherConfig,
) -> torch.Tensor:
    probs = torch.sigmoid(logits) * mask.to(dtype=logits.dtype)
    target_values = torch.clamp(targets, 0.0, 1.0) * mask.to(dtype=logits.dtype)
    true_pos = (probs * target_values).sum(dim=1)
    false_pos = (probs * (1.0 - target_values) * mask.to(dtype=logits.dtype)).sum(dim=1)
    false_neg = ((1.0 - probs) * target_values).sum(dim=1)
    denom = (
        true_pos
        + float(config.tversky_alpha) * false_pos
        + float(config.tversky_beta) * false_neg
        + float(config.eps)
    )
    score = (true_pos + float(config.eps)) / denom
    loss = (1.0 - score).clamp(min=0.0, max=1.0)
    gamma = float(config.tversky_gamma)
    if gamma != 1.0:
        loss = torch.pow(loss, gamma)
    return loss.mean()


def _logit_loss_components(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor,
    config: TabenticsDiakrinoFSTeacherConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_f = mask.to(dtype=logits.dtype)
    pos_mass = (torch.clamp(targets, 0.0, 1.0) * mask_f).sum()
    neg_mass = ((1.0 - torch.clamp(targets, 0.0, 1.0)) * mask_f).sum()
    pos_weight = None
    if bool(config.dynamic_pos_weight) and float(pos_mass.detach().cpu()) > 0.0:
        pos_weight = (neg_mass / pos_mass.clamp(min=float(config.eps))).clamp(min=1.0, max=1.0e4)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    if float(config.focal_gamma) > 0.0:
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        bce = bce * torch.pow((1.0 - pt).clamp(min=0.0), float(config.focal_gamma))
    alpha = float(config.focal_alpha)
    bce = bce * (targets * alpha + (1.0 - targets) * (1.0 - alpha))
    bce_denom = pos_mass.clamp(min=1.0) if bool(config.positive_normalized_bce) else mask_f.sum().clamp(min=1.0)
    masked_bce = (bce * mask_f).sum() / bce_denom

    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, neg_inf)
    log_probs = F.log_softmax(masked_logits, dim=1)
    target_values = torch.clamp(targets, min=0.0) * mask.to(dtype=targets.dtype)
    target_sum = target_values.sum(dim=1, keepdim=True)
    rows_with_targets = target_sum.squeeze(1) > 0.0
    if bool(torch.any(rows_with_targets).detach().cpu()):
        target_dist = target_values / target_sum.clamp(min=float(config.eps))
        row_listwise = -(target_dist * log_probs).sum(dim=1)
        listwise = row_listwise[rows_with_targets].mean()
    else:
        listwise = logits.new_tensor(0.0)
    if float(config.tversky_weight) > 0.0:
        tversky = _soft_tversky_loss(logits, targets, mask=mask, config=config)
    else:
        tversky = logits.new_tensor(0.0)
    pairwise = _pairwise_rank_loss(logits, targets, mask=mask, config=config)
    total = (
        float(config.bce_weight) * masked_bce
        + float(config.listwise_weight) * listwise
        + float(config.tversky_weight) * tversky
        + float(config.pairwise_rank_weight) * pairwise
    )
    return total, masked_bce, listwise, tversky, pairwise


def _pairwise_rank_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor,
    config: TabenticsDiakrinoFSTeacherConfig,
) -> torch.Tensor:
    if float(config.pairwise_rank_weight) <= 0.0:
        return logits.new_tensor(0.0)
    losses: list[torch.Tensor] = []
    hard_targets = targets > 0.0
    negative_limit = max(0, int(config.pairwise_rank_negatives))
    margin = float(config.pairwise_rank_margin)
    for row_index in range(int(logits.shape[0])):
        row_mask = mask[row_index]
        pos_idx = torch.nonzero(row_mask & hard_targets[row_index], as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(row_mask & ~hard_targets[row_index], as_tuple=False).squeeze(-1)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        neg_logits = logits[row_index, neg_idx]
        if negative_limit > 0 and neg_idx.numel() > negative_limit:
            hard = torch.topk(neg_logits, k=negative_limit, largest=True).indices
            neg_logits = neg_logits[hard]
        pos_logits = logits[row_index, pos_idx]
        pos_weights = torch.clamp(targets[row_index, pos_idx], min=0.0).to(dtype=logits.dtype)
        pos_weights = pos_weights / pos_weights.sum().clamp(min=float(config.eps))
        rank_terms = F.softplus(neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1) + margin).mean(dim=1)
        losses.append((rank_terms * pos_weights).sum())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _soft_listwise_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, neg_inf)
    log_probs = F.log_softmax(masked_logits, dim=1)
    target_values = torch.clamp(targets, min=0.0, max=1.0) * mask.to(dtype=targets.dtype)
    target_sum = target_values.sum(dim=1, keepdim=True)
    rows_with_targets = target_sum.squeeze(1) > 0.0
    if not bool(torch.any(rows_with_targets).detach().cpu()):
        return logits.new_tensor(0.0)
    target_dist = target_values / target_sum.clamp(min=float(eps))
    row_loss = -(target_dist * log_probs).sum(dim=1)
    return row_loss[rows_with_targets].mean()


def _proxy_pairwise_rank_loss(
    logits: torch.Tensor,
    proxy_targets: torch.Tensor,
    hard_targets: torch.Tensor,
    *,
    mask: torch.Tensor,
    config: TabenticsDiakrinoFSTeacherConfig,
) -> torch.Tensor:
    if float(config.proxy_pairwise_rank_weight) <= 0.0:
        return logits.new_tensor(0.0)
    losses: list[torch.Tensor] = []
    proxy_values = torch.clamp(proxy_targets, min=0.0, max=1.0)
    positive = proxy_values > 0.0
    hard_positive = hard_targets > 0.0
    negative_limit = max(0, int(config.proxy_pairwise_rank_negatives))
    margin = float(config.proxy_pairwise_rank_margin)
    for row_index in range(int(logits.shape[0])):
        row_mask = mask[row_index]
        pos_idx = torch.nonzero(row_mask & positive[row_index], as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(row_mask & ~hard_positive[row_index] & (proxy_values[row_index] <= 0.0), as_tuple=False).squeeze(-1)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        neg_logits = logits[row_index, neg_idx]
        if negative_limit > 0 and neg_idx.numel() > negative_limit:
            hard = torch.topk(neg_logits, k=negative_limit, largest=True).indices
            neg_logits = neg_logits[hard]
        pos_logits = logits[row_index, pos_idx]
        pos_weights = proxy_values[row_index, pos_idx].to(dtype=logits.dtype)
        pos_weights = pos_weights / pos_weights.sum().clamp(min=float(config.eps))
        rank_terms = F.softplus(neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1) + margin).mean(dim=1)
        losses.append((rank_terms * pos_weights).sum())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _row_logit_std(logits: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    mask_f = mask.to(dtype=logits.dtype)
    valid_count = mask_f.sum(dim=1).clamp(min=1.0)
    row_mean = (logits * mask_f).sum(dim=1) / valid_count
    centered = (logits - row_mean.unsqueeze(1)) * mask_f
    return torch.sqrt((centered * centered).sum(dim=1) / valid_count + float(eps))


def _row_rate_logit_mse(
    predicted_rate: torch.Tensor,
    target_rate: torch.Tensor,
    *,
    active_rows: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    active_rows = active_rows.to(dtype=torch.bool)
    if not bool(torch.any(active_rows).detach().cpu()):
        return predicted_rate.new_tensor(0.0)
    pred = predicted_rate[active_rows].clamp(min=float(eps), max=1.0 - float(eps))
    target = target_rate[active_rows].clamp(min=float(eps), max=1.0 - float(eps))
    return F.mse_loss(torch.logit(pred), torch.logit(target))


def _broadcast_aux_valid_mask(valid_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if tuple(valid_mask.shape) == tuple(target.shape):
        return valid_mask
    if valid_mask.ndim == target.ndim - 1 and tuple(valid_mask.shape) == tuple(target.shape[:-1]):
        return valid_mask.unsqueeze(-1).expand_as(target)
    if (
        valid_mask.ndim == target.ndim
        and tuple(valid_mask.shape[:-1]) == tuple(target.shape[:-1])
        and int(valid_mask.shape[-1]) == 1
    ):
        return valid_mask.expand_as(target)
    try:
        return torch.broadcast_to(valid_mask, target.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"auxiliary valid mask shape {tuple(valid_mask.shape)} is not broadcastable to target shape {tuple(target.shape)}"
        ) from exc


def _masked_smooth_l1(
    predictions: torch.Tensor | None,
    targets: torch.Tensor | None,
    valid: torch.Tensor | None,
    *,
    feature_mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    if predictions is None or targets is None or valid is None:
        raise ValueError("masked SmoothL1 inputs must not be None")
    target_values = targets.to(dtype=predictions.dtype)
    if tuple(target_values.shape) != tuple(predictions.shape):
        target_values = torch.broadcast_to(target_values, predictions.shape)
    valid_mask = valid.to(dtype=torch.bool)
    if feature_mask is not None:
        feature_valid = feature_mask.to(dtype=torch.bool)
        if valid_mask.ndim == feature_valid.ndim + 1:
            valid_mask = valid_mask & feature_valid.unsqueeze(-1)
        elif valid_mask.ndim == feature_valid.ndim:
            valid_mask = valid_mask & feature_valid
    valid_mask = _broadcast_aux_valid_mask(valid_mask, predictions)
    if not bool(torch.any(valid_mask).detach().cpu()):
        return predictions.new_tensor(0.0)
    return F.smooth_l1_loss(predictions[valid_mask], target_values[valid_mask], beta=float(beta))


def _masked_gaussian_nll(
    mean: torch.Tensor | None,
    logvar: torch.Tensor | None,
    targets: torch.Tensor | None,
    valid: torch.Tensor | None,
    *,
    feature_mask: torch.Tensor,
    logvar_max: float = 4.0,
) -> torch.Tensor:
    if mean is None or logvar is None or targets is None or valid is None:
        raise ValueError("masked Gaussian NLL inputs must not be None")
    target_values = targets.to(dtype=mean.dtype)
    if tuple(target_values.shape) != tuple(mean.shape):
        target_values = torch.broadcast_to(target_values, mean.shape)
    valid_mask = valid.to(dtype=torch.bool)
    feature_valid = feature_mask.to(dtype=torch.bool)
    if valid_mask.ndim == feature_valid.ndim + 1:
        valid_mask = valid_mask & feature_valid.unsqueeze(-1)
    elif valid_mask.ndim == feature_valid.ndim:
        valid_mask = valid_mask & feature_valid
    valid_mask = _broadcast_aux_valid_mask(valid_mask, mean)
    if not bool(torch.any(valid_mask).detach().cpu()):
        return mean.new_tensor(0.0)
    clamped_logvar = logvar.clamp(min=-6.0, max=float(logvar_max))
    nll = 0.5 * torch.exp(-clamped_logvar) * torch.square(mean - target_values) + 0.5 * clamped_logvar
    return nll[valid_mask].mean()


def _masked_feature_ce(
    logits: torch.Tensor | None,
    targets: torch.Tensor | None,
    valid: torch.Tensor | None,
    *,
    feature_mask: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if logits is None or targets is None or valid is None:
        raise ValueError("masked feature CE inputs must not be None")
    valid_mask = valid.to(dtype=torch.bool) & feature_mask.to(dtype=torch.bool)
    if not bool(torch.any(valid_mask).detach().cpu()):
        return logits.new_tensor(0.0)
    class_count = int(logits.shape[-1])
    labels = targets.to(dtype=torch.long).clamp(min=0, max=max(0, class_count - 1))
    return F.cross_entropy(logits[valid_mask], labels[valid_mask], label_smoothing=float(label_smoothing))


def _masked_row_ce(
    logits: torch.Tensor | None,
    targets: torch.Tensor | None,
    valid: torch.Tensor | None,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if logits is None or targets is None or valid is None:
        raise ValueError("masked row CE inputs must not be None")
    valid_mask = valid.to(dtype=torch.bool)
    if not bool(torch.any(valid_mask).detach().cpu()):
        return logits.new_tensor(0.0)
    class_count = int(logits.shape[-1])
    labels = targets.to(dtype=torch.long).clamp(min=0, max=max(0, class_count - 1))
    return F.cross_entropy(logits[valid_mask], labels[valid_mask], label_smoothing=float(label_smoothing))


def fs_teacher_loss(
    outputs: TabenticsDiakrinoFSTeacherOutputs,
    teacher_targets: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    config: TabenticsDiakrinoFSTeacherConfig | None = None,
) -> tuple[torch.Tensor, JsonDict]:
    """Masked focal BCE plus listwise distillation to winner feature sets."""

    _ensure_torch()
    cfg = config or TabenticsDiakrinoFSTeacherConfig()
    mask = outputs.feature_valid_mask if valid_mask is None else valid_mask.to(dtype=torch.bool)
    targets = torch.clamp(teacher_targets.to(dtype=outputs.logits.dtype), 0.0, 1.0)
    teacher_total, masked_bce, listwise, tversky, pairwise = _logit_loss_components(
        outputs.logits,
        targets,
        mask=mask,
        config=cfg,
    )
    total = float(cfg.teacher_loss_weight) * teacher_total
    deep_supervision = outputs.logits.new_tensor(0.0)
    if (
        float(cfg.teacher_loss_weight) > 0.0
        and outputs.refiner_step_logits
        and float(cfg.refiner_deep_supervision_weight) > 0.0
    ):
        step_losses: list[torch.Tensor] = []
        # The last recurrent step is already the primary loss.  Earlier steps
        # receive a light auxiliary target to encourage useful anytime salience.
        for step_logits in outputs.refiner_step_logits[:-1]:
            step_loss, _step_bce, _step_listwise, _step_tversky, _step_pairwise = _logit_loss_components(
                step_logits,
                targets,
                mask=mask,
                config=cfg,
            )
            step_losses.append(step_loss)
        if step_losses:
            deep_supervision = torch.stack(step_losses).mean()
            total = total + float(cfg.teacher_loss_weight) * float(cfg.refiner_deep_supervision_weight) * deep_supervision
    equilibrium = outputs.logits.new_tensor(0.0)
    if (
        float(cfg.teacher_loss_weight) > 0.0
        and len(outputs.refiner_step_logits) > 1
        and float(cfg.refiner_equilibrium_weight) > 0.0
    ):
        equilibrium_terms = [
            stability_mse_loss(
                outputs.refiner_step_logits[index],
                outputs.refiner_step_logits[index - 1].detach(),
                valid_mask=mask,
                eps=float(cfg.eps),
            )
            for index in range(1, len(outputs.refiner_step_logits))
        ]
        equilibrium = torch.stack(equilibrium_terms).mean()
        total = total + float(cfg.teacher_loss_weight) * float(cfg.refiner_equilibrium_weight) * equilibrium
    selector_total = outputs.logits.new_tensor(0.0)
    selector_cardinality = outputs.logits.new_tensor(0.0)
    selector_entropy = outputs.logits.new_tensor(0.0)
    if float(cfg.selector_gate_weight) > 0.0:
        selector_total, _selector_bce, _selector_listwise, _selector_tversky, _selector_pairwise = _logit_loss_components(
            outputs.selector_gate_logits,
            targets,
            mask=mask,
            config=cfg,
        )
        total = total + float(cfg.selector_gate_weight) * selector_total
    if float(cfg.selector_cardinality_weight) > 0.0:
        mask_f = mask.to(dtype=outputs.selector_gate_values.dtype)
        expected_cardinality = (outputs.selector_gate_values * mask_f).sum(dim=1)
        hard_cardinality = ((targets > 0.0).to(dtype=outputs.selector_gate_values.dtype) * mask_f).sum(dim=1)
        soft_cardinality = (targets * mask_f).sum(dim=1)
        target_cardinality = torch.where(hard_cardinality > 0.0, hard_cardinality, soft_cardinality)
        valid_count = mask_f.sum(dim=1).clamp(min=1.0)
        normalizer_name = str(getattr(cfg, "selector_cardinality_normalizer", "valid")).lower()
        if normalizer_name in {"target", "target_cardinality", "positive"}:
            normalizer = target_cardinality.clamp(min=1.0)
        elif normalizer_name in {"sqrt", "sqrt_valid_target"}:
            normalizer = torch.sqrt((valid_count * target_cardinality.clamp(min=1.0)).clamp(min=1.0))
        else:
            normalizer = valid_count
        cardinality_ratio = (expected_cardinality - target_cardinality) / normalizer
        ratio_clip = float(getattr(cfg, "selector_cardinality_ratio_clip", 0.0))
        if ratio_clip > 0.0:
            cardinality_ratio = cardinality_ratio.clamp(-ratio_clip, ratio_clip)
        selector_cardinality = torch.square(cardinality_ratio).mean()
        total = total + float(cfg.selector_cardinality_weight) * selector_cardinality
    if float(cfg.selector_entropy_weight) > 0.0:
        p = outputs.selector_gate_values.clamp(min=float(cfg.eps), max=1.0 - float(cfg.eps))
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        selector_entropy = (entropy * mask.to(dtype=entropy.dtype)).sum() / mask.to(dtype=entropy.dtype).sum().clamp(min=1.0)
        total = total + float(cfg.selector_entropy_weight) * selector_entropy
    proxy_listwise = outputs.logits.new_tensor(0.0)
    proxy_pairwise = outputs.logits.new_tensor(0.0)
    if outputs.proxy_relevance_targets is not None:
        proxy_targets = torch.clamp(outputs.proxy_relevance_targets.to(dtype=outputs.logits.dtype), 0.0, 1.0)
        proxy_targets = torch.maximum(proxy_targets, (targets > 0.0).to(dtype=outputs.logits.dtype))
        proxy_targets = proxy_targets * mask.to(dtype=outputs.logits.dtype)
        if float(getattr(cfg, "proxy_listwise_weight", 0.0)) > 0.0:
            proxy_listwise = _soft_listwise_loss(
                outputs.logits,
                proxy_targets,
                mask=mask,
                eps=float(cfg.eps),
            )
            total = total + float(getattr(cfg, "proxy_listwise_weight", 0.0)) * proxy_listwise
        if float(getattr(cfg, "proxy_pairwise_rank_weight", 0.0)) > 0.0:
            proxy_pairwise = _proxy_pairwise_rank_loss(
                outputs.logits,
                proxy_targets,
                targets,
                mask=mask,
                config=cfg,
            )
            total = total + float(getattr(cfg, "proxy_pairwise_rank_weight", 0.0)) * proxy_pairwise
    support_prediction = outputs.logits.new_tensor(0.0)
    if (
        float(cfg.support_prediction_weight) > 0.0
        and outputs.support_class_logits is not None
        and outputs.support_labels is not None
        and outputs.support_valid is not None
    ):
        class_count = int(outputs.support_class_logits.shape[-1])
        labels = outputs.support_labels.clamp(min=0, max=max(0, class_count - 1))
        valid_rows = outputs.support_valid & (outputs.support_labels >= 0) & (outputs.support_labels < class_count)
        if bool(torch.any(valid_rows).detach().cpu()):
            support_prediction = F.cross_entropy(outputs.support_class_logits[valid_rows], labels[valid_rows])
            total = total + float(cfg.support_prediction_weight) * support_prediction
    reconstruction = outputs.logits.new_tensor(0.0)
    population_reconstruction = outputs.logits.new_tensor(0.0)
    population_class_reconstruction = outputs.logits.new_tensor(0.0)
    population_family = outputs.logits.new_tensor(0.0)
    population_support_type = outputs.logits.new_tensor(0.0)
    population_param = outputs.logits.new_tensor(0.0)
    population_param_nll = outputs.logits.new_tensor(0.0)
    population_dependency = outputs.logits.new_tensor(0.0)
    population_dependence_type = outputs.logits.new_tensor(0.0)
    population_task_family = outputs.logits.new_tensor(0.0)
    population_task_variant = outputs.logits.new_tensor(0.0)
    population_coeff = outputs.logits.new_tensor(0.0)
    if (
        float(cfg.reconstruction_weight) > 0.0
        and outputs.reconstruction_predictions is not None
        and outputs.reconstruction_targets is not None
        and outputs.reconstruction_valid is not None
    ):
        recon_valid = outputs.reconstruction_valid.to(dtype=torch.bool)
        if bool(torch.any(recon_valid).detach().cpu()):
            reconstruction = F.mse_loss(
                outputs.reconstruction_predictions[recon_valid],
                outputs.reconstruction_targets.to(dtype=outputs.reconstruction_predictions.dtype)[recon_valid],
            )
            total = total + float(cfg.reconstruction_weight) * reconstruction
    if (
        float(cfg.population_reconstruction_weight) > 0.0
        and outputs.population_reconstruction_predictions is not None
        and outputs.population_reconstruction_targets is not None
        and outputs.population_reconstruction_valid is not None
    ):
        pop_valid = outputs.population_reconstruction_valid.to(dtype=torch.bool)
        pop_valid = pop_valid & mask.unsqueeze(-1)
        if bool(torch.any(pop_valid).detach().cpu()):
            predictions = outputs.population_reconstruction_predictions
            targets_pop = outputs.population_reconstruction_targets.to(dtype=predictions.dtype)
            population_reconstruction = F.mse_loss(predictions[pop_valid], targets_pop[pop_valid])
            total = total + float(cfg.population_reconstruction_weight) * population_reconstruction
    if (
        float(cfg.population_class_reconstruction_weight) > 0.0
        and outputs.population_class_reconstruction_predictions is not None
        and outputs.population_class_reconstruction_targets is not None
        and outputs.population_class_reconstruction_valid is not None
    ):
        pop_class_valid = outputs.population_class_reconstruction_valid.to(dtype=torch.bool)
        pop_class_valid = pop_class_valid & mask.unsqueeze(-1)
        if bool(torch.any(pop_class_valid).detach().cpu()):
            predictions = outputs.population_class_reconstruction_predictions
            targets_pop_class = outputs.population_class_reconstruction_targets.to(dtype=predictions.dtype)
            population_class_reconstruction = F.mse_loss(
                predictions[pop_class_valid],
                targets_pop_class[pop_class_valid],
            )
            total = total + float(cfg.population_class_reconstruction_weight) * population_class_reconstruction
    if float(getattr(cfg, "population_family_weight", 0.0)) > 0.0 and outputs.population_family_logits is not None:
        population_family = _masked_feature_ce(
            outputs.population_family_logits,
            outputs.population_family_targets,
            outputs.population_family_valid,
            feature_mask=mask,
            label_smoothing=0.03,
        )
        total = total + float(getattr(cfg, "population_family_weight", 0.0)) * population_family
    if float(getattr(cfg, "population_support_type_weight", 0.0)) > 0.0 and outputs.population_support_type_logits is not None:
        population_support_type = _masked_feature_ce(
            outputs.population_support_type_logits,
            outputs.population_support_type_targets,
            outputs.population_support_type_valid,
            feature_mask=mask,
        )
        total = total + float(getattr(cfg, "population_support_type_weight", 0.0)) * population_support_type
    if float(getattr(cfg, "population_param_weight", 0.0)) > 0.0 and outputs.population_param_predictions is not None:
        population_param = _masked_smooth_l1(
            outputs.population_param_predictions,
            outputs.population_param_targets,
            outputs.population_param_valid,
            feature_mask=mask,
            beta=0.5,
        )
        total = total + float(getattr(cfg, "population_param_weight", 0.0)) * population_param
    if (
        float(getattr(cfg, "population_param_nll_weight", 0.0)) > 0.0
        and outputs.population_param_predictions is not None
        and outputs.population_param_logvar_predictions is not None
    ):
        population_param_nll = _masked_gaussian_nll(
            outputs.population_param_predictions,
            outputs.population_param_logvar_predictions,
            outputs.population_param_targets,
            outputs.population_param_valid,
            feature_mask=mask,
            logvar_max=float(getattr(cfg, "population_param_logvar_max", 4.0)),
        )
        total = total + float(getattr(cfg, "population_param_nll_weight", 0.0)) * population_param_nll
    if float(getattr(cfg, "population_dependency_weight", 0.0)) > 0.0 and outputs.population_dependency_predictions is not None:
        population_dependency = _masked_smooth_l1(
            outputs.population_dependency_predictions,
            outputs.population_dependency_targets,
            outputs.population_dependency_valid,
            feature_mask=mask,
            beta=1.0,
        )
        total = total + float(getattr(cfg, "population_dependency_weight", 0.0)) * population_dependency
    if float(getattr(cfg, "population_coeff_weight", 0.0)) > 0.0 and outputs.population_coeff_predictions is not None:
        population_coeff = _masked_smooth_l1(
            outputs.population_coeff_predictions,
            outputs.population_coeff_targets,
            outputs.population_coeff_valid,
            feature_mask=mask,
            beta=0.5,
        )
        total = total + float(getattr(cfg, "population_coeff_weight", 0.0)) * population_coeff
    if float(getattr(cfg, "population_dependence_type_weight", 0.0)) > 0.0 and outputs.population_dependence_type_logits is not None:
        population_dependence_type = _masked_row_ce(
            outputs.population_dependence_type_logits,
            outputs.population_dependence_type_targets,
            outputs.population_dependence_type_valid,
        )
        total = total + float(getattr(cfg, "population_dependence_type_weight", 0.0)) * population_dependence_type
    if float(getattr(cfg, "population_task_family_weight", 0.0)) > 0.0 and outputs.population_task_family_logits is not None:
        population_task_family = _masked_row_ce(
            outputs.population_task_family_logits,
            outputs.population_task_family_targets,
            outputs.population_task_family_valid,
        )
        total = total + float(getattr(cfg, "population_task_family_weight", 0.0)) * population_task_family
    if float(getattr(cfg, "population_task_variant_weight", 0.0)) > 0.0 and outputs.population_task_variant_logits is not None:
        population_task_variant = _masked_row_ce(
            outputs.population_task_variant_logits,
            outputs.population_task_variant_targets,
            outputs.population_task_variant_valid,
        )
        total = total + float(getattr(cfg, "population_task_variant_weight", 0.0)) * population_task_variant
    query_classification = outputs.logits.new_tensor(0.0)
    query_evidence_auxiliary = outputs.logits.new_tensor(0.0)
    query_selector_relevance = outputs.logits.new_tensor(0.0)
    query_selector_relevance_listwise = outputs.logits.new_tensor(0.0)
    query_gate_cardinality = outputs.logits.new_tensor(0.0)
    query_gate_entropy = outputs.logits.new_tensor(0.0)
    query_accuracy = outputs.logits.new_tensor(0.0)
    query_confidence = outputs.logits.new_tensor(0.0)
    query_valid_examples = outputs.logits.new_tensor(0.0)
    if (
        outputs.query_class_logits is not None
        and outputs.query_labels is not None
        and outputs.query_class_valid is not None
    ):
        query_logits = outputs.query_class_logits
        query_labels = outputs.query_labels.to(device=query_logits.device, dtype=torch.long)
        query_class_count = int(query_logits.shape[-1])
        label_valid = (
            (query_labels >= 0)
            & (query_labels < query_class_count)
            & outputs.query_class_valid.gather(1, query_labels.clamp(min=0, max=max(0, query_class_count - 1)).unsqueeze(1)).squeeze(1)
        )
        query_valid_examples = label_valid.to(dtype=query_logits.dtype).sum()
        if bool(torch.any(label_valid).detach().cpu()):
            if float(getattr(cfg, "query_classification_weight", 0.0)) > 0.0:
                query_classification = query_classification_cross_entropy(
                    query_logits[label_valid],
                    query_labels[label_valid],
                    label_smoothing=float(getattr(cfg, "query_classification_label_smoothing", 0.0)),
                    class_balance=str(getattr(cfg, "query_classification_class_balance", "none")),
                )
                total = total + float(getattr(cfg, "query_classification_weight", 0.0)) * query_classification
            with torch.no_grad():
                query_predictions = query_logits[label_valid].argmax(dim=-1)
                query_accuracy = (query_predictions == query_labels[label_valid]).to(dtype=query_logits.dtype).mean()
                query_confidence = torch.softmax(query_logits[label_valid], dim=-1).amax(dim=-1).mean()
        if (
            float(getattr(cfg, "query_evidence_auxiliary_weight", 0.0)) > 0.0
            and outputs.query_feature_class_evidence is not None
            and outputs.query_feature_class_gates is not None
            and bool(torch.any(label_valid).detach().cpu())
        ):
            query_feature_class_valid = mask.unsqueeze(-1) & outputs.query_class_valid.unsqueeze(1)
            evidence = torch.where(
                query_feature_class_valid,
                outputs.query_feature_class_evidence,
                torch.zeros_like(outputs.query_feature_class_evidence),
            )
            gates = torch.where(
                query_feature_class_valid,
                outputs.query_feature_class_gates,
                torch.zeros_like(outputs.query_feature_class_gates),
            )
            if bool(getattr(cfg, "query_evidence_auxiliary_detach_gates", True)):
                gates = gates.detach()
            gate_mass = gates.sum(dim=1).clamp(min=1.0)
            evidence_logits = (evidence * gates).sum(dim=1) / torch.sqrt(gate_mass)
            evidence_logits = torch.where(
                outputs.query_class_valid,
                evidence_logits,
                torch.full_like(evidence_logits, -30.0),
            )
            query_evidence_auxiliary = query_classification_cross_entropy(
                evidence_logits[label_valid],
                query_labels[label_valid],
                label_smoothing=float(getattr(cfg, "query_classification_label_smoothing", 0.0)),
                class_balance=str(getattr(cfg, "query_classification_class_balance", "none")),
            )
            total = total + float(getattr(cfg, "query_evidence_auxiliary_weight", 0.0)) * query_evidence_auxiliary
        relevance_targets = outputs.proxy_relevance_targets
        if relevance_targets is None:
            relevance_targets = targets
        else:
            relevance_targets = torch.maximum(
                relevance_targets.to(dtype=outputs.logits.dtype).clamp(min=0.0, max=1.0),
                (targets > 0.0).to(dtype=outputs.logits.dtype),
            )
        relevance_targets = relevance_targets.to(dtype=outputs.logits.dtype).clamp(min=0.0, max=1.0)
        relevance_mask = mask & torch.isfinite(relevance_targets)
        relevance_targets = torch.where(relevance_mask, relevance_targets, torch.zeros_like(relevance_targets))
        if (
            float(getattr(cfg, "query_selector_relevance_weight", 0.0)) > 0.0
            and bool(torch.any((relevance_targets > 0.0) & relevance_mask).detach().cpu())
        ):
            flat_logits = outputs.selector_gate_logits[relevance_mask]
            flat_targets = relevance_targets[relevance_mask].to(dtype=flat_logits.dtype)
            positives = flat_targets.sum().clamp(min=1.0)
            negatives = (flat_targets.numel() - flat_targets.sum()).clamp(min=1.0)
            pos_weight = (negatives / positives).clamp(min=1.0, max=32.0)
            weights = 1.0 + (pos_weight - 1.0) * flat_targets
            per_feature = F.binary_cross_entropy_with_logits(flat_logits, flat_targets, reduction="none")
            query_selector_relevance = (per_feature * weights).sum() / weights.sum().clamp(min=1.0)
            total = total + float(getattr(cfg, "query_selector_relevance_weight", 0.0)) * query_selector_relevance
        if (
            float(getattr(cfg, "query_selector_relevance_listwise_weight", 0.0)) > 0.0
            and bool(torch.any((relevance_targets > 0.0) & relevance_mask).detach().cpu())
        ):
            query_selector_relevance_listwise = _soft_listwise_loss(
                outputs.selector_gate_logits,
                relevance_targets,
                mask=relevance_mask,
                eps=float(cfg.eps),
            )
            total = total + float(getattr(cfg, "query_selector_relevance_listwise_weight", 0.0)) * query_selector_relevance_listwise
        if (
            outputs.query_feature_class_gates is not None
            and outputs.query_class_valid is not None
            and float(getattr(cfg, "query_gate_cardinality_weight", 0.0)) > 0.0
        ):
            gate_mask = mask.unsqueeze(-1) & outputs.query_class_valid.unsqueeze(1)
            if bool(torch.any(gate_mask).detach().cpu()):
                gate_valid = gate_mask.to(dtype=outputs.query_feature_class_gates.dtype)
                per_class_counts = gate_valid.sum(dim=1).clamp(min=1.0)
                gate_fraction = (outputs.query_feature_class_gates * gate_valid).sum(dim=1) / per_class_counts
                class_valid = outputs.query_class_valid.to(dtype=torch.bool)
                target_fraction = max(float(cfg.eps), float(getattr(cfg, "query_gate_target_fraction", 0.03)))
                query_gate_cardinality = torch.square(
                    (gate_fraction[class_valid] - target_fraction) / target_fraction
                ).mean()
                total = total + float(getattr(cfg, "query_gate_cardinality_weight", 0.0)) * query_gate_cardinality
        if (
            outputs.query_feature_class_gates is not None
            and outputs.query_class_valid is not None
            and float(getattr(cfg, "query_gate_entropy_weight", 0.0)) > 0.0
        ):
            gate_mask = mask.unsqueeze(-1) & outputs.query_class_valid.unsqueeze(1)
            if bool(torch.any(gate_mask).detach().cpu()):
                p = outputs.query_feature_class_gates.clamp(min=float(cfg.eps), max=1.0 - float(cfg.eps))
                entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
                query_gate_entropy = (entropy * gate_mask.to(dtype=entropy.dtype)).sum() / gate_mask.to(dtype=entropy.dtype).sum().clamp(min=1.0)
                total = total + float(getattr(cfg, "query_gate_entropy_weight", 0.0)) * query_gate_entropy
    # --- v-next query ICL CE (#179): a DIRECT loss on the ICL head's own logits,
    # so the head stays gradient-live while the zero-init blend scale is 0. ---
    query_icl = outputs.logits.new_tensor(0.0)
    query_icl_accuracy = outputs.logits.new_tensor(0.0)
    if (
        float(getattr(cfg, "query_icl_weight", 0.0)) > 0.0
        and outputs.query_icl_logits is not None
        and outputs.query_labels is not None
        and outputs.query_class_valid is not None
    ):
        icl_logits = outputs.query_icl_logits
        icl_labels = outputs.query_labels.to(device=icl_logits.device, dtype=torch.long)
        icl_class_count = int(icl_logits.shape[-1])
        icl_label_valid = (
            (icl_labels >= 0)
            & (icl_labels < icl_class_count)
            & outputs.query_class_valid.gather(
                1, icl_labels.clamp(min=0, max=max(0, icl_class_count - 1)).unsqueeze(1)
            ).squeeze(1)
        )
        if bool(torch.any(icl_label_valid).detach().cpu()):
            query_icl = query_classification_cross_entropy(
                icl_logits[icl_label_valid],
                icl_labels[icl_label_valid],
                label_smoothing=float(getattr(cfg, "query_icl_label_smoothing", 0.0)),
                class_balance=str(getattr(cfg, "query_classification_class_balance", "none")),
            )
            total = total + float(getattr(cfg, "query_icl_weight", 0.0)) * query_icl
            with torch.no_grad():
                icl_predictions = icl_logits[icl_label_valid].argmax(dim=-1)
                query_icl_accuracy = (
                    (icl_predictions == icl_labels[icl_label_valid]).to(dtype=icl_logits.dtype).mean()
                )
    # --- v-next pairwise redundancy loss (#179 → #164/#165): masked BCE on
    # graded SCM link-strength targets over the sampled pairs. ---
    redundancy_pair = outputs.logits.new_tensor(0.0)
    if (
        float(getattr(cfg, "redundancy_pair_weight", 0.0)) > 0.0
        and outputs.redundancy_pair_logits is not None
        and outputs.redundancy_pair_targets is not None
        and outputs.redundancy_pair_valid is not None
    ):
        pair_valid = outputs.redundancy_pair_valid.to(dtype=torch.bool)
        if bool(torch.any(pair_valid).detach().cpu()):
            pair_targets = outputs.redundancy_pair_targets.to(
                dtype=outputs.redundancy_pair_logits.dtype
            ).clamp(0.0, 1.0)
            redundancy_pair = F.binary_cross_entropy_with_logits(
                outputs.redundancy_pair_logits[pair_valid],
                pair_targets[pair_valid],
            )
            total = total + float(getattr(cfg, "redundancy_pair_weight", 0.0)) * redundancy_pair
    # --- v-next R1 conformal selection surrogate (#179): maximize soft recall
    # subject to a soft-FDP-at-q penalty. The honest FDR guarantee stays with
    # the post-hoc conformal calibrator; this shapes the score it consumes. ---
    conformal_selection = outputs.logits.new_tensor(0.0)
    conformal_soft_fdp = outputs.logits.new_tensor(0.0)
    conformal_soft_recall = outputs.logits.new_tensor(0.0)
    if (
        float(getattr(cfg, "conformal_selection_weight", 0.0)) > 0.0
        and outputs.conformal_selection_probs is not None
    ):
        conformal_mask_f = mask.to(dtype=outputs.logits.dtype)
        conformal_relevance = outputs.proxy_relevance_targets
        if conformal_relevance is None:
            conformal_linked = targets > 0.0
        else:
            conformal_linked = (
                conformal_relevance.to(dtype=outputs.logits.dtype).clamp(0.0, 1.0)
                > float(getattr(cfg, "conformal_relevance_threshold", 0.05))
            ) | (targets > 0.0)
        conformal_linked_f = conformal_linked.to(dtype=outputs.logits.dtype) * conformal_mask_f
        conformal_sel = outputs.conformal_selection_probs.to(dtype=outputs.logits.dtype) * conformal_mask_f
        conformal_sel_mass = conformal_sel.sum(dim=1)
        conformal_linked_mass = conformal_linked_f.sum(dim=1)
        conformal_rows = conformal_linked_mass > 0.0
        if bool(torch.any(conformal_rows).detach().cpu()):
            soft_fdp = (conformal_sel * (1.0 - conformal_linked_f)).sum(dim=1) / conformal_sel_mass.clamp(
                min=float(cfg.eps)
            )
            soft_recall = (conformal_sel * conformal_linked_f).sum(dim=1) / conformal_linked_mass.clamp(min=1.0)
            conformal_q = float(
                min(max(float(getattr(cfg, "conformal_target_fdr", 0.1)), float(cfg.eps)), 1.0)
            )
            conformal_row_loss = -soft_recall + float(getattr(cfg, "conformal_fdp_penalty", 4.0)) * F.softplus(
                (soft_fdp - conformal_q) / conformal_q
            )
            conformal_selection = conformal_row_loss[conformal_rows].mean()
            total = total + float(getattr(cfg, "conformal_selection_weight", 0.0)) * conformal_selection
            with torch.no_grad():
                conformal_soft_fdp = soft_fdp[conformal_rows].mean()
                conformal_soft_recall = soft_recall[conformal_rows].mean()
    logit_spread_penalty = outputs.logits.new_tensor(0.0)
    if float(cfg.logit_spread_weight) > 0.0:
        row_std = _row_logit_std(outputs.logits, mask, float(cfg.eps))
        shortfall = (float(cfg.logit_spread_target) - row_std).clamp(min=0.0)
        logit_spread_penalty = torch.square(shortfall).mean()
        total = total + float(cfg.logit_spread_weight) * logit_spread_penalty
    branch_logit_spread_penalty = outputs.logits.new_tensor(0.0)
    if float(getattr(cfg, "branch_logit_spread_weight", 0.0)) > 0.0:
        branch_stds = torch.stack(
            [
                _row_logit_std(branch_logits, mask, float(cfg.eps))
                for branch_logits in (
                    outputs.prior_logits,
                    outputs.screening_logits,
                    outputs.series_logits,
                    outputs.residual_logits,
                    outputs.selector_gate_logits,
                    outputs.class_extras_logits,
                )
            ],
            dim=0,
        )
        excess = (branch_stds - float(getattr(cfg, "branch_logit_spread_max", 4.0))).clamp(min=0.0)
        branch_logit_spread_penalty = torch.square(excess).mean()
        total = total + float(getattr(cfg, "branch_logit_spread_weight", 0.0)) * branch_logit_spread_penalty
    mask_f = mask.to(dtype=outputs.logits.dtype)
    valid_count = mask_f.sum(dim=1).clamp(min=1.0)
    row_target_rate = (targets * mask_f).sum(dim=1) / valid_count
    rows_with_targets = ((targets > 0.0).to(dtype=outputs.logits.dtype) * mask_f).sum(dim=1) > 0.0
    rate_calibration = outputs.logits.new_tensor(0.0)
    if float(getattr(cfg, "rate_calibration_weight", 0.0)) > 0.0:
        row_pred_rate = (torch.sigmoid(outputs.logits) * mask_f).sum(dim=1) / valid_count
        rate_calibration = _row_rate_logit_mse(
            row_pred_rate,
            row_target_rate,
            active_rows=rows_with_targets,
            eps=float(cfg.eps),
        )
        total = total + float(getattr(cfg, "rate_calibration_weight", 0.0)) * rate_calibration
    selector_rate_calibration = outputs.logits.new_tensor(0.0)
    if float(getattr(cfg, "selector_rate_calibration_weight", 0.0)) > 0.0:
        selector_mask_f = mask.to(dtype=outputs.selector_gate_values.dtype)
        selector_valid_count = selector_mask_f.sum(dim=1).clamp(min=1.0)
        row_selector_rate = (outputs.selector_gate_values * selector_mask_f).sum(dim=1) / selector_valid_count
        selector_rate_calibration = _row_rate_logit_mse(
            row_selector_rate.to(dtype=outputs.logits.dtype),
            row_target_rate,
            active_rows=rows_with_targets,
            eps=float(cfg.eps),
        )
        total = total + float(getattr(cfg, "selector_rate_calibration_weight", 0.0)) * selector_rate_calibration
    z_loss = outputs.logits.new_tensor(0.0)
    z_loss_weight = float(getattr(cfg, "z_loss_weight", 0.0))
    if z_loss_weight > 0.0:
        z_mask = mask.to(dtype=torch.float32)
        z_denom = z_mask.sum().clamp(min=1.0)
        z_logits = (outputs.logits.to(dtype=torch.float32) ** 2 * z_mask).sum() / z_denom
        z_gate = (outputs.selector_gate_logits.to(dtype=torch.float32) ** 2 * z_mask).sum() / z_denom
        z_loss = (z_logits + z_gate).to(dtype=outputs.logits.dtype)
        total = total + z_loss_weight * z_loss
    with torch.no_grad():
        final_logit_std = _row_logit_std(outputs.logits, mask, float(cfg.eps)).mean()
        hard_targets = (targets > 0).to(dtype=targets.dtype)
        pred_rate = (torch.sigmoid(outputs.logits) * mask.to(dtype=targets.dtype)).sum() / mask.to(dtype=targets.dtype).sum().clamp(min=1.0)
        pos_rate = (hard_targets * mask.to(dtype=targets.dtype)).sum() / mask.to(dtype=targets.dtype).sum().clamp(min=1.0)
        target_mean = (targets * mask.to(dtype=targets.dtype)).sum() / mask.to(dtype=targets.dtype).sum().clamp(min=1.0)
        gate_mean = (outputs.selector_gate_values * mask.to(dtype=outputs.selector_gate_values.dtype)).sum() / mask.to(dtype=outputs.selector_gate_values.dtype).sum().clamp(min=1.0)
        gate_cardinality = (outputs.selector_gate_values * mask.to(dtype=outputs.selector_gate_values.dtype)).sum(dim=1).mean()
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_focal_bce": float(masked_bce.detach().cpu()),
        "loss_listwise": float(listwise.detach().cpu()),
        "loss_tversky": float(tversky.detach().cpu()),
        "loss_pairwise_rank": float(pairwise.detach().cpu()),
        "loss_selector_gate": float(selector_total.detach().cpu()),
        "loss_selector_cardinality": float(selector_cardinality.detach().cpu()),
        "loss_selector_entropy": float(selector_entropy.detach().cpu()),
        "loss_z": float(z_loss.detach().cpu()),
        "loss_proxy_listwise": float(proxy_listwise.detach().cpu()),
        "loss_proxy_pairwise_rank": float(proxy_pairwise.detach().cpu()),
        "loss_support_prediction": float(support_prediction.detach().cpu()),
        "loss_reconstruction": float(reconstruction.detach().cpu()),
        "loss_population_reconstruction": float(population_reconstruction.detach().cpu()),
        "loss_population_class_reconstruction": float(population_class_reconstruction.detach().cpu()),
        "loss_population_family": float(population_family.detach().cpu()),
        "loss_population_support_type": float(population_support_type.detach().cpu()),
        "loss_population_param": float(population_param.detach().cpu()),
        "loss_population_param_nll": float(population_param_nll.detach().cpu()),
        "loss_population_dependency": float(population_dependency.detach().cpu()),
        "loss_population_dependence_type": float(population_dependence_type.detach().cpu()),
        "loss_population_task_family": float(population_task_family.detach().cpu()),
        "loss_population_task_variant": float(population_task_variant.detach().cpu()),
        "loss_population_coeff": float(population_coeff.detach().cpu()),
        "loss_query_classification": float(query_classification.detach().cpu()),
        "loss_query_evidence_auxiliary": float(query_evidence_auxiliary.detach().cpu()),
        "loss_query_selector_relevance": float(query_selector_relevance.detach().cpu()),
        "loss_query_selector_relevance_listwise": float(query_selector_relevance_listwise.detach().cpu()),
        "loss_query_gate_cardinality": float(query_gate_cardinality.detach().cpu()),
        "loss_query_gate_entropy": float(query_gate_entropy.detach().cpu()),
        "query_accuracy": float(query_accuracy.detach().cpu()),
        "query_valid_examples": float(query_valid_examples.detach().cpu()),
        "query_confidence": float(query_confidence.detach().cpu()),
        "loss_query_icl": float(query_icl.detach().cpu()),
        "query_icl_accuracy": float(query_icl_accuracy.detach().cpu()),
        "loss_redundancy_pair": float(redundancy_pair.detach().cpu()),
        "loss_conformal_selection": float(conformal_selection.detach().cpu()),
        "conformal_soft_fdp": float(conformal_soft_fdp.detach().cpu()),
        "conformal_soft_recall": float(conformal_soft_recall.detach().cpu()),
        "loss_logit_spread": float(logit_spread_penalty.detach().cpu()),
        "loss_branch_logit_spread": float(branch_logit_spread_penalty.detach().cpu()),
        "loss_rate_calibration": float(rate_calibration.detach().cpu()),
        "loss_selector_rate_calibration": float(selector_rate_calibration.detach().cpu()),
        "loss_refiner_deep_supervision": float(deep_supervision.detach().cpu()),
        "loss_refiner_equilibrium": float(equilibrium.detach().cpu()),
        "refiner_step_count": len(outputs.refiner_step_logits),
        "target_positive_rate": float(pos_rate.detach().cpu()),
        "target_mean": float(target_mean.detach().cpu()),
        "prediction_mean": float(pred_rate.detach().cpu()),
        "logit_std_final": float(final_logit_std.detach().cpu()),
        "selector_gate_mean": float(gate_mean.detach().cpu()),
        "selector_expected_cardinality": float(gate_cardinality.detach().cpu()),
    }


def stability_mse_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MSE between per-episode z-scored logits for the same candidate features."""

    _ensure_torch()
    mask = valid_mask.to(dtype=logits_a.dtype)
    count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean_a = (logits_a * mask).sum(dim=1, keepdim=True) / count
    mean_b = (logits_b * mask).sum(dim=1, keepdim=True) / count
    za = (logits_a - mean_a) * mask
    zb = (logits_b - mean_b) * mask
    std_a = torch.sqrt((za * za).sum(dim=1, keepdim=True) / count + eps)
    std_b = torch.sqrt((zb * zb).sum(dim=1, keepdim=True) / count + eps)
    diff = ((logits_a - mean_a) / std_a - (logits_b - mean_b) / std_b) * mask
    return (diff * diff).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad() if torch is not None else (lambda fn: fn)
def topk_feature_indices(
    logits: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    k: int,
    feature_indices: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Return top-k local or original feature indices for each batch item."""

    _ensure_torch()
    selected: list[torch.Tensor] = []
    for batch_index in range(logits.shape[0]):
        valid = torch.nonzero(valid_mask[batch_index], as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            selected.append(valid)
            continue
        scores = logits[batch_index, valid]
        take = min(int(k), int(valid.numel()))
        local = valid[torch.topk(scores, k=take, largest=True).indices]
        if feature_indices is not None:
            selected.append(feature_indices[batch_index, local])
        else:
            selected.append(local)
    return selected


__all__ = [
    "TabenticsDiakrinoFSTeacher",
    "TabenticsDiakrinoFSTeacherBatch",
    "TabenticsDiakrinoFSTeacherConfig",
    "TabenticsDiakrinoFSTeacherOutputs",
    "compute_distribution_series",
    "compute_fs_screening_features",
    "fs_teacher_loss",
    "stability_mse_loss",
    "topk_feature_indices",
]

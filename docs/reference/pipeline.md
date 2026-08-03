---
title: Pipeline
nav_order: 2
parent: Reference
---

Leakage-safe end-to-end DF+FS+classification pipeline configuration, execution, and reproducibility helpers.

Package source: [`tabnetics.pipeline`](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/pipeline)

## Package overview

Stable DF+FS pipeline surface with the promoted leakage-safe FS->DF workflow (``df_stage_position="after_fs"``), so distribution fitting operates on the feature space that survives selection rather than the full raw matrix.

## Stable exports

- `class ClassificationConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L512). Stage-2 classifier backend configuration.
- `DFFSSafeBundleError` (export) - Exported by `tabnetics.pipeline`.
- `DFFSClassifier` (export) - Exported by `tabnetics.pipeline`.
- `class DFFSConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L738).
- `class DistributionFitterConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L433).
- `class DistributionFeatureSelectionPipeline` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L4262). Integrated DF+FS pipeline with strict leakage-safe protocol.
- `class FittedPipelineComponents` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L2264). Train-only fitted state consumed by the public sklearn estimator.
- `class IncompleteFeatureSelectionError(RuntimeError)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L2718). Fail-closed selector outcome that must not enter Stage 2.
- `FitResamplingContext` (export) - Exported by `tabnetics.pipeline`.
- `LeakageAudit` (export) - Exported by `tabnetics.pipeline`.
- `ResolvedSplit` (export) - Exported by `tabnetics.pipeline`.
- `ResolvedSplitPlan` (export) - Exported by `tabnetics.pipeline`.
- `ResamplingContractError` (export) - Exported by `tabnetics.pipeline`.
- `ResamplingPolicy` (export) - Exported by `tabnetics.pipeline`.
- `SAFE_DFFS_BUNDLE_ARTIFACT` (export) - Exported by `tabnetics.pipeline`.
- `SAFE_DFFS_BUNDLE_SCHEMA_VERSION` (export) - Exported by `tabnetics.pipeline`.
- `SAFE_DFFS_BUNDLE_TRUST_MODE` (export) - Exported by `tabnetics.pipeline`.
- `SAFE_DFFS_ROUTE_ID` (export) - Exported by `tabnetics.pipeline`.
- `SafeBundleIntegrityError` (export) - Exported by `tabnetics.pipeline`.
- `SafeBundleSchemaError` (export) - Exported by `tabnetics.pipeline`.
- `SafeDFFSInferenceModel` (export) - Exported by `tabnetics.pipeline`.
- `SplitAssignment` (export) - Exported by `tabnetics.pipeline`.
- `TRAINING_BALANCE_METHODS` (export) - Exported by `tabnetics.pipeline`.
- `TRAINING_BALANCE_SCHEMA_VERSION` (export) - Exported by `tabnetics.pipeline`.
- `TrainingBalanceConfig` (export) - Exported by `tabnetics.pipeline`.
- `TrainingBalanceContractError` (export) - Exported by `tabnetics.pipeline`.
- `TrainingBalanceProvenance` (export) - Exported by `tabnetics.pipeline`.
- `TrainingBalanceResult` (export) - Exported by `tabnetics.pipeline`.
- `UnsupportedSafeBundleStateError` (export) - Exported by `tabnetics.pipeline`.
- `create_safe_dffs_bundle` (export) - Exported by `tabnetics.pipeline`.
- `apply_training_balance` (export) - Exported by `tabnetics.pipeline`.
- `load_safe_dffs_bundle` (export) - Exported by `tabnetics.pipeline`.
- `resolve_assignment` (export) - Exported by `tabnetics.pipeline`.
- `resolve_cv` (export) - Exported by `tabnetics.pipeline`.
- `resolve_holdout` (export) - Exported by `tabnetics.pipeline`.
- `resolve_leave_one_source_out` (export) - Exported by `tabnetics.pipeline`.

## Module details

### `tabnetics.pipeline.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/__init__.py)

Stable DF+FS pipeline surface with the promoted leakage-safe FS->DF workflow (``df_stage_position="after_fs"``), so distribution fitting operates on the feature space that survives selection rather than the full raw matrix.

No top-level public symbols are exported directly from this module.

### `tabnetics.pipeline.config`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/config.py)

Pipeline configuration dataclasses for the stable DF+FS runtime, including the promoted ``df_stage_position="after_fs"`` ordering used by the benchmark and validation entrypoints.

No top-level public symbols are exported directly from this module.

### `tabnetics.pipeline.pipeline`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py)

Integrated distribution-fitting + feature-selection pipeline.

- `class SupportProfile` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L364).
- `class DataAuditReport` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L374).
- `class DistributionFitSummary` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L394).
- `class DistributionFitterConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L433).
- `class ClassificationConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L512). Stage-2 classifier backend configuration.
- `class DFFSConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L738).
- `class PipelineRunResult` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L2149).
- `class NestedPairingEvaluationError(RuntimeError)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L2213). Fail closed when an opt-in nested pairing contract cannot be honored.
- `class FittedPipelineComponents` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L2264). Train-only fitted state consumed by the public sklearn estimator.
- `class IncompleteFeatureSelectionError(RuntimeError)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L2718). Fail-closed selector outcome that must not enter Stage 2.
- `class UnsafeLegacyBundleError(ValueError)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L2911). Raised before any legacy arbitrary-pickle payload is decoded.
- `class DFFSReproducibleModel` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L2915). Legacy trusted-runtime inference helper backed by arbitrary pickle.
- `def load_df_fs_model_bundle(path: str, *, trusted_legacy_pickle: bool = False) -> DFFSReproducibleModel` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L3359).
- `class DistributionFitter` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L3464). Wrapper around UnifiedDistributionSelectorV6 with auditing and diagnostics.
- `class DistributionFeatureSelectionPipeline` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L4262). Integrated DF+FS pipeline with strict leakage-safe protocol.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

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

- `class ClassificationConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L372). Stage-2 classifier backend configuration.
- `class DFFSConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L512).
- `class DistributionFitterConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L311).
- `class DistributionFeatureSelectionPipeline` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L2597). Integrated DF+FS pipeline with strict leakage-safe protocol.

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

- `class SupportProfile` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L242).
- `class DataAuditReport` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L252).
- `class DistributionFitSummary` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L272).
- `class DistributionFitterConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L311).
- `class ClassificationConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L372). Stage-2 classifier backend configuration.
- `class DFFSConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L512).
- `class PipelineRunResult` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L1419).
- `class DFFSReproducibleModel` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L1608). Inference helper that can be reconstructed from a JSON model bundle.
- `def load_df_fs_model_bundle(path: str) -> DFFSReproducibleModel` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L1968).
- `class DistributionFitter` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L2066). Wrapper around UnifiedDistributionSelectorV6 with auditing and diagnostics.
- `class DistributionFeatureSelectionPipeline` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/pipeline/pipeline.py#L2597). Integrated DF+FS pipeline with strict leakage-safe protocol.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

---
title: How It Works
nav_order: 9
---

Tabnetics is organized around a leakage-safe HDLSS pipeline rather than a loose collection of selectors.

## End-to-end flow

1. `tabnetics.datasets` loads benchmark or user-supplied tabular data, keeps dataset metadata close to the run, and exposes catalog/meta-feature helpers used by both benchmarking and validation.
2. `tabnetics.distribution` evaluates univariate distribution families, attaches fit diagnostics, and produces CDF-based transforms for the feature matrix. In the packaged runtime this stage defaults to `df_stage_position="after_fs"`, so it operates on the post-selection feature space.
3. `tabnetics.feature_selection` runs the configured selector portfolio, computes oracle signals, and aggregates selectors into an MNPO-weighted feature set.
4. `tabnetics.classification` applies regime-aware classifier families and the classifier oracle to pick the downstream model for the selected features. Optional conformal diagnostics augment the classifier output with coverage and prediction-set information; they should be read as uncertainty metrics, not as point-accuracy optimizers.
5. `tabnetics.pipeline` orchestrates the train/test split, audit rules, feature transformations, selection stages, classification, and reproducibility payloads.

## Supporting modules

- `tabnetics.core` holds shared runtime helpers, package-path discovery, and lower-level MNPO primitives that other packages build on.
- `tabnetics.multiomics` provides explicit MINT and multi-block PLS helpers when the data is split into real omics blocks rather than a benchmark-style synthetic split; the benchmark-only `split_halves` adapter is just a convenience stress-test path.
- `tabnetics.domains` keeps domain-specific routing and adapters separate from the generic pipeline surface.

## Benchmarking and validation

- `tabnetics.benchmarks` contains method-set profiles, CLI entrypoints, and the benchmark runner used for systematic comparisons.
- `tabnetics.validation` builds campaign plans, shards work, runs validation suites, and records promotion-oriented evidence across benchmark catalogs.
- Installed wrappers mirror the module entrypoints: `tabnetics-benchmark`, `tabnetics-validation-plan`, `tabnetics-validation-shard`, and `tabnetics-validation-suite`.
- Evidence-bearing validation-catalog runs use the HuggingFace bundle as the authoritative operational mirror of the public upstream sources and default to `dataset_integrity_policy=error`; synthetic fallback is not part of the public validation path.

## Practical reading order

Start with [Using Tabnetics](USING.md) if you want to run the library, then read [Methods and References](BACKGROUND.md) for the algorithmic background, and finally use the [Reference](reference/index.md) pages to inspect the package boundaries and source-linked API symbols.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

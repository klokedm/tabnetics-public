---
title: Reference
nav_order: 6
has_children: true
---

Module-by-module reference for the packaged `tabnetics` library surface.

The pages below combine curated package summaries with AST-derived API symbols and source links into the public repo.

## Packages

- [Core Utilities](core.md) - MNPO primitives, runtime helpers, and package-level compatibility utilities.
- [Pipeline](pipeline.md) - Leakage-safe end-to-end DF+FS+classification pipeline configuration, execution, and reproducibility helpers.
- [Distribution Fitting](distribution.md) - Parametric family fitting, transform metadata, and the unified distribution selector used by the pipeline.
- [Feature Selection](feature-selection.md) - Feature-selector orchestration, MNPO-facing config, method contracts, and stable result surfaces.
- [Classification](classification.md) - Regime-aware classifier backends, classifier-oracle logic, and the MNPO final-model selector.
- [Datasets](datasets.md) - Validation catalogs, dataset containers, loaders, registry metadata, and dataset meta-feature helpers.
- [Domains](domains.md) - Domain-specific routing and helpers for bioinformatics and face-oriented workflows.
- [Multi-omics](multiomics.md) - Explicit multi-block integration helpers for DIABLO-style PLS and MINT batch-correction workflows.
- [Benchmarks](benchmarks.md) - Benchmark profiles, CLI entrypoints, gaming detectors, and the benchmark runner surface.
- [Validation](validation.md) - Validation planning, sharding, suite execution, and campaign-specific job builders.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

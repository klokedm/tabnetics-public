---
title: Benchmarks
nav_order: 9
parent: Reference
---

Benchmark profiles, CLI entrypoints, gaming detectors, and the benchmark runner surface.

Package source: [`tabnetics.benchmarks`](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/benchmarks)

## Package overview

Benchmark runner and profile surface backing ``tabnetics-benchmark`` / ``python -m tabnetics.benchmarks.cli``, exposing the profile registry for systematic paired comparisons and the runner that enforces the validation-catalog data policy: evidence-bearing runs use the HuggingFace mirror of public upstream sources and do not silently fall back to synthetic proxies.

## Related modules

- `tabnetics.benchmarks.cli` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/benchmarks/cli.py). Installed benchmark CLI entrypoint backing ``tabnetics-benchmark`` / ``python -m tabnetics.benchmarks.cli``; the packaged surface defaults to ``df_stage_position="after_fs"`` and enforces the validation-catalog data policy: the HuggingFace bundle is the operational mirror of public upstream sources for evidence-bearing runs, and synthetic fallback is forbidden there.
- `tabnetics.benchmarks.gaming_detectors` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/benchmarks/gaming_detectors.py). Read-only anti-gaming diagnostics for benchmark result analysis.
- `tabnetics.benchmarks.profiles` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/benchmarks/profiles.py). Benchmark method-set profiles.
- `tabnetics.benchmarks.runner` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/benchmarks/runner.py). Run integrated DF+FS benchmarking with SOTA comparison and ablations.

## Module details

### `tabnetics.benchmarks.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/benchmarks/__init__.py)

Benchmark runner and profile surface backing ``tabnetics-benchmark`` / ``python -m tabnetics.benchmarks.cli``, exposing the profile registry for systematic paired comparisons and the runner that enforces the validation-catalog data policy: evidence-bearing runs use the HuggingFace mirror of public upstream sources and do not silently fall back to synthetic proxies.

No top-level public symbols are exported directly from this module.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

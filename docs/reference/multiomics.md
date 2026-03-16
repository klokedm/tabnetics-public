---
title: Multi-omics
nav_order: 8
parent: Reference
---

Explicit multi-block integration helpers for DIABLO-style PLS and MINT batch-correction workflows.

Package source: [`tabnetics.multiomics`](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/multiomics)

## Package overview

Optional multi-omics integration helpers centered on DIABLO-style multi-block PLS and MINT-style study-aware integration in the sense of [Singh et al. 2019](https://doi.org/10.1093/bioinformatics/bty1054) and [Rohart et al. 2017](https://doi.org/10.1186/s12859-017-1553-8).

## Stable exports

- `class MultiBlockPLSDA` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/multiomics/integration.py#L28). Supervised multi-block PLS-DA (DIABLO-style).
- `class MINTIntegrator` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/multiomics/integration.py#L272). MINT-style study-aware multi-block integrator.

## Module details

### `tabnetics.multiomics.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/multiomics/__init__.py)

Optional multi-omics integration helpers centered on DIABLO-style multi-block PLS and MINT-style study-aware integration in the sense of [Singh et al. 2019](https://doi.org/10.1093/bioinformatics/bty1054) and [Rohart et al. 2017](https://doi.org/10.1186/s12859-017-1553-8).

No top-level public symbols are exported directly from this module.

### `tabnetics.multiomics.integration`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/multiomics/integration.py)

DIABLO/MINT-style multi-omics integration (VAL12_Suggestions §4.1).

- `class MultiBlockPLSDA` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/multiomics/integration.py#L28). Supervised multi-block PLS-DA (DIABLO-style).
- `class MINTIntegrator` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/multiomics/integration.py#L272). MINT-style study-aware multi-block integrator.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

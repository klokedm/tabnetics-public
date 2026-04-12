---
title: Domains
nav_order: 7
parent: Reference
---

Domain-specific routing and helpers for bioinformatics and face-oriented workflows.

Package source: [`tabnetics.domains`](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/domains)

## Package overview

Top-level domain-specific helpers that keep bioinformatics routing, benchmark-time multi-omics adapters, and face-domain projection helpers separate from the generic DF+FS pipeline surface.

## Stable exports

- `class DatasetDomainContext` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/domains/base.py#L13). Minimal domain metadata needed by domain-specialized helpers.
- `def base_dataset_name(dataset_name: str) -> str` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/domains/base.py#L27). Strip integrated-dataset suffixes and return the stable dataset id.
- `bio` (module) - [Source](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/domains/bio). Subpackage exported from `tabnetics.domains`.
- `face` (module) - [Source](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/domains/face). Subpackage exported from `tabnetics.domains`.
- `def resolve_dataset_catalog_context(dataset_name: str) -> DatasetDomainContext` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/domains/base.py#L35). Resolve catalog-backed domain metadata with a small fallback heuristic.

## Related modules

- `tabnetics.domains.base` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/domains/base.py). Shared dataset/domain context helpers.
- `tabnetics.domains.bio.__init__` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/domains/bio/__init__.py). Bioinformatics-specific helpers.
- `tabnetics.domains.face.__init__` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/domains/face/__init__.py). Face-domain helpers.

## Module details

### `tabnetics.domains.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/domains/__init__.py)

Top-level domain-specific helpers that keep bioinformatics routing, benchmark-time multi-omics adapters, and face-domain projection helpers separate from the generic DF+FS pipeline surface.

No top-level public symbols are exported directly from this module.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

---
title: Datasets
nav_order: 6
parent: Reference
---

Validation catalogs, dataset containers, loaders, registry metadata, and dataset meta-feature helpers.

Package source: [`tabnetics.datasets`](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/datasets)

## Package overview

Dataset registry, catalogs, loaders, and meta-features for validation-catalog and benchmark runs; when the HuggingFace bundle is configured, it acts as the authoritative operational mirror of the public upstream validation data sources.

## Stable exports

- `CATALOG` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/datasets/validation_catalog.py#L54). Module-level constant exported by the package surface.
- `DATASET_SETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/datasets/validation_catalog.py#L55). Module-level constant exported by the package surface.
- `def extract_meta_features(X: np.ndarray, y: np.ndarray, *, expanded: bool = False, skip_distance_matrix: bool = False) -> Dict[str, float]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/datasets/meta_features.py#L33). Extract dataset meta-features useful for tier assignment and analysis.

## Related modules

- `tabnetics.datasets.loaders` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/datasets/loaders.py). Dataset loader exports.
- `tabnetics.datasets.meta_features` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/datasets/meta_features.py). Dataset meta-feature extraction helpers.
- `tabnetics.datasets.registry` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/datasets/registry.py). DatasetSpec registry (single source of truth for validation + benchmark datasets).
- `tabnetics.datasets.validation_catalog` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/datasets/validation_catalog.py). Validation catalog and dataset-set definitions.

## Module details

### `tabnetics.datasets.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/datasets/__init__.py)

Dataset registry, catalogs, loaders, and meta-features for validation-catalog and benchmark runs; when the HuggingFace bundle is configured, it acts as the authoritative operational mirror of the public upstream validation data sources.

No top-level public symbols are exported directly from this module.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

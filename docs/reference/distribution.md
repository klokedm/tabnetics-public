---
title: Distribution Fitting
nav_order: 3
parent: Reference
---

Parametric family fitting, transform metadata, and the unified distribution selector used by the pipeline.

Package source: [`tabnetics.distribution`](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/distribution)

## Package overview

Distribution fitting and transform helpers.

## Stable exports

- `class BootstrapGOFSelector` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/bootstrap.py#L14). Monte-Carlo goodness-of-fit selector tailored to *small* samples.
- `def select_best_distribution_bootstrap(data: np.ndarray, distributions: Dict[str, sps.rv_continuous] | None = None, statistic: str = 'cvm', n_boot: int = 2000, random_state: Any = None) -> Tuple[str, List[Dict[str, Any]]]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/bootstrap.py#L90). Return (best_dist_name, sorted_result_list).
- `def select_best_distribution_bic(data: np.ndarray, distributions: Dict[str, sps.rv_continuous] | None = None) -> Tuple[str, List[Dict[str, Any]]]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/bootstrap.py#L133). Return (best_dist_name, sorted_result_list) using BIC.
- `def auto_select_distribution(data: np.ndarray, *, small_sample_threshold: int = 30, statistic: str = 'cvm', n_boot: int = 2000, random_state: Any = None) -> Tuple[str, List[Dict[str, Any]]]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/bootstrap.py#L166). Dispatch to the most appropriate selection strategy.
- `class DistributionFeatures` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L73). Represents statistical features extracted from data, used to guide distribution selection. These features help in applying heuristics and bonuses for more accurate fitting.
- `class TransformInfo` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L237). Stores information about data transformations (shifting, scaling) applied prior to fitting a distribution. Used to reverse-transform fitted parameters.
- `class LRTResult` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L276). Stores results of Likelihood Ratio Test between nested distributions.
- `class CVResult` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L291). Stores cross-validation results for a distribution.
- `class FitResult` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L306). Holds the results of fitting a single distribution to data, including parameters, goodness-of-fit statistics, and any applied transformations.
- `class UnifiedDistributionSelectorV6` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L366). Hybrid distribution selector combining statistical fitting, goodness-of-fit tests, and feature-based heuristics to identify the best-fitting distribution for given data. This version incorporates improvements for robustness and accuracy, including LRT and CV.

## Module details

### `tabnetics.distribution.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/__init__.py)

Distribution fitting and transform helpers.

No top-level public symbols are exported directly from this module.

### `tabnetics.distribution.bootstrap`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/bootstrap.py)

- `class BootstrapGOFSelector` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/bootstrap.py#L14). Monte-Carlo goodness-of-fit selector tailored to *small* samples.
- `def select_best_distribution_bootstrap(data: np.ndarray, distributions: Dict[str, sps.rv_continuous] | None = None, statistic: str = 'cvm', n_boot: int = 2000, random_state: Any = None) -> Tuple[str, List[Dict[str, Any]]]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/bootstrap.py#L90). Return (best_dist_name, sorted_result_list).
- `def select_best_distribution_bic(data: np.ndarray, distributions: Dict[str, sps.rv_continuous] | None = None) -> Tuple[str, List[Dict[str, Any]]]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/bootstrap.py#L133). Return (best_dist_name, sorted_result_list) using BIC.
- `def auto_select_distribution(data: np.ndarray, *, small_sample_threshold: int = 30, statistic: str = 'cvm', n_boot: int = 2000, random_state: Any = None) -> Tuple[str, List[Dict[str, Any]]]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/bootstrap.py#L166). Dispatch to the most appropriate selection strategy.

### `tabnetics.distribution.selector`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py)

- `class DistributionFeatures` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L73). Represents statistical features extracted from data, used to guide distribution selection. These features help in applying heuristics and bonuses for more accurate fitting.
- `class TransformInfo` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L237). Stores information about data transformations (shifting, scaling) applied prior to fitting a distribution. Used to reverse-transform fitted parameters.
- `class LRTResult` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L276). Stores results of Likelihood Ratio Test between nested distributions.
- `class CVResult` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L291). Stores cross-validation results for a distribution.
- `class FitResult` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L306). Holds the results of fitting a single distribution to data, including parameters, goodness-of-fit statistics, and any applied transformations.
- `class UnifiedDistributionSelectorV6` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/distribution/selector.py#L366). Hybrid distribution selector combining statistical fitting, goodness-of-fit tests, and feature-based heuristics to identify the best-fitting distribution for given data. This version incorporates improvements for robustness and accuracy, including LRT and CV.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

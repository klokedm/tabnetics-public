---
title: Feature Selection
nav_order: 4
parent: Reference
---

Feature-selector orchestration, MNPO-facing config, method contracts, and stable result surfaces.

Package source: [`tabnetics.feature_selection`](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/feature_selection)

## Package overview

Feature selection sub-package (Phase 2+3+6).

## Stable exports

- `class FeatureSelector` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/base.py#L142). Advanced Feature Selector with two strategies: 1. `mnpo_portfolio` (default): MNPO (Nash Multi-Portfolio Optimization) selection. 2. `legacy_voting`: legacy weighted ensemble voting.
- `class FeatureSelectionResult` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/result.py#L15). Comprehensive result object for feature selection process.
- `class FeatureSelectorConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L669). Top-level configuration for ``FeatureSelector``.
- `class OracleConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L37). MNPO oracle controls (Phase 6A, T-R-180).
- `METHOD_REGISTRY` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/registry.py#L482). Module-level constant exported by the package surface.
- `class MethodSpec` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/registry.py#L10). Specification for a single feature selection method.
- `def get_method_weights() -> dict[str, float]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/registry.py#L495). Return {key: legacy_weight} for all registered methods.
- `def get_experimental_keys() -> set[str]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/registry.py#L500). Return the set of keys whose maturity is 'experimental'.
- `class MethodContract(ABC)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/contracts.py#L18). Execution contract for one feature-selection method.
- `class FeatureSelectorMethodContract(MethodContract)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/contracts.py#L46). Thin adapter that routes the contract to an existing selector method.
- `def build_default_method_contracts(selector) -> Dict[str, MethodContract]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/contracts.py#L88). Build contracts for all registry-backed methods with callable handlers.

## Module details

### `tabnetics.feature_selection.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/__init__.py)

Feature selection sub-package (Phase 2+3+6).

No top-level public symbols are exported directly from this module.

### `tabnetics.feature_selection.base`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/base.py)

- `class FeatureSelector` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/base.py#L142). Advanced Feature Selector with two strategies: 1. `mnpo_portfolio` (default): MNPO (Nash Multi-Portfolio Optimization) selection. 2. `legacy_voting`: legacy weighted ensemble voting.

### `tabnetics.feature_selection.config`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py)

Configuration dataclasses for FeatureSelector.

- `DEFAULT_SELECTOR_PENALTY_MAP` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L29). Module-level constant exported by the package surface.
- `class OracleConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L37). MNPO oracle controls (Phase 6A, T-R-180).
- `class MNPOConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L206). Mirror-descent Nash portfolio optimisation parameters.
- `class StabilityConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L361). Stability selection, cluster-stability, decorrelated-stability, and IPSS parameters.
- `class WrapperConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L397). Wrapper refinement and iterative-pruning parameters.
- `class MulticlassConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L423). One-vs-All, ECOC, NSC, and class-Pareto multiclass parameters.
- `class CopulaConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L447). D-vine copula knockoff (DTDCKe) parameters.
- `class PrefilterConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L472). Prefilter blend configuration for feature pool reduction.
- `class ScreeningConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L510). Tier 2 interaction-aware screening configuration.
- `class EvaluationConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L536). Multi-classifier evaluation proxy configuration.
- `class MethodConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L558). Per-method hyper-parameters for mRMR, k-TSP, and HSIC Lasso.
- `class FeatureSelectorConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/config.py#L669). Top-level configuration for ``FeatureSelector``.

### `tabnetics.feature_selection.result`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/result.py)

FeatureSelectionResult dataclass — comprehensive output of the feature selection pipeline.

- `class FeatureSelectionResult` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/result.py#L15). Comprehensive result object for feature selection process.

### `tabnetics.feature_selection.registry`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/registry.py)

Canonical registry of all feature selection methods.

- `class MethodSpec` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/registry.py#L10). Specification for a single feature selection method.
- `METHOD_REGISTRY` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/registry.py#L482). Module-level constant exported by the package surface.
- `def get_method_weights() -> dict[str, float]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/registry.py#L495). Return {key: legacy_weight} for all registered methods.
- `def get_experimental_keys() -> set[str]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/registry.py#L500). Return the set of keys whose maturity is 'experimental'.
- `def method_excluded_by_default(spec: 'MethodSpec', enabled_methods) -> bool` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/registry.py#L505). Hard opt-in gate for ``default_enabled=False`` methods.

### `tabnetics.feature_selection.contracts`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/contracts.py)

Method execution contracts for feature-selection methods.

- `class MethodContract(ABC)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/contracts.py#L18). Execution contract for one feature-selection method.
- `class FeatureSelectorMethodContract(MethodContract)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/contracts.py#L46). Thin adapter that routes the contract to an existing selector method.
- `def build_default_method_contracts(selector) -> Dict[str, MethodContract]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/feature_selection/contracts.py#L88). Build contracts for all registry-backed methods with callable handlers.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

---
title: Auto Router
nav_order: 0
parent: Reference
---

Packaged V25 score-router artifact and runtime calibration helpers.

Package source: [`tabnetics.auto_router`](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/auto_router)

## Package overview

Packaged auto-router for tabnetics runtime calibration.

## Stable exports

- `AUTO_ROUTER_ARTIFACT_VERSION` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L45). Module-level constant exported by the package surface.
- `class AutoRouterOutput` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L616). Auto-router decision emitted by the packaged V25 score model.
- `class DescriptorOODGate` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L501). Robust per-dimension descriptor envelope persisted with router artifacts.
- `class ScoreExpandedRouter` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L744). Predict candidate BA/F1 scores and select a runnable tabnetics profile.
- `class ScoreRouterConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L728).
- `def apply_router_output(config: Any, output: AutoRouterOutput) -> Any` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L1290). Apply an auto-router decision onto a DFFSConfig-like object.
- `def compute_dataset_descriptor(X: Any, y: Any, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L398). Compute the dataset-only descriptor used by the packaged auto-router.
- `def default_artifact_path() -> Path` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L1201). Return the bundled V25 router artifact directory.
- `def load_default_auto_router(model_dir: Optional[Path | str] = None) -> ScoreExpandedRouter` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L1211). Construct a fresh router from the exact artifact bytes for each call.
- `def predict_auto_router(X: Any, y: Any, *, metadata: Optional[Mapping[str, Any]] = None, model_dir: Optional[Path | str] = None, descriptor_ood_gate_enabled: bool = False, crossfit_uncertainty_enabled: bool = False) -> AutoRouterOutput` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L1218). Compute a dataset descriptor and emit the V25 auto-router decision.
- `CandidateUncertaintyPolicy` (export) - Exported by `tabnetics.auto_router`.
- `CrossFitRouterUncertaintyArtifact` (export) - Exported by `tabnetics.auto_router`.
- `RouterOutcomeRow` (export) - Exported by `tabnetics.auto_router`.
- `RouterUncertaintyError` (export) - Exported by `tabnetics.auto_router`.
- `fit_crossfit_router_uncertainty` (export) - Exported by `tabnetics.auto_router`.
- `MISSINGNESS_DESCRIPTOR_KEYS` (export) - Exported by `tabnetics.auto_router`.
- `MISSINGNESS_DESCRIPTOR_POLICY` (export) - Exported by `tabnetics.auto_router`.
- `compute_missingness_descriptors` (export) - Exported by `tabnetics.auto_router`.
- `missingness_descriptor_artifact_status` (export) - Exported by `tabnetics.auto_router`.
- `missingness_descriptor_model_input_enabled` (export) - Exported by `tabnetics.auto_router`.

## Module details

### `tabnetics.auto_router.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/__init__.py)

Packaged auto-router for tabnetics runtime calibration.

No top-level public symbols are exported directly from this module.

### `tabnetics.auto_router.runtime`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py)

Runtime auto-router for tabnetics pipeline configuration.

- `AUTO_ROUTER_ARTIFACT_VERSION` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L45). Module-level constant exported by the package surface.
- `SCORE_ROUTER_ARTIFACT_TYPE` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L46). Module-level constant exported by the package surface.
- `def compute_dataset_descriptor(X: Any, y: Any, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L398). Compute the dataset-only descriptor used by the packaged auto-router.
- `class DescriptorOODResult` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L467). Descriptor-space OOD evaluation for one auto-router decision.
- `class DescriptorOODGate` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L501). Robust per-dimension descriptor envelope persisted with router artifacts.
- `class AutoRouterOutput` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L616). Auto-router decision emitted by the packaged V25 score model.
- `class ScoreRouterConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L728).
- `class ScoreExpandedRouter` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L744). Predict candidate BA/F1 scores and select a runnable tabnetics profile.
- `def default_artifact_path() -> Path` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L1201). Return the bundled V25 router artifact directory.
- `def load_default_auto_router(model_dir: Optional[Path | str] = None) -> ScoreExpandedRouter` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L1211). Construct a fresh router from the exact artifact bytes for each call.
- `def predict_auto_router(X: Any, y: Any, *, metadata: Optional[Mapping[str, Any]] = None, model_dir: Optional[Path | str] = None, descriptor_ood_gate_enabled: bool = False, crossfit_uncertainty_enabled: bool = False) -> AutoRouterOutput` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L1218). Compute a dataset descriptor and emit the V25 auto-router decision.
- `def apply_router_output(config: Any, output: AutoRouterOutput) -> Any` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L1290). Apply an auto-router decision onto a DFFSConfig-like object.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

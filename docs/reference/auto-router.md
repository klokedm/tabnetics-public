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

- `AUTO_ROUTER_ARTIFACT_VERSION` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L28). Module-level constant exported by the package surface.
- `class AutoRouterOutput` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L366). Auto-router decision emitted by the packaged V25 score model.
- `class ScoreExpandedRouter` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L420). Predict candidate BA/F1 scores and select a runnable tabnetics profile.
- `class ScoreRouterConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L407).
- `def apply_router_output(config: Any, output: AutoRouterOutput) -> Any` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L705). Apply an auto-router decision onto a DFFSConfig-like object.
- `def compute_dataset_descriptor(X: Any, y: Any, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L306). Compute the dataset-only descriptor used by the packaged auto-router.
- `def default_artifact_path() -> Path` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L617). Return the bundled V25 router artifact directory.
- `def load_default_auto_router(model_dir: Optional[Path | str] = None) -> ScoreExpandedRouter` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L632). Load the default packaged auto-router, cached by artifact path.
- `def predict_auto_router(X: Any, y: Any, *, metadata: Optional[Mapping[str, Any]] = None, model_dir: Optional[Path | str] = None) -> AutoRouterOutput` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L639). Compute a dataset descriptor and emit the V25 auto-router decision.

## Module details

### `tabnetics.auto_router.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/__init__.py)

Packaged auto-router for tabnetics runtime calibration.

No top-level public symbols are exported directly from this module.

### `tabnetics.auto_router.runtime`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py)

Runtime auto-router for tabnetics pipeline configuration.

- `AUTO_ROUTER_ARTIFACT_VERSION` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L28). Module-level constant exported by the package surface.
- `SCORE_ROUTER_ARTIFACT_TYPE` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L29). Module-level constant exported by the package surface.
- `def compute_dataset_descriptor(X: Any, y: Any, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L306). Compute the dataset-only descriptor used by the packaged auto-router.
- `class AutoRouterOutput` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L366). Auto-router decision emitted by the packaged V25 score model.
- `class ScoreRouterConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L407).
- `class ScoreExpandedRouter` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L420). Predict candidate BA/F1 scores and select a runnable tabnetics profile.
- `def default_artifact_path() -> Path` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L617). Return the bundled V25 router artifact directory.
- `def load_default_auto_router(model_dir: Optional[Path | str] = None) -> ScoreExpandedRouter` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L632). Load the default packaged auto-router, cached by artifact path.
- `def predict_auto_router(X: Any, y: Any, *, metadata: Optional[Mapping[str, Any]] = None, model_dir: Optional[Path | str] = None) -> AutoRouterOutput` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L639). Compute a dataset descriptor and emit the V25 auto-router decision.
- `def apply_router_output(config: Any, output: AutoRouterOutput) -> Any` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/auto_router/runtime.py#L705). Apply an auto-router decision onto a DFFSConfig-like object.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

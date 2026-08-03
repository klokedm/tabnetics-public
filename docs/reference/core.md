---
title: Core Utilities
nav_order: 1
parent: Reference
---

MNPO primitives, runtime helpers, and package-level compatibility utilities.

Package source: [`tabnetics.core`](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/core)

## Package overview

Core numerical, compatibility, and runtime helpers.

## Stable exports

- `def configure_runtime_environment() -> None` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/core/runtime.py#L32). Apply BLAS and CUDA safety defaults before heavy imports.
- `def find_repo_root(anchor: str | Path) -> Path` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/core/paths.py#L23). Resolve the monorepo root when available, else fall back to the project root.
- `def find_repo_root_or_none(anchor: str | Path) -> Path | None` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/core/paths.py#L34). Like :func:`find_repo_root` but returns ``None`` when called from an installed package (e.g. inside ``site-packages``) where no project root or repo root can be located. Safe for module-level use.
- `def get_sklearn_n_jobs() -> int` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/core/runtime.py#L45). Get the thread-local sklearn n_jobs value.
- `def resolve_sklearn_n_jobs(n_jobs: int = 1) -> int` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/core/runtime.py#L50). Normalize a configured sklearn worker count to a concrete positive value.
- `def set_sklearn_n_jobs(n_jobs: int = 1) -> None` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/core/runtime.py#L40). Set the thread-local sklearn n_jobs value used across the package.
- `def sklearn_n_jobs_scope(n_jobs: int = 1) -> Iterator[int]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/core/runtime.py#L63). Bind sklearn worker count for one operation and restore prior thread state.

## Related modules

- `tabnetics.core.runtime` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/core/runtime.py). Shared runtime bootstrap helpers for stable package entrypoints.
- `tabnetics.core.paths` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/core/paths.py). Filesystem path helpers for packaged tabnetics modules.
- `tabnetics.core.mnpo` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/core/mnpo.py). Shared MNPO (Nash Multi-Portfolio Optimization) utilities.

## Module details

### `tabnetics.core.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/core/__init__.py)

Core numerical, compatibility, and runtime helpers.

No top-level public symbols are exported directly from this module.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

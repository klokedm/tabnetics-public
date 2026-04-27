---
title: Auto Router
nav_order: 3
---

# Auto Router

Tabnetics 1.1.0 ships a packaged **V25 calibrated score-router** and enables it by default in `DFFSConfig`. The router is a small scikit-learn/joblib artifact bundled inside the Python package under `tabnetics.auto_router`; it does not need Hugging Face, network access, or a separate model download.

The router chooses among supported pipeline candidates before feature selection starts. It predicts balanced accuracy and macro-F1 for each candidate from descriptors computed directly from the user's training data, then applies a calibrated conservative policy. The descriptor intentionally excludes validation-only fields such as historical hard/easy labels, holdout membership, or any dataset identity signal.

## Default Usage

The normal pipeline path now uses the auto-router:

```python
from tabnetics.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline

config = DFFSConfig(random_seed=42, n_jobs=4)
pipeline = DistributionFeatureSelectionPipeline(config)

result = pipeline.run(X, y, dataset_name="my_dataset", seed=42)
```

During `run()` or `run_pre_split()`, tabnetics computes the router descriptor on the training split only, selects a candidate profile, disables the router on the delegated inner run to avoid recursion, and records the decision in the result metadata.

To disable auto-routing and use explicit flags/defaults:

```python
config = DFFSConfig(auto_router_enabled=False)
```

To inspect the router directly:

```python
from tabnetics.auto_router import predict_auto_router

decision = predict_auto_router(X_train, y_train)
print(decision.metadata["selected_candidate_id"])
print(decision.enabled_methods)
```

## What It Can Change

V25 selects among 12 supported candidates trained from finite, observed validation profiles. The candidate surface covers:

- Method-set breadth: 5-method compact profiles, 16-method full profiles, and one 35-method broad profile.
- Distribution-fitting order: `df_stage_position="after_fs"` and selected `before_fs` candidates.
- Classifier selection: sklearn legacy, sklearn MNPO-hybrid, and FLAML/tune-first variants.
- Classifier oracle depth: `classifier_oracle_k` values 1, 2, and 3.

It does not freely synthesize arbitrary flags. If the router is uncertain, the calibrated policy can fall back to the current default-like candidate.

## Evidence Summary

The packaged model is the V25 calibrated MLP score-router trained with 10-fold dataset-level CV. Training excluded the frozen holdout dataset IDs and used only dataset-computable descriptors plus candidate action encodings.

| Evidence slice | Result |
|---|---|
| Training policy groups | 513 |
| Training datasets | 57 |
| Candidate profiles | 12 |
| Mean balanced-accuracy delta vs current default | +0.0038 |
| Mean macro-F1 delta vs current default | +0.0053 |
| Non-default selections | 124 / 513 |
| Policy-defaulted selections | 264 / 513 |
| Harm > 0.01 BA vs default | 31 / 513 |
| Severe harm > 0.03 BA vs default | 24 / 513 |

The latest available frozen-router holdout evidence predates V25 and should be treated as context, not as completed V25 holdout validation: the Val-22 frozen-router predecessor was negative on the primary-decision holdout slice (mean BA delta -0.0139 over 45 dataset-seed groups) and neutral on replay. That is why the V25 router is calibrated conservatively, keeps a default fallback path, and reports its decision metadata.

## Rationale

The validation campaigns showed that a single static default is serviceable but leaves value on the table: some datasets prefer compact feature-selection stacks, some prefer broader portfolios, and a smaller number prefer alternative distribution-stage or classifier-oracle settings. Manual flag selection is not a good user interface for that evidence.

The auto-router moves those decisions into a reproducible model:

- It uses features available on any new dataset.
- It chooses only from profiles that have actually been run.
- It optimizes both balanced accuracy and macro-F1.
- It applies calibrated lower-confidence behavior instead of chasing raw predicted gains.
- It keeps explicit opt-out support for reproducibility studies and ablations.

The current recommendation is to use the default auto-router for ordinary library usage, and set `auto_router_enabled=False` when reproducing legacy validation profiles or when an experiment needs fully manual flags.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

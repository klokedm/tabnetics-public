---
title: Classification
nav_order: 5
parent: Reference
---

Regime-aware classifier backends, classifier-oracle logic, and the MNPO final-model selector.

Package source: [`tabnetics.classification`](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/classification)

## Package overview

Stage-2 classifier backend surface.

## Stable exports

- `REGIME_HDLSS_EXTREME` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L92). Module-level constant exported by the package surface.
- `REGIME_HDLSS_MODERATE` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L93). Module-level constant exported by the package surface.
- `REGIME_STANDARD` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L94). Module-level constant exported by the package surface.
- `REGIME_POOLS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L100). Module-level constant exported by the package surface.
- `CLASSIFIER_COMPLEXITY_PRIOR` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L157). Module-level constant exported by the package surface.
- `FLAML_NATIVE_BY_FAMILY` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L272). Module-level constant exported by the package surface.
- `def classify_regime(n_samples: int, n_features: int) -> str` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L794). Classify dataset regime used for classifier-family gating.
- `class OracleCandidateStats` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L866).
- `class PLSDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L881). Minimal PLS-DA classifier wrapper using one-vs-rest targets.
- `class DistanceBasedDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L915). DBDA — Aoshima & Yata (2014). Bias-corrected distance classifier for HDLSS.
- `class GeometricalQuadraticDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L960). GQDA — Aoshima & Yata (2015). Bias-corrected geometric QDA for HDLSS.
- `class BiasCorrectedLinearSVM(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1008). BC-SVM (linear kernel) — Nakayama, Yata & Aoshima (2017).
- `class RandomProjectionEnsembleClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1072). Random-projection soft-voting ensemble for HDLSS stabilization.
- `class SparseGroupLassoNNClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1178). Sparse Group Lasso Neural Network (SGLNN) — Yang (2020).
- `class RandomFourierFeaturesClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1430). Approximate kernel classifier via Random Fourier Features (RFF) + LR.
- `class NearestSubspaceClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1534). Classify by minimal reconstruction error from per-class principal subspaces.
- `class SpatialMedianDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1633). Robust distance classifier using spatial medians instead of sample means.
- `class CopulaDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1721). Discriminant analysis exploiting Gaussian-copula-style structure.
- `class TabMClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1828). Lightweight multi-head MLP inspired by TabM.
- `class RealMLPClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1977). Lightweight regularised deep MLP inspired by RealMLP.
- `class CPDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2170). Confusion-Pursuit Discriminant Analysis.
- `class ClassifierBackend(ABC)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2425).
- `class SklearnBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2458). Backwards-compatible sklearn classifier search backend.
- `class FLAMLBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2955). FLAML-powered AutoML backend for the final classifier stage.
- `class OptunaBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3180). Optuna-powered HPO backend for Stage-2 classifier selection (T-R-211).
- `class ClassifierOracle` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3683). Compute MNPO classifier oracles and Nash-selection weights.
- `class MNPOClassifierBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L4503). MNPO-hybrid backend: regime gating + oracle selection + per-family HPO.

## Module details

### `tabnetics.classification.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/__init__.py)

Stage-2 classifier backend surface.

No top-level public symbols are exported directly from this module.

### `tabnetics.classification.backends`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py)

Classifier backend abstraction for Stage-2 final model selection.

- `REGIME_HDLSS_EXTREME` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L92). Module-level constant exported by the package surface.
- `REGIME_HDLSS_MODERATE` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L93). Module-level constant exported by the package surface.
- `REGIME_STANDARD` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L94). Module-level constant exported by the package surface.
- `REGIME_POOLS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L100). Module-level constant exported by the package surface.
- `CLASSIFIER_COMPLEXITY_PRIOR` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L157). Module-level constant exported by the package surface.
- `FLAML_NATIVE_BY_FAMILY` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L272). Module-level constant exported by the package surface.
- `def classify_regime(n_samples: int, n_features: int) -> str` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L794). Classify dataset regime used for classifier-family gating.
- `class OracleCandidateStats` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L866).
- `class PLSDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L881). Minimal PLS-DA classifier wrapper using one-vs-rest targets.
- `class DistanceBasedDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L915). DBDA — Aoshima & Yata (2014). Bias-corrected distance classifier for HDLSS.
- `class GeometricalQuadraticDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L960). GQDA — Aoshima & Yata (2015). Bias-corrected geometric QDA for HDLSS.
- `class BiasCorrectedLinearSVM(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1008). BC-SVM (linear kernel) — Nakayama, Yata & Aoshima (2017).
- `class RandomProjectionEnsembleClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1072). Random-projection soft-voting ensemble for HDLSS stabilization.
- `class SparseGroupLassoNNClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1178). Sparse Group Lasso Neural Network (SGLNN) — Yang (2020).
- `class RandomFourierFeaturesClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1430). Approximate kernel classifier via Random Fourier Features (RFF) + LR.
- `class NearestSubspaceClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1534). Classify by minimal reconstruction error from per-class principal subspaces.
- `class SpatialMedianDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1633). Robust distance classifier using spatial medians instead of sample means.
- `class CopulaDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1721). Discriminant analysis exploiting Gaussian-copula-style structure.
- `class TabMClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1828). Lightweight multi-head MLP inspired by TabM.
- `class RealMLPClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1977). Lightweight regularised deep MLP inspired by RealMLP.
- `class CPDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2170). Confusion-Pursuit Discriminant Analysis.
- `class ClassifierBackend(ABC)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2425).
- `class SklearnBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2458). Backwards-compatible sklearn classifier search backend.
- `class FLAMLBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2955). FLAML-powered AutoML backend for the final classifier stage.
- `class OptunaBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3180). Optuna-powered HPO backend for Stage-2 classifier selection (T-R-211).
- `class ClassifierOracle` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3683). Compute MNPO classifier oracles and Nash-selection weights.
- `class MNPOClassifierBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L4503). MNPO-hybrid backend: regime gating + oracle selection + per-family HPO.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

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

- `class ClassifierCandidateAdmissionError(RuntimeError)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L125). Raised when an explicit candidate request leaves no constructible model.
- `class SampleWeightRoutingError(ValueError)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L138). Raised when requested weights cannot be safely routed to an estimator.
- `class NativeCategoricalStage2Error(RuntimeError)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L142). Fail-closed error for the narrow native categorical Stage-2 adapter.
- `class NativeCategoricalStage2Adapter` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L158). Concrete DataFrame adapter for one admitted native categorical family.
- `def resolve_native_categorical_stage2_adapter(classifier_name: str) -> NativeCategoricalStage2Adapter` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L324). Resolve an actually importable singleton native categorical adapter.
- `def fit_native_categorical_stage2_singleton(*, adapter: NativeCategoricalStage2Adapter, X_train: Any, y_train: Sequence[Any], categorical_columns: Sequence[str], seed: int, fold_view_factory: Callable[[np.ndarray, np.ndarray], tuple[Any, Any, Sequence[str], Mapping[str, Any]]], cv_splits: int = 5, max_train_test_gap: float = 0.0, n_jobs: int = 1) -> Tuple[BaseEstimator, str, float, float, int, Dict[str, Any]]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L374). Cross-validate one explicit native categorical estimator without fallback.
- `def fit_estimator_with_sample_weight(model: BaseEstimator, X: np.ndarray, y: np.ndarray, sample_weight: Optional[Sequence[float]] = None) -> str` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L596). Fit an estimator with an explicit, fail-closed weight route.
- `REGIME_POOLS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L637). Module-level constant exported by the package surface.
- `CLASSIFIER_COMPLEXITY_PRIOR` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L638). Module-level constant exported by the package surface.
- `FLAML_NATIVE_BY_FAMILY` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L743). Module-level constant exported by the package surface.
- `def classify_regime(n_samples: int, n_features: int) -> str` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1277). Classify dataset regime used for classifier-family gating.
- `class OracleCandidateStats` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1844).
- `class PLSDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1864). Minimal PLS-DA classifier wrapper using one-vs-rest targets.
- `class DistanceBasedDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1898). DBDA — Aoshima & Yata (2014). Bias-corrected distance classifier for HDLSS.
- `class GeometricalQuadraticDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1943). GQDA — Aoshima & Yata (2015). Bias-corrected geometric QDA for HDLSS.
- `class HDRDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1991). High-Dimensional Regularized DA via CDM eigendecomposition.
- `class DWDClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2138). Distance-Weighted Discrimination — generalized mean distance formulation.
- `class SparsePLSDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2295). Sparse PLS-DA with L1-penalized loadings and BER-driven component selection.
- `class ECOCClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2481). Error-Correcting Output Codes wrapper with soft Hamming decoding.
- `class BiasCorrectedLinearSVM(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2596). BC-SVM (linear kernel) — Nakayama, Yata & Aoshima (2017).
- `class RandomProjectionEnsembleClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2660). Random-projection soft-voting ensemble for HDLSS stabilization.
- `class SparseGroupLassoNNClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2766). Sparse Group Lasso Neural Network (SGLNN) — Yang (2020).
- `class RandomFourierFeaturesClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3018). Approximate kernel classifier via Random Fourier Features (RFF) + LR.
- `class NearestSubspaceClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3122). Classify by minimal reconstruction error from per-class principal subspaces.
- `class SpatialMedianDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3238). Robust distance classifier using spatial medians instead of sample means.
- `class CopulaDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3326). Discriminant analysis exploiting Gaussian-copula-style structure.
- `class AblatableCADAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3433). Narrowed, ablatable CADA v1.
- `class TabMClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3595). Lightweight multi-head MLP inspired by TabM.
- `class RealMLPClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3744). Lightweight regularised deep MLP inspired by RealMLP.
- `class CPDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3937). Confusion-Pursuit Discriminant Analysis.
- `class ClassifierBackend(ABC)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L4192).
- `class SklearnBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L4228). Backwards-compatible sklearn classifier search backend.
- `class FLAMLBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L5342). FLAML-powered AutoML backend for the final classifier stage.
- `class OptunaBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L5650). Optuna-powered HPO backend for Stage-2 classifier selection (T-R-211).
- `class ClassifierOracle` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L6308). Compute MNPO classifier oracles and Nash-selection weights.
- `class MNPOClassifierBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L7364). MNPO-hybrid backend: regime gating + oracle selection + per-family HPO.

## Module details

### `tabnetics.classification.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/__init__.py)

Stage-2 classifier backend surface.

No top-level public symbols are exported directly from this module.

### `tabnetics.classification.backends`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py)

Classifier backend abstraction for Stage-2 final model selection.

- `class ClassifierCandidateAdmissionError(RuntimeError)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L125). Raised when an explicit candidate request leaves no constructible model.
- `class SampleWeightRoutingError(ValueError)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L138). Raised when requested weights cannot be safely routed to an estimator.
- `class NativeCategoricalStage2Error(RuntimeError)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L142). Fail-closed error for the narrow native categorical Stage-2 adapter.
- `class NativeCategoricalStage2Adapter` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L158). Concrete DataFrame adapter for one admitted native categorical family.
- `def resolve_native_categorical_stage2_adapter(classifier_name: str) -> NativeCategoricalStage2Adapter` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L324). Resolve an actually importable singleton native categorical adapter.
- `def fit_native_categorical_stage2_singleton(*, adapter: NativeCategoricalStage2Adapter, X_train: Any, y_train: Sequence[Any], categorical_columns: Sequence[str], seed: int, fold_view_factory: Callable[[np.ndarray, np.ndarray], tuple[Any, Any, Sequence[str], Mapping[str, Any]]], cv_splits: int = 5, max_train_test_gap: float = 0.0, n_jobs: int = 1) -> Tuple[BaseEstimator, str, float, float, int, Dict[str, Any]]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L374). Cross-validate one explicit native categorical estimator without fallback.
- `def fit_estimator_with_sample_weight(model: BaseEstimator, X: np.ndarray, y: np.ndarray, sample_weight: Optional[Sequence[float]] = None) -> str` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L596). Fit an estimator with an explicit, fail-closed weight route.
- `REGIME_POOLS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L637). Module-level constant exported by the package surface.
- `CLASSIFIER_COMPLEXITY_PRIOR` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L638). Module-level constant exported by the package surface.
- `FLAML_NATIVE_BY_FAMILY` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L743). Module-level constant exported by the package surface.
- `def classify_regime(n_samples: int, n_features: int) -> str` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1277). Classify dataset regime used for classifier-family gating.
- `class OracleCandidateStats` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1844).
- `class PLSDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1864). Minimal PLS-DA classifier wrapper using one-vs-rest targets.
- `class DistanceBasedDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1898). DBDA — Aoshima & Yata (2014). Bias-corrected distance classifier for HDLSS.
- `class GeometricalQuadraticDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1943). GQDA — Aoshima & Yata (2015). Bias-corrected geometric QDA for HDLSS.
- `class HDRDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L1991). High-Dimensional Regularized DA via CDM eigendecomposition.
- `class DWDClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2138). Distance-Weighted Discrimination — generalized mean distance formulation.
- `class SparsePLSDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2295). Sparse PLS-DA with L1-penalized loadings and BER-driven component selection.
- `class ECOCClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2481). Error-Correcting Output Codes wrapper with soft Hamming decoding.
- `class BiasCorrectedLinearSVM(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2596). BC-SVM (linear kernel) — Nakayama, Yata & Aoshima (2017).
- `class RandomProjectionEnsembleClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2660). Random-projection soft-voting ensemble for HDLSS stabilization.
- `class SparseGroupLassoNNClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L2766). Sparse Group Lasso Neural Network (SGLNN) — Yang (2020).
- `class RandomFourierFeaturesClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3018). Approximate kernel classifier via Random Fourier Features (RFF) + LR.
- `class NearestSubspaceClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3122). Classify by minimal reconstruction error from per-class principal subspaces.
- `class SpatialMedianDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3238). Robust distance classifier using spatial medians instead of sample means.
- `class CopulaDiscriminantAnalysis(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3326). Discriminant analysis exploiting Gaussian-copula-style structure.
- `class AblatableCADAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3433). Narrowed, ablatable CADA v1.
- `class TabMClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3595). Lightweight multi-head MLP inspired by TabM.
- `class RealMLPClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3744). Lightweight regularised deep MLP inspired by RealMLP.
- `class CPDAClassifier(ClassifierMixin, BaseEstimator)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L3937). Confusion-Pursuit Discriminant Analysis.
- `class ClassifierBackend(ABC)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L4192).
- `class SklearnBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L4228). Backwards-compatible sklearn classifier search backend.
- `class FLAMLBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L5342). FLAML-powered AutoML backend for the final classifier stage.
- `class OptunaBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L5650). Optuna-powered HPO backend for Stage-2 classifier selection (T-R-211).
- `class ClassifierOracle` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L6308). Compute MNPO classifier oracles and Nash-selection weights.
- `class MNPOClassifierBackend(ClassifierBackend)` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/classification/backends.py#L7364). MNPO-hybrid backend: regime gating + oracle selection + per-family HPO.

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).

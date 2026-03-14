"""Classifier backend abstraction for Stage-2 final model selection.

This module is intentionally scoped to the final classifier step after feature
selection. It does not participate in Stage-1 feature-selection oracle scoring.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.covariance import LedoitWolf, OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVC

from tabnetics.core.runtime import get_sklearn_n_jobs as _get_sklearn_n_jobs
try:
    from tabnetics.core.compat import make_logistic_regression
except Exception as exc:
    from tabnetics.core.compat import make_logistic_regression  # type: ignore

try:  # Optional dependency.
    from catboost import CatBoostClassifier  # type: ignore
except Exception as exc:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore

try:  # Optional dependency.
    from lightgbm import LGBMClassifier  # type: ignore
except Exception as exc:  # pragma: no cover
    LGBMClassifier = None  # type: ignore

try:  # Optional dependency.
    from pytabkit import RealMLP_TD_Classifier as _RealMLP_TD_Classifier  # type: ignore
except Exception:  # pragma: no cover
    _RealMLP_TD_Classifier = None  # type: ignore

try:  # Optional dependency.
    from pytabkit import TabM_D_Classifier as _TabM_D_Classifier  # type: ignore
except Exception:  # pragma: no cover
    _TabM_D_Classifier = None  # type: ignore

try:
    from tabnetics.core.mnpo import (
        aggregate_payoff_matrix,
        compute_banzhaf_values,
        fit_shapley_weights,
        fit_tritrust_weights,
        james_stein_shrinkage,
        matrix_from_scalar_scores,
        mirror_descent_reference_regularized,
        pairwise_pref_from_fold_scores,
    )
except Exception as exc:
    from tabnetics.core.mnpo import (  # type: ignore
        aggregate_payoff_matrix,
        compute_banzhaf_values,
        fit_shapley_weights,
        fit_tritrust_weights,
        james_stein_shrinkage,
        matrix_from_scalar_scores,
        mirror_descent_reference_regularized,
        pairwise_pref_from_fold_scores,
    )


logger = logging.getLogger(__name__)

OptionalModelBuildResult = Tuple[Optional[BaseEstimator], Optional[str]]
OptionalModelBuildReturn = Union[Optional[BaseEstimator], OptionalModelBuildResult]


REGIME_HDLSS_EXTREME = "hdlss_extreme"
REGIME_HDLSS_MODERATE = "hdlss_moderate"
REGIME_STANDARD = "standard"

# Tree-based model families subject to train/test gap gating and complexity penalty.
_TREE_MODEL_NAMES: frozenset = frozenset({"rf", "extra_tree", "lgbm", "catboost", "xgb"})

# Pools intentionally conservative in HDLSS; tree families only in standard regime.
REGIME_POOLS: Dict[str, Tuple[str, ...]] = {
    REGIME_HDLSS_EXTREME: (
        "lr",
        "elastic_net_lr",
        "svm_linear",
        "bc_svm_linear",
        "rp_ensemble",
        "dlda",
        "shrinkage_lda",
        "nsc",
        "pls_da_classifier",
        "nb",
        "dbda",
        "gqda",
        "sglnn",
        "rff_lr",
        "near_subspace",
        "spatial_median_da",
        "copula_da",
        "cpda",
        "tabm",
        "realmlp",
        "tabm_official",
        "realmlp_td",
    ),
    REGIME_HDLSS_MODERATE: (
        "lr",
        "elastic_net_lr",
        "svm_linear",
        "bc_svm_linear",
        "rp_ensemble",
        "svm_rbf",
        "dlda",
        "shrinkage_lda",
        "nsc",
        "pls_da_classifier",
        "gpc",
        "nb",
        "knn",
        "vote_ensemble",
        "tabpfn",
        "dbda",
        "gqda",
        "sglnn",
        "rff_lr",
        "near_subspace",
        "spatial_median_da",
        "copula_da",
        "cpda",
        "tabm",
        "realmlp",
        "tabm_official",
        "realmlp_td",
    ),
}

# Complexity prior: higher means simpler / preferred under HDLSS uncertainty.
CLASSIFIER_COMPLEXITY_PRIOR: Dict[str, float] = {
    "lr": 1.00,
    "elastic_net_lr": 0.98,
    "svm_linear": 0.96,
    "bc_svm_linear": 0.95,
    "dbda": 0.93,
    "rp_ensemble": 0.94,
    "dlda": 0.92,
    "shrinkage_lda": 0.92,
    "gqda": 0.91,
    "sglnn": 0.82,
    "rff_lr": 0.84,
    "near_subspace": 0.89,
    "spatial_median_da": 0.91,
    "copula_da": 0.87,
    "cpda": 0.85,
    "tabm": 0.78,
    "realmlp": 0.76,
    "tabm_official": 0.72,
    "realmlp_td": 0.70,
    "nsc": 0.90,
    "pls_da_classifier": 0.88,
    "nb": 0.86,
    "svm_rbf": 0.74,
    "gpc": 0.68,
    "knn": 0.62,
    "vote_ensemble": 0.58,
    "tabpfn": 0.54,
    "rf": 0.42,
    "extra_tree": 0.36,
    "xgb": 0.34,
    "lgbm": 0.34,
    "catboost": 0.34,
}

_MULTICLASS_COMPLEXITY_PRIOR_THRESHOLD = 5


def _adjust_complexity_priors(
    n_classes: int,
    *,
    threshold: int = _MULTICLASS_COMPLEXITY_PRIOR_THRESHOLD,
) -> Dict[str, float]:
    """Flatten complexity priors for multiclass tasks (K >= *threshold*).

    When K is large, global safe-model bias (LR dominance) hurts more than
    model complexity.  Compress the prior gap so that performance evidence
    dominates the complexity oracle matrix.
    """
    if int(n_classes) < int(threshold):
        return dict(CLASSIFIER_COMPLEXITY_PRIOR)
    # Linear compression: gap between best (1.0) and worst (0.34) shrinks
    # from 0.66 to ~0.20, keeping relative ordering intact.
    return {k: 0.7 + 0.3 * v for k, v in CLASSIFIER_COMPLEXITY_PRIOR.items()}


FLAML_NATIVE_BY_FAMILY: Dict[str, str] = {
    "lr": "lrl2",
    "rf": "rf",
    "xgb": "xgboost",
    "lgbm": "lgbm",
    "extra_tree": "extra_tree",
    "catboost": "catboost",
}

_ALIAS_GROUPS: Dict[str, str] = {
    "dlda": "lda_shrink",
    "shrinkage_lda": "lda_shrink",
}


class _LabelEncodedEstimator(ClassifierMixin, BaseEstimator):
    """Wrap an estimator so it can train on encoded labels and predict originals."""

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_arr = np.asarray(y).ravel()
        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y_arr)
        self.estimator.fit(np.asarray(X, dtype=float), y_enc)
        self.classes_ = np.asarray(self._label_encoder.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.asarray(self.estimator.predict(np.asarray(X, dtype=float))).ravel()
        if not hasattr(self, "_label_encoder"):
            return pred
        pred_i = np.asarray(np.rint(pred), dtype=int)
        pred_i = np.clip(pred_i, 0, int(len(self._label_encoder.classes_) - 1))
        return self._label_encoder.inverse_transform(pred_i)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("Wrapped estimator does not expose predict_proba")
        return np.asarray(self.estimator.predict_proba(np.asarray(X, dtype=float)), dtype=float)


def classify_regime(n_samples: int, n_features: int) -> str:
    """Classify dataset regime used for classifier-family gating."""
    n = int(max(1, n_samples))
    p = int(max(0, n_features))
    p_over_n = float(p) / float(max(1, n))
    if n < 50 or p_over_n > 500.0:
        return REGIME_HDLSS_EXTREME
    if n < 200 and p_over_n > 50.0:
        return REGIME_HDLSS_MODERATE
    return REGIME_STANDARD


def _pairwise_pref_from_fold_scores(scores_i: np.ndarray, scores_j: np.ndarray, *, pairwise_delta: float = 0.01) -> float:
    """Empirical pairwise preference used by MNPO performance oracles."""
    return float(
        pairwise_pref_from_fold_scores(
            scores_i,
            scores_j,
            pairwise_delta=float(pairwise_delta),
        )
    )


def _unique_with_alias_handling(names: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Deduplicate candidates while collapsing alias groups (T-CS-028)."""
    out: List[str] = []
    dropped: List[str] = []
    seen_names: Set[str] = set()
    seen_groups: Set[str] = set()
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        group = _ALIAS_GROUPS.get(name, name)
        if group in seen_groups:
            dropped.append(name)
            continue
        if name in seen_names:
            continue
        out.append(name)
        seen_names.add(name)
        seen_groups.add(group)
    return out, dropped


def _format_exception_summary(exc: BaseException) -> str:
    detail = str(exc).strip()
    name = type(exc).__name__
    return name if not detail else f"{name}: {detail}"


def _normalize_optional_model_build_result(result: OptionalModelBuildReturn) -> OptionalModelBuildResult:
    if isinstance(result, tuple) and len(result) == 2:
        model, reason = result
        return (
            model if isinstance(model, BaseEstimator) or model is None else None,
            None if reason is None else str(reason),
        )
    model = result if isinstance(result, BaseEstimator) or result is None else None
    return model, None


def _james_stein_shrinkage(
    weights: Dict[str, float],
    *,
    effective_n: Optional[float] = None,
) -> Dict[str, float]:
    """James-Stein shrinkage for oracle weights (T-R-224)."""
    return james_stein_shrinkage(weights, effective_n=effective_n)


@dataclass(frozen=True)
class OracleCandidateStats:
    name: str
    scores: np.ndarray
    mean_score: float
    std_score: float
    min_mean_ratio: float
    complexity_score: float
    calibration_score: float
    ece_score: float  # Expected Calibration Error (VAL12_Suggestions §2.1)
    bbc_corrected_score: float
    bbc_ci_low: float
    bbc_ci_high: float


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """Minimal PLS-DA classifier wrapper using one-vs-rest targets."""

    def __init__(self, n_components: int = 2, scale: bool = True):
        self.n_components = int(max(1, n_components))
        self.scale = bool(scale)

    def fit(self, X: np.ndarray, y: np.ndarray):
        x = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).ravel()
        self.classes_ = np.unique(y_arr)
        if self.classes_.size < 2:
            raise ValueError("PLSDAClassifier requires at least two classes.")

        y_one_hot = np.zeros((y_arr.size, self.classes_.size), dtype=float)
        for idx, cls in enumerate(self.classes_):
            y_one_hot[:, idx] = (y_arr == cls).astype(float)

        max_components = int(max(1, min(x.shape[1], x.shape[0] - 1, self.classes_.size)))
        n_comp = int(max(1, min(self.n_components, max_components)))
        self._model = PLSRegression(n_components=n_comp, scale=self.scale)
        self._model.fit(x, y_one_hot)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_score = np.asarray(self._model.predict(np.asarray(X, dtype=float)), dtype=float)
        if y_score.ndim == 1:
            y_score = y_score[:, None]
        if y_score.shape[1] == 1:
            return np.where(y_score.ravel() >= 0.5, self.classes_[1], self.classes_[0])
        best = np.argmax(y_score, axis=1)
        return np.asarray([self.classes_[i] for i in best])


class DistanceBasedDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    """DBDA — Aoshima & Yata (2014). Bias-corrected distance classifier for HDLSS.

    Assigns each sample to the class minimising
    ``||x - mean_i||^2 - tr(S_i) / n_i``, where the second term corrects
    the systematic bias of the squared distance in high-dimensional settings.

    Reference: M. Aoshima and K. Yata, "A distance-based, misclassification
    rate adjusted classifier for multiclass, high-dimensional data",
    Ann. Inst. Stat. Math. (2014).
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.class_stats_: List[Tuple[np.ndarray, float, int]] = []
        for c in self.classes_:
            Xc = X[y == c]
            mean = Xc.mean(axis=0)
            tr_cov = float(np.sum((Xc - mean) ** 2) / max(1, Xc.shape[0] - 1))
            self.class_stats_.append((mean, tr_cov, int(Xc.shape[0])))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([
            np.sum((X - mu) ** 2, axis=1) - tr_s / max(1, n)
            for mu, tr_s, n in self.class_stats_
        ])
        return self.classes_[np.argmin(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([
            np.sum((X - mu) ** 2, axis=1) - tr_s / max(1, n)
            for mu, tr_s, n in self.class_stats_
        ])
        # Convert negative scores to probabilities via softmax on negated scores.
        neg = -scores
        neg -= neg.max(axis=1, keepdims=True)
        exp = np.exp(neg)
        return exp / exp.sum(axis=1, keepdims=True)


class GeometricalQuadraticDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    """GQDA — Aoshima & Yata (2015). Bias-corrected geometric QDA for HDLSS.

    Assigns each sample to the class minimising
    ``d * ||x - mean_i||^2 / tr(S_i) + d * log(tr(S_i)) - d / n_i``.
    Extends DBDA to heteroscedastic class covariances.

    Reference: M. Aoshima and K. Yata, "Geometric Classifier for Multiclass,
    High-Dimensional Data", Sequential Anal. 34, 279-294 (2015).
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_features_ = int(X.shape[1])
        self.class_stats_: List[Tuple[np.ndarray, float, int]] = []
        for c in self.classes_:
            Xc = X[y == c]
            mean = Xc.mean(axis=0)
            tr_cov = float(np.sum((Xc - mean) ** 2) / max(1, Xc.shape[0] - 1))
            self.class_stats_.append((mean, tr_cov, int(Xc.shape[0])))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        d = self.n_features_
        scores = np.column_stack([
            d * np.sum((X - mu) ** 2, axis=1) / max(tr_s, 1e-12)
            + d * np.log(max(tr_s, 1e-12)) - d / max(1, n)
            for mu, tr_s, n in self.class_stats_
        ])
        return self.classes_[np.argmin(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        d = self.n_features_
        scores = np.column_stack([
            d * np.sum((X - mu) ** 2, axis=1) / max(tr_s, 1e-12)
            + d * np.log(max(tr_s, 1e-12)) - d / max(1, n)
            for mu, tr_s, n in self.class_stats_
        ])
        neg = -scores
        neg -= neg.max(axis=1, keepdims=True)
        exp = np.exp(neg)
        return exp / exp.sum(axis=1, keepdims=True)


class BiasCorrectedLinearSVM(ClassifierMixin, BaseEstimator):
    """BC-SVM (linear kernel) — Nakayama, Yata & Aoshima (2017).

    Wraps a standard linear SVM and shifts the decision function by a
    bias-correction term derived from the kernel Gram matrices of each
    class.  The correction compensates for the systematic decision-
    boundary shift caused by the curse of dimensionality in p >> n.

    Binary only; use with OVA/ECOC wrapping for multiclass.

    Reference: Y. Nakayama, K. Yata, M. Aoshima, "Support vector machine
    and its bias correction in high-dimension, low-sample-size settings",
    J. Statist. Plann. Inference 191 (2017) 88–100.
    """

    def __init__(self, *, C: float = 1.0, random_state: Optional[int] = None):
        self.C = C
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("BiasCorrectedLinearSVM supports binary classification only.")
        self._svc = SVC(
            kernel="linear", C=float(self.C),
            class_weight="balanced", random_state=self.random_state,
        )
        self._svc.fit(X, y)
        # Compute bias-correction term from linear kernel Gram matrices.
        X1 = X[y == self.classes_[0]]
        X2 = X[y == self.classes_[1]]
        n1, n2 = X1.shape[0], X2.shape[0]
        G1 = X1 @ X1.T  # (n1, n1)
        G2 = X2 @ X2.T  # (n2, n2)
        G12 = X1 @ X2.T  # (n1, n2)
        eta1 = (np.trace(G1) / max(1, n1 - 1)
                - np.sum(G1) / max(1, n1 * (n1 - 1))) if n1 > 1 else 0.0
        eta2 = (np.trace(G2) / max(1, n2 - 1)
                - np.sum(G2) / max(1, n2 * (n2 - 1))) if n2 > 1 else 0.0
        delta_num = eta1 / max(1, n1) - eta2 / max(1, n2)
        capital_delta = (
            np.sum(G1) / max(1, n1 ** 2)
            + np.sum(G2) / max(1, n2 ** 2)
            - 2.0 * np.sum(G12) / max(1, n1 * n2)
        )
        self.bc_term_ = float(delta_num / capital_delta) if abs(capital_delta) > 1e-10 else 0.0
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._svc.decision_function(np.asarray(X, dtype=float))) - self.bc_term_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores < 0, self.classes_[0], self.classes_[1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        # Platt-style sigmoid calibration.
        prob_class1 = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - prob_class1, prob_class1])


class RandomProjectionEnsembleClassifier(ClassifierMixin, BaseEstimator):
    """Random-projection soft-voting ensemble for HDLSS stabilization."""

    def __init__(
        self,
        n_estimators: int = 9,
        n_components: Optional[int] = None,
        max_components: int = 64,
        random_state: Optional[int] = None,
        lr_max_iter: int = 5000,
    ):
        self.n_estimators = int(max(1, n_estimators))
        self.n_components = None if n_components is None else int(max(2, n_components))
        self.max_components = int(max(2, max_components))
        self.random_state = None if random_state is None else int(random_state)
        self.lr_max_iter = int(max(500, lr_max_iter))

    def _resolve_n_components(self, X: np.ndarray) -> int:
        n_samples, n_features = np.asarray(X, dtype=float).shape
        if self.n_components is not None:
            return int(max(2, min(self.n_components, n_features)))
        auto = int(math.ceil(math.sqrt(max(2, n_features))))
        auto = int(min(auto, self.max_components, n_features))
        if n_samples > 2:
            auto = int(min(auto, max(2, n_samples - 1)))
        return int(max(2, auto))

    def fit(self, X: np.ndarray, y: np.ndarray):
        x = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).ravel()
        if hasattr(self, "_fallback_estimator"):
            delattr(self, "_fallback_estimator")
        self.classes_ = np.unique(y_arr)
        if self.classes_.size < 2:
            raise ValueError("RandomProjectionEnsembleClassifier requires at least two classes.")

        n_components = self._resolve_n_components(x)
        self.n_features_in_ = int(x.shape[1])
        self.members_: List[Tuple[GaussianRandomProjection, BaseEstimator]] = []
        self.member_projection_dim_ = int(n_components)
        rng_seed = 0 if self.random_state is None else int(self.random_state)

        for est_idx in range(int(self.n_estimators)):
            projector = GaussianRandomProjection(
                n_components=int(n_components),
                random_state=int(rng_seed + est_idx),
            )
            estimator = make_pipeline(
                StandardScaler(),
                make_logistic_regression(
                    random_state=int(rng_seed + est_idx),
                    max_iter=int(self.lr_max_iter),
                    solver="lbfgs",
                    penalty="l2",
                    class_weight="balanced",
                ),
            )
            try:
                x_proj = projector.fit_transform(x)
                estimator.fit(x_proj, y_arr)
                self.members_.append((projector, estimator))
            except Exception:
                continue

        if not self.members_:
            fallback = make_pipeline(
                StandardScaler(),
                make_logistic_regression(
                    random_state=int(rng_seed),
                    max_iter=int(self.lr_max_iter),
                    solver="lbfgs",
                    penalty="l2",
                    class_weight="balanced",
                ),
            )
            fallback.fit(x, y_arr)
            self._fallback_estimator = fallback
            self.members_ = []
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if hasattr(self, "_fallback_estimator"):
            return np.asarray(self._fallback_estimator.predict_proba(x), dtype=float)
        if not getattr(self, "members_", None):
            raise AttributeError("Model is not fitted.")

        probs_accum: Optional[np.ndarray] = None
        for projector, estimator in self.members_:
            x_proj = projector.transform(x)
            member_probs = np.asarray(estimator.predict_proba(x_proj), dtype=float)
            if probs_accum is None:
                probs_accum = np.zeros_like(member_probs, dtype=float)
            probs_accum += member_probs
        assert probs_accum is not None
        probs_accum /= float(max(1, len(self.members_)))
        row_sums = np.sum(probs_accum, axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
        return probs_accum / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        best = np.argmax(probs, axis=1)
        return np.asarray([self.classes_[int(i)] for i in best])


class SparseGroupLassoNNClassifier(ClassifierMixin, BaseEstimator):
    """Sparse Group Lasso Neural Network (SGLNN) — Yang (2020).

    Single-hidden-layer neural network with a sparse group lasso (L1 + L2,1)
    penalty on the input weight matrix, producing true feature-level sparsity.
    Designed for HDLSS classification where p >> n.

    Reference: Yang, K. (2020). *Statistical Machine Learning Theory and
    Methods for High-Dimensional, Low-Sample-Size Problems*.  PhD dissertation,
    Michigan State University, Ch. 3.  Theorem 3.1 guarantees classification
    risk converges to the Bayes risk under HDLSS sparsity assumptions.

    Parameters
    ----------
    n_hidden : int or None
        Number of hidden units.  ``None`` → ``min(50, max(5, n_samples // 3))``.
    lambda_sgl : float
        Sparse group lasso penalty strength.  Cross-validated when ``cv_lambda``
        is True.
    alpha_mix : float in (0, 1)
        Mixing ratio between L1 (``alpha_mix``) and group L2 (``1 - alpha_mix``)
        penalty components.
    max_iter : int
        Maximum coordinate-descent iterations over the input weight matrix.
    tol : float
        Convergence tolerance on the relative change of the loss.
    cv_lambda : bool
        If True, choose ``lambda_sgl`` via internal 3-fold stratified CV from a
        logarithmic grid.
    random_state : int or None
        Seed for weight initialisation and CV splitting.
    """

    def __init__(
        self,
        n_hidden: Optional[int] = None,
        lambda_sgl: float = 0.01,
        alpha_mix: float = 0.5,
        max_iter: int = 200,
        tol: float = 1e-4,
        cv_lambda: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_hidden = n_hidden
        self.lambda_sgl = float(max(0.0, lambda_sgl))
        self.alpha_mix = float(min(1.0, max(0.0, alpha_mix)))
        self.max_iter = int(max(1, max_iter))
        self.tol = float(max(1e-12, tol))
        self.cv_lambda = cv_lambda
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Row-wise softmax with numerical stability."""
        z_shift = z - z.max(axis=1, keepdims=True)
        e = np.exp(z_shift)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _soft_threshold(v: np.ndarray, lam: float) -> np.ndarray:
        """Element-wise soft-thresholding operator."""
        return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)

    def _resolve_n_hidden(self, n_samples: int) -> int:
        if self.n_hidden is not None:
            return int(max(2, self.n_hidden))
        return int(min(50, max(5, n_samples // 3)))

    def _init_weights(self, p: int, m: int, k: int, rng: np.random.RandomState):
        """Xavier-style initialisation for input (theta) and output (beta)."""
        scale_theta = float(np.sqrt(2.0 / (p + m)))
        theta = rng.randn(p, m).astype(float) * scale_theta
        scale_beta = float(np.sqrt(2.0 / (m + k)))
        beta = rng.randn(m, k).astype(float) * scale_beta
        b_hidden = np.zeros(m, dtype=float)
        b_out = np.zeros(k, dtype=float)
        return theta, beta, b_hidden, b_out

    def _forward(
        self,
        X: np.ndarray,
        theta: np.ndarray,
        beta: np.ndarray,
        b_hidden: np.ndarray,
        b_out: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass returning (hidden_pre, hidden_act, logits)."""
        h_pre = X @ theta + b_hidden  # (n, m)
        h_act = np.maximum(h_pre, 0.0)  # ReLU
        logits = h_act @ beta + b_out  # (n, k)
        return h_pre, h_act, logits

    def _cross_entropy_loss(
        self,
        probs: np.ndarray,
        Y_onehot: np.ndarray,
        theta: np.ndarray,
        lam: float,
    ) -> float:
        """Cross-entropy + sparse group lasso on theta."""
        n = float(probs.shape[0])
        ce = -float(np.sum(Y_onehot * np.log(np.clip(probs, 1e-12, 1.0)))) / n
        # L1 term
        l1 = float(np.sum(np.abs(theta)))
        # Group L2 term: column-wise L2 norms (each column = one hidden unit group)
        col_norms = np.sqrt(np.sum(theta ** 2, axis=0))
        group_l2 = float(np.sum(col_norms))
        penalty = lam * (self.alpha_mix * l1 + (1.0 - self.alpha_mix) * group_l2)
        return ce + penalty

    def _fit_one(
        self,
        X: np.ndarray,
        Y_onehot: np.ndarray,
        lam: float,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Train a single SGLNN with fixed lambda via coordinate descent on theta."""
        n_samples, p = X.shape
        k = Y_onehot.shape[1]
        m = self._resolve_n_hidden(n_samples)

        theta, beta, b_hidden, b_out = self._init_weights(p, m, k, rng)
        lr_beta = 0.01
        prev_loss = float("inf")

        for iteration in range(self.max_iter):
            # -- Forward --
            h_pre, h_act, logits = self._forward(X, theta, beta, b_hidden, b_out)
            probs = self._softmax(logits)

            # -- Output layer gradient (standard gradient descent) --
            d_logits = (probs - Y_onehot) / float(n_samples)  # (n, k)
            grad_beta = h_act.T @ d_logits  # (m, k)
            grad_b_out = d_logits.sum(axis=0)  # (k,)
            beta -= lr_beta * grad_beta
            b_out -= lr_beta * grad_b_out

            # -- Hidden layer backprop --
            d_hidden = d_logits @ beta.T  # (n, m)
            d_hidden[h_pre <= 0.0] = 0.0  # ReLU derivative

            grad_theta = X.T @ d_hidden  # (p, m)
            grad_b_hidden = d_hidden.sum(axis=0)  # (m,)
            b_hidden -= lr_beta * grad_b_hidden

            # -- Proximal step on theta (sparse group lasso) --
            # Step 1: gradient step
            theta_tilde = theta - lr_beta * grad_theta
            # Step 2: L1 soft-threshold
            l1_threshold = lam * self.alpha_mix * lr_beta
            theta_tilde = self._soft_threshold(theta_tilde, l1_threshold)
            # Step 3: group (column-wise) L2 proximal
            group_threshold = lam * (1.0 - self.alpha_mix) * lr_beta
            for j in range(m):
                col_norm = float(np.linalg.norm(theta_tilde[:, j]))
                if col_norm > group_threshold:
                    theta_tilde[:, j] *= (1.0 - group_threshold / col_norm)
                else:
                    theta_tilde[:, j] = 0.0
            theta = theta_tilde

            # -- Convergence check --
            loss = self._cross_entropy_loss(self._softmax(self._forward(X, theta, beta, b_hidden, b_out)[2]),
                                            Y_onehot, theta, lam)
            if abs(prev_loss - loss) / max(abs(prev_loss), 1e-12) < self.tol:
                break
            prev_loss = loss

        return theta, beta, b_hidden, b_out, loss

    def _cv_select_lambda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Y_onehot: np.ndarray,
        rng: np.random.RandomState,
    ) -> float:
        """3-fold stratified CV over a log-spaced lambda grid."""
        lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        best_lam = float(self.lambda_sgl)
        best_score = -1.0

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(rng.randint(0, 2**31)))
        for lam in lambdas:
            fold_scores: List[float] = []
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                Y_tr = Y_onehot[train_idx]
                theta, beta, b_h, b_o, _ = self._fit_one(X_tr, Y_tr, lam, rng)
                _, _, logits_val = self._forward(X_val, theta, beta, b_h, b_o)
                preds = np.argmax(logits_val, axis=1)
                fold_scores.append(float(balanced_accuracy_score(y[val_idx], preds)))
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_lam = lam
        return best_lam

    # ------------------------------------------------------------------
    # Sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_features_in_ = int(X.shape[1])
        if self.classes_.size < 2:
            raise ValueError("SparseGroupLassoNNClassifier requires at least two classes.")

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        k = int(self.classes_.size)
        Y_onehot = np.zeros((len(y_enc), k), dtype=float)
        Y_onehot[np.arange(len(y_enc)), y_enc] = 1.0

        rng = np.random.RandomState(self.random_state if self.random_state is not None else 0)

        # Optional CV for lambda selection
        lam = self.lambda_sgl
        if self.cv_lambda and X.shape[0] >= 9:
            lam = self._cv_select_lambda(X, y_enc, Y_onehot, rng)
        self.lambda_sgl_ = float(lam)

        self.theta_, self.beta_, self.b_hidden_, self.b_out_, self.train_loss_ = (
            self._fit_one(X, Y_onehot, lam, rng)
        )

        # Record selected feature mask (rows of theta with nonzero norm)
        row_norms = np.sqrt(np.sum(self.theta_ ** 2, axis=1))
        self.selected_features_ = np.where(row_norms > 1e-10)[0]
        self.n_selected_features_ = int(self.selected_features_.size)

        self._label_encoder = le
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        _, _, logits = self._forward(X, self.theta_, self.beta_, self.b_hidden_, self.b_out_)
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        best = np.argmax(probs, axis=1)
        return np.asarray([self.classes_[int(i)] for i in best])


class RandomFourierFeaturesClassifier(ClassifierMixin, BaseEstimator):
    """Approximate kernel classifier via Random Fourier Features (RFF) + LR.

    Maps inputs to a randomised feature space that approximates a Gaussian RBF
    kernel, then trains logistic regression in the lifted space.  This yields
    **nonlinear decision boundaries** while keeping the effective dimension
    controllable — critical for HDLSS safety.

    The only nonlinear classifier in the HDLSS-EXTREME pool besides SGLNN,
    adding a genuinely different inductive bias (kernel approximation vs
    sparse neural-network).

    Reference: A. Rahimi and B. Recht, "Random Features for Large-Scale
    Kernel Machines", NIPS 2007.  See also Cannings & Samworth (2017, JRSSB)
    for theoretical analysis of random-projection ensembles in high dimensions.

    Parameters
    ----------
    n_features_rff : int or None
        Number of random Fourier features.  ``None`` → ``min(2 * sqrt(p), 256, 2n)``.
    gamma : str or float
        Kernel bandwidth.  ``'auto'`` → ``1 / (p * Var(X))``, same as sklearn
        ``gamma='scale'``.
    random_state : int or None
        Seed for reproducibility.
    lr_max_iter : int
        Maximum iterations for the downstream logistic regression.
    """

    def __init__(
        self,
        n_features_rff: Optional[int] = None,
        gamma: Union[str, float] = "auto",
        random_state: Optional[int] = None,
        lr_max_iter: int = 5000,
    ):
        self.n_features_rff = n_features_rff
        self.gamma = gamma
        self.random_state = random_state
        self.lr_max_iter = int(max(500, lr_max_iter))

    def _resolve_rff_dim(self, n_samples: int, n_features: int) -> int:
        if self.n_features_rff is not None:
            return int(max(4, self.n_features_rff))
        auto = int(min(2 * math.ceil(math.sqrt(max(2, n_features))), 256))
        auto = int(min(auto, max(4, 2 * n_samples)))
        return int(max(4, auto))

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, (int, float)):
            return float(max(1e-12, self.gamma))
        # 'auto' / 'scale' → 1 / (p * var)
        var = float(np.var(X))
        p = float(X.shape[1])
        return 1.0 / max(p * var, 1e-12)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("RandomFourierFeaturesClassifier requires >= 2 classes.")

        n_samples, n_features = X.shape
        self.n_features_in_ = int(n_features)

        # Standardise input
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)

        gamma = self._resolve_gamma(X_sc)
        D = self._resolve_rff_dim(n_samples, n_features)

        rng = np.random.RandomState(
            self.random_state if self.random_state is not None else 0
        )
        # Sample RFF weights from N(0, 2*gamma * I) and biases from U[0, 2*pi]
        self.omega_ = rng.randn(n_features, D).astype(float) * math.sqrt(2.0 * gamma)
        self.bias_ = rng.uniform(0.0, 2.0 * math.pi, size=D).astype(float)
        self.rff_scale_ = math.sqrt(2.0 / D)

        Z = self.rff_scale_ * np.cos(X_sc @ self.omega_ + self.bias_)

        self._lr = make_logistic_regression(
            random_state=self.random_state if self.random_state is not None else 0,
            max_iter=self.lr_max_iter,
            solver="lbfgs",
            penalty="l2",
            class_weight="balanced",
        )
        self._lr.fit(Z, y)
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(np.asarray(X, dtype=float))
        return self.rff_scale_ * np.cos(X_sc @ self.omega_ + self.bias_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._lr.predict_proba(self._transform(X)), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._lr.predict(self._transform(X))


class NearestSubspaceClassifier(ClassifierMixin, BaseEstimator):
    """Classify by minimal reconstruction error from per-class principal subspaces.

    For each class, compute the top-*k* principal components of the centred
    within-class data.  A query is classified by projecting onto every class
    subspace and choosing the class whose subspace gives the smallest residual
    (reconstruction error).

    This produces a fundamentally different decision boundary from
    centroid-based classifiers (LDA, NSC):  a sample near the *manifold* of
    class A but far from its centroid is assigned to A — the opposite of what
    LDA would do.

    Reference: E. Oja, "Subspace Methods of Pattern Recognition", 1983.
    Also related to CRC (Zhang et al., 2012, arXiv:1204.2358) and Roy et al.
    (2019, arXiv:1902.03295) on distance-based generalisations for HDLSS.

    Parameters
    ----------
    n_components : int or None
        Number of principal components per class subspace.
        ``None`` → ``min(n_class - 1, 10)``.
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_features_in_ = int(X.shape[1])

        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)

        self.subspaces_: List[Tuple[np.ndarray, np.ndarray]] = []  # (mean, basis)
        for c in self.classes_:
            Xc = X_sc[y == c]
            mean_c = Xc.mean(axis=0)
            Xc_centred = Xc - mean_c
            n_c = Xc_centred.shape[0]

            # Resolve rank
            if self.n_components is not None:
                k = int(max(1, min(self.n_components, n_c - 1, Xc_centred.shape[1])))
            else:
                k = int(max(1, min(n_c - 1, 10, Xc_centred.shape[1])))

            if k >= min(n_c, Xc_centred.shape[1]):
                # Full rank — orthonormalise for a valid projector
                Q, _ = np.linalg.qr(Xc_centred.T, mode="reduced")
                self.subspaces_.append((mean_c, Q))  # (p, min(n_c, p))
            else:
                # Truncated SVD via economy SVD
                # In HDLSS (p >> n), X^T X is (n_c, n_c) — cheaper to eigendecompose.
                if n_c <= Xc_centred.shape[1]:
                    G = Xc_centred @ Xc_centred.T  # (n_c, n_c)
                    eigvals, eigvecs = np.linalg.eigh(G)
                    # Take top-k (eigenvalues sorted ascending)
                    idx = np.argsort(eigvals)[::-1][:k]
                    # Compute principal directions in feature space
                    V = Xc_centred.T @ eigvecs[:, idx]  # (p, k)
                    # Normalise
                    norms = np.linalg.norm(V, axis=0, keepdims=True)
                    norms = np.where(norms < 1e-12, 1.0, norms)
                    V = V / norms
                else:
                    cov = Xc_centred.T @ Xc_centred / max(1, n_c - 1)
                    eigvals, V = np.linalg.eigh(cov)
                    idx = np.argsort(eigvals)[::-1][:k]
                    V = V[:, idx]
                self.subspaces_.append((mean_c, V))  # (p, k)
        return self

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_classes) residual norms."""
        X_sc = self._scaler.transform(np.asarray(X, dtype=float))
        errors = np.empty((X_sc.shape[0], len(self.classes_)), dtype=float)
        for i, (mean_c, basis) in enumerate(self.subspaces_):
            centred = X_sc - mean_c
            proj = centred @ basis @ basis.T  # project onto subspace
            residual = centred - proj
            errors[:, i] = np.sum(residual ** 2, axis=1)
        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        errors = self._reconstruction_errors(X)
        return self.classes_[np.argmin(errors, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        errors = self._reconstruction_errors(X)
        # Convert errors to probabilities via softmax on negated errors.
        neg = -errors
        neg -= neg.max(axis=1, keepdims=True)
        exp = np.exp(neg)
        return exp / exp.sum(axis=1, keepdims=True)


class SpatialMedianDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    """Robust distance classifier using spatial medians instead of sample means.

    The spatial median (also called the *L1-median* or *geometric median*) is
    the point minimising the sum of Euclidean distances to all observations.
    It has a breakdown point of 0.5, compared to 0 for the sample mean, making
    it resistant to outliers and high-leverage points.

    In HDLSS settings the "distance concentration" phenomenon affects means
    and medians differently. This implementation is a lightweight robust
    distance classifier inspired by existing spatial-/median-based
    high-dimensional classification work, rather than a claim of a new
    literature-standard method family.

    Classification rule:
        ``argmin_c  ||x - median_c||^2 - bias_c``
    where ``bias_c`` corrects for heteroscedastic class scatter, analogous
    to the GQDA bias term.

    References: P. Hall et al., "Median-Based Classifiers for High-Dimensional
    Data", JASA 104(488), 2009; S. Minsker, "Geometric median and robust
    estimation in Banach spaces", Bernoulli 21(4), 2015; Y. Vardi and
    C.-H. Zhang, "The multivariate L1-median and associated data depth",
    PNAS 97(4), 2000.
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-6):
        self.max_iter = int(max(10, max_iter))
        self.tol = float(max(1e-12, tol))

    @staticmethod
    def _spatial_median(X: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
        """Weiszfeld's algorithm for the geometric (spatial) median."""
        n = X.shape[0]
        if n == 1:
            return X[0].copy()
        median = np.mean(X, axis=0).copy()
        for _ in range(max_iter):
            diffs = X - median
            dists = np.linalg.norm(diffs, axis=1, keepdims=True)
            dists = np.where(dists < 1e-12, 1e-12, dists)
            weights = 1.0 / dists
            new_median = np.sum(X * weights, axis=0) / np.sum(weights)
            if np.linalg.norm(new_median - median) < tol:
                median = new_median
                break
            median = new_median
        return median

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_features_in_ = int(X.shape[1])

        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)

        self.class_stats_: List[Tuple[np.ndarray, float, int]] = []
        for c in self.classes_:
            Xc = X_sc[y == c]
            med = self._spatial_median(Xc, self.max_iter, self.tol)
            # Mean squared radial scatter around the spatial median.
            dists_to_med = np.sqrt(np.sum((Xc - med) ** 2, axis=1))
            scatter = float(np.mean(dists_to_med ** 2)) if Xc.shape[0] > 1 else 1.0
            self.class_stats_.append((med, scatter, int(Xc.shape[0])))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(np.asarray(X, dtype=float))
        scores = np.column_stack([
            np.sum((X_sc - med) ** 2, axis=1) - scatter / max(1, n)
            for med, scatter, n in self.class_stats_
        ])
        return self.classes_[np.argmin(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(np.asarray(X, dtype=float))
        scores = np.column_stack([
            np.sum((X_sc - med) ** 2, axis=1) - scatter / max(1, n)
            for med, scatter, n in self.class_stats_
        ])
        neg = -scores
        neg -= neg.max(axis=1, keepdims=True)
        exp = np.exp(neg)
        return exp / exp.sum(axis=1, keepdims=True)


class CopulaDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    """Discriminant analysis exploiting Gaussian-copula-style structure.

    The DF pipeline transforms each feature via ``Phi^{-1}(F_hat(x))``,
    mapping arbitrary marginals to standard Gaussian.  This is exactly a
    Gaussian copula transform.  CopulaDA fits a class-specific Gaussian model
    on these CDF-transformed features,
    which is equivalent to fitting a **Gaussian copula** per class on the
    original data.

    This backend is intended for the DF-transformed feature space.  When the
    upstream CDF-to-Gaussian transform is disabled, the classifier reduces to a
    shrinkage-Gaussian discriminant model on the raw stage-2 inputs.

    Unlike standard LDA which assumes Gaussian features, CopulaDA allows
    **arbitrary marginal distributions** (exponential, Weibull, etc.) while
    only requiring the *dependence structure* to be Gaussian.  This is a
    much weaker (and more realistic) assumption for biological/tabular data.

    The classifier uses Bayes' rule with the copula log-likelihoods.

    This backend is a simplified Gaussian-copula discriminant classifier in the
    style of CODA and related semiparametric copula-DA work. It uses a direct
    classwise Gaussian fit after the upstream marginal Gaussianization step,
    rather than reproducing the full estimation machinery from those papers.

    References: F. Han, T. Zhao, and H. Liu, "CODA: High Dimensional Copula
    Discriminant Analysis", JMLR 14, 2013; F. Tekle and R. de Leon,
    "Gaussian copula distributions for mixed data, with application in
    discrimination", J. Stat. Comput. Simul. 86(9), 2016; L. Wang et al.,
    "High-dimensional integrative copula discriminant analysis for multiomics
    data", Stat. Med. 39(30), 2020.

    Parameters
    ----------
    shrinkage : str
        Covariance shrinkage method.  ``'ledoit_wolf'`` (default) or ``'oas'``.
    """

    def __init__(self, shrinkage: str = "ledoit_wolf"):
        self.shrinkage = shrinkage

    def _shrink_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute shrinkage covariance matrix."""
        n, p = X.shape
        if n <= 1:
            return np.eye(p, dtype=float)
        sample_cov = np.cov(X, rowvar=False, ddof=1)
        if sample_cov.ndim == 0:
            return np.array([[float(sample_cov)]], dtype=float)
        shrinkage = str(self.shrinkage or "ledoit_wolf").strip().lower()
        if shrinkage == "ledoit_wolf":
            self.shrinkage_ = "ledoit_wolf"
            return np.asarray(LedoitWolf().fit(X).covariance_, dtype=float)
        if shrinkage == "oas":
            self.shrinkage_ = "oas"
            return np.asarray(OAS().fit(X).covariance_, dtype=float)
        raise ValueError(
            "CopulaDiscriminantAnalysis shrinkage must be 'ledoit_wolf' or 'oas'."
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        n_total = len(y)
        self.n_features_in_ = int(X.shape[1])

        self.class_params_: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        for c in self.classes_:
            Xc = X[y == c]
            mean_c = Xc.mean(axis=0)
            cov_c = self._shrink_covariance(Xc)
            # Precompute inverse and log-determinant for scoring
            try:
                cov_inv = np.linalg.inv(cov_c)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_c)
            sign, logdet = np.linalg.slogdet(cov_c)
            logdet_val = float(logdet) if sign > 0 else 0.0
            prior = float(Xc.shape[0]) / max(1, n_total)
            self.class_params_.append((mean_c, cov_inv, logdet_val, prior))
        return self

    def _log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_classes) log-likelihood scores."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        p = X.shape[1]
        ll = np.empty((n, len(self.classes_)), dtype=float)
        for i, (mean_c, cov_inv, logdet_val, prior) in enumerate(self.class_params_):
            diff = X - mean_c
            mahal = np.sum(diff @ cov_inv * diff, axis=1)
            ll[:, i] = -0.5 * (mahal + logdet_val + p * np.log(2.0 * np.pi)) + np.log(max(prior, 1e-12))
        return ll

    def predict(self, X: np.ndarray) -> np.ndarray:
        ll = self._log_likelihoods(X)
        return self.classes_[np.argmax(ll, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ll = self._log_likelihoods(X)
        ll -= ll.max(axis=1, keepdims=True)
        exp = np.exp(ll)
        return exp / exp.sum(axis=1, keepdims=True)


class TabMClassifier(ClassifierMixin, BaseEstimator):
    """Lightweight multi-head MLP inspired by TabM.

    The official TabM line uses efficient ensembling with multiple predictions
    per object and, by default, heavy parameter sharing across the implicit
    ensemble members. This numpy backend borrows the central idea of
    per-head multiplicative feature modulation plus averaged multi-head
    predictions, but it is intentionally a simplified, resource-bounded
    approximation rather than a faithful reimplementation of the official
    architecture.

    Each head sees ``x * scale_m`` where ``scale_m`` is a learned per-head
    per-feature scaling vector, passes the modulated inputs through a small
    MLP, and contributes logits that are averaged across heads.

    Reference: Y. Gorishniy et al., "TabM: Advancing Tabular Deep Learning
    With Parameter-Efficient Ensembling", arXiv:2410.24210, 2024.

    Parameters
    ----------
    n_heads : int or None
        Number of ensemble heads.  ``None`` → ``min(8, max(2, n // 5))``.
    n_hidden : int or None
        Hidden layer width per head.  ``None`` → ``min(64, max(8, n // 2))``.
    weight_decay : float
        L2 regularisation strength on all weight matrices.
    max_iter : int
        Maximum training epochs.
    lr : float
        Learning rate for SGD.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_heads: Optional[int] = None,
        n_hidden: Optional[int] = None,
        weight_decay: float = 1e-3,
        max_iter: int = 300,
        lr: float = 0.01,
        random_state: Optional[int] = None,
    ):
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.weight_decay = float(max(0.0, weight_decay))
        self.max_iter = int(max(1, max_iter))
        self.lr = float(max(1e-8, lr))
        self.random_state = random_state

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_shift = z - z.max(axis=1, keepdims=True)
        e = np.exp(z_shift)
        return e / e.sum(axis=1, keepdims=True)

    def _resolve_dims(self, n: int, p: int):
        heads = self.n_heads if self.n_heads is not None else int(min(8, max(2, n // 5)))
        hidden = self.n_hidden if self.n_hidden is not None else int(min(64, max(8, n // 2)))
        return int(max(2, heads)), int(max(4, hidden))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_features_in_ = int(X.shape[1])
        n, p = X.shape
        k = int(self.classes_.size)
        if k < 2:
            raise ValueError("TabMClassifier requires >= 2 classes.")

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        Y = np.zeros((n, k), dtype=float)
        Y[np.arange(n), y_enc] = 1.0

        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)

        M, H = self._resolve_dims(n, p)
        rng = np.random.RandomState(
            self.random_state if self.random_state is not None else 0
        )

        # Per-head multiplicative feature scales: shape (M, p)
        self.scales_ = rng.uniform(0.5, 1.5, size=(M, p)).astype(float)

        # Shared architecture init per head: W1 (p,H), b1 (H,), W2 (H,k), b2 (k,)
        sc1 = float(np.sqrt(2.0 / (p + H)))
        sc2 = float(np.sqrt(2.0 / (H + k)))
        self.heads_: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for m in range(M):
            W1 = rng.randn(p, H).astype(float) * sc1
            b1 = np.zeros(H, dtype=float)
            W2 = rng.randn(H, k).astype(float) * sc2
            b2 = np.zeros(k, dtype=float)
            self.heads_.append((self.scales_[m], W1, b1, W2, b2))

        # Train each head
        for epoch in range(self.max_iter):
            for m_idx in range(M):
                scl, W1, b1, W2, b2 = self.heads_[m_idx]
                X_m = X_sc * scl  # per-head feature scaling
                h = np.maximum(X_m @ W1 + b1, 0.0)  # ReLU
                logits = h @ W2 + b2
                probs = self._softmax(logits)

                # Backprop
                dl = (probs - Y) / float(n)
                gW2 = h.T @ dl + self.weight_decay * W2
                gb2 = dl.sum(axis=0)
                dh = dl @ W2.T
                dh[h <= 0.0] = 0.0
                gW1 = X_m.T @ dh + self.weight_decay * W1
                gb1 = dh.sum(axis=0)
                # Scale gradient: d(loss)/d(scale_j) = sum_i x_ij * dh_ij @ W1[j,:]
                g_scl = np.sum(X_sc * (dh @ W1.T), axis=0)

                W1 -= self.lr * gW1
                b1 -= self.lr * gb1
                W2 -= self.lr * gW2
                b2 -= self.lr * gb2
                scl_new = scl - self.lr * g_scl
                # Clamp scales to (0.01, 5.0) for stability
                scl_new = np.clip(scl_new, 0.01, 5.0)
                self.heads_[m_idx] = (scl_new, W1, b1, W2, b2)

        self.scales_ = np.vstack([head[0] for head in self.heads_])
        self._label_encoder = le
        return self

    def _predict_logits(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(np.asarray(X, dtype=float))
        logits_sum = np.zeros((X_sc.shape[0], int(self.classes_.size)), dtype=float)
        for scl, W1, b1, W2, b2 in self.heads_:
            X_m = X_sc * scl
            h = np.maximum(X_m @ W1 + b1, 0.0)
            logits_sum += h @ W2 + b2
        return logits_sum / float(len(self.heads_))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._predict_logits(X)
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class RealMLPClassifier(ClassifierMixin, BaseEstimator):
    """Lightweight regularised deep MLP inspired by RealMLP.

    This backend is a small numpy-only approximation of the RealMLP /
    "Better by Default" family rather than a full reproduction of the official
    recipe. It keeps the main HDLSS-relevant ingredients we can support
    cheaply here: deep MLP blocks, per-sample layer normalisation, dropout,
    and L2 regularisation.

    Architecture (numpy-only, no PyTorch dependency):
      Input → [LayerNorm → Linear → ReLU → Dropout] × depth → Linear → Softmax

    HDLSS safety:
    - Hidden widths auto-scale to ``min(n, max(16, ceil(sqrt(p))))``
    - Aggressive dropout (0.3–0.5) prevents overfitting on small samples
    - Layer normalisation stabilises optimisation with small batches
    - L2 weight decay on all weight matrices

    Gradient handling is also intentionally simplified for robustness and low
    dependency surface: dropout backprop uses the expected-value scaling and
    the layer-normalisation derivative is approximated.

    This classifier adds **deep supervised representation learning** to the
    HDLSS pool with a simpler implementation than the official RealMLP stack.

    References: Y. Gorishniy et al., "Revisiting Deep Learning Models for
    Tabular Data", NeurIPS 2021; D. Holzmuller et al., "Better by Default:
    Strong Pre-Tuned MLP Baseline for Tabular Prediction", 2024.

    Parameters
    ----------
    depth : int
        Number of hidden layers (default 2).
    n_hidden : int or None
        Hidden layer width.  ``None`` → ``min(n, max(16, ceil(sqrt(p))))``.
    dropout : float
        Dropout rate applied after each hidden layer during training.
    weight_decay : float
        L2 regularisation strength.
    max_iter : int
        Maximum training epochs.
    lr : float
        Learning rate for SGD.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        depth: int = 2,
        n_hidden: Optional[int] = None,
        dropout: float = 0.3,
        weight_decay: float = 1e-3,
        max_iter: int = 400,
        lr: float = 0.01,
        random_state: Optional[int] = None,
    ):
        self.depth = int(max(1, depth))
        self.n_hidden = n_hidden
        self.dropout = float(min(0.9, max(0.0, dropout)))
        self.weight_decay = float(max(0.0, weight_decay))
        self.max_iter = int(max(1, max_iter))
        self.lr = float(max(1e-8, lr))
        self.random_state = random_state

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_shift = z - z.max(axis=1, keepdims=True)
        e = np.exp(z_shift)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _layer_norm(X: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-sample layer normalisation (not per-batch)."""
        mu = X.mean(axis=1, keepdims=True)
        var = X.var(axis=1, keepdims=True)
        X_norm = (X - mu) / np.sqrt(var + eps)
        return X_norm, mu, var

    def _resolve_hidden(self, n: int, p: int) -> int:
        if self.n_hidden is not None:
            return int(max(4, self.n_hidden))
        return int(min(n, max(16, math.ceil(math.sqrt(p)))))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_features_in_ = int(X.shape[1])
        n, p = X.shape
        k = int(self.classes_.size)
        if k < 2:
            raise ValueError("RealMLPClassifier requires >= 2 classes.")

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        Y = np.zeros((n, k), dtype=float)
        Y[np.arange(n), y_enc] = 1.0

        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)

        H = self._resolve_hidden(n, p)
        rng = np.random.RandomState(
            self.random_state if self.random_state is not None else 0
        )

        # Build layer stack: (W, b) per layer
        # Layer 0: p -> H, Layers 1..(depth-1): H -> H, Output: H -> k
        layers: List[Tuple[np.ndarray, np.ndarray]] = []
        dims = [p] + [H] * self.depth + [k]
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            sc = float(np.sqrt(2.0 / (fan_in + fan_out)))
            W = rng.randn(fan_in, fan_out).astype(float) * sc
            b = np.zeros(fan_out, dtype=float)
            layers.append((W, b))

        # Training loop
        for epoch in range(self.max_iter):
            # Forward pass
            activations = [X_sc]
            pre_norms = []
            for i in range(len(layers) - 1):
                W, b = layers[i]
                z = activations[-1] @ W + b
                # Layer normalisation
                z_norm, _, _ = self._layer_norm(z)
                pre_norms.append(z_norm)
                # ReLU
                h = np.maximum(z_norm, 0.0)
                # Dropout (training mode: mask + scale)
                if self.dropout > 0:
                    mask = (rng.rand(*h.shape) > self.dropout).astype(float)
                    h = h * mask / max(1.0 - self.dropout, 1e-12)
                activations.append(h)
            # Output layer (no norm, no activation)
            W_out, b_out = layers[-1]
            logits = activations[-1] @ W_out + b_out
            probs = self._softmax(logits)

            # Backprop
            dl = (probs - Y) / float(n)  # (n, k)
            # Output layer gradients
            gW = activations[-1].T @ dl + self.weight_decay * W_out
            gb = dl.sum(axis=0)

            # Buffer pre-update weights for correct chain-rule propagation,
            # then apply the output-layer update.
            pre_update_weights = [layers[j][0] for j in range(len(layers))]
            layers[-1] = (W_out - self.lr * gW, b_out - self.lr * gb)

            # Hidden layers (reverse order)
            d_next = dl
            for i in range(len(layers) - 2, -1, -1):
                W, b = layers[i]
                W_above = pre_update_weights[i + 1]
                d_next = d_next @ W_above.T
                # Dropout derivative (same mask concept — approximate with scale)
                if self.dropout > 0:
                    d_next = d_next / max(1.0 - self.dropout, 1e-12)
                # ReLU derivative
                z_norm = pre_norms[i]
                d_next = d_next * (z_norm > 0).astype(float)
                # LayerNorm derivative (simplified — treat as identity for gradient)
                gW_i = activations[i].T @ d_next + self.weight_decay * W
                gb_i = d_next.sum(axis=0)
                layers[i] = (W - self.lr * gW_i, b - self.lr * gb_i)

        self.layers_ = layers
        self._label_encoder = le
        return self

    def _predict_logits(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(np.asarray(X, dtype=float))
        h = X_sc
        for i in range(len(self.layers_) - 1):
            W, b = self.layers_[i]
            z = h @ W + b
            z_norm, _, _ = self._layer_norm(z)
            h = np.maximum(z_norm, 0.0)
            # No dropout at inference
        W_out, b_out = self.layers_[-1]
        return h @ W_out + b_out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(self._predict_logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class CPDAClassifier(ClassifierMixin, BaseEstimator):
    """Confusion-Pursuit Discriminant Analysis.

    CPDA is an experimental error-conditioned refinement of shrinkage LDA.
    Each round fits a scaled shrinkage-LDA, measures out-of-fold errors when
    feasible, removes a small slice of features that repeatedly place
    misclassified samples closer to the predicted wrong class than the true
    class, then blends the round-wise probabilities at inference time.

    This implementation intentionally avoids stronger claims such as exact
    Sherman-Morrison leave-one-out updates. It uses ordinary out-of-fold
    predictions when class counts allow and falls back gracefully on the fitted
    round model for ultra-small folds.
    """

    def __init__(
        self,
        max_rounds: int = 4,
        elim_frac: float = 0.10,
        blend_alpha: float = 0.55,
        internal_cv: int = 5,
        min_misclassified: int = 3,
        min_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.max_rounds = int(max(1, max_rounds))
        self.elim_frac = float(min(0.50, max(0.01, elim_frac)))
        self.blend_alpha = float(min(0.95, max(0.05, blend_alpha)))
        self.internal_cv = int(max(2, internal_cv))
        self.min_misclassified = int(max(1, min_misclassified))
        self.min_features = min_features
        self.random_state = random_state

    @staticmethod
    def _normalize_proba(proba: np.ndarray) -> np.ndarray:
        arr = np.asarray(proba, dtype=float)
        arr = np.clip(arr, 0.0, None)
        denom = np.sum(arr, axis=1, keepdims=True)
        denom[denom <= 0.0] = 1.0
        return arr / denom

    @staticmethod
    def _make_lda() -> LinearDiscriminantAnalysis:
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    def _resolve_min_features(self, *, n_features: int, n_classes: int) -> int:
        if self.min_features is not None:
            return int(max(n_classes, min(int(self.min_features), int(n_features))))
        return int(max(n_classes, min(int(n_features), max(n_classes + 1, 2 * n_classes))))

    def _fit_round_model(
        self,
        X_sub: np.ndarray,
        y_enc: np.ndarray,
    ) -> Tuple[StandardScaler, LinearDiscriminantAnalysis, np.ndarray]:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(np.asarray(X_sub, dtype=float))
        model = self._make_lda()
        model.fit(X_sc, y_enc)
        return scaler, model, X_sc

    def _oof_predict_proba(
        self,
        X_sub: np.ndarray,
        y_enc: np.ndarray,
        *,
        full_scaler: StandardScaler,
        full_model: LinearDiscriminantAnalysis,
    ) -> Tuple[np.ndarray, str, int]:
        X_arr = np.asarray(X_sub, dtype=float)
        y_arr = np.asarray(y_enc, dtype=int).ravel()
        n = int(X_arr.shape[0])
        counts = np.bincount(y_arr, minlength=int(self.classes_.size))
        positive_counts = counts[counts > 0]
        min_count = int(np.min(positive_counts)) if positive_counts.size else 0
        n_splits = int(min(self.internal_cv, max(0, min_count)))
        if n_splits >= 2 and n >= n_splits:
            try:
                cv = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                proba = np.zeros((n, int(self.classes_.size)), dtype=float)
                seen = np.zeros(n, dtype=bool)
                for train_idx, test_idx in cv.split(X_arr, y_arr):
                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_arr[train_idx])
                    X_test_sc = scaler.transform(X_arr[test_idx])
                    model = self._make_lda()
                    model.fit(X_train_sc, y_arr[train_idx])
                    proba[test_idx] = self._normalize_proba(model.predict_proba(X_test_sc))
                    seen[test_idx] = True
                if bool(np.all(seen)):
                    return proba, "stratified_oof", int(n_splits)
            except Exception:
                pass
        fitted = self._normalize_proba(full_model.predict_proba(full_scaler.transform(X_arr)))
        return fitted, "resubstitution_fallback", 0

    def _feature_confusion_scores(
        self,
        X_sc: np.ndarray,
        y_enc: np.ndarray,
        oof_proba: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.argmax(np.asarray(oof_proba, dtype=float), axis=1)
        mis_idx = np.flatnonzero(preds != y_enc)
        n_features = int(X_sc.shape[1])
        scores = np.zeros(n_features, dtype=float)
        if mis_idx.size == 0:
            return scores, mis_idx

        class_means = np.zeros((int(self.classes_.size), n_features), dtype=float)
        for cls_idx in range(int(self.classes_.size)):
            cls_mask = y_enc == cls_idx
            if bool(np.any(cls_mask)):
                class_means[cls_idx] = np.mean(X_sc[cls_mask], axis=0)

        total_weight = 0.0
        for idx in mis_idx:
            true_cls = int(y_enc[idx])
            pred_cls = int(preds[idx])
            margin = float(oof_proba[idx, pred_cls] - oof_proba[idx, true_cls])
            weight = max(1e-6, margin)
            true_dist = np.abs(X_sc[idx] - class_means[true_cls])
            pred_dist = np.abs(X_sc[idx] - class_means[pred_cls])
            scores += weight * np.clip(true_dist - pred_dist, 0.0, None)
            total_weight += weight
        if total_weight > 0.0:
            scores /= float(total_weight)
        return scores, mis_idx

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).ravel()
        if X_arr.ndim != 2:
            raise ValueError("CPDAClassifier expects a 2D feature matrix.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y_arr)
        self.classes_ = np.asarray(self._label_encoder.classes_)
        n_samples, n_features = X_arr.shape
        n_classes = int(self.classes_.size)
        if n_classes < 2:
            raise ValueError("CPDAClassifier requires at least two classes.")

        active = np.arange(n_features, dtype=int)
        min_features = self._resolve_min_features(n_features=n_features, n_classes=n_classes)

        round_states: List[Dict[str, Any]] = []
        round_scores: List[float] = []
        stop_reason = "max_rounds"

        for round_idx in range(self.max_rounds):
            X_sub = X_arr[:, active]
            scaler, model, X_sc = self._fit_round_model(X_sub, y_enc)
            oof_proba, diagnostic_mode, diagnostic_splits = self._oof_predict_proba(
                X_sub,
                y_enc,
                full_scaler=scaler,
                full_model=model,
            )
            preds = np.argmax(oof_proba, axis=1)
            round_bal_acc = float(balanced_accuracy_score(y_enc, preds))
            scores, mis_idx = self._feature_confusion_scores(X_sc, y_enc, oof_proba)

            round_states.append(
                {
                    "features": active.copy(),
                    "scaler": scaler,
                    "model": model,
                    "oof_balanced_accuracy": round_bal_acc,
                    "diagnostic_mode": str(diagnostic_mode),
                    "diagnostic_splits": int(diagnostic_splits),
                    "misclassified_count": int(mis_idx.size),
                    "confusion_scores": scores.copy(),
                }
            )
            round_scores.append(round_bal_acc)

            max_remove = int(max(0, active.size - min_features))
            positive = np.flatnonzero(scores > 1e-10)
            if mis_idx.size < int(self.min_misclassified):
                stop_reason = "too_few_misclassified"
                break
            if max_remove <= 0:
                stop_reason = "min_features_reached"
                break
            if positive.size == 0:
                stop_reason = "no_positive_confusion_scores"
                break

            remove_count = int(max(1, math.floor(self.elim_frac * active.size)))
            remove_count = int(min(remove_count, max_remove, positive.size))
            if remove_count <= 0:
                stop_reason = "no_removal_budget"
                break

            ranked = positive[np.argsort(scores[positive])]
            remove_local = np.sort(ranked[-remove_count:])
            active = np.delete(active, remove_local)
            stop_reason = "continue"

        n_rounds = int(len(round_states))
        self.rounds_ = list(round_states)
        self.n_rounds_ = int(n_rounds)
        self.round_feature_counts_ = [int(len(state["features"])) for state in round_states]
        self.round_oof_balanced_accuracy_ = [float(s) for s in round_scores]
        self.round_diagnostic_modes_ = [str(state["diagnostic_mode"]) for state in round_states]
        self.stop_reason_ = str(stop_reason)
        self.final_active_features_ = np.asarray(
            round_states[-1]["features"] if round_states else np.arange(n_features, dtype=int),
            dtype=int,
        )
        self.n_features_in_ = int(n_features)

        if n_rounds <= 0:
            raise RuntimeError("CPDAClassifier failed to fit any round.")

        geom = np.asarray(
            [
                (1.0 - self.blend_alpha) * (self.blend_alpha ** (n_rounds - idx - 1))
                for idx in range(n_rounds)
            ],
            dtype=float,
        )
        quality = np.asarray([max(1e-3, 0.25 + s) for s in round_scores], dtype=float)
        weights = geom * quality
        weights /= float(np.sum(weights))
        self.round_weights_ = weights
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if not hasattr(self, "rounds_") or not self.rounds_:
            raise AttributeError("CPDAClassifier is not fitted.")
        blended = np.zeros((X_arr.shape[0], int(self.classes_.size)), dtype=float)
        for weight, state in zip(np.asarray(self.round_weights_, dtype=float), self.rounds_):
            feats = np.asarray(state["features"], dtype=int)
            scaler = state["scaler"]
            model = state["model"]
            X_sc = scaler.transform(X_arr[:, feats])
            proba = self._normalize_proba(model.predict_proba(X_sc))
            blended += float(weight) * proba
        return self._normalize_proba(blended)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        pred_enc = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(np.asarray(pred_enc, dtype=int))


class ClassifierBackend(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit_and_select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        seed: int,
        n_classes: int,
        class_counts: np.ndarray,
        cv_splits: int = 5,
        scoring: str = "balanced_accuracy",
    ) -> Tuple[BaseEstimator, str, float, float, int, Dict[str, Any]]:
        """Return (model, name, score_mean, score_std, n_splits, meta)."""

    def supports_dataset(
        self,
        *,
        n_samples: int,
        n_features: int,
        n_classes: int,
        class_counts: Optional[np.ndarray] = None,
    ) -> bool:
        return bool(n_samples >= 2 and n_features >= 1 and n_classes >= 2)

    def get_candidates(self) -> Optional[Dict[str, BaseEstimator]]:
        return None


class SklearnBackend(ClassifierBackend):
    """Backwards-compatible sklearn classifier search backend."""

    def __init__(
        self,
        *,
        candidate_names: Sequence[str],
        lr_max_iter: int = 10000,
        use_hybrid_score: bool = False,
        hybrid_balanced_weight: float = 0.6,
        hybrid_macro_f1_weight: float = 0.4,
        allow_tree_models: bool = True,
        max_train_test_gap: float = 0.0,
        tree_complexity_penalty_enabled: bool = False,
        tree_complexity_penalty_strength: float = 0.1,
        n_jobs: int = 1,
        build_xgb_model_fn: Optional[Callable[[np.ndarray, int], OptionalModelBuildReturn]] = None,
        build_tabpfn_model_fn: Optional[Callable[[int], OptionalModelBuildReturn]] = None,
        warn_missing_backend_fn: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        self._candidate_names = tuple(str(c) for c in candidate_names if str(c))
        self.lr_max_iter = int(max(500, lr_max_iter))
        self.use_hybrid_score = bool(use_hybrid_score)
        self.hybrid_balanced_weight = float(max(0.0, hybrid_balanced_weight))
        self.hybrid_macro_f1_weight = float(max(0.0, hybrid_macro_f1_weight))
        self.allow_tree_models = bool(allow_tree_models)
        self.max_train_test_gap = float(max(0.0, max_train_test_gap))
        self.tree_complexity_penalty_enabled = bool(tree_complexity_penalty_enabled)
        self.tree_complexity_penalty_strength = float(max(0.0, tree_complexity_penalty_strength))
        self.n_jobs = int(max(1, n_jobs))
        self._build_xgb_model_fn = build_xgb_model_fn
        self._build_tabpfn_model_fn = build_tabpfn_model_fn
        self._warn_missing_backend_fn = warn_missing_backend_fn
        self._last_candidates: Dict[str, BaseEstimator] = {}
        self._last_candidate_build_failures: Dict[str, str] = {}

    def name(self) -> str:
        return "sklearn"

    def get_candidates(self) -> Optional[Dict[str, BaseEstimator]]:
        return dict(self._last_candidates)

    def _warn_missing(self, model_name: str, package_name: str, reason: Optional[str] = None) -> None:
        if self._warn_missing_backend_fn is not None:
            self._warn_missing_backend_fn(model_name, package_name, reason)

    def _tree_allowed(self, X_train: np.ndarray) -> bool:
        if not self.allow_tree_models:
            return False
        n = int(max(1, X_train.shape[0]))
        p = int(max(1, X_train.shape[1]))
        return bool(n >= 200 and (float(p) / float(n)) <= 50.0)

    def _build_candidates(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int,
    ) -> Dict[str, BaseEstimator]:
        requested = set(self._candidate_names)
        build_failures: Dict[str, str] = {}

        models: Dict[str, BaseEstimator] = {
            "lr": make_logistic_regression(
                random_state=seed,
                max_iter=self.lr_max_iter,
                solver="lbfgs",
                penalty="l2",
                class_weight="balanced",
            ),
            "svm_rbf": SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                random_state=seed,
            ),
        }

        if "elastic_net_lr" in requested:
            models["elastic_net_lr"] = make_logistic_regression(
                random_state=seed,
                max_iter=max(5000, self.lr_max_iter),
                solver="saga",
                penalty="elasticnet",
                l1_ratio=0.5,
                class_weight="balanced",
            )

        tree_allowed = self._tree_allowed(np.asarray(X_train, dtype=float))

        if "rf" in requested and tree_allowed:
            models["rf"] = RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=_get_sklearn_n_jobs(),
            )

        if "extra_tree" in requested and tree_allowed:
            models["extra_tree"] = ExtraTreesClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=_get_sklearn_n_jobs(),
            )

        if "lgbm" in requested and tree_allowed:
            if LGBMClassifier is None:
                self._warn_missing("lgbm", "lightgbm")
            else:
                models["lgbm"] = LGBMClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=seed,
                    n_jobs=_get_sklearn_n_jobs(),
                )

        if "catboost" in requested and tree_allowed:
            if CatBoostClassifier is None:
                self._warn_missing("catboost", "catboost")
            else:
                models["catboost"] = CatBoostClassifier(
                    depth=6,
                    learning_rate=0.05,
                    n_estimators=250,
                    loss_function="MultiClass" if int(np.unique(y_train).size) > 2 else "Logloss",
                    verbose=False,
                    random_seed=seed,
                    allow_writing_files=False,
                )

        if "knn" in requested:
            n_neighbors = int(max(3, min(11, X_train.shape[0] // 5)))
            if n_neighbors % 2 == 0:
                n_neighbors += 1
            max_valid = int(max(1, X_train.shape[0] - 1))
            n_neighbors = int(max(1, min(n_neighbors, max_valid)))
            models["knn"] = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"),
            )

        if "svm_linear" in requested:
            models["svm_linear"] = make_pipeline(
                StandardScaler(),
                SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=seed),
            )

        if "rp_ensemble" in requested:
            models["rp_ensemble"] = RandomProjectionEnsembleClassifier(
                n_estimators=9,
                n_components=None,
                max_components=64,
                random_state=seed,
                lr_max_iter=max(3000, self.lr_max_iter),
            )

        # T-CS-028: keep only one alias-equivalent LDA family per run.
        use_dlda = "dlda" in requested
        use_sh_lda = "shrinkage_lda" in requested
        if use_dlda or use_sh_lda:
            lda = make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            )
            if use_dlda:
                models["dlda"] = lda
            elif use_sh_lda:
                models["shrinkage_lda"] = lda

        if "nb" in requested:
            models["nb"] = make_pipeline(StandardScaler(), GaussianNB())

        if "vote_ensemble" in requested:
            vote_estimators = [
                (
                    "lr",
                    make_logistic_regression(
                        random_state=seed,
                        max_iter=self.lr_max_iter,
                        solver="lbfgs",
                        penalty="l2",
                        class_weight="balanced",
                    ),
                ),
                (
                    "svm_linear",
                    make_pipeline(
                        StandardScaler(),
                        SVC(
                            kernel="linear",
                            C=1.0,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ),
                ("nb", make_pipeline(StandardScaler(), GaussianNB())),
            ]
            models["vote_ensemble"] = VotingClassifier(
                estimators=vote_estimators,
                voting="hard",
                n_jobs=_get_sklearn_n_jobs(),
            )

        if "xgb" in requested and tree_allowed and self._build_xgb_model_fn is not None:
            xgb_model, failure_reason = _normalize_optional_model_build_result(
                self._build_xgb_model_fn(y_train, seed)
            )
            if xgb_model is None:
                build_failures["xgb"] = str(failure_reason or "unavailable")
                self._warn_missing("xgb", "xgboost", failure_reason)
            else:
                models["xgb"] = xgb_model

        if "tabpfn" in requested and self._build_tabpfn_model_fn is not None:
            tabpfn_model, failure_reason = _normalize_optional_model_build_result(
                self._build_tabpfn_model_fn(seed)
            )
            if tabpfn_model is None:
                build_failures["tabpfn"] = str(failure_reason or "unavailable")
                self._warn_missing("tabpfn", "tabpfn", failure_reason)
            else:
                models["tabpfn"] = tabpfn_model

        if "dbda" in requested:
            models["dbda"] = make_pipeline(
                StandardScaler(),
                DistanceBasedDiscriminantAnalysis(),
            )
        if "gqda" in requested:
            models["gqda"] = make_pipeline(
                StandardScaler(),
                GeometricalQuadraticDiscriminantAnalysis(),
            )
        if "bc_svm_linear" in requested:
            models["bc_svm_linear"] = make_pipeline(
                StandardScaler(),
                BiasCorrectedLinearSVM(C=1.0, random_state=seed),
            )
        if "sglnn" in requested:
            models["sglnn"] = make_pipeline(
                StandardScaler(),
                SparseGroupLassoNNClassifier(random_state=seed),
            )
        if "nsc" in requested:
            models["nsc"] = make_pipeline(
                StandardScaler(),
                NearestCentroid(shrink_threshold=0.2),
            )
        if "pls_da_classifier" in requested:
            models["pls_da_classifier"] = make_pipeline(
                StandardScaler(),
                PLSDAClassifier(n_components=4, scale=True),
            )
        if "rff_lr" in requested:
            models["rff_lr"] = RandomFourierFeaturesClassifier(
                random_state=seed,
                lr_max_iter=max(3000, self.lr_max_iter),
            )
        if "near_subspace" in requested:
            models["near_subspace"] = NearestSubspaceClassifier()
        if "spatial_median_da" in requested:
            models["spatial_median_da"] = SpatialMedianDiscriminantAnalysis()
        if "copula_da" in requested:
            models["copula_da"] = CopulaDiscriminantAnalysis()
        if "cpda" in requested:
            models["cpda"] = CPDAClassifier(random_state=seed)
        if "tabm" in requested:
            models["tabm"] = TabMClassifier(random_state=seed)
        if "realmlp" in requested:
            models["realmlp"] = RealMLPClassifier(random_state=seed)
        if "tabm_official" in requested:
            if _TabM_D_Classifier is not None:
                models["tabm_official"] = _TabM_D_Classifier(
                    random_state=seed, device="cpu", verbosity=0,
                )
            else:
                build_failures["tabm_official"] = "pytabkit not installed"
                self._warn_missing("tabm_official", "pytabkit", "pytabkit not installed")
        if "realmlp_td" in requested:
            if _RealMLP_TD_Classifier is not None:
                models["realmlp_td"] = _RealMLP_TD_Classifier(
                    random_state=seed, device="cpu", verbosity=0,
                    n_cv=1, n_refit=0,
                )
            else:
                build_failures["realmlp_td"] = "pytabkit not installed"
                self._warn_missing("realmlp_td", "pytabkit", "pytabkit not installed")
        if "gpc" in requested and int(X_train.shape[0]) <= 200:
            models["gpc"] = make_pipeline(
                StandardScaler(),
                GaussianProcessClassifier(random_state=seed),
            )

        self._last_candidate_build_failures = dict(build_failures)
        return models

    def fit_and_select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        seed: int,
        n_classes: int,
        class_counts: np.ndarray,
        cv_splits: int = 5,
        scoring: str = "balanced_accuracy",
    ) -> Tuple[BaseEstimator, str, float, float, int, Dict[str, Any]]:
        _ = (n_classes, scoring)
        models = self._build_candidates(X_train=np.asarray(X_train), y_train=np.asarray(y_train), seed=int(seed))
        self._last_candidates = dict(models)
        build_failures = dict(self._last_candidate_build_failures)

        candidate_names_raw = [name for name in self._candidate_names if name in models]
        candidate_names, alias_dropped = _unique_with_alias_handling(candidate_names_raw)
        constructed_candidates = tuple(candidate_names)
        if not candidate_names:
            candidate_names = ["lr"]

        counts = np.asarray(class_counts, dtype=int).ravel()
        if counts.size == 0 or np.min(counts) < 2:
            return models["lr"], "lr", float("nan"), float("nan"), 0, {
                "model_cv_runtime_containment_reason": "insufficient_class_counts",
                "model_cv_constructed_candidates": tuple(constructed_candidates),
                "model_cv_candidate_build_failures": dict(build_failures),
                "model_cv_failed_candidates": tuple(),
                "model_cv_candidate_failure_reasons": {},
                "model_cv_evaluated_candidates": tuple(),
                "model_cv_candidate_scores": {},
                "model_cv_alias_dropped": tuple(alias_dropped),
                "classification_backend_used": self.name(),
            }

        n_splits = int(max(2, min(int(cv_splits), int(np.min(counts)))))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))

        best_name = candidate_names[0]
        best_score = -np.inf
        best_std = float("nan")
        best_n = 0
        score_map: Dict[str, float] = {}
        gap_map: Dict[str, float] = {}
        gap_rejected: List[str] = []
        failed_reasons: Dict[str, str] = {}
        evaluated = []

        # Whether train/test gap gating is active.
        _gap_active = bool(self.max_train_test_gap > 0)
        _penalty_active = bool(self.tree_complexity_penalty_enabled and self.tree_complexity_penalty_strength > 0)
        _need_train_score = bool(_gap_active or _penalty_active)

        def _eval_one_candidate(name: str) -> Dict[str, Any]:
            """Evaluate a single candidate model and return structured telemetry."""
            model = models[name]
            try:
                cv_result = cross_validate(
                    model, X_train, y_train, cv=cv,
                    scoring="balanced_accuracy", n_jobs=1,
                    return_train_score=_need_train_score,
                )
                bal_scores = np.asarray(cv_result["test_score"], dtype=float).ravel()
                bal_scores = bal_scores[np.isfinite(bal_scores)]
                if bal_scores.size == 0:
                    return {
                        "name": str(name),
                        "status": "failed",
                        "reason": "no_finite_test_score",
                    }

                # Compute CV train/validation gap (uses only training data).
                gap = 0.0
                if _need_train_score:
                    train_scores = np.asarray(cv_result["train_score"], dtype=float).ravel()
                    finite_train = train_scores[np.isfinite(train_scores)]
                    if finite_train.size > 0:
                        gap = float(np.mean(finite_train) - np.mean(bal_scores))

                # Hard gate: reject candidates whose gap exceeds threshold.
                if _gap_active and gap > self.max_train_test_gap:
                    return {
                        "name": str(name),
                        "status": "gap_rejected",
                        "gap": float(gap),
                    }

                score_arr = bal_scores
                score = float(np.mean(score_arr))

                if self.use_hybrid_score:
                    try:
                        f1m = np.asarray(
                            cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1),
                            dtype=float,
                        ).ravel()
                        f1m = f1m[np.isfinite(f1m)]
                        if f1m.size == bal_scores.size and f1m.size > 0:
                            w_bal = float(max(0.0, self.hybrid_balanced_weight))
                            w_f1 = float(max(0.0, self.hybrid_macro_f1_weight))
                            denom = float(max(1e-12, w_bal + w_f1))
                            score_arr = (w_bal * bal_scores + w_f1 * f1m) / denom
                            score = float(np.mean(score_arr))
                    except Exception as exc:
                        pass

                # Soft penalty: reduce score for tree models proportional to gap.
                if _penalty_active and str(name) in _TREE_MODEL_NAMES and gap > 0:
                    score = score - self.tree_complexity_penalty_strength * gap

                std = float(np.std(score_arr, ddof=1)) if score_arr.size > 1 else 0.0
                return {
                    "name": str(name),
                    "status": "ok",
                    "score": float(score),
                    "std": float(std),
                    "n": int(score_arr.size),
                    "gap": float(gap),
                }
            except Exception as exc:
                return {
                    "name": str(name),
                    "status": "failed",
                    "reason": _format_exception_summary(exc),
                }

        # CL-1: Parallel candidate evaluation via ThreadPoolExecutor.
        if self.n_jobs > 1 and len(candidate_names) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
                futures = {pool.submit(_eval_one_candidate, name): name for name in candidate_names}
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    name = str(result.get("name", futures[fut]))
                    if "gap" in result:
                        gap_map[name] = float(result["gap"])
                    status = str(result.get("status", "failed"))
                    if status == "gap_rejected":
                        gap_rejected.append(name)
                        continue
                    if status != "ok":
                        failed_reasons[name] = str(result.get("reason", "evaluation_failed"))
                        continue
                    score = float(result["score"])
                    std = float(result["std"])
                    n = int(result["n"])
                    score_map[name] = score
                    evaluated.append(name)
                    if score > best_score:
                        best_name = name
                        best_score = score
                        best_std = std
                        best_n = n
        else:
            for name in candidate_names:
                result = _eval_one_candidate(name)
                rname = str(result.get("name", name))
                if "gap" in result:
                    gap_map[rname] = float(result["gap"])
                status = str(result.get("status", "failed"))
                if status == "gap_rejected":
                    gap_rejected.append(rname)
                    continue
                if status != "ok":
                    failed_reasons[rname] = str(result.get("reason", "evaluation_failed"))
                    continue
                score = float(result["score"])
                std = float(result["std"])
                n = int(result["n"])
                score_map[rname] = score
                evaluated.append(rname)
                if score > best_score:
                    best_name = rname
                    best_score = score
                    best_std = std
                    best_n = n

        meta = {
            "model_cv_constructed_candidates": tuple(constructed_candidates),
            "model_cv_candidate_build_failures": dict(build_failures),
            "model_cv_failed_candidates": tuple(sorted(failed_reasons)),
            "model_cv_candidate_failure_reasons": dict(failed_reasons),
            "model_cv_evaluated_candidates": tuple(evaluated),
            "model_cv_candidate_scores": dict(score_map),
            "model_cv_alias_dropped": tuple(alias_dropped),
            "model_cv_train_test_gaps": dict(gap_map),
            "model_cv_gap_rejected": tuple(gap_rejected),
            "classification_backend_used": self.name(),
        }
        if not np.isfinite(best_score):
            return models["lr"], "lr", float("nan"), float("nan"), 0, meta
        return models[best_name], best_name, float(best_score), float(best_std), int(best_n), meta


class FLAMLBackend(ClassifierBackend):
    """FLAML-powered AutoML backend for the final classifier stage."""

    def __init__(
        self,
        *,
        time_budget: int = 60,
        estimator_list: Tuple[str, ...] = ("lgbm", "xgboost", "rf", "extra_tree", "lrl2"),
        metric: str = "macro_f1",
        n_jobs: int = 1,
        min_n_for_automl: int = 50,
        min_n_per_class_for_automl: int = 10,
        min_n_per_class_for_cv: int = 5,
        max_p_over_n_for_automl: int = 200,
    ):
        self.time_budget = int(max(1, time_budget))
        self.estimator_list = tuple(str(e) for e in estimator_list if str(e))
        self.metric = self._normalize_metric_key(metric, default="macro_f1")
        self.n_jobs = int(max(1, n_jobs))
        self.min_n_for_automl = int(max(2, min_n_for_automl))
        self.min_n_per_class_for_automl = int(max(2, min_n_per_class_for_automl))
        self.min_n_per_class_for_cv = int(max(2, min_n_per_class_for_cv))
        self.max_p_over_n_for_automl = int(max(1, max_p_over_n_for_automl))

    def name(self) -> str:
        return "flaml"

    def supports_dataset(
        self,
        *,
        n_samples: int,
        n_features: int,
        n_classes: int,
        class_counts: Optional[np.ndarray] = None,
    ) -> bool:
        if not super().supports_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_counts=class_counts,
        ):
            return False
        if int(n_samples) < int(self.min_n_for_automl):
            return False
        if float(n_features) / float(max(1, n_samples)) > float(self.max_p_over_n_for_automl):
            return False
        if class_counts is not None:
            counts = np.asarray(class_counts, dtype=int).ravel()
            if counts.size == 0 or int(np.min(counts)) < int(self.min_n_per_class_for_automl):
                return False
        return True

    @staticmethod
    def _normalize_metric_key(metric: str, *, default: str = "macro_f1") -> str:
        s = str(metric or "").strip().lower()
        if s in {"balanced_accuracy", "balanced-accuracy", "bal_acc", "accuracy"}:
            return "accuracy"
        if s in {"f1_macro", "macro_f1"}:
            return "macro_f1"
        if s in {"f1_micro", "micro_f1"}:
            return "micro_f1"
        if s in {"f1"}:
            return "f1"
        if s in {"roc_auc", "roc_auc_ovr", "roc_auc_ovo", "log_loss", "ap"}:
            return s

        d = str(default or "macro_f1").strip().lower()
        if d in {"balanced_accuracy", "balanced-accuracy", "bal_acc", "accuracy"}:
            return "accuracy"
        if d in {"f1_macro", "macro_f1"}:
            return "macro_f1"
        if d in {"f1_micro", "micro_f1"}:
            return "micro_f1"
        if d in {"f1"}:
            return "f1"
        if d in {"roc_auc", "roc_auc_ovr", "roc_auc_ovo", "log_loss", "ap"}:
            return d
        return "macro_f1"

    @staticmethod
    def _map_scoring(scoring: str, fallback_metric: str) -> str:
        """Map sklearn-style metric to FLAML metric key.

        FLAML on this stack does not accept ``balanced_accuracy`` as metric key.
        """
        s = str(scoring or "").strip().lower()
        if s in {"balanced_accuracy", "balanced-accuracy", "bal_acc"}:
            return "accuracy"
        if s in {"accuracy"}:
            return "accuracy"
        if s in {"f1_macro", "macro_f1"}:
            return "macro_f1"
        if s in {"f1_micro", "micro_f1"}:
            return "micro_f1"
        if s in {"f1"}:
            return "f1"
        return FLAMLBackend._normalize_metric_key(fallback_metric, default="macro_f1")

    def fit_and_select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        seed: int,
        n_classes: int,
        class_counts: np.ndarray,
        cv_splits: int = 5,
        scoring: str = "balanced_accuracy",
    ) -> Tuple[BaseEstimator, str, float, float, int, Dict[str, Any]]:
        counts = np.asarray(class_counts, dtype=int).ravel()
        if counts.size == 0 or int(np.min(counts)) < int(self.min_n_per_class_for_cv):
            lr = make_logistic_regression(
                random_state=int(seed),
                max_iter=10000,
                solver="lbfgs",
                penalty="l2",
                class_weight="balanced",
            )
            return lr, "lr", float("nan"), float("nan"), 0, {
                "classification_backend_used": "sklearn_fallback",
                "classification_guard_reason": "min_n_per_class_for_cv",
            }

        try:
            from flaml import AutoML
        except Exception as exc:  # pragma: no cover
            raise ImportError("FLAML is not installed.") from exc

        x = np.asarray(X_train, dtype=float)
        y_raw = np.asarray(y_train).ravel()
        # T-CS-002: use explicit encoding to avoid non-zero-based label edge cases.
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(y_raw)

        if not self.supports_dataset(
            n_samples=int(x.shape[0]),
            n_features=int(x.shape[1]) if x.ndim == 2 else 0,
            n_classes=int(n_classes),
            class_counts=counts,
        ):
            raise RuntimeError("FLAMLBackend does not support this dataset regime.")

        n_splits = int(max(2, min(int(cv_splits), int(np.min(counts)))))
        metric = self._map_scoring(scoring, self.metric)

        automl = AutoML()
        automl.fit(
            x,
            y,
            task="classification",
            time_budget=int(self.time_budget),
            estimator_list=list(self.estimator_list),
            metric=metric,
            n_jobs=int(self.n_jobs),
            seed=int(seed),
            eval_method="cv",
            n_splits=int(n_splits),
            verbose=0,
        )
        best_loss = float(getattr(automl, "best_loss", float("nan")))
        best_estimator = str(getattr(automl, "best_estimator", "unknown"))
        best_val = float(1.0 - best_loss) if np.isfinite(best_loss) else float("nan")
        meta = {
            "classification_backend_used": self.name(),
            "flaml_best_estimator": best_estimator,
            "flaml_best_loss": best_loss,
            "flaml_best_config": getattr(automl, "best_config", {}),
            "flaml_time_budget": int(self.time_budget),
            "flaml_metric": str(metric),
            "label_encoder_classes": [str(c) for c in np.asarray(label_enc.classes_).ravel()],
        }
        model = getattr(automl, "model", None)
        if model is None:
            raise RuntimeError("FLAML fit completed without a model.")
        wrapped = _LabelEncodedEstimator(model)
        # Prime wrapper with observed classes for metadata/predict compatibility.
        wrapped._label_encoder = label_enc
        wrapped.classes_ = np.asarray(label_enc.classes_)
        return wrapped, f"flaml_{best_estimator}", best_val, float("nan"), int(n_splits), meta


class OptunaBackend(ClassifierBackend):
    """Optuna-powered HPO backend for Stage-2 classifier selection (T-R-211)."""

    def __init__(
        self,
        *,
        candidate_names: Sequence[str],
        time_budget: int = 120,
        n_trials: int = 25,
        min_n_for_automl: int = 50,
        min_n_per_class_for_automl: int = 10,
        min_n_per_class_for_cv: int = 5,
        max_p_over_n_for_automl: int = 200,
        lr_max_iter: int = 10000,
        use_hybrid_score: bool = False,
        hybrid_balanced_weight: float = 0.6,
        hybrid_macro_f1_weight: float = 0.4,
        allow_tree_models: bool = True,
        n_jobs: int = 1,
        build_xgb_model_fn: Optional[Callable[[np.ndarray, int], OptionalModelBuildReturn]] = None,
        build_tabpfn_model_fn: Optional[Callable[[int], OptionalModelBuildReturn]] = None,
        warn_missing_backend_fn: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        self._candidate_names = tuple(str(c) for c in candidate_names if str(c))
        self.time_budget = int(max(1, time_budget))
        self.n_trials = int(max(1, n_trials))
        self.min_n_for_automl = int(max(2, min_n_for_automl))
        self.min_n_per_class_for_automl = int(max(2, min_n_per_class_for_automl))
        self.min_n_per_class_for_cv = int(max(2, min_n_per_class_for_cv))
        self.max_p_over_n_for_automl = int(max(1, max_p_over_n_for_automl))
        self.lr_max_iter = int(max(500, lr_max_iter))
        self.use_hybrid_score = bool(use_hybrid_score)
        self.hybrid_balanced_weight = float(max(0.0, hybrid_balanced_weight))
        self.hybrid_macro_f1_weight = float(max(0.0, hybrid_macro_f1_weight))
        self.allow_tree_models = bool(allow_tree_models)
        self.n_jobs = int(max(1, n_jobs))
        self._build_xgb_model_fn = build_xgb_model_fn
        self._build_tabpfn_model_fn = build_tabpfn_model_fn
        self._warn_missing_backend_fn = warn_missing_backend_fn
        self._last_candidates: Dict[str, BaseEstimator] = {}

    def name(self) -> str:
        return "optuna"

    def get_candidates(self) -> Optional[Dict[str, BaseEstimator]]:
        return dict(self._last_candidates)

    def supports_dataset(
        self,
        *,
        n_samples: int,
        n_features: int,
        n_classes: int,
        class_counts: Optional[np.ndarray] = None,
    ) -> bool:
        if not super().supports_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_counts=class_counts,
        ):
            return False
        if int(n_samples) < int(self.min_n_for_automl):
            return False
        if float(n_features) / float(max(1, n_samples)) > float(self.max_p_over_n_for_automl):
            return False
        if class_counts is not None:
            counts = np.asarray(class_counts, dtype=int).ravel()
            if counts.size == 0 or int(np.min(counts)) < int(self.min_n_per_class_for_automl):
                return False
        return True

    def _build_candidates(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int,
    ) -> Dict[str, BaseEstimator]:
        sk_backend = SklearnBackend(
            candidate_names=tuple(self._candidate_names),
            lr_max_iter=int(self.lr_max_iter),
            use_hybrid_score=bool(self.use_hybrid_score),
            hybrid_balanced_weight=float(self.hybrid_balanced_weight),
            hybrid_macro_f1_weight=float(self.hybrid_macro_f1_weight),
            allow_tree_models=bool(self.allow_tree_models),
            build_xgb_model_fn=self._build_xgb_model_fn,
            build_tabpfn_model_fn=self._build_tabpfn_model_fn,
            warn_missing_backend_fn=self._warn_missing_backend_fn,
        )
        models = sk_backend._build_candidates(
            X_train=np.asarray(X_train, dtype=float),
            y_train=np.asarray(y_train).ravel(),
            seed=int(seed),
        )
        self._last_candidates = dict(models)
        return models

    def _evaluate_candidate_scores(
        self,
        model: BaseEstimator,
        *,
        X: np.ndarray,
        y: np.ndarray,
        cv: StratifiedKFold,
    ) -> np.ndarray:
        bal_scores = np.asarray(
            cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy"),
            dtype=float,
        ).ravel()
        bal_scores = bal_scores[np.isfinite(bal_scores)]
        if bal_scores.size == 0:
            return np.asarray([], dtype=float)
        if not self.use_hybrid_score:
            return bal_scores

        try:
            f1m = np.asarray(
                cross_val_score(model, X, y, cv=cv, scoring="f1_macro"),
                dtype=float,
            ).ravel()
            f1m = f1m[np.isfinite(f1m)]
            if f1m.size == bal_scores.size and f1m.size > 0:
                w_bal = float(max(0.0, self.hybrid_balanced_weight))
                w_f1 = float(max(0.0, self.hybrid_macro_f1_weight))
                denom = float(max(1e-12, w_bal + w_f1))
                return (w_bal * bal_scores + w_f1 * f1m) / denom
        except Exception as exc:
            pass
        return bal_scores

    def _build_tuned_model(
        self,
        *,
        family: str,
        params: Dict[str, Any],
        fallback: BaseEstimator,
        seed: int,
        n_samples: int,
        n_features: int,
        n_classes: int,
        y_train: np.ndarray,
    ) -> BaseEstimator:
        fam = str(family)
        p = dict(params or {})

        if fam == "lr":
            return make_logistic_regression(
                random_state=int(seed),
                max_iter=int(self.lr_max_iter),
                solver="lbfgs",
                penalty="l2",
                C=float(p.get("C", 1.0)),
                class_weight="balanced",
            )
        if fam == "elastic_net_lr":
            return make_logistic_regression(
                random_state=int(seed),
                max_iter=max(5000, int(self.lr_max_iter)),
                solver="saga",
                penalty="elasticnet",
                C=float(p.get("C", 1.0)),
                l1_ratio=float(np.clip(float(p.get("l1_ratio", 0.5)), 0.0, 1.0)),
                class_weight="balanced",
            )
        if fam == "svm_rbf":
            gamma_val: Any = p.get("gamma", "scale")
            if isinstance(gamma_val, (int, float)):
                gamma_val = float(max(1e-8, gamma_val))
            else:
                gamma_val = str(gamma_val)
            return SVC(
                kernel="rbf",
                C=float(max(1e-6, p.get("C", 10.0))),
                gamma=gamma_val,
                class_weight="balanced",
                random_state=int(seed),
            )
        if fam == "svm_linear":
            return make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="linear",
                    C=float(max(1e-6, p.get("C", 1.0))),
                    class_weight="balanced",
                    random_state=int(seed),
                ),
            )
        if fam in {"dlda", "shrinkage_lda"}:
            return make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage=float(np.clip(float(p.get("shrinkage", 0.5)), 0.0, 1.0)),
                ),
            )
        if fam == "nsc":
            return make_pipeline(
                StandardScaler(),
                NearestCentroid(
                    shrink_threshold=float(max(0.0, p.get("shrink_threshold", 0.2))),
                ),
            )
        if fam == "pls_da_classifier":
            max_components = int(max(1, min(n_features, n_samples - 1, max(2, n_classes))))
            n_comp = int(max(1, min(int(p.get("n_components", 2)), max_components)))
            return make_pipeline(
                StandardScaler(),
                PLSDAClassifier(n_components=n_comp, scale=True),
            )
        if fam == "nb":
            return make_pipeline(
                StandardScaler(),
                GaussianNB(var_smoothing=float(max(1e-12, p.get("var_smoothing", 1e-9)))),
            )
        if fam == "knn":
            max_nn = int(max(1, min(31, n_samples - 1)))
            n_neighbors = int(max(1, min(int(p.get("n_neighbors", 7)), max_nn)))
            return make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=str(p.get("weights", "distance")),
                ),
            )
        if fam == "rf":
            return RandomForestClassifier(
                n_estimators=int(max(50, p.get("n_estimators", 200))),
                max_depth=None if p.get("max_depth", None) in {None, 0} else int(p.get("max_depth")),
                min_samples_leaf=int(max(1, p.get("min_samples_leaf", 2))),
                class_weight="balanced_subsample",
                random_state=int(seed),
                n_jobs=_get_sklearn_n_jobs(),
            )
        if fam == "extra_tree":
            return ExtraTreesClassifier(
                n_estimators=int(max(50, p.get("n_estimators", 250))),
                max_depth=None if p.get("max_depth", None) in {None, 0} else int(p.get("max_depth")),
                min_samples_leaf=int(max(1, p.get("min_samples_leaf", 2))),
                class_weight="balanced",
                random_state=int(seed),
                n_jobs=_get_sklearn_n_jobs(),
            )
        if fam == "lgbm" and LGBMClassifier is not None:
            return LGBMClassifier(
                n_estimators=int(max(50, p.get("n_estimators", 250))),
                learning_rate=float(max(1e-4, p.get("learning_rate", 0.05))),
                num_leaves=int(max(4, p.get("num_leaves", 31))),
                random_state=int(seed),
                n_jobs=_get_sklearn_n_jobs(),
            )
        if fam == "catboost" and CatBoostClassifier is not None:
            return CatBoostClassifier(
                depth=int(max(2, p.get("depth", 6))),
                learning_rate=float(max(1e-4, p.get("learning_rate", 0.05))),
                n_estimators=int(max(50, p.get("n_estimators", 250))),
                loss_function="MultiClass" if int(np.unique(y_train).size) > 2 else "Logloss",
                verbose=False,
                random_seed=int(seed),
                allow_writing_files=False,
            )

        return clone(fallback)

    @staticmethod
    def _extract_family_params(study_params: Dict[str, Any], family: str) -> Dict[str, Any]:
        prefix = f"{str(family)}__"
        out: Dict[str, Any] = {}
        for key, val in dict(study_params or {}).items():
            k = str(key)
            if k.startswith(prefix):
                out[k[len(prefix):]] = val
        return out

    def fit_and_select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        seed: int,
        n_classes: int,
        class_counts: np.ndarray,
        cv_splits: int = 5,
        scoring: str = "balanced_accuracy",
    ) -> Tuple[BaseEstimator, str, float, float, int, Dict[str, Any]]:
        _ = scoring
        counts = np.asarray(class_counts, dtype=int).ravel()
        if counts.size == 0 or int(np.min(counts)) < int(self.min_n_per_class_for_cv):
            lr = make_logistic_regression(
                random_state=int(seed),
                max_iter=10000,
                solver="lbfgs",
                penalty="l2",
                class_weight="balanced",
            )
            return lr, "lr", float("nan"), float("nan"), 0, {
                "classification_backend_used": "sklearn_fallback",
                "classification_guard_reason": "min_n_per_class_for_cv",
            }

        x = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train).ravel()

        if not self.supports_dataset(
            n_samples=int(x.shape[0]),
            n_features=int(x.shape[1]) if x.ndim == 2 else 0,
            n_classes=int(n_classes),
            class_counts=counts,
        ):
            raise RuntimeError("OptunaBackend does not support this dataset regime.")

        try:
            import optuna  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("Optuna is not installed.") from exc

        models = self._build_candidates(X_train=x, y_train=y, seed=int(seed))
        candidate_names_raw = [name for name in self._candidate_names if name in models]
        candidate_names, alias_dropped = _unique_with_alias_handling(candidate_names_raw)
        if not candidate_names:
            candidate_names = ["lr"]
            models["lr"] = make_logistic_regression(
                random_state=int(seed),
                max_iter=10000,
                solver="lbfgs",
                penalty="l2",
                class_weight="balanced",
            )

        n_splits = int(max(2, min(int(cv_splits), int(np.min(counts)))))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))

        score_map: Dict[str, float] = {}
        std_map: Dict[str, float] = {}

        def _eval_baseline(name: str) -> Optional[Tuple[str, float, float]]:
            try:
                arr = self._evaluate_candidate_scores(models[name], X=x, y=y, cv=cv)
                if arr.size == 0:
                    return None
                return (str(name), float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0)
            except Exception as exc:
                return None

        # CL-1: Parallel baseline candidate evaluation.
        if self.n_jobs > 1 and len(candidate_names) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
                futures = {pool.submit(_eval_baseline, name): name for name in candidate_names}
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    if result is not None:
                        score_map[result[0]] = result[1]
                        std_map[result[0]] = result[2]
        else:
            for name in candidate_names:
                result = _eval_baseline(name)
                if result is not None:
                    score_map[result[0]] = result[1]
                    std_map[result[0]] = result[2]

        if score_map:
            best_name = max(score_map, key=score_map.get)
            best_score = float(score_map[best_name])
            best_std = float(std_map.get(best_name, float("nan")))
        else:
            best_name = "lr"
            best_score = float("nan")
            best_std = float("nan")

        # Baseline fallback: default models only.
        best_model = clone(
            models.get(
                best_name,
                models.get("lr", make_logistic_regression(class_weight="balanced")),
            )
        )
        optuna_meta: Dict[str, Any] = {
            "classification_backend_used": self.name(),
            "optuna_time_budget": int(self.time_budget),
            "optuna_n_trials_requested": int(self.n_trials),
            "optuna_n_trials_completed": 0,
            "optuna_best_family": str(best_name),
            "optuna_best_params": {},
            "optuna_best_value": float(best_score) if np.isfinite(best_score) else float("nan"),
            "optuna_used_tuned_params": False,
            "model_cv_evaluated_candidates": tuple(str(k) for k in score_map.keys()),
            "model_cv_candidate_scores": dict(score_map),
            "model_cv_alias_dropped": tuple(alias_dropped),
        }

        def _objective(trial: Any) -> float:
            fam = str(trial.suggest_categorical("family", candidate_names))
            pref = f"{fam}__"
            params: Dict[str, Any] = {}

            if fam in {"lr", "elastic_net_lr", "svm_rbf", "svm_linear"}:
                params["C"] = float(trial.suggest_float(f"{pref}C", 1e-3, 1e2, log=True))
            if fam == "svm_rbf":
                params["gamma"] = float(trial.suggest_float(f"{pref}gamma", 1e-5, 1e1, log=True))
            if fam == "elastic_net_lr":
                params["l1_ratio"] = float(trial.suggest_float(f"{pref}l1_ratio", 0.0, 1.0))
            if fam in {"dlda", "shrinkage_lda"}:
                params["shrinkage"] = float(trial.suggest_float(f"{pref}shrinkage", 0.0, 1.0))
            if fam == "nsc":
                params["shrink_threshold"] = float(
                    trial.suggest_float(f"{pref}shrink_threshold", 0.0, 1.0)
                )
            if fam == "pls_da_classifier":
                max_components = int(max(1, min(x.shape[1], x.shape[0] - 1, max(2, int(n_classes)))))
                params["n_components"] = int(
                    trial.suggest_int(f"{pref}n_components", 1, max_components)
                )
            if fam == "nb":
                params["var_smoothing"] = float(
                    trial.suggest_float(f"{pref}var_smoothing", 1e-12, 1e-2, log=True)
                )
            if fam == "knn":
                max_nn = int(max(1, min(31, x.shape[0] - 1)))
                params["n_neighbors"] = int(trial.suggest_int(f"{pref}n_neighbors", 1, max_nn))
                params["weights"] = str(
                    trial.suggest_categorical(f"{pref}weights", ["uniform", "distance"])
                )
            if fam in {"rf", "extra_tree"}:
                params["n_estimators"] = int(trial.suggest_int(f"{pref}n_estimators", 100, 400))
                params["max_depth"] = int(trial.suggest_int(f"{pref}max_depth", 2, 16))
                params["min_samples_leaf"] = int(trial.suggest_int(f"{pref}min_samples_leaf", 1, 8))
            if fam == "lgbm" and LGBMClassifier is not None:
                params["n_estimators"] = int(trial.suggest_int(f"{pref}n_estimators", 100, 400))
                params["learning_rate"] = float(
                    trial.suggest_float(f"{pref}learning_rate", 1e-3, 3e-1, log=True)
                )
                params["num_leaves"] = int(trial.suggest_int(f"{pref}num_leaves", 8, 128))
            if fam == "catboost" and CatBoostClassifier is not None:
                params["depth"] = int(trial.suggest_int(f"{pref}depth", 3, 10))
                params["learning_rate"] = float(
                    trial.suggest_float(f"{pref}learning_rate", 1e-3, 3e-1, log=True)
                )
                params["n_estimators"] = int(trial.suggest_int(f"{pref}n_estimators", 100, 400))

            model = self._build_tuned_model(
                family=fam,
                params=params,
                fallback=models[fam],
                seed=int(seed),
                n_samples=int(x.shape[0]),
                n_features=int(x.shape[1]),
                n_classes=int(n_classes),
                y_train=y,
            )
            arr = self._evaluate_candidate_scores(model, X=x, y=y, cv=cv)
            if arr.size == 0:
                return 0.0
            return float(np.mean(arr))

        try:
            sampler = optuna.samplers.TPESampler(seed=int(seed))
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(
                _objective,
                n_trials=int(self.n_trials),
                timeout=int(self.time_budget),
                n_jobs=1,
            )
            optuna_meta["optuna_n_trials_completed"] = int(len(getattr(study, "trials", []) or []))
            best_trial = getattr(study, "best_trial", None)
            if best_trial is not None:
                fam = str(getattr(best_trial, "params", {}).get("family", best_name))
                if fam in models:
                    fam_params = self._extract_family_params(getattr(best_trial, "params", {}), fam)
                    tuned = self._build_tuned_model(
                        family=fam,
                        params=fam_params,
                        fallback=models[fam],
                        seed=int(seed),
                        n_samples=int(x.shape[0]),
                        n_features=int(x.shape[1]),
                        n_classes=int(n_classes),
                        y_train=y,
                    )
                    arr = self._evaluate_candidate_scores(tuned, X=x, y=y, cv=cv)
                    if arr.size > 0:
                        best_model = tuned
                        best_name = str(fam)
                        best_score = float(np.mean(arr))
                        best_std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                        optuna_meta["optuna_best_family"] = str(best_name)
                        optuna_meta["optuna_best_params"] = dict(fam_params)
                        optuna_meta["optuna_best_value"] = float(best_score)
                        optuna_meta["optuna_used_tuned_params"] = bool(len(fam_params) > 0)
        except Exception as exc:
            optuna_meta["optuna_failure_reason"] = str(type(exc).__name__)

        best_model.fit(x, y)
        return (
            best_model,
            f"optuna_{best_name}",
            float(best_score) if np.isfinite(best_score) else float("nan"),
            float(best_std) if np.isfinite(best_std) else float("nan"),
            int(n_splits),
            optuna_meta,
        )


class ClassifierOracle:
    """Compute MNPO classifier oracles and Nash-selection weights."""

    def __init__(
        self,
        *,
        pairwise_delta: float = 0.01,
        weighting_mode: str = "tritrust",
        include_calibration: bool = False,
        include_james_stein: bool = False,
        enable_hoeffding_racing: bool = False,
        hoeffding_delta: float = 0.10,
        enable_bbc: bool = False,
        bbc_bootstrap_rounds: int = 200,
        bbc_ci_level: float = 0.90,
        n_classes: int = 2,
        include_diversity: bool = False,
        flatten_multiclass_complexity: bool = True,
    ):
        self.pairwise_delta = float(max(0.0, pairwise_delta))
        self.weighting_mode = str(weighting_mode or "tritrust").strip().lower()
        if self.weighting_mode not in {"tritrust", "uniform", "shapley", "banzhaf"}:
            self.weighting_mode = "tritrust"
        self.include_calibration = bool(include_calibration)
        self.include_james_stein = bool(include_james_stein)
        self.enable_hoeffding_racing = bool(enable_hoeffding_racing)
        self.hoeffding_delta = float(np.clip(hoeffding_delta, 1e-6, 0.99))
        self.enable_bbc = bool(enable_bbc)
        self.bbc_bootstrap_rounds = int(max(0, bbc_bootstrap_rounds))
        self.bbc_ci_level = float(np.clip(bbc_ci_level, 0.50, 0.999))
        self.n_classes = int(max(2, n_classes))
        self.include_diversity = bool(include_diversity)
        self.flatten_multiclass_complexity = bool(flatten_multiclass_complexity)

    @staticmethod
    def _n_splits(y: np.ndarray, cv_splits: int) -> int:
        _, counts = np.unique(np.asarray(y).ravel(), return_counts=True)
        if counts.size == 0:
            return 0
        return int(max(2, min(int(cv_splits), int(np.min(counts)))))

    @staticmethod
    def _multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
        y_arr = np.asarray(y_true).ravel()
        p = np.asarray(proba, dtype=float)
        if p.ndim != 2 or y_arr.size != p.shape[0] or p.shape[0] == 0:
            return float("nan")
        classes = np.unique(y_arr)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        # Align classes with columns if dimensions mismatch.
        if p.shape[1] != int(classes.size):
            return float("nan")
        one_hot = np.zeros_like(p, dtype=float)
        for row, c in enumerate(y_arr):
            idx = class_to_idx.get(c)
            if idx is not None and 0 <= idx < one_hot.shape[1]:
                one_hot[row, idx] = 1.0
        return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))

    @staticmethod
    def _expected_calibration_error(
        y_true: np.ndarray,
        proba: np.ndarray,
        *,
        n_bins: int = 15,
    ) -> float:
        """Expected Calibration Error (ECE) via equal-width binning.

        For multiclass: uses per-sample max-predicted-class confidence vs
        correctness (top-label ECE).  Returns a value in [0, 1] where lower
        is better.  NaN on degenerate inputs.
        """
        y = np.asarray(y_true).ravel()
        p = np.asarray(proba, dtype=float)
        if p.ndim != 2 or y.size != p.shape[0] or p.shape[0] == 0:
            return float("nan")
        # Top-label ECE: compare max predicted probability to correctness
        confidences = np.max(p, axis=1)
        pred_labels = np.argmax(p, axis=1)
        # Map true labels to column indices
        classes = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        if p.shape[1] != int(classes.size):
            return float("nan")
        true_idx = np.array([class_to_idx.get(c, -1) for c in y], dtype=int)
        if np.any(true_idx < 0):
            return float("nan")
        correct = (pred_labels == true_idx).astype(float)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = float(y.size)
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b == n_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            n_bin = int(mask.sum())
            if n_bin == 0:
                continue
            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(correct[mask]))
            ece += (float(n_bin) / n_total) * abs(avg_acc - avg_conf)
        return float(ece)

    def _evaluate_fold(
        self,
        model: BaseEstimator,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
    ) -> Tuple[float, float, float]:
        est = clone(model)
        est.fit(X_tr, y_tr)
        y_pred = np.asarray(est.predict(X_va)).ravel()
        bal = float(balanced_accuracy_score(y_va, y_pred))

        brier = float("nan")
        ece = float("nan")
        if self.include_calibration and hasattr(est, "predict_proba"):
            try:
                proba = np.asarray(est.predict_proba(X_va), dtype=float)
                if np.unique(y_va).size <= 2 and proba.ndim == 2 and proba.shape[1] == 2:
                    y01 = LabelEncoder().fit_transform(np.asarray(y_va).ravel())
                    brier = float(
                        np.mean((proba[:, 1] - np.asarray(y01, dtype=float)) ** 2)
                    )
                else:
                    brier = self._multiclass_brier_score(y_va, proba)
                ece = self._expected_calibration_error(y_va, proba)
            except Exception as exc:
                brier = float("nan")
                ece = float("nan")
        return bal, brier, ece, y_pred

    def _score_candidates(
        self,
        candidates: Dict[str, BaseEstimator],
        candidate_names: Sequence[str],
        X: np.ndarray,
        y: np.ndarray,
        *,
        seed: int,
        cv_splits: int,
    ) -> Tuple[List[OracleCandidateStats], Dict[str, Any], Dict[str, np.ndarray]]:
        """Score candidates via stratified CV.

        Returns (stats, race_meta, oof_preds) where *oof_preds* maps candidate
        name -> array of OOF predicted labels (same length as *y*; entries are
        set only at validation-fold indices, others left as ``-1``).
        """
        names = [str(n) for n in candidate_names if str(n) in candidates]
        if not names:
            return [], {"racing_applied": False, "racing_eliminated": []}, {}

        n_splits = self._n_splits(y, cv_splits)
        if n_splits <= 1:
            return [], {"racing_applied": False, "racing_eliminated": []}, {}
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        splits = list(splitter.split(np.asarray(X, dtype=float), np.asarray(y).ravel()))

        fold_scores: Dict[str, List[float]] = {name: [] for name in names}
        fold_brier: Dict[str, List[float]] = {name: [] for name in names}
        fold_ece: Dict[str, List[float]] = {name: [] for name in names}
        # OOF predictions for diversity matrix (Fix 3).
        n_total = int(np.asarray(y).ravel().size)
        oof_preds: Dict[str, np.ndarray] = {name: np.full(n_total, -1, dtype=object) for name in names}

        active = list(names)
        eliminated: List[str] = []
        racing_applied = False

        for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
            X_tr = np.asarray(X, dtype=float)[tr_idx]
            y_tr = np.asarray(y).ravel()[tr_idx]
            X_va = np.asarray(X, dtype=float)[va_idx]
            y_va = np.asarray(y).ravel()[va_idx]

            for name in list(active):
                model = candidates[name]
                try:
                    score, brier, ece_val, y_pred_fold = self._evaluate_fold(model, X_tr, y_tr, X_va, y_va)
                    oof_preds[name][va_idx] = y_pred_fold
                except Exception as exc:
                    score, brier, ece_val = float("nan"), float("nan"), float("nan")
                fold_scores[name].append(float(score))
                fold_brier[name].append(float(brier))
                fold_ece[name].append(float(ece_val))

            if (
                self.enable_hoeffding_racing
                and len(active) > 2
                and fold_idx >= 2
            ):
                racing_applied = True
                means: Dict[str, float] = {}
                radii: Dict[str, float] = {}
                for name in active:
                    vals = np.asarray(fold_scores[name], dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        means[name] = -np.inf
                        radii[name] = 1.0
                        continue
                    m = float(np.mean(vals))
                    rad = float(math.sqrt(math.log(2.0 / self.hoeffding_delta) / (2.0 * vals.size)))
                    means[name] = m
                    radii[name] = rad
                best_lb = max(float(means[n] - radii[n]) for n in active)
                next_active: List[str] = []
                for name in active:
                    ub = float(means[name] + radii[name])
                    if ub + 1e-12 < best_lb:
                        eliminated.append(name)
                    else:
                        next_active.append(name)
                active = next_active if next_active else active

        stats: List[OracleCandidateStats] = []
        rng = np.random.default_rng(int(seed))
        alpha_lo = float((1.0 - self.bbc_ci_level) / 2.0)
        alpha_hi = float(1.0 - alpha_lo)

        # Current-profile fix: flatten complexity priors on larger multiclass tasks
        # so the oracle is less biased toward globally "safe" linear families.
        adjusted_priors = (
            _adjust_complexity_priors(self.n_classes)
            if self.flatten_multiclass_complexity
            else dict(CLASSIFIER_COMPLEXITY_PRIOR)
        )

        for name in names:
            vals = np.asarray(fold_scores.get(name, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            mean_score = float(np.mean(vals))
            std_score = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            min_ratio = float(np.min(vals) / max(1e-12, mean_score)) if np.isfinite(mean_score) and mean_score > 0 else 0.0

            complexity = float(adjusted_priors.get(name, 0.5))

            brier_vals = np.asarray(fold_brier.get(name, []), dtype=float)
            brier_vals = brier_vals[np.isfinite(brier_vals)]
            if brier_vals.size > 0:
                calibration = float(np.clip(1.0 - float(np.mean(brier_vals)), 0.0, 1.0))
            else:
                calibration = 0.5

            ece_vals = np.asarray(fold_ece.get(name, []), dtype=float)
            ece_vals = ece_vals[np.isfinite(ece_vals)]
            mean_ece = float(np.mean(ece_vals)) if ece_vals.size > 0 else float("nan")

            bbc_corr = float(mean_score)
            ci_low = float("nan")
            ci_high = float("nan")
            if self.enable_bbc and self.bbc_bootstrap_rounds > 0 and vals.size > 1:
                boots = []
                for _ in range(int(self.bbc_bootstrap_rounds)):
                    sample = rng.choice(vals, size=vals.size, replace=True)
                    boots.append(float(np.mean(sample)))
                boot_arr = np.asarray(boots, dtype=float)
                bias = float(np.mean(boot_arr) - mean_score)
                bbc_corr = float(mean_score - bias)
                ci_low = float(np.quantile(boot_arr, alpha_lo))
                ci_high = float(np.quantile(boot_arr, alpha_hi))

            stats.append(
                OracleCandidateStats(
                    name=str(name),
                    scores=vals,
                    mean_score=float(mean_score),
                    std_score=float(std_score),
                    min_mean_ratio=float(np.clip(min_ratio, 0.0, 1.0)),
                    complexity_score=float(np.clip(complexity, 0.0, 1.0)),
                    calibration_score=float(np.clip(calibration, 0.0, 1.0)),
                    ece_score=float(mean_ece),
                    bbc_corrected_score=float(bbc_corr),
                    bbc_ci_low=float(ci_low),
                    bbc_ci_high=float(ci_high),
                )
            )

        return stats, {
            "racing_applied": bool(racing_applied),
            "racing_eliminated": list(eliminated),
            "n_splits": int(n_splits),
        }, oof_preds

    @staticmethod
    def _build_diversity_matrix(
        oof_preds: Dict[str, np.ndarray],
        names: Sequence[str],
        y_true: np.ndarray,
    ) -> np.ndarray:
        """Build a pairwise diversity matrix from OOF predictions.

        For each pair (i, j), diversity is the fraction of samples where
        exactly one classifier is correct and the other is wrong (double-fault
        diversity).  Higher diversity means more complementary error patterns.
        Returns a matrix in [0, 1] suitable for use as an oracle matrix.
        """
        m = len(names)
        div = np.full((m, m), 0.5, dtype=float)
        y = np.asarray(y_true).ravel()
        for i, j in combinations(range(m), 2):
            pi = oof_preds.get(names[i])
            pj = oof_preds.get(names[j])
            if pi is None or pj is None:
                continue
            # Only compare indices where both classifiers produced predictions.
            valid = (np.asarray(pi) != -1) & (np.asarray(pj) != -1)
            if int(valid.sum()) < 4:
                continue
            ci = np.asarray(pi)[valid] == y[valid]
            cj = np.asarray(pj)[valid] == y[valid]
            # Diversity = fraction where exactly one is correct.
            disagreement = float(np.mean(ci != cj))
            # Transform to pairwise preference: higher diversity → higher score.
            # Both share the same diversity so matrix is symmetric (= 0.5).
            div[i, j] = 0.5
            div[j, i] = 0.5
        # For use as an oracle matrix, convert to scalar scores and
        # build a proper asymmetric pairwise matrix.
        # Score each candidate by mean diversity with all others.
        div_scores: List[float] = []
        for i in range(m):
            dvals = []
            for j in range(m):
                if i == j:
                    continue
                pi = oof_preds.get(names[i])
                pj = oof_preds.get(names[j])
                if pi is None or pj is None:
                    continue
                valid = (np.asarray(pi) != -1) & (np.asarray(pj) != -1)
                if int(valid.sum()) < 4:
                    continue
                ci = np.asarray(pi)[valid] == y[valid]
                cj = np.asarray(pj)[valid] == y[valid]
                dvals.append(float(np.mean(ci != cj)))
            div_scores.append(float(np.mean(dvals)) if dvals else 0.0)

        mat, _ = matrix_from_scalar_scores(
            div_scores,
            tie_margin=0.01,
            use_qre_smoothing=False,
            qre_temperature_gamma=1.0,
        )
        return np.asarray(mat, dtype=float)

    def run(
        self,
        candidates: Dict[str, BaseEstimator],
        candidate_names: Sequence[str],
        X: np.ndarray,
        y: np.ndarray,
        *,
        seed: int,
        cv_splits: int,
        top_k: int,
    ) -> Dict[str, Any]:
        stats, race_meta, oof_preds = self._score_candidates(
            candidates=candidates,
            candidate_names=candidate_names,
            X=X,
            y=y,
            seed=seed,
            cv_splits=cv_splits,
        )
        if not stats:
            return {
                "selected_names": ["lr"],
                "weights": {"lr": 1.0},
                "oracle_weights": {"performance": 1.0},
                "payoff": np.asarray([[0.0]], dtype=float),
                "candidate_stats": {},
                "oracle_matrices": {"performance": np.asarray([[0.5]], dtype=float)},
                "race_meta": race_meta,
            }

        names = [s.name for s in stats]
        m = int(len(names))

        perf = np.full((m, m), 0.5, dtype=float)
        for i, j in combinations(range(m), 2):
            p = _pairwise_pref_from_fold_scores(
                stats[i].scores,
                stats[j].scores,
                pairwise_delta=self.pairwise_delta,
            )
            perf[i, j] = float(np.clip(p, 0.0, 1.0))
            perf[j, i] = float(np.clip(1.0 - p, 0.0, 1.0))
        np.fill_diagonal(perf, 0.5)

        robustness_scores = [float(s.min_mean_ratio) for s in stats]
        complexity_scores = [float(s.complexity_score) for s in stats]
        calibration_scores = [float(s.calibration_score) for s in stats]

        robust_mat, _ = matrix_from_scalar_scores(
            robustness_scores,
            tie_margin=0.01,
            use_qre_smoothing=False,
            qre_temperature_gamma=1.0,
        )
        complexity_mat, _ = matrix_from_scalar_scores(
            complexity_scores,
            tie_margin=0.01,
            use_qre_smoothing=False,
            qre_temperature_gamma=1.0,
        )

        oracle_mats: Dict[str, np.ndarray] = {
            "performance": np.asarray(perf, dtype=float),
            "robustness": np.asarray(robust_mat, dtype=float),
            "complexity": np.asarray(complexity_mat, dtype=float),
        }
        if self.include_calibration:
            cal_mat, _ = matrix_from_scalar_scores(
                calibration_scores,
                tie_margin=0.01,
                use_qre_smoothing=False,
                qre_temperature_gamma=1.0,
            )
            oracle_mats["calibration"] = np.asarray(cal_mat, dtype=float)

        # Fix 3: diversity oracle matrix from OOF prediction complementarity.
        if self.include_diversity and oof_preds and m >= 2:
            div_mat = self._build_diversity_matrix(oof_preds, names, np.asarray(y).ravel())
            if div_mat.shape == (m, m):
                oracle_mats["diversity"] = div_mat

        if self.weighting_mode == "uniform":
            oracle_weights = {name: 1.0 for name in oracle_mats}
        elif self.weighting_mode == "shapley":
            oracle_weights, _ = fit_shapley_weights(
                oracle_mats,
                reference="performance",
            )
        elif self.weighting_mode == "banzhaf":
            oracle_weights, _ = compute_banzhaf_values(
                oracle_mats,
                reference="performance",
            )
        else:
            oracle_weights = fit_tritrust_weights(
                oracle_mats,
                reference="performance",
                allow_negative=True,
                no_flip_oracles={"complexity", "calibration", "diversity"},
            )
            if self.include_james_stein:
                fold_counts = [
                    int(np.isfinite(np.asarray(s.scores, dtype=float).ravel()).sum())
                    for s in stats
                ]
                n_eff = float(min(fold_counts)) if fold_counts else 1.0
                oracle_weights = _james_stein_shrinkage(oracle_weights, effective_n=n_eff)

        payoff = aggregate_payoff_matrix(oracle_mats, oracle_weights)
        prior = np.full(m, 1.0 / float(m), dtype=float)
        p_star = mirror_descent_reference_regularized(
            np.asarray(payoff, dtype=float),
            np.asarray(prior, dtype=float),
            steps=300,
            eta=0.15,
            lambda_=0.08,
            tol_kl=1e-7,
            return_history=False,
        )
        p_star = np.asarray(np.nan_to_num(p_star, nan=0.0, posinf=0.0, neginf=0.0), dtype=float).ravel()
        if p_star.size != m or float(np.sum(p_star)) <= 1e-12:
            p_star = np.full(m, 1.0 / float(m), dtype=float)
        else:
            p_star = p_star / float(np.sum(p_star))

        order = list(np.argsort(-p_star))
        k = int(max(1, min(int(top_k), m)))
        selected = [names[i] for i in order[:k]]

        stats_map = {
            s.name: {
                "mean_score": float(s.mean_score),
                "std_score": float(s.std_score),
                "min_mean_ratio": float(s.min_mean_ratio),
                "complexity_score": float(s.complexity_score),
                "calibration_score": float(s.calibration_score),
                "ece_score": float(s.ece_score),
                "bbc_corrected_score": float(s.bbc_corrected_score),
                "bbc_ci_low": float(s.bbc_ci_low),
                "bbc_ci_high": float(s.bbc_ci_high),
                "n_folds": int(s.scores.size),
            }
            for s in stats
        }

        return {
            "selected_names": list(selected),
            "weights": {names[i]: float(p_star[i]) for i in range(m)},
            "oracle_weights": {str(k0): float(v0) for k0, v0 in oracle_weights.items()},
            "payoff": np.asarray(payoff, dtype=float),
            "candidate_stats": stats_map,
            "oracle_matrices": {key: np.asarray(val, dtype=float) for key, val in oracle_mats.items()},
            "race_meta": race_meta,
        }


class MNPOClassifierBackend(ClassifierBackend):
    """MNPO-hybrid backend: regime gating + oracle selection + per-family HPO."""

    def __init__(
        self,
        *,
        candidate_names: Sequence[str],
        oracle_k: int = 1,
        oracle_weighting_mode: str = "tritrust",
        oracle_include_calibration: bool = True,
        oracle_include_james_stein: bool = True,
        enable_hoeffding_racing: bool = True,
        hoeffding_delta: float = 0.10,
        enable_bbc: bool = True,
        bbc_bootstrap_rounds: int = 200,
        bbc_ci_level: float = 0.90,
        enable_ensemble: bool = False,
        multiclass_ensemble_threshold: int = 4,
        oracle_behavior_profile: str = "current",
        flaml_time_budget: int = 60,
        flaml_metric: str = "accuracy",
        flaml_n_jobs: int = 1,
        use_per_family_flaml: bool = True,
        min_n_for_automl: int = 50,
        min_n_per_class_for_automl: int = 10,
        min_n_per_class_for_cv: int = 5,
        max_p_over_n_for_automl: int = 200,
        lr_max_iter: int = 10000,
        use_hybrid_score: bool = False,
        hybrid_balanced_weight: float = 0.6,
        hybrid_macro_f1_weight: float = 0.4,
        n_jobs: int = 1,
        build_xgb_model_fn: Optional[Callable[[np.ndarray, int], OptionalModelBuildReturn]] = None,
        build_tabpfn_model_fn: Optional[Callable[[int], OptionalModelBuildReturn]] = None,
        warn_missing_backend_fn: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        self._candidate_names = tuple(str(c) for c in candidate_names if str(c))
        self.oracle_k = int(max(1, oracle_k))
        self.oracle_weighting_mode = str(oracle_weighting_mode or "tritrust")
        self.oracle_include_calibration = bool(oracle_include_calibration)
        self.oracle_include_james_stein = bool(oracle_include_james_stein)
        self.enable_hoeffding_racing = bool(enable_hoeffding_racing)
        self.hoeffding_delta = float(np.clip(hoeffding_delta, 1e-6, 0.99))
        self.enable_bbc = bool(enable_bbc)
        self.bbc_bootstrap_rounds = int(max(0, bbc_bootstrap_rounds))
        self.bbc_ci_level = float(np.clip(bbc_ci_level, 0.50, 0.999))
        self.enable_ensemble = bool(enable_ensemble)
        self.multiclass_ensemble_threshold = int(max(0, multiclass_ensemble_threshold))
        behavior_profile = str(oracle_behavior_profile or "current").strip().lower()
        if behavior_profile not in {"current", "val18_compat"}:
            behavior_profile = "current"
        self.oracle_behavior_profile = behavior_profile
        self.flaml_time_budget = int(max(1, flaml_time_budget))
        self.flaml_metric = FLAMLBackend._normalize_metric_key(flaml_metric, default="accuracy")
        self.flaml_n_jobs = int(max(1, flaml_n_jobs))
        self.use_per_family_flaml = bool(use_per_family_flaml)
        self.min_n_for_automl = int(max(2, min_n_for_automl))
        self.min_n_per_class_for_automl = int(max(2, min_n_per_class_for_automl))
        self.min_n_per_class_for_cv = int(max(2, min_n_per_class_for_cv))
        self.max_p_over_n_for_automl = int(max(1, max_p_over_n_for_automl))
        self.lr_max_iter = int(max(500, lr_max_iter))
        self.use_hybrid_score = bool(use_hybrid_score)
        self.hybrid_balanced_weight = float(max(0.0, hybrid_balanced_weight))
        self.hybrid_macro_f1_weight = float(max(0.0, hybrid_macro_f1_weight))
        self.n_jobs = int(max(1, n_jobs))
        self._build_xgb_model_fn = build_xgb_model_fn
        self._build_tabpfn_model_fn = build_tabpfn_model_fn
        self._warn_missing_backend_fn = warn_missing_backend_fn
        self._last_candidates: Dict[str, BaseEstimator] = {}

    def name(self) -> str:
        return "mnpo_hybrid"

    def get_candidates(self) -> Optional[Dict[str, BaseEstimator]]:
        return dict(self._last_candidates)

    def _warn_missing(self, model_name: str, package_name: str, reason: Optional[str] = None) -> None:
        if self._warn_missing_backend_fn is not None:
            self._warn_missing_backend_fn(model_name, package_name, reason)

    def _filtered_candidates_by_regime(self, *, n_samples: int, n_features: int) -> Tuple[str, List[str], List[str]]:
        regime = classify_regime(n_samples=int(n_samples), n_features=int(n_features))
        names = list(self._candidate_names) if self._candidate_names else ["lr", "svm_rbf"]
        allowed = REGIME_POOLS.get(regime)
        dropped_by_regime: List[str] = []
        if allowed is not None:
            keep = []
            allowed_set = set(allowed)
            for name in names:
                if name in allowed_set:
                    keep.append(name)
                else:
                    dropped_by_regime.append(name)
            names = keep
        names, alias_dropped = _unique_with_alias_handling(names)
        dropped = list(dropped_by_regime) + list(alias_dropped)
        if not names:
            names = ["lr"]
        return regime, names, dropped

    def _make_sklearn_backend(self, *, candidate_names: Sequence[str], allow_tree_models: bool) -> SklearnBackend:
        return SklearnBackend(
            candidate_names=tuple(candidate_names),
            lr_max_iter=int(self.lr_max_iter),
            use_hybrid_score=bool(self.use_hybrid_score),
            hybrid_balanced_weight=float(self.hybrid_balanced_weight),
            hybrid_macro_f1_weight=float(self.hybrid_macro_f1_weight),
            allow_tree_models=bool(allow_tree_models),
            n_jobs=int(getattr(self, 'n_jobs', 1) or 1),
            build_xgb_model_fn=self._build_xgb_model_fn,
            build_tabpfn_model_fn=self._build_tabpfn_model_fn,
            warn_missing_backend_fn=self._warn_missing_backend_fn,
        )

    def _per_family_budget(self, weights: Dict[str, float], selected: Sequence[str]) -> Dict[str, int]:
        total = int(max(1, self.flaml_time_budget))
        if len(selected) <= 1:
            return {str(selected[0]): total} if selected else {}

        raw = np.asarray([float(max(0.0, weights.get(name, 0.0))) for name in selected], dtype=float)
        if float(np.sum(raw)) <= 1e-12:
            raw = np.full(len(selected), 1.0 / float(max(1, len(selected))), dtype=float)
        else:
            raw = raw / float(np.sum(raw))

        # Keep FLAML's practical floor where possible, but never violate
        # total-budget feasibility when many families are selected.
        floor = 15 if total >= 15 * len(selected) else 1
        budgets = {str(name): int(max(floor, int(round(total * raw[idx])))) for idx, name in enumerate(selected)}
        used = int(sum(budgets.values()))
        if used > total:
            # Normalize down while keeping floor.
            over = int(used - total)
            for name in sorted(selected, key=lambda n: budgets[str(n)], reverse=True):
                if over <= 0:
                    break
                can_cut = int(max(0, budgets[str(name)] - floor))
                if can_cut <= 0:
                    continue
                cut = int(min(can_cut, over))
                budgets[str(name)] -= cut
                over -= cut
        return budgets

    def _fit_family_with_flaml(
        self,
        *,
        family_name: str,
        fallback_model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int,
        n_classes: int,
        class_counts: np.ndarray,
        cv_splits: int,
        scoring: str,
        budget: int,
    ) -> Tuple[BaseEstimator, str, float, float, int, Dict[str, Any]]:
        fam = str(family_name)
        native = FLAML_NATIVE_BY_FAMILY.get(fam)
        if native is None:
            return fallback_model, fam, float("nan"), float("nan"), 0, {
                "classification_backend_used": "mnpo_hybrid_sklearn",
                "mnpo_flaml_fallback_reason": "family_not_supported_by_flaml",
                "mnpo_selected_family": fam,
            }

        backend = FLAMLBackend(
            time_budget=int(max(1, budget)),
            estimator_list=(str(native),),
            metric=str(self.flaml_metric),
            n_jobs=int(self.flaml_n_jobs),
            min_n_for_automl=int(self.min_n_for_automl),
            min_n_per_class_for_cv=int(self.min_n_per_class_for_cv),
            min_n_per_class_for_automl=int(self.min_n_per_class_for_automl),
            max_p_over_n_for_automl=int(self.max_p_over_n_for_automl),
        )
        try:
            model, model_name, score, std, n_splits, meta = backend.fit_and_select(
                np.asarray(X_train, dtype=float),
                np.asarray(y_train).ravel(),
                seed=int(seed),
                n_classes=int(n_classes),
                class_counts=np.asarray(class_counts, dtype=int),
                cv_splits=int(cv_splits),
                scoring=str(scoring),
            )
            meta = dict(meta or {})
            meta["mnpo_selected_family"] = str(fam)
            return model, model_name, float(score), float(std), int(n_splits), meta
        except Exception as exc:
            return fallback_model, fam, float("nan"), float("nan"), 0, {
                "classification_backend_used": "mnpo_hybrid_sklearn",
                "mnpo_flaml_fallback_reason": str(type(exc).__name__),
                "mnpo_selected_family": fam,
            }

    def fit_and_select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        seed: int,
        n_classes: int,
        class_counts: np.ndarray,
        cv_splits: int = 5,
        scoring: str = "balanced_accuracy",
    ) -> Tuple[BaseEstimator, str, float, float, int, Dict[str, Any]]:
        x = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train).ravel()
        counts = np.asarray(class_counts, dtype=int).ravel()
        if counts.size == 0 or int(np.min(counts)) < 2:
            lr = make_logistic_regression(
                random_state=int(seed),
                max_iter=10000,
                solver="lbfgs",
                penalty="l2",
                class_weight="balanced",
            )
            return lr, "lr", float("nan"), float("nan"), 0, {
                "classification_backend_used": "mnpo_hybrid_fallback",
                "classification_guard_reason": "insufficient_class_counts",
            }

        regime, filtered, dropped = self._filtered_candidates_by_regime(
            n_samples=int(x.shape[0]),
            n_features=int(x.shape[1]),
        )
        allow_tree_models = regime == REGIME_STANDARD
        sk_backend = self._make_sklearn_backend(
            candidate_names=tuple(filtered),
            allow_tree_models=bool(allow_tree_models),
        )
        candidates = sk_backend._build_candidates(X_train=x, y_train=y, seed=int(seed))
        self._last_candidates = dict(candidates)

        candidate_names = [name for name in filtered if name in candidates]
        if not candidate_names:
            candidate_names = ["lr"]

        behavior_profile = str(getattr(self, "oracle_behavior_profile", "current") or "current")
        use_val18_compat = behavior_profile == "val18_compat"

        # Current-profile fix: broaden the selected family set automatically on
        # larger multiclass tasks so the final stage is less likely to collapse
        # onto a single globally safe family.
        mc_threshold = int(getattr(self, 'multiclass_ensemble_threshold', 4))
        effective_ensemble = bool(self.enable_ensemble)
        effective_k = int(self.oracle_k)
        if not use_val18_compat and int(n_classes) >= mc_threshold and mc_threshold > 0:
            effective_ensemble = True
            effective_k = int(max(effective_k, 3))

        # Current-profile fix: add an OOF diversity oracle on multiclass tasks.
        include_diversity = (not use_val18_compat) and int(n_classes) >= 3

        oracle = ClassifierOracle(
            weighting_mode=str(self.oracle_weighting_mode),
            include_calibration=bool(self.oracle_include_calibration),
            include_james_stein=bool(self.oracle_include_james_stein),
            enable_hoeffding_racing=bool(self.enable_hoeffding_racing),
            hoeffding_delta=float(self.hoeffding_delta),
            enable_bbc=bool(self.enable_bbc),
            bbc_bootstrap_rounds=int(self.bbc_bootstrap_rounds),
            bbc_ci_level=float(self.bbc_ci_level),
            n_classes=int(n_classes),
            include_diversity=bool(include_diversity),
            flatten_multiclass_complexity=not use_val18_compat,
        )
        oracle_out = oracle.run(
            candidates=candidates,
            candidate_names=candidate_names,
            X=x,
            y=y,
            seed=int(seed),
            cv_splits=int(cv_splits),
            top_k=int(effective_k),
        )

        selected = list(oracle_out.get("selected_names") or ["lr"])
        selected = [name for name in selected if name in candidates] or ["lr"]
        weights = dict(oracle_out.get("weights") or {})

        top_name = str(selected[0])
        top_model = candidates[top_name]
        top_stats = dict((oracle_out.get("candidate_stats") or {}).get(top_name) or {})
        top_score = float(top_stats.get("bbc_corrected_score", top_stats.get("mean_score", float("nan"))))
        top_std = float(top_stats.get("std_score", float("nan")))
        n_splits = int((oracle_out.get("race_meta") or {}).get("n_splits", 0) or 0)

        model: BaseEstimator = top_model
        model_name = f"mnpo_{top_name}"
        hpo_meta: Dict[str, Any] = {}

        if self.use_per_family_flaml:
            budgets = self._per_family_budget(weights, selected)
            if effective_ensemble and len(selected) > 1:
                estimators: List[Tuple[str, BaseEstimator]] = []
                ens_meta: Dict[str, Any] = {}
                for fam in selected:
                    fam_model = candidates[fam]
                    budget = int(budgets.get(fam, max(15, int(self.flaml_time_budget // max(1, len(selected))))))
                    tuned_model, tuned_name, _, _, _, fam_meta = self._fit_family_with_flaml(
                        family_name=str(fam),
                        fallback_model=fam_model,
                        X_train=x,
                        y_train=y,
                        seed=int(seed),
                        n_classes=int(n_classes),
                        class_counts=counts,
                        cv_splits=int(cv_splits),
                        scoring=str(scoring),
                        budget=int(budget),
                    )
                    estimators.append((str(fam), tuned_model))
                    ens_meta[str(fam)] = {
                        "model_name": str(tuned_name),
                        "meta": dict(fam_meta or {}),
                        "budget": int(budget),
                    }
                model = VotingClassifier(estimators=estimators, voting="hard", n_jobs=_get_sklearn_n_jobs())
                model_name = "mnpo_ensemble_" + "__".join(str(f) for f in selected)
                hpo_meta = {
                    "mnpo_hpo_mode": "per_family_ensemble",
                    "mnpo_hpo_by_family": ens_meta,
                }
            else:
                budget = int(budgets.get(top_name, self.flaml_time_budget))
                tuned_model, tuned_name, tuned_score, tuned_std, tuned_n, tuned_meta = self._fit_family_with_flaml(
                    family_name=top_name,
                    fallback_model=top_model,
                    X_train=x,
                    y_train=y,
                    seed=int(seed),
                    n_classes=int(n_classes),
                    class_counts=counts,
                    cv_splits=int(cv_splits),
                    scoring=str(scoring),
                    budget=int(budget),
                )
                model = tuned_model
                model_name = str(tuned_name)
                if np.isfinite(tuned_score):
                    top_score = float(tuned_score)
                if np.isfinite(tuned_std):
                    top_std = float(tuned_std)
                if int(tuned_n) > 0:
                    n_splits = int(tuned_n)
                hpo_meta = {
                    "mnpo_hpo_mode": "per_family_single",
                    "mnpo_hpo_budget": int(budget),
                    "mnpo_hpo_meta": dict(tuned_meta or {}),
                }

        meta = {
            "classification_backend_used": self.name(),
            "classification_regime": str(regime),
            "classification_regime_pool": list(REGIME_POOLS.get(regime, tuple()) or tuple()),
            "classification_regime_dropped_candidates": list(dropped),
            "mnpo_selected_classifier": str(top_name),
            "mnpo_selected_candidates": list(selected),
            "mnpo_candidate_weights": {str(k): float(v) for k, v in weights.items()},
            "mnpo_oracle_weights": {
                str(k): float(v)
                for k, v in dict(oracle_out.get("oracle_weights") or {}).items()
            },
            "mnpo_candidate_stats": dict(oracle_out.get("candidate_stats") or {}),
            "mnpo_race_meta": dict(oracle_out.get("race_meta") or {}),
            "mnpo_oracle_behavior_profile": str(behavior_profile),
            "mnpo_multiclass_ensemble_auto": bool(effective_ensemble and not self.enable_ensemble),
            "mnpo_multiclass_diversity_enabled": bool(include_diversity),
            "mnpo_multiclass_complexity_flattened": bool(
                (not use_val18_compat)
                and int(n_classes) >= int(_MULTICLASS_COMPLEXITY_PRIOR_THRESHOLD)
            ),
            "mnpo_effective_oracle_k": int(effective_k),
        }
        meta.update(hpo_meta)
        return model, str(model_name), float(top_score), float(top_std), int(n_splits), meta

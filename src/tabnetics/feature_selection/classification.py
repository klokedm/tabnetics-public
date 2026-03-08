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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .methods.embedded import _get_sklearn_n_jobs
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

try:
    from tabnetics.core.mnpo import (
        aggregate_payoff_matrix,
        fit_tritrust_weights,
        james_stein_shrinkage,
        matrix_from_scalar_scores,
        mirror_descent_reference_regularized,
        pairwise_pref_from_fold_scores,
    )
except Exception as exc:
    from tabnetics.core.mnpo import (  # type: ignore
        aggregate_payoff_matrix,
        fit_tritrust_weights,
        james_stein_shrinkage,
        matrix_from_scalar_scores,
        mirror_descent_reference_regularized,
        pairwise_pref_from_fold_scores,
    )


logger = logging.getLogger(__name__)


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
        "dlda",
        "shrinkage_lda",
        "nsc",
        "pls_da_classifier",
        "nb",
    ),
    REGIME_HDLSS_MODERATE: (
        "lr",
        "elastic_net_lr",
        "svm_linear",
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
    ),
}

# Complexity prior: higher means simpler / preferred under HDLSS uncertainty.
CLASSIFIER_COMPLEXITY_PRIOR: Dict[str, float] = {
    "lr": 1.00,
    "elastic_net_lr": 0.98,
    "svm_linear": 0.96,
    "dlda": 0.92,
    "shrinkage_lda": 0.92,
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
        build_xgb_model_fn: Optional[Callable[[np.ndarray, int], Optional[BaseEstimator]]] = None,
        build_tabpfn_model_fn: Optional[Callable[[int], Optional[BaseEstimator]]] = None,
        warn_missing_backend_fn: Optional[Callable[[str, str], None]] = None,
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

    def name(self) -> str:
        return "sklearn"

    def get_candidates(self) -> Optional[Dict[str, BaseEstimator]]:
        return dict(self._last_candidates)

    def _warn_missing(self, model_name: str, package_name: str) -> None:
        if self._warn_missing_backend_fn is not None:
            self._warn_missing_backend_fn(model_name, package_name)

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
            xgb_model = self._build_xgb_model_fn(y_train, seed)
            if xgb_model is None:
                self._warn_missing("xgb", "xgboost")
            else:
                models["xgb"] = xgb_model

        if "tabpfn" in requested and self._build_tabpfn_model_fn is not None:
            tabpfn_model = self._build_tabpfn_model_fn(seed)
            if tabpfn_model is None:
                self._warn_missing("tabpfn", "tabpfn")
            else:
                models["tabpfn"] = tabpfn_model

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
        if "gpc" in requested and int(X_train.shape[0]) <= 200:
            models["gpc"] = make_pipeline(
                StandardScaler(),
                GaussianProcessClassifier(random_state=seed),
            )

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

        candidate_names_raw = [name for name in self._candidate_names if name in models]
        candidate_names, alias_dropped = _unique_with_alias_handling(candidate_names_raw)
        if not candidate_names:
            candidate_names = ["lr"]

        counts = np.asarray(class_counts, dtype=int).ravel()
        if counts.size == 0 or np.min(counts) < 2:
            return models["lr"], "lr", float("nan"), float("nan"), 0, {
                "model_cv_runtime_containment_reason": "insufficient_class_counts",
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
        evaluated = []

        # Whether train/test gap gating is active.
        _gap_active = bool(self.max_train_test_gap > 0)
        _penalty_active = bool(self.tree_complexity_penalty_enabled and self.tree_complexity_penalty_strength > 0)
        _need_train_score = bool(_gap_active or _penalty_active)

        def _eval_one_candidate(name: str) -> Optional[Tuple[str, float, float, int]]:
            """Evaluate a single candidate model; returns (name, score, std, n) or None."""
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
                    return None

                # Compute CV train/validation gap (uses only training data).
                gap = 0.0
                if _need_train_score:
                    train_scores = np.asarray(cv_result["train_score"], dtype=float).ravel()
                    finite_train = train_scores[np.isfinite(train_scores)]
                    if finite_train.size > 0:
                        gap = float(np.mean(finite_train) - np.mean(bal_scores))
                gap_map[str(name)] = gap

                # Hard gate: reject candidates whose gap exceeds threshold.
                if _gap_active and gap > self.max_train_test_gap:
                    gap_rejected.append(str(name))
                    return None

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
                return (str(name), float(score), float(std), int(score_arr.size))
            except Exception as exc:
                return None

        # CL-1: Parallel candidate evaluation via ThreadPoolExecutor.
        if self.n_jobs > 1 and len(candidate_names) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
                futures = {pool.submit(_eval_one_candidate, name): name for name in candidate_names}
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    if result is None:
                        continue
                    name, score, std, n = result
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
                if result is None:
                    continue
                rname, score, std, n = result
                score_map[rname] = score
                evaluated.append(rname)
                if score > best_score:
                    best_name = rname
                    best_score = score
                    best_std = std
                    best_n = n

        meta = {
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
        build_xgb_model_fn: Optional[Callable[[np.ndarray, int], Optional[BaseEstimator]]] = None,
        build_tabpfn_model_fn: Optional[Callable[[int], Optional[BaseEstimator]]] = None,
        warn_missing_backend_fn: Optional[Callable[[str, str], None]] = None,
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
    ):
        self.pairwise_delta = float(max(0.0, pairwise_delta))
        self.weighting_mode = str(weighting_mode or "tritrust").strip().lower()
        if self.weighting_mode not in {"tritrust", "uniform"}:
            self.weighting_mode = "tritrust"
        self.include_calibration = bool(include_calibration)
        self.include_james_stein = bool(include_james_stein)
        self.enable_hoeffding_racing = bool(enable_hoeffding_racing)
        self.hoeffding_delta = float(np.clip(hoeffding_delta, 1e-6, 0.99))
        self.enable_bbc = bool(enable_bbc)
        self.bbc_bootstrap_rounds = int(max(0, bbc_bootstrap_rounds))
        self.bbc_ci_level = float(np.clip(bbc_ci_level, 0.50, 0.999))

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
        return bal, brier, ece

    def _score_candidates(
        self,
        candidates: Dict[str, BaseEstimator],
        candidate_names: Sequence[str],
        X: np.ndarray,
        y: np.ndarray,
        *,
        seed: int,
        cv_splits: int,
    ) -> Tuple[List[OracleCandidateStats], Dict[str, Any]]:
        names = [str(n) for n in candidate_names if str(n) in candidates]
        if not names:
            return [], {"racing_applied": False, "racing_eliminated": []}

        n_splits = self._n_splits(y, cv_splits)
        if n_splits <= 1:
            return [], {"racing_applied": False, "racing_eliminated": []}
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        splits = list(splitter.split(np.asarray(X, dtype=float), np.asarray(y).ravel()))

        fold_scores: Dict[str, List[float]] = {name: [] for name in names}
        fold_brier: Dict[str, List[float]] = {name: [] for name in names}
        fold_ece: Dict[str, List[float]] = {name: [] for name in names}

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
                    score, brier, ece_val = self._evaluate_fold(model, X_tr, y_tr, X_va, y_va)
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

        for name in names:
            vals = np.asarray(fold_scores.get(name, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            mean_score = float(np.mean(vals))
            std_score = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            min_ratio = float(np.min(vals) / max(1e-12, mean_score)) if np.isfinite(mean_score) and mean_score > 0 else 0.0

            complexity = float(CLASSIFIER_COMPLEXITY_PRIOR.get(name, 0.5))

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
        }

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
        stats, race_meta = self._score_candidates(
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

        if self.weighting_mode == "uniform":
            oracle_weights = {name: 1.0 for name in oracle_mats}
        else:
            oracle_weights = fit_tritrust_weights(
                oracle_mats,
                reference="performance",
                allow_negative=True,
                no_flip_oracles={"complexity", "calibration"},
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
        build_xgb_model_fn: Optional[Callable[[np.ndarray, int], Optional[BaseEstimator]]] = None,
        build_tabpfn_model_fn: Optional[Callable[[int], Optional[BaseEstimator]]] = None,
        warn_missing_backend_fn: Optional[Callable[[str, str], None]] = None,
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

    def _warn_missing(self, model_name: str, package_name: str) -> None:
        if self._warn_missing_backend_fn is not None:
            self._warn_missing_backend_fn(model_name, package_name)

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

        oracle = ClassifierOracle(
            weighting_mode=str(self.oracle_weighting_mode),
            include_calibration=bool(self.oracle_include_calibration),
            include_james_stein=bool(self.oracle_include_james_stein),
            enable_hoeffding_racing=bool(self.enable_hoeffding_racing),
            hoeffding_delta=float(self.hoeffding_delta),
            enable_bbc=bool(self.enable_bbc),
            bbc_bootstrap_rounds=int(self.bbc_bootstrap_rounds),
            bbc_ci_level=float(self.bbc_ci_level),
        )
        oracle_out = oracle.run(
            candidates=candidates,
            candidate_names=candidate_names,
            X=x,
            y=y,
            seed=int(seed),
            cv_splits=int(cv_splits),
            top_k=int(self.oracle_k),
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
            if self.enable_ensemble and len(selected) > 1:
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
        }
        meta.update(hpo_meta)
        return model, str(model_name), float(top_score), float(top_std), int(n_splits), meta

"""Conservative fold-local elastic-net logistic path selection."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a core dependency
    pd = None  # type: ignore[assignment]

from tabnetics.core.compat import Parallel, delayed, make_logistic_regression


class ElasticNetPathSelectionError(RuntimeError):
    """Raised when no complete, converged path point can be selected."""


@dataclass(frozen=True, slots=True)
class ElasticNetPathResult:
    """Fold-aggregate result for one ``(C, l1_ratio)`` point."""

    C: float
    l1_ratio: float
    mean_score: float
    standard_error: float
    mean_average_precision: float
    mean_log_loss: float
    valid: bool
    failure_reason: str | None = None


def select_one_standard_error(
    results: Sequence[ElasticNetPathResult],
    *,
    metric: str,
) -> tuple[ElasticNetPathResult, float]:
    """Select the most regularized point inside the best point's 1-SE set.

    Smaller ``C`` is always preferred, followed by larger ``l1_ratio``.
    ``log_loss`` is minimized and ``average_precision`` is maximized.
    """

    metric_key = str(metric).strip().lower()
    if metric_key not in {"log_loss", "average_precision"}:
        raise ValueError("metric must be 'log_loss' or 'average_precision'.")
    valid = [
        result
        for result in results
        if result.valid
        and np.isfinite(result.mean_score)
        and np.isfinite(result.standard_error)
    ]
    if not valid:
        raise ElasticNetPathSelectionError("elastic_net_path_no_valid_configuration")
    if metric_key == "log_loss":
        best = min(
            valid,
            key=lambda item: (
                item.mean_score,
                -item.mean_average_precision,
                item.C,
                -item.l1_ratio,
            ),
        )
        threshold = float(best.mean_score + best.standard_error)
        eligible = [item for item in valid if item.mean_score <= threshold]
    else:
        best = max(
            valid,
            key=lambda item: (
                item.mean_score,
                -item.C,
                item.l1_ratio,
            ),
        )
        threshold = float(best.mean_score - best.standard_error)
        eligible = [item for item in valid if item.mean_score >= threshold]
    selected = min(
        eligible,
        key=lambda item: (
            item.C,
            -item.l1_ratio,
            -item.mean_average_precision,
        ),
    )
    return selected, threshold


class ElasticNetPathClassifier(ClassifierMixin, BaseEstimator):
    """Logistic elastic-net classifier with conservative nested path choice.

    Scaling, estimator fitting, and metric evaluation are repeated inside each
    validation fold. A path point is ineligible if any required fold warns on
    convergence, reaches ``max_iter``, or emits nonfinite state.
    """

    def __init__(
        self,
        *,
        C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
        l1_ratio_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        selection_metric: str = "log_loss",
        cv: int | Any = 3,
        class_weight: str | dict[Any, float] | None = None,
        max_iter: int = 10000,
        tol: float = 1e-4,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.C_grid = C_grid
        self.l1_ratio_grid = l1_ratio_grid
        self.selection_metric = selection_metric
        self.cv = cv
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _validate_configuration(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...], str]:
        Cs = tuple(float(value) for value in self.C_grid)
        ratios = tuple(float(value) for value in self.l1_ratio_grid)
        if not Cs or any(not np.isfinite(value) or value <= 0.0 for value in Cs):
            raise ValueError("C_grid must contain finite positive values.")
        if len(set(Cs)) != len(Cs):
            raise ValueError("C_grid must not contain duplicate values.")
        if not ratios or any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in ratios
        ):
            raise ValueError("l1_ratio_grid must contain finite values in [0, 1].")
        if len(set(ratios)) != len(ratios):
            raise ValueError("l1_ratio_grid must not contain duplicate values.")
        metric = str(self.selection_metric).strip().lower()
        if metric not in {"log_loss", "average_precision"}:
            raise ValueError(
                "selection_metric must be 'log_loss' or 'average_precision'."
            )
        if (
            int(self.max_iter) <= 0
            or not np.isfinite(float(self.tol))
            or float(self.tol) <= 0
        ):
            raise ValueError("max_iter and tol must be positive.")
        if int(self.n_jobs) == 0 or int(self.n_jobs) < -1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        return Cs, ratios, metric

    @staticmethod
    def _coerce_sample_weight(
        sample_weight: Sequence[float] | None,
        *,
        n_rows: int,
    ) -> np.ndarray | None:
        if sample_weight is None:
            return None
        weights = np.asarray(sample_weight, dtype=float).ravel()
        if weights.size != int(n_rows):
            raise ValueError("sample_weight length must match X and y.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sample_weight must be finite and nonnegative.")
        if not float(np.sum(weights)) > 0.0:
            raise ValueError("sample_weight must have positive total mass.")
        return weights

    def _split_pairs(
        self, X: np.ndarray, y: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if isinstance(self.cv, (int, np.integer)):
            requested = int(self.cv)
            if requested < 2:
                raise ValueError("cv must be at least 2.")
            _, counts = np.unique(y, return_counts=True)
            if counts.size == 0 or int(np.min(counts)) < requested:
                raise ElasticNetPathSelectionError(
                    "elastic_net_path_insufficient_class_count_for_cv"
                )
            splitter = StratifiedKFold(
                n_splits=requested,
                shuffle=True,
                random_state=self.random_state,
            )
            raw = splitter.split(X, y)
        elif hasattr(self.cv, "split") and callable(self.cv.split):
            raw = self.cv.split(X, y)
        else:
            raw = iter(self.cv)
        pairs = [
            (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
            for train, test in raw
        ]
        if len(pairs) < 2:
            raise ElasticNetPathSelectionError(
                "elastic_net_path_requires_at_least_two_folds"
            )
        expected = set(range(int(y.size)))
        test_multiplicity = np.zeros(int(y.size), dtype=int)
        for fold_index, (train, test) in enumerate(pairs):
            if train.size == 0 or test.size == 0:
                raise ElasticNetPathSelectionError(
                    f"elastic_net_path_empty_fold:{fold_index}"
                )
            if set(train.tolist()) & set(test.tolist()):
                raise ElasticNetPathSelectionError(
                    f"elastic_net_path_fold_overlap:{fold_index}"
                )
            if np.unique(train).size != train.size or np.unique(test).size != test.size:
                raise ElasticNetPathSelectionError(
                    f"elastic_net_path_duplicate_index:{fold_index}"
                )
            if not set(train.tolist()).issubset(expected) or not set(
                test.tolist()
            ).issubset(expected):
                raise ElasticNetPathSelectionError(
                    f"elastic_net_path_index_out_of_bounds:{fold_index}"
                )
            if set(train.tolist()) | set(test.tolist()) != expected:
                raise ElasticNetPathSelectionError(
                    f"elastic_net_path_incomplete_partition:{fold_index}"
                )
            if (
                np.unique(y[train]).size != np.unique(y).size
                or np.unique(y[test]).size != np.unique(y).size
            ):
                raise ElasticNetPathSelectionError(
                    f"elastic_net_path_missing_class_fold:{fold_index}"
                )
            test_multiplicity[test] += 1
        if np.any(test_multiplicity != 1):
            raise ElasticNetPathSelectionError(
                "elastic_net_path_uneven_validation_coverage"
            )
        return pairs

    def _new_model(self, *, C: float, l1_ratio: float) -> Any:
        return make_logistic_regression(
            C=float(C),
            l1_ratio=float(l1_ratio),
            penalty="elasticnet",
            solver="saga",
            class_weight=self.class_weight,
            max_iter=int(self.max_iter),
            tol=float(self.tol),
            random_state=self.random_state,
            warm_start=True,
        )

    @staticmethod
    def _average_precision(
        y_true: np.ndarray,
        probabilities: np.ndarray,
        classes: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> float:
        if classes.size == 2:
            target = (np.asarray(y_true) == classes[1]).astype(int)
            return float(
                average_precision_score(
                    target,
                    probabilities[:, 1],
                    sample_weight=sample_weight,
                )
            )
        targets = label_binarize(y_true, classes=classes)
        return float(
            average_precision_score(
                targets,
                probabilities,
                average="macro",
                sample_weight=sample_weight,
            )
        )

    @staticmethod
    def _fit_has_invalid_state(
        model: Any, caught: Iterable[warnings.WarningMessage]
    ) -> str | None:
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            return "convergence_warning"
        n_iter = np.asarray(getattr(model, "n_iter_", []), dtype=int).ravel()
        if n_iter.size == 0:
            return "missing_iteration_record"
        if np.any(n_iter >= int(model.max_iter)):
            return "iteration_limit_reached"
        coef = np.asarray(getattr(model, "coef_", []), dtype=float)
        intercept = np.asarray(getattr(model, "intercept_", []), dtype=float)
        if (
            coef.size == 0
            or not np.all(np.isfinite(coef))
            or not np.all(np.isfinite(intercept))
        ):
            return "nonfinite_fitted_parameters"
        return None

    @staticmethod
    def _require_class_weight_mass(
        y: np.ndarray,
        weights: np.ndarray | None,
        classes: np.ndarray,
        *,
        context: str,
    ) -> None:
        if weights is None:
            return
        for label in classes:
            mass = float(np.sum(weights[np.asarray(y) == label]))
            if not np.isfinite(mass) or mass <= 0.0:
                raise ElasticNetPathSelectionError(
                    f"elastic_net_path_nonpositive_class_weight_mass:{context}:{label}"
                )

    def _evaluate_fold_ratio_path(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        train: np.ndarray,
        test: np.ndarray,
        l1_ratio: float,
        C_values: Sequence[float],
        sample_weight: np.ndarray | None,
        classes: np.ndarray,
        fold_index: int,
    ) -> list[dict[str, Any]]:
        train_weight = None if sample_weight is None else sample_weight[train]
        test_weight = None if sample_weight is None else sample_weight[test]
        self._require_class_weight_mass(
            y[train], train_weight, classes, context=f"fold_{fold_index}_train"
        )
        self._require_class_weight_mass(
            y[test], test_weight, classes, context=f"fold_{fold_index}_validation"
        )
        scaler = StandardScaler()
        scaler.fit(X[train], sample_weight=train_weight)
        X_train = scaler.transform(X[train])
        X_test = scaler.transform(X[test])
        ordered_C = tuple(sorted(float(value) for value in C_values))
        model = self._new_model(C=ordered_C[0], l1_ratio=l1_ratio)
        path_id = f"fold_{fold_index}:l1_ratio_{l1_ratio:.17g}"
        records: list[dict[str, Any]] = []
        broken_reason: str | None = None
        for path_position, C in enumerate(ordered_C):
            if broken_reason is not None:
                records.append(
                    {
                        "fold": fold_index,
                        "path_id": path_id,
                        "path_position": path_position,
                        "C": C,
                        "l1_ratio": l1_ratio,
                        "warm_start_reused": path_position > 0,
                        "valid": False,
                        "failure_reason": f"upstream_path_failure:{broken_reason}",
                    }
                )
                continue
            model.set_params(C=C)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X_train, y[train], sample_weight=train_weight)
                invalid = self._fit_has_invalid_state(model, caught)
                if invalid is not None:
                    raise ElasticNetPathSelectionError(invalid)
                probabilities = np.asarray(model.predict_proba(X_test), dtype=float)
                if (
                    probabilities.shape != (test.size, classes.size)
                    or not np.all(np.isfinite(probabilities))
                    or np.any(probabilities < 0.0)
                    or np.any(probabilities > 1.0)
                ):
                    raise ElasticNetPathSelectionError(
                        "nonfinite_or_invalid_probability"
                    )
                loss = float(
                    log_loss(
                        y[test],
                        probabilities,
                        labels=classes,
                        sample_weight=test_weight,
                    )
                )
                ap = self._average_precision(
                    y[test], probabilities, classes, test_weight
                )
                if not (np.isfinite(loss) and np.isfinite(ap)):
                    raise ElasticNetPathSelectionError("nonfinite_validation_score")
                records.append(
                    {
                        "fold": fold_index,
                        "path_id": path_id,
                        "path_position": path_position,
                        "C": C,
                        "l1_ratio": l1_ratio,
                        "warm_start_reused": path_position > 0,
                        "log_loss": loss,
                        "average_precision": ap,
                        "n_iter": int(np.max(np.asarray(model.n_iter_, dtype=int))),
                        "valid": True,
                        "failure_reason": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - invalidate arbitrary estimator failure
                broken_reason = f"{type(exc).__name__}:{exc}"
                records.append(
                    {
                        "fold": fold_index,
                        "path_id": path_id,
                        "path_position": path_position,
                        "C": C,
                        "l1_ratio": l1_ratio,
                        "warm_start_reused": path_position > 0,
                        "valid": False,
                        "failure_reason": broken_reason,
                    }
                )
        return records

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Sequence[float] | None = None,
    ) -> ElasticNetPathClassifier:
        """Evaluate the path and refit the selected point on all training rows."""

        Cs, ratios, metric = self._validate_configuration()
        input_is_frame = bool(pd is not None and isinstance(X, pd.DataFrame))
        if input_is_frame:
            names = tuple(str(value) for value in X.columns)
            if len(set(names)) != len(names):
                raise ValueError("DataFrame feature names must be unique.")
        else:
            names = ()
        X_arr, y_arr = check_X_y(
            X, y, accept_sparse=False, dtype=float, ensure_all_finite=True
        )
        classes = np.unique(y_arr)
        if classes.size < 2:
            raise ValueError("ElasticNetPathClassifier requires at least two classes.")
        weights = self._coerce_sample_weight(sample_weight, n_rows=int(y_arr.size))
        self._require_class_weight_mass(y_arr, weights, classes, context="final")
        pairs = self._split_pairs(X_arr, y_arr)

        path_tasks = [
            (fold_index, train, test, ratio)
            for fold_index, (train, test) in enumerate(pairs)
            for ratio in sorted(ratios, reverse=True)
        ]
        if int(self.n_jobs) == 1:
            path_outputs = [
                self._evaluate_fold_ratio_path(
                    X_arr,
                    y_arr,
                    train=train,
                    test=test,
                    l1_ratio=ratio,
                    C_values=Cs,
                    sample_weight=weights,
                    classes=classes,
                    fold_index=fold_index,
                )
                for fold_index, train, test, ratio in path_tasks
            ]
        else:
            path_outputs = Parallel(n_jobs=int(self.n_jobs), prefer="threads")(
                delayed(self._evaluate_fold_ratio_path)(
                    X_arr,
                    y_arr,
                    train=train,
                    test=test,
                    l1_ratio=ratio,
                    C_values=Cs,
                    sample_weight=weights,
                    classes=classes,
                    fold_index=fold_index,
                )
                for fold_index, train, test, ratio in path_tasks
            )
        fold_records = [record for output in path_outputs for record in output]
        results: list[ElasticNetPathResult] = []
        for C in Cs:
            for ratio in ratios:
                matching = [
                    record
                    for record in fold_records
                    if float(record["C"]) == C and float(record["l1_ratio"]) == ratio
                ]
                valid = len(matching) == len(pairs) and all(
                    bool(record["valid"]) for record in matching
                )
                losses = [
                    float(record["log_loss"]) for record in matching if record["valid"]
                ]
                aps = [
                    float(record["average_precision"])
                    for record in matching
                    if record["valid"]
                ]
                failure_reason = next(
                    (
                        str(record["failure_reason"])
                        for record in matching
                        if not record["valid"]
                    ),
                    None,
                )
                selected_values = losses if metric == "log_loss" else aps
                if valid:
                    mean_score = float(np.mean(selected_values))
                    standard_error = float(
                        np.std(selected_values, ddof=1)
                        / math.sqrt(len(selected_values))
                    )
                    mean_loss = float(np.mean(losses))
                    mean_ap = float(np.mean(aps))
                else:
                    mean_score = float("nan")
                    standard_error = float("nan")
                    mean_loss = float("nan")
                    mean_ap = float("nan")
                results.append(
                    ElasticNetPathResult(
                        C=C,
                        l1_ratio=ratio,
                        mean_score=mean_score,
                        standard_error=standard_error,
                        mean_average_precision=mean_ap,
                        mean_log_loss=mean_loss,
                        valid=valid,
                        failure_reason=failure_reason,
                    )
                )

        selected, threshold = select_one_standard_error(results, metric=metric)
        scaler = StandardScaler()
        scaler.fit(X_arr, sample_weight=weights)
        model = self._new_model(C=selected.C, l1_ratio=selected.l1_ratio)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(scaler.transform(X_arr), y_arr, sample_weight=weights)
        invalid = self._fit_has_invalid_state(model, caught)
        if invalid is not None:
            raise ElasticNetPathSelectionError(
                f"elastic_net_path_final_fit_invalid:{invalid}"
            )

        self.classes_ = np.asarray(model.classes_)
        self.n_features_in_ = int(X_arr.shape[1])
        if input_is_frame:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        self._input_kind_ = "dataframe" if input_is_frame else "array"
        self.scaler_ = scaler
        self.model_ = model
        self.selected_C_ = float(selected.C)
        self.selected_l1_ratio_ = float(selected.l1_ratio)
        self.selection_threshold_ = float(threshold)
        self.selection_metric_ = metric
        self.path_results_ = tuple(results)
        self.cv_results_ = {
            "params": tuple(
                {"C": result.C, "l1_ratio": result.l1_ratio} for result in results
            ),
            "mean_test_score": np.asarray([result.mean_score for result in results]),
            "standard_error": np.asarray([result.standard_error for result in results]),
            "mean_log_loss": np.asarray([result.mean_log_loss for result in results]),
            "mean_average_precision": np.asarray(
                [result.mean_average_precision for result in results]
            ),
            "valid": np.asarray([result.valid for result in results], dtype=bool),
            "failure_reason": tuple(result.failure_reason for result in results),
            "fold_records": tuple(fold_records),
        }
        coefficients = np.asarray(model.coef_, dtype=float)
        self.support_mask_ = np.any(np.abs(coefficients) > 0.0, axis=0)
        self.convergence_records_ = tuple(fold_records)
        self.path_model_builds_ = len(path_tasks)
        self.path_fit_attempts_ = sum(
            1
            for record in fold_records
            if not str(record.get("failure_reason") or "").startswith(
                "upstream_path_failure:"
            )
        )
        self.provenance_ = {
            "selection_metric": metric,
            "one_se_threshold": self.selection_threshold_,
            "selected_C": self.selected_C_,
            "selected_l1_ratio": self.selected_l1_ratio_,
            "n_folds": len(pairs),
            "class_weight": self.class_weight,
            "sample_weight_used": weights is not None,
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "random_state": self.random_state,
            "path_order": "smallest_C_then_largest_l1_ratio",
            "warm_start": True,
            "path_model_builds": self.path_model_builds_,
            "path_fit_attempts": self.path_fit_attempts_,
            "n_jobs": int(self.n_jobs),
        }
        return self

    def _validated_inference(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes=["model_", "scaler_", "classes_"])
        is_frame = bool(pd is not None and isinstance(X, pd.DataFrame))
        if self._input_kind_ == "dataframe":
            if not is_frame:
                raise ValueError(
                    "A classifier fitted on a DataFrame requires a DataFrame."
                )
            received = tuple(str(value) for value in X.columns)
            expected = tuple(str(value) for value in self.feature_names_in_)
            if received != expected:
                raise ValueError("DataFrame feature identity/order differs from fit.")
        elif is_frame:
            raise ValueError(
                "A classifier fitted on an array rejects named DataFrame inference."
            )
        array = check_array(X, accept_sparse=False, dtype=float, ensure_all_finite=True)
        if int(array.shape[1]) != int(self.n_features_in_):
            raise ValueError("Inference feature width differs from fit.")
        return np.asarray(self.scaler_.transform(array), dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._validated_inference(X)))

    def predict_proba(self, X: Any) -> np.ndarray:
        probabilities = np.asarray(
            self.model_.predict_proba(self._validated_inference(X)), dtype=float
        )
        if not np.all(np.isfinite(probabilities)):
            raise ElasticNetPathSelectionError("nonfinite_inference_probability")
        return probabilities

    def decision_function(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.decision_function(self._validated_inference(X)))

    def get_support(self, indices: bool = False) -> np.ndarray:
        check_is_fitted(self, attributes=["support_mask_"])
        if indices:
            return np.flatnonzero(self.support_mask_)
        return np.asarray(self.support_mask_, dtype=bool).copy()


__all__ = [
    "ElasticNetPathClassifier",
    "ElasticNetPathResult",
    "ElasticNetPathSelectionError",
    "select_one_standard_error",
]

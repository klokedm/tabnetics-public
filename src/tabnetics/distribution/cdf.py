"""Train-only literal probability-scale CDF transformation.

The transformer in this module deliberately stays independent from the
pipeline's private Gaussianizing transforms.  Each feature is fitted from the
rows supplied to :meth:`fit`, and inference replays either the selected SciPy
family or an exact empirical mid-CDF built from those training rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
import scipy.stats as sps
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a core dependency
    pd = None  # type: ignore[assignment]

from tabnetics.datasets.schema import (
    DatasetSchema,
    FeatureLineage,
    FeatureRole,
    FeatureSpec,
    SchemaContractError,
)
from tabnetics.distribution.selector import UnifiedDistributionSelectorV6


class CDFTransformError(ValueError):
    """Raised when fitting or replay cannot satisfy the CDF contract."""


@dataclass(frozen=True, slots=True)
class FittedCDFFeature:
    """Serializable fitted state for one feature marginal."""

    feature_name: str
    family: str | None
    params: tuple[float, ...]
    empirical_values: tuple[float, ...]
    empirical_counts: tuple[int, ...]
    n_total: int
    n_observed: int
    n_missing: int
    fit_mode: str
    fallback_reason: str | None
    gof_p_min: float | None

    def to_record(self) -> dict[str, Any]:
        """Return metadata without copying the empirical reference sample."""

        return {
            "feature_name": self.feature_name,
            "family": self.family,
            "params": list(self.params),
            "n_total": int(self.n_total),
            "n_observed": int(self.n_observed),
            "n_missing": int(self.n_missing),
            "n_unique": len(self.empirical_values),
            "fit_mode": self.fit_mode,
            "fallback_reason": self.fallback_reason,
            "gof_p_min": self.gof_p_min,
        }


def _empirical_mid_cdf(values: np.ndarray, model: FittedCDFFeature) -> np.ndarray:
    unique = np.asarray(model.empirical_values, dtype=float)
    counts = np.asarray(model.empirical_counts, dtype=np.int64)
    if unique.size == 0 or counts.size != unique.size or int(model.n_observed) <= 0:
        raise CDFTransformError(
            f"Feature {model.feature_name!r} has invalid empirical CDF state."
        )
    cumulative = np.cumsum(counts, dtype=np.int64)
    left_positions = np.searchsorted(unique, values, side="left")
    right_positions = np.searchsorted(unique, values, side="right")
    less = np.where(
        left_positions > 0,
        cumulative[np.maximum(left_positions - 1, 0)],
        0,
    )
    less_or_equal = np.where(
        right_positions > 0,
        cumulative[np.maximum(right_positions - 1, 0)],
        0,
    )
    return (less.astype(float) + less_or_equal.astype(float)) / (
        2.0 * float(model.n_observed)
    )


class FittedCDFTransformer(TransformerMixin, BaseEstimator):
    """Map numeric features to literal CDF values using training-only state.

    Parameters are intentionally sklearn-clone compatible. ``distributions``
    may be ``None`` for the public selector's default SciPy family set or a
    mapping of replayable family names to SciPy distribution objects.
    """

    _VALID_CRITERIA = frozenset(
        {
            "simple",
            "cvm_p",
            "ks_p",
            "bic",
            "aic",
            "aicc",
            "cv",
            "cv_loglik",
            "crps",
            "mnpo_oracle",
        }
    )

    def __init__(
        self,
        *,
        distributions: Mapping[str, Any] | None = None,
        criterion: str = "simple",
        min_gof_p: float = 0.0,
        clip: tuple[float, float] = (1e-8, 1.0 - 1e-8),
        missing_policy: str = "propagate",
        nonfinite_policy: str = "raise",
        random_state: int | None = None,
        n_jobs: int = 1,
        robust_mode: bool = True,
        use_adaptive_strategy: bool = True,
        use_lrt: bool = True,
        use_cv: bool = True,
        use_lmoment_prescreen: bool = False,
        lmoment_prescreen_max_candidates: int = 0,
        fit_estimator: str = "mle",
        min_parametric_unique: int = 2,
    ) -> None:
        self.distributions = distributions
        self.criterion = criterion
        self.min_gof_p = min_gof_p
        self.clip = clip
        self.missing_policy = missing_policy
        self.nonfinite_policy = nonfinite_policy
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.robust_mode = robust_mode
        self.use_adaptive_strategy = use_adaptive_strategy
        self.use_lrt = use_lrt
        self.use_cv = use_cv
        self.use_lmoment_prescreen = use_lmoment_prescreen
        self.lmoment_prescreen_max_candidates = lmoment_prescreen_max_candidates
        self.fit_estimator = fit_estimator
        self.min_parametric_unique = min_parametric_unique

    def _validate_configuration(self) -> tuple[float, float]:
        criterion = str(self.criterion).strip().lower()
        if criterion not in self._VALID_CRITERIA:
            raise CDFTransformError(
                f"criterion must be one of {sorted(self._VALID_CRITERIA)!r}."
            )
        if len(tuple(self.clip)) != 2:
            raise CDFTransformError("clip must contain exactly two bounds.")
        low, high = (float(value) for value in self.clip)
        if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low < high < 1.0):
            raise CDFTransformError("clip must satisfy 0 < low < high < 1.")
        min_gof_p = float(self.min_gof_p)
        if not np.isfinite(min_gof_p) or not 0.0 <= min_gof_p <= 1.0:
            raise CDFTransformError("min_gof_p must be finite and in [0, 1].")
        if str(self.missing_policy).strip().lower() not in {"propagate", "raise"}:
            raise CDFTransformError("missing_policy must be 'propagate' or 'raise'.")
        if str(self.nonfinite_policy).strip().lower() != "raise":
            raise CDFTransformError("nonfinite_policy currently supports only 'raise'.")
        if int(self.n_jobs) == 0 or int(self.n_jobs) < -1:
            raise CDFTransformError("n_jobs must be -1 or a positive integer.")
        if int(self.lmoment_prescreen_max_candidates) < 0:
            raise CDFTransformError(
                "lmoment_prescreen_max_candidates must be non-negative."
            )
        if str(self.fit_estimator).strip().lower() not in {"mle", "mps"}:
            raise CDFTransformError("fit_estimator must be 'mle' or 'mps'.")
        if int(self.min_parametric_unique) < 2:
            raise CDFTransformError("min_parametric_unique must be at least 2.")
        if self.distributions is not None:
            if not isinstance(self.distributions, Mapping) or not self.distributions:
                raise CDFTransformError(
                    "distributions must be a non-empty mapping or None."
                )
            if any(not str(name).strip() for name in self.distributions):
                raise CDFTransformError("distribution names must be non-empty strings.")
        return low, high

    @staticmethod
    def _numeric_matrix(X: Any) -> np.ndarray:
        try:
            array = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise CDFTransformError(
                "FittedCDFTransformer requires numeric features."
            ) from exc
        if array.ndim != 2 or array.shape[1] <= 0:
            raise CDFTransformError(
                f"Expected a 2-D matrix with at least one feature, got {array.shape}."
            )
        return array

    @staticmethod
    def _empirical_state(
        values: np.ndarray,
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        unique, counts = np.unique(np.asarray(values, dtype=float), return_counts=True)
        return tuple(float(value) for value in unique), tuple(
            int(value) for value in counts
        )

    def _distribution_for(self, family: str) -> Any | None:
        if self.distributions is not None and family in self.distributions:
            return self.distributions[family]
        return getattr(sps, family, None)

    def _fit_feature(
        self,
        values: np.ndarray,
        feature_name: str,
        *,
        n_total: int,
    ) -> FittedCDFFeature:
        empirical_values, empirical_counts = self._empirical_state(values)
        base = {
            "feature_name": str(feature_name),
            "empirical_values": empirical_values,
            "empirical_counts": empirical_counts,
            "n_total": int(n_total),
            "n_observed": int(values.size),
            "n_missing": int(n_total - values.size),
        }
        if len(empirical_values) < int(self.min_parametric_unique):
            return FittedCDFFeature(
                family=None,
                params=(),
                fit_mode="empirical",
                fallback_reason=(
                    "constant_feature"
                    if len(empirical_values) <= 1
                    else "insufficient_unique_for_parametric"
                ),
                gof_p_min=None,
                **base,
            )

        selector = UnifiedDistributionSelectorV6(
            distributions=(
                None if self.distributions is None else dict(self.distributions)
            ),
            n_jobs=int(self.n_jobs),
            random_state=self.random_state,
            robust_mode=bool(self.robust_mode),
            use_adaptive_strategy=bool(self.use_adaptive_strategy),
            use_lrt=bool(self.use_lrt),
            use_cv=bool(self.use_cv),
            use_lmoment_prescreen=bool(self.use_lmoment_prescreen),
            lmoment_prescreen_max_candidates=int(self.lmoment_prescreen_max_candidates),
            fit_estimator=str(self.fit_estimator).strip().lower(),
        )
        try:
            family, result, _ = selector.select_best_distribution(
                values,
                criterion=str(self.criterion).strip().lower(),
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - selector families have heterogeneous failures
            return FittedCDFFeature(
                family=None,
                params=(),
                fit_mode="empirical",
                fallback_reason=f"selector_error:{type(exc).__name__}",
                gof_p_min=None,
                **base,
            )
        if family is None or result is None or result.params is None:
            return FittedCDFFeature(
                family=None,
                params=(),
                fit_mode="empirical",
                fallback_reason="no_accepted_fit",
                gof_p_min=None,
                **base,
            )
        try:
            params = tuple(float(value) for value in result.params)
        except (TypeError, ValueError):
            params = ()
        p_values = [
            float(value)
            for value in (getattr(result, "ks_p", None), getattr(result, "cvm_p", None))
            if value is not None and np.isfinite(float(value))
        ]
        gof_p_min = min(p_values) if p_values else None
        rejection: str | None = None
        if not params or not np.all(np.isfinite(np.asarray(params, dtype=float))):
            rejection = "nonfinite_or_empty_parameters"
        elif gof_p_min is None or gof_p_min < float(self.min_gof_p):
            rejection = "gof_below_threshold"
        dist = self._distribution_for(str(family))
        if rejection is None and dist is None:
            rejection = "unreplayable_family"
        if rejection is None:
            try:
                replay = np.asarray(dist.cdf(values, *params), dtype=float)
                if replay.shape != values.shape or not np.all(np.isfinite(replay)):
                    rejection = "invalid_training_replay"
            except Exception:  # noqa: BLE001 - SciPy families raise heterogeneous errors
                rejection = "training_replay_error"
        return FittedCDFFeature(
            family=(str(family) if rejection is None else None),
            params=(params if rejection is None else ()),
            fit_mode=("parametric" if rejection is None else "empirical"),
            fallback_reason=rejection,
            gof_p_min=gof_p_min,
            **base,
        )

    def fit(self, X: Any, y: Any = None) -> FittedCDFTransformer:
        """Fit every marginal from ``X`` only; ``y`` is ignored."""

        del y
        low, high = self._validate_configuration()
        array = self._numeric_matrix(X)
        if np.any(np.isinf(array)):
            raise CDFTransformError("Infinite values are not accepted during fit.")
        if str(self.missing_policy).strip().lower() == "raise" and np.any(
            np.isnan(array)
        ):
            raise CDFTransformError(
                "NaN values are disabled by missing_policy='raise'."
            )

        self.source_schema_ = DatasetSchema.from_input(X)
        self._input_kind_ = (
            "dataframe" if pd is not None and isinstance(X, pd.DataFrame) else "array"
        )
        self.n_features_in_ = int(array.shape[1])
        self.feature_names_in_ = np.asarray(
            self.source_schema_.feature_names, dtype=object
        )
        models: list[FittedCDFFeature] = []
        for index, feature_name in enumerate(self.source_schema_.feature_names):
            observed = np.asarray(array[:, index], dtype=float)
            observed = observed[~np.isnan(observed)]
            if observed.size == 0:
                raise CDFTransformError(
                    f"Feature {feature_name!r} has no finite training observations."
                )
            models.append(
                self._fit_feature(
                    observed,
                    feature_name,
                    n_total=int(array.shape[0]),
                )
            )
        self.feature_models_ = tuple(models)
        self.clip_ = (float(low), float(high))
        self.scipy_version_ = str(scipy.__version__)

        output_features = tuple(
            FeatureSpec(
                name=feature.name,
                role=FeatureRole.CONTINUOUS,
                dtype="float64",
                source_name=feature.source_name,
                annotations=feature.annotations,
            )
            for feature in self.source_schema_.features
        )
        output_lineage = tuple(
            FeatureLineage.from_parameters(
                output_name=model.feature_name,
                operation="literal_probability_cdf",
                input_names=(model.feature_name,),
                source_schema_hash=self.source_schema_.fingerprint,
                parameters={
                    "fit_mode": model.fit_mode,
                    "family": model.family,
                    "clip": list(self.clip_),
                    "fallback_reason": model.fallback_reason,
                },
            )
            for model in self.feature_models_
        )
        self.output_schema_ = DatasetSchema(
            features=output_features,
            lineage=output_lineage,
            metadata={
                "operation": "literal_probability_cdf",
                "source_schema_fingerprint": self.source_schema_.fingerprint,
                "scipy_version": self.scipy_version_,
            },
        )
        resolved_distributions = (
            sorted(str(name) for name in self.distributions)
            if self.distributions is not None
            else sorted(UnifiedDistributionSelectorV6._get_default_distributions())
        )
        self.provenance_ = {
            "resolved_config": {
                "distributions": resolved_distributions,
                "criterion": str(self.criterion).strip().lower(),
                "min_gof_p": float(self.min_gof_p),
                "clip": list(self.clip_),
                "missing_policy": str(self.missing_policy).strip().lower(),
                "nonfinite_policy": str(self.nonfinite_policy).strip().lower(),
                "random_state": self.random_state,
                "n_jobs": int(self.n_jobs),
                "robust_mode": bool(self.robust_mode),
                "use_adaptive_strategy": bool(self.use_adaptive_strategy),
                "use_lrt": bool(self.use_lrt),
                "use_cv": bool(self.use_cv),
                "use_lmoment_prescreen": bool(self.use_lmoment_prescreen),
                "lmoment_prescreen_max_candidates": int(
                    self.lmoment_prescreen_max_candidates
                ),
                "fit_estimator": str(self.fit_estimator).strip().lower(),
                "min_parametric_unique": int(self.min_parametric_unique),
            },
            "scipy_version": self.scipy_version_,
            "source_schema_fingerprint": self.source_schema_.fingerprint,
            "output_schema_fingerprint": self.output_schema_.fingerprint,
            "features": [model.to_record() for model in self.feature_models_],
        }
        self.last_transform_provenance_ = None
        return self

    def _validate_input_kind(self, X: Any) -> None:
        is_frame = bool(pd is not None and isinstance(X, pd.DataFrame))
        if self._input_kind_ == "dataframe" and not is_frame:
            raise SchemaContractError(
                "A transformer fitted on a DataFrame requires a DataFrame at inference."
            )
        if self._input_kind_ == "array" and is_frame:
            raise SchemaContractError(
                "A transformer fitted on a positional array rejects named DataFrame inference."
            )
        self.source_schema_.validate_inference_input(X)

    def transform(self, X: Any) -> Any:
        """Replay fitted marginal CDFs without refitting on inference rows."""

        check_is_fitted(self, attributes=["feature_models_", "source_schema_", "clip_"])
        self._validate_input_kind(X)
        array = self._numeric_matrix(X)
        if np.any(np.isinf(array)):
            raise CDFTransformError(
                "Infinite values are not accepted during transform."
            )
        if str(self.missing_policy).strip().lower() == "raise" and np.any(
            np.isnan(array)
        ):
            raise CDFTransformError(
                "NaN values are disabled by missing_policy='raise'."
            )

        out = np.full(array.shape, np.nan, dtype=float)
        fallback_counts: dict[str, int] = {}
        for index, model in enumerate(self.feature_models_):
            column = np.asarray(array[:, index], dtype=float)
            valid = ~np.isnan(column)
            if not np.any(valid):
                fallback_counts[model.feature_name] = 0
                continue
            values = column[valid]
            empirical = _empirical_mid_cdf(values, model)
            transformed = empirical.copy()
            fallback_count = int(values.size) if model.fit_mode == "empirical" else 0
            if model.fit_mode == "parametric" and model.family is not None:
                dist = self._distribution_for(model.family)
                if dist is not None:
                    try:
                        candidate = np.asarray(
                            dist.cdf(values, *model.params), dtype=float
                        )
                        usable = (
                            np.isfinite(candidate)
                            & (candidate >= 0.0)
                            & (candidate <= 1.0)
                        )
                        transformed[usable] = candidate[usable]
                        fallback_count = int(np.sum(~usable))
                    except Exception:  # noqa: BLE001 - replay falls back per value by contract
                        fallback_count = int(values.size)
                else:
                    fallback_count = int(values.size)
            out[valid, index] = np.clip(transformed, self.clip_[0], self.clip_[1])
            fallback_counts[model.feature_name] = fallback_count
        self.last_transform_provenance_ = {
            "rows": int(array.shape[0]),
            "parametric_value_fallback_counts": fallback_counts,
            "source_schema_fingerprint": self.source_schema_.fingerprint,
        }
        if pd is not None and isinstance(X, pd.DataFrame):
            return pd.DataFrame(out, index=X.index, columns=X.columns)
        return out

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_names_in_"])
        fitted = tuple(str(value) for value in self.feature_names_in_)
        if (
            input_features is not None
            and tuple(str(value) for value in input_features) != fitted
        ):
            raise SchemaContractError(
                "input_features do not match fitted feature identity/order."
            )
        return np.asarray(fitted, dtype=object)

    def get_output_schema(self) -> DatasetSchema:
        """Return the immutable output feature/lineage contract."""

        check_is_fitted(self, attributes=["output_schema_"])
        return self.output_schema_


__all__ = [
    "CDFTransformError",
    "FittedCDFFeature",
    "FittedCDFTransformer",
]

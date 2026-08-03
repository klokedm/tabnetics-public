"""Sklearn-facing train-only estimator for the DF+FS pipeline."""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from tabnetics.datasets.schema import DatasetSchema

from .pipeline import DFFSConfig, DistributionFeatureSelectionPipeline
from .resampling import FitResamplingContext


class DFFSClassifier(ClassifierMixin, BaseEstimator):
    """A sklearn-compatible, full-data fitted DF+FS classifier.

    ``fit`` resolves selector and model policy through training-only CV, then
    refits the selected model on every supplied row.  It never delegates to
    evaluation-oriented ``run`` or ``run_pre_split``.
    """

    def __init__(
        self,
        *,
        config: DFFSConfig | None = None,
        dataset_name: str = "dataset",
        schema: DatasetSchema | None = None,
        feature_alignment: str = "strict",
        random_state: int | None = None,
    ) -> None:
        self.config = config
        self.dataset_name = dataset_name
        self.schema = schema
        self.feature_alignment = feature_alignment
        self.random_state = random_state

    def _resolved_alignment(self) -> str:
        alignment = str(self.feature_alignment or "strict").strip().lower()
        if alignment not in {"strict", "reorder"}:
            raise ValueError(
                "feature_alignment must be 'strict' or the explicit 'reorder' policy."
            )
        return alignment

    def _build_fit_config(self) -> DFFSConfig:
        config = copy.deepcopy(self.config) if self.config is not None else DFFSConfig()
        if not isinstance(config, DFFSConfig):
            raise TypeError("config must be a DFFSConfig or None.")
        if self.random_state is not None:
            config.random_seed = int(self.random_state)
        return config

    def fit(
        self,
        X: Any,
        y: Sequence[Any],
        sample_weight: Optional[Sequence[float]] = None,
        *,
        batch_labels: Optional[Sequence[Any]] = None,
        resampling_context: FitResamplingContext | None = None,
    ) -> "DFFSClassifier":
        """Fit train-only components and refit the chosen model on all rows."""

        config = self._build_fit_config()
        alignment = self._resolved_alignment()
        typed_requested = bool(
            self.schema is not None
            or bool(getattr(config, "typed_input_enabled", False))
        )
        if typed_requested:
            fit_X = X
            fit_y = np.asarray(y)
        else:
            validated_X, fit_y = validate_data(
                self,
                X,
                y,
                reset=True,
                dtype=None,
                ensure_2d=True,
                ensure_all_finite="allow-nan",
            )
            # ``validate_data`` intentionally returns an ndarray for pandas
            # inputs.  Preserve the original numeric DataFrame for the
            # pipeline so its fitted schema can bind names and dtypes rather
            # than relying solely on sklearn's feature-name warning path.
            fit_X = X if hasattr(X, "columns") else validated_X
        pipeline = DistributionFeatureSelectionPipeline(config)
        components = pipeline._fit_components(
            fit_X,
            fit_y,
            dataset_name=str(self.dataset_name or "dataset"),
            seed=int(config.random_seed),
            batch_labels=batch_labels,
            sample_weight=sample_weight,
            schema=self.schema,
            resampling_context=resampling_context,
        )
        self._pipeline_ = pipeline
        self.components_ = components
        self.classes_ = np.asarray(components.classes).copy()
        self.n_features_in_ = int(
            components.source_schema.n_features
            if components.source_schema is not None
            else components.runtime_model.n_input_features
        )
        if components.source_schema is not None:
            self.feature_names_in_ = np.asarray(
                components.source_schema.feature_names, dtype=object
            )
        self.feature_alignment_ = alignment
        self.model_name_ = str(components.model_name)
        self.fit_provenance_ = dict(components.fit_provenance)
        self.config_snapshot_ = dict(components.config_snapshot)
        return self

    def _components_for_inference(self):
        check_is_fitted(self, attributes=["components_", "classes_"])
        return self.components_

    def _validate_inference_input(self, X: Any) -> Any:
        components = self._components_for_inference()
        if components.source_schema is not None:
            shape = getattr(X, "shape", None)
            if shape is not None and len(shape) == 2 and int(shape[1]) != int(self.n_features_in_):
                raise ValueError(
                    f"X has {int(shape[1])} features, but DFFSClassifier is expecting "
                    f"{int(self.n_features_in_)} features as input."
                )
            return X
        return validate_data(
            self,
            X,
            reset=False,
            dtype=None,
            ensure_2d=True,
        )

    def transform(
        self,
        X: Any,
        *,
        batch_labels: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        components = self._components_for_inference()
        transformed = components.transform(
            self._validate_inference_input(X),
            batch_labels=batch_labels,
            alignment=self.feature_alignment_,
        )
        self.last_inference_schema_report_ = dict(
            components.last_inference_schema_report
        )
        return transformed

    def predict(
        self,
        X: Any,
        *,
        batch_labels: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        components = self._components_for_inference()
        prediction = components.predict(
            self._validate_inference_input(X),
            batch_labels=batch_labels,
            alignment=self.feature_alignment_,
        )
        self.last_inference_schema_report_ = dict(
            components.last_inference_schema_report
        )
        return prediction

    def predict_proba(
        self,
        X: Any,
        *,
        batch_labels: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        components = self._components_for_inference()
        probabilities = components.predict_proba(
            self._validate_inference_input(X),
            batch_labels=batch_labels,
            alignment=self.feature_alignment_,
        )
        self.last_inference_schema_report_ = dict(
            components.last_inference_schema_report
        )
        return probabilities

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return selected feature names when lineage preserves them."""

        components = self._components_for_inference()
        if input_features is not None:
            received = np.asarray(input_features, dtype=object).ravel()
            if int(received.size) != int(self.n_features_in_):
                raise ValueError(
                    "input_features should have length equal to n_features_in_."
                )
            expected = getattr(self, "feature_names_in_", None)
            if expected is not None and not np.array_equal(received, expected):
                raise ValueError("input_features is not equal to feature_names_in_.")
        selected = dict(components.selected_feature_schema or {})
        records = list(selected.get("features") or [])
        names = [str(record.get("name")) for record in records if record.get("name")]
        model = components.runtime_model.classifier_model
        width = int(getattr(model, "n_features_in_", 0) or 0)
        if width <= 0:
            width = int(self.transform(np.zeros((1, self.n_features_in_), dtype=float)).shape[1])
        if names:
            ratio_meta = dict(components.runtime_model.stage2_ratio_meta or {})
            if bool(ratio_meta.get("stage2_ratio_features_applied", False)):
                base_names = tuple(names)
                used = set(names)
                for position, pair in enumerate(list(ratio_meta.get("stage2_ratio_pairs", []) or [])):
                    numerator = int(pair.get("numerator", -1))
                    denominator = int(pair.get("denominator", -1))
                    if 0 <= numerator < len(base_names) and 0 <= denominator < len(base_names):
                        base = (
                            f"stage2_ratio_{base_names[numerator]}_over_"
                            f"{base_names[denominator]}"
                        )
                    else:
                        base = f"stage2_ratio_{position}"
                    candidate = base
                    suffix = 1
                    while candidate in used:
                        candidate = f"{base}_{suffix}"
                        suffix += 1
                    names.append(candidate)
                    used.add(candidate)
            while len(names) < width:
                names.append(f"feature_{len(names)}")
            return np.asarray(names[:width], dtype=object)
        return np.asarray([f"feature_{index}" for index in range(width)], dtype=object)

    def to_safe_bundle(self) -> dict[str, Any]:
        """Export an allowlisted non-executable v2 bundle when the route supports it."""

        components = self._components_for_inference()
        from .bundle import create_safe_dffs_bundle

        return create_safe_dffs_bundle(components)


__all__ = ["DFFSClassifier"]

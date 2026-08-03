from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold

import tabnetics.classification.backends as classifier_backends
from tabnetics.datasets.schema import DatasetSchema, FeatureRole
from tabnetics.pipeline.pipeline import (
    ClassificationConfig,
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
)
from tabnetics.pipeline.preprocessing import (
    FoldLocalPreprocessor,
    TransformedInput,
    TypedInputCapabilityError,
)


def _frame(rows: int = 30) -> tuple[pd.DataFrame, np.ndarray, DatasetSchema]:
    frame = pd.DataFrame(
        {
            "number": np.arange(rows, dtype=float),
            "category": ["a", "b"] * (rows // 2) + (["a"] if rows % 2 else []),
        }
    )
    y = np.asarray([0, 1] * (rows // 2) + ([0] if rows % 2 else []))
    schema = DatasetSchema.from_dataframe(
        frame,
        roles={
            "number": FeatureRole.CONTINUOUS,
            "category": FeatureRole.CATEGORICAL,
        },
    )
    return frame, y, schema


def _native_config(estimator: str) -> DFFSConfig:
    classifier = ClassificationConfig(
        model_candidates=(estimator,),
        include_elastic_net_model=False,
        include_rf_model=False,
        include_knn_model=False,
        include_svm_linear_model=False,
        include_dlda_model=False,
        include_nb_model=False,
        native_categorical_stage2_enabled=True,
        native_categorical_stage2_estimator=estimator,
        stage2_ratio_augmentation_enabled=False,
        conformal_enabled=False,
        posthoc_calibration_enabled=False,
        runtime_containment_enabled=False,
    )
    return DFFSConfig(
        typed_input_enabled=True,
        enabled_methods=("anova_f",),
        n_final_features=2,
        fs_fraction=1.0,
        prefilter_top_k=2,
        prefilter_variance_floor_enabled=False,
        folding_method="none",
        apply_cdf_transform=False,
        enable_ratio_features=False,
        batch_correction="none",
        calibration_reporting_enabled=False,
        classification=classifier,
    )


class _RecordingCatBoost(BaseEstimator, ClassifierMixin):
    calls: list[dict[str, object]] = []

    def __init__(
        self,
        depth: int = 6,
        learning_rate: float = 0.05,
        n_estimators: int = 250,
        loss_function: str = "Logloss",
        verbose: bool = False,
        random_seed: int = 0,
        allow_writing_files: bool = False,
    ) -> None:
        self.depth = depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.loss_function = loss_function
        self.verbose = verbose
        self.random_seed = random_seed
        self.allow_writing_files = allow_writing_files

    @classmethod
    def _record(cls, method: str, X: object, **extra: object) -> None:
        assert isinstance(X, pd.DataFrame)
        assert "category" in X.columns
        assert isinstance(X["category"].dtype, pd.CategoricalDtype)
        cls.calls.append(
            {
                "method": method,
                "columns": tuple(str(value) for value in X.columns),
                "category_dtype": str(X["category"].dtype),
                "categories": tuple(str(value) for value in X["category"].cat.categories),
                **extra,
            }
        )

    def fit(self, X: object, y: object, cat_features: object = None) -> "_RecordingCatBoost":
        self._record("fit", X, cat_features=tuple(cat_features or ()))
        self.classes_ = np.unique(np.asarray(y).ravel())
        return self

    def predict(self, X: object) -> np.ndarray:
        self._record("predict", X)
        return np.full(len(X), self.classes_[0])  # type: ignore[arg-type]

    def predict_proba(self, X: object) -> np.ndarray:
        self._record("predict_proba", X)
        return np.tile(np.asarray([[0.7, 0.3]]), (len(X), 1))  # type: ignore[arg-type]


class _RecordingLightGBM(BaseEstimator, ClassifierMixin):
    calls: list[dict[str, object]] = []

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 0,
        n_jobs: int = 1,
        verbosity: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity

    @classmethod
    def _record(cls, method: str, X: object) -> None:
        assert isinstance(X, pd.DataFrame)
        assert "category" in X.columns
        assert isinstance(X["category"].dtype, pd.CategoricalDtype)
        cls.calls.append(
            {
                "method": method,
                "columns": tuple(str(value) for value in X.columns),
                "category_dtype": str(X["category"].dtype),
                "categories": tuple(str(value) for value in X["category"].cat.categories),
            }
        )

    def fit(self, X: object, y: object) -> "_RecordingLightGBM":
        self._record("fit", X)
        self.classes_ = np.unique(np.asarray(y).ravel())
        return self

    def predict(self, X: object) -> np.ndarray:
        self._record("predict", X)
        return np.full(len(X), self.classes_[0])  # type: ignore[arg-type]

    def predict_proba(self, X: object) -> np.ndarray:
        self._record("predict_proba", X)
        return np.tile(np.asarray([[0.6, 0.4]]), (len(X), 1))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("estimator", "backend_class", "backend_attr", "adapter_identity"),
    [
        (
            "catboost",
            _RecordingCatBoost,
            "CatBoostClassifier",
            "catboost_dataframe_named_cat_features_v1",
        ),
        (
            "lgbm",
            _RecordingLightGBM,
            "LGBMClassifier",
            "lightgbm_pandas_categorical_dtype_v1",
        ),
    ],
)
def test_native_stage2_routes_selected_dataframe_through_cv_final_predict_and_proba(
    monkeypatch: pytest.MonkeyPatch,
    estimator: str,
    backend_class: type[BaseEstimator],
    backend_attr: str,
    adapter_identity: str,
) -> None:
    frame, y, schema = _frame()
    calls = getattr(backend_class, "calls")
    calls.clear()
    monkeypatch.setattr(classifier_backends, backend_attr, backend_class)

    result = DistributionFeatureSelectionPipeline(_native_config(estimator)).run_pre_split(
        frame.iloc[:24].copy(),
        y[:24],
        frame.iloc[24:].copy(),
        y[24:],
        schema=schema,
        seed=17,
        capture_diagnostics=True,
    )

    route = result.typed_preprocessing["preprocessor"]["native_categorical_route"]
    expected_columns = tuple(route["selected_native_columns"])
    expected_categories = tuple(route["selected_category_vocabularies"]["category"])
    assert route["status"] == "final_fit_predict_proba_complete"
    assert route["adapter_identity"] == adapter_identity
    assert route["selected_numeric_columns"] == route["selected_native_columns"]
    assert route["selected_categorical_columns"] == ["category"]
    assert any(call["method"] == "fit" for call in calls)
    assert any(call["method"] == "predict" for call in calls)
    assert any(call["method"] == "predict_proba" for call in calls)
    assert all(call["columns"] == expected_columns for call in calls)
    assert all(call["categories"] == expected_categories for call in calls)
    if estimator == "catboost":
        assert all(
            call["cat_features"] == ("category",)
            for call in calls
            if call["method"] == "fit"
        )


def test_native_stage2_fails_before_fit_for_mixed_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, y, schema = _frame()
    _RecordingCatBoost.calls.clear()
    monkeypatch.setattr(classifier_backends, "CatBoostClassifier", _RecordingCatBoost)
    config = _native_config("catboost")
    config.classification.model_candidates = ("catboost", "lr")

    with pytest.raises(TypedInputCapabilityError) as raised:
        DistributionFeatureSelectionPipeline(config).run_pre_split(
            frame.iloc[:24], y[:24], frame.iloc[24:], y[24:], schema=schema
        )

    assert raised.value.code == "native_stage2_mixed_candidates"
    assert not _RecordingCatBoost.calls


def test_native_stage2_fails_before_fit_when_dependency_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, y, schema = _frame()
    monkeypatch.setattr(classifier_backends, "CatBoostClassifier", None)

    with pytest.raises(TypedInputCapabilityError) as raised:
        DistributionFeatureSelectionPipeline(_native_config("catboost")).run_pre_split(
            frame.iloc[:24], y[:24], frame.iloc[24:], y[24:], schema=schema
        )

    assert raised.value.code == "native_stage2_dependency_unavailable"


def test_native_stage2_rejects_sparse_input_before_numeric_adapter() -> None:
    config = _native_config("catboost")
    matrix = sparse.csr_matrix(np.eye(8, dtype=float))
    y = np.asarray([0, 1] * 4)

    with pytest.raises(TypedInputCapabilityError) as raised:
        DistributionFeatureSelectionPipeline(config).run_pre_split(
            matrix[:6], y[:6], matrix[6:], y[6:]
        )

    assert raised.value.code == "native_stage2_sparse_unavailable"


def test_native_stage2_bridge_rejects_duplicate_out_of_range_and_reordered_views() -> None:
    frame, _, schema = _frame(8)
    preprocessor = FoldLocalPreprocessor().fit(frame, schema=schema)
    bridge = preprocessor.native_stage2_bridge()
    native = preprocessor.transform_with_schema(
        frame,
        schema=schema,
        output_mode="native_categorical",
    )

    with pytest.raises(TypedInputCapabilityError) as duplicate:
        preprocessor.select_native_stage2_view(
            native,
            bridge=bridge,
            selected_numeric_positions=(0, 0),
        )
    assert duplicate.value.code == "native_stage2_selection_duplicate"

    with pytest.raises(TypedInputCapabilityError) as out_of_range:
        preprocessor.select_native_stage2_view(
            native,
            bridge=bridge,
            selected_numeric_positions=(99,),
        )
    assert out_of_range.value.code == "native_stage2_selection_out_of_range"

    reordered = TransformedInput(
        X=native.X.loc[:, list(reversed(native.X.columns))],
        schema=native.schema,
        source_schema=native.source_schema,
        output_mode=native.output_mode,
        metadata=native.metadata,
    )
    with pytest.raises(TypedInputCapabilityError) as order:
        preprocessor.select_native_stage2_view(
            reordered,
            bridge=bridge,
            selected_numeric_positions=(0, 1),
        )
    assert order.value.code == "native_stage2_view_column_order_mismatch"


def test_native_stage2_bridge_rejects_wrong_source_schema() -> None:
    frame, _, schema = _frame(8)
    preprocessor = FoldLocalPreprocessor().fit(frame, schema=schema)
    bridge = preprocessor.native_stage2_bridge()
    native = preprocessor.transform_with_schema(
        frame,
        schema=schema,
        output_mode="native_categorical",
    )
    wrong_source_schema = DatasetSchema.from_dataframe(
        frame,
        roles={
            "number": FeatureRole.CONTINUOUS,
            "category": FeatureRole.CATEGORICAL,
        },
        metadata={"schema_variant": "wrong_source_for_native_bridge_test"},
    )
    assert wrong_source_schema.fingerprint != schema.fingerprint
    wrong_source = TransformedInput(
        X=native.X,
        schema=native.schema,
        source_schema=wrong_source_schema,
        output_mode=native.output_mode,
        metadata=native.metadata,
    )

    with pytest.raises(TypedInputCapabilityError) as mismatch:
        preprocessor.select_native_stage2_view(
            wrong_source,
            bridge=bridge,
            selected_numeric_positions=(0, 1),
        )

    assert mismatch.value.code == "native_stage2_view_source_schema_mismatch"


@pytest.mark.parametrize(
    ("estimator", "backend_class", "backend_attr"),
    [
        ("catboost", _RecordingCatBoost, "CatBoostClassifier"),
        ("lgbm", _RecordingLightGBM, "LGBMClassifier"),
    ],
)
def test_native_stage2_cv_fits_category_vocabularies_on_fold_train_only(
    monkeypatch: pytest.MonkeyPatch,
    estimator: str,
    backend_class: type[BaseEstimator],
    backend_attr: str,
) -> None:
    outer_train_rows = 24
    frame = pd.DataFrame(
        {
            "number": np.zeros(28, dtype=float),
            "category": [f"category_{index:02d}" for index in range(28)],
        }
    )
    y = np.asarray([0] * 12 + [1] * 12 + [0, 1, 0, 1])
    schema = DatasetSchema.from_dataframe(
        frame,
        roles={
            "number": FeatureRole.CONTINUOUS,
            "category": FeatureRole.CATEGORICAL,
        },
    )
    calls = getattr(backend_class, "calls")
    calls.clear()
    monkeypatch.setattr(classifier_backends, backend_attr, backend_class)

    seed = 17
    result = DistributionFeatureSelectionPipeline(_native_config(estimator)).run_pre_split(
        frame.iloc[:outer_train_rows].copy(),
        y[:outer_train_rows],
        frame.iloc[outer_train_rows:].copy(),
        y[outer_train_rows:],
        schema=schema,
        seed=seed,
        capture_diagnostics=True,
    )

    model_seed = int(
        np.random.SeedSequence(seed).spawn(5)[3].generate_state(
            1, dtype=np.uint32
        )[0]
    )
    fold_splits = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=model_seed).split(
            frame.iloc[:outer_train_rows], y[:outer_train_rows]
        )
    )
    sentinel_categories = ("__tabnetics_unknown__", "__tabnetics_missing__")
    expected_fold_vocabularies = [
        tuple(
            sorted(
                f"builtins.str:{value!r}"
                for value in frame.iloc[train_idx]["category"].tolist()
            )
        )
        + sentinel_categories
        for train_idx, _ in fold_splits
    ]
    expected_final_vocabulary = tuple(
        sorted(
            f"builtins.str:{value!r}"
            for value in frame.iloc[:outer_train_rows]["category"].tolist()
        )
    ) + sentinel_categories
    fit_vocabularies = [
        tuple(call["categories"])
        for call in calls
        if call["method"] == "fit"
    ]

    assert len(fit_vocabularies) == 6
    assert fit_vocabularies[:-1] == expected_fold_vocabularies
    assert fit_vocabularies[-1] == expected_final_vocabulary
    assert all(
        set(vocabulary) < set(expected_final_vocabulary)
        for vocabulary in fit_vocabularies[:-1]
    )
    cv_route = result.typed_preprocessing["preprocessor"][
        "native_categorical_route"
    ]
    assert cv_route["cv_preprocessor_scope"] == "fold_local"
    assert [
        int(record["preprocessor_fit_rows"])
        for record in cv_route["cv_fold_views"]
    ] == [len(train_idx) for train_idx, _ in fold_splits]

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone

from tabnetics.datasets.schema import (
    DatasetSchema,
    FeatureAnnotation,
    FeatureLineage,
    FeatureRole,
    FeatureSpec,
    SchemaContractError,
)
from tabnetics.feature_selection.registry import METHOD_REGISTRY
from tabnetics.pipeline.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline
from tabnetics.pipeline.preprocessing import (
    FeatureSelectorRuntimeFacts,
    FoldLocalPreprocessor,
    TypedInputCapabilityError,
    guarded_sparse_to_dense,
    is_sparse_input,
    resolve_feature_selector_capabilities,
)


def _typed_frame(rows: int = 30) -> pd.DataFrame:
    half = rows // 2
    return pd.DataFrame(
        {
            "number": np.r_[np.linspace(0.0, 1.0, half), np.linspace(1.0, 2.0, rows - half)],
            "category": ["a", "b"] * half + (["a"] if rows % 2 else []),
            "text": ["alpha common"] * half + ["beta common"] * (rows - half),
            "when": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "group": ["site-a"] * rows,
        }
    )


def _typed_schema(frame: pd.DataFrame) -> DatasetSchema:
    return DatasetSchema.from_dataframe(
        frame,
        roles={
            "number": FeatureRole.CONTINUOUS,
            "category": FeatureRole.CATEGORICAL,
            "text": FeatureRole.TEXT,
            "when": FeatureRole.TIME,
            "group": FeatureRole.GROUP,
        },
        annotations={
            "number": [
                FeatureAnnotation(
                    source="unit-test",
                    version="1",
                    identifier="numeric-panel",
                    source_hash="source-sha",
                    version_hash="version-sha",
                )
            ]
        },
    )


def test_dataset_schema_is_hashable_round_trippable_and_order_bound() -> None:
    frame = _typed_frame(6)
    schema = _typed_schema(frame)

    assert isinstance(hash(schema), int)
    assert DatasetSchema.from_record(schema.to_record()).fingerprint == schema.fingerprint
    assert schema.select((0, 1)).feature_names == ("number", "category")
    with pytest.raises(TypeError):
        schema.metadata_dict["unexpected"] = "mutation"  # type: ignore[index]
    with pytest.raises(SchemaContractError, match="columns do not match"):
        schema.validate_input(frame.loc[:, list(reversed(frame.columns))])
    with pytest.raises(SchemaContractError, match="outputs must match"):
        DatasetSchema(
            features=(FeatureSpec(name="number", role=FeatureRole.CONTINUOUS),),
            lineage=(
                FeatureLineage(
                    output_name="number",
                    operation="identity",
                    input_names=("number",),
                ),
                FeatureLineage(
                    output_name="ghost",
                    operation="identity",
                    input_names=("number",),
                ),
            ),
        )


def test_fold_local_preprocessor_handles_unseen_categories_and_pickle() -> None:
    train = _typed_frame(6)
    test = pd.DataFrame(
        {
            "number": [np.nan, 3.0],
            "category": ["not-seen-during-fit", None],
            "text": ["unseen token", None],
            "when": ["2020-02-01", None],
            "group": ["site-b", "site-c"],
        }
    )
    schema = _typed_schema(train)
    preprocessor = FoldLocalPreprocessor(text_hash_buckets=4).fit(train, schema=schema)

    transformed = preprocessor.transform_with_schema(test, schema=schema)
    category_index = list(preprocessor.get_feature_names_out()).index("category")
    assert transformed.X.shape[1] == len(preprocessor.get_feature_names_out())
    assert transformed.X[0, category_index] == pytest.approx(0.0)
    assert transformed.X[1, category_index] == pytest.approx(-1.0)
    assert np.isfinite(transformed.X).all()

    native = preprocessor.transform_with_schema(
        test,
        schema=schema,
        output_mode="native_categorical",
    )
    assert native.output_mode == "native_categorical"
    assert str(native.X.loc[native.X.index[0], "category"]) == "__tabnetics_unknown__"
    assert str(native.X.loc[native.X.index[1], "category"]) == "__tabnetics_missing__"
    with pytest.raises(TypedInputCapabilityError) as unavailable:
        preprocessor.transform_for_classifier(
            test,
            classifier_name="catboost",
            dependency_facts={"catboost": False},
        )
    assert unavailable.value.code == "native_categorical_classifier_unavailable"
    with pytest.raises(TypedInputCapabilityError) as unknown:
        preprocessor.transform_for_classifier(test, classifier_name="not-a-classifier")
    assert unknown.value.code == "native_categorical_classifier_unknown"

    lineage = {
        record.output_name: record
        for record in preprocessor.get_output_schema().lineage
    }
    assert lineage["text__text_len"].operation == "text_character_length"
    assert dict(lineage["text__text_len"].parameters_dict) == {}
    assert lineage["text__text_hash"].operation == "text_value_hash"
    assert lineage["text__text_hash"].parameters_dict["hash_buckets"] == 1024
    assert lineage["text__tfidf_hash_00"].operation == "text_tfidf_hash_train_only"
    assert lineage["text__tfidf_hash_00"].parameters_dict["hash_buckets"] == 4
    text_hash_index = list(preprocessor.get_feature_names_out()).index("text__text_hash")
    assert 0.0 <= transformed.X[0, text_hash_index] < 1024.0

    restored = pickle.loads(pickle.dumps(preprocessor))
    np.testing.assert_allclose(restored.transform(test), transformed.X)
    assert clone(preprocessor).get_params()["text_hash_buckets"] == 4


def test_sparse_typed_preprocessing_preserves_sparse_or_fails_before_dense_allocation() -> None:
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1.0, np.nan, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 4.0, 1.0, 0.0],
                [0.0, 6.0, 0.0, 1.0],
            ]
        )
    )
    preprocessor = FoldLocalPreprocessor().fit(matrix)
    transformed = preprocessor.transform_with_schema(matrix)

    assert sparse.issparse(transformed.X)
    assert transformed.X.dtype == np.float64
    assert all(feature.dtype == "float64" for feature in transformed.schema.features)
    assert np.isfinite(transformed.X.data).all()
    with pytest.raises(TypedInputCapabilityError, match="sparse-to-dense"):
        guarded_sparse_to_dense(matrix, max_elements=3, callsite="test")
    dense = guarded_sparse_to_dense(matrix, max_elements=16, callsite="test")
    np.testing.assert_allclose(dense, matrix.toarray())

    pandas_sparse = pd.DataFrame.sparse.from_spmatrix(
        sparse.eye(4, format="csr"),
        columns=["a", "b", "c", "d"],
    )
    assert is_sparse_input(pandas_sparse)
    pandas_preprocessor = FoldLocalPreprocessor().fit(pandas_sparse)
    pandas_transformed = pandas_preprocessor.transform_with_schema(pandas_sparse)
    assert sparse.issparse(pandas_transformed.X)
    assert pandas_preprocessor.input_schema_.metadata_dict["input_kind"] == "pandas_sparse_dataframe"

    pipeline = DistributionFeatureSelectionPipeline(
        DFFSConfig(typed_input_enabled=True, typed_sparse_dense_max_elements=0)
    )
    with pytest.raises(TypedInputCapabilityError) as unsafe:
        pipeline._prepare_typed_train_test_inputs(
            X_train=pandas_sparse.iloc[:3],
            X_test=pandas_sparse.iloc[3:],
            schema=None,
        )
    assert unsafe.value.code == "sparse_to_dense_unsafe"


def test_non_numeric_ndarray_requires_explicit_roles_before_numeric_coercion() -> None:
    matrix = np.asarray([["a", "1"], ["b", "2"]], dtype=object)

    with pytest.raises(SchemaContractError, match="explicit role for every feature"):
        FoldLocalPreprocessor().fit(matrix)
    with pytest.raises(SchemaContractError, match="missing roles"):
        DatasetSchema.from_input(matrix, roles={"x0": FeatureRole.CATEGORICAL})

    schema = DatasetSchema.from_input(
        matrix,
        roles={"x0": FeatureRole.CATEGORICAL, "x1": FeatureRole.COUNT},
    )
    transformed = FoldLocalPreprocessor().fit_transform_with_schema(
        matrix,
        schema=schema,
    )
    assert np.any(transformed.X[:, 0] > 0.0)
    np.testing.assert_allclose(transformed.X[:, 1], np.asarray([1.0, 2.0]))


def test_time_and_group_roles_are_excluded_from_predictor_views() -> None:
    train = _typed_frame(6)
    schema = _typed_schema(train)
    preprocessor = FoldLocalPreprocessor().fit(train, schema=schema)
    reference = preprocessor.transform(train)
    shifted_time = train.copy()
    shifted_time["when"] = pd.date_range("2099-01-01", periods=len(train), freq="D")
    shifted = preprocessor.transform(shifted_time)

    assert "when" in preprocessor.excluded_features_
    assert "group" in preprocessor.excluded_features_
    assert all(not name.startswith("when") for name in preprocessor.get_feature_names_out())
    np.testing.assert_allclose(shifted, reference)


def test_feature_selector_capabilities_cover_every_registered_method() -> None:
    runtime = FeatureSelectorRuntimeFacts(
        input_has_categorical=True,
        input_has_missing=True,
        fold_local_adapter="numeric",
    )
    records = [
        resolve_feature_selector_capabilities(method, runtime=runtime)
        for method in METHOD_REGISTRY
    ]

    assert len(records) == len(METHOD_REGISTRY)
    assert all(record.categorical_input.value == "supported" for record in records)
    assert all(record.missing_values.value == "supported" for record in records)
    sparse_native = resolve_feature_selector_capabilities(
        "anova_f",
        runtime=FeatureSelectorRuntimeFacts(
            input_is_sparse=True,
            fold_local_adapter="sparse_native",
        ),
    )
    assert sparse_native.availability.value == "unsupported"
    assert "sparse_input:sparse_native_unimplemented" in sparse_native.availability_reasons


def test_pipeline_runs_typed_input_fold_locally_and_retains_lineage() -> None:
    frame = _typed_frame(30)
    y = np.asarray([0] * 15 + [1] * 15)
    schema = _typed_schema(frame)
    config = DFFSConfig(
        typed_input_enabled=True,
        enabled_methods=("anova_f",),
        n_final_features=3,
        fs_fraction=1.0,
        prefilter_top_k=24,
        prefilter_variance_floor_enabled=False,
        folding_method="none",
        apply_cdf_transform=False,
    )

    result = DistributionFeatureSelectionPipeline(config).run(
        frame,
        y,
        dataset_name="typed-input-unit",
        schema=schema,
        seed=7,
        capture_artifacts=True,
        capture_diagnostics=True,
    )

    assert result.n_features_total == schema.n_features
    assert result.input_schema["fingerprint"] == schema.fingerprint
    assert result.model_input_schema["fingerprint"] != schema.fingerprint
    assert result.selected_feature_indices_original == tuple()
    assert result.selected_feature_schema["fingerprint"]
    assert result.typed_preprocessing["preprocessor"]["fit_rows"] == result.n_train
    assert result.config_snapshot["typed_feature_selector_admission"]["admitted_methods"] == ["anova_f"]
    assert result.run_diagnostics["typed_input"]["input_schema"]["fingerprint"] == schema.fingerprint
    assert result.model_bundle["artifact_error"] == "TypedInputCapabilityError"
    native_route = result.typed_preprocessing["preprocessor"]["native_categorical_route"]
    assert native_route["available"] is False
    assert native_route["status"] == "disabled_by_config"
    assert native_route["reason"] == "native_categorical_stage2_disabled"

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from tabnetics.datasets.schema import (
    DatasetSchema,
    FeatureRole,
    FeatureSpec,
    SchemaAlignmentMode,
    SchemaContractError,
)


def _typed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "continuous": pd.Series([1.0, 2.5, 3.0], dtype="float64"),
            "category": pd.Series(["low", "high", "low"], dtype="category"),
            "text": pd.Series(["alpha", "beta", "gamma"], dtype="string"),
            "count": pd.Series([1, 2, 3], dtype="Int64"),
            "flag": pd.Series([True, False, True], dtype="boolean"),
        }
    )


def _typed_schema(frame: pd.DataFrame) -> DatasetSchema:
    return DatasetSchema.from_dataframe(
        frame,
        roles={
            "continuous": FeatureRole.CONTINUOUS,
            "category": FeatureRole.CATEGORICAL,
            "text": FeatureRole.TEXT,
            "count": FeatureRole.COUNT,
            "flag": FeatureRole.BINARY,
        },
    )


@pytest.mark.parametrize(
    ("build_input", "error_fragment"),
    [
        (lambda frame: frame.drop(columns="category"), "missing=['category']"),
        (
            lambda frame: frame.assign(unexpected=np.arange(len(frame), dtype=float)),
            "unexpected=['unexpected']",
        ),
        (
            lambda frame: frame.loc[:, list(reversed(frame.columns))],
            "columns are reordered",
        ),
    ],
)
def test_inference_schema_strict_rejects_missing_extra_and_reordered_columns(
    build_input: object,
    error_fragment: str,
) -> None:
    frame = _typed_frame()
    schema = _typed_schema(frame)

    with pytest.raises(SchemaContractError, match=re.escape(error_fragment)):
        schema.validate_inference_input(build_input(frame))  # type: ignore[operator]


@pytest.mark.parametrize(
    ("column", "replacement", "expected_dtype", "received_dtype"),
    [
        (
            "category",
            lambda frame: frame["category"].astype(object),
            "category",
            "object",
        ),
        (
            "continuous",
            lambda frame: frame["continuous"].astype("category"),
            "float64",
            "category",
        ),
        (
            "text",
            lambda frame: pd.Series([1, 2, 3], dtype="Int64"),
            "string",
            "Int64",
        ),
    ],
)
def test_inference_schema_strict_rejects_semantic_dtype_and_role_drift(
    column: str,
    replacement: object,
    expected_dtype: str,
    received_dtype: str,
) -> None:
    frame = _typed_frame()
    schema = _typed_schema(frame)
    drifted = frame.copy()
    drifted[column] = replacement(frame)  # type: ignore[operator]

    with pytest.raises(
        SchemaContractError,
        match=(
            rf"feature '{column}'.*stored_dtype='{expected_dtype}'.*"
            rf"dtype='{received_dtype}'"
        ),
    ):
        schema.validate_inference_input(drifted)


def test_inference_schema_reorder_alignment_is_explicit_and_recorded() -> None:
    frame = _typed_frame()
    schema = _typed_schema(frame)
    reordered = frame.loc[:, ["flag", "category", "text", "continuous", "count"]]

    aligned, report = schema.align_inference_input(
        reordered,
        alignment_mode=SchemaAlignmentMode.REORDER,
    )

    assert tuple(reordered.columns) == ("flag", "category", "text", "continuous", "count")
    assert tuple(aligned.columns) == schema.feature_names
    assert report.alignment_applied is True
    assert report.typed_semantics_verified is True
    assert report.received_feature_names == tuple(reordered.columns)
    assert report.output_feature_names == schema.feature_names
    assert report.to_record()["alignment_applied"] is True
    assert report.to_record()["alignment_mode"] == "reorder"

    with pytest.raises(SchemaContractError, match="never fills or drops"):
        schema.align_inference_input(
            reordered.drop(columns="category"),
            alignment_mode="reorder",
        )


def test_legacy_positional_numeric_and_sparse_inputs_remain_untyped() -> None:
    matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    schema = DatasetSchema.from_input(matrix)

    checked_matrix, matrix_report = schema.align_inference_input(matrix)
    checked_sparse, sparse_report = schema.align_inference_input(sparse.csr_matrix(matrix))

    assert checked_matrix is matrix
    assert sparse.issparse(checked_sparse)
    assert matrix_report.input_kind == "ndarray"
    assert sparse_report.input_kind == "sparse"
    assert matrix_report.typed_semantics_verified is False
    assert sparse_report.typed_semantics_verified is False
    assert matrix_report.output_feature_names == ("x0", "x1")

    with pytest.raises(SchemaContractError, match="expected=2, received=3"):
        schema.validate_inference_input(np.ones((2, 3), dtype=float))


def test_typed_schema_rejects_positional_input_that_cannot_verify_semantics() -> None:
    frame = _typed_frame()
    schema = _typed_schema(frame)

    with pytest.raises(SchemaContractError, match="cannot establish typed semantic equivalence"):
        schema.validate_inference_input(frame.to_numpy())

    manual_typed_schema = DatasetSchema(
        features=(
            FeatureSpec(
                name="category",
                role=FeatureRole.CATEGORICAL,
                dtype="category",
            ),
        )
    )
    with pytest.raises(SchemaContractError, match="cannot establish typed semantic equivalence"):
        manual_typed_schema.validate_inference_input(np.ones((2, 1), dtype=float))

    with pytest.raises(SchemaContractError, match="only available for pandas DataFrame"):
        DatasetSchema.from_input(np.ones((2, 2))).validate_inference_input(
            np.ones((2, 2)),
            alignment_mode="reorder",
        )

import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
import scipy.stats as sps
from sklearn.base import clone

from tabnetics.datasets.schema import SchemaContractError
from tabnetics.distribution import CDFTransformError, FittedCDFTransformer


def test_parametric_family_is_replayed_on_literal_probability_scale():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "normal": rng.normal(loc=2.0, scale=1.5, size=180),
            "shifted": rng.normal(loc=-1.0, scale=0.4, size=180),
        }
    )
    transformer = FittedCDFTransformer(
        distributions={"norm": sps.norm},
        random_state=7,
    ).fit(frame)

    transformed = transformer.transform(frame.iloc[:12])

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.index.equals(frame.index[:12])
    assert tuple(transformed.columns) == tuple(frame.columns)
    assert np.all((transformed.to_numpy() > 0.0) & (transformed.to_numpy() < 1.0))
    assert {model.fit_mode for model in transformer.feature_models_} == {"parametric"}
    first = transformer.feature_models_[0]
    expected = sps.norm.cdf(frame["normal"].iloc[:12].to_numpy(), *first.params)
    np.testing.assert_allclose(transformed["normal"], expected)
    assert transformer.get_output_schema().feature_names == ("normal", "shifted")
    assert (
        transformer.get_output_schema().lineage[0].operation
        == "literal_probability_cdf"
    )


def test_empirical_mid_cdf_handles_ties_constants_and_support_clipping():
    tied = np.repeat([0.0, 1.0, 2.0], [4, 4, 4]).reshape(-1, 1)
    transformer = FittedCDFTransformer(
        distributions={"norm": sps.norm},
        min_gof_p=1.0,
        clip=(1e-4, 1.0 - 1e-4),
    ).fit(tied)
    assert transformer.feature_models_[0].fit_mode == "empirical"

    replay = transformer.transform(np.array([[-100.0], [0.0], [1.0], [2.0], [100.0]]))
    np.testing.assert_allclose(
        replay[:, 0],
        np.array([1e-4, 1.0 / 6.0, 0.5, 5.0 / 6.0, 1.0 - 1e-4]),
    )

    constant = FittedCDFTransformer().fit(np.full((12, 1), 7.0))
    assert constant.feature_models_[0].fallback_reason == "constant_feature"
    assert constant.transform(np.array([[7.0]]))[0, 0] == pytest.approx(0.5)

    low_unique = FittedCDFTransformer(min_parametric_unique=4).fit(tied)
    assert low_unique.feature_models_[0].fit_mode == "empirical"
    assert (
        low_unique.feature_models_[0].fallback_reason
        == "insufficient_unique_for_parametric"
    )


def test_missing_values_propagate_and_infinities_fail_closed():
    values = np.arange(12, dtype=float).reshape(-1, 1)
    values[2, 0] = np.nan
    transformer = FittedCDFTransformer(
        distributions={"norm": sps.norm}, missing_policy="propagate"
    ).fit(values)
    assert transformer.feature_models_[0].n_total == 12
    assert transformer.feature_models_[0].n_observed == 11
    assert transformer.feature_models_[0].n_missing == 1
    assert transformer.provenance_["resolved_config"]["missing_policy"] == "propagate"
    replay = transformer.transform(np.array([[np.nan], [3.0]]))
    assert np.isnan(replay[0, 0])
    assert np.isfinite(replay[1, 0])

    with pytest.raises(CDFTransformError, match="Infinite"):
        transformer.transform(np.array([[np.inf]]))
    with pytest.raises(CDFTransformError, match="missing_policy"):
        FittedCDFTransformer(missing_policy="raise").fit(values)


def test_dataframe_lineage_is_strict_and_cross_kind_inference_fails():
    frame = pd.DataFrame({"a": np.arange(12.0), "b": np.arange(12.0) ** 2})
    transformer = FittedCDFTransformer(distributions={"norm": sps.norm}).fit(frame)

    with pytest.raises(SchemaContractError):
        transformer.transform(frame[["b", "a"]])
    with pytest.raises(SchemaContractError, match="requires a DataFrame"):
        transformer.transform(frame.to_numpy())

    positional = FittedCDFTransformer(distributions={"norm": sps.norm}).fit(
        frame.to_numpy()
    )
    with pytest.raises(SchemaContractError, match="rejects named DataFrame"):
        positional.transform(frame)


def test_transform_is_training_state_only_and_pickle_stable():
    training = np.linspace(-2.0, 2.0, 40).reshape(-1, 1)
    transformer = FittedCDFTransformer(
        distributions={"norm": sps.norm},
        min_gof_p=1.0,
    ).fit(training)
    before = transformer.transform(np.array([[0.0], [1000.0]]))
    _ = transformer.transform(np.linspace(-1e6, 1e6, 200).reshape(-1, 1))
    after = transformer.transform(np.array([[0.0], [1000.0]]))
    np.testing.assert_array_equal(before, after)

    restored = pickle.loads(pickle.dumps(transformer))
    np.testing.assert_array_equal(
        before, restored.transform(np.array([[0.0], [1000.0]]))
    )
    assert restored.provenance_ == transformer.provenance_


def test_clone_joblib_and_resolved_selector_configuration(tmp_path):
    frame = pd.DataFrame(
        {
            "observed": np.linspace(-2.0, 2.0, 40),
            "partly_missing": np.r_[np.linspace(0.0, 1.0, 37), [np.nan] * 3],
        }
    )
    transformer = FittedCDFTransformer(
        distributions={"norm": sps.norm},
        random_state=29,
        use_lrt=False,
        use_cv=False,
        use_lmoment_prescreen=True,
        lmoment_prescreen_max_candidates=1,
        fit_estimator="mle",
    )
    cloned = clone(transformer)
    assert cloned.get_params(deep=False)["use_cv"] is False
    assert cloned.get_params(deep=False)["lmoment_prescreen_max_candidates"] == 1

    transformer.fit(frame)
    config = transformer.provenance_["resolved_config"]
    assert config["use_lrt"] is False
    assert config["use_cv"] is False
    assert config["use_lmoment_prescreen"] is True
    assert config["lmoment_prescreen_max_candidates"] == 1
    assert config["fit_estimator"] == "mle"
    feature_records = transformer.provenance_["features"]
    assert feature_records[0]["n_total"] == 40
    assert feature_records[0]["n_observed"] == 40
    assert feature_records[0]["n_missing"] == 0
    assert feature_records[0]["n_unique"] == 40
    assert feature_records[1]["n_total"] == 40
    assert feature_records[1]["n_observed"] == 37
    assert feature_records[1]["n_missing"] == 3
    assert feature_records[1]["n_unique"] == 37

    path = tmp_path / "cdf.joblib"
    joblib.dump(transformer, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(
        transformer.transform(frame.iloc[:8]),
        restored.transform(frame.iloc[:8]),
    )


def test_parametric_support_boundary_is_clipped_and_runtime_failure_uses_empirical():
    training = np.linspace(0.0, 1.0, 80).reshape(-1, 1)
    uniform = FittedCDFTransformer(
        distributions={"uniform": sps.uniform},
        clip=(1e-5, 1.0 - 1e-5),
    ).fit(training)
    boundary = uniform.transform(np.array([[-10.0], [10.0]]))
    np.testing.assert_allclose(boundary[:, 0], [1e-5, 1.0 - 1e-5])

    normal = FittedCDFTransformer(distributions={"norm": sps.norm}).fit(
        np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    )
    assert normal.feature_models_[0].fit_mode == "parametric"

    class BrokenReplay:
        @staticmethod
        def cdf(values, *params):
            del values, params
            raise FloatingPointError("runtime replay failure")

    normal.distributions = {"norm": BrokenReplay()}
    replay = normal.transform(np.array([[0.0], [100.0]]))
    np.testing.assert_allclose(replay[:, 0], [0.5, 1.0 - 1e-8])
    assert (
        normal.last_transform_provenance_["parametric_value_fallback_counts"]["x0"] == 2
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clip": (0.0, 1.0)},
        {"criterion": "unknown"},
        {"min_gof_p": 2.0},
        {"nonfinite_policy": "coerce"},
        {"lmoment_prescreen_max_candidates": -1},
        {"fit_estimator": "unknown"},
        {"min_parametric_unique": 1},
    ],
)
def test_invalid_configuration_fails_before_fitting(kwargs):
    with pytest.raises(CDFTransformError):
        FittedCDFTransformer(**kwargs).fit(np.arange(12.0).reshape(-1, 1))

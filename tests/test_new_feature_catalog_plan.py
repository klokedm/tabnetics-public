"""Opt-in catalog smoke plan for the fitted CDF and path classifier APIs."""

import os

import numpy as np
import pytest
import scipy.stats as sps

from tabnetics.classification import ElasticNetPathClassifier
from tabnetics.datasets.registry import DATASET_REGISTRY
from tabnetics.distribution import FittedCDFTransformer
from tabnetics.validation.suite import (
    _generate_distribution_cases,
    load_feature_selection_dataset,
)

CATALOG_PLAN = {
    "balanced_low_dimensional": "df_synthetic_parametric",
    "imbalanced": "lung_gordon",
    "hdlss": "leukemia_golub",
    "mixed_discrete_heavy": "madelon_nips03",
}


def test_new_feature_catalog_plan_references_registered_datasets():
    assert set(CATALOG_PLAN.values()).issubset(DATASET_REGISTRY)
    assert DATASET_REGISTRY["df_synthetic_parametric"].pipeline == "df"
    assert all(
        DATASET_REGISTRY[dataset_id].pipeline == "fs"
        for dataset_id in ("lung_gordon", "leukemia_golub", "madelon_nips03")
    )


@pytest.mark.beyondarena_external
@pytest.mark.skipif(
    os.environ.get("TABNETICS_RUN_NEW_FEATURE_CATALOG") != "1",
    reason="set TABNETICS_RUN_NEW_FEATURE_CATALOG=1 for the planned catalog smoke",
)
def test_new_feature_catalog_smoke():
    distribution_spec = DATASET_REGISTRY["df_synthetic_parametric"]
    case = _generate_distribution_cases(distribution_spec, seed=20260801)[0]
    cdf = FittedCDFTransformer(
        distributions={str(case.true_family): getattr(sps, str(case.true_family))},
        random_state=20260801,
    ).fit(np.asarray(case.data, dtype=float).reshape(-1, 1))
    cdf_values = cdf.transform(np.asarray(case.data[:10], dtype=float).reshape(-1, 1))
    assert np.all((cdf_values > 0.0) & (cdf_values < 1.0))

    for offset, dataset_id in enumerate(
        ("lung_gordon", "leukemia_golub", "madelon_nips03")
    ):
        loaded = load_feature_selection_dataset(
            DATASET_REGISTRY[dataset_id],
            seed=20260801 + offset,
            allow_synthetic_fallback=False,
            sample_cap=256,
            feature_cap=128,
            source_policy="real_only",
        )
        X = np.asarray(loaded.X, dtype=float)
        y = np.asarray(loaded.y).ravel()
        cdf_input = X[:, : min(8, X.shape[1])]
        cdf = FittedCDFTransformer(
            distributions={"norm": sps.norm},
            min_gof_p=1.0 if dataset_id == "madelon_nips03" else 0.0,
            random_state=20260801 + offset,
        ).fit(cdf_input)
        transformed = cdf.transform(cdf_input)
        assert transformed.shape[0] == X.shape[0]
        if dataset_id == "madelon_nips03":
            assert any(
                np.unique(cdf_input[:, index]).size < cdf_input.shape[0]
                for index in range(cdf_input.shape[1])
            )
            assert all(model.fit_mode == "empirical" for model in cdf.feature_models_)
        model = ElasticNetPathClassifier(
            C_grid=(0.01,),
            l1_ratio_grid=(0.0,),
            cv=3,
            class_weight="balanced",
            max_iter=10000,
            random_state=20260801 + offset,
        ).fit(X, y)
        probabilities = model.predict_proba(X[: min(12, X.shape[0])])
        assert probabilities.shape[1] == np.unique(y).size
        assert np.all(np.isfinite(probabilities))

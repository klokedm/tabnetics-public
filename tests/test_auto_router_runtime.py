import numpy as np

from tabnetics.auto_router import (
    AUTO_ROUTER_ARTIFACT_VERSION,
    apply_router_output,
    compute_dataset_descriptor,
    load_default_auto_router,
    predict_auto_router,
)
from tabnetics.pipeline.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline


def _toy_dataset():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(48, 16))
    y = np.asarray([0, 1, 0, 1] * 12)
    return X, y


def test_v25_auto_router_artifact_loads_and_predicts():
    X, y = _toy_dataset()
    router = load_default_auto_router()
    descriptor = compute_dataset_descriptor(X, y)

    assert router.fitted_
    assert router.feature_names
    assert descriptor["feature_vector"]["n"] == 48.0

    output = predict_auto_router(X, y)
    assert output.enabled_methods
    assert output.metadata["router_type"] == "score_expanded_router_v1"
    assert output.metadata["selected_candidate_id"]
    assert output.to_snapshot()["auto_router_version"] == AUTO_ROUTER_ARTIFACT_VERSION


def test_apply_router_output_updates_pipeline_config():
    X, y = _toy_dataset()
    output = predict_auto_router(X, y)
    cfg = DFFSConfig(auto_router_enabled=True)

    apply_router_output(cfg, output)

    assert cfg.auto_router_enabled is False
    assert tuple(output.enabled_methods) == tuple(cfg.enabled_methods)
    assert getattr(cfg.dist_config, "family_set") in {"v6", "flex"}
    assert getattr(cfg, "classification_backend") in {"sklearn", "flaml", "optuna"}
    assert cfg.auto_router_last_decision["auto_router_selected_candidate_id"]


def test_pipeline_resolves_auto_router_by_default():
    X, y = _toy_dataset()
    pipe = DistributionFeatureSelectionPipeline(DFFSConfig(n_jobs=1))

    active_config, metadata = pipe._resolve_meta_learning_runtime_config(X, y)

    assert active_config is not None
    assert active_config.auto_router_enabled is False
    assert metadata["auto_router_used"] is True
    assert metadata["auto_router_selected_candidate_id"]

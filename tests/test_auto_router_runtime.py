import hashlib
import json

import joblib
import numpy as np
import pytest

from tabnetics.auto_router.dependence import (
    DEPENDENCE_DESCRIPTOR_CAPS,
    DEPENDENCE_DESCRIPTOR_KEYS,
    DEPENDENCE_DESCRIPTOR_POLICY,
)
from tabnetics.auto_router.missingness import (
    MISSINGNESS_DESCRIPTOR_KEYS,
    MISSINGNESS_DESCRIPTOR_POLICY,
)
from tabnetics.auto_router import (
    AUTO_ROUTER_ARTIFACT_VERSION,
    DescriptorOODGate,
    ScoreExpandedRouter,
    ScoreRouterConfig,
    apply_router_output,
    compute_dataset_descriptor,
    default_artifact_path,
    load_default_auto_router,
    predict_auto_router,
)
from tabnetics.auto_router.uncertainty import (
    RouterOutcomeRow,
    fit_crossfit_router_uncertainty,
    router_descriptor_schema_sha256,
)
from tabnetics.pipeline.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline


class _LinearModel:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights


class _VectorModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, X):
        return self.values[: np.asarray(X).shape[0]]


def _toy_dataset():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(48, 16))
    y = np.asarray([0, 1, 0, 1] * 12)
    return X, y


def _dependence_descriptor_training_metadata(**updates):
    metadata = {
        "dependence_descriptor_policy": DEPENDENCE_DESCRIPTOR_POLICY,
        "dependence_descriptor_keys": list(DEPENDENCE_DESCRIPTOR_KEYS),
        "dependence_descriptor_caps": dict(DEPENDENCE_DESCRIPTOR_CAPS),
    }
    metadata.update(updates)
    return metadata


def _missingness_descriptor_training_metadata(**updates):
    metadata = {
        "missingness_descriptor_policy": MISSINGNESS_DESCRIPTOR_POLICY,
        "missingness_descriptor_keys": list(MISSINGNESS_DESCRIPTOR_KEYS),
    }
    metadata.update(updates)
    return metadata


def _toy_score_router(descriptor_ood_gate=None):
    candidates = [
        {
            "id": "default",
            "enabled_methods": ["anova_f"],
            "method_count": 1,
            "config_overrides": {"df_stage_position": "after_fs"},
            "classification_overrides": {"backend": "sklearn"},
            "action_features": {"action::wide": 0.0},
        },
        {
            "id": "wide",
            "enabled_methods": ["anova_f", "mutual_information"],
            "method_count": 2,
            "config_overrides": {"df_stage_position": "before_fs"},
            "classification_overrides": {"backend": "sklearn"},
            "action_features": {"action::wide": 1.0},
        },
    ]
    return ScoreExpandedRouter.from_components(
        feature_names=("log_n",),
        action_feature_names=("action::wide",),
        candidates=candidates,
        default_candidate_id="default",
        feature_median=(0.0,),
        feature_low=(0.0,),
        feature_high=(1.0,),
        score_models={
            "balanced_accuracy": _LinearModel([0.0, 0.80]),
            "macro_f1": _LinearModel([0.0, 0.70]),
        },
        descriptor_ood_gate=descriptor_ood_gate,
        config=ScoreRouterConfig(confidence_margin_scale=0.01),
    )


def _cost_tie_break_router(*, margin=0.0, beats=None):
    candidates = [
        {
            "id": "default",
            "enabled_methods": ["m"] * 16,
            "method_count": 16,
            "config_overrides": {},
            "classification_overrides": {},
            "action_features": {"action::score": 0.800},
        },
        {
            "id": "expensive",
            "enabled_methods": ["m"] * 35,
            "method_count": 35,
            "config_overrides": {},
            "classification_overrides": {},
            "action_features": {"action::score": 0.900},
        },
        {
            "id": "cheap",
            "enabled_methods": ["m"] * 5,
            "method_count": 5,
            "config_overrides": {},
            "classification_overrides": {},
            "action_features": {"action::score": 0.895},
        },
    ]
    score_models = {
        "balanced_accuracy": _LinearModel([1.0]),
        "macro_f1": _LinearModel([1.0]),
    }
    if beats is not None:
        score_models["beats_default"] = _VectorModel(beats)
    return ScoreExpandedRouter.from_components(
        feature_names=(),
        action_feature_names=("action::score",),
        candidates=candidates,
        default_candidate_id="default",
        feature_median=(),
        feature_low=(),
        feature_high=(),
        score_models=score_models,
        config=ScoreRouterConfig(
            confidence_margin_scale=0.01,
            cost_tie_break_margin=float(margin),
            beats_default_probability_threshold=0.5 if beats is not None else 0.0,
        ),
    )


def _toy_descriptor_ood_gate():
    return DescriptorOODGate(
        feature_names=("log_n",),
        feature_median=np.asarray([0.0], dtype=float),
        feature_scale=np.asarray([1.0], dtype=float),
        feature_low=np.asarray([0.0], dtype=float),
        feature_high=np.asarray([1.0], dtype=float),
        z_threshold=4.0,
    ).to_manifest()


def _dependence_descriptor_ood_gate():
    return DescriptorOODGate(
        feature_names=("mutual_info_mean",),
        feature_median=np.asarray([0.0], dtype=float),
        feature_scale=np.asarray([0.01], dtype=float),
        feature_low=np.asarray([0.0], dtype=float),
        feature_high=np.asarray([0.0], dtype=float),
        z_threshold=4.0,
    ).to_manifest()


def _missingness_descriptor_ood_gate():
    return DescriptorOODGate(
        feature_names=("mean_missing_fraction",),
        feature_median=np.asarray([0.0], dtype=float),
        feature_scale=np.asarray([0.01], dtype=float),
        feature_low=np.asarray([0.0], dtype=float),
        feature_high=np.asarray([0.0], dtype=float),
        z_threshold=4.0,
    ).to_manifest()


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


def test_router_artifact_is_fresh_and_bound_to_current_bytes(tmp_path):
    source = default_artifact_path()
    for filename in ("manifest.json", "score_models.joblib"):
        (tmp_path / filename).write_bytes((source / filename).read_bytes())

    first = load_default_auto_router(tmp_path)
    first.candidates_[0]["id"] = "poisoned_cache_entry"
    second = load_default_auto_router(tmp_path)

    assert second.candidates_[0]["id"] != "poisoned_cache_entry"
    first_identity = dict(second.artifact_identity_)
    assert first_identity["manifest_sha256"]
    assert first_identity["score_models_sha256"]

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_metadata"] = {
        **dict(manifest.get("training_metadata", {}) or {}),
        "phase4_same_path_replacement": True,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    replaced = load_default_auto_router(tmp_path)
    replaced_identity = dict(replaced.artifact_identity_)
    assert replaced_identity["artifact_sha256"] != first_identity["artifact_sha256"]
    assert replaced_identity["manifest_sha256"] != first_identity["manifest_sha256"]
    snapshot = replaced.predict({"feature_vector": {}}).to_snapshot()
    assert snapshot["auto_router_artifact_sha256"] == replaced_identity["artifact_sha256"]


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


def test_pipeline_keeps_auto_router_disabled_by_default():
    X, y = _toy_dataset()
    config = DFFSConfig(n_jobs=1)
    pipe = DistributionFeatureSelectionPipeline(config)

    active_config, metadata = pipe._resolve_meta_learning_runtime_config(X, y)

    assert config.auto_router_enabled is False
    assert active_config is None
    assert metadata == {}


def test_pipeline_resolves_auto_router_when_explicitly_enabled():
    X, y = _toy_dataset()
    pipe = DistributionFeatureSelectionPipeline(
        DFFSConfig(n_jobs=1, auto_router_enabled=True)
    )

    active_config, metadata = pipe._resolve_meta_learning_runtime_config(X, y)

    assert active_config is not None
    assert active_config.auto_router_enabled is False
    assert metadata["auto_router_used"] is True
    assert metadata["auto_router_selected_candidate_id"]


def test_dataset_descriptor_accepts_numeric_diakrino_dispersion_metadata():
    X, y = _toy_dataset()

    descriptor = compute_dataset_descriptor(
        X,
        y,
        metadata={
            "diakrino_router_dispersion_descriptor": {
                "summary": {
                    "chunk_mean_std": 0.12,
                    "n_chunks": 2,
                    "source": "diakrino_selection_logits",
                }
            }
        },
    )

    features = descriptor["feature_vector"]
    assert features["diakrino_dispersion_chunk_mean_std"] == 0.12
    assert features["diakrino_dispersion_n_chunks"] == 2.0
    assert "diakrino_dispersion_source" not in features


def test_dataset_descriptor_emits_bounded_dependence_descriptors():
    rng = np.random.default_rng(7)
    y = np.asarray([0, 1] * 30)
    signal = y.astype(float) + rng.normal(scale=0.03, size=y.shape[0])
    X = np.column_stack(
        [
            signal,
            signal + rng.normal(scale=0.01, size=y.shape[0]),
            rng.normal(size=y.shape[0]),
            rng.normal(size=y.shape[0]),
        ]
    )

    features = compute_dataset_descriptor(X, y)["feature_vector"]

    assert features["mutual_info_mean"] > 0.0
    assert features["mutual_info_std"] >= 0.0
    assert 0.0 <= features["pairwise_redundancy_ratio"] <= 1.0


def test_dataset_descriptor_rejects_non_1d_labels():
    y = np.asarray([0, 1] * 20)
    X = np.column_stack([y.astype(float), 1.0 - y.astype(float)])

    valid = compute_dataset_descriptor(X, y)["feature_vector"]

    assert valid["mutual_info_mean"] > 0.0
    with pytest.raises(ValueError, match="auto-router descriptor labels must be 1D"):
        compute_dataset_descriptor(X, y.reshape(-1, 1))


def test_dataset_descriptor_rejects_xy_row_mismatch_and_1d_x():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.asarray([0, 1, 0, 1], dtype=int)

    with pytest.raises(ValueError, match="auto-router descriptor X and labels must have the same number of rows"):
        compute_dataset_descriptor(X, y)
    with pytest.raises(ValueError, match="auto-router descriptor X must be 2D"):
        compute_dataset_descriptor(np.arange(6, dtype=float), np.asarray([0, 1, 0, 1, 0, 1]))


def test_dataset_descriptor_preserves_support_missingness_patterns():
    X = np.asarray(
        [
            [1.0, 2.0, np.nan, np.nan],
            [1.0, np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0, 4.0],
            [np.nan, 2.0, np.nan, np.nan],
            [1.0, 2.0, 3.0, np.nan],
        ]
    )
    y = np.asarray([0, 1, 0, 1, 0])

    features = compute_dataset_descriptor(X, y)["feature_vector"]

    assert features["mean_missing_fraction"] == pytest.approx(9.0 / 20.0)
    assert features["missingness_column_concentration"] == pytest.approx(1.0 / 9.0)
    assert features["missingness_monotone_flag"] == 0.0

    permuted = compute_dataset_descriptor(X[[3, 1, 4, 0, 2]][:, [2, 0, 3, 1]], y[[3, 1, 4, 0, 2]])[
        "feature_vector"
    ]
    for key in MISSINGNESS_DESCRIPTOR_KEYS:
        assert permuted[key] == pytest.approx(features[key])


def test_dataset_descriptor_detects_nested_monotone_missingness():
    X = np.asarray(
        [
            [1.0, 2.0, 3.0, np.nan],
            [1.0, 2.0, np.nan, np.nan],
            [1.0, np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0, 4.0],
        ]
    )
    y = np.asarray([0, 1, 0, 1])

    original = compute_dataset_descriptor(X, y)["feature_vector"]
    permuted = compute_dataset_descriptor(X[:, [2, 0, 3, 1]], y)["feature_vector"]

    assert original["missingness_monotone_flag"] == 1.0
    for key in MISSINGNESS_DESCRIPTOR_KEYS:
        assert permuted[key] == pytest.approx(original[key])


def test_score_router_requires_exact_missingness_artifact_contract():
    kwargs = dict(
        feature_names=MISSINGNESS_DESCRIPTOR_KEYS,
        action_feature_names=(),
        candidates=[
            {
                "id": "default",
                "enabled_methods": ["anova_f"],
                "method_count": 1,
                "config_overrides": {},
                "classification_overrides": {},
                "action_features": {},
            }
        ],
        default_candidate_id="default",
        feature_median=(0.0,) * len(MISSINGNESS_DESCRIPTOR_KEYS),
        feature_low=(0.0,) * len(MISSINGNESS_DESCRIPTOR_KEYS),
        feature_high=(1.0,) * len(MISSINGNESS_DESCRIPTOR_KEYS),
        score_models={
            "balanced_accuracy": _LinearModel(np.ones(len(MISSINGNESS_DESCRIPTOR_KEYS))),
            "macro_f1": _LinearModel(np.ones(len(MISSINGNESS_DESCRIPTOR_KEYS))),
        },
    )
    descriptor = {
        "feature_vector": dict(
            zip(MISSINGNESS_DESCRIPTOR_KEYS, (0.2, 0.3, 0.4, 1.0))
        )
    }

    legacy = ScoreExpandedRouter.from_components(**kwargs)
    incomplete = ScoreExpandedRouter.from_components(
        **kwargs,
        metadata={"missingness_descriptor_policy": MISSINGNESS_DESCRIPTOR_POLICY},
    )
    opted_in = ScoreExpandedRouter.from_components(
        **kwargs,
        metadata=_missingness_descriptor_training_metadata(),
    )

    assert np.allclose(legacy._descriptor_vector(descriptor), np.zeros((1, 3)))
    assert np.allclose(incomplete._descriptor_vector(descriptor), np.zeros((1, 3)))
    assert np.allclose(opted_in._descriptor_vector(descriptor), [[0.2, 0.3, 0.4]])
    legacy_snapshot = legacy.predict(descriptor).to_snapshot()
    opted_in_snapshot = opted_in.predict(descriptor).to_snapshot()
    assert legacy_snapshot["auto_router_missingness_descriptor_model_input_enabled"] is False
    assert (
        opted_in_snapshot["auto_router_missingness_descriptor_policy"]
        == MISSINGNESS_DESCRIPTOR_POLICY
    )
    assert opted_in_snapshot["auto_router_missingness_descriptor_model_input_enabled"] is True


def test_descriptor_ood_gate_zeroes_missingness_for_legacy_artifacts():
    kwargs = dict(
        feature_names=MISSINGNESS_DESCRIPTOR_KEYS,
        action_feature_names=(),
        candidates=[
            {
                "id": "default",
                "enabled_methods": ["anova_f"],
                "method_count": 1,
                "config_overrides": {},
                "classification_overrides": {},
                "action_features": {},
            }
        ],
        default_candidate_id="default",
        feature_median=(0.0,) * len(MISSINGNESS_DESCRIPTOR_KEYS),
        feature_low=(0.0,) * len(MISSINGNESS_DESCRIPTOR_KEYS),
        feature_high=(1.0,) * len(MISSINGNESS_DESCRIPTOR_KEYS),
        score_models={
            "balanced_accuracy": _LinearModel(np.zeros(len(MISSINGNESS_DESCRIPTOR_KEYS))),
            "macro_f1": _LinearModel(np.zeros(len(MISSINGNESS_DESCRIPTOR_KEYS))),
        },
        descriptor_ood_gate=_missingness_descriptor_ood_gate(),
        config=ScoreRouterConfig(descriptor_ood_gate_enabled=True),
    )
    descriptor = {"feature_vector": {"mean_missing_fraction": 0.25}}

    legacy = ScoreExpandedRouter.from_components(**kwargs)
    opted_in = ScoreExpandedRouter.from_components(
        **kwargs,
        metadata=_missingness_descriptor_training_metadata(),
    )

    legacy_output = legacy.predict(descriptor, descriptor_ood_gate_enabled=True)
    opted_in_output = opted_in.predict(descriptor, descriptor_ood_gate_enabled=True)

    assert legacy_output.metadata["descriptor_ood_defaulted"] is False
    assert legacy_output.metadata["descriptor_ood_outside_features"] == []
    assert opted_in_output.metadata["descriptor_ood_defaulted"] is True
    assert opted_in_output.metadata["descriptor_ood_outside_features"] == [
        "mean_missing_fraction"
    ]


def test_score_router_zeroes_dependence_descriptors_for_legacy_artifacts():
    kwargs = dict(
        feature_names=("mutual_info_mean", "mutual_info_std", "pairwise_redundancy_ratio"),
        action_feature_names=(),
        candidates=[
            {
                "id": "default",
                "enabled_methods": ["anova_f"],
                "method_count": 1,
                "config_overrides": {},
                "classification_overrides": {},
                "action_features": {},
            }
        ],
        default_candidate_id="default",
        feature_median=(0.0, 0.0, 0.0),
        feature_low=(-10.0, -10.0, -10.0),
        feature_high=(10.0, 10.0, 10.0),
        score_models={
            "balanced_accuracy": _LinearModel([1.0, 1.0, 1.0]),
            "macro_f1": _LinearModel([1.0, 1.0, 1.0]),
        },
    )
    descriptor = {
        "feature_vector": {
            "mutual_info_mean": 0.5,
            "mutual_info_std": 0.25,
            "pairwise_redundancy_ratio": 0.9,
        }
    }

    legacy = ScoreExpandedRouter.from_components(**kwargs)
    policy_only = ScoreExpandedRouter.from_components(
        **kwargs,
        metadata={"dependence_descriptor_policy": DEPENDENCE_DESCRIPTOR_POLICY},
    )
    wrong_caps = ScoreExpandedRouter.from_components(
        **kwargs,
        metadata=_dependence_descriptor_training_metadata(
            dependence_descriptor_caps={
                **dict(DEPENDENCE_DESCRIPTOR_CAPS),
                "pairwise_max_pairs": int(DEPENDENCE_DESCRIPTOR_CAPS["pairwise_max_pairs"]) + 1,
            }
        ),
    )
    opted_in = ScoreExpandedRouter.from_components(
        **kwargs,
        metadata=_dependence_descriptor_training_metadata(),
    )
    output = legacy.predict(descriptor)
    snapshot = output.to_snapshot()

    assert np.allclose(legacy._descriptor_vector(descriptor), [[0.0, 0.0, 0.0]])
    assert np.allclose(policy_only._descriptor_vector(descriptor), [[0.0, 0.0, 0.0]])
    assert np.allclose(wrong_caps._descriptor_vector(descriptor), [[0.0, 0.0, 0.0]])
    assert np.allclose(opted_in._descriptor_vector(descriptor), [[0.5, 0.25, 0.9]])
    assert output.metadata["dependence_descriptor_model_input_enabled"] is False
    assert snapshot["auto_router_dependence_descriptor_policy"] == ""
    assert snapshot["auto_router_dependence_descriptor_model_input_enabled"] is False


def test_descriptor_ood_gate_zeroes_dependence_descriptors_for_legacy_artifacts():
    kwargs = dict(
        feature_names=("mutual_info_mean", "mutual_info_std", "pairwise_redundancy_ratio"),
        action_feature_names=(),
        candidates=[
            {
                "id": "default",
                "enabled_methods": ["anova_f"],
                "method_count": 1,
                "config_overrides": {},
                "classification_overrides": {},
                "action_features": {},
            }
        ],
        default_candidate_id="default",
        feature_median=(0.0, 0.0, 0.0),
        feature_low=(-10.0, -10.0, -10.0),
        feature_high=(10.0, 10.0, 10.0),
        score_models={
            "balanced_accuracy": _LinearModel([0.0, 0.0, 0.0]),
            "macro_f1": _LinearModel([0.0, 0.0, 0.0]),
        },
        descriptor_ood_gate=_dependence_descriptor_ood_gate(),
        config=ScoreRouterConfig(descriptor_ood_gate_enabled=True),
    )
    descriptor = {"feature_vector": {"mutual_info_mean": 0.5}}

    legacy = ScoreExpandedRouter.from_components(**kwargs)
    opted_in = ScoreExpandedRouter.from_components(
        **kwargs,
        metadata=_dependence_descriptor_training_metadata(),
    )

    legacy_output = legacy.predict(descriptor, descriptor_ood_gate_enabled=True)
    opted_in_output = opted_in.predict(descriptor, descriptor_ood_gate_enabled=True)

    assert legacy_output.metadata["descriptor_ood_defaulted"] is False
    assert legacy_output.metadata["descriptor_ood_outside_features"] == []
    assert legacy_output.to_snapshot()["auto_router_dependence_descriptor_model_input_enabled"] is False
    assert opted_in_output.metadata["descriptor_ood_defaulted"] is True
    assert opted_in_output.metadata["descriptor_ood_outside_features"] == ["mutual_info_mean"]
    assert (
        opted_in_output.to_snapshot()["auto_router_dependence_descriptor_policy"]
        == DEPENDENCE_DESCRIPTOR_POLICY
    )
    assert opted_in_output.to_snapshot()["auto_router_dependence_descriptor_model_input_enabled"] is True


def test_cost_tie_break_is_default_off_and_prefers_cheapest_within_margin():
    descriptor = {"feature_vector": {}}

    disabled = _cost_tie_break_router(margin=0.0).predict(descriptor)
    enabled = _cost_tie_break_router(margin=0.01).predict(descriptor)

    assert disabled.metadata["selected_candidate_id"] == "expensive"
    assert disabled.metadata["cost_tie_break_applied"] is False
    assert enabled.metadata["selected_candidate_id"] == "cheap"
    assert enabled.metadata["cost_tie_break_applied"] is True
    assert enabled.metadata["cost_tie_break_raw_candidate_id"] == "expensive"
    assert enabled.metadata["cost_tie_break_selected_candidate_id"] == "cheap"
    ranked = {row["candidate_id"]: row for row in enabled.metadata["ranked_candidates"]}
    assert ranked["cheap"]["cost_tie_break_selected"] is True
    snapshot = enabled.to_snapshot()
    assert snapshot["auto_router_cost_tie_break_applied"] is True
    assert snapshot["auto_router_cost_tie_break_selected_candidate_id"] == "cheap"


def test_cost_tie_break_cannot_bypass_beats_default_guard():
    descriptor = {"feature_vector": {}}
    router = _cost_tie_break_router(margin=0.01, beats=[1.0, 1.0, 0.0])

    output = router.predict(descriptor)

    assert output.metadata["selected_candidate_id"] == "expensive"
    assert output.metadata["cost_tie_break_applied"] is False
    assert output.metadata["cost_tie_break_eligible_candidate_ids"] == ["expensive"]


def test_cost_tie_break_appends_selected_candidate_outside_raw_top5():
    candidates = [
        {
            "id": "default",
            "enabled_methods": ["m"] * 16,
            "method_count": 16,
            "config_overrides": {},
            "classification_overrides": {},
            "action_features": {"action::score": 0.800},
        }
    ]
    for idx, score in enumerate([0.900, 0.899, 0.898, 0.897, 0.896], start=1):
        candidates.append(
            {
                "id": f"expensive_{idx}",
                "enabled_methods": ["m"] * (30 + idx),
                "method_count": 30 + idx,
                "config_overrides": {},
                "classification_overrides": {},
                "action_features": {"action::score": score},
            }
        )
    candidates.append(
        {
            "id": "cheap",
            "enabled_methods": ["m"] * 5,
            "method_count": 5,
            "config_overrides": {},
            "classification_overrides": {},
            "action_features": {"action::score": 0.895},
        }
    )
    router = ScoreExpandedRouter.from_components(
        feature_names=(),
        action_feature_names=("action::score",),
        candidates=candidates,
        default_candidate_id="default",
        feature_median=(),
        feature_low=(),
        feature_high=(),
        score_models={
            "balanced_accuracy": _LinearModel([1.0]),
            "macro_f1": _LinearModel([1.0]),
        },
        config=ScoreRouterConfig(cost_tie_break_margin=0.01),
    )

    output = router.predict({"feature_vector": {}})

    assert output.metadata["selected_candidate_id"] == "cheap"
    ranked = output.metadata["ranked_candidates"]
    assert [row["candidate_id"] for row in ranked[:5]] == [f"expensive_{idx}" for idx in range(1, 6)]
    assert ranked[-1]["candidate_id"] == "cheap"
    assert ranked[-1]["cost_tie_break_selected"] is True


def test_descriptor_ood_gate_is_opt_in_and_uses_raw_preclip_descriptor():
    router = _toy_score_router(descriptor_ood_gate=_toy_descriptor_ood_gate())
    descriptor = {"feature_vector": {"log_n": 10.0}}

    disabled = router.predict(descriptor, descriptor_ood_gate_enabled=False)
    assert disabled.metadata["selected_candidate_id"] == "wide"
    assert disabled.metadata["auto_router_ood_defaulted"] is False
    assert disabled.metadata["descriptor_ood_gate_enabled"] is False

    enabled = router.predict(descriptor, descriptor_ood_gate_enabled=True)
    assert enabled.metadata["selected_candidate_id"] == "default"
    assert enabled.metadata["raw_selected_candidate_id"] == "wide"
    assert enabled.metadata["policy_defaulted"] is True
    assert enabled.metadata["policy_margin_defaulted"] is False
    assert enabled.metadata["auto_router_ood_defaulted"] is True
    assert enabled.confidence == 0.0
    assert enabled.metadata["descriptor_ood_score"] > 0.0
    assert enabled.metadata["descriptor_ood_max_zscore"] >= 10.0
    assert enabled.metadata["descriptor_ood_outside_features"] == ["log_n"]
    snapshot = enabled.to_snapshot()
    assert snapshot["auto_router_ood_defaulted"] is True
    assert snapshot["auto_router_raw_selected_candidate_id"] == "wide"
    assert snapshot["auto_router_descriptor_ood_gate_enabled"] is True
    assert snapshot["auto_router_descriptor_ood_max_zscore"] >= 10.0


def test_descriptor_ood_gate_missing_manifest_key_loads_as_none(tmp_path):
    router = _toy_score_router()
    manifest = {
        "artifact_type": "score_expanded_router_v1",
        "feature_names": list(router.feature_names_),
        "action_feature_names": list(router.action_feature_names_),
        "candidates": router.candidates_,
        "default_candidate_id": router.default_candidate_id_,
        "feature_median": router.feature_median_.tolist(),
        "feature_low": router.feature_low_.tolist(),
        "feature_high": router.feature_high_.tolist(),
        "config": {"confidence_margin_scale": 0.01},
        "training_metadata": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    joblib.dump(router.score_models_, tmp_path / "score_models.joblib")

    loaded = ScoreExpandedRouter.load(tmp_path)
    assert loaded.descriptor_ood_gate_ is None
    output = loaded.predict({"feature_vector": {"log_n": 10.0}}, descriptor_ood_gate_enabled=True)
    assert output.metadata["selected_candidate_id"] == "wide"
    assert output.metadata["auto_router_ood_defaulted"] is False


def test_crossfit_uncertainty_is_default_off_and_fails_closed_without_artifact():
    router = _toy_score_router(descriptor_ood_gate=_toy_descriptor_ood_gate())
    descriptor = {"feature_vector": {"log_n": 0.5}}

    legacy = router.predict(descriptor)
    guarded = router.predict(descriptor, crossfit_uncertainty_enabled=True)

    assert legacy.metadata["selected_candidate_id"] == "wide"
    assert guarded.metadata["selected_candidate_id"] == "default"
    assert guarded.metadata["crossfit_uncertainty_applied"] is False
    assert guarded.metadata["crossfit_uncertainty_fallback_reason"] == "uncertainty_artifact_unavailable"
    assert guarded.metadata["policy_defaulted"] is True


def test_crossfit_uncertainty_requires_an_enabled_descriptor_ood_gate():
    gate = _toy_descriptor_ood_gate()
    gate["enabled"] = False
    router = _toy_score_router(descriptor_ood_gate=gate)
    router.artifact_identity_ = {"artifact_sha256": hashlib.sha256(b"router").hexdigest()}
    rows = []
    for source_index, source_id in enumerate(("source_a", "source_b", "source_c", "source_d")):
        for candidate_id, delta in (("default", 0.0), ("wide", 0.10)):
            rows.append(
                RouterOutcomeRow(
                    dataset_id=f"dataset_{source_index}", source_id=source_id, seed=11,
                    split_fingerprint=hashlib.sha256(f"{source_id}-{candidate_id}".encode()).hexdigest(),
                    descriptor_sha256=hashlib.sha256(source_id.encode()).hexdigest(),
                    candidate_id=candidate_id,
                    fold_id="fold_a" if source_index % 2 else "fold_b",
                    predicted_delta_utility=delta + 0.05,
                    realized_delta_utility=delta,
                    beats_default_probability=0.90,
                )
            )
    router.crossfit_uncertainty_artifact_ = fit_crossfit_router_uncertainty(
        rows,
        base_router_sha256=router.artifact_identity_["artifact_sha256"],
        descriptor_schema_sha256=router_descriptor_schema_sha256(router.feature_names_),
        frozen_source_ids=("frozen",), minimum_support=2,
    )

    output = router.predict(
        {"feature_vector": {"log_n": 0.5}}, crossfit_uncertainty_enabled=True
    )

    assert output.metadata["selected_candidate_id"] == "default"
    assert output.metadata["crossfit_uncertainty_fallback_reason"] == "descriptor_ood_gate_unavailable_or_disabled"


def test_crossfit_uncertainty_selects_only_bound_candidate_and_binds_router_identity():
    router = _toy_score_router(descriptor_ood_gate=_toy_descriptor_ood_gate())
    router.artifact_identity_ = {"artifact_sha256": hashlib.sha256(b"router").hexdigest()}
    rows = []
    for source_index, source_id in enumerate(("source_a", "source_b", "source_c", "source_d")):
        for candidate_id, delta in (("default", 0.0), ("wide", 0.10)):
            rows.append(
                RouterOutcomeRow(
                    dataset_id=f"dataset_{source_index}", source_id=source_id, seed=11,
                    split_fingerprint=hashlib.sha256(f"{source_id}-{candidate_id}".encode()).hexdigest(),
                    descriptor_sha256=hashlib.sha256(source_id.encode()).hexdigest(),
                    candidate_id=candidate_id,
                    fold_id="fold_a" if source_index % 2 else "fold_b",
                    predicted_delta_utility=delta + 0.05,
                    realized_delta_utility=delta,
                    beats_default_probability=0.90,
                )
            )
    router.crossfit_uncertainty_artifact_ = fit_crossfit_router_uncertainty(
        rows,
        base_router_sha256=router.artifact_identity_["artifact_sha256"],
        descriptor_schema_sha256=router_descriptor_schema_sha256(router.feature_names_),
        frozen_source_ids=("frozen",), minimum_support=2,
    )

    output = router.predict(
        {"feature_vector": {"log_n": 0.5}}, crossfit_uncertainty_enabled=True
    )

    assert output.metadata["selected_candidate_id"] == "wide"
    assert output.metadata["crossfit_uncertainty_applied"] is True
    assert output.metadata["crossfit_uncertainty_fallback_reason"] == ""
    assert output.to_snapshot()["auto_router_crossfit_uncertainty_enabled"] is True

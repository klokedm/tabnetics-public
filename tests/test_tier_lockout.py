import numpy as np

from tabnetics.pipeline.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline


def _xy(seed: int = 11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(48, 12)).astype(float)
    y = np.array([0] * 24 + [1] * 24, dtype=int)
    return X, y


def _base_config(**overrides):
    cfg = DFFSConfig(
        random_seed=11,
        fs_fraction=0.5,
        n_final_features=8,
        max_dist_features=12,
        prefilter_top_k=24,
        enabled_methods=("gradient_boosting", "linear_svm", "mutual_information", "anova_f"),
        enable_maqc_pairing=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _stub_catalog_tier(monkeypatch, pipe: DistributionFeatureSelectionPipeline, tier: str):
    monkeypatch.setattr(
        pipe,
        "_resolve_dataset_catalog_context",
        lambda _dataset_name: {
            "dataset_id": "mock",
            "display_name": "mock",
            "domain": "",
            "tier": str(tier).strip().lower(),
            "is_face_domain": False,
            "found_in_catalog": True,
        },
    )


def test_tier_lockout_disabled_keeps_configured_methods(monkeypatch):
    X, y = _xy()
    pipe = DistributionFeatureSelectionPipeline(_base_config())
    _stub_catalog_tier(monkeypatch, pipe, "easy")

    policy = pipe._resolve_method_policy("mock", X, y)
    assert bool(policy["tier_policy_applied"]) is False
    assert tuple(policy["enabled_methods"]) == tuple(pipe.config.enabled_methods)


def test_tier_lockout_historical_routes_to_fallback(monkeypatch):
    X, y = _xy()
    pipe = DistributionFeatureSelectionPipeline(
        _base_config(
            tier_lockout_enabled=True,
            tier_lockout_tier="easy",
            tier_lockout_difficulty_source="historical",
            tier_lockout_fallback_methods=("linear_svm", "anova_f"),
        )
    )
    _stub_catalog_tier(monkeypatch, pipe, "easy")

    policy = pipe._resolve_method_policy("mock", X, y)
    assert bool(policy["tier_policy_applied"]) is True
    assert str(policy["tier_policy_mode"]) == "lockout"
    assert tuple(policy["enabled_methods"]) == ("linear_svm", "anova_f")
    assert str(policy["enabled_methods_source"]) == "tier_lockout:easy"


def test_tier_lockout_empty_fallback_uses_configured_methods(monkeypatch):
    X, y = _xy()
    pipe = DistributionFeatureSelectionPipeline(
        _base_config(
            tier_lockout_enabled=True,
            tier_lockout_tier="easy",
            tier_lockout_fallback_methods=tuple(),
        )
    )
    _stub_catalog_tier(monkeypatch, pipe, "easy")

    policy = pipe._resolve_method_policy("mock", X, y)
    assert bool(policy["tier_policy_applied"]) is True
    assert tuple(policy["enabled_methods"]) == tuple(pipe.config.enabled_methods)


def test_tier_lockout_meta_features_source(monkeypatch):
    X, y = _xy()
    pipe = DistributionFeatureSelectionPipeline(
        _base_config(
            tier_lockout_enabled=True,
            tier_lockout_tier="easy",
            tier_lockout_difficulty_source="meta_features",
            tier_lockout_fallback_methods=("linear_svm",),
        )
    )
    _stub_catalog_tier(monkeypatch, pipe, "hard")
    monkeypatch.setattr(pipe, "_meta_features_to_tier", lambda _meta: "easy")

    policy = pipe._resolve_method_policy("mock", X, y)
    assert bool(policy["tier_policy_applied"]) is True
    assert str(policy["tier_policy_source"]) == "meta_features"
    assert tuple(policy["enabled_methods"]) == ("linear_svm",)


def test_tier_lockout_choose_candidate_uses_lockout_stack(monkeypatch):
    X, y = _xy()
    pipe = DistributionFeatureSelectionPipeline(
        _base_config(
            tier_lockout_enabled=True,
            tier_lockout_tier="easy",
            tier_lockout_fallback_methods=("linear_svm",),
            enable_maqc_pairing=True,
            maqc_pairing_method_sets=(("gradient_boosting",),),
            maqc_pairing_method_set_names=("alt",),
        )
    )
    _stub_catalog_tier(monkeypatch, pipe, "easy")

    captured = {"methods": tuple()}

    def _fake_eval(*, enabled_methods, **kwargs):
        captured["methods"] = tuple(enabled_methods)
        return {
            "candidate_name": "configured_enabled_methods",
            "enabled_methods": tuple(enabled_methods),
            "X_train_sel": np.asarray(kwargs["X_train_full"], dtype=float),
            "X_test_sel": np.asarray(kwargs["X_test_full"], dtype=float),
            "selected_indices": tuple(range(kwargs["X_train_full"].shape[1])),
            "model": None,
            "model_name": "dummy",
            "model_cv_score": 0.0,
            "model_cv_score_std": 0.0,
            "model_cv_score_n_splits": 1,
            "model_cv_meta": {},
        }

    monkeypatch.setattr(pipe, "_evaluate_selector_candidate", _fake_eval)
    out = pipe._choose_selector_candidate(
        X_fs=X,
        y_fs=y,
        X_train_full=X,
        X_test_full=X,
        y_train_full=y,
        seed=11,
        dataset_name="mock",
    )

    assert captured["methods"] == ("linear_svm",)
    assert bool(out["tier_policy_applied"]) is True
    assert str(out["enabled_methods_source"]) == "tier_lockout:easy"

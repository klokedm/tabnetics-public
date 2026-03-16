import numpy as np

from tabnetics.pipeline.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline


def _xy(n_samples: int = 60, n_features: int = 18, seed: int = 101):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(float)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2), dtype=int)
    return X, y


def _base_config(**overrides):
    cfg = DFFSConfig(
        random_seed=17,
        fs_fraction=0.5,
        n_final_features=10,
        max_dist_features=14,
        prefilter_top_k=28,
        enabled_methods=("gradient_boosting", "linear_svm", "mutual_information", "anova_f", "mrmr_jmi"),
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


def test_regime_gating_disabled_keeps_default_policy(monkeypatch):
    X, y = _xy(n_samples=40, n_features=160, seed=11)  # n/c=20, p/n=4
    pipe = DistributionFeatureSelectionPipeline(_base_config(regime_gating_enabled=False))
    _stub_catalog_tier(monkeypatch, pipe, "very_hard")

    policy = pipe._resolve_method_policy("mock", X, y)
    assert bool(policy["regime_policy_enabled"]) is False
    assert bool(policy["regime_policy_applied"]) is False
    assert tuple(policy["enabled_methods"]) == tuple(pipe.config.enabled_methods)


def test_regime_gating_very_hard_routes_to_simple_fallback(monkeypatch):
    X, y = _xy(n_samples=60, n_features=180, seed=13)  # p/n=3, n/c=30
    pipe = DistributionFeatureSelectionPipeline(
        _base_config(
            regime_gating_enabled=True,
            regime_gating_simple_methods=("linear_svm", "anova_f"),
            regime_gating_very_hard_portfolio_max_methods=4,
            regime_gating_very_hard_copula_derandomize_runs=5,
            regime_gating_very_hard_min_classes=2,  # T-R-270: lowered from default 5 so binary data triggers gate
        )
    )
    _stub_catalog_tier(monkeypatch, pipe, "very_hard")

    policy = pipe._resolve_method_policy("mock", X, y)
    assert bool(policy["regime_policy_applied"]) is True
    assert str(policy["regime_policy_mode"]) == "very_hard_fallback"
    assert bool(policy["regime_policy_bypass_fs"]) is False
    assert tuple(policy["enabled_methods"]) == ("linear_svm", "anova_f")
    overrides = dict(policy.get("regime_policy_selector_overrides") or {})
    assert str(overrides.get("selection_strategy")) == "legacy_voting"
    assert int(overrides.get("fs_adaptive_size_max", 0)) == 2
    assert int(overrides.get("fs_copula_derandomize_runs", 0)) == 5


def test_regime_gating_low_p_over_n_bypass_takes_precedence(monkeypatch):
    X, y = _xy(n_samples=120, n_features=80, seed=19)  # p/n < 1.0
    pipe = DistributionFeatureSelectionPipeline(
        _base_config(regime_gating_enabled=True, regime_gating_low_p_over_n_threshold=2.0)
    )
    _stub_catalog_tier(monkeypatch, pipe, "very_hard")

    policy = pipe._resolve_method_policy("mock", X, y)
    assert bool(policy["regime_policy_applied"]) is True
    assert str(policy["regime_policy_mode"]) == "low_p_over_n_bypass"
    assert bool(policy["regime_policy_bypass_fs"]) is True
    assert str(policy["regime_policy_bypass_mode"]) == "fast_univariate_filter"


def test_regime_gating_choose_candidate_uses_bypass_path(monkeypatch):
    X, y = _xy(n_samples=120, n_features=80, seed=23)  # p/n < 2.0
    pipe = DistributionFeatureSelectionPipeline(
        _base_config(regime_gating_enabled=True, regime_gating_low_p_over_n_threshold=2.0)
    )
    _stub_catalog_tier(monkeypatch, pipe, "hard")

    captured = {"called": False}

    def _fake_bypass(**kwargs):
        captured["called"] = True
        return {
            "candidate_name": "configured_enabled_methods",
            "enabled_methods": tuple(kwargs["enabled_methods"]),
            "X_train_sel": np.asarray(kwargs["X_train_full"], dtype=float),
            "X_test_sel": np.asarray(kwargs["X_test_full"], dtype=float),
            "selected_indices": tuple(range(int(np.asarray(kwargs["X_train_full"]).shape[1]))),
            "model": None,
            "model_name": "dummy",
            "model_cv_score": 0.0,
            "model_cv_score_std": 0.0,
            "model_cv_score_n_splits": 1,
            "model_cv_meta": {},
            "stage2_ratio_meta": {},
            "seed_schedule": {"root_seed": int(kwargs["seed"])},
            "selector_overrides_applied": {"selection_strategy": "regime_bypass_all_features"},
            "fs_selection_summary": {},
            "fs_diagnostics": {},
            "_fitted_selector": None,
            "_selection_result": None,
        }

    monkeypatch.setattr(pipe, "_evaluate_selector_bypass_candidate", _fake_bypass)
    out = pipe._choose_selector_candidate(
        X_fs=X,
        y_fs=y,
        X_train_full=X,
        X_test_full=X,
        y_train_full=y,
        seed=23,
        dataset_name="mock",
    )

    assert bool(captured["called"]) is True
    assert str(out["enabled_methods_source"]) == "regime_gate:low_p_over_n"
    assert bool(out["regime_policy_bypass_fs"]) is True

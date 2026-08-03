from __future__ import annotations

import numpy as np

from tabnetics.pipeline.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline


DIAKRINO_METHODS = ("diakrino_prior", "diakrino_screening_prior")


def _xy(n_samples: int, n_features: int, seed: int = 167):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(float)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2), dtype=int)
    return X, y


def _base_config(**overrides):
    cfg = DFFSConfig(
        random_seed=167,
        fs_fraction=0.5,
        n_final_features=8,
        max_dist_features=16,
        prefilter_top_k=32,
        enabled_methods=(
            "gradient_boosting",
            "linear_svm",
            "mutual_information",
            "anova_f",
            *DIAKRINO_METHODS,
        ),
        enable_maqc_pairing=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_diakrino_regime_conditional_disabled_keeps_configured_diakrino_methods():
    X, y = _xy(n_samples=240, n_features=40)
    pipe = DistributionFeatureSelectionPipeline(_base_config(diakrino_regime_conditional=False))

    policy = pipe._resolve_method_policy("mock", X, y)

    assert bool(policy["diakrino_regime_conditional_enabled"]) is False
    assert bool(policy["diakrino_regime_conditional_applied"]) is False
    assert tuple(policy["enabled_methods"]) == tuple(pipe.config.enabled_methods)


def test_diakrino_regime_conditional_keeps_diakrino_methods_in_hdlss_moderate():
    X, y = _xy(n_samples=100, n_features=6000)
    pipe = DistributionFeatureSelectionPipeline(_base_config(diakrino_regime_conditional=True))

    policy = pipe._resolve_method_policy("mock", X, y)

    assert bool(policy["diakrino_regime_conditional_enabled"]) is True
    assert bool(policy["diakrino_regime_conditional_allowed"]) is True
    assert str(policy["diakrino_regime_conditional_regime"]) == "hdlss_moderate"
    assert bool(policy["diakrino_regime_conditional_applied"]) is False
    assert tuple(policy["enabled_methods"]) == tuple(pipe.config.enabled_methods)


def test_diakrino_regime_conditional_removes_only_diakrino_methods_in_standard_regime():
    X, y = _xy(n_samples=240, n_features=40)
    pipe = DistributionFeatureSelectionPipeline(_base_config(diakrino_regime_conditional=True))

    policy = pipe._resolve_method_policy("mock", X, y)

    assert str(policy["diakrino_regime_conditional_regime"]) == "standard"
    assert bool(policy["diakrino_regime_conditional_allowed"]) is False
    assert bool(policy["diakrino_regime_conditional_applied"]) is True
    assert tuple(policy["diakrino_regime_conditional_removed_methods"]) == DIAKRINO_METHODS
    assert tuple(policy["enabled_methods"]) == (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
    )
    assert str(policy["enabled_methods_source"]) == "config+diakrino_regime_conditional:standard"


def test_diakrino_regime_conditional_composes_after_tier_routing():
    X, y = _xy(n_samples=240, n_features=40)
    pipe = DistributionFeatureSelectionPipeline(
        _base_config(
            diakrino_regime_conditional=True,
            tier_routing_enabled=True,
            tier_routing_difficulty_classifier="historical",
            tier_routing_table={
                "hard": ("linear_svm", "anova_f", "diakrino_prior", "diakrino_screening_prior")
            },
        )
    )
    pipe._resolve_dataset_catalog_context = lambda _dataset_name: {
        "dataset_id": "mock",
        "display_name": "mock",
        "domain": "",
        "tier": "hard",
        "is_face_domain": False,
        "found_in_catalog": True,
    }

    policy = pipe._resolve_method_policy("mock", X, y)

    assert bool(policy["tier_policy_applied"]) is True
    assert tuple(policy["enabled_methods"]) == ("linear_svm", "anova_f")
    assert tuple(policy["diakrino_regime_conditional_removed_methods"]) == DIAKRINO_METHODS
    assert str(policy["enabled_methods_source"]) == (
        "tier_routing:hard+diakrino_regime_conditional:standard"
    )


def test_diakrino_regime_conditional_choose_candidate_uses_gated_methods(monkeypatch):
    X, y = _xy(n_samples=240, n_features=40)
    pipe = DistributionFeatureSelectionPipeline(_base_config(diakrino_regime_conditional=True))
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
        seed=167,
        dataset_name="mock",
    )

    assert captured["methods"] == (
        "gradient_boosting",
        "linear_svm",
        "mutual_information",
        "anova_f",
    )
    assert bool(out["diakrino_regime_conditional_applied"]) is True
    assert tuple(out["diakrino_regime_conditional_removed_methods"]) == DIAKRINO_METHODS

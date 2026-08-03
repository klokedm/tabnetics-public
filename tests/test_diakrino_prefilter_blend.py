"""DIAKRINO prefilter replay and protected-core augmentation contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tabnetics.feature_selection.config import FeatureSelectorConfig
from tabnetics.pipeline.pipeline import (
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
    DistributionFitterConfig,
)


def _data(n=40, p=10, target=7, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, p))
    # features 0,1,2 carry the signal; `target` is pure noise (won't survive top-3 alone)
    for j in (0, 1, 2):
        X[:, j] += 3.0 * y
    return X.astype(float), y, target


def _keep(cfg, X, y, external=None):
    pipe = DistributionFeatureSelectionPipeline(cfg)
    pipe._diakrino_external_feature_scores = external
    _, _, keep_idx = pipe._rank_prefilter(X, y, X, seed=0)
    return set(int(i) for i in keep_idx)


def _prefilter(cfg, X, y, external=None):
    pipe = DistributionFeatureSelectionPipeline(cfg)
    pipe._diakrino_external_feature_scores = external
    X_train, X_test, keep_idx = pipe._rank_prefilter(X, y, X, seed=0)
    return pipe, X_train, X_test, np.asarray(keep_idx, dtype=int)


def _stub_candidate_stages(pipe, monkeypatch, selected_indices=(0, 1)):
    class _CoreOnlySelector:
        mnpo_diagnostics_ = {}

        def __init__(self):
            self.fit_width = None
            self.build_kwargs = {}

        def fit_transform(self, X_fit, y_fit, n_final_features, return_result_object):
            self.fit_width = int(X_fit.shape[1])
            return X_fit[:, selected_indices], None

        def transform(self, X_eval):
            return np.asarray(X_eval, dtype=float)[:, selected_indices]

        def get_selected_features_indices(self):
            return np.asarray(selected_indices, dtype=int)

    selector = _CoreOnlySelector()

    def build_selector(**kwargs):
        selector.build_kwargs = dict(kwargs)
        return selector

    monkeypatch.setattr(pipe, "_build_feature_selector", build_selector)
    monkeypatch.setattr(
        pipe,
        "_apply_post_selection_distribution_transform",
        lambda **kwargs: (
            kwargs["X_train_selected"],
            kwargs["X_test_selected"],
            [],
            {},
            0.0,
        ),
    )
    monkeypatch.setattr(
        pipe,
        "_apply_folding_stage",
        lambda **kwargs: (
            kwargs["X_train_fs_input"],
            kwargs["X_test_fs_input"],
            {},
        ),
    )
    monkeypatch.setattr(
        pipe,
        "_stage2_ratio_augmentation",
        lambda **kwargs: (kwargs["X_train_sel"], kwargs["X_test_sel"], {}),
    )
    monkeypatch.setattr(
        pipe,
        "_select_model_via_cv_scored",
        lambda *args, **kwargs: (object(), "stub", 0.5, 0.0, 2, {}),
    )
    return selector


def test_protected_prefilter_config_normalizes_bounds_and_probe_indices():
    cfg = DFFSConfig(
        diakrino_prefilter_mode="unknown",
        diakrino_prefilter_max_extras=-3,
        diakrino_prefilter_shadow_probe_indices=(7, -1, 7, 2),
    )

    assert cfg.diakrino_prefilter_mode == "protected_union"
    assert cfg.diakrino_prefilter_max_extras == 0
    assert cfg.diakrino_prefilter_shadow_probe_indices == (2, 7)


def test_protected_union_adds_target_without_evicting_classical_core():
    X, y, target = _data()
    cfg = DFFSConfig(
        prefilter_top_k=3,
        diakrino_prefilter_enabled=True,
        diakrino_prefilter_mode="protected_union",
        diakrino_prefilter_lambda=0.9,
        diakrino_prefilter_max_extras=1,
    )
    onehot = np.zeros(X.shape[1])
    onehot[target] = 1.0
    baseline = _keep(DFFSConfig(prefilter_top_k=3), X, y, external=None)
    pipe, _, _, keep_idx = _prefilter(cfg, X, y, external=onehot)
    augmented = set(int(i) for i in keep_idx)

    assert target not in baseline
    assert baseline <= augmented
    assert target in augmented
    assert len(augmented) == len(baseline) + 1
    state = pipe._last_diakrino_prefilter_state
    assert state["mode"] == "protected_union"
    assert state["diakrino_addition_budget"] == 1
    assert state["diakrino_extra_original_indices"] == (target,)


def test_legacy_fixed_budget_blend_is_explicitly_eviction_capable():
    X, y, target = _data()
    cfg = DFFSConfig(
        prefilter_top_k=3,
        diakrino_prefilter_enabled=True,
        diakrino_prefilter_mode="legacy_fixed_budget_blend",
        diakrino_prefilter_lambda=0.9,
    )
    onehot = np.zeros(X.shape[1])
    onehot[target] = 1.0
    baseline = _keep(DFFSConfig(prefilter_top_k=3), X, y, external=None)
    pipe, _, _, keep_idx = _prefilter(cfg, X, y, external=onehot)
    replayed = set(int(i) for i in keep_idx)

    assert target in replayed
    assert len(replayed) == len(baseline)
    assert not baseline <= replayed
    assert pipe._last_diakrino_prefilter_state["reason"] == "legacy_fixed_budget_blend_can_evict"


def test_disabled_is_strict_noop():
    X, y, target = _data()
    onehot = np.zeros(X.shape[1])
    onehot[target] = 1.0
    _, base_train, base_test, baseline = _prefilter(
        DFFSConfig(prefilter_top_k=3), X, y, external=None
    )
    _, disabled_train, disabled_test, disabled = _prefilter(
        DFFSConfig(
            prefilter_top_k=3,
            diakrino_prefilter_enabled=False,
            diakrino_prefilter_lambda=0.9,
        ),
        X,
        y,
        external=onehot,
    )
    np.testing.assert_array_equal(disabled, baseline)
    np.testing.assert_array_equal(disabled_train, base_train)
    np.testing.assert_array_equal(disabled_test, base_test)


def test_disabled_run_metadata_matches_default_and_enabled_run_keeps_provenance():
    X, y, target = _data()
    onehot = np.zeros(X.shape[1])
    onehot[target] = 1.0
    common = {
        "random_seed": 5,
        "fs_fraction": 1.0,
        "n_final_features": 2,
        "prefilter_top_k": 3,
        "folding_method": "none",
        "apply_cdf_transform": False,
        "enabled_methods": ("mutual_information", "anova_f"),
        "enable_maqc_pairing": True,
        "eval_models_enabled": False,
        "model_candidates": ("lr",),
    }

    def run(cfg, external):
        return DistributionFeatureSelectionPipeline(cfg).run_pre_split(
            X_train=X[:30],
            y_train=y[:30],
            X_test=X[30:],
            y_test=y[30:],
            external_feature_scores=external,
            capture_diagnostics=True,
        )

    def stable_metadata(value):
        if isinstance(value, dict):
            return {
                key: stable_metadata(item)
                for key, item in value.items()
                if not str(key).endswith(("_wall_seconds", "_time_sec"))
            }
        if isinstance(value, list):
            return [stable_metadata(item) for item in value]
        return value

    default_result = run(DFFSConfig(**common), None)
    disabled_result = run(
        DFFSConfig(
            **common,
            diakrino_prefilter_enabled=False,
            diakrino_prefilter_mode="legacy_fixed_budget_blend",
            diakrino_prefilter_lambda=0.9,
            diakrino_prefilter_max_extras=4,
            diakrino_prefilter_shadow_probe_indices=(target,),
        ),
        onehot,
    )
    lambda_zero_result = run(
        DFFSConfig(
            **common,
            diakrino_prefilter_enabled=True,
            diakrino_prefilter_mode="legacy_fixed_budget_blend",
            diakrino_prefilter_lambda=0.0,
            diakrino_prefilter_max_extras=4,
            diakrino_prefilter_shadow_probe_indices=(target,),
        ),
        onehot,
    )

    assert json.dumps(stable_metadata(disabled_result.config_snapshot), sort_keys=True) == json.dumps(
        stable_metadata(default_result.config_snapshot), sort_keys=True
    )
    assert json.dumps(stable_metadata(lambda_zero_result.config_snapshot), sort_keys=True) == json.dumps(
        stable_metadata(default_result.config_snapshot), sort_keys=True
    )
    default_stages = default_result.run_diagnostics["pipeline_stages"]
    disabled_stages = disabled_result.run_diagnostics["pipeline_stages"]
    lambda_zero_stages = lambda_zero_result.run_diagnostics["pipeline_stages"]
    assert json.dumps(stable_metadata(disabled_stages["prefilter"]), sort_keys=True) == json.dumps(
        stable_metadata(default_stages["prefilter"]), sort_keys=True
    )
    assert json.dumps(
        stable_metadata(disabled_stages["feature_selection"]), sort_keys=True
    ) == json.dumps(
        stable_metadata(default_stages["feature_selection"]), sort_keys=True
    )
    assert json.dumps(stable_metadata(lambda_zero_stages["prefilter"]), sort_keys=True) == json.dumps(
        stable_metadata(default_stages["prefilter"]), sort_keys=True
    )
    assert json.dumps(
        stable_metadata(lambda_zero_stages["feature_selection"]), sort_keys=True
    ) == json.dumps(
        stable_metadata(default_stages["feature_selection"]), sort_keys=True
    )
    new_snapshot_keys = {
        "diakrino_prefilter_enabled",
        "diakrino_prefilter_mode",
        "diakrino_prefilter_lambda",
        "diakrino_prefilter_max_extras",
        "diakrino_prefilter_score_column",
        "diakrino_prefilter_shadow_probe_indices",
        "diakrino_protected_augmentation",
    }
    assert new_snapshot_keys.isdisjoint(default_result.config_snapshot)
    assert "diakrino_prefilter" not in default_stages["prefilter"]
    assert "diakrino_protected_augmentation" not in default_stages["feature_selection"]

    enabled_result = run(
        DFFSConfig(
            **{
                **common,
                "enabled_methods": (
                    "anova_f",
                    "diakrino_prior",
                    "diakrino_conformal_selection",
                ),
            },
            diakrino_prefilter_enabled=True,
            diakrino_prefilter_lambda=0.9,
            diakrino_prefilter_max_extras=1,
        ),
        onehot,
    )
    enabled_snapshot = enabled_result.config_snapshot
    assert enabled_snapshot["diakrino_prefilter_enabled"] is True
    assert enabled_snapshot["diakrino_prefilter_mode"] == "protected_union"
    assert enabled_snapshot["diakrino_prefilter_max_extras"] == 1
    assert enabled_snapshot["enabled_methods"] == ["anova_f"]
    assert enabled_snapshot["effective_enabled_methods"] == ["anova_f"]
    assert enabled_snapshot["requested_enabled_methods"] == [
        "anova_f",
        "diakrino_prior",
        "diakrino_conformal_selection",
    ]
    assert enabled_snapshot["maqc_pairing_score_space"] == (
        "classical_only_before_diakrino_augmentation"
    )
    assert "maqc_pairing_augmented_cv_score" in enabled_snapshot
    assert enabled_snapshot["maqc_pairing_augmented_selected_feature_count"] == 3
    assert enabled_snapshot["diakrino_protected_augmentation"]["diakrino_additions"] == 1
    assert enabled_snapshot["diakrino_protected_augmentation"][
        "protected_effective_methods"
    ] == ["anova_f"]
    enabled_stages = enabled_result.run_diagnostics["pipeline_stages"]
    assert enabled_stages["prefilter"]["diakrino_prefilter"]["protection_active"] is True
    assert enabled_stages["feature_selection"]["diakrino_protected_augmentation"][
        "protected_core_retention_rate"
    ] == 1.0
    assert set(
        enabled_stages["feature_selection"]["detailed"]["config"]["enabled_methods"]
    ) == {"anova_f"}

    legacy_result = run(
        DFFSConfig(
            **common,
            diakrino_prefilter_enabled=True,
            diakrino_prefilter_mode="legacy_fixed_budget_blend",
            diakrino_prefilter_lambda=0.9,
            diakrino_prefilter_max_extras=4,
            diakrino_prefilter_shadow_probe_indices=(target,),
        ),
        onehot,
    )
    legacy_snapshot = legacy_result.config_snapshot
    assert legacy_snapshot["diakrino_prefilter_enabled"] is True
    assert legacy_snapshot["diakrino_prefilter_mode"] == "legacy_fixed_budget_blend"
    assert legacy_snapshot["diakrino_prefilter_lambda"] == pytest.approx(0.9)
    assert legacy_snapshot["diakrino_prefilter_max_extras"] == 4
    assert legacy_snapshot["diakrino_prefilter_shadow_probe_indices"] == [target]
    legacy_stages = legacy_result.run_diagnostics["pipeline_stages"]
    assert legacy_stages["prefilter"]["diakrino_prefilter"]["mode"] == (
        "legacy_fixed_budget_blend"
    )
    assert legacy_stages["prefilter"]["diakrino_prefilter"]["reason"] == (
        "legacy_fixed_budget_blend_can_evict"
    )
    assert "diakrino_protected_augmentation" not in legacy_stages["feature_selection"]


def test_lambda_zero_is_noop():
    X, y, target = _data()
    onehot = np.zeros(X.shape[1])
    onehot[target] = 1.0
    baseline = _keep(DFFSConfig(prefilter_top_k=3), X, y, external=None)
    lam0 = _keep(DFFSConfig(prefilter_top_k=3, diakrino_prefilter_enabled=True,
                            diakrino_prefilter_lambda=0.0), X, y, external=onehot)
    assert lam0 == baseline


def test_width_mismatch_is_noop_not_crash():
    X, y, target = _data()
    bad = np.ones(X.shape[1] + 1)  # wrong length => must be ignored, not misaligned
    cfg = DFFSConfig(prefilter_top_k=3, diakrino_prefilter_enabled=True, diakrino_prefilter_lambda=0.9)
    baseline = _keep(DFFSConfig(prefilter_top_k=3), X, y, external=None)
    mismatched = _keep(cfg, X, y, external=bad)
    assert mismatched == baseline


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    (
        ("top_k_covers_all", "protected_union_active"),
        ("rank_prefilter_disabled", "protected_union_active"),
        ("zero_budget", "zero_extra_budget"),
        ("missing_scores", "missing_or_misaligned_scores"),
        ("misaligned_scores", "missing_or_misaligned_scores"),
        ("constant_scores", "noninformative_scores"),
        ("ranked_inside_classical", "protected_union_active"),
    ),
)
def test_protection_activates_for_every_fail_closed_branch(case, expected_reason, monkeypatch):
    X, y, _ = _data()
    cfg_kwargs = {
        "prefilter_top_k": 3,
        "diakrino_prefilter_enabled": True,
        "diakrino_prefilter_lambda": 1.0,
        "diakrino_prefilter_max_extras": 1,
    }
    external = np.zeros(X.shape[1])
    external[7] = 1.0
    if case == "top_k_covers_all":
        cfg_kwargs["prefilter_top_k"] = X.shape[1]
    elif case == "rank_prefilter_disabled":
        cfg_kwargs["use_rank_prefilter"] = False
    elif case == "zero_budget":
        cfg_kwargs["diakrino_prefilter_max_extras"] = 0
    elif case == "missing_scores":
        external = None
    elif case == "misaligned_scores":
        external = np.ones(X.shape[1] + 1)
    elif case == "constant_scores":
        external = np.ones(X.shape[1])
    elif case == "ranked_inside_classical":
        external = np.zeros(X.shape[1])
        external[0] = 1.0

    pipe, X_pool, _, _ = _prefilter(DFFSConfig(**cfg_kwargs), X, y, external=external)
    state = pipe._last_diakrino_prefilter_state

    assert state["protection_active"] is True
    assert state["reason"] == expected_reason
    assert pipe._diakrino_protected_selection_context(
        len(state["active_original_indices"])
    ) is not None
    assert pipe._protected_classical_methods(
        ("anova_f", "diakrino_prior", "diakrino_screening_prior", "diakrino_conformal_selection")
    ) == ("anova_f",)
    _stub_candidate_stages(pipe, monkeypatch, selected_indices=(0,))
    result = pipe._evaluate_selector_candidate(
        X_fs=X_pool,
        y_fs=y,
        X_train_full=X_pool,
        X_test_full=X_pool,
        y_train_full=y,
        seed=0,
        enabled_methods=(
            "anova_f",
            "diakrino_prior",
            "diakrino_screening_prior",
            "diakrino_conformal_selection",
        ),
        candidate_name=case,
    )
    diag = result["diakrino_protected_augmentation"]
    assert result["enabled_methods"] == ("anova_f",)
    assert diag["protected_core_retention_rate"] == 1.0
    assert set(diag["protected_core_original_indices"]) <= set(
        diag["final_original_indices"]
    )
    if case == "ranked_inside_classical":
        assert state["diakrino_extra_original_indices"] == (3,)
        assert state["diakrino_ranked_candidate_original_indices"][0] == 0


def test_constant_diakrino_scores_do_not_add_arbitrary_extras():
    X, y, _ = _data()
    cfg = DFFSConfig(
        prefilter_top_k=3,
        diakrino_prefilter_enabled=True,
        diakrino_prefilter_lambda=1.0,
        diakrino_prefilter_max_extras=2,
    )
    baseline = _keep(DFFSConfig(prefilter_top_k=3), X, y)
    pipe, _, _, keep_idx = _prefilter(cfg, X, y, external=np.ones(X.shape[1]))

    assert set(int(i) for i in keep_idx) == baseline
    assert pipe._last_diakrino_prefilter_state["reason"] == "noninformative_scores"
    assert pipe._last_diakrino_prefilter_state["diakrino_valid_finite_candidate_count"] == X.shape[1]


def test_shadow_probe_is_rejected_and_budget_backfills_next_eligible_extra():
    X, y, target = _data()
    baseline = _keep(DFFSConfig(prefilter_top_k=3), X, y)
    backfill = next(i for i in range(X.shape[1]) if i not in baseline and i != target)
    scores = -np.arange(X.shape[1], dtype=float)
    scores[target] = 100.0
    scores[backfill] = 99.0
    cfg = DFFSConfig(
        prefilter_top_k=3,
        diakrino_prefilter_enabled=True,
        diakrino_prefilter_lambda=0.9,
        diakrino_prefilter_max_extras=1,
        diakrino_prefilter_shadow_probe_indices=(target,),
    )
    pipe, _, _, keep_idx = _prefilter(cfg, X, y, external=scores)

    state = pipe._last_diakrino_prefilter_state
    assert set(int(i) for i in keep_idx) == baseline | {backfill}
    assert target not in keep_idx
    assert state["diakrino_extra_original_indices"] == (backfill,)
    assert state["diakrino_admitted_outside_candidate_count"] == 1
    assert state["diakrino_eligible_outside_candidate_count"] > 1
    assert state["diakrino_budget_scan_count"] == 2
    assert state["diakrino_budget_scan_exhausted"] is False
    assert state["shadow_probe_candidate_original_indices"] == (target,)
    assert state["shadow_probe_candidate_count"] == 1
    assert state["shadow_probe_candidate_denominator"] == 2
    assert state["shadow_probe_candidate_fraction"] == pytest.approx(0.5)


def test_classical_top_rank_does_not_consume_outside_extra_budget():
    X, y, _ = _data()
    baseline = _keep(DFFSConfig(prefilter_top_k=3), X, y)
    classical_top = min(baseline)
    backfill = next(i for i in range(X.shape[1]) if i not in baseline)
    scores = -np.arange(X.shape[1], dtype=float)
    scores[classical_top] = 100.0
    scores[backfill] = 99.0
    cfg = DFFSConfig(
        prefilter_top_k=3,
        diakrino_prefilter_enabled=True,
        diakrino_prefilter_lambda=1.0,
        diakrino_prefilter_max_extras=1,
    )

    pipe, _, _, keep_idx = _prefilter(cfg, X, y, external=scores)
    state = pipe._last_diakrino_prefilter_state

    assert set(int(i) for i in keep_idx) == baseline | {backfill}
    assert state["diakrino_extra_original_indices"] == (backfill,)
    assert state["diakrino_ranked_candidate_original_indices"][:2] == (
        classical_top,
        backfill,
    )
    assert state["diakrino_admitted_outside_candidate_count"] == 1
    assert state["diakrino_eligible_outside_candidate_count"] > 1
    assert state["diakrino_budget_scan_count"] == 2
    assert state["diakrino_budget_scan_exhausted"] is False
    assert state["shadow_probe_candidate_count"] == 0
    assert state["shadow_probe_candidate_denominator"] == 2
    assert state["shadow_probe_candidate_fraction"] == pytest.approx(0.0)


def test_nonfinite_scores_are_never_admitted_when_finite_budget_underfills():
    X, y, _ = _data()
    baseline = _keep(DFFSConfig(prefilter_top_k=3), X, y)
    classical_top = min(baseline)
    finite_extra = next(i for i in range(X.shape[1]) if i not in baseline)
    scores = np.full(X.shape[1], np.nan, dtype=float)
    scores[classical_top] = 2.0
    scores[finite_extra] = 1.0
    nonfinite_outside = [
        i for i in range(X.shape[1]) if i not in baseline and i != finite_extra
    ]
    scores[nonfinite_outside[0]] = np.inf
    scores[nonfinite_outside[1]] = -np.inf
    cfg = DFFSConfig(
        prefilter_top_k=3,
        diakrino_prefilter_enabled=True,
        diakrino_prefilter_lambda=1.0,
        diakrino_prefilter_max_extras=4,
    )

    pipe, _, _, keep_idx = _prefilter(cfg, X, y, external=scores)
    state = pipe._last_diakrino_prefilter_state
    admitted_or_ranked = set(state["diakrino_extra_original_indices"]) | set(
        state["diakrino_ranked_candidate_original_indices"]
    )

    assert set(int(i) for i in keep_idx) == baseline | {finite_extra}
    assert state["diakrino_extra_original_indices"] == (finite_extra,)
    assert state["diakrino_valid_finite_candidate_count"] == 2
    assert state["diakrino_eligible_outside_candidate_count"] == 1
    assert state["diakrino_admitted_outside_candidate_count"] == 1
    assert state["diakrino_budget_scan_count"] == 2
    assert state["diakrino_budget_scan_exhausted"] is True
    assert all(np.isfinite(scores[i]) for i in admitted_or_ranked)


def test_final_narrowing_keeps_classical_selection_and_appends_noise(monkeypatch):
    X, y, target = _data()
    onehot = np.zeros(X.shape[1])
    onehot[target] = 1.0
    cfg = DFFSConfig(
        prefilter_top_k=3,
        n_final_features=2,
        folding_method="none",
        diakrino_prefilter_enabled=True,
        diakrino_prefilter_lambda=0.9,
        diakrino_prefilter_max_extras=1,
    )
    pipe, X_union, _, keep_idx = _prefilter(cfg, X, y, external=onehot)
    selector = _stub_candidate_stages(pipe, monkeypatch)

    result = pipe._evaluate_selector_candidate(
        X_fs=X_union,
        y_fs=y,
        X_train_full=X_union,
        X_test_full=X_union,
        y_train_full=y,
        seed=0,
        enabled_methods=("anova_f", "diakrino_prior"),
        candidate_name="adversarial_onehot",
    )

    diag = result["diakrino_protected_augmentation"]
    assert selector.fit_width == 3
    assert diag["protected_core_size"] == 2
    assert diag["protected_core_retained_size"] == 2
    assert diag["protected_core_retention_rate"] == 1.0
    assert diag["diakrino_additions"] == 1
    assert diag["diakrino_extra_original_indices"] == [target]
    assert set(diag["protected_core_original_indices"]) <= set(diag["final_original_indices"])
    assert target in diag["final_original_indices"]
    assert result["X_train_sel"].shape[1] == 3
    assert result["enabled_methods"] == ("anova_f",)
    assert result["requested_enabled_methods"] == ("anova_f", "diakrino_prior")
    assert selector.build_kwargs["enabled_methods"] == ("anova_f",)
    replayed = result["_fitted_selector"].transform(X_union)
    np.testing.assert_allclose(replayed, result["X_train_sel"])
    assert set(diag["final_original_indices"]) == set(int(keep_idx[i]) for i in result["selected_indices"])


def test_ranked_candidate_inside_classical_pool_is_readded_after_selection(monkeypatch):
    X, y, _ = _data()
    scores = np.zeros(X.shape[1])
    scores[0] = 1.0
    cfg = DFFSConfig(
        prefilter_top_k=3,
        n_final_features=2,
        folding_method="none",
        diakrino_prefilter_enabled=True,
        diakrino_prefilter_lambda=1.0,
        diakrino_prefilter_max_extras=1,
    )
    pipe, X_pool, _, keep_idx = _prefilter(cfg, X, y, external=scores)
    selector = _stub_candidate_stages(pipe, monkeypatch, selected_indices=(1, 2))

    result = pipe._evaluate_selector_candidate(
        X_fs=X_pool,
        y_fs=y,
        X_train_full=X_pool,
        X_test_full=X_pool,
        y_train_full=y,
        seed=0,
        enabled_methods=("anova_f", "diakrino_conformal_selection"),
        candidate_name="inside_classical",
    )

    diag = result["diakrino_protected_augmentation"]
    assert pipe._last_diakrino_prefilter_state["diakrino_extra_original_indices"] == (3,)
    assert diag["protected_core_original_indices"] == [1, 2]
    assert diag["diakrino_extra_original_indices"] == [0]
    assert diag["final_original_indices"] == [1, 2, 0]
    assert len(diag["final_original_indices"]) == len(set(diag["final_original_indices"]))
    assert result["enabled_methods"] == ("anova_f",)
    assert result["requested_enabled_methods"] == (
        "anova_f",
        "diakrino_conformal_selection",
    )
    assert selector.build_kwargs["protected_classical_core"] is True
    np.testing.assert_allclose(
        result["_fitted_selector"].transform(X_pool), result["X_train_sel"]
    )
    assert [int(keep_idx[i]) for i in result["selected_indices"]] == [1, 2, 0]


def test_structured_config_constructs_a_fully_classical_protected_selector():
    fs_config = FeatureSelectorConfig(
        enabled_methods={
            "anova_f",
            "diakrino_prior",
            "diakrino_screening_prior",
            "diakrino_conformal_selection",
        }
    )
    fs_config.methods.diakrino_prior_sidecar_path = "/untrusted/diakrino-sidecar"
    fs_config.methods.diakrino_conformal_selection_enabled = True
    fs_config.mnpo.use_diakrino_selector_prior = True
    fs_config.mnpo.oracle.use_diakrino_selector_prior = True
    fs_config.mnpo.oracle.use_diakrino_relevance_oracle = True
    cfg = DFFSConfig(
        fs_config=fs_config,
        diakrino_sidecar_path="/untrusted/diakrino-sidecar",
        diakrino_conformal_selection_enabled=True,
        fs_use_diakrino_relevance_oracle=True,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)

    selector = pipe._build_feature_selector(
        seed=3,
        enabled_methods=(
            "anova_f",
            "diakrino_prior",
            "diakrino_screening_prior",
            "diakrino_conformal_selection",
        ),
        protected_classical_core=True,
    )

    assert set(selector.enabled_methods) == {"anova_f"}
    assert selector.diakrino_prior_sidecar_path == ""
    assert selector.diakrino_conformal_selection_enabled is False
    assert selector.use_diakrino_selector_prior is False
    assert selector.oracle.use_diakrino_selector_prior is False
    assert selector.oracle.use_diakrino_relevance_oracle is False
    assert fs_config.methods.diakrino_prior_sidecar_path == "/untrusted/diakrino-sidecar"
    assert fs_config.methods.diakrino_conformal_selection_enabled is True
    assert fs_config.mnpo.use_diakrino_selector_prior is True
    assert fs_config.mnpo.oracle.use_diakrino_selector_prior is True


def test_maqc_pairing_freezes_classical_winner_before_diakrino_augmentation(monkeypatch):
    X, y, target = _data()
    scores = np.zeros(X.shape[1])
    scores[target] = 1.0
    cfg = DFFSConfig(
        prefilter_top_k=3,
        diakrino_prefilter_enabled=True,
        diakrino_prefilter_lambda=1.0,
        diakrino_prefilter_max_extras=1,
        enable_maqc_pairing=True,
        maqc_pairing_method_set_names=("classical_winner",),
        maqc_pairing_method_sets=(("anova_f",),),
        enabled_methods=("mutual_information",),
        maqc_pairing_min_improvement=0.0,
        maqc_pairing_min_improvement_se_mult=0.0,
    )
    pipe, X_pool, _, _ = _prefilter(cfg, X, y, external=scores)
    calls = []

    def fake_evaluate(**kwargs):
        name = str(kwargs["candidate_name"])
        augmented = bool(kwargs.get("append_diakrino_extras", True))
        calls.append((name, augmented))
        if name == "classical_winner":
            score = 0.90 if not augmented else 0.10
            selected = (0, 1, 3) if augmented else (0, 1)
        else:
            score = 0.70 if not augmented else 0.99
            selected = (1, 2, 3) if augmented else (1, 2)
        methods = tuple(str(m) for m in kwargs["enabled_methods"])
        return {
            "candidate_name": name,
            "enabled_methods": methods,
            "requested_enabled_methods": methods,
            "model_cv_score": score,
            "model_cv_score_std": 0.01,
            "model_cv_score_n_splits": 3,
            "selected_indices": selected,
            "diakrino_protected_augmentation": {
                "augmentation_deferred_for_pairing": not augmented,
            },
        }

    monkeypatch.setattr(pipe, "_evaluate_selector_candidate", fake_evaluate)

    selected = pipe._choose_selector_candidate(
        X_fs=X_pool,
        y_fs=y,
        X_train_full=X_pool,
        X_test_full=X_pool,
        y_train_full=y,
        seed=0,
    )

    assert calls == [
        ("classical_winner", False),
        ("configured_enabled_methods", False),
        ("classical_winner", True),
    ]
    assert selected["candidate_name"] == "classical_winner"
    assert selected["selected_indices"] == (0, 1, 3)
    assert selected["pairing_meta"]["maqc_pairing_score_space"] == (
        "classical_only_before_diakrino_augmentation"
    )
    assert selected["pairing_meta"]["maqc_pairing_selected_cv_score"] == pytest.approx(0.90)
    assert selected["pairing_meta"]["maqc_pairing_augmented_cv_score"] == pytest.approx(0.10)


def test_diakrino_sidecar_resolution_diagnostics_records_loaded_dataset(tmp_path):
    pd = pytest.importorskip("pandas")
    feature_dir = tmp_path / "feature_logits"
    feature_dir.mkdir()
    pd.DataFrame(
        {
            "dataset_id": ["alpha"] * 3,
            "feature_index": [0, 1, 2],
            "chunk_id": [0, 0, 0],
            "prior_logit": [1.0, 2.0, 3.0],
        }
    ).to_parquet(feature_dir / "alpha.parquet", index=False)

    cfg = DFFSConfig(diakrino_sidecar_path=str(tmp_path), diakrino_sidecar_dataset_id="alpha")
    pipe = DistributionFeatureSelectionPipeline(cfg)
    pipe._active_diakrino_dataset_id = "alpha"

    diag = pipe._diakrino_sidecar_resolution_diagnostics(3)

    assert diag["active_dataset_id"] == "alpha"
    assert diag["pipeline"]["status"] == "loaded"
    assert diag["pipeline"]["requested_dataset_id"] == "alpha"
    assert diag["pipeline"]["source_path"].endswith("feature_logits/alpha.parquet")
    assert diag["pipeline"]["n_features_match"] is True
    assert diag["distribution"]["status"] == "loaded"


def test_diakrino_sidecar_resolution_diagnostics_records_missing_dataset(tmp_path):
    cfg = DFFSConfig(diakrino_sidecar_path=str(tmp_path), diakrino_sidecar_dataset_id="missing")
    pipe = DistributionFeatureSelectionPipeline(cfg)
    pipe._active_diakrino_dataset_id = "missing"

    diag = pipe._diakrino_sidecar_resolution_diagnostics(3)

    assert diag["pipeline"]["status"] == "missing_or_unreadable"
    assert diag["pipeline"]["loaded"] is False


def test_diakrino_family_agreement_audit_counts_discrete_ids_before_decode(tmp_path):
    pd = pytest.importorskip("pandas")
    gamma = np.full(36, -9.0, dtype=float)
    gamma[4] = 9.0
    discrete = np.full(36, -9.0, dtype=float)
    discrete[31] = 9.0
    path = tmp_path / "sidecar.parquet"
    pd.DataFrame(
        {
            "feature_index": [0, 1],
            "chunk_id": [0, 0],
            "population_family_logits": [gamma, discrete],
        }
    ).to_parquet(path, index=False)
    cfg = DFFSConfig(
        dist_config=DistributionFitterConfig(
            diakrino_family_prescreen_enabled=True,
            diakrino_sidecar_path=str(path),
        )
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)

    audit = pipe._diakrino_family_agreement_audit(
        {
            0: {"family": "gamma"},
            1: {"family": "norm"},
        }
    )

    assert audit["enabled"] is True
    assert audit["loaded"] is True
    assert audit["n_features_audited"] == 2
    assert audit["n_compared"] == 1
    assert audit["n_agree"] == 1
    assert audit["agreement_rate"] == pytest.approx(1.0)
    assert audit["n_diakrino_discrete_or_nuisance"] == 1
    assert audit["examples"][1]["diakrino_family_id"] == 31
    assert audit["examples"][1]["skip_reason"] == "diakrino_discrete_or_nuisance_family"

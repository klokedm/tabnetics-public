import logging

import numpy as np
import pytest

from tabnetics.feature_selection.config import OracleConfig
from tabnetics.feature_selection.mnpo.oracles import (
    apply_bayesian_shrinkage_to_weights,
    compute_shapley_bayesian_shrinkage,
    estimate_oracle_preferences,
    fit_tritrust_weights,
)
from tabnetics.core.mnpo import (
    aggregate_payoff_matrix,
    fit_shapley_weights,
)


def _make_evaluation(
    *,
    n_candidates: int = 5,
    n_folds: int = 5,
    n_features: int = 32,
    n_samples: int = 140,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    names = [f"m{i}" for i in range(n_candidates)]
    evaluation = {}
    for i, name in enumerate(names):
        base = 0.55 + 0.05 * i
        perf = np.clip(rng.normal(base, 0.03, size=n_folds), 0.0, 1.0)
        selected = np.sort(
            rng.choice(n_features, size=min(n_features, 5 + i), replace=False)
        )
        evaluation[name] = {
            "performance_scores": np.asarray(perf, dtype=float),
            "performance_mean": float(np.mean(perf)),
            "stability": float(rng.uniform(0.2, 0.9)),
            "complexity": float(1.0 - len(selected) / max(1, n_features)),
            "robustness": float(np.min(perf)),
            "prediction_signal": rng.normal(0.0, 1.0, size=n_folds * 8),
            "target_signal": rng.integers(0, 2, size=n_folds * 8),
            "selected_indices": selected,
            "score_vector": rng.uniform(0.0, 1.0, size=n_features),
            "performance_scores_by_model": {},
            "n_samples": int(n_samples),
            "n_features": int(n_features),
        }
    return names, evaluation


def _estimate(names, evaluation, **kwargs):
    params = dict(
        pairwise_delta=0.01,
        use_cvar=False,
        cvar_alpha=0.33,
        use_tail_risk_oracle=False,
        tail_risk_alpha=0.33,
        use_qre_smoothing=False,
        qre_temperature_gamma=1.0,
        use_regret_oracle=False,
        use_stability_oracle=False,
        use_complexity_oracle=False,
        use_robust_oracle=False,
        use_diversity_oracle=False,
        diversity_oracle_mode="legacy_jaccard",
        diversity_redundancy_weight=0.6,
        diversity_complementarity_weight=0.35,
        performance_oracle_mode="single",
        weighting_mode="tritrust",
        shapley_n_coalitions_max=4096,
        use_ubayfs=False,
        ubayfs_n_bootstrap=32,
        ubayfs_min_n=100,
        ubayfs_prior_weight=0.0,
    )
    params.update(kwargs)
    return estimate_oracle_preferences(names, evaluation, **params)


def test_cvar_oracle_added_with_finite_matrix():
    names, evaluation = _make_evaluation(seed=10)
    oracle_matrices, oracle_scores, _, _ = _estimate(
        names, evaluation, use_cvar=True, cvar_alpha=0.5
    )
    assert "cvar" in oracle_matrices
    assert oracle_matrices["cvar"].shape == (len(names), len(names))
    assert np.isfinite(oracle_matrices["cvar"]).all()
    assert np.isfinite(oracle_scores["cvar"]).all()


def test_cvar_oracle_ties_when_all_candidates_have_equal_scores():
    names, evaluation = _make_evaluation(seed=11)
    for name in names:
        evaluation[name]["performance_scores"] = np.array([0.7, 0.7, 0.7], dtype=float)
        evaluation[name]["performance_mean"] = 0.7
    oracle_matrices, _, _, _ = _estimate(names, evaluation, use_cvar=True, cvar_alpha=0.2)
    np.testing.assert_allclose(oracle_matrices["cvar"], np.full((len(names), len(names)), 0.5))


def test_cvar_oracle_handles_single_fold_scores():
    names, evaluation = _make_evaluation(n_folds=1, seed=12)
    oracle_matrices, oracle_scores, _, _ = _estimate(names, evaluation, use_cvar=True, cvar_alpha=0.01)
    assert np.isfinite(oracle_scores["cvar"]).all()
    assert np.isfinite(oracle_matrices["cvar"]).all()


def test_complementarity_mode_applies_for_candidate_count_gt_four():
    names, evaluation = _make_evaluation(n_candidates=5, seed=13)
    oracle_matrices, _, oracle_components, _ = _estimate(
        names,
        evaluation,
        use_diversity_oracle=True,
        diversity_oracle_mode="complementarity",
    )
    assert "diversity" in oracle_matrices
    assert oracle_components["diversity"]["mode"] == "complementarity"


def test_complementarity_mode_falls_back_for_small_candidate_count():
    names, evaluation = _make_evaluation(n_candidates=4, seed=14)
    oracle_matrices, _, oracle_components, _ = _estimate(
        names,
        evaluation,
        use_diversity_oracle=True,
        diversity_oracle_mode="complementarity",
    )
    assert "diversity" in oracle_matrices
    assert oracle_components.get("diversity", {}).get("mode") != "complementarity"


def test_ubayfs_gate_skips_when_n_below_minimum():
    names, evaluation = _make_evaluation(n_samples=80, seed=15)
    oracle_matrices, _, oracle_components, _ = _estimate(
        names,
        evaluation,
        use_ubayfs=True,
        ubayfs_min_n=100,
    )
    assert "ubayfs" not in oracle_matrices
    assert oracle_components["ubayfs"]["applied"] is False
    assert oracle_components["ubayfs"]["reason"] == "min_n_gate"


def test_ubayfs_oracle_added_and_deterministic_for_same_seeded_input():
    names, evaluation = _make_evaluation(n_samples=150, seed=16)
    mats_a, scores_a, comps_a, _ = _estimate(
        names,
        evaluation,
        use_ubayfs=True,
        ubayfs_n_bootstrap=16,
        ubayfs_min_n=100,
        ubayfs_prior_weight=0.0,
    )
    mats_b, scores_b, comps_b, _ = _estimate(
        names,
        evaluation,
        use_ubayfs=True,
        ubayfs_n_bootstrap=16,
        ubayfs_min_n=100,
        ubayfs_prior_weight=0.0,
    )
    assert "ubayfs" in mats_a
    np.testing.assert_allclose(mats_a["ubayfs"], mats_b["ubayfs"])
    np.testing.assert_allclose(scores_a["ubayfs"], scores_b["ubayfs"])
    assert comps_a["ubayfs"]["applied"] is True
    assert comps_b["ubayfs"]["applied"] is True


def test_ubayfs_logs_warning_for_large_prior_weight(caplog):
    names, evaluation = _make_evaluation(n_samples=140, seed=17)
    with caplog.at_level(logging.WARNING):
        _estimate(
            names,
            evaluation,
            use_ubayfs=True,
            ubayfs_n_bootstrap=8,
            ubayfs_min_n=100,
            ubayfs_prior_weight=0.8,
        )
    assert any("prior_weight" in rec.message for rec in caplog.records)


def test_shapley_weights_gate_returns_uniform_when_coalition_cap_exceeded():
    perf = np.array([[0.5, 0.8], [0.2, 0.5]], dtype=float)
    stab = np.array([[0.5, 0.7], [0.3, 0.5]], dtype=float)
    comp = np.array([[0.5, 0.6], [0.4, 0.5]], dtype=float)
    with pytest.warns(RuntimeWarning):
        weights, meta = fit_shapley_weights(
            {"performance": perf, "stability": stab, "complexity": comp},
            max_coalitions=4,  # 2^3=8 > 4 => gate
        )
    assert meta["applied"] is False
    assert meta["reason"] == "coalition_cap_exceeded"
    assert all(v == pytest.approx(1.0 / 3.0) for v in weights.values())


def test_shapley_weights_favor_reference_agreement():
    perf = np.array([[0.5, 0.9], [0.1, 0.5]], dtype=float)
    anti = np.array([[0.5, 0.1], [0.9, 0.5]], dtype=float)
    weights, meta = fit_shapley_weights(
        {"performance": perf, "anti": anti},
        reference="performance",
        max_coalitions=16,
    )
    assert meta["applied"] is True
    assert weights["performance"] > weights["anti"]


def test_shapley_and_tritrust_equivalent_on_uniform_marginal_oracles():
    a = np.full((3, 3), 0.5, dtype=float)
    mats = {"performance": a, "stability": a.copy(), "complexity": a.copy()}
    w_shapley, _ = fit_shapley_weights(mats, max_coalitions=32)
    w_tritrust = fit_tritrust_weights(mats)
    agg_shapley = aggregate_payoff_matrix(mats, w_shapley)
    agg_tritrust = aggregate_payoff_matrix(mats, w_tritrust)
    np.testing.assert_allclose(agg_shapley, agg_tritrust, atol=1e-12)


def test_oracle_config_object_overrides_flat_kwargs():
    names, evaluation = _make_evaluation(seed=18)
    oracle_cfg = OracleConfig(
        use_cvar=True,
        cvar_alpha=0.20,
        use_ubayfs=True,
        ubayfs_n_bootstrap=8,
        ubayfs_min_n=100,
        ubayfs_prior_weight=0.0,
    )
    mats, _, comps, _ = _estimate(
        names,
        evaluation,
        use_cvar=False,
        use_ubayfs=False,
        oracle_config=oracle_cfg,
    )
    assert "cvar" in mats
    assert "ubayfs" in mats
    assert comps["ubayfs"]["applied"] is True


def test_shapley_bayesian_shrinkage_lambda_decreases_with_more_folds():
    names_small, eval_small = _make_evaluation(n_folds=3, seed=21)
    names_large, eval_large = _make_evaluation(n_folds=15, seed=22)
    meta_small = compute_shapley_bayesian_shrinkage(
        evaluation=eval_small,
        candidate_names=names_small,
        prior_strength=8.0,
    )
    meta_large = compute_shapley_bayesian_shrinkage(
        evaluation=eval_large,
        candidate_names=names_large,
        prior_strength=8.0,
    )
    assert float(meta_small["shrinkage_lambda"]) > float(meta_large["shrinkage_lambda"])


def test_apply_bayesian_shrinkage_moves_weights_toward_uniform():
    weights = {"a": 0.90, "b": 0.10}
    shrunk = apply_bayesian_shrinkage_to_weights(weights, shrinkage_lambda=0.5)
    assert shrunk["a"] < 0.90
    assert shrunk["b"] > 0.10
    assert pytest.approx(1.0, abs=1e-12) == sum(shrunk.values())


def test_interaction_oracle_skips_below_n_gate():
    names, evaluation = _make_evaluation(n_samples=120, seed=31)
    x_pool = np.random.default_rng(31).normal(size=(120, 48))
    mats, _, comps, _ = _estimate(
        names,
        evaluation,
        use_interaction_oracle=True,
        interaction_oracle_min_n_train=150,
        X_pool=x_pool,
    )
    assert "interaction_density" not in mats
    assert comps["interaction_density"]["applied"] is False
    assert comps["interaction_density"]["reason"] == "min_n_gate"


def test_interaction_oracle_added_when_gate_passes():
    names, evaluation = _make_evaluation(n_samples=180, seed=32)
    x_pool = np.random.default_rng(32).normal(size=(180, 64))
    mats, scores, comps, meta = _estimate(
        names,
        evaluation,
        use_interaction_oracle=True,
        interaction_oracle_min_n_train=150,
        X_pool=x_pool,
    )
    assert "interaction_density" in mats
    assert mats["interaction_density"].shape == (len(names), len(names))
    assert np.isfinite(mats["interaction_density"]).all()
    assert np.isfinite(scores["interaction_density"]).all()
    assert comps["interaction_density"]["applied"] is True
    assert "interaction_density" in meta

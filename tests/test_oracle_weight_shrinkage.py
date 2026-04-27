import numpy as np

import tabnetics.core.mnpo as mnpo_core
from tabnetics.feature_selection.mnpo import portfolio
from tabnetics.core.mnpo import james_stein_shrinkage


def test_james_stein_shrinkage_contracts_toward_mean():
    weights = {"performance": 1.0, "complexity": 0.0, "diversity": 0.0, "stability": 0.0}
    shrunk = james_stein_shrinkage(weights, effective_n=5)
    mu = np.mean(list(weights.values()))
    assert abs(shrunk["performance"] - mu) < abs(weights["performance"] - mu)
    assert abs(shrunk["complexity"] - mu) < abs(weights["complexity"] - mu)


def test_james_stein_shrinkage_bypasses_small_k():
    weights = {"performance": 0.8, "complexity": 0.2}
    assert james_stein_shrinkage(weights, effective_n=5) == weights


def test_james_stein_shrinkage_depends_on_effective_sample_size():
    weights = {"performance": 0.92, "complexity": 0.03, "diversity": 0.03, "stability": 0.02}
    mu = np.mean(list(weights.values()))
    shrunk_low_n = james_stein_shrinkage(weights, effective_n=3)
    shrunk_high_n = james_stein_shrinkage(weights, effective_n=30)
    assert abs(shrunk_low_n["performance"] - mu) < abs(shrunk_high_n["performance"] - mu)
    assert abs(shrunk_low_n["complexity"] - mu) < abs(shrunk_high_n["complexity"] - mu)


def _run_portfolio(monkeypatch, *, weighting_mode: str, js_enabled: bool):
    method_results = {
        "m1": (
            {"selected_indices": np.array([0, 1]), "scores": {0: 1.0, 1: 0.8}, "method": "m1"},
            np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0]),
        ),
        "m2": (
            {"selected_indices": np.array([1, 2]), "scores": {1: 1.0, 2: 0.7}, "method": "m2"},
            np.array([0.0, 1.0, 0.7, 0.0, 0.0, 0.0]),
        ),
        "m3": (
            {"selected_indices": np.array([2, 3]), "scores": {2: 1.0, 3: 0.6}, "method": "m3"},
            np.array([0.0, 0.0, 1.0, 0.6, 0.0, 0.0]),
        ),
    }
    method_runtimes = {"m1": 1.0, "m2": 1.0, "m3": 1.0}

    def _fake_runtime_race_candidates(X, y, candidates, **kwargs):
        return candidates, {"runtime_racing_applied": False}

    def _fake_evaluate_candidate_library(X, y, candidates, **kwargs):
        out = {}
        for idx, name in enumerate(candidates.keys()):
            out[name] = {
                "performance_scores": np.array([0.8 - idx * 0.05, 0.79 - idx * 0.05, 0.78 - idx * 0.05]),
                "performance_mean": float(0.8 - idx * 0.05),
                "complexity": float(idx + 1),
                "runtime": 1.0,
            }
        return out

    def _fake_estimate_oracle_preferences(candidate_names, evaluation, **kwargs):
        n = len(candidate_names)
        half = np.full((n, n), 0.5, dtype=float)
        return (
            {
                "performance": half.copy(),
                "complexity": half.copy(),
                "diversity": half.copy(),
                "stability": half.copy(),
            },
            {
                "performance": np.array([0.9, 0.8, 0.7]),
                "complexity": np.array([0.2, 0.5, 0.8]),
                "diversity": np.array([0.4, 0.4, 0.4]),
                "stability": np.array([0.7, 0.6, 0.5]),
            },
            {},
            {},
        )

    def _fake_mirror_descent(payoff, reference_prior, **kwargs):
        n = payoff.shape[0]
        return np.full(n, 1.0 / float(n), dtype=float), []

    monkeypatch.setattr(portfolio, "runtime_race_candidates", _fake_runtime_race_candidates)
    monkeypatch.setattr(portfolio, "evaluate_candidate_library", _fake_evaluate_candidate_library)
    monkeypatch.setattr(portfolio, "estimate_oracle_preferences", _fake_estimate_oracle_preferences)
    monkeypatch.setattr(portfolio, "aggregate_payoff_matrix", lambda oracle_matrices, oracle_weights: np.zeros((3, 3)))
    monkeypatch.setattr(portfolio, "mirror_descent_mnpo", _fake_mirror_descent)
    monkeypatch.setattr(
        portfolio,
        "fit_tritrust_weights",
        lambda oracle_matrices: {name: 1.0 / float(len(oracle_matrices)) for name in oracle_matrices},
    )
    monkeypatch.setattr(
        mnpo_core,
        "compute_banzhaf_values",
        lambda oracle_matrices, reference, max_coalitions: (
            {"performance": 1.0, "complexity": 0.0, "diversity": 0.0, "stability": 0.0},
            {"applied": True},
        ),
    )

    _, _, _, _, summary = portfolio.mnpo_select_features(
        X_uncorr=np.random.default_rng(11).normal(size=(20, 6)),
        y=np.array([0, 1] * 10),
        n_target=3,
        n_final_features=2,
        method_results=method_results,
        method_runtimes=method_runtimes,
        safe_normalize_scores_fn=lambda all_scores, selected, n_features: np.asarray(all_scores, dtype=float),
        calculate_weighted_votes_fn=lambda method_results, n_features: (np.zeros(n_features, dtype=float), {}),
        get_inner_cv_splits_fn=lambda X, y: [],
        fit_and_score_fold_fn=lambda *args, **kwargs: (0.5, np.zeros(1)),
        augment_training_data_fn=lambda X, y: (X, y),
        mnpo_include_legacy_consensus=False,
        mnpo_include_majority_consensus=False,
        mnpo_consensus_exclude_methods=tuple(),
        mnpo_consensus_exclude_protect_top_k=0,
        use_tritrust=True,
        use_oracle_redundancy_penalty=False,
        compute_tremble_sensitivity=False,
        mirror_descent_steps=10,
        mirror_descent_eta=0.1,
        mirror_descent_lambda=0.1,
        wrapper_refine_enabled=False,
        rank_aggregation_mode="none",
        portfolio_size=3,
        adaptive_portfolio_sizing_enabled=False,
        use_diversity_oracle=False,
        runtime_racing_enabled=False,
        runtime_racing_mode="single_stage",
        runtime_racing_proxy_splits=1,
        runtime_racing_keep_fraction=1.0,
        runtime_racing_min_candidates=1,
        runtime_racing_runtime_weight=0.0,
        runtime_racing_stages=1,
        runtime_racing_confidence_bound="none",
        runtime_racing_delta=0.1,
        use_robust_oracle=False,
        complexity_use_runtime_penalty=False,
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
        diversity_oracle_mode="legacy_jaccard",
        oracle_weighting_mode=weighting_mode,
        shapley_n_coalitions_max=32,
        shapley_bayesian_shrinkage=False,
        shapley_bayesian_prior_strength=8.0,
        use_interaction_oracle=False,
        interaction_oracle_min_n_train=10,
        interaction_oracle_pool_size_cap=4,
        interaction_oracle_pair_cap=16,
        use_ubayfs=False,
        ubayfs_n_bootstrap=4,
        ubayfs_min_n=10,
        ubayfs_prior_weight=0.0,
        use_conformal_uq=False,
        conformal_uq_alpha=0.10,
        conformal_uq_min_folds=3,
        fold_preference_mode="vote",
        use_conformal_efficiency=False,
        conformal_efficiency_method="split",
        oracle_weight_js_shrinkage=js_enabled,
        payoff_shrinkage_kappa=0.0,
        diversity_redundancy_weight=0.6,
        diversity_complementarity_weight=0.35,
        wrapper_refine_top_k=3,
        wrapper_refine_max_add=1,
        wrapper_refine_min_gain=0.0,
        performance_oracle_mode="single",
        mnpo_paradigm_aware_prior_enabled=False,
        mnpo_interaction_floor=0.0,
        rashomon_enabled=False,
        rashomon_max_models=4,
        rashomon_score_tolerance=0.01,
        selector_penalty_map={},
        random_state=11,
    )
    return summary


def test_oracle_weight_js_shrinkage_is_only_active_in_banzhaf_mode(monkeypatch):
    summary_banzhaf = _run_portfolio(monkeypatch, weighting_mode="banzhaf", js_enabled=True)
    summary_uniform = _run_portfolio(monkeypatch, weighting_mode="uniform", js_enabled=True)

    expected = james_stein_shrinkage(
        {"performance": 1.0, "complexity": 0.0, "diversity": 0.0, "stability": 0.0},
        effective_n=3,
    )
    assert summary_banzhaf["oracle_weight_js_shrinkage"] is True
    assert summary_banzhaf["oracle_weights"] == expected
    assert summary_uniform["oracle_weight_js_shrinkage"] is False

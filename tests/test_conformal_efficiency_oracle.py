import numpy as np
from sklearn.datasets import make_classification

from tabnetics.feature_selection.conformal import compute_conformal_singleton_rate
from tabnetics.feature_selection.mnpo.oracles import estimate_oracle_preferences
from tabnetics.core.compat import make_logistic_regression


def _make_model():
    return make_logistic_regression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=11,
    )


def test_conformal_efficiency_split_and_aps_paths_produce_singleton_rate():
    X, y = make_classification(
        n_samples=80,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        random_state=11,
    )
    split_meta = compute_conformal_singleton_rate(
        model=_make_model(),
        X_train=X[:50],
        y_train=y[:50],
        X_eval=X[50:],
        y_eval=y[50:],
        alpha=0.10,
        method="split",
        seed=11,
    )
    aps_meta = compute_conformal_singleton_rate(
        model=_make_model(),
        X_train=X[:50],
        y_train=y[:50],
        X_eval=X[50:],
        y_eval=y[50:],
        alpha=0.10,
        method="aps",
        seed=11,
    )

    for meta, method in ((split_meta, "split"), (aps_meta, "aps")):
        assert meta["conformal_efficiency_method"] == method
        assert 0.0 <= float(meta["conformal_singleton_rate"]) <= 1.0
        assert "conformal_coverage" in meta


def test_conformal_efficiency_oracle_respects_min_fold_gate():
    names = ["stable", "noisy"]
    evaluation = {
        "stable": {
            "performance_scores": np.array([0.8, 0.8]),
            "performance_mean": 0.8,
            "conformal_singleton_rates": np.array([0.9, 0.9]),
        },
        "noisy": {
            "performance_scores": np.array([0.7, 0.7]),
            "performance_mean": 0.7,
            "conformal_singleton_rates": np.array([0.4, 0.5]),
        },
    }
    matrices, _, components, _ = estimate_oracle_preferences(
        names,
        evaluation,
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
        use_interaction_oracle=False,
        use_ubayfs=False,
        use_conformal_uq=False,
        use_conformal_efficiency=True,
        conformal_efficiency_method="aps",
    )
    assert "conformal_efficiency" not in matrices
    assert components["conformal_efficiency"]["applied"] is False
    assert components["conformal_efficiency"]["reason"] == "min_folds_gate"


def test_conformal_efficiency_oracle_emits_diagnostics():
    names = ["stable", "noisy"]
    evaluation = {
        "stable": {
            "performance_scores": np.array([0.8, 0.82, 0.79]),
            "performance_mean": 0.803,
            "conformal_singleton_rates": np.array([0.95, 0.90, 0.92]),
            "conformal_coverages": np.array([0.93, 0.92, 0.94]),
        },
        "noisy": {
            "performance_scores": np.array([0.7, 0.71, 0.69]),
            "performance_mean": 0.700,
            "conformal_singleton_rates": np.array([0.45, 0.55, 0.50]),
            "conformal_coverages": np.array([0.92, 0.91, 0.90]),
        },
    }
    matrices, scores, components, _ = estimate_oracle_preferences(
        names,
        evaluation,
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
        use_interaction_oracle=False,
        use_ubayfs=False,
        use_conformal_uq=False,
        use_conformal_efficiency=True,
        conformal_efficiency_method="aps",
    )

    assert "conformal_efficiency" in matrices
    assert scores["conformal_efficiency"][0] > scores["conformal_efficiency"][1]
    comp = components["conformal_efficiency"]
    assert comp["applied"] is True
    assert "stable" in comp["conformal_singleton_rate_mean"]
    assert "stable" in comp["conformal_singleton_rate_std"]
    assert "stable" in comp["conformal_coverage_mean"]
    assert float(comp["conformal_efficiency_target_coverage"]) == 0.9


def test_conformal_efficiency_oracle_penalizes_undercoverage():
    names = ["high_singleton_undercover", "moderate_singleton_wellcovered"]
    evaluation = {
        "high_singleton_undercover": {
            "performance_scores": np.array([0.8, 0.81, 0.82]),
            "performance_mean": 0.81,
            "conformal_singleton_rates": np.array([0.95, 0.96, 0.94]),
            "conformal_coverages": np.array([0.72, 0.75, 0.74]),
        },
        "moderate_singleton_wellcovered": {
            "performance_scores": np.array([0.79, 0.80, 0.81]),
            "performance_mean": 0.80,
            "conformal_singleton_rates": np.array([0.60, 0.58, 0.62]),
            "conformal_coverages": np.array([0.93, 0.92, 0.91]),
        },
    }
    _, scores, components, _ = estimate_oracle_preferences(
        names,
        evaluation,
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
        use_interaction_oracle=False,
        use_ubayfs=False,
        use_conformal_uq=False,
        use_conformal_efficiency=True,
        conformal_efficiency_method="aps",
    )
    utility = components["conformal_efficiency"]["conformal_efficiency_utility"]
    assert utility["high_singleton_undercover"] < utility["moderate_singleton_wellcovered"]
    assert scores["conformal_efficiency"][0] < scores["conformal_efficiency"][1]

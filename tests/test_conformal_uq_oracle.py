import logging

import numpy as np

from tabnetics.feature_selection.mnpo.oracles import (
    _compute_conformal_interval_width,
    estimate_oracle_preferences,
)


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
        use_conformal_uq=False,
        conformal_uq_alpha=0.10,
        conformal_uq_min_folds=5,
    )
    params.update(kwargs)
    return estimate_oracle_preferences(names, evaluation, **params)


def _base_eval(scores_a, scores_b):
    return {
        "stable": {
            "performance_scores": np.asarray(scores_a, dtype=float),
            "performance_mean": float(np.mean(scores_a)),
        },
        "noisy": {
            "performance_scores": np.asarray(scores_b, dtype=float),
            "performance_mean": float(np.mean(scores_b)),
        },
    }


def test_conformal_width_zero_for_constant_scores():
    width = _compute_conformal_interval_width(np.full(15, 0.7, dtype=float), alpha=0.10)
    assert width == 0.0


def test_conformal_reliability_prefers_stable_fold_scores():
    names = ["stable", "noisy"]
    eval_map = _base_eval(
        np.full(15, 0.80, dtype=float),
        np.array([0.2, 0.9, 0.4, 0.8, 0.3, 0.7, 0.5, 0.9, 0.1, 0.8, 0.2, 0.7, 0.4, 0.9, 0.3]),
    )
    mats, _, _, _ = _estimate(
        names,
        eval_map,
        use_conformal_uq=True,
        conformal_uq_alpha=0.10,
        conformal_uq_min_folds=5,
    )
    assert "conformal_reliability" in mats
    mat = mats["conformal_reliability"]
    assert float(mat[0, 1]) > 0.5


def test_conformal_oracle_absent_by_default():
    names = ["stable", "noisy"]
    eval_map = _base_eval(np.full(15, 0.70), np.full(15, 0.65))
    mats, _, _, _ = _estimate(names, eval_map)
    assert "conformal_reliability" not in mats


def test_conformal_min_folds_gate_skips_oracle(caplog):
    names = ["stable", "noisy"]
    eval_map = _base_eval(np.full(3, 0.7), np.full(3, 0.6))
    with caplog.at_level(logging.WARNING):
        mats, _, comps, _ = _estimate(
            names,
            eval_map,
            use_conformal_uq=True,
            conformal_uq_min_folds=5,
        )
    assert "conformal_reliability" not in mats
    assert comps["conformal_reliability"]["applied"] is False
    assert any("Conformal UQ oracle skipped" in rec.message for rec in caplog.records)

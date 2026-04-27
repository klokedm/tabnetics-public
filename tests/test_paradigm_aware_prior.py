import numpy as np

from tabnetics.feature_selection.mnpo.portfolio import apply_paradigm_aware_prior_floor


def test_paradigm_prior_no_interaction_candidates_no_change():
    prior = np.array([0.5, 0.3, 0.2], dtype=float)
    names = ["gradient_boosting", "linear_svm", "anova_f"]
    out, meta = apply_paradigm_aware_prior_floor(prior, names, interaction_floor=0.12)
    np.testing.assert_allclose(out, prior / np.sum(prior))
    assert meta["reason"] in {"no_interaction_candidates", "already_satisfied"}


def test_paradigm_prior_applies_floor_when_missing_mass():
    prior = np.array([0.92, 0.06, 0.02], dtype=float)
    names = ["gradient_boosting", "linear_svm", "ktsp"]
    out, meta = apply_paradigm_aware_prior_floor(prior, names, interaction_floor=0.12)
    assert abs(float(np.sum(out)) - 1.0) < 1e-8
    assert float(out[2]) >= 0.12 - 1e-8
    assert bool(meta["applied"]) is True


def test_paradigm_prior_already_satisfied_keeps_distribution():
    prior = np.array([0.6, 0.2, 0.2], dtype=float)
    names = ["gradient_boosting", "ktsp", "joint_auc_l1"]
    out, meta = apply_paradigm_aware_prior_floor(prior, names, interaction_floor=0.12)
    np.testing.assert_allclose(out, prior / np.sum(prior))
    assert meta["reason"] == "already_satisfied"


def test_paradigm_prior_zero_floor_is_noop():
    prior = np.array([0.7, 0.2, 0.1], dtype=float)
    names = ["gradient_boosting", "linear_svm", "copula_knockoff"]
    out, meta = apply_paradigm_aware_prior_floor(prior, names, interaction_floor=0.0)
    np.testing.assert_allclose(out, prior / np.sum(prior))
    assert meta["reason"] == "zero_floor"


def test_paradigm_prior_normalizes_invalid_input():
    prior = np.array([np.nan, np.inf, -1.0, 0.0], dtype=float)
    names = ["gradient_boosting", "linear_svm", "ktsp", "copula_knockoff"]
    out, _ = apply_paradigm_aware_prior_floor(prior, names, interaction_floor=0.12)
    assert np.all(np.isfinite(out))
    assert abs(float(np.sum(out)) - 1.0) < 1e-8

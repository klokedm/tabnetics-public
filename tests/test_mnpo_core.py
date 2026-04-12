import numpy as np
import pytest

from tabnetics.core.mnpo import (
    pairwise_pref_from_fold_scores,
    pairwise_pref_logistic,
    shrink_payoff_matrix,
)


def test_pairwise_pref_logistic_is_continuous():
    pref = pairwise_pref_logistic(
        [0.82, 0.79, 0.77, 0.81],
        [0.71, 0.75, 0.70, 0.72],
        pairwise_delta=0.0,
    )
    assert 0.5 < pref < 1.0


def test_pairwise_pref_logistic_exact_tie_returns_half():
    pref = pairwise_pref_logistic([0.8, 0.8, 0.8], [0.8, 0.8, 0.8], pairwise_delta=0.01)
    assert pref == pytest.approx(0.5)


def test_pairwise_pref_logistic_respects_practical_tie_margin():
    pref = pairwise_pref_logistic(
        [0.801, 0.802, 0.801, 0.802],
        [0.800, 0.801, 0.800, 0.801],
        pairwise_delta=0.01,
    )
    assert pref == pytest.approx(0.5)


def test_pairwise_pref_logistic_zero_variance_effect_remains_strictly_internal():
    pref = pairwise_pref_logistic(
        [0.83, 0.83, 0.83, 0.83],
        [0.79, 0.79, 0.79, 0.79],
        pairwise_delta=0.01,
    )
    assert 0.5 < pref < 1.0


def test_vote_mode_preference_matches_empirical_vote():
    pref = pairwise_pref_from_fold_scores(
        [0.80, 0.78, 0.81, 0.79],
        [0.77, 0.79, 0.76, 0.79],
        pairwise_delta=0.01,
    )
    assert pref == pytest.approx(0.625)


def test_shrink_payoff_matrix_is_identity_at_zero_kappa():
    payoff = np.array(
        [
            [0.0, 0.8, -0.4],
            [-0.8, 0.0, 0.2],
            [0.4, -0.2, 0.0],
        ],
        dtype=float,
    )
    shrunk, meta = shrink_payoff_matrix(payoff, kappa=0.0)
    np.testing.assert_allclose(shrunk, payoff)
    assert meta["applied"] is False


def test_shrink_payoff_matrix_stronger_when_variance_is_lower():
    low_var = np.array(
        [
            [0.0, 0.10, 0.11],
            [-0.10, 0.0, 0.09],
            [-0.11, -0.09, 0.0],
        ],
        dtype=float,
    )
    high_var = np.array(
        [
            [0.0, 0.95, -0.10],
            [-0.95, 0.0, 0.45],
            [0.10, -0.45, 0.0],
        ],
        dtype=float,
    )
    shrunk_low, meta_low = shrink_payoff_matrix(low_var, kappa=0.15)
    shrunk_high, meta_high = shrink_payoff_matrix(high_var, kappa=0.15)

    assert meta_low["applied"] is True
    assert meta_high["applied"] is True
    assert float(meta_low["alpha"]) > float(meta_high["alpha"])
    assert float(np.linalg.norm(shrunk_low)) < float(np.linalg.norm(low_var))
    assert float(np.linalg.norm(shrunk_high)) < float(np.linalg.norm(high_var))

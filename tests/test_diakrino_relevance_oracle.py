"""S4: diakrino_relevance oracle in estimate_oracle_preferences.

The oracle scores each candidate by the mean chunk-calibrated DIAKRINO relevance of the
features it SELECTED (= dot/|selected|) and expands it to an m×m preference matrix —
biasing method trust without ever selecting.  Pins: it produces correct scalars when
enabled, is absent when disabled, and gates on missing-vector / min-n.
"""

from __future__ import annotations

import numpy as np

from tabnetics.feature_selection.mnpo.oracles import estimate_oracle_preferences


def _evaluation(n_samples=120):
    return {
        "m_hi": {  # selects the two DIAKRINO-favoured features
            "performance_scores": np.array([0.80, 0.82, 0.79]),
            "performance_mean": 0.803,
            "selected_indices": np.array([0, 1]),
            "n_samples": n_samples, "n_features": 6,
        },
        "m_lo": {  # selects two DIAKRINO-disfavoured features
            "performance_scores": np.array([0.70, 0.72, 0.69]),
            "performance_mean": 0.703,
            "selected_indices": np.array([2, 3]),
            "n_samples": n_samples, "n_features": 6,
        },
    }


def _call(*, use_diakrino, vector, n_samples=120):
    names = ["m_hi", "m_lo"]
    X_pool = np.zeros((n_samples, 6))
    return estimate_oracle_preferences(
        names, _evaluation(n_samples),
        pairwise_delta=0.01,
        use_tail_risk_oracle=False, tail_risk_alpha=0.33,
        use_qre_smoothing=False, qre_temperature_gamma=1.0,
        use_regret_oracle=False,
        use_stability_oracle=False, use_complexity_oracle=False,
        use_robust_oracle=False, use_diversity_oracle=False,
        diversity_oracle_mode="legacy_jaccard",
        diversity_redundancy_weight=0.6, diversity_complementarity_weight=0.35,
        X_pool=X_pool,
        diakrino_relevance_vector=vector,
        use_diakrino_relevance_oracle=use_diakrino,
        oracle_config=None,
        random_state=0,
    )


def test_relevance_oracle_scores_mass_capture():
    rel = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # features 0,1 favoured
    matrices, scores, components, _meta = _call(use_diakrino=True, vector=rel)
    assert "diakrino_relevance" in matrices and "diakrino_relevance" in scores
    # m_hi selected {0,1} -> mean 1.0 ; m_lo selected {2,3} -> mean 0.0
    assert np.allclose(scores["diakrino_relevance"], [1.0, 0.0])
    assert components["diakrino_relevance"]["applied"] is True
    # preference matrix favours m_hi over m_lo
    assert matrices["diakrino_relevance"][0, 1] > 0.5


def test_disabled_produces_no_oracle():
    rel = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    matrices, scores, components, _meta = _call(use_diakrino=False, vector=rel)
    assert "diakrino_relevance" not in matrices
    assert "diakrino_relevance" not in scores


def test_missing_vector_gates_off():
    matrices, scores, components, _meta = _call(use_diakrino=True, vector=None)
    assert "diakrino_relevance" not in matrices
    assert components["diakrino_relevance"]["applied"] is False
    assert components["diakrino_relevance"]["reason"] == "missing_diakrino_vector"


def test_min_n_gate():
    rel = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    matrices, scores, components, _meta = _call(use_diakrino=True, vector=rel, n_samples=5)
    assert "diakrino_relevance" not in matrices
    assert components["diakrino_relevance"]["applied"] is False
    assert components["diakrino_relevance"]["reason"] == "min_n_gate"

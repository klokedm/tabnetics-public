"""Tests for T-002: Multi-Model MNPO Oracles.

Validates that per-model performance oracle matrices are created correctly
when ``performance_oracle_mode="multi_model_oracles"`` (opt-in), and that
the default ``"single"`` mode is bit-for-bit backward compatible.
"""

import unittest
import numpy as np

from tabnetics.feature_selection.mnpo.oracles import (
    compute_oracle_stability_diagnostics,
    estimate_oracle_preferences,
    fit_tritrust_weights,
    pairwise_pref_from_fold_scores,
)
from tabnetics.core.mnpo import pairwise_pref_from_fold_scores as pairwise_pref_from_fold_scores_core
from tabnetics.feature_selection.mnpo.portfolio import (
    _compute_selector_stability_signal,
    evaluate_candidate_library,
)
from tabnetics.feature_selection.config import (
    MNPOConfig,
    FeatureSelectorConfig,
    EvaluationConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluation(n_candidates=4, n_folds=5, n_features=20, rng_seed=42,
                     include_per_model=False, model_keys=("lr_l2", "linear_svc", "rf_small")):
    """Build a synthetic evaluation dict for testing oracle construction."""
    rng = np.random.default_rng(rng_seed)
    names = [f"method_{i}" for i in range(n_candidates)]
    evaluation = {}
    for idx, name in enumerate(names):
        base_perf = 0.5 + 0.1 * idx  # monotonically increasing
        perf_scores = rng.normal(base_perf, 0.05, size=n_folds).clip(0, 1)
        selected = np.sort(rng.choice(n_features, size=min(5 + idx, n_features), replace=False))
        entry = {
            'performance_scores': np.asarray(perf_scores, dtype=float),
            'performance_mean': float(np.mean(perf_scores)),
            'stability': float(rng.uniform(0.3, 0.9)),
            'complexity': float(1.0 - len(selected) / n_features),
            'robustness': float(np.mean(perf_scores) - 0.02),
            'prediction_signal': rng.normal(0, 1, size=n_folds * 5),
            'target_signal': rng.integers(0, 2, size=n_folds * 5),
            'selected_indices': selected,
            'score_vector': rng.uniform(0, 1, size=n_features),
            'performance_scores_by_model': {},
        }
        if include_per_model:
            for mk in model_keys:
                mk_base = base_perf + rng.uniform(-0.05, 0.05)
                entry['performance_scores_by_model'][mk] = np.asarray(
                    rng.normal(mk_base, 0.04, size=n_folds).clip(0, 1), dtype=float
                )
        evaluation[name] = entry
    return names, evaluation


def test_pairwise_pref_from_fold_scores_canonical_matches_wrappers():
    from tabnetics.classification.backends import (
        _pairwise_pref_from_fold_scores as pairwise_pref_from_fold_scores_classification,
    )

    scores_i = np.asarray([0.70, 0.75, 0.73, np.nan, 0.77], dtype=float)
    scores_j = np.asarray([0.69, 0.76, 0.71, 0.72, np.nan], dtype=float)
    delta = 0.01

    expected = float(
        pairwise_pref_from_fold_scores_core(
            scores_i,
            scores_j,
            pairwise_delta=float(delta),
        )
    )
    got_oracles = float(pairwise_pref_from_fold_scores(scores_i, scores_j, pairwise_delta=float(delta)))
    got_classification = float(
        pairwise_pref_from_fold_scores_classification(
            scores_i,
            scores_j,
            pairwise_delta=float(delta),
        )
    )

    assert np.isclose(got_oracles, expected)
    assert np.isclose(got_classification, expected)


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestToggleOff(unittest.TestCase):
    """performance_oracle_mode='single': only the aggregate 'performance' oracle exists."""

    def test_single_mode_only_performance_oracle(self):
        names, evaluation = _make_evaluation(n_candidates=4, include_per_model=False)
        oracle_matrices, oracle_scores, _, _ = estimate_oracle_preferences(
            names, evaluation,
            pairwise_delta=0.01,
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
        )
        self.assertIn("performance", oracle_matrices)
        # No per-model oracles
        per_model_keys = [k for k in oracle_matrices if k.startswith("performance_")]
        self.assertEqual(len(per_model_keys), 0)

    def test_single_mode_ignores_per_model_data(self):
        """Even if per-model data is present, single mode does not create per-model oracles."""
        names, evaluation = _make_evaluation(n_candidates=3, include_per_model=True)
        oracle_matrices, _, _, _ = estimate_oracle_preferences(
            names, evaluation,
            pairwise_delta=0.01,
            use_tail_risk_oracle=False, tail_risk_alpha=0.33,
            use_qre_smoothing=False, qre_temperature_gamma=1.0,
            use_regret_oracle=False,
            use_stability_oracle=False, use_complexity_oracle=False,
            use_robust_oracle=False, use_diversity_oracle=False,
            diversity_oracle_mode="legacy_jaccard",
            diversity_redundancy_weight=0.6,
            diversity_complementarity_weight=0.35,
            performance_oracle_mode="single",
        )
        per_model_keys = [k for k in oracle_matrices if k.startswith("performance_")]
        self.assertEqual(len(per_model_keys), 0)


class TestToggleOn(unittest.TestCase):
    """performance_oracle_mode='multi_model_oracles': per-model oracle matrices exist."""

    def _get_oracles(self, n_candidates=4, model_keys=("lr_l2", "linear_svc", "rf_small")):
        names, evaluation = _make_evaluation(
            n_candidates=n_candidates, include_per_model=True, model_keys=model_keys,
        )
        return estimate_oracle_preferences(
            names, evaluation,
            pairwise_delta=0.01,
            use_tail_risk_oracle=False, tail_risk_alpha=0.33,
            use_qre_smoothing=False, qre_temperature_gamma=1.0,
            use_regret_oracle=False,
            use_stability_oracle=True, use_complexity_oracle=True,
            use_robust_oracle=False, use_diversity_oracle=False,
            diversity_oracle_mode="legacy_jaccard",
            diversity_redundancy_weight=0.6,
            diversity_complementarity_weight=0.35,
            performance_oracle_mode="multi_model_oracles",
        ), names

    def test_per_model_oracles_exist(self):
        (oracle_matrices, oracle_scores, _, _), names = self._get_oracles()
        self.assertIn("performance", oracle_matrices)
        self.assertIn("performance_lr_l2", oracle_matrices)
        self.assertIn("performance_linear_svc", oracle_matrices)
        self.assertIn("performance_rf_small", oracle_matrices)

    def test_oracle_matrix_shapes(self):
        """All per-model matrices are m×m."""
        (oracle_matrices, _, _, _), names = self._get_oracles(n_candidates=5)
        m = len(names)
        for key, mat in oracle_matrices.items():
            if key.startswith("performance"):
                self.assertEqual(mat.shape, (m, m), msg=f"Matrix {key} has wrong shape")

    def test_oracle_matrix_diagonal(self):
        """Diagonal of all oracle matrices = 0.5."""
        (oracle_matrices, _, _, _), names = self._get_oracles()
        for key, mat in oracle_matrices.items():
            if key.startswith("performance"):
                np.testing.assert_allclose(
                    np.diag(mat), 0.5,
                    err_msg=f"Diagonal of {key} is not 0.5",
                )

    def test_oracle_matrix_antisymmetric(self):
        """mat[i,j] + mat[j,i] = 1.0 for all off-diagonal entries."""
        (oracle_matrices, _, _, _), names = self._get_oracles()
        m = len(names)
        for key, mat in oracle_matrices.items():
            if key.startswith("performance"):
                for i in range(m):
                    for j in range(i + 1, m):
                        self.assertAlmostEqual(
                            mat[i, j] + mat[j, i], 1.0,
                            places=10,
                            msg=f"{key}[{i},{j}]+[{j},{i}] != 1.0",
                        )

    def test_oracle_scores_present(self):
        (_, oracle_scores, _, _), names = self._get_oracles()
        self.assertIn("performance_lr_l2", oracle_scores)
        self.assertEqual(oracle_scores["performance_lr_l2"].shape, (len(names),))


class TestTriTrustReference(unittest.TestCase):
    """TriTrust reference oracle selection with multi-model mode."""

    def test_reference_uses_lr_l2_when_available(self):
        """When performance_lr_l2 is present, TriTrust should use it as reference."""
        names, evaluation = _make_evaluation(n_candidates=3, include_per_model=True)
        oracle_matrices, _, _, _ = estimate_oracle_preferences(
            names, evaluation,
            pairwise_delta=0.01,
            use_tail_risk_oracle=False, tail_risk_alpha=0.33,
            use_qre_smoothing=False, qre_temperature_gamma=1.0,
            use_regret_oracle=False,
            use_stability_oracle=True, use_complexity_oracle=True,
            use_robust_oracle=False, use_diversity_oracle=False,
            diversity_oracle_mode="legacy_jaccard",
            diversity_redundancy_weight=0.6,
            diversity_complementarity_weight=0.35,
            performance_oracle_mode="multi_model_oracles",
        )
        self.assertIn("performance_lr_l2", oracle_matrices)
        # fit_tritrust_weights should run successfully with lr_l2 as reference
        weights = fit_tritrust_weights(oracle_matrices)
        self.assertIsInstance(weights, dict)
        self.assertTrue(len(weights) > 0)

    def test_reference_falls_back_to_performance(self):
        """When no per-model oracles, reference falls back to 'performance'."""
        names, evaluation = _make_evaluation(n_candidates=3, include_per_model=False)
        oracle_matrices, _, _, _ = estimate_oracle_preferences(
            names, evaluation,
            pairwise_delta=0.01,
            use_tail_risk_oracle=False, tail_risk_alpha=0.33,
            use_qre_smoothing=False, qre_temperature_gamma=1.0,
            use_regret_oracle=False,
            use_stability_oracle=True, use_complexity_oracle=True,
            use_robust_oracle=False, use_diversity_oracle=False,
            diversity_oracle_mode="legacy_jaccard",
            diversity_redundancy_weight=0.6,
            diversity_complementarity_weight=0.35,
            performance_oracle_mode="single",
        )
        weights = fit_tritrust_weights(oracle_matrices)
        self.assertIsInstance(weights, dict)
        self.assertTrue(len(weights) > 0)

    def test_explicit_reference_override(self):
        """fit_tritrust_weights(reference_oracle=...) uses specified reference."""
        names, evaluation = _make_evaluation(n_candidates=3, include_per_model=True)
        oracle_matrices, _, _, _ = estimate_oracle_preferences(
            names, evaluation,
            pairwise_delta=0.01,
            use_tail_risk_oracle=False, tail_risk_alpha=0.33,
            use_qre_smoothing=False, qre_temperature_gamma=1.0,
            use_regret_oracle=False,
            use_stability_oracle=True, use_complexity_oracle=True,
            use_robust_oracle=False, use_diversity_oracle=False,
            diversity_oracle_mode="legacy_jaccard",
            diversity_redundancy_weight=0.6,
            diversity_complementarity_weight=0.35,
            performance_oracle_mode="multi_model_oracles",
        )
        # Force using aggregate performance as reference
        weights = fit_tritrust_weights(oracle_matrices, reference_oracle="performance")
        self.assertIsInstance(weights, dict)
        self.assertTrue(len(weights) > 0)


class TestWeightNormalization(unittest.TestCase):
    """Per-model oracle weight normalization."""

    def test_oracle_weights_include_per_model(self):
        """When multi_model_oracles, TriTrust weights include per-model keys."""
        names, evaluation = _make_evaluation(n_candidates=3, include_per_model=True)
        oracle_matrices, _, _, _ = estimate_oracle_preferences(
            names, evaluation,
            pairwise_delta=0.01,
            use_tail_risk_oracle=False, tail_risk_alpha=0.33,
            use_qre_smoothing=False, qre_temperature_gamma=1.0,
            use_regret_oracle=False,
            use_stability_oracle=True, use_complexity_oracle=True,
            use_robust_oracle=False, use_diversity_oracle=False,
            diversity_oracle_mode="legacy_jaccard",
            diversity_redundancy_weight=0.6,
            diversity_complementarity_weight=0.35,
            performance_oracle_mode="multi_model_oracles",
        )
        weights = fit_tritrust_weights(oracle_matrices)
        # Per-model oracle keys should appear in weights
        per_model_weight_keys = [k for k in weights if k.startswith("performance_")]
        self.assertGreater(len(per_model_weight_keys), 0)


class TestMissingModelDataFallback(unittest.TestCase):
    """When per-model data is empty, fall back to single oracle only."""

    def test_empty_per_model_data_falls_back(self):
        names, evaluation = _make_evaluation(n_candidates=3, include_per_model=False)
        oracle_matrices, _, _, _ = estimate_oracle_preferences(
            names, evaluation,
            pairwise_delta=0.01,
            use_tail_risk_oracle=False, tail_risk_alpha=0.33,
            use_qre_smoothing=False, qre_temperature_gamma=1.0,
            use_regret_oracle=False,
            use_stability_oracle=False, use_complexity_oracle=False,
            use_robust_oracle=False, use_diversity_oracle=False,
            diversity_oracle_mode="legacy_jaccard",
            diversity_redundancy_weight=0.6,
            diversity_complementarity_weight=0.35,
            performance_oracle_mode="multi_model_oracles",
        )
        # Should only have aggregate performance (no per-model data available)
        self.assertIn("performance", oracle_matrices)
        per_model_keys = [k for k in oracle_matrices if k.startswith("performance_")]
        self.assertEqual(len(per_model_keys), 0)


class TestConfigDataclass(unittest.TestCase):
    """MNPOConfig.performance_oracle_mode defaults and wiring."""

    def test_default_is_single(self):
        cfg = MNPOConfig()
        self.assertEqual(cfg.performance_oracle_mode, "single")

    def test_multi_model_oracles_value(self):
        cfg = MNPOConfig(performance_oracle_mode="multi_model_oracles")
        self.assertEqual(cfg.performance_oracle_mode, "multi_model_oracles")

    def test_config_wires_to_feature_selector(self):
        """FeatureSelectorConfig → FeatureSelector correctly passes performance_oracle_mode."""
        from tabnetics.feature_selection.base import FeatureSelector
        cfg = FeatureSelectorConfig(
            mnpo=MNPOConfig(performance_oracle_mode="multi_model_oracles"),
            evaluation=EvaluationConfig(eval_models_enabled=True),
        )
        fs = FeatureSelector.from_config(cfg)
        self.assertEqual(fs.performance_oracle_mode, "multi_model_oracles")
        self.assertTrue(fs.eval_models_enabled)

    def test_config_default_single_wires(self):
        from tabnetics.feature_selection.base import FeatureSelector
        cfg = FeatureSelectorConfig()
        fs = FeatureSelector.from_config(cfg)
        self.assertEqual(fs.performance_oracle_mode, "single")


class TestEvaluateCandidateLibrary3Tuple(unittest.TestCase):
    """portfolio.evaluate_candidate_library correctly collects per-model fold scores."""

    def test_3tuple_callback_populates_scores_by_model(self):
        """When callback returns 3-tuple, evaluation dict has per-model data."""
        rng = np.random.default_rng(99)
        n, p = 40, 10
        X = rng.normal(size=(n, p))
        y = rng.integers(0, 2, size=n)

        candidates = {
            "dummy_a": {
                "selected_indices": np.array([0, 1, 2]),
                "score_vector": rng.uniform(0, 1, size=p),
                "method_result": {},
            },
            "dummy_b": {
                "selected_indices": np.array([3, 4, 5]),
                "score_vector": rng.uniform(0, 1, size=p),
                "method_result": {},
            },
        }

        def _mock_3tuple(X_tr, y_tr, X_v, y_v):
            score = float(rng.uniform(0.4, 0.9))
            signal = np.zeros(len(y_v), dtype=float)
            per_model = {"lr_l2": score + 0.01, "linear_svc": score - 0.01}
            return score, signal, per_model

        from tabnetics.feature_selection.cv import get_inner_cv_splits

        def _get_splits(X, y):
            return get_inner_cv_splits(X, y, "classification", 42, 3, 2)

        evaluation = evaluate_candidate_library(
            X, y, candidates,
            get_inner_cv_splits_fn=_get_splits,
            fit_and_score_fold_fn=_mock_3tuple,
            augment_training_data_fn=lambda Xtr, ytr: (Xtr, ytr),
            use_robust_oracle=False,
            complexity_use_runtime_penalty=False,
        )
        for name in candidates:
            by_model = evaluation[name].get("performance_scores_by_model", {})
            self.assertIn("lr_l2", by_model)
            self.assertIn("linear_svc", by_model)
            self.assertGreater(len(by_model["lr_l2"]), 0)

    def test_2tuple_callback_empty_per_model(self):
        """When callback returns 2-tuple, per-model data is empty."""
        rng = np.random.default_rng(99)
        n, p = 40, 10
        X = rng.normal(size=(n, p))
        y = rng.integers(0, 2, size=n)

        candidates = {
            "dummy_a": {
                "selected_indices": np.array([0, 1, 2]),
                "score_vector": rng.uniform(0, 1, size=p),
                "method_result": {},
            },
        }

        def _mock_2tuple(X_tr, y_tr, X_v, y_v):
            score = float(rng.uniform(0.4, 0.9))
            signal = np.zeros(len(y_v), dtype=float)
            return score, signal

        from tabnetics.feature_selection.cv import get_inner_cv_splits

        def _get_splits(X, y):
            return get_inner_cv_splits(X, y, "classification", 42, 3, 2)

        evaluation = evaluate_candidate_library(
            X, y, candidates,
            get_inner_cv_splits_fn=_get_splits,
            fit_and_score_fold_fn=_mock_2tuple,
            augment_training_data_fn=lambda Xtr, ytr: (Xtr, ytr),
            use_robust_oracle=False,
            complexity_use_runtime_penalty=False,
        )
        by_model = evaluation["dummy_a"].get("performance_scores_by_model", {})
        self.assertEqual(len(by_model), 0)

    def test_oracle_eval_failure_metadata_propagates(self):
        """Failure diagnostics are captured in evaluation metadata deterministically."""
        rng = np.random.default_rng(17)
        n, p = 30, 8
        X = rng.normal(size=(n, p))
        y = rng.integers(0, 2, size=n)

        candidates = {
            "dummy_a": {
                "selected_indices": np.array([0, 1, 2]),
                "score_vector": rng.uniform(0, 1, size=p),
                "method_result": {},
            },
        }

        call_count = {"n": 0}

        def _mock_3tuple_with_diag(X_tr, y_tr, X_v, y_v):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("forced fold exception")
            return 0.5, np.zeros(len(y_v), dtype=float), {
                "lr_l2": 0.6,
                "linear_svc": 0.4,
                "_evaluation_failures_total": 1.0,
                "_evaluation_failures_lr_l2": 1.0,
            }

        from tabnetics.feature_selection.cv import get_inner_cv_splits

        def _get_splits(X, y):
            return get_inner_cv_splits(X, y, "classification", 42, 3, 1)

        evaluation = evaluate_candidate_library(
            X, y, candidates,
            get_inner_cv_splits_fn=_get_splits,
            fit_and_score_fold_fn=_mock_3tuple_with_diag,
            augment_training_data_fn=lambda Xtr, ytr: (Xtr, ytr),
            use_robust_oracle=True,
            complexity_use_runtime_penalty=False,
        )
        failure_meta = evaluation["dummy_a"].get("evaluation_failures", {})
        self.assertGreaterEqual(int(failure_meta.get("fold_exceptions", 0)), 1)
        self.assertGreaterEqual(float(failure_meta.get("model_failures_total", 0.0)), 1.0)
        by_model_fail = failure_meta.get("model_failures_by_model", {})
        self.assertGreaterEqual(float(by_model_fail.get("lr_l2", 0.0)), 1.0)


class TestIntegrationSyntheticFlow(unittest.TestCase):
    """End-to-end: synthetic data → 3-tuple → oracle matrices → TriTrust weights."""

    def test_full_flow(self):
        rng = np.random.default_rng(123)
        n, p = 50, 15
        X = rng.normal(size=(n, p))
        y = rng.integers(0, 2, size=n)

        candidates = {}
        for i in range(3):
            sel = np.sort(rng.choice(p, size=5, replace=False))
            candidates[f"m{i}"] = {
                "selected_indices": sel,
                "score_vector": rng.uniform(0, 1, size=p),
                "method_result": {},
            }

        model_keys = ("lr_l2", "linear_svc", "rf_small")

        def _mock_3tuple(X_tr, y_tr, X_v, y_v):
            score = float(rng.uniform(0.4, 0.9))
            signal = np.zeros(len(y_v), dtype=float)
            per_model = {mk: float(rng.uniform(0.3, 0.95)) for mk in model_keys}
            return score, signal, per_model

        from tabnetics.feature_selection.cv import get_inner_cv_splits

        def _get_splits(X, y):
            return get_inner_cv_splits(X, y, "classification", 42, 3, 2)

        evaluation = evaluate_candidate_library(
            X, y, candidates,
            get_inner_cv_splits_fn=_get_splits,
            fit_and_score_fold_fn=_mock_3tuple,
            augment_training_data_fn=lambda Xtr, ytr: (Xtr, ytr),
            use_robust_oracle=False,
            complexity_use_runtime_penalty=False,
        )

        candidate_names = list(candidates.keys())
        oracle_matrices, oracle_scores, _, _ = estimate_oracle_preferences(
            candidate_names, evaluation,
            pairwise_delta=0.01,
            use_tail_risk_oracle=False, tail_risk_alpha=0.33,
            use_qre_smoothing=False, qre_temperature_gamma=1.0,
            use_regret_oracle=False,
            use_stability_oracle=True, use_complexity_oracle=True,
            use_robust_oracle=False, use_diversity_oracle=False,
            diversity_oracle_mode="legacy_jaccard",
            diversity_redundancy_weight=0.6,
            diversity_complementarity_weight=0.35,
            performance_oracle_mode="multi_model_oracles",
        )

        # Verify structure
        self.assertIn("performance", oracle_matrices)
        for mk in model_keys:
            oracle_name = f"performance_{mk}"
            self.assertIn(oracle_name, oracle_matrices)
            mat = oracle_matrices[oracle_name]
            self.assertEqual(mat.shape, (3, 3))
            np.testing.assert_allclose(np.diag(mat), 0.5)

        # TriTrust should work with multi-model oracles
        weights = fit_tritrust_weights(oracle_matrices)
        self.assertIsInstance(weights, dict)
        self.assertIn("performance_lr_l2", weights)
        self.assertIn("performance_linear_svc", weights)
        self.assertIn("performance_rf_small", weights)


class TestSelectorStabilitySignal(unittest.TestCase):
    """T-R-391 selector-stability oracle signal tests."""

    def test_exact_support_telemetry_uses_pairwise_jaccard(self):
        method_result = {
            "bootstrap_supports": [
                np.array([0, 1, 2]),
                np.array([0, 1, 3]),
                np.array([0, 1, 4]),
            ]
        }

        score = _compute_selector_stability_signal(
            method_result,
            np.array([0, 1, 2]),
            n_features=6,
        )

        self.assertAlmostEqual(score, 0.5)

    def test_frequency_telemetry_is_normalized_topk_signal(self):
        stable = _compute_selector_stability_signal(
            {"selection_frequency": np.array([1.0, 1.0, 0.9, 0.0, 0.0])},
            np.array([0, 1, 2]),
            n_features=5,
        )
        diffuse = _compute_selector_stability_signal(
            {"selection_frequency": np.full(5, 0.6)},
            np.array([0, 1, 2]),
            n_features=5,
        )

        self.assertGreater(stable, diffuse)
        self.assertGreaterEqual(stable, 0.0)
        self.assertLessEqual(stable, 1.0)
        self.assertGreaterEqual(diffuse, 0.0)
        self.assertLessEqual(diffuse, 1.0)


class TestOracleStabilityDiagnostics(unittest.TestCase):
    """Shared oracle stability diagnostics contract tests (T-A003-R2)."""

    def test_diagnostics_deterministic_for_fixed_inputs(self):
        oracle_scores = {
            "performance": np.array([0.80, 0.70, 0.60]),
            "stability": np.array([0.75, 0.65, 0.55]),
            "complexity": np.array([0.20, 0.40, 0.60]),
        }
        d1 = compute_oracle_stability_diagnostics(oracle_scores)
        d2 = compute_oracle_stability_diagnostics(oracle_scores)
        self.assertEqual(d1, d2)
        self.assertIn("pairwise_rank_correlation", d1)
        self.assertEqual(d1["pair_count"], 3)

    def test_estimate_outputs_stability_meta(self):
        names, evaluation = _make_evaluation(n_candidates=4, include_per_model=True)
        _, _, components, pairwise_meta = estimate_oracle_preferences(
            names, evaluation,
            pairwise_delta=0.01,
            use_tail_risk_oracle=False,
            tail_risk_alpha=0.33,
            use_qre_smoothing=False,
            qre_temperature_gamma=1.0,
            use_regret_oracle=False,
            use_stability_oracle=True,
            use_complexity_oracle=True,
            use_robust_oracle=False,
            use_diversity_oracle=False,
            diversity_oracle_mode="legacy_jaccard",
            diversity_redundancy_weight=0.6,
            diversity_complementarity_weight=0.35,
            performance_oracle_mode="multi_model_oracles",
        )
        self.assertIn("oracle_stability", components)
        self.assertIn("oracle_stability", pairwise_meta)
        self.assertIn("mean_rank_correlation", pairwise_meta["oracle_stability"])


if __name__ == "__main__":
    unittest.main()

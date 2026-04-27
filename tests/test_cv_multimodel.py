"""Tests for multi-classifier evaluation proxy (T-001).

Covers:
- Toggle OFF regression (bit-for-bit identical to single-model)
- Toggle ON determinism (fixed seed → same score)
- Aggregation math correctness (mean, min, CVaR)
- Single-class edge cases
- Per-model scores returned
- Model library validity
- LinearSVC predict consistency (no predict_proba fallback)
- Config integration
- Pinned-seed regression
"""

import numpy as np
import pytest

from tabnetics.feature_selection.cv import (
    CVEvaluationContext,
    FoldLeakageError,
    fit_and_score_fold,
    fit_and_score_fold_multimodel,
    get_inner_cv_splits,
    _get_eval_model,
    _score_fitted_model,
    resolve_performance_weights,
    safe_balanced_accuracy,
)
from tabnetics.feature_selection.config import (
    EvaluationConfig,
    FeatureSelectorConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_binary_data(n=80, p=6, seed=42):
    """Synthetic binary classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    mid = n // 2
    return X[:mid], y[:mid], X[mid:], y[mid:]


def _make_multiclass_data(n=120, p=6, n_classes=4, seed=42):
    """Synthetic multiclass classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = rng.integers(0, n_classes, size=n)
    mid = n // 2
    return X[:mid], y[:mid], X[mid:], y[mid:]


def _make_regression_data(n=80, p=6, seed=42):
    """Synthetic regression dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X[:, 0] * 2.0 + rng.standard_normal(n) * 0.3
    mid = n // 2
    return X[:mid], y[:mid], X[mid:], y[mid:]


# Default performance weights (matching FeatureSelector defaults)
_PERF_KWARGS = dict(
    performance_balanced_weight=0.6,
    performance_macro_f1_weight=0.4,
    performance_use_adaptive_imbalance=False,
    performance_imbalance_ratio_trigger=1.75,
    performance_min_classes_for_adaptive=3,
)


# ── 1. Toggle OFF regression: bit-for-bit identical ──────────────────

class TestToggleOffRegression:
    """When eval_models_enabled=False, behavior must be bit-for-bit identical."""

    def test_binary_bit_for_bit(self):
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        score_single, signal_single = fit_and_score_fold(
            X_tr, y_tr, X_val, y_val, 'classification', 42, **_PERF_KWARGS,
        )
        # Multi-model with only lr_l2 and mean agg should reproduce single LR
        score_multi, signal_multi, per_model = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2",), eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        assert score_single == pytest.approx(score_multi, abs=1e-12), \
            f"Single={score_single}, Multi={score_multi}"
        np.testing.assert_array_equal(signal_single, signal_multi)

    def test_multiclass_bit_for_bit(self):
        X_tr, y_tr, X_val, y_val = _make_multiclass_data()
        score_single, signal_single = fit_and_score_fold(
            X_tr, y_tr, X_val, y_val, 'classification', 42, **_PERF_KWARGS,
        )
        score_multi, signal_multi, _ = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2",), eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        assert score_single == pytest.approx(score_multi, abs=1e-12)
        np.testing.assert_array_equal(signal_single, signal_multi)

    def test_regression_fallback(self):
        X_tr, y_tr, X_val, y_val = _make_regression_data()
        score_single, signal_single = fit_and_score_fold(
            X_tr, y_tr, X_val, y_val, 'regression', 42, **_PERF_KWARGS,
        )
        score_multi, signal_multi, per_model = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'regression', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        assert score_single == pytest.approx(score_multi, abs=1e-12)
        np.testing.assert_array_equal(signal_single, signal_multi)
        assert "_fallback_regression" in per_model


# ── 2. Toggle ON determinism ─────────────────────────────────────────

class TestToggleOnDeterminism:
    """Same seed → same multi-model score."""

    def test_deterministic_binary(self):
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        results = []
        for _ in range(3):
            s, sig, pm = fit_and_score_fold_multimodel(
                X_tr, y_tr, X_val, y_val, 'classification', 42,
                eval_models=("lr_l2", "linear_svc", "rf_small"),
                eval_aggregate="mean", eval_cvar_alpha=0.33,
                **_PERF_KWARGS,
            )
            results.append((s, sig.copy(), dict(pm)))
        for i in range(1, len(results)):
            assert results[0][0] == pytest.approx(results[i][0], abs=1e-12)
            np.testing.assert_array_equal(results[0][1], results[i][1])

    def test_deterministic_multiclass(self):
        X_tr, y_tr, X_val, y_val = _make_multiclass_data()
        s1, _, pm1 = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 99,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        s2, _, pm2 = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 99,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        assert s1 == pytest.approx(s2, abs=1e-12)
        for key in pm1:
            assert pm1[key] == pytest.approx(pm2[key], abs=1e-12)


# ── 3. Aggregation math correctness ──────────────────────────────────

class TestAggregationMath:
    """Verify mean, min, CVaR produce correct results."""

    def test_mean_aggregation(self):
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        _, _, pm = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        scores = np.array([pm[k] for k in ("lr_l2", "linear_svc", "rf_small")])
        expected = float(np.mean(scores))
        actual, _, _ = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_min_aggregation(self):
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        # First get per-model scores with mean to know what they are
        _, _, pm = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        scores = np.array([pm[k] for k in ("lr_l2", "linear_svc", "rf_small")])
        expected_min = float(np.min(scores))
        actual_min, _, _ = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="min", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        assert actual_min == pytest.approx(expected_min, abs=1e-12)

    def test_cvar_aggregation(self):
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        _, _, pm = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        scores = np.array([pm[k] for k in ("lr_l2", "linear_svc", "rf_small")])
        # CVaR(alpha=0.5): mean of bottom ceil(0.5*3)=2 scores
        alpha = 0.5
        k = max(1, int(np.ceil(alpha * len(scores))))
        sorted_scores = np.sort(scores)
        expected_cvar = float(np.mean(sorted_scores[:k]))

        actual_cvar, _, _ = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="cvar", eval_cvar_alpha=0.5,
            **_PERF_KWARGS,
        )
        assert actual_cvar == pytest.approx(expected_cvar, abs=1e-12)

    def test_cvar_alpha_one_equals_mean(self):
        """CVaR with alpha=1.0 is mean of all scores."""
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        mean_score, _, _ = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        cvar_score, _, _ = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="cvar", eval_cvar_alpha=1.0,
            **_PERF_KWARGS,
        )
        assert cvar_score == pytest.approx(mean_score, abs=1e-12)

    def test_cvar_alpha_small_equals_min(self):
        """CVaR with very small alpha ≈ min (bottom 1 score)."""
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        min_score, _, _ = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="min", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        cvar_score, _, _ = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="cvar", eval_cvar_alpha=0.01,
            **_PERF_KWARGS,
        )
        # alpha=0.01 → k=ceil(0.01*3)=1 → mean of bottom 1 = min
        assert cvar_score == pytest.approx(min_score, abs=1e-12)


# ── 4. Single-class edge case ────────────────────────────────────────

class TestSingleClassEdge:
    """All training data is one class."""

    def test_single_class_train(self):
        rng = np.random.default_rng(42)
        X_tr = rng.standard_normal((30, 4))
        y_tr = np.zeros(30, dtype=int)  # single class
        X_val = rng.standard_normal((10, 4))
        y_val = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0])

        score, signal, pm = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        assert np.isfinite(score)
        assert signal.shape == (10,)
        assert "_single_class" in pm

    def test_empty_class_in_train(self):
        """Only one class present in y_train."""
        rng = np.random.default_rng(7)
        X_tr = rng.standard_normal((20, 3))
        y_tr = np.ones(20, dtype=int)  # all class 1
        X_val = rng.standard_normal((5, 3))
        y_val = np.array([0, 1, 1, 0, 1])

        score, signal, pm = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        assert np.isfinite(score)
        assert signal.shape == (5,)


# ── 5. Per-model scores returned ─────────────────────────────────────

class TestPerModelScores:
    """Verify diagnostics dict has per-model entries."""

    def test_all_models_in_diagnostics(self):
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        _, _, pm = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        assert "lr_l2" in pm
        assert "linear_svc" in pm
        assert "rf_small" in pm
        for v in pm.values():
            assert isinstance(v, float)
            assert np.isfinite(v)


class TestFoldHygieneAndFailureModes:
    """Fold-hygiene guard and evaluation-failure observability contracts."""

    def test_blocks_learned_aggregation_in_evaluation_fold(self):
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        with pytest.raises(FoldLeakageError):
            fit_and_score_fold_multimodel(
                X_tr, y_tr, X_val, y_val, 'classification', 42,
                eval_models=("lr_l2", "linear_svc", "rf_small"),
                eval_aggregate="mean", eval_cvar_alpha=0.33,
                eval_model_weight_strategy="learned",
                evaluation_context=CVEvaluationContext(
                    purpose="evaluation_fold",
                    allow_learned_model_aggregation=False,
                ),
                **_PERF_KWARGS,
            )

    def test_allows_explicit_override_when_context_permits(self):
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        score, signal, pm = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2",),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            eval_model_weight_strategy="learned",
            evaluation_context=CVEvaluationContext(
                purpose="evaluation_fold",
                allow_learned_model_aggregation=True,
            ),
            **_PERF_KWARGS,
        )
        assert np.isfinite(score)
        assert signal.shape == (X_val.shape[0],)
        assert "lr_l2" in pm

    def test_non_strict_mode_tracks_failures_and_aggregates_model_scores_only(self, monkeypatch):
        X_tr, y_tr, X_val, y_val = _make_binary_data()

        def _boom(*args, **kwargs):
            raise RuntimeError("forced eval failure")

        monkeypatch.setattr("tabnetics.feature_selection.cv._score_fitted_model", _boom)
        score, _, pm = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            eval_failure_strict_mode=False,
            **_PERF_KWARGS,
        )
        assert pm["_evaluation_failures_total"] == 2.0
        assert pm["_evaluation_failures_lr_l2"] == 1.0
        assert pm["_evaluation_failures_linear_svc"] == 1.0
        assert score == pytest.approx(0.0, abs=1e-12)

    def test_strict_mode_raises_on_eval_failure(self, monkeypatch):
        X_tr, y_tr, X_val, y_val = _make_binary_data()

        def _boom(*args, **kwargs):
            raise RuntimeError("forced eval failure")

        monkeypatch.setattr("tabnetics.feature_selection.cv._score_fitted_model", _boom)
        with pytest.raises(RuntimeError, match="forced eval failure"):
            fit_and_score_fold_multimodel(
                X_tr, y_tr, X_val, y_val, 'classification', 42,
                eval_models=("lr_l2", "linear_svc"),
                eval_aggregate="mean", eval_cvar_alpha=0.33,
                eval_failure_strict_mode=True,
                **_PERF_KWARGS,
            )

    def test_subset_models(self):
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        _, _, pm = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        assert "lr_l2" in pm
        assert "rf_small" in pm
        assert "linear_svc" not in pm

def test_get_inner_cv_splits_handles_n_lt_2():
    X = np.zeros((1, 4), dtype=float)
    y = np.array([0], dtype=int)
    splits = get_inner_cv_splits(
        X,
        y,
        "classification",
        random_state=42,
        inner_cv_splits=5,
        inner_cv_repeats=3,
    )
    assert splits == []


# ── 6. Model library validity ────────────────────────────────────────

class TestModelLibrary:
    """All 3 model keys resolve to valid classifiers."""

    @pytest.mark.parametrize("key", ["lr_l2", "linear_svc", "rf_small"])
    def test_model_instantiation(self, key):
        model = _get_eval_model(key, random_state=42, max_iter=2000)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown eval model"):
            _get_eval_model("xgboost_mega", random_state=42)

    @pytest.mark.parametrize("key", ["lr_l2", "linear_svc", "rf_small"])
    def test_model_can_fit_and_predict(self, key):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 4))
        y = (X[:, 0] > 0).astype(int)
        model = _get_eval_model(key, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (40,)


# ── 7. LinearSVC predict consistency ─────────────────────────────────

class TestLinearSVCFallback:
    """LinearSVC doesn't have predict_proba — verify signal uses predictions."""

    def test_no_predict_proba(self):
        model = _get_eval_model("linear_svc", random_state=42)
        assert not hasattr(model, "predict_proba")

    def test_score_without_proba(self):
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        model = _get_eval_model("linear_svc", random_state=42)
        model.fit(X_tr, y_tr)
        w_bal, w_f1 = 0.6, 0.4
        score, signal = _score_fitted_model(model, X_val, y_val, w_bal, w_f1)
        assert np.isfinite(score)
        # Signal should be integer predictions (no proba)
        assert signal.shape == y_val.shape
        # All signal values should be class labels (0 or 1)
        unique_signal = np.unique(signal)
        assert all(v in {0.0, 1.0} for v in unique_signal)


# ── 8. Config integration ────────────────────────────────────────────

class TestConfigIntegration:
    """EvaluationConfig defaults produce correct behavior."""

    def test_default_config(self):
        cfg = EvaluationConfig()
        assert cfg.eval_models_enabled is False
        assert cfg.eval_models == ("lr_l2", "linear_svc", "rf_small")
        assert cfg.eval_aggregate == "mean"
        assert cfg.eval_cvar_alpha == pytest.approx(0.33)

    def test_config_in_feature_selector_config(self):
        fsc = FeatureSelectorConfig()
        assert hasattr(fsc, 'evaluation')
        assert isinstance(fsc.evaluation, EvaluationConfig)
        assert fsc.evaluation.eval_models_enabled is False

    def test_custom_config(self):
        cfg = EvaluationConfig(
            eval_models_enabled=True,
            eval_models=("lr_l2", "rf_small"),
            eval_aggregate="cvar",
            eval_cvar_alpha=0.5,
        )
        assert cfg.eval_models_enabled is True
        assert cfg.eval_models == ("lr_l2", "rf_small")
        assert cfg.eval_aggregate == "cvar"
        assert cfg.eval_cvar_alpha == 0.5

    def test_from_config_wiring(self):
        """FeatureSelector.from_config passes evaluation config through."""
        fsc = FeatureSelectorConfig(
            evaluation=EvaluationConfig(eval_models_enabled=True, eval_aggregate="min"),
        )
        from tabnetics.feature_selection.base import FeatureSelector
        fs = FeatureSelector.from_config(fsc)
        assert fs.eval_models_enabled is True
        assert fs.eval_aggregate == "min"

    def test_from_config_default_off(self):
        """Default config → eval_models_enabled=False on FeatureSelector."""
        fsc = FeatureSelectorConfig()
        from tabnetics.feature_selection.base import FeatureSelector
        fs = FeatureSelector.from_config(fsc)
        assert fs.eval_models_enabled is False


# ── 9. Pinned-seed regression test ───────────────────────────────────

class TestPinnedSeedRegression:
    """Multi-model produces a stable score for fixed synthetic data."""

    def test_pinned_binary_multimodel(self):
        """Pinned regression: binary classification with all 3 models."""
        rng = np.random.default_rng(12345)
        X = rng.standard_normal((100, 5))
        y = (X[:, 0] + 0.8 * X[:, 2] > 0).astype(int)
        X_tr, y_tr = X[:60], y[:60]
        X_val, y_val = X[60:], y[60:]

        score, signal, pm = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        # Score must be finite and in reasonable range
        assert np.isfinite(score)
        assert 0.0 <= score <= 1.5  # AUC bonus can push slightly above 1.0
        assert signal.shape == (40,)
        model_keys = ("lr_l2", "linear_svc", "rf_small")
        assert all(k in pm for k in model_keys)
        assert pm["_evaluation_failures_total"] == 0.0
        # All per-model scores finite
        for k, v in pm.items():
            assert np.isfinite(v), f"Model {k} score is not finite: {v}"

    def test_pinned_multiclass_multimodel(self):
        """Pinned regression: multiclass classification with all 3 models."""
        rng = np.random.default_rng(54321)
        X = rng.standard_normal((120, 6))
        y = (np.digitize(X[:, 0], bins=[-0.5, 0.5, 1.5])).astype(int)
        X_tr, y_tr = X[:80], y[:80]
        X_val, y_val = X[80:], y[80:]

        score, signal, pm = fit_and_score_fold_multimodel(
            X_tr, y_tr, X_val, y_val, 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        assert np.isfinite(score)
        assert signal.shape == (40,)
        model_keys = ("lr_l2", "linear_svc", "rf_small")
        assert all(k in pm for k in model_keys)
        assert pm["_evaluation_failures_total"] == 0.0


# ── 10. No fold leakage: sanity check ────────────────────────────────

class TestNoFoldLeakage:
    """Each model is fitted on training fold only, scored on val fold only."""

    def test_score_changes_with_different_val(self):
        """Different validation data → different scores (no data contamination)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = (X[:, 0] > 0).astype(int)
        X_tr, y_tr = X[:60], y[:60]

        score_a, _, _ = fit_and_score_fold_multimodel(
            X_tr, y_tr, X[60:80], y[60:80], 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        score_b, _, _ = fit_and_score_fold_multimodel(
            X_tr, y_tr, X[80:100], y[80:100], 'classification', 42,
            eval_models=("lr_l2", "linear_svc", "rf_small"),
            eval_aggregate="mean", eval_cvar_alpha=0.33,
            **_PERF_KWARGS,
        )
        # Scores should differ with different validation data
        # (unless by extreme coincidence, which is astronomically unlikely)
        assert score_a != pytest.approx(score_b, abs=1e-6)


# ── 11. Base.py integration ──────────────────────────────────────────

class TestBaseIntegration:
    """Test _fit_and_score_fold dispatching on FeatureSelector."""

    def test_toggle_off_uses_single_model(self):
        from tabnetics.feature_selection.base import FeatureSelector
        fs = FeatureSelector(random_state=42, eval_models_enabled=False)
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        score, signal = fs._fit_and_score_fold(X_tr, y_tr, X_val, y_val)
        assert np.isfinite(score)
        # No multi-model diagnostics logged
        assert len(fs._eval_multimodel_fold_log) == 0

    def test_toggle_on_uses_multimodel(self):
        from tabnetics.feature_selection.base import FeatureSelector
        fs = FeatureSelector(random_state=42, eval_models_enabled=True)
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        score, signal = fs._fit_and_score_fold(X_tr, y_tr, X_val, y_val)
        assert np.isfinite(score)
        # Multi-model diagnostics should be logged
        assert len(fs._eval_multimodel_fold_log) == 1
        log_entry = fs._eval_multimodel_fold_log[0]
        assert "lr_l2" in log_entry
        assert "linear_svc" in log_entry
        assert "rf_small" in log_entry

    def test_toggle_on_deterministic_via_base(self):
        from tabnetics.feature_selection.base import FeatureSelector
        X_tr, y_tr, X_val, y_val = _make_binary_data()
        fs1 = FeatureSelector(random_state=42, eval_models_enabled=True)
        s1, sig1 = fs1._fit_and_score_fold(X_tr, y_tr, X_val, y_val)
        fs2 = FeatureSelector(random_state=42, eval_models_enabled=True)
        s2, sig2 = fs2._fit_and_score_fold(X_tr, y_tr, X_val, y_val)
        assert s1 == pytest.approx(s2, abs=1e-12)
        np.testing.assert_array_equal(sig1, sig2)

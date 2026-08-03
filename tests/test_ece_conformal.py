"""Tests for ECE computation and conformal reporting (VAL12_Suggestions §2.1-2.2)."""

import numpy as np
import pytest


class TestExpectedCalibrationError:
    """Verify ECE computation via ClassifierOracle._expected_calibration_error."""

    @staticmethod
    def _ece(y_true, proba, n_bins=15):
        from tabnetics.classification.backends import ClassifierOracle

        return ClassifierOracle._expected_calibration_error(
            np.asarray(y_true),
            np.asarray(proba, dtype=float),
            n_bins=n_bins,
        )

    def test_perfect_calibration_binary(self):
        """Perfectly calibrated predictions should have ECE near 0."""
        # 100 samples, predicted confidence matches accuracy
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.randint(0, 2, size=n)
        # Perfect calibration: predict_proba matches the true label exactly
        proba = np.zeros((n, 2), dtype=float)
        for i in range(n):
            proba[i, y_true[i]] = 0.95
            proba[i, 1 - y_true[i]] = 0.05
        ece = self._ece(y_true, proba)
        assert np.isfinite(ece)
        assert ece < 0.1, f"Near-perfect calibration should have low ECE, got {ece}"

    def test_terrible_calibration(self):
        """Highly miscalibrated predictions should have high ECE."""
        n = 200
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=n)
        # Always predict class 0 with 99% confidence
        proba = np.zeros((n, 2), dtype=float)
        proba[:, 0] = 0.99
        proba[:, 1] = 0.01
        ece = self._ece(y_true, proba)
        assert np.isfinite(ece)
        # About half are wrong but all are confident → high ECE
        assert ece > 0.3, f"Miscalibrated predictions should have high ECE, got {ece}"

    def test_ece_range(self):
        """ECE should be in [0, 1]."""
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.randint(0, 3, size=n)
        proba = rng.dirichlet([1, 1, 1], size=n)
        ece = self._ece(y_true, proba)
        assert np.isfinite(ece)
        assert 0.0 <= ece <= 1.0, f"ECE should be in [0, 1], got {ece}"

    def test_ece_1d_proba_returns_nan(self):
        """1D probability array should return NaN."""
        y_true = np.array([0, 1, 0, 1])
        proba = np.array([0.2, 0.8, 0.3, 0.7])
        ece = self._ece(y_true, proba)
        assert np.isnan(ece), "1D proba should return NaN"

    def test_ece_empty_returns_nan(self):
        """Empty inputs should return NaN."""
        ece = self._ece(np.array([]), np.empty((0, 2)))
        assert np.isnan(ece), "Empty input should return NaN"

    def test_ece_wrong_num_classes_returns_nan(self):
        """Mismatched number of classes in proba vs y_true should return NaN."""
        y_true = np.array([0, 1, 2])
        proba = np.array([[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]])  # 2 cols but 3 classes
        ece = self._ece(y_true, proba)
        assert np.isnan(ece), "Mismatched classes should return NaN"

    def test_ece_uniform_predictions(self):
        """Uniform predictions (always 1/K confidence) should have some ECE."""
        n = 100
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 3, size=n)
        proba = np.full((n, 3), 1.0 / 3.0)
        ece = self._ece(y_true, proba)
        assert np.isfinite(ece)
        # Uniform => confidence = 1/3, accuracy ~= 1/3, so ECE should be small
        assert ece < 0.15, f"Uniform predictions should have small ECE, got {ece}"

    def test_ece_multiclass(self):
        """ECE should work for multiclass (K > 2)."""
        rng = np.random.RandomState(42)
        n = 150
        k = 5
        y_true = rng.randint(0, k, size=n)
        proba = rng.dirichlet(np.ones(k), size=n)
        ece = self._ece(y_true, proba)
        assert np.isfinite(ece)
        assert 0.0 <= ece <= 1.0

    def test_ece_custom_bins(self):
        """ECE with different bin counts should still be valid."""
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.randint(0, 2, size=n)
        proba = rng.dirichlet([1, 1], size=n)
        ece_5 = self._ece(y_true, proba, n_bins=5)
        ece_50 = self._ece(y_true, proba, n_bins=50)
        assert np.isfinite(ece_5)
        assert np.isfinite(ece_50)
        assert 0.0 <= ece_5 <= 1.0
        assert 0.0 <= ece_50 <= 1.0


class TestECEInOracleCandidateStats:
    """Verify ECE flows through the ClassifierOracle pipeline."""

    def test_ece_field_in_stats(self):
        from tabnetics.classification.backends import OracleCandidateStats

        stats = OracleCandidateStats(
            name="test",
            scores=np.array([0.8, 0.9]),
            mean_score=0.85,
            std_score=0.05,
            min_mean_ratio=0.94,
            worst_class_recall_score=0.80,
            complexity_score=0.9,
            calibration_score=0.7,
            cvar_score=0.85,
            ece_score=0.12,
            bbc_corrected_score=0.85,
            bbc_ci_low=0.80,
            bbc_ci_high=0.90,
        )
        assert stats.ece_score == 0.12
        assert stats.worst_class_recall_score == 0.80

    def test_ece_in_oracle_run(self):
        """ClassifierOracle.run() should report ece_score in candidate_stats."""
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from tabnetics.classification.backends import ClassifierOracle

        rng = np.random.RandomState(42)
        n, p = 60, 5
        X = rng.randn(n, p)
        y = (X[:, 0] > 0).astype(int)

        candidates = {
            "nb": make_pipeline(StandardScaler(), GaussianNB()),
        }
        oracle = ClassifierOracle(
            weighting_mode="uniform",
            include_calibration=True,
        )
        result = oracle.run(
            candidates=candidates,
            candidate_names=["nb"],
            X=X,
            y=y,
            seed=42,
            cv_splits=3,
            top_k=1,
        )
        stats = result.get("candidate_stats", {})
        assert "nb" in stats
        nb_stats = stats["nb"]
        assert "ece_score" in nb_stats, "ece_score should be in candidate_stats output"
        ece = nb_stats["ece_score"]
        # ECE should be finite when calibration is enabled and model has predict_proba
        assert np.isfinite(ece), f"ECE should be finite for GaussianNB, got {ece}"
        assert 0.0 <= ece <= 1.0

    def test_ece_nan_without_calibration(self):
        """ECE should be NaN when calibration is disabled."""
        from sklearn.svm import SVC

        from tabnetics.classification.backends import ClassifierOracle

        rng = np.random.RandomState(42)
        n, p = 40, 4
        X = rng.randn(n, p)
        y = rng.randint(0, 2, size=n)

        candidates = {
            "svm": SVC(kernel="linear"),
        }
        oracle = ClassifierOracle(
            weighting_mode="uniform",
            include_calibration=False,
        )
        result = oracle.run(
            candidates=candidates,
            candidate_names=["svm"],
            X=X,
            y=y,
            seed=42,
            cv_splits=3,
            top_k=1,
        )
        stats = result.get("candidate_stats", {})
        assert "svm" in stats
        ece = stats["svm"].get("ece_score", None)
        # With calibration off, ECE should be NaN
        assert ece is not None
        assert np.isnan(ece), f"ECE should be NaN with calibration off, got {ece}"


class TestConformalEmptySetRate:
    """Verify empty_set_rate is reported in conformal output."""

    def test_empty_set_rate_in_output(self):
        """compute_split_conformal_sets should include empty_set_rate."""
        from sklearn.naive_bayes import GaussianNB

        from tabnetics.feature_selection.conformal import compute_split_conformal_sets

        rng = np.random.RandomState(42)
        n_train, n_test, p = 100, 30, 5
        X_train = rng.randn(n_train, p)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_test = rng.randn(n_test, p)
        y_test = (X_test[:, 0] > 0).astype(int)

        model = GaussianNB()

        result = compute_split_conformal_sets(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            y_eval=y_test,
            alpha=0.10,
            calibration_fraction=0.25,
            min_calibration=10,
            seed=42,
        )

        assert "classifier_conformal_empty_set_rate" in result, (
            "empty_set_rate should be in conformal output"
        )
        rate = result["classifier_conformal_empty_set_rate"]
        if result.get("classifier_conformal_applied", False):
            assert np.isfinite(rate)
            assert 0.0 <= rate <= 1.0
        # If not applied (e.g. too few calibration samples), NaN is acceptable

    def test_empty_set_rate_default_nan(self):
        """Default conformal meta should have empty_set_rate as NaN."""
        from tabnetics.feature_selection.conformal import compute_split_conformal_sets

        # Trigger skip by providing too few samples
        X = np.array([[1.0], [2.0]])
        y = np.array([0, 1])
        result = compute_split_conformal_sets(
            model=None,
            X_train=X,
            y_train=y,
            X_eval=X,
            y_eval=y,
            alpha=0.10,
            calibration_fraction=0.25,
            min_calibration=20,
            seed=42,
        )
        assert "classifier_conformal_empty_set_rate" in result
        assert np.isnan(result["classifier_conformal_empty_set_rate"])


class TestCalibrationReportingConfig:
    """Verify calibration_reporting_enabled config toggle exists."""

    def test_config_has_calibration_reporting_enabled(self):
        from tabnetics.pipeline.pipeline import DFFSConfig

        config = DFFSConfig()
        assert hasattr(config, "calibration_reporting_enabled")
        assert config.calibration_reporting_enabled is False, (
            "calibration_reporting_enabled should default to False (opt-in)"
        )

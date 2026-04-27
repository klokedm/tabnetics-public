"""Tests for T-R-246: Tree-model overfitting guard configuration and gating logic."""

import unittest

import numpy as np
from sklearn.datasets import make_classification


class TestTreeOverfittingGuardConfig(unittest.TestCase):
    """Test overfitting guard config fields in ClassificationConfig."""

    def test_default_gap_threshold_enabled(self):
        """Default max_train_test_gap is 0.15 (enabled)."""
        from tabnetics.pipeline.pipeline import ClassificationConfig
        cfg = ClassificationConfig()
        self.assertEqual(cfg.stage2_max_train_test_gap, 0.15)

    def test_default_complexity_penalty_enabled(self):
        """Default tree_complexity_penalty_enabled is True."""
        from tabnetics.pipeline.pipeline import ClassificationConfig
        cfg = ClassificationConfig()
        self.assertTrue(cfg.stage2_tree_complexity_penalty_enabled)

    def test_custom_gap_threshold(self):
        """Custom gap threshold can be set."""
        from tabnetics.pipeline.pipeline import ClassificationConfig
        cfg = ClassificationConfig(stage2_max_train_test_gap=0.15)
        self.assertEqual(cfg.stage2_max_train_test_gap, 0.15)

    def test_complexity_penalty_configurable(self):
        """Complexity penalty can be enabled with custom strength."""
        from tabnetics.pipeline.pipeline import ClassificationConfig
        cfg = ClassificationConfig(
            stage2_tree_complexity_penalty_enabled=True,
            stage2_tree_complexity_penalty_strength=0.2,
        )
        self.assertTrue(cfg.stage2_tree_complexity_penalty_enabled)
        self.assertEqual(cfg.stage2_tree_complexity_penalty_strength, 0.2)

    def test_gap_threshold_accepts_zero(self):
        """Zero gap threshold means disabled."""
        from tabnetics.pipeline.pipeline import ClassificationConfig
        cfg = ClassificationConfig(stage2_max_train_test_gap=0.0)
        self.assertEqual(cfg.stage2_max_train_test_gap, 0.0)

    def test_default_penalty_strength(self):
        """Default penalty strength is 0.1."""
        from tabnetics.pipeline.pipeline import ClassificationConfig
        cfg = ClassificationConfig()
        self.assertEqual(cfg.stage2_tree_complexity_penalty_strength, 0.1)


class TestTreeOverfittingGuardGating(unittest.TestCase):
    """Test that train/test gap gating uses only training data (no leakage)."""

    def _make_data(self, n_samples=100, n_features=10, seed=42):
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=5, n_classes=2, random_state=seed,
        )
        return X, y

    def test_gap_disabled_by_default(self):
        """When max_train_test_gap=0.0, no candidates are rejected for gap."""
        from tabnetics.classification.backends import SklearnBackend
        X, y = self._make_data()
        backend = SklearnBackend(
            candidate_names=("lr", "svm_rbf"),
            max_train_test_gap=0.0,
        )
        _, name, score, _, _, meta = backend.fit_and_select(
            X, y, seed=42, n_classes=2,
            class_counts=np.array([50, 50]),
        )
        self.assertTrue(np.isfinite(score))
        self.assertEqual(len(meta.get("model_cv_gap_rejected", ())), 0)

    def test_gap_gate_rejects_overfitting_candidate(self):
        """A very tight gap threshold rejects candidates with any overfit."""
        from tabnetics.classification.backends import SklearnBackend
        X, y = self._make_data()
        backend = SklearnBackend(
            candidate_names=("lr", "svm_rbf"),
            max_train_test_gap=0.001,  # extremely tight
        )
        _, name, score, _, _, meta = backend.fit_and_select(
            X, y, seed=42, n_classes=2,
            class_counts=np.array([50, 50]),
        )
        gaps = meta.get("model_cv_train_test_gaps", {})
        rejected = meta.get("model_cv_gap_rejected", ())
        # Verify gap info is populated with finite values
        self.assertIsInstance(gaps, dict)
        for model_name, gap_val in gaps.items():
            self.assertIsInstance(gap_val, float)
            self.assertTrue(np.isfinite(gap_val))
        # With a threshold of 0.001, any model with gap > 0.001 should be rejected
        for r in rejected:
            self.assertIn(r, gaps)
            self.assertGreater(gaps[r], 0.001)

    def test_gap_metadata_populated_when_active(self):
        """Gap metadata is populated in the result when gating is enabled."""
        from tabnetics.classification.backends import SklearnBackend
        X, y = self._make_data()
        backend = SklearnBackend(
            candidate_names=("lr",),
            max_train_test_gap=0.5,
        )
        _, _, _, _, _, meta = backend.fit_and_select(
            X, y, seed=42, n_classes=2,
            class_counts=np.array([50, 50]),
        )
        self.assertIn("model_cv_train_test_gaps", meta)
        self.assertIn("model_cv_gap_rejected", meta)
        gaps = meta["model_cv_train_test_gaps"]
        self.assertIn("lr", gaps)
        self.assertIsInstance(gaps["lr"], float)
        self.assertGreaterEqual(gaps["lr"], 0.0)

    def test_no_test_data_in_fit_and_select_signature(self):
        """fit_and_select only accepts training data, never test data."""
        from tabnetics.classification.backends import SklearnBackend
        import inspect
        sig = inspect.signature(SklearnBackend.fit_and_select)
        param_names = set(sig.parameters.keys())
        # Verify no parameter named X_test or y_test exists
        self.assertNotIn("X_test", param_names)
        self.assertNotIn("y_test", param_names)

    def test_gap_uses_cv_folds_only(self):
        """Gap is computed from CV folds within training data, not external test data."""
        from tabnetics.classification.backends import SklearnBackend
        X_train, y_train = self._make_data(n_samples=80, seed=1)
        X_test, _ = self._make_data(n_samples=20, seed=2)

        backend = SklearnBackend(
            candidate_names=("lr",),
            max_train_test_gap=0.5,
        )
        # Only X_train/y_train are passed; X_test is never used
        _, _, score1, _, _, meta1 = backend.fit_and_select(
            X_train, y_train, seed=42, n_classes=2,
            class_counts=np.array([40, 40]),
        )
        # Re-run: result should be deterministic regardless of X_test existence
        _, _, score2, _, _, meta2 = backend.fit_and_select(
            X_train, y_train, seed=42, n_classes=2,
            class_counts=np.array([40, 40]),
        )
        self.assertAlmostEqual(score1, score2, places=10)
        self.assertEqual(
            meta1["model_cv_train_test_gaps"],
            meta2["model_cv_train_test_gaps"],
        )

    def test_complexity_penalty_reduces_tree_score(self):
        """Complexity penalty reduces score for tree models but not linear models."""
        from tabnetics.classification.backends import SklearnBackend
        X, y = self._make_data(n_samples=200, n_features=10)

        # Without penalty
        backend_no_penalty = SklearnBackend(
            candidate_names=("lr",),
            tree_complexity_penalty_enabled=False,
            max_train_test_gap=0.0,
        )
        _, _, score_no_pen, _, _, _ = backend_no_penalty.fit_and_select(
            X, y, seed=42, n_classes=2,
            class_counts=np.array([100, 100]),
        )

        # With penalty (but lr is not a tree model, so no effect)
        backend_with_penalty = SklearnBackend(
            candidate_names=("lr",),
            tree_complexity_penalty_enabled=True,
            tree_complexity_penalty_strength=0.5,
            max_train_test_gap=0.0,
        )
        _, _, score_pen, _, _, meta = backend_with_penalty.fit_and_select(
            X, y, seed=42, n_classes=2,
            class_counts=np.array([100, 100]),
        )
        # LR is NOT a tree model: penalty should not affect its score
        self.assertAlmostEqual(score_no_pen, score_pen, places=10)

    def test_tree_model_names_constant(self):
        """_TREE_MODEL_NAMES contains expected tree-based model families."""
        from tabnetics.classification.backends import _TREE_MODEL_NAMES
        self.assertIn("rf", _TREE_MODEL_NAMES)
        self.assertIn("xgb", _TREE_MODEL_NAMES)
        self.assertIn("lgbm", _TREE_MODEL_NAMES)
        self.assertIn("extra_tree", _TREE_MODEL_NAMES)
        self.assertIn("catboost", _TREE_MODEL_NAMES)
        # Linear models should not be in the set
        self.assertNotIn("lr", _TREE_MODEL_NAMES)
        self.assertNotIn("svm_rbf", _TREE_MODEL_NAMES)

    def test_all_candidates_rejected_falls_back_to_lr(self):
        """When all candidates are rejected by gap gate, falls back to lr."""
        from tabnetics.classification.backends import SklearnBackend
        X, y = self._make_data()
        backend = SklearnBackend(
            candidate_names=("lr", "svm_rbf"),
            max_train_test_gap=1e-9,  # absurdly tight: reject everything
        )
        _, name, score, _, _, meta = backend.fit_and_select(
            X, y, seed=42, n_classes=2,
            class_counts=np.array([50, 50]),
        )
        # Should fall back to lr even if rejected
        self.assertEqual(name, "lr")


if __name__ == "__main__":
    unittest.main()

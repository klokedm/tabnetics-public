"""Tests for PLS-DA class-count guardrail in df_fs_pipeline.py (T-DS1).

Val-3 finding: PLS-DA regressions on datasets with <5 classes (-0.08 TOX at C=4).
Guardrail: folding_pls_min_classes = 5 (default). Configurable via DFFSConfig.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from tabnetics.pipeline.pipeline import DFFSConfig, DistributionFeatureSelectionPipeline


def _make_pipeline(
    folding_method: str = "pls_da",
    folding_pls_min_classes: int = 5,
    folding_pls_min_n_per_class: int = 3,
    folding_pls_max_imbalance_ratio: float = 6.0,
) -> DistributionFeatureSelectionPipeline:
    """Create a pipeline configured with the given folding method and min-class guard."""
    cfg = DFFSConfig(
        folding_method=folding_method,
        folding_pls_min_classes=folding_pls_min_classes,
        folding_pls_min_n_per_class=folding_pls_min_n_per_class,
        folding_pls_max_imbalance_ratio=folding_pls_max_imbalance_ratio,
    )
    return DistributionFeatureSelectionPipeline(config=cfg)


class TestPlsDaGuard:
    """PLS-DA folding should be skipped when n_classes < folding_pls_min_classes."""

    def test_pls_da_skipped_for_binary(self):
        """Binary dataset (2 classes) → guard fires, data returned unchanged."""
        X, y = make_classification(
            n_samples=80, n_features=50, n_classes=2,
            n_informative=10, random_state=42,
        )
        X_train, X_test = X[:60], X[60:]
        y_train = y[:60]

        pipe = _make_pipeline("pls_da")
        X_tr_out, X_te_out, meta = pipe._apply_folding_stage(
            X_train, X_test, y_train, seed=42,
        )

        assert meta["folding_reason"] == "pls_da_insufficient_classes", (
            f"Expected pls_da_insufficient_classes, got {meta['folding_reason']}"
        )
        np.testing.assert_array_equal(X_tr_out, X_train)
        np.testing.assert_array_equal(X_te_out, X_test)

    def test_pls_da_skipped_for_3_classes(self):
        """C=3 dataset → guard fires with default min_classes=5."""
        X, y = make_classification(
            n_samples=90, n_features=40, n_classes=3,
            n_informative=15, n_clusters_per_class=1,
            random_state=99,
        )
        X_train, X_test = X[:70], X[70:]
        y_train = y[:70]

        pipe = _make_pipeline("pls_da", folding_pls_min_classes=5)
        X_tr_out, X_te_out, meta = pipe._apply_folding_stage(
            X_train, X_test, y_train, seed=42,
        )

        assert meta["folding_reason"] == "pls_da_insufficient_classes", (
            "PLS-DA guard should fire for C=3 with min_classes=5"
        )
        np.testing.assert_array_equal(X_tr_out, X_train)
        np.testing.assert_array_equal(X_te_out, X_test)

    def test_pls_da_skipped_for_4_classes(self):
        """C=4 dataset → guard fires with default min_classes=5 (Val-3 regression: TOX -0.08)."""
        X, y = make_classification(
            n_samples=120, n_features=50, n_classes=4,
            n_informative=20, n_clusters_per_class=1,
            random_state=42,
        )
        X_train, X_test = X[:100], X[100:]
        y_train = y[:100]

        pipe = _make_pipeline("pls_da", folding_pls_min_classes=5)
        X_tr_out, X_te_out, meta = pipe._apply_folding_stage(
            X_train, X_test, y_train, seed=42,
        )

        assert meta["folding_reason"] == "pls_da_insufficient_classes", (
            "PLS-DA guard should fire for C=4 with min_classes=5"
        )
        np.testing.assert_array_equal(X_tr_out, X_train)
        np.testing.assert_array_equal(X_te_out, X_test)

    def test_pls_da_runs_for_5_classes(self):
        """C=5 (boundary) → PLS-DA applied with default min_classes=5."""
        X, y = make_classification(
            n_samples=150, n_features=60, n_classes=5,
            n_informative=25, n_clusters_per_class=1,
            random_state=123,
        )
        X_train, X_test = X[:120], X[120:]
        y_train = y[:120]

        pipe = _make_pipeline("pls_da", folding_pls_min_classes=5)
        X_tr_out, X_te_out, meta = pipe._apply_folding_stage(
            X_train, X_test, y_train, seed=42,
        )

        assert meta["folding_reason"] != "pls_da_insufficient_classes", (
            "PLS-DA guard should NOT fire for C=5 (boundary)"
        )
        components_used = meta.get("folding_pls_components_used", 0)
        assert components_used > 0, "PLS-DA should have produced at least 1 component"
        # C=5 yields max (C-1)=4 components
        assert components_used <= 4, (
            f"Expected ≤4 components for C=5, got {components_used}"
        )

    def test_pls_da_runs_for_8_classes(self):
        """C=8 → PLS-DA applied (Val-3: +0.16 NCI at C=9)."""
        X, y = make_classification(
            n_samples=200, n_features=80, n_classes=8,
            n_informative=30, n_clusters_per_class=1,
            random_state=77,
        )
        X_train, X_test = X[:160], X[160:]
        y_train = y[:160]

        pipe = _make_pipeline("pls_da", folding_pls_min_classes=5)
        X_tr_out, X_te_out, meta = pipe._apply_folding_stage(
            X_train, X_test, y_train, seed=42,
        )

        assert meta["folding_reason"] != "pls_da_insufficient_classes", (
            "PLS-DA guard should NOT fire for C=8"
        )
        assert meta.get("folding_applied", False) or meta.get("folding_pls_components_used", 0) > 0

    def test_pls_da_custom_min_classes_3(self):
        """Override guardrail to min_classes=3 → C=4 should run PLS-DA."""
        X, y = make_classification(
            n_samples=120, n_features=50, n_classes=4,
            n_informative=20, n_clusters_per_class=1,
            random_state=42,
        )
        X_train, X_test = X[:100], X[100:]
        y_train = y[:100]

        pipe = _make_pipeline("pls_da", folding_pls_min_classes=3)
        X_tr_out, X_te_out, meta = pipe._apply_folding_stage(
            X_train, X_test, y_train, seed=42,
        )

        assert meta["folding_reason"] != "pls_da_insufficient_classes", (
            "PLS-DA with min_classes=3 should NOT fire guard for C=4"
        )
        assert meta.get("folding_pls_components_used", 0) > 0

    def test_pls_da_components_capped_by_classes(self):
        """Verify max components = C-1 for PLS-DA."""
        X, y = make_classification(
            n_samples=150, n_features=60, n_classes=5,
            n_informative=25, n_clusters_per_class=1,
            random_state=123,
        )
        X_train, X_test = X[:120], X[120:]
        y_train = y[:120]

        cfg = DFFSConfig(
            folding_method="pls_da",
            folding_pls_components=10,
            folding_pls_min_classes=5,  # boundary: exactly 5 classes
        )
        pipe = DistributionFeatureSelectionPipeline(config=cfg)
        X_tr_out, X_te_out, meta = pipe._apply_folding_stage(
            X_train, X_test, y_train, seed=42,
        )

        components_used = meta.get("folding_pls_components_used", 0)
        assert components_used <= 4, (
            f"Expected ≤4 components for C=5, got {components_used}"
        )
        assert components_used > 0, "PLS-DA should have produced at least 1 component"

    def test_default_min_classes_is_5(self):
        """Verify DFFSConfig default for folding_pls_min_classes is 5."""
        cfg = DFFSConfig()
        assert cfg.folding_pls_min_classes == 5, (
            f"Expected default folding_pls_min_classes=5, got {cfg.folding_pls_min_classes}"
        )
        assert cfg.folding_pls_min_n_per_class == 3
        assert cfg.folding_pls_max_imbalance_ratio == pytest.approx(6.0)

    def test_meta_contains_min_classes(self):
        """Verify folding metadata includes the min_classes value."""
        X, y = make_classification(
            n_samples=80, n_features=50, n_classes=2,
            n_informative=10, random_state=42,
        )
        X_train, X_test = X[:60], X[60:]
        y_train = y[:60]

        pipe = _make_pipeline("pls_da", folding_pls_min_classes=5)
        _, _, meta = pipe._apply_folding_stage(
            X_train, X_test, y_train, seed=42,
        )

        assert "folding_pls_min_classes" in meta, "Meta should contain folding_pls_min_classes"
        assert meta["folding_pls_min_classes"] == 5
        assert "folding_pls_min_n_per_class" in meta
        assert "folding_pls_max_imbalance_ratio" in meta

    def test_pls_da_binary_runs_when_guards_relaxed(self):
        """Binary PLS-DA can run when thresholds are intentionally relaxed."""
        X, y = make_classification(
            n_samples=120, n_features=40, n_classes=2,
            n_informative=12, random_state=42,
        )
        X_train, X_test = X[:90], X[90:]
        y_train = y[:90]

        pipe = _make_pipeline(
            "pls_da",
            folding_pls_min_classes=2,
            folding_pls_min_n_per_class=2,
            folding_pls_max_imbalance_ratio=20.0,
        )
        _, _, meta = pipe._apply_folding_stage(
            X_train, X_test, y_train, seed=42,
        )
        assert str(meta.get("folding_reason")) == "ok"
        assert int(meta.get("folding_pls_components_used", 0)) > 0

    def test_pls_da_blocked_by_min_n_per_class(self):
        """PLS-DA guard blocks when smallest class has too few samples."""
        X, y = make_classification(
            n_samples=120,
            n_features=30,
            n_classes=3,
            n_informative=10,
            n_clusters_per_class=1,
            weights=[0.90, 0.08, 0.02],
            random_state=7,
        )
        X_train, X_test = X[:90], X[90:]
        y_train = y[:90]

        pipe = _make_pipeline(
            "pls_da",
            folding_pls_min_classes=3,
            folding_pls_min_n_per_class=4,
            folding_pls_max_imbalance_ratio=50.0,
        )
        _, _, meta = pipe._apply_folding_stage(
            X_train, X_test, y_train, seed=11,
        )
        assert meta["folding_reason"] == "pls_da_insufficient_per_class"

    def test_pls_da_blocked_by_multiclass_imbalance_ratio(self):
        """Multiclass imbalance gate blocks extreme class-ratio cases."""
        X, y = make_classification(
            n_samples=180,
            n_features=45,
            n_classes=4,
            n_informative=12,
            n_clusters_per_class=1,
            weights=[0.76, 0.14, 0.07, 0.03],
            random_state=17,
        )
        X_train, X_test = X[:140], X[140:]
        y_train = y[:140]

        pipe = _make_pipeline(
            "pls_da",
            folding_pls_min_classes=4,
            folding_pls_min_n_per_class=2,
            folding_pls_max_imbalance_ratio=5.0,
        )
        _, _, meta = pipe._apply_folding_stage(
            X_train, X_test, y_train, seed=3,
        )
        assert meta["folding_reason"] == "pls_da_class_imbalance"

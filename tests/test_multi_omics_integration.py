"""Tests for multi-omics integration (VAL12_Suggestions §4.1)."""

import numpy as np
import pytest

from tabnetics.multiomics.integration import MultiBlockPLSDA, MINTIntegrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_block_data(n=120, p1=10, p2=8, seed=42):
    """Create synthetic 2-block data with class-discriminating signal."""
    rng = np.random.RandomState(seed)
    y = np.tile([0, 1, 2], int(np.ceil(n / 3)))[:n]

    # Block 1: features 0-2 discriminate class 0.
    X1 = rng.randn(n, p1)
    X1[y == 0, :3] += 2.0

    # Block 2: features 0-2 discriminate class 1.
    X2 = rng.randn(n, p2)
    X2[y == 1, :3] += 2.0

    return X1, X2, y


# ---------------------------------------------------------------------------
# MultiBlockPLSDA tests
# ---------------------------------------------------------------------------


class TestMultiBlockPLSDA:
    """Verify multi-block PLS-DA fitting and transformation."""

    def test_fit_returns_self(self):
        X1, X2, y = _make_two_block_data()
        model = MultiBlockPLSDA(n_components=2)
        result = model.fit([(X1, "block1"), (X2, "block2")], y)
        assert result is model

    def test_super_scores_shape(self):
        X1, X2, y = _make_two_block_data(n=60)
        model = MultiBlockPLSDA(n_components=3)
        model.fit([(X1, "block1"), (X2, "block2")], y)
        assert model.super_scores_.shape == (60, 3)

    def test_block_loadings_shapes(self):
        X1, X2, y = _make_two_block_data()
        model = MultiBlockPLSDA(n_components=2)
        model.fit([(X1, "b1"), (X2, "b2")], y)
        assert len(model.block_loadings_) == 2
        assert model.block_loadings_[0].shape == (X1.shape[1], 2)
        assert model.block_loadings_[1].shape == (X2.shape[1], 2)

    def test_transform_shape(self):
        X1, X2, y = _make_two_block_data(n=120)
        model = MultiBlockPLSDA(n_components=2)
        model.fit([(X1[:80], "b1"), (X2[:80], "b2")], y[:80])
        T = model.transform([(X1[80:], "b1"), (X2[80:], "b2")])
        assert T.shape == (40, 2)

    def test_fit_transform_equals_super_scores(self):
        X1, X2, y = _make_two_block_data()
        model = MultiBlockPLSDA(n_components=2)
        T = model.fit_transform([(X1, "b1"), (X2, "b2")], y)
        np.testing.assert_array_equal(T, model.super_scores_)

    def test_feature_importance_per_block(self):
        X1, X2, y = _make_two_block_data()
        model = MultiBlockPLSDA(n_components=2)
        model.fit([(X1, "block_a"), (X2, "block_b")], y)
        imp = model.get_feature_importance()
        assert "block_a" in imp
        assert "block_b" in imp
        assert imp["block_a"].shape == (X1.shape[1],)
        assert imp["block_b"].shape == (X2.shape[1],)

    def test_informative_features_have_higher_importance(self):
        """Discriminating features should have higher importance."""
        X1, X2, y = _make_two_block_data(n=200, seed=7)
        model = MultiBlockPLSDA(n_components=2)
        model.fit([(X1, "b1"), (X2, "b2")], y)
        imp = model.get_feature_importance()
        # Block 1: features 0-2 are informative.
        info_imp = np.mean(imp["b1"][:3])
        noise_imp = np.mean(imp["b1"][3:])
        assert info_imp > noise_imp, (
            f"Informative features ({info_imp:.4f}) should have higher "
            f"importance than noise ({noise_imp:.4f})"
        )

    def test_latent_scores_improve_separability(self):
        """Latent scores should yield better class separation than random."""
        X1, X2, y = _make_two_block_data(n=120, seed=13)
        model = MultiBlockPLSDA(n_components=2)
        T = model.fit_transform([(X1, "b1"), (X2, "b2")], y)
        # Compute between-class variance / within-class variance.
        classes = np.unique(y)
        grand_mean = np.mean(T[:, 0])
        between = sum(
            np.sum(y == c) * (np.mean(T[y == c, 0]) - grand_mean) ** 2
            for c in classes
        )
        within = sum(np.sum((T[y == c, 0] - np.mean(T[y == c, 0])) ** 2) for c in classes)
        ratio = between / max(1e-12, within)
        # Should have some class separation.
        assert ratio > 0.01, f"Fisher ratio too low: {ratio:.4f}"

    def test_single_block_works(self):
        X1, _, y = _make_two_block_data(n=60)
        model = MultiBlockPLSDA(n_components=2)
        model.fit([(X1, "only")], y)
        assert model.super_scores_.shape == (60, 2)

    def test_binary_classification(self):
        rng = np.random.RandomState(42)
        n = 80
        X1 = rng.randn(n, 6)
        X2 = rng.randn(n, 4)
        y = np.repeat([0, 1], 40)
        X1[:40, :2] += 1.5
        X2[:40, :2] -= 1.5
        model = MultiBlockPLSDA(n_components=2)
        model.fit([(X1, "b1"), (X2, "b2")], y)
        assert model.super_scores_.shape == (80, 2)

    def test_deterministic(self):
        X1, X2, y = _make_two_block_data()
        m1 = MultiBlockPLSDA(n_components=2)
        m1.fit([(X1, "b1"), (X2, "b2")], y)
        m2 = MultiBlockPLSDA(n_components=2)
        m2.fit([(X1, "b1"), (X2, "b2")], y)
        np.testing.assert_allclose(m1.super_scores_, m2.super_scores_, atol=1e-10)

    def test_explained_variance_nonnegative(self):
        X1, X2, y = _make_two_block_data()
        model = MultiBlockPLSDA(n_components=2)
        model.fit([(X1, "b1"), (X2, "b2")], y)
        assert np.all(model.explained_variance_ >= 0.0)

    def test_mismatched_samples_raises(self):
        X1, X2, y = _make_two_block_data(n=60)
        with pytest.raises(ValueError, match="samples"):
            MultiBlockPLSDA().fit([(X1[:50], "b1"), (X2, "b2")], y)

    def test_mismatched_y_raises(self):
        X1, X2, y = _make_two_block_data(n=60)
        with pytest.raises(ValueError, match="y length"):
            MultiBlockPLSDA().fit([(X1, "b1"), (X2, "b2")], y[:30])

    def test_unfitted_transform_raises(self):
        model = MultiBlockPLSDA()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.transform([(np.zeros((5, 3)), "b1")])


# ---------------------------------------------------------------------------
# MINTIntegrator tests
# ---------------------------------------------------------------------------


class TestMINTIntegrator:
    """Verify MINT-style study-aware integration."""

    def test_fit_returns_self(self):
        X1, X2, y = _make_two_block_data(n=120)
        studies = np.array(["A"] * 60 + ["B"] * 60)
        model = MINTIntegrator(n_components=2)
        result = model.fit([(X1, "b1"), (X2, "b2")], y, studies)
        assert result is model

    def test_transform_shape(self):
        X1, X2, y = _make_two_block_data(n=120)
        studies = np.array(["A"] * 60 + ["B"] * 60)
        model = MINTIntegrator(n_components=2)
        model.fit([(X1[:80], "b1"), (X2[:80], "b2")], y[:80], studies[:80])
        T = model.transform([(X1[80:], "b1"), (X2[80:], "b2")], studies[80:])
        assert T.shape == (40, 2)

    def test_study_centering_reduces_batch_effect(self):
        """MINT should reduce study-induced mean shifts."""
        rng = np.random.RandomState(42)
        n = 120
        y = np.repeat([0, 1], 60)
        X1 = rng.randn(n, 6)
        X2 = rng.randn(n, 4)
        X1[:60, :2] += 1.0  # class signal
        studies = np.array(["A"] * 60 + ["B"] * 60)
        # Inject study effect.
        X1[studies == "B"] += 3.0
        X2[studies == "B"] += 3.0

        model_no_mint = MultiBlockPLSDA(n_components=2)
        T_no = model_no_mint.fit_transform([(X1, "b1"), (X2, "b2")], y)

        model_mint = MINTIntegrator(n_components=2)
        T_mint = model_mint.fit_transform(
            [(X1, "b1"), (X2, "b2")], y, studies,
        )

        # Study-mean gap in latent space should be smaller with MINT.
        gap_no = abs(np.mean(T_no[studies == "A", 0]) - np.mean(T_no[studies == "B", 0]))
        gap_mint = abs(np.mean(T_mint[studies == "A", 0]) - np.mean(T_mint[studies == "B", 0]))
        assert gap_mint <= gap_no + 0.1, (
            f"MINT should reduce study gap: {gap_mint:.3f} > {gap_no:.3f}"
        )

    def test_feature_importance_accessible(self):
        X1, X2, y = _make_two_block_data(n=120)
        studies = np.array(["A"] * 60 + ["B"] * 60)
        model = MINTIntegrator(n_components=2)
        model.fit([(X1, "b1"), (X2, "b2")], y, studies)
        imp = model.get_feature_importance()
        assert "b1" in imp
        assert "b2" in imp

    def test_unseen_study_handled(self):
        """Transform with unseen study should not crash."""
        X1, X2, y = _make_two_block_data(n=120)
        studies_train = np.array(["A"] * 60 + ["B"] * 60)
        model = MINTIntegrator(n_components=2)
        model.fit([(X1, "b1"), (X2, "b2")], y, studies_train)
        studies_test = np.array(["C"] * 40)  # unseen
        T = model.transform([(X1[80:], "b1"), (X2[80:], "b2")], studies_test)
        assert T.shape == (40, 2)
        assert np.all(np.isfinite(T))

    def test_deterministic(self):
        X1, X2, y = _make_two_block_data(n=60)
        studies = np.array(["A"] * 30 + ["B"] * 30)
        m1 = MINTIntegrator(n_components=2)
        T1 = m1.fit_transform([(X1, "b1"), (X2, "b2")], y, studies)
        m2 = MINTIntegrator(n_components=2)
        T2 = m2.fit_transform([(X1, "b1"), (X2, "b2")], y, studies)
        np.testing.assert_allclose(T1, T2, atol=1e-10)

    def test_unfitted_raises(self):
        model = MINTIntegrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.transform(
                [(np.zeros((5, 3)), "b1")],
                np.array(["A"] * 5),
            )

    def test_study_labels_mismatch_raises(self):
        X1, X2, y = _make_two_block_data(n=60)
        with pytest.raises(ValueError, match="mismatch"):
            MINTIntegrator().fit(
                [(X1, "b1"), (X2, "b2")], y,
                np.array(["A"] * 30),  # wrong length
            )

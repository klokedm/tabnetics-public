"""Tests for fold-safe batch correction modes (VAL12_Suggestions §3.2).

Covers the two new modes: combat_seq (count-preserving) and center_scale
(simple per-batch centering and scaling), in addition to verifying that the
existing combat mode continues to work correctly with fold-safe semantics.
"""

import numpy as np

from tabnetics.feature_selection.prefilter import (
    apply_batch_correction_model,
    fit_batch_correction_model,
)


def _make_count_batches(seed: int = 7):
    """Create synthetic count-like data with batch effects."""
    rng = np.random.default_rng(seed)
    n = 120
    p = 15
    # Generate Poisson-distributed count data.
    base_rate = rng.uniform(5.0, 50.0, size=p)
    X = rng.poisson(lam=base_rate[None, :], size=(n, p)).astype(float)
    y = np.tile(np.array([0, 1, 2], dtype=int), int(np.ceil(n / 3)))[:n]
    batches = np.array(["A"] * (n // 2) + ["B"] * (n - (n // 2)), dtype=object)
    # Inject multiplicative batch effect (realistic for RNA-seq).
    X[batches == "B", :6] = (X[batches == "B", :6] * 2.5 + 3.0)
    X[batches == "B", 6:10] = (X[batches == "B", 6:10] * 0.5 + 1.0)
    perm = rng.permutation(n)
    return X[perm], y[perm], batches[perm]


def _make_shifted_batches(seed: int = 7):
    """Create synthetic continuous data with batch effects."""
    rng = np.random.default_rng(seed)
    n = 120
    p = 20
    X = rng.normal(loc=0.0, scale=1.0, size=(n, p)).astype(float)
    y = np.tile(np.array([0, 1, 2], dtype=int), int(np.ceil(n / 3)))[:n]
    batches = np.array(["A"] * (n // 2) + ["B"] * (n - (n // 2)), dtype=object)
    X[batches == "B", :8] += 2.5
    X[batches == "B", 8:14] *= 1.8
    perm = rng.permutation(n)
    return X[perm], y[perm], batches[perm]


def _batch_mean_gap(X, batches, feature):
    x = np.asarray(X, dtype=float)[:, int(feature)]
    b = np.asarray(batches, dtype=object).ravel()
    a = x[b == "A"]
    c = x[b == "B"]
    if a.size == 0 or c.size == 0:
        return 0.0
    return float(abs(np.mean(a) - np.mean(c)))


# ---------------------------------------------------------------------------
# ComBat-seq tests
# ---------------------------------------------------------------------------


class TestCombatSeqBatchCorrection:
    """Verify combat_seq mode preserves non-negativity and reduces batch effect."""

    def test_combat_seq_reduces_batch_mean_gap(self):
        X, _, batches = _make_count_batches(seed=13)
        n_train = 80
        X_train, X_test = X[:n_train], X[n_train:]
        b_train, b_test = batches[:n_train], batches[n_train:]

        model, fit_meta = fit_batch_correction_model(
            X_train,
            batch_labels=b_train,
            mode="combat_seq",
            combat_prior_strength=8.0,
        )
        X_train_corr, X_test_corr, apply_meta = apply_batch_correction_model(
            X_train,
            X_test,
            model=model,
            batch_labels_train=b_train,
            batch_labels_test=b_test,
        )

        assert fit_meta["batch_correction_applied"] is True
        assert fit_meta["batch_correction_mode_applied"] == "combat_seq"
        assert apply_meta["batch_correction_apply_reason"] == "ok"
        # Batch mean gap should be reduced for affected features.
        pre_gap = _batch_mean_gap(X_train, b_train, feature=0)
        post_gap = _batch_mean_gap(X_train_corr, b_train, feature=0)
        assert post_gap < pre_gap, (
            f"combat_seq should reduce batch mean gap: {post_gap} >= {pre_gap}"
        )

    def test_combat_seq_preserves_non_negativity(self):
        X, _, batches = _make_count_batches(seed=17)
        n_train = 80
        model, _ = fit_batch_correction_model(
            X[:n_train],
            batch_labels=batches[:n_train],
            mode="combat_seq",
        )
        X_train_corr, X_test_corr, _ = apply_batch_correction_model(
            X[:n_train],
            X[n_train:],
            model=model,
            batch_labels_train=batches[:n_train],
            batch_labels_test=batches[n_train:],
        )

        assert np.all(X_train_corr >= 0.0), "combat_seq output should be non-negative"
        assert np.all(X_test_corr >= 0.0), "combat_seq output should be non-negative"

    def test_combat_seq_output_shapes_match(self):
        X, _, batches = _make_count_batches(seed=23)
        n_train = 80
        model, _ = fit_batch_correction_model(
            X[:n_train],
            batch_labels=batches[:n_train],
            mode="combat_seq",
        )
        X_train_corr, X_test_corr, _ = apply_batch_correction_model(
            X[:n_train],
            X[n_train:],
            model=model,
            batch_labels_train=batches[:n_train],
            batch_labels_test=batches[n_train:],
        )

        assert X_train_corr.shape == X[:n_train].shape
        assert X_test_corr.shape == X[n_train:].shape

    def test_combat_seq_alias(self):
        """combat-seq and combatseq should map to combat_seq."""
        X, _, batches = _make_count_batches(seed=29)
        n_train = 80
        for alias in ["combat-seq", "combatseq"]:
            model, meta = fit_batch_correction_model(
                X[:n_train],
                batch_labels=batches[:n_train],
                mode=alias,
            )
            assert meta["batch_correction_mode_applied"] == "combat_seq"


# ---------------------------------------------------------------------------
# Center-scale tests
# ---------------------------------------------------------------------------


class TestCenterScaleBatchCorrection:
    """Verify center_scale mode reduces batch mean/scale differences."""

    def test_center_scale_reduces_batch_mean_gap(self):
        X, _, batches = _make_shifted_batches(seed=13)
        n_train = 80
        X_train, X_test = X[:n_train], X[n_train:]
        b_train, b_test = batches[:n_train], batches[n_train:]

        model, fit_meta = fit_batch_correction_model(
            X_train,
            batch_labels=b_train,
            mode="center_scale",
        )
        X_train_corr, X_test_corr, apply_meta = apply_batch_correction_model(
            X_train,
            X_test,
            model=model,
            batch_labels_train=b_train,
            batch_labels_test=b_test,
        )

        assert fit_meta["batch_correction_applied"] is True
        assert fit_meta["batch_correction_mode_applied"] == "center_scale"
        assert apply_meta["batch_correction_apply_reason"] == "ok"
        # Batch mean gap should be reduced for shifted features.
        pre_gap = _batch_mean_gap(X_train, b_train, feature=0)
        post_gap = _batch_mean_gap(X_train_corr, b_train, feature=0)
        assert post_gap < pre_gap, (
            f"center_scale should reduce batch mean gap: {post_gap} >= {pre_gap}"
        )

    def test_center_scale_output_shapes_match(self):
        X, _, batches = _make_shifted_batches(seed=17)
        n_train = 80
        model, _ = fit_batch_correction_model(
            X[:n_train],
            batch_labels=batches[:n_train],
            mode="center_scale",
        )
        X_train_corr, X_test_corr, _ = apply_batch_correction_model(
            X[:n_train],
            X[n_train:],
            model=model,
            batch_labels_train=batches[:n_train],
            batch_labels_test=batches[n_train:],
        )

        assert X_train_corr.shape == X[:n_train].shape
        assert X_test_corr.shape == X[n_train:].shape

    def test_center_scale_alias(self):
        """center-scale and centerscale should map to center_scale."""
        X, _, batches = _make_shifted_batches(seed=29)
        n_train = 80
        for alias in ["center-scale", "centerscale"]:
            model, meta = fit_batch_correction_model(
                X[:n_train],
                batch_labels=batches[:n_train],
                mode=alias,
            )
            assert meta["batch_correction_mode_applied"] == "center_scale"

    def test_center_scale_test_gap_reduced(self):
        """center_scale should also reduce batch gap on test data."""
        X, _, batches = _make_shifted_batches(seed=37)
        n_train = 80
        X_train, X_test = X[:n_train], X[n_train:]
        b_train, b_test = batches[:n_train], batches[n_train:]

        model, _ = fit_batch_correction_model(
            X_train, batch_labels=b_train, mode="center_scale",
        )
        _, X_test_corr, _ = apply_batch_correction_model(
            X_train,
            X_test,
            model=model,
            batch_labels_train=b_train,
            batch_labels_test=b_test,
        )

        pre_gap = _batch_mean_gap(X_test, b_test, feature=0)
        post_gap = _batch_mean_gap(X_test_corr, b_test, feature=0)
        assert post_gap < pre_gap


# ---------------------------------------------------------------------------
# Fold-safety tests (no leakage)
# ---------------------------------------------------------------------------


class TestFoldSafety:
    """Verify that batch correction is fold-safe (no test info leaks to fit)."""

    def test_fold_safe_combat_seq_deterministic(self):
        """Running combat_seq twice with same data → identical output."""
        X, _, batches = _make_count_batches(seed=42)
        n_train = 80
        model1, _ = fit_batch_correction_model(
            X[:n_train], batch_labels=batches[:n_train], mode="combat_seq",
        )
        model2, _ = fit_batch_correction_model(
            X[:n_train], batch_labels=batches[:n_train], mode="combat_seq",
        )
        Xtr1, Xte1, _ = apply_batch_correction_model(
            X[:n_train], X[n_train:],
            model=model1,
            batch_labels_train=batches[:n_train],
            batch_labels_test=batches[n_train:],
        )
        Xtr2, Xte2, _ = apply_batch_correction_model(
            X[:n_train], X[n_train:],
            model=model2,
            batch_labels_train=batches[:n_train],
            batch_labels_test=batches[n_train:],
        )
        np.testing.assert_allclose(Xtr1, Xtr2, atol=1e-12)
        np.testing.assert_allclose(Xte1, Xte2, atol=1e-12)

    def test_fold_safe_center_scale_deterministic(self):
        """Running center_scale twice with same data → identical output."""
        X, _, batches = _make_shifted_batches(seed=42)
        n_train = 80
        model1, _ = fit_batch_correction_model(
            X[:n_train], batch_labels=batches[:n_train], mode="center_scale",
        )
        model2, _ = fit_batch_correction_model(
            X[:n_train], batch_labels=batches[:n_train], mode="center_scale",
        )
        Xtr1, Xte1, _ = apply_batch_correction_model(
            X[:n_train], X[n_train:],
            model=model1,
            batch_labels_train=batches[:n_train],
            batch_labels_test=batches[n_train:],
        )
        Xtr2, Xte2, _ = apply_batch_correction_model(
            X[:n_train], X[n_train:],
            model=model2,
            batch_labels_train=batches[:n_train],
            batch_labels_test=batches[n_train:],
        )
        np.testing.assert_allclose(Xtr1, Xtr2, atol=1e-12)
        np.testing.assert_allclose(Xte1, Xte2, atol=1e-12)

    def test_fold_safe_no_test_leakage(self):
        """Model fit on train only; changing test data should not change train output."""
        X, _, batches = _make_shifted_batches(seed=51)
        n_train = 80
        X_test_a = X[n_train:]
        X_test_b = X[n_train:].copy()
        X_test_b[:, 0] += 100.0  # drastically modify test data

        model, _ = fit_batch_correction_model(
            X[:n_train], batch_labels=batches[:n_train], mode="center_scale",
        )
        Xtr_a, _, _ = apply_batch_correction_model(
            X[:n_train], X_test_a,
            model=model,
            batch_labels_train=batches[:n_train],
            batch_labels_test=batches[n_train:],
        )
        Xtr_b, _, _ = apply_batch_correction_model(
            X[:n_train], X_test_b,
            model=model,
            batch_labels_train=batches[:n_train],
            batch_labels_test=batches[n_train:],
        )
        np.testing.assert_allclose(
            Xtr_a, Xtr_b, atol=1e-12,
            err_msg="Changing test data should not affect train correction",
        )

    def test_unknown_test_batch_handled_gracefully(self):
        """Unknown batch in test set should be reported, not crash."""
        X, _, batches = _make_count_batches(seed=61)
        n_train = 80
        b_test = batches[n_train:].copy()
        b_test[:5] = "UNSEEN_BATCH"

        model, _ = fit_batch_correction_model(
            X[:n_train], batch_labels=batches[:n_train], mode="combat_seq",
        )
        _, _, meta = apply_batch_correction_model(
            X[:n_train], X[n_train:],
            model=model,
            batch_labels_train=batches[:n_train],
            batch_labels_test=b_test,
        )
        assert int(meta["batch_correction_unknown_test_batches"]) >= 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBatchCorrectionEdgeCases:
    """Edge cases for new batch correction modes."""

    def test_combat_seq_missing_labels_noop(self):
        X, _, _ = _make_count_batches(seed=71)
        model, meta = fit_batch_correction_model(
            X[:70], batch_labels=None, mode="combat_seq",
        )
        assert meta["batch_correction_applied"] is False
        assert meta["batch_correction_fit_reason"] == "missing_batch_labels"

    def test_center_scale_missing_labels_noop(self):
        X, _, _ = _make_shifted_batches(seed=71)
        model, meta = fit_batch_correction_model(
            X[:70], batch_labels=None, mode="center_scale",
        )
        assert meta["batch_correction_applied"] is False
        assert meta["batch_correction_fit_reason"] == "missing_batch_labels"

    def test_combat_seq_single_batch_noop(self):
        X, _, _ = _make_count_batches(seed=73)
        batches_single = np.array(["A"] * 70, dtype=object)
        model, meta = fit_batch_correction_model(
            X[:70], batch_labels=batches_single, mode="combat_seq",
        )
        assert meta["batch_correction_applied"] is False
        assert meta["batch_correction_fit_reason"] == "single_batch"

    def test_center_scale_single_batch_noop(self):
        X, _, _ = _make_shifted_batches(seed=73)
        batches_single = np.array(["A"] * 70, dtype=object)
        model, meta = fit_batch_correction_model(
            X[:70], batch_labels=batches_single, mode="center_scale",
        )
        assert meta["batch_correction_applied"] is False
        assert meta["batch_correction_fit_reason"] == "single_batch"

    def test_combat_seq_zero_counts_handled(self):
        """Data with many zeros (typical RNA-seq) should not crash."""
        rng = np.random.default_rng(81)
        n, p = 100, 10
        X = rng.poisson(lam=2.0, size=(n, p)).astype(float)
        # Many zeros in sparse data
        X[X < 1] = 0.0
        batches = np.array(["A"] * 50 + ["B"] * 50, dtype=object)
        X[batches == "B"] += 5.0

        model, meta = fit_batch_correction_model(
            X, batch_labels=batches, mode="combat_seq",
        )
        assert meta["batch_correction_applied"] is True
        Xtr_corr, Xte_corr, _ = apply_batch_correction_model(
            X[:80], X[80:],
            model=model,
            batch_labels_train=batches[:80],
            batch_labels_test=batches[80:],
        )
        assert np.all(np.isfinite(Xtr_corr))
        assert np.all(np.isfinite(Xte_corr))
        assert np.all(Xtr_corr >= 0.0)

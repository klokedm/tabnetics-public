import numpy as np

from tabnetics.feature_selection.prefilter import (
    apply_batch_correction_model,
    fit_batch_correction_model,
)


def _make_shifted_batches(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = 120
    p = 20
    X = rng.normal(loc=0.0, scale=1.0, size=(n, p)).astype(float)
    y = np.tile(np.array([0, 1, 2], dtype=int), int(np.ceil(n / 3)))[:n]
    batches = np.array(["A"] * (n // 2) + ["B"] * (n - (n // 2)), dtype=object)
    # Inject a center-specific shift/scale pattern.
    X[batches == "B", :8] += 2.5
    X[batches == "B", 8:14] *= 1.8
    perm = rng.permutation(n)
    return X[perm], y[perm], batches[perm]


def _batch_mean_gap(X: np.ndarray, batches: np.ndarray, feature: int) -> float:
    x = np.asarray(X, dtype=float)[:, int(feature)]
    b = np.asarray(batches, dtype=object).ravel()
    a = x[b == "A"]
    c = x[b == "B"]
    if a.size == 0 or c.size == 0:
        return 0.0
    return float(abs(np.mean(a) - np.mean(c)))


def _batch_quantile_gap(X: np.ndarray, batches: np.ndarray, feature: int, q: float) -> float:
    x = np.asarray(X, dtype=float)[:, int(feature)]
    b = np.asarray(batches, dtype=object).ravel()
    a = x[b == "A"]
    c = x[b == "B"]
    if a.size == 0 or c.size == 0:
        return 0.0
    return float(abs(np.quantile(a, q) - np.quantile(c, q)))


def test_combat_batch_correction_reduces_batch_mean_gap():
    X, _, batches = _make_shifted_batches(seed=13)
    n_train = 80
    X_train, X_test = X[:n_train], X[n_train:]
    b_train, b_test = batches[:n_train], batches[n_train:]

    model, fit_meta = fit_batch_correction_model(
        X_train,
        batch_labels=b_train,
        mode="combat",
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
    assert fit_meta["batch_correction_mode_applied"] == "combat"
    assert apply_meta["batch_correction_apply_reason"] == "ok"
    assert _batch_mean_gap(X_train_corr, b_train, feature=0) < _batch_mean_gap(X_train, b_train, feature=0)
    assert _batch_mean_gap(X_test_corr, b_test, feature=0) < _batch_mean_gap(X_test, b_test, feature=0)


def test_cdf_center_batch_correction_reduces_quantile_gap():
    X, _, batches = _make_shifted_batches(seed=29)
    n_train = 84
    X_train, X_test = X[:n_train], X[n_train:]
    b_train, b_test = batches[:n_train], batches[n_train:]

    model, fit_meta = fit_batch_correction_model(
        X_train,
        batch_labels=b_train,
        mode="cdf_center",
        cdf_center_n_quantiles=41,
        cdf_center_clip_quantiles=(0.02, 0.98),
    )
    X_train_corr, X_test_corr, apply_meta = apply_batch_correction_model(
        X_train,
        X_test,
        model=model,
        batch_labels_train=b_train,
        batch_labels_test=b_test,
    )

    assert fit_meta["batch_correction_applied"] is True
    assert fit_meta["batch_correction_mode_applied"] == "cdf_center"
    assert apply_meta["batch_correction_apply_reason"] == "ok"
    pre_gap = _batch_quantile_gap(X_test, b_test, feature=10, q=0.9)
    post_gap = _batch_quantile_gap(X_test_corr, b_test, feature=10, q=0.9)
    assert post_gap < pre_gap
    assert X_train_corr.shape == X_train.shape
    assert X_test_corr.shape == X_test.shape


def test_batch_correction_missing_labels_is_safe_noop():
    X, _, _ = _make_shifted_batches(seed=17)
    model, fit_meta = fit_batch_correction_model(
        X[:70],
        batch_labels=None,
        mode="combat",
    )
    X_train_corr, X_test_corr, apply_meta = apply_batch_correction_model(
        X[:70],
        X[70:],
        model=model,
        batch_labels_train=None,
        batch_labels_test=None,
    )

    assert fit_meta["batch_correction_applied"] is False
    assert fit_meta["batch_correction_fit_reason"] == "missing_batch_labels"
    assert apply_meta["batch_correction_apply_reason"] == "mode_none"
    np.testing.assert_allclose(X_train_corr, X[:70], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(X_test_corr, X[70:], atol=0.0, rtol=0.0)


def test_batch_correction_unknown_test_batch_is_reported():
    X, _, batches = _make_shifted_batches(seed=23)
    n_train = 80
    X_train, X_test = X[:n_train], X[n_train:]
    b_train = batches[:n_train]
    b_test = np.array(["A"] * len(X_test), dtype=object)
    b_test[:5] = "C"  # unseen center in fit model

    model, _ = fit_batch_correction_model(
        X_train,
        batch_labels=b_train,
        mode="combat",
    )
    _, _, apply_meta = apply_batch_correction_model(
        X_train,
        X_test,
        model=model,
        batch_labels_train=b_train,
        batch_labels_test=b_test,
    )
    assert int(apply_meta["batch_correction_unknown_test_batches"]) >= 5

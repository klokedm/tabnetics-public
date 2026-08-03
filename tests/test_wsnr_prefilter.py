import numpy as np

from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.prefilter import (
    _wsnr_maybe_stabilize_counts,
    build_prefilter_union_pool,
    wsnr_scores,
)


def _normalize(v):
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _toy_binary(seed: int = 42):
    rng = np.random.RandomState(seed)
    n, p = 120, 40
    y = rng.randint(0, 2, size=n)
    X = rng.normal(size=(n, p))
    X[:, 0] = y + 0.05 * rng.normal(size=n)
    X[:, 1] = (1 - y) + 0.08 * rng.normal(size=n)
    X[:, 2] = 0.5 * y + 0.15 * rng.normal(size=n)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _toy_count_binary(seed: int = 123):
    rng = np.random.RandomState(seed)
    n, p = 120, 40
    y = rng.randint(0, 2, size=n)
    lib_size = rng.lognormal(mean=0.0, sigma=0.8, size=n)
    base = rng.uniform(0.002, 0.020, size=p)
    signal = np.zeros(p, dtype=float)
    signal[0] = 0.030
    signal[1] = 0.020
    lam = lib_size[:, None] * (base[None, :] + signal[None, :] * y[:, None])
    X = rng.poisson(np.clip(lam, 1e-8, None)).astype(float)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def test_wsnr_scores_shape_and_finite():
    X, y = _toy_binary()
    scores = wsnr_scores(X, y)
    assert scores.shape == (X.shape[1],)
    assert np.all(np.isfinite(scores))


def test_wsnr_scores_prioritize_informative_features():
    X, y = _toy_binary()
    scores = wsnr_scores(X, y)
    top10 = set(np.argsort(scores)[::-1][:10].tolist())
    assert 0 in top10
    assert 1 in top10


def test_wsnr_scores_non_binary_returns_zero_vector():
    rng = np.random.RandomState(7)
    X = rng.normal(size=(90, 20))
    y = rng.randint(0, 3, size=90)
    scores = wsnr_scores(X, y)
    assert scores.shape == (20,)
    assert np.allclose(scores, 0.0)


def test_wsnr_scores_single_class_returns_zero_vector():
    rng = np.random.RandomState(9)
    X = rng.normal(size=(64, 16))
    y = np.zeros(64, dtype=int)
    scores = wsnr_scores(X, y)
    assert scores.shape == (16,)
    assert np.allclose(scores, 0.0)


def test_wsnr_scores_handles_empty_feature_matrix():
    X = np.zeros((20, 0), dtype=float)
    y = np.random.RandomState(11).randint(0, 2, size=20)
    scores = wsnr_scores(X, y)
    assert scores.shape == (0,)


def test_wsnr_count_matrix_stabilization_changes_rankings():
    X, y = _toy_count_binary()
    scores_raw = wsnr_scores(X, y, wsnr_stabilize_counts=False)
    scores_stab = wsnr_scores(X, y, wsnr_stabilize_counts=True)
    assert scores_raw.shape == scores_stab.shape
    assert not np.allclose(scores_raw, scores_stab)


def test_wsnr_count_stabilization_toggle_disables_guard():
    X, _ = _toy_count_binary()
    no_guard = _wsnr_maybe_stabilize_counts(X, wsnr_stabilize_counts=False, data_domain="auto")
    guarded = _wsnr_maybe_stabilize_counts(X, wsnr_stabilize_counts=True, data_domain="auto")
    np.testing.assert_allclose(no_guard, X)
    assert not np.allclose(guarded, X)


def test_wsnr_float_matrix_not_stabilized_by_heuristic():
    X, y = _toy_binary()
    scores_raw = wsnr_scores(X, y, wsnr_stabilize_counts=False)
    scores_guard = wsnr_scores(X, y, wsnr_stabilize_counts=True)
    np.testing.assert_allclose(scores_raw, scores_guard)


def test_prefilter_union_wsnr_strategy_selects_signal_features():
    X, y = _toy_binary()
    base = np.abs(np.corrcoef(np.column_stack([X, y]), rowvar=False)[:-1, -1])
    idx = build_prefilter_union_pool(
        X,
        y,
        max_features=14,
        strategies=("mi_ftest_blend", "wsnr"),
        nondefault_budget_fraction=0.10,
        base_scores=base,
        mi_scores=base,
        f_scores=base,
        normalize_fn=_normalize,
        random_state=17,
        problem_type="classification",
    )
    chosen = set(idx.tolist())
    assert idx.size == 14
    assert 0 in chosen


def test_feature_selector_wsnr_flag_appends_strategy():
    fs = FeatureSelector(
        problem_type="classification",
        random_state=5,
        prefilter_union_enabled=False,
        prefilter_wsnr_enabled=True,
        prefilter_strategies=("mi_ftest_blend",),
    )
    assert "wsnr" in set(fs.prefilter_strategies)


def test_feature_selector_wsnr_prefilter_path_executes_binary():
    X, y = _toy_binary()
    fs = FeatureSelector(
        problem_type="classification",
        random_state=19,
        prefilter_union_enabled=True,
        prefilter_wsnr_enabled=True,
        prefilter_strategies=("mi_ftest_blend",),
    )
    idx = fs._prefilter_feature_pool(X, y, max_features=12)
    assert idx.size == 12

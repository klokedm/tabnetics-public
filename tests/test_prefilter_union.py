import numpy as np

from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.prefilter import build_prefilter_union_pool


def _normalize(v):
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _toy():
    rng = np.random.RandomState(42)
    y = rng.randint(0, 2, size=80)
    X = rng.normal(size=(80, 40))
    X[:, 0] = y + 0.05 * rng.normal(size=80)
    X[:, 1] = y + 0.08 * rng.normal(size=80)
    base = np.abs(np.corrcoef(np.column_stack([X, y]), rowvar=False)[:-1, -1])
    return X, y, np.nan_to_num(base, nan=0.0), np.nan_to_num(base, nan=0.0), np.nan_to_num(base, nan=0.0)


def test_prefilter_union_default_strategy_returns_capped_pool():
    X, y, base, mi, f = _toy()
    idx = build_prefilter_union_pool(
        X,
        y,
        max_features=12,
        strategies=("mi_ftest_blend",),
        nondefault_budget_fraction=0.1,
        base_scores=base,
        mi_scores=mi,
        f_scores=f,
        normalize_fn=_normalize,
        random_state=11,
        problem_type="classification",
    )
    assert idx.size == 12


def test_prefilter_union_with_rf_importance_adds_candidates():
    X, y, base, mi, f = _toy()
    idx = build_prefilter_union_pool(
        X,
        y,
        max_features=14,
        strategies=("mi_ftest_blend", "rf_importance"),
        nondefault_budget_fraction=0.10,
        base_scores=base,
        mi_scores=mi,
        f_scores=f,
        normalize_fn=_normalize,
        random_state=11,
        problem_type="classification",
    )
    assert idx.size == 14
    assert 0 in set(idx.tolist())


def test_prefilter_union_with_relieff_scores_adds_candidates():
    X, y, base, mi, f = _toy()
    idx = build_prefilter_union_pool(
        X,
        y,
        max_features=14,
        strategies=("mi_ftest_blend", "relieff_scores"),
        nondefault_budget_fraction=0.10,
        base_scores=base,
        mi_scores=mi,
        f_scores=f,
        normalize_fn=_normalize,
        random_state=13,
        problem_type="classification",
    )
    assert idx.size == 14
    assert 1 in set(idx.tolist())


def test_prefilter_union_ignores_unknown_strategies():
    X, y, base, mi, f = _toy()
    idx = build_prefilter_union_pool(
        X,
        y,
        max_features=10,
        strategies=("mi_ftest_blend", "unknown_strategy"),
        nondefault_budget_fraction=0.10,
        base_scores=base,
        mi_scores=mi,
        f_scores=f,
        normalize_fn=_normalize,
        random_state=19,
        problem_type="classification",
    )
    assert idx.size == 10


def test_prefilter_union_caps_output_to_max_features():
    X, y, base, mi, f = _toy()
    idx = build_prefilter_union_pool(
        X,
        y,
        max_features=8,
        strategies=("mi_ftest_blend", "rf_importance", "relieff_scores"),
        nondefault_budget_fraction=0.25,
        base_scores=base,
        mi_scores=mi,
        f_scores=f,
        normalize_fn=_normalize,
        random_state=23,
        problem_type="classification",
    )
    assert idx.size == 8


def test_feature_selector_prefilter_union_path_executes():
    X, y, _, _, _ = _toy()
    fs = FeatureSelector(
        problem_type="classification",
        random_state=5,
        prefilter_union_enabled=True,
        prefilter_strategies=("mi_ftest_blend", "rf_importance"),
        prefilter_nondefault_budget_fraction=0.10,
    )
    idx = fs._prefilter_feature_pool(X, y, max_features=12)
    assert idx.size == 12

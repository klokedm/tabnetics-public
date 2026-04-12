import numpy as np
from sklearn.datasets import make_classification

from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.mnpo.portfolio import (
    extract_portfolio,
    resolve_adaptive_portfolio_size,
)


def _make_data(seed: int = 7):
    X, y = make_classification(
        n_samples=72,
        n_features=80,
        n_informative=12,
        n_redundant=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def test_resolve_adaptive_portfolio_size_disabled_returns_requested_bound():
    w = np.array([0.6, 0.3, 0.1], dtype=float)
    k, meta = resolve_adaptive_portfolio_size(
        w,
        portfolio_size=4,
        adaptive_enabled=False,
        adaptive_size_min=2,
        adaptive_size_max=5,
    )
    assert int(k) == 3
    assert str(meta["reason"]) == "disabled"


def test_resolve_adaptive_portfolio_size_head_heavy_prefers_small_k():
    w = np.array([0.78, 0.12, 0.05, 0.03, 0.02], dtype=float)
    k, meta = resolve_adaptive_portfolio_size(
        w,
        portfolio_size=5,
        adaptive_enabled=True,
        adaptive_size_min=2,
        adaptive_size_max=5,
    )
    assert 2 <= int(k) <= 4
    assert str(meta["reason"]) == "ok"
    assert int(meta["k_mass"]) >= 1
    assert int(meta["k_elbow"]) >= 1


def test_resolve_adaptive_portfolio_size_variance_penalty_shrinks_k():
    w = np.array([0.85, 0.10, 0.03, 0.01, 0.01], dtype=float)
    k_base, _ = resolve_adaptive_portfolio_size(
        w,
        portfolio_size=5,
        adaptive_enabled=True,
        adaptive_size_min=2,
        adaptive_size_max=5,
        adaptive_sizing_variance_penalty=False,
    )
    k_pen, meta_pen = resolve_adaptive_portfolio_size(
        w,
        portfolio_size=5,
        adaptive_enabled=True,
        adaptive_size_min=2,
        adaptive_size_max=5,
        adaptive_sizing_variance_penalty=True,
        adaptive_sizing_variance_penalty_strength=1.0,
    )
    assert int(k_pen) <= int(k_base)
    assert bool(meta_pen["variance_penalty_enabled"]) is True


def test_resolve_adaptive_portfolio_size_missing_bounds_falls_back():
    w = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)
    k, meta = resolve_adaptive_portfolio_size(
        w,
        portfolio_size=3,
        adaptive_enabled=True,
        adaptive_size_min=None,
        adaptive_size_max=5,
    )
    assert int(k) == 3
    assert str(meta["reason"]) == "missing_bounds"


def test_feature_selector_emits_adaptive_portfolio_metadata():
    X, y = _make_data()
    sel = FeatureSelector(
        random_state=11,
        n_bootstrap_iterations=1,
        inner_cv_splits=3,
        inner_cv_repeats=1,
        mirror_descent_steps=40,
        enabled_methods={"mutual_information", "anova_f", "linear_svm"},
        selection_strategy="mnpo_portfolio",
        portfolio_size=5,
        adaptive_portfolio_sizing_enabled=True,
        adaptive_size_min=3,
        adaptive_size_max=5,
    )

    _, result = sel.fit_transform(X, y, n_final_features=12, return_result_object=True)
    mnpo = dict(result.method_results.get("mnpo_portfolio", {}) or {})
    meta = dict(mnpo.get("adaptive_portfolio_sizing", {}) or {})

    assert bool(meta.get("adaptive_enabled", False)) is True
    assert int(mnpo.get("portfolio_size_requested", 0)) == 5
    assert 3 <= int(mnpo.get("portfolio_size_effective", 0)) <= 5


def _entry(indices, signal):
    return {
        "selected_indices": np.asarray(indices, dtype=int),
        "prediction_signal": np.asarray(signal, dtype=float),
    }


def test_extract_portfolio_blocks_near_clone_by_overlap_alone():
    names = ["m0", "m1", "m2"]
    weights = np.asarray([0.90, 0.80, 0.70], dtype=float)
    evaluation = {
        "m0": _entry(range(10), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "m1": _entry([0, 1, 2, 3, 4, 5, 6, 7, 8, 20], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        "m2": _entry(range(30, 40), [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
    }
    selected = extract_portfolio(
        names, weights, evaluation, portfolio_size=2, use_diversity_oracle=True
    )
    assert selected == [0, 2]


def test_extract_portfolio_blocks_near_clone_by_correlation_alone():
    names = ["m0", "m1", "m2"]
    weights = np.asarray([0.90, 0.80, 0.70], dtype=float)
    base_signal = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4], dtype=float)
    evaluation = {
        "m0": _entry(range(10), base_signal),
        "m1": _entry(range(20, 30), 2.0 * base_signal + 1.0),
        "m2": _entry(range(40, 50), [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]),
    }
    selected = extract_portfolio(
        names, weights, evaluation, portfolio_size=2, use_diversity_oracle=True
    )
    assert selected == [0, 2]


def test_extract_portfolio_keeps_diverse_pair():
    names = ["m0", "m1"]
    weights = np.asarray([0.90, 0.80], dtype=float)
    evaluation = {
        "m0": _entry(range(10), [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]),
        "m1": _entry(range(15, 25), [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]),
    }
    selected = extract_portfolio(
        names, weights, evaluation, portfolio_size=2, use_diversity_oracle=True
    )
    assert selected == [0, 1]


def test_extract_portfolio_keeps_old_and_block_cases_blocked():
    names = ["m0", "m1", "m2"]
    weights = np.asarray([0.90, 0.80, 0.70], dtype=float)
    evaluation = {
        "m0": _entry(range(10), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "m1": _entry(range(10), [0.01, 1.01, 2.01, 3.01, 4.01, 5.01, 6.01, 7.01, 8.01, 9.01]),
        "m2": _entry(range(30, 40), [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
    }
    selected = extract_portfolio(
        names, weights, evaluation, portfolio_size=2, use_diversity_oracle=True
    )
    assert selected == [0, 2]

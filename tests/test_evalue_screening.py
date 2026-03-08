import numpy as np

from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.methods.screening import screen_features
from tabnetics.feature_selection.prefilter import ebh_support, pvalues_to_evalues


def test_pvalues_to_evalues_basic_monotone():
    pvals = np.array([0.5, 0.25, 0.1], dtype=float)
    evals = pvalues_to_evalues(pvals)
    assert np.all(np.isfinite(evals))
    assert float(evals[2]) > float(evals[1]) > float(evals[0])


def test_pvalues_to_evalues_handles_nan_and_bounds():
    pvals = np.array([np.nan, -1.0, 2.0, 1e-20], dtype=float)
    evals = pvalues_to_evalues(pvals)
    assert evals.shape == (4,)
    assert np.all(np.isfinite(evals))
    assert float(np.max(evals)) >= 1.0


def test_ebh_support_empty_input():
    out = ebh_support(np.array([], dtype=float), alpha=0.2)
    assert out.size == 0


def test_ebh_support_selects_large_evalues():
    e_vals = np.array([30.0, 14.0, 1.0, 0.5, 0.2], dtype=float)
    out = ebh_support(e_vals, alpha=0.2)
    assert np.array_equal(out, np.array([0, 1], dtype=int))


def test_screen_features_evalue_disabled_returns_none():
    X = np.random.RandomState(0).normal(size=(40, 10))
    y = np.random.RandomState(1).randint(0, 2, size=40)
    out = screen_features(X, y, enabled=False, method="evalue")
    assert out is None


def test_screen_features_evalue_selects_signal_feature():
    rng = np.random.RandomState(3)
    y = rng.randint(0, 2, size=80)
    X = rng.normal(size=(80, 16))
    X[:, 0] = y.astype(float) + 0.05 * rng.normal(size=80)
    out = screen_features(
        X,
        y,
        enabled=True,
        method="evalue",
        evalue_alpha=0.2,
        evalue_min_features=1,
    )
    assert out is not None
    out = np.asarray(out, dtype=int)
    assert 0 in set(out.tolist())


def test_screen_features_evalue_respects_min_features_fallback():
    rng = np.random.RandomState(7)
    X = rng.normal(size=(60, 20))
    y = rng.randint(0, 2, size=60)
    out = screen_features(
        X,
        y,
        enabled=True,
        method="evalue",
        evalue_alpha=1e-6,
        evalue_min_features=5,
    )
    assert out is not None
    assert int(np.asarray(out, dtype=int).size) == 5


def test_feature_selector_with_evalue_screening_runs():
    rng = np.random.RandomState(11)
    y = rng.randint(0, 2, size=90)
    X = rng.normal(size=(90, 30))
    X[:, 0] = y.astype(float) + 0.1 * rng.normal(size=90)
    fs = FeatureSelector(
        problem_type="classification",
        random_state=11,
        enabled_methods={"mutual_information", "anova_f", "mrmr_jmi"},
        screening_enabled=True,
        screening_method="evalue",
        screening_evalue_alpha=0.2,
        screening_evalue_min_features=10,
    )
    X_sel, result = fs.fit_transform(X, y, n_final_features=8, return_result_object=True)
    assert X_sel.shape[1] == 8
    assert result.config["screening_method"] == "evalue"

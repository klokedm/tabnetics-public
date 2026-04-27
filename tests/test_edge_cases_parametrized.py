from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import tabnetics.distribution.selector as uds_mod
from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.methods.embedded import linear_svm_selection
from tabnetics.feature_selection.methods.filter import (
    anova_f_selection,
    chi_square_selection,
    mrmr_jmi_selection,
    mutual_information_selection,
    relieff_selection,
    wmw_auc_selection,
)
from tabnetics.feature_selection.methods.knockoff import copula_knockoff_selection
from tabnetics.feature_selection.methods.pairwise import ktsp_selection
from tabnetics.feature_selection.preprocessing import remove_correlated_features


def _toy_xy(n: int, p: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(int(n), int(p)))
    if int(n) <= 0:
        y = np.asarray([], dtype=int)
    else:
        y = np.asarray([i % 2 for i in range(int(n))], dtype=int)
    return X, y


def _var_scorer(X: np.ndarray, y: np.ndarray, random_state: int | None = None) -> np.ndarray:
    del y, random_state
    if X.size == 0:
        return np.asarray([], dtype=float)
    return np.nan_to_num(np.var(X, axis=0), nan=0.0, posinf=0.0, neginf=0.0)


def _f_scorer(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    del y
    scores = _var_scorer(X, np.asarray([]))
    return scores, np.zeros(scores.shape[0], dtype=float)


def _normalize(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=float).ravel()
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _prefilter_small(X: np.ndarray, y: np.ndarray, max_features: int) -> np.ndarray:
    del y
    n_features = int(np.asarray(X).shape[1]) if np.asarray(X).ndim == 2 else 0
    keep = int(min(n_features, max(0, min(int(max_features), 16))))
    return np.arange(keep, dtype=int)


def _call_mutual_information(X: np.ndarray, y: np.ndarray):
    return mutual_information_selection(X, y, 5, _var_scorer, 0)


def _call_anova(X: np.ndarray, y: np.ndarray):
    return anova_f_selection(X, y, 5, _f_scorer)


def _call_wmw_auc(X: np.ndarray, y: np.ndarray):
    return wmw_auc_selection(X, y, 5, "classification")


def _call_chi_square(X: np.ndarray, y: np.ndarray):
    return chi_square_selection(X, y, 5)


def _call_relieff(X: np.ndarray, y: np.ndarray):
    return relieff_selection(X, y, 5, n_neighbors=3)


def _call_mrmr(X: np.ndarray, y: np.ndarray):
    return mrmr_jmi_selection(
        X,
        y,
        5,
        random_state=0,
        mi_scorer=_var_scorer,
        prefilter_fn=_prefilter_small,
        normalize_fn=_normalize,
        mrmr_max_features=16,
        mrmr_redundancy_weight=0.5,
    )


def _call_ktsp(X: np.ndarray, y: np.ndarray):
    return ktsp_selection(
        X,
        y,
        5,
        problem_type="classification",
        random_state=0,
        prefilter_fn=_prefilter_small,
        ktsp_max_features=16,
        ktsp_max_pairs=64,
        ktsp_k_pairs=8,
    )


def _call_linear_svm(X: np.ndarray, y: np.ndarray):
    return linear_svm_selection(
        X,
        y,
        5,
        n_bootstrap_iterations=1,
        problem_type="classification",
        random_state=0,
        linear_svm_max_iter=300,
    )


METHOD_CALLS = {
    "mutual_information": _call_mutual_information,
    "anova_f": _call_anova,
    "wmw_auc": _call_wmw_auc,
    "chi_square": _call_chi_square,
    "relieff": _call_relieff,
    "mrmr_jmi": _call_mrmr,
    "ktsp": _call_ktsp,
    "linear_svm": _call_linear_svm,
}


@pytest.mark.parametrize("method_name", sorted(METHOD_CALLS.keys()))
@pytest.mark.parametrize("n", [0, 1, 2, 5])
@pytest.mark.parametrize("p", [0, 1, 500])
def test_parametrized_edge_cases_method_entry_points(method_name: str, n: int, p: int) -> None:
    X, y = _toy_xy(n, p, seed=(n * 1000 + p + len(method_name)))
    result, all_scores = METHOD_CALLS[method_name](X, y)
    assert isinstance(result, dict)
    assert isinstance(all_scores, dict)

    if n < 2 or p == 0:
        assert result == {}
        assert all_scores == {}


@pytest.mark.parametrize(
    "call_fn",
    [_call_mutual_information, _call_anova, _call_chi_square, _call_relieff, _call_linear_svm],
)
def test_nan_injection_no_unhandled_exceptions(call_fn) -> None:
    X, y = _toy_xy(5, 20, seed=123)
    X[:, 0] = np.nan
    X[0, 1] = np.nan
    result, all_scores = call_fn(X, y)
    assert isinstance(result, dict)
    assert isinstance(all_scores, dict)


def test_copula_knockoff_short_circuits_for_small_n() -> None:
    class DummyKnockoff:
        pass

    X, y = _toy_xy(1, 10, seed=9)
    result, all_scores = copula_knockoff_selection(
        X,
        y,
        5,
        CopulaKnockoffSelectorClass=DummyKnockoff,
        copula_knockoff_draws=10,
        copula_alpha_kn=0.1,
        copula_alpha_ebh=0.2,
        copula_truncation_level=None,
        copula_generator="copula",
        copula_deepdrk_latent_fraction=0.35,
        copula_deepdrk_noise_scale=1.0,
        copula_derandomize_runs=1,
        copula_stabilizer_runs=1,
        copula_stabilizer_use_ebh=False,
        copula_stabilizer_seed_stride=997,
        random_state=0,
    )
    assert result == {}
    assert all_scores == {}


def test_preprocessing_remove_correlated_handles_empty_and_tiny_inputs() -> None:
    empty_df = pd.DataFrame(np.empty((3, 0)))
    out_empty, dropped_empty = remove_correlated_features(empty_df, threshold=0.9)
    assert out_empty.shape == empty_df.shape
    assert dropped_empty == []

    tiny_df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    out_tiny, dropped_tiny = remove_correlated_features(tiny_df, threshold=0.9)
    assert list(out_tiny.columns) == ["a", "b", "c"]
    assert dropped_tiny == []


def test_transform_resets_eval_multimodel_fold_log_across_calls() -> None:
    selector = FeatureSelector()
    selector.selected_features_indices_ = np.asarray([0], dtype=int)
    X = np.ones((3, 1), dtype=float)

    selector._eval_multimodel_fold_log = [{"fold": 1}]
    _ = selector.transform(X)
    assert selector._eval_multimodel_fold_log == []

    selector._eval_multimodel_fold_log = [{"fold": 2}, {"fold": 3}]
    _ = selector.transform(X)
    assert selector._eval_multimodel_fold_log == []


def test_run_selection_methods_timeout_path_marks_method_timeout() -> None:
    selector = FeatureSelector(
        enabled_methods={"mutual_information"},
        method_timeout_seconds=0.05,
    )

    def _slow_method(*args, **kwargs):
        del args, kwargs
        time.sleep(0.2)
        return {"selected_indices": np.asarray([0], dtype=int), "scores": {0: 1.0}}, {0: 1.0}

    selector._mutual_information_selection = _slow_method  # type: ignore[method-assign]
    X, y = _toy_xy(8, 4, seed=77)
    results, _ = selector._run_selection_methods(X, y, n_target=2)
    payload = results["mutual_information"][0]
    assert bool(payload.get("timed_out", False)) is True


def test_run_selection_methods_exception_path_isolated_per_method() -> None:
    selector = FeatureSelector(
        enabled_methods={"mutual_information"},
        method_timeout_seconds=0.0,
    )

    def _raising_method(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("intentional-test-error")

    selector._mutual_information_selection = _raising_method  # type: ignore[method-assign]
    X, y = _toy_xy(8, 4, seed=78)
    results, _ = selector._run_selection_methods(X, y, n_target=2)
    assert results["mutual_information"] == ({}, {})


def test_fit_transform_warns_on_hdlss_regime() -> None:
    selector = FeatureSelector(
        enabled_methods={"mutual_information"},
        n_bootstrap_iterations=1,
        inner_cv_splits=2,
        inner_cv_repeats=1,
        method_timeout_seconds=0.0,
    )
    X, y = _toy_xy(6, 300, seed=90)
    with pytest.warns(RuntimeWarning, match="HDLSS regime detected"):
        X_selected, result = selector.fit_transform(X, y, n_final_features=5, return_result_object=True)
    assert X_selected.shape[0] == 6
    assert result is not None


def test_parallel_distribution_fit_timeout_marks_failed_results(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFuture:
        def __init__(self) -> None:
            self._cancelled = False

        def cancel(self) -> bool:
            self._cancelled = True
            return True

        def result(self, timeout=None):
            del timeout
            return None

    class DummyExecutor:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def submit(self, fn, args):
            del fn, args
            return DummyFuture()

    def _as_completed_timeout(_futures, timeout=None):
        del _futures, timeout
        raise uds_mod.FuturesTimeoutError()

    monkeypatch.setattr(uds_mod, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(uds_mod, "as_completed", _as_completed_timeout)

    selector = uds_mod.UnifiedDistributionSelectorV6(n_jobs=2)
    data = np.linspace(0.1, 1.0, num=16)
    dist_name = next(iter(selector.distributions.keys()))
    results = selector._fit_distributions_parallel(
        data,
        features=None,
        dist_names=[dist_name],
        verbose=False,
    )
    assert len(results) == 1
    assert bool(results[0].success) is False
    assert "timeout" in str(results[0].error).lower()

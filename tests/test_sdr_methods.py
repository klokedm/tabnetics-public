import numpy as np
from sklearn.datasets import make_classification

from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.config import FeatureSelectorConfig, MethodConfig
from tabnetics.feature_selection.methods.sdr import (
    pfc_sdr_selection,
    save_sdr_selection,
    sir_sdr_selection,
)


def _make_multiclass(seed: int = 41):
    X, y = make_classification(
        n_samples=96,
        n_features=52,
        n_informative=14,
        n_redundant=6,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def _prefilter_fn(X, y, max_features):
    del y
    n_features = int(np.asarray(X).shape[1])
    keep = int(max(2, min(n_features, int(max_features))))
    return np.arange(keep, dtype=int)


def _normalize(v):
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _assert_method_output(method_result, all_scores, n_features: int, n_target: int):
    assert isinstance(method_result, dict)
    assert isinstance(all_scores, dict)
    assert "selected_indices" in method_result
    idx = np.asarray(method_result["selected_indices"], dtype=int).ravel()
    assert idx.size <= n_target
    if idx.size:
        assert bool(np.all(idx >= 0))
        assert bool(np.all(idx < n_features))
    assert "all_scores" in method_result
    vec = np.asarray(method_result["all_scores"], dtype=float).ravel()
    assert vec.size == n_features


def test_sdr_methods_return_valid_payloads():
    X, y = _make_multiclass(seed=101)
    n_target = 10
    common_kwargs = dict(
        problem_type="classification",
        sdr_min_classes=3,
        sdr_prefilter_max_features=40,
        sdr_n_components=3,
        sdr_covariance_ridge=1e-3,
        prefilter_fn=_prefilter_fn,
        normalize_fn=_normalize,
        random_state=17,
    )

    sir_result, sir_scores = sir_sdr_selection(X, y, n_target, **common_kwargs)
    save_result, save_scores = save_sdr_selection(X, y, n_target, **common_kwargs)
    pfc_result, pfc_scores = pfc_sdr_selection(X, y, n_target, **common_kwargs)

    _assert_method_output(sir_result, sir_scores, X.shape[1], n_target)
    _assert_method_output(save_result, save_scores, X.shape[1], n_target)
    _assert_method_output(pfc_result, pfc_scores, X.shape[1], n_target)
    assert sir_result.get("sdr_method") == "sir"
    assert save_result.get("sdr_method") == "save"
    assert pfc_result.get("sdr_method") == "pfc"


def test_sdr_methods_respect_min_classes_gate():
    X, y = _make_multiclass(seed=202)
    result, scores = sir_sdr_selection(
        X,
        y,
        8,
        problem_type="classification",
        sdr_min_classes=4,
        sdr_prefilter_max_features=32,
        sdr_n_components=2,
        sdr_covariance_ridge=1e-3,
        prefilter_fn=_prefilter_fn,
        normalize_fn=_normalize,
    )
    assert result == {}
    assert scores == {}


def test_feature_selector_from_config_populates_sdr_fields():
    cfg = FeatureSelectorConfig(
        enabled_methods={"sir_sdr"},
        methods=MethodConfig(
            sdr_min_classes=4,
            sdr_prefilter_max_features=300,
            sdr_n_components=2,
            sdr_covariance_ridge=5e-3,
        ),
    )
    fs = FeatureSelector.from_config(cfg)
    assert int(fs.sdr_min_classes) == 4
    assert int(fs.sdr_prefilter_max_features) == 300
    assert int(fs.sdr_n_components) == 2
    assert abs(float(fs.sdr_covariance_ridge) - 5e-3) < 1e-12

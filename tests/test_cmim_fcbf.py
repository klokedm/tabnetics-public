import numpy as np
from sklearn.datasets import make_classification

from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.methods.filter import (
    cmim_selection,
    fcbf_selection,
    mrmr_jmi_selection,
)
from tabnetics.feature_selection.registry import METHOD_REGISTRY


def _identity_prefilter(X, y, max_features):
    p = int(np.asarray(X).shape[1])
    k = int(max(1, min(max_features, p)))
    return np.arange(k, dtype=int)


def _normalize(v):
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _make_binary(n=120, p=40, seed=7):
    X, y = make_classification(
        n_samples=n,
        n_features=p,
        n_informative=8,
        n_redundant=6,
        n_repeated=0,
        n_classes=2,
        random_state=seed,
        shuffle=False,
    )
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _jaccard(a, b):
    sa = set(int(x) for x in np.asarray(a, dtype=int).tolist())
    sb = set(int(x) for x in np.asarray(b, dtype=int).tolist())
    if not sa and not sb:
        return 1.0
    return float(len(sa.intersection(sb)) / max(1, len(sa.union(sb))))


def test_registry_has_cmim_and_fcbf_methodspec():
    assert "cmim" in METHOD_REGISTRY
    assert "fcbf" in METHOD_REGISTRY


def test_cmim_and_fcbf_marked_experimental():
    assert METHOD_REGISTRY["cmim"].maturity == "experimental"
    assert METHOD_REGISTRY["fcbf"].maturity == "experimental"


def test_fcbf_invalid_shape_returns_empty():
    res, scores = fcbf_selection(
        np.array([], dtype=float),
        np.array([], dtype=int),
        n_target_features=5,
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
    )
    assert res == {}
    assert scores == {}


def test_cmim_invalid_shape_returns_empty():
    res, scores = cmim_selection(
        np.array([], dtype=float),
        np.array([], dtype=int),
        n_target_features=5,
        random_state=0,
        mi_scorer=lambda X, y, random_state: np.zeros(0, dtype=float),
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
    )
    assert res == {}
    assert scores == {}


def test_cmim_respects_min_sample_gate():
    X, y = _make_binary(n=40, p=30, seed=11)
    res, _ = cmim_selection(
        X,
        y,
        n_target_features=6,
        random_state=0,
        mi_scorer=lambda X, y, random_state: np.abs(np.corrcoef(X, rowvar=False)[0])[: X.shape[1]],
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
        min_samples=60,
    )
    assert res == {}


def test_fcbf_returns_top_k_on_informative_binary():
    X, y = _make_binary(n=140, p=50, seed=13)
    res, all_scores = fcbf_selection(
        X,
        y,
        n_target_features=10,
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
        max_features=40,
        n_bins=8,
    )
    selected = np.asarray(res["selected_indices"], dtype=int)
    assert selected.size == 10
    assert len(all_scores) == X.shape[1]
    assert int(np.sum(selected < 20)) >= 3


def test_cmim_returns_top_k_on_informative_binary():
    X, y = _make_binary(n=160, p=50, seed=17)
    fs = FeatureSelector(problem_type="classification", random_state=0)
    res, all_scores = cmim_selection(
        X,
        y,
        n_target_features=12,
        random_state=0,
        mi_scorer=fs.mi_scorer,
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
        max_features=40,
        min_samples=60,
        n_bins=8,
    )
    selected = np.asarray(res["selected_indices"], dtype=int)
    assert selected.size == 12
    assert len(all_scores) == X.shape[1]
    assert int(np.sum(selected < 20)) >= 3


def test_mrmr_default_criterion_and_mode():
    X, y = _make_binary(n=120, p=40, seed=19)
    fs = FeatureSelector(problem_type="classification", random_state=0)
    res, _ = mrmr_jmi_selection(
        X,
        y,
        n_target_features=8,
        random_state=0,
        mi_scorer=fs.mi_scorer,
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
        mrmr_max_features=30,
        mrmr_redundancy_weight=0.55,
        mrmr_mi_redundancy_enabled=False,
        mrmr_mi_n_bins=8,
    )
    assert res["criterion"] == "mrmr_jmi"
    assert res["redundancy_mode"] == "pearson"


def test_mrmr_tiered_mode_small_n_uses_pearson():
    X, y = _make_binary(n=70, p=35, seed=23)
    fs = FeatureSelector(problem_type="classification", random_state=0)
    res, _ = mrmr_jmi_selection(
        X,
        y,
        n_target_features=7,
        random_state=0,
        mi_scorer=fs.mi_scorer,
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
        mrmr_max_features=30,
        mrmr_redundancy_weight=0.55,
        mrmr_mi_redundancy_enabled=True,
        mrmr_mi_n_bins=8,
    )
    assert res["criterion"] == "mrmr_jmi_mi"
    assert res["redundancy_mode"] == "pearson"


def test_mrmr_tiered_mode_medium_n_uses_binned_mi():
    X, y = _make_binary(n=100, p=35, seed=29)
    fs = FeatureSelector(problem_type="classification", random_state=0)
    res, _ = mrmr_jmi_selection(
        X,
        y,
        n_target_features=7,
        random_state=0,
        mi_scorer=fs.mi_scorer,
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
        mrmr_max_features=30,
        mrmr_redundancy_weight=0.55,
        mrmr_mi_redundancy_enabled=True,
        mrmr_mi_n_bins=8,
    )
    assert res["redundancy_mode"] == "binned_mi"


def test_mrmr_tiered_mode_large_n_uses_knn_mi():
    X, y = _make_binary(n=180, p=35, seed=31)
    fs = FeatureSelector(problem_type="classification", random_state=0)
    res, _ = mrmr_jmi_selection(
        X,
        y,
        n_target_features=7,
        random_state=0,
        mi_scorer=fs.mi_scorer,
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
        mrmr_max_features=30,
        mrmr_redundancy_weight=0.55,
        mrmr_mi_redundancy_enabled=True,
        mrmr_mi_n_bins=8,
    )
    assert res["redundancy_mode"] == "knn_mi"


def test_feature_selector_dispatch_runs_cmim_and_fcbf():
    X, y = _make_binary(n=120, p=40, seed=41)
    fs = FeatureSelector(
        problem_type="classification",
        random_state=0,
        enabled_methods={"cmim", "fcbf"},
        cmim_min_samples=60,
    )
    method_results, _ = fs._run_selection_methods(X, y, n_target=8)
    assert "cmim" in method_results
    assert "fcbf" in method_results
    cmim_res, _ = method_results["cmim"]
    fcbf_res, _ = method_results["fcbf"]
    assert "selected_indices" in cmim_res
    assert "selected_indices" in fcbf_res


def test_cmim_overlap_with_mrmr_on_linear_data():
    X, y = _make_binary(n=180, p=60, seed=53)
    fs = FeatureSelector(problem_type="classification", random_state=0)
    cmim_res, _ = cmim_selection(
        X,
        y,
        n_target_features=12,
        random_state=0,
        mi_scorer=fs.mi_scorer,
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
        max_features=50,
        min_samples=60,
        n_bins=8,
    )
    mrmr_res, _ = mrmr_jmi_selection(
        X,
        y,
        n_target_features=12,
        random_state=0,
        mi_scorer=fs.mi_scorer,
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
        mrmr_max_features=50,
        mrmr_redundancy_weight=0.55,
        mrmr_mi_redundancy_enabled=True,
        mrmr_mi_n_bins=8,
    )
    overlap = _jaccard(cmim_res["selected_indices"], mrmr_res["selected_indices"])
    assert overlap >= 0.50


def test_cmim_differs_from_mrmr_on_xor_like_interaction_data():
    rng = np.random.RandomState(71)
    n = 220
    z0 = rng.randint(0, 2, size=n).astype(float)
    z1 = rng.randint(0, 2, size=n).astype(float)
    y = np.logical_xor(z0 > 0.5, z1 > 0.5).astype(int)
    X = rng.normal(scale=0.4, size=(n, 40))
    X[:, 0] = z0 + 0.1 * rng.normal(size=n)
    X[:, 1] = z1 + 0.1 * rng.normal(size=n)
    X[:, 2] = y + 0.2 * rng.normal(size=n)
    X[:, 3] = y + 0.2 * rng.normal(size=n)

    fs = FeatureSelector(problem_type="classification", random_state=0)
    cmim_res, _ = cmim_selection(
        X,
        y,
        n_target_features=10,
        random_state=0,
        mi_scorer=fs.mi_scorer,
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
        max_features=35,
        min_samples=60,
        n_bins=8,
    )
    mrmr_res, _ = mrmr_jmi_selection(
        X,
        y,
        n_target_features=10,
        random_state=0,
        mi_scorer=fs.mi_scorer,
        prefilter_fn=_identity_prefilter,
        normalize_fn=_normalize,
        mrmr_max_features=35,
        mrmr_redundancy_weight=0.55,
        mrmr_mi_redundancy_enabled=True,
        mrmr_mi_n_bins=8,
    )
    overlap = _jaccard(cmim_res["selected_indices"], mrmr_res["selected_indices"])
    assert overlap <= 0.80

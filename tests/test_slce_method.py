import numpy as np

from tabnetics.feature_selection.methods.slce import slce_centroid_encoder_selection


def _normalize(v):
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _prefilter_all(X, y, *, max_features):
    n_features = int(np.asarray(X).shape[1])
    return np.arange(min(n_features, int(max_features)), dtype=int)


def _make_binary(seed: int = 30, *, n: int = 90, p: int = 64):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, size=n)
    X = rng.normal(size=(n, p))
    X[:, 0] += 0.9 * y
    X[:, 1] += 0.7 * (1 - y)
    X[:, 2] += 0.4 * y
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def test_slce_returns_feature_set_and_metadata_on_binary_hdlss():
    X, y = _make_binary()
    res, all_scores = slce_centroid_encoder_selection(
        X,
        y,
        n_target_features=12,
        problem_type="classification",
        random_state=30,
        slce_prefilter_max_features=48,
        slce_min_samples=30,
        slce_ridge=1.0,
        prefilter_fn=_prefilter_all,
        normalize_fn=_normalize,
    )
    assert len(res.get("selected_indices", [])) == 12
    assert int(res.get("slce_pool_size", 0)) > 0
    assert str(res.get("slce_solver", "")) in {"woodbury", "direct"}
    assert len(all_scores) == X.shape[1]


def test_slce_gate_blocks_non_binary_or_small_n():
    X, y = _make_binary(n=24)
    res_small, _ = slce_centroid_encoder_selection(
        X,
        y,
        n_target_features=8,
        problem_type="classification",
        random_state=31,
        slce_prefilter_max_features=48,
        slce_min_samples=30,
        slce_ridge=1.0,
        prefilter_fn=_prefilter_all,
        normalize_fn=_normalize,
    )
    assert res_small == {}

    rng = np.random.RandomState(32)
    X_mc = rng.normal(size=(90, 40))
    y_mc = rng.randint(0, 3, size=90)
    res_mc, _ = slce_centroid_encoder_selection(
        X_mc,
        y_mc,
        n_target_features=8,
        problem_type="classification",
        random_state=32,
        slce_prefilter_max_features=40,
        slce_min_samples=30,
        slce_ridge=1.0,
        prefilter_fn=_prefilter_all,
        normalize_fn=_normalize,
    )
    assert res_mc == {}

import numpy as np

from tabnetics.feature_selection.methods import multiclass as multiclass_mod


def _norm(v):
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if (mx - mn) < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def test_ecoc_binary_relevance_cache_parity():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 12))
    y = rng.integers(0, 2, size=60)

    no_cache = multiclass_mod.ecoc_binary_relevance_scores(
        X,
        y,
        n_features=12,
        random_state=7,
        ova_linear_backend="linear_svm_l1",
        linear_svm_max_iter=200,
        normalize_fn=_norm,
    )

    cache = {}
    with_cache = multiclass_mod.ecoc_binary_relevance_scores(
        X,
        y,
        n_features=12,
        random_state=7,
        ova_linear_backend="linear_svm_l1",
        linear_svm_max_iter=200,
        normalize_fn=_norm,
        score_cache=cache,
        cache_context_key=("ecoc", "task1", 60, 30, 30, "linear_svm_l1", 200, 12),
    )

    np.testing.assert_allclose(no_cache, with_cache, atol=0.0, rtol=0.0)


def test_ecoc_binary_relevance_cache_reuses_result(monkeypatch):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 10))
    y = rng.integers(0, 2, size=50)

    calls = {"mi": 0}
    original_mi = multiclass_mod.mutual_info_classif

    def _wrapped_mi(*args, **kwargs):
        calls["mi"] += 1
        return original_mi(*args, **kwargs)

    monkeypatch.setattr(multiclass_mod, "mutual_info_classif", _wrapped_mi)

    cache = {}
    key = ("ova", "class0", 50, int(np.sum(y == 1)), int(np.sum(y == 0)), "linear_svm_l1", 200, 10)
    _ = multiclass_mod.ecoc_binary_relevance_scores(
        X,
        y,
        n_features=10,
        random_state=3,
        ova_linear_backend="linear_svm_l1",
        linear_svm_max_iter=200,
        normalize_fn=_norm,
        score_cache=cache,
        cache_context_key=key,
    )
    _ = multiclass_mod.ecoc_binary_relevance_scores(
        X,
        y,
        n_features=10,
        random_state=3,
        ova_linear_backend="linear_svm_l1",
        linear_svm_max_iter=200,
        normalize_fn=_norm,
        score_cache=cache,
        cache_context_key=key,
    )

    assert calls["mi"] == 1

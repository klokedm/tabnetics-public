import numpy as np

from tabnetics.feature_selection.methods.hsic import hsic_lasso_selection
from tabnetics.feature_selection.prefilter import center_kernel_matrix, rbf_kernel_1d


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
    return np.arange(min(int(max_features), n_features), dtype=int)


def _make_binary(seed: int, *, n: int = 72, p: int = 40):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, size=n)
    X = rng.normal(size=(n, p))
    X[:, 0] = y + 0.05 * rng.normal(size=n)
    X[:, 1] = (1 - y) + 0.07 * rng.normal(size=n)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def test_hsic_uses_binary_delta_kernel_when_gate_satisfied():
    X, y = _make_binary(10, n=72)
    res, _ = hsic_lasso_selection(
        X,
        y,
        n_target_features=10,
        problem_type="classification",
        random_state=10,
        hsic_lasso_prefilter_max_features=40,
        hsic_lasso_target_sigma=0.0,
        hsic_lasso_feature_sigma=0.0,
        hsic_lasso_alpha=0.01,
        hsic_lasso_max_iter=2000,
        hsic_lasso_relevance_blend=0.20,
        hsic_lasso_binary_delta_enabled=True,
        hsic_lasso_binary_delta_min_samples=30,
        prefilter_fn=_prefilter_all,
        normalize_fn=_normalize,
        rbf_kernel_1d_fn=rbf_kernel_1d,
        center_kernel_matrix_fn=center_kernel_matrix,
    )
    assert res["hsic_lasso_target_kernel"] == "delta_binary"
    assert bool(res["hsic_lasso_binary_delta_applied"]) is True


def test_hsic_binary_delta_gate_blocks_small_n():
    X, y = _make_binary(11, n=24)
    res, _ = hsic_lasso_selection(
        X,
        y,
        n_target_features=8,
        problem_type="classification",
        random_state=11,
        hsic_lasso_prefilter_max_features=24,
        hsic_lasso_target_sigma=0.0,
        hsic_lasso_feature_sigma=0.0,
        hsic_lasso_alpha=0.01,
        hsic_lasso_max_iter=2000,
        hsic_lasso_relevance_blend=0.20,
        hsic_lasso_binary_delta_enabled=True,
        hsic_lasso_binary_delta_min_samples=30,
        prefilter_fn=_prefilter_all,
        normalize_fn=_normalize,
        rbf_kernel_1d_fn=rbf_kernel_1d,
        center_kernel_matrix_fn=center_kernel_matrix,
    )
    assert res["hsic_lasso_target_kernel"] == "rbf_label"
    assert bool(res["hsic_lasso_binary_delta_applied"]) is False


def test_hsic_binary_delta_can_be_disabled():
    X, y = _make_binary(12, n=72)
    res, _ = hsic_lasso_selection(
        X,
        y,
        n_target_features=10,
        problem_type="classification",
        random_state=12,
        hsic_lasso_prefilter_max_features=40,
        hsic_lasso_target_sigma=0.0,
        hsic_lasso_feature_sigma=0.0,
        hsic_lasso_alpha=0.01,
        hsic_lasso_max_iter=2000,
        hsic_lasso_relevance_blend=0.20,
        hsic_lasso_binary_delta_enabled=False,
        hsic_lasso_binary_delta_min_samples=30,
        prefilter_fn=_prefilter_all,
        normalize_fn=_normalize,
        rbf_kernel_1d_fn=rbf_kernel_1d,
        center_kernel_matrix_fn=center_kernel_matrix,
    )
    assert res["hsic_lasso_target_kernel"] == "rbf_label"
    assert bool(res["hsic_lasso_binary_delta_applied"]) is False


def test_hsic_multiclass_uses_rbf_label_kernel():
    rng = np.random.RandomState(13)
    X = rng.normal(size=(72, 32))
    y = rng.randint(0, 3, size=72)
    res, _ = hsic_lasso_selection(
        X,
        y,
        n_target_features=10,
        problem_type="classification",
        random_state=13,
        hsic_lasso_prefilter_max_features=32,
        hsic_lasso_target_sigma=0.0,
        hsic_lasso_feature_sigma=0.0,
        hsic_lasso_alpha=0.01,
        hsic_lasso_max_iter=2000,
        hsic_lasso_relevance_blend=0.20,
        hsic_lasso_binary_delta_enabled=True,
        hsic_lasso_binary_delta_min_samples=30,
        prefilter_fn=_prefilter_all,
        normalize_fn=_normalize,
        rbf_kernel_1d_fn=rbf_kernel_1d,
        center_kernel_matrix_fn=center_kernel_matrix,
    )
    assert res["hsic_lasso_target_kernel"] == "rbf_label"
    assert bool(res["hsic_lasso_binary_delta_applied"]) is False

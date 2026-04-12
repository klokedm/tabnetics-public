import numpy as np

from tabnetics.feature_selection.prefilter import (
    _detect_rnaseq_data,
    _rnaseq_transform,
    rnaseq_nb_lrt_scores,
)


def _make_nb_counts(
    *,
    n_samples: int = 90,
    n_features: int = 40,
    n_signal: int = 6,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2), dtype=int)
    X = np.zeros((n_samples, n_features), dtype=float)
    r = 8.0

    for j in range(n_features):
        if j < n_signal:
            mu0, mu1 = 20.0, 65.0
        else:
            mu0, mu1 = 30.0, 30.0

        p0 = r / (r + mu0)
        p1 = r / (r + mu1)
        X[y == 0, j] = rng.negative_binomial(r, p0, size=int(np.sum(y == 0)))
        X[y == 1, j] = rng.negative_binomial(r, p1, size=int(np.sum(y == 1)))
    return X, y


def test_detect_rnaseq_forced_domain():
    X = np.array([[0, 1, 2], [3, 0, 4]], dtype=float)
    is_rnaseq, meta = _detect_rnaseq_data(X, data_domain="rnaseq")
    assert is_rnaseq is True
    assert meta["reason"] == "forced_domain"


def test_detect_rnaseq_auto_count_like():
    X = np.array(
        [
            [0, 12, 0, 5],
            [0, 8, 0, 4],
            [1, 15, 0, 0],
            [0, 9, 1, 3],
        ],
        dtype=float,
    )
    is_rnaseq, meta = _detect_rnaseq_data(X, data_domain="auto")
    assert is_rnaseq is True
    assert meta["reason"] in {"count_like_heuristic", "forced_domain"}


def test_rnaseq_transform_handles_all_zero_matrix():
    X = np.zeros((12, 7), dtype=float)
    out, meta = _rnaseq_transform(X, data_domain="rnaseq", enabled=True)
    assert bool(meta["rnaseq_transform_applied"]) is True
    assert np.isfinite(out).all()
    assert np.allclose(out, 0.0)


def test_rnaseq_transform_handles_all_zero_column():
    X, _ = _make_nb_counts(n_samples=40, n_features=10, n_signal=3, seed=11)
    X[:, 2] = 0.0
    out, meta = _rnaseq_transform(X, data_domain="rnaseq", enabled=True)
    assert bool(meta["rnaseq_transform_applied"]) is True
    assert np.isfinite(out).all()
    assert np.allclose(out[:, 2], 0.0)


def test_rnaseq_nb_lrt_recovers_signal_on_synthetic_counts():
    X, y = _make_nb_counts(n_samples=100, n_features=50, n_signal=8, seed=7)
    scores, meta = rnaseq_nb_lrt_scores(X, y, data_domain="rnaseq", alpha=0.10)
    assert bool(meta["rnaseq_nb_lrt_applied"]) is True
    assert np.isfinite(scores).all()
    top10 = set(np.argsort(scores)[::-1][:10].tolist())
    signal = set(range(8))
    assert len(top10.intersection(signal)) >= 4


def test_rnaseq_nb_lrt_skips_non_rnaseq_data():
    rng = np.random.default_rng(13)
    X = rng.normal(0.0, 1.0, size=(60, 20))
    y = np.array([0] * 30 + [1] * 30, dtype=int)
    scores, meta = rnaseq_nb_lrt_scores(X, y, data_domain="auto", alpha=0.10)
    assert bool(meta["rnaseq_nb_lrt_applied"]) is False
    assert meta["rnaseq_nb_lrt_reason"] in {"not_rnaseq", "negative_values"}
    assert np.allclose(scores, 0.0)

import numpy as np
import pytest

import tabnetics.feature_selection.copula as copula_module
from tabnetics.feature_selection.copula import (
    CopulaKnockoffSelector,
    _ebh_support,
    _knockoff_threshold,
)


def test_knockoff_threshold_returns_infinity_when_no_valid_cutoff():
    # Monotone-positive W cannot satisfy the knockoff+ ratio for alpha=0.1.
    W = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=float)
    threshold = _knockoff_threshold(W, alpha=0.1)
    assert np.isinf(threshold)


def test_knockoff_threshold_simulation_controls_empirical_fdr():
    rng = np.random.default_rng(2026)
    alpha = 0.20
    p = 200
    n_signals = 30
    n_repeats = 350

    fdp_values = []
    for _ in range(n_repeats):
        w_signal = rng.normal(loc=3.0, scale=1.0, size=n_signals)
        w_null = rng.standard_t(df=4, size=p - n_signals)
        is_null = np.concatenate(
            [
                np.zeros(n_signals, dtype=bool),
                np.ones(p - n_signals, dtype=bool),
            ]
        )
        W = np.concatenate([w_signal, w_null])
        perm = rng.permutation(p)
        W = W[perm]
        is_null = is_null[perm]

        threshold = _knockoff_threshold(W, alpha=alpha)
        selected = np.where(W >= threshold)[0]
        if selected.size == 0:
            fdp_values.append(0.0)
        else:
            fdp_values.append(float(np.mean(is_null[selected])))

    empirical_fdr = float(np.mean(fdp_values))
    # Finite-sample Monte-Carlo tolerance around target alpha=0.20.
    assert empirical_fdr <= 0.25


def test_ebh_support_selects_prefix_of_sorted_evalues():
    e_values = np.array([30.0, 14.0, 1.0, 0.5, 0.2], dtype=float)
    support = _ebh_support(e_values, alpha=0.2)
    assert np.array_equal(support, np.array([0, 1], dtype=int))


def test_copula_selector_avoids_degenerate_duplicated_2p_vine(monkeypatch):
    fit_dims = []
    captured = {}

    class FakeVine:
        def __init__(self, p):
            self._p = p
            self._duplicate_2p = False

        @classmethod
        def from_data(cls, data, structure=None, controls=None):
            arr = np.asarray(data, dtype=float)
            p = int(arr.shape[1])
            fit_dims.append(p)
            obj = cls(p)
            if p % 2 == 0 and p > 2:
                half = p // 2
                obj._duplicate_2p = bool(np.allclose(arr[:, :half], arr[:, half:]))
            return obj

        def rosenblatt(self, x):
            return np.asarray(x, dtype=float)

        def inverse_rosenblatt(self, u):
            arr = np.asarray(u, dtype=float)
            if self._duplicate_2p and arr.shape[1] == self._p:
                half = self._p // 2
                return np.concatenate([arr[:, :half], arr[:, :half]], axis=1)
            return arr

    class FakeDVine:
        def __init__(self, order):
            self.order = order

    class FakeControls:
        def __init__(self, **kwargs):
            pass

    def fake_lcd_stat(X, Xtilde, y, cv_kwargs):
        captured["X"] = np.asarray(X, dtype=float)
        captured["Xtilde"] = np.asarray(Xtilde, dtype=float)
        p = int(X.shape[1])
        return np.linspace(1.0, 0.1, num=p, dtype=float)

    monkeypatch.setattr(copula_module, "Vinecop", FakeVine)
    monkeypatch.setattr(copula_module, "DVineStructure", FakeDVine)
    monkeypatch.setattr(copula_module, "FitControlsVinecop", FakeControls)
    monkeypatch.setattr(copula_module, "_lcd_stat", fake_lcd_stat)

    rng = np.random.default_rng(2026)
    X = rng.normal(size=(48, 6)).astype(float)
    y = rng.integers(0, 2, size=48).astype(int)

    selector = CopulaKnockoffSelector(
        M=1,
        alpha_kn=0.2,
        alpha_ebh=0.2,
        conditional_bridge_rho=0.45,
        random_state=7,
    )
    selector.fit(X, y)

    assert fit_dims, "expected at least one vine fit call"
    # Regression guard: knockoff generation must not fit a duplicated 2p vine.
    assert max(fit_dims) == X.shape[1]
    assert selector.truncation_level_effective_.get("vine_2p", "missing") is None

    X_orig = np.asarray(captured["X"], dtype=float)
    X_knock = np.asarray(captured["Xtilde"], dtype=float)
    corr = []
    for j in range(X_orig.shape[1]):
        c = np.corrcoef(X_orig[:, j], X_knock[:, j])[0, 1]
        corr.append(abs(float(c)) if np.isfinite(c) else 0.0)
    # Knockoffs should not collapse to originals across all dimensions.
    assert float(np.mean(corr)) < 0.995


def test_repeated_fit_same_instance_is_deterministic(monkeypatch):
    class FakeVine:
        def __init__(self, p):
            self._p = p

        @classmethod
        def from_data(cls, data, structure=None, controls=None):
            return cls(int(np.asarray(data).shape[1]))

        def rosenblatt(self, x):
            return np.asarray(x, dtype=float)

        def inverse_rosenblatt(self, u):
            return np.asarray(u, dtype=float)

    class FakeDVine:
        def __init__(self, order):
            self.order = order

    class FakeControls:
        def __init__(self, **kwargs):
            pass

    def fake_lcd_stat(X, Xtilde, y, cv_kwargs):
        p = int(X.shape[1])
        rng = np.random.RandomState(int(cv_kwargs.get("random_state", 0)))
        return rng.standard_normal(p)

    monkeypatch.setattr(copula_module, "Vinecop", FakeVine)
    monkeypatch.setattr(copula_module, "DVineStructure", FakeDVine)
    monkeypatch.setattr(copula_module, "FitControlsVinecop", FakeControls)
    monkeypatch.setattr(copula_module, "_lcd_stat", fake_lcd_stat)

    rng = np.random.default_rng(123)
    X = rng.normal(size=(36, 5)).astype(float)
    y = rng.integers(0, 2, size=36).astype(int)

    selector = CopulaKnockoffSelector(
        M=4,
        alpha_kn=0.2,
        alpha_ebh=0.2,
        conditional_bridge_rho=0.4,
        random_state=17,
    )
    selector.fit(X, y)
    e_avg_1 = selector.e_avg_.copy()
    support_1 = selector.support_.copy()

    selector.fit(X, y)
    e_avg_2 = selector.e_avg_.copy()
    support_2 = selector.support_.copy()

    np.testing.assert_allclose(e_avg_1, e_avg_2, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(support_1, support_2)


def test_low_information_diagnostics_contract(monkeypatch):
    class FakeVine:
        def __init__(self, p):
            self._p = p

        @classmethod
        def from_data(cls, data, structure=None, controls=None):
            return cls(int(np.asarray(data).shape[1]))

        def rosenblatt(self, x):
            return np.asarray(x, dtype=float)

        def inverse_rosenblatt(self, u):
            return np.asarray(u, dtype=float)

    class FakeDVine:
        def __init__(self, order):
            self.order = order

    class FakeControls:
        def __init__(self, **kwargs):
            pass

    def fake_lcd_stat(X, Xtilde, y, cv_kwargs):
        # Strongly negative W -> no support expected.
        return -np.ones(int(X.shape[1]), dtype=float)

    monkeypatch.setattr(copula_module, "Vinecop", FakeVine)
    monkeypatch.setattr(copula_module, "DVineStructure", FakeDVine)
    monkeypatch.setattr(copula_module, "FitControlsVinecop", FakeControls)
    monkeypatch.setattr(copula_module, "_lcd_stat", fake_lcd_stat)

    rng = np.random.default_rng(3)
    X = rng.normal(size=(24, 6)).astype(float)
    y = rng.integers(0, 2, size=24).astype(int)

    selector = CopulaKnockoffSelector(M=2, random_state=11)
    selector.fit(X, y)
    diag = dict(selector.low_information_diagnostics_)
    assert diag["n_samples"] == 24
    assert diag["n_features"] == 6
    assert diag["n_effective"] == 6
    assert "reason_code" in diag
    assert isinstance(diag["reason_code"], str)


def test_deepdrk_generator_runs_without_vinecop(monkeypatch):
    monkeypatch.setattr(copula_module, "Vinecop", None)
    monkeypatch.setattr(
        copula_module,
        "_lcd_stat",
        lambda X, Xtilde, y, cv_kwargs: np.linspace(1.0, 0.1, num=int(X.shape[1]), dtype=float),
    )
    rng = np.random.default_rng(123)
    X = rng.normal(size=(40, 8)).astype(float)
    y = rng.integers(0, 2, size=40).astype(int)

    selector = CopulaKnockoffSelector(
        M=2,
        alpha_kn=0.2,
        alpha_ebh=0.2,
        generator="deepdrk",
        deepdrk_latent_fraction=0.4,
        deepdrk_noise_scale=1.1,
        random_state=7,
    )
    selector.fit(X, y)
    assert selector.e_avg_.shape == (8,)
    assert selector.support_.ndim == 1
    assert str(selector.truncation_level_effective_.get("generator")) == "deepdrk"


def test_deepdrk_generator_is_deterministic_with_same_seed(monkeypatch):
    monkeypatch.setattr(copula_module, "Vinecop", None)
    monkeypatch.setattr(
        copula_module,
        "_lcd_stat",
        lambda X, Xtilde, y, cv_kwargs: np.random.RandomState(
            int(cv_kwargs.get("random_state", 0))
        ).standard_normal(int(X.shape[1])),
    )
    rng = np.random.default_rng(99)
    X = rng.normal(size=(36, 7)).astype(float)
    y = rng.integers(0, 3, size=36).astype(int)

    s1 = CopulaKnockoffSelector(
        M=3,
        generator="deepdrk",
        deepdrk_latent_fraction=0.3,
        deepdrk_noise_scale=0.8,
        random_state=17,
    ).fit(X, y)
    s2 = CopulaKnockoffSelector(
        M=3,
        generator="deepdrk",
        deepdrk_latent_fraction=0.3,
        deepdrk_noise_scale=0.8,
        random_state=17,
    ).fit(X, y)

    np.testing.assert_allclose(s1.e_avg_, s2.e_avg_, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(s1.support_, s2.support_)


def test_copula_mode_without_vinecop_raises_importerror(monkeypatch):
    monkeypatch.setattr(copula_module, "Vinecop", None)
    rng = np.random.default_rng(8)
    X = rng.normal(size=(24, 5)).astype(float)
    y = rng.integers(0, 2, size=24).astype(int)

    with pytest.raises(ImportError):
        CopulaKnockoffSelector(M=1, generator="copula", random_state=3).fit(X, y)

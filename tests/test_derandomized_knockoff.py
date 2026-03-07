import numpy as np

from tabnetics.feature_selection.methods.knockoff import copula_knockoff_selection


class _FakeCopulaSelector:
    def __init__(
        self,
        M,
        alpha_kn,
        alpha_ebh,
        truncation_level,
        generator,
        deepdrk_latent_fraction,
        deepdrk_noise_scale,
        show_progress,
        random_state,
    ):
        self.random_state = int(random_state)
        self.support_ = np.array([], dtype=int)
        self.e_avg_ = np.array([], dtype=float)
        self.truncation_level_effective_ = {"vine_x": None, "vine_2p": None}
        self.low_information_diagnostics_ = {
            "reason_code": "ok",
            "n_nonzero_e_values": 1,
            "n_support": 1,
        }

    def fit(self, X, y):
        p = int(np.asarray(X).shape[1])
        rng = np.random.RandomState(self.random_state)
        self.e_avg_ = np.full(p, 0.2, dtype=float)
        self.e_avg_[2 % p] = 50.0 + float(rng.randint(0, 3))
        self.support_ = np.array([2 % p], dtype=int)
        return self

    def get_support(self):
        return np.asarray(self.support_, dtype=int)

    def get_weights(self, eps: float = 1e-6):
        w = np.asarray(self.e_avg_, dtype=float)
        denom = float(np.max(w)) + float(eps)
        return w / denom if denom > 0 else w


class _EmptySupportSelector(_FakeCopulaSelector):
    def fit(self, X, y):
        p = int(np.asarray(X).shape[1])
        self.e_avg_ = np.zeros(p, dtype=float)
        self.support_ = np.array([], dtype=int)
        self.low_information_diagnostics_ = {
            "reason_code": "all_zero_e_values",
            "n_nonzero_e_values": 0,
            "n_support": 0,
        }
        return self


class _ShortWeightSelector(_FakeCopulaSelector):
    def get_weights(self, eps: float = 1e-6):
        # Deliberately wrong length to test padding path.
        return np.array([1.0, 0.5], dtype=float)


def _toy_data():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(36, 8)).astype(float)
    y = rng.integers(0, 2, size=36).astype(int)
    return X, y


def _run(selector_cls, **kwargs):
    X, y = _toy_data()
    return copula_knockoff_selection(
        X,
        y,
        n_target_features=4,
        CopulaKnockoffSelectorClass=selector_cls,
        copula_knockoff_draws=10,
        copula_alpha_kn=0.1,
        copula_alpha_ebh=0.2,
        copula_truncation_level=None,
        copula_generator="copula",
        copula_deepdrk_latent_fraction=0.35,
        copula_deepdrk_noise_scale=1.0,
        copula_derandomize_runs=int(kwargs.get("copula_derandomize_runs", 1)),
        copula_stabilizer_runs=int(kwargs.get("copula_stabilizer_runs", 1)),
        copula_stabilizer_use_ebh=bool(kwargs.get("copula_stabilizer_use_ebh", False)),
        copula_stabilizer_seed_stride=997,
        random_state=11,
    )


def test_invalid_shape_returns_empty():
    res, scores = copula_knockoff_selection(
        np.zeros((1, 0), dtype=float),
        np.zeros(1, dtype=int),
        n_target_features=3,
        CopulaKnockoffSelectorClass=_FakeCopulaSelector,
        copula_knockoff_draws=5,
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
        random_state=7,
    )
    assert res == {}
    assert scores == {}


def test_legacy_single_run_uses_selector_support():
    res, _ = _run(_FakeCopulaSelector, copula_derandomize_runs=1, copula_stabilizer_runs=1)
    sel = np.asarray(res["selected_indices"], dtype=int)
    assert int(sel.size) >= 1
    assert int(sel[0]) == 2
    assert bool(res["copula_derandomized_mode"]) is False


def test_derandomized_mode_aggregates_evalues_with_ebh():
    res, _ = _run(_FakeCopulaSelector, copula_derandomize_runs=5, copula_stabilizer_runs=1)
    sel = np.asarray(res["selected_indices"], dtype=int)
    assert 2 in set(sel.tolist())
    assert bool(res["copula_derandomized_mode"]) is True
    assert int(res["copula_derandomize_runs"]) == 5


def test_derandomized_metadata_fields_present():
    res, _ = _run(_FakeCopulaSelector, copula_derandomize_runs=4, copula_stabilizer_runs=3)
    assert "copula_derandomized_e_values" in res
    assert "copula_stabilizer_legacy_runs" in res
    assert int(res["copula_stabilizer_runs"]) == 4
    assert int(res["copula_stabilizer_legacy_runs"]) == 3


def test_derandomized_empty_support_falls_back_to_top_weights():
    res, _ = _run(_EmptySupportSelector, copula_derandomize_runs=6, copula_stabilizer_runs=1)
    sel = np.asarray(res["selected_indices"], dtype=int)
    assert int(sel.size) >= 1
    assert bool(res["copula_stabilizer_fallback_used"]) is True


def test_legacy_stabilizer_ebh_path_keeps_metadata():
    res, _ = _run(
        _FakeCopulaSelector,
        copula_derandomize_runs=1,
        copula_stabilizer_runs=3,
        copula_stabilizer_use_ebh=True,
    )
    assert bool(res["copula_derandomized_mode"]) is False
    assert bool(res["copula_stabilizer_use_ebh"]) is True
    assert int(res["copula_stabilizer_runs"]) == 3


def test_weight_padding_path_handles_short_vectors():
    res, scores = _run(_ShortWeightSelector, copula_derandomize_runs=1, copula_stabilizer_runs=1)
    assert len(scores) == 8
    assert set(range(8)).issuperset(set(int(k) for k in scores.keys()))
    assert np.asarray(res["copula_stabilizer_support_frequency"]).shape[0] == 8


def test_derandomize_runs_takes_priority_over_legacy_stabilizer_runs():
    res, _ = _run(_FakeCopulaSelector, copula_derandomize_runs=7, copula_stabilizer_runs=2)
    assert int(res["copula_derandomize_runs"]) == 7
    assert int(res["copula_stabilizer_runs"]) == 7
    assert int(res["copula_stabilizer_legacy_runs"]) == 2

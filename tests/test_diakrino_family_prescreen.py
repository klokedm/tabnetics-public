"""S5: DIAKRINO family prescreen in DistributionFitter.generate_candidates.

Pins: default-off is a strict no-op; when enabled it narrows the candidate families to
the DIAKRINO top-K (continuous) while always keeping the mandatory floor; and a high-entropy
(undecided) family head falls through to the full set.  Replay-safe by construction:
scipy still selects the final family from the kept set and that family persists as usual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as sps

from tabnetics.distribution.selector import FitResult, TransformInfo
from tabnetics.pipeline.pipeline import DistributionFitter, DistributionFitterConfig

# Canonical family ids (0..30 continuous): 4=gamma, 5=lognorm.
GAMMA_ID, LOGNORM_ID = 4, 5


def _sidecar(tmp_path):
    # feature 0: peaked on {gamma, lognorm}; feature 1: uniform logits (high entropy)
    peaked = np.full(36, -10.0); peaked[GAMMA_ID] = 10.0; peaked[LOGNORM_ID] = 10.0
    flat = np.zeros(36)
    df = pd.DataFrame({
        "feature_index": [0, 1],
        "chunk_id": [0, 0],
        "population_family_logits": [peaked.tolist(), flat.tolist()],
        "prior_logit": [1.0, 1.0],
    })
    p = tmp_path / "ds.parquet"
    df.to_parquet(p)
    return str(p)


def _positive_data():
    rng = np.random.default_rng(0)
    return rng.lognormal(mean=0.0, sigma=1.0, size=400)  # positive support


class _FakeSelector:
    def __init__(self, results):
        self._results = list(results)

    def select_best_distribution(self, data, criterion="simple", verbose=False):
        return self._results[0].name, self._results[0], list(self._results)


def _fit_result(name: str, p_value: float) -> FitResult:
    return FitResult(
        name=name,
        params=(1.0, 0.0, 1.0) if name == "gamma" else (0.0, 1.0),
        transform_info=TransformInfo(),
        cvm_p=float(p_value),
        ks_p=float(p_value),
        success=True,
    )


def _patch_fake_selector(monkeypatch, fitter: DistributionFitter, results):
    monkeypatch.setattr(
        fitter,
        "generate_candidates",
        lambda audit, feature_index=-1: {"norm": sps.norm, "gamma": sps.gamma},
    )
    monkeypatch.setattr(
        fitter,
        "_build_selector",
        lambda *args, **kwargs: _FakeSelector(results),
    )


def test_default_is_strict_noop():
    data = _positive_data()
    off = DistributionFitter(DistributionFitterConfig())
    audit = off.audit_data(data)
    base = set(off.generate_candidates(audit, feature_index=0).keys())
    # same fitter, explicit feature_index, no sidecar -> identical
    assert base == set(off.generate_candidates(audit, feature_index=0).keys())
    assert base == set(off.generate_candidates(audit).keys())  # default arg too


def test_prescreen_narrows_and_keeps_mandatory(tmp_path):
    data = _positive_data()
    off = DistributionFitter(DistributionFitterConfig())
    audit = off.audit_data(data)
    full = set(off.generate_candidates(audit, feature_index=0).keys())

    on = DistributionFitter(DistributionFitterConfig(
        diakrino_family_prescreen_enabled=True, diakrino_family_prescreen_top_k=2,
        diakrino_sidecar_path=_sidecar(tmp_path),
    ))
    kept = set(on.generate_candidates(audit, feature_index=0).keys())

    assert kept <= full and kept != full          # strictly narrowed
    assert {"gamma", "lognorm"} <= kept            # DIAKRINO top-K survive (both positive-support)
    # mandatory floor (intersected with positive support) is preserved
    assert {"norm", "t", "gamma", "lognorm", "weibull_min"} <= kept
    assert kept                                     # never empty


def test_high_entropy_falls_through(tmp_path):
    data = _positive_data()
    off = DistributionFitter(DistributionFitterConfig())
    audit = off.audit_data(data)
    full = set(off.generate_candidates(audit, feature_index=1).keys())

    on = DistributionFitter(DistributionFitterConfig(
        diakrino_family_prescreen_enabled=True, diakrino_family_prescreen_top_k=2,
        diakrino_sidecar_path=_sidecar(tmp_path),
    ))
    # feature 1 has uniform logits -> high entropy -> no prescreen
    kept = set(on.generate_candidates(audit, feature_index=1).keys())
    assert kept == full


def test_missing_sidecar_is_noop(tmp_path):
    data = _positive_data()
    off = DistributionFitter(DistributionFitterConfig())
    audit = off.audit_data(data)
    full = set(off.generate_candidates(audit, feature_index=0).keys())
    on = DistributionFitter(DistributionFitterConfig(
        diakrino_family_prescreen_enabled=True, diakrino_sidecar_path="/nonexistent/x.parquet",
    ))
    assert set(on.generate_candidates(audit, feature_index=0).keys()) == full


def test_family_prior_lambda_zero_preserves_selector_choice(tmp_path, monkeypatch):
    results = [_fit_result("norm", 0.80), _fit_result("gamma", 0.60)]
    fitter = DistributionFitter(DistributionFitterConfig(
        compute_crps_uq_decomposition=False,
        diakrino_family_prior_lambda=0.0,
        diakrino_sidecar_path=_sidecar(tmp_path),
    ))
    _patch_fake_selector(monkeypatch, fitter, results)

    summary = fitter.select_best_distribution(_positive_data(), criterion="simple", feature_index=0)

    assert summary.family == "norm"


def test_family_prior_can_softly_reorder_simple_selection(tmp_path, monkeypatch):
    results = [_fit_result("norm", 0.80), _fit_result("gamma", 0.60)]
    fitter = DistributionFitter(DistributionFitterConfig(
        compute_crps_uq_decomposition=False,
        diakrino_family_prior_lambda=0.03,
        diakrino_sidecar_path=_sidecar(tmp_path),
    ))
    _patch_fake_selector(monkeypatch, fitter, results)

    summary = fitter.select_best_distribution(_positive_data(), criterion="simple", feature_index=0)

    assert summary.family == "gamma"


def test_family_prior_does_not_affect_explicit_non_simple_criterion(tmp_path, monkeypatch):
    results = [_fit_result("norm", 0.80), _fit_result("gamma", 0.60)]
    fitter = DistributionFitter(DistributionFitterConfig(
        compute_crps_uq_decomposition=False,
        diakrino_family_prior_lambda=0.03,
        diakrino_sidecar_path=_sidecar(tmp_path),
    ))
    _patch_fake_selector(monkeypatch, fitter, results)

    summary = fitter.select_best_distribution(_positive_data(), criterion="bic", feature_index=0)

    assert summary.family == "norm"

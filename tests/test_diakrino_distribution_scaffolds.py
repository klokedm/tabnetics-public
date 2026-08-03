from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabnetics.pipeline.pipeline import (
    DFFSConfig,
    DFFSReproducibleModel,
    DistributionFeatureSelectionPipeline,
    DistributionFitSummary,
    DistributionFitter,
    DistributionFitterConfig,
)


def _sidecar(tmp_path, rows: list[list[float]]) -> str:
    df = pd.DataFrame(
        {
            "feature_index": list(range(len(rows))),
            "chunk_id": [0 for _ in rows],
            "population_family_logits": rows,
            "prior_logit": [0.0 for _ in rows],
        }
    )
    path = tmp_path / "feature_logits.parquet"
    df.to_parquet(path)
    return str(path)


def _flat_logits() -> list[float]:
    return [0.0] * 36


def _discrete_logits(family_id: int = 31) -> list[float]:
    row = [-8.0] * 36
    row[int(family_id)] = 8.0
    return row


def _train_test() -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray([-3.0, -1.5, -0.5, 0.25, 0.75, 1.5, 2.25, 3.0], dtype=float).reshape(-1, 1)
    test = np.asarray([-2.5, -0.1, 1.0, 4.0], dtype=float).reshape(-1, 1)
    return train, test


class _IdentityTransformer:
    def transform(self, X):
        return np.asarray(X, dtype=float)


def _norm_summary(pipe: DistributionFeatureSelectionPipeline, feature_index: int, train_col: np.ndarray):
    audit = pipe.dist_fitter.audit_data(train_col)
    return DistributionFitSummary(
        feature_index=int(feature_index),
        family="norm",
        params=(0.0, 1.0),
        cvm_p=0.8,
        ks_p=0.8,
        simple_score=0.1,
        confidence_set=("norm",),
        rejected=False,
        audit=audit,
        selected_family_support="real",
        candidates_pre_filter=2,
        candidates_post_filter=2,
    )


def _distribution_plan_from_result(result: dict, test: np.ndarray) -> dict:
    summary = result["summary"]
    return {
        "schema_version": "1.0",
        "apply_cdf_transform": True,
        "n_input_features": int(test.shape[1]),
        "feature_plans": [
            {
                "feature_index": int(result["feat_idx"]),
                "family": None if summary.family is None else str(summary.family),
                "params": None if summary.params is None else [float(v) for v in tuple(summary.params)],
                "weight": float(result["weight"]),
                "train_mean": float(result["train_mean"]),
                "train_std": float(result["train_std"]),
                "fallback_meta": dict(result.get("fallback_meta") or {}),
                "applied": True,
            }
        ],
    }


def _replay_feature(result: dict, test: np.ndarray) -> np.ndarray:
    plan = _distribution_plan_from_result(result, test)
    model = object.__new__(DFFSReproducibleModel)
    model.distribution_plan = plan
    model._dist_fitter = DistributionFitter(DistributionFitterConfig())
    return DFFSReproducibleModel._apply_distribution_transforms(model, test, test.copy())


def _serialized_replay(result: dict, test: np.ndarray) -> np.ndarray:
    model = DFFSReproducibleModel(
        n_input_features=int(test.shape[1]),
        imputer=_IdentityTransformer(),
        batch_model=None,
        face_meta={},
        face_pca=None,
        face_lda=None,
        ratio_meta={},
        scaler_base=_IdentityTransformer(),
        distribution_plan=_distribution_plan_from_result(result, test),
        prefilter_indices=list(range(int(test.shape[1]))),
        folding_meta={},
        folding_transformer=None,
        folding_standardize_mean=None,
        folding_standardize_scale=None,
        selector=_IdentityTransformer(),
        stage2_ratio_meta={},
        classifier_model=_IdentityTransformer(),
    )
    restored = DFFSReproducibleModel.from_json_dict(
        model.to_json_dict(),
        trusted_legacy_pickle=True,
    )
    return restored.transform(test)


def test_diakrino_skip_fit_discrete_routes_to_replayable_rank_gaussian(tmp_path):
    path = _sidecar(tmp_path, [_discrete_logits()])
    train, test = _train_test()
    cfg = DFFSConfig(
        dist_config=DistributionFitterConfig(
            compute_dip=False,
            diakrino_skip_fit_discrete_enabled=True,
            diakrino_sidecar_path=path,
        )
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)

    result = pipe._fit_transform_one_feature(0, train, test, seed=11)

    assert result["apply_reason"] == "diakrino_skip_fit_discrete_rank_gaussian"
    assert result["summary"].fit_method == "diakrino_skip_fit_discrete_rank_transform"
    assert result["fallback_meta"]["diakrino_family_argmax_id"] == 31
    replayed = _replay_feature(result, test)
    np.testing.assert_allclose(replayed[:, 0], result["test_z"], atol=1e-12)


def test_diakrino_skip_fit_discrete_survives_serialized_replay_without_sidecar(tmp_path):
    path = _sidecar(tmp_path, [_discrete_logits()])
    train, test = _train_test()
    cfg = DFFSConfig(
        dist_config=DistributionFitterConfig(
            compute_dip=False,
            diakrino_skip_fit_discrete_enabled=True,
            diakrino_sidecar_path=path,
        )
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)

    result = pipe._fit_transform_one_feature(0, train, test, seed=11)
    Path(path).unlink()

    replayed = _serialized_replay(result, test)

    np.testing.assert_allclose(replayed[:, 0], result["test_z"], atol=1e-12)


def test_diakrino_cdf_trust_gate_routes_to_replayable_rank_gaussian(tmp_path, monkeypatch):
    path = _sidecar(tmp_path, [_flat_logits()])
    train, test = _train_test()
    cfg = DFFSConfig(
        diakrino_sidecar_path=path,
        diakrino_cdf_trust_gate_enabled=True,
        diakrino_cdf_trust_entropy_threshold=0.5,
        diakrino_cdf_trust_fallback="rank_gaussian",
        dist_config=DistributionFitterConfig(compute_dip=False),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)

    monkeypatch.setattr(
        pipe.dist_fitter,
        "select_best_distribution",
        lambda data, criterion, feature_index, audit: _norm_summary(pipe, feature_index, data),
    )

    result = pipe._fit_transform_one_feature(0, train, test, seed=11)

    assert result["apply_reason"] == "diakrino_cdf_trust_rank_gaussian"
    assert result["summary"].fit_method == "diakrino_cdf_trust_rank_transform"
    assert result["fallback_meta"]["diakrino_family_entropy"] > 0.99
    replayed = _replay_feature(result, test)
    np.testing.assert_allclose(replayed[:, 0], result["test_z"], atol=1e-12)


def test_diakrino_cdf_trust_gate_survives_serialized_replay_without_sidecar(tmp_path, monkeypatch):
    path = _sidecar(tmp_path, [_flat_logits()])
    train, test = _train_test()
    cfg = DFFSConfig(
        diakrino_sidecar_path=path,
        diakrino_cdf_trust_gate_enabled=True,
        diakrino_cdf_trust_entropy_threshold=0.5,
        diakrino_cdf_trust_fallback="rank_gaussian",
        dist_config=DistributionFitterConfig(compute_dip=False),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    monkeypatch.setattr(
        pipe.dist_fitter,
        "select_best_distribution",
        lambda data, criterion, feature_index, audit: _norm_summary(pipe, feature_index, data),
    )

    result = pipe._fit_transform_one_feature(0, train, test, seed=11)
    Path(path).unlink()

    replayed = _serialized_replay(result, test)

    np.testing.assert_allclose(replayed[:, 0], result["test_z"], atol=1e-12)


def test_diakrino_stability_surrogate_replaces_bootstrap_when_sidecar_present(tmp_path, monkeypatch):
    path = _sidecar(tmp_path, [_flat_logits()])
    train, test = _train_test()
    cfg = DFFSConfig(
        diakrino_sidecar_path=path,
        use_distribution_stability_weight=True,
        diakrino_stability_surrogate_enabled=True,
        dist_config=DistributionFitterConfig(compute_dip=False),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    monkeypatch.setattr(
        pipe.dist_fitter,
        "select_best_distribution",
        lambda data, criterion, feature_index, audit: _norm_summary(pipe, feature_index, data),
    )
    monkeypatch.setattr(
        pipe,
        "_family_stability_bootstrap",
        lambda *args, **kwargs: pytest.fail("bootstrap should not run when DIAKRINO entropy is available"),
    )

    result = pipe._fit_transform_one_feature(0, train, test, seed=11)

    assert result["apply_reason"] == "ok"
    assert result["stability_source"] == "diakrino_entropy_surrogate"
    assert result["stability_weight"] == pytest.approx(0.5, abs=1e-12)
    assert result["weight"] == pytest.approx(0.5, abs=1e-12)


def test_diakrino_stability_surrogate_falls_back_to_bootstrap_without_sidecar(monkeypatch):
    train, test = _train_test()
    cfg = DFFSConfig(
        use_distribution_stability_weight=True,
        diakrino_stability_surrogate_enabled=True,
        dist_config=DistributionFitterConfig(compute_dip=False),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    monkeypatch.setattr(
        pipe.dist_fitter,
        "select_best_distribution",
        lambda data, criterion, feature_index, audit: _norm_summary(pipe, feature_index, data),
    )
    monkeypatch.setattr(pipe, "_family_stability_bootstrap", lambda *args, **kwargs: 0.25)

    result = pipe._fit_transform_one_feature(0, train, test, seed=11)

    assert result["stability_source"] == "bootstrap"
    assert result["stability_weight"] == pytest.approx(0.625, abs=1e-12)

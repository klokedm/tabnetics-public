import sys
import random

import numpy as np
import scipy.stats as sps
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

import tabnetics.pipeline.pipeline as df_fs_pipeline
from tabnetics.pipeline.pipeline import (
    DataAuditReport,
    DFFSConfig,
    DistributionFitSummary,
    DistributionFeatureSelectionPipeline,
    DistributionFitter,
    DistributionFitterConfig,
    SupportProfile,
)


def _small_dataset(seed: int = 42):
    X, y = make_classification(
        n_samples=90,
        n_features=140,
        n_informative=20,
        n_redundant=12,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.1,
        flip_y=0.03,
        weights=[0.5, 0.3, 0.2],
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def test_safe_roc_auc_bundle_hard_voting_ensemble_has_curve_points():
    X, y = make_classification(
        n_samples=120,
        n_features=18,
        n_informative=10,
        n_redundant=2,
        n_classes=2,
        random_state=13,
    )
    model = VotingClassifier(
        estimators=[
            ("lr_a", LogisticRegression(max_iter=4000, solver="lbfgs")),
            ("lr_b", LogisticRegression(max_iter=4000, solver="lbfgs", C=0.25)),
        ],
        voting="hard",
        n_jobs=1,
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    pipe = DistributionFeatureSelectionPipeline(DFFSConfig())
    roc_meta = pipe._safe_roc_auc_bundle(
        model=model,
        X_eval=np.asarray(X, dtype=float),
        y_true=np.asarray(y),
        y_pred=np.asarray(y_pred),
    )

    assert str(roc_meta.get("roc_auc_source")) == "hard_vote_fraction"
    assert np.isfinite(float(roc_meta.get("roc_auc", float("nan"))))
    points = tuple(roc_meta.get("roc_curve_points", tuple()) or tuple())
    assert len(points) >= 2
    assert points[0] == (0.0, 0.0)
    assert points[-1] == (1.0, 1.0)

    by_method = pipe._collect_roc_curves_by_method(
        model=model,
        model_name="mnpo_ensemble_lr_a__lr_b",
        X_eval=np.asarray(X, dtype=float),
        y_true=np.asarray(y),
        y_pred=np.asarray(y_pred),
    )
    assert "mnpo_ensemble_lr_a__lr_b" in by_method
    assert "lr_a" in by_method
    assert "lr_b" in by_method
    root = dict(by_method["mnpo_ensemble_lr_a__lr_b"] or {})
    caps = dict(root.get("roc_metric_capabilities") or {})
    assert bool(caps.get("supports_hard_vote_fraction", False)) is True
    assert str(caps.get("selected_source", "")) == "hard_vote_fraction"


def test_safe_roc_auc_bundle_multiclass_includes_ova_curves():
    X, y = make_classification(
        n_samples=150,
        n_features=22,
        n_informative=12,
        n_redundant=3,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=29,
    )
    model = LogisticRegression(max_iter=5000, solver="lbfgs")
    model.fit(X, y)
    y_pred = model.predict(X)

    pipe = DistributionFeatureSelectionPipeline(DFFSConfig())
    roc_meta = pipe._safe_roc_auc_bundle(
        model=model,
        X_eval=np.asarray(X, dtype=float),
        y_true=np.asarray(y),
        y_pred=np.asarray(y_pred),
    )

    assert str(roc_meta.get("roc_curve_type", "")) == "ovr_micro"
    assert np.isfinite(float(roc_meta.get("roc_auc", float("nan"))))
    assert np.isfinite(float(roc_meta.get("roc_auc_macro_ovr", float("nan"))))
    assert np.isfinite(float(roc_meta.get("roc_auc_micro_ovr", float("nan"))))
    ova = dict(roc_meta.get("roc_curve_ova") or {})
    assert len(ova) >= 3
    for payload in ova.values():
        pts = list(payload.get("roc_curve_points") or [])
        assert len(pts) >= 2


def test_distribution_fitter_support_filtering_unit_interval_prefers_bounded_families():
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, size=200)

    fitter = DistributionFitter(
        DistributionFitterConfig(
            robust_mode=True,
            use_adaptive_strategy=True,
            use_lrt=False,
            use_cv=False,
            use_support_filtering=True,
        )
    )
    audit = fitter.audit_data(x)
    candidates = fitter.generate_candidates(audit)

    assert audit.support.inferred_support == "unit_interval"
    assert "beta" in candidates
    assert "uniform" in candidates
    assert "gamma" not in candidates


def test_distribution_fitter_support_filtering_real_data_with_negative_mass_excludes_positive_only_families():
    rng = np.random.default_rng(123)
    x = np.concatenate(
        [
            rng.normal(loc=-1.5, scale=0.8, size=120),
            rng.normal(loc=1.5, scale=0.8, size=80),
        ]
    ).astype(float)

    fitter = DistributionFitter(DistributionFitterConfig(use_support_filtering=True))
    audit = fitter.audit_data(x)
    candidates = fitter.generate_candidates(audit)

    positive_only = {
        "expon",
        "gamma",
        "lognorm",
        "weibull_min",
        "pareto",
        "invweibull",
        "invgauss",
        "geninvgauss",
        "invgamma",
        "fisk",
        "genpareto",
        "gengamma",
    }
    assert audit.support.inferred_support == "real"
    assert float(audit.frac_negative) >= 0.05
    assert set(candidates.keys()).isdisjoint(positive_only)


def test_distribution_fitter_extended_family_set_includes_opt_in_families():
    rng = np.random.default_rng(42)
    x = rng.gamma(shape=2.0, scale=1.0, size=200).astype(float)

    fitter = DistributionFitter(
        DistributionFitterConfig(
            family_set="extended",
            robust_mode=True,
            use_adaptive_strategy=True,
            use_lrt=False,
            use_cv=False,
            use_support_filtering=True,
        )
    )
    audit = fitter.audit_data(x)
    candidates = fitter.generate_candidates(audit)

    assert audit.support.inferred_support == "positive"
    assert "genpareto" in candidates
    assert "invgamma" in candidates
    assert "fisk" in candidates


def test_distribution_fitter_flex_family_set_includes_opt_in_flexible_families():
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=200).astype(float)

    fitter = DistributionFitter(
        DistributionFitterConfig(
            family_set="flex",
            robust_mode=True,
            use_adaptive_strategy=True,
            use_lrt=False,
            use_cv=False,
            use_support_filtering=True,
        )
    )
    audit = fitter.audit_data(x)
    candidates = fitter.generate_candidates(audit)

    assert audit.support.inferred_support == "real"
    assert "tukeylambda" in candidates
    assert "genhyperbolic" in candidates


def test_distribution_fitter_can_compute_ad_and_qq_pp_diagnostics():
    rng = np.random.default_rng(7)
    x = rng.normal(loc=0.0, scale=1.0, size=120).astype(float)

    fitter = DistributionFitter(
        DistributionFitterConfig(
            robust_mode=True,
            use_adaptive_strategy=True,
            use_lrt=False,
            use_cv=False,
            use_support_filtering=True,
            compute_ad=True,
            ad_bootstrap_samples=10,
            compute_qq_pp=True,
            random_state=7,
        )
    )
    summary = fitter.select_best_distribution(x, criterion="simple", feature_index=0)

    assert summary.family is not None
    assert summary.params is not None
    assert summary.ad_stat is not None
    assert summary.ad_p is not None
    assert summary.qq_r2 is not None
    assert summary.pp_mae is not None


def test_distribution_fitter_can_compute_crps_and_supports_crps_criterion():
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=160).astype(float)

    fitter = DistributionFitter(
        DistributionFitterConfig(
            robust_mode=True,
            use_adaptive_strategy=True,
            use_lrt=False,
            use_cv=False,
            use_support_filtering=True,
            compute_crps=True,
            crps_mc_samples=64,
            crps_data_subsample=128,
            random_state=42,
        )
    )
    summary = fitter.select_best_distribution(x, criterion="crps", feature_index=0)

    assert summary.family is not None
    assert summary.params is not None
    assert summary.crps is not None
    assert np.isfinite(float(summary.crps))


def test_crps_uq_decomposition_is_zero_for_identical_members():
    total, alea, epi = df_fs_pipeline._crps_uq_decompose_gaussian_ensemble(
        means=[0.0, 0.0],
        stds=[1.0, 1.0],
        weights=[0.5, 0.5],
    )

    assert np.isfinite(total)
    assert np.isfinite(alea)
    assert np.isfinite(epi)
    assert abs(total - alea) < 1e-10
    assert abs(epi) < 1e-10


def test_distribution_fitter_can_compute_crps_uq_decomposition_when_opted_in():
    rng = np.random.default_rng(23)
    x = rng.normal(loc=0.25, scale=1.5, size=180).astype(float)

    fitter = DistributionFitter(
        DistributionFitterConfig(
            robust_mode=True,
            use_adaptive_strategy=True,
            use_lrt=False,
            use_cv=False,
            use_support_filtering=False,
            confidence_margin=999.0,  # include a large confidence set for UQ decomposition
            compute_crps_uq_decomposition=True,
            random_state=23,
        )
    )
    summary = fitter.select_best_distribution(x, criterion="simple", feature_index=0)

    assert summary.family is not None
    assert summary.params is not None
    assert summary.crps_uq_total is not None
    assert summary.crps_uq_aleatoric is not None
    assert summary.crps_uq_epistemic is not None
    assert float(summary.crps_uq_total) >= float(summary.crps_uq_aleatoric) - 1e-10
    assert float(summary.crps_uq_epistemic) >= -1e-10


def test_mnpo_oracle_can_include_crps_when_opted_in():
    rng = np.random.default_rng(17)
    x = rng.gamma(shape=2.0, scale=1.0, size=220).astype(float)

    fitter = DistributionFitter(
        DistributionFitterConfig(
            robust_mode=True,
            use_adaptive_strategy=True,
            use_lrt=False,
            use_cv=False,
            use_support_filtering=True,
            mnpo_include_crps=True,
            crps_mc_samples=32,
            crps_data_subsample=128,
            random_state=17,
        )
    )
    summary = fitter.select_best_distribution(x, criterion="mnpo_oracle", feature_index=0)

    assert summary.family is not None
    assert summary.params is not None
    assert summary.crps is not None
    assert np.isfinite(float(summary.crps))


def test_mnpo_oracle_can_include_preq_predictive_oracle_when_opted_in():
    rng = np.random.default_rng(11)
    x = rng.normal(loc=0.0, scale=1.0, size=220).astype(float)

    from tabnetics.distribution.selector import UnifiedDistributionSelectorV6

    selector = UnifiedDistributionSelectorV6(
        distributions={"norm": sps.norm, "laplace": sps.laplace},
        robust_mode=False,
        use_adaptive_strategy=False,
        use_lrt=False,
        use_cv=False,
        n_jobs=1,
        mnpo_include_preq=True,
        random_state=11,
    )
    best_name, best_result, all_results = selector.select_best_distribution(x, criterion="mnpo_oracle", verbose=False)

    assert best_name is not None
    assert best_result is not None
    assert getattr(best_result, "preq_loglik_mean", None) is not None
    assert np.isfinite(float(getattr(best_result, "preq_loglik_mean")))
    assert any(
        getattr(r, "preq_loglik_mean", None) is not None and np.isfinite(float(getattr(r, "preq_loglik_mean")))
        for r in all_results
        if getattr(r, "success", False)
    )


def test_unified_selector_interval_likelihood_uses_randomized_pit_uniformity_tests():
    import zlib
    from scipy.stats import cramervonmises, kstest

    rng = np.random.default_rng(123)
    x = np.round(rng.normal(loc=0.0, scale=1.0, size=240)).astype(float)

    from tabnetics.distribution.selector import UnifiedDistributionSelectorV6

    selector = UnifiedDistributionSelectorV6(
        distributions={"norm": sps.norm},
        robust_mode=False,
        use_adaptive_strategy=False,
        use_lrt=False,
        use_cv=False,
        n_jobs=1,
        interval_likelihood=True,
        interval_delta=1.0,
        random_state=123,
    )

    best_name, best_result, _ = selector.select_best_distribution(x, criterion="simple", verbose=False)

    assert best_name == "norm"
    assert best_result is not None
    assert best_result.params is not None

    params = tuple(best_result.params)
    delta = 1.0
    half = 0.5 * delta
    lo = x - half
    hi = x + half

    u_lo = np.asarray(sps.norm.cdf(lo, *params), dtype=float).ravel()
    u_hi = np.asarray(sps.norm.cdf(hi, *params), dtype=float).ravel()
    u_lo = np.clip(np.nan_to_num(u_lo, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    u_hi = np.clip(np.nan_to_num(u_hi, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    interval_prob = np.clip(u_hi - u_lo, 0.0, 1.0)

    base_seed = 123
    fam_seed = int(zlib.crc32("norm".encode("utf-8")) & 0xFFFFFFFF)
    pit_rng = np.random.default_rng(int(base_seed) ^ fam_seed ^ 0x9E3779B9)
    u = u_lo + pit_rng.random(size=interval_prob.size) * interval_prob
    u = np.clip(u, 1e-12, 1.0 - 1e-12)

    ks_stat, ks_p = kstest(u, "uniform")
    cvm_res = cramervonmises(u, "uniform")
    loglik = float(np.sum(np.log(np.maximum(interval_prob, 1e-300))))

    assert np.isclose(float(best_result.ks_stat), float(ks_stat))
    assert np.isclose(float(best_result.ks_p), float(ks_p))
    assert np.isclose(float(best_result.cvm_stat), float(cvm_res.statistic))
    assert np.isclose(float(best_result.cvm_p), float(cvm_res.pvalue))
    assert np.isclose(float(best_result.loglik), float(loglik))


def test_unified_selector_mps_estimator_marks_fit_method():
    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.2, scale=1.3, size=160).astype(float)

    from tabnetics.distribution.selector import UnifiedDistributionSelectorV6

    selector = UnifiedDistributionSelectorV6(
        distributions={"norm": sps.norm},
        robust_mode=False,
        use_adaptive_strategy=False,
        use_lrt=False,
        use_cv=False,
        n_jobs=1,
        fit_estimator="mps",
        mps_maxiter=80,
        mps_tol=1e-6,
        random_state=123,
    )
    best_name, best_result, _ = selector.select_best_distribution(x, criterion="simple", verbose=False)

    assert best_name == "norm"
    assert best_result is not None
    assert bool(best_result.success) is True
    assert str(best_result.fit_method) == "MPS"


def test_unified_selector_lmoment_prescreen_limits_candidate_count():
    rng = np.random.default_rng(1234)
    x = rng.gamma(shape=2.5, scale=1.0, size=180).astype(float)

    from tabnetics.distribution.selector import UnifiedDistributionSelectorV6

    dist_names = [
        "norm",
        "t",
        "laplace",
        "gamma",
        "lognorm",
        "weibull_min",
        "pareto",
        "tukeylambda",
        "gennorm",
    ]
    dists = {name: getattr(sps, name) for name in dist_names}

    selector = UnifiedDistributionSelectorV6(
        distributions=dists,
        robust_mode=False,
        use_adaptive_strategy=True,
        use_lrt=False,
        use_cv=False,
        n_jobs=1,
        use_lmoment_prescreen=True,
        lmoment_prescreen_max_candidates=6,
        random_state=1234,
    )
    best_name, best_result, all_results = selector.select_best_distribution(x, criterion="simple", verbose=False)

    assert best_name is not None
    assert best_result is not None
    assert bool(best_result.success) is True
    assert len(all_results) <= 6


def test_unified_selector_lmoment_prescreen_demotes_positive_only_families_on_real_negative_data():
    rng = np.random.default_rng(2026)
    x = np.concatenate(
        [
            rng.normal(loc=-2.0, scale=1.0, size=120),
            rng.normal(loc=0.5, scale=0.7, size=80),
        ]
    ).astype(float)

    from tabnetics.distribution.selector import DistributionFeatures, UnifiedDistributionSelectorV6

    names = ["norm", "t", "laplace", "johnsonsu", "gamma", "expon", "lognorm", "weibull_min", "pareto"]
    selector = UnifiedDistributionSelectorV6(
        distributions={name: getattr(sps, name) for name in names},
        robust_mode=False,
        use_adaptive_strategy=True,
        use_lrt=False,
        use_cv=False,
        n_jobs=1,
        use_lmoment_prescreen=True,
        lmoment_prescreen_max_candidates=5,
        random_state=2026,
    )
    features = DistributionFeatures.from_data(x)
    kept = selector._lmoment_prescreen_distribution_names(names, features)

    assert float(features.frac_negative) >= 0.05
    assert len(kept) <= 5
    assert set(kept).isdisjoint({"gamma", "expon", "lognorm", "weibull_min", "pareto"})


def test_unified_selector_positive_only_bonus_is_zero_when_data_has_negative_mass():
    rng = np.random.default_rng(17)
    x = rng.normal(loc=-2.0, scale=1.0, size=180).astype(float)

    from tabnetics.distribution.selector import DistributionFeatures, UnifiedDistributionSelectorV6

    selector = UnifiedDistributionSelectorV6(
        distributions={"expon": sps.expon, "gamma": sps.gamma},
        robust_mode=False,
        use_adaptive_strategy=True,
        use_lrt=False,
        use_cv=False,
        n_jobs=1,
        random_state=17,
    )
    features = DistributionFeatures.from_data(x)
    assert bool(features.is_positive) is False
    assert selector._calculate_feature_bonus(features, "expon") == 0.0
    assert selector._calculate_feature_bonus(features, "gamma") == 0.0


def test_unified_selector_exponential_bonus_regression_guard_on_true_positive_support():
    rng = np.random.default_rng(18)
    x = rng.exponential(scale=1.0, size=240).astype(float)

    from tabnetics.distribution.selector import DistributionFeatures, UnifiedDistributionSelectorV6

    selector = UnifiedDistributionSelectorV6(
        distributions={"expon": sps.expon},
        robust_mode=False,
        use_adaptive_strategy=True,
        use_lrt=False,
        use_cv=False,
        n_jobs=1,
        random_state=18,
    )
    features = DistributionFeatures.from_data(x)
    bonus = selector._calculate_feature_bonus(features, "expon")

    assert bool(features.is_positive) is True
    assert float(features.frac_negative) == 0.0
    assert float(bonus) > 0.0


def test_unified_selector_prepare_data_logs_warning_for_large_positive_shift(caplog):
    rng = np.random.default_rng(19)
    x = rng.normal(loc=-10.0, scale=1.0, size=220).astype(float)

    from tabnetics.distribution.selector import DistributionFeatures, UnifiedDistributionSelectorV6

    selector = UnifiedDistributionSelectorV6(
        distributions={"expon": sps.expon},
        robust_mode=False,
        use_adaptive_strategy=True,
        use_lrt=False,
        use_cv=False,
        n_jobs=1,
        random_state=19,
    )
    features = DistributionFeatures.from_data(x)
    with caplog.at_level("WARNING"):
        selector._prepare_data(x, "expon", features)
    assert any("Large positive-support shift applied" in rec.message for rec in caplog.records)


def test_distribution_fitter_can_compute_dip_and_multimodality_flags():
    rng = np.random.default_rng(123)
    x = np.concatenate(
        [
            rng.normal(loc=-4.0, scale=1.0, size=160),
            rng.normal(loc=4.0, scale=1.0, size=160),
        ]
    ).astype(float)

    fitter = DistributionFitter(
        DistributionFitterConfig(
            robust_mode=True,
            use_adaptive_strategy=True,
            use_lrt=False,
            use_cv=False,
            use_support_filtering=True,
            compute_dip=True,
            dip_hist_bins=60,
        )
    )
    audit = fitter.audit_data(x)

    assert audit.dip_stat is not None
    assert audit.mode_count is not None
    assert audit.is_multimodal is not None
    assert int(audit.mode_count) >= 2
    assert bool(audit.is_multimodal) is True


def test_distribution_fitter_populates_rejection_reason_for_basic_gate_cases():
    fitter = DistributionFitter(
        DistributionFitterConfig(
            use_support_filtering=True,
            use_lrt=False,
            use_cv=False,
        )
    )

    summary_small = fitter.select_best_distribution(np.asarray([0.0, 1.0, 2.0], dtype=float), feature_index=0)
    assert summary_small.rejection_reason == "insufficient_clean"

    summary_const = fitter.select_best_distribution(np.ones(64, dtype=float), feature_index=1)
    assert summary_const.rejection_reason == "near_constant"


def test_distribution_fitter_zero_inflation_flag_triggers_on_many_zeros():
    rng = np.random.default_rng(7)
    x = np.concatenate([np.zeros(40), rng.gamma(shape=2.0, scale=1.0, size=120)]).astype(float)

    fitter = DistributionFitter(DistributionFitterConfig(use_support_filtering=True))
    audit = fitter.audit_data(x)

    assert float(audit.support.frac_zero) > 0.20
    assert bool(audit.zero_inflated) is True


def test_pipeline_supports_mnpo_oracle_dist_criterion_and_emits_weights():
    X, y = _small_dataset(123)

    cfg = DFFSConfig(
        random_seed=77,
        fs_fraction=0.4,
        n_final_features=12,
        max_dist_features=12,
        prefilter_top_k=90,
        dist_criterion="mnpo_oracle",
        multimodal_fallback="none",
        dist_config=DistributionFitterConfig(
            robust_mode=True,
            use_adaptive_strategy=True,
            use_lrt=False,
            use_cv=False,
            compute_budget="standard",
            use_support_filtering=True,
            compute_dip=False,
        ),
    )

    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=77)

    weights = [s.mnpo_weight for s in res.distribution_summaries if s.family is not None]
    assert any(w is not None for w in weights)


def test_pipeline_reproducible_split_and_metrics():
    X, y = _small_dataset(123)

    cfg = DFFSConfig(
        random_seed=77,
        fs_fraction=0.4,
        n_final_features=14,
        max_dist_features=24,
        prefilter_top_k=90,
    )

    pipe_a = DistributionFeatureSelectionPipeline(cfg)
    res_a = pipe_a.run(X, y, dataset_name="tiny", seed=77)

    pipe_b = DistributionFeatureSelectionPipeline(cfg)
    res_b = pipe_b.run(X, y, dataset_name="tiny", seed=77)

    assert res_a.split_indices_train == res_b.split_indices_train
    assert res_a.split_indices_test == res_b.split_indices_test
    assert abs(res_a.balanced_accuracy - res_b.balanced_accuracy) < 1e-12
    assert abs(res_a.macro_f1 - res_b.macro_f1) < 1e-12


def test_pipeline_emits_reproducible_model_bundle_and_structured_diagnostics():
    X, y = _small_dataset(321)

    cfg = DFFSConfig(
        random_seed=77,
        fs_fraction=0.4,
        n_final_features=12,
        max_dist_features=16,
        prefilter_top_k=80,
        multimodal_fallback="none",
        dist_config=DistributionFitterConfig(compute_dip=False),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(
        X,
        y,
        dataset_name="tiny_bundle",
        seed=77,
        capture_artifacts=True,
        capture_diagnostics=True,
    )

    bundle = dict(res.model_bundle or {})
    assert bundle.get("artifact_type") == "df_fs_model_bundle"
    assert int(bundle.get("n_input_features", 0)) == int(X.shape[1])
    assert "serialization" in bundle

    deployed = df_fs_pipeline.DFFSReproducibleModel.from_json_dict(bundle)
    test_idx = np.asarray(res.split_indices_test, dtype=int)
    X_test = np.asarray(X, dtype=float)[test_idx]
    y_test = np.asarray(y).ravel()[test_idx]
    y_pred = np.asarray(deployed.predict(X_test), dtype=np.asarray(y_test).dtype).ravel()
    assert y_pred.shape[0] == y_test.shape[0]
    bal = pipe._safe_balanced_accuracy(y_test, y_pred)
    assert np.isfinite(float(bal))
    assert abs(float(bal) - float(res.balanced_accuracy)) < 1e-12

    diag = dict(res.run_diagnostics or {})
    assert diag.get("artifact_type") == "df_fs_run_diagnostics"
    stages = dict(diag.get("pipeline_stages") or {})
    assert "distribution_stage" in stages
    assert "feature_selection" in stages
    assert "classifier_selection" in stages


def test_pipeline_supports_after_fs_distribution_stage_ordering():
    X, y = _small_dataset(322)

    cfg = DFFSConfig(
        random_seed=79,
        fs_fraction=0.4,
        n_final_features=12,
        max_dist_features=12,
        prefilter_top_k=80,
        df_stage_position="after_fs",
        folding_method="none",
        multimodal_fallback="none",
        dist_config=DistributionFitterConfig(compute_dip=False),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(
        X,
        y,
        dataset_name="tiny_after_fs",
        seed=79,
        capture_artifacts=True,
        capture_diagnostics=True,
    )

    assert str(res.config_snapshot.get("df_stage_position", "")) == "after_fs"
    assert str(res.config_snapshot.get("df_stage_position_effective", "")) == "after_fs"
    assert str(res.config_snapshot.get("df_stage_source_space", "")) == "prefilter_raw"
    assert int(res.n_dist_features_fitted) >= 0

    diag = dict(res.run_diagnostics or {})
    dist_stage = dict((dict(diag.get("pipeline_stages") or {})).get("distribution_stage") or {})
    dist_summary = dict(dist_stage.get("summary") or {})
    assert str(dist_summary.get("df_stage_position", "")) == "after_fs"
    assert str(dist_summary.get("df_stage_source_space", "")) == "prefilter_raw"

    bundle = dict(res.model_bundle or {})
    deployed = df_fs_pipeline.DFFSReproducibleModel.from_json_dict(bundle)
    test_idx = np.asarray(res.split_indices_test, dtype=int)
    X_test = np.asarray(X, dtype=float)[test_idx]
    y_test = np.asarray(y).ravel()[test_idx]
    y_pred = np.asarray(deployed.predict(X_test), dtype=np.asarray(y_test).dtype).ravel()
    bal = pipe._safe_balanced_accuracy(y_test, y_pred)
    assert np.isfinite(float(bal))
    assert abs(float(bal) - float(res.balanced_accuracy)) < 1e-12


def test_pipeline_classifier_mnpo_diagnostics_include_oracle_matrices_and_payoff():
    X, y = _small_dataset(987)

    cls_cfg = df_fs_pipeline.ClassificationConfig(
        selection_mode="mnpo_hybrid",
        backend="sklearn",
        model_candidates=("lr", "svm_linear", "nb"),
        oracle_k=2,
        oracle_enable_bbc=False,
        oracle_enable_hoeffding_racing=False,
        oracle_use_per_family_flaml=False,
        min_n_for_automl=2,
        min_n_per_class_for_cv=2,
        min_n_per_class_for_automl=2,
        max_p_over_n_for_automl=10000,
    )
    cfg = DFFSConfig(
        random_seed=91,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        classification=cls_cfg,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(
        X,
        y,
        dataset_name="tiny_mnpo_diag",
        seed=91,
        capture_diagnostics=True,
    )

    diag = dict(res.run_diagnostics or {})
    stages = dict(diag.get("pipeline_stages") or {})
    classifier_stage = dict(stages.get("classifier_selection") or {})
    mnpo_diag = dict(classifier_stage.get("mnpo_diagnostics") or {})

    assert str(classifier_stage.get("classification_selection_mode", "")) == "mnpo_hybrid"
    assert str(classifier_stage.get("mnpo_selected_classifier", "")) != ""
    assert list(mnpo_diag.get("candidate_order") or []) != []
    assert str(mnpo_diag.get("selected_classifier", "")) == str(
        classifier_stage.get("mnpo_selected_classifier", "")
    )
    assert int(mnpo_diag.get("effective_oracle_k", 0) or 0) == 2

    payoff = np.asarray(mnpo_diag.get("payoff") or [], dtype=float)
    candidate_order = list(mnpo_diag.get("candidate_order") or [])
    assert payoff.shape == (len(candidate_order), len(candidate_order))

    oracle_matrices = dict(mnpo_diag.get("oracle_matrices") or {})
    assert "performance" in oracle_matrices
    for matrix in oracle_matrices.values():
        mat = np.asarray(matrix, dtype=float)
        assert mat.shape == payoff.shape

    solver_meta = dict(mnpo_diag.get("solver_meta") or {})
    assert str(solver_meta.get("reference_oracle", "")) == "performance"
    assert int(solver_meta.get("top_k", 0) or 0) == 2


def test_pipeline_run_pre_split_matches_run_for_same_split_indices():
    X, y = _small_dataset(456)

    cfg = DFFSConfig(
        random_seed=19,
        fs_fraction=0.4,
        n_final_features=14,
        max_dist_features=24,
        prefilter_top_k=90,
    )

    pipe_split = DistributionFeatureSelectionPipeline(cfg)
    idx_all = np.arange(X.shape[0], dtype=int)
    train_idx, test_idx = pipe_split._split_indices(idx_all, y, seed=cfg.random_seed)

    res_pre = pipe_split.run_pre_split(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        dataset_name="tiny",
        seed=cfg.random_seed,
        split_indices_train=train_idx,
        split_indices_test=test_idx,
    )

    pipe_run = DistributionFeatureSelectionPipeline(cfg)
    res_run = pipe_run.run(X, y, dataset_name="tiny", seed=cfg.random_seed)

    assert res_pre.split_indices_train == res_run.split_indices_train
    assert res_pre.split_indices_test == res_run.split_indices_test
    assert abs(res_pre.balanced_accuracy - res_run.balanced_accuracy) < 1e-12
    assert abs(res_pre.macro_f1 - res_run.macro_f1) < 1e-12


def test_pipeline_combat_batch_correction_snapshot_fields_present():
    X, y = _small_dataset(333)
    batch_labels = np.array(["center_a"] * (X.shape[0] // 2) + ["center_b"] * (X.shape[0] - X.shape[0] // 2), dtype=object)
    # Inject a simple center-specific location shift so correction has signal.
    X = np.asarray(X, dtype=float)
    X[batch_labels == "center_b", :12] += 1.8

    cfg = DFFSConfig(
        random_seed=33,
        fs_fraction=0.4,
        n_final_features=12,
        max_dist_features=20,
        prefilter_top_k=80,
        batch_correction="combat",
        batch_correction_combat_prior_strength=8.0,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=33, batch_labels=batch_labels)

    snap = dict(res.config_snapshot or {})
    assert snap.get("batch_correction_mode_requested") == "combat"
    assert snap.get("batch_correction_mode_applied") == "combat"
    assert bool(snap.get("batch_correction_applied", False)) is True
    assert snap.get("batch_correction_fit_reason") == "ok"
    assert snap.get("batch_correction_apply_reason") == "ok"
    assert int(snap.get("batch_correction_n_batches", 0)) >= 2


def test_pipeline_run_pre_split_rejects_mismatched_batch_labels():
    X, y = _small_dataset(444)
    cfg = DFFSConfig(random_seed=44, batch_correction="combat")
    pipe = DistributionFeatureSelectionPipeline(cfg)
    idx_all = np.arange(X.shape[0], dtype=int)
    train_idx, test_idx = pipe._split_indices(idx_all, y, seed=44)

    with pytest.raises(ValueError, match="batch_labels_train"):
        pipe.run_pre_split(
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_test=X[test_idx],
            y_test=y[test_idx],
            dataset_name="tiny",
            seed=44,
            split_indices_train=train_idx,
            split_indices_test=test_idx,
            batch_labels_train=np.array(["a"] * (len(train_idx) - 1), dtype=object),
            batch_labels_test=np.array(["a"] * len(test_idx), dtype=object),
        )


def test_pipeline_fs_fraction_applied_to_train_subset_size():
    X, y = _small_dataset(999)

    cfg = DFFSConfig(
        random_seed=11,
        fs_fraction=0.35,
        n_final_features=12,
        max_dist_features=20,
        prefilter_top_k=80,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=11)

    expected_fs = int(round(cfg.fs_fraction * res.n_train))
    assert abs(res.n_fs_subset - expected_fs) <= 1
    assert res.selected_features_count > 0
    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)


def test_pipeline_respects_configured_test_size():
    X, y = _small_dataset(314)

    cfg = DFFSConfig(
        random_seed=5,
        test_size=0.30,
        fs_fraction=0.4,
        n_final_features=12,
        max_dist_features=20,
        prefilter_top_k=80,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=5)

    assert res.n_test == 27
    assert res.n_train == 63


def test_pipeline_max_train_samples_enforces_absolute_train_cap_on_large_dataset():
    # This targets the Artificial-HDLSS protocol: force a tiny train split even
    # when the dataset is large. Previously, the implementation clamped
    # test_size to <=0.95 which broke absolute caps for N >> 1000.
    X, y = make_classification(
        n_samples=2000,
        n_features=40,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.5, 0.5],
        class_sep=1.0,
        flip_y=0.01,
        random_state=123,
    )
    X = X.astype(float)
    y = y.astype(int)

    cfg = DFFSConfig(
        random_seed=7,
        test_size=0.20,
        max_train_samples=50,
        fs_fraction=0.4,
        n_final_features=12,
        max_dist_features=20,
        prefilter_top_k=80,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    idx_all = np.arange(X.shape[0], dtype=int)
    train_idx, test_idx = pipe._split_indices(idx_all, y, seed=cfg.random_seed)

    assert train_idx.size == 50
    assert test_idx.size == 1950


def test_pipeline_max_train_samples_preserves_minimum_80_20_holdout():
    X, y = make_classification(
        n_samples=40,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.5, 0.5],
        random_state=99,
    )
    X = X.astype(float)
    y = y.astype(int)

    cfg = DFFSConfig(
        random_seed=13,
        test_size=0.20,
        max_train_samples=36,  # Would violate 80/20 if applied blindly.
        fs_fraction=0.4,
        n_final_features=8,
        max_dist_features=10,
        prefilter_top_k=20,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    idx_all = np.arange(X.shape[0], dtype=int)
    train_idx, test_idx = pipe._split_indices(idx_all, y, seed=cfg.random_seed)

    assert train_idx.size == 32
    assert test_idx.size == 8


def test_pipeline_max_train_samples_overrides_test_size_and_is_recorded():
    X, y = _small_dataset(2718)

    cfg = DFFSConfig(
        random_seed=21,
        test_size=0.20,
        max_train_samples=12,
        fs_fraction=0.4,
        n_final_features=12,
        max_dist_features=20,
        prefilter_top_k=80,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=21)

    assert res.n_train == 12
    assert res.n_test == 78
    assert res.config_snapshot["max_train_samples"] == 12


def test_cdf_reliability_gate_can_block_unreliable_transforms():
    X, y = _small_dataset(812)

    cfg = DFFSConfig(
        random_seed=9,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=15,
        cdf_reliability_gate=True,
        cdf_min_gof_p=1.1,  # impossible threshold forces skip.
        prefilter_top_k=70,
        multimodal_fallback="none",
        dist_config=DistributionFitterConfig(compute_dip=False),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=9)

    assert res.n_dist_features_fitted > 0
    assert res.n_dist_features_transformed == 0
    assert res.n_dist_skipped_unreliable >= 0


def test_df_fastpath_flags_are_noop_after_cleanup():
    X, y = _small_dataset(2026)

    X_mod = X.copy()
    x0 = np.asarray(X_mod[:, 0], dtype=float)
    X_mod[:, 0] = (x0 > np.median(x0)).astype(float) * 1e6  # high-variance binary feature

    cfg = DFFSConfig(
        random_seed=12,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=5,
        prefilter_top_k=70,
        df_fastpath_enabled=True,
        df_fastpath_trigger="low_unique",
        df_fastpath_unique_ratio_threshold=0.10,
        df_fastpath_n_unique_threshold=3,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X_mod, y, dataset_name="tiny", seed=12)

    families = [s.family for s in res.distribution_summaries if s.family is not None]
    assert "fastpath_rank_gauss" not in families
    assert res.n_dist_features_transformed > 0
    assert np.isfinite(res.balanced_accuracy)


def test_df_fastpath_flags_emit_deprecation_warnings():
    df_fs_pipeline._DEPRECATED_TOGGLE_WARNED.clear()
    cfg = DFFSConfig(
        df_fastpath_enabled=True,
        df_fastpath_trigger="low_unique",
        df_fastpath_small_n_threshold=128,
        df_fastpath_unique_ratio_threshold=0.10,
        df_fastpath_n_unique_threshold=3,
    )
    with pytest.warns(DeprecationWarning):
        _ = DistributionFeatureSelectionPipeline(cfg)


def test_pipeline_can_build_selector_from_feature_selector_config():
    from tabnetics.feature_selection.config import FeatureSelectorConfig, MNPOConfig

    fs_cfg = FeatureSelectorConfig(
        random_state=999,
        selection_strategy="mnpo_portfolio",
        enabled_methods={"linear_svm", "mutual_information"},
        mnpo=MNPOConfig(portfolio_size=4),
    )
    cfg = DFFSConfig(
        fs_config=fs_cfg,
        fs_portfolio_size=7,  # run-level override should win
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    selector = pipe._build_feature_selector(seed=17, enabled_methods=("linear_svm", "mutual_information"))

    assert int(selector.random_state) == 17
    assert int(selector.portfolio_size) == 7
    assert set(selector.enabled_methods) == {"linear_svm", "mutual_information"}


def test_pipeline_wires_val7_oracle_and_adaptive_sizing_flags_to_selector():
    cfg = DFFSConfig(
        fs_portfolio_size=6,
        fs_adaptive_portfolio_sizing_enabled=True,
        fs_adaptive_sizing_variance_penalty=True,
        fs_adaptive_sizing_variance_penalty_strength=0.75,
        fs_oracle_weighting_mode="shapley",
        fs_shapley_n_coalitions_max=1024,
        fs_shapley_bayesian_shrinkage=True,
        fs_shapley_bayesian_prior_strength=9.0,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    selector = pipe._build_feature_selector(seed=31, enabled_methods=("linear_svm", "mutual_information"))

    assert bool(selector.adaptive_portfolio_sizing_enabled) is True
    assert bool(selector.adaptive_sizing_variance_penalty) is True
    assert float(selector.adaptive_sizing_variance_penalty_strength) == pytest.approx(0.75)
    assert str(selector.oracle_weighting_mode) == "shapley"
    assert int(selector.shapley_n_coalitions_max) == 1024
    assert bool(selector.shapley_bayesian_shrinkage) is True
    assert float(selector.shapley_bayesian_prior_strength) == pytest.approx(9.0)


def test_pipeline_supports_cluster_stability_and_mi_diversity_toggles():
    X, y = _small_dataset(925)

    cfg = DFFSConfig(
        random_seed=19,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("cluster_stability", "linear_svm", "mutual_information", "anova_f"),
        use_diversity_oracle=True,
        fs_diversity_oracle_mode="mi_redundancy",
        fs_performance_use_adaptive_imbalance=True,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=19)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert res.config_snapshot["fs_diversity_oracle_mode"] == "mi_redundancy"
    assert np.isclose(res.config_snapshot["fs_diversity_complementarity_weight"], 0.35)
    assert bool(res.config_snapshot["fs_performance_use_adaptive_imbalance"]) is True


def test_pipeline_supports_expanded_model_harness_toggles_snapshot():
    X, y = _small_dataset(948)

    cfg = DFFSConfig(
        random_seed=33,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        include_elastic_net_model=True,
        include_rf_model=True,
        include_knn_model=True,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=33)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert bool(res.config_snapshot["include_elastic_net_model"]) is True
    assert bool(res.config_snapshot["include_rf_model"]) is True
    assert bool(res.config_snapshot["include_knn_model"]) is True
    assert res.model_name in {"lr", "svm_rbf", "elastic_net_lr", "rf", "knn", "nb", "svm_linear", "dlda"}


def test_pipeline_supports_op1_timeout_and_iter_controls_snapshot():
    X, y = _small_dataset(952)

    cfg = DFFSConfig(
        random_seed=37,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
        fs_method_timeout_seconds=12.5,
        fs_linear_svm_max_iter=12000,
        model_cv_lr_max_iter=11000,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=37)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert np.isclose(float(res.config_snapshot["fs_method_timeout_seconds"]), 12.5)
    assert int(res.config_snapshot["fs_linear_svm_max_iter"]) == 12000
    assert int(res.config_snapshot["model_cv_lr_max_iter"]) == 11000


def test_pipeline_model_cv_runtime_containment_populates_snapshot():
    X, y = _small_dataset(953)

    cfg = DFFSConfig(
        random_seed=38,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        model_candidates=("lr", "svm_rbf", "dlda", "svm_linear", "knn"),
        include_dlda_model=True,
        include_svm_linear_model=True,
        include_knn_model=True,
        model_cv_runtime_containment_enabled=True,
        model_cv_runtime_max_candidates=2,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=38)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert bool(res.config_snapshot["model_cv_runtime_containment_enabled"]) is True
    assert bool(res.config_snapshot["model_cv_runtime_containment_applied"]) is True
    assert str(res.config_snapshot["model_cv_runtime_containment_reason"]) == "hard_cap"
    assert tuple(res.config_snapshot["model_cv_effective_candidates"]) == ("lr", "svm_rbf")
    assert set(res.config_snapshot["model_cv_dropped_candidates"]) >= {"dlda", "svm_linear", "knn"}
    assert int(res.config_snapshot["model_cv_runtime_cap"]) == 2


def test_pipeline_optuna_backend_falls_back_when_optuna_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    X, y = _small_dataset(954)

    cls_cfg = df_fs_pipeline.ClassificationConfig(
        selection_mode="legacy",
        backend="optuna",
        optuna_time_budget=15,
        optuna_n_trials=2,
        min_n_for_automl=2,
        min_n_per_class_for_cv=2,
        min_n_per_class_for_automl=2,
        max_p_over_n_for_automl=10000,
    )
    cfg = DFFSConfig(
        random_seed=39,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        classification=cls_cfg,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=39)

    assert np.isfinite(res.balanced_accuracy)
    assert str(res.config_snapshot.get("classification_backend_requested")) == "optuna"
    assert str(res.config_snapshot.get("classification_backend_used")) == "sklearn"
    assert str(res.config_snapshot.get("classification_backend_fallback_reason")) == "ImportError"


def test_pipeline_classifier_conformal_snapshot_fields():
    X, y = _small_dataset(955)

    cls_cfg = df_fs_pipeline.ClassificationConfig(
        conformal_enabled=True,
        conformal_alpha=0.12,
        conformal_calibration_fraction=0.30,
        conformal_min_calibration=12,
        conformal_output_sets=False,
    )
    cfg = DFFSConfig(
        random_seed=40,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        classification=cls_cfg,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=40)

    assert np.isfinite(res.balanced_accuracy)
    assert bool(res.config_snapshot.get("classifier_conformal_enabled", False)) is True
    assert "classifier_conformal_applied" in res.config_snapshot
    if bool(res.config_snapshot.get("classifier_conformal_applied", False)):
        assert 0.0 <= float(res.config_snapshot.get("classifier_conformal_coverage", float("nan"))) <= 1.0
        assert float(res.config_snapshot.get("classifier_conformal_set_size_mean", float("nan"))) >= 1.0


def test_pipeline_supports_maqc_pairing_snapshot_fields():
    X, y = _small_dataset(951)

    enabled = ("gradient_boosting", "linear_svm", "mutual_information", "anova_f", "mrmr_jmi")
    cfg = DFFSConfig(
        random_seed=36,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=enabled,
        enable_maqc_pairing=True,
        maqc_pairing_method_sets=(enabled,),
        maqc_pairing_method_set_names=("strict_plus_mrmr",),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=36)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert bool(res.config_snapshot["enable_maqc_pairing"]) is True
    assert bool(res.config_snapshot.get("maqc_pairing_enabled", False)) is True
    assert str(res.config_snapshot.get("enabled_methods_source", "")) == "maqc_pairing"
    assert str(res.config_snapshot.get("maqc_pairing_selected_fs_name", "")) != ""


def test_select_model_via_cv_skips_unavailable_optional_backends(monkeypatch):
    X, y = _small_dataset(949)

    monkeypatch.setattr(df_fs_pipeline, "_XGBClassifier", None)
    monkeypatch.setattr(df_fs_pipeline, "_TabPFNClassifier", None)

    cfg = DFFSConfig(
        random_seed=34,
        model_candidates=("xgb", "tabpfn"),
        include_xgb_model=True,
        include_tabpfn_model=True,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    _, model_name = pipe._select_model_via_cv(X[:64], y[:64], seed=34)

    assert model_name == "lr"


def test_pipeline_records_tabpfn_build_failure_in_snapshot(monkeypatch):
    X, y = _small_dataset(956)

    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "_build_tabpfn_model",
        lambda self, seed: (None, "RuntimeError: tabpfn init exploded"),
    )

    cfg = DFFSConfig(
        random_seed=41,
        model_candidates=("lr", "tabpfn"),
        include_tabpfn_model=True,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=41)

    assert np.isfinite(res.balanced_accuracy)
    assert "tabpfn" in tuple(res.config_snapshot.get("model_cv_effective_candidates", ()))
    assert "tabpfn" not in tuple(res.config_snapshot.get("model_cv_constructed_candidates", ()))
    assert "tabpfn" not in tuple(res.config_snapshot.get("model_cv_evaluated_candidates", ()))
    assert dict(res.config_snapshot.get("model_cv_candidate_build_failures", {})).get("tabpfn") == (
        "RuntimeError: tabpfn init exploded"
    )


def test_pipeline_records_tabpfn_eval_failure_in_snapshot(monkeypatch):
    X, y = _small_dataset(957)

    class _ExplodingEstimator(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            raise RuntimeError("tabpfn fit exploded")

        def predict(self, X):
            return np.zeros(np.asarray(X).shape[0], dtype=int)

    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "_build_tabpfn_model",
        lambda self, seed: (_ExplodingEstimator(), None),
    )

    cfg = DFFSConfig(
        random_seed=42,
        model_candidates=("lr", "tabpfn"),
        include_tabpfn_model=True,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=42)

    assert np.isfinite(res.balanced_accuracy)
    assert "tabpfn" in tuple(res.config_snapshot.get("model_cv_effective_candidates", ()))
    assert "tabpfn" in tuple(res.config_snapshot.get("model_cv_constructed_candidates", ()))
    assert "tabpfn" in tuple(res.config_snapshot.get("model_cv_failed_candidates", ()))
    assert "tabpfn" not in tuple(res.config_snapshot.get("model_cv_evaluated_candidates", ()))
    assert "RuntimeError: tabpfn fit exploded" in str(
        dict(res.config_snapshot.get("model_cv_candidate_failure_reasons", {})).get("tabpfn", "")
    )


def test_pipeline_supports_decorrelated_stability_controls():
    X, y = _small_dataset(932)

    cfg = DFFSConfig(
        random_seed=22,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("decorrelated_stability", "linear_svm", "mutual_information", "anova_f"),
        fs_decorrelated_stability_eps=5e-3,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=22)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert np.isclose(float(res.config_snapshot["fs_decorrelated_stability_eps"]), 5e-3)


def test_pipeline_supports_subspace_stability_method():
    X, y = _small_dataset(933)

    cfg = DFFSConfig(
        random_seed=23,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("subspace_stability", "linear_svm", "mutual_information", "anova_f"),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=23)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "subspace_stability" in set(res.config_snapshot["enabled_methods"])


def test_pipeline_supports_tigress_stability_method():
    X, y = _small_dataset(937)

    cfg = DFFSConfig(
        random_seed=25,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("tigress_stability", "linear_svm", "mutual_information", "anova_f"),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=25)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "tigress_stability" in set(res.config_snapshot["enabled_methods"])


def test_pipeline_supports_rank_aggregation_mode_snapshot():
    X, y = _small_dataset(939)

    cfg = DFFSConfig(
        random_seed=27,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("stability_subsample", "linear_svm", "mutual_information", "anova_f"),
        fs_rank_aggregation_mode="rra",
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=27)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert str(res.config_snapshot["fs_rank_aggregation_mode"]) == "rra"


def test_pipeline_supports_wrapper_refinement_snapshot():
    X, y = _small_dataset(941)

    cfg = DFFSConfig(
        random_seed=28,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("stability_subsample", "linear_svm", "mutual_information", "anova_f"),
        fs_wrapper_refine_enabled=True,
        fs_wrapper_refine_top_k=12,
        fs_wrapper_refine_max_add=6,
        fs_wrapper_refine_min_gain=0.0,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=28)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert bool(res.config_snapshot["fs_wrapper_refine_enabled"]) is True
    assert int(res.config_snapshot["fs_wrapper_refine_top_k"]) == 12
    assert int(res.config_snapshot["fs_wrapper_refine_max_add"]) == 6
    assert np.isclose(float(res.config_snapshot["fs_wrapper_refine_min_gain"]), 0.0)


def test_pipeline_supports_ova_ensemble_method():
    X, y = _small_dataset(943)

    cfg = DFFSConfig(
        random_seed=30,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("ova_ensemble", "linear_svm", "mutual_information", "anova_f"),
        fs_ova_negative_ratio=1.7,
        fs_ova_aggregation_mode="p_norm",
        fs_ova_aggregation_p=3.0,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=30)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "ova_ensemble" in set(res.config_snapshot["enabled_methods"])
    assert np.isclose(float(res.config_snapshot["fs_ova_negative_ratio"]), 1.7)
    assert int(res.config_snapshot["fs_ova_min_classes"]) == 5
    assert str(res.config_snapshot["fs_ova_aggregation_mode"]) == "p_norm"
    assert np.isclose(float(res.config_snapshot["fs_ova_aggregation_p"]), 3.0)


def test_pipeline_supports_ecoc_class_aware_method_snapshot():
    X, y = _small_dataset(947)

    cfg = DFFSConfig(
        random_seed=33,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("ecoc_class_aware", "linear_svm", "mutual_information", "anova_f"),
        fs_ecoc_min_classes=3,
        fs_ecoc_max_ovo_pairs=5,
        fs_ecoc_random_code_bits=2,
        fs_ecoc_class_complexity_weight=1.1,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=33)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "ecoc_class_aware" in set(res.config_snapshot["enabled_methods"])
    assert int(res.config_snapshot["fs_ecoc_min_classes"]) == 3
    assert int(res.config_snapshot["fs_ecoc_max_ovo_pairs"]) == 5
    assert int(res.config_snapshot["fs_ecoc_random_code_bits"]) == 2
    assert np.isclose(float(res.config_snapshot["fs_ecoc_class_complexity_weight"]), 1.1)


def test_pipeline_supports_joint_multiclass_method_snapshot():
    X, y = _small_dataset(949)

    cfg = DFFSConfig(
        random_seed=35,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("joint_multiclass_support", "linear_svm", "mutual_information", "anova_f"),
        fs_joint_multiclass_min_classes=3,
        fs_joint_multiclass_max_features=120,
        fs_joint_multiclass_path_grid_size=4,
        fs_joint_multiclass_min_c=0.08,
        fs_joint_multiclass_max_c=1.4,
        fs_joint_multiclass_l1_ratio=0.6,
        fs_joint_multiclass_univariate_blend=0.25,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=35)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "joint_multiclass_support" in set(res.config_snapshot["enabled_methods"])
    assert int(res.config_snapshot["fs_joint_multiclass_path_grid_size"]) == 4
    assert np.isclose(float(res.config_snapshot["fs_joint_multiclass_l1_ratio"]), 0.6)
    assert np.isclose(float(res.config_snapshot["fs_joint_multiclass_univariate_blend"]), 0.25)


def test_pipeline_supports_dove_method_snapshot():
    X, y = _small_dataset(952)

    cfg = DFFSConfig(
        random_seed=38,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("dove_class_specific", "linear_svm", "mutual_information", "anova_f"),
        fs_dove_min_classes=3,
        fs_dove_max_pairs_per_class=3,
        fs_dove_path_grid_size=4,
        fs_dove_specificity_weight=0.4,
        fs_dove_minority_boost=0.7,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=38)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "dove_class_specific" in set(res.config_snapshot["enabled_methods"])
    assert int(res.config_snapshot["fs_dove_max_pairs_per_class"]) == 3
    assert int(res.config_snapshot["fs_dove_path_grid_size"]) == 4
    assert np.isclose(float(res.config_snapshot["fs_dove_specificity_weight"]), 0.4)
    assert np.isclose(float(res.config_snapshot["fs_dove_minority_boost"]), 0.7)


def test_pipeline_supports_sparse_multinomial_method_snapshot():
    X, y = _small_dataset(953)

    cfg = DFFSConfig(
        random_seed=39,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("sparse_multinomial", "linear_svm", "mutual_information", "anova_f"),
        fs_sparse_multinomial_min_classes=3,
        fs_sparse_multinomial_max_features=120,
        fs_sparse_multinomial_path_grid_size=4,
        fs_sparse_multinomial_min_c=0.08,
        fs_sparse_multinomial_max_c=1.4,
        fs_sparse_multinomial_backend="elasticnet",
        fs_sparse_multinomial_l1_ratio=0.6,
        fs_sparse_multinomial_univariate_blend=0.3,
        fs_sparse_multinomial_max_iter=3500,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=39)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "sparse_multinomial" in set(res.config_snapshot["enabled_methods"])
    assert int(res.config_snapshot["fs_sparse_multinomial_path_grid_size"]) == 4
    assert np.isclose(float(res.config_snapshot["fs_sparse_multinomial_min_c"]), 0.08)
    assert str(res.config_snapshot["fs_sparse_multinomial_backend"]) == "elasticnet"
    assert np.isclose(float(res.config_snapshot["fs_sparse_multinomial_l1_ratio"]), 0.6)
    assert int(res.config_snapshot["fs_sparse_multinomial_max_iter"]) == 3500


def test_pipeline_supports_nsc_method_snapshot():
    X, y = _small_dataset(955)

    cfg = DFFSConfig(
        random_seed=41,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("nearest_shrunken_centroid", "linear_svm", "mutual_information", "anova_f"),
        fs_nsc_shrinkage_grid_size=7,
        fs_nsc_min_classes=3,
        fs_nsc_thresholding_mode="auto",
        fs_nsc_order_quantile=0.80,
        fs_nsc_deep_shrinkage_search=True,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=41)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "nearest_shrunken_centroid" in set(res.config_snapshot["enabled_methods"])
    assert int(res.config_snapshot["fs_nsc_shrinkage_grid_size"]) == 7
    assert int(res.config_snapshot["fs_nsc_min_classes"]) == 3
    assert str(res.config_snapshot["fs_nsc_thresholding_mode"]) == "auto"
    assert np.isclose(float(res.config_snapshot["fs_nsc_order_quantile"]), 0.80)
    assert bool(res.config_snapshot["fs_nsc_deep_shrinkage_search"]) is True


def test_pipeline_supports_class_pareto_snapshot():
    X, y = _small_dataset(957)

    cfg = DFFSConfig(
        random_seed=43,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("class_pareto_front", "linear_svm", "mutual_information", "anova_f"),
        fs_class_pareto_min_classes=3,
        fs_class_pareto_top_per_class=24,
        fs_class_pareto_global_fraction=0.30,
        fs_class_pareto_minority_boost=0.70,
        fs_class_pareto_kw_weight=0.20,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=43)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "class_pareto_front" in set(res.config_snapshot["enabled_methods"])
    assert int(res.config_snapshot["fs_class_pareto_top_per_class"]) == 24
    assert np.isclose(float(res.config_snapshot["fs_class_pareto_global_fraction"]), 0.30)
    assert np.isclose(float(res.config_snapshot["fs_class_pareto_kw_weight"]), 0.20)


def test_pipeline_supports_hsic_lasso_snapshot():
    X, y = _small_dataset(958)

    cfg = DFFSConfig(
        random_seed=44,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("hsic_lasso", "linear_svm", "mutual_information", "anova_f"),
        fs_hsic_lasso_alpha=0.02,
        fs_hsic_lasso_prefilter_max_features=64,
        fs_hsic_lasso_feature_sigma=0.0,
        fs_hsic_lasso_target_sigma=0.0,
        fs_hsic_lasso_relevance_blend=0.30,
        fs_hsic_lasso_max_iter=2500,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=44)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "hsic_lasso" in set(res.config_snapshot["enabled_methods"])
    assert np.isclose(float(res.config_snapshot["fs_hsic_lasso_alpha"]), 0.02)
    assert int(res.config_snapshot["fs_hsic_lasso_prefilter_max_features"]) == 64
    assert np.isclose(float(res.config_snapshot["fs_hsic_lasso_relevance_blend"]), 0.30)
    assert int(res.config_snapshot["fs_hsic_lasso_max_iter"]) == 2500


def test_pipeline_supports_folding_stage_snapshot():
    X, y = _small_dataset(956)

    cfg = DFFSConfig(
        random_seed=42,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        folding_method="rff",
        folding_n_components=64,
        folding_rff_gamma=0.25,
        folding_prefilter_k=80,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=42)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert bool(res.config_snapshot["folding_applied"]) is True
    assert str(res.config_snapshot["folding_method"]) == "rff"
    assert int(res.config_snapshot["folding_output_dim"]) == 64
    assert str(res.config_snapshot.get("df_stage_source_space", "")) == "prefilter_raw"
    assert len(res.selected_feature_indices_original) > 0


def test_pipeline_supports_pls_da_folding_snapshot():
    X, y = _small_dataset(959)

    cfg = DFFSConfig(
        random_seed=45,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        folding_method="pls_da",
        folding_pls_components=2,
        folding_pls_scale=False,
        folding_pls_min_classes=3,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(
        X,
        y,
        dataset_name="tiny",
        seed=45,
        capture_artifacts=True,
        capture_diagnostics=True,
    )

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert bool(res.config_snapshot["folding_applied"]) is True
    assert str(res.config_snapshot["folding_method"]) == "pls_da"
    assert int(res.config_snapshot["folding_pls_components"]) == 2
    assert bool(res.config_snapshot["folding_pls_scale"]) is False
    assert int(res.config_snapshot["folding_output_dim"]) >= 1
    assert str(res.config_snapshot.get("df_stage_source_space", "")) == "prefilter_raw"
    assert len(res.selected_feature_indices_original) > 0

    diag = dict(res.run_diagnostics or {})
    dist_stage = dict((dict(diag.get("pipeline_stages") or {})).get("distribution_stage") or {})
    dist_summary = dict(dist_stage.get("summary") or {})
    assert str(dist_summary.get("df_stage_position", "")) == "after_fs"
    assert str(dist_summary.get("df_stage_source_space", "")) == "prefilter_raw"

    bundle = dict(res.model_bundle or {})
    deployed = df_fs_pipeline.DFFSReproducibleModel.from_json_dict(bundle)
    test_idx = np.asarray(res.split_indices_test, dtype=int)
    X_test = np.asarray(X, dtype=float)[test_idx]
    y_test = np.asarray(y).ravel()[test_idx]
    y_pred = np.asarray(deployed.predict(X_test), dtype=np.asarray(y_test).dtype).ravel()
    bal = pipe._safe_balanced_accuracy(y_test, y_pred)
    assert np.isfinite(float(bal))
    assert abs(float(bal) - float(res.balanced_accuracy)) < 1e-12


def test_pipeline_pls_da_folding_guard_and_component_cap():
    # Binary: PLS-DA folding is disabled (insufficient classes).
    Xb, yb = make_classification(
        n_samples=64,
        n_features=120,
        n_informative=14,
        n_redundant=10,
        n_classes=2,
        weights=[0.55, 0.45],
        class_sep=1.1,
        random_state=1001,
    )
    Xb = np.asarray(Xb, dtype=float)
    yb = np.asarray(yb, dtype=int)

    cfg_bin = DFFSConfig(
        random_seed=11,
        fs_fraction=0.5,
        n_final_features=8,
        max_dist_features=12,
        prefilter_top_k=60,
        folding_method="pls_da",
        folding_pls_components=16,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
    )
    res_bin = DistributionFeatureSelectionPipeline(cfg_bin).run(Xb, yb, dataset_name="tiny_bin", seed=11)
    assert bool(res_bin.config_snapshot["folding_applied"]) is False
    assert str(res_bin.config_snapshot["folding_reason"]) == "pls_da_insufficient_classes"

    # Multiclass: C=4 with default min_classes=5 should skip.
    Xm, ym = make_classification(
        n_samples=96,
        n_features=80,
        n_informative=18,
        n_redundant=12,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.1,
        random_state=1002,
    )
    Xm = np.asarray(Xm, dtype=float)
    ym = np.asarray(ym, dtype=int)

    cfg_multi = DFFSConfig(
        random_seed=12,
        fs_fraction=0.5,
        n_final_features=10,
        max_dist_features=12,
        prefilter_top_k=70,
        folding_method="pls_da",
        folding_pls_components=16,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
    )
    res_multi = DistributionFeatureSelectionPipeline(cfg_multi).run(Xm, ym, dataset_name="tiny_multi", seed=12)
    assert bool(res_multi.config_snapshot["folding_applied"]) is False
    assert str(res_multi.config_snapshot["folding_reason"]) == "pls_da_insufficient_classes"

    # Override guardrail to validate component cap behavior.
    cfg_multi_relaxed = DFFSConfig(
        random_seed=12,
        fs_fraction=0.5,
        n_final_features=10,
        max_dist_features=12,
        prefilter_top_k=70,
        folding_method="pls_da",
        folding_pls_components=16,
        folding_pls_min_classes=3,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
    )
    res_multi_relaxed = DistributionFeatureSelectionPipeline(cfg_multi_relaxed).run(
        Xm, ym, dataset_name="tiny_multi", seed=12
    )
    assert bool(res_multi_relaxed.config_snapshot["folding_applied"]) is True
    assert int(res_multi_relaxed.config_snapshot["folding_pls_components_used"]) == 3
    assert int(res_multi_relaxed.config_snapshot["folding_output_dim"]) == 3


def test_pipeline_rff_folding_auto_gamma_uses_scale_default():
    X, y = _small_dataset(960)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    cfg = DFFSConfig(
        random_seed=13,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=40,
        folding_method="rff",
        folding_n_components=64,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
    )
    res = DistributionFeatureSelectionPipeline(cfg).run(X, y, dataset_name="tiny_rff", seed=13)
    assert bool(res.config_snapshot["folding_applied"]) is True
    assert len(res.selected_feature_indices_original) > 0
    assert float(res.config_snapshot["folding_rff_gamma"]) == pytest.approx(
        1.0 / float(len(res.selected_feature_indices_original))
    )


def test_pipeline_folding_stage_standardizes_low_variance_maps():
    X, y = make_classification(
        n_samples=28,
        n_features=160,
        n_informative=18,
        n_redundant=12,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=123,
    )
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    cfg = DFFSConfig(
        random_seed=11,
        fs_fraction=0.5,
        n_final_features=8,
        max_dist_features=12,
        prefilter_top_k=120,
        folding_method="rff",
        folding_n_components=256,
        folding_rff_gamma=0.2,
        folding_prefilter_k=120,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny_small", seed=11)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert bool(res.config_snapshot["folding_standardize_applied"]) is True
    assert np.isfinite(float(res.config_snapshot["folding_standardize_min_train_std"]))


def test_pipeline_applies_face_projection_on_catalog_face_dataset_when_enabled():
    X, y = _small_dataset(957)

    cfg = DFFSConfig(
        random_seed=43,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enable_face_domain_projection=True,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="orlraws10p", seed=43)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert bool(res.config_snapshot["face_projection_is_face_domain"]) is True
    assert bool(res.config_snapshot["face_projection_applied"]) is True
    assert str(res.config_snapshot["face_projection_mode"]) in {"pca_lda", "pca_only"}
    assert int(res.config_snapshot["face_projection_output_dim"]) >= 1
    assert len(res.selected_feature_indices_original) == 0


def test_pipeline_supports_runtime_racing_controls_snapshot():
    X, y = _small_dataset(954)

    cfg = DFFSConfig(
        random_seed=40,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
        fs_runtime_racing_enabled=True,
        fs_runtime_racing_proxy_splits=2,
        fs_runtime_racing_keep_fraction=0.5,
        fs_runtime_racing_min_candidates=2,
        fs_runtime_racing_runtime_weight=0.2,
        fs_runtime_racing_mode="successive_halving",
        fs_runtime_racing_stages=3,
        fs_runtime_racing_confidence_bound="hoeffding",
        fs_runtime_racing_delta=0.08,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=40)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert bool(res.config_snapshot["fs_runtime_racing_enabled"]) is True
    assert int(res.config_snapshot["fs_runtime_racing_proxy_splits"]) == 2
    assert np.isclose(float(res.config_snapshot["fs_runtime_racing_keep_fraction"]), 0.5)
    assert int(res.config_snapshot["fs_runtime_racing_min_candidates"]) == 2
    assert np.isclose(float(res.config_snapshot["fs_runtime_racing_runtime_weight"]), 0.2)
    assert str(res.config_snapshot["fs_runtime_racing_mode"]) == "successive_halving"
    assert int(res.config_snapshot["fs_runtime_racing_stages"]) == 3
    assert str(res.config_snapshot["fs_runtime_racing_confidence_bound"]) == "hoeffding"
    assert np.isclose(float(res.config_snapshot["fs_runtime_racing_delta"]), 0.08)


def test_pipeline_supports_ova_sparse_quota_and_model_extension_snapshot():
    X, y = _small_dataset(960)

    cfg = DFFSConfig(
        random_seed=46,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
        fs_ova_enable_calibration=True,
        fs_ova_calibration_cv=4,
        fs_sparse_multinomial_screening_mode="prefilter_aggressive",
        fs_sparse_multinomial_screening_keep_fraction=0.65,
        fs_sparse_multinomial_screening_min_features=70,
        fs_sparse_multinomial_screening_fallback_on_failure=False,
        fs_per_class_quota_enabled=True,
        fs_per_class_quota_min_per_class=2,
        fs_per_class_quota_max_fraction=0.5,
        include_nb_model=True,
        include_vote_ensemble_model=True,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=46)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert bool(res.config_snapshot["fs_ova_enable_calibration"]) is True
    assert int(res.config_snapshot["fs_ova_calibration_cv"]) == 4
    assert str(res.config_snapshot["fs_sparse_multinomial_screening_mode"]) == "prefilter_aggressive"
    assert np.isclose(
        float(res.config_snapshot["fs_sparse_multinomial_screening_keep_fraction"]),
        0.65,
    )
    assert int(res.config_snapshot["fs_sparse_multinomial_screening_min_features"]) == 70
    assert bool(res.config_snapshot["fs_sparse_multinomial_screening_fallback_on_failure"]) is False
    assert bool(res.config_snapshot["fs_per_class_quota_enabled"]) is True
    assert int(res.config_snapshot["fs_per_class_quota_min_per_class"]) == 2
    assert np.isclose(float(res.config_snapshot["fs_per_class_quota_max_fraction"]), 0.5)
    assert bool(res.config_snapshot["include_nb_model"]) is True
    assert bool(res.config_snapshot["include_vote_ensemble_model"]) is True


def test_pipeline_supports_iterative_pruning_controls_snapshot():
    X, y = _small_dataset(948)

    cfg = DFFSConfig(
        random_seed=34,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("iterative_redundancy_pruning", "linear_svm", "mutual_information", "anova_f"),
        fs_iterative_pruning_pool_factor=2.1,
        fs_iterative_pruning_max_rounds=14,
        fs_iterative_pruning_min_improvement=-0.01,
        fs_iterative_pruning_max_cumulative_loss=0.015,
        fs_iterative_pruning_redundancy_weight=0.7,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=34)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "iterative_redundancy_pruning" in set(res.config_snapshot["enabled_methods"])
    assert np.isclose(float(res.config_snapshot["fs_iterative_pruning_pool_factor"]), 2.1)
    assert int(res.config_snapshot["fs_iterative_pruning_max_rounds"]) == 14
    assert np.isclose(float(res.config_snapshot["fs_iterative_pruning_min_improvement"]), -0.01)
    assert np.isclose(float(res.config_snapshot["fs_iterative_pruning_max_cumulative_loss"]), 0.015)
    assert np.isclose(float(res.config_snapshot["fs_iterative_pruning_redundancy_weight"]), 0.7)


def test_pipeline_supports_iterative_pruning_bounded_controls_snapshot():
    X, y = _small_dataset(950)

    cfg = DFFSConfig(
        random_seed=36,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("iterative_redundancy_pruning_bounded", "linear_svm", "mutual_information", "anova_f"),
        fs_iterative_pruning_bounded_prefilter_cap=140,
        fs_iterative_pruning_bounded_candidate_fraction=0.30,
        fs_iterative_pruning_bounded_min_candidates=3,
        fs_iterative_pruning_bounded_max_evaluations=12,
        fs_iterative_pruning_bounded_max_runtime_seconds=9.0,
        fs_iterative_pruning_bounded_multiclass_scale=0.65,
        fs_iterative_pruning_bounded_imbalance_trigger=2.2,
        fs_iterative_pruning_bounded_imbalance_scale=0.70,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=36)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert "iterative_redundancy_pruning_bounded" in set(res.config_snapshot["enabled_methods"])
    assert int(res.config_snapshot["fs_iterative_pruning_bounded_prefilter_cap"]) == 140
    assert np.isclose(float(res.config_snapshot["fs_iterative_pruning_bounded_candidate_fraction"]), 0.30)
    assert int(res.config_snapshot["fs_iterative_pruning_bounded_max_evaluations"]) == 12
    assert np.isclose(float(res.config_snapshot["fs_iterative_pruning_bounded_imbalance_scale"]), 0.70)


def test_pipeline_supports_iterative_pruning_bounded_cpss_and_pareto_snapshot():
    X, y = _small_dataset(951)

    cfg = DFFSConfig(
        random_seed=37,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("iterative_redundancy_pruning_bounded", "linear_svm", "mutual_information", "anova_f"),
        fs_iterative_pruning_bounded_use_cpss_overlay=True,
        fs_iterative_pruning_bounded_cpss_pairs=3,
        fs_iterative_pruning_bounded_cpss_stability_threshold=0.58,
        fs_iterative_pruning_bounded_cpss_min_stable_features=2,
        fs_iterative_pruning_bounded_cpss_min_jaccard=0.25,
        fs_iterative_pruning_bounded_cpss_max_score_drop=0.015,
        fs_iterative_pruning_class_pareto_prefilter_enabled=True,
        fs_iterative_pruning_class_pareto_min_classes=3,
        fs_iterative_pruning_class_pareto_top_per_class=24,
        fs_iterative_pruning_class_pareto_global_fraction=0.30,
        fs_iterative_pruning_class_pareto_minority_boost=0.7,
        fs_iterative_pruning_class_pareto_stability_gate_enabled=True,
        fs_iterative_pruning_class_pareto_stability_subsamples=5,
        fs_iterative_pruning_class_pareto_stability_fraction=0.72,
        fs_iterative_pruning_class_pareto_stability_threshold=0.56,
        fs_iterative_pruning_class_pareto_stability_min_overlap=0.45,
        fs_iterative_pruning_class_pareto_stability_min_stable_features=2,
        fs_iterative_pruning_class_pareto_stability_fallback_on_failure=False,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=37)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert bool(res.config_snapshot["fs_iterative_pruning_bounded_use_cpss_overlay"]) is True
    assert int(res.config_snapshot["fs_iterative_pruning_bounded_cpss_pairs"]) == 3
    assert np.isclose(float(res.config_snapshot["fs_iterative_pruning_bounded_cpss_stability_threshold"]), 0.58)
    assert bool(res.config_snapshot["fs_iterative_pruning_class_pareto_prefilter_enabled"]) is True
    assert int(res.config_snapshot["fs_iterative_pruning_class_pareto_min_classes"]) == 3
    assert np.isclose(float(res.config_snapshot["fs_iterative_pruning_class_pareto_global_fraction"]), 0.30)
    assert bool(res.config_snapshot["fs_iterative_pruning_class_pareto_stability_gate_enabled"]) is True
    assert int(res.config_snapshot["fs_iterative_pruning_class_pareto_stability_subsamples"]) == 5
    assert np.isclose(float(res.config_snapshot["fs_iterative_pruning_class_pareto_stability_fraction"]), 0.72)
    assert np.isclose(float(res.config_snapshot["fs_iterative_pruning_class_pareto_stability_min_overlap"]), 0.45)
    assert bool(res.config_snapshot["fs_iterative_pruning_class_pareto_stability_fallback_on_failure"]) is False


def test_pipeline_supports_loss_guided_stability_controls():
    X, y = _small_dataset(934)

    cfg = DFFSConfig(
        random_seed=24,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("stability_subsample", "linear_svm", "mutual_information", "anova_f"),
        fs_stability_use_loss_guided_validation=True,
        fs_stability_validation_fraction=0.30,
        fs_stability_validation_quantile=0.50,
        fs_stability_validation_min_samples=5,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=24)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert bool(res.config_snapshot["fs_stability_use_loss_guided_validation"]) is True
    assert np.isclose(float(res.config_snapshot["fs_stability_validation_fraction"]), 0.30)
    assert np.isclose(float(res.config_snapshot["fs_stability_validation_quantile"]), 0.50)
    assert int(res.config_snapshot["fs_stability_validation_min_samples"]) == 5


def test_pipeline_supports_ipss_and_eats_controls():
    X, y = _small_dataset(936)

    cfg = DFFSConfig(
        random_seed=23,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("ipss", "linear_svm", "mutual_information", "anova_f"),
        fs_ipss_path_grid_size=4,
        fs_ipss_target_fdr=0.20,
        fs_ipss_use_eats_threshold=True,
        fs_ipss_eats_min_threshold=0.50,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=23)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert bool(res.config_snapshot["fs_ipss_use_eats_threshold"]) is True
    assert int(res.config_snapshot["fs_ipss_path_grid_size"]) == 4
    assert np.isclose(res.config_snapshot["fs_ipss_target_fdr"], 0.20)


def test_pipeline_supports_portfolio_and_gate_controls_in_snapshot():
    X, y = _small_dataset(960)

    enabled = ("decorrelated_stability", "ipss", "linear_svm", "mutual_information", "anova_f")
    cfg = DFFSConfig(
        random_seed=40,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=enabled,
        fs_portfolio_size=5,
        fs_portfolio_size_guard="raise",
        fs_mnpo_consensus_exclude_methods=("linear_svm", "gradient_boosting"),
        fs_mnpo_consensus_exclude_protect_top_k=5,
        fs_mnpo_include_legacy_consensus=False,
        fs_mnpo_include_majority_consensus=False,
        fs_ipss_gate_min_classes=4,
        fs_ipss_gate_min_p_over_n=0.0,
        fs_decorrelated_stability_min_max_abs_corr=0.33,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=40)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert int(res.config_snapshot["fs_portfolio_size"]) == 5
    assert str(res.config_snapshot["fs_portfolio_size_guard"]) == "raise"
    assert "linear_svm" in set(res.config_snapshot.get("fs_mnpo_consensus_exclude_methods", []) or ())
    assert "gradient_boosting" in set(res.config_snapshot.get("fs_mnpo_consensus_exclude_methods", []) or ())
    assert int(res.config_snapshot.get("fs_mnpo_consensus_exclude_protect_top_k", 0)) == 5
    assert bool(res.config_snapshot.get("fs_mnpo_include_legacy_consensus", True)) is False
    assert bool(res.config_snapshot.get("fs_mnpo_include_majority_consensus", True)) is False
    assert int(res.config_snapshot["fs_ipss_gate_min_classes"]) == 4
    assert np.isclose(float(res.config_snapshot["fs_ipss_gate_min_p_over_n"]), 0.0)
    assert np.isclose(float(res.config_snapshot["fs_decorrelated_stability_min_max_abs_corr"]), 0.33)


def test_pipeline_supports_copula_knockoff_controls_in_snapshot():
    X, y = _small_dataset(944)

    cfg = DFFSConfig(
        random_seed=31,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=16,
        prefilter_top_k=90,
        enabled_methods=("linear_svm", "mutual_information", "anova_f"),
        fs_copula_knockoff_draws=7,
        fs_copula_alpha_kn=0.12,
        fs_copula_alpha_ebh=0.23,
        fs_copula_truncation_level=3,
        fs_copula_stabilizer_runs=3,
        fs_copula_stabilizer_use_ebh=True,
        fs_copula_stabilizer_seed_stride=101,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=31)

    assert np.isfinite(res.balanced_accuracy)
    assert np.isfinite(res.macro_f1)
    assert res.selected_features_count > 0
    assert int(res.config_snapshot["fs_copula_knockoff_draws"]) == 7
    assert np.isclose(float(res.config_snapshot["fs_copula_alpha_kn"]), 0.12)
    assert np.isclose(float(res.config_snapshot["fs_copula_alpha_ebh"]), 0.23)
    assert int(res.config_snapshot["fs_copula_truncation_level"]) == 3
    assert int(res.config_snapshot["fs_copula_stabilizer_runs"]) == 3
    assert bool(res.config_snapshot["fs_copula_stabilizer_use_ebh"]) is True
    assert int(res.config_snapshot["fs_copula_stabilizer_seed_stride"]) == 101


def _mock_summary(feature_index: int, gof_p: float = 0.2, cset_size: int = 2, has_heaping: bool = False) -> DistributionFitSummary:
    support = SupportProfile(
        inferred_support="real",
        frac_zero=0.0,
        min_value=-2.0,
        max_value=2.0,
        unique_ratio=1.0,
        is_near_constant=False,
    )
    audit = DataAuditReport(
        n_raw=40,
        n_clean=40,
        n_missing=0,
        n_unique=40,
        support=support,
        has_heaping=has_heaping,
        heaping_delta=None,
        outlier_fraction=0.0,
    )
    return DistributionFitSummary(
        feature_index=int(feature_index),
        family="norm",
        params=(0.0, 1.0),
        cvm_p=float(gof_p),
        ks_p=float(gof_p),
        simple_score=0.1,
        confidence_set=tuple(["norm"] * max(1, cset_size)),
        rejected=False,
        audit=audit,
    )


def test_cdf_block_gating_budget_fallback_preserves_per_feature_behavior():
    X, y = _small_dataset(2026)

    cfg = DFFSConfig(
        random_seed=21,
        fs_fraction=0.4,
        n_final_features=10,
        max_dist_features=20,
        cdf_reliability_gate=True,
        cdf_min_gof_p=0.0,
        cdf_block_gating_cv=True,
        cdf_block_gating_time_budget_sec=0.0,
        prefilter_top_k=80,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    res = pipe.run(X, y, dataset_name="tiny", seed=21)

    assert res.n_dist_features_fitted > 0
    assert res.n_dist_features_transformed > 0
    assert bool(res.cdf_block_gating_budget_hit) is True
    assert res.config_snapshot["cdf_block_gating_cv"] is True


def test_cdf_block_gating_can_skip_blocks_when_cv_prefers_baseline():
    cfg = DFFSConfig(
        cdf_block_gating_cv=True,
        cdf_block_gating_n_blocks=2,
        cdf_block_gating_min_block_size=2,
        cdf_block_gating_cv_splits=2,
        cdf_block_gating_max_blocks=2,
        cdf_block_gating_time_budget_sec=5.0,
        cdf_block_gating_min_improvement=0.0,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)

    X_train_base = np.full((30, 4), 1.0, dtype=float)
    y_train = np.array([0] * 15 + [1] * 15, dtype=int)
    transformed_payloads = {i: (np.full(30, -1.0, dtype=float), np.full(10, -1.0, dtype=float)) for i in range(4)}
    summary_map = {i: _mock_summary(i, gof_p=0.2, cset_size=2) for i in range(4)}

    # Deterministic scorer: baseline mean(=1.0) beats transformed mean(=-1.0).
    pipe._block_cv_score = lambda Xb, yb, seed: float(np.mean(Xb))  # type: ignore[method-assign]

    selected, stats = pipe._apply_cdf_block_gating_cv(
        X_train_base=X_train_base,
        y_train=y_train,
        transformed_payloads=transformed_payloads,
        summary_map=summary_map,
        seed=7,
    )

    assert selected == set()
    assert stats["n_blocks_evaluated"] >= 1
    assert stats["n_features_skipped_cv"] == 4


def test_audit_distribution_summaries_breaks_down_rejections_and_cdf_skips():
    cfg = DFFSConfig(
        cdf_reliability_gate=True,
        cdf_skip_heaped_features=True,
        cdf_min_gof_p=0.25,
        cdf_max_confidence_set=2,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)

    def _audit(
        inferred_support: str = "real",
        n_clean: int = 50,
        is_near_constant: bool = False,
        has_heaping: bool = False,
    ) -> DataAuditReport:
        support = SupportProfile(
            inferred_support=inferred_support,
            frac_zero=0.0,
            min_value=0.0,
            max_value=1.0,
            unique_ratio=0.50,
            is_near_constant=is_near_constant,
        )
        return DataAuditReport(
            n_raw=n_clean,
            n_clean=n_clean,
            n_missing=0,
            n_unique=max(1, int(round(n_clean * support.unique_ratio))),
            support=support,
            has_heaping=has_heaping,
            heaping_delta=None,
            outlier_fraction=0.0,
        )

    # 1) Immediate rejection: near-constant.
    s_near_const = DistributionFitSummary(
        feature_index=0,
        family=None,
        params=None,
        cvm_p=float("nan"),
        ks_p=float("nan"),
        simple_score=float("nan"),
        confidence_set=tuple(),
        rejected=True,
        audit=_audit(is_near_constant=True),
    )

    # 2) DF fit succeeded but GOF rejects (both p-values under threshold).
    s_gof_reject = DistributionFitSummary(
        feature_index=1,
        family="norm",
        params=(0.0, 1.0),
        cvm_p=0.0,
        ks_p=0.0,
        simple_score=1.0,
        confidence_set=("norm",),
        rejected=True,
        audit=_audit(),
    )

    # 3) Non-rejected but skipped by reliability gate due to heaping.
    s_heaped = DistributionFitSummary(
        feature_index=2,
        family="norm",
        params=(0.0, 1.0),
        cvm_p=0.9,
        ks_p=0.9,
        simple_score=1.0,
        confidence_set=("norm",),
        rejected=False,
        audit=_audit(has_heaping=True),
    )

    # 4) Non-rejected but skipped by reliability gate due to low GOF.
    s_low_gof = DistributionFitSummary(
        feature_index=3,
        family="norm",
        params=(0.0, 1.0),
        cvm_p=0.1,
        ks_p=0.1,
        simple_score=1.0,
        confidence_set=("norm",),
        rejected=False,
        audit=_audit(),
    )

    # 5) Non-rejected and should apply CDF transform.
    s_ok = DistributionFitSummary(
        feature_index=4,
        family="norm",
        params=(0.0, 1.0),
        cvm_p=0.9,
        ks_p=0.9,
        simple_score=1.0,
        confidence_set=("norm",),
        rejected=False,
        audit=_audit(),
    )

    summary = pipe.audit_distribution_summaries([s_near_const, s_gof_reject, s_heaped, s_low_gof, s_ok])

    assert summary["n_total"] == 5
    assert summary["n_rejected"] == 2
    assert summary["rejection_reasons"].get("near_constant", 0) == 1
    assert summary["rejection_reasons"].get("gof_reject", 0) == 1
    assert summary["cdf_gate_enabled"] is True
    assert summary["n_non_rejected"] == 3
    assert summary["n_heaped"] == 1
    assert summary["n_cdf_should_apply"] == 1
    assert summary["n_cdf_skipped"] == 2
    assert summary["cdf_skip_reasons"].get("heaping", 0) == 1
    assert summary["cdf_skip_reasons"].get("low_gof", 0) == 1


def test_multimodal_fallback_routes_feature_to_gmm_branch_not_rejected():
    rng = np.random.default_rng(29)
    train_col = np.concatenate(
        [
            rng.normal(loc=-3.0, scale=0.7, size=120),
            rng.normal(loc=3.5, scale=0.8, size=120),
        ]
    ).astype(float)
    test_col = np.concatenate(
        [
            rng.normal(loc=-2.8, scale=0.8, size=80),
            rng.normal(loc=3.2, scale=0.9, size=80),
        ]
    ).astype(float)

    cfg = DFFSConfig(
        random_seed=29,
        multimodal_fallback="gmm",
        dist_config=DistributionFitterConfig(
            use_support_filtering=True,
            use_lrt=False,
            use_cv=False,
            compute_dip=True,
            dip_hist_bins=60,
        ),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    payload = pipe._fit_transform_one_feature(
        feat_idx=0,
        X_train_imp=train_col.reshape(-1, 1),
        X_test_imp=test_col.reshape(-1, 1),
        seed=29,
    )
    summary = payload["summary"]

    assert payload["apply_reason"] == "multimodal_fallback"
    assert bool(payload["rejected"]) is False
    assert summary.rejection_reason == "multimodal_fallback"
    assert bool(summary.rejected) is False
    assert summary.family is not None and str(summary.family).startswith("multimodal_fallback_")
    assert payload["train_z"] is not None
    assert payload["test_z"] is not None
    fallback_meta = dict(payload.get("fallback_meta") or {})
    assert str(fallback_meta.get("fallback_mode", "")) == "gmm"
    fit_diag = dict(fallback_meta.get("fit_diagnostics") or {})
    assert str(fit_diag.get("criterion", "")) == "bic"
    assert int(fit_diag.get("selected_components", 0)) in {2, 3}
    assert np.isfinite(float(fit_diag.get("selected_bic", float("nan"))))
    assert np.isfinite(float(fit_diag.get("selected_aic", float("nan"))))
    assert "candidate_metrics" in fit_diag


def test_multimodal_fallback_mnpo_bridge_assigns_weight_and_summary_fit_diagnostics():
    rng = np.random.default_rng(41)
    train_col = np.concatenate(
        [
            rng.normal(loc=-2.8, scale=0.7, size=100),
            rng.normal(loc=2.6, scale=0.8, size=100),
        ]
    ).astype(float)
    test_col = np.concatenate(
        [
            rng.normal(loc=-2.7, scale=0.8, size=70),
            rng.normal(loc=2.5, scale=0.9, size=70),
        ]
    ).astype(float)

    cfg = DFFSConfig(
        random_seed=41,
        dist_criterion="mnpo_oracle",
        multimodal_fallback="gmm",
        dist_config=DistributionFitterConfig(
            use_support_filtering=True,
            use_lrt=False,
            use_cv=False,
            compute_dip=True,
            dip_hist_bins=60,
        ),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    payload = pipe._fit_transform_one_feature(
        feat_idx=0,
        X_train_imp=train_col.reshape(-1, 1),
        X_test_imp=test_col.reshape(-1, 1),
        seed=41,
    )

    summary = payload["summary"]
    assert summary.rejection_reason == "multimodal_fallback"
    assert summary.fit_method == "multimodal_gmm_bic"
    assert summary.mnpo_weight == pytest.approx(1.0)
    assert np.isfinite(float(summary.bic))
    assert np.isfinite(float(summary.aic))
    assert np.isfinite(float(summary.loglik))

    fallback_meta = dict(payload.get("fallback_meta") or {})
    mnpo_context = dict(fallback_meta.get("mnpo_context") or {})
    assert bool(mnpo_context.get("included_in_mnpo", False)) is True
    assert float(mnpo_context.get("assigned_weight", 0.0)) == pytest.approx(1.0)


def test_multimodal_fallback_unimodal_data_uses_standard_path():
    quant_train = np.linspace(0.01, 0.99, 200)
    quant_test = np.linspace(0.02, 0.98, 120)
    train_col = sps.norm.ppf(quant_train).astype(float)
    test_col = sps.norm.ppf(quant_test).astype(float)

    cfg = DFFSConfig(
        random_seed=31,
        multimodal_fallback="gmm",
        dist_config=DistributionFitterConfig(
            use_support_filtering=True,
            use_lrt=False,
            use_cv=False,
            compute_dip=True,
            dip_hist_bins=50,
        ),
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)
    payload = pipe._fit_transform_one_feature(
        feat_idx=0,
        X_train_imp=train_col.reshape(-1, 1),
        X_test_imp=test_col.reshape(-1, 1),
        seed=31,
    )
    assert payload["apply_reason"] != "multimodal_fallback"


def test_distribution_summary_export_includes_rejection_and_candidate_fields():
    support = SupportProfile(
        inferred_support="real",
        frac_zero=0.0,
        min_value=-3.0,
        max_value=4.0,
        unique_ratio=1.0,
        is_near_constant=False,
    )
    audit = DataAuditReport(
        n_raw=120,
        n_clean=120,
        n_missing=0,
        n_unique=120,
        support=support,
        has_heaping=False,
        heaping_delta=None,
        outlier_fraction=0.0,
        frac_negative=0.35,
    )
    summary = DistributionFitSummary(
        feature_index=7,
        family="multimodal_fallback_gmm",
        params=None,
        cvm_p=float("nan"),
        ks_p=float("nan"),
        simple_score=float("nan"),
        confidence_set=tuple(),
        rejected=False,
        audit=audit,
        rejection_reason="multimodal_fallback",
        selected_family_support="real",
        candidates_pre_filter=19,
        candidates_post_filter=8,
    )
    row = df_fs_pipeline._dist_summary_to_dict(summary)
    assert row["rejection_reason"] == "multimodal_fallback"
    assert row["selected_family_support"] == "real"
    assert int(row["candidates_pre_filter"]) == 19
    assert int(row["candidates_post_filter"]) == 8


def test_audit_distribution_summaries_support_conflict_and_multimodal_attribution():
    cfg = DFFSConfig(cdf_reliability_gate=True)
    pipe = DistributionFeatureSelectionPipeline(cfg)

    real_support = SupportProfile(
        inferred_support="real",
        frac_zero=0.0,
        min_value=-2.0,
        max_value=2.0,
        unique_ratio=1.0,
        is_near_constant=False,
    )
    positive_support = SupportProfile(
        inferred_support="positive",
        frac_zero=0.0,
        min_value=0.0,
        max_value=4.0,
        unique_ratio=1.0,
        is_near_constant=False,
    )
    audit_real = DataAuditReport(
        n_raw=80,
        n_clean=80,
        n_missing=0,
        n_unique=80,
        support=real_support,
        has_heaping=False,
        heaping_delta=None,
        outlier_fraction=0.0,
        frac_negative=0.30,
    )
    audit_pos = DataAuditReport(
        n_raw=80,
        n_clean=80,
        n_missing=0,
        n_unique=80,
        support=positive_support,
        has_heaping=False,
        heaping_delta=None,
        outlier_fraction=0.0,
        frac_negative=0.0,
    )
    s_conflict = DistributionFitSummary(
        feature_index=0,
        family="gamma",
        params=(1.2, 0.0, 1.0),
        cvm_p=0.0,
        ks_p=0.0,
        simple_score=1.0,
        confidence_set=("gamma",),
        rejected=True,
        audit=audit_real,
        rejection_reason="support_conflict",
        selected_family_support="positive",
    )
    s_gof = DistributionFitSummary(
        feature_index=1,
        family="norm",
        params=(0.0, 1.0),
        cvm_p=0.0,
        ks_p=0.0,
        simple_score=1.0,
        confidence_set=("norm",),
        rejected=True,
        audit=audit_pos,
        rejection_reason="gof_reject",
        selected_family_support="real",
    )
    s_fallback = DistributionFitSummary(
        feature_index=2,
        family="multimodal_fallback_gmm",
        params=None,
        cvm_p=float("nan"),
        ks_p=float("nan"),
        simple_score=float("nan"),
        confidence_set=tuple(),
        rejected=False,
        audit=audit_real,
        rejection_reason="multimodal_fallback",
        selected_family_support="real",
    )

    summary = pipe.audit_distribution_summaries([s_conflict, s_gof, s_fallback])
    assert summary["n_total"] == 3
    assert summary["n_rejected"] == 2
    assert summary["support_conflict_selected"] == 1
    assert summary["true_gof_failure"] == 1
    assert summary["multimodal_fallback"] == 1
    assert summary["n_non_rejected"] == 1
    assert summary["rejection_reasons"].get("multimodal_fallback", 0) == 1


def test_evaluate_selector_candidate_seedsequence_does_not_mutate_global_rng(monkeypatch):
    class _FakeSelectionResult:
        selected_feature_indices = np.array([0, 1], dtype=int)
        selected_feature_votes = {}
        all_features_info = {}
        method_results = {}
        eliminated_features = {}
        feature_importance_mean = {}
        feature_importance_variance = {}
        unstable_feature_indices = []
        importance_uq = {}
        config = {}

        def to_summary_dict(self):
            return {}

    class _FakeSelector:
        def __init__(self):
            self.mnpo_diagnostics_ = {}

        def fit_transform(self, X, y, n_final_features, return_result_object):
            return np.asarray(X, dtype=float), _FakeSelectionResult()

        def transform(self, X):
            return np.asarray(X, dtype=float)

        def get_selected_features_indices(self):
            return np.array([0, 1], dtype=int)

    cfg = DFFSConfig(
        random_seed=77,
        n_final_features=2,
        enable_maqc_pairing=False,
    )
    pipe = DistributionFeatureSelectionPipeline(cfg)

    seen = {"selector_seed": [], "model_seed": []}

    def _fake_build_feature_selector(seed, enabled_methods, **kwargs):
        seen["selector_seed"].append(int(seed))
        return _FakeSelector()

    def _fake_select_model_via_cv_scored(X_train, y_train, seed):
        seen["model_seed"].append(int(seed))
        return LogisticRegression(max_iter=1000), "lr", 0.5, 0.0, 2, {}

    monkeypatch.setattr(pipe, "_build_feature_selector", _fake_build_feature_selector)
    monkeypatch.setattr(pipe, "_select_model_via_cv_scored", _fake_select_model_via_cv_scored)

    random.seed(999)
    np.random.seed(999)
    py_expected = random.random()
    np_expected = float(np.random.random())

    random.seed(999)
    np.random.seed(999)

    X_fs = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=float)
    y_fs = np.asarray([0, 1, 0], dtype=int)
    X_train = np.asarray([[0.2, 0.1], [0.7, 0.8], [0.9, 1.0]], dtype=float)
    X_test = np.asarray([[0.4, 0.4], [0.6, 0.7]], dtype=float)
    y_train = np.asarray([0, 1, 0], dtype=int)

    out_a = pipe._evaluate_selector_candidate(
        X_fs=X_fs,
        y_fs=y_fs,
        X_train_full=X_train,
        X_test_full=X_test,
        y_train_full=y_train,
        seed=101,
        enabled_methods=("mutual_information", "anova_f"),
        candidate_name="a",
    )
    out_b = pipe._evaluate_selector_candidate(
        X_fs=X_fs,
        y_fs=y_fs,
        X_train_full=X_train,
        X_test_full=X_test,
        y_train_full=y_train,
        seed=202,
        enabled_methods=("mutual_information", "anova_f"),
        candidate_name="b",
    )

    py_after = random.random()
    np_after = float(np.random.random())
    assert py_after == py_expected
    assert np_after == np_expected
    assert len(set(seen["selector_seed"])) == 2
    assert len(set(seen["model_seed"])) == 2
    assert out_a["seed_schedule"]["root_seed"] == 101
    assert out_b["seed_schedule"]["root_seed"] == 202


def test_defaults_enable_multimodal_gmm_fallback_and_dip():
    cfg = DFFSConfig()
    assert str(cfg.multimodal_fallback) == "gmm"
    assert bool(cfg.dist_config.compute_dip) is True

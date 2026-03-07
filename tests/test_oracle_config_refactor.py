import pytest

from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.config import (
    FeatureSelectorConfig,
    MNPOConfig,
    OracleConfig,
)


def test_oracle_config_sanitizes_invalid_modes_and_bounds():
    cfg = OracleConfig(
        diversity_mode="invalid",
        performance_oracle_mode="bad",
        weighting_mode="bad",
        fold_preference_mode="bad",
        conformal_efficiency_method="bad",
        cvar_alpha=3.0,
        shapley_n_coalitions_max=1,
        shapley_bayesian_prior_strength=-2.0,
        ubayfs_n_bootstrap=0,
        ubayfs_min_n=0,
        ubayfs_prior_weight=-2.0,
        conformal_uq_alpha=2.0,
        conformal_uq_min_folds=0,
    )
    assert cfg.diversity_mode == "legacy_jaccard"
    assert cfg.performance_oracle_mode == "single"
    assert cfg.weighting_mode == "tritrust"
    assert cfg.fold_preference_mode == "vote"
    assert cfg.conformal_efficiency_method == "split"
    assert cfg.cvar_alpha == pytest.approx(1.0)
    assert cfg.shapley_n_coalitions_max == 2
    assert cfg.shapley_bayesian_prior_strength == pytest.approx(1e-6)
    assert cfg.ubayfs_n_bootstrap == 1
    assert cfg.ubayfs_min_n == 1
    assert cfg.ubayfs_prior_weight == pytest.approx(0.0)
    assert cfg.conformal_uq_alpha == pytest.approx(0.49)
    assert cfg.conformal_uq_min_folds == 2


def test_mnpo_config_has_nested_oracle_by_default():
    cfg = MNPOConfig()
    assert isinstance(cfg.oracle, OracleConfig)
    assert cfg.oracle.weighting_mode == "tritrust"


def test_mnpo_legacy_aliases_warn_and_override_nested_oracle():
    with pytest.warns(DeprecationWarning):
        cfg = MNPOConfig(
            oracle=OracleConfig(
                use_cvar=False,
                cvar_alpha=0.33,
                weighting_mode="uniform",
                fold_preference_mode="vote",
                use_ubayfs=False,
                ubayfs_n_bootstrap=8,
                ubayfs_min_n=150,
                ubayfs_prior_weight=0.0,
                use_conformal_efficiency=False,
                conformal_efficiency_method="split",
                oracle_weight_js_shrinkage=False,
            ),
            use_cvar=True,
            cvar_alpha=0.20,
            oracle_weighting_mode="shapley",
            fold_preference_mode="logistic",
            use_ubayfs=True,
            ubayfs_n_bootstrap=17,
            ubayfs_min_n=123,
            ubayfs_prior_weight=0.4,
            diversity_oracle_mode="complementarity",
            use_conformal_efficiency=True,
            conformal_efficiency_method="aps",
            oracle_weight_js_shrinkage=True,
            payoff_shrinkage_kappa=0.15,
        )
    assert cfg.oracle.use_cvar is True
    assert cfg.oracle.cvar_alpha == pytest.approx(0.20)
    assert cfg.oracle.weighting_mode == "shapley"
    assert cfg.oracle.fold_preference_mode == "logistic"
    assert cfg.oracle.use_ubayfs is True
    assert cfg.oracle.ubayfs_n_bootstrap == 17
    assert cfg.oracle.ubayfs_min_n == 123
    assert cfg.oracle.ubayfs_prior_weight == pytest.approx(0.4)
    assert cfg.oracle.diversity_mode == "complementarity"
    assert cfg.oracle.use_conformal_efficiency is True
    assert cfg.oracle.conformal_efficiency_method == "aps"
    assert cfg.oracle.oracle_weight_js_shrinkage is True
    assert cfg.payoff_shrinkage_kappa == pytest.approx(0.15)


def test_from_config_prefers_nested_oracle_values():
    cfg = FeatureSelectorConfig(
        mnpo=MNPOConfig(
            oracle=OracleConfig(
                use_cvar=True,
                cvar_alpha=0.25,
                weighting_mode="uniform",
                fold_preference_mode="logistic",
                shapley_bayesian_shrinkage=True,
                shapley_bayesian_prior_strength=4.0,
                use_ubayfs=True,
                ubayfs_n_bootstrap=11,
                ubayfs_min_n=88,
                ubayfs_prior_weight=0.2,
                use_conformal_uq=True,
                conformal_uq_alpha=0.12,
                conformal_uq_min_folds=6,
                use_conformal_efficiency=True,
                conformal_efficiency_method="aps",
                oracle_weight_js_shrinkage=True,
                diversity_mode="complementarity",
            ),
            payoff_shrinkage_kappa=0.2,
        ),
    )
    fs = FeatureSelector.from_config(cfg)
    assert fs.use_cvar is True
    assert fs.cvar_alpha == pytest.approx(0.25)
    assert fs.oracle_weighting_mode == "uniform"
    assert fs.fold_preference_mode == "logistic"
    assert fs.shapley_bayesian_shrinkage is True
    assert fs.shapley_bayesian_prior_strength == pytest.approx(4.0)
    assert fs.use_ubayfs is True
    assert fs.ubayfs_n_bootstrap == 11
    assert fs.ubayfs_min_n == 88
    assert fs.ubayfs_prior_weight == pytest.approx(0.2)
    assert fs.use_conformal_uq is True
    assert fs.conformal_uq_alpha == pytest.approx(0.12)
    assert fs.conformal_uq_min_folds == 6
    assert fs.use_conformal_efficiency is True
    assert fs.conformal_efficiency_method == "aps"
    assert fs.oracle_weight_js_shrinkage is True
    assert fs.payoff_shrinkage_kappa == pytest.approx(0.2)
    assert fs.diversity_oracle_mode == "complementarity"


def test_from_config_flat_aliases_remain_backward_compatible():
    with pytest.warns(DeprecationWarning):
        cfg = FeatureSelectorConfig(
            mnpo=MNPOConfig(
                use_cvar=True,
                cvar_alpha=0.31,
                oracle_weighting_mode="shapley",
                use_ubayfs=True,
                ubayfs_n_bootstrap=9,
                ubayfs_min_n=77,
                ubayfs_prior_weight=0.3,
                diversity_oracle_mode="complementarity",
            ),
        )
    fs = FeatureSelector.from_config(cfg)
    assert fs.use_cvar is True
    assert fs.cvar_alpha == pytest.approx(0.31)
    assert fs.oracle_weighting_mode == "shapley"
    assert fs.use_ubayfs is True
    assert fs.ubayfs_n_bootstrap == 9
    assert fs.ubayfs_min_n == 77
    assert fs.ubayfs_prior_weight == pytest.approx(0.3)
    assert fs.diversity_oracle_mode == "complementarity"


def test_prefilter_wsnr_flag_wires_from_config():
    cfg = FeatureSelectorConfig()
    cfg.prefilter.union_enabled = True
    cfg.prefilter.strategies = ("mi_ftest_blend",)
    cfg.prefilter.wsnr_enabled = True
    fs = FeatureSelector.from_config(cfg)
    assert fs.prefilter_wsnr_enabled is True
    assert "wsnr" in set(fs.prefilter_strategies)

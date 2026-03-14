"""Tests for Phase 3: Configuration dataclasses and ``from_config()`` factory.

Validates that:
1. All config dataclasses instantiate with defaults.
2. ``FeatureSelector.from_config(FeatureSelectorConfig())`` produces an
   object whose mapped attributes match those from the plain constructor.
3. Non-default overrides via config propagate correctly through ``from_config()``.
4. Sub-configs can be constructed and modified independently.
"""

import pytest
from dataclasses import fields

from tabnetics.feature_selection.config import (
    CopulaConfig,
    FeatureSelectorConfig,
    MethodConfig,
    MNPOConfig,
    MulticlassConfig,
    StabilityConfig,
    WrapperConfig,
)
from tabnetics.feature_selection import FeatureSelector


# ── helpers ───────────────────────────────────────────────────────────

def _mapped_attrs(config: FeatureSelectorConfig):
    """Return the set of ``FeatureSelector`` attribute names that
    ``from_config()`` explicitly maps (derived from the config fields)."""
    # Top-level flat fields
    attrs = {f.name for f in fields(config) if not hasattr(f.default_factory if f.default_factory is not type else None, '__call__')  # skip sub-configs
             and f.name not in ('mnpo', 'stability', 'wrapper', 'multiclass', 'copula', 'methods')}
    # Sub-config fields  (attribute names match between config and FeatureSelector)
    for sub_name in ('mnpo', 'stability', 'wrapper', 'multiclass', 'copula', 'methods'):
        sub = getattr(config, sub_name)
        for f in fields(sub):
            attrs.add(f.name)
    return attrs


# ── 1. Defaults instantiate cleanly ──────────────────────────────────

class TestConfigDefaults:
    """All config dataclasses must be default-constructible."""

    def test_mnpo_config_defaults(self):
        cfg = MNPOConfig()
        assert cfg.portfolio_size == 6
        assert cfg.use_tritrust is True

    def test_mnpo_deprecated_tail_risk_toggles_warn(self):
        with pytest.warns(DeprecationWarning):
            _ = MNPOConfig(use_tail_risk_oracle=True, tail_risk_alpha=0.20)

    def test_stability_config_defaults(self):
        cfg = StabilityConfig()
        assert cfg.stability_subsample_fraction == 0.5
        assert cfg.ipss_importance_model == "linear_svm"
        assert cfg.stability_target_pfer == pytest.approx(1.0)

    def test_wrapper_config_defaults(self):
        cfg = WrapperConfig()
        assert cfg.wrapper_refine_enabled is False
        assert cfg.iterative_pruning_max_rounds == 32

    def test_multiclass_config_defaults(self):
        cfg = MulticlassConfig()
        assert cfg.nsc_thresholding_mode == "soft"
        assert cfg.ecoc_min_classes == 4

    def test_copula_config_defaults(self):
        cfg = CopulaConfig()
        assert cfg.copula_knockoff_draws == 30
        assert cfg.copula_truncation_level == 5

    def test_method_config_defaults(self):
        cfg = MethodConfig()
        assert cfg.mrmr_max_features == 320

    def test_top_level_config_defaults(self):
        cfg = FeatureSelectorConfig()
        assert cfg.n_folds == 5
        assert cfg.selection_strategy == "mnpo_portfolio"
        assert cfg.enabled_methods is None
        assert isinstance(cfg.mnpo, MNPOConfig)
        assert isinstance(cfg.stability, StabilityConfig)
        assert isinstance(cfg.wrapper, WrapperConfig)
        assert isinstance(cfg.multiclass, MulticlassConfig)
        assert isinstance(cfg.copula, CopulaConfig)
        assert isinstance(cfg.methods, MethodConfig)


# ── 2. from_config with all defaults matches plain constructor ───────

class TestFromConfigDefaults:
    """``FeatureSelector.from_config(FeatureSelectorConfig())`` should
    produce attribute values identical to ``FeatureSelector()``."""

    def test_from_config_matches_plain_init(self):
        fs_plain = FeatureSelector()
        cfg = FeatureSelectorConfig()
        fs_cfg = FeatureSelector.from_config(cfg)

        mapped = _mapped_attrs(cfg)
        mismatches = []
        for attr in sorted(mapped):
            v_plain = getattr(fs_plain, attr, '<MISSING>')
            v_cfg = getattr(fs_cfg, attr, '<MISSING>')
            if v_plain != v_cfg:
                mismatches.append(f"  {attr}: plain={v_plain!r}  config={v_cfg!r}")
        assert not mismatches, (
            "Attribute mismatches between plain init and from_config:\n"
            + "\n".join(mismatches)
        )

    def test_from_config_returns_feature_selector(self):
        fs = FeatureSelector.from_config(FeatureSelectorConfig())
        assert isinstance(fs, FeatureSelector)


# ── 3. Non-default overrides propagate correctly ─────────────────────

class TestFromConfigOverrides:
    """Verify that custom values propagate through ``from_config()``."""

    def test_override_general(self):
        cfg = FeatureSelectorConfig(
            random_state=0,
            n_folds=10,
            problem_type="regression",
            selection_strategy="legacy_voting",
        )
        fs = FeatureSelector.from_config(cfg)
        assert fs.random_state == 0
        assert fs.n_folds == 10
        assert fs.problem_type == "regression"
        assert fs.selection_strategy == "legacy_voting"

    def test_override_mnpo(self):
        cfg = FeatureSelectorConfig(
            mnpo=MNPOConfig(
                portfolio_size=10,
                use_tail_risk_oracle=True,
                diversity_oracle_mode="pid_mi",
                rashomon_enabled=True,
                rashomon_max_models=14,
                rashomon_score_tolerance=0.02,
            ),
        )
        fs = FeatureSelector.from_config(cfg)
        assert fs.portfolio_size == 10
        assert fs.use_tail_risk_oracle is True
        assert fs.diversity_oracle_mode == "pid_mi"
        assert fs.rashomon_enabled is True
        assert fs.rashomon_max_models == 14
        assert fs.rashomon_score_tolerance == pytest.approx(0.02)

    def test_override_stability(self):
        cfg = FeatureSelectorConfig(
            stability=StabilityConfig(
                stability_subsample_fraction=0.7,
                ipss_target_fdr=0.05,
                stability_threshold_method="cpss",
                stability_target_pfer=0.75,
            ),
        )
        fs = FeatureSelector.from_config(cfg)
        assert fs.stability_subsample_fraction == 0.7
        assert fs.ipss_target_fdr == 0.05
        assert fs.stability_threshold_method == "cpss"
        assert fs.stability_target_pfer == pytest.approx(0.75)

    def test_override_wrapper(self):
        cfg = FeatureSelectorConfig(
            wrapper=WrapperConfig(
                wrapper_refine_enabled=True,
                iterative_pruning_bounded_use_cpss_overlay=True,
            ),
        )
        fs = FeatureSelector.from_config(cfg)
        assert fs.wrapper_refine_enabled is True
        assert fs.iterative_pruning_bounded_use_cpss_overlay is True

    def test_override_copula(self):
        cfg = FeatureSelectorConfig(
            copula=CopulaConfig(
                copula_knockoff_draws=50,
                copula_alpha_kn=0.05,
                copula_generator="deepdrk",
                copula_deepdrk_latent_fraction=0.4,
                copula_deepdrk_noise_scale=1.1,
            ),
        )
        fs = FeatureSelector.from_config(cfg)
        assert fs.copula_knockoff_draws == 50
        assert fs.copula_alpha_kn == 0.05
        assert fs.copula_generator == "deepdrk"
        assert fs.copula_deepdrk_latent_fraction == pytest.approx(0.4)
        assert fs.copula_deepdrk_noise_scale == pytest.approx(1.1)

    def test_override_multiclass(self):
        cfg = FeatureSelectorConfig(
            multiclass=MulticlassConfig(nsc_thresholding_mode="hard"),
        )
        fs = FeatureSelector.from_config(cfg)
        assert fs.nsc_thresholding_mode == "hard"

    def test_override_methods(self):
        cfg = FeatureSelectorConfig(
            methods=MethodConfig(mrmr_max_features=128, ktsp_k_pairs=12),
        )
        fs = FeatureSelector.from_config(cfg)
        assert fs.mrmr_max_features == 128
        assert fs.ktsp_k_pairs == 12

    def test_enabled_methods_set(self):
        cfg = FeatureSelectorConfig(
            enabled_methods={"stability_selection", "gradient_boosting"},
        )
        fs = FeatureSelector.from_config(cfg)
        assert fs.enabled_methods == {"stability_selection", "gradient_boosting"}

    def test_adaptive_sizing_bounds_accept_valid(self):
        cfg = FeatureSelectorConfig(
            mnpo=MNPOConfig(
                portfolio_size=8,
                adaptive_portfolio_sizing_enabled=True,
                adaptive_size_min=6,
                adaptive_size_max=10,
            )
        )
        fs = FeatureSelector.from_config(cfg)
        assert fs.adaptive_portfolio_sizing_enabled is True
        assert fs.adaptive_size_min == 6
        assert fs.adaptive_size_max == 10

    def test_adaptive_sizing_bounds_reject_missing(self):
        cfg = FeatureSelectorConfig(
            mnpo=MNPOConfig(
                portfolio_size=8,
                adaptive_portfolio_sizing_enabled=True,
                adaptive_size_min=None,
                adaptive_size_max=10,
            )
        )
        with pytest.raises(ValueError, match="requires both"):
            FeatureSelector.from_config(cfg)

    def test_adaptive_sizing_bounds_reject_out_of_range_reference(self):
        cfg = FeatureSelectorConfig(
            mnpo=MNPOConfig(
                portfolio_size=12,
                adaptive_portfolio_sizing_enabled=True,
                adaptive_size_min=6,
                adaptive_size_max=10,
            )
        )
        with pytest.raises(ValueError, match="portfolio_size must lie within"):
            FeatureSelector.from_config(cfg)


# ── 4. Sub-config isolation ──────────────────────────────────────────

class TestSubConfigIsolation:
    """Modifying one sub-config must not affect others."""

    def test_independent_subconfigs(self):
        cfg = FeatureSelectorConfig()
        cfg.mnpo.portfolio_size = 12
        assert cfg.stability.stability_subsample_fraction == 0.5  # untouched

    def test_two_configs_independent(self):
        c1 = FeatureSelectorConfig()
        c2 = FeatureSelectorConfig()
        c1.mnpo.portfolio_size = 99
        assert c2.mnpo.portfolio_size == 6  # default intact

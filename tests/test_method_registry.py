"""Unit tests for the METHOD_REGISTRY canonical source of truth."""
import pytest
from tabnetics.feature_selection.registry import (
    METHOD_REGISTRY,
    MethodSpec,
    get_method_weights,
    get_experimental_keys,
)


class TestMethodRegistry:
    """Structural invariants that must hold for the registry."""

    def test_registry_is_non_empty(self):
        assert len(METHOD_REGISTRY) > 0

    def test_all_values_are_method_specs(self):
        for key, spec in METHOD_REGISTRY.items():
            assert isinstance(spec, MethodSpec), f"{key} is not a MethodSpec"

    def test_keys_match_spec_key_field(self):
        for key, spec in METHOD_REGISTRY.items():
            assert key == spec.key, f"dict key {key!r} != spec.key {spec.key!r}"

    def test_no_duplicate_fn_names(self):
        fn_names = [s.fn_name for s in METHOD_REGISTRY.values()]
        assert len(fn_names) == len(set(fn_names)), (
            f"Duplicate fn_names: {[f for f in fn_names if fn_names.count(f) > 1]}"
        )

    def test_maturity_values_valid(self):
        valid = {'experimental', 'stable', 'deprecated'}
        for key, spec in METHOD_REGISTRY.items():
            assert spec.maturity in valid, f"{key}: invalid maturity {spec.maturity!r}"

    def test_paradigm_values_valid(self):
        valid = {'filter', 'stability', 'embedded', 'wrapper', 'multiclass', 'pairwise', 'knockoff'}
        for key, spec in METHOD_REGISTRY.items():
            assert spec.paradigm in valid, f"{key}: invalid paradigm {spec.paradigm!r}"

    def test_legacy_weights_positive(self):
        for key, spec in METHOD_REGISTRY.items():
            assert spec.legacy_weight > 0, f"{key}: weight must be positive"

    def test_ga_svm_rfe_excluded(self):
        assert 'ga_svm_rfe' not in METHOD_REGISTRY

    def test_no_space_in_keys(self):
        """Catches the old 'k tsp' typo."""
        for key in METHOD_REGISTRY:
            assert ' ' not in key, f"Key {key!r} contains a space"

    def test_get_method_weights_matches_registry(self):
        weights = get_method_weights()
        assert len(weights) == len(METHOD_REGISTRY)
        for key, w in weights.items():
            assert w == METHOD_REGISTRY[key].legacy_weight

    def test_get_experimental_keys_matches_registry(self):
        exp = get_experimental_keys()
        expected = {k for k, v in METHOD_REGISTRY.items() if v.maturity == 'experimental'}
        assert exp == expected

    def test_known_method_count(self):
        """38 methods after adding SIR/SAVE/PFC SDR selectors."""
        assert len(METHOD_REGISTRY) == 38

    # --- Spot-check specific methods ---

    def test_boruta_weight(self):
        assert METHOD_REGISTRY['boruta'].legacy_weight == 2.0

    def test_copula_knockoff_weight(self):
        assert METHOD_REGISTRY['copula_knockoff'].legacy_weight == 4.0

    def test_ktsp_key_no_space(self):
        assert 'ktsp' in METHOD_REGISTRY
        assert METHOD_REGISTRY['ktsp'].legacy_weight == 1.1

    def test_multiclass_methods_require_multiclass(self):
        mc_keys = {
            'ova_ensemble', 'ecoc_class_aware', 'joint_multiclass_support',
            'dove_class_specific', 'sparse_multinomial',
            'nearest_shrunken_centroid', 'class_pareto_front',
            'sir_sdr', 'save_sdr', 'pfc_sdr',
        }
        for key in mc_keys:
            assert METHOD_REGISTRY[key].requires_multiclass, f"{key} should require multiclass"

    def test_binary_only_methods(self):
        assert METHOD_REGISTRY["joint_auc_l1"].binary_only, "joint_auc_l1 should remain binary_only"
        assert not METHOD_REGISTRY["wmw_auc"].binary_only, "wmw_auc should support multiclass dispatch"

    def test_experimental_methods_match_old_set(self):
        """The old hardcoded experimental_methods set, canonicalised + new filters."""
        old_set = {
            'mrmr_jmi', 'ktsp', 'ova_ensemble', 'ecoc_class_aware',
            'joint_multiclass_support', 'dove_class_specific',
            'sparse_multinomial', 'class_pareto_front', 'hsic_lasso',
            'slce_centroid_encoder', 'sir_sdr', 'save_sdr', 'pfc_sdr',
            'iterative_redundancy_pruning', 'iterative_redundancy_pruning_bounded',
            'stability_subsample', 'tigress_stability', 'subspace_stability',
            'ipss', 'cluster_stability',
            'chi_square', 'relieff', 'fcbf', 'cmim',
            'oaenet', 'treeshap',
        }
        assert get_experimental_keys() == old_set


# ---------------------------------------------------------------------------
# T-P3-INFRA-002: requires_gpu field + dispatcher guard
# ---------------------------------------------------------------------------

class TestRequiresGpu:
    """Tests for the requires_gpu MethodSpec field and dispatcher guard."""

    def test_requires_gpu_field_default_false(self):
        """All current MethodSpec entries should have requires_gpu=False."""
        for key, spec in METHOD_REGISTRY.items():
            assert spec.requires_gpu is False, (
                f"MethodSpec '{key}' unexpectedly has requires_gpu=True"
            )

    def test_requires_gpu_field_exists(self):
        """MethodSpec dataclass has a requires_gpu field defaulting to False."""
        spec = MethodSpec(key="dummy", label="Dummy", fn_name="_dummy")
        assert hasattr(spec, "requires_gpu")
        assert spec.requires_gpu is False

    def test_gpu_spec_skipped_when_no_gpu(self):
        """A method with requires_gpu=True is skipped when GPU unavailable."""
        import numpy as np
        from tabnetics.feature_selection.base import FeatureSelector
        import tabnetics.feature_selection.registry as reg_mod

        fs = FeatureSelector(problem_type="classification")
        # Force _gpu_available to False
        fs.__dict__["_gpu_available"] = False

        fake_spec = MethodSpec(
            key="fake_gpu_method",
            label="Fake GPU Method",
            fn_name="_fake_gpu_fn",
            requires_gpu=True,
        )

        original_registry = dict(reg_mod.METHOD_REGISTRY)
        reg_mod.METHOD_REGISTRY["fake_gpu_method"] = fake_spec
        try:
            X = np.random.RandomState(0).randn(20, 5)
            y = np.array([0] * 10 + [1] * 10)
            results, runtimes = fs._run_selection_methods(X, y, n_target=3)
            assert "fake_gpu_method" not in results
        finally:
            reg_mod.METHOD_REGISTRY.clear()
            reg_mod.METHOD_REGISTRY.update(original_registry)

    def test_gpu_spec_not_filtered_when_gpu_available(self):
        """Method with requires_gpu=True passes GPU gate when GPU available."""
        import numpy as np
        from tabnetics.feature_selection.base import FeatureSelector
        import tabnetics.feature_selection.registry as reg_mod

        fs = FeatureSelector(problem_type="classification")
        fs.__dict__["_gpu_available"] = True

        fake_spec = MethodSpec(
            key="fake_gpu_method2",
            label="Fake GPU Method 2",
            fn_name="_nonexistent_fn",  # skipped by getattr, not GPU gate
            requires_gpu=True,
        )

        original_registry = dict(reg_mod.METHOD_REGISTRY)
        reg_mod.METHOD_REGISTRY["fake_gpu_method2"] = fake_spec
        try:
            X = np.random.RandomState(0).randn(20, 5)
            y = np.array([0] * 10 + [1] * 10)
            # Should not crash
            results, runtimes = fs._run_selection_methods(X, y, n_target=3)
        finally:
            reg_mod.METHOD_REGISTRY.clear()
            reg_mod.METHOD_REGISTRY.update(original_registry)

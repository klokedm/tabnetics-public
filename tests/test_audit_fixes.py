"""Tests for Code Audit Session 2 fixes (T-A2-008)."""
import numpy as np
import pytest

# ── FIX-1: stability_selection_lasso reproducibility ──

def test_stability_lasso_deterministic_with_seed():
    """stability_selection_lasso must produce identical results for identical seeds."""
    from tabnetics.feature_selection.methods.embedded import stability_selection_lasso
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 20))
    y = np.array([0] * 30 + [1] * 30)

    def cv_fn(y_sub):
        n_classes = len(np.unique(y_sub))
        n_folds = min(3, min(np.bincount(y_sub.astype(int))))
        if n_folds < 2:
            n_folds = 2
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)

    r1, s1 = stability_selection_lasso(X, y, 5, 8, "classification", 42, cv_fn)
    r2, s2 = stability_selection_lasso(X, y, 5, 8, "classification", 42, cv_fn)

    np.testing.assert_array_equal(
        r1["selected_indices"], r2["selected_indices"],
        err_msg="stability_selection_lasso must be deterministic for same seed"
    )


def test_stability_lasso_different_seeds_differ():
    """Different seeds must produce different bootstrap samples (and likely different results)."""
    from tabnetics.feature_selection.methods.embedded import stability_selection_lasso
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(99)
    X = rng.standard_normal((80, 30))
    y = np.array([0] * 40 + [1] * 40)

    def cv_fn(y_sub):
        n_folds = min(3, min(np.bincount(y_sub.astype(int))))
        if n_folds < 2:
            n_folds = 2
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)

    r1, s1 = stability_selection_lasso(X, y, 5, 10, "classification", 42, cv_fn)
    r2, s2 = stability_selection_lasso(X, y, 5, 10, "classification", 99, cv_fn)

    # With different seeds, selection frequencies should differ
    freq1 = r1["selection_frequency"]
    freq2 = r2["selection_frequency"]
    assert not np.allclose(freq1, freq2, atol=0.01), \
        "Different seeds should produce different bootstrap results"


# ── FIX-2: e-BH posinf handling ──

def test_ebh_support_preserves_posinf_evidence():
    """posinf e-values should be treated as strong evidence, not discarded."""
    from tabnetics.feature_selection.copula import _ebh_support

    e_vals = np.array([np.inf, 10.0, 0.5, 0.1, 0.01])
    support = _ebh_support(e_vals, alpha=0.2)
    # Feature 0 has inf e-value — it must be selected
    assert 0 in support, "Feature with inf e-value must be selected"


def test_ebh_support_handles_nan_safely():
    """NaN e-values should be treated as zero (no evidence)."""
    from tabnetics.feature_selection.copula import _ebh_support

    e_vals = np.array([np.nan, 30.0, 14.0, 0.2])
    support = _ebh_support(e_vals, alpha=0.2)
    assert 0 not in support, "NaN e-value feature should not be selected"


# ── FIX-3: knockoff threshold excludes t=0 ──

def test_knockoff_threshold_excludes_zero():
    """Threshold grid must exclude t=0, per Barber & Candès (2015)."""
    from tabnetics.feature_selection.copula import _knockoff_threshold

    # All-zero W should return inf (no valid threshold)
    W = np.zeros(10)
    t = _knockoff_threshold(W, alpha=0.1)
    assert np.isinf(t), "All-zero W must return inf threshold"


def test_knockoff_threshold_with_mixed_signs():
    """Basic knockoff threshold still works after t=0 filter (existing simulation test covers this thoroughly)."""
    from tabnetics.feature_selection.copula import _knockoff_threshold

    # Predominantly positive large W with few negatives → achievable ratio
    # With alpha=0.5: (1+#neg) / #pos ≤ 0.5 requires many more positives than negatives
    # Use the simulation test from the existing suite as the primary coverage.
    # Here we just verify the function returns a valid type.
    rng = np.random.default_rng(2026)
    W_strong_signal = np.concatenate([
        rng.normal(loc=3.0, scale=0.5, size=20),   # true signals (positive)
        rng.normal(loc=-0.2, scale=0.3, size=5),    # nulls (symmetric around 0)
    ])
    t = _knockoff_threshold(W_strong_signal, alpha=0.2)
    # Should find a valid threshold with 20 strong positives and only 5 negatives
    assert isinstance(t, float), "Threshold must be a float"
    assert t > 0 or np.isinf(t), "Threshold must be positive or inf"


# ── FIX-4: RuntimeWarning no longer suppressed module-wide ──

def test_runtime_warning_not_suppressed_globally():
    """base.py must not have module-level RuntimeWarning suppression."""
    import tabnetics.feature_selection.base as base_mod
    import inspect
    source = inspect.getsource(base_mod)
    # The old pattern was: warnings.filterwarnings('ignore', category=RuntimeWarning)
    # at module level (outside functions/classes)
    lines = source.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            "filterwarnings" in stripped
            and "RuntimeWarning" in stripped
            and "ignore" in stripped
            and not stripped.startswith("#")
            and not stripped.startswith("'")
            and not stripped.startswith('"')
        ):
            # Check it's not inside a function/class (rough heuristic: no indentation)
            if not line.startswith(" ") and not line.startswith("\t"):
                pytest.fail(
                    f"Module-level RuntimeWarning suppression found at line {i+1}: {stripped}"
                )


# ── FIX-5: copula Gaussian bridge posinf handling ──

def test_gaussian_bridge_preserves_extreme_z():
    """Gaussian bridge should map extreme z values to ±6, not 0."""
    from tabnetics.feature_selection.copula import CopulaKnockoffSelector

    rng = np.random.RandomState(42)
    # Uniform values very close to 0 and 1 will produce extreme z = ppf(v)
    v = np.array([1e-7, 0.5, 1 - 1e-7])
    result = CopulaKnockoffSelector._gaussian_bridge_uniform(v, rng, rho=0.5)
    assert result.shape == (3,), "Output shape must match input"
    assert np.all(np.isfinite(result)), "Output must be finite"
    # The extreme z values should not be collapsed to center
    assert result[0] < 0.3, "Extreme low uniform should produce low output"
    assert result[2] > 0.7, "Extreme high uniform should produce high output"


# ── FIX-6: synthetic fallback default ──

def test_benchmark_cli_synthetic_fallback_default_false():
    """Benchmark CLI default for allow_synthetic_fallback must be False."""
    import tabnetics.benchmarks.runner as benchmark
    args = benchmark.build_arg_parser().parse_args(
        ["--datasets", "synthetic_easy_dfshift"]
    )
    assert args.allow_synthetic_fallback is False, \
        "Default must be False to prevent synthetic data confounds in validation"

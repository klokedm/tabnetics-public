"""Regression coverage for bounded, directional-cache mRMR selection."""

from __future__ import annotations

import numpy as np
import pytest

import tabnetics.feature_selection.methods.filter as filter_mod
from tabnetics.benchmarks import runner as benchmark_runner
from tabnetics.feature_selection.base import FeatureSelector
from tabnetics.feature_selection.config import FeatureSelectorConfig, MethodConfig
from tabnetics.feature_selection.mnpo.portfolio import selector_result_eligibility
from tabnetics.pipeline.pipeline import (
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
    IncompleteFeatureSelectionError,
)


def _identity_prefilter(X, y, max_features):
    del y
    return np.arange(min(int(np.asarray(X).shape[1]), int(max_features)), dtype=int)


def _variance_relevance(X, y, random_state=None):
    del y, random_state
    return np.nan_to_num(np.var(np.asarray(X, dtype=float), axis=0), nan=0.0)


def _ordered_relevance(X, y, random_state=None):
    del y, random_state
    return np.arange(np.asarray(X).shape[1], 0, -1, dtype=float)


def _normalize(scores):
    values = np.asarray(scores, dtype=float).ravel()
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _dataset(n_samples=64, n_features=12, seed=13):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, size=n_samples)
    X = rng.normal(size=(n_samples, n_features))
    X[:, 0] += 1.5 * y
    X[:, 1] = 0.7 * X[:, 0] + rng.normal(scale=0.2, size=n_samples)
    return X, y


def _base_kwargs(**overrides):
    values = {
        "random_state": 17,
        "mi_scorer": _variance_relevance,
        "prefilter_fn": _identity_prefilter,
        "normalize_fn": _normalize,
        "mrmr_max_features": 64,
        "mrmr_redundancy_weight": 0.55,
        "mrmr_mi_redundancy_enabled": False,
        "mrmr_mi_n_bins": 8,
        "mrmr_max_unique_pair_evaluations": 0,
        "mrmr_max_runtime_seconds": 0.0,
        "mrmr_budget_fallback_mode": "empty",
    }
    values.update(overrides)
    return values


def _scores_vector(scores, n_features):
    return np.asarray([float(scores[i]) for i in range(n_features)], dtype=float)


def _uncached_reference(X, y, n_target_features, *, use_pair_scorer=False, **kwargs):
    """Pre-cache greedy reference preserving the historical aggregation order."""
    X_arr = np.asarray(X)
    n_samples, n_features = X_arr.shape
    pool_idx = kwargs["prefilter_fn"](
        X,
        y,
        max_features=min(kwargs["mrmr_max_features"], n_features),
    )
    X_pool = X[:, pool_idx]
    relevance = np.asarray(
        kwargs["mi_scorer"](X_pool, y, random_state=kwargs["random_state"]),
        dtype=float,
    ).ravel()
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)
    relevance_norm = kwargs["normalize_fn"](relevance)
    target = int(min(max(1, n_target_features), X_pool.shape[1]))

    mode = "pearson"
    X_disc = None
    if kwargs["mrmr_mi_redundancy_enabled"]:
        if n_samples < 80:
            mode = "pearson"
        elif n_samples < 150:
            mode = "binned_mi"
        else:
            mode = "knn_mi"
        if mode == "binned_mi":
            X_disc = np.column_stack(
                [
                    filter_mod._safe_rank_bin(X_pool[:, idx], kwargs["mrmr_mi_n_bins"])
                    for idx in range(X_pool.shape[1])
                ]
            )
    X_disc_arr = np.asarray(X_disc) if X_disc is not None else np.zeros((0, 0), dtype=int)

    selected = []
    selected_set = set()
    criterion_scores = np.zeros(X_pool.shape[1], dtype=float)
    if X_pool.shape[1] <= 1:
        selected = [0]
        criterion_scores[0] = float(relevance_norm[0]) if relevance_norm.size else 0.0
    else:
        first = int(np.argmax(relevance_norm))
        selected.append(first)
        selected_set.add(first)
        criterion_scores[first] = relevance_norm[first]

        while len(selected) < target:
            best_idx = None
            best_score = -np.inf
            for idx in range(X_pool.shape[1]):
                if idx in selected_set:
                    continue
                if use_pair_scorer:
                    values = []
                    for selected_idx in selected:
                        try:
                            value = filter_mod._mrmr_pair_redundancy_score(
                                X_pool,
                                X_disc_arr,
                                idx,
                                selected_idx,
                                mode=mode,
                                random_state=kwargs["random_state"],
                            )
                        except Exception:
                            continue
                        if np.isfinite(value):
                            values.append(float(value))
                    redundancy = float(np.mean(values)) if values else 0.0
                else:
                    redundancy = filter_mod._mi_redundancy_score(
                        X_pool,
                        X_disc_arr,
                        idx,
                        selected,
                        mode=mode,
                        random_state=kwargs["random_state"],
                    )
                score = float(
                    relevance_norm[idx] - kwargs["mrmr_redundancy_weight"] * redundancy
                )
                criterion_scores[idx] = score
                if score > best_score:
                    best_score = score
                    best_idx = idx
            assert best_idx is not None
            selected.append(int(best_idx))
            selected_set.add(int(best_idx))

    selected_indices = pool_idx[np.asarray(selected, dtype=int)][:n_target_features]
    all_scores = np.zeros(n_features, dtype=float)
    all_scores[pool_idx] = criterion_scores
    all_scores = np.nan_to_num(all_scores, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(np.min(all_scores)) if all_scores.size else 0.0
    if minimum < 0.0:
        all_scores = all_scores - minimum
    return selected_indices, all_scores


@pytest.mark.parametrize(
    ("n_samples", "mi_redundancy_enabled", "mode"),
    [
        (64, False, "pearson"),
        (100, True, "binned_mi"),
        (160, True, "knn_mi"),
    ],
)
def test_directional_cache_is_bit_identical_to_uncached_reference(
    n_samples, mi_redundancy_enabled, mode
):
    X, y = _dataset(n_samples=n_samples, n_features=11, seed=n_samples)
    kwargs = _base_kwargs(mrmr_mi_redundancy_enabled=mi_redundancy_enabled)
    expected_selected, expected_scores = _uncached_reference(X, y, 5, **kwargs)

    result, scores = filter_mod.mrmr_jmi_selection(X, y, 5, **kwargs)

    np.testing.assert_array_equal(result["selected_indices"], expected_selected)
    np.testing.assert_array_equal(_scores_vector(scores, X.shape[1]), expected_scores)
    assert result["complete"] is True
    assert result["redundancy_mode_effective"] == mode
    assert result["criterion_family"] == "mrmr_difference"
    assert result["budget_status"] == "unlimited"


def test_directional_cache_keeps_asymmetric_pair_scores_and_exact_work_counts(monkeypatch):
    X, y = _dataset(n_samples=64, n_features=7, seed=31)
    kwargs = _base_kwargs(mi_scorer=_ordered_relevance)
    calls = []

    def asymmetric_pair_score(X_pool, X_disc, idx, selected_idx, *, mode, random_state):
        del X_pool, X_disc, mode, random_state
        calls.append((int(idx), int(selected_idx)))
        return float(1000 * int(idx) + int(selected_idx))

    monkeypatch.setattr(filter_mod, "_mrmr_pair_redundancy_score", asymmetric_pair_score)
    expected_selected, expected_scores = _uncached_reference(
        X,
        y,
        4,
        use_pair_scorer=True,
        **kwargs,
    )
    calls.clear()

    result, scores = filter_mod.mrmr_jmi_selection(X, y, 4, **kwargs)
    expected_uncached, expected_unique = filter_mod._mrmr_work_estimates(7, 4)

    np.testing.assert_array_equal(result["selected_indices"], expected_selected)
    np.testing.assert_array_equal(_scores_vector(scores, X.shape[1]), expected_scores)
    assert len(calls) == expected_unique
    assert result["unique_pair_evaluations"] == expected_unique
    assert result["estimated_unique_pair_evaluations"] == expected_unique
    assert result["estimated_uncached_pair_evaluations"] == expected_uncached
    assert result["cache_hits"] == expected_uncached - expected_unique
    assert all((right, left) not in calls for left, right in calls)


def test_pair_evaluation_budget_fails_closed_without_partial_candidate():
    X, y = _dataset(n_samples=64, n_features=8, seed=41)
    kwargs = _base_kwargs(
        mi_scorer=_ordered_relevance,
        mrmr_max_unique_pair_evaluations=1,
    )

    result, scores = filter_mod.mrmr_jmi_selection(X, y, 4, **kwargs)

    assert result["complete"] is False
    assert result["incomplete"] is True
    assert result["budget_status"] == "exhausted"
    assert result["stop_reason"] == "unique_pair_evaluation_budget_exhausted"
    assert result["selected_indices"].size == 0
    assert result["partial_selected_count"] == 1
    assert result["unique_pair_evaluations"] == 1
    assert np.array_equal(_scores_vector(scores, X.shape[1]), np.zeros(X.shape[1]))


def test_pair_evaluation_budget_boundary_and_relevance_only_fallback():
    X, y = _dataset(n_samples=64, n_features=7, seed=43)
    _, expected_unique = filter_mod._mrmr_work_estimates(7, 4)
    complete_kwargs = _base_kwargs(
        mi_scorer=_ordered_relevance,
        mrmr_max_unique_pair_evaluations=expected_unique,
    )
    complete_result, _ = filter_mod.mrmr_jmi_selection(X, y, 4, **complete_kwargs)
    assert complete_result["complete"] is True
    assert complete_result["budget_status"] == "within_budget"
    assert complete_result["unique_pair_evaluations"] == expected_unique

    fallback_kwargs = _base_kwargs(
        mi_scorer=_ordered_relevance,
        mrmr_max_unique_pair_evaluations=1,
        mrmr_budget_fallback_mode="relevance_only",
    )
    fallback_result, _ = filter_mod.mrmr_jmi_selection(X, y, 4, **fallback_kwargs)
    assert fallback_result["complete"] is False
    assert fallback_result["fallback_applied"] is True
    assert fallback_result["selected_indices"].size == 4
    assert fallback_result["budget_fallback_mode_requested"] == "relevance_only"


@pytest.mark.parametrize("selection_strategy", ("legacy_voting", "mnpo_portfolio"))
def test_empty_budget_breached_mrmr_fails_closed_in_final_aggregation(
    selection_strategy,
):
    X, y = _dataset(n_samples=64, n_features=8, seed=41)
    selector = FeatureSelector(
        n_bootstrap_iterations=1,
        random_state=41,
        problem_type="classification",
        selection_strategy=selection_strategy,
        enabled_methods=("mrmr_jmi",),
        mrmr_max_unique_pair_evaluations=1,
        mrmr_budget_fallback_mode="empty",
        inner_cv_splits=2,
        inner_cv_repeats=1,
        mirror_descent_steps=5,
        portfolio_size=1,
    )

    X_selected, result = selector.fit_transform(
        X, y, n_final_features=4, return_result_object=True
    )

    status = result.config["selector_candidate_statuses"]["mrmr_jmi"]
    assert X_selected.shape == (X.shape[0], 0)
    assert result.selected_feature_indices.size == 0
    assert result.config["selection_aggregation_status"] == (
        "fail_closed_no_eligible_selector_evidence"
    )
    assert result.config["selection_aggregation_fail_closed"] is True
    assert status["status"] == "incomplete_excluded"
    assert status["complete"] is False
    assert status["incomplete"] is True
    assert status["fallback_applied"] is False
    assert status["budget_status"] == "exhausted"
    assert status["stop_reason"] == "unique_pair_evaluation_budget_exhausted"
    assert status["mnpo_candidate_eligible"] is False
    assert status["legacy_vote_eligible"] is False
    if selection_strategy == "mnpo_portfolio":
        assert "mnpo_portfolio" not in result.method_results
        assert selector.mnpo_diagnostics_["status"] == "no_eligible_mnpo_candidates"


@pytest.mark.parametrize(
    ("payload", "expected_status", "malformed_field"),
    [
        ({"budget_exhausted": True}, "incomplete_excluded", None),
        ({"budget_status": "exhausted"}, "incomplete_excluded", None),
        (
            {"stop_reason": "runtime_budget_exhausted"},
            "incomplete_excluded",
            None,
        ),
        ({"complete": "false"}, "malformed_execution_state", "complete"),
    ],
)
def test_selector_execution_state_fails_closed_without_truthiness_coercion(
    payload,
    expected_status,
    malformed_field,
):
    status = selector_result_eligibility(payload)

    assert status["status"] == expected_status
    assert status["complete"] is None
    assert status["incomplete"] is True
    assert status["mnpo_candidate_eligible"] is False
    assert status["mnpo_consensus_eligible"] is False
    assert status["legacy_vote_eligible"] is False
    assert status["fail_closed"] is True
    if malformed_field is None:
        assert status["execution_state_malformed"] is False
    else:
        assert status["execution_state_malformed"] is True
        assert status["execution_state_malformed_fields"] == [malformed_field]


def test_budget_state_without_complete_preserves_explicit_relevance_only_fallback():
    status = selector_result_eligibility(
        {
            "budget_status": "exhausted",
            "fallback_applied": True,
            "budget_fallback_mode_requested": "relevance_only",
        }
    )

    assert status["status"] == "relevance_only_fallback"
    assert status["complete"] is None
    assert status["incomplete"] is True
    assert status["mnpo_candidate_eligible"] is False
    assert status["mnpo_consensus_eligible"] is False
    assert status["legacy_vote_eligible"] is True
    assert status["fail_closed"] is False


def test_pipeline_raises_typed_error_before_stage2_for_fail_closed_mrmr(
    monkeypatch,
):
    X, y = _dataset(n_samples=64, n_features=8, seed=41)
    config = DFFSConfig(
        random_seed=41,
        fs_fraction=1.0,
        n_final_features=4,
        selection_strategy="legacy_voting",
        enabled_methods=("mrmr_jmi",),
        prefilter_top_k=8,
        apply_cdf_transform=False,
        fs_inner_cv_splits=2,
        fs_inner_cv_repeats=1,
        fs_mrmr_max_unique_pair_evaluations=1,
        fs_mrmr_budget_fallback_mode="empty",
    )
    stage2_called = False

    def _stage2_must_not_run(*args, **kwargs):
        nonlocal stage2_called
        del args, kwargs
        stage2_called = True
        raise AssertionError("Stage 2 must not run after a fail-closed selection.")

    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "_select_model_via_cv_scored",
        _stage2_must_not_run,
    )

    with pytest.raises(IncompleteFeatureSelectionError) as exc_info:
        DistributionFeatureSelectionPipeline(config).run_pre_split(
            X_train=X[:48],
            y_train=y[:48],
            X_test=X[48:],
            y_test=y[48:],
            dataset_name="mrmr_pipeline_boundary",
            seed=41,
        )

    error = exc_info.value
    diagnostics = dict(error.diagnostics)
    mrmr = diagnostics["mrmr_jmi_execution"]
    assert stage2_called is False
    assert error.code == "incomplete_feature_selection"
    assert diagnostics["candidate_name"] == "configured_enabled_methods"
    assert diagnostics["selection_aggregation_status"] == (
        "fail_closed_no_eligible_selector_evidence"
    )
    assert diagnostics["selection_aggregation_fail_closed"] is True
    assert diagnostics["selected_feature_count"] == 0
    assert mrmr["complete"] is False
    assert mrmr["incomplete"] is True
    assert mrmr["fallback_applied"] is False
    assert mrmr["budget_status"] == "exhausted"
    assert mrmr["budget_exhausted"] is True
    assert mrmr["stop_reason"] == "unique_pair_evaluation_budget_exhausted"


def test_relevance_only_mrmr_is_excluded_from_mnpo_candidates_and_consensus():
    X, y = _dataset(n_samples=64, n_features=8, seed=41)
    selector = FeatureSelector(
        n_bootstrap_iterations=1,
        random_state=41,
        problem_type="classification",
        selection_strategy="mnpo_portfolio",
        enabled_methods=("mrmr_jmi", "mutual_information"),
        mrmr_max_unique_pair_evaluations=1,
        mrmr_budget_fallback_mode="relevance_only",
        inner_cv_splits=2,
        inner_cv_repeats=1,
        mirror_descent_steps=5,
        portfolio_size=2,
    )

    _, result = selector.fit_transform(X, y, n_final_features=4, return_result_object=True)

    summary = result.method_results["mnpo_portfolio"]
    status = summary["selector_candidate_statuses"]["mrmr_jmi"]
    assert status["status"] == "relevance_only_fallback"
    assert status["complete"] is False
    assert status["incomplete"] is True
    assert status["fallback_applied"] is True
    assert status["budget_status"] == "exhausted"
    assert status["stop_reason"] == "unique_pair_evaluation_budget_exhausted"
    assert status["mnpo_candidate_eligible"] is False
    assert status["mnpo_consensus_eligible"] is False
    assert status["legacy_vote_eligible"] is True
    assert "mrmr_jmi" in summary["mnpo_degraded_excluded_methods"]
    assert "mrmr_jmi" not in summary["mnpo_consensus_source_methods"]
    assert "mrmr_jmi" not in selector.mnpo_diagnostics_["candidate_names"]


def test_relevance_only_mrmr_is_labeled_when_used_as_mnpo_fallback():
    X, y = _dataset(n_samples=64, n_features=8, seed=41)
    selector = FeatureSelector(
        n_bootstrap_iterations=1,
        random_state=41,
        problem_type="classification",
        selection_strategy="mnpo_portfolio",
        enabled_methods=("mrmr_jmi",),
        mrmr_max_unique_pair_evaluations=1,
        mrmr_budget_fallback_mode="relevance_only",
        inner_cv_splits=2,
        inner_cv_repeats=1,
        mirror_descent_steps=5,
        portfolio_size=1,
    )

    X_selected, result = selector.fit_transform(
        X, y, n_final_features=4, return_result_object=True
    )

    status = result.config["selector_candidate_statuses"]["mrmr_jmi"]
    mrmr_selected = result.method_results["mrmr_jmi"]["selected_indices"]
    assert X_selected.shape == (X.shape[0], 4)
    assert mrmr_selected.size == 4
    assert result.selected_feature_indices.size == 4
    assert all(value > 0.0 for value in result.selected_feature_votes.values())
    assert result.config["selection_aggregation_status"] == "mnpo_legacy_vote_fallback"
    assert result.config["selection_aggregation_fail_closed"] is False
    assert status["status"] == "relevance_only_fallback"
    assert status["fallback_applied"] is True
    assert status["mnpo_candidate_eligible"] is False
    assert status["legacy_vote_eligible"] is True
    assert selector.mnpo_diagnostics_["status"] == "no_eligible_mnpo_candidates"


def test_runtime_budget_boundary_is_explicit_and_deterministic(monkeypatch):
    X, y = _dataset(n_samples=64, n_features=8, seed=47)

    class Clock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return 0.0 if self.calls == 1 else 1.0

    monkeypatch.setattr(filter_mod, "perf_counter", Clock())
    result, scores = filter_mod.mrmr_jmi_selection(
        X,
        y,
        4,
        **_base_kwargs(
            mi_scorer=_ordered_relevance,
            mrmr_max_runtime_seconds=0.5,
        ),
    )

    assert result["complete"] is False
    assert result["stop_reason"] == "runtime_budget_exhausted"
    assert result["budget_status"] == "exhausted"
    assert result["unique_pair_evaluations"] == 0
    assert result["selected_indices"].size == 0
    assert np.array_equal(_scores_vector(scores, X.shape[1]), np.zeros(X.shape[1]))


def test_nonfinite_constant_and_pair_scorer_failure_are_telemetried(monkeypatch):
    X, y = _dataset(n_samples=64, n_features=7, seed=53)
    X[:, 1] = 1.0
    original_pair_scorer = filter_mod._mrmr_pair_redundancy_score

    def flaky_pair_scorer(X_pool, X_disc, idx, selected_idx, *, mode, random_state):
        if int(idx) == 2 and int(selected_idx) == 0:
            raise RuntimeError("injected pair failure")
        return original_pair_scorer(
            X_pool,
            X_disc,
            idx,
            selected_idx,
            mode=mode,
            random_state=random_state,
        )

    monkeypatch.setattr(filter_mod, "_mrmr_pair_redundancy_score", flaky_pair_scorer)
    result, _ = filter_mod.mrmr_jmi_selection(
        X,
        y,
        4,
        **_base_kwargs(mi_scorer=_ordered_relevance),
    )

    assert result["complete"] is True
    assert result["pair_evaluation_failures"] >= 1
    assert result["pair_nonfinite_values"] >= 1


def test_config_validates_unlimited_mrmr_budgets_and_propagates_to_selector():
    methods = MethodConfig(
        mrmr_max_unique_pair_evaluations=0,
        mrmr_max_runtime_seconds=0.0,
        mrmr_budget_fallback_mode="relevance_only",
    )
    selector = FeatureSelector.from_config(FeatureSelectorConfig(methods=methods))
    assert selector.mrmr_max_unique_pair_evaluations == 0
    assert selector.mrmr_max_runtime_seconds == 0.0
    assert selector.mrmr_budget_fallback_mode == "relevance_only"

    with pytest.raises(ValueError, match="mrmr_max_unique_pair_evaluations"):
        MethodConfig(mrmr_max_unique_pair_evaluations=-1)
    with pytest.raises(ValueError, match="mrmr_max_runtime_seconds"):
        MethodConfig(mrmr_max_runtime_seconds=-1.0)
    with pytest.raises(ValueError, match="mrmr_budget_fallback_mode"):
        MethodConfig(mrmr_budget_fallback_mode="partial")


def test_pipeline_and_benchmark_cli_preserve_opt_in_mrmr_budget_contract() -> None:
    config = DFFSConfig(
        fs_mrmr_max_unique_pair_evaluations=37,
        fs_mrmr_max_runtime_seconds=2.5,
        fs_mrmr_budget_fallback_mode="relevance_only",
    )
    pipeline = DistributionFeatureSelectionPipeline(config)
    selector = pipeline._build_feature_selector(
        seed=7,
        enabled_methods=("mrmr_jmi",),
        dataset_name="mrmr-budget-unit",
    )

    assert selector.mrmr_max_unique_pair_evaluations == 37
    assert selector.mrmr_max_runtime_seconds == pytest.approx(2.5)
    assert selector.mrmr_budget_fallback_mode == "relevance_only"
    structured_pipeline = DistributionFeatureSelectionPipeline(
        DFFSConfig(
            fs_config=FeatureSelectorConfig(),
            fs_mrmr_max_unique_pair_evaluations=37,
            fs_mrmr_max_runtime_seconds=2.5,
            fs_mrmr_budget_fallback_mode="relevance_only",
        )
    )
    structured_selector = structured_pipeline._build_feature_selector(
        seed=7,
        enabled_methods=("mrmr_jmi",),
        dataset_name="mrmr-budget-structured-unit",
    )
    assert structured_selector.mrmr_max_unique_pair_evaluations == 37
    assert structured_selector.mrmr_max_runtime_seconds == pytest.approx(2.5)
    assert structured_selector.mrmr_budget_fallback_mode == "relevance_only"
    snapshot = pipeline._config_snapshot()
    assert snapshot["fs_mrmr_max_unique_pair_evaluations"] == 37
    assert snapshot["fs_mrmr_max_runtime_seconds"] == pytest.approx(2.5)
    assert snapshot["fs_mrmr_budget_fallback_mode"] == "relevance_only"
    clone = benchmark_runner.clone_config(config)
    assert clone.fs_mrmr_max_unique_pair_evaluations == 37
    assert clone.fs_mrmr_max_runtime_seconds == pytest.approx(2.5)
    assert clone.fs_mrmr_budget_fallback_mode == "relevance_only"

    parser = benchmark_runner.build_arg_parser()
    args = parser.parse_args(
        [
            "--fs-mrmr-max-unique-pair-evaluations",
            "41",
            "--fs-mrmr-max-runtime-seconds",
            "3.0",
            "--fs-mrmr-budget-fallback-mode",
            "relevance_only",
        ]
    )
    spec = next(iter(benchmark_runner.BENCHMARK_DATASETS.values()))
    cli_config = benchmark_runner._build_base_config(args, spec, seed=11)
    assert cli_config.fs_mrmr_max_unique_pair_evaluations == 41
    assert cli_config.fs_mrmr_max_runtime_seconds == pytest.approx(3.0)
    assert cli_config.fs_mrmr_budget_fallback_mode == "relevance_only"

    with pytest.raises(ValueError, match="fs_mrmr_max_unique_pair_evaluations"):
        DFFSConfig(fs_mrmr_max_unique_pair_evaluations=-1)
    with pytest.raises(ValueError, match="fs_mrmr_max_runtime_seconds"):
        DFFSConfig(fs_mrmr_max_runtime_seconds=-1.0)
    with pytest.raises(ValueError, match="fs_mrmr_budget_fallback_mode"):
        DFFSConfig(fs_mrmr_budget_fallback_mode="partial")

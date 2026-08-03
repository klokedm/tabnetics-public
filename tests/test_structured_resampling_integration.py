"""Adversarial integration tests for structured resampling propagation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import tabnetics.pipeline.pipeline as pipeline_module
import tabnetics.pipeline.resampling as resampling_module
import tabnetics.classification.backends as classifier_backends
import tabnetics.feature_selection.base as feature_selection_base
import tabnetics.feature_selection.conformal as feature_selection_conformal
import tabnetics.feature_selection.cv as feature_selection_cv
import tabnetics.feature_selection.mnpo.oracles as mnpo_oracles
from tabnetics.pipeline import (
    FitResamplingContext,
    ResamplingPolicy,
    SplitAssignment,
)
from tabnetics.pipeline.pipeline import (
    ClassificationConfig,
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
    DistributionFitterConfig,
)


def _minimal_config(
    *,
    classifier_candidates: tuple[str, ...] = ("lr",),
    fs_methods: tuple[str, ...] = ("anova_f",),
) -> DFFSConfig:
    classification = ClassificationConfig(
        model_candidates=classifier_candidates,
        include_elastic_net_model=False,
        include_rf_model=False,
        include_knn_model=False,
        include_svm_linear_model=False,
        include_dlda_model=False,
        include_nb_model=False,
        runtime_containment_enabled=False,
        stage2_ratio_augmentation_enabled=False,
        conformal_enabled=False,
    )
    return DFFSConfig(
        random_seed=17,
        test_size=0.20,
        fs_fraction=1.0,
        n_final_features=3,
        enabled_methods=fs_methods,
        selection_strategy="legacy_voting",
        use_rank_prefilter=False,
        prefilter_union_enabled=False,
        screening_enabled=False,
        folding_method="none",
        apply_cdf_transform=False,
        max_dist_features=3,
        multimodal_fallback="none",
        eval_models_enabled=False,
        use_stability_oracle=False,
        use_complexity_oracle=False,
        use_robust_oracle=False,
        use_diversity_oracle=False,
        fs_use_conformal_efficiency=False,
        classification=classification,
        dist_config=DistributionFitterConfig(
            use_lrt=False,
            use_cv=False,
            compute_dip=False,
        ),
    )


def test_pre_split_remaps_supplied_inner_assignments_to_outer_train_positions():
    outer_train = (8, 1, 6, 3, 10, 5, 12, 7)
    outer_test = (0, 2, 4, 9, 11, 13, 14, 15)
    assignments = (
        SplitAssignment(
            scope="outer",
            split_id="published-holdout",
            train_indices=outer_train,
            test_indices=outer_test,
            source="caller",
        ),
        SplitAssignment(
            scope="selector_oracle_cv",
            split_id="published-inner-0",
            train_indices=(8, 1, 10, 5),
            test_indices=(6, 3, 12, 7),
            source="caller",
            allow_unassigned=True,
        ),
        SplitAssignment(
            scope="selector_oracle_cv",
            split_id="published-inner-1",
            train_indices=(6, 3, 12, 7),
            test_indices=(8, 1, 10, 5),
            source="caller",
            allow_unassigned=True,
        ),
    )
    context = FitResamplingContext.iid(16).with_supplied_splits(assignments)
    y = np.tile(np.asarray([0, 1], dtype=int), 8)
    pipeline = DistributionFeatureSelectionPipeline(_minimal_config())

    outer_plan = pipeline._resolve_outer_split_plan(
        y,
        seed=17,
        resampling_context=context,
        supplied_split_id="published-holdout",
    )
    (
        full_context,
        fit_context,
        prepared_outer_plan,
        prepared_train,
        prepared_test,
    ) = pipeline._prepare_pre_split_resampling(
        y_train=y[np.asarray(outer_train, dtype=int)],
        y_test=y[np.asarray(outer_test, dtype=int)],
        split_indices_train=outer_train,
        split_indices_test=outer_test,
        batch_labels_train=None,
        batch_labels_test=None,
        resampling_context=context,
        resolved_outer_split=outer_plan,
    )

    assert full_context is context
    assert prepared_outer_plan.fingerprint == outer_plan.fingerprint
    assert prepared_train == outer_train
    assert prepared_test == outer_test
    assert fit_context.row_ids == outer_train
    assert tuple(
        (split.train_indices, split.test_indices)
        for split in fit_context.supplied_splits
    ) == (
        ((0, 1, 4, 5), (2, 3, 6, 7)),
        ((2, 3, 6, 7), (0, 1, 4, 5)),
    )
    assert all(
        split.parent_context_fingerprint == fit_context.base_fingerprint
        for split in fit_context.supplied_splits
    )

    pipeline._active_resampling_plans = {}
    inner_plan = pipeline._resolve_inner_split_plan(
        fit_context,
        y[np.asarray(outer_train, dtype=int)],
        purpose="selector_oracle_cv",
        n_splits=5,
        n_repeats=1,
        seed=999,
        stratified=True,
    )

    assert inner_plan.index_pairs() == (
        ((0, 1, 4, 5), (2, 3, 6, 7)),
        ((2, 3, 6, 7), (0, 1, 4, 5)),
    )
    recorded = pipeline._active_resampling_plans["selector_oracle_cv"]
    assert recorded["plan_fingerprint"] == inner_plan.fingerprint
    assert recorded["n_splits"] == 2


def test_selector_reuse_restores_configured_methods_for_later_iid_fit():
    configured_methods = {"stability_lasso", "anova_f"}
    X = np.random.default_rng(23).normal(size=(60, 6))
    y = np.tile(np.asarray([0, 1], dtype=int), 30)
    X[:, 0] += 4.0 * y
    X[:, 1] += 2.0 * y
    selector = feature_selection_base.FeatureSelector(
        enabled_methods=configured_methods,
        selection_strategy="legacy_voting",
        variance_threshold=0.0,
        correlation_threshold=1.0,
        n_bootstrap_iterations=3,
        screening_enabled=False,
    )

    _, grouped_result = selector.fit_transform(
        X,
        y,
        n_final_features=2,
        resampling_plan_provider=lambda **kwargs: (),
        resampling_policy="group",
    )

    assert selector.enabled_methods == configured_methods
    assert selector._fit_enabled_methods == {"anova_f"}
    assert set(grouped_result.method_results) == {"anova_f"}
    assert selector.resampling_diagnostics_["excluded_methods"] == [
        "stability_lasso"
    ]

    _, iid_result = selector.fit_transform(
        X,
        y,
        n_final_features=2,
        resampling_policy="iid",
    )

    assert selector.enabled_methods == configured_methods
    assert selector._fit_enabled_methods == configured_methods
    assert set(iid_result.method_results) == configured_methods
    assert selector.resampling_diagnostics_["requested_methods"] == [
        "anova_f",
        "stability_lasso",
    ]
    assert selector.resampling_diagnostics_["effective_methods"] == [
        "anova_f",
        "stability_lasso",
    ]
    assert selector.resampling_diagnostics_["excluded_methods"] == []


def test_structured_selector_requires_resolved_plan_provider():
    X = np.random.default_rng(27).normal(size=(20, 4))
    y = np.tile(np.asarray([0, 1], dtype=int), 10)
    selector = feature_selection_base.FeatureSelector(
        enabled_methods={"anova_f"},
        selection_strategy="legacy_voting",
        variance_threshold=0.0,
        correlation_threshold=1.0,
        n_bootstrap_iterations=3,
        screening_enabled=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "non_iid_internal_resampling_unsupported:"
            "feature_selection_resampling_plan_provider_missing"
        ),
    ):
        selector.fit_transform(
            X,
            y,
            n_final_features=2,
            resampling_policy="group",
        )


def test_structured_selector_disables_private_resampling_diagnostics(
    monkeypatch,
):
    X = np.random.default_rng(29).normal(size=(40, 6))
    y = np.tile(np.asarray([0, 1], dtype=int), 20)
    X[:, 0] += 3.0 * y
    selector = feature_selection_base.FeatureSelector(
        enabled_methods={"anova_f"},
        selection_strategy="legacy_voting",
        variance_threshold=0.0,
        correlation_threshold=1.0,
        n_bootstrap_iterations=3,
        screening_enabled=False,
        use_conformal_efficiency=True,
        rashomon_enabled=True,
    )

    def fail_if_called(*args: Any, **kwargs: Any):
        del args, kwargs
        raise AssertionError("structured selector used a private conformal split")

    monkeypatch.setattr(
        feature_selection_conformal,
        "compute_conformal_singleton_rate",
        fail_if_called,
    )
    selector.fit_transform(
        X,
        y,
        n_final_features=2,
        resampling_plan_provider=lambda **kwargs: (),
        resampling_policy="group",
    )

    assert selector._fit_use_conformal_efficiency is False
    assert selector.oracle.use_conformal_efficiency is False
    assert selector._fit_rashomon_enabled is False
    assert selector.resampling_diagnostics_["requested_oracles"] == {
        "conformal_efficiency": True,
    }
    assert selector.resampling_diagnostics_["effective_oracles"] == {
        "conformal_efficiency": False,
    }
    assert selector.resampling_diagnostics_["excluded_oracles"] == [
        "conformal_efficiency",
    ]
    assert selector.resampling_diagnostics_["oracle_exclusion_reason"] == (
        "non_iid_internal_resampling_unsupported:"
        "feature_selection_conformal_efficiency"
    )
    assert selector.resampling_diagnostics_[
        "requested_post_selection_diagnostics"
    ] == {"rashomon_importance": True}
    assert selector.resampling_diagnostics_[
        "effective_post_selection_diagnostics"
    ] == {"rashomon_importance": False}
    assert selector.resampling_diagnostics_[
        "excluded_post_selection_diagnostics"
    ] == ["rashomon_importance"]
    assert selector.resampling_diagnostics_["post_selection_exclusion_reason"] == (
        "non_iid_internal_resampling_unsupported:"
        "feature_selection_rashomon_importance"
    )
    assert selector._compute_fold_conformal_efficiency(
        X[:20], y[:20], X[20:], y[20:]
    ) == {}

    selector.fit_transform(
        X,
        y,
        n_final_features=2,
        resampling_policy="iid",
    )

    assert selector._fit_use_conformal_efficiency is True
    assert selector.oracle.use_conformal_efficiency is True
    assert selector._fit_rashomon_enabled is True
    assert selector.resampling_diagnostics_["effective_oracles"] == {
        "conformal_efficiency": True,
    }
    assert selector.resampling_diagnostics_["excluded_oracles"] == []
    assert selector.resampling_diagnostics_["effective_post_selection_diagnostics"] == {
        "rashomon_importance": True,
    }
    assert selector.resampling_diagnostics_["excluded_post_selection_diagnostics"] == []


def test_grouped_pipeline_records_zero_overlap_and_exclusion_provenance(
    monkeypatch,
):
    n_rows = 40
    rng = np.random.default_rng(31)
    X = rng.normal(size=(n_rows, 6))
    y = np.tile(np.asarray([0, 1], dtype=int), n_rows // 2)
    patient_ids = np.repeat(np.arange(n_rows // 2), 2)
    site_ids = np.repeat(np.arange(n_rows // 4), 4)
    context = FitResamplingContext(
        n_rows=n_rows,
        patient_ids=tuple(patient_ids.tolist()),
        site_ids=tuple(site_ids.tolist()),
        policy=ResamplingPolicy(
            kind="stratified_group",
            enforced_boundaries=("patient_ids", "site_ids"),
        ),
    )

    iid_splitter_calls: list[str] = []

    class ForbiddenIIDSplitter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            iid_splitter_calls.append("constructor")
            raise AssertionError("structured pipeline instantiated an IID splitter")

    def forbid_train_test_split(*args: Any, **kwargs: Any):
        del args, kwargs
        iid_splitter_calls.append("train_test_split")
        raise AssertionError("structured pipeline executed train_test_split")

    monkeypatch.setattr(
        resampling_module,
        "train_test_split",
        forbid_train_test_split,
    )
    for module, names in (
        (
            resampling_module,
            ("KFold", "RepeatedKFold", "RepeatedStratifiedKFold", "StratifiedKFold"),
        ),
        (pipeline_module, ("StratifiedKFold", "StratifiedShuffleSplit")),
        (classifier_backends, ("StratifiedKFold",)),
        (
            feature_selection_base,
            (
                "KFold",
                "LeaveOneOut",
                "RepeatedKFold",
                "RepeatedStratifiedKFold",
                "StratifiedKFold",
                "StratifiedShuffleSplit",
            ),
        ),
        (
            feature_selection_cv,
            (
                "KFold",
                "LeaveOneOut",
                "RepeatedKFold",
                "RepeatedStratifiedKFold",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, ForbiddenIIDSplitter)

    monkeypatch.setattr(
        feature_selection_conformal,
        "StratifiedShuffleSplit",
        ForbiddenIIDSplitter,
    )

    config = _minimal_config(
        classifier_candidates=("cpda", "spls_da_classifier", "sglnn", "lr"),
        fs_methods=("stability_lasso", "anova_f"),
    )
    config.fs_use_conformal_efficiency = True
    config.fs_rashomon_enabled = True

    result = DistributionFeatureSelectionPipeline(config).run(
        X,
        y,
        dataset_name="structured-groups",
        seed=17,
        resampling_context=context,
        capture_diagnostics=True,
    )

    assert iid_splitter_calls == []
    assert result.resampling_policy["kind"] == "stratified_group"
    assert result.leakage_audit["row_overlap_count"] == 0
    assert result.leakage_audit["identity_overlap_counts"] == {
        "patient_ids": 0,
        "site_ids": 0,
    }

    outer_audit = result.resampling_trace["outer_plan"]["splits"][0]["audit"]
    assert outer_audit["identity_overlap_counts"] == {
        "patient_ids": 0,
        "site_ids": 0,
    }
    inner_splits = result.resampling_trace["inner_plans"][
        "classifier_selection_cv"
    ]["splits"]
    assert inner_splits
    assert all(
        split["source"] == "resolver_stratified_group_cv"
        and split["audit"]["identity_overlap_counts"]
        == {"patient_ids": 0, "site_ids": 0}
        for split in inner_splits
    )

    stages = result.run_diagnostics["pipeline_stages"]
    fs_resampling = stages["feature_selection"]["detailed"]["config"][
        "resampling"
    ]
    assert fs_resampling == {
        "policy": "stratified_group",
        "provider_present": True,
        "requested_methods": ["anova_f", "stability_lasso"],
        "effective_methods": ["anova_f"],
        "excluded_methods": ["stability_lasso"],
        "exclusion_reason": "non_iid_internal_resampling_unsupported",
        "requested_oracles": {"conformal_efficiency": True},
        "effective_oracles": {"conformal_efficiency": False},
        "excluded_oracles": ["conformal_efficiency"],
        "oracle_exclusion_reason": (
            "non_iid_internal_resampling_unsupported:"
            "feature_selection_conformal_efficiency"
        ),
        "requested_post_selection_diagnostics": {"rashomon_importance": True},
        "effective_post_selection_diagnostics": {"rashomon_importance": False},
        "excluded_post_selection_diagnostics": ["rashomon_importance"],
        "post_selection_exclusion_reason": (
            "non_iid_internal_resampling_unsupported:"
            "feature_selection_rashomon_importance"
        ),
    }
    classifier_selection = stages["classifier_selection"]
    assert classifier_selection["model_cv_structured_resampling_excluded"] == {
        "cpda": (
            "non_iid_internal_resampling_unsupported:"
            "classifier:cpda:unsupported"
        ),
        "sglnn": (
            "non_iid_internal_resampling_unsupported:"
            "classifier:sglnn:unsupported"
        ),
        "spls_da_classifier": (
            "non_iid_internal_resampling_unsupported:"
            "classifier:spls_da_classifier:unsupported"
        ),
    }
    assert classifier_selection["model_cv_resampling_source"] == "resolved_plan"
    assert result.config_snapshot["enabled_methods"] == ["anova_f"]


def test_default_result_snapshot_persists_structured_exclusion_and_plan_provenance():
    n_rows = 40
    X = np.random.default_rng(41).normal(size=(n_rows, 6))
    y = np.tile(np.asarray([0, 1], dtype=int), n_rows // 2)
    context = FitResamplingContext(
        n_rows=n_rows,
        patient_ids=tuple(np.repeat(np.arange(n_rows // 2), 2).tolist()),
        site_ids=tuple(np.repeat(np.arange(n_rows // 4), 4).tolist()),
        policy=ResamplingPolicy(
            kind="stratified_group",
            enforced_boundaries=("patient_ids", "site_ids"),
        ),
    )

    result = DistributionFeatureSelectionPipeline(
        _minimal_config(
            classifier_candidates=("sglnn", "lr"),
            fs_methods=("stability_lasso", "anova_f"),
        )
    ).run(
        X,
        y,
        dataset_name="default-structured-provenance",
        seed=17,
        resampling_context=context,
    )
    snapshot = result.config_snapshot

    assert snapshot["requested_enabled_methods"] == [
        "stability_lasso",
        "anova_f",
    ]
    assert snapshot["effective_enabled_methods"] == ["anova_f"]
    assert snapshot["feature_selection_resampling"] == {
        "policy": "stratified_group",
        "provider_present": True,
        "requested_methods": ["anova_f", "stability_lasso"],
        "effective_methods": ["anova_f"],
        "excluded_methods": ["stability_lasso"],
        "exclusion_reason": "non_iid_internal_resampling_unsupported",
        "requested_oracles": {"conformal_efficiency": False},
        "effective_oracles": {"conformal_efficiency": False},
        "excluded_oracles": [],
        "oracle_exclusion_reason": "",
        "requested_post_selection_diagnostics": {"rashomon_importance": False},
        "effective_post_selection_diagnostics": {"rashomon_importance": False},
        "excluded_post_selection_diagnostics": [],
        "post_selection_exclusion_reason": "",
    }
    assert snapshot["model_cv_structured_resampling_policy"] == "stratified_group"
    assert snapshot["model_cv_structured_resampling_excluded"] == {
        "sglnn": (
            "non_iid_internal_resampling_unsupported:"
            "classifier:sglnn:unsupported"
        )
    }
    assert snapshot["model_cv_resampling_source"] == "resolved_plan"
    assert snapshot["model_cv_resampling_plan"] == result.resampling_trace[
        "inner_plans"
    ]["classifier_selection_cv"]


def test_structured_mnpo_excludes_rashomon_private_cv(monkeypatch):
    n_rows = 80
    rng = np.random.default_rng(53)
    y = np.tile(np.asarray([0, 1], dtype=int), n_rows // 2)
    X = rng.normal(size=(n_rows, 12))
    X[:, :3] += 2.0 * y[:, None]
    context = FitResamplingContext(
        n_rows=n_rows,
        patient_ids=tuple(np.repeat(np.arange(n_rows // 2), 2).tolist()),
        policy=ResamplingPolicy(
            kind="stratified_group",
            enforced_boundaries=("patient_ids",),
        ),
    )
    private_splitter_calls: list[str] = []

    class ForbiddenPrivateSplitter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            private_splitter_calls.append("constructor")
            raise AssertionError("structured Rashomon route used private IID CV")

    monkeypatch.setattr(mnpo_oracles, "StratifiedKFold", ForbiddenPrivateSplitter)
    monkeypatch.setattr(mnpo_oracles, "KFold", ForbiddenPrivateSplitter)
    config = _minimal_config(
        classifier_candidates=("lr",),
        fs_methods=("anova_f", "mutual_information"),
    )
    config.selection_strategy = "mnpo_portfolio"
    config.n_final_features = 10
    config.fs_portfolio_size = 2
    config.fs_adaptive_portfolio_sizing_enabled = False
    config.fs_rashomon_enabled = True

    result = DistributionFeatureSelectionPipeline(config).run(
        X,
        y,
        dataset_name="structured-rashomon",
        seed=17,
        resampling_context=context,
    )

    assert private_splitter_calls == []
    diagnostics = result.config_snapshot["feature_selection_resampling"]
    assert diagnostics["requested_post_selection_diagnostics"] == {
        "rashomon_importance": True,
    }
    assert diagnostics["effective_post_selection_diagnostics"] == {
        "rashomon_importance": False,
    }
    assert diagnostics["excluded_post_selection_diagnostics"] == [
        "rashomon_importance",
    ]
    assert diagnostics["post_selection_exclusion_reason"] == (
        "non_iid_internal_resampling_unsupported:"
        "feature_selection_rashomon_importance"
    )


def test_temporal_pipeline_uses_forward_outer_and_inner_assignments(monkeypatch):
    n_rows = 40
    rng = np.random.default_rng(47)
    timestamps_ordered = np.repeat(np.arange(n_rows // 2), 2)
    y_ordered = np.tile(np.asarray([0, 1], dtype=int), n_rows // 2)
    X_ordered = rng.normal(size=(n_rows, 6))
    permutation = rng.permutation(n_rows)
    X = X_ordered[permutation]
    y = y_ordered[permutation]
    timestamps = timestamps_ordered[permutation]
    context = FitResamplingContext(
        n_rows=n_rows,
        timestamps=tuple(timestamps.tolist()),
        policy=ResamplingPolicy(kind="blocked_temporal"),
    )
    pipeline = DistributionFeatureSelectionPipeline(_minimal_config())
    captured_plans = {}
    original_resolver = pipeline._resolve_inner_split_plan

    def capture_inner_plan(*args: Any, **kwargs: Any):
        plan = original_resolver(*args, **kwargs)
        captured_plans[str(kwargs["purpose"])] = plan
        return plan

    monkeypatch.setattr(pipeline, "_resolve_inner_split_plan", capture_inner_plan)
    result = pipeline.run(
        X,
        y,
        dataset_name="structured-time",
        seed=17,
        resampling_context=context,
    )

    outer_train = np.asarray(result.split_indices_train, dtype=int)
    outer_test = np.asarray(result.split_indices_test, dtype=int)
    assert np.max(timestamps[outer_train]) < np.min(timestamps[outer_test])
    assert result.leakage_audit["temporal_order_ok"] is True

    classifier_plan = captured_plans["classifier_selection_cv"]
    fit_timestamps = timestamps[outer_train]
    assert classifier_plan.splits
    for split in classifier_plan.splits:
        train = np.asarray(split.train_indices, dtype=int)
        test = np.asarray(split.test_indices, dtype=int)
        assert np.max(fit_timestamps[train]) < np.min(fit_timestamps[test])
        assert split.audit.temporal_order_ok is True
        assert split.assignment.source == "resolver_expanding_temporal_cv"

    recorded_splits = result.resampling_trace["inner_plans"][
        "classifier_selection_cv"
    ]["splits"]
    assert all(
        split["source"] == "resolver_expanding_temporal_cv"
        and split["audit"]["temporal_order_ok"] is True
        for split in recorded_splits
    )

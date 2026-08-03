"""Contract tests for leak-free nested MAQC pairing evaluation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tabnetics.datasets.schema import DatasetSchema, FeatureRole
from tabnetics.pipeline import FitResamplingContext, ResamplingPolicy, SplitAssignment
from tabnetics.pipeline.pipeline import (
    ClassificationConfig,
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
    DistributionFitterConfig,
    NestedPairingEvaluationError,
)
from tabnetics.pipeline.preprocessing import FoldLocalPreprocessor


def _config(
    *,
    mode: str = "nested_cv",
    enabled_methods: tuple[str, ...] = ("weak",),
    method_sets: tuple[tuple[str, ...], ...] = (("good",),),
    outer_splits: int = 3,
    outer_repeats: int = 1,
    max_outer_evaluations: int = 0,
    max_runtime_seconds: float = 0.0,
) -> DFFSConfig:
    return DFFSConfig(
        random_seed=17,
        enabled_methods=enabled_methods,
        enable_maqc_pairing=True,
        maqc_pairing_method_sets=method_sets,
        maqc_pairing_method_set_names=tuple(
            "candidate_" + str(index) for index in range(len(method_sets))
        ),
        maqc_pairing_score_mode=mode,
        maqc_pairing_outer_splits=outer_splits,
        maqc_pairing_outer_repeats=outer_repeats,
        maqc_pairing_bbc_bootstrap_rounds=20,
        maqc_pairing_bbc_ci_level=0.90,
        maqc_pairing_max_outer_evaluations=max_outer_evaluations,
        maqc_pairing_max_runtime_seconds=max_runtime_seconds,
        selection_strategy="legacy_voting",
        classification=ClassificationConfig(
            selection_mode="legacy",
            backend="sklearn",
            model_candidates=("lr",),
            include_elastic_net_model=False,
            include_rf_model=False,
            include_knn_model=False,
            include_svm_linear_model=False,
            include_dlda_model=False,
            include_nb_model=False,
            stage2_ratio_augmentation_enabled=False,
            conformal_enabled=False,
        ),
    )


def _labels(n_rows: int = 24) -> np.ndarray:
    return np.asarray([index % 2 for index in range(n_rows)], dtype=int)


def _raw_context(
    pipeline: DistributionFeatureSelectionPipeline,
    *,
    context: FitResamplingContext | None = None,
    X: Any | None = None,
    schema: Any = None,
) -> Any:
    y = _labels()
    X_value = (
        np.column_stack((np.arange(y.size), np.arange(y.size) + 100.0))
        if X is None
        else X
    )
    fit_context = context or FitResamplingContext.iid(y.size)
    pipeline._active_resampling_plans = {}
    return pipeline._prepare_nested_pairing_raw_context(
        X_train=X_value,
        y_train=y,
        schema=schema,
        batch_labels=None,
        fit_resampling_context=fit_context,
        dataset_name="nested-contract",
        seed=17,
        configured_methods=pipeline.config.enabled_methods,
        external_feature_scores=None,
        reentrancy_guard=False,
    )


def _fake_clone_result(
    self: DistributionFeatureSelectionPipeline,
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
    **kwargs: Any,
) -> SimpleNamespace:
    """Cheap clone result with deterministic candidate-specific OOF predictions."""

    methods = set(str(method) for method in self.config.enabled_methods)
    y_test_arr = np.asarray(y_test).ravel()
    if "good" in methods:
        y_pred = y_test_arr.copy()
    elif "trap" in methods:
        # The trap is intentionally worse out of fold although a later full
        # fit can report an arbitrary optimistic diagnostic CV score.
        y_pred = 1 - np.asarray(y_test_arr, dtype=int)
    else:
        y_pred = np.zeros(y_test_arr.size, dtype=int)
    weights = kwargs.get("sample_weight_test")
    weight_arr = (
        np.ones(y_test_arr.size, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float).ravel()
    )
    capture = {
        "y_true": tuple(y_test_arr.tolist()),
        "y_pred": tuple(np.asarray(y_pred).tolist()),
        "sample_weights": tuple(weight_arr.tolist()),
    }
    sink = kwargs.get("_evaluation_prediction_sink")
    if not callable(sink):
        raise AssertionError("nested clone did not receive a private prediction sink")
    sink(capture)
    balanced_accuracy = self._safe_balanced_accuracy(
        y_test_arr,
        y_pred,
        sample_weight=weight_arr,
    )
    plan = kwargs["resolved_outer_split"]
    return SimpleNamespace(
        split_indices_train=tuple(kwargs["split_indices_train"]),
        split_indices_test=tuple(kwargs["split_indices_test"]),
        outer_split_fingerprint=str(plan.primary.fingerprint),
        leakage_audit={"ok": True, "reason": "ok"},
        balanced_accuracy=float(balanced_accuracy),
    )


def _fake_raw_cv_candidate(**kwargs: Any) -> dict[str, Any]:
    """Model an optimistic inner-CV diagnostic available only to raw pairing."""

    methods = tuple(str(method) for method in kwargs["enabled_methods"])
    score = 0.99 if "trap" in methods else 0.50
    return {
        "candidate_name": str(kwargs["candidate_name"]),
        "enabled_methods": methods,
        "model_cv_score": float(score),
        "model_cv_score_std": 0.0,
        "model_cv_score_n_splits": 5,
    }


def _assert_no_raw_prediction_fields(value: Any) -> None:
    if isinstance(value, dict):
        assert "y_true" not in value
        assert "y_pred" not in value
        assert "sample_weights" not in value
        for child in value.values():
            _assert_no_raw_prediction_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_raw_prediction_fields(child)


def test_pairing_score_mode_validation_and_snapshot_fields():
    with pytest.raises(ValueError, match="maqc_pairing_score_mode"):
        DFFSConfig(maqc_pairing_score_mode="optimistic")

    cfg = _config(mode="nested_bbc", max_outer_evaluations=9, max_runtime_seconds=12.5)
    snapshot = DistributionFeatureSelectionPipeline(cfg)._config_snapshot()

    assert snapshot["maqc_pairing_score_mode"] == "nested_bbc"
    assert snapshot["maqc_pairing_outer_splits"] == 3
    assert snapshot["maqc_pairing_bbc_bootstrap_rounds"] == 20
    assert snapshot["maqc_pairing_max_outer_evaluations"] == 9
    assert snapshot["maqc_pairing_max_runtime_seconds"] == pytest.approx(12.5)


def test_raw_cv_mode_does_not_prepare_nested_context():
    cfg = _config(mode="raw_cv", outer_splits=0, outer_repeats=0)
    pipeline = DistributionFeatureSelectionPipeline(cfg)

    raw_context = _raw_context(pipeline)

    assert raw_context is None
    assert cfg.maqc_pairing_outer_splits == 0
    assert cfg.maqc_pairing_outer_repeats == 0
    snapshot = pipeline._config_snapshot()
    assert "maqc_pairing_score_mode" not in snapshot
    assert "maqc_pairing_outer_splits" not in snapshot


def test_nested_selection_uses_outer_scores_not_final_fit_cv(monkeypatch):
    cfg = _config(
        enabled_methods=("trap",),
        method_sets=(("good",),),
    )
    pipeline = DistributionFeatureSelectionPipeline(cfg)
    raw_context = _raw_context(pipeline)
    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run_pre_split",
        _fake_clone_result,
    )
    final_fit_calls: list[tuple[str, ...]] = []

    def fake_final_fit(**kwargs: Any) -> dict[str, Any]:
        final_fit_calls.append(tuple(kwargs["enabled_methods"]))
        return {
            "candidate_name": str(kwargs["candidate_name"]),
            # Deliberately optimistic diagnostic. It must not reselect trap.
            "model_cv_score": 1.0,
            "model_cv_score_std": 0.0,
            "model_cv_score_n_splits": 5,
        }

    monkeypatch.setattr(pipeline, "_evaluate_selector_candidate", fake_final_fit)
    selected = pipeline._choose_nested_selector_candidate(
        raw_context=raw_context,
        configured_methods=cfg.enabled_methods,
        X_fs=np.zeros((16, 2), dtype=float),
        y_fs=_labels(16),
        X_train_full=np.zeros((24, 2), dtype=float),
        X_test_full=np.zeros((8, 2), dtype=float),
        y_train_full=_labels(),
        seed=17,
        dataset_name="nested-contract",
        selector_overrides=None,
        post_df_source_raw_train=None,
        post_df_source_raw_test=None,
        post_df_source_base_train=None,
        post_df_source_base_test=None,
        post_df_source_space="prefilter_raw",
    )

    metadata = dict(selected["pairing_meta"])
    assert final_fit_calls == [("good",)]
    assert selected["enabled_methods_source"] == "maqc_pairing_nested_cv"
    assert metadata["maqc_pairing_selected_fs_name"] == "candidate_0"
    assert metadata["maqc_pairing_score_space"] == "nested_oof"
    assert metadata["maqc_pairing_final_fit_model_cv_score"] == pytest.approx(1.0)
    assert metadata["maqc_pairing_final_fit_model_cv_used_for_selection"] is False


def test_nested_bbc_is_deterministic_and_does_not_persist_predictions(monkeypatch):
    cfg = _config(mode="nested_bbc")
    pipeline = DistributionFeatureSelectionPipeline(cfg)
    raw_context = _raw_context(pipeline)
    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run_pre_split",
        _fake_clone_result,
    )

    first = pipeline._run_nested_pairing_evaluation(
        raw_context=raw_context,
        configured_methods=cfg.enabled_methods,
    )
    second = pipeline._run_nested_pairing_evaluation(
        raw_context=raw_context,
        configured_methods=cfg.enabled_methods,
    )
    first_meta = dict(first["pairing_meta"])
    second_meta = dict(second["pairing_meta"])

    for key in (
        "maqc_pairing_bbc_corrected_score",
        "maqc_pairing_bbc_ci_low",
        "maqc_pairing_bbc_ci_high",
        "maqc_pairing_bbc_selection_frequency",
        "maqc_pairing_bbc_valid_draws",
    ):
        assert first_meta[key] == second_meta[key]
    assert first_meta["maqc_pairing_bbc_score_space"] == "bbc_oob_fold_mean"
    assert first_meta["maqc_pairing_bbc_reference_score_space"] == (
        "nested_oof_fold_mean"
    )
    assert first_meta["maqc_pairing_bbc_valid_draws"] == 20
    assert first_meta["maqc_pairing_outer_evaluations_completed"] == 6
    _assert_no_raw_prediction_fields(first_meta)


def test_raw_cv_optimism_trap_is_rejected_by_nested_cv_and_bbc(monkeypatch):
    raw_cfg = _config(
        mode="raw_cv",
        enabled_methods=("good",),
        method_sets=(("trap",),),
    )
    raw_pipeline = DistributionFeatureSelectionPipeline(raw_cfg)
    monkeypatch.setattr(
        raw_pipeline,
        "_evaluate_selector_candidate",
        _fake_raw_cv_candidate,
    )

    raw_selected = raw_pipeline._choose_selector_candidate(
        X_fs=np.zeros((24, 2), dtype=float),
        y_fs=_labels(),
        X_train_full=np.zeros((24, 2), dtype=float),
        X_test_full=np.zeros((8, 2), dtype=float),
        y_train_full=_labels(),
        seed=17,
        dataset_name="raw-cv-optimism-trap",
    )

    assert raw_selected["candidate_name"] == "candidate_0"
    assert raw_selected["enabled_methods"] == ("trap",)

    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run_pre_split",
        _fake_clone_result,
    )
    for mode in ("nested_cv", "nested_bbc"):
        nested_cfg = _config(
            mode=mode,
            enabled_methods=("good",),
            method_sets=(("trap",),),
        )
        nested_pipeline = DistributionFeatureSelectionPipeline(nested_cfg)
        evidence = nested_pipeline._run_nested_pairing_evaluation(
            raw_context=_raw_context(nested_pipeline),
            configured_methods=nested_cfg.enabled_methods,
        )

        assert evidence["selected_name"] == "configured_enabled_methods"
        assert evidence["selected_methods"] == ("good",)
        assert evidence["pairing_meta"]["maqc_pairing_raw_best_fs_name"] == (
            "configured_enabled_methods"
        )


def test_nested_bbc_bootstrap_uses_the_root_reversion_rule(monkeypatch):
    cfg = _config(
        mode="nested_bbc",
        enabled_methods=("weak",),
        method_sets=(("good",),),
    )
    cfg.maqc_pairing_min_improvement = 0.75
    pipeline = DistributionFeatureSelectionPipeline(cfg)
    helper_calls: list[dict[str, np.ndarray]] = []
    original_helper = (
        DistributionFeatureSelectionPipeline._select_nested_pairing_from_fold_scores
    )

    def record_selection_helper(self, **kwargs: Any) -> dict[str, Any]:
        helper_calls.append(
            {
                str(name): np.asarray(scores, dtype=float).copy()
                for name, scores in kwargs["scores_by_candidate"].items()
            }
        )
        return original_helper(self, **kwargs)

    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run_pre_split",
        _fake_clone_result,
    )
    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "_select_nested_pairing_from_fold_scores",
        record_selection_helper,
    )

    evidence = pipeline._run_nested_pairing_evaluation(
        raw_context=_raw_context(pipeline),
        configured_methods=cfg.enabled_methods,
    )
    metadata = dict(evidence["pairing_meta"])

    assert evidence["selected_name"] == "configured_enabled_methods"
    assert metadata["maqc_pairing_raw_best_fs_name"] == "candidate_0"
    assert metadata["maqc_pairing_reverted"] is True
    assert metadata["maqc_pairing_bbc_selection_frequency"] == {
        "candidate_0": 0.0,
        "configured_enabled_methods": 1.0,
    }
    assert metadata["maqc_pairing_bbc_raw_best_frequency"] == {
        "candidate_0": 1.0,
        "configured_enabled_methods": 0.0,
    }
    assert metadata["maqc_pairing_bbc_reverted_draws"] == 20
    assert metadata["maqc_pairing_bbc_revert_reason_frequency"] == {
        "below_min_improvement": 1.0
    }
    assert metadata["maqc_pairing_bbc_reference_nested_oof_fold_mean"] == (
        pytest.approx(metadata["maqc_pairing_selected_cv_score"])
    )
    assert len(helper_calls) == cfg.maqc_pairing_bbc_bootstrap_rounds + 1
    assert all(
        scores["candidate_0"].shape == scores["configured_enabled_methods"].shape
        == (cfg.maqc_pairing_outer_splits,)
        for scores in helper_calls
    )


def test_nested_bbc_rejects_repeats_and_grouped_context_before_fold_work():
    repeated = DistributionFeatureSelectionPipeline(
        _config(mode="nested_bbc", outer_repeats=2)
    )
    with pytest.raises(NestedPairingEvaluationError) as repeated_error:
        _raw_context(repeated)
    assert repeated_error.value.code == "nested_bbc_repeats_unsupported"

    groups = tuple("g" + str(index // 4) for index in range(24))
    grouped_context = FitResamplingContext(
        n_rows=24,
        groups=groups,
        policy=ResamplingPolicy(kind="group", enforced_boundaries=("groups",)),
    )
    grouped = DistributionFeatureSelectionPipeline(_config(mode="nested_bbc"))
    with pytest.raises(NestedPairingEvaluationError) as grouped_error:
        _raw_context(grouped, context=grouped_context)
    assert grouped_error.value.code == "nested_bbc_resampling_policy_unsupported"


def test_nested_modes_reject_diakrino_candidate_sets_and_classifier_paths():
    diakrino_methods = DistributionFeatureSelectionPipeline(
        _config(method_sets=(("diakrino_feature_selector",),))
    )
    with pytest.raises(NestedPairingEvaluationError) as method_error:
        _raw_context(diakrino_methods)
    assert method_error.value.code == "nested_pairing_unsupported_composition"
    assert "diakrino_selector_method" in method_error.value.diagnostics["unsupported"]

    cfg = _config()
    cfg.classification.include_tabpfn_model = True
    diakrino_classifier = DistributionFeatureSelectionPipeline(cfg)
    with pytest.raises(NestedPairingEvaluationError) as classifier_error:
        _raw_context(diakrino_classifier)
    assert classifier_error.value.code == "nested_pairing_unsupported_composition"
    assert "diakrino_classifier" in classifier_error.value.diagnostics["unsupported"]


def test_nested_mode_rejects_duplicate_candidate_names_before_fold_work():
    cfg = _config(method_sets=(("good",), ("other",)))
    cfg.maqc_pairing_method_set_names = ("same", "same")
    pipeline = DistributionFeatureSelectionPipeline(cfg)

    with pytest.raises(NestedPairingEvaluationError) as error:
        _raw_context(pipeline)

    assert error.value.code == "nested_pairing_duplicate_candidate_name"


def test_nested_cv_keeps_grouped_raw_dataframe_rows_and_schema(monkeypatch):
    pandas = pytest.importorskip("pandas")
    groups = tuple("g" + str(index // 4) for index in range(24))
    grouped_context = FitResamplingContext(
        n_rows=24,
        groups=groups,
        policy=ResamplingPolicy(kind="group", enforced_boundaries=("groups",)),
    )
    raw_frame = pandas.DataFrame(
        {
            "token": ["train_only_" + str(index) for index in range(24)],
            "value": np.arange(24, dtype=float),
        }
    )
    schema_marker = object()
    pipeline = DistributionFeatureSelectionPipeline(_config(mode="nested_cv"))
    raw_context = _raw_context(
        pipeline,
        context=grouped_context,
        X=raw_frame,
        schema=schema_marker,
    )
    observed: list[tuple[Any, Any, Any]] = []

    def observe_raw_clone(self, X_train, y_train, X_test, y_test, **kwargs):
        observed.append((X_train.copy(), X_test.copy(), kwargs.get("schema")))
        return _fake_clone_result(self, X_train, y_train, X_test, y_test, **kwargs)

    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run_pre_split",
        observe_raw_clone,
    )
    evidence = pipeline._run_nested_pairing_evaluation(
        raw_context=raw_context,
        configured_methods=pipeline.config.enabled_methods,
    )

    assert evidence["pairing_meta"]["maqc_pairing_outer_policy"] == "group"
    assert len(observed) == 6
    for X_train, X_test, schema in observed:
        assert list(X_train.columns) == ["token", "value"]
        assert list(X_test.columns) == ["token", "value"]
        assert schema is schema_marker
        assert all(str(value).startswith("train_only_") for value in X_train["token"])
        assert all(str(value).startswith("train_only_") for value in X_test["token"])


def test_nested_cv_fits_typed_vocabularies_and_idf_per_outer_fold(monkeypatch):
    pandas = pytest.importorskip("pandas")
    y = _labels()
    context = FitResamplingContext.iid(y.size)
    classification = ClassificationConfig(
        selection_mode="legacy",
        backend="sklearn",
        model_candidates=("lr",),
        include_elastic_net_model=False,
        include_rf_model=False,
        include_knn_model=False,
        include_svm_linear_model=False,
        include_dlda_model=False,
        include_nb_model=False,
        stage2_ratio_augmentation_enabled=False,
        conformal_enabled=False,
    )
    cfg = DFFSConfig(
        random_seed=17,
        typed_input_enabled=True,
        typed_text_hash_buckets=32,
        fs_fraction=1.0,
        n_final_features=3,
        max_dist_features=3,
        enabled_methods=("anova_f",),
        enable_maqc_pairing=True,
        maqc_pairing_score_mode="nested_cv",
        maqc_pairing_outer_splits=3,
        maqc_pairing_min_train_per_class=2,
        selection_strategy="legacy_voting",
        use_rank_prefilter=False,
        prefilter_union_enabled=False,
        screening_enabled=False,
        folding_method="none",
        apply_cdf_transform=False,
        use_stability_oracle=False,
        use_complexity_oracle=False,
        use_robust_oracle=False,
        use_diversity_oracle=False,
        fs_use_conformal_efficiency=False,
        classification=classification,
        dist_config=DistributionFitterConfig(use_lrt=False, use_cv=False),
    )
    pipeline = DistributionFeatureSelectionPipeline(cfg)
    probe_context = _raw_context(
        pipeline,
        context=context,
        X=np.zeros((y.size, 2), dtype=float),
    )
    token_by_row = np.empty(y.size, dtype=object)
    for fold_index, split in enumerate(probe_context.outer_plan.splits):
        token_by_row[np.asarray(split.test_indices, dtype=int)] = (
            "heldout_category_" + str(fold_index)
        )
    frame = pandas.DataFrame(
        {
            "signal": y.astype(float) + np.linspace(0.0, 0.01, num=y.size),
            "category": token_by_row.tolist(),
            "memo": ["heldout_text_" + str(value) for value in token_by_row],
        }
    )
    schema = DatasetSchema.from_dataframe(
        frame,
        roles={
            "signal": FeatureRole.CONTINUOUS,
            "category": FeatureRole.CATEGORICAL,
            "memo": FeatureRole.TEXT,
        },
    )
    raw_context = _raw_context(
        pipeline,
        context=context,
        X=frame,
        schema=schema,
    )
    observed: list[tuple[set[str], tuple[str, ...], np.ndarray]] = []
    original_fit = FoldLocalPreprocessor.fit

    def record_fold_preprocessor_fit(self, X, y=None, *, schema=None):
        fitted = original_fit(self, X, y=y, schema=schema)
        if int(X.shape[0]) == 16:
            observed.append(
                (
                    {str(value) for value in X["category"].tolist()},
                    tuple(fitted.category_levels_["category"]),
                    np.asarray(fitted.text_idf_["memo"], dtype=float).copy(),
                )
            )
        return fitted

    monkeypatch.setattr(FoldLocalPreprocessor, "fit", record_fold_preprocessor_fit)
    evidence = pipeline._run_nested_pairing_evaluation(
        raw_context=raw_context,
        configured_methods=cfg.enabled_methods,
    )
    full_preprocessor = FoldLocalPreprocessor(text_hash_buckets=32).fit(
        frame,
        schema=schema,
    )
    full_idf = np.asarray(full_preprocessor.text_idf_["memo"], dtype=float)

    assert evidence["pairing_meta"]["maqc_pairing_outer_policy"] == "iid"
    assert len(observed) == cfg.maqc_pairing_outer_splits
    expected_tokens = {str(value) for value in token_by_row.tolist()}
    for train_tokens, category_levels, text_idf in observed:
        heldout_tokens = expected_tokens - train_tokens
        assert len(heldout_tokens) == 1
        assert all(
            heldout_token not in category_key
            for heldout_token in heldout_tokens
            for category_key in category_levels
        )
        assert not np.allclose(text_idf, full_idf)


def test_nested_cv_accepts_supplied_outer_fold_assignments(monkeypatch):
    assignments = (
        SplitAssignment(
            scope="maqc_pairing_nested_outer",
            split_id="supplied-0",
            train_indices=tuple(range(8, 24)),
            test_indices=tuple(range(0, 8)),
            source="test",
        ),
        SplitAssignment(
            scope="maqc_pairing_nested_outer",
            split_id="supplied-1",
            train_indices=tuple(range(0, 8)) + tuple(range(16, 24)),
            test_indices=tuple(range(8, 16)),
            source="test",
        ),
        SplitAssignment(
            scope="maqc_pairing_nested_outer",
            split_id="supplied-2",
            train_indices=tuple(range(0, 16)),
            test_indices=tuple(range(16, 24)),
            source="test",
        ),
    )
    context = FitResamplingContext.iid(24).with_supplied_splits(
        assignments,
        policy=ResamplingPolicy(kind="supplied"),
    )
    pipeline = DistributionFeatureSelectionPipeline(_config(mode="nested_cv"))
    raw_context = _raw_context(pipeline, context=context)
    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run_pre_split",
        _fake_clone_result,
    )

    evidence = pipeline._run_nested_pairing_evaluation(
        raw_context=raw_context,
        configured_methods=pipeline.config.enabled_methods,
    )

    assert evidence["pairing_meta"]["maqc_pairing_outer_policy"] == "supplied"
    assert evidence["pairing_meta"]["maqc_pairing_outer_fold_count"] == 3


def test_nested_cv_preserves_temporal_outer_fold_order(monkeypatch):
    timestamps = tuple(range(24))
    temporal_context = FitResamplingContext(
        n_rows=24,
        timestamps=timestamps,
        policy=ResamplingPolicy(kind="blocked_temporal"),
    )
    pipeline = DistributionFeatureSelectionPipeline(_config(mode="nested_cv"))
    raw_context = _raw_context(pipeline, context=temporal_context)
    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run_pre_split",
        _fake_clone_result,
    )

    evidence = pipeline._run_nested_pairing_evaluation(
        raw_context=raw_context,
        configured_methods=pipeline.config.enabled_methods,
    )

    assert evidence["pairing_meta"]["maqc_pairing_outer_policy"] == "blocked_temporal"
    for split in raw_context.outer_plan.splits:
        train_times = {timestamps[index] for index in split.train_indices}
        test_times = {timestamps[index] for index in split.test_indices}
        assert max(train_times) < min(test_times)
        assert split.audit.temporal_order_ok is True


def test_nested_cv_propagates_nonuniform_fold_weights(monkeypatch):
    weights = tuple(0.25 + float(index) / 5.0 for index in range(24))
    weighted_context = FitResamplingContext(n_rows=24, sample_weights=weights)
    pipeline = DistributionFeatureSelectionPipeline(_config(mode="nested_cv"))
    raw_context = _raw_context(pipeline, context=weighted_context)
    observed: list[tuple[tuple[int, ...], tuple[float, ...]]] = []

    def observe_weighted_clone(self, X_train, y_train, X_test, y_test, **kwargs):
        test_indices = tuple(int(value) for value in kwargs["split_indices_test"])
        test_weights = tuple(
            float(value) for value in kwargs["sample_weight_test"]
        )
        observed.append((test_indices, test_weights))
        return _fake_clone_result(self, X_train, y_train, X_test, y_test, **kwargs)

    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run_pre_split",
        observe_weighted_clone,
    )
    evidence = pipeline._run_nested_pairing_evaluation(
        raw_context=raw_context,
        configured_methods=pipeline.config.enabled_methods,
    )

    assert len(observed) == 6
    for test_indices, test_weights in observed:
        assert test_weights == tuple(weights[index] for index in test_indices)
    _assert_no_raw_prediction_fields(evidence["pairing_meta"])


def test_nested_cv_rejects_supplied_plan_with_wrong_declared_fold_count():
    assignments = (
        SplitAssignment(
            scope="maqc_pairing_nested_outer",
            split_id="only-fold",
            train_indices=tuple(range(8, 24)),
            test_indices=tuple(range(0, 8)),
            source="test",
        ),
    )
    context = FitResamplingContext.iid(24).with_supplied_splits(
        assignments,
        policy=ResamplingPolicy(kind="supplied"),
    )
    pipeline = DistributionFeatureSelectionPipeline(_config(mode="nested_cv"))

    with pytest.raises(NestedPairingEvaluationError) as error:
        _raw_context(pipeline, context=context)

    assert error.value.code == "nested_pairing_outer_fold_count"


def test_nested_evaluation_and_runtime_caps_fail_closed(monkeypatch):
    capped = DistributionFeatureSelectionPipeline(
        _config(max_outer_evaluations=1)
    )
    with pytest.raises(NestedPairingEvaluationError) as cap_error:
        _raw_context(capped)
    assert cap_error.value.code == "nested_pairing_evaluation_cap"

    runtime_capped = DistributionFeatureSelectionPipeline(
        _config(max_runtime_seconds=0.1)
    )
    raw_context = _raw_context(runtime_capped)
    timestamps = iter((0.0, 1.0))
    monkeypatch.setattr(runtime_capped, "_timer", lambda: next(timestamps))
    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run_pre_split",
        _fake_clone_result,
    )
    with pytest.raises(NestedPairingEvaluationError) as runtime_error:
        runtime_capped._run_nested_pairing_evaluation(
            raw_context=raw_context,
            configured_methods=runtime_capped.config.enabled_methods,
        )
    assert runtime_error.value.code == "nested_pairing_runtime_cap"


def test_nested_cv_executes_the_real_pipeline_and_persists_aggregate_only():
    rng = np.random.default_rng(404)
    y = np.repeat(np.asarray([0, 1], dtype=int), 24)
    X = rng.normal(size=(48, 10))
    X[:, 0] += 2.0 * y
    X[:, 1] += 0.8 * y
    classification = ClassificationConfig(
        selection_mode="legacy",
        backend="sklearn",
        model_candidates=("lr",),
        include_elastic_net_model=False,
        include_rf_model=False,
        include_knn_model=False,
        include_svm_linear_model=False,
        include_dlda_model=False,
        include_nb_model=False,
        stage2_ratio_augmentation_enabled=False,
        conformal_enabled=False,
    )
    cfg = DFFSConfig(
        random_seed=404,
        fs_fraction=1.0,
        n_final_features=3,
        enabled_methods=("anova_f",),
        enable_maqc_pairing=True,
        maqc_pairing_method_sets=(("mutual_information",),),
        maqc_pairing_method_set_names=("mutual_information",),
        maqc_pairing_score_mode="nested_cv",
        maqc_pairing_outer_splits=2,
        maqc_pairing_min_train_per_class=2,
        selection_strategy="legacy_voting",
        use_rank_prefilter=False,
        prefilter_union_enabled=False,
        screening_enabled=False,
        folding_method="none",
        apply_cdf_transform=False,
        max_dist_features=3,
        use_stability_oracle=False,
        use_complexity_oracle=False,
        use_robust_oracle=False,
        use_diversity_oracle=False,
        fs_use_conformal_efficiency=False,
        classification=classification,
        dist_config=DistributionFitterConfig(use_lrt=False, use_cv=False),
    )

    result = DistributionFeatureSelectionPipeline(cfg).run(
        X,
        y,
        dataset_name="nested-real-smoke",
        seed=404,
    )
    snapshot = result.config_snapshot

    assert np.isfinite(result.balanced_accuracy)
    assert snapshot["maqc_pairing_score_mode"] == "nested_cv"
    assert snapshot["maqc_pairing_score_space"] == "nested_oof"
    assert snapshot["maqc_pairing_outer_evaluations_completed"] == 4
    assert snapshot["maqc_pairing_final_fit_model_cv_used_for_selection"] is False
    _assert_no_raw_prediction_fields(snapshot)
    _assert_no_raw_prediction_fields(result.run_diagnostics)

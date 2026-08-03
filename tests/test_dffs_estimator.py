from __future__ import annotations

import copy
import io
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator
from tabnetics.pipeline import (
    FitResamplingContext,
    ResamplingPolicy,
    UnsupportedSafeBundleStateError,
    load_safe_dffs_bundle,
)
from tabnetics.pipeline.estimator import DFFSClassifier
from tabnetics.pipeline.pipeline import (
    DFFSConfig,
    DistributionFeatureSelectionPipeline,
    UnsafeLegacyBundleError,
)
from tabnetics.pipeline.preprocessing import TypedInputCapabilityError


def _dataset(seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=60,
        n_features=11,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=seed,
    )
    X = np.asarray(X, dtype=float)
    # This is the historic v1 replay failure shape: the selector is fit after
    # a leading variance-floor removal, so the mask must be replayed too.
    X[:, 0] = 1.0
    return X, np.asarray(y)


def _fast_config(
    seed: int = 17,
    *,
    typed: bool = False,
    apply_cdf_transform: bool = False,
    folding_method: str = "none",
    stage2_ratio_augmentation_enabled: bool = False,
    stage2_ratio_max_features: int = 16,
    n_final_features: int = 4,
) -> DFFSConfig:
    return DFFSConfig(
        random_seed=seed,
        typed_input_enabled=typed,
        fs_fraction=1.0,
        n_final_features=n_final_features,
        enabled_methods=("anova_f",),
        selection_strategy="legacy_voting",
        use_rank_prefilter=False,
        apply_cdf_transform=apply_cdf_transform,
        folding_method=folding_method,
        stage2_ratio_augmentation_enabled=stage2_ratio_augmentation_enabled,
        stage2_ratio_max_features=stage2_ratio_max_features,
        model_candidates=("lr",),
        include_elastic_net_model=False,
        include_rf_model=False,
        include_knn_model=False,
        include_svm_linear_model=False,
        include_dlda_model=False,
        include_nb_model=False,
    )


def _diakrino_dataset(seed: int = 71) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.tile(np.asarray([0, 1], dtype=int), 40)
    X = rng.normal(size=(y.size, 11))
    X[:, 0] = 1.0
    X[:, 1] += 4.0 * y
    X[:, 2] -= 3.5 * y
    X[:, 3] += 3.0 * y
    frame = pd.DataFrame(
        X,
        columns=[f"measurement_{index}" for index in range(X.shape[1])],
    )
    return frame, y


def _write_diakrino_sidecar(
    root,
    scores: np.ndarray,
    *,
    dataset_id: str = "alpha",
) -> None:
    feature_dir = root / "feature_logits"
    feature_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(scores, dtype=float).ravel()
    pd.DataFrame(
        {
            "dataset_id": [dataset_id] * values.size,
            "feature_index": np.arange(values.size),
            "chunk_id": np.zeros(values.size, dtype=int),
            "prior_logit": values,
        }
    ).to_parquet(feature_dir / f"{dataset_id}.parquet", index=False)


def _diakrino_config(root, *, seed: int = 71) -> DFFSConfig:
    config = _fast_config(seed, n_final_features=2)
    config.use_rank_prefilter = True
    config.prefilter_top_k = 3
    config.diakrino_prefilter_enabled = True
    config.diakrino_prefilter_mode = "protected_union"
    config.diakrino_prefilter_lambda = 1.0
    # Admit both the leading constant probe and one usable extra.  The
    # constant is removed by the variance floor, exercising post-variance
    # selector coordinates during runtime replay.
    config.diakrino_prefilter_max_extras = 2
    config.diakrino_sidecar_path = str(root)
    config.diakrino_sidecar_dataset_id = "alpha"
    return config


def test_fit_is_train_only_and_refits_every_supplied_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _dataset()

    def _evaluation_forbidden(*args, **kwargs):
        raise AssertionError("DFFSClassifier.fit must not invoke evaluation helpers")

    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline, "run", _evaluation_forbidden
    )
    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run_pre_split",
        _evaluation_forbidden,
    )
    estimator = DFFSClassifier(config=_fast_config(), random_state=17)
    estimator.fit(X, y)

    provenance = dict(estimator.fit_provenance_)
    assert provenance["training_only"] is True
    assert provenance["evaluation_metrics_emitted"] is False
    assert int(provenance["n_fit_rows"]) == X.shape[0]
    assert isinstance(provenance["diakrino_sidecar_resolution"], dict)
    assert provenance["diakrino_prefilter"]["configured"] is False
    assert provenance["diakrino_prefilter"]["applied"] is False
    assert estimator.components_.runtime_model.metadata["n_train"] == X.shape[0]
    assert estimator.predict(X[:5]).shape == (5,)
    assert estimator.predict_proba(X[:5]).shape == (5, 2)


def test_train_only_diakrino_replays_without_sidecar_and_round_trips(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, y = _diakrino_dataset()
    scores = np.linspace(-2.0, 2.0, num=frame.shape[1])
    scores[0] = 10.0
    scores[8] = 9.0
    _write_diakrino_sidecar(tmp_path, scores)
    config = _diakrino_config(tmp_path)

    def _evaluation_forbidden(*args, **kwargs):
        raise AssertionError("DFFSClassifier.fit must not invoke evaluation helpers")

    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run",
        _evaluation_forbidden,
    )
    monkeypatch.setattr(
        DistributionFeatureSelectionPipeline,
        "run_pre_split",
        _evaluation_forbidden,
    )
    estimator = DFFSClassifier(
        config=config,
        dataset_name="alpha",
        random_state=71,
    ).fit(frame, y)
    runtime = estimator.components_.runtime_model
    provenance = dict(estimator.fit_provenance_)
    state = dict(provenance["diakrino_prefilter"])

    assert provenance["schema_version"] == "tabnetics_train_only_components_v1"
    assert state["schema_version"] == "1.0"
    assert state["configured"] is True
    assert state["mode"] == "protected_union"
    assert state["protection_active"] is True
    assert state["original_identity_available"] is True
    assert state["original_identity_reason"] == "original_feature_indices"
    assert 0 in state["diakrino_extra_original_indices"]
    assert 0 not in state["active_original_indices"]
    initial = np.asarray(runtime.prefilter_indices, dtype=int)
    variance = np.asarray(runtime.variance_keep_indices, dtype=int)
    active = np.asarray(state["active_original_indices"], dtype=int)
    np.testing.assert_array_equal(
        initial,
        np.asarray(state["initial_original_indices"], dtype=int),
    )
    np.testing.assert_array_equal(active, initial[variance])

    selected = tuple(estimator.components_.selected_model_input_indices)
    augmentation = dict(provenance["diakrino_protected_augmentation"])
    assert tuple(augmentation["final_original_indices"]) == selected
    assert tuple(estimator.get_feature_names_out()) == tuple(
        frame.columns[index] for index in selected
    )
    probe = frame.iloc[:18]
    transformed = estimator.transform(probe)
    prediction = estimator.predict(probe)
    probabilities = estimator.predict_proba(probe)
    assert transformed.shape == (probe.shape[0], len(selected))
    np.testing.assert_allclose(
        transformed,
        runtime.transform(probe.to_numpy()),
        rtol=0.0,
        atol=0.0,
    )

    repeated = DFFSClassifier(
        config=config,
        dataset_name="alpha",
        random_state=71,
    ).fit(frame, y)
    np.testing.assert_array_equal(
        repeated.components_.runtime_model.prefilter_indices,
        runtime.prefilter_indices,
    )
    np.testing.assert_array_equal(
        repeated.components_.runtime_model.variance_keep_indices,
        runtime.variance_keep_indices,
    )
    assert repeated.components_.selected_model_input_indices == selected
    assert repeated.fit_provenance_["diakrino_prefilter"] == state
    assert repeated.fit_provenance_["diakrino_protected_augmentation"] == augmentation
    np.testing.assert_allclose(
        repeated.transform(probe), transformed, rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(repeated.predict(probe), prediction)

    sidecar = tmp_path / "feature_logits" / "alpha.parquet"
    sidecar.rename(sidecar.with_suffix(".parquet.removed"))
    np.testing.assert_allclose(
        estimator.transform(probe), transformed, rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(estimator.predict(probe), prediction)
    np.testing.assert_allclose(
        estimator.predict_proba(probe),
        probabilities,
        rtol=0.0,
        atol=0.0,
    )

    restored = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_allclose(
        restored.transform(probe), transformed, rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(restored.predict(probe), prediction)
    np.testing.assert_allclose(
        restored.predict_proba(probe), probabilities, rtol=0.0, atol=0.0
    )
    assert tuple(restored.get_feature_names_out()) == tuple(
        estimator.get_feature_names_out()
    )
    assert restored.components_.selected_model_input_indices == selected
    np.testing.assert_array_equal(
        restored.components_.runtime_model.prefilter_indices,
        runtime.prefilter_indices,
    )
    assert restored.fit_provenance_ == estimator.fit_provenance_
    assert json.dumps(restored.config_snapshot_, sort_keys=True) == json.dumps(
        estimator.config_snapshot_, sort_keys=True
    )

    joblib = pytest.importorskip("joblib")
    payload = io.BytesIO()
    joblib.dump(estimator, payload)
    payload.seek(0)
    joblib_restored = joblib.load(payload)
    np.testing.assert_allclose(
        joblib_restored.transform(probe), transformed, rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(joblib_restored.predict(probe), prediction)
    assert joblib_restored.fit_provenance_ == estimator.fit_provenance_
    assert joblib_restored.config_snapshot_ == estimator.config_snapshot_

    with pytest.raises(UnsupportedSafeBundleStateError, match="DIAKRINO/native"):
        estimator.to_safe_bundle()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("duplicate", "train_only_diakrino_prefilter_indices_invalid"),
        ("out_of_range", "train_only_diakrino_prefilter_indices_invalid"),
        ("active_mismatch", "train_only_diakrino_prefilter_indices_invalid"),
        ("selected_schema_mismatch", "train_only_diakrino_selected_mapping_invalid"),
    ],
)
def test_train_only_diakrino_fitted_state_validation_fails_closed(
    tmp_path,
    mutation: str,
    expected_code: str,
) -> None:
    frame, y = _diakrino_dataset(73)
    scores = np.linspace(-3.0, 3.0, num=frame.shape[1])
    scores[0] = 10.0
    scores[8] = 9.0
    _write_diakrino_sidecar(tmp_path, scores)
    estimator = DFFSClassifier(
        config=_diakrino_config(tmp_path, seed=73),
        dataset_name="alpha",
        random_state=73,
    ).fit(frame, y)
    components = estimator.components_
    provenance = copy.deepcopy(estimator.fit_provenance_)
    selected_schema = copy.deepcopy(components.selected_feature_schema)
    state = provenance["diakrino_prefilter"]
    width = components.runtime_model.n_input_features

    if mutation == "duplicate":
        state["initial_original_indices"] = np.asarray(
            [state["initial_original_indices"][0]] * 2,
            dtype=int,
        )
    elif mutation == "out_of_range":
        state["initial_original_indices"] = [width]
    elif mutation == "active_mismatch":
        state["active_original_indices"] = list(
            reversed(state["active_original_indices"])
        )
    else:
        selected_schema.pop("fingerprint", None)
        selected_schema["features"][0]["name"] = "tampered_name"

    with pytest.raises(TypedInputCapabilityError) as exc_info:
        estimator._pipeline_._validate_train_only_diakrino_fitted_state(
            runtime_model=components.runtime_model,
            fit_provenance=provenance,
            source_schema=components.source_schema,
            model_input_schema=components.model_input_schema,
            selected_model_input_indices=components.selected_model_input_indices,
            selected_feature_schema=selected_schema,
        )
    assert exc_info.value.code == expected_code


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("missing", "missing_or_misaligned_scores"),
        ("misaligned", "missing_or_misaligned_scores"),
        ("noninformative", "noninformative_scores"),
    ],
)
def test_train_only_diakrino_abstention_retains_classical_replay(
    tmp_path,
    case: str,
    expected_reason: str,
) -> None:
    frame, y = _diakrino_dataset(75)
    config = _diakrino_config(tmp_path, seed=75)
    if case == "misaligned":
        _write_diakrino_sidecar(tmp_path, np.arange(frame.shape[1] - 1, dtype=float))
    elif case == "noninformative":
        _write_diakrino_sidecar(tmp_path, np.ones(frame.shape[1], dtype=float))

    estimator = DFFSClassifier(
        config=config,
        dataset_name="alpha",
        random_state=75,
    ).fit(frame, y)
    runtime = estimator.components_.runtime_model
    state = dict(estimator.fit_provenance_["diakrino_prefilter"])

    assert state["configured"] is True
    assert state["protection_active"] is True
    assert state["applied"] is False
    assert state["reason"] == expected_reason
    assert state["diakrino_extra_original_indices"] == []
    assert state["diakrino_ranked_candidate_original_indices"] == []
    assert tuple(runtime.prefilter_indices) == tuple(
        state["classical_pool_original_indices"]
    )
    assert estimator.transform(frame.iloc[:5]).shape[0] == 5


@pytest.mark.parametrize(
    ("updates", "expected_code", "expected_break"),
    [
        (
            {"diakrino_prefilter_mode": "legacy_fixed_budget_blend"},
            "train_only_diakrino_legacy_prefilter_unsupported",
            None,
        ),
        (
            {"enable_face_domain_projection": True},
            "train_only_diakrino_original_identity_required",
            "face_projection",
        ),
        (
            {"enable_ratio_features": True},
            "train_only_diakrino_original_identity_required",
            "ratio_generation",
        ),
        (
            {"df_stage_position": "before_fs", "folding_method": "rff"},
            "train_only_diakrino_original_identity_required",
            "pre_fs_folding",
        ),
    ],
)
def test_train_only_diakrino_rejects_nonreplayable_compositions(
    tmp_path,
    updates: dict[str, object],
    expected_code: str,
    expected_break: str | None,
) -> None:
    frame, y = _diakrino_dataset(77)
    config = _diakrino_config(tmp_path, seed=77)
    for field, value in updates.items():
        setattr(config, field, value)

    with pytest.raises(TypedInputCapabilityError) as exc_info:
        DFFSClassifier(
            config=config,
            dataset_name="alpha",
            random_state=77,
        ).fit(frame, y)
    assert exc_info.value.code == expected_code
    if expected_break is not None:
        assert expected_break in exc_info.value.diagnostics["identity_breaks"]


def test_fitted_components_replay_variance_floor_map_after_trusted_round_trip() -> None:
    X, y = _dataset(21)
    estimator = DFFSClassifier(config=_fast_config(21), random_state=21).fit(X, y)
    runtime_model = estimator.components_.runtime_model

    assert tuple(runtime_model.variance_keep_indices) == tuple(range(1, X.shape[1]))
    payload = runtime_model.to_json_dict()
    restored = type(runtime_model).from_json_dict(
        payload,
        trusted_legacy_pickle=True,
    )
    np.testing.assert_allclose(
        restored.transform(X[:12]),
        estimator.transform(X[:12]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(restored.predict(X[:12]), estimator.predict(X[:12]))


def test_legacy_bundle_refuses_to_decode_without_explicit_trust() -> None:
    X, y = _dataset(23)
    estimator = DFFSClassifier(config=_fast_config(23), random_state=23).fit(X, y)

    with pytest.raises(UnsafeLegacyBundleError, match="trusted_legacy_pickle=True"):
        type(estimator.components_.runtime_model).from_json_dict(
            estimator.components_.runtime_model.to_json_dict()
        )


@pytest.mark.parametrize("untrusted_value", [False, 0, 1, "true", "false", object()])
def test_legacy_bundle_requires_literal_true_trust_opt_in(
    untrusted_value: object,
) -> None:
    X, y = _dataset(24)
    estimator = DFFSClassifier(config=_fast_config(24), random_state=24).fit(X, y)

    with pytest.raises(UnsafeLegacyBundleError, match="trusted_legacy_pickle=True"):
        type(estimator.components_.runtime_model).from_json_dict(
            estimator.components_.runtime_model.to_json_dict(),
            trusted_legacy_pickle=untrusted_value,  # type: ignore[arg-type]
        )


def test_clone_params_and_feature_names_are_sklearn_compatible() -> None:
    X, y = _dataset(31)
    estimator = DFFSClassifier(config=_fast_config(31), random_state=31)
    cloned = clone(estimator)
    assert cloned.get_params(deep=False)["random_state"] == 31

    estimator.fit(X, y)
    names = estimator.get_feature_names_out()
    assert names.ndim == 1
    assert names.size == estimator.transform(X[:1]).shape[1]
    estimator.set_params(feature_alignment="strict")
    assert estimator.get_params(deep=False)["feature_alignment"] == "strict"


def test_focused_sklearn_check_estimator_passes_for_numeric_route() -> None:
    check_estimator(
        DFFSClassifier(config=_fast_config(37), random_state=37),
        legacy=False,
    )


def test_typed_schema_alignment_is_explicit_and_recorded() -> None:
    rng = np.random.default_rng(41)
    y = np.repeat([0, 1], 30)
    frame = pd.DataFrame(
        {
            "signal": rng.normal(loc=y, scale=0.4),
            "category": pd.Categorical(np.where(y == 1, "case", "control")),
            "text": np.where(y == 1, "positive marker", "negative marker"),
        }
    )
    strict = DFFSClassifier(config=_fast_config(41, typed=True), random_state=41).fit(
        frame, y
    )
    with pytest.raises(Exception, match="reordered"):
        strict.predict(frame[["category", "signal", "text"]])

    reordered = DFFSClassifier(
        config=_fast_config(41, typed=True),
        random_state=41,
        feature_alignment="reorder",
    ).fit(frame, y)
    prediction = reordered.predict(frame[["category", "signal", "text"]])
    assert prediction.shape == (frame.shape[0],)
    report = dict(reordered.components_.last_inference_schema_report)
    assert report["alignment_applied"] is True
    assert report["typed_semantics_verified"] is True


def test_numeric_dataframe_schema_alignment_and_dtype_contract_are_preserved() -> None:
    X, y = _dataset(43)
    frame = pd.DataFrame(
        X, columns=[f"measurement_{index}" for index in range(X.shape[1])]
    )

    strict = DFFSClassifier(config=_fast_config(43), random_state=43).fit(frame, y)
    assert tuple(strict.components_.source_schema.feature_names) == tuple(frame.columns)
    assert tuple(strict.components_.model_input_schema.feature_names) == tuple(
        frame.columns
    )
    with pytest.raises(Exception, match="reordered"):
        strict.predict(frame.loc[:, list(reversed(frame.columns))])
    with pytest.raises(Exception, match="stored_dtype='float64'.*dtype='object'"):
        strict.predict(frame.astype({"measurement_0": object}))

    reordered = DFFSClassifier(
        config=_fast_config(43),
        random_state=43,
        feature_alignment="reorder",
    ).fit(frame, y)
    prediction = reordered.predict(frame.loc[:, list(reversed(frame.columns))])
    assert prediction.shape == (frame.shape[0],)
    report = dict(reordered.last_inference_schema_report_)
    assert report["alignment_applied"] is True
    assert report["typed_semantics_verified"] is True


def test_fitted_components_replay_post_selection_ratio_route() -> None:
    X, y = _dataset(53)
    train_X, test_X = X[:48], X[48:]
    train_y = y[:48]
    test_y = y[48:]
    config = _fast_config(
        53,
        apply_cdf_transform=True,
        folding_method="pls_da",
        stage2_ratio_augmentation_enabled=True,
        stage2_ratio_max_features=2,
    )

    estimator = DFFSClassifier(config=config, random_state=53).fit(train_X, train_y)
    runtime = estimator.components_.runtime_model
    ratio_meta = dict(runtime.stage2_ratio_meta)
    assert ratio_meta["stage2_ratio_features_applied"] is True
    assert int(ratio_meta["stage2_ratio_features_added"]) > 0

    transformed = estimator.transform(test_X)
    assert transformed.shape[1] == runtime.classifier_model.n_features_in_
    names = estimator.get_feature_names_out()
    assert names.size == transformed.shape[1]
    assert all(name.startswith("stage2_ratio_") for name in names[-2:])
    np.testing.assert_array_equal(
        estimator.predict(test_X),
        runtime.classifier_model.predict(transformed),
    )

    restored = type(runtime).from_json_dict(
        runtime.to_json_dict(),
        trusted_legacy_pickle=True,
    )
    np.testing.assert_allclose(
        restored.transform(test_X), transformed, rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(restored.predict(test_X), estimator.predict(test_X))

    # Evaluation receives held-out covariates, but they must not alter the
    # train-fitted Stage-2 route or its resulting deployed model.
    evaluation = DistributionFeatureSelectionPipeline(
        _fast_config(
            53,
            apply_cdf_transform=True,
            folding_method="pls_da",
            stage2_ratio_augmentation_enabled=True,
            stage2_ratio_max_features=2,
        )
    ).run_pre_split(
        train_X,
        train_y,
        test_X,
        test_y,
        dataset_name="stage2_ratio_parity",
        seed=53,
        capture_artifacts=True,
    )
    evaluation_runtime = type(runtime).from_json_dict(
        evaluation.model_bundle,
        trusted_legacy_pickle=True,
    )
    np.testing.assert_allclose(
        evaluation_runtime.transform(test_X),
        transformed,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        evaluation_runtime.predict_proba(test_X),
        estimator.predict_proba(test_X),
        rtol=0.0,
        atol=0.0,
    )


def test_fitted_components_replay_feature_cap_route_and_safe_bundle() -> None:
    X, y = _dataset(59)
    config = _fast_config(59, n_final_features=10)
    config.fs_max_selected_features_ratio = 0.04
    config.fs_max_selected_features_cap = 2
    estimator = DFFSClassifier(config=config, random_state=59).fit(X, y)
    runtime = estimator.components_.runtime_model

    assert runtime.classifier_model.n_features_in_ == 2
    transformed = estimator.transform(X[:12])
    assert transformed.shape == (12, 2)
    restored = type(runtime).from_json_dict(
        runtime.to_json_dict(),
        trusted_legacy_pickle=True,
    )
    np.testing.assert_allclose(
        restored.transform(X[:12]), transformed, rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(restored.predict(X[:12]), estimator.predict(X[:12]))

    portable = load_safe_dffs_bundle(estimator.to_safe_bundle())
    np.testing.assert_allclose(
        portable.transform(X[:12]), transformed, rtol=0.0, atol=1e-12
    )
    np.testing.assert_array_equal(portable.predict(X[:12]), estimator.predict(X[:12]))


def test_stage2_ratio_admission_is_invariant_to_out_of_support_evaluation_rows() -> (
    None
):
    X, y = _dataset(60)
    train_X, train_y = X[:48], y[:48]
    config = _fast_config(
        60,
        stage2_ratio_augmentation_enabled=True,
        stage2_ratio_max_features=2,
    )
    estimator = DFFSClassifier(config=config, random_state=60).fit(train_X, train_y)
    selected = tuple(estimator.components_.selected_model_input_indices)
    pair = dict(
        estimator.components_.runtime_model.stage2_ratio_meta["stage2_ratio_pairs"][0]
    )
    numerator = selected[int(pair["numerator"])]
    denominator = selected[int(pair["denominator"])]

    # These values lie far outside the train support.  Older evaluation code
    # silently dropped the ratio route when the resulting held-out ratio was
    # non-finite, changing the final model dimensionality.
    adversarial_test = X[48:].copy()
    adversarial_test[:, numerator] = -1e12
    adversarial_test[:, denominator] = 1e12
    evaluation = DistributionFeatureSelectionPipeline(
        _fast_config(
            60,
            stage2_ratio_augmentation_enabled=True,
            stage2_ratio_max_features=2,
        )
    ).run_pre_split(
        train_X,
        train_y,
        adversarial_test,
        y[48:],
        dataset_name="stage2_ratio_train_only_admission",
        seed=60,
        capture_artifacts=True,
    )
    evaluation_runtime = type(estimator.components_.runtime_model).from_json_dict(
        evaluation.model_bundle,
        trusted_legacy_pickle=True,
    )

    assert evaluation_runtime.classifier_model.n_features_in_ == (
        estimator.components_.runtime_model.classifier_model.n_features_in_
    )
    np.testing.assert_allclose(
        evaluation_runtime.transform(adversarial_test),
        estimator.transform(adversarial_test),
        rtol=0.0,
        atol=0.0,
    )


def test_fitted_components_replay_low_p_over_n_bypass_route() -> None:
    X, y = _dataset(61)
    config = _fast_config(61)
    config.regime_gating_enabled = True
    config.regime_gating_low_p_over_n_threshold = 1.0
    config.regime_gating_low_p_over_n_mode = "fast_univariate_filter"
    config.regime_gating_low_p_over_n_filter_max_k = 3
    estimator = DFFSClassifier(config=config, random_state=61).fit(X, y)
    runtime = estimator.components_.runtime_model

    assert runtime.classifier_model.n_features_in_ == 3
    transformed = estimator.transform(X[:12])
    assert transformed.shape == (12, 3)
    restored = type(runtime).from_json_dict(
        runtime.to_json_dict(),
        trusted_legacy_pickle=True,
    )
    np.testing.assert_allclose(
        restored.transform(X[:12]), transformed, rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(restored.predict(X[:12]), estimator.predict(X[:12]))

    portable = load_safe_dffs_bundle(estimator.to_safe_bundle())
    np.testing.assert_allclose(
        portable.transform(X[:12]), transformed, rtol=0.0, atol=1e-12
    )
    np.testing.assert_array_equal(portable.predict(X[:12]), estimator.predict(X[:12]))


def test_numeric_estimator_imputes_missing_values_before_prediction() -> None:
    X, y = _dataset(63)
    X[0, 1] = np.nan
    X[7, 4] = np.nan
    estimator = DFFSClassifier(config=_fast_config(63), random_state=63).fit(X, y)

    probabilities = estimator.predict_proba(X[:12])
    assert probabilities.shape == (12, 2)
    assert np.all(np.isfinite(probabilities))
    portable = load_safe_dffs_bundle(estimator.to_safe_bundle())
    np.testing.assert_allclose(
        portable.predict_proba(X[:12]), probabilities, rtol=0.0, atol=1e-12
    )


def test_fit_consumes_structured_resampling_context() -> None:
    X, y = _dataset(65)
    groups = tuple(f"group_{index // 3}" for index in range(X.shape[0]))
    context = FitResamplingContext(
        n_rows=X.shape[0],
        groups=groups,
        policy=ResamplingPolicy(
            kind="stratified_group",
            enforced_boundaries=("groups",),
        ),
    )
    estimator = DFFSClassifier(config=_fast_config(65), random_state=65).fit(
        X,
        y,
        resampling_context=context,
    )

    provenance = dict(estimator.fit_provenance_)
    assert provenance["fit_context_fingerprint"] == context.fingerprint
    selection = dict(provenance["classifier_selection"])
    assert selection["model_cv_structured_resampling_policy"] == "stratified_group"


def test_fit_consumes_explicit_weighted_resampling_context() -> None:
    X, y = _dataset(47)
    weights = np.linspace(0.7, 1.3, num=X.shape[0])
    context = FitResamplingContext.iid(X.shape[0], sample_weights=weights)
    estimator = DFFSClassifier(config=_fast_config(47), random_state=47).fit(
        X,
        y,
        sample_weight=weights,
        resampling_context=context,
    )

    provenance = dict(estimator.fit_provenance_)
    assert provenance["fit_context_fingerprint"] == context.fingerprint
    weights_meta = dict(provenance["sample_weight_provenance"])
    assert weights_meta["sample_weight_requested"] is True
    assert weights_meta["sample_weight_stage2_fit_consumed"] is True

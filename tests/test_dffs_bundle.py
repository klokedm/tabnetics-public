"""Focused coverage for restricted non-executable DFFS v2/v3 bundle routes."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tabnetics.datasets.schema import DatasetSchema
from tabnetics.pipeline.bundle import (
    SAFE_DFFS_BUNDLE_TRUST_MODE,
    SafeBundleIntegrityError,
    SafeBundleSchemaError,
    UnsupportedSafeBundleStateError,
    create_safe_dffs_bundle,
    load_safe_dffs_bundle,
)
from tabnetics.pipeline.balancing import TrainingBalanceConfig, _make_provenance
from tabnetics.pipeline.estimator import DFFSClassifier
from tabnetics.pipeline.pipeline import (
    DFFSConfig,
    DFFSReproducibleModel,
    FittedPipelineComponents,
    _FixedIndexFeatureSelector,
)
from tabnetics.pipeline.resampling import FitResamplingContext


def _sealed_after_edit(bundle: dict) -> dict:
    """Model an adversary who can recompute the public integrity checksum."""

    body = copy.deepcopy(bundle)
    body.pop("integrity")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    body["integrity"] = {
        "algorithm": "sha256",
        "scope": "bundle_without_integrity",
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    return body


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _refresh_config_hash(bundle: dict) -> dict:
    bundle["hashes"]["config_sha256"] = _sha256_json(bundle["config"])
    return _sealed_after_edit(bundle)


def _supported_components() -> tuple[FittedPipelineComponents, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(712)
    raw = rng.normal(size=(48, 5))
    raw[3, 0] = np.nan
    raw[17, 4] = np.nan
    frame = pd.DataFrame(raw, columns=[f"f{index}" for index in range(raw.shape[1])])
    schema = DatasetSchema.from_dataframe(frame)

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    # The route deliberately exercises each positional mapping, not merely a
    # selector applied directly to the original column order.
    prefilter = np.asarray([2, 4, 0, 1, 3], dtype=int)
    variance = np.asarray([0, 2, 4], dtype=int)
    selector_positions = np.asarray([2, 0], dtype=int)
    selected_input_positions = np.asarray([3, 2], dtype=int)
    y = (1.3 * scaled[:, 3] - 0.8 * scaled[:, 2] > 0.0).astype(int)
    classifier = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=23).fit(
        scaled[:, selected_input_positions], y
    )
    runtime = DFFSReproducibleModel(
        n_input_features=raw.shape[1],
        imputer=imputer,
        batch_model={"mode": "none", "n_features": raw.shape[1], "label_to_code": {}},
        face_meta={"face_projection_applied": False},
        face_pca=None,
        face_lda=None,
        ratio_meta={"ratio_features_applied": False, "ratio_pairs": []},
        scaler_base=scaler,
        distribution_plan={
            "apply_cdf_transform": False,
            "feature_plans": [],
            "dist_feature_indices": [],
            "df_stage_position": "after_fs",
        },
        prefilter_indices=tuple(prefilter.tolist()),
        folding_meta={"folding_applied": False},
        folding_transformer=None,
        folding_standardize_mean=None,
        folding_standardize_scale=None,
        selector=_FixedIndexFeatureSelector(selector_positions),
        stage2_ratio_meta={
            "stage2_ratio_features_applied": False,
            "stage2_ratio_pairs": [],
        },
        classifier_model=classifier,
        variance_keep_indices=tuple(variance.tolist()),
        metadata={"model_name": "lr"},
    )
    components = FittedPipelineComponents(
        runtime_model=runtime,
        classes=np.asarray(classifier.classes_),
        fit_resampling_context=FitResamplingContext.iid(raw.shape[0]),
        config_snapshot={
            "random_seed": 23,
            "diakrino_prefilter_enabled": False,
            "training_balance": TrainingBalanceConfig().to_dict(),
            "training_balance_provenance": {},
        },
        model_name="lr",
        source_schema=schema,
        model_input_schema=schema,
        selected_feature_schema=schema.select(
            selected_input_positions.tolist(),
            operation="feature_selection_output",
        ).to_record(),
        selected_model_input_indices=tuple(selected_input_positions.tolist()),
    )
    return components, frame, y


def _fast_numeric_config(seed: int) -> DFFSConfig:
    """The real train-only numeric route used by the estimator-focused tests."""

    return DFFSConfig(
        random_seed=seed,
        fs_fraction=1.0,
        n_final_features=4,
        enabled_methods=("anova_f",),
        selection_strategy="legacy_voting",
        use_rank_prefilter=False,
        apply_cdf_transform=False,
        folding_method="none",
        stage2_ratio_augmentation_enabled=False,
        model_candidates=("lr",),
        include_elastic_net_model=False,
        include_rf_model=False,
        include_knn_model=False,
        include_svm_linear_model=False,
        include_dlda_model=False,
        include_nb_model=False,
    )


def test_safe_v2_bundle_replays_supported_components_without_pickle() -> None:
    components, frame, _ = _supported_components()

    bundle = create_safe_dffs_bundle(components)
    loaded = load_safe_dffs_bundle(json.dumps(bundle, sort_keys=True))

    assert bundle["trust_mode"] == SAFE_DFFS_BUNDLE_TRUST_MODE
    assert bundle["model"]["classifier"]["kind"] == "sklearn_logistic_regression_coefficients"
    assert "pickle" not in json.dumps(bundle, sort_keys=True).lower()
    assert bundle["hashes"]["config_sha256"]
    assert bundle["hashes"]["model_sha256"]
    assert bundle["hashes"]["package_sha256"]
    assert bundle["schemas"]["source"]["fingerprint"]
    assert bundle["schemas"]["selected"]["fingerprint"]

    expected_transform = components.runtime_model.transform(frame.to_numpy())
    expected_proba = components.runtime_model.predict_proba(frame.to_numpy())
    np.testing.assert_allclose(loaded.transform(frame), expected_transform, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(loaded.predict_proba(frame), expected_proba, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(loaded.predict(frame), components.runtime_model.predict(frame.to_numpy()))
    np.testing.assert_array_equal(loaded.classes_, components.classes)
    assert loaded.get_feature_names_out().tolist() == ["f3", "f2"]


def test_sealed_legacy_safe_v2_without_balance_metadata_loads_as_disabled() -> None:
    components, frame, _ = _supported_components()
    bundle = create_safe_dffs_bundle(components)
    legacy = copy.deepcopy(bundle)
    legacy["schema_version"] = "2"
    legacy["route"]["id"] = "dffs_numeric_median_standard_lr_positional_v2"
    legacy["package"]["codec"] = "dffs_numeric_median_standard_lr_positional_v2"
    legacy["config"].pop("training_balance")
    legacy["config"].pop("training_balance_provenance")
    encoded_config = json.dumps(
        legacy["config"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    legacy["hashes"]["config_sha256"] = hashlib.sha256(encoded_config).hexdigest()
    encoded_package = json.dumps(
        legacy["package"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    legacy["hashes"]["package_sha256"] = hashlib.sha256(encoded_package).hexdigest()
    loaded = load_safe_dffs_bundle(_sealed_after_edit(legacy))
    np.testing.assert_array_equal(
        loaded.predict(frame.iloc[:3]),
        components.runtime_model.predict(frame.iloc[:3].to_numpy()),
    )


def test_safe_bundle_persists_and_verifies_enabled_balance_metadata() -> None:
    components, frame, y = _supported_components()
    config = TrainingBalanceConfig(method="propensity_match", propensity_n_splits=2)
    X = components.runtime_model.transform(frame.to_numpy())
    provenance = _make_provenance(
        config=config,
        seed=23,
        X_input=X,
        y_input=y,
        X_output=X,
        y_output=y,
        context_input=components.fit_resampling_context,
        context_output=components.fit_resampling_context,
        matched_pairs=int(y.size // 2),
        diagnostics={"bundle_contract_fixture": True},
    ).to_dict()
    components.config_snapshot["training_balance"] = config.to_dict()
    components.config_snapshot["training_balance_provenance"] = {
        "fit_components_final_fit": provenance
    }
    bundle = create_safe_dffs_bundle(components)
    load_safe_dffs_bundle(bundle)
    assert bundle["config"]["training_balance"]["method"] == "propensity_match"
    assert bundle["schema_version"] == "3"
    assert "sampler" not in json.dumps(bundle["config"]["training_balance"]).lower()

    missing = copy.deepcopy(bundle)
    missing["config"]["training_balance_provenance"] = {}
    with pytest.raises(SafeBundleSchemaError):
        load_safe_dffs_bundle(_sealed_after_edit(missing))

    stripped = copy.deepcopy(bundle)
    stripped["config"].pop("training_balance")
    stripped["config"].pop("training_balance_provenance")
    stripped_config = json.dumps(
        stripped["config"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    stripped["hashes"]["config_sha256"] = hashlib.sha256(
        stripped_config
    ).hexdigest()
    with pytest.raises(SafeBundleSchemaError):
        load_safe_dffs_bundle(_sealed_after_edit(stripped))

    tampered = copy.deepcopy(bundle)
    tampered["config"]["training_balance_provenance"]["fit_components_final_fit"][
        "matched_pairs"
    ] += 1
    with pytest.raises(SafeBundleIntegrityError):
        load_safe_dffs_bundle(_sealed_after_edit(tampered))


def test_safe_v3_bundle_dual_loads_legacy_balance_v1_metadata() -> None:
    components, frame, y = _supported_components()
    config = TrainingBalanceConfig(method="propensity_match", propensity_n_splits=2)
    X = components.runtime_model.transform(frame.to_numpy())
    provenance = _make_provenance(
        config=config,
        seed=23,
        X_input=X,
        y_input=y,
        X_output=X,
        y_output=y,
        context_input=components.fit_resampling_context,
        context_output=components.fit_resampling_context,
        matched_pairs=int(y.size // 2),
        diagnostics={"legacy_balance_fixture": True},
    ).to_dict()
    components.config_snapshot["training_balance"] = config.to_dict()
    components.config_snapshot["training_balance_provenance"] = {
        "fit_components_final_fit": provenance
    }
    legacy = create_safe_dffs_bundle(components)
    legacy_config = legacy["config"]["training_balance"]
    legacy_config["schema_version"] = "1.0"
    legacy_provenance = legacy["config"]["training_balance_provenance"][
        "fit_components_final_fit"
    ]
    legacy_provenance["schema_version"] = "1.0"
    legacy_provenance["config"] = copy.deepcopy(legacy_config)
    legacy_provenance["config_fingerprint"] = _sha256_json(legacy_config)
    unsigned = dict(legacy_provenance)
    unsigned.pop("provenance_fingerprint")
    legacy_provenance["provenance_fingerprint"] = _sha256_json(unsigned)

    loaded = load_safe_dffs_bundle(_refresh_config_hash(legacy))
    np.testing.assert_array_equal(
        loaded.predict(frame.iloc[:3]),
        components.runtime_model.predict(frame.iloc[:3].to_numpy()),
    )


def test_safe_v3_bundle_rejects_balance_v2_method_mismatch() -> None:
    components, frame, y = _supported_components()
    config = TrainingBalanceConfig(method="random_under")
    X = components.runtime_model.transform(frame.to_numpy())
    provenance = _make_provenance(
        config=config,
        seed=23,
        X_input=X,
        y_input=y,
        X_output=X,
        y_output=y,
        context_input=components.fit_resampling_context,
        context_output=components.fit_resampling_context,
        diagnostics={"random_under_fixture": True},
    ).to_dict()
    components.config_snapshot["training_balance"] = config.to_dict()
    components.config_snapshot["training_balance_provenance"] = {
        "fit_components_final_fit": provenance
    }
    bundle = create_safe_dffs_bundle(components)
    bundle["config"]["training_balance_provenance"]["fit_components_final_fit"][
        "method"
    ] = "random_over"

    with pytest.raises(SafeBundleSchemaError, match="disagrees"):
        load_safe_dffs_bundle(_refresh_config_hash(bundle))


def test_safe_v2_bundle_accepts_runtime_only_with_generated_positional_schema() -> None:
    components, frame, _ = _supported_components()

    bundle = create_safe_dffs_bundle(components.runtime_model)
    loaded = load_safe_dffs_bundle(bundle)

    assert bundle["source_kind"] == "runtime_model_only_generated_positional_schema"
    np.testing.assert_array_equal(
        loaded.predict(frame.to_numpy()),
        components.runtime_model.predict(frame.to_numpy()),
    )


def test_safe_v2_bundle_exports_real_fast_numeric_fitted_components() -> None:
    X, y = make_classification(
        n_samples=60,
        n_features=11,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=731,
    )
    X = np.asarray(X, dtype=float)
    X[:, 0] = 1.0
    frame = pd.DataFrame(X, columns=[f"measurement_{index}" for index in range(X.shape[1])])
    estimator = DFFSClassifier(config=_fast_numeric_config(731), random_state=731).fit(frame, y)

    # These are inactive DIAKRINO defaults.  They must remain provenance metadata,
    # not turn an otherwise numeric safe route into a false rejection.
    assert estimator.components_.config_snapshot["diakrino_conformal_target_fdp"] == 0.2
    assert estimator.components_.config_snapshot["tabentics_diakrino_max_features"] == 256
    bundle = create_safe_dffs_bundle(estimator.components_)
    loaded = load_safe_dffs_bundle(bundle)

    assert [feature["name"] for feature in bundle["schemas"]["source"]["features"]] == list(frame.columns)
    np.testing.assert_allclose(
        loaded.predict_proba(frame.iloc[:12]), estimator.predict_proba(frame.iloc[:12]), rtol=0.0, atol=1e-12
    )
    np.testing.assert_array_equal(loaded.predict(frame.iloc[:12]), estimator.predict(frame.iloc[:12]))
    with pytest.raises(SafeBundleSchemaError, match="schema mismatch"):
        loaded.predict(frame.iloc[:12, ::-1])


def test_safe_v2_bundle_rejects_tampering_before_prediction() -> None:
    components, _, _ = _supported_components()
    bundle = create_safe_dffs_bundle(components)
    tampered = copy.deepcopy(bundle)
    tampered["model"]["classifier"]["coef"][0][0] += 0.25

    with pytest.raises(SafeBundleIntegrityError, match="integrity"):
        load_safe_dffs_bundle(tampered)


def test_safe_v2_bundle_rejects_recomputed_unsupported_classifier_state() -> None:
    components, _, _ = _supported_components()
    bundle = create_safe_dffs_bundle(components)
    tampered = copy.deepcopy(bundle)
    tampered["model"]["classifier"]["kind"] = "pickle_callable"

    with pytest.raises(SafeBundleSchemaError, match="allowlisted"):
        load_safe_dffs_bundle(_sealed_after_edit(tampered))


def test_safe_v2_bundle_rejects_schema_order_and_dtype_drift() -> None:
    components, frame, _ = _supported_components()
    loaded = load_safe_dffs_bundle(create_safe_dffs_bundle(components))

    with pytest.raises(SafeBundleSchemaError, match="schema mismatch"):
        loaded.predict(frame.loc[:, list(reversed(frame.columns))])
    with pytest.raises(SafeBundleSchemaError, match="schema mismatch"):
        loaded.predict(frame.astype({"f0": object}))


def test_safe_v2_bundle_fails_closed_for_batch_state() -> None:
    components, _, _ = _supported_components()
    components.runtime_model.batch_model = {
        "mode": "combat",
        "label_to_code": {"batch_a": 0, "batch_b": 1},
    }

    with pytest.raises(UnsupportedSafeBundleStateError, match="batch-correction"):
        create_safe_dffs_bundle(components)


@pytest.mark.parametrize(
    ("field", "value", "error_fragment"),
    [
        ("diakrino_prefilter_enabled", True, "DIAKRINO/native"),
        ("native_categorical_stage2_enabled", True, "DIAKRINO/native"),
    ],
)
def test_safe_v2_bundle_fails_closed_for_active_diakrino_or_native_state(
    field: str,
    value: object,
    error_fragment: str,
) -> None:
    components, _, _ = _supported_components()
    components.config_snapshot[field] = value

    with pytest.raises(UnsupportedSafeBundleStateError, match=error_fragment):
        create_safe_dffs_bundle(components)


def test_safe_v2_bundle_fails_closed_for_typed_preprocessor_state() -> None:
    components, _, _ = _supported_components()
    components.typed_preprocessor = object()  # type: ignore[assignment]

    with pytest.raises(UnsupportedSafeBundleStateError, match="typed-preprocessor"):
        create_safe_dffs_bundle(components)

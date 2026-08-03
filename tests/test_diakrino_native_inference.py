from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest

from tabnetics.classification import diakrino_native as diakrino_native_module
from tabnetics.classification.diakrino_native import (
    TabenticsDiakrinoNativeOOMError,
    TabenticsDiakrinoNativeClassifier,
    _score_query_chunks_with_oom_retry,
    apply_temperature_to_proba,
    apply_robust_scaler,
    class_stats_numpy,
    fit_robust_scaler,
    load_tabentics_diakrino_fs_classifier,
    marginal_stats_numpy,
    predict_tabentics_diakrino_proba,
    resolve_torch_device,
    run_tabentics_diakrino_native,
    screening_features_numpy,
    select_tabentics_diakrino_features,
)
from tabnetics.feature_selection.diakrino_sidecar import N_CANONICAL_FAMILIES


def test_episode_stats_derive_from_supplied_training_xy():
    X = np.array(
        [
            [0.0, 1.0, np.nan],
            [1.0, 1.5, 5.0],
            [5.0, 4.0, 1.0],
            [6.0, 4.5, 1.5],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    center, scale = fit_robust_scaler(X)
    scaled, missing = apply_robust_scaler(X, center, scale, clip_value=6.0)
    class_stats, class_valid, marginal = class_stats_numpy(scaled, missing, y, max_classes=3)
    feature_valid = np.ones(X.shape[1], dtype=bool)
    class_mask = np.array([True, True, False], dtype=bool)
    screening = screening_features_numpy(marginal, class_stats, class_valid & class_mask[None, :], feature_valid)

    assert center.shape == (3,)
    assert scale.shape == (3,)
    assert missing[0, 2]
    assert class_stats.shape == (3, 3, 24)
    assert marginal.shape == (3, 5)
    assert screening.shape == (3, 18)
    assert np.isfinite(screening).all()


@pytest.mark.parametrize(
    ("X", "missing", "message"),
    [
        (np.arange(12, dtype=np.float32), np.zeros(12, dtype=bool), "statistics X must be 2D"),
        (np.zeros((3, 4), dtype=np.float32), np.zeros(12, dtype=bool), "missing mask must be 2D"),
        (np.zeros((3, 4), dtype=np.float32), np.zeros((3, 5), dtype=bool), "must have the same shape"),
        (np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=bool), "at least one row"),
        (np.empty((3, 0), dtype=np.float32), np.empty((3, 0), dtype=bool), "at least one feature"),
    ],
)
def test_marginal_stats_rejects_malformed_direct_inputs(X, missing, message):
    with pytest.raises(ValueError, match=message):
        marginal_stats_numpy(X, missing)


@pytest.mark.parametrize(
    ("y", "message"),
    [
        (np.asarray([[0], [1], [0], [1]], dtype=np.int64), "class-stat labels must be 1D"),
        (np.asarray([0, 1, 0], dtype=np.int64), "same number of rows"),
        (np.asarray(["case", "control", "case", "control"], dtype=object), "integer-encoded"),
    ],
)
def test_class_stats_rejects_malformed_direct_labels(y, message):
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    missing = np.zeros_like(X, dtype=bool)

    with pytest.raises(ValueError, match=message):
        class_stats_numpy(X, missing, y, max_classes=2)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"marginal": np.zeros((3, 4), dtype=np.float32)}, "marginal stats must have shape"),
        ({"class_stats": np.zeros((3, 2, 23), dtype=np.float32)}, "class stats must have shape"),
        ({"class_stats_valid": np.zeros((3, 3), dtype=bool)}, "class_stats_valid must match"),
        ({"feature_valid": np.zeros((3, 1), dtype=bool)}, "feature_valid must be 1D"),
        ({"feature_valid": np.ones(2, dtype=bool)}, "must agree on feature count"),
    ],
)
def test_screening_features_rejects_malformed_direct_inputs(updates, message):
    payload = {
        "marginal": np.zeros((3, 5), dtype=np.float32),
        "class_stats": np.zeros((3, 2, 24), dtype=np.float32),
        "class_stats_valid": np.zeros((3, 2), dtype=bool),
        "feature_valid": np.ones(3, dtype=bool),
    }
    payload.update(updates)

    with pytest.raises(ValueError, match=message):
        screening_features_numpy(
            payload["marginal"],
            payload["class_stats"],
            payload["class_stats_valid"],
            payload["feature_valid"],
        )


def test_select_features_falls_back_to_variance_topk():
    X = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 1.0, 2.0],
            [0.0, 8.0, 2.0, 4.0],
        ],
        dtype=np.float32,
    )

    selected, meta = select_tabentics_diakrino_features(X, 2)

    assert selected.tolist() == [1, 3]
    assert meta["native_diakrino_feature_cap_policy"] == "variance_topk"
    assert meta["native_diakrino_sidecar_loaded"] is False
    assert meta["native_diakrino_sidecar_used"] is False
    assert meta["native_diakrino_sidecar_status"] == "not_configured"
    assert meta["native_diakrino_sidecar_reason"] == "sidecar_path_not_configured"


def test_select_features_all_features_does_not_require_or_claim_sidecar():
    X = np.arange(12, dtype=np.float32).reshape(3, 4)

    selected, meta = select_tabentics_diakrino_features(
        X,
        10,
        sidecar_path="/tmp/does-not-need-to-exist.parquet",
        dataset_id="not_a_catalog_dataset",
    )

    assert selected.tolist() == [0, 1, 2, 3]
    assert meta["native_diakrino_feature_cap_policy"] == "all_features"
    assert meta["native_diakrino_feature_score_column"] == ""
    assert meta["native_diakrino_sidecar_loaded"] is False
    assert meta["native_diakrino_sidecar_used"] is False
    assert meta["native_diakrino_sidecar_status"] == "not_needed"
    assert meta["native_diakrino_sidecar_reason"] == "feature_budget_covers_all_features"


@pytest.mark.parametrize(
    ("X", "message"),
    [
        (np.arange(12, dtype=np.float32), "X_train must be 2D"),
        (np.empty((0, 4), dtype=np.float32), "at least one training row"),
        (np.empty((3, 0), dtype=np.float32), "at least one feature"),
    ],
)
def test_select_features_rejects_malformed_direct_training_matrix(X, message):
    with pytest.raises(ValueError, match=message):
        select_tabentics_diakrino_features(X, 2)


def test_select_features_uses_chunk_zscore_sidecar_and_discrete_skip(tmp_path):
    pd = pytest.importorskip("pandas")
    family_cont = np.zeros(N_CANONICAL_FAMILIES, dtype=np.float64)
    family_cont[0] = 5.0
    family_discrete = np.zeros(N_CANONICAL_FAMILIES, dtype=np.float64)
    family_discrete[31] = 5.0
    path = tmp_path / "toy.parquet"
    pd.DataFrame(
        {
            "dataset_id": ["toy"] * 5,
            "feature_index": [0, 1, 2, 3, 4],
            "chunk_id": [0, 0, 0, 0, 0],
            "feature_selection_logit": [0.0, 100.0, 5.0, 4.0, 3.0],
            "population_family_logits": [family_cont, family_discrete, family_cont, family_cont, family_cont],
        }
    ).to_parquet(path, index=False)
    X = np.arange(20, dtype=np.float32).reshape(4, 5)

    selected, meta = select_tabentics_diakrino_features(X, 3, sidecar_path=path, dataset_id="toy")

    assert selected.tolist() == [2, 3, 4]
    assert meta["native_diakrino_sidecar_loaded"] is True
    assert meta["native_diakrino_sidecar_used"] is True
    assert meta["native_diakrino_sidecar_status"] == "used"
    assert meta["native_diakrino_sidecar_reason"] == "ok"
    assert meta["native_diakrino_feature_cap_policy"] == "sidecar_chunk_zscore_feature_selection_logit_then_variance"


def test_select_features_aligns_unsorted_sidecar_by_feature_index(tmp_path):
    pd = pytest.importorskip("pandas")
    family_cont = np.zeros(N_CANONICAL_FAMILIES, dtype=np.float64)
    family_cont[0] = 5.0
    family_discrete = np.zeros(N_CANONICAL_FAMILIES, dtype=np.float64)
    family_discrete[31] = 5.0
    path = tmp_path / "unsorted.parquet"
    pd.DataFrame(
        {
            "dataset_id": ["toy"] * 5,
            "feature_index": [4, 0, 3, 1, 2],
            "chunk_id": [0, 0, 0, 0, 0],
            "feature_selection_logit": [40.0, 10.0, 30.0, 100.0, 20.0],
            "population_family_logits": [family_cont, family_cont, family_cont, family_discrete, family_cont],
        }
    ).to_parquet(path, index=False)
    X = np.arange(20, dtype=np.float32).reshape(4, 5)

    selected, meta = select_tabentics_diakrino_features(X, 3, sidecar_path=path, dataset_id="toy")

    assert selected.tolist() == [4, 3, 2]
    assert meta["native_diakrino_sidecar_loaded"] is True
    assert meta["native_diakrino_sidecar_used"] is True


def test_select_features_blank_dataset_id_does_not_filter_direct_sidecar(tmp_path):
    pd = pytest.importorskip("pandas")
    family_cont = np.zeros(N_CANONICAL_FAMILIES, dtype=np.float64)
    family_cont[0] = 5.0
    path = tmp_path / "single_dataset.parquet"
    pd.DataFrame(
        {
            "dataset_id": ["toy"] * 4,
            "feature_index": [0, 1, 2, 3],
            "chunk_id": [0, 0, 0, 0],
            "feature_selection_logit": [1.0, 2.0, 20.0, 30.0],
            "population_family_logits": [family_cont, family_cont, family_cont, family_cont],
        }
    ).to_parquet(path, index=False)
    X = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [9.0, 8.0, 1.0, 2.0],
            [18.0, 16.0, 2.0, 4.0],
        ],
        dtype=np.float32,
    )

    selected, meta = select_tabentics_diakrino_features(X, 2, sidecar_path=path, dataset_id="")

    assert selected.tolist() == [3, 2]
    assert meta["native_diakrino_sidecar_loaded"] is True
    assert meta["native_diakrino_sidecar_used"] is True
    assert meta["native_diakrino_sidecar_status"] == "used"


def test_select_features_loaded_but_unusable_sidecar_reports_variance_fallback(tmp_path):
    pd = pytest.importorskip("pandas")
    path = tmp_path / "short.parquet"
    pd.DataFrame(
        {
            "dataset_id": ["toy"] * 2,
            "feature_index": [0, 1],
            "chunk_id": [0, 0],
            "feature_selection_logit": [10.0, 9.0],
        }
    ).to_parquet(path, index=False)
    X = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 1.0, 2.0],
            [0.0, 8.0, 2.0, 4.0],
        ],
        dtype=np.float32,
    )

    selected, meta = select_tabentics_diakrino_features(X, 2, sidecar_path=path, dataset_id="toy")

    assert selected.tolist() == [1, 3]
    assert meta["native_diakrino_sidecar_loaded"] is True
    assert meta["native_diakrino_sidecar_used"] is False
    assert meta["native_diakrino_sidecar_status"] == "loaded_unusable"
    assert meta["native_diakrino_sidecar_reason"] == "score_column_missing_or_short"
    assert meta["native_diakrino_feature_cap_policy"] == "variance_topk"
    assert meta["native_diakrino_feature_score_column"] == ""


def test_select_features_reports_sidecar_load_error_and_variance_fallback(monkeypatch):
    from tabnetics.feature_selection import diakrino_sidecar as diakrino_sidecar_module

    def fail_load(*args, **kwargs):
        raise RuntimeError("broken sidecar manifest")

    monkeypatch.setattr(diakrino_sidecar_module.DiakrinoSidecar, "load", fail_load)
    X = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 1.0, 2.0],
            [0.0, 8.0, 2.0, 4.0],
        ],
        dtype=np.float32,
    )

    selected, meta = select_tabentics_diakrino_features(
        X,
        2,
        sidecar_path="/tmp/malformed-sidecar.json",
        dataset_id="toy",
    )

    assert selected.tolist() == [1, 3]
    assert meta["native_diakrino_feature_cap_policy"] == "variance_topk"
    assert meta["native_diakrino_feature_score_column"] == ""
    assert meta["native_diakrino_sidecar_loaded"] is False
    assert meta["native_diakrino_sidecar_used"] is False
    assert meta["native_diakrino_sidecar_status"] == "load_error"
    assert meta["native_diakrino_sidecar_reason"] == "RuntimeError: broken sidecar manifest"


def test_resolve_torch_device_cpu_is_explicit():
    pytest.importorskip("torch")

    assert str(resolve_torch_device("cpu")) == "cpu"


def test_resolve_torch_device_preserves_indexed_cuda_when_available(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert str(resolve_torch_device("cuda:0")) == "cuda:0"


def test_probability_temperature_calibration_preserves_rows():
    proba = np.array([[0.01, 0.99], [0.45, 0.55]], dtype=float)

    softened = apply_temperature_to_proba(proba, temperature=2.0)

    assert softened.shape == proba.shape
    assert np.allclose(softened.sum(axis=1), 1.0)
    assert softened[0, 1] < proba[0, 1]
    assert softened[1, 1] < proba[1, 1]


def test_native_estimator_uses_supplied_training_xy_without_catalog(monkeypatch):
    calls: list[tuple[tuple[int, int], tuple[int, int], str]] = []

    class _Encoder:
        classes_ = np.array([0, 1])

        def transform(self, values):
            return np.asarray(values, dtype=int)

    def fake_predict(X_train, y_train, X_query, **kwargs):
        calls.append((tuple(np.asarray(X_train).shape), tuple(np.asarray(X_query).shape), str(kwargs["dataset_name"])))
        proba = np.tile(np.array([[0.25, 0.75]], dtype=float), (int(np.asarray(X_query).shape[0]), 1))
        return proba, {"native_diakrino_used_features": int(np.asarray(X_train).shape[1])}, _Encoder()

    monkeypatch.setattr(diakrino_native_module, "predict_tabentics_diakrino_proba", fake_predict)
    X = np.arange(40, dtype=np.float32).reshape(10, 4)
    y = np.array([0, 1] * 5, dtype=int)
    clf = TabenticsDiakrinoNativeClassifier(
        checkpoint="/tmp/native-diakrino-smoke.pt",
        dataset_id="toy_dataset",
        calibrate_probabilities=False,
    )

    clf.fit(X, y)
    proba = clf.predict_proba(X[:3])

    assert proba.shape == (3, 2)
    assert calls == [((10, 4), (3, 4), "toy_dataset")]


def test_native_estimator_blank_dataset_id_does_not_fabricate_dataset_name(monkeypatch):
    calls: list[str] = []

    class _Encoder:
        classes_ = np.array([0, 1])

        def transform(self, values):
            return np.asarray(values, dtype=int)

    def fake_predict(X_train, y_train, X_query, **kwargs):
        calls.append(str(kwargs["dataset_name"]))
        proba = np.tile(np.array([[0.4, 0.6]], dtype=float), (int(np.asarray(X_query).shape[0]), 1))
        return proba, {}, _Encoder()

    monkeypatch.setattr(diakrino_native_module, "predict_tabentics_diakrino_proba", fake_predict)
    X = np.arange(40, dtype=np.float32).reshape(10, 4)
    y = np.array([0, 1] * 5, dtype=int)
    clf = TabenticsDiakrinoNativeClassifier(
        checkpoint="/tmp/native-diakrino-smoke.pt",
        dataset_id="",
        calibrate_probabilities=False,
    )

    clf.fit(X, y)
    clf.predict_proba(X[:2])

    assert calls == [""]


def test_native_estimator_rejects_query_feature_count_mismatch(monkeypatch):
    def fake_predict(*args, **kwargs):  # pragma: no cover - should not be reached
        raise AssertionError("predict_tabentics_diakrino_proba should not be called")

    monkeypatch.setattr(diakrino_native_module, "predict_tabentics_diakrino_proba", fake_predict)
    X = np.arange(40, dtype=np.float32).reshape(10, 4)
    y = np.array([0, 1] * 5, dtype=int)
    clf = TabenticsDiakrinoNativeClassifier(
        checkpoint="/tmp/native-diakrino-smoke.pt",
        calibrate_probabilities=False,
    )
    clf.fit(X, y)

    with pytest.raises(ValueError, match="expected 4 features, got 5"):
        clf.predict_proba(np.zeros((2, 5), dtype=np.float32))


def test_native_predict_helper_rejects_query_feature_count_before_checkpoint_load():
    X_train = np.arange(24, dtype=np.float32).reshape(8, 3)
    y_train = np.array([0, 1] * 4, dtype=int)
    X_query = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="same feature count as X_train"):
        predict_tabentics_diakrino_proba(
            X_train,
            y_train,
            X_query,
            dataset_name="shape_guard",
            seed=0,
            checkpoint="/tmp/does-not-need-to-exist.pt",
            max_features=3,
            batch_size=2,
            device="cpu",
        )


def test_native_predict_helper_rejects_training_label_length_before_checkpoint_load():
    X_train = np.arange(24, dtype=np.float32).reshape(8, 3)
    y_train = np.array([0, 1, 0], dtype=int)
    X_query = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="same number of rows"):
        predict_tabentics_diakrino_proba(
            X_train,
            y_train,
            X_query,
            dataset_name="shape_guard",
            seed=0,
            checkpoint="/tmp/does-not-need-to-exist.pt",
            max_features=3,
            batch_size=2,
            device="cpu",
        )


def test_native_predict_helper_rejects_single_class_before_torch_import(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if str(name) == "torch":
            raise AssertionError("torch should not be imported for invalid native DIAKRINO labels")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    X_train = np.arange(24, dtype=np.float32).reshape(8, 3)
    y_train = np.zeros(8, dtype=int)
    X_query = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="at least two classes"):
        predict_tabentics_diakrino_proba(
            X_train,
            y_train,
            X_query,
            dataset_name="label_guard",
            seed=0,
            checkpoint="/tmp/does-not-need-to-exist.pt",
            max_features=3,
            batch_size=2,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("X_train", "y_train", "X_query", "message"),
    [
        (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=int),
            np.zeros((2, 3), dtype=np.float32),
            "at least one training row",
        ),
        (
            np.arange(24, dtype=np.float32).reshape(8, 3),
            np.array([0, 1] * 4, dtype=int),
            np.empty((0, 3), dtype=np.float32),
            "at least one query row",
        ),
    ],
)
def test_native_predict_helper_rejects_empty_episode_arrays_before_checkpoint_load(
    X_train, y_train, X_query, message
):
    with pytest.raises(ValueError, match=message):
        predict_tabentics_diakrino_proba(
            X_train,
            y_train,
            X_query,
            dataset_name="empty_guard",
            seed=0,
            checkpoint="/tmp/does-not-need-to-exist.pt",
            max_features=3,
            batch_size=2,
            device="cpu",
        )


def test_native_metric_helper_rejects_unseen_test_labels_before_checkpoint_load(monkeypatch):
    def fake_predict(*args, **kwargs):  # pragma: no cover - should not be reached
        raise AssertionError("predict_tabentics_diakrino_proba should not be called")

    monkeypatch.setattr(diakrino_native_module, "predict_tabentics_diakrino_proba", fake_predict)
    X_train = np.arange(24, dtype=np.float32).reshape(8, 3)
    y_train = np.array([0, 1] * 4, dtype=int)
    X_test = np.zeros((3, 3), dtype=np.float32)
    y_test = np.array([0, 1, 2], dtype=int)

    with pytest.raises(ValueError, match="metric labels must all be present in y_train"):
        run_tabentics_diakrino_native(
            X_train,
            y_train,
            X_test,
            y_test,
            dataset_name="metric_guard",
            seed=0,
            checkpoint="/tmp/does-not-need-to-exist.pt",
            max_features=3,
            batch_size=2,
            device="cpu",
        )


def test_native_estimator_fit_rejects_training_label_length_mismatch():
    clf = TabenticsDiakrinoNativeClassifier(
        checkpoint="/tmp/native-diakrino-smoke.pt",
        calibrate_probabilities=False,
    )

    with pytest.raises(ValueError, match="same number of rows"):
        clf.fit(np.zeros((4, 3), dtype=np.float32), np.array([0, 1], dtype=int))


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=int), "at least one training row"),
        (np.zeros((4, 0), dtype=np.float32), np.array([0, 1, 0, 1], dtype=int), "at least one feature"),
        (np.zeros((4, 3), dtype=np.float32), np.zeros(4, dtype=int), "at least two classes"),
    ],
)
def test_native_estimator_fit_rejects_invalid_training_arrays_before_sklearn_import(
    monkeypatch, X, y, message
):
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if str(name).startswith(("sklearn.metrics", "sklearn.model_selection", "sklearn.preprocessing")):
            raise AssertionError("sklearn fit helpers should not be imported for invalid native DIAKRINO arrays")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    clf = TabenticsDiakrinoNativeClassifier(
        checkpoint="/tmp/native-diakrino-smoke.pt",
        calibrate_probabilities=False,
    )

    with pytest.raises(ValueError, match=message):
        clf.fit(X, y)


def _tiny_native_classifier_config():
    from tabnetics.classification.tabentics_diakrino_fs_classifier import TabenticsDiakrinoFSClassifierConfig

    return TabenticsDiakrinoFSClassifierConfig(
        d_model=16,
        n_heads=2,
        context_layers=1,
        query_layers=1,
        class_layers=1,
        ffn_expansion=1,
        dropout=0.0,
        max_classes=3,
        max_feature_tokens=8,
        use_distribution_series=False,
        fs_refiner_steps=0,
        feature_position_encoding="rope_fourier",
        position_encoding_scale_init=0.25,
        position_frequency_bands=2,
        label_smoothing=0.0,
    )


def _test_model_state_sha256(torch, state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_trusted_classifier_bundle(root, *, usable=True, state_variant=None, seed=7):
    torch = pytest.importorskip("torch")
    from tabnetics.classification.tabentics_diakrino_fs_classifier import TabenticsDiakrinoFSClassifier

    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    cfg = _tiny_native_classifier_config()
    model = TabenticsDiakrinoFSClassifier(cfg)
    state = dict(model.state_dict())
    if state_variant == "missing":
        state.pop("class_logit_head.1.weight")
    elif state_variant == "shape":
        state["class_logit_head.1.weight"] = state["class_logit_head.1.weight"][:0]
    elif state_variant is not None:
        raise AssertionError(f"unknown test state variant: {state_variant}")

    state_sha256 = _test_model_state_sha256(torch, state)
    split_sha256 = "1" * 64
    row_sha256 = "2" * 64
    feature_sha256 = "3" * 64
    data_config_sha256 = "4" * 64
    task_metrics = [
        {
            "group_id": f"group_{index:02d}",
            "examples": 32,
            "negative_log_likelihood": 0.5,
            "chance_negative_log_likelihood": 0.6,
            "nll_improvement_vs_chance": 0.1,
            "expected_calibration_error": 0.05,
        }
        for index in range(20)
    ]
    bootstrap = {
        "schema_version": "tabentics_diakrino_task_admission_bootstrap_v1",
        "resampling_unit": "heldout_dataset_or_world_group",
        "group_weighting": "equal",
        "seed": 50,
        "samples": 10000,
        "lower_quantile": 0.025,
        "upper_quantile": 0.975,
        "nll_improvement_vs_chance": {
            "task_mean": 0.1,
            "lower_quantile_value": 0.1,
            "upper_quantile_value": 0.1,
        },
        "expected_calibration_error": {
            "task_mean": 0.05,
            "lower_quantile_value": 0.05,
            "upper_quantile_value": 0.05,
        },
    }
    hard_minimums = {
        "tasks": 20,
        "examples": 640,
        "examples_per_task": 32,
        "classes": 2,
        "bootstrap_samples": 10000,
        "ece_bins": 10,
    }
    admission_policy = {
        "accuracy_margin": 0.02,
        "nll_improvement_margin": 0.0,
        "max_ece": 0.10,
        "ece_interpretation": "operational_confidence_ece_guardrail_not_proof_of_multiclass_calibration",
        "ece_bins": 10,
        "bootstrap_samples": 10000,
        "bootstrap_lower_quantile": 0.025,
        "bootstrap_upper_quantile": 0.975,
        "bootstrap_unit": "heldout_dataset_or_world_group_equal_weight",
        "minimum_tasks": 20,
        "minimum_examples": 640,
        "minimum_examples_per_task": 32,
        "minimum_classes": 2,
        "hard_minimums": hard_minimums,
    }
    effective_serving_config = {
        "schema_version": "tabentics_diakrino_classifier_serving_config_v1",
        "admission_policy": admission_policy,
    }
    effective_serving_config_sha256 = hashlib.sha256(
        json.dumps(effective_serving_config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    checks = {
        "accuracy_margin": bool(usable),
        "balanced_accuracy_margin": True,
        "group_disjoint": True,
        "heldout_evaluation": True,
        "minimum_classes": True,
        "minimum_examples": True,
        "minimum_examples_per_task": True,
        "minimum_tasks": True,
        "task_bootstrap_contract": True,
        "ece_bins_contract": True,
        "nll_task_bootstrap_margin": True,
        "ece_threshold_predeclared": True,
        "ece_task_bootstrap_upper_bound": True,
        "provenance_hashes_present": True,
        "support_joint_policy": True,
        "task_accuracy_ci_margin": True,
        "task_balanced_accuracy_ci_margin": True,
        "trained_classifier_head": True,
    }
    failed_checks = [] if usable else ["accuracy_margin"]
    head_trust = {
        "schema_version": "tabentics_diakrino_classifier_head_trust_v3",
        "gate": "heldout_query_head_vs_per_episode_chance",
        "evaluation_scope": "heldout_dataset_or_world_groups",
        "trained_classifier_head": True,
        "usable_by_core_candidate": bool(usable),
        "accuracy_margin": 0.02,
        "nll_improvement_margin": 0.0,
        "max_ece": 0.10,
        "minimum_tasks": 20,
        "minimum_examples": 640,
        "minimum_examples_per_task": 32,
        "minimum_classes": 2,
        "requested_minimum_tasks": 20,
        "requested_minimum_examples": 640,
        "requested_minimum_examples_per_task": 32,
        "requested_minimum_classes": 2,
        "checks": checks,
        "failed_checks": failed_checks,
        "reason": (
            "heldout_trust_gate_passed"
            if usable
            else "heldout_trust_gate_failed:accuracy_margin"
        ),
        "checkpoint_state_sha256": state_sha256,
        "split_assignment_sha256": split_sha256,
        "row_set_sha256": row_sha256,
        "feature_schema_sha256": feature_sha256,
        "model_input_content_sha256": "5" * 64,
        "effective_serving_config": effective_serving_config,
        "effective_serving_config_sha256": effective_serving_config_sha256,
        "metrics": {
            "task_count": 20,
            "example_count": 640,
            "min_task_examples": 32,
            "min_episode_classes": 2,
            "ece_bins": 10,
            "task_admission_bootstrap": bootstrap,
            "task_metrics": task_metrics,
        },
    }
    split_record = {"assignment_sha256": split_sha256, "group_overlap_count": 0}
    source_hashes = {
        "warm_start_fs_teacher_sha256": "",
        "warm_start_classifier_sha256": "",
        "resume_checkpoint_sha256": "",
    }
    provenance = {
        "artifact_role": "final_heldout_evaluated_classifier",
        "checkpoint_state_sha256": state_sha256,
        "split_record": split_record,
        "heldout_row_set_sha256": row_sha256,
        "heldout_feature_schema_sha256": feature_sha256,
        "data_config_sha256": data_config_sha256,
        "head_nll_improvement_margin": 0.0,
        "head_max_ece": 0.10,
        "heldout_evaluator": {
            "minimum_tasks": 20,
            "minimum_examples": 640,
            "minimum_examples_per_task": 32,
            "minimum_classes": 2,
            "nll_improvement_margin": 0.0,
            "max_ece": 0.10,
        },
        "effective_serving_config": effective_serving_config,
        "effective_serving_config_sha256": effective_serving_config_sha256,
        **source_hashes,
    }
    summary = {
        "head_trust_record": head_trust,
        "effective_serving_config": effective_serving_config,
        "effective_serving_config_sha256": effective_serving_config_sha256,
        "checkpoint_provenance": provenance,
        "split_record": split_record,
    }
    checkpoint = root / "fs_classifier.pt"
    torch.save(
        {
            "checkpoint_format": "classifier",
            "artifact_role": "final_heldout_evaluated_classifier",
            "config": cfg.__dict__,
            "model_state_dict": state,
            "summary": summary,
            "provenance": provenance,
            "head_trust_record": head_trust,
            "effective_serving_config": effective_serving_config,
            "effective_serving_config_sha256": effective_serving_config_sha256,
        },
        checkpoint,
    )
    checkpoint_sha256 = _sha256_file(checkpoint)
    trust_record = {
        **head_trust,
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    trust_path = root / "classifier_trust_record.json"
    trust_path.write_text(json.dumps(trust_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "tabentics_diakrino_classifier_artifacts_v1",
        "checkpoint": {
            "path": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
            "state_sha256": state_sha256,
            "format": "classifier",
            "artifact_role": "final_heldout_evaluated_classifier",
        },
        "trust_record": {"path": trust_path.name, "sha256": _sha256_file(trust_path)},
        "summary": {"path": summary_path.name, "sha256": _sha256_file(summary_path)},
        "source_hashes": source_hashes,
        "data_config_sha256": data_config_sha256,
        "split_assignment_sha256": split_sha256,
        "effective_serving_config": effective_serving_config,
        "effective_serving_config_sha256": effective_serving_config_sha256,
    }
    (root / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint


def _rewrite_bundle_head_trust(root, mutator):
    torch = pytest.importorskip("torch")
    checkpoint = root / "fs_classifier.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head_trust = json.loads(json.dumps(payload["head_trust_record"]))
    mutator(head_trust)
    payload["head_trust_record"] = head_trust
    payload["summary"]["head_trust_record"] = head_trust
    torch.save(payload, checkpoint)
    trust_record = {
        **head_trust,
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    trust_path = root / "classifier_trust_record.json"
    trust_path.write_text(json.dumps(trust_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"]["bytes"] = checkpoint.stat().st_size
    manifest["checkpoint"]["sha256"] = _sha256_file(checkpoint)
    manifest["trust_record"]["sha256"] = _sha256_file(trust_path)
    manifest["summary"]["sha256"] = _sha256_file(summary_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checkpoint


def test_loader_verifies_pristine_classifier_artifact_bundle(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = _write_trusted_classifier_bundle(tmp_path)

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is True
    assert report["checkpoint_trust_bundle_reason"] == "artifact_trust_bundle_verified"
    assert report["checkpoint_trust_bundle_failures"] == []
    assert report["checkpoint_complete_state"] is True
    assert report["checkpoint_loaded_state_matches"] is True
    assert report["checkpoint_loaded_state_sha256"] == report["checkpoint_state_sha256"]
    assert report["checkpoint_comparable_for_classification"] is True
    assert report["checkpoint_usable_by_core_candidate"] is True
    assert report["checkpoint_usability_reason"] == "heldout_trust_gate_passed"
    assert report["checkpoint_sha256"] == _sha256_file(checkpoint)


def test_loader_preserves_valid_but_failed_external_trust_gate_for_diagnostics(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = _write_trusted_classifier_bundle(tmp_path, usable=False)

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is True
    assert report["checkpoint_complete_state"] is True
    assert report["checkpoint_comparable_for_classification"] is True
    assert report["checkpoint_usable_by_core_candidate"] is False
    assert report["checkpoint_usability_reason"] == "heldout_trust_gate_failed:accuracy_margin"


def test_loader_rejects_legacy_point_estimate_trust_schema(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = _write_trusted_classifier_bundle(tmp_path)
    trust_path = tmp_path / "classifier_trust_record.json"
    trust = json.loads(trust_path.read_text(encoding="utf-8"))
    trust["schema_version"] = "tabentics_diakrino_classifier_head_trust_v2"
    trust_path.write_text(json.dumps(trust, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trust_record"]["sha256"] = _sha256_file(trust_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is False
    assert report["checkpoint_comparable_for_classification"] is False
    assert "trust_record_schema_mismatch" in report["checkpoint_trust_bundle_failures"]


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        ("bootstrap_samples", "task_admission_bootstrap_contract_mismatch"),
        ("nll_bound", "task_admission_statistics_mismatch"),
        ("ece_bound", "task_admission_statistics_mismatch"),
        ("task_examples", "trust_record_statistical_checks_inconsistent"),
        ("ece_bins", "task_admission_bootstrap_contract_mismatch"),
        ("max_ece", "admission_thresholds_mismatch"),
        ("oversized_max_ece", "admission_thresholds_mismatch"),
    ],
)
def test_loader_recomputes_v3_statistical_evidence(tmp_path, mutation, expected_failure):
    torch = pytest.importorskip("torch")
    _write_trusted_classifier_bundle(tmp_path)

    def mutate(trust):
        metrics = trust["metrics"]
        bootstrap = metrics["task_admission_bootstrap"]
        if mutation == "bootstrap_samples":
            bootstrap["samples"] = 9999
        elif mutation == "nll_bound":
            bootstrap["nll_improvement_vs_chance"]["lower_quantile_value"] = 0.2
        elif mutation == "ece_bound":
            bootstrap["expected_calibration_error"]["upper_quantile_value"] = 0.2
        elif mutation == "task_examples":
            metrics["task_metrics"][0]["examples"] = 31
        elif mutation == "ece_bins":
            metrics["ece_bins"] = 9
        elif mutation == "max_ece":
            trust["max_ece"] = None
        elif mutation == "oversized_max_ece":
            trust["max_ece"] = 10**400
        else:  # pragma: no cover
            raise AssertionError(mutation)

    checkpoint = _rewrite_bundle_head_trust(tmp_path, mutate)
    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is False
    assert report["checkpoint_comparable_for_classification"] is False
    assert expected_failure in report["checkpoint_trust_bundle_failures"]


@pytest.mark.parametrize(
    ("artifact_name", "expected_reason"),
    [
        ("artifact_manifest.json", "artifact_manifest_missing"),
        ("classifier_trust_record.json", "classifier_trust_record_missing"),
    ],
)
def test_loader_rejects_missing_classifier_trust_bundle_artifact(tmp_path, artifact_name, expected_reason):
    torch = pytest.importorskip("torch")
    checkpoint = _write_trusted_classifier_bundle(tmp_path)
    (tmp_path / artifact_name).unlink()

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is False
    assert report["checkpoint_comparable_for_classification"] is False
    assert report["checkpoint_usable_by_core_candidate"] is False
    assert report["checkpoint_trust_bundle_reason"] == expected_reason


@pytest.mark.parametrize(
    ("artifact_name", "expected_reason"),
    [
        ("artifact_manifest.json", "artifact_manifest_malformed"),
        ("classifier_trust_record.json", "classifier_trust_record_malformed"),
    ],
)
def test_loader_rejects_malformed_classifier_trust_bundle_artifact(tmp_path, artifact_name, expected_reason):
    torch = pytest.importorskip("torch")
    checkpoint = _write_trusted_classifier_bundle(tmp_path)
    (tmp_path / artifact_name).write_text("{not-json", encoding="utf-8")

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is False
    assert report["checkpoint_comparable_for_classification"] is False
    assert report["checkpoint_trust_bundle_reason"] == expected_reason


def test_loader_rejects_checkpoint_byte_tamper(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = _write_trusted_classifier_bundle(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = "class_logit_head.1.weight"
    payload["model_state_dict"][key] = payload["model_state_dict"][key] + 1.0
    torch.save(payload, checkpoint)

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is False
    assert report["checkpoint_comparable_for_classification"] is False
    assert "checkpoint_sha256_mismatch" in report["checkpoint_trust_bundle_failures"]
    assert "checkpoint_state_sha256_mismatch" in report["checkpoint_trust_bundle_failures"]


@pytest.mark.parametrize("artifact_name", ["artifact_manifest.json", "classifier_trust_record.json"])
def test_loader_rejects_manifest_or_trust_record_edit(tmp_path, artifact_name):
    torch = pytest.importorskip("torch")
    checkpoint = _write_trusted_classifier_bundle(tmp_path)
    artifact_path = tmp_path / artifact_name
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact_name == "artifact_manifest.json":
        artifact["checkpoint"]["sha256"] = "f" * 64
    else:
        artifact["reason"] = "heldout_trust_gate_failed:edited"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is False
    assert report["checkpoint_comparable_for_classification"] is False
    expected = "checkpoint_sha256_mismatch" if artifact_name == "artifact_manifest.json" else "trust_record_sha256_mismatch"
    assert expected in report["checkpoint_trust_bundle_failures"]


@pytest.mark.parametrize("artifact_name", ["artifact_manifest.json", "classifier_trust_record.json"])
def test_loader_rejects_swapped_manifest_or_trust_record(tmp_path, artifact_name):
    torch = pytest.importorskip("torch")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    checkpoint = _write_trusted_classifier_bundle(first_root, seed=11)
    _write_trusted_classifier_bundle(second_root, seed=13)
    shutil.copyfile(second_root / artifact_name, first_root / artifact_name)

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is False
    assert report["checkpoint_comparable_for_classification"] is False
    assert report["checkpoint_trust_bundle_failures"]


def test_loader_rejects_canonical_state_mismatch_when_file_bindings_are_updated(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = _write_trusted_classifier_bundle(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = "class_logit_head.1.bias"
    payload["model_state_dict"][key] = payload["model_state_dict"][key] + 1.0
    torch.save(payload, checkpoint)

    trust_path = tmp_path / "classifier_trust_record.json"
    trust = json.loads(trust_path.read_text(encoding="utf-8"))
    trust["checkpoint_sha256"] = _sha256_file(checkpoint)
    trust["checkpoint_bytes"] = checkpoint.stat().st_size
    trust_path.write_text(json.dumps(trust, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"]["sha256"] = _sha256_file(checkpoint)
    manifest["checkpoint"]["bytes"] = checkpoint.stat().st_size
    manifest["trust_record"]["sha256"] = _sha256_file(trust_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is False
    assert report["checkpoint_comparable_for_classification"] is False
    assert "checkpoint_sha256_mismatch" not in report["checkpoint_trust_bundle_failures"]
    assert "trust_record_sha256_mismatch" not in report["checkpoint_trust_bundle_failures"]
    assert "checkpoint_state_sha256_mismatch" in report["checkpoint_trust_bundle_failures"]


@pytest.mark.parametrize(
    ("identity_field", "expected_failure"),
    [
        ("checkpoint_state_sha256", "checkpoint_state_sha256_mismatch"),
        ("split_assignment_sha256", "split_assignment_sha256_mismatch"),
        ("row_set_sha256", "row_set_sha256_mismatch"),
        ("feature_schema_sha256", "feature_schema_sha256_mismatch"),
    ],
)
def test_loader_rejects_rehashed_external_identity_edit(tmp_path, identity_field, expected_failure):
    torch = pytest.importorskip("torch")
    checkpoint = _write_trusted_classifier_bundle(tmp_path)
    trust_path = tmp_path / "classifier_trust_record.json"
    trust = json.loads(trust_path.read_text(encoding="utf-8"))
    trust[identity_field] = "e" * 64
    trust_path.write_text(json.dumps(trust, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trust_record"]["sha256"] = _sha256_file(trust_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is False
    assert report["checkpoint_comparable_for_classification"] is False
    assert "trust_record_sha256_mismatch" not in report["checkpoint_trust_bundle_failures"]
    assert expected_failure in report["checkpoint_trust_bundle_failures"]


@pytest.mark.parametrize("state_variant", ["missing", "shape"])
def test_loader_rejects_trusted_bundle_with_incomplete_classifier_state(tmp_path, state_variant):
    torch = pytest.importorskip("torch")
    checkpoint = _write_trusted_classifier_bundle(tmp_path, state_variant=state_variant)

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_trust_bundle_verified"] is True
    assert report["checkpoint_complete_state"] is False
    assert report["checkpoint_comparable_for_classification"] is False
    assert report["checkpoint_usable_by_core_candidate"] is False
    assert report["checkpoint_comparability_reason"] == "classifier_checkpoint_incomplete_state"


def test_loader_marks_unbundled_classifier_checkpoint_as_noncomparable(tmp_path):
    torch = pytest.importorskip("torch")
    from tabnetics.classification.tabentics_diakrino_fs_classifier import TabenticsDiakrinoFSClassifier

    cfg = _tiny_native_classifier_config()
    checkpoint = tmp_path / "classifier.pt"
    model = TabenticsDiakrinoFSClassifier(cfg)
    torch.save(
        {
            "checkpoint_format": "classifier",
            "model_config": cfg.__dict__,
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_format"] == "classifier"
    assert report["checkpoint_comparable_for_classification"] is False
    assert report["checkpoint_comparability_reason"] == "artifact_manifest_missing"
    assert report["checkpoint_usable_by_core_candidate"] is False
    assert report["checkpoint_usability_reason"] == "artifact_manifest_missing"
    assert report["checkpoint_trust_bundle_verified"] is False
    assert report["loaded_classifier_head_count"] > 0


def test_loader_rebuilds_classifier_feature_selector_from_saved_config(tmp_path):
    torch = pytest.importorskip("torch")
    from tabnetics.classification.tabentics_diakrino_fs_classifier import TabenticsDiakrinoFSClassifier
    from tabnetics.classification.tabentics_diakrino_fs_teacher import (
        TabenticsDiakrinoFSTeacher,
        TabenticsDiakrinoFSTeacherConfig,
    )

    cfg = _tiny_native_classifier_config()
    teacher_cfg = TabenticsDiakrinoFSTeacherConfig(
        d_model=16,
        n_heads=2,
        context_layers=1,
        ffn_expansion=1,
        dropout=0.0,
        max_classes=3,
        max_feature_tokens=8,
        use_distribution_series=False,
        series_samples=4,
        refiner_steps=0,
        joint_sample_mode="induced",
        joint_sample_size=4,
        joint_sample_layers=1,
        joint_sample_width=16,
        joint_sample_induced_points=4,
        joint_sample_heads=2,
        joint_sample_scale_init=0.1,
        joint_cell_fourier_bands=1,
        conformal_head_mode="scores",
    )
    checkpoint = tmp_path / "classifier_vnext_selector.pt"
    model = TabenticsDiakrinoFSClassifier(cfg)
    model.feature_selector = TabenticsDiakrinoFSTeacher(teacher_cfg)
    state = model.state_dict()
    torch.save(
        {
            "checkpoint_format": "classifier",
            "config": cfg.__dict__,
            "feature_selector_config": teacher_cfg.__dict__,
            "model_state_dict": state,
            "head_trust_record": {
                "usable_by_core_candidate": True,
                "reason": "clears_predeclared_margin",
            },
        },
        checkpoint,
    )

    loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert loaded.feature_selector.joint_sample_encoder is not None
    assert loaded.feature_selector.conformal_head is not None
    assert report["checkpoint_format"] == "classifier"
    assert report["checkpoint_comparable_for_classification"] is False
    assert report["checkpoint_usable_by_core_candidate"] is False
    assert report["checkpoint_usability_reason"] == "artifact_manifest_missing"
    assert report["embedded_head_trust_record"]["usable_by_core_candidate"] is True
    assert report["loaded_classifier_head_count"] > 0
    assert report["loaded_count"] == len(state)


def test_loader_hard_labels_fs_teacher_checkpoint_as_non_comparable(tmp_path):
    torch = pytest.importorskip("torch")
    from tabnetics.classification.tabentics_diakrino_fs_teacher import TabenticsDiakrinoFSTeacher

    cfg = _tiny_native_classifier_config()
    checkpoint = tmp_path / "fs_teacher.pt"
    teacher = TabenticsDiakrinoFSTeacher(cfg.fs_teacher_config())
    torch.save(
        {
            "checkpoint_format": "fs_teacher",
            "model_config": cfg.__dict__,
            "model_state_dict": teacher.state_dict(),
        },
        checkpoint,
    )

    _loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert report["checkpoint_format"] == "fs_teacher"
    assert report["checkpoint_comparable_for_classification"] is False
    assert report["checkpoint_comparability_reason"] == "fs_teacher_checkpoint_without_trained_classifier_head"
    assert report["loaded_classifier_head_count"] == 0


def test_predict_helper_rejects_fs_teacher_only_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from tabnetics.classification.tabentics_diakrino_fs_teacher import TabenticsDiakrinoFSTeacher

    cfg = _tiny_native_classifier_config()
    checkpoint = tmp_path / "fs_teacher_only.pt"
    teacher = TabenticsDiakrinoFSTeacher(cfg.fs_teacher_config())
    torch.save(
        {
            "checkpoint_format": "fs_teacher",
            "model_config": cfg.__dict__,
            "model_state_dict": teacher.state_dict(),
        },
        checkpoint,
    )
    X_train = np.arange(48, dtype=np.float32).reshape(12, 4)
    y_train = np.array([0, 1] * 6, dtype=int)
    X_query = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="not comparable for classification"):
        predict_tabentics_diakrino_proba(
            X_train,
            y_train,
            X_query,
            dataset_name="non_comparable_guard",
            seed=0,
            checkpoint=checkpoint,
            max_features=4,
            batch_size=2,
            device="cpu",
        )


def test_loader_rebuilds_fs_teacher_geometry_from_checkpoint_config(tmp_path):
    torch = pytest.importorskip("torch")
    from tabnetics.classification.tabentics_diakrino_fs_teacher import (
        TabenticsDiakrinoFSTeacher,
        TabenticsDiakrinoFSTeacherConfig,
    )

    cfg = TabenticsDiakrinoFSTeacherConfig(
        d_model=16,
        n_heads=2,
        context_layers=1,
        ffn_expansion=1,
        dropout=0.0,
        max_classes=3,
        max_feature_tokens=8,
        use_distribution_series=False,
        series_samples=4,
        refiner_steps=0,
        joint_sample_mode="induced",
        joint_sample_size=4,
        joint_sample_layers=1,
        joint_sample_width=16,
        joint_sample_induced_points=4,
        joint_sample_heads=2,
        joint_sample_scale_init=0.1,
        joint_cell_fourier_bands=1,
        conformal_head_mode="scores",
    )
    teacher = TabenticsDiakrinoFSTeacher(cfg)
    checkpoint = tmp_path / "vnext_teacher.pt"
    state = teacher.state_dict()
    torch.save(
        {
            "checkpoint_format": "fs_teacher",
            "model_config": cfg.__dict__,
            "model_state_dict": state,
        },
        checkpoint,
    )

    loaded, report = load_tabentics_diakrino_fs_classifier(checkpoint, map_location=torch.device("cpu"))

    assert loaded.feature_selector.joint_sample_encoder is not None
    assert loaded.feature_selector.conformal_head is not None
    assert report["checkpoint_comparable_for_classification"] is False
    assert report["loaded_count"] == len(state)
    assert report["discarded_count"] == 0


def test_predict_helper_rejects_classifier_checkpoint_without_trust_record(tmp_path):
    torch = pytest.importorskip("torch")
    from tabnetics.classification.tabentics_diakrino_fs_classifier import TabenticsDiakrinoFSClassifier

    cfg = _tiny_native_classifier_config()
    checkpoint = tmp_path / "untrusted_classifier.pt"
    model = TabenticsDiakrinoFSClassifier(cfg)
    torch.save(
        {
            "checkpoint_format": "classifier",
            "model_config": cfg.__dict__,
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    X_train = np.arange(48, dtype=np.float32).reshape(12, 4)
    y_train = np.array([0, 1] * 6, dtype=int)
    X_query = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="not comparable for classification"):
        predict_tabentics_diakrino_proba(
            X_train,
            y_train,
            X_query,
            dataset_name="untrusted_guard",
            seed=0,
            checkpoint=checkpoint,
            max_features=4,
            batch_size=2,
            device="cpu",
        )


def test_native_estimator_tiny_checkpoint_predicts_finite_calibrated_probabilities(tmp_path):
    pytest.importorskip("torch")
    from sklearn.metrics import log_loss

    checkpoint = _write_trusted_classifier_bundle(tmp_path)
    rng = np.random.default_rng(19)
    X = rng.normal(size=(24, 6)).astype(np.float32)
    y = np.asarray([0, 1] * 12, dtype=int)
    X[y == 1, :2] += 0.5
    clf = TabenticsDiakrinoNativeClassifier(
        checkpoint=str(checkpoint),
        max_features=6,
        batch_size=4,
        device="cpu",
        calibrate_probabilities=True,
        calibration_fraction=0.25,
        calibration_min_samples=8,
        random_state=3,
    )

    clf.fit(X, y)
    proba = clf.predict_proba(X[:8])

    assert proba.shape == (8, 2)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(log_loss(y[:8], proba, labels=[0, 1]))
    assert clf.calibration_meta_["native_diakrino_probability_calibration"] == "temperature_holdout"
    assert clf.native_diakrino_meta_["native_diakrino_checkpoint_comparable"] is True
    assert clf.native_diakrino_meta_["native_diakrino_checkpoint_usable_by_core_candidate"] is True
    assert clf.native_diakrino_meta_["native_diakrino_checkpoint_usability_reason"] == "heldout_trust_gate_passed"
    assert clf.native_diakrino_meta_["native_diakrino_checkpoint_trust_bundle_verified"] is True
    assert clf.native_diakrino_meta_["native_diakrino_checkpoint_complete_state"] is True


def test_predict_helper_cached_support_matches_legacy_and_stamps_contract(tmp_path):
    checkpoint = _write_trusted_classifier_bundle(tmp_path)
    rng = np.random.default_rng(198)
    X_train = rng.normal(size=(513, 6)).astype(np.float32)
    y_train = np.arange(513, dtype=int) % 2
    X_train[0, 0] = np.nan
    X_query = rng.normal(size=(8, 6)).astype(np.float32)
    X_query[-1, -1] = np.nan

    legacy, legacy_meta, _ = predict_tabentics_diakrino_proba(
        X_train,
        y_train,
        X_query,
        dataset_name="cache_contract",
        seed=198,
        checkpoint=checkpoint,
        max_features=6,
        batch_size=4,
        device="cpu",
    )
    cached, cached_meta, _ = predict_tabentics_diakrino_proba(
        X_train,
        y_train,
        X_query,
        dataset_name="cache_contract",
        seed=198,
        checkpoint=checkpoint,
        max_features=6,
        batch_size=4,
        device="cpu",
        support_joint_serving_cache=True,
        retry_cuda_oom_microbatch=True,
    )

    np.testing.assert_allclose(cached, legacy, rtol=0.0, atol=1e-7)
    assert np.array_equal(cached.argmax(axis=1), legacy.argmax(axis=1))
    assert legacy_meta["native_diakrino_support_joint_serving_cache_used"] is False
    assert cached_meta["native_diakrino_support_joint_serving_cache_requested"] is True
    assert cached_meta["native_diakrino_support_joint_serving_cache_used"] is True
    assert cached_meta["native_diakrino_cuda_oom_retry_enabled"] is True
    assert cached_meta["native_diakrino_cuda_oom_retry_count"] == 0
    assert cached_meta["native_diakrino_effective_min_batch_size"] == 4
    assert cached_meta["native_diakrino_joint_support_rows_total"] == 513
    assert cached_meta["native_diakrino_joint_support_rows_used"] == 512


def test_oom_retry_replays_only_failed_chunk_in_order():
    calls = []
    cleared = []

    def score(start, stop):
        calls.append((start, stop))
        if (start, stop) == (0, 4):
            raise RuntimeError("CUDA out of memory")
        return np.arange(start, stop, dtype=float)[:, None]

    parts, retries, minimum = _score_query_chunks_with_oom_retry(
        total_rows=6,
        requested_batch_size=4,
        score_range=score,
        retry_enabled=True,
        is_retryable_oom=lambda exc: "out of memory" in str(exc).lower(),
        clear_device_cache=lambda: cleared.append(True),
        dataset_name="ordered",
    )

    assert np.array_equal(np.concatenate(parts).ravel(), np.arange(6, dtype=float))
    assert calls == [(0, 4), (0, 1), (1, 2), (2, 3), (3, 4), (4, 6)]
    assert retries == 1
    assert minimum == 1
    assert len(cleared) == 1


def test_oom_retry_microbatch_one_failure_is_structured_and_terminal():
    def score(_start, _stop):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(TabenticsDiakrinoNativeOOMError) as captured:
        _score_query_chunks_with_oom_retry(
            total_rows=1,
            requested_batch_size=1,
            score_range=score,
            retry_enabled=True,
            is_retryable_oom=lambda _exc: True,
            clear_device_cache=lambda: None,
            dataset_name="terminal",
        )

    assert captured.value.dataset_name == "terminal"
    assert captured.value.query_start == 0
    assert captured.value.requested_batch_size == 1
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_oom_retry_does_not_swallow_non_oom_runtime_errors():
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _score_query_chunks_with_oom_retry(
            total_rows=4,
            requested_batch_size=4,
            score_range=lambda _start, _stop: (_ for _ in ()).throw(RuntimeError("shape mismatch")),
            retry_enabled=True,
            is_retryable_oom=lambda _exc: False,
            clear_device_cache=lambda: pytest.fail("cache clear must not run"),
            dataset_name="non_oom",
        )

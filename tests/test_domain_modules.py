import numpy as np

from tabnetics.domains.base import resolve_dataset_catalog_context
from tabnetics.domains.bio import (
    apply_multiomics_adapter_train_test,
    infer_prefilter_data_domain,
)
from tabnetics.domains.face import apply_face_domain_projection


def _face_like_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), 8)
    X = rng.normal(size=(24, 18))
    X[y == 0, :4] += 2.0
    X[y == 1, 4:8] += 2.0
    X[y == 2, 8:12] += 2.0
    return X, y


def _omics_like_train_test(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.tile(np.array([0, 1, 2]), 6)
    X = rng.normal(size=(18, 8))
    X[y == 0, :2] += 2.5
    X[y == 1, 2:4] += 2.5
    X[y == 2, 4:6] += 2.5
    return X[:12], y[:12], X[12:]


def test_resolve_dataset_catalog_context_marks_face_domain() -> None:
    context = resolve_dataset_catalog_context("orlraws10p")

    assert context.dataset_id == "orlraws10p"
    assert context.found_in_catalog is True
    assert context.is_face_domain is True


def test_infer_prefilter_data_domain_detects_rnaseq() -> None:
    assert infer_prefilter_data_domain("RNA-seq counts") == "rnaseq"
    assert infer_prefilter_data_domain("Affy HG-U133A") == "auto"


def test_apply_face_domain_projection_extracts_face_specific_state() -> None:
    X, y = _face_like_data(11)
    context = resolve_dataset_catalog_context("orlraws10p")

    result = apply_face_domain_projection(
        X_train_imp=X[:18],
        y_train=y[:18],
        X_test_imp=X[18:],
        enabled=True,
        dataset_name="orlraws10p",
        dataset_context=context,
        seed=11,
    )

    assert result.meta["face_projection_applied"] is True
    assert result.meta["face_projection_is_face_domain"] is True
    assert result.state.pca_model is not None
    assert result.X_train.shape[1] >= 1
    assert np.all(np.isfinite(result.X_train))
    assert np.all(np.isfinite(result.X_test))


def test_apply_multiomics_adapter_train_test_appends_latent_features() -> None:
    X_train, y_train, X_test = _omics_like_train_test(7)

    X_train_out, X_test_out, meta = apply_multiomics_adapter_train_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        adapter_mode="split_halves",
        integrator="mb_plsda",
        n_components=2,
    )

    assert meta["multiomics_adapter_applied"] is True
    assert int(meta["multiomics_latent_dim"]) >= 1
    assert X_train_out.shape[1] > X_train.shape[1]
    assert X_test_out.shape[1] == X_train_out.shape[1]
    assert np.all(np.isfinite(X_train_out))
    assert np.all(np.isfinite(X_test_out))


def test_apply_multiomics_adapter_train_test_uses_metadata_blocks() -> None:
    X_train, y_train, X_test = _omics_like_train_test(9)

    X_train_out, X_test_out, meta = apply_multiomics_adapter_train_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        adapter_mode="metadata_blocks",
        integrator="mb_plsda",
        n_components=2,
        feature_blocks={
            "transcriptomics": (0, 1, 2, 3),
            "proteomics": (4, 5, 6, 7),
        },
    )

    assert meta["multiomics_adapter_applied"] is True
    assert meta["multiomics_feature_blocks_available"] is True
    assert meta["multiomics_block_names"] == ["transcriptomics", "proteomics"]
    assert meta["multiomics_block_sizes"] == [4, 4]
    assert X_train_out.shape[1] > X_train.shape[1]
    assert X_test_out.shape[1] == X_train_out.shape[1]
    assert np.all(np.isfinite(X_train_out))
    assert np.all(np.isfinite(X_test_out))


def test_apply_multiomics_adapter_train_test_metadata_blocks_missing_is_noop() -> None:
    X_train, y_train, X_test = _omics_like_train_test(13)

    X_train_out, X_test_out, meta = apply_multiomics_adapter_train_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        adapter_mode="metadata_blocks",
        integrator="mb_plsda",
        n_components=2,
        feature_blocks={"transcriptomics": (0, 1, 2, 3)},
    )

    assert meta["multiomics_adapter_applied"] is False
    assert meta["multiomics_adapter_reason"] == "metadata_blocks_missing"
    assert meta["multiomics_feature_blocks_available"] is False
    assert X_train_out.shape == X_train.shape
    assert X_test_out.shape == X_test.shape

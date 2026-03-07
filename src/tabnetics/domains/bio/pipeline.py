"""Bioinformatics-specific pipeline helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def infer_prefilter_data_domain(platform: Any) -> str:
    """Infer whether the benchmark prefilter should use RNA-seq handling."""
    text = str(platform or "").strip().lower()
    if "rna" in text:
        return "rnaseq"
    return "auto"


def apply_multiomics_adapter_train_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    adapter_mode: str,
    integrator: str,
    n_components: int,
    batch_labels_train: Optional[np.ndarray] = None,
    batch_labels_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply the benchmark-style split-halves multiomics adapter to train/test data."""
    mode = str(adapter_mode or "none").strip().lower()
    integrator_mode = str(integrator or "mb_plsda").strip().lower()
    meta: Dict[str, Any] = {
        "multiomics_adapter_mode": str(mode),
        "multiomics_integrator": str(integrator_mode),
        "multiomics_adapter_applied": False,
        "multiomics_adapter_reason": "disabled",
        "multiomics_n_blocks": 0,
        "multiomics_block_sizes": [],
        "multiomics_latent_dim": 0,
    }
    X_train_arr = np.asarray(X_train, dtype=float)
    X_test_arr = np.asarray(X_test, dtype=float)
    y_train_arr = np.asarray(y_train).ravel()
    if mode == "none":
        return X_train_arr, X_test_arr, meta
    if X_train_arr.ndim != 2 or X_test_arr.ndim != 2:
        meta["multiomics_adapter_reason"] = "invalid_matrix_shape"
        return X_train_arr, X_test_arr, meta
    if X_train_arr.shape[1] != X_test_arr.shape[1]:
        meta["multiomics_adapter_reason"] = "feature_dim_mismatch"
        return X_train_arr, X_test_arr, meta
    if X_train_arr.shape[1] < 4:
        meta["multiomics_adapter_reason"] = "too_few_features"
        return X_train_arr, X_test_arr, meta
    if mode != "split_halves":
        meta["multiomics_adapter_reason"] = "unsupported_mode"
        return X_train_arr, X_test_arr, meta
    if integrator_mode not in {"mb_plsda", "mint"}:
        integrator_mode = "mb_plsda"
        meta["multiomics_integrator"] = str(integrator_mode)

    mid = int(X_train_arr.shape[1] // 2)
    if mid <= 0 or mid >= int(X_train_arr.shape[1]):
        meta["multiomics_adapter_reason"] = "invalid_split"
        return X_train_arr, X_test_arr, meta

    train_blocks = [
        (np.asarray(X_train_arr[:, :mid], dtype=float), "omics_block_a"),
        (np.asarray(X_train_arr[:, mid:], dtype=float), "omics_block_b"),
    ]
    test_blocks = [
        (np.asarray(X_test_arr[:, :mid], dtype=float), "omics_block_a"),
        (np.asarray(X_test_arr[:, mid:], dtype=float), "omics_block_b"),
    ]
    meta["multiomics_n_blocks"] = 2
    meta["multiomics_block_sizes"] = [
        int(train_blocks[0][0].shape[1]),
        int(train_blocks[1][0].shape[1]),
    ]

    n_comp = int(max(1, min(int(n_components), 8, max(1, int(X_train_arr.shape[0]) - 1))))
    try:
        try:
            from tabnetics.multiomics.integration import MINTIntegrator, MultiBlockPLSDA
        except Exception:
            from tabnetics.multiomics.integration import MINTIntegrator, MultiBlockPLSDA  # type: ignore

        if integrator_mode == "mb_plsda":
            model = MultiBlockPLSDA(n_components=n_comp)
            latent_train = np.asarray(model.fit_transform(train_blocks, y_train_arr), dtype=float)
            latent_test = np.asarray(model.transform(test_blocks), dtype=float)
        else:
            if batch_labels_train is None or batch_labels_test is None:
                meta["multiomics_adapter_reason"] = "mint_requires_batch_labels"
                return X_train_arr, X_test_arr, meta
            train_studies = np.asarray(batch_labels_train, dtype=object).ravel()
            test_studies = np.asarray(batch_labels_test, dtype=object).ravel()
            if int(np.unique(train_studies).size) < 2:
                meta["multiomics_adapter_reason"] = "mint_requires_multiple_batches"
                return X_train_arr, X_test_arr, meta
            model = MINTIntegrator(n_components=n_comp)
            model.fit(train_blocks, y_train_arr, study_labels=train_studies)
            latent_train = np.asarray(
                model.transform(train_blocks, study_labels=train_studies),
                dtype=float,
            )
            latent_test = np.asarray(
                model.transform(test_blocks, study_labels=test_studies),
                dtype=float,
            )

        if (
            latent_train.ndim != 2
            or latent_test.ndim != 2
            or latent_train.shape[0] != X_train_arr.shape[0]
            or latent_test.shape[0] != X_test_arr.shape[0]
        ):
            meta["multiomics_adapter_reason"] = "latent_transform_failed"
            return X_train_arr, X_test_arr, meta
        X_train_out = np.hstack([X_train_arr, latent_train])
        X_test_out = np.hstack([X_test_arr, latent_test])
        meta["multiomics_adapter_applied"] = True
        meta["multiomics_adapter_reason"] = "ok"
        meta["multiomics_latent_dim"] = int(latent_train.shape[1])
        meta["multiomics_output_features"] = int(X_train_out.shape[1])
        return X_train_out, X_test_out, meta
    except Exception as exc:
        meta["multiomics_adapter_reason"] = f"adapter_error:{type(exc).__name__}"
        return X_train_arr, X_test_arr, meta

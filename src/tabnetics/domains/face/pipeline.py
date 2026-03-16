"""Face-domain pipeline helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tabnetics.domains.base import DatasetDomainContext


@dataclass
class FaceProjectionState:
    """Fitted state needed to replay face-domain transforms at inference time."""

    pca_model: Optional[Any]
    lda_model: Optional[Any]
    dataset_name: str
    seed: int


@dataclass
class FaceProjectionResult:
    """Output bundle for the optional face-domain projection stage."""

    X_train: np.ndarray
    X_test: np.ndarray
    meta: Dict[str, Any]
    state: FaceProjectionState


def apply_face_domain_projection(
    X_train_imp: np.ndarray,
    y_train: np.ndarray,
    X_test_imp: np.ndarray,
    *,
    enabled: bool,
    dataset_name: str,
    dataset_context: DatasetDomainContext,
    seed: int,
) -> FaceProjectionResult:
    """Apply the opt-in Fisherfaces-style face projection when appropriate."""
    state = FaceProjectionState(
        pca_model=None,
        lda_model=None,
        dataset_name=str(dataset_name),
        seed=int(seed),
    )
    meta: Dict[str, Any] = {
        "enable_face_domain_projection": bool(enabled),
        "face_projection_applied": False,
        "face_projection_mode": "none",
        "face_projection_reason": "disabled",
        "face_projection_dataset_id": str(dataset_context.dataset_id),
        "face_projection_domain": str(dataset_context.domain),
        "face_projection_is_face_domain": bool(dataset_context.is_face_domain),
        "face_projection_found_in_catalog": bool(dataset_context.found_in_catalog),
        "face_projection_pca_components": 0,
        "face_projection_lda_components": 0,
        "face_projection_output_dim": int(X_train_imp.shape[1]) if X_train_imp.ndim == 2 else 0,
    }
    if not bool(enabled):
        return FaceProjectionResult(
            X_train=np.asarray(X_train_imp, dtype=float),
            X_test=np.asarray(X_test_imp, dtype=float),
            meta=meta,
            state=state,
        )

    if not bool(dataset_context.is_face_domain):
        meta["face_projection_reason"] = "not_face_domain"
        return FaceProjectionResult(
            X_train=np.asarray(X_train_imp, dtype=float),
            X_test=np.asarray(X_test_imp, dtype=float),
            meta=meta,
            state=state,
        )

    y_arr = np.asarray(y_train)
    classes = np.unique(y_arr)
    n_classes = int(classes.size)
    if n_classes < 2:
        meta["face_projection_reason"] = "insufficient_classes"
        return FaceProjectionResult(
            X_train=np.asarray(X_train_imp, dtype=float),
            X_test=np.asarray(X_test_imp, dtype=float),
            meta=meta,
            state=state,
        )

    n_train = int(X_train_imp.shape[0])
    n_features = int(X_train_imp.shape[1])
    pca_components = int(min(n_features, max(1, n_train - n_classes)))
    if pca_components <= 0:
        meta["face_projection_reason"] = "invalid_pca_components"
        return FaceProjectionResult(
            X_train=np.asarray(X_train_imp, dtype=float),
            X_test=np.asarray(X_test_imp, dtype=float),
            meta=meta,
            state=state,
        )

    try:
        pca = PCA(
            n_components=pca_components,
            svd_solver="full",
            random_state=int(seed),
        )
        X_train_pca = np.asarray(pca.fit_transform(X_train_imp), dtype=float)
        X_test_pca = np.asarray(pca.transform(X_test_imp), dtype=float)
        state.pca_model = pca
    except Exception:
        meta["face_projection_reason"] = "pca_failed"
        return FaceProjectionResult(
            X_train=np.asarray(X_train_imp, dtype=float),
            X_test=np.asarray(X_test_imp, dtype=float),
            meta=meta,
            state=state,
        )

    meta["face_projection_applied"] = True
    meta["face_projection_mode"] = "pca_only"
    meta["face_projection_reason"] = "pca_only"
    meta["face_projection_pca_components"] = int(X_train_pca.shape[1])
    meta["face_projection_output_dim"] = int(X_train_pca.shape[1])

    lda_components = int(min(max(1, n_classes - 1), X_train_pca.shape[1]))
    if lda_components <= 0:
        return FaceProjectionResult(
            X_train=X_train_pca,
            X_test=X_test_pca,
            meta=meta,
            state=state,
        )

    try:
        lda = LinearDiscriminantAnalysis()
        X_train_lda = np.asarray(lda.fit_transform(X_train_pca, y_arr), dtype=float)
        X_test_lda = np.asarray(lda.transform(X_test_pca), dtype=float)
        if X_train_lda.ndim == 1:
            X_train_lda = X_train_lda.reshape(-1, 1)
        if X_test_lda.ndim == 1:
            X_test_lda = X_test_lda.reshape(-1, 1)
        if X_train_lda.shape[1] <= 0:
            return FaceProjectionResult(
                X_train=X_train_pca,
                X_test=X_test_pca,
                meta=meta,
                state=state,
            )
        meta["face_projection_mode"] = "pca_lda"
        meta["face_projection_reason"] = "pca_lda"
        meta["face_projection_lda_components"] = int(X_train_lda.shape[1])
        meta["face_projection_output_dim"] = int(X_train_lda.shape[1])
        state.lda_model = lda
        return FaceProjectionResult(
            X_train=X_train_lda,
            X_test=X_test_lda,
            meta=meta,
            state=state,
        )
    except Exception:
        meta["face_projection_reason"] = "lda_failed_fallback_pca"
        return FaceProjectionResult(
            X_train=X_train_pca,
            X_test=X_test_pca,
            meta=meta,
            state=state,
        )

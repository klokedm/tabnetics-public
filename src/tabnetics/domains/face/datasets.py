"""Face-domain dataset builders."""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, TypeVar

import numpy as np
from scipy import ndimage
from sklearn.datasets import fetch_olivetti_faces


logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@dataclass
class FaceProxyDataset:
    """Loaded face-domain proxy dataset in tabular form."""

    X: np.ndarray
    y: np.ndarray
    data_source: str
    notes: str = ""


def _retry_with_backoff(
    fn: Callable[[], _T],
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    label: str = "network_call",
) -> _T:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (OSError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None) or getattr(response, "status", None)
            if status is not None and 400 <= int(status) < 500:
                raise
            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    "[%s] attempt %d/%d failed (%s); retrying in %.1fs...",
                    label,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error("[%s] all %d attempts failed; last error: %s", label, max_retries + 1, exc)
    assert last_exc is not None
    raise last_exc


def _resize_face_image(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    src = np.asarray(image, dtype=float)
    if src.ndim != 2:
        side = int(round(np.sqrt(float(src.size))))
        if side <= 0 or side * side != int(src.size):
            raise RuntimeError(f"Cannot infer 2D image shape from vector length {src.size}.")
        src = src.reshape(side, side)
    target_h = int(max(1, target_shape[0]))
    target_w = int(max(1, target_shape[1]))
    if src.shape == (target_h, target_w):
        return src
    zoom_h = float(target_h) / float(src.shape[0])
    zoom_w = float(target_w) / float(src.shape[1])
    resized = np.asarray(ndimage.zoom(src, (zoom_h, zoom_w), order=1), dtype=float)
    if resized.shape != (target_h, target_w):
        out = np.zeros((target_h, target_w), dtype=float)
        h = min(target_h, resized.shape[0])
        w = min(target_w, resized.shape[1])
        out[:h, :w] = resized[:h, :w]
        resized = out
    return resized


def load_face_proxy_dataset(spec: Any, seed: int) -> FaceProxyDataset:
    """Build a tabular face-domain proxy dataset from Olivetti faces."""
    params = dict(getattr(spec, "params", {}) or {})
    profile = dict(params.get("synthetic_profile", {}))
    n_classes = int(max(2, profile.get("n_classes", 10)))
    target_n = int(max(20, profile.get("n_samples", 100)))
    target_p = int(max(64, profile.get("n_features", 4096)))
    resize_raw = params.get("face_resize_shape", None)
    if resize_raw is None:
        side = int(round(np.sqrt(float(target_p))))
        resize_shape = (side, side)
    else:
        resize_shape = (
            int(max(1, int(resize_raw[0]))),
            int(max(1, int(resize_raw[1]))),
        )

    faces = _retry_with_backoff(
        lambda: fetch_olivetti_faces(shuffle=False, download_if_missing=True),
        max_retries=3,
        label="olivetti_faces",
    )
    images = np.asarray(faces.images, dtype=float)
    targets = np.asarray(faces.target, dtype=int)
    classes = np.unique(targets)
    dataset_id = str(getattr(spec, "dataset_id", "") or "")
    if classes.size < n_classes:
        raise RuntimeError(
            f"Olivetti proxy has {classes.size} classes, required {n_classes} for dataset {dataset_id}."
        )

    chosen = np.asarray(classes[:n_classes], dtype=int)
    rng = np.random.default_rng(int(seed) + 7919)
    per_class_target = int(max(1, int(np.ceil(float(target_n) / float(n_classes)))))

    images_accum: List[np.ndarray] = []
    labels_accum: List[int] = []
    for local_label, cls in enumerate(chosen.tolist()):
        cls_images = images[targets == cls]
        if cls_images.size == 0:
            continue
        order = np.arange(cls_images.shape[0], dtype=int)
        rng.shuffle(order)
        selected = [
            np.asarray(cls_images[idx], dtype=float)
            for idx in order[: min(per_class_target, cls_images.shape[0])]
        ]
        while len(selected) < per_class_target:
            base = np.asarray(cls_images[int(rng.integers(0, cls_images.shape[0]))], dtype=float)
            shift_r = int(rng.integers(-2, 3))
            shift_c = int(rng.integers(-2, 3))
            aug = np.roll(base, shift=(shift_r, shift_c), axis=(0, 1))
            aug = np.clip(aug + rng.normal(0.0, 0.01, size=aug.shape), 0.0, 1.0)
            selected.append(np.asarray(aug, dtype=float))

        for img in selected:
            images_accum.append(img)
            labels_accum.append(int(local_label))

    if not images_accum:
        raise RuntimeError(f"Failed to construct Olivetti proxy samples for {dataset_id}.")

    X_img = np.asarray(images_accum, dtype=float)
    y = np.asarray(labels_accum, dtype=int)
    perm = rng.permutation(X_img.shape[0])
    X_img = X_img[perm]
    y = y[perm]
    X_img = X_img[:target_n]
    y = y[:target_n]

    target_h, target_w = int(resize_shape[0]), int(resize_shape[1])
    X = np.zeros((X_img.shape[0], target_p), dtype=float)
    for i in range(X_img.shape[0]):
        resized = _resize_face_image(X_img[i], (target_h, target_w))
        flat = np.asarray(resized, dtype=float).ravel()
        if flat.size < target_p:
            pad = np.zeros((target_p - flat.size,), dtype=float)
            flat = np.concatenate([flat, pad], axis=0)
        elif flat.size > target_p:
            flat = flat[:target_p]
        X[i, :] = flat

    notes = (
        f"olivetti_proxy_faces; resize={target_h}x{target_w}; "
        f"target_n={target_n}; target_p={target_p}"
    )
    return FaceProxyDataset(
        X=X,
        y=y,
        data_source=f"sklearn:olivetti_faces_proxy:{dataset_id}",
        notes=notes,
    )

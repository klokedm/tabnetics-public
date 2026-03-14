"""Conformal prediction-set utilities for Stage-2 classifiers.

This module provides:
1. Split-conformal prediction sets (default, lightweight, model-agnostic).
2. MAPIE-style APS/RAPS integration (opt-in, requires ``mapie`` package).

Both modes use a train/calibration split from the provided training set and
return diagnostics that can be attached to benchmark artifacts without changing
default behavior.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    den = np.sum(e, axis=1, keepdims=True)
    den[~np.isfinite(den) | (den <= 0.0)] = 1.0
    return e / den


def _safe_score_matrix(model: Any, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    x = np.asarray(X, dtype=float)
    cls = np.asarray(classes).ravel()
    n = int(x.shape[0]) if x.ndim == 2 else int(np.asarray(x).shape[0])
    c = int(max(1, cls.size))

    def _normalize_rows(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=float)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, 0.0, None)
        row_sums = np.sum(out, axis=1, keepdims=True)
        bad = (~np.isfinite(row_sums)) | (row_sums <= 1e-12)
        if np.any(bad):
            out[bad.ravel()] = 1.0 / float(max(1, out.shape[1]))
            row_sums = np.sum(out, axis=1, keepdims=True)
        return out / row_sums

    if hasattr(model, "predict_proba"):
        try:
            raw = np.asarray(model.predict_proba(x), dtype=float)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            if raw.shape[0] != n:
                raise ValueError("predict_proba size mismatch")
            model_classes = np.asarray(getattr(model, "classes_", cls)).ravel()
            if raw.shape[1] == c and model_classes.size == c and np.array_equal(model_classes, cls):
                return _normalize_rows(raw)
            aligned = np.zeros((n, c), dtype=float)
            class_to_col = {k: i for i, k in enumerate(model_classes.tolist())}
            for idx, key in enumerate(cls.tolist()):
                j = class_to_col.get(key)
                if j is not None and 0 <= int(j) < raw.shape[1]:
                    aligned[:, idx] = raw[:, int(j)]
            return _normalize_rows(aligned)
        except Exception as exc:
            pass

    if hasattr(model, "decision_function"):
        try:
            decision = np.asarray(model.decision_function(x), dtype=float)
            if decision.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-decision))
                mat = np.vstack([1.0 - p1, p1]).T
                if c == 2:
                    return _normalize_rows(mat)
                # If class count mismatches (rare), fall through to one-hot fallback.
            elif decision.ndim == 2 and decision.shape[0] == n:
                if decision.shape[1] == c:
                    return _normalize_rows(_softmax(decision))
        except Exception as exc:
            pass

    try:
        pred = np.asarray(model.predict(x)).ravel()
    except Exception as exc:
        pred = np.asarray([], dtype=cls.dtype)
    out = np.zeros((n, c), dtype=float)
    idx_map = {k: i for i, k in enumerate(cls.tolist())}
    for i in range(min(n, pred.size)):
        j = idx_map.get(pred[i])
        if j is None:
            j = 0
        out[i, int(j)] = 1.0
    row_sums = np.sum(out, axis=1, keepdims=True)
    zero_rows = row_sums.ravel() <= 0.0
    if np.any(zero_rows):
        out[zero_rows] = 1.0 / float(max(1, c))
    return out


def _quantile_higher(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    qq = float(np.clip(q, 0.0, 1.0))
    try:
        return float(np.quantile(arr, qq, method="higher"))
    except TypeError:  # numpy<1.22
        return float(np.quantile(arr, qq, interpolation="higher"))


def compute_split_conformal_sets(
    *,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: Optional[np.ndarray] = None,
    alpha: float = 0.10,
    calibration_fraction: float = 0.25,
    min_calibration: int = 20,
    seed: int = 0,
    include_prediction_sets: bool = False,
) -> Dict[str, Any]:
    """Compute split-conformal prediction sets and summary diagnostics."""
    x_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train).ravel()
    x_ev = np.asarray(X_eval, dtype=float)
    y_ev = None if y_eval is None else np.asarray(y_eval).ravel()

    alpha_val = float(np.clip(alpha, 1e-4, 0.49))
    cal_frac = float(np.clip(calibration_fraction, 0.05, 0.95))
    min_cal = int(max(2, min_calibration))

    out: Dict[str, Any] = {
        "classifier_conformal_enabled": True,
        "classifier_conformal_applied": False,
        "classifier_conformal_skip_reason": "",
        "classifier_conformal_alpha": float(alpha_val),
        "classifier_conformal_calibration_fraction": float(cal_frac),
        "classifier_conformal_min_calibration": int(min_cal),
        "classifier_conformal_calibration_size": 0,
        "classifier_conformal_fit_size": 0,
        "classifier_conformal_qhat": float("nan"),
        "classifier_conformal_threshold": float("nan"),
        "classifier_conformal_set_size_mean": float("nan"),
        "classifier_conformal_set_size_median": float("nan"),
        "classifier_conformal_singleton_rate": float("nan"),
        "classifier_conformal_empty_set_rate": float("nan"),
        "classifier_conformal_coverage": float("nan"),
        "classifier_conformal_classes": [],
        "classifier_conformal_prediction_sets": [],
    }

    if x_tr.ndim != 2 or x_ev.ndim != 2:
        out["classifier_conformal_skip_reason"] = "invalid_input_rank"
        return out
    if x_tr.shape[1] != x_ev.shape[1]:
        out["classifier_conformal_skip_reason"] = "feature_dim_mismatch"
        return out
    if y_tr.size != x_tr.shape[0]:
        out["classifier_conformal_skip_reason"] = "train_label_mismatch"
        return out

    classes = np.unique(y_tr)
    n_classes = int(classes.size)
    if n_classes < 2:
        out["classifier_conformal_skip_reason"] = "single_class"
        return out
    out["classifier_conformal_classes"] = [str(c) for c in classes.tolist()]

    n_train = int(x_tr.shape[0])
    min_required = int(max(min_cal + n_classes, 2 * n_classes + 2))
    if n_train < min_required:
        out["classifier_conformal_skip_reason"] = "insufficient_train_samples"
        return out

    cal_size = int(max(min_cal, round(cal_frac * float(n_train))))
    cal_size = int(min(cal_size, n_train - max(2, n_classes)))
    if cal_size < min_cal or cal_size < n_classes:
        out["classifier_conformal_skip_reason"] = "insufficient_calibration_size"
        return out

    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=int(cal_size), random_state=int(seed))
        tr_idx, cal_idx = next(splitter.split(np.zeros((n_train, 1)), y_tr))
    except Exception as exc:
        out["classifier_conformal_skip_reason"] = "stratified_split_failed"
        return out

    x_fit = x_tr[np.asarray(tr_idx, dtype=int)]
    y_fit = y_tr[np.asarray(tr_idx, dtype=int)]
    x_cal = x_tr[np.asarray(cal_idx, dtype=int)]
    y_cal = y_tr[np.asarray(cal_idx, dtype=int)]
    if np.unique(y_fit).size < 2:
        out["classifier_conformal_skip_reason"] = "fit_split_single_class"
        return out

    try:
        est = clone(model)
        est.fit(x_fit, y_fit)
    except Exception as exc:
        out["classifier_conformal_skip_reason"] = "model_refit_failed"
        return out

    classes_fit = np.asarray(getattr(est, "classes_", np.unique(y_fit))).ravel()
    if classes_fit.size < 2:
        out["classifier_conformal_skip_reason"] = "fitted_model_single_class"
        return out

    score_cal = _safe_score_matrix(est, x_cal, classes_fit)
    class_to_idx = {k: i for i, k in enumerate(classes_fit.tolist())}
    idx = np.asarray([class_to_idx.get(v, -1) for v in y_cal], dtype=int)
    valid = idx >= 0
    if int(np.sum(valid)) < min_cal:
        out["classifier_conformal_skip_reason"] = "insufficient_valid_calibration_points"
        return out
    rows = np.arange(idx.size, dtype=int)[valid]
    true_prob = score_cal[rows, idx[valid]]
    nonconformity = 1.0 - np.asarray(true_prob, dtype=float)
    n_cal = int(nonconformity.size)
    q_level = float(math.ceil((n_cal + 1) * (1.0 - alpha_val)) / float(max(1, n_cal)))
    q_hat = _quantile_higher(nonconformity, q_level)
    if not np.isfinite(q_hat):
        out["classifier_conformal_skip_reason"] = "invalid_qhat"
        return out
    threshold = float(1.0 - q_hat)

    score_eval = _safe_score_matrix(est, x_ev, classes_fit)
    pred_sets_idx: List[np.ndarray] = []
    for row in np.asarray(score_eval, dtype=float):
        keep = np.where(np.asarray(row, dtype=float) >= threshold - 1e-12)[0]
        if keep.size == 0:
            keep = np.asarray([int(np.argmax(row))], dtype=int)
        pred_sets_idx.append(np.asarray(keep, dtype=int))

    set_sizes = np.asarray([int(v.size) for v in pred_sets_idx], dtype=float)
    coverage = float("nan")
    if y_ev is not None and y_ev.size == len(pred_sets_idx):
        hit = []
        for label, keep_idx in zip(y_ev, pred_sets_idx):
            labels = {classes_fit[int(i)] for i in keep_idx.tolist()}
            hit.append(1.0 if label in labels else 0.0)
        coverage = float(np.mean(np.asarray(hit, dtype=float))) if hit else float("nan")

    out["classifier_conformal_applied"] = True
    out["classifier_conformal_calibration_size"] = int(n_cal)
    out["classifier_conformal_fit_size"] = int(x_fit.shape[0])
    out["classifier_conformal_qhat"] = float(q_hat)
    out["classifier_conformal_threshold"] = float(threshold)
    out["classifier_conformal_set_size_mean"] = float(np.mean(set_sizes)) if set_sizes.size else float("nan")
    out["classifier_conformal_set_size_median"] = float(np.median(set_sizes)) if set_sizes.size else float("nan")
    out["classifier_conformal_singleton_rate"] = (
        float(np.mean(set_sizes <= 1.0)) if set_sizes.size else float("nan")
    )
    out["classifier_conformal_empty_set_rate"] = (
        float(np.mean(set_sizes == 0.0)) if set_sizes.size else float("nan")
    )
    out["classifier_conformal_coverage"] = float(coverage)

    if include_prediction_sets:
        serialised: List[List[str]] = []
        for keep_idx in pred_sets_idx:
            serialised.append([str(classes_fit[int(i)]) for i in keep_idx.tolist()])
        out["classifier_conformal_prediction_sets"] = serialised

    return out


# ---------------------------------------------------------------------------
# MAPIE APS/RAPS integration (VAL12_Suggestions §2.3)
# ---------------------------------------------------------------------------

# MAPIE >=1.0 uses SplitConformalClassifier / CrossConformalClassifier.
# MAPIE <1.0 used MapieClassifier.  We support both.
_MAPIE_AVAILABLE = False
_MAPIE_LEGACY = False  # True → old MapieClassifier API (mapie <1.0)

try:
    from mapie.classification import SplitConformalClassifier as _SplitConformal  # type: ignore
    from mapie.classification import CrossConformalClassifier as _CrossConformal  # type: ignore
    from mapie.classification import RAPSConformityScore as _RAPSScore  # type: ignore
    _MAPIE_AVAILABLE = True
except Exception:
    try:
        from mapie.classification import MapieClassifier  # type: ignore  # noqa: F401
        _MAPIE_AVAILABLE = True
        _MAPIE_LEGACY = True
    except Exception:  # pragma: no cover
        pass


def mapie_available() -> bool:
    """Return True if the ``mapie`` package is importable."""
    return bool(_MAPIE_AVAILABLE)


def compute_mapie_conformal_sets(
    *,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: Optional[np.ndarray] = None,
    alpha: float = 0.10,
    method: str = "aps",
    seed: int = 0,
    cv_folds: int = 5,
    include_prediction_sets: bool = False,
) -> Dict[str, Any]:
    """Compute MAPIE-style conformal prediction sets.

    Parameters
    ----------
    method : str
        ``"aps"`` for Adaptive Prediction Sets, ``"raps"`` for Regularized
        APS, ``"cross"`` for cross-conformal via MAPIE's internal CV.
    cv_folds : int
        Number of CV folds for cross-conformal (``method="cross"``).  Ignored
        for ``"aps"`` and ``"raps"`` which use ``cv="prefit"``.
    """
    x_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train).ravel()
    x_ev = np.asarray(X_eval, dtype=float)
    y_ev = None if y_eval is None else np.asarray(y_eval).ravel()

    alpha_val = float(np.clip(alpha, 1e-4, 0.49))
    method_key = str(method or "aps").strip().lower()
    if method_key not in {"aps", "raps", "cross"}:
        method_key = "aps"

    prefix = "classifier_conformal_mapie_"
    out: Dict[str, Any] = {
        f"{prefix}enabled": True,
        f"{prefix}applied": False,
        f"{prefix}skip_reason": "",
        f"{prefix}method": str(method_key),
        f"{prefix}alpha": float(alpha_val),
        f"{prefix}set_size_mean": float("nan"),
        f"{prefix}set_size_median": float("nan"),
        f"{prefix}singleton_rate": float("nan"),
        f"{prefix}empty_set_rate": float("nan"),
        f"{prefix}coverage": float("nan"),
        f"{prefix}classes": [],
        f"{prefix}prediction_sets": [],
    }

    if not _MAPIE_AVAILABLE:
        out[f"{prefix}skip_reason"] = "mapie_not_installed"
        return out

    if x_tr.ndim != 2 or x_ev.ndim != 2:
        out[f"{prefix}skip_reason"] = "invalid_input_rank"
        return out
    if x_tr.shape[1] != x_ev.shape[1]:
        out[f"{prefix}skip_reason"] = "feature_dim_mismatch"
        return out
    if y_tr.size != x_tr.shape[0]:
        out[f"{prefix}skip_reason"] = "train_label_mismatch"
        return out

    classes = np.unique(y_tr)
    n_classes = int(classes.size)
    if n_classes < 2:
        out[f"{prefix}skip_reason"] = "single_class"
        return out
    out[f"{prefix}classes"] = [str(c) for c in classes.tolist()]

    try:
        from sklearn.base import clone as sk_clone
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_enc = le.fit_transform(y_tr)

        if method_key == "cross":
            # Cross-conformal: MAPIE handles internal CV.
            n_folds = int(max(2, min(int(cv_folds), int(np.min(np.bincount(y_enc))))))
            if _MAPIE_LEGACY:
                mapie_clf = MapieClassifier(
                    estimator=sk_clone(model),
                    cv=int(n_folds),
                    method="score",
                    random_state=int(seed),
                )
                mapie_clf.fit(x_tr, y_enc)
            else:
                # MAPIE >=1.0 enforces "lac" for binary targets.
                cross_score = "lac" if n_classes <= 2 else "aps"
                mapie_clf = _CrossConformal(
                    estimator=sk_clone(model),
                    confidence_level=1.0 - float(alpha_val),
                    conformity_score=cross_score,
                    cv=int(n_folds),
                    random_state=int(seed),
                )
                mapie_clf.fit_conformalize(x_tr, y_enc)
        else:
            # APS / RAPS: prefit on a train/calibration split.
            from sklearn.model_selection import StratifiedShuffleSplit as SSS

            n_train = int(x_tr.shape[0])
            cal_size = int(max(20, round(0.25 * float(n_train))))
            cal_size = int(min(cal_size, n_train - max(2, n_classes)))
            if cal_size < n_classes or cal_size < 10:
                out[f"{prefix}skip_reason"] = "insufficient_calibration_size"
                return out

            spl = SSS(n_splits=1, test_size=cal_size, random_state=int(seed))
            tr_idx, cal_idx = next(spl.split(np.zeros((n_train, 1)), y_enc))

            est = sk_clone(model)
            est.fit(x_tr[tr_idx], y_enc[tr_idx])

            if _MAPIE_LEGACY:
                mapie_method = "cumulated_score" if method_key == "aps" else "raps"
                mapie_clf = MapieClassifier(
                    estimator=est,
                    cv="prefit",
                    method=mapie_method,
                )
                mapie_clf.fit(x_tr[cal_idx], y_enc[cal_idx])
            else:
                # MAPIE >=1.0 enforces "lac" for binary targets.
                if n_classes <= 2:
                    score = "lac"
                else:
                    score = "aps" if method_key == "aps" else _RAPSScore()
                mapie_clf = _SplitConformal(
                    estimator=est,
                    confidence_level=1.0 - float(alpha_val),
                    conformity_score=score,
                    prefit=True,
                    random_state=int(seed),
                )
                mapie_clf.conformalize(x_tr[cal_idx], y_enc[cal_idx])

        # Predict on evaluation set.
        if _MAPIE_LEGACY:
            _, pred_sets = mapie_clf.predict(x_ev, alpha=float(alpha_val))
        else:
            _, pred_sets = mapie_clf.predict_set(x_ev)
        # pred_sets shape: (n_eval, n_classes, 1) — boolean mask
        if pred_sets.ndim == 3:
            pred_sets = pred_sets[:, :, 0]

        set_sizes = np.sum(pred_sets.astype(int), axis=1).astype(float)

        coverage = float("nan")
        if y_ev is not None and y_ev.size == pred_sets.shape[0]:
            y_ev_enc = le.transform(np.asarray(y_ev).ravel())
            hits = []
            for i, lab in enumerate(y_ev_enc):
                hits.append(1.0 if bool(pred_sets[i, int(lab)]) else 0.0)
            coverage = float(np.mean(hits)) if hits else float("nan")

        out[f"{prefix}applied"] = True
        out[f"{prefix}set_size_mean"] = float(np.mean(set_sizes)) if set_sizes.size else float("nan")
        out[f"{prefix}set_size_median"] = float(np.median(set_sizes)) if set_sizes.size else float("nan")
        out[f"{prefix}singleton_rate"] = (
            float(np.mean(set_sizes <= 1.0)) if set_sizes.size else float("nan")
        )
        out[f"{prefix}empty_set_rate"] = (
            float(np.mean(set_sizes == 0.0)) if set_sizes.size else float("nan")
        )
        out[f"{prefix}coverage"] = float(coverage)

        if include_prediction_sets:
            serialised_mapie: List[List[str]] = []
            inv_classes = le.classes_
            for row in pred_sets:
                included = [str(inv_classes[j]) for j in range(len(inv_classes)) if bool(row[j])]
                serialised_mapie.append(included)
            out[f"{prefix}prediction_sets"] = serialised_mapie

    except Exception as exc:
        out[f"{prefix}applied"] = False
        out[f"{prefix}skip_reason"] = str(type(exc).__name__)

    return out


def compute_conformal_singleton_rate(
    *,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: Optional[np.ndarray] = None,
    alpha: float = 0.10,
    method: str = "split",
    seed: int = 0,
) -> Dict[str, Any]:
    """Dispatch to a conformal backend and return singleton-rate diagnostics."""
    method_key = str(method or "split").strip().lower()
    if method_key not in {"split", "aps"}:
        method_key = "split"

    out: Dict[str, Any] = {
        "conformal_efficiency_method": str(method_key),
        "conformal_efficiency_applied": False,
        "conformal_efficiency_skip_reason": "",
        "conformal_singleton_rate": float("nan"),
        "conformal_set_size_mean": float("nan"),
        "conformal_set_size_median": float("nan"),
        "conformal_coverage": float("nan"),
    }

    if method_key == "aps":
        result = compute_mapie_conformal_sets(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            alpha=float(alpha),
            method="aps",
            seed=int(seed),
            include_prediction_sets=False,
        )
        applied = bool(result.get("classifier_conformal_mapie_applied", False))
        out.update(
            {
                "conformal_efficiency_applied": bool(applied),
                "conformal_efficiency_skip_reason": str(
                    result.get("classifier_conformal_mapie_skip_reason", "") or ""
                ),
                "conformal_singleton_rate": float(
                    result.get("classifier_conformal_mapie_singleton_rate", float("nan"))
                ),
                "conformal_set_size_mean": float(
                    result.get("classifier_conformal_mapie_set_size_mean", float("nan"))
                ),
                "conformal_set_size_median": float(
                    result.get("classifier_conformal_mapie_set_size_median", float("nan"))
                ),
                "conformal_coverage": float(
                    result.get("classifier_conformal_mapie_coverage", float("nan"))
                ),
            }
        )
        return out

    result = compute_split_conformal_sets(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        alpha=float(alpha),
        seed=int(seed),
        include_prediction_sets=False,
    )
    applied = bool(result.get("classifier_conformal_applied", False))
    out.update(
        {
            "conformal_efficiency_applied": bool(applied),
            "conformal_efficiency_skip_reason": str(
                result.get("classifier_conformal_skip_reason", "") or ""
            ),
            "conformal_singleton_rate": float(
                result.get("classifier_conformal_singleton_rate", float("nan"))
            ),
            "conformal_set_size_mean": float(
                result.get("classifier_conformal_set_size_mean", float("nan"))
            ),
            "conformal_set_size_median": float(
                result.get("classifier_conformal_set_size_median", float("nan"))
            ),
            "conformal_coverage": float(
                result.get("classifier_conformal_coverage", float("nan"))
            ),
        }
    )
    return out

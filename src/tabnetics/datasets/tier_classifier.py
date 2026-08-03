"""Shared learned tier-classification and complexity helpers.

This module keeps the runtime tier/complexity logic in one place so the
pipeline, prefilter, and oracle-conditioning paths all consume the same
signals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier


TIER_LABELS: Tuple[str, ...] = ("easy", "medium", "hard", "very_hard")

BASE_META_FEATURE_KEYS: Tuple[str, ...] = (
    "n",
    "p",
    "p_over_n",
    "class_count",
    "class_balance_entropy",
    "correlation_spectrum_decay",
    "heaping_fraction",
)

EXPANDED_META_FEATURE_KEYS: Tuple[str, ...] = (
    *BASE_META_FEATURE_KEYS,
    "fisher_f1",
    "f2_overlap",
    "n1_borderline",
    "n2_nn_ratio",
    "lsc",
    "t4_pca_ratio",
    "intrinsic_dim",
    "correlation_alpha",
    "signal_eigenvalue_fraction",
)

DEFAULT_MODEL_PATH = Path(__file__).with_name("tier_classifier_model.json")
DEFAULT_VAL20_COMPOSITE_PROFILE_IDS: Tuple[str, ...] = (
    "V20_C01_candidate_a_full64",
    "V20_C02_candidate_b_full64",
    "V20_C03_candidate_c_full64",
    "V20_C04_current_default_full64",
)

__tabnetics_execution_isolated_state__ = {
    "DEFAULT_MODEL_PATH": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
    "EXPANDED_META_FEATURE_KEYS": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}


@dataclass(frozen=True)
class TierPrediction:
    tier: str
    mode: str
    model_source: str
    used_features: Tuple[str, ...]
    fallback_applied: bool
    confidence: float
    model_path: str = ""
    model_sha256: str = ""
    model_size_bytes: int = 0
    fallback_reason: str = ""

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "tier": str(self.tier),
            "mode": str(self.mode),
            "model_source": str(self.model_source),
            "model_path": str(self.model_path),
            "model_sha256": str(self.model_sha256),
            "model_size_bytes": int(self.model_size_bytes),
            "used_features": list(self.used_features),
            "fallback_applied": bool(self.fallback_applied),
            "fallback_reason": str(self.fallback_reason),
            "confidence": float(self.confidence),
        }


def _clean_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _safe_meta(meta_features: Optional[Dict[str, float]]) -> Dict[str, float]:
    meta = dict(meta_features or {})
    return {str(key): _clean_float(value, 0.0) for key, value in meta.items()}


def _has_runtime_meta_signal(meta_features: Optional[Dict[str, float]]) -> bool:
    if not meta_features:
        return False
    known_keys = {
        "n",
        "p",
        "p_over_n",
        "class_count",
        "class_balance_entropy",
        "fisher_f1",
        "f2_overlap",
        "n1_borderline",
        "intrinsic_dim",
        "signal_eigenvalue_fraction",
    }
    return any(str(key) in known_keys for key in dict(meta_features).keys())


def heuristic_tier(meta_features: Optional[Dict[str, float]]) -> str:
    """Preserve the historical hand-tuned tier heuristic."""
    meta = _safe_meta(meta_features)
    p_over_n = float(meta.get("p_over_n", 0.0))
    class_count = float(meta.get("class_count", 2.0))
    entropy = float(meta.get("class_balance_entropy", 1.0))
    n_samples = float(meta.get("n", 0.0))

    if p_over_n >= 220.0 or class_count >= 8.0 or n_samples < 40.0:
        return "very_hard"
    if p_over_n >= 90.0 or class_count >= 5.0 or entropy < 0.45:
        return "hard"
    if p_over_n >= 30.0 or class_count >= 3.0 or entropy < 0.70:
        return "medium"
    return "easy"


def samples_per_class(meta_features: Optional[Dict[str, float]]) -> float:
    meta = _safe_meta(meta_features)
    n_samples = float(max(0.0, meta.get("n", 0.0)))
    class_count = float(max(1.0, meta.get("class_count", 1.0)))
    return float(n_samples / class_count)


def normalized_complexity_score(meta_features: Optional[Dict[str, float]]) -> float:
    """Compute a stable [0, 1] dataset-complexity score.

    High scores indicate datasets where retaining more features and relying on
    robustness-oriented portfolio behavior is reasonable. Noise-heavy datasets
    with weak signal-eigenstructure are intentionally pushed down so adaptive
    prefiltering does not blindly keep more features for every high-p dataset.
    """
    meta = _safe_meta(meta_features)
    fisher_f1 = max(0.0, float(meta.get("fisher_f1", 0.0)))
    n1_borderline = float(np.clip(meta.get("n1_borderline", 0.0), 0.0, 1.0))
    intrinsic_dim = max(0.0, float(meta.get("intrinsic_dim", 0.0)))
    signal_fraction = float(np.clip(meta.get("signal_eigenvalue_fraction", 0.0), 0.0, 1.0))
    overlap = float(np.clip(meta.get("f2_overlap", 0.0), 0.0, 1.0))
    p = max(1.0, float(meta.get("p", 1.0)))
    n = max(1.0, float(meta.get("n", 1.0)))
    max_rank = max(1.0, min(p, n))
    intrinsic_ratio = float(np.clip(intrinsic_dim / max_rank, 0.0, 1.0))
    separation_difficulty = float(np.clip(1.0 / (1.0 + fisher_f1), 0.0, 1.0))

    score = (
        0.30 * n1_borderline
        + 0.25 * separation_difficulty
        + 0.20 * intrinsic_ratio
        + 0.15 * signal_fraction
        + 0.10 * overlap
    )
    return float(np.clip(score, 0.0, 1.0))


def adaptive_prefilter_top_k(
    *,
    base_top_k: int,
    n_features: int,
    meta_features: Optional[Dict[str, float]],
    scaling_factor: float = 0.5,
) -> int:
    base = int(max(1, base_top_k))
    total_features = int(max(1, n_features))
    score = normalized_complexity_score(meta_features)
    effective = int(round(base * (1.0 + float(max(0.0, scaling_factor)) * score)))
    return int(np.clip(effective, 1, total_features))


def classifier_oracle_shrinkage_factor(meta_features: Optional[Dict[str, float]]) -> float:
    """Return a multiplicative factor for James-Stein shrinkage strength."""
    if not _has_runtime_meta_signal(meta_features):
        return 1.0
    tier = heuristic_tier(meta_features)
    mapping = {
        "easy": 1.30,
        "medium": 1.00,
        "hard": 0.85,
        "very_hard": 0.70,
    }
    return float(mapping.get(str(tier), 1.0))


def adjust_oracle_weights_for_complexity(
    oracle_weights: Dict[str, float],
    meta_features: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Tilt oracle weights based on the inferred dataset regime."""
    weights = {str(key): float(value) for key, value in dict(oracle_weights or {}).items()}
    if not weights:
        return {}
    if not _has_runtime_meta_signal(meta_features):
        return dict(weights)
    tier = heuristic_tier(meta_features)
    factors: Dict[str, float] = {}
    if tier == "easy":
        factors = {
            "stability": 1.25,
            "diversity": 0.80,
            "robustness": 0.90,
            "complexity": 0.95,
        }
    elif tier == "hard":
        factors = {
            "robustness": 1.20,
            "complexity": 1.10,
            "diversity": 1.05,
            "stability": 0.95,
        }
    elif tier == "very_hard":
        factors = {
            "robustness": 1.25,
            "complexity": 1.15,
            "diversity": 0.90,
            "stability": 1.05,
        }
    if not factors:
        return dict(weights)
    adjusted: Dict[str, float] = {}
    for name, value in weights.items():
        factor = float(factors.get(str(name), 1.0))
        adjusted[str(name)] = float(value * factor)
    return adjusted


def _load_tier_classifier_model_with_identity(
    path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse a learned-tier model from its exact current bytes."""

    model_path = DEFAULT_MODEL_PATH if path is None else Path(path)
    payload_bytes = model_path.read_bytes()
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tier classifier model must decode to a JSON object")
    identity = {
        "schema_version": "tabnetics_tier_model_identity_v1",
        "model_path": str(model_path.resolve()),
        "model_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "model_size_bytes": int(len(payload_bytes)),
    }
    return dict(payload), identity


def load_tier_classifier_model(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load a learned-tier model without retaining path-keyed mutable state."""

    payload, _identity = _load_tier_classifier_model_with_identity(path)
    return payload


def _predict_tree_node(tree_payload: Dict[str, Any], meta_features: Dict[str, float]) -> Tuple[str, Tuple[str, ...], float]:
    feature_trace: List[str] = []
    node = dict(tree_payload)
    while "leaf" not in node:
        feature = str(node.get("feature", "") or "").strip()
        threshold = _clean_float(node.get("threshold"), 0.0)
        if not feature:
            break
        feature_trace.append(feature)
        value = _clean_float(meta_features.get(feature), 0.0)
        branch = "left" if value <= threshold else "right"
        child = node.get(branch)
        if not isinstance(child, dict):
            break
        node = dict(child)
    leaf = node.get("leaf", {})
    if not isinstance(leaf, dict) or not leaf:
        return heuristic_tier(meta_features), tuple(feature_trace), 0.0
    predicted = str(node.get("prediction", "") or "").strip().lower()
    if predicted not in TIER_LABELS:
        predicted = str(max(leaf.items(), key=lambda item: _clean_float(item[1], 0.0))[0]).strip().lower()
    confidence = float(max(_clean_float(v, 0.0) for v in leaf.values()))
    return predicted, tuple(feature_trace), confidence


def predict_tier_with_details(
    meta_features: Optional[Dict[str, float]],
    *,
    mode: str = "heuristic",
    model_path: Optional[Path] = None,
) -> TierPrediction:
    meta = _safe_meta(meta_features)
    requested_mode = str(mode or "heuristic").strip().lower()
    if requested_mode != "learned":
        tier = heuristic_tier(meta)
        return TierPrediction(
            tier=str(tier),
            mode="heuristic",
            model_source="heuristic",
            used_features=tuple(),
            fallback_applied=False,
            confidence=1.0,
            fallback_reason="",
        )
    try:
        model_payload, model_identity = _load_tier_classifier_model_with_identity(model_path)
        required = tuple(str(name) for name in model_payload.get("required_features", ()) if str(name))
        if any(name not in meta for name in required):
            raise KeyError("missing_required_features")
        tree = dict(model_payload.get("tree", {}) or {})
        tier, used_features, confidence = _predict_tree_node(tree, meta)
        return TierPrediction(
            tier=str(tier),
            mode="learned",
            model_source=str(Path(model_payload.get("model_path", model_path or DEFAULT_MODEL_PATH)).name),
            used_features=tuple(used_features),
            fallback_applied=False,
            confidence=float(confidence),
            model_path=str(model_identity["model_path"]),
            model_sha256=str(model_identity["model_sha256"]),
            model_size_bytes=int(model_identity["model_size_bytes"]),
            fallback_reason="",
        )
    except Exception as exc:
        tier = heuristic_tier(meta)
        requested_path = DEFAULT_MODEL_PATH if model_path is None else Path(model_path)
        return TierPrediction(
            tier=str(tier),
            mode="learned",
            model_source="heuristic_fallback",
            used_features=tuple(),
            fallback_applied=True,
            confidence=0.0,
            model_path=str(requested_path),
            model_sha256="",
            model_size_bytes=0,
            fallback_reason=type(exc).__name__,
        )


def predict_tier(
    meta_features: Optional[Dict[str, float]],
    *,
    mode: str = "heuristic",
    model_path: Optional[Path] = None,
) -> str:
    return str(
        predict_tier_with_details(
            meta_features,
            mode=mode,
            model_path=model_path,
        ).tier
    )


def derive_ground_truth_tier(
    *,
    winner_score: float,
    best_composite_score: float,
    meta_features: Dict[str, float],
) -> str:
    oracle_gap = float(best_composite_score) - float(winner_score)
    if samples_per_class(meta_features) < 10.0 or oracle_gap > 0.10:
        return "very_hard"
    if oracle_gap < 0.005:
        return "easy"
    if oracle_gap <= 0.015:
        return "medium"
    return "hard"


def resolve_composite_training_target(
    score_map: Dict[str, float],
    *,
    winner_profile_id: Optional[str],
    composite_profile_ids: Sequence[str] = DEFAULT_VAL20_COMPOSITE_PROFILE_IDS,
) -> Dict[str, Any]:
    winner_id = str(winner_profile_id or "").strip()
    if not winner_id:
        raise ValueError(
            "Val-20 composite winner unresolved. Pass winner_profile_id explicitly after Wave 3 completes."
        )
    composite_scores: Dict[str, float] = {}
    missing_profiles: List[str] = []
    for profile_id in composite_profile_ids:
        key = str(profile_id)
        if key in score_map:
            composite_scores[key] = float(score_map[key])
        else:
            missing_profiles.append(key)
    if winner_id not in composite_scores:
        available = ", ".join(sorted(composite_scores.keys())[:8]) or "none"
        raise ValueError(
            f"Winner profile {winner_id!r} is missing from cross-campaign scores. Available composite scores: {available}."
        )
    if len(composite_scores) < 2:
        raise ValueError(
            "Need at least two resolved Val-20 composite scores per dataset before training the learned tier classifier."
        )
    best_profile_id, best_score = max(
        composite_scores.items(),
        key=lambda item: float(item[1]),
    )
    winner_score = float(composite_scores[winner_id])
    return {
        "winner_profile_id": str(winner_id),
        "winner_score": float(winner_score),
        "best_composite_profile_id": str(best_profile_id),
        "best_composite_score": float(best_score),
        "oracle_gap": float(best_score - winner_score),
        "available_profiles": tuple(sorted(composite_scores.keys())),
        "missing_profiles": tuple(missing_profiles),
    }


def _serialize_tree_node(
    tree: Any,
    node_id: int,
    *,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
) -> Dict[str, Any]:
    left_id = int(tree.children_left[node_id])
    right_id = int(tree.children_right[node_id])
    if left_id == right_id:
        probs = np.asarray(tree.value[node_id][0], dtype=float)
        total = float(np.sum(probs))
        if total > 0.0:
            probs = probs / total
        leaf = {
            str(class_labels[idx]): float(probs[idx])
            for idx in range(min(len(class_labels), probs.size))
        }
        prediction = str(class_labels[int(np.argmax(probs))]) if probs.size else "medium"
        return {
            "prediction": str(prediction),
            "leaf": leaf,
        }
    feature_index = int(tree.feature[node_id])
    threshold = float(tree.threshold[node_id])
    return {
        "feature": str(feature_names[feature_index]),
        "threshold": float(threshold),
        "left": _serialize_tree_node(
            tree,
            left_id,
            feature_names=feature_names,
            class_labels=class_labels,
        ),
        "right": _serialize_tree_node(
            tree,
            right_id,
            feature_names=feature_names,
            class_labels=class_labels,
        ),
    }


def _heuristic_predictions(meta_rows: Sequence[Dict[str, float]]) -> np.ndarray:
    return np.asarray([heuristic_tier(row) for row in meta_rows], dtype=object)


def _evaluate_lodocv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> Tuple[np.ndarray, float]:
    loo = LeaveOneOut()
    preds: List[str] = []
    for train_idx, test_idx in loo.split(X):
        model = DecisionTreeClassifier(
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            class_weight="balanced",
            random_state=int(random_state),
        )
        model.fit(X[train_idx], y[train_idx])
        pred = str(model.predict(X[test_idx])[0])
        preds.append(pred)
    pred_arr = np.asarray(preds, dtype=object)
    accuracy = float(np.mean(pred_arr == y))
    _ = tuple(feature_names)
    return pred_arr, accuracy


def _confusion_matrix_dict(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Dict[str, int]]:
    matrix = confusion_matrix(
        np.asarray(y_true, dtype=object),
        np.asarray(y_pred, dtype=object),
        labels=list(TIER_LABELS),
    )
    return {
        str(TIER_LABELS[i]): {
            str(TIER_LABELS[j]): int(matrix[i, j])
            for j in range(len(TIER_LABELS))
        }
        for i in range(len(TIER_LABELS))
    }


def train_serialized_tier_classifier(
    rows: Sequence[Dict[str, Any]],
    *,
    random_state: int = 42,
) -> Dict[str, Any]:
    usable_rows = [dict(row) for row in rows if str(row.get("label", "") or "") in TIER_LABELS]
    if len(usable_rows) < 8:
        raise ValueError("Need at least 8 labeled datasets to train tier classifier.")

    feature_names = tuple(EXPANDED_META_FEATURE_KEYS)
    X = np.asarray(
        [
            [_clean_float(dict(row.get("meta_features", {})).get(name), 0.0) for name in feature_names]
            for row in usable_rows
        ],
        dtype=float,
    )
    y = np.asarray([str(row["label"]) for row in usable_rows], dtype=object)

    full_model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=int(random_state),
    )
    full_model.fit(X, y)
    importances = np.asarray(full_model.feature_importances_, dtype=float)
    ranked = np.argsort(importances)[::-1]
    selected_indices = [int(idx) for idx in ranked if float(importances[idx]) > 0.0][:5]
    if not selected_indices:
        selected_indices = list(range(min(5, len(feature_names))))
    selected_features = tuple(feature_names[idx] for idx in selected_indices)
    X_sel = X[:, selected_indices]

    best_depth = 3
    best_leaf = 2
    best_pred = np.asarray([], dtype=object)
    best_acc = -1.0
    for max_depth in (2, 3, 4):
        for min_samples_leaf in (2, 3, 4):
            preds, acc = _evaluate_lodocv(
                X_sel,
                y,
                feature_names=selected_features,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=int(random_state),
            )
            if acc > best_acc + 1e-12:
                best_depth = int(max_depth)
                best_leaf = int(min_samples_leaf)
                best_pred = np.asarray(preds, dtype=object)
                best_acc = float(acc)

    final_model = DecisionTreeClassifier(
        max_depth=int(best_depth),
        min_samples_leaf=int(best_leaf),
        class_weight="balanced",
        random_state=int(random_state),
    )
    final_model.fit(X_sel, y)

    heuristic_pred = _heuristic_predictions(
        [dict(row.get("meta_features", {}) or {}) for row in usable_rows]
    )
    heuristic_acc = float(np.mean(heuristic_pred == y))

    per_tier_accuracy: Dict[str, float] = {}
    for tier in TIER_LABELS:
        mask = y == tier
        if int(np.sum(mask)) <= 0:
            per_tier_accuracy[str(tier)] = float("nan")
            continue
        per_tier_accuracy[str(tier)] = float(np.mean(best_pred[mask] == y[mask]))

    payload = {
        "schema_version": 1,
        "model_type": "decision_tree",
        "selected_features": list(selected_features),
        "required_features": list(selected_features),
        "training_rows": int(len(usable_rows)),
        "tree_hyperparameters": {
            "max_depth": int(best_depth),
            "min_samples_leaf": int(best_leaf),
            "random_state": int(random_state),
        },
        "metrics": {
            "lodocv_accuracy": float(best_acc),
            "heuristic_accuracy": float(heuristic_acc),
            "per_tier_accuracy": {
                str(key): None if not np.isfinite(value) else float(value)
                for key, value in per_tier_accuracy.items()
            },
            "confusion_matrix": _confusion_matrix_dict(y, best_pred),
            "heuristic_confusion_matrix": _confusion_matrix_dict(y, heuristic_pred),
        },
        "tree": _serialize_tree_node(
            final_model.tree_,
            0,
            feature_names=selected_features,
            class_labels=tuple(str(v) for v in final_model.classes_.tolist()),
        ),
    }
    return payload


def _load_cross_campaign_scores(merged_csv_path: Path) -> Dict[str, Dict[str, float]]:
    import pandas as pd

    frame = pd.read_csv(merged_csv_path)
    required = {"dataset_id", "profile_id", "balanced_accuracy"}
    if not required.issubset(set(frame.columns)):
        raise ValueError(f"{merged_csv_path} missing required columns: {sorted(required)}")
    frame = frame.dropna(subset=["dataset_id", "profile_id", "balanced_accuracy"]).copy()
    frame["dataset_id"] = frame["dataset_id"].astype(str)
    frame["profile_id"] = frame["profile_id"].astype(str)
    frame["balanced_accuracy"] = frame["balanced_accuracy"].astype(float)
    grouped = (
        frame.groupby(["dataset_id", "profile_id"], sort=True)["balanced_accuracy"]
        .mean()
        .reset_index()
    )
    out: Dict[str, Dict[str, float]] = {}
    for row in grouped.itertuples(index=False):
        out.setdefault(str(row.dataset_id), {})[str(row.profile_id)] = float(row.balanced_accuracy)
    return out


def build_cross_campaign_tier_rows(
    *,
    merged_csv_path: Path,
    dataset_ids: Optional[Sequence[str]] = None,
    winner_profile_id: Optional[str] = None,
    composite_profile_ids: Sequence[str] = DEFAULT_VAL20_COMPOSITE_PROFILE_IDS,
    require_complete_profiles: bool = True,
    seed: int = 11,
    sample_cap: int = 0,
    feature_cap: int = 0,
) -> List[Dict[str, Any]]:
    try:
        from tabnetics.datasets.meta_features import extract_meta_features
        from tabnetics.datasets.registry import DATASET_REGISTRY
        from tabnetics.validation.suite import load_feature_selection_dataset
    except Exception:
        from tabnetics.datasets.meta_features import extract_meta_features  # type: ignore
        from tabnetics.datasets.registry import DATASET_REGISTRY  # type: ignore
        from tabnetics.validation.suite import load_feature_selection_dataset  # type: ignore

    scores_by_dataset = _load_cross_campaign_scores(Path(merged_csv_path))
    selected_ids = list(dataset_ids) if dataset_ids is not None else sorted(scores_by_dataset.keys())
    rows: List[Dict[str, Any]] = []
    blocked_datasets: List[Dict[str, str]] = []
    for dataset_id in selected_ids:
        score_map = dict(scores_by_dataset.get(str(dataset_id), {}) or {})
        if not score_map:
            continue
        try:
            target = resolve_composite_training_target(
                score_map,
                winner_profile_id=winner_profile_id,
                composite_profile_ids=composite_profile_ids,
            )
        except ValueError as exc:
            blocked_datasets.append(
                {
                    "dataset_id": str(dataset_id),
                    "reason": str(exc),
                }
            )
            continue
        if str(target.get("winner_profile_id", "")).strip() == "":
            continue
        spec = DATASET_REGISTRY.get(str(dataset_id))
        if spec is None:
            continue
        try:
            loaded = load_feature_selection_dataset(
                spec,
                seed=int(seed),
                allow_synthetic_fallback=False,
                sample_cap=int(sample_cap),
                feature_cap=int(feature_cap),
                source_policy="real_only",
                require_hf_source=False,
            )
        except Exception as load_exc:
            blocked_datasets.append(
                {
                    "dataset_id": str(dataset_id),
                    "reason": f"load failed: {load_exc}",
                }
            )
            continue
        X = np.asarray(loaded.X, dtype=float)
        y = np.asarray(loaded.y).ravel()
        meta = extract_meta_features(X, y, expanded=True)
        label = derive_ground_truth_tier(
            winner_score=float(target["winner_score"]),
            best_composite_score=float(target["best_composite_score"]),
            meta_features=meta,
        )
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "label": str(label),
                "meta_features": {str(key): float(value) for key, value in meta.items()},
                "winner_profile_id": str(target["winner_profile_id"]),
                "winner_score": float(target["winner_score"]),
                "best_composite_profile_id": str(target["best_composite_profile_id"]),
                "best_composite_score": float(target["best_composite_score"]),
                "oracle_gap": float(target["oracle_gap"]),
            }
        )
    if blocked_datasets and bool(require_complete_profiles):
        preview = "; ".join(
            f"{row['dataset_id']}: {row['reason']}" for row in blocked_datasets[:5]
        )
        suffix = " ..." if len(blocked_datasets) > 5 else ""
        raise RuntimeError(
            "Val-20 composite-oracle labels are not ready for learned tier training. "
            f"Blocked datasets={len(blocked_datasets)}. {preview}{suffix}"
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the serialized tier classifier.")
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=Path("run_artifacts") / "cross_campaign_merged.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--winner-profile-id",
        type=str,
        default="",
        help=(
            "Resolved Val-20 composite winner profile ID used as the always-winner baseline. "
            "Leave blank only while scaffolding; training will fail with a clear placeholder error."
        ),
    )
    parser.add_argument(
        "--allow-partial-composites",
        action="store_true",
        help=(
            "Allow partial cross-campaign composite coverage by skipping blocked datasets. "
            "Use only for scaffolding diagnostics before Val-20 Wave 3 completes."
        ),
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sample-cap", type=int, default=0)
    parser.add_argument("--feature-cap", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = build_cross_campaign_tier_rows(
        merged_csv_path=Path(args.merged_csv),
        winner_profile_id=str(args.winner_profile_id or "").strip() or None,
        require_complete_profiles=not bool(args.allow_partial_composites),
        seed=int(args.seed),
        sample_cap=int(args.sample_cap),
        feature_cap=int(args.feature_cap),
    )
    payload = train_serialized_tier_classifier(rows, random_state=int(args.seed))
    payload["model_path"] = str(Path(args.output).name)
    payload["data_summary"] = {
        "n_datasets": int(len(rows)),
        "dataset_ids": [str(row["dataset_id"]) for row in rows],
        "label_counts": {
            str(label): int(sum(1 for row in rows if str(row.get("label")) == label))
            for label in TIER_LABELS
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

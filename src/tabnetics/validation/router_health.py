"""Auto-router predicted-vs-realized health analyzer.

This module is intentionally report-only. It consumes validation or score-router
corpus rows that already contain auto-router decision snapshots, joins realized
selected/default scores, and emits targeted retrain harvest rows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised when scipy is present in the env
    from scipy.stats import binomtest as _scipy_binomtest
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except Exception:  # pragma: no cover - fallback for minimal environments
    _scipy_binomtest = None
    _scipy_wilcoxon = None


DEFAULT_BASELINE_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "auto_router"
    / "artifacts"
    / "v25_calibrated_score_router"
    / "manifest.json"
)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        if not np.isfinite(float(value)):
            return bool(default)
        return bool(int(value))
    text = str(value if value is not None else "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(default)


def _parse_maybe_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except Exception:
            return {}
    return {}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    return value


@dataclass(frozen=True)
class RouterHealthBaseline:
    """V25 OOF baseline used to anchor router health tests."""

    name: str = "v25_calibrated_score_router"
    n_policy_groups: int = 0
    policy_defaulted_rate: float = 0.0
    non_default_rate: float = 0.0
    mean_delta_balanced_accuracy_vs_default: float = 0.0
    mean_delta_macro_f1_vs_default: float = 0.0
    mean_delta_utility_vs_default: float = 0.0
    beats_default_probability_threshold: float = 0.0
    decision_threshold: float = 0.0
    balanced_accuracy_lcb_offset: float = 0.0
    macro_f1_lcb_offset: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_policy_groups": int(self.n_policy_groups),
            "policy_defaulted_rate": float(self.policy_defaulted_rate),
            "non_default_rate": float(self.non_default_rate),
            "mean_delta_balanced_accuracy_vs_default": float(
                self.mean_delta_balanced_accuracy_vs_default
            ),
            "mean_delta_macro_f1_vs_default": float(self.mean_delta_macro_f1_vs_default),
            "mean_delta_utility_vs_default": float(self.mean_delta_utility_vs_default),
            "beats_default_probability_threshold": float(
                self.beats_default_probability_threshold
            ),
            "decision_threshold": float(self.decision_threshold),
            "balanced_accuracy_lcb_offset": float(self.balanced_accuracy_lcb_offset),
            "macro_f1_lcb_offset": float(self.macro_f1_lcb_offset),
        }


def load_router_health_baseline(path: Optional[Path | str] = None) -> RouterHealthBaseline:
    """Load baseline rates from a score-router manifest or summary JSON."""

    manifest_path = Path(path) if path is not None else DEFAULT_BASELINE_ARTIFACT
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "training_metadata" in payload:
        metadata = dict(payload.get("training_metadata", {}) or {})
        config = dict(payload.get("config", {}) or {})
        selected_family = str(metadata.get("selected_model_family", "") or "")
        summaries = list(metadata.get("model_family_summaries", []) or [])
        selected = next(
            (item for item in summaries if str(item.get("family", "")) == selected_family),
            summaries[-1] if summaries else {},
        )
        policy = dict(selected.get("calibrated_policy", {}) or {})
        name = str(payload.get("artifact_type", "score_router"))
    else:
        policy = dict(payload.get("calibrated_policy", payload) or {})
        config = dict(payload.get("config", {}) or {})
        name = str(payload.get("name", "router_health_baseline"))
    n_groups = int(policy.get("n_policy_groups", payload.get("n_policy_groups", 0)) or 0)
    policy_defaulted_count = int(policy.get("policy_defaulted_count", 0) or 0)
    non_default_count = int(policy.get("non_default_selected_count", 0) or 0)
    mean_ba = _safe_float(policy.get("mean_delta_balanced_accuracy_vs_default"), 0.0)
    mean_f1 = _safe_float(policy.get("mean_delta_macro_f1_vs_default"), 0.0)
    return RouterHealthBaseline(
        name=name,
        n_policy_groups=n_groups,
        policy_defaulted_rate=float(policy_defaulted_count / n_groups) if n_groups else 0.0,
        non_default_rate=float(non_default_count / n_groups) if n_groups else 0.0,
        mean_delta_balanced_accuracy_vs_default=mean_ba,
        mean_delta_macro_f1_vs_default=mean_f1,
        mean_delta_utility_vs_default=float(0.7 * mean_ba + 0.3 * mean_f1),
        beats_default_probability_threshold=_safe_float(
            policy.get(
                "beats_default_probability_threshold",
                config.get("beats_default_probability_threshold", 0.0),
            ),
            0.0,
        ),
        decision_threshold=_safe_float(
            policy.get("decision_threshold", config.get("decision_threshold", 0.0)),
            0.0,
        ),
        balanced_accuracy_lcb_offset=_safe_float(
            policy.get(
                "balanced_accuracy_lcb_offset",
                config.get("balanced_accuracy_lcb_offset", 0.0),
            ),
            0.0,
        ),
        macro_f1_lcb_offset=_safe_float(
            policy.get("macro_f1_lcb_offset", config.get("macro_f1_lcb_offset", 0.0)),
            0.0,
        ),
    )


def _extract_router_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    cfg = _parse_maybe_json(row.get("config_snapshot"))
    if isinstance(cfg, Mapping):
        nested = cfg.get("auto_router_last_decision", {})
        if isinstance(nested, Mapping):
            snapshot.update({str(k): v for k, v in nested.items()})
    nested_direct = _parse_maybe_json(row.get("auto_router_last_decision"))
    if isinstance(nested_direct, Mapping):
        snapshot.update({str(k): v for k, v in nested_direct.items()})
    for key, value in dict(row).items():
        if str(key).startswith("auto_router_"):
            snapshot[str(key)] = value
    return snapshot


def _normalize_selected_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row_raw in pd.DataFrame(frame).iterrows():
        row = row_raw.to_dict()
        snapshot = _extract_router_snapshot(row)
        has_router_signal = bool(snapshot) or any(
            key in row
            for key in (
                "selected_candidate_id",
                "policy_defaulted",
                "pred_balanced_accuracy",
                "predicted_balanced_accuracy",
                "beats_default_probability",
            )
        )
        if not has_router_signal:
            continue
        used_raw = snapshot.get("auto_router_used", snapshot.get("auto_router_enabled", None))
        used = True if used_raw is None else _safe_bool(used_raw, False)
        if not used:
            continue
        selected_id = str(
            snapshot.get(
                "auto_router_selected_candidate_id",
                row.get("selected_candidate_id", row.get("candidate_id", "")),
            )
            or ""
        )
        default_id = str(
            snapshot.get("auto_router_default_candidate_id", row.get("default_candidate_id", ""))
            or ""
        )
        out = dict(row)
        out.update(
            {
                "router_used": bool(used),
                "selected_candidate_id": selected_id,
                "raw_selected_candidate_id": str(
                    snapshot.get("auto_router_raw_selected_candidate_id", selected_id) or selected_id
                ),
                "default_candidate_id": default_id,
                "policy_defaulted": _safe_bool(
                    snapshot.get("auto_router_policy_defaulted", row.get("policy_defaulted", False)),
                    False,
                ),
                "beats_default_probability": _safe_float(
                    snapshot.get(
                        "auto_router_beats_default_probability",
                        row.get("beats_default_probability", row.get("pred_beats_default_probability")),
                    ),
                    float("nan"),
                ),
                "predicted_balanced_accuracy": _safe_float(
                    snapshot.get(
                        "auto_router_predicted_balanced_accuracy",
                        row.get("predicted_balanced_accuracy", row.get("pred_balanced_accuracy")),
                    ),
                    float("nan"),
                ),
                "predicted_macro_f1": _safe_float(
                    snapshot.get(
                        "auto_router_predicted_macro_f1",
                        row.get("predicted_macro_f1", row.get("pred_macro_f1")),
                    ),
                    float("nan"),
                ),
                "predicted_utility": _safe_float(
                    snapshot.get("auto_router_predicted_utility", row.get("predicted_utility")),
                    float("nan"),
                ),
                "calibrated_utility": _safe_float(
                    snapshot.get("auto_router_calibrated_utility", row.get("calibrated_utility")),
                    float("nan"),
                ),
                "utility_margin": _safe_float(
                    snapshot.get(
                        "auto_router_utility_margin",
                        row.get("utility_margin", row.get("predicted_utility_margin")),
                    ),
                    float("nan"),
                ),
                "realized_balanced_accuracy": _safe_float(
                    row.get("selected_balanced_accuracy", row.get("balanced_accuracy")),
                    float("nan"),
                ),
                "realized_macro_f1": _safe_float(
                    row.get("selected_macro_f1", row.get("macro_f1")),
                    float("nan"),
                ),
                "auto_router_dependence_descriptor_policy": str(
                    snapshot.get(
                        "auto_router_dependence_descriptor_policy",
                        row.get("auto_router_dependence_descriptor_policy", ""),
                    )
                    or ""
                ),
                "auto_router_dependence_descriptor_model_input_enabled": _safe_bool(
                    snapshot.get(
                        "auto_router_dependence_descriptor_model_input_enabled",
                        row.get("auto_router_dependence_descriptor_model_input_enabled", False),
                    ),
                    False,
                ),
                "auto_router_missingness_descriptor_policy": str(
                    snapshot.get(
                        "auto_router_missingness_descriptor_policy",
                        row.get("auto_router_missingness_descriptor_policy", ""),
                    )
                    or ""
                ),
                "auto_router_missingness_descriptor_model_input_enabled": _safe_bool(
                    snapshot.get(
                        "auto_router_missingness_descriptor_model_input_enabled",
                        row.get(
                            "auto_router_missingness_descriptor_model_input_enabled",
                            False,
                        ),
                    ),
                    False,
                ),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _default_key_columns(frame: pd.DataFrame) -> list[str]:
    candidates = ["dataset_id", "seed", "protocol"]
    return [col for col in candidates if col in frame.columns]


def _prepare_default_rows(default_frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    if default_frame is None or pd.DataFrame(default_frame).empty:
        return pd.DataFrame()
    default = pd.DataFrame(default_frame).copy()
    key_cols = _default_key_columns(default)
    if not key_cols:
        return pd.DataFrame()
    agg = {
        "balanced_accuracy": "mean",
        "macro_f1": "mean",
    }
    default = default.groupby(key_cols, dropna=False).agg(agg).reset_index()
    return default.rename(
        columns={
            "balanced_accuracy": "default_balanced_accuracy",
            "macro_f1": "default_macro_f1",
        }
    )


def join_router_realized_scores(
    selected_frame: pd.DataFrame,
    default_frame: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join router snapshots with realized selected/default scores."""

    selected = _normalize_selected_rows(selected_frame)
    if selected.empty:
        return selected
    if "default_balanced_accuracy" not in selected.columns:
        selected["default_balanced_accuracy"] = np.nan
    if "default_macro_f1" not in selected.columns:
        selected["default_macro_f1"] = np.nan
    default = _prepare_default_rows(default_frame)
    key_cols = [col for col in _default_key_columns(selected) if col in default.columns]
    if not default.empty and key_cols:
        selected = selected.merge(default, on=key_cols, how="left", suffixes=("", "_joined"))
        for metric in ("balanced_accuracy", "macro_f1"):
            col = f"default_{metric}"
            joined = f"{col}_joined"
            if joined in selected.columns:
                selected[col] = selected[col].where(
                    np.isfinite(pd.to_numeric(selected[col], errors="coerce")),
                    selected[joined],
                )
                selected = selected.drop(columns=[joined])
    selected["realized_utility"] = (
        0.7 * pd.to_numeric(selected["realized_balanced_accuracy"], errors="coerce")
        + 0.3 * pd.to_numeric(selected["realized_macro_f1"], errors="coerce")
    )
    selected["default_utility"] = (
        0.7 * pd.to_numeric(selected["default_balanced_accuracy"], errors="coerce")
        + 0.3 * pd.to_numeric(selected["default_macro_f1"], errors="coerce")
    )
    selected["default_arm_available"] = np.isfinite(selected["default_utility"].to_numpy(dtype=float))
    selected["delta_balanced_accuracy_vs_default"] = (
        pd.to_numeric(selected["realized_balanced_accuracy"], errors="coerce")
        - pd.to_numeric(selected["default_balanced_accuracy"], errors="coerce")
    )
    selected["delta_macro_f1_vs_default"] = (
        pd.to_numeric(selected["realized_macro_f1"], errors="coerce")
        - pd.to_numeric(selected["default_macro_f1"], errors="coerce")
    )
    selected["delta_utility_vs_default"] = selected["realized_utility"] - selected["default_utility"]
    selected["realized_beats_default"] = selected["delta_utility_vs_default"] > 0.0
    selected["prediction_residual_utility"] = (
        selected["realized_utility"] - pd.to_numeric(selected["predicted_utility"], errors="coerce")
    )
    return selected


def _binomial_test(k: int, n: int, p: float) -> dict[str, Any]:
    p = float(np.clip(p, 0.0, 1.0))
    if n <= 0:
        return {"n": int(n), "successes": int(k), "expected_probability": p, "p_value": None}
    if _scipy_binomtest is None:
        return {
            "n": int(n),
            "successes": int(k),
            "expected_probability": p,
            "observed_rate": float(k / n),
            "p_value": None,
            "method": "unavailable",
        }
    result = _scipy_binomtest(int(k), int(n), p)
    return {
        "n": int(n),
        "successes": int(k),
        "expected_probability": p,
        "observed_rate": float(k / n),
        "p_value": float(result.pvalue),
        "method": "binomtest",
    }


def _wilcoxon_centered(values: Sequence[float], center: float) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "center": float(center), "p_value": None}
    centered = arr - float(center)
    if np.allclose(centered, 0.0):
        return {
            "n": int(arr.size),
            "center": float(center),
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "wilcoxon_all_zero",
        }
    if _scipy_wilcoxon is None:
        return {
            "n": int(arr.size),
            "center": float(center),
            "statistic": None,
            "p_value": None,
            "method": "unavailable",
        }
    result = _scipy_wilcoxon(centered, zero_method="wilcox", alternative="two-sided")
    return {
        "n": int(arr.size),
        "center": float(center),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "method": "wilcoxon_centered_on_baseline",
    }


def analyze_router_health(
    selected_frame: pd.DataFrame,
    default_frame: Optional[pd.DataFrame] = None,
    *,
    baseline: Optional[RouterHealthBaseline] = None,
    alpha: float = 0.05,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return summary, joined rows, and retrain-harvest rows."""

    baseline = baseline or load_router_health_baseline()
    joined = join_router_realized_scores(selected_frame, default_frame)
    if joined.empty:
        summary = {
            "n_router_rows": 0,
            "baseline": baseline.to_dict(),
            "alarm_retrain_router": False,
            "status": "no_router_rows",
        }
        return summary, joined, pd.DataFrame()

    default_available = joined[joined["default_arm_available"].astype(bool)].copy()
    policy_defaulted_count = int(joined["policy_defaulted"].astype(bool).sum())
    policy_test = _binomial_test(
        policy_defaulted_count,
        int(len(joined)),
        baseline.policy_defaulted_rate,
    )
    hit_test: dict[str, Any]
    wilcoxon_utility: dict[str, Any]
    wilcoxon_ba: dict[str, Any]
    if default_available.empty:
        hit_test = {
            "n": 0,
            "successes": 0,
            "expected_probability": None,
            "p_value": None,
            "status": "default_arm_missing",
        }
        wilcoxon_utility = {
            "n": 0,
            "center": baseline.mean_delta_utility_vs_default,
            "p_value": None,
            "status": "default_arm_missing",
        }
        wilcoxon_ba = {
            "n": 0,
            "center": baseline.mean_delta_balanced_accuracy_vs_default,
            "p_value": None,
            "status": "default_arm_missing",
        }
    else:
        pred_probs = pd.to_numeric(
            default_available["beats_default_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        expected_p = float(np.nanmean(pred_probs[np.isfinite(pred_probs)])) if np.isfinite(pred_probs).any() else 0.5
        hit_test = _binomial_test(
            int(default_available["realized_beats_default"].astype(bool).sum()),
            int(len(default_available)),
            expected_p,
        )
        wilcoxon_utility = _wilcoxon_centered(
            default_available["delta_utility_vs_default"].to_numpy(dtype=float),
            baseline.mean_delta_utility_vs_default,
        )
        wilcoxon_ba = _wilcoxon_centered(
            default_available["delta_balanced_accuracy_vs_default"].to_numpy(dtype=float),
            baseline.mean_delta_balanced_accuracy_vs_default,
        )

    utility_mean = (
        float(pd.to_numeric(default_available["delta_utility_vs_default"], errors="coerce").mean())
        if not default_available.empty
        else float("nan")
    )
    ba_mean = (
        float(pd.to_numeric(default_available["delta_balanced_accuracy_vs_default"], errors="coerce").mean())
        if not default_available.empty
        else float("nan")
    )
    residual_mean = float(
        pd.to_numeric(joined["prediction_residual_utility"], errors="coerce").mean()
    )
    policy_rate = float(policy_defaulted_count / max(len(joined), 1))
    hit_observed = hit_test.get("observed_rate")
    hit_expected = hit_test.get("expected_probability")
    policy_alarm = (
        policy_test.get("p_value") is not None
        and float(policy_test["p_value"]) < float(alpha)
        and abs(policy_rate - baseline.policy_defaulted_rate) > 0.05
    )
    hit_alarm = (
        hit_test.get("p_value") is not None
        and float(hit_test["p_value"]) < float(alpha)
        and hit_observed is not None
        and hit_expected is not None
        and float(hit_observed) < float(hit_expected)
    )
    delta_alarm = (
        wilcoxon_utility.get("p_value") is not None
        and float(wilcoxon_utility["p_value"]) < float(alpha)
        and np.isfinite(utility_mean)
        and utility_mean < baseline.mean_delta_utility_vs_default
    )
    harvest = emit_router_harvest_rows(joined)
    summary = {
        "status": "ok",
        "alpha": float(alpha),
        "baseline": baseline.to_dict(),
        "n_router_rows": int(len(joined)),
        "n_default_arm_rows": int(len(default_available)),
        "default_arm_available_rate": float(len(default_available) / max(len(joined), 1)),
        "policy_defaulted_count": int(policy_defaulted_count),
        "policy_defaulted_rate": policy_rate,
        "policy_defaulted_binomial": policy_test,
        "beats_default_binomial": hit_test,
        "delta_utility_wilcoxon_vs_baseline": wilcoxon_utility,
        "delta_balanced_accuracy_wilcoxon_vs_baseline": wilcoxon_ba,
        "mean_delta_utility_vs_default": utility_mean,
        "mean_delta_balanced_accuracy_vs_default": ba_mean,
        "mean_prediction_residual_utility": residual_mean,
        "alarm_policy_defaulted_rate": bool(policy_alarm),
        "alarm_beats_default_hit_rate": bool(hit_alarm),
        "alarm_realized_delta": bool(delta_alarm),
        "alarm_retrain_router": bool(policy_alarm or hit_alarm or delta_alarm),
        "n_harvest_rows": int(len(harvest)),
    }
    return summary, joined, harvest


def emit_router_harvest_rows(joined: pd.DataFrame) -> pd.DataFrame:
    """Emit score-router retrain harvest rows in a corpus-compatible shape."""

    if pd.DataFrame(joined).empty:
        return pd.DataFrame()
    frame = pd.DataFrame(joined).copy()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        payload = {
            "dataset_id": str(row.get("dataset_id", "")),
            "seed": _safe_int(row.get("seed"), 0),
            "candidate_id": str(row.get("selected_candidate_id", "")),
            "raw_selected_candidate_id": str(row.get("raw_selected_candidate_id", "")),
            "default_candidate_id": str(row.get("default_candidate_id", "")),
            "dataset_name": str(row.get("dataset_name", "")),
            "domain": str(row.get("domain", "")),
            "effective_tier": str(row.get("effective_tier", row.get("tier", ""))),
            "balanced_accuracy": _safe_float(row.get("realized_balanced_accuracy"), float("nan")),
            "macro_f1": _safe_float(row.get("realized_macro_f1"), float("nan")),
            "default_balanced_accuracy": _safe_float(row.get("default_balanced_accuracy"), float("nan")),
            "default_macro_f1": _safe_float(row.get("default_macro_f1"), float("nan")),
            "delta_balanced_accuracy_vs_default": _safe_float(
                row.get("delta_balanced_accuracy_vs_default"), float("nan")
            ),
            "delta_macro_f1_vs_default": _safe_float(
                row.get("delta_macro_f1_vs_default"), float("nan")
            ),
            "pred_balanced_accuracy": _safe_float(row.get("predicted_balanced_accuracy"), float("nan")),
            "pred_macro_f1": _safe_float(row.get("predicted_macro_f1"), float("nan")),
            "predicted_utility": _safe_float(row.get("predicted_utility"), float("nan")),
            "calibrated_utility": _safe_float(row.get("calibrated_utility"), float("nan")),
            "predicted_utility_margin": _safe_float(row.get("utility_margin"), float("nan")),
            "beats_default_probability": _safe_float(
                row.get("beats_default_probability"), float("nan")
            ),
            "policy_defaulted": bool(row.get("policy_defaulted", False)),
            "router_health_harvest": True,
        }
        for key in (
            "auto_router_dependence_descriptor_policy",
            "auto_router_dependence_descriptor_model_input_enabled",
            "auto_router_missingness_descriptor_policy",
            "auto_router_missingness_descriptor_model_input_enabled",
        ):
            if key in row:
                payload[key] = row.get(key)
        for key in (
            "effective_enabled_methods",
            "cfg_df_stage_position",
            "cfg_classification_backend",
            "cfg_classifier_oracle_k",
            "cfg_classifier_selection_mode",
        ):
            if key in row:
                payload[key] = row.get(key)
        rows.append(payload)
    return pd.DataFrame(rows)


def predict_router_on_descriptors(
    descriptor_frame: pd.DataFrame,
    *,
    model_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Run packaged score-router inference for precomputed descriptor rows."""

    from tabnetics.auto_router import load_default_auto_router

    router = load_default_auto_router(model_dir)
    rows: list[dict[str, Any]] = []
    metadata_cols = {"dataset_id", "dataset_name", "seed", "config", "protocol"}
    for _, raw in pd.DataFrame(descriptor_frame).iterrows():
        row = raw.to_dict()
        descriptor = {
            "dataset_id": str(row.get("dataset_id", "")),
            "dataset_name": str(row.get("dataset_name", "")),
            "feature_vector": {
                str(name): _safe_float(row.get(name), 0.0)
                for name in router.feature_names
                if name in row
            },
        }
        output = router.predict(descriptor)
        rows.append({k: row.get(k) for k in metadata_cols if k in row} | output.to_snapshot())
    return pd.DataFrame(rows)


def merge_descriptor_predictions(
    realized_frame: pd.DataFrame,
    descriptor_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach descriptor-only router predictions to realized result rows."""

    realized = pd.DataFrame(realized_frame).copy()
    preds = pd.DataFrame(descriptor_predictions).copy()
    if realized.empty or preds.empty:
        return realized
    key_candidates = ("dataset_id", "seed", "protocol")
    keys = [key for key in key_candidates if key in realized.columns and key in preds.columns]
    if not keys and "dataset_id" in realized.columns and "dataset_id" in preds.columns:
        keys = ["dataset_id"]
    if not keys:
        raise ValueError("Cannot merge descriptor predictions without a shared dataset_id/key column.")
    pred_cols = [col for col in preds.columns if str(col).startswith("auto_router_")]
    merged = realized.merge(preds[keys + pred_cols], on=keys, how="left", suffixes=("", "_descriptor"))
    for col in pred_cols:
        desc_col = f"{col}_descriptor"
        if desc_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[desc_col]
        else:
            merged[col] = merged[col].where(merged[col].notna(), merged[desc_col])
        merged = merged.drop(columns=[desc_col])
    return merged


def write_router_health_outputs(
    selected_frame: pd.DataFrame,
    *,
    output_dir: Path | str,
    default_frame: Optional[pd.DataFrame] = None,
    baseline: Optional[RouterHealthBaseline] = None,
    prefix: str = "router_health",
    alpha: float = 0.05,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary, joined, harvest = analyze_router_health(
        selected_frame,
        default_frame,
        baseline=baseline,
        alpha=alpha,
    )
    summary_path = output_path / f"{prefix}_summary.json"
    joined_path = output_path / f"{prefix}_joined.csv"
    harvest_path = output_path / f"{prefix}_harvest.csv"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    joined.to_csv(joined_path, index=False)
    harvest.to_csv(harvest_path, index=False)
    return {
        "summary": summary_path,
        "joined": joined_path,
        "harvest": harvest_path,
    }


def _read_csv_optional(path: str) -> Optional[pd.DataFrame]:
    text = str(path or "").strip()
    if not text:
        return None
    return pd.read_csv(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--descriptor-csv", default="")
    parser.add_argument("--router-model-dir", default="")
    parser.add_argument("--default-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="router_health")
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    selected = pd.read_csv(args.results_csv)
    if str(args.descriptor_csv or "").strip():
        descriptor_predictions = predict_router_on_descriptors(
            pd.read_csv(args.descriptor_csv),
            model_dir=(Path(args.router_model_dir) if str(args.router_model_dir or "").strip() else None),
        )
        selected = merge_descriptor_predictions(selected, descriptor_predictions)
    default = _read_csv_optional(args.default_csv)
    baseline = load_router_health_baseline(args.baseline_json or None)
    paths = write_router_health_outputs(
        selected,
        output_dir=args.output_dir,
        default_frame=default,
        baseline=baseline,
        prefix=str(args.prefix),
        alpha=float(args.alpha),
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

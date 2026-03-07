#!/usr/bin/env python3
"""Analyze Val-14 feature-effect matrix and activation-gate checks.

Primary outputs:
- per-feature paired contrasts (dataset-level mean BA deltas)
- Wilcoxon signed-rank p-values + BH-FDR q-values
- activation checks for pre-launch smoke gating
- interpretation caveats for high-risk infrastructure profiles
- group-sparse seed-stability diagnostics
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from tabnetics.validation.core.aggregate import aggregate


PROFILE_IDS: List[str] = [
    "a_control",
    "d_default",
    "v14_ref",
    "v14_no_bh",
    "v14_no_varfloor",
    "v14_no_copula5",
    "v14_add_stability",
    "v14_add_pareto",
    "v14_no_extreme_multiclass",
    "v14_tritrust",
    "v14_tabpfn",
    "v14_ipss",
    "v14_ipss_eats",
    "v14_group_sparse",
    "v14_batch_combat",
    "v14_mapie_aps",
    "v14_mapie_raps",
    "v14_mapie_cross",
    "v14_multiomics_adapter",
    "v14_gates_only",
]
PROFILE_SET = set(PROFILE_IDS)

PRIMARY_CONTRASTS: List[Tuple[str, str, str]] = [
    ("v14_no_bh", "v14_ref", "bh_prefilter"),
    ("v14_no_varfloor", "v14_ref", "variance_floor"),
    ("v14_no_copula5", "v14_ref", "copula_5_vs_3"),
    ("v14_add_stability", "v14_ref", "stability_weighting"),
    ("v14_add_pareto", "v14_ref", "pareto_sizing"),
    ("v14_no_extreme_multiclass", "v14_ref", "extreme_multiclass_gate"),
    ("v14_tritrust", "v14_ref", "oracle_weighting"),
    ("v14_tabpfn", "v14_ref", "tabpfn"),
    ("v14_ipss", "v14_ref", "ipss"),
    ("v14_group_sparse", "v14_ref", "group_sparse"),
    ("v14_batch_combat", "v14_ref", "batch_combat"),
    ("v14_mapie_aps", "v14_ref", "mapie_aps"),
    ("v14_mapie_raps", "v14_ref", "mapie_raps"),
    ("v14_mapie_cross", "v14_ref", "mapie_cross"),
    ("v14_multiomics_adapter", "v14_ref", "multiomics_adapter"),
    ("v14_ipss_eats", "v14_ipss", "eats_over_ipss"),
]

ANCHOR_CONTRASTS: List[Tuple[str, str, str]] = [
    ("v14_ref", "d_default", "ref_vs_prod"),
    ("v14_ref", "a_control", "ref_vs_simple"),
    ("v14_gates_only", "d_default", "gates_only_vs_prod"),
]

HIGH_RISK_ACTIVATION_ONLY_PROFILES: Dict[str, Dict[str, str]] = {
    "v14_batch_combat": {
        "risk_level": "HIGH",
        "scope": "activation_only",
        "rationale": (
            "ComBat with source/kmeans2 pseudo-batch labels is a systems/activation check "
            "only in Val-14 and is not scientific evidence for biological batch correction."
        ),
    },
    "v14_multiomics_adapter": {
        "risk_level": "HIGH",
        "scope": "activation_only",
        "rationale": (
            "Multi-omics split-halves adapter is a synthetic infrastructure check in Val-14 "
            "and is not scientific evidence for real multi-omics integration."
        ),
    },
}

GROUP_STABILITY_MIN_MEAN_JACCARD = 0.35
GROUP_STABILITY_MIN_DOMINANT_SIGNATURE = 0.55


def _extract_profile(job_id: str) -> str:
    m = re.match(r"^val14(?:_activation)?/([^/]+)/", str(job_id))
    return str(m.group(1)) if m else "unknown"


def _safe_wilcoxon(values: np.ndarray) -> Tuple[float, float, int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    if np.allclose(arr, 0.0):
        return 0.0, 1.0, int(arr.size)
    try:
        stat, p = wilcoxon(arr, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p), int(arr.size)
    except Exception:
        return float("nan"), float("nan"), int(arr.size)


def _paired_effect_sizes(values: np.ndarray) -> Tuple[float, float]:
    """Compute paired effect sizes on per-dataset deltas.

    Returns:
    - Cohen's d_z (paired): mean(delta) / std(delta, ddof=1)
    - Rank-biserial correlation from signed-rank decomposition
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan")

    if n >= 2:
        std = float(np.std(arr, ddof=1))
        if std > 0.0:
            cohen_d = float(np.mean(arr) / std)
        else:
            mean_val = float(np.mean(arr))
            if np.isclose(mean_val, 0.0):
                cohen_d = 0.0
            else:
                cohen_d = float(np.sign(mean_val) * np.inf)
    else:
        cohen_d = float("nan")

    nonzero = arr[~np.isclose(arr, 0.0)]
    if nonzero.size == 0:
        rank_biserial = 0.0
    else:
        ranks = rankdata(np.abs(nonzero), method="average")
        sum_pos = float(np.sum(ranks[nonzero > 0.0]))
        sum_neg = float(np.sum(ranks[nonzero < 0.0]))
        denom = float(sum_pos + sum_neg)
        rank_biserial = float((sum_pos - sum_neg) / denom) if denom > 0.0 else 0.0

    return float(cohen_d), float(rank_biserial)


def _bh_fdr(pvalues: Sequence[float]) -> List[float]:
    p = np.asarray([float(x) for x in pvalues], dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if valid.size == 0:
        return list(out)

    order = valid[np.argsort(p[valid])]
    m = float(valid.size)
    ranked_q = np.empty(valid.size, dtype=float)
    prev = 1.0
    for i in range(valid.size - 1, -1, -1):
        rank = float(i + 1)
        idx = int(order[i])
        q = min(prev, (p[idx] * m) / rank)
        ranked_q[i] = q
        prev = q
    for i, idx in enumerate(order):
        out[int(idx)] = float(ranked_q[i])
    return list(out)


def _parse_listlike(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("("):
        try:
            raw = ast.literal_eval(text)
            if isinstance(raw, (list, tuple, set)):
                return [str(v) for v in raw]
        except Exception:
            pass
    if "," in text:
        return [t.strip() for t in text.split(",") if t.strip()]
    return [text]


def _parse_int_set(value: Any) -> set[int]:
    out: set[int] = set()
    for token in _parse_listlike(value):
        txt = str(token).strip()
        if not txt:
            continue
        try:
            out.add(int(txt))
        except Exception:
            continue
    return out


def _compute_contrast_table(
    df_runs: pd.DataFrame,
    contrasts: Sequence[Tuple[str, str, str]],
    *,
    apply_bh: bool,
) -> pd.DataFrame:
    work = df_runs.copy()
    if "config" in work.columns:
        work = work[work["config"].astype(str).eq("baseline")]
    work = work[work["profile"].isin(PROFILE_SET)]
    work = work[np.isfinite(pd.to_numeric(work.get("balanced_accuracy"), errors="coerce"))]

    grouped = (
        work.groupby(["dataset_id", "profile"], dropna=False)["balanced_accuracy"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot(index="dataset_id", columns="profile", values="balanced_accuracy")

    rows: List[Dict[str, Any]] = []
    for left, right, label in contrasts:
        if left not in pivot.columns or right not in pivot.columns:
            rows.append(
                {
                    "contrast": label,
                    "left": left,
                    "right": right,
                    "n_datasets": 0,
                    "mean_delta": float("nan"),
                    "median_delta": float("nan"),
                    "hl_delta": float("nan"),
                    "wins": 0,
                    "losses": 0,
                    "ties": 0,
                    "wilcoxon_stat": float("nan"),
                    "wilcoxon_p": float("nan"),
                }
            )
            continue

        pair = pivot[[left, right]].dropna(how="any")
        delta = (pair[left] - pair[right]).to_numpy(dtype=float)
        stat, p, n = _safe_wilcoxon(delta)
        cohen_d, rank_biserial = _paired_effect_sizes(delta)
        rows.append(
            {
                "contrast": label,
                "left": left,
                "right": right,
                "n_datasets": int(n),
                "mean_delta": float(np.mean(delta)) if n else float("nan"),
                "median_delta": float(np.median(delta)) if n else float("nan"),
                "hl_delta": float(
                    np.median(
                        (delta[:, None] + delta[None, :])
                        [np.triu_indices(len(delta))]
                        / 2.0
                    )
                ) if n else float("nan"),
                "wins": int(np.sum(delta > 0.0)) if n else 0,
                "losses": int(np.sum(delta < 0.0)) if n else 0,
                "ties": int(np.sum(delta == 0.0)) if n else 0,
                "cohen_d": float(cohen_d),
                "rank_biserial_r": float(rank_biserial),
                "wilcoxon_stat": float(stat),
                "wilcoxon_p": float(p),
            }
        )

    out = pd.DataFrame(rows)
    if apply_bh:
        out["bh_fdr_q"] = _bh_fdr(out["wilcoxon_p"].tolist())
        out["significant_fdr_0_05"] = (
            pd.to_numeric(out["bh_fdr_q"], errors="coerce").fillna(1.0) <= 0.05
        ).astype(int)
    return out


def _add_interpretation_scope(df_contrasts: pd.DataFrame) -> pd.DataFrame:
    out = df_contrasts.copy()
    out["interpretation_scope"] = "scientific"
    out["interpretation_note"] = ""
    for profile, meta in HIGH_RISK_ACTIVATION_ONLY_PROFILES.items():
        mask = out["left"].astype(str).eq(profile) | out["right"].astype(str).eq(profile)
        out.loc[mask, "interpretation_scope"] = str(meta.get("scope", "activation_only"))
        out.loc[mask, "interpretation_note"] = str(meta.get("rationale", ""))
    return out


def _activation_checks(df_runs: pd.DataFrame, *, min_rate: float) -> pd.DataFrame:
    work = df_runs.copy()
    work = work[work["profile"].isin(PROFILE_SET)]

    def _series(name: str, default: Any) -> pd.Series:
        if name in work.columns:
            return work[name]
        return pd.Series([default] * len(work), index=work.index)

    # Normalize commonly used columns.
    work["selected_features"] = pd.to_numeric(_series("selected_features", np.nan), errors="coerce")
    work["fs_cap_max_allowed"] = pd.to_numeric(_series("fs_cap_max_allowed", np.nan), errors="coerce")
    work["regime_policy_trigger_low_p_over_n"] = pd.to_numeric(
        _series("regime_policy_trigger_low_p_over_n", 0), errors="coerce"
    ).fillna(0.0)
    work["classifier_conformal_method"] = _series("classifier_conformal_method", "").astype(str)
    work["classifier_conformal_mapie_applied"] = pd.to_numeric(
        _series("classifier_conformal_mapie_applied", 0), errors="coerce"
    ).fillna(0.0)
    work["classifier_conformal_enabled"] = pd.to_numeric(
        _series("classifier_conformal_enabled", 0), errors="coerce"
    ).fillna(0.0)
    work["multiomics_adapter_applied"] = pd.to_numeric(
        _series("multiomics_adapter_applied", 0), errors="coerce"
    ).fillna(0.0)
    work["multiomics_n_blocks"] = pd.to_numeric(_series("multiomics_n_blocks", 0), errors="coerce").fillna(0.0)
    work["fs_ipss_use_eats_threshold"] = pd.to_numeric(
        _series("fs_ipss_use_eats_threshold", 0), errors="coerce"
    ).fillna(0.0)
    work["fs_stability_threshold_method"] = _series("fs_stability_threshold_method", "fixed").astype(str)

    parsed_methods: List[set[str]] = []
    for raw in _series("effective_enabled_methods", "[]"):
        parsed_methods.append({str(v) for v in _parse_listlike(raw)})
    work["_methods"] = parsed_methods

    rows: List[Dict[str, Any]] = []

    def _append_check(
        check_id: str,
        profile: str,
        description: str,
        eligible_mask: pd.Series,
        active_mask: pd.Series,
        threshold: Optional[float] = None,
    ) -> None:
        th = float(min_rate if threshold is None else threshold)
        eligible = int(eligible_mask.sum())
        active = int((eligible_mask & active_mask).sum())
        rate = float(active / eligible) if eligible > 0 else float("nan")
        passed = bool(eligible > 0 and np.isfinite(rate) and rate >= th)
        rows.append(
            {
                "check_id": check_id,
                "profile": profile,
                "description": description,
                "eligible_rows": eligible,
                "active_rows": active,
                "activation_rate": rate,
                "required_rate": th,
                "passed": int(passed),
            }
        )

    ref_mask = work["profile"].eq("v14_ref")
    low_p_mask = ref_mask & work["regime_policy_trigger_low_p_over_n"].ge(1.0)
    _append_check(
        "v14_ref_low_p_mode",
        "v14_ref",
        "Low p/n gate uses fast_univariate_filter",
        low_p_mask,
        _series("regime_policy_bypass_mode", "").astype(str).eq("fast_univariate_filter"),
        threshold=1.0,
    )
    cap_eligible = ref_mask & work["fs_cap_max_allowed"].gt(0)
    _append_check(
        "v14_ref_fs_cap",
        "v14_ref",
        "Selected features respect post-FS cap",
        cap_eligible,
        work["selected_features"].le(work["fs_cap_max_allowed"]),
        threshold=1.0,
    )

    ipss_mask = work["profile"].eq("v14_ipss")
    _append_check(
        "v14_ipss_active",
        "v14_ipss",
        "IPSS method present in effective enabled methods",
        ipss_mask,
        work["_methods"].apply(lambda s: "ipss" in s),
    )

    ipss_eats_mask = work["profile"].eq("v14_ipss_eats")
    _append_check(
        "v14_ipss_eats_active",
        "v14_ipss_eats",
        "IPSS+EATS toggles active",
        ipss_eats_mask,
        work["fs_ipss_use_eats_threshold"].ge(1.0)
        | work["fs_stability_threshold_method"].str.lower().eq("eats"),
    )

    gs_mask = work["profile"].eq("v14_group_sparse")
    _append_check(
        "v14_group_sparse_active",
        "v14_group_sparse",
        "group_sparse_lasso present in effective enabled methods",
        gs_mask,
        work["_methods"].apply(lambda s: "group_sparse_lasso" in s),
    )

    bc_mask = work["profile"].eq("v14_batch_combat")
    _append_check(
        "v14_batch_combat_active",
        "v14_batch_combat",
        "ComBat batch correction applied",
        bc_mask,
        _series("batch_correction_mode_applied", "").astype(str).eq("combat"),
    )

    for method in ("aps", "raps", "cross"):
        pid = f"v14_mapie_{method}"
        m_mask = work["profile"].eq(pid) & work["classifier_conformal_enabled"].ge(1.0)
        _append_check(
            f"{pid}_active",
            pid,
            f"MAPIE {method} applied",
            m_mask,
            work["classifier_conformal_method"].astype(str).str.lower().eq(method)
            & work["classifier_conformal_mapie_applied"].ge(1.0),
        )

    mo_mask = work["profile"].eq("v14_multiomics_adapter")
    _append_check(
        "v14_multiomics_adapter_active",
        "v14_multiomics_adapter",
        "Multi-omics adapter applied with >=2 blocks",
        mo_mask,
        work["multiomics_adapter_applied"].ge(1.0) & work["multiomics_n_blocks"].ge(2.0),
    )

    return pd.DataFrame(rows)


def _build_interpretation_caveats(df_runs: pd.DataFrame) -> pd.DataFrame:
    work = df_runs.copy()
    if "config" in work.columns:
        work = work[work["config"].astype(str).eq("baseline")]

    rows: List[Dict[str, Any]] = []
    for profile, meta in HIGH_RISK_ACTIVATION_ONLY_PROFILES.items():
        sub = work[work["profile"].astype(str).eq(str(profile))].copy()
        rows.append(
            {
                "profile": str(profile),
                "risk_level": str(meta.get("risk_level", "HIGH")),
                "interpretation_scope": str(meta.get("scope", "activation_only")),
                "rationale": str(meta.get("rationale", "")),
                "rows_observed": int(sub.shape[0]),
                "datasets_observed": int(sub["dataset_id"].nunique()) if "dataset_id" in sub.columns else 0,
                "mean_balanced_accuracy": (
                    float(pd.to_numeric(sub.get("balanced_accuracy"), errors="coerce").mean())
                    if sub.shape[0] > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _group_sparse_stability(df_runs: pd.DataFrame) -> pd.DataFrame:
    work = df_runs.copy()
    if "config" in work.columns:
        work = work[work["config"].astype(str).eq("baseline")]
    work = work[work["profile"].astype(str).eq("v14_group_sparse")].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "n_seed_runs",
                "mean_selected_features",
                "std_selected_features",
                "mean_pairwise_jaccard",
                "min_pairwise_jaccard",
                "mean_pairwise_overlap_coeff",
                "selection_signature_count",
                "dominant_signature_fraction",
                "stability_flag",
                "flag_reason",
                "metric_scope",
                "threshold_mean_jaccard",
                "threshold_dominant_signature_fraction",
            ]
        )

    if "selected_feature_indices_original" not in work.columns:
        return pd.DataFrame(
            [
                {
                    "dataset_id": "__missing_column__",
                    "n_seed_runs": 0,
                    "mean_selected_features": float("nan"),
                    "std_selected_features": float("nan"),
                    "mean_pairwise_jaccard": float("nan"),
                    "min_pairwise_jaccard": float("nan"),
                    "mean_pairwise_overlap_coeff": float("nan"),
                    "selection_signature_count": 0,
                    "dominant_signature_fraction": float("nan"),
                    "stability_flag": 1,
                    "flag_reason": "missing selected_feature_indices_original column",
                    "metric_scope": "feature_set_proxy",
                    "threshold_mean_jaccard": GROUP_STABILITY_MIN_MEAN_JACCARD,
                    "threshold_dominant_signature_fraction": GROUP_STABILITY_MIN_DOMINANT_SIGNATURE,
                }
            ]
        )

    work["_selected_feature_set"] = [
        _parse_int_set(raw) for raw in work["selected_feature_indices_original"]
    ]

    rows: List[Dict[str, Any]] = []
    for dataset_id, grp in work.groupby("dataset_id", dropna=False):
        sets: List[set[int]] = [set(v) for v in grp["_selected_feature_set"].tolist()]
        n_runs = int(len(sets))
        selected_counts = [int(len(s)) for s in sets]

        signature_counts: Dict[Tuple[int, ...], int] = {}
        for s in sets:
            key = tuple(sorted(s))
            signature_counts[key] = int(signature_counts.get(key, 0) + 1)
        dominant_frac = (
            float(max(signature_counts.values()) / max(1, n_runs)) if signature_counts else float("nan")
        )

        pairwise_jaccard: List[float] = []
        pairwise_overlap: List[float] = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                a = sets[i]
                b = sets[j]
                union = float(len(a | b))
                inter = float(len(a & b))
                jacc = float(inter / union) if union > 0 else 1.0
                min_size = float(min(len(a), len(b)))
                overlap = float(inter / min_size) if min_size > 0 else 1.0
                pairwise_jaccard.append(jacc)
                pairwise_overlap.append(overlap)

        mean_jaccard = (
            float(np.mean(np.asarray(pairwise_jaccard, dtype=float)))
            if pairwise_jaccard
            else float("nan")
        )
        min_jaccard = (
            float(np.min(np.asarray(pairwise_jaccard, dtype=float)))
            if pairwise_jaccard
            else float("nan")
        )
        mean_overlap = (
            float(np.mean(np.asarray(pairwise_overlap, dtype=float)))
            if pairwise_overlap
            else float("nan")
        )
        mean_sel = (
            float(np.mean(np.asarray(selected_counts, dtype=float)))
            if selected_counts
            else float("nan")
        )
        std_sel = (
            float(np.std(np.asarray(selected_counts, dtype=float), ddof=1))
            if len(selected_counts) > 1
            else 0.0
        )

        reasons: List[str] = []
        if n_runs < 2:
            reasons.append("insufficient_seed_runs")
        if np.isfinite(mean_jaccard) and mean_jaccard < GROUP_STABILITY_MIN_MEAN_JACCARD:
            reasons.append("low_pairwise_jaccard")
        if (
            np.isfinite(dominant_frac)
            and dominant_frac < GROUP_STABILITY_MIN_DOMINANT_SIGNATURE
        ):
            reasons.append("low_dominant_signature_fraction")

        rows.append(
            {
                "dataset_id": str(dataset_id),
                "n_seed_runs": int(n_runs),
                "mean_selected_features": float(mean_sel),
                "std_selected_features": float(std_sel),
                "mean_pairwise_jaccard": float(mean_jaccard),
                "min_pairwise_jaccard": float(min_jaccard),
                "mean_pairwise_overlap_coeff": float(mean_overlap),
                "selection_signature_count": int(len(signature_counts)),
                "dominant_signature_fraction": float(dominant_frac),
                "stability_flag": int(len(reasons) > 0),
                "flag_reason": ";".join(reasons),
                "metric_scope": "feature_set_proxy_for_correlation_group_stability",
                "threshold_mean_jaccard": float(GROUP_STABILITY_MIN_MEAN_JACCARD),
                "threshold_dominant_signature_fraction": float(GROUP_STABILITY_MIN_DOMINANT_SIGNATURE),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["stability_flag", "mean_pairwise_jaccard"], ascending=[False, True]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Val-14 effect matrix (paired contrasts + BH-FDR + activation checks)."
    )
    parser.add_argument("--root-out-dir", type=str, required=True)
    parser.add_argument(
        "--plan",
        type=str,
        default="pod_validation/plan_6_val14.json",
        help="Plan JSON used for aggregation.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: <root-out-dir>/_analysis).",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip aggregate step and use existing _aggregate CSVs.",
    )
    parser.add_argument(
        "--activation-gate",
        action="store_true",
        help="Fail with non-zero exit code when activation checks do not pass.",
    )
    parser.add_argument(
        "--activation-min-rate",
        type=float,
        default=0.90,
        help="Minimum activation rate required per activation check.",
    )
    args = parser.parse_args()

    root_out = Path(args.root_out_dir).expanduser().resolve()
    if not root_out.exists():
        raise FileNotFoundError(f"Missing root output dir: {root_out}")

    plan_path = Path(args.plan).expanduser().resolve()
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing plan file: {plan_path}")

    aggregate_dir = root_out / "_aggregate"
    if not args.skip_aggregate:
        aggregate(root_out, plan_path=plan_path, out_dir=aggregate_dir)

    runs_path = aggregate_dir / "benchmark_df_fs_runs__all_jobs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(
            f"Missing aggregated runs CSV: {runs_path}. Run aggregation first or omit --skip-aggregate."
        )

    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir or "").strip() else (root_out / "_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_runs = pd.read_csv(runs_path)
    if "job_id" not in df_runs.columns:
        raise RuntimeError("Runs CSV is missing required column 'job_id'.")

    df_runs["profile"] = df_runs["job_id"].map(_extract_profile)
    df_runs = df_runs[df_runs["profile"].isin(PROFILE_SET)].copy()

    primary = _compute_contrast_table(df_runs, PRIMARY_CONTRASTS, apply_bh=True)
    primary = _add_interpretation_scope(primary)
    anchors = _compute_contrast_table(df_runs, ANCHOR_CONTRASTS, apply_bh=False)
    activation = _activation_checks(df_runs, min_rate=float(args.activation_min_rate))
    caveats = _build_interpretation_caveats(df_runs)
    group_sparse_stability = _group_sparse_stability(df_runs)

    primary_csv = out_dir / "val14_primary_contrasts.csv"
    anchor_csv = out_dir / "val14_anchor_contrasts.csv"
    activation_csv = out_dir / "val14_activation_checks.csv"
    caveats_csv = out_dir / "val14_interpretation_caveats.csv"
    group_sparse_stability_csv = out_dir / "val14_group_sparse_stability.csv"

    primary.to_csv(primary_csv, index=False)
    anchors.to_csv(anchor_csv, index=False)
    activation.to_csv(activation_csv, index=False)
    caveats.to_csv(caveats_csv, index=False)
    group_sparse_stability.to_csv(group_sparse_stability_csv, index=False)

    gate_required = {
        "v14_ref_low_p_mode",
        "v14_ref_fs_cap",
        "v14_ipss_active",
        "v14_ipss_eats_active",
        "v14_group_sparse_active",
        "v14_batch_combat_active",
        "v14_mapie_aps_active",
        "v14_mapie_raps_active",
        "v14_mapie_cross_active",
        "v14_multiomics_adapter_active",
    }
    gate_subset = activation[activation["check_id"].isin(gate_required)].copy()
    gate_pass = bool((gate_subset["passed"].astype(int) == 1).all()) if not gate_subset.empty else False

    summary = {
        "rows": int(df_runs.shape[0]),
        "profiles_observed": sorted({str(p) for p in df_runs["profile"].unique()}),
        "primary_contrast_count": int(primary.shape[0]),
        "primary_significant_fdr_0_05": int(primary.get("significant_fdr_0_05", pd.Series(dtype=int)).sum()),
        "high_risk_activation_only_profiles": sorted(HIGH_RISK_ACTIVATION_ONLY_PROFILES.keys()),
        "activation_check_count": int(activation.shape[0]),
        "activation_gate_required_checks": sorted(gate_required),
        "activation_gate_pass": bool(gate_pass),
        "activation_min_rate": float(args.activation_min_rate),
        "group_sparse_stability_rows": int(group_sparse_stability.shape[0]),
        "group_sparse_stability_flagged": int(
            pd.to_numeric(group_sparse_stability.get("stability_flag"), errors="coerce").fillna(0).sum()
        ),
        "group_sparse_stability_thresholds": {
            "mean_pairwise_jaccard_min": float(GROUP_STABILITY_MIN_MEAN_JACCARD),
            "dominant_signature_fraction_min": float(GROUP_STABILITY_MIN_DOMINANT_SIGNATURE),
        },
    }

    summary_path = out_dir / "val14_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {primary_csv}")
    print(f"Wrote: {anchor_csv}")
    print(f"Wrote: {activation_csv}")
    print(f"Wrote: {caveats_csv}")
    print(f"Wrote: {group_sparse_stability_csv}")
    print(f"Wrote: {summary_path}")
    print(
        "Interpretation caveat: v14_batch_combat and v14_multiomics_adapter are "
        "activation/infrastructure checks only (not scientific evidence)."
    )
    flagged = int(
        pd.to_numeric(group_sparse_stability.get("stability_flag"), errors="coerce").fillna(0).sum()
    )
    print(
        "Group-sparse stability flags: "
        + f"{flagged}/{int(group_sparse_stability.shape[0])} datasets "
        + f"(thresholds: mean_jaccard>={GROUP_STABILITY_MIN_MEAN_JACCARD:.2f}, "
        + f"dominant_signature>={GROUP_STABILITY_MIN_DOMINANT_SIGNATURE:.2f})"
    )
    print(
        "Activation gate: "
        + ("PASS" if gate_pass else "FAIL")
        + f" (required={len(gate_required)} checks, min_rate={float(args.activation_min_rate):.2f})"
    )

    if bool(args.activation_gate) and not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

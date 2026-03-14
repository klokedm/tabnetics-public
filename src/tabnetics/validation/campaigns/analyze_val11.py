#!/usr/bin/env python3
"""Analyze Val-11 v2 seven-profile 2x2 factorial paired results.

Profiles:
  A simple_control                - simple FS + simple classifier (baseline)
  B mnpo_improved                  - MNPO FS + MNPO classifier (full improved)
  C simple_fs_mnpo_clf             - simple FS + MNPO classifier (isolates classifier)
  D mnpo_fs_simple_clf             - MNPO FS + simple classifier (isolates FS)
  E mnpo_improved_cvar             - B + CVaR tail-risk oracle
  F mnpo_improved_experimental_fs  - B + 11 experimental FS features
  G mnpo_improved_alt_screening    - B + alt screening/estimator/knockoffs

Factorial contrasts (from the 2x2 {FS} x {Classifier} design):
  - Classifier main effect:  (B + C) / 2 - (A + D) / 2
  - FS main effect:          (B + D) / 2 - (A + C) / 2
  - FS x Classifier interaction: B - C - D + A
  - CVaR tail-risk effect:   E - B
  - Experimental FS effect:  F - B
  - Alt screening effect:    G - B

All pairwise comparisons use Wilcoxon signed-rank with Bonferroni correction
(21 pairs, alpha_adj ~= 0.0024).
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tabnetics.validation.core.aggregate import aggregate


# -- Profile definitions -------------------------------------------------------
PROFILE_A = "simple_control"
PROFILE_B = "mnpo_improved"
PROFILE_C = "simple_fs_mnpo_clf"
PROFILE_D = "mnpo_fs_simple_clf"
PROFILE_E = "mnpo_improved_cvar"
PROFILE_F = "mnpo_improved_experimental_fs"
PROFILE_G = "mnpo_improved_alt_screening"

ALL_PROFILES = [PROFILE_A, PROFILE_B, PROFILE_C, PROFILE_D, PROFILE_E, PROFILE_F, PROFILE_G]
PROFILE_SET = set(ALL_PROFILES)

# 2x2 factorial cell mapping
FACTORIAL_CELLS = {
    "simple_fs__simple_clf": PROFILE_A,
    "simple_fs__mnpo_clf": PROFILE_C,
    "mnpo_fs__simple_clf": PROFILE_D,
    "mnpo_fs__mnpo_clf": PROFILE_B,
}

# Bonferroni-corrected alpha for 21 pairwise comparisons (C(7,2))
N_PAIRS = 21
ALPHA_BONFERRONI = 0.05 / 21  # ~0.00238


def _safe_wilcoxon(values: pd.Series) -> Dict[str, float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n": 0}
    if np.allclose(clean.to_numpy(), 0.0):
        return {"statistic": 0.0, "pvalue": 1.0, "n": int(clean.shape[0])}
    try:
        stat, p = wilcoxon(clean.to_numpy(), alternative="two-sided", zero_method="wilcox")
        return {"statistic": float(stat), "pvalue": float(p), "n": int(clean.shape[0])}
    except ValueError:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n": int(clean.shape[0])}


def _wins(values: pd.Series) -> Dict[str, int]:
    clean = values.dropna().astype(float)
    return {
        "left_wins": int((clean > 0).sum()),
        "right_wins": int((clean < 0).sum()),
        "ties": int((clean == 0).sum()),
    }


def _extract_profile(job_id: str) -> str:
    m = re.match(r"^val11/([^/]+)/", str(job_id))
    return str(m.group(1)) if m else "unknown"


def _compute_factorial_contrasts(
    ba_pivot: pd.DataFrame,
) -> Dict[str, Dict[str, object]]:
    """Compute the four factorial contrasts from the 2x2 design."""
    contrasts: Dict[str, Dict[str, object]] = {}

    # Need all 4 factorial profiles present
    factorial_profiles = [PROFILE_A, PROFILE_B, PROFILE_C, PROFILE_D]
    has_factorial = ba_pivot.dropna(subset=factorial_profiles)

    if has_factorial.empty:
        return contrasts

    a = has_factorial[PROFILE_A].to_numpy()
    b = has_factorial[PROFILE_B].to_numpy()
    c = has_factorial[PROFILE_C].to_numpy()
    d = has_factorial[PROFILE_D].to_numpy()

    # Classifier main effect: MNPO clf vs simple clf
    # (B + C) / 2 - (A + D) / 2  (positive = MNPO clf better)
    clf_effect = (b + c) / 2.0 - (a + d) / 2.0
    clf_series = pd.Series(clf_effect)
    contrasts["classifier_main_effect"] = {
        "description": "MNPO ensemble clf vs simple single-model clf",
        "formula": "(B + C) / 2 - (A + D) / 2",
        "positive_means": "MNPO classifier better",
        "mean": float(clf_series.mean()),
        "median": float(clf_series.median()),
        "std": float(clf_series.std()),
        "wilcoxon": _safe_wilcoxon(clf_series),
        "n": int(has_factorial.shape[0]),
    }

    # FS main effect: MNPO FS vs simple FS
    # (B + D) / 2 - (A + C) / 2  (positive = MNPO FS better)
    fs_effect = (b + d) / 2.0 - (a + c) / 2.0
    fs_series = pd.Series(fs_effect)
    contrasts["fs_main_effect"] = {
        "description": "MNPO broad FS vs simple compact FS",
        "formula": "(B + D) / 2 - (A + C) / 2",
        "positive_means": "MNPO FS better",
        "mean": float(fs_series.mean()),
        "median": float(fs_series.median()),
        "std": float(fs_series.std()),
        "wilcoxon": _safe_wilcoxon(fs_series),
        "n": int(has_factorial.shape[0]),
    }

    # Interaction: FS x Classifier
    # B - C - D + A  (positive = synergy, negative = redundancy)
    interaction = b - c - d + a
    int_series = pd.Series(interaction)
    contrasts["fs_x_classifier_interaction"] = {
        "description": "FS x Classifier interaction (synergy vs redundancy)",
        "formula": "B - C - D + A",
        "positive_means": "stages have synergy (combined > sum of parts)",
        "mean": float(int_series.mean()),
        "median": float(int_series.median()),
        "std": float(int_series.std()),
        "wilcoxon": _safe_wilcoxon(int_series),
        "n": int(has_factorial.shape[0]),
    }

    # CVaR tail-risk effect: E - B  (positive = CVaR helps)
    if PROFILE_E in ba_pivot.columns:
        has_cvar = has_factorial.dropna(subset=[PROFILE_E])
        if not has_cvar.empty:
            cvar_effect = has_cvar[PROFILE_E].to_numpy() - has_cvar[PROFILE_B].to_numpy()
            cvar_series = pd.Series(cvar_effect)
            contrasts["cvar_tail_risk_effect"] = {
                "description": "CVaR oracle value (tail-risk improvement)",
                "formula": "E - B",
                "positive_means": "CVaR oracle helps",
                "mean": float(cvar_series.mean()),
                "median": float(cvar_series.median()),
                "std": float(cvar_series.std()),
                "wilcoxon": _safe_wilcoxon(cvar_series),
                "n": int(has_cvar.shape[0]),
                "min_ba_E": float(has_cvar[PROFILE_E].min()),
                "min_ba_B": float(has_cvar[PROFILE_B].min()),
            }

    # Experimental FS features effect: F - B  (positive = experimental FS helps)
    if PROFILE_F in ba_pivot.columns:
        has_exp = has_factorial.dropna(subset=[PROFILE_F])
        if not has_exp.empty:
            exp_effect = has_exp[PROFILE_F].to_numpy() - has_exp[PROFILE_B].to_numpy()
            exp_series = pd.Series(exp_effect)
            contrasts["experimental_fs_effect"] = {
                "description": "Experimental FS features value (11 features stacked on B)",
                "formula": "F - B",
                "positive_means": "Experimental FS features help",
                "mean": float(exp_series.mean()),
                "median": float(exp_series.median()),
                "std": float(exp_series.std()),
                "wilcoxon": _safe_wilcoxon(exp_series),
                "n": int(has_exp.shape[0]),
            }

    # Alt screening/fitting effect: G - B  (positive = alt methods help)
    if PROFILE_G in ba_pivot.columns:
        has_alt = has_factorial.dropna(subset=[PROFILE_G])
        if not has_alt.empty:
            alt_effect = has_alt[PROFILE_G].to_numpy() - has_alt[PROFILE_B].to_numpy()
            alt_series = pd.Series(alt_effect)
            contrasts["alt_screening_effect"] = {
                "description": "Alt screening/fitting value (STIR, MPS, DeepDRK vs defaults)",
                "formula": "G - B",
                "positive_means": "Alt screening/fitting methods help",
                "mean": float(alt_series.mean()),
                "median": float(alt_series.median()),
                "std": float(alt_series.std()),
                "wilcoxon": _safe_wilcoxon(alt_series),
                "n": int(has_alt.shape[0]),
            }

    return contrasts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Val-11 v2 seven-profile 2x2 factorial paired comparisons."
    )
    parser.add_argument(
        "--root-out-dir",
        type=str,
        required=True,
        help="Val-11 v2 merged output root (contains val11/... and _status).",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="pod_validation/plan_4_val11.json",
        help="Plan JSON used for aggregation.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Analysis output directory (default: <root-out-dir>/_analysis).",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip aggregate step and read existing _aggregate CSVs.",
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

    summary_path = aggregate_dir / "benchmark_df_fs_summary__all_jobs.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing aggregated summary CSV: {summary_path}. "
            "Run aggregation first or omit --skip-aggregate."
        )

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root_out / "_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    if "job_id" not in df.columns:
        raise ValueError(f"Expected 'job_id' column in {summary_path}")

    df["profile"] = df["job_id"].map(_extract_profile)
    df = df[df["profile"].isin(PROFILE_SET)].copy()
    if df.empty:
        raise ValueError("No Val-11 v2 rows found in aggregated summary.")

    required_cols: List[str] = ["dataset_id", "tier", "profile", "balanced_accuracy_mean", "fs_time_sec_mean"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in summary: {missing_cols}")

    # -- Aggregate per dataset x profile ----------------------------------------
    grouped = (
        df.groupby(["dataset_id", "tier", "profile"], as_index=False)[["balanced_accuracy_mean", "fs_time_sec_mean"]]
        .mean()
        .rename(columns={"balanced_accuracy_mean": "ba", "fs_time_sec_mean": "fs_time"})
    )

    ba_pivot = grouped.pivot_table(index=["dataset_id", "tier"], columns="profile", values="ba", aggfunc="first")
    fs_pivot = grouped.pivot_table(index=["dataset_id", "tier"], columns="profile", values="fs_time", aggfunc="first")
    merged = ba_pivot.join(fs_pivot, lsuffix="_ba", rsuffix="_fs").reset_index()

    # Complete datasets: need at least A, B, C, D for factorial analysis
    factorial_profiles = [PROFILE_A, PROFILE_B, PROFILE_C, PROFILE_D]
    complete = merged.dropna(subset=factorial_profiles).copy()
    if complete.empty:
        raise ValueError("No complete paired datasets with all four factorial profiles present.")

    print(f"[val11] paired_datasets={complete.shape[0]} (factorial-complete)")

    # -- 1. Factorial contrasts -------------------------------------------------
    contrasts = _compute_factorial_contrasts(complete)

    print("\n=== Factorial Contrasts ===")
    for name, c in contrasts.items():
        sig = "***" if c["wilcoxon"]["pvalue"] < ALPHA_BONFERRONI else ""
        print(f"  {name}: mean={c['mean']:+.4f}, p={c['wilcoxon']['pvalue']:.4g} {sig}")

    # -- 2. All 10 pairwise Wilcoxon tests (Bonferroni) -------------------------
    pairwise_results: List[Dict[str, object]] = []
    for left, right in combinations(ALL_PROFILES, 2):
        if left not in complete.columns or right not in complete.columns:
            continue
        pair_df = complete.dropna(subset=[left, right])
        if pair_df.empty:
            continue
        delta = pair_df[left] - pair_df[right]
        w = _safe_wilcoxon(delta)
        wins = _wins(delta)
        pairwise_results.append({
            "left": left,
            "right": right,
            "mean_delta": float(delta.mean()),
            "median_delta": float(delta.median()),
            "wilcoxon": w,
            "wins": wins,
            "significant_bonferroni": bool(w["pvalue"] < ALPHA_BONFERRONI),
        })

    print(f"\n=== Pairwise Comparisons (n={len(pairwise_results)}, alpha_adj={ALPHA_BONFERRONI}) ===")
    for pr in pairwise_results:
        sig = "***" if pr["significant_bonferroni"] else ""
        print(f"  {pr['left']} vs {pr['right']}: d={pr['mean_delta']:+.4f}, p={pr['wilcoxon']['pvalue']:.4g} {sig}")

    # -- 3. Per-profile summary -------------------------------------------------
    profile_summary: Dict[str, Dict[str, float]] = {}
    for p in ALL_PROFILES:
        if p in complete.columns:
            vals = complete[p].dropna()
            fs_col = f"{p}_fs"
            fs_vals = complete[fs_col].dropna() if fs_col in complete.columns else pd.Series(dtype=float)
            profile_summary[p] = {
                "mean_ba": float(vals.mean()),
                "median_ba": float(vals.median()),
                "std_ba": float(vals.std()),
                "min_ba": float(vals.min()),
                "n": int(vals.shape[0]),
                "mean_fs_time": float(fs_vals.mean()) if not fs_vals.empty else float("nan"),
            }

    print("\n=== Per-Profile Summary ===")
    for p in ALL_PROFILES:
        if p in profile_summary:
            s = profile_summary[p]
            print(f"  {p}: mean_ba={s['mean_ba']:.4f}, min_ba={s['min_ba']:.4f}, fs_time={s['mean_fs_time']:.0f}s")

    # -- 4. Per-tier analysis ---------------------------------------------------
    tier_rows: List[Dict[str, object]] = []
    for tier_name, tier_df in complete.groupby("tier"):
        row: Dict[str, object] = {"tier": str(tier_name), "n": int(tier_df.shape[0])}
        for p in ALL_PROFILES:
            if p in tier_df.columns:
                row[f"mean_ba_{p}"] = float(tier_df[p].mean())
        # Factorial contrasts within tier
        tier_contrasts = _compute_factorial_contrasts(tier_df)
        for cname, cval in tier_contrasts.items():
            row[f"{cname}_mean"] = cval["mean"]
            row[f"{cname}_p"] = cval["wilcoxon"]["pvalue"]
        tier_rows.append(row)

    print("\n=== Per-Tier Factorial Contrasts ===")
    for tr in tier_rows:
        clf_mean = tr.get("classifier_main_effect_mean", "N/A")
        fs_mean = tr.get("fs_main_effect_mean", "N/A")
        print(f"  {tr['tier']} (n={tr['n']}): clf_effect={clf_mean}, fs_effect={fs_mean}")

    # -- 5. Success criteria evaluation -----------------------------------------
    criteria: Dict[str, Dict[str, object]] = {}

    # C1: Classifier main effect positive and significant
    if "classifier_main_effect" in contrasts:
        c1 = contrasts["classifier_main_effect"]
        criteria["C1_classifier_main_effect"] = {
            "pass": bool(c1["mean"] > 0 and c1["wilcoxon"]["pvalue"] < ALPHA_BONFERRONI),
            "mean": c1["mean"],
            "pvalue": c1["wilcoxon"]["pvalue"],
        }

    # C2: FS main effect magnitude and direction
    if "fs_main_effect" in contrasts:
        c2 = contrasts["fs_main_effect"]
        criteria["C2_fs_main_effect"] = {
            "direction": "positive" if c2["mean"] > 0 else "negative",
            "significant": bool(c2["wilcoxon"]["pvalue"] < ALPHA_BONFERRONI),
            "mean": c2["mean"],
            "pvalue": c2["wilcoxon"]["pvalue"],
        }

    # C3: Interaction assessment
    if "fs_x_classifier_interaction" in contrasts:
        c3 = contrasts["fs_x_classifier_interaction"]
        criteria["C3_interaction"] = {
            "type": "synergy" if c3["mean"] > 0 else "redundancy" if c3["mean"] < 0 else "additive",
            "significant": bool(c3["wilcoxon"]["pvalue"] < ALPHA_BONFERRONI),
            "mean": c3["mean"],
            "pvalue": c3["wilcoxon"]["pvalue"],
        }

    # C4: CVaR tail-risk
    if "cvar_tail_risk_effect" in contrasts:
        c4 = contrasts["cvar_tail_risk_effect"]
        min_e = c4.get("min_ba_E", float("nan"))
        min_b = c4.get("min_ba_B", float("nan"))
        criteria["C4_cvar_tail_risk"] = {
            "pass": bool(min_e > min_b + 0.01) if not np.isnan(min_e) else False,
            "min_ba_E": min_e,
            "min_ba_B": min_b,
            "mean_effect": c4["mean"],
        }

    # C5: No easy-tier regression (B >= A - 0.005 on easy tier)
    easy_tier = [t for t in tier_rows if t["tier"] == "easy"]
    if easy_tier:
        ba_a_easy = easy_tier[0].get(f"mean_ba_{PROFILE_A}", float("nan"))
        ba_b_easy = easy_tier[0].get(f"mean_ba_{PROFILE_B}", float("nan"))
        if not np.isnan(ba_a_easy) and not np.isnan(ba_b_easy):
            criteria["C5_no_easy_regression"] = {
                "pass": bool(ba_b_easy >= ba_a_easy - 0.005),
                "ba_A_easy": float(ba_a_easy),
                "ba_B_easy": float(ba_b_easy),
            }

    # C6: Runtime improvement (B FS time / Val-10 MNPO FS time <= 0.65)
    if PROFILE_B in profile_summary and not np.isnan(profile_summary[PROFILE_B]["mean_fs_time"]):
        criteria["C6_runtime_improvement"] = {
            "mean_fs_time_B": profile_summary[PROFILE_B]["mean_fs_time"],
            "note": "Compare against Val-10 MNPO FS time (3984s) manually",
        }

    # C7: Experimental FS features (F vs B)
    if "experimental_fs_effect" in contrasts:
        c7 = contrasts["experimental_fs_effect"]
        criteria["C7_experimental_fs_value"] = {
            "pass": bool(c7["mean"] > 0 and c7["wilcoxon"]["pvalue"] < ALPHA_BONFERRONI),
            "mean_effect": c7["mean"],
            "pvalue": c7["wilcoxon"]["pvalue"],
            "note": "Positive means experimental FS features improve BA vs vanilla B",
        }

    # C8: Alt screening/fitting (G vs B)
    if "alt_screening_effect" in contrasts:
        c8 = contrasts["alt_screening_effect"]
        criteria["C8_alt_screening_value"] = {
            "pass": bool(c8["mean"] > 0 and c8["wilcoxon"]["pvalue"] < ALPHA_BONFERRONI),
            "mean_effect": c8["mean"],
            "pvalue": c8["wilcoxon"]["pvalue"],
            "note": "Positive means STIR/MPS/DeepDRK improve BA vs evalue/MLE/default",
        }

    print("\n=== Success Criteria ===")
    for cname, cval in criteria.items():
        status = cval.get("pass", "assess")
        print(f"  {cname}: {status} -- {cval}")

    # -- 6. Write outputs -------------------------------------------------------
    dataset_table_path = out_dir / "val11v2_dataset_comparison.csv"
    report_json_path = out_dir / "val11v2_comparison_report.json"
    report_md_path = out_dir / "VAL11_V2_ANALYSIS.md"

    # Dataset-level CSV with all profile BA values
    out_cols = ["dataset_id", "tier"]
    for p in ALL_PROFILES:
        if p in complete.columns:
            out_cols.append(p)
    for p in ALL_PROFILES:
        fs_col = f"{p}_fs"
        if fs_col in complete.columns:
            out_cols.append(fs_col)
    complete[out_cols].sort_values(["tier", "dataset_id"]).to_csv(dataset_table_path, index=False)

    # Full JSON report
    report = {
        "summary_csv": str(summary_path),
        "n_paired_datasets": int(complete.shape[0]),
        "profiles": profile_summary,
        "factorial_contrasts": {k: _json_safe(v) for k, v in contrasts.items()},
        "pairwise_comparisons": pairwise_results,
        "per_tier": tier_rows,
        "success_criteria": {k: _json_safe(v) for k, v in criteria.items()},
        "alpha_bonferroni": ALPHA_BONFERRONI,
        "n_pairwise_tests": N_PAIRS,
    }
    report_json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Markdown report
    md_lines = _build_markdown_report(report, contrasts, pairwise_results, tier_rows, criteria, profile_summary)
    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\n[val11v2] wrote={dataset_table_path}")
    print(f"[val11v2] wrote={report_json_path}")
    print(f"[val11v2] wrote={report_md_path}")


def _json_safe(obj: object) -> object:
    """Convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _build_markdown_report(
    report: Dict,
    contrasts: Dict,
    pairwise_results: List[Dict],
    tier_rows: List[Dict],
    criteria: Dict,
    profile_summary: Dict,
) -> List[str]:
    """Build a markdown report from the analysis results."""
    lines = [
        "# Val-11 v2 Factorial Analysis",
        "",
        f"**Paired datasets:** {report['n_paired_datasets']}",
        f"**Bonferroni-corrected alpha:** {ALPHA_BONFERRONI} (for {N_PAIRS} pairwise tests)",
        "",
        "## 2x2 Factorial Design",
        "",
        "| | Simple Classifier (A, D) | MNPO Ensemble Classifier (B, C) |",
        "|---|---|---|",
    ]

    def _ba_str(p: str) -> str:
        if p in profile_summary:
            return f"{profile_summary[p]['mean_ba']:.4f}"
        return "N/A"

    lines.append(f"| **Simple FS** | A: {_ba_str(PROFILE_A)} | C: {_ba_str(PROFILE_C)} |")
    lines.append(f"| **MNPO FS** | D: {_ba_str(PROFILE_D)} | B: {_ba_str(PROFILE_B)} |")

    lines += [
        "",
        f"Profile E (B + CVaR): {_ba_str(PROFILE_E)}",
        f"Profile F (B + experimental FS): {_ba_str(PROFILE_F)}",
        f"Profile G (B + alt screening): {_ba_str(PROFILE_G)}",
        "",
        "## Factorial Contrasts",
        "",
        "| Contrast | Formula | Mean | p-value | Significant |",
        "|----------|---------|------|---------|-------------|",
    ]
    for cname, cval in contrasts.items():
        sig = "Yes ***" if cval["wilcoxon"]["pvalue"] < ALPHA_BONFERRONI else "No"
        lines.append(f"| {cname} | {cval['formula']} | {cval['mean']:+.4f} | {cval['wilcoxon']['pvalue']:.4g} | {sig} |")

    lines += [
        "",
        "## Pairwise Comparisons (Bonferroni-corrected)",
        "",
        "| Left | Right | Mean d | p-value | Sig |",
        "|------|-------|--------|---------|-----|",
    ]
    for pr in pairwise_results:
        sig = "***" if pr["significant_bonferroni"] else ""
        lines.append(
            f"| {pr['left']} | {pr['right']} | {pr['mean_delta']:+.4f} "
            f"| {pr['wilcoxon']['pvalue']:.4g} | {sig} |"
        )

    lines += [
        "",
        "## Per-Profile Summary",
        "",
        "| Profile | Mean BA | Min BA | Std | FS Time (s) |",
        "|---------|---------|--------|-----|-------------|",
    ]
    for p in ALL_PROFILES:
        if p in profile_summary:
            s = profile_summary[p]
            lines.append(f"| {p} | {s['mean_ba']:.4f} | {s['min_ba']:.4f} | {s['std_ba']:.4f} | {s['mean_fs_time']:.0f} |")

    lines += ["", "## Per-Tier Analysis", ""]
    if tier_rows:
        tier_df = pd.DataFrame(tier_rows)
        display_cols = ["tier", "n"]
        for p in ALL_PROFILES:
            col = f"mean_ba_{p}"
            if col in tier_df.columns:
                display_cols.append(col)
        for c in ["classifier_main_effect_mean", "fs_main_effect_mean"]:
            if c in tier_df.columns:
                display_cols.append(c)
        existing = [c for c in display_cols if c in tier_df.columns]
        lines.append(tier_df[existing].to_markdown(index=False))

    lines += [
        "",
        "## Success Criteria",
        "",
        "| Criterion | Result | Details |",
        "|-----------|--------|---------|",
    ]
    for cname, cval in criteria.items():
        status = cval.get("pass", "assess")
        details = ", ".join(f"{k}={v}" for k, v in cval.items() if k != "pass")
        lines.append(f"| {cname} | {status} | {details} |")

    lines += [
        "",
        "## Key Interpretation",
        "",
        "- If **C1 significant** and **C2 not significant**: MNPO's value comes from "
        "the ensemble classifier, not game-theoretic FS. Redirect development toward "
        "classifier optimization.",
        "- If **C1 and C2 both significant** with **C3 approx 0**: both stages add value "
        "independently. Keep full MNPO with pruned methods.",
        "- If **C3 > 0 (synergy)**: FS and classifier work together; full pipeline justified.",
        "- If **C4 passes**: CVaR oracle improves worst-case; add to default config.",
        "- If **F > B** significantly: experimental FS features add value; promote to default.",
        "- If **G > B** significantly: alt screening/fitting methods improve results; "
        "consider replacing evalue/MLE/default knockoffs.",
    ]

    return lines


if __name__ == "__main__":
    main()

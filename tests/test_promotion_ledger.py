from __future__ import annotations

import json
import sys
from pathlib import Path

from tabnetics.validation.core.ledger import (
    append_rows,
    infer_intervention_class,
    load_rows,
    report,
    rows_from_val4_report,
)


def _sample_val4_report() -> dict:
    return {
        "promotion_decision": {
            "recommended_profile": "broad_oracle",
            "promotion_gate": "strict",
            "confidence": "high",
        },
        "pair_reports": {
            "B_vs_A": {
                "candidate": "broad_oracle",
                "baseline": "baseline",
                "observed_dataset_overlap": 3,
                "mean_delta": 0.02,
                "wilcoxon_p": 0.031,
                "tier_breakdown": {
                    "easy": {
                        "n": 2,
                        "mean_delta": 0.015,
                        "worst_delta": 0.0,
                        "best_delta": 0.03,
                        "improved": 1,
                        "same": 1,
                        "regressed": 0,
                    },
                    "hard": {
                        "n": 1,
                        "mean_delta": 0.03,
                        "worst_delta": 0.03,
                        "best_delta": 0.03,
                        "improved": 1,
                        "same": 0,
                        "regressed": 0,
                    },
                },
                "gate": {
                    "strict": {"verdict": "PASS", "overall_mean": 0.02},
                    "promotion_decision": {"mode_used": "strict", "verdict": "PASS", "promote": True},
                },
                "gaming_detectors": {"any_flagged": False},
            },
            "P_vs_A": {
                "candidate": "prefilter_probe",
                "baseline": "baseline",
                "observed_dataset_overlap": 2,
                "mean_delta": -0.01,
                "wilcoxon_p": 0.5,
                "tier_breakdown": {
                    "medium": {
                        "n": 2,
                        "mean_delta": -0.01,
                        "worst_delta": -0.02,
                        "best_delta": 0.0,
                        "improved": 0,
                        "same": 1,
                        "regressed": 1,
                    }
                },
                "gate": {
                    "strict": {"verdict": "FAIL", "overall_mean": -0.01},
                    "promotion_decision": {"mode_used": "strict", "verdict": "FAIL", "promote": False},
                },
                "gaming_detectors": {"any_flagged": True},
            },
        },
    }


def test_rows_from_val4_report_and_report_aggregate(tmp_path: Path) -> None:
    rows = rows_from_val4_report(
        _sample_val4_report(),
        campaign="val4",
        evidence_links=["https://github.com/klokedm/tabnetics/issues/206#issuecomment-1"],
        source_artifacts={"json_report": tmp_path / "report.json"},
    )
    ledger = tmp_path / "promotion_attempts.jsonl"

    assert append_rows(ledger, rows) == 2
    loaded = load_rows(ledger)
    summary = report(loaded)

    assert loaded[0]["schema_version"] == "tabnetics_promotion_ledger_v1"
    assert loaded[0]["profile_id"] == "broad_oracle"
    assert loaded[0]["parent_baseline"] == "baseline"
    assert loaded[0]["intervention_class"] == "oracle"
    assert loaded[0]["wins"] == 2
    assert loaded[0]["ties"] == 1
    assert loaded[0]["losses"] == 0
    assert loaded[0]["wilcoxon_p"] == 0.031
    assert "champion" not in loaded[0]
    assert "challenger" not in loaded[0]
    assert summary["by_intervention_class"]["oracle"]["dataset_win_rate"] == 2 / 3
    assert summary["by_intervention_class"]["oracle"]["mean_verified_delta_balanced_accuracy"] == 0.02
    assert summary["by_intervention_class"]["prefilter"]["losses"] == 1


def test_infer_intervention_class_uses_stable_taxonomy() -> None:
    assert infer_intervention_class("rank_prefilter_probe") == "prefilter"
    assert infer_intervention_class("broad_oracle") == "oracle"
    assert infer_intervention_class("chunk_zscore_calibration") == "calibration"
    assert infer_intervention_class("new_methods") == "method_pool"


def test_val4_main_appends_ledger_rows(tmp_path: Path, monkeypatch) -> None:
    from tabnetics.validation.campaigns import analyze_val4

    root = tmp_path / "root"
    root.mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text('{"jobs":[]}\n', encoding="utf-8")
    json_out = tmp_path / "analysis.json"
    md_out = tmp_path / "analysis.md"
    ledger_out = tmp_path / "promotion_attempts.jsonl"

    monkeypatch.setattr(
        analyze_val4,
        "analyze_val4",
        lambda **_kwargs: _sample_val4_report(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_val4",
            "--root-out-dir",
            str(root),
            "--plan",
            str(plan),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
            "--ledger-out",
            str(ledger_out),
            "--ledger-evidence-link",
            "https://github.com/klokedm/tabnetics/issues/206#issuecomment-1",
        ],
    )

    analyze_val4.main()

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    rows = load_rows(ledger_out)
    assert payload["promotion_ledger"]["row_count"] == 2
    assert payload["promotion_ledger"]["history_authority"].startswith("GitHub issues")
    assert len(rows) == 2
    assert rows[0]["source_artifacts"]["json_report"] == str(json_out)
    assert rows[0]["evidence_links"] == ["https://github.com/klokedm/tabnetics/issues/206#issuecomment-1"]


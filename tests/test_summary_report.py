"""Tests for T-010: compact per-run summary report.

Validates:
- FeatureSelectionResult.to_summary_dict() schema and field types
- _build_run_summary() schema and field types
- schema_version presence
- Determinism (same config/seed → same summary, modulo timestamps)
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fs_result(**overrides: Any):
    """Create a minimal FeatureSelectionResult for testing."""
    from tabnetics.feature_selection.result import FeatureSelectionResult

    defaults: Dict[str, Any] = {
        "selected_feature_indices": np.array([0, 3, 7]),
        "selected_feature_votes": {0: 0.9, 3: 0.8, 7: 0.6},
        "all_features_info": {
            0: {"name": "f0"},
            3: {"name": "f3"},
            7: {"name": "f7"},
        },
        "method_results": {
            "mi_filter": {"score": 0.85, "oracle_weight": 0.4},
            "lasso_embedded": {"score": 0.75, "oracle_weight": 0.35},
            "svm_rfe": {"score": 0.70},
        },
        "eliminated_features": {
            "low_variance": [1, 2],
            "correlation": [4, 5, 6],
        },
        "config": {
            "method_set": "mnpo_joint_multiclass_extended",
            "selection_strategy": "mnpo",
        },
    }
    defaults.update(overrides)
    return FeatureSelectionResult(**defaults)


def _make_run_rows():
    """Create minimal benchmark run rows for testing _build_run_summary."""
    return [
        {
            "dataset_id": "leukemia_golub",
            "seed": 11,
            "config": "baseline",
            "protocol": "holdout",
            "balanced_accuracy": 0.95,
            "macro_f1": 0.94,
            "hybrid_score": 0.945,
            "roc_auc": 0.972,
            "roc_curve_type": "binary",
            "roc_auc_source": "predict_proba",
            "roc_curve_points": [[0.0, 0.0], [0.1, 0.9], [1.0, 1.0]],
            "roc_curves_by_method": {
                "svm_rbf": {
                    "roc_auc": 0.972,
                    "roc_curve_type": "binary",
                    "roc_auc_source": "predict_proba",
                    "roc_curve_points": [[0.0, 0.0], [0.1, 0.9], [1.0, 1.0]],
                }
            },
            "selected_features": 15,
            "model": "svm_rbf",
            "tier": "easy",
            "fs_time_sec": 1.2,
            "dist_time_sec": 0.5,
            "transform_time_sec": 0.3,
            "n_dist_features_fitted": 100,
            "n_dist_features_transformed": 80,
            "n_dist_rejected": 5,
            "n_dist_skipped_unreliable": 3,
            "n_dist_skipped_block_cv": 2,
            "n_low_gof_downweighted": 1,
            "enabled_methods_source": "config",
        },
        {
            "dataset_id": "leukemia_golub",
            "seed": 23,
            "config": "baseline",
            "protocol": "holdout",
            "balanced_accuracy": 0.93,
            "macro_f1": 0.92,
            "hybrid_score": 0.925,
            "roc_auc": 0.948,
            "roc_curve_type": "binary",
            "roc_auc_source": "predict_proba",
            "roc_curve_points": [[0.0, 0.0], [0.2, 0.85], [1.0, 1.0]],
            "roc_curves_by_method": {
                "svm_rbf": {
                    "roc_auc": 0.948,
                    "roc_curve_type": "binary",
                    "roc_auc_source": "predict_proba",
                    "roc_curve_points": [[0.0, 0.0], [0.2, 0.85], [1.0, 1.0]],
                }
            },
            "selected_features": 14,
            "model": "svm_rbf",
            "tier": "easy",
            "fs_time_sec": 1.1,
            "dist_time_sec": 0.6,
            "transform_time_sec": 0.2,
            "n_dist_features_fitted": 100,
            "n_dist_features_transformed": 78,
            "n_dist_rejected": 6,
            "n_dist_skipped_unreliable": 4,
            "n_dist_skipped_block_cv": 1,
            "n_low_gof_downweighted": 2,
            "enabled_methods_source": "config",
        },
    ]


def _make_metadata():
    """Create minimal metadata dict for testing _build_run_summary."""
    return {
        "datasets": ["leukemia_golub"],
        "seeds": [11, 23],
        "fs_method_set": "mnpo_joint_multiclass_extended",
        "compute_budget": "standard",
        "dist_criterion": "simple",
        "config_flags": {
            "test_size": 0.20,
            "fs_portfolio_size": 5,
        },
        "candidate_rerun_counts": {"candidate_a": 3, "candidate_b": 1},
        "shadow_evaluator": {
            "enabled": True,
            "frozen_subset_id": "pilot_v1",
            "concordance_rate": 0.75,
            "n_compared": 8,
            "disagreement_count": 2,
        },
    }


# ---------------------------------------------------------------------------
# FeatureSelectionResult.to_summary_dict tests
# ---------------------------------------------------------------------------


class TestFeatureSelectionResultSummary:
    """Tests for FeatureSelectionResult.to_summary_dict()."""

    def test_schema_version_present(self):
        fs = _make_fs_result()
        summary = fs.to_summary_dict()
        assert "schema_version" in summary
        assert summary["schema_version"] == "1.0"

    def test_required_fields_present(self):
        fs = _make_fs_result()
        summary = fs.to_summary_dict()
        required = {
            "schema_version",
            "fs_method_preset",
            "selection_strategy",
            "portfolio_candidates",
            "oracle_weights",
            "oracle_stability",
            "copula_low_information",
            "n_features_selected",
            "selected_features",
            "n_methods_run",
            "eliminated_features_counts",
        }
        assert required.issubset(set(summary.keys())), (
            f"Missing fields: {required - set(summary.keys())}"
        )

    def test_field_types(self):
        fs = _make_fs_result()
        summary = fs.to_summary_dict()
        assert isinstance(summary["schema_version"], str)
        assert isinstance(summary["fs_method_preset"], str)
        assert isinstance(summary["selection_strategy"], str)
        assert isinstance(summary["portfolio_candidates"], list)
        assert isinstance(summary["oracle_weights"], dict)
        assert isinstance(summary["oracle_stability"], dict)
        assert isinstance(summary["copula_low_information"], dict)
        assert isinstance(summary["n_features_selected"], int)
        assert isinstance(summary["selected_features"], list)
        assert isinstance(summary["n_methods_run"], int)
        assert isinstance(summary["eliminated_features_counts"], dict)

    def test_portfolio_candidates_sorted(self):
        fs = _make_fs_result()
        summary = fs.to_summary_dict()
        assert summary["portfolio_candidates"] == sorted(summary["portfolio_candidates"])

    def test_oracle_weights_extracted(self):
        fs = _make_fs_result()
        summary = fs.to_summary_dict()
        # mi_filter and lasso_embedded have oracle_weight; svm_rfe does not
        assert "mi_filter" in summary["oracle_weights"]
        assert "lasso_embedded" in summary["oracle_weights"]
        assert "svm_rfe" not in summary["oracle_weights"]
        assert summary["oracle_weights"]["mi_filter"] == 0.4
        assert summary["oracle_weights"]["lasso_embedded"] == 0.35

    def test_extracts_oracle_stability_and_copula_low_info(self):
        fs = _make_fs_result(
            method_results={
                "mnpo_portfolio": {
                    "oracle_pairwise_meta": {
                        "oracle_stability": {
                            "mean_rank_correlation": 0.75,
                            "min_rank_correlation": 0.20,
                        }
                    }
                },
                "copula_knockoff": {
                    "copula_low_information": {
                        "reason_counts": {"ok": 1},
                        "fallback_used": False,
                    }
                },
            }
        )
        summary = fs.to_summary_dict()
        assert summary["oracle_stability"]["mean_rank_correlation"] == pytest.approx(0.75)
        assert summary["copula_low_information"]["reason_counts"]["ok"] == 1

    def test_selected_features_sorted(self):
        fs = _make_fs_result()
        summary = fs.to_summary_dict()
        assert summary["selected_features"] == [0, 3, 7]
        assert summary["n_features_selected"] == 3

    def test_eliminated_features_counts(self):
        fs = _make_fs_result()
        summary = fs.to_summary_dict()
        assert summary["eliminated_features_counts"]["low_variance"] == 2
        assert summary["eliminated_features_counts"]["correlation"] == 3

    def test_fs_method_preset_from_config(self):
        fs = _make_fs_result()
        summary = fs.to_summary_dict()
        assert summary["fs_method_preset"] == "mnpo_joint_multiclass_extended"

    def test_empty_method_results(self):
        fs = _make_fs_result(method_results={})
        summary = fs.to_summary_dict()
        assert summary["portfolio_candidates"] == []
        assert summary["oracle_weights"] == {}
        assert summary["n_methods_run"] == 0

    def test_none_selected_features(self):
        fs = _make_fs_result(selected_feature_indices=None)
        summary = fs.to_summary_dict()
        assert summary["selected_features"] == []
        assert summary["n_features_selected"] == 0

    def test_json_serialisable(self):
        fs = _make_fs_result()
        summary = fs.to_summary_dict()
        # Must round-trip through JSON without error
        json_str = json.dumps(summary, default=str)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0"

    def test_determinism_same_config(self):
        """Same config/seed → same summary (excluding timestamp if any)."""
        fs1 = _make_fs_result()
        fs2 = _make_fs_result()
        s1 = fs1.to_summary_dict()
        s2 = fs2.to_summary_dict()
        assert s1 == s2


# ---------------------------------------------------------------------------
# _build_run_summary tests
# ---------------------------------------------------------------------------


class TestBuildRunSummary:
    """Tests for _build_run_summary() from run_df_fs_sota_benchmark."""

    @pytest.fixture()
    def summary(self, tmp_path):
        from tabnetics.benchmarks.runner import _build_run_summary

        rows = _make_run_rows()
        failures = [{"dataset_id": "nci60_ross", "seed": 37, "error": "timeout"}]
        metadata = _make_metadata()
        return _build_run_summary(
            rows=rows,
            failures=failures,
            metadata=metadata,
            run_dir=tmp_path,
        )

    def test_schema_version_present(self, summary):
        assert summary["schema_version"] == "1.0"

    def test_required_top_level_fields(self, summary):
        required = {
            "schema_version",
            "timestamp",
            "run_dir",
            "fs_method_preset",
            "datasets",
            "seeds",
            "compute_budget",
            "dist_criterion",
            "n_total_runs",
            "n_failures",
            "results",
            "config_snapshot",
            "anti_gaming_telemetry",
            "shadow_evaluator",
        }
        assert required.issubset(set(summary.keys())), (
            f"Missing fields: {required - set(summary.keys())}"
        )

    def test_field_types_top_level(self, summary):
        assert isinstance(summary["schema_version"], str)
        assert isinstance(summary["timestamp"], str)
        assert isinstance(summary["fs_method_preset"], str)
        assert isinstance(summary["datasets"], list)
        assert isinstance(summary["seeds"], list)
        assert isinstance(summary["n_total_runs"], int)
        assert isinstance(summary["n_failures"], int)
        assert isinstance(summary["results"], list)
        assert isinstance(summary["config_snapshot"], dict)
        assert isinstance(summary["anti_gaming_telemetry"], dict)
        assert isinstance(summary["shadow_evaluator"], dict)

    def test_result_count(self, summary):
        assert summary["n_total_runs"] == 2
        assert summary["n_failures"] == 1
        assert len(summary["results"]) == 2

    def test_result_entry_fields(self, summary):
        entry = summary["results"][0]
        for key in (
            "dataset_id", "seed", "config", "protocol",
            "balanced_accuracy", "macro_f1", "hybrid_score", "roc_auc",
            "roc_curve_type", "roc_auc_source", "roc_curve_points", "roc_curves_by_method",
            "selected_features", "model", "tier",
            "runtime", "df_diagnostics", "enabled_methods_source", "oracle_diagnostics",
        ):
            assert key in entry, f"Missing field: {key}"

    def test_runtime_breakdown(self, summary):
        rt = summary["results"][0]["runtime"]
        assert "fs_time_sec" in rt
        assert "dist_time_sec" in rt
        assert "transform_time_sec" in rt

    def test_df_diagnostics(self, summary):
        diag = summary["results"][0]["df_diagnostics"]
        for key in (
            "n_dist_features_fitted",
            "n_dist_features_transformed",
            "n_dist_rejected",
            "n_dist_skipped_unreliable",
            "n_dist_skipped_block_cv",
            "n_low_gof_downweighted",
        ):
            assert key in diag, f"Missing DF diagnostic field: {key}"

    def test_json_serialisable(self, summary):
        json_str = json.dumps(summary, default=str)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0"

    def test_determinism_modulo_timestamp(self, tmp_path):
        from tabnetics.benchmarks.runner import _build_run_summary

        rows = _make_run_rows()
        failures = []
        metadata = _make_metadata()
        s1 = _build_run_summary(rows=rows, failures=failures, metadata=metadata, run_dir=tmp_path)
        s2 = _build_run_summary(rows=rows, failures=failures, metadata=metadata, run_dir=tmp_path)
        # Exclude timestamp for determinism comparison
        s1c = {k: v for k, v in s1.items() if k != "timestamp"}
        s2c = {k: v for k, v in s2.items() if k != "timestamp"}
        # NaN-aware comparison: NaN != NaN breaks plain dict equality,
        # so serialise via json (which converts NaN to null) for a stable check.
        import json, math

        def _nan_safe_json(obj):
            """Serialise to JSON string, replacing NaN/Inf with None for stable comparison."""
            return json.dumps(obj, sort_keys=True, default=str,
                              allow_nan=False,
                              indent=None)

        def _replace_nan(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: _replace_nan(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_replace_nan(v) for v in obj]
            return obj

        assert _replace_nan(s1c) == _replace_nan(s2c)

    def test_fs_method_preset(self, summary):
        assert summary["fs_method_preset"] == "mnpo_joint_multiclass_extended"

"""S1: probe/validation split-disjointness leakage guard.

The decisive S1 calibration-vs-representation replay must test ONLY on rows the DIAKRINO
probe held out (its query split); otherwise the feature logits saw the test rows and the
verdict is leaked.  This pins the guard: it raises on a leaky split, accepts a probe-aligned
one, honours the explicit opt-out, and ignores non-probe rankers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

nnfs = pytest.importorskip("run_nn_fs_tabnetics_validation")


def _make_probe_ranker(tmp_path, query_rows=None):
    feat_dir = tmp_path / "feature_logits"
    feat_dir.mkdir(parents=True, exist_ok=True)
    if query_rows is not None:
        qdir = tmp_path / "query_class_logits"
        qdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"query_row_index": list(query_rows), "query_split": ["query"] * len(query_rows)}
        ).to_parquet(qdir / "ds.parquet")
    return nnfs.ProbeLogitsRanker(logits_root=feat_dir, score_column="prior_logit")


def test_non_probe_ranker_is_noop():
    # None / non-ProbeLogitsRanker => guard does nothing
    nnfs._assert_probe_split_leakage_safe(
        ranker=None, split_policy_used="pipeline_seed",
        test_idx=np.arange(10), dataset_id="ds", allow_leaky=False,
    )


def test_probe_normalize_default_is_chunk_zscore():
    args = nnfs.parse_args([])
    assert args.probe_normalize == "chunk_zscore"

    explicit_legacy = nnfs.parse_args(["--probe-normalize", "none"])
    assert explicit_legacy.probe_normalize == "none"


def test_probe_calibration_ablation_modes_are_selectable():
    expected = {
        "none",
        "chunk_zscore",
        "chunk_rank01",
        "chunk_ecdf",
        "chunk_minmax",
        "chunk_robust_iqr",
        "chunk_softmax_temp",
        "blend",
    }
    assert expected.issubset(set(nnfs.ProbeLogitsRanker.NORMALIZE_MODES))
    for mode in expected:
        args = nnfs.parse_args(["--probe-normalize", mode])
        assert args.probe_normalize == mode


def test_probe_extra_normalizers_are_finite_and_chunk_local(tmp_path):
    scores = np.asarray([0.0, 10.0, 100.0, 110.0, np.nan], dtype=float)
    chunks = np.asarray([0, 0, 1, 1, -1], dtype=np.int64)
    valid = np.isfinite(scores)

    ranker = nnfs.ProbeLogitsRanker(logits_root=tmp_path, score_column="prior_logit", normalize="chunk_minmax")
    minmax = ranker._normalized_probe(scores, chunks, valid)
    assert np.allclose(minmax[:4], [0.0, 1.0, 0.0, 1.0])

    for mode in ("chunk_ecdf", "chunk_robust_iqr", "chunk_softmax_temp"):
        ranker_mode = nnfs.ProbeLogitsRanker(logits_root=tmp_path, score_column="prior_logit", normalize=mode)
        out = ranker_mode._normalized_probe(scores, chunks, valid)
        assert out.shape == scores.shape
        assert np.all(np.isfinite(out))
        assert np.all((out >= 0.0) & (out <= 1.0))
        assert out[1] > out[0]
        assert out[3] > out[2]


def test_probe_chunk_zscore_drift_summary_quantifies_calibration():
    scores = np.asarray([0.0, 2.0, 100.0, 102.0], dtype=float)
    chunks = np.asarray([0, 0, 1, 1], dtype=np.int64)
    summary = nnfs._probe_chunk_zscore_drift_summary(scores, chunks, np.isfinite(scores))

    assert summary["n_chunks"] == 2
    assert summary["n_valid_scores"] == 4
    assert summary["raw_chunk_mean_range"] == pytest.approx(100.0)
    assert summary["chunk_zscore_chunk_mean_range"] == pytest.approx(0.0, abs=1e-6)
    assert summary["chunk_zscore_drift_shrink_ratio"] == pytest.approx(0.0, abs=1e-6)
    assert summary["passes_chunk_zscore_shrink_check"] is True
    assert summary["chunk_logit_mean_drift_count"] == 2
    assert summary["chunk_logit_mean_drift"][0]["chunk_id"] == 0
    assert summary["chunk_logit_mean_drift"][0]["raw_logit_mean"] == pytest.approx(1.0)
    assert summary["chunk_logit_mean_drift"][0]["raw_mean_offset_from_global"] == pytest.approx(-50.0)
    assert summary["chunk_logit_mean_drift"][0]["chunk_zscore_logit_mean"] == pytest.approx(0.0, abs=1e-6)


def test_probe_logits_audit_outputs_calibration_columns(tmp_path):
    pytest.importorskip("pyarrow")
    feat_dir = tmp_path / "feature_logits"
    feat_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "feature_index": [0, 1, 2, 3],
            "chunk_id": [0, 0, 1, 1],
            "prior_logit": [0.0, 2.0, 100.0, 102.0],
        }
    ).to_parquet(feat_dir / "ds.parquet")
    ranker = nnfs.ProbeLogitsRanker(logits_root=feat_dir, score_column="prior_logit", normalize="chunk_zscore")
    X = np.zeros((6, 4), dtype=float)
    y = np.asarray([0, 1, 0, 1, 0, 1])
    result = ranker.rank(X, y, seed=11, dataset_id="ds")
    columns = nnfs._nn_probe_calibration_columns(result)

    assert result.panel_audit.calibration_summary["normalization_mode"] == "chunk_zscore"
    assert result.panel_audit.calibration_summary["zscore_applied"] is True
    assert columns["normalization_mode"] == "chunk_zscore"
    assert columns["normalization_family"] == "chunk_zscore"
    assert columns["calibration_mode"] == "within_chunk_mean_std_then_global_rank01"
    assert columns["zscore_applied"] is True
    assert columns["nn_probe_score_column"] == "prior_logit"
    assert columns["nn_probe_normalize"] == "chunk_zscore"
    assert columns["nn_probe_normalization_family"] == "chunk_zscore"
    assert columns["nn_probe_calibration"] == "within_chunk_mean_std_then_global_rank01"
    assert columns["nn_probe_zscore_applied"] is True
    assert columns["nn_probe_n_chunks"] == 2
    assert columns["nn_probe_raw_chunk_mean_range"] == pytest.approx(100.0)
    assert columns["nn_probe_chunk_zscore_chunk_mean_range"] == pytest.approx(0.0, abs=1e-6)
    assert columns["nn_probe_chunk_zscore_shrink_pass"] is True
    assert columns["nn_probe_chunk_logit_mean_drift_count"] == 2
    drift_rows = json.loads(columns["nn_probe_chunk_logit_mean_drift_json"])
    assert drift_rows[1]["raw_logit_mean"] == pytest.approx(101.0)
    assert drift_rows[1]["chunk_zscore_logit_mean"] == pytest.approx(0.0, abs=1e-6)


def test_diakrino_prior_calibrate_parser_accepts_sidecar_output_modes():
    for mode in (
        "none",
        "chunk_zscore",
        "chunk_rank01",
        "chunk_ecdf",
        "chunk_minmax",
        "chunk_robust_iqr",
        "chunk_softmax_temp",
        "blend",
    ):
        args = nnfs.parse_args(["--diakrino-prior-calibrate", mode])
        assert args.diakrino_prior_calibrate == mode


def test_probe_logits_fuses_prior_and_screening_columns(tmp_path):
    pytest.importorskip("pyarrow")
    feat_dir = tmp_path / "feature_logits"
    feat_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "feature_index": [0, 1, 2, 3],
            "chunk_id": [0, 0, 1, 1],
            "prior_logit": [1.0, 0.0, 0.0, -1.0],
            "screening_logit": [0.0, 1.0, 0.0, -1.0],
        }
    ).to_parquet(feat_dir / "ds.parquet")
    ranker = nnfs.ProbeLogitsRanker(
        logits_root=feat_dir,
        score_column="fuse:prior+screening",
        normalize="none",
        fusion_prior_weight=0.75,
    )
    X = np.zeros((6, 4), dtype=float)
    y = np.asarray([0, 1, 0, 1, 0, 1])
    result = ranker.rank(X, y, seed=11, dataset_id="ds")

    assert int(result.feature_order[0]) == 0
    assert result.scores[0] > result.scores[1] > result.scores[2] > result.scores[3]
    assert "fuse:prior+screening" in result.panel_audit.strategy
    assert "fusion_prior=0.750" in result.panel_audit.strategy


def test_provenance_disjoint_split_passes(tmp_path):
    ranker = _make_probe_ranker(tmp_path, query_rows=[0, 1, 2, 3, 4])
    nnfs._assert_probe_split_leakage_safe(
        ranker=ranker, split_policy_used="pipeline_seed",
        test_idx=np.array([1, 2, 3]), dataset_id="ds", allow_leaky=False,
    )  # test rows ⊆ probe query rows -> safe


def test_provenance_overlapping_support_rows_raises(tmp_path):
    ranker = _make_probe_ranker(tmp_path, query_rows=[0, 1, 2, 3, 4])
    with pytest.raises(RuntimeError, match="PROBE SPLIT LEAKAGE"):
        nnfs._assert_probe_split_leakage_safe(
            ranker=ranker, split_policy_used="pipeline_seed",
            test_idx=np.array([3, 4, 7]), dataset_id="ds", allow_leaky=False,  # 7 was a support row
        )


def test_allow_leaky_override_suppresses_raise(tmp_path):
    ranker = _make_probe_ranker(tmp_path, query_rows=[0, 1, 2, 3, 4])
    nnfs._assert_probe_split_leakage_safe(
        ranker=ranker, split_policy_used="pipeline_seed",
        test_idx=np.array([3, 4, 7]), dataset_id="ds", allow_leaky=True,  # explicit opt-out
    )


def test_no_provenance_requires_hf_probe_policy(tmp_path):
    ranker = _make_probe_ranker(tmp_path, query_rows=None)  # no query_class_logits sibling
    with pytest.raises(RuntimeError, match="requires"):
        nnfs._assert_probe_split_leakage_safe(
            ranker=ranker, split_policy_used="pipeline_seed",
            test_idx=np.array([1, 2, 3]), dataset_id="ds", allow_leaky=False,
        )
    # probe-aligned policy is accepted as the disjointness guarantee
    nnfs._assert_probe_split_leakage_safe(
        ranker=ranker, split_policy_used="hf_probe:12345",
        test_idx=np.array([1, 2, 3]), dataset_id="ds", allow_leaky=False,
    )

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tabnetics.feature_selection import FeatureSelector
from tabnetics.feature_selection.diakrino_conformal_selection import (
    bh_reject,
    conformal_pvalues,
    load_sidecar_conformal_selection,
    select_with_conformal_fdp,
)
from tabnetics.feature_selection.diakrino_qualification import DEFAULT_REQUIRED_GATES, SCHEMA_VERSION
from tabnetics.feature_selection.diakrino_sidecar import N_CANONICAL_FAMILIES
from tabnetics.feature_selection.registry import METHOD_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = REPO_ROOT / "bsc-run/phase0_probe/conformal_selection.py"


def _reference_module():
    spec = importlib.util.spec_from_file_location("phase0_conformal_selection", REFERENCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(*, overall_pass: bool = True, failed_gate: str = "") -> dict:
    gates = []
    for name in DEFAULT_REQUIRED_GATES:
        passed = bool(overall_pass and name != failed_gate)
        gates.append(
            {
                "name": name,
                "required": True,
                "status": "pass" if passed else "fail",
                "pass": passed,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "overall_pass": bool(overall_pass and not failed_gate),
        "gates": gates,
    }


def _write_record(path: Path, record: dict) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _family_logits(top_id: int) -> np.ndarray:
    out = np.full(N_CANONICAL_FAMILIES, -8.0, dtype=np.float64)
    out[int(top_id)] = 8.0
    return out


def test_bh_and_conformal_pvalues_match_phase0_probe_reference() -> None:
    reference = _reference_module()
    scores = np.array([0.25, 2.1, 1.5, -0.5, 3.2, 0.9], dtype=np.float64)
    null_scores = np.array([0.1, 0.4, 1.0, -0.2], dtype=np.float64)
    q = 0.40

    ours = conformal_pvalues(scores, null_scores)
    theirs = reference.conformal_pvalues(scores, null_scores)

    assert np.allclose(ours, theirs)
    assert np.array_equal(bh_reject(ours, q), reference.bh_reject(theirs, q))


def test_conformal_selection_controls_empirical_fdp_on_synthetic_scores() -> None:
    rng = np.random.default_rng(0)
    q = 0.20
    false_discoveries = 0
    selected_total = 0

    for _ in range(200):
        n_link = int(rng.integers(8, 24))
        n_null = int(rng.integers(40, 90))
        scores = np.r_[rng.normal(2.8, 1.0, n_link), rng.normal(0.0, 1.0, n_null)]
        linked = np.r_[np.ones(n_link, dtype=bool), np.zeros(n_null, dtype=bool)]
        result = select_with_conformal_fdp(scores, scores[~linked], target_fdp=q)
        false_discoveries += int(np.count_nonzero(result.selected_mask & ~linked))
        selected_total += int(np.count_nonzero(result.selected_mask))

    empirical_fdp = false_discoveries / max(1, selected_total)
    assert selected_total > 0
    assert empirical_fdp <= q + 0.05


def test_diakrino_conformal_method_is_hard_opt_in_default_disabled() -> None:
    spec = METHOD_REGISTRY["diakrino_conformal_selection"]
    assert spec.default_enabled is False
    assert spec.requires_gpu is False

    selector = FeatureSelector(enabled_methods={"diakrino_conformal_selection"})
    results, scores = selector._diakrino_conformal_selection(np.zeros((4, 3)), np.array([0, 1, 0, 1]), 2)

    assert results == {}
    assert scores == {}
    assert selector._diakrino_conformal_selection_meta["reason"] == "disabled"
    assert selector._diakrino_conformal_selection_meta["normalization_mode"] == "chunk_zscore"
    assert selector._diakrino_conformal_selection_meta["calibration_mode"] == "within_chunk_mean_std_then_split_conformal_bh"
    assert selector._diakrino_conformal_selection_meta["zscore_applied"] is True
    assert selector._diakrino_conformal_selection_meta["nn_probe_normalize"] == "chunk_zscore"
    assert selector._diakrino_conformal_selection_meta["nn_probe_zscore_applied"] is True


def test_sidecar_conformal_selection_fails_closed_without_qualification_or_column(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    sidecar_path = tmp_path / "sidecar.parquet"
    pd.DataFrame(
        {
            "feature_index": [0, 1, 2],
            "chunk_id": [0, 0, 0],
            "prior_logit": [1.0, 2.0, 3.0],
        }
    ).to_parquet(sidecar_path, index=False)

    missing_record = load_sidecar_conformal_selection(
        sidecar_path=str(sidecar_path),
        n_columns=3,
        qualification_record=str(tmp_path / "missing.json"),
        enabled=True,
    )
    assert missing_record.selected_indices.size == 0
    assert missing_record.diagnostics["reason"] == "record_unreadable"

    record_path = _write_record(tmp_path / "pass.json", _record())
    missing_column = load_sidecar_conformal_selection(
        sidecar_path=str(sidecar_path),
        n_columns=3,
        qualification_record=str(record_path),
        enabled=True,
    )
    assert missing_column.selected_indices.size == 0
    assert missing_column.diagnostics["reason"] == "missing_conformal_score"


def test_sidecar_conformal_selection_uses_chunk_zscore_output_diagnostics(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    sidecar_path = tmp_path / "sidecar.parquet"
    pd.DataFrame(
        {
            "feature_index": [0, 1, 2, 3, 4, 5],
            "chunk_id": [0, 0, 0, 0, 0, 0],
            "conformal_score": [-2.0, -1.0, 0.0, 1.0, 4.0, 5.0],
        }
    ).to_parquet(sidecar_path, index=False)
    record_path = _write_record(tmp_path / "pass.json", _record())

    result = load_sidecar_conformal_selection(
        sidecar_path=str(sidecar_path),
        n_columns=6,
        qualification_record=str(record_path),
        enabled=True,
        target_fdp=0.50,
        null_fraction=0.50,
        min_null_scores=3,
    )

    assert result.selected_indices.tolist() == [3, 4, 5]
    assert result.diagnostics["applied"] is True
    assert result.diagnostics["normalize"] == "chunk_zscore"
    assert result.diagnostics["normalization_mode"] == "chunk_zscore"
    assert result.diagnostics["normalization_family"] == "chunk_zscore"
    assert result.diagnostics["calibration"] == "within_chunk_mean_std_then_split_conformal_bh"
    assert result.diagnostics["calibration_mode"] == "within_chunk_mean_std_then_split_conformal_bh"
    assert result.diagnostics["zscore_applied"] is True
    assert result.diagnostics["nn_probe_normalize"] == "chunk_zscore"
    assert result.diagnostics["nn_probe_normalization_family"] == "chunk_zscore"
    assert result.diagnostics["nn_probe_zscore_applied"] is True


def test_feature_selector_conformal_selection_applies_discrete_family_skip(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    sidecar_path = tmp_path / "sidecar.parquet"
    pd.DataFrame(
        {
            "feature_index": [0, 1, 2, 3, 4, 5],
            "chunk_id": [0, 0, 0, 0, 0, 0],
            "population_family_logits": [
                _family_logits(0),
                _family_logits(0),
                _family_logits(0),
                _family_logits(0),
                _family_logits(0),
                _family_logits(31),
            ],
            "conformal_score": [-2.0, -1.0, 0.0, 4.0, 5.0, 8.0],
        }
    ).to_parquet(sidecar_path, index=False)
    record_path = _write_record(tmp_path / "pass.json", _record())
    selector = FeatureSelector(
        enabled_methods={"diakrino_conformal_selection"},
        diakrino_prior_sidecar_path=str(sidecar_path),
        diakrino_conformal_selection_enabled=True,
        diakrino_conformal_qualification_record=str(record_path),
        diakrino_conformal_target_fdp=0.90,
        diakrino_conformal_null_fraction=0.50,
        diakrino_conformal_min_null_scores=3,
    )

    results, all_scores = selector._diakrino_conformal_selection(
        np.zeros((6, 6)),
        np.array([0, 1, 0, 1, 0, 1]),
        2,
    )

    assert 5 not in all_scores
    assert 5 not in results.get("selected_indices", [])
    assert results["selected_indices"].tolist() == [3, 4]
    assert selector._diakrino_conformal_selection_meta["normalize"] == "chunk_zscore"

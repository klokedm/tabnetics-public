"""S3: diakrino_prior / diakrino_screening_prior MNPO candidate methods.

Pins: registry entries are opt-in (default_enabled=False), the method gracefully skips
without a sidecar, selects by the calibrated sidecar score when present, and correctly
gathers original-indexed scores onto the X_uncorr column space via feature_mapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabnetics.feature_selection.registry import (
    METHOD_REGISTRY,
    method_excluded_by_default,
)
from tabnetics.feature_selection.methods.diakrino_prior import diakrino_prior_selection


def _write_sidecar(tmp_path, prior_logits, chunk_ids=None):
    n = len(prior_logits)
    df = pd.DataFrame({
        "feature_index": np.arange(n, dtype=int),
        "chunk_id": np.zeros(n, dtype=int) if chunk_ids is None else np.asarray(chunk_ids),
        "prior_logit": np.asarray(prior_logits, dtype=float),
        "screening_logit": np.asarray(prior_logits, dtype=float)[::-1],
    })
    p = tmp_path / "ds.parquet"
    df.to_parquet(p)
    return str(p)


def test_registry_entries_are_opt_in():
    for key in ("diakrino_prior", "diakrino_screening_prior"):
        spec = METHOD_REGISTRY[key]
        assert spec.default_enabled is False
        assert spec.requires_gpu is False
        assert spec.maturity == "experimental"
        # excluded under run-all; included only when explicitly named
        assert method_excluded_by_default(spec, None) is True
        assert method_excluded_by_default(spec, {key}) is False


def test_skips_without_sidecar():
    X = np.random.default_rng(0).normal(size=(20, 6))
    y = np.array([0, 1] * 10)
    res, allsc = diakrino_prior_selection(
        X, y, 2, sidecar_path="", score_column="prior_logit",
        calibrate="none", feature_mapping=np.arange(6), top_k=0,
    )
    assert res == {} and allsc == {}


def test_selects_top_by_sidecar_score(tmp_path):
    path = _write_sidecar(tmp_path, prior_logits=[1, 2, 3, 4, 5, 6])
    X = np.zeros((20, 6))
    y = np.array([0, 1] * 10)
    res, allsc = diakrino_prior_selection(
        X, y, 2, sidecar_path=path, score_column="prior_logit",
        calibrate="none", feature_mapping=np.arange(6), top_k=0,
    )
    assert set(int(i) for i in res["selected_indices"]) == {4, 5}  # highest two scores
    assert len(allsc) == 6


def test_selects_from_sidecar_root_by_dataset_id(tmp_path):
    feature_dir = tmp_path / "feature_logits"
    feature_dir.mkdir()
    pd.DataFrame(
        {
            "dataset_id": ["alpha"] * 3,
            "feature_index": [0, 1, 2],
            "chunk_id": [0, 0, 0],
            "prior_logit": [100.0, 1.0, 0.0],
        }
    ).to_parquet(feature_dir / "alpha.parquet", index=False)
    pd.DataFrame(
        {
            "dataset_id": ["beta"] * 3,
            "feature_index": [0, 1, 2],
            "chunk_id": [0, 0, 0],
            "prior_logit": [0.0, 1.0, 100.0],
        }
    ).to_parquet(feature_dir / "beta.parquet", index=False)

    X = np.zeros((10, 3))
    y = np.array([0, 1] * 5)
    res, _ = diakrino_prior_selection(
        X, y, 1, sidecar_path=str(tmp_path), dataset_id="beta",
        score_column="prior_logit", calibrate="none", feature_mapping=np.arange(3),
    )

    assert list(res["selected_indices"]) == [2]


def test_feature_mapping_gathers_original_scores(tmp_path):
    # original prior_logits = [1,2,3,4,5,6]; X_uncorr 3 cols map to originals [2,0,5]
    path = _write_sidecar(tmp_path, prior_logits=[1, 2, 3, 4, 5, 6])
    X = np.zeros((10, 3))
    y = np.array([0, 1] * 5)
    res, _ = diakrino_prior_selection(
        X, y, 2, sidecar_path=path, score_column="prior_logit",
        calibrate="none", feature_mapping=np.array([2, 0, 5]), top_k=0,
    )
    # col scores = [3,1,6] -> top-2 cols are 2 (=6) and 0 (=3)
    assert set(int(i) for i in res["selected_indices"]) == {0, 2}


def test_misaligned_mapping_skips(tmp_path):
    path = _write_sidecar(tmp_path, prior_logits=[1, 2, 3])
    X = np.zeros((10, 5))  # 5 cols but mapping references original index 9 (out of range)
    y = np.array([0, 1] * 5)
    res, allsc = diakrino_prior_selection(
        X, y, 2, sidecar_path=path, score_column="prior_logit",
        calibrate="none", feature_mapping=np.array([0, 1, 2, 9, 4]), top_k=0,
    )
    assert res == {} and allsc == {}  # out-of-range original index -> skip, never misalign

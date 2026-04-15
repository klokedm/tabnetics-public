import numpy as np
import pandas as pd
import pytest

from tabnetics.validation.campaigns.analyze_val14 import _compute_contrast_table


def _runs_for_contrast(left: str, right: str, deltas: list[float]) -> pd.DataFrame:
    rows = []
    base = 0.60
    for i, delta in enumerate(deltas, start=1):
        ds = f"ds{i:02d}"
        rows.append(
            {
                "dataset_id": ds,
                "profile": left,
                "balanced_accuracy": float(base + delta),
                "config": "baseline",
            }
        )
        rows.append(
            {
                "dataset_id": ds,
                "profile": right,
                "balanced_accuracy": float(base),
                "config": "baseline",
            }
        )
    return pd.DataFrame(rows)


def test_compute_contrast_table_includes_effect_size_columns():
    df = _runs_for_contrast("v14_ref", "d_default", [0.10, 0.20, 0.30])
    table = _compute_contrast_table(
        df,
        [("v14_ref", "d_default", "ref_vs_prod")],
        apply_bh=True,
    )
    assert {"cohen_d", "rank_biserial_r"}.issubset(set(table.columns))
    row = table.iloc[0]
    assert float(row["n_datasets"]) == 3
    assert float(row["rank_biserial_r"]) == pytest.approx(1.0)
    assert np.isfinite(float(row["cohen_d"]))
    assert float(row["cohen_d"]) > 0.0


def test_compute_contrast_table_rank_biserial_is_bounded():
    df = _runs_for_contrast("v14_ref", "d_default", [0.10, -0.20, 0.30, -0.05, 0.0])
    table = _compute_contrast_table(
        df,
        [("v14_ref", "d_default", "ref_vs_prod")],
        apply_bh=False,
    )
    row = table.iloc[0]
    r_rb = float(row["rank_biserial_r"])
    assert -1.0 <= r_rb <= 1.0

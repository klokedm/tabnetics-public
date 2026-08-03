from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabnetics.feature_selection.diakrino_budget_authority import (
    build_p4_budget_authority,
    read_p4_budget_authority,
    write_p4_budget_authority,
)
from tabnetics.feature_selection.diakrino_identity import DiakrinoSidecarIdentityError


def _p4_row() -> dict[str, object]:
    return {
        "dataset_id": "fixture_ds",
        "seed": 11,
        "diakrino_feature_selection_arm": "protected_native_null_jmi",
        "diakrino_campaign_contract_sha256": "a" * 64,
        "diakrino_identity_binding_sha256": "b" * 64,
        "diakrino_native_nulls_sha256": "c" * 64,
        "diakrino_paired_inference_views_sha256": "d" * 64,
        "diakrino_protected_core_original_indices_json": "[1,3,5]",
        "diakrino_extra_original_indices_json": "[7,9]",
        "diakrino_additions": 2,
    }


def test_p4_budget_authority_binds_exact_task_and_result(tmp_path: Path) -> None:
    row = _p4_row()
    path = write_p4_budget_authority(tmp_path / "p4_authority.json", row)

    payload, digest = read_p4_budget_authority(
        path,
        dataset_id="fixture_ds",
        seed=11,
        campaign_contract_sha256="a" * 64,
        binding_sha256="b" * 64,
        native_nulls_sha256="c" * 64,
        paired_inference_views_sha256="d" * 64,
    )

    assert digest
    assert payload["p4_realized_additions"] == 2
    assert payload["protected_core_indices"] == [1, 3, 5]
    assert payload["p4_result_row_sha256"]


def test_p4_budget_authority_rejects_crosswired_native_null(tmp_path: Path) -> None:
    path = write_p4_budget_authority(tmp_path / "p4_authority.json", _p4_row())

    with pytest.raises(DiakrinoSidecarIdentityError, match="native_nulls_sha256"):
        read_p4_budget_authority(
            path,
            dataset_id="fixture_ds",
            seed=11,
            campaign_contract_sha256="a" * 64,
            binding_sha256="b" * 64,
            native_nulls_sha256="e" * 64,
            paired_inference_views_sha256="d" * 64,
        )


def test_p4_budget_authority_rejects_tampered_protected_core(tmp_path: Path) -> None:
    path = write_p4_budget_authority(tmp_path / "p4_authority.json", _p4_row())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protected_core_indices"] = [1, 3, 6]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DiakrinoSidecarIdentityError, match="protected core digest"):
        read_p4_budget_authority(
            path,
            dataset_id="fixture_ds",
            seed=11,
            campaign_contract_sha256="a" * 64,
            binding_sha256="b" * 64,
            native_nulls_sha256="c" * 64,
            paired_inference_views_sha256="d" * 64,
        )

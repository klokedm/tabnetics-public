from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tabnetics.feature_selection import FeatureSelector
from tabnetics.feature_selection.diakrino_qualification import (
    CONSUMER_CLASS_REQUIRED_GATES,
    DEFAULT_REQUIRED_GATES,
    SCHEMA_VERSION,
    qualification_gate_status,
    record_allows_gated_consumers,
)
from tabnetics.feature_selection.diakrino_trust import default_diakrino_sidecar_trust_record


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


def test_core_qualification_gate_fails_closed_for_missing_or_failed_records(tmp_path: Path) -> None:
    missing = qualification_gate_status(tmp_path / "missing.json")
    assert missing["allowed"] is False
    assert missing["reason"] == "record_unreadable"

    failed_path = _write_record(tmp_path / "failed.json", _record(failed_gate="sidecar_reemit"))
    failed = qualification_gate_status(failed_path)
    assert failed["allowed"] is False
    assert failed["reason"] == "overall_pass_false"
    assert "sidecar_reemit" in failed["failed_gates"]
    assert record_allows_gated_consumers(json.loads(failed_path.read_text(encoding="utf-8"))) is False


def test_core_qualification_gate_allows_only_all_required_passes(tmp_path: Path) -> None:
    record_path = _write_record(tmp_path / "pass.json", _record())
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    status = qualification_gate_status(record_path)

    assert status["allowed"] is True
    assert status["reason"] == "allowed"
    assert status["failed_gates"] == []
    assert record_allows_gated_consumers(payload) is True

    missing_selector_gate = dict(payload)
    missing_selector_gate["gates"] = [
        gate for gate in payload["gates"] if gate["name"] != "selector_weights_candidate_probe"
    ]
    assert record_allows_gated_consumers(missing_selector_gate) is False


def test_diakrino_selector_prior_requires_qualification_record_and_trusted_sidecar(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    sidecar_path = tmp_path / "sidecar.parquet"
    pd.DataFrame(
        {
            "feature_index": [0, 1],
            "selector_weight_boruta": [1.0, 1.0],
        }
    ).to_parquet(sidecar_path, index=False)

    selector = FeatureSelector(use_diakrino_selector_prior=True, diakrino_prior_sidecar_path=str(sidecar_path))
    assert selector._compute_diakrino_selector_weights() is None
    assert selector._diakrino_selector_prior_gate_meta["reason"] == "missing_qualification_record"

    record_path = _write_record(tmp_path / "pass.json", _record())
    qualified = FeatureSelector(
        use_diakrino_selector_prior=True,
        diakrino_prior_sidecar_path=str(sidecar_path),
        diakrino_selector_prior_qualification_record=str(record_path),
    )
    weights = qualified._compute_diakrino_selector_weights()

    assert weights is None
    assert qualified._diakrino_selector_prior_gate_meta["allowed"] is False
    assert qualified._diakrino_selector_prior_gate_meta["reason"] == "missing_sidecar_trust_record_after_qualification"
    assert qualified._diakrino_selector_prior_gate_meta["sidecar_trust_record_present"] is False


def test_diakrino_selector_prior_does_not_consume_report_only_candidate_weights(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    feature_dir = tmp_path / "feature_logits"
    feature_dir.mkdir()
    sidecar_path = feature_dir / "toy.parquet"
    pd.DataFrame(
        {
            "dataset_id": ["toy", "toy"],
            "feature_index": [0, 1],
            "selector_weights_candidate": [
                json.dumps([0.25, 0.45, 0.075, 0.075, 0.15]),
                json.dumps([0.25, 0.45, 0.075, 0.075, 0.15]),
            ],
        }
    ).to_parquet(sidecar_path, index=False)
    trust = default_diakrino_sidecar_trust_record(checkpoint_sha256="abc123")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_sha256": "abc123",
                "diakrino_sidecar_trust_record": trust,
                "sidecars": [{"dataset_id": "toy", "feature_logits_path": "feature_logits/toy.parquet"}],
            }
        ),
        encoding="utf-8",
    )
    record_path = _write_record(tmp_path / "pass.json", _record())

    selector = FeatureSelector(
        use_diakrino_selector_prior=True,
        diakrino_prior_sidecar_path=str(manifest),
        diakrino_prior_dataset_id="toy",
        diakrino_selector_prior_qualification_record=str(record_path),
    )

    assert selector._compute_diakrino_selector_weights() is None
    assert selector._diakrino_selector_prior_gate_meta["allowed"] is False
    assert selector._diakrino_selector_prior_gate_meta["reason"] == "selector_weights_untrusted_after_qualification"
    assert selector._diakrino_selector_prior_gate_meta["sidecar_trust_record_present"] is True


def test_diakrino_selector_prior_passes_sidecar_trust_record_to_mnpo(tmp_path: Path, monkeypatch) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    feature_dir = tmp_path / "feature_logits"
    feature_dir.mkdir()
    pd.DataFrame(
        {
            "dataset_id": ["toy", "toy"],
            "feature_index": [0, 1],
            "selector_weight_boruta": [1.0, 1.0],
        }
    ).to_parquet(feature_dir / "toy.parquet", index=False)
    trust = default_diakrino_sidecar_trust_record(checkpoint_sha256="abc123")
    trust["heads"]["selector_weights"] = {
        "status": "trusted_for_smoke",
        "consumer_allowed": True,
        "evidence": {"verdict": "test_manifest_contract"},
    }
    trust["selector_prior_calibration"]["anchor_weights"] = {
        "mnpo_broad_stable": 0.0,
        "strict_plus_mrmr": 0.0,
        "boruta": 1.0,
        "copula_knockoff": 0.0,
        "stability_lasso": 0.0,
    }
    trust["selector_prior_calibration"]["raw_weight"] = 0.0
    trust["selector_prior_calibration"]["max_blend_weight"] = 0.10
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_sha256": "abc123",
                "diakrino_sidecar_trust_record": trust,
                "sidecars": [{"dataset_id": "toy", "feature_logits_path": "feature_logits/toy.parquet"}],
            }
        ),
        encoding="utf-8",
    )
    record_path = _write_record(tmp_path / "pass.json", _record())
    captured = {}

    def fake_mnpo_select_features(*args, **kwargs):
        captured.update(kwargs)
        return np.array([0], dtype=np.int64)

    import tabnetics.feature_selection.mnpo.portfolio as portfolio

    monkeypatch.setattr(portfolio, "mnpo_select_features", fake_mnpo_select_features)
    selector = FeatureSelector(
        use_diakrino_selector_prior=True,
        diakrino_prior_sidecar_path=str(manifest),
        diakrino_prior_dataset_id="toy",
        diakrino_selector_prior_qualification_record=str(record_path),
    )

    selector._mnpo_select_features(
        np.zeros((4, 2), dtype=np.float32),
        np.array([0, 1, 0, 1], dtype=np.int64),
        1,
        1,
        {},
        {},
    )

    assert captured["diakrino_selector_prior"] == {"boruta": pytest.approx(1.0)}
    trust_record = captured["diakrino_selector_prior_trust_record"]
    assert trust_record["checkpoint_sha256"] == "abc123"
    assert trust_record["selector_prior_calibration"]["max_blend_weight"] == pytest.approx(0.10)
    assert selector._diakrino_selector_prior_gate_meta["sidecar_trust_record_present"] is True


# --- T-DIAKRINO-NAT-12 granular consumer-class gating -----------------------------

FS_GATES = CONSUMER_CLASS_REQUIRED_GATES["feature_selection"]


def _record_with_passing(pass_gates, *, fail_status: str = "fail") -> dict:
    """Record where exactly ``pass_gates`` are measured-pass; the rest ``fail``."""
    passing = {str(name) for name in pass_gates}
    gates = []
    for name in DEFAULT_REQUIRED_GATES:
        ok = name in passing
        gates.append(
            {
                "name": name,
                "required": True,
                "status": "pass" if ok else fail_status,
                "pass": ok,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "overall_pass": all(g["pass"] for g in gates),
        "gates": gates,
    }


def test_feature_selection_class_passes_while_dead_head_classes_stay_closed() -> None:
    # The v-next-C LAWA reality: FS gates pass, but query-ICL / param / structural fail.
    rec = _record_with_passing(FS_GATES)
    assert rec["overall_pass"] is False

    # Strict whole-checkpoint policy (no class) still fails closed.
    assert record_allows_gated_consumers(rec) is False
    # The feature-selection consumer class is enabled.
    assert record_allows_gated_consumers(rec, consumer_class="feature_selection") is True
    # Everything that depends on the dead heads stays fail-closed.
    for cls in ("classifier", "warm_start", "redundancy", "all"):
        assert record_allows_gated_consumers(rec, consumer_class=cls) is False


def test_unknown_consumer_class_fails_closed() -> None:
    rec = _record_with_passing(DEFAULT_REQUIRED_GATES)  # everything passes
    assert record_allows_gated_consumers(rec, consumer_class="feature_selection") is True
    assert record_allows_gated_consumers(rec, consumer_class="does_not_exist") is False


def test_feature_selection_class_fails_closed_on_not_run_fs_gate(tmp_path: Path) -> None:
    rec = _record_with_passing(FS_GATES)
    for gate in rec["gates"]:
        if gate["name"] == "s1_chunk_calibration_replay":
            gate["status"] = "not_run"
            gate["pass"] = False
    assert record_allows_gated_consumers(rec, consumer_class="feature_selection") is False

    path = _write_record(tmp_path / "not_run.json", rec)
    status = qualification_gate_status(path, consumer_class="feature_selection")
    assert status["allowed"] is False
    assert "s1_chunk_calibration_replay" in status["failed_gates"]


def test_qualification_gate_status_feature_selection_class(tmp_path: Path) -> None:
    path = _write_record(tmp_path / "fs.json", _record_with_passing(FS_GATES))

    fs_status = qualification_gate_status(path, consumer_class="feature_selection")
    assert fs_status["allowed"] is True
    assert fs_status["reason"] == "allowed"
    assert fs_status["failed_gates"] == []
    assert fs_status["consumer_class"] == "feature_selection"

    # The strict default remains blocked (overall_pass is false).
    strict = qualification_gate_status(path)
    assert strict["allowed"] is False
    assert strict["reason"] == "overall_pass_false"

    unknown = qualification_gate_status(path, consumer_class="nope")
    assert unknown["allowed"] is False
    assert unknown["reason"] == "unknown_consumer_class"

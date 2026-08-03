from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tabnetics.benchmarks.result_journal import (
    AtomicResultJournal,
    DuplicateResultKeyError,
    ResultJournalContextError,
    deterministic_result_key,
)


KEY_FIELDS = ("dataset_id", "split_id", "method", "metric", "seed")


def _row(*, status: str = "ok", metric_value: float = 0.75) -> dict[str, object]:
    return {
        "dataset_id": "dataset-a",
        "split_id": "0",
        "method": "tabnetics-current",
        "metric": "roc_auc",
        "seed": 42,
        "status": status,
        "metric_value": metric_value,
        "skip_reason": pd.NA,
    }


def test_deterministic_result_key_uses_only_declared_identity_fields() -> None:
    first = _row(status="ok", metric_value=0.75)
    second = {**first, "status": "error", "metric_value": 0.1}

    assert deterministic_result_key(first, key_fields=KEY_FIELDS) == deterministic_result_key(
        second,
        key_fields=KEY_FIELDS,
    )


def test_atomic_result_journal_recovers_committed_rows_and_ignores_uncommitted_temp(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results.journal"
    journal = AtomicResultJournal(root, key_fields=KEY_FIELDS, context={"backend": "current"})
    key = journal.commit(_row())
    interrupted_temp = journal.rows_dir / ".interrupted.json.tmp.123"
    interrupted_temp.write_text('{"partial":', encoding="utf-8")

    resumed = AtomicResultJournal(root, key_fields=KEY_FIELDS, context={"backend": "current"})

    assert resumed.committed_keys == {key}
    assert len(resumed) == 1
    assert resumed.rows() == [
        {
            **_row(),
            "skip_reason": None,
        }
    ]
    assert resumed.to_frame().columns.tolist() == list(_row())


def test_atomic_result_journal_fails_closed_on_duplicate_and_conflicting_keys(
    tmp_path: Path,
) -> None:
    journal = AtomicResultJournal(tmp_path / "results.journal", key_fields=KEY_FIELDS)
    journal.commit(_row())

    with pytest.raises(DuplicateResultKeyError, match="committed row is identical"):
        journal.commit(_row())
    with pytest.raises(DuplicateResultKeyError, match=r"conflicting result key.*metric_value"):
        journal.commit(_row(status="error", metric_value=0.1))

    assert len(journal) == 1
    assert journal.rows()[0]["status"] == "ok"


def test_atomic_result_journal_rejects_incompatible_resume_context(tmp_path: Path) -> None:
    root = tmp_path / "results.journal"
    AtomicResultJournal(root, key_fields=KEY_FIELDS, context={"backend": "current"})

    with pytest.raises(ResultJournalContextError, match="context does not match"):
        AtomicResultJournal(root, key_fields=KEY_FIELDS, context={"backend": "diakrino"})

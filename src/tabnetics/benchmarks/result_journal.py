"""Atomic, deterministic result-row journaling for resumable benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


JOURNAL_FORMAT_VERSION = 1


class ResultJournalError(RuntimeError):
    """Base error for a result journal that cannot be used safely."""


class DuplicateResultKeyError(ResultJournalError):
    """Raised when a caller attempts to commit an existing result key."""


class ResultJournalContextError(ResultJournalError):
    """Raised when an existing journal belongs to a different run context."""


class CorruptResultJournalError(ResultJournalError):
    """Raised when committed journal data fails integrity validation."""


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    raise TypeError(f"result journal values must be JSON scalars or containers; got {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def deterministic_result_key(row: Mapping[str, Any], *, key_fields: Sequence[str]) -> str:
    """Return a stable content key for the selected result identity fields."""

    fields = tuple(str(field) for field in key_fields)
    if not fields:
        raise ValueError("result journal key_fields must not be empty")
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"result row is missing deterministic key fields: {missing}")
    identity = {field: _json_value(row[field]) for field in fields}
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create ``path`` from a fully-fsynced temporary inode without replacing it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorruptResultJournalError(f"cannot read committed journal record {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorruptResultJournalError(f"committed journal record is not an object: {path}")
    return payload


class AtomicResultJournal:
    """Directory journal with one immutable, atomically committed JSON file per row.

    Files left with a ``.tmp.`` name were never committed and are ignored on
    recovery. A committed key is immutable: duplicate and conflicting writes
    both raise instead of silently replacing evidence.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        key_fields: Sequence[str],
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.rows_dir = self.root / "rows"
        self.manifest_path = self.root / "manifest.json"
        self.key_fields = tuple(str(field) for field in key_fields)
        if not self.key_fields:
            raise ValueError("result journal key_fields must not be empty")
        self.context = _json_value(context or {})
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_manifest()
        self._keys = self._load_committed_keys()

    def _expected_manifest(self) -> dict[str, Any]:
        return {
            "format_version": JOURNAL_FORMAT_VERSION,
            "key_fields": list(self.key_fields),
            "context": self.context,
            "context_sha256": hashlib.sha256(_canonical_json(self.context).encode("utf-8")).hexdigest(),
        }

    def _initialize_manifest(self) -> None:
        expected = self._expected_manifest()
        if not self.manifest_path.exists():
            try:
                _atomic_create_json(self.manifest_path, expected)
            except FileExistsError:
                pass
        actual = _load_json(self.manifest_path)
        if actual != expected:
            raise ResultJournalContextError(
                "result journal context does not match this run; "
                f"journal={self.root}, expected={expected}, actual={actual}"
            )

    def _load_record(self, path: Path) -> dict[str, Any]:
        payload = _load_json(path)
        if payload.get("format_version") != JOURNAL_FORMAT_VERSION:
            raise CorruptResultJournalError(f"unsupported committed journal record version: {path}")
        row = payload.get("row")
        if not isinstance(row, dict):
            raise CorruptResultJournalError(f"committed journal record has no row object: {path}")
        row_order = payload.get("row_order")
        if (
            not isinstance(row_order, list)
            or not all(isinstance(field, str) for field in row_order)
            or len(row_order) != len(row)
            or len(set(row_order)) != len(row_order)
            or set(row_order) != set(row)
        ):
            raise CorruptResultJournalError(f"committed journal record has invalid row order: {path}")
        payload["row"] = {field: row[field] for field in row_order}
        row = payload["row"]
        key = deterministic_result_key(row, key_fields=self.key_fields)
        if payload.get("key") != key or path.stem != key:
            raise CorruptResultJournalError(
                f"committed journal key mismatch: path={path}, calculated_key={key}"
            )
        return payload

    def _load_committed_keys(self) -> set[str]:
        keys: set[str] = set()
        for path in sorted(self.rows_dir.glob("*.json")):
            payload = self._load_record(path)
            key = str(payload["key"])
            if key in keys:
                raise CorruptResultJournalError(f"duplicate committed result key in journal: {key}")
            keys.add(key)
        return keys

    def key_for(self, row: Mapping[str, Any]) -> str:
        return deterministic_result_key(row, key_fields=self.key_fields)

    def contains(self, row: Mapping[str, Any]) -> bool:
        key = self.key_for(row)
        with self._lock:
            return key in self._keys

    @property
    def committed_keys(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._keys)

    def commit(self, row: Mapping[str, Any]) -> str:
        normalized = _json_value(dict(row))
        key = deterministic_result_key(normalized, key_fields=self.key_fields)
        path = self.rows_dir / f"{key}.json"
        payload = {
            "format_version": JOURNAL_FORMAT_VERSION,
            "key": key,
            "key_values": {field: normalized[field] for field in self.key_fields},
            "row_order": list(normalized),
            "row": normalized,
        }
        with self._lock:
            if key in self._keys or path.exists():
                self._raise_duplicate(path, key=key, incoming=normalized)
            try:
                _atomic_create_json(path, payload)
            except FileExistsError:
                self._raise_duplicate(path, key=key, incoming=normalized)
            self._keys.add(key)
        return key

    def _raise_duplicate(self, path: Path, *, key: str, incoming: Mapping[str, Any]) -> None:
        existing = self._load_record(path).get("row", {})
        identity = {field: incoming.get(field) for field in self.key_fields}
        if existing == incoming:
            raise DuplicateResultKeyError(
                f"duplicate result key {key} for identity {identity}; committed row is identical"
            )
        differing = sorted(
            field for field in set(existing).union(incoming) if existing.get(field) != incoming.get(field)
        )
        raise DuplicateResultKeyError(
            f"conflicting result key {key} for identity {identity}; differing fields={differing}"
        )

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            payloads = [self._load_record(path) for path in sorted(self.rows_dir.glob("*.json"))]
        payloads.sort(key=lambda payload: _canonical_json(payload["key_values"]))
        return [dict(payload["row"]) for payload in payloads]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.rows())

    def __len__(self) -> int:
        with self._lock:
            return len(self._keys)


__all__ = [
    "AtomicResultJournal",
    "CorruptResultJournalError",
    "DuplicateResultKeyError",
    "JOURNAL_FORMAT_VERSION",
    "ResultJournalContextError",
    "ResultJournalError",
    "deterministic_result_key",
]

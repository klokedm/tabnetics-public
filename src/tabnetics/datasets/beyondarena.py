"""BeyondArena/DataFoundry metadata, split, and preprocessing contracts.

The public BeyondArena release stores each dataset as a DataFoundry-style
artifact directory with JSON metadata plus a ``dataset.parquet`` payload.  This
module keeps the core integration manifest-first: unit tests can exercise the
metadata/split contracts without downloading the full corpus, while callers who
have local or HF-cached artifacts can opt into materializing the parquet data.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CONTAINER_METADATA = "container_metadata.json"
DATASET_METADATA = "dataset_metadata.dataset-mold-v1.json"
TASK_METADATA = "task_metadata.predictive-ml-task-mold-v1.json"
SPLIT_METADATA = "experiment_metadata.predictive-ml-splits-mold-v1.json"
DTYPES_METADATA = "dtypes.json"
DATASET_PARQUET = "dataset.parquet"
TEXT_CACHE_BASENAME = "tabarena_text_cache_Qwen3-Embedding-8B-d32-r1.parquet"
_TEXT_TOKEN_PATTERN = re.compile(r"\b\w+\b")
__tabnetics_execution_isolated_state__ = {
    "_TEXT_TOKEN_PATTERN": {
        "provider": "clean_isolated_reference_v1",
        "dependencies": (),
    },
}

BEYONDARENA_HF_REPO_ID = "TabArena/BeyondArena"
BEYONDARENA_TASK_METADATA_CSV_URL = (
    "https://raw.githubusercontent.com/autogluon/tabarena/main/"
    "packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/"
    "BeyondArena_tasks_metadata.csv"
)
BEYONDARENA_CORE_TASKS_CSV_URL = (
    "https://raw.githubusercontent.com/autogluon/tabarena/main/"
    "packages/tabarena/src/tabarena/contexts/beyondarena/data/"
    "BeyondArena_core_tasks.csv"
)
BEYONDARENA_EXPECTED_ACCEPTED_DATASETS = 142
BEYONDARENA_EXPECTED_TASK_METADATA_ROWS = 3722
BEYONDARENA_EXPECTED_CORE_TASK_ROWS = 507

_CLASSIFICATION_PROBLEM_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "classification",
    "binary",
    "multiclass",
}
_REGRESSION_PROBLEM_TYPES = {"regression", "quantile_regression"}
_LOWER_IS_BETTER_METRICS = {
    "log_loss",
    "mae",
    "mse",
    "rmse",
    "root_mean_squared_error",
}
_HIGHER_IS_BETTER_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "f1",
    "macro_f1",
    "r2",
    "roc_auc",
}


class BeyondArenaUnavailableError(RuntimeError):
    """Raised when an opt-in BeyondArena operation cannot be performed."""


@dataclass(frozen=True)
class BeyondArenaDatasetSpec:
    """Normalized metadata for one BeyondArena/DataFoundry artifact."""

    beyondarena_id: str
    dataset_name: str
    artifact_dir: Optional[Path]
    artifact_revision: Optional[str]
    task_type: str
    target_column: str
    problem_type: str
    raw_problem_type: str
    objective_metric: str
    stratify_on: Optional[str] = None
    time_on: Optional[str] = None
    group_on: Optional[str] = None
    group_labels: Optional[str] = None
    group_time_on: Optional[str] = None
    data_tags: Tuple[str, ...] = ()
    source: str = ""
    license: str = ""
    domain: str = ""
    has_text_cache: bool = False
    has_dataset: bool = False
    has_high_cardinality: bool = False
    high_cardinality_columns: Tuple[str, ...] = ()
    has_text_features: bool = False
    text_columns: Tuple[str, ...] = ()
    is_high_dimensional: bool = False
    n_features: Optional[int] = None
    n_samples: Optional[int] = None
    metadata_paths: Dict[str, str] = field(default_factory=dict)
    skip_reason: Optional[str] = None

    @property
    def is_iid(self) -> bool:
        return self.task_type == "iid"

    @property
    def is_grouped(self) -> bool:
        return self.task_type == "grouped"

    @property
    def is_temporal(self) -> bool:
        return self.task_type == "temporal"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.artifact_dir is not None:
            payload["artifact_dir"] = str(self.artifact_dir)
        return payload


@dataclass(frozen=True)
class BeyondArenaTaskMetadataRow:
    """One committed TabArena BeyondArena task-metadata CSV row.

    The upstream CSV is split-grain: one row per dataset/repeat/fold.  The
    normalized ``split`` field mirrors TabArena's task-grid convention:
    ``split = n_folds * repeat + fold`` per dataset.
    """

    dataset_name: str
    problem_type: str
    is_classification: bool
    target_name: str
    eval_metric: str
    tabarena_task_name: str
    task_id_str: str
    data_foundry_uri: str
    task_type: str
    repeat: int
    fold: int
    split_index: str
    split: int
    stratify_on: Optional[str] = None
    time_on: Optional[str] = None
    group_on: Optional[str] = None
    group_time_on: Optional[str] = None
    group_labels: Optional[str] = None
    split_time_horizon: Optional[str] = None
    split_time_horizon_unit: Optional[str] = None
    has_datetime: bool = False
    has_text: bool = False
    has_categorical: bool = False
    has_numerical: bool = False
    has_binary: bool = False
    has_high_cardinality_categorical: bool = False
    num_instances: Optional[int] = None
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    num_instance_groups: Optional[int] = None
    num_text_cols: Optional[int] = None
    num_high_cardinality_cats: Optional[int] = None
    num_cols_after_preprocessing: Optional[int] = None
    missing_value_fraction: Optional[float] = None
    domain: str = ""
    dataset_year: str = ""
    source: str = ""
    num_instances_train: Optional[int] = None
    num_instances_test: Optional[int] = None
    num_instance_groups_train: Optional[int] = None
    num_instance_groups_test: Optional[int] = None
    num_classes_train: Optional[int] = None
    num_classes_test: Optional[int] = None
    num_features_train: Optional[int] = None
    num_features_test: Optional[int] = None

    @property
    def split_id(self) -> str:
        return self.split_index

    @property
    def normalized_task_type(self) -> str:
        return "iid" if self.task_type == "random" else self.task_type

    @property
    def normalized_problem_type(self) -> str:
        return _normalize_problem_type(self.problem_type)

    @property
    def has_text_features(self) -> bool:
        return bool(self.has_text or (self.num_text_cols or 0) > 0)

    @property
    def has_high_cardinality_features(self) -> bool:
        return bool(
            self.has_high_cardinality_categorical
            or (self.num_high_cardinality_cats or 0) > 0
        )

    @property
    def is_high_dimensional(self) -> bool:
        width = self.num_cols_after_preprocessing
        if width is None:
            width = self.num_features
        return bool(width is not None and int(width) > 100)

    @property
    def metric_lower_is_better(self) -> bool:
        return beyondarena_metric_lower_is_better(self.eval_metric)

    @property
    def key(self) -> Tuple[str, int]:
        return (self.tabarena_task_name, self.split)

    def to_dataset_spec(self) -> BeyondArenaDatasetSpec:
        """Project committed CSV metadata onto the local manifest spec type."""

        return BeyondArenaDatasetSpec(
            beyondarena_id=self.tabarena_task_name,
            dataset_name=self.dataset_name,
            artifact_dir=None,
            artifact_revision=self.data_foundry_uri or self.task_id_str,
            task_type=self.normalized_task_type,
            target_column=self.target_name,
            problem_type=self.normalized_problem_type,
            raw_problem_type=self.problem_type,
            objective_metric=self.eval_metric,
            stratify_on=self.stratify_on,
            time_on=self.time_on,
            group_on=self.group_on,
            group_labels=self.group_labels,
            group_time_on=self.group_time_on,
            data_tags=(self.task_type,),
            source=self.source,
            license="",
            domain=self.domain,
            has_text_cache=False,
            has_dataset=False,
            has_high_cardinality=self.has_high_cardinality_features,
            high_cardinality_columns=(),
            has_text_features=self.has_text_features,
            text_columns=(),
            is_high_dimensional=self.is_high_dimensional,
            n_features=self.num_features,
            n_samples=self.num_instances,
            metadata_paths={
                "data_foundry_uri": self.data_foundry_uri,
                "task_id_str": self.task_id_str,
                "split_index": self.split_index,
                "split": str(self.split),
            },
            skip_reason="official metadata row only; local dataset.parquet not materialized",
        )


@dataclass(frozen=True)
class BeyondArenaCoreTaskRow:
    """One row from the official BeyondArena core subset CSV."""

    dataset: str
    split: int

    @property
    def key(self) -> Tuple[str, int]:
        return (self.dataset, self.split)


@dataclass(frozen=True)
class BeyondArenaLoadedDataset:
    """Loaded BeyondArena tabular data with metadata."""

    X: pd.DataFrame
    y: pd.Series
    frame: pd.DataFrame
    spec: BeyondArenaDatasetSpec
    data_source: str
    notes: str = ""


@dataclass(frozen=True)
class BeyondArenaSplit:
    """One official or fallback train/test split."""

    split_id: str
    repeat: str
    fold: str
    train_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]
    source: str = "official"
    allow_temporal_train_after_test: bool = False


@dataclass(frozen=True)
class BeyondArenaSplitBundle:
    """Parsed split metadata for one dataset."""

    dataset_id: str
    task_type: str
    comment: str
    splits: Tuple[BeyondArenaSplit, ...]
    source_path: Optional[Path] = None

    @property
    def split_ids(self) -> Tuple[str, ...]:
        return tuple(split.split_id for split in self.splits)


@dataclass(frozen=True)
class BeyondArenaInnerValidationPolicy:
    """Inner-CV policy aligned with the BeyondArena protocol description."""

    n_train_rows: int
    repeats: int
    folds: int
    stratified: bool
    group_column: Optional[str] = None
    time_column: Optional[str] = None

    @property
    def policy_id(self) -> str:
        prefix = f"{self.repeats}x{self.folds}" if self.repeats > 1 else f"{self.folds}fold"
        suffix = "stratified" if self.stratified else "plain"
        if self.group_column:
            suffix = f"{suffix}_grouped"
        if self.time_column:
            suffix = f"{suffix}_temporal"
        return f"{prefix}_{suffix}"


@dataclass(frozen=True)
class BeyondArenaPreprocessingProfile:
    """Opt-in preprocessing switches for BeyondArena parity/ablation runs."""

    profile_id: str = "beyondarena_local_fallback"
    encode_dates: bool = True
    use_text_cache: bool = True
    text_fallback: str = "tfidf_hash"  # tfidf_hash | length_hash
    text_tfidf_hash_buckets: int = 8
    group_encoding: str = "auto"  # auto | drop | hash50
    group_hash_buckets: int = 50
    high_cardinality_threshold: int = 50
    high_cardinality_encoder: str = "train_ordinal"


@dataclass(frozen=True)
class BeyondArenaPreprocessedFrame:
    """Feature frame plus preprocessing metadata for result rows."""

    X: pd.DataFrame
    y: pd.Series
    metadata: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def _normalize_problem_type(raw: Any) -> str:
    key = str(raw or "").strip().lower()
    if key in _CLASSIFICATION_PROBLEM_TYPES or "classification" in key:
        return "classification"
    if key in _REGRESSION_PROBLEM_TYPES or "regression" in key:
        return "regression"
    return key or "unknown"


def beyondarena_metric_lower_is_better(metric: Any) -> bool:
    """Return the optimization direction for official BeyondArena metrics."""

    key = str(metric or "").strip().lower()
    if key in _LOWER_IS_BETTER_METRICS:
        return True
    if key in _HIGHER_IS_BETTER_METRICS:
        return False
    return True


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _string_or_none(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    return str(value)


def _string_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value)


def _bool_value(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return bool(value)


def _int_or_none(value: Any) -> Optional[int]:
    if _is_missing(value):
        return None
    return int(value)


def _float_or_none(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    return float(value)


def _infer_task_type(
    *,
    data_tags: Sequence[str],
    group_on: Optional[str],
    time_on: Optional[str],
) -> str:
    tags = {str(tag).strip().lower() for tag in data_tags}
    if time_on or "temporal" in tags:
        return "temporal"
    if group_on or "grouped" in tags:
        return "grouped"
    return "iid"


def _stable_hash_bucket(value: Any, buckets: int) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % max(1, int(buckets))


def _dtype_column_names(dtypes: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(str(key) for key in dtypes.keys())


def _detect_text_columns(dtypes: Mapping[str, Any]) -> Tuple[str, ...]:
    out: List[str] = []
    for name, dtype in dtypes.items():
        dtype_text = str(dtype).strip().lower()
        if "text" in dtype_text or dtype_text in {"string_text", "free_text"}:
            out.append(str(name))
    return tuple(out)


def _detect_high_cardinality_columns(dtypes: Mapping[str, Any]) -> Tuple[str, ...]:
    out: List[str] = []
    for name, dtype in dtypes.items():
        if isinstance(dtype, Mapping):
            card = dtype.get("cardinality") or dtype.get("n_unique")
            try:
                if card is not None and int(card) >= 50:
                    out.append(str(name))
                    continue
            except Exception:
                pass
            marker = str(dtype.get("semantic_type", "")).lower()
        else:
            marker = str(dtype).lower()
        if "high_cardinality" in marker or "high-cardinality" in marker:
            out.append(str(name))
    return tuple(out)


def _metadata_path(path: Path, filename: str) -> Path:
    return path / filename


def iter_local_beyondarena_artifact_dirs(root: str | Path) -> Iterator[Path]:
    """Yield local artifact directories that contain BeyondArena task metadata."""

    root_path = Path(root)
    if (root_path / TASK_METADATA).exists():
        yield root_path
        return
    for task_file in sorted(root_path.rglob(TASK_METADATA)):
        yield task_file.parent


def load_beyondarena_spec(artifact_dir: str | Path) -> BeyondArenaDatasetSpec:
    """Load one local DataFoundry-style artifact directory in manifest-only mode."""

    path = Path(artifact_dir)
    task_path = _metadata_path(path, TASK_METADATA)
    dataset_path = _metadata_path(path, DATASET_METADATA)
    container_path = _metadata_path(path, CONTAINER_METADATA)
    dtypes_path = _metadata_path(path, DTYPES_METADATA)
    split_path = _metadata_path(path, SPLIT_METADATA)

    if not task_path.exists():
        raise FileNotFoundError(f"Missing required BeyondArena task metadata: {task_path}")

    task_meta = _read_json(task_path)
    dataset_meta = _read_json(dataset_path) if dataset_path.exists() else {}
    container_meta = _read_json(container_path) if container_path.exists() else {}
    dtypes = _read_json(dtypes_path) if dtypes_path.exists() else {}

    dataset_name = str(
        dataset_meta.get("unique_name")
        or path.parent.name
        or path.name
    )
    data_tags = tuple(str(tag) for tag in dataset_meta.get("data_tags", ()) or ())
    group_on = task_meta.get("group_on")
    time_on = task_meta.get("time_on")
    task_type = _infer_task_type(data_tags=data_tags, group_on=group_on, time_on=time_on)
    target = str(task_meta.get("target_column_name") or "")
    raw_problem = str(task_meta.get("problem_type") or "")
    dtype_cols = tuple(col for col in _dtype_column_names(dtypes) if col != target)
    text_cols = tuple(col for col in _detect_text_columns(dtypes) if col != target)
    high_card_cols = tuple(col for col in _detect_high_cardinality_columns(dtypes) if col != target)
    tag_text = {str(tag).strip().lower() for tag in data_tags}
    high_dim = bool(len(dtype_cols) >= 1000 or "high-dimensional" in tag_text or "high dimensional" in tag_text)
    high_card = bool(high_card_cols or "high-cardinality" in tag_text or "high cardinality" in tag_text)
    text_cache = path / TEXT_CACHE_BASENAME
    parquet_path = path / DATASET_PARQUET
    revision = str(container_meta.get("checksum") or container_meta.get("uuid") or path.name)

    metadata_paths = {
        "artifact_dir": str(path),
        "container": str(container_path) if container_path.exists() else "",
        "dataset": str(dataset_path) if dataset_path.exists() else "",
        "task": str(task_path),
        "splits": str(split_path) if split_path.exists() else "",
        "dtypes": str(dtypes_path) if dtypes_path.exists() else "",
        "parquet": str(parquet_path) if parquet_path.exists() else "",
        "text_cache": str(text_cache) if text_cache.exists() else "",
    }

    return BeyondArenaDatasetSpec(
        beyondarena_id=dataset_name,
        dataset_name=dataset_name,
        artifact_dir=path,
        artifact_revision=revision,
        task_type=task_type,
        target_column=target,
        problem_type=_normalize_problem_type(raw_problem),
        raw_problem_type=raw_problem,
        objective_metric=str(task_meta.get("objective_metric_name") or ""),
        stratify_on=task_meta.get("stratify_on"),
        time_on=time_on,
        group_on=group_on,
        group_labels=task_meta.get("group_labels"),
        group_time_on=task_meta.get("group_time_on"),
        data_tags=data_tags,
        source=str(dataset_meta.get("dataset_source") or ""),
        license=str(dataset_meta.get("license") or ""),
        domain=str(dataset_meta.get("domain_str") or ""),
        has_text_cache=text_cache.exists(),
        has_dataset=parquet_path.exists(),
        has_high_cardinality=high_card,
        high_cardinality_columns=high_card_cols,
        has_text_features=bool(text_cols or text_cache.exists() or "text" in tag_text),
        text_columns=text_cols,
        is_high_dimensional=high_dim,
        n_features=len(dtype_cols) if dtype_cols else None,
        n_samples=None,
        metadata_paths=metadata_paths,
        skip_reason=None if parquet_path.exists() else "dataset.parquet not present in local artifact",
    )


def discover_local_beyondarena_specs(root: str | Path) -> Tuple[BeyondArenaDatasetSpec, ...]:
    """Index all local artifact directories under *root* without reading parquet data."""

    specs = [load_beyondarena_spec(path) for path in iter_local_beyondarena_artifact_dirs(root)]
    return tuple(sorted(specs, key=lambda spec: (spec.beyondarena_id, str(spec.artifact_dir or ""))))


def discover_hf_beyondarena_specs(
    *,
    repo_id: str = BEYONDARENA_HF_REPO_ID,
    revision: str = "main",
    limit: Optional[int] = None,
) -> Tuple[BeyondArenaDatasetSpec, ...]:
    """Manifest-only Hugging Face discovery.

    Only JSON metadata files are downloaded to the HF cache; parquet payloads
    and text-cache artifacts are not materialized.  This is intentionally
    optional so CI can remain fully offline.
    """

    try:
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise BeyondArenaUnavailableError(
            "huggingface_hub is required for remote BeyondArena manifest discovery"
        ) from exc

    api = HfApi()
    task_paths: List[str] = []
    for entry in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", revision=revision, recursive=True):
        path = str(getattr(entry, "path", ""))
        if path.endswith(TASK_METADATA):
            task_paths.append(path)
            if limit is not None and len(task_paths) >= int(limit):
                break

    specs: List[BeyondArenaDatasetSpec] = []
    for task_path in sorted(task_paths):
        parent = str(Path(task_path).parent).replace("\\", "/")
        local_files: Dict[str, Path] = {}
        for filename in (CONTAINER_METADATA, DATASET_METADATA, TASK_METADATA, SPLIT_METADATA, DTYPES_METADATA):
            remote = f"{parent}/{filename}"
            try:
                local_files[filename] = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        revision=revision,
                        filename=remote,
                    )
                )
            except Exception:
                continue
        if TASK_METADATA not in local_files:
            continue
        specs.append(load_beyondarena_spec(local_files[TASK_METADATA].parent))

    return tuple(sorted(specs, key=lambda spec: spec.beyondarena_id))


def _read_csv_frame(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source)


def _require_csv_columns(df: pd.DataFrame, columns: Sequence[str], *, source_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")


def _with_official_split_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_csv_columns(
        out,
        ("tabarena_task_name", "repeat", "fold"),
        source_name="BeyondArena task metadata CSV",
    )
    repeat = pd.to_numeric(out["repeat"], errors="raise").astype(int)
    fold = pd.to_numeric(out["fold"], errors="raise").astype(int)
    if "split_index" not in out.columns:
        out["split_index"] = [f"r{r}f{f}" for r, f in zip(repeat, fold)]
    if "split" not in out.columns:
        n_folds = fold.groupby(out["tabarena_task_name"].astype(str)).transform("max") + 1
        out["split"] = (n_folds * repeat + fold).astype(int)
    return out


def _task_metadata_row_from_series(row: pd.Series) -> BeyondArenaTaskMetadataRow:
    return BeyondArenaTaskMetadataRow(
        dataset_name=_string_value(row["dataset_name"]),
        problem_type=_string_value(row["problem_type"]),
        is_classification=_bool_value(row["is_classification"]),
        target_name=_string_value(row["target_name"]),
        eval_metric=_string_value(row["eval_metric"]),
        tabarena_task_name=_string_value(row["tabarena_task_name"]),
        task_id_str=_string_value(row["task_id_str"]),
        data_foundry_uri=_string_value(row["data_foundry_uri"]),
        task_type=_string_value(row["task_type"]),
        repeat=int(row["repeat"]),
        fold=int(row["fold"]),
        split_index=_string_value(row["split_index"]),
        split=int(row["split"]),
        stratify_on=_string_or_none(row.get("stratify_on")),
        time_on=_string_or_none(row.get("time_on")),
        group_on=_string_or_none(row.get("group_on")),
        group_time_on=_string_or_none(row.get("group_time_on")),
        group_labels=_string_or_none(row.get("group_labels")),
        split_time_horizon=_string_or_none(row.get("split_time_horizon")),
        split_time_horizon_unit=_string_or_none(row.get("split_time_horizon_unit")),
        has_datetime=_bool_value(row.get("has_datetime")),
        has_text=_bool_value(row.get("has_text")),
        has_categorical=_bool_value(row.get("has_categorical")),
        has_numerical=_bool_value(row.get("has_numerical")),
        has_binary=_bool_value(row.get("has_binary")),
        has_high_cardinality_categorical=_bool_value(row.get("has_high_cardinality_categorical")),
        num_instances=_int_or_none(row.get("num_instances")),
        num_features=_int_or_none(row.get("num_features")),
        num_classes=_int_or_none(row.get("num_classes")),
        num_instance_groups=_int_or_none(row.get("num_instance_groups")),
        num_text_cols=_int_or_none(row.get("num_text_cols")),
        num_high_cardinality_cats=_int_or_none(row.get("num_high_cardinality_cats")),
        num_cols_after_preprocessing=_int_or_none(row.get("num_cols_after_preprocessing")),
        missing_value_fraction=_float_or_none(row.get("missing_value_fraction")),
        domain=_string_value(row.get("domain")),
        dataset_year=_string_value(row.get("dataset_year")),
        source=_string_value(row.get("source")),
        num_instances_train=_int_or_none(row.get("num_instances_train")),
        num_instances_test=_int_or_none(row.get("num_instances_test")),
        num_instance_groups_train=_int_or_none(row.get("num_instance_groups_train")),
        num_instance_groups_test=_int_or_none(row.get("num_instance_groups_test")),
        num_classes_train=_int_or_none(row.get("num_classes_train")),
        num_classes_test=_int_or_none(row.get("num_classes_test")),
        num_features_train=_int_or_none(row.get("num_features_train")),
        num_features_test=_int_or_none(row.get("num_features_test")),
    )


def load_beyondarena_task_metadata_csv(
    source: str | Path | pd.DataFrame = BEYONDARENA_TASK_METADATA_CSV_URL,
) -> Tuple[BeyondArenaTaskMetadataRow, ...]:
    """Load the official committed BeyondArena task metadata CSV.

    This is metadata-only and does not materialize DataFoundry parquet payloads.
    """

    df = _read_csv_frame(source)
    _require_csv_columns(
        df,
        (
            "dataset_name",
            "problem_type",
            "is_classification",
            "target_name",
            "eval_metric",
            "tabarena_task_name",
            "task_id_str",
            "data_foundry_uri",
            "task_type",
            "repeat",
            "fold",
        ),
        source_name="BeyondArena task metadata CSV",
    )
    df = _with_official_split_column(df)
    return tuple(_task_metadata_row_from_series(row) for _, row in df.iterrows())


def load_beyondarena_core_tasks_csv(
    source: str | Path | pd.DataFrame = BEYONDARENA_CORE_TASKS_CSV_URL,
) -> Tuple[BeyondArenaCoreTaskRow, ...]:
    """Load the official committed BeyondArena core subset CSV."""

    df = _read_csv_frame(source)
    _require_csv_columns(df, ("dataset", "split"), source_name="BeyondArena core tasks CSV")
    rows: List[BeyondArenaCoreTaskRow] = []
    for _, row in df.iterrows():
        rows.append(
            BeyondArenaCoreTaskRow(
                dataset=_string_value(row["dataset"]),
                split=int(row["split"]),
            )
        )
    return tuple(rows)


def select_beyondarena_core_task_rows(
    metadata: Iterable[BeyondArenaTaskMetadataRow],
    core_tasks: Iterable[BeyondArenaCoreTaskRow],
    *,
    strict: bool = True,
) -> Tuple[BeyondArenaTaskMetadataRow, ...]:
    """Select official task-metadata rows matching the BeyondArena core subset."""

    by_key: Dict[Tuple[str, int], BeyondArenaTaskMetadataRow] = {}
    for row in metadata:
        by_key[row.key] = row
    selected: List[BeyondArenaTaskMetadataRow] = []
    missing: List[Tuple[str, int]] = []
    seen: set[Tuple[str, int]] = set()
    for task in core_tasks:
        key = task.key
        if key in seen:
            continue
        match = by_key.get(key)
        if match is None:
            missing.append(key)
            continue
        selected.append(match)
        seen.add(key)
    if missing and strict:
        preview = ", ".join(f"{dataset}:{split}" for dataset, split in missing[:5])
        raise ValueError(
            f"{len(missing)} BeyondArena core task rows were not present in task metadata: {preview}"
        )
    return tuple(selected)


def select_beyondarena_core_dataset_split_rows(
    metadata: Iterable[BeyondArenaTaskMetadataRow],
    core_tasks: Iterable[BeyondArenaCoreTaskRow],
    *,
    strict: bool = True,
) -> Tuple[BeyondArenaTaskMetadataRow, ...]:
    """Select all official split rows for datasets present in the core subset."""

    metadata_rows = tuple(metadata)
    core_rows = tuple(core_tasks)
    dataset_order: List[str] = []
    for task in core_rows:
        if task.dataset not in dataset_order:
            dataset_order.append(task.dataset)
    by_dataset: Dict[str, List[BeyondArenaTaskMetadataRow]] = {dataset: [] for dataset in dataset_order}
    for row in metadata_rows:
        if row.tabarena_task_name in by_dataset:
            by_dataset[row.tabarena_task_name].append(row)
    missing = [dataset for dataset in dataset_order if not by_dataset.get(dataset)]
    if missing and strict:
        preview = ", ".join(str(dataset) for dataset in missing[:5])
        raise ValueError(
            f"{len(missing)} BeyondArena core datasets were not present in task metadata: {preview}"
        )
    selected: List[BeyondArenaTaskMetadataRow] = []
    for dataset in dataset_order:
        selected.extend(
            sorted(
                by_dataset.get(dataset, ()),
                key=lambda row: (int(row.repeat), int(row.fold), int(row.split), row.split_index),
            )
        )
    return tuple(selected)


def load_beyondarena_dataset(
    artifact_dir: str | Path,
    *,
    manifest_only: bool = False,
) -> BeyondArenaLoadedDataset | BeyondArenaDatasetSpec:
    """Load one BeyondArena artifact.

    With ``manifest_only=True`` this returns ``BeyondArenaDatasetSpec`` and does
    not touch the parquet payload.  Otherwise, the local ``dataset.parquet`` is
    read and a ``BeyondArenaLoadedDataset`` is returned.
    """

    spec = load_beyondarena_spec(artifact_dir)
    if manifest_only:
        return spec
    parquet_path = Path(artifact_dir) / DATASET_PARQUET
    if not parquet_path.exists():
        raise BeyondArenaUnavailableError(f"Missing BeyondArena parquet payload: {parquet_path}")
    try:
        frame = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise BeyondArenaUnavailableError(f"Unable to read {parquet_path}: {exc}") from exc
    if spec.target_column not in frame.columns:
        raise ValueError(
            f"BeyondArena target column {spec.target_column!r} not present in {parquet_path}"
        )
    y = frame[spec.target_column].copy()
    X = frame.drop(columns=[spec.target_column]).copy()
    spec = BeyondArenaDatasetSpec(
        **{**spec.to_dict(), "artifact_dir": spec.artifact_dir, "n_samples": int(len(frame))}
    )
    return BeyondArenaLoadedDataset(
        X=X,
        y=y,
        frame=frame,
        spec=spec,
        data_source=str(parquet_path),
        notes="loaded from local BeyondArena/DataFoundry artifact",
    )


def load_beyondarena_splits(artifact_dir: str | Path) -> BeyondArenaSplitBundle:
    """Parse official DataFoundry split metadata."""

    spec = load_beyondarena_spec(artifact_dir)
    path = Path(artifact_dir) / SPLIT_METADATA
    if not path.exists():
        raise FileNotFoundError(f"Missing BeyondArena split metadata: {path}")
    payload = _read_json(path)
    raw_splits = payload.get("splits", {})
    if not isinstance(raw_splits, Mapping):
        raise ValueError(f"Expected object-valued splits in {path}")

    parsed: List[BeyondArenaSplit] = []
    allow_temporal_train_after_test = _allows_temporal_train_after_test(payload, spec)
    for repeat, folds in raw_splits.items():
        if not isinstance(folds, Mapping):
            continue
        for fold, pair in folds.items():
            if not isinstance(pair, Sequence) or len(pair) < 2:
                continue
            train_raw, test_raw = pair[0], pair[1]
            train = tuple(int(i) for i in train_raw)
            test = tuple(int(i) for i in test_raw)
            parsed.append(
                BeyondArenaSplit(
                    split_id=f"{repeat}:{fold}",
                    repeat=str(repeat),
                    fold=str(fold),
                    train_indices=train,
                    test_indices=test,
                    source="official",
                    allow_temporal_train_after_test=allow_temporal_train_after_test,
                )
            )
    if not parsed:
        raise ValueError(f"No parseable BeyondArena splits found in {path}")
    return BeyondArenaSplitBundle(
        dataset_id=spec.beyondarena_id,
        task_type=spec.task_type,
        comment=str(payload.get("splits_comment") or ""),
        splits=tuple(parsed),
        source_path=path,
    )


def _allows_temporal_train_after_test(
    split_payload: Mapping[str, Any],
    spec: BeyondArenaDatasetSpec,
) -> bool:
    """Return whether official temporal metadata permits future-dated train rows."""

    if spec.task_type != "temporal":
        return False
    comment = str(split_payload.get("splits_comment") or "").lower()
    return "future" in comment and "included in the train data" in comment


def build_beyondarena_fallback_split(
    spec: BeyondArenaDatasetSpec,
    *,
    n_samples: int,
    seed: int = 0,
    test_fraction: float = 0.2,
    allow_non_iid: bool = False,
) -> BeyondArenaSplit:
    """Build a deterministic IID fallback split.

    Non-IID datasets require explicit opt-in so grouped/temporal tasks do not
    silently degrade to IID holdouts.
    """

    if spec.task_type != "iid" and not bool(allow_non_iid):
        raise BeyondArenaUnavailableError(
            f"IID fallback split refused for non-IID BeyondArena dataset "
            f"{spec.beyondarena_id!r} (task_type={spec.task_type!r})"
        )
    rng = np.random.default_rng(int(seed))
    indices = np.arange(int(n_samples))
    rng.shuffle(indices)
    n_test = max(1, int(round(float(test_fraction) * int(n_samples))))
    test = tuple(sorted(int(i) for i in indices[:n_test]))
    train = tuple(sorted(int(i) for i in indices[n_test:]))
    return BeyondArenaSplit(
        split_id=f"fallback_iid_seed{seed}",
        repeat="fallback",
        fold="0",
        train_indices=train,
        test_indices=test,
        source="fallback_iid",
    )


def build_beyondarena_inner_validation_policy(
    spec: BeyondArenaDatasetSpec,
    *,
    n_train_rows: int,
) -> BeyondArenaInnerValidationPolicy:
    """Return the BeyondArena inner-validation policy for a train fold."""

    n_rows = int(n_train_rows)
    if n_rows < 500:
        repeats, folds = 5, 5
    else:
        repeats, folds = 1, 8
    return BeyondArenaInnerValidationPolicy(
        n_train_rows=n_rows,
        repeats=repeats,
        folds=folds,
        stratified=spec.problem_type == "classification",
        group_column=spec.group_on if spec.task_type == "grouped" else None,
        time_column=spec.time_on if spec.task_type == "temporal" else None,
    )


def build_beyondarena_resampling_context(
    frame: pd.DataFrame,
    spec: BeyondArenaDatasetSpec,
    splits: Sequence[BeyondArenaSplit],
) -> Any:
    """Adapt BeyondArena row metadata and official splits to the core contract."""

    from tabnetics.pipeline.resampling import (
        FitResamplingContext,
        ResamplingPolicy,
        SplitAssignment,
    )

    n_rows = int(len(frame))
    is_classification = (
        spec.problem_type in _CLASSIFICATION_PROBLEM_TYPES
        or _normalize_problem_type(spec.problem_type) == "classification"
    )

    def column_values(column: Optional[str]) -> tuple[Any, ...]:
        if not column:
            return tuple()
        if column not in frame.columns:
            raise BeyondArenaUnavailableError(
                f"BeyondArena resampling field {column!r} is missing from the materialized frame"
            )
        return tuple(
            None if pd.isna(value) else value
            for value in frame[column].tolist()
        )

    groups: tuple[Any, ...] = tuple()
    timestamps: tuple[Any, ...] = tuple()
    boundaries: tuple[str, ...] = tuple()
    if spec.is_grouped:
        groups = column_values(spec.group_on)
        boundaries = ("groups",)
        policy_kind = "stratified_group" if is_classification else "group"
    elif spec.is_temporal:
        timestamps = column_values(spec.time_on)
        policy_kind = "blocked_temporal"
    else:
        policy_kind = "stratified" if is_classification else "iid"

    policy = ResamplingPolicy(
        kind=policy_kind,
        enforced_boundaries=boundaries,
        require_class_coverage=bool(is_classification),
        require_full_coverage=True,
    )
    base_context = FitResamplingContext(
        n_rows=n_rows,
        row_ids=tuple(range(n_rows)),
        groups=groups,
        timestamps=timestamps,
        policy=policy,
    )
    assignments = tuple(
        SplitAssignment(
            scope="outer",
            split_id=str(split.split_id),
            train_indices=tuple(split.train_indices),
            test_indices=tuple(split.test_indices),
            source=f"beyondarena_{split.source}",
            allow_unassigned=(
                len(set(split.train_indices).union(split.test_indices)) < n_rows
            ),
            metadata=(
                ("repeat", str(split.repeat)),
                ("fold", str(split.fold)),
                (
                    "allow_temporal_train_after_test",
                    bool(split.allow_temporal_train_after_test),
                ),
            ),
        )
        for split in splits
    )
    if not assignments:
        raise BeyondArenaUnavailableError(
            "At least one BeyondArena split is required to build a resampling context"
        )
    return base_context.with_supplied_splits(assignments, policy=policy)


def validate_beyondarena_split_leakage(
    frame: pd.DataFrame,
    spec: BeyondArenaDatasetSpec,
    split: BeyondArenaSplit,
) -> Dict[str, Any]:
    """Run leakage guards for one official/fallback split."""

    n_rows = len(frame)
    train = np.asarray(split.train_indices, dtype=int)
    test = np.asarray(split.test_indices, dtype=int)
    if np.any(train < 0) or np.any(test < 0) or np.any(train >= n_rows) or np.any(test >= n_rows):
        return {"ok": False, "reason": "split index out of bounds", "split_id": split.split_id}
    overlap = set(train.tolist()).intersection(set(test.tolist()))
    if overlap:
        return {"ok": False, "reason": "train/test row index overlap", "split_id": split.split_id}

    checks: Dict[str, Any] = {
        "ok": True,
        "split_id": split.split_id,
        "row_overlap": False,
        "group_overlap": None,
        "temporal_order_ok": None,
        "temporal_train_after_test_allowed": False,
    }
    if spec.group_on and spec.group_on in frame.columns:
        train_groups = set(frame.iloc[train][spec.group_on].dropna().astype(str).tolist())
        test_groups = set(frame.iloc[test][spec.group_on].dropna().astype(str).tolist())
        group_overlap = train_groups.intersection(test_groups)
        checks["group_overlap"] = bool(group_overlap)
        if group_overlap:
            return {
                **checks,
                "ok": False,
                "reason": "group labels cross train/test split",
                "overlap_count": len(group_overlap),
            }
    if spec.time_on and spec.time_on in frame.columns:
        train_time = pd.to_datetime(frame.iloc[train][spec.time_on], errors="coerce")
        test_time = pd.to_datetime(frame.iloc[test][spec.time_on], errors="coerce")
        if train_time.notna().any() and test_time.notna().any():
            order_ok = bool(train_time.max() <= test_time.min())
            checks["temporal_order_ok"] = order_ok
            if not order_ok:
                if bool(getattr(split, "allow_temporal_train_after_test", False)):
                    checks["temporal_train_after_test_allowed"] = True
                    return checks
                return {
                    **checks,
                    "ok": False,
                    "reason": "temporal train timestamps exceed test horizon",
                }
    return checks


def _train_values(series: pd.Series, train_indices: Sequence[int]) -> pd.Series:
    if not train_indices:
        return series
    return series.iloc[list(train_indices)]


def _append_group_hash_columns(
    X: pd.DataFrame,
    group_values: pd.Series,
    *,
    buckets: int,
    prefix: str,
) -> pd.DataFrame:
    out = X.copy()
    bucket_ids = group_values.map(lambda value: _stable_hash_bucket(value, buckets))
    for bucket in range(int(buckets)):
        out[f"{prefix}{bucket:02d}"] = (bucket_ids == bucket).astype(float)
    return out


def _encode_dates(X: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    out = X.copy()
    encoded: List[str] = []
    for col in columns:
        if col not in out.columns:
            continue
        dt = pd.to_datetime(out[col], errors="coerce")
        if dt.notna().sum() == 0:
            continue
        safe = str(col)
        out[f"{safe}__year"] = dt.dt.year.fillna(0).astype(float)
        out[f"{safe}__month"] = dt.dt.month.fillna(0).astype(float)
        out[f"{safe}__day"] = dt.dt.day.fillna(0).astype(float)
        out[f"{safe}__dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(float)
        ordinal = dt.map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
        out[f"{safe}__ordinal"] = pd.Series(ordinal, index=out.index).fillna(0).astype(float)
        out = out.drop(columns=[col])
        encoded.append(safe)
    return out, tuple(encoded)


def _tokenize_text(value: Any) -> Tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(_TEXT_TOKEN_PATTERN.findall(str(value).lower()))


def _hashed_tfidf_frame(
    text: pd.Series,
    *,
    train_indices: Sequence[int],
    buckets: int,
    prefix: str,
) -> pd.DataFrame:
    safe_buckets = int(max(1, buckets))
    train_positions = [int(i) for i in train_indices if 0 <= int(i) < len(text)]
    if not train_positions:
        train_positions = list(range(len(text)))
    doc_freq = np.zeros(safe_buckets, dtype=float)
    train_text = text.iloc[train_positions]
    for value in train_text:
        seen = {_stable_hash_bucket(token, safe_buckets) for token in _tokenize_text(value)}
        for bucket in seen:
            doc_freq[bucket] += 1.0
    n_train = max(1, len(train_text))
    idf = np.log((1.0 + float(n_train)) / (1.0 + doc_freq)) + 1.0

    values = np.zeros((len(text), safe_buckets), dtype=float)
    for row_idx, value in enumerate(text):
        tokens = _tokenize_text(value)
        if not tokens:
            continue
        counts: Dict[int, int] = {}
        for token in tokens:
            bucket = _stable_hash_bucket(token, safe_buckets)
            counts[bucket] = counts.get(bucket, 0) + 1
        denom = float(len(tokens))
        for bucket, count in counts.items():
            values[row_idx, bucket] = (float(count) / denom) * float(idf[bucket])
    columns = [f"{prefix}__tfidf_hash_{bucket:02d}" for bucket in range(safe_buckets)]
    return pd.DataFrame(values, index=text.index, columns=columns)


def _encode_text_fallback(
    X: pd.DataFrame,
    text_columns: Sequence[str],
    *,
    train_indices: Sequence[int],
    profile: BeyondArenaPreprocessingProfile,
) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    out = X.copy()
    encoded: List[str] = []
    fallback = str(profile.text_fallback or "tfidf_hash").strip().lower()
    if fallback not in {"length_hash", "tfidf_hash"}:
        raise ValueError(f"unsupported BeyondArena text fallback: {profile.text_fallback!r}")
    for col in text_columns:
        if col not in out.columns:
            continue
        text = out[col].fillna("").astype(str)
        out[f"{col}__text_len"] = text.map(len).astype(float)
        out[f"{col}__text_hash"] = text.map(lambda value: _stable_hash_bucket(value, 1024)).astype(float)
        if fallback == "tfidf_hash":
            tfidf = _hashed_tfidf_frame(
                text,
                train_indices=train_indices,
                buckets=int(profile.text_tfidf_hash_buckets),
                prefix=str(col),
            )
            out = pd.concat([out, tfidf], axis=1)
        out = out.drop(columns=[col])
        encoded.append(str(col))
    return out, tuple(encoded)


def apply_beyondarena_preprocessing(
    frame: pd.DataFrame,
    spec: BeyondArenaDatasetSpec,
    *,
    split: Optional[BeyondArenaSplit] = None,
    train_indices: Optional[Sequence[int]] = None,
    profile: Optional[BeyondArenaPreprocessingProfile] = None,
    text_cache_path: Optional[str | Path] = None,
) -> BeyondArenaPreprocessedFrame:
    """Apply deterministic, opt-in BeyondArena preprocessing fallbacks."""

    if spec.target_column not in frame.columns:
        raise ValueError(f"target column {spec.target_column!r} missing from frame")
    profile = profile or BeyondArenaPreprocessingProfile()
    train_idx = tuple(int(i) for i in (train_indices or (split.train_indices if split else tuple(range(len(frame))))))
    y = frame[spec.target_column].copy()
    X = frame.drop(columns=[spec.target_column]).copy()

    metadata: Dict[str, Any] = {
        "preprocessing_profile": profile.profile_id,
        "dataset_id": spec.beyondarena_id,
        "task_type": spec.task_type,
        "target_column": spec.target_column,
        "date_columns_encoded": (),
        "text_columns_encoded": (),
        "text_cache_used": False,
        "text_fallback": profile.text_fallback,
        "text_tfidf_hash_buckets": int(profile.text_tfidf_hash_buckets),
        "group_handling": "none",
        "group_hash_buckets": 0,
        "high_cardinality_encoder": profile.high_cardinality_encoder,
        "max_categorical_cardinality": 0,
        "high_cardinality_columns": (),
    }

    date_candidates: List[str] = []
    for col in (spec.time_on, spec.group_time_on):
        if col:
            date_candidates.append(str(col))
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]) and str(col) not in date_candidates:
            date_candidates.append(str(col))
    if profile.encode_dates:
        X, encoded_dates = _encode_dates(X, date_candidates)
        metadata["date_columns_encoded"] = encoded_dates

    if profile.use_text_cache:
        cache = Path(text_cache_path) if text_cache_path is not None else (
            spec.artifact_dir / TEXT_CACHE_BASENAME if spec.artifact_dir is not None else None
        )
        if cache is not None and cache.exists():
            try:
                cache_df = pd.read_parquet(cache)
            except Exception as exc:
                raise BeyondArenaUnavailableError(f"Unable to read BeyondArena text cache {cache}: {exc}") from exc
            forbidden = {spec.target_column}
            leaked = forbidden.intersection(set(cache_df.columns))
            if leaked:
                raise ValueError(f"text cache contains forbidden target columns: {sorted(leaked)}")
            cache_df = cache_df.reset_index(drop=True)
            if len(cache_df) != len(X):
                raise ValueError(
                    f"text cache row count {len(cache_df)} does not match frame row count {len(X)}"
                )
            X = pd.concat(
                [X.reset_index(drop=True), cache_df.add_prefix("textcache__").reset_index(drop=True)],
                axis=1,
            )
            metadata["text_cache_used"] = True
    text_cols = tuple(col for col in spec.text_columns if col in X.columns)
    if text_cols and not bool(metadata["text_cache_used"]):
        X, encoded_text = _encode_text_fallback(
            X,
            text_cols,
            train_indices=train_idx,
            profile=profile,
        )
        metadata["text_columns_encoded"] = encoded_text

    if spec.group_on and spec.group_on in X.columns:
        mode = profile.group_encoding
        if mode == "auto":
            mode = "hash50" if str(spec.group_labels or "").lower() == "per_group" else "drop"
        if mode == "hash50":
            X = _append_group_hash_columns(
                X,
                X[spec.group_on],
                buckets=int(profile.group_hash_buckets),
                prefix=f"{spec.group_on}__group_hash_",
            )
            X = X.drop(columns=[spec.group_on])
            metadata["group_handling"] = "hash50"
            metadata["group_hash_buckets"] = int(profile.group_hash_buckets)
        elif mode == "drop":
            X = X.drop(columns=[spec.group_on])
            metadata["group_handling"] = "drop_index"

    categorical_cols = [
        str(col)
        for col in X.columns
        if (
            pd.api.types.is_object_dtype(X[col].dtype)
            or pd.api.types.is_string_dtype(X[col].dtype)
            or isinstance(X[col].dtype, pd.CategoricalDtype)
        )
    ]
    high_card_cols: List[str] = []
    max_card = 0
    for col in categorical_cols:
        train = _train_values(X[col], train_idx)
        card = int(train.nunique(dropna=True))
        max_card = max(max_card, card)
        if card >= int(profile.high_cardinality_threshold):
            high_card_cols.append(col)
        categories = {value: idx for idx, value in enumerate(sorted(train.dropna().astype(str).unique()))}
        X[col] = X[col].astype(str).map(categories).fillna(-1).astype(float)
    metadata["max_categorical_cardinality"] = int(max_card)
    metadata["high_cardinality_columns"] = tuple(sorted(set(high_card_cols).union(spec.high_cardinality_columns)))

    return BeyondArenaPreprocessedFrame(X=X, y=y, metadata=metadata)


def build_beyondarena_smoke_subset(
    specs: Iterable[BeyondArenaDatasetSpec],
    *,
    max_items: int = 6,
) -> Tuple[str, ...]:
    """Pick a representative smoke subset from available metadata."""

    ordered = sorted(specs, key=lambda spec: spec.beyondarena_id)
    selectors = (
        (
            "iid",
            lambda s: (
                s.task_type == "iid"
                and not s.has_text_features
                and not s.has_text_cache
                and not s.has_high_cardinality
                and not s.is_high_dimensional
            ),
        ),
        ("grouped", lambda s: s.task_type == "grouped"),
        ("temporal", lambda s: s.task_type == "temporal"),
        ("text", lambda s: s.has_text_features or s.has_text_cache),
        ("high_cardinality", lambda s: s.has_high_cardinality),
        ("high_dimensional", lambda s: s.is_high_dimensional),
    )
    chosen: List[str] = []
    for _label, predicate in selectors:
        for spec in ordered:
            if spec.beyondarena_id in chosen:
                continue
            if predicate(spec):
                chosen.append(spec.beyondarena_id)
                break
    return tuple(chosen[: int(max_items)])


def build_beyondarena_smoke_task_rows(
    metadata: Iterable[BeyondArenaTaskMetadataRow],
    *,
    max_items: int = 6,
) -> Tuple[BeyondArenaTaskMetadataRow, ...]:
    """Pick representative official split rows for a cheap BeyondArena smoke run."""

    ordered = sorted(metadata, key=_smoke_task_row_rank)
    selectors = (
        (
            "iid",
            lambda row: (
                row.normalized_task_type == "iid"
                and not row.has_text_features
                and not row.has_high_cardinality_features
                and not row.is_high_dimensional
            ),
        ),
        ("grouped", lambda row: row.normalized_task_type == "grouped"),
        ("temporal", lambda row: row.normalized_task_type == "temporal"),
        ("text", lambda row: row.has_text_features),
        ("high_cardinality", lambda row: row.has_high_cardinality_features),
        ("high_dimensional", lambda row: row.is_high_dimensional),
    )
    chosen: List[BeyondArenaTaskMetadataRow] = []
    chosen_keys: set[Tuple[str, int]] = set()
    chosen_datasets: set[str] = set()
    for _label, predicate in selectors:
        match: Optional[BeyondArenaTaskMetadataRow] = None
        for row in ordered:
            if row.key in chosen_keys or row.tabarena_task_name in chosen_datasets:
                continue
            if predicate(row):
                match = row
                break
        if match is None:
            for row in ordered:
                if row.key in chosen_keys:
                    continue
                if predicate(row):
                    match = row
                    break
        if match is not None:
            chosen.append(match)
            chosen_keys.add(match.key)
            chosen_datasets.add(match.tabarena_task_name)
    return tuple(chosen[: int(max_items)])


def _large_int_when_missing(value: Optional[int]) -> int:
    return 10**12 if value is None else int(value)


def _smoke_task_row_rank(row: BeyondArenaTaskMetadataRow) -> Tuple[Any, ...]:
    width = row.num_cols_after_preprocessing
    if width is None:
        width = row.num_features
    return (
        row.normalized_problem_type != "classification",
        _large_int_when_missing(row.num_instances_train),
        _large_int_when_missing(row.num_instances_test),
        _large_int_when_missing(width),
        bool(row.has_text_features),
        bool(row.has_high_cardinality_features),
        bool(row.is_high_dimensional),
        row.tabarena_task_name,
        int(row.split),
    )


def _current_feasibility_rank(row: BeyondArenaTaskMetadataRow) -> Tuple[Any, ...]:
    width = row.num_cols_after_preprocessing
    if width is None:
        width = row.num_features
    task_type_rank = {"iid": 0, "grouped": 1, "temporal": 2}.get(row.normalized_task_type, 9)
    return (
        bool(row.has_text_features),
        bool(row.has_high_cardinality_features),
        bool(row.is_high_dimensional),
        task_type_rank,
        _large_int_when_missing(row.num_instances_train),
        _large_int_when_missing(row.num_instances_test),
        _large_int_when_missing(width),
        row.tabarena_task_name,
        int(row.split),
    )


def build_beyondarena_current_feasibility_task_rows(
    metadata: Iterable[BeyondArenaTaskMetadataRow],
    *,
    max_items: int = 1,
) -> Tuple[BeyondArenaTaskMetadataRow, ...]:
    """Pick small classification rows for current-tabnetics runner proof.

    This subset is intentionally not a replacement for smoke/core validation.
    It gives operators a deterministic manifest for cheap current-default
    provenance checks before launching slower Stage-1 smoke rows.
    """

    candidates = [
        row
        for row in metadata
        if row.normalized_problem_type == "classification" and row.is_classification
    ]
    ordered = sorted(candidates, key=_current_feasibility_rank)
    chosen: List[BeyondArenaTaskMetadataRow] = []
    chosen_keys: set[Tuple[str, int]] = set()
    chosen_datasets: set[str] = set()
    for row in ordered:
        if len(chosen) >= int(max_items):
            break
        if row.tabarena_task_name in chosen_datasets:
            continue
        chosen.append(row)
        chosen_keys.add(row.key)
        chosen_datasets.add(row.tabarena_task_name)
    for row in ordered:
        if len(chosen) >= int(max_items):
            break
        if row.key in chosen_keys:
            continue
        chosen.append(row)
        chosen_keys.add(row.key)
    return tuple(chosen[: int(max_items)])


BEYONDARENA_SMOKE_DATASET_SET = (
    "amazon_employee_access",
    "amex_non_iid_1m",
    "hotel_booking_demand",
    "california_house_prices_2020",
    "bioresponse",
    "home_credit_default_risk",
)


__all__ = [
    "BEYONDARENA_CORE_TASKS_CSV_URL",
    "BEYONDARENA_EXPECTED_ACCEPTED_DATASETS",
    "BEYONDARENA_EXPECTED_CORE_TASK_ROWS",
    "BEYONDARENA_EXPECTED_TASK_METADATA_ROWS",
    "BEYONDARENA_HF_REPO_ID",
    "BEYONDARENA_SMOKE_DATASET_SET",
    "BEYONDARENA_TASK_METADATA_CSV_URL",
    "BeyondArenaCoreTaskRow",
    "BeyondArenaDatasetSpec",
    "BeyondArenaInnerValidationPolicy",
    "BeyondArenaLoadedDataset",
    "BeyondArenaPreprocessedFrame",
    "BeyondArenaPreprocessingProfile",
    "BeyondArenaSplit",
    "BeyondArenaSplitBundle",
    "BeyondArenaTaskMetadataRow",
    "BeyondArenaUnavailableError",
    "apply_beyondarena_preprocessing",
    "beyondarena_metric_lower_is_better",
    "build_beyondarena_current_feasibility_task_rows",
    "build_beyondarena_fallback_split",
    "build_beyondarena_inner_validation_policy",
    "build_beyondarena_resampling_context",
    "build_beyondarena_smoke_subset",
    "build_beyondarena_smoke_task_rows",
    "discover_hf_beyondarena_specs",
    "discover_local_beyondarena_specs",
    "iter_local_beyondarena_artifact_dirs",
    "load_beyondarena_core_tasks_csv",
    "load_beyondarena_dataset",
    "load_beyondarena_spec",
    "load_beyondarena_splits",
    "load_beyondarena_task_metadata_csv",
    "select_beyondarena_core_dataset_split_rows",
    "select_beyondarena_core_task_rows",
    "validate_beyondarena_split_leakage",
]
